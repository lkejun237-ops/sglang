# SPDX-License-Identifier: Apache-2.0
"""Run unchanged on CUDA for the frozen exact padding gate."""

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.spatial_padding import (
    spatial_pad,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("time", [1, 2, 4, 8, 10, 17])
@pytest.mark.parametrize(
    "padding",
    [(1, 1, 1, 1, 0, 0), (0, 1, 0, 1, 0, 0), (2, 0, 1, 3, 0, 0), (0, 0, 0, 0, 0, 0)],
)
@pytest.mark.parametrize(
    "layout",
    [
        "contiguous",
        "spatial_stride",
        "channel_stride",
        "temporal_slice",
        "channels_last",
        "channels_last_temporal_slice",
    ],
)
def test_bitwise(device, dtype, time, padding, layout):
    x = torch.randn(2, 4, time + 2, 8, 12, device=device, dtype=dtype)
    if layout == "spatial_stride":
        x = x[..., ::2]
    elif layout == "channel_stride":
        x = x[:, ::2]
    elif layout == "temporal_slice":
        x = x[:, :, 2:]
    elif layout == "channels_last_temporal_slice":
        x = x.contiguous(memory_format=torch.channels_last_3d)[:, :, 1:-1]
    elif layout == "channels_last":
        x = x.contiguous(memory_format=torch.channels_last_3d)
    else:
        x = x[:, :, :time].contiguous()
    # Compare bytes, including signed zero; no arithmetic tolerance.
    expected = F.pad(x, padding, mode="reflect")
    actual = spatial_pad(x, padding, "reflect")
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize(
    "padding,mode",
    [
        ((1, 1, 1, 1, 1, 0), "reflect"),
        ((-1, 0, 0, 0, 0, 0), "reflect"),
        ((1, 1, 1, 1, 0, 0), "replicate"),
        ((1, 1, 1, 1, 0, 0), "constant"),
    ],
)
def test_fallback(padding, mode):
    x = torch.randn(1, 2, 3, 5, 6)
    assert torch.equal(spatial_pad(x, padding, mode), F.pad(x, padding, mode=mode))


def test_invalid_reflection_preserves_error():
    x = torch.randn(1, 2, 3, 5, 6)
    with pytest.raises(RuntimeError):
        spatial_pad(x, (6, 0, 0, 0, 0, 0), "reflect")


def test_fast_path_is_used(monkeypatch):
    original = F.pad
    calls = []

    def record(x, padding, **kwargs):
        calls.append((x.ndim, padding))
        return original(x, padding, **kwargs)

    monkeypatch.setattr(F, "pad", record)
    spatial_pad(torch.randn(2, 4, 8, 5, 6), (1, 1, 1, 1, 0, 0), "reflect")
    assert calls == [(3, (1, 1, 1, 1))]


@pytest.mark.parametrize("device", DEVICES)
def test_channels_last_autograd_uses_native_path(device):
    source = torch.randn(2, 4, 3, 8, 12, device=device, requires_grad=True)
    x = source.contiguous(memory_format=torch.channels_last_3d)
    padding = (1, 1, 2, 1, 0, 0)
    actual = torch.autograd.grad(
        spatial_pad(x, padding, "reflect").sum(), source, retain_graph=True
    )[0]
    expected = torch.autograd.grad(F.pad(x, padding, "reflect").sum(), source)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
