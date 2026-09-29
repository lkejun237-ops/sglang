# SPDX-License-Identifier: Apache-2.0
"""Same-device bitwise and memory-safety checks for the Zing KV writer."""

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as m
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _apply_rope,
    _apply_rope_kv_into,
    _can_fuse_rope_kv_write,
)


@pytest.mark.parametrize("heads", [56, 28, 14])
@pytest.mark.parametrize("rows", [1, 52, 2296, 2298])
@pytest.mark.parametrize("head_stride", [128, 384])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_writer_strides_and_sentinels(heads, rows, head_stride, device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setattr(m, "_ZING_EXPERIMENTAL_STRIDED_KV_WRITE", True)
    # Nonzero input offset, padded rows, interleaved heads and partial last CTA.
    width = (heads - 1) * head_stride + 128
    stride = width + 257
    storage = torch.randn(17 + rows * stride, device=device).bfloat16()
    k = storage.as_strided((rows, heads, 128), (stride, head_stride, 1), 17)
    v = k.clone(memory_format=torch.contiguous_format)
    original = storage.clone()
    c = torch.randn(rows, 96, device=device).bfloat16()
    outputs = [
        torch.full((rows + 2, heads, 128), -123, device=device, dtype=k.dtype)
        for _ in range(2)
    ]
    ok, ov = (x[1:-1] for x in outputs)
    assert _can_fuse_rope_kv_write(k, v, c, ok, ov) == (
        device == "cuda"
        and torch.version.hip is None
        and (head_stride == 128 or heads in (14, 28))
    )
    _apply_rope_kv_into(k, v, c, ok, ov)
    assert torch.equal(ok, _apply_rope(k, c))
    assert torch.equal(ov, v)
    assert torch.equal(ok[..., 96:], k[..., 96:])
    assert torch.equal(storage, original)
    for x in outputs:
        assert (x[0] == -123).all() and (x[-1] == -123).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_writer_rejects_unsafe_and_unsupported_layouts(monkeypatch):
    monkeypatch.setattr(m, "_ZING_EXPERIMENTAL_STRIDED_KV_WRITE", True)
    k = torch.randn(52, 28, 384, device="cuda").bfloat16()[..., :128]
    v = k.clone()
    c = torch.randn(52, 96, device="cuda").bfloat16()
    ok, ov = torch.empty_like(v), torch.empty_like(v)
    assert not _can_fuse_rope_kv_write(k, v, c, ok, ok)
    assert not _can_fuse_rope_kv_write(k, v, c, v, ov)
    assert not _can_fuse_rope_kv_write(
        k.as_strided(k.shape, (128, 384, 1)), v, c, ok, ov
    )
    assert not _can_fuse_rope_kv_write(
        k.as_strided(k.shape, (28 * 384, 256, 1)), v, c, ok, ov
    )
    assert not _can_fuse_rope_kv_write(
        k[..., ::2], v[..., ::2], c[:, :48], ok[..., ::2], ov[..., ::2]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("heads", [14, 28])
@torch.no_grad()
def test_disabled_strided_writer_preserves_existing_paths(heads, monkeypatch):
    monkeypatch.setattr(m, "_ZING_EXPERIMENTAL_STRIDED_KV_WRITE", False)
    k = torch.randn(52, heads, 384, device="cuda").bfloat16()[..., :128]
    v = k.clone()
    c = torch.randn(52, 96, device="cuda").bfloat16()
    ok, ov = torch.empty_like(v), torch.empty_like(v)
    assert not _can_fuse_rope_kv_write(k, v, c, ok, ov)
    _apply_rope_kv_into(k, v, c, ok, ov)
    assert torch.equal(ok, _apply_rope(k, c)) and torch.equal(ov, v)
    # Disabling the new layout must not disable the previously fused writer.
    assert _can_fuse_rope_kv_write(k.contiguous(), v, c, ok, ov) == (
        torch.version.hip is None
    )
