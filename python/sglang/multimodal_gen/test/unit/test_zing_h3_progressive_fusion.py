# SPDX-License-Identifier: Apache-2.0
"""Real tensor parity, cache ownership and progressive fallback for opt-in assembly."""

from importlib.util import find_spec

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.norm import (
    TemporalIsolatedGroupNorm,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive_fusion import (
    fused_conv_input,
)

CUDA_KERNELS = torch.cuda.is_available() and find_spec("triton") is not None


@pytest.mark.parametrize("norm_enabled", [False, True])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize(
    "time,history,offset", [(1, 2, 0), (8, 2, 0), (3, 4, 1), (2, 0, 0)]
)
@pytest.mark.parametrize("mode", ["reflect", "constant"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(not CUDA_KERNELS, reason="Requires CUDA and Triton")
@torch.no_grad()
def test_assembly_matches_original_and_owns_cache(
    norm_enabled, channels_last, time, history, offset, mode, dtype
):
    torch.manual_seed(512)
    x = torch.randn(2, 32, time, 7, 9, device="cuda", dtype=dtype)
    fmt = torch.channels_last_3d if channels_last else torch.contiguous_format
    x = x.contiguous(memory_format=fmt)
    previous = (
        torch.randn(2, 32, history, 7, 9, device="cuda", dtype=dtype)
        if history
        else None
    )
    if previous is not None:
        previous = previous.contiguous(memory_format=fmt)
    norm = (
        TemporalIsolatedGroupNorm(8, 32, eps=1e-6).to(device="cuda", dtype=dtype)
        if norm_enabled
        else None
    )
    if norm is not None:
        norm.weight.uniform_(0.5, 1.5)
        norm.bias.uniform_(-0.3, 0.3)
    current = F.silu(norm(x)) if norm is not None else x
    joined = torch.cat((previous, current), 2) if history else current
    expected = F.pad(joined[:, :, offset:], (1, 1, 1, 1, 0, 0), mode=mode)
    actual, cache = fused_conv_input(
        x, previous, history, offset, (1, 1), mode, channels_last, norm
    )
    atol = 0 if norm is None else (0.016 if dtype == torch.bfloat16 else 2e-5)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)
    assert actual.is_contiguous(memory_format=fmt)
    if history:
        torch.testing.assert_close(cache, joined[:, :, -history:], atol=atol, rtol=atol)
        saved = cache.clone()
        x.zero_()
        previous.zero_()
        actual.zero_()
        assert torch.equal(cache, saved)
    else:
        assert cache is None


@pytest.mark.skipif(not CUDA_KERNELS, reason="Requires CUDA and Triton")
@torch.no_grad()
def test_initial_zero_history_and_temporal_norm_isolation():
    x = torch.randn(1, 32, 8, 7, 9, device="cuda", dtype=torch.bfloat16)
    norm = TemporalIsolatedGroupNorm(8, 32, eps=1e-6).cuda().bfloat16()
    expected = F.pad(F.silu(norm(x)), (1, 1, 1, 1, 0, 0), mode="reflect")
    a, _ = fused_conv_input(x, None, 2, 0, (1, 1), "reflect", True, norm)
    assert torch.count_nonzero(a[:, :, :2]) == 0
    torch.testing.assert_close(a[:, :, 2:], expected, atol=0.016, rtol=0.016)
    x[:, :, -1] += 5
    b, _ = fused_conv_input(x, None, 2, 0, (1, 1), "reflect", True, norm)
    assert torch.equal(a[:, :, :-1], b[:, :, :-1])


@pytest.mark.skipif(not CUDA_KERNELS, reason="Requires CUDA and Triton")
@pytest.mark.parametrize("near_constant", [False, True])
@torch.no_grad()
def test_multi_partition_statistics_and_strided_views(near_constant):
    # Eleven partial reductions per group, including a short final partition.
    x = torch.randn(2, 64, 3, 37, 35, device="cuda").transpose(-1, -2)
    if near_constant:
        x = x * 1e-5 + 3
    norm = TemporalIsolatedGroupNorm(8, 64, eps=1e-6).cuda()
    norm.weight.uniform_(0.5, 1.5)
    norm.bias.uniform_(-0.2, 0.2)
    actual, _ = fused_conv_input(x, None, 2, 0, (1, 1), "reflect", True, norm)
    expected = F.pad(F.silu(norm(x)), (1, 1, 1, 1, 0, 0), mode="reflect")
    torch.testing.assert_close(
        actual[:, :, 2:], expected, atol=6e-4 if near_constant else 2e-5, rtol=2e-5
    )


def test_cpu_unsupported_path_is_explicit():
    x = torch.randn(1, 32, 1, 7, 9)
    assert fused_conv_input(x, None, 2, 0, (1, 1), "reflect", False) is None


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("role", ["shortcut", "downsample"])
@torch.no_grad()
def test_pointwise_dispatch_preserves_explicit_input_layout(
    monkeypatch, prepared, role
):
    from test_zing_h3_progressive_kernel import make_codecs

    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
        progressive_fusion,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.conv import (
        BaseConv3d,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive import (
        CausalEncoderCache,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive_layout import (
        make_encoder_layout,
    )

    model = make_codecs(ch_mult=(1, 2, 4)).modules["video_vae"]
    shortcut = model.encoder.down[1].block[0].nin_shortcut
    if role == "downsample":
        # Exercise the channel-only downsample policy without spatial/temporal stride.
        shortcut = BaseConv3d(32, 64, kernel_size=1, causal=True).double()
        model.encoder.down[0].downsample = shortcut
    weights = {
        layer: layer.weight.contiguous(memory_format=torch.channels_last_3d)
        for layer in model.encoder.modules()
        if isinstance(layer, torch.nn.Conv3d)
    }
    layout = make_encoder_layout(model, weights=weights if prepared else None)
    assert shortcut.weight.is_contiguous()
    assert shortcut.weight.is_contiguous(memory_format=torch.channels_last_3d)
    assert layout.convs[shortcut].input_memory_format == torch.contiguous_format
    assert layout.convs[model.encoder.conv_in].input_memory_format == (
        torch.channels_last_3d if prepared else torch.contiguous_format
    )
    assert (
        layout.convs[model.encoder.down[2].block[0].conv1].input_memory_format
        == torch.contiguous_format
    )
    x = torch.randn(1, 32, 3, 7, 9, dtype=torch.float64)
    if prepared:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    cache = CausalEncoderCache(model)
    cache.fused_conv_input = False
    expected = cache.conv(shortcut, x, {}, "shortcut", layout.convs[shortcut])
    calls = []

    def pointwise_assembly(
        x, previous, history, offset, padding, mode, channels_last, norm
    ):
        # CPU dispatch probe only; the CUDA test below exercises the real kernel.
        assert previous is None and history == offset == 0
        assert padding == (0, 0) and norm is None
        calls.append(channels_last)
        fmt = torch.channels_last_3d if channels_last else torch.contiguous_format
        return x.contiguous(memory_format=fmt), None

    monkeypatch.setattr(progressive_fusion, "fused_conv_input", pointwise_assembly)
    cache.fused_conv_input = True
    states = {}
    actual = cache.conv(shortcut, x, states, "shortcut", layout.convs[shortcut])
    assert calls == [False]
    assert actual.stride() == expected.stride()
    assert torch.equal(actual, expected)
    assert states == {"shortcut": (None, 3)}


def test_all_encoder_switches_keep_cpu_stream_lifecycle(monkeypatch):
    from test_zing_h3_progressive_kernel import make_codecs

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
        create_session,
    )

    codecs = make_codecs()
    for name in (
        "MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT",
        "MINIMAX_H3_PROGRESSIVE_FUSED_CONV_INPUT",
    ):
        monkeypatch.delenv(name, raising=False)
    default_session = create_session(codecs, capacity_frames=34)
    assert default_session.cache.fused_norm_input
    assert default_session.cache.fused_conv_input
    default_session.dispose()
    pixels = torch.randint(256, (3, 34, 8, 8), dtype=torch.uint8)

    def run():
        session = create_session(codecs, capacity_frames=34)
        session.append(pixels, frame_start=0)
        session.finish()
        values = []
        while session.ready:
            values.append(session.pop().values)
        result = torch.cat(values, 2)
        counts = dict(session.cache.optimization_counts)
        session.dispose()
        assert session.state_usage().retained_bytes == 0
        return result, counts

    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT", "0")
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_FUSED_CONV_INPUT", "0")
    expected, _ = run()
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT", "1")
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_FUSED_CONV_INPUT", "1")
    actual, counts = run()
    assert torch.equal(actual, expected)
    assert counts["fallback"] > 0 and counts["norm_input"] == counts["conv_input"] == 0


@pytest.mark.skipif(not CUDA_KERNELS, reason="Requires CUDA and Triton")
@pytest.mark.parametrize(
    "norm_enabled,input_enabled", [(False, True), (True, False), (True, True)]
)
@torch.no_grad()
def test_real_cuda_cnn_progressive_cache_and_retry(
    monkeypatch, norm_enabled, input_enabled
):
    from test_zing_h3_progressive_kernel import make_codecs

    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive import (
        CausalEncoderCache,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive_layout import (
        make_encoder_layout,
    )

    model = (
        make_codecs(ch_mult=(1, 2, 4))
        .modules["video_vae"]
        .cuda()
        .bfloat16()
        .requires_grad_(False)
    )
    weights = {
        layer: layer.weight.contiguous(memory_format=torch.channels_last_3d)
        for layer in model.encoder.modules()
        if isinstance(layer, torch.nn.Conv3d)
    }
    model._progressive_encoder_layouts = (
        make_encoder_layout(model),
        make_encoder_layout(model, weights=weights),
    )
    conv3d = F.conv3d
    pointwise_calls = []

    def check_pointwise_input(x, weight, *args, **kwargs):
        if weight.shape[2:] == (1, 1, 1):
            assert x.is_contiguous()
            pointwise_calls.append(x.stride())
        return conv3d(x, weight, *args, **kwargs)

    monkeypatch.setattr(F, "conv3d", check_pointwise_input)
    pixels = torch.randn(1, 3, 17, 8, 8, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT", "0")
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_FUSED_CONV_INPUT", "0")
    baseline = CausalEncoderCache(
        model, encode_mode="performance", encode_dtype="bfloat16"
    )
    monkeypatch.setenv(
        "MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT", str(int(norm_enabled))
    )
    monkeypatch.setenv(
        "MINIMAX_H3_PROGRESSIVE_FUSED_CONV_INPUT", str(int(input_enabled))
    )
    candidate = CausalEncoderCache(
        model, encode_mode="performance", encode_dtype="bfloat16"
    )
    for start, end in ((0, 1), (1, 9), (9, 17)):
        expected, astate, awork = baseline.encode(pixels[:, :, start:end])
        actual, bstate, bwork = candidate.encode(pixels[:, :, start:end])
        if norm_enabled:
            torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)
        else:
            assert torch.equal(actual, expected)
        baseline.commit(astate, awork)
        candidate.commit(bstate, bwork)
    assert (
        candidate.optimization_counts["norm_input" if norm_enabled else "conv_input"]
        > 0
    )
    assert pointwise_calls
    snapshot = {
        key: (value, value.clone(), seen)
        for key, (value, seen) in candidate.states.items()
        if value is not None
    }
    forward = model.quant_conv.forward

    def fail(*args):
        raise RuntimeError("injected quantization failure")

    monkeypatch.setattr(model.quant_conv, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        candidate.encode(pixels[:, :, :8])
    for key, (value, saved, seen) in snapshot.items():
        assert candidate.states[key][0] is value
        assert candidate.states[key][1] == seen
        assert torch.equal(value, saved)
    monkeypatch.setattr(model.quant_conv, "forward", forward)
    candidate.clear()
    assert not candidate.states
    actual, _, _ = candidate.encode(pixels[:, :, :1])
    baseline.clear()
    expected, _, _ = baseline.encode(pixels[:, :, :1])
    torch.testing.assert_close(
        actual,
        expected,
        atol=0.025 if norm_enabled else 0,
        rtol=0.025 if norm_enabled else 0,
    )
