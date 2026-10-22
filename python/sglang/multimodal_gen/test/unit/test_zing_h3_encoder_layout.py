# SPDX-License-Identifier: Apache-2.0
"""Layout propagation and temporal isolation, independent of GPU timing."""

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.norm import (
    FusedGroupNorm3D,
    TemporalIsolatedGroupNorm,
    _merge_time_to_batch,
    _split_time_from_batch,
    progressive_norm_silu,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive_layout import (
    make_encoder_layout,
)


@pytest.mark.parametrize("batch,time", [(1, 1), (1, 8), (3, 2), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("affine", [False, True])
def test_temporal_norm_preserves_layout_and_independent_planes(
    batch, time, dtype, fused, affine
):
    torch.manual_seed(349)
    x = torch.randn(batch, 32, time, 7, 9, dtype=dtype)
    cl = x.contiguous(memory_format=torch.channels_last_3d)
    merged = _merge_time_to_batch(cl)
    assert merged.is_contiguous()
    split = _split_time_from_batch(merged, batch, torch.channels_last_3d)
    assert split.is_contiguous(memory_format=torch.channels_last_3d)
    torch.testing.assert_close(split, x, atol=0, rtol=0)
    norm = (
        FusedGroupNorm3D(8, 32, affine=affine, use_t_isolated_gn=True)
        if fused
        else TemporalIsolatedGroupNorm(8, 32, affine=affine)
    ).to(dtype=dtype)
    # Reference / CPU callers retain their original contiguous output, even
    # when callers supply a channels-last tensor.
    legacy = norm(cl)
    assert legacy.is_contiguous()
    torch.testing.assert_close(legacy, norm(x), atol=0, rtol=0)
    actual = progressive_norm_silu(cl, norm)
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    expected = legacy if fused else torch.nn.functional.silu(legacy)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    changed = cl.clone()
    changed[:, :, -1] += 100
    torch.testing.assert_close(
        progressive_norm_silu(changed, norm)[:, :, :-1],
        actual[:, :, :-1],
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("batch_size", [1, 3])
def test_layout_cache_stride_phase_retry_and_reset(monkeypatch, batch_size):
    from test_zing_h3_progressive_kernel import configure_tiling, make_codecs

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
        create_session,
    )

    codecs = make_codecs()
    model = configure_tiling(codecs, False, "reflect")
    monkeypatch.setenv(
        "MINIMAX_H3_PROGRESSIVE_ENCODER_TILE_BATCH_SIZE", str(batch_size)
    )
    pixels = torch.randint(256, (3, 34, 24, 24), dtype=torch.uint8)
    baseline = create_session(codecs, capacity_frames=34)
    baseline.append(pixels, frame_start=0)
    expected = []
    while baseline.ready:
        expected.append(baseline.pop().values)
    baseline.finish()
    candidate = create_session(codecs, capacity_frames=34)
    # Force the layout mechanics on CPU; production eligibility remains CUDA/BF16.
    assert not candidate.cache.channels_last
    weights = {
        layer: layer.weight.detach().clone(memory_format=torch.channels_last_3d)
        for layer in model.encoder.modules()
        if isinstance(layer, torch.nn.Conv3d) and layer.kernel_size != (1, 1, 1)
    }
    candidate.cache._multi_frame_layout = make_encoder_layout(model, weights=weights)
    candidate.append(pixels, frame_start=0)
    first = candidate.pop()
    torch.testing.assert_close(first.values, expected[0], atol=1e-12, rtol=1e-12)
    previous = {
        k: (v[0], v[0].clone(), v[1])
        for k, v in candidate.cache.states.items()
        if v[0] is not None
    }
    original = model.quant_conv.forward

    def fail(_):
        raise RuntimeError("Injected layout failure")

    monkeypatch.setattr(model.quant_conv, "forward", fail)
    with pytest.raises(RuntimeError, match="Injected"):
        candidate.pop()
    for key, (tensor, saved, seen) in previous.items():
        assert candidate.cache.states[key][0] is tensor
        assert candidate.cache.states[key][1] == seen
        torch.testing.assert_close(tensor, saved, atol=0, rtol=0)
        # A failed multi-frame transition preserves the original startup history.
        if key[1] == "in":
            assert tensor.is_contiguous()
    monkeypatch.setattr(model.quant_conv, "forward", original)
    for value in expected[1:]:
        torch.testing.assert_close(
            candidate.pop().values, value, atol=1e-12, rtol=1e-12
        )
    candidate.finish()
    assert candidate.state_usage().retained_bytes == 0
    candidate.reset()
    candidate.append(pixels[:, :1], frame_start=0)
    torch.testing.assert_close(
        candidate.pop().values, expected[0], atol=1e-12, rtol=1e-12
    )
    candidate.dispose()
    assert candidate.state_usage().retained_bytes == 0


@pytest.mark.parametrize("batch,time", [(1, 1), (3, 8)])
def test_channels_last_temporal_slice_padding(batch, time):
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.spatial_padding import (
        spatial_pad,
    )

    x = torch.randn(batch, 32, time + 2, 9, 11).contiguous(
        memory_format=torch.channels_last_3d
    )[:, :, 1:-1]
    padding = (1, 2, 3, 1, 0, 0)
    result = spatial_pad(x, padding, "reflect")
    assert result.is_contiguous(memory_format=torch.channels_last_3d)
    torch.testing.assert_close(
        result, torch.nn.functional.pad(x, padding, "reflect"), atol=0, rtol=0
    )


@pytest.mark.parametrize("mode", ["performance", "deterministic"])
@pytest.mark.parametrize("dtype", ["float64", "float32", "bfloat16"])
def test_cpu_does_not_enter_cuda_layout_path(monkeypatch, mode, dtype):
    from test_zing_h3_progressive_kernel import make_codecs

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
        create_session,
    )

    codecs = make_codecs()
    codecs.runtime["vae_config"].update(encode_mode=mode, encode_dtype=dtype)
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST", "1")
    session = create_session(codecs, capacity_frames=8)
    assert not session.cache.channels_last
    assert session.cache._multi_frame_layout is session.cache._native_layout
    session.dispose()


@pytest.mark.parametrize("enabled,expected", [(None, True), ("0", False), ("1", True)])
def test_prepared_cuda_layout_defaults_on_with_explicit_opt_out(
    monkeypatch, enabled, expected
):
    from test_zing_h3_progressive_kernel import make_codecs

    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive import (
        CausalEncoderCache,
    )

    if enabled is None:
        monkeypatch.delenv("MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST", raising=False)
    else:
        monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST", enabled)
    model = make_codecs().modules["video_vae"]
    weights = {
        layer: layer.weight.detach().clone(memory_format=torch.channels_last_3d)
        for layer in model.encoder.modules()
        if isinstance(layer, torch.nn.Conv3d) and layer.kernel_size != (1, 1, 1)
    }
    native = make_encoder_layout(model)
    mixed = make_encoder_layout(model, weights=weights)
    model._progressive_encoder_layouts = (native, mixed)
    # Exercise eligibility only; real CUDA arithmetic is checked by the GPU gate.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    cache = CausalEncoderCache(
        model,
        tile_mode="replicated",
        encode_mode="performance",
        encode_dtype="bfloat16",
    )
    assert cache.channels_last is expected
    assert cache._multi_frame_layout is (mixed if expected else native)
    # A running session retains the strategy chosen at construction.
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST", "0" if expected else "1")
    assert cache.channels_last is expected


def test_layout_strategy_keeps_original_stage_and_pointwise_convolutions(monkeypatch):
    from test_zing_h3_progressive_kernel import make_codecs

    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive import (
        CausalEncoderCache,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_cnn import (
        EncoderFCN3D,
    )

    model = make_codecs().modules["video_vae"]
    model.encoder = (
        EncoderFCN3D(
            ch=32,
            ch_mult=[1, 2, 2],
            space_down=[1, 1, 1],
            time_down=[1, 2, 2],
            num_res_blocks=1,
            in_channels=3,
            z_channels=24,
            double_z=True,
            causal=True,
            use_t_isolated_gn=True,
        )
        .double()
        .eval()
    )
    weights = {
        layer: layer.weight.detach().clone(memory_format=torch.channels_last_3d)
        for layer in model.encoder.modules()
        if isinstance(layer, torch.nn.Conv3d) and layer.kernel_size != (1, 1, 1)
    }
    originals = {
        name: (p.data_ptr(), p.stride()) for name, p in model.named_parameters()
    }
    legacy = set(model.encoder.down[2].modules())
    original_conv = torch.nn.functional.conv3d
    observed = set()

    def conv(x, weight, *args, **kwargs):
        for layer in model.encoder.modules():
            if not isinstance(layer, torch.nn.Conv3d):
                continue
            if weight is layer.weight or weight is weights.get(layer):
                keep_native = layer in legacy or layer.kernel_size == (1, 1, 1)
                assert weight is (layer.weight if keep_native else weights[layer])
                if keep_native:
                    assert x.is_contiguous()
                else:
                    # Native CPU padding may choose a different output layout;
                    # the serving-weight selection is independent of that fallback.
                    assert weight.is_contiguous(memory_format=torch.channels_last_3d)
                observed.add(layer)
                break
        return original_conv(x, weight, *args, **kwargs)

    cache = CausalEncoderCache(model, tile_mode="replicated")
    cache._multi_frame_layout = make_encoder_layout(model, weights=weights)
    monkeypatch.setattr(torch.nn.functional, "conv3d", conv)
    with torch.no_grad():
        cache.forward(torch.randn(1, 3, 8, 8, 8, dtype=torch.float64), "test")
    assert observed == {
        layer for layer in model.encoder.modules() if isinstance(layer, torch.nn.Conv3d)
    }
    assert any(layer.kernel_size == (1, 1, 1) for layer in observed)
    assert originals == {
        name: (p.data_ptr(), p.stride()) for name, p in model.named_parameters()
    }


@pytest.mark.parametrize("fail_frame", [None, 1, 8])
def test_explicit_warmup_is_separate_and_cleans_temporary_cache(
    monkeypatch, fail_frame
):
    from test_zing_h3_progressive_kernel import make_codecs

    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
        progressive,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        ZingH3Codecs,
    )

    codecs = make_codecs()
    codecs.runtime["vae_config"]["encode_dtype"] = "bfloat16"
    model = codecs.modules["video_vae"]
    model.encoder_tiling, model.tile_size = True, 8
    events = []

    class WarmupCache:
        def __init__(self, prepared, **kwargs):
            assert prepared is model
            assert kwargs == dict(
                tile_mode="replicated",
                encode_mode="performance",
                encode_dtype="bfloat16",
            )
            events.append("construct")

        def forward(self, pixels, key):
            assert not torch.is_grad_enabled()
            assert key == "startup"
            assert pixels.shape == (1, 3, pixels.shape[2], 8, 8)
            events.append(pixels.shape[2])
            if pixels.shape[2] == fail_frame:
                raise RuntimeError("Injected tile warmup failure")
            return None, pixels.shape[2], 1

        def commit(self, pending, work):
            events.append(("commit", pending))

        def clear(self):
            events.append("clear")

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(progressive, "CausalEncoderCache", WarmupCache)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    ZingH3Codecs.prepare_visual_encoder_weights(codecs)
    assert events == []  # Preparing weights must not run inference or synchronize.
    if fail_frame:
        with pytest.raises(RuntimeError, match="Injected tile warmup"):
            ZingH3Codecs.warmup_encoder(codecs)
        assert events[-1] == "clear" and "sync" not in events
    else:
        ZingH3Codecs.warmup_encoder(codecs)
        assert events == [
            "construct",
            1,
            ("commit", 1),
            8,
            ("commit", 8),
            "clear",
            "sync",
        ]


@pytest.mark.parametrize("prepared", [False, True])
def test_encoder_warmup_skips_unprepared_or_native_layout(monkeypatch, prepared):
    from test_zing_h3_progressive_kernel import make_codecs

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        ZingH3Codecs,
    )

    codecs = make_codecs()
    if prepared:
        ZingH3Codecs.prepare_visual_encoder_weights(codecs)

    def unexpected_sync(device):
        pytest.fail("Native CPU encoder must not enter CUDA warmup")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_sync)
    ZingH3Codecs.warmup_encoder(codecs)
