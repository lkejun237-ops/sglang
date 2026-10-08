# SPDX-License-Identifier: Apache-2.0
"""Real small CNN kernels on CPU; not trained-weight/GPU acceptance."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_cnn import (
    EncoderFCN3D,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
    create_session,
    expected_progressive_slices,
)


@pytest.fixture
def codecs():
    return make_codecs()


def make_codecs():
    torch.manual_seed(7)
    model = nn.Module()
    model.encoder = (
        EncoderFCN3D(
            ch=32,
            ch_mult=[1, 1, 1],
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
    model.quant_conv = nn.Conv3d(48, 48, 1).double().eval()
    model.encoder_tiling = False
    return SimpleNamespace(
        modules={"video_vae": model},
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        temporal_ratio=4,
        runtime={
            "vae_config": {"encode_dtype": "float64", "encode_mode": "performance"}
        },
    )


def full(codecs, pixels):
    model = codecs.modules["video_vae"]
    values = []
    with torch.no_grad():
        for start in range(0, pixels.shape[1], 17):
            x = pixels[:, start : start + 17]
            n = x.shape[1]
            if n < 17 and pixels.shape[1] != 1:
                x = torch.cat((x, x[:, -1:].expand(-1, 17 - n, -1, -1)), 1)
            x = x[None].double() / 255
            mean = x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
            std = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
            values.append(
                model.quant_conv(model.encoder((x - mean) / std))[
                    :, :, : 1 + (n - 1 + 3) // 4
                ]
            )
    return torch.cat(values, 2)


@pytest.mark.parametrize("frames", range(1, 35))
def test_real_cnn_causal_cache_and_eos(codecs, frames):
    pixels = torch.randint(256, (3, frames, 8, 8), dtype=torch.uint8)
    stream = create_session(codecs, capacity_frames=8)
    values, indices = [], []
    for i in range(frames):
        stream.append(pixels[:, i : i + 1], frame_start=i)
        while stream.ready:
            item = stream.pop()
            values.append(item.values)
            indices.append((item.start, item.end, item.source_dependency_end))
    stream.finish()
    while stream.ready:
        item = stream.pop()
        values.append(item.values)
        indices.append((item.start, item.end, item.source_dependency_end))
    assert indices == expected_progressive_slices(frames)
    # Float64 analytical test only. Production BF16 acceptance remains exact.
    torch.testing.assert_close(
        torch.cat(values, 2), full(codecs, pixels), atol=1e-12, rtol=1e-12
    )
    stream.dispose()
    assert stream.state_usage().retained_bytes == 0


def test_session_storage_failure_and_reset(codecs):
    pixels = torch.randint(256, (3, 8, 8, 8), dtype=torch.uint8)
    a = create_session(codecs, capacity_frames=8)
    b = create_session(codecs, capacity_frames=8)
    saved = pixels.clone()
    a.append(pixels, frame_start=0)
    pixels.zero_()
    b.append(saved, frame_start=0)
    torch.testing.assert_close(a.pop().values, b.pop().values, atol=0, rtol=0)
    usage = a.state_usage()
    with pytest.raises(ValueError):
        a.append(saved, frame_start=8)
    assert a.accepted_frames == 8 and a.state_usage() == usage
    a.reset()
    assert a.state_usage().retained_bytes == 0
    a.append(saved[:, :1], frame_start=0)
    a.pop()
    a.finish()
    assert a.state_usage().retained_bytes == 0
    a.dispose()
    a.dispose()
    with pytest.raises(ValueError):
        a.reset()


def test_fixed_work_per_clip_and_bounded_storage(codecs):
    stream = create_session(codecs, capacity_frames=17)
    pixels = torch.randint(256, (3, 17, 8, 8), dtype=torch.uint8)
    totals, peaks = [], []
    first = None
    for clip in range(4):
        stream.append(pixels, frame_start=17 * clip)
        peak = stream.state_usage().retained_bytes
        while stream.ready:
            item = stream.pop()
            if first is None:
                first = item.values.clone()
                item.values.zero_()
            peak = max(peak, stream.state_usage().retained_bytes)
        totals.append(stream.state_usage().temporal_positions)
        peaks.append(peak)
        assert stream.state_usage().retained_bytes == 0
    assert totals == [totals[0] * n for n in range(1, 5)]
    assert len(set(peaks)) == 1


def test_kernel_exception_keeps_state_retryable(codecs, monkeypatch):
    stream = create_session(codecs, capacity_frames=17)
    pixels = torch.randint(256, (3, 17, 8, 8), dtype=torch.uint8)
    stream.append(pixels, frame_start=0)
    stream.pop()
    usage = stream.state_usage()
    original = stream.cache.model.quant_conv.forward

    def fail(x):
        raise RuntimeError("injected kernel failure")

    monkeypatch.setattr(stream.cache.model.quant_conv, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        stream.pop()
    assert stream.emitted_latents == 1
    assert stream.state_usage() == usage
    monkeypatch.setattr(stream.cache.model.quant_conv, "forward", original)
    assert stream.pop().end == 3


def configure_tiling(codecs, stack, padding="zeros"):
    from types import MethodType

    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
        AutoencoderKL,
        AutoencoderKLLegacy,
    )

    model = codecs.modules["video_vae"]
    model.encoder = (
        EncoderFCN3D(
            ch=32,
            ch_mult=[1, 1, 1],
            space_down=[1, 2, 2],
            time_down=[1, 2, 2],
            num_res_blocks=1,
            in_channels=3,
            z_channels=24,
            double_z=True,
            causal=True,
            padding_mode=padding,
            use_t_isolated_gn=True,
        )
        .double()
        .eval()
    )
    model.encoder_tiling = True
    model.parallel_tiling = False
    model.stack_tiling = stack
    model.tile_size = 16
    model.tile_overlap_min = 8
    model.vae_ratio = 4
    model._blend_weight_cache = {}
    for name in (
        "split_tiles",
        "blend",
        "_assemble_tiles",
        "_local_tile_indices",
        "_run_tile_tasks",
        "tiled_encode",
    ):
        setattr(model, name, MethodType(getattr(AutoencoderKL, name), model))
    model.encode = MethodType(AutoencoderKLLegacy.encode, model)
    return model


@pytest.mark.parametrize("stack", [False, True])
@pytest.mark.parametrize("padding", ["zeros", "reflect"])
def test_native_tiling_and_packet_invariance(codecs, stack, padding):
    model = configure_tiling(codecs, stack, padding)
    pixels = torch.randint(256, (3, 17, 24, 24), dtype=torch.uint8)
    x = pixels[None].double() / 255
    x = (x - x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)) / x.new_tensor(
        (0.229, 0.224, 0.225)
    ).view(1, 3, 1, 1, 1)
    with torch.no_grad():
        reference = model.tiled_encode(x)
    results = []
    for pattern in ([1], [17], [3, 2, 7, 5]):
        stream = create_session(codecs, capacity_frames=17)
        cursor = step = 0
        values = []
        while cursor < 17:
            n = min(pattern[step % len(pattern)], 17 - cursor)
            stream.append(pixels[:, cursor : cursor + n], frame_start=cursor)
            cursor += n
            step += 1
            while stream.ready:
                values.append(stream.pop().values)
        stream.finish()
        results.append(torch.cat(values, 2))
        torch.testing.assert_close(results[-1], reference, atol=1e-12, rtol=1e-12)
    for result in results[1:]:
        torch.testing.assert_close(result, results[0], atol=0, rtol=0)


@pytest.mark.parametrize("watermark,tokens", [(1, 1), (9, 3)])
def test_cached_outputs_ignore_future_pixels(codecs, watermark, tokens):
    source = torch.randint(256, (3, 17, 8, 8), dtype=torch.uint8)
    changed = source.clone()
    changed[:, watermark:] = 255 - changed[:, watermark:]
    outputs = []
    for pixels in (source, changed):
        stream = create_session(codecs, capacity_frames=17)
        values = []
        for i in range(17):
            stream.append(pixels[:, i : i + 1], frame_start=i)
            while stream.ready:
                values.append(stream.pop().values)
        stream.finish()
        outputs.append(torch.cat(values, 2))
        stream.dispose()
    torch.testing.assert_close(
        outputs[0][:, :, :tokens], outputs[1][:, :, :tokens], atol=0, rtol=0
    )


def test_progressive_tile_mode_is_independent_and_frozen(codecs, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_ENCODER_TILING", "parallel")
    stream = create_session(codecs, capacity_frames=8)
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_ENCODER_TILING", "replicated")
    stream.append(torch.zeros(3, 1, 8, 8, dtype=torch.uint8), frame_start=0)
    stream.pop()
    receipt = stream.cache.last_execution
    assert receipt["requested_mode"] == "parallel"
    assert receipt["effective_mode"] == "replicated"
    assert receipt["fallback_reason"] == "non_tiled_encoder"
    assert receipt["gather_calls"] == 0
    assert receipt["source_dependency_end"] == 1
    stream.dispose()
    assert stream.state_usage().retained_bytes == 0
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_ENCODER_TILING", "typo")
    with pytest.raises(ValueError, match="tiling"):
        create_session(codecs, capacity_frames=8)


def test_cached_tile_geometry_cannot_change(codecs):
    model = configure_tiling(codecs, False)
    stream = create_session(codecs, capacity_frames=17)
    stream.append(torch.zeros(3, 17, 24, 24, dtype=torch.uint8), frame_start=0)
    stream.pop()
    usage = stream.state_usage()
    model.stack_tiling = True
    with pytest.raises(ValueError, match="layout"):
        stream.pop()
    assert stream.state_usage() == usage
    stream.dispose()


@pytest.mark.parametrize(
    "size,mode,dtype,stack,expected_reason",
    [
        (2, "deterministic", "bfloat16", False, None),
        (4, "deterministic", "bfloat16", False, None),
        (1, "deterministic", "bfloat16", False, "single_rank_or_uninitialized_group"),
        (8, "deterministic", "bfloat16", False, "unvalidated_group_size"),
        (2, "performance", "bfloat16", False, None),
        (4, "performance", "bfloat16", False, None),
        (2, "deterministic", "float32", False, "unvalidated_encode_dtype"),
        (2, "deterministic", "bfloat16", True, "unvalidated_stack_tiling"),
    ],
)
def test_default_tile_policy_requires_validated_configuration(
    codecs, monkeypatch, size, mode, dtype, stack, expected_reason
):
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import klvae

    monkeypatch.delenv("MINIMAX_H3_PROGRESSIVE_ENCODER_TILING", raising=False)
    monkeypatch.delenv("MINIMAX_H3_PROGRESSIVE_ENCODER_PERFORMANCE_AUTO", raising=False)
    group = SimpleNamespace(world_size=size)
    monkeypatch.setattr(klvae, "get_tile_parallel_state", lambda: (0, size))
    monkeypatch.setattr(klvae, "get_decode_parallel_group_coordinator", lambda: group)
    configure_tiling(codecs, stack)
    codecs.runtime["vae_config"].update(encode_mode=mode, encode_dtype=dtype)
    stream = create_session(codecs, capacity_frames=8)
    assert stream.cache.requested_mode == "auto"
    assert stream.cache.tile_mode == (
        "parallel" if expected_reason is None else "replicated"
    )
    assert stream.cache.default_fallback_reason == expected_reason
    assert (stream.cache.group is group) == (expected_reason is None)
    stream.dispose()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_performance_auto_override_is_session_scoped(codecs, monkeypatch, enabled, size):
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import klvae

    monkeypatch.delenv("MINIMAX_H3_PROGRESSIVE_ENCODER_TILING", raising=False)
    name = "MINIMAX_H3_PROGRESSIVE_ENCODER_PERFORMANCE_AUTO"
    monkeypatch.setenv(name, str(int(enabled)))
    monkeypatch.setattr(klvae, "get_tile_parallel_state", lambda: (0, size))
    group = SimpleNamespace(world_size=size)
    monkeypatch.setattr(klvae, "get_decode_parallel_group_coordinator", lambda: group)
    configure_tiling(codecs, False)
    codecs.runtime["vae_config"].update(
        encode_mode="performance", encode_dtype="bfloat16"
    )
    stream = create_session(codecs, capacity_frames=8)
    expected = "parallel" if enabled and size in (2, 4) else "replicated"
    assert stream.cache.tile_mode == expected
    monkeypatch.setenv(name, str(int(not enabled)))
    stream.reset()
    assert stream.cache.tile_mode == expected
    assert stream.cache.performance_auto is enabled
    stream.dispose()


@pytest.mark.parametrize(
    "key,value", [("encode_mode", "deterministic"), ("encode_dtype", "float32")]
)
def test_session_rejects_changed_numerical_policy(codecs, key, value):
    stream = create_session(codecs, capacity_frames=8)
    stream.append(torch.zeros(3, 1, 8, 8, dtype=torch.uint8), frame_start=0)
    codecs.runtime["vae_config"][key] = value
    with pytest.raises(ValueError, match="policy cannot change"):
        stream.pop()
    assert stream.emitted_latents == 0
    stream.dispose()


def test_explicit_replicated_overrides_approved_default(codecs, monkeypatch):
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import klvae

    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_ENCODER_TILING", "replicated")
    monkeypatch.setattr(klvae, "get_tile_parallel_state", lambda: (0, 2))
    monkeypatch.setattr(
        klvae,
        "get_decode_parallel_group_coordinator",
        lambda: SimpleNamespace(world_size=2),
    )
    configure_tiling(codecs, False)
    codecs.runtime["vae_config"].update(
        encode_mode="deterministic", encode_dtype="bfloat16"
    )
    stream = create_session(codecs, capacity_frames=8)
    assert stream.cache.requested_mode == "replicated"
    assert stream.cache.group is None
    stream.dispose()


@pytest.mark.parametrize("batch_size", [1, 2, 3, 6])
@pytest.mark.parametrize("frames", [1, 8, 17, 19, 34])
def test_bounded_tile_batches_preserve_history(codecs, monkeypatch, batch_size, frames):
    configure_tiling(codecs, False)
    name = "MINIMAX_H3_PROGRESSIVE_ENCODER_TILE_BATCH_SIZE"
    monkeypatch.setenv(name, "1")
    baseline = create_session(codecs, capacity_frames=frames + 8)
    monkeypatch.setenv(name, str(batch_size))
    candidate = create_session(codecs, capacity_frames=frames + 8)
    pixels = torch.randint(256, (3, frames, 32, 32), dtype=torch.uint8)
    for stream in (baseline, candidate):
        stream.append(pixels, frame_start=0)
        stream.finish()
    while baseline.ready:
        expected, actual = baseline.pop(), candidate.pop()
        assert (actual.start, actual.end, actual.source_dependency_end) == (
            expected.start,
            expected.end,
            expected.source_dependency_end,
        )
        torch.testing.assert_close(
            actual.values, expected.values, atol=1e-12, rtol=1e-12
        )
        receipt = candidate.cache.last_execution
        ids = [i for batch in receipt["tile_batches"] for i in batch]
        assert sorted(ids) == receipt["local_tile_ids"]
        assert max(map(len, receipt["tile_batches"])) <= batch_size
        if candidate.cache.states:
            assert {key[0] for key in candidate.cache.states} == {
                tuple(batch) for batch in receipt["tile_batches"]
            }
    assert not candidate.ready
    candidate.reset()
    assert candidate.state_usage().retained_bytes == 0
    candidate.dispose()
    assert candidate.state_usage().retained_bytes == 0


@pytest.mark.parametrize("value", ["0", "-1", "2.5", "bad"])
def test_tile_batch_rejects_invalid_size(codecs, monkeypatch, value):
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_ENCODER_TILE_BATCH_SIZE", value)
    with pytest.raises(ValueError):
        create_session(codecs, capacity_frames=8)


def test_bounded_batches_group_unequal_edges_and_restore_order(codecs, monkeypatch):
    model = configure_tiling(codecs, True)
    model.split_tiles = lambda length, is_decoder=False: (
        [0, 8, 20],
        [16, 16, 12],
        [8, 4],
    )
    name = "MINIMAX_H3_PROGRESSIVE_ENCODER_TILE_BATCH_SIZE"
    monkeypatch.setenv(name, "1")
    baseline = create_session(codecs, capacity_frames=17)
    monkeypatch.setenv(name, "3")
    candidate = create_session(codecs, capacity_frames=17)
    pixels = torch.randint(256, (3, 17, 32, 32), dtype=torch.uint8)
    for stream in (baseline, candidate):
        stream.append(pixels, frame_start=0)
    while baseline.ready:
        torch.testing.assert_close(
            candidate.pop().values, baseline.pop().values, atol=1e-12, rtol=1e-12
        )
        receipt = candidate.cache.last_execution
        assert receipt["tile_batches"] == [[0, 1, 3], [4], [2, 5], [6, 7], [8]]
        assert not receipt["stack_tiling"]
    candidate.finish()
    baseline.finish()
    assert candidate.state_usage().retained_bytes == 0


def test_batched_failure_keeps_all_tile_histories_retryable(codecs, monkeypatch):
    model = configure_tiling(codecs, False)
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_ENCODER_TILE_BATCH_SIZE", "2")
    candidate = create_session(codecs, capacity_frames=17)
    candidate.append(
        torch.randint(256, (3, 17, 32, 32), dtype=torch.uint8), frame_start=0
    )
    candidate.pop()
    previous = dict(candidate.cache.states)
    usage = candidate.state_usage()
    original = model.quant_conv.forward
    calls = 0

    def fail_on_second_group(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Injected batched tile failure")
        return original(value)

    monkeypatch.setattr(model.quant_conv, "forward", fail_on_second_group)
    with pytest.raises(RuntimeError, match="Injected"):
        candidate.pop()
    assert candidate.state_usage() == usage
    assert all(candidate.cache.states[k] is v for k, v in previous.items())
    monkeypatch.setattr(model.quant_conv, "forward", original)
    assert candidate.pop().start == 1
    candidate.dispose()
    assert candidate.state_usage().retained_bytes == 0


@pytest.mark.parametrize("mode", [None, "off", "reflect2d"])
def test_spatial_padding_default_override_and_session_snapshot(
    codecs, monkeypatch, mode
):
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.spatial_padding import (
        spatial_pad,
    )

    key = "MINIMAX_H3_PROGRESSIVE_SPATIAL_PADDING"
    if mode is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, mode)
    stream = create_session(codecs, capacity_frames=8)
    expected = mode or "reflect2d"
    assert stream.cache.spatial_padding == expected
    assert stream.cache._pad is (
        spatial_pad if expected == "reflect2d" else torch.nn.functional.pad
    )
    monkeypatch.setenv(key, "off" if expected == "reflect2d" else "reflect2d")
    assert stream.cache.spatial_padding == expected
    stream.dispose()
