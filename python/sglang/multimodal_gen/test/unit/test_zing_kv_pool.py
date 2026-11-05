# SPDX-License-Identifier: Apache-2.0
"""Zing realtime KV pool contract, capacity and admission (#171, CPU only)."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
    ZingKVGeometry,
    ZingKVPoolAdmissionError,
    ZingKVPoolConfig,
    ZingKVPoolPlan,
    ZingKVRequest,
    cleaned_free_memory,
    peak_rows,
    token_capacity,
)
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

# Historical capacity oracle explicitly exercises before-source eviction.
peak_rows = partial(peak_rows, window_eviction_timing="before_source")
build_plan = partial(ZingKVPoolPlan.build, token_capacity=250000)


def _geometry(lengths=5, split=False, short_edge=720):
    return ZingKVGeometry(16, (1, 2, 2), short_edge, ChunkPlan(lengths, split))


def _config(**overrides):
    values = dict(
        window_eviction_timing="before_source",
        max_target_area=1280 * 704,
        max_target_long_edge=1280,
    )
    values.update(overrides)
    return ZingKVPoolConfig(**values)


def _request(h=480, w=832, *, source=True, prefix=1000, **overrides):
    values = dict(
        target_height=h,
        window_eviction_timing="before_source",
        target_width=w,
        source_shape=(h, w) if source else None,
        prefix_rows=prefix,
        window_chunks=21,
        sink_chunks=3,
        rope_position_mode="block_relative",
        rope_max_frame_gap=12,
    )
    values.update(overrides)
    return ZingKVRequest(**values)


@torch.no_grad()
def _simulated_peak(
    plan,
    window,
    *,
    prefix_rows,
    per_frame,
    has_source,
    chunks,
    audio_rows=None,
    source_per_frame=None,
):
    """Drive the real cache and record rows held at each denoise step."""
    cache = ZingH3KVCache(
        1,
        window_policy=window,
        rope_position_mode="block_relative",
        window_eviction_timing="before_source",
    )

    def layers(rows):
        return ((torch.zeros(rows, 1, 1), torch.zeros(rows, 1, 1)),)

    def block(rows, index):
        return torch.zeros(rows, 3, dtype=torch.float64), torch.full((rows,), index)

    peak = 0
    if prefix_rows:
        pos, idx = block(prefix_rows, 0)
        cache.update(pos, idx, "prefix", 0, layers(prefix_rows))
        peak = cache._length
    for chunk_id in range(chunks):
        chunk = plan.chunk(chunk_id)
        frames = chunk.end - chunk.start
        rows = per_frame * frames
        cache.prepare_chunk(chunk.target_block)
        if has_source:
            source_rows = (
                rows if source_per_frame is None else source_per_frame * frames
            )
            pos, idx = block(source_rows, chunk.source_block)
            cache.update(pos, idx, "source", frames, layers(source_rows))
        # Target audio rows share the target transaction.
        rows += 0 if audio_rows is None else audio_rows(chunk)
        pos, idx = block(rows, chunk.target_block)
        cache.update(pos, idx, "active", frames)
        # The denoising target occupies the scratch tail past committed rows.
        peak = max(peak, cache._length + rows)
        cache.update(pos, idx, "final", frames, layers(rows))
    cache.clear()
    return peak


@pytest.mark.parametrize("lengths,split", [(5, False), ([1, 2, 2], False), (3, True)])
def test_mixed_spatial_grids_match_real_cache_peak(lengths, split):
    geometry, policy = _geometry(lengths, split, short_edge=736), ZingWindowPolicy(4, 1)
    expected = peak_rows(
        geometry,
        policy,
        prefix_rows=7,
        rows_per_frame=6,
        has_source=True,
        source_rows_per_frame=2,
    )
    actual = _simulated_peak(
        geometry.chunk_plan,
        policy,
        prefix_rows=7,
        per_frame=6,
        source_per_frame=2,
        has_source=True,
        chunks=20,
    )
    assert actual == expected


def test_source_480_target_736_admission_retains_slot_bound():
    plan = build_plan(_config(max_target_area=1280 * 736), _geometry(short_edge=736))
    request = _request(736, 1280, source_shape=(480, 832))
    rows = plan.admit(request)
    assert rows == 1000 + (920 + 390) * 5 * 21
    assert rows < plan.admit(_request(736, 1280)) <= plan.slot_rows
    with pytest.raises(ZingKVPoolAdmissionError):
        plan.admit(_request(736, 1280, source_shape=(768, 1280)))


@pytest.mark.parametrize("lengths,split", [(5, False), (3, True), ([1, 2, 2], False)])
@pytest.mark.parametrize("window,sink", [(21, 3), (4, 1), (2, 0), (1, 0)])
@pytest.mark.parametrize("has_source", [True, False])
@pytest.mark.parametrize("prefix_rows", [0, 7])
def test_peak_matches_real_cache_across_evictions(
    lengths, split, window, sink, has_source, prefix_rows
):
    geometry = _geometry(lengths, split)
    policy = ZingWindowPolicy(window, sink)
    expected = peak_rows(
        geometry,
        policy,
        prefix_rows=prefix_rows,
        rows_per_frame=3,
        has_source=has_source,
    )
    # Run well past the window so at least two evictions cycle every phase.
    chunks = window + 3 * len(geometry.chunk_plan.lengths) + 4
    actual = _simulated_peak(
        geometry.chunk_plan,
        policy,
        prefix_rows=prefix_rows,
        per_frame=3,
        has_source=has_source,
        chunks=chunks,
    )
    assert expected == actual


def test_peak_is_independent_of_duration():
    geometry, policy = _geometry(), ZingWindowPolicy(21, 3)
    kwargs = dict(prefix_rows=0, rows_per_frame=390, has_source=True)
    unbounded = peak_rows(geometry, policy, **kwargs)
    assert unbounded == 2 * 390 * 5 * 21
    # A finite session shorter than the window holds only what it produced.
    assert peak_rows(geometry, policy, total_frames=12, **kwargs) == 2 * 390 * 12


def test_first_chunk_split_counts_short_sinks():
    geometry, policy = _geometry(5, True), ZingWindowPolicy(3, 2)
    # Sinks are the 1-frame and 4-frame split chunks, plus one current chunk.
    assert (
        peak_rows(geometry, policy, prefix_rows=0, rows_per_frame=1, has_source=False)
        == 1 + 4 + 5
    )


def test_slot_capacity_and_budget():
    plan = build_plan(_config(), _geometry())
    assert plan.slot_rows == 250000
    bytes_ = plan.slot_bytes(
        num_layers=50, local_heads=56, head_dim=128, element_size=2
    )
    assert bytes_ == 3 * plan.slot_rows * 50 * 56 * 128 * 2


@pytest.mark.parametrize("h,w", [(480, 832), (704, 1248), (704, 1280), (832, 480)])
def test_smaller_canvases_fit_the_max_slot(h, w):
    plan = build_plan(_config(), _geometry())
    rows = plan.admit(_request(h, w))
    assert rows <= plan.slot_rows
    assert rows == 1000 + 2 * (h // 32) * (w // 32) * 5 * 21


def test_source_free_session_uses_target_rows_only():
    plan = build_plan(_config(), _geometry())
    assert plan.admit(_request(source=False)) == 1000 + 390 * 5 * 21


@pytest.mark.parametrize(
    "request_kwargs",
    [
        dict(h=480, w=848),  # not aligned to the 32-pixel grid
        dict(h=736, w=1280),  # short edge above the checkpoint bound
        dict(h=320, w=1312),  # long edge above the pool bound
        dict(h=704, w=1280, prefix=250001),  # total capacity exceeded
        dict(source_shape=(481, 864)),  # source has its own grid checks
        dict(window_chunks=None),
        dict(window_chunks=12),
        dict(sink_chunks=0),
        dict(rope_position_mode="absolute"),
        dict(rope_max_frame_gap=6),
        dict(generates_audio=True),
    ],
)
def test_out_of_contract_requests_are_rejected(request_kwargs):
    plan = build_plan(_config(), _geometry(short_edge=704))
    h, w = request_kwargs.pop("h", 480), request_kwargs.pop("w", 832)
    prefix = request_kwargs.pop("prefix", 1000)
    with pytest.raises(ZingKVPoolAdmissionError) as error:
        plan.admit(_request(h, w, prefix=prefix, **request_kwargs))
    assert error.value.code == "kv_pool_request_unsupported"
    assert error.value.message.isascii()


def test_area_bound_rejects_aligned_canvas():
    plan = build_plan(_config(max_target_area=480 * 832), _geometry())
    plan.admit(_request(480, 832))
    with pytest.raises(ZingKVPoolAdmissionError):
        plan.admit(_request(512, 832))


def _pipeline(**values):
    defaults = dict(
        window_chunks=21,
        sink_chunks=3,
        rope_position_mode="block_relative",
        rope_max_frame_gap=12,
        window_eviction_timing="before_source",
    )
    defaults.update(values)
    return SimpleNamespace(**{f"zing_kv_pool_{k}": v for k, v in defaults.items()})


def test_pipeline_config_disabled_by_default():
    assert ZingKVPoolConfig.from_pipeline_config(_pipeline(), max_sessions=8) is None


def test_pipeline_config_enforces_single_active_session():
    with pytest.raises(ValueError, match="one active"):
        ZingKVPoolConfig.from_pipeline_config(
            _pipeline(), max_sessions=2, mem_fraction_static=0.8
        )
    config = ZingKVPoolConfig.from_pipeline_config(
        _pipeline(window_chunks=4, sink_chunks=1),
        max_sessions=1,
        mem_fraction_static=0.8,
    )
    assert (config.pool_size, config.window_chunks, config.sink_chunks) == (1, 4, 1)


@pytest.mark.parametrize("entrypoint", ["pipeline", "dotted_args"])
def test_retired_capacity_fields_fail_instead_of_disabling_pool(entrypoint):
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3CausalPipelineConfig,
    )

    config = ZingH3CausalPipelineConfig()
    with pytest.raises(ValueError, match="mem-fraction-static"):
        if entrypoint == "pipeline":
            config.update_pipeline_config({"zing_kv_pool_size": 1})
        else:
            config.update_config_from_dict(
                {"pipeline.zing_kv_pool_max_prefix_rows": 1024}, prefix="pipeline"
            )


def test_static_budget_uses_pre_load_available_memory():
    gib = 1024**3
    rows, budget = token_capacity(
        before_load=80 * gib,
        after_load=50 * gib,
        mem_fraction_static=0.9,
        num_layers=50,
        local_heads=28,
        head_dim=128,
        element_size=2,
    )
    assert budget == 42 * gib
    assert rows == budget // (3 * 50 * 28 * 128 * 2)
    smaller, _ = token_capacity(
        before_load=80 * gib,
        after_load=50 * gib,
        mem_fraction_static=0.8,
        num_layers=50,
        local_heads=28,
        head_dim=128,
        element_size=2,
    )
    assert smaller < rows
    with pytest.raises(ValueError, match="Insufficient"):
        token_capacity(
            before_load=80 * gib,
            after_load=1,
            mem_fraction_static=0.8,
            num_layers=50,
            local_heads=28,
            head_dim=128,
            element_size=2,
        )


def test_static_budget_preserves_additional_runtime_reserve():
    kwargs = dict(
        before_load=80 << 30,
        after_load=50 << 30,
        mem_fraction_static=0.99,
        num_layers=50,
        local_heads=28,
        head_dim=128,
        element_size=2,
    )
    rows, budget = token_capacity(**kwargs, additional_runtime_reserve_bytes=8 << 30)
    ordinary_rows, ordinary_budget = token_capacity(**kwargs)
    assert budget == ordinary_budget - (8 << 30)
    assert rows < ordinary_rows


@pytest.mark.parametrize("prompt_prefix_frames", [0, 5])
def test_kv_fit_does_not_admit_oversized_forward_workspace(prompt_prefix_frames):
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        ZING_FIXED_WORKSPACE_BYTES,
        ZING_WORKSPACE_BYTES_PER_FEATURE,
    )

    geometry = _geometry(lengths=[3, 2], short_edge=736)
    request = _request(prefix=24574)
    # R29 original OOM: 24,574 prefix + three source + three target frames.
    forward_rows = 26914
    workspace = (
        ZING_FIXED_WORKSPACE_BYTES
        + ZING_WORKSPACE_BYTES_PER_FEATURE * 3584 * forward_rows
    )
    plan = build_plan(
        _config(),
        geometry,
        runtime_budget_bytes=workspace,
        local_hidden_width=3584,
        prefix_hidden_width=3584,
    )
    assert (
        plan.admit(request, prompt_prefix_frames=prompt_prefix_frames) < plan.slot_rows
    )
    smaller = build_plan(
        _config(),
        geometry,
        runtime_budget_bytes=workspace - 1,
        local_hidden_width=3584,
        prefix_hidden_width=3584,
    )
    with pytest.raises(ZingKVPoolAdmissionError) as caught:
        smaller.admit(request, prompt_prefix_frames=prompt_prefix_frames)
    assert "forward workspace" in caught.value.detail
    assert "forward workspace" not in str(caught.value)
    # Same slot still admits a normal session after the refusal.
    smaller.admit(_request(prefix=3259), prompt_prefix_frames=prompt_prefix_frames)


@pytest.mark.parametrize(
    "prefix,joint,forward_rows", [(0, False, 3510), (24574, True, 28864)]
)
def test_workspace_covers_cross_chunk_final_and_joint_prompt_replay(
    prefix, joint, forward_rows
):
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        ZING_FIXED_WORKSPACE_BYTES,
        ZING_WORKSPACE_BYTES_PER_FEATURE,
    )

    request = _request(prefix=prefix)
    # Three chunks for cross-chunk execution; joint prompt replacement also
    # retains five clean frames in the same forward as prefix/source/target.
    required = (
        ZING_FIXED_WORKSPACE_BYTES
        + ZING_WORKSPACE_BYTES_PER_FEATURE * 3584 * forward_rows
    )
    kwargs = dict(local_hidden_width=3584, prefix_hidden_width=3584)
    plan = build_plan(
        _config(), _geometry(lengths=[3, 2]), runtime_budget_bytes=required, **kwargs
    )
    plan.admit(request, prompt_prefix_frames=5, prompt_joint=joint)
    smaller = build_plan(
        _config(), plan.geometry, runtime_budget_bytes=required - 1, **kwargs
    )
    with pytest.raises(ZingKVPoolAdmissionError):
        smaller.admit(request, prompt_prefix_frames=5, prompt_joint=joint)


@pytest.mark.parametrize("sp", [1, 2, 4])
def test_replicated_prefix_workspace_does_not_shrink_with_sp(sp):
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        ZING_FIXED_WORKSPACE_BYTES,
        ZING_WORKSPACE_BYTES_PER_FEATURE,
    )

    request = _request(prefix=24574)
    local_width = 7168 // sp
    # The initial joint forward shards rows; serial prefix rebuilding uses
    # all 24,574 rows at width 7,168 on every rank (R33's actual OOM path).
    features = max(local_width * 26914, 7168 * 24574)
    workspace = ZING_FIXED_WORKSPACE_BYTES + ZING_WORKSPACE_BYTES_PER_FEATURE * features
    kwargs = dict(local_hidden_width=local_width, prefix_hidden_width=7168)
    plan = build_plan(
        _config(), _geometry(lengths=[3, 2]), runtime_budget_bytes=workspace, **kwargs
    )
    plan.admit(request, prompt_prefix_frames=5)
    smaller = build_plan(
        _config(), plan.geometry, runtime_budget_bytes=workspace - 1, **kwargs
    )
    with pytest.raises(ZingKVPoolAdmissionError, match="kv_pool_request_unsupported"):
        smaller.admit(request, prompt_prefix_frames=5)
    smaller.admit(_request(prefix=3259), prompt_prefix_frames=5)


def test_measurement_collects_then_releases_idle_cache(monkeypatch):
    calls = []
    import sglang.multimodal_gen.runtime.realtime.zing_kv_pool as module

    monkeypatch.setattr(module.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: calls.append("read") or (123, 456)
    )
    assert cleaned_free_memory("cuda:0") == 123
    assert calls == ["gc", "sync", "empty", "read"]


def test_prefix_and_history_share_capacity():
    plan = build_plan(_config(), _geometry(), token_capacity=5000)
    # A prefix larger than the old independent quota fits a short request.
    assert plan.admit(_request(prefix=4500, source=False, total_frames=1)) == 4890
    with pytest.raises(ZingKVPoolAdmissionError):
        plan.admit(_request(prefix=4500, source=True, total_frames=1))


@pytest.mark.parametrize(
    "overrides",
    [dict(window_chunks=None), dict(sink_chunks=21), dict(max_target_area=0)],
)
def test_invalid_contract(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def _packer(lengths=5, split=False):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
        ZingH3Packer,
    )

    arch = SimpleNamespace(patch_size=(1, 2, 2), latents_dim=2, audio_latents_dim=2)
    return ZingH3Packer(arch, num_frame_per_block=lengths, split_first_latent=split)


def _audio_geometry(packer, short_edge=720):
    return ZingKVGeometry(
        16,
        (1, 2, 2),
        short_edge,
        packer.chunk_plan,
        video_t_span=packer._video_t_span,
        time_span_period=len(packer.visual_token_frame_spans),
    )


@pytest.mark.parametrize("lengths,split", [(5, False), (3, True), ([1, 2, 2], False)])
def test_audio_frames_match_packer_mapping(lengths, split):
    packer = _packer(lengths, split)
    geometry = _audio_geometry(packer)
    total = 41
    audio_length = int(packer._video_t_span(total)) + 5  # finite tail remainder
    for chunk in packer.chunk_plan.chunks(total):
        tail = chunk.end == total
        expected = packer._audio_indices_for_video_span(
            audio_length, chunk.start, chunk.end, total
        )
        assert geometry.audio_frames(
            chunk.start, chunk.end, tail=tail, audio_length=audio_length
        ) == len(expected)
        # A continuous session never reaches EOS, so nothing is absorbed.
        continuous = packer._audio_indices_for_video_span(
            10**6, chunk.start, chunk.end, 10**6
        )
        assert geometry.audio_frames(chunk.start, chunk.end, tail=False) == len(
            continuous
        )


@pytest.mark.parametrize("lengths,split", [(5, False), (3, True), ([1, 2, 2], False)])
@pytest.mark.parametrize("window,sink", [(21, 3), (4, 1)])
@pytest.mark.parametrize("finite", [False, True])
def test_audio_peak_matches_real_cache(lengths, split, window, sink, finite):
    packer = _packer(lengths, split)
    geometry, policy = _audio_geometry(packer), ZingWindowPolicy(window, sink)
    plan = packer.chunk_plan
    # Long enough for the fractional audio pattern to cycle past the window.
    chunks = window + 12 * len(plan.lengths) * 5
    total = plan.chunk(chunks - 1).end if finite else None
    audio_length = int(packer._video_t_span(total)) + 4 if finite else None

    def audio_rows(chunk):
        tail = finite and chunk.end == total
        indices = packer._audio_indices_for_video_span(
            audio_length if finite else 10**6,
            chunk.start,
            chunk.end,
            total if finite else 10**6,
        )
        assert (
            tail or indices == list(range(indices[0], indices[0] + len(indices)))
            if indices
            else True
        )
        return 2 * len(indices)

    expected = peak_rows(
        geometry,
        policy,
        prefix_rows=5,
        rows_per_frame=3,
        has_source=True,
        total_frames=total,
        has_audio=True,
        audio_length=audio_length,
    )
    actual = _simulated_peak(
        plan,
        policy,
        prefix_rows=5,
        per_frame=3,
        has_source=True,
        chunks=chunks,
        audio_rows=audio_rows,
    )
    assert expected == actual
    silent = peak_rows(
        geometry, policy, prefix_rows=5, rows_per_frame=3, has_source=True
    )
    assert expected > silent


def test_slots_reserve_continuous_target_audio():
    packer = _packer()
    plan = build_plan(_config(), _audio_geometry(packer))
    silent = build_plan(_config(), _geometry())
    assert plan.slot_rows == silent.slot_rows
    request = _request(704, 1280, prefix=4096, generates_audio=True)
    assert plan.admit(request) < plan.slot_rows
    assert plan.admit(_request(generates_audio=True)) < plan.slot_rows
    # Without a time mapping the pool cannot account for target audio.
    with pytest.raises(ZingKVPoolAdmissionError):
        silent.admit(_request(generates_audio=True))


def test_finite_audio_tail_beyond_the_slot_is_rejected():
    packer = _packer()
    plan = build_plan(_config(), _audio_geometry(packer))
    frames = 5 * 21 * 3
    # The float span of the last frame lands just above an integer, so its
    # ceil already covers one extra latent; the tail must exceed that.
    audio = 50000
    kwargs = dict(generates_audio=True, total_frames=frames, audio_length=audio)
    # Slots reserve continuous (v2v) audio; a smaller finite session has room
    # for its EOS audio remainder, one at the full bounds does not.
    assert plan.admit(_request(480, 832, **kwargs)) < plan.slot_rows
    with pytest.raises(ZingKVPoolAdmissionError):
        plan.admit(_request(704, 1280, prefix=4096, **kwargs))
