# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401, F811
"""Delayed window timing contracts. CPU arithmetic is not GPU acceptance.

The compatibility probe in _sampler deliberately exercises the old engine even
before the new constructor argument exists: the red trace must demonstrate an
incorrect eviction boundary, rather than merely a missing Python argument.
"""

import inspect

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import allocate_layer_stores
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
    model,
    request,
)


def _sampler(model, packer, **kwargs):
    if "window_eviction_timing" in inspect.signature(ZingH3Sampler).parameters:
        return ZingH3Sampler(
            model, packer, window_eviction_timing="after_step0", **kwargs
        )
    sampler = ZingH3Sampler(model, packer, **kwargs)
    sampler.window_eviction_timing = "after_step0"
    return sampler


def _retained(policy, current):
    return [i for i in range(current) if policy.retains(i, current)]


def _ids(cache):
    return [c.temporal_chunk_id for c in cache.committed]


def _require_cache_api(cache):
    assert callable(getattr(cache, "begin_chunk", None)), (
        "missing two-stage begin_chunk boundary"
    )
    assert callable(getattr(cache, "complete_step0_and_evict", None)), (
        "missing scheduler-completion eviction boundary"
    )


def _cache(window=3, sink=1, pooled=False):
    kwargs = dict(window_policy=ZingWindowPolicy(window, sink))
    if "window_eviction_timing" in inspect.signature(ZingH3KVCache).parameters:
        kwargs["window_eviction_timing"] = "after_step0"
    if pooled:
        kwargs["storage"] = allocate_layer_stores(
            128,
            num_layers=2,
            local_heads=1,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
        )
    cache = ZingH3KVCache(2, **kwargs)
    if "window_eviction_timing" not in inspect.signature(ZingH3KVCache).parameters:
        cache.window_eviction_timing = "after_step0"
    return cache


def _rows(block):
    return torch.tensor([[float(block), 0, 0], [float(block), 0, 1]]), torch.full(
        (2,), block
    )


def _layers(value):
    return tuple(
        (
            torch.full((2, 1, 2), float(value + layer)),
            torch.full((2, 1, 2), float(100 + value + layer)),
        )
        for layer in range(2)
    )


def _commit(cache, chunk):
    cache.begin_chunk(chunk.target_block)
    pos, blocks = _rows(chunk.source_block)
    cache.update(pos, blocks, "source", 1, _layers(chunk.source_block))
    pos, blocks = _rows(chunk.target_block)
    cache.update(pos, blocks, "active", 1)
    cache.complete_step0_and_evict(chunk.target_block)
    cache.update(pos, blocks, "final", 1, _layers(chunk.target_block))


@torch.no_grad()
@pytest.mark.parametrize("mode", ["off", "source_first", "cross_chunk"])
@pytest.mark.parametrize("steps,source", [(2, True), (3, True), (2, False)])
def test_real_attention_reads_old_history_until_scheduler_step0(
    model, monkeypatch, mode, steps, source
):
    packer, inputs, visual, _ = request(model, source=source, audio=False)
    inputs.visual_shape = (7, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 7, 2, 2) if source else None
    visual = torch.randn(1, 2, 7, 2, 2)
    sampler = _sampler(
        model,
        packer,
        sigma_points=steps,
        window_chunks=1,
        sink_chunks=0,
        rope_position_mode="absolute",
        joint_forward=mode,
    )
    trace = []
    original = ZingH3KVCache.update

    def update(cache, positions, blocks, phase, frames, layers=None, validation=None):
        if phase in {"source", "active", "final"}:
            block = int(blocks[0])
            chunk_id = block // 2 if phase == "source" else block // 2 - 1
            trace.append((phase, chunk_id, tuple(_ids(cache))))
        return original(cache, positions, blocks, phase, frames, layers, validation)

    monkeypatch.setattr(ZingH3KVCache, "update", update)
    sampler.sample(inputs, visual)
    assert torch.isfinite(visual).all()
    # First forward of each active chunk observes the preceding final, even W=1.
    # Later forwards and final observe the shrunken committed window.
    seen_active = set()
    for phase, chunk_id, history in trace:
        first_active = phase == "active" and chunk_id not in seen_active
        expected = () if chunk_id == 0 else (chunk_id - 1,)
        if phase == "source" or first_active:
            assert history == expected, (
                f"{mode} chunk={chunk_id} {phase}: old history {expected} was evicted before step0; actual={history}"
            )
        else:
            assert history == (), (
                f"{mode} chunk={chunk_id} {phase}: committed history must shrink after scheduler step0"
            )
        if phase == "active":
            seen_active.add(chunk_id)
    assert len(seen_active) == 4


@pytest.mark.parametrize(
    "window,sink,pooled",
    [
        (w, s, p)
        for w, s in [(1, 0), (3, 0), (3, 1), (21, 0), (21, 3)]
        for p in [False, True]
    ]
    + [(None, 0, False)],
)
@torch.no_grad()
def test_cache_preserves_pending_active_and_raw_values_after_compaction(
    window, sink, pooled
):
    cache = _cache(window, sink, pooled)
    _require_cache_api(cache)
    plan = ChunkPlan(1)
    # More than a window boundary, with differently valued layers and branches.
    for i in range(26):
        chunk = plan.chunk(i)
        old_ids = _ids(cache)
        cache.begin_chunk(chunk.target_block)
        assert _ids(cache) == old_ids
        source_pos, source_blocks = _rows(chunk.source_block)
        cache.update(
            source_pos, source_blocks, "source", 1, _layers(chunk.source_block)
        )
        target_pos, target_blocks = _rows(chunk.target_block)
        cache.update(target_pos, target_blocks, "active", 1)
        active = cache.active
        pending = cache.pending_source
        pending_values = [(k.clone(), v.clone()) for k, v in pending.layers]
        cache.complete_step0_and_evict(chunk.target_block)
        assert _ids(cache) == _retained(cache.window_policy, i)
        assert cache.active is active and cache.pending_source is pending
        assert cache._length == sum(
            c.positions.shape[0] for c in cache._stored_chunks()
        )
        offset = 0
        for retained in cache._stored_chunks():
            assert retained.offset == offset
            offset += retained.positions.shape[0]
            for layer, (k, v) in enumerate(retained.layers):
                torch.testing.assert_close(
                    k, cache._stores[layer][0][retained.offset : offset]
                )
                torch.testing.assert_close(
                    v, cache._stores[layer][1][retained.offset : offset]
                )
                if retained is not pending:
                    cid = retained.temporal_chunk_id
                    wanted_k = torch.cat(
                        (_layers(2 * cid + 1)[layer][0], _layers(2 * cid + 2)[layer][0])
                    )
                    wanted_v = torch.cat(
                        (_layers(2 * cid + 1)[layer][1], _layers(2 * cid + 2)[layer][1])
                    )
                    torch.testing.assert_close(k, wanted_k, rtol=0, atol=0)
                    torch.testing.assert_close(v, wanted_v, rtol=0, atol=0)
        for pair, expected in zip(pending.layers, pending_values):
            for actual, wanted in zip(pair, expected):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        cache.update(target_pos, target_blocks, "final", 1, _layers(chunk.target_block))
        assert cache.active is cache.pending_source is None
        assert cache.last_block == chunk.target_block
    cache.clear()
    assert getattr(cache, "eviction_target_block", None) is None
    assert not cache.committed and cache._length == 0


@torch.no_grad()
def test_cache_rejects_duplicate_wrong_and_unfinished_boundaries():
    cache = _cache(1, 0)
    _require_cache_api(cache)
    with pytest.raises(ValueError):
        cache.complete_step0_and_evict(2)
    cache.begin_chunk(2)
    with pytest.raises(ValueError):
        cache.begin_chunk(2)
    with pytest.raises(ValueError):
        cache.complete_step0_and_evict(4)
    pos, blocks = _rows(2)
    cache.update(pos, blocks, "active", 1)
    cache.complete_step0_and_evict(2)
    with pytest.raises(ValueError):
        cache.complete_step0_and_evict(2)
    cache.update(pos, blocks, "final", 1, _layers(2))
    with pytest.raises(ValueError):
        cache.begin_chunk(6)
    cache.clear()
    _commit(cache, ChunkPlan(1).chunk(0))
    assert cache.last_block == 2


@torch.no_grad()
def test_10000_chunk_state_machine_remains_bounded():
    cache = _cache(3, 1)
    _require_cache_api(cache)
    peak_capacity = 0
    for i in range(10000):
        _commit(cache, ChunkPlan(1).chunk(i))
        assert len(cache.committed) <= 3
        assert len(cache._chunk_geometry) <= 5
        assert len(cache._rope_coefficients) <= 5
        peak_capacity = max(
            peak_capacity,
            max((pair[0].shape[0] for pair in cache._stores.values()), default=0),
        )
        assert peak_capacity <= 64
    assert cache.stats["store_allocations"] < 20


def test_delayed_policy_graph_is_conservatively_eager(model):
    from types import SimpleNamespace

    cache = _cache()
    reason = model._denoise_graph_rejection(
        SimpleNamespace(x=torch.zeros(1)),
        options=None,
        cache=cache,
        host_layout=None,
        ulysses_size=1,
        mask=None,
    )
    assert reason == "window_eviction_timing"


@torch.no_grad()
@pytest.mark.parametrize("mode", ["source_first", "cross_chunk"])
@pytest.mark.parametrize("rope", ["absolute", "block_relative"])
def test_independent_sequential_real_attention_oracle(model, monkeypatch, mode, rope):
    packer, inputs, visual, _ = request(model, audio=False)
    inputs.visual_shape = (9, 2, 2)
    torch.manual_seed(914)
    inputs.source_visual = torch.randn(1, 2, 9, 2, 2)
    visual = torch.randn(1, 2, 9, 2, 2)
    snapshots = []
    original = ZingH3KVCache.update

    def update(cache, positions, blocks, phase, frames, layers=None, validation=None):
        result = original(cache, positions, blocks, phase, frames, layers, validation)
        if phase in {"source", "final"}:
            snapshots.append(
                (phase, positions.clone(), [(k.clone(), v.clone()) for k, v in layers])
            )
        return result

    monkeypatch.setattr(ZingH3KVCache, "update", update)
    outputs, states = [], []
    for selected in ("off", mode):
        sampler = _sampler(
            model,
            packer,
            sigma_points=3,
            window_chunks=1,
            sink_chunks=0,
            rope_position_mode=rope,
            joint_forward=selected,
        )
        if selected == "off":

            def forbidden(*args, **kwargs):
                raise AssertionError("sequential reference entered joint API")

            monkeypatch.setattr(sampler, "_predict_joint", forbidden)
        snapshots.clear()
        torch.manual_seed(71)
        outputs.append(sampler.sample(inputs, visual.clone())[0])
        states.append(list(snapshots))
        if selected == "cross_chunk":
            assert (
                sampler.last_run_metrics["joint_fallbacks"].get("window_eviction", 0)
                == 0
            )
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.008, atol=0.008)
    assert len(states[0]) == len(states[1])
    for a, b in zip(*states):
        assert a[0] == b[0]
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
        for ap, bp in zip(a[2], b[2]):
            for av, bv in zip(ap, bp):
                torch.testing.assert_close(av, bv, rtol=0.008, atol=0.008)


def test_policy_cannot_change_mid_session(model):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = _sampler(model, packer, sigma_points=2)
    state = sampler.initialize(inputs, visual)
    try:
        with pytest.raises((AttributeError, ValueError)):
            sampler.window_eviction_timing = "before_source"
        with pytest.raises((AttributeError, ValueError)):
            state.cache.window_eviction_timing = "before_source"
    finally:
        state.close()


@torch.no_grad()
@pytest.mark.parametrize("mode", ["off", "source_first", "cross_chunk"])
@pytest.mark.parametrize(
    "failure", ["source", "active", "final", "scheduler", "compaction"]
)
def test_delayed_failure_closes_whole_state(model, monkeypatch, mode, failure):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = _sampler(
        model,
        packer,
        sigma_points=2,
        window_chunks=1,
        sink_chunks=0,
        joint_forward=mode,
    )
    state = sampler.initialize(inputs, visual)
    if failure == "scheduler":

        def fail(*args, **kwargs):
            raise RuntimeError("injected scheduler")

        monkeypatch.setattr(sampler.video_scheduler, "step", fail)
    elif failure == "compaction":

        def fail(*args, **kwargs):
            raise RuntimeError("injected compaction")

        monkeypatch.setattr(state.cache, "_compact_store", fail)
    else:
        original = state.cache.update

        def update(positions, blocks, phase, *args, **kwargs):
            if phase == failure:
                raise RuntimeError("injected phase")
            return original(positions, blocks, phase, *args, **kwargs)

        monkeypatch.setattr(state.cache, "update", update)
    with pytest.raises(RuntimeError, match="injected"):
        while state.index < len(state.spans):
            sampler.step(state)
    assert state.closed
    assert state.cache.active is state.cache.pending_source is None
    assert not state.cache.committed
    assert state.cache.eviction_target_block is None


@pytest.mark.parametrize("bad", ["policy", "capacity"])
def test_sp_boundary_rejects_divergent_rank_before_forward(model, monkeypatch, bad):
    from types import SimpleNamespace

    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.models.dits import zing_h3

    packer, *_ = request(model, audio=False)
    sampler = _sampler(model, packer, sigma_points=2)
    cache = _cache()
    assert callable(getattr(sampler, "_agree_sp_boundary", None)), (
        "missing pre-collective cache agreement"
    )
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (4, 0))
    monkeypatch.setattr(
        zing_h3, "get_sp_group", lambda: SimpleNamespace(cpu_group=object())
    )

    def gather(votes, payload, group):
        votes[:] = [payload.copy() for _ in range(4)]
        votes[2]["policy" if bad == "policy" else "capacity_ok"] = (
            "before_source" if bad == "policy" else False
        )

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(ValueError, match="rank"):
        sampler._agree_sp_boundary(cache, 2)
    assert cache.eviction_target_block is None and cache.active is None


@pytest.mark.parametrize("solver,batch_size", [("euler", 2), ("few_step", 20)])
def test_delayed_ready_batch_rng_and_resume(model, monkeypatch, solver, batch_size):
    from sglang.multimodal_gen.test.unit import test_zing_h3_joint as joint_tests

    def sampler(*args, **kwargs):
        return ZingH3Sampler(*args, window_eviction_timing="after_step0", **kwargs)

    monkeypatch.setattr(joint_tests, "ZingH3Sampler", sampler)
    joint_tests.test_ready_lookahead_matches_streaming_and_preserves_rng(
        model, 1, 0, solver, batch_size
    )


def test_delayed_ready_close_releases_prefetched_state(model, monkeypatch):
    from sglang.multimodal_gen.test.unit import test_zing_h3_joint as joint_tests

    def sampler(*args, **kwargs):
        return ZingH3Sampler(*args, window_eviction_timing="after_step0", **kwargs)

    monkeypatch.setattr(joint_tests, "ZingH3Sampler", sampler)
    joint_tests.test_ready_lookahead_close_releases_prefetched_state(model)


@torch.no_grad()
def test_zero_step_rejects_and_clears_open_boundary(model):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = _sampler(model, packer, sigma_points=2)
    state = sampler.initialize(inputs, visual)
    sampler.video_scheduler.timesteps = torch.empty(0)
    with pytest.raises(ValueError, match="at least one"):
        sampler.step(state)
    assert state.closed and state.cache.eviction_target_block is None


@pytest.mark.parametrize("bad", [[], {}])
def test_non_string_policy_reports_value_error(model, bad):
    packer, *_ = request(model, audio=False)
    with pytest.raises(ValueError, match="timing"):
        ZingH3Sampler(model, packer, window_eviction_timing=bad)
    with pytest.raises(ValueError, match="timing"):
        ZingH3KVCache(2, window_eviction_timing=bad)


def test_initial_policy_agreement_includes_default_rank(model, monkeypatch):
    from types import SimpleNamespace

    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.models.dits import zing_h3

    packer, *_ = request(model, audio=False)
    sampler = ZingH3Sampler(model, packer)
    assert callable(getattr(sampler, "_agree_sp_policy", None)), (
        "missing initial policy agreement"
    )
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (4, 0))
    monkeypatch.setattr(
        zing_h3, "get_sp_group", lambda: SimpleNamespace(cpu_group=object())
    )

    def gather(votes, payload, group):
        votes[:] = [payload, payload, "after_step0", payload]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(ValueError, match="rank"):
        sampler._agree_sp_policy()


@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize("mode", ["source_first", "cross_chunk"])
@torch.no_grad()
def test_random_scheduler_draws_exactly_once_in_video_audio_order(model, audio, mode):
    packer, inputs, visual, sound = request(model, audio=audio)
    draws, outputs = [], []
    for selected in ("off", mode):
        sampler = _sampler(
            model,
            packer,
            sigma_points=3,
            window_chunks=1,
            sink_chunks=0,
            rope_position_mode="absolute",
            joint_forward=selected,
        )
        trace = []
        for kind, scheduler in [
            ("video", sampler.video_scheduler),
            ("audio", sampler.audio_scheduler),
        ]:
            original = scheduler.step

            def step(prediction, sample, *, step_index, kind=kind, original=original):
                draw = torch.rand(())
                trace.append((kind, step_index, float(draw)))
                result = original(prediction, sample, step_index=step_index)
                return (result[0] + draw * 0.0001, result[1])

            scheduler.step = step
        torch.manual_seed(519)
        outputs.append(
            sampler.sample(
                inputs, visual.clone(), None if sound is None else sound.clone()
            )
        )
        draws.append(trace)
        blocks = len(sampler.last_run_metrics["block_spans"])
        expected = [
            (kind, step)
            for _ in range(blocks)
            for step in range(2)
            for kind in (["video", "audio"] if audio else ["video"])
        ]
        assert [(kind, step) for kind, step, draw in trace] == expected
    assert draws[0] == draws[1]
    for expected, actual in zip(outputs[0], outputs[1]):
        if expected is not None:
            torch.testing.assert_close(expected, actual, rtol=0.008, atol=0.008)


@torch.no_grad()
def test_joint_precheck_votes_before_deciding_fallback(model, monkeypatch):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = _sampler(model, packer, sigma_points=2, joint_forward="source_first")
    state = sampler.initialize(inputs, visual)
    source, target, start, end = state.spans[0]
    sampler._begin_chunk(state.cache, state.positive, source, target)
    sampler._set_block_latents(
        state.positive, target, state.visual[:, :, start:end], None
    )
    packs = [
        sampler._select_block(
            state.positive, source, phase="source", video_t=1, audio_t=None
        ),
        sampler._select_block(
            state.positive, target, phase="target", video_t=0, audio_t=None
        ),
    ]
    votes = []

    def reject(cache, target, **kwargs):
        votes.append(kwargs)
        return False

    monkeypatch.setattr(sampler, "_agree_sp_boundary", reject)
    try:
        result = sampler._predict_joint(
            packs,
            ("source", "active"),
            state.visual[:, :, start:end].shape[2:],
            state.cache,
        )
        assert result is None, "joint precheck must honor peer fallback before forward"
        assert len(votes) == 1 and votes[0]["joint_eligible"] is True
        assert state.cache.pending_source is state.cache.active is None
    finally:
        state.close()


@torch.no_grad()
def test_delayed_policy_recording_graph_never_captures(model, monkeypatch):
    from sglang.multimodal_gen.test.unit.test_zing_h3_denoise_graph import (
        RecordingGraphBackend,
        _graph_mode,
    )

    backend = RecordingGraphBackend()
    graphs = _graph_mode(model, monkeypatch, backend)
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = _sampler(model, packer, sigma_points=3, window_chunks=1, sink_chunks=0)
    sampler.sample(inputs, visual)
    assert backend.captures == 0
    assert graphs.routes["window_eviction_timing"] > 0


@pytest.mark.parametrize("key", ["guidance", "schedule", "layout_signature"])
def test_sp_boundary_rejects_different_collective_contract(model, monkeypatch, key):
    from types import SimpleNamespace

    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.models.dits import zing_h3

    packer, *_ = request(model, audio=False)
    sampler = _sampler(model, packer, sigma_points=3)
    cache = _cache()
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (4, 0))
    monkeypatch.setattr(
        zing_h3, "get_sp_group", lambda: SimpleNamespace(cpu_group=object())
    )

    def gather(votes, payload, group):
        votes[:] = [payload.copy() for _ in range(4)]
        votes[1][key] = ("divergent collective configuration",)

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(ValueError, match="rank"):
        sampler._agree_sp_boundary(cache, 2)
    assert not cache.committed and cache.active is None


@torch.no_grad()
def test_initial_prefix_capacity_is_voted_before_model(model, monkeypatch):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = _sampler(model, packer, sigma_points=2)
    original = sampler._agree_sp_boundary
    checked = []

    def agree(cache, target, **kwargs):
        if cache._length == 0:
            prefix_rows = kwargs["layout_signature"][2][0][1]
            assert prefix_rows > 0
            assert kwargs.get("required_rows", 0) == prefix_rows, (
                "prefix capacity must be voted before model work"
            )
            checked.append(prefix_rows)
        return original(cache, target, **kwargs)

    monkeypatch.setattr(sampler, "_agree_sp_boundary", agree)
    state = sampler.initialize(inputs, visual)
    try:
        assert checked and state.cache.prefix.positions.shape[0] == checked[0]
    finally:
        state.close()


@torch.no_grad()
def test_ready_prefix_workspace_capacity_refuses_before_forward(model, monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance

    packer, inputs, visual, _ = request(model, audio=False)
    sampler = _sampler(model, packer, sigma_points=2, window_chunks=1, sink_chunks=0)
    sampler.prefix_workspace = allocate_layer_stores(
        1,
        num_layers=len(model.blocks),
        local_heads=model.arch.num_attention_heads,
        head_dim=model.arch.attention_head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )
    session = SourceReadySession(
        sampler,
        inputs,
        seed=71,
        max_pending_chunks=4,
        max_pending_bytes=1000000,
        total_latent_frames=visual.shape[2],
    )
    chunk = packer.chunk_plan.chunk(0, visual.shape[2])
    session.append(
        inputs.source_visual[:, :, chunk.start : chunk.end],
        visual[:, :, chunk.start : chunk.end],
        provenance=fixture_provenance(1, 20, chunk.start, chunk.end),
    )
    forwards = []
    original = model.forward

    def forward(*args, **kwargs):
        forwards.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", forward)
    with pytest.raises(ValueError, match="(?i)(capacity|rank|pooled)"):
        session.step()
    assert session.closed and not session.cache.committed
    assert (
        session.cache.prefix
        is session.cache.active
        is session.cache.pending_source
        is None
    )
    assert not forwards, "undersized donor must be refused before entering model work"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
