# SPDX-License-Identifier: Apache-2.0
"""Zing realtime KV pool leases, SP agreement and stage wiring (#171, CPU only)."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.realtime import (
    ZingH3RealtimeState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
    ZingH3GenerationStage,
)
from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
    RealtimeSession,
)
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
    ZingKVGeometry,
    ZingKVPool,
    ZingKVPoolAdmissionError,
    ZingKVPoolConfig,
    ZingKVPoolPlan,
    ZingKVRequest,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_kv_pool_cache import (
    WINDOW,
    _generate,
    _inputs,
    _pointers,
    _sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
    model,
)

LAYOUT = dict(
    num_layers=2,
    local_heads=2,
    head_dim=8,
    dtype=torch.bfloat16,
    device=torch.device("cpu"),
)


def _plan(pool_size=2, **overrides):
    config = ZingKVPoolConfig(
        pool_size=pool_size,
        max_target_area=64 * 64,
        max_target_long_edge=64,
        max_prefix_rows=16,
        **{**WINDOW, **overrides},
    )
    return ZingKVPoolPlan.build(config, ZingKVGeometry(16, (1, 2, 2), 64, ChunkPlan(2)))


def _request(size=64, **overrides):
    values = dict(
        target_height=size,
        target_width=size,
        source_shape=(size, size),
        prefix_rows=3,
        window_chunks=WINDOW["window_chunks"],
        sink_chunks=WINDOW["sink_chunks"],
        rope_position_mode="block_relative",
        rope_max_frame_gap=12,
    )
    values.update(overrides)
    return ZingKVRequest(**values)


def test_pool_allocates_only_session_slots(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime import zing_kv_pool

    allocated_rows = []
    allocate = zing_kv_pool.allocate_layer_stores

    def track(rows, **layout):
        allocated_rows.append(rows)
        return allocate(rows, **layout)

    monkeypatch.setattr(zing_kv_pool, "allocate_layer_stores", track)
    plan = _plan()
    pool = ZingKVPool.allocate(plan, **LAYOUT)
    assert allocated_rows == [plan.slot_rows] * plan.config.pool_size
    assert pool.allocated_bytes() == plan.config.pool_size * plan.slot_bytes(
        num_layers=2, local_heads=2, head_dim=8, element_size=2
    )


@torch.no_grad()
def test_source_initialization_and_rolling_switch_do_not_use_a_donor(
    model, monkeypatch
):

    def forbidden(*args):
        raise AssertionError("initialization/rolling switch allocated a donor")

    from sglang.multimodal_gen.runtime.realtime import zing_kv_pool

    storage = _storage_for(model, 64)
    monkeypatch.setattr(zing_kv_pool, "allocate_layer_stores", forbidden)
    actual, _, acquired = _source_stream(model, storage)
    assert torch.isfinite(actual).all()
    assert acquired == [3]


def test_sessions_hold_exclusive_slots_until_release():
    pool = ZingKVPool.allocate(_plan(), **LAYOUT)
    first, second, third = RealtimeSession(), RealtimeSession(), RealtimeSession()
    a = pool.acquire(first, _request())
    b = pool.acquire(second, _request(32))
    assert a.slot_id != b.slot_id and pool.available == 0
    assert _pointers(a.kv_cache) != _pointers(b.kv_cache)
    with pytest.raises(ZingKVPoolAdmissionError) as error:
        pool.acquire(third, _request())
    assert error.value.code == "kv_pool_exhausted"
    assert str(error.value).splitlines()[0] == (
        "kv_pool_exhausted: The server is busy. Please try again later."
    )
    slot_id, pointers = a.slot_id, _pointers(a.kv_cache)
    a.release()
    a.release()  # idempotent
    assert pool.available == 1 and a.released
    c = pool.acquire(third, _request(32))
    assert c.slot_id == slot_id and _pointers(c.kv_cache) == pointers


def test_one_session_cannot_hold_two_slots():
    pool = ZingKVPool.allocate(_plan(), **LAYOUT)
    session = RealtimeSession()
    lease = pool.acquire(session, _request())
    with pytest.raises(ZingKVPoolAdmissionError) as error:
        pool.acquire(session, _request())
    assert error.value.code == "kv_pool_request_unsupported"
    assert pool.available == 1
    lease.release()
    # A released lease (e.g. a finished block session) allows a new session.
    pool.acquire(session, _request())


def test_rejected_request_still_joins_the_agreement():
    calls = []
    pool = ZingKVPool.allocate(
        _plan(), agree=lambda ok: calls.append(ok) or ok, **LAYOUT
    )
    with pytest.raises(ZingKVPoolAdmissionError) as error:
        pool.acquire(RealtimeSession(), _request(prefix_rows=17))
    assert error.value.code == "kv_pool_request_unsupported"
    assert "17" not in str(error.value)  # technical detail stays in logs
    assert calls == [False] and pool.available == 2


def test_sp_ranks_roll_back_a_partial_acquisition():
    world, barrier, lock = 2, Barrier(2), Lock()
    votes = []

    def agree(ok):
        with lock:
            votes.append(ok)
        barrier.wait()
        result = all(votes)
        barrier.wait()
        return result

    pools = [ZingKVPool.allocate(_plan(1), agree=agree, **LAYOUT) for _ in range(world)]
    # Rank 1 already lost its only slot, e.g. to a leaked lease.
    pools[1]._pool.acquire()

    def rank(index):
        try:
            pools[index].acquire(RealtimeSession(), _request())
        except ZingKVPoolAdmissionError as error:
            return error.code

    with ThreadPoolExecutor(world) as executor:
        codes = list(executor.map(rank, range(world)))
    assert codes == ["kv_pool_exhausted"] * world
    assert pools[0].available == 1  # the rank that succeeded rolled back


class _FailingState(BaseRealtimeState):
    def dispose(self):
        raise RuntimeError("dispose failed")


class _OtherState(BaseRealtimeState):
    disposed = False

    def dispose(self):
        type(self).disposed = True


def test_session_dispose_releases_the_lease_when_a_state_fails():
    pool = ZingKVPool.allocate(_plan(1), **LAYOUT)
    session = RealtimeSession()
    session.get_or_create_state(_FailingState)
    session.get_or_create_state(_OtherState)
    pool.acquire(session, _request())
    with pytest.raises(RuntimeError, match="dispose failed"):
        session.dispose()
    assert _OtherState.disposed and pool.available == 1
    assert session.get_state(_OtherState) is None


@torch.no_grad()
def test_block_state_leases_on_initialize_and_releases_on_dispose(model):
    sampler = _sampler(model)
    arch = model.arch
    plan = _plan(1)
    pool = ZingKVPool.allocate(
        plan,
        num_layers=len(model.blocks),
        local_heads=arch.num_attention_heads,
        head_dim=arch.attention_head_dim,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    session, state = RealtimeSession(), ZingH3RealtimeState()
    state.sampler = sampler
    stage = object.__new__(ZingH3GenerationStage)
    stage.kv_pool = pool
    stage.codecs = SimpleNamespace(spatial_ratio=16)
    inputs = _inputs(5)
    inputs.audio_shape = None
    batch = SimpleNamespace(session=session)
    torch.manual_seed(2)
    visual = torch.randn(1, 2, 5, 2, 2)
    expected, _, _ = _generate(sampler, inputs, visual)
    state.sampling = sampler.initialize(
        inputs,
        visual.clone(),
        acquire_storage=stage._realtime_storage(batch, state, inputs),
    )
    assert state.kv_lease is session.zing_kv_lease and pool.available == 0
    assert state.sampling.cache.pooled
    while state.sampling.index < len(state.sampling.spans):
        sampler.step(state.sampling)
    assert torch.equal(state.sampling.visual, expected)
    state.dispose()
    assert pool.available == 1 and state.kv_lease is None
    state.dispose()  # repeated close is harmless
    assert pool.available == 1


@torch.no_grad()
def test_block_state_rejection_leaves_the_pool_intact(model):
    sampler = _sampler(model)
    pool = ZingKVPool.allocate(_plan(1), **LAYOUT)
    state = ZingH3RealtimeState()
    state.sampler = sampler
    stage = object.__new__(ZingH3GenerationStage)
    stage.kv_pool, stage.codecs = pool, SimpleNamespace(spatial_ratio=16)
    inputs = _inputs(5, context=30)  # prefix above max_prefix_rows=16
    inputs.audio_shape = None
    with pytest.raises(ZingKVPoolAdmissionError):
        sampler.initialize(
            inputs,
            torch.randn(1, 2, 5, 2, 2),
            acquire_storage=stage._realtime_storage(
                SimpleNamespace(session=RealtimeSession()), state, inputs
            ),
        )
    assert pool.available == 1 and state.kv_lease is None


def test_stage_allocates_the_pool_and_logs_readiness(monkeypatch, caplog):
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    monkeypatch.setattr(parallel_state, "get_ulysses_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "get_sp_parallel_rank", lambda: 1)
    stage = object.__new__(ZingH3GenerationStage)
    stage.kv_pool = None
    stage.codecs = SimpleNamespace(
        spatial_ratio=16, patch_size=(1, 2, 2), max_short_edge=64
    )
    stage.packer = SimpleNamespace(
        chunk_plan=ChunkPlan(2),
        _video_t_span=lambda n: 5 / 3 * n,
        visual_token_frame_spans=(1,),
    )
    arch = SimpleNamespace(num_layers=2, num_attention_heads=4, attention_head_dim=8)
    pipeline = SimpleNamespace(
        dit_config=SimpleNamespace(arch_config=arch),
        dit_precision="bf16",
        zing_kv_pool_size=2,
        zing_kv_pool_max_target_area=64 * 64,
        zing_kv_pool_max_target_long_edge=64,
        zing_kv_pool_max_prefix_rows=16,
        zing_kv_pool_window_chunks=3,
        zing_kv_pool_sink_chunks=1,
        zing_kv_pool_rope_position_mode="block_relative",
        zing_kv_pool_rope_max_frame_gap=12,
    )
    stage.config = pipeline
    server_args = SimpleNamespace(
        pipeline_config=pipeline, realtime_max_sessions_per_worker=2
    )
    with caplog.at_level(logging.INFO):
        stage.initialize_realtime_kv_cache_pool(server_args)
    assert stage.realtime_kv_cache_pool_enabled and stage.kv_pool.size == 2
    record = next(
        r.getMessage()
        for r in caplog.records
        if "ZING_KV_CACHE_POOL_READY_JSON" in r.getMessage()
    )
    ready = json.loads(record.split(" ", 1)[1])
    assert ready["num_attention_heads_per_rank"] == 2 and ready["rank"] == 1
    assert ready["reserves_target_audio"]
    assert ready["slot_rows"] == stage.kv_pool.plan.slot_rows
    assert ready["pool_allocated_bytes"] == 2 * ready["slot_bytes"]
    assert stage.kv_pool.allocated_bytes() == ready["pool_allocated_bytes"]

    disabled = object.__new__(ZingH3GenerationStage)
    disabled.kv_pool = None
    pipeline.zing_kv_pool_size = None
    pipeline.zing_kv_pool_max_target_area = None
    pipeline.zing_kv_pool_max_target_long_edge = None
    pipeline.zing_kv_pool_max_prefix_rows = None
    disabled.initialize_realtime_kv_cache_pool(server_args)
    assert not disabled.realtime_kv_cache_pool_enabled


def _source_stream(model, storage=None):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import (
        fixture_provenance,
    )

    sampler = _sampler(model)
    sampler.processor.num_frame_per_block = [3, 2]
    sampler.processor.split_first_latent = True
    torch.manual_seed(11)
    inputs = _inputs(12)
    inputs.audio_shape = None
    source, noise = inputs.source_visual, torch.randn(1, 2, 12, 2, 2)
    new_prompt = {
        "context": torch.randn(7, 12).bfloat16(),
        "row_role": torch.zeros(7, dtype=torch.long),
    }
    acquired = []

    def acquire(positive, ready):
        acquired.append(int((positive["block_idx"] == 0).sum()))
        return storage

    session = SourceReadySession(
        sampler,
        inputs,
        seed=5,
        max_pending_chunks=1,
        max_pending_bytes=1 << 20,
        acquire_storage=acquire if storage is not None else None,
        prompt_switch_prefix_latent_frames=5,
        initial_prompt_event_id=0,
        initial_prompt_version=0,
    )
    assert (session.cache is None) == (storage is not None)
    outputs = []
    for chunk in sampler.processor.chunk_plan.chunks(12):
        if chunk.chunk_id == 3:
            session.update_prompt(new_prompt, 1)
        session.append(
            source[:, :, chunk.start : chunk.end],
            noise[:, :, chunk.start : chunk.end],
            provenance=fixture_provenance(
                100 + chunk.chunk_id, 200 + chunk.end, chunk.start, chunk.end
            ),
            final=chunk.end == 12,
        )
        outputs.append(session.step().visual)
    stats = dict(session.cache.stats)
    assert session.complete and session.closed
    return torch.cat(outputs, dim=2), stats, acquired


@torch.no_grad()
def test_source_stream_leases_at_first_step_and_matches_growth_path(model):
    expected, grown, _ = _source_stream(model)
    assert grown["compactions"] >= 2
    storage = _storage_for(model, 64)
    pointers = _pointers(storage)
    actual, pooled, acquired = _source_stream(model, storage)
    assert torch.equal(actual, expected)
    assert acquired == [3]  # once, with the packed prefix size
    assert pooled["store_allocations"] == pooled["rotated_allocations"] == 0
    assert _pointers(storage) == pointers


def _storage_for(model, rows):
    from sglang.multimodal_gen.test.unit.test_zing_h3_kv_pool_cache import _storage

    return _storage(model, rows)


@torch.no_grad()
@pytest.mark.parametrize("with_reference", [False, True])
def test_prompt_budget_rejection_preserves_live_pool_and_recovers(
    model, with_reference
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance

    sampler = _sampler(model)
    sampler.processor.num_frame_per_block = [3, 2]
    sampler.processor.split_first_latent = True
    plan = ZingKVPoolPlan.build(
        _plan(1).config,
        ZingKVGeometry(16, (1, 2, 2), 64, ChunkPlan([3, 2], True)),
    )
    pool = ZingKVPool.allocate(plan, **LAYOUT)
    inputs = _inputs(80)
    source, noise = inputs.source_visual, torch.randn_like(inputs.source_visual)
    inputs.source_visual = inputs.audio_shape = None
    inputs.condition_media = []
    if with_reference:
        inputs.conditions = [dict(content_id="ref", role="reference")]
        inputs.media = [dict(id="ref", type="image")]
        inputs.visual = {"ref": torch.zeros(1, 2, 1, 2, 2)}
    stage = object.__new__(ZingH3GenerationStage)
    stage.kv_pool, stage.packer = pool, sampler.processor
    stage.codecs = SimpleNamespace(spatial_ratio=16)
    candidate = {}
    stage.encode_condition_for_batch = lambda *args: candidate
    state = ZingSourceStreamState()
    state.inputs = inputs
    owner = RealtimeSession()
    session = state.sampler = SourceReadySession(
        sampler,
        inputs,
        seed=5,
        max_pending_chunks=1,
        max_pending_bytes=1 << 20,
        acquire_storage=state._pooled_storage(stage, owner, sampler),
        prompt_switch_prefix_latent_frames=5,
        initial_prompt_event_id=0,
    )
    state.prompt, state.prompt_event = "old", 0

    def step():
        start = session.next_latent
        end = start + session.next_chunk_size
        session.append(
            source[:, :, start:end],
            noise[:, :, start:end],
            provenance=fixture_provenance(1, end, start, end),
        )
        return session.step()

    def update(rows, event_id):
        candidate.update(
            context=torch.ones(rows, 12).bfloat16(),
            row_role=torch.zeros(rows, dtype=torch.long),
        )
        return state._update_prompt(
            stage,
            SimpleNamespace(
                sampling_params=SimpleNamespace(prompt=str(event_id)),
                realtime_event_id=event_id,
                realtime_prompt_version=event_id,
            ),
        )

    for _ in range(3):
        assert step().prompt_event_id == 0
    pointers = _pointers(state.kv_lease.kv_cache)
    history = session.cache.history(0)
    active, applied = session.prompt, session.applied_prompt
    rng = session.rng.cpu_rng.clone()
    # A valid pending update must be superseded even when the newer one fails.
    assert update(7, 1) is None and session.pending_prompt.event_id == 1
    assert update(40, 2) == {"code": "CONDITION_INVALID", "event_id": 2}
    assert session.pending_prompt is None
    assert session.prompt is active and session.applied_prompt is applied
    assert state.prompt_event == 1  # Last accepted candidate, not rejected event 2.
    assert torch.equal(rng, session.rng.cpu_rng)
    assert all(torch.equal(a, b) for a, b in zip(history, session.cache.history(0)))
    assert _pointers(state.kv_lease.kv_cache) == pointers and pool.available == 0
    # Continue beyond the late-overflow reproduction's eighteenth output block.
    for _ in range(18):
        assert step().prompt_event_id == 0
    # The inclusive limit counts reference rows as well as text context rows.
    if with_reference:
        assert update(16, 3) == {"code": "CONDITION_INVALID", "event_id": 3}
    assert update(16 - int(with_reference), 4) is None
    assert step().prompt_event_id == 4
    assert _pointers(state.kv_lease.kv_cache) == pointers
    session.dispose()
    state.kv_lease.release()
    assert pool.available == 1


@torch.no_grad()
def test_source_stream_parity_survives_in_place_qk_norm_and_dirty_memory(
    model, monkeypatch
):
    # CUDA normalizes Q/K in place, so K is a QKV view and the cache stages raw K;
    # emulate that on CPU. Fill every pooled and growth buffer with NaN so any
    # read of rows the cache never wrote poisons the output.
    from sglang.multimodal_gen.runtime.models.dits import zing_h3
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        ZingH3KVCache,
    )

    def in_place_qk_norm(module, q, k):
        q.copy_(module.q_norm(q))
        k.copy_(module.k_norm(k))
        return q, k

    staged = []
    stage_raw_k = ZingH3KVCache.stage_raw_k

    def counted(self, *args, **kwargs):
        staged.append(1)
        return stage_raw_k(self, *args, **kwargs)

    new_empty = torch.Tensor.new_empty

    def dirty_new_empty(self, *args, **kwargs):
        tensor = new_empty(self, *args, **kwargs)
        if tensor.is_floating_point():
            tensor.fill_(float("nan"))
        return tensor

    monkeypatch.setattr(zing_h3, "_qk_norm", in_place_qk_norm)
    monkeypatch.setattr(ZingH3KVCache, "stage_raw_k", counted)
    monkeypatch.setattr(torch.Tensor, "new_empty", dirty_new_empty)

    expected, _, _ = _source_stream(model)
    storage = _storage_for(model, 64)
    for store in storage:
        for tensor in (store.k, store.v, store.rotated_k):
            tensor.fill_(float("nan"))
    actual, _, _ = _source_stream(model, storage)
    assert staged
    assert torch.isfinite(expected).all()
    assert torch.equal(actual, expected)
