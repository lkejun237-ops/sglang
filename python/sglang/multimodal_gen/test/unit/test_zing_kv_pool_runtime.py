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


def _plan(pool_size=1, **overrides):
    config = ZingKVPoolConfig(
        max_target_area=64 * 64,
        max_target_long_edge=64,
        **{**WINDOW, **overrides},
    )
    return ZingKVPoolPlan.build(
        config, ZingKVGeometry(16, (1, 2, 2), 64, ChunkPlan(2)), token_capacity=80
    )


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
    first, third = RealtimeSession(), RealtimeSession()
    a = pool.acquire(first, _request())
    assert pool.available == 0
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
    assert pool.available == 0
    lease.release()
    # A released lease (e.g. a finished block session) allows a new session.
    pool.acquire(session, _request())


def test_rejected_request_still_joins_the_agreement():
    calls = []
    pool = ZingKVPool.allocate(
        _plan(), agree=lambda ok: calls.append(ok) or ok, **LAYOUT
    )
    with pytest.raises(ZingKVPoolAdmissionError) as error:
        pool.acquire(RealtimeSession(), _request(prefix_rows=81))
    assert error.value.code == "kv_pool_request_unsupported"
    assert "81" not in str(error.value)  # technical detail stays in logs
    assert calls == [False] and pool.available == 1


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
    fail = True

    def dispose(self):
        if self.fail:
            raise RuntimeError("dispose failed")


class _OtherState(BaseRealtimeState):
    disposed = False

    def dispose(self):
        type(self).disposed = True


def test_session_dispose_retains_the_lease_until_failed_state_is_cleaned():
    pool = ZingKVPool.allocate(_plan(1), **LAYOUT)
    session = RealtimeSession()
    failed = session.get_or_create_state(_FailingState)
    session.get_or_create_state(_OtherState)
    pool.acquire(session, _request())
    with pytest.raises(RuntimeError, match="dispose failed"):
        session.dispose()
    assert _OtherState.disposed and pool.available == 0
    assert session.get_state(_OtherState) is None
    assert session.get_state(_FailingState) is failed
    with pytest.raises(ZingKVPoolAdmissionError):
        pool.acquire(RealtimeSession(), _request())
    failed.fail = False
    session.dispose()
    assert pool.available == 1
    assert session.get_state(_FailingState) is None
    session.dispose()
    assert pool.available == 1


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
    inputs = _inputs(5, context=81)  # prefix exceeds the entire shared capacity
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


@pytest.mark.parametrize("peer_budget", ["local", "smaller", "error"])
def test_stage_allocates_the_pool_and_logs_readiness(monkeypatch, caplog, peer_budget):
    from sglang.multimodal_gen.runtime.conditioning import worker
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    def coordinate(value):
        if isinstance(value, tuple) and len(value) == 3 and isinstance(value[1], int):
            if peer_budget == "error":
                return [value, ("Insufficient peer memory", None, value[2])]
            if peer_budget == "smaller":
                return [value, (None, 7, value[2] - 1)]
        return [value]

    monkeypatch.setattr(worker, "coordinate_validation", coordinate)
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage.ZING_MIN_RUNTIME_BYTES",
        0,
    )
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
        zing_kv_pool_window_chunks=3,
        zing_kv_pool_sink_chunks=1,
        zing_kv_pool_rope_position_mode="block_relative",
        zing_kv_pool_rope_max_frame_gap=12,
        zing_kv_pool_window_eviction_timing="after_step0",
    )
    stage.config = pipeline
    server_args = SimpleNamespace(
        pipeline_config=pipeline,
        realtime_max_sessions_per_worker=1,
        mem_fraction_static=0.8,
        zing_block0_cache_mode="off",
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage.cleaned_free_memory",
        lambda device: 20000,
    )
    if peer_budget == "error":
        with pytest.raises(ValueError, match="at least one rank"):
            stage.initialize_realtime_kv_cache_pool(server_args, before_load=20000)
        assert stage.kv_pool is None
        return
    with caplog.at_level(logging.INFO):
        stage.initialize_realtime_kv_cache_pool(server_args, before_load=20000)
    assert stage.realtime_kv_cache_pool_enabled and stage.kv_pool.size == 1
    record = next(
        r.getMessage()
        for r in caplog.records
        if "ZING_KV_CACHE_POOL_READY_JSON" in r.getMessage()
    )
    ready = json.loads(record.split(" ", 1)[1])
    assert ready["num_attention_heads_per_rank"] == 2 and ready["rank"] == 1
    assert ready["shared_condition_and_target_budget"]
    assert ready["runtime_budget_bytes"] == (3999 if peer_budget == "smaller" else 4000)
    assert ready["prefix_hidden_width"] == 32
    assert stage.kv_pool.plan.local_hidden_width == 16
    assert ready["slot_rows"] == stage.kv_pool.plan.slot_rows
    if peer_budget == "smaller":
        assert ready["slot_rows"] == 7
    assert ready["pool_allocated_bytes"] == ready["slot_bytes"]
    assert stage.kv_pool.allocated_bytes() == ready["pool_allocated_bytes"]

    disabled = object.__new__(ZingH3GenerationStage)
    disabled.kv_pool = None
    server_args.mem_fraction_static = None
    disabled.initialize_realtime_kv_cache_pool(server_args)
    assert not disabled.realtime_kv_cache_pool_enabled


def _source_stream(model, storage=None):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance

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
def test_reference_admission_counts_candidate_rows(model):
    from copy import copy

    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.media_geometry import (
        SourceGeometry,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )

    sampler = _sampler(model)
    stage = object.__new__(ZingH3GenerationStage)
    stage.kv_pool = SimpleNamespace(plan=_plan(1))
    stage.packer, stage.codecs = sampler.processor, SimpleNamespace(spatial_ratio=16)
    state = ZingSourceStreamState()
    state.inputs = _inputs(8)
    state.inputs.source_visual = state.inputs.audio_shape = None
    height, width = (
        n * stage.codecs.spatial_ratio for n in state.inputs.visual_shape[-2:]
    )
    state.source_geometry = SourceGeometry(height, width, height, width, height, width)
    state.sampler = SimpleNamespace(sampler=sampler)
    condition = dict(
        context=torch.zeros(16, 12).bfloat16(),
        row_role=torch.zeros(16, dtype=torch.long),
    )
    request = state._admit_prompt_update(stage, condition)
    from dataclasses import replace

    stage.kv_pool.plan = replace(
        stage.kv_pool.plan, slot_rows=stage.kv_pool.plan.admit(request)
    )
    candidate = copy(state.inputs)
    candidate.conditions = [
        dict(content_id="new", role="reference", latent_modalities=["visual"])
    ]
    candidate.media = [dict(id="new", type="image")]
    candidate.visual = {"new": torch.ones(1, 2, 1, 2, 2)}
    with pytest.raises(ConditionError):
        state._admit_prompt_update(stage, condition, candidate)
    assert state.inputs.visual == {}


@torch.no_grad()
@pytest.mark.parametrize("with_reference", [False, True])
def test_prompt_budget_rejection_preserves_live_pool_and_recovers(
    model, with_reference
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.media_geometry import (
        SourceGeometry,
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
        token_capacity=80,
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
    height, width = (n * stage.codecs.spatial_ratio for n in inputs.visual_shape[-2:])
    state.source_geometry = SourceGeometry(height, width, height, width, height, width)
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
                extra={"zing_source_command": {}},
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
    assert update(81, 2) == {"code": "CONDITION_INVALID", "event_id": 2}
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
        assert update(62, 3) == {"code": "CONDITION_INVALID", "event_id": 3}
    assert update(62 - int(with_reference), 4) is None
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
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache

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


@pytest.mark.parametrize("pool_independent", [False, True])
def test_compute_input_gate_runs_before_encoding(monkeypatch, pool_independent):
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        codecs as module,
    )

    codecs = object.__new__(module.ZingH3Codecs)
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import ZingComputeLimits

    codecs.compute_plan = (
        ZingComputeLimits((32, 32), 64) if pool_independent else _plan()
    )
    codecs.pipeline_config = SimpleNamespace(zing_reference_max_images=2)
    codecs.max_short_edge = 64
    codecs.conditioning_mode = "local"
    codecs.modules = codecs.runtime = {}
    calls = []
    monkeypatch.setattr(
        module, "encode_condition", lambda *a, **kw: calls.append(kw) or {}
    )
    image = dict(id="ref", type="image", frames=torch.zeros(3, 1, 32, 32))
    codecs.encode_condition("accepted", [image])
    assert calls[0]["max_rows"] == 16384
    for prompt, media in [
        ("x" * 65537, [image]),
        ("ok", [image] * 3),
        ("ok", [dict(image, frames=torch.zeros(3, 1, 96, 96))]),
    ]:
        with pytest.raises(ConditionError):
            codecs.encode_condition(prompt, media)
    assert len(calls) == 1
    with pytest.raises(ConditionError):
        codecs.validate_compute_inputs("ok", [image], target_shape=(96, 96))


@torch.no_grad()
def test_finite_prompt_rejection_keeps_generation_equal_to_control(model):
    from contextlib import nullcontext
    from copy import deepcopy

    original = _inputs(80)
    original.audio_shape = None
    original.condition_media = []
    states = []
    for _ in range(2):
        stage = object.__new__(ZingH3GenerationStage)
        stage.kv_pool = ZingKVPool.allocate(_plan(), **LAYOUT)
        stage.codecs = SimpleNamespace(spatial_ratio=16)
        stage._use = lambda *args: nullcontext(model)
        sampler = _sampler(model)
        sampler.processor.num_frame_per_block = [3, 2]
        sampler.processor.split_first_latent = True
        stage.packer = sampler.processor
        state = ZingH3RealtimeState()
        state.inputs, state.sampler = deepcopy(original), sampler
        state.prompt = state.requested_prompt = "initial"
        state.prompt_event = 0
        stage.encode_condition_for_batch = lambda batch, prompt, media: dict(
            context=torch.ones(81 if prompt == "oversized" else 5, 12).bfloat16(),
            row_role=torch.zeros(81 if prompt == "oversized" else 5, dtype=torch.long),
        )
        batch = SimpleNamespace(
            sampling_params=SimpleNamespace(prompt="initial", seed=13),
            realtime_event_id=0,
            session=RealtimeSession(),
        )
        states.append((stage, state, batch))

    def advance():
        for stage, state, batch in states:
            stage._sample_realtime_block(batch, state)
        assert torch.equal(states[0][1].sampling.visual, states[1][1].sampling.visual)
        assert torch.equal(
            states[0][1].sampling.ready.rng.cpu_rng,
            states[1][1].sampling.ready.rng.cpu_rng,
        )

    for _ in range(3):
        advance()
    stage, state, batch = states[1]
    pointers = _pointers(state.kv_lease.kv_cache)
    batch.sampling_params.prompt, batch.realtime_event_id = "oversized", 1
    advance()
    assert state.conditioning_notice == {"code": "CONDITION_INVALID", "event_id": 1}
    assert state.prompt == state.requested_prompt == "initial"
    assert state.sampling.ready.pending_prompt is None
    assert _pointers(state.kv_lease.kv_cache) == pointers
    batch.sampling_params.prompt = "initial"
    for _ in range(3):
        advance()
    for _, _, current in states:
        current.sampling_params.prompt, current.realtime_event_id = "valid", 2
    for _ in range(6):
        advance()
    assert state.prompt == "valid" and state.conditioning_notice is None
    for current_stage, current_state, _ in states:
        current_state.dispose()
        assert current_stage.kv_pool.available == 1


@pytest.mark.parametrize(
    "kind", ["boundary", "long_target", "long_source", "large_source", "audio"]
)
def test_finite_compute_rejection_precedes_all_encoders(kind):
    from contextlib import nullcontext

    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        ZingH3Codecs,
    )

    codecs = object.__new__(ZingH3Codecs)
    codecs.compute_plan = _plan()
    codecs.pipeline_config = SimpleNamespace(zing_reference_max_images=2)
    codecs.max_short_edge, codecs.max_area_config = 768, {"source": 64 * 64}
    codecs.temporal_ratio, codecs.canvas_multiple, codecs.target_fps = 4, 32, 24
    codecs.spatial_ratio, codecs.patch_size = 16, (1, 2, 2)
    codecs.vae_frames_per_chunk, codecs.vae_latents_per_chunk = 17, 5
    codecs.max_source_target_aspect_ratio_factor = 1.5
    frames = (
        721
        if kind in ("long_target", "long_source")
        else 720
        if kind == "boundary"
        else 1
    )
    source = (
        dict(
            type="video",
            fps=24,
            frames=torch.zeros(
                3, frames, *([736, 736] if kind == "large_source" else [32, 32])
            ),
        )
        if kind in ("long_source", "large_source")
        else None
    )
    target = dict(
        type="video_audio" if kind == "audio" else "video",
        output=dict(width=32, height=32, frames=frames),
    )
    codecs._parse_inference_request = lambda request: ("prompt", [], target, [], source)
    codecs._decode_media = lambda item, **kwargs: item
    calls = []

    def encode(*args):
        calls.append(args)
        raise RuntimeError("Reached encoder")

    expected = RuntimeError if kind == "boundary" else ConditionError
    with pytest.raises(expected):
        codecs.prepare(
            {"h3_max_area_config": {"source": 736 * 736}},
            lambda name: nullcontext(),
            conditioning=encode,
        )
    assert bool(calls) == (kind == "boundary")


@pytest.mark.parametrize("limit", ["kv", "compute", "source_compute"])
def test_cached_update_rejection_precedes_gpu_restore(monkeypatch, limit):
    from copy import deepcopy

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        preparation_cache,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        ZingH3Codecs,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )

    stage = object.__new__(ZingH3GenerationStage)
    stage.kv_pool = ZingKVPool.allocate(_plan(), **LAYOUT)
    stage.codecs = object.__new__(ZingH3Codecs)
    stage.codecs.compute_plan = stage.kv_pool.plan
    stage.codecs.pipeline_config = SimpleNamespace(zing_reference_max_images=2)
    stage.codecs.spatial_ratio = 16
    active = _inputs(1)
    active.source_visual = active.audio_shape = None
    active.pixel_frame_count = 1
    candidate = deepcopy(active)
    rows = 81 if limit == "kv" else 3
    candidate.positive = dict(
        context=torch.ones(rows, 12).bfloat16(),
        row_role=torch.zeros(rows, dtype=torch.long),
    )
    if limit == "compute":
        candidate.pixel_frame_count = 721
    if limit == "source_compute":
        candidate.source_visual = torch.zeros(1, 4, 1, 46, 46)
    entry = SimpleNamespace(inputs=candidate, request=_request(32, prefix_rows=rows))
    state = ZingSourceStreamState()
    state.inputs, state.prompt, state.prompt_event = active, "current", 0
    state.block0_cache = SimpleNamespace(lookup=lambda key: entry)
    state._block0_update_key = lambda *args: ("cached", {})
    state.sampler = SimpleNamespace(discard_pending_prompt=lambda: None)
    owner = RealtimeSession()
    lease = stage.kv_pool.acquire(owner, _request(32))
    pointers = _pointers(lease.kv_cache)

    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected cache data was copied to GPU")

    monkeypatch.setattr(preparation_cache, "inputs_on", forbidden)
    notice = state._update_prompt(
        stage,
        SimpleNamespace(
            sampling_params=SimpleNamespace(prompt="new"), realtime_event_id=1
        ),
    )
    assert notice == {"code": "CONDITION_INVALID", "event_id": 1}
    assert state.inputs is active and state.prompt == "current"
    assert stage.kv_pool.available == 0 and _pointers(lease.kv_cache) == pointers
    lease.release()
    assert stage.kv_pool.available == 1


def test_reference_compute_area_is_independent_of_output_canvas():
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        ZingH3Codecs,
    )

    codecs = object.__new__(ZingH3Codecs)
    codecs.compute_plan = _plan()
    codecs.pipeline_config = SimpleNamespace(zing_reference_max_images=2)
    codecs.max_short_edge = 2048
    codecs.spatial_ratio = 16
    # The default checkpoint normalizes a 16:9 reference to 1344x768,
    # even when the output is 832x480 or 1280x736.
    media = dict(type="image", frames=torch.zeros(3, 1, 768, 1344, dtype=torch.uint8))
    codecs.validate_compute_inputs("prompt", [media], target_shape=(64, 64))
    inputs = SimpleNamespace(
        audio_shape=None,
        audio={},
        pixel_frame_count=17,
        visual_shape=(5, 4, 4),
        positive={"context": torch.zeros(3, 8)},
        conditions=[{}],
        visual={"ref": torch.zeros(1, 24, 1, 48, 84)},
    )
    codecs.validate_prepared_compute_inputs(inputs)
    media["frames"] = torch.zeros(3, 1, 800, 1344, dtype=torch.uint8)
    with pytest.raises(ConditionError):
        codecs.validate_compute_inputs("prompt", [media])
    inputs.visual["ref"] = torch.zeros(1, 24, 1, 50, 84)
    with pytest.raises(ConditionError):
        codecs.validate_prepared_compute_inputs(inputs)


@pytest.mark.parametrize(
    "operation,limit",
    [
        ("condition", "kv"),
        ("lookup", "kv"),
        ("prefill", "kv"),
        ("condition", "switch"),
        ("lookup", "switch"),
    ],
)
def test_disaggregated_cache_rejection_precedes_tensor_restore(
    monkeypatch, operation, limit
):
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        dit_preparation,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        ZingH3Codecs,
    )

    stage = object.__new__(ZingH3GenerationStage)
    if limit == "switch":
        plan = ZingKVPoolPlan.build(
            ZingKVPoolConfig(window_chunks=1, sink_chunks=0),
            ZingKVGeometry(16, (1, 2, 2), 64, ChunkPlan([3, 2])),
            token_capacity=28,
        )
        request = _request(64, source_shape=(32, 32), window_chunks=1, sink_chunks=0)
    else:
        plan, request = _plan(), _request(32, prefix_rows=81)
    stage.kv_pool = ZingKVPool.allocate(plan, **LAYOUT)
    stage.codecs = object.__new__(ZingH3Codecs)
    stage.codecs.compute_plan = stage.kv_pool.plan
    stage.codecs.pipeline_config = SimpleNamespace(zing_reference_max_images=2)
    stage.codecs.spatial_ratio = 16
    candidate = _inputs(
        1, size=4 if limit == "switch" else 2, context=3 if limit == "switch" else 81
    )
    candidate.source_visual = candidate.audio_shape = None
    candidate.pixel_frame_count = 1
    entry = SimpleNamespace(inputs=candidate, request=request)
    stage.block0_cache = SimpleNamespace(
        lookup=lambda *args, **kwargs: entry, last_lookup={"cache_result": "l1_hit"}
    )
    owner = RealtimeSession()
    lease = (
        None
        if operation == "prefill"
        else stage.kv_pool.acquire(
            owner, request if limit == "switch" else _request(32)
        )
    )
    available = stage.kv_pool.available
    pointers = None if lease is None else _pointers(lease.kv_cache)

    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected cache data reached GPU restoration")

    monkeypatch.setattr(dit_preparation, "inputs_on", forbidden)
    preparation = dit_preparation.DiTPreparation(stage, forbidden)
    arguments = {"cache_key": "a" * 64, "prompt_update": limit == "switch"}
    if operation == "condition":
        with pytest.raises(ConditionError):
            preparation.cached_condition(arguments, initial=False)
    else:
        assert preparation.control(
            operation, arguments, busy=operation == "lookup"
        ) == {"code": "CONDITION_INVALID"}
    assert stage.kv_pool.available == available
    if lease is not None:
        assert _pointers(lease.kv_cache) == pointers
        lease.release()


@pytest.mark.parametrize("limit", ["compute", "kv", "host_cache"])
def test_disaggregated_prefill_rejection_precedes_tensor_restore(monkeypatch, limit):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        dit_preparation,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        ZingH3Codecs,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
        H3InferenceInputs,
    )

    stage = object.__new__(ZingH3GenerationStage)
    stage.kv_pool = ZingKVPool.allocate(_plan(), **LAYOUT)
    stage.codecs = object.__new__(ZingH3Codecs)
    stage.codecs.compute_plan = stage.kv_pool.plan
    stage.codecs.pipeline_config = SimpleNamespace(zing_reference_max_images=2)
    stage.codecs.spatial_ratio = 16
    candidate = H3InferenceInputs(
        output_type="video",
        conditions=[],
        media=[],
        positive=dict(
            context=torch.ones(3 if limit == "host_cache" else 81, 12),
            row_role=torch.zeros(3 if limit == "host_cache" else 81),
        ),
        visual={},
        audio={},
        source_visual=None,
        visual_shape=(1, 2, 2),
        audio_shape=None,
        pixel_frame_count=721 if limit == "compute" else 1,
        sample_count=None,
        sample_rate=24000,
    )
    stage.block0_cache = SimpleNamespace(
        lookup=lambda *a, **kw: None,
        can_store=lambda rows: False,
        last_lookup={"cache_result": "miss", "miss_reason": "not_resident"},
    )
    stage.packer = SimpleNamespace(
        visual_channels=2,
        pack_inference_layout=lambda *args: {
            "block_idx": torch.zeros(3 if limit == "host_cache" else 81)
        },
    )
    config = stage.kv_pool.plan.config
    sampler = SimpleNamespace(
        window_policy=config.window_policy,
        rope_position_mode=config.rope_position_mode,
        rope_max_frame_gap=config.rope_max_frame_gap,
        window_eviction_timing=config.window_eviction_timing,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected preparation reached GPU restoration")

    monkeypatch.setattr(dit_preparation, "inputs_on", forbidden)
    preparation = dit_preparation.DiTPreparation(stage, lambda: sampler)
    assert preparation.control(
        "prefill",
        {
            "cache_key": "a" * 64,
            "inputs": dit_preparation.prepared_message(candidate),
            "source_shape": (2, 2),
        },
        busy=False,
    ) == {
        "code": "cache_capacity_exceeded"
        if limit == "host_cache"
        else "CONDITION_INVALID"
    }
    assert stage.kv_pool.available == 1


def test_dit_worker_forwards_static_memory_ratio_before_model_loading(monkeypatch):
    import sys

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        dit_worker,
        encoder_worker,
    )

    class StopBeforeModelLoading(RuntimeError):
        pass

    captured = {}

    class Args:
        @staticmethod
        def from_kwargs(**kwargs):
            captured.update(kwargs)
            raise StopBeforeModelLoading

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dit-worker",
            "--model",
            "/unused",
            "--sp-degree",
            "2",
            "--mem-fraction-static",
            "0.8",
        ],
    )
    monkeypatch.setattr(
        encoder_worker, "initialize_runtime_imports", lambda: (Args, lambda args: None)
    )
    with pytest.raises(StopBeforeModelLoading):
        dit_worker.main()
    assert captured["mem_fraction_static"] == 0.8
    assert captured["sp_degree"] == 2


@pytest.mark.parametrize("timing", ["before_source", "after_step0"])
@pytest.mark.parametrize("has_source", [False, True])
@torch.no_grad()
def test_small_window_prompt_switch_checks_rebuild_and_next_step(
    model, timing, has_source
):
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
        ZingH3Packer,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.realtime import (
        ZingH3BlockSession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
        ZingH3Sampler,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import peak_rows
    from sglang.multimodal_gen.test.unit.test_zing_h3_kv_pool_cache import _storage

    for supports_switch in (False, True):
        packer = ZingH3Packer(
            model.arch, num_frame_per_block=[3, 2], split_first_latent=False
        )
        sampler = ZingH3Sampler(
            model,
            packer,
            sigma_points=2,
            window_chunks=1,
            sink_chunks=0,
            window_eviction_timing=timing,
        )
        sampler.device = torch.device("cpu")
        inputs = _inputs(20, size=4)
        inputs.source_visual = torch.randn(1, 2, 20, 2, 2) if has_source else None
        inputs.audio_shape = None
        geometry = ZingKVGeometry(
            16, tuple(model.arch.patch_size), 64, packer.chunk_plan
        )
        arguments = dict(
            prefix_rows=3,
            rows_per_frame=4,
            has_source=has_source,
            source_rows_per_frame=1 if has_source else None,
            total_frames=20,
            window_eviction_timing=timing,
        )
        rows = peak_rows(
            geometry,
            sampler.window_policy,
            prompt_prefix_frames=5 if supports_switch else 0,
            **arguments,
        )
        stage = object.__new__(ZingH3GenerationStage)
        stage.packer, stage.codecs = packer, SimpleNamespace(spatial_ratio=16)
        stage.kv_pool = SimpleNamespace(
            plan=ZingKVPoolPlan.build(
                ZingKVPoolConfig(
                    window_chunks=1, sink_chunks=0, window_eviction_timing=timing
                ),
                geometry,
                token_capacity=rows,
            )
        )
        session = ZingH3BlockSession(
            sampler,
            inputs,
            torch.randn(1, 2, 20, 4, 4),
            seed=2,
            initial_rng_state=(torch.get_rng_state(), None),
            acquire_storage=lambda layout: _storage(model, rows),
        )
        try:
            session.step()
            session.step()
            prefix = session.ready.cache.prefix
            candidate = dict(inputs.positive, context=inputs.positive["context"] + 0.01)

            def admit():
                stage.admit_realtime_condition(
                    sampler,
                    inputs,
                    candidate,
                    source_shape=(2, 2) if has_source else None,
                    total_frames=20,
                )

            if supports_switch:
                admit()
                session.update_prompt(candidate, 1, 1)
            else:
                with pytest.raises(ConditionError) as error:
                    admit()
                assert error.value.code == "CONDITION_INVALID"
                assert session.ready.cache.prefix is prefix
            session.step()
            assert session.prompt_updated is supports_switch
            assert not session.ready.closed
        finally:
            session.close()
