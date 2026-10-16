# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401, F811
# Imported pytest fixtures are discovered by name and injected into test arguments.
"""Delayed-window contracts with real CPU DiT and production SP packing.

Only communication is exchanged through blocking four-rank barriers. This is
neither CUDA/NCCL validation nor a GPU performance measurement.
"""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    sampling,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_joint import (
    four_head_model,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (
    cpu_execution_options,
    model,
    request,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_sp_cpu import _CPUCollectives


def test_sp4_rank_local_prompt_transition_rejects_before_model(
    four_head_model, monkeypatch
):
    entry = "step"
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.models.dits import zing_h3
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        allocate_layer_stores,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance

    comm = _CPUCollectives(4)
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (4, comm.thread.rank))
    monkeypatch.setattr(
        zing_h3,
        "get_sp_group",
        lambda: SimpleNamespace(cpu_group=object(), ulysses_group=None),
    )

    def gather_objects(output, value, group=None):
        output[:] = comm.exchange(value, lambda values, rank: list(values))

    monkeypatch.setattr(sampling.dist, "all_gather_object", gather_objects)
    monkeypatch.setattr(zing_h3, "gather_sequence_varlen", comm.gather)
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: 4)
    monkeypatch.setattr(usp, "get_ulysses_parallel_rank", lambda: comm.thread.rank)
    monkeypatch.setattr(usp, "_ipc_ready_group", lambda: None)
    monkeypatch.setattr(usp, "_usp_all_to_all_single_varlen", comm.all_to_all)
    rank_models = [
        zing_h3.ZingH3DiTModel(
            ZingH3DiTConfig(arch_config=four_head_model.arch), {}
        ).eval()
        for _ in range(4)
    ]
    for rank_model in rank_models:
        rank_model.load_state_dict(four_head_model.state_dict())
    original_forward = ZingH3Sampler._forward
    observed = []
    preparing_failure = [False] * 4
    pack_errors = []
    pools = []
    leases = []
    sessions = []

    def forward(*args, **kwargs):
        observed.append((comm.thread.rank, preparing_failure[comm.thread.rank]))
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(ZingH3Sampler, "_forward", forward)
    packer, inputs, visual, _ = request(four_head_model, audio=False)
    original_pack_chunk = packer.pack_chunk

    def pack_chunk(inputs, prompt, visual, audio, **kwargs):
        failing_chunk = 1
        if kwargs["chunk"].chunk_id == failing_chunk:
            preparing_failure[comm.thread.rank] = True
        if (
            entry == "lookahead"
            and comm.thread.rank == 1
            and kwargs["chunk"].chunk_id == 1
        ):
            prompt = {k: v.clone() for k, v in prompt.items()}
            prompt["row_role"][0] = 4
        try:
            return original_pack_chunk(inputs, prompt, visual, audio, **kwargs)
        except ValueError as error:
            pack_errors.append((comm.thread.rank, str(error)))
            raise

    monkeypatch.setattr(packer, "pack_chunk", pack_chunk)
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, None)
    prefix_rows = int((layout["block_idx"] == 0).sum())
    assert prefix_rows > 1
    caches = []
    original_clear = ZingH3KVCache.clear

    def clear(cache):
        original_clear(cache)
        caches.append(cache)

    monkeypatch.setattr(ZingH3KVCache, "clear", clear)

    def worker(rank):
        comm.thread.rank = rank
        sampler = ZingH3Sampler(
            rank_models[rank],
            packer,
            window_eviction_timing="after_step0",
            sigma_points=2,
            joint_forward="cross_chunk",
        )
        sampler.prefix_workspace = allocate_layer_stores(
            128,
            num_layers=len(four_head_model.blocks),
            local_heads=1,
            head_dim=8,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        storage = allocate_layer_stores(
            128,
            num_layers=len(four_head_model.blocks),
            local_heads=1,
            head_dim=8,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        from sglang.multimodal_gen.runtime.realtime.causal_kv_cache_pool import (
            RealtimeKVCachePool,
            RealtimeKVCachePoolSlot,
        )

        pool = RealtimeKVCachePool([RealtimeKVCachePoolSlot(0, [storage], [])])
        pools.append(pool)
        held = []

        def acquire(*args):
            lease = pool.acquire()
            held.append(lease)
            leases.append(lease)
            return lease.kv_cache[0]

        session = SourceReadySession(
            sampler,
            inputs,
            seed=51,
            max_pending_chunks=4,
            max_pending_bytes=1000000,
            total_latent_frames=3,
            acquire_storage=acquire,
        )
        sessions.append(session)
        chunk = packer.chunk_plan.chunk(0, 3)
        session.append(
            inputs.source_visual[:, :, chunk.start : chunk.end],
            visual[:, :, chunk.start : chunk.end],
            provenance=fixture_provenance(
                chunk.chunk_id + 1, 20, chunk.start, chunk.end
            ),
        )
        if entry == "lookahead":
            next_chunk = packer.chunk_plan.chunk(1, 3)
            session.append(
                inputs.source_visual[:, :, next_chunk.start : next_chunk.end],
                visual[:, :, next_chunk.start : next_chunk.end],
                provenance=fixture_provenance(
                    next_chunk.chunk_id + 1, 20, next_chunk.start, next_chunk.end
                ),
            )
        session.step()
        next_chunk = packer.chunk_plan.chunk(1, 3)
        session.append(
            inputs.source_visual[:, :, next_chunk.start : next_chunk.end],
            visual[:, :, next_chunk.start : next_chunk.end],
            provenance=fixture_provenance(
                next_chunk.chunk_id + 1, 20, next_chunk.start, next_chunk.end
            ),
        )
        if rank == 1:
            condition = {k: v.clone() for k, v in inputs.positive.items()}
            session.update_prompt(condition, "changed")
        try:
            if entry == "step":
                session.step()
            else:
                list(session.drain_ready())
        except BaseException as error:
            return type(error).__name__, str(error), session.closed
        finally:
            session.dispose()
            for lease in held:
                lease.release()
        return "NO_ERROR", "", session.closed

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, rank) for rank in range(4)]
        results = [f.result(timeout=30) for f in futures]
    print("R3_PROMPT_TRANSITION_RESULTS", entry, results)
    assert all(kind == "ValueError" and closed for kind, message, closed in results), (
        results
    )

    assert observed, "The first chunk must execute production CPU DiT forwards"
    assert not any(after_prepare for rank, after_prepare in observed), observed
    assert all(
        session.closed and session.pending_bytes == 0 and not session.pending
        for session in sessions
    )
    assert all(
        session.inputs
        is session.prompt
        is session.applied_prompt
        is session.sampler
        is None
        for session in sessions
    )
    assert all(session._drain_tensors == [] for session in sessions)
    assert all(
        pool.available == pool.size and pool.acquire_count == pool.release_count
        for pool in pools
    )
    assert all(lease.released for lease in leases)
    unique = {id(cache): cache for cache in caches}
    assert len(unique) >= 4
    for cache in unique.values():
        assert cache.positions() is None
        assert cache.active is cache.pending_source is None
        assert cache.eviction_target_block is None
        assert cache._stores == {}


@torch.no_grad()
def test_sp4_distinct_boundary_purposes_cannot_match_host_vote(model, monkeypatch):
    """Two legal operations have equal rows/layout/cache but different purposes."""
    from sglang.multimodal_gen.runtime.models.dits import zing_h3

    packer, inputs, visual, _ = request(model, audio=False)
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (1, 0))
    pairs = []
    for _ in range(4):
        sampler = ZingH3Sampler(
            model, packer, sigma_points=2, window_eviction_timing="after_step0"
        )
        state = sampler.initialize(inputs, visual)
        pairs.append((sampler, state))
    comm = _CPUCollectives(4)
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (4, comm.thread.rank))
    monkeypatch.setattr(
        zing_h3, "get_sp_group", lambda: SimpleNamespace(cpu_group=object())
    )
    payloads = []

    def gather_objects(output, value, group=None):
        payloads.append(value.copy())
        output[:] = comm.exchange(value, lambda values, rank: list(values))

    monkeypatch.setattr(sampling.dist, "all_gather_object", gather_objects)

    def worker(rank):
        comm.thread.rank = rank
        sampler, state = pairs[rank]
        source, target, _, _ = state.spans[0]
        try:
            if rank == 1:
                sampler._precheck_replacement_prefix(
                    state.cache, state.positive, source, target
                )
            else:
                sampler._begin_chunk(state.cache, state.positive, source, target)
        except BaseException as error:
            return type(error).__name__, str(error)
        return "NO_ERROR", ""

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(worker, range(4)))
        print("R3_HOST_PHASE_RESULTS", results)
        assert len(payloads) == 4
        assert len({payload["required_rows"] for payload in payloads}) == 1
        assert all(
            payload["layout_signature"] == payloads[0]["layout_signature"]
            for payload in payloads
        )
        assert all(kind == "ValueError" for kind, message in results), results
        assert all(state.cache.positions() is None for sampler, state in pairs)
        assert all(
            state.cache.eviction_target_block is None for sampler, state in pairs
        )
    finally:
        for sampler, state in pairs:
            state.close()
