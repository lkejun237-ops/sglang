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


@pytest.mark.parametrize("entry", ["step", "drain", "lookahead"])
def test_sp4_ready_prepare_failure_is_agreed(four_head_model, monkeypatch, entry):
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
    monkeypatch.setattr(zing_h3, "get_tp_world_size", lambda: 1)
    monkeypatch.setattr(
        zing_h3,
        "get_sp_group",
        lambda: SimpleNamespace(
            cpu_group=object(), ulysses_group=None, ring_world_size=1
        ),
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
    packer.num_frame_per_block = [3, 2]
    packer.split_first_latent = True
    original_pack_chunk = packer.pack_chunk

    def pack_chunk(inputs, prompt, visual, audio, **kwargs):
        failing_chunk = 1 if entry == "lookahead" else 0
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
        assert session.cache is None
        condition = {k: v.clone() for k, v in inputs.positive.items()}
        if rank == 1 and entry != "lookahead":
            condition["row_role"][0] = 4
        # Inject malformed initial conditioning at the prepare boundary.
        session.prompt = session._snapshot(condition, "changed")
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
    print("R2_READY_PREPARE_RESULTS", entry, results)
    assert all(kind == "ValueError" and closed for kind, message, closed in results), (
        results
    )
    assert pack_errors == [
        (
            1,
            "H3 context row_role must align with context and contain only condition roles",
        )
    ]
    assert not any(after_prepare for rank, after_prepare in observed), (
        "Prepare consensus must reject before any subsequent model collective",
        observed,
    )
    if entry == "lookahead":
        assert observed, "The valid first chunk must execute real DiT forwards"
    else:
        assert observed == []
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
    # Initial failure may reject before any cache is allocated.
    if entry == "lookahead":
        assert len(unique) >= 4
    for cache in unique.values():
        assert cache.positions() is None
        assert cache.active is cache.pending_source is None
        assert cache.eviction_target_block is None
        assert cache._stores == {}


@pytest.mark.parametrize("failure", ["plan", "pack_runtime"])
def test_sp4_ready_pure_prepare_errors_are_agreed(
    four_head_model, monkeypatch, failure
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
    monkeypatch.setattr(zing_h3, "get_tp_world_size", lambda: 1)
    monkeypatch.setattr(
        zing_h3,
        "get_sp_group",
        lambda: SimpleNamespace(
            cpu_group=object(), ulysses_group=None, ring_world_size=1
        ),
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
    packer.num_frame_per_block = [3, 2]
    packer.split_first_latent = True
    original_pack_chunk = packer.pack_chunk

    def pack_chunk(inputs, prompt, visual, audio, **kwargs):
        failing_chunk = 1 if entry == "lookahead" else 0
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
        packer, inputs, visual, _ = request(four_head_model, audio=False)
        packer.num_frame_per_block = [3, 2]
        packer.split_first_latent = True
        original_pack = packer.pack_chunk

        def observe_pack(*args, **kwargs):
            preparing_failure[rank] = True
            try:
                return original_pack(*args, **kwargs)
            except (ValueError, RuntimeError) as error:
                pack_errors.append((rank, str(error)))
                raise

        packer.pack_chunk = observe_pack
        sampler = ZingH3Sampler(
            rank_models[rank],
            packer,
            window_eviction_timing="after_step0",
            sigma_points=2,
            joint_forward="cross_chunk",
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
        assert session.cache is None
        condition = {k: v.clone() for k, v in inputs.positive.items()}
        if rank == 1:
            if failure == "plan":
                packer.num_frame_per_block = 3
            else:
                # Real tensor-device validation fails inside production packing,
                # before any model or CUDA operation is involved.
                condition["row_role"] = condition["row_role"].to("meta")
        # Inject malformed initial conditioning at the prepare boundary.
        session.prompt = session._snapshot(condition, "changed")
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
    print("R2_READY_PREPARE_RESULTS", entry, results)
    assert all(kind == "ValueError" and closed for kind, message, closed in results), (
        results
    )
    if failure == "pack_runtime":
        assert pack_errors == [(1, "Tensor.item() cannot be called on meta tensors")]
    else:
        assert pack_errors == []
    assert not any(after_prepare for rank, after_prepare in observed), (
        "Prepare consensus must reject before any subsequent model collective",
        observed,
    )
    if entry == "lookahead":
        assert observed, "The valid first chunk must execute real DiT forwards"
    else:
        assert observed == []
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
    # Initial failure may reject before any cache is allocated.
    if entry == "lookahead":
        assert len(unique) >= 4
    for cache in unique.values():
        assert cache.positions() is None
        assert cache.active is cache.pending_source is None
        assert cache.eviction_target_block is None
        assert cache._stores == {}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
