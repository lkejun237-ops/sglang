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


@pytest.mark.parametrize(
    "case,joint",
    [
        ("cold", "cross_chunk"),
        ("warm", "cross_chunk"),
        ("empty_step", "cross_chunk"),
        ("empty_step", "off"),
        ("empty_step", "source_first"),
        ("empty_drain", "cross_chunk"),
        ("empty_drain", "off"),
        ("empty_drain", "source_first"),
        ("mixed_entry", "cross_chunk"),
        ("unequal_drain", "off"),
        ("unequal_drain", "source_first"),
    ],
)
def test_sp4_ready_route_divergence_rejects_and_cleans(
    four_head_model, monkeypatch, case, joint
):
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
    pools, leases, sessions = [], [], []

    def forward(self, packed, cache, update):
        rank = comm.thread.rank
        phases = tuple(packed.get("_zing_joint_phases", (update,)))
        observed.append((rank, phases, tuple(packed["block_idx"].unique().tolist())))
        return original_forward(self, packed, cache, update)

    monkeypatch.setattr(ZingH3Sampler, "_forward", forward)
    packer, inputs, visual, _ = request(four_head_model, audio=False)
    packer.num_frame_per_block = [3, 2]
    packer.split_first_latent = True
    if case == "warm":
        inputs.visual_shape = (5, 2, 2)
        inputs.source_visual = torch.cat(
            (inputs.source_visual, inputs.source_visual[:, :, :2]), dim=2
        )
        visual = torch.cat((visual, visual[:, :, :2]), dim=2)
    total = inputs.visual_shape[0]
    sample_entries = []
    warm_predictions = []
    original_cross = ZingH3Sampler._ready_cross_prediction

    def cross(self, *args):
        prediction = original_cross(self, *args)
        if prediction is not None:
            warm_predictions.append(comm.thread.rank)
        return prediction

    monkeypatch.setattr(ZingH3Sampler, "_ready_cross_prediction", cross)
    original_sample = ZingH3Sampler._sample_block

    def sample(self, *args, **kwargs):
        sample_entries.append(
            (comm.thread.rank, args[8], kwargs.get("first_prediction") is not None)
        )
        return original_sample(self, *args, **kwargs)

    monkeypatch.setattr(ZingH3Sampler, "_sample_block", sample)
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, None)
    prefix_rows = int((layout["block_idx"] == 0).sum())
    assert prefix_rows > 1
    caches = []
    created_caches = []
    original_init = ZingH3KVCache.__init__

    def initialize(cache, *args, **kwargs):
        original_init(cache, *args, **kwargs)
        created_caches.append(cache)

    monkeypatch.setattr(ZingH3KVCache, "__init__", initialize)
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
            joint_forward=joint,
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
            total_latent_frames=total,
            acquire_storage=acquire,
        )
        sessions.append(session)
        count = 2 if rank != 1 else 1
        if case == "warm":
            count += 1
        elif case in ("empty_step", "empty_drain"):
            count = 0 if rank == 1 else 1
        elif case == "mixed_entry":
            count = 2
        for chunk_id in range(count):
            chunk = packer.chunk_plan.chunk(chunk_id, total)
            session.append(
                inputs.source_visual[:, :, chunk.start : chunk.end],
                visual[:, :, chunk.start : chunk.end],
                provenance=fixture_provenance(
                    chunk.chunk_id + 1, 20, chunk.start, chunk.end
                ),
            )
        try:
            if case == "empty_step" or (case == "mixed_entry" and rank == 1):
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
    print("READY_ROUTE_RESULTS", case, joint, results)
    # All ranks must reject through one host decision, never an A2A/host
    # payload collision or a peer barrier timeout. finally cleanup is checked
    # separately: disposal by the owner must not mask an open failed session.
    assert all(kind == "ValueError" and closed for kind, message, closed in results), (
        results
    )
    assert len({message for kind, message, closed in results}) == 1, results
    assert all("mismatch" in message for kind, message, closed in results), results

    print("READY_ROUTE_MODEL_ENTRIES", observed)
    # Prefix/source/step0 work shared before the cold/warm disagreement is
    # permitted. A final on the divergent chunk (or any later model) is not.
    if case in ("cold", "warm", "mixed_entry"):
        divergent_target = packer.chunk_plan.chunk(
            1 if case == "warm" else 0, total
        ).target_block
        assert not any(
            phase == "final" and block == divergent_target
            for rank, phases, blocks in observed
            for phase, block in zip(phases, blocks, strict=True)
        ), observed
        assert not any(
            max(blocks, default=0) > divergent_target
            for rank, phases, blocks in observed
        ), observed
    elif case in ("empty_step", "empty_drain"):
        assert not observed, observed
    else:
        # off/source_first may finish the common first chunk, but cannot enter
        # a model for the next chunk after one peer ran out of ready data.
        first_target = packer.chunk_plan.chunk(0, total).target_block
        assert not any(
            max(blocks, default=0) > first_target for rank, phases, blocks in observed
        ), observed
    if case == "warm":
        assert sorted(warm_predictions) == list(range(4)), warm_predictions
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
    # Refusal before allocation needs no artificial cache. Every cache that
    # was actually initialized must have been cleared.
    unique = {id(cache): cache for cache in created_caches}
    assert set(unique) <= {id(cache) for cache in caches}
    for cache in unique.values():
        assert cache.positions() is None
        assert cache.active is cache.pending_source is None
        assert cache.eviction_target_block is None
        assert cache._stores == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
