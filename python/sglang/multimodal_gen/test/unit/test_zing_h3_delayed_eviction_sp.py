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
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy
from sglang.multimodal_gen.test.unit.test_zing_h3_joint import (
    _joint_cache_snapshot,
    four_head_model,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (
    cpu_execution_options,
    model,
    request,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_sp_cpu import _CPUCollectives


@pytest.mark.parametrize("fault", ["policy", "target", "last_block", "eviction_target"])
def test_sp4_boundary_rejects_logical_disagreement_before_model(
    four_head_model, monkeypatch, fault
):
    from sglang.multimodal_gen.runtime.models.dits import zing_h3

    comm = _CPUCollectives(4)
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (4, comm.thread.rank))
    monkeypatch.setattr(
        zing_h3, "get_sp_group", lambda: SimpleNamespace(cpu_group=object())
    )

    def gather_objects(output, value, group=None):
        output[:] = comm.exchange(value, lambda values, rank: list(values))

    monkeypatch.setattr(sampling.dist, "all_gather_object", gather_objects)
    # Missing coordinator is a behavioral contract failure before implementing
    # rank agreement, rather than a module-import failure.
    assert callable(getattr(ZingH3Sampler, "_agree_sp_boundary", None)), (
        "Delayed eviction lacks host rank consensus"
    )
    packer, _, _, _ = request(four_head_model, audio=False)

    def worker(rank):
        comm.thread.rank = rank
        cache = ZingH3KVCache(
            len(four_head_model.blocks),
            window_policy=ZingWindowPolicy(window_chunks=3),
            window_eviction_timing="after_step0",
        )
        sampler = ZingH3Sampler(
            four_head_model, packer, window_eviction_timing="after_step0"
        )
        target = 2
        if rank == 1:
            if fault == "policy":
                cache._window_eviction_timing = "before_source"
            elif fault == "target":
                target = 4
            elif fault == "last_block":
                cache.last_block = 2
            else:
                cache.eviction_target_block = 4
        try:
            sampler._agree_sp_boundary(cache, target)
        except ValueError as error:
            assert str(error)
            assert cache.positions() is None
            assert cache.active is cache.pending_source is None
            assert cache.eviction_target_block is None
            return True
        pytest.fail("Every rank must reject logical disagreement")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, rank) for rank in range(4)]
        assert all(f.result(timeout=30) for f in futures)


@pytest.mark.parametrize("mode", ["source_first", "cross_chunk"])
@pytest.mark.parametrize("world", [4])
@pytest.mark.parametrize("source_hw", [(2, 2), (2, 4)])
@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("rope_mode", ["absolute", "block_relative"])
def test_after_step0_sp4_matches_independent_sequential(
    four_head_model, monkeypatch, mode, world, source_hw, pooled, rope_mode
):
    """SP4 must fuse real branches, including split boundaries and empty local heads."""
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.models.dits import zing_h3
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        allocate_layer_stores,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_sp_cpu import _CPUCollectives

    model = four_head_model
    packer, inputs, _, _ = request(model, audio=False)
    packer = type(packer)(model.arch, num_frame_per_block=[3, 2])
    inputs.visual_shape = (15, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 15, *source_hw)
    visual = torch.randn(1, 2, 15, 2, 2)
    comm = _CPUCollectives(world)
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (world, comm.thread.rank))
    monkeypatch.setattr(
        zing_h3,
        "get_sp_group",
        lambda: SimpleNamespace(ulysses_group=None, cpu_group=object()),
    )
    monkeypatch.setattr(zing_h3, "gather_sequence_varlen", comm.gather)
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: world)
    monkeypatch.setattr(usp, "get_ulysses_parallel_rank", lambda: comm.thread.rank)
    monkeypatch.setattr(usp, "_ipc_ready_group", lambda: None)
    monkeypatch.setattr(usp, "_usp_all_to_all_single_varlen", comm.all_to_all)

    def gather_objects(output, value, group=None):
        output[:] = comm.exchange(value, lambda values, rank: list(values))

    monkeypatch.setattr(sampling.dist, "all_gather_object", gather_objects)
    original_update = ZingH3KVCache.update
    original_forward = ZingH3Sampler._forward
    original_project = zing_h3.ZingH3DiTModel._project_outputs
    scheduler_type = type(ZingH3Sampler(model, packer).video_scheduler)
    original_scheduler_step = scheduler_type.step

    def scheduler_step(self, prediction, sample, *args, **kwargs):
        result = original_scheduler_step(self, prediction, sample, *args, **kwargs)
        if kwargs.get("step_index") == 0:
            comm.thread.step0_latents.append(result[0].clone())
        return result

    monkeypatch.setattr(scheduler_type, "step", scheduler_step)
    # Real SP ranks own independent models, including mutable derived buffers.
    rank_models = [
        zing_h3.ZingH3DiTModel(ZingH3DiTConfig(arch_config=model.arch), {}).eval()
        for _ in range(world)
    ]
    for rank_model in rank_models:
        rank_model.load_state_dict(model.state_dict())

    def update(cache, positions, blocks, phase, frames, layers=None, validation=None):
        original_update(cache, positions, blocks, phase, frames, layers, validation)
        assert cache.raw_store
        comm.thread.snapshots.append(
            _joint_cache_snapshot(cache, phase, blocks, positions, layers)
        )

    def forward(sampler, packed, cache, update):
        result = original_forward(sampler, packed, cache, update)
        if update in ("active", "joint"):
            rows = int((packed["block_idx"] == packed["block_idx"][-1]).sum())
            comm.thread.targets.append(result[0][-rows:].clone())
        if update == "joint":
            comm.thread.phases.append(packed["_zing_joint_phases"])
        return result

    def project(self, hidden, p, layout, *, joint_ends=None):
        if joint_ends is not None:
            comm.thread.layouts.append(
                (joint_ends, layout.row_start, hidden.shape[0], layout.sequence_splits)
            )
        return original_project(self, hidden, p, layout, joint_ends=joint_ends)

    monkeypatch.setattr(ZingH3KVCache, "update", update)
    monkeypatch.setattr(ZingH3Sampler, "_forward", forward)
    monkeypatch.setattr(zing_h3.ZingH3DiTModel, "_project_outputs", project)

    @torch.no_grad()
    def worker(rank, selected):
        comm.thread.rank = rank
        comm.thread.snapshots, comm.thread.targets = [], []
        comm.thread.phases, comm.thread.layouts = [], []
        comm.thread.step0_latents = []
        sampler = ZingH3Sampler(
            rank_models[rank],
            packer,
            sigma_points=3,
            joint_forward=selected,
            window_chunks=3,
            sink_chunks=1,
            rope_position_mode=rope_mode,
            window_eviction_timing="after_step0",
        )
        storage = (
            allocate_layer_stores(
                128,
                num_layers=len(model.blocks),
                local_heads=model.arch.num_attention_heads // world,
                head_dim=model.arch.attention_head_dim,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )
            if pooled
            else None
        )
        state = sampler.initialize(inputs, visual.clone(), storage=storage)
        if selected == "off":

            def forbidden_joint(*args, **kwargs):
                raise AssertionError("Independent sequential reference called joint")

            sampler._predict_joint = forbidden_joint
            sampler._ready_cross_prediction = forbidden_joint
        try:
            if selected == "cross_chunk":
                sampler._sample_cross_chunk(state)
            else:
                while state.index < len(state.spans):
                    sampler.step(state)
            assert state.cache.last_block == 12
            assert state.cache.active is state.cache.pending_source is None
            assert state.cache.stats["compactions"] >= 3
            if pooled:
                assert state.cache.stats["store_allocations"] == 0
                assert state.cache.stats["rotated_allocations"] == 0
            return (
                state.visual.clone(),
                list(comm.thread.snapshots),
                list(comm.thread.targets),
                dict(sampler.last_run_metrics),
                list(comm.thread.phases),
                list(comm.thread.layouts),
                list(comm.thread.step0_latents),
            )
        finally:
            state.close()
            assert state.cache.positions() is None
            assert state.cache.active is state.cache.pending_source is None

    runs = []
    for selected in ("off", mode):
        with ThreadPoolExecutor(max_workers=world) as executor:
            futures = [executor.submit(worker, rank, selected) for rank in range(world)]
            runs.append([future.result(timeout=90) for future in futures])
    for expected, actual in zip(*runs):
        torch.testing.assert_close(actual[0], expected[0], atol=0.008, rtol=0.008)
        assert len(actual[1]) == len(expected[1])
        for (am, at), (em, et) in zip(actual[1], expected[1]):
            assert am == em
            assert len(at) == len(et)
            for value, reference in zip(at, et):
                torch.testing.assert_close(value, reference, atol=0.008, rtol=0.008)
        assert len(actual[2]) == len(expected[2])
        for value, reference in zip(actual[2], expected[2]):
            torch.testing.assert_close(value, reference, atol=0.008, rtol=0.008)
        assert len(actual[6]) == len(expected[6]) > 3
        for value, reference in zip(actual[6], expected[6]):
            torch.testing.assert_close(value, reference, atol=0.008, rtol=0.008)
        assert actual[3]["joint_fallbacks"].get("window_eviction", 0) == 0
        counts = actual[3]["forward_counts"]
        assert counts[mode] > 0
        assert actual[3]["dit_forwards"] < expected[3]["dit_forwards"]
        phases = (
            ("source", "active")
            if mode == "source_first"
            else ("final", "source", "active")
        )
        assert phases in actual[4]
        assert actual[4].count(phases) == counts[mode]
        assert "sp_size" not in actual[3]["joint_fallbacks"]
    if world == 4:
        layouts = [layout for result in runs[1] for layout in result[5]]
        assert any(
            start < stop < start + rows
            for ends, start, rows, _ in layouts
            for stop in ends[:-1]
        )
        assert any(
            stop <= start or begin >= start + rows
            for ends, start, rows, _ in layouts
            for begin, stop in zip((0, *ends[:-1]), ends)
        )
        assert any(len(set(splits)) > 1 for _, _, _, splits in layouts)


@pytest.mark.parametrize("sequential_fits", [False, True])
def test_sp4_capacity_vote_is_global_and_preserves_policy(
    four_head_model, monkeypatch, sequential_fits
):
    from sglang.multimodal_gen.runtime.models.dits import zing_h3
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        allocate_layer_stores,
    )

    comm = _CPUCollectives(4)
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (4, comm.thread.rank))
    monkeypatch.setattr(
        zing_h3, "get_sp_group", lambda: SimpleNamespace(cpu_group=object())
    )

    def gather_objects(output, value, group=None):
        output[:] = comm.exchange(value, lambda values, rank: list(values))

    monkeypatch.setattr(sampling.dist, "all_gather_object", gather_objects)
    assert callable(getattr(ZingH3Sampler, "_agree_sp_boundary", None)), (
        "Delayed eviction lacks host rank consensus"
    )
    packer, _, _, _ = request(four_head_model, audio=False)

    def worker(rank):
        comm.thread.rank = rank
        storage = allocate_layer_stores(
            8 if rank == 1 else 32,
            num_layers=len(four_head_model.blocks),
            local_heads=1,
            head_dim=8,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        cache = ZingH3KVCache(
            len(four_head_model.blocks),
            window_policy=ZingWindowPolicy(window_chunks=3),
            window_eviction_timing="after_step0",
            storage=storage,
        )
        sampler = ZingH3Sampler(
            four_head_model, packer, window_eviction_timing="after_step0"
        )
        try:
            # Eligibility differs only because one rank has insufficient joint
            # scratch. Every rank still fits the same-policy sequential peak.
            route = sampler._agree_sp_boundary(
                cache,
                2,
                cross=True,
                joint_eligible=cache._length + 9 <= cache.capacity,
                required_rows=8 if sequential_fits else 9,
            )
        except ValueError:
            assert not sequential_fits
            assert cache.positions() is None
            assert cache.eviction_target_block is None
            return "rejected"
        assert sequential_fits
        assert route is False
        assert cache.window_eviction_timing == "after_step0"
        cache.begin_chunk(2)
        assert cache.eviction_target_block == 2
        cache.clear()
        return "sequential"

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, rank) for rank in range(4)]
        assert [f.result(timeout=30) for f in futures] == [
            "sequential" if sequential_fits else "rejected"
        ] * 4


def test_sp4_initialize_rejects_sampler_policy_mismatch_before_model(
    four_head_model, monkeypatch
):
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.models.dits import zing_h3

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

    def forward(*args, **kwargs):
        observed.append(comm.thread.rank)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(ZingH3Sampler, "_forward", forward)
    packer, inputs, visual, _ = request(four_head_model, audio=False)

    def worker(rank):
        comm.thread.rank = rank
        sampler = ZingH3Sampler(
            rank_models[rank],
            packer,
            window_eviction_timing="before_source" if rank == 1 else "after_step0",
        )
        with pytest.raises(ValueError, match="(?i)(policy|timing|rank|mismatch)"):
            sampler.initialize(inputs, visual.clone())
        return True

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, rank) for rank in range(4)]
        assert all(f.result(timeout=30) for f in futures)
    assert observed == [], (
        "Policy consensus must reject before prefix/model collectives"
    )


def test_sp4_one_rank_rope_unsupported_uses_global_sequential_route(
    four_head_model, monkeypatch
):
    """A real transaction rejects rank-one coordinates; no joint rank enters A2A."""
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.models.dits import zing_h3

    comm = _CPUCollectives(4)
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (4, comm.thread.rank))
    monkeypatch.setattr(
        zing_h3,
        "get_sp_group",
        lambda: SimpleNamespace(cpu_group=object(), ulysses_group=None),
    )
    monkeypatch.setattr(zing_h3, "gather_sequence_varlen", comm.gather)
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: 4)
    monkeypatch.setattr(usp, "get_ulysses_parallel_rank", lambda: comm.thread.rank)
    monkeypatch.setattr(usp, "_ipc_ready_group", lambda: None)
    monkeypatch.setattr(usp, "_usp_all_to_all_single_varlen", comm.all_to_all)

    def gather_objects(output, value, group=None):
        output[:] = comm.exchange(value, lambda values, rank: list(values))

    monkeypatch.setattr(sampling.dist, "all_gather_object", gather_objects)
    rank_models = [
        zing_h3.ZingH3DiTModel(
            ZingH3DiTConfig(arch_config=four_head_model.arch), {}
        ).eval()
        for _ in range(4)
    ]
    for rank_model in rank_models:
        rank_model.load_state_dict(four_head_model.state_dict())
    packer, inputs, visual, _ = request(four_head_model, audio=False)
    original_forward = ZingH3Sampler._forward

    def forward(sampler, packed, cache, update):
        comm.thread.phases.append(update)
        return original_forward(sampler, packed, cache, update)

    monkeypatch.setattr(ZingH3Sampler, "_forward", forward)

    @torch.no_grad()
    def worker(rank):
        comm.thread.rank = rank
        comm.thread.phases = []
        sampler = ZingH3Sampler(
            rank_models[rank],
            packer,
            sigma_points=3,
            joint_forward="source_first",
            window_chunks=3,
            sink_chunks=1,
            window_eviction_timing="after_step0",
        )
        state = sampler.initialize(inputs, visual.clone())
        try:
            if rank == 1:
                positions = state.positive["position_ids"]
                positions[state.positive["block_idx"] == 1, 0] += 100
            sampler.step(state)
            assert "joint" not in comm.thread.phases
            assert sampler.last_run_metrics["forward_counts"]["source_first"] == 0
            assert state.cache.last_block == 2
            assert state.cache.active is state.cache.pending_source is None
            assert state.cache.eviction_target_block is None
            assert state.cache.window_eviction_timing == "after_step0"
            assert torch.isfinite(state.visual).all()
            return True
        finally:
            state.close()
            assert state.cache.positions() is None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, rank) for rank in range(4)]
        assert all(f.result(timeout=30) for f in futures)


@pytest.mark.parametrize(
    "disagreement",
    [
        "guidance",
        "solver",
        "same_length_grid",
        "source_presence",
        "source_geometry",
        "target_geometry",
        "row_roles",
        "target_audio_presence",
    ],
)
def test_sp4_initialize_rejects_configuration_or_geometry_mismatch(
    four_head_model, monkeypatch, disagreement
):
    from copy import deepcopy

    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.models.dits import zing_h3

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

    def forward(*args, **kwargs):
        observed.append(comm.thread.rank)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(ZingH3Sampler, "_forward", forward)
    packer, inputs, visual, sound = request(four_head_model, audio=True)

    def worker(rank):
        comm.thread.rank = rank
        options = dict(window_eviction_timing="after_step0", sigma_points=3)
        local_inputs = deepcopy(inputs)
        local_visual = visual.clone()
        local_audio = None
        if disagreement == "same_length_grid":
            options.update(solver="few_step", denoising_step_list=[1000, 500, 100])
        if rank == 1:
            if disagreement == "guidance":
                options["guidance_scale"] = 2.0
            elif disagreement == "solver":
                options.update(solver="few_step", denoising_step_list=[1000, 500, 100])
            elif disagreement == "same_length_grid":
                options["denoising_step_list"] = [1000, 600, 100]
            elif disagreement == "source_presence":
                local_inputs.source_visual = None
            elif disagreement == "source_geometry":
                local_inputs.source_visual = torch.randn(1, 2, 3, 2, 4)
            elif disagreement == "target_geometry":
                local_inputs.visual_shape = (3, 2, 4)
                local_visual = torch.randn(1, 2, 3, 2, 4)
            elif disagreement == "row_roles":
                local_inputs.positive["row_role"][0] = 1
            elif disagreement == "target_audio_presence":
                local_audio = sound.clone()
        sampler = ZingH3Sampler(rank_models[rank], packer, **options)
        with pytest.raises(ValueError, match="(?i)(policy|timing|rank|mismatch)"):
            sampler.initialize(local_inputs, local_visual, local_audio)
        return True

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, rank) for rank in range(4)]
        assert all(f.result(timeout=30) for f in futures)
    assert observed == [], (
        "Configuration/geometry consensus must reject before prefix/model collectives"
    )


def test_sp4_initialize_rejects_one_rank_prefix_capacity_before_model(
    four_head_model, monkeypatch
):
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.models.dits import zing_h3
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        allocate_layer_stores,
    )

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

    def forward(*args, **kwargs):
        observed.append(comm.thread.rank)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(ZingH3Sampler, "_forward", forward)
    packer, inputs, visual, _ = request(four_head_model, audio=False)
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
            rank_models[rank], packer, window_eviction_timing="after_step0"
        )
        storage = allocate_layer_stores(
            prefix_rows - 1 if rank == 1 else 128,
            num_layers=len(four_head_model.blocks),
            local_heads=1,
            head_dim=8,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        with pytest.raises(ValueError, match="(?i)(capacity|rank|mismatch)"):
            sampler.initialize(inputs, visual.clone(), storage=storage)
        return True

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, rank) for rank in range(4)]
        assert all(f.result(timeout=30) for f in futures)
    assert observed == [], (
        "Capacity consensus must reject before prefix/model collectives"
    )

    unique = {id(cache): cache for cache in caches}
    assert len(unique) == 4
    for cache in unique.values():
        assert cache.positions() is None
        assert cache.active is cache.pending_source is None
        assert cache.eviction_target_block is None
        assert cache._stores == {}


def test_sp4_ready_rejects_one_rank_prefix_slot_capacity_before_model(
    four_head_model, monkeypatch
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

    def forward(*args, **kwargs):
        observed.append(comm.thread.rank)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(ZingH3Sampler, "_forward", forward)
    packer, inputs, visual, _ = request(four_head_model, audio=False)
    packer.num_frame_per_block = [3, 2]
    packer.split_first_latent = True
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
            rank_models[rank], packer, window_eviction_timing="after_step0"
        )
        storage = allocate_layer_stores(
            prefix_rows - 1 if rank == 1 else 128,
            num_layers=len(four_head_model.blocks),
            local_heads=1,
            head_dim=8,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        session = SourceReadySession(
            sampler,
            inputs,
            seed=51,
            max_pending_chunks=4,
            max_pending_bytes=1000000,
            total_latent_frames=3,
            acquire_storage=lambda *args: storage,
        )
        chunk = packer.chunk_plan.chunk(0, 3)
        session.append(
            inputs.source_visual[:, :, chunk.start : chunk.end],
            visual[:, :, chunk.start : chunk.end],
            provenance=fixture_provenance(
                chunk.chunk_id + 1, 20, chunk.start, chunk.end
            ),
        )
        with pytest.raises(ValueError, match="(?i)(capacity|rank|mismatch)"):
            session.step()
        assert session.closed
        assert all(value == 0 for value in session.memory_snapshot().values())
        return True

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, rank) for rank in range(4)]
        assert all(f.result(timeout=30) for f in futures)
    assert observed == [], (
        "Capacity consensus must reject before prefix/model collectives"
    )

    unique = {id(cache): cache for cache in caches}
    assert len(unique) >= 4
    for cache in unique.values():
        assert cache.positions() is None
        assert cache.active is cache.pending_source is None
        assert cache.eviction_target_block is None
        assert cache._stores == {}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
