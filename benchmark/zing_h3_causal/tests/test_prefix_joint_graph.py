# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811
import copy
from types import SimpleNamespace

import prefix_joint as pj
import pytest
import torch
from prefix_joint_graph import PrefixJointGraphs

from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    ZingH3Packer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_denoise_graph import (
    WarmupRecordingGraphBackend,
    _storage,
    _weights_free_model,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
    model,
    request,
)


def setup_case(model, rope):
    model.arch.ar_condition_timestep = 0.0
    model.arch.reference_visual_timestep = 0.999
    packer = ZingH3Packer(
        model.arch, num_frame_per_block=[3, 2], split_first_latent=True
    )
    sampler = ZingH3Sampler(
        model,  # noqa: F401
        packer,
        solver="few_step",
        denoising_step_list=[1000, 750],
        window_chunks=6,
        sink_chunks=3,
        rope_position_mode=rope,
        window_eviction_timing="after_step0",
        joint_forward="cross_chunk",
        ar_condition_timestep=0.0,
        reference_visual_timestep=0.999,
    )
    sampler.device = torch.device("cpu")
    sampler.video_scheduler.to("cpu")

    def cache(storage=None):
        return ZingH3KVCache(
            model.arch.num_layers,
            window_policy=sampler.window_policy,
            rope_position_mode=rope,
            rope_max_frame_gap=12,
            window_eviction_timing="after_step0",
            storage=storage,
        )

    def packs(refs=True):
        _, inputs, _, _ = request(model, audio=False)
        if refs:
            inputs.visual = {"r": torch.randn(1, 2, 1, 2, 2)}
            inputs.media = [dict(id="r", type="image")]
            inputs.conditions = [dict(content_id="r", role="reference")]
        inputs.visual_shape = (8, 2, 2)
        inputs.source_visual = None
        history = torch.randn(1, 2, 5, 2, 2)
        noise = torch.randn(1, 2, 3, 2, 2)
        origin, _ = sampler.rebuild_prompt_prefix(
            inputs, inputs.positive, history, cache()
        )
        current_inputs = copy.copy(inputs)
        current_inputs.source_visual = torch.randn(1, 2, 3, 2, 2)
        layout = packer.pack_chunk(
            current_inputs,
            inputs.positive,
            noise,
            None,
            chunk=packer.chunk_plan.chunk(3),
            media_origin=origin,
        )
        current = SimpleNamespace(positive=layout, spans=[(7, 8, 0, 3)], visual=noise)
        return pj.packs_for(sampler, inputs, inputs.positive, history, current)

    return sampler, cache, packs


@pytest.mark.parametrize("rope", ["absolute", "block_relative"])
@pytest.mark.parametrize("refs", [False, True])
@torch.no_grad()
def test_weights_free_mixed_prefix_matches_resident_and_replays(
    model, tmp_path, rope, refs
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import video_plan

    sampler, cache, make_packs = setup_case(model, rope)
    schedule = sampler.video_scheduler
    plan = video_plan(
        "few_step", schedule.timesteps, 0.0, 0.999, sigmas=schedule.sigmas
    )
    candidate = _weights_free_model(model, tmp_path, plan)
    graphs = PrefixJointGraphs(backend=WarmupRecordingGraphBackend())
    pooled = cache(_storage(candidate))
    for _ in range(4):
        sampler.generator = model
        packs = make_packs(refs)
        expected_cache = cache()
        expected, _ = pj.joint(sampler, packs, expected_cache)
        sampler.generator = candidate
        eager_cache = cache()
        eager, _ = pj.joint(sampler, packs, eager_cache)
        pooled.clear()
        fixed, _ = graphs.run(sampler, packs, pooled)
        # Same full-group BF16 projections; inspect output and every KV layer.
        for actual, state in ((eager, eager_cache), (fixed, pooled)):
            torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)
            assert state.last_block == expected_cache.last_block
            assert state.active[0] == expected_cache.active[0]
            assert torch.equal(state.positions(), expected_cache.positions())
            for layer in range(model.arch.num_layers):
                for got, want in zip(
                    state.history(layer), expected_cache.history(layer)
                ):
                    torch.testing.assert_close(got, want, atol=0.008, rtol=0.008)
    assert graphs.routes[("prefix_separate", "replay")] == 2
    assert candidate.adaln_table.hits > 0
    assert candidate.adaln_table.stats()["fallback"] == 0


@pytest.mark.parametrize("fixed", [False, True])
@torch.no_grad()
def test_weights_free_missing_full_prefix_batch_clears_cache(model, tmp_path, fixed):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
        batch_key,
        video_plan,
    )

    sampler, cache, make_packs = setup_case(model, "block_relative")
    packs = make_packs()
    schedule = sampler.video_scheduler
    candidate = _weights_free_model(
        model,
        tmp_path,
        video_plan("few_step", schedule.timesteps, 0.0, 0.999, sigmas=schedule.sigmas),
    )
    # Keep both singleton entries: they must not substitute for this full batch.
    del candidate.adaln_table.keys[batch_key(torch.tensor([0.0, 0.999]))]
    sampler.generator = candidate
    state = cache(_storage(candidate))
    with pytest.raises(ValueError, match="complete timestep batch"):
        if fixed:
            PrefixJointGraphs(capture_enabled=False).run(sampler, packs, state)
        else:
            pj.joint(sampler, packs, state)
    assert state.prefix is None and not state.committed
    assert state.pending_source is None and state.active is None


@torch.no_grad()
def test_weights_free_prefix_sp2_matches_same_sp_resident(model, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.models.dits import zing_h3
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import video_plan
    from sglang.multimodal_gen.test.unit.test_zing_h3_sp_cpu import _CPUCollectives

    sampler, cache, make_packs = setup_case(model, "block_relative")
    packs = make_packs()
    schedule = sampler.video_scheduler
    plan = video_plan(
        "few_step", schedule.timesteps, 0.0, 0.999, sigmas=schedule.sigmas
    )
    residents = [
        zing_h3.ZingH3DiTModel(ZingH3DiTConfig(arch_config=model.arch), {}).eval()
        for _ in range(2)
    ]
    for resident in residents:
        resident.load_state_dict(model.state_dict())
    candidates = [
        _weights_free_model(model, tmp_path / str(rank), plan) for rank in range(2)
    ]
    for candidate in residents + candidates:
        candidate.experimental_stable_shapes = True
    comm = _CPUCollectives(2)
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (2, comm.thread.rank))
    monkeypatch.setattr(
        zing_h3,
        "get_sp_group",
        lambda: SimpleNamespace(ulysses_group=None, cpu_group=comm),
    )
    monkeypatch.setattr(zing_h3, "gather_sequence_varlen", comm.gather)
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: 2)
    monkeypatch.setattr(usp, "get_ulysses_parallel_rank", lambda: comm.thread.rank)
    monkeypatch.setattr(usp, "_ipc_ready_group", lambda: None)
    monkeypatch.setattr(usp, "_usp_all_to_all_single_varlen", comm.all_to_all)
    monkeypatch.setattr(torch.distributed, "all_gather_object", comm.gather_objects)

    def gather(output, value, group=None):
        values = comm.exchange(value, lambda values, rank: list(values))
        for target, source in zip(output, values):
            target.copy_(source)

    monkeypatch.setattr(torch.distributed, "all_gather", gather)

    @torch.no_grad()
    def worker(rank):
        comm.thread.rank = rank
        local_sampler = copy.copy(sampler)
        results = []
        for candidate in (residents[rank], candidates[rank]):
            local_sampler.generator = candidate
            state = cache()
            output, _ = pj.joint(local_sampler, packs, state)
            results.append((output, state))
        (expected, reference), (actual, state) = results
        torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)
        assert torch.equal(state.positions(), reference.positions())
        for layer in range(model.arch.num_layers):
            for got, want in zip(state.history(layer), reference.history(layer)):
                torch.testing.assert_close(got, want, atol=0.008, rtol=0.008)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(worker, rank) for rank in range(2)]:
            future.result(timeout=60)


@pytest.mark.parametrize("rope", ["absolute", "block_relative"])
@torch.no_grad()
def test_replay_refreshes_all_six_branches(model, rope):
    sampler, cache, packs = setup_case(model, rope)

    class BeforeCaptureBackend(WarmupRecordingGraphBackend):
        def capture(self, fn, warmed=None):
            assert not graphs.graphs  # Previous generation must already be gone.
            return super().capture(fn, warmed=warmed)

    graphs = PrefixJointGraphs(backend=BeforeCaptureBackend(), max_graphs=1)
    pooled = cache(_storage(model))
    # Same shape, changed actual context/reference/history/source/noise on every
    # run. Recording replay executes only captured ATen operations, no Python.
    for refs in (True, True, True, True, False, False, False):
        current = packs(refs)
        reference = cache()
        expected, _ = pj.joint(sampler, current, reference)
        pooled.clear()
        actual, _ = graphs.run(sampler, current, pooled)
        torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)
        torch.testing.assert_close(
            pooled.positions(), reference.positions(), atol=0, rtol=0
        )
        assert pooled.active[0] == reference.active[0]
        for layer in range(model.arch.num_layers):
            for got, want in zip(pooled.history(layer), reference.history(layer)):
                torch.testing.assert_close(got, want, atol=0.008, rtol=0.008)
        assert all(
            n not in model.__dict__
            for n in ("_forward_body", "_projection", "_project_outputs")
        )
    assert graphs.routes == {
        ("prefix_separate", "warmup"): 2,
        ("prefix_separate", "capture"): 2,
        ("prefix_separate", "replay"): 3,
        ("prefix_separate", "shape_reset"): 1,
    }
    assert len(graphs.graphs) == 1


@torch.no_grad()
def test_failure_preserves_hooks_and_rejects_live_cache(model, monkeypatch):
    sampler, cache, packs = setup_case(model, "block_relative")
    graphs = PrefixJointGraphs(backend=WarmupRecordingGraphBackend())
    pooled = cache(_storage(model))
    current = packs()
    graphs.run(sampler, current, pooled)
    before = pooled.positions().clone()
    with pytest.raises(ValueError, match="cleared logical cache"):
        graphs.run(sampler, current, pooled)
    torch.testing.assert_close(pooled.positions(), before)
    pooled.clear()

    def fail(*args, **kwargs):
        raise RuntimeError("injected forward failure")

    monkeypatch.setattr(model, "_fixed_block_attention", fail)
    with pytest.raises(Exception):
        graphs.run(sampler, current, pooled)
    assert pooled.prefix is None and not pooled.committed
    assert all(
        n not in model.__dict__
        for n in ("_forward_body", "_projection", "_project_outputs")
    )


def test_prefix_capture_pools_are_independent(monkeypatch):
    from prefix_joint_graph import PrefixGraphBackend

    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        CudaGraphBackend,
    )

    def capture(self, fn, warmed=None):
        assert self._pool is None
        self._pool = object()
        return self._pool, fn()

    monkeypatch.setattr(CudaGraphBackend, "capture", capture)
    backend = PrefixGraphBackend()
    captures = [backend.capture(lambda: ()) for _ in range(3)]
    assert len({id(pool) for pool, _ in captures}) == 3


@pytest.mark.parametrize("splits", [(2, 2), (2, 1)])
def test_graph_output_gather_uses_ulysses_group(splits, monkeypatch):
    from sglang.multimodal_gen.runtime.models.dits import zing_h3

    group = object()
    monkeypatch.setattr(
        zing_h3, "get_sp_group", lambda: SimpleNamespace(ulysses_group=group)
    )

    def gather(outputs, inputs, *, group):
        assert group is expected_group
        outputs[0].copy_(inputs)
        outputs[1].copy_(inputs + 10)

    expected_group = group
    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    seen = []

    def video_out(hidden):
        seen.append(hidden.clone())
        return (hidden,)

    m = SimpleNamespace(
        experimental_stable_shapes=True,
        final_layer=SimpleNamespace(video_out=video_out, audio_out=lambda h: (h,)),
        arch=SimpleNamespace(latents_dim=2),
    )
    layout = SimpleNamespace(
        sequence_splits=splits,
        vrows=torch.arange(sum(splits)),
        arows=torch.empty(0, dtype=torch.long),
        local_vrows=None,
        local_arows=None,
    )
    local = torch.arange(4.0).reshape(2, 2)
    zing_h3.ZingH3DiTModel._project_outputs(
        m, local, 1, layout, joint_ends=(sum(splits),), mixed_prefix=True
    )
    torch.testing.assert_close(seen[0], torch.cat([local, (local + 10)[: splits[1]]]))
