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
