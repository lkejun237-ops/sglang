# SPDX-License-Identifier: Apache-2.0
"""Startup replay uses real request preparation, posterior sampling and DiT."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context_or_none,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
    ZingH3Codecs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_warmup import (
    warmup_dit,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_worker import (
    create_link_session,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    ZingH3Packer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
    ZingH3GenerationStage,
)
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
    ZingKVGeometry,
    ZingKVPool,
    ZingKVPoolConfig,
    ZingKVPoolPlan,
    peak_rows,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model as model_fixture

model = model_fixture


@pytest.fixture
def codecs():
    codec = object.__new__(ZingH3Codecs)
    codec.modules = {}
    codec.runtime = {"vae_config": {"posterior_seed": 0, "encode_mode": "performance"}}
    codec.conditioning_mode = "external"
    codec.spatial_ratio = 16
    codec.patch_size = (1, 2, 2)
    codec.temporal_ratio = 4
    codec.vae_frames_per_chunk = 17
    codec.vae_latents_per_chunk = 5
    codec.target_fps = 24
    codec.max_short_edge = 32
    codec.canvas_multiple = 32
    codec.max_area_config = dict(
        reference_image=1024, reference_video=1024, source=1024
    )
    codec.max_source_target_aspect_ratio_factor = 2.0
    codec.audio_sample_rate = 32000
    codec.va = SimpleNamespace(
        latent_channels=2, latents_mean=(0.0, 0.0), latents_std=(1.0, 1.0)
    )
    codec.pipeline_config = SimpleNamespace(zing_reference_max_images=0)
    return codec


def session_factory(model, codecs, joint_forward, *, pooled=False, output=None):
    packer = ZingH3Packer(
        model.arch, num_frame_per_block=[3, 2], split_first_latent=False
    )

    def sampler():
        return ZingH3Sampler(
            model,
            packer,
            solver="few_step",
            denoising_step_list=[1000, 500],
            window_chunks=21,
            sink_chunks=3,
            joint_forward=joint_forward,
        )

    stage = None
    if pooled:
        sample = sampler()
        config = ZingKVPoolConfig(
            max_target_area=1024,
            max_target_long_edge=32,
            window_chunks=sample.window_policy.window_chunks,
            sink_chunks=sample.window_policy.sink_chunks,
            window_eviction_timing=sample.window_eviction_timing,
        )
        geometry = ZingKVGeometry(
            codecs.spatial_ratio,
            tuple(model.arch.patch_size),
            codecs.max_short_edge,
            packer.chunk_plan,
            video_t_span=packer._video_t_span,
            time_span_period=len(packer.visual_token_frame_spans),
        )
        stage = object.__new__(ZingH3GenerationStage)
        stage.codecs, stage.packer = codecs, packer
        plan = ZingKVPoolPlan.build(
            config,
            geometry,
            token_capacity=peak_rows(
                geometry,
                config.window_policy,
                prefix_rows=64,
                rows_per_frame=geometry.rows_per_frame(32, 32),
                has_source=True,
                window_eviction_timing=config.window_eviction_timing,
            ),
        )
        stage.kv_pool = ZingKVPool.allocate(
            plan,
            num_layers=model.arch.num_layers,
            local_heads=model.arch.num_attention_heads,
            head_dim=model.arch.attention_head_dim,
            dtype=torch.bfloat16,
            device=next(model.parameters()).device,
        )
        codecs.compute_plan = plan

    workers, outputs = [], []

    def create(init, deliver):
        async def collect(result):
            outputs.append(result.visual.detach().clone())
            if output is not None:
                output(result)
            await deliver(result)

        worker = create_link_session(sampler(), codecs, init, collect, stage=stage)
        workers.append(worker)
        return worker

    return create, workers, outputs, stage


@pytest.mark.parametrize("joint_forward", ["off", "cross_chunk"])
@pytest.mark.parametrize("pooled", [False, True])
def test_single_latent_replay_completes_and_releases_for_next_request(
    model, codecs, joint_forward, pooled
):
    create, workers, outputs, stage = session_factory(
        model, codecs, joint_forward, pooled=pooled
    )
    random_state = torch.random.get_rng_state().clone()
    context = get_forward_context_or_none()
    for count in (1, 2):
        summary = warmup_dit(model, codecs, create, height=32, width=32)
        assert summary["role"] == "dit"
        assert summary["warmed_up"] is True
        assert summary["elapsed_s"] >= 0.0
        assert (
            summary["width"] == 32
            and summary["height"] == 32
            and summary["latent_frames"] == 1
        )
        assert len(workers) == len(outputs) == count
        assert workers[-1].closed
        assert workers[-1].state.closed
        assert workers[-1].state.inputs is None
        assert not workers[-1].bridge.buffered_latents
        assert outputs[-1].shape == (1, 2, 1, 2, 2)
        assert torch.isfinite(outputs[-1]).all()
        if stage is not None:
            assert stage.kv_pool.available == 1
        assert torch.equal(random_state, torch.random.get_rng_state())
        assert get_forward_context_or_none() is context
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["raise", "nonfinite"])
def test_failed_replay_propagates_and_releases_pool(model, codecs, failure):
    def fail(result):
        if failure == "raise":
            raise RuntimeError("Injected warmup output failure")
        result.visual.fill_(float("nan"))

    create, workers, _, stage = session_factory(
        model, codecs, "cross_chunk", pooled=True, output=fail
    )
    context = get_forward_context_or_none()
    with pytest.raises(RuntimeError):
        warmup_dit(model, codecs, create, height=32, width=32)
    assert len(workers) == 1
    assert workers[0].closed
    assert workers[0].state.inputs is None
    assert stage.kv_pool.available == 1
    assert get_forward_context_or_none() is context


def test_invalid_canvas_fails_before_creating_session(model, codecs):
    def create(*args):
        pytest.fail("Invalid input must not create a session")

    with pytest.raises(ValueError, match="latent-aligned canvas"):
        warmup_dit(model, codecs, create, height=31, width=32)
