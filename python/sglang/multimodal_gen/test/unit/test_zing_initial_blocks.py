# SPDX-License-Identifier: Apache-2.0
"""The two #297 deployments, checked against real small-model sampling."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.zing_sampling import (
    parse_noise_grid,
    validate_initial_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.realtime_sampling import (
    ZingH3RealtimeSampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model  # noqa: F401
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
    request,
)


@pytest.mark.parametrize("count", [0, 3])
def test_deployment_grids(count):
    regular = parse_noise_grid("1000,750")
    initial = parse_noise_grid("1000,750,500,250") if count else None
    validate_initial_blocks("few_step", regular, initial, count)
    assert regular == [1000, 750]


@pytest.mark.parametrize("count", [0, 3])
def test_deployment_cli(count):
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    argv = [
        "--model-path",
        "unused",
        "--zing-solver",
        "few_step",
        "--zing-denoising-step-list",
        "1000,750",
        "--zing-initial-block-count",
        str(count),
    ]
    if count:
        argv += ["--zing-initial-denoising-step-list", "1000,750,500,250"]
    args = parser.parse_args(argv)
    args.pipeline_config = SimpleNamespace(zing_adaln_table_path=None)
    ServerArgs._validate_zing_sampling(args)
    assert args.zing_denoising_step_list == [1000, 750]
    assert args.zing_initial_block_count == count


@pytest.mark.parametrize("count", [0, 3])
@torch.no_grad()
def test_actual_target_steps_and_new_generation(model, count):  # noqa: F811
    packer, inputs, _, _ = request(model, audio=False)
    packer.num_frame_per_block = [3, 2]
    packer.split_first_latent = True
    inputs.visual_shape = (10, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 10, 2, 2)
    noise = torch.randn_like(inputs.source_visual)
    sampler = ZingH3RealtimeSampler(
        model,
        packer,
        solver="few_step",
        denoising_step_list=[1000, 750],
        initial_denoising_step_list=[1000, 750, 500, 250] if count else None,
        initial_block_count=count,
        window_chunks=None,
        sink_chunks=0,
        rope_position_mode="absolute",
    )
    calls = []
    schedules = [sampler.video_scheduler]
    if count:
        schedules.append(sampler.initial_schedulers[0])
    for schedule in schedules:
        original = schedule.step

        def tracked(*args, _original=original, **kwargs):
            calls.append(kwargs["step_index"])
            return _original(*args, **kwargs)

        schedule.step = tracked
    expected = [4, 4, 4, 2, 2] if count else [2, 2, 2, 2, 2]
    for _ in range(2):
        calls.clear()
        state = sampler.initialize(inputs, noise)
        cache = state.cache
        try:
            while state.index < len(state.spans):
                if state.index == 2:
                    sampler.update_condition(state, inputs)
                sampler.step(state)
                assert state.cache is cache
            assert [
                r["latent_frames"] for r in sampler.last_run_metrics["block_schedules"]
            ] == [1, 2, 2, 3, 2]
            assert [
                r["denoise_steps"] for r in sampler.last_run_metrics["block_schedules"]
            ] == expected
            assert calls == [step for steps in expected for step in range(steps)]
        finally:
            state.close()


@pytest.mark.parametrize("count", [0, 3])
@torch.no_grad()
def test_ready_chunks_use_generation_index(model, count):  # noqa: F811
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance

    packer, inputs, _, _ = request(model, audio=False)
    packer.num_frame_per_block = [3, 2]
    packer.split_first_latent = True
    inputs.visual_shape = (10, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 10, 2, 2)
    noise = torch.randn_like(inputs.source_visual)
    sampler = ZingH3RealtimeSampler(
        model,
        packer,
        solver="few_step",
        denoising_step_list=[1000, 750],
        initial_denoising_step_list=[1000, 750, 500, 250] if count else None,
        initial_block_count=count,
        joint_forward="cross_chunk",
    )
    session = SourceReadySession(
        sampler,
        inputs,
        seed=42,
        max_pending_chunks=20,
        max_pending_bytes=1000000,
        total_latent_frames=10,
    )
    try:
        for chunk in packer.chunk_plan.chunks(10):
            session.append(
                inputs.source_visual[:, :, chunk.start : chunk.end],
                noise[:, :, chunk.start : chunk.end],
                provenance=fixture_provenance(
                    chunk.chunk_id + 1, 20, chunk.start, chunk.end
                ),
            )
        executed = []
        for _ in session.drain_ready():
            executed.append(
                sampler.last_run_metrics["last_block_schedule"]["denoise_steps"]
            )
        assert executed == ([4, 4, 4, 2, 2] if count else [2, 2, 2, 2, 2])
    finally:
        session.dispose()


def test_deployment_schedule_is_only_used_by_realtime(model):  # noqa: F811
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
        ZingH3Sampler,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )

    stage = object.__new__(ZingH3GenerationStage)
    stage.modules = {"transformer": model}
    stage.packer, _, _, _ = request(model, audio=False)
    stage.runtime = {
        "timestep_shift": 8.0,
        "audio_timestep_shift": 3.0,
        "num_train_timestep": 1000,
    }
    stage.deployment_sampling = {
        "solver": "few_step",
        "denoising_step_list": [1000, 750],
        "initial_denoising_step_list": [1000, 750, 500, 250],
        "initial_block_count": 3,
    }
    params = SimpleNamespace(
        zing_solver="few_step",
        zing_denoising_step_list=[1000, 500],
        num_inference_steps=50,
        guidance_scale=1.0,
    )
    offline = stage._build_sampler(params)
    realtime = stage._build_realtime_sampler(params)
    assert type(offline) is ZingH3Sampler
    assert type(realtime) is ZingH3RealtimeSampler
    expected = ZingH3Sampler(
        model,
        stage.packer,
        solver="few_step",
        denoising_step_list=[1000, 500],
        video_shift=8.0,
    )
    torch.testing.assert_close(
        offline.video_scheduler.timesteps, expected.video_scheduler.timesteps
    )
    assert realtime.denoising_step_list == [1000, 750]
    assert realtime.initial_block_count == 3


def test_realtime_step_restores_schedulers_on_failure(model, monkeypatch):  # noqa: F811
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
        ZingH3Sampler,
    )

    packer, _, _, _ = request(model, audio=False)
    sampler = ZingH3RealtimeSampler(
        model,
        packer,
        solver="few_step",
        denoising_step_list=[1000, 750],
        initial_denoising_step_list=[1000, 750, 500, 250],
        initial_block_count=3,
    )
    regular = sampler.video_scheduler, sampler.audio_scheduler

    def fail(self, state, *, inputs=None):
        assert self.video_scheduler is self.initial_schedulers[0]
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(ZingH3Sampler, "step", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        sampler.step(SimpleNamespace(index=0))
    assert (sampler.video_scheduler, sampler.audio_scheduler) == regular
