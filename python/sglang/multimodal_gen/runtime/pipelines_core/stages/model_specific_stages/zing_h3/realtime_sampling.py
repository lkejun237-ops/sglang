# SPDX-License-Identifier: Apache-2.0
"""Realtime-only initial-block schedules; the offline sampler stays unchanged."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.zing_sampling import validate_initial_blocks
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_zing_h3_causal_renoise import (
    ZingH3CausalRenoiseSchedule,
)

from .sampling import ZingH3Sampler, ZingH3SamplingState


@dataclass
class ZingH3RealtimeSamplingState(ZingH3SamplingState):
    # Source-ready spans start at local index zero; this cursor spans the generation.
    generation_block_index: int | None = None


class ZingH3RealtimeSampler(ZingH3Sampler):
    def __init__(
        self,
        transformer,
        processor,
        *,
        solver="euler",
        denoising_step_list=None,
        initial_denoising_step_list=None,
        initial_block_count=0,
        **kwargs,
    ):
        validate_initial_blocks(
            solver,
            denoising_step_list,
            initial_denoising_step_list,
            initial_block_count,
        )
        if (
            initial_block_count
            and getattr(transformer, "adaln_table", None) is not None
        ):
            raise ValueError(
                "Zing mixed schedules do not support offline single-schedule AdaLN tables"
            )
        super().__init__(
            transformer,
            processor,
            solver=solver,
            denoising_step_list=denoising_step_list,
            **kwargs,
        )
        self.initial_block_count = initial_block_count
        self.denoising_step_list = denoising_step_list
        self.initial_denoising_step_list = initial_denoising_step_list
        self.initial_schedulers = None
        if initial_block_count:
            self.initial_schedulers = tuple(
                ZingH3CausalRenoiseSchedule(
                    initial_denoising_step_list,
                    num_train_timesteps=kwargs.get("num_train_timesteps", 1000),
                    shift=shift,
                )
                for shift in (
                    kwargs.get("video_shift", 5.0),
                    kwargs.get("audio_shift", 3.0),
                )
            )

    def prepare_schedules(self, device):
        pairs = [(self.video_scheduler, self.audio_scheduler)]
        if self.initial_schedulers is not None:
            pairs.append(self.initial_schedulers)
        for pair in pairs:
            for schedule, modality in zip(pair, ("video", "audio")):
                schedule.to(device)
                schedule.diagnostic_modality = modality

    def schedules_for(self, state):
        index = getattr(state, "generation_block_index", None)
        if index is None:
            index = state.index
        if index < self.initial_block_count:
            return self.initial_schedulers
        return self.video_scheduler, self.audio_scheduler

    def initialize(self, inputs, visual, audio=None, **kwargs):
        self.prepare_schedules(visual.device)
        state = super().initialize(inputs, visual, audio, **kwargs)
        self.last_run_metrics["block_schedules"] = []
        return state

    def step(self, state, *, inputs=None):
        # The existing step owns cache/condition/error handling. Bind schedules only
        # for this synchronous block, then restore even on cancellation or failure.
        # A sampler belongs to the deployment's single active generation.
        regular = self.video_scheduler, self.audio_scheduler
        self.video_scheduler, self.audio_scheduler = self.schedules_for(state)
        try:
            return super().step(state, inputs=inputs)
        finally:
            self.video_scheduler, self.audio_scheduler = regular

    def _sample_block(
        self,
        visual,
        audio,
        pos,
        neg,
        cache,
        vs,
        aus,
        source,
        target,
        start,
        end,
        *,
        first_prediction=None,
        defer_final=False,
    ):
        result = super()._sample_block(
            visual,
            audio,
            pos,
            neg,
            cache,
            vs,
            aus,
            source,
            target,
            start,
            end,
            first_prediction=first_prediction,
            defer_final=defer_final,
        )
        receipt = {
            "target_block": int(target),
            "latent_frames": end - start,
            "denoise_steps": len(vs.timesteps),
            "noise_grid": self.initial_denoising_step_list
            if self.initial_schedulers is not None and vs is self.initial_schedulers[0]
            else self.denoising_step_list,
        }
        self.last_run_metrics["last_block_schedule"] = receipt
        if "block_schedules" in self.last_run_metrics:
            self.last_run_metrics["block_schedules"].append(receipt)
        return result
