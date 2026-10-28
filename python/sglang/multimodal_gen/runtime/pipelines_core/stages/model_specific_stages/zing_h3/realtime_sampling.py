# SPDX-License-Identifier: Apache-2.0
"""Realtime-only initial-block schedules; the offline sampler stays unchanged."""

from dataclasses import dataclass

import torch

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
        # Admission must see the regular plan even while step() binds the initial
        # plan for a prompt update inside one synchronous block.
        self.regular_schedulers = self.video_scheduler, self.audio_scheduler
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
        pairs = [self.regular_schedulers]
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
        return self.regular_schedulers

    def _admit_adaln(self, positive, negative, spans, audio):
        if not self.initial_block_count and self.joint_forward == "off":
            return super()._admit_adaln(positive, negative, spans, audio)
        table = getattr(self.generator, "adaln_table", None)
        if table is None:
            return None
        if audio is not None:
            raise ValueError("Zing weights-free AdaLN supports video requests only")
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
            initial_blocks_plan,
            video_plan,
        )

        schedules = [self.regular_schedulers[0]]
        if self.initial_schedulers is not None:
            schedules.append(self.initial_schedulers[0])
        plans = [
            video_plan(
                self.solver,
                schedule.timesteps.cpu(),
                self.ar_condition_timestep,
                self.reference_visual_timestep,
                sigmas=schedule.sigmas.cpu(),
            )
            for schedule in schedules
        ]
        plan = (
            initial_blocks_plan(*plans, self.initial_block_count)
            if self.initial_block_count
            else plans[0]
        )
        timesteps = sorted(
            {t for schedule in schedules for t in self._host_list(schedule.timesteps)}
        )
        batches = []

        def add(roles, *, video_t=1, final=False):
            if roles.numel():
                batches.append(
                    torch.unique(
                        self._block_timesteps(roles, video_t=video_t, final=final),
                        sorted=True,
                    )
                )

        # Inspect actual packed roles, not just the manifest's declared batch list.
        # Both plans are required even for a short request that ends before switching.
        # Joint source_first/cross_chunk project each unique timestep separately:
        # the clean source/final and active singleton batches below cover them.
        # Prefix forwards still require their original full mixed-role batch.
        for layout in (positive, negative):
            if layout is None:
                continue
            host = self._layout_index(layout).host
            roles, blocks = host["row_role"], host["block_idx"]
            if bool(((roles == 5) | (roles == 11)).any()):
                raise ValueError(
                    "Zing weights-free AdaLN does not support audio row roles"
                )
            if not bool(torch.isin(roles, torch.tensor([0, 1, 4, 7, 10])).all()):
                raise ValueError(
                    "Zing weights-free AdaLN received unsupported row roles"
                )
            add(roles[blocks == 0])
            for source, target, _, _ in spans:
                add(roles[(blocks == source) & (roles == 7)])
                target_roles = roles[(blocks == target) & (roles == 10)]
                for t in timesteps:
                    add(target_roles, video_t=t)
                add(target_roles, final=True)
        return table.prepare(
            batches,
            device=self.device,
            schedule=plan["schedule"],
            embedder=self.generator.time_embedder,
        )

    def initialize(self, inputs, visual, audio=None, **kwargs):
        self.prepare_schedules(visual.device)
        state = super().initialize(inputs, visual, audio, **kwargs)
        self.last_run_metrics["block_schedules"] = []
        return state

    def step(self, state, *, first_prediction=None):
        # The existing step owns cache/condition/error handling. Bind schedules only
        # for this synchronous block, then restore even on cancellation or failure.
        # A sampler belongs to the deployment's single active generation.
        regular = self.video_scheduler, self.audio_scheduler
        self.video_scheduler, self.audio_scheduler = self.schedules_for(state)
        try:
            if first_prediction is None:
                return super().step(state)
            return super().step(state, first_prediction=first_prediction)
        finally:
            self.video_scheduler, self.audio_scheduler = regular

    def _ready_cross_prediction(self, previous, current):
        # Lookahead belongs to the next generation block, including across the
        # initial/regular boundary. Leave the offline lookahead path unchanged.
        regular = self.video_scheduler, self.audio_scheduler
        self.video_scheduler, self.audio_scheduler = self.schedules_for(current)
        try:
            return super()._ready_cross_prediction(previous, current)
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
