# SPDX-License-Identifier: Apache-2.0
"""AR Euler and CD re-noise block generation with isolated positive KV."""

import math

import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_zing_h3_causal_renoise import (
    ZingH3CausalRenoiseSchedule,
)

from ._sampling_helpers import ZingBlockSamplingHelpers


class ZingH3EulerSchedule:
    def __init__(self, sigma_points, shift):
        if sigma_points < 2 or not math.isfinite(shift) or shift <= 0:
            raise ValueError(
                "Zing Euler requires at least two sigma points and a positive shift"
            )
        base = torch.linspace(1, 0, sigma_points, dtype=torch.float32)
        self.sigmas = torch.unique_consecutive(shift * base / (1 + (shift - 1) * base))
        self.timesteps = 1 - self.sigmas[:-1]

    def to(self, device):
        self.sigmas, self.timesteps = self.sigmas.to(device), self.timesteps.to(device)
        return self

    def step(self, prediction, sample, *, step_index):
        sigma = self.sigmas[step_index]
        denoised = (
            sample.float() + (1 - self.timesteps[step_index]) * prediction.float()
        )
        ratio = self.sigmas[step_index + 1] / sigma
        return (ratio * sample.float() + (1 - ratio) * denoised).to(
            sample.dtype
        ), denoised


class ZingH3Sampler(ZingBlockSamplingHelpers):
    def __init__(
        self,
        transformer,
        packer,
        *,
        solver="euler",
        sigma_points=50,
        denoising_step_list=None,
        num_train_timesteps=1000,
        video_shift=5.0,
        audio_shift=3.0,
        guidance_scale=1.0,
    ):
        if not math.isfinite(guidance_scale) or guidance_scale < 1:
            raise ValueError("Zing guidance_scale must be finite and at least one")
        if solver not in {"euler", "few_step"}:
            raise ValueError("Zing solver must be euler or few_step")
        if solver == "euler" and denoising_step_list is not None:
            raise ValueError("An explicit re-noise grid requires solver=few_step")
        self.generator, self.processor = transformer, packer
        self.guidance_scale, self.do_cfg = guidance_scale, guidance_scale > 1

        def schedule(shift):
            if solver == "euler":
                return ZingH3EulerSchedule(sigma_points, shift)
            if denoising_step_list is None:
                raise ValueError(
                    "Zing few_step requires the distilled checkpoint's explicit noise grid"
                )
            return ZingH3CausalRenoiseSchedule(
                denoising_step_list,
                num_train_timesteps=num_train_timesteps,
                shift=shift,
            )

        self.video_scheduler, self.audio_scheduler = (
            schedule(video_shift),
            schedule(audio_shift),
        )

    @torch.no_grad()
    def sample(self, inputs, visual, audio=None):
        self.device = visual.device
        self.processor._validate_visual(visual)
        if audio is not None:
            self.processor._validate_audio(audio)
        # Keep FP32 master latents through H3's FP32 input projections.
        visual = visual.float().clone()
        audio = None if audio is None else audio.float().clone()
        pos = self.processor.pack_inference_layout(
            inputs, inputs.positive, visual, audio
        )
        neg = (
            self.processor.pack_inference_layout(inputs, None, visual, audio)
            if self.do_cfg
            else None
        )
        spans = self._block_spans(pos, inputs.visual_shape)
        vs, aus = (
            self.video_scheduler.to(self.device),
            self.audio_scheduler.to(self.device),
        )
        cache = ZingH3KVCache(len(self.generator.blocks))
        try:
            if bool((pos["block_idx"] == 0).any()):
                self._forward(
                    self._select_block(pos, 0, phase="prefix", video_t=1, audio_t=1),
                    cache,
                    "prefix",
                )
            for source, target, start, end in spans:
                video = visual[:, :, start:end]
                audio_indices = self._audio_indices(pos, target)
                sound = None if audio is None else audio[:, :, audio_indices]
                if bool((pos["block_idx"] == source).any()):
                    self._forward(
                        self._select_block(
                            pos, source, phase="source", video_t=1, audio_t=None
                        ),
                        cache,
                        "source",
                    )
                for step in range(len(vs.timesteps)):
                    vt = float(vs.timesteps[step])
                    at = None if sound is None else float(aus.timesteps[step])
                    self._set_block_latents(pos, target, video, sound)
                    packed = self._select_block(
                        pos, target, phase="target", video_t=vt, audio_t=at
                    )
                    vp, ap = self._predict(packed, video.shape[2:], cache, "active")
                    if self.do_cfg:
                        self._set_block_latents(neg, target, video, sound)
                        uncond = self._select_block(
                            neg, target, phase="target", video_t=vt, audio_t=at
                        )
                        vu, au = self._predict(uncond, video.shape[2:], None, None)
                        vp = vu + self.guidance_scale * (vp - vu)
                        if ap is not None:
                            ap = au + self.guidance_scale * (ap - au)
                    # Preserve video-then-audio RNG consumption for CD parity.
                    video = vs.step(vp, video, step_index=step)[0]
                    if sound is not None:
                        sound = aus.step(ap, sound, step_index=step)[0]
                visual[:, :, start:end] = video
                if audio is not None:
                    audio[:, :, audio_indices] = sound
                self._commit(pos, target, video, sound, cache)
            return visual, audio
        finally:
            cache.clear()
