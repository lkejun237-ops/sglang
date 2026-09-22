# SPDX-License-Identifier: Apache-2.0
"""AR Euler and CD re-noise block generation with isolated positive KV."""

import math
from dataclasses import dataclass

import torch
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_minimax_h3_euler_ancestral import (
    _minimax_h3_euler_interpolate,
    _minimax_h3_rf_v_to_x0,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_zing_h3_causal_renoise import (
    ZingH3CausalRenoiseSchedule,
)
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy
from sglang.multimodal_gen.runtime.utils.profiler import maybe_record_function
from sglang.multimodal_gen.runtime.utils.zing_diagnostics import current_diagnostic

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
        denoised = _minimax_h3_rf_v_to_x0(
            sample.float(), prediction.float(), self.timesteps[step_index]
        )
        ratio = self.sigmas[step_index + 1] / sigma
        return _minimax_h3_euler_interpolate(sample.float(), denoised, ratio).to(
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
        window_chunks=21,
        sink_chunks=3,
        rope_position_mode="block_relative",
        rope_max_frame_gap=12,
    ):
        if not math.isfinite(guidance_scale) or guidance_scale < 1:
            raise ValueError("Zing guidance_scale must be finite and at least one")
        if solver not in {"euler", "few_step"}:
            raise ValueError("Zing solver must be euler or few_step")
        if solver == "euler" and denoising_step_list is not None:
            raise ValueError("An explicit re-noise grid requires solver=few_step")
        self.generator, self.processor = transformer, packer
        self.window_policy = ZingWindowPolicy(window_chunks, sink_chunks)
        ZingWindowPolicy.validate_rope(rope_position_mode, rope_max_frame_gap)
        self.rope_position_mode = rope_position_mode
        self.rope_max_frame_gap = rope_max_frame_gap
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
    def initialize(self, inputs, visual, audio=None):
        diagnostic = current_diagnostic()
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
        vs = self.video_scheduler.to(self.device)
        aus = self.audio_scheduler.to(self.device)
        vs.diagnostic_modality, aus.diagnostic_modality = "video", "audio"
        forward_counts = {
            "prefix": 0,
            "source": 0,
            "denoise": 0,
            "unconditional": 0,
            "commit": 0,
        }

        self._forward_counts = forward_counts
        self.last_run_metrics = {
            "window_chunks": self.window_policy.window_chunks,
            "sink_chunks": self.window_policy.sink_chunks,
            "rope_position_mode": self.rope_position_mode,
            "rope_max_frame_gap": self.rope_max_frame_gap,
            "block_spans": [
                {
                    "source_block": source,
                    "target_block": target,
                    "start": start,
                    "end": end,
                    "frames": end - start,
                }
                for source, target, start, end in spans
            ],
            "video_sigma_points": int(vs.sigmas.numel()),
            "video_denoise_steps_per_block": int(vs.timesteps.numel()),
            "guidance_forwards_per_step": 2 if self.do_cfg else 1,
            "forward_counts": forward_counts,
            "dit_forwards": 0,
        }
        cache = ZingH3KVCache(
            len(self.generator.blocks),
            window_policy=self.window_policy,
            rope_position_mode=self.rope_position_mode,
            rope_max_frame_gap=self.rope_max_frame_gap,
            reserve_rows=int(pos["block_idx"].numel()),
        )
        try:
            if diagnostic:
                diagnostic.phase = "prefix"
            if bool((pos["block_idx"] == 0).any()):
                self._forward(
                    self._select_block(pos, 0, phase="prefix", video_t=1, audio_t=1),
                    cache,
                    "prefix",
                )
        except BaseException:
            cache.clear()
            raise
        return ZingH3SamplingState(visual, audio, pos, neg, spans, cache)

    @torch.no_grad()
    def update_condition(self, state, inputs):
        """Replace prefix conditioning at a block boundary, retaining target history."""
        if (
            state.closed
            or state.cache.active is not None
            or state.cache.pending_source is not None
        ):
            raise ValueError("Zing conditioning requires a completed block boundary")
        layout = self.processor.pack_inference_layout(
            inputs, inputs.positive, state.visual, state.audio
        )
        if self._block_spans(layout, inputs.visual_shape) != state.spans:
            raise ValueError("Zing prompt update changed target geometry")
        # Anchor the end of the new prompt to the original media origin so
        # absolute positions of committed and future video rows never move.
        old_target = state.positive["block_idx"] > 0
        new_target = layout["block_idx"] > 0
        shift = (
            state.positive["position_ids"][old_target][0, 0]
            - layout["position_ids"][new_target][0, 0]
        )
        layout["position_ids"][:, 0] += shift
        if not torch.allclose(
            layout["position_ids"][new_target],
            state.positive["position_ids"][old_target],
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError("Zing prompt update changed absolute target positions")
        # Reuse the original values exactly, including fractional frame positions.
        layout["position_ids"][new_target] = state.positive["position_ids"][old_target]
        prefix_cache = ZingH3KVCache(
            len(self.generator.blocks),
            window_policy=self.window_policy,
            rope_position_mode=self.rope_position_mode,
            rope_max_frame_gap=self.rope_max_frame_gap,
        )
        try:
            if bool((layout["block_idx"] == 0).any()):
                self._forward(
                    self._select_block(layout, 0, phase="prefix", video_t=1, audio_t=1),
                    prefix_cache,
                    "prefix",
                )
            state.cache.replace_prefix(prefix_cache.prefix)
            state.positive = layout
        finally:
            prefix_cache.clear()

    @torch.no_grad()
    def step(self, state, *, inputs=None):
        """Denoise one block; any execution failure closes the sampling state."""
        if state.closed or state.index >= len(state.spans):
            raise ValueError("Zing sampling state is closed or complete")
        visual, audio, cache = state.visual, state.audio, state.cache
        self.device = visual.device
        source, target, start, end = state.spans[state.index]
        try:
            cache.prepare_chunk(target)
            if inputs is not None:
                self.update_condition(state, inputs)
            history_rows = sum(c.positions.shape[0] for c in cache._history())
            with maybe_record_function(
                f"Zing.causal_block{target}.latent{start}-{end}.history_rows{history_rows}"
            ):
                self._sample_block(
                    visual,
                    audio,
                    state.positive,
                    state.negative,
                    cache,
                    self.video_scheduler,
                    self.audio_scheduler,
                    source,
                    target,
                    start,
                    end,
                )
        except BaseException:
            state.close()
            raise
        finally:
            self.last_run_metrics["dit_forwards"] = sum(self._forward_counts.values())
        state.index += 1
        return start, end

    def _sample_block(
        self, visual, audio, pos, neg, cache, vs, aus, source, target, start, end
    ):
        diagnostic = current_diagnostic()
        video = visual[:, :, start:end]
        audio_indices = self._audio_indices(pos, target)
        sound = None if audio is None else audio[:, :, audio_indices]
        if diagnostic:
            diagnostic.phase = f"block{source}.source"
        if bool((pos["block_idx"] == source).any()):
            self._forward(
                self._select_block(
                    pos, source, phase="source", video_t=1, audio_t=None
                ),
                cache,
                "source",
            )
        for step in range(len(vs.timesteps)):
            if diagnostic:
                diagnostic.phase = f"block{target}.step{step}"
            with maybe_record_function(f"Zing.denoise_step.block{target}.step{step}"):
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
        if diagnostic:
            diagnostic.phase = f"block{target}.commit"
        self._commit(pos, target, video, sound, cache)
        visual[:, :, start:end] = video
        if audio is not None:
            audio[:, :, audio_indices] = sound
        if diagnostic:
            diagnostic.record("block_complete", video=visual, audio=audio)

    @torch.no_grad()
    def sample(self, inputs, visual, audio=None):
        state = self.initialize(inputs, visual, audio)
        try:
            while state.index < len(state.spans):
                self.step(state)
            return state.visual, state.audio
        finally:
            state.close()
            del self._forward_counts


@dataclass
class ZingH3SamplingState:
    visual: torch.Tensor
    audio: torch.Tensor | None
    positive: dict
    negative: dict | None
    spans: list
    cache: ZingH3KVCache
    index: int = 0
    closed: bool = False

    def close(self):
        self.cache.clear()
        self.visual = self.audio = self.positive = self.negative = None
        self.spans = []
        self.closed = True
