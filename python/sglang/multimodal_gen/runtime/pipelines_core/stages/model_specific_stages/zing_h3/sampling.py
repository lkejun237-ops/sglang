# SPDX-License-Identifier: Apache-2.0
"""AR Euler and CD re-noise block generation with isolated positive KV."""

import math
from copy import copy
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
from .packing import H3RowRole


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
        return (
            _minimax_h3_euler_interpolate(sample.float(), denoised, ratio).to(
                sample.dtype
            ),
            denoised,
        )


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
        joint_forward="off",
        ar_condition_timestep=1.0,
        reference_visual_timestep=1.0,
    ):
        if not math.isfinite(guidance_scale) or guidance_scale < 1:
            raise ValueError("Zing guidance_scale must be finite and at least one")
        if solver not in {"euler", "few_step"}:
            raise ValueError("Zing solver must be euler or few_step")
        if solver == "euler" and denoising_step_list is not None:
            raise ValueError("An explicit re-noise grid requires solver=few_step")
        if joint_forward not in {"off", "source_first", "cross_chunk"}:
            raise ValueError(
                "Zing joint_forward must be off, source_first, or cross_chunk"
            )
        self.joint_forward = joint_forward
        for value in (ar_condition_timestep, reference_visual_timestep):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    "Zing condition timesteps must be finite and within [0, 1]"
                )
        self.solver = solver
        self.ar_condition_timestep = ar_condition_timestep
        self.reference_visual_timestep = reference_visual_timestep
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
        # Preallocated donor storage for prompt updates of pooled sessions.
        self.prefix_workspace = None

    def _admit_adaln(self, positive, negative, spans, audio):
        table = getattr(self.generator, "adaln_table", None)
        if table is None:
            return None
        if audio is not None:
            raise ValueError("Zing weights-free AdaLN supports video requests only")
        if self.joint_forward != "off":
            raise ValueError("Zing weights-free AdaLN does not support joint forwards")
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
            schedule_contract,
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

        timesteps = self._host_list(self.video_scheduler.timesteps)
        for layout in [positive, negative]:
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
        schedule = schedule_contract(
            self.solver,
            self.video_scheduler.timesteps.cpu(),
            self.ar_condition_timestep,
            self.reference_visual_timestep,
            sigmas=self.video_scheduler.sigmas.cpu(),
        )
        return table.prepare(
            batches,
            device=self.device,
            schedule=schedule,
            embedder=self.generator.time_embedder,
        )

    def prefix_donor_cache(self):
        """Cache that computes a replacement prefix before replace_prefix.

        Pooled sessions borrow the worker's prefix workspace, so a prompt
        update never allocates K/V outside the pool budget. The donor is
        cleared right after its prefix is copied into the session slot.
        """
        return ZingH3KVCache(
            len(self.generator.blocks),
            window_policy=self.window_policy,
            rope_position_mode=self.rope_position_mode,
            rope_max_frame_gap=self.rope_max_frame_gap,
            storage=self.prefix_workspace,
        )

    def rebuild_prompt_prefix(self, inputs, condition, visual_prefix, cache):
        """Replace all logical KV with a new condition and clean latent prefix."""
        prefix_inputs = copy(inputs)
        prefix_inputs.positive = condition
        prefix_inputs.source_visual = None
        layout, chunks = self.processor.pack_prompt_switch_prefix(
            prefix_inputs, condition, visual_prefix
        )
        cache.clear()
        if bool((layout["block_idx"] == 0).any()):
            self._forward(
                self._select_block(layout, 0, phase="prefix", video_t=1.0, audio_t=1.0),
                cache,
                "prefix",
            )
        for chunk in chunks:
            self._commit_clean_prefix(
                layout,
                chunk.target_block,
                visual_prefix[:, :, chunk.start : chunk.end],
                cache,
            )
        target = layout["row_role"] == int(H3RowRole.TARGET_VISUAL)
        return float(layout["position_ids"][target][0, 0]), chunks[-1].end

    @torch.no_grad()
    def initialize(
        self, inputs, visual, audio=None, *, storage=None, acquire_storage=None
    ):
        """Pack the session and create its cache.

        ``acquire_storage(layout)`` leases pooled storage once the packed
        prefix size is known and before any forward writes K/V.
        """
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

        if self.joint_forward != "off":
            forward_counts.update(source_first=0, cross_chunk=0)
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
        if self.joint_forward != "off":
            self.last_run_metrics.update(
                joint_forward=self.joint_forward, joint_fallbacks={}
            )
        adaln_lease = self._admit_adaln(pos, neg, spans, audio)
        if getattr(self.generator, "adaln_table", None) is not None:
            self.last_run_metrics["adaln_deployment"] = (
                self.generator.adaln_deployment_receipt()
            )
        cache = None
        try:
            if acquire_storage is not None:
                storage = acquire_storage(pos)
            cache = ZingH3KVCache(
                len(self.generator.blocks),
                window_policy=self.window_policy,
                rope_position_mode=self.rope_position_mode,
                rope_max_frame_gap=self.rope_max_frame_gap,
                reserve_rows=int(pos["block_idx"].numel()),
                storage=storage,
            )
            if diagnostic:
                diagnostic.phase = "prefix"
            if bool((pos["block_idx"] == 0).any()):
                self._forward(
                    self._select_block(pos, 0, phase="prefix", video_t=1, audio_t=1),
                    cache,
                    "prefix",
                )
        except BaseException:
            if cache is not None:
                cache.clear()
            if adaln_lease is not None:
                adaln_lease.close()
            raise
        return ZingH3SamplingState(
            visual, audio, pos, neg, spans, cache, adaln_lease=adaln_lease
        )

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
        adaln_lease = self._admit_adaln(
            layout, state.negative, state.spans, state.audio
        )
        prefix_cache = None
        try:
            prefix_cache = self.prefix_donor_cache()
            if bool((layout["block_idx"] == 0).any()):
                self._forward(
                    self._select_block(layout, 0, phase="prefix", video_t=1, audio_t=1),
                    prefix_cache,
                    "prefix",
                )
            state.cache.replace_prefix(prefix_cache.prefix)
            state.positive = layout
        finally:
            if prefix_cache is not None:
                prefix_cache.clear()
            if adaln_lease is not None:
                adaln_lease.close()

    @torch.no_grad()
    def step(self, state, *, inputs=None):
        """Denoise one block; any execution failure closes the sampling state."""
        if state.closed or state.index >= len(state.spans):
            raise ValueError("Zing sampling state is closed or complete")
        if self.joint_forward == "cross_chunk":
            reasons = self.last_run_metrics["joint_fallbacks"]
            reasons["streaming_boundary"] = reasons.get("streaming_boundary", 0) + 1
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
        diagnostic = current_diagnostic()
        video = visual[:, :, start:end]
        audio_indices = self._audio_indices(pos, target)
        sound = None if audio is None else audio[:, :, audio_indices]
        if diagnostic:
            diagnostic.phase = f"block{source}.source"
        joint_source = first_prediction is None and self._joint_eligible(
            pos, cache, source, target, audio
        )
        if (
            first_prediction is None
            and not joint_source
            and self._has_block(pos, source)
        ):
            self._forward(
                self._select_block(
                    pos, source, phase="source", video_t=1, audio_t=None
                ),
                cache,
                "source",
            )
        video_timesteps = self._host_list(vs.timesteps)
        audio_timesteps = None if sound is None else self._host_list(aus.timesteps)
        for step in range(len(vs.timesteps)):
            if diagnostic:
                diagnostic.phase = f"block{target}.step{step}"
            with maybe_record_function(f"Zing.denoise_step.block{target}.step{step}"):
                vt = video_timesteps[step]
                at = None if sound is None else audio_timesteps[step]
                self._set_block_latents(pos, target, video, sound)
                packed = self._select_block(
                    pos, target, phase="target", video_t=vt, audio_t=at
                )
                if step == 0 and first_prediction is not None:
                    vp, ap = first_prediction
                elif step == 0 and joint_source:
                    source_packed = self._select_block(
                        pos, source, phase="source", video_t=1, audio_t=None
                    )
                    prediction = self._predict_joint(
                        [source_packed, packed],
                        ("source", "active"),
                        video.shape[2:],
                        cache,
                    )
                    if prediction is None:
                        self._forward(source_packed, cache, "source")
                        prediction = self._predict(
                            packed, video.shape[2:], cache, "active"
                        )
                    vp, ap = prediction
                else:
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
        if not defer_final:
            self._commit(pos, target, video, sound, cache)
        visual[:, :, start:end] = video
        if audio is not None:
            audio[:, :, audio_indices] = sound
        if diagnostic:
            diagnostic.record("block_complete", video=visual, audio=audio)
        return video, sound

    def _joint_eligible(self, pos, cache, source, target, audio, *, cross=False):
        if self.joint_forward == "off":
            return False
        from sglang.multimodal_gen.runtime.models.dits.zing_h3 import get_ulysses_ctx

        reason = None
        roles = self._layout_index(pos).host["row_role"]
        if get_ulysses_ctx()[0] not in (1, 2):
            reason = "sp_size"
        elif self.do_cfg:
            reason = "cfg"
        elif audio is not None or bool(((roles == 5) | (roles == 11)).any()):
            reason = "audio"
        elif not self._has_block(pos, source):
            reason = "source_not_ready"
        elif not cache.raw_store or cache.local_frames != -1:
            reason = "cache_layout"
        elif cross and not all(
            cache.window_policy.retains(i, target // 2 - 1)
            for i in [c.temporal_chunk_id for c in cache.committed] + [target // 2 - 2]
        ):
            reason = "window_eviction"
        elif cache.pooled:
            blocks = self._layout_index(pos).host["block_idx"]
            selected = (blocks == source) | (blocks == target)
            if cross:
                selected = selected | (blocks == target - 2)
            if cache._length + int(selected.sum()) > cache.capacity:
                reason = "pooled_scratch_capacity"
        if reason is not None:
            counts = self.last_run_metrics["joint_fallbacks"]
            counts[reason] = counts.get(reason, 0) + 1
        return reason is None

    def _ready_cross_prediction(self, previous, current):
        """Join two separately packed ready chunks without moving RNG draws."""
        source, target, start, end = current.spans[0]
        old_target = previous.spans[0][1]
        cache, pos = current.cache, current.positive
        if not self._joint_eligible(
            pos, cache, source, target, current.audio, cross=True
        ):
            return None
        self._set_block_latents(
            previous.positive, old_target, previous.visual, previous.audio
        )
        self._set_block_latents(pos, target, current.visual, current.audio)
        packs = [
            self._select_block(
                previous.positive,
                old_target,
                phase="target",
                video_t=1,
                audio_t=None,
                final=True,
            ),
            self._select_block(pos, source, phase="source", video_t=1, audio_t=None),
            self._select_block(
                pos,
                target,
                phase="target",
                video_t=self._host_list(self.video_scheduler.timesteps)[0],
                audio_t=None,
            ),
        ]
        # The previous final is absent from current's separate packed layout.
        if (
            cache.pooled
            and cache._length + sum(p["x"].shape[0] for p in packs) > cache.capacity
        ):
            reasons = self.last_run_metrics["joint_fallbacks"]
            reasons["pooled_scratch_capacity"] = (
                reasons.get("pooled_scratch_capacity", 0) + 1
            )
            return None
        return self._predict_joint(
            packs, ("final", "source", "active"), current.visual.shape[2:], cache
        )

    def _sample_cross_chunk(self, state):
        """Offline-only lookahead: all source/noise inputs are already available.

        Streaming step() still commits before returning each block. This loop
        does not prepare noise early or change scheduler/RNG consumption.
        """
        previous = None
        for source, target, start, end in state.spans:
            pos, cache = state.positive, state.cache
            first = None
            if previous is not None:
                old_target, video, sound = previous
                if self._joint_eligible(
                    pos, cache, source, target, state.audio, cross=True
                ):
                    self._set_block_latents(pos, old_target, video, sound)
                    self._set_block_latents(
                        pos, target, state.visual[:, :, start:end], None
                    )
                    packs = [
                        self._select_block(
                            pos,
                            old_target,
                            phase="target",
                            video_t=1,
                            audio_t=None,
                            final=True,
                        ),
                        self._select_block(
                            pos, source, phase="source", video_t=1, audio_t=None
                        ),
                        self._select_block(
                            pos,
                            target,
                            phase="target",
                            video_t=self._host_list(self.video_scheduler.timesteps)[0],
                            audio_t=None,
                        ),
                    ]
                    first = self._predict_joint(
                        packs,
                        ("final", "source", "active"),
                        state.visual[:, :, start:end].shape[2:],
                        cache,
                    )
                if first is None:
                    self._commit(pos, old_target, video, sound, cache)
            if first is None:
                cache.prepare_chunk(target)
            video, sound = self._sample_block(
                state.visual,
                state.audio,
                pos,
                state.negative,
                cache,
                self.video_scheduler,
                self.audio_scheduler,
                source,
                target,
                start,
                end,
                first_prediction=first,
                defer_final=True,
            )
            previous = (target, video, sound)
            state.index += 1
        if previous is not None:
            self._commit(state.positive, *previous, state.cache)
        self.last_run_metrics["dit_forwards"] = sum(self._forward_counts.values())

    @torch.no_grad()
    def sample(self, inputs, visual, audio=None):
        state = self.initialize(inputs, visual, audio)
        try:
            if self.joint_forward == "cross_chunk":
                self._sample_cross_chunk(state)
            else:
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
    adaln_lease: object = None

    def close(self):
        self.cache.clear()
        if self.adaln_lease is not None:
            self.adaln_lease.close()
            self.adaln_lease = None
        self.visual = self.audio = self.positive = self.negative = None
        self.spans = []
        self.closed = True
