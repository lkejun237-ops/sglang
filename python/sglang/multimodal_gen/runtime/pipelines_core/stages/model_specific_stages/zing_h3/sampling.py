# SPDX-License-Identifier: Apache-2.0
"""AR Euler and CD re-noise block generation with isolated positive KV."""

import math
from copy import copy
from dataclasses import dataclass

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_minimax_h3_euler_ancestral import (
    _minimax_h3_euler_interpolate,
    _minimax_h3_rf_v_to_x0,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_zing_h3_causal_renoise import (
    ZingH3CausalRenoiseSchedule,
)
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
    ZingKVCapacityError,
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
        window_eviction_timing="after_step0",
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
        if window_eviction_timing not in ("before_source", "after_step0"):
            raise ValueError("Invalid Zing window eviction timing")
        self._window_eviction_timing = window_eviction_timing
        self._delayed_policy_requires_agreement = (
            window_eviction_timing == "after_step0"
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

    @property
    def window_eviction_timing(self):
        """A session's window policy cannot change after construction."""
        return self._window_eviction_timing

    def _admit_adaln(self, positive, negative, spans, audio):
        table = getattr(self.generator, "adaln_table", None)
        if table is None:
            return None
        if audio is not None:
            raise ValueError("Zing weights-free AdaLN supports video requests only")
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

    def pack_prompt_prefix(self, inputs, condition, visual_prefix):
        """Prepare condition + 1/2/2 history without changing the live KV cache."""
        prefix_inputs = copy(inputs)
        prefix_inputs.positive, prefix_inputs.source_visual = condition, None
        layout, chunks = self.processor.pack_prompt_switch_prefix(
            prefix_inputs, condition, visual_prefix
        )
        packs = [self._select_block(layout, 0, phase="prefix", video_t=1, audio_t=1)]
        for chunk in chunks:
            self._set_block_latents(
                layout,
                chunk.target_block,
                visual_prefix[:, :, chunk.start : chunk.end],
                None,
            )
            packs.append(
                self._select_block(
                    layout,
                    chunk.target_block,
                    phase="target",
                    video_t=1,
                    audio_t=None,
                    final=True,
                )
            )
        host = self._layout_index(layout).host
        target = host["row_role"] == int(H3RowRole.TARGET_VISUAL)
        return packs, float(host["position_ids"][target][0, 0]), chunks[-1].end

    def rebuild_prompt_prefix(
        self, inputs, condition, visual_prefix, cache, *, restore_prefix=None
    ):
        """Replace all logical KV with a new condition and clean latent prefix."""
        prefix_inputs = copy(inputs)
        prefix_inputs.positive = condition
        prefix_inputs.source_visual = None
        layout, chunks = self.processor.pack_prompt_switch_prefix(
            prefix_inputs, condition, visual_prefix
        )
        cache.clear()
        if restore_prefix is not None:
            restore_prefix(cache)
        elif bool((layout["block_idx"] == 0).any()):
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
        self._agree_sp_policy()
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
                window_eviction_timing=self.window_eviction_timing,
                rope_position_mode=self.rope_position_mode,
                rope_max_frame_gap=self.rope_max_frame_gap,
                reserve_rows=int(pos["block_idx"].numel()),
                storage=storage,
            )
            first_source, first_target, _, _ = spans[0]
            self._agree_sp_boundary(
                cache,
                first_target,
                phase="initial_prefix",
                required_rows=int(
                    (self._layout_index(pos).host["block_idx"] == 0).sum()
                ),
                layout_signature=self._collective_layout_signature(
                    pos, first_source, first_target
                ),
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
    def step(self, state, *, first_prediction=None):
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
            if first_prediction is None:
                self._begin_chunk(cache, state.positive, source, target)
            history_rows = sum(c.positions.shape[0] for c in cache._history())
            with maybe_record_function(
                f"Zing.causal_block{target}.latent{start}-{end}.history_rows{history_rows}"
            ):
                warm_start = (
                    {}
                    if first_prediction is None
                    else {"first_prediction": first_prediction}
                )
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
                    **warm_start,
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
        if not len(vs.timesteps):
            raise ValueError("Zing denoising requires at least one scheduler step")
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
                if step == 0:
                    cache.complete_step0_and_evict(target)
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

    def _collective_config(self):
        return {
            "guidance": (self.guidance_scale, self.do_cfg),
            "schedule": (
                self.solver,
                tuple(self._host_list(self.video_scheduler.timesteps)),
                tuple(self._host_list(self.video_scheduler.sigmas)),
                tuple(self._host_list(self.audio_scheduler.timesteps)),
                tuple(self._host_list(self.audio_scheduler.sigmas)),
            ),
            "condition_timesteps": (
                self.ar_condition_timestep,
                self.reference_visual_timestep,
            ),
        }

    def _collective_layout_signature(self, pos, source, target):
        """Host geometry that determines modality and collective row shapes.

        Absolute coordinate values may produce a joint RoPE fallback; they do
        not change communication shapes, so preserve that supported fallback.
        """
        host = self._layout_index(pos).host
        blocks, roles, positions = (
            host[k] for k in ("block_idx", "row_role", "position_ids")
        )
        selected = (blocks == 0) | (blocks == source) | (blocks == target)
        geometry = []
        for block in (0, source, target):
            current = positions[blocks == block]
            geometry.append(
                (
                    block,
                    current.shape[0],
                    tuple(int(current[:, axis].unique().numel()) for axis in range(3)),
                )
            )
        return (
            tuple(blocks[selected].tolist()),
            tuple(roles[selected].tolist()),
            tuple(geometry),
            tuple(
                (name, tuple(pos[name].shape))
                for name in (
                    "clean_visual",
                    "clean_audio",
                    "visual",
                    "audio",
                    "x",
                    "audio_x",
                    "context",
                )
                if name in pos
            ),
        )

    def _agree_sp_policy(self):
        """Every real SP rank agrees the policy before any policy-specific votes."""
        from sglang.multimodal_gen.runtime.models.dits import zing_h3

        world, _ = zing_h3.get_ulysses_ctx()
        if world == 1:
            return
        group = getattr(zing_h3.get_sp_group(), "cpu_group", None)
        if group is None:
            if self.window_eviction_timing == "after_step0":
                raise ValueError(
                    "Zing delayed window rank agreement requires a CPU group"
                )
            return
        votes = [None] * world
        config = {"policy": self.window_eviction_timing, **self._collective_config()}
        dist.all_gather_object(votes, config, group=group)
        if any(vote != config for vote in votes):
            raise ValueError("Zing window policy differs across ranks")

    def _agree_sp_boundary(
        self,
        cache,
        target,
        *,
        cross=False,
        joint_eligible=None,
        required_rows=0,
        precheck_error=False,
        local_error=None,
        layout_signature=None,
        phase=None,
        needs_prefix_update=None,
        ready_route=None,
    ):
        """Vote on logical state and capacity before entering model collectives."""
        from sglang.multimodal_gen.runtime.models.dits import zing_h3

        world, _ = zing_h3.get_ulysses_ctx()
        if not self._delayed_policy_requires_agreement:
            return joint_eligible
        payload = {
            **self._collective_config(),
            "phase": phase,
            "needs_prefix_update": needs_prefix_update,
            "ready_route": ready_route,
            "layout_signature": layout_signature,
            "required_rows": required_rows,
            "cache_ready": cache is not None,
            "active_geometry": None
            if cache is None or cache.active is None
            else tuple(cache.active[1].shape),
            "policy": self.window_eviction_timing
            if cache is None
            else cache.window_eviction_timing,
            "sampler_policy": self.window_eviction_timing,
            "joint_mode": self.joint_forward,
            "window": (
                self.window_policy.window_chunks
                if cache is None
                else cache.window_policy.window_chunks,
                self.window_policy.sink_chunks
                if cache is None
                else cache.window_policy.sink_chunks,
            ),
            "rope": (self.rope_position_mode, self.rope_max_frame_gap)
            if cache is None
            else (cache.rope_position_mode, cache.rope_max_frame_gap),
            "steps": len(self.video_scheduler.timesteps),
            "stored": tuple(
                (c.block, c.offset, c.positions.shape[0])
                for c in (() if cache is None else cache._stored_chunks())
            ),
            "target": target,
            "last_block": None if cache is None else cache.last_block,
            "eviction_target": None if cache is None else cache.eviction_target_block,
            "pending": None
            if cache is None or cache.pending_source is None
            else cache.pending_source.block,
            "active": None
            if cache is None or cache.active is None
            else cache.active[0],
            "committed": tuple(
                c.temporal_chunk_id for c in (() if cache is None else cache.committed)
            ),
            "retained": tuple(
                c.temporal_chunk_id
                for c in (() if cache is None else cache.committed)
                if cache.window_policy.retains(c.temporal_chunk_id, target // 2 - 1)
            ),
            "capacity_ok": cache is None
            or cache.window_eviction_timing == "before_source"
            or not cache.pooled
            or cache._length + required_rows <= cache.capacity,
            "joint_eligible": joint_eligible,
            "precheck_error": precheck_error,
        }
        if world == 1:
            # Preserve typed local failures. Only multi-rank runs need a shared
            # mismatch error; the prepare caller still owns its original error.
            if local_error is not None:
                raise local_error
            if not payload["capacity_ok"]:
                raise ZingKVCapacityError("Zing fixed KV slot capacity exceeded")
            if precheck_error:
                raise ValueError("Zing boundary preparation failed")
            return joint_eligible
        votes = [payload]
        if world > 1:
            group = getattr(zing_h3.get_sp_group(), "cpu_group", None)
            if group is None:
                if cache is not None:
                    cache.clear()
                raise ValueError(
                    "Zing delayed window rank agreement requires a CPU group"
                )
            votes = [None] * world
            dist.all_gather_object(votes, payload, group=group)
        logical = {
            k: v
            for k, v in payload.items()
            if k not in {"capacity_ok", "joint_eligible", "precheck_error"}
        }
        if (
            any(
                {k: v for k, v in vote.items() if k in logical} != logical
                for vote in votes
            )
            or not all(vote["capacity_ok"] for vote in votes)
            or any(vote["precheck_error"] for vote in votes)
        ):
            if cache is not None:
                cache.clear()
            raise ValueError("Zing delayed window rank state or capacity mismatch")
        return (
            None
            if joint_eligible is None
            else all(vote["joint_eligible"] for vote in votes)
        )

    def _precheck_initial_prefix(self, cache, pos, source, target):
        """Agree empty-cache prefix and first-step capacity before prefill."""
        blocks = self._layout_index(pos).host["block_idx"]
        rows = int(((blocks == 0) | (blocks == source) | (blocks == target)).sum())
        self._agree_sp_boundary(
            cache,
            target,
            phase="prefix_initialize",
            required_rows=rows,
            layout_signature=self._collective_layout_signature(pos, source, target),
        )

    def _begin_chunk(self, cache, pos, source, target):
        blocks = self._layout_index(pos).host["block_idx"]
        rows = int(((blocks == source) | (blocks == target)).sum())
        self._agree_sp_boundary(
            cache,
            target,
            phase="chunk_begin",
            required_rows=rows,
            layout_signature=self._collective_layout_signature(pos, source, target),
        )
        cache.begin_chunk(target)

    def _joint_eligible(self, pos, cache, source, target, audio, *, cross=False):
        if self.joint_forward == "off":
            return False
        from sglang.multimodal_gen.runtime.models.dits.zing_h3 import get_ulysses_ctx

        reason = None
        roles = self._layout_index(pos).host["row_role"]
        if get_ulysses_ctx()[0] not in (1, 2, 4):
            reason = "sp_size"
        elif self.do_cfg:
            reason = "cfg"
        elif audio is not None or bool(((roles == 5) | (roles == 11)).any()):
            reason = "audio"
        elif not self._has_block(pos, source):
            reason = "source_not_ready"
        elif not cache.raw_store or cache.local_frames != -1:
            reason = "cache_layout"
        elif (
            cross
            and cache.window_eviction_timing == "before_source"
            and not all(
                cache.window_policy.retains(i, target // 2 - 1)
                for i in [c.temporal_chunk_id for c in cache.committed]
                + [target // 2 - 2]
            )
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
        blocks = self._layout_index(pos).host["block_idx"]
        selected = (blocks == source) | (blocks == target)
        if cross:
            selected = selected | (blocks == target - 2)
        return self._agree_sp_boundary(
            cache,
            target,
            phase="joint_eligibility",
            cross=cross,
            joint_eligible=reason is None,
            required_rows=int(selected.sum()),
            layout_signature=self._collective_layout_signature(pos, source, target),
        )

    def _ready_cross_prediction(self, previous, current):
        """Join two separately packed ready chunks without moving RNG draws."""
        old_target = previous.spans[0][1]
        self._set_block_latents(
            previous.positive, old_target, previous.visual, previous.audio
        )
        final = self._select_block(
            previous.positive,
            old_target,
            phase="target",
            video_t=1,
            audio_t=None,
            final=True,
        )
        return self._resume_cross_prediction(final, current)

    def _resume_cross_packs(self, final, current):
        source, target, _, _ = current.spans[0]
        cache, pos = current.cache, current.positive
        if not self._joint_eligible(
            pos, cache, source, target, current.audio, cross=True
        ):
            return None
        self._set_block_latents(pos, target, current.visual, current.audio)
        schedules = getattr(self, "schedules_for", None)
        vs = schedules(current)[0] if schedules is not None else self.video_scheduler
        packs = [
            final,
            self._select_block(pos, source, phase="source", video_t=1, audio_t=None),
            self._select_block(
                pos,
                target,
                phase="target",
                video_t=self._host_list(vs.timesteps)[0],
                audio_t=None,
            ),
        ]
        # The previous final is absent from current's separate packed layout.
        rows = sum(p["x"].shape[0] for p in packs)
        capacity_ok = not cache.pooled or cache._length + rows <= cache.capacity
        eligible = self._agree_sp_boundary(
            cache,
            target,
            phase="ready_cross_capacity",
            cross=True,
            joint_eligible=capacity_ok,
            required_rows=rows,
            layout_signature=self._collective_layout_signature(pos, source, target),
        )
        if not eligible:
            reasons = self.last_run_metrics["joint_fallbacks"]
            reasons["pooled_scratch_capacity"] = (
                reasons.get("pooled_scratch_capacity", 0) + 1
            )
            return None
        return packs

    def _resume_cross_prediction(self, final, current):
        packs = self._resume_cross_packs(final, current)
        if packs is None:
            return None
        return self._predict_joint(
            packs,
            ("final", "source", "active"),
            current.visual.shape[2:],
            current.cache,
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
                self._begin_chunk(cache, pos, source, target)
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
