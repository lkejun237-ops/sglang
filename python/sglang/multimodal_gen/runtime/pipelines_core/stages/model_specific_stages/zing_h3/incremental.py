# SPDX-License-Identifier: Apache-2.0
"""Fixture-driven, source-ready execution with optional temporal window; not a v1 service adapter.

Initial noise must be supplied from a frozen native run. Generating it online
would change channel-major RNG assignment and needs a separate model contract.
"""

import os
from collections import deque
from copy import copy
from dataclasses import dataclass, replace
from typing import NamedTuple

import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import TemporalChunk
from sglang.multimodal_gen.runtime.realtime.zing_component_metrics import (
    ComponentTimer,
    component_timer,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import Provenance
from sglang.multimodal_gen.runtime.utils.zing_ttff_timing import measure

from .packing import H3InferenceInputs, H3RowRole
from .realtime import ZingH3RealtimeState
from .realtime_sampling import ZingH3RealtimeSampler, ZingH3RealtimeSamplingState


def _bytes(tensor):
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


# One [3, 2] native epoch. Periodic sink refreshes are anchored to these
# boundaries, so the rebuilt history always starts at a clean cycle.
_SINK_CYCLE_LATENTS = 5
_SINK_METHODS = ("copy", "rebuild")
AUTO_SINK_CYCLES_ENV = "SGLANG_ZING_AUTO_SINK_CYCLES"
AUTO_SINK_METHOD_ENV = "SGLANG_ZING_AUTO_SINK_METHOD"


@dataclass(frozen=True)
class ReadyLatentChunk:
    chunk: TemporalChunk
    model_chunk: TemporalChunk
    source: torch.Tensor | None
    noise: torch.Tensor
    audio: torch.Tensor | None
    audio_start: int
    # Immutable #111 provenance supplied by the codec/fixture bridge.
    # This worker does not manufacture pixel ranges from latent indices.
    provenance: Provenance | None

    @property
    def nbytes(self):
        return sum(_bytes(t) for t in (self.source, self.noise, self.audio))


@dataclass(frozen=True)
class ChunkResult:
    chunk: TemporalChunk
    visual: torch.Tensor
    audio: torch.Tensor | None
    provenance: Provenance | None
    prompt_event_id: int | str | None
    prompt_version: int | None = None
    sink_event_id: int | str | None = None
    sink_chunks: int = 0
    sink_chunk_ids: tuple[int, ...] = ()
    sink_config_event_id: int | str | None = None
    sink_refresh_count: int = 0
    sink_refresh_method: str | None = None
    sink_refresh_latent: int | None = None


@dataclass(frozen=True)
class AutoSinkConfig:
    """Periodic refresh schedule; a wire event supersedes the env default."""

    enabled: bool = False
    interval_cycles: int | None = None
    method: str = "copy"
    event_id: int | str | None = None
    # Chunks already owned when the schedule was configured finish unrefreshed.
    after_chunk_id: int = -1


@dataclass(frozen=True)
class PendingSinkIntent:
    """One manual or automatic sink change waiting for a later chunk boundary."""

    operation: str
    method: str
    event_id: int | str | None
    after_chunk_id: int
    is_auto: bool = False


class PromptSnapshot(NamedTuple):
    condition: dict[str, torch.Tensor]
    event_id: int | str | None
    prompt_version: int | None
    inputs: H3InferenceInputs | None = None
    reuse_prefix: bool = False
    base_event_id: int | str | None = None


class SourceReadySession:
    """One writer, bounded ready queue, absolute cursors, optional bounded KV.

    append copies ready tensors into owned storage, so small views cannot retain
    whole codec clips. A failed step closes the session and releases its state.
    The returned output transfers ownership to the caller; this object stores
    no output/provenance history. Caller must budget its output queue and
    must not issue consumed merely because this queue accepted a chunk.
    """

    def __init__(
        self,
        sampler,
        inputs,
        *,
        seed,
        max_pending_chunks,
        max_pending_bytes,
        max_pending_spans=4096,
        total_latent_frames=None,
        initial_rng_state=None,
        acquire_storage=None,
        prompt_switch_prefix_latent_frames=5,
        initial_prompt_event_id=None,
        initial_prompt_version=None,
        defer_final=False,
        prompt_joint=False,
        prefix_loader=None,
        metric_geometry=None,
    ):
        if any(
            type(n) is not int or n < 1
            for n in (max_pending_chunks, max_pending_bytes, max_pending_spans)
        ):
            raise ValueError("Ready queue budgets must be positive integers")
        # Validate finite geometry without allocating a complete session layout.
        sampler.processor.chunk_plan.chunk(0, total_latent_frames)
        sampler._agree_sp_policy()
        if (
            type(prompt_switch_prefix_latent_frames) is not int
            or prompt_switch_prefix_latent_frames != 5
        ):
            raise ValueError(
                "prompt_switch_prefix_latent_frames currently only supports 5"
            )
        self.prompt_prefix_chunk_count = len(
            sampler.processor.prompt_switch_prefix_chunks()
        )
        if defer_final and (
            prompt_switch_prefix_latent_frames != 5
            or sampler.joint_forward != "cross_chunk"
            or sampler.do_cfg
            or sampler.window_eviction_timing != "after_step0"
        ):
            raise ValueError(
                "Deferred final requires cross-chunk, five-frame history, no CFG and after-step0 eviction"
            )
        self.prompt_joint = prompt_joint
        self.prefix_loader = prefix_loader
        self._prefix_load_attempted = False
        self._cached_prefix_ready = False
        self._pending_prefix = None
        self.defer_final = defer_final
        self.sampler = sampler
        # Reuse the deployment's graph owner; explicit test/eager overrides are
        # respected, and the existing graph environment switch controls startup.
        model = sampler.generator
        if not hasattr(model, "chunk_graphs"):
            from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
                ZingChunkGraphs,
            )

            denoise_graphs = getattr(model, "denoise_graphs", None)
            model.chunk_graphs = (
                ZingChunkGraphs(max_graphs=denoise_graphs.max_graphs)
                if denoise_graphs is not None
                else None
            )
        self.plan = sampler.processor.chunk_plan
        self.inputs = copy(inputs)
        self.metric_geometry = metric_geometry or {}
        self.inputs.source_visual = self.inputs.positive = None
        if hasattr(self.inputs, "condition_media"):
            self.inputs.condition_media = None
        self.seed = seed
        self.max_pending_chunks, self.max_pending_bytes = (
            max_pending_chunks,
            max_pending_bytes,
        )
        self.total_latent_frames = total_latent_frames
        self.pending = deque()
        self.pending_bytes = 0
        self.pending_spans = 0
        self.max_pending_spans = max_pending_spans
        self.next_chunk_id = self.next_latent = self.next_audio = self.next_pixel = 0
        self.committed_chunk_id = self.committed_latent = self.committed_pixel = 0
        # A pooled session leases its slot at the first step, once the packed
        # prefix size is known; until then it holds no cache.
        self.acquire_storage = acquire_storage
        self.adaln_lease = None
        self.cache = None if acquire_storage is not None else self._new_cache(None)
        self.rng = ZingH3RealtimeState()
        if initial_rng_state is not None:
            cpu_rng, cuda_rng = initial_rng_state
            # A frozen native run must hand over the state *after* initial-noise
            # generation, not restart re-noise from the request seed.
            torch.Generator(device="cpu").set_state(cpu_rng)
            self.rng.cpu_rng = cpu_rng.clone()
            self.rng.cuda_rng = None if cuda_rng is None else cuda_rng.clone()
        self.prompt_switch_prefix_latent_frames = prompt_switch_prefix_latent_frames
        self.prompt = self._snapshot(
            inputs.positive, initial_prompt_event_id, initial_prompt_version
        )
        self.pending_prompt = None
        self.pending_sink = None
        self.sink_event_id = None
        self.auto_sink = AutoSinkConfig()
        self.sink_config_event_id = None
        self.sink_anchor_latent = 0
        self.sink_refresh_count = 0
        self.sink_refresh_method = None
        self.sink_refresh_latent = None
        # (latent_start, latent_end, clean source, clean target) for sink rebuild.
        self.recent_paired = None
        self.epoch_chunk_base = 0
        self.applied_prompt = None
        self.restore_initial_prefix = None
        self.capture_initial_prefix = None
        self.prepare_prompt_prefix = None
        self.recent_visual = None
        self.model_chunk_id = 0
        self.source_dependency_end = 0
        self.media_origin = None
        self.ended = self.closed = False
        self.draining = False
        self._drain_continuation = None
        self._drain_tensors = []
        self._drain_continuation = None
        self.audio_enabled = None
        self.device = None
        sampler._forward_counts = dict.fromkeys(
            ("prefix", "source", "denoise", "unconditional", "commit"), 0
        )
        sampler.last_run_metrics = {
            "dit_forwards": 0,
            "forward_counts": sampler._forward_counts,
        }

        if sampler.joint_forward != "off":
            sampler._forward_counts.update(source_first=0, cross_chunk=0)
            sampler.last_run_metrics.update(
                joint_forward=sampler.joint_forward, joint_fallbacks={}
            )

        self.metrics = sampler.last_run_metrics
        if self._supports_dynamic_sink():
            self._apply_env_auto_sink()

    def _new_cache(self, storage):
        sampler = self.sampler
        return ZingH3KVCache(
            len(sampler.generator.blocks),
            window_policy=sampler.window_policy,
            window_eviction_timing=sampler.window_eviction_timing,
            rope_position_mode=sampler.rope_position_mode,
            rope_max_frame_gap=sampler.rope_max_frame_gap,
            storage=storage,
        )

    @staticmethod
    def _snapshot(condition, event_id, prompt_version=None, inputs=None):
        return PromptSnapshot(
            {k: v.detach().clone() for k, v in condition.items()},
            event_id,
            prompt_version,
            inputs,
        )

    def update_prompt(
        self,
        condition,
        event_id,
        prompt_version=None,
        *,
        inputs=None,
        min_source_frame=None,
        reuse_prefix=False,
    ):
        if self.closed or self.draining:
            raise ValueError("The source-ready session is closed or draining")
        if reuse_prefix:
            if (
                condition is not self.prompt.condition
                or inputs is not None
                or self.applied_prompt is not self.prompt
            ):
                raise ValueError(
                    "GPU prefix reuse must name the current applied condition"
                )
            candidate = PromptSnapshot(
                condition, event_id, prompt_version, None, True, self.prompt.event_id
            )
        else:
            candidate = self._snapshot(condition, event_id, prompt_version, inputs)
        if min_source_frame is not None and (
            type(min_source_frame) is not int or min_source_frame < self.committed_pixel
        ):
            raise ValueError("The prompt boundary is no longer available")
        self.pending_prompt_source_frame = min_source_frame
        self.pending_prompt = candidate

    def discard_pending_prompt(self):
        """Drop an encoded candidate superseded by a newer control event."""
        if self.closed or self.draining:
            raise ValueError("The source-ready session is closed or draining")
        self.pending_prompt = None
        self.pending_prompt_source_frame = None

    def _supports_dynamic_sink(self):
        policy = self.sampler.window_policy
        return (
            self.plan.lengths == (3, 2)
            and not self.plan.split_first_latent
            and policy.window_chunks is not None
            and policy.window_chunks >= 4
            and policy.sink_chunks in (0, 2)
            and not self.audio_enabled
        )

    def _require_dynamic_sink(self):
        if not self._supports_dynamic_sink():
            raise ValueError(
                "Dynamic sinks require unsplit [3,2] video, sink 0/2 and a window of at least four chunks"
            )

    def update_sink(self, operation, event_id, method="copy"):
        """Latest-wins intent; previously queued chunks finish under the old policy."""
        if self.closed or self.draining:
            raise ValueError("The source-ready session is closed or draining")
        if operation not in {"off", "refresh"} or method not in _SINK_METHODS:
            raise ValueError(
                "Sink updates require an off or refresh operation with a copy or rebuild method"
            )
        if method == "rebuild" and operation != "refresh":
            raise ValueError("Sink rebuild only applies to a refresh")
        self._require_dynamic_sink()
        after = (
            self.pending[-1].chunk.chunk_id
            if self.pending
            else self.committed_chunk_id - 1
        )
        self.pending_sink = PendingSinkIntent(operation, method, event_id, after)

    def configure_auto_sink(
        self, enabled, *, interval_cycles=None, method="copy", event_id=None
    ):
        """Replace the periodic schedule and restart its cycle at this boundary."""
        if self.closed or self.draining:
            raise ValueError("The source-ready session is closed or draining")
        if type(enabled) is not bool or method not in _SINK_METHODS:
            raise ValueError(
                "Auto sink requires a boolean switch and a copy or rebuild method"
            )
        if enabled:
            if type(interval_cycles) is not int or interval_cycles < 1:
                raise ValueError("Auto sink requires a positive cycle interval")
        elif interval_cycles is not None:
            raise ValueError("Disabling auto sink must not carry a cycle interval")
        self._require_dynamic_sink()
        after = (
            self.pending[-1].chunk.chunk_id
            if self.pending
            else self.committed_chunk_id - 1
        )
        self.auto_sink = AutoSinkConfig(
            enabled=enabled,
            interval_cycles=interval_cycles if enabled else None,
            method=method,
            event_id=event_id,
            after_chunk_id=after,
        )
        self.sink_config_event_id = event_id
        self.sink_anchor_latent = self.committed_latent

    def _apply_env_auto_sink(self):
        """Seed the session default so a client can enable refresh out of band."""
        method = (os.environ.get(AUTO_SINK_METHOD_ENV) or "copy").strip() or "copy"
        if method not in _SINK_METHODS:
            raise ValueError("Auto sink method must be copy or rebuild")
        cycles = (os.environ.get(AUTO_SINK_CYCLES_ENV) or "").strip()
        if cycles in {"", "0"}:
            return
        if not cycles.isdigit() or int(cycles) < 1:
            raise ValueError("Auto sink cycles must be a positive integer")
        self.auto_sink = AutoSinkConfig(
            enabled=True,
            interval_cycles=int(cycles),
            method=method,
            event_id=None,
        )
        self.sink_anchor_latent = 0

    @property
    def _auto_sink_due(self):
        # S0 pauses the schedule: without a two-chunk sink there is no anchor
        # to move, and a cold session has no cache yet.
        interval = self.auto_sink.interval_cycles
        return (
            self.auto_sink.enabled
            and interval is not None
            and self.cache is not None
            and self.cache.window_policy.sink_chunks == 2
            and self.committed_latent >= _SINK_CYCLE_LATENTS
            and self.committed_chunk_id > self.auto_sink.after_chunk_id
            and self.committed_latent - self.sink_anchor_latent
            >= _SINK_CYCLE_LATENTS * interval
        )

    def _sink_intent(self, ready):
        """Sink change applied at this chunk's start, or None to keep the window."""
        pending = self.pending_sink
        if pending is not None and ready.chunk.chunk_id > pending.after_chunk_id:
            # Rebuild waits for a full cycle boundary; copy and off apply here.
            return None if pending.method == "rebuild" else pending
        if (
            self._auto_sink_due
            and self.auto_sink.method == "copy"
            and ready.chunk.start % _SINK_CYCLE_LATENTS == 0
        ):
            return PendingSinkIntent(
                "refresh", "copy", None, ready.chunk.chunk_id - 1, is_auto=True
            )
        return None

    def _apply_pending_sink(self, ready, current):
        intent = self._sink_intent(ready)
        if intent is None:
            return
        sampler, cache = self.sampler, self.cache
        sampler._agree_sp_boundary(
            cache,
            ready.model_chunk.target_block,
            phase="sink_update",
            ready_route=intent,
        )
        if cache.active is not None:
            # The latest delivered picture may still lack its clean final KV.
            final = sampler._deferred_final(cache, self.recent_visual, current.positive)
            sampler._forward(final, cache, "final")
        cache.update_sink(intent.operation, current_id=ready.model_chunk.chunk_id)
        if not intent.is_auto:
            self.sink_event_id = intent.event_id
            self.pending_sink = None
        if intent.operation == "refresh":
            self.sink_anchor_latent = ready.chunk.start
            self.sink_refresh_count += 1
            self.sink_refresh_method = "copy"
            self.sink_refresh_latent = ready.chunk.start

    @property
    def prompt_switch_ready(self):
        # Repacking the last five latents as 0..4 must preserve codec phase.
        return (
            self.recent_visual is not None
            and self.recent_visual.shape[2] == self.prompt_switch_prefix_latent_frames
            and self.committed_latent % 5 == 0
        )

    @property
    def sink_rebuild_ready(self):
        """A paired five-latent clean window ending exactly at this boundary."""
        paired = self.recent_paired
        window = self.prompt_switch_prefix_latent_frames
        return (
            self.cache is not None
            and self.cache.prefix is not None
            and self.applied_prompt is self.prompt
            and paired is not None
            and paired[0] == self.committed_latent - window
            and paired[1] == self.committed_latent
            and paired[2].shape[2] == window
            and paired[3].shape[2] == window
            and self.committed_latent >= window
            and self.committed_latent % window == 0
        )

    def _sink_rebuild_intent(self):
        """Rebuild intent that may run once the next chunk starts on a cycle boundary."""
        intent = self.pending_sink
        if intent is not None:
            if (
                intent.method == "rebuild"
                and self.committed_chunk_id > intent.after_chunk_id
            ):
                return intent
            return None
        if self._auto_sink_due and self.auto_sink.method == "rebuild":
            return PendingSinkIntent(
                "refresh", "rebuild", None, self.committed_chunk_id, is_auto=True
            )
        return None

    def _remember_paired(self, ready, visual):
        """Keep the tail clean source+target latents a sink rebuild replays."""
        if not self._supports_dynamic_sink() or ready.source is None:
            self.recent_paired = None
            return
        count = ready.chunk.end - ready.chunk.start
        source = ready.source.detach().float().clone()
        target = visual.detach().float().clone()
        if source.shape[2] != count or target.shape[2] != count:
            self.recent_paired = None
            return
        previous = self.recent_paired
        if previous is None or previous[1] != ready.chunk.start:
            combined_source, combined_target = source, target
        else:
            combined_source = torch.cat((previous[2], source), dim=2)
            combined_target = torch.cat((previous[3], target), dim=2)
        keep = min(self.prompt_switch_prefix_latent_frames, combined_source.shape[2])
        self.recent_paired = (
            ready.chunk.end - keep,
            ready.chunk.end,
            combined_source[:, :, -keep:].clone(),
            combined_target[:, :, -keep:].clone(),
        )

    def _activate_pending_sink_rebuild(self, ready):
        """Replay the paired window as local epoch blocks 1..4 before packing."""
        window = self.prompt_switch_prefix_latent_frames
        if (
            ready.chunk.start % _SINK_CYCLE_LATENTS
            or ready.model_chunk.chunk_id != self.prompt_prefix_chunk_count
            or ready.model_chunk.start != window
            or not self.sink_rebuild_ready
        ):
            return
        # A prompt switch already replayed its own epoch and absorbed the intent.
        intent = self._sink_rebuild_intent()
        if intent is None:
            return
        error = None
        try:
            _, _, source_prefix, target_prefix = self.recent_paired
            layout, _ = self.sampler.processor.pack_sink_rebuild_prefix(
                self.inputs, self.prompt.condition, source_prefix, target_prefix
            )
            host = self.sampler._layout_index(layout).host
            prefix = self.cache.validate_prefix_reset()
            block0_pos = host["position_ids"][host["block_idx"] == 0]
            if (
                prefix is None
                or prefix.host_positions is None
                or not torch.equal(prefix.host_positions, block0_pos)
            ):
                raise ValueError(
                    "Resident prefix geometry differs from sink rebuild layout"
                )
        except Exception as exc:
            error = exc
        self.sampler._agree_sp_boundary(
            self.cache,
            0,
            phase="sink_rebuild_prepare",
            ready_route=(intent.is_auto, intent.event_id),
            precheck_error=error is not None,
            local_error=error,
        )
        if error is not None:
            raise error
        policy = self.cache.window_policy
        self.cache.window_policy = replace(
            policy, sink_chunks=2, sink_start=0, min_chunk_id=0, active_sinks=None
        )
        origin, latent_end = self.sampler.rebuild_sink_prefix(
            self.inputs,
            self.prompt.condition,
            source_prefix,
            target_prefix,
            self.cache,
        )
        if latent_end != window:
            raise ValueError("Sink rebuild prefix does not meet the next model block")
        self.media_origin = origin
        self.model_chunk_id = self.prompt_prefix_chunk_count
        self.epoch_chunk_base = self.committed_chunk_id - self.prompt_prefix_chunk_count
        self._pending_prefix = None
        self._cached_prefix_ready = False
        if not intent.is_auto:
            self.sink_event_id = intent.event_id
            self.pending_sink = None
        self.sink_anchor_latent = ready.chunk.start
        self.sink_refresh_count += 1
        self.sink_refresh_method = "rebuild"
        self.sink_refresh_latent = ready.chunk.start

    def _auto_sink_blocks(self, tail_latent):
        """True while queued work already reaches the next auto refresh threshold."""
        interval = self.auto_sink.interval_cycles
        if not self.auto_sink.enabled or interval is None:
            return False
        # A cold session has no cache yet; its first block still runs under the
        # deployment policy, which is what the refresh will later restore.
        sink_chunks = (
            self.sampler.window_policy.sink_chunks
            if self.cache is None
            else self.cache.window_policy.sink_chunks
        )
        return (
            sink_chunks == 2
            and tail_latent >= _SINK_CYCLE_LATENTS
            and tail_latent - self.sink_anchor_latent >= _SINK_CYCLE_LATENTS * interval
        )

    def _next_model_chunk(self):
        boundary = getattr(self, "pending_prompt_source_frame", None) or 0
        if self.pending:
            # Drain queued work before planning the first block of a new epoch.
            if (
                self.pending_prompt is not None
                or self.pending_sink is not None
                or self._auto_sink_blocks(self.pending[-1].chunk.end)
            ):
                return None
            previous = self.pending[-1].model_chunk
            return self.plan.chunk(previous.chunk_id + 1)
        if (
            self.pending_prompt is not None
            and self.prompt_switch_ready
            and self.committed_pixel >= boundary
        ):
            return self.plan.chunk(self.prompt_prefix_chunk_count)
        # A due rebuild replans this cycle as the epoch chunk(2); a half cycle
        # keeps producing native chunks until the pair window closes.
        if self._sink_rebuild_intent() is not None and self.sink_rebuild_ready:
            return self.plan.chunk(self.prompt_prefix_chunk_count)
        return self.plan.chunk(self.model_chunk_id)

    @property
    def next_chunk_size(self):
        """Latents required for the next native-phase model block.

        A ready chunk already owned by this session must be consumed before a
        prompt update can reset the phase.  This keeps cross-chunk lookahead
        from mixing a pre-switch boundary with the new local epoch.
        """
        chunk = self._next_model_chunk()
        if chunk is None:
            return None
        size = chunk.end - chunk.start
        if self.total_latent_frames is not None:
            size = min(size, self.total_latent_frames - self.next_latent)
        return size if size > 0 else None

    def _activate_pending_prompt(self, ready, *, defer_prefix=False):
        if (
            self.pending_prompt is None
            or not self.prompt_switch_ready
            or self.committed_pixel
            < (getattr(self, "pending_prompt_source_frame", None) or 0)
            or ready.chunk.start % 5 != 0
            or ready.model_chunk.chunk_id != self.prompt_prefix_chunk_count
            or ready.model_chunk.start != 5
        ):
            return
        self._rebuild_pending_prompt(defer_prefix=defer_prefix)

    def activate_prompt_at_source_boundary(self, boundary, *, defer_prefix=False):
        """Prepare an admitted condition without needing a future source chunk.

        The source adapter has fenced admission at a full codec clip. Only
        already committed history is used; global source and RNG cursors stay
        unchanged. Negotiated joint execution can retain frozen packs until new
        source arrives; that receipt means source-ready, not completed KV.
        A failed destructive rebuild closes the session.
        """
        if (
            self.closed
            or self.ended
            or self.draining
            or self.pending
            or type(boundary) is not int
            or boundary < 1
            or self.committed_pixel != boundary
            or self.next_pixel != boundary
            or self.committed_latent != self.next_latent
            or self.next_latent % 5 != 0
            or self.pending_prompt is None
            or not self.prompt_switch_ready
        ):
            raise ValueError("The condition switch boundary is not ready")
        try:
            self._rebuild_pending_prompt(
                defer_prefix=defer_prefix and self.prompt_joint
            )
        except BaseException:
            self.dispose()
            raise
        return dict(
            event_id=self.prompt.event_id,
            prompt_version=self.prompt.prompt_version,
            source_frame_index=boundary,
            **({"kv_deferred": True} if self._pending_prefix is not None else {}),
        )

    def _rebuild_pending_prompt(self, *, defer_prefix=False):
        if self.audio_enabled:
            raise ValueError("Prompt switching currently supports video-only sessions")
        candidate = self.pending_prompt
        policy = self.cache.window_policy
        count = policy.sink_chunks
        if (
            self.pending_sink is not None
            and self.committed_chunk_id > self.pending_sink.after_chunk_id
        ):
            intent = self.pending_sink
            self.sink_event_id = intent.event_id
            count = 2 if intent.operation == "refresh" else 0
            self.pending_sink = None
        # Chunk identities restart with the prompt epoch; only the on/off state survives.
        self.cache.window_policy = replace(
            policy, sink_chunks=count, sink_start=0, min_chunk_id=0
        )
        inputs = self.inputs if candidate.inputs is None else candidate.inputs
        restore_prefix = (
            None
            if self.prepare_prompt_prefix is None
            else self.prepare_prompt_prefix(candidate)
        )
        reuse = candidate.reuse_prefix
        error = None
        if reuse:
            try:
                if (
                    candidate.base_event_id != self.prompt.event_id
                    or candidate.condition is not self.prompt.condition
                ):
                    raise ValueError(
                        "Resident prefix condition changed before application"
                    )
                packs, _, _ = self.sampler.pack_prompt_prefix(
                    inputs, candidate.condition, self.recent_visual
                )
                host = self.sampler._layout_index(packs[0]).host
                prefix = self.cache.validate_prefix_reset()
                if (
                    prefix is None
                    or prefix.host_positions is None
                    or not torch.equal(prefix.host_positions, host["position_ids"])
                ):
                    raise ValueError(
                        "Resident prefix geometry differs from the rebuilt layout"
                    )
            except Exception as exc:
                error = exc
        self.sampler._agree_sp_boundary(
            self.cache,
            0,
            phase="resident_prefix_prepare",
            ready_route=(reuse,),
            precheck_error=error is not None,
            local_error=error,
        )
        if error is not None:
            raise error
        if reuse:
            with measure(
                getattr(self, "ttff_timing", None), "history_kv_rebuild", cuda=True
            ):
                origin, latent_end = self.sampler.rebuild_prompt_prefix(
                    inputs,
                    candidate.condition,
                    self.recent_visual,
                    self.cache,
                    reuse_prefix=True,
                )
            self.metrics["gpu_prefix_reuses"] = (
                self.metrics.get("gpu_prefix_reuses", 0) + 1
            )
        elif (
            defer_prefix
            and (self.prefix_loader is not None or self.prompt_joint)
            and (restore_prefix is None or self.sampler.joint_forward == "cross_chunk")
        ):
            prepared, error = None, None
            try:
                prepared = self.sampler.pack_prompt_prefix(
                    inputs, candidate.condition, self.recent_visual
                )
            except Exception as exc:
                error = exc
            self.sampler._agree_sp_boundary(
                self.cache,
                8,
                phase="prompt_prefix_prepare",
                precheck_error=error is not None,
                local_error=error,
            )
            if error is not None:
                raise error
            packs, origin, latent_end = prepared
            self._cached_prefix_ready = self._load_prefix(
                inputs, candidate, packs[0], restore_prefix=restore_prefix
            )
            if self._cached_prefix_ready:
                self._pending_prefix = packs[1:]
            elif self.prompt_joint:
                self._pending_prefix = packs
                self.cache.clear()
            else:
                with measure(
                    getattr(self, "ttff_timing", None), "history_kv_rebuild", cuda=True
                ):
                    origin, latent_end = self.sampler.rebuild_prompt_prefix(
                        inputs, candidate.condition, self.recent_visual, self.cache
                    )
        else:
            with measure(
                getattr(self, "ttff_timing", None), "history_kv_rebuild", cuda=True
            ):
                origin, latent_end = self.sampler.rebuild_prompt_prefix(
                    inputs,
                    candidate.condition,
                    self.recent_visual,
                    self.cache,
                    **({"restore_prefix": restore_prefix} if restore_prefix else {}),
                )
        if latent_end != 5:
            raise ValueError("Prompt-switch prefix does not meet the next model block")
        if candidate.inputs is not None:
            self.inputs = copy(candidate.inputs)
            self.inputs.source_visual = self.inputs.positive = None
            self.inputs.condition_media = None
        self.prompt = self.applied_prompt = candidate
        if self.pending_prompt is candidate:
            self.pending_prompt = None
            self.pending_prompt_source_frame = None
        self.media_origin = origin
        self.model_chunk_id = self.prompt_prefix_chunk_count
        self.epoch_chunk_base = self.committed_chunk_id - self.prompt_prefix_chunk_count
        # The new epoch's first five latents are a fresh refresh anchor, so an
        # auto refresh already due at this boundary is absorbed by the rebuild.
        self.sink_anchor_latent = self.committed_latent

    def _load_prefix(self, inputs, prompt, prefix, *, restore_prefix=None):
        """Consume an external writer's prefix at a quiescent session boundary.

        False must leave the cache untouched. True publishes a complete prefix
        in this leased slot, with no dynamic history. The external writer owns
        content identity and any cross-stream dependency; we never fetch KV.
        """
        if self.prefix_loader is None and restore_prefix is None:
            return False
        error, hit = None, False
        try:
            if restore_prefix is not None:
                self.cache.clear()
                restore_prefix(self.cache)
                hit = True
            else:
                hit = self.prefix_loader(self.cache, inputs, prompt, prefix)
            if type(hit) is not bool:
                raise ValueError("Prefix loader must return a boolean")
            if hit:
                from sglang.multimodal_gen.runtime.models.dits.zing_h3 import (
                    get_ulysses_ctx,
                )

                cache = self.cache
                host = self.sampler._layout_index(prefix).host
                chunk = cache.prefix
                if (
                    not cache.pooled
                    or not cache.raw_store
                    or chunk is None
                    or cache.committed
                    or cache.active is not None
                    or cache.pending_source is not None
                    or cache.last_block != 0
                    or chunk.offset != 0
                    or cache._length != host["block_idx"].numel()
                    or chunk.host_positions is None
                    or not torch.equal(chunk.host_positions, host["position_ids"])
                    or chunk.block != 0
                ):
                    raise ValueError(
                        "Prefix loader must publish matching prefix-only pooled KV"
                    )
                if (
                    self.sampler.joint_forward != "cross_chunk"
                    or self.sampler.do_cfg
                    or self.audio_enabled
                    or self.sampler.solver != "few_step"
                    or self.sampler.window_eviction_timing != "after_step0"
                    or get_ulysses_ctx()[0] not in (1, 2, 4)
                ):
                    raise ValueError(
                        "Cached prefix requires video-only cross-chunk joint execution"
                    )
        except Exception as exc:
            error = exc
        self.sampler._agree_sp_boundary(
            self.cache,
            0,
            phase="cached_prefix_load",
            ready_route=(hit,),
            precheck_error=error is not None,
            local_error=error,
        )
        if error is not None:
            raise error
        key = "cached_prefix_hits" if hit else "cached_prefix_misses"
        self.metrics[key] = self.metrics.get(key, 0) + 1
        return hit

    def _prefix_prediction(self, current, graph_eligible):
        """Consume the session-owned prefix at the first target of the new epoch."""
        if self._pending_prefix is None:
            return None
        from sglang.multimodal_gen.runtime.models.dits.zing_h3 import get_ulysses_ctx
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
            ZingChunkGraphs,
        )
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_joint import (
            prefix_joint_phases,
        )
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_prefix_graph import (
            PrefixJointGraphs,
        )

        sampler, model, cache = self.sampler, self.sampler.generator, self.cache
        prefix_target = self.plan.chunk(self.prompt_prefix_chunk_count).target_block
        size, _ = get_ulysses_ctx()
        heads = int(
            getattr(getattr(model, "arch", None), "num_attention_heads", 0) or 0
        )
        supported = (
            cache.raw_store
            and cache.local_frames == -1
            and sampler.joint_forward == "cross_chunk"
            and not sampler.do_cfg
            and sampler.solver == "few_step"
            and size >= 1
            and (heads <= 0 or heads % size == 0)
            and sampler.window_eviction_timing == "after_step0"
            and current.audio is None
            and current.spans[0][:2] == (prefix_target - 1, prefix_target)
            and (size == 1 or model.experimental_stable_shapes)
        )
        sampler._agree_sp_boundary(
            cache,
            prefix_target,
            phase="prompt_prefix_route",
            ready_route=(supported, graph_eligible),
        )
        packs, self._pending_prefix = self._pending_prefix, None
        if not supported:
            if self._cached_prefix_ready and self.plan.split_first_latent:
                raise ValueError(
                    "Cached prompt switch requires supported joint execution"
                )
            # Unsupported execution modes retain an already restored condition KV.
            if not self._cached_prefix_ready:
                sampler._forward(packs[0], cache, "prefix")
                packs = packs[1:]
            for packed in packs:
                host = sampler._layout_index(packed).host
                cache.update(
                    packed["position_ids"],
                    packed["block_idx"],
                    "active",
                    int(host["position_ids"][:, 0].unique().numel()),
                    validation=(host["position_ids"], host["block_idx"]),
                )
                sampler._forward(packed, cache, "final")
            reasons = self.metrics.setdefault("prompt_joint_fallbacks", {})
            if self.plan.split_first_latent:
                reasons["unsupported_layout"] = reasons.get("unsupported_layout", 0) + 1
            return None
        error = None
        try:
            source, target, *_ = current.spans[0]
            vs, _ = ZingChunkGraphs._schedules(sampler, current)
            packs += [
                sampler._select_block(
                    current.positive, source, phase="source", video_t=1, audio_t=None
                ),
                sampler._select_block(
                    current.positive,
                    target,
                    phase="target",
                    video_t=sampler._host_list(vs.timesteps)[0],
                    audio_t=None,
                ),
            ]
        except Exception as exc:
            error = exc
        sampler._agree_sp_boundary(
            cache,
            prefix_target,
            phase="prompt_joint_prepare",
            precheck_error=error is not None,
            local_error=error,
        )
        if error is not None:
            raise error
        if graph_eligible:
            graphs = getattr(model, "prefix_graphs", None)
            if graphs is None:
                graphs = model.prefix_graphs = PrefixJointGraphs(
                    capture_enabled=model.chunk_graphs.capture_enabled
                )
            before = graphs.routes.copy()
            video, audio = graphs.run(sampler, packs, cache)
            routes = self.metrics.setdefault("prompt_joint_graph_routes", {})
            for (kind, route), count in (graphs.routes - before).items():
                key = f"{kind}.{route}"
                routes[key] = routes.get(key, 0) + count
            sampler._forward_counts["prompt_joint"] = (
                sampler._forward_counts.get("prompt_joint", 0) + 1
            )
        else:
            prediction = sampler._predict_joint(
                packs,
                prefix_joint_phases(
                    self.prompt_prefix_chunk_count, cached=self._cached_prefix_ready
                ),
                current.visual.shape[2:],
                cache,
            )
            if prediction is None:
                raise RuntimeError(
                    "Prompt joint ranks rejected the prepared transaction"
                )
            return prediction
        return sampler.processor.unpatchify_visual(
            video, 1, current.visual.shape[2:]
        ), None

    def _remember_visual(self, visual):
        limit = self.prompt_switch_prefix_latent_frames
        value = visual.detach()
        if self.recent_visual is not None:
            value = torch.cat((self.recent_visual, value), dim=2)
        self.recent_visual = value[:, :, -limit:].clone()

    def append(
        self, source, noise, *, provenance, audio=None, audio_start=0, final=False
    ):
        if source is None or not isinstance(provenance, Provenance):
            raise ValueError("Source-ready chunks require source and provenance")
        return self._append(
            source,
            noise,
            provenance=provenance,
            audio=audio,
            audio_start=audio_start,
            final=final,
        )

    def append_block(self, source, noise, *, final=False):
        """Queue a block from an already encoded request, without source transport."""
        return self._append(source, noise, provenance=None, final=final)

    def _append(
        self, source, noise, *, provenance, audio=None, audio_start=0, final=False
    ):
        if self.closed or self.ended or self.draining:
            raise ValueError("The source-ready session no longer accepts chunks")
        packer = self.sampler.processor
        if packer.chunk_plan != self.plan:
            raise ValueError("The checkpoint chunk plan changed within the session")
        packer._validate_visual(noise)
        if self.defer_final and (audio is not None or noise.shape[2] > 5):
            raise ValueError(
                "Deferred final requires video-only chunks fitting five-frame history"
            )
        if not noise.is_floating_point():
            raise ValueError("Ready noise must be floating-point latents")
        if source is not None:
            packer._validate_visual(source)
            if not source.is_floating_point():
                raise ValueError("Ready source must be floating-point latents")
            if source.shape[2] != noise.shape[2] or source.device != noise.device:
                raise ValueError(
                    "Ready source and target must share temporal length and device"
                )
        if tuple(noise.shape[3:]) != tuple(self.inputs.visual_shape[1:]):
            raise ValueError("Ready target spatial geometry changed")
        if self.device is not None and noise.device != self.device:
            raise ValueError("Ready chunk device changed within the session")
        if (
            type(final) is not bool
            or type(audio_start) is not int
            or audio_start != self.next_audio
        ):
            raise ValueError("Ready audio cursor or final flag is invalid")
        if provenance is not None and provenance.pixels.start != self.next_pixel:
            raise ValueError("Ready source provenance must be continuous")
        if self.audio_enabled is not None and (audio is not None) != self.audio_enabled:
            raise ValueError("Ready target audio mode changed within the session")
        total = self.total_latent_frames
        if final:
            end = self.next_latent + noise.shape[2]
            if total is not None and end != total:
                raise ValueError("Final chunk does not cover the finite latent target")
            total = end
        model_chunk = self._next_model_chunk()
        if model_chunk is None:
            raise ValueError("Ready data cannot cross a pending prompt phase boundary")
        expected = model_chunk.end - model_chunk.start
        count = noise.shape[2]
        if count != expected and not (final and 0 < count < expected):
            raise ValueError("Ready data does not cover the next checkpoint chunk")
        chunk = TemporalChunk(
            self.next_chunk_id, self.next_latent, self.next_latent + count
        )
        model_chunk = TemporalChunk(
            model_chunk.chunk_id, model_chunk.start, model_chunk.start + count
        )
        if audio is not None:
            packer._validate_audio(audio)
            if audio.device != noise.device:
                raise ValueError("Ready video and audio must share a device")
            # Preserve native audio grouping; only the actual EOS absorbs a tail.
            end_t = packer._video_t_span(chunk.end)
            expected = len(
                [
                    i
                    for i in range(audio_start, audio_start + audio.shape[2])
                    if i < end_t
                ]
            )
            if not final and (
                expected != audio.shape[2] or audio_start + audio.shape[2] < end_t
            ):
                raise ValueError("Ready audio must cover the native temporal chunk")
        spans = 0 if provenance is None else len(provenance.source_alignment)
        nbytes = sum(_bytes(t) for t in (source, noise, audio))
        if (
            len(self.pending) >= self.max_pending_chunks
            or self.pending_bytes + nbytes > self.max_pending_bytes
            or self.pending_spans + spans > self.max_pending_spans
        ):
            raise ValueError("The source-ready queue is full")
        owned = ReadyLatentChunk(
            chunk,
            model_chunk,
            None if source is None else source.detach().clone(),
            noise.detach().clone(),
            None if audio is None else audio.detach().clone(),
            audio_start,
            provenance,
        )
        self.pending.append(owned)
        self.pending_bytes += nbytes
        self.pending_spans += spans
        self.next_chunk_id += 1
        self.next_latent = chunk.end
        if provenance is not None:
            self.next_pixel = provenance.pixels.end
        self.next_audio += 0 if audio is None else audio.shape[2]
        self.audio_enabled = audio is not None
        self.device = noise.device
        if final:
            self.total_latent_frames = total
            self.ended = True

    def finish(self):
        if self.closed or self.ended or self.draining:
            raise ValueError("The source-ready session has already ended")
        if (
            self.total_latent_frames is not None
            and self.next_latent != self.total_latent_frames
        ):
            raise ValueError("Source ended before the finite latent target")
        self.ended = True
        self.total_latent_frames = self.next_latent
        if not self.pending:
            self.dispose()

    @property
    def complete(self):
        return (
            self.ended
            and not self.pending
            and self.committed_chunk_id == self.next_chunk_id
        )

    def _agree_ready_route(self, entry, *, first_prediction=False):
        # An empty valid caller must participate before its peers prepare or
        # enter a model. Use the committed cursor, never an optional queue head.
        self.sampler._agree_sp_boundary(
            self.cache,
            2 * (self.committed_chunk_id + 1),
            phase="source_ready_route",
            ready_route=(
                entry,
                bool(self.pending),
                len(self.pending) > 1,
                first_prediction,
                self.ended,
                self.total_latent_frames,
                self.defer_final,
                self.cache is not None and self.cache.active is not None,
                self.prompt_joint,
                self.prefix_loader is not None,
                (self.pending_sink, self.auto_sink, self._auto_sink_due),
            ),
        )

    @torch.no_grad()
    def step(self):
        if self.closed or self.draining:
            raise ValueError("The source-ready session is closed or draining")
        if self.defer_final:
            raise ValueError("Deferred final requires drain_ready execution")
        if self._drain_continuation is not None:
            raise ValueError("Prepared continuation requires drain_ready execution")
        try:
            self._agree_ready_route("step")
            if not self.pending:
                return None  # Wait for source. Never repeat frames or advance RNG.
            with component_timer("dit", **self.metric_geometry) as timer:
                result = self._step()
                if result.provenance is not None:
                    timer.frames = result.provenance.pixels.count
                return result
        except BaseException:
            self.dispose()
            raise

    def _prepare(self, ready, prompt, media_origin, previous=None):
        sampler, packer = self.sampler, self.sampler.processor
        sampler.device = ready.noise.device
        model_chunk = ready.model_chunk
        positive = negative = visual = audio = None
        prepare_error = None
        try:
            if packer.chunk_plan != self.plan:
                raise ValueError("The checkpoint chunk plan changed within the session")
            if (
                ready.noise.is_cuda
                and self.rng.cpu_rng is not None
                and self.rng.cuda_rng is None
            ):
                raise ValueError(
                    "Frozen CUDA sampling requires the initial CUDA RNG state"
                )
            inputs = copy(self.inputs)
            inputs.positive, inputs.source_visual = prompt.condition, ready.source
            visual = ready.noise.float().clone()
            audio = None if ready.audio is None else ready.audio.float().clone()
            kwargs = dict(
                chunk=model_chunk,
                audio_start=ready.audio_start,
                media_origin=media_origin,
            )
            positive = packer.pack_chunk(
                inputs, prompt.condition, visual, audio, **kwargs
            )
            negative = (
                packer.pack_chunk(inputs, None, visual, audio, **kwargs)
                if sampler.do_cfg
                else None
            )
        except BaseException as error:
            # Validate and pack only a candidate; no prefix/model or slot has
            # been touched. Peers must still join its prepare-status vote.
            prepare_error = error
        if sampler._delayed_policy_requires_agreement:
            previous_layout = (
                None
                if previous is None
                else sampler._collective_layout_signature(
                    previous.positive, *previous.spans[0][:2]
                )
            )
            sampler._agree_sp_boundary(
                self.cache,
                model_chunk.target_block,
                phase="source_ready_prepare",
                needs_prefix_update=prompt is not self.applied_prompt,
                precheck_error=prepare_error is not None,
                local_error=prepare_error,
                layout_signature=(
                    "source_ready_prepare",
                    ready.model_chunk.chunk_id,
                    ready.model_chunk.start,
                    ready.model_chunk.end,
                    ready.audio_start,
                    previous_layout,
                ),
            )
        if prepare_error is not None:
            raise prepare_error
        if isinstance(sampler, ZingH3RealtimeSampler):
            sampler.prepare_schedules(sampler.device)
        else:
            sampler.video_scheduler.to(sampler.device)
            sampler.audio_scheduler.to(sampler.device)
        spans = [
            (model_chunk.source_block, model_chunk.target_block, 0, visual.shape[2])
        ]
        lease = sampler._admit_adaln(positive, negative, spans, audio)
        if self.adaln_lease is not None:
            self.adaln_lease.close()
        # This session owns the lease, just as it owns the shared KV cache.
        # Outer step/drain cleanup releases it if pool acquisition or prefix fails.
        self.adaln_lease = lease
        if self.cache is None:
            self.cache = self._new_cache(self.acquire_storage(positive, ready))
        origin = media_origin
        if origin is None:
            target = positive["row_role"] == int(H3RowRole.TARGET_VISUAL)
            origin = float(positive["position_ids"][target][0, 0])
        current = ZingH3RealtimeSamplingState(
            visual,
            audio,
            positive,
            negative,
            spans,
            self.cache,
            generation_block_index=int(ready.chunk.chunk_id),
        )
        if self.prefix_loader is not None and not self._prefix_load_attempted:
            self._prefix_load_attempted = True
            prefix = sampler._select_block(
                positive, 0, phase="prefix", video_t=1, audio_t=1
            )
            self._cached_prefix_ready = self._load_prefix(self.inputs, prompt, prefix)
        return current, origin

    def _initial_prediction(self, current, prompt):
        """Join an uncached initial condition with source and target step0."""
        from sglang.multimodal_gen.runtime.models.dits.zing_h3 import get_ulysses_ctx
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_joint import (
            INITIAL_PREFIX_PHASES,
        )

        sampler, cache = self.sampler, self.cache
        source, target, *_ = current.spans[0]
        if (
            not sampler.initial_prefix_joint
            or prompt is self.applied_prompt
            or self._cached_prefix_ready
            or self.restore_initial_prefix is not None
            or self._pending_prefix is not None
            or cache.prefix is not None
            or cache.last_block != 0
            or cache.active is not None
            or (source, target) != (1, 2)
            or sampler.solver != "few_step"
            or not sampler._has_block(current.positive, 0)
            or (
                get_ulysses_ctx()[0] > 1
                and not sampler.generator.experimental_stable_shapes
            )
        ):
            return None
        if not sampler._joint_eligible(
            current.positive, cache, source, target, current.audio
        ):
            return None
        sampler._precheck_initial_prefix(cache, current.positive, source, target)
        packs = error = None
        try:
            vs, _ = (
                sampler.schedules_for(current)
                if hasattr(sampler, "schedules_for")
                else (sampler.video_scheduler, sampler.audio_scheduler)
            )
            packs = [
                sampler._select_block(
                    current.positive, 0, phase="prefix", video_t=1, audio_t=1
                ),
                sampler._select_block(
                    current.positive, source, phase="source", video_t=1, audio_t=None
                ),
                sampler._select_block(
                    current.positive,
                    target,
                    phase="target",
                    video_t=sampler._host_list(vs.timesteps)[0],
                    audio_t=None,
                ),
            ]
        except Exception as failure:
            error = failure
        sampler._agree_sp_boundary(
            cache,
            target,
            phase="initial_joint_prepare",
            precheck_error=error is not None,
            local_error=error,
        )
        if error is not None:
            raise error
        return sampler._predict_joint(
            packs, INITIAL_PREFIX_PHASES, current.visual.shape[2:], cache
        )

    def _apply_prefix(self, current, prompt):
        if prompt is self.applied_prompt:
            return
        if self._cached_prefix_ready:
            return
        # Later prompt changes rebuild the full rolling prefix before _prepare.
        # This path only initializes an empty session cache.
        if self.cache.prefix is not None or self.cache.last_block:
            raise ValueError("Prompt changes require a rolling prefix rebuild")
        sampler, positive = self.sampler, current.positive
        source, target, _, _ = current.spans[0]
        sampler._precheck_initial_prefix(self.cache, positive, source, target)
        if self.restore_initial_prefix is not None:
            self.restore_initial_prefix(self.cache)
            self.restore_initial_prefix = None
            return
        if bool((positive["block_idx"] == 0).any()):
            sampler._forward(
                sampler._select_block(
                    positive, 0, phase="prefix", video_t=1, audio_t=1
                ),
                self.cache,
                "prefix",
            )
        if self.capture_initial_prefix is not None:
            self.capture_initial_prefix(self.cache.prefix)
            self.capture_initial_prefix = None

    def _finish(self, ready, current, prompt, origin):
        # A joint miss publishes the new prefix in its first target forward.
        if self.capture_initial_prefix is not None:
            self.capture_initial_prefix(self.cache.prefix)
            self.capture_initial_prefix = None
        dependency = self.source_dependency_end
        provenance = ready.provenance
        if provenance is not None:
            dependency = max(dependency, provenance.source_dependency_end)
            provenance = Provenance(
                provenance.pixels, provenance.source_alignment, dependency
            )
        result = ChunkResult(
            ready.chunk,
            current.visual,
            current.audio,
            provenance,
            prompt.event_id,
            prompt.prompt_version,
            self.sink_event_id,
            self.cache.window_policy.sink_chunks,
            tuple(
                self.epoch_chunk_base + i
                for i in range(
                    self.cache.window_policy.sink_start,
                    min(
                        self.cache.window_policy.sink_start
                        + self.cache.window_policy.sink_chunks,
                        ready.model_chunk.chunk_id + 1,
                    ),
                )
            ),
            self.sink_config_event_id,
            self.sink_refresh_count,
            self.sink_refresh_method,
            self.sink_refresh_latent,
        )
        self._remember_visual(current.visual)
        self._remember_paired(ready, current.visual)
        self.pending.popleft()
        self.pending_bytes -= ready.nbytes
        self.pending_spans -= (
            0 if ready.provenance is None else len(ready.provenance.source_alignment)
        )
        self.committed_chunk_id += 1
        self.committed_latent = ready.chunk.end
        if ready.provenance is not None:
            self.committed_pixel = ready.provenance.pixels.end
        self.media_origin, self.applied_prompt = origin, prompt
        self.model_chunk_id += 1
        self.source_dependency_end = dependency
        if self.complete:
            self.dispose()
        return result

    def _step(self):
        ready = self.pending[0]
        self._activate_pending_prompt(ready, defer_prefix=True)
        self._activate_pending_sink_rebuild(ready)
        prompt = self.prompt
        current, origin = self._prepare(ready, prompt, self.media_origin)
        with self.rng.random_stream(self.sampler.device, self.seed):
            self._apply_pending_sink(ready, current)
            with measure(
                getattr(self, "ttff_timing", None), "initial_joint_step0", cuda=True
            ):
                first = self._initial_prediction(current, prompt)
            if first is None:
                with measure(
                    getattr(self, "ttff_timing", None),
                    "condition_prefix_apply",
                    cuda=True,
                ):
                    self._apply_prefix(current, prompt)
            if self._pending_prefix is None:
                self.sampler.step(current, first_prediction=first)
            else:
                with measure(
                    getattr(self, "ttff_timing", None), "prompt_joint_step0", cuda=True
                ):
                    first = self._prefix_prediction(current, False)
                self.sampler.step(current, first_prediction=first)
        return self._finish(ready, current, prompt, origin)

    @torch.no_grad()
    def drain_ready(self, *, max_outputs=None):
        """Drain ready chunks, committing each result before yielding.

        The caller must close this synchronous iterator on decoder failure or
        cancellation; abandoned work closes the session, including a next
        target whose step0 may already be active. When ``max_outputs`` stops a
        bounded drain cleanly after a cross-chunk lookahead, the continuation
        is retained for the next ``drain_ready`` call.
        """
        if max_outputs is not None and (
            type(max_outputs) is not int or max_outputs < 1
        ):
            raise ValueError("max_outputs must be a positive integer")
        if self.closed or self.draining:
            raise ValueError("The source-ready session is closed or draining")
        if max_outputs is not None and (
            type(max_outputs) is not int or max_outputs < 1
        ):
            raise ValueError("Output limit must be a positive integer")
        self.draining = True
        finished, emitted = False, 0
        timer = None
        try:
            self._agree_ready_route(
                "drain_entry", first_prediction=self._drain_continuation is not None
            )
            if not self.pending:
                self._drain_continuation = None
                finished = True
                return
            timer = ComponentTimer("dit", **self.metric_geometry)
            sampler = self.sampler
            if self._drain_continuation is not None:
                current, first, prompt, origin = self._drain_continuation
                self._drain_continuation = None
            else:
                self._activate_pending_prompt(self.pending[0], defer_prefix=True)
                self._activate_pending_sink_rebuild(self.pending[0])
                prompt = self.prompt
                current, origin = self._prepare(
                    self.pending[0], prompt, self.media_origin
                )
                first = None
            emitted = 0
            while True:
                # This vote also covers a warm prediction and sequential modes.
                # A drained peer cannot return while another prepares next ready.
                self._agree_ready_route(
                    "drain_iteration", first_prediction=first is not None
                )
                if not self.pending:
                    break
                if current is None:
                    self._activate_pending_prompt(self.pending[0], defer_prefix=True)
                    self._activate_pending_sink_rebuild(self.pending[0])
                    prompt, origin = self.prompt, self.media_origin
                    current, _ = self._prepare(self.pending[0], prompt, origin)
                ready = self.pending[0]
                next_state = next_first = None
                with self.rng.random_stream(sampler.device, self.seed):
                    self._apply_pending_sink(ready, current)
                    source, target, start, end = current.spans[0]
                    graphs = getattr(sampler.generator, "chunk_graphs", None)
                    graph_eligible = graphs is not None and graphs.eligible(
                        sampler, self.cache, current
                    )
                    cold_init = (
                        graph_eligible
                        and prompt is not self.applied_prompt
                        and self.cache.prefix is None
                        and self.cache.last_block == 0
                        and self.cache.active is None
                    )
                    sampler._agree_sp_boundary(
                        self.cache,
                        target,
                        phase="source_ready_graph_route",
                        ready_route=(graph_eligible, first is not None, cold_init),
                        needs_prefix_update=prompt is not self.applied_prompt,
                        layout_signature=sampler._collective_layout_signature(
                            current.positive, source, target
                        ),
                    )
                    if self._pending_prefix is not None:
                        with measure(
                            getattr(self, "ttff_timing", None),
                            "prompt_joint_step0",
                            cuda=True,
                        ):
                            first = self._prefix_prediction(current, graph_eligible)
                    if cold_init:
                        with measure(
                            getattr(self, "ttff_timing", None),
                            "initial_joint_graph",
                            cuda=True,
                        ):
                            first = graphs.initialization(sampler, current)
                    else:
                        if first is None:
                            with measure(
                                getattr(self, "ttff_timing", None),
                                "initial_joint_step0",
                                cuda=True,
                            ):
                                first = self._initial_prediction(current, prompt)
                        if first is None:
                            with measure(
                                getattr(self, "ttff_timing", None),
                                "condition_prefix_apply",
                                cuda=True,
                            ):
                                self._apply_prefix(current, prompt)
                    # A prompt rebuild has already cleared the old active block.
                    # Otherwise its values are the tail of the existing history.
                    if (
                        self.defer_final
                        and first is None
                        and self.cache.active is not None
                    ):
                        final = None
                        prepare_error = None
                        try:
                            final = sampler._deferred_final(
                                self.cache, self.recent_visual, current.positive
                            )
                        except Exception as exc:
                            prepare_error = exc
                        sampler._agree_sp_boundary(
                            self.cache,
                            target,
                            phase="source_deferred_prepare",
                            precheck_error=prepare_error is not None,
                            local_error=prepare_error,
                        )
                        if prepare_error is not None:
                            raise prepare_error
                        packs = sampler._resume_cross_packs(final, current)
                        if packs is None:
                            sampler._forward(final, self.cache, "final")
                        elif graph_eligible:
                            with measure(
                                getattr(self, "ttff_timing", None),
                                "resume_joint_graph",
                                cuda=True,
                            ):
                                first = graphs.resume(sampler, current, packs)
                        else:
                            first = sampler._predict_joint(
                                packs,
                                ("final", "source", "active"),
                                current.visual.shape[2:],
                                self.cache,
                            )
                            if first is None:
                                sampler._forward(final, self.cache, "final")
                        final = packs = None
                    defer_final = (
                        self.defer_final
                        and (not self.ended or len(self.pending) > 1)
                        and (len(self.pending) == 1 or max_outputs == 1)
                    )
                    if graph_eligible:
                        if first is None:
                            with measure(
                                getattr(self, "ttff_timing", None),
                                "source_step0_graph",
                                cuda=True,
                            ):
                                first = graphs.recovery(sampler, current)
                        if (
                            not defer_final
                            and len(self.pending) > 1
                            and self.pending_prompt is None
                            and self.pending_sink is None
                            and not self._auto_sink_blocks(ready.chunk.end)
                        ):
                            next_state, _ = self._prepare(
                                self.pending[1], prompt, origin, previous=current
                            )
                        with measure(
                            getattr(self, "ttff_timing", None),
                            "denoise_graph",
                            cuda=True,
                        ):
                            video, next_first = graphs.chunk(
                                sampler,
                                current,
                                first,
                                next_state,
                                defer_final=defer_final,
                            )
                        sound = None
                    else:
                        if first is None:
                            sampler._begin_chunk(
                                self.cache, current.positive, source, target
                            )
                        vs, aus = (
                            sampler.schedules_for(current)
                            if isinstance(sampler, ZingH3RealtimeSampler)
                            else (sampler.video_scheduler, sampler.audio_scheduler)
                        )
                        lookahead = (
                            not defer_final
                            and sampler.joint_forward == "cross_chunk"
                            and len(self.pending) > 1
                            and self.pending_prompt is None
                            and self.pending_sink is None
                            and not self._auto_sink_blocks(ready.chunk.end)
                        )
                        with measure(
                            getattr(self, "ttff_timing", None),
                            "denoise_steps",
                            cuda=True,
                        ):
                            video, sound = sampler._sample_block(
                                current.visual,
                                current.audio,
                                current.positive,
                                current.negative,
                                self.cache,
                                vs,
                                aus,
                                source,
                                target,
                                start,
                                end,
                                first_prediction=first,
                                defer_final=lookahead or defer_final,
                            )
                        if lookahead:
                            next_state, _ = self._prepare(
                                self.pending[1], prompt, origin, previous=current
                            )
                            next_first = sampler._ready_cross_prediction(
                                current, next_state
                            )
                            if next_first is None:
                                sampler._commit(
                                    current.positive, target, video, sound, self.cache
                                )
                        elif sampler.joint_forward == "cross_chunk" and not defer_final:
                            reasons = sampler.last_run_metrics["joint_fallbacks"]
                            reason = "eos" if self.ended else "source_not_ready"
                            reasons[reason] = reasons.get(reason, 0) + 1
                    sampler.last_run_metrics["dit_forwards"] = sum(
                        sampler._forward_counts.values()
                    )
                result = self._finish(ready, current, prompt, origin)
                # Drop the completed chunk's packed inputs and scheduler
                # temporaries before yielding; only lookahead survives decode.
                current, first = next_state, next_first
                ready = video = sound = None
                self._drain_tensors = []
                if current is not None:
                    for layout in (current.positive, current.negative):
                        if layout is not None:
                            self._drain_tensors.extend(
                                v
                                for v in layout.values()
                                if isinstance(v, torch.Tensor)
                            )
                    self._drain_tensors.extend(
                        v for v in (current.visual, current.audio) if v is not None
                    )
                    if first is not None:
                        self._drain_tensors.extend(v for v in first if v is not None)
                if result.provenance is not None:
                    timer.frames = result.provenance.pixels.count
                timer.finish()
                timer = None
                yield result
                emitted += 1
                if self.closed:
                    break
                if max_outputs is not None and emitted >= max_outputs:
                    if current is not None:
                        self._drain_continuation = current, first, prompt, origin
                    break
                timer = ComponentTimer("dit", **self.metric_geometry)
            finished = True
        finally:
            if timer is not None and not finished:
                # An unfinished iteration has no committed output frames.
                timer.finish(error=RuntimeError("DiT iteration did not complete"))
            self.draining = False
            if self._drain_continuation is None:
                self._drain_tensors = []
            if not finished:
                self.dispose()

    def memory_snapshot(self):
        """Retained tensor storage only; excludes allocator/workspace/output owners.

        On-demand accounting, not a GPU peak measurement or admission guarantee.
        Full-history KV grows; a temporal window bounds retained chunk count.
        Prefix/reference size and caller-owned outputs are separate budgets.
        """

        def storage_bytes(tensors):
            storages = {}
            for tensor in tensors:
                storage = tensor.untyped_storage()
                storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
            return sum(storages.values())

        kv = () if self.cache is None else self.cache.owned_tensors()
        static = []
        if self.inputs is not None:
            static.extend(self.inputs.visual.values())
            static.extend(self.inputs.audio.values())
        for prompt in (self.prompt, self.pending_prompt, self.applied_prompt):
            if prompt is not None:
                static.extend(prompt.condition.values())
                if prompt.inputs is not None:
                    static.extend(prompt.inputs.positive.values())
                    static.extend(prompt.inputs.visual.values())
                    static.extend(prompt.inputs.audio.values())
                    static.extend(
                        item["frames"]
                        for item in prompt.inputs.condition_media or []
                        if "frames" in item
                    )
        if self.recent_visual is not None:
            static.append(self.recent_visual)
        return {
            "ready_bytes": self.pending_bytes,
            "ready_chunks": len(self.pending),
            "ready_provenance_spans": self.pending_spans,
            "kv_storage_bytes": storage_bytes(kv),
            "kv_temporal_chunks": 0
            if self.cache is None
            else len(self.cache.committed),
            "static_storage_bytes": storage_bytes(static),
            "execution_storage_bytes": storage_bytes(self._drain_tensors),
        }

    def kv_counters(self):
        """Pool evidence: logical rows, fixed capacity and main-store growth.

        Lifetime counters, kept out of memory_snapshot(), which reports only
        retained state and is all zero after dispose.
        """
        cache = self.cache
        if cache is None:
            return {"kv_pooled": self.acquire_storage is not None}
        return {
            "kv_pooled": cache.pooled,
            "kv_capacity_rows": cache.capacity,
            "kv_logical_rows": cache._length,
            "kv_prefix_rows": (
                0 if cache.prefix is None else int(cache.prefix.positions.shape[0])
            ),
            **{f"kv_{name}": value for name, value in cache.stats.items()},
        }

    def dispose(self):
        if self.cache is not None:
            self.cache.clear()
        if self.adaln_lease is not None:
            self.adaln_lease.close()
            self.adaln_lease = None
        self._drain_tensors = []
        self._drain_continuation = None
        self._pending_prefix = None
        self.pending.clear()
        self.pending_bytes = 0
        self.pending_spans = 0
        self.rng.dispose()
        if self.sampler is not None:
            self.sampler.processor._prefix_memo = None
        self.inputs = self.prompt = self.pending_prompt = self.applied_prompt = None
        self.pending_sink = None
        self.recent_paired = None
        self.recent_visual = self.sampler = None
        self.prefix_loader = None
        self.restore_initial_prefix = self.capture_initial_prefix = None
        self.prepare_prompt_prefix = None
        self._cached_prefix_ready = False
        self.closed = True
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
