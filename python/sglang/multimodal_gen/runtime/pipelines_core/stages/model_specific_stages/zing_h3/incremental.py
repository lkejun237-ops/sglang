# SPDX-License-Identifier: Apache-2.0
"""Fixture-driven, source-ready execution with optional temporal window; not a v1 service adapter.

Initial noise must be supplied from a frozen native run. Generating it online
would change channel-major RNG assignment and needs a separate model contract.
"""

from collections import deque
from copy import copy
from dataclasses import dataclass
from typing import NamedTuple

import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import TemporalChunk
from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    Provenance,
)

from .packing import H3InferenceInputs, H3RowRole
from .realtime import ZingH3RealtimeState
from .realtime_sampling import ZingH3RealtimeSampler, ZingH3RealtimeSamplingState


def _bytes(tensor):
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


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


class PromptSnapshot(NamedTuple):
    condition: dict[str, torch.Tensor]
    event_id: int | str | None
    prompt_version: int | None
    inputs: H3InferenceInputs | None = None


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
        sampler.processor.prompt_switch_prefix_chunks()
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
        self.applied_prompt = None
        self.recent_visual = None
        self.model_chunk_id = 0
        self.source_dependency_end = 0
        self.media_origin = None
        self.ended = self.closed = False
        self.draining = False
        self._drain_tensors = []
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

    def update_prompt(self, condition, event_id, prompt_version=None, *, inputs=None):
        if self.closed or self.draining:
            raise ValueError("The source-ready session is closed or draining")
        candidate = self._snapshot(condition, event_id, prompt_version, inputs)
        self.pending_prompt = candidate

    def discard_pending_prompt(self):
        """Drop an encoded candidate superseded by a newer control event."""
        if self.closed or self.draining:
            raise ValueError("The source-ready session is closed or draining")
        self.pending_prompt = None

    @property
    def prompt_switch_ready(self):
        return (
            self.recent_visual is not None
            and self.recent_visual.shape[2] == self.prompt_switch_prefix_latent_frames
        )

    def _next_model_chunk(self):
        if self.pending:
            if self.pending_prompt is not None and self.prompt_switch_ready:
                return None
            previous = self.pending[-1].model_chunk
            return self.plan.chunk(previous.chunk_id + 1)
        if self.pending_prompt is not None and self.prompt_switch_ready:
            return self.plan.chunk(3)
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

    def _activate_pending_prompt(self, ready):
        if (
            self.pending_prompt is None
            or not self.prompt_switch_ready
            or ready.model_chunk.chunk_id != 3
            or ready.model_chunk.start != 5
        ):
            return
        if self.audio_enabled:
            raise ValueError("Prompt switching currently supports video-only sessions")
        candidate = self.pending_prompt
        inputs = self.inputs if candidate.inputs is None else candidate.inputs
        origin, latent_end = self.sampler.rebuild_prompt_prefix(
            inputs, candidate.condition, self.recent_visual, self.cache
        )
        if latent_end != ready.model_chunk.start:
            raise ValueError("Prompt-switch prefix does not meet the next model block")
        if candidate.inputs is not None:
            self.inputs = copy(candidate.inputs)
            self.inputs.source_visual = self.inputs.positive = None
            self.inputs.condition_media = None
        self.prompt = self.applied_prompt = candidate
        if self.pending_prompt is candidate:
            self.pending_prompt = None
        self.media_origin = origin
        self.model_chunk_id = ready.model_chunk.chunk_id

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
            ),
        )

    @torch.no_grad()
    def step(self):
        if self.closed or self.draining:
            raise ValueError("The source-ready session is closed or draining")
        try:
            self._agree_ready_route("step")
            if not self.pending:
                return None  # Wait for source. Never repeat frames or advance RNG.
            return self._step()
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
        return current, origin

    def _apply_prefix(self, current, prompt):
        if prompt is self.applied_prompt:
            return
        # Later prompt changes rebuild the full rolling prefix before _prepare.
        # This path only initializes an empty session cache.
        if self.cache.prefix is not None or self.cache.last_block:
            raise ValueError("Prompt changes require a rolling prefix rebuild")
        sampler, positive = self.sampler, current.positive
        source, target, _, _ = current.spans[0]
        sampler._precheck_initial_prefix(self.cache, positive, source, target)
        if bool((positive["block_idx"] == 0).any()):
            sampler._forward(
                sampler._select_block(
                    positive, 0, phase="prefix", video_t=1, audio_t=1
                ),
                self.cache,
                "prefix",
            )

    def _finish(self, ready, current, prompt, origin):
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
        )
        self._remember_visual(current.visual)
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
        self._activate_pending_prompt(ready)
        prompt = self.prompt
        current, origin = self._prepare(ready, prompt, self.media_origin)
        with self.rng.random_stream(self.sampler.device, self.seed):
            self._apply_prefix(current, prompt)
            self.sampler.step(current)
        return self._finish(ready, current, prompt, origin)

    @torch.no_grad()
    def drain_ready(self):
        """Drain a finite ready batch, committing each result before yielding.

        Lookahead never survives this synchronous iterator. The caller must
        close it on decoder failure/cancellation; abandoned work closes the
        session, including a next target whose step0 may already be active.
        """
        if self.closed or self.draining:
            raise ValueError("The source-ready session is closed or draining")
        self.draining = True
        finished = False
        try:
            self._agree_ready_route("drain_entry")
            if not self.pending:
                finished = True
                return
            self._activate_pending_prompt(self.pending[0])
            prompt = self.prompt
            current, origin = self._prepare(self.pending[0], prompt, self.media_origin)
            sampler = self.sampler
            first = None
            while True:
                # This vote also covers a warm prediction and sequential modes.
                # A drained peer cannot return while another prepares next ready.
                self._agree_ready_route(
                    "drain_iteration", first_prediction=first is not None
                )
                if not self.pending:
                    break
                if current is None:
                    self._activate_pending_prompt(self.pending[0])
                    prompt, origin = self.prompt, self.media_origin
                    current, _ = self._prepare(self.pending[0], prompt, origin)
                ready = self.pending[0]
                next_state = next_first = None
                with self.rng.random_stream(sampler.device, self.seed):
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
                    if cold_init:
                        first = graphs.initialization(sampler, current)
                    else:
                        self._apply_prefix(current, prompt)
                    if graph_eligible:
                        if first is None:
                            first = graphs.recovery(sampler, current)
                        if len(self.pending) > 1 and self.pending_prompt is None:
                            next_state, _ = self._prepare(
                                self.pending[1], prompt, origin, previous=current
                            )
                        video, next_first = graphs.chunk(
                            sampler, current, first, next_state
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
                            sampler.joint_forward == "cross_chunk"
                            and len(self.pending) > 1
                            and self.pending_prompt is None
                        )
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
                            defer_final=lookahead,
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
                        elif sampler.joint_forward == "cross_chunk":
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
                yield result
                if self.closed:
                    break
            finished = True
        finally:
            self.draining = False
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
        self.pending.clear()
        self.pending_bytes = 0
        self.pending_spans = 0
        self.rng.dispose()
        self.inputs = self.prompt = self.pending_prompt = self.applied_prompt = None
        self.recent_visual = self.sampler = None
        self.closed = True
