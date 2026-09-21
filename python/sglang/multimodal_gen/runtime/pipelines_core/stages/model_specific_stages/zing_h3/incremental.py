# SPDX-License-Identifier: Apache-2.0
"""Fixture-driven, source-ready execution with optional temporal window; not a v1 service adapter.

Initial noise must be supplied from a frozen native run. Generating it online
would change channel-major RNG assignment and needs a separate model contract.
"""

from collections import deque
from copy import copy
from dataclasses import dataclass

import torch

from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    Provenance,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import TemporalChunk

from .packing import H3RowRole
from .realtime import ZingH3RealtimeState
from .sampling import ZingH3SamplingState


def _bytes(tensor):
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


@dataclass(frozen=True)
class ReadyLatentChunk:
    chunk: TemporalChunk
    source: torch.Tensor
    noise: torch.Tensor
    audio: torch.Tensor | None
    audio_start: int
    # Immutable #111 provenance supplied by the codec/fixture bridge.
    # This worker does not manufacture pixel ranges from latent indices.
    provenance: Provenance

    @property
    def nbytes(self):
        return sum(_bytes(t) for t in (self.source, self.noise, self.audio))


@dataclass(frozen=True)
class ChunkResult:
    chunk: TemporalChunk
    visual: torch.Tensor
    audio: torch.Tensor | None
    provenance: Provenance
    prompt_event_id: int | None


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
    ):
        if any(
            type(n) is not int or n < 1
            for n in (max_pending_chunks, max_pending_bytes, max_pending_spans)
        ):
            raise ValueError("Ready queue budgets must be positive integers")
        # Validate finite geometry without allocating a complete session layout.
        sampler.processor.chunk_plan.chunk(0, total_latent_frames)
        self.sampler = sampler
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
        self.cache = ZingH3KVCache(
            len(sampler.generator.blocks),
            window_policy=sampler.window_policy,
            rope_position_mode=sampler.rope_position_mode,
            rope_max_frame_gap=sampler.rope_max_frame_gap,
        )
        self.rng = ZingH3RealtimeState()
        if initial_rng_state is not None:
            cpu_rng, cuda_rng = initial_rng_state
            # A frozen native run must hand over the state *after* initial-noise
            # generation, not restart re-noise from the request seed.
            torch.Generator(device="cpu").set_state(cpu_rng)
            self.rng.cpu_rng = cpu_rng.clone()
            self.rng.cuda_rng = None if cuda_rng is None else cuda_rng.clone()
        self.prompt = self._snapshot(inputs.positive, None)
        self.applied_prompt = None
        self.source_dependency_end = 0
        self.media_origin = None
        self.ended = self.closed = False
        self.audio_enabled = None
        self.device = None
        sampler._forward_counts = dict.fromkeys(
            ("prefix", "source", "denoise", "unconditional", "commit"), 0
        )
        sampler.last_run_metrics = {
            "dit_forwards": 0,
            "forward_counts": sampler._forward_counts,
        }

    @staticmethod
    def _snapshot(condition, event_id):
        return ({k: v.detach().clone() for k, v in condition.items()}, event_id)

    def update_prompt(self, condition, event_id):
        if self.closed:
            raise ValueError("The source-ready session is closed")
        self.prompt = self._snapshot(condition, event_id)

    def append(
        self, source, noise, *, provenance, audio=None, audio_start=0, final=False
    ):
        if self.closed or self.ended:
            raise ValueError("The source-ready session no longer accepts chunks")
        packer = self.sampler.processor
        if packer.chunk_plan != self.plan:
            raise ValueError("The checkpoint chunk plan changed within the session")
        packer._validate_visual(source)
        packer._validate_visual(noise)
        if not source.is_floating_point() or not noise.is_floating_point():
            raise ValueError("Ready source and noise must be floating-point latents")
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
        if not isinstance(provenance, Provenance):
            raise ValueError("Ready chunks require explicit source provenance")
        if provenance.pixels.start != self.next_pixel:
            raise ValueError("Ready source provenance must be continuous")
        if self.audio_enabled is not None and (audio is not None) != self.audio_enabled:
            raise ValueError("Ready target audio mode changed within the session")
        total = self.total_latent_frames
        if final:
            end = self.next_latent + noise.shape[2]
            if total is not None and end != total:
                raise ValueError("Final chunk does not cover the finite latent target")
            total = end
        chunk = self.plan.chunk(self.next_chunk_id, total)
        if chunk is None or noise.shape[2] != chunk.end - chunk.start:
            raise ValueError("Ready data does not cover the next checkpoint chunk")
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
        nbytes = sum(_bytes(t) for t in (source, noise, audio))
        if (
            len(self.pending) >= self.max_pending_chunks
            or self.pending_bytes + nbytes > self.max_pending_bytes
            or self.pending_spans + len(provenance.source_alignment)
            > self.max_pending_spans
        ):
            raise ValueError("The source-ready queue is full")
        owned = ReadyLatentChunk(
            chunk,
            source.detach().clone(),
            noise.detach().clone(),
            None if audio is None else audio.detach().clone(),
            audio_start,
            provenance,
        )
        self.pending.append(owned)
        self.pending_bytes += nbytes
        self.pending_spans += len(provenance.source_alignment)
        self.next_chunk_id += 1
        self.next_latent = chunk.end
        self.next_pixel = provenance.pixels.end
        self.next_audio += 0 if audio is None else audio.shape[2]
        self.audio_enabled = audio is not None
        self.device = noise.device
        if final:
            self.total_latent_frames = total
            self.ended = True

    def finish(self):
        if self.closed or self.ended:
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

    @torch.no_grad()
    def step(self):
        if self.closed:
            raise ValueError("The source-ready session is closed")
        if not self.pending:
            return None  # Wait for source. Never repeat frames or advance RNG.
        try:
            return self._step()
        except BaseException:
            self.dispose()
            raise

    def _step(self):
        ready = self.pending[0]
        sampler, packer = self.sampler, self.sampler.processor
        if packer.chunk_plan != self.plan:
            raise ValueError("The checkpoint chunk plan changed within the session")
        if (
            ready.noise.is_cuda
            and self.rng.cpu_rng is not None
            and self.rng.cuda_rng is None
        ):
            raise ValueError("Frozen CUDA sampling requires the initial CUDA RNG state")
        sampler.device = ready.noise.device
        sampler.video_scheduler.to(sampler.device)
        sampler.audio_scheduler.to(sampler.device)
        prompt = self.prompt
        inputs = copy(self.inputs)
        inputs.positive, inputs.source_visual = prompt[0], ready.source
        visual = ready.noise.float().clone()
        audio = None if ready.audio is None else ready.audio.float().clone()
        kwargs = dict(
            chunk=ready.chunk,
            audio_start=ready.audio_start,
            media_origin=self.media_origin,
        )
        positive = packer.pack_chunk(inputs, prompt[0], visual, audio, **kwargs)
        negative = (
            packer.pack_chunk(inputs, None, visual, audio, **kwargs)
            if sampler.do_cfg
            else None
        )
        origin = self.media_origin
        if origin is None:
            target = positive["row_role"] == int(H3RowRole.TARGET_VISUAL)
            origin = float(positive["position_ids"][target][0, 0])
        current = ZingH3SamplingState(
            visual,
            audio,
            positive,
            negative,
            [(ready.chunk.source_block, ready.chunk.target_block, 0, visual.shape[2])],
            self.cache,
        )
        with self.rng.random_stream(sampler.device, self.seed):
            if prompt is not self.applied_prompt:
                prefix_cache = ZingH3KVCache(len(sampler.generator.blocks))
                try:
                    if bool((positive["block_idx"] == 0).any()):
                        sampler._forward(
                            sampler._select_block(
                                positive, 0, phase="prefix", video_t=1, audio_t=1
                            ),
                            prefix_cache,
                            "prefix",
                        )
                    self.cache.prefix = prefix_cache.prefix
                finally:
                    prefix_cache.clear()
            sampler.step(current)
            dependency = max(
                self.source_dependency_end, ready.provenance.source_dependency_end
            )
            provenance = Provenance(
                ready.provenance.pixels, ready.provenance.source_alignment, dependency
            )
            result = ChunkResult(ready.chunk, visual, audio, provenance, prompt[1])
        self.pending.popleft()
        self.pending_bytes -= ready.nbytes
        self.pending_spans -= len(ready.provenance.source_alignment)
        self.committed_chunk_id += 1
        self.committed_latent = ready.chunk.end
        self.committed_pixel = ready.provenance.pixels.end
        self.media_origin, self.applied_prompt = origin, prompt
        self.source_dependency_end = dependency
        if self.complete:
            self.dispose()
        return result

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

        kv = []
        for chunk in self.cache._history():
            kv.append(chunk.positions)
            kv.extend(t for layer in chunk.layers for t in layer)
        static = []
        if self.inputs is not None:
            static.extend(self.inputs.visual.values())
            static.extend(self.inputs.audio.values())
        for prompt in (self.prompt, self.applied_prompt):
            if prompt is not None:
                static.extend(prompt[0].values())
        return {
            "ready_bytes": self.pending_bytes,
            "ready_chunks": len(self.pending),
            "ready_provenance_spans": self.pending_spans,
            "kv_storage_bytes": storage_bytes(kv),
            "kv_temporal_chunks": len(self.cache.committed),
            "static_storage_bytes": storage_bytes(static),
        }

    def dispose(self):
        self.cache.clear()
        self.pending.clear()
        self.pending_bytes = 0
        self.pending_spans = 0
        self.rng.dispose()
        self.inputs = self.prompt = self.applied_prompt = self.sampler = None
        self.closed = True
