# SPDX-License-Identifier: Apache-2.0
"""Bounded ownership transfer from encoder clips to checkpoint AR chunks."""

from collections import deque
from dataclasses import dataclass

import torch

from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import TemporalChunk
from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange, Provenance

from .streaming_noise import STREAMING_NOISE_VERSION, indexed_video_noise


def source_pixel_end(latent_end, bridge, *, frame_end=None):
    """Original absolute latent-to-pixel boundary, clipped only at source EOS."""
    clip_index, last_offset = divmod(latent_end - 1, bridge.clip_latents)
    end = clip_index * bridge.clip_frames + 1 + last_offset * bridge.ratio
    return end if frame_end is None else min(end, frame_end)


@dataclass(frozen=True)
class SourceChunk:
    chunk: TemporalChunk
    source: torch.Tensor
    noise: torch.Tensor
    provenance: Provenance
    final: bool


class SourceChunkBridge:
    """No model kernels or whole-video layouts; pop transfers one owned chunk.

    ``append`` receives stable posterior samples, never moments, including
    multiple contiguous slices within a codec clip. Pixel alignment follows
    checkpoint clip geometry; dependency is the actual encoder watermark and
    may include lookahead. Source IDs come from the service ledger.
    """

    def __init__(
        self,
        plan,
        *,
        clip_frames,
        clip_latents,
        temporal_ratio,
        capacity_latents,
        capacity_bytes,
        seed,
        provenance_for,
        target_shape=None,
    ):
        if any(
            type(v) is not int or v < 1
            for v in (
                clip_frames,
                clip_latents,
                temporal_ratio,
                capacity_latents,
                capacity_bytes,
            )
        ):
            raise ValueError("Bridge geometry and capacities must be positive integers")
        if capacity_latents < max(plan.lengths) + clip_latents:
            raise ValueError("Bridge capacity must hold an AR chunk and encoder clip")
        self.plan = plan
        self.clip_frames, self.clip_latents, self.ratio = (
            clip_frames,
            clip_latents,
            temporal_ratio,
        )
        self.capacity_latents, self.capacity_bytes = capacity_latents, capacity_bytes
        self.seed, self.provenance_for = seed, provenance_for
        if target_shape is not None and (
            len(target_shape) != 3
            or any(type(v) is not int or v < 1 for v in target_shape)
        ):
            raise ValueError("Target latent shape must be positive [C,H,W]")
        self.target_shape = target_shape
        self.values = None
        self.pixel_ends, self.dependencies = deque(), deque()
        self.accepted_latents = self.accepted_pixels = self.chunk_id = (
            self.emitted_pixels
        ) = 0
        self.ended = self.closed = False
        self.short_tail = False

    @property
    def buffered_latents(self):
        return len(self.dependencies)

    @property
    def buffered_bytes(self):
        return (
            0
            if self.values is None
            else self.values.numel() * self.values.element_size()
        )

    def append(self, source, *, latent_start, pixel_end, source_dependency_end):
        if self.ended or self.closed:
            raise ValueError("Source bridge is closed")
        if source.ndim != 5 or not source.is_floating_point() or min(source.shape) < 1:
            raise ValueError("Source bridge requires sampled video latents")
        count = source.shape[2]
        # Compute alignment from absolute latent positions, not the size of the
        # latest network packet or the encoder's dependency/lookahead watermark.
        clip_index, offset = divmod(self.accepted_latents, self.clip_latents)
        clip_start = clip_index * self.clip_frames
        nominal_ends = [
            min(clip_start + self.clip_frames, clip_start + 1 + i * self.ratio)
            for i in range(offset, offset + count)
        ]
        previous_end = (
            clip_start
            if offset + count == 1
            else clip_start + 1 + (offset + count - 2) * self.ratio
        )
        if (
            type(latent_start) is not int
            or latent_start != self.accepted_latents
            or type(pixel_end) is not int
            or offset + count > self.clip_latents
            or not max(self.accepted_pixels, previous_end)
            < pixel_end
            <= nominal_ends[-1]
            or type(source_dependency_end) is not int
            or source_dependency_end < pixel_end
        ):
            raise ValueError(
                "Source codec slice has inconsistent geometry or dependency"
            )
        # Only a final codec clip may be short; accepting another would change
        # the absolute latent-to-pixel mapping and silently corrupt alignment.
        if self.short_tail:
            raise ValueError("A partial source clip must be the final clip")
        signature = (*source.shape[:2], *source.shape[3:], source.device, source.dtype)
        if self.accepted_latents and signature != self.signature:
            raise ValueError("Source bridge geometry changed")
        nbytes = source.numel() * source.element_size()
        if (
            self.buffered_latents + count > self.capacity_latents
            or self.buffered_bytes + nbytes > self.capacity_bytes
        ):
            raise ValueError("Source bridge capacity exceeded")
        owned = (
            source.detach().clone()
            if self.values is None
            else torch.cat((self.values, source.detach()), 2)
        )
        self.pixel_ends.extend(min(pixel_end, end) for end in nominal_ends)
        self.dependencies.extend([source_dependency_end] * count)
        self.values, self.signature = owned, signature
        self.accepted_latents += count
        self.accepted_pixels = pixel_end
        self.short_tail = pixel_end < nominal_ends[-1]

    def finish(self):
        if self.closed or self.ended or not self.accepted_pixels:
            raise ValueError("Source bridge cannot finish in its current state")
        self.ended = True

    @property
    def ready(self):
        if self.closed:
            return False
        chunk = self.plan.chunk(
            self.chunk_id, self.accepted_latents if self.ended else None
        )
        return chunk is not None and chunk.end <= self.accepted_latents

    def ready_for(self, count):
        """Whether ``count`` model-phase latents, or a final tail, can transfer."""
        if type(count) is not int or count < 1:
            raise ValueError("Requested source chunk length must be a positive integer")
        if self.closed or not self.buffered_latents:
            return False
        return self.buffered_latents >= count or self.ended

    def pop(self, count=None):
        if count is None:
            if not self.ready:
                return None
            chunk = self.plan.chunk(
                self.chunk_id, self.accepted_latents if self.ended else None
            )
        else:
            if not self.ready_for(count):
                return None
            count = min(count, self.buffered_latents)
            start = self.accepted_latents - self.buffered_latents
            chunk = TemporalChunk(self.chunk_id, start, start + count)
        if chunk is None:
            return None
        count = chunk.end - chunk.start
        pixel_end = self.pixel_ends[count - 1]
        dependency = max(self.dependencies[i] for i in range(count))
        provenance = self.provenance_for(
            PixelRange(self.emitted_pixels, pixel_end), dependency
        )
        if (
            not isinstance(provenance, Provenance)
            or provenance.pixels != PixelRange(self.emitted_pixels, pixel_end)
            or provenance.source_dependency_end != dependency
        ):
            raise ValueError("Source provenance does not match the codec chunk")
        source = self.values[:, :, :count].clone()
        channels, height, width = self.target_shape or (
            source.shape[1],
            *source.shape[-2:],
        )
        noise = indexed_video_noise(
            (source.shape[0], channels, count, height, width),
            seed=self.seed,
            latent_start=chunk.start,
            purpose="target_initial",
            device=source.device,
            dtype=torch.float32,
            version=STREAMING_NOISE_VERSION,
        )
        result = SourceChunk(
            chunk,
            source,
            noise,
            provenance,
            self.ended and chunk.end == self.accepted_latents,
        )
        self.values = (
            self.values[:, :, count:].clone() if count < self.buffered_latents else None
        )
        for _ in range(count):
            self.pixel_ends.popleft()
            self.dependencies.popleft()
        self.chunk_id += 1
        self.emitted_pixels = pixel_end
        return result

    def dispose(self):
        self.values = None
        self.pixel_ends.clear()
        self.dependencies.clear()
        self.provenance_for = None
        self.closed = True
