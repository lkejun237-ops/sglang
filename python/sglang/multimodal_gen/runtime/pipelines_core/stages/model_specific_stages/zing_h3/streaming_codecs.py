# SPDX-License-Identifier: Apache-2.0
"""Pull-based H3 codec buffers, independent of transport and AR chunk boundaries.

This module is not wired into the public realtime adapter. Source results are
posterior moments: choosing a streaming posterior RNG contract is a prerequisite
for producing native-equivalent sampled latents with unknown total length.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    PixelRange,
    Provenance,
)


@dataclass(frozen=True)
class CodecSlice:
    values: torch.Tensor
    start: int
    end: int
    source_dependency_end: int


@dataclass(frozen=True)
class DecodedVideoSlice(CodecSlice):
    provenance: Provenance | None = None


class SourcePixelBuffer:
    """Own at most ``capacity_frames`` source pixels; encode one clip per pop.

    ``values`` in pop results are moments with an absolute latent interval.
    Dependencies are exclusive pixel watermarks, including real lookahead but
    excluding EOS replication. References must use the separate offline path.
    """

    def __init__(self, codecs, *, capacity_frames):
        self.codecs = codecs
        self.clip = codecs.vae_frames_per_chunk
        self.tokens = codecs.vae_latents_per_chunk
        self.ratio = codecs.temporal_ratio
        if capacity_frames < self.clip:
            raise ValueError("Source capacity must hold a complete VAE clip")
        self.capacity_frames = capacity_frames
        self._pixels = None
        self.accepted_frames = self.consumed_frames = self.emitted_latents = 0
        self.ended = self.disposed = False

    @property
    def minimum_credit_frames(self):
        return self.clip

    @property
    def buffered_frames(self):
        return 0 if self._pixels is None else self._pixels.shape[1]

    @property
    def ready(self):
        return not self.disposed and (
            self.buffered_frames >= self.clip
            or (self.ended and self.buffered_frames > 0)
        )

    def append(self, pixels, *, frame_start):
        if self.disposed or self.ended:
            raise ValueError("Source stream is closed")
        if pixels.ndim != 4 or pixels.shape[0] != 3 or pixels.shape[1] < 1:
            raise ValueError("Source pixels must have shape [3, frames, height, width]")
        if pixels.dtype != torch.uint8 or min(pixels.shape[2:]) < 1:
            raise ValueError("Source pixels must be nonempty RGB bytes")
        if frame_start != self.accepted_frames:
            raise ValueError("Source frames must be contiguous")
        if self.buffered_frames + pixels.shape[1] > self.capacity_frames:
            raise ValueError("Source pixel capacity exceeded")
        signature = (pixels.shape[0], *pixels.shape[2:], pixels.dtype, pixels.device)
        if self.accepted_frames and signature != self._signature:
            raise ValueError("Source pixel format cannot change")
        joined = (
            pixels.clone()
            if self._pixels is None
            else torch.cat((self._pixels, pixels), 1)
        )
        self._pixels = joined
        self._signature = signature
        self.accepted_frames += pixels.shape[1]

    @torch.no_grad()
    def pop(self):
        if not self.ready:
            return None
        count = min(self.clip, self.buffered_frames)
        pixels = self._pixels[:, :count]
        single = self.ended and self.accepted_frames == 1
        if count < self.clip and not single:
            pixels = torch.cat(
                (pixels, pixels[:, -1:].expand(-1, self.clip - count, -1, -1)), 1
            )
        # Do not mutate cursors or release pixels until the kernel succeeds.
        moments = self.codecs.encode_visual_moments(pixels)
        length = 1 if single else 1 + math.ceil((count - 1) / self.ratio)
        if moments.ndim != 5 or moments.shape[2] < length:
            raise ValueError("H3 encoder returned too few latent moments")
        values = moments[:, :, :length].clone()
        result = CodecSlice(
            values,
            self.emitted_latents,
            self.emitted_latents + length,
            self.consumed_frames + count,
        )
        remaining = (
            self._pixels[:, count:].clone() if count < self.buffered_frames else None
        )
        self._pixels = remaining
        self.consumed_frames += count
        self.emitted_latents += length
        return result

    def finish(self):
        if self.disposed or self.ended:
            raise ValueError("Source stream is already closed")
        if not self.accepted_frames:
            raise ValueError("Source stream requires at least one frame")
        self.ended = True

    def drain(self):
        if not self.ended:
            raise ValueError("Source stream must end before draining")
        while self.ready:
            yield self.pop()

    def dispose(self):
        self._pixels = None
        self.codecs = None
        self.disposed = True


class StableVideoDecoder:
    """Keep only the preceding H3 clip and pending bounded latent context.

    The caller supplies committed generated latents and actual source dependency
    watermarks. Pop is retryable on decode failure. Returned tensors belong to
    the caller; downstream queue capacity is a separate responsibility.
    """

    def __init__(
        self,
        codecs,
        *,
        capacity_latents,
        provenance_for: Callable[[PixelRange, int], Provenance] | None = None,
    ):
        self.codecs = codecs
        self.clip = codecs.vae_frames_per_chunk
        self.tokens = codecs.vae_latents_per_chunk
        self.drop = codecs.va.vae_token_drop
        self.overlap = (-self.drop) % self.tokens
        if not 0 <= self.drop < self.tokens:
            raise ValueError("Unsupported H3 token drop")
        model = codecs.modules["video_vae"]
        if getattr(model, "isolated_first_frame", False) or getattr(
            model, "isolated_last_frame", False
        ):
            raise ValueError("Streaming requires the standard H3 temporal layout")
        if capacity_latents < 2 * self.tokens + self.overlap:
            raise ValueError("Decoder capacity must hold preceding clip and lookahead")
        self.provenance_for = provenance_for
        self.capacity_latents = capacity_latents
        self._latent = None
        self._dependencies = []
        self.latent_start = self.accepted_latents = self.emitted_frames = 0
        self.total_frames = None
        self.disposed = False

    @property
    def buffered_latents(self):
        return len(self._dependencies)

    @property
    def ready(self):
        if self.disposed:
            return False
        if self.total_frames is not None:
            return self.emitted_frames < self.total_frames
        # With zero overlap, five latents could still be a 14..17-pixel
        # EOS tail. Wait for the next real token before committing all 17.
        lookahead = max(1, self.overlap)
        stable = max(0, (self.accepted_latents - lookahead) // self.tokens) * self.clip
        return self.emitted_frames < stable

    def append(self, latent, *, latent_start, source_dependency_end):
        if self.disposed or self.total_frames is not None:
            raise ValueError("Decoder stream is closed")
        if latent.ndim != 5 or latent.shape[2] < 1:
            raise ValueError(
                "Video latents must have shape [batch, channels, time, height, width]"
            )
        if latent_start != self.accepted_latents:
            raise ValueError("Video latents must be contiguous")
        if not isinstance(source_dependency_end, int) or source_dependency_end < 0:
            raise ValueError("Source dependency must be a nonnegative pixel watermark")
        if self.buffered_latents + latent.shape[2] > self.capacity_latents:
            raise ValueError("Decoder latent capacity exceeded")
        signature = (*latent.shape[:2], *latent.shape[3:], latent.dtype, latent.device)
        if self.accepted_latents and signature != self._signature:
            raise ValueError("Video latent format cannot change")
        joined = (
            latent.detach().clone()
            if self._latent is None
            else torch.cat((self._latent, latent.detach()), 2)
        )
        self._latent = joined
        self._signature = signature
        self._dependencies.extend([source_dependency_end] * latent.shape[2])
        self.accepted_latents += latent.shape[2]

    @torch.no_grad()
    def pop(self):
        if not self.ready:
            return None
        end = self.emitted_frames + self.clip
        if self.total_frames is not None:
            end = min(end, self.total_frames)
        offset = self.latent_start // self.tokens * self.clip
        # Limit even EOS work to the window(s) required for this output clip.
        needed = min(
            self.accepted_latents,
            (self.emitted_frames // self.clip + 1) * self.tokens + self.overlap,
        )
        window = self._latent[:, :, : needed - self.latent_start]
        values = self.codecs.decode_visual_range(
            window, self.emitted_frames - offset, end - offset
        ).clone()
        dependency = max(self._dependencies[: needed - self.latent_start])
        pixels = PixelRange(self.emitted_frames, end)
        provenance = None
        if self.provenance_for is not None:
            # The service owns alignment/credits. Resolve and validate before
            # changing codec cursors; a failed lookup can be retried safely.
            provenance = self.provenance_for(pixels, dependency)
            if (
                not isinstance(provenance, Provenance)
                or provenance.pixels != pixels
                or provenance.source_dependency_end != dependency
            ):
                raise ValueError("Provenance must match decoded pixels and dependency")
        result = DecodedVideoSlice(
            values, self.emitted_frames, end, dependency, provenance
        )
        keep = max(0, end // self.clip - 1) * self.tokens
        discard = max(0, keep - self.latent_start)
        remaining = self._latent[:, :, discard:].clone()
        self._latent = remaining
        self._dependencies = self._dependencies[discard:]
        self.latent_start += discard
        self.emitted_frames = end
        if self.total_frames == end:
            self._latent = None
            self._dependencies.clear()
        return result

    def finish(self, *, total_frames):
        if self.disposed or self.total_frames is not None:
            raise ValueError("Decoder stream is already closed")
        if (
            not isinstance(total_frames, int)
            or total_frames < 1
            or total_frames < self.emitted_frames
        ):
            raise ValueError("Invalid final video frame count")
        clips, tail = divmod(total_frames - 1, self.clip)
        expected = (
            clips * self.tokens + 1 + math.ceil(tail / self.codecs.temporal_ratio)
        )
        if expected != self.accepted_latents:
            raise ValueError("Final video frame count does not match latent count")
        self.total_frames = total_frames

    def drain(self):
        if self.total_frames is None:
            raise ValueError("Decoder stream must end before draining")
        while self.ready:
            yield self.pop()

    def dispose(self):
        self._latent = None
        self._dependencies.clear()
        self.codecs = None
        self.provenance_for = None
        self.disposed = True
