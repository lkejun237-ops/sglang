# SPDX-License-Identifier: Apache-2.0
"""Bounded source provenance around the existing session-owned TAEH3 stream."""

import math

import torch

from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange

from .streaming_codecs import DecodedVideoSlice


class SourceTAEH3Decoder:
    def __init__(self, codecs, *, capacity_latents, provenance_for):
        self.stream = codecs.modules["taeh3_decoder"].new_stream()
        self.capacity_latents = capacity_latents
        self.provenance_for = provenance_for
        self.pending = None
        self.accepted_latents = self.emitted_frames = self.dependency = 0
        self.total_frames = None
        self.closed = self.drained = False

    @property
    def buffered_latents(self):
        return sum(
            value.shape[2] if value is not None else 0
            for value in (self.pending, self.stream.pending, self.stream.last_latent)
        )

    @property
    def ready(self):
        return not self.closed and (
            self.pending is not None
            or (self.total_frames is not None and not self.drained)
        )

    def append(self, latent, *, latent_start, source_dependency_end):
        if self.closed or self.total_frames is not None:
            raise ValueError("Decoder stream is closed")
        if latent_start != self.accepted_latents:
            raise ValueError("Video latents must be contiguous")
        if latent.ndim != 5 or latent.shape[1] != 24 or latent.shape[2] < 1:
            raise ValueError("TAEH3 expects nonempty N24THW latents")
        if type(source_dependency_end) is not int or source_dependency_end < 0:
            raise ValueError("Source dependency must be a nonnegative pixel watermark")
        if self.buffered_latents + latent.shape[2] > self.capacity_latents:
            raise ValueError("Decoder latent capacity exceeded")
        self.pending = (
            latent.detach().clone()
            if self.pending is None
            else torch.cat((self.pending, latent.detach()), dim=2)
        )
        self.accepted_latents += latent.shape[2]
        # Recurrent decoder state depends on the preceding stream; an absolute
        # high watermark preserves this dependency without retaining history.
        self.dependency = max(self.dependency, source_dependency_end)

    @torch.no_grad()
    def pop(self):
        if not self.ready:
            return None
        parts = []
        if self.pending is not None:
            parts.append(self.stream.push(self.pending))
            self.pending = None
        if self.total_frames is not None:
            parts.append(self.stream.finish(self.total_frames))
            self.drained = True
        values = torch.cat(parts, dim=2)
        end = self.emitted_frames + values.shape[2]
        pixels = (
            PixelRange(self.emitted_frames, end) if end > self.emitted_frames else None
        )
        provenance = self.provenance_for(pixels, self.dependency) if pixels else None
        if pixels and (
            provenance.pixels != pixels
            or provenance.source_dependency_end != self.dependency
        ):
            raise ValueError("Provenance must match decoded pixels and dependency")
        result = DecodedVideoSlice(
            values, self.emitted_frames, end, self.dependency, provenance
        )
        self.emitted_frames = end
        return result

    def finish(self, *, total_frames):
        if self.closed or self.total_frames is not None:
            raise ValueError("Decoder stream is already closed")
        if type(total_frames) is not int or total_frames < max(1, self.emitted_frames):
            raise ValueError("Invalid final video frame count")
        clips, tail = divmod(total_frames - 1, 17)
        if clips * 5 + 1 + math.ceil(tail / 4) != self.accepted_latents:
            raise ValueError("Final video frame count does not match latent count")
        self.total_frames = total_frames

    def dispose(self):
        self.stream.close()
        self.pending = self.provenance_for = None
        self.closed = True
