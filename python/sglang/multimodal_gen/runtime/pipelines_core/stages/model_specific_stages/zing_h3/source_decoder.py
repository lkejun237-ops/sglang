# SPDX-License-Identifier: Apache-2.0
"""Bounded source provenance around the existing session-owned TAEH3 stream."""

import math
import os

import torch

from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange

from .streaming_codecs import DecodedVideoSlice


class SourceTAEH3Decoder:
    def __init__(self, codecs, *, capacity_latents, provenance_for):
        progressive = os.environ.get("SGLANG_ZING_EXPERIMENTAL_PROGRESSIVE_DECODE", "0")
        if progressive not in ("0", "1"):
            raise ValueError(
                "SGLANG_ZING_EXPERIMENTAL_PROGRESSIVE_DECODE must be 0 or 1"
            )
        decoder = codecs.modules["taeh3_decoder"]
        self.stream = (
            decoder.new_stream(progressive=True)
            if progressive == "1"
            else decoder.new_stream()
        )
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


class SourceLynnDecoder:
    """Bounded source provenance around a session-owned Lynn stream.

    Every pop runs decode_step on all pending latents. Until EOS the last
    latent may still cover padded source frames, so its at most four frames
    are held back; only frames proven valid by a later latent or by
    finish(total_frames) are delivered, never emitted and withdrawn.
    """

    def __init__(self, codecs, *, capacity_latents, provenance_for):
        module = codecs.modules[codecs.decoder.module_key]
        self.stream = module.new_stream()
        self.protocol = module.protocol
        self.capacity_latents = capacity_latents
        self.provenance_for = provenance_for
        self.pending = self.held = None
        self.accepted_latents = self.emitted_frames = self.dependency = 0
        self.total_frames = None
        self.closed = self.drained = False

    @property
    def buffered_latents(self):
        return 0 if self.pending is None else self.pending.shape[2]

    @property
    def valid_frames(self):
        if self.total_frames is not None:
            return self.total_frames
        return self.protocol.frame_count(max(0, self.accepted_latents - 1))

    @property
    def ready(self):
        return not self.closed and (
            self.pending is not None
            or self.valid_frames > self.emitted_frames
            or (self.total_frames is not None and not self.drained)
        )

    def append(self, latent, *, latent_start, source_dependency_end):
        if self.closed or self.total_frames is not None:
            raise ValueError("Decoder stream is closed")
        if latent_start != self.accepted_latents:
            raise ValueError("Video latents must be contiguous")
        if latent.ndim != 5 or latent.shape[1] != 24 or latent.shape[2] < 1:
            raise ValueError("Lynn expects nonempty N24THW latents")
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
        # Attention history chains through every earlier latent, so keep the
        # cumulative high watermark rather than inferring a W-latent bound.
        self.dependency = max(self.dependency, source_dependency_end)

    @torch.no_grad()
    def pop(self):
        if not self.ready:
            return None
        end = self.valid_frames
        pixels = (
            PixelRange(self.emitted_frames, end) if end > self.emitted_frames else None
        )
        # Check provenance before the irreversible decode_step.
        provenance = self.provenance_for(pixels, self.dependency) if pixels else None
        if pixels and (
            provenance.pixels != pixels
            or provenance.source_dependency_end != self.dependency
        ):
            raise ValueError("Provenance must match decoded pixels and dependency")
        if self.pending is not None:
            try:
                frames = self.stream.push(self.pending)
            except BaseException:
                # Partially written KV cannot be retried; the generation must be disposed.
                self.closed = True
                raise
            self.pending = None
            self.held = (
                frames if self.held is None else torch.cat((self.held, frames), dim=2)
            )
        count = end - self.emitted_frames
        if self.held is None or self.held.shape[2] < count:
            raise ValueError("Lynn decoded fewer frames than the source provides")
        values = self.held[:, :, :count]
        self.held = self.held[:, :, count:].clone()
        if self.total_frames is not None:
            self.stream.finish(self.total_frames)
            self.held, self.drained = None, True
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
        count = self.accepted_latents
        if not count or not self.protocol.frame_count(
            count - 1
        ) < total_frames <= self.protocol.frame_count(count):
            raise ValueError("Final video frame count does not match latent count")
        self.total_frames = total_frames

    def dispose(self):
        self.stream.close()
        self.pending = self.held = self.provenance_for = None
        self.closed = True
