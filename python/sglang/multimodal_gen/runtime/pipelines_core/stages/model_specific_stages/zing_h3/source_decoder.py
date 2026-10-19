# SPDX-License-Identifier: Apache-2.0
"""Bounded source provenance around a session-owned Lynn stream."""

import torch

from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange

from .streaming_codecs import DecodedVideoSlice


class SourceLynnDecoder:
    """Bounded source provenance around a session-owned Lynn stream.

    Each append declares the actual output pixel boundary independently of
    source dependencies. Lynn has no lookahead: pop emits every valid frame,
    discards only short-tail padding, and retains no RGB waiting for EOS.
    """

    def __init__(self, codecs, *, capacity_latents, provenance_for):
        module = codecs.modules[codecs.decoder.module_key]
        self.stream = codecs.decoder.new_stream(module, frame_limit=None)
        self.protocol = module.protocol
        self.capacity_latents = capacity_latents
        self.provenance_for = provenance_for
        self.pending = self.held = None
        self.accepted_latents = self.emitted_frames = self.dependency = 0
        self.total_frames = None
        self.valid_frame_end = 0
        self.closed = self.drained = False

    @property
    def buffered_latents(self):
        return 0 if self.pending is None else self.pending.shape[2]

    @property
    def valid_frames(self):
        return self.valid_frame_end

    @property
    def ready(self):
        return not self.closed and (
            self.pending is not None
            or self.valid_frames > self.emitted_frames
            or (self.total_frames is not None and not self.drained)
        )

    def append(self, latent, *, latent_start, source_dependency_end, valid_frame_end):
        if self.closed or self.total_frames is not None:
            raise ValueError("Decoder stream is closed")
        if latent_start != self.accepted_latents:
            raise ValueError("Video latents must be contiguous")
        if latent.ndim != 5 or latent.shape[1] != 24 or latent.shape[2] < 1:
            raise ValueError("Lynn expects nonempty N24THW latents")
        if type(source_dependency_end) is not int or source_dependency_end < 0:
            raise ValueError("Source dependency must be a nonnegative pixel watermark")
        if self.valid_frame_end < self.protocol.frame_count(self.accepted_latents):
            raise ValueError("Decoder cannot append after a partial final latent")
        count = self.accepted_latents + latent.shape[2]
        if (
            type(valid_frame_end) is not int
            or not self.protocol.frame_count(count - 1)
            < valid_frame_end
            <= self.protocol.frame_count(count)
            or valid_frame_end <= self.valid_frame_end
            or valid_frame_end > source_dependency_end
        ):
            raise ValueError("Invalid valid output frame boundary")
        if self.buffered_latents + latent.shape[2] > self.capacity_latents:
            raise ValueError("Decoder latent capacity exceeded")
        self.pending = (
            latent.detach().clone()
            if self.pending is None
            else torch.cat((self.pending, latent.detach()), dim=2)
        )
        self.accepted_latents = count
        self.valid_frame_end = valid_frame_end
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
        # Empty shape metadata supports an EOS pop after the last RGB was
        # already delivered; clone avoids retaining the decoded allocation.
        self.held = self.held[:, :, :0].clone()
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
        if not self.accepted_latents or total_frames != self.valid_frame_end:
            raise ValueError("Final video frame count does not match latent count")
        self.total_frames = total_frames

    def dispose(self):
        self.stream.close()
        self.pending = self.held = self.provenance_for = None
        self.closed = True
