# SPDX-License-Identifier: Apache-2.0
"""Bounded source provenance around a session-owned Lynn stream."""

import torch

from sglang.multimodal_gen.runtime.realtime.zing_component_metrics import (
    record_completed_frames,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange

from .streaming_codecs import DecodedVideoSlice


def check_latent_append(
    *,
    latent,
    latent_start,
    source_dependency_end,
    valid_frame_end,
    accepted_latents,
    valid_frame_end_so_far,
    protocol,
    capacity_latents,
    buffered_latents,
):
    """Validate one append against the decoder boundary contract; return the new count."""
    if latent_start != accepted_latents:
        raise ValueError("Video latents must be contiguous")
    if latent.ndim != 5 or latent.shape[1] != 24 or latent.shape[2] < 1:
        raise ValueError("Lynn expects nonempty N24THW latents")
    if type(source_dependency_end) is not int or source_dependency_end < 0:
        raise ValueError("Source dependency must be a nonnegative pixel watermark")
    if valid_frame_end_so_far < protocol.frame_count(accepted_latents):
        raise ValueError("Decoder cannot append after a partial final latent")
    count = accepted_latents + latent.shape[2]
    if (
        type(valid_frame_end) is not int
        or not protocol.frame_count(count - 1)
        < valid_frame_end
        <= protocol.frame_count(count)
        or valid_frame_end <= valid_frame_end_so_far
        or valid_frame_end > source_dependency_end
    ):
        raise ValueError("Invalid valid output frame boundary")
    if buffered_latents + latent.shape[2] > capacity_latents:
        raise ValueError("Decoder latent capacity exceeded")
    return count


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
        count = check_latent_append(
            latent=latent,
            latent_start=latent_start,
            source_dependency_end=source_dependency_end,
            valid_frame_end=valid_frame_end,
            accepted_latents=self.accepted_latents,
            valid_frame_end_so_far=self.valid_frame_end,
            protocol=self.protocol,
            capacity_latents=self.capacity_latents,
            buffered_latents=self.buffered_latents,
        )
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
    def pop(self, *, max_latents=None):
        """Emit a legal temporal slice while retaining undecoded latents.

        Splitting is before decode_step, not after full RGB materialization.
        The unbounded default preserves existing callers and numerical order.
        """
        if max_latents is not None and (
            type(max_latents) is not int or max_latents < 1
        ):
            raise ValueError("Decoder latent quantum must be positive")
        if not self.ready:
            return None
        count = self.buffered_latents
        take = count if max_latents is None else min(count, max_latents)
        end = self.valid_frames
        if max_latents is not None:
            end = min(end, self.protocol.frame_count(self.stream.latents_seen + take))
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
                frames = self.stream.push(self.pending[:, :, :take])
            except BaseException:
                # Partially written KV cannot be retried; the generation must be disposed.
                self.closed = True
                raise
            self.pending = self.pending[:, :, take:] if take < count else None
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
        if self.total_frames is not None and self.pending is None:
            self.stream.finish(self.total_frames)
            self.held, self.drained = None, True
        result = DecodedVideoSlice(
            values, self.emitted_frames, end, self.dependency, provenance
        )
        self.emitted_frames = end
        record_completed_frames(
            "decoder", count, width=values.shape[-1], height=values.shape[-2]
        )
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
