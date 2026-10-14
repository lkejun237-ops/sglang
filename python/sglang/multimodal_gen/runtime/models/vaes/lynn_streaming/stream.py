# SPDX-License-Identifier: Apache-2.0
"""Request-private Lynn stream: one decode_step per committed latent, RGB in [0, 1]."""

import torch


class LynnStream:
    """Same push/finish/close shape as TAEH3Stream.

    `frame_limit` is the known valid length of a finite request, so frames past
    it are never emitted. A failed push leaves the KV half-written; the stream
    is then closed and the generation must be disposed, never retried.
    """

    def __init__(self, model, *, frame_limit=None):
        if frame_limit is not None and (
            type(frame_limit) is not int or frame_limit < 1
        ):
            raise ValueError("Lynn frame limit must be a positive integer")
        self.model = model
        self.protocol = model.protocol
        self.frame_limit = frame_limit
        self.state = model.init_stream()
        self.latents_seen = self.frames_emitted = 0
        self.empty = None
        self.closed = False

    @property
    def produced_frames(self):
        return self.protocol.frame_count(self.latents_seen)

    def close(self):
        self.state = None
        self.closed = True

    @torch.no_grad()
    def push(self, latent):
        if self.closed:
            raise ValueError("Lynn stream is closed")
        if latent.ndim != 5 or latent.shape[2] < 1:
            raise ValueError("Lynn expects nonempty normalized N24THW latents")
        device = self.model.latent_mean.device
        latent = latent.to(device=device)
        outputs = []
        try:
            with torch.autocast(device.type, dtype=self.model.compute_dtype):
                for z_t in latent.split(1, dim=2):
                    frames, self.state = self.model.decode_step(z_t, self.state)
                    outputs.append(frames)
                    self.latents_seen += 1
        except BaseException:
            self.close()
            raise
        result = torch.cat(outputs, dim=2).clamp_(0, 1)
        self.empty = result.new_empty((*result.shape[:2], 0, *result.shape[3:]))
        if self.frame_limit is not None:
            result = result[:, :, : max(0, self.frame_limit - self.frames_emitted)]
        self.frames_emitted += result.shape[2]
        return result

    @torch.no_grad()
    def finish(self, frames):
        """Validate the final length; Lynn emits on push, so nothing is left to flush.

        Without `frame_limit` the caller owns trimming frames past `frames`.
        """
        if self.closed or not self.latents_seen:
            raise ValueError("Lynn cannot finish an empty or closed stream")
        # The last latent must contribute at least one valid frame; no latent is repeated to pad.
        if (
            not self.protocol.frame_count(self.latents_seen - 1)
            < frames
            <= self.produced_frames
        ):
            raise ValueError("Lynn requested frame count is inconsistent with latents")
        if self.frame_limit is not None and frames != self.frames_emitted:
            raise ValueError(
                "Lynn finish requires every valid frame to be pushed first"
            )
        empty = self.empty
        self.close()
        return empty

    def decode(self, latent, frames):
        """Offline path: the deployment stream over all latents, trimmed to `frames`."""
        result = self.push(latent)
        self.finish(frames)
        return result
