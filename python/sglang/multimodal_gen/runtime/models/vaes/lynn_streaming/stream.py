# SPDX-License-Identifier: Apache-2.0
"""Request-private Lynn stream: one decode_step per committed latent, RGB in [0, 1]."""

import traceback

import torch


class LynnStream:
    """Session-owned push/finish/close streaming interface.

    `frame_limit` is the known valid length of a finite request, so frames past
    it are never emitted. A failed push leaves the KV half-written; the stream
    is then closed and the generation must be disposed, never retried.

    `group` is the SP group whose ranks split the tiles of every decode_step;
    None decodes all tiles on this rank. Ownership is fixed for the stream.
    """

    def __init__(self, model, *, frame_limit=None, group=None):
        if frame_limit is not None and (
            type(frame_limit) is not int or frame_limit < 1
        ):
            raise ValueError("Lynn frame limit must be a positive integer")
        self.model = model
        self.protocol = model.protocol
        self.frame_limit = frame_limit
        self.group = group
        self.state = model.init_stream()
        self.latents_seen = self.frames_emitted = 0
        self.empty = None
        self.closed = False

    @property
    def produced_frames(self):
        return self.protocol.frame_count(self.latents_seen)

    def close(self):
        if self.state is not None:
            # A failure traceback may still refer to the state after close.
            self.state.pixel_send = self.state.pixel_recv = None
        self.state = None
        self.closed = True

    @torch.no_grad()
    def push(self, latent):
        if self.closed:
            raise ValueError("Lynn stream is closed")
        if latent.ndim != 5 or latent.shape[2] < 1:
            raise ValueError("Lynn expects nonempty normalized N24THW latents")
        try:
            device = self.model.latent_mean.device
            latent = latent.to(device=device)
            outputs = []
            with torch.autocast(device.type, dtype=self.model.compute_dtype):
                for z_t in latent.split(1, dim=2):
                    frames, self.state = self.model.decode_step(
                        z_t, self.state, self.group
                    )
                    outputs.append(frames)
                    self.latents_seen += 1
            result = torch.cat(outputs, dim=2).clamp_(0, 1)
            self.empty = result.new_empty((*result.shape[:2], 0, *result.shape[3:]))
            if self.frame_limit is not None:
                result = result[:, :, : max(0, self.frame_limit - self.frames_emitted)]
            self.frames_emitted += result.shape[2]
            return result
        except BaseException as exc:
            self.close()
            # Completed decoder frames may own views into the pixel workspace.
            # Release their locals even when the caller retains this exception;
            # keep the original exception and traceback locations for diagnosis.
            pending, seen = [exc], set()
            while pending:
                error = pending.pop()
                if id(error) in seen:
                    continue
                seen.add(id(error))
                # Wrappers can retain independent frames through either link;
                # preserve both links while handling even cyclic chains once.
                pending.extend(
                    linked
                    for linked in (error.__cause__, error.__context__)
                    if linked is not None
                )
                traceback.clear_frames(error.__traceback__)
            raise

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
