# SPDX-License-Identifier: Apache-2.0
"""Opt-in H3 decoder with bounded, session-owned temporal memory.

Input is diffusion-normalized NCTHW; output is RGB NCTHW in [0, 1].
The native VAE's latent statistics and ImageNet pixel transform do not apply.
"""

import hashlib
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from .upstream import TAEHV, MemBlock, SuperMemBlock, TPool

WEIGHTS_SHA256 = "af92965c2d7986a89a757e7cccd26f9eeeff0c3f0d5495eb168aeb2d6d9be9ba"
SOURCE_COMMIT = "011dfc2112197741c540e0bdd5b7b67bcc930771"


@contextmanager
def decoder_precision(dtype):
    """FP32 means IEEE convolution, with the caller's DiT flags restored."""
    if dtype != torch.float32:
        yield
        return
    backend = torch.backends.cudnn
    with backend.flags(
        enabled=backend.enabled,
        benchmark=backend.benchmark,
        benchmark_limit=backend.benchmark_limit,
        deterministic=backend.deterministic,
        allow_tf32=False,
    ):
        yield


class TAEH3Decoder(nn.Module):
    def __init__(self, checkpoint_path, expected_sha256):
        super().__init__()
        path = Path(checkpoint_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("TAEH3 checkpoint SHA256 mismatch")
        # Construction initializes parameters before loading; do not alter the
        # caller's CPU random stream (DiT seeds and audio remain independent).
        with torch.random.fork_rng(devices=[]):
            model = TAEHV(path, arch_name="taeh3")
        self.decoder = model.decoder
        self.requires_grad_(False)
        self.eval()

    def new_stream(self, *, progressive=False):
        return TAEH3Stream(self, progressive=progressive)

    @torch.no_grad()
    def decode(self, latent, frames):
        stream = self.new_stream()
        outputs = []
        # Fixed-size batches bound activation memory and make partitioning
        # independent of the caller's AR schedule.
        for chunk in latent.split(5, dim=2):
            outputs.append(stream.push(chunk))
        outputs.append(stream.finish(frames))
        result = torch.cat(outputs, dim=2)
        if result.shape[2] < frames:
            raise ValueError("TAEH3 decoded fewer frames than requested")
        return result[:, :, :frames]


class TAEH3Stream:
    """One stream per request; model weights may be shared across sessions.

    Execute each layer in temporal batches, carrying its last *input* into the
    next batch. This is the upstream parallel algorithm with a nonzero prefix
    memory. Never distribute time chunks to independent decoder instances.
    """

    def __init__(self, model, *, progressive=False):
        if type(progressive) is not bool:
            raise ValueError("TAEH3 progressive flag must be boolean")
        self.model = model
        self.progressive = progressive
        self.reset()

    def reset(self):
        self.memory = [None] * len(self.model.decoder)
        self.pending = self.tail = self.last_latent = None
        self.latents_seen = self.raw_frames = self.frames_emitted = 0
        self.raw_frame_shape = None
        self.decoded_latents = 0
        self.closed = False

    def close(self):
        self.memory = []
        self.pending = self.tail = self.last_latent = None
        self.closed = True

    def _empty(self, latent):
        return latent.new_empty(
            (latent.shape[0], 3, 0, latent.shape[3] * 16, latent.shape[4] * 16)
        )

    def _decode(self, latent):
        # Honor the explicitly selected precision even under pipeline autocast.
        with (
            decoder_precision(latent.dtype),
            torch.autocast(device_type=latent.device.type, enabled=False),
        ):
            start = self.decoded_latents
            end = start + latent.shape[2]
            with torch.profiler.record_function(f"Zing.TAEH3.decode.latents_{start}_{end}"):
                result = self._decode_layers(latent)
            self.decoded_latents = end
            return result

    def _decode_layers(self, latent):
        x = latent.transpose(1, 2).contiguous()
        n, t, c, h, w = x.shape
        x = x.reshape(n * t, c, h, w)
        for index, layer in enumerate(self.model.decoder):
            if isinstance(layer, (MemBlock, SuperMemBlock)):
                _, c, h, w = x.shape
                sequence = x.reshape(n, -1, c, h, w)
                past = self.memory[index]
                if past is None:
                    past = torch.zeros_like(sequence[:, :1])
                shifted = torch.cat((past, sequence[:, :-1]), dim=1)
                self.memory[index] = sequence[:, -1:].clone()
                x = layer(x, shifted.reshape_as(x))
            elif isinstance(layer, TPool):
                raise ValueError("TAEH3 decoder cannot contain temporal pooling")
            else:
                x = layer(x)
        return x.reshape(n, -1, *x.shape[1:])

    def _accept_raw(self, raw):
        self.raw_frame_shape = (raw.shape[0], *raw.shape[2:])
        # Prefix trim is per five-latent group, not just at stream startup.
        positions = torch.arange(raw.shape[1], device=raw.device) + self.raw_frames
        self.raw_frames += raw.shape[1]
        kept = raw[:, positions.remainder(20) >= 3]
        kept = torch.nn.functional.pixel_shuffle(kept, 2).clamp(0, 1).transpose(1, 2)
        self.tail = kept if self.tail is None else torch.cat((self.tail, kept), dim=2)
        count = max(0, self.tail.shape[2] - 12)
        output = self.tail[:, :, :count]
        self.tail = self.tail[:, :, count:].clone()
        self.frames_emitted += count
        return output

    @torch.no_grad()
    def push(self, latent):
        if self.closed:
            raise ValueError("TAEH3 stream is closed")
        if latent.ndim != 5 or latent.shape[1] != 24 or latent.shape[2] < 1:
            raise ValueError("TAEH3 expects nonempty N24THW latents")
        parameter = next(self.model.parameters())
        latent = latent.to(device=parameter.device, dtype=parameter.dtype)
        if self.last_latent is not None and (
            latent.shape[:2] + latent.shape[3:]
            != self.last_latent.shape[:2] + self.last_latent.shape[3:]
        ):
            raise ValueError("TAEH3 stream shape changed")
        self.last_latent = latent[:, :, -1:].clone()
        self.latents_seen += latent.shape[2]
        self.pending = (
            latent if self.pending is None else torch.cat((self.pending, latent), dim=2)
        )
        outputs = []
        while self.pending.shape[2] >= (1 if self.progressive else 5):
            count = min(2, self.pending.shape[2]) if self.progressive else 5
            chunk, self.pending = self.pending[:, :, :count], self.pending[:, :, count:]
            outputs.append(self._accept_raw(self._decode(chunk)))
        self.pending = self.pending.clone()
        return torch.cat(outputs, dim=2) if outputs else self._empty(latent)

    @torch.no_grad()
    def finish(self, frames):
        if self.closed or self.last_latent is None:
            raise ValueError("TAEH3 cannot finish an empty or closed stream")
        # Same request-level latent extension as native H3. Upstream then pads
        # RAW pixels to a 20-frame group before applying the final 12-frame drop.
        padded = (
            5
            if frames == 1 and self.latents_seen == 1
            else max(7, self.latents_seen + (-(self.latents_seen + 3)) % 5)
        )
        outputs = []
        if padded > self.latents_seen:
            outputs.append(
                self.push(
                    self.last_latent.expand(-1, -1, padded - self.latents_seen, -1, -1)
                )
            )
        if self.pending.shape[2]:
            raw = self._decode(self.pending)
            raw = torch.nn.functional.pad(
                raw, (0, 0, 0, 0, 0, 0, 0, -raw.shape[1] % 20)
            )
            outputs.append(self._accept_raw(raw))
        elif self.progressive and self.raw_frames % 20:
            # Small pushes already decoded the final latent. Preserve upstream
            # RAW-pixel zero padding before per-group trim and the 12-frame tail.
            raw = self.last_latent.new_zeros(
                (
                    self.raw_frame_shape[0],
                    (-self.raw_frames) % 20,
                    *self.raw_frame_shape[1:],
                )
            )
            outputs.append(self._accept_raw(raw))
        result = torch.cat(outputs, dim=2) if outputs else self._empty(self.last_latent)
        available = self.frames_emitted
        previously_emitted = available - result.shape[2]
        self.close()
        if frames < previously_emitted or frames > available:
            raise ValueError("TAEH3 requested frame count is inconsistent with latents")
        return result[:, :, : frames - previously_emitted]
