# SPDX-License-Identifier: Apache-2.0
"""Opt-in indexed noise for unknown-length video; not legacy native parity.

The CPU float32 draw, hash encoding and per-frame shape are part of v1.
Record the PyTorch version and geometry with the request for replay.
"""

import hashlib

import torch

STREAMING_NOISE_VERSION = "indexed-cpu-v1"


def indexed_video_noise(
    shape,
    *,
    seed,
    latent_start,
    purpose,
    device,
    dtype,
    version=STREAMING_NOISE_VERSION,
):
    """Return [B,C,T,H,W] noise without reading or advancing global RNG.

    Source posterior and target initial noise use independent domains. Memory
    is proportional to this requested slice, never the preceding time horizon.
    """
    if version != STREAMING_NOISE_VERSION:
        raise ValueError("Unsupported streaming noise version")
    if purpose not in ("source_posterior", "target_initial"):
        raise ValueError("Unsupported streaming noise purpose")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    if type(latent_start) is not int or latent_start < 0:
        raise ValueError("Latent start must be a nonnegative integer")
    if len(shape) != 5 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("Noise shape must be positive [B,C,T,H,W]")
    if not dtype.is_floating_point:
        raise ValueError("Noise dtype must be floating point")
    batch, channels, frames, height, width = shape
    noise = torch.empty(shape, dtype=torch.float32, device="cpu")
    for offset in range(frames):
        key = f"{version}:{purpose}:{seed}:{latent_start + offset}".encode("ascii")
        frame_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        generator = torch.Generator(device="cpu").manual_seed(frame_seed)
        noise[:, :, offset] = torch.randn(
            (batch, channels, height, width), generator=generator, dtype=torch.float32
        )
    return noise.to(device=device, dtype=dtype)
