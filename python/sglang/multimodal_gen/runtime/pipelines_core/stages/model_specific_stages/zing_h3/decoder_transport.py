# SPDX-License-Identifier: Apache-2.0
"""Private same-node transport for the dedicated decoder service.

The two-rank group connects the DiT leader (0) and the decoder leader (1),
outside both the DiT SP world and the decoder tile world. Only normalized
latents cross it; RGB returns through the shared-memory callback channel.
"""

import math

import torch

from sglang.multimodal_gen.runtime.tensor_bridge import bridge_group as _bridge_group
from sglang.multimodal_gen.runtime.tensor_bridge import (
    receive_tensor,
    send_tensor,
)

_BRIDGE_PREFIX = "zing-decoder-bridge"
_MAX_TRANSFER_BYTES = 256 << 20


def bridge_group(store, rank, *, backend="nccl", timeout_seconds=120):
    """Connect DiT leader (0) and decoder leader (1) on a private group."""
    return _bridge_group(
        store,
        rank,
        prefix=_BRIDGE_PREFIX,
        backend=backend,
        timeout_seconds=timeout_seconds,
    )


def check_latent(shape, dtype):
    """Validate one [1,24,T,H,W] latent that the DiT leader hands to the decoder."""
    if len(shape) != 5 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("Invalid decoder latent shape")
    if shape[0] != 1 or shape[1] != 24:
        raise ValueError("Invalid decoder latent dimensions")
    size = torch.empty((), dtype=dtype).element_size()
    if math.prod(shape) * size > _MAX_TRANSFER_BYTES:
        raise ValueError("Decoder latent exceeds transfer capacity")


def send_latent(group, latent, *, peer):
    """Send a normalized latent to `peer` from the DiT leader."""
    value = latent.detach().contiguous()
    check_latent(tuple(int(n) for n in value.shape), value.dtype)
    send_tensor(group, value, peer=peer)


def receive_latent(group, shape, dtype, device):
    """Receive a normalized latent from the DiT leader."""
    check_latent(shape, dtype)
    return receive_tensor(group, shape, dtype, device, peer=0)
