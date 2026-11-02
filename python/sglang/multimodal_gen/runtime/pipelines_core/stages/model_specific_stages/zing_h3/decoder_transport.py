# SPDX-License-Identifier: Apache-2.0
"""Private same-node transport for the dedicated decoder service.

The two-rank group connects the DiT leader (0) and the decoder leader (1),
outside both the DiT SP world and the decoder tile world. Only normalized
latents cross it; RGB returns through the shared-memory callback channel.
"""

from sglang.multimodal_gen.runtime.disaggregation.transport import nccl_bridge

_BRIDGE_PREFIX = "zing-decoder-bridge"
_MAX_TRANSFER_BYTES = 256 << 20


def bridge_group(store, rank, *, backend="nccl", timeout_seconds=120, prefix=None):
    """Connect DiT leader (0) and decoder leader (1) on a private group."""
    try:
        return nccl_bridge.create_bridge_group(
            store,
            rank,
            prefix=prefix or _BRIDGE_PREFIX,
            backend=backend,
            timeout_seconds=timeout_seconds,
        )
    except ValueError as exc:
        raise ValueError("Unsupported decoder bridge backend") from exc


def check_latent(shape, dtype):
    """Validate one [1,24,T,H,W] latent that the DiT leader hands to the decoder."""
    if len(shape) != 5 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("Invalid decoder latent shape")
    if shape[0] != 1 or shape[1] != 24:
        raise ValueError("Invalid decoder latent dimensions")
    nccl_bridge.validate_bridge_tensor(
        shape,
        dtype,
        expected_ndim=5,
        expected_batch=1,
        expected_channels=24,
        max_bytes=_MAX_TRANSFER_BYTES,
    )


def send_latent(group, latent, *, peer):
    """Send a normalized latent to `peer` from the DiT leader."""
    check_latent(tuple(int(n) for n in latent.shape), latent.dtype)
    nccl_bridge.send_bridge_tensor(group, latent, peer=peer, tag=0)


def receive_latent(group, shape, dtype, device):
    """Receive a normalized latent from the DiT leader."""
    check_latent(shape, dtype)
    return nccl_bridge.receive_bridge_tensor(group, shape, dtype, device, peer=0, tag=0)
