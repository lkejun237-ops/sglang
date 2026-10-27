# SPDX-License-Identifier: Apache-2.0
"""Private same-node transport for the experimental encoder service.

The two-rank group is separate from both encoder collectives and DiT SP. Using
the process-group methods directly avoids registering it in either default world.
Only GPU moments cross this group; source RGB originates on the host.
"""

import torch

from sglang.multimodal_gen.runtime.disaggregation.transport import nccl_bridge


def bridge_group(store, rank, *, backend="nccl", timeout_seconds=120):
    """Connect DiT leader (0) and Encoder leader (1), outside either SP world."""
    try:
        return nccl_bridge.create_bridge_group(
            store,
            rank,
            prefix="zing-encoder-bridge",
            backend=backend,
            timeout_seconds=timeout_seconds,
        )
    except ValueError as exc:
        raise ValueError("Unsupported encoder bridge backend") from exc


def receive_moments(group, shape, dtype, device):
    """Receive [1,48,T,H,W] posterior mean/logvar; the consumer samples latents.

    T is one or two latent positions. Session, interval and dependency metadata
    travel on the control channel, not in this tensor.
    """
    if len(shape) != 5 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("Invalid encoder moments shape")
    if shape[0] != 1 or shape[1] != 48 or shape[2] > 2:
        raise ValueError("Invalid progressive encoder moments dimensions")
    if dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        raise ValueError("Invalid encoder moments dtype")
    return nccl_bridge.receive_bridge_tensor(
        group,
        shape,
        dtype,
        device,
        peer=1,
        tag=0,
        expected_ndim=5,
        expected_batch=1,
        expected_channels=48,
        max_temporal=2,
        max_bytes=256 << 20,
        allowed_dtypes=(torch.float32, torch.float16, torch.bfloat16, torch.float64),
    )
