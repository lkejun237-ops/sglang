# SPDX-License-Identifier: Apache-2.0
"""Private same-node transport for the experimental encoder service.

The two-rank group is separate from both encoder collectives and DiT SP. Using
the process-group methods directly avoids registering it in either default world.
Only GPU moments cross this group; source RGB originates on the host.
"""

from datetime import timedelta

import torch
import torch.distributed as dist


def bridge_group(store, rank, *, backend="nccl", timeout_seconds=120):
    """Connect DiT leader (0) and Encoder leader (1), outside either SP world."""
    store = dist.PrefixStore("zing-encoder-bridge", store)
    timeout = timedelta(seconds=timeout_seconds)
    if backend == "gloo":
        return dist.ProcessGroupGloo(store, rank, 2, timeout)
    if backend != "nccl":
        raise ValueError("Unsupported encoder bridge backend")
    options = dist.ProcessGroupNCCL.Options()
    options._timeout = timeout
    return dist.ProcessGroupNCCL(store, rank, 2, options)


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
    size = torch.empty((), dtype=dtype).element_size()
    import math

    if math.prod(shape) * size > 256 << 20:
        raise ValueError("Encoder moments exceed transfer capacity")
    value = torch.empty(shape, dtype=dtype, device=device)
    group.recv([value], 1, 0).wait()
    return value
