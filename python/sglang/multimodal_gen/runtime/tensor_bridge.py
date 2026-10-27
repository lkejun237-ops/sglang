# SPDX-License-Identifier: Apache-2.0
"""Two-process tensor transport outside either participant's default world.

Callers own message metadata, device selection and buffer lifetime. Work.wait()
orders CUDA work on the calling stream; synchronize that stream before handing
storage to another thread or releasing the last producer reference.
"""

from datetime import timedelta

import torch
import torch.distributed as dist


def bridge_group(store, rank, *, prefix, backend="nccl", timeout_seconds=120):
    store = dist.PrefixStore(prefix, store)
    timeout = timedelta(seconds=timeout_seconds)
    if backend == "gloo":
        return dist.ProcessGroupGloo(store, rank, 2, timeout)
    if backend != "nccl":
        raise ValueError("Unsupported tensor bridge backend")
    options = dist.ProcessGroupNCCL.Options()
    options._timeout = timeout
    return dist.ProcessGroupNCCL(store, rank, 2, options)


def send_tensor(group, value, *, peer):
    group.send([value], peer, 0).wait()


def receive_tensor(group, shape, dtype, device, *, peer):
    value = torch.empty(shape, dtype=dtype, device=device)
    receive_into(group, value, peer=peer)
    return value


def receive_into(group, value, *, peer):
    group.recv([value], peer, 0).wait()
