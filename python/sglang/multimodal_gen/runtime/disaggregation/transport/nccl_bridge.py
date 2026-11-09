# SPDX-License-Identifier: Apache-2.0
"""Private same-node 2-rank bridge transport for disaggregated streaming components."""

from __future__ import annotations

import math
import time
from datetime import timedelta
from typing import Sequence

import torch
import torch.distributed as dist


def wait_bridge_probe(tensor: torch.Tensor, *, timeout_seconds: float) -> None:
    """Confirm a startup probe on the CPU without synchronizing the whole GPU.

    Call on the same thread/stream immediately after send/recv Work.wait().
    Keep the probe tensor alive until the stream event proves completion.
    This is a connection readiness boundary, not a per-frame transfer fence.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Bridge probe timeout must be finite and positive")
    if not tensor.is_cuda:
        return
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(tensor.device))
    deadline = time.monotonic() + timeout_seconds
    while not event.query():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("NCCL bridge startup probe did not complete")
        time.sleep(min(0.001, remaining))


def create_bridge_group(
    store: dist.Store,
    rank: int,
    *,
    prefix: str,
    backend: str = "nccl",
    timeout_seconds: int = 120,
) -> dist.ProcessGroup:
    """Connect two ranks on a private ProcessGroup outside the default world."""
    store = dist.PrefixStore(prefix, store)
    timeout = timedelta(seconds=timeout_seconds)
    if backend == "gloo":
        return dist.ProcessGroupGloo(store, rank, 2, timeout)
    if backend == "nccl":
        options = dist.ProcessGroupNCCL.Options()
        options._timeout = timeout
        return dist.ProcessGroupNCCL(store, rank, 2, options)
    raise ValueError(f"Unsupported bridge backend: {backend}")


def validate_bridge_tensor(
    shape: Sequence[int],
    dtype: torch.dtype,
    *,
    expected_ndim: int | None = 5,
    expected_batch: int | None = 1,
    expected_channels: int | None = None,
    max_temporal: int | None = None,
    max_bytes: int = 256 << 20,
    allowed_dtypes: Sequence[torch.dtype] | None = (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
    ),
) -> None:
    """Validate shape, dtype, and capacity bounds for bridge transfers."""
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) == 0
        or any(type(n) is not int or n < 1 for n in shape)
    ):
        raise ValueError("Invalid bridge tensor shape")
    if expected_ndim is not None and len(shape) != expected_ndim:
        raise ValueError("Invalid bridge tensor shape ndim")
    if expected_batch is not None and shape[0] != expected_batch:
        raise ValueError("Invalid bridge tensor batch dimensions")
    if expected_channels is not None and shape[1] != expected_channels:
        raise ValueError("Invalid bridge tensor channel dimensions")
    if max_temporal is not None and shape[2] > max_temporal:
        raise ValueError("Invalid bridge tensor temporal dimensions")
    if allowed_dtypes is not None and dtype not in allowed_dtypes:
        raise ValueError("Invalid bridge tensor dtype")
    element_size = torch.empty((), dtype=dtype).element_size()
    total_bytes = math.prod(shape) * element_size
    if total_bytes > max_bytes:
        raise ValueError("Bridge tensor exceeds transfer capacity")


def send_bridge_tensor(
    group: dist.ProcessGroup,
    tensor: torch.Tensor,
    *,
    peer: int = 1,
    tag: int = 0,
    **validate_kwargs,
) -> None:
    """Validate and synchronously send a contiguous tensor across the bridge."""
    if validate_kwargs:
        validate_bridge_tensor(
            tuple(int(n) for n in tensor.shape), tensor.dtype, **validate_kwargs
        )
    value = tensor.detach().contiguous()
    group.send([value], peer, tag).wait()


def receive_bridge_tensor(
    group: dist.ProcessGroup,
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device | str,
    *,
    peer: int = 0,
    tag: int = 0,
    **validate_kwargs,
) -> torch.Tensor:
    """Validate and synchronously receive a tensor across the bridge."""
    if validate_kwargs:
        validate_bridge_tensor(shape, dtype, **validate_kwargs)
    value = torch.empty(shape, dtype=dtype, device=device)
    group.recv([value], peer, tag).wait()
    return value


def abort_bridge_group(group):
    """Abort a failed bridge; never drain unmatched operations as a fallback."""
    abort = getattr(group, "abort", None)
    if abort is None:
        raise RuntimeError("Bridge abort is unavailable; worker restart is required")
    abort()
