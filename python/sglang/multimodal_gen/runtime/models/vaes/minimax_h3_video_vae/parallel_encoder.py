# SPDX-License-Identifier: Apache-2.0
"""Bind the shared progressive encoder to a dedicated worker process group."""

from types import SimpleNamespace

import torch.distributed as dist

from .progressive import CausalEncoderCache


def create_encoder_cache(model, group, *, control_group, encode_mode, encode_dtype):
    size = dist.get_world_size(group)
    if size not in (1, 2, 4):
        raise ValueError("Encoder parallel size must be 1, 2, or 4")
    if dist.get_backend(control_group) != "gloo":
        raise ValueError("Encoder control group must use Gloo")
    ranks = dist.get_process_group_ranks(group)
    if dist.get_process_group_ranks(control_group) != ranks:
        raise ValueError("Encoder data and control groups must have identical ranks")
    coordinator = SimpleNamespace(
        rank_in_group=dist.get_rank(group),
        world_size=size,
        ranks=ranks,
        device_group=group,
        cpu_group=control_group,
    )
    return CausalEncoderCache(
        model,
        group=coordinator,
        encode_mode=encode_mode,
        encode_dtype=encode_dtype,
    )
