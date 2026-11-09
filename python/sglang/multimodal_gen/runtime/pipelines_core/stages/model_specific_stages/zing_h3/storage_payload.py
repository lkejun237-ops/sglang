# SPDX-License-Identifier: Apache-2.0
"""Conservative byte bounds without copying large conditioning tensors."""

from dataclasses import fields, is_dataclass

import torch


def serialization_bound(value, *, deduplicate=True):
    # torch.save serializes complete underlying CPU storages, including views.
    # Count those storages once and leave bounded pickle/ZIP metadata headroom.
    storages = set()
    nodes = 0

    def measure(item, depth=0):
        nonlocal nodes
        nodes += 1
        if depth > 64 or nodes > 100000:
            raise ValueError("Conditioning metadata exceeds serialization bounds")
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            identity = (str(item.device), storage.data_ptr(), storage.nbytes())
            if deduplicate and identity in storages:
                return 0
            storages.add(identity)
            return storage.nbytes() + 4096
        if is_dataclass(item) and not isinstance(item, type):
            return 256 + sum(
                measure(getattr(item, f.name), depth + 1) for f in fields(item)
            )
        if isinstance(item, torch.dtype):
            return 256
        if isinstance(item, dict):
            return 256 + sum(
                measure(k, depth + 1) + measure(v, depth + 1) for k, v in item.items()
            )
        if isinstance(item, (list, tuple)):
            return 256 + sum(measure(v, depth + 1) for v in item)
        if isinstance(item, str):
            return len(item.encode()) + 256
        if isinstance(item, bytes):
            return len(item) + 256
        if item is None or isinstance(item, (bool, int, float)):
            return 256
        raise TypeError("Unsupported conditioning serialization value")

    size = measure(value)
    return size + size // 100 + 65536
