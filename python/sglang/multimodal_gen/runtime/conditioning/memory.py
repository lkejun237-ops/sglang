# SPDX-License-Identifier: Apache-2.0
"""Physical tensor-storage receipts for explicit conditioning memory audits.

Owner totals may overlap. Only storage_union_bytes/device_bytes are additive.
CUDA allocator allocated and reserved are separate observations, never summed.
This helper does not synchronize or reset peaks; benchmark callers control that.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import torch


def _tensors(value, seen):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if value is None or id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.nn.Module):
        yield from value.parameters(recurse=True)
        yield from value.buffers(recurse=True)
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensors(child, seen)
    elif isinstance(value, (tuple, list, deque)):
        for child in value:
            yield from _tensors(child, seen)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _tensors(getattr(value, field.name), seen)
    elif hasattr(value, "__dict__"):
        for child in vars(value).values():
            yield from _tensors(child, seen)


def memory_receipt(owners, *, phase, rank, allocator=None):
    """Return a JSON-compatible union of storages reachable from named owners.

    Callers name condition current/pending/clones, KV, codec and graph pools
    explicitly. Tensor views retain the full physical backing storage size.
    ``allocator`` permits frozen CPU tests or externally collected GPU counters.
    """
    storages = {}
    for owner, value in sorted(owners.items()):
        for tensor in _tensors(value, set()):
            if tensor.device.type == "meta":
                continue
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage._cdata)
            entry = storages.setdefault(
                key,
                {
                    "bytes": storage.nbytes(),
                    "device": str(tensor.device),
                    "owners": set(),
                    "dtypes": set(),
                },
            )
            entry["owners"].add(owner)
            entry["dtypes"].add(str(tensor.dtype))
    device_bytes, owner_bytes, records = {}, {}, []
    for entry in storages.values():
        device = entry["device"]
        device_bytes[device] = device_bytes.get(device, 0) + entry["bytes"]
        for owner in entry["owners"]:
            owner_bytes[owner] = owner_bytes.get(owner, 0) + entry["bytes"]
        records.append(
            {
                **entry,
                "owners": sorted(entry["owners"]),
                "dtypes": sorted(entry["dtypes"]),
            }
        )
    if allocator is None:
        allocator = {}
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                allocator[f"cuda:{index}"] = {
                    "allocated": torch.cuda.memory_allocated(index),
                    "reserved": torch.cuda.memory_reserved(index),
                    "peak_allocated": torch.cuda.max_memory_allocated(index),
                    "peak_reserved": torch.cuda.max_memory_reserved(index),
                }
    observed = {}
    for device, counters in allocator.items():
        observed[device] = {
            **counters,
            "unattributed_allocated": max(
                0, counters["allocated"] - device_bytes.get(device, 0)
            ),
        }
    return {
        "phase": phase,
        "rank": rank,
        "storage_union_bytes": sum(device_bytes.values()),
        "device_bytes": dict(sorted(device_bytes.items())),
        "owner_bytes": dict(sorted(owner_bytes.items())),
        "storages": sorted(
            records,
            key=lambda entry: (entry["device"], entry["owners"], entry["bytes"]),
        ),
        "allocator": observed,
    }
