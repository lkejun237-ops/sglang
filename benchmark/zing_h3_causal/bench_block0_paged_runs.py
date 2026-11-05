#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Single-cell H2D/D2H comparison: old page-major copies vs coalesced L1 runs."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from bench_block0_paged_l1 import measure

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.paged_kv import (
    PagedKV,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
    Block0PreparationCache,
)


class PageMajorKV(PagedKV):
    """Frozen pre-coalescing shape/slices from PR499 fa171b20ee."""

    @property
    def shape(self):
        return (*self.pages.shape[1:3], self.rows, *self.pages.shape[4:])

    def slices(self):
        page_rows = self.pages.shape[3]
        for layer in range(self.shape[0]):
            for part in range(2):
                for index, page in enumerate(self.page_ids):
                    start = index * page_rows
                    valid = min(page_rows, self.rows - start)
                    yield layer, part, start, self.pages[page, layer, part, :valid]


def allocation_plan(rows, page_rows, layout):
    """Use the actual allocator/LRU with tiny payloads; keep the resulting IDs fixed."""
    count = math.ceil(rows / page_rows)
    total = count * 2
    first = torch.empty((1, 1, 1), dtype=torch.bfloat16, device="cuda")
    pool = NS(
        _pool=NS(_slots=[NS(kv_cache=[NS(k=first)])]),
        plan=NS(slot_rows=max(rows, 3 * page_rows)),
    )
    cache = Block0PreparationCache(
        pool,
        max_bytes=total * 2 * page_rows * 2,
        identity={},
        disk_cache=None,
        page_rows=page_rows,
    )
    operations = []
    if layout != "contiguous":
        remaining, index = total, 0
        while remaining:
            pages = min(1 if layout == "scattered" else (2, 3)[index % 2], remaining)
            key = str(index)
            data = cache._take_pages(key, pages * page_rows)
            cache.entries[key] = NS(page_ids=data.page_ids, disk_task=None)
            operations.append({"op": "allocate", "key": key, "page_ids": data.page_ids})
            remaining -= pages
            index += 1
        for key in list(cache.entries)[::2]:
            operations.append({"op": "unload", "key": key})
            cache._release(key)
    ids = cache._take_pages("target", rows).page_ids
    operations.append({"op": "allocate", "key": "target", "page_ids": ids})
    return {
        "total_pages": total,
        "page_ids": ids,
        "operations": operations,
        "evictions": cache.evictions,
    }


def pattern(rows, layer, part, *, device="cpu", offset=0):
    return (
        ((torch.arange(rows, device=device) + layer * 3 + part * 11 + offset) % 127)
        .to(torch.bfloat16)
        .reshape(rows, 1, 1)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--page-rows", type=int, default=2048)
    p.add_argument(
        "--layout", choices=["contiguous", "churn", "scattered"], required=True
    )
    p.add_argument("--variant", choices=["page-major", "coalesced"], required=True)
    p.add_argument("--layers", type=int, default=50)
    p.add_argument("--local-heads", type=int, default=28)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        min(a.rows, a.page_rows, a.layers, a.local_heads, a.head_dim, a.repeats) < 1
        or a.warmup < 0
    ):
        p.error("Sizes/repeats must be positive and warmup nonnegative")
    plan = allocation_plan(a.rows, a.page_rows, a.layout)
    ids, total = tuple(plan["page_ids"]), plan["total_pages"]
    shape = (a.layers, 2, total, a.page_rows, a.local_heads, a.head_dim)
    if a.variant == "page-major":
        shape = (total, a.layers, 2, a.page_rows, a.local_heads, a.head_dim)
    pages = torch.full(shape, float("nan"), dtype=torch.bfloat16, pin_memory=True)
    cls = PageMajorKV if a.variant == "page-major" else PagedKV
    data = cls(pages, ids, a.rows)
    for layer, part, start, value in data.slices():
        value.copy_(pattern(a.rows, layer, part)[start : start + len(value)])
    destination = torch.full(
        (a.layers, 2, a.rows + 1, a.local_heads, a.head_dim),
        77,
        dtype=torch.bfloat16,
        device="cuda",
    )
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    h2d = measure(data, destination, stream, a.warmup, a.repeats)
    for layer in range(a.layers):
        for part in range(2):
            assert bool(
                (
                    destination[layer, part, : a.rows]
                    == pattern(a.rows, layer, part, device="cuda")
                ).all()
            )
            destination[layer, part, : a.rows].copy_(
                pattern(a.rows, layer, part, device="cuda", offset=17)
            )
    assert bool((destination[:, :, a.rows] == 77).all())
    for _, _, _, value in data.slices():
        value.fill_(float("nan"))
    torch.cuda.synchronize()
    d2h = measure(data, destination, stream, a.warmup, a.repeats, direction="d2h")
    for layer, part, start, value in data.slices():
        assert bool(
            (
                value
                == pattern(a.rows, layer, part, offset=17)[start : start + len(value)]
            ).all()
        )
    # Check every unused page and the tail; D2H must never write padding/other owners.
    for page in range(total):
        value = pages[page] if a.variant == "page-major" else pages[:, :, page]
        if page not in ids:
            assert bool(value.isnan().all())
    tail = a.rows % a.page_rows
    if tail:
        value = pages[ids[-1]] if a.variant == "page-major" else pages[:, :, ids[-1]]
        assert bool(value[:, :, tail:].isnan().all())
    props = torch.cuda.get_device_properties(0)
    result = {
        "scope": "Per-rank H2D/D2H component; not model/E2E. Each cell needs a fresh process.",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "hardware": {
            "name": props.name,
            "cc": [props.major, props.minor],
            "total_bytes": props.total_memory,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "allocation_plan": plan,
        "run_count": 1 + sum(b != c + 1 for c, b in zip(ids, ids[1:])),
        "copy_calls": sum(1 for _ in data.slices()),
        "payload_bytes": data.numel() * data.element_size(),
        "h2d": h2d,
        "d2h": d2h,
        "verified": {
            "all_h2d_values": True,
            "all_d2h_values": True,
            "unused_pages": True,
            "tail_padding": True,
            "device_guard": True,
        },
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["run_count", "copy_calls", "payload_bytes", "verified"]
            }
        )
    )


if __name__ == "__main__":
    main()
