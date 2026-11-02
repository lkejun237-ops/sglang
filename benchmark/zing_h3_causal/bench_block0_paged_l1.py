#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare dense and paged Block0 IO/restore; no model or service is launched."""

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
    _read_kv,
    _write_kv,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.paged_kv import (
    PagedKV,
    kv_slices,
)


def summary(values):
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.95
    low, high = math.floor(index), math.ceil(index)
    return {
        "n": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[low] + (ordered[high] - ordered[low]) * (index - low),
        "samples_ms": values,
    }


def measure(source, destination, stream, warmup, repeats, *, direction="h2d"):
    host, cuda, wall = [], [], []
    for trial in range(warmup + repeats):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        started = time.perf_counter()
        with torch.cuda.stream(stream):
            begin.record()
            if isinstance(source, PagedKV):
                for layer, part, start, value in source.slices():
                    device_value = destination[
                        layer, part, start : start + value.shape[0]
                    ]
                    target, origin = (
                        (device_value, value)
                        if direction == "h2d"
                        else (value, device_value)
                    )
                    target.copy_(origin, non_blocking=True)
            else:
                for layer in range(source.shape[0]):
                    for part in range(2):
                        device_value = destination[layer, part, : source.shape[2]]
                        target, origin = (
                            (device_value, source[layer, part])
                            if direction == "h2d"
                            else (source[layer, part], device_value)
                        )
                        target.copy_(origin, non_blocking=True)
            end.record()
        submitted = time.perf_counter()
        end.synchronize()
        finished = time.perf_counter()
        if trial >= warmup:
            host.append((submitted - started) * 1000)
            wall.append((finished - started) * 1000)
            cuda.append(begin.elapsed_time(end))
    return {
        "host_submit": summary(host),
        "cuda_span": summary(cuda),
        "host_to_completion": summary(wall),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[2855, 4886, 12288, 40960]
    )
    parser.add_argument("--page-rows", type=int, nargs="+", default=[2048])
    parser.add_argument("--max-prefix-rows", type=int, default=40960)
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--local-heads", type=int, default=28)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--io-dir",
        type=Path,
        required=True,
        help="Existing scratch directory; creates/removes only its own KV file",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    positive = [
        *args.rows,
        *args.page_rows,
        args.max_prefix_rows,
        args.layers,
        args.local_heads,
        args.head_dim,
        args.repeats,
    ]
    if min(positive) < 1 or args.warmup < 0 or max(args.rows) > args.max_prefix_rows:
        parser.error("Sizes/repeats must be positive and rows must fit max-prefix-rows")
    if not args.io_dir.is_dir():
        parser.error("io-dir must already exist")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    report = {
        "scope": "Per-rank component benchmark; L2 reads are warm-cache, not cold NVMe. Not E2E performance evidence.",
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "hardware": {
            "name": props.name,
            "cc": [props.major, props.minor],
            "total_bytes": props.total_memory,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "results": [],
    }
    dense = torch.empty(
        (args.layers, 2, args.max_prefix_rows, args.local_heads, args.head_dim),
        dtype=torch.bfloat16,
        pin_memory=True,
    )
    dense.fill_(1)
    stream = torch.cuda.Stream(device=device)
    import tempfile

    with tempfile.NamedTemporaryFile(
        dir=args.io_dir, prefix="block0-paged-", suffix=".bin"
    ) as handle:
        path = Path(handle.name)
        for rows in args.rows:
            source = dense[:, :, :rows]
            destination = torch.empty(
                (args.layers, 2, rows + 1, args.local_heads, args.head_dim),
                dtype=dense.dtype,
                device=device,
            )
            destination.fill_(77)
            stream.wait_stream(torch.cuda.current_stream(device))
            _write_kv(path, source)
            dense_times = measure(
                source, destination, stream, args.warmup, args.repeats
            )
            for page_rows in args.page_rows:
                count = (rows + page_rows - 1) // page_rows
                # Scattered pages with holes; the data payload never includes holes.
                pages = torch.empty(
                    (
                        args.layers,
                        2,
                        count * 2,
                        page_rows,
                        args.local_heads,
                        args.head_dim,
                    ),
                    dtype=dense.dtype,
                    pin_memory=True,
                )
                ids = list(range(0, count * 2, 2))
                random.Random(42).shuffle(ids)
                paged = PagedKV(pages, tuple(ids), rows)
                for value in kv_slices(paged):
                    value.fill_(1)
                paged_times = measure(
                    paged, destination, stream, args.warmup, args.repeats
                )
                io = {}
                for name, target in [("dense", source), ("paged", paged)]:
                    samples = []
                    for trial in range(args.warmup + args.repeats):
                        start = time.perf_counter()
                        _read_kv(path, target)
                        if trial >= args.warmup:
                            samples.append((time.perf_counter() - start) * 1000)
                    io[name] = summary(samples)
                assert bool((destination[:, :, rows, 0, 0] == 77).all())
                assert bool((destination[:, :, rows - 1, 0, 0] == 1).all())
                report["results"].append(
                    {
                        "rows": rows,
                        "page_rows": page_rows,
                        "pages": count,
                        "tail_waste_rows": count * page_rows - rows,
                        "tail_waste_fraction": 1 - rows / (count * page_rows),
                        "dense_slot_waste_fraction": 1 - rows / args.max_prefix_rows,
                        "dense_h2d": dense_times,
                        "paged_h2d": paged_times,
                        "l2_to_l1": io,
                        "paged_copy_calls": args.layers * 2 * count,
                    }
                )
                del paged, pages
            del destination
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
