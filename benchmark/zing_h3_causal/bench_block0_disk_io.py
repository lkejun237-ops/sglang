#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only Block0 raw KV IO timing with observed file-page residency.

Run on an independently verified local NVMe mount. The default shape matches
the historical SP2 maximum-prefix slot: 50 layers, 12,288 rows, 28 local heads,
128 head dimensions, BF16, K and V. This measures synthetic KV file IO into
preallocated pageable CPU tensors, not pinned allocation, H2D or model latency.
"""

import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import mmap
import os
import shutil
import statistics
import time
from pathlib import Path


def resident_fraction(path):
    """Observe Linux file-page residency without reading/faulting the pages."""
    length = path.stat().st_size
    pages = (length + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    libc.mincore.restype = ctypes.c_int
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), length, access=mmap.ACCESS_COPY) as mapping:
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            vector = (ctypes.c_ubyte * pages)()
            if libc.mincore(address, length, vector) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            return sum(value & 1 for value in vector) / pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--rows", type=int, default=12288)
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--local-heads", type=int, default=28)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if (
        min(
            args.rows,
            args.layers,
            args.local_heads,
            args.head_dim,
            args.ranks,
            args.trials,
        )
        < 1
    ):
        parser.error("Shape, rank count and trial count must be positive")

    import torch

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        disk_cache,
    )

    digest = hashlib.sha256(Path(disk_cache.__file__).read_bytes()).hexdigest()
    if digest != args.source_sha256:
        raise ValueError("Imported disk cache differs from the candidate source")
    torch.set_num_threads(1)
    shape = (args.layers, 2, args.rows, args.local_heads, args.head_dim)
    shard_bytes = 2
    for dimension in shape:
        shard_bytes *= dimension
    total_bytes = shard_bytes * args.ranks
    root = args.root.resolve()
    if shutil.disk_usage(root.parent).free < total_bytes * 2:
        raise ValueError("Insufficient free space for the isolated IO sample")
    root.mkdir(exist_ok=False)
    paths = [root / f"rank-{rank}.bin" for rank in range(args.ranks)]
    sources = [
        torch.full(shape, rank + 1, dtype=torch.bfloat16) for rank in range(args.ranks)
    ]
    # Touch target pages before timing, like the service's preallocated slots.
    targets = [torch.zeros_like(value) for value in sources]
    samples = []

    def write(rank):
        started = time.perf_counter()
        disk_cache._write_kv(paths[rank], sources[rank])
        closed = time.perf_counter()
        # Runtime publication does not promise fsync. Record it separately here
        # so DONTNEED can actually discard clean test-file pages before a read.
        with paths[rank].open("rb") as handle:
            os.fsync(handle.fileno())
        return dict(
            rank=rank,
            close_s=closed - started,
            fsync_s=time.perf_counter() - closed,
        )

    def read(rank):
        started = time.perf_counter()
        disk_cache._read_kv(paths[rank], targets[rank])
        return dict(rank=rank, seconds=time.perf_counter() - started)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.ranks) as pool:
        for trial in range(args.trials):
            started = time.perf_counter()
            writes = list(pool.map(write, range(args.ranks)))
            write_wall = time.perf_counter() - started
            if any(path.stat().st_size != shard_bytes for path in paths):
                raise AssertionError("KV file size differs from the declared shape")
            for path in paths:
                with path.open("rb") as handle:
                    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            before = [resident_fraction(path) for path in paths]
            started = time.perf_counter()
            cold_reads = list(pool.map(read, range(args.ranks)))
            cold_wall = time.perf_counter() - started
            if any(not torch.equal(a, b) for a, b in zip(sources, targets)):
                raise AssertionError("KV read differs from the written tensor")
            warm_before = [resident_fraction(path) for path in paths]
            started = time.perf_counter()
            warm_reads = list(pool.map(read, range(args.ranks)))
            warm_wall = time.perf_counter() - started
            if any(not torch.equal(a, b) for a, b in zip(sources, targets)):
                raise AssertionError("Warm KV read differs from the written tensor")
            sample = dict(
                trial=trial + 1,
                write_and_fsync_wall_s=write_wall,
                writes=writes,
                after_dontneed_resident_fraction=before,
                cold_verified=all(value <= 0.01 for value in before),
                cold_read_wall_s=cold_wall,
                cold_reads=cold_reads,
                before_warm_resident_fraction=warm_before,
                warm_verified=all(value >= 0.99 for value in warm_before),
                warm_read_wall_s=warm_wall,
                warm_reads=warm_reads,
                exact=True,
            )
            samples.append(sample)
            print(json.dumps(sample), flush=True)
    result = dict(
        ok=True,
        source_sha256=digest,
        torch_version=torch.__version__,
        shape_per_rank=shape,
        dtype="bfloat16",
        ranks=args.ranks,
        bytes_per_rank=shard_bytes,
        bytes_per_key=total_bytes,
        target="preallocated pageable CPU tensors; no GPU/H2D",
        concurrency="one IO thread per rank; simultaneous shard IO",
        samples=samples,
        median_s={
            field: statistics.median(sample[field] for sample in samples)
            for field in (
                "write_and_fsync_wall_s",
                "cold_read_wall_s",
                "warm_read_wall_s",
            )
        },
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
