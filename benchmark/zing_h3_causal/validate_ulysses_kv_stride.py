# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --nproc-per-node={1,2,4}; emits per-rank raw evidence."""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist


def validate_transactions(core, k, v):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
    from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

    k, v = k[:7], v[:7]

    def coefficients(pos):
        half = pos[:, 0:1].float() * torch.arange(1, 49, device=k.device)[None] / 100
        return torch.cat((half.cos(), half.sin()), -1).bfloat16()

    results = []
    for window, sink, mode in [
        (2, 0, "block_relative"),
        (21, 3, "block_relative"),
        (None, 0, "absolute"),
    ]:
        caches = [
            ZingH3KVCache(
                1, window_policy=ZingWindowPolicy(window, sink), rope_position_mode=mode
            )
            for _ in range(2)
        ]

        def run(phase, block, rows):
            pos = torch.zeros(rows, 3, device=k.device)
            pos[:, 0] = block // 2
            pos[:, 1] = torch.arange(rows, device=k.device)
            blocks = torch.full((rows,), block, device=k.device, dtype=torch.long)
            outputs = []
            for variant, cache in enumerate(caches):
                cache.validate(pos, blocks, phase, 0 if phase == "prefix" else 1)
                mapped, history = cache.attention_positions(
                    pos, prefix=phase == "prefix"
                )
                cache.bind_attention_positions(history)
                c = coefficients(mapped)

                def write(a, b, ok, ov):
                    if variant:
                        core._apply_rope_kv_into(a, b, c, ok, ov)
                    else:
                        core._apply_rope_into(a, c, ok)
                        ov.copy_(b)

                if cache.raw_store:
                    ok, ov = cache.raw_attention_view(
                        0,
                        k[:rows],
                        v[:rows],
                        coefficients,
                        core._apply_rope_into,
                        write,
                    )
                    capture = (
                        (cache.stage_raw_k(0, k[:rows]), ov[-rows:])
                        if phase != "active"
                        else None
                    )
                else:
                    ok, ov = cache.attention_view(0, k[:rows], v[:rows], write)
                    capture = (ok[-rows:], ov[-rows:]) if phase != "active" else None
                outputs.append((ok.clone(), ov.clone()))
                cache.update(
                    pos,
                    blocks,
                    phase,
                    0 if phase == "prefix" else 1,
                    None if capture is None else (capture,),
                )
            for a, b in zip(*outputs):
                assert torch.equal(a, b), "attention cache differs"
            a, b = caches
            assert a._length == b._length and a.last_block == b.last_block
            assert torch.equal(a.positions(), b.positions())
            for x, y in zip(a._stores[0], b._stores[0]):
                assert torch.equal(x[: a._length], y[: b._length]), (
                    "authoritative cache differs"
                )

        run("prefix", 0, 3)
        evictions = 0
        for chunk in range(70):
            target = caches[0].last_block + 2
            for cache in caches:
                before = len(cache.committed)
                cache.prepare_chunk(target)
                if len(cache.committed) < before:
                    evictions += 1
            run("source", target - 1, 7)
            run("active", target, 5 if chunk == 69 else 7)
            run("active", target, 5 if chunk == 69 else 7)
            run("final", target, 5 if chunk == 69 else 7)
            if chunk == 30:
                for cache in caches:
                    cache.clear()
                    assert cache.positions() is None and not list(cache.owned_tensors())
                run("prefix", 0, 4)
        assert window is None or evictions // 2 >= 3
        for cache in caches:
            cache.prepare_chunk(cache.last_block + 2)
            pos = torch.zeros(7, 3, device=k.device)

            def fail(*args):
                raise RuntimeError("injected writer failure")

            try:
                if cache.raw_store:
                    mapped, history = cache.attention_positions(pos)
                    cache.bind_attention_positions(history)

                    cache.raw_attention_view(
                        0, k, v, coefficients, core._apply_rope_into, fail
                    )
                else:
                    cache.attention_view(0, k, v, fail)
            except RuntimeError as error:
                assert str(error) == "injected writer failure"
            finally:
                cache.clear()
            assert not list(cache.owned_tensors()) and cache.positions() is None
        results.append(
            dict(
                window=window,
                sink=sink,
                mode=mode,
                transactions=282,
                evictions_per_variant=evictions // 2,
                bitwise=True,
                prefix_rebuilt=True,
                failure_cleanup=True,
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SGLANG_ZING_EXPERIMENTAL_STRIDED_KV_WRITE") != "1":
        parser.error(
            "Set SGLANG_ZING_EXPERIMENTAL_STRIDED_KV_WRITE=1 for this experiment"
        )
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(world_size=world, rank=rank, local_rank=local)
    initialize_model_parallel(
        sequence_parallel_degree=world, ulysses_degree=world, ring_degree=1
    )
    from sglang.kernels.ops.diffusion.rope import zing_rope_kv_write as kernel
    from sglang.multimodal_gen.runtime.layers.usp import (
        _usp_input_all_to_all_varlen_qkv,
    )
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as m

    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(
        rank=rank,
        world=world,
        experimental_strided_writer=m._ZING_EXPERIMENTAL_STRIDED_KV_WRITE,
        modules={
            x.__name__: dict(
                file=x.__file__,
                sha256=hashlib.sha256(Path(x.__file__).read_bytes()).hexdigest(),
            )
            for x in [m, kernel]
        },
        samples=[],
    )
    torch.manual_seed(42)

    def timing(fn):
        for _ in range(10):
            fn()
        times = []
        for _ in range(5):
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            for _ in range(100):
                fn()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b) / 100)
        return times

    with torch.no_grad():
        for rows in [1, 52, 390, 780, 2296, 2298, 5040, 11756]:
            splits = [rows // world + (i < rows % world) for i in range(world)]
            if min(splits) == 0:
                continue
            start = sum(splits[:rank])
            stop = start + splits[rank]
            full = torch.randn(3, rows, 56, 128, device="cuda", dtype=torch.bfloat16)
            # Same tensor on all ranks: deterministic source for exact shard checks.
            dist.broadcast(full, src=0)
            q, k, v = full[:, start:stop].unbind(0)
            packed = _usp_input_all_to_all_varlen_qkv(
                q[None].contiguous(), k[None].contiguous(), v[None].contiguous(), splits
            )
            q, k, v = (x[0] for x in packed.chunk(3, -1))
            if world == 1:
                q, k, v = full.unbind(0)
            h = 56 // world
            for got, reference in zip((q, k, v), full):
                assert torch.equal(got, reference[:, rank * h : (rank + 1) * h])
            original = packed.clone()
            c = torch.randn(rows, 96, device="cuda", dtype=torch.bfloat16)
            buffers = [
                torch.full((rows + 2, h, 128), -123, device="cuda", dtype=k.dtype)
                for _ in range(2)
            ]
            ok, ov = (x[1:-1] for x in buffers)

            def eager():
                m._apply_rope_into(k, c, ok)
                ov.copy_(v)

            def direct():
                m._apply_rope_kv_into(k, v, c, ok, ov)

            def repack():
                # Local batch3 option includes the cost of packing all Q/K/V.
                p = (
                    packed.reshape(1, rows, h, 3, 128)
                    .permute(3, 0, 1, 2, 4)
                    .contiguous()
                )
                m._apply_rope_kv_into(p[1, 0], p[2, 0], c, ok, ov)

            eager()
            expected_k = ok.clone()
            expected_v = ov.clone()
            assert m._can_fuse_rope_kv_write(k, v, c, ok, ov)
            for fn in (direct, repack):
                fn()
                assert torch.equal(ok, expected_k) and torch.equal(ov, expected_v)
                assert torch.equal(packed, original)
                assert torch.equal(ok[..., 96:], k[..., 96:])
                for x in buffers:
                    assert (x[0] == -123).all() and (x[-1] == -123).all()
            record = dict(
                rows=rows,
                heads=h,
                shape=list(k.shape),
                q_stride=list(q.stride()),
                k_stride=list(k.stride()),
                v_stride=list(v.stride()),
                baseline_rejection="head_stride" if world > 1 else None,
                bitwise=True,
            )
            # Fixed route order; preserve all samples. Model A/B uses reverse order.
            record["ms"] = {}
            for name, fn in [("eager", eager), ("direct", direct), ("batch3", repack)]:
                record["ms"][name] = timing(fn)
            record["direct_speedup"] = statistics.median(
                record["ms"]["eager"]
            ) / statistics.median(record["ms"]["direct"])
            result["samples"].append(record)
            (args.output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    dict(rank=rank, rows=rows, direct_speedup=record["direct_speedup"])
                ),
                flush=True,
            )
    with torch.no_grad():
        result["transactions"] = validate_transactions(m, k, v)
    (args.output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
