"""Compare the existing fixed-KV RoPE writer with the runtime fallback.

Run on a CUDA worker. Each mode has separate eager timing, warmed CUDA Graph
timing and an optional trace. Inputs use the strided layout returned by Ulysses.
These component timings complement the client benchmark's DiT/E2E measurements.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

# Initialize the diffusion runtime before importing its model helpers.
import sglang.multimodal_gen.runtime.models.dits.zing_h3  # noqa: F401
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _apply_rope
from sglang.multimodal_gen.runtime.models.dits.zing_h3_rope import (
    CURRENT_KV_ROUTES,
    apply_current_rope,
    apply_current_rope_kv,
)


def timed(call, *, repeats, iterations):
    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / iterations)
    return {"samples_ms": samples, "median_ms": statistics.median(samples)}


def trace_call(call, path):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        call()
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(path))
    kernels = [
        event
        for event in json.loads(path.read_text())["traceEvents"]
        if event.get("cat") == "kernel"
    ]
    return {
        "path": str(path),
        "kernel_count": len(kernels),
        "kernel_sum_us": sum(event.get("dur", 0) for event in kernels),
        "kernel_names": sorted({event["name"] for event in kernels}),
    }


@torch.inference_mode()
def benchmark(args):
    torch.manual_seed(args.seed)
    os.environ["SGLANG_ZING_FUSED_CURRENT_ROPE"] = "1"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "source_revision": args.source_revision,
        "config": vars(args),
        "hardware": {
            "name": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
        },
        "cases": [],
    }
    for rows in args.rows:
        packed = torch.randn(rows, args.heads, 384, device="cuda", dtype=torch.bfloat16)
        q, k, v = packed.chunk(3, dim=-1)
        coefficients = torch.randn(rows, 96, device="cuda", dtype=q.dtype)
        capacity = rows + 128
        stores = [
            torch.randn(capacity, args.heads, 128, device="cuda", dtype=q.dtype)
            for _ in range(3)
        ]
        raw, rotated, values = stores
        original = [tensor.clone() for tensor in stores]
        write_rows = torch.arange(64, 64 + rows, device="cuda")
        clean = rows // 3
        plan = SimpleNamespace(
            stores=((raw, values),), rotated=(rotated,), clean_rows=clean
        )
        tensors = {"write_rows": write_rows}

        def reference():
            out_q = _apply_rope(q, coefficients)
            rotated.index_copy_(0, write_rows, _apply_rope(k, coefficients))
            values.index_copy_(0, write_rows, v)
            raw.index_copy_(0, write_rows[:clean], k[:clean])
            return out_q

        expected_q = reference().clone()
        expected = [tensor.clone() for tensor in stores]
        record = {"rows": rows, "heads": args.heads, "clean_rows": clean, "modes": {}}
        for mode in ("off", "on"):
            os.environ["SGLANG_ZING_FUSED_CURRENT_ROPE_KV"] = (
                "1" if mode == "on" else "0"
            )
            CURRENT_KV_ROUTES.clear()

            def call():
                out_q = apply_current_rope_kv(q, k, v, coefficients, plan, 0, tensors)
                if out_q is not None:
                    return out_q
                # Disabling the combined writer leaves the existing standalone
                # Q and K RoPE kernels enabled, exactly as in the fixed runtime.
                out_q = apply_current_rope(q, coefficients)
                rotated.index_copy_(0, write_rows, apply_current_rope(k, coefficients))
                values.index_copy_(0, write_rows, v)
                raw.index_copy_(0, write_rows[:clean], k[:clean])
                return out_q

            for tensor, before in zip(stores, original):
                tensor.copy_(before)
            actual_q = call()
            torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
            for tensor, expected_tensor in zip(stores, expected):
                torch.testing.assert_close(tensor, expected_tensor, rtol=0, atol=0)
            if mode == "on" and CURRENT_KV_ROUTES["fused"] != 1:
                raise RuntimeError(
                    f"Fused writer did not run: {dict(CURRENT_KV_ROUTES)}"
                )
            for _ in range(20):
                call()
            torch.cuda.synchronize()
            data = {"equal": True, "routes_before_capture": dict(CURRENT_KV_ROUTES)}
            data["eager"] = timed(
                call, repeats=args.repeats, iterations=args.iterations
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_q = call()
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_q, expected_q, rtol=0, atol=0)
            data["graph"] = timed(
                graph.replay, repeats=args.repeats, iterations=args.iterations
            )
            if args.trace:
                data["eager_trace"] = trace_call(
                    call, output / f"rows-{rows}-{mode}-eager.json"
                )
                data["graph_trace"] = trace_call(
                    graph.replay, output / f"rows-{rows}-{mode}-graph.json"
                )
            record["modes"][mode] = data
            del graph, graph_q
        result["cases"].append(record)
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "rows": rows,
                    "modes": {
                        name: {
                            kind: value[kind]["median_ms"]
                            for kind in ("eager", "graph")
                        }
                        for name, value in record["modes"].items()
                    },
                }
            ),
            flush=True,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--rows", nargs="+", type=int, default=[780, 1170, 1950, 1840, 2760, 4600]
    )
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if min(*args.rows, args.heads, args.iterations, args.repeats) <= 0:
        parser.error("rows, heads, iterations and repeats must be positive")
    benchmark(args)


if __name__ == "__main__":
    main()
