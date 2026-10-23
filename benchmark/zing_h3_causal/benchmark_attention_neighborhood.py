"""SP2/4 A/B for QK norm, input exchange, history RoPE and current KV writes.

Run with torchrun on CUDA workers. These are component measurements: a common
projection-buffer clone supplies fresh inputs on every replay, and neither
projection GEMM nor Attention is timed. Full-model results use the client bench.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from benchmark_current_rope_kv import timed, trace_call

from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.usp import _usp_input_all_to_all_varlen_qkv
from sglang.multimodal_gen.runtime.models.dits.zing_h3_rope import (
    apply_current_qknorm_rope_kv,
    apply_current_rope,
    apply_current_rope_kv,
    apply_history_rope_into,
)

VARIANTS = ("baseline", "writer", "sp", "norm", "live")


@torch.inference_mode()
def run(args):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_distributed_environment(world, rank, local_rank=rank, backend="nccl")
    initialize_model_parallel(sequence_parallel_degree=world, ulysses_degree=world)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    torch.manual_seed(123)
    module = SimpleNamespace(head_dim=128)
    for name in ("q_norm", "k_norm"):
        norm = torch.nn.RMSNorm(128, eps=1e-6, device="cuda", dtype=torch.bfloat16)
        norm.weight.copy_(torch.randn_like(norm.weight))
        setattr(module, name, norm)
    heads = 56 // world
    # The first cases also exercise uneven and empty source shards with NCCL.
    cases = [("uneven", [5, 3] if world == 2 else [3, 0, 5, 2], 7, 32)]
    if world == 2:
        cases.append(("empty", [0, 7], 7, 32))
    for grid in (390, 920):
        for window in args.windows:
            for latent in (2, 3, 5) if window == 6 else (3,):
                rows = grid * latent
                splits = [rows // world + (r < rows % world) for r in range(world)]
                # Explicit synthetic prefix and pool budget; not an E2E estimate.
                history = 4096 + window * 3 * grid
                capacity = 15360 + (window + 2) * 3 * grid
                cases.append(
                    (f"grid{grid}-w{window}-t{latent}", splits, history, capacity)
                )
    for name, splits, history, capacity in cases:
        rows = sum(splits)
        seed = torch.randn(
            splits[rank], 3 * 56 * 128, device="cuda", dtype=torch.bfloat16
        )
        coefficients = torch.randn(rows, 96, device="cuda", dtype=seed.dtype)
        history_coeff = torch.randn(capacity, 96, device="cuda", dtype=seed.dtype)
        raw, rotated, values = [
            torch.randn(capacity, heads, 128, device="cuda", dtype=seed.dtype)
            for _ in range(3)
        ]
        write_rows = torch.arange(history, history + rows, device="cuda")
        live = torch.tensor([history], dtype=torch.int32, device="cuda")
        clean = rows // 3
        plan = SimpleNamespace(
            stores=((raw, values),), rotated=(rotated,), clean_rows=clean
        )
        tensors = {"write_rows": write_rows}
        record = dict(
            name=name, splits=splits, history=history, capacity=capacity, modes={}
        )
        expected = None
        for variant in VARIANTS:
            os.environ["SGLANG_ZING_FUSED_CURRENT_ROPE_KV"] = str(
                int(variant != "baseline")
            )
            os.environ["SGLANG_ZING_FUSED_CURRENT_ROPE"] = "1"
            os.environ["SGLANG_ZING_FUSED_HISTORY_ROPE"] = "1"
            after_sp = variant in ("norm", "live")
            direct = variant in ("sp", "norm", "live")

            def call():
                projection = seed.clone()
                q, k, v = (x.reshape(-1, 56, 128) for x in projection.chunk(3, -1))
                if not after_sp and q.shape[0]:
                    fused_inplace_qknorm(
                        q,
                        k,
                        module.q_norm.weight,
                        module.k_norm.weight,
                        eps=module.q_norm.eps,
                    )
                shard = tuple(
                    x[None] if direct else x[None].contiguous() for x in (q, k, v)
                )
                exchanged = _usp_input_all_to_all_varlen_qkv(
                    *shard, splits, destination_major=direct
                )
                q, k, v = (x[0] for x in exchanged.chunk(3, -1))
                apply_history_rope_into(
                    raw,
                    history_coeff,
                    rotated,
                    live_rows=live if variant == "live" else None,
                )
                if after_sp:
                    return apply_current_qknorm_rope_kv(
                        module, q, k, v, coefficients, plan, 0, tensors
                    )
                out = apply_current_rope_kv(q, k, v, coefficients, plan, 0, tensors)
                if out is None:
                    out = apply_current_rope(q, coefficients)
                    rotated.index_copy_(
                        0, write_rows, apply_current_rope(k, coefficients)
                    )
                    values.index_copy_(0, write_rows, v)
                    raw.index_copy_(0, write_rows[:clean], k[:clean])
                return out

            result = call()
            actual = [
                result,
                raw[history : history + clean],
                rotated[: history + rows],
                values[history : history + rows],
            ]
            if expected is None:
                expected = [x.clone() for x in actual]
            else:
                for value, reference in zip(actual, expected):
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)
            for _ in range(10):
                call()
            dist.barrier()
            torch.cuda.synchronize()
            eager = timed(call, repeats=args.repeats, iterations=args.iterations)
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                captured = call()
            for _ in range(10):
                graph.replay()
            dist.barrier()
            torch.cuda.synchronize()
            replay = timed(
                graph.replay, repeats=args.repeats, iterations=args.iterations
            )
            torch.testing.assert_close(captured, expected[0], rtol=0, atol=0)
            mode = dict(equal=True, eager=eager, graph=replay)
            if args.trace and name.endswith("w6-t3"):
                mode["trace"] = trace_call(
                    graph.replay, output / f"{name}-{variant}-rank{rank}.json"
                )
            record["modes"][variant] = mode
            del graph, captured
        records.append(record)
        (output / f"rank{rank}.json").write_text(
            json.dumps(
                dict(
                    source_revision=args.source_revision,
                    world=world,
                    rank=rank,
                    cases=records,
                ),
                indent=2,
            )
        )
        if rank == 0:
            print(
                json.dumps(
                    dict(
                        case=name,
                        graph_ms={
                            k: v["graph"]["median_ms"]
                            for k, v in record["modes"].items()
                        },
                    )
                ),
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--windows", nargs="+", type=int, default=[6, 12, 21])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trace", action="store_true")
    run(parser.parse_args())
