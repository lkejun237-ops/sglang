# SPDX-License-Identifier: Apache-2.0
"""Single-GPU offline-table loading and real projection correctness smoke.

Prepare the exact deployment plan first. This exercises the production DiT
loader with the default weights-free mode, then compares every table batch and
projection against independent checkpoint F.linear calls. It is not a video
quality or performance benchmark and does not load encoders or decoders.
"""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F
from build_adaln_table import ensure
from safetensors import safe_open

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
)
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    set_global_server_args,
)


@torch.no_grad()
def run(model_path, plan_path, table_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This smoke requires exactly one visible CUDA GPU")
    device = torch.device("cuda:0")
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    table_path, build_status = ensure(model_path, plan_path, table_path, "cuda")
    server_args = ServerArgs.from_kwargs(
        model_path=str(model_path),
        num_gpus=1,
        tp_size=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        performance_mode="speed",
        warmup_mode="off",
        pipeline_config={
            "video_decoder_backend": "native",
            "zing_adaln_table_path": str(table_path),
        },
    )
    # Deliberately omit zing_adaln_mode: the production default must select it.
    assert server_args.pipeline_config.zing_adaln_mode == "weights_free"
    set_global_server_args(server_args)
    model = TransformerLoader().load_customized(
        str(Path(model_path) / "transformer"), server_args, "transformer"
    )
    model.eval()
    receipt = model.adaln_deployment_receipt()
    assert receipt["mode"] == "weights_free"
    assert receipt["projection_parameters"] == []
    assert receipt["projection_storage_bytes"] == 0
    table = model.adaln_table
    assert table.checkpoint_verified and table.device == device
    plan = json.loads(Path(plan_path).read_text())
    batches = [torch.tensor(values, dtype=torch.float32) for values in plan["batches"]]
    lease = table.prepare(
        batches,
        device=device,
        schedule=plan["schedule"],
        embedder=model.time_embedder,
    )
    comparisons = 0
    input_shapes = []
    try:
        embeddings = [
            F.silu(model.time_embedder(t.to(device))).bfloat16() for t in batches
        ]
        with ExitStack() as stack:
            index = {}
            for path in sorted(
                (Path(model_path) / "transformer").glob("*.safetensors")
            ):
                handle = stack.enter_context(
                    safe_open(str(path), framework="pt", device="cpu")
                )
                for key in handle.keys():
                    if key in index:
                        raise ValueError("Duplicate checkpoint tensor: " + key)
                    index[key] = handle
            for slot in [*range(model.arch.num_layers), "final"]:
                final = slot == "final"
                prefix = (
                    "backbone.norm_out.linear"
                    if final
                    else f"backbone.transformer_blocks.{slot}.adaln_proj.linear"
                )
                weight = (
                    index[prefix + ".weight"]
                    .get_tensor(prefix + ".weight")
                    .to(device=device, dtype=torch.bfloat16)
                )
                bias = (
                    index[prefix + ".bias"]
                    .get_tensor(prefix + ".bias")
                    .to(device=device, dtype=torch.bfloat16)
                )
                parts = 2 if final else 6
                width = parts * model.arch.hidden_size
                for timestep, embedding in zip(batches, embeddings):
                    expected = F.linear(embedding, weight, bias).view(-1, width)
                    actual = torch.cat(
                        table.project(slot, None, embedding, table.lookup(timestep)),
                        dim=-1,
                    )
                    assert actual.dtype == expected.dtype == torch.bfloat16
                    assert actual.shape == expected.shape
                    assert (
                        torch.isfinite(actual).all() and torch.isfinite(expected).all()
                    )
                    if not torch.equal(actual, expected):
                        raise AssertionError(
                            f"AdaLN projection mismatch at slot {slot}, batch {timestep.tolist()}"
                        )
                    comparisons += 1
                    if slot == 0:
                        input_shapes.append(list(embedding.shape))
                del weight, bias
                print(
                    json.dumps({"verified_slot": slot, "comparisons": comparisons}),
                    flush=True,
                )
    finally:
        lease.close()
    stats = table.stats()
    assert stats["active_plans"] == 0
    assert stats["misses"] == stats["fallback"] == 0
    return dict(
        status="PASS",
        scope="real_checkpoint_loader_and_all_full_batch_adaln_projections",
        gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        table_path=str(table_path),
        table_build_status=build_status,
        checkpoint=str(model_path),
        schedule=plan["schedule"],
        num_modalities=model.arch.num_modalities,
        layers=model.arch.num_layers,
        embedding_shapes=input_shapes,
        projection_comparisons=comparisons,
        all_torch_equal=True,
        deployment=receipt,
        table=stats,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        unvalidated=["end_to_end_video", "performance", "multi_gpu"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--table", help="Use a fresh path, or reuse an exactly matching table"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model_path, args.plan, args.table)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
