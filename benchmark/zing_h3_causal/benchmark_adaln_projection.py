# SPDX-License-Identifier: Apache-2.0
"""Single-GPU resident projection replay, not an end-to-end business benchmark.

--batches is a JSON list of ordered unique-t batches. Declare whether these
were captured or derived from a schedule; keep the provenance. This measures the
original modules/weights and cache lookup together, leaving timestep/SiLU in
both A and B. It does not measure token indexing, modulation, KV, or media.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from safetensors import safe_open


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def main(args):
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        MiniMaxH3AdalnProj,
        MiniMaxH3TimeEmbedder,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_cache import (
        ZingH3AdalnCache,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("This frozen replay requires exactly one visible SM103 GPU")
    for key, value in {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29531",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
    }.items():
        os.environ.setdefault(key, value)
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    root = args.model / "transformer"
    config = ZingH3DiTConfig()
    config.update_model_arch(json.loads((root / "config.json").read_text()))
    arch = config.arch_config
    if arch.num_modalities != 4 or arch.adaln_out_features != 24 * arch.hidden_size:
        raise ValueError("Replay requires the resident four-modality Zing checkpoint")
    index_path = root / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    weights = []

    def load(module, names):
        state = {}
        for destination, source in names.items():
            with safe_open(root / index[source], framework="pt", device="cpu") as shard:
                tensor = shard.get_tensor(source)
            weights.append(
                {
                    "name": source,
                    "shard": index[source],
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": hashlib.sha256(
                        tensor.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                }
            )
            state[destination] = tensor
        module.load_state_dict(state, strict=True)
        return module.cuda().eval()

    embedder = load(
        MiniMaxH3TimeEmbedder(arch, prefix="time_embedder"),
        {
            f"{dst}.{suffix}": f"backbone.time_embedder.{src}.{suffix}"
            for dst, src in (("proj_in", "linear_1"), ("proj_out", "linear_2"))
            for suffix in ("weight", "bias")
        },
    )
    modules = []
    for slot in list(range(arch.num_layers)) + ["final"]:
        final = slot == "final"
        module = MiniMaxH3AdalnProj(
            arch,
            arch.final_adaln_out_features if final else arch.adaln_out_features,
            None,
            prefix=str(slot),
            expand_ratio=2 if final else 6,
            modality_num=1 if final else 4,
        )
        prefix = (
            "backbone.norm_out"
            if final
            else f"backbone.transformer_blocks.{slot}.adaln_proj"
        )
        modules.append(
            load(
                module,
                {f"linear.{s}": f"{prefix}.linear.{s}" for s in ("weight", "bias")},
            )
        )
    batch_values = json.loads(args.batches.read_text())
    batches = [
        torch.tensor(t, device="cuda", dtype=torch.float32) for t in batch_values
    ]
    if not batches or any(
        t.ndim != 1 or not t.numel() or not torch.equal(t, torch.unique(t, sorted=True))
        for t in batches
    ):
        raise ValueError("Replay batches must contain complete sorted unique-t vectors")
    cache = ZingH3AdalnCache(args.budget_bytes)

    def replay(enabled):
        for t in batches:
            embedding = F.silu(embedder(t.float())).bfloat16()
            key = cache.begin(t, embedding, embedder, 0) if enabled else None
            for slot, module in enumerate(modules):
                if enabled:
                    cache.project(slot, module, embedding, key)
                else:
                    module(embedding)

    comparisons = 0
    for t in batches:
        embedding = F.silu(embedder(t.float())).bfloat16()
        key = cache.begin(t, embedding, embedder, 0)
        for slot, module in enumerate(modules):
            expected = module(embedding)
            for _ in range(2):
                actual = cache.project(slot, module, embedding, key)
                if not all(
                    a.shape == b.shape
                    and a.dtype == b.dtype
                    and torch.equal(a, b)
                    and bool(torch.isfinite(b).all())
                    for a, b in zip(expected, actual)
                ):
                    raise RuntimeError(
                        f"Projection exact comparison failed at slot {slot}"
                    )
                comparisons += 1
    cache.clear()
    samples = []
    for enabled in (False, True, True, False):
        cache.clear()
        replay(enabled)  # full warmup
        torch.cuda.synchronize()
        for repeat in range(args.repeats):
            start = perf_counter()
            replay(enabled)
            torch.cuda.synchronize()
            samples.append(
                {
                    "enabled": enabled,
                    "repeat": repeat,
                    "seconds": perf_counter() - start,
                }
            )
    cold = []
    for _ in range(args.repeats):
        cache.clear()
        torch.cuda.synchronize()
        start = perf_counter()
        replay(True)
        torch.cuda.synchronize()
        cold.append(perf_counter() - start)
    if args.profile:
        replay(True)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profiler:
            replay(True)
        profiler.export_chrome_trace(str(args.output / "hot.trace.json"))
    report = {
        "kind": "resident_projection_replay_not_business_acceptance",
        "model": str(args.model),
        "config_sha256": digest(root / "config.json"),
        "index_sha256": digest(index_path),
        "batches_sha256": digest(args.batches),
        "batches": batch_values,
        "batches_origin": args.batches_origin,
        "weights": weights,
        "comparisons_exact": comparisons,
        "samples": samples,
        "cold_seconds": cold,
        "cache": cache.stats(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(torch.cuda.get_device_properties(0)),
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "e2e_fps": None,
        "business_acceptance": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps({"comparisons_exact": comparisons, "cache": cache.stats()}),
        flush=True,
    )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument(
        "--batches-origin", choices=("captured", "schedule-derived"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-bytes", type=int, default=134217728)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parsed = parser.parse_args()
    if parsed.repeats < 5:
        parser.error("At least five samples per ordering are required")
    main(parsed)
