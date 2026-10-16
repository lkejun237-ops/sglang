# SPDX-License-Identifier: Apache-2.0
"""Offline table builder; never loads the full DiT or submits GPU jobs."""

import argparse
import json
import re
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
    ZingH3DiTArchConfig,
    resolve_zing_modality_tags,
)
from sglang.multimodal_gen.configs.zing_sampling import validate_initial_blocks
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3AdalnProj,
    MiniMaxH3TimeEmbedder,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
    arithmetic_contract,
    checkpoint_identity,
    initial_blocks_plan,
    table_key,
    table_staleness,
    video_plan,
    write_table,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_zing_h3_causal_renoise import (
    ZingH3CausalRenoiseSchedule,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3EulerSchedule,
)


def make_plan(args):
    count = getattr(args, "initial_block_count", 0)
    initial_grid = getattr(args, "initial_denoising_step_list", None)
    validate_initial_blocks(args.solver, args.denoising_step_list, initial_grid, count)
    if args.solver == "euler":
        schedule = ZingH3EulerSchedule(args.sigma_points, args.video_shift)
    else:
        if args.denoising_step_list is None:
            raise ValueError("few_step requires an explicit checkpoint noise grid")
        schedule = ZingH3CausalRenoiseSchedule(
            args.denoising_step_list,
            num_train_timesteps=args.num_train_timesteps,
            shift=args.video_shift,
        )
    plan = video_plan(
        args.solver,
        schedule.timesteps,
        args.condition,
        args.reference,
        sigmas=schedule.sigmas,
    )
    if count:
        initial = ZingH3CausalRenoiseSchedule(
            initial_grid,
            num_train_timesteps=args.num_train_timesteps,
            shift=args.video_shift,
        )
        plan = initial_blocks_plan(
            plan,
            video_plan(
                args.solver,
                initial.timesteps,
                args.condition,
                args.reference,
                sigmas=initial.sigmas,
            ),
            count,
        )
    Path(args.output).write_text(json.dumps(plan, indent=2) + "\n")


def locate(model_path):
    model_path = Path(model_path)
    root = model_path.parent if model_path.name == "transformer" else model_path
    transformer = root / "transformer"
    files = sorted(transformer.glob("*.safetensors"))
    deployment = json.loads((root / "zing_config.json").read_text())
    return root, files, transformer / "config.json", deployment


def check(model_path, table, plan_path=None):
    """Return (reasons, schedule check) comparing a table to the current export."""
    _, files, config_path, deployment = locate(model_path)
    schedule = None
    if plan_path is not None:
        schedule = json.loads(Path(plan_path).read_text())["schedule"]
    return table_staleness(table, files, config_path, deployment, schedule)


def default_table(model_path, plan_path, device):
    root, files, config_path, deployment = locate(model_path)
    key = table_key(
        identity=checkpoint_identity(files, config_path),
        provenance=deployment,
        schedule=json.loads(Path(plan_path).read_text())["schedule"],
        arithmetic=arithmetic_contract(torch.device(device)),
    )
    return root / f"adaln-table-{key}"


def ensure(model_path, plan_path, table, device):
    """Idempotent: the default directory name is the full contract key."""
    explicit = table is not None
    table = Path(table) if explicit else default_table(model_path, plan_path, device)
    if table.exists():
        reasons = check(model_path, table, plan_path)
        manifest = json.loads((table / "manifest.json").read_text())
        if manifest.get("arithmetic") != arithmetic_contract(torch.device(device)):
            reasons.append("build device or software arithmetic changed")
        if reasons:
            raise ValueError(
                f"Existing Zing AdaLN table {table} is stale ({'; '.join(reasons)}); "
                "it is never overwritten. "
                + ("Use a new --table path." if explicit else "Remove it to rebuild.")
            )
        return table, "current"
    build(model_path, plan_path, table, device)
    left = check(model_path, table, plan_path)
    if left:
        raise ValueError("Built Zing AdaLN table is still stale: " + "; ".join(left))
    return table, "built"


@torch.no_grad()
def build(model_path, plan_path, output, device):
    if Path(output).exists():
        raise ValueError("Zing AdaLN output directory already exists")
    model_path = Path(model_path)
    root = model_path.parent if model_path.name == "transformer" else model_path
    transformer = (
        model_path if model_path.name == "transformer" else root / "transformer"
    )
    config_path = transformer / "config.json"
    config = json.loads(config_path.read_text())
    if config.pop("_class_name", None) != "ZingH3DiTModel":
        raise ValueError(
            "Zing AdaLN builder requires the normalized Zing serving export"
        )
    arch = ZingH3DiTArchConfig()
    for name, value in config.items():
        if not hasattr(arch, name):
            raise ValueError(f"Unknown Zing architecture field: {name}")
        setattr(arch, name, value)
    tags = resolve_zing_modality_tags(arch.num_modalities, arch.modality_tags)
    if arch.adaln_curve_grid is not None or arch.adaln_affine_input_dim is not None:
        raise ValueError("Zing AdaLN builder does not support curve checkpoints")
    deployment = json.loads((root / "zing_config.json").read_text())
    provenance = deployment.get("provenance", {})
    if (
        not re.fullmatch(r"[0-9a-f]{64}", deployment.get("checkpoint_sha256", ""))
        or provenance.get("checkpoint_source") not in {"generator", "ema"}
        or not re.fullmatch(r"[0-9a-f]{40}", provenance.get("native_sha", ""))
        or not provenance.get("component_versions")
        or deployment.get("training_effective_config", {})
        .get("lora_config", {})
        .get("enabled", False)
        and provenance.get("lora_merged") is not True
    ):
        raise ValueError(
            "Zing AdaLN builder requires effective checkpoint and merged-LoRA provenance"
        )
    plan = json.loads(Path(plan_path).read_text())
    if (
        plan["schedule"]["condition"] != arch.ar_condition_timestep
        or plan["schedule"]["reference"] != arch.reference_visual_timestep
    ):
        raise ValueError(
            "Zing AdaLN plan prefix timesteps do not match the effective checkpoint"
        )
    device = torch.device(device)
    # Builder is single-rank TP=SP=1, even when the worker will use SP2.
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    files = sorted(transformer.glob("*.safetensors"))
    if not files:
        raise ValueError("Zing AdaLN builder found no serving checkpoint shards")
    identity = checkpoint_identity(files, config_path)
    with ExitStack() as stack:
        index = {}
        for file in files:
            handle = stack.enter_context(
                safe_open(str(file), framework="pt", device="cpu")
            )
            for key in handle.keys():
                if key in index:
                    raise ValueError(f"Duplicate checkpoint tensor: {key}")
                index[key] = handle

        def copy(module, mapping):
            parameters = dict(module.named_parameters())
            for source, target in mapping.items():
                value = index[source].get_tensor(source)
                param = parameters[target]
                if tuple(value.shape) != tuple(param.shape):
                    raise ValueError(f"Invalid Zing AdaLN weight shape: {source}")
                param.copy_(value.to(dtype=param.dtype))
            # Builder copies directly and never uses loader callbacks. Remove
            # parameter -> bound loader -> module cycles before streaming on.
            for param in module.parameters():
                if hasattr(param, "weight_loader"):
                    delattr(param, "weight_loader")

        with torch.device("meta"):
            embedder = MiniMaxH3TimeEmbedder(arch, prefix="time_embedder")
        embedder.to_empty(device=device).eval()
        copy(
            embedder,
            {
                f"backbone.time_embedder.linear_{i}.{part}": f"proj_{target}.{part}"
                for i, target in [(1, "in"), (2, "out")]
                for part in ["weight", "bias"]
            },
        )

        def projections():
            for slot in [*range(arch.num_layers), "final"]:
                final = slot == "final"
                with torch.device("meta"):
                    module = MiniMaxH3AdalnProj(
                        arch,
                        arch.final_adaln_out_features
                        if final
                        else arch.adaln_out_features,
                        None,
                        prefix="adaln_proj",
                        expand_ratio=2 if final else 6,
                        modality_num=1 if final else arch.num_modalities,
                    )
                module.to_empty(device=device).eval()
                prefix = (
                    "backbone.norm_out.linear"
                    if final
                    else f"backbone.transformer_blocks.{slot}.adaln_proj.linear"
                )
                copy(
                    module,
                    {
                        f"{prefix}.{part}": f"linear.{part}"
                        for part in ["weight", "bias"]
                    },
                )
                yield slot, module
                del module

        manifest = write_table(
            output,
            arch=arch,
            tags=tags,
            embedder=embedder,
            projections=projections(),
            plan=plan,
            identity=identity,
            provenance=deployment,
        )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--solver", choices=["euler", "few_step"], required=True)
    plan.add_argument("--sigma-points", type=int, default=50)
    plan.add_argument("--video-shift", type=float, default=5.0)
    plan.add_argument("--denoising-step-list", type=int, nargs="+")
    plan.add_argument("--initial-block-count", type=int, default=0, choices=[0, 3])
    plan.add_argument("--initial-denoising-step-list", type=int, nargs="+")
    plan.add_argument("--num-train-timesteps", type=int, default=1000)
    plan.add_argument("--condition", type=float, default=1.0)
    plan.add_argument("--reference", type=float, default=1.0)
    plan.add_argument("--output", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--model-path", required=True)
    builder.add_argument("--plan", required=True)
    builder.add_argument("--output", required=True)
    builder.add_argument("--device", required=True, choices=["cpu", "cuda"])
    checker = commands.add_parser(
        "check", help="exit 0 if the table matches the export, 2 if a rebuild is needed"
    )
    checker.add_argument("--model-path", required=True)
    checker.add_argument("--table", required=True)
    checker.add_argument("--plan", help="also require this schedule plan")
    ensurer = commands.add_parser(
        "ensure", help="check, and build a paired table only if stale or missing"
    )
    ensurer.add_argument("--model-path", required=True)
    ensurer.add_argument("--plan", required=True)
    ensurer.add_argument("--table", help="default: <export>/adaln-table-<key>")
    ensurer.add_argument("--device", required=True, choices=["cpu", "cuda"])
    args = parser.parse_args()
    if args.command == "check":
        reasons = check(args.model_path, args.table, args.plan)
        print(
            json.dumps(
                {"status": "rebuild_required" if reasons else "ok", "reasons": reasons},
                indent=2,
            )
        )
        sys.exit(2 if reasons else 0)
    elif args.command == "ensure":
        table, status = ensure(args.model_path, args.plan, args.table, args.device)
        print(json.dumps({"status": status, "table": str(table)}, indent=2))
    elif args.command == "plan":
        make_plan(args)
    else:
        build(args.model_path, args.plan, args.output, args.device)


if __name__ == "__main__":
    main()
