# SPDX-License-Identifier: Apache-2.0
"""Apply B300/H200 Zing acceleration recipes before importing the runtime.

The server and separately started component services use this same launcher.
Asset paths, role placement and byte budgets remain normal SGLang arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

# Pin live switches and regional policy from PR #647. Prepared model assets
# remain untouched; explicit environment/CLI values can override the recipe.
COMMON_ENV = {
    "NCCL_GRAPH_REGISTER": "0",
    "SGLANG_ZING_SOURCE_PREPARE": "1",
    "SGLANG_ZING_REMOTE_DECODE_PROGRESSIVE": "1",
    "SGLANG_ZING_SOURCE_PROMPT_JOINT": "1",
    "SGLANG_ZING_EXPERIMENTAL_ATTENTION": "fa4",
    "SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES": "1",
    "SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN": "1",
    "SGLANG_ZING_EXPERIMENTAL_FA4_METADATA_REUSE": "1",
    "SGLANG_ZING_ADALN_PROJECTION_CACHE": "1",
    "SGLANG_ZING_FUSED_HISTORY_ROPE": "1",
    "SGLANG_ZING_FUSED_CURRENT_ROPE": "1",
    "SGLANG_ZING_FUSED_KV_COMPACT": "1",
    "SGLANG_ZING_GROUPED_HISTORY_ROPE": "1",
    "SGLANG_ZING_CHUNK_ROPE_REUSE": "1",
    "SGLANG_ZING_HISTORY_ROPE_PLAN": "0",
    # #402, #405: strided SP packing and live-prefix/cross-graph RoPE reuse.
    "SGLANG_ZING_SP_DESTINATION_MAJOR": "1",
    "SGLANG_ZING_FUSED_QKNORM_ROPE_KV": "1",
    "SGLANG_ZING_HISTORY_ROPE_LIVE_ROWS": "1",
    "SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH": "1",
    "SGLANG_ZING_INPLACE_RESIDUAL": "1",
    "SGLANG_ZING_FUSED_PRE_LINEAR_QUANT": "1",
    "SGLANG_ZING_DENOISE_CUDA_GRAPH": "1",
    "SGLANG_ZING_DENOISE_CUDA_GRAPH_SP": "1",
    "SGLANG_ZING_DENOISE_CUDA_GRAPH_MAX": "32",
    "SGLANG_ZING_VISUAL_ENCODE_MODE": "performance",
    "MINIMAX_H3_PROGRESSIVE_ENCODER_TILING": "auto",
    "MINIMAX_H3_PROGRESSIVE_ENCODER_PERFORMANCE_AUTO": "1",
    "MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST": "1",
    "MINIMAX_H3_PROGRESSIVE_SPATIAL_PADDING": "reflect2d",
    # #404/#437: keep profile building and remote execution in agreement.
    "SGLANG_H3_QWEN_HF_RMSNORM": "1",
    "SGLANG_H3_QWEN_FUSED_MROPE": "1",
    "SGLANG_H3_QWEN_FUSED_GATE_UP": "1",
    "SGLANG_H3_QWEN_TEXT_FA4": "1",
    "SGLANG_H3_QWEN_VISION_FA4": "1",
    "SGLANG_H3_QWEN_FP8": "1",
}

ENCODER_ENV = {
    name: value
    for name, value in COMMON_ENV.items()
    if name == "SGLANG_ZING_VISUAL_ENCODE_MODE"
    or name.startswith("MINIMAX_H3_PROGRESSIVE_")
}
CONDITIONING_ENV = {
    name: value
    for name, value in COMMON_ENV.items()
    if name.startswith("SGLANG_H3_QWEN_")
}
# Lynn preparation fusion is selected automatically by its runtime guards;
# attention/precision come from the trained checkpoint, not DiT env switches.
DECODER_ENV = {}
COMPONENTS = ("server", "dit", "conditioning", "encoder", "decoder")

# #410/#452/#439/#455. Use the current CLI rather than retired environment
# aliases. StoreBoolean accepts explicit true/false; no value means true.
SERVER_DEFAULTS = {
    "--performance-mode": "speed",
    "--warmup-mode": "off",
    "--dist-timeout": "600",
    "--zing-solver": "few_step",
    "--zing-denoising-step-list": "1000,500",
    "--zing-initial-block-count": "3",
    "--zing-initial-denoising-step-list": "1000,750,500,250",
    "--zing-reference-max-images": "21",
    "--encoder-parallel": "replicate",
    "--attention-backend": "torch_sdpa",
    "--use-fsdp-inference": "false",
    "--realtime-session-lease-ttl-s": "1200",
    "--realtime-session-idle-timeout-s": "1200",
    "--dit-cpu-offload": "false",
    "--dit-layerwise-offload": "false",
    "--text-encoder-cpu-offload": "false",
    "--vae-cpu-offload": "false",
    # Generic whole-model compilation/Graph is distinct from the Zing-specific
    # compiled AdaLN and bounded denoise Graph switches above.
    "--enable-torch-compile": "false",
    "--enable-cuda-graph": "false",
    "--zing-conditioning-mode": "remote",
    "--zing-conditioning-prefetch": "true",
    "--zing-source-encoder-mode": "overlap",
    "--zing-source-encoder-prefetch": "true",
    "--zing-source-async-output": "true",
    "--zing-source-decoder-overlap": "true",
    "--zing-source-work-quantum": "1",
    "--zing-source-joint-forward": "cross_chunk",
    "--zing-source-defer-final": "true",
    "--zing-source-prompt-joint": "true",
    "--conditioning-result-transport": "nccl",
    "--mem-fraction-static": "0.8",
    "--zing-block0-cache-mode": "on",
    "--zing-block0-cache-host-gb": "416",
    "--zing-block0-cache-host-total-gb": "416",
    "--zing-block0-cache-disk-gb": "700",
    "--zing-block0-cache-storage": "local",
    "--zing-block0-cache-gc": None,
    "--zing-block0-cache-min-free-gb": "128",
    "--zing-block0-cache-inflight-gb": "128",
    "--zing-block0-cache-max-writers": "2",
    "--zing-block0-cache-gc-high-watermark": "0.9",
    "--zing-block0-cache-gc-low-watermark": "0.8",
    "--zing-block0-cache-capacity-dir": "/var/lib/sglang/capacity",
    "--realtime-max-sessions": "1",
    "--realtime-max-sessions-per-worker": "1",
}

# Existing #438 launcher owns child readiness, GPU isolation and cleanup.
ROLE_DEFAULTS = {
    "--encoder-gpus": "2",
    "--denoiser-gpus": "0,1",
    "--decoder-gpus": "3",
    "--num-gpus": "2",
    "--tp-size": "1",
    "--sp-degree": "2",
    "--ulysses-degree": "2",
}

DIT_DEFAULTS = {
    option: value
    for option, value in ROLE_DEFAULTS.items()
    if option not in {"--encoder-gpus", "--denoiser-gpus", "--decoder-gpus"}
}


def server_policy(profile: str, cache_dir: str | None = None) -> dict:
    """Shared CLI policy for generic and supervised launchers."""
    defaults = dict(SERVER_DEFAULTS)
    defaults["--realtime-session-max-lifetime-s"] = (
        "7800" if profile == "b300" else "7200"
    )
    defaults["--quantization"] = "fp8" if profile == "b300" else "none"
    if profile == "b300":
        defaults["--direct-gpu-weight-loading"] = "false"
    cache_dir = cache_dir or (
        "/var/cache/sglang/block0/shanghai"
        if profile == "h200"
        else "/var/cache/sglang/block0/thailand-480-restore4"
    )
    defaults["--zing-block0-cache-dir"] = cache_dir
    defaults["--zing-block0-cache-budget-root"] = str(Path(cache_dir).parent)
    return defaults


def policy_arguments(policy: Mapping) -> list[str]:
    return [
        token
        for option, value in policy.items()
        for token in ((option,) if value is None else (option, value))
    ]


@dataclass(frozen=True)
class GPUInfo:
    name: str
    capability: tuple[int, int]
    total_memory_mib: int
    multiprocessor_count: int


def resolve_profile(requested: str, gpus: list[GPUInfo]) -> str:
    if not gpus:
        raise ValueError("Zing hardware recipes require a visible CUDA GPU")
    profiles = []
    for gpu in gpus:
        if gpu.capability == (9, 0) and gpu.total_memory_mib >= 130 * 1024:
            profiles.append("h200")
        elif gpu.capability == (10, 3) and gpu.total_memory_mib >= 240 * 1024:
            profiles.append("b300")
        else:
            raise ValueError(
                f"No B300/H200 recipe for {gpu.name}: "
                f"CC={gpu.capability}, memory={gpu.total_memory_mib} MiB"
            )
    if len(set(profiles)) != 1:
        raise ValueError("Use one hardware family per Zing recipe invocation")
    actual = profiles[0]
    if requested not in ("auto", actual):
        raise ValueError(
            f"Requested {requested} recipe does not match detected {actual}"
        )
    return actual


def detect_gpus() -> list[GPUInfo]:
    import torch

    return [
        GPUInfo(
            name=p.name,
            capability=(p.major, p.minor),
            total_memory_mib=p.total_memory // (1024**2),
            multiprocessor_count=p.multi_processor_count,
        )
        for p in (
            torch.cuda.get_device_properties(i)
            for i in range(torch.cuda.device_count())
        )
    ]


def build_launch_contract(
    *,
    requested_profile: str,
    resolution: str,
    component: str,
    gpus: list[GPUInfo],
    arguments: list[str],
    environment: Mapping[str, str],
    worker_gpus: int = 1,
    mode: str = "integrated",
    sp_degree: int = 1,
) -> dict:
    if mode == "independent":
        from sglang.multimodal_gen.tools.zing_h3_role_launcher import (
            build_role_contract,
        )

        return build_role_contract(
            requested_profile=requested_profile,
            resolution=resolution,
            component=component,
            gpus=gpus,
            arguments=arguments,
            environment=environment,
            worker_gpus=worker_gpus,
            sp_degree=sp_degree,
        )
    if mode != "integrated" or sp_degree != 1:
        raise ValueError("Use --mode independent for launcher-level --sp-degree")
    profile = resolve_profile(requested_profile, gpus)
    if resolution not in ("480", "736") or component not in COMPONENTS:
        raise ValueError(
            "Choose resolution 480/736 and component server/dit/conditioning/encoder/decoder"
        )
    if component not in ("encoder", "decoder") and worker_gpus != 1:
        raise ValueError("--worker-gpus applies only to standalone Encoder/Decoder")
    defaults = dict(
        {
            "encoder": ENCODER_ENV,
            "decoder": DECODER_ENV,
            "conditioning": CONDITIONING_ENV,
        }.get(component, COMMON_ENV)
    )
    # Only the exact tested Blackwell 480 shape has the fixed-FA4 fast path.
    # The runtime additionally gates SP2, prefix, dtype, grid and FA4 version.
    fixed = (
        profile == "b300"
        and resolution == "480"
        and all(gpu.multiprocessor_count == 148 for gpu in gpus)
    )
    if component in ("server", "dit"):
        if profile == "b300":
            defaults["SGLANG_ZING_QUANT_SCOPE"] = "ffn-attention"
        defaults["SGLANG_ZING_FUSED_CURRENT_ROPE_KV"] = "1" if fixed else "0"
        defaults["SGLANG_ZING_FA4_FIXED"] = "1" if fixed else "0"
    effective = {name: environment.get(name, value) for name, value in defaults.items()}
    supplied = {arg.split("=", 1)[0] for arg in arguments if arg.startswith("--")}
    if component in ("server", "dit"):
        server_defaults = server_policy(
            profile, argument_value(arguments, "--zing-block0-cache-dir")
        )
        if component == "dit":
            if supplied.intersection(
                {"--encoder-gpus", "--denoiser-gpus", "--decoder-gpus"}
            ):
                raise ValueError("DiT-only mode cannot start local role GPU workers")
            if len(gpus) < 2:
                raise ValueError("The SP2 DiT recipe needs two visible GPUs")
            server_defaults.update(DIT_DEFAULTS)
        # Never partially replace an explicitly selected role layout. Advanced
        # deployments can supply their own complete #438 role arguments.
        if component == "server" and not supplied.intersection(
            {"--encoder-gpus", "--denoiser-gpus", "--decoder-gpus"}
        ):
            if len(gpus) < 4:
                raise ValueError(
                    "Default E1 + SP2 + D1 recipe needs four visible GPUs; "
                    "provide an explicit role layout for a different deployment"
                )
            server_defaults.update(ROLE_DEFAULTS)
        additions = policy_arguments(
            {
                option: value
                for option, value in server_defaults.items()
                if option not in supplied
            }
        )
        module = "sglang.multimodal_gen.runtime.launch_server"
    elif component in ("encoder", "decoder"):
        if not 1 <= worker_gpus <= len(gpus):
            raise ValueError(
                "Worker GPU count must be between 1 and the visible GPU count"
            )
        additions = [
            "--standalone",
            "--nproc-per-node",
            str(worker_gpus),
            "-m",
            (
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
                f"zing_h3.{component}_worker"
            ),
        ]
        module = "torch.distributed.run"
    else:
        additions = []
        if (
            arguments[:1] == ["serve"]
            and "--conditioning-result-transport" not in supplied
        ):
            # Subcommands must precede their options in conditioning.cli.
            arguments = [*arguments, "--conditioning-result-transport", "nccl"]
        module = "sglang.multimodal_gen.runtime.conditioning.cli"
    return {
        "schema_version": "zing-h3-acceleration/v1",
        "profile": profile,
        "resolution": resolution,
        "component": component,
        "hardware": [asdict(gpu) for gpu in gpus],
        "environment": effective,
        "environment_overrides": sorted(set(defaults) & set(environment)),
        "command": [sys.executable, "-m", module, *additions, *arguments],
        "validation": "Current combined 480/736 GPU performance, memory and quality not validated",
    }


def argument_value(arguments, option):
    # Match argparse: the last explicit occurrence wins.
    value = None
    for index, token in enumerate(arguments):
        if token.startswith(option + "="):
            value = token.split("=", 1)[1]
        elif token == option and index + 1 < len(arguments):
            value = arguments[index + 1]
    return value


def validate_launch_assets(arguments, profile, *, model_option="--model-path"):
    """Require the prepared PR #647 policy without rewriting shared assets."""
    path = argument_value(arguments, "--pipeline-config-path")
    if path is None:
        raise ValueError("The regional server recipe requires --pipeline-config-path")
    pipeline = json.loads(Path(path).read_text())
    model = argument_value(arguments, model_option)
    if model is None:
        raise ValueError(f"The recipe requires {model_option}")
    realtime = json.loads((Path(model) / "zing_realtime_profile.json").read_text())
    if (
        pipeline.get("zing_kv_pool_window_chunks"),
        pipeline.get("zing_kv_pool_sink_chunks"),
    ) != (6, 0) or realtime.get("source_attention_window") != {
        "window_chunks": 6,
        "sink_chunks": 0,
    }:
        raise ValueError("Prepare matching W6/S0 profile and KV pool before launching")
    if (
        pipeline.get("zing_kv_pool_window_eviction_timing", "after_step0")
        != "after_step0"
        or realtime.get("window_eviction_timing", "after_step0") != "after_step0"
    ):
        raise ValueError(
            "The regional recipe requires after_step0 in profile and KV pool"
        )
    if profile == "h200" and (
        pipeline.get("zing_adaln_mode", "weights_free") != "weights_free"
        or not pipeline.get("zing_adaln_table_path")
    ):
        raise ValueError("The H200 recipe requires a prepared AdaLN table")
    if profile == "b300" and (
        pipeline.get("zing_adaln_mode") != "resident"
        or pipeline.get("zing_adaln_table_path")
    ):
        raise ValueError("The B300 recipe requires resident AdaLN")
    retired = {
        "zing_kv_pool_" + name
        for name in (
            "size",
            "max_target_area",
            "max_target_long_edge",
            "max_prefix_rows",
        )
    }
    if retired.intersection(pipeline):
        raise ValueError("Remove retired KV capacity fields; use --mem-fraction-static")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("auto", "b300", "h200", "h100"), default="auto"
    )
    parser.add_argument("--resolution", choices=("480", "736"), required=True)
    parser.add_argument("--component", choices=COMPONENTS, default="server")
    parser.add_argument(
        "--worker-gpus",
        type=int,
        default=1,
        help="Local ranks for standalone Encoder/Decoder (default: one GPU).",
    )
    parser.add_argument(
        "--mode", choices=("integrated", "independent"), default="integrated"
    )
    parser.add_argument("--sp-degree", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    contract = build_launch_contract(
        requested_profile=args.profile,
        resolution=args.resolution,
        component=args.component,
        gpus=detect_gpus(),
        arguments=arguments,
        environment=os.environ,
        worker_gpus=args.worker_gpus,
        mode=args.mode,
        sp_degree=args.sp_degree,
    )
    print(json.dumps(contract, indent=2), flush=True)
    if not args.dry_run:
        if args.mode == "integrated" and args.component in ("server", "dit"):
            validate_launch_assets(arguments, contract["profile"])
        elif args.mode == "independent" and args.component == "dit":
            validate_launch_assets(
                arguments, contract["profile"], model_option="--model"
            )
        environment = dict(os.environ)
        for name in contract.get("environment_unset", ()):
            environment.pop(name, None)
        environment.update(contract["environment"])
        os.execvpe(contract["command"][0], contract["command"], environment)


if __name__ == "__main__":
    main()
