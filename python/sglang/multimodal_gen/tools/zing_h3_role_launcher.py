# SPDX-License-Identifier: Apache-2.0
"""Independent worker commands; group ownership stays with the existing client."""

from __future__ import annotations

import importlib.util
import ipaddress
import sys
from dataclasses import asdict
from pathlib import Path


def _load_sibling_profile_launcher():
    mod_name = "sglang.multimodal_gen.tools.zing_h3_profile_launcher"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = Path(__file__).with_name("zing_h3_profile_launcher.py")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_profile_mod = _load_sibling_profile_launcher()
COMMON_ENV = _profile_mod.COMMON_ENV
CONDITIONING_ENV = _profile_mod.CONDITIONING_ENV
DECODER_ENV = _profile_mod.DECODER_ENV
ENCODER_ENV = _profile_mod.ENCODER_ENV
argument_value = _profile_mod.argument_value
policy_arguments = _profile_mod.policy_arguments
resolve_profile = _profile_mod.resolve_profile
server_policy = _profile_mod.server_policy

STAGES = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3."
)
DISTRIBUTED_ENV = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)

TOPOLOGY_WORKAROUNDS = {
    "cross_host_sm90_h100_h200": {
        "environment": {
            "NCCL_MIN_NCHANNELS": "2",
            "NCCL_MIN_CTAS": "2",
        },
        "nccl_version": "2.27.3+cuda12.9",
        "topology": "cross-host K3s H100 (SM90, 80 GiB) <-> PAI-DLC H200 (SM90, 140 GiB)",
        "reason": (
            "Align cross-host SM90 PyNcclCommunicator channel/CTA count during "
            "independent WorkerGroupMaster build_group handshake; not a host quota."
        ),
    }
}

SCOPED_MEMORY_POLICIES = {
    "h200_model1027_480_bf16_sp1_block0_cache": {
        "mem_fraction_static": "0.875",
        "conditions": {
            "model_id": "model1027",
            "checkpoint_step": "global_step_010000",
            "num_frame_per_block": [3, 2],
            "split_first_latent": False,
            "profile": "h200",
            "resolution": "480",
            "precision": "bf16",
            "quantization": None,
            "sp_degree": 1,
            "zing_block0_cache_mode": "on",
        },
        "reason": (
            "Measured static KV/cache reservation for model1027 832x480 BF16 SP1 "
            "with resident L1 Block0 cache on a single 140 GiB H200; not a general H200 default."
        ),
    }
}


def resolve_topology_workaround(topology: str | None) -> dict:
    if not topology:
        return {}
    if topology not in TOPOLOGY_WORKAROUNDS:
        raise ValueError(f"Unknown topology workaround profile: {topology}")
    return dict(TOPOLOGY_WORKAROUNDS[topology])


def resolve_scoped_dit_memory_fraction(
    *,
    profile: str,
    resolution: str,
    sp_degree: int,
    quantization: str | None,
    cache_mode: str,
    model_id: str | None = None,
) -> str:
    scoped = SCOPED_MEMORY_POLICIES["h200_model1027_480_bf16_sp1_block0_cache"]
    cond = scoped["conditions"]
    if (
        model_id == cond["model_id"]
        and profile == cond["profile"]
        and resolution == cond["resolution"]
        and sp_degree == cond["sp_degree"]
        and quantization in (None, "none")
        and cache_mode == cond["zing_block0_cache_mode"]
    ):
        return scoped["mem_fraction_static"]
    return server_policy(profile)["--mem-fraction-static"]


def build_role_contract(
    *,
    requested_profile,
    resolution,
    component,
    gpus,
    arguments,
    environment,
    worker_gpus=1,
    sp_degree=1,
):
    if component not in ("encoder", "dit", "decoder"):
        raise ValueError("Independent mode requires encoder, dit or decoder")
    if resolution not in ("480", "736") or sp_degree not in (1, 2):
        raise ValueError("Choose resolution 480/736 and SP degree 1/2")
    if worker_gpus != 1 or (component != "dit" and sp_degree != 1):
        raise ValueError(
            "Independent Encoder/Decoder use one GPU; use --sp-degree for DiT"
        )
    ranks = sp_degree if component == "dit" else 1
    if len(gpus) != ranks:
        raise ValueError(f"Expose exactly {ranks} allocated GPU(s) for this role")
    if component in ("encoder", "decoder") and all(
        g.capability == (9, 0) and 0 < g.total_memory_mib < 130 * 1024 for g in gpus
    ):
        profile = "h100"
        if requested_profile not in ("auto", profile):
            raise ValueError("Requested profile does not match the H100 role device")
    else:
        profile = resolve_profile(requested_profile, gpus)

    supplied = {a.split("=", 1)[0] for a in arguments if a.startswith("--")}
    protected = {
        "--service-count",
        "--max-sessions",
        "--advertise-host",
        "--sp-degree",
        "--static-connection",
    }
    if any(
        option not in protected and any(full.startswith(option) for full in protected)
        for option in supplied
    ):
        raise ValueError(
            "Use complete worker option names for role and protocol settings"
        )
    forbidden = supplied.intersection(
        {
            "--model-path",
            "--encoder-gpus",
            "--denoiser-gpus",
            "--decoder-gpus",
            "--num-gpus",
            "--tp-size",
            "--ulysses-degree",
            "--static-connection",
        }
    )
    if forbidden:
        raise ValueError(
            "Independent workers do not accept server/static options: "
            + ", ".join(sorted(forbidden))
        )
    service_count = argument_value(arguments, "--service-count")
    if service_count is not None:
        if not service_count.isdigit() or int(service_count) < 1:
            raise ValueError("--service-count must be a positive integer")
        if component == "dit" and int(service_count) != 1:
            raise ValueError("This recipe starts one service per role for DiT")
    if argument_value(arguments, "--max-sessions") not in (None, "1"):
        raise ValueError("This recipe supports one active session per service")
    host = argument_value(arguments, "--advertise-host")
    if not host or host.lower() == "localhost":
        raise ValueError("Provide --advertise-host reachable by the other workers")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None  # Worker APIs also accept a deployment-resolved hostname.
    if address is not None and (address.is_loopback or address.is_unspecified):
        raise ValueError("--advertise-host cannot be loopback or a wildcard address")

    defaults = {"--service-count": "1"}
    if component == "encoder":
        env = {
            **ENCODER_ENV,
            **CONDITIONING_ENV,
            "SGLANG_ZING_QWEN_PREFIX_BYTES": str(20 * 1024**3),
            "NCCL_GRAPH_REGISTER": "0",
            "SGLANG_ZING_SOURCE_PREPARE": "1",
            "SGLANG_ZING_SOURCE_PROMPT_JOINT": "1",
        }
        defaults["--session-max-lifetime-s"] = server_policy(profile)[
            "--realtime-session-max-lifetime-s"
        ]
        module = "sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server"
    elif component == "decoder":
        env = dict(DECODER_ENV)
        module = STAGES + "decoder_worker"
    else:
        env = {
            k: v
            for k, v in COMMON_ENV.items()
            if k not in ENCODER_ENV and k not in CONDITIONING_ENV
        }
        # SP1 uses the runtime's guarded attention/RoPE paths, not SP2 fixed-shape admission.
        if profile == "b300" or argument_value(arguments, "--quantization") == "fp8":
            env["SGLANG_ZING_QUANT_SCOPE"] = "ffn-attention"
        env["SGLANG_ZING_FA4_FIXED"] = "0"
        env["SGLANG_ZING_FUSED_CURRENT_ROPE_KV"] = "0"
        policy = server_policy(
            profile, argument_value(arguments, "--zing-block0-cache-dir")
        )
        defaults.update(
            {
                k: v
                for k, v in policy.items()
                if k.startswith("--zing-block0-cache-")
                or k
                in {
                    "--mem-fraction-static",
                    "--quantization",
                    "--zing-solver",
                    "--zing-denoising-step-list",
                    "--zing-initial-block-count",
                    "--zing-initial-denoising-step-list",
                    "--zing-source-joint-forward",
                }
            }
        )
        effective_initial_blocks = argument_value(
            arguments, "--zing-initial-block-count"
        )
        if effective_initial_blocks is None:
            effective_initial_blocks = defaults.get("--zing-initial-block-count")
        if effective_initial_blocks == "0":
            defaults.pop("--zing-initial-denoising-step-list", None)
        if argument_value(arguments, "--sp-degree") not in (None, str(sp_degree)):
            raise ValueError("Worker --sp-degree must match launcher --sp-degree")
        # Native workers use None for BF16; the integrated launcher sentinel
        # "none" is not a registered quantization method.
        if defaults.get("--quantization") == "none":
            del defaults["--quantization"]
        defaults["--sp-degree"] = str(sp_degree)
        if any(a.startswith("--zing-source-defer-final=") for a in arguments):
            raise ValueError(
                "Use --zing-source-defer-final or --no-zing-source-defer-final"
            )
        if not supplied.intersection(
            {"--zing-source-defer-final", "--no-zing-source-defer-final"}
        ):
            defaults["--zing-source-defer-final"] = None
        module = STAGES + "dit_worker"

    additions = policy_arguments(
        {k: v for k, v in defaults.items() if k not in supplied}
    )
    if component == "decoder" or ranks > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            str(ranks),
            "-m",
            module,
            *additions,
            *arguments,
        ]
    else:
        command = [sys.executable, "-m", module, *additions, *arguments]
    return {
        "schema_version": "zing-h3-acceleration/v1",
        "mode": "independent",
        "profile": profile,
        "resolution": resolution,
        "component": component,
        "hardware": [asdict(g) for g in gpus],
        "environment": {k: environment.get(k, v) for k, v in env.items()},
        "environment_overrides": sorted(set(env) & set(environment)),
        "environment_unset": list(DISTRIBUTED_ENV),
        "command": command,
        "group_management": "External client uses the existing worker connection APIs",
        "validation": "GPU fit, inter-host NCCL, quality and performance not validated",
    }
