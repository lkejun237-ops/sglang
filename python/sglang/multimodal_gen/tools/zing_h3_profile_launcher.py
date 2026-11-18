# SPDX-License-Identifier: Apache-2.0
"""Apply B300/H200 Zing acceleration recipes before importing the runtime.

The server and separately started component services use this same launcher.
Asset paths, role placement and byte budgets remain normal SGLang arguments.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
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
    "--zing-initial-block-count": "0",
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


def policy_source_metadata() -> dict[str, str]:
    here = Path(__file__).resolve()
    role_launcher = here.with_name("zing_h3_role_launcher.py")
    digest = hashlib.sha256()
    digest.update(here.read_bytes())
    if role_launcher.is_file():
        digest.update(role_launcher.read_bytes())
    return {
        "schema_version": "zing-h3-acceleration/v1",
        "launcher_module": "sglang.multimodal_gen.tools.zing_h3_profile_launcher",
        "role_launcher_module": "sglang.multimodal_gen.tools.zing_h3_role_launcher",
        "policy_source_sha256": digest.hexdigest(),
    }


def build_integrated_group_commands(
    bundle: Mapping,
    *,
    python: str = sys.executable,
    fixed_roles_script: str | Path | None = None,
) -> dict:
    """Build the 4-GPU integrated Whitebox role argv dictionary from versioned policy."""
    c, b = bundle["policy"], bundle["bindings"]
    m = c["model"]
    root, state = Path(b["asset_root"]), Path(b["state_root"])
    distributed = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        "-m",
    ]
    runtime_module = "sglang.multimodal_gen.runtime."
    workers = runtime_module + "pipelines_core.stages.model_specific_stages.zing_h3."
    dit_entry = (
        [python, str(fixed_roles_script)]
        if fixed_roles_script is not None
        else [
            python,
            "-m",
            "sglang.multimodal_gen.tools.zing_h3_profile_launcher",
            "_fixed_roles_server",
        ]
    )
    dit = [
        *dit_entry,
        "--model-path",
        str(root / "model"),
        "--pipeline-config-path",
        str(root / "pipeline.json"),
        "--mem-fraction-static",
        str(m["mem_fraction_static"]),
        "--num-gpus",
        DIT_DEFAULTS["--num-gpus"],
        "--sp-degree",
        DIT_DEFAULTS["--sp-degree"],
        "--ulysses-degree",
        DIT_DEFAULTS["--ulysses-degree"],
        "--ring-degree",
        "1",
        "--encoder-parallel",
        SERVER_DEFAULTS["--encoder-parallel"],
        "--attention-backend",
        SERVER_DEFAULTS["--attention-backend"],
        "--performance-mode",
        SERVER_DEFAULTS["--performance-mode"],
        "--warmup-mode",
        SERVER_DEFAULTS["--warmup-mode"],
        "--dit-cpu-offload",
        SERVER_DEFAULTS["--dit-cpu-offload"],
        "--dit-layerwise-offload",
        SERVER_DEFAULTS["--dit-layerwise-offload"],
        "--text-encoder-cpu-offload",
        SERVER_DEFAULTS["--text-encoder-cpu-offload"],
        "--vae-cpu-offload",
        SERVER_DEFAULTS["--vae-cpu-offload"],
        "--use-fsdp-inference",
        SERVER_DEFAULTS["--use-fsdp-inference"],
        "--realtime-max-sessions",
        SERVER_DEFAULTS["--realtime-max-sessions"],
        "--realtime-max-sessions-per-worker",
        SERVER_DEFAULTS["--realtime-max-sessions-per-worker"],
        "--host",
        "0.0.0.0",
        "--port",
        "30000",
        "--scheduler-port",
        "31005",
        "--master-port",
        "31006",
        "--dist-timeout",
        SERVER_DEFAULTS["--dist-timeout"],
        "--realtime-session-lease-ttl-s",
        SERVER_DEFAULTS["--realtime-session-lease-ttl-s"],
        "--realtime-session-idle-timeout-s",
        SERVER_DEFAULTS["--realtime-session-idle-timeout-s"],
        "--realtime-session-max-lifetime-s",
        str(max(7200, c["play_seconds"] + 600)),
        "--zing-solver",
        SERVER_DEFAULTS["--zing-solver"],
        "--zing-denoising-step-list",
        ",".join(map(str, m["denoising_steps"])),
        "--zing-initial-block-count",
        str(m["initial_blocks"]),
        *(
            [
                "--zing-initial-denoising-step-list",
                ",".join(map(str, m["initial_steps"])),
            ]
            if m["initial_blocks"] > 0 and m.get("initial_steps")
            else []
        ),
        "--zing-reference-max-images",
        str(m["reference_max_images"]),
        "--zing-block0-cache-mode",
        SERVER_DEFAULTS["--zing-block0-cache-mode"],
        "--zing-block0-cache-host-gb",
        str(m["cache_host_gb"]),
        "--zing-block0-cache-dir",
        m["cache_dir"],
        "--zing-block0-cache-disk-gb",
        str(m["cache_disk_gb"]),
        "--zing-block0-cache-storage",
        SERVER_DEFAULTS["--zing-block0-cache-storage"],
        "--zing-conditioning-mode",
        SERVER_DEFAULTS["--zing-conditioning-mode"],
        "--conditioning-service-url",
        "http://127.0.0.1:31002",
        "--conditioning-expected-profile",
        str(root / "conditioning-profile.json"),
        "--conditioning-config",
        str(root / "worker.secret.json"),
        "--zing-block0-cache-gc",
        "--zing-block0-cache-budget-root",
        str(Path(m["cache_dir"]).parent),
        "--zing-block0-cache-min-free-gb",
        str(
            m.get(
                "cache_disk_reserve_gb",
                int(SERVER_DEFAULTS["--zing-block0-cache-min-free-gb"]),
            )
        ),
        "--zing-block0-cache-inflight-gb",
        str(
            m.get(
                "cache_disk_inflight_gb",
                int(SERVER_DEFAULTS["--zing-block0-cache-inflight-gb"]),
            )
        ),
        "--zing-block0-cache-max-writers",
        str(
            m.get(
                "cache_disk_max_writers",
                int(SERVER_DEFAULTS["--zing-block0-cache-max-writers"]),
            )
        ),
        "--zing-block0-cache-gc-high-watermark",
        SERVER_DEFAULTS["--zing-block0-cache-gc-high-watermark"],
        "--zing-block0-cache-gc-low-watermark",
        SERVER_DEFAULTS["--zing-block0-cache-gc-low-watermark"],
        "--zing-block0-cache-capacity-dir",
        m.get(
            "cache_capacity_dir", SERVER_DEFAULTS["--zing-block0-cache-capacity-dir"]
        ),
        "--zing-block0-cache-host-total-gb",
        str(m.get("cache_host_total_gb", m["cache_host_gb"])),
    ]
    if m.get("quantization", "none") == "fp8":
        dit += ["--quantization", "fp8", "--direct-gpu-weight-loading", "false"]
    if m.get("cache_backend", "local") == "mooncake":
        cache_config = bundle["bindings"].get("cache_config")
        if not isinstance(cache_config, str) or not Path(cache_config).is_absolute():
            raise ValueError("Mooncake requires an absolute host cache configuration")
        dit += [
            "--zing-block0-cache-backend",
            "mooncake",
            "--zing-block0-mooncake-config",
            cache_config,
        ]
    quantum = m["source_work_quantum"]
    dit += ["--zing-source-work-quantum", str(quantum)]
    dit += ["--zing-source-encoder-mode", "overlap" if quantum else "serial"]
    for flag in (
        "--zing-source-encoder-prefetch",
        "--zing-source-async-output",
        "--zing-source-decoder-overlap",
        "--zing-source-defer-final",
    ):
        dit += [flag, "true" if quantum else "false"]
    prompt_joint = bundle["environments"]["model"]["SGLANG_ZING_SOURCE_PROMPT_JOINT"]
    if prompt_joint not in ("0", "1"):
        raise ValueError("Source prompt joint policy must be 0 or 1")
    dit += ["--zing-source-prompt-joint", "true" if prompt_joint == "1" else "false"]
    return {
        "encoder": {
            "devices": m["encoder_gpus"],
            "argv": distributed
            + [
                workers + "encoder_worker",
                "--model",
                str(root / "model"),
                "--port",
                "31001",
                "--timeout-seconds",
                "1800",
            ],
        },
        "conditioning": {
            "devices": m["encoder_gpus"],
            "argv": [
                python,
                "-m",
                runtime_module + "conditioning.cli",
                "serve",
                "--config",
                str(root / "service.secret.json"),
                "--profile",
                str(root / "conditioning-profile.json"),
                "--host",
                "127.0.0.1",
                "--port",
                "31002",
            ],
        },
        "decoder": {
            "devices": m["decoder_gpus"],
            "argv": distributed
            + [
                workers + "decoder_worker",
                "--static-connection",
                "--model",
                str(root / "model"),
                "--port",
                "31003",
            ],
        },
        "dit": {"devices": m["dit_gpus"], "argv": dit},
        "heartbeat": {
            "devices": [],
            "argv": [
                python,
                "-m",
                runtime_module + "entrypoints.realtime_worker_heartbeat",
                "--coordinator-url",
                b["coordinator_url"],
                "--health-url",
                "http://127.0.0.1:30000/health",
                "--state-url",
                "http://127.0.0.1:30000/v1/realtime_worker/state",
                "--worker-id",
                b["worker_id"],
                "--worker-epoch-file",
                str(state / "worker-epoch"),
                "--role",
                "denoiser",
                "--endpoint",
                f"ws://{b['advertise_ip']}:30000/v1/realtime_video/generate",
                "--reservation-endpoint",
                f"http://{b['advertise_ip']}:30000/v1/realtime_worker",
                "--az",
                b["availability_zone"],
                "--capacity",
                "1",
                "--model-revision",
                b["routing_revision"],
                "--routing-revision",
                b["routing_revision"],
                "--vae-fingerprint",
                "local",
            ],
        },
    }


def stop_children(children, *, os_module=os) -> None:
    """Stop owned process groups only; never search for/kill arbitrary GPU users."""
    for process, log in reversed(children):
        with contextlib.suppress(ProcessLookupError):
            os_module.killpg(process.pid, signal.SIGTERM)
    for process, log in reversed(children):
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os_module.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        log.close()


def wait_http(
    process,
    url: str,
    headers: Mapping[str, str],
    timeout: float = 600,
    *,
    urllib_request=urllib.request,
    time_module=time,
) -> None:
    deadline = time_module.monotonic() + timeout
    while time_module.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Owned role exited before readiness")
        try:
            with urllib_request.urlopen(
                urllib_request.Request(url, headers=headers), timeout=3
            ) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time_module.sleep(1)
    raise TimeoutError("Owned role readiness timed out")


def role_metrics_environment(role, environment, *, directory=None):
    """Configure before importing prometheus; ranks share one role boot directory.

    Integrated supervisors always allocate afresh, even if their parent inherited
    a directory. Independent deploy may supply its own fresh, receipted directory.
    Direct launcher users own the lifecycle of an explicitly supplied directory.
    """
    return {
        **environment,
        "WORLD_MODEL_METRIC_SERVICE": "qwen_vlm" if role == "conditioning" else role,
        "WORLD_MODEL_METRIC_LANE": environment.get(
            "WORLD_MODEL_METRIC_LANE", "default"
        ),
        "PROMETHEUS_MULTIPROC_DIR": directory
        or tempfile.mkdtemp(prefix=f"zing-metrics-{role}-"),
    }


def supervise_integrated_group(
    plan: Mapping[str, Mapping],
    *,
    bundle: Mapping,
    state: Path,
    expected: list[str],
    config: Mapping,
    evidence: Mapping,
    token: str,
    retries: int = 0,
    subprocess_module=subprocess,
    os_module=os,
    signal_module=signal,
    time_module=time,
    wait_http_fn=None,
    stop_children_fn=None,
) -> None:
    """Supervise an integrated 4-GPU Whitebox model group with process-group cleanup."""
    wait_http_impl = wait_http_fn or wait_http
    stop_children_impl = stop_children_fn or (
        lambda children: stop_children(children, os_module=os_module)
    )
    signal_module.signal(signal.SIGUSR1, signal.SIG_IGN)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal_module.signal(sig, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    for attempt in range(retries + 1):
        children = []
        status = {"attempt": attempt, "phase": "starting"}
        try:
            worker_epoch = uuid.uuid4().hex
            (state / "worker-epoch").write_text(worker_epoch)
            for role, item in plan.items():
                env = dict(os_module.environ, **bundle["environments"]["model"])
                env["WORKER_EPOCH"] = worker_epoch
                env["WORKER_EPOCH_FILE"] = str(state / "worker-epoch")
                env["WHITEBOX_CHECKPOINT_SHA256"] = config["checkpoint_sha256"]
                env["WHITEBOX_MODEL_CONFIG_SHA256"] = evidence["asset_config_sha256"][
                    "model/zing_config.json"
                ]
                for key in (
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                    "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "SGLANG_DIFFUSION_PLATFORM_OVERRIDE",
                ):
                    env.pop(key, None)
                env["CUDA_VISIBLE_DEVICES"] = ",".join(
                    expected[i] for i in item["devices"]
                )
                env = role_metrics_environment(role, env)
                from sglang.multimodal_gen.runtime.utils.bounded_process_log import (
                    BoundedProcessLog,
                )

                log = BoundedProcessLog(state / f"{role}.log")
                try:
                    process = subprocess_module.Popen(
                        item["argv"],
                        env=env,
                        stdout=subprocess_module.PIPE,
                        stderr=subprocess_module.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log.close()
                    raise
                log.attach(process.stdout)
                children.append((process, log))
                if role in ("encoder", "decoder"):
                    from sglang.multimodal_gen.runtime.disaggregation.role_launcher import (
                        wait_ready,
                    )

                    wait_ready(
                        process,
                        31001 if role == "encoder" else 31003,
                        timeout_s=600,
                    )
                elif role == "conditioning":
                    wait_http_impl(
                        process,
                        "http://127.0.0.1:31002/v1/conditioning/profile",
                        {"Authorization": "Bearer " + token},
                    )
                elif role == "dit":
                    wait_http_impl(process, "http://127.0.0.1:30000/health", {})
            status["phase"] = "ready"
            (state / "status.json").write_text(json.dumps(status))
            while all(p.poll() is None for p, _ in children):
                time_module.sleep(1)
            raise RuntimeError("Owned model role exited")
        except (SystemExit, KeyboardInterrupt):
            status["phase"] = "stopped"
            (state / "status.json").write_text(json.dumps(status))
            raise
        except Exception as error:
            status.update(phase="failed", error_type=type(error).__name__)
            (state / "status.json").write_text(json.dumps(status))
            if attempt == retries:
                raise
        finally:
            stop_children_impl(children)


def fixed_roles_main(argv: list[str] | None = None) -> None:
    """DiT entrypoint for an already allocated and partitioned E1+SP2+D1 job."""
    import multiprocessing as mp

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 2 or len(set(visible)) != 2 or not all(visible):
        raise ValueError("DiT requires two distinct pre-partitioned GPU devices")
    mp.set_start_method("spawn", force=True)
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        initialize_runtime_imports,
    )

    initialize_runtime_imports()
    from sglang.multimodal_gen.runtime.launch_server import dispatch_launch
    from sglang.multimodal_gen.runtime.server_args.server_args import (
        prepare_server_args,
    )
    from sglang.multimodal_gen.runtime.utils.common import kill_process_tree

    args = prepare_server_args(sys.argv[1:] if argv is None else argv)
    if args.num_gpus != 2 or any(
        (args.encoder_gpus, args.denoiser_gpus, args.decoder_gpus)
    ):
        raise ValueError("The parent launcher owns role allocation")
    args.disagg_encoder_port = 31001
    args.disagg_decoder_port = 31003
    try:
        dispatch_launch(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


B300_MODULE = "sglang.multimodal_gen.tools.zing_b300_launcher"
B300_ENVIRONMENT = {
    **COMMON_ENV,
    "SGLANG_LIGHTWEIGHT_RUNTIME": "1",
    "SGLANG_DIFFUSION_STAGE_LOGGING": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "TOKENIZERS_PARALLELISM": "false",
}


def b300_parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Single-node Zing B300 recipe: E1/Qwen + DiT SP2 + Lynn D1, using shared hardware defaults."
    )
    for name in (
        "model-path",
        "lynn-model-dir",
        "conditioning-service-config",
        "conditioning-config",
        "conditioning-profile",
        "run-dir",
    ):
        value.add_argument("--" + name, type=Path, required=True)
    value.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Four device IDs/UUIDs, in DiT0-1, Encoder/Qwen, Decoder order. Must be allocated to this deployment.",
    )
    value.add_argument("--host", default="0.0.0.0")
    value.add_argument("--quantization", choices=("none", "fp8"), default="fp8")
    value.add_argument(
        "--quantization-scope",
        choices=("ffn", "fc1", "interior-qkv-fc1", "ffn-attention"),
        default="ffn-attention",
    )
    value.add_argument("--port", type=int, default=30000)
    value.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without writing files or initializing CUDA.",
    )
    return value


def b300_build_plan(args, *, server_module: str = B300_MODULE) -> dict:
    devices = [part.strip() for part in args.gpus.split(",")]
    if len(devices) != 4 or any(not part for part in devices) or len(set(devices)) != 4:
        raise ValueError("--gpus requires four distinct device IDs or UUIDs")
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None and not set(devices) <= set(inherited.split(",")):
        raise ValueError(
            "--gpus must select devices from the inherited CUDA_VISIBLE_DEVICES allocation"
        )
    if not 1024 <= args.port <= 65530:
        raise ValueError("--port must be between 1024 and 65530")
    model, lynn, run = (
        p.resolve() for p in (args.model_path, args.lynn_model_dir, args.run_dir)
    )
    if run.is_relative_to(model):
        raise ValueError("--run-dir must be outside the source model directory")
    if run.exists():
        raise ValueError(
            "--run-dir must be a new directory; existing deployment files are never overwritten"
        )
    for path in (
        model / "zing_config.json",
        model / "zing_realtime_profile.json",
        lynn / "decoder.pt",
        args.conditioning_service_config,
        args.conditioning_config,
        args.conditioning_profile,
    ):
        if not path.is_file():
            raise ValueError(f"Required deployment file is missing: {path}")
    service = json.loads(args.conditioning_service_config.read_text())
    if service.get("native", {}).get("gpu_id") != 0:
        raise ValueError(
            "Conditioning service native.gpu_id must be 0 within its isolated GPU"
        )
    client = json.loads(args.conditioning_config.read_text())
    token = client.get("bearer_token", client.get("token"))
    if not isinstance(token, str) or token not in service.get("tokens", {}):
        raise ValueError(
            "Conditioning client token must be registered in the service configuration"
        )
    for config in (service, client):
        if config.get("result_transport", "nccl") != "nccl":
            raise ValueError(
                "This same-node recipe requires NCCL conditioning transport"
            )
    pipeline = {
        "decoder_config": {
            "type": "lynn_streaming",
            "model_dir": str(lynn),
            "checkpoint_path": str(lynn / "decoder.pt"),
            "precision": "bf16",
            "weights": "ema",
        },
        "dit_precision": "bf16",
        "zing_adaln_mode": "resident",
        "zing_kv_pool_window_chunks": 6,
        "zing_kv_pool_sink_chunks": 0,
        "zing_kv_pool_rope_position_mode": "block_relative",
        "zing_kv_pool_rope_max_frame_gap": 12,
        "zing_kv_pool_window_eviction_timing": "after_step0",
    }
    base = args.port
    policy = server_policy("b300", str(run / "block0-cache"))
    policy["--quantization"] = args.quantization
    if args.quantization == "none":
        policy.pop("--direct-gpu-weight-loading", None)
    server = [
        "--model-path",
        str(run / "model"),
        "--pipeline-config-path",
        str(run / "pipeline.json"),
        "--num-gpus",
        "2",
        "--sp-degree",
        "2",
        "--ulysses-degree",
        "2",
        "--ring-degree",
        "1",
        "--conditioning-service-url",
        f"http://127.0.0.1:{base + 2}",
        "--conditioning-expected-profile",
        str(args.conditioning_profile.resolve()),
        "--conditioning-config",
        str(args.conditioning_config.resolve()),
        "--host",
        args.host,
        "--port",
        str(base),
        "--scheduler-port",
        str(base + 4),
        "--master-port",
        str(base + 5),
        *policy_arguments(policy),
    ]
    environment = dict(B300_ENVIRONMENT)
    if args.quantization == "fp8":
        environment["SGLANG_ZING_QUANT_SCOPE"] = args.quantization_scope
    worker_prefix = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        "-m",
    ]
    stages = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3."
    return {
        "model_source": str(model),
        "run_dir": str(run),
        "pipeline": pipeline,
        "environment": environment,
        "roles": {
            "encoder": {
                "devices": devices[2],
                "command": worker_prefix
                + [
                    stages + "encoder_worker",
                    "--model",
                    str(run / "model"),
                    "--port",
                    str(base + 1),
                ],
            },
            "conditioning": {
                "devices": devices[2],
                "command": [
                    sys.executable,
                    "-m",
                    "sglang.multimodal_gen.runtime.conditioning.cli",
                    "serve",
                    "--config",
                    str(args.conditioning_service_config.resolve()),
                    "--profile",
                    str(args.conditioning_profile.resolve()),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(base + 2),
                ],
            },
            "decoder": {
                "devices": devices[3],
                "command": worker_prefix
                + [
                    stages + "decoder_worker",
                    "--static-connection",
                    "--model",
                    str(run / "model"),
                    "--port",
                    str(base + 3),
                    "--decoder-config",
                    json.dumps(pipeline["decoder_config"]),
                ],
            },
            "server": {
                "devices": ",".join(devices[:2]),
                "command": [
                    sys.executable,
                    "-m",
                    server_module,
                    "_server",
                    str(base + 1),
                    str(base + 3),
                    *server,
                ],
            },
        },
    }


def b300_materialize(plan: Mapping) -> None:
    run, source = Path(plan["run_dir"]), Path(plan["model_source"])
    run.mkdir(parents=True)
    model = run / "model"
    model.mkdir()
    for item in source.iterdir():
        if item.name != "zing_realtime_profile.json":
            (model / item.name).symlink_to(
                item.resolve(), target_is_directory=item.is_dir()
            )
    profile = json.loads((source / "zing_realtime_profile.json").read_text())
    profile["source_attention_window"] = {"window_chunks": 6, "sink_chunks": 0}
    profile["window_eviction_timing"] = plan["pipeline"][
        "zing_kv_pool_window_eviction_timing"
    ]
    profile.setdefault("source_encoder", {}).pop("remote", None)
    profile.pop("source_decoder", None)
    (model / "zing_realtime_profile.json").write_text(
        json.dumps(profile, indent=2) + "\n"
    )
    for name, value in (("pipeline", plan["pipeline"]), ("launch-plan", plan)):
        (run / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")


def b300_check_hardware(devices: str, env: Mapping[str, str]) -> list[dict]:
    code = "import json,torch; print(json.dumps([dict(cc=list(torch.cuda.get_device_capability(i)),mib=torch.cuda.get_device_properties(i).total_memory//(1<<20)) for i in range(torch.cuda.device_count())]))"
    result = subprocess.check_output(
        [sys.executable, "-c", code],
        env={**env, "CUDA_VISIBLE_DEVICES": devices},
        text=True,
    )
    gpus = json.loads(result)
    if len(gpus) != 4 or any(
        gpu["cc"] != [10, 3] or gpu["mib"] < 240 * 1024 for gpu in gpus
    ):
        raise ValueError(
            "This B300 recipe requires four CC10.3 devices with at least 240 GiB CUDA-visible memory each"
        )
    return gpus


def b300_wait_conditioning(
    process, url: str, token: str, timeout: float = 1800
) -> None:
    deadline = time.monotonic() + timeout
    request = urllib.request.Request(
        url + "/v1/conditioning/profile", headers={"Authorization": "Bearer " + token}
    )
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "Conditioning service exited before readiness; see conditioning.log"
            )
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError("Conditioning service readiness deadline exceeded")


def b300_stop_processes(processes) -> None:
    for sig, timeout in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        for process in reversed(processes):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        for process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass


def b300_check_ports(base: int) -> None:
    for port in range(base, base + 6):
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", port))


def b300_serve(
    args,
    plan: Mapping,
    *,
    check_ports_fn=None,
    check_hardware_fn=None,
    materialize_fn=None,
    wait_conditioning_fn=None,
    stop_processes_fn=None,
) -> None:
    from sglang.multimodal_gen.runtime.disaggregation.role_launcher import wait_ready

    env = {**os.environ, **plan["environment"]}
    if args.quantization == "none":
        env.pop("SGLANG_ZING_QUANT_SCOPE", None)
    devices = args.gpus
    (check_ports_fn or b300_check_ports)(args.port)
    hardware = (check_hardware_fn or b300_check_hardware)(devices, env)
    (materialize_fn or b300_materialize)(plan)
    run = Path(plan["run_dir"])
    (run / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    processes, logs = [], []

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    previous = {
        sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for name, role in plan["roles"].items():
            log = (run / (name + ".log")).open("wb")
            logs.append(log)
            process = subprocess.Popen(
                role["command"],
                env=role_metrics_environment(
                    name, {**env, "CUDA_VISIBLE_DEVICES": role["devices"]}
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            if name in ("encoder", "decoder"):
                wait_ready(
                    process, args.port + (1 if name == "encoder" else 3), timeout_s=1800
                )
            elif name == "conditioning":
                config = json.loads(args.conditioning_config.read_text())
                (wait_conditioning_fn or b300_wait_conditioning)(
                    process,
                    f"http://127.0.0.1:{args.port + 2}",
                    config.get("bearer_token", config.get("token")),
                )
        print(
            f"Zing B300 processes started; server log: {run / 'server.log'}", flush=True
        )
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        raise RuntimeError("A deployment process exited; see role logs")
    finally:
        (stop_processes_fn or b300_stop_processes)(processes)
        for log in logs:
            log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def b300_server_main(argv: list[str]) -> None:
    import multiprocessing as mp

    from sglang.multimodal_gen.runtime.launch_server import dispatch_launch
    from sglang.multimodal_gen.runtime.server_args import prepare_server_args

    mp.set_start_method("spawn", force=True)
    args = prepare_server_args(argv[2:])
    args.disagg_encoder_port, args.disagg_decoder_port = map(int, argv[:2])
    dispatch_launch(args)


def b300_main(
    argv: list[str] | None = None, *, server_module: str = B300_MODULE
) -> None:
    raw = sys.argv[1:] if argv is None else argv
    if raw[:1] == ["_server"]:
        b300_server_main(raw[1:])
        return
    cli = b300_parser()
    args = cli.parse_args(raw)
    try:
        plan = b300_build_plan(args, server_module=server_module)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
        else:
            b300_serve(args, plan)
    except (ValueError, OSError, RuntimeError, TimeoutError) as error:
        cli.exit(1, f"{error}\n")
    except KeyboardInterrupt:
        cli.exit(130, "Deployment stopped.\n")


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
        if profile == "b300" or argument_value(arguments, "--quantization") == "fp8":
            defaults["SGLANG_ZING_QUANT_SCOPE"] = "ffn-attention"
        defaults["SGLANG_ZING_FUSED_CURRENT_ROPE_KV"] = "1" if fixed else "0"
        defaults["SGLANG_ZING_FA4_FIXED"] = "1" if fixed else "0"
    effective = {name: environment.get(name, value) for name, value in defaults.items()}
    supplied = {arg.split("=", 1)[0] for arg in arguments if arg.startswith("--")}
    if component in ("server", "dit"):
        server_defaults = server_policy(
            profile, argument_value(arguments, "--zing-block0-cache-dir")
        )
        effective_initial_blocks = argument_value(
            arguments, "--zing-initial-block-count"
        )
        if effective_initial_blocks is None:
            effective_initial_blocks = server_defaults.get("--zing-initial-block-count")
        if effective_initial_blocks == "0":
            server_defaults.pop("--zing-initial-denoising-step-list", None)
        # Mooncake owns Block0 capacity and persistence through its mounted
        # config. Do not append the legacy local-disk budget defaults (which
        # can otherwise pin hundreds of GiB and make the command appear to
        # use the old local backend alongside Mooncake).
        if argument_value(arguments, "--zing-block0-cache-backend") == "mooncake":
            for option in (
                "--zing-block0-cache-host-gb",
                "--zing-block0-cache-host-total-gb",
                "--zing-block0-cache-disk-gb",
                "--zing-block0-cache-storage",
                "--zing-block0-cache-dir",
                "--zing-block0-cache-budget-root",
                "--zing-block0-cache-min-free-gb",
                "--zing-block0-cache-inflight-gb",
                "--zing-block0-cache-max-writers",
                "--zing-block0-cache-gc",
                "--zing-block0-cache-gc-high-watermark",
                "--zing-block0-cache-gc-low-watermark",
                "--zing-block0-cache-capacity-dir",
            ):
                server_defaults.pop(option, None)
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
    if sys.argv[1:2] == ["_fixed_roles_server"]:
        fixed_roles_main(sys.argv[2:])
        return
    if sys.argv[1:2] == ["_server"]:
        b300_server_main(sys.argv[2:])
        return
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
        if args.mode == "independent":
            environment = role_metrics_environment(
                args.component,
                environment,
                directory=environment.get("PROMETHEUS_MULTIPROC_DIR"),
            )
        os.execvpe(contract["command"][0], contract["command"], environment)


if __name__ == "__main__":
    main()
