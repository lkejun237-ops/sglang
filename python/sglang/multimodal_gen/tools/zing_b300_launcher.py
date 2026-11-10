# SPDX-License-Identifier: Apache-2.0
"""Single-node Zing B300 recipe: E1/Qwen + DiT SP2 + Lynn D1, using shared hardware defaults."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from sglang.multimodal_gen.tools.zing_h3_profile_launcher import (
    COMMON_ENV,
    policy_arguments,
    server_policy,
)

MODULE = "sglang.multimodal_gen.tools.zing_b300_launcher"
ENVIRONMENT = {
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


def parser():
    value = argparse.ArgumentParser(description=__doc__)
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


def build_plan(args):
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
    environment = dict(ENVIRONMENT)
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
                    MODULE,
                    "_server",
                    str(base + 1),
                    str(base + 3),
                    *server,
                ],
            },
        },
    }


def materialize(plan):
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
    # The streaming request policy and the pool must agree; pool size alone does not select W6.
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


def check_hardware(devices, env):
    # Query in a disposable child so no supervisor/DiT parent owns a CUDA context.
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


def wait_conditioning(process, url, token, timeout=1800):
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


def stop_processes(processes):
    # Dedicated process groups include torchrun ranks and scheduler descendants.
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


def check_ports(base):
    # Reject an existing listener before any readiness probe can reach another deployment.
    for port in range(base, base + 6):
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", port))


def serve(args, plan):
    from sglang.multimodal_gen.runtime.disaggregation.role_launcher import wait_ready

    env = {**os.environ, **plan["environment"]}
    if args.quantization == "none":
        env.pop("SGLANG_ZING_QUANT_SCOPE", None)
    devices = args.gpus
    check_ports(args.port)
    hardware = check_hardware(devices, env)
    materialize(plan)
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
                env={**env, "CUDA_VISIBLE_DEVICES": role["devices"]},
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
                wait_conditioning(
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
        stop_processes(processes)
        for log in logs:
            log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def server_main(argv):
    # Narrowing happens in the subprocess environment before importing runtime/platform code.
    import multiprocessing as mp

    from sglang.multimodal_gen.runtime.launch_server import dispatch_launch
    from sglang.multimodal_gen.runtime.server_args import prepare_server_args

    mp.set_start_method("spawn", force=True)
    args = prepare_server_args(argv[2:])
    args.disagg_encoder_port, args.disagg_decoder_port = map(int, argv[:2])
    dispatch_launch(args)


def main():
    if sys.argv[1:2] == ["_server"]:
        server_main(sys.argv[2:])
        return
    cli = parser()
    args = cli.parse_args()
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
        else:
            serve(args, plan)
    except (ValueError, OSError, RuntimeError, TimeoutError) as error:
        cli.exit(1, f"{error}\n")
    except KeyboardInterrupt:
        cli.exit(130, "Deployment stopped.\n")


if __name__ == "__main__":
    main()
