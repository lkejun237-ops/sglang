# SPDX-License-Identifier: Apache-2.0
"""Run the fixed Zing dataset from a published SGLang checkpoint."""

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import bench_client
from build_benchmark_report import build_report, load_run
from prepare_whitebox_video_smoke import open_media


def checkpoint_settings(model):
    """Read published sampling settings without importing the GPU runtime."""
    runtime_path = model / "zing_config.json"
    profile_path = model / "zing_realtime_profile.json"
    runtime = json.loads(runtime_path.read_text())
    profile = json.loads(profile_path.read_text())
    steps = profile.get("denoising_step_list")
    if (
        runtime.get("causal") is not True
        or profile.get("model_config_sha256") != bench_client.sha256_file(runtime_path)
        or profile.get("solver") != "few_step"
        or not isinstance(steps, list)
        or len(steps) != 4
        or any(type(step) is not int for step in steps)
        or any(a <= b for a, b in zip(steps, steps[1:]))
        or not 0 < steps[-1] < steps[0] <= runtime["num_train_timestep"]
        or type(profile.get("guidance_scale")) not in (int, float)
        or not math.isfinite(profile["guidance_scale"])
    ):
        raise ValueError("Checkpoint realtime profile is missing or inconsistent")
    return {
        "seed": 123,
        "num_inference_steps": len(steps),
        "guidance_scale": profile["guidance_scale"],
        "zing_solver": profile["solver"],
        "zing_denoising_step_list": steps,
        "zing_realtime_decode_mode": "stable",
    }, {
        "model_path": str(model),
        "model_config_sha256": bench_client.sha256_file(runtime_path),
        "realtime_profile_sha256": bench_client.sha256_file(profile_path),
    }


def prepare_media(selected, root, refresh=False, receipt=None):
    """Verify cached assets or atomically publish downloads outside request timing."""
    if receipt is None:
        receipt = {}
    receipt.update(assets=[], status="failed")
    started = time.monotonic()
    seen = set()
    try:
        for label, manifest, case_id in selected:
            case = bench_client.load_cases(manifest)[case_id]
            for uri, asset in bench_client.expected_assets(case).items():
                if uri in seen:
                    continue
                seen.add(uri)
                path = bench_client.local_source_path(uri, root)
                entry = dict(
                    uri=uri,
                    path=str(path.resolve()),
                    bytes=asset["bytes"],
                    sha256=asset["sha256"],
                    cache_state="refresh"
                    if refresh
                    else "hit"
                    if path.exists()
                    else "miss",
                    status="failed",
                )
                receipt["assets"].append(entry)
                asset_started = time.monotonic()
                try:
                    if path.exists() and not refresh:
                        if (
                            path.stat().st_size != asset["bytes"]
                            or bench_client.sha256_file(path) != asset["sha256"]
                        ):
                            raise ValueError(
                                f"Cached benchmark asset does not match: {path}"
                            )
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        # Keep the last valid entry until its replacement is verified.
                        with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
                            staged = Path(temporary) / "asset"
                            with (
                                staged.open("wb") as target,
                                open_media(uri, None) as source,
                            ):
                                while data := source.read(1024 * 1024):
                                    target.write(data)
                            if (
                                staged.stat().st_size != asset["bytes"]
                                or bench_client.sha256_file(staged) != asset["sha256"]
                            ):
                                raise ValueError(
                                    f"Downloaded benchmark asset does not match: {uri}"
                                )
                            os.replace(staged, path)
                    entry["status"] = "verified"
                finally:
                    entry["duration_seconds"] = time.monotonic() - asset_started
        receipt["status"] = "verified"
        return receipt
    finally:
        receipt["duration_seconds"] = time.monotonic() - started


def wait_ready(process, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "Benchmark server exited before readiness; see server.log"
            )
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    raise TimeoutError("Benchmark server startup timed out; see server.log")


def stop_server(process):
    """Clean up only this runner's process group, including GPU workers."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-path", type=Path, required=True)
    result.add_argument(
        "--source-revision", help="Full Git SHA when running a frozen source archive"
    )
    selection = result.add_mutually_exclusive_group()
    selection.add_argument(
        "--case", choices=["all", *bench_client.DEFAULT_CASES], default="all"
    )
    selection.add_argument(
        "--case-set", help="Collection name or JSON file; default: whitebox-basic"
    )
    result.add_argument(
        "--media-root", type=Path, default=Path.home() / ".cache/sglang/zing-benchmark"
    )
    result.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-download selected assets from their original URIs and verify before replacement",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results") / datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    result.add_argument(
        "--pipeline-config",
        type=Path,
        help="Optional decoder/deployment overrides; otherwise use checkpoint pipeline_config.json or native decoder",
    )
    result.add_argument("--sp-degree", type=int, choices=(1, 2), default=2)
    result.add_argument("--port", type=int, default=30000)
    result.add_argument("--warmup-runs", type=int, default=1)
    result.add_argument("--measured-runs", type=int, default=3)
    result.add_argument("--stage-timing", choices=("off", "host"), default="host")
    result.add_argument("--startup-timeout", type=float, default=900)
    result.add_argument("--timeout", type=float, default=900)
    result.add_argument("--preflight-only", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("Select a valid port")
    if (
        args.warmup_runs < 1
        or args.measured_runs < 1
        or any(
            not math.isfinite(value) or value <= 0
            for value in (args.timeout, args.startup_timeout)
        )
    ):
        raise ValueError("Warmup/measured counts and timeouts must be positive")
    model = args.model_path.resolve(strict=True)
    sampling, deployment = checkpoint_settings(model)
    bundled = model / "pipeline_config.json"
    pipeline = args.pipeline_config or (bundled if bundled.is_file() else None)
    overrides = (
        json.loads(pipeline.read_text())
        if pipeline
        else {"video_decoder_backend": "native"}
    )
    if not isinstance(overrides, dict):
        raise ValueError("Pipeline configuration must be a JSON object")
    decoder_config = overrides.get("decoder_config")
    if decoder_config is not None:
        for key in ("checkpoint_path", "model_dir"):
            if decoder_config.get(key):
                path = Path(decoder_config[key])
                decoder_config[key] = str(
                    (path if path.is_absolute() else model / path).resolve(strict=True)
                )
    collection, selected = bench_client.select_cases(args)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    preparation = {}
    try:
        prepare_media(selected, args.media_root, args.refresh_cache, preparation)
    finally:
        bench_client.write_json(output / "media_preparation.json", preparation)
    deployment["media_preparation"] = preparation
    for label, manifest, case_id in selected:
        bench_client.prepare_case(label, manifest, args.media_root, sampling, case_id)
    bench_client.write_json(output / "sampling.json", sampling)
    bench_client.write_json(output / "pipeline.json", overrides)
    command = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.launch_server",
        "--model-path",
        str(model),
        "--pipeline-config-path",
        str(output / "pipeline.json"),
        "--num-gpus",
        str(args.sp_degree),
        "--sp-degree",
        str(args.sp_degree),
        "--ulysses-degree",
        str(args.sp_degree),
        "--ring-degree",
        "1",
        "--encoder-parallel",
        "replicate",
        "--attention-backend",
        "torch_sdpa",
        "--performance-mode",
        "speed",
        "--zing-solver",
        "few_step",
        "--zing-denoising-step-list",
        "1000,500",
        "--zing-initial-block-count",
        "3",
        "--zing-initial-denoising-step-list",
        "1000,750,500,250",
        "--warmup-mode",
        "off",
        "--dit-cpu-offload",
        "false",
        "--dit-layerwise-offload",
        "false",
        "--text-encoder-cpu-offload",
        "false",
        "--vae-cpu-offload",
        "false",
        "--use-fsdp-inference",
        "false",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    deployment.update(
        sampling_schedule={
            "solver": "few_step",
            "initial_block_count": 3,
            "initial_denoising_step_list": [1000, 750, 500, 250],
            "denoising_step_list": [1000, 500],
        },
        layout="finite-local-encoding",
        resident=True,
        concurrency=1,
        sp_degree=args.sp_degree,
        pipeline_config=overrides,
        server_command=command,
        stage_timing=args.stage_timing,
    )
    bench_client.write_json(output / "deployment.json", deployment)
    client = [
        "--url",
        f"ws://127.0.0.1:{args.port}/v1/realtime_video/generate",
        "--case",
        args.case,
        "--media-root",
        str(args.media_root.resolve()),
        "--sampling-params",
        str(output / "sampling.json"),
        "--deployment",
        str(output / "deployment.json"),
        "--output-dir",
        str(output / "client"),
        "--warmup-runs",
        str(args.warmup_runs),
        "--measured-runs",
        str(args.measured_runs),
        "--timeout",
        str(args.timeout),
        "--save-media",
    ]
    if args.stage_timing != "off":
        client.append("--receive-stage-timing")
    else:
        client.append("--no-stage-timing")
    if args.case_set:
        # Resolve before server startup; the client consumes the same collection file.
        client[client.index("--case") : client.index("--case") + 2] = [
            "--case-set",
            collection["path"],
        ]
    if args.source_revision:
        client.extend(["--source-revision", args.source_revision])
    if args.preflight_only:
        bench_client.main([*client, "--preflight-only"])
        return
    # Refuse an occupied endpoint instead of benchmarking an unrelated service.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    env = dict(os.environ)
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            wait_ready(process, args.port, args.startup_timeout)
            bench_client.main(client)
        finally:
            stop_server(process)
            if (output / "client/summary.json").is_file():
                build_report(
                    [load_run(args.stage_timing, output / "client")],
                    output / "report.html",
                )
    print(output / "report.html")


if __name__ == "__main__":
    main()
