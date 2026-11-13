# SPDX-License-Identifier: Apache-2.0
"""Run exact product replays with SM90 decoding off/on on an isolated GPU.

The one-GPU layout uses text CPU offload and is not the deployed three-role
performance layout. Both cells use identical settings. Full videos are encoded
only after request timing; all original frames and condition updates are kept.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

from benchmark_artifacts import ExperimentRetention


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def live_group_members(group):
    """Ignore reaped-worker zombies; they cannot execute or retain GPU memory."""
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = path.read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        fields = text[text.rfind(")") + 2 :].split()
        if int(fields[2]) == group and fields[0] not in ("Z", "X"):
            members.append(int(path.parent.name))
    return members


def terminate(signum, frame):
    raise SystemExit(128 + signum)


def main():
    signal.signal(signal.SIGTERM, terminate)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--lynn", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--adaln-table", type=Path, required=True)
    p.add_argument("--measured-runs", type=int, default=3)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--port", type=int, default=30693)
    args = p.parse_args()
    expected = os.environ.get("PR693_EXPECTED_GPU")
    if expected:
        visible = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
            )
            .strip()
            .splitlines()
        )
        if visible != [expected]:
            raise RuntimeError("Review GPU identity differs from reservation")
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if processes:
            raise RuntimeError("Review GPU became occupied before execution")
    # Catch missing GPU dependencies and deep-engine regressions before loading DiT.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(
                Path(__file__).resolve().parents[2]
                / "python/sglang/multimodal_gen/test/unit/test_zing_lynn_execution.py"
            ),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=s=16x16:r=24",
            "-frames:v",
            "1",
            "-c:v",
            "libx264",
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    checkout = Path(__file__).resolve().parents[2]
    model = args.output / "model"
    model.mkdir()
    for f in args.model.iterdir():
        if f.name != "zing_realtime_profile.json":
            (model / f.name).symlink_to(f.resolve())
    profile = json.loads((args.model / "zing_realtime_profile.json").read_text())
    profile["source_encoder"].pop("remote", None)
    write(model / "zing_realtime_profile.json", profile)
    pipeline = args.output / "pipeline.json"
    write(
        pipeline,
        {
            "decoder_config": {
                "type": "lynn_streaming",
                "checkpoint_path": str(args.lynn / "decoder.pt"),
                "model_dir": str(args.lynn),
                "precision": "bf16",
                "weights": "ema",
            },
            "zing_adaln_mode": "weights_free",
            "zing_adaln_table_path": str(args.adaln_table),
            "zing_kv_pool_window_chunks": 6,
            "zing_kv_pool_sink_chunks": 3,
        },
    )
    retention = ExperimentRetention(args.output)
    workers_stopped = True
    try:
        for cell, fast in (("decoder-block", "0"), ("decoder-sm90", "1")):
            dest = args.output / cell
            dest.mkdir()
            cache = retention.create_cache(f"scratch/{cell}")
            env = dict(os.environ, SGLANG_LYNN_SM90_FAST_DECODE=fast)
            # Validate the exact CLI cache settings before loading GPU models.
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.block0_identity import (
                startup_config,
            )

            cache_flags = {
                "--zing-block0-cache-mode": "on",
                "--zing-block0-cache-dir": str(cache),
                "--zing-block0-cache-host-gb": 16,
                "--zing-block0-cache-disk-gb": 64,
                "--zing-block0-cache-storage": "local",
                "--zing-block0-cache-min-free-gb": 64,
                "--zing-block0-cache-inflight-gb": 16,
                "--zing-block0-cache-max-writers": 1,
            }
            startup_config(
                SimpleNamespace(
                    **{
                        key[2:].replace("-", "_"): value
                        for key, value in cache_flags.items()
                    }
                )
            )
            command = [
                sys.executable,
                "-u",
                "-m",
                "sglang.multimodal_gen.runtime.launch_server",
                "--model-path",
                str(model),
                "--pipeline-config-path",
                str(pipeline),
                "--num-gpus",
                "1",
                "--sp-degree",
                "1",
                "--ulysses-degree",
                "1",
                "--ring-degree",
                "1",
                "--performance-mode",
                "speed",
                "--attention-backend",
                "torch_sdpa",
                "--dit-cpu-offload",
                "false",
                "--dit-layerwise-offload",
                "false",
                "--text-encoder-cpu-offload",
                "true",
                "--vae-cpu-offload",
                "false",
                "--use-fsdp-inference",
                "false",
                "--mem-fraction-static",
                "0.60",
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
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ]
            for flag, value in cache_flags.items():
                command.extend((flag, str(value)))
            write(
                dest / "deployment.json",
                {
                    "source_sha": args.source_sha,
                    "command": command,
                    "sm90_fast_decode": fast,
                    "layout": "single-GPU text-offload; not production three-role",
                },
            )
            with (dest / "server.log").open("w") as log:
                server = subprocess.Popen(
                    command,
                    env=env,
                    cwd=checkout,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                workers_stopped = False
                try:
                    deadline = time.monotonic() + 900
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(
                                f"{cell} server exited: {server.returncode}"
                            )
                        try:
                            with urlopen(
                                f"http://127.0.0.1:{args.port}/health", timeout=2
                            ) as r:
                                if r.status == 200:
                                    break
                        except OSError:
                            pass
                        if time.monotonic() > deadline:
                            raise TimeoutError("Server readiness timed out")
                        time.sleep(2)
                    client = [
                        sys.executable,
                        str(
                            checkout
                            / "benchmark/zing_h3_causal/bench_streaming_source.py"
                        ),
                        "--url",
                        f"ws://127.0.0.1:{args.port}/v1/realtime_video/generate",
                        "--case-set",
                        "product-case-20261103",
                        "--media-root",
                        str(args.media_root),
                        "--output-dir",
                        str(dest / "client"),
                        "--media-mode",
                        "full",
                        "--warmup-runs",
                        str(args.warmup_runs),
                        "--measured-runs",
                        str(args.measured_runs),
                        "--timeout",
                        "3600",
                        "--seed",
                        "123",
                    ]
                    write(dest / "client-command.json", client)
                    subprocess.run(client, cwd=checkout, env=env, check=True)
                finally:
                    if server.poll() is None:
                        os.killpg(server.pid, signal.SIGTERM)
                        try:
                            server.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            os.killpg(server.pid, signal.SIGKILL)
                            server.wait(timeout=15)
                    # A dead distributed worker may remain as a zombie until
                    # container PID 1 reaps it. Check live members, not killpg(0).
                    if live_group_members(server.pid):
                        os.killpg(server.pid, signal.SIGKILL)
                        deadline = time.monotonic() + 10
                        while (
                            live_group_members(server.pid)
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.2)
                    workers_stopped = not live_group_members(server.pid)
                    if not workers_stopped:
                        raise RuntimeError("Owned worker process group is still alive")
            write(dest / "COMPLETE.json", {"complete": True})
        write(args.output / "COMPLETE.json", {"complete": True})
    finally:
        try:
            retention.finish(workers_stopped=workers_stopped)
        finally:
            retention.close()


if __name__ == "__main__":
    main()
