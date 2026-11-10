# SPDX-License-Identifier: Apache-2.0
"""Bounded real-model load probe for the independent hardware-profile launcher.

Run prepare with no visible GPU in the final image/interpreter, then load on an
allocated GPU. Asset paths are explicit; shared checkpoints are never modified.
The health receipt proves loading only. Use WorkerGroupMaster separately for
real cross-host tensor traffic, and retain that separate acceptance boundary.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

MODULE = "sglang.multimodal_gen.tools.zing_h3_profile_launcher"


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(args):
    args.root.mkdir(parents=True, exist_ok=False)
    model = args.root / "model"
    model.mkdir()
    identities = {}
    for source in args.model.iterdir():
        target = model / source.name
        if source.name in {
            "zing_config.json",
            "model_index.json",
            "zing_realtime_profile.json",
        }:
            data = source.read_bytes()
            identities[source.name] = hashlib.sha256(data).hexdigest()
            target.write_bytes(data)
        else:
            target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    profile = json.loads((model / "zing_realtime_profile.json").read_text())
    profile["source_attention_window"] = {"window_chunks": 6, "sink_chunks": 0}
    profile["window_eviction_timing"] = "after_step0"
    write(model / "zing_realtime_profile.json", profile)
    decoder = {
        "type": "lynn_streaming",
        "model_dir": str(args.lynn),
        "checkpoint_path": str(args.lynn / "decoder.pt"),
        "precision": "bf16",
        "weights": "ema",
    }
    assert Path(decoder["checkpoint_path"]).is_file()
    write(args.root / "decoder.json", decoder)
    # Encoder owns native VAE and must not receive Lynn decoder settings.
    write(args.root / "encoder.json", {"zing_reference_max_images": 21})
    pipeline = {
        "decoder_config": decoder,
        "zing_kv_pool_window_chunks": 6,
        "zing_kv_pool_sink_chunks": 0,
        "zing_kv_pool_rope_position_mode": "block_relative",
        "zing_kv_pool_rope_max_frame_gap": 12,
        "zing_kv_pool_window_eviction_timing": "after_step0",
        "zing_adaln_mode": "weights_free",
        "zing_adaln_table_path": str(args.adaln_table) if args.adaln_table else None,
        "zing_reference_max_images": 21,
    }
    write(args.root / "dit.json", pipeline)
    for role in ("encoder", "decoder", "dit"):
        module = (
            "sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server"
            if role == "encoder"
            else "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3."
            + role
            + "_worker"
        )
        subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    write(
        args.root / "PREPARED.json",
        {
            "passed": True,
            "revision": args.revision,
            "original_metadata_sha256": identities,
            "model": str(args.model),
            "lynn": str(args.lynn),
        },
    )


def worker_command(args, role):
    command = [
        sys.executable,
        "-m",
        MODULE,
        "--mode",
        "independent",
        "--component",
        role,
        "--profile",
        "h200" if role == "dit" else "h100",
        "--resolution",
        "480",
        "--",
        "--model",
        str(args.root / "model"),
        "--advertise-host",
        args.advertise_host,
        "--http-port",
        str(args.http_port),
    ]
    if role == "decoder":
        command += ["--decoder-config", (args.root / "decoder.json").read_text()]
    else:
        command += [
            "--pipeline-config-path",
            str(args.root / ("dit.json" if role == "dit" else "encoder.json")),
        ]
    if role == "encoder":
        command += ["--port", str(args.http_port + 1)]
    if role == "dit":
        command += [
            "--zing-block0-cache-dir",
            str(args.root / "dit" / "block0-cache"),
            "--timeout-seconds",
            "300",
        ]
    return command


def load(args):
    prepared = json.loads((args.root / "PREPARED.json").read_text())
    assert prepared["passed"] and prepared["revision"] == args.revision
    output = args.root / args.role
    output.mkdir(exist_ok=False)
    retention = None
    if args.role == "dit":
        from benchmark_artifacts import ExperimentRetention

        retention = ExperimentRetention(output)
        retention.create_cache("block0-cache")
    command = worker_command(args, args.role)
    write(output / "command.json", command)
    receipt = {
        "role": args.role,
        "revision": args.revision,
        "loaded": False,
        "cleaned": False,
    }
    child = None
    stopped = False

    def terminate(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        with (output / "worker.log").open("wb") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            receipt["pid"] = child.pid
            deadline = time.monotonic() + args.startup_timeout
            opener = build_opener(ProxyHandler({}))
            while not stopped and time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(
                        f"Worker exited before readiness: {child.returncode}"
                    )
                try:
                    with opener.open(
                        f"http://127.0.0.1:{args.http_port}/health", timeout=2
                    ) as response:
                        health = json.load(response)
                    receipt.update(loaded=True, health=health)
                    write(output / "LOAD_READY.json", receipt)
                    break
                except (OSError, ValueError):
                    time.sleep(2)
            if not receipt["loaded"]:
                raise TimeoutError("Worker did not reach model readiness")
            receipt["nvidia_smi"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,name,memory.total,memory.used",
                    "--format=csv",
                ],
                text=True,
            )
            until = time.monotonic() + args.hold_seconds
            while (
                not stopped
                and time.monotonic() < until
                and not (args.root / "STOP").exists()
            ):
                if child.poll() is not None:
                    raise RuntimeError(f"Loaded worker exited: {child.returncode}")
                time.sleep(2)
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            receipt.update(cleaned=True, returncode=child.returncode)
        write(output / "LOAD_RESULT.json", receipt)
        if retention is not None:
            try:
                retention.finish(workers_stopped=receipt["cleaned"])
            finally:
                retention.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "load"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--lynn", type=Path)
    p.add_argument("--adaln-table", type=Path)
    p.add_argument("--role", choices=("encoder", "decoder", "dit"), default="encoder")
    p.add_argument("--advertise-host", required=True)
    p.add_argument("--http-port", type=int, required=True)
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--hold-seconds", type=float, default=300)
    args = p.parse_args()
    if args.mode == "prepare":
        if args.model is None or args.lynn is None:
            p.error("prepare requires --model and --lynn")
        prepare(args)
    else:
        load(args)


if __name__ == "__main__":
    main()
