"""Bounded, isolated E1/SP2/D1 service for server-side cache acceptance.

Run only inside an explicitly allocated four-GPU test job. This does not join a
production pool, change routing, restart workers, or discover free GPUs. Clients
use WorkerGroupMaster and the ordinary Encoder/Decoder protocols separately.
"""

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

from worker_group_master import LocalWorkers, http_request

REPO = Path(__file__).resolve().parents[2]
MODULE = "sglang.multimodal_gen.tools.zing_h3_profile_launcher"


def prepare(plan, root, host):
    """Generate config only; model bytes remain on the read-only asset mount."""
    source = Path(plan["asset_root"])
    manifest = source / "manifest.json"
    if (
        hashlib.sha256(manifest.read_bytes()).hexdigest()
        != plan["asset_manifest_sha256"]
    ):
        raise ValueError("Model asset manifest differs from the frozen test input")
    model_config = json.loads((source / "model/zing_config.json").read_text())
    if (
        model_config["num_frame_per_block"] != [3, 2]
        or model_config["split_first_latent"] is not False
        or model_config["checkpoint_sha256"] != plan["checkpoint_sha256"]
    ):
        raise ValueError("Model identity or first-block geometry changed")
    # Config/profile files are immutable input; no runtime source overrides.
    model = root / "model"
    model.mkdir()
    for entry in (source / "model").iterdir():
        (model / entry.name).symlink_to(entry)
    pipeline = dict(plan["dit_pipeline"])
    pipeline["zing_adaln_table_path"] = str(source / "adaln-table-unsplit")
    decoder = pipeline["decoder_config"] = dict(pipeline["decoder_config"])
    decoder.update(
        checkpoint_path=str(source / "lynn/decoder.pt"), model_dir=str(source / "lynn")
    )
    dit = root / "dit-pipeline.json"
    dit.write_text(json.dumps(pipeline))
    encoder = root / "encoder-pipeline.json"
    encoder.write_text(
        json.dumps({"zing_reference_max_images": pipeline["zing_reference_max_images"]})
    )
    cache = json.loads(
        (REPO / "deploy/whitebox/cache/test-shanghai-copy.example.json").read_text()
    )
    cache["tiers"].update(
        state_directory=str(root / "cache-state"),
        disk_directory=str(root / "cache-disk"),
        allow_overlay_l2=True,
        prefix=plan["cache_prefix"],
    )
    if not cache["tiers"]["prefix"].startswith(
        "sglang-block0-cache/v2/test-shanghai-copy/validation/"
    ):
        raise ValueError("Test cache must have an isolated validation prefix")
    cache_path = root / "cache.json"
    cache_path.write_text(json.dumps(cache))
    native = ["--model", str(model), "--advertise-host", host]
    role_args = {
        "encoder": [
            *native,
            "--pipeline-config-path",
            str(encoder),
            "--zing-block0-mooncake-config",
            str(cache_path),
            "--http-port",
            "31000",
            "--port",
            "30000",
        ],
        "dit": [
            *native,
            "--pipeline-config-path",
            str(dit),
            "--http-port",
            "33000",
            "--zing-block0-cache-mode",
            "on",
            "--zing-block0-cache-backend",
            "mooncake",
            "--zing-block0-mooncake-config",
            str(cache_path),
            "--zing-block0-cache-storage",
            "local",
            "--zing-block0-cache-dir",
            str(root / "unused-local-cache"),
            "--timeout-seconds",
            "600",
        ],
        "decoder": [
            *native,
            "--decoder-config",
            json.dumps(decoder),
            "--http-port",
            "36000",
        ],
    }
    workers = {}
    for role, gpu in (("encoder", "0"), ("dit", "1,2"), ("decoder", "3")):
        command = [
            sys.executable,
            "-m",
            MODULE,
            "--mode",
            "independent",
            "--component",
            role,
            "--profile",
            "h200",
            "--resolution",
            "480",
        ]
        if role == "dit":
            command += ["--sp-degree", "2"]
        workers[role] = {
            "http": f"http://{host}:"
            + {"encoder": "31000", "dit": "33000", "decoder": "36000"}[role],
            "cwd": str(REPO),
            "env": {"CUDA_VISIBLE_DEVICES": gpu},
            "command": [*command, "--", *role_args[role]],
        }
    (root / "workers.json").write_text(json.dumps({"workers": workers}, indent=2))
    return cache_path, workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--seconds", type=int, default=7200)
    args = parser.parse_args()
    if not 600 <= args.seconds <= 10800:
        raise ValueError("Test lifetime must be bounded to 10 minutes through 3 hours")
    plan = json.loads(args.plan.read_text())
    source = json.loads((REPO / "release.json").read_text())["source_sha"]
    if source != plan["source_sha"]:
        raise ValueError(
            "The test plan and complete image have different source identities"
        )
    args.root.mkdir(parents=True, exist_ok=False)
    host = "127.0.0.1"
    if not args.preflight:
        # The CUDA dependency image need not contain the iproute2 executable.
        # Read PAI's VPC interface without installing runtime dependencies.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            try:
                address = fcntl.ioctl(
                    sock.fileno(), 0x8915, struct.pack("256s", b"net0")
                )
                host = socket.inet_ntoa(address[20:24])
            except OSError:
                host = socket.gethostbyname(socket.gethostname())
    cache, workers = prepare(plan, args.root, host)
    if args.preflight:
        subprocess.run(
            [
                sys.executable,
                str(REPO / "deploy/whitebox/verify_cache_dependencies.py"),
                "--allow-missing-driver",
            ],
            check=True,
            timeout=90,
        )
        # Exercise the real transport version gate without allocating a GPU.
        from sglang.multimodal_gen.runtime.realtime.source_transport import (
            SourceTransportBudget,
        )

        SourceTransportBudget().websocket_options()
        for module in (
            "sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server",
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_worker",
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker",
        ):
            subprocess.run(
                [sys.executable, "-m", module, "--help"],
                check=True,
                timeout=90,
                stdout=subprocess.DEVNULL,
            )
        (args.root / "PREP_COMPLETE.json").write_text(
            json.dumps(
                {
                    "source": source,
                    "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                    "gpu_test": False,
                }
            )
        )
        print("CACHE_SERVICE_PREFLIGHT_PASS", flush=True)
        return
    import torch

    if torch.cuda.device_count() != 4:
        raise ValueError("Exactly four scheduler-allocated GPUs must be visible")
    # A test-owned management capability, never printed or put in model logs.
    # The test client can read this exact private file through the job terminal.
    if not os.environ.get("SGLANG_CACHE_ADMIN_TOKEN"):
        token = secrets.token_urlsafe(48)
        path = args.root / "cache-admin-token.secret"
        with os.fdopen(
            os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w"
        ) as stream:
            stream.write(token)
        os.environ["SGLANG_CACHE_ADMIN_TOKEN"] = token
    hardware = [
        {
            "uuid": str(torch.cuda.get_device_properties(i).uuid),
            "bytes": torch.cuda.get_device_properties(i).total_memory,
        }
        for i in range(4)
    ]
    stopped = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: stopped.set())
    launcher = LocalWorkers(args.root / "workers.json", args.root / "logs")
    store = None
    deadline = time.monotonic() + args.seconds
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        os.environ.pop(name, None)
    with (args.root / "store.log").open("w") as log:
        try:
            store = subprocess.Popen(
                [
                    sys.executable,
                    str(REPO / "deploy/whitebox/cache_store.py"),
                    "--config",
                    str(cache),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            ready = args.root / "cache-state/store-ready.json"
            until = time.monotonic() + 90
            while not ready.exists():
                if (
                    store.poll() is not None
                    or stopped.is_set()
                    or time.monotonic() >= until
                ):
                    raise RuntimeError("Shared Store failed readiness")
                stopped.wait(0.2)
            addresses = launcher.start_missing({})
            until = time.monotonic() + 1800
            while True:
                launcher.check()
                if stopped.is_set() or time.monotonic() >= until:
                    raise RuntimeError("Model readiness interrupted or timed out")
                try:
                    health = {
                        role: http_request(address, "/health", timeout=2)
                        for role, address in addresses.items()
                    }
                    break
                except (OSError, RuntimeError):
                    stopped.wait(2)
            (args.root / "ready.json").write_text(
                json.dumps(
                    {
                        "source": source,
                        "hardware": hardware,
                        "addresses": addresses,
                        "health": health,
                    },
                    indent=2,
                )
            )
            print("CACHE_MODEL_SERVICE_READY", json.dumps(addresses), flush=True)
            while not stopped.wait(2) and time.monotonic() < deadline:
                launcher.check()
                if store.poll() is not None:
                    raise RuntimeError("Shared Store exited")
        finally:
            try:
                launcher.close()
            finally:
                if store is not None and store.poll() is None:
                    store.terminate()
                    store.wait(timeout=30)
            (args.root / "children-stopped.json").write_text(
                json.dumps(
                    {
                        "workers": [p.poll() for _, p, _ in launcher.owned],
                        "store": store.poll() if store else None,
                    }
                )
            )


if __name__ == "__main__":
    main()
