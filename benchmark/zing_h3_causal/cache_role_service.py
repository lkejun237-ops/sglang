"""Launch one allocated test role from a complete, immutable model image.

This independently versioned deployment helper never overlays application code.
Its plan pins the image's runtime commit and read-only model metadata. It neither
allocates GPUs nor modifies other workers or Coordinator state.
"""

import argparse
import fcntl
import hashlib
import ipaddress
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

IMAGE_ROOT = Path("/opt/whitebox")


def validate_hardware(plan, device, visible_count):
    profile = plan.get("gpu_profile", "h200")
    if profile not in {"h100", "h200"} or visible_count != 1:
        raise ValueError("This role requires one explicitly selected H100/H200")
    if profile == "h100" and plan["role"] not in {"encoder", "decoder"}:
        raise ValueError("This test plan supports H100 only for Encoder/Decoder")
    minimum = {"h100": 70, "h200": 130}[profile] * 2**30
    if (device.major, device.minor) != (9, 0) or device.total_memory < minimum:
        raise ValueError("Allocated hardware differs from the frozen role profile")
    if profile == "h100" and device.total_memory >= 100 * 2**30:
        raise ValueError("The H100 plan must not silently consume an H200")
    gpu = str(device.uuid)
    if not gpu.startswith("GPU-"):
        gpu = "GPU-" + gpu
    expected = plan.get("expected_gpu_uuid")
    if profile == "h100" and not expected:
        raise ValueError("A host-managed H100 requires its reserved GPU UUID")
    if expected is not None and expected != gpu:
        raise ValueError("The visible GPU differs from the reserved GPU UUID")
    return gpu


def read_plan(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("The role plan differs from its frozen digest")
    value = json.loads(raw)
    if value["role"] not in {"encoder", "dit", "decoder"}:
        raise ValueError("Unknown model role")
    return value


def prepare(plan, root, host):
    resolution = plan.get("resolution", "480")
    if resolution not in ("480", "736"):
        raise ValueError("Expected a 480 or 736 role profile")
    release = json.loads((IMAGE_ROOT / "release.json").read_text())
    if release["source_sha"] != plan["runtime_source"]:
        raise ValueError("The complete runtime image and role plan differ")
    assets = Path(plan["asset_root"])
    if (
        hashlib.sha256((assets / "manifest.json").read_bytes()).hexdigest()
        != plan["asset_manifest_sha256"]
    ):
        raise ValueError("Model asset manifest changed")
    expected_config = plan["model_metadata"]["zing_config.json"]
    actual_config = json.loads((assets / "model/zing_config.json").read_text())
    if actual_config != expected_config:
        raise ValueError("Model configuration changed")
    if (
        actual_config["num_frame_per_block"] != [3, 2]
        or actual_config["split_first_latent"] is not False
    ):
        raise ValueError("Unexpected first-block geometry")
    model = root / "model"
    model.mkdir()
    for path in (assets / "model").iterdir():
        destination = model / path.name
        if path.name == "zing_config.json":
            # The profile binds the exact export bytes, not a reserialized dict.
            destination.symlink_to(path)
        elif path.name in plan["model_metadata"]:
            destination.write_text(json.dumps(plan["model_metadata"][path.name]))
        else:
            destination.symlink_to(path)
    profile = json.loads((model / "zing_realtime_profile.json").read_text())
    if (
        profile["model_config_sha256"]
        != hashlib.sha256((model / "zing_config.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Streaming profile and model configuration digest differ")
    cache_path = root / "cache.json"
    cache_path.write_text(json.dumps(plan["cache"]))
    role = plan["role"]
    encoder_ports = plan.get(
        "encoder_ports", {"http": 31000, "input_ws": 30000, "rendezvous": 32000}
    )
    if (
        set(encoder_ports) != {"http", "input_ws", "rendezvous"}
        or any(
            type(p) is not int or not 1024 <= p <= 65535 for p in encoder_ports.values()
        )
        or len(set(encoder_ports.values())) != 3
    ):
        raise ValueError("Encoder ports must be three distinct unprivileged ports")
    common = [
        "--model",
        str(model),
        "--advertise-host",
        host,
        "--worker-id",
        plan["worker_id"],
    ]
    port = {"encoder": encoder_ports["http"], "dit": 33000, "decoder": 36000}[role]
    native = [*common, "--http-port", str(port)]
    if role == "encoder":
        native += [
            "--port",
            str(encoder_ports["input_ws"]),
            "--rendezvous-port",
            str(encoder_ports["rendezvous"]),
            "--zing-block0-mooncake-config",
            str(cache_path),
        ]
    elif role == "dit":
        pipeline = dict(plan["pipeline"])
        pipeline["zing_adaln_table_path"] = str(assets / "adaln-table-unsplit")
        if "decoder_config" in pipeline:
            pipeline["decoder_config"] = {
                **pipeline["decoder_config"],
                "checkpoint_path": str(assets / "lynn/decoder.pt"),
                "model_dir": str(assets / "lynn"),
            }
        pipeline_path = root / "pipeline.json"
        pipeline_path.write_text(json.dumps(pipeline))
        native += [
            "--pipeline-config-path",
            str(pipeline_path),
            "--mem-fraction-static",
            "0.875",
            "--timeout-seconds",
            "600",
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
            "--zing-source-joint-forward",
            "cross_chunk",
            "--zing-solver",
            "few_step",
            "--zing-denoising-step-list",
            "1000,500",
            "--zing-initial-block-count",
            "0",
        ]
    else:
        decoder = {
            **plan["decoder_config"],
            "checkpoint_path": str(assets / "lynn/decoder.pt"),
            "model_dir": str(assets / "lynn"),
        }
        native += ["--decoder-config", json.dumps(decoder)]
    command = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.tools.zing_h3_profile_launcher",
        "--mode",
        "independent",
        "--component",
        role,
        "--profile",
        plan.get("gpu_profile", "h200"),
        "--resolution",
        resolution,
        "--sp-degree",
        "1",
        "--",
        *native,
    ]
    return cache_path, command, "http://" + host + ":" + str(port)


def repair_metadata_only(plan, root):
    """Restore original JSON bytes only when every parsed model value is equal."""
    source = Path(plan["asset_root"]) / "model/zing_config.json"
    target = root / "model/zing_config.json"
    raw, old = source.read_bytes(), target.read_bytes()
    if (
        json.loads(raw) != json.loads(old)
        or json.loads(raw) != plan["model_metadata"]["zing_config.json"]
    ):
        raise ValueError("Model values differ; refusing a metadata-only repair")
    expected = hashlib.sha256(raw).hexdigest()
    profile = json.loads((root / "model/zing_realtime_profile.json").read_text())
    if profile["model_config_sha256"] != expected:
        raise ValueError("The profile does not reference the original asset bytes")
    if old != raw:
        if target.is_symlink():
            raise ValueError("Refusing to alter a linked immutable asset")
        temporary = target.with_name("zing_config.json.metadata-repair")
        with temporary.open("xb") as stream:
            stream.write(raw)
        temporary.replace(target)
    return {
        "old_sha256": hashlib.sha256(old).hexdigest(),
        "sha256": expected,
        "parameters_unchanged": True,
        "application_source_changed": False,
    }


def stop_child(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--repair-metadata-only", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    plan = read_plan(args.plan, args.plan_sha256)
    # Import only the image's complete application. The helper mount is not an
    # application source search path.
    os.environ["PYTHONPATH"] = str(IMAGE_ROOT / "python")
    sys.path.insert(0, str(IMAGE_ROOT / "python"))
    import sglang

    if Path(sglang.__file__).resolve() != IMAGE_ROOT / "python/sglang/__init__.py":
        raise ValueError("Runtime source resolved outside the verified image")
    if args.repair_metadata_only:
        if (
            json.loads((IMAGE_ROOT / "release.json").read_text())["source_sha"]
            != plan["runtime_source"]
        ):
            raise ValueError("Runtime image identity changed")
        print(json.dumps(repair_metadata_only(plan, args.root)), flush=True)
        return
    args.root.mkdir(parents=True, exist_ok=False)
    host = "127.0.0.1"
    if not args.preflight:
        if plan.get("advertise_host"):
            host = str(ipaddress.IPv4Address(plan["advertise_host"]))
            if ipaddress.ip_address(host).is_loopback:
                raise ValueError("A shared test role must advertise its real host")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, 0))  # Prove the advertised address belongs here.
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                host = socket.inet_ntoa(
                    fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", b"net0"))[
                        20:24
                    ]
                )
    cache_path, command, address = prepare(plan, args.root, host)
    from sglang.multimodal_gen.runtime.realtime.cache_tiers import CacheTiers

    CacheTiers.parse(plan["cache"]["tiers"])
    if args.preflight:
        from sglang.multimodal_gen.runtime.realtime.source_transport import (
            SourceTransportBudget,
        )

        SourceTransportBudget().websocket_options()
        from sglang.multimodal_gen.configs.pipeline_configs.zing_h3_decoder import (
            normalize_decoder_spec,
        )

        if plan["role"] == "decoder":
            normalize_decoder_spec(
                plan["decoder_config"],
                video_decoder_backend=None,
                vae_decode_precision="bf16",
            )
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "role": plan["role"],
                    "runtime_source": plan["runtime_source"],
                    "gpu_test": False,
                }
            ),
            flush=True,
        )
        return
    import torch

    if torch.cuda.device_count() != 1:
        raise ValueError("This role requires exactly one allocated GPU")
    device = torch.cuda.get_device_properties(0)
    gpu = validate_hardware(plan, device, torch.cuda.device_count())
    environment = dict(os.environ)
    environment.update(plan["environment"])
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(name, None)
    for name, path in plan["secret_files"].items():
        secret = Path(path).read_text().strip()
        if len(secret) < 32:
            raise ValueError("A role credential is unavailable")
        environment[name] = secret
    worker = {**plan["worker_config"], "hardware_ids": [gpu]}
    worker_path = args.root / "worker.json"
    worker_path.write_text(json.dumps(worker))
    environment["SGLANG_ROLE_POOL_WORKER_CONFIG"] = str(worker_path)
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    children, handles = [], []
    try:
        if plan["role"] == "dit":
            handles.append((args.root / "store.log").open("w"))
            store = subprocess.Popen(
                [
                    sys.executable,
                    str(IMAGE_ROOT / "deploy/whitebox/cache_store.py"),
                    "--config",
                    str(cache_path),
                ],
                env=environment,
                stdout=handles[-1],
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append(store)
            ready = Path(plan["cache"]["tiers"]["state_directory"]) / "store-ready.json"
            deadline = time.monotonic() + 90
            while not ready.exists():
                if (
                    store.poll() is not None
                    or stopped.is_set()
                    or time.monotonic() >= deadline
                ):
                    raise RuntimeError("Native shared Store failed startup")
                stopped.wait(0.2)
        handles.append((args.root / "worker.log").open("w"))
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=handles[-1],
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append(process)
        deadline = time.monotonic() + 1800
        while True:
            if (
                any(p.poll() is not None for p in children)
                or stopped.is_set()
                or time.monotonic() >= deadline
            ):
                raise RuntimeError("Role startup interrupted or failed")
            try:
                with urllib.request.urlopen(address + "/health", timeout=2) as response:
                    health = json.load(response)
                if health.get("ready") is True:
                    break
            except OSError:
                pass
            stopped.wait(2)
        receipt = {
            "role": plan["role"],
            "runtime_source": plan["runtime_source"],
            "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "worker_id": plan["worker_id"],
            "gpu": gpu,
            "http": address,
            "ready_epoch": time.time(),
            "health": health,
        }
        (args.root / "ready.json").write_text(json.dumps(receipt, indent=2))
        print("CACHE_ROLE_READY=" + json.dumps(receipt), flush=True)
        while not stopped.wait(2):
            if any(p.poll() is not None for p in children):
                print(
                    "CACHE_ROLE_CHILD_EXIT=" + json.dumps([p.poll() for p in children]),
                    flush=True,
                )
                # PAI retains stdout after a failed Pod can no longer be exec'd.
                # Preserve a bounded failure tail, not only the wrapper error.
                for name in ("worker.log", "store.log"):
                    path = args.root / name
                    if path.exists():
                        with path.open("rb") as log:
                            log.seek(max(0, path.stat().st_size - (128 << 10)))
                            print("CACHE_ROLE_FAILURE_LOG=" + name, flush=True)
                            print(log.read().decode(errors="replace"), flush=True)
                raise RuntimeError(
                    "Owned role or cache process exited; no automatic restart"
                )
    finally:
        for process in reversed(children):
            stop_child(process)
        for handle in handles:
            handle.close()
        (args.root / "children-stopped.json").write_text(
            json.dumps([p.returncode for p in children])
        )


if __name__ == "__main__":
    main()
