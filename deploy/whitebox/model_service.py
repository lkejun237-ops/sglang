"""Manual, all-role launcher for an existing four-GPU allocation.

This does not allocate GPUs, submit PAI jobs, stop other processes, copy weights,
clear caches, alter admission or activate a regional deployment. The platform
operator owns those actions. Without --run it only prints the command plan.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

from config import fingerprint, render
from output_paths import outside_checkout
from prepare_assets import verify_prepared

MODULE = "sglang.multimodal_gen.runtime."
HERE = Path(__file__).resolve().parent


def commands(bundle, *, python=sys.executable):
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
    workers = MODULE + "pipelines_core.stages.model_specific_stages.zing_h3."
    dit = [
        python,
        str(HERE / "fixed_roles.py"),
        "--model-path",
        str(root / "model"),
        "--pipeline-config-path",
        str(root / "pipeline.json"),
        "--mem-fraction-static",
        str(m["mem_fraction_static"]),
        "--num-gpus",
        "2",
        "--sp-degree",
        "2",
        "--ulysses-degree",
        "2",
        "--ring-degree",
        "1",
        "--encoder-parallel",
        "replicate",
        "--attention-backend",
        "torch_sdpa",
        "--performance-mode",
        "speed",
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
        "--realtime-max-sessions",
        "1",
        "--realtime-max-sessions-per-worker",
        "1",
        "--host",
        "0.0.0.0",
        "--port",
        "30000",
        "--scheduler-port",
        "31005",
        "--master-port",
        "31006",
        "--dist-timeout",
        "600",
        "--realtime-session-lease-ttl-s",
        "1200",
        "--realtime-session-idle-timeout-s",
        "1200",
        "--realtime-session-max-lifetime-s",
        str(max(7200, c["play_seconds"] + 600)),
        "--zing-solver",
        "few_step",
        "--zing-denoising-step-list",
        ",".join(map(str, m["denoising_steps"])),
        "--zing-initial-block-count",
        str(m["initial_blocks"]),
        "--zing-initial-denoising-step-list",
        ",".join(map(str, m["initial_steps"])),
        "--zing-reference-max-images",
        str(m["reference_max_images"]),
        "--zing-block0-cache-mode",
        "on",
        "--zing-block0-cache-host-gb",
        str(m["cache_host_gb"]),
        "--zing-block0-cache-dir",
        m["cache_dir"],
        "--zing-block0-cache-disk-gb",
        str(m["cache_disk_gb"]),
        "--zing-block0-cache-storage",
        "local",
        "--zing-conditioning-mode",
        "remote",
        "--conditioning-service-url",
        "http://127.0.0.1:31002",
        "--conditioning-expected-profile",
        str(root / "conditioning-profile.json"),
        "--conditioning-config",
        str(root / "worker.secret.json"),
    ]
    # One local filesystem policy is shared by all instances, not multiplied
    # by their rank/worker count. Old bundles get the same conservative guards.
    dit += [
        "--zing-block0-cache-gc",
        "--zing-block0-cache-budget-root",
        str(Path(m["cache_dir"]).parent),
        "--zing-block0-cache-min-free-gb",
        str(m.get("cache_disk_reserve_gb", 128)),
        "--zing-block0-cache-inflight-gb",
        str(m.get("cache_disk_inflight_gb", 128)),
        "--zing-block0-cache-max-writers",
        str(m.get("cache_disk_max_writers", 2)),
        "--zing-block0-cache-gc-high-watermark",
        "0.9",
        "--zing-block0-cache-gc-low-watermark",
        "0.8",
        "--zing-block0-cache-capacity-dir",
        m.get("cache_capacity_dir", "/var/lib/sglang/capacity"),
        "--zing-block0-cache-host-total-gb",
        str(m.get("cache_host_total_gb", m["cache_host_gb"])),
    ]
    if m.get("quantization", "none") == "fp8":
        # Native online quantization needs the source BF16 weights on Host
        # before post-load conversion; it is not a serialized FP8 checkpoint.
        dit += ["--quantization", "fp8", "--direct-gpu-weight-loading", "false"]
    # Explicit CLI values win over ServerArgs defaults and parent environment.
    # In particular quantum=0 must disable the optional quantum-one scheduling.
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
                MODULE + "conditioning.cli",
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
                MODULE + "entrypoints.realtime_worker_heartbeat",
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


def hardware(bundle, expected):
    import torch

    if torch.cuda.device_count() != 4 or len(expected) != 4 or len(set(expected)) != 4:
        raise ValueError("Exactly four explicitly owned GPU UUIDs must be visible")
    facts = []
    for i in range(4):
        p = torch.cuda.get_device_properties(i)
        uid = str(p.uuid)
        uid = uid if uid.startswith("GPU-") else "GPU-" + uid
        if (
            uid != expected[i]
            or [p.major, p.minor] != bundle["policy"]["gpu"]["capability"]
            or p.total_memory < bundle["policy"]["gpu"]["minimum_memory_gib"] * 2**30
        ):
            raise ValueError("Allocated GPUs do not match the reviewed regional plan")
        facts.append(
            {
                "uuid": uid,
                "memory_bytes": p.total_memory,
                "capability": [p.major, p.minor],
            }
        )
    return facts


def stop_children(children):
    # Own process groups only; never search for/kill arbitrary GPU users.
    for process, log in reversed(children):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    for process, log in reversed(children):
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        log.close()


def wait_http(process, url, headers, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Owned role exited before readiness")
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=3
            ) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError("Owned role readiness timed out")


def run(bundle, expected, *, retries=0):
    if retries != 0:
        raise ValueError(
            "Hard faults require quarantining and manual recovery; automatic restart is disabled"
        )
    state = outside_checkout(bundle["bindings"]["state_root"])
    state.mkdir(parents=True, exist_ok=True)
    with (state / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        facts = hardware(bundle, expected)
        root = Path(bundle["bindings"]["asset_root"])
        verify_prepared(root, bundle)
        pipeline = json.loads((root / "pipeline.json").read_text())
        config = json.loads((root / "model/zing_config.json").read_text())
        if "sink_chunks" in bundle["policy"]["model"]:
            profile = json.loads(
                (root / "model/zing_realtime_profile.json").read_text()
            )
            window = profile.get("source_attention_window", {})
            if (
                type(window.get("window_chunks")) is not int
                or window["window_chunks"] < 4
                or window.get("sink_chunks") != 0
                or pipeline.get("zing_kv_pool_sink_chunks") != 0
                or pipeline.get("zing_kv_pool_window_chunks") != window["window_chunks"]
            ):
                raise ValueError(
                    "Prepare matching S0 profile and KV pool before launching Whitebox"
                )
        if (
            config.get("num_frame_per_block") != [3, 2]
            or config.get("split_first_latent") is not False
        ):
            raise ValueError(
                "Prepared model must preserve [3,2] and split_first_latent=false"
            )
        mode = bundle["policy"]["model"]["adaln_mode"]
        if mode == "resident" and (
            pipeline.get("zing_adaln_mode") != "resident"
            or pipeline.get("zing_adaln_table_path")
        ):
            raise ValueError("Expected prepared resident AdaLN configuration")
        if mode == "table" and not pipeline.get("zing_adaln_table_path"):
            raise ValueError("Expected prepared AdaLN table configuration")
        worker_config = json.loads((root / "worker.secret.json").read_text())
        token = worker_config.get("bearer_token", worker_config.get("token"))
        if not isinstance(token, str) or not token:
            raise ValueError("Conditioning credential is missing")
        if (
            config.get("checkpoint_sha256")
            != bundle["policy"]["model"]["checkpoint_sha256"]
        ):
            raise ValueError("Prepared checkpoint identity differs from release policy")
        plan = commands(bundle)
        plan["decoder"]["argv"] += [
            "--decoder-config",
            json.dumps(pipeline["decoder_config"], sort_keys=True),
        ]
        evidence = {
            "config_sha256": bundle["config_sha256"],
            "hardware": facts,
            "asset_config_sha256": {
                str(p.relative_to(root)): __import__("hashlib")
                .sha256(p.read_bytes())
                .hexdigest()
                for p in (
                    root / "pipeline.json",
                    root / "model/zing_config.json",
                    root / "model/zing_realtime_profile.json",
                    root / "conditioning-profile.json",
                )
            },
        }
        (state / "launch.json").write_text(json.dumps(evidence, indent=2))
        # Signals never restart a healthy model. A new rollout replaces the whole
        # owner explicitly after draining sessions via the existing platform.
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
        for attempt in range(retries + 1):
            children = []
            status = {"attempt": attempt, "phase": "starting"}
            try:
                worker_epoch = uuid.uuid4().hex
                (state / "worker-epoch").write_text(worker_epoch)
                for role, item in plan.items():
                    env = dict(os.environ, **bundle["environments"]["model"])
                    # An in-place restart may inherit the old launcher's epoch.
                    # Fence it explicitly in every role, not only in heartbeat.
                    env["WORKER_EPOCH"] = worker_epoch
                    env["WORKER_EPOCH_FILE"] = str(state / "worker-epoch")
                    env["WHITEBOX_CHECKPOINT_SHA256"] = config["checkpoint_sha256"]
                    env["WHITEBOX_MODEL_CONFIG_SHA256"] = evidence[
                        "asset_config_sha256"
                    ]["model/zing_config.json"]
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
                    from sglang.multimodal_gen.runtime.utils.bounded_process_log import (
                        BoundedProcessLog,
                    )

                    log = BoundedProcessLog(state / f"{role}.log")
                    try:
                        process = subprocess.Popen(
                            item["argv"],
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
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
                        wait_http(
                            process,
                            "http://127.0.0.1:31002/v1/conditioning/profile",
                            {"Authorization": "Bearer " + token},
                        )
                    elif role == "dit":
                        wait_http(process, "http://127.0.0.1:30000/health", {})
                status["phase"] = "ready"
                (state / "status.json").write_text(json.dumps(status))
                while all(p.poll() is None for p, _ in children):
                    time.sleep(1)
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
                stop_children(children)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--run", action="store_true")
    p.add_argument(
        "--gpu-uuids", help="Four allocated UUIDs, in the reviewed role order"
    )
    p.add_argument(
        "--retries",
        type=int,
        choices=(0,),
        default=0,
        help="Automatic whole-group recovery is disabled",
    )
    args = p.parse_args()
    supplied = json.loads(args.bundle.read_text())
    bundle = render(supplied["policy"], supplied["bindings"])
    if fingerprint(supplied) != fingerprint(bundle):
        raise ValueError("Bundle changed after compilation")
    if not args.run:
        print(json.dumps(commands(bundle), indent=2))
    else:
        if not args.gpu_uuids:
            p.error("--run requires explicit --gpu-uuids")
        run(bundle, args.gpu_uuids.split(","), retries=args.retries)


if __name__ == "__main__":
    main()
