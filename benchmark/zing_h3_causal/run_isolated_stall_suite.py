# SPDX-License-Identifier: Apache-2.0
"""Owned-instance serving-stall-v1: capture, cold, full instance, recovery, DiT.

Only use a fresh task root and scheduler-allocated devices. All child process
handles belong to this runner; no external service is reserved or stopped.
"""

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

from archive_stall_artifacts import archive_to_directory
from owned_process import owned_popen, verify_owned_cleanup


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def role_layout(config, visible_count):
    sp = config["sp"]
    roles = config.get(
        "role_devices",
        {"dit": list(range(sp)), "encoder": [sp], "decoder": [sp + 1]},
    )
    if config.get("dit_only") is True:
        if (
            sp != 4
            or visible_count != 4
            or roles != dict(dit=[0, 1, 2, 3], encoder=[], decoder=[])
        ):
            raise ValueError(
                "DiT-only collection requires exactly four visible model GPUs"
            )
        return roles
    if set(roles) != {"dit", "encoder", "decoder"}:
        raise ValueError("All three model roles require explicit device assignments")
    indices = [i for values in roles.values() for i in values]
    if (
        len(roles["dit"]) != sp
        or len(set(roles["dit"])) != sp
        or len(roles["encoder"]) != 1
        or not roles["decoder"]
        or len(set(roles["decoder"])) != len(roles["decoder"])
        or any(type(i) is not int or not 0 <= i < visible_count for i in indices)
        or set(indices) != set(range(visible_count))
    ):
        raise ValueError("Role assignments differ from the allocated GPU devices")
    if (
        len(indices) != visible_count
        and config.get("allow_shared_role_devices") is not True
    ):
        raise ValueError("Shared model devices require an explicit test configuration")
    if roles["dit"][0] in roles["encoder"] + roles["decoder"]:
        raise ValueError(
            "NCCL model peers must use a different GPU from the DiT leader"
        )
    if len(roles["decoder"]) > 1 and shared_role_devices(roles):
        raise ValueError("Parallel decoder testing requires dedicated role devices")
    return roles


def shared_role_devices(roles):
    indices = [i for values in roles.values() for i in values]
    return len(set(indices)) != len(indices)


def numeric_cuda_devices(original, count):
    values = (
        [str(i) for i in range(count)]
        if original is None
        else [value.strip() for value in original.split(",")]
    )
    if (
        len(values) != count
        or any(not value.isascii() or not value.isdecimal() for value in values)
        or len({int(value) for value in values}) != count
    ):
        raise ValueError(
            "The allocated CUDA ordinal mapping must be numeric and unique"
        )
    return [str(int(value)) for value in values]


def stop(process):
    import psutil

    # A reaped PID may have been reused. Never resolve it into a new owner.
    if process.poll() is not None:
        verify_owned_cleanup(process)
        return

    def wait_children(children):
        # Some container kernels reject pidfd_open used by psutil.wait_procs.
        # Poll the same Process identities instead; zombies no longer own GPUs.
        deadline = time.monotonic() + 10
        while True:
            remaining = []
            for child in children:
                with contextlib.suppress(psutil.NoSuchProcess):
                    if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                        remaining.append(child)
            if not remaining or time.monotonic() >= deadline:
                return remaining
            time.sleep(0.05)

    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
    # torchrun may create separate worker sessions. psutil retains PID identity
    # and refuses signals to a process that reused the original worker PID.
    for child in descendants:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.terminate()
    remaining = wait_children(descendants)
    for child in remaining:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()
    remaining = wait_children(remaining)
    if remaining:
        raise RuntimeError("Owned role descendants remain after cleanup")
    verify_owned_cleanup(process)


class Runner:
    def stage_timing_directory(self):
        identity = hashlib.sha256(str(self.out).encode()).hexdigest()[:16]
        return (
            Path(self.c.get("stage_timing_tmpfs", "/dev/shm/zing-stall-timing"))
            / identity
        )

    def wait_stage_timing(self):
        deadline = time.monotonic() + 30
        while True:
            busy = []
            for path in Runner.stage_timing_directory(self).rglob("*-health.json"):
                try:
                    row = json.loads(path.read_text())
                    if any(
                        row.get(key)
                        for key in ["pending", "queued", "error", "dropped"]
                    ):
                        busy.append(str(path))
                except (OSError, ValueError):
                    busy.append(str(path))
            if not busy:
                return None
            if time.monotonic() >= deadline:
                return busy
            time.sleep(0.05)

    def preserve_stage_timing(self, *, stopped=True):
        source = Runner.stage_timing_directory(self)
        if source.exists():
            shutil.copytree(source, self.out / "timing", dirs_exist_ok=True)
            write(
                self.out / "TIMING_RETENTION.json",
                dict(
                    source=str(source),
                    target=str(self.out / "timing"),
                    bytes=sum(
                        p.stat().st_size for p in source.rglob("*") if p.is_file()
                    ),
                    source_removed=False,
                    snapshot_after_owned_roles_stopped=stopped,
                    scope="Only the finite isolated test uses local RAM timing; raw files retained before pod exit",
                ),
            )

    def __init__(self, config):
        self.c = config
        self.root = Path(config["root"])
        self.out = self.root / "results"
        self.out.mkdir(parents=True, exist_ok=False)
        if config.get("isolated_instance") is not True:
            raise ValueError("An explicitly isolated test allocation is required")
        self.bench = Path(__file__).resolve().parent
        self.children = []
        self.instance = []
        self.model = Path(config["model"])
        self.sp = config["sp"]
        self.ports = config["ports"]
        self.switch = Path(config["diagnostic_switch"])
        self.switch.unlink(missing_ok=True)
        import torch

        self.torch = torch
        count = torch.cuda.device_count()
        self.roles = role_layout(config, count)
        self.cuda_devices = numeric_cuda_devices(
            os.environ.get("CUDA_VISIBLE_DEVICES"), count
        )
        self.hardware = []
        self.uuids = []
        for i in range(count):
            p = torch.cuda.get_device_properties(i)
            uuid = str(p.uuid)
            uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
            row = dict(
                index=i,
                uuid=uuid,
                name=p.name,
                cc=list(torch.cuda.get_device_capability(i)),
                bytes=p.total_memory,
            )
            if row["cc"] != config["expected_cc"]:
                raise ValueError(
                    "Allocated GPU capability differs from frozen CPU receipt"
                )
            self.hardware.append(row)
            self.uuids.append(uuid)
        before = Path(config["before_import_gpu_apps"]).read_text()
        if any(uuid in before for uuid in self.uuids):
            raise ValueError(
                "An allocated GPU was occupied before this runner imported CUDA"
            )
        inventory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
        )
        nvml_devices = dict(
            tuple(part.strip() for part in line.split(","))
            for line in inventory.splitlines()
            if line
        )
        if [nvml_devices.get(i) for i in self.cuda_devices] != self.uuids:
            raise ValueError(
                "CUDA and NVML ordinals differ from the allocated GPU UUIDs"
            )
        topology = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True
        )
        (self.out / "HARDWARE_TOPOLOGY.txt").write_text(
            topology.stdout + topology.stderr
        )
        for port in self.ports.values():
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
        write(
            self.out / "HARDWARE.json",
            dict(
                devices=self.hardware,
                cuda_device_ordinals=self.cuda_devices,
                topology_returncode=topology.returncode,
                role_devices=self.roles,
                shared_role_devices=shared_role_devices(self.roles),
                torch=torch.__version__,
                cuda=torch.version.cuda,
            ),
        )
        write(self.out / "CONFIG.json", config)
        self.memory_monitor = self.start(
            "gpu-telemetry",
            [
                "nvidia-smi",
                "--id=" + ",".join(self.uuids),
                "--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw",
                "--format=csv",
                "--loop=1",
            ],
            [],
            group=False,
        )

    def phase(self, name):
        write(self.out / "STATUS.json", dict(phase=name, time=time.time()))
        print("PHASE", name, flush=True)

    def environment(self, devices, label):
        c = self.c
        e = dict(os.environ)
        e.update(
            CUDA_VISIBLE_DEVICES=",".join(self.cuda_devices[i] for i in devices),
            PYTHONPATH=":".join(
                [c["deps"], str(self.bench), str(Path(c["source"]) / "python")]
            ),
            SGLANG_LIGHTWEIGHT_RUNTIME="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            TOKENIZERS_PARALLELISM="false",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            SGLANG_ZING_EXPERIMENTAL_ATTENTION="fa4",
            SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES="1",
            SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN="1",
            SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM="1",
            SGLANG_ZING_VISUAL_ENCODE_MODE="performance",
            MINIMAX_H3_PROGRESSIVE_ENCODER_PERFORMANCE_AUTO="1",
            NCCL_GRAPH_REGISTER="0",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            STALL_MEMORY_DIR=str(self.out / "memory"),
            STALL_DIAGNOSTIC_SWITCH=str(self.switch),
            STALL_ENCODER_PORT=str(self.ports["encoder"]),
            STALL_DECODER_PORT=str(self.ports["decoder"]),
            SGLANG_ZING_STAGE_TIMING_DIR=str(
                Runner.stage_timing_directory(self) / label
            ),
            STALL_RETAIN_STAGE_TIMING="1",
        )
        # Only the explicit shared fallback needs to return inactive cached
        # reservations to the other processes on its card between sessions.
        e["STALL_SHARED_ALLOCATOR_TRIM"] = (
            "1" if shared_role_devices(self.roles) else "0"
        )
        for key in [
            "FP8",
            "TEXT_FA4",
            "VISION_FA4",
            "HF_RMSNORM",
            "FUSED_MROPE",
            "FUSED_GATE_UP",
        ]:
            e["SGLANG_H3_QWEN_" + key] = "0"
        for key in [
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ]:
            e.pop(key, None)
        return e

    def start(self, label, cmd, devices, extra=None, *, group=True):
        env = self.environment(devices, label)
        env.update(extra or {})
        log = (self.out / (label + ".log")).open("w")
        p = owned_popen(
            cmd,
            env=env,
            cwd=self.c["source"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.children.append((p, log))
        write(self.out / (label + "-command.json"), cmd)
        if group:
            self.instance.append(p)
        return p

    def wait_http(self, process, url, token=None):
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Owned service exited before readiness: " + url)
            try:
                q = urllib.request.Request(
                    url, headers={"Authorization": "Bearer " + token} if token else {}
                )
                with urllib.request.urlopen(q, timeout=3) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
            time.sleep(2)
        raise TimeoutError("Owned service readiness timeout")

    def launch_instance(self, label, *, capture=False):
        self.phase(label + "-loading")
        self.instance = []
        from sglang.multimodal_gen.runtime.conditioning.resources import ResourceLimits

        token = secrets.token_urlsafe(32)
        service = dict(
            tokens={token: dict(tenant="stall-regression", admin=True)},
            native=dict(model_path=str(self.model), gpu_id=0),
            result_transport="nccl",
            limits=dataclasses.asdict(ResourceLimits(request_timeout_s=600)),
        )
        client = dict(
            bearer_token=token,
            result_transport="nccl",
            conditioning_timeout_s=600,
            retry_initial_s=0.05,
            retry_max_s=0.5,
            limits=dict(
                max_request_bytes=512 << 20,
                max_response_bytes=256 << 20,
                max_condition_rows=16384,
            ),
        )
        for name, data in [
            ("service.secret.json", service),
            ("worker.secret.json", client),
        ]:
            write(self.root / name, data)
            (self.root / name).chmod(0o600)
        entry = str(self.bench / "isolated_stall_worker.py")
        p = self.ports
        enc = self.start(
            label + "-encoder",
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--nproc-per-node=1",
                "--master-addr=127.0.0.1",
                "--master-port=" + str(p["encoder_master"]),
                entry,
                "encoder",
                "--model",
                str(self.model),
                "--port",
                str(p["encoder"]),
                "--timeout-seconds",
                "1800",
            ],
            self.roles["encoder"],
        )
        from run_encoder_overlap_matrix import server_command, wait_encoder

        wait_encoder(enc, p["encoder"])
        qwen = self.start(
            label + "-qwen",
            [
                sys.executable,
                entry,
                "qwen",
                "serve",
                "--config",
                str(self.root / "service.secret.json"),
                "--profile",
                self.c["conditioning_profile"],
                "--host",
                "127.0.0.1",
                "--port",
                str(p["qwen"]),
            ],
            self.roles["encoder"],
        )
        self.wait_http(
            qwen, f"http://127.0.0.1:{p['qwen']}/v1/conditioning/profile", token
        )
        from types import SimpleNamespace

        from sglang.multimodal_gen.runtime.disaggregation.role_launcher import (
            build_role_command,
            wait_ready,
        )
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        spec = json.loads(Path(self.c["pipeline"]).read_text())["decoder_config"]
        command = build_role_command(
            SimpleNamespace(
                model_path=str(self.model),
                pipeline_config=SimpleNamespace(decoder_config=spec),
                decoder_gpus=",".join(map(str, range(len(self.roles["decoder"])))),
            ),
            RoleType.DECODER,
            p["decoder"],
        )
        module = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker"
        standalone = command.index("--standalone")
        command[standalone : standalone + 1] = [
            "--nnodes=1",
            "--master-addr=127.0.0.1",
            "--master-port=" + str(p["decoder_master"]),
        ]
        idx = command.index(module)
        command = command[: idx - 1] + [entry, "decoder"] + command[idx + 1 :]
        dec = self.start(label + "-decoder", command, self.roles["decoder"])
        wait_ready(dec, p["decoder"], timeout_s=1800)
        command = server_command(self.model, Path(self.c["pipeline"]), self.sp)
        command[command.index("--port") + 1] = str(p["http"])
        command = [
            sys.executable,
            entry,
            "server",
            *command[3:],
            "--scheduler-port",
            str(p["scheduler"]),
            "--master-port",
            str(p["master"]),
            "--dist-timeout",
            "600",
            "--realtime-session-lease-ttl-s",
            "1200",
            "--realtime-session-idle-timeout-s",
            "1200",
            "--realtime-session-max-lifetime-s",
            "7200",
            "--zing-solver",
            "few_step",
            "--zing-denoising-step-list",
            "1000,500",
            "--zing-initial-block-count",
            "3",
            "--zing-initial-denoising-step-list",
            "1000,750,500,250",
            "--zing-reference-max-images",
            "21",
            "--zing-conditioning-mode",
            "remote",
            "--conditioning-service-url",
            f"http://127.0.0.1:{p['qwen']}",
            "--conditioning-expected-profile",
            self.c["conditioning_profile"],
            "--conditioning-config",
            str(self.root / "worker.secret.json"),
            "--zing-block0-cache-mode",
            "off",
            "--mem-fraction-static",
            str(self.c.get("mem_fraction_static", 0.8)),
            # Reuse only the owned role bridges across serial sessions. Closing
            # per-session channels synchronously can delay the final RGB batch.
            "--disagg-encoder-ports",
            str(p["encoder"]),
            "--disagg-decoder-ports",
            str(p["decoder"]),
        ]
        extra = (
            dict(
                SGLANG_ZING_COMPONENT_CAPTURE_DIR=str(self.root / "captures"),
                SGLANG_ZING_COMPONENT_CAPTURE_SCOPE="dit",
            )
            if capture
            else {}
        )
        server = self.start(label + "-dit", command, self.roles["dit"], extra)
        self.wait_http(server, f"http://127.0.0.1:{p['http']}/health")
        write(
            self.out / (label + "-ready.json"),
            dict(time=time.time(), monotonic_ns=time.monotonic_ns(), capture=capture),
        )

    def shutdown_instance(self):
        busy = self.wait_stage_timing()
        errors = []
        for process in reversed(self.instance):
            try:
                stop(process)
            except Exception as error:
                errors.append(str(error))
        self.instance = []
        self.preserve_stage_timing(stopped=not errors)
        if errors or busy:
            raise RuntimeError(
                f"Owned diagnostic cleanup incomplete: stop={errors}, timing={busy}"
            )

    def client(
        self,
        label,
        *,
        mode=None,
        width=None,
        height=None,
        warmup=False,
        diagnostic=False,
        case_set=None,
    ):
        self.phase(label)
        output = self.out / label
        common = [
            "--media-root",
            self.c["media"],
            "--output-dir",
            str(output),
            "--url",
            f"ws://127.0.0.1:{self.ports['http']}/v1/realtime_video/generate",
        ]
        if mode is not None:
            cmd = [
                sys.executable,
                str(self.bench / "run_stall_regression.py"),
                *common,
                "--frozen-inputs",
                self.c["frozen_inputs"],
                "--archive-dir",
                self.c["archive_dir"],
                "--mode",
                mode,
            ]
        else:
            selector = (
                Path(case_set)
                if case_set is not None
                else self.root / "few-images-case.json"
            )
            if case_set is None and not selector.exists():
                from case_sets import load_case_set

                collection = load_case_set()
                write(
                    selector,
                    dict(
                        name=collection["definition"]["name"],
                        cases=[collection["definition"]["cases"][0]],
                    ),
                )
            cmd = [
                sys.executable,
                str(self.bench / "bench_streaming_source.py"),
                "--media-mode",
                "raw" if diagnostic else "metrics",
                *common,
                "--case-set",
                str(selector),
                "--source-rgb-manifest",
                self.c["frozen_inputs"],
                "--width",
                str(width),
                "--height",
                str(height),
                "--source-width",
                str(width),
                "--source-height",
                str(height),
                "--warmup-runs",
                "0",
                "--measured-runs",
                "1",
                "--seed",
                "123",
                "--timeout",
                "3600",
                "--batch-pattern",
                "1",
            ]
            if not diagnostic:
                cmd.append("--no-stage-timing")
            else:
                cmd += [
                    "--retention-budget-bytes",
                    str(self.c.get("raw_retention_budget_bytes", 2 << 30)),
                ]
            if warmup:
                raise ValueError(
                    "Cold companion requests retain their explicit outer warm identity"
                )
        if mode is None:
            write(
                self.out / (label + "-active-request.json"),
                dict(
                    phase=label,
                    index=0,
                    output=str(output),
                    case_set=str(selector),
                ),
            )
        with (self.out / (label + ".log")).open("w") as log:
            subprocess.run(
                cmd,
                env=self.environment([], label),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=14 * 3600,
            )
        if mode is None:
            if diagnostic:
                self.switch.unlink(missing_ok=True)
            requests = [
                json.loads(line)
                for line in (output / "requests.jsonl").read_text().splitlines()
                if line
            ]
            if not requests or any(
                not r.get("success") or not r.get("delivery_complete") for r in requests
            ):
                raise RuntimeError("Cold/warm request failed")
            for raw in output.rglob("*.rgb"):
                archive_to_directory(raw, self.c["archive_dir"], remove_verified=True)
        return output

    def cold_component_diagnostic(self, width, height):
        label = f"cold-components-{width}x{height}"
        self.switch.touch()
        try:
            self.launch_instance(label)
            self.client(label, width=width, height=height, diagnostic=True)
        finally:
            self.shutdown_instance()
            self.switch.unlink(missing_ok=True)

    def check_capture_gc(self):
        self.phase("cuda-graph-gc")
        process = self.start(
            "cuda-graph-gc",
            [
                sys.executable,
                str(self.bench / "check_cuda_graph_gc.py"),
                "--output",
                str(self.out / "CUDA_GRAPH_GC.json"),
            ],
            self.roles["dit"][:1],
            group=False,
        )
        if process.wait(timeout=900):
            raise RuntimeError("CUDA graph GC safety regression failed")
        process = self.start(
            "cuda-graph-stream",
            [
                sys.executable,
                str(self.bench / "check_cuda_graph_stream.py"),
                "--output",
                str(self.out / "CUDA_GRAPH_STREAM.json"),
            ],
            self.roles["dit"][:1],
            group=False,
        )
        if process.wait(timeout=180):
            raise RuntimeError("CUDA graph stream isolation regression failed")

    def run(self):
        # First inference must precede capture: a separate capture process still
        # primes disk compiler caches shared by subsequent process restarts.
        for width, height in [(832, 480), (1280, 736)]:
            for restart in range(3):
                label = f"S3-{width}x{height}-restart{restart}"
                self.launch_instance(label)
                self.client(label + "-cold", width=width, height=height)
                for repeat in range(3):
                    self.client(label + f"-warm{repeat}", width=width, height=height)
                self.shutdown_instance()
        self.check_capture_gc()
        # Separate fresh processes provide uncached Qwen/reference VAE evidence;
        # warmed formal requests may legitimately skip both components.
        for width, height in [(832, 480), (1280, 736)]:
            self.cold_component_diagnostic(width, height)
        self.launch_instance("capture", capture=True)
        self.client("capture-inputs", mode="capture")
        self.shutdown_instance()
        self.launch_instance("instance")
        self.client("instance-matrix", mode="full")
        self.switch.touch()
        try:
            self.client("diagnostic", mode="diagnostic")
        finally:
            self.switch.unlink(missing_ok=True)
        self.run_lifecycle()
        self.shutdown_instance()
        self.replay_dit()
        self.phase("complete")

    def run_lifecycle(self):
        self.phase("lifecycle")
        with (self.out / "lifecycle.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(self.bench / "run_stall_lifecycle.py"),
                    "--url",
                    f"ws://127.0.0.1:{self.ports['http']}/v1/realtime_video/generate",
                    "--media-root",
                    self.c["media"],
                    "--frozen-inputs",
                    self.c["frozen_inputs"],
                    "--output-dir",
                    str(self.out / "lifecycle"),
                ],
                env=self.environment([], "lifecycle"),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=2400,
            )
        for raw in (self.out / "lifecycle").rglob("*.rgb"):
            archive_to_directory(raw, self.c["archive_dir"], remove_verified=True)

    def replay_dit(self):
        captures = sorted((self.root / "captures").iterdir())
        if len(captures) != 2:
            raise ValueError("Expected exactly two full long-session captures")
        for index, capture in enumerate(captures):
            self.phase("dit-replay")
            complete = json.loads((capture / "COMPLETE.json").read_text())
            if complete["input_frames"] != 15921 or complete["capture_scope"] != "dit":
                raise ValueError("Incomplete long-session DiT capture")
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--nproc-per-node=" + str(self.sp),
                "--master-addr=127.0.0.1",
                "--master-port=" + str(self.ports["replay_master"]),
                str(self.bench / "benchmark_dit_replay.py"),
                "--model",
                str(self.model),
                "--pipeline-config",
                self.c["pipeline"],
                "--capture",
                str(capture),
                "--output",
                str(self.out / f"D1-{index}"),
                "--measured-runs",
                "3",
                "--require-captured-kv-pool",
            ]
            process = self.start(f"D1-{index}", cmd, self.roles["dit"], group=False)
            if process.wait(timeout=14400):
                raise RuntimeError("Full DiT replay failed")
            self.phase("dit-resident-soak")
            soak_cmd = [*cmd, "--minimum-round-seconds", "600"]
            soak_cmd[soak_cmd.index("--output") + 1] = str(self.out / f"SOAK-{index}")
            process = self.start(
                f"SOAK-{index}", soak_cmd, self.roles["dit"], group=False
            )
            if process.wait(timeout=14400):
                raise RuntimeError("Resident complete-case DiT soak failed")

    def close(self):
        errors = []
        for process, log in reversed(self.children):
            try:
                stop(process)
            except Exception as error:
                errors.append(str(error))
            log.close()
        self.preserve_stage_timing(stopped=not errors)
        if errors:
            raise RuntimeError(f"Owned role cleanup failed: {errors}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    ready = json.loads(Path(config["cpu_receipt"]).read_text())
    if not ready.get("passed"):
        raise ValueError("Frozen CPU preparation is incomplete")
    for name, digest in ready["controls"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen control changed: " + name)
    # Argument parsing alone does not import the actual Lynn implementation.
    # Check it before starting any GPU role; older serving images can parse
    # --help successfully while missing the repository's pinned Diffusers API.
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        initialize_runtime_imports,
    )

    initialize_runtime_imports()
    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.loader import (  # noqa: F401
        load_lynn_decoder,
    )

    runner = None
    try:
        runner = Runner(config)
        runner.run()
    except BaseException:
        if runner is not None:
            (runner.out / "FAILURE.txt").write_text(traceback.format_exc())
            runner.phase("failed")
        raise
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
