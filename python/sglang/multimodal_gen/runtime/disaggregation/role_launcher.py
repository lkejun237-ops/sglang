# SPDX-License-Identifier: Apache-2.0
"""Unified process lifecycle launcher for disaggregated diffusion roles."""

from __future__ import annotations

import atexit
import copy
import ctypes
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Any

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.utils.common import kill_process_tree

logger = logging.getLogger(__name__)

START_PORT = 31003


def free_port(start: int = START_PORT, avoid: tuple[int, ...] = ()) -> int:
    """Find the first available port for loopback binding outside `avoid`."""
    for port in range(max(1024, start), 65536):
        if port in avoid:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("No free port available for disaggregated role worker")


def resolve_role_devices(role_or_args, caller: str | None = None) -> Any:
    """Translate role GPU indices into physical device IDs from caller's snapshot.

    All roles index the same caller CUDA_VISIBLE_DEVICES snapshot taken at launch,
    eliminating startup order dependencies.
    """
    visible = [part.strip() for part in (caller or "").split(",") if part.strip()]

    def pick(gpus_str: str | None) -> str | None:
        if not gpus_str:
            return None
        parsed_indices: list[int] = []
        for p in gpus_str.split(","):
            token = p.strip()
            if not token:
                continue
            try:
                idx = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"GPU index {token!r} is out of range or not an integer"
                ) from exc
            if idx < 0:
                raise ValueError(f"GPU index {idx} is out of range")
            if visible and idx >= len(visible):
                raise ValueError(
                    f"GPU index {idx} is not visible to the launcher (out of range)"
                )
            parsed_indices.append(idx)
        if visible:
            return ",".join(visible[idx] for idx in parsed_indices)
        return ",".join(str(idx) for idx in parsed_indices)

    if isinstance(role_or_args, str):
        return pick(role_or_args)

    encoder_devices = pick(getattr(role_or_args, "encoder_gpus", None))
    denoiser_devices = pick(getattr(role_or_args, "denoiser_gpus", None))
    decoder_devices = pick(getattr(role_or_args, "decoder_gpus", None))
    return encoder_devices, denoiser_devices, decoder_devices


def assign_denoiser_visible_devices(server_args, denoiser_devices: str | None) -> None:
    """Narrow the main process to the denoiser devices before any CUDA context exists."""
    if denoiser_devices is not None:
        try:
            import torch

            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "assign_denoiser_visible_devices must be called before CUDA runtime initialization in caller process"
                )
        except ImportError:
            pass
        os.environ["CUDA_VISIBLE_DEVICES"] = denoiser_devices


def build_role_command(
    server_args,
    role: RoleType,
    port: int,
    *,
    gpus: str | None = None,
    max_sessions: int | None = None,
) -> list[str]:
    """Construct command to start a role worker via torch.distributed.run."""
    eff_max_sessions = (
        max_sessions
        if max_sessions is not None
        else getattr(server_args, "disagg_max_sessions", None)
    )
    if role == RoleType.ENCODER:
        encoder_gpus = (
            gpus if gpus is not None else getattr(server_args, "encoder_gpus", None)
        )
        nproc = (
            len([x for x in encoder_gpus.split(",") if x.strip()])
            if encoder_gpus
            else 1
        )
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={nproc}",
            "-m",
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
            "zing_h3.encoder_worker",
            "--model",
            str(server_args.model_path),
            "--port",
            str(port),
        ]
        if eff_max_sessions is not None and int(eff_max_sessions) > 1:
            cmd.extend(["--max-sessions", str(int(eff_max_sessions))])
        return cmd
    elif role == RoleType.DECODER:
        config = getattr(
            getattr(server_args, "pipeline_config", None), "decoder_config", None
        )
        if not isinstance(config, dict):
            raise ValueError(
                "Dedicated Zing source decoder requires a pipeline decoder_config"
            )
        decoder_gpus = (
            gpus if gpus is not None else getattr(server_args, "decoder_gpus", None)
        )
        nproc = (
            len([x for x in decoder_gpus.split(",") if x.strip()])
            if decoder_gpus
            else 1
        )
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={nproc}",
            "-m",
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
            "zing_h3.decoder_worker",
            "--static-connection",
            "--model",
            str(server_args.model_path),
            "--port",
            str(port),
            "--decoder-config",
            json.dumps(config, sort_keys=True),
        ]
        if eff_max_sessions is not None and int(eff_max_sessions) > 1:
            cmd.extend(["--max-sessions", str(int(eff_max_sessions))])
        return cmd
    else:
        raise ValueError(f"Unsupported role for build_role_command: {role}")


def wait_ready(process, port: int, timeout_s: float = 600) -> None:
    """Poll the role worker's control socket until ready or failed."""
    import zmq

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Worker process exited before readiness")
        with zmq.Context() as context:
            socket_ = context.socket(zmq.REQ)
            socket_.setsockopt(zmq.LINGER, 0)
            socket_.setsockopt(zmq.RCVTIMEO, 1000)
            socket_.setsockopt(zmq.SNDTIMEO, 1000)
            socket_.connect(f"tcp://127.0.0.1:{port}")
            try:
                socket_.send_json({"op": "ping"})
                if socket_.recv_json().get("ok"):
                    return
            except zmq.Again:
                pass
            finally:
                socket_.close()
        time.sleep(0.5)
    raise TimeoutError(f"Worker on port {port} readiness deadline exceeded")


def _kill_tree(pid: int) -> None:
    kill_process_tree(pid, include_parent=True)


def stop(process) -> None:
    """Terminate a process and its child tree."""
    if process is None or not hasattr(process, "poll") or process.poll() is not None:
        return
    _kill_tree(process.pid)


def stop_role_workers(server_args, processes: list | None = None) -> None:
    """Stop all active role worker processes and reset port state on server_args."""
    for proc in processes or []:
        stop(proc)
    if server_args is not None:
        server_args.disagg_encoder_port = None
        server_args.disagg_decoder_port = None
        server_args.disagg_encoder_ports = None
        server_args.disagg_decoder_ports = None


def _die_with_parent() -> None:
    """Ask the OS to send SIGKILL to this process when its parent terminates."""
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(
        1, signal.SIGKILL
    )  # PR_SET_PDEATHSIG


def _stop_on_signal(processes: list) -> None:
    """Ensure graceful teardown of role workers on SIGTERM/SIGINT."""
    for name in ("SIGTERM", "SIGINT"):
        signum = getattr(signal, name)
        if signal.getsignal(signum) != signal.SIG_DFL:
            continue

        def handler(received, _frame, signum=signum):
            for p in processes:
                stop(p)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        signal.signal(signum, handler)


def start_role_workers(
    server_args,
    *,
    encoder_devices: str | None = None,
    decoder_devices: str | None = None,
    encoder_device_groups: list[str] | None = None,
    decoder_device_groups: list[str] | None = None,
    max_sessions: int | None = None,
) -> list:
    """Start required role worker subprocesses, probe readiness, and roll back on error."""
    if encoder_device_groups is not None:
        enc_groups = [g for g in encoder_device_groups if g]
    elif getattr(server_args, "encoder_gpus", None) and encoder_devices:
        enc_groups = [encoder_devices]
    else:
        enc_groups = []

    if decoder_device_groups is not None:
        dec_groups = [g for g in decoder_device_groups if g]
    elif getattr(server_args, "decoder_gpus", None) and decoder_devices:
        dec_groups = [decoder_devices]
    else:
        dec_groups = []

    if not enc_groups and not dec_groups:
        return []

    avoid = [
        p
        for attr in ("port", "scheduler_port", "master_port")
        if isinstance(p := getattr(server_args, attr, None), int) and p > 0
    ]
    parent_dies = _die_with_parent if sys.platform.startswith("linux") else None

    enc_ports: list[int] = []
    dec_ports: list[int] = []
    processes = []
    _stop_on_signal(processes)
    started_roles = []

    try:
        for dev_group in enc_groups:
            enc_port = free_port(avoid=tuple(avoid))
            avoid.append(enc_port)
            enc_ports.append(enc_port)
            if encoder_device_groups is None and max_sessions is None:
                enc_cmd = build_role_command(server_args, RoleType.ENCODER, enc_port)
            else:
                enc_cmd = build_role_command(
                    server_args,
                    RoleType.ENCODER,
                    enc_port,
                    gpus=dev_group,
                    max_sessions=max_sessions,
                )
            enc_env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=dev_group,
                PYTHONUNBUFFERED="1",
            )
            enc_proc = subprocess.Popen(enc_cmd, env=enc_env, preexec_fn=parent_dies)
            processes.append(enc_proc)
            atexit.register(stop, enc_proc)
            started_roles.append((enc_proc, enc_port, "encoder"))

        for dev_group in dec_groups:
            dec_port = free_port(avoid=tuple(avoid))
            avoid.append(dec_port)
            dec_ports.append(dec_port)
            if decoder_device_groups is None and max_sessions is None:
                dec_cmd = build_role_command(server_args, RoleType.DECODER, dec_port)
            else:
                dec_cmd = build_role_command(
                    server_args,
                    RoleType.DECODER,
                    dec_port,
                    gpus=dev_group,
                    max_sessions=max_sessions,
                )
            dec_env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=dev_group,
                PYTHONUNBUFFERED="1",
            )
            dec_proc = subprocess.Popen(dec_cmd, env=dec_env, preexec_fn=parent_dies)
            processes.append(dec_proc)
            atexit.register(stop, dec_proc)
            started_roles.append((dec_proc, dec_port, "decoder"))

        for proc, port, role_name in started_roles:
            wait_ready(proc, port)

    except BaseException:
        stop_role_workers(server_args, processes)
        raise

    if enc_ports:
        server_args.disagg_encoder_port = enc_ports[0]
        server_args.disagg_encoder_ports = list(enc_ports)
    if dec_ports:
        server_args.disagg_decoder_port = dec_ports[0]
        server_args.disagg_decoder_ports = list(dec_ports)

    return processes


def _with_disagg_role_policies(
    profile: dict,
    *,
    encoder_port: int | None = None,
    decoder_port: int | None = None,
    encoder_ports: list[int] | None = None,
    decoder_ports: list[int] | None = None,
) -> dict:
    """Inject disaggregated encoder and decoder ports into runtime profile."""
    if (
        encoder_port is None
        and decoder_port is None
        and not encoder_ports
        and not decoder_ports
    ):
        return profile
    result = copy.deepcopy(profile)
    if encoder_port is not None or encoder_ports:
        enc_cfg = result.setdefault("source_encoder", {})
        enc_cfg.setdefault("mode", "progressive")
        enc_cfg.setdefault("contract_version", "h3-clip17-progressive-v1")
        enc_cfg.setdefault(
            "factory",
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.zing_h3.progressive_encoder:create_session",
        )
        enc_remote = enc_cfg.setdefault("remote", {})
        enc_remote["port"] = (
            encoder_port if encoder_port is not None else encoder_ports[0]
        )
        if encoder_ports:
            enc_remote["ports"] = list(encoder_ports)
        enc_remote.setdefault("mode", "overlap")
    if decoder_port is not None or decoder_ports:
        dec_remote = result.setdefault("source_decoder", {}).setdefault("remote", {})
        dec_remote["port"] = (
            decoder_port if decoder_port is not None else decoder_ports[0]
        )
        if decoder_ports:
            dec_remote["ports"] = list(decoder_ports)
    return result
