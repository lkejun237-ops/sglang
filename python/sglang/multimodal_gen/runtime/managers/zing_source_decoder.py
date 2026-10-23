# SPDX-License-Identifier: Apache-2.0
"""Single-controller lifecycle for a dedicated Zing source decoder process.

The deployment names the device list for the decoder; this module derives the
decoder's port, starts the worker before the DiT ranks and stops it on exit.
It stays importable without the GPU worker stack so the launch contract can be
tested on CPU-only hosts.
"""

import atexit
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time

from sglang.multimodal_gen.runtime.utils.logging_utils import logger

START_PORT = 31003


def free_port(start=START_PORT, avoid=()):
    """First port that accepts a loopback bind outside *avoid*."""
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
    raise RuntimeError("No free port for the Zing source decoder")


def resolve_role_devices(server_args, caller: str) -> tuple[str, str]:
    """Translate the user's indices into physical device IDs, from the caller's list.

    Both roles index the same CUDA_VISIBLE_DEVICES snapshot taken at launch, so
    the result never depends on the order in which the two roles are started.
    """
    visible = caller.split(",")

    def pick(indices):
        indices = list(indices)
        for index in indices:
            if index >= len(visible):
                raise ValueError(f"GPU index {index} is not visible to the launcher")
        return ",".join(visible[index] for index in indices)

    dit = pick(int(part) for part in server_args.zing_dit_gpus.split(","))
    decoder = pick(int(part) for part in server_args.zing_decoder_gpus.split(","))
    return dit, decoder


def assign_dit_visible_devices(server_args, dit_devices: str) -> None:
    """Narrow this process to the DiT devices before any CUDA context exists."""
    os.environ["CUDA_VISIBLE_DEVICES"] = dit_devices


def command(server_args, port: int) -> list[str]:
    config = getattr(server_args.pipeline_config, "decoder_config", None)
    if not isinstance(config, dict):
        raise ValueError(
            "Dedicated Zing source decoder requires a pipeline decoder_config"
        )
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={len(server_args.zing_decoder_gpus.split(','))}",
        "-m",
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "zing_h3.decoder_worker",
        "--model",
        str(server_args.model_path),
        "--port",
        str(port),
        "--decoder-config",
        json.dumps(config, sort_keys=True),
    ]


def wait_ready(process, port: int, timeout_s: float = 600) -> None:
    import zmq

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Zing source decoder exited before readiness")
        # A REQ socket cannot send again after a timed-out receive, so probe
        # each attempt on a fresh socket.
        with zmq.Context() as context:
            socket_ = context.socket(zmq.REQ)
            socket_.setsockopt(zmq.LINGER, 0)
            socket_.setsockopt(zmq.RCVTIMEO, 1000)
            socket_.setsockopt(zmq.SNDTIMEO, 1000)
            socket_.connect(f"tcp://127.0.0.1:{port}")
            try:
                socket_.send_json(dict(op="ping"))
                if socket_.recv_json().get("ok"):
                    return
            except zmq.Again:
                pass
            finally:
                socket_.close()
        time.sleep(0.5)
    raise TimeoutError("Zing source decoder readiness deadline exceeded")


def _kill_tree(pid: int) -> None:
    from sglang.multimodal_gen.runtime.utils.common import kill_process_tree

    kill_process_tree(pid, include_parent=True)


def stop(process) -> None:
    if process is None or process.poll() is not None:
        return
    _kill_tree(process.pid)


def _die_with_parent() -> None:
    """Ask the kernel to kill the decoder when the server process dies.

    atexit covers a clean interpreter exit; a SIGTERM or SIGKILL to the server
    never reaches it, so the decoder would keep holding its GPUs.
    """
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(
        1, signal.SIGKILL
    )  # PR_SET_PDEATHSIG


def _stop_on_signal(process) -> None:
    """Stop the decoder on SIGTERM/SIGINT, then finish the interrupted signal."""
    for name in ("SIGTERM", "SIGINT"):
        signum = getattr(signal, name)
        if signal.getsignal(signum) != signal.SIG_DFL:
            # Another shutdown owner (e.g. the HTTP server) already exits cleanly.
            continue

        def handler(received, _frame, signum=signum):
            stop(process)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        signal.signal(signum, handler)


def start(server_args, decoder_devices: str):
    """Start the decoder on its physical devices, publish its port, abort if unready."""
    if not server_args.zing_decoder_gpus:
        return None
    port = free_port(avoid=(server_args.port,))
    launch = command(server_args, port)
    environment = dict(
        os.environ, CUDA_VISIBLE_DEVICES=decoder_devices, PYTHONUNBUFFERED="1"
    )
    parent_dies = _die_with_parent if sys.platform.startswith("linux") else None
    process = subprocess.Popen(launch, env=environment, preexec_fn=parent_dies)
    atexit.register(stop, process)
    _stop_on_signal(process)
    try:
        wait_ready(process, port)
    except BaseException:
        stop(process)
        raise
    server_args.zing_source_decoder_port = port
    logger.info(
        "Zing source decoder ready: %d GPU(s) %s on port %d",
        len(decoder_devices.split(",")),
        decoder_devices,
        port,
    )
    return process
