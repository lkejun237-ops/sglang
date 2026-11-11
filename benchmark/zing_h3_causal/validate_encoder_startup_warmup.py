# SPDX-License-Identifier: Apache-2.0
"""Observe one real Encoder startup; no DiT, Decoder, communication group or replay.

The child runs the ordinary Encoder entrypoint with its default single service.
A Python call observer records only warmup boundaries and source-VAE forwards;
it never replaces model methods. Timings include observer overhead and are not
performance benchmark results. Startup requires an allocated, isolated GPU;
--preflight checks the same image and model metadata without loading weights.
"""

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import runpy
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

ENCODER = "sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server"
STAGES = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3"
)
PROGRESSIVE = (
    "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive"
)


async def _transport_loopback(options):
    import uvicorn
    import websockets

    async def echo(scope, receive, send):
        await receive()
        await send({"type": "websocket.accept"})
        message = await receive()
        await send({"type": "websocket.send", "bytes": message["bytes"]})
        await send({"type": "websocket.close"})

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        server = uvicorn.Server(
            uvicorn.Config(echo, log_level="error", lifespan="off", **options)
        )
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("WebSocket preflight server exited")
                if time.monotonic() >= deadline:
                    raise TimeoutError("WebSocket preflight server did not start")
                await asyncio.sleep(0.01)
            url = f"ws://127.0.0.1:{listener.getsockname()[1]}"
            async with websockets.connect(
                url,
                compression=None,
                max_size=options["ws_max_size"],
                max_queue=options["ws_max_queue"],
                open_timeout=5,
                close_timeout=5,
            ) as connection:
                payload = b"encoder-startup-preflight"
                await asyncio.wait_for(connection.send(payload), 5)
                if await asyncio.wait_for(connection.recv(), 5) != payload:
                    raise RuntimeError("WebSocket preflight echo did not match")
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 10)
    return {"passed": True, "payload_bytes": len(payload)}


def _preflight(args):
    """Check the real image and model metadata without entering model lifespan."""
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        revision=args.revision,
        model=str(args.model),
        mode="preflight",
        python=sys.version,
        passed=False,
        packages={},
        modules={},
        files={},
    )
    try:
        for package in (
            "torch",
            "transformers",
            "safetensors",
            "fastapi",
            "uvicorn",
            "websockets",
            "triton",
            "flashinfer-python",
            "pyzmq",
            "sglang",
        ):
            try:
                receipt["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                receipt["packages"][package] = None
        # This is the ordinary entrypoint and bootstrap, not CPU platform stubs.
        encoder = importlib.import_module(ENCODER)
        bootstrap = importlib.import_module(STAGES + ".encoder_worker")
        bootstrap.initialize_runtime_imports()
        for name in (
            ENCODER,
            STAGES + ".encoder_worker",
            STAGES + ".codecs",
            STAGES + ".codec_warmup",
            PROGRESSIVE,
            "sglang.multimodal_gen.runtime.conditioning.native",
            "sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl",
            "sglang.multimodal_gen.runtime.models.vaes.minimax_h3",
            "sglang.multimodal_gen.runtime.realtime.source_transport",
            "torch",
            "transformers",
            "uvicorn",
            "websockets",
        ):
            module = importlib.import_module(name)
            receipt["modules"][name] = str(Path(module.__file__).resolve())

        def record(path):
            info = dict(path=str(path), exists=path.is_file())
            receipt["files"][str(path.relative_to(args.model))] = info
            if not info["exists"]:
                raise FileNotFoundError(
                    f"Required Encoder component is missing: {path}"
                )
            info.update(resolved_path=str(path.resolve()), bytes=path.stat().st_size)
            if path.suffix == ".json":
                content = path.read_bytes()
                info["sha256"] = hashlib.sha256(content).hexdigest()
                return json.loads(content)

        configs = {}
        for relative in (
            "model_index.json",
            "zing_config.json",
            "transformer/config.json",
            "video_vae/config.json",
            "video_vae/source/model.safetensors",
            "text_encoder/config.json",
            "tokenizer/tokenizer_config.json",
        ):
            configs[relative] = record(args.model / relative)
        for component in ("text_encoder", "tokenizer", "processor"):
            files = sorted(
                path for path in (args.model / component).rglob("*") if path.is_file()
            )
            if not files:
                raise FileNotFoundError(f"Encoder component is empty: {component}")
            for path in files:
                record(path)
            if component == "text_encoder":
                if not any(path.suffix == ".safetensors" for path in files):
                    raise FileNotFoundError("Encoder text weights are missing")
                for path in files:
                    if path.name.endswith(".safetensors.index.json"):
                        for shard in set(
                            json.loads(path.read_text())["weight_map"].values()
                        ):
                            record(path.parent / shard)
        pipeline = None
        if args.pipeline_config_path is not None:
            content = args.pipeline_config_path.read_bytes()
            pipeline = json.loads(content)
            receipt["pipeline_config"] = dict(
                path=str(args.pipeline_config_path),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        # create_apps parses the production ServerArgs. Model loading and CUDA
        # execution are inside its lifespan, which preflight never enters.
        _, server_args = encoder.create_apps(args.model, pipeline_config=pipeline)
        from transformers import AutoConfig

        from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
            MiniMaxH3VideoVAEConfig,
        )
        from sglang.multimodal_gen.runtime.realtime.source_transport import (
            SourceTransportBudget,
        )

        source_config = MiniMaxH3VideoVAEConfig()
        source_config.update_model_arch(configs["video_vae/config.json"])
        text_config = AutoConfig.from_pretrained(
            str(args.model / "text_encoder"), local_files_only=True
        )
        receipt["parsed_configs"] = dict(
            server_args=type(server_args).__name__,
            pipeline=type(server_args.pipeline_config).__name__,
            source_vae=type(source_config).__name__,
            text_encoder=type(text_config).__name__,
        )
        budget = SourceTransportBudget.from_args(server_args)
        budget.validate_sessions(1)
        receipt["websocket_options"] = budget.websocket_options()
        receipt["loopback"] = asyncio.run(
            _transport_loopback(receipt["websocket_options"])
        )
        receipt["passed"] = True
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("ENCODER_PREFLIGHT_PASS " + str(args.output / "result.json"), flush=True)


def _worker(args):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    import torch

    device = torch.cuda.get_device_properties(0)
    (args.output / "runtime.json").write_text(
        json.dumps(
            dict(
                revision=args.revision,
                python=sys.version,
                torch_version=torch.__version__,
                torch_path=str(Path(torch.__file__).resolve()),
                cuda_version=torch.version.cuda,
                device_name=device.name,
                device_capability=[device.major, device.minor],
                total_memory_bytes=device.total_memory,
            ),
            indent=2,
        )
        + "\n"
    )
    descriptor = os.open(
        args.output / "warmup.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    local = threading.local()
    targets = {
        (STAGES + ".codec_warmup", "warmup_encoder"): "encoder",
        (STAGES + ".codec_warmup", "warmup_text_encoder"): "text",
        (STAGES + ".codecs", "warmup_source_vae"): "source_vae",
    }

    def observe(frame, event, result):
        if event not in {"call", "return"}:
            return
        module, name = frame.f_globals.get("__name__"), frame.f_code.co_name
        kind = targets.get((module, name))
        if (
            module == PROGRESSIVE
            and name == "forward"
            and getattr(local, "source_warmup", False)
            and type(frame.f_locals.get("self")).__name__ == "CausalEncoderCache"
        ):
            kind = "source_forward"
        if kind is None:
            return
        if kind == "source_vae":
            local.source_warmup = event == "call"
        row = dict(
            kind=kind, event=event, monotonic_ns=time.monotonic_ns(), pid=os.getpid()
        )
        if kind == "source_forward":
            value = frame.f_locals.get("x")
            row["shape"] = list(value.shape)
            row["device"] = str(value.device)
            if event == "return":
                row["returned_output"] = result is not None
        if kind == "text" and event == "return":
            value = frame.f_locals.get("context")
            if value is not None:
                row.update(
                    context_shape=list(value.shape),
                    dtype=str(value.dtype),
                    device=str(value.device),
                )
        os.write(descriptor, (json.dumps(row) + "\n").encode())

    sys.setprofile(observe)
    threading.setprofile(observe)
    try:
        sys.argv = [
            ENCODER,
            "--model",
            str(args.model),
            "--host",
            "127.0.0.1",
            "--advertise-host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--http-port",
            str(args.http_port),
            "--rendezvous-port",
            str(args.rendezvous_port),
        ]
        if args.pipeline_config_path is not None:
            sys.argv += ["--pipeline-config-path", str(args.pipeline_config_path)]
        runpy.run_module(ENCODER, run_name="__main__")
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        os.close(descriptor)


def _validate(events, health):
    def select(kind, event):
        return [row for row in events if row["kind"] == kind and row["event"] == event]

    for kind in ("encoder", "source_vae", "text"):
        if len(select(kind, "call")) != 1 or len(select(kind, "return")) != 1:
            raise RuntimeError(f"Expected exactly one {kind} warmup call and return")
    source = select("source_forward", "call")
    source_returns = select("source_forward", "return")
    if [row["shape"][2] for row in source] != [1, 8] or len(source_returns) != 2:
        raise RuntimeError(
            "Source VAE warmup must execute the existing 1-frame and 8-frame forwards"
        )
    if not all(row["returned_output"] for row in source_returns):
        raise RuntimeError("Source VAE warmup did not return both outputs")
    text = select("text", "return")[0]
    if len(text.get("context_shape", [])) != 2 or min(text["context_shape"]) < 1:
        raise RuntimeError("Text warmup did not return nonempty Qwen features")
    if not all(row.get("device", "").startswith("cuda") for row in [*source, text]):
        raise RuntimeError("Warmup was not executed on CUDA")
    outer_start = select("encoder", "call")[0]["monotonic_ns"]
    source_start = select("source_vae", "call")[0]["monotonic_ns"]
    source_end = select("source_vae", "return")[0]["monotonic_ns"]
    text_start = select("text", "call")[0]["monotonic_ns"]
    text_end = text["monotonic_ns"]
    outer_end = select("encoder", "return")[0]["monotonic_ns"]
    if (
        not outer_start
        <= source_start
        < source_end
        <= text_start
        < text_end
        <= outer_end
    ):
        raise RuntimeError("Encoder warmup did not execute source VAE before Qwen")
    if any(row.get("ready") and row["monotonic_ns"] < outer_end for row in health):
        raise RuntimeError("Encoder reported readiness before warmup completed")
    if not any(row.get("status") == 503 for row in health):
        raise RuntimeError("No not-ready health response was observed during startup")
    if not health[-1].get("ready"):
        raise RuntimeError("Encoder did not reach readiness")


def _controller(args):
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        str(args.model),
        "--output",
        str(args.output),
        "--revision",
        args.revision,
        "--port",
        str(args.port),
        "--http-port",
        str(args.http_port),
        "--rendezvous-port",
        str(args.rendezvous_port),
    ]
    if args.pipeline_config_path is not None:
        command += ["--pipeline-config-path", str(args.pipeline_config_path)]
    receipt = dict(
        revision=args.revision,
        model=str(args.model),
        service_count=1,
        command=command,
        passed=False,
        cleaned=False,
        health=[],
    )
    child = None
    error = None
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    previous_handlers = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        with (args.output / "worker.log").open("wb") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            receipt["pid"] = child.pid
            opener = build_opener(ProxyHandler({}))
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if stopped:
                    raise RuntimeError("Encoder warmup validation was interrupted")
                if child.poll() is not None:
                    raise RuntimeError(
                        f"Encoder exited before readiness: {child.returncode}"
                    )
                row = dict(monotonic_ns=time.monotonic_ns(), ready=False)
                try:
                    try:
                        response = opener.open(
                            f"http://127.0.0.1:{args.http_port}/health", timeout=1
                        )
                    except HTTPError as exc:
                        response = exc
                    with response:
                        body = json.load(response)
                        row.update(
                            status=response.code,
                            ready=body.get("ready") is True,
                            role=body.get("role"),
                        )
                    if row["role"] != "encoder" or row["status"] not in (200, 503):
                        raise RuntimeError("Unexpected Encoder health response")
                except (OSError, ValueError) as exc:
                    row["error"] = type(exc).__name__
                row["monotonic_ns"] = time.monotonic_ns()
                receipt["health"].append(row)
                if row["ready"] and row.get("status") == 200:
                    break
                time.sleep(0.2)
            else:
                raise TimeoutError("Encoder startup warmup exceeded the deadline")
            events = [
                json.loads(line)
                for line in (args.output / "warmup.jsonl").read_text().splitlines()
            ]
            receipt["events"] = events
            _validate(events, receipt["health"])
            receipt["passed"] = True
    except Exception as exc:
        error = str(exc)
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=30)
                receipt["cleaned"] = True
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
                receipt["cleaned"] = True
                error = error or "Encoder required forced shutdown"
            receipt["returncode"] = child.returncode
        if error is not None:
            receipt.update(passed=False, error=error)
        (args.output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    if not receipt["passed"]:
        raise RuntimeError(error or "Encoder warmup validation failed")
    print("ENCODER_STARTUP_WARMUP_PASS " + str(args.output / "result.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pipeline-config-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--revision", required=True, help="Pushed source commit used by the GPU run"
    )
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--http-port", type=int, default=31000)
    parser.add_argument("--rendezvous-port", type=int, default=32000)
    parser.add_argument("--startup-timeout", type=float, default=900)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    modes.add_argument(
        "--preflight",
        action="store_true",
        help="Check imports, model metadata and local transport without loading weights",
    )
    args = parser.parse_args()
    args.model, args.output = args.model.resolve(), args.output.resolve()
    if args.pipeline_config_path is not None:
        args.pipeline_config_path = args.pipeline_config_path.resolve()
    if args.startup_timeout <= 0:
        parser.error("startup-timeout must be positive")
    if len({args.port, args.http_port, args.rendezvous_port}) != 3:
        parser.error("Worker ports must not overlap")
    if args.worker:
        _worker(args)
    elif args.preflight:
        _preflight(args)
    else:
        _controller(args)


if __name__ == "__main__":
    main()
