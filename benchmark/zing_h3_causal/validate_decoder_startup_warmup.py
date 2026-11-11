# SPDX-License-Identifier: Apache-2.0
"""Validate one Decoder startup shared by three services, without other roles.

The ordinary decoder_worker consumes one latent through its existing warmup.
The observer only records Python calls and tensor metadata; it does not replace
model methods. Observed timings include overhead and are not benchmarks.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import runpy
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

from validate_encoder_startup_warmup import STAGES

DECODER = STAGES + ".decoder_worker"
LYNN = "sglang.multimodal_gen.runtime.models.vaes.lynn_streaming"


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _preflight(args):
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        revision=args.revision,
        mode="preflight",
        model=str(args.model),
        service_count=args.service_count,
        passed=False,
        modules={},
        files={},
    )
    try:
        receipt["python"] = sys.version
        receipt["packages"] = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "uvicorn",
                "websockets",
                "fastapi",
                "msgspec",
                "pyzmq",
            )
        }
        worker = importlib.import_module(DECODER)
        ServerArgs, _ = worker.initialize_runtime_imports()
        for name in (
            DECODER,
            STAGES + ".codec_warmup",
            LYNN + ".loader",
            LYNN + ".model",
            LYNN + ".stream",
            "sglang.multimodal_gen.runtime.realtime.worker_http",
            "sglang.multimodal_gen.runtime.realtime.decoder_output",
            "torch",
            "uvicorn",
            "websockets",
        ):
            module = importlib.import_module(name)
            receipt["modules"][name] = str(Path(module.__file__).resolve())
        # Mirror the worker's actual parsing; do not instantiate the model.
        server_args = ServerArgs.from_kwargs(
            model_path=str(args.model),
            num_gpus=1,
            video_decoder_backend="native",
            attention_backend="torch_sdpa",
            warmup_mode="off",
        )
        spec = worker.normalize_decoder_spec(
            json.loads(args.decoder_config),
            video_decoder_backend=None,
            vae_decode_precision="fp32",
        )
        if spec.type != "lynn_streaming":
            raise ValueError("The decoder service supports only lynn_streaming")
        loader = importlib.import_module(LYNN + ".loader")
        architecture = loader.read_architecture(spec.model_dir)
        for path in (
            args.decoder_config_path,
            args.model / "model_index.json",
            args.model / "zing_config.json",
            Path(spec.model_dir) / "config.json",
            Path(spec.checkpoint_path),
        ):
            record = dict(exists=path.is_file(), resolved_path=str(path.resolve()))
            receipt["files"][str(path)] = record
            if not record["exists"]:
                raise FileNotFoundError(
                    f"Required Decoder component is missing: {path}"
                )
            record["bytes"] = path.stat().st_size
            if path.suffix == ".json":
                content = path.read_bytes()
                json.loads(content)
                record["sha256"] = hashlib.sha256(content).hexdigest()
        receipt.update(
            server_args=type(server_args).__name__,
            pipeline=type(server_args.pipeline_config).__name__,
            decoder=spec.redacted(),
            selected_decoder_config=json.loads(args.decoder_config),
            latent_channels=architecture["latent_channels"],
            decoder_config_sha256=hashlib.sha256(
                args.decoder_config.encode()
            ).hexdigest(),
            passed=True,
        )
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write_json(args.output / "result.json", receipt)
    print("DECODER_PREFLIGHT_PASS " + str(args.output / "result.json"), flush=True)


def _observer(descriptor):
    local = threading.local()
    targets = {
        (STAGES + ".codec_warmup", "warmup_decoder"): "warmup",
        (LYNN + ".model", "new_stream"): "new_stream",
        (LYNN + ".model", "decode_step"): "decode_step",
        (LYNN + ".stream", "push"): "push",
        (LYNN + ".stream", "finish"): "finish",
    }

    def observe(frame, event, result):
        if event not in {"call", "return"}:
            return
        kind = targets.get((frame.f_globals.get("__name__"), frame.f_code.co_name))
        if kind is None or (kind != "warmup" and not getattr(local, "active", False)):
            return
        if kind == "warmup":
            local.active = event == "call"
        row = dict(kind=kind, event=event, monotonic_ns=time.monotonic_ns())
        values = frame.f_locals
        if kind in {"push", "decode_step"}:
            latent = values.get("latent" if kind == "push" else "z_t")
            row.update(shape=list(latent.shape), device=str(latent.device))
            if event == "return" and result is not None:
                output = result if kind == "push" else result[0]
                row["output_shape"] = list(output.shape)
        if kind == "new_stream":
            row.update(
                frame_limit=values["frame_limit"], group_is_none=values["group"] is None
            )
        if kind == "warmup" and event == "return":
            latent, output, stream = (
                values.get("latent"),
                values.get("output"),
                values.get("stream"),
            )
            row.update(
                latent_shape=None if latent is None else list(latent.shape),
                output_shape=None if output is None else list(output.shape),
                error=values.get("error") is not None,
                stream_closed=stream is not None and stream.closed,
                stream_state_released=stream is not None and stream.state is None,
            )
        os.write(descriptor, (json.dumps(row) + "\n").encode())

    return observe


def _worker(args):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    import torch

    properties = torch.cuda.get_device_properties(0)
    _write_json(
        args.output / "runtime.json",
        dict(
            revision=args.revision,
            python=sys.version,
            torch_version=torch.__version__,
            torch_path=str(Path(torch.__file__).resolve()),
            cuda_version=torch.version.cuda,
            device_name=properties.name,
            device_capability=[properties.major, properties.minor],
            total_memory_bytes=properties.total_memory,
        ),
    )
    descriptor = os.open(
        args.output / "warmup.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    observe = _observer(descriptor)
    sys.setprofile(observe)
    threading.setprofile(observe)
    try:
        sys.argv = [
            DECODER,
            "--model",
            str(args.model),
            "--decoder-config",
            args.decoder_config,
            "--http-host",
            "127.0.0.1",
            "--advertise-host",
            "127.0.0.1",
            "--service-count",
            str(args.service_count),
            "--port",
            str(args.port),
            "--http-port",
            str(args.http_port),
            "--output-port",
            str(args.output_port),
        ]
        try:
            runpy.run_module(DECODER, run_name="__main__")
        except KeyboardInterrupt:
            if not (args.output / "stop_requested.json").is_file():
                raise
            # SIGINT unwinds the ordinary worker's finally block before arriving
            # here. The test owns this rank and finishes its process group too.
            logging.info(
                "Decoder test control interruption completed; not a business shutdown request"
            )
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        os.close(descriptor)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _validate(events, health, ports):
    selected = {}
    for kind in ("warmup", "new_stream", "push", "decode_step", "finish"):
        for event in ("call", "return"):
            rows = [
                row for row in events if (row["kind"], row["event"]) == (kind, event)
            ]
            if len(rows) != 1:
                raise RuntimeError(f"Expected exactly one Decoder {kind} {event}")
            selected[kind, event] = rows[0]
    completed = selected["warmup", "return"]
    if completed["error"] or completed["output_shape"] != [1, 3, 1, 480, 832]:
        raise RuntimeError("Decoder warmup did not complete one 832x480 frame")
    if not completed["stream_closed"] or not completed["stream_state_released"]:
        raise RuntimeError("Decoder warmup stream was not released")
    new_stream = selected["new_stream", "call"]
    if new_stream["frame_limit"] != 1 or not new_stream["group_is_none"]:
        raise RuntimeError("Decoder warmup did not create one local frame stream")
    for kind in ("push", "decode_step"):
        row = selected[kind, "call"]
        if (
            len(row["shape"]) != 5
            or row["shape"][2] != 1
            or not row["device"].startswith("cuda")
        ):
            raise RuntimeError("Decoder warmup did not consume one CUDA latent")
        if not selected[kind, "return"].get("output_shape"):
            raise RuntimeError("Decoder warmup did not return decoded pixels")
    for row in events:
        if (
            not selected["warmup", "call"]["monotonic_ns"]
            <= row["monotonic_ns"]
            <= completed["monotonic_ns"]
        ):
            raise RuntimeError("Decoder observations fell outside the startup warmup")
    for port in ports:
        samples = [row for row in health if row["port"] == port]
        if not any(row.get("status") == 503 for row in samples) or not samples[-1].get(
            "ready"
        ):
            raise RuntimeError(f"Decoder service {port} did not transition to ready")
        if any(
            row.get("ready") and row["monotonic_ns"] < completed["monotonic_ns"]
            for row in samples
        ):
            raise RuntimeError("Decoder service reported readiness before warmup")


def _controller(args):
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", *sys.argv[1:]]
    ports = list(range(args.http_port, args.http_port + args.service_count))
    receipt = dict(
        revision=args.revision,
        passed=False,
        cleaned=False,
        service_count=args.service_count,
        command=command,
        health=[],
    )
    environment = os.environ | dict(
        LOCAL_RANK="0",
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(args.master_port),
    )
    child = None
    interrupted = False

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    previous = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        with (args.output / "worker.log").open("wb") as log:
            child = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            opener, ready = build_opener(ProxyHandler({})), set()
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if interrupted or child.poll() is not None:
                    raise RuntimeError("Decoder stopped before readiness")
                for port in ports:
                    row = dict(port=port, ready=False)
                    try:
                        try:
                            response = opener.open(
                                f"http://127.0.0.1:{port}/health", timeout=1
                            )
                        except HTTPError as exc:
                            response = exc
                        with response:
                            body = json.load(response)
                            row.update(
                                status=response.code, ready=body.get("ready") is True
                            )
                        if body.get("role") != "decoder" or row["status"] not in (
                            200,
                            503,
                        ):
                            raise RuntimeError("Unexpected Decoder health response")
                    except (OSError, ValueError) as exc:
                        row["error"] = type(exc).__name__
                    row["monotonic_ns"] = time.monotonic_ns()
                    receipt["health"].append(row)
                    if row["ready"] and row.get("status") == 200:
                        ready.add(port)
                    else:
                        ready.discard(port)
                if ready == set(ports):
                    break
                time.sleep(0.2)
            else:
                raise TimeoutError("Decoder startup warmup exceeded the deadline")
            receipt["events"] = [
                json.loads(line)
                for line in (args.output / "warmup.jsonl").read_text().splitlines()
            ]
            _validate(receipt["events"], receipt["health"], ports)
            receipt["passed"] = True
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if child is not None:
            receipt["stop_method"] = "test_control_interrupt"
            _write_json(
                args.output / "stop_requested.json",
                dict(
                    signal="SIGINT",
                    reason=receipt["stop_method"],
                    monotonic_ns=time.monotonic_ns(),
                ),
            )
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
                receipt["error"] = "Decoder required forced shutdown"
            receipt.update(cleaned=True, returncode=child.returncode)
            if child.returncode != 0:
                receipt.setdefault(
                    "error", f"Decoder shutdown returned {child.returncode}"
                )
        if "error" in receipt:
            receipt["passed"] = False
        _write_json(args.output / "result.json", receipt)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if not receipt["passed"]:
        raise RuntimeError(receipt.get("error", "Decoder startup warmup failed"))
    print("DECODER_STARTUP_WARMUP_PASS " + str(args.output / "result.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--decoder-config-path", type=Path, required=True)
    parser.add_argument("--service-count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", type=int, default=37000)
    parser.add_argument("--http-port", type=int, default=36000)
    parser.add_argument("--output-port", type=int, default=38000)
    parser.add_argument("--master-port", type=int, default=35000)
    parser.add_argument("--startup-timeout", type=float, default=900)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.model, args.output = args.model.resolve(), args.output.resolve()
    args.decoder_config_path = args.decoder_config_path.resolve()
    config = json.loads(args.decoder_config_path.read_text())
    if isinstance(config, dict) and "decoder_config" in config:
        config = config["decoder_config"]
    if not isinstance(config, dict):
        parser.error("decoder-config must be a JSON object")
    args.decoder_config = json.dumps(config)
    if args.service_count < 1:
        parser.error("service-count must be positive")
    ports = [
        args.master_port,
        *range(args.port, args.port + args.service_count),
        *range(args.http_port, args.http_port + args.service_count),
        *range(args.output_port, args.output_port + args.service_count),
    ]
    if len(set(ports)) != len(ports) or not all(0 < port < 65536 for port in ports):
        parser.error("Worker ports must be distinct valid TCP ports")
    if args.startup_timeout <= 0:
        parser.error("startup-timeout must be positive")
    if args.preflight:
        _preflight(args)
    elif args.worker:
        _worker(args)
    else:
        _controller(args)


if __name__ == "__main__":
    main()
