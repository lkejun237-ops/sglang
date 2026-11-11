# SPDX-License-Identifier: Apache-2.0
"""Observe the ordinary SP2 DiT startup warmup, without Encoder or Decoder.

Both ranks run dit_worker through torchrun. A narrow call observer records the
existing private one-latent request and its cleanup, without replacing model
methods. These observed startup timings are not performance benchmarks.
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

from validate_decoder_startup_warmup import _write_json
from validate_encoder_startup_warmup import STAGES

DIT = STAGES + ".dit_worker"
WARMUP = STAGES + ".dit_warmup"
RANKS = (0, 1)


def _worker_arguments(args):
    command = [
        "--model",
        str(args.model),
        "--pipeline-config-path",
        str(args.pipeline_config_path),
        "--sp-degree",
        "2",
        "--http-host",
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
    if args.quantization is not None:
        command += ["--quantization", args.quantization]
    if args.mem_fraction_static is not None:
        command += ["--mem-fraction-static", str(args.mem_fraction_static)]
    return command


def _preflight(args):
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        revision=args.revision,
        mode="preflight",
        sp_degree=2,
        passed=False,
        files={},
        modules={},
    )
    try:
        receipt["python"] = sys.version
        receipt["packages"] = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "safetensors",
                "uvicorn",
                "fastapi",
                "msgspec",
                "pyzmq",
            )
        }
        worker = importlib.import_module(DIT)
        bootstrap = importlib.import_module(STAGES + ".encoder_worker")
        ServerArgs, _ = bootstrap.initialize_runtime_imports()
        for name in (
            DIT,
            WARMUP,
            STAGES + ".stage",
            STAGES + ".dit_link",
            "sglang.multimodal_gen.runtime.models.dits.zing_h3",
            "sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table",
            "sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader",
            "sglang.multimodal_gen.runtime.realtime.worker_parallel",
            "sglang.multimodal_gen.runtime.realtime.worker_group_dit",
            "torch",
        ):
            module = importlib.import_module(name)
            receipt["modules"][name] = str(Path(module.__file__).resolve())
        cli = worker.argument_parser().parse_args(_worker_arguments(args))
        # Match dit_worker.main's ServerArgs without entering its model loader.
        server = ServerArgs.from_kwargs(
            model_path=str(cli.model),
            pipeline_config_path=str(cli.pipeline_config_path),
            num_gpus=2,
            quantization=cli.quantization,
            sp_degree=2,
            ulysses_degree=2,
            ring_degree=1,
            encoder_parallel="replicate",
            attention_backend="torch_sdpa",
            dit_cpu_offload=False,
            dit_layerwise_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=False,
            use_fsdp_inference=False,
            performance_mode="speed",
            warmup_mode="off",
            mem_fraction_static=cli.mem_fraction_static,
            realtime_max_sessions=1,
            realtime_max_sessions_per_worker=1,
            zing_source_joint_forward=cli.zing_source_joint_forward,
            **{
                name: value
                for name, value in vars(cli).items()
                if name.startswith("zing_block0_cache_")
                or name == "zing_block0_mooncake_config"
            },
        )

        def record(path):
            row = dict(exists=path.is_file(), resolved_path=str(path.resolve()))
            receipt["files"][str(path)] = row
            if not row["exists"]:
                raise FileNotFoundError(f"Required DiT asset is missing: {path}")
            row["bytes"] = path.stat().st_size
            if path.suffix == ".json":
                content = path.read_bytes()
                row["sha256"] = hashlib.sha256(content).hexdigest()
                return json.loads(content)

        record(args.pipeline_config_path)
        for name in ("model_index.json", "zing_config.json", "transformer/config.json"):
            record(args.model / name)
        server.pipeline_config.vae_config.update_model_arch(
            record(args.model / "video_vae/config.json")
        )
        weights = sorted((args.model / "transformer").glob("*.safetensors"))
        if not weights:
            raise FileNotFoundError("DiT transformer weights are missing")
        for path in weights:
            record(path)
        for path in sorted(
            (args.model / "transformer").glob("*.safetensors.index.json")
        ):
            for shard in set(record(path)["weight_map"].values()):
                record(path.parent / shard)
        table = server.pipeline_config.resolve_adaln_table_path(str(args.model))
        if table is not None:
            manifest = record(Path(table) / "manifest.json")
            record(Path(table) / "table.safetensors")
            receipt["adaln_arithmetic"] = manifest.get("arithmetic")
        adapter = importlib.import_module(
            "sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_h3_source_adapter"
        )
        profile = adapter._with_source_pipeline_policy(
            record(args.model / "zing_realtime_profile.json"),
            joint_forward=server.zing_source_joint_forward,
        )
        receipt.update(
            pipeline=type(server.pipeline_config).__name__,
            sampling=adapter._source_sampling_schedule(server, profile),
            quantization=server.quantization,
            mem_fraction_static=server.mem_fraction_static,
            adaln_table=table,
            passed=True,
        )
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write_json(args.output / "result.json", receipt)
    print("DIT_PREFLIGHT_PASS " + str(args.output / "result.json"), flush=True)


def _observer(descriptor, rank):
    def observe(frame, event, result):
        if (
            event not in {"call", "return"}
            or frame.f_globals.get("__name__") != WARMUP
            or frame.f_code.co_name != "warmup_dit"
        ):
            return
        values = frame.f_locals
        parallel = values.get("parallel")
        row = dict(
            event=event,
            rank=rank,
            monotonic_ns=time.monotonic_ns(),
            sp_world_size=getattr(parallel, "world_size", None),
            sp_rank=getattr(parallel, "rank_in_group", None),
        )
        if event == "return":
            outputs, completed = values.get("outputs", []), values.get("completed", [])
            channel = values.get("channel")
            row.update(
                error=values.get("error") is not None,
                source_shape=None
                if values.get("moments") is None
                else list(values["moments"].shape),
                output_shapes=[list(item["tensor"].shape) for item in outputs],
                output_devices=[str(item["tensor"].device) for item in outputs],
                completed_latents=[item.get("latent_frames") for item in completed],
                expected_shape=list(values.get("expected", ())),
                request_released=channel is not None
                and channel.service.owner is None
                and channel.service.worker is None,
                queue_empty=channel is not None
                and not channel.service.queue
                and not channel.replies,
            )
        os.write(descriptor, (json.dumps(row) + "\n").encode())

    return observe


def _worker(args):
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    runtime = dict(
        rank=rank,
        local_rank=local_rank,
        world_size=int(os.environ["WORLD_SIZE"]),
        pid=os.getpid(),
        revision=args.revision,
    )
    # Publish rank ownership even if importing torch or CUDA setup fails.
    _write_json(args.output / f"runtime.rank{rank}.json", runtime)
    logging.basicConfig(
        level=logging.INFO, format=f"rank={rank} %(asctime)s %(levelname)s %(message)s"
    )
    import torch

    properties = torch.cuda.get_device_properties(local_rank)
    runtime.update(
        python=sys.version,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device_name=properties.name,
        device_capability=[properties.major, properties.minor],
        total_memory_bytes=properties.total_memory,
    )
    _write_json(args.output / f"runtime.rank{rank}.json", runtime)
    descriptor = os.open(
        args.output / f"warmup.rank{rank}.jsonl",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    observer = _observer(descriptor, rank)
    sys.setprofile(observer)
    threading.setprofile(observer)
    outcome = dict(rank=rank, passed=False, test_control_interrupt=False)
    try:
        sys.argv = [DIT, *_worker_arguments(args)]
        try:
            runpy.run_module(DIT, run_name="__main__")
        except KeyboardInterrupt:
            if not (args.output / "stop_requested.json").is_file():
                raise
            outcome["test_control_interrupt"] = True
            logging.info(
                "DiT test control interruption completed; not a business shutdown request"
            )
        if not outcome["test_control_interrupt"]:
            raise RuntimeError("DiT worker returned without the test stop request")
        if torch.distributed.is_initialized():
            raise RuntimeError(
                "DiT worker did not clean up its distributed environment"
            )
        outcome["passed"] = True
    except BaseException as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        os.close(descriptor)
        _write_json(args.output / f"exit.rank{rank}.json", outcome)


def _validate(events, health):
    completed = []
    for rank in RANKS:
        rows = events.get(rank, [])
        if [row["event"] for row in rows] != ["call", "return"]:
            raise RuntimeError(f"Rank {rank} did not run exactly one DiT warmup")
        if any(
            row["rank"] != rank or row["sp_rank"] != rank or row["sp_world_size"] != 2
            for row in rows
        ):
            raise RuntimeError(
                "DiT warmup did not participate in the expected SP2 group"
            )
        result = rows[-1]
        expected = result["expected_shape"]
        if (
            len(expected) != 5
            or expected[0] != 1
            or expected[2] != 1
            or result["output_shapes"] != [expected]
            or result["completed_latents"] != [1]
            or result["error"]
        ):
            raise RuntimeError(f"Rank {rank} did not complete one DiT output latent")
        if result["source_shape"] != [expected[0], expected[1] * 2, *expected[2:]]:
            raise RuntimeError(
                f"Rank {rank} did not consume one source posterior latent"
            )
        if result["output_devices"] != [f"cuda:{rank}"]:
            raise RuntimeError(f"Rank {rank} did not produce its CUDA output")
        if not result["request_released"] or not result["queue_empty"]:
            raise RuntimeError(f"Rank {rank} left the private warmup request occupied")
        if result["monotonic_ns"] <= rows[0]["monotonic_ns"]:
            raise RuntimeError("Invalid DiT warmup observation order")
        completed.append(result)
    if completed[0]["expected_shape"] != completed[1]["expected_shape"]:
        raise RuntimeError("SP2 warmup ranks disagree on the output shape")
    if (
        not health
        or not health[-1].get("ready")
        or not any(row.get("status") == 503 for row in health)
    ):
        raise RuntimeError("DiT health did not transition from not ready to ready")
    if any(
        row.get("ready")
        and row["monotonic_ns"] < max(item["monotonic_ns"] for item in completed)
        for row in health
    ):
        raise RuntimeError("DiT reported readiness before both rank warmups completed")


def _rank_processes(output):
    result = {}
    for rank in RANKS:
        try:
            row = json.loads((output / f"runtime.rank{rank}.json").read_text())
            result[rank] = row["pid"]
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return result


def _signal(pid, signum, *, group=False):
    try:
        (os.killpg if group else os.kill)(pid, signum)
    except ProcessLookupError:
        pass


def _stop(child, args, receipt):
    receipt["stop_method"] = "test_control_interrupt"
    _write_json(
        args.output / "stop_requested.json",
        dict(signal="SIGINT", reason=receipt["stop_method"]),
    )
    ranks = _rank_processes(args.output)
    # torchrun owns separate rank processes. Signal them directly so its agent
    # can observe their zero exits instead of interrupting the agent itself.
    for pid in ranks.values():
        _signal(pid, signal.SIGINT)
    if len(ranks) != 2:
        _signal(child.pid, signal.SIGTERM, group=True)
    try:
        child.wait(timeout=60)
    except subprocess.TimeoutExpired:
        receipt["shutdown_error"] = "DiT test required forced shutdown"
    finally:
        # A crashed launcher may leave ranks alive. These are only this test's
        # registered PIDs; never address unrelated processes on the GPU host.
        for pid in _rank_processes(args.output).values():
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            _signal(pid, signal.SIGKILL)
            receipt.setdefault("shutdown_error", "DiT rank survived its launcher")
        if child.poll() is None:
            _signal(child.pid, signal.SIGKILL, group=True)
            child.wait(timeout=10)
    receipt["returncode"] = child.returncode
    exits = []
    for rank in RANKS:
        path = args.output / f"exit.rank{rank}.json"
        exits.append(
            json.loads(path.read_text())
            if path.is_file()
            else dict(rank=rank, passed=False)
        )
    receipt["rank_exits"] = exits
    receipt["cleaned"] = (
        child.returncode == 0
        and all(
            row.get("passed") and row.get("test_control_interrupt") for row in exits
        )
        and "shutdown_error" not in receipt
    )
    if not receipt["cleaned"]:
        receipt.setdefault(
            "shutdown_error", "DiT ranks or torchrun did not exit cleanly"
        )


def _controller(args):
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--max-restarts=0",
        str(Path(__file__).resolve()),
        "--worker",
        *sys.argv[1:],
    ]
    receipt = dict(
        revision=args.revision,
        passed=False,
        cleaned=False,
        sp_degree=2,
        command=command,
        health=[],
    )
    interrupted = False
    child = None

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    previous = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        with (args.output / "worker.log").open("wb") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            opener = build_opener(ProxyHandler({}))
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if interrupted or child.poll() is not None:
                    raise RuntimeError("DiT test stopped before readiness")
                row = dict(ready=False)
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
                            status=response.code, ready=body.get("ready") is True
                        )
                    if body.get("role") != "dit" or row["status"] not in (200, 503):
                        raise RuntimeError("Unexpected DiT health response")
                except (OSError, ValueError) as exc:
                    row["error"] = type(exc).__name__
                row["monotonic_ns"] = time.monotonic_ns()
                receipt["health"].append(row)
                if row["ready"] and row.get("status") == 200:
                    break
                time.sleep(0.2)
            else:
                raise TimeoutError("DiT startup warmup exceeded the deadline")
            events = {
                rank: [
                    json.loads(line)
                    for line in (args.output / f"warmup.rank{rank}.jsonl")
                    .read_text()
                    .splitlines()
                ]
                for rank in RANKS
            }
            receipt["events"] = events
            _validate(events, receipt["health"])
            receipt["warmup_passed"] = receipt["passed"] = True
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if child is not None:
                _stop(child, args, receipt)
        except Exception as exc:
            receipt["shutdown_error"] = f"{type(exc).__name__}: {exc}"
        if "error" in receipt or "shutdown_error" in receipt:
            receipt["passed"] = False
        _write_json(args.output / "result.json", receipt)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if not receipt["passed"]:
        raise RuntimeError(
            receipt.get(
                "error", receipt.get("shutdown_error", "DiT startup warmup failed")
            )
        )
    print("DIT_STARTUP_WARMUP_PASS " + str(args.output / "result.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pipeline-config-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--quantization")
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument("--port", type=int, default=34000)
    parser.add_argument("--http-port", type=int, default=33000)
    parser.add_argument("--rendezvous-port", type=int, default=35000)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.model, args.output = args.model.resolve(), args.output.resolve()
    args.pipeline_config_path = args.pipeline_config_path.resolve()
    ports = (args.port, args.http_port, args.rendezvous_port)
    if len(set(ports)) != 3 or not all(0 < port < 65536 for port in ports):
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
