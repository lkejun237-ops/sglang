# SPDX-License-Identifier: Apache-2.0
"""Opt-in benchmark worker launcher and allocation-history checkpoints."""

import argparse
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path


def install():
    # Keep the launcher as multiprocessing's main module so spawn workers install
    # the same observer before model loading. No forward hooks alter Graph routes.
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        initialize_runtime_imports,
    )

    initialize_runtime_imports()
    import memory_probe

    memory_probe.install()


if os.environ.get("ZING_BENCHMARK_MEMORY_DIR"):
    install()


def checkpoint(directory, label, operation, expected, timeout):
    if not label.replace("-", "").isalnum():
        raise ValueError("Use an alphanumeric checkpoint label with optional hyphens")
    path = directory / "memory-control.json"
    memory = directory / "memory"
    if len(list(memory.glob("process-*.json"))) != expected:
        raise ValueError("Worker identity count does not match --workers")
    sequence = json.loads(path.read_text())["sequence"] + 1 if path.exists() else 1
    temporary = path.with_suffix(".next")
    temporary.write_text(
        json.dumps(dict(sequence=sequence, label=label, operation=operation))
    )
    os.replace(temporary, path)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        errors = list(memory.glob("ERROR-*.txt"))
        if errors:
            raise RuntimeError("Memory observer failed: " + errors[0].name)
        acks = [
            json.loads(p.read_text()) for p in memory.glob(f"ack-{sequence}-*.json")
        ]
        if len(acks) == expected and len({v["pid"] for v in acks}) == expected:
            if not all(v["passed"] and v["label"] == label for v in acks):
                raise ValueError("Worker checkpoint acknowledgement failed")
            return acks
        if len(acks) > expected:
            raise ValueError("More workers acknowledged than requested")
        time.sleep(0.1)
    raise TimeoutError(
        f"Checkpoint {label} did not receive {expected} worker acknowledgements"
    )


def sample_nvml(output, duration, interval):
    """Read physical-card counters separately from the allocator observers."""
    deadline = time.monotonic() + duration
    with output.open("x") as target:
        while time.monotonic() < deadline:
            rows = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            ).splitlines()
            target.write(json.dumps(dict(monotonic=time.monotonic(), rows=rows)) + "\n")
            target.flush()
            time.sleep(min(interval, max(0, deadline - time.monotonic())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser(
        "run", help="Launch an existing server, encoder or Qwen service"
    )
    run.add_argument("--role", choices=("server", "encoder", "qwen"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--max-entries", type=int, default=8_000_000)
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    control = sub.add_parser(
        "checkpoint", help="Snapshot/reset/disable all explicitly counted workers"
    )
    control.add_argument("--output", type=Path, required=True)
    control.add_argument("--label", required=True)
    control.add_argument(
        "--operation", choices=("snapshot", "reset", "disable"), default="snapshot"
    )
    control.add_argument("--workers", type=int, required=True)
    control.add_argument("--timeout", type=float, default=300)
    nvml = sub.add_parser(
        "sample", help="Bounded NVML sampling without model execution"
    )
    nvml.add_argument("--output", type=Path, required=True)
    nvml.add_argument("--duration", type=float, required=True)
    nvml.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.action == "sample":
        if any(not math.isfinite(v) or v <= 0 for v in (args.duration, args.interval)):
            parser.error("Duration and interval must be positive")
        sample_nvml(args.output, args.duration, args.interval)
        return
    if args.action == "checkpoint":
        if args.workers <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error("Workers and timeout must be positive")
        print(
            json.dumps(
                checkpoint(
                    args.output, args.label, args.operation, args.workers, args.timeout
                )
            )
        )
        return
    if args.max_entries <= 0:
        parser.error("--max-entries must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        ZING_BENCHMARK_MEMORY_DIR=str(args.output.resolve()),
        ZING_BENCHMARK_ROLE=args.role,
        ZING_BENCHMARK_MEMORY_ENTRIES=str(args.max_entries),
    )
    install()
    sys.argv = [
        args.role,
        *(args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments),
    ]
    if args.role == "server":
        from sglang.multimodal_gen.runtime import launch_server

        multiprocessing.set_start_method("spawn", force=True)
        try:
            launch_server.dispatch_launch(
                launch_server.prepare_server_args(sys.argv[1:])
            )
        finally:
            launch_server.kill_process_tree(os.getpid(), include_parent=False)
    elif args.role == "encoder":
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
            main as run_encoder,
        )

        run_encoder()
    else:
        from sglang.multimodal_gen.runtime.conditioning.cli import main as run_qwen

        run_qwen()


if __name__ == "__main__":
    main()
