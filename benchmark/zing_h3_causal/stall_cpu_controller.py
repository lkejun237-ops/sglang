# SPDX-License-Identifier: Apache-2.0
"""Execute SHA-bound CPU preparation/readback tasks; never submit GPU jobs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--harness-source", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-seconds", type=int, default=24 * 3600)
    args = parser.parse_args()
    if not args.root.is_absolute() or not 1 <= args.max_seconds <= 48 * 3600:
        raise ValueError(
            "Expected an absolute owned task directory and bounded duration"
        )
    sys.path.insert(0, str(args.harness_source / "benchmark/zing_h3_causal"))
    from frozen_source import verify_source

    verify_source(args.harness_source, args.source_revision)
    import torch
    from owned_process import owned_popen
    from run_isolated_stall_suite import stop

    if torch.cuda.device_count() or torch.cuda.is_initialized():
        raise RuntimeError(
            "CPU controller must have zero CUDA devices and no CUDA context"
        )
    root = args.root
    queue = root / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    write(
        root / "READY.json",
        dict(
            time=time.time(),
            source_revision=args.source_revision,
            controller_source_revision=os.environ.get("STALL_CONTROLLER_CODE_REVISION"),
            controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            cuda_count=0,
            scope="CPU task exit status only; individual results require independent acceptance",
        ),
    )
    print("CPU_CONTROLLER_READY", flush=True)
    deadline = time.monotonic() + args.max_seconds
    done = set()
    budget_exhausted = False
    while time.monotonic() < deadline and not (root / "STOP").exists():
        for request in sorted(queue.glob("*.json")):
            if deadline - time.monotonic() <= 120:
                budget_exhausted = True
                break
            if request.name in done:
                continue
            result_path = root / (request.stem + ".result.json")
            # A restarted controller must never replay an already executed task.
            if result_path.exists():
                prior = json.loads(result_path.read_text())
                if (
                    prior.get("request_sha256")
                    != hashlib.sha256(request.read_bytes()).hexdigest()
                ):
                    raise ValueError(
                        "A completed CPU task name was reused with different contents"
                    )
                done.add(request.name)
                continue
            result = dict(
                task=request.name,
                started=time.time(),
                passed=False,
                request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),
                scope="Task execution only; not GPU or numerical acceptance",
            )
            process = None
            temporary = None
            try:
                task = json.loads(request.read_text())
                relative = Path(task["script"])
                script = root / relative
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not script.resolve().is_relative_to(root.resolve())
                ):
                    raise ValueError(
                        "Task script must remain inside the owned controller root"
                    )
                data = script.read_bytes()
                if (
                    len(data) > 2 << 20
                    or hashlib.sha256(data).hexdigest() != task["sha256"]
                ):
                    raise ValueError("CPU task script hash mismatch")
                arguments = task.get("args", [])
                timeout = task.get("timeout", 1800)
                if (
                    not isinstance(arguments, list)
                    or not all(isinstance(v, str) for v in arguments)
                    or type(timeout) is not int
                    or not 1 <= timeout <= 7200
                ):
                    raise ValueError(
                        "Expected string task arguments and a bounded timeout"
                    )
                temporary = tempfile.TemporaryDirectory(prefix="stall-cpu-task-")
                local = Path(temporary.name) / "task.py"
                local.write_bytes(data)
                with (root / (request.stem + ".log")).open("xb") as log:
                    remaining = deadline - time.monotonic() - 120
                    if remaining <= 0:
                        raise TimeoutError(
                            "Controller budget exhausted before task launch"
                        )
                    process = owned_popen(
                        [sys.executable, "-u", str(local), *arguments],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    result["returncode"] = process.wait(timeout=min(timeout, remaining))
            except BaseException:
                result["error"] = traceback.format_exc()
            finally:
                try:
                    if process is not None:
                        stop(process)
                except BaseException:
                    result["cleanup_error"] = traceback.format_exc()
                finally:
                    if temporary is not None:
                        try:
                            temporary.cleanup()
                        except BaseException:
                            result["temporary_cleanup_error"] = traceback.format_exc()
                result["ended"] = time.time()
                result["passed"] = (
                    result.get("returncode") == 0
                    and not result.get("error")
                    and not result.get("cleanup_error")
                    and not result.get("temporary_cleanup_error")
                )
                write(result_path, result)
                done.add(request.name)
                print("CPU_TASK_RESULT", json.dumps(result), flush=True)
                if result.get("cleanup_error"):
                    raise RuntimeError(
                        "Owned CPU task cleanup failed; controller stopped"
                    )
        if budget_exhausted:
            break
        time.sleep(5)
    write(
        root / "FINISHED.json",
        dict(
            time=time.time(),
            processed=len(done),
            budget_exhausted=budget_exhausted,
            pending_tasks=[
                p.name for p in sorted(queue.glob("*.json")) if p.name not in done
            ],
        ),
    )
    print("CPU_CONTROLLER_FINISHED", flush=True)


if __name__ == "__main__":
    main()
