"""Exercise cleanup deadlines in disposable CPU-only Linux PID namespaces."""

import argparse
import base64
import json
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "python/sglang/multimodal_gen/runtime/realtime"
HARNESS = """
import base64, json, os, sys, time, types
sources = SOURCES
def load(name):
    module = types.ModuleType(name)
    exec(compile(base64.b64decode(sources[name]), name + '.py', 'exec'), module.__dict__)
    return module
watchdog = load('cleanup_watchdog')
mode = sys.argv[1]
def worker():
    print(json.dumps({'worker_pid': os.getpid()}), flush=True)
    try:
        with watchdog.CleanupWatchdog(0.5):
            if mode == 'hang':
                time.sleep(10)
            if mode == 'failure':
                raise RuntimeError('injected cleanup failure')
    except RuntimeError:
        # Catching a cleanup exception must not disarm its deadline.
        time.sleep(10)
if mode == 'guard':
    worker()
else:
    load('worker_process').run_worker(worker)
"""


def check(image):
    present = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=15,
    )
    if present.returncode:
        subprocess.run(["docker", "pull", image], check=True, timeout=180)
    sources = {
        name: base64.b64encode((RUNTIME / (name + ".py")).read_bytes()).decode()
        for name in ("cleanup_watchdog", "worker_process")
    }
    harness = HARNESS.replace("SOURCES", repr(sources))
    for mode, expected in (
        ("success", 0),
        ("hang", 137),
        ("failure", 137),
        ("guard", 70),
    ):
        name = "zing-pid1-" + uuid.uuid4().hex
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--cpus",
            "0.25",
            "--memory",
            "96m",
            "--pids-limit",
            "24",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--entrypoint",
            "python",
            image,
            "-u",
            "-c",
            harness,
            mode,
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            print(
                json.dumps(
                    dict(mode=mode, returncode=result.returncode, expected=expected)
                )
            )
            if result.returncode != expected:
                raise AssertionError(result.stdout + result.stderr)
            lines = result.stdout.strip().splitlines()
            assert len(lines) == 1, "Worker unexpectedly restarted"
            pid = json.loads(lines[0])["worker_pid"]
            assert (pid == 1) == (mode == "guard"), "Wrong PID namespace boundary"
        finally:
            # Only the uniquely named container created by this test is eligible.
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="python:3.12-slim")
    check(parser.parse_args().image)
