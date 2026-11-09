"""Verify Room init reaping, child exits and SIGTERM in isolated CPU containers."""

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path


def docker(*args, timeout=30):
    return subprocess.run(
        ["docker", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=timeout,
    )


def check(image, probe, mode):
    name = "whitebox-init-check-" + uuid.uuid4().hex
    result = {"mode": mode, "status": "failed"}
    args = [
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=128m",
        "--mount",
        f"type=bind,src={probe},dst=/app/run_service.py,readonly",
    ]
    try:
        if mode != "signal":
            process = docker("run", *args, image, mode)
            result.update(
                exitCode=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
            )
            assert process.returncode == (42 if mode == "exit" else 0), result
        else:
            created = docker("run", "--detach", *args, image, mode)
            assert created.returncode == 0, created.stderr
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                logs = docker("logs", name)
                if "room-init-signal-ready" in logs.stdout:
                    break
                time.sleep(0.05)
            else:
                raise AssertionError("Room process did not become signal-ready")
            sent = docker("kill", "--signal=TERM", name)
            assert sent.returncode == 0, sent.stderr
            exited = docker("wait", name)
            logs = docker("logs", name)
            result.update(
                exitCode=exited.stdout.strip(), stdout=logs.stdout, stderr=logs.stderr
            )
            assert exited.returncode == 0 and exited.stdout.strip() == "23", result
            assert "room-init-signal-received" in logs.stdout, result
        result["status"] = "passed"
    except (AssertionError, subprocess.TimeoutExpired) as error:
        result["error"] = str(error)
    finally:
        cleanup = docker("rm", "--force", name)
        result["ownedContainerRemoved"] = cleanup.returncode == 0
        if cleanup.returncode != 0:
            result["status"] = "failed"
            result["cleanupError"] = cleanup.stderr
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.trials > 0
    inspected = docker("image", "inspect", args.image)
    assert inspected.returncode == 0, inspected.stderr
    image = json.loads(inspected.stdout)[0]
    probe = Path(__file__).resolve().parents[3] / "deploy/whitebox/room_init_probe.py"
    receipt = {
        "image": image["Id"],
        "entrypoint": image["Config"]["Entrypoint"],
        "results": [],
        "status": "failed",
        "network": "none",
        "gpu": False,
    }
    for trial in range(1, args.trials + 1):
        for mode in ("reap", "exit", "signal"):
            result = check(image["Id"], probe, mode)
            receipt["results"].append({"trial": trial, **result})
    if all(row["status"] == "passed" for row in receipt["results"]):
        receipt["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
