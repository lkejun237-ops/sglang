#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect real stopped-container evidence for one reviewed original create.

Runs on the Docker lifecycle owner's host (Python 3.6+). Never starts or stops
a container. The original coordinator performs a read-only validation unless
the operator explicitly supplies --apply. No credentials appear in arguments.
"""

import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import threading
from pathlib import Path, PurePosixPath


def bounded_command(args, limit):
    child = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if args[1] == "logs" else subprocess.DEVNULL,
    )
    deadline = threading.Timer(30, child.kill)
    deadline.start()
    try:
        data = child.stdout.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Docker evidence exceeds its bound")
        if child.wait() != 0:
            raise ValueError("Docker evidence is unavailable")
        return data
    finally:
        deadline.cancel()
        if child.poll() is None:
            child.kill()
        child.wait()
        child.stdout.close()


def read_file(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("A regular original evidence file is required")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if not data or len(data) > limit:
        raise ValueError("Original evidence is empty or oversized")
    return data


def encoded(data):
    return base64.b64encode(data).decode("ascii")


def collect_observations(audit_path, approved_sha256, references, run=bounded_command):
    raw = read_file(audit_path, 262144)
    if hashlib.sha256(raw).hexdigest() != approved_sha256:
        raise ValueError("The reviewed retirement evidence has changed")
    audit = json.loads(raw)
    container_id = audit["container_id"]
    if not re.fullmatch(r"[a-f0-9]{64}", container_id):
        raise ValueError("An exact original container ID is required")
    observed = json.loads(run(["docker", "inspect", container_id], 2 * 1024 * 1024))[0]
    if (
        observed["Id"] != container_id
        or observed["State"]["Running"] is not False
        or observed["State"]["Pid"] != 0
        or observed["State"]["Status"] != "exited"
    ):
        raise ValueError("The original container retirement is not confirmed")
    container = {
        "Id": observed["Id"],
        "Image": observed["Image"],
        "State": {
            key: observed["State"][key]
            for key in ("Status", "Running", "Pid", "StartedAt", "FinishedAt")
        },
        "HostConfig": {"RestartPolicy": observed["HostConfig"]["RestartPolicy"]},
    }
    sources = {}
    for name, path in audit["startup_boundary"]["source_paths"].items():
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or "\0" in path
            or ".." in PurePosixPath(path).parts
            or PurePosixPath(path).suffix not in {".py", ".ts", ".js", ".mjs"}
        ):
            raise ValueError("An exact reviewed source-file path is required")
        archive = run(["docker", "cp", container_id + ":" + path, "-"], 3 * 1024 * 1024)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            members = tar.getmembers()
            if (
                len(members) != 1
                or not members[0].isfile()
                or members[0].size > 2 * 1024 * 1024
            ):
                raise ValueError("The original source file is unavailable")
            sources[name] = encoded(tar.extractfile(members[0]).read())
    log = run(
        [
            "docker",
            "logs",
            "--timestamps",
            "--since",
            container["State"]["StartedAt"],
            "--until",
            container["State"]["FinishedAt"],
            "--tail",
            "20000",
            container_id,
        ],
        2 * 1024 * 1024,
    )
    marker = (
        "CLI startup failed trace_id="
        + audit["operation_id"]
        + " code="
        + audit["startup_boundary"]["error_code"]
    )
    # Forward only this original operation's diagnostic, never other sessions' logs.
    lines = [line for line in log.splitlines() if marker.encode() in line]
    return {
        "audit": encoded(raw),
        "container": container,
        "startup_log": encoded(b"\n".join(lines) + b"\n"),
        "sources": sources,
        "references": {
            name: encoded(read_file(path, 2 * 1024 * 1024))
            for name, path in references.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--approved-audit-sha256", required=True)
    parser.add_argument("--epoch-evidence", required=True)
    parser.add_argument("--stop-evidence", required=True)
    parser.add_argument("--source-review", required=True)
    parser.add_argument("--coordinator-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{64}", args.coordinator_id):
        raise ValueError("An exact coordinator container ID is required")
    coordinator = json.loads(
        bounded_command(["docker", "inspect", args.coordinator_id], 2 * 1024 * 1024)
    )[0]
    if (
        coordinator["Id"] != args.coordinator_id
        or coordinator["State"]["Running"] is not True
    ):
        raise ValueError("The original coordinator is unavailable")
    payload = collect_observations(
        args.audit,
        args.approved_audit_sha256,
        {
            "epoch": args.epoch_evidence,
            "stop": args.stop_evidence,
            "source_review": args.source_review,
        },
    )
    data = json.dumps(payload).encode()
    if len(data) > 12 * 1024 * 1024:
        raise ValueError("Retirement observations exceed their bound")
    command = [
        "docker",
        "exec",
        "-i",
        args.coordinator_id,
        "python",
        "-B",
        "-m",
        "sglang.multimodal_gen.runtime.realtime.whitebox.owner_retirement_maintenance",
        "--approved-audit-sha256",
        args.approved_audit_sha256,
    ]
    if args.apply:
        command.append("--apply")
    result = subprocess.run(
        command,
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    if result.returncode or len(result.stdout) > 65536:
        raise ValueError("The original retirement outcome remains unconfirmed")
    receipt = json.loads(result.stdout)
    if not isinstance(receipt, dict) or receipt.get("validated") is not True:
        raise ValueError("The original retirement outcome remains unconfirmed")
    print(json.dumps(receipt))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            json.dumps(
                {
                    "error": "OWNER_RETIREMENT_NOT_CONFIRMED",
                    "errorType": type(error).__name__,
                }
            )
        )
        sys.exit(1)
