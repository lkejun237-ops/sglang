"""Snapshot the authorized Shanghai copy without printing private settings."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

SOCKET = "unix:///run/worldkit-audio-docker/docker.sock"
FIELDS = (
    "WHITEBOX_ROLE",
    "WHITEBOX_CLI_FPS",
    "WHITEBOX_MODEL_SOURCE_SAMPLING",
    "WHITEBOX_MODEL_STARTUP_PREPARE",
    "WHITEBOX_BOUND_COMPUTE",
    "WHITEBOX_MANUAL_INFERENCE",
    "COORDINATOR_URL",
    "WHITEBOX_RELEASE_BUNDLE",
)


def run(*args):
    return subprocess.check_output(args, universal_newlines=True).strip()


def private(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    os.chmod(str(path), 0o600)


def snapshot(root):
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    result = {"observed_at": time.time(), "root": str(root), "containers": []}
    for cid in run("docker", "-H", SOCKET, "ps", "-q").split():
        obj = json.loads(run("docker", "-H", SOCKET, "inspect", cid))[0]
        name = obj["Name"].lstrip("/")
        if not name.startswith(("whitebox-sh-copy-", "whitebox-sh-test-")):
            continue
        env = dict(e.split("=", 1) for e in obj["Config"].get("Env", []) if "=" in e)
        role = env.get("WHITEBOX_ROLE")
        if role not in ("room", "gateway"):
            continue
        private(root / (role + "-inspect.private.json"), obj)
        release = json.loads(
            run("docker", "-H", SOCKET, "exec", cid, "cat", "/app/release.json")
        )
        mounts = [
            {k: m.get(k) for k in ("Type", "Source", "Destination", "RW")}
            for m in obj["Mounts"]
        ]
        bundle = env.get("WHITEBOX_RELEASE_BUNDLE")
        if bundle:
            private(
                root / (role + "-bundle.private.json"),
                json.loads(run("docker", "-H", SOCKET, "exec", cid, "cat", bundle)),
            )
        result["containers"].append(
            dict(
                id=obj["Id"],
                name=name,
                role=role,
                image_id=obj["Image"],
                image=obj["Config"]["Image"],
                started=obj["State"]["StartedAt"],
                release=release,
                env={k: env[k] for k in FIELDS if k in env},
                mounts=mounts,
            )
        )
        if role == "room":
            req = urllib.request.Request(
                "http://127.0.0.1:8092/internal/worlds/runtime",
                data=b'{"operation":"status"}',
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + env["WHITEBOX_INTERNAL_TOKEN"],
                },
            )
            value = json.load(urllib.request.urlopen(req, timeout=25))
            private(root / "runtime.private.json", value)
            result["runtime"] = {
                key: item
                if isinstance(item, (int, float, bool)) or item is None
                else {"count": len(item)}
                if isinstance(item, (list, dict))
                else "<text>"
                for key, item in value.items()
            }
    configs = []
    for path in Path("/etc/nginx").rglob("*.conf"):
        if (
            path.is_file()
            and b"shcopy.world-test-platform.loopit.com.cn" in path.read_bytes()
        ):
            data = path.read_bytes()
            saved = root / (path.name + ".private.txt")
            saved.write_bytes(data)
            os.chmod(str(saved), 0o600)
            configs.append(
                dict(
                    path=str(path),
                    sha256=hashlib.sha256(data).hexdigest(),
                    routes=[
                        line.strip()
                        for line in data.decode().splitlines()
                        if "location " in line
                    ],
                )
            )
    result["nginx"] = configs
    result["free_bytes"] = shutil.disk_usage("/var/lib/whitebox-shanghai-copy").free
    private(root / "summary.private.json", result)
    print(json.dumps(result))


def builder():
    result = {
        "free_bytes": shutil.disk_usage("/opt").free,
        "mounts": run("findmnt", "-n", "-o", "TARGET,SOURCE,FSTYPE"),
    }
    root = Path("/opt/whitebox-adaptive-test-20261109")
    result["previous_own_build_files"] = (
        run("du", "-x", "-h", "--max-depth=2", str(root)) if root.exists() else ""
    )
    for directory in (Path("/opt"), Path("/var/lib")):
        result[str(directory)] = [
            p.name for p in directory.iterdir() if "whitebox" in p.name
        ]
    result["memory"] = Path("/proc/meminfo").read_text().splitlines()[:3]
    result["own_export_markers"] = []
    for parent in (
        Path("/opt/whitebox-adaptive-test-20261109"),
        Path("/opt/whitebox-unified-progressive-20261111"),
    ):
        for context in parent.glob("*/context"):
            result["own_export_markers"].append(
                {
                    "path": str(context),
                    "markers": [p.name for p in context.glob("*.json")],
                }
            )
    result["containers"] = json.loads(run("podman", "ps", "-a", "--format", "json"))
    # Commands and full environments are intentionally omitted.
    result["containers"] = [
        {k: x.get(k) for k in ("Id", "Names", "Image", "State")}
        for x in result["containers"]
    ]
    print(json.dumps(result))


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("snapshot", "builder"))
    p.add_argument("--root", type=Path)
    args = p.parse_args()
    if args.mode == "snapshot":
        if args.root is None or not str(args.root).startswith(
            "/var/lib/whitebox-shanghai-copy/managed-parallel-startup-20261113/"
        ):
            raise ValueError("Expected a private Shanghai-copy stage")
        snapshot(args.root)
    else:
        builder()
