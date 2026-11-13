"""Reclaim only hash-verified, unreferenced Git exports from this chat's old builds."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOTS = {
    "/opt/whitebox-adaptive-test-20261109/cpu-ccda28a6a3f3/context",
    "/opt/whitebox-adaptive-test-20261109/room-r2-8552a0d4609e/context",
    "/opt/whitebox-adaptive-test-20261109/room-r3-017396b0c737/context",
    "/opt/whitebox-unified-progressive-20261111/model-a7d61072d3b6/context",
    "/opt/whitebox-unified-progressive-20261111/room-a7d61072d3b6/context",
}


def referenced_paths():
    paths = []
    for engine in ("podman", "docker"):
        if shutil.which(engine) is None:
            continue
        ids = subprocess.check_output(
            [engine, "ps", "-aq"], universal_newlines=True
        ).split()
        if ids:
            rows = json.loads(subprocess.check_output([engine, "inspect", *ids]))
            paths += [m.get("Source", "") for r in rows for m in r.get("Mounts", [])]
    for proc in Path("/proc").glob("[0-9]*"):
        for link in [proc / "cwd", *list((proc / "fd").glob("*"))]:
            try:
                paths.append(os.readlink(str(link)))
            except (OSError, PermissionError):
                pass
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    spec = json.loads(a.manifest.read_text())
    references = referenced_paths()
    result = []
    for entry in spec["exports"]:
        root = Path(entry["path"])
        if str(root) not in ROOTS or root.is_symlink() or not root.is_dir():
            raise ValueError("Unexpected or missing export")
        if any(x == str(root) or x.startswith(str(root) + "/") for x in references):
            raise ValueError("Export still referenced")
        expected = entry["files"]
        actual = {
            str(f.relative_to(root)): f
            for f in root.rglob("*")
            if f.is_file() or f.is_symlink()
        }
        # Strict equality protects unknown additions; no generated or user files
        # are discarded just because a directory was originally our build export.
        if set(actual) != set(expected):
            raise ValueError(
                "Export file inventory differs: "
                + json.dumps(
                    {
                        "path": str(root),
                        "missing": sorted(set(expected) - set(actual))[:8],
                        "extra": sorted(set(actual) - set(expected))[:8],
                    }
                )
            )
        total = 0
        for name, path in actual.items():
            data = (
                os.readlink(str(path)).encode()
                if path.is_symlink()
                else path.read_bytes()
            )
            if hashlib.sha256(data).hexdigest() != expected[name]:
                raise ValueError("Export content changed: " + name)
            total += len(data)
        result.append({"path": str(root), "bytes": total, "source": entry["source"]})
    # Recheck references after hashing; deleting these private exports never
    # touches images, containers, configs, logs, model assets or shared caches.
    references = referenced_paths()
    for entry in result:
        if any(
            x == entry["path"] or x.startswith(entry["path"] + "/") for x in references
        ):
            raise ValueError("Export became referenced")
    if a.execute:
        for entry in result:
            shutil.rmtree(entry["path"])
    print(
        json.dumps(
            {
                "execute": a.execute,
                "exports": result,
                "free_bytes": shutil.disk_usage("/opt").free,
            }
        )
    )


if __name__ == "__main__":
    main()
