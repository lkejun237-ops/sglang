# SPDX-License-Identifier: Apache-2.0
"""Verify archive contents against independently supplied Git commit objects."""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def verify_source(root, source_sha, manifest_path=None):
    root = Path(root).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("Source SHA must be a full Git commit SHA")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    try:
        if git("rev-parse", "--show-toplevel").decode().strip() != str(root):
            raise ValueError("Source root must contain its own Git repository")
        if git("cat-file", "-t", source_sha).strip() != b"commit":
            raise ValueError("Source SHA must identify a commit")
        entries = git("ls-tree", "-r", "-z", source_sha).split(b"\0")
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "Frozen commit objects are required, including on archive workers"
        ) from exc
    tracked = {}
    for entry in filter(None, entries):
        meta, name = entry.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        if kind != "blob" or mode not in ("100644", "100755", "120000"):
            raise ValueError("Unsupported tracked source entry")
        tracked[os.fsdecode(name)] = (mode, oid)
    manifest = None
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest.get("source_sha") != source_sha:
            raise ValueError("Source manifest commit differs from the frozen commit")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != set(tracked):
            raise ValueError(
                "Source manifest must contain exactly the commit's tracked paths"
            )
    for name, (mode, oid) in tracked.items():
        path = root / name
        if path.parent.resolve() != path.parent or path.is_symlink() != (
            mode == "120000"
        ):
            raise ValueError(f"Frozen source file type differs: {name}")
        try:
            data = (
                os.fsencode(os.readlink(path))
                if mode == "120000"
                else path.read_bytes()
            )
        except OSError as exc:
            raise ValueError(f"Frozen source file missing: {name}") from exc
        blob = b"blob " + str(len(data)).encode() + b"\0" + data
        if hashlib.sha1(blob).hexdigest() != oid:
            raise ValueError(f"Frozen source file differs: {name}")
        if manifest and hashlib.sha256(data).hexdigest() != manifest["files"][name]:
            raise ValueError(f"Frozen source manifest hash differs: {name}")
