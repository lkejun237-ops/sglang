# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import shutil
import subprocess

import pytest
from frozen_source import verify_source


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()

    git("init", "-q")
    files = {"progressive.py": b"encoder\n", "validator.py": b"validator\n"}
    for name, data in files.items():
        (root / name).write_bytes(data)
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "source",
    )
    sha = git("rev-parse", "HEAD")
    manifest = {
        "source_sha": sha,
        "files": {n: hashlib.sha256(d).hexdigest() for n, d in files.items()},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return root, sha, path, manifest


def test_complete(source):
    root, sha, path, _ = source
    verify_source(root, sha, path)
    verify_source(root, sha)


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "omitted",
        "extra",
        "modified",
        "missing",
        "forged_hash",
        "wrong_sha",
        "no_git",
    ],
)
def test_rejects_incomplete_or_modified_source(source, change):
    root, sha, path, manifest = source
    if change == "empty":
        manifest["files"] = {}
    elif change == "omitted":
        del manifest["files"]["progressive.py"]
        (root / "progressive.py").write_text("modified")
    elif change == "extra":
        manifest["files"]["extra.py"] = hashlib.sha256(b"").hexdigest()
        (root / "extra.py").write_bytes(b"")
    elif change in ("modified", "forged_hash"):
        (root / "validator.py").write_bytes(b"modified")
        if change == "forged_hash":
            manifest["files"]["validator.py"] = hashlib.sha256(b"modified").hexdigest()
    elif change == "missing":
        (root / "progressive.py").unlink()
    elif change == "wrong_sha":
        manifest["source_sha"] = "0" * 40
    elif change == "no_git":
        shutil.rmtree(root / ".git")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_source(root, sha, path)
