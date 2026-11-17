#!/bin/sh
# Export the pinned SDK, adding only the recorded Atlas fixture used by Room checks.
set -eu
exec python3 - "$1" "$2" <<'PY'
import os
import subprocess
import sys

repo, revision = sys.argv[1:]
git = ["git", "-C", repo]


def entries(tree):
    names = subprocess.check_output([*git, "ls-tree", "-z", "--name-only", tree])
    return [os.fsdecode(name) for name in names.split(b"\0") if name]

root = entries(revision)
paths = [name for name in root if name != "asset-library"]
if "asset-library" in root:
    paths.extend(
        "asset-library/" + name
        for name in entries(revision + ":asset-library")
        if name not in {"subjects", "tests"}
    )
fixture = "asset-library/tests/fixtures/atlas-recording"
if subprocess.check_output([*git, "ls-tree", revision, "--", fixture]):
    paths.append(fixture)
os.execvp("git", [*git, "--literal-pathspecs", "archive", "--format=tar", revision, "--", *paths])
PY
