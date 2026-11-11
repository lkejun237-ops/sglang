"""Immutable git archive preparation and verified container-local extraction (stdlib)."""

import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

OSS_MOUNT = Path("/mnt/oss")
SOURCE_CACHE = Path("/var/lib/sglang-orchestrate/cache")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha(value, length=64):
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError("Expected a full lowercase hexadecimal digest")
    return value


def write_json(path, value):
    with open(path, "x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def validate_manifest(manifest):
    if manifest.get("schema") != 1 or set(manifest) != {
        "schema",
        "commit",
        "sha256",
        "uri",
        "tools",
    }:
        raise ValueError("Unsupported source manifest schema")
    sha(manifest["commit"], 40)
    sha(manifest["sha256"])
    suffix = "/" + manifest["commit"] + "/" + manifest["sha256"] + "/source.tar"
    if not re.fullmatch(
        r"oss://[a-z0-9-]+/[A-Za-z0-9_./-]+", manifest["uri"]
    ) or not manifest["uri"].endswith(suffix):
        raise ValueError("Source URI must be content-addressed")


def prepare(repo, commit, output, oss_prefix):
    sha(commit, 40)
    if not re.fullmatch(r"oss://[a-z0-9-]+/[A-Za-z0-9_./-]+", oss_prefix):
        raise ValueError("Expected an OSS prefix without credentials or query strings")
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", commit + "^{commit}"], text=True
    ).strip()
    if actual != commit:
        raise ValueError("Commit does not resolve exactly")
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "source.tar"
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=tar",
            "--output=" + str(archive.resolve()),
            commit,
        ],
        check=True,
    )
    checksum = digest(archive)
    manifest = {
        "schema": 1,
        "commit": commit,
        "sha256": checksum,
        "uri": oss_prefix.rstrip("/") + "/" + commit + "/" + checksum + "/source.tar",
    }
    with tarfile.open(archive) as tar:
        manifest["tools"] = {}
        for name in ("launch.py", "source.py", "select_gpu.py", "orchestrate.py"):
            member = "deploy/sglang/" + name
            try:
                stream = tar.extractfile(member)
            except KeyError:
                continue
            manifest["tools"][name] = hashlib.sha256(stream.read()).hexdigest()
    write_json(output / "source.json", manifest)
    return manifest


def unpack(archive, expected, destination):
    if digest(archive) != sha(expected):
        raise ValueError("Source archive SHA256 mismatch")
    # New, private directories only: never overlay a previous release or image source.
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            links = {m.name for m in members if m.issym() or m.islnk()}
            for m in members:
                p = PurePosixPath(m.name)
                if (
                    p.is_absolute()
                    or ".." in p.parts
                    or not (m.isfile() or m.isdir() or m.issym())
                ):
                    raise ValueError("Unsafe archive member")
                if any(str(parent) in links for parent in p.parents):
                    raise ValueError("Archive member traverses a symlink")
                if m.issym():
                    target = (destination / m.name).parent / m.linkname
                    if not target.resolve().is_relative_to(destination.resolve()):
                        raise ValueError("Archive symlink escapes release")
            tar.extractall(destination)
    except BaseException:
        import shutil

        shutil.rmtree(destination)
        raise


def fetch(manifest, destination):
    validate_manifest(manifest)
    uri = manifest["uri"]
    if not re.fullmatch(r"oss://[a-z0-9-]+/[A-Za-z0-9_./-]+", uri):
        raise ValueError("Invalid OSS URI")
    key = uri.split("/", 3)[3]
    if PurePosixPath(key).is_absolute() or ".." in PurePosixPath(key).parts:
        raise ValueError("OSS key cannot traverse local mount paths")
    candidates = (OSS_MOUNT / key, SOURCE_CACHE / manifest["sha256"] / "source.tar")
    local = next((path for path in candidates if path.is_file()), None)
    if local is not None:
        # A present but corrupt candidate fails closed; never hide it with a download.
        unpack(local, manifest["sha256"], destination)
    else:
        with tempfile.TemporaryDirectory(prefix="sglang-download-") as tmp:
            archive = Path(tmp) / "source.tar"
            subprocess.run(
                ["ossutil", "cp", uri, str(archive)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            unpack(archive, manifest["sha256"], destination)
    write_json(
        destination.parent / "verified-source.json",
        {"commit": manifest["commit"], "archive_sha256": manifest["sha256"]},
    )


def bootstrap():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--python", default="/opt/sglang/bin/python")
    args = p.parse_args()
    manifest = json.loads(args.manifest)
    root = Path(tempfile.mkdtemp(prefix="sglang-release-")) / "source"
    fetch(manifest, root)
    config = root.parent / "role.json"
    write_json(config, json.loads(args.config))
    # Execute only code from the verified archive. Environment is filtered by launcher.
    os.execv(
        args.python,
        [
            args.python,
            str(root / "deploy/sglang/launch.py"),
            "run",
            "--config",
            str(config),
            "--source",
            str(root),
            "--commit",
            manifest["commit"],
            "--archive-sha256",
            manifest["sha256"],
        ],
    )


if __name__ == "__main__":
    bootstrap()
