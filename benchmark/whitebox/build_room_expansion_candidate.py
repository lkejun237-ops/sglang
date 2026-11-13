"""Full immutable-source CPU/Room test images; never deploys or edits containers.

Input is a clean exported Git context with a per-file SHA256 manifest. Production
builds continue to use deploy/whitebox/release.py and its main ancestry checks.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--engine", choices=("docker", "podman"), default="podman")
    p.add_argument("--tag-prefix", required=True)
    p.add_argument("--component", choices=("cpu", "room", "both"), default="both")
    p.add_argument("--browser-mirror")
    a = p.parse_args()
    if a.browser_mirror:
        mirror = urlsplit(a.browser_mirror)
        if not (
            mirror.scheme == "http"
            and mirror.hostname == "127.0.0.1"
            and mirror.port
            and not (
                mirror.username or mirror.password or mirror.query or mirror.fragment
            )
            and mirror.path in ("", "/")
        ):
            raise ValueError(
                "Use the verified loopback browser cache without credentials"
            )
    root = a.context.resolve()
    manifest = json.loads((root / "candidate-source.json").read_text())
    for key in ("source", "sdk"):
        if not re.fullmatch("[a-f0-9]{40}", manifest[key]):
            raise ValueError("Exact source identities required")
    if not re.fullmatch(r"codex/[A-Za-z0-9._/-]+", manifest["branch"]):
        raise ValueError("Use an isolated test branch")
    for name, digest in manifest["files"].items():
        path = root / name
        if path.is_symlink():
            data = os.readlink(path).encode()
        else:
            data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Export content changed: " + name)
    if (root / "benchmark/whitebox/sdk-revision.txt").read_text().strip() != manifest[
        "sdk"
    ]:
        raise ValueError("The exported SDK does not match the source pin")
    if shutil.disk_usage(root).free < 21 * 1024**3:
        raise ValueError("At least 21 GiB of free build space is required")
    sys.path.insert(0, str(root / "deploy/whitebox"))
    from image_storage import inspect_local

    def engine_run(argv, **kwargs):
        args = [a.engine, *argv[1:]]
        if a.engine == "podman" and args[1:3] == ["image", "save"]:
            args[3:3] = ["--format", "docker-archive"]
        output = subprocess.check_output(args, **kwargs)
        if a.engine == "podman" and args[1:3] == ["image", "inspect"]:
            # Podman returns the same config digest without Docker's sha256:
            # prefix. The archive gate still verifies the actual config bytes.
            records = json.loads(output)
            for record in records:
                if re.fullmatch(r"[a-f0-9]{64}", record.get("Id", "")):
                    record["Id"] = "sha256:" + record["Id"]
            return json.dumps(records)
        return output

    components = ("cpu", "room") if a.component == "both" else (a.component,)
    results = []
    for component in components:
        tag = a.tag_prefix + "-" + component
        command = [
            a.engine,
            "build",
            "--file",
            "docker/whitebox.Dockerfile",
            "--target",
            "release" if component == "cpu" else "room",
            "--tag",
            tag,
            "--build-arg",
            "WHITEBOX_SOURCE_SHA=" + manifest["source"],
            "--build-arg",
            "WHITEBOX_SOURCE_BRANCH=" + manifest["branch"],
            "--build-arg",
            "WHITEBOX_SDK_SHA=" + manifest["sdk"],
            "--build-arg",
            "WHITEBOX_TEST_ALLOW_DEVELOPMENT_RELEASE=1",
            "--build-arg",
            "PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple",
            "--build-arg",
            "GOPROXY=https://goproxy.cn,direct",
            "--label",
            "org.opencontainers.image.revision=" + manifest["source"],
            "--label",
            "com.seedleap.worldkit.revision=" + manifest["sdk"],
        ]
        if a.engine == "podman":
            command += [
                "--layers",
                "--jobs",
                "2",
                "--cpu-period",
                "100000",
                "--cpu-quota",
                "400000",
                "--format",
                "docker",
            ]
        if a.browser_mirror:
            command += [
                "--network",
                "host",
                "--build-arg",
                "PLAYWRIGHT_DOWNLOAD_HOST=" + a.browser_mirror.rstrip("/"),
            ]
        subprocess.run(command + ["."], cwd=root, check=True)
        result = dict(
            component=component,
            tag=tag,
            source=manifest["source"],
            sdk=manifest["sdk"],
            branch=manifest["branch"],
            storage=inspect_local(tag, component, run=engine_run),
        )
        results.append(result)
        (root / "candidate-build-results.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
