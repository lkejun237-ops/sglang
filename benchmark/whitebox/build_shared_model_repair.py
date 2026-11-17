"""Rebuild the shared test model using its verified, cached native build recipe.

Only release identity and full source change. No application overlay, GPU
allocation, dependency upgrade, container replacement or image cleanup occurs.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

OLD = "06d04b805c96ab0d3d5dbc6035659902f9c4aabc"
BRANCH = "codex/managed-parallel-startup-ttff-prefix"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    assert re.fullmatch("[a-f0-9]{40}", args.source)
    context = args.context.resolve()
    previous = Path("/var/tmp/cachetiers-integration-" + OLD[:12])
    dockerfile = "docker/whitebox-model.Dockerfile"
    assert (context / dockerfile).read_text() == (
        previous / "context" / dockerfile
    ).read_text(), "Dependency recipe changed; re-estimate the build before running"
    command = json.loads((previous / "model-build-command.json").read_text())
    old_tag = command[command.index("--tag") + 1]
    subprocess.run(["podman", "image", "exists", old_tag], check=True)
    # Reuse the exact dependency layers. The original application COPY is ~90 MB;
    # allow 1 GiB of new layers twice plus the standard 15 GiB host reserve.
    assert shutil.disk_usage("/var/lib/containers").free >= 17 * 1024**3
    assert (
        sum(
            p.stat().st_size
            for p in (context / "python/sglang").rglob("*")
            if p.is_file()
        )
        < 512 * 1024**2
    )
    tag = old_tag.split(":cachetiers-", 1)[0] + ":ttff-repair-" + args.source[:12]
    command = [
        tag
        if value == old_tag
        else value.replace(OLD, args.source).replace(
            "codex/mooncake-integration-test-20261113", BRANCH
        )
        for value in command
    ]
    assert command[-1] == "." and command[0:2] == ["podman", "build"]
    output = context.parent
    (output / "build-command.json").write_text(json.dumps(command, indent=2))
    with (output / "build.log").open("w") as log:
        subprocess.run(
            command, cwd=context, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    record = json.loads(subprocess.check_output(["podman", "image", "inspect", tag]))[0]
    assert (
        record["Config"]["Labels"]["org.opencontainers.image.revision"] == args.source
    )
    (output / "build-result.json").write_text(
        json.dumps(
            dict(
                source=args.source,
                tag=tag,
                image_id=record["Id"],
                dockerfile_sha256=hashlib.sha256(
                    (context / dockerfile).read_bytes()
                ).hexdigest(),
            ),
            indent=2,
        )
    )
    print("BUILT", args.source, tag, flush=True)


if __name__ == "__main__":
    main()
