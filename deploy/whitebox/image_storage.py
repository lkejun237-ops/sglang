"""Reject CPU images which retain hidden application/SDK layers.

Inspect history as well as the visible filesystem: deleting a directory in a
new layer doesn't reclaim the original COPY. This is a build contract, not a
replacement for source digest verification or the native image tests.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

if __package__:
    from .docker_storage import GIB, check_headroom
else:
    from docker_storage import GIB, check_headroom


def validate_history(config, component):
    if component not in {"cpu", "room"}:
        raise ValueError("Storage validation supports CPU and Room images only")
    history = config.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("Image layer history is required")
    sdk, source = [], []
    for entry in history:
        if not isinstance(entry, dict):
            raise ValueError("Invalid image layer history")
        command = entry.get("created_by", "")
        if not isinstance(command, str):
            raise ValueError("Invalid image layer command")
        if entry.get("empty_layer") or not re.search(r"\b(?:COPY|ADD)\b", command):
            continue
        if "/opt/worldkit" in command or ".whitebox-sdk" in command:
            sdk.append(command)
        # Native codec COPY targets are below this directory, but are not a
        # second copy of the complete runtime source tree.
        if re.search(r"(?:^|\s)/app/python/sglang/?(?:\s|$)", command):
            source.append(command)
    expected_sdk = 1 if component == "room" else 0
    if len(sdk) != expected_sdk:
        raise ValueError(
            f"{component} requires {expected_sdk} SDK COPY layer(s), found {len(sdk)}; "
            "do not inherit full Room layers as dependencies, even after rm"
        )
    if len(source) != 1:
        raise ValueError(
            f"Expected one complete runtime source COPY, found {len(source)}"
        )
    return {"component": component, "sdk_copies": len(sdk), "source_copies": 1}


def saved_config(archive_path, identity, platform):
    """Read exact content-addressed history, never infer layers from display sizes."""
    with tarfile.open(archive_path) as archive:

        def read(name, digest=None):
            try:
                member = archive.getmember(name)
            except KeyError as error:
                raise ValueError(
                    "Image configuration identity is unavailable"
                ) from error
            if not member.isfile() or member.size > 16 * 1024**2:
                raise ValueError("Invalid image configuration metadata")
            raw = archive.extractfile(member).read()
            actual = "sha256:" + hashlib.sha256(raw).hexdigest()
            if digest is not None and actual != digest:
                raise ValueError("Image configuration identity changed")
            return json.loads(raw), actual

        def blob(digest):
            if not isinstance(digest, str) or not re.fullmatch(
                r"sha256:[a-f0-9]{64}", digest
            ):
                raise ValueError("Invalid image configuration identity")
            return read("blobs/sha256/" + digest.split(":", 1)[1], digest)

        # Classic Docker IDs identify config JSON; the containerd store may use
        # an OCI index ID. Follow only its matching platform's sealed descriptors.
        if "blobs/sha256/" + identity.split(":", 1)[1] in archive.getnames():
            value, digest = blob(identity)
            for _ in range(4):
                if "rootfs" in value:
                    break
                if "manifests" in value:
                    matching = [
                        row
                        for row in value["manifests"]
                        if row.get("platform", {}).get("os") == platform[0]
                        and row.get("platform", {}).get("architecture") == platform[1]
                    ]
                    if len(matching) != 1:
                        raise ValueError(
                            "Image platform identity is ambiguous or unavailable"
                        )
                    value, digest = blob(matching[0].get("digest"))
                elif "config" in value:
                    value, digest = blob(value["config"].get("digest"))
                else:
                    raise ValueError("Invalid image configuration identity")
            else:
                raise ValueError("Image configuration descriptor depth exceeded")
        else:
            manifest, _ = read("manifest.json")
            if not isinstance(manifest, list) or len(manifest) != 1:
                raise ValueError("Image configuration identity is ambiguous")
            value, digest = read(manifest[0]["Config"], identity)
        if (value.get("os"), value.get("architecture")) != platform:
            raise ValueError("Final image configuration platform identity changed")
        history, layers = value.get("history"), value.get("rootfs", {}).get("diff_ids")
        if (
            not isinstance(history, list)
            or not isinstance(layers, list)
            or sum(not row.get("empty_layer", False) for row in history) != len(layers)
        ):
            raise ValueError("Image history does not match its filesystem layers")
        return value, digest


def inspect_local(image, component, *, run=subprocess.check_output):
    # Resolve the mutable tag once; save only that immutable local image identity.
    data = json.loads(
        run(["docker", "image", "inspect", image], universal_newlines=True)
    )[0]
    identity = data["Id"]
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", identity):
        raise ValueError("Invalid Docker image identity")
    if component not in {"cpu", "room"}:
        raise ValueError("Storage validation supports CPU and Room images only")
    size = data["Size"]
    if type(size) is not int or size < 0:
        raise ValueError("Invalid Docker image size")
    with tempfile.TemporaryDirectory(prefix="whitebox-image-history-") as temporary:
        # Export can live on a different filesystem from DockerRootDir. Check
        # this actual scratch directory immediately before its large write.
        incoming = max(6 if component == "room" else 1, size / GIB)
        try:
            check_headroom(
                incoming,
                read=lambda: {"free_bytes": shutil.disk_usage(temporary).free},
            )
        except ValueError as error:
            raise ValueError("Insufficient image export disk headroom") from error
        path = Path(temporary) / "image.tar"
        run(["docker", "image", "save", "--output", str(path), identity])
        config, digest = saved_config(
            path, identity, (data["Os"], data["Architecture"])
        )
    return {
        **validate_history(config, component),
        "image_id": identity,
        "config_digest": digest,
        "logical_bytes": data["Size"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--component", choices=("cpu", "room"), required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_local(args.image, args.component), indent=2))


if __name__ == "__main__":
    main()
