# SPDX-License-Identifier: Apache-2.0
"""Retain one terminal owned experiment, including partial data; never remove sources."""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_parts(value, store, verified):
    identity = tuple(
        (part["path"], part["offset"], part["bytes"], part["sha256"])
        for part in value["parts"]
    )
    expected = (value["bytes"], value["sha256"])
    if identity in verified:
        if verified[identity] != expected:
            raise ValueError("Conflicting manifest for identical exact archive paths")
        return
    whole, offset = hashlib.sha256(), 0
    for part in value["parts"]:
        path = Path(part["path"])
        if (
            not path.is_absolute()
            or ".." in path.parts
            or path.is_symlink()
            or path.resolve() != path
            or not path.resolve().is_relative_to(store)
            or type(part["bytes"]) is not int
            or part["bytes"] < 1
            or part["offset"] != offset
        ):
            raise ValueError(
                "RGB archive parts must bind exact contiguous mounted-store paths"
            )
        digest, count = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
                whole.update(block)
                count += len(block)
        if count != part["bytes"] or digest.hexdigest() != part["sha256"]:
            raise ValueError("Existing RGB archive part checksum mismatch")
        offset += count
    if offset != value["bytes"] or whole.hexdigest() != value["sha256"]:
        raise ValueError("Existing RGB archive whole-file checksum mismatch")
    verified[identity] = expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--harness-source", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--owner-user-id", required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    store = args.store_root.resolve()
    if not output.is_relative_to(store):
        raise ValueError(
            "Archive output must remain within the declared artifact store"
        )
    if root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError("Archive output and original experiment must be disjoint")
    proof = json.loads(args.proof.read_text())
    writer = proof["roots"][root.name]
    if (
        writer["status"] not in ["Succeeded", "Failed", "Stopped"]
        or str(writer["user_id"]) != args.owner_user_id
        or writer["root"] != str(root)
        or proof["scope"]
        != "Exact PAI job terminal states checked before bulk retention"
    ):
        raise ValueError(
            "Exact owned experiment writer must be terminal before retention"
        )
    sys.path.insert(0, str(args.harness_source / "benchmark/zing_h3_causal"))
    from frozen_source import verify_source

    verify_source(args.harness_source, args.source_revision)
    from archive_stall_artifacts import archive_to_directory

    output.mkdir(parents=True, exist_ok=True)
    if (output / "INDEX.json").exists():
        raise ValueError("Retention index already exists; use a new immutable output")
    skip = {
        "source",
        "runtime-source",
        "baseline-source",
        "harness-source",
        "deps",
        "model",
        "lynn",
        "media",
        ".git",
        "torch-cache",
        "cache",
        "inductor-cache",
        "triton-cache",
    }
    suffixes = {
        ".pt",
        ".nsys-rep",
        ".sqlite",
        ".prof",
        ".json",
        ".jsonl",
        ".csv",
        ".log",
        ".txt",
        ".py",
        ".sh",
        ".xml",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".mp4",
        ".wav",
        ".npy",
        ".npz",
        ".html",
        ".js",
        ".css",
        ".partial",
        # Retain explicitly copied compiler diagnostics; cache directories
        # themselves remain excluded above.
        ".ttir",
        ".ttgir",
        ".llir",
        ".ptx",
        ".cubin",
    }
    files, rgb = [], set()
    for current, directories, names in os.walk(root, followlinks=False):
        selected_sources = "generated-sources" in Path(current).relative_to(root).parts
        directories[:] = [
            name
            for name in directories
            if (
                name not in skip
                or selected_sources
                and name in {"inductor-cache", "triton-cache"}
            )
            and "secret" not in name.lower()
            and "private" not in name.lower()
            and not (Path(current) / name).is_symlink()
        ]
        for name in names:
            path = Path(current) / name
            if (
                path.is_symlink()
                or "secret" in name.lower()
                or "private" in name.lower()
            ):
                continue
            if path.suffix == ".rgb":
                rgb.add(path)
            elif path.suffix in suffixes or name in [
                "COMPLETE",
                "PLANNED_RANK_FAILURE",
            ]:
                files.append(path)
                # Normal benchmark retention may already have removed the local
                # RGB after verified upload. Its sidecar still binds real raw
                # objects and must participate in the final independent read.
                if name.endswith(".rgb.archive.json"):
                    rgb.add(path.with_name(name[: -len(".archive.json")]))
    record = dict(
        root=str(root),
        writer=writer,
        proof_sha256=sha(args.proof),
        started=time.time(),
        source_removed=False,
        rgb=[],
        scope="Original experiment traces/tensors/media/metadata, including failures and partial files. Model/dependency/compiler caches, symlinked input assets and secrets excluded; their frozen identities remain in run controls.",
    )
    verified = {}
    for path in sorted(rgb):
        manifest = path.with_name(path.name + ".archive.json")
        if manifest.exists():
            value = json.loads(manifest.read_text())
            if not value["verified"] or value["path"] != str(path):
                raise ValueError("Existing RGB archive does not bind this exact source")
            verify_parts(value, store, verified)
            absent = not path.exists()
            leftover = not absent and path.stat().st_size == 0 and value["bytes"] > 0
            if (
                not absent
                and not leftover
                and (
                    path.stat().st_size != value["bytes"]
                    or sha(path) != value["sha256"]
                )
            ):
                raise ValueError("Archived RGB source changed")
        else:
            absent = leftover = False
            value = archive_to_directory(path, output / "rgb", remove_verified=False)
            files.append(manifest)
        record["rgb"].append(
            dict(
                path=str(path),
                manifest=value,
                zero_length_leftover=leftover,
                source_already_absent=absent,
            )
        )
    target = output / "originals.tar"
    members = []
    # The complete source bytes remain untouched, including on any failure.
    with target.open("xb") as stream, tarfile.open(fileobj=stream, mode="w") as archive:
        for path in sorted(set(files)):
            before = path.stat()
            digest = sha(path)
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError(
                    "A source writer is still modifying the terminal experiment"
                )
            members.append(
                dict(
                    path=str(path.relative_to(root)),
                    bytes=before.st_size,
                    sha256=digest,
                )
            )
    (output / "MEMBERS.json").write_text(json.dumps(members, indent=2))
    record["bundle"] = archive_to_directory(
        target, output / "bundle", remove_verified=False
    )
    record["members_sha256"] = sha(output / "MEMBERS.json")
    record["existing_rgb_exact_composites_verified"] = len(verified)
    record["completed"] = time.time()
    record["independent_verification_required"] = True
    with (output / "INDEX.json").open("x") as stream:
        json.dump(record, stream, indent=2)
    print("OWNED_STALL_RAW_RETAINED", root.name, len(members), len(rgb))


if __name__ == "__main__":
    main()
