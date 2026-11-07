# SPDX-License-Identifier: Apache-2.0
"""Archive owned experiment files using scoped PUT/HEAD URLs; never log URLs.

The delivery JSON is private: {slots:[{key,put_url,head_url}], next_slot:0}.
A verified public manifest contains only object keys, byte counts and hashes.
Call between requests; archive IO is excluded from inference timing.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

PART_BYTES = 3 << 30


def archive_file(path, delivery_path, *, remove_verified=False, part_bytes=PART_BYTES):
    path, delivery_path = Path(path), Path(delivery_path)
    if part_bytes < 1 or part_bytes > PART_BYTES:
        raise ValueError("Invalid archive part size")
    manifest = path.with_name(path.name + ".archive.json")
    if manifest.exists():
        raise ValueError("Archive manifest already exists")
    rows, whole = [], hashlib.sha256()
    total = path.stat().st_size
    with path.open("rb") as stream:
        offset = 0
        while offset < total:
            size = min(part_bytes, total - offset)
            digest, md5 = hashlib.sha256(), hashlib.md5()
            remaining = size
            while remaining:
                block = stream.read(min(8 << 20, remaining))
                if not block:
                    raise EOFError("Archive source changed during hashing")
                digest.update(block)
                md5.update(block)
                whole.update(block)
                remaining -= len(block)
            stream.seek(offset)
            delivery = json.loads(delivery_path.read_text())
            index = delivery.get("next_slot", 0)
            slot = delivery["slots"][index]
            delivery["next_slot"] = index + 1
            temporary = delivery_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(delivery))
            temporary.chmod(0o600)
            temporary.replace(delivery_path)

            def chunks():
                remaining = size
                while remaining:
                    block = stream.read(min(8 << 20, remaining))
                    if not block:
                        raise EOFError("Archive source changed during upload")
                    remaining -= len(block)
                    yield block

            request = urllib.request.Request(
                slot["put_url"],
                data=chunks(),
                method="PUT",
                headers={
                    "Content-Length": str(size),
                    "Content-Type": "application/octet-stream",
                },
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                etag = response.headers.get("ETag", "").strip('"').lower()
                if response.status != 200 or etag != md5.hexdigest():
                    raise RuntimeError("Archive upload checksum mismatch")
            with urllib.request.urlopen(
                urllib.request.Request(slot["head_url"], method="HEAD"), timeout=60
            ) as response:
                if (
                    int(response.headers["Content-Length"]) != size
                    or response.headers["ETag"].strip('"').lower() != md5.hexdigest()
                ):
                    raise RuntimeError("Archive readback checksum mismatch")
            rows.append(
                dict(
                    key=slot["key"],
                    offset=offset,
                    bytes=size,
                    sha256=digest.hexdigest(),
                    md5=md5.hexdigest(),
                )
            )
            offset += size
    receipt = dict(
        schema_version=1,
        path=str(path),
        bytes=total,
        sha256=whole.hexdigest(),
        parts=rows,
        verified=True,
    )
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    # Only the exact file supplied by the caller is removable, after verified PUT and HEAD.
    if remove_verified:
        if path.stat().st_size != total:
            raise RuntimeError("Archive source size changed before removal")
        path.unlink()
    return receipt


def archive_to_directory(
    path, directory, *, remove_verified=False, part_bytes=PART_BYTES
):
    """Use a mounted private artifact store; verify every byte by readback.

    The caller provides an experiment-owned destination. No service credentials
    or signed URLs are needed. All work executes between requests.
    """
    import shutil
    import uuid

    path, directory = Path(path), Path(directory)
    if not 0 < part_bytes <= PART_BYTES:
        raise ValueError("Invalid archive part size")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = path.with_name(path.name + ".archive.json")
    if manifest.exists():
        raise ValueError("Archive manifest already exists")
    whole, rows, offset = hashlib.sha256(), [], 0
    size = path.stat().st_size
    with path.open("rb") as source:
        while offset < size:
            expected = min(part_bytes, size - offset)
            digest = hashlib.sha256()
            remaining = expected
            while remaining:
                block = source.read(min(8 << 20, remaining))
                if not block:
                    raise EOFError("Archive source changed during hashing")
                digest.update(block)
                whole.update(block)
                remaining -= len(block)
            digest = digest.hexdigest()
            target = directory / digest
            if not target.exists():
                source.seek(offset)
                # An interrupted upload must not leave a truncated object under
                # a content hash that a later retry treats as complete. Retain
                # partial bytes separately; the source remains until verification.
                partial = directory / f".{digest}.{uuid.uuid4().hex}.partial"
                with partial.open("xb") as output:
                    remaining = expected
                    while remaining:
                        block = source.read(min(8 << 20, remaining))
                        if not block:
                            raise EOFError("Archive source changed during upload")
                        output.write(block)
                        remaining -= len(block)
                partial.replace(target)
            readback = hashlib.sha256()
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    readback.update(block)
            expected = min(part_bytes, size - offset)
            if target.stat().st_size != expected or readback.hexdigest() != digest:
                raise RuntimeError("Archive mounted-store readback checksum mismatch")
            rows.append(
                dict(path=str(target), offset=offset, bytes=expected, sha256=digest)
            )
            offset += expected
    receipt = dict(
        schema_version=1,
        path=str(path),
        bytes=size,
        sha256=whole.hexdigest(),
        parts=rows,
        verified=True,
    )
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    shutil.copyfile(manifest, directory / (whole.hexdigest() + ".manifest.json"))
    if remove_verified:
        if path.stat().st_size != size:
            raise RuntimeError("Archive source size changed before removal")
        path.unlink()
    return receipt
