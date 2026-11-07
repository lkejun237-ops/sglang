# SPDX-License-Identifier: Apache-2.0
"""Independently stream exact archive objects and every original tar member."""

import argparse
import hashlib
import io
import json
import os
import tarfile
import time
import traceback
from pathlib import Path


class Parts(io.RawIOBase):
    def __init__(self, manifest, store):
        self.manifest, self.store = manifest, store
        self.index = self.total = self.part_bytes = 0
        self.stream = None
        self.part_hash = None
        self.whole = hashlib.sha256()

    def readable(self):
        return True

    def read(self, size=-1):
        size = 8 << 20 if size < 0 else size
        result = bytearray()
        while len(result) < size:
            if self.stream is None:
                if self.index == len(self.manifest["parts"]):
                    break
                part = self.manifest["parts"][self.index]
                path = Path(part["path"])
                if (
                    not path.is_absolute()
                    or path.resolve() != path
                    or not path.is_relative_to(self.store)
                    or part["offset"] != self.total
                    or part["bytes"] < 1
                ):
                    raise ValueError("Invalid exact archive part path or offset")
                self.stream = path.open("rb")
                self.part_hash, self.part_bytes = hashlib.sha256(), 0
            block = self.stream.read(size - len(result))
            if block:
                self.part_hash.update(block)
                self.whole.update(block)
                self.part_bytes += len(block)
                self.total += len(block)
                result.extend(block)
            else:
                part = self.manifest["parts"][self.index]
                self.stream.close()
                self.stream = None
                if (
                    self.part_bytes != part["bytes"]
                    or self.part_hash.hexdigest() != part["sha256"]
                ):
                    raise ValueError("Archive part size or checksum differs")
                self.index += 1
        return bytes(result)

    def finish(self):
        while self.read(8 << 20):
            pass
        if (
            self.total != self.manifest["bytes"]
            or self.whole.hexdigest() != self.manifest["sha256"]
        ):
            raise ValueError("Whole archive size or checksum differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reader-job-id", required=True)
    args = parser.parse_args()
    raw = args.index.read_bytes()
    index = json.loads(raw)
    store = args.store_root.resolve()
    if not args.index.resolve().is_relative_to(store) or not index.get("completed"):
        raise ValueError("Expected a completed archive index within the owned store")
    proof = dict(
        started=time.time(),
        index_sha256=hashlib.sha256(raw).hexdigest(),
        passed=False,
        source_removed=False,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        reader_job_id=args.reader_job_id,
        rgb=[],
        scope="Independent exact-path whole-stream and tar-member readback; source data remain untouched",
    )
    try:
        members_raw = (args.index.parent / "MEMBERS.json").read_bytes()
        if hashlib.sha256(members_raw).hexdigest() != index["members_sha256"]:
            raise ValueError("Frozen member index changed")
        members = json.loads(members_raw)
        expected = {row["path"]: row for row in members}
        if len(expected) != len(members):
            raise ValueError("Duplicate tar member identity")
        reader, seen = Parts(index["bundle"], store), set()
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                if (
                    not member.isfile()
                    or member.name not in expected
                    or member.name in seen
                ):
                    raise ValueError("Unexpected tar member")
                row = expected[member.name]
                if member.size != row["bytes"]:
                    raise ValueError("Tar member size differs")
                digest, count = hashlib.sha256(), 0
                with archive.extractfile(member) as source:
                    for block in iter(lambda: source.read(8 << 20), b""):
                        digest.update(block)
                        count += len(block)
                if count != row["bytes"] or digest.hexdigest() != row["sha256"]:
                    raise ValueError("Tar member checksum differs")
                seen.add(member.name)
        reader.finish()
        if seen != set(expected):
            raise ValueError("Missing tar members")
        proof["bundle"] = dict(
            bytes=reader.total,
            sha256=reader.whole.hexdigest(),
            members=len(seen),
            exact_part_paths=[part["path"] for part in index["bundle"]["parts"]],
        )
        verified = {}
        for row in index["rgb"]:
            manifest = row["manifest"]
            identity = tuple(
                (part["path"], part["offset"], part["bytes"], part["sha256"])
                for part in manifest["parts"]
            )
            expected = (manifest["bytes"], manifest["sha256"])
            if identity in verified:
                if verified[identity] != expected:
                    raise ValueError(
                        "Conflicting manifest for identical exact archive paths"
                    )
            else:
                reader = Parts(manifest, store)
                reader.finish()
                verified[identity] = (reader.total, reader.whole.hexdigest())
            proof["rgb"].append(
                dict(
                    source_path=row["path"],
                    bytes=verified[identity][0],
                    sha256=verified[identity][1],
                    exact_part_paths=[
                        part["path"] for part in row["manifest"]["parts"]
                    ],
                )
            )
        proof["rgb_exact_composites_verified"] = len(verified)
        proof["passed"] = True
    except BaseException:
        proof["error"] = traceback.format_exc()
        raise
    finally:
        proof["ended"] = time.time()
        with args.output.open("x") as output:
            json.dump(proof, output, indent=2)
    print(
        "INDEPENDENT_OWNED_STALL_ARCHIVE_VERIFIED",
        proof["bundle"]["members"],
        len(proof["rgb"]),
    )


if __name__ == "__main__":
    main()
