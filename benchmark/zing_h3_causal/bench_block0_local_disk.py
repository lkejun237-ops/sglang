#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Same-host Block0 file publication/GC probe, without model dependencies.

Executes the candidate's actual Block0DiskCache and JSON IO definitions. Tensor
serialization is replaced with JSON and KV with fixed byte patterns; real tensor,
conditioning and collective semantics are verified separately in CPU CI.
"""

import argparse
import ast
import base64
import hashlib
import json
import logging
import math
import os
import shutil
import socket
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--index-source", type=Path)
    parser.add_argument("--index-sha256")
    args = parser.parse_args()
    source = args.source.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != args.source_sha256:
        raise ValueError("Candidate source digest differs")
    tree = ast.parse(source)
    definitions = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        and n.name in {"_dump_json", "_read_json", "Block0DiskCache"}
    ]
    scope = dict(
        json=json,
        os=os,
        uuid=uuid,
        math=math,
        shutil=shutil,
        threading=threading,
        time=time,
        Path=Path,
        logger=logging.getLogger("probe"),
        SCHEMA="probe",
        DISK_FORMAT=4,
        CONDITIONS_SCHEMA="probe-conditions",
        encode_conditions=lambda value: value,
    )
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]), str(args.source), "exec"
        ),
        scope,
    )
    if args.index_source is not None:
        index_source = args.index_source.read_bytes()
        if hashlib.sha256(index_source).hexdigest() != args.index_sha256:
            raise ValueError("Candidate index source digest differs")
        index_definitions = [
            n
            for n in ast.parse(index_source).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        ]
        scope["base64"] = base64
        exec(
            compile(
                ast.Module(body=index_definitions, type_ignores=[]),
                str(args.index_source),
                "exec",
            ),
            scope,
        )
    dump = scope["_dump_json"]
    scope["torch"] = SimpleNamespace(save=lambda value, path: dump(path, value))
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        mount = Path(fields[4])
        if root == mount or mount in root.parents:
            mounts.append((mount, fields[fields.index("-") + 1]))
    mount, filesystem = max(mounts, key=lambda item: len(str(item[0])))
    if mount == Path("/"):
        raise ValueError("Probe root is not on a separate mounted filesystem")
    try:
        fd = os.open(root / "leader.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        role = 1
    else:
        os.close(fd)
        role = 0
    fd = os.open(
        root / f"node-{role}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
    )
    os.close(fd)
    node = dict(
        role=role,
        pid=os.getpid(),
        hostname=socket.gethostname(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        source_sha256=digest,
        mount=str(mount),
        filesystem=filesystem,
        topology="same-host",
    )
    dump(root / f"node-{role}.json", node)
    deadline = time.monotonic() + 280

    def wait(name):
        path = root / name
        while not path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for {name}")
            time.sleep(0.02)
        return path

    def mark(name):
        dump(root / name, {"role": role})

    peer = json.loads(wait(f"node-{1 - role}.json").read_text())
    if peer["boot_id"] != node["boot_id"] or peer["pid"] == node["pid"]:
        raise ValueError("Same-host probe requires distinct processes on one kernel")
    if peer["source_sha256"] != digest:
        raise ValueError("Workers use different candidate sources")
    cache_root = root / "cache"
    cache_root.mkdir(exist_ok=True)
    owner = scope["Block0DiskCache"](
        cache_root,
        max_bytes=1 << 30,
        identity={
            "zing_config": {"name": "probe"},
            "sp": {"size": 2},
            "kv_dtype": "bytes",
        },
        world_size=2,
    )
    manifest = dict(
        identity=owner.identity, sp_world_size=2, geometry={}, prompt_preview="probe"
    )
    payloads = [b"a" * (128 << 10), b"b" * (128 << 10)]
    publications = 0

    def publish():
        nonlocal publications
        for rank in (0, 1):
            tmp = Path(owner.new_tmp_dir("case"))
            data = tmp / "kv.bin"
            data.write_bytes(payloads[role])
            dump(tmp / f"kvmeta-{rank}.json", {"shape": [128 << 10], "dtype": "bytes"})
            if owner.commit_entry(
                "case", {"writer": role}, manifest, {rank: data}, str(tmp), force=True
            ):
                publications += 1

    reads, misses = 0, 0

    def read():
        nonlocal reads, misses
        try:
            entry = owner._valid_manifest(owner._entry_dir("case"), "case")
            if entry is None:
                misses += 1
                return
            for rank in (0, 1):
                folder = owner._entry_dir("case") / f"rank-{rank}"
                if (folder / "kv.bin").read_bytes() not in payloads:
                    raise AssertionError("Observed a partial or mixed KV file")
                if json.loads((folder / "conditions.pt").read_text())["writer"] not in (
                    0,
                    1,
                ):
                    raise AssertionError("Invalid conditions publication")
            reads += 1
        except FileNotFoundError:
            misses += 1

    errors = []

    def writer():
        try:
            for _ in range(60):
                publish()
                if time.monotonic() > deadline:
                    raise TimeoutError("Publication loop exceeded deadline")
        except Exception as exc:
            errors.append(exc)
        finally:
            mark(f"published-{role}")

    mark(f"ready-{role}")
    wait(f"ready-{1 - role}")
    worker = threading.Thread(target=writer)
    worker.start()
    while worker.is_alive() or not (root / f"published-{1 - role}").exists():
        read()
        if time.monotonic() > deadline:
            raise TimeoutError("Concurrent writer did not finish")
        time.sleep(0.002)
    worker.join()
    if errors:
        raise errors[0]
    read()
    if not reads or not publications:
        raise AssertionError("No successful publication/read samples")
    publication_result = dict(
        role=role, publications=publications, complete_reads=reads, misses=misses
    )
    dump(root / f"publication-pass-{role}.json", publication_result)
    print(json.dumps(dict(phase="publication", **publication_result)), flush=True)
    mark(f"observed-{role}")
    wait(f"observed-{1 - role}")

    # Test peer-process mtime/size visibility on a still-open writer.
    progress = root / "progress.bin"
    if role == 0:
        with progress.open("wb") as handle:
            handle.write(b"first")
            handle.flush()
            mark("mtime-first")
            wait("mtime-read")
            time.sleep(1.1)
            handle.write(b"second")
            handle.flush()
        mark("mtime-second")
        wait("mtime-pass")
    else:
        wait("mtime-first")
        before = progress.stat()
        mark("mtime-read")
        wait("mtime-second")
        after = progress.stat()
        if after.st_size != 11 or after.st_mtime <= before.st_mtime:
            raise AssertionError("Closed local file size/mtime is not visible")
        result = dict(
            before_size=before.st_size,
            after_size=after.st_size,
            before_mtime_ns=before.st_mtime_ns,
            after_mtime_ns=after.st_mtime_ns,
        )
        dump(root / "mtime-result.json", result)
        print(json.dumps(dict(phase="mtime", **result)), flush=True)
        mark("mtime-pass")

    # One process deletes while the other publishes; misses/recreated keys are legal.
    mark(f"delete-ready-{role}")
    wait(f"delete-ready-{1 - role}")
    for _ in range(30):
        if role == 0:
            publish()
        else:
            owner.delete_entries(keys=["case"])
        read()
    mark(f"delete-done-{role}")
    wait(f"delete-done-{1 - role}")
    if role == 0:
        publish()
        mark("recovered")
    wait("recovered")
    if owner._valid_manifest(owner._entry_dir("case"), "case") is None:
        raise AssertionError("Local entry did not recover after delete/write race")
    read()
    mark(f"recovery-read-{role}")
    wait(f"recovery-read-{1 - role}")

    # Leave a real process-exit staging orphan, collect it from the other process.
    if role == 0:
        pid = os.fork()
        if pid == 0:
            tmp = Path(owner.new_tmp_dir("crashed"))
            (tmp / "partial.bin").write_bytes(b"partial")
            for path in [tmp / "partial.bin", tmp]:
                os.utime(path, (1, 1))
            dump(root / "orphan.json", {"path": str(tmp)})
            os._exit(1)
        _, status = os.waitpid(pid, 0)
        if os.waitstatus_to_exitcode(status) != 1:
            raise AssertionError("Crash fixture did not exit as intended")
        mark("crash-exited")
        wait("gc-pass")
    else:
        wait("crash-exited")
        orphan = Path(json.loads(wait("orphan.json").read_text())["path"])
        usage = owner.collect()
        if orphan.exists() or usage["failed"] or usage["root_entries"] != 1:
            raise AssertionError("Peer-process GC did not reclaim orphan")
        mark("gc-pass")
    if args.index_source is not None:
        index = scope["CacheIndex"](cache_root)
        for _ in range(20):
            index.bind("probe-game/游戏", f"stage-{role}", str(role) * 64)
            index.bind("probe-game/游戏", f"stage-{role}", "a" * 64)
        mark(f"indexed-{role}")
        wait(f"indexed-{1 - role}")
        expected = ["0" * 64, "1" * 64, "a" * 64]
        if index.select("probe-game/游戏") != expected:
            raise AssertionError("Concurrent index registration lost an association")
        if len(index.select("probe-game/游戏", f"stage-{role}")) != 2:
            raise AssertionError("Group selection differs")
        # Recreate the manager as a process-restart/persistence check. KV deletion
        # must leave associations available, including references to absent keys.
        index = scope["CacheIndex"](cache_root)
        if index.select("probe-game/游戏") != expected:
            raise AssertionError("Index persistence differs")
        if list((cache_root / "index").rglob("*.tmp")):
            raise AssertionError("Index publication left temporary files")
        dump(root / f"index-pass-{role}.json", {"keys": expected, "group_entries": 2})
        mark(f"index-checked-{role}")
        wait(f"index-checked-{1 - role}")
    result = dict(
        node,
        ok=True,
        peer=peer,
        publications=publications,
        complete_reads=reads,
        misses=misses,
        mtime_visible=True,
        delete_race_recovered=True,
        orphan_gc=True,
        index_atomic_persistence=args.index_source is not None,
        tensor_serialization="JSON stand-in; covered separately in CPU CI",
        mtime=json.loads(wait("mtime-result.json").read_text()),
    )
    dump(root / f"node-{role}-pass.json", result)
    wait(f"node-{1 - role}-pass.json")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
