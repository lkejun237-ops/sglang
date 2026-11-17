# SPDX-License-Identifier: Apache-2.0
"""Real Block0 providers: shared Store, capacity, rank failures and hit latency.

No model weights are loaded. Uses actual pooled ZingH3KVCache transactions.
Run with the same source/image on CPU preparation and GPU workers.
"""

import argparse
import base64
import hashlib
import io
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace as NS


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def master_preflight(root):
    root.mkdir(exist_ok=True, parents=True)
    http_port, rpc_port = port(), port()
    with (root / "master-preflight.log").open("w") as log:
        proc = subprocess.Popen(
            [
                "mooncake_master",
                f"--port={rpc_port}",
                f"--metrics_port={http_port}",
                "--enable_offload=false",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for _ in range(100):
                assert proc.poll() is None, "Master preflight exited"
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{http_port}/get_all_keys",
                        timeout=0.5,
                    ) as response:
                        assert response.read() == b""
                        return
                except OSError:
                    time.sleep(0.1)
            raise TimeoutError("Master admin endpoint preflight")
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def key(case, seed=0):
    return hashlib.sha256(json.dumps([case, seed], sort_keys=True).encode()).hexdigest()


def pattern(rows, heads, dim, plane, seed, device):
    import torch

    row = torch.arange(rows, dtype=torch.int32, device=device)[:, None]
    col = torch.arange(heads * dim, dtype=torch.int32, device=device)[None, :]
    return (
        (15360 + (row * 17 + col * 13 + plane * 31 + seed * 53) % 1024)
        .to(torch.int16)
        .reshape(rows, heads, dim)
    )


def make(case, rank=0):
    import torch

    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import ZingLayerStore
    from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

    layers, rows, heads, dim = (case[k] for k in ("layers", "rows", "heads", "dim"))
    capacity = rows + 7
    arena = torch.full(
        (layers, 2, capacity + 2, heads, dim), -12345, dtype=torch.int16, device="cuda"
    )
    data = arena.view(torch.bfloat16)
    stores = [
        ZingLayerStore(
            data[layer, 0, 1:-1],
            data[layer, 1, 1:-1],
            torch.zeros_like(data[layer, 0, 1:-1]),
        )
        for layer in range(layers)
    ]
    pool = NS(
        _pool=NS(_slots=[NS(kv_cache=stores)]),
        plan=NS(slot_rows=capacity),
    )
    cache = ZingH3KVCache(
        layers,
        window_policy=ZingWindowPolicy(6, 0),
        rope_position_mode="block_relative",
        storage=stores,
    )
    return pool, cache, arena


def prepared():
    import torch

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
        H3InferenceInputs,
    )

    return H3InferenceInputs(
        output_type="video",
        conditions=[],
        media=[],
        positive={"context": torch.ones(2, 4), "row_role": torch.zeros(2)},
        visual={"ref": torch.ones(1, 2, 1, 2, 2)},
        audio={},
        source_visual=None,
        visual_shape=(1, 2, 2),
        audio_shape=None,
        pixel_frame_count=1,
        sample_count=None,
        sample_rate=24000,
        condition_media=[],
    )


def archive_reference(folder, expected_manifest, case, device):
    import torch

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.block0_conditions import (
        decode_conditions,
    )

    raw = (folder / "manifest.json").read_bytes()
    if (
        hashlib.sha256(raw).hexdigest() != expected_manifest
        or (folder / "complete").read_bytes() != b""
    ):
        raise ValueError("Incomplete archive or unexpected manifest")
    manifest = json.loads(raw)
    shape = (case["layers"], 2, case["rows"], case["heads"], case["dim"])
    expected = {f"l{i:02d}" for i in range(shape[0])} | {"cond"}
    ref = torch.empty(shape, dtype=torch.int16, device=device)
    conditions = None
    for obj in manifest["objects"]:
        name = obj["key"].rsplit(":", 1)[1]
        if (
            name not in expected
            or obj["file"] != hashlib.sha256(obj["key"].encode()).hexdigest()
        ):
            raise ValueError("Unexpected archive object")
        expected.remove(name)
        data = bytearray((folder / obj["file"]).read_bytes())
        if (
            len(data) != obj["bytes"]
            or hashlib.sha256(data).hexdigest() != obj["sha256"]
        ):
            raise ValueError("Archive payload digest mismatch")
        if name == "cond":
            conditions = decode_conditions(
                torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            )
        else:
            dest = ref[int(name[1:])]
            dest.copy_(torch.frombuffer(data, dtype=torch.int16).reshape(dest.shape))
    if (
        expected
        or conditions is None
        or tuple(conditions["positions"].shape) != (shape[2], 3)
    ):
        raise ValueError("Incomplete archive geometry")
    return ref, conditions


def populate(cache, arena, case, seed=0, reference=None):
    import torch

    cache.clear()
    rows = case["rows"]
    arena.fill_(-12345)
    if reference is not None:
        ref, conditions = reference
        arena[:, :, 1 : rows + 1].copy_(ref)
        positions = conditions["positions"]
    else:
        for layer in range(case["layers"]):
            for part in range(2):
                arena[layer, part, 1 : rows + 1].copy_(
                    pattern(
                        rows,
                        case["heads"],
                        case["dim"],
                        layer * 2 + part,
                        seed,
                        arena.device,
                    )
                )
        positions = torch.arange(rows).reshape(-1, 1).repeat(1, 3)
    blocks = torch.zeros(rows, dtype=torch.long)
    cache.update(
        positions.to(arena.device),
        blocks.to(arena.device),
        "prefix",
        0,
        layers=[(slot.k[:rows], slot.v[:rows]) for slot in cache.storage],
        validation=(positions, blocks),
    )


def verify(arena, case, seed=0, reference=None):
    import torch

    rows = case["rows"]
    assert (arena[:, :, 0] == -12345).all().item(), "leading guard"
    assert (arena[:, :, rows + 1 :] == -12345).all().item(), "tail or guard overwritten"
    if reference is not None:
        assert torch.equal(arena[:, :, 1 : rows + 1], reference[0]), (
            "archive KV content mismatch"
        )
        return
    for layer in range(case["layers"]):
        for part in range(2):
            assert torch.equal(
                arena[layer, part, 1 : rows + 1],
                pattern(
                    rows,
                    case["heads"],
                    case["dim"],
                    layer * 2 + part,
                    seed,
                    arena.device,
                ),
            ), ("payload", layer, part)


def identity(case, world=1):
    return dict(
        model="block0-provider-component",
        geometry=case,
        sp=dict(size=world, ulysses=world, ring=1),
        kv_dtype="torch.bfloat16",
    )


def provider(pool, case, config, rank=0, world=1, **kwargs):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        Block0PreparationCache,
    )

    return Block0PreparationCache(
        pool,
        backend="mooncake",
        identity=identity(case, world),
        config=config,
        rank=rank,
        world_size=world,
        **kwargs,
    )


def connect(config, segment=0):
    from mooncake.store import MooncakeDistributedStore

    from sglang.multimodal_gen.runtime.realtime.block0_rdma import block0_rdma_devices

    devices = block0_rdma_devices(config)
    client = MooncakeDistributedStore()
    rc = client.setup(
        config["local_hostname"],
        "P2PHANDSHAKE",
        segment,
        0,
        "rdma",
        devices,
        config["master_server_address"],
    )
    assert rc == 0, ("setup", rc)
    return client


def worker(root, role):
    import torch

    torch.set_num_threads(4)
    config = json.loads((root / "config.json").read_text())
    rank = int(os.environ.get("LOCAL_RANK", "0")) if role == "sp" else 0
    torch.cuda.set_device(rank)
    if role == "store":
        client = connect(config, config["segment_bytes"])
        save(
            root / "store-ready.json",
            dict(pid=os.getpid(), bytes=config["segment_bytes"]),
        )
        try:
            while not (root / "stop-store").exists():
                time.sleep(0.1)
        finally:
            client.close()
        return
    if role == "sp":
        return sp_worker(root, config, rank)
    cases = config["cases"]
    result = dict(role=role, pid=os.getpid(), passed=False, cases=[])
    props = torch.cuda.get_device_properties(rank)
    result["gpu"] = dict(
        name=props.name,
        total_memory=props.total_memory,
        capability=list(torch.cuda.get_device_capability(rank)),
    )
    with torch.no_grad():
        for case in cases:
            pool, cache, arena = make(case)
            reference = None
            fill_args = (prepared(), "condition", None)
            if "archive" in config:
                reference = archive_reference(
                    Path(config["archive"]),
                    config["manifest_sha256"],
                    case,
                    arena.device,
                )
                conditions = reference[1]
                fill_args = (
                    conditions["inputs"],
                    conditions.get("conditioning_identity"),
                    conditions.get("request"),
                )
            backend = provider(pool, case, config)
            try:
                if role == "writer":
                    populate(cache, arena, case, reference=reference)
                    backend.fill(key(case), *fill_args, cache)
                    assert backend.finish_fill() is not None
                    backend.register(key(case), "component", case["name"])
                    arena.fill_(-12345)
                    torch.cuda.synchronize()
                    item = dict(name=case["name"], published=True, source_reused=True)
                elif role == "capacity":
                    item = capacity(backend, cache, arena, case)
                elif role == "orphan-reader":
                    other = provider(pool, case, config, 0, 2)
                    try:
                        assert other.lookup(key(case)) is None
                    finally:
                        other.close()
                    populate(cache, arena, case)
                    backend.fill(key(case), prepared(), "condition", None, cache)
                    assert backend.finish_fill() is not None
                    cache.clear()
                    arena.fill_(-12345)
                    backend.restore(cache, backend.lookup(key(case))).commit(cache)
                    verify(arena, case)
                    item = dict(
                        name=case["name"],
                        incomplete_sp_missed=True,
                        store_usable_after_kill=True,
                        payload_checked=True,
                    )
                else:
                    entry = backend.lookup(key(case))
                    assert entry is not None
                    assert backend.select_keys("component", case["name"]) == [key(case)]
                    transfer = backend.restore(cache, entry)
                    # Queue a real GPU consumer before the test adds any global fence.
                    first = torch.eq(
                        arena[0, 0, 1 : case["rows"] + 1],
                        reference[0][0, 0]
                        if reference is not None
                        else pattern(
                            case["rows"], case["heads"], case["dim"], 0, 0, arena.device
                        ),
                    ).all()
                    transfer.commit(cache)
                    assert first.item(), "consumer stream observed incomplete KV"
                    assert cache.prefix is not None
                    verify(arena, case, reference=reference)
                    item = dict(
                        name=case["name"],
                        shared_hit=True,
                        payload_checked=True,
                        guards_checked=True,
                    )
                    if role == "reader":
                        item.update(
                            performance(
                                backend, cache, arena, pool, case, reference, fill_args
                            )
                        )
                    # Deliberate corruption must be caught by the full verifier.
                    arena[-1, -1, case["rows"], -1, -1] += 1
                    try:
                        verify(arena, case, reference=reference)
                    except AssertionError:
                        item["corruption_detected"] = True
                    else:
                        raise AssertionError("verifier missed injected corruption")
                result["cases"].append(item)
                save(root / (role + ".json"), result)
                print("CASE_PASS", json.dumps(item), flush=True)
            finally:
                backend.close()
                del cache, pool, arena
                torch.cuda.empty_cache()
    result["passed"] = True
    save(root / (role + ".json"), result)


def performance(backend, cache, arena, pool, case, reference, fill_args):
    import torch

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        Block0PreparationCache,
    )

    pages = math.ceil(case["rows"] / 2048)
    max_bytes = pages * 2048 * case["layers"] * 2 * case["heads"] * case["dim"] * 2
    local = Block0PreparationCache(
        pool, max_bytes=max_bytes, identity=identity(case), disk_cache=None
    )
    local.fill(key(case), *fill_args, cache)
    assert local.finish_fill() is not None
    trials = []
    for iteration in range(7):
        modes = [("mooncake", backend), ("local", local)]
        if iteration % 2:
            modes.reverse()
        for label, owner in modes:
            cache.clear()
            arena.fill_(-12345)
            torch.cuda.synchronize()
            start = time.perf_counter()
            entry = owner.lookup(key(case))
            looked_up = time.perf_counter()
            assert entry is not None
            transfer = owner.restore(cache, entry)
            transfer.wait()
            restored = time.perf_counter()
            transfer.commit(cache)
            torch.cuda.synchronize()
            end = time.perf_counter()
            verify(arena, case, reference=reference)
            trials.append(
                dict(
                    mode=label,
                    warmup=iteration < 2,
                    lookup_ms=(looked_up - start) * 1000,
                    restore_ms=(restored - looked_up) * 1000,
                    commit_ms=(end - restored) * 1000,
                    hit_total_ms=(end - start) * 1000,
                    payload_checked=True,
                )
            )
    metrics = {}
    for label in ("mooncake", "local"):
        selected = [t for t in trials if t["mode"] == label and not t["warmup"]]
        metrics[label] = {}
        for name in ("lookup_ms", "restore_ms", "commit_ms", "hit_total_ms"):
            values = sorted(t[name] for t in selected)
            metrics[label][name] = dict(
                median=statistics.median(values),
                p95=values[math.ceil(0.95 * len(values)) - 1],
                n=len(values),
            )
    # The local pool is a benchmark control, not part of MooncakeBackend.
    del local
    return dict(
        payload_bytes=case["layers"]
        * 2
        * case["rows"]
        * case["heads"]
        * case["dim"]
        * 2,
        trials=trials,
        metrics=metrics,
        p95_method="nearest-rank",
    )


def capacity(backend, cache, arena, case):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.mooncake_cache import (
        CacheTransferError,
    )

    outcomes = []

    def inventory(seeds, phase):
        existing = set(backend._list_objects())  # Admin API does not renew leases.
        entries = []
        layer_bytes = 2 * case["rows"] * case["heads"] * case["dim"] * 2
        for seed in seeds:
            cache_key = key(case, seed)
            layers = [name in existing for name in backend._layer_keys(cache_key)]
            conditions = backend._cond_key(cache_key) in existing
            manifest = backend._manifest_key(cache_key) in existing
            entries.append(
                dict(
                    seed=seed,
                    layers=layers,
                    conditions=conditions,
                    manifest=manifest,
                    complete=all(layers) and conditions and manifest,
                )
            )
        stranded = sum(
            sum(entry["layers"]) * layer_bytes
            for entry in entries
            if not entry["complete"]
        )
        return dict(phase=phase, entries=entries, stranded_kv_bytes=stranded)

    for seed in range(30):
        populate(cache, arena, case, seed)
        try:
            backend.fill(key(case, seed), prepared(), "condition", None, cache)
            status = (
                "published" if backend.finish_fill() is not None else "capacity_failed"
            )
        except CacheTransferError:
            status = "capacity_failed"
        outcomes.append(dict(seed=seed, status=status))
    snapshots = [inventory(range(30), "after_pressure")]
    # Let ordinary read leases expire; no hard/soft pin or force remove is used.
    time.sleep(11)
    snapshots.append(inventory(range(30), "leases_expired"))
    populate(cache, arena, case, 99)
    backend.fill(key(case, 99), prepared(), "condition", None, cache)
    assert backend.finish_fill() is not None, "Store unusable after pressure"
    snapshots.append(inventory([*range(30), 99], "after_retry"))
    assert all(s["stranded_kv_bytes"] == 0 for s in snapshots), snapshots
    hits = misses = 0
    for seed in [*range(30), 99]:
        entry = backend.lookup(key(case, seed))
        if entry is None:
            misses += 1
            continue
        cache.clear()
        arena.fill_(-12345)
        backend.restore(cache, entry).commit(cache)
        verify(arena, case, seed)
        hits += 1
    assert misses or any(v["status"] == "capacity_failed" for v in outcomes), (
        "capacity not exercised"
    )
    assert hits > 1, "group eviction did not preserve multiple complete entries"
    return dict(
        name=case["name"],
        outcomes=outcomes,
        snapshots=snapshots,
        verified_hits=hits,
        misses=misses,
        retry_after_pressure=True,
        payload_checked=True,
    )


def sp_worker(root, config, rank):
    import datetime

    import torch
    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.conditioning import worker as coordination
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.mooncake_cache import (
        CacheTransferError,
    )

    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=20))
    group = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=8))
    coordination._sp_group = lambda: NS(world_size=2, cpu_group=group, ranks=[0, 1])
    case = config["cases"][0]
    pool, cache, arena = make(case, rank)
    backend = provider(pool, case, config, rank, 2)
    result = dict(rank=rank, fault=config.get("fault"), passed=False)
    if config.get("fault") == "kill" and rank == 1:

        class KillAfterShard:
            def __init__(self, client):
                self.client = client

            def __getattr__(self, name):
                return getattr(self.client, name)

            def put(self, name, payload, replicate_config):
                rc = self.client.put(name, payload, replicate_config)
                if name.endswith(":cond") and rc == 0:
                    save(
                        root / "kill-receipt.json",
                        dict(
                            pid=os.getpid(),
                            stage="shard_complete_before_manifest",
                            rank=rank,
                            signal="SIGKILL",
                        ),
                    )
                    os.kill(os.getpid(), signal.SIGKILL)
                return rc

        backend.client = KillAfterShard(backend.client)
    try:
        with torch.no_grad():
            populate(cache, arena, case, rank)
            backend.fill(key(case), prepared(), "condition", None, cache)
            assert backend.finish_fill() is not None
            cache.clear()
            arena.fill_(-12345)
            if config.get("fault") == "error" and rank == 1:
                # Inject the local failure; the CPU and device collectives remain real.
                class ReadFault:
                    def __init__(self, client):
                        self.client = client

                    def __getattr__(self, name):
                        return getattr(self.client, name)

                    def batch_get_into_multi_buffers(self, *args):
                        return [-704] * case["layers"]

                backend.client = ReadFault(backend.client)
            try:
                backend.restore(cache, backend.lookup(key(case))).commit(cache)
            except CacheTransferError:
                assert config.get("fault") == "error"
                result["uniform_failure"] = True
            else:
                assert config.get("fault") is None
                verify(arena, case, rank)
                value = torch.ones(1, device="cuda")
                dist.all_reduce(value)
                assert value.item() == 2
                result["nccl_consumed"] = True
        result["passed"] = True
    finally:
        backend.close()
        save(root / f"rank{rank}.json", result)
        dist.destroy_process_group()


def transport_markers(log, devices):
    markers = [
        "Transfer engine auto discovery is disabled for protocol: rdma",
        "Using specified RDMA devices: " + devices,
        "rdma_transport.cpp:",
        *["RDMA device: " + name + "," for name in devices.split(",")],
    ]
    if "using TCP transport only" in log:
        return []
    return markers if all(marker in log for marker in markers) else []


def emit_archive_receipt(protocol, result, reader):
    raw = json.dumps(
        {"result": result, "reader": reader}, separators=(",", ":")
    ).encode()
    digest = hashlib.sha256(raw).hexdigest()
    data = base64.b64encode(raw).decode("ascii")
    chunks = [data[i : i + 1024] for i in range(0, len(data), 1024)]
    for i, chunk in enumerate(chunks):
        print("ARCHIVE_RECEIPT", protocol, digest, i, len(chunks), chunk, flush=True)


def run(
    root,
    purpose,
    cases,
    segment_bytes,
    *,
    fault=None,
    archive=None,
    manifest_sha256=None,
    rdma_devices="",
):
    from sglang.multimodal_gen.runtime.realtime.block0_rdma import block0_rdma_devices

    rdma_devices = block0_rdma_devices({"rdma_devices": rdma_devices})
    root.mkdir(parents=True, exist_ok=False)
    ip = (
        "127.0.0.1"
        if archive is not None
        else socket.gethostbyname(socket.gethostname())
    )
    config = dict(
        local_hostname=ip,
        master_server_address=f"{ip}:{port()}",
        master_http_url=f"http://{ip}:{port()}",
        protocol="rdma",
        rdma_devices=rdma_devices,
        namespace="b0_component",
        segment_bytes=segment_bytes,
        cases=cases,
        fault=fault,
    )
    if archive is not None:
        config.update(
            archive=str(archive),
            manifest_sha256=manifest_sha256,
        )
    save(root / "config.json", config)
    env = dict(
        os.environ,
        MC_STORE_MEMCPY="0",
        GLOG_logtostderr="1",
        MC_STORE_PIN_MEMORY_MAX_BYTES=str(segment_bytes),
    )
    for name in (
        "MC_FORCE_SHM",
        "MC_INTRANODE_NVLINK",
        "MC_FORCE_MNNVL",
    ):
        env.pop(name, None)
    children, logs = [], []
    roles = (
        ["capacity"]
        if purpose == "capacity"
        else ["writer", "reader", "reader-restart"]
    )
    result = dict(protocol_requested="rdma", purpose=purpose, passed=False)

    def start(command, label):
        handle = (root / (label + ".log")).open("w")
        logs.append(handle)
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append(proc)
        return proc

    try:
        master = start(
            [
                "mooncake_master",
                "--port=" + config["master_server_address"].rsplit(":", 1)[1],
                "--metrics_port=" + config["master_http_url"].rsplit(":", 1)[1],
                "--enable_offload=false",
            ],
            "master",
        )
        for _ in range(200):
            assert master.poll() is None, "Master exited"
            try:
                with urllib.request.urlopen(
                    config["master_http_url"] + "/get_all_keys", timeout=0.5
                ):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise TimeoutError("Master HTTP startup")
        service = start(
            [sys.executable, "-u", __file__, "--root", str(root), "--role", "store"],
            "store",
        )
        for _ in range(900):
            assert service.poll() is None, "Store exited"
            if (root / "store-ready.json").exists():
                break
            time.sleep(0.1)
        else:
            raise TimeoutError("Store startup")
        pin_log = (root / "store.log").read_text()
        assert "cudaHostRegister succeeded for" in pin_log, "Store pool not pinned"
        assert f"size={segment_bytes}, pinned={segment_bytes}" in pin_log, (
            "Store pool not fully pinned"
        )
        if purpose == "sp":
            proc = start(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node=2",
                    __file__,
                    "--root",
                    str(root),
                    "--role",
                    "sp",
                ],
                "sp",
            )
            code = proc.wait(timeout=120)
            if fault == "kill":
                assert code != 0 and (root / "kill-receipt.json").exists()
                reader = start(
                    [
                        sys.executable,
                        "-u",
                        __file__,
                        "--root",
                        str(root),
                        "--role",
                        "orphan-reader",
                    ],
                    "orphan-reader",
                )
                assert reader.wait(timeout=120) == 0
                result["real_rank_sigkill"] = True
                result["kill_stage"] = "shard_complete_before_manifest"
            else:
                assert code == 0, "SP component failed"
                assert all(
                    json.loads((root / f"rank{r}.json").read_text())["passed"]
                    for r in range(2)
                )
        else:
            for role in roles:
                proc = start(
                    [
                        sys.executable,
                        "-u",
                        __file__,
                        "--root",
                        str(root),
                        "--role",
                        role,
                    ],
                    role,
                )
                assert proc.wait(timeout=600 if archive is not None else 240) == 0, (
                    role + " failed"
                )
                assert json.loads((root / (role + ".json")).read_text())["passed"]
                assert service.poll() is None
        result["passed"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        (root / "stop-store").touch()
        for proc in reversed(children):
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
        for handle in logs:
            handle.close()
        result["exit_codes"] = [p.returncode for p in children]
        expected_logs = [
            root / "store.log",
            *(
                [root / "sp.log"]
                if purpose == "sp"
                else [root / (r + ".log") for r in roles]
            ),
        ]
        evidence = {
            path.stem: transport_markers(path.read_text(), rdma_devices)
            if path.exists()
            else []
            for path in expected_logs
        }
        result["transport_log_confirmed"] = all(evidence.values())
        if archive is not None:
            result["transport_evidence"] = evidence
        result["passed"] = result["passed"] and result["transport_log_confirmed"]
        save(root / "result.json", result)
        if not result["passed"]:
            for path in root.glob("*.log"):
                print("ERROR_LOG", path.name, path.read_text()[-7000:], flush=True)
        print("COMPONENT_RESULT", json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", default="suite")
    parser.add_argument(
        "--suite", choices=("all", "transfer", "sp", "black-lake-l1"), default="all"
    )
    parser.add_argument(
        "--archive", type=Path, help="Read-only completed Black Lake archive directory"
    )
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--rdma-devices", default="")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        assert pattern(13, 2, 3, 0, 0, "cpu").shape == (13, 2, 3)
        master_preflight(args.root)
        print("BENCH_PREFLIGHT_PASS")
        return
    if args.role != "suite":
        return worker(args.root, args.role)
    import torch

    required_gpus = 2 if args.suite in ("all", "sp") else 1
    assert torch.cuda.device_count() >= required_gpus, (
        "Insufficient GPUs for selected suite"
    )
    odd = dict(name="odd", layers=3, rows=13, heads=2, dim=3)
    full = dict(name="block0-50-layer", layers=50, rows=12288, heads=28, dim=128)
    pressure = dict(name="capacity", layers=3, rows=1024, heads=8, dim=128)
    results = []
    if args.suite == "black-lake-l1":
        if not args.archive or not args.manifest_sha256:
            parser.error("black-lake-l1 requires --archive and --manifest-sha256")
        case = dict(name="black-lake-l1", layers=50, rows=3998, heads=56, dim=128)
        result = run(
            args.root / "rdma",
            "transfer",
            [case],
            12 << 30,
            archive=args.archive.resolve(),
            manifest_sha256=args.manifest_sha256,
            rdma_devices=args.rdma_devices,
        )
        results.append(result)
        reader_file = args.root / "rdma" / "reader.json"
        reader = json.loads(reader_file.read_text()) if reader_file.exists() else None
        emit_archive_receipt("rdma", result, reader)
    if args.suite in ("all", "transfer"):
        results.append(
            run(
                args.root / "capacity",
                "capacity",
                [pressure],
                256 << 20,
                rdma_devices=args.rdma_devices,
            )
        )
        results.append(
            run(
                args.root / "rdma",
                "transfer",
                [odd, full],
                16 << 30,
                rdma_devices=args.rdma_devices,
            )
        )
    if args.suite in ("all", "sp"):
        for label, fault in (("sp", None), ("sp-error", "error"), ("sp-kill", "kill")):
            results.append(
                run(
                    args.root / label,
                    "sp",
                    [odd],
                    256 << 20,
                    fault=fault,
                    rdma_devices=args.rdma_devices,
                )
            )
    passed = all(v["passed"] for v in results)
    save(
        args.root / "GPU_RESULT.json",
        dict(
            passed=passed,
            suite=args.suite,
            results=results,
            sp_tested=args.suite in ("all", "sp"),
        ),
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
