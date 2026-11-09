# SPDX-License-Identifier: Apache-2.0
"""Real OS-lock concurrency and small real-tensor storage failure contracts."""

import errno
import gc
import io
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
    Block0DiskCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
    Block0PreparationCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.storage_budget import (
    StorageBudget,
    tree_bytes,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.storage_payload import (
    serialization_bound,
)
from sglang.multimodal_gen.runtime.utils.shared_capacity import (
    CapacityUnavailable,
    SharedCapacity,
)
from sglang.multimodal_gen.test.unit.realtime import zing_cache_fixtures as fixtures

cpu_dma = fixtures.cpu_dma
disk_commit = fixtures.disk_commit


def test_live_machine_lease_cannot_be_stolen_and_dead_process_is_reaped(tmp_path):
    script = """
import sys
from sglang.multimodal_gen.runtime.utils.shared_capacity import SharedCapacity
lease=SharedCapacity(sys.argv[1]).reserve('host-cache',60,100)
print('ready',flush=True)
sys.stdin.read()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "python")},
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        capacity = SharedCapacity(tmp_path)
        with pytest.raises(CapacityUnavailable, match="exhausted"):
            capacity.reserve("host-cache", 50, 100)
        with capacity.reserve("host-cache", 40, 100):
            assert sum(r["bytes"] for r in capacity.snapshot()) == 100
        process.kill()
        process.wait(timeout=5)
        with capacity.reserve("host-cache", 100, 100):
            assert len(capacity.snapshot()) == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


def test_conflicting_limits_and_busy_lock_fail_closed(tmp_path):
    capacity = SharedCapacity(tmp_path, lock_timeout=0.01)
    with capacity.reserve("host-cache", 40, 100):
        with pytest.raises(CapacityUnavailable, match="differs"):
            capacity.reserve("host-cache", 30, 200)
        with capacity.locked():
            with pytest.raises(CapacityUnavailable, match="busy"):
                capacity.reserve("host-cache", 30, 100)


def storage(root, name="one", **kwargs):
    namespace = root / name
    namespace.mkdir(exist_ok=True)
    (namespace / ".tmp").mkdir(exist_ok=True)
    return StorageBudget(
        root,
        namespace,
        limit=kwargs.get("limit", 10000),
        reserve=kwargs.get("reserve", 0),
        inflight=kwargs.get("inflight", 10000),
        writers=kwargs.get("writers", 2),
    )


def staging(budget, name):
    path = budget.namespace / ".tmp" / name
    path.mkdir()
    return path


def test_ranks_instances_legacy_and_partially_written_data_share_one_budget(tmp_path):
    (tmp_path / "legacy-v2.bin").write_bytes(b"x" * 1000)
    first, second = storage(tmp_path), storage(tmp_path, "two")
    a, b = staging(first, "rank0"), staging(second, "rank1")
    with first.reserve_write("key", a, 6000):
        (a / "partial").write_bytes(b"x" * 2000)
        # Written bytes replace outstanding reservation, they do not double-charge.
        with second.reserve_write("key", b, 3000):
            with pytest.raises(CapacityUnavailable):
                first.reserve_write("another", staging(first, "extra"), 1)
    assert (tmp_path / "legacy-v2.bin").stat().st_size == 1000


def test_disk_minimum_free_and_writer_count_are_independent_limits(
    tmp_path, monkeypatch
):
    first = storage(tmp_path, reserve=900, writers=1)
    monkeypatch.setattr(os, "statvfs", lambda _: NS(f_bavail=1000, f_frsize=1))
    with pytest.raises(CapacityUnavailable):
        first.reserve_write("key", staging(first, "too-large"), 101)
    with first.reserve_write("key", staging(first, "a"), 40):
        with first.reserve_write("key", staging(first, "b"), 40):
            with pytest.raises(CapacityUnavailable):
                first.reserve_write("other-key", staging(first, "c"), 1)


def test_collector_has_one_owner_and_can_fail_over(tmp_path):
    first, second = storage(tmp_path), storage(tmp_path, "two")
    with first.collector() as a:
        assert a
        with second.collector() as b:
            assert not b
    with second.collector() as b:
        assert b


def test_payload_bound_counts_full_view_storage():
    backing = torch.arange(100000, dtype=torch.float32)
    data = {"view": backing[:1], "again": backing[:2], "text": "x" * 10000}
    output = io.BytesIO()
    torch.save(data, output)
    assert serialization_bound(data) >= len(output.getbuffer())


def test_gc_preserves_readers_and_never_collects_unregistered_legacy(tmp_path):
    root = tmp_path / "current"
    root.mkdir()
    identity = {"model": "fixture", "sp": {"size": 1}}
    original = Block0DiskCache(root, max_bytes=10**7, identity=identity)
    for key in ["first", "second", "third"]:
        disk_commit(original, key)
    size = tree_bytes(root)
    legacy = tmp_path / "v2"
    legacy.mkdir()
    (legacy / "keep").write_bytes(b"legacy")
    cache = Block0DiskCache(
        root,
        max_bytes=size,
        identity=identity,
        budget_root=tmp_path,
        gc_high_watermark=0.8,
        gc_low_watermark=0.4,
    )
    entry = cache.lookup("first")
    assert entry is not None
    result = cache.collect()
    assert result["collector"] and result["deleted_count"] == 2
    assert (root / "entries/first/rank-0/kv.bin").is_file()
    assert (legacy / "keep").read_bytes() == b"legacy"
    entry.close()


def test_full_disk_declines_write_before_raw_io_and_cleans_reservation(
    tmp_path, monkeypatch
):
    root = tmp_path / "cache"
    root.mkdir()
    cache = Block0DiskCache(root, max_bytes=2 << 20, identity={}, budget_root=tmp_path)
    tmp = cache.new_tmp_dir("key")
    monkeypatch.setattr(os, "statvfs", lambda _: NS(f_bavail=0, f_frsize=1))
    assert cache.save_rank_kv("key", 0, torch.zeros((1, 2, 8)), tmp) is None
    assert not list(Path(tmp).iterdir())
    cache.discard_tmp(tmp)
    assert not cache.budget.state.snapshot()


def test_l1_multiple_instances_and_external_arena_reference_own_machine_quota(
    tmp_path, cpu_dma
):
    store = NS(k=torch.zeros(8, 2, 3), v=torch.zeros(8, 2, 3))
    pool = NS(_pool=NS(_slots=[NS(kv_cache=[store])]), plan=NS(slot_rows=8))

    def allocate():
        return Block0PreparationCache(
            pool,
            max_bytes=4 << 20,
            identity={},
            disk_cache=None,
            page_rows=8,
            capacity_dir=tmp_path,
            host_total_bytes=8 << 20,
        )

    a, b = allocate(), allocate()
    with pytest.raises(CapacityUnavailable):
        allocate()
    arena = a.pages
    a.close()
    with pytest.raises(CapacityUnavailable):
        allocate()
    del arena
    gc.collect()
    c = allocate()
    b.close()
    c.close()
    gc.collect()
    assert SharedCapacity(tmp_path).snapshot() == []


def test_persistence_queue_is_bounded_without_blocking_cache_owner(cpu_dma):
    store = NS(k=torch.zeros(8, 2, 3), v=torch.zeros(8, 2, 3))
    pool = NS(_pool=NS(_slots=[NS(kv_cache=[store])]), plan=NS(slot_rows=8))
    disk = NS(is_enabled=lambda: True, local_rank=0, close=lambda: None)
    owner = Block0PreparationCache(
        pool, max_bytes=4096, identity={}, disk_cache=disk, page_rows=8
    )
    release = threading.Event()
    owner._write_disk = lambda _: release.wait(5)
    entries = [NS(key=str(i), disk_task=None, log_context=None) for i in range(3)]
    try:
        for entry in entries:
            owner._persist(entry)
        assert entries[2].disk_task.is_set()
        assert not entries[0].disk_task.is_set()
        assert not entries[1].disk_task.is_set()
    finally:
        release.set()
        owner.close()
    assert all(e.disk_task.is_set() for e in entries)


def test_manual_delete_and_gc_keep_active_readers_and_old_live_writers(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    identity = {"model": "test", "sp": {"size": 1}}
    disk_commit(Block0DiskCache(root, identity=identity), "ready")
    cache = Block0DiskCache(
        root, identity=identity, max_bytes=10**7, budget_root=tmp_path
    )
    entry = cache.lookup("ready")
    # The HTTP/CLI manager is created without a model or capacity arguments.
    manager = Block0DiskCache(root)
    result = manager.delete_entries(keys=["ready"])
    assert result["failed"] == [{"key": "ready", "code": "entry_busy"}]
    tmp = cache.new_tmp_dir("writing")
    assert cache.reserve_write("writing", tmp, 1024)
    partial = Path(tmp) / "partial"
    partial.write_bytes(b"x")
    for path in (partial, Path(tmp)):
        os.utime(path, (time.time() - 3600,) * 2)
    cache.collect()
    assert partial.is_file()
    cache.discard_tmp(tmp)
    entry.close()
    assert cache.delete_entries(keys=["ready"])["deleted"] == ["ready"]


def test_conditions_admission_happens_before_deserialization(tmp_path, monkeypatch):
    cache = Block0DiskCache(tmp_path, identity={"model": "test", "sp": {"size": 1}})
    disk_commit(cache, "ready")

    def deny(_size):
        raise CapacityUnavailable("L1 auxiliary budget exhausted")

    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("read before admission")
    )
    assert cache.lookup("ready", admit_conditions=deny) is None


def test_index_metadata_cannot_bypass_shared_capacity(tmp_path):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.cache_index import (
        CacheIndex,
    )

    budget = storage(tmp_path)
    index = CacheIndex(budget.namespace)
    index.bind("game", None, "first")
    before = tree_bytes(budget.namespace)
    (tmp_path / "legacy").write_bytes(b"x" * (10000 - before))
    with pytest.raises(CapacityUnavailable):
        index.bind("game", None, "second")
    assert index.select("game") == ["first"]


def test_mid_write_full_disk_releases_reservation_and_keeps_l1(
    tmp_path, monkeypatch, cpu_dma
):
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache as module

    store = NS(k=torch.zeros(8, 2, 3), v=torch.zeros(8, 2, 3))
    pool = NS(_pool=NS(_slots=[NS(kv_cache=[store])]), plan=NS(slot_rows=8))
    root = tmp_path / "cache"
    root.mkdir()
    disk = Block0DiskCache(root, max_bytes=2 << 20, identity={}, budget_root=tmp_path)
    owner = Block0PreparationCache(
        pool, max_bytes=4096, identity={}, disk_cache=disk, page_rows=8
    )
    from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import (
        inputs,
    )

    entry = NS(
        key="ready",
        data=torch.zeros(1, 2, 8),
        inputs=inputs(),
        positions=torch.zeros(1, 3),
        conditioning_identity=None,
        request={},
        force=False,
    )
    owner.entries[entry.key] = entry
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache._block0_geometry",
        lambda _: {},
    )

    def full(path, _tensor, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError(errno.ENOSPC, "full")

    monkeypatch.setattr(module, "_write_kv", full)
    assert owner._write_disk(entry) is None
    assert owner.entries[entry.key] is entry
    assert not list(disk.tmp_root.iterdir())
    assert not disk.budget.state.snapshot()
    owner.close()


def test_namespace_replacement_and_overlap_fail_closed(tmp_path):
    first = storage(tmp_path)
    nested = first.namespace / "nested"
    nested.mkdir()
    with pytest.raises(CapacityUnavailable, match="overlap"):
        StorageBudget(
            tmp_path, nested, limit=10000, reserve=0, inflight=10000, writers=2
        )
    first.namespace.rename(tmp_path / "retired")
    first.namespace.mkdir()
    with pytest.raises(CapacityUnavailable, match="replaced"):
        first.roots()


def test_namespace_handle_lives_with_budget_and_replacement_still_fails(tmp_path):
    first = storage(tmp_path)
    fd = first._namespace_finalizer.peek()[2][0]
    pinned = os.fstat(fd)
    assert pinned.st_ino == first.namespace.stat().st_ino
    assert not os.get_inheritable(fd)
    first.namespace.rename(tmp_path / "old")
    first.namespace.mkdir()
    assert os.fstat(fd).st_ino == (tmp_path / "old").stat().st_ino
    with pytest.raises(CapacityUnavailable, match="replaced"):
        first.roots()
    del first
    gc.collect()
    with pytest.raises(OSError) as closed:
        os.fstat(fd)
    assert closed.value.errno == errno.EBADF


def test_registration_failure_closes_namespace_handle(tmp_path, monkeypatch):
    first = storage(tmp_path)
    first.namespace.rename(tmp_path / "old")
    first.namespace.mkdir()
    opened = []
    real_open = os.open

    def observe(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == first.namespace and flags & os.O_DIRECTORY:
            opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", observe)
    with pytest.raises(CapacityUnavailable, match="replaced"):
        storage(tmp_path)
    assert len(opened) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(opened[0])
    assert closed.value.errno == errno.EBADF


def test_disk_owner_crash_preserves_bytes_until_orphan_collection(tmp_path):
    budget = storage(tmp_path, limit=10000)
    path = staging(budget, "worker")
    script = """
import sys
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location('budget', Path('python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/storage_budget.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
StorageBudget = module.StorageBudget
b = StorageBudget(sys.argv[1], sys.argv[2], limit=10000, reserve=0, inflight=10000, writers=2)
lease = b.reserve_write('key', sys.argv[3], 9000)
(Path(sys.argv[3])/'partial').write_bytes(b'x'*2000)
print('ready',flush=True)
sys.stdin.read()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), str(budget.namespace), str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "python")},
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(CapacityUnavailable):
            budget.reserve_write("next", staging(budget, "next"), 2000)
        child.kill()
        child.wait(timeout=5)
        assert tree_bytes(path) == 2000
        with pytest.raises(CapacityUnavailable):
            budget.reserve_write("next", budget.namespace / ".tmp/next", 9000)
        with budget.reserve_write("next", budget.namespace / ".tmp/next", 7000):
            assert len(budget.state.snapshot()) == 1
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        child.stdin.close()
        child.stdout.close()


def test_machine_launch_plan_cannot_be_bypassed(tmp_path, monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.block0_identity import (
        host_capacity,
    )

    monkeypatch.setenv("SGLANG_CACHE_CAPACITY_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_CACHE_HOST_TOTAL_BYTES", "10000000000")
    monkeypatch.setenv("SGLANG_CACHE_INSTANCE_HOST_BYTES", "4000000000")
    assert host_capacity(NS(zing_block0_cache_host_gb=4)) == (str(tmp_path), 10**10)
    with pytest.raises(ValueError, match="exceeds"):
        host_capacity(NS(zing_block0_cache_host_gb=5))
    with pytest.raises(ValueError, match="bypass"):
        host_capacity(
            NS(
                zing_block0_cache_host_gb=4,
                zing_block0_cache_capacity_dir=str(tmp_path / "other"),
            )
        )


def test_real_dit_parser_accepts_all_capacity_launch_options(tmp_path):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_worker import (
        argument_parser,
    )

    cli = argument_parser().parse_args(
        [
            "--model",
            str(tmp_path),
            "--zing-block0-cache-budget-root",
            str(tmp_path),
            "--zing-block0-cache-min-free-gb",
            "128",
            "--zing-block0-cache-inflight-gb",
            "128",
            "--zing-block0-cache-max-writers",
            "2",
            "--zing-block0-cache-gc-high-watermark",
            "0.9",
            "--zing-block0-cache-capacity-dir",
            str(tmp_path),
            "--zing-block0-cache-host-total-gb",
            "416",
            "--zing-block0-cache-gc",
        ]
    )
    assert cli.zing_block0_cache_host_total_gb == 416
    assert cli.zing_block0_cache_gc is True


def test_busy_l1_writer_skips_optional_fill_without_overwriting_live_pages(cpu_dma):
    from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import (
        inputs,
    )

    stores = [NS(k=torch.zeros(8, 2, 3), v=torch.zeros(8, 2, 3))]
    owner = fixtures.cpu_cache(stores, entries=1, rows=8)
    source = NS(
        prefix=NS(
            positions=torch.zeros(8, 3),
            layers=[(torch.ones(8, 2, 3), torch.ones(8, 2, 3))],
        ),
        _model_signature="test",
    )
    owner.fill("first", inputs(), "condition", "request", source)
    owner.finish_fill()
    entry = owner.entries["first"]
    entry.disk_task = threading.Event()
    try:
        assert owner.fill("second", inputs(), "condition", "request", source) is False
        assert list(owner.entries) == ["first"]
        assert torch.all(owner.pages == 1)
        assert owner.manage("unload", ["first"]) == {"code": "cache_busy"}
    finally:
        entry.disk_task.set()
    owner.fill("second", inputs(), "condition", "request", source)
    owner.finish_fill()
    assert list(owner.entries) == ["second"]
    owner.close()
