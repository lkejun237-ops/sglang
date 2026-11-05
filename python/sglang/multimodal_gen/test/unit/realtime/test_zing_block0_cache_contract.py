# SPDX-License-Identifier: Apache-2.0
"""CPU-only startup, namespace, byte-budget and persisted-shard contracts."""

import errno
import json
import logging
import multiprocessing
import os
import threading
import time
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    block0_identity,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.block0_conditions import (
    CONDITIONS_SCHEMA,
    encode_conditions,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
    Block0DiskCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
    Block0PreparationCache,
    preparation_key,
)
from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import (  # noqa: F401
    cpu_dma,
    disk_commit,
)


def _write_pickle_marker(path):
    Path(path).write_text("Unrestricted pickle executed")


class _PicklePayload:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return _write_pickle_marker, (self.marker,)


def _assert_snapshot_equal(expected, actual):
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        assert actual.device.type == "cpu"
        assert expected.dtype == actual.dtype and torch.equal(expected, actual)
    elif is_dataclass(expected):
        for field in fields(expected):
            _assert_snapshot_equal(
                getattr(expected, field.name), getattr(actual, field.name)
            )
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for name in expected:
            _assert_snapshot_equal(expected[name], actual[name])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for a, b in zip(expected, actual):
            _assert_snapshot_equal(a, b)
    else:
        assert actual == expected


def identity(sp=1, model="weights-a"):
    return {
        "zing_config": {"checkpoint_sha256": model, "vae_config": {}},
        "sp": {"size": sp, "ulysses": sp, "ring": 1},
        "kv_dtype": "torch.bfloat16",
        "cache_schema_version": block0_identity.SCHEMA,
    }


def disk(root, *, sp=1, rank=0, model="weights-a", budget=1 << 24):
    root.mkdir(parents=True, exist_ok=True)
    return Block0DiskCache(
        root, max_bytes=budget, identity=identity(sp, model), rank=rank, world_size=sp
    )


def test_startup_off_does_not_require_or_touch_storage():
    assert block0_identity.startup_config(NS(zing_block0_cache_mode="off")) is None


def test_no_gpu_pool_skips_cache_even_with_missing_parameters(monkeypatch, caplog):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
        ZingKVPoolConfig,
    )

    monkeypatch.setattr(ZingKVPoolConfig, "from_pipeline_config", lambda *a, **kw: None)
    stage = object.__new__(ZingH3GenerationStage)
    with caplog.at_level("INFO"):
        stage.initialize_realtime_kv_cache_pool(
            NS(pipeline_config=object(), realtime_max_sessions_per_worker=1)
        )
    assert stage.block0_cache is None and stage.kv_pool is None
    assert "no GPU KV pool" in caplog.text


@pytest.mark.parametrize("value", [None, 0, -1, True, float("nan"), float("inf")])
def test_invalid_or_missing_budget_fails_startup(value):
    with pytest.raises(ValueError, match="positive decimal GB"):
        block0_identity.budget_bytes(value, "budget")


@pytest.mark.parametrize("mode", ["late", "early", "overlap", "bad"])
def test_production_mode_has_only_on_and_off(mode):
    with pytest.raises(ValueError, match="on or off"):
        block0_identity.startup_config(NS(zing_block0_cache_mode=mode))


def test_budget_is_decimal_gb_and_missing_mount_is_not_created(tmp_path, monkeypatch):
    args = NS(
        zing_block0_cache_mode="on",
        zing_block0_cache_host_gb=800,
        zing_block0_cache_disk_gb=4000,
        zing_block0_cache_dir=str(tmp_path / "unmounted"),
    )
    with pytest.raises(ValueError, match="existing mounted directory"):
        block0_identity.startup_config(args)
    assert not (tmp_path / "unmounted").exists()
    args.zing_block0_cache_dir = str(tmp_path)
    monkeypatch.setattr(block0_identity, "storage_mounted", lambda root: False)
    with pytest.raises(ValueError, match="mounted before"):
        block0_identity.startup_config(args)
    monkeypatch.setattr(block0_identity, "storage_mounted", lambda root: True)
    host, budget, root = block0_identity.startup_config(args)
    assert host == 800_000_000_000 and budget == 4_000_000_000_000 and root == tmp_path


def test_pinned_slots_are_sized_by_bytes_and_floor_unused_remainder(cpu_dma):
    store = NS(k=torch.zeros(8, 2, 3), v=torch.zeros(8, 2, 3))
    pool = NS(_pool=NS(_slots=[NS(kv_cache=[store])]), plan=NS(slot_rows=8))
    slot_bytes = 2 * 8 * 2 * 3 * 4
    owner = Block0PreparationCache(
        pool,
        max_bytes=3 * slot_bytes + 17,
        identity=identity(),
        disk_cache=None,
        page_rows=8,
    )
    assert owner.pages.shape[2] == 3
    assert owner.pages.numel() * owner.pages.element_size() == 3 * slot_bytes
    with pytest.raises(ValueError, match="one pinned KV page"):
        Block0PreparationCache(
            pool,
            max_bytes=slot_bytes - 1,
            identity=identity(),
            disk_cache=None,
            page_rows=8,
        )


def test_key_scopes_config_sp_dtype_but_not_labels_or_environment(monkeypatch):
    codecs = NS(remote_read_fs=None)
    config = dict(width=32, height=32, fps=24)

    def key(metadata=None, request=None):
        return preparation_key(
            codecs, request or {"prompt": "scene"}, None, config, metadata or identity()
        )

    original = key()
    assert len(original) == 64
    for name, value in (
        ("zing_config", {"checkpoint_sha256": "changed"}),
        ("sp", {"size": 2}),
        ("kv_dtype", "torch.float32"),
        ("cache_schema_version", "new"),
    ):
        assert key({**identity(), name: value}) != original
    for field in (
        "game_id",
        "group_id",
        "event_id",
        "request_id",
        "session_id",
        "force",
        "cache_write",
    ):
        assert key(request={"prompt": "scene", field: "ignored"}) == original
    monkeypatch.setenv("SGLANG_ZING_SOURCE_PROMPT_JOINT", "1")
    assert key() == original
    assert block0_identity.fingerprint({"b": 2, "a": 1}) == block0_identity.fingerprint(
        {"a": 1, "b": 2}
    )


def test_identity_reads_only_export_config_and_votes_across_sp(tmp_path, monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning import worker
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state, "get_sp_group", lambda: NS(world_size=2, ring_world_size=1)
    )
    monkeypatch.setattr(parallel_state, "get_ulysses_parallel_world_size", lambda: 2)
    monkeypatch.setattr(worker, "coordinate_validation", lambda value: [value, value])
    pool = NS(
        _pool=NS(_slots=[NS(kv_cache=[NS(k=torch.zeros(1, dtype=torch.bfloat16))])])
    )
    stage = NS(model_path=str(tmp_path), kv_pool=pool)
    path = tmp_path / "zing_config.json"
    path.write_text(
        '{"checkpoint_sha256":"abc", "checkpoint":"/old/path", "format_version":2}'
    )
    before = block0_identity.build_identity(stage, NS(), device="cpu")
    path.write_text(
        '{"format_version": 2, "checkpoint":"/new/path", "checkpoint_sha256":"abc"}'
    )
    assert block0_identity.build_identity(stage, NS(), device="cpu") == before
    assert before["sp"]["size"] == 2 and "checkpoint" not in before["zing_config"]
    path.write_text('{"checkpoint_sha256":"changed"}')
    assert block0_identity.build_identity(stage, NS(), device="cpu") != before
    monkeypatch.setattr(
        worker, "coordinate_validation", lambda value: [value, ("rank failed", None)]
    )
    with pytest.raises(ValueError, match="differs or is invalid"):
        block0_identity.build_identity(stage, NS(), device="cpu")


def test_sp2_only_publishes_after_both_shards_and_keeps_conditions_per_rank(tmp_path):
    a, b = disk(tmp_path, sp=2), disk(tmp_path, sp=2, rank=1)
    ka, _ = disk_commit(a, "case", value=1, rank=0)
    folder = a._entry_dir("case")
    assert not (folder / "rank-1/ready.json").exists()
    assert a.lookup("case") is None and b.lookup("case") is None
    kb, _ = disk_commit(b, "case", value=2, rank=1)
    assert (folder / "rank-1/ready.json").is_file()
    for owner, expected in ((a, ka), (b, kb)):
        hit = owner.lookup("case")
        assert hit is not None
        target = torch.empty_like(expected)
        owner.load_kv_into(hit, target)
        assert torch.equal(target, expected)
        assert torch.equal(
            hit.conditions["inputs"].positive["context"],
            torch.full_like(
                hit.conditions["inputs"].positive["context"],
                expected[0, 0, 0, 0, 0].item(),
            ),
        )
    (folder / "rank-1/kv.bin").write_bytes(b"truncated")
    assert a.lookup("case") is None and b.lookup("case") is None


def test_same_key_shards_from_independent_fills_are_interchangeable(tmp_path):
    owner = disk(tmp_path, sp=2)
    disk_commit(owner, "case", rank=0, generation="a", value=7)
    disk_commit(owner, "case", rank=1, generation="b", value=9)
    entry = owner.lookup("case")
    assert entry is not None and "generation" not in entry.manifest
    for rank, value in ((0, 7), (1, 9)):
        target = torch.empty(2, 2, 3, 2, 2, dtype=torch.bfloat16)
        entry.load_kv_into(target, rank=rank)
        assert torch.equal(target, torch.full_like(target, value))


def test_restart_identity_mismatch_legacy_and_tampering_are_misses(tmp_path):
    owner = disk(tmp_path)
    kv, _ = disk_commit(owner, "case")
    restarted = disk(tmp_path)
    hit = restarted.lookup("case")
    target = torch.empty_like(kv)
    restarted.load_kv_into(hit, target)
    assert torch.equal(kv, target)
    assert disk(tmp_path, model="other-weights").lookup("case") is None
    assert disk(tmp_path, sp=2).lookup("case") is None
    legacy = tmp_path / "ca/case"
    legacy.mkdir(parents=True)
    (legacy / ".complete").touch()
    path = owner._entry_dir("case") / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"]["zing_config"]["checkpoint_sha256"] = "wrong"
    path.write_text(json.dumps(manifest))
    assert owner.lookup("case") is None
    assert owner.lookup("missing-legacy-key") is None


def test_same_byte_size_wrong_dtype_or_geometry_is_rejected(tmp_path):
    owner = disk(tmp_path)
    kv, _ = disk_commit(owner, "case")
    hit = owner.lookup("case")
    for target in (
        torch.empty_like(kv, dtype=torch.float16),
        torch.empty(kv.numel(), dtype=kv.dtype),
    ):
        with pytest.raises(ValueError, match="shape/dtype"):
            owner.load_kv_into(hit, target)


def test_periodic_budget_covers_all_models_without_write_time_eviction(tmp_path):
    a = disk(tmp_path)
    for index in range(55):
        disk_commit(a, f"case-{index:02d}")
    usage = a.collect()
    assert usage["root_entries"] == 55
    a.max_bytes = usage["root_bytes"] // 2
    b = disk(tmp_path, model="model-b")
    disk_commit(b, "new")
    assert len(a._scan()) == 56  # Writes do not enforce a hard budget.
    usage = a.collect()
    assert usage["root_bytes"] <= a.max_bytes * a.gc_low_watermark
    assert not a._entry_dir("case-00").exists()
    assert b.lookup("new") is not None
    assert any(e["key"] == "new" for e in usage["entries"])
    assert any(e["key"].startswith("case-") for e in usage["entries"])


def test_usage_is_unknown_before_gc_and_only_collector_sets_watermark(tmp_path):
    a = disk(tmp_path, budget=123456)
    b = disk(tmp_path, budget=999999)
    assert b.max_bytes == 999999
    manager = Block0DiskCache(tmp_path)
    assert manager.list_entries()["root_bytes"] is None
    assert manager.list_entries()["sampled_at"] is None
    disk_commit(a, "case")
    assert not a.usage_path.exists()
    a.collect()
    assert manager.list_entries()["max_bytes"] == 123456
    assert manager.list_entries()["total_entries"] == 1
    assert not (tmp_path / "cache_config.json").exists()


def test_management_cli_reuses_store_and_reports_partial_failure(
    tmp_path, monkeypatch, capsys
):
    from sglang.multimodal_gen.tools.zing_block0_cache import main

    owner = disk(tmp_path)
    disk_commit(owner, "case")
    owner.collect()
    root = ["--cache-dir", str(tmp_path)]
    scope = []
    assert main(root + ["list", "--limit", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["total_entries"] == 1
    assert main(root + ["delete"] + scope + ["case"]) == 0
    assert json.loads(capsys.readouterr().out)["deleted"] == ["case"]
    assert main(root + ["clear"] + scope) == 0
    assert json.loads(capsys.readouterr().out)["deleted"] == []
    monkeypatch.setattr(
        Block0DiskCache,
        "delete_entries",
        lambda *a, **kw: {"failed": [{"key": "case", "code": "delete_failed"}]},
    )
    assert main(root + ["delete"] + scope + ["case"]) == 1
    assert json.loads(capsys.readouterr().out)["failed"]


def test_force_preserves_other_shards_and_accepts_same_key_mixed_fills(tmp_path):
    a, b = disk(tmp_path, sp=2), disk(tmp_path, sp=2, rank=1)
    for rank, owner in enumerate((a, b)):
        disk_commit(owner, "case", rank=rank, value=1)
    old = a.lookup("case")
    disk_commit(a, "case", value=7, force=True)
    assert a.lookup("case") is not None and b.lookup("case") is not None
    target = torch.empty(2, 2, 3, 2, 2, dtype=torch.bfloat16)
    a.load_kv_into(old, target)
    assert (target == 7).all()
    b.load_kv_into(b.lookup("case"), target)
    assert (target == 1).all()
    disk_commit(b, "case", rank=1, value=8, force=True)
    b.load_kv_into(b.lookup("case"), target)
    assert (target == 8).all()
    disk_commit(a, "case", value=99)
    a.load_kv_into(a.lookup("case"), target)
    assert (target == 7).all()


def test_management_is_metadata_only_and_partially_successful(tmp_path, monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        disk_cache,
    )

    a = disk(tmp_path)
    disk_commit(a, "a")
    disk_commit(a, "b")
    manager = Block0DiskCache(tmp_path)
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("Metadata loaded tensors")
    )
    a.collect()
    view = manager.list_entries(limit=1)
    assert view["root_entries"] == view["total_entries"] == 2
    assert len(view["entries"]) == 1 and view["next_offset"] == 1
    assert view["entries"][0]["prompt_preview"] == "prompt-a"
    assert manager.list_entries(cache_key="a")["total_entries"] == 1
    remove = disk_cache.shutil.rmtree

    def fail_one(path, *args, **kwargs):
        if Path(path) == a._entry_dir("b"):
            raise PermissionError("fixture denied")
        return remove(path, *args, **kwargs)

    monkeypatch.setattr(disk_cache.shutil, "rmtree", fail_one)
    result = manager.delete_entries()
    assert result["deleted"] == ["a"]
    assert result["failed"] == [{"key": "b", "code": "delete_failed"}]
    assert manager.list_entries()["total_entries"] == 2  # Stale snapshot.
    a.collect()
    assert manager.list_entries()["total_entries"] == 1
    result = manager.delete_entries(keys=["a"])
    assert result["deleted"] == [] and result["missing"] == ["a"]
    with pytest.raises(ValueError, match="valid cache keys"):
        manager.delete_entries(keys=["../outside"])


def test_delete_does_not_invalidate_l1_and_background_write_can_recreate(
    tmp_path, monkeypatch, cpu_dma
):
    from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import (
        inputs,
    )
    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import cpu_cache

    monkeypatch.setenv("SGLANG_ZING_DISK_CACHE_DIR", str(tmp_path))
    stores = [NS(k=torch.zeros(8, 2, 3), v=torch.zeros(8, 2, 3))]
    owner = cpu_cache(stores, rows=8, entries=2)
    live = NS(
        prefix=NS(
            positions=torch.zeros(3, 3),
            layers=[(torch.ones(3, 2, 3), torch.ones(3, 2, 3))],
        ),
        _model_signature="model",
    )
    owner.fill("case", inputs(), "condition", "request", live)
    owner.finish_fill()
    owner.wait_for_disk()
    old = owner.entries["case"]
    started, release = threading.Event(), threading.Event()
    write = owner._write_disk

    def delayed(entry):
        started.set()
        assert release.wait(10)
        write(entry)

    monkeypatch.setattr(owner, "_write_disk", delayed)
    try:
        owner.fill("case", inputs(), "condition", "request", live, force=True)
        owner.finish_fill()
        assert started.wait(5)
        # The local prefill is complete while the actual disk writer is blocked.
        fresh = owner.lookup("case")
        assert fresh is not old and fresh.force
        manager = Block0DiskCache(tmp_path)
        result = manager.delete_entries(keys=["case"])
        assert result["deleted"] == ["case"]
        assert owner.lookup("case") is fresh
        assert owner.disk_cache.lookup("case") is None
    finally:
        release.set()
        owner.wait_for_disk()
    entry = owner.disk_cache.lookup("case")
    assert entry is not None


def test_failed_async_write_leaves_local_cache_and_next_request_usable(
    tmp_path, monkeypatch, cpu_dma, caplog
):
    from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import (
        inputs,
    )
    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import cpu_cache

    monkeypatch.setenv("SGLANG_ZING_DISK_CACHE_DIR", str(tmp_path))
    owner = cpu_cache([NS(k=torch.zeros(8, 2, 3))], rows=8, entries=1)
    monkeypatch.setattr(
        owner, "_write_disk", lambda _: (_ for _ in ()).throw(OSError("failed"))
    )
    live = NS(
        prefix=NS(
            positions=torch.zeros(3, 3),
            layers=[(torch.ones(3, 2, 3), torch.ones(3, 2, 3))],
        ),
        _model_signature="model",
    )
    for key in ("first", "next"):
        owner.fill(key, inputs(), "condition", "request", live, force=True)
        owner.finish_fill()
        owner.wait_for_disk()
        assert owner.lookup(key) is not None
    failures = [
        (level, value)
        for level, value in _cache_log_records(caplog)
        if value["event"] == "l2_write" and value["result"] == "failed"
    ]
    assert [value["cache_key"] for _, value in failures] == ["first", "next"]
    assert all(
        level == logging.WARNING and value["reason"] == "OSError"
        for level, value in failures
    )


def test_l1_sp_votes_accept_equivalent_hits_and_reject_partial_or_wrong_key(
    cpu_dma, monkeypatch
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        preparation_cache,
    )
    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import cpu_cache

    owner = cpu_cache([NS(k=torch.zeros(8, 2, 3))])
    owner.entries["case"] = NS(positions=torch.zeros(3, 3))
    for remote, expected in (
        (("case", 3), True),
        (("case", None), False),
        (("case", 5), False),
        (("other", 3), False),
    ):
        monkeypatch.setattr(
            preparation_cache, "coordinate_validation", lambda vote: [vote, remote]
        )
        assert (owner.lookup("case") is not None) == expected


def _wait_for_file(path):
    deadline = time.monotonic() + 30
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {path.name}")
        time.sleep(0.01)


def _crash_during_staging(root):
    owner = disk(Path(root))
    tmp = Path(owner.new_tmp_dir("case"))
    (tmp / "partial.bin").write_bytes(b"partial")
    (Path(root) / "staged").write_text(str(tmp))
    _wait_for_file(Path(root) / "release")


def test_crashed_writer_preserves_formal_files_and_gc_reclaims_staging(tmp_path):
    owner = disk(tmp_path)
    kv, _ = disk_commit(owner, "case")
    ctx = multiprocessing.get_context("spawn")
    writer = ctx.Process(target=_crash_during_staging, args=(str(tmp_path),))
    try:
        writer.start()
        _wait_for_file(tmp_path / "staged")
        writer.terminate()
        writer.join(10)
        target = torch.empty_like(kv)
        owner.load_kv_into(owner.lookup("case"), target)
        assert torch.equal(target, kv)
        stale = Path((tmp_path / "staged").read_text())
        for path in [*stale.rglob("*"), stale]:
            os.utime(path, (1, 1))
        owner.collect()
        assert not stale.exists()
    finally:
        if writer.is_alive():
            writer.terminate()
        writer.join(10)


def _competing_sp_writer(root, label, value, barrier):
    owners = [disk(Path(root), sp=2, rank=rank) for rank in range(2)]
    barrier.wait(30)
    for iteration in range(4):
        for rank in range(2):
            disk_commit(
                owners[rank],
                "race",
                value=value,
                rank=rank,
                generation=f"{label}-{iteration}",
                force=True,
            )


def test_multiprocess_sp_writes_and_reads_never_accept_partial_files(tmp_path):
    observer = disk(tmp_path, sp=2)
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    writers = [
        ctx.Process(
            target=_competing_sp_writer, args=(str(tmp_path), label, value, barrier)
        )
        for label, value in (("a", 7), ("b", 9))
    ]
    try:
        for process in writers:
            process.start()
        deadline = time.monotonic() + 50
        while any(process.is_alive() for process in writers):
            assert time.monotonic() < deadline, "Competing writers did not finish"
            entry = observer.lookup("race")
            if entry is not None:
                target = torch.empty(2, 2, 3, 2, 2, dtype=torch.bfloat16)
                try:
                    observer.load_kv_into(entry, target)
                except ValueError:
                    pass  # A replacement between conditions and KV is a safe miss.
                else:
                    assert (target == 7).all() or (target == 9).all()
            time.sleep(0.01)
        for process in writers:
            process.join(10)
            assert process.exitcode == 0
        # Even if the race ended with a miss, a later ordinary fill can recover.
        manager = Block0DiskCache(tmp_path)
        manager.delete_entries(keys=["race"])
        for rank in range(2):
            disk_commit(
                disk(tmp_path, sp=2, rank=rank),
                "race",
                rank=rank,
                generation="recovered",
            )
        assert observer.lookup("race") is not None
    finally:
        for process in writers:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(10)


def test_read_and_management_do_not_scan_or_rewrite_metadata(tmp_path, monkeypatch):
    owner = disk(tmp_path)
    kv, _ = disk_commit(owner, "case")
    owner.collect()
    before = owner.usage_path.read_bytes()
    monkeypatch.setattr(owner, "_scan", lambda: pytest.fail("Read scanned the root"))
    entry = owner.lookup("case")
    owner.load_kv_into(entry, torch.empty_like(kv))
    assert owner.list_entries()["total_entries"] == 1
    assert owner.usage_path.read_bytes() == before
    assert not (tmp_path / ".index.lock").exists()


def test_read_failure_after_startup_is_a_miss(tmp_path):
    owner = disk(tmp_path)
    disk_commit(owner, "case")
    (owner._entry_dir("case") / "rank-0/conditions.pt").write_bytes(b"broken")
    assert owner.lookup("case") is None


def test_manifest_without_identity_is_never_accepted(tmp_path):
    owner = disk(tmp_path)
    disk_commit(owner, "case")
    path = owner._entry_dir("case") / "manifest.json"
    value = deepcopy(json.loads(path.read_text()))
    del value["identity"]
    path.write_text(json.dumps(value))
    assert owner.lookup("case") is None


def test_safe_conditions_roundtrip_rebuilds_inputs_and_pool_request(tmp_path):
    owner = disk(tmp_path)
    disk_commit(owner, "case")
    before = owner.lookup("case").conditions
    path = owner._entry_dir("case") / "rank-0/conditions.pt"
    prepared = before["inputs"]
    prepared.source_visual = torch.arange(8).reshape(1, 2, 1, 2, 2).bfloat16()
    prepared.audio = {"ref": torch.ones(2, 3, 4)}
    prepared.audio_shape = (2, 3, 4)
    prepared.sample_count = 100
    prepared.max_area_config = {"max_area": 1024}
    torch.save(encode_conditions(before), path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == CONDITIONS_SCHEMA
    assert payload["records"] == {
        "inputs": "H3InferenceInputs",
        "request": "ZingKVRequest",
    }
    assert type(payload["conditions"]["inputs"]) is dict
    assert type(payload["conditions"]["request"]) is dict
    after = disk(tmp_path).lookup("case").conditions
    _assert_snapshot_equal(before, after)


def test_pickle_payload_is_rejected_without_execution_or_fallback(tmp_path):
    owner = disk(tmp_path)
    disk_commit(owner, "case")
    path = owner._entry_dir("case") / "rank-0/conditions.pt"
    payload = torch.load(path, weights_only=True)
    marker = tmp_path / "pickle-executed"
    payload["conditions"]["request"] = _PicklePayload(marker)
    torch.save(payload, path)
    assert owner.lookup("case") is None
    assert not marker.exists()


@pytest.mark.parametrize(
    "corruption", ["legacy", "unknown_record", "missing_field", "tensor_type", "shape"]
)
def test_unsafe_or_invalid_conditions_are_misses(tmp_path, corruption):
    owner = disk(tmp_path)
    disk_commit(owner, "case")
    path = owner._entry_dir("case") / "rank-0/conditions.pt"
    payload = torch.load(path, weights_only=True)
    if corruption == "legacy":
        payload = payload["conditions"]
    elif corruption == "unknown_record":
        payload["records"]["inputs"] = "attacker.ChosenClass"
    elif corruption == "missing_field":
        del payload["conditions"]["inputs"]["positive"]
    elif corruption == "tensor_type":
        payload["conditions"]["inputs"]["positive"]["context"] = "not a tensor"
    else:
        payload["conditions"]["inputs"]["visual_shape"] = (1, 2)
    torch.save(payload, path)
    assert owner.lookup("case") is None


def test_legacy_conditions_manifest_is_rejected_before_deserialization(
    tmp_path, monkeypatch
):
    owner = disk(tmp_path)
    disk_commit(owner, "case")
    path = owner._entry_dir("case") / "rank-0/ready.json"
    manifest = json.loads(path.read_text())
    del manifest["conditions_schema"]
    path.write_text(json.dumps(manifest))
    calls = []
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: calls.append(kwargs))
    assert owner.lookup("case") is None
    assert calls == []


def test_writer_rejects_custom_objects_before_serialization(tmp_path):
    owner = disk(tmp_path)
    disk_commit(owner, "case")
    conditions = owner.lookup("case").conditions
    marker = tmp_path / "pickle-executed"
    conditions["request"] = _PicklePayload(marker)
    with pytest.raises(ValueError, match="plain data"):
        encode_conditions(conditions)
    assert not marker.exists()


def _age_tree(folder, stamp):
    for path in [*folder.rglob("*"), folder]:
        os.utime(path, (stamp, stamp))


def test_gc_uses_child_mtime_and_runs_without_more_writes(tmp_path):
    owner = disk(tmp_path)
    disk_commit(owner, "valid")
    fresh = Path(owner.new_tmp_dir("fresh"))
    (fresh / "data").write_bytes(b"active")
    os.utime(fresh, (1, 1))  # Parent mtime does not describe a child's writes.
    stale = Path(owner.new_tmp_dir("stale"))
    (stale / "data").write_bytes(b"abandoned")
    _age_tree(stale, 1)
    incomplete = owner._entry_dir("incomplete")
    incomplete.mkdir(parents=True)
    (incomplete / "partial").write_bytes(b"orphan")
    _age_tree(incomplete, 1)
    _age_tree(owner._entry_dir("valid"), 1)
    orphan = owner._entry_dir("valid") / "manifest.orphan.tmp"
    orphan.write_bytes(b"partial json")
    os.utime(orphan, (1, 1))
    owner.collect()
    assert fresh.exists() and not stale.exists() and not incomplete.exists()
    assert not orphan.exists() and owner.lookup("valid") is not None
    _age_tree(fresh, 1)
    owner.collect()
    assert not fresh.exists()


def test_gc_delete_failure_does_not_claim_reclaimed_bytes(tmp_path, monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        disk_cache,
    )

    owner = disk(tmp_path)
    disk_commit(owner, "case")
    before = owner.collect()["root_bytes"]
    owner.max_bytes = 1

    def denied(*args, **kwargs):
        raise PermissionError("fixture denied")

    monkeypatch.setattr(disk_cache.shutil, "rmtree", denied)
    result = owner.collect()
    assert result["root_bytes"] == before and result["failed"] == ["case"]
    assert owner.lookup("case") is not None


def test_only_explicit_collector_rank_starts_one_periodic_worker(tmp_path, monkeypatch):
    owner = disk(tmp_path)
    follower = disk(tmp_path, sp=2, rank=1)
    assert owner._gc_thread is None and follower._gc_thread is None
    ran = threading.Event()
    monkeypatch.setattr(owner, "collect", ran.set)
    try:
        follower.start_gc()
        assert follower._gc_thread is None
        owner.start_gc()
        worker = owner._gc_thread
        owner.start_gc()
        assert owner._gc_thread is worker and ran.wait(5)
    finally:
        owner.close()
    assert not worker.is_alive()


def test_open_reader_keeps_complete_inode_during_replacement(tmp_path, monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        disk_cache,
    )

    owner = disk(tmp_path)
    kv, _ = disk_commit(owner, "case", value=1)
    entry = owner.lookup("case")
    readinto = disk_cache._readinto
    replaced = False

    def replace_during_read(handle, target):
        nonlocal replaced
        if not replaced:
            replaced = True
            disk_commit(owner, "case", value=9, force=True)
        return readinto(handle, target)

    monkeypatch.setattr(disk_cache, "_readinto", replace_during_read)
    target = torch.empty_like(kv)
    owner.load_kv_into(entry, target)
    assert (target == 1).all()
    owner.load_kv_into(owner.lookup("case"), target)
    assert (target == 9).all()


def test_delete_between_conditions_and_kv_is_a_load_failure(tmp_path):
    owner = disk(tmp_path)
    kv, _ = disk_commit(owner, "case")
    entry = owner.lookup("case")
    owner.delete_entries(keys=["case"])
    with pytest.raises(OSError):
        owner.load_kv_into(entry, torch.empty_like(kv))
    assert owner.lookup("case") is None


@pytest.mark.parametrize(
    "phase", ["mkdir", "write", "close", "conditions", "replace", "metadata"]
)
def test_enospc_at_each_write_stage_never_publishes_partial_files(
    tmp_path, monkeypatch, phase
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        disk_cache,
    )

    owner = disk(tmp_path)
    kv, _ = disk_commit(owner, "case", value=1)
    old = owner.lookup("case")

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "fixture full")

    if phase == "mkdir":
        monkeypatch.setattr(Path, "mkdir", full)
        assert owner.new_tmp_dir("case") is None
    else:
        tmp = owner.new_tmp_dir("case")
        if phase == "write":
            monkeypatch.setattr(disk_cache, "_write_kv", full)
        if phase == "close":
            import builtins

            real_open = builtins.open

            class FailClose:
                def __enter__(self):
                    self.handle = real_open(Path(tmp) / "block0_kv_rank_0.bin", "wb")
                    return self.handle

                def __exit__(self, *args):
                    self.handle.close()
                    full()

            monkeypatch.setattr(
                disk_cache, "open", lambda *args, **kwargs: FailClose(), raising=False
            )
        path = owner.save_rank_kv("case", 0, kv * 9, tmp)
        if phase in ("write", "close"):
            assert path is None
        else:
            if phase == "conditions":
                monkeypatch.setattr(torch, "save", full)
            elif phase == "replace":
                monkeypatch.setattr(disk_cache.os, "replace", full)
            else:
                monkeypatch.setattr(disk_cache, "_dump_json", full)
            assert (
                owner.commit_entry(
                    "case", old.conditions, old.manifest, {0: path}, tmp, force=True
                )
                is None
            )
        owner.discard_tmp(tmp)
    monkeypatch.undo()
    target = torch.empty_like(kv)
    owner.load_kv_into(owner.lookup("case"), target)
    assert (target == 1).all() or (target == 9).all()
    assert not list(owner.tmp_root.glob("*"))


def _bind_index_worker(root, group, ready, start):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.cache_index import (
        CacheIndex,
    )

    index = CacheIndex(root)
    ready.set()
    if not start.wait(120):
        raise TimeoutError("Cache index workers did not start together")
    for _ in range(5):
        index.bind("game/with unicode 游戏", group, "a" * 64)
        index.bind("game/with unicode 游戏", group, "b" * 64)


def test_index_atomic_associations_deduplicate_groups_and_survive_entity_gc(tmp_path):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.cache_index import (
        CacheIndex,
    )

    index = CacheIndex(tmp_path)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    ready = [context.Event() for _ in range(2)]
    workers = [
        context.Process(
            target=_bind_index_worker,
            args=(str(tmp_path), group, worker_ready, start),
        )
        for group, worker_ready in zip(("stage/1", "stage/2"), ready)
    ]
    try:
        for process in workers:
            process.start()
        # Spawn imports the CUDA-enabled Python stack even for this CPU test.
        # Wait for imports separately, then exercise concurrent writes within
        # the original work deadline instead of letting startup consume it.
        deadline = time.monotonic() + 120
        for worker_ready in ready:
            assert worker_ready.wait(max(0, deadline - time.monotonic()))
        start.set()
        for process in workers:
            process.join(30)
            assert process.exitcode == 0
    finally:
        start.set()
        for process in workers:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(10)
    assert index.select("game/with unicode 游戏") == ["a" * 64, "b" * 64]
    assert index.select("game/with unicode 游戏", "stage/1", "a" * 64) == ["a" * 64]
    index.bind("game/with unicode 游戏", None, "c" * 64)
    assert len(index.select("game/with unicode 游戏")) == 3
    assert not (tmp_path / "entries").exists()  # Index does not create KV.
    with pytest.raises(ValueError, match="does not belong"):
        index.select("other-game", cache_key="a" * 64)
    assert not list((tmp_path / "index").rglob("*.tmp"))


def test_online_lookup_never_touches_disk_and_preload_unload_use_real_files(
    tmp_path, cpu_dma, monkeypatch
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.cache_index import (
        CacheIndex,
    )
    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import cpu_cache

    writer = disk(tmp_path)
    disk_commit(writer, "a" * 64, value=7)
    index = CacheIndex(tmp_path)
    index.bind("game", "stage", "a" * 64)
    owner = cpu_cache(
        [NS(k=torch.zeros(8, 2, 2, dtype=torch.bfloat16)) for _ in range(2)],
        rows=8,
        entries=2,
    )
    owner.identity, owner.disk_cache = writer.identity, writer
    lookup = writer.lookup
    monkeypatch.setattr(
        writer, "lookup", lambda *a, **kw: pytest.fail("Online lookup read disk")
    )
    monkeypatch.setattr(
        writer, "may_contain", lambda *a: pytest.fail("Online hint touched disk")
    )
    assert owner.lookup("a" * 64) is None
    assert not owner.may_contain("a" * 64)
    monkeypatch.setattr(writer, "lookup", lookup)
    result = owner.manage("preload", ["a" * 64, "b" * 64])
    assert not result["all_ready"]
    assert [item["status"] for item in result["results"]] == ["ready", "missing"]
    assert all(
        torch.all(value == 7)
        for _, _, _, value in owner.entries["a" * 64].data.slices()
    )
    assert owner.manage("preload", ["a" * 64])["all_ready"]
    before = list(owner.entries)
    # Missing keys consume no pages; key count alone no longer rejects preload.
    missing = owner.manage("preload", ["a", "b", "c"])
    assert all(item["status"] == "missing" for item in missing["results"])
    assert list(owner.entries) == before
    owner.manage("unload", ["a" * 64])
    assert not owner.status(["a" * 64])["all_ready"]
    assert writer.lookup("a" * 64) is not None
    assert index.select("game", "stage") == ["a" * 64]
    assert owner.free_pages.count(True) == 2


def test_preload_rechecks_actual_residency_and_sp_ready(tmp_path, cpu_dma, monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        preparation_cache,
    )
    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import cpu_cache

    owner = cpu_cache([NS(k=torch.zeros(8, 2, 2))], rows=8, entries=2)

    def loaded(key, **kwargs):
        owner.entries.clear()  # A previously successful load was evicted.
        owner.entries[key] = NS(positions=torch.zeros(1, 3))
        return owner.entries[key]

    monkeypatch.setattr(owner, "lookup", loaded)
    monkeypatch.setattr(owner, "_preload_plan", lambda keys: ([1, 1], 2, 2))
    result = owner.manage("preload", ["a", "b"])
    assert all(item["status"] == "ready" for item in result["results"])
    assert not result["all_ready"]
    monkeypatch.setattr(
        preparation_cache,
        "coordinate_validation",
        lambda value: [value, ({}, *value[1:])],
    )
    assert not owner.status(["b"])["all_ready"]


def test_reference_resize_scopes_effective_inputs_without_runtime_fingerprint():
    from sglang.multimodal_gen.configs.zing_media import ZingResizeLimits

    codecs = NS(remote_read_fs=None, reference_resize=None)
    config = dict(width=832, height=480, fps=24)

    def key():
        return preparation_key(codecs, {"prompt": "scene"}, None, config, identity())

    original = key()
    codecs.reference_resize = ZingResizeLimits(max_short_edge=480, max_area=399360)
    resized = key()
    assert resized != original
    codecs.reference_resize = ZingResizeLimits(max_short_edge=384, max_area=245760)
    assert key() not in (original, resized)
    codecs.reference_resize = None
    assert key() == original


def test_key_ignores_repeated_source_fps_but_retains_model_and_media_fps():
    codecs = NS(remote_read_fs=None)
    config = dict(width=32, height=32, fps=24)
    request = {"prompt": "scene"}
    original = preparation_key(codecs, request, None, config, identity())
    assert (
        preparation_key(codecs, request, None, {**config, "fps": 60}, identity())
        == original
    )
    assert (
        preparation_key(codecs, request, None, {"width": 32, "height": 32}, identity())
        == original
    )
    changed_model = identity()
    changed_model["zing_config"]["h3_media_config"] = {"fps": 60}
    assert preparation_key(codecs, request, None, config, changed_model) != original
    video = {
        **request,
        "media": [{"type": "video", "payload": b"same frames", "fps": 24}],
    }
    other = deepcopy(video)
    other["media"][0]["fps"] = 60
    assert preparation_key(codecs, video, None, config, identity()) != preparation_key(
        codecs, other, None, config, identity()
    )


def _cache_log_records(caplog):
    prefix = "zing_cache_event "
    return [
        (record.levelno, json.loads(record.getMessage()[len(prefix) :]))
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


def test_cache_lookup_and_eviction_logs_have_context_without_prompt(cpu_dma, caplog):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.cache_logging import (
        cache_scope,
    )
    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import cpu_cache

    caplog.set_level(logging.DEBUG)
    owner = cpu_cache([NS(k=torch.zeros(8, 2, 2))], rows=8, entries=1)
    owner.entries["a"] = NS(page_ids=(0,), disk_task=None, positions=torch.zeros(1, 3))
    owner.free_pages[:] = [False] * owner.total_pages
    with cache_scope(
        operation_id="switch-a", session_id="session", event_id=3, game_id="game"
    ):
        assert owner.lookup("a") is not None
        assert owner.lookup("b") is None
        assert owner._take_pages("b", 1).page_ids == (0,)
    records = _cache_log_records(caplog)
    assert [(r["event"], r["result"]) for _, r in records] == [
        ("lookup", "hit"),
        ("lookup", "miss"),
        ("l1_evict", "completed"),
    ]
    for level, value in records:
        assert level == logging.INFO
        assert value["operation_id"] == "switch-a" and value["session_id"] == "session"
        assert value["event_id"] == 3 and value["game_id"] == "game"
        assert value["rank"] == 0 and value["instance"]
        assert {"cache_key", "reason", "elapsed_host_ms"} <= value.keys()
    assert records[1][1]["next_action"] == "recompute"
    assert not records[1][1]["allow_disk"]


@pytest.mark.parametrize("fails", [False, True])
def test_async_cache_write_keeps_fill_context_and_reports_actual_result(
    tmp_path, cpu_dma, caplog, monkeypatch, fails
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.cache_logging import (
        cache_scope,
    )
    from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import (
        inputs,
    )
    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import cpu_cache

    caplog.set_level(logging.DEBUG)
    owner = cpu_cache([NS(k=torch.zeros(8, 2, 2))], rows=8, entries=1)
    owner.disk_cache = disk(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def write(entry):
        entered.set()
        assert release.wait(5)
        if fails:
            raise OSError("secret prompt must not appear in logs")
        return {"key": entry.key}

    monkeypatch.setattr(owner, "_write_disk", write)
    prefix = NS(
        positions=torch.zeros(3, 3),
        layers=[(torch.ones(3, 2, 2), torch.ones(3, 2, 2))],
    )
    with cache_scope(
        operation_id="prefill-a",
        session_id="session-a",
        game_id="game-a",
        group_id="stage-a",
    ):
        owner.fill(
            "a",
            inputs(),
            "condition",
            {"prompt": "secret prompt"},
            NS(prefix=prefix, _model_signature="model"),
            force=True,
        )
    try:
        with cache_scope(operation_id="unrelated-b", game_id="game-b"):
            owner.finish_fill()
            assert entered.wait(5)
            assert not any(
                v["event"] == "l2_write" and v["result"] == "completed"
                for _, v in _cache_log_records(caplog)
            )
    finally:
        release.set()
        owner.wait_for_disk()
    records = _cache_log_records(caplog)
    assert all(
        v["operation_id"] == "prefill-a" and v["game_id"] == "game-a"
        for _, v in records
    )
    writes = [
        (level, value) for level, value in records if value["event"] == "l2_write"
    ]
    assert [v["result"] for _, v in writes] == [
        "started",
        "failed" if fails else "completed",
    ]
    assert writes[-1][0] == (logging.WARNING if fails else logging.INFO)
    assert writes[-1][1]["elapsed_host_ms"] >= 0
    assert "secret prompt" not in caplog.text
    assert "a" in owner.entries  # Persistence failure still leaves a usable L1.
    cpu_dma[1].synchronize.assert_not_called()  # Logging adds no DMA-stream fence.


def test_nonzero_cache_rank_uses_debug_for_success_and_warning_for_failure(caplog):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.cache_logging import (
        cache_event,
        cache_scope,
    )

    caplog.set_level(logging.DEBUG)
    with cache_scope(operation_id="op"):
        cache_event("lookup", "hit", rank=1, cache_key="a")
        cache_event("l2_write", "failed", rank=1, cache_key="a", level=logging.WARNING)
    values = _cache_log_records(caplog)
    assert [level for level, _ in values] == [logging.DEBUG, logging.WARNING]
    assert all(value["rank"] == 1 for _, value in values)


def test_gc_logs_reclamation_and_keeps_empty_scans_and_list_at_debug(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    owner = disk(tmp_path, budget=1)
    disk_commit(owner, "a")
    staging = Path(owner.new_tmp_dir("stale"))
    (staging / "partial").write_bytes(b"temporary")
    for path in [*staging.rglob("*"), staging]:
        os.utime(path, (1, 1))
    caplog.clear()
    owner.collect()
    owner.collect()
    owner.list_entries()
    values = _cache_log_records(caplog)
    gc = [(level, value) for level, value in values if value["event"] == "gc"]
    assert gc[0][0] == logging.INFO and gc[0][1]["result"] == "completed"
    assert gc[0][1]["capacity_deleted_count"] == 1
    assert gc[0][1]["stale_deleted_count"] == 1
    assert gc[0][1]["reclaimed_bytes"] > 0
    assert gc[1][0] == logging.DEBUG and gc[1][1]["result"] == "idle"
    assert values[-1][0] == logging.DEBUG and values[-1][1]["event"] == "list"


def test_cache_delete_partial_failure_is_warning_and_preserves_results(
    tmp_path, caplog, monkeypatch
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        disk_cache,
    )

    caplog.set_level(logging.DEBUG)
    owner = disk(tmp_path)
    disk_commit(owner, "a")
    disk_commit(owner, "b")
    remove = disk_cache.shutil.rmtree

    def fail_one(path, *args, **kwargs):
        if Path(path).name == "b":
            raise PermissionError("denied")
        return remove(path, *args, **kwargs)

    monkeypatch.setattr(disk_cache.shutil, "rmtree", fail_one)
    value = owner.delete_entries()
    assert value["deleted"] == ["a"] and value["failed"] == [
        {"key": "b", "code": "delete_failed"}
    ]
    level, summary = next(
        (level, v)
        for level, v in reversed(_cache_log_records(caplog))
        if v["event"] == "clear"
    )
    assert level == logging.WARNING and summary["result"] == "partial"
    assert summary["deleted_count"] == summary["failed_count"] == 1
    by_key = {
        v["cache_key"]: (level, v["result"])
        for level, v in _cache_log_records(caplog)
        if v["event"] == "l2_delete"
    }
    assert by_key == {
        "a": (logging.INFO, "completed"),
        "b": (logging.WARNING, "failed"),
    }


def test_control_logs_partial_preload_and_busy_rejection(
    tmp_path, cpu_dma, caplog, monkeypatch
):
    from sglang.multimodal_gen.runtime.conditioning import worker
    from sglang.multimodal_gen.runtime.entrypoints.control_requests import ZingCacheReq
    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import cpu_cache

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(worker, "broadcast_receipt", lambda value: value)
    owner = cpu_cache([NS(k=torch.zeros(8, 2, 2))], rows=8, entries=1)
    owner.disk_cache = disk(tmp_path)
    owner.register("a", "game", "stage")
    context = {"operation_id": "http-op", "session_id": "control-session"}
    result = owner.control(
        ZingCacheReq("preload", "game", "stage", log_context=context), busy=False
    )
    assert not result["all_ready"] and result["results"] == [
        {"cache_key": "a", "status": "missing"}
    ]
    value = owner.control(
        ZingCacheReq("unload", "game", "stage", log_context=context), busy=True
    )
    assert value["code"] == "cache_busy"
    records = [
        (level, v)
        for level, v in _cache_log_records(caplog)
        if v["event"] in {"preload", "unload"}
    ]
    assert [(v["event"], v["result"]) for _, v in records] == [
        ("preload", "started"),
        ("preload", "partial"),
        ("unload", "started"),
        ("unload", "rejected"),
    ]
    assert records[1][0] == records[3][0] == logging.WARNING
    assert all(
        v["operation_id"] == "http-op" and v["group_id"] == "stage" for _, v in records
    )
    assert records[1][1]["missing_count"] == 1
    assert records[3][1]["reason"] == "cache_busy"


def test_paged_status_counts_pending_bytes_and_aggregates_sp(cpu_dma, monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        preparation_cache,
    )
    from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import (
        inputs,
    )
    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import cpu_cache

    owner = cpu_cache([NS(k=torch.zeros(8, 2, 2))], rows=8, entries=3, page_rows=4)
    source = NS(
        prefix=NS(
            positions=torch.zeros(5, 3),
            layers=[(torch.ones(5, 2, 2), torch.ones(5, 2, 2))],
        ),
        _model_signature="model",
    )
    owner.fill("a", inputs(), "condition", "request", source)
    status = owner.status(["a"])
    assert "capacity" not in status
    assert {
        k: status[k]
        for k in [
            "page_rows",
            "total_pages",
            "free_pages",
            "budget_bytes",
            "reserved_bytes",
            "used_bytes",
            "entries",
        ]
    } == dict(
        page_rows=4,
        total_pages=6,
        free_pages=4,
        budget_bytes=768,
        reserved_bytes=768,
        used_bytes=256,
        entries=0,
    )
    assert not status["all_ready"]  # pending occupies pages but is not published.
    owner.finish_fill()
    assert owner.status(["a"])["all_ready"]
    monkeypatch.setattr(
        preparation_cache,
        "coordinate_validation",
        lambda snapshot: [snapshot, ({}, 5, 6, 128, 768)],
    )
    status = owner.status(["a"])
    assert status["free_pages"] == 4  # SP guarantee uses the minimum, not sum.
    assert status["budget_bytes"] == status["reserved_bytes"] == 1536
    assert status["used_bytes"] == 384
    assert not status["all_ready"]
