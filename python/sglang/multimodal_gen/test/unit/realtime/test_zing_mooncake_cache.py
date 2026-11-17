# SPDX-License-Identifier: Apache-2.0
"""CPU contracts: real tensor bytes, injected Store errors, and real Gloo ranks."""

import ctypes
import datetime
import json
import multiprocessing
import os
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    preparation_cache as prep,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.block0_identity import (
    startup_config,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.mooncake_cache import (
    CacheTransferError,
    MooncakeBackend,
)
from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import (
    inputs,
)

KEY = "a" * 64
OTHER = "b" * 64


class MemoryStore:
    """Byte transport double, not evidence for real Mooncake allocation/leases."""

    def __init__(self, objects=None):
        self.objects = {} if objects is None else objects
        self.registrations = {}
        self.register_calls = 0
        self.capacity = None
        self.fail_write = False
        self.fail_read = False
        self.before_write = None
        self.exit_on_conditions = False
        self.closed = False
        self.groups = {}
        self.fail_keys = set()
        self.fail_suffix = None
        self.exist_queries = []
        self.before_exists = None

    def register_buffer(self, ptr, size):
        self.registrations[ptr] = size
        self.register_calls += 1
        return 0

    def unregister_buffer(self, ptr):
        del self.registrations[ptr]
        return 0

    def put(self, key, payload, config):
        assert len(config.group_ids) == 1
        if self.exit_on_conditions and key.endswith(":cond"):
            os._exit(17)
        if key in self.fail_keys or (
            self.fail_suffix and key.endswith(self.fail_suffix)
        ):
            return -100
        if key in self.objects:
            return -705
        if (
            self.capacity is not None
            and sum(map(len, self.objects.values())) + len(payload) > self.capacity
        ):
            return -100
        self.objects[key] = bytes(payload)
        self.groups[key] = config.group_ids[0]
        return 0

    def get(self, key):
        return self.objects.get(key, b"")

    def is_exist(self, key):
        return int(key in self.objects)

    def get_size(self, key):
        return len(self.objects.get(key, b""))

    def batch_is_exist(self, keys):
        keys = list(keys)
        self.exist_queries.append(keys)
        if self.before_exists:
            self.before_exists(keys)
        return [self.is_exist(key) for key in keys]

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes, config):
        assert len(config.group_ids) == len(keys)
        if self.before_write:
            self.before_write()
        return [
            -100
            if self.fail_write
            else self.put(
                key,
                b"".join(
                    ctypes.string_at(ptr, size) for ptr, size in zip(pair, counts)
                ),
                NS(group_ids=[group]),
            )
            for key, pair, counts, group in zip(keys, ptrs, sizes, config.group_ids)
        ]

    def batch_get_into_multi_buffers(self, keys, ptrs, sizes):
        result = []
        for key, pair, counts in zip(keys, ptrs, sizes):
            data = self.get(key)
            if self.fail_read or len(data) != sum(counts):
                result.append(-704)
                continue
            offset = 0
            for ptr, size in zip(pair, counts):
                ctypes.memmove(ptr, data[offset : offset + size], size)
                offset += size
            result.append(len(data))
        return result

    def remove(self, key):
        if key not in self.objects:
            return -704
        del self.objects[key]
        return 0

    def batch_remove(self, keys):
        return [self.remove(key) for key in keys]

    def close(self):
        self.closed = True


def provider(store=None, *, rank=0, world=1, rows=13, layers=3, config=None):
    store = MemoryStore() if store is None else store
    storage = [
        NS(
            k=torch.full((rows + 2, 2, 3), -99, dtype=torch.bfloat16),
            v=torch.full((rows + 2, 2, 3), -99, dtype=torch.bfloat16),
        )
        for _ in range(layers)
    ]
    for layer, slot in enumerate(storage):
        slot.k[1:-1].fill_(rank * 50 + layer * 2 + 1)
        slot.v[1:-1].fill_(rank * 50 + layer * 2 + 2)
    pool = NS(
        _pool=NS(_slots=[NS(kv_cache=storage)]),
        plan=NS(slot_rows=rows + 2),
    )
    cache = NS(
        storage=storage,
        prefix=NS(
            positions=torch.arange(rows).reshape(-1, 1).repeat(1, 3),
            layers=[(slot.k[1:-1], slot.v[1:-1]) for slot in storage],
        ),
    )
    cache._reserve_store = lambda layer, count, k, v: (k[1:-1], v[1:-1])
    backend = prep.Block0PreparationCache(
        pool,
        backend="mooncake",
        identity={"sp": {"size": world}},
        config=config if config is not None else {"tiers": {"snapshot_bytes": 1 << 20}},
        rank=rank,
        world_size=world,
        client=store,
        list_keys=lambda: list(store.objects),
        replicate_config_factory=NS,
    )
    return backend, cache, store


def write(backend, cache, key=KEY):
    backend.fill(key, inputs(), "condition", None, cache)
    return backend.finish_fill()


def assert_restored(cache, *, rank=0):
    for layer, slot in enumerate(cache.storage):
        assert (slot.k[1:-1] == rank * 50 + layer * 2 + 1).all()
        assert (slot.v[1:-1] == rank * 50 + layer * 2 + 2).all()
        assert (slot.k[[0, -1]] == -99).all()
        assert (slot.v[[0, -1]] == -99).all()


def test_startup_mooncake_never_requires_disk_or_host_budget(tmp_path):
    path = tmp_path / "client.json"
    path.write_text(json.dumps({"master_server_address": "localhost:50051"}))
    result = startup_config(
        NS(
            zing_block0_cache_mode="on",
            zing_block0_cache_backend="mooncake",
            zing_block0_mooncake_config=str(path),
        )
    )
    assert result["backend"] == "mooncake"


def test_disabled_mooncake_does_not_read_client_configuration(tmp_path):
    assert (
        startup_config(
            NS(
                zing_block0_cache_mode="off",
                zing_block0_cache_backend="mooncake",
                zing_block0_mooncake_config=str(tmp_path / "not-created.json"),
            )
        )
        is None
    )


def test_client_config_alone_does_not_enable_mooncake(tmp_path, monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        block0_identity,
    )

    monkeypatch.setattr(block0_identity, "storage_mounted", lambda _: True)
    result = startup_config(
        NS(
            zing_block0_cache_mode="on",
            zing_block0_cache_host_gb=1,
            zing_block0_cache_disk_gb=2,
            zing_block0_cache_dir=str(tmp_path),
            zing_block0_mooncake_config=str(tmp_path / "not-created.json"),
        )
    )
    assert result == (10**9, 2 * 10**9, tmp_path.resolve())


def test_optional_fill_uses_physical_slot_capacity_and_returns_acceptance():
    b, cache, _ = provider()
    try:
        assert b.can_store(15)
        assert not b.can_store(0)
        assert not b.can_store(16)
        assert b.fill(KEY, inputs(), "condition", None, cache, required=False)
        b.register(KEY, "game", "group")
        assert b.finish_fill(wait=True) is not None
        assert b.select_keys("game", "group") == [KEY]
    finally:
        b.close()


def test_stage_starts_mooncake_with_real_cpu_kv_pool_without_disk(
    tmp_path, monkeypatch
):
    from sglang.multimodal_gen.runtime.distributed import parallel_state
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        block0_identity,
        disk_cache,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        stage as stage_module,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )
    from sglang.multimodal_gen.runtime.platforms import current_platform
    from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan

    store = MemoryStore()
    setup_calls = []
    store.setup = lambda *a: (setup_calls.append(a), 0)[1]
    monkeypatch.setitem(
        sys.modules,
        "mooncake.store",
        NS(MooncakeDistributedStore=lambda: store, ReplicateConfig=NS),
    )
    monkeypatch.setattr(parallel_state, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_sp_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        current_platform, "get_local_torch_device", lambda: torch.device("cpu")
    )
    reserve = stage_module.ZING_MIN_RUNTIME_BYTES
    monkeypatch.setattr(
        stage_module, "cleaned_free_memory", lambda _: reserve + (64 << 10)
    )
    monkeypatch.setattr(
        block0_identity,
        "build_identity",
        lambda *a, **k: {
            "sp": {"size": 1},
            "zing_config": {},
        },
    )
    monkeypatch.setattr(
        disk_cache, "Block0DiskCache", lambda *a, **k: pytest.fail("disk accessed")
    )
    config = tmp_path / "client.json"
    config.write_text(
        json.dumps(
            {
                "master_server_address": "localhost:50051",
                "master_http_url": "http://localhost:9003",
                "rdma_devices": "unit-hca",
            }
        )
    )
    stage = object.__new__(ZingH3GenerationStage)
    stage.codecs = NS(spatial_ratio=16, patch_size=(1, 2, 2), max_short_edge=64)
    stage.packer = NS(
        chunk_plan=ChunkPlan(2),
        _video_t_span=lambda n: 5 / 3 * n,
        visual_token_frame_spans=(1,),
    )
    stage.config = NS(
        dit_config=NS(
            arch_config=NS(num_layers=2, num_attention_heads=4, attention_head_dim=8)
        ),
        dit_precision="bf16",
        zing_kv_pool_size=1,
        zing_kv_pool_max_target_area=4096,
        zing_kv_pool_max_target_long_edge=64,
        zing_kv_pool_max_prefix_rows=16,
        zing_kv_pool_window_chunks=3,
        zing_kv_pool_sink_chunks=1,
        zing_kv_pool_rope_position_mode="block_relative",
        zing_kv_pool_rope_max_frame_gap=12,
        zing_kv_pool_window_eviction_timing="after_step0",
    )
    stage.initialize_realtime_kv_cache_pool(
        NS(
            pipeline_config=stage.config,
            realtime_max_sessions_per_worker=1,
            mem_fraction_static=0.5,
            zing_block0_cache_mode="on",
            zing_block0_cache_backend="mooncake",
            zing_block0_mooncake_config=str(config),
        ),
        before_load=2 * reserve,
    )
    try:
        assert stage.kv_pool.size == 1
        assert isinstance(stage.block0_cache, MooncakeBackend)
        assert not hasattr(stage.block0_cache, "pages")
        assert store.register_calls > 0
        assert setup_calls[0][2:6] == (0, 64 << 20, "rdma", "unit-hca")
    finally:
        stage.block0_cache.close()


def test_new_client_reuses_data_without_local_pages_or_disk(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        disk_cache,
    )

    monkeypatch.setattr(
        disk_cache, "Block0DiskCache", lambda *a, **k: pytest.fail("disk accessed")
    )
    a, source, store = provider()
    b, target, _ = provider(MemoryStore(store.objects))
    try:
        assert isinstance(a, MooncakeBackend)
        assert not hasattr(a, "pages") and not hasattr(a, "disk_cache")
        entry = write(a, source)
        assert entry.key == KEY
        a.register(KEY, "game", "group")
        a.close()
        assert b.select_keys("game", "group") == [KEY]
        entry = b.lookup(KEY, allow_disk=True)
        assert entry is not None and entry.model_signature is None
        for slot in target.storage:
            slot.k[1:-1].fill_(-7)
            slot.v[1:-1].fill_(-7)
        b.restore(target, entry).wait()
        assert_restored(target)
        assert b.cpu_capacity_bytes == 0
        assert b.client.register_calls == len(target.storage) * 2
    finally:
        a.close()
        b.close()


def test_diagnostic_entry_count_never_calls_shared_store():
    b, cache, _ = provider()
    try:
        b._list_keys_override = lambda: pytest.fail("diagnostic HTTP enumeration")
        assert b.entry_count == 0
        assert write(b, cache) is not None
        assert b.entry_count == 1
    finally:
        b.close()


def test_dit_prefill_hit_reports_shared_lookup_without_local_disk():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_preparation import (
        DiTPreparation,
    )

    backend, source, _ = provider()
    admitted = []
    try:
        entry = write(backend, source)
        owner = DiTPreparation(
            NS(block0_cache=backend, admit_cached_preparation=admitted.append),
            lambda: pytest.fail("A cached prefill must not recompute"),
        )
        result = owner.control("prefill", {"cache_key": KEY}, busy=False)
        assert result["hit"] is True
        assert result["cache_key"] == entry.key
        assert result["entries"] == 1
        assert result["_disk_descriptor"] is None
        assert admitted[0].key == entry.key
        assert backend.last_lookup["cache_result"] == "l1_hit"
        assert backend.last_lookup["backend"] == "mooncake"
        assert backend.last_lookup["disk_allowed"] is False
        assert backend.last_lookup["lookup_host_ms"] >= 0
        assert backend.lookup(OTHER) is None
        assert backend.last_lookup["cache_result"] == "miss"
    finally:
        backend.close()


def test_shared_restore_report_and_dispose_do_not_require_cuda_events(caplog):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.block0_telemetry import (
        Block0Telemetry,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )

    backend, source, _ = provider()
    try:
        entry = write(backend, source)
        state = ZingSourceStreamState()
        state.block0_cache = backend
        state.block0_entry = entry
        state.block0_key = entry.key
        state.block0_prompt_version = 1
        state.block0_telemetry = Block0Telemetry()
        state.block0_restore = backend.restore(source, entry)
        assert state.block0_restore.end is None
        report = state._take_block0_report([{"prompt_version": 1}])
        assert report["cache_key"] == entry.key
        assert "restore" not in state.block0_telemetry.cuda
        assert state._take_block0_report([{"prompt_version": 1}]) is None
        state.dispose()
        assert state.block0_cache is None
    finally:
        backend.close()


def test_native_group_spans_ranks_and_business_bindings():
    store = MemoryStore()
    a, cache, _ = provider(store, world=2)
    b, _, _ = provider(store, rank=1, world=2)
    try:
        # Group identity excludes the rank and business selection labels.
        assert a._replicate_config(KEY, 3).group_ids == [a._base(KEY)] * 3
        assert b._replicate_config(KEY).group_ids == [a._base(KEY)]
        assert a._replicate_config(OTHER).group_ids != [a._base(KEY)]
        # Publication needs collectives; use a single-rank writer below.
        a.world_size = 1
        write(a, cache)
        a.register(KEY, "game-a", "group-a")
        a.register(KEY, "game-b", "group-b")
        assert len(store.objects) == 7  # 3 layers, conditions, manifest, 2 labels
        assert set(store.groups.values()) == {a._base(KEY)}
        assert ":g3:" in a._namespace
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("index_first", [False, True])
def test_post_publication_check_covers_early_and_late_index(index_first):
    b, cache, store = provider()
    release = threading.Event()
    try:
        if index_first:
            store.before_write = lambda: release.wait(10)
            b.fill(KEY, inputs(), "condition", None, cache)
            b.register(KEY, "game", "group")
            # An unpublished association must not refresh a partial group's
            # lease. finish_fill checks it after all payload members are ready.
            assert store.exist_queries == []
            # Native pressure can discard the early index before KV completes.
            association = next(k for k in store.objects if ":idx:" in k)
            del store.objects[association]
            release.set()
            assert b.finish_fill() is not None
        else:
            assert write(b, cache) is not None
        if not index_first:
            b.register(KEY, "game", "group")
        manifest = b._manifest_key(KEY)
        checks = [q for q in store.exist_queries if q[0] == manifest]
        assert checks
        assert set(checks[-1]) >= {manifest, b._cond_key(KEY), *b._layer_keys(KEY)}
        association = next(k for k in store.objects if ":idx:" in k)
        assert association in checks[-1]
        assert b._pending_indices == set()
        # The original pre-publication KV-only check is retained.
        assert list(b._layer_keys(KEY)) in store.exist_queries
        assert b.lookup(KEY) is not None
    finally:
        release.set()
        b.close()


def test_post_publication_missing_shard_fails_without_deleting_other_members():
    b, cache, store = provider()
    try:

        def evict_before_check(keys):
            if keys[0] == b._manifest_key(KEY):
                store.objects.pop(b._layer_keys(KEY)[1], None)

        store.before_exists = evict_before_check
        assert write(b, cache) is None
        assert b.write_failures == 1
        # No rollback: another instance can still own/reuse these same-key objects.
        assert b._manifest_key(KEY) in store.objects
        assert b._cond_key(KEY) in store.objects
        assert b._layer_keys(KEY)[0] in store.objects
        assert b.lookup(KEY) is None
        store.before_exists = None
        assert write(b, cache) is not None
        b.restore(cache, b.lookup(KEY)).wait()
        assert_restored(cache)
    finally:
        b.close()


@pytest.mark.parametrize("failure", ["layer", "conditions", "manifest"])
def test_partial_failure_retry_and_same_key_reuse_keep_shared_objects(failure):
    b, cache, store = provider()
    other, target, _ = provider(store)
    try:
        failed = {"layer": ":kv:l01", "conditions": ":cond", "manifest": ":manifest"}[
            failure
        ]
        store.fail_suffix = failed
        assert write(b, cache) is None
        assert store.objects  # Real partial success, rather than all writes failing.
        assert not store.get(b._manifest_key(KEY))
        assert b.lookup(KEY) is None
        assert set(store.groups.values()) == {b._base(KEY)}
        partial = dict(store.objects)
        store.fail_suffix = None
        assert write(other, target) is not None
        assert all(store.objects[key] == value for key, value in partial.items())
        other.restore(target, other.lookup(KEY)).wait()
        assert_restored(target)
        published = dict(store.objects)
        # A failed concurrent/repeated fill must not roll back another writer's
        # complete result. Every key is immutable and still belongs to its group.
        store.fail_suffix = failed
        assert write(b, cache) is None
        assert all(store.objects[key] == value for key, value in published.items())
        assert other.lookup(KEY) is not None
    finally:
        b.close()
        other.close()


def test_full_store_never_publishes_partial_and_can_retry():
    b, cache, store = provider()
    try:
        write(b, cache)
        store.capacity = sum(map(len, store.objects.values())) + 400
        b.fill(OTHER, inputs(), "condition", None, cache)
        assert b.finish_fill() is None
        assert b.write_failures == 1
        assert not store.get(b._manifest_key(OTHER))
        assert b.lookup(OTHER) is None
        assert b.lookup(KEY) is not None
        store.capacity = None
        write(b, cache, OTHER)
        b.restore(cache, b.lookup(OTHER))
        assert_restored(cache)
    finally:
        b.close()


@pytest.mark.parametrize("damage", ["layer", "conditions", "geometry", "short_read"])
def test_incomplete_or_invalid_cache_never_installs_prefix(damage):
    b, cache, store = provider()
    try:
        write(b, cache)
        entry = b.lookup(KEY)
        if damage == "layer":
            store.remove(entry.data.keys[-1])
        elif damage == "conditions":
            store.objects[b._cond_key(KEY)] = b"not a torch archive"
        elif damage == "geometry":
            header = json.loads(store.get(b._manifest_key(KEY)))
            header["geometry"]["dtype"] = "float32"
            store.objects[b._manifest_key(KEY)] = json.dumps(header).encode()
        else:
            store.fail_read = True
            with pytest.raises(CacheTransferError):
                b.restore(cache, entry)
            store.fail_read = False
            b.restore(cache, entry)
            assert_restored(cache)
            return
        assert b.lookup(KEY) is None
    finally:
        b.close()


def test_publication_waits_for_background_write_and_close_keeps_buffers():
    b, cache, store = provider()
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def delay():
        entered.set()
        assert release.wait(10)

    store.before_write = delay
    try:
        b.fill(KEY, inputs(), "condition", None, cache)
        assert entered.wait(5)
        assert not store.get(b._manifest_key(KEY))
        closer = threading.Thread(target=lambda: (b.close(), closed.set()))
        closer.start()
        assert not closed.wait(0.05)
        assert store.registrations
        release.set()
        closer.join(10)
        assert closed.is_set() and not store.registrations
        assert not store.get(b._manifest_key(KEY))
    finally:
        release.set()
        b.close()


def test_shared_unload_and_preload_missing_without_disk():
    b, cache, store = provider()
    try:
        assert b.manage("preload", [KEY])["results"][0]["status"] == "missing"
        write(b, cache)
        assert b.status([KEY])["all_ready"]
        assert b.manage("unload", [KEY])["results"][0]["status"] == "unloaded"
        assert not b.status([KEY])["all_ready"]
        assert not store.objects
        write(b, cache)
        assert b.lookup(KEY) is not None
    finally:
        b.close()


@pytest.mark.parametrize("operation", ["list_entries", "delete_entries"])
def test_disk_api_rejects_mooncake_before_constructing_disk(monkeypatch, operation):
    from fastapi import HTTPException

    from sglang.multimodal_gen.runtime.entrypoints.openai import zing_cache_api
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_preparation import (
        DiTPreparation,
    )

    monkeypatch.setattr(
        zing_cache_api,
        "get_global_server_args",
        lambda: NS(
            zing_block0_cache_backend="mooncake", zing_block0_cache_dir="/unused"
        ),
    )
    monkeypatch.setattr(
        zing_cache_api,
        "_manager",
        lambda *a: pytest.fail("disk accessed"),
    )
    with pytest.raises(HTTPException) as raised:
        zing_cache_api._call(operation)
    assert raised.value.status_code == 501
    stage = NS(block0_cache=NS(backend="mooncake"))
    assert DiTPreparation(stage, None).control(operation, {}, busy=False) == {
        "code": "cache_operation_unsupported"
    }


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize(
    "method,path", [("get", ""), ("delete", "/entries/" + KEY), ("post", "/clear")]
)
def test_l2_http_routes_reject_mooncake_without_touching_disk(
    monkeypatch, remote, method, path
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.entrypoints.openai import zing_cache_api

    monkeypatch.setattr(
        zing_cache_api,
        "get_global_server_args",
        lambda: NS(zing_block0_cache_backend="mooncake"),
    )
    monkeypatch.setattr(
        zing_cache_api, "_manager", lambda *a: pytest.fail("disk accessed")
    )
    app = FastAPI()
    app.include_router(zing_cache_api.router)
    if remote:

        async def cache_disk(*a, **k):
            return {"code": "cache_operation_unsupported"}

        app.state.realtime_execution_client = NS(cache_disk=cache_disk)
    with TestClient(app) as client:
        response = client.request(method, "/v1/zing/cache" + path)
    assert response.status_code == 501
    assert response.json() == {"detail": {"code": "cache_operation_unsupported"}}


def test_independent_bindings_do_not_overwrite_shared_index():
    a, _, store = provider()
    b, _, _ = provider(MemoryStore(store.objects))
    try:
        a.register(KEY, "game", "one")
        b.register(OTHER, "game", "two")
        a.register(KEY, "game", "one")
        assert a.select_keys("game") == sorted([KEY, OTHER])
        assert b.select_keys("game", "one") == [KEY]
        assert a.select_keys("game", "two") == [OTHER]
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize(
    "backend, code",
    [("local", "cache_capacity_busy"), ("mooncake", "cache_unavailable")],
)
def test_explicit_dit_prefill_reports_unpublished_entry(monkeypatch, backend, code):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        dit_preparation,
    )

    monkeypatch.setattr(dit_preparation, "fill_prepared_prefix", lambda *a, **k: None)
    monkeypatch.setattr(dit_preparation, "prepared_inputs", lambda v: NS(positive={}))
    monkeypatch.setattr(dit_preparation, "inputs_on", lambda v, device: v)
    lookups = []
    cache = NS(
        backend=backend,
        lookup=lambda key, **kwargs: lookups.append(kwargs),
        last_lookup={"cache_result": "miss", "miss_reason": "not_resident"},
    )
    transformer = NS(parameters=lambda: iter([torch.zeros(1)]))
    owner = dit_preparation.DiTPreparation(
        NS(
            block0_cache=cache,
            modules={"transformer": transformer},
            codecs=NS(validate_prepared_compute_inputs=lambda _: None),
            admit_realtime_condition=lambda *a, **k: None,
        ),
        lambda: None,
    )
    assert owner.control(
        "prefill",
        dict(
            cache_key=KEY,
            resident_only=backend == "mooncake",
            inputs={},
            request={},
            source_shape=[1, 1],
        ),
        busy=False,
    ) == {"code": code}

    assert lookups == (
        [{"resident_only": True}] if backend == "mooncake" else [{"allow_disk": True}]
    )


def test_integrated_prefill_reports_unpublished_entry(monkeypatch, caplog):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )

    monkeypatch.setattr(ZingSourceStreamState, "_prefill", lambda *a, **k: None)
    with pytest.raises(ValueError, match="did not publish"):
        ZingSourceStreamState().prefill(
            {},
            stage=NS(block0_cache=None),
            batch=NS(extra={}),
        )
    assert '"result":"failed"' in caplog.text
    assert '"result":"completed"' not in caplog.text


def _rank_worker(rank, rendezvous, objects, outcome, fault):
    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.conditioning import worker

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=5),
    )
    worker._sp_group = lambda: NS(
        world_size=2, cpu_group=dist.group.WORLD, ranks=[0, 1]
    )
    store = MemoryStore(objects)
    store.fail_write = fault == "error" and rank == 1
    store.exit_on_conditions = fault == "exit" and rank == 1
    b, cache, _ = provider(store, rank=rank, world=2)
    try:
        write(b, cache)
        entry = b.lookup(KEY)
        assert entry is not None
        if fault == "index":
            if rank == 0:

                def fail_index():
                    from http.client import IncompleteRead

                    raise IncompleteRead(b"partial")

                b._list_keys_override = fail_index
            response = b.control(
                NS(operation="status", game_id="game", group_id=None, cache_key=None),
                busy=False,
            )
            assert response["code"] == "cache_unavailable"
            outcome[rank] = "unavailable"
            return
        if fault in ("remove", "batch_remove"):
            if rank == (0 if fault == "remove" else 1):

                def fail_remove(*args):
                    raise RuntimeError("transport lost during unload")

                setattr(store, fault, fail_remove)
            assert b.manage("unload", [KEY])["code"] == "cache_busy"
            outcome[rank] = "busy"
            return
        b.restore(cache, entry)
        assert_restored(cache, rank=rank)
        outcome[rank] = "ok"
    except Exception:
        outcome[rank] = "failed"
    finally:
        b.close()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "fault", [None, "error", "exit", "remove", "batch_remove", "index"]
)
def test_real_cpu_ranks_publish_or_exit_without_partial_manifest(tmp_path, fault):
    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as manager:
        objects, outcome = manager.dict(), manager.dict()
        rendezvous = "file://" + str(tmp_path / "rendezvous")
        children = [
            ctx.Process(
                target=_rank_worker, args=(r, rendezvous, objects, outcome, fault)
            )
            for r in range(2)
        ]
        try:
            for child in children:
                child.start()
            deadline = time.monotonic() + 40
            for child in children:
                child.join(max(0, deadline - time.monotonic()))
            assert not any(child.is_alive() for child in children), (
                "SP peer hung after rank failure"
            )
            if fault is None:
                assert dict(outcome) == {0: "ok", 1: "ok"}
                assert any(key.endswith(":manifest") for key in objects)
            elif fault in ("remove", "batch_remove"):
                assert dict(outcome) == {0: "busy", 1: "busy"}
            elif fault == "index":
                assert dict(outcome) == {0: "unavailable", 1: "unavailable"}
            else:
                assert outcome[0] == "failed"
                assert not any(key.endswith(":manifest") for key in objects)
                if fault == "exit":
                    assert children[1].exitcode == 17
        finally:
            for child in children:
                if child.is_alive():
                    child.terminate()
                child.join(5)


def test_different_world_preparations_never_mix_failed_attempt_objects():
    first, original, store = provider()
    second, replacement, _ = provider(store)
    try:
        # A partially uploaded S0 A must not donate KV to a later S0 B fill.
        store.fail_suffix = ":cond"
        assert write(first, original) is None
        failed_keys = set(store.objects)
        for k, v in replacement.prefix.layers:
            k.fill_(71)
            v.fill_(72)
        store.fail_suffix = None
        committed = write(second, replacement)
        assert committed is not None
        assert not failed_keys.intersection(committed.data.keys)
        for slot in original.storage:
            slot.k.fill_(-99)
            slot.v.fill_(-99)
        first.restore(original, first.lookup(KEY)).wait()
        for slot in original.storage:
            assert (slot.k[1:-1] == 71).all()
            assert (slot.v[1:-1] == 72).all()
    finally:
        first.close()
        second.close()


def test_optional_writeback_does_not_block_lookup_and_owns_immutable_kv():
    b, cache, store = provider(config={"tiers": {"snapshot_bytes": 1 << 20}})
    entered, release = threading.Event(), threading.Event()

    def delay():
        entered.set()
        assert release.wait(5)

    store.before_write = delay
    try:
        assert b.fill(KEY, inputs(), "condition", None, cache, required=False)
        assert entered.wait(2)
        # A hot-path lookup must not wait for the blocked background Store put.
        before = time.monotonic()
        assert b.lookup(KEY) is None
        assert time.monotonic() - before < 0.5
        for slot in cache.storage:
            slot.k.fill_(-99)
            slot.v.fill_(-99)
        release.set()
        assert b.finish_fill(wait=True) is not None
        b.restore(cache, b.lookup(KEY)).wait()
        assert_restored(cache)
    finally:
        release.set()
        b.close()


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_optional_fill_persists_reference_media_received_on_accelerator(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    backend, cache, _ = provider()
    try:
        prepared = inputs()
        original = prepared.condition_media[0]["frames"].clone()
        prepared.condition_media[0]["frames"] = original.to(device)
        assert backend.fill(KEY, prepared, "condition", None, cache, required=False)
        assert backend.finish_fill(wait=True) is not None
        persisted = backend.lookup(KEY).inputs.condition_media[0]["frames"]
        assert persisted.device.type == "cpu" and torch.equal(persisted, original)
        assert prepared.condition_media[0]["frames"].device.type == device
    finally:
        backend.close()


def test_optional_writeback_respects_explicit_snapshot_budget():
    b, cache, store = provider(config={"tiers": {"snapshot_bytes": 1}})
    try:
        assert not b.fill(KEY, inputs(), "condition", None, cache, required=False)
        assert not store.objects and b.pending is None
    finally:
        b.close()


def test_polls_wait_for_native_promotion_receipt():
    backend, cache, _ = provider()
    try:
        entry = write(backend, cache)
        backend.tiers = replace(backend.tiers, l2=True, online_wait_ms=0)
        future = Future()
        backend._loads[KEY] = future
        # The group is resident, but its original L2 retrieval is still running.
        assert backend.lookup(KEY) is None
        future.set_result({"source_layer": "l2", "entry": entry, "stages": {}})
        assert backend.lookup(KEY) is not None
        assert backend.last_lookup["cache_result"] == "l2_hit"
        assert backend.lookup(KEY) is not None
        assert backend.last_lookup["cache_result"] == "l1_hit"
    finally:
        backend.close()


def test_disk_marker_promotes_group_before_second_descriptor_query():
    backend, cache, _ = provider()
    try:
        entry = write(backend, cache)
        result = backend._load_tiers(KEY, marker_layers={"l2"})
        assert result["entry"].data.artifact_id == entry.data.artifact_id
        assert result["source_layer"] == "l2"
    finally:
        backend.close()


def test_disk_promotion_reads_large_members_without_using_metadata_scratch():
    backend, cache, store = provider()
    try:
        entry = write(backend, cache)
        key = entry.data.keys[0]
        backend._metadata_bytes = len(store.objects[key]) - 1
        # The registered read must work even when this member exceeds SDK scratch.
        policy = replace(backend.tiers, l2=True)
        backend.replicas.policy = policy
        backend.replicas.layers = lambda keys: {k: {"l2"} for k in keys}
        backend.replicas.wait = lambda keys, tier, deadline: True
        registrations = store.register_calls
        assert backend.replicas.promote([key], float("inf"))
        assert store.register_calls == registrations + 1
    finally:
        backend.close()
