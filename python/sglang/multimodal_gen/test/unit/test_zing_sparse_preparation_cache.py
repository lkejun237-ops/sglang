# SPDX-License-Identifier: Apache-2.0
"""Compressed prefix export/restore contracts; CUDA and real games are separate."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
    fill_prepared_prefix,
)
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import allocate_layer_stores
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy
from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import inputs
from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import (  # noqa: F401
    cpu_cache,
    cpu_dma,
    dense_kv,
)


def storage():
    return allocate_layer_stores(
        512,
        num_layers=2,
        local_heads=1,
        head_dim=128,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )


def cache(*, arena=None, retain=False):
    return ZingH3KVCache(
        2,
        window_policy=ZingWindowPolicy(6, 0),
        rope_position_mode="block_relative",
        storage=arena,
        retain_prefix_layers=retain,
    )


def prefix_data(rows=130):
    g = torch.Generator().manual_seed(42)
    positions = torch.zeros((rows, 3), dtype=torch.float64)
    positions[:, 2] = torch.arange(rows)
    layers = [
        (
            torch.randn((rows, 1, 128), generator=g, dtype=torch.bfloat16),
            torch.randn((rows, 1, 128), generator=g, dtype=torch.bfloat16),
        )
        for _ in range(2)
    ]
    return positions, layers


def publish(target, positions, layers):
    blocks = torch.zeros(len(positions), dtype=torch.long)
    target.update(
        positions, blocks, "prefix", 0, layers, validation=(positions, blocks)
    )


def compressed_bytes(target):
    return [
        {
            name: value.contiguous().reshape(-1).view(torch.uint8).clone()
            for name, value in store.as_kernel_dict(store.num_blocks).items()
            if isinstance(value, torch.Tensor)
        }
        for store in target._compressed_kv_stores
    ]


def equal_compressed(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert left.keys() == right.keys()
        for name in left:
            assert torch.equal(left[name], right[name]), name


@torch.no_grad()
def test_raw_prefix_views_exist_only_for_explicit_export(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_SPARSE_THREE_TIER", "1")
    positions, layers = prefix_data()
    normal, exporter = cache(), cache(retain=True)
    publish(normal, positions, layers)
    publish(exporter, positions, layers)
    assert normal.prefix.layers == ()
    assert len(exporter.prefix.layers) == 2
    assert exporter.prefix.layers[0][0].data_ptr() != layers[0][0].data_ptr()
    assert torch.equal(exporter.prefix.layers[0][0], layers[0][0])
    equal_compressed(compressed_bytes(normal), compressed_bytes(exporter))
    exporter.clear()
    assert exporter.prefix is None
    assert all(s.num_blocks == 0 for s in exporter._compressed_kv_stores)


@torch.no_grad()
@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.usefixtures("cpu_dma")
def test_prepared_prefix_round_trip_preserves_compressed_bytes(
    monkeypatch, tmp_path, disk
):
    monkeypatch.setenv("SGLANG_ZING_SPARSE_THREE_TIER", "1")
    if disk:
        monkeypatch.setenv("SGLANG_ZING_DISK_CACHE_DIR", str(tmp_path / "cache"))
    else:
        monkeypatch.delenv("SGLANG_ZING_DISK_CACHE_DIR", raising=False)
    arena = storage()
    host = cpu_cache(arena, rows=512, page_rows=32)
    positions, layers = prefix_data()
    created = []
    expected = []

    def initialize(prepared, visual, *, storage, retain_prefix_layers):
        assert retain_prefix_layers is True
        value = cache(arena=storage, retain=retain_prefix_layers)
        publish(value, positions, layers)
        created.append(value)
        expected.extend(compressed_bytes(value))
        return NS(cache=value, close=value.clear)

    model = torch.nn.Linear(1, 1)
    stage = NS(
        block0_cache=host,
        kv_pool=None,
        packer=NS(visual_channels=1),
        _use=lambda *a: nullcontext(model),
    )
    sampler = NS(
        processor=NS(pack_inference_layout=lambda *a: {}), initialize=initialize
    )
    key = "a" * 64
    try:
        entry = fill_prepared_prefix(
            stage, sampler, inputs(), owner=object(), key=key, identity=None, request={}
        )
        assert created[0].prefix is None
        assert entry is not None
        assert torch.equal(
            dense_kv(entry.data), torch.stack([torch.stack(pair) for pair in layers])
        )
        if disk:
            assert entry.disk_task is not None and entry.disk_task.wait(10)
            host.close()
            host = cpu_cache(arena, rows=512, page_rows=32)
            entry = host.lookup(key, allow_disk=True)
            assert entry is not None
        else:
            assert host.lookup(key) is entry
        target = cache(arena=arena)
        transfer = host.restore(target, entry)
        transfer.commit(target)
        transfer.wait()
        assert target.prefix.layers == ()
        equal_compressed(expected, compressed_bytes(target))
    finally:
        host.close()


@torch.no_grad()
def test_resident_reset_keeps_only_prefix_even_if_source_shares_chunk_id(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_SPARSE_THREE_TIER", "1")
    target = cache()
    positions, layers = prefix_data()
    publish(target, positions, layers)
    original = compressed_bytes(target)
    counts = target._prefix_compressed_blocks
    ptrs = [s.k_res_fp8.data_ptr() for s in target._compressed_kv_stores]
    for store, pair in zip(target._compressed_kv_stores, layers):
        # First source can have the same -1 chunk ID as static conditioning.
        store.append_chunk(pair[0][:17], pair[1][:17], chunk_id=-1)
        assert store.num_blocks > counts[0]
    target.last_block = 2
    target.reset_history_keep_prefix()
    assert target._prefix_compressed_blocks == counts
    assert target._length == len(positions) and target.last_block == 0
    assert not target._stores and not target.committed
    assert [s.k_res_fp8.data_ptr() for s in target._compressed_kv_stores] == ptrs
    equal_compressed(original, compressed_bytes(target))


@torch.no_grad()
def test_missing_compressed_prefix_boundary_fails_closed(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_SPARSE_THREE_TIER", "1")
    target = cache()
    publish(target, *prefix_data())
    target._prefix_compressed_blocks = None
    with pytest.raises(ValueError, match="complete static KV"):
        target.reset_history_keep_prefix()


@torch.no_grad()
def test_fp8_shift_mask_kernel_and_wire_dtypes(monkeypatch):
    from sglang.multimodal_gen.runtime.models.dits import zing_h3_prefix_codec as codec

    monkeypatch.setenv("SGLANG_ZING_SPARSE_THREE_TIER", "1")
    source = cache(arena=storage())
    publish(source, *prefix_data())
    store = source._compressed_kv_stores[0]
    mask = store.as_kernel_dict()["block_is_shifted"]
    assert mask.dtype == torch.bool
    assert mask.is_contiguous()
    mask[1] = True

    rows = source.prefix.positions.shape[0]
    blob = codec.pack_layer(store, rows)
    field = next(
        f
        for f in codec.fields(rows, store.num_heads, store.head_dim)
        if f.name == "block_is_shifted"
    )
    assert field.dtype == torch.int32
    legacy_flags = blob.narrow(0, field.offset, field.size).view(torch.int32)
    assert legacy_flags.tolist() == [0, 1, 0]
    legacy_flags.copy_(torch.tensor([1, 0, 1], dtype=torch.int32))
    codec.unpack_layer(blob, store, rows)
    assert mask.tolist() == [True, False, True]


@torch.no_grad()
@pytest.mark.parametrize("required", [True, False])
def test_mooncake_transfer_rebuilds_private_compressed_prefix(monkeypatch, required):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        Block0PreparationCache,
    )
    from sglang.multimodal_gen.test.unit.realtime.test_zing_mooncake_cache import (
        MemoryStore,
    )

    monkeypatch.setenv("SGLANG_ZING_SPARSE_THREE_TIER", "1")
    arena = storage()
    source = cache(arena=arena, retain=True)
    publish(source, *prefix_data())
    expected = compressed_bytes(source)
    pool = NS(_pool=NS(_slots=[NS(kv_cache=arena)]), plan=NS(slot_rows=512))
    transport = MemoryStore()
    backend = Block0PreparationCache(
        pool,
        backend="mooncake",
        identity={"sp": {"size": 1}},
        config={"tiers": {"snapshot_bytes": 1 << 20}},
        rank=0,
        world_size=1,
        client=transport,
        list_keys=lambda: list(transport.objects),
        replicate_config_factory=NS,
    )
    try:
        assert backend.fill("b" * 64, inputs(), None, None, source, required=required)
        entry = backend.finish_fill(wait=True)
        assert entry is not None
        assert source.prefix.layers == ()
        assert len(transport.registrations) == 4  # Only the two-layer pool remains.
        source.clear()
        for slot in arena:
            slot.k.fill_(99)
            slot.v.fill_(99)
        target = cache(arena=arena)
        transfer = backend.restore(target, entry)
        transfer.commit(target)
        transfer.wait()
        equal_compressed(expected, compressed_bytes(target))
        assert target.prefix.layers == ()
        # Transport is a byte-level double; this test does not claim native
        # Mooncake TCP/RDMA leases or OSS persistence have been exercised.
    finally:
        backend.close()


def compressed_backend(arena, transport, *, compressed=True):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_compressed_view import (
        LAYOUT_VERSION,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        Block0PreparationCache,
    )

    pool = NS(_pool=NS(_slots=[NS(kv_cache=arena)]), plan=NS(slot_rows=512))
    identity = {"sp": {"size": 1}, "execution": {"attention": {"mode": "dense"}}}
    if compressed:
        identity["execution"]["attention"] = {
            "mode": "sparse_095",
            "three_tier": True,
            "compressed_layout": LAYOUT_VERSION,
        }
    return Block0PreparationCache(
        pool,
        backend="mooncake",
        identity=identity,
        config={"tiers": {"snapshot_bytes": 1 << 20}},
        rank=0,
        world_size=1,
        client=transport,
        list_keys=lambda: list(transport.objects),
        replicate_config_factory=NS,
    )


@torch.no_grad()
@pytest.mark.parametrize("required", [True, False])
def test_mooncake_fp8_round_trip_never_requantizes(monkeypatch, required):
    from sglang.multimodal_gen.runtime.models.dits import zing_h3_prefix_codec as codec
    from sglang.multimodal_gen.test.unit.realtime.test_zing_mooncake_cache import (
        MemoryStore,
    )

    monkeypatch.setenv("SGLANG_ZING_SPARSE_THREE_TIER", "1")
    arena = storage()
    source = cache(arena=arena, retain=False)
    publish(source, *prefix_data())
    assert source.prefix.layers == ()
    expected = compressed_bytes(source)
    transport = MemoryStore()
    backend = compressed_backend(arena, transport)
    if not required:
        positions, layers = prefix_data(rows=17)
        positions[:, 0] = 1
        blocks = torch.ones(17, dtype=torch.long)
        source.update(
            positions, blocks, "source", 1, layers, validation=(positions, blocks)
        )
    try:
        assert backend.kv_encoding == codec.FP8 and not backend.requires_raw_prefix
        assert backend.fill("c" * 64, inputs(), None, None, source, required=required)
        entry = backend.finish_fill(wait=True)
        assert entry.data.encoding == codec.FP8
        assert (
            entry.data.storage_bytes() < entry.data.numel() * entry.data.element_size()
        )
        assert all(
            len(transport.objects[key]) == entry.data.layer_bytes()
            for key in entry.data.keys
        )
        assert len(transport.registrations) == 4
        entry = backend.lookup("c" * 64)
        assert entry is not None and entry.data.encoding == codec.FP8
        source.clear()
        target = cache(arena=arena)

        def no_quantization(*args, **kwargs):
            pytest.fail("FP8 cache hit must not invoke quantization")

        for store in target._compressed_kv_stores:
            monkeypatch.setattr(store, "append_chunk", no_quantization)
        transfer = backend.restore(target, entry)
        transfer.commit(target)
        transfer.wait()
        assert transfer.compressed_payload is None
        assert target.prefix.layers == ()
        equal_compressed(expected, compressed_bytes(target))
        for store in target._compressed_kv_stores:
            assert store.block_valid_len[:3].tolist() == [64, 64, 2]
            assert store.block_chunk_id[:3].tolist() == [-1, -1, -1]
        assert len(transport.registrations) == 4
    finally:
        backend.close()


@torch.no_grad()
def test_dense_worker_cannot_consume_sparse_variant(monkeypatch):
    from sglang.multimodal_gen.test.unit.realtime.test_zing_mooncake_cache import (
        MemoryStore,
    )

    monkeypatch.setenv("SGLANG_ZING_SPARSE_THREE_TIER", "1")
    arena = storage()
    source = cache(arena=arena)
    publish(source, *prefix_data())
    transport = MemoryStore()
    sparse = compressed_backend(arena, transport)
    dense = compressed_backend(
        storage(), MemoryStore(transport.objects), compressed=False
    )
    try:
        assert sparse._namespace != dense._namespace
        sparse.fill("f" * 64, inputs(), None, None, source)
        assert sparse.finish_fill(wait=True) is not None
        assert dense.lookup("f" * 64) is None
        assert sparse.lookup("f" * 64) is not None
        assert dense.requires_raw_prefix
    finally:
        sparse.close()
        dense.close()


@torch.no_grad()
def test_mooncake_rejects_wrong_encoding_and_truncated_fp8(monkeypatch):
    import json

    from sglang.multimodal_gen.runtime.models.dits import zing_h3_prefix_codec as codec
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.mooncake_cache import (
        CacheTransferError,
    )
    from sglang.multimodal_gen.test.unit.realtime.test_zing_mooncake_cache import (
        MemoryStore,
    )

    monkeypatch.setenv("SGLANG_ZING_SPARSE_THREE_TIER", "1")
    arena = storage()
    source = cache(arena=arena)
    publish(source, *prefix_data())
    transport = MemoryStore()
    backend = compressed_backend(arena, transport)
    try:
        backend.fill("d" * 64, inputs(), None, None, source)
        entry = backend.finish_fill(wait=True)
        marker = backend._manifest_key("d" * 64)
        original = transport.objects[marker]
        header = json.loads(original)
        header["kv_encoding"] = codec.BF16
        transport.objects[marker] = json.dumps(header).encode()
        assert backend.lookup("d" * 64) is None
        transport.objects[marker] = original
        transport.objects[entry.data.keys[0]] = transport.objects[entry.data.keys[0]][
            :-1
        ]
        target = cache(arena=arena)
        with pytest.raises(CacheTransferError):
            backend.restore(target, entry)
        assert target.prefix is None and target._length == 0
        assert len(transport.registrations) == 4
    finally:
        backend.close()


def test_bf16_and_fp8_are_separate_worker_recipe_formats():
    from sglang.multimodal_gen.runtime.models.dits import zing_h3_prefix_codec as codec
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_compressed_view import (
        LAYOUT_VERSION,
    )

    assert (
        codec.encoding_for_identity({"execution": {"attention": {"mode": "dense"}}})
        == codec.BF16
    )
    assert (
        codec.encoding_for_identity(
            {
                "execution": {
                    "attention": {
                        "mode": "sparse_095",
                        "three_tier": True,
                        "compressed_layout": LAYOUT_VERSION,
                    }
                }
            }
        )
        == codec.FP8
    )
    with pytest.raises(ValueError, match="layout"):
        codec.encoding_for_identity(
            {
                "execution": {
                    "attention": {"three_tier": True, "compressed_layout": "unknown"}
                }
            }
        )


@torch.no_grad()
@pytest.mark.parametrize("tier", ["l2", "l3"])
def test_fp8_bytes_survive_native_recall_and_archive_paths(monkeypatch, tier):
    from dataclasses import replace

    from sglang.multimodal_gen.runtime.models.dits import zing_h3_prefix_codec as codec
    from sglang.multimodal_gen.runtime.realtime.cache_archive import CacheArchive
    from sglang.multimodal_gen.runtime.realtime.cache_tiers import NativeReplicas
    from sglang.multimodal_gen.test.unit.realtime.test_zing_cache_tiers import (
        Objects,
        policy,
        replica,
    )
    from sglang.multimodal_gen.test.unit.realtime.test_zing_mooncake_cache import (
        MemoryStore,
    )

    monkeypatch.setenv("SGLANG_ZING_SPARSE_THREE_TIER", "1")
    arena = storage()
    source = cache(arena=arena)
    publish(source, *prefix_data())
    expected = compressed_bytes(source)
    transport = MemoryStore()
    backend = compressed_backend(arena, transport)
    key = "e" * 64
    try:
        backend.fill(key, inputs(), None, None, source)
        entry = backend.finish_fill(wait=True)
        encoded = {key: transport.objects[key] for key in entry.data.keys}
        if tier == "l3":
            backend.archive = CacheArchive(
                policy(), backend._namespace, objects=Objects()
            )
            backend.bind_world(key, "world-fp8")
            result = backend.archive.submit(
                "world-fp8",
                key,
                backend._archive_manifest(entry),
                backend._export_object,
            ).result(10)
            assert result["state"] == "persisted"
            transport.objects.clear()
        else:
            states = {name: "l2" for name in transport.objects}
            transport.batch_get_replica_desc = lambda keys: {
                k: [replica(states[k])] if k in states else [] for k in keys
            }
            get = transport.get

            def promote(name):
                value = get(name)
                if value:
                    states[name] = "l1"
                return value

            transport.get = promote
            backend.tiers = replace(backend.tiers, l2=True)
            backend.replicas = NativeReplicas(transport, backend.tiers)
        recalled = backend._load_tiers(key)
        assert recalled["source_layer"] == tier
        assert recalled["entry"].data.encoding == codec.FP8
        assert all(transport.objects[k] == value for k, value in encoded.items())
        source.clear()
        target = cache(arena=arena)
        transfer = backend.restore(target, recalled["entry"])
        transfer.commit(target)
        transfer.wait()
        equal_compressed(expected, compressed_bytes(target))
        # Native promotion and object storage are doubles; actual SSD/OSS
        # throughput and durability require separate infrastructure validation.
    finally:
        backend.close()


@torch.no_grad()
@pytest.mark.parametrize("fault", ["return", "raise"])
def test_failed_native_unregister_retains_buffer_and_refuses_new_transfers(
    monkeypatch, fault
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.mooncake_cache import (
        CacheTransferError,
    )
    from sglang.multimodal_gen.test.unit.realtime.test_zing_mooncake_cache import (
        MemoryStore,
    )

    monkeypatch.setenv("SGLANG_ZING_SPARSE_THREE_TIER", "1")
    arena = storage()
    source = cache(arena=arena)
    publish(source, *prefix_data())
    transport = MemoryStore()
    backend = compressed_backend(arena, transport)
    try:
        backend.fill("1" * 64, inputs(), None, None, source)
        entry = backend.finish_fill(wait=True)
        pools = set(transport.registrations)
        original = transport.unregister_buffer

        def fail_owned(ptr):
            if ptr not in pools:
                if fault == "raise":
                    raise RuntimeError("injected unregister failure")
                return -1
            return original(ptr)

        monkeypatch.setattr(transport, "unregister_buffer", fail_owned)
        with pytest.raises(CacheTransferError):
            backend.restore(cache(arena=arena), entry)
        assert len(backend._registered) == 6  # Both temporary buffers remain owned.
        registrations = transport.register_calls
        with pytest.raises(CacheTransferError):
            backend.restore(cache(arena=arena), entry)
        assert transport.register_calls == registrations
        monkeypatch.setattr(transport, "unregister_buffer", original)
    finally:
        backend.close()
    assert transport.closed and not backend._registered
