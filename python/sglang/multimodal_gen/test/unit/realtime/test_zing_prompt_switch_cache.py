# SPDX-License-Identifier: Apache-2.0
"""Shared startup/switch cache: real CPU KV arithmetic and transport stand-ins."""

import json
import time
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.conditioning.worker import WorkerConditioningState
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    source_session,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
    preparation_key,
)
from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import inputs
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model as model_fixture


@pytest.fixture
def model():
    return model_fixture.__wrapped__()


@pytest.fixture
def cpu_dma(monkeypatch):
    """Run real tensor copies/KV commits, replacing only CUDA transport on CPU."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        preparation_cache,
    )

    stream = NS(wait_stream=Mock(), wait_event=Mock(), synchronize=Mock())
    events = []

    class Event:
        def __init__(self, **kwargs):
            self.synchronized = False
            events.append(self)

        def record(self):
            pass

        def synchronize(self):
            self.synchronized = True

        def query(self):
            return self.synchronized

    empty = torch.empty

    def allocate(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(
        preparation_cache, "coordinate_validation", lambda value: [value]
    )
    return events, stream


def cpu_cache(storage, *, entries=6, rows=32):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        Block0PreparationCache,
    )

    pool = NS(
        _pool=NS(_slots=[NS(kv_cache=storage)]),
        plan=NS(config=NS(max_prefix_rows=rows)),
    )
    return Block0PreparationCache(pool, "overlap", entries=entries)


def test_lru_reuses_preallocated_slots_and_waits_for_fill(cpu_dma):
    stores = [NS(k=torch.zeros(8, 2, 3), v=torch.zeros(8, 2, 3)) for _ in range(2)]
    owner = cpu_cache(stores, entries=4, rows=8)
    pointers = [owner.data[i].data_ptr() for i in range(4)]

    def fill(key, value, rows=3):
        cache = NS(
            prefix=NS(
                positions=torch.zeros(rows, 3),
                layers=[
                    (torch.full((rows, 2, 3), value), torch.full((rows, 2, 3), -value))
                    for _ in stores
                ],
            ),
            _model_signature="model",
        )
        owner.fill(key, inputs(), "condition", "request", cache)
        return owner.lookup(key)

    for index, key in enumerate("ABCD"):
        entry = fill(key, float(index + 1))
        assert entry.slot == index
        assert cpu_dma[0][-1].synchronized
    assert owner.lookup("A").data[0, 0, 0, 0, 0] == 1
    entry = fill("E", 5.0, rows=7)
    assert entry.slot == 1  # B is oldest after A was accessed.
    assert owner.lookup("B") is None
    assert owner.lookup("A") is not None
    assert owner.evictions == 1
    assert torch.equal(entry.data[:, 0], torch.full_like(entry.data[:, 0], 5))
    assert torch.equal(entry.data[:, 1], torch.full_like(entry.data[:, 1], -5))
    assert [owner.data[i].data_ptr() for i in range(4)] == pointers
    # Repeated fill neither allocates nor evicts an existing ready entry.
    fill("E", 99.0)
    assert owner.lookup("E").data[0, 0, 0, 0, 0] == 99
    assert owner.evictions == 1


@pytest.mark.parametrize("entries", [0, -1, True, 1.5])
def test_reject_invalid_entry_capacity(entries):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        Block0PreparationCache,
    )

    with pytest.raises(ValueError, match="positive integer"):
        Block0PreparationCache(None, "overlap", entries=entries)


def test_partial_sp_hit_falls_back(cpu_dma, monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        preparation_cache,
    )

    owner = cpu_cache([NS(k=torch.zeros(8, 2, 3))])
    owner.entries["A"] = object()
    monkeypatch.setattr(
        preparation_cache, "coordinate_validation", lambda value: [value, ("A", False)]
    )
    assert owner.lookup("A") is None
    assert owner.hits == 0 and owner.misses == 1


def test_post_switch_fill_waits_for_new_epoch_output():
    state = source_session.ZingSourceStreamState()
    prefix = object()
    state.sampler = NS(cache=NS(prefix=prefix), committed_pixel=21)
    state.block0_cache = NS(fill=Mock())
    state.block0_key = "B"
    state.block0_inputs = inputs()
    state.block0_condition_identity = "B-condition"
    state.block0_request = object()
    state.ledger = NS(output_frames=17)
    state._remember_block0(prefix)
    state._fill_block0_after_output()
    state.ledger.output_frames = 21  # Draining only old prompt output.
    state._fill_block0_after_output()
    state.block0_cache.fill.assert_not_called()
    state.ledger.output_frames = 22
    state._fill_block0_after_output()
    state._fill_block0_after_output()
    state.block0_cache.fill.assert_called_once()
    assert state.block0_cache.fill.call_args.args[0] == "B"


def test_key_reuses_accepted_media_after_uri_changes(tmp_path):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        preparation_request,
    )

    path = tmp_path / "ref.png"
    path.write_bytes(b"accepted")
    codecs = NS(remote_read_fs=None)
    raw = dict(prompt="A", media=[dict(type="image", uri=str(path))])
    accepted = preparation_request(codecs, raw)
    config = dict(width=32, height=32, fps=24)
    a = preparation_key(codecs, accepted, None, config, {})
    assert a == preparation_key(codecs, raw, None, config, {})
    path.write_bytes(b"replaced")
    assert preparation_key(codecs, raw, None, config, {}) != a
    assert preparation_key(codecs, {**accepted, "prompt": "A"}, None, config, {}) == a
    assert preparation_key(codecs, {**accepted, "prompt": "B"}, None, config, {}) != a


@torch.no_grad()
@pytest.mark.parametrize("use_drain", [False, True])
def test_startup_a_switch_b_a_reuses_kv_and_preserves_outputs(
    model, cpu_dma, use_drain
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
        ZingH3Packer,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
        ZingH3Sampler,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance
    from sglang.multimodal_gen.test.unit.test_zing_h3_kv_pool_cache import (
        _pointers,
        _storage,
    )

    base = inputs()
    base.visual = {}
    base.positive = dict(
        context=torch.randn(3, 12).bfloat16(), row_role=torch.zeros(3, dtype=torch.long)
    )
    positive_b = dict(
        context=torch.cat(
            (base.positive["context"] + 0.25, base.positive["context"][:2] - 0.1)
        ),
        row_role=torch.zeros(5, dtype=torch.long),
    )
    base.visual_shape = (30, 2, 2)
    source = torch.randn(1, 2, 30, 2, 2)
    noise = torch.randn_like(source)
    conditions = {"A": base.positive, "B": positive_b}

    def run(cached):
        packer = ZingH3Packer(
            model.arch, num_frame_per_block=[3, 2], split_first_latent=True
        )
        sampler = ZingH3Sampler(
            model,
            packer,
            solver="few_step",
            denoising_step_list=[1000, 500],
            window_chunks=6,
            sink_chunks=3,
        )
        stores = _storage(model, 256)
        pointers = _pointers(stores)
        cache = cpu_cache(stores) if cached else None
        ready = SourceReadySession(
            sampler,
            base,
            seed=123,
            max_pending_chunks=1,
            max_pending_bytes=1 << 20,
            acquire_storage=lambda *_: stores,
            initial_prompt_event_id=0,
            prompt_joint=use_drain,
        )
        state = source_session.ZingSourceStreamState()
        state.sampler, state.inputs = ready, base
        state.ledger = NS(output_frames=0)
        state.block0_cache = cache
        state.block0_inputs = base
        state.block0_condition_identity = "A-condition"
        state.block0_request = "pool-request"
        state.block0_input_request = dict(prompt="A", media=[], conditions=[])
        codecs = NS(remote_read_fs=None)
        config = dict(width=32, height=32, fps=24)
        state.block0_key_args = (codecs, None, config, {})
        state.block0_key = preparation_key(
            codecs, state.block0_input_request, None, config, {}
        )
        if cached:
            ready.capture_initial_prefix = state._remember_block0
            ready.prepare_prompt_prefix = state._prepare_block0_switch
        condition_state = WorkerConditioningState()
        stage = NS(
            codecs=codecs,
            encode_condition_for_batch=Mock(
                side_effect=lambda batch, prompt, media: conditions[prompt]
            ),
            kv_pool=NS(acquire=Mock()),
        )
        state._admit_prompt_update = Mock(return_value="pool-request")
        results, snapshots, histories = [], [], []
        prefix_calls = 0
        forward = sampler._forward

        def counted(packed, cache, phase):
            nonlocal prefix_calls
            prefix_calls += phase == "prefix"
            return forward(packed, cache, phase)

        sampler._forward = counted
        for i in range(12):
            state._fill_block0_after_output()
            applied = ready.applied_prompt
            if applied is not None and applied.inputs is not None:
                state.inputs = applied.inputs
            if ready.prompt_switch_ready:
                # Switch after complete codec cycles so every candidate applies.
                prompt = "B" if (ready.committed_latent // 5) % 2 else "A"
                batch = NS(
                    extra={},
                    sampling_params=NS(prompt=prompt),
                    realtime_event_id=i,
                    realtime_prompt_version=i,
                    session=NS(get_or_create_state=lambda _: condition_state),
                )
                assert state._update_prompt(stage, batch) is None
            start, end = ready.next_latent, ready.next_latent + ready.next_chunk_size
            ready.append(
                source[:, :, start:end],
                noise[:, :, start:end],
                provenance=fixture_provenance(1, end, start, end),
            )
            rng = torch.get_rng_state().clone()
            result = list(ready.drain_ready())[0] if use_drain else ready.step()
            assert torch.equal(rng, torch.get_rng_state())
            results.append(
                (result.visual.clone(), result.prompt_event_id, result.prompt_version)
            )
            snapshots.append(
                [(k.clone(), v.clone()) for k, v in ready.cache.prefix.layers]
            )
            histories.append(
                [
                    tuple(value.clone() for value in ready.cache.history(layer))
                    for layer in range(len(stores))
                ]
            )
            assert _pointers(stores) == pointers
            state.ledger.output_frames = ready.committed_pixel
        assert [row[1] for row in results] == [0, 0, 0, 3, 3, 5, 5, 7, 7, 9, 9, 11]
        state._fill_block0_after_output()
        if cached:
            cache.finish_fill()
            assert len(cache.entries) == 2 and cache.evictions == 0
            assert stage.encode_condition_for_batch.call_count == 1  # Only first B.
            assert prefix_calls == 2  # Startup A and cold B; all returns restore.
            assert ready.cache.stats["store_allocations"] == 0
        else:
            assert prefix_calls == 6
        stage.kv_pool.acquire.assert_not_called()  # Switch keeps the original slot.
        ready.dispose()
        return results, snapshots, histories

    expected, actual = run(False), run(True)
    for left, right in zip(expected[0], actual[0]):
        assert torch.equal(left[0], right[0])
        assert left[1:] == right[1:]
    for expected_layers, actual_layers in zip(
        expected[1] + expected[2], actual[1] + actual[2]
    ):
        for expected_kv, actual_kv in zip(expected_layers, actual_layers):
            assert all(torch.equal(a, b) for a, b in zip(expected_kv, actual_kv))


def test_message_request_and_prompt_switch_use_the_same_key():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        preparation_request,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_media_contract import (
        codecs as media_codecs,
    )

    codecs = media_codecs.__wrapped__()
    raw = dict(
        messages=[
            dict(role="user", content=[dict(type="text", text="A")]),
            dict(
                role="target",
                content=[
                    dict(type="video", output=dict(frames=17, height=32, width=32))
                ],
                conditions=[],
            ),
        ]
    )
    config = dict(width=32, height=32, fps=24)
    initial = preparation_key(codecs, raw, None, config, {})
    accepted = preparation_request(codecs, raw)
    assert initial == preparation_key(
        codecs, {**accepted, "prompt": "A"}, None, config, {}
    )
    assert initial != preparation_key(
        codecs, {**accepted, "prompt": "B"}, None, config, {}
    )


@pytest.mark.parametrize("failure", ["media", "admission", "encode"])
def test_failed_update_drops_pending_but_preserves_active_cache(tmp_path, failure):
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    state = source_session.ZingSourceStreamState()
    active = inputs()
    state.inputs = active
    state.block0_key = "active-A"
    state.block0_input_request = dict(prompt="A", media=[], conditions=[])
    state.pending_block0 = "older-pending"
    state.sampler = NS(discard_pending_prompt=Mock(), update_prompt=Mock())
    codecs = NS(remote_read_fs=None)
    state.block0_key_args = (codecs, None, dict(width=32, height=32, fps=24), {})
    state.block0_cache = NS(
        lookup=Mock(
            return_value=NS(inputs=active, conditioning_identity="cached")
            if failure == "admission"
            else None
        )
    )
    stage = NS(
        codecs=codecs,
        encode_condition_for_batch=Mock(
            side_effect=ConditionError("CONDITION_INVALID", "Encoding failed")
        ),
    )
    state._admit_prompt_update = Mock(
        side_effect=ConditionError("CONDITION_INVALID", "This update is too large")
    )
    command = dict(prompt="B", event_id=3)
    if failure == "media":
        command["references"] = [
            dict(type="image", id="new", uri=str(tmp_path / "missing.png"))
        ]
    batch = NS(
        extra={"zing_prompt_update": command},
        sampling_params=NS(prompt="A"),
        realtime_event_id=0,
    )
    notice = state._update_prompt(stage, batch)
    assert notice == dict(code="CONDITION_INVALID", event_id=3)
    state.sampler.discard_pending_prompt.assert_called_once()
    state.sampler.update_prompt.assert_not_called()
    assert state.pending_block0 is None
    assert state.inputs is active and state.block0_key == "active-A"
    if failure == "media":
        state.block0_cache.lookup.assert_not_called()
    if failure == "admission":
        stage.encode_condition_for_batch.assert_not_called()


def disk_cache(tmp_path, monkeypatch, **env):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
        Block0DiskCache,
    )

    monkeypatch.setenv("SGLANG_ZING_DISK_CACHE_DIR", str(tmp_path / "disk"))
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    return Block0DiskCache()


def disk_commit(cache, key, value=1.0, rows=3, layers=2, rank=0):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
        Block0DiskCache,
    )

    assert isinstance(cache, Block0DiskCache)
    kv = torch.full((layers, 2, rows, 2, 2), value, dtype=torch.bfloat16)
    tmp = cache.new_tmp_dir(key)
    path = cache.save_rank_kv(key, rank, kv, tmp)
    conditions = {
        "identity": f"{key}-condition",
        "inputs": torch.full((rows,), value, dtype=torch.bfloat16),
        "positions": torch.arange(rows).reshape(rows, 1).repeat(1, 3),
    }
    manifest = {
        "geometry": {"prefix_rows": rows},
        "prompt_preview": f"prompt-{key}",
    }
    entry = cache.commit_entry(key, conditions, manifest, {rank: path}, tmp)
    assert entry is not None
    return kv, entry


def test_disk_cache_roundtrip_complete_marker_and_index(tmp_path, monkeypatch):
    cache = disk_cache(tmp_path, monkeypatch)
    key = "abcdef012345"
    kv, entry = disk_commit(cache, key, value=2.5)

    folder = cache.cache_dir / key[:2] / key
    assert folder.is_dir()
    assert (folder / ".complete").is_file()
    assert not any(cache.tmp_root.iterdir())
    manifest = json.loads((folder / "manifest.json").read_text())
    for name in (
        "early_key",
        "created_at",
        "last_accessed_at",
        "size_bytes",
        "sp_world_size",
        "geometry",
        "prompt_preview",
    ):
        assert name in manifest
    assert manifest["early_key"] == key
    assert manifest["sp_world_size"] == 1
    assert manifest["size_bytes"] == entry["size_bytes"]

    index = json.loads((cache.cache_dir / "lru_index.json").read_text())
    assert index["version"] == 1
    assert index["stats"]["total_entries"] == 1
    assert index["stats"]["total_bytes"] == entry["size_bytes"]
    assert index["stats"]["max_entries"] == cache.max_entries
    assert index["stats"]["max_bytes"] == cache.max_bytes
    assert [item["key"] for item in index["entries"]] == [key]

    hit = cache.lookup(key, 0)
    assert hit is not None
    restored = torch.zeros_like(kv)
    assert hit.load_kv_into(restored) is restored
    assert torch.equal(restored, kv)
    assert hit.conditions["identity"] == f"{key}-condition"
    assert hit.manifest["early_key"] == key
    accessed = json.loads((cache.cache_dir / "lru_index.json").read_text())
    assert (
        accessed["entries"][0]["last_accessed_at"]
        >= index["entries"][0]["last_accessed_at"]
    )


def test_disk_index_self_heals_when_missing_or_corrupt(tmp_path, monkeypatch):
    cache = disk_cache(tmp_path, monkeypatch)
    key = "012345abcdef"
    disk_commit(cache, key)
    index_path = cache.cache_dir / "lru_index.json"

    index_path.unlink()
    assert cache.lookup(key, 0) is not None
    assert index_path.is_file()

    index_path.write_text("{not json")
    assert cache.lookup(key, 0) is not None
    rebuilt = json.loads(index_path.read_text())
    assert rebuilt["version"] == 1
    assert [item["key"] for item in rebuilt["entries"]] == [key]


def test_disk_lru_evicts_by_entries_and_bytes(tmp_path, monkeypatch):
    cache = disk_cache(tmp_path, monkeypatch, SGLANG_ZING_DISK_CACHE_MAX_ENTRIES=2)
    for key in ("a1", "b2", "c3"):
        disk_commit(cache, key)
        time.sleep(0.002)
    keys = [
        item["key"]
        for item in json.loads((cache.cache_dir / "lru_index.json").read_text())[
            "entries"
        ]
    ]
    assert keys == ["b2", "c3"]
    assert not (cache.cache_dir / "a1"[:2] / "a1").exists()
    assert (cache.cache_dir / "c3"[:2] / "c3" / ".complete").is_file()

    byte_cache = disk_cache(tmp_path / "bytes", monkeypatch)
    _, first = disk_commit(byte_cache, "d4")
    byte_cache.max_bytes = int(first["size_bytes"] * 1.5)
    disk_commit(byte_cache, "e5")
    keys = [
        item["key"]
        for item in json.loads((byte_cache.cache_dir / "lru_index.json").read_text())[
            "entries"
        ]
    ]
    assert keys == ["e5"]
    assert not (byte_cache.cache_dir / "d4"[:2] / "d4").exists()


def test_l2_disk_hit_promotes_into_l1_and_matches_values(
    tmp_path, monkeypatch, cpu_dma
):
    monkeypatch.setenv("SGLANG_ZING_DISK_CACHE_DIR", str(tmp_path / "disk"))
    stores = [NS(k=torch.zeros(8, 2, 3), v=torch.zeros(8, 2, 3)) for _ in range(2)]
    owner = cpu_cache(stores, entries=2, rows=8)
    assert owner.disk_cache.is_enabled()
    prefix = NS(
        positions=torch.zeros(3, 3),
        layers=[
            (torch.full((3, 2, 3), 7.0), torch.full((3, 2, 3), -7.0)) for _ in stores
        ],
    )
    live = NS(prefix=prefix, _model_signature="model")
    owner.fill("L2", inputs(), "L2-condition", "L2-request", live)
    owner.finish_fill()
    owner.wait_for_disk()
    old = owner.entries["L2"]

    fresh = cpu_cache(stores, entries=2, rows=8)
    assert fresh.lookup("L2") is not None
    promoted = fresh.entries["L2"]
    assert fresh.hits == 1 and fresh.misses == 0
    assert promoted.slot == 0
    assert promoted.conditioning_identity == "L2-condition"
    assert torch.equal(promoted.data, old.data)


@torch.no_grad()
@pytest.mark.parametrize("legacy_signature", [False, True])
@pytest.mark.parametrize("reuse_cache", [False, True])
def test_l2_restore_rebinds_process_local_signature(
    model, cpu_dma, tmp_path, monkeypatch, legacy_signature, reuse_cache
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
    from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy
    from sglang.multimodal_gen.test.unit.test_zing_h3_kv_pool_cache import _storage

    monkeypatch.setenv("SGLANG_ZING_DISK_CACHE_DIR", str(tmp_path / "disk"))
    stores = _storage(model, 32)
    positions = torch.zeros(3, 3)
    blocks = torch.zeros(3, dtype=torch.long)
    layers = [(torch.randn_like(s.k[:3]), torch.randn_like(s.v[:3])) for s in stores]
    source = ZingH3KVCache(
        len(stores),
        storage=stores,
        window_policy=ZingWindowPolicy(window_chunks=6, sink_chunks=3),
    )
    source.update(positions, blocks, "prefix", 0, layers=layers)
    model._bind_cache(source, positions)
    old_signature = source._model_signature
    owner = cpu_cache(stores, entries=2)
    owner.fill("restart", inputs(), "condition", "pool-request", source)
    owner.finish_fill()
    owner.wait_for_disk()
    path = owner.disk_cache._entry_dir("restart") / "conditions.pt"
    conditions = torch.load(path, weights_only=False)
    if legacy_signature:
        conditions["model_signature"] = old_signature
        torch.save(conditions, path)
    else:
        assert "model_signature" not in conditions

    restarted = model_fixture.__wrapped__()  # Same weights, new parameter identities.
    assert all(
        torch.equal(a, b) for a, b in zip(model.parameters(), restarted.parameters())
    )
    destination = ZingH3KVCache(
        len(stores),
        storage=_storage(restarted, 32),
        window_policy=ZingWindowPolicy(window_chunks=6, sink_chunks=3),
    )
    if reuse_cache:
        restarted._bind_cache(destination, torch.empty(0, 3))
        destination.clear()  # Existing prompt-switch boundary.
    fresh = cpu_cache(destination.storage, entries=2)
    promoted = fresh.lookup("restart")
    assert promoted is not None
    fresh.restore(destination, promoted).commit(destination)
    restarted._bind_cache(destination, positions)
    assert destination._model_signature != old_signature
    for expected, actual in zip(layers, destination.prefix.layers):
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    # A later in-process model mutation must still be rejected with live KV.
    next(restarted.parameters()).add_(1)
    with pytest.raises(ValueError, match="model changed within a cached session"):
        restarted._bind_cache(destination, positions)


def test_disk_cache_fails_open_on_invalid_path(tmp_path, monkeypatch):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("x")
    monkeypatch.setenv("SGLANG_ZING_DISK_CACHE_DIR", str(blocked))
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
        Block0DiskCache,
    )

    cache = Block0DiskCache()
    assert not cache.is_enabled()
    assert cache.lookup("safe", 0) is None
    assert cache.new_tmp_dir("safe") is None
    assert cache.save_rank_kv("safe", 0, torch.zeros(2), None) is None
    assert cache.commit_entry("safe", {}, {}, None, None) is None


def test_dispose_drains_transfers_before_releasing_gpu_slot():
    state = source_session.ZingSourceStreamState()
    actions = []
    state.block0_cache = NS(
        finish_fill=lambda: actions.append("fill"),
        mode="overlap",
        data=torch.empty(0),
        entries={},
        hits=0,
        misses=0,
        evictions=0,
    )
    state.block0_restore = NS(
        end=NS(synchronize=lambda: actions.append("restore")),
        begin=NS(elapsed_time=lambda _: 0),
        submit_ms=0,
        ready_at_consumer=False,
        lead_ms=0,
    )
    state._dispose_states = lambda: actions.append("states")
    state.kv_lease = NS(release=lambda: actions.append("release"))
    state.dispose()
    assert actions == ["fill", "restore", "states", "release"]
    assert state.kv_lease is None and state.block0_cache is None
