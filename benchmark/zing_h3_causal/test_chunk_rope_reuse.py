# SPDX-License-Identifier: Apache-2.0
"""Exact cache oracle; runs unchanged on CPU and CUDA via ZING_TEST_DEVICE.

A temporal chunk groups source + target/audio for one latent time interval.
RoPE reuse tracks source and target parts separately, even after their commit
merges them. These fixtures use two token rows per part, not checkpoint sizes;
see CHUNK_ROPE_REUSE.md for the real [1, 2, 2] latent-frame cycle.
"""

import os

import pytest
import torch
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import allocate_layer_stores
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

DEVICE = os.environ.get("ZING_TEST_DEVICE", "cpu")


def positions(t):
    return torch.tensor([[t, 0, 0], [t, 1, 0]], device=DEVICE)


def kv(p):
    k = torch.arange(p.shape[0] * 8, device=DEVICE).reshape(-1, 1, 8).float()
    return k, k + 10


def commit(cache, t, block, phase):
    p = positions(t)
    if phase == "final":
        cache.update(p, torch.full((2,), block, device=DEVICE), "active", 1)
    cache.update(
        p,
        torch.full((2,), block, device=DEVICE),
        phase,
        0 if phase == "prefix" else 1,
        [kv(p)],
    )


def view(cache, t, override=None, fail=False, batches=None):
    current, history = cache.attention_positions(positions(t))
    if override is not None:
        history = override(history)
    cache.bind_attention_positions(history)
    k, v = kv(current)
    rebuilt = []

    def coeff(p):
        return p.sum(-1).float().reshape(-1, 1, 1)

    def rotate(k, r, out):
        rebuilt.append(k.shape[0])
        out.copy_(k + r)

    def write(k, v, out_k, out_v):
        out_k.copy_(k + coeff(current))
        out_v.copy_(v)
        if fail:
            raise RuntimeError("Injected tail failure")

    def batch(xs, coefficients, out):
        if batches is not None:
            batches.append([x.shape[0] for x in xs])
        offset = 0
        for x, r in zip(xs, coefficients):
            rotate(x, r, out[offset : offset + x.shape[0]])
            offset += x.shape[0]

    keys, values = cache.raw_attention_view(
        0, k, v, coeff, rotate, write, rotate_batch=batch
    )
    hist = cache.history(0)
    if hist is not None:
        assert torch.equal(
            keys, torch.cat((hist[0] + coeff(history), k + coeff(current)))
        )
        assert torch.equal(values, torch.cat((hist[1], v)))
    return rebuilt


def cache(monkeypatch, window=4, mode="block_relative"):
    monkeypatch.setenv("SGLANG_ZING_CHUNK_ROPE_REUSE", "1")
    return ZingH3KVCache(
        1,
        window_policy=ZingWindowPolicy(
            window_chunks=window, sink_chunks=0 if window is None else 1
        ),
        rope_position_mode=mode,
    )


@torch.inference_mode()
def test_append_same_length_coordinate_change_and_failure(monkeypatch):
    c = cache(monkeypatch)
    commit(c, 0, 0, "prefix")
    commit(c, 10, 2, "final")
    commit(c, 20, 4, "final")
    assert sum(view(c, 30)) == 6
    assert not view(c, 30)

    def middle(p):
        p = p.clone()
        p[2:4, 0] += 1
        return p

    assert sum(view(c, 30, middle)) == 2
    assert sum(view(c, 30)) == 2
    with pytest.raises(RuntimeError, match="Injected"):
        view(c, 30, fail=True)
    assert 0 not in c._rotated_store_signatures
    assert sum(view(c, 30)) == 6
    c.clear()
    assert not c._chunk_geometry and not c._rotated_stores
    assert not list(c.owned_tensors())


@pytest.mark.parametrize("mode", ["absolute", "block_relative"])
@torch.inference_mode()
def test_window_repeated_eviction_and_prefix_replace(monkeypatch, mode):
    c = cache(monkeypatch, mode=mode)
    commit(c, 0, 0, "prefix")
    for i in range(1, 12):
        c.prepare_chunk(2 * i)
        view(c, 100 * i)
        commit(c, 100 * i, 2 * i - 1, "source")
        view(c, 100 * i)
        assert not view(c, 100 * i)
        commit(c, 100 * i, 2 * i, "final")
        view(c, 100 * i + 1)
        assert not view(c, 100 * i + 1)
    donor = cache(monkeypatch, mode=mode)
    commit(donor, 7, 0, "prefix")
    c.replace_prefix(donor.prefix)
    donor.clear()
    view(c, 1200)
    assert not view(c, 1200)


@torch.inference_mode()
def test_raw_replacement_and_geometry_mutation(monkeypatch):
    c = cache(monkeypatch)
    commit(c, 0, 0, "prefix")
    commit(c, 10, 2, "final")
    view(c, 20)
    old_k, old_v = c.committed[0].layers[0]
    c.committed[0].layers = ((old_k.clone() + 1, old_v),)
    assert sum(view(c, 20)) == 2
    c.committed[0].positions[:, 1] += 1
    assert sum(view(c, 20)) == 2


@torch.inference_mode()
def test_same_anchor_prefix_replace_rebuilds_only_prefix(monkeypatch):
    c = cache(monkeypatch)
    commit(c, 0, 0, "prefix")
    commit(c, 10, 2, "final")
    view(c, 20)
    donor = cache(monkeypatch)
    commit(donor, 0, 0, "prefix")
    donor.prefix.layers[0][0].add_(1)
    c.replace_prefix(donor.prefix)
    donor.clear()
    assert sum(view(c, 20)) == 2


@torch.no_grad()
def test_versioned_raw_inplace_write_never_reuses_stale_rotation(monkeypatch):
    c = cache(monkeypatch)
    commit(c, 0, 0, "prefix")
    commit(c, 10, 2, "final")
    view(c, 20)
    c.committed[0].layers[0][0].add_(5)
    assert sum(view(c, 20)) > 0


@torch.inference_mode()
def test_failed_rotation_unpublishes_even_unchanged_chunks(monkeypatch):
    c = cache(monkeypatch)
    commit(c, 0, 0, "prefix")
    commit(c, 10, 2, "final")
    view(c, 20)
    current, history = c.attention_positions(positions(20))
    history[2:, 0] += 1
    c.bind_attention_positions(history)
    k, v = kv(current)

    def fail(k, r, out):
        out.fill_(float("nan"))
        raise RuntimeError("Injected rotation failure")

    with pytest.raises(RuntimeError, match="Injected"):
        c.raw_attention_view(0, k, v, lambda p: p, fail, lambda *a: None)
    assert 0 not in c._rotated_store_signatures
    assert sum(view(c, 20)) == 4


def test_full_history_absolute_preserves_single_store(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_CHUNK_ROPE_REUSE", "1")
    c = ZingH3KVCache(1)
    assert c.single_store and not c.raw_store


@torch.no_grad()
def test_owned_tail_writes_do_not_invalidate_versioned_history(monkeypatch):
    c = cache(monkeypatch)
    commit(c, 0, 0, "prefix")
    commit(c, 10, 2, "final")
    assert sum(view(c, 20)) == 4
    assert not view(c, 20)
    commit(c, 20, 3, "source")
    assert sum(view(c, 20)) == 2
    assert not view(c, 20)
    # An external edit followed by an owned write must still be detected.
    c.prefix.layers[0][0].add_(3)
    c.stage_raw_k(0, kv(positions(20))[0])
    assert sum(view(c, 20)) == 6


@torch.inference_mode()
def test_inference_owned_raw_is_versioned_and_detects_external_edits(monkeypatch):
    c = cache(monkeypatch)
    commit(c, 0, 0, "prefix")
    commit(c, 10, 2, "final")
    view(c, 20)
    assert not torch.is_inference(c._stores[0][0])
    c.prefix.layers[0][0].add_(9)
    assert sum(view(c, 20)) == 4


@torch.no_grad()
def test_source_identity_survives_target_merge_without_grow(monkeypatch):
    c = cache(monkeypatch, window=None)
    c.reserve_rows = 64
    commit(c, 0, 0, "prefix")
    commit(c, 10, 1, "source")
    view(c, 10)
    commit(c, 10, 2, "final")
    assert sum(view(c, 20)) == 2
    assert not view(c, 20)


@pytest.mark.parametrize("mode", ["absolute", "block_relative"])
@torch.inference_mode()
def test_pooled_chunk_reuse_keeps_slot_and_tracks_writes(monkeypatch, mode):
    monkeypatch.setenv("SGLANG_ZING_CHUNK_ROPE_REUSE", "1")
    storage = allocate_layer_stores(
        32,
        num_layers=1,
        local_heads=1,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device(DEVICE),
    )
    c = ZingH3KVCache(
        1,
        window_policy=ZingWindowPolicy(window_chunks=4, sink_chunks=1),
        rope_position_mode=mode,
        storage=storage,
    )
    commit(c, 0, 0, "prefix")
    for i in range(1, 10):
        c.prepare_chunk(2 * i)
        commit(c, 10 * i, 2 * i, "final")
        view(c, 10 * i + 1)
        assert not view(c, 10 * i + 1)
    assert c._rotated_stores[0].data_ptr() == storage[0].rotated_k.data_ptr()
    assert c.stats["store_allocations"] == c.stats["rotated_allocations"] == 0
    assert not torch.is_inference(storage[0].k)
    c.prefix.layers[0][0].add_(1)
    assert view(c, 91)
    assert not view(c, 91)
    # Prefix scratch writes are cache-owned and must not invalidate history.
    k, v = kv(positions(100))
    c.stage_prefix_kv(0, k, v, 0, 1)
    assert not view(c, 91)
    donor = cache(monkeypatch, mode=mode)
    commit(donor, 7, 0, "prefix")
    c.committed[0].layers[0][0].add_(2)
    c.replace_prefix(donor.prefix)
    view(c, 91)
    assert not view(c, 91)
    assert c._rotated_stores[0].data_ptr() == storage[0].rotated_k.data_ptr()


@torch.no_grad()
def test_grouped_contiguous_misses_preserve_middle_hit(monkeypatch):
    c = cache(monkeypatch, window=None)
    c.reserve_rows = 64
    for i in range(5):
        commit(c, i * 10, i * 2, "prefix" if i == 0 else "final")
    batches = []
    assert sum(view(c, 60, batches=batches)) == 10
    assert batches == [[2, 2, 2, 2, 2]]
    batches.clear()
    assert not view(c, 60, batches=batches)
    assert not batches
    middle = c._rotated_stores[0][4:6].clone()

    def change(p):
        p = p.clone()
        p[:4, 0] += 1
        p[6:, 0] += 1
        return p

    assert sum(view(c, 60, override=change, batches=batches)) == 8
    assert batches == [[2, 2], [2, 2]]
    assert torch.equal(c._rotated_stores[0][4:6], middle)


@torch.no_grad()
def test_grouped_failure_unpublishes_all_ranges(monkeypatch):
    c = cache(monkeypatch)
    commit(c, 0, 0, "prefix")
    commit(c, 10, 2, "final")
    current, history = c.attention_positions(positions(20))
    c.bind_attention_positions(history)
    k, v = kv(current)

    def fail(xs, tables, out):
        out.fill_(float("nan"))
        raise RuntimeError("Injected grouped failure")

    with pytest.raises(RuntimeError, match="Injected grouped"):
        c.raw_attention_view(
            0,
            k,
            v,
            lambda p: p.sum(-1).float().reshape(-1, 1, 1),
            None,
            None,
            rotate_batch=fail,
        )
    assert 0 not in c._rotated_store_signatures
    batches = []
    assert sum(view(c, 20, batches=batches)) == 4
    assert batches == [[2, 2]]


@torch.no_grad()
def test_store_growth_rebuilds_changed_storage(monkeypatch):
    c = cache(monkeypatch, window=None)
    c.reserve_rows = 0
    commit(c, 0, 0, "prefix")
    commit(c, 10, 2, "final")
    view(c, 20)
    old = c._rotated_stores[0]
    old_rows = old[:4].clone()
    # A larger current tail reserves new raw storage and rebinds history.
    k, v = kv(positions(20))
    c._reserve_store(0, old.shape[0] + 64, k, v)
    batches = []
    view(c, 20, batches=batches)
    assert c._rotated_stores[0].data_ptr() != old.data_ptr()
    assert torch.equal(c._rotated_stores[0][:4], old_rows)
    assert batches == [[2, 2]]


@torch.no_grad()
def test_later_grouped_failure_discards_previous_publication(monkeypatch):
    c = cache(monkeypatch, window=None)
    for i in range(3):
        commit(c, i * 10, i * 2, "prefix" if i == 0 else "final")
    view(c, 40)
    current, history = c.attention_positions(positions(40))
    changed = history.clone()
    changed[:2, 0] += 1
    changed[4:, 0] += 1
    c.bind_attention_positions(changed)
    k, v = kv(current)
    calls = []

    def coeff(p):
        return p.sum(-1).float().reshape(-1, 1, 1)

    def batch(xs, tables, out):
        calls.append(len(xs))
        out.copy_(xs[0] + tables[0])
        if len(calls) == 2:
            raise RuntimeError("Injected second batch failure")

    with pytest.raises(RuntimeError, match="second batch"):
        c.raw_attention_view(0, k, v, coeff, None, None, rotate_batch=batch)
    assert calls == [1, 1]
    assert 0 not in c._rotated_store_signatures
    assert c.last_rope_reuse["rebuilt_rows"] == 2
    batches = []
    assert sum(view(c, 40, batches=batches)) == 6
    assert batches == [[2, 2, 2]]
