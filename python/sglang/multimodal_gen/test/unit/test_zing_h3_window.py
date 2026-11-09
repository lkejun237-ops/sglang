# SPDX-License-Identifier: Apache-2.0
"""Window ownership and release tests; real-checkpoint acceptance is separate."""

import weakref

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy


@pytest.mark.parametrize(
    "window,sink",
    [(0, 0), (-1, 0), (True, 0), (21, True), (21, -1), (21, 21), (None, 3), (21.0, 3)],
)
def test_invalid_window(window, sink):
    with pytest.raises(ValueError):
        ZingWindowPolicy(window, sink)


def test_window_policy_active_sinks_validation_and_retains():
    default_policy = ZingWindowPolicy(21, 3)
    assert default_policy.effective_sinks() == (0, 1, 2)
    for u in (0, 1, 2):
        assert default_policy.is_sink(u)
        assert default_policy.retains(u, current_id=50)
    assert not default_policy.is_sink(3)
    assert not default_policy.retains(3, current_id=50)

    refreshed_step1 = ZingWindowPolicy(21, 3, active_sinks=(0, 2, 15))
    assert refreshed_step1.effective_sinks() == (0, 2, 15)
    assert refreshed_step1.is_sink(0)
    assert not refreshed_step1.is_sink(1)
    assert refreshed_step1.is_sink(2)
    assert refreshed_step1.is_sink(15)
    assert not refreshed_step1.retains(1, current_id=33)
    assert refreshed_step1.retains(2, current_id=33)
    assert refreshed_step1.retains(15, current_id=33)
    # Local sliding window for current_id=33 keeps [33 - (21 - 3 - 1) = 16 .. 32].
    assert not refreshed_step1.retains(14, current_id=33)
    assert refreshed_step1.retains(16, current_id=33)

    for bad_sinks in [
        (0, 1),  # wrong length
        (0, 1, 2, 3),  # wrong length
        (1, 2, 3),  # first sink must be 0
        (0, 5, 4),  # not strictly increasing
        (0, 2, 2),  # duplicate
        (-1, 0, 1),  # negative
        (0, True, 2),  # bool not allowed
        [0, 1, 2],  # must be tuple
    ]:
        with pytest.raises(ValueError):
            ZingWindowPolicy(21, 3, active_sinks=bad_sinks)


@torch.no_grad()
def test_sink_refresh_and_off_preserve_real_source_target_rows():
    plan = ChunkPlan([3, 2], False)
    cache = ZingH3KVCache(
        1, window_policy=ZingWindowPolicy(5, 2), window_eviction_timing="after_step0"
    )
    for i in range(10):
        _commit(cache, plan.chunk(i), delayed=True)
    cache.update_sink("refresh", current_id=10)
    assert [c.temporal_chunk_id for c in cache.committed] == [8, 9]
    assert sum(c.frames for c in cache.committed) == 5
    for chunk in cache.committed:
        k, v = chunk.layers[0]
        assert k.shape[0] == 4  # Both source and target survived compaction.
        assert (k == chunk.temporal_chunk_id).all()
        assert (v == 0).all()
    for i in (10, 11, 12):
        _commit(cache, plan.chunk(i), delayed=True)
    assert [c.temporal_chunk_id for c in cache.committed] == [8, 9, 10, 11, 12]
    cache.update_sink("off", current_id=13)
    assert [c.temporal_chunk_id for c in cache.committed] == [10, 11, 12]
    for i in (13, 14):
        _commit(cache, plan.chunk(i), delayed=True)
    assert [c.temporal_chunk_id for c in cache.committed] == [10, 11, 12, 13, 14]
    _commit(cache, plan.chunk(15), delayed=True)
    assert [c.temporal_chunk_id for c in cache.committed] == [11, 12, 13, 14, 15]
    before = cache.committed.copy()
    cache.update_sink("off", current_id=16)
    assert cache.committed == before


@torch.no_grad()
@pytest.mark.parametrize("count", [0, 1, 2])
def test_sink_refresh_before_history_fills(count):
    plan = ChunkPlan([3, 2], False)
    cache = ZingH3KVCache(
        1, window_policy=ZingWindowPolicy(5, 0), window_eviction_timing="after_step0"
    )
    for i in range(count):
        _commit(cache, plan.chunk(i), delayed=True)
    cache.update_sink("refresh", current_id=count)
    assert cache.window_policy == ZingWindowPolicy(5, 2)
    for i in range(count, 9):
        _commit(cache, plan.chunk(i), delayed=True)
    assert [c.temporal_chunk_id for c in cache.committed] == [0, 1, 6, 7, 8]


@torch.no_grad()
@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize(
    "metadata_device",
    [
        "meta",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"),
        ),
    ],
)
def test_graph_source_metadata_merges_with_device_final(pooled, metadata_device):
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        allocate_layer_stores,
    )

    storage = (
        allocate_layer_stores(
            8,
            num_layers=1,
            local_heads=1,
            head_dim=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        if pooled
        else None
    )
    cache = ZingH3KVCache(
        1,
        storage=storage,
        window_policy=ZingWindowPolicy(5, 2),
        window_eviction_timing="after_step0",
    )
    positions = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.float64)
    source, target = torch.full((2,), 1), torch.full((2,), 2)

    def layers(k, v):
        return (
            (
                torch.full((2, 1, 2), k, dtype=torch.float32),
                torch.full((2, 1, 2), v, dtype=torch.float32),
            ),
        )

    cache.begin_chunk(2)
    cache.update(
        positions, source, "source", 2, layers(1.0, 2.0), validation=(positions, source)
    )
    # Meta makes an accidental device read fail on CPU CI. Production supplies
    # CUDA positions here after the preceding source was published by a graph.
    device_positions = positions.to(metadata_device)
    cache.update(device_positions, target, "active", 2, validation=(positions, target))
    cache.complete_step0_and_evict(2)
    cache.update(
        device_positions,
        target,
        "final",
        2,
        layers(3.0, 4.0),
        validation=(positions, target),
    )
    chunk = cache.committed[0]
    expected_positions = torch.cat((positions, positions))
    assert torch.equal(chunk.positions, expected_positions)
    assert torch.equal(chunk.host_positions, expected_positions)
    k, v = chunk.layers[0]
    assert torch.equal(k[:, 0, 0], torch.tensor([1.0, 1.0, 3.0, 3.0]))
    assert torch.equal(v[:, 0, 0], torch.tensor([2.0, 2.0, 4.0, 4.0]))
    cache.update_sink("refresh", current_id=1)
    assert cache.committed == [chunk]
    cache.update_sink("off", current_id=1)
    assert not cache.committed and cache.pending_source is None and cache.active is None


@pytest.mark.parametrize("field", ["sink_start", "min_chunk_id"])
@pytest.mark.parametrize("value", [-1, True, 0.5])
def test_sink_anchor_metadata_requires_nonnegative_integers(field, value):
    with pytest.raises(ValueError):
        ZingWindowPolicy(5, 2, **{field: value})


def _commit(cache, chunk, *, delayed=False):
    n = chunk.end - chunk.start
    # Fractional visual positions and interleaved audio rows remain absolute.
    pos = torch.tensor(
        [[chunk.start * 5 / 3, 0, 0], [chunk.start * 5 / 3 + 0.5, 0, 1]],
        dtype=torch.float64,
    )
    layers = ((torch.ones(2, 1, 2) * chunk.chunk_id, torch.zeros(2, 1, 2)),)
    if delayed:
        cache.begin_chunk(chunk.target_block)
    else:
        cache.prepare_chunk(chunk.target_block)
    cache.update(pos, torch.full((2,), chunk.source_block), "source", n, layers)
    cache.update(pos, torch.full((2,), chunk.target_block), "active", n)
    if delayed:
        cache.complete_step0_and_evict(chunk.target_block)
    cache.update(pos, torch.full((2,), chunk.target_block), "final", n, layers)


@torch.no_grad()
@pytest.mark.parametrize("lengths", [5, [1, 2, 2]])
@pytest.mark.parametrize("window,sink", [(21, 3), (1, 0), (4, 3)])
def test_temporal_selection_and_storage_release(lengths, window, sink):
    cache = ZingH3KVCache(
        1,
        window_policy=ZingWindowPolicy(window, sink),
        window_eviction_timing="before_source",
    )
    plan = ChunkPlan(lengths)
    for i in range(65):
        old = {c.temporal_chunk_id: weakref.ref(c) for c in cache.committed}
        _commit(cache, plan.chunk(i))
        expected = list(range(min(i + 1, sink))) + list(
            range(max(sink, i - (window - sink) + 1), i + 1)
        )
        assert [c.temporal_chunk_id for c in cache.committed] == expected
        assert cache.last_block == 2 * (i + 1)
        for chunk_id, ref in old.items():
            assert (ref() is not None) == (chunk_id in expected)
        for c in cache.committed:
            assert c.positions[0, 0] == plan.chunk(c.temporal_chunk_id).start * 5 / 3
            assert c.positions.shape[0] == 4  # Source and target/audio are one entry.
        del c
    cache.clear()
    assert cache.positions() is None and cache.history(0) is None


@torch.no_grad()
def test_full_history_and_session_isolation():
    full = ZingH3KVCache(1, window_eviction_timing="before_source")
    bounded = ZingH3KVCache(
        1, window_policy=ZingWindowPolicy(21, 3), window_eviction_timing="before_source"
    )
    for i in range(35):
        for cache in (full, bounded):
            _commit(cache, ChunkPlan(5).chunk(i))
    assert len(full.committed) == 35 and len(bounded.committed) == 21
    bounded.clear()
    assert len(full.committed) == 35


@pytest.mark.parametrize(
    "mode,gap",
    [
        ("bad", 12),
        ([], 12),
        ("block_relative", 0),
        ("absolute", True),
        ("absolute", 1.5),
    ],
)
def test_rope_policy_rejects_invalid_parameters(mode, gap):
    with pytest.raises(ValueError):
        ZingH3KVCache(1, rope_position_mode=mode, rope_max_frame_gap=gap)


@torch.no_grad()
def test_relative_rope_shares_modality_time_and_preserves_absolute_storage():
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingKVChunk

    cache = ZingH3KVCache(1, rope_position_mode="block_relative", rope_max_frame_gap=12)
    prefix = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)
    cache.prefix = ZingKVChunk(prefix, 0, ())
    # Source/target and stereo audio revisit the same (fractional) timestamps.
    old = torch.tensor([[2.0, 0, 0], [3.5, 1, 1], [2.0, 2, 2]], dtype=torch.float64)
    recent = torch.tensor([[102.0, 0, 0], [103.5, 1, 1]], dtype=torch.float64)
    cache.committed = [ZingKVChunk(old, 2, ()), ZingKVChunk(recent, 44, ())]
    current = torch.tensor(
        [[105.0, 0, 0], [105.5, 1, 1], [105.0, 2, 2]], dtype=torch.float64
    )
    absolute = cache.positions().clone()
    q, k = cache.attention_positions(current)
    torch.testing.assert_close(
        k[:, 0],
        torch.tensor([0.0, 1.0, 2.0, 3.5, 2.0, 15.5, 17.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        q[:, 0], torch.tensor([18.5, 19.0, 18.5], dtype=torch.float64)
    )
    assert torch.equal(cache.positions(), absolute)
    assert torch.equal(q[:, 1:], current[:, 1:])
    assert torch.equal(k[:, 1:], absolute[:, 1:])
    # A rank-local slice must use the global mapping, not a different origin.
    assert torch.equal(q[1:], cache.attention_positions(current)[0][1:])
    cache.rope_position_mode = "absolute"
    q, k = cache.attention_positions(current)
    assert q is current and torch.equal(k, absolute)


@torch.no_grad()
def test_relative_rope_no_sink_long_history_has_bounded_positions():
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingKVChunk

    cache = ZingH3KVCache(1, rope_position_mode="block_relative")
    for offset in (0, 100, 1000000):
        history = torch.tensor([[offset + 2.0, 0, 0], [offset + 3.0, 0, 0]])
        cache.committed = [ZingKVChunk(history, 2, ())]
        current = torch.tensor([[offset + 4.0, 0, 0]])
        q, k = cache.attention_positions(current)
        assert q[0, 0] == 2
        assert k[:, 0].tolist() == [0, 1]


@torch.no_grad()
def test_relative_sink_to_recent_gap_stays_bounded_after_long_eviction():
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingKVChunk

    cache = ZingH3KVCache(1, rope_position_mode="block_relative", rope_max_frame_gap=12)
    sink = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]])
    for offset in (100, 1000000):
        recent = torch.tensor([[float(offset), 0, 0], [float(offset + 1), 0, 0]])
        key = torch.ones(2, 1, 2)
        cache.committed = [
            ZingKVChunk(sink, 2, ((key, key),)),
            ZingKVChunk(recent, 200, ((key, key),)),
        ]
        q, k = cache.attention_positions(torch.tensor([[float(offset + 2), 0, 0]]))
        assert k[:, 0].tolist() == [0, 1, 13, 14]
        assert q[0, 0] == 15
        assert cache.committed[1].layers[0][0] is key
        assert cache.committed[1].positions is recent


@torch.no_grad()
def test_default_delayed_cache_completes_manual_step0_before_final():
    cache = ZingH3KVCache(1, window_policy=ZingWindowPolicy(2, 0))
    assert cache.window_eviction_timing == "after_step0"
    plan = ChunkPlan(1)
    for index in range(5):
        previous = [chunk.temporal_chunk_id for chunk in cache.committed]
        chunk = plan.chunk(index)
        cache.begin_chunk(chunk.target_block)
        assert [entry.temporal_chunk_id for entry in cache.committed] == previous
        pos = torch.tensor([[index, 0, 0]], dtype=torch.float64)
        layers = ((torch.full((1, 1, 2), float(index)), torch.zeros(1, 1, 2)),)
        cache.update(pos, torch.tensor([chunk.source_block]), "source", 1, layers)
        cache.update(pos, torch.tensor([chunk.target_block]), "active", 1)
        assert [entry.temporal_chunk_id for entry in cache.committed] == previous
        cache.complete_step0_and_evict(chunk.target_block)
        assert cache.eviction_target_block is None
        assert [entry.temporal_chunk_id for entry in cache.committed] == list(
            range(max(0, index - 1), index)
        )
        cache.update(pos, torch.tensor([chunk.target_block]), "final", 1, layers)
        assert [entry.temporal_chunk_id for entry in cache.committed] == list(
            range(max(0, index - 1), index + 1)
        )
        assert cache.pending_source is cache.active is None
    cache.clear()
    assert cache.history(0) is None


@torch.no_grad()
def test_rolling_scene_sink_refresh_two_step_parity_and_stability_guard(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        allocate_layer_stores,
    )

    monkeypatch.setenv("ZING_SPARSE_SINK_REFRESH", "1")
    monkeypatch.setenv("ZING_SPARSE_SINK_REFRESH_COS_THR", "-0.35")
    monkeypatch.setenv("ZING_SPARSE_SINK_REFRESH_CAND_MIN_Z", "0.0")
    monkeypatch.setenv("ZING_SPARSE_SINK_REFRESH_STREAK", "16")
    monkeypatch.setenv("ZING_SPARSE_SINK_REFRESH_COOLDOWN", "24")
    monkeypatch.setenv("ZING_SPARSE_SINK_REFRESH_MAX_AGE", "60")
    monkeypatch.setenv("ZING_SPARSE_SINK_REFRESH_MAX_ARMED_WAIT", "16")

    def make_cache():
        stores = allocate_layer_stores(
            256,
            num_layers=1,
            local_heads=1,
            head_dim=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        return ZingH3KVCache(
            1,
            window_policy=ZingWindowPolicy(21, 3),
            window_eviction_timing="after_step0",
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
            storage=stores,
        )

    base = make_cache()
    refreshed = make_cache()
    # Keep baseline disabled so we can compare exact row counts at every chunk.
    base._sink_refresh_enabled = False
    plan = ChunkPlan([3, 2])

    def commit_step(cache, chunk, *, z_scene=None, z_cand=0.0):
        frames = chunk.end - chunk.start
        pos = torch.tensor(
            [
                [float(chunk.start + f), 0.0, float(col)]
                for f in range(frames)
                for col in (0, 1)
            ],
            dtype=torch.float64,
        )
        rows = pos.shape[0]
        src_kv = (
            torch.full((rows, 1, 2), -(chunk.chunk_id + 1.0)),
            torch.full((rows, 1, 2), -(chunk.chunk_id + 101.0)),
        )
        tgt_kv = (
            torch.full((rows, 1, 2), float(chunk.chunk_id + 1.0)),
            torch.full((rows, 1, 2), float(chunk.chunk_id + 101.0)),
        )
        cache.begin_chunk(chunk.target_block)
        cache.update(
            pos, torch.full((rows,), chunk.source_block), "source", frames, (src_kv,)
        )
        cache.update(pos, torch.full((rows,), chunk.target_block), "active", frames)
        if z_scene is not None:
            cache.record_sink_refresh_signal(z_scene, z_cand)
        cache.complete_step0_and_evict(chunk.target_block)
        cache.update(
            pos, torch.full((rows,), chunk.target_block), "final", frames, (tgt_kv,)
        )
        return tgt_kv

    expected_targets = {}
    for u in range(95):
        chunk = plan.chunk(u)
        # Chunks 0..2: initial sinks (0=target-only 3f, 1=target-only 2f, 2=target-only 3f)
        if u < 3:
            frames = chunk.end - chunk.start
            pos = torch.tensor(
                [
                    [float(chunk.start + f), 0.0, float(col)]
                    for f in range(frames)
                    for col in (0, 1)
                ],
                dtype=torch.float64,
            )
            rows = pos.shape[0]
            tgt_kv = (
                torch.full((rows, 1, 2), float(u + 1.0)),
                torch.full((rows, 1, 2), float(u + 101.0)),
            )
            for c_obj in (base, refreshed):
                c_obj.begin_chunk(chunk.target_block)
                c_obj.update(
                    pos, torch.full((rows,), chunk.target_block), "active", frames
                )
                c_obj.complete_step0_and_evict(chunk.target_block)
                c_obj.update(
                    pos,
                    torch.full((rows,), chunk.target_block),
                    "final",
                    frames,
                    (tgt_kv,),
                )
            expected_targets[u] = tgt_kv
            continue

        commit_step(base, chunk)

        # Inject scene change signals starting at u=24:
        # At u=24..39 (16 chunks), z_scene=-1.0 triggers ARMED at u=39.
        # At u=39 (odd, cand=21) and u=40 (even, cand=22), keep z_cand=-0.5 (< 0.0, panning unstable) -> stays ARMED!
        # At u=41 (odd, cand=23), set z_cand=0.5 (>= 0.0, stable!) -> triggers STEP1_ODD -> active_sinks=(0, 2, 23)
        # At u=42 (even, cand=24) -> completes STEP2_EVEN -> active_sinks=(0, 23, 24)
        z_s, z_c = 0.5, 0.5
        if 24 <= u <= 39:
            z_s = -1.0
            z_c = -0.5
        elif u == 40:
            z_s = -1.0
            z_c = -0.5
        elif u == 41:
            z_s = -1.0
            z_c = 0.5
        expected_targets[u] = commit_step(refreshed, chunk, z_scene=z_s, z_cand=z_c)

        # Invariant 1: Total live row count _length must be 100% identical to no-refresh baseline at every step!
        assert refreshed._length == base._length, (
            f"Row count mismatch at u={u}: {refreshed._length} != {base._length}"
        )
        # Invariant 2: All compaction spans are strictly monotonic and move left only (source >= dest)
        dest = 0
        for src_off, span_rows in refreshed._last_compaction_spans:
            assert src_off >= dest, (
                f"Compaction moved right at u={u}: {src_off} < {dest}"
            )
            dest += span_rows

        if u in (39, 40):
            # Panning stability guard holds promotion in ARMED state
            assert refreshed._sink_refresh_state == "ARMED"
            assert refreshed.window_policy.effective_sinks() == (0, 1, 2)
        elif u == 41:
            # Odd candidate u_cand = 41 - 18 = 23 promoted to slot 1, sorted active_sinks = (0, 2, 23)
            assert refreshed._sink_refresh_state == "STEP1_ODD"
            assert refreshed.window_policy.effective_sinks() == (0, 2, 23)
            s_chunks = refreshed.committed[:3]
            assert [c.temporal_chunk_id for c in s_chunks] == [0, 2, 23]
            # Trimmed chunk 23 must have only its clean target rows (2 frames * 2 cols = 4 rows)
            assert s_chunks[2].positions.shape[0] == 4
            torch.testing.assert_close(
                s_chunks[2].layers[0][0], expected_targets[23][0]
            )
            torch.testing.assert_close(
                s_chunks[2].layers[0][1], expected_targets[23][1]
            )
        elif u == 42:
            # Even candidate u_cand = 42 - 18 = 24 promoted to slot 2, sorted active_sinks = (0, 23, 24)
            assert refreshed._sink_refresh_state == "COOLDOWN"
            assert refreshed.window_policy.effective_sinks() == (0, 23, 24)
            assert refreshed.stats["sink_refreshes"] == 1
            s_chunks = refreshed.committed[:3]
            assert [c.temporal_chunk_id for c in s_chunks] == [0, 23, 24]
            assert [c.positions.shape[0] for c in s_chunks] == [6, 4, 6]
            for s_u, s_c in zip((0, 23, 24), s_chunks):
                torch.testing.assert_close(s_c.layers[0][0], expected_targets[s_u][0])
                torch.testing.assert_close(s_c.layers[0][1], expected_targets[s_u][1])

    # By u=94, max_age=60 from sink_u=23 (age >= 60 at u=83) must have triggered a second full cycle refresh!
    assert refreshed.stats["sink_refreshes"] >= 2
    assert refreshed.window_policy.effective_sinks()[0] == 0
    assert refreshed.window_policy.effective_sinks()[1] > 23
    refreshed.clear()
    assert refreshed.window_policy.effective_sinks() == (0, 1, 2)
    assert refreshed._sink_refresh_state == "COOLDOWN"


@torch.no_grad()
def test_rolling_scene_sink_refresh_auto_descriptor_on_6_chunk_window(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        allocate_layer_stores,
    )

    monkeypatch.setenv("ZING_SCENE_SINK_REFRESH", "1")
    monkeypatch.setenv("ZING_SPARSE_SINK_REFRESH_COS_THR", "-0.35")
    monkeypatch.setenv("ZING_SPARSE_SINK_REFRESH_CAND_MIN_Z", "0.0")
    monkeypatch.setenv("ZING_SPARSE_SINK_REFRESH_STREAK", "16")
    monkeypatch.setenv("ZING_SPARSE_SINK_REFRESH_COOLDOWN", "24")
    monkeypatch.setenv("ZING_SPARSE_SINK_REFRESH_MAX_AGE", "60")

    stores = allocate_layer_stores(
        128,
        num_layers=1,
        local_heads=1,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    cache = ZingH3KVCache(
        1,
        window_policy=ZingWindowPolicy(6, 3),
        window_eviction_timing="after_step0",
        rope_position_mode="block_relative",
        rope_max_frame_gap=12,
        storage=stores,
    )
    for split_first in (False, True):
        plan = ChunkPlan([3, 2], split_first_latent=split_first)
        for initial_blocks in (3, 0):
            cache.clear()
            base_stores = allocate_layer_stores(
                128,
                num_layers=1,
                local_heads=1,
                head_dim=4,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            base_cache = ZingH3KVCache(
                1,
                window_policy=ZingWindowPolicy(6, 3),
                window_eviction_timing="after_step0",
                rope_position_mode="block_relative",
                rope_max_frame_gap=12,
                storage=base_stores,
            )
            base_cache._sink_refresh_enabled = False
            for u in range(50):
                chunk = plan.chunk(u)
                frames = chunk.end - chunk.start
                pos = torch.tensor(
                    [
                        [float(chunk.start + f), 0.0, float(col)]
                        for f in range(frames)
                        for col in (0, 1)
                    ],
                    dtype=torch.float64,
                )
                rows = pos.shape[0]
                if u < 20:
                    vec = torch.tensor([1.0, 0.02 * (u % 3), 0.0, 0.0])
                else:
                    vec = torch.tensor([0.0, 1.0, 0.02 * (u % 3), 0.0])
                tgt_k = vec.view(1, 1, 4).expand(rows, 1, 4).clone()
                tgt_v = torch.full((rows, 1, 4), float(u))
                for c_obj in (base_cache, cache):
                    c_obj.begin_chunk(chunk.target_block)
                    if u >= initial_blocks:
                        src_k = (-vec).view(1, 1, 4).expand(rows, 1, 4).clone()
                        src_v = torch.full((rows, 1, 4), -float(u))
                        c_obj.update(
                            pos,
                            torch.full((rows,), chunk.source_block),
                            "source",
                            frames,
                            ((src_k, src_v),),
                        )
                    c_obj.update(
                        pos, torch.full((rows,), chunk.target_block), "active", frames
                    )
                    c_obj.complete_step0_and_evict(chunk.target_block)
                    c_obj.update(
                        pos,
                        torch.full((rows,), chunk.target_block),
                        "final",
                        frames,
                        ((tgt_k, tgt_v),),
                    )
                assert cache._length == base_cache._length, (
                    f"Length mismatch at split={split_first}, init={initial_blocks}, u={u}: "
                    f"{cache._length} != {base_cache._length}"
                )

            sinks = cache.window_policy.effective_sinks()
            assert sinks[0] == 0
            assert sinks[1] >= 20
            if split_first:
                assert sinks[1] % 2 == 0
                assert sinks[2] == sinks[1] + 2
            else:
                assert sinks[1] % 2 == 1
                assert sinks[2] == sinks[1] + 1

    assert cache.stats["sink_refreshes"] == 4
    assert len(cache._sink_refresh_all_history) == 4


@torch.no_grad()
def test_stationary_scene_never_triggers_false_sink_refresh(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        allocate_layer_stores,
    )

    monkeypatch.setenv("ZING_SCENE_SINK_REFRESH", "1")
    monkeypatch.delenv("ZING_SPARSE_SINK_REFRESH_MAX_AGE", raising=False)

    stores = allocate_layer_stores(
        128,
        num_layers=1,
        local_heads=1,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    cache = ZingH3KVCache(
        1,
        window_policy=ZingWindowPolicy(6, 3),
        window_eviction_timing="after_step0",
        rope_position_mode="block_relative",
        rope_max_frame_gap=12,
        storage=stores,
    )
    plan = ChunkPlan([3, 2])
    for u in range(110):
        chunk = plan.chunk(u)
        frames = chunk.end - chunk.start
        pos = torch.tensor(
            [
                [float(chunk.start + f), 0.0, float(col)]
                for f in range(frames)
                for col in (0, 1)
            ],
            dtype=torch.float64,
        )
        rows = pos.shape[0]
        if u < 80:
            # Stationary room shot: tiny window jitter (std ~ 0.005) and positive sink similarity (~0.97)
            vec = (
                torch.tensor([1.0, 0.25, 0.0, 0.0])
                if u < 3
                else torch.tensor([1.0, 0.01 * (u % 3), 0.0, 0.0])
            )
        else:
            # Genuine 180-degree camera turn at u >= 80
            vec = torch.tensor([0.0, 1.0, 0.01 * (u % 3), 0.0])
        tgt_k = vec.view(1, 1, 4).expand(rows, 1, 4).clone()
        tgt_v = torch.full((rows, 1, 4), float(u))
        cache.begin_chunk(chunk.target_block)
        if u >= 3:
            src_k = (-vec).view(1, 1, 4).expand(rows, 1, 4).clone()
            src_v = torch.full((rows, 1, 4), -float(u))
            cache.update(
                pos,
                torch.full((rows,), chunk.source_block),
                "source",
                frames,
                ((src_k, src_v),),
            )
        cache.update(pos, torch.full((rows,), chunk.target_block), "active", frames)
        cache.complete_step0_and_evict(chunk.target_block)
        cache.update(
            pos,
            torch.full((rows,), chunk.target_block),
            "final",
            frames,
            ((tgt_k, tgt_v),),
        )
        if u < 80:
            assert cache.window_policy.effective_sinks() == (0, 1, 2), (
                f"Stationary camera at u={u} falsely evicted opening sinks: "
                f"{cache.window_policy.effective_sinks()}"
            )
            assert cache.stats.get("sink_refreshes", 0) == 0

    assert cache.stats.get("sink_refreshes", 0) == 1
    assert cache.window_policy.effective_sinks()[0] == 0
    assert cache.window_policy.effective_sinks()[1] >= 80


@torch.no_grad()
@pytest.mark.parametrize("pooled", [False, True])
def test_history_reset_retains_prefix_storage_bytes_and_signature(pooled):
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        allocate_layer_stores,
    )

    storage = (
        allocate_layer_stores(
            64,
            num_layers=1,
            local_heads=1,
            head_dim=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        if pooled
        else None
    )
    cache = ZingH3KVCache(1, window_policy=ZingWindowPolicy(6, 0), storage=storage)
    positions = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.float64)
    blocks = torch.zeros(2, dtype=torch.long)
    cache.update(
        positions,
        blocks,
        "prefix",
        0,
        [(torch.full((2, 1, 2), 7.0), torch.full((2, 1, 2), 11.0))],
        validation=(positions, blocks),
    )
    signature = ("model", "precision", "layout")
    cache.bind_model(signature)
    plan = ChunkPlan([3, 2], False)
    for i in range(4):
        _commit(cache, plan.chunk(i), delayed=True)
    prefix = cache.prefix
    tensors = [
        (k.data_ptr(), v.data_ptr(), k.clone(), v.clone()) for k, v in prefix.layers
    ]
    stores = cache._stores
    cache.reset_history_keep_prefix()
    assert (
        cache.prefix is prefix
        and cache._stores is stores
        and cache._model_signature == signature
    )
    assert cache._length == 2 and not cache.committed and cache.last_block == 0
    for pair, (kp, vp, k, v) in zip(prefix.layers, tensors):
        assert pair[0].data_ptr() == kp and pair[1].data_ptr() == vp
        assert torch.equal(pair[0], k) and torch.equal(pair[1], v)
    _commit(cache, plan.chunk(0), delayed=True)
    assert cache.committed and cache.prefix is prefix
    assert torch.equal(prefix.layers[0][0], tensors[0][2])
    assert torch.equal(prefix.layers[0][1], tensors[0][3])
