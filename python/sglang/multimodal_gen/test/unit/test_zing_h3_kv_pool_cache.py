# SPDX-License-Identifier: Apache-2.0
"""Pooled ZingH3KVCache storage: parity, reuse and no growth (#171, CPU only)."""

import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
from sglang.multimodal_gen.runtime.layers import usp
from sglang.multimodal_gen.runtime.models.dits import zing_h3
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _apply_rope
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    ZingH3Packer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
    ZingKVCapacityError,
    allocate_layer_stores,
)
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
    model,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_sp_cpu import _CPUCollectives

WINDOW = dict(window_chunks=3, sink_chunks=1)


def _inputs(frames, size=2, context=3):
    return SimpleNamespace(
        positive={
            "context": torch.randn(context, 12).bfloat16(),
            "row_role": torch.tensor([0, 1, 0] * (context // 3) + [0] * (context % 3)),
        },
        visual={},
        audio={},
        source_visual=torch.randn(1, 2, frames, size, size),
        conditions=[],
        media=[],
        visual_shape=(frames, size, size),
    )


def _sampler(model):
    packer = ZingH3Packer(model.arch, num_frame_per_block=2)
    sampler = ZingH3Sampler(model, packer, sigma_points=2, **WINDOW)
    sampler.device = torch.device("cpu")
    return sampler


def _storage(model, rows, heads=None):
    arch = model.arch
    return allocate_layer_stores(
        rows,
        num_layers=len(model.blocks),
        local_heads=heads or arch.num_attention_heads,
        head_dim=arch.attention_head_dim,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )


def _pointers(storage):
    return [(s.k.data_ptr(), s.v.data_ptr(), s.rotated_k.data_ptr()) for s in storage]


@torch.no_grad()
def _generate(sampler, inputs, visual, storage=None, prompts=()):
    state = sampler.initialize(inputs, visual.clone(), storage=storage)
    prompts = list(prompts)
    while state.index < len(state.spans):
        sampler.step(state)
        if prompts and state.index == 2:
            inputs.positive = prompts.pop(0)
            sampler.update_condition(state, inputs)
    stats, cache = dict(state.cache.stats), state.cache
    result = state.visual.clone()
    state.close()
    return result, stats, cache


@torch.no_grad()
def test_pooled_sampling_matches_growth_path_across_evictions(model):
    sampler = _sampler(model)
    torch.manual_seed(1)
    inputs, visual = _inputs(9), torch.randn(1, 2, 9, 2, 2)
    expected, grown, _ = _generate(sampler, inputs, visual)
    # 9 latent frames in 2-frame chunks is five chunks: two evictions at 3/1.
    assert grown["compactions"] >= 2 and grown["store_allocations"] > 0
    storage = _storage(model, 64)
    pointers = _pointers(storage)
    actual, pooled, cache = _generate(sampler, inputs, visual, storage)
    assert torch.equal(actual, expected)
    assert pooled["store_allocations"] == pooled["rotated_allocations"] == 0
    assert pooled["compactions"] == grown["compactions"]
    assert _pointers(storage) == pointers
    assert cache.prefix is None and not cache.committed and cache._length == 0


@torch.no_grad()
def test_slot_is_reused_by_larger_then_smaller_then_larger_sessions(model):
    sampler = _sampler(model)
    storage = _storage(model, 64)
    pointers = _pointers(storage)
    for size in (4, 2, 4):
        torch.manual_seed(size)
        inputs, visual = _inputs(5, size), torch.randn(1, 2, 5, size, size)
        expected, _, _ = _generate(sampler, inputs, visual)
        actual, stats, _ = _generate(sampler, inputs, visual, storage)
        # A previous session's rows must not leak into this one.
        assert torch.equal(actual, expected)
        assert stats["store_allocations"] == stats["rotated_allocations"] == 0
    assert _pointers(storage) == pointers


@torch.no_grad()
@pytest.mark.parametrize("lengths", [(7, 2), (4, 2), (2, 9)])
def test_prompt_update_shifts_history_inside_the_slot(model, lengths, monkeypatch):
    # A prefix row-count change smaller than a chunk overlaps the move; it must
    # be staged inside the slot, never through a request-time allocation.
    replace_in_place = ZingH3KVCache._replace_prefix_in_place
    shifts = []

    def no_allocation(*args, **kwargs):
        raise AssertionError("pooled prefix replacement allocated a tensor")

    def guarded(self, prefix, old_rows, new_rows):
        shifts.append(new_rows - old_rows)
        with monkeypatch.context() as scope:
            for name in ("clone", "new_empty", "new_zeros", "contiguous"):
                scope.setattr(torch.Tensor, name, no_allocation)
            for name in ("empty", "empty_like", "zeros", "zeros_like", "cat"):
                scope.setattr(torch, name, no_allocation)
            return replace_in_place(self, prefix, old_rows, new_rows)

    monkeypatch.setattr(ZingH3KVCache, "_replace_prefix_in_place", guarded)
    sampler = _sampler(model)
    torch.manual_seed(3)
    inputs, visual = _inputs(9), torch.randn(1, 2, 9, 2, 2)
    prompts = [
        {
            "context": torch.randn(n, 12).bfloat16(),
            "row_role": torch.zeros(n, dtype=torch.long),
        }
        for n in lengths
    ]
    original = inputs.positive
    expected, _, _ = _generate(sampler, inputs, visual, prompts=prompts[:1])
    inputs.positive = original
    storage = _storage(model, 64)
    pointers = _pointers(storage)
    sampler.prefix_workspace = _storage(model, 16)
    workspace = _pointers(sampler.prefix_workspace)
    actual, stats, _ = _generate(sampler, inputs, visual, storage, prompts[:1])
    assert torch.equal(actual, expected)
    assert shifts == [lengths[0] - 3]  # +1 and -1 overlap their own chunk
    assert stats["store_allocations"] == stats["rotated_allocations"] == 0
    assert _pointers(storage) == pointers
    assert _pointers(sampler.prefix_workspace) == workspace


@torch.no_grad()
def test_pooled_cache_rejects_growth_beyond_the_slot(model):
    sampler = _sampler(model)
    inputs, visual = _inputs(9), torch.randn(1, 2, 9, 2, 2)
    storage = _storage(model, 8)
    with pytest.raises(ZingKVCapacityError):
        _generate(sampler, inputs, visual, storage)
    # The failed session is closed; the slot serves the next session intact.
    small = _inputs(3)
    _generate(sampler, small, torch.randn(1, 2, 3, 2, 2), _storage(model, 64))


def test_pooled_storage_requires_a_finite_raw_window(model):
    storage = _storage(model, 8)
    with pytest.raises(ValueError, match="finite-window"):
        ZingH3KVCache(len(model.blocks), storage=storage)
    with pytest.raises(ValueError, match="every layer"):
        ZingH3KVCache(
            len(model.blocks) + 1,
            window_policy=ZingWindowPolicy(**WINDOW),
            rope_position_mode="block_relative",
            storage=storage,
        )


@torch.no_grad()
def test_pooled_cache_rejects_a_mismatched_head_shard(model):
    sampler = _sampler(model)
    storage = _storage(model, 64, heads=1)
    with pytest.raises(ValueError, match="does not match the pool slot"):
        _generate(sampler, _inputs(3), torch.randn(1, 2, 3, 2, 2), storage)


def _transactions(sampler, layout, blocks):
    out = []
    for block in range(blocks):
        phase = "prefix" if block == 0 else ("target" if block % 2 == 0 else "source")
        target = phase == "target"
        out.append(
            (
                sampler._select_block(
                    layout,
                    block,
                    phase=phase,
                    video_t=0.25 if target else 1.0,
                    audio_t=None,
                ),
                "active" if target else phase,
            )
        )
        if target:
            out.append(
                (
                    sampler._select_block(
                        layout,
                        block,
                        phase="target",
                        video_t=1,
                        audio_t=None,
                        final=True,
                    ),
                    "final",
                )
            )
    return out


@torch.no_grad()
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("stable_shapes", [False, True])
@pytest.mark.parametrize("timing", ["before_source", "after_step0"])
def test_sp_pooled_raw_store_matches_unpooled(
    model, monkeypatch, world, stable_shapes, timing
):
    """SP raw K and replicated-prefix shards are staged into the slot."""
    if world == 4:
        arch = replace(
            model.arch,
            num_attention_heads=4,
            hidden_size=32,
            adaln_out_features=768,
            final_adaln_out_features=64,
        )
        model = zing_h3.ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
        torch.manual_seed(7)
        for name, param in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                param.fill_(1)
            else:
                param.copy_(torch.randn(param.shape) * 0.08)
        model.rope.inv_freq.fill_(1)
    model.experimental_stable_shapes = stable_shapes
    # CPU ranks share one model, unlike production's process-local SP models.
    # Initialize the lazy timestep buffer before threads can replace it after
    # another rank has bound its cache to the model's buffer identities.
    model.time_embedder(torch.zeros(1))
    sampler = _sampler(model)
    torch.manual_seed(5)
    inputs, visual = _inputs(7), torch.randn(1, 2, 7, 2, 2)
    layout = sampler.processor.pack_inference_layout(
        inputs, inputs.positive, visual, None
    )
    transactions = _transactions(sampler, layout, 9)
    communication = _CPUCollectives(world)
    monkeypatch.setattr(
        zing_h3, "get_ulysses_ctx", lambda: (world, communication.thread.rank)
    )
    monkeypatch.setattr(
        zing_h3, "get_sp_group", lambda: SimpleNamespace(ulysses_group=None)
    )
    monkeypatch.setattr(zing_h3, "gather_sequence_varlen", communication.gather)
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: world)
    monkeypatch.setattr(
        usp, "get_ulysses_parallel_rank", lambda: communication.thread.rank
    )
    monkeypatch.setattr(usp, "_ipc_ready_group", lambda: None)
    monkeypatch.setattr(usp, "_usp_all_to_all_single_varlen", communication.all_to_all)
    heads = model.arch.num_attention_heads // world
    staged = {"raw_k": 0, "prefix": 0}
    for name, key in (("stage_raw_k", "raw_k"), ("stage_prefix_kv", "prefix")):
        original = getattr(ZingH3KVCache, name)

        def counted(self, *args, _original=original, _key=key, **kwargs):
            if self.pooled:
                staged[_key] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(ZingH3KVCache, name, counted)

    def cache(storage):
        # Keep the historical prepare_chunk oracle explicit; delayed mode uses the default.
        policy = {"window_eviction_timing": timing} if timing == "before_source" else {}
        result = ZingH3KVCache(
            len(model.blocks),
            window_policy=ZingWindowPolicy(**WINDOW),
            rope_position_mode="block_relative",
            storage=storage,
            **policy,
        )
        assert result.window_eviction_timing == timing
        return result

    def run(pooled):
        def rank_worker(rank):
            communication.thread.rank = rank
            with torch.no_grad():
                storage = _storage(model, 64, heads) if pooled else None
                target = cache(storage)
                outputs = []
                for packed, phase in transactions:
                    if phase == "source":
                        block = int(packed["block_idx"][0]) + 1
                        if timing == "before_source":
                            target.prepare_chunk(block)
                        else:
                            target.begin_chunk(block)
                    outputs.append(sampler._forward(packed, target, phase))
                    if timing == "after_step0" and phase == "active":
                        target.complete_step0_and_evict(int(packed["block_idx"][0]))
                return outputs, target, storage

        with ThreadPoolExecutor(max_workers=world) as executor:
            futures = [executor.submit(rank_worker, rank) for rank in range(world)]
            results, errors = [], []
            for rank, future in enumerate(futures):
                try:
                    results.append(future.result())
                except Exception as error:
                    errors.append((rank, error))
            if errors:
                details = "\n".join(
                    f"CPU SP rank {rank}, pooled={pooled}, timing={timing}:\n"
                    + "".join(
                        traceback.format_exception(
                            type(error), error, error.__traceback__
                        )
                    )
                    for rank, error in errors
                )
                raise RuntimeError(details) from errors[0][1]
            return results

    unpooled, pooled_runs = run(False), run(True)
    # Every SP commit stages raw K into the slot; a replicated prefix stages
    # its head shard instead of keeping a full-head copy.
    assert staged["raw_k"] > 0
    assert (staged["prefix"] > 0) == stable_shapes
    for (expected, reference, _), (actual, pooled, storage) in zip(
        unpooled, pooled_runs
    ):
        for got, want in zip(actual, expected):
            for value, ref in zip(got, want):
                if ref is not None:
                    assert torch.equal(value, ref)
        assert pooled.stats["store_allocations"] == 0
        assert pooled.stats["rotated_allocations"] == 0
        for layer in range(len(model.blocks)):
            for value, ref in zip(pooled.history(layer), reference.history(layer)):
                assert torch.equal(value, ref)
            # Committed rows are views of this rank's slot, not external copies.
            assert pooled.history(layer)[0].shape[1] == heads
            assert pooled._stores[layer][0].data_ptr() == storage[layer].k.data_ptr()


@torch.no_grad()
def test_compaction_moves_rows_without_cloning(model, monkeypatch):
    clones, compacting = [], []
    original = torch.Tensor.clone
    compact = ZingH3KVCache._compact_store

    def counting_clone(self, *args, **kwargs):
        if compacting:
            clones.append(self.numel())
        return original(self, *args, **kwargs)

    def tracked_compact(self):
        compacting.append(True)
        try:
            return compact(self)
        finally:
            compacting.pop()

    monkeypatch.setattr(ZingH3KVCache, "_compact_store", tracked_compact)
    monkeypatch.setattr(torch.Tensor, "clone", counting_clone)
    sampler = _sampler(model)
    torch.manual_seed(1)
    inputs, visual = _inputs(9), torch.randn(1, 2, 9, 2, 2)
    _, stats, _ = _generate(sampler, inputs, visual, _storage(model, 64))
    assert stats["compactions"] >= 2
    assert clones == []


@pytest.mark.parametrize(
    "destination, source, rows",
    [(0, 5, 5), (0, 7, 5), (0, 2, 7), (3, 4, 9), (1, 3, 2), (0, 1, 12), (2, 3, 16)],
)
def test_move_rows_matches_cloned_copy_for_overlapping_left_shifts(
    destination, source, rows
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        _move_rows as move_rows,
    )

    tensor = torch.arange(20 * 3, dtype=torch.float32).view(20, 3)
    expected = tensor.clone()
    expected[destination : destination + rows] = tensor[source : source + rows].clone()
    move_rows(tensor, destination, source, rows)
    assert torch.equal(tensor, expected)


@pytest.mark.parametrize(
    "spans",
    [
        [],
        [(0, 4), (4, 6)],
        [(0, 3), (7, 5), (14, 2)],
        [(5, 4), (12, 6)],
        [(0, 2), (3, 1), (9, 8)],
    ],
)
def test_move_store_rows_moves_only_the_listed_rows_left(spans):
    cache = ZingH3KVCache.__new__(ZingH3KVCache)
    cache._stores = {
        layer: (
            torch.randn(24, 2, 4, generator=torch.Generator().manual_seed(layer)),
            torch.randn(24, 2, 4, generator=torch.Generator().manual_seed(9 + layer)),
        )
        for layer in range(3)
    }
    before = {
        layer: tuple(tensor.clone() for tensor in pair)
        for layer, pair in cache._stores.items()
    }
    cache.move_store_rows(spans)
    live = sum(rows for _, rows in spans)
    for layer, pair in cache._stores.items():
        for tensor, original in zip(pair, before[layer]):
            pieces = [original[o : o + r] for o, r in spans] or [original[:0]]
            expected = torch.cat(pieces)
            assert torch.equal(tensor[:live], expected)
            # Rows past the live length are never read and stay untouched.
            assert torch.equal(tensor[live:], original[live:])


@pytest.mark.parametrize(
    "spans,runs",
    [
        # Eviction after the prefix: every later chunk shares one shift.
        ([(0, 10), (30, 20), (50, 20)], [(10, 30, 40)]),
        # Unequal shifts cannot share a run.
        ([(0, 5), (10, 5), (20, 5)], [(5, 10, 5), (10, 20, 5)]),
        # A gap between moved chunks with the same shift splits the run.
        ([(0, 4), (8, 4), (20, 4)], [(4, 8, 4), (8, 20, 4)]),
        ([(0, 4), (4, 6)], []),
    ],
)
def test_compaction_runs_merge_adjacent_chunks_with_one_shift(spans, runs):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        _compaction_runs,
    )

    assert _compaction_runs(spans) == runs


@torch.no_grad()
def test_compaction_moves_every_layer_in_one_fused_call_per_run(model, monkeypatch):
    from sglang.multimodal_gen.runtime.models.dits import zing_h3_cache

    sampler = _sampler(model)
    torch.manual_seed(1)
    inputs, visual = _inputs(9), torch.randn(1, 2, 9, 2, 2)
    expected, _, _ = _generate(sampler, inputs, visual, _storage(model, 64))
    calls = []

    def reference(tensors, runs):
        calls.append((len(tensors), list(runs)))
        for tensor in tensors:
            for destination, source, rows in runs:
                tensor[destination : destination + rows] = tensor[
                    source : source + rows
                ].clone()

    monkeypatch.setattr(zing_h3_cache, "_fused_compaction_rejection", lambda *a: None)
    monkeypatch.setattr(zing_h3_cache, "_fused_move_rows", reference)
    actual, stats, _ = _generate(sampler, inputs, visual, _storage(model, 64))
    assert torch.equal(actual, expected)
    assert len(calls) == stats["compactions"] >= 2
    assert all(count == 2 * len(model.blocks) and runs for count, runs in calls)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shift", [16, 17, 390, 1170])
@torch.no_grad()
def test_fused_move_rows_matches_reference_for_overlapping_left_shifts(shift):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        _fused_compaction_rejection,
        _fused_move_rows,
    )

    torch.manual_seed(5)
    tensors = [
        torch.randn(4000, 7, 128, device="cuda", dtype=torch.bfloat16) for _ in range(6)
    ]
    runs = [(100, 100 + shift, 2600)]
    expected = [t.clone() for t in tensors]
    for t in expected:
        t[100:2700] = t[100 + shift : 2700 + shift].clone()
    assert _fused_compaction_rejection(tensors, runs) is None
    _fused_move_rows(tensors, runs)
    for actual, reference in zip(tensors, expected):
        assert torch.equal(actual, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_fused_compaction_rejects_unsafe_or_unsupported_moves():
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        _fused_compaction_rejection,
    )

    t = torch.zeros(64, 2, 128, device="cuda", dtype=torch.bfloat16)
    assert _fused_compaction_rejection([t], [(0, 4, 20)]) == "shift"
    assert _fused_compaction_rejection([t.cpu()], [(0, 32, 20)]) == "device"
    assert _fused_compaction_rejection([t[:, :1]], [(0, 32, 20)]) == "layout"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_fused_compaction_never_synchronizes_the_host():
    """Fresh store addresses (a grown store) must not add a blocking H2D copy."""
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        _fused_move_rows,
    )

    tensors = [
        torch.randn(256, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    ]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _fused_move_rows(tensors, [(0, 64, 128)])
    finally:
        torch.cuda.set_sync_debug_mode("default")


@torch.no_grad()
def test_model_routes_current_q_through_the_dispatch(model, monkeypatch):
    calls = []
    eager = _apply_rope

    def spy(x, rope):
        calls.append(tuple(x.shape))
        return eager(x, rope)

    monkeypatch.setattr(zing_h3, "apply_current_rope", spy, raising=False)
    sampler = _sampler(model)
    torch.manual_seed(3)
    inputs, visual = _inputs(5), torch.randn(1, 2, 5, 2, 2)
    expected, _, _ = _generate(sampler, inputs, visual)
    assert calls, "current Q/K RoPE bypassed zing_h3.apply_current_rope"
    monkeypatch.setattr(zing_h3, "apply_current_rope", eager, raising=False)
    torch.manual_seed(3)
    actual, _, _ = _generate(sampler, inputs, visual)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("offset,rows", [(0, 3), (2, 3), (10, 5), (12, 0)])
def test_unchanged_rebind_allocates_no_probe_views(monkeypatch, strided, offset, rows):
    cache = ZingH3KVCache(1)
    keys, values = (torch.randn(24 if strided else 12, 2, 4) for _ in range(2))
    if strided:
        keys, values = keys[::2], values[::2]
    cache._stores[0] = (keys, values)
    old = ((keys[offset : offset + rows], values[offset : offset + rows]),)
    cache.prefix = SimpleNamespace(
        offset=offset, positions=torch.zeros(rows, 3), layers=old
    )
    sliced = []
    original = torch.Tensor.__getitem__

    def tracked(tensor, index):
        if tensor is keys or tensor is values:
            sliced.append(index)
        return original(tensor, index)

    monkeypatch.setattr(torch.Tensor, "__getitem__", tracked)
    cache._rebind(0)
    assert not sliced
    assert cache.prefix.layers is old

    if rows == 0:
        return

    # A moved chunk must receive fresh views; a planning shadow must leave the
    # original tuple and its Tensor offsets untouched until publication.
    cache.prefix.offset = 1
    cache._rebind(0)
    assert len(sliced) == 2
    assert cache.prefix.layers is not old
    assert torch.equal(cache.prefix.layers[0][0], keys[1 : 1 + rows])
    assert old[0][0].storage_offset() == keys[offset : offset + rows].storage_offset()


@pytest.mark.parametrize("strided", [False, True])
def test_pooled_layer_ranges_materialize_stable_views_on_demand(monkeypatch, strided):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import _StoreLayerViews

    stores = tuple((torch.randn(24, 2, 4), torch.randn(24, 2, 4)) for _ in range(2))
    if strided:
        stores = tuple((k[::2], v[::2]) for k, v in stores)
    expected = tuple((k[2:5], v[2:5]) for k, v in stores)
    identities = {id(t) for pair in stores for t in pair}
    original = torch.Tensor.__getitem__
    slices = []

    def tracked(tensor, index):
        if id(tensor) in identities:
            slices.append(index)
        return original(tensor, index)

    monkeypatch.setattr(torch.Tensor, "__getitem__", tracked)
    layers = _StoreLayerViews(stores, 2, 5)
    assert len(layers) == 2 and not slices
    assert layers[0] is layers[0]
    assert len(slices) == 2
    assert layers[-1] is layers[1]
    assert tuple(layers) == layers[:]
    assert len(slices) == 4
    for actual, reference in zip(layers, expected):
        for a, b in zip(actual, reference):
            assert torch.equal(a, b)
            assert a.data_ptr() == b.data_ptr() and a.stride() == b.stride()
    for invalid in (2, -3):
        with pytest.raises(IndexError):
            layers[invalid]


@torch.no_grad()
def test_pooled_planning_compaction_keeps_layer_views_lazy_and_old_ranges_intact(
    monkeypatch,
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        ZingKVChunk,
        _StoreLayerViews,
    )

    layers = 50
    storage = allocate_layer_stores(
        16,
        num_layers=layers,
        local_heads=1,
        head_dim=2,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    cache = ZingH3KVCache(
        layers,
        storage=storage,
        window_policy=ZingWindowPolicy(window_chunks=3, sink_chunks=1),
    )
    cache._stores = {i: (slot.k, slot.v) for i, slot in enumerate(storage)}
    cache.prefix = ZingKVChunk(
        torch.zeros(3, 3), 0, cache._chunk_layers(0, 3), offset=0
    )
    cache.committed = [
        ZingKVChunk(torch.zeros(3, 3), 2, cache._chunk_layers(5, 3), offset=5)
    ]
    cache._length = 8
    old_layers = cache.committed[0].layers
    old_view = old_layers[0][0]
    old_pointer = old_view.data_ptr()
    identities = {id(t) for pair in cache._stores.values() for t in pair}
    original = torch.Tensor.__getitem__
    slices = []

    def tracked(tensor, index):
        if id(tensor) in identities:
            slices.append(index)
        return original(tensor, index)

    monkeypatch.setattr(torch.Tensor, "__getitem__", tracked)
    shadow = cache.planning_copy()
    shadow._compact_store()
    assert not slices
    assert shadow.committed[0].offset == 3
    assert isinstance(shadow.committed[0].layers, _StoreLayerViews)
    assert all(not c.layers._views for c in shadow._stored_chunks())
    assert cache.committed[0].offset == 5
    assert cache.committed[0].layers is old_layers
    assert old_layers[0][0] is old_view and old_view.data_ptr() == old_pointer
    assert not slices
    cache.publish_plan(shadow)
    current = cache.committed[0].layers[0][0]
    assert len(slices) == 2
    assert (
        current.data_ptr() == storage[0].k.data_ptr() + 3 * storage[0].k.stride(0) * 2
    )
    assert old_view.data_ptr() == old_pointer
