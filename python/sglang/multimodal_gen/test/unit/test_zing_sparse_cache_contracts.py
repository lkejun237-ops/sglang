# SPDX-License-Identifier: Apache-2.0
"""Representation and ownership contracts, independent of CUDA availability."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import allocate_layer_stores
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy


def make_cache(monkeypatch, *, retain=False):
    monkeypatch.setenv("SGLANG_ZING_SPARSE_THREE_TIER", "1")
    return ZingH3KVCache(
        1,
        window_policy=ZingWindowPolicy(4, 0),
        rope_position_mode="block_relative",
        storage=allocate_layer_stores(
            512,
            num_layers=1,
            local_heads=1,
            head_dim=128,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        ),
        retain_prefix_layers=retain,
    )


def commit(cache, block, phase, value, rows=64):
    pos = torch.zeros(rows, 3)
    pos[:, 0] = block
    pos[:, 2] = torch.arange(rows)
    blocks = torch.full((rows,), block, dtype=torch.long)
    pair = (
        torch.full((rows, 1, 128), value, dtype=torch.bfloat16),
        torch.full((rows, 1, 128), value + 1, dtype=torch.bfloat16),
    )
    if phase == "final":
        cache.update(pos, blocks, "active", 1)
    cache.update(
        pos,
        blocks,
        phase,
        0 if phase == "prefix" else 1,
        layers=[pair],
        validation=(pos, blocks),
    )
    return pair


@torch.no_grad()
def test_compressed_source_and_target_keep_the_same_temporal_owner(monkeypatch):
    cache = make_cache(monkeypatch)
    commit(cache, 0, "prefix", 1)
    for i in range(3):
        commit(cache, 2 * i + 1, "source", 10 + i)
        commit(cache, 2 * i + 2, "final", 20 + i)
    store = cache._compressed_kv_stores[0]
    assert store.block_chunk_id[: store.num_blocks].tolist() == [-1, 0, 0, 1, 1, 2, 2]
    cache._sync_compressed_kv_store(cache.committed[1:])
    assert store.block_chunk_id[: store.num_blocks].tolist() == [-1, 1, 1, 2, 2]
    assert store.k_blk_mean[0, : store.num_blocks, 0].tolist() == [1, 11, 21, 12, 22]


@torch.no_grad()
def test_planning_never_compacts_authoritative_compressed_tensors(monkeypatch):
    cache = make_cache(monkeypatch)
    commit(cache, 0, "prefix", 1)
    for i in range(3):
        commit(cache, 2 * i + 1, "source", 10 + i)
        commit(cache, 2 * i + 2, "final", 20 + i)
    store = cache._compressed_kv_stores[0]
    before = store.block_chunk_id.clone(), store.k_blk_mean.clone(), store.num_blocks
    shadow = cache.planning_copy()
    shadow._sync_compressed_kv_store(shadow.committed[-1:])
    assert store.num_blocks == before[2]
    assert torch.equal(store.block_chunk_id, before[0])
    assert torch.equal(store.k_blk_mean, before[1])


@torch.no_grad()
def test_compressed_commit_failure_is_not_published(monkeypatch):
    cache = make_cache(monkeypatch)
    store = cache._compressed_kv_stores[0]

    def broken(*args, **kwargs):
        raise RuntimeError("injected compressed write failure")

    monkeypatch.setattr(store, "append_chunk", broken)
    with pytest.raises(RuntimeError, match="compressed write failure"):
        commit(cache, 0, "prefix", 1)
    assert cache.prefix is None and cache._length == 0
    assert store.num_blocks == 0


def test_world_key_excludes_s0_pixels_and_native_preview_shape():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        preparation_key,
    )

    codecs = SimpleNamespace(remote_read_fs=None)
    request = {"world_id": "world-a", "prompt": "fixed", "media": []}
    config = {"width": 32, "height": 32}
    identity = {"world_scoped_key": True, "model": "frozen"}
    previews = [
        None,
        {"frames": torch.zeros(3, 1, 32, 32, dtype=torch.uint8)},
        {"frames": torch.ones(3, 1, 48, 24, dtype=torch.uint8)},
    ]
    keys = [preparation_key(codecs, request, p, config, identity) for p in previews]
    assert len(set(keys)) == 1
    assert (
        preparation_key(
            codecs, {**request, "world_id": "world-b"}, None, config, identity
        )
        != keys[0]
    )


@torch.no_grad()
def test_planning_clear_cannot_zero_the_live_pool(monkeypatch):
    cache = make_cache(monkeypatch)
    commit(cache, 0, "prefix", 1)
    store = cache._compressed_kv_stores[0]
    before = store.block_valid_len.clone(), store.k_blk_mean.clone(), store.num_blocks
    shadow = cache.planning_copy()
    shadow.clear()
    assert torch.equal(store.block_valid_len, before[0])
    assert torch.equal(store.k_blk_mean, before[1])
    assert store.num_blocks == before[2]


@torch.no_grad()
def test_target_only_sink_and_prefix_reset_preserve_physical_parts(monkeypatch):
    cache = make_cache(monkeypatch)
    commit(cache, 0, "prefix", 1, rows=65)
    for i in range(2):
        commit(cache, 2 * i + 1, "source", 10 + i, rows=17)
        commit(cache, 2 * i + 2, "final", 20 + i, rows=17)
    cache._sink_refresh_enabled = True
    cache._stage_sink_refresh_signal_async()
    assert cache.committed[0].descriptor is not None
    torch.testing.assert_close(
        cache.committed[0].descriptor, torch.full((128,), 128**-0.5)
    )
    cache._trim_chunk_to_target(cache.committed[0])
    cache._compact_store()
    store = cache._compressed_kv_stores[0]
    assert store.k_blk_mean[0, : store.num_blocks, 0].tolist() == [1, 1, 20, 11, 21]
    cache.reset_history_keep_prefix()
    commit(cache, 1, "source", 15, rows=17)
    assert cache._compressed_part_layout == ((0, 65), (65, 17))
    assert store.k_blk_mean[0, : store.num_blocks, 0].tolist() == [1, 1, 15]


@torch.no_grad()
def test_fixed_clean_writes_replay_device_values_without_host_publication(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        ZingFixedForwardKVPlan,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_compressed_view import (
        commit_layer,
        mutable_store_views,
    )

    cache = make_cache(monkeypatch)
    commit(cache, 0, "prefix", 1, rows=65)
    shadow = cache.planning_copy()
    pos = torch.zeros(17, 3)
    pos[:, 0] = 1
    pos[:, 2] = torch.arange(17)
    plan = ZingFixedForwardKVPlan(
        shadow,
        pos,
        current_blocks=torch.ones(17, dtype=torch.long),
        ends=(17,),
        clean_rows=17,
    )
    store = plan._compressed_kv_stores[0]
    k, v = plan.stores[0]
    k[:17].fill_(3)
    v[:17].fill_(4)
    operations = []

    class Recorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func not in {
                torch.ops.aten._local_scalar_dense.default,
                torch.ops.aten.nonzero.default,
            }, "Device value escaped to host"
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            operations.append((func, args, kwargs, out))
            return out

    views = tuple(mutable_store_views(store, 2, 3))
    originals = [x.clone() for x in views]
    with Recorder():
        commit_layer(plan, 0, plan.tensors)
    assert store.num_blocks == 2
    assert store.k_blk_mean[0, 2, 0] == 3
    for dst, src in zip(views, originals):
        dst.copy_(src)
    assert store.block_valid_len[2] == 0
    k[:17].fill_(9)
    v[:17].fill_(11)
    plan.tensors["compressed_owners"].fill_(7)
    for func, args, kwargs, out in operations:
        if func.is_view or func in {
            torch.ops.aten.empty.memory_format,
            torch.ops.aten.empty_like.default,
            torch.ops.aten.empty_strided.default,
        }:
            continue
        result = func(*args, **kwargs)
        if not func._schema.is_mutable:
            pairs = (
                zip(result, out) if isinstance(out, (tuple, list)) else [(result, out)]
            )
            for fresh, recorded in pairs:
                if isinstance(recorded, torch.Tensor):
                    recorded.copy_(fresh)
    assert store.k_blk_mean[0, 2, 0] == 9 and store.v_blk_mean[0, 2, 0] == 11
    assert store.block_chunk_id[2] == 7 and store.block_valid_len[2] == 17
    assert store.num_blocks == 2  # Exactly one later host publication owns this.


def test_graph_cannot_replay_stale_block_roles_at_the_same_shape(monkeypatch):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    monkeypatch.delenv("ZING_SPARSE_STATE_FILE", raising=False)
    monkeypatch.setenv("ZING_SPARSE_DISABLE_CHUNK_GRAPH", "0")

    class FrozenHostBackend:
        capturing = False

        def capture(self, fn, **kwargs):
            self.capturing = True
            try:
                output = fn()
            finally:
                self.capturing = False
            return object(), output

        def replay(self, graph):
            pass  # Captured host choices do not execute Python again.

    runner = ZingChunkGraphs(backend=FrozenHostBackend(), capture_enabled=True)

    def run(role):
        kv = SimpleNamespace(
            _compressed_branch_specs={
                0: SimpleNamespace(blk_kind_sig=(4, ((role, role + 1),), None, ()))
            },
            compressed_attention_layout=object(),
            compressed_clean_parts=(),
            clean_rows=0,
            rows=0,
            write_start=0,
            reused_history_rows=0,
        )
        plan = SimpleNamespace(signature=("same-shape",), kv_plan=kv)
        return runner._run("steady", [plan], {}, lambda _: (torch.tensor(role),), ())[
            0
        ].item()

    assert run(0) == 0  # warmup
    assert run(0) == 0  # capture
    assert run(1) == 1  # same geometry, different prefix/reference routing
    assert run(1) == 1
    assert run(0) == 0


@pytest.mark.parametrize("index,step_1000", [(0, 1000), (1, 500)])
def test_compressed_eager_uses_same_schedule_and_host_rows_as_graph(
    monkeypatch, index, step_1000
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3 import (
        ZingH3DiTModel,
        ZingHostLayout,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_sparse_attn import (
        attach_sparse_specs_to_plan,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3._sampling_helpers import (
        ZingBlockSamplingHelpers,
        _guard_host_metadata,
        _host_metadata,
    )

    class Generator:
        __call__ = ZingH3DiTModel.forward

        def _validate_packed_inputs(self, *args, **kwargs):
            return 64, 0, 1, None, None, None

    model = Generator()
    sampler = SimpleNamespace(
        generator=model,
        device=torch.device("cpu"),
        video_scheduler=SimpleNamespace(timesteps=torch.tensor([0.25, 0.75])),
        denoising_step_list=[1000, 500],
    )
    host = ZingHostLayout(
        seq_lens=torch.tensor([64]),
        clean_seq_lens=torch.tensor([0]),
        block_idx=torch.full((64,), 14),
        position_ids=torch.zeros(64, 3, dtype=torch.long),
        row_role=torch.ones(64, dtype=torch.long),
    )
    host_t = torch.full((64,), (0.25, 0.75)[index])
    packed = {
        **{
            k: getattr(host, k).clone()
            for k in (
                "seq_lens",
                "clean_seq_lens",
                "block_idx",
                "position_ids",
                "row_role",
            )
        },
        "t": host_t.clone(),
        "_zing_host_layout": host,
        "_zing_host_t": host_t,
        "x": torch.zeros(64, 4),
        "audio_x": torch.empty(0, 4),
        "context": torch.zeros(1, 4),
    }
    _guard_host_metadata(packed)
    kv = SimpleNamespace(write_start=13 * 64, ends=[64])
    history = [("source", u, u * 128, (u + 1) * 128) for u in range(6)]

    def route(context, rows):
        return attach_sparse_specs_to_plan(
            context, history, None, rows, "active", kv, {"mode": "sparse_095"}
        )[0]

    expected = route(sampler, packed)
    assert (expected.step_idx, expected.step_1000) == (index, step_1000)

    def plan(context, shadow, rows, phase, *, generator):
        assert context is sampler and generator is model and phase == "active"
        guarded = _host_metadata(rows)
        assert guarded["host_layout"] is host and guarded["host_t"] is host_t
        got = route(context, rows)
        assert (got.step_idx, got.step_1000) == (expected.step_idx, expected.step_1000)
        rows["t"] = rows["t"].clone()
        assert _host_metadata(rows) is None
        return SimpleNamespace(execute=lambda: ("video", "audio"))

    monkeypatch.setattr(ZingChunkGraphs, "_forward_plan", staticmethod(plan))

    def reject_read(self, *args, **kwargs):
        pytest.fail("unexpected device-to-host metadata read")

    monkeypatch.setattr(torch.Tensor, "cpu", reject_read)
    cache = SimpleNamespace(
        _compressed_kv_enabled=True,
        storage=True,
        planning_copy=lambda: None,
        publish_plan=lambda _: None,
    )
    assert ZingBlockSamplingHelpers._forward(sampler, packed, cache, "active") == (
        "video",
        "audio",
    )
