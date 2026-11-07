# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811
"""Fixed device forward preserves eager arithmetic without cache bookkeeping."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
    ZingFixedForwardKVPlan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model  # noqa: F401
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
    request,
)


def _kv_plan(
    stores,
    rotated,
    *,
    capacity,
    packed_rows,
    clean,
    ends,
    history,
    current,
    history_positions=None,
    rotates=None,
):
    """Fixed-shape plan with the production tensors and device methods.

    ``rotates`` defaults to rotating whenever there is any history, as the
    production plan does for the first forward that sees it.
    """
    rotates = history > 0 if rotates is None else rotates
    tensors = {
        "current_positions": current.clone(),
        "write_rows": torch.arange(history, history + packed_rows),
    }
    if rotates:
        positions = current.new_zeros(capacity, 3)
        if history_positions is not None:
            positions[:history] = history_positions
        tensors["positions"] = positions
    previous = 0
    for branch, stop in enumerate(ends):
        tensors[f"cu_k.{branch}"] = torch.tensor([0, history + stop], dtype=torch.int32)
        tensors[f"cu_q.{branch}"] = torch.tensor(
            [0, stop - previous], dtype=torch.int32
        )
        previous = stop
    plan = SimpleNamespace(
        stores=stores,
        rotated=rotated,
        capacity=capacity,
        rows=packed_rows,
        clean_rows=clean,
        ends=tuple(ends),
        tensors=tensors,
        rotates_history=rotates,
    )
    plan.write_current = MethodType(ZingFixedForwardKVPlan.write_current, plan)
    plan.persist = MethodType(ZingFixedForwardKVPlan.persist, plan)
    return plan


def _stores(model, capacity, heads=None, layers=None):
    heads = model.arch.num_attention_heads if heads is None else heads
    shape = (capacity, heads, model.arch.attention_head_dim)
    count = len(model.blocks) if layers is None else layers
    stores = tuple(
        (
            torch.zeros(shape, dtype=torch.bfloat16),
            torch.zeros(shape, dtype=torch.bfloat16),
        )
        for _ in range(count)
    )
    return stores, tuple(torch.zeros(shape, dtype=torch.bfloat16) for _ in range(count))


def _workspace(model, packed, history=0, clean=False, heads=None):
    rows = packed["position_ids"].shape[0]
    stores, rotated = _stores(model, 128, heads)
    generator = torch.Generator().manual_seed(7)
    for k, v in stores:
        # Raw history is arbitrary but finite; rows past it stay zero.
        k[:history] = (
            torch.randn(k[:history].shape, generator=generator) * 0.1
        ).bfloat16()
        v[:history] = (
            torch.randn(v[:history].shape, generator=generator) * 0.1
        ).bfloat16()
    return _kv_plan(
        stores,
        rotated,
        capacity=128,
        packed_rows=rows,
        clean=rows if clean else 0,
        ends=(rows,),
        history=history,
        current=packed["position_ids"],
        history_positions=torch.arange(history * 3).reshape(history, 3),
    )


@pytest.mark.parametrize("phase", ["active", "source", "prefix"])
@torch.no_grad()
def test_fixed_forward_matches_eager_and_does_not_prepare_commit(model, phase):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, None)
    packed = sampler._select_block(
        layout,
        {"prefix": 0, "source": 1, "active": 2}[phase],
        phase="target" if phase == "active" else phase,
        video_t=0.3 if phase == "active" else 1.0,
        audio_t=1.0,
    )
    kv = _workspace(model, packed, clean=phase != "active")
    assert callable(getattr(model, "prepare_fixed_forward", None)), (
        "fixed device forward preparation is missing"
    )
    before = [(k.clone(), v.clone()) for k, v in kv.stores]
    plan = model.prepare_fixed_forward(packed=packed, kv_plan=kv, phase=phase)
    for pair, old in zip(kv.stores, before):
        for value, expected in zip(pair, old):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    expected = sampler._forward(packed, None, None)
    actual = plan.execute()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0.008, rtol=0.008)
    from sglang.multimodal_gen.test.unit.test_zing_h3_denoise_graph import (
        RecordingGraphBackend,
    )

    backend = RecordingGraphBackend()
    calls = []

    def execute():
        calls.append(True)
        return plan.execute(dict(plan.tensors))

    graph, recorded = backend.capture(execute)
    backend.replay(graph)
    assert len(calls) == 1
    for a, b in zip(actual, recorded):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    if phase == "active":
        # Only raw K is persistent state; the rows are scratch past the history.
        assert all(torch.count_nonzero(k) == 0 for k, _ in kv.stores)
    else:
        assert all(
            torch.count_nonzero(k) > 0 and torch.count_nonzero(v) > 0
            for k, v in kv.stores
        )


@torch.no_grad()
@pytest.mark.parametrize("live_only", [False, True])
def test_fixed_forward_replays_changed_history_with_one_signature(
    model, monkeypatch, live_only
):
    from sglang.multimodal_gen.runtime.models.dits import zing_h3
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import _Entry
    from sglang.multimodal_gen.test.unit.test_zing_h3_denoise_graph import (
        RecordingGraphBackend,
    )

    monkeypatch.setenv("SGLANG_ZING_HISTORY_ROPE_LIVE_ROWS", str(int(live_only)))
    observed = []
    rotate = zing_h3.apply_history_rope_into

    def record_live_rows(*args, live_rows=None, row_range=None):
        observed.append(live_rows)
        return rotate(*args, live_rows=live_rows, row_range=row_range)

    monkeypatch.setattr(zing_h3, "apply_history_rope_into", record_live_rows)
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, None)
    packed = sampler._select_block(layout, 2, phase="target", video_t=0.3, audio_t=1.0)
    assert callable(getattr(model, "prepare_fixed_forward", None)), (
        "fixed device forward preparation is missing"
    )
    kv = _workspace(model, packed, history=3)
    if live_only:
        kv.tensors["history_live_rows"] = torch.tensor([3], dtype=torch.int32)
    plan = model.prepare_fixed_forward(packed=packed, kv_plan=kv, phase="active")
    backend = RecordingGraphBackend()
    static = {name: value.clone() for name, value in plan.tensors.items()}
    graph, output = backend.capture(lambda: plan.execute(static))
    captured_lengths = tuple(observed)
    entry = _Entry(graph, static, output)
    for history in (1, 7, 3):
        changed = _workspace(model, packed, history=history)
        if live_only:
            changed.tensors["history_live_rows"] = torch.tensor(
                [history], dtype=torch.int32
            )
        # The graph owns the stores; only device metadata and history differ.
        for (k, v), (new_k, new_v) in zip(kv.stores, changed.stores):
            k.copy_(new_k)
            v.copy_(new_v)
        other = model.prepare_fixed_forward(
            packed=packed, kv_plan=changed, phase="active"
        )
        entry.refresh(other.tensors)
        reference = tuple(v.clone() for v in plan.execute(other.tensors))
        backend.replay(graph)
        assert captured_lengths
        assert {
            None if value is None else value.item() for value in captured_lengths
        } == {history if live_only else None}
        for actual, expected in zip(output, reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Store addresses are deliberately a separate identity component.
        assert plan.signature[:-1] == other.signature[:-1]
    assert backend.captures == 1


@torch.no_grad()
def test_fixed_joint_preserves_three_branches_and_joint_projections(model, monkeypatch):
    assert callable(getattr(model, "prepare_fixed_forward", None)), (
        "fixed device forward preparation is missing"
    )
    packer, inputs, _, _ = request(model, audio=False)
    inputs.visual_shape = (7, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 7, 2, 2)
    visual = torch.randn(1, 2, 7, 2, 2)
    original = ZingH3Sampler._forward
    seen = []

    def forward(sampler, packed, cache, phase):
        if phase != "joint" or len(packed["_zing_joint_phases"]) != 3:
            return original(sampler, packed, cache, phase)
        rows = packed["position_ids"].shape[0]
        block = packed["block_idx"]
        ends = tuple(
            int((block <= value).sum()) for value in block.unique_consecutive()
        )
        positions, history_positions = cache.attention_positions(
            packed["position_ids"], host_current=packed["position_ids"]
        )
        history = 0 if history_positions is None else history_positions.shape[0]
        stores, rotated = _stores(model, 128)
        if history:
            for layer, (k, v) in enumerate(stores):
                hk, hv = cache.history(layer)
                k[:history].copy_(hk)
                v[:history].copy_(hv)
        kv = _kv_plan(
            stores,
            rotated,
            capacity=128,
            packed_rows=rows,
            clean=int(packed["clean_seq_lens"].sum()),
            ends=ends,
            history=history,
            current=positions,
            history_positions=history_positions,
        )
        plan = model.prepare_fixed_forward(packed=packed, kv_plan=kv, phase="joint")
        counts = {"qkv": 0, "mlp": 0}

        def hook(kind):
            def count(*args):
                counts[kind] += 1

            return count

        handles = [
            module.register_forward_hook(hook(kind))
            for b in model.blocks
            for kind, module in [("qkv", b.attn.qkv_proj), ("mlp", b.mlp)]
        ]
        try:
            actual = plan.execute()
        finally:
            for handle in handles:
                handle.remove()
        assert counts == {"qkv": len(model.blocks), "mlp": len(model.blocks)}
        expected = original(sampler, packed, cache, phase)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0.008, rtol=0.008)
        seen.append(ends)
        return expected

    monkeypatch.setattr(ZingH3Sampler, "_forward", forward)
    sampler = ZingH3Sampler(
        model,  # noqa: F401
        packer,
        sigma_points=3,
        joint_forward="cross_chunk",
        rope_position_mode="block_relative",
    )
    sampler.sample(inputs, visual)
    assert seen and all(len(ends) == 3 for ends in seen)


@torch.no_grad()
def test_fixed_clean_phase_rejects_dirty_timestep_before_persistent_write(model):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, None)
    packed = sampler._select_block(layout, 1, phase="source", video_t=1.0, audio_t=1.0)
    packed["t"].fill_(0.5)
    packed["_zing_host_t"].fill_(0.5)
    kv = _workspace(model, packed, clean=True)
    with pytest.raises(ValueError, match="clean"):
        model.prepare_fixed_forward(packed=packed, kv_plan=kv, phase="source")
    assert all(
        torch.count_nonzero(k) == 0 and torch.count_nonzero(v) == 0
        for k, v in kv.stores
    )


@torch.no_grad()
def test_fixed_prepare_honors_replaced_device_timestep_over_stale_host_copy(model):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, None)
    packed = sampler._select_block(layout, 1, phase="source", video_t=1.0, audio_t=1.0)
    packed["t"] = torch.full_like(packed["t"], 0.5)
    kv = _workspace(model, packed, clean=True)
    with pytest.raises(ValueError, match="clean"):
        model.prepare_fixed_forward(packed=packed, kv_plan=kv, phase="source")
    assert all(
        torch.count_nonzero(k) == 0 and torch.count_nonzero(v) == 0
        for k, v in kv.stores
    )


@pytest.mark.parametrize("world", [2, 4])
@torch.no_grad()
def test_fixed_stable_prefix_preserves_replicated_compute_and_head_shard(
    model, monkeypatch, world
):
    from dataclasses import replace

    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.models.dits import zing_h3
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
    from sglang.multimodal_gen.test.unit.test_zing_h3_denoise_graph import (
        RecordingGraphBackend,
    )

    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", "1")
    arch = replace(
        model.arch,
        num_attention_heads=4,
        hidden_size=32,
        adaln_out_features=6 * model.arch.num_modalities * 32,
        final_adaln_out_features=64,
    )
    model = zing_h3.ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
    torch.manual_seed(17)
    for name, value in model.named_parameters():
        value.copy_(
            torch.ones_like(value)
            if "norm" in name and name.endswith("weight")
            else torch.randn_like(value) * 0.08
        )
    model.rope.inv_freq.fill_(1)
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, None)
    packed = sampler._select_block(layout, 0, phase="prefix", video_t=1.0, audio_t=1.0)
    rows = packed["position_ids"].shape[0]
    original = model._block_forward
    seen = []

    def observe(current, value, **kwargs):
        body = kwargs["inputs"]
        assert body.layout.sequence_splits is None
        assert value.shape[0] == rows
        seen.append(kwargs["index"])
        return original(current, value, **kwargs)

    monkeypatch.setattr(model, "_block_forward", observe)
    heads = arch.num_attention_heads // world
    for rank in range(world):
        monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (world, rank))
        kv = _workspace(model, packed, clean=True, heads=heads)
        plan = model.prepare_fixed_forward(packed=packed, kv_plan=kv, phase="prefix")
        cache = ZingH3KVCache(len(model.blocks), rope_position_mode="block_relative")
        expected = sampler._forward(packed, cache, "prefix")
        seen.clear()
        actual = plan.execute()
        assert seen == list(range(len(model.blocks)))
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        for layer, (k, v) in enumerate(kv.stores):
            expected_k, expected_v = cache.history(layer)
            assert expected_k.shape == (rows, heads, arch.attention_head_dim)
            torch.testing.assert_close(k[:rows], expected_k, rtol=0, atol=0)
            torch.testing.assert_close(v[:rows], expected_v, rtol=0, atol=0)
        backend = RecordingGraphBackend()
        graph, _ = backend.capture(lambda: plan.execute(dict(plan.tensors)))
        before = [(k[:rows].clone(), v[:rows].clone()) for k, v in kv.stores]
        for k, v in kv.stores:
            k.zero_()
            v.zero_()
        backend.replay(graph)
        for (k, v), (expected_k, expected_v) in zip(kv.stores, before):
            torch.testing.assert_close(k[:rows], expected_k, rtol=0, atol=0)
            torch.testing.assert_close(v[:rows], expected_v, rtol=0, atol=0)


@pytest.mark.parametrize("world,deep", [(2, False), (4, False), (2, True)])
@torch.no_grad()
def test_fixed_sp_prefix_and_uneven_three_branch_joint_match_same_sp(
    model, monkeypatch, world, deep
):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.models.dits import zing_h3
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        sampling,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_sp_cpu import _CPUCollectives

    arch = replace(
        model.arch,
        num_attention_heads=4,
        hidden_size=32,
        adaln_out_features=6 * model.arch.num_modalities * 32,
        final_adaln_out_features=64,
    )
    if deep:
        monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", "1")
        arch = replace(
            arch,
            num_layers=50,
            token_refiner_num_layers=2,
            hidden_size=256,
            num_attention_heads=2,
            attention_head_dim=128,
            ffn_hidden_size=512,
            num_modalities=5,
            modality_tags=dict(
                video=4, text=0, audio=3, source_visual=1, condition_visual=2
            ),
            adaln_out_features=6 * 5 * 256,
            final_adaln_out_features=512,
        )
    base = zing_h3.ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
    torch.manual_seed(17)
    for name, value in base.named_parameters():
        value.copy_(
            torch.ones_like(value)
            if "norm" in name and name.endswith("weight")
            else torch.randn_like(value) * 0.08
        )
    base.rope.inv_freq.fill_(1)
    rank_models = [
        zing_h3.ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
        for _ in range(world)
    ]
    for rank_model in rank_models:
        rank_model.load_state_dict(base.state_dict())
    packer, inputs, _, _ = request(base, audio=False)
    packer = type(packer)(arch, num_frame_per_block=[3, 2])
    inputs.visual_shape = (15, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 15, 2, 4)
    visual = torch.randn(1, 2, 15, 2, 2)
    comm = _CPUCollectives(world)
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (world, comm.thread.rank))
    monkeypatch.setattr(
        zing_h3,
        "get_sp_group",
        lambda: SimpleNamespace(ulysses_group=None, cpu_group=comm),
    )
    monkeypatch.setattr(zing_h3, "gather_sequence_varlen", comm.gather)
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: world)
    monkeypatch.setattr(usp, "get_ulysses_parallel_rank", lambda: comm.thread.rank)
    monkeypatch.setattr(usp, "_ipc_ready_group", lambda: None)
    monkeypatch.setattr(usp, "_usp_all_to_all_single_varlen", comm.all_to_all)
    monkeypatch.setattr(sampling.dist, "all_gather_object", comm.gather_objects)
    original = ZingH3Sampler._forward

    def forward(sampler, packed, cache, phase):
        if phase != "prefix" and not (
            phase == "joint" and len(packed["_zing_joint_phases"]) == 3
        ):
            return original(sampler, packed, cache, phase)
        rows = packed["position_ids"].shape[0]
        clean = int(packed["clean_seq_lens"].sum())
        ends = (
            (rows,)
            if phase == "prefix"
            else tuple(
                int((packed["block_idx"] <= value).sum())
                for value in packed["block_idx"].unique_consecutive()
            )
        )
        current, history_positions = cache.attention_positions(
            packed["position_ids"],
            prefix=phase == "prefix",
            host_current=packed["_zing_host_layout"].position_ids,
        )
        history = 0 if history_positions is None else history_positions.shape[0]
        capacity = 256 if deep else 1024
        stores = tuple(
            (
                torch.zeros(
                    capacity,
                    arch.num_attention_heads // world,
                    arch.attention_head_dim,
                    dtype=torch.bfloat16,
                ),
                torch.zeros(
                    capacity,
                    arch.num_attention_heads // world,
                    arch.attention_head_dim,
                    dtype=torch.bfloat16,
                ),
            )
            for _ in base.blocks
        )
        if history:
            for layer, (k, v) in enumerate(stores):
                hk, hv = cache.history(layer)
                k[:history].copy_(hk)
                v[:history].copy_(hv)
        assert history_positions is None or history_positions.dtype == current.dtype
        rotated = tuple(torch.zeros_like(k) for k, _ in stores)
        kv = _kv_plan(
            stores,
            rotated,
            capacity=capacity,
            packed_rows=rows,
            clean=clean,
            ends=ends,
            history=history,
            current=current,
            history_positions=history_positions,
        )
        plan = sampler.generator.prepare_fixed_forward(
            packed=packed, kv_plan=kv, phase=phase
        )
        actual = tuple(value.clone() for value in plan.execute())
        expected = original(sampler, packed, cache, phase)
        comm.thread.pairs.extend(
            (a.clone(), b.clone()) for a, b in zip(actual, expected)
        )
        if phase == "prefix":
            comm.thread.prefix += 1
        else:
            start = sum(
                zing_h3.compute_sequence_splits(rows, world)[: comm.thread.rank]
            )
            local = zing_h3.compute_sequence_splits(rows, world)[comm.thread.rank]
            comm.thread.joint.append((ends, start, local))
        return expected

    monkeypatch.setattr(ZingH3Sampler, "_forward", forward)

    if deep:
        original_randn_like = torch.randn_like

        def rank_randn_like(value, **kwargs):
            if hasattr(comm.thread, "rng"):
                return torch.randn(
                    value.shape,
                    dtype=value.dtype,
                    device=value.device,
                    generator=comm.thread.rng,
                    **kwargs,
                )
            return original_randn_like(value, **kwargs)

        monkeypatch.setattr(torch, "randn_like", rank_randn_like)

    @torch.no_grad()
    def worker(rank):
        comm.thread.rank = rank
        if deep:
            # Each CPU thread emulates an independent rank's identical RNG.
            comm.thread.rng = torch.Generator().manual_seed(97)
        comm.thread.prefix = 0
        comm.thread.joint = []
        comm.thread.pairs = []
        schedule = (
            dict(
                solver="few_step",
                denoising_step_list=[1000, 750, 500, 250],
                video_shift=8,
                audio_shift=8,
            )
            if deep
            else dict(sigma_points=3)
        )
        sampler = ZingH3Sampler(
            rank_models[rank],
            packer,
            **schedule,
            joint_forward="cross_chunk",
            rope_position_mode="block_relative",
        )
        sampler.sample(inputs, visual)
        return comm.thread.prefix, comm.thread.joint, comm.thread.pairs

    with ThreadPoolExecutor(max_workers=world) as executor:
        futures = [executor.submit(worker, rank) for rank in range(world)]
        results, errors = [], []
        for rank, future in enumerate(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append((rank, repr(error)))
    assert not errors, f"Fixed SP worker failures: {errors}"
    for _, _, pairs in results:
        for actual, expected in pairs:
            torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)
    assert all(prefix == 1 and joints for prefix, joints, _ in results)
    if world == 4:
        assert any(
            start < stop < start + local
            for _, joints, _ in results
            for ends, start, local in joints
            for stop in ends[:-1]
        )
        assert any(
            stop <= start or begin >= start + local
            for _, joints, _ in results
            for ends, start, local in joints
            for begin, stop in zip((0, *ends[:-1]), ends)
        )


def _rope(model, positions):
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        _rope_cos_sin_cache,
    )

    return _rope_cos_sin_cache(model.rope(positions[None]), dtype=torch.bfloat16)


@torch.no_grad()
def test_slot_attention_inputs_match_the_gathered_reference_bitwise(model):
    """In-slot K/V equal the gather + where + full-capacity rotation they replace."""
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        _apply_rope,
        _apply_rope_into,
    )

    torch.manual_seed(5)
    capacity, history, rows, clean = 24, 7, 5, 3
    heads, dim = model.arch.num_attention_heads, model.arch.attention_head_dim
    old_k = torch.randn(capacity, heads, dim).bfloat16()
    old_v = torch.randn(capacity, heads, dim).bfloat16()
    # Rows past the history are stale slot contents that must never matter.
    stale_k, stale_v = old_k.clone(), old_v.clone()
    current_k = torch.randn(rows, heads, dim).bfloat16()
    current_v = torch.randn(rows, heads, dim).bfloat16()
    history_positions = torch.randint(0, 40, (history, 3))
    current_positions = torch.randint(0, 40, (rows, 3))

    # Reference: the original padded gather with a zeroed tail.
    gather = torch.tensor(
        [*range(history), *range(capacity, capacity + rows)]
        + [0] * (capacity - history)
    )
    valid = (torch.arange(capacity + rows) < history + rows)[:, None, None]
    padded = torch.cat((history_positions, current_positions)).new_zeros(
        capacity + rows, 3
    )
    padded[: history + rows] = torch.cat((history_positions, current_positions))
    reference_k = torch.where(
        valid,
        _apply_rope(
            torch.where(
                valid, torch.cat((old_k, current_k)).index_select(0, gather), 0
            ),
            _rope(model, padded),
        ),
        0,
    )
    reference_v = torch.where(
        valid, torch.cat((old_v, current_v)).index_select(0, gather), 0
    )

    # In-slot path, exactly as the fixed attention executes it.
    stores = ((stale_k, stale_v),)
    rotated = (torch.full_like(old_k, 3.0),)
    plan = _kv_plan(
        stores,
        rotated,
        capacity=capacity,
        packed_rows=rows,
        clean=clean,
        ends=(rows,),
        history=history,
        current=current_positions,
        history_positions=history_positions,
    )
    full = new_positions = plan.tensors["positions"]
    assert new_positions.shape == (capacity, 3)
    _apply_rope_into(stale_k, _rope(model, full), rotated[0])
    plan.write_current(
        0, _apply_rope(current_k, _rope(model, current_positions)), current_k, current_v
    )
    live = history + rows
    assert torch.equal(rotated[0][:live], reference_k[:live])
    assert torch.equal(stale_v[:live], reference_v[:live])
    # Clean raw K persists unrotated at the write rows; active rows do not.
    assert torch.equal(stale_k[history : history + clean], current_k[:clean])
    assert torch.equal(stale_k[history + clean :], old_k[history + clean :])
    assert torch.equal(stale_k[:history], old_k[:history])


@torch.no_grad()
def test_fixed_forward_never_reads_rows_past_the_live_length(model):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, None)
    packed = sampler._select_block(layout, 2, phase="target", video_t=0.3, audio_t=1.0)
    rows = packed["position_ids"].shape[0]
    history = 5
    results = []
    for poison in (None, float("nan"), float("inf")):
        kv = _workspace(model, packed, history=history)
        if poison is not None:
            for k, v in kv.stores:
                k[history + rows :] = poison
                v[history + rows :] = poison
            for rotated in kv.rotated:
                rotated[history + rows :] = poison
        plan = model.prepare_fixed_forward(packed=packed, kv_plan=kv, phase="active")
        results.append(tuple(value.clone() for value in plan.execute()))
    for value in results[0]:
        assert torch.isfinite(value).all()
    for other in results[1:]:
        for actual, expected in zip(other, results[0]):
            assert torch.equal(actual, expected)


def _fake_cache(model_layers, capacity, chunk_rows):
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        allocate_layer_stores,
    )

    chunks, offset = [], 0
    for count in chunk_rows:
        chunks.append(
            SimpleNamespace(offset=offset, positions=torch.zeros(count, 3).long())
        )
        offset += count
    history = torch.arange(offset * 3).reshape(offset, 3) if offset else None
    return SimpleNamespace(
        storage=allocate_layer_stores(
            capacity,
            num_layers=model_layers,
            local_heads=2,
            head_dim=8,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        ),
        capacity=capacity,
        _length=offset,
        _history=lambda: chunks,
        attention_positions=lambda current, prefix=False, host_current=None: (
            current + 100,
            history,
        ),
    )


def test_plan_writes_front_rows_and_shares_one_rotation_per_history(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_HISTORY_ROPE_LIVE_ROWS", "0")
    current = torch.arange(12).reshape(4, 3)

    def plan(cache, *, after=None, clean=0):
        return ZingFixedForwardKVPlan(
            cache, current, ends=(2, 4), clean_rows=clean, after=after
        )

    cache = _fake_cache(2, 32, (3, 4))
    first = plan(cache, clean=2)
    assert first.rotates_history and first.history_positions.shape == (7, 3)
    assert sorted(first.tensors) == [
        "cu_k.0",
        "cu_k.1",
        "cu_q.0",
        "cu_q.1",
        "current_positions",
        "positions",
        "write_rows",
    ]
    assert torch.equal(first.tensors["write_rows"], torch.arange(7, 11))
    assert first.tensors["cu_k.0"].tolist() == [0, 9]
    assert first.tensors["cu_k.1"].tolist() == [0, 11]
    assert first.tensors["cu_q.0"].tolist() == [0, 2]
    assert first.tensors["cu_q.1"].tolist() == [0, 2]
    positions = first.tensors["positions"]
    assert positions.shape == (32, 3)
    assert torch.equal(positions[:7], first.history_positions)
    assert not positions[7:].any()
    assert torch.equal(first.tensors["current_positions"], current + 100)
    assert [id(r) for r in first.rotated] == [id(s.rotated_k) for s in cache.storage]

    # Equal coordinates and no write into the history: the rotation is reused,
    # also through a plan that reused it itself.
    second = plan(cache, after=first)
    third = plan(cache, after=second)
    for reused in (second, third):
        assert not reused.rotates_history and "positions" not in reused.tensors
    # A longer history (rows were persisted) is a new rotation.
    longer = plan(_fake_cache(2, 32, (3, 4, 2)), after=third)
    assert longer.rotates_history
    # No history means nothing to rotate, and nothing is inherited from it.
    empty = plan(_fake_cache(2, 32, ()), after=longer)
    assert not empty.rotates_history and empty.rotated_positions is None
    assert plan(cache, after=empty).rotates_history


def test_history_live_length_has_stable_device_metadata_shape(monkeypatch):
    monkeypatch.delenv("SGLANG_ZING_HISTORY_ROPE_LIVE_ROWS", raising=False)
    current = torch.arange(12).reshape(4, 3)
    for chunks in ((3,), (3, 4), (2,)):
        plan = ZingFixedForwardKVPlan(
            _fake_cache(1, 32, chunks), current, ends=(4,), clean_rows=0
        )
        live = plan.tensors["history_live_rows"]
        assert live.dtype == torch.int32 and live.shape == (1,)
        assert live.item() == sum(chunks)
        assert live.device == plan.stores[0][0].device


@torch.no_grad()
@pytest.mark.parametrize(
    "mutation", ["raw_layer_1", "spatial_coordinate", "unpublished"]
)
def test_cross_graph_prefix_requires_all_layers_and_exact_coordinates(
    monkeypatch, mutation
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

    monkeypatch.delenv("SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH", raising=False)
    storage = _fake_cache(2, 32, ()).storage
    cache = ZingH3KVCache(
        2,
        storage=storage,
        rope_position_mode="block_relative",
        window_policy=ZingWindowPolicy(3, 1),
    )
    positions = torch.tensor([[0, 0, 0], [0, 1, 0]])
    blocks = torch.zeros(2, dtype=torch.long)
    cache.update(
        positions,
        blocks,
        "prefix",
        0,
        [
            (torch.randn(2, 2, 8).bfloat16(), torch.randn(2, 2, 8).bfloat16())
            for _ in storage
        ],
        validation=(positions, blocks),
    )

    def prepare():
        shadow = cache.planning_copy()
        current = torch.tensor([[1, 0, 0]])
        plan = ZingFixedForwardKVPlan(shadow, current, ends=(1,), clean_rows=0)
        shadow.bind_attention_positions(plan.history_positions)
        plan.prepare_history_reuse(shadow)
        return plan, shadow

    first, shadow = prepare()
    assert first.tensors["history_rope_range"].tolist() == [0, 2]
    ZingChunkGraphs._discard_rotated(cache, shadow)
    cache.publish_plan(shadow)
    reused, _ = prepare()
    assert reused.tensors["history_rope_range"].tolist() == [2, 2]
    if mutation == "raw_layer_1":
        storage[1].k[0].add_(1)
    elif mutation == "spatial_coordinate":
        cache.prefix.host_positions[0, 1] += 1
    else:
        cache._rotated_store_signatures.clear()
    rebuilt, shadow = prepare()
    assert rebuilt.tensors["history_rope_range"].tolist() == [0, 2]
    if mutation == "raw_layer_1":
        assert shadow._raw_write_epochs.get(0, 0) == 0
        assert shadow._raw_write_epochs[1] == 1
    assert reused.tensors.keys() == rebuilt.tensors.keys()
    assert (
        reused.tensors["history_rope_range"].shape
        == rebuilt.tensors["history_rope_range"].shape
    )


def test_plan_host_inputs_use_pinned_asynchronous_copies(monkeypatch):
    from sglang.multimodal_gen.runtime.models.dits import zing_h3_cache

    # A pageable host-to-device copy synchronizes the stream, which would make
    # planning wait for the previous graph. Every host-built input is staged.
    staged = []

    def record(tensor, device):
        assert tensor.device.type == "cpu"
        staged.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(zing_h3_cache, "_host_to_device", record)
    monkeypatch.delenv("SGLANG_ZING_HISTORY_ROPE_LIVE_ROWS", raising=False)
    current = torch.arange(12).reshape(4, 3)
    ZingFixedForwardKVPlan(
        _fake_cache(1, 32, (3, 4)), current, ends=(2, 4), clean_rows=2
    )
    assert sorted(staged) == sorted([(4, 3), (1,), (32, 3), (2,), (2,), (2,), (2,)])


def test_host_to_device_copies_from_pinned_memory_without_blocking():
    from unittest.mock import MagicMock

    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        _host_to_device,
    )

    host = MagicMock()
    host.device = torch.device("cpu")
    _host_to_device(host, torch.device("cuda", 0))
    host.pin_memory.return_value.to.assert_called_once_with(
        torch.device("cuda", 0), non_blocking=True
    )
    cpu = torch.ones(2)
    assert _host_to_device(cpu, torch.device("cpu")) is cpu


def test_plan_admission_covers_every_row_written_into_the_slot():
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        ZingKVCapacityError,
    )

    current = torch.zeros(4, 3).long()
    # Active rows are not persisted but still occupy the slot as scratch.
    with pytest.raises(ZingKVCapacityError):
        ZingFixedForwardKVPlan(
            _fake_cache(1, 10, (4, 3)), current, ends=(4,), clean_rows=0
        )
    ZingFixedForwardKVPlan(_fake_cache(1, 11, (4, 3)), current, ends=(4,), clean_rows=0)
    gapped = _fake_cache(1, 32, (3, 4))
    gapped._history()[1].offset = 5
    with pytest.raises(ValueError, match="front rows"):
        ZingFixedForwardKVPlan(gapped, current, ends=(4,), clean_rows=0)
    stale = _fake_cache(1, 32, (3, 4))
    stale._length = 9
    with pytest.raises(ValueError, match="every live row"):
        ZingFixedForwardKVPlan(stale, current, ends=(4,), clean_rows=0)


def test_pool_slots_start_finite():
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        allocate_layer_stores,
    )

    for store in allocate_layer_stores(
        6,
        num_layers=2,
        local_heads=2,
        head_dim=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    ):
        for tensor in (store.k, store.v, store.rotated_k):
            assert not tensor.any()


@pytest.mark.parametrize("fused_kv", [False, True])
@torch.no_grad()
def test_fixed_forward_rotates_current_qk_once_across_kv_routes(
    model, monkeypatch, fused_kv
):
    from sglang.multimodal_gen.runtime.models.dits import zing_h3
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _apply_rope

    packer, inputs, visual, _ = request(model, audio=False)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, None)
    packed = sampler._select_block(layout, 1, phase="source", video_t=1.0, audio_t=1.0)
    expected = sampler._forward(packed, None, None)
    calls = []

    def current(x, rope):
        calls.append("current")
        return _apply_rope(x, rope)

    def current_kv(q, k, v, rope, plan, layer, tensors, *, default=False):
        calls.append("kv")
        if not fused_kv:
            return None
        plan.write_current(layer, _apply_rope(k, rope), k, v, tensors=tensors)
        return _apply_rope(q, rope)

    monkeypatch.setattr(zing_h3, "apply_current_rope", current)
    monkeypatch.setattr(zing_h3, "apply_current_rope_kv", current_kv)
    kv = _workspace(model, packed, clean=True)
    actual = model.prepare_fixed_forward(
        packed=packed, kv_plan=kv, phase="source"
    ).execute()
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=0.008, rtol=0.008)
    assert calls.count("kv") == len(model.blocks)
    assert calls.count("current") == (0 if fused_kv else 2 * len(model.blocks))


def test_target_geometry_separates_fixed_forward_signatures(model):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, None)
    packed = sampler._select_block(layout, 2, phase="target", video_t=0.3, audio_t=1.0)
    kv = _workspace(model, packed, history=3)

    def signature(grid):
        return model.prepare_fixed_forward(
            packed=packed, kv_plan=kv, phase="active", target_grid=grid
        ).signature

    legacy = signature(None)
    assert signature((15, 26)) == legacy
    assert signature((40, 23)) == legacy
    assert signature((23, 40)) != legacy


def test_adaln_table_project_plan_caches_concatenated_layer_projections():
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
        ZingAdalnTable,
    )

    adaln = ZingAdalnTable.__new__(ZingAdalnTable)
    adaln.device = torch.device("cpu")
    adaln.ready = None
    adaln.hits = 0
    adaln._plan_cache = {}
    adaln.values = {
        f"batch.{b}.slot.{slot}": torch.randn(
            4, (2 if slot == "final" else 6) * 8, dtype=torch.bfloat16
        )
        for b in range(3)
        for slot in (0, 1, "final")
    }
    first = adaln.project_plan([0, 2], 2)
    second = adaln.project_plan([0, 2], 2)
    assert first is second
    assert adaln.hits == 12
    for slot in (0, 1, "final"):
        parts = [adaln.project(slot, None, None, idx) for idx in (0, 2)]
        for i, values in enumerate(zip(*parts)):
            torch.testing.assert_close(
                first[f"projection.{slot}.{i}"], torch.cat(values)
            )


@torch.no_grad()
def test_planning_copy_store_chunk_skips_layer_tensor_views():
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
    from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

    storage = _fake_cache(2, 32, ()).storage
    cache = ZingH3KVCache(
        2,
        storage=storage,
        rope_position_mode="block_relative",
        window_policy=ZingWindowPolicy(3, 1),
    )
    positions = torch.tensor([[0, 0, 0], [0, 1, 0]])
    blocks = torch.zeros(2, dtype=torch.long)
    cache.update(
        positions,
        blocks,
        "prefix",
        0,
        [
            (
                torch.zeros(2, 2, 8, dtype=torch.bfloat16),
                torch.zeros(2, 2, 8, dtype=torch.bfloat16),
            )
            for _ in storage
        ],
        validation=(positions, blocks),
    )
    shadow = cache.planning_copy()
    src_pos = torch.tensor([[1, 0, 0], [1, 1, 0]])
    src_blk = torch.ones(2, dtype=torch.long)
    before = [(s.k.clone(), s.v.clone()) for s in storage]
    shadow.update(
        src_pos,
        src_blk,
        "source",
        1,
        [
            (
                torch.ones(2, 2, 8, dtype=torch.bfloat16),
                torch.ones(2, 2, 8, dtype=torch.bfloat16),
            )
            for _ in storage
        ],
        validation=(src_pos, src_blk),
    )
    # Planning publishes lazy ranges without materializing per-layer views or
    # writing source data into the live cache's authoritative store.
    assert shadow.pending_source.offset == 2
    assert shadow.pending_source.layers._views == {}
    assert shadow._length == 4
    assert cache.pending_source is None and cache._length == 2
    for store, (keys, values) in zip(storage, before, strict=True):
        torch.testing.assert_close(store.k, keys, rtol=0, atol=0)
        torch.testing.assert_close(store.v, values, rtol=0, atol=0)
