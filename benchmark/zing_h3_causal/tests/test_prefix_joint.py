# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811
import copy
from types import SimpleNamespace

import prefix_joint as pj
import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
    ZingChunkGraphs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    ZingH3Packer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
    model,
    request,
)


@pytest.mark.parametrize("rope", ["absolute", "block_relative"])
@pytest.mark.parametrize("refs", [False, True])
@torch.no_grad()
def test_prefix_joint_model_and_kv(model, rope, refs):
    # Exercise the mixed timestep grouping responsible for the real-fixture error.
    model.arch.ar_condition_timestep = 0.0
    model.arch.reference_visual_timestep = 0.999

    _, inputs, _, _ = request(model, audio=False)
    if refs:
        # Use the actual packer's native reference convention, from a known
        # model contract test, rather than adding anonymous token padding.
        inputs.visual = {"r": torch.randn(1, 2, 1, 2, 2)}
        inputs.media = [dict(id="r", type="image")]
        inputs.conditions = [dict(content_id="r", role="reference")]
    inputs.visual_shape = (8, 2, 2)
    inputs.source_visual = None
    history = torch.randn(1, 2, 5, 2, 2)
    noise = torch.randn(1, 2, 3, 2, 2)
    packer = ZingH3Packer(
        model.arch, num_frame_per_block=[3, 2], split_first_latent=True
    )
    sampler = ZingH3Sampler(
        model,
        packer,
        solver="few_step",
        denoising_step_list=[1000, 750],
        window_chunks=6,
        sink_chunks=3,
        rope_position_mode=rope,
        window_eviction_timing="after_step0",
        joint_forward="cross_chunk",
        ar_condition_timestep=model.arch.ar_condition_timestep,
        reference_visual_timestep=model.arch.reference_visual_timestep,
    )
    sampler.device = torch.device("cpu")
    sampler.video_scheduler.to("cpu")

    def cache():
        return ZingH3KVCache(
            model.arch.num_layers,
            window_policy=sampler.window_policy,
            rope_position_mode=rope,
            rope_max_frame_gap=12,
            window_eviction_timing="after_step0",
        )

    a = cache()
    origin, _ = sampler.rebuild_prompt_prefix(inputs, inputs.positive, history, a)
    current_inputs = copy.copy(inputs)
    current_inputs.source_visual = torch.randn(1, 2, 3, 2, 2)
    layout = packer.pack_chunk(
        current_inputs,
        inputs.positive,
        noise,
        None,
        chunk=packer.chunk_plan.chunk(3),
        media_origin=origin,
    )
    current = SimpleNamespace(positive=layout, spans=[(7, 8, 0, 3)], visual=noise)
    packs = pj.packs_for(sampler, inputs, inputs.positive, history, current)
    assert bool((packs[0]["row_role"] == 4).any()) == refs
    a.begin_chunk(8)
    v, _ = sampler._forward(
        ZingChunkGraphs._joint_packed(packs[-2:], ("source", "active")), a, "joint"
    )
    expected = v[-packs[-1]["x"].shape[0] :]
    b = cache()
    actual, _ = pj.joint(sampler, packs, b)
    assert "_forward_body" not in model.__dict__
    assert "_projection" not in model.__dict__
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)
    assert a.last_block == b.last_block == 6 and a.active[0] == b.active[0] == 8
    torch.testing.assert_close(a.positions(), b.positions(), atol=0, rtol=0)
    for layer in range(model.arch.num_layers):
        for got, want in zip(b.history(layer), a.history(layer)):
            torch.testing.assert_close(got, want, atol=0.008, rtol=0.008)


@pytest.mark.parametrize("second_device", ["cpu", "cuda"])
def test_metadata_snapshot_mixed_devices(second_device):
    if second_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA only")
    first = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    second = torch.tensor([[6.0, 7.0, 8.0]], device=second_device)
    cache = SimpleNamespace(
        _history=lambda: [
            SimpleNamespace(positions=first),
            SimpleNamespace(positions=second),
        ]
    )
    value = pj.host_positions(cache)
    torch.testing.assert_close(value, torch.arange(9.0).reshape(3, 3), atol=0, rtol=0)
    assert first.device.type == "cpu" and second.device.type == second_device


@torch.no_grad()
def test_grouped_projection_preserves_prefix_batch_and_shared_zero(model):
    model.arch.ar_condition_timestep = 0.0
    model.arch.reference_visual_timestep = 0.999

    def pack(times, roles):
        return dict(
            t=torch.tensor(times), row_role=torch.tensor(roles), x=torch.zeros(1)
        )

    packs = [
        pack([0.0, 0.999], [0, 4]),
        *[pack([1.0], [10]) for _ in range(3)],
        pack([1.0], [7]),
        pack([0.0], [10]),
    ]
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_joint import (
        prefix_modulation,
    )

    times, sizes, indices, local, ends = prefix_modulation(
        torch.cat([p["t"] for p in packs]),
        torch.cat([p["row_role"] for p in packs]),
        (2, 3, 4, 5, 6, 7),
        model.modality_tags,
        model.arch.num_modalities,
    )
    assert sizes == (2, 1, 1)
    assert times[0] == times[-1] == 0
    assert local.tolist() == [0, 1, 2, 2, 2, 2, 3]
    assert indices[-1] == 3 * model.arch.num_modalities + model.modality_tags["video"]
    assert ends == (1, 2, 3, 4, 5, 6)


@torch.no_grad()
@pytest.mark.parametrize("refs", [False, True])
@pytest.mark.parametrize("budget", [0, 1 << 20])
def test_dynamic_prefix_cache_matches_resident_and_reuses_active(model, refs, budget):
    from test_prefix_joint_graph import setup_case

    sampler, cache, make_packs = setup_case(model, "block_relative")
    model.adaln_projection_cache_enabled = False
    packs = make_packs(refs)
    reference = cache()
    expected, _ = pj.joint(sampler, packs, reference)
    expected_active, _ = sampler._forward(packs[-1], reference, "active")

    model.adaln_projection_cache_enabled = True
    model.adaln_projection_cache_bytes = budget
    state = cache()
    modules = [block.adaln_proj for block in model.blocks]
    modules.append(model.final_layer.adaln_proj)
    batches = [[] for _ in modules]
    hooks = [
        module.register_forward_pre_hook(
            lambda _module, args, seen=seen: seen.append(args[0].shape[0])
        )
        for module, seen in zip(modules, batches)
    ]
    try:
        for _ in range(2):
            # Prompt replacement/session reset must discard old projections.
            actual, _ = pj.joint(sampler, packs, state)
            projections = state.adaln_projection_cache
            assert projections is not None
            groups = [2, 1, 1] if refs else [1, 1]
            assert all(seen == groups for seen in batches)
            assert projections.hits == 0
            assert projections.misses == (len(groups) * len(modules) if budget else 0)
            torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)

            active, _ = sampler._forward(packs[-1], state, "active")
            torch.testing.assert_close(active, expected_active, atol=0.008, rtol=0.008)
            assert projections.hits == (len(modules) if budget else 0)
            assert all(seen == groups + ([] if budget else [1]) for seen in batches)
            assert state.last_block == reference.last_block
            assert state.active[0] == reference.active[0]
            assert torch.equal(state.active[1], reference.active[1])
            assert torch.equal(state.positions(), reference.positions())
            for layer in range(model.arch.num_layers):
                for got, want in zip(state.history(layer), reference.history(layer)):
                    torch.testing.assert_close(got, want, atol=0.008, rtol=0.008)
            state.clear()
            assert state.adaln_projection_cache is None
            assert projections.bytes == 0 and not projections.entries
            for seen in batches:
                seen.clear()
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def test_native_prefix_failure_clears_cache(model, monkeypatch):
    from test_prefix_joint_graph import setup_case

    sampler, cache, make_packs = setup_case(model, "block_relative")
    packs, state = make_packs(), cache()
    original = type(model)._forward_body

    def fail(*args, **kwargs):
        raise RuntimeError("injected forward failure")

    monkeypatch.setattr(model, "_block_forward", fail)
    with pytest.raises(RuntimeError, match="injected forward failure"):
        pj.joint(sampler, packs, state)
    assert state.prefix is None and not state.committed and state.active is None
    assert state.adaln_projection_cache is None
    assert type(model)._forward_body is original
    assert "_forward_body" not in model.__dict__
