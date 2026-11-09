# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811
"""Joint forwards must preserve the sequential branch/cache oracle."""

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    sampling,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model  # noqa: F401
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
    request,
)


def _replicated_eligibility_votes(output, value, group=None):
    """Single-lane eligibility seams assume matching peer state, without model work."""
    output[:] = [value] * len(output)


@pytest.mark.parametrize("mode", ["source_first", "cross_chunk"])
@pytest.mark.parametrize("window,sink", [(21, 3), (3, 1), (1, 0)])
@pytest.mark.parametrize("rope_mode", ["absolute", "block_relative"])
def test_joint_matches_sequential(model, mode, window, sink, rope_mode):
    packer, inputs, visual, _ = request(model, audio=False)
    inputs.visual_shape = (12, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 12, 2, 2)
    visual = torch.randn(1, 2, 12, 2, 2)
    packer = type(packer)(model.arch, num_frame_per_block=[1, 2, 2])
    outputs, runs = [], []
    for selected in ("off", mode):
        sampler = ZingH3Sampler(
            model,
            packer,
            sigma_points=3,
            window_chunks=window,
            sink_chunks=sink,
            rope_position_mode=rope_mode,
            joint_forward=selected,
        )
        torch.manual_seed(91)
        outputs.append(sampler.sample(inputs, visual)[0])
        runs.append(sampler.last_run_metrics)
    torch.testing.assert_close(outputs[1], outputs[0], atol=0.008, rtol=0.008)
    assert runs[1]["forward_counts"]["source_first"] > 0
    assert runs[1]["dit_forwards"] < runs[0]["dit_forwards"]
    for kind, shapes in runs[1]["joint_row_shapes"].items():
        assert sum(shapes.values()) == runs[1]["forward_counts"][kind]
        assert all(
            len(shape.split("+")) == (2 if kind == "source_first" else 3)
            for shape in shapes
        )
    if window == 21:
        blocks = len(runs[0]["block_spans"])
        steps = runs[0]["video_denoise_steps_per_block"]
        assert runs[0]["dit_forwards"] == (steps + 2) * blocks + 1
        expected = (
            (steps + 1) * blocks + 1 if mode == "source_first" else steps * blocks + 2
        )
        assert runs[1]["dit_forwards"] == expected
    if mode == "cross_chunk" and window > 1:
        assert runs[1]["forward_counts"]["cross_chunk"] > 0


def test_joint_option_rejects_typo(model):
    packer, *_ = request(model)
    with pytest.raises(ValueError, match="joint_forward"):
        ZingH3Sampler(model, packer, joint_forward="typo")


@pytest.mark.parametrize("ends", [(2, 5), (1, 3, 5), (1, 3, 4, 5), (1, 2, 3, 4, 5)])
def test_branch_attention_is_block_bidirectional(ends):
    import torch.nn.functional as F

    from sglang.multimodal_gen.runtime.models.dits.zing_h3 import _joint_attention

    torch.manual_seed(12)
    q = torch.randn(5, 2, 8)
    k, v = torch.randn(8, 2, 8), torch.randn(8, 2, 8)
    actual = _joint_attention(q, k, v, ends, 0.3, "torch_sdpa", None)
    mask = torch.zeros(5, 8, dtype=torch.bool)
    start = 0
    for stop in ends:
        mask[start:stop, : 3 + stop] = True
        start = stop
    expected = F.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        attn_mask=mask,
        scale=0.3,
    ).transpose(0, 1)
    torch.testing.assert_close(actual, expected)
    changed = v.clone()
    changed[3 + ends[0] :] += 100
    other = _joint_attention(q, k, changed, ends, 0.3, "torch_sdpa", None)
    torch.testing.assert_close(other[: ends[0]], actual[: ends[0]], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["source_first", "cross_chunk"])
def test_clean_kv_and_active_geometry_match(model, monkeypatch, mode):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache

    packer, inputs, visual, _ = request(model, audio=False)
    snapshots = []
    original = ZingH3KVCache.update

    def update(self, positions, blocks, phase, frames, layers=None, validation=None):
        original(self, positions, blocks, phase, frames, layers, validation)
        if phase != "prefix":
            snapshots.append(
                (
                    phase,
                    blocks.clone(),
                    positions.clone(),
                    None
                    if layers is None
                    else [(k.clone(), v.clone()) for k, v in layers],
                )
            )

    monkeypatch.setattr(ZingH3KVCache, "update", update)
    runs = []
    for selected in ("off", mode):
        snapshots.clear()
        ZingH3Sampler(model, packer, sigma_points=3, joint_forward=selected).sample(
            inputs, visual
        )
        runs.append(list(snapshots))
    assert len(runs[0]) == len(runs[1])
    for a, b in zip(*runs):
        assert a[0] == b[0]
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
        torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
        if a[3] is not None:
            for (ak, av), (bk, bv) in zip(a[3], b[3]):
                torch.testing.assert_close(ak, bk, rtol=0.008, atol=0.008)
                torch.testing.assert_close(av, bv, rtol=0.008, atol=0.008)


def test_joint_failure_clears_session(model, monkeypatch):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = ZingH3Sampler(model, packer, sigma_points=3, joint_forward="source_first")
    state = sampler.initialize(inputs, visual)

    def fail(*args, **kwargs):
        raise RuntimeError("injected output head failure")

    monkeypatch.setattr(model, "_project_outputs", fail)
    with pytest.raises(RuntimeError, match="injected"):
        sampler.step(state)
    assert state.closed
    assert state.cache.active is state.cache.pending_source is None
    assert state.cache.positions() is None


@pytest.mark.parametrize("audio,cfg,reason", [(True, 1, "audio"), (False, 2, "cfg")])
def test_unvalidated_combinations_fall_back(model, audio, cfg, reason):
    packer, inputs, visual, sound = request(model, audio=audio)
    sampler = ZingH3Sampler(
        model, packer, sigma_points=3, guidance_scale=cfg, joint_forward="cross_chunk"
    )
    actual = sampler.sample(inputs, visual, sound)
    expected = ZingH3Sampler(model, packer, sigma_points=3, guidance_scale=cfg).sample(
        inputs, visual, sound
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    if audio:
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert sampler.last_run_metrics["joint_fallbacks"][reason] > 0
    assert sampler.last_run_metrics["forward_counts"]["source_first"] == 0
    assert sampler.last_run_metrics["forward_counts"]["cross_chunk"] == 0


@pytest.mark.parametrize("mode", ["source_first", "cross_chunk"])
def test_joint_preserves_renoise_rng(model, mode):
    packer, inputs, visual, _ = request(model, audio=False)
    outputs, rng = [], []
    for selected in ("off", mode):
        torch.manual_seed(72)
        sampler = ZingH3Sampler(
            model,
            packer,
            solver="few_step",
            denoising_step_list=[1000, 750, 500, 250],
            joint_forward=selected,
        )
        outputs.append(sampler.sample(inputs, visual)[0])
        rng.append(torch.get_rng_state())
    torch.testing.assert_close(rng[0], rng[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.008, atol=0.008)


def test_cross_chunk_streaming_commits_before_delivery(model):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = ZingH3Sampler(model, packer, sigma_points=3, joint_forward="cross_chunk")
    state = sampler.initialize(inputs, visual)
    try:
        sampler.step(state)
        assert state.cache.last_block == 2
        assert state.cache.active is state.cache.pending_source is None
        assert sampler.last_run_metrics["joint_fallbacks"]["streaming_boundary"] == 1
        sampler.step(state)
        assert state.cache.last_block == 4
    finally:
        state.close()


@pytest.mark.parametrize("mode", ["source_first", "cross_chunk"])
@pytest.mark.parametrize("source_hw", [(2, 2), (4, 4)])
def test_joint_sp2_layout(model, monkeypatch, mode, source_hw):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.models.dits import zing_h3
    from sglang.multimodal_gen.test.unit.test_zing_h3_sp_cpu import _CPUCollectives

    packer, inputs, visual, _ = request(model, audio=False)
    # Real source and target can have different spatial token counts.
    inputs.source_visual = torch.randn(1, 2, 3, *source_hw)
    expected = ZingH3Sampler(model, packer, sigma_points=3).sample(inputs, visual)[0]
    comm = _CPUCollectives(2)
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (2, comm.thread.rank))
    monkeypatch.setattr(
        zing_h3,
        "get_sp_group",
        lambda: SimpleNamespace(ulysses_group=None, cpu_group=comm),
    )
    monkeypatch.setattr(zing_h3, "gather_sequence_varlen", comm.gather)
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: 2)
    monkeypatch.setattr(usp, "get_ulysses_parallel_rank", lambda: comm.thread.rank)
    monkeypatch.setattr(usp, "_ipc_ready_group", lambda: None)
    monkeypatch.setattr(usp, "_usp_all_to_all_single_varlen", comm.all_to_all)
    monkeypatch.setattr(sampling.dist, "all_gather_object", comm.gather_objects)

    @torch.no_grad()
    def worker(rank):
        comm.thread.rank = rank
        sampler = ZingH3Sampler(model, packer, sigma_points=3, joint_forward=mode)
        return sampler.sample(inputs, visual)[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, range(2)))
    for actual in results:
        torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)


def test_source_first_uses_pooled_scratch(model):
    from sglang.multimodal_gen.test.unit.test_zing_h3_kv_pool_cache import _storage

    packer, inputs, visual, _ = request(model, audio=False)
    outputs = []
    for mode in ("off", "source_first"):
        sampler = ZingH3Sampler(model, packer, sigma_points=3, joint_forward=mode)
        state = sampler.initialize(inputs, visual, storage=_storage(model, 40))
        try:
            while state.index < len(state.spans):
                sampler.step(state)
            outputs.append(state.visual.clone())
            assert state.cache.stats["store_allocations"] == 0
            if mode != "off":
                assert sampler.last_run_metrics["forward_counts"]["source_first"] == 2
        finally:
            state.close()
    torch.testing.assert_close(outputs[0], outputs[1], atol=0.008, rtol=0.008)


def test_different_rope_views_fall_back_before_model_work(model):
    packer, inputs, visual, _ = request(model, audio=False)
    outputs = []
    for mode in ("off", "source_first"):
        sampler = ZingH3Sampler(model, packer, sigma_points=3, joint_forward=mode)
        state = sampler.initialize(inputs, visual)
        try:
            pos = state.positive
            pos["position_ids"][pos["block_idx"] == 1, 0] += 100
            sampler.step(state)
            outputs.append(state.visual.clone())
            if mode != "off":
                assert (
                    sampler.last_run_metrics["joint_fallbacks"]["different_rope_views"]
                    == 1
                )
                assert sampler.last_run_metrics["forward_counts"]["source_first"] == 0
        finally:
            state.close()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def test_unvalidated_sp_size_is_observable_fallback(model, monkeypatch):
    from sglang.multimodal_gen.runtime.models.dits import zing_h3

    packer, inputs, visual, _ = request(model, audio=False)
    sampler = ZingH3Sampler(model, packer, sigma_points=3, joint_forward="source_first")
    state = sampler.initialize(inputs, visual)
    try:
        from types import SimpleNamespace

        monkeypatch.setattr(
            zing_h3, "get_sp_group", lambda: SimpleNamespace(cpu_group=object())
        )
        monkeypatch.setattr(
            sampling.dist,
            "all_gather_object",
            _replicated_eligibility_votes,
        )
        monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (8, 0))
        assert not sampler._joint_eligible(state.positive, state.cache, 1, 2, None)
        assert sampler.last_run_metrics["joint_fallbacks"]["sp_size"] == 1
    finally:
        state.close()


def test_cross_commit_failure_cannot_expose_previous_final(model, monkeypatch):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache

    packer, inputs, visual, _ = request(model, audio=False)
    original = ZingH3KVCache.update
    failed = []

    def update(cache, positions, blocks, phase, *args, **kwargs):
        if phase == "source" and int(blocks[0]) == 3:
            assert cache.last_block == 2  # first clean commit has run
            failed.append(cache)
            raise RuntimeError("injected second commit failure")
        return original(cache, positions, blocks, phase, *args, **kwargs)

    monkeypatch.setattr(ZingH3KVCache, "update", update)
    sampler = ZingH3Sampler(model, packer, sigma_points=3, joint_forward="cross_chunk")
    with pytest.raises(RuntimeError, match="second commit"):
        sampler.sample(inputs, visual)
    assert len(failed) == 1
    assert failed[0].last_block == 0
    assert failed[0].positions() is None
    assert failed[0].pending_source is failed[0].active is None


@pytest.mark.parametrize("mode", ["source_first", "cross_chunk"])
def test_target_step0_matches_sequential(model, monkeypatch, mode):
    packer, inputs, visual, _ = request(model, audio=False)
    original = ZingH3Sampler._forward
    snapshots = {}

    def forward(sampler, packed, cache, update):
        result = original(sampler, packed, cache, update)
        if update in ("active", "joint"):
            target = int(packed["block_idx"][-1])
            rows = int(
                ((packed["block_idx"] == target) & (packed["row_role"] == 10)).sum()
            )
            snapshots.setdefault(target, result[0][-rows:].clone())
        return result

    monkeypatch.setattr(ZingH3Sampler, "_forward", forward)
    runs = []
    for selected in ("off", mode):
        snapshots.clear()
        ZingH3Sampler(model, packer, sigma_points=3, joint_forward=selected).sample(
            inputs, visual
        )
        runs.append(dict(snapshots))
    assert runs[0].keys() == runs[1].keys() == {2, 4}
    for target in runs[0]:
        torch.testing.assert_close(
            runs[0][target], runs[1][target], rtol=0.008, atol=0.008
        )


@pytest.mark.parametrize("cached", [False, True])
def test_joint_preserves_scalar_timestep_projection_batches(model, cached):
    packer, inputs, visual, _ = request(model, audio=False)
    model.adaln_projection_cache_enabled = cached
    batches = []
    modules = [model.time_embedder, model.final_layer.adaln_proj]
    modules.extend(block.adaln_proj for block in model.blocks)
    hooks = [
        module.register_forward_pre_hook(
            lambda _module, args: batches.append(args[0].shape[0])
        )
        for module in modules
    ]
    try:
        sampler = ZingH3Sampler(
            model, packer, sigma_points=3, joint_forward="cross_chunk"
        )
        sampler.sample(inputs, visual)
    finally:
        for hook in hooks:
            hook.remove()
    assert sampler.last_run_metrics["forward_counts"]["cross_chunk"] > 0
    assert batches and set(batches) == {1}


@pytest.mark.parametrize("mode", ["source_first", "cross_chunk"])
def test_joint_keeps_output_head_branch_shapes(model, mode):
    packer, inputs, visual, _ = request(model, audio=False)
    # Source and target sizes differ, as in the real-checkpoint input.
    inputs.source_visual = torch.randn(1, 2, 3, 4, 4)
    batches = []
    hook = model.final_layer.video_out.register_forward_pre_hook(
        lambda _module, args: batches.append(args[0].shape[0])
    )
    try:
        sampler = ZingH3Sampler(model, packer, sigma_points=3, joint_forward=mode)
        sampler.sample(inputs, visual)
    finally:
        hook.remove()
    # The checkpoint plan is [2, 1]; joint traversal must not make a 10-row
    # source+target head or a 7-row final+source+target head.
    assert set(batches) <= {0, 1, 2, 4, 8}
    assert {1, 2, 4, 8} <= set(batches)
    assert sampler.last_run_metrics["forward_counts"]["source_first"] > 0


@pytest.mark.parametrize("window,sink", [(21, 3), (3, 1), (1, 0)])
@pytest.mark.parametrize("solver", ["euler", "few_step"])
@pytest.mark.parametrize("batch_size", [2, 20])
def test_ready_lookahead_matches_streaming_and_preserves_rng(
    model, window, sink, solver, batch_size
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance

    packer, inputs, _, _ = request(model, audio=False)
    packer = type(packer)(
        model.arch, num_frame_per_block=[3, 2], split_first_latent=True
    )
    inputs.visual_shape = (10, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 10, 2, 2)
    noise = torch.randn_like(inputs.source_visual)
    runs, outputs, rngs = [], [], []
    for mode in ("off", "cross_chunk"):
        sampler = ZingH3Sampler(
            model,
            packer,
            joint_forward=mode,
            window_chunks=window,
            sink_chunks=sink,
            solver=solver,
            sigma_points=3,
            denoising_step_list=[1000, 750, 500, 250] if solver == "few_step" else None,
        )
        session = SourceReadySession(
            sampler,
            inputs,
            seed=51,
            max_pending_chunks=20,
            max_pending_bytes=1000000,
            total_latent_frames=noise.shape[2],
        )
        parts = []
        for chunk in packer.chunk_plan.chunks(10):
            session.append(
                inputs.source_visual[:, :, chunk.start : chunk.end],
                noise[:, :, chunk.start : chunk.end],
                provenance=fixture_provenance(
                    chunk.chunk_id + 1, 20, chunk.start, chunk.end
                ),
            )
            if len(session.pending) < batch_size and chunk.end != 10:
                continue
            iterator = (
                session.drain_ready()
                if mode == "cross_chunk"
                else iter(session.step, None)
            )
            for result in iterator:
                parts.append(result.visual)
                assert session.cache.last_block == result.chunk.target_block
                assert session.committed_latent == result.chunk.end
                if not session.pending:
                    assert session.memory_snapshot()["execution_storage_bytes"] == 0
        outputs.append(torch.cat(parts, dim=2))
        rngs.append(session.rng.cpu_rng.clone())
        runs.append(sampler.last_run_metrics)
        session.finish()
        assert session.complete and session.closed
    torch.testing.assert_close(outputs[1], outputs[0], atol=0.008, rtol=0.008)
    assert torch.equal(*rngs)
    assert runs[1]["dit_forwards"] < runs[0]["dit_forwards"]
    if window > 1:
        assert runs[1]["forward_counts"]["cross_chunk"] > 0


def test_ready_lookahead_close_releases_prefetched_state(model):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance

    packer, inputs, noise, _ = request(model, audio=False)
    packer.num_frame_per_block = [3, 2]
    packer.split_first_latent = True
    sampler = ZingH3Sampler(model, packer, joint_forward="cross_chunk", sigma_points=3)
    session = SourceReadySession(
        sampler,
        inputs,
        seed=51,
        max_pending_chunks=20,
        max_pending_bytes=1000000,
        total_latent_frames=noise.shape[2],
    )
    for chunk in packer.chunk_plan.chunks(noise.shape[2]):
        session.append(
            inputs.source_visual[:, :, chunk.start : chunk.end],
            noise[:, :, chunk.start : chunk.end],
            provenance=fixture_provenance(
                chunk.chunk_id + 1, 20, chunk.start, chunk.end
            ),
        )
    iterator = session.drain_ready()
    next(iterator)
    for operation in (
        lambda: session.update_prompt(inputs.positive, 2),
        session.step,
        session.finish,
    ):
        with pytest.raises(ValueError):
            operation()
    iterator.close()
    assert session.closed and not session.pending and not session._drain_tensors
    assert all(value == 0 for value in session.memory_snapshot().values())


@pytest.mark.parametrize("mode", ["source_first", "cross_chunk"])
def test_joint_preserves_checkpoint_condition_timesteps_and_five_modalities(
    model, mode
):
    """The joint body must retain main's checkpoint-driven modality routing."""
    from dataclasses import replace

    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.models.dits.zing_h3 import ZingH3DiTModel

    arch = replace(
        model.arch,
        num_modalities=5,
        adaln_out_features=6 * 5 * model.arch.hidden_size,
        modality_tags=dict(
            video=4, text=0, audio=3, source_visual=1, condition_visual=2
        ),
        ar_condition_timestep=0.875,
        reference_visual_timestep=0.999,
    )
    candidate = ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
    torch.manual_seed(7)
    with torch.no_grad():
        for name, param in candidate.named_parameters():
            param.copy_(
                torch.ones_like(param)
                if "norm" in name and name.endswith("weight")
                else torch.randn_like(param) * 0.08
            )
        candidate.rope.inv_freq.fill_(1)
    packer, inputs, visual, _ = request(candidate, audio=False)
    inputs.visual = {"ref": torch.randn(1, 2, 1, 2, 2)}
    inputs.media = [{"id": "ref", "type": "image"}]
    inputs.conditions = [{"content_id": "ref", "role": "reference"}]
    outputs = []
    for selected in ("off", mode):
        sampler = ZingH3Sampler(
            candidate,
            packer,
            sigma_points=3,
            joint_forward=selected,
            ar_condition_timestep=arch.ar_condition_timestep,
            reference_visual_timestep=arch.reference_visual_timestep,
        )
        torch.manual_seed(91)
        outputs.append(sampler.sample(inputs, visual)[0])
        if selected != "off":
            assert sampler.last_run_metrics["forward_counts"][mode] > 0
    torch.testing.assert_close(outputs[1], outputs[0], atol=0.008, rtol=0.008)


@pytest.fixture
def four_head_model(model):
    from dataclasses import replace

    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.models.dits import zing_h3

    arch = replace(
        model.arch,
        num_attention_heads=4,
        hidden_size=32,
        adaln_out_features=6 * model.arch.num_modalities * 32,
        final_adaln_out_features=64,
    )
    result = zing_h3.ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
    torch.manual_seed(7)
    with torch.no_grad():
        for name, param in result.named_parameters():
            if "norm" in name and name.endswith("weight"):
                param.fill_(1)
            else:
                param.copy_(torch.randn(param.shape) * 0.08)
        result.rope.inv_freq.fill_(1)
    return result


def _joint_cache_snapshot(cache, phase, blocks, positions, layers):
    source = cache.pending_source
    active = cache.active
    metadata = (
        phase,
        cache.last_block,
        tuple((c.block, c.frames) for c in cache.committed),
        None if source is None else (source.block, source.frames),
        None if active is None else active[0],
    )
    tensors = [blocks.clone(), positions.clone()]
    if active is not None:
        tensors.append(active[1].clone())
    for chunk in cache._history():
        tensors.append(chunk.positions.clone())
        tensors.extend(value.clone() for pair in chunk.layers for value in pair)
    if layers is not None:
        tensors.extend(value.clone() for pair in layers for value in pair)
    return metadata, tensors


@pytest.mark.parametrize("mode", ["source_first", "cross_chunk"])
@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("source_hw", [(2, 2), (2, 4)])
@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("rope_mode", ["absolute", "block_relative"])
def test_joint_sp_matches_same_sp_sequential_cache_and_outputs(
    four_head_model, monkeypatch, mode, world, source_hw, pooled, rope_mode
):
    """SP4 must fuse real branches, including split boundaries and empty local heads."""
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.models.dits import zing_h3
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        allocate_layer_stores,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_sp_cpu import _CPUCollectives

    model = four_head_model
    packer, inputs, _, _ = request(model, audio=False)
    packer = type(packer)(model.arch, num_frame_per_block=[3, 2])
    inputs.visual_shape = (15, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 15, *source_hw)
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
    original_update = ZingH3KVCache.update
    original_forward = ZingH3Sampler._forward
    original_project = zing_h3.ZingH3DiTModel._project_outputs
    # Real SP ranks own independent models, including mutable derived buffers.
    rank_models = [
        zing_h3.ZingH3DiTModel(ZingH3DiTConfig(arch_config=model.arch), {}).eval()
        for _ in range(world)
    ]
    for rank_model in rank_models:
        rank_model.load_state_dict(model.state_dict())

    def update(cache, positions, blocks, phase, frames, layers=None, validation=None):
        original_update(cache, positions, blocks, phase, frames, layers, validation)
        assert cache.raw_store
        comm.thread.snapshots.append(
            _joint_cache_snapshot(cache, phase, blocks, positions, layers)
        )

    def forward(sampler, packed, cache, update):
        result = original_forward(sampler, packed, cache, update)
        if update in ("active", "joint"):
            rows = int((packed["block_idx"] == packed["block_idx"][-1]).sum())
            comm.thread.targets.append(result[0][-rows:].clone())
        if update == "joint":
            comm.thread.phases.append(packed["_zing_joint_phases"])
        return result

    def project(self, hidden, p, layout, *, joint_ends=None):
        if joint_ends is not None:
            comm.thread.layouts.append(
                (joint_ends, layout.row_start, hidden.shape[0], layout.sequence_splits)
            )
        return original_project(self, hidden, p, layout, joint_ends=joint_ends)

    monkeypatch.setattr(ZingH3KVCache, "update", update)
    monkeypatch.setattr(ZingH3Sampler, "_forward", forward)
    monkeypatch.setattr(zing_h3.ZingH3DiTModel, "_project_outputs", project)

    @torch.no_grad()
    def worker(rank, selected):
        comm.thread.rank = rank
        comm.thread.snapshots, comm.thread.targets = [], []
        comm.thread.phases, comm.thread.layouts = [], []
        sampler = ZingH3Sampler(
            rank_models[rank],
            packer,
            sigma_points=3,
            joint_forward=selected,
            window_chunks=3,
            sink_chunks=1,
            rope_position_mode=rope_mode,
        )
        storage = (
            allocate_layer_stores(
                128,
                num_layers=len(model.blocks),
                local_heads=model.arch.num_attention_heads // world,
                head_dim=model.arch.attention_head_dim,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )
            if pooled
            else None
        )
        state = sampler.initialize(inputs, visual.clone(), storage=storage)
        try:
            if selected == "cross_chunk":
                sampler._sample_cross_chunk(state)
            else:
                while state.index < len(state.spans):
                    sampler.step(state)
            assert state.cache.last_block == 12
            assert state.cache.active is state.cache.pending_source is None
            assert state.cache.stats["compactions"] >= 3
            if pooled:
                assert state.cache.stats["store_allocations"] == 0
                assert state.cache.stats["rotated_allocations"] == 0
            return (
                state.visual.clone(),
                list(comm.thread.snapshots),
                list(comm.thread.targets),
                dict(sampler.last_run_metrics),
                list(comm.thread.phases),
                list(comm.thread.layouts),
            )
        finally:
            state.close()
            assert state.cache.positions() is None
            assert state.cache.active is state.cache.pending_source is None

    runs = []
    for selected in ("off", mode):
        with ThreadPoolExecutor(max_workers=world) as executor:
            futures = [executor.submit(worker, rank, selected) for rank in range(world)]
            runs.append([future.result() for future in futures])
    for expected, actual in zip(*runs):
        torch.testing.assert_close(actual[0], expected[0], atol=0.008, rtol=0.008)
        assert len(actual[1]) == len(expected[1])
        for (am, at), (em, et) in zip(actual[1], expected[1]):
            assert am == em
            assert len(at) == len(et)
            for value, reference in zip(at, et):
                torch.testing.assert_close(value, reference, atol=0.008, rtol=0.008)
        assert len(actual[2]) == len(expected[2])
        for value, reference in zip(actual[2], expected[2]):
            torch.testing.assert_close(value, reference, atol=0.008, rtol=0.008)
        counts = actual[3]["forward_counts"]
        assert counts[mode] > 0
        assert actual[3]["dit_forwards"] < expected[3]["dit_forwards"]
        phases = (
            ("source", "active")
            if mode == "source_first"
            else ("final", "source", "active")
        )
        assert phases in actual[4]
        assert actual[4].count(phases) == counts[mode]
        assert "sp_size" not in actual[3]["joint_fallbacks"]
    if world == 4:
        layouts = [layout for result in runs[1] for layout in result[5]]
        assert any(
            start < stop < start + rows
            for ends, start, rows, _ in layouts
            for stop in ends[:-1]
        )
        assert any(
            stop <= start or begin >= start + rows
            for ends, start, rows, _ in layouts
            for begin, stop in zip((0, *ends[:-1]), ends)
        )
        assert any(len(set(splits)) > 1 for _, _, _, splits in layouts)


@pytest.mark.parametrize("mode", ["source_first", "cross_chunk"])
@pytest.mark.parametrize("world,reason", [(4, "different_rope_views"), (8, "sp_size")])
def test_joint_sp_rope_conflict_and_unsupported_size_fall_back_before_joint_work(
    four_head_model, monkeypatch, mode, world, reason
):
    from sglang.multimodal_gen.runtime.models.dits import zing_h3

    model = four_head_model
    packer, inputs, visual, _ = request(model, audio=False)
    # Only the admission/transaction seam needs SP here; eager work stays SP1.
    original_ctx = zing_h3.get_ulysses_ctx
    original_joint = ZingH3Sampler._joint_eligible
    from types import SimpleNamespace

    monkeypatch.setattr(
        zing_h3, "get_sp_group", lambda: SimpleNamespace(cpu_group=object())
    )
    monkeypatch.setattr(
        sampling.dist,
        "all_gather_object",
        _replicated_eligibility_votes,
    )

    def eligible(sampler, *args):
        monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (world, 0))
        try:
            return original_joint(sampler, *args)
        finally:
            monkeypatch.setattr(zing_h3, "get_ulysses_ctx", original_ctx)

    monkeypatch.setattr(ZingH3Sampler, "_joint_eligible", eligible)
    outputs = []
    for selected in ("off", mode):
        sampler = ZingH3Sampler(model, packer, sigma_points=3, joint_forward=selected)
        state = sampler.initialize(inputs, visual.clone())
        try:
            pos = state.positive
            if reason == "different_rope_views":
                pos["position_ids"][pos["block_idx"] == 1, 0] += 100
            sampler.step(state)
            outputs.append(state.visual.clone())
            if selected != "off":
                assert sampler.last_run_metrics["joint_fallbacks"].get(reason, 0) == 1
                assert sampler.last_run_metrics["forward_counts"]["source_first"] == 0
                assert sampler.last_run_metrics["forward_counts"]["cross_chunk"] == 0
                assert state.cache.last_block == 2
                assert state.cache.active is state.cache.pending_source is None
                if world == 4:
                    assert "sp_size" not in sampler.last_run_metrics["joint_fallbacks"]
        finally:
            state.close()
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)


def test_bounded_output_drain_resumes_joint_lookahead_without_recomputing(model):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance

    packer, inputs, _, _ = request(model, audio=False)
    packer = type(packer)(
        model.arch, num_frame_per_block=[3, 2], split_first_latent=True
    )
    inputs.visual_shape = (10, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 10, 2, 2)
    noise = torch.randn_like(inputs.source_visual)
    runs = []
    for limit in (None, 1):
        sampler = ZingH3Sampler(
            model,
            packer,
            joint_forward="cross_chunk",
            window_chunks=6,
            sink_chunks=3,
            solver="few_step",
            denoising_step_list=[1000, 500],
        )
        session = SourceReadySession(
            sampler,
            inputs,
            seed=51,
            max_pending_chunks=20,
            max_pending_bytes=1000000,
            total_latent_frames=10,
        )
        try:
            for chunk in packer.chunk_plan.chunks(10):
                session.append(
                    inputs.source_visual[:, :, chunk.start : chunk.end],
                    noise[:, :, chunk.start : chunk.end],
                    provenance=fixture_provenance(
                        chunk.chunk_id + 1, 20, chunk.start, chunk.end
                    ),
                )
            parts = []
            while session.pending:
                outputs = list(session.drain_ready(max_outputs=limit))
                if limit is not None:
                    assert len(outputs) == 1 and not session.draining
                parts.extend(result.visual for result in outputs)
                assert not session.closed
            runs.append(
                (
                    torch.cat(parts, dim=2),
                    session.rng.cpu_rng.clone(),
                    dict(session.metrics["forward_counts"]),
                )
            )
            assert session._drain_continuation is None
            assert session.memory_snapshot()["execution_storage_bytes"] == 0
        finally:
            session.dispose()
        assert all(value == 0 for value in session.memory_snapshot().values())
    torch.testing.assert_close(runs[0][0], runs[1][0], rtol=0, atol=0)
    assert torch.equal(runs[0][1], runs[1][1])
    assert runs[0][2] == runs[1][2] and runs[1][2]["cross_chunk"] > 0
