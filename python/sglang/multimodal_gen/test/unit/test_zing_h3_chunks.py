# SPDX-License-Identifier: Apache-2.0
"""CPU structure/lifecycle checks, not real-checkpoint quality validation."""

from copy import copy

import pytest
import torch

from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    PixelRange,
    SourceSpan,
    Provenance,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
    SourceReadySession,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (
    cpu_execution_options,  # noqa: F401
    model as model_fixture,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import request


@pytest.fixture
def model():
    return model_fixture.__wrapped__()


def fixture_provenance(source_id, dependency, start=0, end=1):
    return Provenance(
        PixelRange(start, end),
        (SourceSpan(source_id, start, start, end - start),),
        dependency,
    )


@pytest.mark.parametrize("lengths", [5, [1, 2, 2], [3, 2]])
@pytest.mark.parametrize("split", [False, True])
def test_plan_matches_native_cycle_and_tail(lengths, split):
    from itertools import cycle

    values = (lengths,) if isinstance(lengths, int) else lengths
    for total in range(1, 160):
        boundaries = [0]
        it = cycle(values)
        while boundaries[-1] < total:
            boundaries.append(min(total, boundaries[-1] + next(it)))
        if split and boundaries[1] > 1:
            boundaries.insert(1, 1)
        plan = ChunkPlan(lengths, split)
        chunks = list(plan.chunks(total))
        assert [(c.start, c.end) for c in chunks] == list(
            zip(boundaries, boundaries[1:])
        )
        assert plan.count(total) == len(chunks)
        assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
        assert [c.target_block for c in chunks] == list(
            range(2, len(chunks) * 2 + 1, 2)
        )
        assert all(c.source_block + 1 == c.target_block for c in chunks)
    assert ChunkPlan([1, 2, 2]).chunk(30).start == 50


@pytest.mark.parametrize(
    "value", [True, 0, -1, 1.5, "5", [], [1, False], [1, 0], [1, 2.0]]
)
def test_plan_rejects_invalid_metadata(value):
    with pytest.raises(ValueError):
        ChunkPlan(value)


def _long_request(model, lengths, frames=12):
    packer, inputs, _, _ = request(model)
    packer.num_frame_per_block = lengths
    inputs.visual_shape = (frames, 2, 2)
    inputs.source_visual = torch.randn(1, 2, frames, 2, 2)
    noise = torch.randn_like(inputs.source_visual)
    audio_frames = int(packer._video_t_span(frames)) + 3
    audio = torch.randn(1, 2, audio_frames, 2)
    return packer, inputs, noise, audio


@torch.no_grad()
@pytest.mark.parametrize("lengths", [5, [1, 2, 2]])
@pytest.mark.parametrize("solver", ["euler", "few_step"])
@pytest.mark.parametrize("guidance", [1, 2])
def test_ready_chunks_match_full_history_frozen_noise(model, lengths, solver, guidance):
    packer, inputs, noise, audio = _long_request(model, lengths)
    inputs.conditions = [{"content_id": "ref", "role": "reference"}]
    inputs.media = [{"id": "ref", "type": "video"}]
    inputs.visual = {"ref": torch.randn(1, 2, 2, 2, 2)}
    inputs.audio = {"ref": torch.randn(1, 2, 3, 2)}
    options = dict(solver=solver, sigma_points=3, guidance_scale=guidance)
    if solver == "few_step":
        options["denoising_step_list"] = [1000, 750, 500, 250]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        # Match actual pipeline order: noise draws precede re-noise sampling.
        noise = torch.randn_like(noise)
        audio = torch.randn_like(audio)
        initial_rng_state = (torch.get_rng_state().clone(), None)
        expected = ZingH3Sampler(model, packer, **options).sample(inputs, noise, audio)
    sampler = ZingH3Sampler(model, packer, **options)
    session = SourceReadySession(
        sampler,
        inputs,
        seed=31,
        max_pending_chunks=2,
        max_pending_bytes=100000,
        initial_rng_state=initial_rng_state,
    )
    outputs, sounds, audio_cursor = [], [], 0
    for chunk in packer.chunk_plan.chunks(noise.shape[2]):
        indices = packer._audio_indices_for_video_span(
            audio.shape[2], chunk.start, chunk.end, noise.shape[2]
        )
        provenance = fixture_provenance(
            100 + chunk.chunk_id, 200 + chunk.end, chunk.start, chunk.end
        )
        session.append(
            inputs.source_visual[:, :, chunk.start : chunk.end],
            noise[:, :, chunk.start : chunk.end],
            audio=audio[:, :, indices],
            audio_start=audio_cursor,
            provenance=provenance,
            final=chunk.end == noise.shape[2],
        )
        audio_cursor += len(indices)
        result = session.step()
        assert result.chunk == chunk and result.provenance == provenance
        assert session.committed_latent == chunk.end
        assert session.pending_bytes == 0
        if not session.complete:
            assert len(session.cache.committed) == chunk.chunk_id + 1
        outputs.append(result.visual)
        sounds.append(result.audio)
        if not session.complete:
            before = session.rng.cpu_rng.clone()
            assert session.step() is None
            assert torch.equal(session.rng.cpu_rng, before)
        torch.randn(47)  # An unrelated session cannot perturb this session RNG.
    assert session.complete
    torch.testing.assert_close(torch.cat(outputs, dim=2), expected[0], rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(sounds, dim=2), expected[1], rtol=0, atol=0)
    session.dispose()
    assert session.inputs is session.prompt is session.sampler is None
    assert session.cache.history(0) is None and not session.pending


@torch.no_grad()
@pytest.mark.parametrize("phase", ["source", "active", "final", "after_commit"])
def test_sampling_failure_closes_state_and_rejects_retry(model, monkeypatch, phase):
    packer, inputs, noise, audio = _long_request(model, [1, 2, 2], frames=6)
    sampler = ZingH3Sampler(
        model,
        packer,
        solver="few_step",
        denoising_step_list=[1000, 500],
        guidance_scale=2,
    )
    state = sampler.initialize(inputs, noise, audio)
    sampler.step(state)
    changed = copy(inputs)
    changed.positive = dict(
        context=torch.randn(7, 12).bfloat16(), row_role=torch.zeros(7, dtype=torch.long)
    )
    original = sampler._forward

    def fail(packed, cache, update):
        result = original(packed, cache, update)
        if update == phase:
            raise RuntimeError("Injected post-forward failure")
        return result

    monkeypatch.setattr(sampler, "_forward", fail)
    original_block = sampler._sample_block
    if phase == "after_commit":

        def fail_after(*args):
            original_block(*args)
            raise RuntimeError("Injected post-commit failure")

        monkeypatch.setattr(sampler, "_sample_block", fail_after)
    with pytest.raises(RuntimeError, match="Injected"):
        sampler.step(state, inputs=changed)
    assert state.closed
    assert state.visual is state.audio is state.positive is state.negative is None
    assert state.cache.history(0) is None
    assert state.cache.active is state.cache.pending_source is None
    assert state.cache.layer_layouts is None
    with pytest.raises(ValueError, match="closed"):
        sampler.step(state)
    state.close()  # Cleanup remains idempotent.


@torch.no_grad()
def test_ready_queue_ownership_budget_prompt_eos(model):
    packer, inputs, noise, _ = _long_request(model, [1, 2, 2], frames=4)
    sampler = ZingH3Sampler(
        model, packer, solver="few_step", denoising_step_list=[1000, 500]
    )
    session = SourceReadySession(
        sampler, inputs, seed=7, max_pending_chunks=1, max_pending_bytes=128
    )
    source = inputs.source_visual
    p = fixture_provenance(91, 20)
    session.append(source[:, :, :1], noise[:, :, :1], provenance=p)
    queued = session.pending[0]
    assert (
        queued.source.untyped_storage().nbytes()
        == queued.source.numel() * queued.source.element_size()
    )
    source[:, :, :1].fill_(100)
    assert not torch.equal(source[:, :, :1], queued.source)
    second_provenance = fixture_provenance(92, 19, 1, 3)
    with pytest.raises(ValueError, match="full"):
        session.append(
            source[:, :, 1:3], noise[:, :, 1:3], provenance=second_provenance
        )
    assert session.next_chunk_id == 1 and len(session.pending) == 1
    session.step()
    session.append(source[:, :, 1:3], noise[:, :, 1:3], provenance=second_provenance)
    context = dict(
        context=torch.randn(5, 12).bfloat16(), row_role=torch.zeros(5, dtype=torch.long)
    )
    session.update_prompt(context, 42)
    frozen = session.prompt[0]["context"].clone()
    context["context"].zero_()
    result = session.step()
    assert result.prompt_event_id == 42
    assert result.provenance.pixels == second_provenance.pixels
    assert (
        result.provenance.source_dependency_end == 20
    )  # Prior full-history dependency.
    assert torch.equal(session.applied_prompt[0]["context"], frozen)
    assert result.chunk.chunk_id == 1 and result.chunk.start == 1
    session.update_prompt(context, 99)
    session.append(
        source[:, :, 3:4],
        noise[:, :, 3:4],
        provenance=fixture_provenance(93, 21, 3, 4),
        final=True,
    )
    final = session.step()
    assert final.chunk.end == 4 and final.prompt_event_id == 99
    assert session.closed and session.cache.history(0) is None
    assert session.complete
    with pytest.raises(ValueError):
        session.append(source[:, :, :1], noise[:, :, :1], provenance=p)
    session.dispose()
    session.dispose()
    with pytest.raises(ValueError, match="closed"):
        session.step()


@torch.no_grad()
def test_32_ready_chunks_pause_finite_eos_and_cancel(model):
    packer, inputs, noise, _ = _long_request(model, [1, 2, 2], frames=53)
    sampler = ZingH3Sampler(model, packer, sigma_points=2)
    session = SourceReadySession(
        sampler,
        inputs,
        seed=1,
        max_pending_chunks=1,
        max_pending_bytes=128,
        total_latent_frames=53,
    )
    before = torch.get_rng_state().clone()
    assert session.step() is None and session.rng.cpu_rng is None
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match="before"):
        session.finish()
    previous_kv = 0
    for chunk in packer.chunk_plan.chunks(53):
        session.append(
            inputs.source_visual[:, :, chunk.start : chunk.end],
            noise[:, :, chunk.start : chunk.end],
            provenance=fixture_provenance(7, 60, chunk.start, chunk.end),
        )
        output = session.step()
        assert output.chunk == chunk
        assert session.committed_pixel == chunk.end
        assert session.pending_bytes == 0 and not session.pending
        assert (
            len(session.cache.committed) == chunk.chunk_id + 1
        )  # Explicit full history, no eviction.
        memory = session.memory_snapshot()
        assert memory["kv_storage_bytes"] > previous_kv
        assert memory["kv_temporal_chunks"] == chunk.chunk_id + 1
        assert memory["ready_bytes"] == 0
        previous_kv = memory["kv_storage_bytes"]
    assert session.committed_chunk_id == 32
    session.finish()
    assert session.complete and session.closed and session.cache.history(0) is None
    assert all(value == 0 for value in session.memory_snapshot().values())

    fresh = SourceReadySession(
        ZingH3Sampler(model, packer, sigma_points=2),
        inputs,
        seed=1,
        max_pending_chunks=1,
        max_pending_bytes=128,
    )
    fresh.append(
        inputs.source_visual[:, :, :1],
        noise[:, :, :1],
        provenance=fixture_provenance(1, 1),
    )
    fresh.dispose()  # Cancellation drops pending ownership, not only KV.
    assert fresh.pending_bytes == 0 and not fresh.pending and fresh.prompt is None


@torch.no_grad()
def test_ready_queue_rejects_byte_overflow_and_invalid_tail_without_mutation(model):
    packer, inputs, noise, _ = _long_request(model, 5, frames=6)
    session = SourceReadySession(
        ZingH3Sampler(model, packer, sigma_points=2),
        inputs,
        seed=1,
        max_pending_chunks=2,
        max_pending_bytes=64,
        total_latent_frames=6,
    )
    with pytest.raises(ValueError, match="next checkpoint chunk"):
        session.append(
            inputs.source_visual[:, :, :1],
            noise[:, :, :1],
            provenance=fixture_provenance(1, 6),
        )
    with pytest.raises(ValueError, match="finite latent target"):
        session.append(
            inputs.source_visual[:, :, :1],
            noise[:, :, :1],
            provenance=fixture_provenance(1, 6),
            final=True,
        )
    with pytest.raises(ValueError, match="full"):
        session.append(
            inputs.source_visual[:, :, :5],
            noise[:, :, :5],
            provenance=fixture_provenance(1, 6, 0, 5),
        )
    assert (
        session.next_latent
        == session.next_pixel
        == session.next_chunk_id
        == session.pending_bytes
        == 0
    )
    assert not session.ended and not session.pending
    session.dispose()


def test_unknown_duration_cannot_be_enumerated_as_a_finite_plan():
    with pytest.raises(ValueError, match="finite"):
        ChunkPlan([1, 2, 2]).count(None)


@torch.no_grad()
def test_interleaved_sessions_preserve_post_noise_rng(model):
    sessions, expected, outputs = [], [], [[], []]
    for seed in (43, 97):
        packer, inputs, noise, _ = _long_request(model, [1, 2, 2], frames=6)
        options = dict(solver="few_step", denoising_step_list=[1000, 750, 500, 250])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            noise = torch.randn_like(noise)
            snapshot = (torch.get_rng_state().clone(), None)
            expected.append(
                ZingH3Sampler(model, packer, **options).sample(inputs, noise)[0]
            )
        sampler = ZingH3Sampler(model, packer, **options)
        session = SourceReadySession(
            sampler,
            inputs,
            seed=seed,
            max_pending_chunks=4,
            max_pending_bytes=1024,
            initial_rng_state=snapshot,
        )
        # The session owns a copy; subsequent fixture mutation is harmless.
        snapshot[0].zero_()
        for chunk in packer.chunk_plan.chunks(6):
            session.append(
                inputs.source_visual[:, :, chunk.start : chunk.end],
                noise[:, :, chunk.start : chunk.end],
                provenance=fixture_provenance(seed, 30, chunk.start, chunk.end),
                final=chunk.end == 6,
            )
        sessions.append(session)
    caller_rng = torch.get_rng_state().clone()
    outputs[1].append(sessions[1].step().visual)
    outputs[0].append(sessions[0].step().visual)
    for _ in range(3):
        for index in (1, 0):
            outputs[index].append(sessions[index].step().visual)
    assert torch.equal(caller_rng, torch.get_rng_state())
    for index, session in enumerate(sessions):
        assert session.complete and session.closed
        torch.testing.assert_close(
            torch.cat(outputs[index], dim=2), expected[index], rtol=0, atol=0
        )
        assert all(value == 0 for value in session.memory_snapshot().values())


@torch.no_grad()
@pytest.mark.parametrize(
    "phase", ["pack", "prefix", "source", "active", "final", "after_commit"]
)
def test_ready_session_failure_releases_all_state_and_preserves_caller_rng(
    model, monkeypatch, phase
):
    packer, inputs, noise, _ = _long_request(model, [1, 2, 2], frames=4)
    sampler = ZingH3Sampler(
        model, packer, solver="few_step", denoising_step_list=[1000, 500]
    )
    session = SourceReadySession(
        sampler, inputs, seed=7, max_pending_chunks=2, max_pending_bytes=1024
    )
    session.append(
        inputs.source_visual[:, :, :1],
        noise[:, :, :1],
        provenance=fixture_provenance(91, 20),
    )
    session.step()
    session.append(
        inputs.source_visual[:, :, 1:3],
        noise[:, :, 1:3],
        provenance=fixture_provenance(92, 20, 1, 3),
    )
    session.append(
        inputs.source_visual[:, :, 3:4],
        noise[:, :, 3:4],
        provenance=fixture_provenance(93, 21, 3, 4),
        final=True,
    )
    session.update_prompt(inputs.positive, 42)
    original = sampler._forward

    def fail_forward(packed, cache, update):
        result = original(packed, cache, update)
        if update == phase:
            raise RuntimeError("Injected session failure")
        return result

    monkeypatch.setattr(sampler, "_forward", fail_forward)
    original_block = sampler._sample_block

    def fail_block(*args):
        original_block(*args)
        raise RuntimeError("Injected session failure")

    if phase == "after_commit":
        monkeypatch.setattr(sampler, "_sample_block", fail_block)
    if phase == "pack":

        def fail_pack(*args, **kwargs):
            raise RuntimeError("Injected session failure")

        monkeypatch.setattr(packer, "pack_chunk", fail_pack)
    caller_rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="Injected"):
        session.step()
    assert session.closed and not session.complete
    assert all(value == 0 for value in session.memory_snapshot().values())
    assert session.rng.cpu_rng is session.rng.cuda_rng is None
    assert session.inputs is session.sampler is session.prompt is None
    assert torch.equal(caller_rng, torch.get_rng_state())
    with pytest.raises(ValueError, match="closed"):
        session.step()
    with pytest.raises(ValueError, match="closed"):
        session.update_prompt(inputs.positive, 43)
    with pytest.raises(ValueError, match="no longer"):
        session.append(
            inputs.source_visual[:, :, :1],
            noise[:, :, :1],
            provenance=fixture_provenance(94, 20),
        )
    session.dispose()
