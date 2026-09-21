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
    options = dict(
        solver=solver,
        sigma_points=3,
        guidance_scale=guidance,
        window_chunks=None,
        sink_chunks=0,
        rope_position_mode="absolute",
    )
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
@pytest.mark.parametrize("window", [None, 21])
def test_32_ready_chunks_pause_finite_eos_and_cancel(model, window):
    packer, inputs, noise, _ = _long_request(model, [1, 2, 2], frames=53)
    sampler = ZingH3Sampler(
        model,
        packer,
        sigma_points=2,
        window_chunks=window,
        sink_chunks=3 if window else 0,
    )
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
        expected_chunks = (
            chunk.chunk_id + 1 if window is None else min(window, chunk.chunk_id + 1)
        )
        assert len(session.cache.committed) == expected_chunks
        memory = session.memory_snapshot()
        if window is None or chunk.chunk_id < window:
            assert memory["kv_storage_bytes"] > previous_kv
        assert memory["kv_temporal_chunks"] == expected_chunks
        assert memory["ready_bytes"] == 0
        previous_kv = memory["kv_storage_bytes"]
    assert session.committed_chunk_id == 32
    session.finish()
    assert session.complete and session.closed and session.cache.history(0) is None
    assert all(value == 0 for value in session.memory_snapshot().values())

    fresh = SourceReadySession(
        ZingH3Sampler(
            model,
            packer,
            sigma_points=2,
            window_chunks=window,
            sink_chunks=3 if window else 0,
        ),
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


@torch.no_grad()
@pytest.mark.parametrize("lengths", [5, [1, 2, 2]])
@pytest.mark.parametrize(
    "failure_phase", [None, "source", "active", "final", "after_commit"]
)
@pytest.mark.parametrize("rope_mode", ["absolute", "block_relative"])
def test_window_ready_matches_finite_or_closes_on_eviction_failure(
    model, monkeypatch, lengths, failure_phase, rope_mode
):
    # Both paths use exactly the same policy, never compare different contexts.
    frames = sum(
        ChunkPlan(lengths).chunk(i).end - ChunkPlan(lengths).chunk(i).start
        for i in range(32)
    )
    packer, inputs, noise, audio = _long_request(model, lengths, frames)
    options = dict(
        solver="few_step",
        denoising_step_list=[1000, 500],
        window_chunks=21,
        sink_chunks=3,
        rope_position_mode=rope_mode,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51)
        rng = (torch.get_rng_state().clone(), None)
        expected = ZingH3Sampler(model, packer, **options).sample(inputs, noise, audio)
    sampler = ZingH3Sampler(model, packer, **options)
    session = SourceReadySession(
        sampler,
        inputs,
        seed=51,
        max_pending_chunks=1,
        max_pending_bytes=1000000,
        initial_rng_state=rng,
    )
    outputs, sounds, audio_cursor = [], [], 0
    for chunk in packer.chunk_plan.chunks(frames):
        indices = packer._audio_indices_for_video_span(
            audio.shape[2], chunk.start, chunk.end, frames
        )
        session.append(
            inputs.source_visual[:, :, chunk.start : chunk.end],
            noise[:, :, chunk.start : chunk.end],
            audio=audio[:, :, indices],
            audio_start=audio_cursor,
            provenance=fixture_provenance(1, chunk.end, chunk.start, chunk.end),
            final=chunk.end == frames,
        )
        audio_cursor += len(indices)
        if chunk.chunk_id == 21 and failure_phase is not None:
            caller_rng = torch.get_rng_state().clone()
            original = sampler._forward

            def fail(*args, **kwargs):
                phase = args[2] if len(args) > 2 else kwargs.get("phase")
                if phase == failure_phase:
                    raise RuntimeError("window failure")
                value = original(*args, **kwargs)
                if phase == "final" and failure_phase == "after_commit":
                    raise RuntimeError("window failure")
                return value

            monkeypatch.setattr(sampler, "_forward", fail)
            with pytest.raises(RuntimeError, match="window failure"):
                session.step()
            assert session.closed and not session.complete
            assert all(value == 0 for value in session.memory_snapshot().values())
            assert session.rng.cpu_rng is session.rng.cuda_rng is None
            assert session.inputs is session.sampler is session.prompt is None
            assert torch.equal(caller_rng, torch.get_rng_state())
            with pytest.raises(ValueError, match="closed"):
                session.step()
            return
        result = session.step()
        outputs.append(result.visual)
        sounds.append(result.audio)
        i = chunk.chunk_id
        expected_ids = (
            []
            if session.complete
            else [*range(min(i + 1, 3)), *range(max(3, i - 17), i + 1)]
        )
        assert [c.temporal_chunk_id for c in session.cache.committed] == expected_ids
        assert session.pending_bytes == 0
    torch.testing.assert_close(torch.cat(outputs, 2), expected[0], rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(sounds, 2), expected[1], rtol=0, atol=0)
    assert session.closed and session.memory_snapshot()["kv_storage_bytes"] == 0
    assert session.rng.cpu_rng is None and not session.pending


def test_incremental_rope_grid_preserves_checkpoint_rounding(model):
    packer, _, _, _ = request(model)
    for offset in (0, 1, 20, 150, 10000):
        length = 7
        spans = torch.tensor(
            [
                packer.rope_frame_scale
                * packer.visual_token_frame_spans[
                    i % len(packer.visual_token_frame_spans)
                ]
                for i in range(offset + length)
            ],
            dtype=torch.float64,
        )
        legacy = torch.cat((torch.zeros(1, dtype=torch.float64), spans.cumsum(0)))
        torch.testing.assert_close(
            packer._video_t_grid(length, 3.5, offset),
            3.5 + legacy[offset : offset + length],
            rtol=0,
            atol=0,
        )


@torch.no_grad()
def test_ready_metadata_budget_is_independent_of_tensor_budget(model):
    packer, inputs, noise, _ = _long_request(model, 5, frames=5)
    sampler = ZingH3Sampler(
        model, packer, sigma_points=2, window_chunks=21, sink_chunks=3
    )
    session = SourceReadySession(
        sampler,
        inputs,
        seed=1,
        max_pending_chunks=1,
        max_pending_bytes=100000,
        max_pending_spans=1,
    )
    provenance = Provenance(
        PixelRange(0, 2), (SourceSpan(1, 0, 0, 1), SourceSpan(2, 1, 1, 1)), 2
    )
    with pytest.raises(ValueError, match="queue is full"):
        session.append(inputs.source_visual, noise, provenance=provenance)
    assert session.pending_spans == session.pending_bytes == session.next_chunk_id == 0
    session.append(
        inputs.source_visual, noise, provenance=fixture_provenance(1, 2, 0, 2)
    )
    assert session.pending_spans == 1
    session.dispose()
    assert session.pending_spans == 0


@torch.no_grad()
def test_prompt_change_after_eviction_preserves_sinks_and_absolute_rope(model):
    packer, inputs, noise, _ = _long_request(model, [1, 2, 2], frames=45)
    sampler = ZingH3Sampler(
        model, packer, sigma_points=2, window_chunks=21, sink_chunks=3
    )
    session = SourceReadySession(
        sampler, inputs, seed=9, max_pending_chunks=1, max_pending_bytes=100000
    )
    sinks = None
    origin = None
    for i, chunk in enumerate(packer.chunk_plan.chunks(45)):
        if i == 22:
            sinks = session.cache.committed[:3]
            origin = session.media_origin
            condition = {
                "context": torch.randn(5, 12).bfloat16(),
                "row_role": torch.tensor([0, 1, 0, 0, 0]),
            }
            session.update_prompt(condition, 17)
        session.append(
            inputs.source_visual[:, :, chunk.start : chunk.end],
            noise[:, :, chunk.start : chunk.end],
            provenance=fixture_provenance(1, chunk.end, chunk.start, chunk.end),
        )
        result = session.step()
        if i >= 22:
            assert result.prompt_event_id == 17
            assert session.media_origin == origin
            assert all(a is b for a, b in zip(sinks, session.cache.committed[:3]))
            # The temporal origin is anchored despite a different prefix length.
            assert (
                session.cache.committed[-1].positions[:, 0].min()
                == packer._video_t_grid(1, origin, chunk.start)[0]
            )
    session.dispose()  # Cancellation after repeated eviction releases history.
    assert all(value == 0 for value in session.memory_snapshot().values())


@torch.no_grad()
@pytest.mark.parametrize("commit_first", [False, True])
def test_window_cancelled_eos_is_not_successful_completion(model, commit_first):
    packer, inputs, noise, _ = _long_request(model, [1, 2, 2], frames=3)
    session = SourceReadySession(
        ZingH3Sampler(model, packer, sigma_points=2, window_chunks=21, sink_chunks=3),
        inputs,
        seed=1,
        max_pending_chunks=1,
        max_pending_bytes=100000,
    )
    session.append(
        inputs.source_visual[:, :, :1],
        noise[:, :, :1],
        provenance=fixture_provenance(1, 1),
        final=not commit_first,
    )
    if commit_first:
        session.step()
        session.append(
            inputs.source_visual[:, :, 1:3],
            noise[:, :, 1:3],
            provenance=fixture_provenance(1, 3, 1, 3),
            final=True,
        )
    assert session.ended and not session.complete
    session.dispose()
    assert session.closed and not session.complete
    assert all(value == 0 for value in session.memory_snapshot().values())
