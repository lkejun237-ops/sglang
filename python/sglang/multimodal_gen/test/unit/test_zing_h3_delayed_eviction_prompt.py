# SPDX-License-Identifier: Apache-2.0
"""Both realtime entries use the fixed-five rolling prompt contract (CPU CI)."""

from copy import copy

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
    SourceReadySession,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.realtime import (
    ZingH3BlockSession,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import (
    _long_request,
    fixture_provenance,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_kv_pool_cache import (
    _pointers,
    _storage,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
    model,
)


def _sampler(model, packer):
    return ZingH3Sampler(
        model,
        packer,
        solver="few_step",
        denoising_step_list=[1000, 500],
        window_chunks=6,
        sink_chunks=3,
    )


@torch.no_grad()
@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("pooled", [False, True])
def test_block_without_updates_matches_frozen_full_request(model, source, pooled):
    packer, inputs, noise, _ = _long_request(model, [3, 2], frames=13)
    if not source:
        inputs.source_visual = None
    initial = torch.get_rng_state().clone()
    sampler = _sampler(model, packer)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(initial)
        expected = sampler.sample(inputs, noise)[0]
    storage = _storage(model, 256) if pooled else None
    block = ZingH3BlockSession(
        _sampler(model, packer),
        inputs,
        noise.clone(),
        seed=7,
        initial_rng_state=(initial, None),
        acquire_storage=(lambda _: storage) if pooled else None,
    )
    try:
        while not block.ready.complete:
            result = block.step()
            assert result.provenance is None
            assert block.spans[block.index - 1][3] == result.chunk.end
        torch.testing.assert_close(block.visual, expected, rtol=0, atol=0)
    finally:
        block.close()


@torch.no_grad()
@pytest.mark.parametrize("pooled", [False, True])
def test_block_and_source_ready_share_latest_wins_and_repeated_rebuilds(model, pooled):
    packer, inputs, noise, _ = _long_request(model, [3, 2], frames=18)
    initial = torch.get_rng_state().clone()
    conditions = {
        event: {
            "context": torch.full((3, 12), event, dtype=torch.bfloat16),
            "row_role": torch.zeros(3, dtype=torch.long),
        }
        for event in (11, 12, 13)
    }

    def run(block_entry):
        sampler = _sampler(model, packer)
        storage = _storage(model, 256) if pooled else None
        pointers = _pointers(storage) if pooled else None
        switches = []
        rebuild = sampler.rebuild_prompt_prefix

        def observe(*args):
            before = torch.get_rng_state().clone()
            out = rebuild(*args)
            assert torch.equal(before, torch.get_rng_state())
            switches.append(args[1]["context"][0, 0].item())
            return out

        sampler.rebuild_prompt_prefix = observe
        if block_entry:
            owner = ZingH3BlockSession(
                sampler,
                inputs,
                noise.clone(),
                seed=7,
                initial_rng_state=(initial, None),
                acquire_storage=(lambda _: storage) if pooled else None,
                prompt_event_id=10,
                prompt_version=1,
            )
            ready = owner.ready
        else:
            ready = SourceReadySession(
                sampler,
                inputs,
                seed=7,
                max_pending_chunks=1,
                max_pending_bytes=1 << 20,
                total_latent_frames=18,
                initial_rng_state=(initial, None),
                acquire_storage=(lambda *_: storage) if pooled else None,
                initial_prompt_event_id=10,
                initial_prompt_version=1,
            )
        outputs, events, spans = [], [], []
        try:
            while not ready.complete:
                index = ready.committed_chunk_id
                if index in (1, 2, 4):
                    event = {1: 11, 2: 12, 4: 13}[index]
                    ready.update_prompt(conditions[event], event, event - 9)
                start = ready.next_latent
                if block_entry:
                    result = owner.step()
                    assert owner.index == len(outputs) + 1
                else:
                    end = start + ready.next_chunk_size
                    ready.append(
                        inputs.source_visual[:, :, start:end],
                        noise[:, :, start:end],
                        provenance=fixture_provenance(1, end, start, end),
                        final=end == 18,
                    )
                    result = ready.step()
                outputs.append(result.visual.clone())
                events.append((result.prompt_event_id, result.prompt_version))
                spans.append((result.chunk.start, result.chunk.end))
                if len(outputs) <= 3:
                    assert not switches and result.prompt_event_id == 10
                if pooled:
                    assert _pointers(storage) == pointers
            assert switches == [12, 13]
            assert events[3:6] == [(12, 3), (12, 3), (13, 4)]
            assert spans[0][0] == 0 and spans[-1][1] == 18
            assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
            return torch.cat(outputs, dim=2), events, spans
        finally:
            (owner.close() if block_entry else ready.dispose())

    block, source = run(True), run(False)
    assert torch.equal(block[0], source[0])
    assert block[1:] == source[1:]


@torch.no_grad()
@pytest.mark.parametrize("source", [False, True])
def test_block_rebuild_failure_closes_ready_cache(model, monkeypatch, source):
    packer, inputs, noise, _ = _long_request(model, [3, 2], frames=10)
    inputs = copy(inputs)
    if not source:
        inputs.source_visual = None
    sampler = _sampler(model, packer)
    storage = _storage(model, 256)
    block = ZingH3BlockSession(
        sampler,
        inputs,
        noise,
        seed=7,
        initial_rng_state=(torch.get_rng_state().clone(), None),
        acquire_storage=lambda _: storage,
    )
    for _ in range(3):
        block.step()
    block.update_prompt(inputs.positive, 1, 1)

    def fail(*args):
        raise RuntimeError("injected rebuild failure")

    monkeypatch.setattr(sampler, "rebuild_prompt_prefix", fail)
    try:
        with pytest.raises(RuntimeError, match="rebuild failure"):
            block.step()
        assert block.ready.closed and block.ready.cache.prefix is None
    finally:
        block.close()
    assert block.visual is block.source is None


@torch.no_grad()
@pytest.mark.parametrize("anonymous", [False, True])
def test_block_stage_reports_only_applied_prompt_and_drops_superseded_candidates(
    model, anonymous
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.realtime import (
        ZingH3RealtimeState,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )

    packer, inputs, _, _ = _long_request(model, [3, 2], frames=23)
    inputs.condition_media = []
    stage = object.__new__(ZingH3GenerationStage)
    stage.packer, stage.kv_pool = packer, None
    stage._use = lambda *_: nullcontext(model)
    calls = []

    def encode(batch, text, media):
        calls.append(text)
        if text == "bad":
            raise ConditionError("CONDITION_BUSY")
        return inputs.positive

    stage.encode_condition_for_batch = encode
    state = ZingH3RealtimeState()
    state.inputs, state.sampler = inputs, _sampler(model, packer)
    state.prompt = state.requested_prompt = "old"
    state.prompt_event, state.effective_prompt_version = (
        (None, None) if anonymous else (10, 1)
    )
    tag = lambda event: None if anonymous else event

    def step(text, event):
        batch = SimpleNamespace(
            sampling_params=SimpleNamespace(prompt=text, seed=7),
            realtime_event_id=tag(event),
            realtime_prompt_version=None if anonymous else event - 9,
        )
        rng = torch.get_rng_state().clone()
        end = stage._sample_realtime_block(batch, state)
        assert torch.equal(rng, torch.get_rng_state())
        return end, state.prompt_event

    try:
        assert step("old", 10) == (1, tag(10))
        assert step("new", 11) == (3, tag(10))
        # The same candidate is retagged without a second conditioning RPC.
        assert step("new", 12) == (5, tag(10))
        assert step("new", 12) == (8, tag(12))
        assert state.prompt == "new"
        assert calls == ["new"]
        # A newer failed event drops the old pending candidate.
        state.sampling.update_prompt(inputs.positive, 13, 4)
        state.requested_prompt = "pending"
        assert step("bad", 14)[1] == tag(12)
        assert state.sampling.ready.pending_prompt is None
        assert state.conditioning_notice == {
            "code": "CONDITION_BUSY",
            "event_id": tag(14),
        }
        assert step("new", 15)[1] == tag(15)
        assert calls == ["new", "bad"]
        step("old", 16)
        assert state.prompt == "new"
        step("old", 16)
        assert state.prompt == "old"
        assert calls == ["new", "bad", "old"]
    finally:
        state.dispose()
