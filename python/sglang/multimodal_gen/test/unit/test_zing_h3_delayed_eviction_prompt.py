# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401, F811
"""Ordinary prompt replacement must refuse before model work on every rank."""

from concurrent.futures import ThreadPoolExecutor
from copy import copy
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    sampling,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import allocate_layer_stores
from sglang.multimodal_gen.test.unit.test_zing_h3_joint import four_head_model
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (
    cpu_execution_options,
    model,
    request,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_sp_cpu import _CPUCollectives


@torch.no_grad()
def test_ordinary_prompt_donor_rejected_before_model(model, monkeypatch):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = ZingH3Sampler(
        model, packer, sigma_points=2, window_eviction_timing="after_step0"
    )
    state = sampler.initialize(inputs, visual)
    sampler.step(state)
    sampler.prefix_workspace = allocate_layer_stores(
        1,
        num_layers=len(model.blocks),
        local_heads=model.arch.num_attention_heads,
        head_dim=model.arch.attention_head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )
    forwards = []
    original = model.forward

    def forward(*args, **kwargs):
        forwards.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", forward)
    with pytest.raises((ValueError, RuntimeError), match="(?i)(capacity|rank|pooled)"):
        sampler.step(state, inputs=inputs)
    assert state.closed
    assert not forwards, "undersized ordinary prompt donor entered model before refusal"


@torch.no_grad()
def test_ordinary_prompt_layout_is_voted(model, monkeypatch):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = ZingH3Sampler(
        model, packer, sigma_points=2, window_eviction_timing="after_step0"
    )
    state = sampler.initialize(inputs, visual)
    sampler.step(state)
    votes = []
    original = sampler._agree_sp_boundary

    def agree(*args, **kwargs):
        votes.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(sampler, "_agree_sp_boundary", agree)
    inputs.positive = {
        "context": torch.randn(7, 12).bfloat16(),
        "row_role": torch.zeros(7, dtype=torch.long),
    }
    try:
        sampler.update_condition(state, inputs)
        assert votes, "ordinary replacement prefix runs with no boundary vote"
    finally:
        state.close()


@pytest.mark.parametrize("entry", ["step", "update_condition"])
@pytest.mark.parametrize("fault", ["donor", "layout", "session_peak"])
def test_sp4_ordinary_prompt_rejects_before_model(
    four_head_model, monkeypatch, entry, fault
):
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.models.dits import zing_h3

    comm = _CPUCollectives(4)
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (4, comm.thread.rank))
    monkeypatch.setattr(
        zing_h3,
        "get_sp_group",
        lambda: SimpleNamespace(cpu_group=object(), ulysses_group=None),
    )
    monkeypatch.setattr(zing_h3, "gather_sequence_varlen", comm.gather)
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: 4)
    monkeypatch.setattr(usp, "get_ulysses_parallel_rank", lambda: comm.thread.rank)
    monkeypatch.setattr(usp, "_ipc_ready_group", lambda: None)
    monkeypatch.setattr(usp, "_usp_all_to_all_single_varlen", comm.all_to_all)

    def gather_objects(output, value, group=None):
        output[:] = comm.exchange(value, lambda values, rank: list(values))

    monkeypatch.setattr(sampling.dist, "all_gather_object", gather_objects)
    rank_models = [
        zing_h3.ZingH3DiTModel(
            ZingH3DiTConfig(arch_config=four_head_model.arch), {}
        ).eval()
        for _ in range(4)
    ]
    for rank_model in rank_models:
        rank_model.load_state_dict(four_head_model.state_dict())
    packer, inputs, visual, _ = request(four_head_model, audio=False)
    observed = []
    original_forward = ZingH3Sampler._forward

    def forward(*args, **kwargs):
        if getattr(comm.thread, "reject_phase", False):
            observed.append(comm.thread.rank)
            # All ranks stop at the observable model boundary so a faulty
            # implementation cannot strand peers in a real model collective.
            raise ValueError("capacity probe reached model before refusal")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(ZingH3Sampler, "_forward", forward)
    states = []

    def stores(rows):
        return allocate_layer_stores(
            rows,
            num_layers=len(four_head_model.blocks),
            local_heads=1,
            head_dim=8,
            dtype=torch.bfloat16,
            device="cpu",
        )

    def worker(rank):
        comm.thread.rank = rank
        with torch.no_grad():
            sampler = ZingH3Sampler(
                rank_models[rank],
                packer,
                sigma_points=2,
                window_eviction_timing="after_step0",
            )
            state = sampler.initialize(inputs, visual.clone(), storage=stores(128))
            states.append(state)
            sampler.step(state)
            changed = copy(inputs)
            prompt_rows = 9 if fault == "layout" and rank == 1 else 7
            changed.positive = {
                "context": torch.ones(prompt_rows, 12).bfloat16(),
                "row_role": torch.zeros(prompt_rows, dtype=torch.long),
            }
            layout = packer.pack_inference_layout(
                changed, changed.positive, state.visual, None
            )
            prefix_rows = int((layout["block_idx"] == 0).sum())
            old_rows = state.cache.prefix.positions.shape[0]
            sampler.prefix_workspace = stores(
                prefix_rows - 1 if fault == "donor" and rank == 1 else 128
            )
            if fault == "session_peak":
                # Replacement alone fits, but adding current source/target does
                # not. The original old-prefix boundary still fits this slot.
                source, target, _, _ = state.spans[state.index]
                scratch = int(
                    (
                        (layout["block_idx"] == source)
                        | (layout["block_idx"] == target)
                    ).sum()
                )
                capacity = max(
                    state.cache._length + prefix_rows - old_rows,
                    state.cache._length + scratch,
                )
                assert state.cache._length + prefix_rows - old_rows <= capacity
                assert state.cache._length + prefix_rows - old_rows + scratch > capacity
                state.cache.storage = stores(capacity if rank == 1 else 128)
            comm.thread.reject_phase = True
            try:
                with pytest.raises(ValueError, match="(?i)(capacity|rank|mismatch)"):
                    if entry == "step":
                        sampler.step(state, inputs=changed)
                    else:
                        sampler.update_condition(state, changed)
                assert state.closed, (
                    "public prompt update failures must close sampling state"
                )
                assert state.cache.positions() is None
                assert state.cache.active is state.cache.pending_source is None
                assert state.cache.eviction_target_block is None
                assert state.cache._stores == {}
            finally:
                state.close()
            return True

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, rank) for rank in range(4)]
            assert all(f.result(timeout=30) for f in futures)
        assert observed == [], (
            "ordinary prompt rejection must precede every rank model forward"
        )
    finally:
        for state in states:
            state.close()


def test_sp4_ordinary_prompt_prepare_failure_is_agreed(four_head_model, monkeypatch):
    """A failed pack must join the same status vote as successful peers."""
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.models.dits import zing_h3

    comm = _CPUCollectives(4)
    status_votes = []
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (4, comm.thread.rank))
    monkeypatch.setattr(
        zing_h3,
        "get_sp_group",
        lambda: SimpleNamespace(cpu_group=object(), ulysses_group=None),
    )
    monkeypatch.setattr(zing_h3, "gather_sequence_varlen", comm.gather)
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: 4)
    monkeypatch.setattr(usp, "get_ulysses_parallel_rank", lambda: comm.thread.rank)
    monkeypatch.setattr(usp, "_ipc_ready_group", lambda: None)
    monkeypatch.setattr(usp, "_usp_all_to_all_single_varlen", comm.all_to_all)

    def gather_objects(output, value, group=None):
        if getattr(comm.thread, "reject_phase", False):
            status_votes.append((comm.thread.rank, value))
        output[:] = comm.exchange(value, lambda values, rank: list(values))

    monkeypatch.setattr(sampling.dist, "all_gather_object", gather_objects)
    rank_models = [
        zing_h3.ZingH3DiTModel(
            ZingH3DiTConfig(arch_config=four_head_model.arch), {}
        ).eval()
        for _ in range(4)
    ]
    for rank_model in rank_models:
        rank_model.load_state_dict(four_head_model.state_dict())
    packer, inputs, visual, _ = request(four_head_model, audio=False)
    observed = []
    original_forward = ZingH3Sampler._forward

    def forward(*args, **kwargs):
        if getattr(comm.thread, "reject_phase", False):
            observed.append(comm.thread.rank)
            # All ranks stop at the observable model boundary so a faulty
            # implementation cannot strand peers in a real model collective.
            raise ValueError("capacity probe reached model before refusal")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(ZingH3Sampler, "_forward", forward)
    states = []

    def stores(rows):
        return allocate_layer_stores(
            rows,
            num_layers=len(four_head_model.blocks),
            local_heads=1,
            head_dim=8,
            dtype=torch.bfloat16,
            device="cpu",
        )

    def worker(rank):
        comm.thread.rank = rank
        with torch.no_grad():
            sampler = ZingH3Sampler(
                rank_models[rank],
                packer,
                sigma_points=2,
                window_eviction_timing="after_step0",
            )
            state = sampler.initialize(inputs, visual.clone(), storage=stores(128))
            states.append(state)
            sampler.step(state)
            changed = copy(inputs)
            changed.positive = {
                "context": torch.ones(7, 12).bfloat16(),
                "row_role": torch.zeros(7, dtype=torch.long),
            }
            if rank == 1:
                # Actual production pack validation rejects a target role in
                # context; no model or packing behavior is mocked.
                changed.positive["row_role"][0] = 4
            comm.thread.reject_phase = True
            try:
                sampler.update_condition(state, changed)
            except BaseException as error:
                return error, state.closed
            finally:
                state.close()
            return None, state.closed

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, rank) for rank in range(4)]
        results = [f.result(timeout=30) for f in futures]
    assert all(isinstance(error, ValueError) and closed for error, closed in results), (
        "A local prompt prepare failure must produce coordinated ValueError on all ranks, without a broken collective",
        results,
    )
    assert sorted(rank for rank, _ in status_votes) == list(range(4)), (
        "all ranks must join one prepare-status vote"
    )
    assert any(rank == 1 and vote["precheck_error"] for rank, vote in status_votes)
    assert not observed, "prepare failure must refuse before donor/model execution"
    for state in states:
        assert state.cache.positions() is None
        assert state.cache.active is state.cache.pending_source is None
        assert state.cache.eviction_target_block is None
        assert state.cache._stores == {}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
