# SPDX-License-Identifier: Apache-2.0
"""Condition validation must reach the leader before any rank commits."""

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.conditioning import worker
from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.prompt_update import (
    coordinate_prompt_admission,
)


@pytest.mark.parametrize("error", [ValueError("Invalid row_role"), ConditionError()])
def test_local_validation_failure_still_votes(monkeypatch, error):
    votes = []
    monkeypatch.setattr(
        worker,
        "coordinate_validation_decision",
        lambda valid: votes.append(valid) or False,
    )

    def validate():
        raise error

    with pytest.raises(ConditionError) as rejected:
        coordinate_prompt_admission(validate)
    assert votes == [False]
    assert rejected.value.__cause__ is error


def test_peer_rejection_does_not_publish_candidate(monkeypatch):
    candidate = object()
    votes = []
    monkeypatch.setattr(
        worker,
        "coordinate_validation_decision",
        lambda valid: votes.append(valid) or False,
    )
    with pytest.raises(ConditionError):
        coordinate_prompt_admission(lambda: candidate)
    assert votes == [True]


def test_accepted_candidate_is_returned_only_after_decision(monkeypatch):
    candidate = object()
    votes = []
    monkeypatch.setattr(
        worker,
        "coordinate_validation_decision",
        lambda valid: votes.append(valid) or True,
    )
    assert coordinate_prompt_admission(lambda: candidate) is candidate
    assert votes == [True]


def test_runtime_failure_is_not_reported_as_invalid_condition(monkeypatch):
    monkeypatch.setattr(
        worker,
        "coordinate_validation_decision",
        lambda _: pytest.fail("Runtime faults require fault cleanup"),
    )

    def validate():
        raise RuntimeError("CUDA failure")

    with pytest.raises(RuntimeError, match="CUDA failure"):
        coordinate_prompt_admission(validate)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("accepted", [False, True])
def test_leader_gathers_votes_and_broadcasts_one_decision(monkeypatch, size, accepted):
    # Nonzero global ranks catch accidental use of a local rank as dst/src.
    ranks = list(range(4, 4 + size))
    votes = [True] * size
    votes[-1] = accepted
    for rank in range(size):
        calls = []
        group = SimpleNamespace(
            world_size=size, rank_in_group=rank, ranks=ranks, cpu_group=object()
        )
        monkeypatch.setattr(worker, "_sp_group", lambda: group)

        def gather(valid, receipts, *, dst, group):
            calls.append("gather")
            assert dst == ranks[0]
            assert valid == votes[rank]
            if rank == 0:
                receipts[:] = votes
            else:
                assert receipts is None

        def broadcast(decision, *, src, group):
            calls.append("broadcast")
            assert src == ranks[0]
            assert decision == ([accepted] if rank == 0 else [None])
            decision[0] = accepted

        monkeypatch.setattr(worker.torch.distributed, "gather_object", gather)
        monkeypatch.setattr(
            worker.torch.distributed, "broadcast_object_list", broadcast
        )
        assert worker.coordinate_validation_decision(votes[rank]) is accepted
        assert calls == ["gather", "broadcast"]
