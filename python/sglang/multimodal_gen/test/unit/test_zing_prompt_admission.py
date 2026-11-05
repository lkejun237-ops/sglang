# SPDX-License-Identifier: Apache-2.0
"""Condition validation must reach the leader before any rank commits."""

import json
import multiprocessing
from datetime import timedelta
from pathlib import Path
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


def _gloo_admission_rank(rank, rendezvous, output):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    group = SimpleNamespace(
        world_size=2, rank_in_group=rank, ranks=[0, 1], cpu_group=dist.group.WORLD
    )
    worker._sp_group = lambda: group
    decisions = []
    try:
        # Fail once on the leader and once on its peer; a valid request follows
        # each refusal on the same process group to expose collective drift.
        for failing_rank, error in [
            (0, ValueError("Invalid packed condition")),
            (1, ConditionError()),
        ]:

            def validate():
                if rank == failing_rank:
                    raise error
                return "candidate"

            try:
                coordinate_prompt_admission(validate)
            except ConditionError:
                decisions.append("rejected")
            else:
                raise AssertionError("A rank accepted a rejected peer candidate")
            decisions.append(coordinate_prompt_admission(lambda: "accepted"))
        Path(output, str(rank) + ".json").write_text(json.dumps(decisions))
    finally:
        dist.destroy_process_group()


def test_two_gloo_ranks_reject_together_then_accept_on_the_same_group(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(
            target=_gloo_admission_rank,
            args=(rank, str(tmp_path / "rendezvous"), str(tmp_path)),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)
            assert not process.is_alive(), "A rank remained blocked after admission"
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    for rank in range(2):
        assert json.loads((tmp_path / f"{rank}.json").read_text()) == [
            "rejected",
            "accepted",
            "rejected",
            "accepted",
        ]
