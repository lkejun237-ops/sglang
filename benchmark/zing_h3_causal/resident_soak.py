# SPDX-License-Identifier: Apache-2.0
"""Repeat complete real sessions in one resident process, without inserted waits."""

import math
import time


def replay_collection_decision(*, valid, exact, minimum_seconds, collect_mismatch):
    """Apply an already synchronized all-rank verdict without relaxing parity."""
    if not valid:
        raise RuntimeError("DiT replay produced invalid or incomplete output")
    if minimum_seconds and not exact and not collect_mismatch:
        raise RuntimeError(
            "Resident soak requires an exact complete-case warmup replay"
        )
    return dict(
        strict_replay_passed=exact,
        numerical_status="PASS" if exact else "FAIL",
        # Collection never substitutes for independent numerical, resource
        # and stall acceptance, even when the warmup is bitexact.
        full_acceptance_passed=False,
        measurement_scope="diagnostic_resident_soak"
        if minimum_seconds and not exact
        else "complete_case_replay",
    )


def all_ranks_reached_duration(work_seconds, minimum_seconds, *, device):
    import torch
    import torch.distributed as dist

    seconds = torch.tensor(work_seconds, dtype=torch.float64, device=device)
    dist.all_reduce(seconds, op=dist.ReduceOp.MIN)
    return seconds.item() >= minimum_seconds


def run_resident_round(
    run_session,
    minimum_seconds,
    *,
    clock=time.perf_counter,
    prepare_session=None,
    on_progress=None,
    all_ranks_complete=None,
):
    if not math.isfinite(minimum_seconds) or minimum_seconds <= 0:
        raise ValueError("Resident soak duration must be finite and positive")
    started = clock()
    sessions = []
    work_seconds = 0
    while True:
        # Each call must finish the original case, including normal cleanup.
        # A failed session aborts the round; do not retry or count partial work.
        if prepare_session is not None:
            prepare_session()
        entered = clock()
        session = run_session()
        returned = clock()
        session_seconds = returned - entered
        work_seconds += session_seconds
        sessions.append(
            dict(
                session,
                resident_session_seconds=session_seconds,
                resident_session_entered_ns=int(entered * 1e9),
                resident_session_returned_ns=int(returned * 1e9),
            )
        )
        progress = dict(
            seconds=work_seconds,
            minimum_seconds=minimum_seconds,
            completed_sessions=len(sessions),
            sessions=sessions,
            scope="Serial complete-case sessions in one resident process; not a longer single context",
            timing_scope="Session work including EOS/disposal; between-session coordination and receipt writes excluded",
        )
        if on_progress is not None:
            on_progress(progress)
        # Distributed callers must agree before any rank starts the next case.
        complete = (
            all_ranks_complete(work_seconds)
            if all_ranks_complete is not None
            else work_seconds >= minimum_seconds
        )
        if complete and work_seconds >= minimum_seconds:
            return dict(
                **progress,
                wall_seconds_including_coordination_and_receipts=clock() - started,
            )
