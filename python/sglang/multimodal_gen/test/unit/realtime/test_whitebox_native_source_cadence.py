# SPDX-License-Identifier: Apache-2.0
"""Fixed-cost three-stage regression, not a real GPU performance prediction.

This deliberately uses 9-frame synthetic work instead of modeling Zing kernels.
The API tests separately cover real 1/8/8 dependencies and 1/4/4 publication.
"""

import heapq
from collections import deque

import pytest
from test_whitebox_source_sampling import item

from sglang.multimodal_gen.runtime.realtime.whitebox.source_sampling import (
    AdaptiveSourceSampler,
)


def pipeline(*, adaptive, cost, duration=120, later_cost=None):
    sampler = AdaptiveSourceSampler()
    sampler.configure_native(
        {
            "source_demand_protocol": "source-demand-v1",
            "source_flow": {
                "protocol": "clip17-source-flow-v1",
                "input_groups": [1, 8, 8],
            },
        },
        session_id="pipeline",
        generation_id="one",
        capacity=25,
    )
    events, inputs, arrivals, selected = [], deque(), [], []
    stage_busy, queues = [False] * 3, [deque() for _ in range(3)]
    sent = received = consumed = cursor = serial = capture = sequence = 0
    latest = None
    waiting_since, waiting_total = 0.0, 0.0

    def schedule(t, kind, data=None):
        nonlocal serial
        serial += 1
        heapq.heappush(events, (t, serial, kind, data))

    def report(t):
        nonlocal sequence, waiting_since, waiting_total
        required = consumed + 9
        waiting = not stage_busy[0] and sent < required
        if waiting and waiting_since is None:
            waiting_since = t
        elif not waiting and waiting_since is not None:
            waiting_total += t - waiting_since
            waiting_since = None
        sequence += 1
        sampler.model_demand(
            dict(
                type="source.demand",
                protocol="source-demand-v1",
                session_id="pipeline",
                generation_id="one",
                sequence=sequence,
                accepted_source_frame_end=sent,
                dispatched_source_frame_end=consumed,
                required_source_frame_end=required,
                queued_source_frames=sent - consumed,
                waiting_for_source=waiting,
                input_wait_ms=(
                    waiting_total
                    + (t - waiting_since if waiting_since is not None else 0)
                )
                * 1000,
            ),
            sent=sent,
            now=t,
        )

    def start_stage(t, stage):
        if stage_busy[stage] or not queues[stage]:
            return
        stage_busy[stage] = True
        queues[stage].popleft()
        service = later_cost if later_cost is not None and t >= 45 else cost
        if stage == 2:
            for offset, count in [(0, 1), (0.04, 4), (0.08, 4)]:
                schedule(t + service + offset, "output", count)
        schedule(t + service + (0.08 if stage == 2 else 0), "stage", stage)

    def start_input(t):
        nonlocal consumed
        if stage_busy[0] or sent - consumed < 9:
            return
        consumed += 9
        queues[0].append(9)
        start_stage(t, 0)
        report(t)

    def pump(t):
        nonlocal cursor, sent
        while latest is not None and latest[0] > cursor and sent - received < 25:
            if adaptive:
                sampler.observe_queue(
                    t,
                    sent=sent,
                    buffered=sent - received,
                    reserve=9,
                    capacity=25,
                    oldest_age=t - inputs[0] if inputs else None,
                    period=sampler.period_us / 1e6 if sampler.period_us else None,
                    enabled=received > 0,
                )
            demand = dict(
                after=cursor,
                buffered=sent - received,
                reserve=9,
                startup=received == 0,
                urgent=sampler.needs_input(sent),
                control_sequence=0,
                semantic_control=True,
            )
            if adaptive and not sampler.admit(latest, **demand):
                break
            chosen = sampler.choose(latest, [], now=t, **demand) if adaptive else latest
            assert chosen[0] > cursor
            cursor = chosen[0]
            selected.append(cursor)
            sent += 1
            inputs.append(t)
            start_input(t)
            assert 0 <= sent - received <= 25 and consumed <= sent

    report(0)
    schedule(0, "capture")
    while events:
        t, _, kind, value = heapq.heappop(events)
        if t > duration:
            break
        if kind == "capture":
            capture += 1
            latest = item(
                capture, round(t * 1e6), observed={"latest": {"held_keys": ["KeyW"]}}
            )
            pump(t)
            schedule(capture / 36, "capture")
        elif kind == "output":
            before = sent - received
            received += value
            for _ in range(value):
                inputs.popleft()
                arrivals.append(t)
            if adaptive:
                sampler.completed(received, t, buffered=before, reserve=9)
            pump(t)
        else:
            stage_busy[value] = False
            if value < 2:
                queues[value + 1].append(9)
                start_stage(t, value + 1)
            start_stage(t, value)
            if value == 0:
                start_input(t)
                report(t)
                pump(t)
    assert len(selected) == len(set(selected))
    return dict(
        early=sum(10 <= t < 30 for t in arrivals) / 20,
        late=sum(90 <= t < 120 for t in arrivals) / 30,
        sampler=sampler.snapshot(),
    )


@pytest.mark.parametrize("cost", [0.15, 0.25, 0.35])
def test_fixed_pipeline_capacity_does_not_decay_with_native_supply_feedback(cost):
    baseline = pipeline(adaptive=False, cost=cost)
    candidate = pipeline(adaptive=True, cost=cost)
    assert candidate["late"] >= baseline["late"] * 0.95
    assert candidate["late"] >= candidate["early"] * 0.95


@pytest.mark.parametrize("before,after", [(0.25, 0.35), (0.35, 0.25)])
def test_native_feedback_recovers_after_real_pipeline_speed_changes(before, after):
    baseline = pipeline(adaptive=False, cost=before, later_cost=after)
    candidate = pipeline(adaptive=True, cost=before, later_cost=after)
    assert candidate["late"] >= baseline["late"] * 0.95
