"""Deterministic capture/consumer schedules; no GPU or deployment is involved."""

import asyncio
import heapq
import math
from collections import deque
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import EngineFrame
from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import LatestFrame
from sglang.multimodal_gen.runtime.realtime.whitebox.render_assets import SourceFrames
from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import SegmentSource
from sglang.multimodal_gen.runtime.realtime.whitebox.source_sampling import (
    AdaptiveSourceSampler,
    wait_for_capture_or_progress,
)


def item(version, pts, *, effective=None, observed=None):
    return version, (
        EngineFrame(
            version,
            version * 1_000_000 // 24,
            bytes([version % 256]) * 3,
            {"ui_tick": version},
            effective_input=effective,
            observed_input=observed,
            provenance={"capture_pts_us": pts},
        ),
        pts / 1_000_000,
    )


def choose(sampler, frames, *, after=None, **kwargs):
    latest = frames[-1]
    params = dict(
        after=sampler.last_version if after is None else after,
        now=latest[1][1],
        buffered=25,
        reserve=9,
        control_sequence=-1,
    )
    params.update(kwargs)
    return sampler.choose(
        latest, [(v, f, t, 3) for v, (f, t) in reversed(frames)], **params
    )


def trained(fps=24):
    sampler = AdaptiveSourceSampler()
    sampler.completed(0, 0, buffered=25, reserve=9)
    sampler.completed(fps, 1, buffered=25, reserve=9)
    assert sampler.period_us == pytest.approx(1_000_000 / fps)
    choose(sampler, [item(1, 0)], after=0, startup=True)
    return sampler


def test_selects_existing_capture_near_target_without_rewriting_identity():
    s = trained()
    frames = [item(2, 33333), item(3, 50000), item(4, 66667)]
    selected = choose(s, frames)
    assert selected is not frames[-1]
    assert selected == frames[1]  # 50ms is closer to 41.667ms than 66.667ms.
    assert selected[1][0].rgb == frames[1][1][0].rgb
    assert selected[1][0].states == {"ui_tick": 3}
    assert selected[1][0].provenance == {"capture_pts_us": 50000}
    assert s.snapshot()["selection_counts"]["uniform"] == 1


@pytest.mark.parametrize("kwargs", [dict(startup=True), dict(urgent=True)])
def test_supply_startup_and_prompt_take_priority_over_even_spacing(kwargs):
    s = trained()
    frames = [item(2, 41667), item(3, 60000)]
    assert choose(s, frames, **kwargs) is frames[-1]


def test_low_water_supplies_every_usable_capture_in_order_instead_of_skipping_to_latest():
    s = trained()
    frames = [item(2, 16667), item(3, 33333), item(4, 50000)]
    assert [choose(s, frames, buffered=i)[0] for i in range(3)] == [2, 3, 4]
    assert s.counts["low_water"] == 3
    # An interaction overrides backlog filling and chooses the freshest view.
    assert (
        choose(s, [item(5, 66667), item(6, 83333)], buffered=0, control_sequence=1)[0]
        == 6
    )


@pytest.mark.parametrize("fps", [60, 48, 40, 36, 30, 24, 20, 8])
def test_one_available_capture_is_always_supplied_at_all_source_rates(fps):
    s = trained()
    selected = []
    for version in range(2, 122):
        frame = item(version, round((version - 1) * 1_000_000 / fps))
        # The consumer is about to run out of input; there is never a timer here.
        assert choose(s, [frame], buffered=0) is frame
        selected.append(s.last_version)
    assert selected == list(range(2, 122))


def test_batched_completions_learn_capacity_but_ignore_duplicates_and_source_starvation():
    s = AdaptiveSourceSampler()
    s.completed(0, 0, buffered=25, reserve=9)
    s.completed(8, 0.25, buffered=17, reserve=9)
    s.completed(8, 0.6, buffered=17, reserve=9)
    assert s.period_us is None
    s.completed(24, 0.75, buffered=17, reserve=9)
    assert s.period_us == 31250  # 32 fps, not clamped to protocol FPS 24.
    s.supply(0, 9)
    s.completed(28, 1.25, buffered=0, reserve=9)
    s.completed(32, 1.75, buffered=0, reserve=9)
    assert s.period_us == 31250  # Never reinterpret the slow CLI as slow GPU.
    s.completed(40, 2.0, buffered=25, reserve=9)
    s.completed(56, 2.5, buffered=25, reserve=9)
    assert s.period_us == 31250


def test_batch_completion_refill_dip_does_not_disable_the_estimator():
    s = AdaptiveSourceSampler()
    s.completed(17, 0, buffered=34, reserve=18)
    # The returned batch briefly leaves 17 outstanding, but real captures are
    # already available to refill the window. This is not an input-starved GPU.
    s.supply(17, 18, available=True)
    s.completed(34, 17 / 24, buffered=34, reserve=18)
    assert s.period_us == pytest.approx(1_000_000 / 24)
    s.supply(0, 18, available=False)
    s.completed(51, 3, buffered=34, reserve=18)
    assert s.period_us == pytest.approx(1_000_000 / 24)
    assert s.counts["rate_observations"] == 1


def test_input_changes_and_unverified_interaction_keep_the_freshest_capture():
    s = trained()
    frames = [item(2, 41667), item(3, 60000)]
    assert choose(s, frames, control_sequence=1) is frames[-1]
    frames = [item(4, 100000), item(5, 125000, effective={"keys": ["W"]})]
    assert choose(s, frames, control_sequence=1) is frames[-1]
    for version in range(6, 12):
        latest = item(version, version * 41667, observed={"latest": {"keys": ["W"]}})
        assert choose(s, [frames[-1], latest], control_sequence=1) is latest
        frames.append(latest)


def test_stale_candidates_resume_floor_and_capture_clock_jumps_do_not_replay_history():
    s = trained()
    frames = [item(2, 41667), item(3, 70000)]
    assert choose(s, frames, now=0.2) is frames[-1]
    assert choose(s, [item(4, 110000), item(8, 140000)], after=7)[0] == 8
    assert s.period_us is None
    assert choose(s, [item(9, 3_000_000)])[0] == 9
    assert choose(s, [item(10, 1)])[0] == 10
    assert s.snapshot()["resets"] >= 3


def test_current_input_receipt_wins_over_an_older_evenly_spaced_capture():
    s = trained()
    a = item(2, 41667, observed={"receipts": [{"sequence": 1}]})
    b = item(3, 60000, observed={"receipts": [{"sequence": 2}]})
    assert choose(s, [a, b]) is b


def test_capture_age_is_bounded_even_when_old_frames_arrive_together():
    s = trained()
    old = item(2, 41667)
    # Observation time is fresh, but original pixels are more than 100ms old.
    old = old[0], (old[1][0], 0.2)
    latest = item(3, 200000)
    assert choose(s, [old, latest], buffered=0) is latest


def test_source_identity_change_cannot_select_an_older_presentation():
    s = trained()
    a = item(2, 41667)
    b = item(3, 60000)
    b = (
        b[0],
        (
            replace(
                b[1][0],
                provenance={
                    **b[1][0].provenance,
                    "cli_frame_header": {
                        "sessionId": "new",
                        "epoch": "new",
                        "source": {"presentationId": "new-view", "sdkEpoch": "sdk"},
                    },
                },
            ),
            b[1][1],
        ),
    )
    assert choose(s, [a, b]) is b


def test_sampler_retains_no_rgb_and_only_inspects_bounded_history():
    s = trained()
    latest = item(12, 100000)
    visited = []

    def history():
        for i in range(100):
            visited.append(i)
            v, (f, t) = item(11 - i, 99000 - i)
            yield v, f, t, 3

    s.choose(
        latest, history(), after=1, now=0.1, buffered=25, reserve=9, control_sequence=-1
    )
    assert len(visited) <= s.max_candidates
    assert not any(isinstance(v, (bytes, EngineFrame)) for v in vars(s).values())


@pytest.mark.parametrize(
    "jitter,improves",
    [
        ([0.012, -0.012, 0.009, -0.009], False),
        ([0.028, 0.012, 0.026, 0.010], False),
        ([0.01, 0.03, 0.05, 0.07, 0.05, 0.03], True),
    ],
)
def test_irregular_ready_opportunities_preserve_samples_and_use_available_choices(
    jitter, improves
):
    # A virtual schedule comparison, not a GPU throughput benchmark. Both policies
    # receive the same 60Hz captures and 24Hz consumer opportunities with jitter.
    s = trained()
    adaptive, latest = [0], [0]
    versions = []
    for n in range(1, 241):
        now = n / 24 + jitter[n % len(jitter)]
        newest = int(now * 60)
        frames = [
            item(i + 1, round(i * 1_000_000 / 60))
            for i in range(max(0, newest - 7), newest + 1)
        ]
        result = choose(s, frames, now=now)
        versions.append(result[0])
        adaptive.append(result[1][0].provenance["capture_pts_us"])
        latest.append(frames[-1][1][0].provenance["capture_pts_us"])
    assert len(set(versions)) == 240
    assert versions == sorted(versions)

    def error(values):
        return sum(((b - a) - 1_000_000 / 24) ** 2 for a, b in zip(values, values[1:]))

    # Integer-microsecond capture timestamps may change the RMS by <1us even
    # when both sequences already have the minimum 2/3-tick spacing variance.
    assert math.sqrt(error(adaptive) / 240) <= math.sqrt(error(latest) / 240) + 1
    if improves:
        assert error(adaptive) < error(latest)
        assert s.counts["uniform"] > 0


def test_config_opt_in_requires_segmented_room_and_does_not_change_cli_default(
    monkeypatch,
):
    import os

    from sglang.multimodal_gen.runtime.realtime.whitebox import session

    for key in list(os.environ):
        if key.startswith("WHITEBOX_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(session, "load_plugin", lambda _: lambda **kwargs: None)
    config, _ = session.from_environment(room=True)
    assert config.model_source_sampling == "latest"
    monkeypatch.setenv("WHITEBOX_MODEL_SOURCE_SAMPLING", "adaptive")
    with pytest.raises(ValueError, match="render segments"):
        session.from_environment(room=True)
    monkeypatch.setenv("WHITEBOX_RENDER_SEGMENTS", "1")
    monkeypatch.setenv("WHITEBOX_CLI_FPS", "60")
    config, _ = session.from_environment(room=True)
    assert config.model_source_sampling == "adaptive"
    assert config.spec.fps == 24  # Indexed model protocol, not a new FPS limiter.
    monkeypatch.setenv("WHITEBOX_MODEL_SOURCE_SAMPLING", "unknown")
    with pytest.raises(ValueError, match="sampling policy"):
        session.from_environment(room=True)


def source():
    return NS(
        config=NS(model_source_sampling="adaptive"),
        latest=LatestFrame(),
        retained=SourceFrames(max_bytes=1 << 20),
        input_sequence=-1,
    )


async def capture(owner, frame):
    version, (value, timestamp) = frame
    owner.retained.append(version, value, timestamp)
    await owner.latest.put((value, timestamp))


def test_segment_source_waits_only_for_real_frames_and_remains_cancellable():
    async def run():
        owner = source()
        view = SegmentSource(owner)
        task = asyncio.create_task(view.get(0))
        await asyncio.sleep(0)
        assert not task.done()
        await capture(owner, item(1, 0))
        result = await asyncio.wait_for(task, 0.2)
        assert result[0] == 1
        waiting = asyncio.create_task(view.get(1))
        await asyncio.sleep(0)
        view.stop()
        assert await asyncio.wait_for(waiting, 0.2) is None

    asyncio.run(run())


@pytest.mark.parametrize("cadence_deferred", [False, True])
def test_real_source_control_dispatch_does_not_wait_for_new_captures(
    tmp_path, cadence_deferred
):
    import msgspec
    from test_whitebox_interactive import BrowserUI, Engine
    from test_whitebox_render_lifecycle import StrictLeases

    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncArchiveManager,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderJournal,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig

    async def run():
        spec = VideoSpec(32, 32)
        engine = Engine(spec)
        engine.stream_id = "view"
        leases = StrictLeases(spec, "clip-barrier-v1")
        archives = AsyncArchiveManager(recorder_slots=3)
        journal = RenderJournal(
            MemoryRoomStore(),
            user_id="user",
            source_session_id="source",
            source_epoch="epoch",
            owner_epoch="room",
        )
        await journal.create()

        async def factory(**_):
            return engine

        owner = SourceSession(
            NS(runtime=NS(players={}), make_segment_leases=lambda *_: leases),
            journal,
            source_epoch="epoch",
            user_id="user",
            session_id="source",
            trace_id="sampling-control",
            config=WhiteboxConfig(
                engine_factory=factory,
                ui_factory=lambda **_: BrowserUI(),
                spool=tmp_path,
                spec=spec,
                room_owned=True,
                capture_clock=True,
                model_source_sampling="adaptive",
                startup_seconds=5,
                drain_seconds=2,
            ),
            archives=archives,
        )
        running = asyncio.Event()
        running.set()

        async def consume():
            while True:
                m = msgspec.msgpack.decode(await owner.recv())
                if m.get("type") in {"whitebox.frame", "frame_batch"}:
                    owner.media_delivered()

        async def produce():
            while True:
                await running.wait()
                await engine.push()
                await asyncio.sleep(0.01)

        async def until(predicate):
            async with asyncio.timeout(5):
                while not predicate():
                    await asyncio.sleep(0.005)

        tasks = []
        try:
            await owner.send(
                msgspec.msgpack.encode(
                    dict(type="init", whitebox={"world_id": "world"}, prompt="initial")
                )
            )
            tasks = [asyncio.create_task(consume()), asyncio.create_task(produce())]
            await until(lambda: owner.ready.is_set())
            await owner.send(
                msgspec.msgpack.encode(
                    dict(
                        type="command",
                        op="render.start",
                        request_id="start",
                        source_session_id="source",
                        source_epoch="epoch",
                        expected_render_revision=0,
                        prompt="initial",
                        seed=42,
                        references=[],
                    )
                )
            )
            await until(
                lambda: owner.current_segment and owner.current_segment.first_decodable
            )
            segment = owner.current_segment
            if cadence_deferred:
                # Exercise the new data wait with the real independent input
                # and Prompt paths; the admission policy is tested separately.
                deferred = []

                def defer(*args, **kwargs):
                    deferred.append(True)
                    return False

                segment.source_view.sampler.admit = defer
                await until(lambda: deferred)
            running.clear()
            await asyncio.sleep(0.1)
            capture_count = owner.submitted
            await owner.send(
                msgspec.msgpack.encode(
                    dict(
                        type="event",
                        kind="input",
                        sequence=owner.input_sequence + 1,
                        payload={"keys": ["W"]},
                    )
                )
            )
            await until(lambda: engine.inputs and engine.inputs[-1] == {"keys": ["W"]})
            held = {"held_keys": ["KeyW"], "held_buttons": [], "events": []}
            start_count = len(engine.inputs)
            for payload in [
                held,
                held,
                {"held_keys": [], "held_buttons": [], "events": []},
            ]:
                await owner.send(
                    msgspec.msgpack.encode(
                        dict(
                            type="event",
                            kind="input",
                            sequence=owner.input_sequence + 1,
                            payload=payload,
                        )
                    )
                )
            assert engine.inputs[start_count:] == [
                held,
                held,
                {"held_keys": [], "held_buttons": [], "events": []},
            ]
            assert (
                owner.sampling_inputs.revision == 3
            )  # Unknown + press + release; renewal still delivered.
            await asyncio.sleep(0.51)  # Existing control rate limit, not sampling.
            await owner.send(
                msgspec.msgpack.encode(
                    dict(
                        type="command",
                        op="prompt.replace",
                        request_id="update",
                        source_session_id="source",
                        source_epoch="epoch",
                        segment_id=segment.segment_id,
                        segment_epoch=segment.segment_epoch,
                        expected_prompt_version=1,
                        prompt_version=2,
                        prompt="updated",
                        effective={
                            "source_frame_index": engine.next - 1,
                            "policy": "at_or_after",
                        },
                    )
                )
            )
            await until(lambda: hasattr(leases.opens[0], "switch_candidate"))
            assert leases.opens[0].switch_candidate["payload"]["text"] == "updated"
            assert (
                owner.submitted == capture_count
            )  # Both commands crossed without a capture.
            assert segment.source_view.sampler is not None
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await owner.close()
            await archives.close()
        assert leases.active == 0

    asyncio.run(run())


def test_persistent_observed_input_requires_fresh_pixels_not_unlimited_admission():
    sampler = trained(20)
    observed = {"latest": {"keys": ["W"]}}

    def offer(version, pts, sequence=1, buffered=25):
        latest = item(version, pts, observed=observed)
        demand = dict(
            after=sampler.last_version,
            buffered=buffered,
            reserve=9,
            control_sequence=sequence,
        )
        if not sampler.admit(latest, **demand):
            return None
        return sampler.choose(latest, [], now=pts / 1_000_000, **demand)

    assert offer(2, 25000)[0] == 2  # One fresh control can advance a slot.
    assert offer(3, 41667) is None
    assert offer(4, 83333) is None
    assert offer(5, 100000)[0] == 5
    assert sampler.counts["control"] == 1
    assert sampler.counts["fresh_unverified"] == 1
    assert sampler.max_extra_age_us == 0
    assert sampler.admission_counts["early_captures"] == 2


def test_fast_control_updates_borrow_bounded_slots_and_low_water_never_waits():
    sampler = trained(20)
    first = item(2, 10000, observed={"latest": {"sequence": 1}})
    demand = dict(after=1, buffered=25, reserve=9, control_sequence=1)
    assert sampler.admit(first, **demand)
    sampler.choose(first, [], now=0.01, **demand)
    second = item(3, 20000, observed={"latest": {"sequence": 2}})
    assert not sampler.admit(
        second, after=2, buffered=25, reserve=9, control_sequence=2
    )
    assert sampler.admit(second, after=2, buffered=0, reserve=9, control_sequence=2)
    assert sampler.admit(second, after=2, buffered=25, reserve=9, urgent=True)


def test_backpressure_rebases_once_without_burst_catchup():
    sampler = trained(20)
    late = item(30, 500000)
    demand = dict(after=1, buffered=25, reserve=9, control_sequence=-1)
    assert sampler.admit(late, **demand)
    assert sampler.choose(late, [], now=0.5, **demand) is late
    assert sampler.next_pts == 550000
    assert not sampler.admit(
        item(31, 516667), after=30, buffered=25, reserve=9, control_sequence=-1
    )


def test_deferred_capture_wakes_on_model_progress_and_eos_keeps_the_last_real_frame():
    async def run():
        owner = source()
        view = SegmentSource(owner)
        await capture(owner, item(1, 0))
        await capture(owner, item(2, 25000))
        condition, progress = asyncio.Condition(), [0]
        waiter = asyncio.create_task(
            wait_for_capture_or_progress(view, 2, condition, lambda: progress[0], 0)
        )
        await asyncio.sleep(0)
        async with condition:
            progress[0] = 1
            condition.notify_all()
        assert await asyncio.wait_for(waiter, 0.2) is None
        assert (await view.get(1))[0] == 2
        ended = asyncio.create_task(
            wait_for_capture_or_progress(view, 2, condition, lambda: progress[0], 1)
        )
        await asyncio.sleep(0)
        await owner.latest.close()
        assert await asyncio.wait_for(ended, 0.2) is None
        assert (await view.get(1))[0] == 2  # A deferred frame is not consumed.
        assert await view.get(2) is None

    asyncio.run(run())


def test_sampling_wait_keeps_simultaneous_pinned_capture_and_cancels_all_waiters():
    async def run():
        owner, condition = source(), asyncio.Condition()
        view = SegmentSource(owner)
        pinned = item(1, 0)
        view.first = pinned
        result = await wait_for_capture_or_progress(view, 100, condition, lambda: 1, 0)
        assert result is None
        assert view.first is pinned
        assert view.sampling_fence is None
        assert await view.get(100) is pinned
        assert view.sampling_fence == 1
        waiting = asyncio.create_task(
            wait_for_capture_or_progress(view, 1, condition, lambda: 0, 0)
        )
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(run())


def simulate_batched_consumer(
    *,
    adaptive,
    batch,
    controls_hz,
    source_fps=36,
    model_fps=18.6,
    duration=60,
    feedback=False,
    semantic_controls=False,
    capacity=34,
):
    """Fixed consumer/capture clocks, including real credit and completion bursts.

    With persistent observed-only input, the old policy always chose latest.
    This baseline reproduces that branch, not a fictitious fixed-FPS limiter.
    """
    sampler = AdaptiveSourceSampler()
    reserve = 18
    sent = received = cursor = 0
    latest = None
    history, queue = deque(maxlen=8), deque()
    selected, outputs = [], []
    events = [(i / source_fps, 1, i + 1) for i in range(int(duration * source_fps) + 1)]
    heapq.heapify(events)
    computing = started = False
    high_water = 0
    while events:
        now, kind, value = heapq.heappop(events)
        sequence = int(now * controls_hz) if controls_hz else 1
        if kind == 1:
            latest = item(
                value,
                round(now * 1_000_000),
                observed={"latest": {"sequence": sequence}},
            )
            history.append((value, *latest[1], 3))
        else:
            sampler.completed(
                received + value, now, buffered=sent - received, reserve=reserve
            )
            for _ in range(value):
                outputs.append((now, queue.popleft()))
            received += value
            computing, started = False, True
        while latest is not None and latest[0] > cursor and sent - received < capacity:
            sampler.supply(sent - received, reserve, available=True)
            if feedback:
                sampler.feedback.observe(
                    now,
                    buffered=sent - received,
                    reserve=reserve,
                    capacity=capacity,
                    oldest_age=now - queue[0][1][1] if queue else None,
                    period=sampler.period_us / 1_000_000 if sampler.period_us else None,
                    enabled=received > 0,
                )
            demand = dict(
                after=cursor,
                buffered=sent - received,
                reserve=reserve,
                startup=received == 0,
                urgent=False,
                control_sequence=sequence,
                **({"semantic_control": True} if semantic_controls else {}),
            )
            if adaptive and not sampler.admit(latest, **demand):
                break
            chosen = (
                sampler.choose(latest, list(reversed(history)), now=now, **demand)
                if adaptive
                else latest
            )
            cursor = chosen[0]
            sent += 1
            selected.append(chosen)
            queue.append(chosen)
            high_water = max(high_water, sent - received)
        sampler.supply(
            sent - received,
            reserve,
            available=latest is not None and latest[0] > cursor,
        )
        if not computing and sent - received >= (batch if started else reserve):
            computing = True
            heapq.heappush(events, (now + batch / model_fps, 0, batch))
    pts = [
        frame.provenance["capture_pts_us"] / 1_000_000
        for _, (frame, captured) in selected
        if 10 <= captured <= duration - 5
    ]
    gaps = sorted(b - a for a, b in zip(pts, pts[1:]))
    output_count = sum(10 <= time <= duration - 5 for time, _ in outputs)
    delays = []
    if controls_hz:
        for seq in range(
            math.ceil(10 * controls_hz), int((duration - 5) * controls_hz)
        ):
            first = next(
                (
                    time
                    for time, (_, (frame, _)) in outputs
                    if frame.observed_input["latest"]["sequence"] >= seq
                ),
                None,
            )
            if first is not None:
                delays.append(first - seq / controls_hz)
    delays.sort()
    return dict(
        count=output_count,
        gap_p95=gaps[int(0.95 * (len(gaps) - 1))],
        input_p95=delays[int(0.95 * (len(delays) - 1))] if delays else 0,
        unique=len({version for version, _ in selected}) == len(selected),
        high_water=high_water,
        extra_age_us=sampler.max_extra_age_us,
        pi_updates=sampler.feedback.updates if hasattr(sampler, "feedback") else 0,
        pi_scale=sampler.feedback.scale if hasattr(sampler, "feedback") else 1,
        source_delay_p95=sorted(
            time - frame[1][1] for time, frame in outputs if 10 <= time <= duration - 5
        )[int(0.95 * (output_count - 1))],
    )


@pytest.mark.parametrize("batch", [9, 17])
@pytest.mark.parametrize("controls_hz", [0, 5, 36])
def test_capture_clock_admission_smooths_bursts_without_losing_throughput(
    batch, controls_hz
):
    baseline = simulate_batched_consumer(
        adaptive=False, batch=batch, controls_hz=controls_hz
    )
    candidate = simulate_batched_consumer(
        adaptive=True, batch=batch, controls_hz=controls_hz
    )
    assert baseline["gap_p95"] > 0.2
    assert candidate["gap_p95"] < 0.1
    assert candidate["count"] == baseline["count"]
    assert candidate["input_p95"] <= baseline["input_p95"] + 1e-8
    assert candidate["unique"] and candidate["high_water"] <= 34
    assert (
        candidate["extra_age_us"] == 0
    )  # Observed input is not effective-input proof.


def test_pinned_start_eos_and_two_players_keep_independent_cursors():
    async def run():
        owner, other = source(), source()
        a, b = SegmentSource(owner), SegmentSource(other)
        a.first = item(1, 0)
        await capture(owner, item(1, 0))
        await capture(owner, item(2, 41667))
        await capture(other, item(1, 400000))
        a.sampler = trained()
        pinned = await a.get(100)
        args = dict(now=0.05, buffered=25, reserve=9, startup=False, urgent=False)
        assert a.select_model_source(pinned, after=100, **args) is pinned
        assert b.sampler.last_version is None
        await owner.latest.close()
        latest = await a.get(1)
        assert a.select_model_source(latest, after=1, **args)[0] == 2
        assert await a.get(2) is None
        assert (await b.get(0))[1][0].provenance["capture_pts_us"] == 400000

    asyncio.run(run())


def test_context_boundary_cannot_be_crossed_backwards_by_sampling():
    async def run():
        owner = source()
        view = SegmentSource(owner)
        native = dict(
            sessionId="native",
            epoch="native-epoch",
            presentationId="view",
            sdkEpoch="sdk",
        )
        view.context_boundary = dict(
            source=native,
            effective=dict(sourceFrameId=2, sourceTimeUs=100, simulationTick=2),
        )
        for version, pts in [(1, 0), (2, 100)]:
            v, (frame, t) = item(version, pts)
            frame = replace(
                frame,
                provenance={
                    **frame.provenance,
                    "cli_frame_header": dict(
                        sessionId="native",
                        epoch="native-epoch",
                        source=dict(
                            presentationId="view",
                            sdkEpoch="sdk",
                            sourceFrameId=version,
                            sourceTimeUs=pts,
                            simulationTick=version,
                        ),
                    ),
                },
            )
            await capture(owner, (v, (frame, t)))
        view.sampler = trained()
        result = await view.get(0)
        assert result[0] == 2
        assert (
            view.select_model_source(
                result,
                after=0,
                now=0.0001,
                buffered=25,
                reserve=9,
                startup=False,
                urgent=False,
            )
            is result
        )
        assert view.context_boundary is None

    asyncio.run(run())


def test_input_renewals_do_not_change_priority_but_edges_and_unknown_inputs_do():
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_sampling import (
        InputChangeTracker,
    )

    tracker = InputChangeTracker()

    def note(**payload):
        return tracker.observe(payload)

    assert note(held_keys=["KeyW"], held_buttons=[], events=[]) == 1
    assert note(held_keys=["KeyW"], held_buttons=[], events=[]) == 1
    assert note(held_keys=[], held_buttons=[], events=[]) == 2
    assert (
        note(held_keys=[], held_buttons=[], events=[{"type": "keyup", "code": "KeyW"}])
        == 3
    )
    assert note(held_keys=[], events=[{"type": "pointermove", "dx": 1}]) == 4
    assert note(held_keys=[], events=[{"type": "pointermove", "dx": 1}]) == 5
    assert note(custom_action="jump") == 6
    assert note(custom_action="jump") == 7
    assert note(held_keys=[]) == 8  # Unknown state cannot establish a renewal.
    assert note(held_keys=[]) == 8


def test_semantic_priority_keeps_effective_context_and_action_fences():
    s = trained()
    first = item(
        2,
        50000,
        observed={
            "latest": {"held_keys": ["KeyW"]},
            "receipts": [{"type": "input.receipt", "sequence": 1}],
        },
    )
    d = dict(
        after=1, buffered=25, reserve=18, control_sequence=1, semantic_control=True
    )
    assert s.admit(first, **d)
    s.choose(first, [], now=0.05, **d)
    renewal = item(
        3,
        60000,
        observed={
            "latest": {"held_keys": ["KeyW"]},
            "receipts": [{"type": "input.receipt", "sequence": 2}],
        },
    )
    assert not s.control_changed(s.input_identity(renewal[1][0]), 1, True)
    action = item(
        3,
        60000,
        observed={"receipts": [{"type": "action.receipt", "actionId": "clicked"}]},
    )
    assert s.control_changed(s.input_identity(action[1][0]), 1, True)
    effective = item(3, 60000, effective={"applied": True})
    assert s.control_changed(s.input_identity(effective[1][0]), 1, True)
    changed = item(3, 60000)
    changed[1][0].provenance["source_epoch"] = "new"
    assert s.control_changed(s.input_identity(changed[1][0]), 1, True)


def test_pi_has_bounded_slew_and_anti_windup_and_freezes_at_protocol_fences():
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_sampling import (
        QueueFeedback,
    )

    q = QueueFeedback()
    scales = []
    for i in range(1001):
        q.observe(
            i * 0.05,
            buffered=34,
            reserve=18,
            capacity=34,
            oldest_age=3,
            period=0.05,
            enabled=True,
        )
        scales.append(q.scale)
    assert q.updates > 40 and 1 < q.scale <= 1.15
    assert max(abs(b - a) for a, b in zip(scales, scales[1:])) <= 0.02000001
    assert abs(q.integral) <= 0.06
    for i in range(1001, 1201):
        q.observe(
            i * 0.05,
            buffered=18,
            reserve=18,
            capacity=34,
            oldest_age=0.1,
            period=0.05,
            enabled=True,
        )
    assert q.scale < 1.02  # It can unwind after overload, not stay saturated.
    q.observe(
        61,
        buffered=25,
        reserve=18,
        capacity=34,
        oldest_age=0.5,
        period=0.05,
        enabled=False,
    )
    assert q.scale == 1 and q.integral == 0 and q.last is None
    q.observe(
        62,
        buffered=25,
        reserve=18,
        capacity=34,
        oldest_age=None,
        period=0.05,
        enabled=True,
    )
    assert q.scale == 1 and q.last is None


def test_pi_queue_limit_does_not_block_starved_input_or_control_dispatch():
    s = trained(20)
    s.feedback.scale = 1.15
    early = item(2, 10000)
    assert not s.admit(early, after=1, buffered=25, reserve=18, control_sequence=9)
    assert s.admit(early, after=1, buffered=17, reserve=18, control_sequence=9)
    assert s.admit(
        early, after=1, buffered=25, reserve=18, control_sequence=9, urgent=True
    )


@pytest.mark.parametrize("batch", [9, 17])
@pytest.mark.parametrize("controls_hz", [0, 5, 36])
def test_pi_keeps_batch_throughput_and_improves_or_preserves_model_age(
    batch, controls_hz
):
    baseline = simulate_batched_consumer(
        adaptive=True, batch=batch, controls_hz=controls_hz
    )
    candidate = simulate_batched_consumer(
        adaptive=True,
        batch=batch,
        controls_hz=controls_hz,
        feedback=True,
        semantic_controls=True,
    )
    assert candidate["count"] == baseline["count"]
    assert candidate["gap_p95"] < 0.085
    assert candidate["input_p95"] <= baseline["input_p95"] + 1e-6
    assert candidate["source_delay_p95"] <= baseline["source_delay_p95"] + 1e-6
    assert candidate["pi_updates"] > 0
    assert (
        candidate["high_water"] <= 34
        and candidate["unique"]
        and candidate["extra_age_us"] == 0
    )


@pytest.mark.parametrize("batch", [9, 17])
def test_tight_native_credit_never_gets_bypassed_by_pi(batch):
    baseline = simulate_batched_consumer(
        adaptive=True, batch=batch, controls_hz=5, capacity=25
    )
    candidate = simulate_batched_consumer(
        adaptive=True,
        batch=batch,
        controls_hz=5,
        capacity=25,
        feedback=True,
        semantic_controls=True,
    )
    assert candidate["count"] == baseline["count"]
    assert candidate["high_water"] <= 25 and candidate["unique"]
    assert candidate["input_p95"] <= baseline["input_p95"] + 1e-6


def test_new_state_receipt_advances_priority_but_renewal_receipt_does_not():
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_sampling import (
        InputChangeTracker,
    )

    t = InputChangeTracker()
    held = {"held_keys": ["KeyW"], "events": []}
    t.observe(held, 0)
    t.observe(held, 1)
    t.observe({"held_keys": []}, 2)
    receipt = lambda seq: [{"type": "input.receipt", "clientSequence": seq}]
    assert t.acknowledged([]) == 0
    assert t.acknowledged(receipt(0)) == 1
    assert t.acknowledged(receipt(1)) == 1
    assert t.acknowledged(receipt(2)) == 2
    for seq in range(3, 2050):
        t.observe({"held_keys": []}, seq)
    assert len(t.receipt_revisions) == 1024
