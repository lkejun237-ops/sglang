"""Server-only periodic prompt updates: bounded state and exact source ownership."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest
from test_whitebox_render_contract import command, setup

from sglang.multimodal_gen.runtime.realtime.whitebox.condition_refresh import (
    ConditionRefreshPolicy,
    PeriodicConditionRefresh,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.opening_presentation import (
    applied_context_operation,
)


async def fixture(tmp_path):
    journal = await setup()
    start = await journal.start(
        command(
            "render.start",
            "start",
            expected_render_revision=0,
            prompt="current prompt",
            references=[],
        )
    )
    await journal.prompt_applied(
        start["segment_id"], 1, dict(first_output_frame=0, num_frames=1)
    )
    await journal.update(lambda s: s["segment"].update(state="streaming"))
    source = NS(
        journal=journal,
        close_requested=False,
        binding_ended=False,
        host=NS(),
        session_id="source",
        context_channel=None,
    )
    segment = NS(
        owner=source,
        segment_id=start["segment_id"],
        segment_epoch=start["segment_epoch"],
        cancel_requested=False,
        source_view=NS(stopped=asyncio.Event()),
        pending_prompt=None,
        condition_switch=None,
        sink=NS(busy=False),
        clip_switch=True,
        source_send_lock=asyncio.Lock(),
        prompt_available=asyncio.Event(),
        prompt_by_event={None: dict(version=1)},
        closed_event=asyncio.Event(),
    )
    clock = NS(now=0.0)
    policy = ConditionRefreshPolicy(tmp_path / "policy.json")
    timer = PeriodicConditionRefresh(segment, policy, clock=lambda: clock.now)
    return journal, timer, clock


def test_operator_policy_hot_reload_zero_and_last_good(tmp_path):
    path = tmp_path / "policy.json"
    policy = ConditionRefreshPolicy(path)
    assert policy.read(0) == (0, 0)
    path.write_text('{"interval_seconds":5}')
    assert policy.read(1) == (5, 1)
    for i, raw in enumerate(
        [
            '{"interval_seconds":-1}',
            '{"interval_seconds":true}',
            '{"interval_seconds":NaN}',
            '{"interval_seconds":Infinity}',
            "{bad",
        ],
        2,
    ):
        path.write_text(raw)
        assert policy.read(i) == (5, 1)
    path.unlink()
    assert policy.read(10) == (5, 1)
    path.write_text('{"interval_seconds":2.5}')
    assert policy.read(11) == (2.5, 2)
    path.write_text('{"interval_seconds":0}')
    assert policy.read(12) == (0, 3)


def test_dynamic_interval_and_disable_cancel_only_unsent_refresh(tmp_path):
    async def run():
        journal, timer, clock = await fixture(tmp_path)
        timer.policy.path.write_text('{"interval_seconds":5}')
        await timer.tick(0)
        timer.observe(dict(num_frames=1, prompt_version=1))
        await timer.tick(4.99)
        assert not journal.snapshot["segment"]["prompt_queue"]
        await timer.tick(5)
        current = journal.snapshot["segment"]
        assert len(current["prompt_queue"]) == 1
        assert current["prompt_queue"][0]["text"] == "current prompt"
        assert "references" not in current["prompt_queue"][0]
        assert len(journal.snapshot["operations"]) == 1
        await timer.tick(100)
        assert len(journal.snapshot["segment"]["prompt_queue"]) == 1
        timer.policy.path.write_text('{"interval_seconds":0}')
        await timer.tick(101)
        assert journal.snapshot["segment"]["prompt_queue"] == []
        assert journal.snapshot["segment"]["accepted_version"] == 1
        timer.policy.path.write_text('{"interval_seconds":2}')
        await timer.tick(102)
        timer.observe(dict(num_frames=1, prompt_version=1))
        await timer.tick(103)
        assert not journal.snapshot["segment"]["prompt_queue"]
        await timer.tick(104)
        op = journal.snapshot["segment"]["periodic_refresh"]["operation"]
        assert await journal.mark_periodic_dispatch(
            timer.segment.segment_id, op["request_id"]
        )
        timer.policy.path.write_text('{"interval_seconds":0}')
        await timer.tick(105)
        assert len(journal.snapshot["segment"]["prompt_queue"]) == 1
        await journal.prompt_applied(
            timer.segment.segment_id, 2, dict(first_output_frame=17, num_frames=17)
        )
        await timer.tick(200)
        assert not journal.snapshot["segment"]["prompt_queue"]

    asyncio.run(run())


def test_long_session_refresh_keeps_user_journal_bounded_and_business_identity(
    tmp_path,
):
    async def run():
        journal, timer, clock = await fixture(tmp_path)

        def business(s):
            s["segment"]["context_segment_id"] = "scene"
            s["segment"]["model_context_applied_request_id"] = "start"
            s["operations"]["start"].update(
                prompt_version=1,
                state="applied",
                model_context={"segmentId": "scene", "conditionId": "original"},
                model_context_lifecycle={"lifecycleId": "life", "generation": 1},
            )

        await journal.update(business)
        original = json.dumps(journal.snapshot["operations"], sort_keys=True)
        for n in range(1440):
            op = await journal.enqueue_periodic_refresh(
                timer.segment.segment_id, timer.segment.segment_epoch, n + 1
            )
            assert op is not None
            assert await journal.mark_periodic_dispatch(
                timer.segment.segment_id, op["request_id"]
            )
            await journal.prompt_applied(
                timer.segment.segment_id,
                n + 2,
                dict(first_output_frame=(n + 1) * 17, num_frames=17),
            )
        assert json.dumps(journal.snapshot["operations"], sort_keys=True) == original
        assert journal.snapshot["segment"]["periodic_refresh"]["completed"] == 1440
        assert len(json.dumps(journal.snapshot)) < 5000
        assert applied_context_operation(journal.snapshot)["request_id"] == "start"

    asyncio.run(run())


@pytest.mark.parametrize("busy", ["sink", "prompt", "closing", "disconnect", "opening"])
def test_periodic_refresh_yields_to_real_controls_and_opening(tmp_path, busy):
    async def run():
        journal, timer, clock = await fixture(tmp_path)
        timer.policy.path.write_text('{"interval_seconds":5}')
        await timer.tick(0)
        timer.observe(dict(num_frames=1, prompt_version=1))
        if busy == "sink":
            timer.segment.sink.busy = True
        if busy == "prompt":
            timer.segment.condition_switch = {}
        if busy == "closing":
            timer.segment.owner.close_requested = True
        if busy == "disconnect":
            timer.segment.owner.host.sessions = {}
        if busy == "opening":
            timer.segment.owner.context_channel = NS(first_presented_supported=True)
        await timer.tick(6)
        assert not journal.snapshot["segment"]["prompt_queue"]

    asyncio.run(run())


@pytest.mark.parametrize("close_inflight", [False, True])
def test_real_source_uses_existing_prompt_wire_and_hot_disable(
    tmp_path, monkeypatch, close_inflight
):
    import msgspec
    from test_whitebox_render_lifecycle import (
        AsyncArchiveManager,
        BrowserUI,
        Engine,
        SourceSession,
        StrictLeases,
        VideoSpec,
        WhiteboxConfig,
    )

    monkeypatch.delenv("WHITEBOX_BOUND_COMPUTE", raising=False)

    async def run():
        spec = VideoSpec(32, 32)
        engine, leases = Engine(spec), StrictLeases(spec, "clip-barrier-v1")
        engine.stream_id = "view"
        archives = AsyncArchiveManager(recorder_slots=3)
        journal = await setup()
        policy = ConditionRefreshPolicy(tmp_path / "policy.json")
        policy.path.write_text('{"interval_seconds":0.1}')

        async def factory(**_):
            return engine

        source = SourceSession(
            NS(
                runtime=NS(players={}),
                make_segment_leases=lambda *_: leases,
                condition_refresh_policy=policy,
            ),
            journal,
            source_epoch="epoch",
            user_id="u",
            session_id="source",
            trace_id="source",
            config=WhiteboxConfig(
                engine_factory=factory,
                ui_factory=lambda **_: BrowserUI(),
                spool=tmp_path,
                spec=spec,
                room_owned=True,
                startup_seconds=8,
                drain_seconds=2,
            ),
            archives=archives,
        )
        messages = []

        async def consume():
            while True:
                m = msgspec.msgpack.decode(await source.recv())
                messages.append(m)
                if m.get("type") in {"whitebox.frame", "frame_batch"}:
                    source.media_delivered()

        async def produce():
            while True:
                await engine.push()
                await asyncio.sleep(0.01)

        async def wait(predicate):
            async with asyncio.timeout(8):
                while not predicate():
                    await asyncio.sleep(0.01)

        tasks = [asyncio.create_task(consume()), asyncio.create_task(produce())]
        try:
            await source.send(
                msgspec.msgpack.encode(
                    dict(
                        type="init",
                        whitebox={"world_id": "world"},
                        prompt="same prompt",
                    )
                )
            )
            await source.ready.wait()
            await source.send(
                msgspec.msgpack.encode(
                    dict(
                        type="command",
                        **command(
                            "render.start",
                            "start-live",
                            expected_render_revision=0,
                            prompt="same prompt",
                        ),
                    )
                )
            )
            await wait(
                lambda: (
                    journal.snapshot["segment"]
                    .get("periodic_refresh", {})
                    .get("completed", 0)
                    >= 2
                )
            )
            assert len(leases.opens) == 1 and leases.active == 1
            worker = leases.opens[0]
            assert len(worker.switched) >= 2 and all(
                s["text"] == "same prompt" and "references" not in s
                for s in worker.switched
            )
            assert len(journal.snapshot["operations"]) == 1
            assert not any(
                m.get("type") == "operation.result"
                and str(m.get("request_id", "")).startswith("periodic-")
                for m in messages
            )
            if close_inflight:
                await wait(
                    lambda: (
                        journal.snapshot["segment"]
                        .get("periodic_refresh", {})
                        .get("operation", {})
                        .get("state")
                        == "dispatching"
                    )
                )
                current = journal.snapshot["segment"]
                await source.send(
                    msgspec.msgpack.encode(
                        dict(
                            type="command",
                            **command(
                                "render.end",
                                "end-active-refresh",
                                segment_id=current["segment_id"],
                                segment_epoch=current["segment_epoch"],
                                expected_render_revision=journal.snapshot[
                                    "render_revision"
                                ],
                                mode="drain",
                            ),
                        )
                    )
                )
                await wait(lambda: source.current_segment.closed_event.is_set())
                assert leases.active == 0
                return
            policy.path.write_text('{"interval_seconds":0}')
            await wait(
                lambda: (
                    policy.seconds == 0
                    and not journal.snapshot["segment"]["prompt_queue"]
                )
            )
            count = len(worker.switched)
            await source.send(
                msgspec.msgpack.encode(
                    dict(
                        type="event", kind="input", sequence=0, payload={"keys": ["w"]}
                    )
                )
            )
            assert len(engine.inputs) == 1
            await asyncio.sleep(0.35)
            assert len(worker.switched) == count
            await source.close()
            assert leases.active == 0
        finally:
            await source.close()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await archives.close()

    asyncio.run(run())


def test_real_prompt_cancels_undispatched_refresh_and_resets_lineage(tmp_path):
    async def run():
        journal, timer, _ = await fixture(tmp_path)
        op = await journal.enqueue_periodic_refresh(
            timer.segment.segment_id, timer.segment.segment_epoch, 1
        )
        change = command(
            "prompt.replace",
            "user-update",
            segment_id=timer.segment.segment_id,
            segment_epoch=timer.segment.segment_epoch,
            expected_prompt_version=2,
            prompt_version=3,
            prompt="different",
            effective={"source_frame_index": 0, "policy": "at_or_after"},
        )
        await journal.prompt(change)
        current = journal.snapshot["segment"]
        assert current["periodic_refresh"]["operation"]["state"] == "cancelled"
        assert (
            len(current["prompt_queue"]) == 1
            and current["prompt_queue"][0]["request_id"] == "user-update"
        )
        assert not await journal.mark_periodic_dispatch(
            timer.segment.segment_id, op["request_id"]
        )
        await journal.prompt_applied(
            timer.segment.segment_id, 3, dict(first_output_frame=17, num_frames=17)
        )
        assert journal.snapshot["segment"]["prompt"] == "different"
        assert "refresh_condition_version" not in journal.snapshot["segment"]

    asyncio.run(run())
