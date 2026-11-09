"""Sink controls retain the original journal, wire order and player ownership."""

import asyncio
import copy
from types import SimpleNamespace

import msgspec
import pytest
from test_whitebox_render_contract import command, setup

from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
    InteractiveSession,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import RenderError
from sglang.multimodal_gen.runtime.realtime.whitebox.sink_control import SinkControl


async def fixture():
    journal = await setup()
    start = await journal.start(
        command(
            "render.start",
            "start",
            expected_render_revision=0,
            prompt="scene",
            references=[],
        )
    )
    await journal.update(
        lambda state: state["segment"].update(state="streaming", applied_version=1)
    )
    sent, receipts, tasks = [], [], []

    async def receipt(value):
        receipts.append(copy.deepcopy(value))

    async def send(value):
        wire = msgspec.msgpack.decode(value)
        sent.append(wire)
        pending = journal.snapshot["operations"].get("change")
        if wire["kind"].startswith("sink.") and pending:
            assert pending["native_event_id"] == wire["event_id"]
            assert pending["state"] == "dispatching"

    def spawn(coro):
        tasks.append(asyncio.create_task(coro))

    owner = SimpleNamespace(
        journal=journal, operation_message=receipt, spawn_control=spawn
    )
    segment = SimpleNamespace(
        owner=owner,
        segment_id=start["segment_id"],
        segment_epoch=start["segment_epoch"],
        pending_prompt=None,
        condition_switch=None,
        cancel_requested=False,
        source_view=SimpleNamespace(stopped=asyncio.Event()),
        source_send_lock=asyncio.Lock(),
        model_wire_lock=asyncio.Lock(),
        model_wire_sequence=0,
        trace_id="sink-test",
        model_ws=SimpleNamespace(send=send),
        config=SimpleNamespace(startup_seconds=1),
        prompt_available=asyncio.Event(),
    )

    async def wire(ws, kind, payload, **kwargs):
        return await InteractiveSession.send_model_event(
            segment, ws, kind, payload, **kwargs
        )

    segment.send_model_event = wire
    sink = segment.sink = SinkControl(segment)
    sink.supported = True
    await sink.output(
        dict(sink_chunks=0, sink_chunk_ids=[], sink_event_id=None, output_frame_start=0)
    )
    scope = {k: start[k] for k in ("segment_id", "segment_epoch")}
    return sink, owner, scope, sent, receipts, tasks


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


def output(event, enabled, start=17, ids=None):
    return dict(
        sink_chunks=2 if enabled else 0,
        sink_chunk_ids=([2, 3] if enabled else []) if ids is None else ids,
        sink_event_id=event,
        output_frame_start=start,
    )


def test_sink_same_wire_ack_is_not_applied_and_duplicate_does_not_refresh():
    async def run():
        sink, owner, scope, sent, receipts, tasks = await fixture()
        cmd = command("model.sink", "change", **scope, action="on")
        await sink.command(cmd)
        assert receipts[-1]["state"] == "accepted"
        await until(lambda: len(sent) == 1)
        event = sent[-1]["event_id"]
        assert sent[-1]["kind"] == "sink.refresh"
        assert await sink.control(
            dict(
                type="control_ack",
                kind="sink.refresh",
                event_id=event,
                stage="accepted",
            )
        )
        assert owner.journal.snapshot["operations"]["change"]["applied"] is False
        # Heartbeats/source continue on exactly the original wire while waiting.
        await sink.segment.send_model_event(sink.segment.model_ws, "heartbeat", {})
        assert sent[-1]["event_id"] == event + 1
        await sink.command(cmd)
        assert receipts[-1]["duplicate"] and len(sent) == 2
        with pytest.raises(RenderError, match="request ID"):
            await sink.command({**cmd, "action": "off"})
        with pytest.raises(RenderError) as err:
            await sink.command({**cmd, "request_id": "overlap", "action": "off"})
        assert err.value.code == "SINK_UPDATE_BUSY"
        await sink.output(output(None, False, start=5))
        assert sink.busy
        await sink.output(output(event, True))
        assert (
            receipts[-1]["state"] == "completed"
            and receipts[-1]["first_output_frame"] == 17
        )
        await sink.command({**cmd, "request_id": "on-again"})
        assert receipts[-1]["noop"] and len(sent) == 2
        await asyncio.gather(*tasks)

    asyncio.run(run())


@pytest.mark.parametrize("action,expected", [("off", False), ("refresh", True)])
def test_sink_off_and_refresh_are_distinct_current_segment_operations(action, expected):
    async def run():
        sink, owner, scope, sent, receipts, tasks = await fixture()
        await sink.output(output(None, True, ids=[0, 1]))
        await sink.command(command("model.sink", "change", **scope, action=action))
        await until(lambda: bool(sent))
        assert sent[0]["kind"] == "sink." + action
        await sink.output(output(sent[0]["event_id"], expected))
        assert owner.journal.snapshot["segment"]["sink"]["enabled"] is expected
        assert not sink.busy
        await asyncio.gather(*tasks)

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("source_epoch", "old", "SOURCE_EPOCH_MISMATCH"),
        ("segment_epoch", "old", "SEGMENT_EPOCH_MISMATCH"),
        ("action", "toggle", "INVALID_COMMAND"),
        ("action", [], "INVALID_COMMAND"),
        ("worker_id", "another-player", "INVALID_COMMAND"),
    ],
)
def test_sink_rejects_stale_or_unscoped_commands(field, value, code):
    async def run():
        sink, owner, scope, sent, receipts, tasks = await fixture()
        cmd = command("model.sink", "bad", **scope, action="on")
        with pytest.raises(RenderError) as err:
            await sink.command({**cmd, field: value})
        assert err.value.code == code
        assert (
            not sent and not tasks and "bad" not in owner.journal.snapshot["operations"]
        )

    asyncio.run(run())


def test_sink_timeout_late_output_and_end_never_claim_unobserved_success():
    async def run():
        sink, owner, scope, sent, receipts, tasks = await fixture()
        sink.segment.config.startup_seconds = 0.01
        await sink.command(command("model.sink", "change", **scope, action="refresh"))
        await until(lambda: bool(tasks))
        await tasks[0]
        assert sink.busy and receipts[-1]["state"] == "reconciling"
        await sink.output(output(sent[0]["event_id"], True))
        assert receipts[-1]["state"] == "completed" and "code" not in receipts[-1]
        await sink.command(command("model.sink", "next", **scope, action="off"))
        await until(lambda: len(sent) == 2)
        await sink.finish()
        assert receipts[-1]["state"] == "reconciling" and not receipts[-1]["applied"]
        await asyncio.gather(*tasks)

    asyncio.run(run())


def test_sink_cancel_before_dispatch_and_unsupported_worker():
    async def run():
        sink, owner, scope, sent, receipts, tasks = await fixture()
        sink.supported = False
        with pytest.raises(RenderError) as err:
            await sink.command(
                command("model.sink", "old-worker", **scope, action="on")
            )
        assert err.value.code == "CAPABILITY_UNAVAILABLE"
        sink.supported = True
        await sink.command(command("model.sink", "change", **scope, action="on"))
        sink.segment.source_view.stopped.set()
        await asyncio.gather(*tasks)
        assert (
            not sent
            and receipts[-1]["state"] == "cancelled"
            and receipts[-1]["never_applied"]
        )

    asyncio.run(run())


def test_sink_prompt_rebuild_updates_anchor_without_replaying_user_request():
    async def run():
        sink, owner, scope, sent, receipts, tasks = await fixture()
        await sink.output(output(None, True, ids=[0, 1]))
        await sink.output(output(None, True, ids=[8, 9]))
        assert owner.journal.snapshot["segment"]["sink"]["sink_chunk_ids"] == [8, 9]
        assert not tasks and not sent
        sink.segment.condition_switch = {}
        with pytest.raises(RenderError) as err:
            await sink.command(
                command("model.sink", "prompt-race", **scope, action="off")
            )
        assert err.value.code == "SINK_UPDATE_BUSY"

    asyncio.run(run())


def test_real_source_segment_routes_sink_and_input_without_waiting_for_model(
    tmp_path, monkeypatch
):
    from test_whitebox_render_lifecycle import (
        AsyncArchiveManager,
        BrowserUI,
        Engine,
        OrderedWorker,
        SourceSession,
        StrictLeases,
        VideoSpec,
        WhiteboxConfig,
    )

    original_send, original_emit = OrderedWorker.send, OrderedWorker.emit

    async def emit(worker, message):
        if message["type"] == "session.ready":
            message["sink_updates"] = True
        if message["type"] in {"frame_batch", "frame_batch_header"}:
            enabled = getattr(worker, "sink_enabled", False)
            message.update(
                sink_chunks=2 if enabled else 0,
                sink_chunk_ids=[0, 1] if enabled else [],
                sink_event_id=getattr(worker, "sink_event", None),
            )
        await original_emit(worker, message)

    async def send(worker, wire):
        message = msgspec.msgpack.decode(wire)
        if message.get("kind") in {"sink.refresh", "sink.off"}:
            worker.sink_event = message["event_id"]
            worker.sink_enabled = message["kind"] == "sink.refresh"
            worker.events.append(message)
            await worker.emit(
                dict(
                    type="control_ack",
                    kind=message["kind"],
                    event_id=worker.sink_event,
                    stage="accepted",
                )
            )
            return
        await original_send(worker, wire)

    monkeypatch.setattr(OrderedWorker, "emit", emit)
    monkeypatch.setattr(OrderedWorker, "send", send)

    async def run():
        spec = VideoSpec(32, 32)
        engine, leases = Engine(spec), StrictLeases(spec, "clip-barrier-v1")
        engine.stream_id = "view"
        archives = AsyncArchiveManager(recorder_slots=3)
        journal = await setup()

        async def factory(**_):
            return engine

        source = SourceSession(
            SimpleNamespace(
                runtime=SimpleNamespace(players={}),
                make_segment_leases=lambda *_: leases,
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
                value = msgspec.msgpack.decode(await source.recv())
                messages.append(value)
                if value.get("type") in {"whitebox.frame", "frame_batch"}:
                    source.media_delivered()

        async def produce():
            while True:
                await engine.push()
                await asyncio.sleep(0.01)

        tasks = [asyncio.create_task(consume()), asyncio.create_task(produce())]
        try:
            await source.send(
                msgspec.msgpack.encode(
                    dict(type="init", whitebox={"world_id": "world"}, prompt="scene")
                )
            )
            await asyncio.wait_for(source.ready.wait(), 5)
            await source.send(
                msgspec.msgpack.encode(
                    dict(
                        type="command",
                        **command(
                            "render.start",
                            "start-live",
                            expected_render_revision=0,
                            prompt="scene",
                        ),
                    )
                )
            )
            # Sink metadata can arrive before the first decodable frame. The
            # command contract admits updates only after streaming begins.
            async with asyncio.timeout(10):
                while (
                    source.current_segment is None
                    or source.current_segment.sink.snapshot is None
                    or (journal.snapshot.get("segment") or {}).get("state")
                    != "streaming"
                ):
                    await asyncio.sleep(0.01)
            segment = source.current_segment
            assert segment.sink.snapshot["enabled"] is False
            scope = dict(
                segment_id=segment.segment_id, segment_epoch=segment.segment_epoch
            )
            for i, action in enumerate(("on", "refresh", "off"), 1):
                cmd = dict(
                    type="command",
                    **command("model.sink", f"sink-{i}", **scope, action=action),
                )
                await source.send(msgspec.msgpack.encode(cmd))
                await source.send(
                    msgspec.msgpack.encode(
                        dict(
                            type="event",
                            kind="input",
                            sequence=i - 1,
                            payload={"keys": ["w"] if i < 3 else []},
                        )
                    )
                )
                assert len(engine.inputs) == i
                async with asyncio.timeout(5):
                    while (
                        journal.snapshot["operations"].get(f"sink-{i}", {}).get("state")
                        != "completed"
                    ):
                        await asyncio.sleep(0.01)
                assert segment.sink.snapshot["enabled"] is (action != "off")
            assert leases.active == 1 and len(leases.opens) == 1
            assert source.current_segment is segment and not engine.closed
            await source.close()
            assert leases.active == 0
        finally:
            await source.close()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await archives.close()

    asyncio.run(run())


def test_sink_cancel_during_dispatch_journal_commit_cannot_send_after_stop():
    async def run():
        sink, owner, scope, sent, receipts, tasks = await fixture()
        original = owner.journal.update
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(change, **fields):
            if (
                sink.pending
                and sink.pending["event_id"] is not None
                and not entered.is_set()
            ):
                entered.set()
                await release.wait()
            return await original(change, **fields)

        owner.journal.update = delayed
        await sink.command(command("model.sink", "change", **scope, action="on"))
        await entered.wait()
        sink.segment.source_view.stopped.set()
        await sink.finish()
        release.set()
        await asyncio.gather(*tasks)
        assert not sent
        assert receipts[-1]["state"] == "reconciling"
        assert receipts[-1].get("never_applied") is not True
        assert owner.journal.snapshot["operations"]["change"]["state"] == "reconciling"

    asyncio.run(run())
