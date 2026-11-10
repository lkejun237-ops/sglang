"""Live input never waits for archive I/O; the SDK remains the receipt authority."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.player_idle import (
    PlayerInputActivity,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.player_transport import relay
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import (
    RoomLog,
    RoomRuntime,
)


async def live_log(tmp_path, monkeypatch, processes):
    captured = {}

    class LogCaptured(Exception):
        pass

    async def initialize(log):
        captured["log"] = log
        raise LogCaptured

    async def touch(*args):
        pass

    monkeypatch.setattr(RoomLog, "initialize", initialize)
    runtime = RoomRuntime(
        directory=SimpleNamespace(touch=touch),
        owner_id="owner",
        driver_factory=None,
        spool=tmp_path,
        archives=SimpleNamespace(
            processes=processes, recorder_cpu=None, fail_open=False, diskless=False
        ),
        spec=VideoSpec(32, 32),
    )
    with pytest.raises(LogCaptured):
        await runtime._start_room(
            "1" * 32,
            {"game_id": "1" * 32, "world_id": "world"},
            asyncio.Lock(),
            24,
            None,
            "trace",
        )
    return runtime, captured["log"]


@pytest.mark.parametrize("processes", [False, True])
def test_live_room_forwards_keys_and_fast_receipts_when_archive_is_full(
    tmp_path, monkeypatch, processes
):
    async def run():
        runtime, log = await live_log(tmp_path, monkeypatch, processes)
        worker = None
        original_file = log.events
        if log.journal:
            if isinstance(log.journal.queue, asyncio.Queue):
                for _ in range(128):
                    log.journal.queue.put_nowait(("full", time.monotonic()))
                worker = asyncio.create_task(asyncio.Event().wait())
                log.journal.task = worker
            else:
                for _ in range(128):
                    log.journal.queue.append((time.monotonic(), 1, "full"))
        else:

            class SlowFile:
                def write(self, value):
                    return original_file.write(value)

                def flush(self):
                    time.sleep(0.25)
                    original_file.flush()

            log.events = SlowFile()

        activity = PlayerInputActivity()
        activity.start(time.monotonic())
        entry = {
            "engine": SimpleNamespace(descriptor={"sessionId": "sdk", "epoch": 0}),
            "input_activity": activity,
            "identity": {"player_id": "player"},
            "game_id": "1" * 32,
        }
        runtime.rooms["1" * 32] = {"log": log}
        runtime.player = lambda *args, **kwargs: entry
        incoming, outgoing = asyncio.Queue(), asyncio.Queue()
        sent, receipts = [], []
        entered = time.monotonic()

        class Browser:
            async def receive(self):
                return await incoming.get()

            async def send_text(self, wire):
                receipts.append(wire)

        class SDK:
            async def send(self, wire):
                assert ("client", "input", 1) in activity.pending
                sent.append(time.monotonic() - entered)
                await outgoing.put(
                    json.dumps(
                        {
                            "type": "input.receipt",
                            "epoch": 0,
                            "clientId": "client",
                            "clientSequence": 1,
                            "simulationTick": 10,
                        }
                    )
                )

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await outgoing.get()

        async def observe(direction, wire):
            await runtime.observe_player_control("stream", "client", direction, wire)

        await incoming.put(
            {
                "type": "websocket.receive",
                "text": json.dumps(
                    {
                        "type": "input.state",
                        "epoch": 0,
                        "input": {"sequence": 1, "heldKeys": ["KeyW"], "keyEdges": []},
                    }
                ),
            }
        )
        task = asyncio.create_task(relay(Browser(), SDK(), observe=observe))
        try:
            await asyncio.sleep(0.3)
            assert sent and sent[0] < 0.2
            assert not task.done()
            assert len(receipts) == 1
            assert activity.pending == {}
            assert activity.held["client"] is True
            assert log.journal.failure == "JOURNAL_QUEUE_FULL"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if worker:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                log.journal.pool.shutdown(wait=False, cancel_futures=True)
            if original_file:
                original_file.close()

    asyncio.run(run())


def test_live_room_journal_finalizes_ordered_original_events(tmp_path, monkeypatch):
    async def run():
        initialize = RoomLog.initialize
        _, log = await live_log(tmp_path, monkeypatch, False)
        await initialize(log)
        await log.event(
            {"player_id": "player"}, "viewer_input_observed", {"sequence": 1}
        )
        await log.event(
            {"player_id": "player"}, "viewer_input_receipt", {"sequence": 1}
        )
        await asyncio.wait_for(log.finish("completed"), 15)
        events = [
            json.loads(line)
            for line in (log.directory / "inputs.jsonl").read_text().splitlines()
        ]
        assert [event["room_seq"] for event in events] == [1, 2]
        assert [event["kind"] for event in events] == [
            "viewer_input_observed",
            "viewer_input_receipt",
        ]
        assert log.manifest["recording_status"] == "complete"
        assert log.journal.task.done()
        assert log.journal.bytes == 0

    asyncio.run(run())


def test_live_room_archive_overflow_is_failed_at_finalization(tmp_path, monkeypatch):
    async def run():
        initialize = RoomLog.initialize
        _, log = await live_log(tmp_path, monkeypatch, False)
        for _ in range(128):
            log.journal.queue.append((time.monotonic(), 1, "full"))
        await log.event({"player_id": "player"}, "viewer_input_observed", {})
        await initialize(log)
        await asyncio.wait_for(log.finish("completed"), 15)
        assert log.manifest["recording_status"] == "failed"
        assert log.manifest["upload_status"] == "failed"
        assert log.manifest["archive_failure"] == "JOURNAL_QUEUE_FULL"
        assert log.journal.task.done()

    asyncio.run(run())


@pytest.mark.parametrize("diskless", [False, True])
@pytest.mark.parametrize("kind", ["batch", "binding"])
def test_legal_semantic_messages_preserve_complete_room_archive(
    tmp_path, diskless, kind
):
    async def run():
        log = RoomLog(
            tmp_path,
            {"game_id": "game", "world_id": "world"},
            "archive",
            process=True,
            fail_open=True,
            diskless=diskless,
        )
        message = {"type": "input.semantic." + kind, "epoch": 0, "generation": 1}
        if kind == "batch":
            message["batch"] = {
                "schemaVersion": 1,
                "sequence": 1,
                "stateRevision": 1,
                "state": {"text": "x" * 65400},
                "events": [],
            }
            # The native batch budget excludes its transport/archive envelopes.
            assert len(json.dumps(message["batch"], separators=(",", ":"))) <= 65536
        else:
            message.update(
                clientId="client",
                inputLeaseMs=500,
                abiVersion="player-input/1",
                controls={
                    "schemaVersion": 1,
                    "adapters": [],
                    "controls": {
                        "move": {
                            "kind": "state",
                            "schema": {"type": "number", "description": "x" * 130000},
                            "neutral": 0,
                        }
                    },
                },
            )
            assert len(json.dumps(message["controls"], separators=(",", ":"))) <= 131072
        entry = {
            "engine": SimpleNamespace(descriptor={"sessionId": "sdk", "epoch": 0}),
            "identity": {"player_id": "player"},
            "game_id": "game",
        }
        runtime = SimpleNamespace(
            player=lambda *args, **kwargs: entry, rooms={"game": {"log": log}}
        )
        await log.event(entry["identity"], "joined", {})
        await RoomRuntime.observe_player_control(
            runtime,
            "stream",
            "client",
            "to-sdk" if kind == "batch" else "from-sdk",
            json.dumps(message, separators=(",", ":")),
        )
        assert not log.failure
        assert log.journal is None or not log.journal.failure
        await log.event(entry["identity"], "finished", {})
        await log.initialize()
        await asyncio.wait_for(log.finish("completed"), 15)
        content = (
            log.archive_payload
            if diskless
            else (log.directory / "inputs.jsonl").read_bytes()
        )
        events = [json.loads(line) for line in content.splitlines()]
        assert [event["room_seq"] for event in events] == [1, 2, 3]
        assert events[1]["payload"]["message"] == message
        assert log.manifest["recording_status"] == "complete"
        assert log.manifest["objects"]["inputs.jsonl"]["bytes"] == len(content)

    asyncio.run(run())


@pytest.mark.parametrize("diskless", [False, True])
@pytest.mark.parametrize("semantic", [False, True])
def test_room_event_size_remains_bounded(tmp_path, diskless, semantic):
    async def run():
        log = RoomLog(
            tmp_path,
            {"game_id": "game", "world_id": "world"},
            "archive",
            process=True,
            fail_open=True,
            diskless=diskless,
        )
        await log.event(
            {},
            "viewer_input_observed" if semantic else "diagnostic",
            {
                "message": {
                    "type": "input.semantic.batch",
                    "data": "x" * (262144 if semantic else 65536),
                }
            },
        )
        assert log.failure if diskless else log.journal.failure

    asyncio.run(run())
