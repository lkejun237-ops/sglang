"""Player instructions use the current trusted CLI consumer, never a model writer."""

import asyncio
import json
import time
from types import SimpleNamespace

import msgspec
import pytest
from test_whitebox_source_conditions import setup

from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession
from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
    CLIConditionChannel,
)


async def owner(tmp_path):
    _, condition, journal, _, _, _ = await setup(tmp_path)
    ids = (await journal.read())["model_context_segment_ids"]
    source = object.__new__(SourceSession)
    source.journal, source.world_driven = journal, True
    source.closed = source.binding_ended = False
    source.session_id, source.source_epoch = "source", "epoch"
    source.outgoing = asyncio.Queue()
    source.control_tokens, source.control_token_time = 4.0, time.monotonic()
    source.engine = SimpleNamespace(stream_id="view")
    sent = asyncio.Queue()

    class Socket:
        async def send(self, wire):
            await sent.put(json.loads(wire))

    view = SimpleNamespace(
        descriptor=dict(
            sessionId="session",
            epoch=1,
            modelContext=dict(
                protocol=condition["protocol"],
                required=True,
                configHash=condition["configHash"],
                gameBuildHash=condition["source"]["gameBuildHash"],
                compilerVersion=1,
                visible=False,
            ),
        ),
        client_id="consumer",
        control=Socket(),
        closed=False,
        stopped=False,
    )
    channel = CLIConditionChannel(view)
    channel.segment = ids["segment_id"]
    channel.consumer = condition["audience"]
    channel.bound.set()
    source.context_channel = channel
    command = dict(
        type="command",
        op="model.context.intervention",
        request_id="player-text",
        source_session_id="source",
        source_epoch="epoch",
        segment_id=ids["segment_id"],
        segment_epoch=ids["segment_epoch"],
        expectedRevision=0,
        prompt="Make the rain cinematic",
    )
    return source, channel, command, sent


def response(request, **fields):
    return {
        k: request[k] for k in ("epoch", "consumerEpoch", "segmentId", "requestId")
    } | dict(
        type="model.context.intervention.result",
        status="accepted",
        revision=1,
        prompt="Make the rain cinematic",
        **fields,
    )


@pytest.mark.parametrize(
    "status", ["accepted", "duplicate", "conflict", "rejected", "current"]
)
def test_instruction_routes_current_cli_scope_and_preserves_browser_identity(
    tmp_path, status
):
    async def scenario():
        source, channel, command, sent = await owner(tmp_path)
        if status == "current":
            command = {
                k: v
                for k, v in command.items()
                if k not in {"prompt", "expectedRevision"}
            }
            command["op"] += ".get"
        before = await source.journal.read()
        task = asyncio.create_task(source.command(command))
        async with asyncio.timeout(1):
            request = await sent.get()
        assert request == dict(
            type=command["op"],
            epoch=1,
            consumerEpoch=channel.consumer["consumerEpoch"],
            segmentId=command["segment_id"],
            requestId="player-text",
            **(
                {}
                if status == "current"
                else dict(expectedRevision=0, prompt="Make the rain cinematic")
            ),
        )
        result = response(request)
        result["status"] = status
        await channel.consume(result)
        await task
        browser = msgspec.msgpack.decode(await source.outgoing.get())
        assert browser == dict(
            type="model.context.intervention.result",
            request_id="player-text",
            source_session_id="source",
            source_id="source",
            source_epoch="epoch",
            view_id="view",
            segment_id=command["segment_id"],
            segment_epoch=command["segment_epoch"],
            status=status,
            revision=1,
            prompt="Make the rain cinematic",
        )
        assert await source.journal.read() == before
        await channel.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_epoch", "old"),
        ("segment_id", "old"),
        ("segment_epoch", "old"),
        ("consumerEpoch", "forged"),
        ("requestId", "forged"),
        ("references", []),
    ],
)
def test_instruction_rejects_stale_or_untrusted_browser_scope(tmp_path, field, value):
    async def scenario():
        source, channel, command, sent = await owner(tmp_path)
        command[field] = value
        await source.command(command)
        assert sent.empty()
        error = msgspec.msgpack.decode(await source.outgoing.get())
        assert error["type"] == "operation.rejected"
        await channel.close()

    asyncio.run(scenario())


def test_instruction_timeout_never_reissues_write_and_read_recovers(
    tmp_path, monkeypatch
):
    from sglang.multimodal_gen.runtime.realtime.whitebox import source_condition_channel

    monkeypatch.setattr(
        source_condition_channel, "INTERVENTION_WAIT_SECONDS", 0.01, raising=False
    )

    async def scenario():
        source, channel, command, sent = await owner(tmp_path)
        await source.command(command)
        assert (await sent.get())["type"] == "model.context.intervention"
        error = msgspec.msgpack.decode(await source.outgoing.get())
        assert error["code"] == "CONTROL_UNAVAILABLE" and sent.empty()
        assert not channel.interventions
        # A late answer cannot affect a subsequent read with a different ID.
        late = dict(
            type="model.context.intervention.result",
            epoch=1,
            consumerEpoch=channel.consumer["consumerEpoch"],
            segmentId=channel.segment,
            requestId="player-text",
            status="accepted",
            revision=1,
            prompt=command["prompt"],
        )
        await channel.consume(late)
        read = {
            k: v for k, v in command.items() if k not in {"prompt", "expectedRevision"}
        }
        read.update(op="model.context.intervention.get", request_id="recover")
        task = asyncio.create_task(source.command(read))
        request = await sent.get()
        result = response(request)
        result["status"] = "current"
        await channel.consume(result)
        await task
        assert (
            msgspec.msgpack.decode(await source.outgoing.get())["status"] == "current"
        )
        await channel.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change", ["epoch", "consumerEpoch", "segmentId", "revision", "prompt", "extra"]
)
def test_bad_cli_result_fails_only_its_instruction(tmp_path, change):
    async def scenario():
        source, channel, command, sent = await owner(tmp_path)
        task = asyncio.create_task(source.command(command))
        request = await sent.get()
        result = response(request)
        result[change] = (
            True if change == "revision" else ([] if change == "prompt" else "invalid")
        )
        channel.offer(result)
        await task
        assert (
            msgspec.msgpack.decode(await source.outgoing.get())["code"]
            == "CONTROL_UNAVAILABLE"
        )
        assert (
            not channel.suspended
            and not channel.interventions
            and channel.sequence == 0
        )
        await channel.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("end", ["close", "reconnect", "suspend"])
def test_instruction_disconnect_and_cleanup_do_not_replay_write(tmp_path, end):
    async def scenario():
        source, channel, command, sent = await owner(tmp_path)
        task = asyncio.create_task(source.command(command))
        assert (await sent.get())["requestId"] == "player-text"
        if end == "close":
            await channel.close()
        elif end == "reconnect":
            await channel.reconnect()
        else:
            await channel.suspend("MODEL_CONTEXT_PROTOCOL_INVALID")
        await task
        assert (
            msgspec.msgpack.decode(await source.outgoing.get())["code"]
            == "CONTROL_UNAVAILABLE"
        )
        assert not channel.interventions
        assert all(
            row["type"] != "model.context.intervention" for row in list(sent._queue)
        )
        await channel.close()

    asyncio.run(scenario())


def test_replaced_room_engine_cannot_relay_instruction(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        LocalRoomEngine,
    )

    async def scenario():
        source, channel, command, sent = await owner(tmp_path)
        engine = object.__new__(LocalRoomEngine)
        engine.closed, engine.stream_id = False, "view"
        engine._context_engine = channel.view
        engine.runtime = SimpleNamespace(player=lambda _: {"engine": object()})
        source.engine = engine
        await source.command(command)
        assert sent.empty()
        assert (
            msgspec.msgpack.decode(await source.outgoing.get())["code"]
            == "SOURCE_EPOCH_MISMATCH"
        )
        await channel.close()

    asyncio.run(scenario())


def test_member_instruction_passes_ordered_admission_without_allowing_second_writer(
    tmp_path,
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderPipeline,
    )

    async def scenario():
        source, channel, command, sent = await owner(tmp_path)
        pipeline = object.__new__(RenderPipeline)
        pipeline.session, pipeline.generation = source, 1
        pipeline.subscription = SimpleNamespace(closed=False, generation=1)
        pipeline.member_source = SimpleNamespace(check=lambda: None)
        pipeline.command_queue = asyncio.Queue(maxsize=32)
        pipeline.command_bytes, pipeline.command_byte_limit = 0, 1 << 20
        for op in ("prompt.replace", "render.start", "render.replace"):
            with pytest.raises(ValueError, match="SDK player owns"):
                pipeline.accept(
                    pipeline.subscription, msgspec.msgpack.encode(dict(command, op=op))
                )
        pipeline.accept(pipeline.subscription, msgspec.msgpack.encode(command))
        worker = asyncio.create_task(pipeline._commands())
        try:
            async with asyncio.timeout(1):
                request = await sent.get()
                channel.offer(response(request))
                while source.outgoing.empty():
                    await asyncio.sleep(0)
            assert (
                msgspec.msgpack.decode(await source.outgoing.get())["status"]
                == "accepted"
            )
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await channel.close()

    asyncio.run(scenario())


def test_reattached_ready_uses_current_reserved_context_segment(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderPipeline,
    )

    async def scenario():
        source, channel, command, _ = await owner(tmp_path)
        source.config = SimpleNamespace(play_seconds=300)
        source.play_elapsed = 0
        source.revision, source.display_mode = 0, "whitebox"
        source.current_model_stream = None
        pipeline = object.__new__(RenderPipeline)
        pipeline.init = {"whitebox": {}}
        pipeline.session, pipeline.generation = source, 2
        pipeline.ready = dict(
            type="session.ready",
            model_context_segment=dict(segment_id="old", segment_epoch="old"),
            source_video_stream={},
            video_stream={},
        )
        pipeline.member_source = object()
        pipeline.first_media_at, pipeline.committed_input = None, -1
        pipeline.stream_id = "view"
        pipeline.host = SimpleNamespace(
            config=source.config,
            grace_seconds=15,
            runtime=SimpleNamespace(epoch="owner", owner_id="local"),
        )
        expected = dict(
            segment_id=command["segment_id"], segment_epoch=command["segment_epoch"]
        )
        assert pipeline.ready_message()["model_context_segment"] == expected
        source.source_state_value = "ready"
        source.wait_elapsed = 0
        await source.status_snapshot()
        assert (
            msgspec.msgpack.decode(await source.outgoing.get())["model_context_segment"]
            == expected
        )
        await channel.close()

    asyncio.run(scenario())


def test_ready_instruction_read_waits_for_initial_cli_subscription(tmp_path):
    async def scenario():
        source, channel, command, sent = await owner(tmp_path)
        channel.bound.clear()
        consumer, channel.consumer = channel.consumer, None
        command = {
            k: v for k, v in command.items() if k not in {"prompt", "expectedRevision"}
        }
        command["op"] += ".get"
        task = asyncio.create_task(source.command(command))
        await asyncio.sleep(0)
        assert source.outgoing.empty() and sent.empty()
        channel.consumer = consumer
        channel.bound.set()
        request = await sent.get()
        result = response(request)
        result.update(status="current", prompt=None, revision=0)
        channel.offer(result)
        await task
        assert (
            msgspec.msgpack.decode(await source.outgoing.get())["status"] == "current"
        )
        await channel.close()

    asyncio.run(scenario())


def test_close_releases_instruction_waiting_for_first_subscription(tmp_path):
    async def scenario():
        source, channel, command, sent = await owner(tmp_path)
        channel.bound.clear()
        channel.consumer = None
        task = asyncio.create_task(source.command(command))
        await asyncio.sleep(0)
        await channel.close()
        async with asyncio.timeout(1):
            await task
        assert sent.empty() and not channel.interventions
        assert (
            msgspec.msgpack.decode(await source.outgoing.get())["code"]
            == "CONTROL_UNAVAILABLE"
        )

    asyncio.run(scenario())


def test_retired_cli_view_cannot_complete_browser_instruction(tmp_path):
    async def scenario():
        source, channel, command, sent = await owner(tmp_path)
        task = asyncio.create_task(source.command(command))
        request = await sent.get()
        channel.view.closed = True
        channel.offer(response(request))
        await task
        result = msgspec.msgpack.decode(await source.outgoing.get())
        assert (
            result["type"] == "operation.rejected"
            and result["code"] == "CONTROL_UNAVAILABLE"
        )
        await channel.close()

    asyncio.run(scenario())


def test_first_native_epoch_zero_accepts_scoped_instruction_result(tmp_path):
    async def scenario():
        source, channel, command, sent = await owner(tmp_path)
        channel.view.descriptor["epoch"] = 0
        task = asyncio.create_task(source.command(command))
        request = await sent.get()
        assert request["epoch"] == 0
        channel.offer(response(request))
        await task
        result = msgspec.msgpack.decode(await source.outgoing.get())
        assert (
            result["type"] == "model.context.intervention.result"
            and result["status"] == "accepted"
        )
        await channel.close()

    asyncio.run(scenario())
