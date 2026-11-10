"""One member control socket preserves the SDK's original wire and authority."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import RenderError
from sglang.multimodal_gen.runtime.realtime.whitebox.sdk_control_relay import (
    SDKControlRelay,
)


def fixture():
    packets = []
    sub = SimpleNamespace(generation=3, closed=False, offer_control=packets.append)
    entry = {}
    descriptor = dict(
        sessionId="sdk-a", baseUrl="http://127.0.0.1:9000", accessToken="a" * 48
    )
    transport = SimpleNamespace(viewer=lambda stream, session: (entry, descriptor))
    source = SimpleNamespace(
        entry=entry,
        check=Mock(),
        record={"stream_id": "view-a"},
        sdk={"sessionId": "sdk-a"},
        runtime=SimpleNamespace(player_transport=transport),
    )
    pipeline = SimpleNamespace(
        closed=False, generation=3, subscription=sub, member_source=source
    )
    return pipeline, sub, packets


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


def test_sdk_tunnel_keeps_binding_and_input_bytes_and_closes_on_generation_change():
    async def run():
        pipeline, sub, packets = fixture()
        relay = SDKControlRelay(pipeline, sub, "client-a")
        incoming, outgoing, connects = asyncio.Queue(), [], []
        closed = asyncio.Event()

        async def send(payload):
            outgoing.append(payload)

        async def close():
            closed.set()

        async def connect(url, **kwargs):
            connects.append((url, kwargs))
            return SimpleNamespace(send=send, recv=incoming.get, close=close)

        relay.task = asyncio.create_task(relay.run(connect=connect))
        await until(lambda: bool(packets))
        assert packets == [
            dict(type="sdk.control", connection_generation=3, state="ready")
        ]
        assert (
            connects[0][0]
            == "ws://127.0.0.1:9000/v1/sessions/sdk-a/control?clientId=client-a&role=controller&ui=1&clock=0"
        )
        assert connects[0][1]["subprotocols"] == [
            "worldkit.stream.v1",
            "worldkit.cap." + "a" * 48,
        ]
        offered = '{"type":"player.binding.offer","bindingSequence":8}'
        await incoming.put(offered)
        await until(lambda: len(packets) == 2)
        assert packets[-1]["payload"] == offered
        ack = '{"type":"player.binding.ack","bindingSequence":8}'
        key = '{"type":"input.state","epoch":5,"bindingSequence":8,"sequence":9,"heldKeys":["KeyW"]}'
        relay.enqueue(ack)
        relay.enqueue(key)
        await until(lambda: len(outgoing) == 2)
        assert outgoing == [ack, key]
        # A replacement attachment cannot send on the old member/SDK lease.
        pipeline.generation = 4
        with pytest.raises(RenderError):
            relay.enqueue(key)
        await incoming.put('{"type":"input.ack","sequence":9}')
        await relay.task
        assert closed.is_set() and relay.closed
        assert len(packets) == 2  # No old ACK delivered to the replacement.

    asyncio.run(run())


def test_sdk_tunnel_does_not_substitute_another_members_descriptor():
    async def run():
        pipeline, sub, packets = fixture()
        pipeline.member_source.runtime.player_transport.viewer = lambda *_: (
            {},
            {"sessionId": "sdk-b"},
        )
        relay = SDKControlRelay(pipeline, sub, "client-a")
        connected = Mock()
        await relay.run(connect=connected)
        connected.assert_not_called()
        assert packets[-1]["state"] == "closed"

    asyncio.run(run())


def test_sdk_tunnel_has_bounded_input_queue_and_rejects_unready_inputs():
    pipeline, sub, packets = fixture()
    relay = SDKControlRelay(pipeline, sub, "client-a")
    with pytest.raises(RenderError) as error:
        relay.enqueue("{}")
    assert error.value.code == "CONTROL_UNAVAILABLE"
    relay.socket = object()
    for _ in range(64):
        relay.enqueue("{}")
    with pytest.raises(RenderError) as error:
        relay.enqueue("{}")
    assert error.value.code == "LIMIT_EXCEEDED" and relay.closed
    assert relay.queued_bytes == 128


def test_real_sdk_socket_preserves_input_lease_packet_and_member_credentials():
    from websockets.asyncio.server import serve

    async def run():
        pipeline, sub, packets = fixture()
        seen = []

        async def sdk(socket):
            seen.append(
                (socket.request.path, socket.request.headers["Sec-WebSocket-Protocol"])
            )
            await socket.send('{"type":"binding.offer","bindingSequence":13}')
            async for payload in socket:
                seen.append(payload)
                await socket.send(payload)

        async with serve(
            sdk, "127.0.0.1", 0, subprotocols=["worldkit.stream.v1"]
        ) as server:
            port = server.sockets[0].getsockname()[1]
            descriptor = dict(
                sessionId="sdk-a",
                baseUrl=f"http://127.0.0.1:{port}",
                accessToken="a" * 48,
            )
            pipeline.member_source.runtime.player_transport.viewer = lambda *_: (
                pipeline.member_source.entry,
                descriptor,
            )
            relay = SDKControlRelay(pipeline, sub, "client-a")
            relay.task = asyncio.create_task(relay.run())
            await until(lambda: len(packets) >= 2)
            raw = '{"type":"input.state","bindingSequence":13,"sequence":21,"heldKeys":["KeyW"]}'
            relay.enqueue(raw)
            await until(lambda: len(packets) >= 3)
            assert seen[1] == raw and packets[2]["payload"] == raw
            assert "/sdk-a/control?" in seen[0][0]
            assert "worldkit.cap." + "a" * 48 in seen[0][1]
            assert not any(
                "accessToken" in str(p) or "a" * 48 in str(p) for p in packets
            )
            relay.stop()
            await asyncio.gather(relay.task, return_exceptions=True)
            assert relay.closed and relay.socket.close_code is not None

    asyncio.run(run())


def test_sdk_tunnel_expires_gameplay_behind_a_live_stalled_socket_without_replaying():
    async def run():
        pipeline, sub, packets = fixture()
        now = [0.0]
        relay = SDKControlRelay(pipeline, sub, "client-a", clock=lambda: now[0])
        incoming, sent = asyncio.Queue(), []
        gate = asyncio.Event()

        async def send(payload):
            if payload == "{}":
                await gate.wait()
            sent.append(payload)

        async def close():
            pass

        async def connect(*args, **kwargs):
            return SimpleNamespace(send=send, recv=incoming.get, close=close)

        relay.task = asyncio.create_task(relay.run(connect=connect))
        await until(lambda: bool(packets))
        relay.enqueue("{}")
        await asyncio.sleep(0)
        stale = '{"type":"input.semantic.batch","epoch":1,"generation":2,"batch":{"sequence":1}}'
        relay.enqueue(stale)
        now[0] = 0.201
        gate.set()
        await relay.task
        assert sent == ["{}"]
        assert relay.closed and relay.queued_bytes == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    "blocked,gameplay", [("send", True), ("archive", True), ("send", False)]
)
def test_sdk_tunnel_retires_queued_gameplay_without_releasing_blocked_handshake(
    blocked, gameplay
):
    async def run():
        pipeline, sub, packets = fixture()
        relay = SDKControlRelay(pipeline, sub, "client-a")
        started, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        incoming, sent = asyncio.Queue(), []

        async def block():
            started.set()
            await release.wait()

        async def send(payload):
            if blocked == "send":
                await block()
            sent.append(payload)

        async def observe(*_):
            if blocked == "archive":
                await block()

        async def close():
            closed.set()

        async def connect(*_, **__):
            return SimpleNamespace(send=send, recv=incoming.get, close=close)

        pipeline.member_source.runtime.observe_player_control = observe
        relay.task = asyncio.create_task(relay.run(connect=connect))
        try:
            await until(lambda: bool(packets))
            relay.enqueue('{"type":"input.semantic.applied"}')
            await asyncio.wait_for(started.wait(), 1)
            batch = '{"type":"input.semantic.batch","generation":1,"batch":{"sequence":1,"stateRevision":1}}'
            if gameplay:
                relay.enqueue(batch)
                await asyncio.wait_for(asyncio.shield(relay.task), 1)
                assert closed.is_set() and relay.closed
                assert not release.is_set() and batch not in sent
                assert relay.queue.empty() and relay.queued_bytes == 0
            else:
                await asyncio.sleep(0.25)
                assert not relay.closed and not closed.is_set()
                release.set()
                await until(lambda: bool(sent))
        finally:
            relay.stop()
            await asyncio.gather(relay.task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("send_fails", [False, True])
def test_sdk_tunnel_correlates_fast_receipts_but_failed_sends_are_not_activity(
    send_fails,
):
    import json

    from sglang.multimodal_gen.runtime.realtime.whitebox.player_idle import (
        PlayerInputActivity,
    )

    async def run():
        pipeline, sub, packets = fixture()
        activity = PlayerInputActivity()
        activity.start(0)
        incoming, forwarded = asyncio.Queue(), asyncio.Event()

        async def observe(stream, client, direction, payload):
            activity.observe(client, direction, json.loads(payload), 10)

        pipeline.member_source.runtime.observe_player_control = observe

        def offer(packet):
            packets.append(packet)
            if packet.get("payload"):
                forwarded.set()

        sub.offer_control = offer

        async def send(payload):
            if send_fails:
                raise ConnectionError("fixture send failed")
            incoming.put_nowait(
                json.dumps(
                    dict(
                        type="input.semantic.receipt",
                        clientId="client-a",
                        generation=1,
                        sequence=1,
                        stateRevision=1,
                        simulationTick=1,
                        meaningful=True,
                    )
                )
            )
            await forwarded.wait()

        async def close():
            pass

        async def connect(*args, **kwargs):
            return SimpleNamespace(send=send, recv=incoming.get, close=close)

        relay = SDKControlRelay(pipeline, sub, "client-a")
        relay.task = asyncio.create_task(relay.run(connect=connect))
        try:
            await until(lambda: bool(packets))
            relay.enqueue(
                json.dumps(
                    dict(
                        type="input.semantic.batch",
                        generation=1,
                        batch=dict(sequence=1, stateRevision=1),
                    )
                )
            )
            if send_fails:
                await asyncio.wait_for(relay.task, 1)
                assert activity.last_input == 0
            else:
                await asyncio.wait_for(forwarded.wait(), 1)
                assert activity.last_input == 10
                assert not activity.pending
        finally:
            relay.stop()
            await asyncio.gather(relay.task, return_exceptions=True)

    asyncio.run(run())
