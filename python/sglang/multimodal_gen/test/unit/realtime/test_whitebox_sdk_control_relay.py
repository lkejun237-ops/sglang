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
