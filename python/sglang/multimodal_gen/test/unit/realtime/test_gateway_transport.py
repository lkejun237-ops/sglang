# SPDX-License-Identifier: Apache-2.0

import asyncio
import socket

import pytest
import uvicorn
from sglang.multimodal_gen.runtime.realtime.gateway_transport import (
    bounded_websocket_protocol,
)
from websockets.asyncio.client import connect


def test_bounded_protocol_keeps_websocket_wire_compatible():
    async def app(scope, receive, send):
        assert scope["type"] == "websocket"
        assert (await receive())["type"] == "websocket.connect"
        await send({"type": "websocket.accept"})
        while True:
            event = await receive()
            if event["type"] == "websocket.disconnect":
                return
            await send({"type": "websocket.send", "bytes": event["bytes"]})

    async def run():
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            lifespan="off",
            log_level="error",
            ws=bounded_websocket_protocol(131072),
        )
        server = uvicorn.Server(config)
        running = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            while not server.started:
                await asyncio.sleep(0.01)
            async with connect(f"ws://127.0.0.1:{port}", compression=None) as client:
                protocol = next(iter(server.server_state.connections))
                assert protocol.transport.get_write_buffer_limits() == (16384, 65536)
                actual = protocol.transport.get_extra_info("socket").getsockopt(
                    socket.SOL_SOCKET, socket.SO_SNDBUF
                )
                assert 131072 <= actual <= 262144
                for sequence in range(8):
                    payload = bytes([sequence]) * 65536
                    await client.send(payload)
                    assert await asyncio.wait_for(client.recv(), 2) == payload
        finally:
            server.should_exit = True
            await asyncio.wait_for(running, 3)
            listener.close()

    asyncio.run(run())


def test_bounded_protocol_rejects_an_unusable_buffer():
    with pytest.raises(ValueError, match="at least"):
        bounded_websocket_protocol(1)
