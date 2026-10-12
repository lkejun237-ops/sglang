# SPDX-License-Identifier: Apache-2.0
"""Real bounded ASGI transport diagnostic; no model or business-load claim."""

import asyncio
import socket

import uvicorn
import websockets


def test_bidirectional_backpressure_times_out_and_next_connection_is_usable():
    async def scenario():
        active_sessions = set()
        finished = asyncio.Event()

        async def app(scope, receive, send):
            await receive()
            await send({"type": "websocket.accept"})
            identity = id(scope)
            active_sessions.add(identity)
            lock = asyncio.Lock()

            async def output():
                while True:
                    async with lock:
                        await send(
                            {"type": "websocket.send", "bytes": b"x" * (2 << 20)}
                        )

            async def inputs():
                while True:
                    message = await receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    async with lock:
                        await send({"type": "websocket.send", "bytes": b"ack"})

            tasks = [asyncio.create_task(output()), asyncio.create_task(inputs())]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                active_sessions.remove(identity)
                finished.set()

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                ws="websockets",
                ws_max_queue=1,
                ws_max_size=8192,
                ws_per_message_deflate=False,
                # Scale the transport clock only for this component regression.
                ws_ping_interval=0.1,
                ws_ping_timeout=0.2,
                lifespan="off",
                log_level="error",
            )
        )
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                await asyncio.sleep(0.01)
            url = f"ws://127.0.0.1:{sock.getsockname()[1]}"
            async with websockets.connect(
                url,
                max_queue=1,
                max_size=4 << 20,
                compression=None,
                ping_interval=0.1,
                ping_timeout=0.2,
                close_timeout=0.1,
            ) as ws:
                for _ in range(68):
                    await ws.send(b"source-control" * 4)
                await ws.recv()
                # The consumer intentionally stops draining the bounded queue.
                protocol = getattr(ws, "protocol", ws)
                async with asyncio.timeout(5):
                    while protocol.close_sent is None:
                        await asyncio.sleep(0.01)
                assert protocol.close_sent.code == 1011
                assert protocol.close_sent.reason == "keepalive ping timeout"
                connections = list(server.server_state.connections)
                assert len(connections) == 1
                assert connections[0].max_queue == 1
                assert len(connections[0].messages) <= 1
                ws.transport.abort()
            await asyncio.wait_for(finished.wait(), 3)
            assert not active_sessions
            finished.clear()
            async with websockets.connect(
                url, max_size=4 << 20, compression=None
            ) as ws:
                assert len(await ws.recv()) == 2 << 20
            await asyncio.wait_for(finished.wait(), 3)
            assert not active_sessions
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
