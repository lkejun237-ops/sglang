"""Real handshakes: RTC control compresses; H.264 WS and limits stay unchanged."""

import asyncio
import socket

import pytest
import uvicorn
from fastapi import FastAPI, WebSocket
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError

from sglang.multimodal_gen.runtime.realtime.whitebox.websocket_protocol import (
    WhiteboxWebSocketProtocol,
)


def test_rtc_only_deflate_and_decompressed_message_limit():
    async def run():
        app = FastAPI()

        @app.websocket("/{path:path}")
        async def echo(ws: WebSocket, path: str):
            await ws.accept()
            try:
                while True:
                    await ws.send_bytes(await ws.receive_bytes())
            except Exception:
                pass

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(16)
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                ws=WhiteboxWebSocketProtocol,
                ws_per_message_deflate=False,
                ws_max_size=4096,
                log_level="error",
                lifespan="off",
            )
        )
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                await asyncio.sleep(0.01)
            root = "ws://127.0.0.1:" + str(sock.getsockname()[1])
            for path, compressed in [
                ("/v1/webrtc/play?protocol=native", True),
                ("/v1/realtime_video/generate", False),
                ("/v1/webrtc/play/other", False),
            ]:
                async with connect(root + path, proxy=None) as ws:
                    extension = ws.response.headers.get("Sec-WebSocket-Extensions", "")
                    assert ("permessage-deflate" in extension) == compressed
                    if compressed:
                        assert "server_no_context_takeover" not in extension
                        assert "server_max_window_bits=12" in extension
                    payload = b"frame-identity-and-ui-" * 100
                    await ws.send(payload)
                    assert await ws.recv() == payload
                    await ws.send(payload + b"next")
                    assert await ws.recv() == payload + b"next"
            async with connect(
                root + "/v1/webrtc/play", compression=None, proxy=None
            ) as ws:
                assert not ws.response.headers.get("Sec-WebSocket-Extensions")
                await ws.send(b"plain")
                assert await ws.recv() == b"plain"
            async with connect(root + "/v1/webrtc/play", proxy=None) as ws:
                await ws.send(b"x" * 10000)
                with pytest.raises(ConnectionClosedError) as error:
                    await ws.recv()
                assert error.value.rcvd.code == 1009
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 5)
            sock.close()

    asyncio.run(run())
