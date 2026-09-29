# SPDX-License-Identifier: Apache-2.0
"""CPU transport boundaries. These tests do not establish model acceptance."""

import asyncio
import socket
from types import SimpleNamespace

import pytest
import uvicorn
import websockets
from websockets.exceptions import ConnectionClosedError

from sglang.multimodal_gen.runtime.realtime.source_transport import (
    SourceTransportBudget,
    SourceTransportError,
    validate_source_dimensions,
)
from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
    SourceLedger,
    SourceLimits,
)


@pytest.mark.parametrize("width,height", [(832, 480), (896, 512), (1280, 736)])
def test_supported_admission(width, height):
    budget = SourceTransportBudget()
    assert budget.source_message_bytes(width, height, 17) <= budget.message_bytes
    pipeline = SimpleNamespace(
        vae_config=SimpleNamespace(
            arch_config=SimpleNamespace(spatial_compression_ratio=16)
        ),
        dit_config=SimpleNamespace(arch_config=SimpleNamespace(patch_size=(1, 2, 2))),
    )
    validate_source_dimensions(width, height, pipeline)
    with pytest.raises(SourceTransportError) as error:
        validate_source_dimensions(1280, 720, pipeline)
    assert error.value.code == "source_dimensions_unaligned"


def test_budget_exact_edges_and_concurrent_sessions():
    required = 17 * 1280 * 736 * 3 + 4096
    assert (
        SourceTransportBudget(required).source_message_bytes(1280, 736, 17) == required
    )
    with pytest.raises(SourceTransportError) as error:
        SourceTransportBudget(required - 1).source_message_bytes(1280, 736, 17)
    assert error.value.code == "source_message_too_large"
    per_session = SourceTransportBudget().session_bytes
    SourceTransportBudget(host_bytes=8 * per_session).validate_sessions(8)
    with pytest.raises(ValueError, match="host budget"):
        SourceTransportBudget(host_bytes=8 * per_session - 1).validate_sessions(8)
    for size in [None, 0, -1, True, 4096]:
        with pytest.raises(ValueError):
            SourceTransportBudget(message_bytes=size)


def test_message_and_credit_rejections_do_not_consume_credit():
    ledger = SourceLedger(
        width=1, height=1, fps=24, limits=SourceLimits(1, 4096, 1, 3, 2)
    )

    def accept(nbytes):
        return ledger.accept(
            dict(
                client_seq=ledger.sequence,
                source_frame_start=ledger.accepted_frames,
                num_frames=1,
                pts_us=ledger.accepted_frames * 1000000 // 24,
                data=b"123",
            ),
            event_id=ledger.event_id + 1,
            message_bytes=nbytes,
        )

    with pytest.raises(SourceTransportError) as error:
        accept(4097)
    assert error.value.code == "source_message_too_large"
    assert ledger.sequence == 0
    accept(4096)
    with pytest.raises(SourceTransportError) as error:
        accept(4096)
    assert error.value.code == "source_credit_exceeded"
    assert ledger.sequence == 1
    ledger.consume(1)
    accept(4096)
    assert ledger.sequence == 2


def test_real_websocket_size_and_receive_queue():
    async def run():
        release = asyncio.Event()

        async def app(scope, receive, send):
            assert scope["type"] == "websocket"
            await receive()
            await send({"type": "websocket.accept"})
            if scope["path"] == "/slow":
                await release.wait()
            while True:
                msg = await receive()
                if msg["type"] == "websocket.disconnect":
                    return
                await send({"type": "websocket.send", "bytes": msg["bytes"]})

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        config = uvicorn.Config(
            app,
            log_level="error",
            lifespan="off",
            **SourceTransportBudget(8192).websocket_options(),
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                await asyncio.sleep(0.01)
            url = f"ws://127.0.0.1:{sock.getsockname()[1]}"
            async with websockets.connect(
                url, compression=None, max_size=8192, max_queue=1
            ) as ws:
                await ws.send(b"x" * 8192)
                assert len(await ws.recv()) == 8192
                await ws.send(b"x" * 8193)
                with pytest.raises(ConnectionClosedError) as error:
                    await ws.recv()
                assert error.value.rcvd.code == 1009
            async with websockets.connect(
                url + "/slow", compression=None, max_queue=1
            ) as ws:
                for _ in range(20):
                    await ws.send(b"x" * 8192)
                await asyncio.sleep(0.1)
                active = [
                    conn
                    for conn in server.server_state.connections
                    if conn.path == "/slow"
                ]
                assert len(active) == 1
                assert active[0].max_queue == 1
                assert len(active[0].messages) == 1
                release.set()
                for _ in range(20):
                    assert len(await asyncio.wait_for(ws.recv(), 3)) == 8192
        finally:
            release.set()
            server.should_exit = True
            await asyncio.wait_for(task, 5)

    asyncio.run(run())


def test_small_append_cannot_hide_large_metadata():
    stream = SourceLedger(
        width=1, height=1, fps=24, limits=SourceLimits(1, 8192, 2, 6, 4)
    )
    payload = dict(
        client_seq=0, source_frame_start=0, num_frames=1, pts_us=0, data=b"123"
    )
    with pytest.raises(SourceTransportError) as error:
        stream.accept(payload, event_id=1, message_bytes=4100)
    assert error.value.code == "source_message_too_large"
    assert stream.sequence == 0
    stream.accept(payload, event_id=1, message_bytes=4099)
    assert stream.sequence == 1


def test_unsupported_transport_dependencies_fail_early(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime import source_transport

    for package, installed in [("uvicorn", "0.52.4"), ("websockets", "17.1")]:
        versions = {"uvicorn": "0.39.0", "websockets": "15.0.1"}
        versions[package] = installed
        monkeypatch.setattr(source_transport, "version", versions.__getitem__)
        with pytest.raises(RuntimeError, match=f"requires {package}"):
            SourceTransportBudget().websocket_options()


def test_backpressured_write_and_pong_share_drain_lock():
    """Reproduce overlapping flow-control waiters without GPU/network timing."""
    from websockets.legacy.protocol import WebSocketCommonProtocol

    async def run():
        protocol = WebSocketCommonProtocol()
        protocol.transport = None
        protocol.pause_writing()
        first = asyncio.create_task(protocol.drain())
        second = asyncio.create_task(protocol.drain())
        try:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not first.done()
            assert not second.done()
            protocol.resume_writing()
            await asyncio.wait_for(asyncio.gather(first, second), 1)
        finally:
            for task in (first, second):
                if not task.done():
                    task.cancel()
            await asyncio.gather(first, second, return_exceptions=True)

    asyncio.run(run())
