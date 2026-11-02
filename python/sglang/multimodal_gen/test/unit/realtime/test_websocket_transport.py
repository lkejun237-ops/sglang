# SPDX-License-Identifier: Apache-2.0
"""Exercise shared WS lifecycle primitives with a socket double, without CUDA."""

import ast
import asyncio
import logging
from pathlib import Path

import pytest


@pytest.fixture
def transport():
    # Load only the unchanged lifecycle primitives; HTTP/codec integration is
    # covered by test_decoder_output_api and the original realtime API tests.
    path = (
        Path(__file__).resolve().parents[3] / "runtime/realtime/websocket_transport.py"
    )
    tree = ast.parse(path.read_text())
    names = {
        "_LockedRealtimeWebSocket",
        "_RealtimeWebSocketSendGroup",
        "accept_realtime_websocket",
        "_drain_realtime_cleanup",
        "cancel_realtime_tasks",
        "_await_realtime_task",
    }
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names)
    namespace = dict(
        asyncio=asyncio,
        logger=logging.getLogger(__name__),
        WebSocket=object,
        WebSocketDisconnect=ConnectionError,
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def test_shared_accept_and_group_keep_header_payload_together(transport):
    async def run():
        class Socket:
            def __init__(self):
                self.accepted = False
                self.messages = []

            async def accept(self):
                self.accepted = True

            async def send_bytes(self, value):
                self.messages.append(value)
                await asyncio.sleep(0)

        raw = Socket()
        ws = await transport["accept_realtime_websocket"](raw)
        assert raw.accepted

        async def frames():
            async with ws.send_group():
                await ws.send_bytes(b"header")
                await asyncio.sleep(0.01)
                await ws.send_bytes(b"payload")

        await asyncio.gather(frames(), ws.send_bytes(b"control"))
        assert raw.messages == [b"header", b"payload", b"control"]

    asyncio.run(run())


def test_cancellation_releases_send_lock_and_joins_tasks(transport):
    async def run():
        entered = asyncio.Event()
        cleaned = []
        ws = transport["_LockedRealtimeWebSocket"](object())

        async def send():
            try:
                async with ws.send_group():
                    entered.set()
                    await asyncio.Event().wait()
            finally:
                cleaned.append(True)

        task = asyncio.create_task(send())
        await entered.wait()
        await transport["cancel_realtime_tasks"](task, None)
        assert cleaned == [True]
        async with ws.send_group():
            pass

    asyncio.run(run())


def test_repeated_cancellation_does_not_interrupt_resource_cleanup(transport):
    async def run():
        release, entered = asyncio.Event(), asyncio.Event()
        completed = []

        async def cleanup():
            entered.set()
            await release.wait()
            completed.append(True)

        task = asyncio.create_task(transport["_drain_realtime_cleanup"](cleanup()))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not completed
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert completed == [True]

    asyncio.run(run())
