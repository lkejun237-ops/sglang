# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi import WebSocketDisconnect
from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import create_app
from sglang.multimodal_gen.runtime.realtime.async_vae_client import GatewayOutputClient
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    encode_message,
)


class _Sockets:
    def __init__(self):
        self.up = asyncio.Queue()
        self.down = asyncio.Queue()

    async def accept(self):
        pass

    async def receive_bytes(self):
        wire = await self.up.get()
        if wire is None:
            raise WebSocketDisconnect(1000)
        return wire

    async def send_bytes(self, wire):
        await self.down.put(wire)

    async def send(self, wire):
        await self.up.put(wire)

    async def recv(self):
        return await self.down.get()

    async def close(self, **_kwargs):
        await self.up.put(None)


@asynccontextmanager
async def _connection(*, enabled=True, session_id="s", app=None):
    if app is None:
        app = create_app(
            object(),
            model_revision="test",
            vae_fingerprint="test",
            internal_output_url="ws://gateway/v1/internal/realtime_output",
        )
    route = await app.state.output_registry.register(session_id, "g", token="test")
    endpoint = next(
        r.endpoint for r in app.routes if r.path == "/v1/internal/realtime_output"
    )
    sockets = _Sockets()

    async def connect(*_args, **_kwargs):
        return sockets

    task = asyncio.create_task(endpoint(sockets))
    client = GatewayOutputClient(
        "ws://gateway/v1/internal/realtime_output",
        session_id=session_id,
        generation_id="g",
        token="test",
        connect_factory=connect,
        media_delivery_ack=enabled,
        timeout_s=0.01,
    )
    await client.open()
    try:
        yield client, route, app
    finally:
        await route.close()
        await client.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _payload(sequence=0, *, session_id="s"):
    return encode_message(
        "media_payload",
        session_id=session_id,
        generation_id="g",
        sequence=sequence,
        payload=b"unchanged-fmp4" * 1024,
    )


def test_delivery_credit_includes_inflight_write_and_has_no_rpc_deadline():
    async def run():
        async with _connection() as (client, route, _):
            assert client.media_delivery_ack_enabled
            sending = asyncio.create_task(client.send(_payload()))
            output = await asyncio.wait_for(route.get_output(), 1)
            assert output.wire == _payload()
            assert route.queue_metrics()["gateway_queue_messages"] == 0
            # A popped item is still in flight. The 10 ms ordinary RPC timeout
            # must not terminate media delivery when the browser is slow.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(sending), 0.04)
            gate = asyncio.create_task(client.wait_writable())
            await asyncio.sleep(0)
            assert not gate.done()
            route.task_done()
            await asyncio.wait_for(sending, 1)
            await asyncio.wait_for(gate, 1)

    asyncio.run(run())


def test_legacy_output_is_unchanged_without_flow_negotiation():
    async def run():
        async with _connection(enabled=False) as (client, route, _):
            assert not client.media_delivery_ack_enabled
            await asyncio.wait_for(client.send(_payload()), 1)
            assert (await route.get_output()).wire == _payload()
            route.task_done()

    asyncio.run(run())


def test_repeated_stalls_remain_bounded_and_do_not_block_another_session():
    async def run():
        async with (
            _connection() as (slow, route, app),
            _connection(session_id="fast", app=app) as (fast, fast_route, _),
        ):
            for sequence in range(4):
                pending = asyncio.create_task(slow.send(_payload(sequence)))
                output = await asyncio.wait_for(route.get_output(), 1)
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(pending), 0.02)
                fast_pending = asyncio.create_task(
                    fast.send(_payload(sequence, session_id="fast"))
                )
                await asyncio.wait_for(fast_route.get_output(), 1)
                fast_route.task_done()
                await asyncio.wait_for(fast_pending, 1)
                assert not pending.done()
                assert route.queue_metrics()["gateway_queue_messages"] == 0
                assert output.wire == _payload(sequence)
                route.task_done()
                await asyncio.wait_for(pending, 1)
            complete = encode_message(
                "media_chunk_complete",
                session_id="s",
                generation_id="g",
                request_id="r",
                chunk_index=0,
                num_frames=4,
                media_transport="h264",
            )
            await asyncio.wait_for(slow.send(complete), 1)
            assert (await route.get_output()).wire == complete
            route.task_done()

    asyncio.run(run())


def test_requested_credit_falls_back_for_old_gateway():
    async def run():
        sockets = _Sockets()
        await sockets.down.put(
            encode_message("session_output_accepted", session_id="s", generation_id="g")
        )

        async def connect(*_args, **_kwargs):
            return sockets

        client = GatewayOutputClient(
            "ws://gateway/output",
            session_id="s",
            generation_id="g",
            token="test",
            media_delivery_ack=True,
            connect_factory=connect,
        )
        await client.open()
        assert not client.media_delivery_ack_enabled
        await asyncio.wait_for(client.send(_payload()), 1)
        await client.close()

    asyncio.run(run())
