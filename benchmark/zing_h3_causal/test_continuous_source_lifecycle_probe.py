# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest
import websockets
from websockets.exceptions import ConnectionClosed

from continuous_source_lifecycle_probe import negotiate


@pytest.mark.parametrize("reason", ["USER_SESSION_LIMIT", "UNRELATED_ERROR"])
def test_admission_close_before_init_is_classified_without_hiding_other_errors(reason):
    async def run():
        async def reject(ws):
            await ws.close(code=1008, reason=reason)

        async def send(ws, init):
            await ws.wait_closed()
            await ws.send(init)

        async def recv(ws):
            return await ws.recv()

        async with websockets.serve(reject, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                if reason == "USER_SESSION_LIMIT":
                    result = await negotiate(ws, b"init", send, recv)
                    assert result == dict(type="error", content=reason, close_code=1008)
                else:
                    with pytest.raises(ConnectionClosed):
                        await negotiate(ws, b"init", send, recv)

    asyncio.run(run())


def test_three_source_batches_match_server_timestamp_and_credit_contract():
    import msgpack
    from continuous_source_lifecycle_probe import source_payload
    from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
        SourceLedger,
        SourceLimits,
    )

    stride = 3
    pixels = bytes(range(153))
    ledger = SourceLedger(
        width=1, height=1, fps=24, limits=SourceLimits(17, 4096, 68, 204, 136)
    )
    for seq, start in enumerate((0, 17, 34)):
        payload = source_payload(pixels, stride, start, 17, seq)
        record = ledger.accept(
            payload, event_id=seq + 1, message_bytes=len(msgpack.packb(payload)) + 128
        )
        assert record.end == start + 17
    assert ledger.accepted_frames == 51


def test_keepalive_sends_protocol_heartbeats_and_stops_with_session():
    import msgpack
    from continuous_source_lifecycle_probe import keepalive

    async def run():
        received = []
        ready = asyncio.Event()

        async def collect(ws):
            async for data in ws:
                received.append(msgpack.unpackb(data, raw=False))
                ready.set()

        async with websockets.serve(collect, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                async with keepalive(ws, interval=0.01):
                    await asyncio.wait_for(ready.wait(), 1)
                await asyncio.sleep(0.03)
                assert received == [dict(type="event", kind="heartbeat")]

    asyncio.run(run())
