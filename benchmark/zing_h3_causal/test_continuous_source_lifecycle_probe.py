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


@pytest.mark.parametrize("completed", [False, True])
def test_inflight_witness_requires_matching_unfinished_scheduler_request(
    tmp_path, completed
):
    import json
    from continuous_source_lifecycle_probe import wait_for_scheduler_inflight

    log = tmp_path / "server.log"
    events = [
        dict(
            event="server.scheduler_forward_start",
            session_id="other",
            request_id="other",
        ),
        dict(
            event="server.scheduler_forward_start",
            session_id="target",
            request_id="finished",
        ),
        dict(
            event="server.scheduler_forward_done",
            session_id="target",
            request_id="finished",
        ),
        dict(
            event="server.scheduler_forward_start",
            session_id="target",
            request_id="current",
        ),
    ]
    if completed:
        events.append(
            dict(
                event="server.scheduler_forward_done",
                session_id="target",
                request_id="current",
            )
        )
    log.write_text(
        "".join("realtime_trace " + json.dumps(event) + "\n" for event in events)
    )
    if completed:
        with pytest.raises(TimeoutError, match="No in-flight"):
            asyncio.run(wait_for_scheduler_inflight(log, "target", timeout=0.02))
    else:
        witness = asyncio.run(wait_for_scheduler_inflight(log, "target", timeout=0.1))
        assert witness["request_id"] == "current"


def test_negotiated_packets_keep_global_offsets_sequence_and_events():
    from continuous_source_lifecycle_probe import append_batches

    packets = []

    async def send(ws, message):
        packets.append(message)

    async def exercise():
        seq, event = await append_batches(
            send, None, bytes(range(153)), 3, 0, 17, 0, 1, 11
        )
        assert (seq, event) == (2, 3)
        # A prompt consumes one event but never consumes a source sequence.
        seq, event = await append_batches(
            send, None, bytes(range(153)), 3, 17, 17, seq, event + 1, 11
        )
        assert (seq, event) == (4, 6)
        assert [x["event_id"] for x in packets] == [1, 2, 4, 5]
        assert [x["payload"]["client_seq"] for x in packets] == [0, 1, 2, 3]
        assert [x["payload"]["source_frame_start"] for x in packets] == [0, 11, 17, 28]
        assert [x["payload"]["num_frames"] for x in packets] == [11, 6, 11, 6]
        assert b"".join(x["payload"]["data"] for x in packets) == bytes(range(102))
        await append_batches(
            send, None, bytes(range(153)), 3, 34, 1, seq + 1, event, 11
        )
        assert (
            packets[-1]["payload"]["client_seq"] == 5
        )  # expected 4: intentional bad sequence

    asyncio.run(exercise())
