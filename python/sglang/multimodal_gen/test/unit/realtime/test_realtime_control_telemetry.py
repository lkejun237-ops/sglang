# SPDX-License-Identifier: Apache-2.0
"""Control receipts must not be followed by video chunk telemetry."""

import asyncio
import time
from types import SimpleNamespace

import msgspec

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import realtime_video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)


def test_control_receipts_leave_no_media_telemetry_for_the_next_handshake():
    messages = []
    identity = {"preparation_id": "prepare-1"}

    class _Ws:
        async def send_bytes(self, wire):
            messages.append(msgspec.msgpack.decode(wire))

    class _Adapter:
        async def send_output(self, ws, session, result, batch):
            await ws.send_bytes(msgspec.msgpack.encode(result.receipt))
            return dict(
                header_pack_ms=0,
                header_write_ms=0,
                raw_payload_build_ms=0,
                raw_write_ms=0,
                ws_write_ms=0,
                raw_bytes=0,
                ws_payload_bytes=0,
                num_frames=0,
                num_batches=0,
                frame_shape=None,
                content_type="",
            )

    async def run():
        session = GenerateSession()
        session.adapter = _Adapter()
        for kind in ("source.prepared", "source.attached"):
            chunk = session.new_chunk(control_only=True)
            receipt = dict(type=kind, identity=identity)
            await realtime_video_api._send_output_and_log(
                _Ws(),
                session,
                chunk,
                SimpleNamespace(
                    block_idx=0, realtime_event_id=None, condition_inputs={}
                ),
                SimpleNamespace(receipt=receipt),
                request_prepare_ms=1,
                scheduler_forward_ms=2,
                chunk_started=time.perf_counter(),
                stage_metrics={},
            )
            # The real output path must not leave an unrelated packet behind
            # to be consumed as the response to the following attach command.
            assert messages == [receipt]
            messages.clear()
            session.generate_chunk_completed(chunk)
        assert session.playable_at is None
        assert session.new_chunk().index == 0

    asyncio.run(run())


def test_zero_frame_media_work_still_emits_chunk_telemetry():
    messages = []

    class _Ws:
        async def send_bytes(self, wire):
            messages.append(msgspec.msgpack.decode(wire))

    session = GenerateSession()
    chunk = session.new_chunk()
    asyncio.run(
        realtime_video_api._send_chunk_telemetry(
            _Ws(),
            session,
            chunk,
            SimpleNamespace(realtime_event_id=None),
            request_prepare_ms=1,
            scheduler_forward_ms=2,
            chunk_total_ms=3,
            send_stats=dict(
                pace_wait_ms=0,
                raw_payload_build_ms=0,
                ws_write_ms=0,
                raw_bytes=0,
                ws_payload_bytes=0,
                num_frames=0,
                content_type="",
            ),
        )
    )
    assert len(messages) == 1
    assert messages[0]["type"] == "chunk_telemetry"
    assert messages[0]["num_frames"] == 0
