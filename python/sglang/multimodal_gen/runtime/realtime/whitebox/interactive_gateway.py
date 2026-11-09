# SPDX-License-Identifier: Apache-2.0
"""Opt-in direct-generate route; authentication stays at the existing boundary."""

import asyncio
import logging
import os
from uuid import uuid4

import msgspec
from starlette.websockets import WebSocketDisconnect

from .interactive import InteractiveSession, ModelLeases
from .metrics import stage
from .reports import SessionReport

log = logging.getLogger(__name__)


async def serve(
    websocket,
    *,
    coordinator,
    connect,
    config,
    archives,
    reports,
    model_revision,
    vae_fingerprint,
    user_id,
    trace_id,
    renew_seconds,
):
    await websocket.accept()
    report = SessionReport(trace_id)
    session = InteractiveSession(
        leases=ModelLeases(
            coordinator,
            connect,
            user_id=user_id,
            model_revision=model_revision,
            vae_fingerprint=vae_fingerprint,
            trace_id=trace_id,
            max_bytes=config.max_message_bytes,
            renew_seconds=renew_seconds,
        ),
        config=config,
        archives=archives,
        reports=reports,
        report=report,
        trace_id=trace_id,
        session_id=uuid4().hex,
        user_id=user_id,
    )
    rtc = None
    tasks = []
    try:
        # Negotiation is bounded and must precede any world/slot allocation.
        wire = await asyncio.wait_for(websocket.receive_bytes(), 30)
        message = msgspec.msgpack.decode(wire)
        if isinstance(message, dict) and message.get("type") == "rtc.prepare":
            if os.getenv("WHITEBOX_WEBRTC_ENABLED") != "1":
                raise ValueError("WebRTC is unavailable on this gateway.")
            from .webrtc import WebRTCMedia

            rtc = WebRTCMedia()
            await websocket.send_bytes(
                msgspec.msgpack.encode(
                    dict(type="rtc.configuration", iceServers=rtc.servers)
                )
            )
            message = msgspec.msgpack.decode(
                await asyncio.wait_for(websocket.receive_bytes(), 30)
            )
            if message.get("type") != "rtc.offer":
                raise ValueError("Expected a WebRTC offer.")
            answer = await asyncio.wait_for(rtc.answer(message.get("sdp")), 20)
            await websocket.send_bytes(
                msgspec.msgpack.encode(dict(type="rtc.answer", sdp=answer))
            )
            await rtc.wait_connected()
            session.video_transport = rtc
            wire = await asyncio.wait_for(websocket.receive_bytes(), 30)
        await session.send(wire)

        async def receive():
            while True:
                await session.send(await websocket.receive_bytes())

        async def send():
            while True:
                payload = await session.recv()
                message = msgspec.msgpack.decode(payload)
                with stage("browser_send", trace_id, bytes=len(payload)):
                    await websocket.send_bytes(payload)
                if message.get("type") in ("whitebox.frame", "frame_batch"):
                    session.media_delivered()
                if message.get("type") == "session.completed":
                    return

        tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass
    except Exception as error:
        log.exception("Interactive session failed trace_id=%s", trace_id)
        from .diagnostics import public_error

        try:
            await websocket.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        version=2,
                        type="error",
                        **public_error(
                            error, trace_id, default="INTERACTIVE_SESSION_FAILED"
                        ),
                    )
                )
            )
        except Exception:
            pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await session.close()
        finally:
            if rtc:
                await rtc.close()
            report.gateway_done.set()
            if report.recording is None:
                report.close()
            try:
                await websocket.close()
            except Exception:
                pass
