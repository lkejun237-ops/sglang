# SPDX-License-Identifier: Apache-2.0
"""Keep the bounded WebSocket receive queue flowing after session completion."""

import asyncio


async def collect_profile_with_input_drain(websocket, collection):
    """Own reception only after the session listener has stopped.

    Terminal application messages are no longer actionable. Consuming them lets
    the transport process PING/PONG while asynchronous profile writers drain.
    No payload queue is accumulated; profile ACK reception resumes at the caller
    after collection, before the server sends the profile result.
    """
    profile = asyncio.create_task(collection)
    receive = asyncio.create_task(websocket.receive_bytes())
    try:
        while not profile.done():
            done, _ = await asyncio.wait(
                (profile, receive), return_when=asyncio.FIRST_COMPLETED
            )
            if receive in done:
                receive.result()
                if not profile.done():
                    receive = asyncio.create_task(websocket.receive_bytes())
        return await profile
    finally:
        for task in (profile, receive):
            if not task.done():
                task.cancel()
        await asyncio.gather(profile, receive, return_exceptions=True)


async def close_with_input_drain(websocket, *, code, reason):
    """Finish the close handshake after the application receiver has stopped.

    A late heartbeat can fill the bounded application queue after the profile
    ACK. Keep transport reads flowing until CLOSE completes; no further input
    is actionable. Use ASGI receive(), which remains valid after sending CLOSE.
    """

    async def drain():
        while True:
            message = await websocket.receive()
            if (
                isinstance(message, dict)
                and message.get("type") == "websocket.disconnect"
            ):
                return

    receive = asyncio.create_task(drain())
    try:
        await websocket.close(code=code, reason=reason)
    finally:
        if not receive.done():
            receive.cancel()
        await asyncio.gather(receive, return_exceptions=True)
