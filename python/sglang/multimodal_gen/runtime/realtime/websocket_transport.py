# SPDX-License-Identifier: Apache-2.0
"""Shared transport lifecycle extracted from the original realtime WS endpoint.

Role-specific initialization and model work stay in the endpoint/adapters.
"""

from __future__ import annotations

import asyncio
import logging

import msgspec.msgpack
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class _LockedRealtimeWebSocket:
    def __init__(self, websocket: WebSocket):
        self._websocket = websocket
        self._send_lock = asyncio.Lock()
        self._send_lock_owner: asyncio.Task | None = None
        self._send_lock_depth = 0

    async def send_bytes(self, payload):
        await self._acquire_send_lock()
        try:
            await self._websocket.send_bytes(payload)
        finally:
            self._release_send_lock()

    async def close(self, *args, **kwargs):
        await self._acquire_send_lock()
        try:
            await self._websocket.close(*args, **kwargs)
        finally:
            self._release_send_lock()

    def send_group(self):
        return _RealtimeWebSocketSendGroup(self)

    def __getattr__(self, name):
        return getattr(self._websocket, name)

    async def _acquire_send_lock(self):
        task = asyncio.current_task()
        if self._send_lock_owner is task:
            self._send_lock_depth += 1
            return
        await self._send_lock.acquire()
        self._send_lock_owner = task
        self._send_lock_depth = 1

    def _release_send_lock(self):
        task = asyncio.current_task()
        if self._send_lock_owner is not task:
            return
        self._send_lock_depth -= 1
        if self._send_lock_depth <= 0:
            self._send_lock_owner = None
            self._send_lock_depth = 0
            self._send_lock.release()


class _RealtimeWebSocketSendGroup:
    def __init__(self, websocket: _LockedRealtimeWebSocket):
        self.websocket = websocket

    async def __aenter__(self):
        await self.websocket._acquire_send_lock()
        return self.websocket

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.websocket._release_send_lock()
        return False


async def _await_realtime_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    try:
        await task
    except (asyncio.CancelledError, WebSocketDisconnect):
        pass
    except Exception as e:
        logger.debug("realtime task exited with error: %s", e)


async def _close_realtime_websocket(
    websocket: WebSocket,
    *,
    code: int,
    reason: str,
    drain_input: bool = False,
) -> None:
    try:
        if drain_input:
            from .stage_profile_delivery import close_with_input_drain

            await close_with_input_drain(websocket, code=code, reason=reason)
        else:
            await websocket.close(code=code, reason=reason)
    except (RuntimeError, WebSocketDisconnect):
        pass


async def _drain_realtime_cleanup(cleanup) -> None:
    """Keep cleanup and its lease alive through repeated handler cancellation.

    Socket cancellation cannot stop worker kernels or the Encoder producer.
    The endpoint must not return admission or unlink inputs before cleanup
    finishes, even if shutdown cancels it a second time.
    """
    task = asyncio.create_task(cleanup, name="realtime-session-cleanup")
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    await task
    if cancelled:
        raise asyncio.CancelledError


async def write_error_msg(
    error_msg: str, websocket: WebSocket, *, code=None, fatal=None, event_id=None
):
    message = {"type": "error", "content": error_msg}
    if code is not None:
        message["code"] = code
    if fatal is not None:
        message["fatal"] = fatal
    if event_id is not None:
        message["event_id"] = event_id
    await websocket.send_bytes(msgspec.msgpack.encode(message))


async def accept_realtime_websocket(websocket):
    await websocket.accept()
    return _LockedRealtimeWebSocket(websocket)


async def cancel_realtime_tasks(*tasks):
    for task in tasks:
        if task is not None and not task.done():
            task.cancel()
    for task in tasks:
        await _await_realtime_task(task)


async def wait_realtime_tasks(tasks):
    return await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
