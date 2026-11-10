# SPDX-License-Identifier: Apache-2.0
"""Optional SDK control tunnel on an already authorized member attachment.

The original SDK socket still owns binding offers, leases and input sequences.
Packets are neither translated into legacy input nor replayed on reconnect.
"""

import asyncio
import re
import time
from collections import deque
from urllib.parse import urlencode

from .control_input_policy import (
    CONTROL_GAMEPLAY_MAX_AGE_SECONDS,
    CONTROL_MAX_QUEUED_MESSAGES,
    is_gameplay_message,
)
from .player_transport import MAX_VIEWER_CONTROL_BYTES, ws_url
from .render_contract import RenderError


class SDKControlRelay:
    def __init__(self, pipeline, subscription, client_id, *, clock=None):
        if not isinstance(client_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}", client_id
        ):
            raise RenderError("INVALID_COMMAND")
        self.pipeline, self.subscription, self.client_id = (
            pipeline,
            subscription,
            client_id,
        )
        self.queue = asyncio.Queue(maxsize=CONTROL_MAX_QUEUED_MESSAGES)
        self.queued_bytes = 0
        self.socket = self.task = None
        self.closed = False
        self.clock = clock or time.monotonic
        self.queued_gameplay = deque()
        self.queue_changed = asyncio.Event()

    def check(self):
        p, sub = self.pipeline, self.subscription
        if (
            self.closed
            or p.closed
            or sub.closed
            or p.subscription is not sub
            or p.generation != sub.generation
        ):
            raise RenderError("SOURCE_EPOCH_MISMATCH")
        p.member_source.check()

    def emit(self, **fields):
        self.check()
        # Binding offers are attachment-local; never enter replay history.
        self.subscription.offer_control(
            dict(
                type="sdk.control",
                connection_generation=self.subscription.generation,
                **fields,
            )
        )

    def enqueue(self, payload):
        self.check()
        if not isinstance(payload, str) or not payload:
            raise RenderError("INVALID_COMMAND")
        size = len(payload.encode())
        if (
            size > MAX_VIEWER_CONTROL_BYTES
            or self.queue.full()
            or self.queued_bytes + size > MAX_VIEWER_CONTROL_BYTES
        ):
            self.stop()
            raise RenderError("LIMIT_EXCEEDED")
        if self.socket is None:
            raise RenderError("CONTROL_UNAVAILABLE", retryable=True)
        gameplay = is_gameplay_message(payload)
        self.queued_bytes += size
        entered = self.clock() if gameplay else None
        self.queue.put_nowait((payload, size, entered))
        if entered is not None:
            self.queued_gameplay.append(entered)
            self.queue_changed.set()

    def stop(self):
        self.closed = True
        if self.task is not None:
            self.task.cancel()

    async def run(self, *, connect=None):
        if connect is None:
            from websockets.asyncio.client import connect
        tasks = []
        try:
            self.check()
            source = self.pipeline.member_source
            transport = source.runtime.player_transport
            entry, descriptor = transport.viewer(
                source.record["stream_id"], source.sdk["sessionId"]
            )
            if entry is not source.entry:
                raise RenderError("SOURCE_EPOCH_MISMATCH")
            self.socket = await connect(
                ws_url(
                    descriptor["baseUrl"],
                    f"/v1/sessions/{descriptor['sessionId']}/control",
                    urlencode(
                        dict(
                            clientId=self.client_id,
                            role="controller",
                            ui="1",
                            clock="0",
                        )
                    ),
                ),
                origin=descriptor["baseUrl"],
                subprotocols=[
                    "worldkit.stream.v1",
                    "worldkit.cap." + descriptor["accessToken"],
                ],
                max_size=MAX_VIEWER_CONTROL_BYTES,
                max_queue=2,
                compression=None,
                open_timeout=5,
                close_timeout=1,
            )
            self.emit(state="ready")

            async def observe(direction, payload):
                hook = getattr(source.runtime, "observe_player_control", None)
                if hook:
                    await hook(
                        source.record["stream_id"], self.client_id, direction, payload
                    )

            async def send():
                while True:
                    payload, size, entered = await self.queue.get()
                    self.queued_bytes -= size
                    if entered is not None:
                        self.queued_gameplay.popleft()
                        self.queue_changed.set()
                    self.check()
                    timeout = 5
                    if entered is not None:
                        timeout = CONTROL_GAMEPLAY_MAX_AGE_SECONDS - (
                            self.clock() - entered
                        )
                        if timeout <= 0:
                            raise RenderError("CONTROL_INPUT_EXPIRED")

                    # A stalled live socket cannot deliver a retired gameplay queue
                    # later. Closing retires its original SDK lease; no resequencing.
                    async def dispatch():
                        # Register pending intent before a fast SDK receipt can arrive.
                        await observe("to-sdk", payload)
                        self.check()
                        await self.socket.send(payload)

                    # Keep dispatch in this task: Python 3.11 wait_for can
                    # swallow cancellation when its child just completed.
                    async with asyncio.timeout(timeout):
                        await dispatch()

            async def expire_queued_gameplay():
                # A preceding handshake/archive await must not hide the oldest
                # gameplay deadline. In-flight gameplay keeps its send budget.
                while True:
                    self.queue_changed.clear()
                    if not self.queued_gameplay:
                        await self.queue_changed.wait()
                        continue
                    remaining = CONTROL_GAMEPLAY_MAX_AGE_SECONDS - (
                        self.clock() - self.queued_gameplay[0]
                    )
                    if remaining <= 0:
                        raise RenderError("CONTROL_INPUT_EXPIRED")
                    try:
                        async with asyncio.timeout(remaining):
                            await self.queue_changed.wait()
                    except asyncio.TimeoutError:
                        raise RenderError("CONTROL_INPUT_EXPIRED") from None

            async def receive():
                while True:
                    self.check()
                    try:
                        async with asyncio.timeout(1):
                            payload = await self.socket.recv()
                    except asyncio.TimeoutError:
                        continue
                    if (
                        not isinstance(payload, str)
                        or len(payload.encode()) > MAX_VIEWER_CONTROL_BYTES
                    ):
                        raise RenderError("INVALID_COMMAND")
                    await observe("from-sdk", payload)
                    self.emit(payload=payload)

            tasks = [
                asyncio.create_task(work)
                for work in (send(), receive(), expire_queued_gameplay())
            ]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except asyncio.CancelledError:
            raise
        except Exception:
            try:
                self.emit(state="closed", code="CONTROL_UNAVAILABLE")
            except Exception:
                pass
        finally:
            self.closed = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.socket is not None:
                await self.socket.close()
            while not self.queue.empty():
                self.queue.get_nowait()
            self.queued_bytes = 0
            self.queued_gameplay.clear()
