# SPDX-License-Identifier: Apache-2.0
"""Optional SDK control tunnel on an already authorized member attachment.

The original SDK socket still owns binding offers, leases and input sequences.
Packets are neither translated into legacy input nor replayed on reconnect.
"""

import asyncio
import re
from urllib.parse import urlencode

from .player_transport import MAX_VIEWER_CONTROL_BYTES, ws_url
from .render_contract import RenderError


class SDKControlRelay:
    def __init__(self, pipeline, subscription, client_id):
        if not isinstance(client_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}", client_id
        ):
            raise RenderError("INVALID_COMMAND")
        self.pipeline, self.subscription, self.client_id = (
            pipeline,
            subscription,
            client_id,
        )
        self.queue = asyncio.Queue(maxsize=64)
        self.queued_bytes = 0
        self.socket = self.task = None
        self.closed = False

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
        self.queued_bytes += size
        self.queue.put_nowait((payload, size))

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
                    payload, size = await self.queue.get()
                    self.queued_bytes -= size
                    self.check()
                    await asyncio.wait_for(self.socket.send(payload), 5)
                    await observe("to-sdk", payload)

            async def receive():
                while True:
                    self.check()
                    try:
                        payload = await asyncio.wait_for(self.socket.recv(), 1)
                    except asyncio.TimeoutError:
                        continue
                    if (
                        not isinstance(payload, str)
                        or len(payload.encode()) > MAX_VIEWER_CONTROL_BYTES
                    ):
                        raise RenderError("INVALID_COMMAND")
                    await observe("from-sdk", payload)
                    self.emit(payload=payload)

            tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
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
