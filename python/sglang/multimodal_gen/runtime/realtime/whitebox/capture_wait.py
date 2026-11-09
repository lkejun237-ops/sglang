# SPDX-License-Identifier: Apache-2.0
"""Bounded media silence for a confirmed, exclusive SDK capture hold."""

import asyncio
import time


class CaptureWait:
    def __init__(self, *, frame_seconds=15, hold_seconds=300, clock=time.monotonic):
        self.frame_seconds, self.hold_seconds, self.clock = (
            frame_seconds,
            hold_seconds,
            clock,
        )
        self.request_id = None
        self.confirmed = False
        self.until = 0.0
        self.resumed_at = 0.0
        self.changed = asyncio.Event()

    def _notify(self):
        previous, self.changed = self.changed, asyncio.Event()
        previous.set()

    def pending(self, request_id):
        if self.request_id is not None:
            raise ValueError("Capture hold is already pending")
        self.request_id = request_id
        self.confirmed = False
        self.until = self.clock() + self.frame_seconds
        self._notify()

    def held(self, request_id):
        if (
            request_id != self.request_id
            or self.confirmed
            or self.clock() >= self.until
        ):
            raise ValueError("Capture hold receipt expired")
        self.confirmed = True
        self.until = self.clock() + self.hold_seconds
        self._notify()

    def resumed(self, request_id):
        if self.request_id != request_id:
            raise ValueError("Capture resume identity mismatch")
        self.request_id, self.until = None, 0.0
        self.resumed_at = self.clock()
        self._notify()

    async def receive(self, operation):
        task = asyncio.create_task(operation)
        deadline = self.clock() + self.frame_seconds
        changed = None
        try:
            while True:
                event = self.changed
                until = (
                    self.until
                    if self.request_id is not None
                    else max(deadline, self.resumed_at + self.frame_seconds)
                )
                changed = asyncio.create_task(event.wait())
                done, _ = await asyncio.wait(
                    (task, changed),
                    timeout=max(0, until - self.clock()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if task in done:
                    return task.result()
                if not changed.done():
                    changed.cancel()
                await asyncio.gather(changed, return_exceptions=True)
                changed = None
                if event.is_set():
                    continue  # A hold/resume also affects an already pending read.
                raise ValueError("The CLI stream did not become ready")
        finally:
            pending = [item for item in (task, changed) if item is not None]
            for item in pending:
                if not item.done():
                    item.cancel()
            try:
                await asyncio.gather(*pending, return_exceptions=True)
            finally:
                # Cancellation can arrive during waiter cleanup, after recv
                # already failed. Preserve that original transport evidence.
                if task.done() and not task.cancelled():
                    task.result()
