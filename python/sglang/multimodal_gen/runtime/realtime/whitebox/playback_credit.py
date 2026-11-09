# SPDX-License-Identifier: Apache-2.0
"""Optional frame receipts bound media buffered beyond the gateway/proxy.

Receipts acknowledge transport delivery, not input application or rendering.
Waiting never drops an H.264 reference frame and never blocks the input reader.
"""

import asyncio


class PlaybackCredit:
    def __init__(self, timeout=15.0):
        self.window = None
        self.received = self.sent = -1
        self.timeout = timeout
        self.changed = asyncio.Condition()

    async def acknowledge(self, payload):
        window, frame = payload.get("window"), payload.get("frame")
        if (
            type(window) is not int
            or not 1 <= window <= 32
            or type(frame) is not int
            or not -1 <= frame <= self.sent
            or (self.window is not None and window != self.window)
        ):
            raise ValueError("Invalid playback receipt")
        async with self.changed:
            self.window = window
            self.received = max(self.received, frame)
            self.changed.notify_all()

    async def reserve(self, frame):
        if frame != self.sent + 1:
            raise ValueError("Invalid playback frame sequence")
        async with self.changed:
            await asyncio.wait_for(
                self.changed.wait_for(
                    lambda: self.window is None or frame - self.received <= self.window
                ),
                timeout=self.timeout,
            )
            self.sent = frame
