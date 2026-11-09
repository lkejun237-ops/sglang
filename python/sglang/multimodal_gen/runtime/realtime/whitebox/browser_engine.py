# SPDX-License-Identifier: Apache-2.0
"""Explicit mock CLI port: browser runs the existing, same-origin world.

Controls make a round trip through the gateway before the browser applies them.
Only one requested capture may be outstanding. This adapter is never selected
by a client: enable it using WHITEBOX_ENGINE_FACTORY in a mock deployment.
"""

import asyncio
import copy
import io
import time

from .contracts import EngineFrame


class BrowserEngine:
    metadata = {"mode": "mock_browser_world", "training_eligible": False}

    def __init__(self, *, world_id, spec, trace_id):
        self.spec = spec
        self.messages = asyncio.Queue(maxsize=32)
        self.pending = None
        self.stopped = False
        self.index = 0
        self.sequence = -1
        self.applied = -1
        self.inputs = {}
        self.history = {}

    async def client_messages(self):
        while True:
            yield await self.messages.get()

    async def send_input(self, sequence, payload):
        if len(self.history) >= 1024:
            raise ValueError("Browser world input backlog exceeded")
        self.history[sequence] = copy.deepcopy(payload)
        self.sequence = sequence
        await self.messages.put(
            dict(type="whitebox.control", sequence=sequence, payload=payload)
        )

    async def receive_frame(self, payload):
        if self.pending is None or self.pending.done():
            if self.stopped:
                return
            raise ValueError("Unrequested browser capture")
        if not isinstance(payload, dict) or payload.get("index") != self.index:
            raise ValueError("Browser capture timeline mismatch")
        sequence = payload.get("applied_sequence")
        data = payload.get("data")
        if (
            type(sequence) is not int
            or not self.applied <= sequence <= self.sequence
            or not isinstance(data, bytes)
            or not 0 < len(data) <= 4 * 1024 * 1024
        ):
            raise ValueError("Invalid browser capture")

        def decode():
            from PIL import Image

            with Image.open(io.BytesIO(data)) as image:
                if image.size != (
                    self.spec.width,
                    self.spec.height,
                ) or image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Invalid browser capture dimensions or format")
                return image.convert("RGB").tobytes()

        rgb = await asyncio.to_thread(decode)
        if self.pending.done():
            return
        events = []
        for seq in sorted(self.history):
            if seq <= sequence:
                self.inputs = self.history[seq]
                events.extend(self.inputs.get("events", []))
        self.history = {
            seq: value for seq, value in self.history.items() if seq > sequence
        }
        self.applied = sequence
        effective = {
            key: value for key, value in self.inputs.items() if key != "events"
        }
        if events:
            effective["events"] = events
        self.pending.set_result(
            EngineFrame(
                self.index,
                self.spec.pts(self.index),
                rgb,
                {
                    "mock": True,
                    "applied_sequence": sequence,
                    "capture": payload.get("states"),
                },
                effective,
            )
        )

    async def frames(self):
        next_capture = time.monotonic()
        while not self.stopped:
            await asyncio.sleep(max(0, next_capture - time.monotonic()))
            if self.stopped:
                break
            self.pending = asyncio.get_running_loop().create_future()
            await self.messages.put(
                dict(
                    type="whitebox.capture",
                    index=self.index,
                    pts_us=self.spec.pts(self.index),
                    width=self.spec.width,
                    height=self.spec.height,
                )
            )
            # A stalled/hidden tab cannot hold a slot indefinitely.
            frame = await asyncio.wait_for(self.pending, 75 if self.index == 0 else 10)
            if frame is None:
                break
            self.pending = None
            yield frame
            self.index += 1
            next_capture = max(next_capture + 1 / self.spec.fps, time.monotonic())

    async def finish(self):
        self.stopped = True
        if self.pending is not None and not self.pending.done():
            self.pending.set_result(None)

    async def close(self):
        await self.finish()


async def factory(**kwargs):
    return BrowserEngine(**kwargs)


class MockUIRenderer:
    metadata = {"mode": "mock_server_overlay", "version": 1}

    async def render(self, rgb, frame, spec):
        def draw():
            from PIL import Image, ImageDraw

            image = Image.frombytes("RGB", (spec.width, spec.height), rgb)
            painter = ImageDraw.Draw(image)
            painter.rectangle((0, 0, spec.width, 24), fill=(20, 30, 45))
            painter.text(
                (8, 6),
                f"MOCK | frame {frame.index} | input {frame.states.get('applied_sequence', -1)}",
                fill=(255, 215, 90),
            )
            return image.tobytes()

        return await asyncio.to_thread(draw)

    async def close(self):
        pass


def ui_factory(**_):
    return MockUIRenderer()
