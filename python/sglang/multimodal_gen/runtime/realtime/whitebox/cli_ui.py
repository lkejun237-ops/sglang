# SPDX-License-Identifier: Apache-2.0
"""Exact source-timed Worldkit UI renderer in a separate Chromium process.

The SDK owns layout, component loading, animations and UI history. PNG crosses
only the local process boundary; public playback and all recordings use H.264.
"""

import asyncio
import base64
import io
import json
import os
import signal
from pathlib import Path

from PIL import Image

from .metrics import stage


class CLIUI:
    metadata = {
        "mode": "worldkit_headless_ui",
        "training_eligible": True,
        "mapping": "source_frame_and_ui_revision",
        "capture": "chromium_png_ipc",
    }

    def __init__(self, process, spec):
        self.process, self.spec = process, spec
        self.messages = []
        self.closed = False

    async def request(self, value):
        if self.closed:
            raise ValueError("The UI renderer is closed")
        async with asyncio.timeout(30):
            self.process.stdin.write((json.dumps(value) + "\n").encode())
            await self.process.stdin.drain()
            line = await self.process.stdout.readline()
            if not line.endswith(b"\n"):
                raise ValueError("The UI renderer stopped")
            return json.loads(line)

    async def consume_ui(self, message):
        if len(self.messages) >= 4096:
            raise ValueError("The UI renderer history exceeded its limit")
        self.messages.append(message)

    async def render(self, rgb, frame, spec):
        messages, self.messages = self.messages, []
        header = frame.provenance["cli_frame_header"]
        with stage("cli_ui_capture", header["sessionId"], service="room_runtime"):
            result = await self.request(
                {
                    "type": "render",
                    "index": frame.index,
                    "header": header,
                    "rgb": base64.b64encode(rgb).decode(),
                    "messages": messages,
                }
            )
        if result.get("index") != frame.index:
            raise ValueError("The UI frame identity changed")

        def decode():
            with Image.open(
                io.BytesIO(base64.b64decode(result["png"], validate=True))
            ) as image:
                if image.format != "PNG" or image.size != (spec.width, spec.height):
                    raise ValueError("The UI image dimensions changed")
                return image.convert("RGB").tobytes()

        return await asyncio.to_thread(decode)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.messages.clear()
        if self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.process.wait(), 10)
            except TimeoutError:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await self.process.wait()


async def factory(*, descriptor, spec):
    cli = Path(os.environ["WHITEBOX_CLI_ENTRYPOINT"])
    root = cli.parents[3]
    entry = root / "packages/stream-player/src/headless.ts"
    if not entry.is_file():
        raise ValueError("The SDK headless UI adapter is missing")
    process = await asyncio.create_subprocess_exec(
        os.environ.get("WHITEBOX_NODE", "node"),
        "--import",
        "tsx",
        str(entry),
        cwd=root,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        limit=8 * 1024 * 1024,
        start_new_session=True,
    )
    renderer = CLIUI(process, spec)
    try:
        browser = {"rendering": os.environ.get("WHITEBOX_CLI_RENDERING", "auto")}
        if os.environ.get("WHITEBOX_CLI_BROWSER_EXECUTABLE"):
            browser["executablePath"] = os.environ["WHITEBOX_CLI_BROWSER_EXECUTABLE"]
        if await renderer.request(
            {"type": "init", "descriptor": descriptor, "browser": browser}
        ) != {"ready": True}:
            raise ValueError("The UI renderer did not become ready")
        return renderer
    except BaseException:
        await renderer.close()
        raise
