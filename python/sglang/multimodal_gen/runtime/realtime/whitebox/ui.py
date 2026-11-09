# SPDX-License-Identifier: Apache-2.0
"""Backend composition independent of the world UI's state schema."""

import asyncio

from PIL import Image

from .contracts import EngineFrame, VideoSpec


class RgbaOverlayRenderer:
    """The world-specific renderer supplies a transparent RGBA UI layer.

    overlay(frame, spec) is async and can call a headless browser, SDK UI runtime
    or native painter. This compositor owns no game state or simulation clock.
    """

    def __init__(self, overlay, *, metadata: dict, close=None):
        self.overlay, self.metadata, self.cleanup = overlay, metadata, close

    async def render(self, rgb: bytes, frame: EngineFrame, spec: VideoSpec) -> bytes:
        rgba = await self.overlay(frame, spec)
        if len(rgba) != spec.width * spec.height * 4:
            raise ValueError("UI overlay must match the video viewport")

        def composite():
            background = Image.frombytes("RGB", (spec.width, spec.height), rgb).convert(
                "RGBA"
            )
            foreground = Image.frombytes("RGBA", (spec.width, spec.height), rgba)
            return (
                Image.alpha_composite(background, foreground).convert("RGB").tobytes()
            )

        return await asyncio.to_thread(composite)

    async def close(self):
        if self.cleanup is not None:
            await self.cleanup()
