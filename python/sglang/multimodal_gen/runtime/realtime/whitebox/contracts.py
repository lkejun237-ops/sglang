# SPDX-License-Identifier: Apache-2.0
"""Ports for the independently developed world CLI and UI renderer.

Only the envelope is shared. States and engine input remain application-owned.
Plugins are loaded from server configuration, never from a browser request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol


@dataclass(frozen=True)
class VideoSpec:
    width: int = 832
    height: int = 480
    fps: int = 24

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 3

    def pts(self, frame: int) -> int:
        return frame * 1_000_000 // self.fps


@dataclass(frozen=True)
class EngineFrame:
    index: int
    pts_us: int
    rgb: bytes
    states: Any
    # Full effective input snapshot, including ordered one-shot events if any.
    # None means the CLI cannot yet prove which inputs affected this frame.
    effective_input: dict[str, Any] | None = None
    # Optional fallback evidence when consumption receipts are unavailable.
    # This is explicitly approximate and never claims an effective input frame.
    observed_input: dict[str, Any] | None = None
    # Source presentation identity / room time; opaque UI state is never coerced.
    provenance: dict[str, Any] | None = None
    # Outer rendering revision; independent of the CLI's opaque states/input.
    rendering: dict[str, Any] | None = None
    references: list[dict[str, Any]] | None = None


def capture_pts(frame: EngineFrame) -> int:
    """Media/archive time; never substitute this into the H3 24fps clock."""
    value = (frame.provenance or {}).get("capture_pts_us", frame.pts_us)
    if type(value) is not int or value < 0:
        raise ValueError("Invalid capture timestamp")
    return value


def presentation_pts(frame: EngineFrame) -> int:
    """Optional model playback placement, independent of real capture/UI time."""
    value = (frame.provenance or {}).get(
        "model_presentation_pts_us", capture_pts(frame)
    )
    if type(value) is not int or value < 0:
        raise ValueError("Invalid presentation timestamp")
    return value


class EngineSession(Protocol):
    def frames(self) -> AsyncIterator[EngineFrame]: ...

    async def send_input(self, sequence: int, payload: dict[str, Any]) -> None: ...

    async def finish(self) -> None:
        """Stop producing new frames and drain already committed frames."""
        ...

    async def close(self) -> None: ...


class EngineFactory(Protocol):
    async def __call__(
        self, *, world_id: str, spec: VideoSpec, trace_id: str
    ) -> EngineSession: ...


class UIRenderer(Protocol):
    # Stable bundle/resource identities for manifest.ui; no credentials or URLs
    # whose validity expires before the recording's retention period.
    metadata: dict[str, Any]

    async def render(
        self, rgb: bytes, frame: EngineFrame, spec: VideoSpec
    ) -> bytes: ...

    async def close(self) -> None: ...


class NoUIRenderer:
    """Explicit opt-in for worlds with no UI, not a substitute for a UI plugin."""

    metadata = {"mode": "none", "version": 1}

    async def render(self, rgb: bytes, frame: EngineFrame, spec: VideoSpec) -> bytes:
        return rgb

    async def close(self) -> None:
        pass


def no_ui_renderer(**_):
    return NoUIRenderer()
