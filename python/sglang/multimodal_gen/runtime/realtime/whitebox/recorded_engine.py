# SPDX-License-Identifier: Apache-2.0
"""Recorded video engine implementing Room EngineSession and capture_control."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
    EngineFactory,
    EngineFrame,
    EngineSession,
    VideoSpec,
)


def _iter_rgb24_via_pyav(
    mp4_path: Path, width: int, height: int, max_frames: int | None = None
) -> Iterable[bytes]:
    import av  # type: ignore[import-not-found]

    container = av.open(str(mp4_path))
    try:
        count = 0
        for packet_frame in container.decode(video=0):
            if max_frames is not None and count >= max_frames:
                break
            if packet_frame.width != width or packet_frame.height != height:
                packet_frame = packet_frame.reformat(
                    width=width, height=height, format="rgb24"
                )
            else:
                packet_frame = packet_frame.reformat(format="rgb24")
            plane = bytes(packet_frame.planes[0])
            expected = width * height * 3
            if len(plane) == expected:
                rgb = plane
            else:
                arr = packet_frame.to_ndarray(format="rgb24")
                rgb = arr.tobytes()
            count += 1
            yield rgb
    finally:
        container.close()


class _FfmpegRgb24Reader:
    """Streams raw RGB24 frames from an .mp4 file via an ffmpeg subprocess."""

    def __init__(self, mp4_path: Path, width: int, height: int) -> None:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg binary is required when PyAV is unavailable")
        self.frame_bytes = width * height * 3
        cmd = [
            ffmpeg_bin,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(mp4_path),
            "-vf",
            f"scale={width}:{height}:flags=bicubic",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def read_frame(self) -> bytes | None:
        if self._proc.stdout is None:
            return None
        chunks: list[bytes] = []
        remaining = self.frame_bytes
        while remaining > 0:
            chunk = self._proc.stdout.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining != 0:
            return None
        return b"".join(chunks)

    def close(self) -> None:
        if self._proc.stdout is not None:
            try:
                self._proc.stdout.close()
            except Exception:
                pass
        if self._proc.stderr is not None:
            try:
                self._proc.stderr.close()
            except Exception:
                pass
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2.0)


class RecordedVideoEngine(EngineSession):
    """Replays a recorded .mp4 video or in-memory RGB frames as a 24fps Room EngineSession."""

    def __init__(
        self,
        spec: VideoSpec | None = None,
        *,
        mp4_path: str | Path | None = None,
        rgb_frames: Sequence[bytes] | Callable[[int, VideoSpec], bytes] | None = None,
        max_frames: int | None = None,
        capture_fps: int | None = None,
        realtime_pacing: bool = True,
        pacing_speed: float = 1.0,
        fast_frame_delay_s: float = 0.002,
        loop: bool = False,
        source_id: str = "recorded-whitebox",
        session_id: str | None = None,
        stream_id: str | None = None,
        epoch: int = 1,
        media_generation: int = 1,
        prefer_ffmpeg: bool = False,
    ) -> None:
        self.spec = spec or VideoSpec()
        if min(self.spec.width, self.spec.height, self.spec.fps) <= 0:
            raise ValueError("VideoSpec dimensions and fps must be positive")
        self.mp4_path = Path(mp4_path) if mp4_path is not None else None
        self.rgb_frames = rgb_frames
        if self.mp4_path is None and self.rgb_frames is None:
            raise ValueError("Either mp4_path or rgb_frames must be provided")
        if self.mp4_path is not None and not self.mp4_path.exists():
            raise FileNotFoundError(f"Recorded video not found: {self.mp4_path}")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when specified")
        if pacing_speed <= 0:
            raise ValueError("pacing_speed must be > 0")

        raw_capture_fps = (
            str(capture_fps)
            if capture_fps is not None
            else os.environ.get("WHITEBOX_CLI_FPS", str(self.spec.fps))
        )
        try:
            resolved_capture_fps = int(raw_capture_fps)
        except ValueError as exc:
            raise ValueError(
                f"WHITEBOX_CLI_FPS must be an integer, got {raw_capture_fps!r}"
            ) from exc
        if resolved_capture_fps < 1 or resolved_capture_fps > 60:
            raise ValueError(
                f"WHITEBOX_CLI_FPS must be between 1 and 60, got {resolved_capture_fps}"
            )

        self.max_frames = max_frames
        self.capture_fps = resolved_capture_fps
        self.metadata = {
            "capture_fps": self.capture_fps,
            "model_fps": self.spec.fps,
        }
        self.realtime_pacing = realtime_pacing
        self.pacing_speed = pacing_speed
        self.fast_frame_delay_s = max(0.0, fast_frame_delay_s)
        self.loop = loop
        self.source_id = source_id
        self.session_id = session_id or f"rec-src-{uuid4().hex[:12]}"
        self.stream_id = stream_id or f"view-{uuid4().hex[:12]}"
        self.epoch = epoch
        self.media_generation = media_generation
        self.prefer_ffmpeg = prefer_ffmpeg

        self.closed = False
        self.inputs: list[tuple[int, dict[str, Any]]] = []
        self.control_log: list[tuple[str, str | None, int]] = []
        self.emitted_frames = 0

        self._last_emitted_index = -1
        self._first_emitted = asyncio.Event()
        self._hold_gate = asyncio.Event()
        self._hold_gate.set()
        self._held_context_id: str | None = None
        self._hold_started_at: float | None = None
        self._paused_pts_offset_us: int = 0
        self._clock_origin: float | None = None
        self._active_reader: _FfmpegRgb24Reader | None = None

    def _frame_provenance(self, index: int, pts_us: int) -> dict[str, Any]:
        return {
            "capture_pts_us": pts_us,
            "cli_frame_header": {
                "sessionId": self.session_id,
                "epoch": self.epoch,
                "mediaGeneration": self.media_generation,
                "outputFrameId": index,
            },
        }

    async def _pace_before_frame(self, index: int) -> None:
        if index <= 0:
            return
        await self._hold_gate.wait()
        if self.closed:
            return
        if self.realtime_pacing:
            assert self._clock_origin is not None
            target_time = self._clock_origin + (
                index / (self.capture_fps * self.pacing_speed)
            )
            delay = target_time - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        elif self.fast_frame_delay_s > 0:
            await asyncio.sleep(self.fast_frame_delay_s)
        else:
            await asyncio.sleep(0)
        while not self._hold_gate.is_set() and not self.closed:
            await self._hold_gate.wait()

    async def _iter_decoded_rgb(self) -> AsyncIterator[bytes]:
        if self.rgb_frames is not None:
            if callable(self.rgb_frames):
                idx = 0
                while self.max_frames is None or idx < self.max_frames:
                    if self.closed:
                        return
                    rgb = self.rgb_frames(idx, self.spec)
                    if len(rgb) != self.spec.frame_bytes:
                        raise ValueError(
                            f"Synthetic frame {idx} size {len(rgb)} != {self.spec.frame_bytes}"
                        )
                    yield rgb
                    idx += 1
                return

            frames_seq = list(self.rgb_frames)
            if not frames_seq:
                return
            emitted = 0
            while True:
                for rgb in frames_seq:
                    if self.closed or (
                        self.max_frames is not None and emitted >= self.max_frames
                    ):
                        return
                    if len(rgb) != self.spec.frame_bytes:
                        raise ValueError(
                            f"Frame {emitted} size {len(rgb)} != {self.spec.frame_bytes}"
                        )
                    yield rgb
                    emitted += 1
                if not self.loop:
                    return

        assert self.mp4_path is not None
        use_pyav = False
        if not self.prefer_ffmpeg:
            try:
                import av as _av  # noqa: F401

                use_pyav = True
            except ImportError:
                use_pyav = False

        emitted = 0
        while True:
            if self.closed or (
                self.max_frames is not None and emitted >= self.max_frames
            ):
                return
            pass_emitted = 0
            if use_pyav:
                remaining = (
                    None if self.max_frames is None else self.max_frames - emitted
                )
                for rgb in _iter_rgb24_via_pyav(
                    self.mp4_path,
                    self.spec.width,
                    self.spec.height,
                    max_frames=remaining,
                ):
                    if self.closed:
                        return
                    yield rgb
                    emitted += 1
                    pass_emitted += 1
                    if self.max_frames is not None and emitted >= self.max_frames:
                        return
            else:
                reader = _FfmpegRgb24Reader(
                    self.mp4_path, self.spec.width, self.spec.height
                )
                self._active_reader = reader
                try:
                    while not self.closed and (
                        self.max_frames is None or emitted < self.max_frames
                    ):
                        rgb = await asyncio.to_thread(reader.read_frame)
                        if rgb is None:
                            break
                        yield rgb
                        emitted += 1
                        pass_emitted += 1
                finally:
                    reader.close()
                    if self._active_reader is reader:
                        self._active_reader = None

            if not self.loop or pass_emitted == 0:
                return

    async def frames(self) -> AsyncIterator[EngineFrame]:
        self._clock_origin = time.monotonic()
        index = 0
        try:
            async for rgb in self._iter_decoded_rgb():
                if self.closed:
                    break
                await self._pace_before_frame(index)
                if self.closed:
                    break
                pts_us = self.spec.pts(index)
                capture_pts_us = (
                    (index * 1_000_000) // self.capture_fps
                ) + self._paused_pts_offset_us
                frame = EngineFrame(
                    index=index,
                    pts_us=pts_us,
                    rgb=rgb,
                    states={"frame": index, "source_id": self.source_id},
                    provenance=self._frame_provenance(index, capture_pts_us),
                )
                self._last_emitted_index = index
                self.emitted_frames = index + 1
                self._first_emitted.set()
                yield frame
                index += 1
        finally:
            self._close_reader()

    async def capture_control(
        self, action: str, context_id: str | None = None
    ) -> dict[str, Any]:
        if action == "hold":
            self._hold_gate.clear()
            if self._last_emitted_index < 0:
                await self._first_emitted.wait()
            f0_index = max(0, self._last_emitted_index)
            self._held_context_id = context_id
            if self._hold_started_at is None:
                self._hold_started_at = time.monotonic()
            self.control_log.append(("hold", context_id, f0_index))
            return {
                "frame": {
                    "sessionId": self.session_id,
                    "epoch": self.epoch,
                    "mediaGeneration": self.media_generation,
                    "outputFrameId": f0_index,
                }
            }
        if action == "resume":
            paused_s = 0.0
            if self._hold_started_at is not None:
                paused_s = max(0.0, time.monotonic() - self._hold_started_at)
                if self._clock_origin is not None:
                    self._clock_origin += paused_s
            paused_ms = round(paused_s * 1000.0, 3)
            self._paused_pts_offset_us += round(paused_ms * 1000)
            self._hold_started_at = None
            self._held_context_id = None
            self._hold_gate.set()
            self.control_log.append(
                ("resume", context_id, max(0, self._last_emitted_index))
            )
            return {"pausedMs": paused_ms}
        raise ValueError(f"Unsupported capture_control action: {action!r}")

    async def send_input(self, sequence: int, payload: dict[str, Any]) -> None:
        self.inputs.append((sequence, payload))

    def _close_reader(self) -> None:
        if self._active_reader is not None:
            self._active_reader.close()
            self._active_reader = None

    async def finish(self) -> None:
        await self.close()

    async def close(self) -> None:
        self.closed = True
        self._hold_gate.set()
        self._first_emitted.set()
        self._close_reader()

    async def retire(self) -> dict[str, Any]:
        await self.close()
        return {
            "state": "retired",
            "source_id": self.source_id,
            "owner_epoch": str(self.epoch),
            "view_id": self.stream_id,
        }


def create_recorded_video_engine(
    mp4_path: str | Path | None = None,
    *,
    rgb_frames: Sequence[bytes] | Callable[[int, VideoSpec], bytes] | None = None,
    max_frames: int | None = None,
    capture_fps: int | None = None,
    realtime_pacing: bool = True,
    pacing_speed: float = 1.0,
    fast_frame_delay_s: float = 0.002,
    loop: bool = False,
    session_id: str | None = None,
    stream_id: str | None = None,
    epoch: int = 1,
    media_generation: int = 1,
    prefer_ffmpeg: bool = False,
    on_created: Callable[[RecordedVideoEngine], None] | None = None,
) -> EngineFactory:
    """Create an EngineFactory compatible with WhiteboxConfig(engine_factory=...)."""

    async def _factory(*args: Any, **kwargs: Any) -> RecordedVideoEngine:
        source_id = "recorded-whitebox"
        spec: VideoSpec | None = None
        if len(args) >= 2 and isinstance(args[1], VideoSpec):
            source_id = str(args[0])
            spec = args[1]
        elif len(args) == 1 and isinstance(args[0], VideoSpec):
            spec = args[0]
        else:
            if "world_id" in kwargs:
                source_id = str(kwargs["world_id"])
            elif "source_id" in kwargs:
                source_id = str(kwargs["source_id"])
            if "spec" in kwargs and isinstance(kwargs["spec"], VideoSpec):
                spec = kwargs["spec"]
        engine = RecordedVideoEngine(
            spec,
            mp4_path=mp4_path,
            rgb_frames=rgb_frames,
            max_frames=max_frames,
            capture_fps=capture_fps,
            realtime_pacing=realtime_pacing,
            pacing_speed=pacing_speed,
            fast_frame_delay_s=fast_frame_delay_s,
            loop=loop,
            source_id=source_id,
            session_id=session_id,
            stream_id=stream_id,
            epoch=epoch,
            media_generation=media_generation,
            prefer_ffmpeg=prefer_ffmpeg,
        )
        if on_created is not None:
            on_created(engine)
        return engine

    return _factory
