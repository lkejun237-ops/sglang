# SPDX-License-Identifier: Apache-2.0
"""Bounded source-frame retention and immutable per-source reference assets."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
from collections import deque
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from .contracts import capture_pts
from .render_contract import MAX_REFERENCE_BYTES, MAX_REFERENCE_IMAGES, RenderError


def durable_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as target:
            json.dump(value, target, ensure_ascii=False, separators=(",", ":"))
            target.flush()
            os.fsync(target.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


class SourceFrames:
    def __init__(self, max_bytes=56 << 20):
        self.max_bytes, self.bytes = max_bytes, 0
        self.frames = deque()

    def append(self, version, frame, captured):
        import msgspec

        metadata = len(msgspec.msgpack.encode(replace(frame, rgb=b"")))
        size = len(frame.rgb) + metadata * 4 + 65536
        # The remaining 8 MiB of the public 64 MiB budget holds at most one
        # pinned start frame. Reject oversized metadata before retaining it.
        if size > 8 << 20:
            raise RenderError("LIMIT_EXCEEDED")
        if size > self.max_bytes:
            raise RenderError("LIMIT_EXCEEDED")
        if self.frames and (
            frame.index <= self.frames[-1][1].index
            or capture_pts(frame) <= capture_pts(self.frames[-1][1])
        ):
            raise RenderError("SOURCE_EPOCH_MISMATCH")
        while self.frames and self.bytes + size > self.max_bytes:
            self.bytes -= self.frames.popleft()[3]
        self.frames.append((version, frame, captured, size))
        self.bytes += size

    def exact(self, boundary):
        for version, frame, captured, _ in self.frames:
            if boundary.get("source_frame_index") == frame.index or boundary.get(
                "source_pts_us"
            ) == capture_pts(frame):
                return version, (frame, captured)
        return None

    def clear(self):
        self.frames.clear()
        self.bytes = 0

    def scene_frame(self, header):
        """Find the exact CLI frame acknowledged by scene publication."""
        for version, frame, captured, _ in self.frames:
            if (frame.provenance or {}).get("cli_frame_header") == header:
                return version, (frame, captured)
        return None


class SourceAssets:
    def __init__(
        self,
        directory,
        *,
        max_bytes=3 * MAX_REFERENCE_BYTES,
        max_items=512,
        memory=False,
    ):
        self.directory = Path(directory) / "render-assets"
        self.memory = memory
        self.payloads = {}
        if not memory:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_bytes, self.max_items = max_bytes, max_items
        self.assets, self.bytes = {}, 0
        self.lock = asyncio.Lock()

    def release_memory(self):
        if self.memory:
            self.payloads.clear()
            self.assets.clear()
            self.bytes = 0

    async def upload(self, format, payload):
        if (
            format not in {"png", "jpeg", "webp"}
            or not isinstance(payload, bytes)
            or not 0 < len(payload) <= 8 << 20
        ):
            raise RenderError("CONDITION_INVALID")
        digest = hashlib.sha256(payload).hexdigest()
        asset_id = "sha256:" + digest
        async with self.lock:
            if asset_id in self.assets:
                return dict(self.assets[asset_id], duplicate=True)
            if (
                len(self.assets) >= self.max_items
                or self.bytes + len(payload) > self.max_bytes
            ):
                raise RenderError("LIMIT_EXCEEDED")

            def write():
                from PIL import Image

                try:
                    with Image.open(io.BytesIO(payload)) as image:
                        if (
                            image.format.lower() != format
                            or image.width * image.height > 16_777_216
                        ):
                            raise RenderError("CONDITION_INVALID")
                        width, height = image.size
                        image.verify()
                except RenderError:
                    raise
                except Exception as error:
                    raise RenderError("CONDITION_INVALID") from error
                name = digest + "." + format
                target = self.directory / name
                if not self.memory:
                    if target.exists():
                        if (
                            target.is_symlink()
                            or hashlib.sha256(target.read_bytes()).hexdigest() != digest
                        ):
                            raise RenderError("CONDITION_INVALID")
                    else:
                        temporary = target.with_name(name + "." + uuid4().hex + ".tmp")
                        try:
                            with temporary.open("xb") as output:
                                output.write(payload)
                                output.flush()
                                os.fsync(output.fileno())
                            temporary.chmod(0o600)
                            os.replace(temporary, target)
                        finally:
                            temporary.unlink(missing_ok=True)
                return dict(
                    asset_id=asset_id,
                    format=format,
                    sha256=digest,
                    bytes=len(payload),
                    width=width,
                    height=height,
                    path=name,
                )

            async def commit():
                descriptor = await asyncio.to_thread(write)
                if self.memory:
                    self.payloads[asset_id] = payload
                else:
                    await asyncio.to_thread(
                        durable_json,
                        self.directory / "assets.json",
                        {**self.assets, asset_id: descriptor},
                    )
                self.assets[asset_id] = descriptor
                self.bytes += len(payload)
                return dict(descriptor)

            # A withdrawn condition must not unlock a still-running disk/thread
            # commit. Finish the asset and metadata atomically before propagating
            # cancellation, including a second cancellation during source close.
            task = asyncio.create_task(commit())
            cancelled = False
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
                except Exception:
                    if not cancelled:
                        raise
            if cancelled:
                task.exception()
                raise asyncio.CancelledError
            return task.result()

    async def resolve(self, references):
        if not isinstance(references, list) or len(references) > MAX_REFERENCE_IMAGES:
            raise RenderError("LIMIT_EXCEEDED")
        selected = []
        total = 0
        for ref in references:
            descriptor = self.assets.get(ref.get("asset_id"))
            if descriptor is None:
                raise RenderError("CONDITION_INVALID")
            total += descriptor["bytes"]
            if total > MAX_REFERENCE_BYTES:
                raise RenderError("LIMIT_EXCEEDED")
            selected.append((ref, descriptor))

        def read():
            result = []
            for ref, descriptor in selected:
                if self.memory:
                    payload = self.payloads[descriptor["asset_id"]]
                else:
                    path = self.directory / descriptor["path"]
                    if path.is_symlink():
                        raise RenderError("CONDITION_INVALID")
                    payload = path.read_bytes()
                if (
                    len(payload) != descriptor["bytes"]
                    or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
                ):
                    raise RenderError("CONDITION_INVALID")
                result.append(
                    dict(id=ref["id"], format=descriptor["format"], payload=payload)
                )
            return result

        return await asyncio.to_thread(read)
