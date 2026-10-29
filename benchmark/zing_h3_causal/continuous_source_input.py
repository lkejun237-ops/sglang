# SPDX-License-Identifier: Apache-2.0
"""Incremental RGB readers; closing a session also stops its video decoder."""

import asyncio
import sys
from contextlib import asynccontextmanager


async def read_packet(reader, size):
    """Pipes may split frames across reads; only EOF permits a short packet."""
    try:
        return await reader.readexactly(size)
    except asyncio.IncompleteReadError as exc:
        return exc.partial


def source_dimensions(args):
    width = getattr(args, "source_width", None)
    height = getattr(args, "source_height", None)
    if (width is None) != (height is None):
        raise ValueError("Specify source width and height together")
    if width is None:
        width, height = args.width, args.height
    if min(width, height) < 1:
        raise ValueError("Source dimensions must be positive")
    return width, height


@asynccontextmanager
async def open_source(args, frame_bytes, stderr_path):
    if getattr(args, "source_video", None):
        width, height = source_dimensions(args)
        with stderr_path.open("wb") as stderr:
            process = await asyncio.create_subprocess_exec(
                args.ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-i",
                args.source_video,
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                f"scale={width}:{height}:flags=bilinear",
                "-vsync",
                "0",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "pipe:1",
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr,
                limit=2 * frame_bytes,
            )

            async def read(size):
                data = await read_packet(process.stdout, size)
                if len(data) < size and await process.wait() != 0:
                    raise RuntimeError(f"Video decode failed; see {stderr_path}")
                return data

            try:
                yield read
            finally:
                if process.returncode is None:
                    process.kill()
                # Drain the bounded stdout pipe so process cleanup cannot wait
                # forever on a paused reader after cancellation or disconnect.
                await process.communicate()
    elif args.source_rgb == "-":
        reader = asyncio.StreamReader(limit=2 * frame_bytes)
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer
        )
        try:
            yield lambda size: read_packet(reader, size)
        finally:
            transport.close()
    else:
        with open(args.source_rgb, "rb") as source:
            yield lambda size: asyncio.to_thread(source.read, size)
