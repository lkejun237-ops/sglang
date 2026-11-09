# SPDX-License-Identifier: Apache-2.0
"""Immediate access units with a source-clock byte budget.

The bridge uses only x264's public API. Its fixed tick removes x264's VFR
lookahead; per-picture VBV replenishment represents elapsed source time.
Presentation timestamps are carried independently, unchanged, by the protocol.
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def library():
    path = Path(
        os.environ.get(
            "WHITEBOX_X264_LIBRARY",
            str(Path(__file__).with_name("native") / "libwhitebox_x264.so"),
        )
    )
    if not path.is_file():
        raise RuntimeError("The immediate H.264 encoder library is unavailable")
    lib = ctypes.CDLL(str(path))
    lib.wb_abi.argtypes = []
    lib.wb_abi.restype = ctypes.c_int
    if lib.wb_abi() != 3:
        raise RuntimeError("The immediate H.264 encoder ABI is unsupported")
    lib.wb_version.argtypes = []
    lib.wb_version.restype = ctypes.c_char_p
    lib.wb_open.argtypes = [ctypes.c_int] * 4
    lib.wb_open.restype = ctypes.c_void_p
    lib.wb_headers.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.wb_headers.restype = ctypes.c_int
    lib.wb_encode.argtypes = (
        [ctypes.c_void_p] * 4
        + [ctypes.c_int] * 3
        + [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int64),
        ]
    )
    lib.wb_encode.restype = ctypes.c_int
    lib.wb_close.argtypes = [ctypes.c_void_p]
    lib.wb_close.restype = None
    lib.wb_bitrate.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.wb_bitrate.restype = ctypes.c_int
    return lib


class NativeCaptureEncoder:
    """Called under H264PlaybackEncoder's lifecycle/encode lock."""

    def __init__(self, spec, *, gop, max_bitrate):
        self.spec = spec
        self.lib = library()
        self.handle = self.lib.wb_open(spec.width, spec.height, gop, max_bitrate)
        if not self.handle:
            raise RuntimeError("The immediate H.264 encoder could not start")
        try:
            pointer = ctypes.c_void_p()
            size = self.lib.wb_headers(self.handle, ctypes.byref(pointer))
            if size <= 0 or size > 1 << 20:
                raise RuntimeError("The H.264 encoder returned invalid headers")
            self.headers = ctypes.string_at(pointer, size)
            self.version = self.lib.wb_version().decode("ascii")
        except BaseException:
            self.close()
            raise

    def encode(self, rgb, pts, *, keyframe, duration_hint_us=None):
        import av
        import numpy as np

        if not self.handle:
            raise ValueError("The H.264 encoder is closed")
        # PyAV owns the plane allocations through the synchronous native call.
        # CDLL releases the GIL; RGB conversion/encoding run in to_thread callers.
        pixels = np.frombuffer(rgb, dtype=np.uint8).reshape(
            self.spec.height, self.spec.width, 3
        )
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24").reformat(
            format="yuv420p"
        )
        pointer, is_keyframe, timestamp = (
            ctypes.c_void_p(),
            ctypes.c_int(),
            ctypes.c_int64(),
        )
        size = self.lib.wb_encode(
            self.handle,
            *(plane.buffer_ptr for plane in frame.planes),
            *(plane.line_size for plane in frame.planes),
            pts,
            duration_hint_us if duration_hint_us is not None else -1,
            int(keyframe),
            ctypes.byref(pointer),
            ctypes.byref(is_keyframe),
            ctypes.byref(timestamp),
        )
        if size <= 0 or timestamp.value != pts:
            raise RuntimeError("The H.264 encoder buffered or reordered a frame")
        return ctypes.string_at(pointer, size), bool(is_keyframe.value)

    def close(self):
        if self.handle:
            self.lib.wb_close(self.handle)
            self.handle = None

    def set_target_bitrate(self, bitrate):
        if not self.handle or self.lib.wb_bitrate(self.handle, bitrate) != 0:
            raise ValueError("The H.264 target bitrate is invalid")
