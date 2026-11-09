# SPDX-License-Identifier: Apache-2.0
"""Worker-owned, fixed-budget RGB encoding, independent of live transport."""

from __future__ import annotations

import re
import time
from fractions import Fraction


class ArchiveVideoEncoder:
    """Use the codec's one-frame VFR lookahead and retain its exact identity.

    There is intentionally no transport-feedback or bitrate-update method.
    Each packet keeps its input identity; paired muxers use the same source PTS.
    """

    def __init__(self, spec, *, bitrate=4_000_000, vbv_ms=2000, keyframe_ms=1000):
        import av
        import numpy as np

        self.np = np  # Import before worker boot ACK, not during the first RGB.
        if type(bitrate) is not int or not 100_000 <= bitrate <= 50_000_000:
            raise ValueError("Invalid archive bitrate")
        if not 250 <= vbv_ms <= 5000 or not 250 <= keyframe_ms <= 5000:
            raise ValueError("Invalid archive encoding interval")
        self.spec = spec
        self.keyframe_us = keyframe_ms * 1000
        self.count = self.output_count = 0
        self.last_pts = self.last_key_pts = None
        self.last_duration = spec.pts(1)
        self.timings = {}
        self.drained = False
        self.last_encode_ms = 0.0
        self.context = av.CodecContext.create("libx264", "w")
        context = self.context
        context.width, context.height = spec.width, spec.height
        context.pix_fmt = "yuv420p"
        context.time_base = Fraction(1, 1_000_000)
        # SPS level must cover every accepted CLI capture rate; VFR rate
        # control and MP4 timestamps still use the actual source-clock deltas.
        context.framerate = Fraction(60)
        context.bit_rate = bitrate
        context.thread_count = 1
        context.options = {
            "preset": "veryfast",
            "tune": "zerolatency",
            "profile": "baseline",
            "bf": "0",
            "forced-idr": "1",
            "g": "300",
            "x264-params": (
                "repeat-headers=1:annexb=1:scenecut=0:rc-lookahead=0:sync-lookahead=0"
                ":force-cfr=0"
                f":vbv-maxrate={bitrate // 1000}"
                f":vbv-bufsize={bitrate * vbv_ms // 1_000_000}"
            ),
        }
        context.open()
        self.stream = dict(
            format="h264",
            bitstream_format="annexb",
            width=spec.width,
            height=spec.height,
            fps=spec.fps,
            encoder="libx264",
            profile="baseline",
            pixel_format="yuv420p",
            preset="veryfast",
            tune="zerolatency",
            b_frames=0,
            rate_control="abr",
            target_bitrate=bitrate,
            max_bitrate=bitrate,
            vbv_buffer_ms=vbv_ms,
            vbv_buffer_bits=bitrate * vbv_ms // 1000,
            keyframe_interval_ms=keyframe_ms,
            rate_control_clock="source_capture_vfr",
            encoder_retained_frames=1,
            implementation="independent-rgb-v2",
            reused_playback_bitstream=False,
            network_adaptive=False,
            encoder_parameters_verified=False,
            encoder_version="libavcodec-"
            + ".".join(map(str, av.library_versions["libavcodec"])),
        )

    def _packets(self, packets):
        output = []
        for packet in packets:
            if packet.pts != packet.dts or packet.pts not in self.timings:
                raise ValueError("Archive encoder changed frame identity")
            index, duration = self.timings.pop(packet.pts)
            if index != self.output_count:
                raise ValueError("Archive encoder reordered frames")
            payload = bytes(packet)
            if index == 0:
                values = dict(
                    re.findall(
                        rb"\b(rc|bitrate|vbv_maxrate|vbv_bufsize)=([a-z0-9]+)", payload
                    )
                )
                expected = {
                    b"bitrate": str(self.stream["target_bitrate"] // 1000).encode(),
                    b"vbv_maxrate": str(self.stream["max_bitrate"] // 1000).encode(),
                    b"vbv_bufsize": str(
                        self.stream["vbv_buffer_bits"] // 1000
                    ).encode(),
                }
                if values.get(b"rc") not in (b"abr", b"cbr") or any(
                    values.get(k) != v for k, v in expected.items()
                ):
                    raise ValueError("Archive encoder did not apply its bitrate budget")
                # x264 describes ABR with an equal VBV ceiling as 'cbr'. No
                # filler is enabled: simple scenes need not consume the budget.
                self.stream.update(
                    encoder_parameters_verified=True,
                    x264_rate_control=values[b"rc"].decode(),
                )
            self.output_count += 1
            output.append(
                (
                    index,
                    dict(
                        encoding="h264",
                        payload=payload,
                        keyframe=packet.is_keyframe,
                        duration_us=duration,
                        duration_provisional=True,
                        pts_us=packet.pts,
                    ),
                )
            )
        return output

    def push(self, index, rgb, pts_us):
        import av

        started = time.perf_counter()
        if self.drained or index != self.count or len(rgb) != self.spec.frame_bytes:
            raise ValueError("Archive RGB frames must be contiguous")
        if (
            type(pts_us) is not int
            or pts_us < 0
            or (self.last_pts is not None and pts_us <= self.last_pts)
        ):
            raise ValueError("Archive source timestamps must increase")
        if self.last_pts is not None:
            self.last_duration = pts_us - self.last_pts
            if self.last_pts in self.timings:
                old_index, _ = self.timings[self.last_pts]
                self.timings[self.last_pts] = old_index, self.last_duration
        self.timings[pts_us] = index, self.last_duration
        frame = av.VideoFrame.from_ndarray(
            self.np.frombuffer(rgb, dtype=self.np.uint8).reshape(
                self.spec.height, self.spec.width, 3
            ),
            format="rgb24",
        )
        frame.pts, frame.time_base = pts_us, self.context.time_base
        frame.duration = self.last_duration
        if self.last_key_pts is None or pts_us - self.last_key_pts >= self.keyframe_us:
            frame.pict_type = av.video.frame.PictureType.I
            self.last_key_pts = pts_us
        output = self._packets(self.context.encode(frame))
        self.last_pts = pts_us
        self.count += 1
        if self.count - self.output_count > 1:
            raise ValueError("Archive encoder exceeded its frame budget")
        self.last_encode_ms = (time.perf_counter() - started) * 1000
        return output

    def drain(self):
        if self.drained:
            return []
        self.drained = True
        output = self._packets(self.context.encode(None))
        if self.timings or self.count != self.output_count:
            raise ValueError("Archive encoder lost frames")
        return output

    def close(self):
        self.timings.clear()
        self.context = None
        self.drained = True
