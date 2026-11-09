# SPDX-License-Identifier: Apache-2.0
"""Low-delay elementary video access units, independent of archive encoders."""

from __future__ import annotations

import re
import threading
from fractions import Fraction

from .contracts import VideoSpec
from .source_encoding import (
    default_preset,
    default_quality,
    encoder_implementation,
    validate_source_encoding,
)

# The native RTC sender negotiates Baseline Level 3.1. Keep the legacy capture
# envelope where it fits, and use the source's actual rate for larger canvases.
H264_LEVEL31_MAX_MACROBLOCKS = 3600
H264_LEVEL31_MAX_MACROBLOCKS_PER_SECOND = 108_000


def level31_supports(spec: VideoSpec, fps: int) -> bool:
    macroblocks = ((spec.width + 15) // 16) * ((spec.height + 15) // 16)
    return (
        macroblocks <= H264_LEVEL31_MAX_MACROBLOCKS
        and macroblocks * fps <= H264_LEVEL31_MAX_MACROBLOCKS_PER_SECOND
    )


def capture_rate_hint(spec: VideoSpec, fps: int) -> int:
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError("H.264 capture rate must be between 1 and 60 FPS")
    return 60 if level31_supports(spec, 60) else fps


class H264PlaybackEncoder:
    """One Annex B access unit per frame, without B-frame reordering.

    Capture-clock mode retains one frame to measure its duration; push/drain
    return explicit identities so its metadata and EOS tail remain aligned.

    The lock also fences cleanup against a to_thread encode still running when
    its asyncio caller is cancelled. Frames are encoded off the event loop.
    """

    def __init__(
        self,
        spec: VideoSpec,
        *,
        max_bitrate: int = 1_500_000,
        gop: int | None = None,
        capture_clock: bool = False,
        immediate_capture: bool = False,
        capture_fps: int = 60,
        intra_refresh: bool = False,
        codec: str = "h264",
        vbv_ms: int = 250,
        preset: str | None = None,
        quality: int | None = None,
        min_qp: int = 0,
    ):
        import av

        if type(max_bitrate) is not int or not 100_000 <= max_bitrate <= 50_000_000:
            raise ValueError("Playback bitrate must be between 100 Kbps and 50 Mbps")
        if type(capture_fps) is not int or not 1 <= capture_fps <= 60:
            raise ValueError("H.264 capture rate must be between 1 and 60 FPS")
        self.spec, self.count = spec, 0
        # The caller supplies the source capture cadence independently of the
        # model's VideoSpec.fps. H.264 may retain its legacy 60-FPS hint.
        self.capture_fps = capture_fps
        self.max_bitrate = self.target_bitrate = self.context_bitrate = max_bitrate
        self.capture_clock = capture_clock
        if type(immediate_capture) is not bool or (
            immediate_capture and not capture_clock
        ):
            raise ValueError("Immediate H.264 encoding requires the capture clock")
        self.immediate_capture = immediate_capture
        if type(intra_refresh) is not bool or (
            intra_refresh and (not capture_clock or immediate_capture)
        ):
            raise ValueError(
                "Source intra refresh requires paired capture-clock encoding"
            )
        settings = validate_source_encoding(
            dict(
                codec=codec,
                bitrate=max_bitrate,
                gop=gop if gop is not None else min(spec.fps, 24),
                vbvMs=vbv_ms,
                preset=default_preset(codec) if preset is None else preset,
                quality=default_quality(codec) if quality is None else quality,
                intraRefresh=intra_refresh,
                minQp=min_qp,
            )
        )
        self.encoding_settings = settings
        self.codec_name = codec
        self.intra_refresh = intra_refresh
        self.native = None
        self.output_count = 0
        self.capture_origin = self.last_capture = None
        self.last_interval_us = (
            spec.pts(1) if codec == "h264" else 1_000_000 // self.capture_fps
        )
        self.pending_timing = {}
        self.drained = False
        if gop is not None and (type(gop) is not int or not 1 <= gop <= 24):
            raise ValueError("Invalid GOP")
        self.independent_frames = False
        self.force_keyframe = threading.Event()
        self.lock = threading.Lock()
        self.context = av.CodecContext.create(encoder_implementation(codec), "w")
        context = self.context
        context.width, context.height = spec.width, spec.height
        context.pix_fmt = "yuv420p"
        context.time_base = Fraction(1, 1_000_000 if capture_clock else spec.fps)
        # This is a real source-rate hint, not a timestamp conversion or a frame
        # limiter. VFR rate control still uses every frame's actual PTS delta.
        context.framerate = Fraction(self.capture_fps if capture_clock else spec.fps)
        context.thread_count = 1
        # Expose the real SPS profile/level at negotiation, while repeat-headers
        # retains SPS/PPS inside every key access unit for Annex B decoding.
        context.flags |= av.codec.context.Flags.global_header
        context.options = {
            "preset": settings["preset"],
            "tune": "zerolatency",
            "profile": "baseline",
            "crf": str(settings["quality"]),
            "bf": "0",
            "g": str(gop or spec.fps),
            "x264-params": (
                "repeat-headers=1:annexb=1:scenecut=0:rc-lookahead=0:sync-lookahead=0"
                f":vbv-maxrate={max_bitrate // 1000}:vbv-bufsize={max_bitrate * vbv_ms // 1_000_000}"
                + (":force-cfr=0" if capture_clock else "")
            ),
        }
        if codec != "h264":
            context.thread_count = 2
            context.flags &= ~av.codec.context.Flags.global_header
            context.gop_size = settings["gop"]
            context.bit_rate = max_bitrate
            if codec == "vp9":
                context.options = {
                    "deadline": "realtime",
                    "cpu-used": settings["preset"],
                    "lag-in-frames": "0",
                    "auto-alt-ref": "0",
                    # Equal minimum/maximum/target selects libvpx CBR. CRF is
                    # ignored in that mode, so expose the real maximum QP.
                    "minrate": str(max_bitrate),
                    "maxrate": str(max_bitrate),
                    "qmax": str(settings["quality"]),
                    "bufsize": str(max_bitrate * vbv_ms // 1000),
                    "max-intra-rate": "300",
                    "row-mt": "1",
                    "drop-threshold": "0",
                }
                # Zero retains the established v2 default option set.
                if settings["minQp"]:
                    context.options["qmin"] = str(settings["minQp"])
            else:
                context.options = {
                    "preset": settings["preset"],
                    # Unsupported SVT parameters must fail startup probing,
                    # rather than silently ignoring the realtime policy.
                    "err_detect": "explode",
                    # SVT >=3 uses lp as a parallelism level, not a thread cap.
                    "svtav1-params": f"rtc=1:pred-struct=1:lookahead=0:lp=2:rc=2:max-qp={settings['quality']}:buf-sz={vbv_ms}"
                    f":buf-initial-sz={max(1, vbv_ms // 2)}:buf-optimal-sz={max(1, vbv_ms // 2)}",
                }
                if settings["minQp"]:
                    context.options["svtav1-params"] += f":min-qp={settings['minQp']}"
        if intra_refresh:
            context.options["x264-params"] += ":intra-refresh=1"
        if immediate_capture and codec != "h264":
            raise ValueError("Native immediate capture only supports H.264")
        if immediate_capture:
            from .native_capture import NativeCaptureEncoder

            self.native = NativeCaptureEncoder(
                spec, gop=gop or spec.fps, max_bitrate=max_bitrate
            )
            extra = self.native.headers
        else:
            context.open()
            extra = context.extradata or b""
        units = re.split(b"\x00\x00\x00?\x01", extra)
        sps = next((unit for unit in units if unit and unit[0] & 31 == 7), None)
        if codec == "h264" and (sps is None or len(sps) < 4):
            if self.native is not None:
                self.native.close()
            raise RuntimeError("H.264 encoder did not provide an Annex B SPS")
        self.stream = dict(
            format="h264",
            bitstream_format="annexb",
            codec="avc1." + sps[1:4].hex()
            if codec == "h264"
            else ("vp09.00.31.08" if codec == "vp9" else "av01.0.05M.08"),
            width=spec.width,
            height=spec.height,
            fps=spec.fps,
            gop=gop or spec.fps,
            max_bitrate=max_bitrate,
            vbv_buffer_ms=vbv_ms,
            encoder="libx264",
            profile="baseline",
            pixel_format="yuv420p",
            preset=settings["preset"],
            tune="zerolatency",
            crf=settings["quality"],
            b_frames=0,
            vbv_buffer_bits=max_bitrate * vbv_ms // 1000,
            implementation="room-paired-au-v1",
            rate_control_clock="source_capture_vfr" if capture_clock else "model_cfr",
            encoder_retained_frames=0 if immediate_capture else int(capture_clock),
            max_encoding_fps=self.capture_fps if capture_clock else spec.fps,
        )
        if codec != "h264":
            self.stream.update(
                format=codec,
                bitstream_format="raw" if codec == "vp9" else "obu",
                encoder=encoder_implementation(codec),
                fps=self.capture_fps if capture_clock else spec.fps,
                profile="0",
                tune="realtime",
                encoder_retained_frames=0,
                quality=settings["quality"],
                quality_semantics="cbr-max-qp",
                rate_control="cbr",
                rate_control_fps=float(context.framerate),
                duration_policy="known_or_previous_capture_interval",
                encoding_policy="source-cbr-low-delay-v2",
                qmax=settings["quality"],
                rate_control_target_bitrate=max_bitrate,
            )
            self.stream.pop("crf", None)
            if codec == "vp9":
                self.stream.update(
                    qmin=settings["minQp"],
                    max_intra_rate_pct=300,
                    software_threads=2,
                    row_mt=True,
                    min_bitrate=max_bitrate,
                )
            else:
                self.stream.update(
                    svt_rtc=True, svt_parallelism_level=2, qmin=settings["minQp"]
                )
        if intra_refresh:
            self.stream["intra_refresh"] = True
        if immediate_capture:
            self.stream.update(
                implementation="room-immediate-au-v1",
                rate_control_clock="elapsed_source_capture",
                duration_policy="archive_next_pts_live_known_or_previous_interval",
                encoder_version=self.native.version,
            )

    def request_keyframe(self):
        self.force_keyframe.set()

    def set_target_bitrate(self, bitrate):
        """Apply this RTC session's native congestion feedback below its ceiling.

        WS never invokes this API. Each new session starts at its configured
        ceiling; feedback cannot change a deployment or another encoder.
        """
        if type(bitrate) is not int or not 100_000 <= bitrate <= self.max_bitrate:
            raise ValueError("The H.264 target exceeds the configured bitrate range")
        with self.lock:
            if self.context is None or self.drained:
                return False
            self.target_bitrate = bitrate
            if self.native is not None:
                self.native.set_target_bitrate(bitrate)
            return True

    def encode(self, index: int, rgb: bytes) -> dict:
        import av
        import numpy as np

        with self.lock:
            if self.capture_clock:
                raise ValueError("Capture-clock encoding requires push and drain")
            if self.context is None or index != self.count:
                raise ValueError("H.264 playback frames must be contiguous")
            if len(rgb) != self.spec.frame_bytes:
                raise ValueError("H.264 playback requires same-size RGB24 frames")
            pixels = np.frombuffer(rgb, dtype=np.uint8).reshape(
                self.spec.height, self.spec.width, 3
            )
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, self.spec.fps)
            if self.independent_frames or self.force_keyframe.is_set():
                self.force_keyframe.clear()
                # Experimental lossy WebRTC data transport: each delivered access
                # unit is decodable even if its predecessor was lost.
                frame.pict_type = av.video.frame.PictureType.I
            packets = self.context.encode(frame)
            if len(packets) != 1 or packets[0].pts != index or packets[0].dts != index:
                raise RuntimeError(
                    "H.264 playback encoder buffered or reordered frames"
                )
            packet = packets[0]
            self.count += 1
            self.output_count += 1
            return dict(
                encoding=self.codec_name,
                content_type="video/" + self.codec_name,
                bitstream_format=self.stream["bitstream_format"],
                codec=self.stream["codec"],
                keyframe=packet.is_keyframe,
                duration_us=self.spec.pts(index + 1) - self.spec.pts(index),
                payload=bytes(packet),
            )

    def push(
        self, index: int, rgb: bytes, capture_pts_us: int, *, duration_hint_us=None
    ) -> list[tuple[int, dict]]:
        """Encode on the source clock, retaining at most one AU for its duration.

        Return the encoded frame's identity, which may precede this input. The
        caller must retain matching metadata until it is returned, including EOS.
        Model input timing and archive/source identities are independent of this
        rate-control clock. No frame is duplicated or discarded.
        """
        if not self.capture_clock:
            return [(index, self.encode(index, rgb))]
        import av
        import numpy as np

        with self.lock:
            if self.context is None or self.drained or index != self.count:
                raise ValueError("H.264 playback frames must be contiguous")
            if (
                type(capture_pts_us) is not int
                or capture_pts_us < 0
                or (
                    self.last_capture is not None
                    and capture_pts_us <= self.last_capture
                )
            ):
                raise ValueError("H.264 capture timestamps must increase")
            if len(rgb) != self.spec.frame_bytes:
                raise ValueError("H.264 playback requires same-size RGB24 frames")
            if self.capture_origin is None:
                self.capture_origin = capture_pts_us
            if self.last_capture is not None:
                self.last_interval_us = capture_pts_us - self.last_capture
            pts = capture_pts_us - self.capture_origin
            if self.immediate_capture:
                if duration_hint_us is not None and (
                    type(duration_hint_us) is not int
                    or not 0 < duration_hint_us < 2**63
                ):
                    raise ValueError("Invalid known source-frame duration")
                forced = self.independent_frames or self.force_keyframe.is_set()
                self.force_keyframe.clear()
                payload, keyframe = self.native.encode(
                    rgb,
                    pts,
                    keyframe=forced,
                    duration_hint_us=duration_hint_us,
                )
                self.count += 1
                self.output_count += 1
                self.last_capture = capture_pts_us
                return [
                    (
                        index,
                        dict(
                            encoding=self.codec_name,
                            content_type="video/" + self.codec_name,
                            bitstream_format="annexb",
                            codec=self.stream["codec"],
                            keyframe=keyframe,
                            payload=payload,
                            duration_us=duration_hint_us or self.last_interval_us,
                            duration_provisional=True,
                            duration_hint_exact=duration_hint_us is not None,
                        ),
                    )
                ]
            retained = []
            if (
                self.target_bitrate != self.context_bitrate
                and index % self.stream["gop"] == 0
            ):
                # PyAV does not expose mutable VBV maxrate. Rollover at the
                # existing GOP boundary, drain the old AU and start with IDR.
                # Source PTS and output identities remain continuous.
                replacement = H264PlaybackEncoder(
                    self.spec,
                    max_bitrate=self.target_bitrate,
                    gop=self.stream["gop"],
                    capture_clock=True,
                    capture_fps=self.capture_fps,
                    **({"intra_refresh": True} if self.intra_refresh else {}),
                    codec=self.codec_name,
                    vbv_ms=self.encoding_settings["vbvMs"],
                    preset=self.encoding_settings["preset"],
                    quality=self.encoding_settings["quality"],
                    min_qp=self.encoding_settings.get("minQp", 0),
                )
                retained = self.context.encode(None)
                self.context = replacement.context
                replacement.context = None
                self.context_bitrate = self.target_bitrate
                if self.codec_name != "h264":
                    self.stream["rate_control_target_bitrate"] = self.context_bitrate
                    if self.codec_name == "vp9":
                        self.stream["min_bitrate"] = self.context_bitrate
            self.pending_timing[pts] = index
            pixels = np.frombuffer(rgb, dtype=np.uint8).reshape(
                self.spec.height, self.spec.width, 3
            )
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = pts, self.context.time_base
            if self.codec_name != "h264":
                if duration_hint_us is not None and (
                    type(duration_hint_us) is not int
                    or not 0 < duration_hint_us < 2**63
                ):
                    raise ValueError("Invalid known source-frame duration")
                # libvpx uses AVFrame.duration for rate control; do not charge
                # every real ~22-FPS frame against the legacy 60-FPS budget.
                # SVT keeps its declared framerate; both preserve original PTS.
                frame.duration = duration_hint_us or self.last_interval_us
            if self.independent_frames or self.force_keyframe.is_set():
                self.force_keyframe.clear()
                frame.pict_type = av.video.frame.PictureType.I
            self.count += 1
            self.last_capture = capture_pts_us
            result = self._timed_packets([*retained, *self.context.encode(frame)])
            if len(self.pending_timing) > 1:
                raise RuntimeError("H.264 capture encoder retained more than one frame")
            return result

    def _timed_packets(self, packets):
        result = []
        for packet in packets:
            if packet.pts != packet.dts or packet.pts not in self.pending_timing:
                raise RuntimeError("H.264 capture encoder reordered frames")
            index = self.pending_timing.pop(packet.pts)
            if index != self.output_count:
                raise RuntimeError("H.264 capture encoder skipped a frame")
            duration = (
                min(self.pending_timing) - packet.pts
                if self.pending_timing
                else self.last_interval_us
            )
            payload = bytes(packet)
            # Recovery-point pictures may carry AV_PKT_FLAG_KEY without IDR.
            # Consumers use keyframe to reopen a dropped reference chain, so
            # this opt-in path advertises only immediate random-access AUs.
            keyframe = packet.is_keyframe
            if self.intra_refresh:
                keyframe = any(
                    unit and unit[0] & 31 == 5
                    for unit in re.split(b"\x00\x00\x00?\x01", payload)
                )
            result.append(
                (
                    index,
                    dict(
                        encoding=self.codec_name,
                        content_type="video/" + self.codec_name,
                        bitstream_format=self.stream["bitstream_format"],
                        codec=self.stream["codec"],
                        keyframe=keyframe,
                        duration_us=duration,
                        payload=payload,
                    ),
                )
            )
            self.output_count += 1
        return result

    def drain(self) -> list[tuple[int, dict]]:
        """Return the retained final AU once, before successful stream completion."""
        if not self.capture_clock:
            self.finish()
            return []
        with self.lock:
            if self.drained:
                return []
            if self.context is None:
                raise ValueError("The H.264 encoder is closed")
            if self.immediate_capture:
                self.drained = True
                if self.output_count != self.count:
                    raise RuntimeError("The H.264 encoder lost a frame")
                return []
            result = self._timed_packets(self.context.encode(None))
            self.drained = True
            if self.pending_timing or self.output_count != self.count:
                raise RuntimeError("H.264 capture encoder did not drain completely")
            return result

    def finish(self) -> None:
        with self.lock:
            if self.native is not None:
                self.native.close()
                self.context = None
                return
            if self.context is not None and self.context.encode(None):
                raise RuntimeError("H.264 playback has unexpected delayed frames")
            self.context = None

    def close(self) -> None:
        with self.lock:
            if self.native is not None:
                self.native.close()
            self.context = None
            self.pending_timing.clear()


def create_source_playback_encoder(spec: VideoSpec, *, settings=None, **kwargs):
    """Keep source codec selection separate from the H.264 model encoder owner."""
    if settings is not None:
        selected = validate_source_encoding(settings)
        kwargs.update(
            codec=selected["codec"],
            max_bitrate=selected["bitrate"],
            gop=selected["gop"],
            vbv_ms=selected["vbvMs"],
            preset=selected["preset"],
            quality=selected["quality"],
            min_qp=selected.get("minQp", 0),
            intra_refresh=selected["intraRefresh"],
        )
    return H264PlaybackEncoder(spec, **kwargs)
