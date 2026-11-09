# SPDX-License-Identifier: Apache-2.0
"""Signed, opt-in source-only RTC settings; omission preserves deployment defaults."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace

from .source_encoding import validate_source_encoding


def canonical_source_encoding(value):
    """Wire settings must already be normalized, not silently repaired after signing."""
    result = validate_source_encoding(value)
    if json.dumps(value, sort_keys=True, allow_nan=False) != json.dumps(
        result, sort_keys=True, allow_nan=False
    ):
        raise ValueError("Source encoder settings must be canonical")
    return result


def requested_source_encoding(world):
    if "source_encoding" not in world:
        return None
    return canonical_source_encoding(world["source_encoding"])


def bind_signed_source_encoding(claims, world):
    """Called before spending a ticket. Both absence and exact settings are signed."""
    if ("source_encoding" in claims) != ("source_encoding" in world):
        raise ValueError("Source encoder authorization does not match")
    selected = requested_source_encoding(world)
    if selected is not None:
        signed = canonical_source_encoding(claims["source_encoding"])
        if signed != selected:
            raise ValueError("Source encoder authorization does not match")
    return selected


def check_source_encoding_scope(message, *, transport, existing_room=False):
    from .render_contract import source_create_matches

    world = message.get("whitebox", {})
    selected = requested_source_encoding(world)
    if selected is not None and (
        message.get("type") != "init"
        or not source_create_matches(message)
        or message.get("control_protocol") != "whitebox-render-v2"
        or transport != "webrtc-rtp"
        or world.get("loading", "complete") != "complete"
        or existing_room
    ):
        raise ValueError("Source encoder overrides require a new native RTC source")
    return selected


def frozen_source_encoding(world, original_world):
    """Reconnect may omit the selection or echo it, never change an existing source."""
    if "source_encoding" not in world:
        return
    selected = canonical_source_encoding(world["source_encoding"])
    original = original_world.get("source_encoding")
    if original is None or selected != canonical_source_encoding(original):
        raise ValueError("The source encoder selection is frozen")


def session_source_config(config, world):
    selected = requested_source_encoding(world)
    return config if selected is None else replace(config, source_encoding=selected)


def source_bitrate(config):
    return (getattr(config, "source_encoding", None) or {}).get(
        "bitrate", config.source_max_bitrate
    )


def source_gop(config):
    return (getattr(config, "source_encoding", None) or {}).get(
        "gop", config.playback_gop
    )


def _encoder_probe(settings, fps):
    import av

    from .contracts import VideoSpec
    from .playback import create_source_playback_encoder

    encoder = create_source_playback_encoder(
        VideoSpec(832, 480), settings=settings, capture_clock=True, capture_fps=fps
    )
    try:
        packets = []
        for i in range(3):
            packets.extend(encoder.push(i, bytes(832 * 480 * 3), i * 1_000_000 // fps))
        packets.extend(encoder.drain())
        decoder = av.CodecContext.create(
            "libdav1d" if settings["codec"] == "av1" else settings["codec"], "r"
        )
        decoder.thread_count = 1
        frames = []
        for _, packet in packets:
            frames.extend(decoder.decode(av.Packet(packet["payload"])))
        frames.extend(decoder.decode(None))
        if len(frames) != 3 or any((f.width, f.height) != (832, 480) for f in frames):
            raise ValueError("The source codec probe failed")
        return {
            "profileLevelId": encoder.stream["codec"][5:]
            if settings["codec"] == "h264"
            else None
        }
    finally:
        encoder.close()


def preflight_source_encoding(settings, fps, executable=None):
    """No listener, Room, CLI or model; bounded real encode/decode and binary check."""
    executable = executable or os.environ.get(
        "WHITEBOX_NATIVE_RTC_BINARY", "/app/whitebox-rtc"
    )
    native = subprocess.run(
        [executable, "--capabilities"], check=True, capture_output=True, timeout=5
    )
    capabilities = json.loads(native.stdout)
    if (
        capabilities.get("protocol") != "native-video-rtp-v1"
        or capabilities.get("sessionSourceEncoding") is not True
        or settings["codec"] not in capabilities.get("sourceCodecs", [])
        or capabilities.get("modelCodec") != "h264"
    ):
        raise ValueError("The native sender does not support session source encoding")
    script = (
        "import json,sys; from sglang.multimodal_gen.runtime.realtime.whitebox."
        "session_encoding import _encoder_probe; "
        "print(json.dumps(_encoder_probe(json.loads(sys.argv[1]),int(sys.argv[2]))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(settings), str(fps)],
        env={**os.environ, "SGLANG_LIGHTWEIGHT_RUNTIME": "1"},
        check=True,
        capture_output=True,
        timeout=15,
    )
    profile = json.loads(result.stdout).get("profileLevelId")
    if settings["codec"] == "h264":
        import re

        if not isinstance(profile, str) or not re.fullmatch("[a-f0-9]{6}", profile):
            raise ValueError("The source codec probe did not provide an H264 profile")
    return profile
