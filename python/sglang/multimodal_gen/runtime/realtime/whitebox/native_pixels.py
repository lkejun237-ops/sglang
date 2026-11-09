# SPDX-License-Identifier: Apache-2.0
"""Worldkit native v2 RGB24 envelopes. H3 indices remain a separate timeline."""

import copy
import json
import math
import re

MAX_HEADER = 65536
MAX_PIXELS = 64 * 1024 * 1024
FRAME_KEYS = set(
    "protocolVersion sessionId epoch mediaGeneration outputFrameId outputPtsUs source uiRevision uiCompleteThroughUs widthPixels heightPixels".split()
)
SOURCE_KEYS = set(
    "presentationId sdkEpoch sourceFrameId sourceTimeUs simulationTick worldRevision".split()
)
OUTER_KEYS = set(
    "kind schemaVersion pixelFormat widthPixels heightPixels rowStrideBytes payloadBytes colorSpace worldBuildHash uiBundleHash input".split()
)


def safe_int(value, minimum=0):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and value == int(value)
        and minimum <= value <= 2**53 - 1
    )


def checked_json(value, depth=0):
    if depth > 32:
        raise ValueError("Native frame JSON is too deep")
    if value is None or type(value) in (bool, str):
        return
    if type(value) in (int, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            checked_json(item, depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"__proto__", "prototype", "constructor"}:
                raise ValueError("Unsafe native frame key")
            checked_json(item, depth + 1)
        return
    raise ValueError("Invalid native JSON value")


def read_frame_identity(value):
    if (
        not isinstance(value, dict)
        or set(value) != FRAME_KEYS
        or not isinstance(value.get("source"), dict)
        or set(value["source"]) != SOURCE_KEYS
    ):
        raise ValueError("Invalid native frame identity")
    source = value["source"]
    if (
        value["protocolVersion"] != 1
        or not isinstance(value["sessionId"], str)
        or not 1 <= len(value["sessionId"]) <= 128
        or not isinstance(source["presentationId"], str)
        or not 1 <= len(source["presentationId"]) <= 128
        or any(not safe_int(value[k]) for k in FRAME_KEYS - {"sessionId", "source"})
        or any(not safe_int(source[k]) for k in SOURCE_KEYS - {"presentationId"})
        or not 1 <= value["widthPixels"] <= 8192
        or not 1 <= value["heightPixels"] <= 8192
        or value["uiCompleteThroughUs"] < source["sourceTimeUs"]
    ):
        raise ValueError("Invalid native frame identity")
    return copy.deepcopy(value)


def decode_native_packet(wire, descriptor, spec):
    if not isinstance(wire, bytes) or not 5 <= len(wire) <= MAX_PIXELS + MAX_HEADER + 4:
        raise ValueError("Invalid native pixel packet")
    length = int.from_bytes(wire[:4], "big")
    if not 1 <= length <= MAX_HEADER or length + 4 >= len(wire):
        raise ValueError("Invalid native pixel header")
    envelope = json.loads(wire[4 : 4 + length])
    checked_json(envelope)
    payload = wire[4 + length :]
    if not isinstance(envelope, dict) or set(envelope) != OUTER_KEYS:
        raise ValueError("Invalid native pixel fields")
    expected = dict(
        kind="worldkit-pixel-frame",
        schemaVersion=2,
        pixelFormat="rgb24",
        colorSpace="srgb",
        widthPixels=spec.width,
        heightPixels=spec.height,
        rowStrideBytes=spec.width * 3,
        payloadBytes=spec.frame_bytes,
    )
    if (
        any(
            envelope.get(key) != value or isinstance(envelope[key], bool)
            for key, value in expected.items()
        )
        or len(payload) != spec.frame_bytes
    ):
        raise ValueError("Invalid native pixel layout")
    for key in ("worldBuildHash", "uiBundleHash"):
        if (
            not isinstance(envelope[key], str)
            or not re.fullmatch(r"[a-f0-9]{64}", envelope[key])
            or envelope[key] != descriptor[key]
        ):
            raise ValueError("Native pixel artifact identity changed")
    frame = read_frame_identity(envelope["input"])
    if (
        frame["sessionId"] != descriptor["sessionId"]
        or frame["epoch"] != descriptor["epoch"]
        or frame["widthPixels"] != spec.width
        or frame["heightPixels"] != spec.height
    ):
        raise ValueError("Native frame session changed")
    for key in FRAME_KEYS - {"sessionId", "source"}:
        frame[key] = int(frame[key])
    for key in SOURCE_KEYS - {"presentationId"}:
        frame["source"][key] = int(frame["source"][key])
    return frame, payload
