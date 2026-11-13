# SPDX-License-Identifier: Apache-2.0
"""Condition-only startup contract shared by Room and the source worker.

The snapshot is source S0, never an extra reference or an accepted video frame.
Only immutable bytes/identities cross this boundary; GPU state stays with its
original worker generation.
"""

from __future__ import annotations

import hashlib
import io
import json
import re

# Existing callers retain v1 unless their original request explicitly selects v2.
PROTOCOL = "whitebox-startup-prepare-v1"
V2_PROTOCOL = "whitebox-startup-prepare-v2"
MAX_SNAPSHOT_PIXELS = 16_777_216
MAX_SNAPSHOT_BYTES = 2 << 20
STARTUP_SECONDS = 60
IDENTITY_FIELDS = (
    "preparation_id",
    "source_session_id",
    "source_epoch",
    "worker_epoch",
    "condition_fingerprint",
)


def preparation_identity(value):
    if not isinstance(value, dict) or set(value) != set(IDENTITY_FIELDS):
        raise ValueError("Invalid startup identity")
    if any(
        not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", v)
        for v in value.values()
    ):
        raise ValueError("Invalid startup identity")
    if not re.fullmatch(r"[a-f0-9]{64}", value["condition_fingerprint"]):
        raise ValueError("Invalid condition fingerprint")
    return {k: value[k] for k in IDENTITY_FIELDS}


def rgb_digest(rgb):
    return hashlib.sha256(rgb).hexdigest()


def validate_snapshot_dimensions(width, height):
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or width * height > MAX_SNAPSHOT_PIXELS
    ):
        raise ValueError("Invalid condition preview dimensions")


def validate_preview_rgb(rgb, width, height):
    validate_snapshot_dimensions(width, height)
    if not isinstance(rgb, bytes) or len(rgb) != width * height * 3:
        raise ValueError("Invalid condition preview pixels")


def snapshot_rgb(frame, width, height, *, protocol=PROTOCOL):
    """Decode once: v1 matches source size; v2 keeps its native condition size."""
    if not isinstance(frame, dict) or set(frame) != {
        "format",
        "width",
        "height",
        "encoded_sha256",
        "rgb_sha256",
        "payload",
    }:
        raise ValueError("Invalid initial frame")
    if protocol == V2_PROTOCOL:
        width, height = frame["width"], frame["height"]
        validate_snapshot_dimensions(width, height)
    elif protocol != PROTOCOL:
        raise ValueError("Unsupported startup protocol")
    raw = frame["payload"]
    if (
        frame["format"] != "png"
        or frame["width"] != width
        or frame["height"] != height
        or not isinstance(raw, bytes)
        or not 0 < len(raw) <= MAX_SNAPSHOT_BYTES
        or rgb_digest(raw) != frame["encoded_sha256"]
    ):
        raise ValueError("Initial frame integrity check failed")
    rgb = decode_snapshot_png(raw, width, height)
    if rgb_digest(rgb) != frame["rgb_sha256"]:
        raise ValueError("Initial frame integrity check failed")
    return rgb


def decode_snapshot_png(raw, width, height):
    """Decode the same native-size, opaque RGB condition for trusted PNG sources."""
    from PIL import Image

    validate_snapshot_dimensions(width, height)
    if (
        not 33 <= len(raw) <= MAX_SNAPSHOT_BYTES
        or raw[:8] != b"\x89PNG\r\n\x1a\n"
        or raw[24] != 8
    ):
        raise ValueError("Initial frame must be an eight-bit PNG")
    if raw[12:16] != b"IHDR" or (
        int.from_bytes(raw[16:20], "big"),
        int.from_bytes(raw[20:24], "big"),
    ) != (width, height):
        raise ValueError("Initial frame dimensions do not match the PNG")
    with Image.open(io.BytesIO(raw)) as image:
        if (
            image.format != "PNG"
            or image.size != (width, height)
            or image.mode not in {"RGB", "RGBA"}
            or getattr(image, "n_frames", 1) != 1
            or any(k in image.info for k in ("icc_profile", "exif"))
        ):
            raise ValueError("Unsupported initial frame format")
        if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
            raise ValueError("Initial frame must be opaque")
        return image.convert("RGB").tobytes()


def condition_fingerprint(
    *,
    rgb,
    prompt,
    references,
    seed,
    source_stream,
    profile_sha256=None,
    protocol=PROTOCOL,
    preview_width=None,
    preview_height=None,
):
    if protocol == V2_PROTOCOL:
        validate_preview_rgb(rgb, preview_width, preview_height)
    elif protocol != PROTOCOL:
        raise ValueError("Unsupported startup protocol")
    value = dict(
        protocol=protocol,
        normalization=(
            "condition-preview-rgb24-v2" if protocol == V2_PROTOCOL else "cli-rgb24-v1"
        ),
        rgb_sha256=rgb_digest(rgb),
        prompt=prompt,
        seed=seed,
        source_stream=source_stream,
        profile_sha256=profile_sha256,
        references=[
            dict(id=r["id"], format=r["format"], sha256=rgb_digest(r["payload"]))
            for r in references
        ],
    )
    if protocol == V2_PROTOCOL:
        value.update(preview_width=preview_width, preview_height=preview_height)
    return rgb_digest(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    )
