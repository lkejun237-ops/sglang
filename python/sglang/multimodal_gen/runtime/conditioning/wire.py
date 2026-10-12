# SPDX-License-Identifier: Apache-2.0
"""One framed canonical JSON header followed by a safetensors payload."""

from __future__ import annotations

import math
import struct

from safetensors.torch import load, save

from .contract import (
    MAX_HEADER_BYTES,
    _digest,
    _invalid,
    _preflight_tensor,
    _tensor_spec,
    canonical_json,
    safe_json,
)
from .inputs import (
    PREPROCESS_VERSION,
    content_identity,
    validate_condition_media,
    validate_prompt,
)


def _validate_deadline(deadline):
    # JSON integers are always mathematically finite, including values too large
    # for a float. The service clamps them before calculating its time budget.
    if (
        type(deadline) not in (int, float)
        or deadline <= 0
        or (type(deadline) is float and not math.isfinite(deadline))
    ):
        _invalid("RPC deadline must be finite and positive")


def pack_message(header, tensors):
    header = canonical_json(header)
    if len(header) > MAX_HEADER_BYTES:
        _invalid("RPC header exceeds the protocol limit")
    try:
        payload = save(
            {
                name: tensor.detach().to("cpu").contiguous().clone()
                for name, tensor in tensors.items()
            }
        )
    except Exception:
        _invalid("Invalid RPC tensors")
    return struct.pack(">I", len(header)) + header + payload


def unpack_message(body, max_header_bytes=MAX_HEADER_BYTES, *, max_bytes=512 << 20):
    if type(body) is not bytes or not 4 < len(body) <= max_bytes:
        _invalid("Invalid RPC body size")
    size = struct.unpack(">I", body[:4])[0]
    if size > min(MAX_HEADER_BYTES, max_header_bytes) or size > len(body) - 4:
        _invalid("Invalid RPC header size")
    header = safe_json(body[4 : 4 + size], max_bytes=max_header_bytes)
    if type(header) is not dict:
        _invalid("RPC header must be an object")
    try:
        tensors = load(body[4 + size :])
    except Exception:
        _invalid("Invalid RPC safetensors payload")
    return header, tensors


def encode_request(
    prompt,
    media,
    profile_id,
    deadline_ms=10000,
    *,
    input_digest=None,
    item_hashes=None,
    preprocess_version=PREPROCESS_VERSION,
):
    validate_prompt(prompt)
    media = validate_condition_media(media)
    _digest(profile_id, "profile")
    _validate_deadline(deadline_ms)
    digest = input_digest or content_identity(
        prompt, media, profile_id, preprocess_version, item_hashes=item_hashes
    )
    _digest(digest, "input")
    return pack_message(
        dict(
            schema_version=1,
            profile_id=profile_id,
            input_digest=digest,
            prompt=prompt,
            media=[
                dict(type=item["type"], fps=item["fps"], tensor=f"media_{i}")
                for i, item in enumerate(media)
            ],
            deadline_ms=deadline_ms,
        ),
        {f"media_{i}": item["frames"] for i, item in enumerate(media)},
    )


def decode_request(body, profile, limits):
    header, tensors = unpack_message(
        body, limits.max_header_bytes, max_bytes=limits.max_request_bytes
    )
    required = {
        "schema_version",
        "profile_id",
        "input_digest",
        "prompt",
        "media",
        "deadline_ms",
    }
    if (
        set(header) != required
        or type(header["schema_version"]) is not int
        or header["schema_version"] != 1
    ):
        _invalid("Unsupported RPC request schema")
    from .contract import ConditionError

    if header["profile_id"] != profile.profile_id:
        raise ConditionError("CONDITION_PROFILE_MISMATCH")
    _digest(header["input_digest"], "input")
    validate_prompt(header["prompt"], limits.max_prompt_bytes)
    deadline = header["deadline_ms"]
    _validate_deadline(deadline)
    entries = header["media"]
    if type(entries) is not list or len(entries) > limits.max_media:
        _invalid("Invalid RPC media list")
    media, names = [], []
    for item in entries:
        if type(item) is not dict or set(item) not in (
            {"type", "fps", "tensor"},
            {"type", "tensor"},
        ):
            _invalid("Invalid RPC media fields")
        name = item["tensor"]
        if type(name) is not str or name not in tensors or name in names:
            _invalid("Invalid RPC media tensor reference")
        names.append(name)
        value = {"type": item["type"], "frames": tensors[name]}
        if "fps" in item:
            value["fps"] = item["fps"]
        media.append(value)
    if set(tensors) != set(names):
        _invalid("RPC payload contains unreferenced tensors")
    media = validate_condition_media(
        media,
        max_media=limits.max_media,
        max_side=limits.max_side,
        max_input_bytes=limits.max_input_bytes,
        max_decoded_pixels=limits.max_decoded_pixels,
    )
    digest = content_identity(
        header["prompt"],
        media,
        profile.profile_id,
        profile.document["preprocess_version"],
    )
    if digest != header["input_digest"]:
        _invalid("RPC input digest does not match its content")
    # For T=1 every positive sampling rate selects only index 0 with timestamp
    # 0 and the same temporal padding. Canonicalize only this remote boundary:
    # tiny legal caller FPS cannot stall the unchanged native sampling loop.
    media = [
        dict(item, fps=240.0) if item["type"] == "video" else item for item in media
    ]
    return header, media


def validate_response(
    header, condition, profile_id, input_digest, max_rows=16384, max_bytes=256 << 20
):
    if (
        type(header) is not dict
        or header.get("schema_version") != 1
        or type(header.get("schema_version")) is not int
    ):
        _invalid("Unsupported RPC response schema")
    if (
        header.get("profile_id") != profile_id
        or header.get("input_digest") != input_digest
    ):
        _invalid("RPC response identity does not match the request")
    if (
        type(header.get("cache")) is not str
        or header["cache"] not in {"hit", "miss", "joined"}
        or header.get("tensors") != ["context", "row_role"]
    ):
        _invalid("Invalid RPC response tensor manifest")
    for name in ("encode_ms", "queue_ms"):
        value = header.get(name)
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            _invalid("Invalid RPC response timing")
    _tensor_spec(condition, max_rows)
    if sum(t.numel() * t.element_size() for t in condition.values()) > max_bytes:
        _invalid("RPC response exceeds the byte budget")
    return condition


def decode_response(
    body,
    profile_id,
    input_digest,
    max_rows=16384,
    max_bytes=256 << 20,
    *,
    max_header_bytes=MAX_HEADER_BYTES,
):
    if type(body) is not bytes or len(body) > max_bytes:
        _invalid("RPC response exceeds the byte budget")
    if len(body) < 4:
        _invalid("Invalid RPC response framing")
    size = struct.unpack(">I", body[:4])[0]
    if size > min(max_header_bytes, MAX_HEADER_BYTES) or size > len(body) - 4:
        _invalid("Invalid RPC response header size")
    _preflight_tensor(body[4 + size :], max_rows, max_bytes)
    header, condition = unpack_message(body, max_header_bytes, max_bytes=max_bytes)
    return validate_response(
        header, condition, profile_id, input_digest, max_rows, max_bytes
    )
