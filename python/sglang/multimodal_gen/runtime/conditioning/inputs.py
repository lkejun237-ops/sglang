# SPDX-License-Identifier: Apache-2.0
"""Ordered single-frame RGB validation and identities for local and remote use."""

from __future__ import annotations

import math

import torch

from .contract import _digest, _invalid, canonical_json, sha256

PREPROCESS_VERSION = "rgb-v1"


def validate_prompt(prompt, max_prompt_bytes=65536):
    if type(prompt) is not str or not prompt:
        _invalid("A nonempty complete prompt is required")
    try:
        size = len(prompt.encode("utf-8"))
    except UnicodeError:
        _invalid("Invalid UTF-8 prompt")
    if size > max_prompt_bytes:
        _invalid("Prompt exceeds the byte budget")
    return prompt


def validate_condition_media(
    media,
    *,
    max_media=16,
    max_side=8192,
    max_input_bytes=512 << 20,
    max_decoded_pixels=178956970,
):
    if any(
        type(n) is not int or n <= 0
        for n in (max_media, max_side, max_input_bytes, max_decoded_pixels)
    ):
        _invalid("Media limits must be positive integers")
    if type(media) is not list or len(media) > min(16, max_media):
        _invalid("Media count exceeds the input budget")
    normalized, size, pixels = [], 0, 0
    for item in media:
        if type(item) is not dict:
            _invalid("Media entries must be objects")
        kind, frames = item.get("type"), item.get("frames")
        if (
            type(kind) is not str
            or kind not in {"image", "video"}
            or not isinstance(frames, torch.Tensor)
        ):
            _invalid("Conditioning requires image or video tensors")
        if (
            frames.device.type != "cpu"
            or frames.dtype != torch.uint8
            or frames.ndim != 4
            or tuple(frames.shape[:2]) != (3, 1)
        ):
            _invalid("Conditioning requires CPU uint8 single-frame CTHW RGB")
        h, w = frames.shape[2:]
        if not 0 < h <= max_side or not 0 < w <= max_side:
            _invalid("Media dimensions exceed the input budget")
        size += frames.numel()
        pixels += h * w
        if size > max_input_bytes or pixels > max_decoded_pixels:
            _invalid("Decoded media exceeds the byte or pixel budget")
        fps = item.get("fps", 1.0 if kind == "image" else None)
        if (
            type(fps) not in (int, float)
            or not 0 < fps <= 240
            or not math.isfinite(fps)
        ):
            _invalid("Invalid media presentation frame rate")
        normalized.append({**item, "fps": float(fps), "frames": frames})
    return normalized


def hash_frame(frames):
    return sha256(frames.contiguous().numpy().tobytes())


def content_identity(
    prompt,
    media,
    profile_id,
    preprocess_version=PREPROCESS_VERSION,
    *,
    item_hashes=None,
):
    validate_prompt(prompt)
    _digest(profile_id, "profile")
    if type(preprocess_version) is not str or not preprocess_version:
        _invalid("Invalid preprocess version")
    media = validate_condition_media(media)
    if item_hashes is not None and (
        type(item_hashes) not in (list, tuple) or len(item_hashes) != len(media)
    ):
        _invalid("Media digest count must match the input")
    items = []
    for index, item in enumerate(media):
        digest = (
            hash_frame(item["frames"]) if item_hashes is None else item_hashes[index]
        )
        _digest(digest, "frame")
        items.append(
            dict(type=item["type"], shape=list(item["frames"].shape), sha256=digest)
        )
    return sha256(
        canonical_json(
            dict(
                profile_id=profile_id,
                preprocess_version=preprocess_version,
                prompt=prompt,
                items=items,
            )
        )
    )
