# SPDX-License-Identifier: Apache-2.0
"""Media fields shared by H3 requests and Zing source/reference extensions.

Transport envelopes, task roles and geometry policies belong to the callers.
A source is the same media descriptor as a reference, with different routing.
"""

import math
from collections.abc import Mapping
from typing import Any


def parse_h3_media_descriptor(value: Any, *, path: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    kind = value.get("type")
    if not isinstance(kind, str) or kind not in {
        "image",
        "video",
        "video_audio",
        "audio",
    }:
        raise ValueError(f"{path}.type must be image, video, video_audio or audio")
    uri = value.get("uri")
    if not isinstance(uri, str) or not uri:
        raise ValueError(f"{path}.uri must be a non-empty string")
    return {"type": kind, "uri": uri}


def optional_nonnegative_media_time(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{path} must be a non-negative finite number")
    return normalized
