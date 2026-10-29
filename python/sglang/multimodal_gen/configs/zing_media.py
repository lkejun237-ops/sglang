# SPDX-License-Identifier: Apache-2.0
"""Fixed source and optional reference spatial limits for Zing media."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ZingResizeLimits:
    max_short_edge: int
    max_area: int

    @classmethod
    def parse(cls, value, *, multiple=1):
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {
            "max_short_edge",
            "max_area",
        }:
            raise ValueError("Zing resize requires max_short_edge and max_area")
        if any(type(v) is not int or v < 1 for v in value.values()):
            raise ValueError("Zing resize limits must be positive integers")
        result = cls(**value)
        if result.max_short_edge < multiple or result.max_area < multiple**2:
            raise ValueError("Zing resize limits cannot hold one spatial grid cell")
        return result


# Realtime source encoding has one fixed, non-upscaling spatial policy.
ZING_SOURCE_RESIZE = ZingResizeLimits(max_short_edge=480, max_area=832 * 480)
