# SPDX-License-Identifier: Apache-2.0
"""Small public cache telemetry; never forward arbitrary model usage."""

import math
import re


def public_cache_receipt(value):
    if not isinstance(value, dict):
        return None
    key = value.get("cache_key", value.get("key"))
    version = value.get("prompt_version")
    result = value.get("cache_result")
    if (
        not isinstance(key, str)
        or not re.fullmatch(r"[a-f0-9]{64}", key)
        or type(version) is not int
        or version < 1
        or result not in {"l1_hit", "l2_hit", "mixed_hit", "miss", "bypass"}
    ):
        return None

    def duration(x):
        return (
            round(x, 3)
            if type(x) in (int, float) and math.isfinite(x) and 0 <= x < 86_400_000
            else None
        )

    details = value.get("timings") if isinstance(value.get("timings"), dict) else value
    stages = {}
    raw_stages = details.get("stages")
    for name, item in list(
        (raw_stages if isinstance(raw_stages, dict) else {}).items()
    )[:32]:
        if (
            isinstance(name, str)
            and re.fullmatch(r"[a-z0-9_]{1,64}", name)
            and isinstance(item, dict)
        ):
            stages[name] = dict(
                host_ms=duration(item.get("host_ms")),
                status=(
                    item.get("status")
                    if item.get("status") in {"executed", "skipped", "error"}
                    else "unknown"
                ),
            )
    reason = value.get("miss_reason")
    return dict(
        cache_key=key,
        prompt_version=version,
        cache_result=result,
        miss_reason=(
            reason
            if isinstance(reason, str) and re.fullmatch(r"[a-z0-9_]{1,64}", reason)
            else None
        ),
        timings=dict(
            elapsed_host_ms=duration(details.get("elapsed_host_ms")),
            stages=stages,
            cuda_ms={
                name: duration(value)
                for name, value in list(
                    (
                        details.get("cuda_ms")
                        if isinstance(details.get("cuda_ms"), dict)
                        else {}
                    ).items()
                )[:32]
                if isinstance(name, str) and re.fullmatch(r"[a-z0-9_]{1,64}", name)
            },
        ),
    )
