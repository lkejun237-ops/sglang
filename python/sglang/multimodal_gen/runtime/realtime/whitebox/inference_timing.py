# SPDX-License-Identifier: Apache-2.0
"""Allowlisted serving timing receipts; no prompts, tensors or arbitrary usage."""

import math
import re
import time
from uuid import uuid4

# Journal snapshots can outlive a Room process. Never subtract a recovered
# monotonic timestamp from a different process's clock.
ROOM_CLOCK_ID = uuid4().hex


def room_admission():
    return dict(clock_id=ROOM_CLOCK_ID, at=time.perf_counter())


def room_elapsed_ms(marker, now):
    if not isinstance(marker, dict) or marker.get("clock_id") != ROOM_CLOCK_ID:
        return None
    began = marker.get("at")
    if type(began) not in (int, float) or not math.isfinite(began):
        return None
    return duration((now - began) * 1000)


def duration(value):
    return (
        round(value, 3)
        if type(value) in (int, float)
        and math.isfinite(value)
        and 0 <= value < 86_400_000
        else None
    )


def public_timing(value):
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return None
    if value.get("role") not in {"encoder", "dit", "decoder", "room"}:
        return None
    report_id = value.get("report_id")
    version = value.get("prompt_version")
    if not isinstance(report_id, str) or not re.fullmatch(r"[a-f0-9]{32}", report_id):
        return None
    if type(version) is not int or version < 1:
        return None
    stages = value.get("stages")
    if not isinstance(stages, list) or len(stages) > 128:
        return None
    result = {
        k: value.get(k)
        for k in ("schema_version", "report_id", "role", "prompt_version")
    }
    result["revision"] = (
        value.get("revision") if type(value.get("revision")) is int else 0
    )
    result["status"] = (
        value.get("status")
        if value.get("status") in {"pending", "complete", "incomplete"}
        else "incomplete"
    )
    for key in (
        "timing_revision",
        "event_id",
        "rank",
        "output_frame_start",
        "num_frames",
        "source_frame_start",
        "source_dependency_end",
    ):
        if type(value.get(key)) is int and value[key] >= 0:
            result[key] = value[key]
    domain = value.get("clock_domain")
    if isinstance(domain, str) and len(domain) <= 256:
        result["clock_domain"] = domain
    result["stages"] = []
    for item in stages:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("stage"), str)
            or not re.fullmatch(r"[a-z0-9_]{1,64}", item["stage"])
        ):
            continue
        result["stages"].append(
            dict(
                stage=item["stage"],
                start_ms=duration(item.get("start_ms")),
                host_ms=duration(item.get("host_ms")),
                device_elapsed_ms=duration(item.get("device_elapsed_ms")),
                status=item.get("status")
                if item.get("status") in {"executed", "skipped", "error"}
                else "unknown",
                device_status=item.get("device_status")
                if item.get("device_status")
                in {"not_requested", "pending", "complete", "error", "unresolved"}
                else "unknown",
            )
        )
    return result


def trigger_source(condition):
    if condition.get("periodic_refresh"):
        return "periodic_refresh"
    if condition.get("source_context") is not None or "area_binding" in condition:
        return "world_context"
    if condition.get("request_id") is not None:
        return "user_control"
    return "initial" if condition.get("version") == 1 else "unknown"
