#!/usr/bin/env python3
"""Export per-case startup/switch timings from downloaded Room archives.

No GPU, service calls, or inference from absent historical instrumentation.
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOM_STAGES = (
    "source_session_to_first_rgb",
    "segment_setup_to_first_rgb",
    "segment_setup_to_source_dispatch",
    "condition_dequeue_to_first_rgb",
    "room_accept_to_first_rgb",
    "room_accept_to_condition_dequeue",
    "condition_dequeue_to_model_send",
    "room_accept_to_model_send",
    "sink_idle_wait",
    "effective_boundary_wait",
    "reference_resolve_store",
    "send_lock_wait",
    "model_wire_lock_wait",
    "prompt_send_call",
    "prompt_send_to_accepted",
    "accepted_to_condition_ready",
    "old_clip_input_wait",
    "old_clip_dispatch_to_condition_ready",
    "condition_ready_to_resume",
    "resume_to_fresh_source_dispatch",
    "source_dispatch_to_first_rgb",
    "resume_to_first_rgb",
)
# These sequential Room-clock intervals close to room_accept_to_first_rgb.
# Other metrics are nested diagnostics and must not be added to this path.
ROOM_PATH = (
    "room_accept_to_model_send",
    "prompt_send_to_accepted",
    "accepted_to_condition_ready",
    "condition_ready_to_resume",
    "resume_to_fresh_source_dispatch",
    "source_dispatch_to_first_rgb",
)


def read_case(directory, environment):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "inference-timing.jsonl"
    reports, caches = {}, {}
    if path.exists():
        with path.open(encoding="utf-8") as file:
            for line in file:
                value = json.loads(line)
                if value.get("role") == "cache":
                    caches[value["prompt_version"]] = value
                elif value.get("report_id"):
                    key = value["report_id"]
                    if value.get("revision", 0) >= reports.get(key, {}).get(
                        "revision", -1
                    ):
                        reports[key] = value
    for value in manifest.get("model_cache_receipts", []):
        caches.setdefault(value["prompt_version"], value)
    switches = {v["prompt_version"]: v for v in manifest.get("condition_switches", [])}
    by_version = defaultdict(list)
    for value in reports.values():
        by_version[value["prompt_version"]].append(value)
    versions = {1, *switches, *caches, *by_version}
    cases, stages = [], []
    identity = manifest.get("model_identity", {})
    for version in sorted(versions):
        values = by_version[version]
        common = dict(
            environment=environment,
            case_id=manifest.get("recording_id", directory.name),
            prompt_version=version,
            request_id=next(
                (v.get("request_id") for v in values if v["role"] == "room"), None
            ),
            source_session_id=manifest.get("source_session_id"),
            created_at=manifest.get("created_at"),
            model_session_id=identity.get("model_session_id"),
            generation_id=identity.get("generation_id"),
            worker_epoch=identity.get("worker_epoch"),
        )
        room = {}
        for value in values:
            for stage in value.get("stages", []):
                stages.append(
                    dict(
                        common,
                        role=value["role"],
                        report_id=value["report_id"],
                        timing_revision=value.get("timing_revision", 0),
                        rank=value.get("rank"),
                        clock_domain=value.get("clock_domain"),
                        report_status=value["status"],
                        **stage,
                    )
                )
                if value["role"] == "room":
                    room[stage["stage"]] = stage.get("host_ms")
        cache = caches.get(version, {})
        cache_timings = cache.get("timings", {})
        for name, entry in cache_timings.get("stages", {}).items():
            stages.append(
                dict(
                    common,
                    role="cache",
                    stage=name,
                    host_ms=entry.get("host_ms"),
                    status=entry.get("status"),
                    device_elapsed_ms=None,
                    device_status="not_requested",
                )
            )
        for name, elapsed in cache_timings.get("cuda_ms", {}).items():
            stages.append(
                dict(
                    common,
                    role="cache",
                    stage=name,
                    host_ms=None,
                    status="executed",
                    device_elapsed_ms=elapsed,
                    device_status="complete" if elapsed is not None else "unresolved",
                )
            )
        triggers = {v.get("trigger_source", "unknown") for v in values} - {"unknown"}
        roles = {v["role"] for v in values}
        timing_revisions = {v.get("timing_revision", 0) for v in values}
        if (
            not values
            and manifest.get("detailed_timing") is False
            and not manifest.get("timing_changes")
        ):
            timing_status = "disabled"
        elif not values:
            timing_status = "not_recorded"
        elif (
            len(timing_revisions) != 1
            or manifest.get("inference_timing", {}).get("incomplete")
            or any(v["status"] != "complete" for v in values)
            or not {"room", "encoder", "dit", "decoder"}.issubset(roles)
        ):
            timing_status = "incomplete"
        else:
            timing_status = "complete"
        path_complete = (
            all(room.get(name) is not None for name in ROOM_PATH)
            and room.get("room_accept_to_first_rgb") is not None
        )
        cases.append(
            dict(
                common,
                kind="first_frame" if version == 1 else "prompt_switch",
                trigger_source=next(iter(triggers))
                if len(triggers) == 1
                else switches.get(version, {}).get("trigger_source", "unknown"),
                **{name + "_ms": room.get(name) for name in ROOM_STAGES},
                client_send_to_room_accept_ms=None,
                client_timing_status="pending_client_integration",
                room_path_status="complete"
                if path_complete
                else "not_recorded"
                if not room
                else "incomplete",
                room_path_closure_error_ms=(
                    sum(room[name] for name in ROOM_PATH)
                    - room["room_accept_to_first_rgb"]
                )
                if path_complete
                else None,
                switch_host_ms=switches.get(version, {}).get("switch_host_ms"),
                readiness_state=switches.get(version, {}).get("readiness_state"),
                cache_result=cache.get("cache_result", "not_recorded"),
                cache_miss_reason=cache.get("miss_reason"),
                measured_roles=",".join(sorted(roles)),
                timing_status=timing_status,
            )
        )
    return cases, stages


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(selected):
    hits = sum(r["cache_result"] in {"l1_hit", "l2_hit", "mixed_hit"} for r in selected)
    misses = sum(r["cache_result"] == "miss" for r in selected)
    return dict(
        count=len(selected),
        cache_hits=hits,
        cache_misses=misses,
        cache_hit_rate=hits / (hits + misses) if hits + misses else None,
        cache_bypass=sum(r["cache_result"] == "bypass" for r in selected),
        cache_unknown=sum(r["cache_result"] == "not_recorded" for r in selected),
        detailed_complete=sum(r["timing_status"] == "complete" for r in selected),
        room_path_complete=sum(
            r.get("room_path_status") == "complete" for r in selected
        ),
        latency_ms={
            name: summarize_latency(selected, name + "_ms")
            for name in (*ROOM_STAGES, "client_send_to_room_accept")
        },
    )


def summarize_latency(rows, field):
    values = sorted(
        r[field]
        for r in rows
        if type(r.get(field)) in (int, float)
        and math.isfinite(r[field])
        and r[field] >= 0
    )
    return dict(
        n=len(values),
        missing=len(rows) - len(values),
        p50=statistics.median(values) if values else None,
        p95_nearest_rank=values[math.ceil(len(values) * 0.95) - 1] if values else None,
        maximum=max(values) if values else None,
    )


def summarize(rows):
    result = {}
    for kind in ("first_frame", "prompt_switch"):
        selected = [r for r in rows if r["kind"] == kind]
        result[kind] = summarize_group(selected)
        result[kind]["by_trigger"] = {
            trigger: summarize_group(
                [r for r in selected if r["trigger_source"] == trigger]
            )
            for trigger in sorted({r["trigger_source"] for r in selected})
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archives",
        type=Path,
        help="Directory containing downloaded manifest.json files",
    )
    parser.add_argument(
        "--environment",
        required=True,
        help="Explicit environment label; never infer from GPU",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, stages = [], []
    for manifest in sorted(args.archives.rglob("manifest.json")):
        value = json.loads(manifest.read_text(encoding="utf-8"))
        if not value.get("model_identity") and not value.get("rendering"):
            continue
        cases, detail = read_case(manifest.parent, args.environment)
        rows.extend(cases)
        stages.extend(detail)
    args.output.mkdir(parents=True, exist_ok=True)
    for kind in ("first_frame", "prompt_switch"):
        write_csv(args.output / f"{kind}.csv", [r for r in rows if r["kind"] == kind])
    write_csv(args.output / "inference_stages.csv", stages)
    summary = json.dumps(summarize(rows), ensure_ascii=False, indent=2)
    (args.output / "summary.json").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
