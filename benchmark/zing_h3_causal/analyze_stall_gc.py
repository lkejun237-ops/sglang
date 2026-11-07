# SPDX-License-Identifier: Apache-2.0
"""Join passive GC intervals with Encoder Host operations and RGB delivery gaps.

This is a correlation report, not a causal or throughput acceptance gate. It
requires the opt-in STALL_ROLE_GC_JSON diagnostic and preserves missing coverage.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


def overlap(a, b, c, d):
    return max(0, min(b, d) - max(a, c)) / 1e6


def distribution(values):
    if not values:
        return dict(n=0, median_ms=None, p95_ms=None, max_ms=None)
    values = sorted(values)
    index = (len(values) - 1) * 0.95
    lo = int(index)
    return dict(
        n=len(values),
        median_ms=statistics.median(values),
        p95_ms=values[lo]
        + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo),
        max_ms=max(values),
    )


def analyze(root):
    from analyze_stream_stalls import analyze_events

    results = root / "work/results"
    events, operations, malformed = [], [], defaultdict(int)
    for path in sorted(results.glob("*.log")):
        with path.open(errors="replace") as stream:
            for line in stream:
                if "STALL_ROLE_GC_JSON " in line:
                    try:
                        row = json.loads(line.split("STALL_ROLE_GC_JSON ", 1)[1])
                    except json.JSONDecodeError:
                        malformed[str(path)] += 1
                        continue
                    row["log"] = str(path)
                    events.append(row)
                if '"encoder_step"' in line:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        malformed[str(path)] += 1
                        continue
                    operations.append(
                        {
                            key: row.get(key)
                            for key in [
                                "encoder_step",
                                "session",
                                "rank",
                                "host_start_ns",
                                "host_end_ns",
                                "shape",
                                "start",
                                "end",
                                "ok",
                            ]
                        }
                    )
    boundaries = results / "REQUEST_BOUNDARIES.json"
    bounds = json.loads(boundaries.read_text()) if boundaries.exists() else []
    sessions = {
        r["session_id"] for r in bounds if r["phase"] == "passive-gc-diagnostic"
    }
    operations = [r for r in operations if r["session"] in sessions]
    by_shape = defaultdict(list)
    for row in operations:
        row["host_ms"] = (row["host_end_ns"] - row["host_start_ns"]) / 1e6
        key = (row["session"], row["encoder_step"], tuple(row["shape"] or []))
        by_shape[key].append(row["host_ms"])
    long_operations = []
    for row in operations:
        key = (row["session"], row["encoder_step"], tuple(row["shape"] or []))
        median = statistics.median(by_shape[key])
        if row["host_ms"] <= 3 * median or row["host_ms"] <= median + 100:
            continue
        row["matched_shape_median_ms"] = median
        row["overlapping_encoder_gc"] = [
            dict(
                event=event,
                overlap_ms=overlap(
                    row["host_start_ns"],
                    row["host_end_ns"],
                    event["start_unix_ns"],
                    event["end_unix_ns"],
                ),
            )
            for event in events
            if event.get("role_label", "").endswith("-encoder")
            and event.get("rank") == row["rank"]
            and overlap(
                row["host_start_ns"],
                row["host_end_ns"],
                event["start_unix_ns"],
                event["end_unix_ns"],
            )
            > 0
        ]
        long_operations.append(row)
    gaps = []
    requests = []
    for path in sorted(results.rglob("requests.jsonl")):
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("server_session_id") not in sessions:
                continue
            raw = [
                json.loads(line)
                for line in Path(row["events_path"]).read_text().splitlines()
                if line
            ]
            report = analyze_events(raw)
            requests.append(
                dict(
                    request_id=row["request_id"],
                    case_id=row["case_id"],
                    success=row.get("success"),
                    delivery_complete=row.get("delivery_complete"),
                    metrics=row.get("metrics"),
                    stall_check={k: v for k, v in report.items() if k != "underflows"},
                )
            )
            frames = [e for e in raw if e["kind"] == "frames"]
            indices = {
                e["header"]["output_frame_start"]: i for i, e in enumerate(frames)
            }
            for gap in report["underflows"]:
                if gap["underflow_seconds"] <= 0.25:
                    continue
                index = indices[gap["output_frame_start"]]
                previous, current = frames[index - 1], frames[index]
                start = previous["header"].get("server_sent_epoch_ms")
                end = current["header"].get("server_sent_epoch_ms")
                matches = []
                if start is not None and end is not None:
                    matches = [
                        dict(
                            event=event,
                            overlap_ms=overlap(
                                start * 1e6,
                                end * 1e6,
                                event["start_unix_ns"],
                                event["end_unix_ns"],
                            ),
                        )
                        for event in events
                        if overlap(
                            start * 1e6,
                            end * 1e6,
                            event["start_unix_ns"],
                            event["end_unix_ns"],
                        )
                        > 0
                    ]
                gaps.append(
                    dict(
                        request_id=row["request_id"],
                        **gap,
                        server_header_gap_ms=None
                        if start is None or end is None
                        else end - start,
                        overlapping_gc=matches,
                    )
                )
    groups = defaultdict(list)
    for event in events:
        groups[
            (
                event.get("role_label"),
                event.get("pid"),
                event.get("rank"),
                event["generation"],
            )
        ].append(event["seconds"] * 1000)
    return dict(
        time=time.time(),
        root=str(root),
        regression_accepted=False,
        scope="Passive diagnostic correlation only. Host includes compute/transport/waits. GC automatic versus explicit cause is unknown; overlap alone does not prove critical-path contribution. Coverage derives from observed role/PID events; no events is not proof of no GC.",
        request_boundaries=bounds,
        diagnostic_sessions=sorted(sessions),
        requests=requests,
        gc_groups=[
            dict(role=k[0], pid=k[1], rank=k[2], generation=k[3], **distribution(v))
            for k, v in groups.items()
        ],
        observed_gc_events=events,
        encoder_shape_groups=[
            dict(session_id=k[0], operation=k[1], shape=k[2], **distribution(v))
            for k, v in by_shape.items()
        ],
        long_encoder_operations=long_operations,
        delivery_gaps=gaps,
        malformed_log_lines=dict(malformed),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--harness-source", type=Path, required=True)
    args = parser.parse_args()
    analyzer = args.harness_source / "benchmark/zing_h3_causal/analyze_stream_stalls.py"
    sys.path.insert(0, str(analyzer.parent))
    result = analyze(args.root)
    result["delivery_analyzer"] = dict(
        path=str(analyzer),
        sha256=hashlib.sha256(analyzer.read_bytes()).hexdigest(),
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                gc_events=len(result["observed_gc_events"]),
                requests=len(result["requests"]),
                gaps=len(result["delivery_gaps"]),
            )
        )
    )


if __name__ == "__main__":
    main()
