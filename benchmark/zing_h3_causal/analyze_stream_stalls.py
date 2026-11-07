# SPDX-License-Identifier: Apache-2.0
"""Audit continuous-source events; delivery symptoms do not identify DiT faults."""

import argparse
import json
import math
from pathlib import Path


def analyze_events(
    events, *, fps=24, startup_buffer_seconds=0.25, stall_threshold_seconds=0.25
):
    """Simulate a fixed initial playback buffer, without excluding later gaps.

    The existing client's receipts contain complete RGB batch arrival times.
    This is a client delivery audit, not a measurement of browser presentation.
    It intentionally cannot certify that the producer or DiT was unstalled.
    """
    if (
        not math.isfinite(fps)
        or fps <= 0
        or any(
            not math.isfinite(v) or v < 0
            for v in (startup_buffer_seconds, stall_threshold_seconds)
        )
    ):
        raise ValueError("Invalid playback policy")
    events = list(events)
    batches = [e for e in events if e["kind"] == "frames"]
    reasons, stalls, gaps = [], [], []
    expected_start, previous, playback_end = 0, None, None
    for event in batches:
        header = event["header"]
        arrival = event["received_seconds"]
        count = header["num_frames"]
        if (
            not isinstance(arrival, (int, float))
            or not math.isfinite(arrival)
            or arrival < 0
            or type(count) is not int
            or count <= 0
            or type(header["output_frame_start"]) is not int
            or header["output_frame_start"] != expected_start
            or (previous is not None and arrival < previous)
        ):
            raise ValueError("Invalid or non-contiguous RGB delivery receipt")
        if previous is None:
            playback_end = arrival + startup_buffer_seconds
        else:
            gaps.append(arrival - previous)
            deficit = max(0, arrival - playback_end)
            if deficit > 0:
                stalls.append(
                    dict(
                        output_frame_start=expected_start,
                        arrival_seconds=arrival,
                        underflow_seconds=deficit,
                    )
                )
            playback_end = max(playback_end, arrival)
        playback_end += count / fps
        expected_start += count
        previous = arrival
    ends = [e for e in events if e["kind"] == "source_end"]
    completed = [e for e in events if e["kind"] == "session.completed"]
    if not batches:
        reasons.append("missing_rgb_delivery")
    if len(ends) != 1 or len(completed) != 1:
        reasons.append("missing_or_duplicate_completion")
    elif (
        ends[0]["frames"] != expected_start
        or completed[0]["message"]["output_frames"] != expected_start
    ):
        reasons.append("incomplete_frame_delivery")
    detected = [s for s in stalls if s["underflow_seconds"] > stall_threshold_seconds]
    severe = [s for s in stalls if s["underflow_seconds"] > 1.0]
    sends = [e for e in events if e["kind"] == "source_send"]
    # Both may be caused by prior server backpressure. Never use them to erase
    # an output stall or to declare the game's/producer's CPU faulty.
    credit = [e["credit_wait_seconds"] for e in sends]
    lag = [e.get("source_schedule_lag_seconds") for e in sends]
    lag = [v for v in lag if v is not None]
    return dict(
        schema_version=2,
        status=(
            "insufficient_evidence"
            if reasons
            else "stall_observed"
            if detected
            else "no_stall_observed"
        ),
        scope="client_rgb_delivery_with_fixed_initial_playback_buffer",
        attribution="unresolved_requires_producer_and_server_stage_evidence",
        reasons=reasons,
        fps=fps,
        startup_buffer_seconds=startup_buffer_seconds,
        stall_threshold_seconds=stall_threshold_seconds,
        output_frames=expected_start,
        batch_count=len(batches),
        ttff_seconds=batches[0]["received_seconds"] if batches else None,
        max_batch_gap_seconds=max(gaps, default=None),
        underflow_count=len(stalls),
        stall_count=len(detected),
        severe_underflow_threshold_seconds=1.0,
        severe_underflow_count=len(severe),
        total_underflow_seconds=sum(s["underflow_seconds"] for s in stalls),
        max_underflow_seconds=max((s["underflow_seconds"] for s in stalls), default=0),
        underflows=stalls,
        source_send_count=len(sends),
        max_credit_wait_seconds=max(credit, default=None),
        max_source_schedule_lag_seconds=max(lag, default=None),
        dit_verdict="not_evaluated",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("--startup-buffer-seconds", type=float, default=0.25)
    parser.add_argument("--stall-threshold-seconds", type=float, default=0.25)
    args = parser.parse_args()
    report = analyze_events(
        [json.loads(line) for line in args.events.read_text().splitlines() if line],
        fps=args.fps,
        startup_buffer_seconds=args.startup_buffer_seconds,
        stall_threshold_seconds=args.stall_threshold_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"])
    return {"no_stall_observed": 0, "stall_observed": 1, "insufficient_evidence": 2}[
        report["status"]
    ]


if __name__ == "__main__":
    raise SystemExit(main())
