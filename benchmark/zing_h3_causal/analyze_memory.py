# SPDX-License-Identifier: Apache-2.0
"""Replay trusted local snapshots using explicit request and process identities."""

import argparse
import gzip
import json
import pickle
from pathlib import Path

from allocation_replay import final_live, replay, stack_groups


def analyze(snapshot, entry, buckets=500):
    trace = snapshot["device_traces"][entry["device"]]
    if not trace:
        raise ValueError("Snapshot has no device allocation history")
    partial = entry["history_may_be_truncated"]
    value = replay(
        trace,
        final_live(snapshot, entry["device"]),
        entry["start_us"],
        partial,
        buckets,
    )
    groups = stack_groups(value.pop("peak_live"))
    value.update(
        case=entry["case"],
        pid=entry["pid"],
        role=entry["role"],
        device=entry["device"],
        partial=partial,
        trace_events=len(trace),
        recorded_start_seconds=(trace[0]["time_us"] - entry["start_us"]) / 1e6,
        peak_seconds=(value.pop("new_peak_us") - entry["start_us"]) / 1e6,
        reconstructed_peak_bytes=value["baseline_bytes"] + value["net_peak_bytes"],
        peak_stacks=groups[:40],
        other_peak_stack_bytes=sum(g["bytes"] for g in groups[40:]),
        stage_timeline=entry.get("stage_timeline", []),
        dit_calls=entry.get("dit_calls", []),
        checkpoint_counter_peak_bytes=entry.get("checkpoint_counter_peak_bytes"),
        observed_counter_max_bytes=entry.get("observed_counter_max_bytes"),
        counter_fell_count=entry.get("counter_fell_count", 0),
    )
    value.pop("origin_us")
    value.pop("new_active_peak_us")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON list: snapshot, device, pid, role, case, start_us, history_may_be_truncated; optional relative Host stage_timeline",
    )
    parser.add_argument(
        "--trust-local-snapshots",
        action="store_true",
        required=True,
        help="Snapshots are pickle files: only use artifacts from your own trusted runs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=500)
    args = parser.parse_args(argv)
    if args.buckets <= 0:
        parser.error("--buckets must be positive")
    profiles = []
    for entry in json.loads(args.manifest.read_text()):
        path = args.manifest.parent / entry["snapshot"]
        with (
            gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
        ) as source:
            snapshot = pickle.load(source)
        profiles.append(analyze(snapshot, entry, args.buckets))
    if not profiles:
        raise ValueError("No allocator snapshots selected")
    args.output.mkdir(parents=True, exist_ok=False)
    payload = dict(
        schema_version=1,
        event_size_scope="Requested bytes, excluding allocator rounding and non-PyTorch allocations",
        profiles=profiles,
        errors=[],
    )
    (args.output / "MEMORY_REPLAY.json").write_text(
        json.dumps(payload, allow_nan=False)
    )


if __name__ == "__main__":
    main()
