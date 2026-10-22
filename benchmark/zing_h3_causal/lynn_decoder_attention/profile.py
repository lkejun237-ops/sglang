# SPDX-License-Identifier: Apache-2.0
"""Attribute GPU kernel durations through launch correlation and Lynn CPU scopes."""

import argparse
import bisect
import collections
import json
from pathlib import Path


def summarize(path):
    events = json.loads(path.read_text())["traceEvents"]
    scopes = collections.defaultdict(list)
    launches = {}
    operators = {}
    for event in events:
        category = event.get("cat")
        args = event.get("args", {})
        if category == "user_annotation" and event["name"].startswith("Lynn."):
            scopes[event["tid"]].append(event)
        if category in ("cuda_runtime", "cuda_driver") and "correlation" in args:
            launches[args["correlation"]] = event
        if category == "cpu_op" and "External id" in args:
            operators[args["External id"]] = event
    starts = {}
    for tid, values in scopes.items():
        values.sort(key=lambda event: event["ts"])
        starts[tid] = [event["ts"] for event in values]
    totals = collections.defaultdict(lambda: dict(kernels=0, gpu_ms=0.0))
    unmatched = 0
    for kernel in events:
        if kernel.get("cat") != "kernel":
            continue
        args = kernel.get("args", {})
        launch = launches.get(args.get("correlation"))
        operator = operators.get(args.get("External id"))
        point = launch or operator
        label = "other"
        if point is not None:
            tid, timestamp = point["tid"], point["ts"]
            active = [
                scope
                for scope in scopes[tid][
                    : bisect.bisect_right(starts.get(tid, []), timestamp)
                ]
                if timestamp < scope["ts"] + scope.get("dur", 0)
                and scope["name"] != "Lynn.block"
                and not scope["name"].startswith("Lynn.decode_step.")
            ]
            if active:
                label = min(active, key=lambda scope: scope["dur"])["name"]
            elif operator is not None:
                # Unscoped history updates sit outside streaming_block in baseline.
                # Restrict attribution to actual H32/D64 preparation operators.
                dimensions = str(operator.get("args", {}).get("Input Dims", []))
                if (
                    operator["name"]
                    in (
                        "aten::cat",
                        "aten::clone",
                        "aten::copy_",
                        "aten::slice",
                    )
                    and ", 32, 64]" in dimensions
                ):
                    label = "Lynn.cat_cache"
        else:
            unmatched += 1
        if kernel["name"].lower().startswith("nccl"):
            label = "collective"
        totals[label]["kernels"] += 1
        totals[label]["gpu_ms"] += kernel["dur"] / 1000
    return dict(
        trace=path.name,
        categories=dict(totals),
        unmatched_launches=unmatched,
        note="Unique GPU kernel durations, not CPU scope spans or wall-time shares; excludes memcpy.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps([summarize(path) for path in args.traces], indent=2) + "\n"
    )
