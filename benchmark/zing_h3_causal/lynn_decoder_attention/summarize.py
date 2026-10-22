# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize frozen SP2 real-latent Decoder acceptance records."""

import argparse
import gzip
import json
import statistics
from pathlib import Path

VARIANTS = ("baseline", "rope-cache", "fused")


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def rgb_fps(ranks, variant, start):
    return statistics.median(
        1000
        * sum(ranks[0][variant, trial, i]["frames"] for i in range(start, 15))
        / sum(
            max(rank[variant, trial, i]["wall_ms"] for rank in ranks)
            for i in range(start, 15)
        )
        for trial in range(3)
    )


def summarize(archive):
    assert archive["counts"] == {"gpu": 18, "cold": 12}
    files = archive["files"]
    report = {}
    for resolution in ("480p", "720p", "1080p"):
        processes = []
        for rep in range(3):
            ranks = []
            for rank in range(2):
                path = f"gpu/{resolution}/paired/rep{rep}/rank{rank}"
                done = files[f"{path}/COMPLETE.json"]
                assert all(
                    done[key]
                    for key in ("passed", "strict_pixel_equality", "fused_hits_all")
                )
                assert done["layers"] == 26 and done["sp"] == 2 and done["tp"] == 1
                for variant in VARIANTS[1:]:
                    rows = files[f"{path}/numerical-{variant}.json"]
                    assert len(rows) == 15
                    assert all(
                        row["equal"]
                        and row["finite"]
                        and row["max_abs"] == row["max_rel"] == row["rmse"] == 0
                        for row in rows
                    )
                indexed = {}
                for trial in range(3):
                    for variant in VARIANTS:
                        rows = files[f"{path}/measured-{trial}-{variant}.json"]
                        assert len(rows) == 15
                        for row in rows:
                            indexed[variant, trial, row["latent"]] = row
                ranks.append(indexed)
            for trial in range(3):
                for i in range(15):
                    assert (
                        len(
                            {
                                rank[variant, trial, i]["frames"]
                                for rank in ranks
                                for variant in VARIANTS
                            }
                        )
                        == 1
                    )
            process = {}
            for variant in VARIANTS:
                steady = [
                    max(rank[variant, trial, i]["wall_ms"] for rank in ranks)
                    for trial in range(3)
                    for i in range(2, 15)
                ]
                first = [
                    max(rank[variant, trial, 0]["wall_ms"] for rank in ranks)
                    for trial in range(3)
                ]
                rows = [
                    rank[variant, trial, i]
                    for rank in ranks
                    for trial in range(3)
                    for i in range(2, 15)
                ]
                process[variant] = dict(
                    p50_ms=statistics.median(steady),
                    p95_ms=percentile(steady, 0.95),
                    warm_first_frame_ms=statistics.median(first),
                    # Use two complete 17-RGB output cycles for video FPS.
                    rgb_fps=rgb_fps(ranks, variant, 5),
                    # Retain the original 13-latent sample as a separate metric.
                    sample_rgb_fps=rgb_fps(ranks, variant, 2),
                    cache_mib=max(row["cache"]["storage"] for row in rows) / 2**20,
                    peak_allocated_mib=max(row["peak"] for row in rows) / 2**20,
                )
            processes.append(process)
        summary = {}
        for variant in VARIANTS:
            selected = [process[variant] for process in processes]
            summary[variant] = {
                key: statistics.median(row[key] for row in selected)
                for key in selected[0]
            }
            reductions = [
                100 * (1 - process[variant]["p50_ms"] / process["baseline"]["p50_ms"])
                for process in processes
            ]
            summary[variant].update(
                paired_reductions_percent=reductions,
                median_reduction_percent=statistics.median(reductions),
            )
        report[resolution] = dict(summary=summary, processes=processes)
    cold = {}
    for variant in ("baseline", "fused"):
        processes = []
        for rep in range(3):
            ranks = []
            for rank in range(2):
                path = f"cold/480p/{variant}/rep{rep}/rank{rank}"
                assert files[f"{path}/COMPLETE.json"]["passed"]
                rows = files[f"{path}/first-use-{variant}.json"]
                assert len(rows) == 15
                ranks.append(rows)
            critical = [max(rank[i]["wall_ms"] for rank in ranks) for i in range(15)]
            processes.append(
                dict(first_three_ms=critical[:3], total_15_ms=sum(critical))
            )
        cold[variant] = dict(
            processes=processes,
            first_three_ms=[
                statistics.median(p["first_three_ms"][i] for p in processes)
                for i in range(3)
            ],
            total_15_ms=statistics.median(p["total_15_ms"] for p in processes),
        )
    report["cold_480p"] = cold
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive", type=Path, help="Private FINAL_METRICS.json.gz readback"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.archive, "rt") as stream:
        report = summarize(json.load(stream))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
