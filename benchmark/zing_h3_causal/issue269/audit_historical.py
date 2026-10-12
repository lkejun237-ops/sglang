# SPDX-License-Identifier: Apache-2.0
"""Describe pre-DiT divergence in the frozen 2026-10-08 online evidence."""

import argparse
import hashlib
import json
from pathlib import Path


def audit(summary):
    arms = summary["arms"]
    baseline = arms["perf-tile-off"]["evidence"]["0"]["warmup_conditions"]
    result = {
        "scope": "Historical warmup hashes; identifies stages, not operators or magnitude",
        "arms": {},
    }
    for arm, evidence in arms.items():
        left = evidence["evidence"]["0"]
        right = evidence["evidence"]["1"]
        a, b = left["warmup_conditions"], right["warmup_conditions"]
        if len(a) != len(b) or len(a) != len(baseline):
            raise ValueError("Warmup operation counts differ")
        if any(x["operation"] != y["operation"] for x, y in zip(a, b)):
            raise ValueError("Rank operation order differs")
        mismatches = [
            {
                "index": i,
                "operation": x["operation"],
                "rank0": x["result"],
                "rank1": y["result"],
            }
            for i, (x, y) in enumerate(zip(a, b))
            if x != y
        ]
        different = [
            {
                "index": i,
                "operation": x["operation"],
                "baseline": x["result"],
                "actual": y["result"],
            }
            for i, (x, y) in enumerate(zip(baseline, a))
            if x != y
        ]
        result["arms"][arm] = {
            "rank_mismatches": mismatches,
            "first_vs_off": different[0] if different else None,
            "source_moments_rank_equal": left["warmup_moments_sha256"]
            == right["warmup_moments_sha256"],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(json.loads(args.summary.read_text()))
    report["summary_sha256"] = hashlib.sha256(args.summary.read_bytes()).hexdigest()
    if args.archive or args.receipt:
        if not args.archive or not args.receipt:
            parser.error("--archive and --receipt must be provided together")
        receipt = json.loads(args.receipt.read_text())
        digest = hashlib.sha256()
        with args.archive.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        sha = digest.hexdigest()
        if sha != receipt["sha256"] or args.archive.stat().st_size != receipt["bytes"]:
            raise ValueError("Historical archive receipt mismatch")
        report["archive_receipt_verified"] = True
        report["archive_sha256"] = sha
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
