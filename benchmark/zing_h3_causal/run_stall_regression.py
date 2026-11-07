# SPDX-License-Identifier: Apache-2.0
"""Run the default full-input instance stall matrix against an isolated endpoint.

Deployment ownership/readiness and isolated DiT replay are separate prerequisites.
This runner never starts, restarts, reserves or stops an existing deployment.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from archive_stall_artifacts import archive_file, archive_to_directory
from case_sets import load_case_set


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--frozen-inputs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--url", default="ws://127.0.0.1:30000/v1/realtime_video/generate")
    archival = p.add_mutually_exclusive_group()
    archival.add_argument("--archive-delivery", type=Path)
    archival.add_argument("--archive-dir", type=Path)
    p.add_argument(
        "--resolutions", nargs="+", choices=["480", "736"], default=["480", "736"]
    )
    p.add_argument("--measured-runs", type=int, default=3)
    p.add_argument("--mode", choices=["full", "capture", "diagnostic"], default="full")
    a = p.parse_args()
    if a.measured_runs < 1:
        p.error("measured-runs must be positive")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    collection = load_case_set()
    rows = []
    for resolution in a.resolutions:
        w, h = (832, 480) if resolution == "480" else (1280, 736)
        for member in collection["members"]:
            cid = member["case_id"]
            if a.mode == "capture" and not cid.endswith("long-session"):
                continue
            selector = a.output_dir / (cid + ".json")
            selector.write_text(
                json.dumps(
                    dict(
                        name=collection["definition"]["name"],
                        cases=[
                            dict(
                                manifest="benchmark/zing_h3_causal/product_case_20261029_cases.jsonl",
                                case_id=cid,
                            )
                        ],
                    )
                )
            )
            rounds = [
                ("S1-warmup", 0, False),
                *[("S1-measured", i, False) for i in range(a.measured_runs)],
            ]
            if a.mode == "capture":
                rounds = [("capture-untimed", 0, False)]
            if a.mode == "diagnostic":
                rounds = [("diagnostic", 0, False)]
            if a.mode == "full" and cid.endswith("long-session"):
                rounds += [("S2-paced", i, True) for i in range(a.measured_runs)]
            for phase, index, paced in rounds:
                out = a.output_dir / f"{w}x{h}" / cid / f"{phase}-{index}"
                cmd = [
                    sys.executable,
                    str(Path(__file__).with_name("bench_streaming_source.py")),
                    "--media-mode",
                    "raw" if a.mode == "diagnostic" else "metrics",
                    "--retention-budget-bytes",
                    str((64 if a.mode == "diagnostic" else 2) << 30),
                    "--url",
                    a.url,
                    "--case-set",
                    str(selector),
                    "--media-root",
                    str(a.media_root),
                    "--source-rgb-manifest",
                    str(a.frozen_inputs),
                    "--width",
                    str(w),
                    "--height",
                    str(h),
                    "--source-width",
                    str(w),
                    "--source-height",
                    str(h),
                    "--output-dir",
                    str(out),
                    "--warmup-runs",
                    "0",
                    "--measured-runs",
                    "1",
                    "--timeout",
                    "3600",
                    "--seed",
                    "123",
                    "--batch-pattern",
                    "1",
                ]
                if phase == "S1-warmup":
                    cmd[cmd.index("--warmup-runs") + 1] = "1"
                    cmd += ["--prewarm-only"]
                if a.mode != "diagnostic":
                    cmd += ["--no-stage-timing"]
                if paced:
                    cmd += ["--paced"]
                (a.output_dir / "active-request.json").write_text(
                    json.dumps(
                        dict(output=str(out), phase=phase, index=index, command=cmd)
                    )
                )
                completed = subprocess.run(cmd, timeout=3900)
                result = dict(
                    resolution=resolution,
                    case_id=cid,
                    phase=phase,
                    index=index,
                    returncode=completed.returncode,
                    output=str(out),
                )
                request_file = out / "requests.jsonl"
                requests = (
                    [
                        json.loads(line)
                        for line in request_file.read_text().splitlines()
                        if line
                    ]
                    if request_file.exists()
                    else []
                )
                result["requests"] = requests
                identities = json.loads(a.frozen_inputs.read_text())
                for request in requests:
                    if request.get("success"):
                        expected = identities[request["case_id"]]
                        if request.get("input_sha256") != expected["sha256"]:
                            raise ValueError(
                                "Actual submitted RGB hash differs from frozen input"
                            )
                        media = request.get("media") or {}
                        if (
                            media.get("output_rgb_bytes") != request["output_rgb_bytes"]
                            or media.get("output_sha256") != request["output_sha256"]
                        ):
                            raise ValueError(
                                "Media hash receipt differs from delivered RGB"
                            )
                        raw = request.get("media_path")
                        if (
                            raw
                            and Path(raw).stat().st_size != request["output_rgb_bytes"]
                        ):
                            raise ValueError(
                                "Raw output byte count differs from delivery receipt"
                            )
                # Failure receipts and partial RGB are retained as well.
                if a.archive_delivery:
                    for raw in out.rglob("*.rgb"):
                        archive_file(raw, a.archive_delivery, remove_verified=True)
                if a.archive_dir:
                    for raw in out.rglob("*.rgb"):
                        archive_to_directory(raw, a.archive_dir, remove_verified=True)
                rows.append(result)
                (a.output_dir / "matrix.json").write_text(json.dumps(rows, indent=2))
                if (
                    completed.returncode
                    or not requests
                    or any(
                        not r.get("success") or not r.get("delivery_complete")
                        for r in requests
                    )
                ):
                    raise RuntimeError(
                        "Instance regression failed; inspect matrix and service logs"
                    )
    (a.output_dir / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                requests=len(rows),
                transport_completed=True,
                verdict="requires_stage_and_capacity_analysis",
                dit_verdict="not_evaluated",
            )
        )
    )


if __name__ == "__main__":
    main()
