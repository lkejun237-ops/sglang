# SPDX-License-Identifier: Apache-2.0
"""Freeze the expected acceptance matrix before launching any workers."""

import argparse
import hashlib
import json
import uuid
from pathlib import Path


def prepare(output, inputs, degrees, lifecycle_sample, reference=False):
    source = Path(inputs) / "inputs.json"
    samples = [row["id"] for row in json.loads(source.read_text())["latents"]]
    degrees = sorted(set([1, *degrees]))
    if not samples or len(set(samples)) != len(samples) or any(p < 1 for p in degrees):
        raise ValueError("Acceptance requires unique samples and positive SP degrees")
    if any(p > 1 for p in degrees) and lifecycle_sample not in samples:
        raise ValueError(
            "The lifecycle sample must be staged for multi-rank acceptance"
        )
    manifest = {
        "run_id": str(uuid.uuid4()),
        "sp": degrees,
        "samples": samples,
        "inputs_json_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "lifecycle_sample": lifecycle_sample,
        "reference": bool(reference),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "acceptance-manifest.json").write_text(json.dumps(manifest, indent=1))
    (output / "sp-summary.json").write_text(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "result": "incomplete",
                "reason": "Workers have not completed reporting",
            },
            indent=1,
        )
    )
    return manifest


def expected_cells(manifest):
    cells = []
    for sp in manifest["sp"]:
        for rank in range(sp):
            for sample in manifest["samples"]:
                for gate, variants in (
                    (
                        "paths",
                        (
                            "offline-vs-preclamp",
                            "finite-vs-offline",
                            "source-vs-offline",
                        ),
                    ),
                    ("ranks-equal", ("preclamp", "offline", "finite")),
                    ("timing", ("decode_step", "offline")),
                ):
                    cells.extend(
                        (sp, rank, gate, sample, variant) for variant in variants
                    )
                if rank == 0:
                    if sp > 1:
                        cells.append((sp, rank, "vs-SP1", sample, "preclamp"))
                    if manifest["reference"]:
                        cells.append((sp, rank, "vs-R", sample, "preclamp"))
            if sp > 1:
                cells.append((sp, rank, "failure", None, "single-rank-failure"))
    return cells


def run_report(args):
    output = Path(args.output)
    manifest_path = output / "acceptance-manifest.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    )
    rows, stale_rows = [], 0
    for path in sorted(output.glob("sp*/metrics-rank*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if manifest is None or row.get("run_id") != manifest["run_id"]:
                stale_rows += 1
                continue
            rows.append(row)
    checks, timing = [], {}
    for row in rows:
        if row["gate"] == "timing":
            timing.setdefault((row["sample"], row["variant"]), {}).setdefault(
                row["sp"], []
            ).append(row)
            continue
        ok = row.get("ok", row.get("equal"))
        if "intervals_ok" in row:
            ok = ok and row["intervals_ok"]
        checks.append(
            {
                k: row.get(k)
                for k in (
                    "sp",
                    "rank",
                    "gate",
                    "sample",
                    "variant",
                    "max_abs",
                    "rmse",
                    "mismatched",
                    "raised",
                )
            }
            | {"ok": bool(ok)}
        )
    speed = []
    for (sample, variant), by_sp in sorted(timing.items()):
        entry = {"sample": sample, "variant": variant}
        for sp, items in sorted(by_sp.items()):
            entry[f"sp{sp}_seconds"] = max(item["seconds"] for item in items)
            if variant == "decode_step":
                entry[f"sp{sp}_owned_tiles"] = [
                    item["owned_tiles"]
                    for item in sorted(items, key=lambda r: r["rank"])
                ]
                entry[f"sp{sp}_state_bytes"] = [
                    item["state_bytes"]
                    for item in sorted(items, key=lambda r: r["rank"])
                ]
                entry[f"sp{sp}_peak_allocated"] = [
                    item["peak_allocated"]
                    for item in sorted(items, key=lambda r: r["rank"])
                ]
        if "sp1_seconds" in entry and "sp2_seconds" in entry:
            entry["sp2_speedup"] = entry["sp1_seconds"] / entry["sp2_seconds"]
        speed.append(entry)
    expected_sp = manifest["sp"] if manifest else []
    samples = manifest["samples"] if manifest else []
    present = {
        (row["sp"], row["rank"], row["gate"], row.get("sample"), row["variant"])
        for row in rows
    }
    cells = expected_cells(manifest) if manifest else []
    missing = [list(cell) for cell in cells if cell not in present]
    if manifest is None:
        missing.append(["acceptance-manifest.json"])
    failed = [c for c in checks if not c["ok"]]
    summary = {
        "run_id": manifest["run_id"] if manifest else None,
        "stale_rows_ignored": stale_rows,
        "sp": expected_sp,
        "samples": samples,
        "checks": len(checks),
        "missing": missing,
        "failed": failed,
        "max_abs": max(
            (c["max_abs"] or 0.0 for c in checks if c["max_abs"] is not None),
            default=None,
        ),
        "result": "failed"
        if failed
        else "incomplete"
        if missing or not checks
        else "passed",
        "timing": speed,
        "rows": checks,
    }
    (output / "sp-summary.json").write_text(json.dumps(summary, indent=1))
    print(
        json.dumps(
            {k: summary[k] for k in ("sp", "checks", "result", "max_abs")}
            | {"missing": missing[:10], "failed": failed[:10]},
            indent=1,
        )
    )
    for entry in speed:
        print(json.dumps(entry))
    return 0 if summary["result"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--sp-degrees", type=int, nargs="+", required=True)
    parser.add_argument("--lifecycle-sample", required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    prepare(
        args.output, args.inputs, args.sp_degrees, args.lifecycle_sample, args.reference
    )
