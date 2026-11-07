# SPDX-License-Identifier: Apache-2.0
"""Independent CPU readback. Collection is not accuracy or capacity acceptance."""

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import time

P = pathlib.Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=P, required=True)
parser.add_argument("--output", type=P, required=True)
parser.add_argument("--source-revision", required=True)
parser.add_argument("--worlds", type=int, nargs="+", default=[2, 3], choices=[2, 3])
args = parser.parse_args()
if len(set(args.worlds)) != len(args.worlds):
    raise ValueError("Decoder worlds must be unique")
R = args.root
ready = json.loads((R / "CPU_READY.json").read_text())
if not ready["passed"] or ready["source_revision"] != args.source_revision:
    raise ValueError("The frozen source revision must match the requested run")
for name in ["COMMON_CAPTURES.json", "FROZEN_RGB.json"]:
    path = R / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != ready["controls"][str(path)]:
        raise ValueError("Frozen Decoder comparison inputs changed")
out = dict(
    time=time.time(),
    root=str(R),
    source_revision=args.source_revision,
    cpu_ready_sha256=hashlib.sha256((R / "CPU_READY.json").read_bytes()).hexdigest(),
    analyzer_sha256=hashlib.sha256(P(__file__).read_bytes()).hexdigest(),
    scope="Same allocated8 GPUs, isolated full15921-frame Decoder replay plus matched403-frame full-instance pilot; does not replace full-instance new-layout matrix; no production deployment",
    parity=[],
    pilot=[],
    errors=[],
)


def load(p):
    return json.loads(p.read_text())


def stats(v):
    v = sorted(v)
    if not v:
        return dict(count=0)
    k = (len(v) - 1) * 0.95
    i = int(k)
    return dict(
        count=len(v),
        median=statistics.median(v),
        p95=v[i] + (v[min(i + 1, len(v) - 1)] - v[i]) * (k - i),
        min=v[0],
        max=v[-1],
    )


identities = (
    load(R / "COMMON_CAPTURES.json") if (R / "COMMON_CAPTURES.json").exists() else []
)
for size in args.worlds:
    for i, identity in enumerate(identities):
        D = R / "work" / f"decoder-parity-d{size}-{i}"
        cell = dict(
            decoder_world=size,
            geometry=identity["geometry"],
            case_id=identity["case_id"],
            input_sha256=identity["input_sha256"],
            capture=identity["path"],
            strict_numerical_accepted=False,
            ranks=[],
        )
        out["parity"].append(cell)
        p = D / "rank0-comparison.json"
        if p.exists():
            comparison = load(p)
            rows = comparison["rows"]
            count = sum(v["elements"] for v in rows)
            rmse = (
                math.sqrt(sum(v["squared_error_sum"] for v in rows) / count)
                if count
                else None
            )
            cell.update(
                comparison_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                full_chunks=len(rows),
                frames=sum(v["frames"] for v in rows),
                all_exact=all(v["exact"] for v in rows),
                max_abs=max((v["max_abs"] for v in rows), default=None),
                rmse=rmse,
                psnr_db=-20 * math.log10(rmse) if rmse else None,
                psnr_scope="RGB data_range1; exact match has no finite PSNR value; no uncalibrated quality threshold",
                strict_numerical_accepted=len(rows) == 1873
                and sum(v["frames"] for v in rows) == 15921
                and all(
                    v["exact"] and v["max_abs"] == 0 and v["squared_error_sum"] == 0
                    for v in rows
                ),
                sample_pngs=[str(p) for p in sorted(D.glob("*.png"))],
            )
        for rank in range(size):
            p = D / f"rank{rank}-measured.json"
            r = dict(rank=rank, collected=False, modes=[])
            cell["ranks"].append(r)
            if not p.exists():
                continue
            runs = load(p)
            r["measured_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            for mode in ["single", "parallel"]:
                values = [v for v in runs if v["mode"] == mode]
                if mode == "single" and rank:
                    continue
                valid = (
                    len(values) == 3
                    and sorted(v["trial"] for v in values) == [0, 1, 2]
                    and all(
                        v["frames"] == 15921
                        and v["rank"] == rank
                        and v["world"] == size
                        and len(v["blocks"]) == 1873
                        and sum(b["frames"] for b in v["blocks"]) == 15921
                        for v in values
                    )
                )
                r["modes"].append(
                    dict(
                        mode=mode,
                        accepted_collection=valid,
                        sessions=len(values),
                        seconds=stats([v["seconds"] for v in values]),
                        fps=stats([v["fps"] for v in values]),
                        push_cuda_elapsed_ms=stats(
                            [b["cuda_elapsed_ms"] for v in values for b in v["blocks"]]
                        ),
                        memory_per_session=[
                            {
                                k: v[k]
                                for k in [
                                    "trial",
                                    "resident_before",
                                    "after_cleanup",
                                    "peak_allocated_bytes",
                                    "peak_reserved_bytes",
                                ]
                            }
                            for v in values
                        ],
                        scope="Isolated Decoder total host time includes EOS/stream close; per-push CUDA elapsed excludes these. All1873 blocks include startup/drain phases; pooled distribution is full fixed case, not a homogeneous kernel group. Process memory includes model, frozen latent input and workspaces; no driver peak claimed.",
                    )
                )
            r["collected"] = (
                (D / f"rank{rank}-COMPLETE.json").exists()
                and all(v["accepted_collection"] for v in r["modes"])
                and len(r["modes"]) == (2 if rank == 0 else 1)
            )
        cell["isolated_collection_accepted"] = all(
            r["collected"] for r in cell["ranks"]
        )
        first = cell["ranks"][0]
        if len(first["modes"]) == 2 and all(
            m["accepted_collection"] for m in first["modes"]
        ):
            single, parallel = first["modes"]
            cell["isolated_time_change_percent"] = (
                parallel["seconds"]["median"] / single["seconds"]["median"] - 1
            ) * 100
pilot = load(R / "PILOT_REQUESTS.json") if (R / "PILOT_REQUESTS.json").exists() else []
out["pilot_requests_observed"] = len(pilot)
frozen = load(R / "FROZEN_RGB.json") if (R / "FROZEN_RGB.json").exists() else {}
out["pilot_receipt_checks"] = []
for indexed in pilot:
    try:
        p = P(indexed["receipt"])
        raw = p.read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines() if line]
        assert len(rows) == 1
        r = rows[0]
        identity = frozen[r["case_id"]]
        obs = r["client_observation"]
        media = r["media"]
        checks = dict(
            identity=r["request_id"] == indexed["request_id"]
            and r["metrics"] == indexed["metrics"],
            input_sha256=r["input_sha256"] == identity["sha256"],
            frames=r["output_frames"]
            == r["expected_frames"]
            == identity["frames"]
            == 403,
            delivery=r["success"]
            and r["delivery_complete"]
            and obs["completed"]
            and obs["consumed"] == 403
            and obs["received"] == 403,
            output_integrity=r["output_sha256"]
            == obs["output_sha256"]
            == media["output_sha256"]
            and r["output_rgb_bytes"]
            == media["output_rgb_bytes"]
            == 403 * identity["width"] * identity["height"] * 3,
            events_exist=P(r["events_path"]).is_file(),
        )
        out["pilot_receipt_checks"].append(
            dict(
                request_id=r["request_id"],
                path=str(p),
                sha256=hashlib.sha256(raw).hexdigest(),
                checks=checks,
                accepted=all(checks.values()),
            )
        )
    except Exception as exc:
        out["errors"].append(dict(request_id=indexed["request_id"], error=repr(exc)))
out["all_pilot_receipts_valid"] = (
    len(pilot) == 16 * (1 + len(args.worlds))
    and len(out["pilot_receipt_checks"]) == 16 * (1 + len(args.worlds))
    and all(r["accepted"] for r in out["pilot_receipt_checks"])
    and not out["errors"]
)
for size in [1, *args.worlds]:
    for geometry in [[832, 480], [1280, 736]]:
        rows = [
            r
            for r in pilot
            if r["decoder_world"] == size and [r["width"], r["height"]] == geometry
        ]
        formal = [r for r in rows if r["phase"] in ["formal0", "formal1", "formal2"]]
        diag = [
            r
            for r in rows
            if r["phase"] in ["diagnostic0", "diagnostic1", "diagnostic2"]
        ]
        valid = (
            len(rows) == 8
            and len({r["phase"] for r in rows}) == 8
            and len(formal) == len(diag) == 3
        )
        cell = dict(
            decoder_world=size,
            geometry=geometry,
            requests=len(rows),
            complete=valid,
            formal_request_ids=[r["request_id"] for r in formal],
            diagnostic_request_ids=[r["request_id"] for r in diag],
            formal_metrics={
                k: stats([r["metrics"][k] for r in formal])
                for k in [
                    "ttff_seconds",
                    "e2e_seconds",
                    "fps",
                    "post_ttff_fps",
                    "tpoc_seconds",
                ]
            },
            short_case_24fps_capacity=valid
            and all(r["metrics"]["post_ttff_fps"] >= 24 for r in formal),
            allocated_gpus=8,
            active_gpus=5 + size,
            unused_gpus=3 - size,
            scope="Warm formal metrics n3 and independent CUDA diagnostics n3; full403-frame original few-images case only. Geometry-specific first request follows fresh service processes after shared disk caches were primed.",
        )
        out["pilot"].append(cell)
for cell in out["pilot"]:
    base = next(
        c
        for c in out["pilot"]
        if c["decoder_world"] == 1 and c["geometry"] == cell["geometry"]
    )
    if cell["complete"] and base["complete"]:
        cell["formal_median_change_percent_vs_d1"] = {
            k: (v["median"] / base["formal_metrics"][k]["median"] - 1) * 100
            for k, v in cell["formal_metrics"].items()
        }
eligible = []
for size in args.worlds:
    numeric = [c for c in out["parity"] if c["decoder_world"] == size]
    capacity = [c for c in out["pilot"] if c["decoder_world"] == size]
    if (
        out["all_pilot_receipts_valid"]
        and len(numeric) == len(capacity) == 2
        and all(
            c["strict_numerical_accepted"] and c["isolated_collection_accepted"]
            for c in numeric
        )
        and all(c["short_case_24fps_capacity"] for c in capacity)
    ):
        eligible.append(size)
out["eligible_layouts_for_full_validation"] = eligible
out["recommended_next_validation_world"] = min(eligible) if eligible else None
out["full_matrix_accepted"] = False
out["production_quality_scope"] = (
    "Exact parity is checked against existing single Decoder on identical real latent input, not native-vs-SGLang generation quality. Any nonexact differences require separate quality assessment; no threshold invented from observed drift."
)
with args.output.open("x") as stream:
    json.dump(out, stream, indent=2)
print("DECODER_READBACK", len(pilot), "eligible_full_validation", eligible)
