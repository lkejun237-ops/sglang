# SPDX-License-Identifier: Apache-2.0
"""Validate startup-JIT and full-model 480p A/B service readback."""

import argparse
import gzip
import json
import statistics
from pathlib import Path


def validate_startup(files):
    assert files["GPU_UNIT_RESULT.json"] == dict(
        tests=19, errors=0, failures=0, skipped=0
    )
    proof = [files[f"proof/rank{rank}/COMPLETE.json"] for rank in range(2)]
    for rank in proof:
        assert rank["passed"] and rank["startup_shapes"] == 18
        assert rank["startup_compiles"]
        assert rank["allocated_after"] == rank["allocated_before"]
        for resolution in ("480p", "720p", "1080p"):
            assert len(rank["resolutions"][resolution]["wall_ms"]) == 15
            assert rank["resolutions"][resolution]["new_preparation_compiles"] == []
    return proof


def summarize_service(files, *, prefix="e2e", measured_requests=3):
    pairs = []
    input_hashes = set()
    architectures = set()
    for rep in range(3):
        pair = dict(rep=rep)
        for variant in ("baseline", "candidate"):
            cell = files[f"{prefix}/rep{rep}/{variant}/COMPLETE.json"]
            assert cell["rep"] == rep and cell["variant"] == variant
            architecture = files[f"{prefix}/rep{rep}/{variant}/model-architecture.json"]
            assert architecture["num_layers"] > 10
            architectures.add(json.dumps(architecture, sort_keys=True))
            assert cell["passed"] and cell["full_model"]
            assert cell["decoder_layers"] == 26 and cell["sp"] == 2 and cell["tp"] == 1
            assert [row["phase"] for row in cell["requests"]] == [
                "warmup",
                *(f"measured-{index}" for index in range(measured_requests)),
            ]
            for request in cell["requests"]:
                row = request["summary"]
                assert row["ended"]
                assert all(
                    row[key] == 720
                    for key in ("submitted", "accepted", "consumed", "received")
                )
                assert row["fps"] > 0 and 0 <= row["ttff_seconds"] <= row["e2e_seconds"]
                assert abs(row["fps"] - 720 / row["e2e_seconds"]) < 1e-8
                input_hashes.add(row["input_sha256"])
            rows = [request["summary"] for request in cell["requests"][1:]]
            pair[variant] = dict(
                startup_ready_s=cell["ready_s"],
                first_request_ttff_s=cell["requests"][0]["summary"]["ttff_seconds"],
                requests=[
                    {key: row[key] for key in ("fps", "ttff_seconds", "e2e_seconds")}
                    for row in rows
                ],
                **{
                    key: statistics.median(row[key] for row in rows)
                    for key in ("fps", "ttff_seconds", "e2e_seconds")
                },
            )
        pair["fps_increase_percent"] = 100 * (
            pair["candidate"]["fps"] / pair["baseline"]["fps"] - 1
        )
        pairs.append(pair)
    assert len(input_hashes) == 1 and len(architectures) == 1
    return dict(
        dit_layers=json.loads(next(iter(architectures)))["num_layers"],
        pairs=pairs,
        median_paired_fps_increase_percent=statistics.median(
            pair["fps_increase_percent"] for pair in pairs
        ),
        all_three_pairs_faster=all(pair["fps_increase_percent"] > 0 for pair in pairs),
        scope="Full-model 480p/SP2/TP1, 3 independent server pairs, 3 measured 720-frame requests per server; warmup excluded.",
    )


def summarize(archive):
    assert archive["counts"] == dict(proof=2, e2e=6)
    report = summarize_service(archive["files"])
    report["startup_proof"] = validate_startup(archive["files"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.archive, "rt") as stream:
        report = summarize(json.load(stream))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
