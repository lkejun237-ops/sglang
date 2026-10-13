# SPDX-License-Identifier: Apache-2.0
"""Validate fixed-case stage provenance and retain rank-local work summaries."""

import argparse
import json
from pathlib import Path

from bench_client import CASES, DEFAULT_CASES, REFERENCE_COUNTS, distribution


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(
    directory,
    ranks=(0, 1),
    *,
    expected_frames=120,
    expected_references=6,
    expected_qwen=1,
):
    requests = jsonl(directory / "client/requests.jsonl")
    chunks = jsonl(directory / "client/chunks.jsonl")
    paths = sorted((directory / "timing").glob("stage-timing-pid*.jsonl"))
    if not paths:
        raise ValueError("No stage timing records")
    rows = [row for path in paths for row in jsonl(path)]
    if len({r["span_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate stage spans")
    health_paths = list((directory / "timing").glob("*-health.json"))
    if len(health_paths) != len(ranks):
        raise ValueError("Missing collector health records")
    for path in health_paths:
        health = json.loads(path.read_text())
        if (
            health["dropped"]
            or health["pending"]
            or health["queued"]
            or health["error"]
        ):
            raise ValueError("Stage collector lost or did not finish samples")
    summaries, used = [], set()
    for request in requests:
        if not request["success"]:
            raise ValueError("Incomplete client request")
        batches = [r for r in chunks if r["request_id"] == request["request_id"]]
        dispatches = {r["server_request_id"] for r in batches}
        if sum(r["num_frames"] for r in batches) != expected_frames:
            raise ValueError("Fixed-case client output is incomplete")
        selected = [r for r in rows if r["dispatch_id"] in dispatches]
        used.update(r["span_id"] for r in selected)
        if {r["rank"] for r in selected} != set(ranks):
            raise ValueError("Missing rank coverage")
        for rank in ranks:
            group = [r for r in selected if r["rank"] == rank]
            if len({r["request_id"] for r in group}) != 1:
                raise ValueError("Stage spans mix request identities")
            for row in group:
                if row["status"] != "ok" or not (
                    row["host_enter_ns"]
                    <= row["host_return_ns"]
                    <= row["host_handoff_ns"]
                ):
                    raise ValueError("Invalid stage boundary or failed span")
                if row["mode"] == "cuda" and row["stage"] != "Conditioning":
                    if row["cuda_status"] != "complete" or row["cuda_stream_ms"] < 0:
                        raise ValueError("Missing CUDA stream interval")
            encoders = [r for r in group if r["stage"] == "Encoder"]
            components = [r["component"] for r in encoders]
            if (
                components.count("qwen") != expected_qwen
                or components.count("reference_vae") != expected_references
                or components.count("source_vae") != 1
            ):
                raise ValueError(
                    "Encoder coverage does not match the expected source/reference contract"
                )
            dits = sorted(
                (r for r in group if r["stage"] == "DiT"),
                key=lambda r: r["block_index"],
            )
            expected_blocks = sorted({r["chunk_index"] for r in batches})
            if [r["block_index"] for r in dits] != expected_blocks:
                raise ValueError("DiT blocks do not match delivered block identities")
            latent_end = 0
            for row in dits:
                start, end = row["latent_range"]
                if start != latent_end or end <= start:
                    raise ValueError("DiT latent ranges are not contiguous")
                latent_end = end
            decoders = sorted(
                (r for r in group if r["stage"] == "Decoder"),
                key=lambda r: r["output_frame_range"][0],
            )
            frame_end, delivered = 0, []
            for row in decoders:
                start, end = row["output_frame_range"]
                if start != frame_end or end <= start:
                    raise ValueError("Decoder frame ranges are not contiguous")
                frame_end = end
                delivered.extend(b["dispatch_id"] for b in row["output_blocks"])
            if (
                frame_end != expected_frames
                or len(delivered) != len(dispatches)
                or set(delivered) != dispatches
            ):
                raise ValueError("Decoder outputs do not match client deliveries")
            summary = {
                "client_request_id": request["request_id"],
                "phase": request["phase"],
                "model_request_id": group[0]["request_id"],
                "rank": rank,
                "clock_domain": group[0]["clock_domain"],
                "stages": {},
            }
            for stage in ("Encoder", "DiT", "Decoder", "Conditioning"):
                spans = [r for r in group if r["stage"] == stage]
                summary["stages"][stage] = {
                    "count": len(spans),
                    "host_call_ms_sum": sum(r["host_call_ms"] for r in spans),
                    "host_call_ms": distribution([r["host_call_ms"] for r in spans]),
                    "cuda_stream_ms_sum": sum(r["cuda_stream_ms"] for r in spans)
                    if spans and all(r["cuda_stream_ms"] is not None for r in spans)
                    else None,
                }
            summaries.append(summary)
    if len(used) != len(rows):
        raise ValueError("Unmatched stage records")
    return {
        "valid": True,
        "qwen_scope": "local_encoder_span"
        if expected_qwen
        else "not_measured_conditioner_callback",
        "client_requests": len(requests),
        "span_count": len(rows),
        "semantics": "rank-local cumulative intervals; nested Conditioning overlaps Encoder; not additive E2E or pure kernel time",
        "requests": summaries,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=DEFAULT_CASES, default="5s")
    parser.add_argument(
        "--qwen-scope",
        choices=("local-span", "conditioner-unmeasured"),
        default="conditioner-unmeasured",
    )
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(
            analyze(
                args.directory,
                expected_frames=CASES[args.case][1],
                expected_references=REFERENCE_COUNTS[args.case],
                expected_qwen=1 if args.qwen_scope == "local-span" else 0,
            ),
            indent=2,
        )
        + "\n"
    )
