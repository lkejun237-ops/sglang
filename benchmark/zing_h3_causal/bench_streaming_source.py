# SPDX-License-Identifier: Apache-2.0
"""Serial fixed-case benchmark against a resident continuous-source endpoint."""

import argparse
import asyncio
import json
import traceback
import uuid
from pathlib import Path

from bench_client import append_jsonl, distribution, summarize, write_json
from build_streaming_report import build_report, load_received_profiles
from continuous_source_client import parser as client_parser
from continuous_source_client import run
from prepare_whitebox_video_smoke import (
    expected_assets,
    load_cases,
    local_source_path,
    sha256_file,
)

HERE = Path(__file__).resolve().parent
CASES = {
    "5s": ("whitebox_video_smoke_cases.jsonl", "zing-whitebox-video-five-triview-0001"),
    "30s-wasteland": (
        "whitebox_video_long_cases.jsonl",
        "zing-whitebox-video-30s-six-triview-0001",
    ),
    "30s-town": (
        "whitebox_video_long_cases.jsonl",
        "zing-whitebox-video-30s-six-triview-0002",
    ),
}
CASE_NAMES = {
    "5s": "5s · 海岸",
    "30s-wasteland": "30s · 废土",
    "30s-town": "30s · 地中海小镇",
}
STREAM_METRICS = (
    "source_to_first_output_seconds",
    "tail_drain_seconds",
    "credit_wait_seconds",
)


def prepare_case(label, media_root, output):
    manifest, case_id = CASES[label]
    case = load_cases(HERE / manifest)[case_id]
    assets, local = [], {}
    for uri, identity in expected_assets(case).items():
        path = local_source_path(uri, media_root).resolve()
        if (
            path.stat().st_size != identity["bytes"]
            or sha256_file(path) != identity["sha256"]
        ):
            raise ValueError(f"Case asset differs from frozen manifest: {path}")
        local[uri] = str(path)
        assets.append({"source_uri": uri, "local_uri": path.as_uri(), **identity})
    user, target = case["messages"]
    spec = target["content"][0]["output"]
    media = {item["id"]: item for item in user["content"] if "id" in item}
    reference_ids = [item["content_id"] for item in target["conditions"]]

    prepared = {
        "label": label,
        "display_name": CASE_NAMES[label],
        "case_id": case_id,
        "frames": spec["frames"],
        "width": spec["width"],
        "height": spec["height"],
        "source_video": local[target["source"]["uri"]],
        "prompt": "\n".join(
            item["text"] for item in user["content"] if item["type"] == "text"
        ),
        "references": [local[media[key]["uri"]] for key in reference_ids],
        "reference_ids": reference_ids,
        "init": {"fps": 24},
        "conditioning": "text_and_references_plus_first_accepted_source_frame",
        "assets": assets,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "streaming-case.json", prepared)
    return prepared


async def benchmark(cases, args):
    results = []
    config = {
        "schema_version": 1,
        "transport": "continuous_source",
        "input_mode": "paced" if args.paced else "unpaced",
        "timing_origin": "client_start_before_connect_and_source_decode",
        "concurrency": 1,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "seed": args.seed,
        "batch_pattern": args.batch_pattern,
        "timeout_seconds": args.timeout,
        "url": args.url,
        "cases": cases,
    }
    write_json(args.output_dir / "config.json", config)
    try:
        for case in cases:
            for phase, count in (
                ("warmup", args.warmup_runs),
                ("measured", args.measured_runs),
            ):
                for index in range(count):
                    request_id = uuid.uuid4().hex
                    output = args.output_dir / case["label"] / f"{phase}-{index}"
                    command = [
                        "--url",
                        args.url,
                        "--source-video",
                        case["source_video"],
                        "--output",
                        str(output),
                        "--prompt",
                        case["prompt"],
                        "--width",
                        str(case["width"]),
                        "--height",
                        str(case["height"]),
                        "--expected-frames",
                        str(case["frames"]),
                        "--seed",
                        str(args.seed),
                        "--timeout",
                        str(args.timeout),
                        "--ffmpeg",
                        args.ffmpeg,
                        "--batch-pattern",
                        *map(str, args.batch_pattern),
                    ]
                    for path, media_id in zip(
                        case["references"], case["reference_ids"], strict=True
                    ):
                        command.extend(
                            ["--reference", path, "--reference-id", media_id]
                        )
                    if not getattr(args, "no_stage_timing", False):
                        command.append("--receive-stage-timing")
                    if args.paced:
                        command.append("--paced")
                    result = {
                        "request_id": request_id,
                        "label": case["label"],
                        "case_id": case["case_id"],
                        "phase": phase,
                        "index": index,
                        "success": False,
                        "error": None,
                        "output_frames": 0,
                        "chunk_count": 0,
                        "metrics": {},
                        "icl_values_seconds": [],
                        "media_path": str(output / "output.rgb"),
                        "input_mode": config["input_mode"],
                        "timing_origin": config["timing_origin"],
                    }
                    try:
                        observation = await run(client_parser().parse_args(command))
                        result.update(
                            success=True,
                            server_session_id=observation["session_id"],
                            input_sha256=observation["input_sha256"],
                            output_sha256=observation["output_sha256"],
                            output_frames=observation["received"],
                            metrics={
                                key: observation[key]
                                for key in (
                                    "ttff_seconds",
                                    "e2e_seconds",
                                    "fps",
                                    "tpoc_seconds",
                                    "icl_seconds",
                                    "stall_seconds",
                                    "ifl_seconds",
                                )
                            },
                            streaming={
                                key: observation[key]
                                for key in (
                                    *STREAM_METRICS,
                                    "first_output_before_source_end",
                                    "dependency_to_output_seconds",
                                    "source_to_output_seconds",
                                )
                            },
                        )
                        result["streaming"]["source_send_span_seconds"] = (
                            observation["last_source_send"]
                            - observation["first_source_send"]
                        )
                    except Exception as exc:
                        result["error"] = f"{type(exc).__name__}: {exc}"
                        result["error_traceback"] = "".join(
                            traceback.format_exception(exc)
                        )
                    chunks = []
                    events = output / "events.jsonl"
                    if events.exists():
                        for line in events.read_text().splitlines():
                            event = json.loads(line)
                            if event["kind"] != "frames":
                                continue
                            header = event["header"]
                            chunks.append(
                                {
                                    "request_id": request_id,
                                    "label": case["label"],
                                    "case_id": case["case_id"],
                                    "phase": phase,
                                    "server_request_id": header.get("request_id"),
                                    "arrival_seconds": event["received_seconds"],
                                    "num_frames": header["num_frames"],
                                    "output_frame_start": header["output_frame_start"],
                                    "source_dependency_end": header[
                                        "source_dependency_end"
                                    ],
                                }
                            )
                    result["chunk_count"] = len(chunks)
                    result["output_frames"] = sum(
                        chunk["num_frames"] for chunk in chunks
                    )
                    result["icl_values_seconds"] = [
                        b["arrival_seconds"] - a["arrival_seconds"]
                        for a, b in zip(chunks, chunks[1:])
                    ]
                    append_jsonl(args.output_dir / "chunks.jsonl", chunks)
                    append_jsonl(args.output_dir / "requests.jsonl", [result])
                    results.append(result)
                    if not result["success"]:
                        raise RuntimeError(
                            f"{case['label']}/{phase}-{index}: {result['error']}"
                        )
    finally:
        summary = summarize(results, cases, args.measured_runs)
        for case in cases:
            passed = [
                row
                for row in results
                if row["label"] == case["label"]
                and row["phase"] == "measured"
                and row["success"]
            ]
            summary[case["label"]]["streaming"] = {
                key: distribution([row["streaming"][key] for row in passed])
                for key in STREAM_METRICS
            }
        write_json(args.output_dir / "summary.json", summary)
        bundle = {"config": config, "requests": results, "summary": summary}
        if not getattr(args, "no_stage_timing", False):
            bundle["server"] = load_received_profiles(args.output_dir, bundle)
        write_json(args.output_dir / "results.json", bundle)
        build_report([bundle], args.output_dir / "results.html")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="ws://127.0.0.1:30000/v1/realtime_video/generate"
    )
    parser.add_argument("--case", choices=["all", *CASES], default="5s")
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paced", action="store_true")
    parser.add_argument("--no-stage-timing", action="store_true")
    parser.add_argument("--batch-pattern", type=int, nargs="+", default=[1])
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if (
        args.warmup_runs < 0
        or args.measured_runs < 1
        or min(args.batch_pattern) < 1
        or args.timeout <= 0
    ):
        parser.error("Invalid repetition count, source batch size or timeout")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    labels = list(CASES) if args.case == "all" else [args.case]
    cases = [
        prepare_case(label, args.media_root, args.output_dir / "cases" / label)
        for label in labels
    ]
    asyncio.run(benchmark(cases, args))


if __name__ == "__main__":
    main()
