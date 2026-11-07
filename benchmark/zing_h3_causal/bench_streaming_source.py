# SPDX-License-Identifier: Apache-2.0
"""Serial fixed-case benchmark against a resident continuous-source endpoint."""

import argparse
import asyncio
import hashlib
import json
import traceback
import uuid
from pathlib import Path

from analyze_stream_stalls import analyze_events
from bench_client import (
    append_jsonl,
    calculate_metrics,
    distribution,
    summarize,
    write_json,
)
from benchmark_artifacts import DEFAULT_RETENTION_BYTES, TIMING_SCOPE, check_retention
from build_streaming_report import build_report, load_received_profiles
from case_sets import DEFAULT_CASE_SET, load_case_set
from continuous_source_client import CLIENT_TIMING
from continuous_source_client import parser as client_parser
from continuous_source_client import run
from continuous_source_media import load_prompt_updates
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


def prepare_case(label, media_root, output, *, manifest=None, case_id=None):
    if manifest is None:
        manifest, case_id = CASES[label]
        manifest = HERE / manifest
    case = load_cases(manifest)[case_id]
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
        "display_name": case.get("metadata", {}).get(
            "display_name", CASE_NAMES.get(label, label)
        ),
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
    if "width" in target["source"] and "height" in target["source"]:
        prepared["source_width"] = target["source"]["width"]
        prepared["source_height"] = target["source"]["height"]
    replay = case.get("metadata", {}).get("streaming_replay")
    if replay is not None:
        prepared["reference_ids"] = replay["initial_reference_ids"]
        if len(prepared["reference_ids"]) != len(prepared["references"]):
            raise ValueError("Replay reference IDs do not match the initial images")
        updates = []
        for update in replay["prompt_updates"]:
            updates.append(
                {
                    **update,
                    "references": [
                        {"id": item["id"], "path": local[item["uri"]]}
                        for item in update["references"]
                    ],
                }
            )
        prepared["replay"] = replay
        prepared["provenance"] = case["provenance"]
        prepared["manifest_sha256"] = sha256_file(Path(manifest))
        prepared["prompt_updates"] = updates
    output.mkdir(parents=True, exist_ok=False)
    if replay is not None:
        updates_path = (output / "prompt-updates.json").resolve()
        write_json(updates_path, prepared["prompt_updates"])
        validated = load_prompt_updates(updates_path)
        if any(item["source_frame_start"] >= prepared["frames"] for item in validated):
            raise ValueError("Replay update lies outside the source video")
        prepared["prompt_updates_path"] = str(updates_path)
    write_json(output / "streaming-case.json", prepared)
    return prepared


def case_geometry(
    case, *, width=None, height=None, source_width=None, source_height=None
):
    """Derive a case while retaining its frozen assets and original canvas."""
    for label, w, h in [
        ("output", width, height),
        ("source", source_width, source_height),
    ]:
        if (w is None) != (h is None) or (w is not None and min(w, h) < 1):
            raise ValueError(f"Specify positive {label} width and height together")
    if width is None and source_width is None:
        return case
    derived = dict(
        case,
        original_case_id=case["case_id"],
        original_manifest_dimensions=[case["width"], case["height"]],
    )
    if width is not None:
        derived.update(width=width, height=height)
    if source_width is not None:
        derived.update(source_width=source_width, source_height=source_height)
    sw, sh = (
        derived.get("source_width", derived["width"]),
        derived.get("source_height", derived["height"]),
    )
    derived["case_id"] += (
        f"-source-{sw}x{sh}-output-{derived['width']}x{derived['height']}"
    )
    return derived


async def benchmark(cases, args):
    results = []
    config = {
        "schema_version": 1,
        "transport": "continuous_source",
        "input_mode": "paced" if args.paced else "unpaced",
        "timing_origin": "client_start_before_connect_and_source_decode",
        "timing_boundaries": CLIENT_TIMING,
        "concurrency": 1,
        "prewarm_only": getattr(args, "prewarm_only", False),
        "cache_write": getattr(args, "cache_write", False),
        "media_mode": getattr(args, "media_mode", "representative"),
        "artifact_timing_scope": TIMING_SCOPE,
        "retention_budget_bytes": getattr(
            args, "retention_budget_bytes", DEFAULT_RETENTION_BYTES
        ),
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "seed": args.seed,
        "batch_pattern": args.batch_pattern,
        "timeout_seconds": args.timeout,
        "url": args.url,
        "decoder_url": getattr(args, "decoder_url", None),
        "cases": cases,
        "case_collection": getattr(args, "case_collection", None),
        "prompt_updates_manifest": (
            dict(
                path=str(args.prompt_updates),
                sha256=hashlib.sha256(
                    Path(args.prompt_updates).read_bytes()
                ).hexdigest(),
            )
            if args.prompt_updates
            else None
        ),
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
                        "--retention-budget-bytes",
                        str(
                            getattr(
                                args, "retention_budget_bytes", DEFAULT_RETENTION_BYTES
                            )
                        ),
                        "--media-mode",
                        (
                            getattr(args, "media_mode", "representative")
                            if phase == "measured"
                            and index == 0
                            or getattr(args, "media_mode", "") == "raw"
                            else "metrics"
                        ),
                        "--url",
                        args.url,
                        "--source-rgb" if "source_rgb" in case else "--source-video",
                        case.get("source_rgb", case["source_video"]),
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
                    if "source_width" in case:
                        command.extend(
                            [
                                "--source-width",
                                str(case["source_width"]),
                                "--source-height",
                                str(case["source_height"]),
                            ]
                        )
                    for path, media_id in zip(
                        case["references"], case["reference_ids"], strict=True
                    ):
                        command.extend(
                            ["--reference", path, "--reference-id", media_id]
                        )
                    if config["decoder_url"]:
                        command.extend(["--decoder-url", config["decoder_url"]])
                    if not getattr(args, "no_stage_timing", False):
                        command.append("--receive-stage-timing")
                    if args.paced:
                        command.append("--paced")
                    if getattr(args, "cache_write", False):
                        command.append("--cache-write")
                    prompt_updates = (
                        case.get("prompt_updates_path") or args.prompt_updates
                    )
                    if prompt_updates:
                        command.extend(["--prompt-updates", str(prompt_updates)])
                    if getattr(args, "sink_updates", None):
                        command.extend(["--sink-updates", str(args.sink_updates)])
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
                        "media_path": None,
                        "events_path": str(output / "events.jsonl"),
                        "input_mode": config["input_mode"],
                        "timing_origin": config["timing_origin"],
                        "expected_frames": case["frames"],
                        "client_observation": None,
                    }
                    try:
                        observation = await run(client_parser().parse_args(command))
                        result.update(
                            success=True,
                            client_observation=observation,
                            media_path=observation.get("media_path"),
                            media=observation.get("media"),
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
                                    "prompt_updates",
                                    "prompt_switch_ttff_ms",
                                    "prompt_update_status_counts",
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
                        failure = output / "failure-summary.json"
                        if failure.exists():
                            try:
                                result["client_observation"] = json.loads(
                                    failure.read_text()
                                )
                            except (OSError, ValueError) as receipt_error:
                                result["receipt_error"] = type(receipt_error).__name__
                    media_receipt = output / "media.json"
                    if media_receipt.is_file():
                        result["media"] = json.loads(media_receipt.read_text())
                        result["media_path"] = result["media"]["media_path"]
                    prompt_evidence = output / "prompt-updates.json"
                    if prompt_evidence.is_file():
                        result.setdefault("streaming", {}).update(
                            json.loads(prompt_evidence.read_text())
                        )
                    chunks = []
                    events = output / "events.jsonl"
                    try:
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
                                        "output_frame_start": header[
                                            "output_frame_start"
                                        ],
                                        "source_dependency_end": header[
                                            "source_dependency_end"
                                        ],
                                        "wire_arrival_seconds": event.get(
                                            "wire_received_seconds"
                                        ),
                                        "init_wire_arrival_seconds": event.get(
                                            "init_wire_received_seconds"
                                        ),
                                        "total_size": header.get("total_size"),
                                    }
                                )
                    except (OSError, ValueError, KeyError, TypeError) as receipt_error:
                        result["success"] = False
                        result["error"] = (
                            result.get("error") or "Invalid media event receipt"
                        )
                        previous = result.get("receipt_error")
                        result["receipt_error"] = (
                            (previous + "; " if previous else "")
                            + "events: "
                            + type(receipt_error).__name__
                        )
                    try:
                        result["stall_check"] = analyze_events(
                            [
                                json.loads(line)
                                for line in (output / "events.jsonl")
                                .read_text()
                                .splitlines()
                                if line
                            ]
                        )
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        result["stall_check"] = {
                            "status": "insufficient_evidence",
                            "dit_verdict": "not_evaluated",
                            "error": type(exc).__name__,
                        }
                    result["chunk_count"] = len(chunks)
                    result["output_frames"] = sum(
                        chunk["num_frames"] for chunk in chunks
                    )
                    result["delivery_complete"] = result["output_frames"] == case[
                        "frames"
                    ] and [c["output_frame_start"] for c in chunks] == [
                        sum(p["num_frames"] for p in chunks[:i])
                        for i in range(len(chunks))
                    ]
                    result["delivery_metrics"] = calculate_metrics(chunks)
                    if result.get("metrics"):
                        result["metrics"].update(
                            {
                                key: result["delivery_metrics"].get(key)
                                for key in ("post_ttff_fps", "first_batch_frames")
                            }
                        )
                    result["output_rgb_bytes"] = (
                        result["output_frames"] * case["width"] * case["height"] * 3
                    )
                    if chunks and all(
                        c["init_wire_arrival_seconds"] is not None for c in chunks
                    ):
                        result["init_send_to_rgb"] = calculate_metrics(
                            [
                                {**c, "arrival_seconds": c["init_wire_arrival_seconds"]}
                                for c in chunks
                            ]
                        )
                    if result["success"] and not result["delivery_complete"]:
                        result.update(success=False, error="Incomplete RGB delivery")
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
            summary[case["label"]]["streaming"]["prompt_switch_ttff_ms"] = distribution(
                [
                    update["latency_ms"]
                    for row in passed
                    for update in row["streaming"]["prompt_updates"]
                    if update["status"] == "applied"
                ]
            )
            summary[case["label"]]["init_send_to_rgb"] = {
                key: distribution(
                    [
                        row["init_send_to_rgb"][key]
                        for row in passed
                        if row.get("init_send_to_rgb", {}).get(key) is not None
                    ]
                )
                for key in (
                    "ttff_seconds",
                    "e2e_seconds",
                    "fps",
                    "post_ttff_fps",
                    "tpoc_seconds",
                )
            }
        write_json(args.output_dir / "summary.json", summary)
        bundle = {"config": config, "requests": results, "summary": summary}
        if not getattr(args, "no_stage_timing", False):
            bundle["server"] = load_received_profiles(args.output_dir, bundle)
        write_json(args.output_dir / "results.json", bundle)
        build_report(
            [bundle],
            args.output_dir / "results.html",
            json_output=args.output_dir / "report.json",
        )
        check_retention(
            args.output_dir,
            budget=getattr(args, "retention_budget_bytes", DEFAULT_RETENTION_BYTES),
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="ws://127.0.0.1:30000/v1/realtime_video/generate"
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", choices=["all", *CASES])
    selection.add_argument(
        "--case-set", help=f"Named collection or JSON file; default: {DEFAULT_CASE_SET}"
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--source-rgb-manifest",
        type=Path,
        help="Frozen RGB identities keyed by derived case ID; verified before request timing",
    )
    parser.add_argument("--decoder-url", help="Separate Decoder output WebSocket URL")
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paced", action="store_true")
    parser.add_argument(
        "--cache-write",
        action="store_true",
        help="Allow warmup and measured requests to populate an enabled Block0 cache",
    )
    parser.add_argument("--no-stage-timing", action="store_true")
    parser.add_argument(
        "--sink-updates", type=Path, help="Ordered sink off/refresh events"
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Override the generated canvas width; pair with --height",
    )
    parser.add_argument("--height", type=int)
    parser.add_argument(
        "--source-width",
        type=int,
        help="Source RGB width; pair with --source-height; requires a resize-enabled server",
    )
    parser.add_argument("--source-height", type=int)
    parser.add_argument(
        "--prompt-updates",
        type=Path,
        help="Prompt update manifest for each serial request",
    )
    parser.add_argument("--batch-pattern", type=int, nargs="+", default=[1])
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument(
        "--prewarm-only",
        action="store_true",
        help="Populate resident caches with normal fixed-case warmup requests, without measured runs.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--media-mode",
        choices=("metrics", "representative", "raw"),
        default="representative",
    )
    parser.add_argument(
        "--retention-budget-bytes",
        type=int,
        default=DEFAULT_RETENTION_BYTES,
        help="Explicit total output limit; defaults to 2 GiB.",
    )
    args = parser.parse_args()
    if args.prewarm_only:
        args.measured_runs = 0
    if (
        args.warmup_runs < 0
        or (args.measured_runs < 1 and not args.prewarm_only)
        or (args.prewarm_only and args.warmup_runs < 1)
        or min(args.batch_pattern) < 1
        or args.timeout <= 0
        or args.retention_budget_bytes <= 64 << 20
    ):
        parser.error(
            "Invalid repetition count, source batch size, timeout or retention budget"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.case_set or args.case in (None, "all"):
        collection = load_case_set(args.case_set or DEFAULT_CASE_SET)
        if args.prompt_updates:
            parser.error("Collection cases provide their own prompt updates")
        cases = [
            prepare_case(
                member["case_id"],
                args.media_root,
                args.output_dir / "cases" / member["case_id"],
                manifest=member["manifest"],
                case_id=member["case_id"],
            )
            for member in collection["members"]
        ]
        args.case_collection = {
            key: value for key, value in collection.items() if key != "members"
        }
        write_json(args.output_dir / "case-set.json", args.case_collection)
    else:
        labels = [args.case]
        cases = [
            prepare_case(label, args.media_root, args.output_dir / "cases" / label)
            for label in labels
        ]
    try:
        cases = [
            case_geometry(
                case,
                width=args.width,
                height=args.height,
                source_width=args.source_width,
                source_height=args.source_height,
            )
            for case in cases
        ]
    except ValueError as exc:
        parser.error(str(exc))
    if args.source_rgb_manifest:
        from frozen_rgb import bind_frozen_sources

        cases = bind_frozen_sources(cases, args.source_rgb_manifest)
    for case in cases:
        write_json(
            args.output_dir / "cases" / case["label"] / "streaming-case.json", case
        )
    if args.preflight_only:
        write_json(args.output_dir / "preflight.json", {"cases": cases})
        return
    asyncio.run(benchmark(cases, args))
    if args.prewarm_only:
        rows = json.loads((args.output_dir / "results.json").read_text())["requests"]
        if not rows or any(not row["success"] for row in rows):
            raise SystemExit("Fixed-case prewarm failed; inspect results.json")


if __name__ == "__main__":
    main()
