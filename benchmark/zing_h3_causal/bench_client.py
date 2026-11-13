# SPDX-License-Identifier: Apache-2.0
"""Fixed-payload, concurrency-one client benchmark for a resident Zing server.

No model imports, GPU profiling, synthetic input, or server startup. A client
chunk is one complete RGB frame-batch delivery, not an internal DiT block.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from benchmark_artifacts import TIMING_SCOPE, MediaCapture, check_retention
from case_sets import DEFAULT_CASE_SET, load_case_set
from prepare_whitebox_video_smoke import (
    expected_assets,
    load_cases,
    local_source_path,
    sha256_file,
)

HERE = Path(__file__).resolve().parent
# Explicit legacy finite-input aliases; case_sets defines the default replay set.
CASES = {
    "5s": ("zing-whitebox-video-five-triview-0001", 120),
    "30s-wasteland": ("zing-whitebox-video-30s-six-triview-0001", 720),
    "30s-town": ("zing-whitebox-video-30s-six-triview-0002", 720),
    "30s-indoor": ("zing-whitebox-video-30s-two-triview-0001", 720),
    "30s": ("zing-whitebox-video-30s-six-reference-0001", 720),
}
DEFAULT_CASES = ("5s", "30s-wasteland", "30s-town", "30s-indoor")
REFERENCE_COUNTS = {
    "5s": 6,
    "30s-wasteland": 7,
    "30s-town": 7,
    "30s-indoor": 3,
    "30s": 6,
}


def default_manifest(label):
    return HERE / (
        "whitebox_video_smoke_cases.jsonl"
        if label == "5s"
        else "whitebox_video_30s_cases.jsonl"
        if label == "30s"
        else "whitebox_video_long_cases.jsonl"
    )


def select_cases(args):
    if args.case != "all":
        label = args.case
        manifest = getattr(
            args,
            "manifest_long" if label in DEFAULT_CASES[1:] else f"manifest_{label}",
            default_manifest(label),
        )
        return None, [(label, manifest, CASES[label][0])]
    collection = load_case_set(args.case_set or DEFAULT_CASE_SET)
    if collection["definition"].get("transport") == "continuous_source":
        raise ValueError(
            "The default collection requires bench_streaming_source.py; "
            "use --case-set whitebox-basic for the legacy finite-input cases"
        )
    aliases = {case_id: label for label, (case_id, _) in CASES.items()}
    selected = [
        (
            aliases.get(member["case_id"], f"case-{index + 1}"),
            member["manifest"],
            member["case_id"],
        )
        for index, member in enumerate(collection["members"])
    ]
    if not args.case_set:
        selected = [
            (
                label,
                getattr(
                    args,
                    "manifest_long"
                    if label in DEFAULT_CASES[1:]
                    else f"manifest_{label}",
                    manifest,
                ),
                case_id,
            )
            for label, manifest, case_id in selected
        ]
    return {
        key: value for key, value in collection.items() if key != "members"
    }, selected


SAMPLING_FIELDS = {
    "seed",
    "num_inference_steps",
    "guidance_scale",
    "zing_solver",
    "zing_denoising_step_list",
    "zing_realtime_decode_mode",
    "zing_window_chunks",
    "zing_sink_chunks",
    "zing_rope_position_mode",
    "zing_rope_max_frame_gap",
    "zing_window_eviction_timing",
}
METRIC_NAMES = ("ttff_seconds", "e2e_seconds", "fps", "post_ttff_fps", "tpoc_seconds")


def distribution(values):
    """Linear-interpolated percentiles; an empty set is unknown, never zero."""
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "median": None, "min": None, "max": None, "p95": None}
    position = (len(ordered) - 1) * 0.95
    lo, hi = math.floor(position), math.ceil(position)
    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo),
    }


def calculate_metrics(chunks):
    if not chunks:
        return {**dict.fromkeys(METRIC_NAMES), "icl_seconds": distribution([])}
    arrivals = [row["arrival_seconds"] for row in chunks]
    if any(not math.isfinite(t) or t < 0 for t in arrivals) or any(
        b < a for a, b in zip(arrivals, arrivals[1:])
    ):
        raise ValueError("Invalid client arrival timestamps")
    frames = sum(row["num_frames"] for row in chunks)
    intervals = [b - a for a, b in zip(arrivals, arrivals[1:])]
    return {
        "ttff_seconds": arrivals[0],
        "e2e_seconds": arrivals[-1],
        "fps": frames / arrivals[-1] if arrivals[-1] > 0 else None,
        "first_batch_frames": chunks[0]["num_frames"],
        "post_ttff_fps": (
            (frames - chunks[0]["num_frames"]) / (arrivals[-1] - arrivals[0])
            if arrivals[-1] > arrivals[0]
            else None
        ),
        "tpoc_seconds": statistics.mean(intervals) if intervals else None,
        "icl_seconds": distribution(intervals),
    }


class Delivery:
    """Validate wire batch order without treating split batches as duplicates."""

    def __init__(self, frames, width, height):
        self.expected_frames = frames
        self.width, self.height = width, height
        self.frames = 0
        self.block, self.part, self.parts = 0, 0, None
        self.block_request_id = None
        self.chunks = []
        self.profile = None
        self.profile_wait_seconds = None

    def accept(self, header, payload, arrival):
        count = header.get("num_frames")
        if type(count) is not int or count < 1:
            raise ValueError("Invalid frame count")
        if (
            header.get("format") != "rgb24"
            or header.get("encoding") not in (None, "raw")
            or header.get("channels") != 3
            or header.get("width") != self.width
            or header.get("height") != self.height
            or not isinstance(payload, bytes)
            or len(payload) != count * self.width * self.height * 3
            or header.get("total_size") != len(payload)
        ):
            raise ValueError("Invalid complete RGB payload")
        parts = header.get("num_frame_batches")
        request_id = header.get("request_id")
        if (
            header.get("chunk_index") != self.block
            or header.get("frame_batch_index") != self.part
            or type(parts) is not int
            or parts < 1
            or self.part >= parts
            or (self.parts is not None and parts != self.parts)
            or not isinstance(request_id, str)
            or not request_id
            or (
                self.block_request_id is not None
                and request_id != self.block_request_id
            )
            or header.get("is_final_frame_batch") is not (self.part == parts - 1)
            or header.get("output_frame_start", self.frames) != self.frames
            or self.frames + count > self.expected_frames
        ):
            raise ValueError("Missing, duplicate, or out-of-order frame batch")
        if (
            not math.isfinite(arrival)
            or arrival < 0
            or (self.chunks and arrival < self.chunks[-1]["arrival_seconds"])
        ):
            raise ValueError("Invalid client arrival timestamp")
        row = {
            "delivery_index": len(self.chunks),
            "server_request_id": request_id,
            "chunk_index": self.block,
            "frame_batch_index": self.part,
            "num_frame_batches": parts,
            "output_frame_start": self.frames,
            "frame_start_origin": "server"
            if "output_frame_start" in header
            else "client_contiguous_batch_order",
            "num_frames": count,
            "media_seconds": count / 24,
            "arrival_seconds": arrival,
            "icl_seconds": arrival - self.chunks[-1]["arrival_seconds"]
            if self.chunks
            else None,
            "payload_bytes": len(payload),
        }
        self.chunks.append(row)
        self.frames += count
        if self.part == parts - 1:
            self.block += 1
            self.part, self.parts, self.block_request_id = 0, None, None
        else:
            self.part += 1
            self.parts, self.block_request_id = parts, request_id
        return row

    def complete(self):
        if self.frames != self.expected_frames or self.part:
            raise ValueError("Generation ended with missing frames or split batches")


def prepare_case(label, manifest, media_root, sampling, case_id=None):
    """Prepare once outside timing; preserve the canonical request's semantics."""
    case_id = case_id or CASES[label][0]
    if not manifest.is_file():
        raise ValueError(f"Fixed {label} manifest is unavailable: {manifest}")
    cases = load_cases(manifest)
    if case_id not in cases:
        raise ValueError(f"Fixed case is unavailable: {case_id}")
    case = cases[case_id]
    if "streaming_replay" in case.get("metadata", {}):
        raise ValueError("Archive replay cases require bench_streaming_source.py")
    user, target = case["messages"]
    conditions = target.get("conditions", [])
    content = target.get("content", [])
    frames = CASES[label][1] if label in CASES else content[0]["output"]["frames"]
    references = REFERENCE_COUNTS[label] if label in CASES else len(conditions)
    if (
        user.get("role") != "user"
        or target.get("role") != "target"
        or len(content) != 1
        or content[0].get("type") != "video"
        or content[0].get("output", {}).get("frames") != frames
        or target.get("source", {}).get("type") != "video"
        or len(conditions) != references
        or any(c.get("latent_modalities") != ["visual"] for c in conditions)
        or len({c.get("content_id") for c in conditions}) != references
    ):
        reference_name = {6: "six", 7: "seven"}.get(references, str(references))
        raise ValueError(
            f"Fixed {label} source/{reference_name}-reference contract does not match"
        )
    media = {c.get("id"): c for c in user.get("content", []) if "id" in c}
    if any(
        media.get(c.get("content_id"), {}).get("type") != "image" for c in conditions
    ):
        raise ValueError("Reference conditions do not map to the expected images")
    contract = case.get("metadata", {}).get("smoke_contract", {})
    if contract and contract.get("target", {}).get("fps") != 24:
        raise ValueError("Fixed payload must declare 24 FPS")
    if label in DEFAULT_CASES and case != load_cases(default_manifest(label))[case_id]:
        raise ValueError(f"Fixed {label} payload differs from the repository case")
    request = copy.deepcopy(case)
    replacements, receipts = {}, []
    for uri, asset in expected_assets(case).items():
        path = local_source_path(uri, media_root).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != asset.get("bytes")
            or sha256_file(path) != asset.get("sha256")
        ):
            raise ValueError(f"Fixed media is missing or does not match: {path}")
        replacements[uri] = path.as_uri()
        receipts.append({"source_uri": uri, "local_uri": path.as_uri(), **asset})

    def rewrite(value):
        if isinstance(value, dict):
            if value.get("uri") in replacements:
                value["uri"] = replacements[value["uri"]]
            for child in value.values():
                rewrite(child)
        elif isinstance(value, list):
            for child in value:
                rewrite(child)

    rewrite(request)
    spec = content[0]["output"]
    prompt = next(c["text"] for c in user["content"] if c.get("type") == "text")
    init = {
        **sampling,
        "type": "init",
        "prompt": prompt,
        "zing_request": request,
        "num_frames": frames,
        "size": f"{spec['width']}x{spec['height']}",
        "fps": 24,
        "realtime_output_format": "raw",
        "realtime_output_pacing": False,
        "realtime_causal_kv_cache_num_frames": -1,
        "realtime_causal_sink_size": 0,
    }
    return {
        "label": label,
        "case_id": case_id,
        "reference_count": references,
        "input_revision": case.get("provenance", {}).get("input_revision"),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "assets": receipts,
        "init": init,
        "width": spec["width"],
        "height": spec["height"],
        "frames": frames,
    }


async def receive(ws, delivery, started, clock=time.perf_counter, sink=None):
    """Timestamp immediately on complete wire receipt, before unpacking/writing."""
    import msgpack
    from websockets.exceptions import ConnectionClosedOK

    while True:
        try:
            wire = await ws.recv()
            received = clock()
        except ConnectionClosedOK:
            ended = clock() - started
            if ws.close_code != 1000 or ws.close_reason != "generation complete":
                raise ValueError("Connection closed without generation completion")
            delivery.complete()
            return ended
        message = msgpack.unpackb(wire, raw=False)
        if not isinstance(message, dict):
            raise ValueError("Invalid server message")
        kind = message.get("type")
        if kind == "error":
            raise RuntimeError(message.get("content", "Generation failed"))
        if kind == "profile_result":
            delivery.complete()
            session_id = message.get("session_id")
            dispatches = {row["server_request_id"] for row in delivery.chunks}
            if (
                delivery.profile is not None
                or not isinstance(session_id, str)
                or not session_id
                or any(
                    not dispatch.startswith(session_id + "_") for dispatch in dispatches
                )
                or any(
                    row.get("request_id") != session_id
                    or row.get("dispatch_id") not in dispatches
                    for row in message.get("stages", [])
                )
            ):
                raise ValueError("Stage profile does not match the completed session")
            profile_path = getattr(delivery, "profile_path", None)
            if profile_path is not None:
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(write_json, profile_path, message)
            delivery.profile = message
            await ws.send(
                msgpack.packb(
                    {"type": "profile_ack", "session_id": session_id}, use_bin_type=True
                )
            )
            delivery.profile_wait_seconds = (
                received - started - delivery.chunks[-1]["arrival_seconds"]
            )
            continue
        if kind == "frame_batch_header":
            payload = await ws.recv()
            received = clock()
        elif kind == "frame_batch":
            payload = message.get("payload")
        else:
            continue
        delivery.accept(message, payload, received - started)
        if sink is not None:
            sink.write(payload)


async def run_request(case, args, phase, index):
    import msgpack
    from websockets.asyncio.client import connect

    request_id = f"{case['label']}-{phase}-{index:03d}"
    delivery = Delivery(case["frames"], case["width"], case["height"])
    delivery.profile_path = args.output_dir / "profiles" / f"{request_id}.json"
    # Serialization and connection setup are outside request timing.
    init = dict(case["init"])
    if getattr(args, "receive_stage_timing", False):
        init["return_stage_timing"] = True
    payload = msgpack.packb(init, use_bin_type=True)
    mode = (
        "raw"
        if getattr(args, "save_media", False)
        else getattr(args, "media_mode", "metrics")
    )
    if mode == "representative" and (phase != "measured" or index != 0):
        mode = "metrics"
    sink = MediaCapture(
        args.output_dir / request_id,
        case["width"],
        case["height"],
        case["frames"],
        mode,
        getattr(args, "ffmpeg", "ffmpeg"),
    )
    result = {
        "request_id": request_id,
        "case_id": case["case_id"],
        "label": case["label"],
        "phase": phase,
        "success": False,
        "error": None,
        "completion_seconds": None,
    }
    try:
        async with connect(
            args.url,
            max_size=None,
            max_queue=1,
            compression=None,
            open_timeout=args.timeout,
            close_timeout=10,
        ) as ws:
            started = time.perf_counter()

            async def heartbeat():
                while True:
                    await asyncio.sleep(5)
                    await ws.send(msgpack.packb({"type": "event", "kind": "heartbeat"}))

            keepalive = asyncio.create_task(heartbeat())
            try:

                async def exchange():
                    await ws.send(payload)
                    return await receive(ws, delivery, started, sink=sink)

                result["completion_seconds"] = await asyncio.wait_for(
                    exchange(), timeout=args.timeout
                )
                result["success"] = True
            finally:
                keepalive.cancel()
                await asyncio.gather(keepalive, return_exceptions=True)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if sink is not None:
            try:
                await asyncio.to_thread(sink.finish)
            except Exception as exc:
                result["success"] = False
                result["error"] = f"{type(exc).__name__}: {exc}"
    result["profile_status"] = (
        delivery.profile["status"]
        if delivery.profile is not None
        else "missing"
        if getattr(args, "receive_stage_timing", False)
        else "not_requested"
    )
    result["profile_wait_seconds"] = delivery.profile_wait_seconds
    result["output_frames"] = delivery.frames
    result["chunk_count"] = len(delivery.chunks)
    result["metrics"] = calculate_metrics(delivery.chunks)
    result["icl_values_seconds"] = [row["icl_seconds"] for row in delivery.chunks[1:]]
    result["media"] = sink.receipt
    result["media_path"] = sink.receipt["media_path"]
    result["output_sha256"] = sink.receipt["output_sha256"]
    result["output_rgb_bytes"] = sink.receipt["output_rgb_bytes"]
    result["input_sha256"] = hashlib.sha256(payload).hexdigest()
    result["input_hash_scope"] = (
        "serialized_init_request; referenced_assets_have_separate_hashes"
    )
    result["artifact_timing_scope"] = sink.timing_scope
    result["ifl_seconds"] = None
    append_jsonl(
        args.output_dir / "chunks.jsonl",
        [{"request_id": request_id, **row} for row in delivery.chunks],
    )
    append_jsonl(args.output_dir / "requests.jsonl", [result])
    return result


def append_jsonl(path, rows):
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, allow_nan=False) + "\n")


def summarize(results, cases, measured_runs):
    summary = {}
    for case in cases:
        rows = [r for r in results if r["label"] == case["label"]]
        measured = [r for r in rows if r["phase"] == "measured"]
        passed = [r for r in measured if r["success"]]
        summary[case["label"]] = {
            "case_id": case["case_id"],
            "planned_measured": measured_runs,
            "attempted_measured": len(measured),
            "not_run_measured": measured_runs - len(measured),
            "succeeded": len(passed),
            "failed": len(measured) - len(passed),
            "warmup_failed": sum(
                not r["success"] for r in rows if r["phase"] == "warmup"
            ),
            "metrics": {
                name: distribution(
                    [
                        r["metrics"].get(name)
                        for r in passed
                        if r["metrics"].get(name) is not None
                    ]
                )
                for name in METRIC_NAMES
            },
            "icl_seconds": distribution(
                [
                    interval
                    for result in passed
                    for interval in result["icl_values_seconds"]
                ]
            ),
        }
    return summary


async def benchmark(cases, args):
    results = []
    try:
        # All requests, including warmups, are sequential and use one endpoint.
        for case in cases:
            for phase, count in (
                ("warmup", args.warmup_runs),
                ("measured", args.measured_runs),
            ):
                for index in range(count):
                    result = await run_request(case, args, phase, index)
                    results.append(result)
                    print(json.dumps(result, allow_nan=False), flush=True)
                    if not result["success"]:
                        # Cleanup after a failed session is not certified. Stop
                        # rather than start another request against a busy server.
                        return False
        return True
    finally:
        summary = summarize(results, cases, args.measured_runs)
        write_json(args.output_dir / "summary.json", summary)
        check_retention(args.output_dir)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--url", default="ws://127.0.0.1:30000/v1/realtime_video/generate"
    )
    selection = result.add_mutually_exclusive_group()
    selection.add_argument("--case", choices=["all", *CASES], default="all")
    selection.add_argument(
        "--case-set",
        help="Finite-input collection; use whitebox-basic for legacy cases",
    )
    result.add_argument(
        "--manifest-5s", type=Path, default=HERE / "whitebox_video_smoke_cases.jsonl"
    )
    result.add_argument(
        "--manifest-30s", type=Path, default=HERE / "whitebox_video_30s_cases.jsonl"
    )
    result.add_argument(
        "--manifest-long", type=Path, default=HERE / "whitebox_video_long_cases.jsonl"
    )
    result.add_argument("--media-root", required=True, type=Path)
    result.add_argument("--sampling-params", required=True, type=Path)
    result.add_argument(
        "--source-revision", help="Full Git revision for a frozen source archive"
    )
    result.add_argument(
        "--deployment",
        required=True,
        type=Path,
        help="Frozen server/model/hardware configuration JSON",
    )
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--warmup-runs", type=int, default=1)
    result.add_argument("--measured-runs", type=int, default=3)
    result.add_argument("--timeout", type=float, default=600)
    result.add_argument(
        "--save-media",
        action="store_true",
        help="Explicitly retain raw RGB (subject to the 2 GiB retention gate)",
    )
    result.add_argument(
        "--receive-stage-timing",
        action="store_true",
        default=True,
        help="Receive this session's server stage profile before normal close",
    )
    result.add_argument(
        "--no-stage-timing", dest="receive_stage_timing", action="store_false"
    )
    result.add_argument(
        "--media-mode",
        choices=("metrics", "representative", "raw", "full"),
        default="representative",
    )
    result.add_argument("--ffmpeg", default="ffmpeg")
    result.add_argument("--preflight-only", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if (
        args.warmup_runs < 1
        or args.measured_runs < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        raise ValueError("Warmup/measured counts and timeout must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        sampling = json.loads(args.sampling_params.read_text())
        if (
            not isinstance(sampling, dict)
            or set(sampling) - SAMPLING_FIELDS
            or not {"seed", "num_inference_steps"} <= sampling.keys()
        ):
            raise ValueError(
                "Provide explicit seed/num_inference_steps and supported sampling fields only"
            )
        collection, selected = select_cases(args)
        cases = [
            prepare_case(label, manifest, args.media_root, sampling, case_id)
            for label, manifest, case_id in selected
        ]
        deployment = json.loads(args.deployment.read_text())
        if not isinstance(deployment, dict) or not deployment:
            raise ValueError("Deployment configuration must be a nonempty JSON object")
        sha = args.source_revision
        if sha is not None:
            if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("Source revision must be a full lowercase Git SHA")
        else:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
            ).strip()
        config = {
            "client_sha": sha,
            "client_file_sha256": sha256_file(Path(__file__)),
            "url": args.url,
            "concurrency": 1,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "deployment": deployment,
            "deployment_source": "operator_supplied_not_server_attested",
            "sampling": sampling,
            "chunk_unit": "complete_client_RGB_frame_batch; server chunks may have multiple batches",
            "t0": "before_init_send_after_connection_and_local_preparation",
            "arrival": "complete_wire_receipt_before_unpack_and_disk_write",
            "e2e_end": "last_valid_media_batch; normal_close_after_cleanup_recorded_separately",
            "input_mode": "complete_source_via_server_readable_local_URIs; not_source_append",
            "ifl": "not_measured_no_individual_frame_timestamps",
            "cases": cases,
            "case_set": collection,
            "receive_stage_timing": args.receive_stage_timing,
            "media_mode": "raw" if args.save_media else args.media_mode,
            "artifact_timing_scope": TIMING_SCOPE,
            "retention_budget_bytes": 2 << 30,
        }
        write_json(args.output_dir / "config.json", config)
        if not args.preflight_only and not asyncio.run(benchmark(cases, args)):
            raise SystemExit(1)
    except Exception as exc:
        write_json(
            args.output_dir / "preflight-error.json",
            {"error": f"{type(exc).__name__}: {exc}"},
        )
        raise


if __name__ == "__main__":
    main()
