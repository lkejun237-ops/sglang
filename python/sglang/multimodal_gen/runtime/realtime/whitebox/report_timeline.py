# SPDX-License-Identifier: Apache-2.0
"""Verified paired archive alignment and bounded applied-output context windows."""

import hashlib
import json
import math
import re

from fastapi import Header, HTTPException, Query
from fastapi.responses import JSONResponse

MAX_READ_BYTES = 8 << 20
MAX_TIMELINE_BYTES = 128 << 10
SHA256 = re.compile(r"[a-f0-9]{64}")


async def rows(reports, manifest, descriptor):
    from .report_context import object_name

    name = descriptor.get("path")
    expected = manifest.get("objects", {}).get(name, {})
    if (
        not object_name(name)
        or type(expected.get("bytes")) is not int
        or not 0 <= expected["bytes"] <= MAX_READ_BYTES
        or not isinstance(expected.get("sha256"), str)
        or not SHA256.fullmatch(expected["sha256"])
    ):
        raise ValueError("index_read_budget_or_digest_missing")
    if manifest.get("archive_storage") == "oss_streaming_v1":
        raw = await reports.archives.read_object(
            manifest["object_prefix"] + name, MAX_READ_BYTES
        )
    else:
        root = reports.spool / manifest["recording_id"]
        path = root / name

        def read():
            if (
                path.is_symlink()
                or root.resolve() not in path.resolve().parents
                or not path.is_file()
                or path.stat().st_size > MAX_READ_BYTES
            ):
                raise ValueError("index_read_failed")
            return path.read_bytes()

        raw = await reports.run_io(read)
    if (
        len(raw) != expected["bytes"]
        or hashlib.sha256(raw).hexdigest() != expected["sha256"]
    ):
        raise ValueError("index_digest_mismatch")

    def parse():
        result = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if any(not isinstance(row, dict) for row in result):
            raise ValueError("index_invalid")
        return result

    return await reports.run_io(parse), expected["sha256"]


def playback(video):
    return (
        video.get("playback_timeline") == "mp4_preserved_pts_v1"
        and type(video.get("playback_time_origin_us")) is int
        and video["playback_time_origin_us"] == 0
        and type(video.get("first_pts_us")) is int
    )


async def comparison(reports, manifest):
    base = dict(
        schema_version=1,
        recording_id=manifest["recording_id"],
        source_session_id=manifest.get("source_session_id"),
        source_epoch=manifest.get("source_epoch"),
        segment_id=manifest.get("segment_id"),
        segment_epoch=manifest.get("segment_epoch"),
        source_artifact_id="video:whitebox",
        model_artifact_id="video:model",
    )

    def unavailable(reason, status="unavailable"):
        return dict(base, status=status, reason=reason)

    if manifest.get("recording_kind") != "render_segment":
        return unavailable("paired_model_recording_required")
    if any(
        not isinstance(manifest.get(key), str) or not manifest[key]
        for key in ("source_session_id", "source_epoch", "segment_id", "segment_epoch")
    ):
        return unavailable("paired_recording_identity_missing")
    if manifest.get("upload_status") != "complete":
        return unavailable("archive_not_complete", "pending")
    if manifest.get("recording_status") != "complete":
        return unavailable("paired_recording_incomplete")
    source, model = (
        manifest.get("videos", {}).get(name, {}) for name in ("whitebox", "model")
    )
    if not source or not model:
        return unavailable("paired_video_missing")
    for video in (source, model):
        artifact = manifest.get("objects", {}).get(video.get("path"), {})
        if (
            not isinstance(artifact.get("sha256"), str)
            or not SHA256.fullmatch(artifact["sha256"])
            or type(artifact.get("bytes")) is not int
            or artifact["bytes"] <= 0
        ):
            return unavailable("paired_video_receipt_missing")
    count = source.get("frames")
    if type(count) is not int or count <= 0 or model.get("frames") != count:
        return unavailable("paired_frame_count_mismatch")
    if not playback(source) or not playback(model):
        return unavailable("archive_playback_timeline_unknown")
    descriptor = manifest.get("report_frames", {})
    if descriptor.get("format") != "report_frame_index_v1":
        return unavailable("paired_frame_index_missing")
    try:
        index, digest = await rows(reports, manifest, descriptor)

        def validate():
            if len(index) != count:
                return unavailable("paired_index_count_mismatch")
            previous = -1
            for frame, row in enumerate(index):
                pts = row.get("archive_pts_us", {})
                left, right = pts.get("whitebox"), pts.get("model")
                if (
                    row.get("schema_version") != 1
                    or row.get("frame") != frame
                    or type(left) is not int
                    or type(right) is not int
                    or left != right
                    or left <= previous
                ):
                    return unavailable("paired_frame_pts_mismatch")
                previous = left
            first, last = index[0]["archive_pts_us"]["model"], previous
            if source["first_pts_us"] != first or model["first_pts_us"] != first:
                return unavailable("paired_video_origin_mismatch")
            return dict(
                base,
                status="aligned",
                method="shared_recording_frame_index_and_pts",
                frames=count,
                first_pts_us=first,
                last_pts_us=last,
                first_playback_seconds=first / 1_000_000,
                last_playback_seconds=last / 1_000_000,
                playback_timeline="mp4_preserved_pts_v1",
                playback_time_origin_us=0,
                index_sha256=digest,
            )

        return await reports.run_io(validate)
    except Exception:
        return unavailable("paired_index_unavailable")


def prompt_context(prompt, references, version, manifest):
    if (
        not isinstance(prompt, str)
        or len(prompt) > 16384
        or not isinstance(references, list)
        or len(references) > 9
    ):
        raise ValueError("output_context_invalid")
    value = dict(text=prompt, sha256=hashlib.sha256(prompt.encode()).hexdigest())
    if type(version) is int and 0 <= version < 2**53:
        value["version"] = version
    artifacts = sorted(
        name for name in manifest.get("objects", {}) if name.startswith("references/")
    )[:64]
    safe = []
    for reference in references:
        name, digest = reference.get("id"), reference.get("sha256")
        if (
            not isinstance(name, str)
            or len(name) > 128
            or not isinstance(digest, str)
            or not SHA256.fullmatch(digest)
        ):
            raise ValueError("output_reference_invalid")
        ref = dict(id=name, sha256=digest)
        for index, path in enumerate(artifacts):
            if manifest["objects"][path].get("sha256") == digest:
                ref["artifact_id"] = "reference:" + str(index)
                break
        safe.append(ref)
    return dict(prompt=value, references=safe)


async def timeline(reports, manifest, start, end, limit):
    base = dict(
        schema_version=1,
        recording_id=manifest["recording_id"],
        identity={
            key: manifest.get(key)
            for key in (
                "recording_id",
                "source_session_id",
                "source_epoch",
                "segment_id",
                "segment_epoch",
            )
        },
        semantic="model_output_applied",
        window=dict(from_seconds=start, to_seconds=end),
        events=[],
        includes_prior_context=False,
        truncated=False,
        next_from_seconds=None,
    )

    def unavailable(reason, status="unavailable"):
        return dict(base, status=status, reason=reason)

    if manifest.get("recording_kind") != "render_segment":
        return unavailable("model_segment_required")
    if manifest.get("upload_status") != "complete":
        return unavailable("archive_not_complete", "pending")
    if manifest.get("recording_status") != "complete":
        return unavailable("model_recording_incomplete")
    if not playback(manifest.get("videos", {}).get("model", {})):
        return unavailable("archive_playback_timeline_unknown")
    try:
        descriptor = manifest.get("output_contexts", {})
        compact = descriptor.get("format") == "model_output_context_v1"
        if compact:
            if descriptor.get("frames") != manifest["videos"]["model"].get("frames"):
                return unavailable("output_context_coverage_unproven")
            contexts, digest = await rows(reports, manifest, descriptor)
        else:
            # Only model segment inputs are written from actual output-selected
            # prompt/ref context. Capture timestamps are NOT playback positions.
            contexts, digest = await rows(reports, manifest, manifest.get("inputs", {}))
        frame_descriptor = manifest.get("report_frames", {})
        if frame_descriptor.get("format") != "report_frame_index_v1":
            return unavailable("output_frame_index_missing")
        frames, _ = await rows(reports, manifest, frame_descriptor)

        def project():
            if not frames or len(frames) != manifest["videos"]["model"].get("frames"):
                raise ValueError("output_frame_count_invalid")
            index = {}
            last_pts = -1
            for number, frame in enumerate(frames):
                pts = frame.get("archive_pts_us", {}).get("model")
                source_frame = frame.get("source_frame_index")
                if (
                    frame.get("frame") != number
                    or type(pts) is not int
                    or pts <= last_pts
                    or source_frame is not None
                    and (type(source_frame) is not int or not 0 <= source_frame < 2**53)
                ):
                    raise ValueError("output_frame_order_invalid")
                last_pts = pts
                index[number] = frame
            coverage = dict(
                from_seconds=frames[0]["archive_pts_us"]["model"] / 1_000_000,
                to_seconds=last_pts / 1_000_000,
            )
            projected, previous, last_position = [], None, -1
            input_end = 0
            for row in contexts:
                if compact:
                    frame = index.get(row.get("frame"))
                    if (
                        frame is None
                        or row.get("archive_pts_us", {}).get("model")
                        != frame["archive_pts_us"]["model"]
                        or row.get("source_frame_index")
                        != frame.get("source_frame_index")
                    ):
                        raise ValueError("output_context_frame_mismatch")
                    prompt = row.get("prompt", {})
                    text, version = prompt.get("text"), prompt.get("version")
                    refs = row.get("references", [])
                    if row.get("schema_version") != 1:
                        raise ValueError("output_context_invalid")
                else:
                    if (
                        type(row.get("start_frame")) is not int
                        or type(row.get("end_frame")) is not int
                        or row["start_frame"] != input_end
                        or row["end_frame"] <= input_end
                        or row["end_frame"] > len(frames)
                    ):
                        raise ValueError("output_input_coverage_invalid")
                    input_end = row["end_frame"]
                    frame = index.get(row.get("start_frame"))
                    if frame is None:
                        raise ValueError("output_context_frame_missing")
                    text, version = (
                        row.get("prompt"),
                        row.get("input", {}).get("rendering", {}).get("prompt_version"),
                    )
                    refs = row.get("input", {}).get("references", [])
                current = prompt_context(text, refs, version, manifest)
                if compact and prompt.get("sha256") != current["prompt"]["sha256"]:
                    raise ValueError("output_prompt_digest_mismatch")
                pts = frame.get("archive_pts_us", {}).get("model")
                if (
                    type(pts) is not int
                    or pts < manifest["videos"]["model"]["first_pts_us"]
                ):
                    raise ValueError("output_context_pts_missing")
                position = pts / 1_000_000
                if position <= last_position:
                    raise ValueError("output_context_order_invalid")
                last_position = position
                if current == previous:
                    continue
                previous = current
                projected.append(
                    dict(
                        frame_index=frame["frame"],
                        source_frame_index=frame.get("source_frame_index"),
                        playback_seconds=position,
                        **current,
                    )
                )
            if not compact and input_end != len(frames):
                raise ValueError("output_input_coverage_incomplete")
            prior = (
                [event for event in projected if event["playback_seconds"] <= start]
                if start <= coverage["to_seconds"]
                else []
            )
            selected = (prior[-1:] if prior else []) + [
                event for event in projected if start < event["playback_seconds"] <= end
            ]
            result = dict(
                base,
                status="available",
                includes_prior_context=bool(prior),
                provenance=(
                    "model_output_context_v1" if compact else "model_output_inputs_v1"
                ),
                index_sha256=digest,
                coverage=coverage,
            )
            budget = 2048
            for event in selected:
                size = len(json.dumps(event, ensure_ascii=False).encode())
                if len(result["events"]) >= limit or budget + size > MAX_TIMELINE_BYTES:
                    if not result["events"]:
                        return unavailable("timeline_event_budget")
                    result.update(
                        truncated=True,
                        next_from_seconds=event["playback_seconds"],
                        reason="timeline_window_budget",
                    )
                    break
                result["events"].append(event)
                budget += size
            if not projected:
                return unavailable("model_output_context_missing")
            return result

        return await reports.run_io(project)
    except Exception:
        return unavailable("output_context_unavailable")


def install_report_timeline(app, reports):
    @app.get("/v1/whitebox/recordings/{recording_id}/context/timeline")
    async def read_timeline(
        recording_id: str,
        authorization: str = Header(default=""),
        segment_id: str = Query(default="", max_length=32),
        segment_epoch: str = Query(default="", max_length=128),
        from_seconds: float = Query(default=0, ge=0),
        to_seconds: float = Query(default=None, ge=0),
        limit: int = Query(default=64, ge=1, le=128),
    ):
        from .report_context import selected

        parent, child = await selected(
            reports, recording_id, authorization, segment_id, segment_epoch
        )
        end = from_seconds + 300 if to_seconds is None else to_seconds
        if (
            not math.isfinite(from_seconds)
            or not math.isfinite(end)
            or end <= from_seconds
            or end - from_seconds > 300
        ):
            raise HTTPException(
                400, "Select a finite timeline window of at most 300 seconds"
            )
        value = await timeline(reports, child or parent, from_seconds, end, limit)
        value["source_recording_id"] = parent["recording_id"]
        return JSONResponse(value, headers={"Cache-Control": "private, no-store"})
