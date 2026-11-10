# SPDX-License-Identifier: Apache-2.0
"""Small, capability-scoped evidence snapshots; never a second archive owner."""

import copy
import hashlib
import json
import time

from fastapi import Header, HTTPException, Query
from fastapi.responses import JSONResponse

IDENTITY_FIELDS = (
    "recording_id",
    "recording_kind",
    "world_id",
    "trace_id",
    "source_session_id",
    "source_epoch",
    "segment_id",
    "segment_epoch",
    "created_at",
)
MULTIPLAYER_FIELDS = (
    "game_id",
    "player_id",
    "stream_id",
    "user_id",
    "owner_id",
    "owner_epoch",
)
MAX_STATES_BYTES = 8 << 20
MAX_CONTEXT_BYTES = 256 << 10


def scalars(value, fields):
    return {
        key: item
        for key in fields
        if isinstance(item := value.get(key), (str, int, float, bool))
        and (not isinstance(item, str) or len(item) <= 1024)
    }


def object_name(value):
    return (
        isinstance(value, str)
        and 0 < len(value) <= 2048
        and not value.startswith("/")
        and "\\" not in value
        and not any(part in {"", ".", ".."} for part in value.split("/"))
        and not any(c in value for c in "?:#\r\n")
    )


def snapshot(manifest, store, segment_id=None):
    identity = scalars(manifest, IDENTITY_FIELDS)
    identity["multiplayer"] = scalars(
        manifest.get("multiplayer") or {}, MULTIPLAYER_FIELDS
    )
    prefix = manifest.get("object_prefix", "")
    if not object_name(prefix.rstrip("/")) or not prefix.endswith("/"):
        raise HTTPException(503, "Recording identity is unavailable")
    objects = manifest.get("objects", {})
    ready = manifest.get("upload_status") == "complete"
    artifacts = []

    def add(artifact_id, role, path, **metadata):
        if not object_name(path):
            return
        expected = objects.get(path, {})
        available = ready and (path == "manifest.json" or bool(expected))
        artifacts.append(
            dict(
                artifact_id=artifact_id,
                role=role,
                recording_id=identity.get("recording_id"),
                availability=(
                    "available"
                    if available
                    else "pending"
                    if not ready
                    else "unavailable"
                ),
                locator=dict(
                    provider="oss",
                    bucket=getattr(store, "bucket", None),
                    region=getattr(store, "worker_config", {}).get("region"),
                    object_key=prefix + path,
                ),
                **scalars(expected, ("sha256", "bytes")),
                **metadata,
                access=dict(artifact_id=artifact_id, segment_id=segment_id),
            )
        )

    add("manifest", "manifest", "manifest.json")
    for role in ("inputs", "states"):
        add(role, role, manifest.get(role, {}).get("path", role + ".jsonl"))
    for name in ("whitebox", "model", "ui_rendered"):
        video = manifest.get("videos", {}).get(name)
        if video:
            add(
                "video:" + name,
                "video",
                video.get("path"),
                stream=name,
                **scalars(
                    video,
                    (
                        "frames",
                        "codec",
                        "first_pts_us",
                        "duration_us",
                        "playback_timeline",
                        "playback_time_origin_us",
                    ),
                ),
            )
    client_ui = manifest.get("client_ui", {})
    if client_ui:
        receipt = manifest.get("_report_client_receipt", {})
        video = receipt.get("video", {})
        if receipt.get("status") == "complete" and receipt.get(
            "recording_id"
        ) == identity.get("recording_id"):
            objects = dict(objects, **{video.get("path", ""): video})
            add(
                "client_ui",
                "video",
                video.get("path"),
                stream="ui_rendered",
                **scalars(video, ("frames", "codec")),
            )
        else:
            artifacts.append(
                dict(
                    artifact_id="client_ui",
                    role="video",
                    stream="ui_rendered",
                    recording_id=identity.get("recording_id"),
                    locator=None,
                    availability=(
                        "unavailable"
                        if client_ui.get("status") in {"failed", "unavailable"}
                        else "pending"
                    ),
                    access=dict(artifact_id="client_ui", segment_id=segment_id),
                )
            )
    # Reference count is bounded independently from arbitrary manifest objects.
    for index, name in enumerate(
        sorted(n for n in objects if n.startswith("references/"))[:64]
    ):
        expected = objects[name]
        digest = expected.get("sha256", "")
        mime = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
        }.get(name.rsplit(".", 1)[-1].lower())
        metadata = {}
        if (
            mime
            and isinstance(digest, str)
            and len(digest) == 64
            and set(digest) <= set("0123456789abcdef")
            and type(expected.get("bytes")) is int
            and expected["bytes"] > 0
        ):
            metadata = dict(
                content_type=mime, content_type_provenance="archive_reference_extension"
            )
        add("reference:" + str(index), "reference", name, **metadata)
    return dict(
        identity=identity,
        identity_provenance="archive_manifest",
        **scalars(
            manifest,
            ("recording_status", "upload_status", "dataset_status", "archive_storage"),
        ),
        timeline=scalars(
            manifest.get("timeline", {}),
            (
                "mode",
                "pts_unit",
                "first_frame",
                "end_frame",
                "end_time_us",
                "alignment",
            ),
        ),
        client_ui_status=client_ui.get("status", "not_applicable"),
        deployment=scalars(
            manifest.get("inference", {}).get("deployment", {}),
            (
                "model_revision",
                "routing_revision",
                "vae_fingerprint",
                "worker_id",
                "model",
                "model_name",
                "source_sha",
                "image_digest",
            ),
        ),
        deployment_provenance="recorded_manifest_metadata",
        artifacts=artifacts,
    )


async def selected(reports, recording_id, authorization, segment_id, segment_epoch=""):
    await reports.access(recording_id, authorization)
    parent = await reports.manifest(recording_id)
    if parent.get("recording_id") != recording_id:
        raise HTTPException(503, "Recording identity is unavailable")
    if segment_epoch:
        matches = [
            entry
            for entry in parent.get("render_segments", [])
            if entry.get("segment_epoch") == segment_epoch
            and (not segment_id or entry.get("recording_id") == segment_id)
        ]
        if len(matches) != 1:
            raise HTTPException(404)
        segment_id = matches[0].get("recording_id", "")
    if segment_id:
        child = await reports.owned_segment(parent, segment_id)
        if child.get("recording_id") != segment_id:
            raise HTTPException(503, "Recording identity is unavailable")
        if segment_epoch and child.get("segment_epoch") != segment_epoch:
            raise HTTPException(404)
        return parent, child
    return parent, None


async def with_client_receipt(reports, manifest):
    if not manifest.get("client_ui") or manifest.get("upload_status") != "complete":
        return manifest
    value = copy.deepcopy(manifest)
    receipt = None
    if getattr(reports.archives, "diskless", False):
        intent, receipt = await reports.client_recordings.streaming.status(
            manifest["recording_id"], manifest
        )
        if intent:
            value["client_ui"]["status"] = intent.get("status", "unknown")
    else:
        path = reports.spool / manifest["recording_id"] / "ui-recording.json"

        def read():
            if (
                path.is_file()
                and not path.is_symlink()
                and path.stat().st_size <= 32768
            ):
                return json.loads(path.read_text())

        receipt = await reports.run_io(read)
    if receipt:
        value["_report_client_receipt"] = receipt
        value["client_ui"]["status"] = receipt.get("status", "unknown")
    return value


async def frame_mapping(
    reports, manifest, frame_index, stream, source_frame_index=None
):
    """Resolve only actual mux timestamps, never extrapolate a frame clock."""
    missing = dict(status="unavailable", reason="archive_frame_mapping_not_requested")
    if frame_index is None and source_frame_index is None:
        return missing
    video = manifest.get("videos", {}).get(stream, {})
    if not isinstance(video.get("first_pts_us"), int):
        return dict(status="unavailable", reason="archive_video_pts_origin_missing")
    if (
        video.get("playback_timeline") != "mp4_preserved_pts_v1"
        or type(video.get("playback_time_origin_us")) is not int
        or video["playback_time_origin_us"] != 0
    ):
        return dict(status="unavailable", reason="archive_playback_timeline_unknown")
    if manifest.get("upload_status") != "complete":
        return dict(status="pending", reason="archive_not_complete")
    sidecar = manifest.get("report_frames", {})
    indexed = sidecar.get("format") == "report_frame_index_v1"
    name = (
        sidecar.get("path")
        if indexed
        else manifest.get("states", {}).get("path", "states.jsonl")
    )
    expected = manifest.get("objects", {}).get(name, {})
    if (
        not object_name(name)
        or not isinstance(expected.get("bytes"), int)
        or expected["bytes"] > MAX_STATES_BYTES
        or not expected.get("sha256")
    ):
        return dict(status="unavailable", reason="states_read_budget_or_digest_missing")
    try:
        if manifest.get("archive_storage") == "oss_streaming_v1":
            raw = await reports.archives.read_object(
                manifest["object_prefix"] + name, MAX_STATES_BYTES
            )
        else:
            path = reports.spool / manifest["recording_id"] / name

            def read():
                root = reports.spool / manifest["recording_id"]
                if (
                    path.is_symlink()
                    or root.resolve() not in path.resolve().parents
                    or not path.is_file()
                    or path.stat().st_size > MAX_STATES_BYTES
                ):
                    raise ValueError("Invalid states archive")
                return path.read_bytes()

            raw = await reports.run_io(read)
        if (
            len(raw) != expected["bytes"]
            or hashlib.sha256(raw).hexdigest() != expected["sha256"]
        ):
            raise ValueError("States archive changed")

        def resolve():
            matches = []
            for line in raw.splitlines():
                row = json.loads(line)
                if frame_index is not None and row.get("frame") != frame_index:
                    continue
                if (
                    source_frame_index is not None
                    and (
                        row.get("source_frame_index")
                        if indexed
                        else row.get("source", {}).get("source_frame_index")
                    )
                    != source_frame_index
                ):
                    continue
                matches.append(row)
                if len(matches) > 1:
                    return dict(
                        status="unavailable", reason="source_frame_mapping_ambiguous"
                    )
            if matches:
                row = matches[0]
                pts = row.get("archive_pts_us", {}).get(stream)
                method = "states_archive_pts"
                if indexed:
                    method = "report_frame_index_v1"
                if pts is None and manifest.get("archive_encoding_version") == 2:
                    pts = row.get("presentation_pts_us")
                    method = "states_presentation_pts"
                if not isinstance(pts, int) or pts < video["first_pts_us"]:
                    return dict(status="unavailable", reason="frame_pts_missing")
                return dict(
                    status="mapped",
                    recording_id=manifest["recording_id"],
                    artifact_id="video:" + stream,
                    frame_index=row["frame"],
                    source_frame_index=source_frame_index,
                    pts_us=pts,
                    seek_seconds=(pts - video["playback_time_origin_us"]) / 1_000_000,
                    playback_timeline=video["playback_timeline"],
                    playback_time_origin_us=video["playback_time_origin_us"],
                    method=method,
                    precision="archive_frame",
                    index_sha256=expected["sha256"],
                    index_path=name,
                )
            return dict(status="unavailable", reason="archive_frame_not_found")

        return await reports.run_io(resolve)
    except Exception:
        return dict(status="unavailable", reason="states_read_failed")


def install_report_context(app, reports):
    from .report_timeline import comparison, install_report_timeline

    install_report_timeline(app, reports)

    @app.get("/v1/whitebox/recordings/{recording_id}/context")
    async def context(
        recording_id: str,
        authorization: str = Header(default=""),
        segment_id: str = Query(default="", max_length=32),
        segment_epoch: str = Query(default="", max_length=128),
        frame_index: int = Query(default=None, ge=0, le=2**53 - 1),
        source_frame_index: int = Query(default=None, ge=0, le=2**53 - 1),
        stream: str = Query(default="whitebox", pattern="^(whitebox|model)$"),
    ):
        parent, child = await selected(
            reports, recording_id, authorization, segment_id, segment_epoch
        )
        if frame_index is not None and source_frame_index is not None:
            raise HTTPException(400, "Select one archive or source frame identity")
        parent = await with_client_receipt(reports, parent)
        store = reports.archives.store
        source = snapshot(parent, store)
        segments = [snapshot(child, store, child["recording_id"])] if child else []
        source["comparison"] = await comparison(reports, parent)
        if child:
            segments[0]["comparison"] = await comparison(reports, child)
        timings = []
        for manifest in (parent, child):
            if manifest is None:
                continue
            live = reports.live.get(manifest["recording_id"])
            rows = live.snapshot() if live else manifest.get("trace", [])
            timings.extend(
                scalars(
                    row,
                    (
                        "stage",
                        "count",
                        "errors",
                        "mean_ms",
                        "min_ms",
                        "max_ms",
                        "last_ms",
                        "age_ms",
                        "p95_ms",
                        "p95_window",
                    ),
                )
                for row in rows[:64]
            )
        value = dict(
            schema_version=1,
            recording_id=recording_id,
            observed_at_ms=int(time.time() * 1000),
            identity=source["identity"],
            identity_provenance="archive_manifest",
            source=source,
            segments=segments,
            segment_selection="exact" if child else "not_requested",
            mapping=await frame_mapping(
                reports, child or parent, frame_index, stream, source_frame_index
            ),
            timings=timings,
        )
        if len(json.dumps(value, allow_nan=False).encode()) > MAX_CONTEXT_BYTES:
            raise HTTPException(503, "Recording context exceeds its read budget")
        return JSONResponse(
            value,
            headers={"Cache-Control": "private, no-store"},
        )

    @app.get("/v1/whitebox/recordings/{recording_id}/context/artifact-access")
    async def artifact_access(
        recording_id: str,
        artifact_id: str = Query(max_length=128),
        segment_id: str = Query(default="", max_length=32),
        authorization: str = Header(default=""),
    ):
        parent, child = await selected(reports, recording_id, authorization, segment_id)
        manifest = await with_client_receipt(reports, child or parent)
        value = snapshot(manifest, reports.archives.store, segment_id or None)
        artifact = next(
            (item for item in value["artifacts"] if item["artifact_id"] == artifact_id),
            None,
        )
        if not artifact:
            raise HTTPException(404)
        if artifact["availability"] != "available":
            raise HTTPException(409, "The recording artifact is not ready yet.")
        signer = getattr(reports.archives.store, "playback_url", None)
        if not signer:
            raise HTTPException(503, "Recording playback is unavailable")
        return JSONResponse(
            dict(
                artifact_id=artifact_id,
                recording_id=value["identity"]["recording_id"],
                url=signer(artifact["locator"]["object_key"]),
                expires_at_ms=int(time.time() * 1000) + 900_000,
            ),
            headers={"Cache-Control": "private, no-store"},
        )
