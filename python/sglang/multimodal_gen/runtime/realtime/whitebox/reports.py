# SPDX-License-Identifier: Apache-2.0
"""Bounded per-session timings and capability-scoped archive status.

The capability is returned only on that session's WebSocket. Its hash is stored
outside the dataset; replay inputs and UI resources require the same capability.
"""

import asyncio
import copy
import hashlib
import hmac
import json
import math
import re
import secrets
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import Header, HTTPException
from fastapi.responses import FileResponse

from sglang.multimodal_gen.runtime.utils.realtime_trace import (
    register_realtime_trace_sink,
    unregister_realtime_trace_sink,
)


class SessionReport:
    def __init__(self, trace_id):
        self.trace_id = trace_id
        self.token = secrets.token_urlsafe(32)
        self.timings = {}
        # Count the post-archive subscriber boundary separately from transport
        # and model-input drops. This includes all attachments of this session.
        self.playback_delivery = {}
        self.lock = threading.RLock()
        self.recording = None
        self.closed = False
        self.gateway_done = asyncio.Event()
        register_realtime_trace_sink(trace_id, self.observe)

    def observe(self, payload):
        name = str(payload.get("event", "unknown"))[:128]
        if isinstance(payload.get("stage"), str):
            name += "." + payload["stage"][:128]
        # Events may carry multiple durations; preserve each named measurement.
        values = {
            key: value
            for key, value in payload.items()
            if key.endswith("_ms")
            and key
            not in {"server_epoch_ms", "server_elapsed_ms", "epoch_ms", "timestamp_ms"}
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0
        }
        if not values:
            return
        with self.lock:
            for field, value in values.items():
                label = name if field == "duration_ms" else f"{name}.{field}"
                if label not in self.timings and len(self.timings) >= 128:
                    continue
                item = self.timings.setdefault(
                    label,
                    dict(
                        count=0,
                        total_ms=0.0,
                        min_ms=value,
                        max_ms=0.0,
                        errors=0,
                        samples=deque(maxlen=512),
                    ),
                )
                item["count"] += 1
                item["total_ms"] += value
                item["min_ms"] = min(item["min_ms"], value)
                item["max_ms"] = max(item["max_ms"], value)
                item["errors"] += int(payload.get("result") == "error")
                item["samples"].append(value)
                item["last_ms"] = value
                item["observed_at"] = time.monotonic()

    def snapshot(self):
        with self.lock:
            return [
                dict(
                    stage=name,
                    count=item["count"],
                    errors=item["errors"],
                    mean_ms=item["total_ms"] / item["count"],
                    min_ms=item["min_ms"],
                    max_ms=item["max_ms"],
                    last_ms=item["last_ms"],
                    age_ms=(time.monotonic() - item["observed_at"]) * 1000,
                    p95_ms=sorted(item["samples"])[
                        math.ceil(len(item["samples"]) * 0.95) - 1
                    ],
                    p95_window=len(item["samples"]),
                )
                for name, item in sorted(self.timings.items())
            ]

    def close(self):
        if not self.closed:
            unregister_realtime_trace_sink(self.trace_id, self.observe)
            self.closed = True


class Reports:
    def __init__(self, spool, archives):
        self.spool, self.archives = Path(spool), archives
        self.live = {}
        self.io_active = 0
        self.io_timeout = 3.0
        self.io_available = asyncio.Event()
        self.io_available.set()
        self.ui_io_waiters = 0
        # One supported UI bundle may fetch up to 128 resources concurrently.
        # Waiters hold no disk worker and share the existing operation deadline.
        self.ui_io_max_waiters = 128

    async def run_io(self, function, *args, wait_for_slot=False):
        """Keep two disk workers; mandatory UI reads may wait in a bounded queue."""
        deadline = asyncio.get_running_loop().time() + self.io_timeout
        if self.io_active >= 2:
            if not wait_for_slot or self.ui_io_waiters >= self.ui_io_max_waiters:
                raise HTTPException(503, "Recording storage is busy")
            self.ui_io_waiters += 1
            try:
                async with asyncio.timeout_at(deadline):
                    while self.io_active >= 2:
                        self.io_available.clear()
                        await self.io_available.wait()
            except TimeoutError:
                raise HTTPException(503, "Recording storage is slow") from None
            finally:
                self.ui_io_waiters -= 1
        if asyncio.get_running_loop().time() >= deadline:
            raise HTTPException(503, "Recording storage is slow")
        # No await separates slot observation from reservation.
        self.io_active += 1
        task = asyncio.create_task(asyncio.to_thread(function, *args))

        def done(result):
            self.io_active -= 1
            self.io_available.set()
            if not result.cancelled():
                result.exception()

        task.add_done_callback(done)
        try:
            return await asyncio.wait_for(
                asyncio.shield(task),
                max(0, deadline - asyncio.get_running_loop().time()),
            )
        except TimeoutError:
            raise HTTPException(503, "Recording storage is slow") from None

    async def access(self, recording_id, authorization, *, wait_for_io=False):
        if recording_id in self.live:
            return self.authorize(recording_id, authorization)
        try:
            # Retained pre-migration recordings remain read-only accessible.
            return await self.run_io(
                self.authorize, recording_id, authorization, wait_for_slot=wait_for_io
            )
        except HTTPException as error:
            if error.status_code != 404 or not getattr(
                self.archives, "diskless", False
            ):
                raise
        if not re.fullmatch(r"[a-f0-9]{32}", recording_id):
            raise HTTPException(404)
        try:
            index = await self.archives.read_index(recording_id)
        except FileNotFoundError:
            raise HTTPException(404) from None
        except Exception:
            raise HTTPException(503, "Recording storage is unavailable") from None
        token = (
            authorization.removeprefix("Bearer ")
            if authorization.startswith("Bearer ")
            else ""
        )
        if not token or not hmac.compare_digest(
            index.get("token_hash", ""), hashlib.sha256(token.encode()).hexdigest()
        ):
            raise HTTPException(404)
        return self.spool / recording_id

    def attach(self, report, recording):
        recording.trace_snapshot = report.snapshot
        report.recording = recording
        self.live[recording.directory.name] = report
        if hasattr(recording, "report_access"):
            recording.report_access(hashlib.sha256(report.token.encode()).hexdigest())
            self._prune()
            return
        root = self.spool / ".reports"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = root / (recording.directory.name + ".json")
        path.write_text(
            json.dumps(
                {"token_hash": hashlib.sha256(report.token.encode()).hexdigest()}
            )
        )
        path.chmod(0o600)
        self.live[recording.directory.name] = report
        self._prune()

    def _prune(self):
        # Active sessions/archives remain observable; closed reports are durable.
        for key, value in list(self.live.items()):
            if len(self.live) <= 128:
                break
            if value.closed:
                del self.live[key]

    def authorize(self, recording_id, authorization):
        if not re.fullmatch(r"[a-f0-9]{32}", recording_id):
            raise HTTPException(404)
        live = self.live.get(recording_id)
        if live is not None:
            if not hmac.compare_digest("Bearer " + live.token, authorization):
                raise HTTPException(404)
            return self.spool / recording_id
        access = self.spool / ".reports" / (recording_id + ".json")
        if not access.is_file():
            raise HTTPException(404)
        token = (
            authorization.removeprefix("Bearer ")
            if authorization.startswith("Bearer ")
            else ""
        )
        expected = json.loads(access.read_text())["token_hash"]
        if not token or not hmac.compare_digest(
            expected, hashlib.sha256(token.encode()).hexdigest()
        ):
            raise HTTPException(404)
        return self.spool / recording_id

    async def manifest(self, recording_id, *, wait_for_io=False):
        report = self.live.get(recording_id)
        recording = report.recording if report else None
        if recording is not None and hasattr(recording, "pair"):
            return dict(recording.manifest)
        if report and getattr(recording, "diskless", False):
            return copy.deepcopy(recording.manifest)
        try:
            return await self.run_io(
                lambda: json.loads(
                    (self.spool / recording_id / "manifest.json").read_text()
                ),
                wait_for_slot=wait_for_io,
            )
        except FileNotFoundError:
            if not getattr(self.archives, "diskless", False):
                raise
        try:
            return await self.archives.read_manifest(recording_id)
        except FileNotFoundError:
            raise HTTPException(404) from None
        except Exception:
            raise HTTPException(503, "Recording storage is unavailable") from None

    async def owned_segment(self, parent, segment_id):
        if (
            parent.get("recording_kind") != "source"
            or not parent.get("source_session_id")
            or not re.fullmatch(r"[a-f0-9]{32}", segment_id)
            or not any(
                entry.get("recording_id") == segment_id
                for entry in parent.get("render_segments", [])
            )
        ):
            raise HTTPException(404)
        child = await self.manifest(segment_id)
        if (
            child.get("recording_kind") != "render_segment"
            or child.get("source_session_id") != parent["source_session_id"]
            or child.get("source_epoch") != parent.get("source_epoch")
            or child.get("object_prefix")
            != parent["object_prefix"] + "segments/" + segment_id + "/"
        ):
            raise HTTPException(404)
        return child

    async def read_inputs(self, recording_id):
        manifest = (
            await self.manifest(recording_id)
            if getattr(self.archives, "diskless", False)
            else {}
        )
        if manifest.get("archive_storage") == "oss_streaming_v1":
            try:
                raw = await self.archives.read_object(
                    manifest["object_prefix"] + "inputs.jsonl", 32 << 20
                )
                expected = manifest.get("objects", {}).get("inputs.jsonl", {})
                if len(raw) != expected.get("bytes") or hashlib.sha256(
                    raw
                ).hexdigest() != expected.get("sha256"):
                    raise ValueError("Input archive changed")
                rows = await asyncio.to_thread(
                    lambda: [
                        json.loads(line) for line in raw.splitlines() if line.strip()
                    ]
                )
                return dict(time_unit="us", interval="[start,end)", intervals=rows)
            except Exception:
                raise HTTPException(503, "Replay inputs are unavailable") from None
        path = self.spool / recording_id / "inputs.jsonl"

        def read():
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size > 32 * 1024 * 1024
            ):
                raise HTTPException(409, "Replay inputs are not available")
            return [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]

        return dict(
            time_unit="us", interval="[start,end)", intervals=await self.run_io(read)
        )

    def install(self, app):
        from .client_recording import ClientRecordings

        @app.get("/v1/whitebox/recordings/{recording_id}/segments/{segment_id}/inputs")
        async def segment_inputs(
            recording_id: str, segment_id: str, authorization: str = Header(default="")
        ):
            await self.access(recording_id, authorization)
            child = await self.owned_segment(
                await self.manifest(recording_id), segment_id
            )
            if child.get("upload_status") != "complete":
                raise HTTPException(409, "The segment recording is not ready yet.")
            return await self.read_inputs(segment_id)

        self.client_recordings = ClientRecordings(self)
        self.client_recordings.install(app)

        @app.get("/v1/whitebox/recordings/{recording_id}/inputs")
        async def recording_inputs(
            recording_id: str, authorization: str = Header(default="")
        ):
            directory = await self.access(recording_id, authorization)
            if (await self.manifest(recording_id)).get("upload_status") != "complete":
                raise HTTPException(409, "Wait for the recording to finish uploading")
            return await self.read_inputs(recording_id)

        @app.get("/v1/whitebox/recordings/{recording_id}/audio/{audio_path:path}")
        async def audio_asset(
            recording_id: str, audio_path: str, authorization: str = Header(default="")
        ):
            from .project_audio import AUDIO_PREFIX, audio_path_allowed

            if not audio_path_allowed(audio_path):
                raise HTTPException(404)
            return await ui_asset(
                recording_id, AUDIO_PREFIX + audio_path, authorization
            )

        @app.get("/v1/whitebox/recordings/{recording_id}/ui/{asset_path:path}")
        async def ui_asset(
            recording_id: str, asset_path: str, authorization: str = Header(default="")
        ):
            directory = await self.access(recording_id, authorization, wait_for_io=True)
            manifest = await self.manifest(recording_id, wait_for_io=True)
            expected = manifest.get("ui", {}).get("files", {}).get(asset_path)
            if manifest.get("archive_storage") == "oss_streaming_v1":
                from fastapi.responses import Response

                from .stream_recording import relative_name

                try:
                    relative_name(asset_path)
                    if not expected:
                        raise FileNotFoundError()
                    live = self.live.get(recording_id)
                    if live and not live.recording.ui_closed:
                        data = await live.recording.get_ui_asset(asset_path)
                    else:
                        data = await self.archives.read_object(
                            manifest["object_prefix"] + "ui_bundle/" + asset_path,
                            32 << 20,
                        )
                    if hashlib.sha256(data).hexdigest() != expected:
                        raise ValueError("UI asset changed")
                except (FileNotFoundError, ValueError):
                    raise HTTPException(404) from None
                except Exception:
                    raise HTTPException(503, "UI assets are unavailable") from None
                return Response(
                    data,
                    media_type="application/octet-stream",
                    headers={
                        "Cache-Control": "private, no-store",
                        "X-Content-SHA256": expected,
                        "X-Content-Type-Options": "nosniff",
                    },
                )
            path = directory / "ui_bundle" / asset_path

            def validate():
                if (
                    not expected
                    or Path(asset_path).is_absolute()
                    or ".." in Path(asset_path).parts
                    or path.is_symlink()
                    or directory.resolve() not in path.resolve().parents
                    or not path.is_file()
                ):
                    raise HTTPException(404)
                from .project_audio import AUDIO_PREFIX

                if asset_path.startswith(AUDIO_PREFIX):
                    if path.stat().st_size > 32 << 20:
                        raise HTTPException(404)
                    data = path.read_bytes()
                    if (
                        len(data) > 32 << 20
                        or hashlib.sha256(data).hexdigest() != expected
                    ):
                        raise HTTPException(404)
                    return data

            sealed = await self.run_io(validate, wait_for_slot=True)
            if sealed is not None:
                from fastapi.responses import Response

                return Response(
                    sealed,
                    media_type="application/octet-stream",
                    headers={
                        "Cache-Control": "private, no-store",
                        "X-Content-SHA256": expected,
                        "X-Content-Type-Options": "nosniff",
                    },
                )
            # Always use fetch with Bearer, then verify hash before executing UI.
            return FileResponse(
                path,
                headers={
                    "Cache-Control": "private, no-store",
                    "X-Content-SHA256": expected,
                    "X-Content-Type-Options": "nosniff",
                },
            )

        @app.get("/v1/whitebox/recordings/{recording_id}")
        async def recording_status(
            recording_id: str, authorization: str = Header(default="")
        ):
            await self.access(recording_id, authorization)
            manifest = await self.manifest(recording_id)
            report = self.live.get(recording_id)
            videos = []
            manifest_url = None
            uploading = self.spool / recording_id in self.archives.active
            if manifest.get("upload_status") == "complete" and not uploading:
                signer = getattr(self.archives.store, "playback_url", None)
                if signer:
                    manifest_url = signer(manifest["object_prefix"] + "manifest.json")
                    for name, video in manifest.get("videos", {}).items():
                        videos.append(
                            dict(
                                name=name,
                                frames=video["frames"],
                                codec=video["codec"],
                                encoder=copy.deepcopy(video.get("encoder", {})),
                                url=signer(manifest["object_prefix"] + video["path"]),
                                expires_at_ms=int(time.time() * 1000) + 900_000,
                            )
                        )
            client_ui = dict(manifest.get("client_ui", {}))
            intent_path = self.spool / ".reports" / recording_id / "client-upload.json"
            receipt_path = self.spool / recording_id / "ui-recording.json"
            intent = receipt = None
            # Active sessions have no client upload yet; polling their live status
            # uses only memory, even when the archive disk is unavailable.
            if (
                manifest.get("upload_status") == "complete"
                and not uploading
                and getattr(self.archives, "diskless", False)
            ):
                intent, receipt = await self.client_recordings.streaming.status(
                    recording_id, manifest
                )
            elif manifest.get("upload_status") == "complete" and not uploading:

                def receipts():
                    return tuple(
                        json.loads(p.read_text()) if p.is_file() else None
                        for p in (intent_path, receipt_path)
                    )

                intent, receipt = await self.run_io(receipts)
            if intent:
                client_ui.update(status=intent["status"], error=intent.get("error"))
            if receipt:
                client_ui.update(status=receipt["status"])
                signer = getattr(self.archives.store, "playback_url", None)
                if signer and receipt["status"] == "complete":
                    video = receipt["video"]
                    videos.append(
                        dict(
                            name="ui_rendered",
                            frames=video["frames"],
                            codec=video["codec"],
                            url=signer(manifest["object_prefix"] + video["path"]),
                            expires_at_ms=int(time.time() * 1000) + 900_000,
                        )
                    )
                    client_ui["receipt_url"] = signer(
                        manifest["object_prefix"] + "ui-recording.json"
                    )
            segments = []
            timings = report.snapshot() if report else manifest.get("trace", [])
            model_queue = None
            entries = (
                manifest.get("render_segments", [])
                if manifest.get("recording_kind") == "source"
                else []
            )
            for index, entry in list(enumerate(entries))[-64:]:
                child_id = entry.get("recording_id", "")
                try:
                    child = await self.owned_segment(manifest, child_id)
                except (HTTPException, FileNotFoundError):
                    segments.append(
                        dict(
                            segment_id=entry.get("segment_id"),
                            recording_status="unavailable",
                            upload_status="unavailable",
                        )
                    )
                    continue
                child_report = self.live.get(child_id)
                if index == len(entries) - 1:
                    child_timings = (
                        child_report.snapshot()
                        if child_report
                        else child.get("trace", [])
                    )
                    merged = {row["stage"]: row for row in timings}
                    merged.update({row["stage"]: row for row in child_timings})
                    timings = list(merged.values())
                    if child_report and hasattr(child_report.recording, "statistics"):
                        model_queue = child_report.recording.statistics()
                segments.append(
                    dict(
                        segment_id=child_id,
                        region_id=entry.get("region_id"),
                        recording_status=child["recording_status"],
                        upload_status=child["upload_status"],
                        archive_failure=child.get("archive_failure"),
                    )
                )
                signer = getattr(self.archives.store, "playback_url", None)
                if (
                    signer
                    and child.get("upload_status") == "complete"
                    and self.spool / child_id not in self.archives.active
                ):
                    for name, video in child.get("videos", {}).items():
                        videos.append(
                            dict(
                                name=f"segment-{index + 1}-{name}",
                                label=f"Model segment {index + 1} · {name}",
                                frames=video["frames"],
                                codec=video["codec"],
                                encoder=copy.deepcopy(video.get("encoder", {})),
                                url=signer(child["object_prefix"] + video["path"]),
                                inputs_path=f"segments/{child_id}/inputs",
                                expires_at_ms=int(time.time() * 1000) + 900_000,
                            )
                        )
            return dict(
                client_ui=client_ui,
                playback_delivery=(
                    copy.deepcopy(report.playback_delivery) if report else None
                ),
                recording_id=recording_id,
                trace_id=manifest["trace_id"],
                recording_status=manifest["recording_status"],
                archive_failure=manifest.get("archive_failure"),
                missing_frame_range=manifest.get("missing_frame_range"),
                last_successful_frame=manifest.get("last_successful_frame"),
                upload_failure=manifest.get("upload_failure"),
                upload_status="uploading" if uploading else manifest["upload_status"],
                dataset_status=manifest["dataset_status"],
                archive_encoding_version=manifest.get("archive_encoding_version", 1),
                timeline=manifest["timeline"],
                videos=videos,
                object_prefix=manifest["object_prefix"],
                archive_uri=(
                    "oss://"
                    + self.archives.store.bucket
                    + "/"
                    + manifest["object_prefix"]
                )
                if getattr(self.archives.store, "bucket", None)
                else None,
                manifest_url=manifest_url,
                multiplayer=manifest.get("multiplayer"),
                timings=timings,
                render_segments=segments,
                segments_truncated=len(entries) > 64,
                model_recording_queue=model_queue,
                recording_queue=(
                    report.recording.statistics()
                    if report and hasattr(report.recording, "statistics")
                    else None
                ),
                upload_queue=(
                    self.archives.upload_statistics()
                    if getattr(self.archives, "diskless", False)
                    else dict(pending=self.archives.upload_pending, workers=1)
                ),
                upload_statistics=manifest.get("upload_statistics"),
            )
