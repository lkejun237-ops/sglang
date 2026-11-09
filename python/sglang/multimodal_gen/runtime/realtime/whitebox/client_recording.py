# SPDX-License-Identifier: Apache-2.0
"""Capability-scoped client recording handoff; no browser rendering in this service.

The untrusted PUT object is never used for playback. An isolated uploader validates
the downloaded bytes against the model timeline and publishes a separate receipt.
The two-video manifest remains immutable after its normal archive upload.
"""

import asyncio
import base64
import hashlib
import json
import re
import secrets
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import av
from fastapi import Header, HTTPException
from pydantic import BaseModel, Field

MAX_BYTES = 256 * 1024 * 1024


class UploadIntent(BaseModel):
    bytes: int = Field(gt=0, le=MAX_BYTES, strict=True)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_md5: str


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.chmod(0o600)
    temporary.replace(path)


def validate_video(path, model_path, intent):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if path.stat().st_size != intent["bytes"] or digest.hexdigest() != intent["sha256"]:
        raise ValueError("Client video checksum mismatch")
    return validate_video_streams(path, model_path)


def validate_video_streams(candidate, reference):
    """Accept paths for retained archives and seekable RAM buffers for new ones."""
    with (
        av.open(candidate if hasattr(candidate, "read") else str(candidate)) as video,
        av.open(reference if hasattr(reference, "read") else str(reference)) as model,
    ):
        if len(video.streams) != 1 or len(video.streams.video) != 1:
            raise ValueError("Exactly one video stream is required")
        actual, expected = video.streams.video[0], model.streams.video[0]
        if actual.codec_context.name != "h264" or (actual.width, actual.height) != (
            expected.width,
            expected.height,
        ):
            raise ValueError("Client video format does not match the model")
        actual.thread_type = "NONE"
        actual.codec_context.thread_count = 1
        expected.codec_context.thread_count = 1
        wanted = iter(model.decode(expected))
        frames = 0
        for frame in video.decode(actual):
            reference = next(wanted, None)
            if (
                reference is None
                or frame.pts is None
                or reference.pts is None
                or abs(
                    float(
                        frame.pts * frame.time_base
                        - reference.pts * reference.time_base
                    )
                )
                > 0.000001
            ):
                raise ValueError(
                    "Client video must preserve every model frame timestamp"
                )
            frames += 1
        if not frames or next(wanted, None) is not None:
            raise ValueError("Client video frame count does not match the model")
        return {
            "frames": frames,
            "codec": "h264",
            "width": actual.width,
            "height": actual.height,
        }


def verify_and_publish(directory, intent_path, store_config):
    """Executed in the existing resource-limited uploader process."""
    from .recording import S3ArchiveStore

    directory, intent_path = Path(directory), Path(intent_path)
    intent = json.loads(intent_path.read_text())
    store = S3ArchiveStore(**store_config)
    candidate = intent_path.with_suffix(".mp4")
    try:
        response = store.client.get_object(
            Bucket=store.bucket, Key=intent["staging_key"]
        )
        body = response["Body"]
        try:
            if response["ContentLength"] != intent["bytes"]:
                raise ValueError("Client video size mismatch")
            total = 0
            with candidate.open("wb") as target:
                for chunk in iter(lambda: body.read(1024 * 1024), b""):
                    total += len(chunk)
                    if total > intent["bytes"]:
                        raise ValueError("Client video exceeds its declared size")
                    target.write(chunk)
        finally:
            body.close()
        video = validate_video(candidate, directory / "model.mp4", intent)
        video.update(
            path=intent["published_path"],
            sha256=intent["sha256"],
            bytes=intent["bytes"],
        )
        manifest = json.loads((directory / "manifest.json").read_text())
        prefix = manifest["object_prefix"]
        asyncio.run(store.upload(candidate, prefix + video["path"]))
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "recording_id": directory.name,
            "upload_id": intent["upload_id"],
            "timeline": "source_capture_vfr",
            "video": video,
        }
        receipt_path = intent_path.with_suffix(".receipt.json")
        save_json(receipt_path, receipt)
        asyncio.run(store.upload(receipt_path, prefix + "ui-recording.json"))
        save_json(directory / "ui-recording.json", receipt)
        intent.update(status="complete", error=None)
    except Exception:
        # Keep failed data and private evidence. Never expose SDK errors or signed URLs.
        intent.update(
            status="failed", error="The UI recording could not be verified or uploaded."
        )
    finally:
        save_json(intent_path, intent)
    return intent["status"]


class ClientRecordings:
    def __init__(self, reports):
        self.reports = reports
        self.jobs = {}
        from .stream_client_recording import StreamingClientRecordings

        self.streaming = StreamingClientRecordings(reports)

    def install(self, app):
        base = "/v1/whitebox/recordings/{recording_id}/ui-recording"

        def prepare_intent(
            recording_id: str,
            body: UploadIntent,
            authorization: str = Header(default=""),
        ):
            directory = self.reports.authorize(recording_id, authorization)
            manifest = json.loads((directory / "manifest.json").read_text())
            archives = self.reports.archives
            if (
                manifest.get("upload_status") != "complete"
                or directory in archives.active
                or manifest.get("ui", {}).get("mode") != "browser"
            ):
                raise HTTPException(
                    409, "Wait for the model recording to finish uploading"
                )
            if not hasattr(archives.store, "playback_client"):
                raise HTTPException(503, "Client recording uploads are unavailable")
            try:
                if len(base64.b64decode(body.content_md5, validate=True)) != 16:
                    raise ValueError()
            except ValueError:
                raise HTTPException(422, "A base64 Content-MD5 is required") from None
            root = self.reports.spool / ".reports" / recording_id
            root.mkdir(mode=0o700, exist_ok=True)
            path = root / "client-upload.json"
            previous = json.loads(path.read_text()) if path.exists() else None
            if previous:
                if any(previous[k] != v for k, v in body.model_dump().items()):
                    raise HTTPException(409, "An upload already owns this recording")
                if previous["status"] in {"verifying", "complete"}:
                    return {
                        "upload_id": previous["upload_id"],
                        "status": previous["status"],
                    }
            else:
                upload_id = secrets.token_hex(16)
                previous = {
                    **body.model_dump(),
                    "upload_id": upload_id,
                    "status": "pending",
                    "published_path": f"client-ui/{upload_id}/ui_rendered.mp4",
                    "staging_key": manifest["object_prefix"]
                    + f"client-ui/{upload_id}/incoming.mp4",
                }
            previous["expires_at"] = int(time.time()) + 900
            save_json(path, previous)
            headers = {"Content-Type": "video/mp4", "Content-MD5": body.content_md5}
            url = archives.store.playback_client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": archives.store.bucket,
                    "Key": previous["staging_key"],
                    "ContentType": "video/mp4",
                    "ContentMD5": body.content_md5,
                },
                ExpiresIn=900,
                HttpMethod="PUT",
            )
            return {
                "upload_id": previous["upload_id"],
                "status": "pending",
                "method": "PUT",
                "url": url,
                "headers": headers,
                "expires_at": previous["expires_at"],
                "max_bytes": MAX_BYTES,
            }

        @app.post(base + "/intent")
        async def intent(
            recording_id: str,
            body: UploadIntent,
            authorization: str = Header(default=""),
        ):
            await self.reports.access(recording_id, authorization)
            manifest = await self.reports.manifest(recording_id)
            if getattr(self.reports.archives, "diskless", False):
                return await self.streaming.prepare(recording_id, body, manifest)
            return await self.reports.run_io(
                prepare_intent, recording_id, body, authorization
            )

        @app.post(base + "/{upload_id}/complete")
        async def complete(
            recording_id: str, upload_id: str, authorization: str = Header(default="")
        ):
            directory = await self.reports.access(recording_id, authorization)
            manifest = await self.reports.manifest(recording_id)
            if getattr(self.reports.archives, "diskless", False):
                return await self.streaming.complete(recording_id, upload_id, manifest)
            path = self.reports.spool / ".reports" / recording_id / "client-upload.json"

            def read_intent():
                if not re.fullmatch(r"[a-f0-9]{32}", upload_id) or not path.is_file():
                    raise HTTPException(404)
                return json.loads(path.read_text())

            value = await self.reports.run_io(read_intent)
            if value["upload_id"] != upload_id:
                raise HTTPException(404)
            if value["status"] == "complete" or recording_id in self.jobs:
                return {"status": value["status"], "upload_id": upload_id}
            archives = self.reports.archives
            if len(self.jobs) >= archives.max_pending:
                raise HTTPException(
                    429, "The upload verifier is busy. Please retry shortly."
                )
            value["status"] = "verifying"
            await self.reports.run_io(save_json, path, value)

            async def verify():
                if hasattr(archives, "verify_client"):
                    try:
                        await archives.verify_client(directory, path)
                    except Exception:
                        value.update(
                            status="failed",
                            error="The upload worker is unavailable. Please retry.",
                        )
                        await self.reports.run_io(save_json, path, value)
                    finally:
                        self.jobs.pop(recording_id, None)
                    return

                from .recording_process import process_pool

                archives.upload_pending += 1
                try:
                    async with archives.upload_gate:
                        if archives.upload_pool is None:
                            archives.upload_pool = process_pool(archives.upload_cpu)
                        try:
                            await asyncio.get_running_loop().run_in_executor(
                                archives.upload_pool,
                                verify_and_publish,
                                directory,
                                path,
                                archives.store.worker_config,
                            )
                        except BrokenProcessPool:
                            await asyncio.to_thread(
                                archives.upload_pool.shutdown, wait=True
                            )
                            archives.upload_pool = None
                            raise
                except Exception:
                    value.update(
                        status="failed",
                        error="The upload worker is unavailable. Please retry.",
                    )
                    save_json(path, value)
                finally:
                    archives.upload_pending -= 1
                    self.jobs.pop(recording_id, None)

            task = asyncio.create_task(verify(), name="client-ui-" + recording_id)
            self.jobs[recording_id] = task
            archives.tasks.add(task)
            task.add_done_callback(archives.tasks.discard)
            return {"status": "verifying", "upload_id": upload_id}
