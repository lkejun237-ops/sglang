# SPDX-License-Identifier: Apache-2.0
"""Optional browser-composed recording verification without local MP4 copies."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import multiprocessing
import queue
import re
import secrets
import time

from fastapi import HTTPException

from .stream_recording import put_json
from .stream_upload import retry_same_bytes, s3_client

MAX_BYTES = 256 << 20
MAX_MODEL_BYTES = 512 << 20


def verify_streaming_client(config, manifest, intent, intent_key):
    from .client_recording import validate_video_streams

    client = s3_client(config)
    bucket = config["bucket"]

    def read(key, limit, expected=None):
        result = client.get_object(Bucket=bucket, Key=key)
        body = result["Body"]
        try:
            size = result.get("ContentLength", limit + 1)
            if size > limit or (expected is not None and size != expected):
                raise ValueError("Client recording size mismatch")
            data = body.read(limit + 1)
            if len(data) != size:
                raise ValueError("Client recording was truncated")
            return data
        finally:
            body.close()

    value = dict(intent)
    try:
        candidate = read(intent["staging_key"], MAX_BYTES, intent["bytes"])
        if hashlib.sha256(candidate).hexdigest() != intent["sha256"]:
            raise ValueError("Client recording checksum mismatch")
        expected = manifest["objects"]["model.mp4"]
        model = read(
            manifest["object_prefix"] + "model.mp4", MAX_MODEL_BYTES, expected["bytes"]
        )
        if hashlib.sha256(model).hexdigest() != expected["sha256"]:
            raise ValueError("Model recording changed")
        video = validate_video_streams(io.BytesIO(candidate), io.BytesIO(model))
        video.update(
            path=intent["published_path"], sha256=intent["sha256"], bytes=len(candidate)
        )
        # Upload exactly the validated bytes. Do not copy a staging object that
        # the capability holder could replace between validation and publication.
        retry_same_bytes(
            lambda: client.put_object(
                Bucket=bucket,
                Key=manifest["object_prefix"] + video["path"],
                Body=candidate,
                ContentType="video/mp4",
                ContentMD5=base64.b64encode(hashlib.md5(candidate).digest()).decode(),
                Metadata={"sha256": intent["sha256"]},
            )
        )
        head = client.head_object(
            Bucket=bucket, Key=manifest["object_prefix"] + video["path"]
        )
        if (
            head.get("ContentLength") != len(candidate)
            or head.get("Metadata", {}).get("sha256") != intent["sha256"]
        ):
            raise ValueError("Published client recording was not confirmed")
        receipt = dict(
            schema_version=1,
            status="complete",
            recording_id=manifest["recording_id"],
            upload_id=intent["upload_id"],
            timeline="source_capture_vfr",
            video=video,
        )
        put_json(
            client, bucket, manifest["object_prefix"] + "ui-recording.json", receipt
        )
        value.update(status="complete", error=None, receipt=receipt)
    except Exception:
        value.update(
            status="failed", error="The UI recording could not be verified or uploaded."
        )
    put_json(client, bucket, intent_key, value)
    return value


def client_worker(replies, config, manifest, intent, intent_key, cpu):
    from .async_recording import limit_child

    limit_child(cpu, 2 << 30, 0)
    try:
        replies.put(
            (True, verify_streaming_client(config, manifest, intent, intent_key))
        )
    except BaseException:
        replies.put((False, None))


class StreamingClientRecordings:
    def __init__(self, reports):
        self.reports = reports
        self.cache, self.jobs = {}, {}
        self.lock = asyncio.Lock()

    def key(self, recording_id):
        # index_key performs the strict UUID check.
        return (
            self.reports.archives.index_key(recording_id).removesuffix(".json")
            + ".client.json"
        )

    async def read(self, recording_id):
        cached = self.cache.get(recording_id)
        if cached and time.monotonic() - cached[0] < 5:
            return cached[1]
        try:
            value = json.loads(
                await self.reports.archives.read_object(self.key(recording_id), 16384)
            )
        except FileNotFoundError:
            # Preserve ownership of optional UI uploads initiated before this
            # migration, without creating or rewriting any retained local file.
            def retained():
                path = (
                    self.reports.spool
                    / ".reports"
                    / recording_id
                    / "client-upload.json"
                )
                receipt = self.reports.spool / recording_id / "ui-recording.json"
                value = json.loads(path.read_text()) if path.is_file() else None
                if value and receipt.is_file():
                    value["receipt"] = json.loads(receipt.read_text())
                return value

            value = await self.reports.run_io(retained)
        self.remember(recording_id, value)
        return value

    def remember(self, recording_id, value):
        self.cache[recording_id] = (time.monotonic(), value)
        while len(self.cache) > 128:
            self.cache.pop(next(iter(self.cache)))

    async def save(self, recording_id, value):
        archives = self.reports.archives
        await archives.io(
            put_json,
            archives.store.client,
            archives.store.bucket,
            self.key(recording_id),
            value,
        )
        self.remember(recording_id, value)

    async def prepare(self, recording_id, body, manifest):
        archives = self.reports.archives
        if (
            manifest.get("upload_status") != "complete"
            or self.reports.spool / recording_id in archives.active
            or manifest.get("ui", {}).get("mode") != "browser"
            or "model" not in manifest.get("videos", {})
        ):
            raise HTTPException(409, "Wait for the model recording to finish uploading")
        if not hasattr(archives.store, "playback_client"):
            raise HTTPException(503, "Client recording uploads are unavailable")
        try:
            if len(base64.b64decode(body.content_md5, validate=True)) != 16:
                raise ValueError()
        except ValueError:
            raise HTTPException(422, "A base64 Content-MD5 is required") from None
        async with self.lock:
            try:
                previous = await self.read(recording_id)
                if previous:
                    self.validate_intent(previous, manifest)
                    if any(
                        previous[k] != value for k, value in body.model_dump().items()
                    ):
                        raise HTTPException(
                            409, "An upload already owns this recording"
                        )
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
                await self.save(recording_id, previous)
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(
                    503, "Client recording storage is unavailable"
                ) from None
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
        return dict(
            upload_id=previous["upload_id"],
            status="pending",
            method="PUT",
            url=url,
            headers={"Content-Type": "video/mp4", "Content-MD5": body.content_md5},
            expires_at=previous["expires_at"],
            max_bytes=MAX_BYTES,
        )

    @staticmethod
    def validate_intent(value, manifest):
        upload_id = value.get("upload_id", "")
        if (
            not isinstance(upload_id, str)
            or not re.fullmatch(r"[a-f0-9]{32}", upload_id)
            or value.get("published_path") != f"client-ui/{upload_id}/ui_rendered.mp4"
            or value.get("staging_key")
            != manifest["object_prefix"] + f"client-ui/{upload_id}/incoming.mp4"
        ):
            raise ValueError("Invalid client recording identity")
        receipt = value.get("receipt")
        if receipt and (
            receipt.get("recording_id") != manifest["recording_id"]
            or receipt.get("upload_id") != upload_id
            or receipt.get("video", {}).get("path") != value["published_path"]
        ):
            raise ValueError("Invalid client recording receipt")

    async def complete(self, recording_id, upload_id, manifest):
        # Reserve the verifier under the same lock as intent creation. Two
        # simultaneous completion requests must not spawn duplicate workers.
        async with self.lock:
            return await self._complete(recording_id, upload_id, manifest)

    async def _complete(self, recording_id, upload_id, manifest):
        if not re.fullmatch(r"[a-f0-9]{32}", upload_id):
            raise HTTPException(404)
        try:
            value = await self.read(recording_id)
        except Exception:
            raise HTTPException(
                503, "Client recording storage is unavailable"
            ) from None
        if not value or value["upload_id"] != upload_id:
            raise HTTPException(404)
        try:
            self.validate_intent(value, manifest)
        except ValueError:
            raise HTTPException(404) from None
        if value["status"] == "complete" or recording_id in self.jobs:
            return {"status": value["status"], "upload_id": upload_id}
        archives = self.reports.archives
        if len(self.jobs) >= archives.max_pending:
            raise HTTPException(
                429, "The upload verifier is busy. Please retry shortly."
            )
        value = dict(value, status="verifying")
        try:
            await self.save(recording_id, value)
        except Exception:
            raise HTTPException(
                503, "Client recording storage is unavailable"
            ) from None

        async def verify():
            process = replies = None
            archives.upload_pending += 1
            try:
                async with asyncio.timeout(120):
                    async with archives.upload_gate:
                        context = multiprocessing.get_context("spawn")
                        replies = context.Queue(1)
                        process = context.Process(
                            target=client_worker,
                            args=(
                                replies,
                                archives.store_config,
                                manifest,
                                value,
                                self.key(recording_id),
                                archives.upload_cpu,
                            ),
                            daemon=True,
                        )
                        starting = asyncio.create_task(asyncio.to_thread(process.start))
                        try:
                            await asyncio.shield(starting)
                        except asyncio.CancelledError:
                            await asyncio.gather(starting, return_exceptions=True)
                            raise
                        while True:
                            try:
                                ok, result = replies.get_nowait()
                                if not ok:
                                    raise RuntimeError("Client verifier failed")
                                self.remember(recording_id, result)
                                return
                            except queue.Empty:
                                if not process.is_alive():
                                    raise RuntimeError("Client verifier exited")
                                await asyncio.sleep(0.05)
            except (Exception, asyncio.CancelledError):
                failed = dict(
                    value,
                    status="failed",
                    error="The upload worker is unavailable. Please retry.",
                )
                self.remember(recording_id, failed)
                try:
                    await self.save(recording_id, failed)
                except Exception:
                    pass
            finally:
                if process and process.pid:
                    if process.is_alive():
                        process.terminate()
                    await asyncio.to_thread(process.join, 1)
                    if process.is_alive():
                        process.kill()
                        await asyncio.to_thread(process.join, 1)
                if replies:
                    replies.cancel_join_thread()
                    replies.close()
                archives.upload_pending -= 1
                self.jobs.pop(recording_id, None)

        task = self.jobs[recording_id] = asyncio.create_task(verify())
        archives.tasks.add(task)
        task.add_done_callback(archives.tasks.discard)
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        return {"status": "verifying", "upload_id": upload_id}

    async def status(self, recording_id, manifest):
        if not manifest.get("client_ui"):
            return None, None
        try:
            value = await self.read(recording_id)
            if value:
                self.validate_intent(value, manifest)
            return value, value.get("receipt") if value else None
        except Exception:
            return None, None
