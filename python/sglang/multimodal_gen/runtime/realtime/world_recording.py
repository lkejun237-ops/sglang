# SPDX-License-Identifier: Apache-2.0
"""Fail-open, bounded recording of authorized world H.264 output."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

PART_SIZE = 8 * 1024 * 1024
QUEUE_MAX_BYTES = 8 * 1024 * 1024
S3_WORKER_THREADS = 8
SHUTDOWN_DRAIN_SECONDS = 10.0

RECORDING_SESSIONS_TOTAL = Counter(
    "world_recording_sessions_total",
    "Authorized world recording sessions by terminal result.",
    ["result"],
)
RECORDING_ACTIVE_SESSIONS = Gauge(
    "world_recording_active_sessions",
    "Authorized world recordings that have not reached a terminal state.",
)
RECORDING_BUFFER_BYTES = Gauge(
    "world_recording_buffer_bytes",
    "Media payload bytes waiting for a recording consumer.",
)
RECORDING_SESSION_PEAK_BUFFER_BYTES = Histogram(
    "world_recording_session_peak_buffer_bytes",
    "Peak queued media payload bytes for one recording.",
    buckets=(
        64 * 1024,
        256 * 1024,
        1024 * 1024,
        2 * 1024 * 1024,
        4 * 1024 * 1024,
        8 * 1024 * 1024,
    ),
)
RECORDING_UPLOAD_BYTES_TOTAL = Counter(
    "world_recording_upload_bytes_total",
    "Media bytes uploaded as successful S3 multipart parts.",
)
RECORDING_PART_UPLOAD_SECONDS = Histogram(
    "world_recording_part_upload_seconds",
    "S3 UploadPart wall time.",
)
RECORDING_FINALIZE_SECONDS = Histogram(
    "world_recording_finalize_seconds",
    "Final part upload and CompleteMultipartUpload wall time.",
)
RECORDING_CALLBACK_FAILURES_TOTAL = Counter(
    "world_recording_callback_failures_total",
    "Recorded callbacks that exhausted their retry budget.",
)


class RunVideoRecorder:
    """One Run's multipart upload.

    ``offer`` is intentionally synchronous and never waits for S3. All methods
    are called on the Gateway event loop; blocking SDK calls run in the
    manager's dedicated executor.
    """

    def __init__(
        self,
        *,
        run_id: str,
        session_id: str,
        generation_id: str,
        trace_id: str,
        bucket: str,
        s3_client: Any,
        executor: ThreadPoolExecutor,
        on_recorded: Callable[[str, str], Any],
    ) -> None:
        self.run_id = run_id
        self.session_id = session_id
        self.generation_id = generation_id
        self.trace_id = trace_id
        self.bucket = bucket
        self.key = f"runs/{run_id}/video.mp4"
        self.video_uri = f"s3://{bucket}/{self.key}"
        self._s3 = s3_client
        self._executor = executor
        self._on_recorded = on_recorded

        self.state = "waiting_init"
        self.result: str | None = None
        self.failure_reason = ""
        self.final_seen = False
        self.payload_count = 0
        self.queued_bytes = 0
        self.peak_queued_bytes = 0
        self._next_sequence = 0
        self._queue: deque[bytes] = deque()
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(
            self._run(), name=f"world-recording-{run_id[:8]}"
        )

    @property
    def task(self) -> asyncio.Task:
        return self._task

    def offer(self, message: dict[str, Any]) -> None:
        """Accept one already-validated Gateway output message without waiting."""
        if self.state in {"failed", "ready"}:
            return
        try:
            message_type = message.get("type")
            if message_type == "media_init":
                self._offer_init(message)
            elif message_type == "media_payload":
                self._offer_payload(message)
            elif (
                message_type == "media_chunk_complete"
                and message.get("is_final_chunk") is True
            ):
                self._offer_final()
        except Exception:  # Recording must never escape into the media route.
            logger.exception(
                "World recording hook failed run_id=%s session_id=%s "
                "trace_id=%s bucket=%s key=%s",
                self.run_id,
                self.session_id,
                self.trace_id,
                self.bucket,
                self.key,
            )
            self.fail("upload_failed")

    def _offer_init(self, message: dict[str, Any]) -> None:
        if self.state != "waiting_init":
            self.fail("incomplete")
            return
        if message.get("codec") != "h264" or message.get("container") != "fmp4":
            self.fail("incomplete")
            return
        self.state = "recording"
        self._wake.set()

    def _offer_payload(self, message: dict[str, Any]) -> None:
        if self.state != "recording":
            self.fail("incomplete")
            return
        sequence = message.get("sequence")
        payload = message.get("payload")
        if type(sequence) is not int or sequence != self._next_sequence:
            self.fail("incomplete")
            return
        if not isinstance(payload, bytes) or not payload:
            self.fail("incomplete")
            return
        if self.queued_bytes + len(payload) > QUEUE_MAX_BYTES:
            self.fail("buffer_full")
            return
        # Keep the already-decoded bytes reference. Do not copy the msgpack wire
        # or coalesce the complete video in memory.
        self._queue.append(payload)
        self.queued_bytes += len(payload)
        self.peak_queued_bytes = max(self.peak_queued_bytes, self.queued_bytes)
        RECORDING_BUFFER_BYTES.inc(len(payload))
        self.payload_count += 1
        self._next_sequence += 1
        self._wake.set()

    def _offer_final(self) -> None:
        if self.state != "recording" or self.payload_count == 0:
            self.fail("incomplete")
            return
        # Seal input synchronously with acceptance of the final marker. Session
        # cleanup may now detach, but no later payload can enter this upload.
        self.final_seen = True
        self.state = "completing"
        self._wake.set()

    def session_closed(self) -> None:
        """Abort sessions that closed without the post-EOF final marker."""
        if not self.final_seen:
            self.fail("incomplete")

    def fail(self, reason: str) -> None:
        if self.state in {"failed", "ready"}:
            return
        self.state = "failed"
        self.failure_reason = reason
        self._clear_queue()
        self._wake.set()
        if reason == "buffer_full":
            logger.warning(
                "World recording buffer is full run_id=%s session_id=%s "
                "trace_id=%s bucket=%s key=%s",
                self.run_id,
                self.session_id,
                self.trace_id,
                self.bucket,
                self.key,
            )

    async def wait(self) -> str:
        await asyncio.shield(self._task)
        return self.result or self.failure_reason or "upload_failed"

    def _pop_payload(self) -> bytes | None:
        if not self._queue:
            return None
        payload = self._queue.popleft()
        self.queued_bytes -= len(payload)
        RECORDING_BUFFER_BYTES.dec(len(payload))
        return payload

    def _clear_queue(self) -> None:
        if self.queued_bytes:
            RECORDING_BUFFER_BYTES.dec(self.queued_bytes)
        self._queue.clear()
        self.queued_bytes = 0

    async def _wait_for_work(self) -> None:
        self._wake.clear()
        if self._queue or self.state in {"failed", "completing", "ready"}:
            return
        await self._wake.wait()

    async def _call_s3(self, method: Callable[..., Any], **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(method, **kwargs))

    async def _upload_part(
        self, upload_id: str, part_number: int, body: bytes
    ) -> dict[str, Any]:
        started = time.perf_counter()
        response = await self._call_s3(
            self._s3.upload_part,
            Bucket=self.bucket,
            Key=self.key,
            UploadId=upload_id,
            PartNumber=part_number,
            Body=body,
        )
        RECORDING_PART_UPLOAD_SECONDS.observe(time.perf_counter() - started)
        etag = response.get("ETag") if isinstance(response, dict) else None
        if not etag:
            raise RuntimeError("S3 UploadPart response is missing ETag")
        RECORDING_UPLOAD_BYTES_TOTAL.inc(len(body))
        return {"ETag": etag, "PartNumber": part_number}

    async def _run(self) -> None:
        upload_id = ""
        parts: list[dict[str, Any]] = []
        part = bytearray()
        try:
            while self.state == "waiting_init":
                await self._wait_for_work()
            if self.state == "failed":
                return

            created = await self._call_s3(
                self._s3.create_multipart_upload,
                Bucket=self.bucket,
                Key=self.key,
                ContentType="video/mp4",
                CacheControl="public, max-age=31536000, immutable",
            )
            upload_id = str(created.get("UploadId") or "")
            if not upload_id:
                raise RuntimeError(
                    "S3 CreateMultipartUpload response is missing UploadId"
                )

            while True:
                payload = self._pop_payload()
                if payload is not None:
                    offset = 0
                    view = memoryview(payload)
                    while offset < len(payload):
                        take = min(PART_SIZE - len(part), len(payload) - offset)
                        part.extend(view[offset : offset + take])
                        offset += take
                        if len(part) == PART_SIZE:
                            body = bytes(part)
                            part.clear()
                            parts.append(
                                await self._upload_part(upload_id, len(parts) + 1, body)
                            )
                    continue

                if self.state == "failed":
                    return
                if self.state == "completing":
                    finalize_started = time.perf_counter()
                    if part:
                        body = bytes(part)
                        part.clear()
                        parts.append(
                            await self._upload_part(upload_id, len(parts) + 1, body)
                        )
                    if not parts:
                        self.fail("incomplete")
                        return
                    await self._call_s3(
                        self._s3.complete_multipart_upload,
                        Bucket=self.bucket,
                        Key=self.key,
                        UploadId=upload_id,
                        MultipartUpload={"Parts": parts},
                    )
                    RECORDING_FINALIZE_SECONDS.observe(
                        time.perf_counter() - finalize_started
                    )
                    self.state = "ready"
                    self.result = "ready"
                    self._fire_recorded_callback()
                    return
                await self._wait_for_work()
        except Exception:
            if self.state != "failed":
                self.fail("upload_failed")
            logger.exception(
                "World recording upload failed run_id=%s session_id=%s "
                "trace_id=%s bucket=%s key=%s",
                self.run_id,
                self.session_id,
                self.trace_id,
                self.bucket,
                self.key,
            )
        finally:
            if self.state != "ready" and upload_id:
                try:
                    await self._call_s3(
                        self._s3.abort_multipart_upload,
                        Bucket=self.bucket,
                        Key=self.key,
                        UploadId=upload_id,
                    )
                except Exception:
                    logger.exception(
                        "World recording multipart abort failed run_id=%s "
                        "session_id=%s trace_id=%s bucket=%s key=%s",
                        self.run_id,
                        self.session_id,
                        self.trace_id,
                        self.bucket,
                        self.key,
                    )
            self._clear_queue()
            if self.result is None:
                self.result = self.failure_reason or "upload_failed"

    def _fire_recorded_callback(self) -> None:
        try:
            pending = self._on_recorded(self.run_id, self.video_uri)
            if inspect.isawaitable(pending) and not isinstance(pending, asyncio.Task):
                pending = asyncio.create_task(pending)
            if isinstance(pending, asyncio.Task):
                pending.add_done_callback(self._record_callback_result)
        except Exception:
            RECORDING_CALLBACK_FAILURES_TOTAL.inc()
            logger.exception(
                "World recorded callback could not be scheduled run_id=%s "
                "session_id=%s trace_id=%s video_uri=%s",
                self.run_id,
                self.session_id,
                self.trace_id,
                self.video_uri,
            )

    def _record_callback_result(self, task: asyncio.Task) -> None:
        try:
            succeeded = task.result()
        except asyncio.CancelledError:
            succeeded = False
        except Exception:
            succeeded = False
        if succeeded is False:
            RECORDING_CALLBACK_FAILURES_TOTAL.inc()
            logger.error(
                "World recorded callback exhausted retries run_id=%s "
                "session_id=%s trace_id=%s video_uri=%s",
                self.run_id,
                self.session_id,
                self.trace_id,
                self.video_uri,
            )


class RecordingManager:
    """Process-level owner for active and detached recording tasks."""

    def __init__(
        self,
        *,
        bucket: str,
        s3_client: Any,
        on_recorded: Callable[[str, str], Any],
    ) -> None:
        self.bucket = bucket
        self._s3 = s3_client
        self._on_recorded = on_recorded
        self._executor = ThreadPoolExecutor(
            max_workers=S3_WORKER_THREADS, thread_name_prefix="world-recording-s3"
        )
        self._recorders: dict[tuple[str, str], RunVideoRecorder] = {}
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    def start(
        self,
        *,
        run_id: str,
        session_id: str,
        generation_id: str,
        trace_id: str,
    ) -> RunVideoRecorder:
        if self._closed:
            raise RuntimeError("recording manager is closed")
        key = (session_id, generation_id)
        current = self._recorders.get(key)
        if current is not None:
            current.fail("incomplete")
        recorder = RunVideoRecorder(
            run_id=run_id,
            session_id=session_id,
            generation_id=generation_id,
            trace_id=trace_id,
            bucket=self.bucket,
            s3_client=self._s3,
            executor=self._executor,
            on_recorded=self._on_recorded,
        )
        self._recorders[key] = recorder
        self._tasks.add(recorder.task)
        RECORDING_ACTIVE_SESSIONS.inc()

        def done(task: asyncio.Task) -> None:
            self._tasks.discard(task)
            if self._recorders.get(key) is recorder:
                self._recorders.pop(key, None)
            RECORDING_ACTIVE_SESSIONS.dec()
            RECORDING_SESSION_PEAK_BUFFER_BYTES.observe(recorder.peak_queued_bytes)
            result = recorder.result or recorder.failure_reason or "upload_failed"
            RECORDING_SESSIONS_TOTAL.labels(result=result).inc()
            if not task.cancelled():
                task.exception()  # consume unexpected task exceptions defensively

        recorder.task.add_done_callback(done)
        return recorder

    def session_closed(self, session_id: str, generation_id: str) -> None:
        recorder = self._recorders.get((session_id, generation_id))
        if recorder is not None:
            recorder.session_closed()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for recorder in tuple(self._recorders.values()):
            recorder.session_closed()
        if self._tasks:
            _, pending = await asyncio.wait(
                tuple(self._tasks), timeout=SHUTDOWN_DRAIN_SECONDS
            )
            if pending:
                logger.error(
                    "World recording shutdown drain timed out with %d task(s)",
                    len(pending),
                )
        self._executor.shutdown(wait=False, cancel_futures=True)


def create_s3_recording_manager(
    *,
    bucket: str,
    region: str,
    on_recorded: Callable[[str, str], Any],
) -> RecordingManager:
    """Build the production S3 client with bounded SDK retries and timeouts."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError("boto3 is required when world recording is enabled") from exc

    client = boto3.client(
        "s3",
        region_name=region,
        config=Config(
            connect_timeout=3,
            read_timeout=10,
            retries={"mode": "standard", "total_max_attempts": 2},
        ),
    )
    return RecordingManager(bucket=bucket, s3_client=client, on_recorded=on_recorded)
