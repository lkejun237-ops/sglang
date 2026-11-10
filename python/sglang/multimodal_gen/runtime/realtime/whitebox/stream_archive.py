# SPDX-License-Identifier: Apache-2.0
"""Best-effort OSS archives. Source admission and frame offers never await IO."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import multiprocessing
import os
import re
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from .archive_rgb import RGBBudget, RGBSlots
from .async_recording import AsyncRecording
from .contracts import capture_pts, presentation_pts
from .metrics import observe_duration
from .recording import ArchiveManager
from .stream_recording import (
    STORAGE,
    put_bytes,
    put_json,
    relative_name,
    streaming_worker,
)


class StreamingRecording(AsyncRecording):
    diskless = True

    def __init__(self, manager, kwargs):
        self.raw_archive = bool(kwargs.get("archive_encoding"))
        self.rgb_slots = {}
        self.reference_waiters, self.reference_confirmed = set(), set()
        self.worker_cpu = None
        self.token_hash = None
        self.ui_engine = None
        self.ui_assets, self.ui_tasks = {}, {}
        self.ui_bytes = 0
        self.ui_closed = False
        self.ui_gate = asyncio.Semaphore(2)
        self.upload_buffer_bytes = self.object_buffer_bytes = self.produced_bytes = 0
        self.confirmed_upload_bytes = 0
        super().__init__(manager, kwargs)
        self.manifest["archive_storage"] = STORAGE
        self.manifest["client_ui"]["status"] = "awaiting_upload"
        if self.raw_archive:
            self.manifest["archive_encoding_version"] = 2
            self.manifest["archive_encoding_settings"] = copy.deepcopy(
                kwargs["archive_encoding"]
            )

    def _raw_offer(self, lane, kind, rgb, build):
        if not self.accepting:
            self.rejected += 1
            return False
        oldest = self.pending or (self.items[0] if self.items else None)
        if oldest and time.monotonic() - oldest[0] > 1.0:
            self.degrade("ARCHIVE_RGB_QUEUE_EXPIRED")
            return False
        try:
            if lane not in self.rgb_slots:
                self.rgb_slots[lane] = RGBSlots(
                    (self.model_spec if lane == "model" else self.spec).frame_bytes,
                    self.manager.rgb_budget,
                )
            pool = self.rgb_slots[lane]
            descriptor = pool.offer(rgb)
        except (BufferError, OSError, ValueError) as error:
            logging.getLogger(__name__).warning(
                "Archive RGB admission failed trace=%s lane=%s frame=%s: %s",
                self.trace_id,
                lane,
                self.last_offered_frame,
                error,
            )
            self.degrade("ARCHIVE_RGB_CAPACITY_EXHAUSTED")
            return False
        if not self.offer(kind, build(descriptor)):
            pool.release(descriptor)
            return False
        return True

    async def prepare_raw(self):
        if not self.raw_archive:
            return
        from .diagnostics import WhiteboxDiagnostic

        if self.closed or self.failure:
            raise WhiteboxDiagnostic("ARCHIVE_CAPACITY_UNAVAILABLE", "startup")
        try:
            names = (
                ("whitebox",)
                if self.kwargs.get("source_only")
                else ("whitebox", "model")
            )
            for name in names:
                if name not in self.rgb_slots:
                    self.rgb_slots[name] = RGBSlots(
                        (self.model_spec if name == "model" else self.spec).frame_bytes,
                        self.manager.rgb_budget,
                    )
            await asyncio.wait_for(
                self.startup_complete.wait(), self.manager.boot_timeout + 1
            )
            if self.failure or not self.worker_ready:
                raise RuntimeError("Archive worker is unavailable")
        except asyncio.CancelledError:
            self.degrade("ARCHIVE_CANCELLED")
            raise
        except Exception as error:
            self.degrade("ARCHIVE_CAPACITY_UNAVAILABLE")
            raise WhiteboxDiagnostic(
                "ARCHIVE_CAPACITY_UNAVAILABLE", "startup"
            ) from error

    async def add_references(self, references, *, wait=False):
        if not (self.raw_archive and wait):
            return await super().add_references(references, wait=wait)
        from .recording import reference_descriptors

        # Confirm one image at a time. A legal 64 MiB collection must not occupy
        # the entire video metadata queue or stop RGB delivery during upload.
        async with asyncio.timeout(self.manager.boot_timeout + self.manager.max_age):
            for reference in references:
                request_id = uuid4().hex
                self.reference_waiters.add(request_id)
                try:
                    if not self.offer("reference_prepare", (request_id, reference)):
                        raise RuntimeError("Reference recording is unavailable")
                    while request_id not in self.reference_confirmed:
                        if self.failure or self.closed:
                            raise RuntimeError(
                                "Reference recording could not be confirmed"
                            )
                        self._acknowledged.clear()
                        await self._acknowledged.wait()
                finally:
                    self.reference_waiters.discard(request_id)
                    self.reference_confirmed.discard(request_id)
        return reference_descriptors(references)

    def source_rgb(self, frame):
        self.last_offered_frame = frame.index
        archived = replace(
            frame,
            rgb=b"",
            provenance=dict(
                frame.provenance or {},
                source_frame_index=frame.index,
                capture_pts_us=capture_pts(frame),
            ),
        )
        return self._raw_offer(
            "whitebox", "source_rgb", frame.rgb, lambda d: (archived, d)
        )

    def selected_source_rgb(self, index, frame):
        return self._raw_offer(
            "whitebox",
            "selected_source_rgb",
            frame.rgb,
            lambda d: (index, presentation_pts(frame), frame.index, d),
        )

    def model_rgb(self, index, frame, rgb, prompt):
        self.last_offered_frame = index
        archived = replace(
            frame,
            index=index,
            rgb=b"",
            pts_us=self.spec.pts(index),
            provenance=dict(
                frame.provenance or {},
                source_frame_index=frame.index,
                capture_pts_us=capture_pts(frame),
            ),
        )
        return self._raw_offer(
            "model", "model_rgb", rgb, lambda d: (archived, d, prompt)
        )

    def _release_rgb(self, kind, value):
        if kind not in ("source_rgb", "selected_source_rgb", "model_rgb"):
            return
        descriptor = value[-1] if kind == "selected_source_rgb" else value[1]
        lane = "model" if kind == "model_rgb" else "whitebox"
        self.rgb_slots[lane].release(descriptor)

    def _close_rgb(self):
        for pool in self.rgb_slots.values():
            pool.close()
        self.rgb_slots.clear()

    def statistics(self):
        value = super().statistics()
        value.update(
            diskless=True,
            disk_bytes=0,
            disk_free_bytes=None,
            disk_budget_bytes=0,
            upload_buffer_bytes=self.upload_buffer_bytes,
            produced_bytes=self.produced_bytes,
            confirmed_upload_bytes=self.confirmed_upload_bytes,
            object_buffer_bytes=self.object_buffer_bytes,
            last_processed_frame=self.last_frame,
            independent_encoding=self.raw_archive,
            worker_cpu=self.worker_cpu,
            rgb_reserved_bytes=sum(p.size for p in self.rgb_slots.values()),
            rgb_inflight_frames=sum(len(p.leases) for p in self.rgb_slots.values()),
            rgb_high_water_frames=sum(p.high_water for p in self.rgb_slots.values()),
            last_successful_frame=(
                self.last_frame if self.manifest["upload_status"] == "complete" else -1
            ),
        )
        return value

    def degrade(self, reason):
        super().degrade(reason)
        self.manifest["last_processed_frame"] = self.last_frame
        self.manifest["last_successful_frame"] = -1
        self.manifest["missing_frame_range"]["start"] = 0

    def report_access(self, token_hash):
        self.token_hash = token_hash
        super().report_access(token_hash)

    def bind_ui(self, engine, metadata):
        self.ui_engine = engine
        self.ui_metadata = copy.deepcopy(metadata)

    async def get_ui_asset(self, name):
        relative_name(name)
        expected = self.manifest.get("ui", {}).get("files", {}).get(name)
        if not expected or self.ui_closed or self.ui_engine is None:
            raise FileNotFoundError("UI asset is unavailable")
        if name in self.ui_assets:
            return self.ui_assets[name]
        if name not in self.ui_tasks:
            if len(self.ui_tasks) >= 128:
                raise ValueError("UI asset count exceeded")

            async def load():
                async with self.ui_gate:
                    data = await asyncio.wait_for(self.ui_engine.ui_asset(name), 10)
                    if (
                        self.ui_closed
                        or self.ui_bytes + len(data) > 32 << 20
                        or hashlib.sha256(data).hexdigest() != expected
                    ):
                        raise ValueError("UI asset validation failed")
                    self.ui_assets[name] = data
                    self.ui_bytes += len(data)
                    # UI service remains available even when archival has failed.
                    if self.accepting:
                        self.offer("asset", (name, data))
                    return data

            self.ui_tasks[name] = asyncio.create_task(load())
            self.ui_tasks[name].add_done_callback(
                lambda t: t.exception() if not t.cancelled() else None
            )
        return await asyncio.shield(self.ui_tasks[name])

    async def capture_assets(self, engine, metadata):
        self.bind_ui(engine, metadata)
        for name in metadata.get("files", {}):
            if self.ui_closed:
                return
            try:
                await self.get_ui_asset(name)
            except (Exception, asyncio.CancelledError):
                if not self.ui_closed:
                    self.degrade("UI_ASSET_FAILED")
                return

    def release_ui(self):
        self.ui_closed = True
        for task in self.ui_tasks.values():
            if not task.done():
                task.cancel()
        self.ui_tasks.clear()
        self.ui_assets.clear()
        self.ui_bytes = 0
        self.ui_engine = None

    async def encoded_video(self, index, encoded, *, name="ui_rendered", pts_us=None):
        self.offer("video", (name, index, encoded, pts_us))

    async def write_metadata(self, frame, prompt):
        from dataclasses import replace

        self.last_offered_frame = frame.index
        self.offer("metadata", (replace(frame, rgb=b""), prompt))

    async def video(self, *_args, **_kwargs):
        # No raw-video/disk fallback is allowed in this manager.
        self.degrade("ENCODED_ARCHIVE_REQUIRED")

    def worker_failed(self, detail):
        detail = detail if isinstance(detail, dict) else {}
        self.manifest["unfinished_uploads_observed"] = detail.get(
            "unfinished_uploads", []
        )
        self.degrade(detail.get("reason", "STREAM_UPLOAD_FAILED"))
        if detail.get("diagnostic"):
            logging.getLogger(__name__).warning(
                "Streaming archive worker: %s", detail["diagnostic"]
            )

    async def _run(self):
        commands = replies = None
        acquired = False
        cpu_reserved = False
        try:
            if self.manager.store_config is None:
                self.degrade("ARCHIVE_UNAVAILABLE")
                return
            if self.manager.recorders.locked():
                self.degrade("RECORDER_CAPACITY_EXHAUSTED")
                return
            await self.manager.recorders.acquire()
            acquired = True
            self.worker_cpu = self.manager.recorder_cpu
            if self.raw_archive:
                self.worker_cpu = self.manager.recorder_cpu_slots.popleft()
                cpu_reserved = True
            context = multiprocessing.get_context("spawn")
            commands, replies = context.Queue(1), context.Queue(2)
            self.process = context.Process(
                target=self.manager.worker_target,
                args=(
                    commands,
                    replies,
                    self.manager.store_config,
                    self.kwargs,
                    copy.deepcopy(self.manifest),
                    self.manager.settings(self),
                    self.worker_cpu,
                    self.manager.worker_memory_bytes,
                ),
                daemon=True,
            )
            starting = asyncio.create_task(asyncio.to_thread(self.process.start))
            try:
                await asyncio.shield(starting)
            except asyncio.CancelledError:
                await asyncio.gather(starting, return_exceptions=True)
                raise
            kind, detail = await self._reply(replies, self.manager.boot_timeout)
            if kind != "boot":
                self.worker_failed(detail)
                raise RuntimeError("Archive worker boot failed")
            self.worker_ready = True
            self.startup_complete.set()
            while not self.failure:
                if not self.items:
                    if self.final_outcome is not None:
                        self.manifest["trace"] = (
                            self.trace_snapshot() if self.trace_snapshot else []
                        )
                        commands.put_nowait(
                            (
                                "finish",
                                (self.final_outcome, copy.deepcopy(self.manifest)),
                            )
                        )
                        finish_deadline = time.monotonic() + self.manager.finish_timeout
                        kind, result = await self._reply(
                            replies, self.manager.finish_timeout
                        )
                        if kind == "rgb_released":
                            self._close_rgb()
                            kind, result = await self._reply(
                                replies, max(0, finish_deadline - time.monotonic())
                            )
                        if kind != "finished":
                            self.worker_failed(result)
                            raise RuntimeError("Archive upload did not finish")
                        self.manifest = result
                        self.upload_buffer_bytes = self.object_buffer_bytes = 0
                        stats = result.get("upload_statistics", {})
                        self.produced_bytes = stats.get("produced_bytes", 0)
                        self.confirmed_upload_bytes = stats.get(
                            "confirmed_upload_bytes", 0
                        )
                        return
                    self.wake.clear()
                    try:
                        await asyncio.wait_for(self.wake.wait(), 0.2)
                    except TimeoutError:
                        if not self.process.is_alive():
                            raise RuntimeError("Archive worker exited")
                        if self.reference_waiters:
                            self.offer("archive_status", None)
                    continue
                self.pending = self.items.popleft()
                queued, size, kind, value = self.pending
                age_limit = (
                    1.0
                    if kind in ("source_rgb", "selected_source_rgb", "model_rgb")
                    else self.manager.max_age
                )
                if time.monotonic() - queued > age_limit:
                    raise TimeoutError("Archive queue expired")
                commands.put_nowait((kind, value))
                reply, result = await self._reply(replies, self.manager.max_age)
                if reply != "ack":
                    self.worker_failed(result)
                    raise RuntimeError("Archive upload failed")
                self._release_rgb(kind, value)
                for codec, milliseconds in result.get("encoding_ms", {}).items():
                    observe_duration(
                        "archive_" + codec + "_encode", milliseconds, self.trace_id
                    )
                self.reference_confirmed.update(
                    request_id
                    for request_id in result.get("reference_receipts", [])
                    if request_id in self.reference_waiters
                )
                self.last_frame = result["last_frame"]
                self.upload_buffer_bytes = result.get("upload_pending_bytes", 0)
                self.object_buffer_bytes = result.get("object_buffer_bytes", 0)
                self.produced_bytes = result.get("produced_bytes", 0)
                self.confirmed_upload_bytes = result.get("confirmed_upload_bytes", 0)
                self.pending_bytes -= size
                self.pending = None
                self._completed_sequence += 1
                self._acknowledged.set()
        except asyncio.CancelledError:
            self.degrade("ARCHIVE_CANCELLED")
        except Exception:
            self.degrade("STREAM_UPLOAD_FAILED_OR_STALLED")
        finally:
            self.startup_complete.set()
            if self.process and self.process.pid is not None:
                # Give normal completion/abort a brief chance; never hold the
                # real-time owner behind a blocked uploader's network timeout.
                await asyncio.to_thread(self.process.join, 0.2)
                if self.process.is_alive():
                    self.process.terminate()
                    await asyncio.to_thread(self.process.join, 1)
                if self.process.is_alive():
                    self.process.kill()
                    await asyncio.to_thread(self.process.join, 1)
            for ipc in (commands, replies):
                if ipc is not None:
                    ipc.cancel_join_thread()
                    ipc.close()
            self.pending = None
            self.items.clear()
            self.pending_bytes = 0
            self.upload_buffer_bytes = self.object_buffer_bytes = 0
            self.kwargs = {}
            # The child has exited before outstanding RGB leases are unlinked.
            self._close_rgb()
            if cpu_reserved:
                self.manager.recorder_cpu_slots.append(self.worker_cpu)
            if acquired:
                self.manager.recorders.release()
            self.closed = True
            self._acknowledged.set()
            if self.failure:
                self.manager.publish_failure(self)

    async def finish(self, outcome):
        for capture in list(getattr(self, "model_input_captures", [])):
            capture.close()
        self.accepting = False
        self.final_outcome = outcome
        self.wake.set()
        await self.task
        if self.failure:
            self.manifest["missing_frame_range"]["end"] = self.last_offered_frame + 1
            self.manager.publish_failure(self)


class StreamingArchiveManager(ArchiveManager):
    diskless = True

    @property
    def supports_rgb(self):
        # Explicitly disabled archival remains a supported local/play-only mode.
        return self.store_config is not None

    def __init__(
        self,
        store=None,
        *,
        index_prefix="whitebox",
        queue_depth=256,
        queue_bytes=32 << 20,
        part_bytes=8 << 20,
        upload_bytes=32 << 20,
        max_age=5.0,
        boot_timeout=15.0,
        finish_timeout=120.0,
        rgb_budget_bytes=96 << 20,
        **kwargs,
    ):
        if type(rgb_budget_bytes) is not int or not (
            32 << 20 <= rgb_budget_bytes <= 512 << 20
        ):
            raise ValueError("Invalid streaming archive RGB budget")
        super().__init__(
            store,
            processes=True,
            queue_depth=queue_depth,
            queue_bytes=queue_bytes,
            **kwargs,
        )
        self.rgb_budget = RGBBudget(limit=rgb_budget_bytes)
        cpus = (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        )
        # RGB encoding is CPU work, unlike the old remux-only writers. Keep
        # active source/segment workers off the same single core when the Room
        # has capacity; never leave the container's existing CPU affinity set.
        self.recorder_cpu_slots = deque(
            reversed(cpus[-self.recorder_slots :])
            if len(cpus) >= self.recorder_slots + 2
            else [self.recorder_cpu] * self.recorder_slots
        )
        if not (
            5 << 20 <= part_bytes <= 16 << 20
            and part_bytes <= upload_bytes <= 64 << 20
            and 1 <= queue_depth <= 512
            and 1 << 20 <= queue_bytes <= 64 << 20
        ):
            raise ValueError("Invalid streaming archive memory budget")
        if min(max_age, boot_timeout, finish_timeout) <= 0:
            raise ValueError("Invalid streaming archive deadline")
        self.part_bytes, self.upload_bytes = part_bytes, upload_bytes
        self.max_age, self.boot_timeout, self.finish_timeout = (
            max_age,
            boot_timeout,
            finish_timeout,
        )
        self.worker_memory_bytes = 2 << 30
        self.disk_bytes = 0
        self.fail_open = True
        self.store_config = getattr(store, "worker_config", None)
        self.worker_target = streaming_worker
        self.live_recordings = {}
        self.index_prefix = index_prefix.strip("/") + "/.archive-index/v1/"
        self.io_active = 0
        self.status_pending = set()

    def index_key(self, recording_id):
        if not re.fullmatch(r"[a-f0-9]{32}", recording_id):
            raise ValueError("Invalid recording identity")
        return self.index_prefix + recording_id + ".json"

    def settings(self, recording):
        return dict(
            part_bytes=self.part_bytes,
            upload_bytes=self.upload_bytes,
            archive_bytes=2 << 30,
            finish_seconds=max(1, self.finish_timeout - 10),
            index_key=self.index_key(recording.directory.name),
        )

    async def create_recording(self, **kwargs):
        # Do not wait for process boot, a network connection, or an upload.
        recording = StreamingRecording(self, kwargs)
        for key, item in list(self.live_recordings.items()):
            if len(self.live_recordings) < 128:
                break
            if item.closed and item.final_outcome is not None:
                self.live_recordings.pop(key)
        self.live_recordings[recording.directory.name] = recording
        recording.manifest["resource_budgets"] = dict(
            queue_bytes=self.queue_bytes,
            queue_items=self.queue_depth,
            upload_bytes=self.upload_bytes,
            part_bytes=self.part_bytes,
            object_buffer_bytes=96 << 20,
            max_age_seconds=self.max_age,
            disk_bytes=0,
            rgb_bytes=self.rgb_budget.limit,
            rgb_frames_per_stream=8,
            reference_upload_bytes=8 << 20,
        )
        return recording

    def start(self, spool):
        self.spool = Path(spool)  # Logical paths only; old recordings stay readable.

    def upload_statistics(self):
        live = [r for r in self.live_recordings.values() if not r.closed]
        return dict(
            pending=len(live) + self.upload_pending,
            workers=len(live) * 2,
            workers_per_archive=2,
            pending_bytes=sum(r.upload_buffer_bytes for r in live),
            object_buffer_bytes=sum(r.object_buffer_bytes for r in live),
            client_verifiers=self.upload_pending,
        )

    def finalize(self, recording, outcome, *, report=None):
        if recording.manifest.get("archive_kind") == "game":
            self.finalize_game(recording, outcome)
            return
        recording.accepting = False
        recording.final_outcome = outcome
        recording.wake.set()
        recording.release_ui()
        self.active.add(recording.directory)

        async def finish():
            try:
                await recording.finish(outcome)
            finally:
                self.active.discard(recording.directory)
                if report:
                    report.gateway_done.set()
                    report.close()

        task = asyncio.create_task(finish())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    def finalize_game(self, recording, outcome):
        """The ordered Room receipt log is also best-effort and memory-only."""
        assert recording.diskless
        if recording.directory in self.active:
            return
        self.active.add(recording.directory)

        async def finish():
            try:
                await recording.finish(outcome)
                if self.store is None:
                    raise RuntimeError("Archive storage is unavailable")
                if not recording.failure:
                    await self.io(
                        put_bytes,
                        self.store.client,
                        self.store.bucket,
                        recording.manifest["object_prefix"] + "inputs.jsonl",
                        recording.archive_payload,
                        "application/x-ndjson",
                        timeout=30,
                    )
                recording.manifest["upload_status"] = (
                    "failed" if recording.failure else "complete"
                )
                await self.io(
                    put_json,
                    self.store.client,
                    self.store.bucket,
                    recording.manifest["object_prefix"] + "manifest.json",
                    recording.manifest,
                )
            except (Exception, asyncio.CancelledError):
                recording.manifest.update(
                    recording_status="failed",
                    upload_status="failed",
                    archive_failure=recording.failure or "GAME_ARCHIVE_UPLOAD_FAILED",
                )
            finally:
                recording.buffer.clear()
                recording.archive_payload = None
                self.active.discard(recording.directory)

        # Refuse excess bookkeeping instead of queuing unbounded completed games.
        # Source admission itself never awaits this upload.
        if len(self.active) > self.max_pending:
            recording.buffer.clear()
            recording.manifest.update(
                recording_status="failed",
                upload_status="failed",
                archive_failure="GAME_ARCHIVE_CAPACITY_EXHAUSTED",
            )
            self.active.discard(recording.directory)
            return
        task = asyncio.create_task(finish())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    async def io(self, function, *args, timeout=15):
        if self.io_active >= 2:
            raise RuntimeError("Archive metadata service is busy")
        self.io_active += 1
        task = asyncio.create_task(asyncio.to_thread(function, *args))

        def done(result):
            self.io_active -= 1
            if not result.cancelled():
                result.exception()

        task.add_done_callback(done)
        return await asyncio.wait_for(asyncio.shield(task), timeout)

    async def read_object(self, key, max_bytes):
        if self.store is None:
            raise FileNotFoundError("Archive storage is unavailable")

        def read():
            try:
                result = self.store.client.get_object(Bucket=self.store.bucket, Key=key)
            except Exception as error:
                code = getattr(error, "response", {}).get("Error", {}).get("Code")
                if code in {"NoSuchKey", "NotFound", "404"}:
                    raise FileNotFoundError("Archive object is unavailable") from None
                raise RuntimeError("Archive storage is unavailable") from None
            body = result["Body"]
            try:
                if result.get("ContentLength", max_bytes + 1) > max_bytes:
                    raise ValueError("Archive object exceeds its read budget")
                data = body.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise ValueError("Archive object exceeds its read budget")
                return data
            finally:
                body.close()

        return await self.io(read)

    async def read_index(self, recording_id):
        value = json.loads(await self.read_object(self.index_key(recording_id), 16384))
        prefix = value.get("object_prefix", "")
        root = self.index_prefix.removesuffix(".archive-index/v1/")
        if (
            value.get("recording_id") != recording_id
            or not prefix.startswith(root)
            or not prefix.endswith(recording_id + "/")
            or ".." in prefix.split("/")
            or value.get("archive_storage") != STORAGE
        ):
            raise ValueError("Invalid archive index")
        if not isinstance(value.get("token_hash"), str) or not re.fullmatch(
            r"[a-f0-9]{64}", value["token_hash"]
        ):
            raise ValueError("Invalid report capability")
        return value

    async def read_manifest(self, recording_id):
        index = await self.read_index(recording_id)
        value = json.loads(
            await self.read_object(index["object_prefix"] + "manifest.json", 2 << 20)
        )
        if (
            value.get("recording_id") != recording_id
            or value.get("object_prefix") != index["object_prefix"]
        ):
            raise ValueError("Invalid archive identity")
        return value

    def publish_failure(self, recording):
        if (
            self.store is None
            or recording.directory in self.status_pending
            or len(self.status_pending) >= 2
        ):
            return
        self.status_pending.add(recording.directory)
        value = copy.deepcopy(recording.manifest)
        token_hash = recording.token_hash

        async def publish():
            try:
                # A final manifest PUT may have succeeded despite a lost reply.
                # Do not overwrite a confirmed archive with an inferred failure.
                try:
                    remote = json.loads(
                        await self.read_object(
                            value["object_prefix"] + "manifest.json", 2 << 20
                        )
                    )
                except FileNotFoundError:
                    remote = None
                if (
                    remote
                    and remote.get("upload_status") == "complete"
                    and remote.get("recording_id") == value["recording_id"]
                    and remote.get("object_prefix") == value["object_prefix"]
                ):
                    recording.manifest = remote
                    recording.failure = None
                    return

                def abort_observed():
                    deadline = time.monotonic() + 10
                    for upload in value.get("unfinished_uploads_observed", [])[:192]:
                        if time.monotonic() >= deadline:
                            break
                        key, upload_id = upload.get("key"), upload.get("upload_id")
                        if (
                            not isinstance(key, str)
                            or not key.startswith(value["object_prefix"])
                            or not isinstance(upload_id, str)
                        ):
                            continue
                        try:
                            self.store.client.abort_multipart_upload(
                                Bucket=self.store.bucket,
                                Key=key,
                                UploadId=upload_id,
                            )
                        except Exception:
                            pass

                if value.get("unfinished_uploads_observed"):
                    await self.io(abort_observed)
                if token_hash:
                    await self.io(
                        put_json,
                        self.store.client,
                        self.store.bucket,
                        self.index_key(recording.directory.name),
                        {
                            "recording_id": recording.directory.name,
                            "token_hash": token_hash,
                            "object_prefix": value["object_prefix"],
                            "archive_storage": STORAGE,
                        },
                    )
                await self.io(
                    put_json,
                    self.store.client,
                    self.store.bucket,
                    value["object_prefix"] + "manifest.json",
                    value,
                )
            except Exception:
                pass  # Best effort: a failed OSS cannot be required for playback.
            finally:
                self.status_pending.discard(recording.directory)

        task = asyncio.create_task(publish())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def close(self):
        for recording in list(self.live_recordings.values()):
            recording.release_ui()
            if not recording.closed:
                recording.final_outcome = recording.final_outcome or "runtime_stopped"
                recording.wake.set()
        pending = [r.task for r in self.live_recordings.values() if not r.task.done()]
        if pending:
            await asyncio.wait(pending, timeout=min(5, self.finish_timeout))
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if self.tasks:
            await asyncio.wait(self.tasks, timeout=3)
            for task in list(self.tasks):
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
