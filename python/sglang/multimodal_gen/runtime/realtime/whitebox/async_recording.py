# SPDX-License-Identifier: Apache-2.0
"""Fail-open archive sidecar for Room pipelines.

The live owner only snapshots bounded metadata and offers immutable frame bytes.
Disk, archive codecs and network upload run in killable, resource-limited child
processes. Neither a full queue nor a dead/hung child raises into the live owner.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import multiprocessing
import os
import queue
import time
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone

from .contracts import presentation_pts
from .metrics import observe_duration
from .recording import ArchiveManager, Recording, reference_descriptors, write_manifest


def limit_child(cpu, memory_bytes=2 * 1024**3, file_bytes=2 * 1024**3):
    import resource

    from .recording_process import worker_init

    worker_init(cpu, 10)
    if os.name == "posix":
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
        if os.uname().sysname == "Linux":
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def _client_upload_child(replies, directory, intent_path, store_config, cpu):
    from .client_recording import verify_and_publish

    limit_child(cpu)
    try:
        replies.put((True, verify_and_publish(directory, intent_path, store_config)))
    except BaseException:
        replies.put((False, None))


def _worker(
    commands,
    replies,
    kwargs,
    cpu,
    memory_bytes,
    disk_bytes,
    reserve_bytes,
    io_bytes_per_second,
):
    import shutil

    limit_child(cpu, memory_bytes, disk_bytes)
    recording = None
    disk_checked_at, disk_last_size = time.monotonic(), 0
    disk_free = None
    last_frame = -1
    try:
        parent = kwargs["directory"].parent
        parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(parent).free < reserve_bytes:
            raise OSError("Archive free-space reserve reached")
        recording = Recording(**kwargs, synchronous=True, reuse_playback=True)
        replies.put(("boot", recording.manifest))
        while True:
            kind, value = commands.get()
            started = time.monotonic()
            if started - disk_checked_at >= 0.5 or kind == "finish":
                used = sum(
                    path.stat().st_size
                    for path in recording.directory.rglob("*")
                    if path.is_file()
                )
                disk_free = shutil.disk_usage(recording.directory).free
                if used > disk_bytes or disk_free < reserve_bytes:
                    raise OSError("Archive disk budget reached")
                delay = (used - disk_last_size) / io_bytes_per_second - (
                    started - disk_checked_at
                )
                if delay > 0:
                    time.sleep(delay)  # Only this archive child is throttled.
                disk_checked_at, disk_last_size = time.monotonic(), used
            if kind == "pair":
                frame, source, encoded, prompt = value
                pts = presentation_pts(frame)
                recording.encoders["whitebox"].append(frame.index, source, pts)
                recording.encoders["model"].append(frame.index, encoded, pts)
                recording.aligned_metadata(replace(frame, rgb=b""), prompt)
                last_frame = frame.index
            elif kind == "source_frame":
                frame, encoded = value
                if not recording.source_only:
                    raise ValueError("Source frames require a source recording")
                recording.encoders["whitebox"].append(
                    frame.index, encoded, frame.provenance["capture_pts_us"]
                )
                recording.aligned_metadata(replace(frame, rgb=b""), "")
                last_frame = frame.index
            elif kind == "state":
                recording.source_state(value)
            elif kind == "configure":
                recording.manifest = value
                write_manifest(recording.directory, recording.manifest)
            elif kind == "references":
                recording.store_references(value)
            elif kind == "asset":
                name, data = value
                target = recording.directory / "ui_bundle" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            elif kind == "report":
                token_hash, created_at = value
                target = recording.directory.parent / ".reports"
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
                import json

                access = target / (recording.directory.name + ".json")
                access.write_text(
                    json.dumps({"token_hash": token_hash, "created_at": created_at})
                )
                access.chmod(0o600)
            elif kind == "finish":
                outcome, manifest = value
                recording.manifest.update(manifest)
                asyncio.run(recording.finish(outcome))
                replies.put(("finished", recording.manifest))
                return
            else:
                raise ValueError("Unsupported archive operation")
            replies.put(
                (
                    "ack",
                    {
                        "last_frame": last_frame,
                        "duration_ms": (time.monotonic() - started) * 1000,
                        "disk_bytes": disk_last_size,
                        "disk_free_bytes": disk_free,
                    },
                )
            )
    except BaseException:
        # Exception details stay in this process; only a stable failure reaches
        # live status. A damaged archive must never be presented as complete.
        if recording:
            try:
                recording.manifest.update(
                    recording_status="failed",
                    upload_status="pending",
                    archive_failure="WORKER_FAILED",
                    last_successful_frame=last_frame,
                )
                recording.manifest["timeline"]["alignment"] = "unverified"
                write_manifest(recording.directory, recording.manifest)
            except Exception:
                pass
        try:
            replies.put(("failed", {"last_frame": last_frame}))
        except Exception:
            pass


def _persist_failure(directory, manifest):
    import json

    try:
        path = directory / "manifest.json"
        previous = json.loads(path.read_text()) if path.exists() else manifest.copy()
        if (
            previous.get("trace_id") != manifest["trace_id"]
            or previous.get("recording_id") != manifest.get("recording_id")
            or previous.get("game_id") != manifest.get("game_id")
        ):
            return
        for key in (
            "recording_status",
            "archive_failure",
            "last_successful_frame",
            "missing_frame_range",
            "upload_status",
            "upload_failure",
        ):
            if key in manifest:
                previous[key] = manifest[key]
        if "timeline" in previous and manifest.get("archive_failure"):
            previous["timeline"]["alignment"] = "unverified"
        directory.mkdir(parents=True, exist_ok=True)
        write_manifest(directory, previous)
    except (OSError, ValueError, KeyError):
        pass  # Disk failure cannot be repaired by waiting in the live pipeline.


async def persist_outcome(directory, manifest):
    child = multiprocessing.get_context("spawn").Process(
        target=_persist_failure, args=(directory, copy.deepcopy(manifest)), daemon=True
    )
    try:
        await asyncio.to_thread(child.start)
        await asyncio.to_thread(child.join, 3)
    except Exception:
        pass
    finally:
        if child.pid and child.is_alive():
            child.kill()
            await asyncio.to_thread(child.join, 1)


def _upload_child(replies, directory, store_config, cpu):
    import json

    from .recording import S3ArchiveStore, publish_archive
    from .reports import SessionReport

    limit_child(cpu)
    report = None
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        report = SessionReport(manifest["trace_id"])
        previous = manifest.get("trace", [])
        asyncio.run(
            publish_archive(
                directory,
                S3ArchiveStore(**store_config),
                concurrency=1,
                retries=2,
                trace_snapshot=lambda: previous + report.snapshot(),
            )
        )
        replies.put((True, json.loads((directory / "manifest.json").read_text())))
    except BaseException:
        replies.put((False, None))
    finally:
        if report:
            report.close()


def _payload_size(value, depth=0):
    if depth > 24:
        raise ValueError("Archive metadata nesting exceeds budget")
    if isinstance(value, (bytes, bytearray, str)):
        return len(value) * (4 if isinstance(value, str) else 1) + 64
    if isinstance(value, dict):
        return 64 + sum(
            _payload_size(k, depth + 1) + _payload_size(v, depth + 1)
            for k, v in value.items()
        )
    if isinstance(value, (tuple, list)):
        return 64 + sum(_payload_size(v, depth + 1) for v in value)
    if hasattr(value, "__dataclass_fields__"):
        return _payload_size(vars(value), depth + 1)
    return 32


class AsyncRecording:
    """Frame admission never waits; condition changes can confirm asset writes."""

    def __init__(self, manager, kwargs):
        self.manager, self.kwargs = manager, kwargs
        self.directory, self.spec, self.trace_id = (
            kwargs[k] for k in ("directory", "spec", "trace_id")
        )
        self.model_spec = kwargs.get("model_spec", self.spec)
        self.trace_snapshot = None
        self.references = [
            dict(value, path=f"references/{i:02d}.{ref['format']}")
            for i, (value, ref) in enumerate(
                zip(reference_descriptors(kwargs["references"]), kwargs["references"])
            )
        ]
        self.manifest = dict(
            schema_version=1,
            recording_id=self.directory.name,
            world_id=kwargs["world_id"],
            created_at=datetime.now(timezone.utc).isoformat(),
            trace_id=self.trace_id,
            object_prefix=kwargs["object_prefix"].rstrip("/") + "/",
            recording_status="finalizing",
            upload_status="pending",
            dataset_status="excluded_unvalidated",
            inference=kwargs["inference"],
            ui=kwargs["ui"],
            quality={"unknown_input_frames": 0},
            timeline=dict(
                fps={"num": self.spec.fps, "den": 1},
                mode="source_capture_vfr",
                model_clock_fps=self.spec.fps,
                pts_unit="us",
                first_frame=0,
                interval="[start,end)",
                alignment="unverified",
            ),
            inputs={"path": "inputs.jsonl", "format": "effective_intervals_v1"},
            states={"path": "states.jsonl", "format": "opaque_frame_envelope_v1"},
            client_ui={
                "owner": "browser",
                "receipt_path": "ui-recording.json",
                "status": "awaiting_upload",
            },
        )
        self.items = deque()
        self.pending_bytes = 0
        self.pending = None
        self.last_frame = -1
        self.last_offered_frame = -1
        self.rejected = 0
        self.disk_used = 0
        self.disk_free = None
        self.failure = None
        self.accepting = True
        self.closed = False
        self.wake = asyncio.Event()
        self._offered_sequence = self._completed_sequence = 0
        self._acknowledged = asyncio.Event()
        self.process = None
        self.worker_ready = False
        self.startup_complete = asyncio.Event()
        self.final_outcome = None
        self.task = asyncio.create_task(
            self._run(), name="room-archive-" + self.directory.name
        )

    def statistics(self):
        oldest = self.pending or (self.items[0] if self.items else None)
        return dict(
            process=True,
            pending_operations=len(self.items) + int(self.pending is not None),
            pending_bytes=self.pending_bytes,
            byte_capacity=self.manager.queue_bytes,
            queue_capacity=self.manager.queue_depth,
            oldest_wait_ms=(time.monotonic() - oldest[0]) * 1000 if oldest else 0,
            rejected_operations=self.rejected,
            disk_bytes=self.disk_used,
            disk_free_bytes=self.disk_free,
            disk_budget_bytes=self.manager.disk_bytes,
            failure=self.failure,
            missing_frame_range=self.manifest.get("missing_frame_range"),
            last_successful_frame=self.last_frame,
            status="failed" if self.failure else "closed" if self.closed else "running",
        )

    def degrade(self, reason):
        if self.failure:
            return
        self.failure = reason
        self.accepting = False
        self.items.clear()
        self.pending_bytes = self.pending[1] if self.pending else 0
        self.manifest.update(
            recording_status="failed",
            archive_failure=reason,
            upload_status="failed",
            upload_failure="ARCHIVE_FAILED",
            last_successful_frame=self.last_frame,
            missing_frame_range={"start": self.last_frame + 1, "end": None},
        )
        self.manifest["timeline"]["alignment"] = "unverified"
        self.wake.set()
        self._acknowledged.set()
        observe_duration("archive_offer_rejected", 0, self.trace_id, result="error")

    def offer(self, kind, value, *, size=None):
        if not self.accepting:
            self.rejected += 1
            return False
        oldest = self.pending or (self.items[0] if self.items else None)
        try:
            size = size if size is not None else _payload_size(value)
            if oldest and time.monotonic() - oldest[0] > self.manager.max_age:
                reason = "QUEUE_TOO_OLD"
            elif (
                len(self.items) + int(self.pending is not None)
                >= self.manager.queue_depth
            ):
                reason = "QUEUE_FULL"
            elif size + self.pending_bytes > self.manager.queue_bytes:
                reason = "BYTE_BUDGET_EXCEEDED"
            else:
                snapshot = copy.deepcopy(value)
                self.items.append((time.monotonic(), size, kind, snapshot))
                self._offered_sequence += 1
                self.pending_bytes += size
                self.wake.set()
                return True
        except (ValueError, RecursionError, MemoryError, TypeError):
            reason = "METADATA_TOO_LARGE"
        self.rejected += 1
        self.degrade(reason)
        return False

    def pair(self, frame, source, encoded, prompt):
        self.last_offered_frame = frame.index
        if not self.accepting:
            self.rejected += 1
            return False
        # bytes are immutable and shared until the bounded feeder serializes them.
        return self.offer("pair", (replace(frame, rgb=b""), source, encoded, prompt))

    def source_frame(self, frame, encoded):
        self.last_offered_frame = frame.index
        if not self.accepting:
            self.rejected += 1
            return False
        return self.offer("source_frame", (replace(frame, rgb=b""), encoded))

    async def configure(self):
        self.offer("configure", self.manifest)

    async def add_references(self, references, *, wait=False):
        accepted = self.offer("references", references)
        if wait:
            if not accepted:
                raise RuntimeError("Reference recording is unavailable")
            target = self._offered_sequence
            async with asyncio.timeout(
                self.manager.boot_timeout + self.manager.max_age
            ):
                while self._completed_sequence < target:
                    if self.failure or self.closed:
                        raise RuntimeError("Reference recording could not be confirmed")
                    self._acknowledged.clear()
                    await self._acknowledged.wait()
        return reference_descriptors(references)

    async def write_source_state(self, message):
        self.offer("state", message)

    def report_access(self, token_hash):
        self.offer("report", (token_hash, time.time()))

    async def capture_assets(self, engine, metadata):
        total = 0
        for name, expected in metadata.get("files", {}).items():
            if not self.accepting:
                return
            try:
                data = await asyncio.wait_for(engine.ui_asset(name), 10)
                total += len(data)
                if total > 32 * 1024**2:
                    raise ValueError("UI asset budget exceeded")
                if (
                    name.startswith("/")
                    or ".." in name.split("/")
                    or hashlib.sha256(data).hexdigest() != expected
                ):
                    raise ValueError("Invalid UI asset")
                self.offer("asset", (name, data))
            except Exception:
                self.degrade("UI_ASSET_FAILED")
                return

    async def _reply(self, replies, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.failure:
                raise RuntimeError(self.failure)
            try:
                return replies.get_nowait()
            except queue.Empty:
                if not self.process.is_alive():
                    raise RuntimeError("WORKER_EXITED")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # Wake as soon as the IPC reply arrives. A 10ms polling floor
                # serialized every frame/state operation and capped the writer
                # near 100 operations/s even when disk writes were fast.
                # The short timeout preserves worker-death/failure checks and
                # bounds the lifetime of a thread after owner cancellation.
                try:
                    return await asyncio.to_thread(
                        replies.get, True, min(0.05, remaining)
                    )
                except queue.Empty:
                    continue
        raise TimeoutError("WORKER_TIMEOUT")

    async def _run(self):
        commands = replies = None
        acquired = False
        try:
            # Capacity exhaustion only degrades this archive; never delay admission.
            if self.manager.recorders.locked():
                self.degrade("RECORDER_CAPACITY_EXHAUSTED")
                return
            await self.manager.recorders.acquire()
            acquired = True
            context = multiprocessing.get_context("spawn")
            commands, replies = context.Queue(maxsize=1), context.Queue(maxsize=2)
            self.process = context.Process(
                target=_worker,
                args=(
                    commands,
                    replies,
                    self.kwargs,
                    self.manager.recorder_cpu,
                    self.manager.worker_memory_bytes,
                    self.manager.disk_bytes,
                    self.manager.reserve_bytes,
                    self.manager.io_bytes_per_second,
                ),
                daemon=True,
            )
            starting = asyncio.create_task(asyncio.to_thread(self.process.start))
            try:
                await asyncio.shield(starting)
            except asyncio.CancelledError:
                # Starting a process in a thread cannot be cancelled. Fence its
                # completion before the finally block checks and stops the child.
                await asyncio.gather(starting, return_exceptions=True)
                raise
            kind, _ = await self._reply(replies, self.manager.boot_timeout)
            if kind != "boot":
                raise RuntimeError("WORKER_BOOT_FAILED")
            self.worker_ready = True
            self.startup_complete.set()
            while not self.failure:
                if not self.items:
                    if self.final_outcome is not None:
                        commands.put_nowait(
                            (
                                "finish",
                                (self.final_outcome, copy.deepcopy(self.manifest)),
                            )
                        )
                        kind, manifest = await self._reply(
                            replies, self.manager.finish_timeout
                        )
                        if kind != "finished":
                            raise RuntimeError("WORKER_FINISH_FAILED")
                        self.manifest = manifest
                        return
                    self.wake.clear()
                    try:
                        await asyncio.wait_for(self.wake.wait(), 0.2)
                    except TimeoutError:
                        if not self.process.is_alive():
                            raise RuntimeError("WORKER_EXITED")
                    continue
                self.pending = self.items.popleft()
                queued, size, operation, value = self.pending
                if time.monotonic() - queued > self.manager.max_age:
                    raise TimeoutError("QUEUE_TOO_OLD")
                observe_duration(
                    "archive_queue_wait",
                    (time.monotonic() - queued) * 1000,
                    self.trace_id,
                )
                commands.put_nowait((operation, value))
                kind, result = await self._reply(replies, self.manager.max_age)
                if kind != "ack":
                    raise RuntimeError("WORKER_FAILED")
                self.last_frame = result["last_frame"]
                self.disk_used = result.get("disk_bytes", 0)
                self.disk_free = result.get("disk_free_bytes")
                observe_duration(
                    "archive_worker_" + operation, result["duration_ms"], self.trace_id
                )
                self.pending_bytes -= size
                self.pending = None
                self._completed_sequence += 1
                self._acknowledged.set()
        except asyncio.CancelledError:
            self.degrade("ARCHIVE_CANCELLED")
        except Exception:
            self.degrade("WORKER_FAILED_OR_STALLED")
        finally:
            self.startup_complete.set()
            if self.process and self.process.pid is not None:
                if self.process.is_alive():
                    self.process.terminate()
                await asyncio.to_thread(self.process.join, 2)
                if self.process.is_alive():
                    self.process.kill()
                    await asyncio.to_thread(self.process.join, 2)
            if self.failure:
                await persist_outcome(self.directory, self.manifest)
            for ipc in (commands, replies):
                if ipc is not None:
                    ipc.cancel_join_thread()
                    ipc.close()
            self.pending = None
            self.items.clear()
            self.pending_bytes = 0
            # Drop init image snapshots as soon as the process is finished.
            self.kwargs = {}
            if acquired:
                self.manager.recorders.release()
            self.closed = True
            self._acknowledged.set()

    async def finish(self, outcome):
        self.accepting = False
        self.final_outcome = outcome
        self.manifest["recording_worker"] = self.statistics()
        self.manifest["trace"] = self.trace_snapshot() if self.trace_snapshot else []
        if self.failure:
            self.manifest["missing_frame_range"]["end"] = self.last_offered_frame + 1
        self.wake.set()
        await self.task
        if self.failure:
            # The child may have failed while gameplay continued. Persist the
            # final missing range, rather than leaving the earlier open range.
            self.manifest["missing_frame_range"]["end"] = self.last_offered_frame + 1
            await persist_outcome(self.directory, self.manifest)


class AsyncArchiveManager(ArchiveManager):
    def __init__(
        self,
        *args,
        max_age=5.0,
        boot_timeout=15.0,
        finish_timeout=30.0,
        worker_memory_bytes=2 * 1024**3,
        disk_bytes=2 * 1024**3,
        reserve_bytes=1024**3,
        io_bytes_per_second=16 * 1024**2,
        **kwargs,
    ):
        super().__init__(*args, processes=True, **kwargs)
        if min(max_age, boot_timeout, finish_timeout) <= 0:
            raise ValueError("Archive deadlines must be positive")
        self.max_age, self.boot_timeout, self.finish_timeout = (
            max_age,
            boot_timeout,
            finish_timeout,
        )
        if (
            min(disk_bytes, io_bytes_per_second, worker_memory_bytes) <= 0
            or reserve_bytes < 0
        ):
            raise ValueError("Invalid archive resource budgets")
        self.worker_memory_bytes = worker_memory_bytes
        self.disk_bytes, self.reserve_bytes, self.io_bytes_per_second = (
            disk_bytes,
            reserve_bytes,
            io_bytes_per_second,
        )
        self.live_recordings = {}
        self.fail_open = True
        self.upload_timeout = 120

    async def create_recording(self, **kwargs):
        for key, item in list(self.live_recordings.items()):
            if len(self.live_recordings) < 128:
                break
            if item.closed and item.directory not in self.active:
                self.live_recordings.pop(key, None)
        recording = AsyncRecording(self, kwargs)
        recording.manifest["resource_budgets"] = dict(
            recorder_slots=self.recorder_slots,
            queue_bytes=self.queue_bytes,
            queue_items=self.queue_depth,
            max_age_seconds=self.max_age,
            worker_address_space_bytes=self.worker_memory_bytes,
            archive_disk_bytes=self.disk_bytes,
            disk_reserve_bytes=self.reserve_bytes,
            archive_write_bytes_per_second=self.io_bytes_per_second,
            upload_deadline_seconds=self.upload_timeout,
        )
        self.live_recordings[recording.directory.name] = recording
        # This is session admission, before the live source pump starts. A cold
        # spawned worker must be ready before a fast source can fill its queue.
        # The existing boot deadline degrades failed archives; it does not turn
        # disk/worker failure into a live-session error or block per-frame offers.
        started = time.monotonic()
        try:
            await recording.startup_complete.wait()
        except asyncio.CancelledError:
            recording.task.cancel()
            await asyncio.gather(recording.task, return_exceptions=True)
            raise
        finally:
            observe_duration(
                "archive_startup_wait",
                (time.monotonic() - started) * 1000,
                recording.trace_id,
            )
        return recording

    def start(self, spool):
        # New archives have bounded attempts in finalize; no unbounded automatic
        # retry loop scanning historical disks in the Room event loop.
        self.spool = spool

    def finalize(self, recording, outcome, *, report=None):
        if isinstance(recording, AsyncRecording):
            recording.accepting = False
            recording.final_outcome = outcome
        self.active.add(recording.directory)

        async def finish():
            try:
                await recording.finish(outcome)
                if report:
                    report.gateway_done.set()
                if self.store is not None and recording.manifest.get(
                    "recording_status"
                ) in {"complete", "partial"}:
                    await self._publish_bounded(recording, report=report)
            finally:
                self.active.discard(recording.directory)
                if report:
                    report.gateway_done.set()
                    report.close()

        task = asyncio.create_task(
            finish(), name="room-archive-finish-" + recording.directory.name
        )
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        # Retrieve failures without propagating them into the gameplay owner.
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    async def _publish_bounded(self, recording, *, report=None):
        if self.upload_pending >= self.max_pending:
            recording.manifest.update(
                upload_status="failed", upload_failure="UPLOAD_CAPACITY_EXHAUSTED"
            )
            await persist_outcome(recording.directory, recording.manifest)
            return
        self.upload_pending += 1
        process = replies = None
        try:
            async with asyncio.timeout(self.upload_timeout):
                async with self.upload_gate:
                    config = getattr(self.store, "worker_config", None)
                    if config is None:
                        # Injected async test stores remain bounded by the same deadline.
                        from .recording import publish_archive

                        await publish_archive(
                            recording.directory, self.store, retries=2, concurrency=1
                        )
                        import json

                        recording.manifest = await asyncio.to_thread(
                            lambda: json.loads(
                                (recording.directory / "manifest.json").read_text()
                            )
                        )
                        return
                    context = multiprocessing.get_context("spawn")
                    replies = context.Queue(1)
                    process = context.Process(
                        target=_upload_child,
                        args=(replies, recording.directory, config, self.upload_cpu),
                        daemon=True,
                    )
                    await asyncio.to_thread(process.start)
                    while True:
                        try:
                            ok, manifest = replies.get_nowait()
                            if not ok:
                                raise RuntimeError("UPLOAD_FAILED")
                            recording.manifest = manifest
                            return
                        except queue.Empty:
                            if not process.is_alive():
                                raise RuntimeError("UPLOADER_EXITED")
                            await asyncio.sleep(0.05)
        except (Exception, asyncio.CancelledError):
            recording.manifest.update(
                upload_status="failed", upload_failure="UPLOAD_FAILED_OR_STALLED"
            )
        finally:
            self.upload_pending -= 1
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
            if recording.manifest.get("upload_status") == "failed":
                await persist_outcome(recording.directory, recording.manifest)

    async def verify_client(self, directory, path):
        """Same bounded uploader budget for optional browser-composited archives."""
        if self.upload_pending >= self.max_pending:
            raise RuntimeError("UPLOAD_CAPACITY_EXHAUSTED")
        self.upload_pending += 1
        child = replies = None
        try:
            async with asyncio.timeout(self.upload_timeout):
                async with self.upload_gate:
                    context = multiprocessing.get_context("spawn")
                    replies = context.Queue(1)
                    child = context.Process(
                        target=_client_upload_child,
                        args=(
                            replies,
                            directory,
                            path,
                            self.store.worker_config,
                            self.upload_cpu,
                        ),
                        daemon=True,
                    )
                    await asyncio.to_thread(child.start)
                    while True:
                        try:
                            ok, result = replies.get_nowait()
                            if not ok:
                                raise RuntimeError("CLIENT_UPLOAD_FAILED")
                            return result
                        except queue.Empty:
                            if not child.is_alive():
                                raise RuntimeError("CLIENT_UPLOAD_EXITED")
                            await asyncio.sleep(0.02)
        finally:
            self.upload_pending -= 1
            if child and child.pid:
                if child.is_alive():
                    child.terminate()
                await asyncio.to_thread(child.join, 1)
                if child.is_alive():
                    child.kill()
                    await asyncio.to_thread(child.join, 1)
            if replies:
                replies.cancel_join_thread()
                replies.close()

    async def close(self):
        for recording in list(self.live_recordings.values()):
            if not recording.closed:
                if recording.final_outcome is None:
                    recording.final_outcome = "runtime_stopped"
                recording.wake.set()
        if self.tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.tasks, return_exceptions=True),
                    self.finish_timeout + 5,
                )
            except TimeoutError:
                pass
        for recording in self.live_recordings.values():
            if not recording.task.done():
                recording.task.cancel()
        await asyncio.gather(
            *(r.task for r in self.live_recordings.values()), return_exceptions=True
        )
