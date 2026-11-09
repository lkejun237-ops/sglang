# SPDX-License-Identifier: Apache-2.0
"""Bounded IPC: disk, archive encoding and metadata have one process owner.

Only one operation is in the executor at a time. The asyncio queue bounds both
bytes and items; ProcessPoolExecutor's otherwise unbounded queue never holds an
entire recording. A slow/failed recorder applies visible backpressure, not drops.
"""

from __future__ import annotations

import asyncio
import copy
import multiprocessing
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from prometheus_client import Gauge

from .metrics import capture_stages, observe_duration, stage
from .recording import (
    Recording,
    S3ArchiveStore,
    publish_archive,
    reference_descriptors,
    write_manifest,
)

_recording = None
PENDING_OPERATIONS = Gauge(
    "world_model_whitebox_recording_pending_operations",
    "Queued and executing archive operations",
)
PENDING_BYTES = Gauge(
    "world_model_whitebox_recording_pending_bytes", "Bounded recorder payload bytes"
)


def worker_init(cpu, nice):
    # Each background worker uses one CPU and one codec thread. Affinity/nice
    # supplement the pod's CPU/memory limits; they are not per-process cgroups.
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    if nice and hasattr(os, "nice"):
        os.nice(nice)


def process_pool(cpu=None, nice=10):
    return ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=worker_init,
        initargs=(cpu, nice),
    )


def _boot(kwargs):
    global _recording
    _recording = Recording(**kwargs, synchronous=True, reuse_playback=True)
    return _recording.manifest, _recording.references, os.getpid()


def _operation(kind, args, enqueued_at):
    started = time.monotonic()
    with capture_stages() as spans:
        if kind == "video":
            name, index, payload, *timing = args
            label = (
                "mux_" if name == "ui_rendered" or _recording.browser_ui else "encode_"
            ) + name
            with stage(label, _recording.trace_id):
                _recording.encoders[name].append(index, payload, *timing)
        elif kind == "metadata":
            with stage("recording_metadata_write", _recording.trace_id):
                _recording.aligned_metadata(*args)
        elif kind == "configure":
            _recording.manifest = args[0]
            write_manifest(_recording.directory, _recording.manifest)
        elif kind == "source_state":
            _recording.source_state(args[0])
        elif kind == "references":
            _recording.store_references(args[0])
        else:
            raise ValueError("Unknown recorder operation")
    return (started - enqueued_at) * 1000, spans


def _finish(outcome, trace, statistics=None):
    _recording.manifest["trace"] = trace
    if statistics:
        _recording.manifest["recording_worker"] = statistics
    asyncio.run(_recording.finish(outcome))
    return _recording.manifest


def _publish(directory, store_config, trace):
    # The client, connection pool, checksum/hash work and upload threads belong
    # to this process. No boto clients or credentials cross the IPC boundary.
    import json

    from .reports import SessionReport

    manifest = json.loads((directory / "manifest.json").read_text())
    if trace is not None:
        manifest["trace"] = trace
        write_manifest(directory, manifest)
    report = SessionReport(manifest["trace_id"])
    previous = [
        row
        for row in manifest.get("trace", [])
        if row["stage"] != "whitebox.oss_upload"
    ]
    error = False
    with capture_stages() as spans:
        try:
            asyncio.run(
                publish_archive(
                    directory,
                    S3ArchiveStore(**store_config),
                    trace_snapshot=lambda: previous + report.snapshot(),
                )
            )
        except Exception:
            error = True
        finally:
            report.close()
    return spans, error


def _failed_manifest(directory, manifest):
    # A dead process may have left unfinalized MP4s. Preserve evidence and never
    # publish them as a complete/aligned dataset or silently retry the session.
    import json

    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except (FileNotFoundError, ValueError):
        pass
    manifest.update(
        recording_status="failed",
        execution_outcome="recording_worker_failed",
        upload_status="pending",
    )
    manifest["timeline"]["alignment"] = "unverified"
    write_manifest(directory, manifest)


class ProcessRecording:
    @classmethod
    async def create(
        cls,
        *,
        queue_depth=24,
        queue_bytes=64 * 1024 * 1024,
        cpu=None,
        release=None,
        **kwargs,
    ):
        if queue_depth < 1 or queue_bytes < kwargs["spec"].frame_bytes:
            raise ValueError("Recorder queue must hold at least one frame")
        self = cls()
        self.directory = kwargs["directory"]
        self.spec, self.trace_id = kwargs["spec"], kwargs["trace_id"]
        self.trace_snapshot = None
        self.pool = process_pool(cpu)
        self.release = release
        self.queue = asyncio.Queue(maxsize=queue_depth)
        self.byte_limit, self.pending_bytes = queue_bytes, 0
        self.pending = {}  # Sequence -> (enqueue time, serialized bytes).
        self.sequence = self.high_water = 0
        self.condition = asyncio.Condition()
        self.accepting = True
        self.failure = None
        self.closed = False
        self.unprocessed_operations = 0
        self.cpu = cpu
        self.worker_pid = None
        try:
            boot = asyncio.ensure_future(self._call(_boot, kwargs))
            try:
                self.manifest, self.references, self.worker_pid = await asyncio.shield(
                    boot
                )
            except asyncio.CancelledError:
                await boot
                await self._call(_finish, "aborted", [])
                raise
        except BaseException:
            await asyncio.to_thread(self.pool.shutdown, wait=True, cancel_futures=True)
            raise
        self.task = asyncio.create_task(self._drain(), name="whitebox-record-ipc")
        return self

    async def _call(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(
            self.pool, function, *args
        )

    def statistics(self):
        now = time.monotonic()
        oldest = min((item[0] for item in self.pending.values()), default=now)
        return dict(
            process=True,
            pending_operations=len(self.pending),
            queue_capacity=self.queue.maxsize,
            high_water=self.high_water,
            unprocessed_operations=self.unprocessed_operations,
            pending_bytes=self.pending_bytes,
            byte_capacity=self.byte_limit,
            oldest_wait_ms=(now - oldest) * 1000,
            status="failed" if self.failure else "closed" if self.closed else "running",
            cpu=self.cpu,
        )

    async def _enqueue(self, kind, args):
        # RGB/encoded payload sizes need no full-frame serialization on the live
        # event loop. The executor's feeder thread performs the actual pickling.
        size = (
            len(args[2]["payload"] if isinstance(args[2], dict) else args[2]) + 512
            if kind == "video"
            else len(pickle.dumps(args, protocol=5))
        )
        if size > self.byte_limit:
            raise ValueError("Recorder operation exceeds its bounded byte budget")
        with stage("recording_backpressure", self.trace_id):
            async with self.condition:
                await self.condition.wait_for(
                    lambda: (
                        self.failure
                        or not self.accepting
                        or (
                            not self.queue.full()
                            and self.pending_bytes + size <= self.byte_limit
                        )
                    )
                )
                if self.failure:
                    raise RuntimeError("Recorder worker failed") from self.failure
                if not self.accepting:
                    raise RuntimeError("Recorder no longer accepts frames")
                self.sequence += 1
                now = time.monotonic()
                self.pending[self.sequence] = (now, size)
                self.pending_bytes += size
                PENDING_OPERATIONS.inc()
                PENDING_BYTES.inc(size)
                self.high_water = max(self.high_water, len(self.pending))
                self.queue.put_nowait((self.sequence, kind, args, now))
                return self.sequence

    async def _drain(self):
        try:
            while (item := await self.queue.get()) is not None:
                sequence, kind, args, enqueued = item
                wait_ms, spans = await self._call(_operation, kind, args, enqueued)
                observe_duration("recording_worker_queue", wait_ms, self.trace_id)
                for span in spans:
                    observe_duration(
                        span["stage"],
                        span["duration_ms"],
                        self.trace_id,
                        result=span["result"],
                        service="recorder",
                    )
                async with self.condition:
                    size = self.pending.pop(sequence)[1]
                    self.pending_bytes -= size
                    PENDING_OPERATIONS.dec()
                    PENDING_BYTES.dec(size)
                    self.condition.notify_all()
        except BaseException as exc:
            async with self.condition:
                self.failure = exc
                self.condition.notify_all()
            observe_duration("recording_worker_queue", 0, self.trace_id, result="error")

    async def video(self, name, index, rgb, pts_us=None):
        await self._enqueue(
            "video",
            (name, index, rgb) if pts_us is None else (name, index, rgb, pts_us),
        )

    async def encoded_video(self, index, encoded, *, name="ui_rendered", pts_us=None):
        await self.video(name, index, dict(encoded), pts_us)

    async def write_metadata(self, frame, prompt):
        await self._enqueue(
            "metadata", (copy.deepcopy(replace(frame, rgb=b"")), prompt)
        )

    async def configure(self):
        await self._enqueue("configure", (copy.deepcopy(self.manifest),))

    async def write_source_state(self, message):
        await self._enqueue("source_state", (copy.deepcopy(message),))

    async def add_references(self, references, *, wait=False):
        sequence = await self._enqueue("references", (copy.deepcopy(references),))
        if wait:
            async with self.condition:
                await self.condition.wait_for(
                    lambda: self.failure or sequence not in self.pending
                )
                if self.failure:
                    raise RuntimeError(
                        "Reference recording could not be confirmed"
                    ) from self.failure
        return reference_descriptors(references)

    async def finish(self, outcome):
        if self.closed:
            return
        async with self.condition:
            self.accepting = False
            self.condition.notify_all()
        try:
            if not self.task.done():
                put = asyncio.create_task(self.queue.put(None))
                try:
                    await asyncio.wait(
                        (put, self.task), return_when=asyncio.FIRST_COMPLETED
                    )
                    if not self.task.done():
                        await self.task
                finally:
                    if not put.done():
                        put.cancel()
                    await asyncio.gather(put, return_exceptions=True)
            await self.task
            trace = self.trace_snapshot() if self.trace_snapshot else []
            self.manifest = await self._call(
                _finish,
                "recording_failed" if self.failure else outcome,
                trace,
                self.statistics(),
            )
        except Exception as exc:
            self.failure = exc
            # File recovery also has a process owner; do not put disk writes on
            # the gateway event loop when the recorder itself has died.
            recovery = process_pool(self.cpu)
            try:
                await asyncio.get_running_loop().run_in_executor(
                    recovery, _failed_manifest, self.directory, self.manifest
                )
            finally:
                await asyncio.to_thread(recovery.shutdown, wait=True)
            raise
        finally:
            self.closed = True
            await asyncio.to_thread(self.pool.shutdown, wait=True, cancel_futures=True)
            PENDING_OPERATIONS.dec(len(self.pending))
            PENDING_BYTES.dec(self.pending_bytes)
            self.unprocessed_operations = len(self.pending)
            self.pending.clear()
            self.pending_bytes = 0
            while not self.queue.empty():
                self.queue.get_nowait()
            if self.release:
                self.release()
