# SPDX-License-Identifier: Apache-2.0
"""Opt-in finite Zing stage spans; CUDA results are collected asynchronously.

Host return and GPU tensor handoff do not imply device completion. CUDA events
cover only the current device/stream and are never mapped onto the host clock.
"""

import asyncio
import atexit
import hashlib
import json
import os
import queue
import tempfile
import threading
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path

_WRITER = None
_LOCK = threading.Lock()
_LOCAL_DIRECTORY = None
_DEPLOYMENT_ACTIVE = False


def new_events():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requested without CUDA")
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def resolve_cuda(row, start, end):
    if start is None or end is None:
        row.setdefault("cuda_stream_ms", None)
        return True
    try:
        if not end.query():
            return False
        row["cuda_stream_ms"] = float(start.elapsed_time(end))
        row["cuda_status"] = "complete"
    except Exception as exc:
        row["cuda_stream_ms"] = None
        row["cuda_status"] = "error"
        row["cuda_error_type"] = type(exc).__name__
    return True


def publish_health(path, value):
    # Readers and process termination must never expose a truncated snapshot.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


class Span:
    def __init__(self, trace, stage, cuda, fields):
        self.trace = trace
        self.start = self.end = None
        self.sent = False
        trace.sequence += 1
        self.row = {
            **trace.identity,
            "span_id": f"{trace.identity.get('rank', 0)}:{trace.identity['dispatch_id']}:{stage}:{trace.sequence}",
            "stage": stage,
            "enqueue_ns": None,
            "host_enter_ns": trace.clock(),
            "host_return_ns": None,
            "host_handoff_ns": None,
            "input_ids": [],
            "output_ids": [],
            **fields,
            "status": "ok",
            "cuda_status": "pending" if cuda else "not_requested",
            "cuda_scope": "current_device_current_stream" if cuda else None,
        }
        if cuda:
            try:
                self.start, self.end = trace.event_factory()
                self.start.record()
            except Exception as exc:
                self.start = self.end = None
                self.row.update(cuda_status="error", cuda_error_type=type(exc).__name__)

    def finish(self, error=None):
        self.row["host_return_ns"] = self.trace.clock()
        self.row["host_call_ms"] = (
            self.row["host_return_ns"] - self.row["host_enter_ns"]
        ) / 1e6
        if self.end is not None:
            try:
                self.end.record()
            except Exception as exc:
                self.start = self.end = None
                self.row.update(cuda_status="error", cuda_error_type=type(exc).__name__)
        if error is not None:
            self.row.update(status="error", error_type=type(error).__name__)
            self.emit()

    def handoff(self, **fields):
        if self.sent:
            return
        self.row.update(fields)
        self.row["host_handoff_ns"] = self.trace.clock()
        self.row["handoff_meaning"] = (
            "caller_accepted_tensor_or_metadata; not_GPU_completion"
        )
        self.emit()

    def emit(self):
        if not self.sent:
            self.sent = True
            self.trace.emitted += 1
            self.trace.submit((self.row, self.start, self.end))


class StageTrace:
    def __init__(
        self,
        request_id,
        dispatch_id,
        submit,
        *,
        clock=time.monotonic_ns,
        event_factory=new_events,
        measure_cuda=False,
        identity=None,
    ):
        self.identity = {
            "request_id": request_id,
            "dispatch_id": dispatch_id,
            **(identity or {}),
        }
        self.submit, self.clock, self.event_factory = submit, clock, event_factory
        self.measure_cuda = measure_cuda
        self.sequence = self.emitted = 0

    def finish(self):
        self.submit(
            (
                {
                    **self.identity,
                    "kind": "dispatch_complete",
                    "span_count": self.emitted,
                },
                None,
                None,
            )
        )

    @contextmanager
    def span(self, stage, *, cuda=False, **fields):
        span = Span(self, stage, cuda, fields)
        try:
            yield span
        except BaseException as exc:
            span.finish(exc)
            raise
        else:
            span.finish()


def session_file_tag(request_id):
    return hashlib.sha256(str(request_id).encode()).hexdigest()


def remove_profile_files(directory, request_id):
    """Remove only this completed request; worker health files stay bounded."""
    tag = session_file_tag(request_id)
    for path in Path(directory).glob(f"stage-timing-pid*-session{tag}.jsonl"):
        path.unlink(missing_ok=True)


class Writer:
    """Bounded nonblocking producer, bounded delayed events, append-only output."""

    def __init__(self, directory, *, capacity=512, event_timeout=30):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"stage-timing-pid{os.getpid()}.jsonl"
        self.health_path = directory / f"stage-timing-pid{os.getpid()}-health.json"
        self.queue = queue.Queue(maxsize=capacity)
        self.capacity, self.event_timeout = capacity, event_timeout
        self.dropped = self.written = 0
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self.run, daemon=True, name="zing-stage-timing"
        )
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)

    def submit(self, item):
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def run(self):
        pending = []
        profile = None
        try:
            while not self.stop.is_set() or pending or not self.queue.empty():
                if len(pending) < self.capacity:
                    try:
                        item = self.queue.get(timeout=0.01)
                        row = item[0]
                        if row.get("request_id") is not None and (
                            profile is None
                            or profile["request_id"] != row["request_id"]
                            or profile.get("generation_id") != row.get("generation_id")
                        ):
                            profile = {
                                "request_id": row["request_id"],
                                "generation_id": row.get("generation_id"),
                                "rank": row.get("rank"),
                                "pid": os.getpid(),
                                "worker_role": row.get("worker_role"),
                                "dispatches": {},
                            }
                        pending.append((time.monotonic(), item))
                    except queue.Empty:
                        if not pending:
                            continue
                else:
                    self.stop.wait(0.01)
                retained = []
                for added, (row, start, end) in pending:
                    if row.get("kind") == "dispatch_complete":
                        # A marker is publishable only after earlier spans from
                        # this dispatch have resolved, including CUDA events.
                        if any(
                            r.get("request_id") == row["request_id"]
                            and r.get("dispatch_id") == row["dispatch_id"]
                            for _, (r, _, _) in retained
                        ):
                            retained.append((added, (row, start, end)))
                        elif (
                            profile
                            and profile["request_id"] == row["request_id"]
                            and profile.get("generation_id") == row.get("generation_id")
                        ):
                            profile["dispatches"][row["dispatch_id"]] = row[
                                "span_count"
                            ]
                        continue
                    if not resolve_cuda(row, start, end):
                        if time.monotonic() - added < self.event_timeout:
                            retained.append((added, (row, start, end)))
                            continue
                        row.update(cuda_status="timeout", cuda_stream_ms=None)
                    path = self.path
                    if row.get("request_id") is not None:
                        tag = session_file_tag(row["request_id"])
                        path = self.path.with_name(
                            f"{self.path.stem}-session{tag}.jsonl"
                        )
                    with path.open("a", encoding="utf-8") as file:
                        file.write(json.dumps(row, allow_nan=False) + "\n")
                    self.written += 1
                pending = retained
                publish_health(
                    self.health_path,
                    {
                        "written": self.written,
                        "dropped": self.dropped,
                        "pending": len(pending),
                        "queued": self.queue.qsize(),
                        "error": None,
                        "profile": profile,
                    },
                )
        except Exception as exc:
            self.error = type(exc).__name__
            # Collection failure must be visible without replacing model output.
            import logging

            logging.getLogger(__name__).exception("Zing stage timing collector failed")


def get_writer(directory):
    global _WRITER
    with _LOCK:
        directory = Path(directory).resolve()
        if _WRITER is not None and _WRITER.path.parent != directory:
            _WRITER.close()
            _WRITER = None
        if _WRITER is None:
            _WRITER = Writer(directory)
        return _WRITER


@contextmanager
def deployment_timing_directory():
    """One directory per deployment, inherited by spawned SP workers."""
    global _DEPLOYMENT_ACTIVE
    if _DEPLOYMENT_ACTIVE:
        yield timing_directory()
        return
    previous = os.environ.get("SGLANG_ZING_STAGE_TIMING_DIR")
    base = Path(previous).resolve() if previous else None
    if base:
        base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="zing-stage-timing-", dir=base
    ) as directory:
        _DEPLOYMENT_ACTIVE = True
        os.environ["SGLANG_ZING_STAGE_TIMING_DIR"] = directory
        try:
            yield Path(directory)
        finally:
            _DEPLOYMENT_ACTIVE = False
            global _WRITER
            if _WRITER is not None and _WRITER.path.parent == Path(directory):
                _WRITER.close()
                _WRITER = None
            if previous is None:
                os.environ.pop("SGLANG_ZING_STAGE_TIMING_DIR", None)
            else:
                os.environ["SGLANG_ZING_STAGE_TIMING_DIR"] = previous


def timing_directory():
    """Lazy fallback for programmatic callers outside the serving launcher."""
    global _LOCAL_DIRECTORY
    directory = os.environ.get("SGLANG_ZING_STAGE_TIMING_DIR")
    if directory:
        return Path(directory).resolve()
    if _LOCAL_DIRECTORY is None:
        _LOCAL_DIRECTORY = tempfile.TemporaryDirectory(prefix="zing-stage-timing-")
        atexit.register(_LOCAL_DIRECTORY.cleanup)
    return Path(_LOCAL_DIRECTORY.name)


def trace_for_ids(
    request_id, dispatch_id, *, enabled=False, directory=None, identity=None
):
    """Request-scoped Host trace, also usable by the dedicated Encoder."""
    if not enabled:
        return None
    directory = (
        Path(directory).resolve() if directory is not None else timing_directory()
    )
    writer = get_writer(directory)
    import torch
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    boot = Path("/proc/sys/kernel/random/boot_id")
    trace_identity = {
        "pid": os.getpid(),
        "rank": rank,
        "generation_id": None,
        "device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "host": os.uname().nodename,
        "clock_domain": f"CLOCK_MONOTONIC:{boot.read_text().strip() if boot.exists() else 'unknown'}",
        "mode": "host",
        "dispatch_id_status": "present" if dispatch_id else "missing",
        **(identity or {}),
    }
    trace = StageTrace(request_id, dispatch_id, writer.submit, identity=trace_identity)
    trace.directory = str(directory)
    return trace


def trace_for_batch(batch):
    return trace_for_ids(
        getattr(batch, "realtime_session_id", None),
        getattr(batch, "request_id", None),
        enabled=bool(
            (getattr(batch, "extra", None) or {}).get("return_stage_timing", False)
        ),
        identity={"generation_id": getattr(batch, "realtime_generation_id", None)},
    )


def stage_span(trace, stage, **fields):
    return trace.span(stage, **fields) if trace is not None else nullcontext(None)


def handoff(span, **fields):
    if span is not None:
        span.handoff(**fields)


def _profile_snapshots(directory, request_id, generation_id):
    snapshots = {}
    for path in Path(directory).glob("stage-timing-pid*-health.json"):
        value = json.loads(path.read_text()).get("profile")
        if (
            value
            and value.get("worker_role") != "dedicated_source_encoder"
            and value["request_id"] == request_id
            and (generation_id is None or value.get("generation_id") == generation_id)
        ):
            snapshots[value["rank"]] = value
    return snapshots


async def collect_profile(
    directory, request_id, ranks, dispatch_count, *, generation_id=None, timeout=35
):
    """Read this session after each rank's asynchronous writer has drained it."""
    started = time.monotonic()
    snapshots = {}
    error = None
    while True:
        try:
            snapshots = await asyncio.to_thread(
                _profile_snapshots, directory, request_id, generation_id
            )
        except (OSError, ValueError):
            error = "Stage collector snapshots could not be read."
        complete = all(
            rank in snapshots and len(snapshots[rank]["dispatches"]) == dispatch_count
            for rank in ranks
        )
        if complete or time.monotonic() - started >= timeout:
            break
        await asyncio.sleep(0.05)
    stages = []
    try:
        for path in Path(directory).glob("stage-timing-pid*.jsonl"):
            with path.open() as source:
                for line in source:
                    row = json.loads(line)
                    if (
                        row.get("request_id") == request_id
                        and row.get("rank") in ranks
                        and (
                            generation_id is None
                            or row.get("generation_id") == generation_id
                        )
                    ):
                        stages.append(row)
    except (OSError, ValueError):
        error = "Stage records could not be read."
    counts = Counter((row["rank"], row["dispatch_id"]) for row in stages)
    health = []
    dispatch_sets = []
    for rank in ranks:
        snapshot = snapshots.get(rank, {})
        dispatches = snapshot.get("dispatches", {})
        dispatch_sets.append(set(dispatches))
        missing = sum(
            max(0, expected - counts[rank, dispatch])
            for dispatch, expected in dispatches.items()
        )
        complete = (
            complete
            and not missing
            and all(
                counts[rank, dispatch] == expected
                for dispatch, expected in dispatches.items()
            )
        )
        health.append(
            {
                "rank": rank,
                "pid": snapshot.get("pid"),
                "request_id": request_id,
                "written": sum(n for (r, _), n in counts.items() if r == rank),
                "dropped": missing,
                "pending": max(0, dispatch_count - len(dispatches)),
                "queued": 0,
                "error": error,
            }
        )
    complete = (
        complete and all(s == dispatch_sets[0] for s in dispatch_sets) and not error
    )
    complete = complete and all(
        row.get("cuda_status") not in {"pending", "error", "timeout"}
        and row.get("status", "ok") == "ok"
        for row in stages
    )
    return {
        "type": "profile_result",
        "session_id": request_id,
        "generation_id": generation_id,
        "status": "complete" if complete else "incomplete",
        "expected_ranks": ranks,
        "missing_ranks": [r for r in ranks if r not in snapshots],
        "stages": stages,
        "health": health,
        "wait_seconds": time.monotonic() - started,
    }


async def collect_continuous_profile(directory, request_id, ranks, *, timeout=35):
    """Read the completed single-active source session, including its Encoder."""
    started = time.monotonic()
    tag = session_file_tag(request_id)
    stages, health = [], []
    while True:
        stages, health = [], []
        readable = True
        try:
            for path in Path(directory).glob(f"stage-timing-pid*-session{tag}.jsonl"):
                stages.extend(
                    json.loads(line) for line in path.read_text().splitlines()
                )
                stem = path.name.split("-session", 1)[0]
                health.append(
                    {
                        "file": stem + "-health.json",
                        **json.loads(path.with_name(stem + "-health.json").read_text()),
                    }
                )
        except (OSError, ValueError):
            readable = False
            stages, health = [], []
        completed = {
            row["rank"]
            for row in stages
            if row.get("component") == "continuous_source_complete"
        }
        complete = (
            readable
            and set(ranks) <= completed
            and all(
                not any(h.get(k) for k in ("pending", "queued", "error"))
                for h in health
            )
            and all(row.get("status") == "ok" for row in stages)
        )
        if complete or time.monotonic() - started >= timeout:
            break
        await asyncio.sleep(0.05)
    return {
        "type": "profile_result",
        "session_id": request_id,
        "status": "complete" if complete else "incomplete",
        "expected_ranks": ranks,
        "stages": stages,
        "health": health,
        "wait_seconds": time.monotonic() - started,
    }


async def discard_profile(directory, request_id, *, timeout=2):
    """Best-effort cleanup after workers released an aborted session."""
    deadline = time.monotonic() + timeout
    tag = session_file_tag(request_id)
    previous = None
    while True:
        files = list(Path(directory).glob(f"stage-timing-pid*-session{tag}.jsonl"))
        state = [(str(p), p.stat().st_size) for p in files]
        healthy = True
        for path in Path(directory).glob("*-health.json"):
            try:
                h = json.loads(path.read_text())
                healthy = healthy and not h.get("pending") and not h.get("queued")
            except (OSError, ValueError):
                healthy = False
        if (healthy and previous == state) or time.monotonic() >= deadline:
            break
        previous = state
        await asyncio.sleep(0.05)
    remove_profile_files(directory, request_id)
