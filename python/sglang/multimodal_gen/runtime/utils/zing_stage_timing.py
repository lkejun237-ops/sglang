# SPDX-License-Identifier: Apache-2.0
"""Opt-in finite Zing stage spans; CUDA results are collected asynchronously.

Host return and GPU tensor handoff do not imply device completion. CUDA events
cover only the current device/stream and are never mapped onto the host clock.
"""

import json
import os
import queue
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

_WRITER = None
_LOCK = threading.Lock()


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
        self.sequence = 0

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

    def submit(self, item):
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def run(self):
        pending = []
        try:
            with self.path.open("a", encoding="utf-8") as file:
                while not self.stop.is_set() or pending or not self.queue.empty():
                    if len(pending) < self.capacity:
                        try:
                            item = self.queue.get(timeout=0.01)
                            pending.append((time.monotonic(), item))
                        except queue.Empty:
                            pass
                    else:
                        self.stop.wait(0.01)
                    retained = []
                    for added, (row, start, end) in pending:
                        if not resolve_cuda(row, start, end):
                            if time.monotonic() - added < self.event_timeout:
                                retained.append((added, (row, start, end)))
                                continue
                            row.update(cuda_status="timeout", cuda_stream_ms=None)
                        file.write(json.dumps(row, allow_nan=False) + "\n")
                        self.written += 1
                    pending = retained
                    file.flush()
                    publish_health(
                        self.health_path,
                        {
                            "written": self.written,
                            "dropped": self.dropped,
                            "pending": len(pending),
                            "queued": self.queue.qsize(),
                            "error": None,
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
        if _WRITER is None:
            _WRITER = Writer(Path(directory))
        return _WRITER


def trace_for_batch(batch):
    mode = os.environ.get("SGLANG_ZING_STAGE_TIMING", "off")
    if mode == "off":
        return None
    if mode not in {"host", "cuda"}:
        raise ValueError("Zing stage timing mode must be off, host, or cuda")
    directory = os.environ.get("SGLANG_ZING_STAGE_TIMING_DIR")
    if not directory:
        raise ValueError("Zing stage timing requires an output directory")
    import torch
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    boot = Path("/proc/sys/kernel/random/boot_id")
    identity = {
        "pid": os.getpid(),
        "rank": rank,
        "device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "host": os.uname().nodename,
        "clock_domain": f"CLOCK_MONOTONIC:{boot.read_text().strip() if boot.exists() else 'unknown'}",
        "mode": mode,
        "dispatch_id_status": "present"
        if getattr(batch, "request_id", None)
        else "missing",
    }
    return StageTrace(
        batch.realtime_session_id,
        getattr(batch, "request_id", None),
        get_writer(directory).submit,
        measure_cuda=mode == "cuda",
        identity=identity,
    )


def stage_span(trace, stage, **fields):
    return trace.span(stage, **fields) if trace is not None else nullcontext(None)


def handoff(span, **fields):
    if span is not None:
        span.handoff(**fields)
