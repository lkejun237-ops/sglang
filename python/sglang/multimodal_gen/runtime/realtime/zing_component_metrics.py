# SPDX-License-Identifier: Apache-2.0
"""Zing component timings, with bounded, non-blocking CUDA event collection.

Host durations measure the call, CUDA durations measure its current stream.
Neither includes input starvation, transport or playback. Frame counters use
pixel frames (sampled media frames for Qwen), never latent frames or calls.
"""

import os
import queue
import threading
import time
from contextlib import contextmanager

from prometheus_client import Counter, Histogram

from .critical_path_metrics import CRITICAL_PATH_BUCKETS, lane_label, service_label

LABELS = (
    "component",
    "operation",
    "service",
    "lane",
    "rank",
    "resolution",
    "measurement",
)
DURATION = Histogram(
    "sglang_zing_component_duration_seconds",
    "Successful component call duration; host call or current CUDA stream.",
    LABELS,
    buckets=CRITICAL_PATH_BUCKETS,
)
FRAMES = Counter(
    "sglang_zing_component_frames_total",
    "Compute frames paired with timing samples; Decoder includes padding.",
    LABELS,
)
COMPLETED_FRAMES = Counter(
    "sglang_zing_component_completed_frames_total",
    "Successfully processed valid pixel frames, independent of timing samples.",
    LABELS[:-1],
)
DROPPED = Counter(
    "sglang_zing_component_timing_dropped_total",
    "CUDA timing samples lost without blocking model execution.",
    ("component", "reason"),
)
ERRORS = Counter(
    "sglang_zing_component_errors_total",
    "Failed component calls, excluded from successful duration and frame totals.",
    ("component",),
)


def new_events():
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def observe(labels, measurement, duration, frames):
    DURATION.labels(*labels, measurement).observe(duration)
    FRAMES.labels(*labels, measurement).inc(frames)


def record_completed_frames(component, frames, *, width, height, operation="forward"):
    """Report at a caller-owned valid-frame boundary, including Decoder trim."""
    if os.environ.get("SGLANG_ZING_COMPONENT_METRICS", "1") == "0" or frames <= 0:
        return
    try:
        COMPLETED_FRAMES.labels(
            component,
            operation,
            service_label(component),
            lane_label(),
            os.environ.get("RANK", "0"),
            f"{width}x{height}",
        ).inc(frames)
    except Exception:
        # Observability must not change the success of frame processing.
        return


class CudaCollector:
    """No tensors, blocking CUDA calls or unbounded history."""

    def __init__(self):
        self.pending = queue.Queue(maxsize=128)
        self.thread = threading.Thread(
            target=self.run, name="zing-component-metrics", daemon=True
        )
        self.thread.start()

    def submit(self, item):
        try:
            self.pending.put_nowait(item)
        except queue.Full:
            DROPPED.labels(item[0][0], "queue_full").inc()

    def collect(self, item):
        labels, start, end, frames, submitted = item
        try:
            if end.query():
                observe(labels, "cuda", start.elapsed_time(end) / 1000, frames)
            elif time.monotonic() - submitted > 120:
                DROPPED.labels(labels[0], "timeout").inc()
            else:
                return False
        except Exception:
            DROPPED.labels(labels[0], "event_error").inc()
        return True

    def run(self):
        while True:
            item = self.pending.get()
            if not self.collect(item):
                self.submit(item)
                # Yield even when an event cannot complete; never spin on CUDA.
                time.sleep(0.02)


_COLLECTOR = None
_LOCK = threading.Lock()


def submit_cuda(item):
    global _COLLECTOR
    with _LOCK:
        if _COLLECTOR is None:
            _COLLECTOR = CudaCollector()
        collector = _COLLECTOR
    collector.submit(item)


class ComponentTimer:
    def __init__(
        self,
        component,
        *,
        frames=0,
        width=None,
        height=None,
        operation="forward",
        completed_frames=None,
    ):
        self.component, self.frames = component, frames
        self.completed_frames = completed_frames
        self.events = None
        self.finished = False
        self.enabled = os.environ.get("SGLANG_ZING_COMPONENT_METRICS", "1") != "0"
        if not self.enabled:
            return
        self.resolution = "conditioning" if component == "qwen_vlm" else "unknown"
        if width and height:
            self.resolution = f"{width}x{height}"
        self.labels = (
            component,
            operation,
            service_label(component),
            lane_label(),
            os.environ.get("RANK", "0"),
        )
        self.started = time.perf_counter()
        try:
            self.events = new_events()
            if self.events is not None:
                self.events[0].record()
        except Exception:
            self.events = None
            DROPPED.labels(component, "event_error").inc()

    def finish(self, error=None):
        if self.finished or not self.enabled:
            return
        self.finished = True
        try:
            if error is not None:
                ERRORS.labels(self.component).inc()
                return
            if self.events is not None:
                try:
                    self.events[1].record()
                except Exception:
                    self.events = None
                    DROPPED.labels(self.component, "event_error").inc()
            labels = (*self.labels, self.resolution)
            observe(labels, "host", time.perf_counter() - self.started, self.frames)
            completed = (
                self.frames if self.completed_frames is None else self.completed_frames
            )
            if self.completed_frames is None or completed > 0:
                COMPLETED_FRAMES.labels(*labels).inc(completed)
            if self.events is not None:
                submit_cuda((labels, *self.events, self.frames, time.monotonic()))
        except Exception:
            # Monitoring must never turn a successful model call into a failure.
            DROPPED.labels(self.component, "event_error").inc()


@contextmanager
def component_timer(component, **kwargs):
    timer = ComponentTimer(component, **kwargs)
    try:
        yield timer
    except BaseException as error:
        timer.finish(error)
        raise
    else:
        timer.finish()
