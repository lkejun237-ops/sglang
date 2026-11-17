# SPDX-License-Identifier: Apache-2.0
"""Small per-condition timings for serving; never wait for a CUDA event.

Host intervals and device stream intervals are separate, potentially nested
measurements. A pending snapshot can be followed by a resolved snapshot with the
same report_id. Owners drain on their existing output/dispatch boundaries.
"""

import os
import socket
import time
import uuid
from contextlib import contextmanager, nullcontext

from .zing_stage_timing import new_events, resolve_cuda


def validate_timing_update(message, revision):
    """Validate a single owner's next revision before any role mutates state."""
    if (
        type(message.get("enabled")) is not bool
        or type(message.get("timing_revision")) is not int
        or message["timing_revision"] != revision + 1
    ):
        raise ValueError("Invalid timing mode or revision")


class InferenceTiming:
    def __init__(
        self,
        role,
        *,
        event_id=None,
        prompt_version=None,
        device=None,
        clock=time.perf_counter,
        event_factory=new_events,
        allowed_stages=None,
        **identity,
    ):
        self.clock, self.event_factory = clock, event_factory
        self.device = device
        self.allowed_stages = allowed_stages
        self.started = clock()
        self.identity = dict(
            schema_version=1,
            report_id=uuid.uuid4().hex,
            role=role,
            event_id=event_id,
            prompt_version=prompt_version,
            clock_domain=f"{socket.gethostname()}:{os.getpid()}:perf_counter",
            **identity,
        )
        self.stages = []
        self.pending = []
        self.revision = 0
        self.finished = False

    @contextmanager
    def measure(self, name, *, cuda=False):
        if self.allowed_stages is not None and name not in self.allowed_stages:
            yield
            return
        row = dict(
            stage=name,
            start_ms=(self.clock() - self.started) * 1000,
            host_ms=None,
            device_elapsed_ms=None,
            status="executed",
            device_status="not_requested",
        )
        start = end = None
        use_cuda = cuda and getattr(self.device, "type", None) == "cuda"
        if cuda and self.device is None:
            import torch

            use_cuda = torch.cuda.is_available()
        if use_cuda:
            try:
                start, end = self.event_factory()
                start.record()
                row["device_status"] = "pending"
            except Exception:
                start = end = None
                row["device_status"] = "error"
        began = self.clock()
        try:
            yield
        except BaseException:
            row["status"] = "error"
            raise
        finally:
            row["host_ms"] = (self.clock() - began) * 1000
            if end is not None:
                try:
                    end.record()
                    self.pending.append((row, start, end))
                except Exception:
                    row["device_status"] = "error"
            self.stages.append(row)
            self.revision += 1

    def record(self, name, value, *, status="executed"):
        if self.allowed_stages is not None and name not in self.allowed_stages:
            return
        self.stages.append(
            dict(
                stage=name,
                start_ms=None,
                host_ms=value,
                device_elapsed_ms=None,
                status=status,
                device_status="not_requested",
            )
        )
        self.revision += 1

    def skip(self, *names):
        for name in names:
            self.record(name, None, status="skipped")

    def snapshot(self, *, final=False):
        retained = []
        for row, start, end in self.pending:
            resolved = {}
            if resolve_cuda(resolved, start, end):
                row["device_elapsed_ms"] = resolved.get("cuda_stream_ms")
                row["device_status"] = resolved.get("cuda_status", "error")
                self.revision += 1
            elif final or self.clock() - self.started >= 30:
                row["device_status"] = "unresolved"
                self.revision += 1
            else:
                retained.append((row, start, end))
        self.pending = retained
        status = "pending" if retained or not self.finished else "complete"
        if any(
            row["device_status"] in {"error", "unresolved"} or row["status"] == "error"
            for row in self.stages
        ):
            status = "incomplete"
        return dict(
            self.identity,
            revision=self.revision,
            status=status,
            stages=[dict(row) for row in self.stages],
        )


class TimingBuffer:
    """Owned by a role, bounded even if downstream stops requesting output."""

    def __init__(self, capacity=64):
        self.capacity = capacity
        self.pending = []
        self.terminal = []

    def add(self, timing):
        timing.finished = True
        if len(self.pending) >= self.capacity:
            old, _ = self.pending.pop(0)
            report = old.snapshot(final=True)
            report["status"] = "incomplete"
            self.terminal.append(report)
            # Each dispatch drains this list; the owner must not enqueue unboundedly.
            self.terminal = self.terminal[-self.capacity :]
        self.pending.append((timing, -1))

    def drain(self, *, final=False):
        reports, self.terminal = self.terminal, []
        retained = []
        for timing, last_revision in self.pending:
            report = timing.snapshot(final=final)
            if report["revision"] != last_revision:
                reports.append(report)
            if timing.pending:
                retained.append((timing, report["revision"]))
        self.pending = retained
        return reports


def measure(timing, name, *, cuda=False):
    return timing.measure(name, cuda=cuda) if timing is not None else nullcontext()
