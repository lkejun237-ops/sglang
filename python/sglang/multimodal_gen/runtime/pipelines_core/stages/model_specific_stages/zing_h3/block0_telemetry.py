# SPDX-License-Identifier: Apache-2.0
"""Bounded preparation spans; host durations never imply GPU completion."""

import time
from contextlib import contextmanager


class Block0Telemetry:
    def __init__(self, *, clock=time.perf_counter):
        self.clock = clock
        self.started = clock()
        self.stages = {}
        self.cache_result = "miss"
        self.miss_reason = None
        self.lookup_details = {}
        self.finished = None
        self.cuda = {}

    def finish(self):
        if self.finished is None:
            self.finished = self.clock()

    @contextmanager
    def measure(self, name, *, cuda=False):
        start = self.clock()
        status = "executed"
        try:
            yield
        except BaseException:
            status = "error"
            raise
        finally:
            self.record(name, (self.clock() - start) * 1000, status=status)

    def record(self, name, value, *, status="executed"):
        if len(self.stages) >= 32 and name not in self.stages:
            return
        previous = self.stages.get(name, {})
        self.stages[name] = {
            "host_ms": (previous.get("host_ms") or 0.0) + max(0.0, float(value)),
            "status": status,
        }

    def skip(self, *names):
        for name in names:
            if name not in self.stages:
                self.stages[name] = {"host_ms": None, "status": "skipped"}

    def snapshot(self):
        return {
            "timing_schema_version": 1,
            "cache_result": self.cache_result,
            "miss_reason": self.miss_reason,
            "lookup": dict(self.lookup_details),
            "elapsed_host_ms": (
                (self.clock() if self.finished is None else self.finished)
                - self.started
            )
            * 1000,
            "stages": {
                **self.lookup_details.get("stages", {}),
                **{name: dict(value) for name, value in self.stages.items()},
            },
            "cuda_ms": dict(self.cuda),
        }
