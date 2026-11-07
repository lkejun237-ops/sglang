# SPDX-License-Identifier: Apache-2.0
"""Observe host GC pauses without changing collection or CUDA scheduling."""

import gc
import json
import os
import socket
import sys
import time
from pathlib import Path


class GCPauseObserver:
    def __init__(self, *, clock=time.perf_counter_ns):
        self.clock = clock
        self.events = []
        self.starts = {}

    def callback(self, phase, info):
        generation = info["generation"]
        now = self.clock()
        if phase == "start":
            self.starts[generation] = now
        elif phase == "stop":
            started = self.starts.pop(generation, None)
            if started is not None:
                self.events.append(
                    dict(
                        generation=generation,
                        start_ns=started,
                        end_ns=now,
                        seconds=(now - started) / 1e9,
                        collected=info["collected"],
                        uncollectable=info["uncollectable"],
                    )
                )

    def __enter__(self):
        gc.callbacks.append(self.callback)
        return self

    def __exit__(self, *exc):
        gc.callbacks.remove(self.callback)


def install_passive_gc_diagnostic():
    """Opt-in gen2 records; the switch never changes GC or CUDA policy."""
    flag = Path(os.environ["STALL_GC_DIAGNOSTIC_SWITCH"])

    class Observer(GCPauseObserver):
        def __init__(self):
            super().__init__()
            self.unix_start = None

        def callback(self, phase, info):
            if info["generation"] != 2:
                return
            if phase == "start":
                if not flag.exists():
                    return
                self.unix_start = time.time_ns()
                super().callback(phase, info)
            elif phase == "stop" and 2 in self.starts:
                super().callback(phase, info)
                event = self.events.pop()
                ended = time.time_ns()
                torch = sys.modules.get("torch")
                dist = getattr(torch, "distributed", None)
                event.update(
                    start_unix_ns=self.unix_start,
                    end_unix_ns=ended,
                    role_label=os.environ.get("STALL_ROLE_GC_LABEL", "unknown-role"),
                    pid=os.getpid(),
                    hostname=socket.gethostname(),
                    rank=dist.get_rank()
                    if dist is not None and dist.is_initialized()
                    else None,
                    automatic_gc_enabled=gc.isenabled(),
                    freeze_count=gc.get_freeze_count(),
                    scope="Passive gen2 only; automatic versus explicit collection unknown; output logging starts after measured end",
                )
                print("STALL_ROLE_GC_JSON " + json.dumps(event), flush=True)

    observer = Observer()
    gc.callbacks.append(observer.callback)
    return lambda: gc.callbacks.remove(observer.callback)
