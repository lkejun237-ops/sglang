# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar

from sglang.multimodal_gen.runtime.realtime.critical_path_metrics import (
    observe_stage_seconds,
)
from sglang.multimodal_gen.runtime.utils.realtime_trace import (
    emit_realtime_trace_payload,
)

logger = logging.getLogger(__name__)
_captured = ContextVar("whitebox_stages", default=None)


@contextmanager
def capture_stages():
    spans = []
    token = _captured.set(spans)
    try:
        yield spans
    finally:
        _captured.reset(token)


@contextmanager
def stage(name: str, trace_id: str, **fields):
    start = time.perf_counter()
    result = "success"
    try:
        yield
    except StopAsyncIteration:
        raise
    except BaseException:
        result = "error"
        raise
    finally:
        elapsed = time.perf_counter() - start
        observe_duration(name, elapsed * 1000, trace_id, result=result, **fields)


def observe_duration(
    name, milliseconds, trace_id, *, result="success", service=None, **fields
):
    service = service or (
        "room_runtime" if os.environ.get("WHITEBOX_ROLE") == "room" else "gateway"
    )
    captured = _captured.get()
    if captured is not None and len(captured) < 32:
        captured.append(dict(stage=name, duration_ms=milliseconds, result=result))
    observe_stage_seconds(
        "whitebox_" + name,
        milliseconds / 1000,
        service=service,
        model=os.environ.get("WORLD_MODEL_METRIC_MODEL", "zing_h3"),
        result=result,
        scope="request",
    )
    emit_realtime_trace_payload(
        logger,
        {
            "event": "whitebox." + name,
            "trace_id": trace_id,
            "server_epoch_ms": int(time.time() * 1000),
            "duration_ms": milliseconds,
            "result": result,
            **fields,
        },
    )
