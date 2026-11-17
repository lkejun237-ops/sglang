# SPDX-License-Identifier: Apache-2.0
"""Small, request-local metadata for cache events, including background writes."""

import json
import logging
import os
import socket
import time
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
_context = ContextVar("zing_cache_log_context", default=None)
_host = socket.gethostname()
_fields = (
    "operation_id",
    "session_id",
    "generation_id",
    "prompt_version",
    "event_id",
    "game_id",
    "group_id",
)


def log_context(**fields):
    value = dict(_context.get() or {})
    value.update(
        {k: v for k, v in fields.items() if k in _fields and isinstance(v, (str, int))}
    )
    if "operation_id" not in value:
        value["operation_id"] = uuid4().hex
    return value


@contextmanager
def cache_scope(context=None, **fields):
    value = log_context(**{**(context or {}), **fields})
    token = _context.set(value)
    try:
        yield value
    finally:
        _context.reset(token)


def batch_context(batch):
    extra = getattr(batch, "extra", {}) or {}
    command = extra.get("zing_prefill", extra.get("zing_source_command", {}))
    session = getattr(batch, "session", None)
    return log_context(
        operation_id=command.get("operation_id") or getattr(batch, "request_id", None),
        session_id=getattr(session, "session_id", getattr(session, "id", None)),
        generation_id=getattr(session, "generation_id", None),
        prompt_version=getattr(batch, "realtime_prompt_version", None),
        event_id=getattr(batch, "realtime_event_id", None),
        game_id=command.get("game_id"),
        group_id=command.get("group_id"),
    )


def cache_event(
    event,
    result,
    *,
    cache_key=None,
    reason=None,
    rank=None,
    started=None,
    level=logging.INFO,
    context=None,
    **details,
):
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    # Only SP rank zero emits normal INFO; warnings retain the failing rank.
    if level == logging.INFO and rank != 0:
        level = logging.DEBUG
    if not logger.isEnabledFor(level):
        return
    value = log_context(**(context or {}))
    value.update(
        instance=f"{_host}:{os.getpid()}",
        rank=rank,
        event=event,
        cache_key=cache_key,
        result=result,
        reason=reason,
        elapsed_host_ms=None
        if started is None
        else (time.perf_counter() - started) * 1000,
        **details,
    )
    logger.log(level, "zing_cache_event %s", json.dumps(value, separators=(",", ":")))
