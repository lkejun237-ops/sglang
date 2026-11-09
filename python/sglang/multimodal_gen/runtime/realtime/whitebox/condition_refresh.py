# SPDX-License-Identifier: Apache-2.0
"""Operator-owned refresh policy; no client command or additional model owner."""

import asyncio
import json
import logging
import math
import time
from pathlib import Path


class ConditionRefreshPolicy:
    def __init__(self, path):
        self.path = Path(path)
        self.seconds = 0.0
        self.revision = 0
        self.next_read = 0.0

    def read(self, now):
        if now < self.next_read:
            return self.seconds, self.revision
        self.next_read = now + 1
        try:
            with self.path.open() as source:
                value = json.loads(source.read(4097))["interval_seconds"]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 86400
            ):
                raise ValueError("Invalid condition refresh interval")
            if value != self.seconds:
                self.seconds = float(value)
                self.revision += 1
                logging.getLogger(__name__).info(
                    "condition_refresh_policy interval_seconds=%s revision=%s",
                    self.seconds,
                    self.revision,
                )
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError):
            logging.getLogger(__name__).warning(
                "Invalid condition refresh policy; retaining interval_seconds=%s",
                self.seconds,
            )
        return self.seconds, self.revision


class PeriodicConditionRefresh:
    def __init__(self, segment, policy, *, clock=time.monotonic):
        self.segment, self.policy, self.clock = segment, policy, clock
        self.revision = -1
        self.interval = 0.0
        self.deadline = None
        self.last_version = None
        self.progress = self.last_progress = 0
        self.failed_revision = None
        self.last_failure = None

    def observe(self, message):
        self.progress += message["num_frames"]
        version = message.get("prompt_version") or 1
        if version != self.last_version:
            self.last_version = version
            self.deadline = self.clock() + self.interval if self.interval else None

    def eligible(self, state):
        segment, source = self.segment, self.segment.owner
        current = state.get("segment")
        if (
            not current
            or current["segment_id"] != segment.segment_id
            or current["segment_epoch"] != segment.segment_epoch
            or current["state"] != "streaming"
            or state["source_state"] != "ready"
            or current["accepted_version"] != current["applied_version"]
            or current["applied_version"] < 1
            or current["rebase_required"]
            or current["prompt_queue"]
            or current.get("validating_prompt")
            or source.close_requested
            or source.binding_ended
            or getattr(source, "closing", False)
            or getattr(source, "pending_end", None)
            or segment.cancel_requested
            or segment.source_view.stopped.is_set()
            or segment.pending_prompt is not None
            or segment.condition_switch is not None
            or segment.sink.busy
            or not segment.clip_switch
            or not state.get("inference", {}).get("enabled", True)
        ):
            return False
        sessions = getattr(source.host, "sessions", None)
        if sessions is not None:
            pipeline = sessions.get(source.session_id)
            if (
                pipeline is None
                or pipeline.subscription is None
                or pipeline.subscription.closed
            ):
                return False
        channel = getattr(source, "context_channel", None)
        if channel is not None and getattr(channel, "suspended", False):
            return False
        if getattr(source, "first_presented_required", False) and not getattr(
            channel, "first_presented_supported", False
        ):
            return False
        if getattr(channel, "first_presented_supported", False):
            from .opening_presentation import (
                applied_context_operation,
                retained_first_presented,
            )

            active, shown = (
                applied_context_operation(state),
                retained_first_presented(state),
            )
            if (
                not active
                or not shown
                or shown.get("first_presented") != "acknowledged"
                or active["model_context_lifecycle"]
                != shown.get("model_context_lifecycle")
            ):
                return False
        return True

    async def tick(self, now=None):
        now = self.clock() if now is None else now
        seconds, revision = self.policy.read(now)
        if revision != self.revision:
            self.revision, self.interval = revision, seconds
            self.deadline = now + seconds if seconds else None
            self.failed_revision = None
        source = self.segment.owner
        if seconds == 0:
            current = source.journal.snapshot.get("segment") or {}
            if (
                current.get("periodic_refresh", {}).get("operation", {}).get("state")
                == "accepted"
            ):
                async with self.segment.source_send_lock:
                    await source.journal.cancel_periodic_refresh(
                        self.segment.segment_id
                    )
            return
        state = source.journal.snapshot
        operation = (
            (state.get("segment") or {})
            .get("periodic_refresh", {})
            .get("operation", {})
        )
        if (
            operation.get("state") in {"failed", "reconciling"}
            and operation.get("request_id") != self.last_failure
        ):
            self.last_failure = operation["request_id"]
            self.failed_revision = revision
            return
        if not self.eligible(state):
            self.deadline = now + seconds
            return
        if (
            self.deadline is None
            or now < self.deadline
            or self.failed_revision == revision
        ):
            return
        if self.progress == self.last_progress:
            self.deadline = now + seconds
            return
        # Share the original source send boundary; no waiting for GPU here.
        async with self.segment.source_send_lock:
            state = await source.journal.read()
            if not self.eligible(state):
                return
            current = state["segment"]
            applied = next(
                (
                    c
                    for c in self.segment.prompt_by_event.values()
                    if c["version"] == current["applied_version"]
                ),
                {},
            )
            operation = await source.journal.enqueue_periodic_refresh(
                self.segment.segment_id,
                self.segment.segment_epoch,
                current["applied_version"],
                area=applied.get("area_binding"),
            )
            self.last_progress = self.progress
            self.deadline = now + seconds
            if operation is not None:
                logging.getLogger(__name__).info(
                    "condition_refresh_enqueued segment=%s version=%s interval_seconds=%s",
                    self.segment.segment_id,
                    operation["prompt_version"],
                    seconds,
                )
                self.segment.prompt_available.set()

    async def run(self):
        try:
            while not self.segment.closed_event.is_set():
                await self.tick()
                try:
                    await asyncio.wait_for(self.segment.closed_event.wait(), 0.25)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception:
            # Disable this optional scheduler. Original condition/segment owner
            # continues to settle any operation already dispatched.
            logging.getLogger(__name__).exception(
                "Condition refresh scheduler stopped for segment %s",
                self.segment.segment_id,
            )
