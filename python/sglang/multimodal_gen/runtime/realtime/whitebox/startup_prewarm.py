# SPDX-License-Identifier: Apache-2.0
"""One candidate window owned by an explicit World-driven inference start."""

from __future__ import annotations

import asyncio
import copy
import logging
from collections import deque
from uuid import uuid4

from .render_contract import RenderError, identity
from .source_conditions import (
    ENVELOPE,
    digest,
    effective,
    envelope,
    integer,
    invalid,
    keys,
    payload,
    request_identity,
    source_reached,
)

PROTOCOL = "worldkit-model-context-prewarm-v1"
WAIT_SECONDS = 5
CLEANUP_MARGIN_SECONDS = 1
DECISION = "model.context.prewarm"
CANCEL = "model.context.prewarm.cancel"
COMMON = (*ENVELOPE, "type", "windowId", "decisionRevision")
CANDIDATE_ID = ("intentId", "candidateRevision")
NATIVE_PHASE = {
    "prefill.accepted": "accepted",
    "prefill.computing": "computing",
    "prefill.memory_ready": "memory-ready",
    "prefill.persisted": "persisted",
    "prefill.failed": "failed",
}


def parse_decision(value):
    kind = value.get("type")
    if kind == DECISION and value.get("decision") == "none":
        keys(value, (*COMMON, "decision"))
    elif kind == DECISION and value.get("decision") == "candidate":
        keys(
            value,
            (
                *COMMON,
                "decision",
                *CANDIDATE_ID,
                "eventId",
                "targetHash",
                "contentHash",
                "lifecycle",
                "prompt",
                "references",
                "effective",
            ),
        )
        identity(value["eventId"])
        digest(value["targetHash"])
        digest(value["contentHash"])
        effective(value["effective"])
        payload(dict(value, onsetOccurrenceIds=[], ongoingOccurrenceIds=[]))
    elif kind == CANCEL:
        keys(value, (*COMMON, *CANDIDATE_ID, "reason"))
        identity(value["reason"])
    else:
        raise invalid()
    envelope(value)
    identity(value["windowId"])
    integer(value["decisionRevision"], 1)
    if "intentId" in value:
        identity(value["intentId"])
        integer(value["candidateRevision"], 1)
    return copy.deepcopy(value)


class StartupPrewarm:
    def __init__(self, owner, segment_id, opening_revision, opening_deadline):
        self.owner = owner
        self.segment_id = segment_id
        self.opening_revision = opening_revision
        self.opening_deadline = opening_deadline
        self.window_id = uuid4().hex
        self.changed = asyncio.Event()
        self.released = asyncio.get_running_loop().create_future()
        self.released.add_done_callback(
            lambda future: future.exception() if not future.cancelled() else None
        )
        self.task = self.sender = None
        self.condition = self.identity = self.decision = self.last_result = None
        self.emit = self.resolve = self.check = None
        self.deadline = None
        self.outbound = deque(maxlen=16)
        self.failure = None
        self.retired = False
        self.first = self.computed = None
        self.request_id = None
        self.request_value = None

    def gate(self, condition, *, emit, resolve, check):
        if condition["segmentId"] != self.segment_id or self.retired:
            return None
        self.condition = condition
        if self.task is None:
            now = asyncio.get_running_loop().time()
            self.deadline = min(
                now + WAIT_SECONDS, self.opening_deadline - CLEANUP_MARGIN_SECONDS
            )
            if self.deadline <= now:
                return None
            self.identity = envelope(condition)
            self.emit, self.resolve, self.check = emit, resolve, check
            self.task = self.owner.spawn_control(self.run())
            self.announce()
        if self.released.done():
            if self.failure is not None:
                raise self.failure
            return None
        return self.opening_deadline, self.released

    def announce(self):
        if self.task is None or self.retired:
            return
        message = dict(
            **self.identity,
            type="model.context.prewarm.window",
            windowId=self.window_id,
            openingRevision=self.opening_revision,
            remainingMs=max(
                0, int((self.deadline - asyncio.get_running_loop().time()) * 1000)
            ),
        )
        self.observe_player(message)
        self.send(message)
        if self.last_result is not None:
            self.send(self.last_result)

    def observe_player(self, message, decision=None):
        observer = getattr(self.owner, "record_startup_prewarm", None)
        if observer is not None:
            observer(self, message, decision)

    def send(self, value):
        if self.retired:
            return
        self.outbound.append(value)
        if self.sender is None or self.sender.done():
            self.sender = self.owner.spawn_control(self.flush())

    async def flush(self):
        while self.outbound and not self.retired:
            await self.emit(self.outbound.popleft())

    def result_message(self, value, phase, **details):
        decision = value or {}
        return dict(
            **self.identity,
            type="model.context.prewarm.result",
            windowId=self.window_id,
            decisionRevision=decision.get("decisionRevision", 0),
            **{key: decision[key] for key in CANDIDATE_ID if key in decision},
            phase=phase,
            **details,
        )

    def result(self, value, phase, *, player_only=False, duration_ms=None, **details):
        message = self.result_message(value, phase, **details)
        observation = (
            dict(message, durationMs=duration_ms)
            if duration_ms is not None
            else message
        )
        self.observe_player(observation, value)
        if not player_only:
            if self.decision is value:
                self.last_result = message
            self.send(message)

    def receive(self, value):
        if self.task is None or self.retired:
            return
        value = parse_decision(value)
        if value["windowId"] != self.window_id or envelope(value) != self.identity:
            return
        current = self.decision
        if current and value["decisionRevision"] <= current["decisionRevision"]:
            if value == current and self.last_result is not None:
                self.send(self.last_result)
            return
        if self.released.done() or current and current.get("decision") != "candidate":
            # Reply to the source without replacing the Player summary of
            # the candidate that actually owned this completed window.
            self.send(self.result_message(value, "skipped", reason="window-closed"))
            return
        if (
            value["type"] == CANCEL
            and current
            and any(value[key] != current[key] for key in CANDIDATE_ID)
        ):
            self.send(
                self.result_message(value, "invalidated", reason="stale-candidate")
            )
            return
        self.decision, self.last_result = value, None
        self.changed.set()

    def current(self, value):
        return (
            not self.retired
            and value is self.decision
            and asyncio.get_running_loop().time() < self.deadline
        )

    async def wait_step(self, task):
        changed = asyncio.create_task(self.changed.wait())
        try:
            done, _ = await asyncio.wait(
                (task, changed),
                timeout=max(0, self.deadline - asyncio.get_running_loop().time()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            return task in done
        finally:
            changed.cancel()
            await asyncio.gather(changed, return_exceptions=True)

    async def prepare(self, value):
        if value["lifecycle"] != self.condition["lifecycle"]:
            raise invalid("MODEL_CONTEXT_STALE_SOURCE")
        refs = await self.resolve(value)
        version = 0
        while True:
            first = await self.owner.latest.get(version)
            if first is None:
                raise RenderError("INVALID_SOURCE")
            self.check()
            version, (frame, _) = first
            if source_reached(frame, value) and source_reached(frame, self.condition):
                return refs, first

    async def run(self):
        prefill = self.owner.control_prefill
        try:
            while (
                not self.retired and asyncio.get_running_loop().time() < self.deadline
            ):
                self.changed.clear()
                value = self.decision
                if value is None:
                    async with asyncio.timeout_at(self.deadline):
                        await self.changed.wait()
                    continue
                if value.get("decision") != "candidate":
                    self.result(value, "skipped", reason=value.get("reason", "none"))
                    break
                prep = asyncio.create_task(self.prepare(value))
                try:
                    ready = await self.wait_step(prep)
                finally:
                    if not prep.done():
                        prep.cancel()
                        await asyncio.gather(prep, return_exceptions=True)
                if not self.current(value) or not ready:
                    await asyncio.gather(prep, return_exceptions=True)
                    if value is self.decision and not self.retired:
                        self.result(value, "skipped", reason="budget-exhausted")
                    continue
                refs, first = prep.result()
                await self.compute(prefill, value, refs, first)
                break
        except TimeoutError:
            self.result(self.decision, "skipped", reason="budget-exhausted")
        except Exception as error:
            if not isinstance(error, (RenderError, ValueError)):
                logging.getLogger(__name__).exception("Startup prewarm failed")
            if self.decision is not None:
                self.result(
                    self.decision,
                    "failed",
                    reason=(
                        error.code
                        if isinstance(error, RenderError)
                        else "prewarm-failed"
                    ),
                )
        finally:
            if prefill.failure is not None:
                self.failure = RenderError("RELEASE_UNKNOWN")
                if self.decision is not None:
                    request = (
                        {"requestId": self.request_id}
                        if self.request_id is not None
                        and self.decision is self.request_value
                        else {}
                    )
                    self.result(
                        self.decision, "failed", reason=self.failure.code, **request
                    )
                self.released.set_exception(self.failure)
            else:
                self.released.set_result(None)

    async def compute(self, prefill, value, refs, first):
        memory_ready = False
        started = asyncio.get_running_loop().time()

        def observe(message):
            nonlocal memory_ready
            kind = message.get("type")
            phase = NATIVE_PHASE.get(kind)
            if not phase:
                return
            self.request_id = message["request_id"]
            self.request_value = value
            if self.retired or value is not self.decision:
                return
            persistence_failed = (
                kind == "prefill.failed" and message.get("memory_ready") is True
            )
            if kind == "prefill.persisted" or persistence_failed:
                if not memory_ready:
                    return
            elif not self.current(value):
                return
            if kind == "prefill.memory_ready":
                memory_ready = True
            details = dict(requestId=self.request_id)
            if isinstance(message.get("cache_key"), str):
                details["cacheKey"] = message["cache_key"]
            if isinstance(message.get("code"), str):
                details["reason"] = message["code"]
            # Persistence failure does not invalidate an established memory entry.
            if persistence_failed:
                self.result(value, "persistence-failed", player_only=True, **details)
                return
            duration = (
                int((asyncio.get_running_loop().time() - started) * 1000)
                if kind == "prefill.memory_ready"
                else None
            )
            self.result(value, phase, duration_ms=duration, **details)

        async with asyncio.timeout_at(self.deadline):
            active = await prefill.start_startup(
                dict(prompt=value["prompt"], seed=self.owner.context_seed),
                refs,
                first,
                current=lambda: self.current(value),
                observe=observe,
            )
        if active is None:
            return
        try:
            ready = await self.wait_step(active)
        finally:
            if not active.done():
                active.cancel()
            await asyncio.gather(active, return_exceptions=True)
        if not ready or not self.current(value):
            if not self.retired:
                details = {"requestId": self.request_id} if self.request_id else {}
                self.result(value, "invalidated", reason="window-ended", **details)
                if self.decision is not value:
                    self.result(self.decision, "skipped", reason="window-closed")
            return
        if memory_ready:
            self.first, self.computed = first, value

    def prepared_for(self, condition):
        return (
            not self.retired
            and self.condition is not None
            and request_identity(condition) == request_identity(self.condition)
            and (
                self.task is None
                or self.released.done()
                and not self.released.cancelled()
                and self.failure is None
            )
        )

    def first_for(self, condition):
        if (
            self.first is None
            or not self.prepared_for(condition)
            or condition["lifecycle"] != self.computed["lifecycle"]
            or envelope(condition) != self.identity
        ):
            return None
        try:
            return self.first if source_reached(self.first[1][0], condition) else None
        except ValueError:
            return None

    def effective_for(self, condition):
        first = self.first_for(condition)
        if first is None:
            return None
        source = first[1][0].provenance["cli_frame_header"]["source"]
        return dict(
            sourceFrameId=source["sourceFrameId"],
            sourceTimeUs=source["sourceTimeUs"],
            simulationTick=source["simulationTick"],
            policy="at-or-after",
        )

    def invalidate(self):
        self.retired = True
        self.changed.set()

    async def close(self):
        self.invalidate()
        if self.task is not None:
            await asyncio.shield(asyncio.gather(self.task, return_exceptions=True))
        if self.sender is not None:
            self.sender.cancel()
            await asyncio.gather(self.sender, return_exceptions=True)
        self.outbound.clear()
        self.first = None
        if self.failure is not None:
            raise self.failure
