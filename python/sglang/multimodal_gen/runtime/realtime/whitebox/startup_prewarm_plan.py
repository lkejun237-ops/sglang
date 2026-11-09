# SPDX-License-Identifier: Apache-2.0
"""Bounded, immutable candidates sharing one startup capture and cleanup gate."""

import asyncio
import copy
import hashlib
import json
import logging
from collections import deque

from .render_contract import RenderError, identity
from .source_conditions import (
    ENVELOPE,
    envelope,
    integer,
    invalid,
    keys,
    source_reached,
)
from .startup_prewarm import CLEANUP_MARGIN_SECONDS, StartupPrewarm, parse_decision

PROTOCOL = "worldkit-model-context-prewarm-plan-v1"
APPEND_PROTOCOL = "worldkit-model-context-prewarm-append-v1"
PREFIX = "model.context.prewarm.plan"
COMMON = (*ENVELOPE, "type", "windowId", "decisionRevision", "planId", "planRevision")
ENTRY = (
    "id",
    "candidateRevision",
    "eventId",
    "targetHash",
    "contentHash",
    "lifecycle",
    "prompt",
    "references",
    "effective",
)
MAX_ENTRIES = 32
MAX_BUDGET_MS = 120000


def parse_plan(value):
    kind = value.get("type")
    if kind in {PREFIX, PREFIX + ".append"}:
        initial = kind == PREFIX
        keys(
            value,
            (
                *COMMON,
                "entries",
                *(("budgetMs", "waitFor") if initial else ()),
                *(("mode",) if initial and "mode" in value else ()),
            ),
        )
        if initial:
            integer(value["budgetMs"])
            if "mode" in value and value["mode"] != "appendable":
                raise invalid()
        entries = value["entries"]
        if (
            initial
            and (
                value["budgetMs"] > MAX_BUDGET_MS
                or value["waitFor"] not in {"memory-ready", "persisted"}
            )
            or not isinstance(entries, list)
            or len(entries) > MAX_ENTRIES
        ):
            raise invalid()
        if not entries and (
            not initial or value["budgetMs"] != 0 and value.get("mode") != "appendable"
        ):
            raise invalid()
        seen = set()
        for entry in entries:
            keys(entry, ENTRY)
            identity(entry["id"])
            if (
                entry["id"] in seen
                or entry["candidateRevision"] != 1
                or type(entry["candidateRevision"]) is not int
            ):
                raise invalid()
            seen.add(entry["id"])
            # Reuse the authoritative candidate/resource validation, not a
            # second interpretation of references or effective frame identity.
            parse_decision(
                dict(
                    **envelope(value),
                    type="model.context.prewarm",
                    windowId=value["windowId"],
                    decisionRevision=value["decisionRevision"],
                    decision="candidate",
                    intentId=entry["id"],
                    **{k: v for k, v in entry.items() if k != "id"},
                )
            )
    elif kind == PREFIX + ".cancel":
        keys(value, (*COMMON, "id", "candidateRevision", "reason"))
        identity(value["id"])
        if (
            type(value["candidateRevision"]) is not int
            or value["candidateRevision"] != 1
        ):
            raise invalid()
        identity(value["reason"])
    elif kind == PREFIX + ".finish":
        keys(value, (*COMMON, "reason"))
        identity(value["reason"])
    else:
        raise invalid()
    envelope(value)
    for field in ("windowId", "planId"):
        identity(value[field])
    integer(value["decisionRevision"], 1)
    if type(value["planRevision"]) is not int or value["planRevision"] != 1:
        raise invalid()
    return copy.deepcopy(value)


class StartupPrewarmPlan(StartupPrewarm):
    def __init__(self, *args):
        super().__init__(*args)
        self.plan_condition = None
        self.entries = {}
        self.history = {}
        self.control_revision = 0
        self.controls = {}
        self.append_supported = False
        self.stopping = False
        self.started_at = None
        self.binding = None
        self.outbound = deque(maxlen=256)

    def binding_identity(self):
        binding = getattr(self.owner, "model_binding", None)
        denoiser = getattr(getattr(binding, "assignment", None), "denoiser", None)
        return (
            binding,
            getattr(denoiser, "worker_id", None),
            getattr(denoiser, "worker_epoch", None),
        )

    def gate(self, condition, *, emit, resolve, check):
        if condition["segmentId"] != self.segment_id or self.retired:
            return None
        # Formal preparation may be withdrawn and replaced while the plan keeps
        # its original frame and candidate inputs. Never retarget that plan.
        if self.task is not None and envelope(condition) != self.identity:
            raise invalid("MODEL_CONTEXT_STALE_SOURCE")
        check()
        self.condition = condition
        if self.task is None:
            self.plan_condition = copy.deepcopy(condition)
            self.started_at = asyncio.get_running_loop().time()
            self.deadline = min(
                self.started_at + MAX_BUDGET_MS / 1000,
                self.opening_deadline - CLEANUP_MARGIN_SECONDS,
            )
            if self.deadline <= self.started_at:
                return None
            self.identity = envelope(condition)
            self.binding = self.binding_identity()
            self.seed = self.owner.context_seed
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
            type=PREFIX + ".window",
            windowId=self.window_id,
            openingRevision=self.opening_revision,
            remainingMs=max(
                0, int((self.deadline - asyncio.get_running_loop().time()) * 1000)
            ),
        )
        if self.append_supported:
            message["prewarmAppendProtocol"] = APPEND_PROTOCOL
        self.observe_player(message)
        self.send(message)
        for message in self.history.values():
            self.send(message)

    def result(self, entry, phase, **details):
        if self.decision is None:
            return
        message = dict(
            **self.identity,
            type=PREFIX + ".result",
            windowId=self.window_id,
            decisionRevision=self.decision["decisionRevision"],
            planId=self.decision["planId"],
            planRevision=self.decision["planRevision"],
            phase=phase,
        )
        if entry is not None:
            message.update(id=entry["id"], candidateRevision=entry["candidateRevision"])
            item = self.entries.get(entry["id"])
            if item is not None:
                message["decisionRevision"] = item["decision_revision"]
        duration = details.pop("duration_ms", None)
        message.update(details)
        category = (
            "persistence"
            if phase in {"persisted", "persistence-failed"}
            else "intent"
            if phase == "invalidated"
            else "skipped"
            if phase == "skipped"
            else "terminal"
            if phase in {"stopping", "finished"}
            else "prepare"
        )
        key = (entry["id"] if entry else None, category)
        if self.history.get(key) == message:
            return
        self.history[key] = message
        self.observe_player(
            dict(message, durationMs=duration) if duration is not None else message,
            entry,
        )
        self.send(message)

    def add_entries(self, value):
        # Validate the entire batch before changing the immutable history.
        if len(self.entries) + len(value["entries"]) > MAX_ENTRIES or any(
            entry["id"] in self.entries for entry in value["entries"]
        ):
            raise invalid()
        if any(
            entry["lifecycle"] != self.plan_condition["lifecycle"]
            for entry in value["entries"]
        ):
            raise invalid("MODEL_CONTEXT_STALE_SOURCE")
        for entry in value["entries"]:
            self.entries[entry["id"]] = {
                "entry": entry,
                "decision_revision": value["decisionRevision"],
                "cancelled": False,
                "memory_ready": False,
                "persisted": False,
                "persistence_failed": False,
                "request_id": None,
                "done": False,
            }

    def receive(self, value):
        if self.task is None or self.retired:
            return
        value = parse_plan(value)
        if value["windowId"] != self.window_id or envelope(value) != self.identity:
            return
        if not self.trusted():
            self.stopping = True
            self.failure = self.failure or RenderError("MODEL_UNAVAILABLE")
            self.changed.set()
            return
        if (
            value.get("mode") == "appendable" or value["type"] == PREFIX + ".append"
        ) and not self.append_supported:
            raise invalid("MODEL_CONTEXT_CAPABILITY_UNSUPPORTED")
        revision = value["decisionRevision"]
        fingerprint = hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).digest()
        previous = self.controls.get(revision)
        if previous is not None:
            if previous != fingerprint:
                raise invalid()
            for message in self.history.values():
                self.send(message)
            return
        if revision <= self.control_revision:
            raise invalid()
        if value["type"] == PREFIX:
            if self.decision is not None:
                return
            self.add_entries(value)
            self.decision = value
            self.deadline = min(
                self.deadline, self.started_at + value["budgetMs"] / 1000
            )
            if self.released.done():
                for entry in value["entries"]:
                    self.result(entry, "skipped", reason="window-closed")
                self.result(None, "finished", reason="window-closed")
        else:
            if self.decision is None or any(
                value[k] != self.decision[k] for k in ("planId", "planRevision")
            ):
                return
            if value["type"] == PREFIX + ".append":
                if self.decision.get("mode") != "appendable":
                    raise invalid("MODEL_CONTEXT_CAPABILITY_UNSUPPORTED")
                closed = (
                    self.stopping
                    or self.released.done()
                    or asyncio.get_running_loop().time() >= self.deadline
                )
                self.add_entries(value)
                if closed:
                    # Retain bounded rejected identities for exact replay, without
                    # reopening the gate or disturbing any prior item result.
                    for entry in value["entries"]:
                        item = self.entries[entry["id"]]
                        item["done"] = item["cancelled"] = True
                        self.result(entry, "skipped", reason="window-closed")
            elif value["type"] == PREFIX + ".finish":
                if self.stopping:
                    raise invalid()
                self.stopping = True
                if not self.released.done():
                    self.result(None, "stopping", reason=value["reason"])
                else:
                    for message in self.history.values():
                        self.send(message)
            else:
                item = self.entries.get(value["id"])
                if (
                    item is None
                    or value["candidateRevision"] != item["entry"]["candidateRevision"]
                ):
                    raise invalid()
                if item["cancelled"]:
                    raise invalid()
                item["cancelled"] = True
                self.result(item["entry"], "invalidated", reason=value["reason"])
        self.controls[revision] = fingerprint
        self.control_revision = revision
        self.changed.set()

    def trusted(self):
        if self.retired or self.binding_identity() != self.binding:
            return False
        try:
            self.check()
        except (RenderError, ValueError):
            self.failure = RenderError("MODEL_CONTEXT_STALE_SOURCE")
            self.stopping = True
            self.changed.set()
            return False
        return True

    def current(self, entry):
        return (
            not self.retired
            and not self.stopping
            and self.failure is None
            and self.trusted()
            and not self.entries[entry["id"]]["cancelled"]
            and asyncio.get_running_loop().time() < self.deadline
        )

    async def wait_current(self, task, entry):
        while self.current(entry):
            self.changed.clear()
            if await self.wait_step(task):
                return True
        return False

    async def prepare(self, entry):
        if entry["lifecycle"] != self.plan_condition["lifecycle"]:
            raise invalid("MODEL_CONTEXT_STALE_SOURCE")
        value = dict(**self.identity, **entry)
        refs = await self.resolve(value)
        if not self.current(entry):
            raise invalid("MODEL_CONTEXT_STALE_SOURCE")
        if self.first is None:
            # Pin once. Later candidates may never pick a newer frame.
            version = 0
            while True:
                first = await self.owner.latest.get(version)
                if first is None:
                    raise RenderError("INVALID_SOURCE")
                if not self.current(entry):
                    raise invalid("MODEL_CONTEXT_STALE_SOURCE")
                version, (frame, _) = first
                if source_reached(frame, value) and source_reached(
                    frame, self.plan_condition
                ):
                    self.first = first
                    break
        if not source_reached(self.first[1][0], value):
            raise invalid("MODEL_CONTEXT_STALE_SOURCE")
        self.check()
        return refs, self.first

    async def compute(self, prefill, entry, refs, first):
        item = self.entries[entry["id"]]
        started = asyncio.get_running_loop().time()

        def observe(message):
            if not self.trusted():
                return
            request_id = message.get("request_id")
            if item["request_id"] not in (None, request_id):
                return
            item["request_id"] = request_id
            phase = {
                "prefill.accepted": "accepted",
                "prefill.computing": "computing",
                "prefill.memory_ready": "memory-ready",
                "prefill.persisted": "persisted",
                "prefill.failed": "failed",
            }.get(message.get("type"))
            if phase is None:
                return
            details = {"requestId": request_id}
            if isinstance(message.get("cache_key"), str):
                details["cacheKey"] = message["cache_key"]
            if isinstance(message.get("code"), str):
                details["reason"] = message["code"]
            if phase == "persisted":
                if not item["memory_ready"]:
                    return
                item["persisted"] = True
            elif phase == "failed" and message.get("memory_ready") is True:
                item["persistence_failed"] = True
                self.result(entry, "persistence-failed", **details)
                self.changed.set()
                return
            elif not self.current(entry):
                return
            elif phase == "memory-ready":
                item["memory_ready"] = True
                self.computed = entry
                details["duration_ms"] = int(
                    (asyncio.get_running_loop().time() - started) * 1000
                )
            self.result(entry, phase, **details)
            self.changed.set()

        async with asyncio.timeout_at(self.deadline):
            active = await prefill.start_startup(
                {"prompt": entry["prompt"], "seed": self.seed},
                refs,
                first,
                current=lambda: self.current(entry),
                observe=observe,
            )
        if active is None:
            return
        try:
            await self.wait_current(active, entry)
        finally:
            if not active.done():
                active.cancel()
            # This wait intentionally outlives the budget. It is the native
            # cleanup barrier; startup must not borrow a second child early.
            await asyncio.gather(active, return_exceptions=True)
        if prefill.failure is not None:
            raise RenderError("RELEASE_UNKNOWN")
        if self.decision["waitFor"] == "persisted" and item["memory_ready"]:
            while (
                self.current(entry)
                and not item["persisted"]
                and not item["persistence_failed"]
            ):
                self.changed.clear()
                async with asyncio.timeout_at(self.deadline):
                    await self.changed.wait()
        if not self.current(entry) and not item["cancelled"]:
            self.result(entry, "invalidated", reason="window-ended")

    async def run(self):
        prefill = self.owner.control_prefill
        try:
            # An absent Host response cannot hold startup indefinitely.
            async with asyncio.timeout_at(min(self.deadline, self.started_at + 5)):
                while self.decision is None and not self.retired:
                    self.changed.clear()
                    await self.changed.wait()
            while True:
                self.changed.clear()
                item = next(
                    (item for item in self.entries.values() if not item["done"]), None
                )
                if item is None:
                    if (
                        self.decision is None
                        or self.decision.get("mode") != "appendable"
                        or self.stopping
                        or not self.trusted()
                        or self.retired
                    ):
                        break
                    async with asyncio.timeout_at(self.deadline):
                        await self.changed.wait()
                    continue
                entry = item["entry"]
                if not self.current(entry):
                    if not item["cancelled"]:
                        self.result(entry, "skipped", reason="window-ended")
                    item["done"] = True
                    continue
                prep = asyncio.create_task(self.prepare(entry))
                try:
                    if await self.wait_current(prep, entry):
                        refs, first = prep.result()
                        await self.compute(prefill, entry, refs, first)
                    elif not item["cancelled"]:
                        self.result(entry, "skipped", reason="window-ended")
                except TimeoutError:
                    self.result(entry, "skipped", reason="budget-exhausted")
                except (RenderError, ValueError) as error:
                    self.result(
                        entry,
                        "failed",
                        reason=error.code
                        if isinstance(error, RenderError)
                        else "prewarm-failed",
                    )
                    if prefill.failure is not None:
                        raise RenderError("RELEASE_UNKNOWN")
                finally:
                    if not prep.done():
                        prep.cancel()
                    await asyncio.gather(prep, return_exceptions=True)
                    item["done"] = True
            if self.binding_identity() != self.binding:
                raise RenderError("MODEL_UNAVAILABLE")
        except TimeoutError:
            pass
        except Exception as error:
            logging.getLogger(__name__).exception("Startup prewarm plan failed")
            self.failure = (
                error
                if isinstance(error, RenderError)
                else RenderError("MODEL_UNAVAILABLE")
            )
        finally:
            for item in self.entries.values():
                if not item["done"] and not item["cancelled"]:
                    self.result(item["entry"], "skipped", reason="window-ended")
            if prefill.failure is not None:
                self.failure = RenderError("RELEASE_UNKNOWN")
            if self.failure is not None:
                self.result(None, "finished", reason=self.failure.code)
                self.released.set_exception(self.failure)
            else:
                self.result(None, "finished")
                self.released.set_result(None)

    def prepared_for(self, condition):
        return super().prepared_for(condition) and (self.task is None or self.trusted())

    def first_for(self, condition):
        if self.binding_identity() != self.binding:
            return None
        return super().first_for(condition) if self.computed is not None else None
