# SPDX-License-Identifier: Apache-2.0
"""Bounded observation for one single-player prewarm window."""

from __future__ import annotations

import re

from .cache_receipt import public_cache_receipt

PROTOCOL = "worldkit-model-prewarm-v1"
OBSERVE = "model.context.prewarm.observe"
OBSERVATION = "model.context.prewarm.observation"
IDENTITY = ("intent_id", "candidate_revision")
CANDIDATE = (*IDENTITY, "event_id", "request_id", "semantic_sha256")
SEGMENT = ("segment_id", "segment_epoch")
OPTIONAL = (*CANDIDATE, *SEGMENT, "reason", "duration_ms", "cache_key")
PREPARE_ORDER = {"accepted": 1, "computing": 2, "memory-ready": 3, "failed": 3}
INTENT = {"cancelled", "invalidated", "fulfilled"}
PERSISTENCE = {"persisted", "persistence-failed"}
HIT = {"l1_hit", "l2_hit", "mixed_hit"}


def identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value)


def revision(value):
    return type(value) is int and 0 <= value < 2**53


def is_prewarm_observation_request(value):
    return (
        isinstance(value, dict)
        and set(value) == {"version", "type"}
        and type(value["version"]) is int
        and value["version"] == 2
        and value["type"] == OBSERVE
    )


class PrewarmPlayerObservations:
    """Retain the current decision across reconnect without retaining model inputs."""

    def __init__(self, source_session_id, source_epoch, consumer_id):
        self.source = dict(
            source_session_id=source_session_id,
            source_epoch=source_epoch,
            consumer_id=consumer_id,
        )
        if not all(map(identifier, self.source.values())):
            raise ValueError("Invalid prewarm observation scope.")
        self.window = None
        self.window_generation = 0
        self.candidate = {}
        self.segment_id = None
        self.segment = {}
        self.decision = 0
        self.latest = {}

    @property
    def capability(self):
        return dict(
            model_prewarm_observation=PROTOCOL,
            model_prewarm_consumer_id=self.source["consumer_id"],
        )

    def open(
        self, window_id, inference_revision, *, segment_id=None, segment_epoch=None
    ):
        if (
            not identifier(window_id)
            or not revision(inference_revision)
            or (segment_id is not None and not identifier(segment_id))
            or (
                segment_epoch is not None
                and (not segment_id or not identifier(segment_epoch))
            )
            or (
                self.window
                and (
                    inference_revision < self.window["inference_revision"]
                    or (
                        inference_revision == self.window["inference_revision"]
                        and (not segment_id or segment_id == self.segment_id)
                    )
                )
            )
        ):
            return None
        segment = (
            dict(segment_id=segment_id, segment_epoch=segment_epoch)
            if segment_epoch
            else {}
        )
        self.candidate, self.segment_id, self.segment = {}, segment_id, segment
        self.decision = 0
        self.latest.clear()
        self.window_generation += 1
        self.window = dict(
            window_id=window_id,
            inference_revision=inference_revision,
            window_generation=self.window_generation,
            **segment,
        )
        return self.event(0, "window", segment)

    def record(self, window_id, decision_revision, status, **fields):
        if (
            not self.window
            or self.window["window_id"] != window_id
            or not revision(decision_revision)
            or not valid_fields(status, fields)
        ):
            return None
        identities = {key: fields[key] for key in IDENTITY if key in fields}
        current = {
            key: self.candidate[key] for key in IDENTITY if key in self.candidate
        }
        replacing = identities != current
        if replacing and (
            decision_revision < self.decision
            or (self.latest and decision_revision == self.decision)
        ):
            return None
        known = self.segment if replacing else {**self.segment, **self.candidate}
        if (
            self.segment_id is not None
            and fields.get("segment_id", self.segment_id) != self.segment_id
            or any(key in fields and fields[key] != val for key, val in known.items())
        ):
            return None
        kind = (
            "prepare"
            if status in PREPARE_ORDER
            else "intent"
            if status in INTENT
            else "persistence"
            if status in PERSISTENCE
            else "skipped"
        )
        previous = None if replacing else self.latest.get(kind)
        if previous and (
            (
                kind == "prepare"
                and previous["status"] != status
                and PREPARE_ORDER[status] <= PREPARE_ORDER[previous["status"]]
            )
            or (kind == "intent" and decision_revision < previous["decision_revision"])
            or (kind == "persistence" and previous["status"] == "persisted")
        ):
            return None
        if replacing:
            self.candidate.clear()
            self.latest.clear()
        self.candidate.update({key: fields[key] for key in CANDIDATE if key in fields})
        self.segment.update({key: fields[key] for key in SEGMENT if key in fields})
        self.decision = max(self.decision, decision_revision)
        return self.save(kind, decision_revision, status, fields)

    def record_cache(
        self,
        window_id,
        *,
        formal_request_id,
        prompt_version,
        segment_id,
        segment_epoch,
        cache,
    ):
        # The caller must resolve the formal request from the committed condition
        # journal before passing the native receipt and its authoritative version.
        if (
            not self.window
            or self.window["window_id"] != window_id
            or not self.candidate.get("request_id")
            or not identifier(formal_request_id)
            or not revision(prompt_version)
            or not identifier(segment_id)
            or not identifier(segment_epoch)
            or (self.segment_id or self.segment.get("segment_id")) != segment_id
            or self.segment.get("segment_epoch") not in (None, segment_epoch)
            or "consume" in self.latest
            or not isinstance(cache, dict)
            or not isinstance(cache.get("cache_result"), str)
        ):
            return None
        receipt = public_cache_receipt(cache)
        if receipt is None or receipt["prompt_version"] != prompt_version:
            return None
        outcome = receipt["cache_result"]
        if outcome not in HIT and outcome != "miss":
            return None
        self.segment.update(segment_id=segment_id, segment_epoch=segment_epoch)
        return self.save(
            "consume",
            self.decision,
            "hit" if outcome in HIT else "miss",
            dict(
                formal_request_id=formal_request_id,
                cache_key=receipt["cache_key"],
                **(
                    {"reason": receipt["miss_reason"]} if receipt["miss_reason"] else {}
                ),
            ),
        )

    def save(self, kind, decision_revision, status, fields):
        event = self.event(decision_revision, status, fields)
        if self.latest.get(kind) == event:
            return None
        self.latest[kind] = event
        return event.copy()

    def event(self, decision_revision, status, fields):
        return dict(
            version=2,
            type=OBSERVATION,
            **self.source,
            **{
                **self.window,
                **self.segment,
                **self.candidate,
                **fields,
            },
            decision_revision=decision_revision,
            status=status,
        )

    def wire(self, event, connection_generation):
        return dict(event, connection_generation=connection_generation)

    def replay(self, connection_generation):
        if not self.window:
            return []
        window = dict(
            version=2,
            type=OBSERVATION,
            **self.source,
            **self.window,
            decision_revision=0,
            status="window",
        )
        events = sorted(
            self.latest.values(), key=lambda event: event["decision_revision"]
        )
        return [self.wire(event, connection_generation) for event in (window, *events)]


def valid_segment(value):
    return not value or (
        set(value) == set(SEGMENT) and all(map(identifier, value.values()))
    )


def valid_fields(status, fields):
    if (
        status not in (*PREPARE_ORDER, *INTENT, *PERSISTENCE, "skipped")
        or set(fields) - set(OPTIONAL)
        or not valid_segment({key: fields[key] for key in SEGMENT if key in fields})
    ):
        return False
    for key, value in fields.items():
        if key == "semantic_sha256":
            valid = isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)
        elif key == "candidate_revision":
            valid = revision(value) and value > 0
        elif key == "duration_ms":
            valid = revision(value)
        else:
            valid = identifier(value)
        if not valid:
            return False
    return (
        ("intent_id" in fields) == ("candidate_revision" in fields)
        and ("intent_id" in fields or not set(fields).intersection(CANDIDATE))
        and (status not in INTENT or "intent_id" in fields)
        and (
            status not in {"accepted", "computing", "memory-ready", *PERSISTENCE}
            or "request_id" in fields
        )
        and ("cache_key" not in fields or status == "memory-ready")
    )


PLAN_PROTOCOL = "worldkit-model-prewarm-plan-v1"
PLAN_OBSERVE = "model.context.prewarm.plan.observe"
PLAN_OBSERVATION = "model.context.prewarm.plan.observation"


def is_plan_observation_request(value):
    return (
        isinstance(value, dict)
        and set(value) == {"version", "type"}
        and type(value["version"]) is int
        and value["version"] == 2
        and value["type"] == PLAN_OBSERVE
    )


class PrewarmPlanObservations:
    """At most 32 candidate histories; disk facts never replace intent facts."""

    def __init__(self, **scope):
        self.source = PrewarmPlayerObservations(**scope).source
        self.window = None
        self.generation = 0
        self.rows = {}
        self.latest = {}
        self.consumed = set()

    @property
    def capability(self):
        return dict(
            model_prewarm_plan_observation=PLAN_PROTOCOL,
            model_prewarm_consumer_id=self.source["consumer_id"],
        )

    def record(self, window, message, entry=None):
        if message["type"].endswith(".window"):
            if self.window and self.window["window_id"] == window.window_id:
                return None
            self.generation += 1
            self.window = dict(
                window_id=window.window_id,
                inference_revision=window.opening_revision,
                window_generation=self.generation,
            )
            self.rows.clear()
            self.latest.clear()
            self.consumed.clear()
            return self.event(0, "window", {})
        if not self.window or self.window["window_id"] != window.window_id:
            return None
        fields = {
            dest: message[src]
            for src, dest in (
                ("planId", "plan_id"),
                ("planRevision", "plan_revision"),
                ("id", "candidate_id"),
                ("candidateRevision", "candidate_revision"),
                ("requestId", "request_id"),
                ("reason", "reason"),
                ("durationMs", "duration_ms"),
                ("cacheKey", "cache_key"),
            )
            if src in message
        }
        if entry:
            fields.update(
                event_id=entry["eventId"], semantic_sha256=entry["contentHash"]
            )
        candidate = fields.get("candidate_id")
        if candidate is not None:
            if candidate not in self.rows and len(self.rows) >= 32:
                return None
            row = self.rows.setdefault(candidate, {})
            row.update(
                {
                    k: v
                    for k, v in fields.items()
                    if k not in {"reason", "duration_ms", "cache_key"}
                }
            )
            fields = {**row, **fields}
        phase = message["phase"]
        if phase != "memory-ready":
            fields.pop("cache_key", None)
        category = (
            "intent"
            if phase in {"invalidated", "fulfilled", "cancelled"}
            else "persistence"
            if phase in PERSISTENCE
            else "terminal"
            if candidate is None
            else "skipped"
            if phase == "skipped"
            else "prepare"
        )
        return self.save(
            (candidate, category), message["decisionRevision"], phase, fields
        )

    def event(self, revision, status, fields):
        return dict(
            version=2,
            type=PLAN_OBSERVATION,
            **self.source,
            **self.window,
            decision_revision=revision,
            status=status,
            **fields,
        )

    def save(self, key, revision, status, fields):
        event = self.event(revision, status, fields)
        previous = self.latest.get(key)
        if previous == event or (previous and previous["status"] == "persisted"):
            return None
        self.latest[key] = event
        return event.copy()

    def record_cache(self, window, entry, operation, cache):
        request = operation["request_id"]
        row = self.rows.get(entry["id"])
        if not row or not row.get("request_id") or request in self.consumed:
            return None
        receipt = public_cache_receipt(cache)
        if receipt is None or receipt["prompt_version"] != operation["prompt_version"]:
            return None
        outcome = receipt["cache_result"]
        if outcome not in HIT and outcome != "miss":
            return None
        self.consumed.add(request)
        if len(self.consumed) > 256:
            self.consumed.pop()
        fields = dict(
            row,
            formal_request_id=request,
            segment_id=operation["segment_id"],
            segment_epoch=operation["segment_epoch"],
            cache_key=receipt["cache_key"],
        )
        # One native formal request produces one result, even when two business
        # candidates contain identical inputs.
        return self.save(
            (entry["id"], "consume"),
            window.decision["decisionRevision"],
            "hit" if outcome in HIT else "miss",
            fields,
        )

    def wire(self, event, connection_generation):
        return dict(event, connection_generation=connection_generation)

    def replay(self, connection_generation):
        if not self.window:
            return []
        events = [self.event(0, "window", {}), *self.latest.values()]
        return [self.wire(event, connection_generation) for event in events]


def dynamic_member_plan_scope(session, member, message):
    """Observation admission reuses the original member, never browser opt-in."""
    from .render_contract import RenderError
    from .startup_prewarm_plan import PROTOCOL

    if (
        member is None
        or getattr(session, "member_prefill", None) is not None
        or not getattr(getattr(session, "inference", None), "manual", False)
    ):
        return None
    capability = getattr(session.engine, "model_context_capability", None) or {}
    if capability.get("prewarmPlanProtocol") != PROTOCOL:
        return None
    try:
        member.check()
    except (RenderError, ValueError):
        return None
    channel = getattr(session, "context_channel", None)
    consumer = getattr(getattr(channel, "view", None), "client_id", None)
    source = session.source_identity()
    scope = {key: source[key] for key in ("source_session_id", "source_epoch")}
    if not consumer or any(message.get(key) != value for key, value in scope.items()):
        return None
    return dict(scope, consumer_id=consumer)
