# SPDX-License-Identifier: Apache-2.0
"""Versioned render controls, independent of source and transport lifetimes.

The journal commits intent before a caller may launch a side effect. The store
has the RoomStore get/CAS surface; payloads contain control data, never pixels.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from contextlib import contextmanager
from uuid import uuid4

PROTOCOL = "whitebox-render-v2"
MAX_OPERATIONS = 1024
MAX_PENDING_PROMPTS = 16
MAX_PENDING_PROMPT_BYTES = 1 << 20
MAX_CONTROL_BYTES = 128 << 10
MAX_JOURNAL_BYTES = 2 << 20
# New manual sources reserve 16 KiB inside the existing hard ceiling: 4 KiB
# for the bounded stop/retry row and terminal source/lease fields, plus 768
# bytes for each of the at most 16 pending prompt/event settlements. These
# settle existing bounded identities; neither pixels nor full prompts grow here.
# Normal admission cannot use this reserve; explicit retries reuse the stop row.
JOURNAL_SHUTDOWN_RESERVE_BYTES = (4 << 10) + MAX_PENDING_PROMPTS * 768
MAX_CAPACITY_WAIT_MS = 10_000
MAX_PROMPT_CHARS = 16384
MAX_PROMPT_BYTES = 64 << 10
MAX_REFERENCE_IMAGES = 21
MAX_REFERENCE_BYTES = 64 << 20
TERMINAL_SEGMENTS = {"ended", "cancelled", "failed"}
OCCUPIED_SEGMENTS = {
    "accepted",
    "waiting_capacity",
    "initializing",
    "streaming",
    "draining",
}

MESSAGES = {
    "SOURCE_PAUSE_BUSY": "The game is already being prepared or paused. Please try again shortly.",
    "SOURCE_PAUSE_UNCONFIRMED": "The game pause could not be confirmed. Check its status or resume the original pause.",
    "SINK_UPDATE_BUSY": "Wait for the current model control to finish.",
    "SINK_UPDATE_UNKNOWN": "The sink operation is unconfirmed. Query its original request.",
    "PLAYER_IDLE_TIMEOUT": "Your play session ended because there was no input. Please start again.",
    "PLAYER_EXPIRED": "Your play session has reached its time limit.",
    "BINDING_BUDGET_EXHAUSTED": "Your play session has reached its time limit.",
    "MODEL_CONTEXT_SUBSCRIPTION_PAUSE_UNSUPPORTED": "This world cannot safely pause model inference. Please use an updated world version.",
    "PREFILL_BUSY": "Please wait for the current rendering operation to finish.",
    "ARCHIVE_CAPACITY_UNAVAILABLE": "Recording is temporarily unavailable. Please try again shortly.",
    "AREA_PRESENTATION_STALE": "The visible area has changed. Please refresh its status.",
    "AREA_RECEIPT_INVALID": "The visible area could not be confirmed.",
    "AREA_PRESENTATION_UNKNOWN": "The confirmation is pending. Please check its original request.",
    "MODEL_PRESENTATION_STALE": "The model opening presentation is no longer current.",
    "MODEL_PRESENTATION_UNCONFIRMED": "The game has not confirmed the opening presentation.",
    "MODEL_PRESENTATION_UNSUPPORTED": "This game needs a Player that supports opening presentation.",
    "INITIALIZATION_FAILED": "The model could not finish initialization. Report the trace ID for diagnosis.",
    "RENDER_FAILED": "The rendering service could not complete this operation.",
    "INVALID_COMMAND": "This request is not valid.",
    "INVALID_SOURCE": "This source is not available.",
    "SOURCE_EPOCH_MISMATCH": "The source has changed. Please reconnect.",
    "SEGMENT_EPOCH_MISMATCH": "This rendering session has changed.",
    "REQUEST_ID_CONFLICT": "This request ID was already used for different content.",
    "STARTUP_ALREADY_EXISTS": "Your previous session is still available. Reconnect to it or close it before starting again.",
    "REVISION_CONFLICT": "The session changed. Please refresh its status.",
    "RENDER_TRANSITION_IN_PROGRESS": "A rendering change is already in progress.",
    "MODEL_UNAVAILABLE": "There is no active rendering session.",
    "CAPACITY_EXHAUSTED": "No rendering slot is currently available.",
    "LIMIT_EXCEEDED": "This request exceeds the session limit.",
    "CONDITION_INVALID": "The rendering instructions are not valid.",
    "CONTROL_UNAVAILABLE": "Session controls are temporarily unavailable.",
    "RELEASE_UNKNOWN": "The previous rendering session is still being cleaned up.",
    "START_SOURCE_EXPIRED": "The requested source frame is no longer available.",
    "PREFILL_INPUT_MISMATCH": "The prepared inputs have changed. Please prepare them again.",
    "EFFECTIVE_BOUNDARY_UNAVAILABLE": "The requested update time is not available.",
    "CAPABILITY_UNAVAILABLE": "This operation is not supported by this source.",
    "ROOM_SESSION_LOST": "This source session is no longer available.",
}


class RenderError(ValueError):
    def __init__(self, code, *, retryable=False):
        self.code = code if code in MESSAGES else "RENDER_FAILED"
        if self.code == "INITIALIZATION_FAILED":
            self.stage = "model"
        self.retryable = retryable
        super().__init__(MESSAGES[self.code])

    def message(self, request_id=None):
        return dict(
            type="operation.rejected",
            code=self.code,
            content=str(self),
            request_id=request_id,
            retryable=self.retryable,
        )


def source_create_matches(message):
    """The old strict create discriminator rejects manual init before compute.

    A permissive extra init field alone would silently enable old servers.
    Keep this pair identical in every trusted admission and transport consumer.
    """
    return (
        message.get("source_operation") == "create.manual"
        and message.get("inference_control") == "manual"
        or message.get("source_operation") == "create"
        and "inference_control" not in message
    )


def canonical(value):
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise RenderError("INVALID_COMMAND") from error


def identity(value):
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 128
        or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.:"
            for char in value
        )
    ):
        raise RenderError("INVALID_COMMAND")
    return value


def prompt_text(value):
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > MAX_PROMPT_CHARS
        or len(value.encode()) > MAX_PROMPT_BYTES
    ):
        raise RenderError("CONDITION_INVALID")
    return value


def boundary(value, *, start=False):
    value = {"mode": "next_available"} if value is None and start else value
    if not isinstance(value, dict):
        raise RenderError("INVALID_COMMAND")
    if start and value == {"mode": "next_available"}:
        return dict(value)
    names = {name for name in ("source_frame_index", "source_pts_us") if name in value}
    allowed = names | ({"policy"} if not start else set())
    if len(names) != 1 or set(value) - allowed:
        raise RenderError("INVALID_COMMAND")
    for name in names:
        if type(value[name]) is not int or value[name] < 0:
            raise RenderError("INVALID_COMMAND")
    if not start and value.get("policy", "at_or_after") not in {
        "strict",
        "at_or_after",
    }:
        raise RenderError("INVALID_COMMAND")
    return copy.deepcopy(value)


def reference_parameters(refs):
    if not isinstance(refs, list) or len(refs) > MAX_REFERENCE_IMAGES:
        raise RenderError("LIMIT_EXCEEDED")
    seen = set()
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) - {
            "asset_id",
            "id",
            "role",
            "subject_id",
            "view",
            "layout",
        }:
            raise RenderError("CONDITION_INVALID")
        ref_id, asset_id = ref.get("id"), ref.get("asset_id")
        if (
            not isinstance(ref_id, str)
            or not 1 <= len(ref_id) <= 128
            or ref_id == "<Video 1>"
        ):
            raise RenderError("CONDITION_INVALID")
        if (
            ref_id in seen
            or not isinstance(asset_id, str)
            or not asset_id.startswith("sha256:")
            or len(asset_id) != 71
            or any(c not in "0123456789abcdef" for c in asset_id[7:])
        ):
            raise RenderError("CONDITION_INVALID")
        if ref.get("role", "scene") not in {"subject", "style", "scene"}:
            raise RenderError("CONDITION_INVALID")
        if ref.get("role") == "subject":
            identity(ref.get("subject_id"))
        if "view" in ref and ref["view"] not in {"front", "side", "back"}:
            raise RenderError("CONDITION_INVALID")
        if "layout" in ref and ref["layout"] != "triptych":
            raise RenderError("CONDITION_INVALID")
        seen.add(ref_id)
    return copy.deepcopy(refs)


def area_binding(command):
    if "area_binding" not in command:
        return {}
    from .area_contract import read_area_identity

    value = command["area_binding"]
    if (
        not isinstance(value, dict)
        or set(value) != {"protocol", "identity"}
        or value["protocol"] != "worldkit-area-condition-v1"
    ):
        raise RenderError("CONDITION_INVALID")
    try:
        return dict(
            area_binding=dict(
                protocol=value["protocol"],
                identity=read_area_identity(value["identity"]),
            )
        )
    except ValueError as error:
        raise RenderError("CONDITION_INVALID") from error


def segment_parameters(command):
    prompt = command.get("prompt")
    if isinstance(prompt, dict):
        if (
            set(prompt) != {"version", "text"}
            or type(prompt["version"]) is not int
            or prompt["version"] != 1
        ):
            raise RenderError("CONDITION_INVALID")
        prompt = prompt["text"]
    prompt = prompt_text(prompt)
    seed = command.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise RenderError("INVALID_COMMAND")
    refs = reference_parameters(command.get("references", []))
    wait = command.get("capacity_wait_ms", 0)
    if type(wait) is not int or not 0 <= wait <= MAX_CAPACITY_WAIT_MS:
        raise RenderError("LIMIT_EXCEEDED")
    region = command.get("region_id", "default")
    identity(region)
    if command.get("prepared_input") is not None:
        raise RenderError("CAPABILITY_UNAVAILABLE")
    return dict(
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        seed=seed,
        references=copy.deepcopy(refs),
        region_id=region,
        start_boundary=boundary(command.get("start_boundary"), start=True),
        capacity_wait_ms=wait if command.get("wait_for_capacity", False) else 0,
        **area_binding(command),
        **(
            {"context_id": identity(command["context_id"])}
            if "context_id" in command
            else {}
        ),
    )


class RenderJournal:
    """A fenced, bounded per-source CAS journal, retaining completed operations."""

    def __init__(
        self,
        store,
        *,
        user_id,
        source_session_id,
        source_epoch,
        owner_epoch,
        retention=86400,
        clock=time.time,
    ):
        self.store, self.clock, self.retention = store, clock, retention
        self.user_id = user_id
        self.source_session_id = identity(source_session_id)
        self.source_epoch, self.owner_epoch = (
            identity(source_epoch),
            identity(owner_epoch),
        )
        self.key = (
            "render-v2:"
            + hashlib.sha256(user_id.encode()).hexdigest()
            + ":"
            + source_session_id
        )
        self.snapshot = None

    def initial(self):
        return dict(
            schema=2,
            user_id=self.user_id,
            source_session_id=self.source_session_id,
            source_epoch=self.source_epoch,
            owner_epoch=self.owner_epoch,
            state_version=0,
            render_revision=0,
            display_revision=0,
            source_state="binding",
            display_mode="whitebox",
            segment=None,
            pending_replace=None,
            operations={},
            created_at=self.clock(),
        )

    def validate_scope(self, state, command=None):
        if state["user_id"] != self.user_id or state["owner_epoch"] != self.owner_epoch:
            raise RenderError("ROOM_SESSION_LOST")
        if state["source_epoch"] != self.source_epoch:
            raise RenderError("SOURCE_EPOCH_MISMATCH")
        if command is not None:
            if command.get("source_session_id") != self.source_session_id:
                raise RenderError("INVALID_SOURCE")
            if command.get("source_epoch") != self.source_epoch:
                raise RenderError("SOURCE_EPOCH_MISMATCH")

    async def read(self):
        raw = await self.store.get(self.key)
        if raw is None:
            raise RenderError("ROOM_SESSION_LOST")
        state = json.loads(raw)
        self.validate_scope(state)
        self.snapshot = copy.deepcopy(state)
        return state

    async def create(self):
        value = self.initial()
        if not await self.store.cas(
            self.key, None, canonical(value).decode(), self.retention
        ):
            return await self.read()
        self.snapshot = copy.deepcopy(value)
        return value

    async def update(self, change, *, settlement=False):
        for _ in range(8):
            before = await self.read()
            after = copy.deepcopy(before)
            result = change(after)
            after["state_version"] += 1
            size = len(canonical(after))
            reserved = settlement(before) if callable(settlement) else settlement
            if size > MAX_JOURNAL_BYTES or (
                before.get("manual_inference_journal")
                and not reserved
                and size > MAX_JOURNAL_BYTES - JOURNAL_SHUTDOWN_RESERVE_BYTES
                and size > len(canonical(before))
            ):
                raise RenderError("LIMIT_EXCEEDED")
            if await self.store.cas(
                self.key,
                canonical(before).decode(),
                canonical(after).decode(),
                self.retention,
            ):
                self.snapshot = copy.deepcopy(after)
                return copy.deepcopy(result)
        raise RenderError("CONTROL_UNAVAILABLE", retryable=True)

    async def command(self, command, transition, *, duplicate=None):
        if not isinstance(command, dict) or len(canonical(command)) > MAX_CONTROL_BYTES:
            raise RenderError("LIMIT_EXCEEDED")
        request_id = identity(command.get("request_id"))
        digest = hashlib.sha256(canonical(command)).hexdigest()

        def apply(state):
            self.validate_scope(state, command)
            previous = state["operations"].get(request_id)
            if previous:
                if previous["payload_sha256"] != digest:
                    raise RenderError("REQUEST_ID_CONFLICT")
                if duplicate is not None:
                    return duplicate(state, previous)
                return {**previous, "duplicate": True}
            # Content/history exhaustion must not prevent the final stop. Keep
            # one bounded end record; retries still resolve their original ID.
            limit = MAX_OPERATIONS + (1 if cleanup(state) else 0)
            if len(state["operations"]) >= limit:
                raise RenderError("LIMIT_EXCEEDED")
            operation = dict(
                operation_id=uuid4().hex,
                request_id=request_id,
                payload_sha256=digest,
                op=command.get("op"),
                state="accepted",
                accepted_at=self.clock(),
            )
            transition(state, operation)
            state["operations"][request_id] = operation
            return operation

        def cleanup(state):
            return command.get("op") in {"render.end", "source.resume"} or (
                command.get("op") == "model.inference.configure"
                and command.get("enabled") is False
                and state.get("inference", {}).get("state") != "disabled"
            )

        return await self.update(apply, settlement=cleanup)

    async def initialize_inference(self, *, manual):
        def initialize(state):
            state.setdefault("manual_inference_journal", manual)
            state.setdefault(
                "inference",
                dict(
                    enabled=not manual,
                    revision=0,
                    state="disabled" if manual else "enabled",
                ),
            )

        await self.update(initialize)

    async def configure_inference(self, command):
        if (
            set(command)
            - {
                "type",
                "version",
                "op",
                "request_id",
                "source_session_id",
                "source_epoch",
                "enabled",
                "expected_inference_revision",
            }
            or command.get("op") != "model.inference.configure"
            or type(command.get("enabled")) is not bool
        ):
            raise RenderError("INVALID_COMMAND")

        def transition(state, operation):
            current = state["inference"]
            revision = command.get("expected_inference_revision")
            if type(revision) is not int or revision != current["revision"]:
                raise RenderError("REVISION_CONFLICT")
            if state["source_state"] != "ready":
                raise RenderError("INVALID_SOURCE")
            if current["state"] == "stopping":
                raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
            if current.get("retry") is not None:
                raise RenderError("REVISION_CONFLICT")
            enabled = command["enabled"]
            stable = current["enabled"] == enabled and current["state"] != "failed"
            if stable:
                operation.update(inference=copy.deepcopy(current), state="completed")
                return
            value = dict(
                enabled=enabled,
                revision=revision + 1,
                state="starting" if enabled else "stopping",
            )
            state["inference"] = value
            segment = state["segment"]
            if (
                not enabled
                and segment
                and segment["state"] in {"initializing", "streaming"}
            ):
                # Stop new condition dispatch while already sent work drains.
                segment["state"] = "draining"
            # The next desired revision supersedes accepted control intent in
            # this CAS, even if its task has never been scheduled. This says
            # nothing about model execution or release; stopping remains busy.
            for previous in state["operations"].values():
                if (
                    previous["op"] == "model.inference.configure"
                    and previous["state"] == "accepted"
                ):
                    previous.update(
                        state="completed",
                        inference=copy.deepcopy(value),
                        superseded_by_inference_revision=value["revision"],
                    )
            # A new opt-in selects model display through this original owner,
            # without a second rate-limited operation. Stable/replayed intent
            # above preserves any later explicit raw/model debug selection.
            display_mode = "model" if enabled else "whitebox"
            if state["display_mode"] != display_mode:
                state["display_mode"] = display_mode
                state["display_revision"] += 1
            operation.update(
                inference=copy.deepcopy(value),
                state="accepted",
                expected_inference_revision=revision,
            )

        def duplicate(state, operation):
            current = state["inference"]
            retry = current.get("retry")
            if operation["state"] == "failed" and command["enabled"] is False:
                if (
                    current["enabled"] is not False
                    or current["state"] != "failed"
                    or not retry
                    or retry["request_id"] != command["request_id"]
                    or retry["expected_inference_revision"]
                    != command["expected_inference_revision"]
                    or operation["inference"]["revision"] != current["revision"]
                ):
                    raise RenderError("REVISION_CONFLICT")
                current.update(state="stopping")
                current.pop("code", None)
                current.pop("retry", None)
                operation.update(state="accepted", inference=copy.deepcopy(current))
                operation.pop("code", None)
                return {**operation, "resumed": True}
            return {**operation, "duplicate": True}

        return await self.command(command, transition, duplicate=duplicate)

    async def settle_inference(
        self, request_id, revision, *, state, code=None, preserve_intent=False
    ):
        def settle(value):
            current = value["inference"]
            operation = value["operations"][request_id]
            if current["revision"] != revision:
                operation.update(
                    state="completed",
                    inference=copy.deepcopy(current),
                    superseded_by_inference_revision=current["revision"],
                )
                return operation
            if preserve_intent:
                # Native reset retires this opening, not the accepted desire to
                # keep inference enabled. Its historical result remains separate
                # from the fresh server-owned lifecycle operation.
                operation.update(
                    inference=dict(
                        enabled=current["enabled"], revision=revision, state=state
                    ),
                    state="failed" if code else "completed",
                )
                if code:
                    operation["code"] = code
                return operation
            current.update(state=state)
            if code:
                current["code"] = code
                if current["enabled"] is False:
                    current["retry"] = dict(
                        request_id=request_id,
                        expected_inference_revision=operation[
                            "expected_inference_revision"
                        ],
                    )
            else:
                current.pop("code", None)
                current.pop("retry", None)
            operation.update(
                inference=copy.deepcopy(current),
                state="failed" if code else "completed",
            )
            if code:
                operation["code"] = code
            return operation

        return await self.update(settle, settlement=True)

    async def result(self, request_id, **fields):
        def apply(state):
            operation = state["operations"].get(request_id)
            if operation is None:
                raise RenderError("INVALID_COMMAND")
            operation.update(fields, updated_at=self.clock())
            return operation

        return await self.update(apply, settlement=True)

    @staticmethod
    def require_revision(state, command):
        revision = command.get("expected_render_revision")
        if type(revision) is not int or revision != state["render_revision"]:
            raise RenderError("REVISION_CONFLICT")

    @staticmethod
    def require_segment(state, command):
        segment = state.get("segment")
        if not segment:
            raise RenderError("MODEL_UNAVAILABLE")
        if (
            command.get("segment_id") != segment["segment_id"]
            or command.get("segment_epoch") != segment["segment_epoch"]
        ):
            raise RenderError("SEGMENT_EPOCH_MISMATCH")
        return segment

    async def start(
        self,
        command,
        *,
        replace=False,
        startup_ids=None,
        model_context=None,
        opening_revision=None,
        model_context_lifecycle=None,
        prepared_model_ids=None,
    ):
        parameters = segment_parameters(command)
        if prepared_model_ids is not None:
            if model_context is None or set(prepared_model_ids) != {
                "model_session_id",
                "generation_id",
            }:
                raise RenderError("REQUEST_ID_CONFLICT")
            prepared_model_ids = {
                key: identity(value) for key, value in prepared_model_ids.items()
            }

        def transition(state, operation):
            inference = state.get("inference", {})
            if not inference.get("enabled", True) or (
                opening_revision is not None
                and inference.get("revision") != opening_revision
            ):
                raise RenderError("MODEL_UNAVAILABLE")
            self.require_revision(state, command)
            if state["source_state"] != "ready":
                raise RenderError("INVALID_SOURCE")
            if state["pending_replace"]:
                raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
            current = state["segment"]
            occupied = current and (
                current["state"] in OCCUPIED_SEGMENTS
                or current["resource_state"] not in {"none", "released"}
            )
            if occupied and not replace:
                raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
            if replace and occupied:
                self.require_segment(state, command)
            state["render_revision"] += 1
            segment = dict(
                segment_id=uuid4().hex,
                segment_epoch=uuid4().hex,
                model_session_id=uuid4().hex,
                generation_id=uuid4().hex,
                state="accepted",
                resource_state="none",
                recording_state="recording",
                prompt_version=1,
                accepted_version=1,
                applied_version=0,
                rebase_required=False,
                prompt_queue=[],
                prompt_events={},
                first_decodable=False,
                request_id=command["request_id"],
                render_revision=state["render_revision"],
                **parameters,
            )
            if startup_ids is not None:
                # Only Room-owned startup or a journal-reserved context handoff
                # supplies these; clients cannot choose model/session identities.
                context_handoff = (
                    model_context is not None
                    and startup_ids == state.get("model_context_segment_ids")
                )
                if (
                    replace
                    or not context_handoff
                    and state["render_revision"] not in {1, 2}
                    or (
                        current is not None
                        and (
                            current["state"] not in TERMINAL_SEGMENTS
                            or current["resource_state"] not in {"none", "released"}
                        )
                    )
                ):
                    raise RenderError("REQUEST_ID_CONFLICT")
                for key in (
                    "segment_id",
                    "segment_epoch",
                    "model_session_id",
                    "generation_id",
                ):
                    identity(startup_ids[key])
                    segment[key] = startup_ids[key]
                if prepared_model_ids is not None:
                    if not context_handoff:
                        raise RenderError("REQUEST_ID_CONFLICT")
                    state["model_context_segment_ids"] = {
                        **startup_ids,
                        **prepared_model_ids,
                    }
                    segment.update(prepared_model_ids)
            elif prepared_model_ids is not None:
                raise RenderError("REQUEST_ID_CONFLICT")
            if occupied:
                state["pending_replace"] = segment
                current["state"] = "draining"
            else:
                state["segment"] = segment
            if model_context is not None:
                segment["prompt_events"] = {
                    event: dict(request_id=command["request_id"], state="accepted")
                    for event in command["one_shot_event_ids"]
                }
                segment["context_segment_id"] = model_context["segmentId"]
                segment["context_references"] = copy.deepcopy(parameters["references"])
                segment["source_context"] = copy.deepcopy(command["source_context"])
                operation.update(
                    model_context=copy.deepcopy(model_context),
                    model_context_effective=copy.deepcopy(
                        command["model_context_effective"]
                    ),
                    prompt_version=1,
                )
                if "model_context_instruction" in command:
                    operation["model_context_instruction"] = copy.deepcopy(
                        command["model_context_instruction"]
                    )
                if model_context_lifecycle is not None:
                    operation["model_context_lifecycle"] = copy.deepcopy(
                        model_context_lifecycle
                    )
            operation.update(
                segment_id=segment["segment_id"],
                segment_epoch=segment["segment_epoch"],
                render_revision=state["render_revision"],
                old_segment_id=current["segment_id"] if occupied else None,
            )

        return await self.command(command, transition)

    async def end(self, command):
        if (
            command.get("mode", "drain") not in {"drain", "cancel"}
            or type(command.get("cancel_pending_replace", False)) is not bool
        ):
            raise RenderError("INVALID_COMMAND")

        def transition(state, operation):
            self.require_revision(state, command)
            segment = self.require_segment(state, command)
            cancelled = None
            if command.get("cancel_pending_replace") and state["pending_replace"]:
                cancelled = state["pending_replace"]["segment_id"]
                pending = state["pending_replace"]
                state["operations"][pending["request_id"]].update(
                    state="cancelled",
                    code="REPLACEMENT_CANCELLED",
                    new_started=False,
                )
                state["pending_replace"] = None
            state["render_revision"] += 1
            if segment["state"] not in TERMINAL_SEGMENTS:
                segment["state"] = "draining"
            operation.update(
                segment_id=segment["segment_id"],
                segment_epoch=segment["segment_epoch"],
                render_revision=state["render_revision"],
                cancelled_pending_segment=cancelled,
                mode=command.get("mode", "drain"),
            )

        return await self.command(command, transition)

    async def display(self, command):
        if command.get("mode") not in {"whitebox", "model"}:
            raise RenderError("INVALID_COMMAND")

        def transition(state, operation):
            revision = command.get("expected_display_revision")
            if type(revision) is not int or revision != state["display_revision"]:
                raise RenderError("REVISION_CONFLICT")
            state["display_revision"] += 1
            state["display_mode"] = command["mode"]
            operation.update(
                state="applied",
                display_revision=state["display_revision"],
                mode=command["mode"],
            )

        return await self.command(command, transition)

    async def prompt(
        self,
        command,
        *,
        model_context=None,
        player_instruction=False,
        model_context_lifecycle=None,
    ):
        text = None if player_instruction else prompt_text(command.get("prompt"))
        effective = (
            dict(source_frame_index=0, policy="at_or_after")
            if player_instruction
            else boundary(command.get("effective"))
        )
        binding = area_binding(command)
        references = (
            reference_parameters(command["references"])
            if "references" in command
            else None
        )
        source_policy = command.get("source_policy")
        if source_policy not in (None, "skip_during_switch"):
            raise RenderError("INVALID_COMMAND")
        if source_policy and effective.get("policy") == "strict":
            raise RenderError("EFFECTIVE_BOUNDARY_UNAVAILABLE")
        events = command.get("one_shot_event_ids", [])
        if (
            not isinstance(events, list)
            or len(events) > 32
            or len(set(map(identity, events))) != len(events)
        ):
            raise RenderError("INVALID_COMMAND")

        def transition(state, operation):
            if not state.get("inference", {}).get("enabled", True):
                raise RenderError("MODEL_UNAVAILABLE")
            segment = self.require_segment(state, command)
            if segment["state"] not in {"initializing", "streaming"}:
                raise RenderError("MODEL_UNAVAILABLE")
            next_text = text
            if player_instruction:
                if state.get("inference", {}).get("state") != "enabled":
                    raise RenderError("MODEL_UNAVAILABLE")
                current = segment.get(
                    "player_instruction", dict(revision=0, prompt=None)
                )
                if command["expectedRevision"] != current["revision"]:
                    operation.update(
                        state="completed",
                        instruction_result=dict(
                            status="conflict", **current, code="REVISION_CONFLICT"
                        ),
                    )
                    return
                original = segment.setdefault(
                    "instruction_base_prompt", segment["prompt"]
                )
                next_text = prompt_text(
                    original
                    + (
                        "\n\n" + command["prompt"]
                        if command["prompt"] is not None
                        else ""
                    )
                )
            expected, version = (
                command.get("expected_prompt_version"),
                command.get("prompt_version"),
            )
            base_applied = command.get("base_applied_prompt_version")
            if model_context is not None or player_instruction:
                if (
                    model_context is not None
                    and segment.get("context_segment_id") != model_context["segmentId"]
                ):
                    raise RenderError("SEGMENT_EPOCH_MISMATCH")
                expected, version = (
                    segment["accepted_version"],
                    segment["accepted_version"] + 1,
                )
                # Trusted conditions replace the complete context. Resolve their
                # recovery base in this CAS, without replaying failed onset IDs.
                base_applied = segment["applied_version"]
            if (
                type(expected) is not int
                or expected != segment["accepted_version"]
                or type(version) is not int
                or version != expected + 1
            ):
                raise RenderError("REVISION_CONFLICT")
            if segment["rebase_required"]:
                if base_applied != segment["applied_version"]:
                    raise RenderError("REVISION_CONFLICT")
            queue = segment["prompt_queue"]
            if queue and queue[0].get("periodic_refresh"):
                automatic = self.prompt_operation(state, segment, queue[0])
                if automatic["state"] == "accepted":
                    queue.pop(0)
                    automatic.update(state="cancelled", never_applied=True)
                    # Keep the admitted revision: the real update below owns
                    # the next version; native ingress permits skipped versions.
            if (
                len(queue) >= MAX_PENDING_PROMPTS
                or sum(len(item["text"].encode()) for item in queue)
                + len(next_text.encode())
                > MAX_PENDING_PROMPT_BYTES
            ):
                raise RenderError("LIMIT_EXCEEDED")
            if queue:
                previous = queue[-1]["effective"]
                key = (
                    "source_frame_index"
                    if "source_frame_index" in effective
                    else "source_pts_us"
                )
                if key not in previous or effective[key] < previous[key]:
                    raise RenderError("EFFECTIVE_BOUNDARY_UNAVAILABLE")
            retry = command.get("retry_of_request_id")
            for event in events:
                old = segment["prompt_events"].get(event)
                if old and not (
                    old["state"] == "never_applied" and retry == old["request_id"]
                ):
                    raise RenderError("REQUEST_ID_CONFLICT")
            # Strict acceptance is finalized by the worker boundary reservation,
            # never by this control-plane queue alone.
            validating = effective.get("policy") == "strict"
            if segment.get("validating_prompt"):
                raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
            item = dict(
                request_id=command["request_id"],
                version=version,
                text=next_text,
                sha256=hashlib.sha256(next_text.encode()).hexdigest(),
                effective=effective,
                one_shot_event_ids=list(events),
                **copy.deepcopy(binding),
            )
            segment["accepted_area_binding"] = copy.deepcopy(
                binding.get("area_binding")
            )
            if model_context is not None:
                item["source_context"] = copy.deepcopy(command["source_context"])
                operation["model_context"] = copy.deepcopy(model_context)
                operation["model_context_effective"] = copy.deepcopy(
                    command["model_context_effective"]
                )
                if "model_context_instruction" in command:
                    operation["model_context_instruction"] = copy.deepcopy(
                        command["model_context_instruction"]
                    )
                if model_context_lifecycle is not None:
                    operation["model_context_lifecycle"] = copy.deepcopy(
                        model_context_lifecycle
                    )
                segment["context_references"] = copy.deepcopy(
                    command["context_references"]
                )
            if references is not None:
                item["references"] = references
            if source_policy is not None:
                item["source_policy"] = source_policy
            if validating:
                segment["validating_prompt"] = item
                operation["state"] = "validating"
            else:
                queue.append(item)
                segment["accepted_version"] = version
                segment["rebase_required"] = False
                for event in events:
                    segment["prompt_events"][event] = dict(
                        request_id=command["request_id"], state="accepted"
                    )
            operation.update(
                segment_id=segment["segment_id"],
                segment_epoch=segment["segment_epoch"],
                prompt_version=version,
                prompt_sha256=item["sha256"],
                requested_effective=effective,
            )
            if player_instruction:
                segment["player_instruction"] = dict(
                    revision=current["revision"] + 1, prompt=command["prompt"]
                )
                operation["instruction_result"] = dict(
                    status="accepted", **segment["player_instruction"]
                )

        return await self.command(command, transition)

    async def player_instruction(self, command):
        """Compose a plain source's text supplement in its original prompt CAS.

        The frozen initial prompt and references stay authoritative. Reads do
        not grow the journal; writes retain its bounded original request IDs.
        """
        state = await self.read()
        self.validate_scope(state, command)
        if command.get("op") not in {
            "model.context.intervention",
            "model.context.intervention.get",
        }:
            raise RenderError("INVALID_COMMAND")
        if not state.get("manual_inference_journal") or state.get(
            "model_context_segment_ids"
        ):
            raise RenderError("CAPABILITY_UNAVAILABLE")
        if (
            state["source_state"] != "ready"
            or state.get("inference", {}).get("state") != "enabled"
            or not state["inference"]["enabled"]
        ):
            raise RenderError("MODEL_UNAVAILABLE")
        segment = self.require_segment(state, command)
        if segment["state"] not in {"initializing", "streaming"}:
            raise RenderError("MODEL_UNAVAILABLE")
        if command["op"] == "model.context.intervention.get":
            return dict(
                status="current",
                **segment.get("player_instruction", dict(revision=0, prompt=None)),
            )
        if (
            type(command.get("expectedRevision")) is not int
            or not 0 <= command["expectedRevision"] <= 2**53 - 1
            or "prompt" not in command
        ):
            raise RenderError("INVALID_COMMAND")
        if command["prompt"] is not None:
            prompt_text(command["prompt"])
        result = await self.prompt(command, player_instruction=True)
        return {
            **result["instruction_result"],
            **(
                {"status": "duplicate"}
                if result.get("duplicate")
                and result["instruction_result"]["status"] == "accepted"
                else {}
            ),
        }

    async def reserve_model_context_segment(self, startup_ids=None):
        """Reserve the original segment identity before subscribing, without init."""

        def reserve(state):
            current = state.get("segment")
            if current is not None and not current.get("context_segment_id"):
                raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
            ids = state.get("model_context_segment_ids")
            if ids is None:
                names = (
                    "segment_id",
                    "segment_epoch",
                    "model_session_id",
                    "generation_id",
                )
                ids = {
                    name: identity(startup_ids[name]) if startup_ids else uuid4().hex
                    for name in names
                }
                state["model_context_segment_ids"] = ids
            return ids

        return await self.update(reserve)

    @staticmethod
    def _model_context_unresolved(operation):
        return operation.get("model_context_execution", {}).get(
            "state", operation["state"]
        ) not in {"applied", "failed", "rejected", "cancelled"}

    async def retire_model_context_source(self, segment_id, reset):
        """Record authenticated source retirement after original execution cleanup.

        Application may remain unknown forever. A verified retired child cannot
        affect the new SDK source, and no original command is replayed or relabelled.
        """

        def retire(state):
            current = state.get("segment")
            if not current or current["segment_id"] != segment_id:
                raise RenderError("SEGMENT_EPOCH_MISMATCH")
            if current["state"] not in TERMINAL_SEGMENTS or current[
                "resource_state"
            ] not in {"none", "released"}:
                raise RenderError("RELEASE_UNKNOWN")
            source = current.get("source_context", {}).get("source", {})
            if (
                set(reset) != {"type", "epoch", "presentationId", "sdkEpoch"}
                or reset["type"] != "source.reset"
                or type(reset["sdkEpoch"]) is not int
                or type(source.get("sdkEpoch")) is not int
                or reset["sdkEpoch"] <= source["sdkEpoch"]
                or reset["epoch"] != source.get("epoch")
            ):
                raise RenderError("SOURCE_EPOCH_MISMATCH")
            for operation in state["operations"].values():
                context = operation.get("model_context", {})
                if (
                    context.get("segmentId") == segment_id
                    and context.get("source") == source
                    and self._model_context_unresolved(operation)
                ):
                    # Proven history already passes successor admission. Only
                    # unresolved execution needs this old-source exemption.
                    operation.setdefault(
                        "model_context_retirement",
                        dict(
                            segment_id=segment_id,
                            segment_epoch=current["segment_epoch"],
                            source_reset=copy.deepcopy(reset),
                        ),
                    )

        return await self.update(retire, settlement=True)

    async def start_model_context_segment(self, command):
        """Explicit source-owner intent after EOS, without accepting prompt input.

        Reservation is idempotent in the original journal. The next complete
        Host condition performs init only after the old model lease is released.
        """
        if (
            set(command)
            - {
                "type",
                "version",
                "op",
                "request_id",
                "source_session_id",
                "source_epoch",
                "expected_render_revision",
            }
            or command.get("op") != "render.context.start"
        ):
            raise RenderError("INVALID_COMMAND")

        def reserve(state, operation):
            self.require_revision(state, command)
            if state["source_state"] != "ready":
                raise RenderError("INVALID_SOURCE")
            current = state.get("segment")
            ids = state.get("model_context_segment_ids")
            if (
                not current
                or not current.get("context_segment_id")
                or current["state"] not in TERMINAL_SEGMENTS
                or current["resource_state"] not in {"none", "released"}
                or state["pending_replace"]
                or not ids
                or ids["segment_id"] != current["segment_id"]
            ):
                raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
            if any(
                item.get("model_context", {}).get("segmentId") == current["segment_id"]
                and self._model_context_unresolved(item)
                and not (
                    item.get("model_context_retirement", {}).get("segment_id")
                    == current["segment_id"]
                    and item.get("model_context_retirement", {}).get("segment_epoch")
                    == current["segment_epoch"]
                )
                for item in state["operations"].values()
            ):
                # Ending alone cannot resolve a lost application result. Only
                # verified source retirement permits a fresh SDK generation;
                # its original request remains queryable with an unknown outcome.
                raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
            ids = {
                name: uuid4().hex
                for name in (
                    "segment_id",
                    "segment_epoch",
                    "model_session_id",
                    "generation_id",
                )
            }
            state["model_context_segment_ids"] = ids
            state["render_revision"] += 1
            operation.update(**ids, render_revision=state["render_revision"])

        return await self.command(command, reserve)

    @staticmethod
    def model_condition_request(value):
        # The original journal owns request allocation. Payload hash deliberately
        # stays outside the request key so conflicting content cannot mint a retry.
        key = {
            k: value[k]
            for k in (
                "protocol",
                "source",
                "audience",
                "segmentId",
                "configHash",
                "compilerVersion",
                "conditionId",
            )
        }
        return "model-context:" + hashlib.sha256(canonical(key)).hexdigest()

    async def model_condition_query(self, query):
        state = await self.read()
        request = self.model_condition_request(query)
        operation = state["operations"].get(request)
        if operation is None:
            return None
        context = operation.get("model_context")
        expected = {
            k: v for k, v in query.items() if k not in {"requestId", "promptVersion"}
        }
        if context != expected or (
            "requestId" in query
            and (
                query["requestId"] != request
                or query["promptVersion"] != operation["prompt_version"]
            )
        ):
            raise RenderError("REQUEST_ID_CONFLICT")
        return operation

    async def model_condition(
        self,
        condition,
        references,
        *,
        seed=42,
        source_first=None,
        opening_revision=None,
        startup_prepared=True,
        first_presented=False,
        prepared_model_ids=None,
    ):
        from .source_conditions import request_identity

        context = request_identity(condition)
        previous = await self.model_condition_query(context)
        if previous:
            return dict(previous, duplicate=True)
        state = await self.read()
        segment = state["segment"]
        command = dict(
            source_session_id=self.source_session_id,
            source_epoch=self.source_epoch,
            request_id=self.model_condition_request(context),
            prompt=condition["prompt"],
            model_context=context,
            **(
                {"model_context_instruction": copy.deepcopy(condition["instruction"])}
                if "instruction" in condition
                else {}
            ),
            one_shot_event_ids=condition["onsetOccurrenceIds"],
            model_context_effective=copy.deepcopy(condition["effective"]),
            source_context={
                "source": condition["source"],
                "effective": condition["effective"],
            },
        )
        if (
            segment is None
            or segment.get("context_segment_id") != condition["segmentId"]
        ):
            ids = state.get("model_context_segment_ids")
            if ids is None or ids["segment_id"] != condition["segmentId"]:
                raise RenderError("SEGMENT_EPOCH_MISMATCH")
            if not startup_prepared:
                raise RenderError("MODEL_UNAVAILABLE")
            command.update(
                op="render.start",
                expected_render_revision=state["render_revision"],
                references=references,
                seed=seed,
            )
            if source_first is not None:
                command["start_boundary"] = {"source_frame_index": source_first}
            if (
                segment is not None
                and segment.get("context_segment_id")
                and segment["resource_state"] == "released"
            ):
                # An explicit successor may see the previous worker's busy
                # heartbeat after verified cleanup. Use original bounded admit,
                # with the same reserved identity; initial admission stays fast.
                command.update(
                    wait_for_capacity=True, capacity_wait_ms=MAX_CAPACITY_WAIT_MS
                )
            return await self.start(
                command,
                startup_ids=ids,
                model_context=context,
                prepared_model_ids=prepared_model_ids,
                **(
                    {"model_context_lifecycle": condition["lifecycle"]}
                    if first_presented
                    else {}
                ),
                **(
                    {"opening_revision": opening_revision}
                    if opening_revision is not None
                    else {}
                ),
            )
        if prepared_model_ids is not None:
            raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
        if segment.get("context_segment_id") != condition["segmentId"]:
            raise RenderError("SEGMENT_EPOCH_MISMATCH")
        command.update(
            op="prompt.replace",
            segment_id=segment["segment_id"],
            segment_epoch=segment["segment_epoch"],
            effective=dict(source_frame_index=0, policy="at_or_after"),
            source_context={
                "source": condition["source"],
                "effective": condition["effective"],
            },
            context_references=references,
            one_shot_event_ids=condition["onsetOccurrenceIds"],
        )
        if references != segment.get("context_references", segment["references"]):
            command["references"] = references
        return await self.prompt(
            command,
            model_context=context,
            **(
                {"model_context_lifecycle": condition["lifecycle"]}
                if first_presented
                else {}
            ),
        )

    async def confirm_opening_presentation(self, command, resolve_offer):
        from .opening_presentation import retained_first_presented

        def confirm(state, operation):
            opening = resolve_offer(command, state)
            prior = retained_first_presented(state)
            if prior is not None and prior["first_presented"] == "pending":
                prior["first_presented"] = "rejected"
                state["operations"][prior["first_presented_confirmation_id"]].update(
                    state="rejected", code="MODEL_PRESENTATION_STALE"
                )
            opening["first_presented"] = "pending"
            opening["first_presented_confirmation_id"] = operation["request_id"]
            state["model_context_first_presented_request_id"] = opening["request_id"]

        return await self.command(command, confirm)

    async def acknowledge_opening_presentation(
        self, request_id, condition_id, status, *, confirmation_id=None
    ):
        from .opening_presentation import (
            applied_context_operation,
            retained_first_presented,
        )

        def still_current(state, opening):
            current = applied_context_operation(state)
            return (
                current is not None
                and opening["model_context"]["segmentId"]
                == current["model_context"]["segmentId"]
                and opening["model_context_lifecycle"]
                == current["model_context_lifecycle"]
            )

        def pending(state):
            opening = retained_first_presented(state)
            if (
                opening is not None
                and opening["first_presented"] == "pending"
                and opening["request_id"] == request_id
                and opening["model_context"]["conditionId"] == condition_id
                and (
                    confirmation_id is None
                    or opening["first_presented_confirmation_id"] == confirmation_id
                )
                # Success requires the current lifecycle. An invalidated
                # original confirmation still needs a negative terminal result.
                and (
                    status not in {"accepted", "duplicate"}
                    or still_current(state, opening)
                )
            ):
                return opening
            return None

        opening = pending(await self.read())
        if opening is None:
            return None
        confirmation_id = opening["first_presented_confirmation_id"]

        def acknowledge(state):
            opening = pending(state)
            if opening is None:
                return None
            confirmation = state["operations"][
                opening["first_presented_confirmation_id"]
            ]
            if status in {"accepted", "duplicate"}:
                opening["first_presented"] = "acknowledged"
                confirmation.update(state="completed")
            elif status == "timeout" and still_current(state, opening):
                opening["first_presented"] = "expired"
                confirmation.update(
                    state="failed", code="MODEL_PRESENTATION_UNCONFIRMED"
                )
            else:
                opening["first_presented"] = "rejected"
                confirmation.update(state="rejected", code="MODEL_PRESENTATION_STALE")
            return confirmation

        return await self.update(acknowledge, settlement=True)

    @staticmethod
    def prompt_operation(state, segment, item):
        if item.get("periodic_refresh"):
            operation = segment.get("periodic_refresh", {}).get("operation")
            if operation is None or operation["request_id"] != item["request_id"]:
                raise RenderError("REQUEST_ID_CONFLICT")
            return operation
        return state["operations"][item["request_id"]]

    async def enqueue_periodic_refresh(
        self, segment_id, segment_epoch, version, *, area=None
    ):
        """One bounded internal update; never consumes user operation capacity."""

        def enqueue(state):
            segment = self.require_segment(
                state, dict(segment_id=segment_id, segment_epoch=segment_epoch)
            )
            previous = segment.get("periodic_refresh", {})
            op = previous.get("operation", {})
            if (
                state["source_state"] != "ready"
                or segment["state"] != "streaming"
                or not state.get("inference", {}).get("enabled", True)
                or segment["applied_version"] != version
                or segment["accepted_version"] != version
                or segment["prompt_queue"]
                or segment.get("validating_prompt")
                or segment["rebase_required"]
                or state.get("pending_replace")
                or segment.get("sink", {}).get("pending_request_id")
                or op.get("state") in {"accepted", "dispatching", "reconciling"}
            ):
                return None
            sequence = previous.get("sequence", 0) + 1
            request_id = "periodic-" + uuid4().hex
            item = dict(
                request_id=request_id,
                version=version + 1,
                text=segment["prompt"],
                sha256=segment["prompt_sha256"],
                effective=dict(source_frame_index=0, policy="at_or_after"),
                one_shot_event_ids=[],
                periodic_refresh=True,
            )
            if area is not None:
                item["area_binding"] = copy.deepcopy(area)
            operation = dict(
                op="prompt.replace",
                request_id=request_id,
                state="accepted",
                periodic_refresh=True,
                prompt_version=version + 1,
                segment_id=segment_id,
                segment_epoch=segment_epoch,
            )
            segment["periodic_refresh"] = dict(
                sequence=sequence,
                completed=previous.get("completed", 0),
                operation=operation,
            )
            segment["prompt_queue"].append(item)
            segment["accepted_version"] = version + 1
            return operation

        return await self.update(enqueue)

    async def mark_periodic_dispatch(self, segment_id, request_id):
        def mark(state):
            segment = state.get("segment")
            if not segment or segment["segment_id"] != segment_id:
                return False
            operation = segment.get("periodic_refresh", {}).get("operation")
            if (
                not operation
                or operation["request_id"] != request_id
                or operation["state"] != "accepted"
            ):
                return False
            operation["state"] = "dispatching"
            return True

        return await self.update(mark)

    async def cancel_periodic_refresh(self, segment_id):
        def cancel(state):
            segment = state.get("segment")
            if not segment or segment["segment_id"] != segment_id:
                return False
            operation = segment.get("periodic_refresh", {}).get("operation", {})
            if operation.get("state") != "accepted":
                return False
            segment["prompt_queue"] = [
                p
                for p in segment["prompt_queue"]
                if p["request_id"] != operation["request_id"]
            ]
            if not segment["prompt_queue"]:
                segment["accepted_version"] = segment["applied_version"]
            operation.update(state="cancelled", never_applied=True)
            return True

        return await self.update(cancel, settlement=True)

    async def prompt_reserved(self, segment_id, request_id, *, accepted, code=None):
        def transition(state):
            segment = state["segment"]
            if not segment or segment["segment_id"] != segment_id:
                raise RenderError("SEGMENT_EPOCH_MISMATCH")
            item = segment.get("validating_prompt")
            if not item or item["request_id"] != request_id:
                raise RenderError("REQUEST_ID_CONFLICT")
            segment["validating_prompt"] = None
            operation = state["operations"][request_id]
            if accepted:
                if item["version"] != segment["accepted_version"] + 1:
                    raise RenderError("REVISION_CONFLICT")
                segment["prompt_queue"].append(item)
                segment["accepted_version"] = item["version"]
                segment["rebase_required"] = False
                for event in item["one_shot_event_ids"]:
                    segment["prompt_events"][event] = dict(
                        request_id=request_id, state="accepted"
                    )
                operation["state"] = "accepted"
            else:
                operation.update(
                    state="rejected",
                    code=code or "EFFECTIVE_BOUNDARY_UNAVAILABLE",
                    never_applied=True,
                )
            return operation

        return await self.update(transition, settlement=True)

    async def prompt_applied(self, segment_id, version, evidence):
        if (
            type(evidence.get("first_output_frame")) is not int
            or type(evidence.get("num_frames")) is not int
            or evidence["num_frames"] < 1
        ):
            raise RenderError("INVALID_COMMAND")

        def transition(state):
            segment = state["segment"]
            if not segment or segment["segment_id"] != segment_id:
                raise RenderError("SEGMENT_EPOCH_MISMATCH")
            if version == 1 and segment["applied_version"] == 0:
                segment["applied_version"] = segment["prompt_version"] = 1
                operation = state["operations"][segment["request_id"]]
                if operation.get("model_context") is not None:
                    for event in segment["prompt_events"].values():
                        if event["request_id"] == operation["request_id"]:
                            event["state"] = "applied"
                    # The start operation also tracks segment lifetime. Keep
                    # proven condition execution monotonic through EOS/failure.
                    operation["model_context_execution"] = dict(
                        state="applied", **copy.deepcopy(evidence)
                    )
                    operation.update(state="applied", **evidence)
                    if "model_context_lifecycle" in operation:
                        segment["model_context_applied_request_id"] = operation[
                            "request_id"
                        ]
                    return operation
                return dict(prompt_version=1, **evidence)
            if (
                not segment["prompt_queue"]
                or segment["prompt_queue"][0]["version"] != version
            ):
                raise RenderError("REVISION_CONFLICT")
            item = segment["prompt_queue"].pop(0)
            periodic = item.get("periodic_refresh", False)
            if periodic:
                segment.setdefault(
                    "refresh_condition_version", segment["applied_version"]
                )
            else:
                segment.pop("refresh_condition_version", None)
            segment.update(
                applied_version=version,
                prompt_version=version,
                prompt=item["text"],
                prompt_sha256=item["sha256"],
            )
            if "references" in item:
                segment["references"] = copy.deepcopy(item["references"])
            operation = self.prompt_operation(state, segment, item)
            operation.update(state="applied", **evidence)
            if periodic:
                segment["periodic_refresh"]["completed"] += 1
                return operation
            if "model_context_lifecycle" in operation:
                segment["model_context_applied_request_id"] = operation["request_id"]
            else:
                segment.pop("model_context_applied_request_id", None)
            for event in item["one_shot_event_ids"]:
                segment["prompt_events"][event]["state"] = "applied"
            return operation

        return await self.update(transition, settlement=True)

    async def prompt_failed(self, segment_id, version, *, never_applied, code):
        def transition(state):
            segment = state["segment"]
            if not segment or segment["segment_id"] != segment_id:
                raise RenderError("SEGMENT_EPOCH_MISMATCH")
            queue = segment["prompt_queue"]
            if not queue or queue[0]["version"] != version:
                raise RenderError("REVISION_CONFLICT")
            first = self.prompt_operation(state, segment, queue[0])
            if not never_applied:
                first.update(state="reconciling", code=code)
                return first
            for index, item in enumerate(queue):
                operation = self.prompt_operation(state, segment, item)
                operation.update(
                    state="failed",
                    code=code if index == 0 else "DEPENDENCY_FAILED",
                    never_applied=True,
                )
                for event in item["one_shot_event_ids"]:
                    segment["prompt_events"][event]["state"] = "never_applied"
            queue.clear()
            if "context_references" in segment:
                # Dropped updates never changed the model's applied image set.
                segment["context_references"] = copy.deepcopy(segment["references"])
            segment["rebase_required"] = True
            return first

        return await self.update(transition, settlement=True)


class SQLiteRenderStore:
    """Durable Room-owned control journal on the existing recording PVC.

    FULL synchronous WAL commits precede accepted replies. A separate bounded
    executor keeps these small control writes away from frame and archive work.
    The source owner epoch is still authoritative after a Room restart: rows are
    queryable evidence, not instructions to reconstruct a lost CLI or H3 session.
    """

    def __init__(self, path, *, clock=time.time):
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="render-control"
        )
        self.admission = asyncio.Semaphore(8)
        self.closed = False
        self._initialize()

    @contextmanager
    def _connect(self):
        import sqlite3

        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            page_size = connection.execute("PRAGMA page_size").fetchone()[0]
            connection.execute("PRAGMA max_page_count=" + str((64 << 20) // page_size))
            connection.execute("PRAGMA journal_size_limit=4194304")
            yield connection
        finally:
            connection.close()

    def _initialize(self):
        import os

        with self._connect() as connection:
            mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if mode != "wal":
                raise RuntimeError("Render journal requires durable WAL storage")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS render_v2 (key TEXT PRIMARY KEY, value TEXT NOT NULL, expires_at REAL NOT NULL)"
            )
        os.chmod(self.path, 0o600)

    async def _run(self, fn):
        import asyncio
        import sqlite3

        if self.closed:
            raise RenderError("CONTROL_UNAVAILABLE")
        async with self.admission:
            try:
                return await asyncio.get_running_loop().run_in_executor(
                    self.executor, fn
                )
            except sqlite3.Error as error:
                raise RenderError("CONTROL_UNAVAILABLE", retryable=True) from error

    async def find_source(self, source_id):
        identity(source_id)

        def read():
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT value FROM render_v2 WHERE substr(key, -?)=? AND expires_at>? LIMIT 2",
                    (len(source_id) + 1, ":" + source_id, self.clock()),
                ).fetchall()
                return json.loads(rows[0][0]) if len(rows) == 1 else None

        return await self._run(read)

    async def get(self, key):
        def read():
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT value, expires_at FROM render_v2 WHERE key=?", (key,)
                ).fetchone()
                return row[0] if row and row[1] > self.clock() else None

        return await self._run(read)

    async def cas(self, key, old, new, retention):
        if not isinstance(new, str) or len(new.encode()) > 2 << 20 or retention < 86400:
            raise RenderError("LIMIT_EXCEEDED")

        def write():
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                # Reclaim only terminal source journals after the retention
                # horizon. An unresolved owner/lease is never expired into reuse.
                connection.execute(
                    "DELETE FROM render_v2 WHERE expires_at<? AND json_extract(value,'$.source_state')='ended'",
                    (self.clock(),),
                )
                row = connection.execute(
                    "SELECT value, expires_at FROM render_v2 WHERE key=?", (key,)
                ).fetchone()
                current = row[0] if row and row[1] > self.clock() else None
                if current != old or (row and current is None):
                    connection.execute("ROLLBACK")
                    return False
                connection.execute(
                    "INSERT INTO render_v2(key,value,expires_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, expires_at=excluded.expires_at",
                    (key, new, self.clock() + retention),
                )
                connection.execute("COMMIT")
                return True

        return await self._run(write)

    async def close(self):
        import asyncio

        self.closed = True
        await asyncio.to_thread(self.executor.shutdown, wait=True)
