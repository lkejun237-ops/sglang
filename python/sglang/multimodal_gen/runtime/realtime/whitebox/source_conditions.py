# SPDX-License-Identifier: Apache-2.0
"""Trusted Worldkit conditions. Preparation owns no model version or side effect.

Wire field order is worldkit-model-context-v1's UTF-8 netstring contract. The
shared SDK fixture checks this boundary; RenderJournal remains the only ledger.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import logging
import math
import re
from collections.abc import Awaitable, Callable

from .opening_presentation import PROTOCOL as FIRST_PRESENTED_PROTOCOL
from .opening_presentation import applied_context_operation, retained_first_presented
from .render_contract import (
    MAX_REFERENCE_BYTES,
    MAX_REFERENCE_IMAGES,
    RenderError,
    canonical,
    identity,
    prompt_text,
)

PROTOCOL = "worldkit-model-context-v1"
ENVELOPE = (
    "protocol",
    "source",
    "audience",
    "segmentId",
    "configHash",
    "compilerVersion",
)
SOURCE = (
    "sessionId",
    "epoch",
    "gameBuildHash",
    "worldId",
    "sourceEpoch",
    "presentationId",
    "sdkEpoch",
)
AUDIENCE = ("consumerId", "consumerEpoch", "playerId", "viewId")
HANDLE = (
    "sessionId",
    "epoch",
    "gameBuildHash",
    "configHash",
    "compilerVersion",
    "resource",
    "sha256",
)
PAYLOAD = (
    *ENVELOPE,
    "lifecycle",
    "prompt",
    "references",
    "onsetOccurrenceIds",
    "ongoingOccurrenceIds",
)
CONDITION = (
    *PAYLOAD,
    "type",
    "conditionId",
    "payloadHash",
    "controlSequence",
    "stateFingerprint",
    "contextRevision",
    "viewRevision",
    "effective",
)


def invalid(code="MODEL_CONTEXT_PROTOCOL_INVALID"):
    return ValueError(code)


def keys(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise invalid()


def integer(value, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**53 - 1:
        raise invalid()


def digest(value):
    if not isinstance(value, str) or not re.fullmatch("[a-f0-9]{64}", value):
        raise invalid()


def envelope(value):
    result = {key: copy.deepcopy(value[key]) for key in ENVELOPE}
    if (
        result["protocol"] != PROTOCOL
        or type(result["compilerVersion"]) is not int
        or result["compilerVersion"] != 1
    ):
        raise invalid()
    digest(result["configHash"])
    identity(result["segmentId"])
    source, audience = result["source"], result["audience"]
    keys(source, SOURCE)
    keys(audience, AUDIENCE)
    for key in SOURCE:
        (
            integer
            if key in ("epoch", "sdkEpoch")
            else digest
            if key == "gameBuildHash"
            else identity
        )(source[key])
    for v in audience.values():
        identity(v)
    return result


def effective(value):
    keys(value, ("sourceFrameId", "sourceTimeUs", "simulationTick", "policy"))
    for name in ("sourceFrameId", "sourceTimeUs", "simulationTick"):
        integer(value[name])
    if value["policy"] != "at-or-after":
        raise invalid()


def payload(value):
    p = {k: copy.deepcopy(value[k]) for k in PAYLOAD}
    envelope(p)
    keys(p["lifecycle"], ("lifecycleId", "generation"))
    identity(p["lifecycle"]["lifecycleId"])
    integer(p["lifecycle"]["generation"], 1)
    if not isinstance(p["prompt"], str):
        raise invalid()
    try:
        prompt_units = len(p["prompt"].encode("utf-16-le")) // 2
    except UnicodeEncodeError as error:
        raise invalid() from error
    if prompt_units > 1048576:
        raise invalid()
    if not isinstance(p["references"], list) or len(p["references"]) > 128:
        raise invalid()
    aliases, pictures = set(), set()
    for ref in p["references"]:
        keys(ref, ("alias", "picture", "format", "sha256", "resource"))
        identity(ref["alias"])
        integer(ref["picture"], 1)
        digest(ref["sha256"])
        h = ref["resource"]
        keys(h, HANDLE)
        if (
            ref["format"] not in ("png", "jpeg", "webp")
            or ref["alias"] in aliases
            or ref["picture"] in pictures
            or h
            != dict(
                sessionId=p["source"]["sessionId"],
                epoch=p["source"]["epoch"],
                gameBuildHash=p["source"]["gameBuildHash"],
                configHash=p["configHash"],
                compilerVersion=1,
                resource=ref["alias"],
                sha256=ref["sha256"],
            )
            or type(h["epoch"]) is not int
            or type(h["compilerVersion"]) is not int
        ):
            raise invalid()
        aliases.add(ref["alias"])
        pictures.add(ref["picture"])
    for name in ("onsetOccurrenceIds", "ongoingOccurrenceIds"):
        values = p[name]
        if not isinstance(values, list) or len(values) > 1024:
            raise invalid()
        for v in values:
            identity(v)
        if len(set(values)) != len(values):
            raise invalid()
    if set(p["onsetOccurrenceIds"]) & set(p["ongoingOccurrenceIds"]):
        raise invalid()
    return p


def encode_payload(value):
    p = payload(value)
    tokens = [
        p["protocol"],
        *(p["source"][k] for k in SOURCE),
        *(p["audience"][k] for k in AUDIENCE),
        p["segmentId"],
        p["configHash"],
        p["compilerVersion"],
        p["lifecycle"]["lifecycleId"],
        p["lifecycle"]["generation"],
        p["prompt"],
        len(p["references"]),
    ]
    for ref in p["references"]:
        tokens.extend(
            (
                ref["alias"],
                ref["picture"],
                ref["format"],
                ref["sha256"],
                *(ref["resource"][k] for k in HANDLE),
            )
        )
    for name in ("onsetOccurrenceIds", "ongoingOccurrenceIds"):
        tokens.extend((len(p[name]), *p[name]))
    encoded = [str(t).encode("utf-8") for t in tokens]
    return b"".join(str(len(v)).encode("ascii") + b":" + v for v in encoded)


def payload_hash(value):
    return hashlib.sha256(encode_payload(value)).hexdigest()


def parse_condition(value):
    keys(value, set(CONDITION) | ({"instruction"} if "instruction" in value else set()))
    if "instruction" in value:
        keys(value["instruction"], {"requestId", "revision"})
        identity(value["instruction"]["requestId"])
        integer(value["instruction"]["revision"], 1)
    if value["type"] != "model.condition":
        raise invalid()
    payload(value)
    identity(value["conditionId"])
    digest(value["payloadHash"])
    digest(value["stateFingerprint"])
    integer(value["controlSequence"], 1)
    integer(value["contextRevision"])
    integer(value["viewRevision"])
    effective(value["effective"])
    if payload_hash(value) != value["payloadHash"]:
        raise invalid("MODEL_CONTEXT_PAYLOAD_HASH")
    return copy.deepcopy(value)


def request_identity(value):
    return {
        **envelope(value),
        "conditionId": identity(value["conditionId"]),
        "payloadHash": value["payloadHash"],
    }


def source_reached(frame, condition):
    """Compare native evidence; never confuse capture numbering with model indices."""
    header = (frame.provenance or {}).get("cli_frame_header", {})
    source = header.get("source", {})
    expected = condition["source"]
    if (
        header.get("sessionId") != expected["sessionId"]
        or header.get("epoch") != expected["epoch"]
        or any(source.get(k) != expected[k] for k in ("presentationId", "sdkEpoch"))
    ):
        raise invalid("MODEL_CONTEXT_STALE_SOURCE")
    boundary = condition["effective"]
    return all(
        type(source.get(k)) is int and source[k] >= boundary[k]
        for k in ("sourceFrameId", "sourceTimeUs", "simulationTick")
    )


class SourceConditions:
    """One authenticated consumer; asynchronous preparation and original journal queries.

    The optional startup owner reads the condition and returns either no gate
    or a shared release future and its absolute event-loop deadline. A successful
    future means the window can release a prepared receipt. The owner settles
    the future and remains responsible for any work it started.
    """

    def __init__(
        self,
        *,
        envelope: dict,
        assets,
        journal,
        read_resource,
        emit,
        submit,
        capabilities,
        validate_reference,
        check,
        startup_release: (
            Callable[
                [dict],
                tuple[float, asyncio.Future[None]]
                | None
                | Awaitable[tuple[float, asyncio.Future[None]] | None],
            ]
            | None
        ) = None,
        startup_effective: Callable[[dict], dict | None] | None = None,
        first_presented_supported: Callable[[], bool] = lambda: False,
        first_presented_result: Callable[[dict], None] | None = None,
    ):
        self.envelope = globals()["envelope"](envelope)
        self.assets, self.journal = assets, journal
        self.read_resource, self.emit, self.submit = read_resource, emit, submit
        self.capabilities, self.check = capabilities, check
        self.validate_reference = validate_reference
        self.first_presented_supported = first_presented_supported
        self.first_presented_result = first_presented_result
        self.first_presented_reply = asyncio.Event()
        self.first_presented_task: asyncio.Task | None = None
        self.first_presented_request_id: str | None = None
        self.startup_release = startup_release
        self.startup_effective = startup_effective
        self.candidate = self.preparation = None
        self.preparations = set()
        self.closed = False
        self.debug = None
        self.instruction_receipt = None
        self.lock = asyncio.Lock()

    def validate(self, value):
        self.check()
        if self.closed or envelope(value) != self.envelope:
            raise invalid("MODEL_CONTEXT_STALE_SOURCE")
        identity(value["conditionId"])
        digest(value["payloadHash"])

    async def receipt(self, value, phase, operation=None, reason=None):
        self.check()
        message = dict(
            request_identity(value), type="model.condition.receipt", phase=phase
        )
        if operation is not None:
            message.update(
                requestId=operation["request_id"],
                promptVersion=operation["prompt_version"],
            )
        if reason:
            message["reason"] = reason
        if phase == "prepared" and self.candidate is not None:
            marker = self.candidate.get("startup_effective")
            if marker is not None:
                message["startupEffective"] = marker
        if phase == "first-presented":
            message["presentedLifecycle"] = operation["model_context_lifecycle"]
        await self.emit(message)
        condition = value
        if "prompt" not in condition and self.candidate is not None:
            condition = self.candidate["condition"]
        instruction = (operation or {}).get("model_context_instruction")
        if instruction is None and request_identity(condition) == request_identity(
            value
        ):
            instruction = condition.get("instruction")
        if (
            self.instruction_receipt is not None
            and phase in {"applied", "failed", "rejected"}
            and instruction is not None
        ):
            await self.instruction_receipt(instruction, phase, reason)
        if self.debug is not None:
            condition = value
            if "prompt" not in condition and self.candidate is not None:
                condition = self.candidate["condition"]
            if "prompt" in condition and request_identity(
                condition
            ) == request_identity(value):
                self.debug(condition, message)

    async def candidate_receipt(self, value, candidate):
        phase = {
            "preparing": "queued",
            "prepared": "prepared",
            "committed": "unknown",
            "rejected": "rejected",
        }[candidate["state"]]
        await self.receipt(value, phase, reason=candidate.get("reason"))

    async def original(self, value, *, query=None):
        # Reconciliation carries the original request/version; condition and
        # commit messages instead normalize to their fixed condition identity.
        operation = await self.journal.model_condition_query(
            request_identity(value) if query is None else query
        )
        if operation is None:
            return False
        if (
            value.get("type") == "model.condition.commit"
            and operation.get("model_context_effective") != value["effective"]
        ):
            raise invalid("MODEL_CONTEXT_COMMIT_CONFLICT")
        phase = {
            "accepted": "accepted",
            "applied": "applied",
            "failed": "failed",
            "rejected": "rejected",
            "cancelled": "failed",
        }.get(
            operation.get("model_context_execution", {}).get(
                "state", operation["state"]
            ),
            "unknown",
        )
        await self.receipt(
            value,
            phase,
            operation,
            operation.get("code") if phase != "applied" else None,
        )
        if phase == "applied" and operation.get("first_presented"):
            self.schedule_first_presented()
        if (
            phase in ("applied", "failed", "rejected")
            and self.candidate
            and request_identity(self.candidate["condition"]) == request_identity(value)
        ):
            self.candidate = None
        return True

    def current_first_presented(self):
        if self.closed:
            return None
        state = self.journal.snapshot or {}
        operation = retained_first_presented(state)
        current = applied_context_operation(state)
        if (
            operation is not None
            and current is not None
            and operation["first_presented"] == "pending"
            and operation["model_context"]["segmentId"]
            == current["model_context"]["segmentId"]
            == self.envelope["segmentId"]
            and operation["model_context_lifecycle"]
            == current["model_context_lifecycle"]
            and all(
                operation["model_context"][key] == self.envelope[key]
                for key in ENVELOPE
            )
        ):
            return operation
        return None

    async def emit_first_presented(self, request_id):
        operation = self.current_first_presented()
        if (
            self.first_presented_supported()
            and operation is not None
            and operation["request_id"] == request_id
        ):
            await self.receipt(operation["model_context"], "first-presented", operation)

    def schedule_first_presented(self):
        operation = retained_first_presented(self.journal.snapshot or {})
        if (
            self.closed
            or operation is None
            or operation.get("first_presented") != "pending"
        ):
            return
        request_id = operation["request_id"]
        if (
            self.first_presented_task is not None
            and not self.first_presented_task.done()
        ):
            if request_id != self.first_presented_request_id:
                self.first_presented_reply.set()
            return
        self.first_presented_request_id = request_id
        self.first_presented_task = asyncio.create_task(
            self.replay_first_presented(request_id)
        )

    async def replay_first_presented(self, request_id):
        confirmation_id = None
        try:
            retained = retained_first_presented(self.journal.snapshot or {})
            if retained is None or retained["request_id"] != request_id:
                return
            condition_id = retained["model_context"]["conditionId"]
            confirmation_id = retained["first_presented_confirmation_id"]
            status = "timeout"
            for delay in (1, 2, 4):
                operation = self.current_first_presented()
                if operation is None or operation["request_id"] != request_id:
                    status = "stale"
                    break
                self.first_presented_reply.clear()
                try:
                    # Reconnection temporarily clears capability negotiation,
                    # but does not invalidate the original game lifecycle.
                    if self.first_presented_supported():
                        await self.original(operation["model_context"])
                        await self.emit_first_presented(request_id)
                    await asyncio.wait_for(self.first_presented_reply.wait(), delay)
                    # An ACK settled this confirmation, or a successor replaced it.
                    return
                except (OSError, ValueError):
                    # This observation cannot hold the source or model output.
                    continue
            await self.settle_first_presented(
                request_id, condition_id, confirmation_id, status
            )
        finally:
            self.first_presented_task = None
            self.first_presented_request_id = None
            pending = retained_first_presented(self.journal.snapshot or {})
            if pending is not None and (
                pending["request_id"] != request_id
                or confirmation_id is not None
                and pending.get("first_presented_confirmation_id") != confirmation_id
            ):
                self.schedule_first_presented()

    async def settle_first_presented(
        self, request_id, condition_id, confirmation_id, status
    ):
        # Delivery and journal settlement have separate bounded work. A lost
        # CAS reply is reconciled against this browser command, never its retry.
        for delay in (0, 1, 2):
            if delay:
                await asyncio.sleep(delay)
            if self.closed:
                return
            try:
                state = await self.journal.read()
                result = state["operations"].get(confirmation_id)
                if result is None:
                    return
                if result["state"] == "accepted":
                    result = await self.journal.acknowledge_opening_presentation(
                        request_id,
                        condition_id,
                        status,
                        confirmation_id=confirmation_id,
                    )
            except (OSError, RenderError):
                continue
            if result is not None:
                self.first_presented_reply.set()
                if self.first_presented_result is not None:
                    self.first_presented_result(result)
                return

    async def acknowledge_first_presented(self, request_id, condition_id, status):
        if status in {"accepted", "duplicate"} and not self.first_presented_supported():
            return False
        operation = (
            self.current_first_presented()
            if status in {"accepted", "duplicate"}
            else retained_first_presented(self.journal.snapshot or {})
        )
        if (
            operation is None
            or operation["request_id"] != request_id
            or operation["model_context"]["conditionId"] != condition_id
        ):
            return False
        try:
            accepted = await self.journal.acknowledge_opening_presentation(
                request_id, condition_id, status
            )
        except (OSError, RenderError):
            # A receipt-store failure leaves the original result unknown and
            # must not suspend otherwise healthy model delivery.
            return False
        if accepted:
            self.first_presented_reply.set()
            if self.first_presented_result is not None:
                self.first_presented_result(accepted)
        return accepted is not None

    async def wait_for_startup_release(self, candidate, state):
        c = candidate["condition"]
        ids = state.get("model_context_segment_ids") or {}
        if self.candidate is not candidate or self.closed:
            return
        self.check()
        if ids.get("segment_id") != c["segmentId"]:
            raise RenderError("SEGMENT_EPOCH_MISMATCH")
        # The startup owner retains the window and its cleanup. This candidate
        # only waits on the shared release decision and its absolute deadline.
        gate = self.startup_release(c)
        if inspect.isawaitable(gate):
            gate = await gate
            self.check()
            latest = await self.journal.read()
            if latest.get("model_context_segment_ids") != ids:
                raise RenderError("SEGMENT_EPOCH_MISMATCH")
            if not latest.get("inference", {}).get("enabled", True) or latest.get(
                "inference", {}
            ).get("revision") != state.get("inference", {}).get("revision"):
                raise RenderError("MODEL_UNAVAILABLE")
        if gate is None:
            return
        deadline, released = gate
        if not math.isfinite(deadline) or deadline <= asyncio.get_running_loop().time():
            raise RenderError("RELEASE_UNKNOWN")
        try:
            async with asyncio.timeout_at(deadline):
                await asyncio.shield(released)
        except TimeoutError as error:
            raise RenderError("RELEASE_UNKNOWN") from error
        except asyncio.CancelledError as error:
            if released.cancelled():
                raise RenderError("RELEASE_UNKNOWN") from error
            raise
        latest = await self.journal.read()
        if latest.get("model_context_segment_ids") != ids:
            raise RenderError("SEGMENT_EPOCH_MISMATCH")
        inference = latest.get("inference", {})
        if not inference.get("enabled", True) or inference.get("revision") != state.get(
            "inference", {}
        ).get("revision"):
            raise RenderError("MODEL_UNAVAILABLE")

    async def prepare_references(self, condition, limits):
        prompt_text(condition["prompt"])
        if len(condition["references"]) > min(
            MAX_REFERENCE_IMAGES, limits["max_references"]
        ):
            raise RenderError("LIMIT_EXCEEDED")
        refs, total = [], 0
        for ref in condition["references"]:
            data = await self.read_resource(ref["resource"])
            self.check()
            if (
                not isinstance(data, bytes)
                or hashlib.sha256(data).hexdigest() != ref["sha256"]
            ):
                raise invalid("MODEL_CONTEXT_RESOURCE_HASH")
            total += len(data)
            if total > min(MAX_REFERENCE_BYTES, limits["max_reference_bytes"]):
                raise RenderError("LIMIT_EXCEEDED")
            asset = await self.assets.upload(ref["format"], data)
            self.validate_reference(asset)
            if asset["format"] != ref["format"]:
                raise invalid("MODEL_CONTEXT_RESOURCE_HASH")
            refs.append(
                dict(
                    id="<Picture %s>" % ref["picture"],
                    asset_id=asset["asset_id"],
                    role="scene",
                )
            )
        return refs

    async def prepare(self, candidate):
        try:
            c = candidate["condition"]
            limits = self.capabilities()
            if len(c["onsetOccurrenceIds"]) > 32:
                raise RenderError("LIMIT_EXCEEDED")
            refs = await self.prepare_references(c, limits)
            # Existing text-only models may retain exactly the bound ordered set.
            state = await self.journal.read()
            current = state.get("segment")
            if current and current.get("context_segment_id") == c["segmentId"]:
                last = current.get("context_references", current["references"])
                if refs != last and not limits["reference_updates"]:
                    raise invalid("REFERENCE_UPDATE_UNSUPPORTED")
            elif self.startup_release is not None:
                await self.wait_for_startup_release(candidate, state)
            async with self.lock:
                if self.candidate is not candidate or self.closed:
                    return
                self.check()
                if self.startup_effective is not None:
                    candidate["startup_effective"] = self.startup_effective(c)
                candidate.update(references=refs, state="prepared")
                await self.receipt(c, "prepared")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            async with self.lock:
                if self.candidate is candidate and not self.closed:
                    code = (
                        error.code
                        if isinstance(error, RenderError)
                        else (
                            str(error)
                            if str(error)
                            in {
                                "MODEL_CONTEXT_RESOURCE_HASH",
                                "REFERENCE_UPDATE_UNSUPPORTED",
                                "MODEL_CONTEXT_STALE_SOURCE",
                            }
                            else "MODEL_CONTEXT_PREPARE_FAILED"
                        )
                    )
                    candidate.update(state="rejected", reason=code)
                    await self.receipt(candidate["condition"], "rejected", reason=code)

    def preparation_done(self, task):
        self.preparations.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        # Native retirement intentionally fences a late preparation or receipt.
        if isinstance(error, ValueError) and str(error) == "MODEL_CONTEXT_STALE_SOURCE":
            return
        if error is not None:
            logging.getLogger(__name__).error(
                "Model condition preparation did not complete", exc_info=error
            )

    async def commit(self, current, committed):
        try:
            operation = await self.submit(committed, current["references"])
        except Exception as error:
            async with self.lock:
                if self.closed:
                    return
                self.check()
                # Preserve an original CAS whose external receipt was lost.
                if not await self.original(committed):
                    if isinstance(error, RenderError) and not error.retryable:
                        current.update(state="rejected", reason=error.code)
                    await self.candidate_receipt(committed, current)
            return
        async with self.lock:
            if not self.closed:
                self.check()
                await self.receipt(committed, "accepted", operation)

    async def handle(self, value):
        kind = value.get("type")
        if kind == "model.condition":
            value = parse_condition(value)
        else:
            extra = (
                ("effective",)
                if kind == "model.condition.commit"
                else ("reason",)
                if kind == "model.condition.withdraw"
                else ()
            )
            if not extra:
                raise invalid()
            keys(
                value,
                (
                    *ENVELOPE,
                    "type",
                    "controlSequence",
                    "conditionId",
                    "payloadHash",
                    *extra,
                ),
            )
            integer(value["controlSequence"], 1)
            if kind == "model.condition.commit":
                effective(value["effective"])
            else:
                identity(value["reason"])
        self.validate(value)
        async with self.lock:
            if self.closed:
                raise invalid("MODEL_CONTEXT_STALE_SOURCE")
            if await self.original(value):
                return
            current = self.candidate
            if current and current["condition"]["conditionId"] == value["conditionId"]:
                if current["condition"]["payloadHash"] != value["payloadHash"]:
                    raise invalid("MODEL_CONTEXT_PAYLOAD_HASH")
                if (
                    kind == "model.condition.commit"
                    and "committed_effective" in current
                    and current["committed_effective"] != value["effective"]
                ):
                    raise invalid("MODEL_CONTEXT_COMMIT_CONFLICT")
            elif current and current["state"] in ("preparing", "prepared", "committed"):
                raise invalid("MODEL_CONTEXT_BACKPRESSURE")
            elif kind != "model.condition":
                await self.receipt(value, "unknown")
                return
            else:
                current = None
            if kind == "model.condition":
                if current:
                    if canonical(current["condition"]) != canonical(value):
                        raise invalid("MODEL_CONTEXT_CONDITION_CONFLICT")
                    await self.candidate_receipt(value, current)
                    return
                current = dict(condition=value, state="preparing")
                self.candidate = current
                await self.receipt(value, "queued")
                if self.closed:
                    return
                self.preparation = asyncio.create_task(self.prepare(current))
                # A withdrawn upload may still own an atomic asset commit.
                # Keep it owned even when a successor becomes the candidate.
                self.preparations.add(self.preparation)
                self.preparation.add_done_callback(self.preparation_done)
            elif kind == "model.condition.withdraw":
                if current["state"] == "committed":
                    await self.receipt(value, "unknown")
                    return
                current["state"] = "withdrawn"
                if self.preparation:
                    self.preparation.cancel()
                self.candidate = None
            elif current["state"] in ("committed", "rejected"):
                await self.candidate_receipt(value, current)
            elif current["state"] == "prepared":
                # Host's commit boundary is fixed; model indices are assigned later.
                committed = dict(
                    current["condition"], effective=copy.deepcopy(value["effective"])
                )
                current["state"] = "committed"
                current["committed_effective"] = copy.deepcopy(value["effective"])
                # Submission may need a future source frame whose sample is
                # delivered by this same control reader. Keep the operation owned
                # through close, while allowing that reader to advance.
                submission = asyncio.create_task(self.commit(current, committed))
                self.preparations.add(submission)
                submission.add_done_callback(self.preparation_done)
                return submission
            else:
                raise invalid("MODEL_CONTEXT_NOT_PREPARED")

    async def reconcile(self, queries):
        if not isinstance(queries, list) or len(queries) > 2048:
            raise invalid()
        for query in queries:
            self.validate(query)
            extras = set(query) - set((*ENVELOPE, "conditionId", "payloadHash"))
            if extras not in (set(), {"requestId", "promptVersion"}):
                raise invalid()
            if extras:
                identity(query["requestId"])
                integer(query["promptVersion"])
            if not await self.original(query, query=query):
                c = self.candidate
                if c and request_identity(c["condition"]) == request_identity(query):
                    await self.candidate_receipt(query, c)
                else:
                    await self.receipt(query, "unknown")

    async def close(self):
        self.closed = True
        tasks = tuple(self.preparations)
        if self.first_presented_task is not None:
            tasks += (self.first_presented_task,)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def validate_capability(value):
    append_field = isinstance(value, dict) and "prewarmAppendProtocol" in value
    plan_field = isinstance(value, dict) and "prewarmPlanProtocol" in value
    opening_field = isinstance(value, dict) and "firstPresentedProtocol" in value
    keys(
        value,
        (
            "protocol",
            "required",
            "configHash",
            "gameBuildHash",
            "compilerVersion",
            "visible",
            *(("firstPresentedProtocol",) if opening_field else ()),
            *(("prewarmPlanProtocol",) if plan_field else ()),
            *(("prewarmAppendProtocol",) if append_field else ()),
        ),
    )
    if (
        value["protocol"] != PROTOCOL
        or value["required"] is not True
        or value["visible"] is not False
        or type(value["compilerVersion"]) is not int
        or value["compilerVersion"] != 1
        or opening_field
        and value["firstPresentedProtocol"] != FIRST_PRESENTED_PROTOCOL
        or append_field
        and (
            not plan_field
            or value["prewarmAppendProtocol"]
            != "worldkit-model-context-prewarm-append-v1"
        )
        or plan_field
        and value["prewarmPlanProtocol"] != "worldkit-model-context-prewarm-plan-v1"
    ):
        raise invalid("MODEL_CONTEXT_CAPABILITY_UNSUPPORTED")
    digest(value["configHash"])
    digest(value["gameBuildHash"])
