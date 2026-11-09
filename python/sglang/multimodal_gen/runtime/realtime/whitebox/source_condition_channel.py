# SPDX-License-Identifier: Apache-2.0
"""Authenticated CLI transport and bounded context/frame recovery on one socket."""

import asyncio
import copy
import json
import logging
import re

from websockets.exceptions import ConnectionClosed

from .opening_presentation import PROTOCOL as FIRST_PRESENTED_PROTOCOL
from .render_contract import RenderError, canonical, identity, prompt_text
from .source_conditions import (
    AUDIENCE,
    ENVELOPE,
    PROTOCOL,
    digest,
    envelope,
    integer,
    invalid,
    keys,
    parse_condition,
    payload,
    request_identity,
    validate_capability,
)

SAMPLE_WAIT_SECONDS = 30
INTERVENTION_WAIT_SECONDS = 10


class CLIConditionChannel:
    """Bounded control worker on the existing authenticated CLI connection.

    Downloads are adapter tasks, never control-socket reads. Exact native sample
    evidence gates pixels independently of UI; no simulation step is performed.
    """

    def __init__(self, view):
        validate_capability(view.descriptor["modelContext"])
        self.view = view
        self.adapter = None
        self.consumer = None
        self.segment = None
        self.requested_segment = None
        self.sequence = 0
        self.samples = {}
        self.retired_samples = set()
        self.snapshots = []
        self.changed = asyncio.Condition()
        self.bound = asyncio.Event()
        self.queue = asyncio.Queue(maxsize=128)
        self.queue_bytes = 0
        self.task = None
        self.closed = False
        self.error = None
        self.transport_generation = 0
        self.suspended = False
        self.prewarm_supported = False
        self.prewarm_plan_supported = False
        self.prewarm_append_supported = False
        self.first_presented_supported = False
        self.require_subscription_pause = False
        self.subscription_probe = False
        self.subscription_probe_ready = asyncio.Event()
        self.subscription_probe_error = None
        self.on_recovery_failed = None
        self.source_generation = 0
        self.reset_task = None
        self.reset_transport_generation = None
        self.interventions = {}

    async def send(self, value):
        generation = getattr(self.view, "transport_generation", 0)
        try:
            await self.view.control.send(json.dumps(value))
        except (OSError, ConnectionClosed):
            # The receipt may already have reached the Host. Recovery queries
            # the same original operation; it never resends a model commit.
            await self.view.recover_model_context_transport(generation)

    def fail_interventions(self):
        # A lost reply is uncertain. The browser recovers with a read, never an
        # automatic write replay, including when the logical consumer survives.
        for _, future in self.interventions.values():
            if not future.done():
                future.set_exception(RenderError("CONTROL_UNAVAILABLE", retryable=True))

    async def intervention(self, command):
        request_id = identity(command["request_id"])
        future = None
        try:
            async with asyncio.timeout(INTERVENTION_WAIT_SECONDS):
                # session.ready may precede the initial subscription reply.
                # Waiting here does not block the shared CLI control reader.
                await self.bound.wait()
                if self.closed or self.view.closed or self.view.stopped or self.error:
                    raise RenderError("CONTROL_UNAVAILABLE", retryable=True)
                if (
                    self.suspended
                    or self.consumer is None
                    or self.requested_segment is not None
                ):
                    raise RenderError("CONTROL_UNAVAILABLE", retryable=True)
                if request_id in self.interventions:
                    raise RenderError("REQUEST_ID_CONFLICT")
                if len(self.interventions) >= 16:
                    raise RenderError("LIMIT_EXCEEDED")
                request = dict(
                    type=command["op"],
                    epoch=self.view.descriptor["epoch"],
                    consumerEpoch=self.consumer["consumerEpoch"],
                    segmentId=self.segment,
                    requestId=request_id,
                )
                if request["type"] == "model.context.intervention":
                    request.update(
                        expectedRevision=command["expectedRevision"],
                        prompt=command["prompt"],
                    )
                future = asyncio.get_running_loop().create_future()
                self.interventions[request_id] = (request, future)
                # Use the current authenticated CLI socket. Its client ID owns
                # the consumer; browser fields can never choose another member.
                await self.view.control.send(json.dumps(request))
                return await future
        except (OSError, ConnectionClosed, TimeoutError) as error:
            raise RenderError("CONTROL_UNAVAILABLE", retryable=True) from error
        finally:
            if future is not None:
                self.interventions.pop(request_id, None)
                if not future.done():
                    future.cancel()

    def intervention_result(self, value):
        request_id = value.get("requestId")
        if not isinstance(request_id, str):
            return
        pending = self.interventions.get(request_id)
        if pending is None:
            return  # Late replies cannot mutate a later read or another request.
        request, future = pending
        if future.done():
            return
        try:
            keys(
                value,
                (
                    "type",
                    "epoch",
                    "consumerEpoch",
                    "segmentId",
                    "requestId",
                    "status",
                    "revision",
                    "prompt",
                    *(("code",) if "code" in value else ()),
                ),
            )
            if (
                any(
                    value[k] != request[k]
                    for k in ("epoch", "consumerEpoch", "segmentId", "requestId")
                )
                or value["segmentId"] != self.segment
                or self.consumer is None
                or value["consumerEpoch"] != self.consumer["consumerEpoch"]
                or self.requested_segment is not None
                or self.suspended
                or self.closed
                or self.view.closed
                or self.view.stopped
            ):
                raise RenderError("CONTROL_UNAVAILABLE", retryable=True)
            if value["status"] not in {
                "accepted",
                "duplicate",
                "conflict",
                "rejected",
                "current",
            }:
                raise ValueError("Invalid intervention status")
            integer(value["epoch"])
            integer(value["revision"])
            if value["prompt"] is not None:
                prompt_text(value["prompt"])
            if "code" in value and (
                not isinstance(value["code"], str)
                or not re.fullmatch(r"[A-Z][A-Z0-9_]{1,95}", value["code"])
            ):
                raise ValueError("Invalid intervention code")
        except (ValueError, TypeError):
            future.set_exception(RenderError("CONTROL_UNAVAILABLE", retryable=True))
        else:
            future.set_result(copy.deepcopy(value))

    async def bind(self, *, segment_id, create_adapter, start_suspended=False):
        if self.task is not None:
            raise invalid("MODEL_CONTEXT_WRITER_CONFLICT")
        self.segment, self.create_adapter = identity(segment_id), create_adapter
        self.task = asyncio.create_task(self.run())
        if start_suspended:
            await self.pause()
        else:
            await self.subscribe()

    async def pause(self):
        """Intentional preview mode preserves unresolved condition history."""
        self.suspended = True
        self.prewarm_supported = False
        self.prewarm_plan_supported = False
        self.prewarm_append_supported = False
        self.first_presented_supported = False
        self.fail_interventions()
        self.discard_controls()
        self.bound.set()
        async with self.changed:
            self.changed.notify_all()
        await self.send(
            dict(type="model.context.preview", epoch=self.view.descriptor["epoch"])
        )

    def prewarm_window(self):
        if self.suspended or not (
            getattr(self, "prewarm_supported", False)
            or getattr(self, "prewarm_plan_supported", False)
        ):
            return None
        get_window = getattr(self.adapter, "startup_window", None)
        window = get_window() if get_window is not None else None
        if window is not None:
            from .startup_prewarm_plan import StartupPrewarmPlan

            if (
                isinstance(window, StartupPrewarmPlan)
                and not self.prewarm_plan_supported
            ):
                return None
        return window

    def subscription_request(self, *, paused=False):
        request = dict(
            type="model.context.subscribe",
            protocol=PROTOCOL,
            epoch=self.view.descriptor["epoch"],
            segmentId=self.requested_segment or self.segment,
        )
        if self.require_subscription_pause:
            request["requireSubscriptionPause"] = True
        if paused:
            request["paused"] = True
        return request

    async def subscribe(self):
        await self.send(self.subscription_request())

    def retire_source(self, reset):
        """Fence an authenticated SDK reset without blocking its shared reader."""
        # Retire preparation before its source checks and feedback writer become
        # stale. Its original task still owns bounded child cleanup and release.
        window = self.prewarm_window()
        if window is not None:
            window.invalidate()
        self.suspended = True
        self.source_generation += 1
        self.reset_transport_generation = self.view.transport_generation
        self.fail_interventions()
        self.discard_controls()
        self.subscription_probe_error = "SOURCE_EPOCH_MISMATCH"
        self.subscription_probe_ready.set()
        previous, adapter = self.reset_task, self.adapter
        self.adapter = self.consumer = None
        self.requested_segment = None
        self.sequence = 0
        self.samples.clear()
        self.retired_samples.clear()
        self.snapshots.clear()
        self.prewarm_supported = self.prewarm_plan_supported = False
        self.prewarm_append_supported = self.first_presented_supported = False

        async def retire():
            # The SDK pixel broker also owns the model/preview delivery gate.
            # Preview transport and original cleanup must progress independently.
            try:
                if previous is not None:
                    await asyncio.shield(previous)
                preview = asyncio.create_task(self.pause())
                try:
                    # Attempt preview delivery, but never wait for transport
                    # after the original resource cleanup has completed.
                    await asyncio.sleep(0)
                    # Preview delivery and original cleanup are independent. A
                    # blocked send must not keep the retired opening alive.
                    if self.on_recovery_failed is not None:
                        await self.on_recovery_failed(
                            "MODEL_CONTEXT_STALE_SOURCE", source_reset=reset
                        )
                finally:
                    if not preview.done():
                        preview.cancel()
                    await asyncio.gather(preview, return_exceptions=True)
            finally:
                if adapter is not None:
                    await adapter.close()
                async with self.changed:
                    self.changed.notify_all()

        self.reset_task = asyncio.create_task(retire())
        self.reset_task.add_done_callback(
            lambda task: task.exception() if not task.cancelled() else None
        )

    async def probe_subscription_pause(self, *, timeout):
        """Prove the actual World binding can pause while preview stays active."""
        while self.reset_task is not None:
            retiring = self.reset_task
            await asyncio.shield(retiring)
            # Explicit retry replaces the old socket before accepting a fresh
            # consumer. Buffered reset errors cannot become its probe response.
            await self.view.recover_model_context_transport(
                self.reset_transport_generation
            )
            if self.reset_task is retiring:
                self.reset_task = None
        self.check()
        if not self.suspended:
            raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
        self.require_subscription_pause = True
        self.subscription_probe = True
        self.subscription_probe_error = None
        self.subscription_probe_ready.clear()
        try:
            async with asyncio.timeout(timeout):
                await self.send(self.subscription_request(paused=True))
                await self.subscription_probe_ready.wait()
            if self.subscription_probe_error:
                raise RenderError(self.subscription_probe_error)
            self.check()
        finally:
            self.subscription_probe = False

    def probe_reply(self, value):
        return self.subscription_probe and value.get("type") in {
            "model.context.subscribed",
            "model.context.error",
        }

    async def switch_segment(self, segment_id):
        self.check()
        was_suspended = self.suspended
        source_generation = self.source_generation
        segment_id = identity(segment_id)
        if self.requested_segment and self.requested_segment != segment_id:
            raise invalid("MODEL_CONTEXT_SEGMENT_SWITCH_PENDING")
        if segment_id != self.segment:
            self.requested_segment = segment_id
        self.suspended = False
        if was_suspended:
            # Drop preview-only packets before opening the new model gate.
            reset_generation = self.reset_transport_generation
            await self.view.recover_model_context_transport(
                reset_generation
                if reset_generation is not None
                else self.view.transport_generation
            )
            if source_generation != self.source_generation:
                raise invalid("MODEL_CONTEXT_STALE_SOURCE")
            self.reset_transport_generation = None
            if reset_generation is not None:
                # The paused probe already replaced this transport. Its native
                # generation guard avoids a second reconnect; open delivery now.
                await self.subscribe()
        else:
            await self.subscribe()

    async def suspend(self, reason):
        """Fail only model delivery; preview may continue without forged mapping."""
        if self.suspended:
            return
        self.suspended = True
        self.first_presented_supported = False
        self.fail_interventions()
        self.discard_controls()
        if self.on_recovery_failed is not None:
            await self.on_recovery_failed(reason)
        self.bound.set()
        async with self.changed:
            self.changed.notify_all()
        await self.send(
            dict(type="model.context.preview", epoch=self.view.descriptor["epoch"])
        )

    async def reconnect(self):
        self.fail_interventions()
        self.check()
        self.prewarm_supported = False
        self.prewarm_plan_supported = False
        self.prewarm_append_supported = False
        self.first_presented_supported = False
        self.bound.clear()
        # Both sockets were replaced, so buffered packets from the old socket
        # cannot be consumed. Keep control history and the original candidate.
        async with self.changed:
            self.transport_generation += 1
            self.samples.clear()
            self.retired_samples.clear()
            self.changed.notify_all()
        await self.view.control.send(
            json.dumps(
                dict(type="model.context.preview", epoch=self.view.descriptor["epoch"])
                if self.suspended
                else self.subscription_request()
            )
        )

    def discard_controls(self):
        while not self.queue.empty():
            self.queue.get_nowait()
        self.queue_bytes = 0

    def offer(self, value):
        if value.get("type") == "model.context.intervention.result":
            self.intervention_result(value)
            return
        if self.suspended and not self.probe_reply(value):
            return
        size = len(canonical(value))
        if self.queue.full() or self.queue_bytes + size > 2 << 20:
            # Stop model admission through its worker, without terminating the
            # shared control reader. The consumed cursor remains authoritative.
            self.discard_controls()
            value = dict(
                type="model.context.error",
                epoch=self.view.descriptor["epoch"],
                code="MODEL_CONTEXT_BACKPRESSURE",
            )
            size = len(canonical(value))
        self.queue_bytes += size
        self.queue.put_nowait((value, size))

    def check(self):
        if self.closed or self.view.closed or self.view.stopped:
            raise invalid("MODEL_CONTEXT_STALE_SOURCE")
        if self.error:
            raise self.error

    def check_generation(self, generation):
        self.check()
        if generation != self.source_generation:
            raise invalid("MODEL_CONTEXT_STALE_SOURCE")

    async def emit(self, receipt):
        self.check()
        if self.suspended:
            # Preview mode has no model-control admission. The journal retains
            # late results for the next explicit subscription to reconcile.
            return
        await self.send(
            dict(
                type="model.context.control",
                epoch=self.view.descriptor["epoch"],
                message=receipt,
            )
        )

    async def run(self):
        try:
            while not self.closed:
                value, size = await self.queue.get()
                self.queue_bytes -= size
                generation = self.source_generation
                try:
                    await self.consume(value)
                except Exception as error:
                    if self.closed:
                        return
                    if generation != self.source_generation:
                        continue  # A retired item cannot fail a fresh probe or source.
                    logging.getLogger(__name__).exception(
                        "Model context control failed"
                    )
                    code = getattr(error, "code", None)
                    if code is None and isinstance(error, ValueError):
                        code = str(error)
                    if not isinstance(code, str) or not re.fullmatch(
                        r"[A-Z][A-Z0-9_]{1,95}", code
                    ):
                        code = "MODEL_CONTEXT_PROTOCOL_INVALID"
                    if self.subscription_probe:
                        self.subscription_probe_error = code
                        self.subscription_probe_ready.set()
                    await self.suspend(code)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.error = error
            self.bound.set()
            async with self.changed:
                self.changed.notify_all()

    def snapshot_metadata(self, value):
        keys(
            value,
            (
                *ENVELOPE,
                "type",
                "controlSequence",
                "baseControlSequence",
                "occurrenceCursor",
                "maxEvictedOccurrenceSequence",
                "contextStateId",
                "contextRevision",
                "viewRevision",
                "compiledConditionId",
                "desired",
                "executions",
                "candidate",
            ),
        )
        if value["type"] != "model.context.snapshot":
            raise invalid()
        for field in (
            "controlSequence",
            "baseControlSequence",
            "occurrenceCursor",
            "maxEvictedOccurrenceSequence",
            "contextRevision",
            "viewRevision",
        ):
            integer(value[field])
        identity(value["contextStateId"])
        if value["compiledConditionId"] is not None:
            identity(value["compiledConditionId"])
        if value["maxEvictedOccurrenceSequence"] > value["occurrenceCursor"]:
            raise invalid()
        candidate = value["candidate"]
        if candidate is not None:
            keys(candidate, ("conditionId", "payloadHash", "status"))
            identity(candidate["conditionId"])
            digest(candidate["payloadHash"])
            if candidate["status"] not in ("preparing", "committed", "reconciling"):
                raise invalid()
        if not isinstance(value["executions"], list) or len(value["executions"]) > 2048:
            raise invalid()
        if value["desired"] is None:
            if value["compiledConditionId"] is not None:
                raise invalid("MODEL_CONTEXT_SNAPSHOT_IDENTITY")
            self.snapshots.clear()
            return
        if envelope(value["desired"]) != envelope(value):
            raise invalid("MODEL_CONTEXT_SNAPSHOT_IDENTITY")
        payload(value["desired"])
        snapshots = self.snapshots + [
            (
                value["controlSequence"],
                {
                    k: value[k]
                    for k in (
                        "contextStateId",
                        "contextRevision",
                        "viewRevision",
                        "compiledConditionId",
                    )
                },
            )
        ]
        # Pending pixels pin the exact prefix mapping they were admitted under.
        # Keep the latest mapping for new samples; only unused history may retire.
        pinned = {len(snapshots) - 1}
        for metadata in self.samples.values():
            match = next(
                (
                    i
                    for i in range(len(snapshots) - 1, -1, -1)
                    if snapshots[i][0] <= metadata["throughControlSequence"]
                ),
                None,
            )
            if match is not None:
                pinned.add(match)
        if len(snapshots) > 128:
            unused = next((i for i in range(len(snapshots)) if i not in pinned), None)
            if unused is None:
                raise invalid("MODEL_CONTEXT_BACKPRESSURE")
            del snapshots[unused]
        self.snapshots = snapshots

    def verify_sample(self, metadata):
        snapshot = next(
            (
                state
                for sequence, state in reversed(self.snapshots)
                if sequence <= metadata["throughControlSequence"]
            ),
            None,
        )
        if snapshot is None or any(metadata[k] != v for k, v in snapshot.items()):
            raise invalid("MODEL_CONTEXT_SAMPLE_IDENTITY")

    def remember_retired(self, keys):
        # Exact old sample evidence, bounded like the live sample queue. A
        # barrier alone cannot identify the generation of an unmatched pixel.
        self.retired_samples.update(keys)
        if len(self.retired_samples) > 128:
            raise invalid("MODEL_CONTEXT_BACKPRESSURE")

    async def retire_samples(self, floor):
        generation = self.source_generation
        async with self.changed:
            self.check_generation(generation)
            self.sample_floor = floor
            retired = [
                key
                for key, metadata in self.samples.items()
                if metadata["throughControlSequence"] < floor
            ]
            self.remember_retired(retired)
            for key in retired:
                del self.samples[key]
            self.changed.notify_all()

    async def consume(self, value):
        self.check()
        generation = self.source_generation
        if self.suspended and not self.probe_reply(value):
            return
        kind = value.get("type")
        if kind == "model.context.intervention.result":
            self.intervention_result(value)
            return
        descriptor = self.view.descriptor
        if kind == "model.context.subscribed":
            if (
                value.get("epoch") != descriptor["epoch"]
                or value.get("protocol") != PROTOCOL
                or value.get("segmentId") not in {self.segment, self.requested_segment}
                or value.get("pendingSegmentId") not in (None, self.requested_segment)
                or value.get("pendingSegmentId") is not None
                and value.get("segmentId") != self.segment
            ):
                raise invalid("MODEL_CONTEXT_STALE_SOURCE")
            if self.require_subscription_pause and (
                value.get("subscriptionPause") is not True
                or value.get("paused") is not self.subscription_probe
            ):
                code = "MODEL_CONTEXT_SUBSCRIPTION_PAUSE_UNSUPPORTED"
                self.subscription_probe_error = code
                self.subscription_probe_ready.set()
                raise RenderError(code)
            consumer = value.get("consumer")
            keys(consumer, AUDIENCE)
            for item in consumer.values():
                identity(item)
            if consumer["consumerId"] != self.view.client_id:
                raise invalid("MODEL_CONTEXT_STALE_SOURCE")
            if self.consumer and consumer != self.consumer:
                raise invalid("MODEL_CONTEXT_STALE_SOURCE")
            integer(value.get("sampleControlFloor"), 1)
            self.sample_floor = value["sampleControlFloor"]
            self.consumer = copy.deepcopy(consumer)
            if self.subscription_probe:
                # An inactive capability proof never binds conditions, resumes
                # model delivery or changes the preview frame admission gate.
                self.subscription_probe_ready.set()
                return
            from .startup_prewarm import PROTOCOL as PREWARM_PROTOCOL

            self.prewarm_supported = (
                value.get("paused") is not True
                and value.get("prewarmProtocol") == PREWARM_PROTOCOL
            )
            from .startup_prewarm_plan import PROTOCOL as PLAN_PROTOCOL

            self.prewarm_plan_supported = (
                value.get("paused") is not True
                and value.get("prewarmPlanProtocol") == PLAN_PROTOCOL
                and descriptor["modelContext"].get("prewarmPlanProtocol")
                == PLAN_PROTOCOL
            )
            from .startup_prewarm_plan import APPEND_PROTOCOL

            self.prewarm_append_supported = (
                self.prewarm_plan_supported
                and value.get("prewarmAppendProtocol") == APPEND_PROTOCOL
                and descriptor["modelContext"].get("prewarmAppendProtocol")
                == APPEND_PROTOCOL
            )
            self.first_presented_supported = (
                value.get("paused") is not True
                and value.get("firstPresentedProtocol") == FIRST_PRESENTED_PROTOCOL
            )
            if self.adapter is not None:
                candidate = self.adapter.candidate
                if candidate is not None:
                    await self.adapter.reconcile(
                        [request_identity(candidate["condition"])]
                    )
                await self.send(
                    dict(
                        type="model.context.control",
                        epoch=descriptor["epoch"],
                        message={
                            **self.adapter.envelope,
                            # The Host may have finished the handoff while the
                            # previous control socket was disconnected.
                            "segmentId": value["segmentId"],
                            "type": "model.context.resume",
                            "afterControlSequence": self.sequence,
                        },
                    )
                )
                window = self.prewarm_window()
                if window is not None:
                    window.append_supported = self.prewarm_append_supported
                    window.announce()
                if self.first_presented_supported:
                    self.adapter.schedule_first_presented()
            self.bound.set()
            return
        if kind == "model.context.error":
            keys(value, ("type", "epoch", "code"))
            if value["epoch"] != descriptor["epoch"]:
                raise invalid("MODEL_CONTEXT_STALE_SOURCE")
            code = value["code"]
            if not isinstance(code, str) or not re.fullmatch(
                r"[A-Z][A-Z0-9_]{1,95}", code
            ):
                raise invalid()
            if self.subscription_probe:
                self.subscription_probe_error = code
                self.subscription_probe_ready.set()
            await self.suspend(code)
            return
        if not self.consumer:
            raise invalid("MODEL_CONTEXT_NOT_SUBSCRIBED")
        if kind in {
            "model.context.prewarm",
            "model.context.prewarm.cancel",
            "model.context.prewarm.plan",
            "model.context.prewarm.plan.append",
            "model.context.prewarm.plan.cancel",
            "model.context.prewarm.plan.finish",
        }:
            window = self.prewarm_window()
            is_plan = kind.startswith("model.context.prewarm.plan")
            supported = (
                self.prewarm_plan_supported if is_plan else self.prewarm_supported
            )
            if window is not None and supported:
                try:
                    window.append_supported = self.prewarm_append_supported
                    window.receive(value)
                except (KeyError, TypeError, ValueError):
                    logging.getLogger(__name__).warning(
                        "Ignored invalid model-context prewarm decision"
                    )
            return
        if kind == "model.context.sample":
            from .native_pixels import read_frame_identity

            if (
                value.get("epoch") != descriptor["epoch"]
                or value.get("consumerId") != self.consumer["consumerId"]
                or value.get("consumerEpoch") != self.consumer["consumerEpoch"]
                or value.get("segmentId") != self.segment
            ):
                raise invalid("MODEL_CONTEXT_STALE_SOURCE")
            header = read_frame_identity(value["header"])
            if (
                header["sessionId"] != descriptor["sessionId"]
                or header["epoch"] != descriptor["epoch"]
            ):
                raise invalid("MODEL_CONTEXT_STALE_SOURCE")
            metadata = value["metadata"]
            keys(
                metadata,
                (
                    "contextStateId",
                    "contextRevision",
                    "viewRevision",
                    "throughControlSequence",
                    "compiledConditionId",
                ),
            )
            identity(metadata["contextStateId"])
            for key in ("contextRevision", "viewRevision", "throughControlSequence"):
                integer(metadata[key])
            if metadata["compiledConditionId"] is not None:
                identity(metadata["compiledConditionId"])
            if metadata["throughControlSequence"] < self.sample_floor:
                async with self.changed:
                    self.remember_retired([canonical(header)])
                    self.changed.notify_all()
                return
            if len(self.samples) >= 128:
                raise invalid("MODEL_CONTEXT_BACKPRESSURE")
            async with self.changed:
                if metadata["throughControlSequence"] <= self.sequence:
                    self.verify_sample(metadata)
                self.samples[canonical(header)] = copy.deepcopy(metadata)
                self.changed.notify_all()
            return
        if kind == "model.context.receipt-result":
            if (
                self.first_presented_supported
                and self.adapter is not None
                and set(value)
                == {"type", "epoch", "phase", "conditionId", "requestId", "status"}
                and value["epoch"] == descriptor["epoch"]
                and value["phase"] == "first-presented"
                and value["status"]
                in ("accepted", "duplicate", "stale", "history-unavailable")
                and isinstance(value["conditionId"], str)
                and isinstance(value["requestId"], str)
            ):
                await self.adapter.acknowledge_first_presented(
                    value["requestId"], value["conditionId"], value["status"]
                )
            return
        if kind == "model.context.reconcile":
            if value.get("epoch") != descriptor["epoch"] or self.adapter is None:
                raise invalid("MODEL_CONTEXT_STALE_SOURCE")
            await self.adapter.reconcile(value["queries"])
            return
        if kind == "model.context.checkpoint":
            keys(
                value,
                ("type", "afterControlSequence", "snapshot", "condition", "queries"),
            )
            integer(value["afterControlSequence"])
            snapshot = value["snapshot"]
            if (
                value["afterControlSequence"] != self.sequence
                or snapshot.get("controlSequence", 0) <= self.sequence
            ):
                raise invalid("MODEL_CONTEXT_CHECKPOINT_CURSOR")
            # Recovery supplies the original prepared candidate, never recompiles it.
            await self.admit_envelope(snapshot)
            self.check_generation(generation)
            self.snapshot_metadata(snapshot)
            candidate = snapshot.get("candidate")
            condition = value["condition"]
            if bool(candidate) != bool(condition):
                raise invalid("MODEL_CONTEXT_CHECKPOINT_CANDIDATE")
            if condition:
                parse_condition(condition)
                if request_identity(condition) != {
                    **envelope(snapshot),
                    "conditionId": candidate["conditionId"],
                    "payloadHash": candidate["payloadHash"],
                }:
                    raise invalid("MODEL_CONTEXT_CHECKPOINT_CANDIDATE")
                if candidate["status"] == "preparing":
                    await self.adapter.handle(condition)
                    self.check_generation(generation)
            await self.adapter.reconcile(value["queries"])
            self.check_generation(generation)
            self.sequence = snapshot["controlSequence"]
            await self.retire_samples(self.sequence + 1)
            self.check_generation(generation)
            await self.ack()
            return
        sequence = value.get("controlSequence")
        integer(sequence, 1)
        e = self.validate_envelope(value, consumed=sequence <= self.sequence)
        if sequence <= self.sequence:
            # The Host retains a consumed prefix across handoff. Do not send
            # old-segment queries through the replacement segment's adapter.
            if e["segmentId"] == self.segment and kind.startswith("model.condition"):
                await self.adapter.reconcile([request_identity(value)])
            return
        if sequence != self.sequence + 1:
            await self.send(
                dict(
                    type="model.context.control",
                    epoch=descriptor["epoch"],
                    message={
                        **e,
                        "type": "model.context.resume",
                        "afterControlSequence": self.sequence,
                    },
                )
            )
            return
        await self.admit_envelope(value)
        self.check_generation(generation)
        if kind == "model.context.snapshot":
            self.snapshot_metadata(value)
            if value.get("desired") is not None:
                if envelope(value["desired"]) != envelope(value):
                    raise invalid("MODEL_CONTEXT_SNAPSHOT_IDENTITY")
                payload(value["desired"])
            else:
                if value.get("compiledConditionId") is not None:
                    raise invalid("MODEL_CONTEXT_SNAPSHOT_IDENTITY")
                await self.retire_samples(sequence + 1)
        else:
            await self.adapter.handle(value)
        self.check_generation(generation)
        self.sequence = sequence
        await self.ack()
        async with self.changed:
            self.check_generation(generation)
            self.changed.notify_all()

    def validate_envelope(self, value, *, consumed=False):
        e = envelope(value)
        descriptor = self.view.descriptor
        capability = descriptor["modelContext"]
        if (
            e["source"]["sessionId"] != descriptor["sessionId"]
            or e["source"]["epoch"] != descriptor["epoch"]
            or e["source"]["gameBuildHash"] != capability["gameBuildHash"]
            or e["configHash"] != capability["configHash"]
            or e["audience"] != self.consumer
            or (
                not consumed
                and e["segmentId"] not in {self.segment, self.requested_segment}
            )
            or (
                self.adapter is not None
                and any(
                    e[key] != self.adapter.envelope[key]
                    for key in ENVELOPE
                    if key != "segmentId"
                )
            )
        ):
            raise invalid("MODEL_CONTEXT_STALE_SOURCE")
        return e

    async def admit_envelope(self, value):
        generation = self.source_generation
        e = self.validate_envelope(value)
        if e["segmentId"] != self.segment:
            # The Host settles the previous committed candidate before it
            # publishes the new envelope. Retain its adapter until that point.
            if self.adapter is not None:
                await self.adapter.close()
                self.check_generation(generation)
            self.adapter = None
            self.segment, self.requested_segment = e["segmentId"], None
            self.snapshots.clear()
            await self.retire_samples(value["controlSequence"])
            self.check_generation(generation)
        if self.adapter is None:

            def check():
                self.check_generation(generation)

            async def emit(receipt):
                check()
                await self.emit(receipt)

            self.adapter = self.create_adapter(e, emit, check)
        elif self.adapter.envelope != e:
            raise invalid("MODEL_CONTEXT_STALE_SOURCE")

    async def ack(self):
        await self.send(
            dict(
                type="model.context.ack",
                epoch=self.view.descriptor["epoch"],
                consumerEpoch=self.consumer["consumerEpoch"],
                throughControlSequence=self.sequence,
            )
        )

    async def frame(self, header):
        from .native_pixels import read_frame_identity

        key = canonical(read_frame_identity(header))
        generation = self.transport_generation
        async with asyncio.timeout(SAMPLE_WAIT_SECONDS):
            async with self.changed:
                await self.changed.wait_for(
                    lambda: (
                        self.error
                        or self.suspended
                        or self.closed
                        or generation != self.transport_generation
                        or key in self.retired_samples
                        or key in self.samples
                        and self.samples[key]["throughControlSequence"] <= self.sequence
                    )
                )
                self.check()
                if self.suspended or generation != self.transport_generation:
                    return None
                if key in self.retired_samples:
                    self.retired_samples.remove(key)
                    return None
                metadata = self.samples.pop(key)
                self.verify_sample(metadata)
                return metadata

    async def close(self):
        if self.closed:
            return
        self.fail_interventions()
        if self.consumer and self.view.control:
            try:
                await self.send(
                    dict(
                        type="model.context.disconnect",
                        epoch=self.view.descriptor["epoch"],
                        consumerEpoch=self.consumer["consumerEpoch"],
                    )
                )
            except Exception:
                pass
        self.closed = True
        self.bound.set()
        if self.adapter:
            await self.adapter.close()
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.reset_task is not None:
            await asyncio.shield(self.reset_task)
        async with self.changed:
            self.changed.notify_all()
