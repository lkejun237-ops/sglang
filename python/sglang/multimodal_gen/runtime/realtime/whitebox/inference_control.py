# SPDX-License-Identifier: Apache-2.0
"""Player-owned inference children; the shared world never owns the model seat."""

import asyncio
import copy
import logging
import os
from types import SimpleNamespace
from typing import NamedTuple
from uuid import uuid4

import msgspec

from .diagnostics import public_error
from .render_contract import (
    TERMINAL_SEGMENTS,
    RenderError,
    prompt_text,
    segment_parameters,
    source_create_matches,
)


class InferenceTaskScope(NamedTuple):
    task: asyncio.Task | None
    revision: int
    source_generation: int | None
    request_id: str


class SourceInference:
    def __init__(self, source, message):
        self.source = source
        mode = message.get("inference_control")
        if mode not in (None, "manual"):
            raise RenderError("INVALID_COMMAND")
        self.manual = mode == "manual"
        self.assets = self.startup = None
        self.prewarm = None
        self.task = None
        self.task_scope = None
        self.recovery = self.recovery_task = None
        self.lock = asyncio.Lock()
        self.frozen = copy.deepcopy(message)
        if self.manual:
            if not source_create_matches(message):
                raise RenderError("INVALID_SOURCE")
            if message.get("startup") is not None:
                from .startup import startup_fingerprint, validate_startup

                if message.get("references") not in (None, []):
                    raise RenderError("CONDITION_INVALID")
                self.assets = validate_startup(message["startup"], source.config.spec)
                self.startup = SimpleNamespace(
                    assets=self.assets,
                    request=copy.deepcopy(message["startup"]),
                    request_fingerprint=startup_fingerprint(message["startup"]),
                    closed=False,
                )
            else:
                # Validate trusted immutable launch inputs before opening a World.
                prompt_text(message.get("prompt"))
                seed = message.get("seed", 42)
                if type(seed) is not int or not 0 <= seed < 2**32:
                    raise RenderError("CONDITION_INVALID")

    def fence(self, *, drain_pending=False):
        source = self.source
        if self.prewarm is not None:
            self.prewarm.invalidate()
        source.current_model_stream = None
        source.display_mode = "whitebox"
        source.revision = source.journal.snapshot.get("display_revision", 0)
        segment = source.current_segment
        if segment is not None:
            segment.publishing = False
            if not drain_pending:
                segment.source_view.stop()
        channel = source.context_channel
        if channel is not None:
            # Fence synchronously, before any socket, journal or cleanup await.
            channel.suspended = True

    def dispatch(self, operation):
        self.cancel_recovery()
        previous = self.task
        if previous is not None and not previous.done():
            previous.cancel()
        self.task = self.source.spawn_control(self.run(operation, previous))
        self.task_scope = InferenceTaskScope(
            task=self.task,
            revision=operation["inference"]["revision"],
            source_generation=self.source_generation(),
            request_id=operation["request_id"],
        )

    def current(self, revision):
        state = self.source.journal.snapshot["inference"]
        if state["revision"] != revision or not state["enabled"]:
            return False
        scope = getattr(self, "task_scope", None)
        if (
            scope is not None
            and scope.revision == revision
            and scope.source_generation != self.source_generation()
        ):
            raise RenderError("SOURCE_EPOCH_MISMATCH")
        return True

    def source_generation(self):
        return getattr(
            getattr(self.source, "context_channel", None),
            "source_generation",
            getattr(self.source, "native_source_generation", None),
        )

    def cancel_recovery(self):
        self.recovery = None
        task = getattr(self, "recovery_task", None)
        if task is not None and not task.done():
            task.cancel()

    def recovering(self, revision=None):
        recovery = getattr(self, "recovery", None)
        source = self.source
        current = source.journal.snapshot["inference"]
        return (
            recovery is not None
            and current["enabled"]
            and current["revision"] == recovery.revision
            and (revision is None or revision == recovery.revision)
            and not any(
                getattr(source, name, False)
                for name in ("closing", "close_requested", "closed", "binding_ended")
            )
        )

    def source_reset(self, generation, retiring):
        """Retain enabled intent; the authenticated channel owns retirement."""
        state = self.source.journal.snapshot["inference"]
        self.cancel_recovery()
        if (
            not self.manual
            or getattr(self.source, "member_prefill", None) is not None
            or not state["enabled"]
            or state.get("code") == "RELEASE_UNKNOWN"
            or any(
                getattr(self.source, name, False)
                for name in ("closing", "close_requested", "closed", "binding_ended")
            )
        ):
            return
        recovery = self.recovery = SimpleNamespace(
            revision=state["revision"],
            generation=generation,
            operation_id=uuid4().hex,
        )
        self.fence()
        # Never dispatch from inside reset_task: enable must await that exact
        # task, including adapter close, before it can reserve fresh compute.
        self.recovery_task = self.source.spawn_control(
            self.recover_source(recovery, retiring)
        )

    async def recover_source(self, recovery, retiring):
        source = self.source

        def current():
            return (
                self.recovery is recovery
                and self.recovering()
                and self.source_generation() == recovery.generation
            )

        async def status(state, code=None):
            def update(value):
                if not current():
                    return
                value["inference"].update(state=state)
                value["inference"].pop("code", None)
                value["inference"].pop("phase", None)
                if state == "starting":
                    value["inference"]["phase"] = "preparing_source"
                if code:
                    value["inference"]["code"] = code

            await source.journal.update(update, settlement=True)
            # Feedback cannot hold source retirement/close behind backpressure.
            source.spawn_control(source.status_snapshot())

        cleaned = False
        try:
            await status("starting")
            await asyncio.shield(retiring)
            cleaned = True
            if not current():
                return
            async with self.lock:
                if not current():
                    return
                if source.context_channel is not None:
                    # A reset before init has no admitted segment to retire,
                    # but its reserved identities still belong to the old SDK.
                    def fresh_reservation(value):
                        if value["segment"] is None:
                            value["model_context_segment_ids"] = None

                    await source.journal.update(fresh_reservation)
                    await source.journal.reserve_model_context_segment()
                operation = dict(
                    operation_id="source-reset:" + recovery.operation_id,
                    inference=dict(enabled=True, revision=recovery.revision),
                )
                self.task = asyncio.current_task()
                self.task_scope = InferenceTaskScope(
                    self.task, recovery.revision, recovery.generation, ""
                )
                await self.enable(operation)
                if current():
                    await status("enabled")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if current():
                code = public_error(
                    error, recovery.operation_id, default="INITIALIZATION_FAILED"
                )["code"]
                if cleaned:
                    try:
                        async with self.lock:
                            await self.disable(
                                reset_generation=recovery.generation,
                                expected_revision=recovery.revision,
                            )
                    except Exception:
                        code = "RELEASE_UNKNOWN"
                else:
                    code = "RELEASE_UNKNOWN"
                await status("failed", code)
        finally:
            if self.recovery is recovery:
                self.recovery = None

    async def run(self, operation, previous=None):
        if previous is not None:
            await asyncio.gather(previous, return_exceptions=True)
        revision = operation["inference"]["revision"]
        source = self.source
        try:
            # A new intent must not hold the inference lock while its native
            # source retirement needs that lock to finish original cleanup.
            retiring = getattr(
                getattr(source, "context_channel", None),
                "reset_task",
                getattr(source, "native_reset_task", None),
            )
            if operation["inference"]["enabled"] and retiring is not None:
                await asyncio.shield(retiring)
            async with self.lock:
                if operation["inference"]["enabled"]:
                    await self.enable(operation)
                    state = "enabled"
                else:
                    await self.disable(drain_pending=True)
                    state = "disabled"
            result = await source.journal.settle_inference(
                operation["request_id"], revision, state=state
            )
            await source.operation_message(result)
            await source.status_snapshot()
        except asyncio.CancelledError:
            if (
                not getattr(source, "closing", False)
                and source.journal.snapshot["inference"]["revision"] != revision
            ):
                result = await source.journal.settle_inference(
                    operation["request_id"], revision, state="completed"
                )
                await source.operation_message(result)
            raise
        except Exception as error:
            # Stop the failed model child. Managed players retain their parent
            # Slot until player teardown; no UI can infer that its seat is free.
            self.fence()
            # Admission rejections carry reason, not RenderError.code. Reuse
            # the public allowlist so capacity failures retain their cause.
            code = public_error(
                error, operation["request_id"], default="INITIALIZATION_FAILED"
            )["code"]
            if operation["inference"]["enabled"]:
                try:
                    scope = getattr(self, "task_scope", None)
                    generation = self.source_generation()
                    if (
                        scope is not None
                        and scope.revision == revision
                        and scope.source_generation != generation
                    ):
                        await self.disable(
                            reset_generation=generation, expected_revision=revision
                        )
                    else:
                        await self.disable()
                except Exception:
                    code = "RELEASE_UNKNOWN"
            result = await source.journal.settle_inference(
                operation["request_id"],
                revision,
                state="failed",
                code=code,
                preserve_intent=self.recovering(revision),
            )
            await source.operation_message(result)
            await source.status_snapshot()

    async def enable(self, operation):
        source = self.source
        revision = operation["inference"]["revision"]
        if source.closed or source.binding_ended:
            raise RenderError("INVALID_SOURCE")
        if getattr(source, "first_presented_required", False):
            source.require_opening_presentation_player()
        channel = source.context_channel
        if self.manual and channel is not None:
            # A paused handshake has no model-condition delivery. Validate the
            # actual sealed World binding before reserving any model compute.
            await channel.probe_subscription_pause(
                timeout=source.config.startup_seconds
            )
        if getattr(source, "first_presented_required", False):
            source.require_opening_presentation_player()
        if not self.current(revision):
            return
        # Unknown previous cleanup cannot be bypassed by creating a new parent.
        binding = source.model_binding
        if binding is not None and binding.lease.resource_state not in {
            "none",
            "released",
        }:
            if binding.closed:
                raise RenderError("RELEASE_UNKNOWN")
        if binding is None:
            source.start_model_binding(self.frozen)
        if source.model_binding is not None:
            await source.model_binding.wait_ready()
        if not self.current(revision):
            return
        channel = source.context_channel
        if channel is not None:
            state = await source.journal.read()
            reserved, segment = state["model_context_segment_ids"], state["segment"]
            pending_context = (
                reserved is not None
                and segment is not None
                and reserved["segment_id"] != segment["segment_id"]
            )
            if pending_context:
                # An authenticated reserved successor has not received a condition
                # yet. Reuse it, but never bypass original execution/release gates.
                if segment["state"] not in TERMINAL_SEGMENTS:
                    raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
                if segment["resource_state"] not in {"none", "released"}:
                    raise RenderError("RELEASE_UNKNOWN")
            if segment is not None and not pending_context:
                command = self.command("render.context.start", operation, state)
                result = await source.journal.start_model_context_segment(command)
                await self.open_prewarm(result["segment_id"], revision)
                await source.handoff_context_segment(
                    result, await source.journal.read()
                )
                await source.operation_message(result)
                expected_segment_id = result["segment_id"]
            else:
                expected_segment_id = state["model_context_segment_ids"]["segment_id"]
                await self.open_prewarm(expected_segment_id, revision)
                if not self.current(revision):
                    return
                await channel.switch_segment(expected_segment_id)
        else:
            async with asyncio.timeout(source.config.startup_seconds):
                while not source.retained.frames:
                    if not self.current(revision):
                        return
                    await asyncio.sleep(0.01)
            if not self.current(revision):
                return
            version, frame, captured, _ = source.retained.frames[-1]
            state = await source.journal.read()
            command = self.command("render.start", operation, state)
            command.update(
                prompt=self.assets.prompt if self.assets else self.frozen["prompt"],
                seed=self.assets.seed if self.assets else self.frozen.get("seed", 42),
                references=copy.deepcopy(source.initial_assets),
                start_boundary=dict(source_frame_index=frame.index),
            )
            segment_parameters(command)
            references = await source.assets.resolve(command["references"])
            if not self.current(revision):
                return
            result = await source.journal.start(command)
            await source.operation_message(result)
            expected_segment_id = result["segment_id"]
            source.current_model_stream = result["segment_epoch"]
            await source.outgoing.put(
                msgspec.msgpack.encode(
                    dict(
                        version=2,
                        type="render.stream",
                        segment_id=result["segment_id"],
                        segment_epoch=result["segment_epoch"],
                        media_stream_id=result["segment_epoch"],
                        render_revision=result["render_revision"],
                        **source.source_identity(),
                    )
                )
            )
            # The original source control owner must survive cancellation of
            # this admission waiter, including archive setup before child run.
            source.spawn_control(
                source._transition_segment(
                    result, references, (version, (frame, captured))
                )
            )
        # The source keeps accepting input while the original child establishes
        # its model connection. Enabled means admission, never visible pixels.
        async with asyncio.timeout(source.config.startup_seconds):
            while self.current(revision):
                if self.prewarm is not None and self.prewarm.failure is not None:
                    raise self.prewarm.failure
                segment = source.current_segment
                if segment is not None and segment.segment_id == expected_segment_id:
                    if segment.closed_event.is_set():
                        raise RenderError(
                            segment.failure_code or "INITIALIZATION_FAILED"
                        )
                    if segment.model_ready.is_set():
                        return
                await asyncio.sleep(0.01)

    async def open_prewarm(self, segment_id, revision):
        source = self.source
        if (
            os.environ.get("WHITEBOX_MODEL_CONTEXT_PREWARM") != "1"
            or not self.manual
            or source.control_prefill is None
            or source.model_binding is None
            or getattr(source, "member_prefill", None) is not None
        ):
            return
        from .startup_prewarm import StartupPrewarm
        from .startup_prewarm_plan import PROTOCOL, StartupPrewarmPlan

        owner = StartupPrewarm
        if getattr(source.engine, "shared_world", False):
            capability = getattr(source.engine, "model_context_capability", None) or {}
            if capability.get("prewarmPlanProtocol") != PROTOCOL:
                return
            # The descriptor authorizes this route. The active subscription
            # must separately acknowledge it before startup_release opens it.
            owner = StartupPrewarmPlan

        if self.prewarm is not None:
            if (self.prewarm.segment_id, self.prewarm.opening_revision) == (
                segment_id,
                revision,
            ):
                return
            await self.prewarm.close()
        self.prewarm = owner(
            source,
            segment_id,
            revision,
            asyncio.get_running_loop().time() + source.config.startup_seconds,
        )

    def command(self, op, operation, state):
        return dict(
            type="command",
            op=op,
            request_id="inference:" + operation["operation_id"] + ":" + op,
            source_session_id=self.source.session_id,
            source_epoch=self.source.source_epoch,
            expected_render_revision=state["render_revision"],
        )

    async def disable(
        self, *, drain_pending=False, reset_generation=None, expected_revision=None
    ):
        source = self.source
        self.fence(drain_pending=drain_pending)
        if reset_generation is None:
            await self.pause_channel()
        cleanup_unknown = False
        if self.prewarm is not None:
            try:
                await self.prewarm.close()
                self.prewarm = None
            except RenderError:
                cleanup_unknown = True
        async with source.segment_lock:
            binding = source.model_binding
            managed_player = (
                binding is not None
                and getattr(
                    getattr(binding, "assignment", None), "player_expires_at", None
                )
                is not None
            )
            segment = source.current_segment
            if segment is not None:
                try:
                    send_lock = getattr(segment, "source_send_lock", None)
                    if drain_pending and (
                        send_lock is not None or segment.pending_prompt is not None
                    ):
                        deadline = (
                            asyncio.get_running_loop().time()
                            + source.config.drain_seconds
                            + 10
                        )
                        # A sender can hold this lock while waiting for the
                        # model wire, before it records pending_prompt. Keep
                        # feeding until that dispatch boundary is resolved.
                        if send_lock is not None:
                            try:
                                async with asyncio.timeout_at(deadline):
                                    async with send_lock:
                                        pass
                            except TimeoutError:
                                # Cancel below; only the original dispatch
                                # evidence may distinguish unsent from unknown.
                                pass
                        if segment.pending_prompt is not None:
                            await source.drain_pending_condition(segment, deadline)
                    await segment.stop(cancel=True)
                except Exception:
                    cleanup_unknown = True
            if source.startup is not None:
                try:
                    await source.startup.close()
                    source.startup = None
                except Exception:
                    cleanup_unknown = True
            if source.control_prefill is not None:
                try:
                    await source.control_prefill.close()
                    if managed_player:
                        from .control_prefill import ControlPrefill

                        source.control_prefill = ControlPrefill(source)
                        binding.on_receipt = source.control_prefill.receipt
                except Exception:
                    cleanup_unknown = True
            if binding is not None and not managed_player:
                await binding.close()
                if binding.lease.resource_state not in {"none", "released"}:
                    raise RenderError("RELEASE_UNKNOWN")
                source.model_binding = source.control_prefill = None
            elif managed_player and binding.closed:
                cleanup_unknown = True
            if segment is not None and segment.resource_state not in {
                "none",
                "released",
            }:
                retry = getattr(segment, "retry_release", None)
                if retry is not None:
                    try:
                        await retry()
                    except Exception:
                        cleanup_unknown = True
                if segment.resource_state not in {"none", "released"}:
                    cleanup_unknown = True
        if cleanup_unknown:
            raise RenderError("RELEASE_UNKNOWN")
        state = await source.journal.read()
        # An accepted child that never acquired compute is safely terminal. Do
        # not settle dispatched conditions; their original recovery barriers stay.
        segment = state["segment"]
        if (
            segment
            and segment["state"] in {"accepted", "waiting_capacity"}
            and segment["resource_state"] == "none"
        ):

            def cancel_unstarted(value):
                value["segment"].update(state="cancelled")
                original = value["operations"][segment["request_id"]]
                original.update(
                    state="cancelled",
                    never_applied=True,
                    code="MODEL_INFERENCE_DISABLED",
                )

            await source.journal.update(cancel_unstarted, settlement=True)
        source.display_mode = "whitebox"

        def raw_display(state):
            if (
                expected_revision is not None
                and state["inference"]["revision"] != expected_revision
            ):
                return  # An accepted intent may still be waiting to dispatch.
            scope = getattr(self, "task_scope", None)
            if (
                reset_generation is not None
                and scope is not None
                and scope.source_generation is not None
                and scope.source_generation >= reset_generation
            ):
                return  # A newer explicit intent owns this display selection.
            if state["display_mode"] != "whitebox":
                state["display_mode"] = "whitebox"
                state["display_revision"] += 1

        await source.journal.update(raw_display, settlement=True)
        source.display_mode = source.journal.snapshot["display_mode"]
        source.revision = source.journal.snapshot["display_revision"]
        # Native retirement already owns an independent, cancellable preview
        # sender. Repeating that send here would hold original settlement and
        # close behind transport backpressure after resources were released.

    async def pause_channel(self):
        channel = self.source.context_channel
        if channel is not None:
            try:
                await channel.pause()
            except Exception:
                # Local admission is already fenced. A preview transport fault
                # must not keep a model reservation alive.
                logging.getLogger(__name__).warning(
                    "Preview transport did not acknowledge inference pause",
                    exc_info=True,
                )

    async def segment_ended(self, segment, code, *, revision=None, notify=True):
        source = self.source
        if (
            not self.manual
            or self.recovering(revision)
            or getattr(source, "closing", False)
            or getattr(source, "close_requested", False)
            or getattr(source, "member_prefill", None) is not None
        ):
            return

        def failed(state):
            inference = state["inference"]
            if (
                not inference["enabled"]
                or source.current_segment is not segment
                or revision is not None
                and inference["revision"] != revision
            ):
                return False
            inference.update(state="failed", code=code or "MODEL_UNAVAILABLE")
            if state["display_mode"] != "whitebox":
                state["display_mode"] = "whitebox"
                state["display_revision"] += 1
            return True

        if await source.journal.update(failed, settlement=True):
            self.fence()
            if notify:
                await self.pause_channel()
                await source.status_snapshot()

    async def retire_preparation(self, code):
        """Retire the reset opening, independent of prewarm/segment creation."""
        scope = getattr(self, "task_scope", None)
        generation = self.source_generation()
        newer_intent = (
            scope is not None
            and generation is not None
            and scope.source_generation == generation
        )
        task = scope.task if scope is not None else self.task
        if newer_intent:
            task = None
        revision = (
            scope.revision
            if scope is not None
            else self.source.journal.snapshot["inference"]["revision"]
        )
        self.fence()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        try:
            async with self.lock:
                await self.disable(
                    reset_generation=generation, expected_revision=revision
                )
        except Exception:
            if not newer_intent:
                # Reporting cannot replace the original cleanup failure.
                try:
                    await self.finish_retired_opening(
                        scope, revision, "RELEASE_UNKNOWN"
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Retired inference control settlement failed"
                    )
            raise
        if not newer_intent:
            await self.finish_retired_opening(scope, revision, code)

    async def finish_retired_opening(self, scope, revision, code):
        # Current intent and the original command are distinct journal records.
        # Retirement already owns preview selection; repeating transport pause
        # here could hide a cleanup failure behind socket backpressure.
        await self.segment_ended(
            self.source.current_segment, code, revision=revision, notify=False
        )
        operation = None
        if scope is not None:
            operation = self.source.journal.snapshot["operations"].get(scope.request_id)
            if operation is not None and operation["state"] == "accepted":
                operation = await self.source.journal.settle_inference(
                    scope.request_id,
                    scope.revision,
                    state="failed",
                    code=code,
                    preserve_intent=self.recovering(revision),
                )
            operation = copy.deepcopy(operation)

        async def publish():
            if operation is not None:
                await self.source.operation_message(operation)
            await self.source.status_snapshot()

        # Close may stop the outgoing reader; feedback must not gate retirement.
        self.source.spawn_control(publish())

    async def close(self):
        self.cancel_recovery()
        recovery = getattr(self, "recovery_task", None)
        if recovery is not None:
            await asyncio.gather(recovery, return_exceptions=True)
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.startup is not None:
            self.startup.closed = True
        if self.assets is not None:
            self.assets.clear()
        self.frozen.clear()
        if self.prewarm is not None:
            await self.prewarm.close()
