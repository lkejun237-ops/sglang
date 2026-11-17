# SPDX-License-Identifier: Apache-2.0
"""A journalled timing override scoped to one active model generation."""

import asyncio

from .render_contract import RenderError


class TimingControl:
    def __init__(self, segment):
        self.segment = segment
        self.supported = False
        self.revision = 0
        self.ever_enabled = segment.detailed_timing
        self.pending = None
        self.idle = asyncio.Event()
        self.idle.set()

    def snapshot(self):
        return dict(
            supported=self.supported,
            enabled=self.segment.detailed_timing,
            revision=self.revision,
            pending_request_id=self.pending["request_id"] if self.pending else None,
        )

    async def command(self, command):
        if (
            set(command)
            - {
                "version",
                "type",
                "op",
                "request_id",
                "source_session_id",
                "source_epoch",
                "segment_id",
                "segment_epoch",
                "enabled",
                "expected_timing_revision",
            }
            or type(command.get("enabled")) is not bool
        ):
            raise RenderError("INVALID_COMMAND")
        segment, owner = self.segment, self.segment.owner

        def admit(state, operation):
            current = owner.journal.require_segment(state, command)
            if (current["segment_id"], current["segment_epoch"]) != (
                segment.segment_id,
                segment.segment_epoch,
            ):
                raise RenderError("SEGMENT_EPOCH_MISMATCH")
            if not self.supported:
                raise RenderError("CAPABILITY_UNAVAILABLE")
            if current["state"] != "streaming" or segment.cancel_requested:
                raise RenderError("MODEL_UNAVAILABLE")
            if (
                self.pending is not None
                or getattr(owner, "pending_end", None) is not None
            ):
                raise RenderError("TIMING_UPDATE_BUSY", retryable=True)
            if (
                type(command.get("expected_timing_revision")) is not int
                or command["expected_timing_revision"] != self.revision
            ):
                raise RenderError("REVISION_CONFLICT")
            current["timing"] = dict(
                self.snapshot(), pending_request_id=command["request_id"]
            )
            operation.update(
                segment_id=segment.segment_id,
                segment_epoch=segment.segment_epoch,
                state="accepted",
                enabled=command["enabled"],
                timing_revision=self.revision + 1,
            )

        result = await owner.journal.command(command, admit)
        if result.get("duplicate"):
            await owner.operation_message(result)
            return
        self.pending = dict(result, event_id=None)
        self.idle.clear()
        await owner.operation_message(result)
        owner.spawn_control(self.dispatch(self.pending))

    async def dispatch(self, pending):
        segment, owner = self.segment, self.segment.owner
        try:
            async with segment.source_send_lock:
                if (
                    self.pending is not pending
                    or segment.cancel_requested
                    or segment.source_view.stopped.is_set()
                ):
                    await self.finish()
                    return
                assignment = segment.assignment

                async def remember(event_id):
                    pending["event_id"] = event_id

                    def reserve(state):
                        current = state.get("segment")
                        if (
                            self.pending is not pending
                            or segment.cancel_requested
                            or segment.source_view.stopped.is_set()
                            or not current
                            or current["segment_id"] != segment.segment_id
                            or current["segment_epoch"] != segment.segment_epoch
                            or current["state"] != "streaming"
                        ):
                            raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
                        state["operations"][pending["request_id"]].update(
                            state="dispatching", native_event_id=event_id
                        )

                    await owner.journal.update(reserve)
                    if (
                        self.pending is not pending
                        or segment.cancel_requested
                        or segment.source_view.stopped.is_set()
                    ):
                        raise RenderError("RENDER_TRANSITION_IN_PROGRESS")

                # Receipts can arrive before ACK, so accept current-generation in-flight
                # reports as soon as ON is dispatched, without claiming it is applied.
                self.ever_enabled |= pending["enabled"]
                await segment.send_model_event(
                    segment.model_ws,
                    "timing.configure",
                    dict(
                        session_id=assignment.session_id,
                        generation_id=assignment.generation_id,
                        enabled=pending["enabled"],
                        timing_revision=pending["timing_revision"],
                    ),
                    before_send=remember,
                )
            await asyncio.wait_for(self.idle.wait(), segment.config.startup_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self.pending is pending:
                await owner.operation_message(
                    await owner.journal.result(
                        pending["request_id"],
                        state="reconciling",
                        code="TIMING_UPDATE_UNKNOWN",
                    )
                )
                # Do not retry an uncertain distributed mutation or report it as OFF.

    async def control(self, message):
        if (
            message.get("type") != "control_ack"
            or message.get("kind") != "timing.configure"
        ):
            return False
        pending, segment = self.pending, self.segment
        assignment = segment.assignment
        if (
            pending is None
            or assignment is None
            or message.get("event_id") != pending["event_id"]
            or message.get("model_session_id") != assignment.session_id
            or message.get("generation_id") != assignment.generation_id
            or message.get("enabled") is not pending["enabled"]
            or message.get("timing_revision") != pending["timing_revision"]
        ):
            return True
        boundaries = message.get("boundaries")
        fields = {
            "encoder": "command_index",
            "dit": "source_frame_end",
            "decoder": "output_frame_start",
        }
        if (
            not isinstance(boundaries, dict)
            or set(boundaries) != set(fields)
            or any(
                not isinstance(boundaries[role], dict)
                or type(boundaries[role].get(field)) is not int
                or boundaries[role][field] < 0
                for role, field in fields.items()
            )
        ):
            return True
        boundaries = {
            role: {field: boundaries[role][field]} for role, field in fields.items()
        }
        owner = segment.owner

        def settle(state):
            current = state.get("segment")
            if (
                not current
                or current["segment_id"] != segment.segment_id
                or current["segment_epoch"] != segment.segment_epoch
                or current["state"] != "streaming"
                or segment.cancel_requested
                or self.pending is not pending
            ):
                return None
            current["timing"] = dict(
                supported=True,
                enabled=pending["enabled"],
                revision=pending["timing_revision"],
                pending_request_id=None,
            )
            operation = state["operations"][pending["request_id"]]
            operation.update(state="completed", applied=True, boundaries=boundaries)
            operation.pop("code", None)
            return operation

        result = await owner.journal.update(settle, settlement=True)
        if result is None:
            return True
        segment.detailed_timing = pending["enabled"]
        self.revision = pending["timing_revision"]
        if segment.recording is not None:
            segment.recording.manifest.setdefault("timing_changes", []).append(
                dict(
                    request_id=pending["request_id"],
                    enabled=pending["enabled"],
                    timing_revision=self.revision,
                    boundaries=boundaries,
                )
            )
        self.pending = None
        self.idle.set()
        await owner.operation_message(result)
        return True

    async def finish(self):
        pending = self.pending
        if pending is None:
            return
        self.pending = None
        self.idle.set()
        sent = pending["event_id"] is not None

        def settle(state):
            current = state.get("segment")
            if current and (current["segment_id"], current["segment_epoch"]) == (
                self.segment.segment_id,
                self.segment.segment_epoch,
            ):
                current.get("timing", {}).update(pending_request_id=None)
            operation = state["operations"][pending["request_id"]]
            operation.update(
                state="reconciling" if sent else "cancelled",
                code="TIMING_UPDATE_UNKNOWN" if sent else "SEGMENT_ENDED",
            )
            return operation

        owner = self.segment.owner
        await owner.operation_message(
            await owner.journal.update(settle, settlement=True)
        )
