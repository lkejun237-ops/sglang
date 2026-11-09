# SPDX-License-Identifier: Apache-2.0
"""One bounded Sink operation on an existing player's model connection."""

import asyncio
import copy

from .render_contract import RenderError


class SinkControl:
    def __init__(self, segment):
        self.segment = segment
        self.supported = False
        self.pending = None
        self.snapshot = None
        self.idle = asyncio.Event()
        self.idle.set()

    @property
    def busy(self):
        return self.pending is not None

    def current(self, state):
        current = state.get("segment")
        if not current or any(
            current.get(key) != getattr(self.segment, key)
            for key in ("segment_id", "segment_epoch")
        ):
            raise RenderError("SEGMENT_EPOCH_MISMATCH")
        return current

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
                "action",
            }
            or not isinstance(command.get("action"), str)
            or command["action"] not in {"on", "off", "refresh"}
        ):
            raise RenderError("INVALID_COMMAND")
        segment, owner = self.segment, self.segment.owner

        def admit(state, operation):
            current = owner.journal.require_segment(state, command)
            self.current(state)
            if not self.supported:
                raise RenderError("CAPABILITY_UNAVAILABLE")
            if current["state"] != "streaming" or self.snapshot is None:
                raise RenderError("MODEL_UNAVAILABLE")
            if (
                self.busy
                or getattr(owner, "pending_end", None) is not None
                or getattr(owner, "close_requested", False)
                or current.get("sink", {}).get("pending_request_id")
                or current.get("prompt_queue")
                or current.get("validating_prompt")
                or segment.pending_prompt is not None
                or segment.condition_switch is not None
                or segment.source_view.stopped.is_set()
                or segment.cancel_requested
            ):
                raise RenderError("SINK_UPDATE_BUSY", retryable=True)
            action = command["action"]
            noop = action != "refresh" and (action == "on") == self.snapshot["enabled"]
            operation.update(
                segment_id=segment.segment_id,
                segment_epoch=segment.segment_epoch,
                action=action,
                state="completed" if noop else "accepted",
                applied=noop,
                noop=noop,
            )
            if noop:
                operation["sink"] = copy.deepcopy(self.snapshot)
            else:
                current["sink"]["pending_request_id"] = command["request_id"]

        result = await owner.journal.command(command, admit)
        if result.get("duplicate") or result["state"] == "completed":
            await owner.operation_message(result)
            return
        self.pending = dict(
            request_id=command["request_id"],
            operation="off" if command["action"] == "off" else "refresh",
            event_id=None,
        )
        self.idle.clear()
        # Queue the receipt before the original model connection can publish output.
        await owner.operation_message(result)
        owner.spawn_control(self.dispatch(self.pending))

    async def dispatch(self, pending):
        segment, owner = self.segment, self.segment.owner
        try:
            async with segment.source_send_lock:
                current = self.current(await owner.journal.read())
                if (
                    current["state"] != "streaming"
                    or segment.source_view.stopped.is_set()
                    or segment.cancel_requested
                    or self.pending is not pending
                ):
                    await self.finish()
                    return

                async def remember(event_id):
                    # Persist the exact wire identity before sending a side effect.
                    pending["event_id"] = event_id

                    def reserve(state):
                        current = self.current(state)
                        if (
                            self.pending is not pending
                            or current["state"] != "streaming"
                            or current["sink"].get("pending_request_id")
                            != pending["request_id"]
                        ):
                            raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
                        state["operations"][pending["request_id"]].update(
                            state="dispatching", native_event_id=event_id
                        )

                    await owner.journal.update(reserve)
                    if (
                        self.pending is not pending
                        or segment.source_view.stopped.is_set()
                        or segment.cancel_requested
                    ):
                        raise RenderError("RENDER_TRANSITION_IN_PROGRESS")

                await segment.send_model_event(
                    segment.model_ws,
                    "sink." + pending["operation"],
                    {},
                    before_send=remember,
                )
            # Do not hold either source or wire lock while waiting for new RGB.
            await asyncio.wait_for(self.idle.wait(), segment.config.startup_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self.pending is pending:
                result = await owner.journal.result(
                    pending["request_id"],
                    state="reconciling",
                    code="SINK_UPDATE_UNKNOWN",
                )
                await owner.operation_message(result)
                # Keep the operation fenced. Late actual output may still settle it.

    async def control(self, message):
        pending = self.pending
        if pending is None or message.get("event_id") != pending["event_id"]:
            return False
        if message.get("type") == "control_ack" and message.get("kind") in {
            "sink.off",
            "sink.refresh",
        }:
            return True  # Native accepted is not proof of a changed output.
        return False

    async def output(self, message):
        if not self.supported:
            return None
        count, ids = message.get("sink_chunks"), message.get("sink_chunk_ids")
        event_id = message.get("sink_event_id")
        if (
            type(count) is not int
            or count not in (0, 2)
            or not isinstance(ids, list)
            or len(ids) > count
            or any(type(i) is not int or i < 0 for i in ids)
            or ids != sorted(set(ids))
        ):
            raise RenderError("CONDITION_INVALID")
        snapshot = dict(
            enabled=bool(count),
            sink_chunks=count,
            sink_chunk_ids=ids,
            sink_event_id=event_id,
        )
        pending = self.pending
        applied = (
            pending is not None
            and pending["event_id"] is not None
            and event_id == pending["event_id"]
        )
        if applied and bool(count) != (pending["operation"] == "refresh"):
            raise RenderError("CONDITION_INVALID")
        if snapshot == self.snapshot and not applied:
            return copy.deepcopy(snapshot)
        owner = self.segment.owner

        def update(state):
            current = self.current(state)
            current["sink"] = dict(
                supported=True,
                **snapshot,
                pending_request_id=None
                if applied or pending is None
                else pending["request_id"],
            )
            if applied:
                operation = state["operations"][pending["request_id"]]
                operation.update(
                    state="completed",
                    applied=True,
                    sink=copy.deepcopy(snapshot),
                    first_output_frame=message["output_frame_start"],
                )
                operation.pop("code", None)
                return operation

        result = await owner.journal.update(update, settlement=True)
        self.snapshot = copy.deepcopy(snapshot)
        if applied:
            self.pending = None
            self.idle.set()
            await owner.operation_message(result)
            self.segment.prompt_available.set()
        return copy.deepcopy(snapshot)

    async def finish(self):
        pending = self.pending
        if pending is None:
            return
        sent = pending["event_id"] is not None
        owner = self.segment.owner

        def settle(state):
            current = self.current(state)
            current["sink"]["pending_request_id"] = None
            operation = state["operations"][pending["request_id"]]
            operation.update(
                state="reconciling" if sent else "cancelled",
                code="SINK_UPDATE_UNKNOWN" if sent else "SEGMENT_ENDED",
                **({} if sent else {"never_applied": True}),
            )
            return operation

        result = await owner.journal.update(settle, settlement=True)
        self.pending = None
        self.idle.set()
        await owner.operation_message(result)
