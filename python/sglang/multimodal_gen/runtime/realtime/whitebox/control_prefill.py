# SPDX-License-Identifier: Apache-2.0
"""Optional prefill requests on the already-bound source control connection."""

import asyncio
import hashlib
import time
from uuid import uuid4

import msgspec

from .contracts import capture_pts
from .render_contract import (
    OCCUPIED_SEGMENTS,
    RenderError,
    canonical,
    identity,
    segment_parameters,
)

PROTOCOL = "whitebox-control-prefill-v1"
MAX_REQUESTS = 256


class ControlPrefill:
    def __init__(self, owner):
        self.owner = owner
        self.context_id = None
        self.first = None
        self.pending_frame = None
        self.requests = {}
        self.active = None
        self.closed = False
        self.expires = 0
        self.pending_proof = None
        self.resumed = True
        self.paused_ms = 0.0
        self.snapshot_mode = False
        self.snapshot_after = 0
        self.snapshot_consumed = False
        self.resume_after = None
        self.recovery = None
        self.failure = None

    def _activity(self):
        runtime = getattr(getattr(self.owner, "host", None), "runtime", None)
        stream = getattr(getattr(self.owner, "engine", None), "stream_id", None)
        return getattr(runtime, "players", {}).get(stream, {}).get("input_activity")

    def _pause_idle(self):
        activity = self._activity()
        if not self.snapshot_mode and activity is not None and not self.closed:
            activity.pause(self.context_id)

    def _resume_idle(self):
        activity = self._activity()
        if not self.snapshot_mode and activity is not None and not self.closed:
            activity.resume(self.context_id, time.monotonic())

    def remaining(self):
        remaining = min(
            self.owner.config.startup_seconds, self.expires - time.monotonic()
        )
        if remaining <= 0:
            raise RenderError("START_SOURCE_EXPIRED")
        return remaining

    def check_deadline(self, now):
        if self.failure is not None:
            raise self.failure
        if (
            self.expires
            and now >= self.expires
            and (not self.resumed or self.snapshot_mode and not self.snapshot_consumed)
            and self.recovery is None
            and not self.closed
        ):
            self.first = None  # Retire the old F0 before cancellation or source resume.
            self.recovery = self.owner.spawn_control(self._expire())

    async def _expire(self):
        try:
            if self.pending_frame is not None and not self.pending_frame.done():
                self.pending_frame.cancel()
            if self.active is not None and not self.active.done():
                self.active.cancel()
                # Source recovery does not grant model reuse while cleanup drains.
                await self.resume(self.context_id)
                done, _ = await asyncio.wait((self.active,), timeout=90)
                if not done:
                    raise RenderError("RELEASE_UNKNOWN")
            else:
                await self.resume(self.context_id)
        except Exception:
            self.failure = RenderError("RELEASE_UNKNOWN")

    def observe(self, version, frame, captured):
        if self.resume_after is not None and version > self.resume_after:
            self.resume_after = None
            self._resume_idle()
        if (
            self.snapshot_mode
            and self.pending_frame is not None
            and not self.pending_frame.done()
            and version > self.snapshot_after
        ):
            header = (frame.provenance or {}).get("cli_frame_header", {})
            if all(
                header.get(key) is not None
                for key in ("sessionId", "epoch", "mediaGeneration", "outputFrameId")
            ):
                # This stream is the existing player's private SDK view. A
                # fresh native frame pins F0 without stopping the shared clock.
                self.pending_proof = dict(header)
        if (
            self.pending_frame is not None
            and not self.pending_frame.done()
            and self.pending_proof is not None
        ):
            header = (frame.provenance or {}).get("cli_frame_header", {})
            if all(
                header.get(key) == self.pending_proof.get(key)
                for key in ("sessionId", "epoch", "mediaGeneration", "outputFrameId")
            ):
                self.pending_frame.set_result((version, (frame, captured)))

    async def available(self):
        state = await self.owner.journal.read()
        if self.closed or self.owner.binding_ended:
            raise RenderError("INVALID_SOURCE")
        if self.failure is not None:
            raise self.failure
        if self.recovery is not None and not self.recovery.done():
            raise RenderError("PREFILL_BUSY")
        if (
            state.get("pending_replace")
            or (state.get("segment") or {}).get("state") in OCCUPIED_SEGMENTS
        ):
            raise RenderError("PREFILL_BUSY")
        return state

    async def prepare(self, command):
        """Explicit context boundary; a future real frame becomes video F0."""
        state = await self.available()
        self.owner.journal.require_revision(state, command)
        if (
            self.active is not None
            and not self.active.done()
            or self.pending_frame is not None
        ):
            raise RenderError("PREFILL_BUSY")
        if not self.resumed:
            await self.resume(self.context_id)
        self.context_id = identity(command["request_id"])
        self.first = None
        self.snapshot_mode = bool(getattr(self.owner.engine, "shared_world", False))
        self.snapshot_after = (
            getattr(self.owner.latest, "version", 0) if self.snapshot_mode else 0
        )
        self.snapshot_consumed = False
        self.resume_after = None
        self.recovery = None
        self.expires = time.monotonic() + (120 if self.snapshot_mode else 270)
        self.paused_ms = 0.0
        self.pending_proof = None
        self.pending_frame = asyncio.get_running_loop().create_future()
        result = await self.owner.journal.command(
            command, lambda s, o: o.update(state="accepted")
        )
        await self.owner.operation_message(result)
        self._pause_idle()

        async def pin():
            try:
                # CLI stops its authoritative simulation clock and publishes
                # exactly one fresh F0. It does not stream repeated still frames.
                if not self.snapshot_mode:
                    self.resumed = False
                    held = await self.owner.engine.capture_control(
                        "hold", self.context_id
                    )
                    self.pending_proof = held["frame"]
                    if not isinstance(self.pending_proof, dict) or any(
                        self.pending_proof.get(k) is None
                        for k in (
                            "sessionId",
                            "epoch",
                            "mediaGeneration",
                            "outputFrameId",
                        )
                    ):
                        raise RenderError("CONDITION_INVALID")
                    for version, frame, captured, _ in self.owner.retained.frames:
                        self.observe(version, frame, captured)
                self.first = await asyncio.wait_for(
                    self.pending_frame, self.remaining()
                )
                frame = self.first[1][0]
                receipt = await self.owner.journal.result(
                    command["request_id"],
                    state="completed",
                    context_id=self.context_id,
                    source_frame_index=frame.index,
                    source_pts_us=capture_pts(frame),
                    source_frame_sha256=hashlib.sha256(frame.rgb).hexdigest(),
                    source_mode="private_view_snapshot"
                    if self.snapshot_mode
                    else "exclusive_capture_hold",
                )
                await self.owner.operation_message(receipt)
            except asyncio.CancelledError:
                await self.owner.journal.result(
                    command["request_id"], state="cancelled"
                )
                raise
            except Exception:
                try:
                    await self.resume(self.context_id)
                except Exception:
                    pass  # Source close/hold expiry owns any uncertain recovery.
                self.first = None
                await self.owner.operation_message(
                    await self.owner.journal.result(
                        command["request_id"],
                        state="failed",
                        code="START_SOURCE_EXPIRED",
                    )
                )
            finally:
                self.pending_frame = None
                self.pending_proof = None

        self.owner.spawn_control(pin())

    async def resume(self, context_id):
        if context_id != self.context_id:
            raise RenderError("PREFILL_INPUT_MISMATCH")
        if self.snapshot_mode:
            if not self.snapshot_consumed and self.first is not None:
                latest = self.owner.latest.value
                if latest is not None:
                    self.paused_ms = (
                        max(0, capture_pts(latest[0]) - capture_pts(self.first[1][0]))
                        / 1000
                    )
                self.snapshot_consumed = True
            return  # Never send hold/resume to a shared World.
        if not self.resumed:
            self._pause_idle()
            after = getattr(self.owner.latest, "version", 0)
            result = await self.owner.engine.capture_control("resume", context_id)
            self.resumed = True
            self.resume_after = after
            if getattr(self.owner.latest, "version", 0) > after:
                self.resume_after = None
                self._resume_idle()
            self.paused_ms = float(result.get("pausedMs", 0))
            await self.owner.outgoing.put(
                msgspec.msgpack.encode(
                    dict(
                        type="prefill.source_resumed",
                        context_id=context_id,
                        paused_ms=result.get("pausedMs"),
                    )
                )
            )

    def resolve(self, context_id):
        if (
            context_id != self.context_id
            or self.first is None
            or time.monotonic() >= self.expires
        ):
            raise RenderError("START_SOURCE_EXPIRED")
        return self.first

    async def receipt(self, message):
        request = self.requests.get(message.get("request_id"))
        if request is None:
            return
        kind = message.get("type")
        first_ready = kind == "prefill.memory_ready" and not request.get(
            "ready_emitted"
        )
        persistence = kind == "prefill.persisted" or (
            kind == "prefill.failed" and message.get("memory_ready") is True
        )
        if persistence and not request.get("ready_emitted"):
            request["pending_persistence"] = message
            return
        if kind == "prefill.memory_ready":
            request["ready_emitted"] = True
        # Do not expose disk paths or internal exception strings on the play
        # socket. Native cache/timing fields are forwarded separately below.
        allowed = {
            "type",
            "request_id",
            "cache_key",
            "cache_result",
            "hit",
            "timings",
            "prefill_to_memory_ms",
            "prefill_to_persisted_ms",
            "memory_ready",
            "code",
            "cache",
            "timing",
            "rank_timings",
            "entries",
            "evictions",
            "prompt_version",
        }
        public = {k: v for k, v in message.items() if k in allowed}
        public.update(
            **self.owner.source_identity(),
            context_id=request["context_id"],
            worker_epoch=self.owner.model_binding.assignment.denoiser.worker_epoch,
        )
        if isinstance(message.get("session_id"), str):
            public["model_session_id"] = message["session_id"]
        public["worker_id"] = self.owner.model_binding.assignment.denoiser.worker_id
        previous = request.get("receipt", {})
        if (
            previous.get("type") == "prefill.persisted"
            and public.get("type") == "prefill.memory_ready"
        ):
            public = {**public, **previous}
        if persistence and kind == "prefill.failed":
            request["receipt"] = {
                **previous,
                "persistence_state": "failed",
                "persistence_code": public.get("code"),
            }
        else:
            request["receipt"] = public
        recording = getattr(self.owner, "recording", None)
        if recording is not None:
            # The existing manifest finalizer persists metadata; no synchronous
            # storage operation is inserted before memory_ready or playback.
            recording.manifest.setdefault("prefill", {})[message["request_id"]] = dict(
                request["receipt"], observed_epoch_ms=time.time() * 1000
            )
        if "startup_observer" in request:
            request["startup_observer"](public)
        else:
            await self.owner.outgoing.put(msgspec.msgpack.encode(public))
        if (
            first_ready
            and request["context_id"] == self.context_id
            and self.first is not None
        ):
            # Ready to accept the user's start command, even while F0 stays held.
            self._resume_idle()
        if (
            kind == "prefill.memory_ready"
            and request.get("pending_persistence") is not None
        ):
            await self.receipt(request.pop("pending_persistence"))

    async def start_startup(self, value, refs, first, *, current, observe):
        """Borrow one child for a trusted startup window on its pinned capture."""
        await self.available()
        parameters = segment_parameters(dict(value, references=refs))
        references = await self.owner.assets.resolve(parameters["references"])
        if not current():
            return None
        if (
            self.active is not None
            and not self.active.done()
            or self.pending_frame is not None
            or not self.resumed
        ):
            raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
        if len(self.requests) >= MAX_REQUESTS:
            raise RenderError("LIMIT_EXCEEDED")
        request_id = uuid4().hex
        self.requests[request_id] = dict(
            context_id=None, receipt={}, startup_observer=observe
        )

        async def run():
            await self.receipt(dict(type="prefill.accepted", request_id=request_id))
            await self._compute(request_id, parameters, references, first)

        self.active = asyncio.create_task(run())
        return self.active

    async def submit(self, message):
        request_id = message.get("request_id")
        try:
            identity(request_id)
            if (
                set(message)
                - {
                    "type",
                    "eventType",
                    "version",
                    "request_id",
                    "payload",
                    "source_session_id",
                    "source_epoch",
                }
                or message.get("type") != "event"
                or message.get("eventType") != "prefill"
            ):
                raise RenderError("INVALID_COMMAND")
            state = await self.owner.journal.read()
            self.owner.journal.validate_scope(state, message)
            payload = message.get("payload")
            if not isinstance(payload, dict) or set(payload) - {
                "context_id",
                "prompt",
                "seed",
                "references",
            }:
                raise RenderError("INVALID_COMMAND")
            parameters = segment_parameters(payload)
            member_prefill = getattr(self.owner, "member_prefill", None)
            if member_prefill is not None:
                member_prefill.validate_conditions(payload)
            fingerprint = hashlib.sha256(canonical(message)).hexdigest()
            previous = self.requests.get(request_id)
            if previous is not None:
                if previous.get("fingerprint") != fingerprint:
                    raise RenderError("REQUEST_ID_CONFLICT")
                await self.owner.outgoing.put(
                    msgspec.msgpack.encode({**previous["receipt"], "duplicate": True})
                )
                return
            await self.available()
            if self.active is not None and not self.active.done():
                raise RenderError("PREFILL_BUSY")
            if len(self.requests) >= MAX_REQUESTS:
                raise RenderError("LIMIT_EXCEEDED")
            first = self.resolve(payload.get("context_id"))
            references = await self.owner.assets.resolve(parameters["references"])
            first = self.resolve(payload.get("context_id"))
            if member_prefill is not None:
                member_prefill.validate_conditions(payload)
                first = self.resolve(payload.get("context_id"))
            self.requests[request_id] = dict(
                fingerprint=fingerprint, context_id=self.context_id, receipt={}
            )
            if member_prefill is not None:
                from .member_prefill import condition_digest

                self.requests[request_id]["condition_sha256"] = condition_digest(
                    payload
                )
            await self.receipt(dict(type="prefill.accepted", request_id=request_id))
            self._pause_idle()
            self.active = asyncio.create_task(
                self._compute(request_id, parameters, references, first)
            )
        except RenderError as error:
            await self.owner.outgoing.put(
                msgspec.msgpack.encode(
                    dict(
                        type="prefill.failed",
                        request_id=request_id,
                        code=error.code,
                        content=str(error),
                        fatal=False,
                    )
                )
            )

    def _pending_persistence(self, request_id):
        return any(
            key != request_id
            and value.get("ready_emitted")
            and value.get("receipt", {}).get("type") != "prefill.persisted"
            and value.get("receipt", {}).get("persistence_state") != "failed"
            for key, value in self.requests.items()
        )

    async def _compute(self, request_id, parameters, references, first):
        child = None
        request = self.requests[request_id]
        startup = "startup_observer" in request

        def remaining():
            return self.owner.config.startup_seconds if startup else self.remaining()

        try:
            geometry = self.owner.model_geometry
            init = dict(
                geometry.model_init(self.owner.upstream.init),
                type="prefill",
                request_id=request_id,
                prompt=parameters["prompt"],
                seed=parameters["seed"],
                references=references,
                source_frame=geometry.pad_rgb(first[1][0].rgb),
                persist=True,
                record_model_inputs=True,
            )
            await self.receipt(dict(type="prefill.computing", request_id=request_id))
            for attempt in range(3):
                remaining()
                child_identity = dict(
                    model_session_id=uuid4().hex, generation_id=uuid4().hex
                )
                child = self.owner.model_binding.borrow(child_identity, kind="prefill")
                try:
                    async with child.open() as (ws, assignment):
                        from .model_inputs import ModelInputCapture

                        recording = getattr(self.owner, "recording", None)
                        capture = (
                            ModelInputCapture(
                                recording,
                                child_identity["model_session_id"],
                                child_identity["generation_id"],
                            )
                            if recording is not None
                            else None
                        )
                        init["record_model_inputs"] = capture is not None
                        if capture is not None:
                            capture.sent(init)
                        try:
                            await ws.send(msgspec.msgpack.encode(init))
                            # One accepted-input receipt may precede the result.
                            # Use a single deadline; receipts cannot extend compute time.
                            deadline = time.monotonic() + remaining()
                            while True:
                                timeout = deadline - time.monotonic()
                                if timeout <= 0:
                                    raise asyncio.TimeoutError()
                                message = msgspec.msgpack.decode(
                                    await asyncio.wait_for(ws.recv(), timeout)
                                )
                                if capture is not None:
                                    capture.received(message)
                                if message.get("type") != "model.input":
                                    break
                        finally:
                            if capture is not None:
                                capture.close()
                        if (
                            message.get("type") != "prefill.memory_ready"
                            or message.get("request_id") != request_id
                        ):
                            raise RenderError("CONDITION_INVALID")
                        timeout = remaining()
                        await asyncio.wait_for(child.completed.wait(), timeout)
                        if child.resource_state != "released":
                            raise RenderError("RELEASE_UNKNOWN")
                    if not startup:
                        self.resolve(request["context_id"])
                    await self.receipt(message)
                    return
                except RenderError as error:
                    # Native async writeback can still hold CPU fill buffers
                    # after memory_ready. Retry only a failed, verified-clean
                    # child while earlier writes are pending, never uncertain
                    # cleanup or an input/protocol rejection. No model output
                    # was emitted; the original request and pinned F0 stay fixed.
                    if (
                        startup
                        or attempt == 2
                        # Internal worker failures are sanitized to RENDER_FAILED.
                        or error.code != "RENDER_FAILED"
                        or child.resource_state != "released"
                        or getattr(child, "failure", None) != "MODEL_OPERATION_FAILED"
                        or not self._pending_persistence(request_id)
                        or self.closed
                    ):
                        raise
                    await self.receipt(
                        dict(type="prefill.retrying", request_id=request_id)
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))
        except asyncio.CancelledError:
            if child is not None and child.resource_state not in {"none", "released"}:
                self.failure = RenderError("RELEASE_UNKNOWN")
            await self.receipt(
                dict(
                    type="prefill.failed",
                    request_id=request_id,
                    code="PREFILL_CANCELLED",
                )
            )
            raise
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "Control prefill failed: %s", request_id
            )
            if child is not None and child.resource_state not in {"none", "released"}:
                self.failure = RenderError("RELEASE_UNKNOWN")
            await self.receipt(
                dict(
                    type="prefill.failed", request_id=request_id, code="PREFILL_FAILED"
                )
            )
            if not startup:
                self.first = None
            if self.failure is None and not self.closed and not startup:
                try:
                    await self.resume(self.context_id)
                except Exception:
                    self.failure = RenderError("RELEASE_UNKNOWN")

    async def cancel(self, command):
        await self.available()
        if command.get("context_id") != self.context_id:
            raise RenderError("PREFILL_INPUT_MISMATCH")
        operation = await self.owner.journal.command(
            command, lambda s, o: o.update(state="accepted")
        )
        await self.owner.operation_message(operation)

        async def stop():
            if self.active is not None and not self.active.done():
                self.active.cancel()
                done, _ = await asyncio.wait((self.active,), timeout=90)
                if not done:
                    self.failure = RenderError("RELEASE_UNKNOWN")
            if self.failure is not None:
                raise self.failure
            if self.pending_frame is not None and not self.pending_frame.done():
                self.pending_frame.cancel()
            await self.resume(self.context_id)
            self.first = None
            await self.owner.operation_message(
                await self.owner.journal.result(
                    command["request_id"], state="completed"
                )
            )

        self.owner.spawn_control(stop())

    async def close(self):
        self.closed = True
        if self.active is not None and not self.active.done():
            self.active.cancel()
            await asyncio.gather(self.active, return_exceptions=True)
        if self.pending_frame is not None and not self.pending_frame.done():
            self.pending_frame.cancel()
        try:
            if not self.resumed:
                await self.resume(self.context_id)
        except Exception:
            # Engine disposal immediately follows; it owns final CLI teardown.
            pass
        finally:
            self.first = None
