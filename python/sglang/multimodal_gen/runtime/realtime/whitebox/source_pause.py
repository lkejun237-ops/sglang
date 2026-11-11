# SPDX-License-Identifier: Apache-2.0
"""Exclusive source-clock pause, independent of model inference and reports."""

import asyncio
import hashlib
import time
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import UUID

from .render_contract import RenderError, canonical

MAXIMUM_PAUSE_MS = 300000


@dataclass
class Pause:
    id: str
    engine: object
    generation: int
    phase: str = "pending"
    cancelled: bool = False
    activity: object = None
    timer: object = None


class SourcePause:
    def __init__(self, owner):
        self.owner = owner
        self.current = None
        self.previous = None
        self.released = set()  # Bounded by the existing per-source command journal.
        self.generation = 0
        self.closed = False
        self.lock = asyncio.Lock()
        self.pause_admissions = {}
        self.prefill_admission = None

    @property
    def supported(self):
        engine = getattr(self.owner, "engine", None)
        metadata = getattr(engine, "metadata", {})
        pipeline = getattr(getattr(self.owner, "host", None), "sessions", {}).get(
            getattr(self.owner, "session_id", None)
        )
        return bool(
            engine is not None
            and callable(getattr(engine, "capture_control", None))
            and metadata.get("mode") == "worldkit_cli"
            and metadata.get("source_codec") == "rgb24"
            and type(metadata.get("max_players")) is int
            and metadata["max_players"] == 1
            and getattr(pipeline, "member_source", None) is None
            and not getattr(engine, "shared_world", False)
            and getattr(self.owner, "member_prefill", None) is None
        )

    @property
    def active(self):
        return self.current is not None or bool(self.pause_admissions)

    @contextmanager
    def reserve_prefill(self):
        # Synchronous reservation must precede Prefill.available/journal awaits.
        if self.active or self.prefill_admission is not None:
            raise RenderError("SOURCE_PAUSE_BUSY")
        token = self.prefill_admission = object()
        try:
            yield
        finally:
            if self.prefill_admission is token:
                self.prefill_admission = None

    def prefill_busy(self):
        prefill = getattr(self.owner, "control_prefill", None)
        return self.prefill_admission is not None or (
            prefill is not None
            and (
                not prefill.resumed
                or getattr(prefill, "pending_frame", None) is not None
                or any(
                    task is not None and not task.done()
                    for task in (
                        getattr(prefill, "active", None),
                        getattr(prefill, "recovery", None),
                    )
                )
                or getattr(prefill, "failure", None) is not None
            )
        )

    def snapshot(self):
        entry = self.current
        if entry is not None:
            return self.value(entry.id, "paused") if entry.phase == "paused" else None
        return self.previous

    @staticmethod
    def value(pause_id, state):
        return dict(pause_id=pause_id, state=state, maximum_pause_ms=MAXIMUM_PAUSE_MS)

    def activity(self, engine):
        runtime = getattr(getattr(self.owner, "host", None), "runtime", None)
        return (
            getattr(runtime, "players", {})
            .get(getattr(engine, "stream_id", None), {})
            .get("input_activity")
        )

    async def command(self, command):
        allowed = {
            "type",
            "version",
            "op",
            "request_id",
            "source_session_id",
            "source_epoch",
            "pause_id",
        }
        pause_id = command.get("pause_id")
        try:
            valid = isinstance(pause_id, str) and str(UUID(pause_id)) == pause_id
        except ValueError:
            valid = False
        if (
            not valid
            or set(command) - allowed
            or command.get("op") not in {"source.pause", "source.resume"}
            or command.get("type", "command") != "command"
            or (
                "version" in command
                and (type(command["version"]) is not int or command["version"] != 2)
            )
        ):
            raise RenderError("INVALID_COMMAND")
        if not self.supported:
            raise RenderError("CAPABILITY_UNAVAILABLE")
        # Durable duplicate replies remain readable while another foreground
        # owner is busy; they never enter capture admission again.
        previous = (
            (self.owner.journal.snapshot or {})
            .get("operations", {})
            .get(command.get("request_id"))
        )
        if previous is not None:
            if (
                previous["payload_sha256"]
                != hashlib.sha256(canonical(command)).hexdigest()
            ):
                raise RenderError("REQUEST_ID_CONFLICT")
            await self.owner.operation_message({**previous, "duplicate": True})
            await self.owner.status_snapshot()
            return
        generation, engine = self.generation, self.owner.engine
        token = None
        if command["op"] == "source.pause":
            if self.prefill_busy() or any(
                value != pause_id for value in self.pause_admissions.values()
            ):
                raise RenderError("SOURCE_PAUSE_BUSY")
            token = object()
            self.pause_admissions[token] = pause_id
        try:
            result = await self.owner.journal.command(
                command, lambda state, operation: operation.update(state="accepted")
            )
            await self.owner.operation_message(result)
            if result.get("duplicate"):
                await self.owner.status_snapshot()
                self.pause_admissions.pop(token, None)
                return
            # Cancel only this original pending hold, before awaiting its receipt.
            if (
                command["op"] == "source.resume"
                and self.current is not None
                and self.current.id == pause_id
            ):
                self.current.cancelled = True
            self.owner.spawn_control(self.apply(command, generation, engine, token))
        except BaseException:
            self.pause_admissions.pop(token, None)
            raise

    async def apply(self, command, generation, engine, token):
        request_id, pause_id = command["request_id"], command["pause_id"]
        try:
            async with self.lock:
                if command["op"] == "source.resume":
                    entry = self.current
                    if entry is not None and entry.id == pause_id:
                        await self.resume(entry)
                    # No-op for an unowned ID, including a failed busy pause.
                    self.released.add(pause_id)
                    value = self.value(pause_id, "running")
                else:
                    if (
                        self.closed
                        or generation != self.generation
                        or getattr(self.owner, "closing", False)
                        or self.owner.binding_ended
                        or self.owner.engine is not engine
                        or not self.supported
                    ):
                        raise RenderError("INVALID_SOURCE")
                    entry = self.current
                    if entry is not None:
                        if entry.id != pause_id:
                            raise RenderError("SOURCE_PAUSE_BUSY")
                        if entry.phase != "paused":
                            raise RenderError("SOURCE_PAUSE_UNCONFIRMED")
                        value = self.value(pause_id, "paused")
                    elif pause_id in self.released:
                        value = self.value(pause_id, "running")
                    else:
                        if self.prefill_busy():
                            raise RenderError("SOURCE_PAUSE_BUSY")
                        entry = Pause(pause_id, self.owner.engine, generation)
                        self.current = entry
                        started = time.monotonic()
                        try:
                            held = await entry.engine.capture_control("hold", pause_id)
                            if (
                                held.get("requestId") != pause_id
                                or held.get("state") != "held"
                            ):
                                raise RenderError("SOURCE_PAUSE_UNCONFIRMED")
                        except Exception as error:
                            entry.phase = "unknown"
                            # The HTTP reply can be lost after the source stopped.
                            # Reconcile only this exact original hold, never guess.
                            try:
                                await self.resume(entry)
                            except Exception:
                                pass
                            raise RenderError("SOURCE_PAUSE_UNCONFIRMED") from error
                        if (
                            entry.cancelled
                            or self.closed
                            or generation != self.generation
                            or self.owner.engine is not entry.engine
                            or getattr(self.owner, "closing", False)
                            or self.owner.binding_ended
                        ):
                            await self.resume(entry)
                            value = self.value(pause_id, "running")
                        else:
                            entry.phase = "paused"
                            entry.activity = self.activity(entry.engine)
                            if entry.activity is not None:
                                entry.activity.pause(pause_id)
                            # Start no later than the SDK's own safety timer.
                            delay = max(
                                0,
                                MAXIMUM_PAUSE_MS / 1000 - (time.monotonic() - started),
                            )
                            entry.timer = asyncio.get_running_loop().call_later(
                                delay, lambda: self.invalidate(entry)
                            )
                            value = self.value(pause_id, "paused")
                result = await self.owner.journal.result(
                    request_id, state="complete", source_pause=value
                )
        except Exception as error:
            code = (
                error.code
                if isinstance(error, RenderError)
                else "SOURCE_PAUSE_UNCONFIRMED"
            )
            result = await self.owner.journal.result(
                request_id, state="failed", code=code
            )
        finally:
            self.pause_admissions.pop(token, None)
        await self.owner.operation_message(result)
        await self.owner.status_snapshot()

    def unwait(self, entry):
        if entry.activity is not None:
            entry.activity.resume(entry.id, time.monotonic())
            entry.activity = None

    async def resume(self, entry):
        if self.current is not entry:
            return
        entry.phase = "resuming"
        try:
            result = await entry.engine.capture_control("resume", entry.id)
        except Exception:
            entry.phase = "unknown"
            self.unwait(entry)
            raise
        if result.get("requestId") != entry.id or result.get("state") != "running":
            entry.phase = "unknown"
            self.unwait(entry)
            raise RenderError("SOURCE_PAUSE_UNCONFIRMED")
        if entry.timer is not None:
            entry.timer.cancel()
        self.unwait(entry)
        self.released.add(entry.id)
        self.previous = self.value(entry.id, "running")
        self.current = None

    def invalidate(self, entry=None):
        if entry is not None and self.current is not entry:
            return
        self.generation += 1
        entry = self.current if entry is None else entry
        if entry is not None and self.current is entry:
            entry.cancelled = True
            # Mark no longer confirmed immediately; a late receipt cannot
            # resurrect pause after a detach/source close or safety deadline.
            entry.phase = "unknown"
            self.unwait(entry)
            self.owner.spawn_control(self.release(entry))

    async def release(self, entry):
        async with self.lock:
            if self.current is not entry:
                return
            try:
                await self.resume(entry)
            except Exception:
                # Keep ownership for an explicit same-ID retry. Never report
                # running on an unknown resume. Idle suppression is already
                # released, so a failed cleanup cannot extend inactivity forever.
                entry.phase = "unknown"
        await self.owner.status_snapshot()

    async def close(self):
        self.closed = True
        self.generation += 1
        entry = self.current
        if entry is not None:
            entry.cancelled = True
            async with self.lock:
                if self.current is entry:
                    await self.resume(entry)
