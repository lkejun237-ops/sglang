# SPDX-License-Identifier: Apache-2.0
"""Authoritative CLI world host; optional Room pipelines own H3 GPU leases.

The server-configured driver owns one world and creates per-player views. Its
join() returns an EngineSession with a renderer attribute. No player can select
a driver, process command, URL or UI bundle through the public gateway.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import msgspec
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from .area_contract import area_source_options
from .contracts import VideoSpec
from .diagnostics import log_player_failure
from .maintenance_process import finish_cleanup
from .metrics import capture_stages, stage
from .model_geometry import validate_room_spec
from .player_idle import PlayerIdlePolicy, PlayerInputActivity, meaningful_input
from .recording import ArchiveManager, json_line, room_event_max_bytes, write_manifest
from .room_startup import (
    FactoryCleanupReceipt,
    MemberCleanupReceipt,
    RoomCleanupReceipt,
    drain_startup_cleanup,
    startup_cleanup_error,
    startup_cleanup_receipt,
)
from .rooms import checked_id, new_id, require_internal_token


class RoomLog:
    def __init__(
        self,
        spool,
        room,
        prefix,
        *,
        process=False,
        cpu=None,
        fail_open=False,
        diskless=False,
    ):
        self.directory = Path(spool) / new_id()
        self.diskless = diskless
        self.buffer = bytearray()
        self.archive_payload = None
        self.failure = None
        self.journal = None
        self.finishing = None
        if not process and not diskless:
            self.directory.mkdir(parents=True)
        self.trace_id, self.trace_snapshot = room["game_id"], None
        self.events = (
            None if process or diskless else (self.directory / "inputs.jsonl").open("w")
        )
        self.start, self.sequence = time.monotonic_ns(), 0
        self.manifest = dict(
            schema_version=2,
            archive_kind="game",
            game_id=room["game_id"],
            world_id=room["world_id"],
            object_prefix=prefix + "/",
            trace_id=self.trace_id,
            recording_status="finalizing",
            upload_status="pending",
            players=[],
            inputs={"path": "inputs.jsonl", "format": "ordered_room_events_v1"},
            clock={"unit": "us", "origin": "room_monotonic", "ordering": "room_seq"},
        )
        if diskless:
            self.manifest.update(
                recording_id=self.directory.name,
                archive_storage="oss_streaming_v1",
                dataset_status="excluded_unvalidated",
            )
        elif process:
            if fail_open:
                from .async_journal import AsyncRoomJournal as RoomJournal
            else:
                from .room_log_process import RoomJournal

            self.journal = RoomJournal(self.directory, self.trace_id, cpu)
        else:
            write_manifest(self.directory, self.manifest)

    async def initialize(self):
        if self.journal:
            await self.journal.start(self.manifest)

    async def event(self, player, kind, payload):
        self.sequence += 1
        event = dict(
            room_seq=self.sequence,
            room_time_us=(time.monotonic_ns() - self.start) // 1000,
            game_id=self.manifest["game_id"],
            **player,
            kind=kind,
            payload=payload,
        )
        if self.diskless:
            if not self.failure:
                try:
                    data = json_line(event).encode()
                    if (
                        len(data) > room_event_max_bytes(event)
                        or len(self.buffer) + len(data) > 8 << 20
                    ):
                        raise ValueError("Game event archive exceeds its memory budget")
                    self.buffer.extend(data)
                except (ValueError, TypeError, RecursionError, MemoryError):
                    self.failure = "GAME_ARCHIVE_MEMORY_BUDGET_EXCEEDED"
                    self.buffer.clear()
            # Receipt sequence remains authoritative even if archiving fails.
        elif self.journal:
            await self.journal.append(event)
        else:
            self.events.write(json_line(event))
            self.events.flush()
        return event

    async def finish(self, outcome):
        # Archive publication and startup rollback share one local flush/reap.
        # Waiting for this task never waits for the optional OSS publisher.
        if self.finishing is None:
            self.finishing = asyncio.create_task(self._finish(outcome))
        return await asyncio.shield(self.finishing)

    async def _finish(self, outcome):
        if self.diskless:
            self.archive_payload = bytes(self.buffer)
            self.buffer.clear()
            self.manifest.update(
                recording_status=(
                    "failed"
                    if self.failure
                    else ("complete" if outcome == "completed" else "partial")
                ),
                execution_outcome=outcome,
                archive_failure=self.failure,
                objects=(
                    {}
                    if self.failure
                    else {
                        "inputs.jsonl": {
                            "bytes": len(self.archive_payload),
                            "sha256": hashlib.sha256(self.archive_payload).hexdigest(),
                        }
                    }
                ),
            )
            return
        if self.journal:
            self.manifest = await self.journal.finish(self.manifest, outcome)
            return
        self.events.close()
        path = self.directory / "inputs.jsonl"
        digest = await asyncio.to_thread(
            lambda: hashlib.sha256(path.read_bytes()).hexdigest()
        )
        self.manifest.update(
            recording_status="complete" if outcome == "completed" else "partial",
            execution_outcome=outcome,
            objects={"inputs.jsonl": {"bytes": path.stat().st_size, "sha256": digest}},
        )
        write_manifest(self.directory, self.manifest)


def game_prefix(prefix, world_id, game_id):
    # World references may contain slashes/unicode. Hash the canonical reference,
    # retain the original in manifests, and never interpolate it into a local path.
    world_key = hashlib.sha256(world_id.encode()).hexdigest()[:32]
    return f"{prefix.rstrip('/')}/{world_key}/{checked_id(game_id)}"


class RoomRuntime:
    def __init__(
        self,
        *,
        directory,
        owner_id,
        driver_factory,
        spool,
        archives=None,
        prefix="white_box_game_record",
        spec=VideoSpec(),
        max_players=8,
        max_rooms=16,
        idle_seconds=45,
        pending_frames=96,
    ):
        self.directory, self.owner_id, self.driver_factory = (
            directory,
            owner_id,
            driver_factory,
        )
        self.epoch = new_id()
        self.spool, self.prefix, self.spec = Path(spool), prefix, spec
        self.archives = archives or ArchiveManager()
        self.max_players, self.max_rooms, self.idle_seconds = (
            max_players,
            max_rooms,
            idle_seconds,
        )
        self.pending_frames = pending_frames
        self.input_idle_policy = PlayerIdlePolicy(
            self.spool / ".runtime/player-idle.json"
        )
        if (
            min(max_players, max_rooms, idle_seconds, pending_frames) <= 0
            or spec.fps != 24
            or min(spec.width, spec.height) <= 0
            or (spec != VideoSpec(1280, 720) and (spec.width % 32 or spec.height % 32))
        ):
            raise ValueError("Invalid room capacity or video settings")
        self.rooms, self.players, self.locks = {}, {}, weakref.WeakValueDictionary()
        self.pending_startups, self.cancelled_startups = {}, {}
        self.join_attempts = {}
        self.retiring, self.retiring_players = {}, {}
        self.member_retirements, self.player_closures = {}, {}

    async def join(self, body, *, startup_scope=None):
        if startup_scope is not None:
            startup_scope.entered = True
        if body.get("startup_prepare") is not True:
            return await self._join(body, startup_scope=startup_scope)
        stream = checked_id(body["stream_id"])
        expires = body.get("startup_expires_at")
        now = time.time()
        self.cancelled_startups = {
            key: expiry
            for key, expiry in self.cancelled_startups.items()
            if expiry > now
        }
        if (
            type(expires) not in (int, float)
            or not math.isfinite(expires)
            or not now < expires <= now + 61
            or stream in self.cancelled_startups
            or stream in self.pending_startups
            or stream in self.players
        ):
            raise ValueError("The startup reservation is no longer valid")
        task = asyncio.create_task(self._join(body, startup_scope=startup_scope))
        self.pending_startups[stream] = task
        try:
            async with asyncio.timeout(max(0, expires - time.time())):
                return await task
        finally:
            self.pending_startups.pop(stream, None)

    async def cancel_startup(self, stream_id):
        stream_id = checked_id(stream_id)
        now = time.time()
        self.cancelled_startups = {
            key: expiry
            for key, expiry in self.cancelled_startups.items()
            if expiry > now
        }
        if (
            stream_id not in self.cancelled_startups
            and len(self.cancelled_startups) >= 1024
        ):
            raise ValueError("Startup cancellation capacity is temporarily full")
        self.cancelled_startups[stream_id] = now + 120
        task = self.pending_startups.get(stream_id)
        if task is not None:
            task.cancel()
            results = await asyncio.gather(task, return_exceptions=True)
            if any(isinstance(result, Exception) for result in results):
                raise ValueError("Startup cleanup could not be confirmed")
        await self.leave(stream_id)

    async def _start_room(
        self,
        game_id,
        route,
        lock,
        requested_fps,
        source_options,
        trace_id,
        *,
        create_context=None,
        verified_caller=None,
        expected_publication=None,
        spec=None,
        startup_scope=None,
    ):
        """Start through the same CLI/log/heartbeat owner while holding its lock."""
        spec = self.spec if spec is None else spec
        validate_room_spec(spec, self.spec)
        owner_epoch = self.epoch
        if len(self.rooms) >= self.max_rooms:
            from .diagnostics import WhiteboxDiagnostic

            raise WhiteboxDiagnostic("CAPACITY_EXHAUSTED", "admission")
        await self.directory.touch(game_id, self.owner_id, self.epoch)
        if startup_scope is not None:
            startup_scope.allocation_started = True
        log = RoomLog(
            self.spool,
            route,
            game_prefix(self.prefix, route["world_id"], game_id),
            # Live input and receipt correlation share this ordered log. Archive
            # backpressure must fail recording, never consume the SDK send budget.
            process=True,
            cpu=self.archives.recorder_cpu,
            fail_open=True,
            diskless=getattr(self.archives, "diskless", False),
        )
        await log.initialize()
        room = dict(
            spec=spec,
            cli_fps=requested_fps,
            **(
                {"expected_publication": expected_publication}
                if expected_publication is not None
                else {}
            ),
            **({"source_options": source_options} if source_options else {}),
            route=route,
            log=log,
            driver=None,
            players=set(),
            input_lock=asyncio.Lock(),
            lock=lock,
        )
        self.rooms[game_id] = room
        room["heartbeat"] = asyncio.create_task(self._heartbeat(game_id))
        try:
            with stage("cli_room_start", trace_id, service="room_runtime"):
                driver = await self.driver_factory(
                    world_id=route["world_id"],
                    **(
                        {"expected_publication": expected_publication}
                        if expected_publication is not None
                        else {}
                    ),
                    **(
                        {
                            "create_context": create_context,
                            "verified_caller": verified_caller,
                        }
                        if create_context is not None
                        else {}
                    ),
                    spec=spec,
                    trace_id=trace_id,
                    **({"source_options": source_options} if source_options else {}),
                    **({"cli_fps": requested_fps} if requested_fps is not None else {}),
                )
                if self.rooms.get(game_id) is not room:
                    await driver.close()
                    raise ValueError("The game ended during startup")
                if expected_publication is not None:
                    from .cli_room import validate_frozen_descriptor

                    try:
                        validate_frozen_descriptor(
                            getattr(driver, "descriptor", None), expected_publication
                        )
                    except BaseException:
                        await driver.close()
                        raise
                room["driver"] = driver
        except BaseException as error:
            if (self.rooms.get(game_id) or self.retiring.get(game_id)) is not room:
                # Another lifecycle owner withdrew this Room during startup.
                # Do not clean a replacement or mint a receipt from absence.
                raise
            factory = startup_cleanup_receipt(error)
            proven = (
                isinstance(factory, FactoryCleanupReceipt)
                and factory.operation_id == trace_id
            )
            complete, interrupted = await drain_startup_cleanup(
                self._end(game_id, "failed", wait_for_log=proven, expected_room=room)
            )
            if proven and complete:
                receipt = RoomCleanupReceipt(factory, owner_epoch)
                raise startup_cleanup_error(
                    asyncio.CancelledError() if interrupted else error, receipt
                ) from error
            if interrupted:
                if proven:
                    raise startup_cleanup_error(
                        asyncio.CancelledError(), factory
                    ) from error
                raise asyncio.CancelledError from error
            raise
        return room

    async def _join(self, body, *, startup_scope=None):
        stream_id = checked_id(body["stream_id"])
        if stream_id in self.join_attempts:
            raise ValueError(
                "The original stream join is still in progress or unconfirmed"
            )
        attempt = {"sdk_started": False}
        self.join_attempts[stream_id] = attempt
        try:
            return await self._join_owned(body, attempt, startup_scope=startup_scope)
        except BaseException as error:
            factory = startup_cleanup_receipt(error)
            factory_retired = (
                isinstance(factory, RoomCleanupReceipt)
                and factory.owner_epoch == self.epoch
                and factory.factory.operation_id == body.get("trace_id")
            )
            if (
                (not attempt["sdk_started"] or factory_retired)
                and stream_id not in self.players
                and stream_id not in self.retiring_players
                and stream_id not in self.member_retirements
            ):
                # No member was requested, or the original factory and Room
                # supplied their scoped cleanup receipt before driver return.
                # This also covers a queued peer whose Room ended while it
                # waited; absence after dispatch is never sufficient.
                self._remember_member_retirement(stream_id, state="not_created")
            if attempt["sdk_started"] and stream_id not in self.member_retirements:
                # A retry cannot relabel an earlier dispatched, unconfirmed
                # join as never-created after its route or owner disappears.
                attempt["failed"] = True
            raise
        finally:
            if self.join_attempts.get(stream_id) is attempt and not attempt.get(
                "failed"
            ):
                del self.join_attempts[stream_id]

    async def _join_owned(self, body, attempt, *, startup_scope=None):
        from .diagnostics import cli_fps_override

        requested_fps = cli_fps_override(body.get("cli_fps"))
        if type(body.get("startup_prepare", False)) is not bool:
            raise ValueError("Invalid startup reservation")
        source_options = area_source_options(body.get("source_options") or {})
        game_id, player_id, stream_id = (
            checked_id(body[k]) for k in ("game_id", "player_id", "stream_id")
        )
        checked_id(body["session_id"])
        spec = VideoSpec(**body["spec"])
        validate_room_spec(spec, self.spec)
        route = await self.directory.resolve(game_id, body["world_id"])
        if route["owner_id"] != self.owner_id:
            raise ValueError("The game belongs to another runtime")
        # Serialize joins only for this game. Other games can start concurrently.
        lock = self.locks.setdefault(game_id, asyncio.Lock())
        async with lock:
            route = await self.directory.resolve(game_id, body["world_id"])
            if getattr(self, "maintenance", None) is not None:
                await self.maintenance.cancel_for_play()
            if stream_id in self.players:
                raise ValueError("The stream is already connected")
            if (
                stream_id in self.retiring_players
                or stream_id in self.member_retirements
            ):
                raise ValueError("The original stream is retiring or retired")
            room = self.rooms.get(game_id)
            if room is not None and room.get("closing"):
                raise ValueError("The game is closing")
            if room is not None and spec != room["spec"]:
                from .diagnostics import WhiteboxDiagnostic

                raise WhiteboxDiagnostic("RESOLUTION_UNSUPPORTED", "validation")
            if (
                room is not None
                and requested_fps is not None
                and requested_fps != room.get("cli_fps")
            ):
                from .diagnostics import WhiteboxDiagnostic

                raise WhiteboxDiagnostic("CLI_FPS_UNSUPPORTED", "validation")
            if room is None:
                attempt["sdk_started"] = True
                room = await self._start_room(
                    game_id,
                    route,
                    lock,
                    requested_fps,
                    source_options,
                    body["trace_id"],
                    expected_publication=body.get("expected_publication"),
                    spec=spec,
                    startup_scope=startup_scope,
                )
            if room.get("control_caller") != body.get("control_caller"):
                raise ValueError("Controlled room admission is required")
            if room.get("expected_publication") != body.get("expected_publication"):
                raise ValueError("The frozen publication identity has changed")
            if room.get("source_options") != source_options:
                raise ValueError("AREA_SOURCE_OPTIONS_CHANGED")
            if len(room["players"]) >= min(
                self.max_players, room["driver"].max_players
            ):
                raise ValueError(
                    "This world does not have another player view available"
                )
            try:
                if startup_scope is not None:
                    startup_scope.allocation_started = True
                attempt["sdk_started"] = True
                engine = await room["driver"].join(
                    player_id=player_id, spec=room["spec"]
                )
                if self.rooms.get(game_id) is not room:
                    await engine.close()
                    await engine.renderer.close()
                    raise ValueError("The game ended while joining")
            except BaseException:
                await finish_cleanup(
                    self._retire_failed_join(room, game_id, player_id, stream_id)
                )
                raise
            if body.get("start_suspended") is True:
                channel = getattr(engine, "model_context", None)
                if channel is not None:
                    channel.suspended = True
            identity = {
                key: body[key]
                for key in ("player_id", "stream_id", "session_id", "user_id")
            }
            prefix = room["log"].manifest["object_prefix"] + "streams/" + stream_id
            entry = dict(
                spec=room["spec"],
                identity=identity,
                engine=engine,
                iterator=engine.frames().__aiter__(),
                game_id=game_id,
                pending={},
                next=0,
                touched=time.monotonic(),
                input_activity=PlayerInputActivity(),
                read_lock=asyncio.Lock(),
                render_lock=asyncio.Lock(),
                trace_id=body["trace_id"],
            )
            if (
                "control_caller" not in room
                and room["driver"].max_players == 1
                and getattr(engine, "metadata", {}).get("project_input") is True
            ):
                entry["source_input"] = True
            self.players[stream_id] = entry
            room["players"].add(stream_id)
            room["ever_occupied"] = True
            room.pop("empty_since", None)
            room["log"].manifest["players"].append(
                {**identity, "object_prefix": prefix + "/"}
            )
            joined = await room["log"].event(identity, "join", {})
            receipt = dict(
                game_id=game_id,
                **identity,
                owner_id=self.owner_id,
                owner_epoch=self.epoch,
                joined_room_time_us=joined["room_time_us"],
                object_prefix=prefix,
                game_object_prefix=room["log"].manifest["object_prefix"],
                engine=getattr(engine, "metadata", {}),
                ui=engine.renderer.metadata,
            )
            entry["join_receipt"] = receipt
            return receipt

    def player(self, stream_id, *, touch=True):
        entry = self.players.get(checked_id(stream_id))
        if entry is None or entry["game_id"] not in self.rooms:
            raise ValueError("The player has left the game")
        if touch:
            entry["touched"] = time.monotonic()
        return entry

    def input_idle_expired(self, entry):
        activity = entry.get("input_activity")
        if activity is None:
            return False
        now = time.monotonic()
        previously_expired = activity.expired
        expired = activity.is_expired(now, self.input_idle_policy.timeout(now))
        if expired and not previously_expired:
            logging.getLogger(__name__).info(
                "PLAYER_IDLE_TIMEOUT stream=%s trace=%s idle_s=%.3f timeout_s=%.3f",
                entry["identity"]["stream_id"],
                entry["trace_id"],
                now - activity.last_input,
                self.input_idle_policy.seconds,
            )
        return expired

    async def next_frame(self, stream_id, *, consumer=None):
        entry = self.player(stream_id, touch=False)
        spec = entry["spec"]
        if not entry.get("explicit_source_owner"):
            entry["touched"] = time.monotonic()
        async with entry["read_lock"]:
            if (
                entry.get("media_reader") is not None
                and entry["media_reader"] is not consumer
            ):
                raise ValueError("The original member source is already consumed")
            if len(entry["pending"]) >= self.pending_frames:
                raise ValueError("The player's video queue is full")
            try:
                with stage("cli_frame_wait", entry["trace_id"], service="room_runtime"):
                    frame = await anext(entry["iterator"])
            except StopAsyncIteration:
                return None
            if (
                frame.index != entry["next"]
                or len(frame.rgb) != spec.frame_bytes
                or frame.pts_us != spec.pts(frame.index)
            ):
                raise ValueError("The world returned an invalid video frame")
            if self.players.get(stream_id) is not entry:
                raise ValueError("The player has left the game")
            activity = entry["input_activity"]
            now = time.monotonic()
            if activity.startup_expired(now):
                raise ValueError(
                    "The first source frame arrived after the startup deadline"
                )
            entry["next"] += 1
            activity.start(now)
            if entry["engine"].renderer.metadata.get("mode") != "browser":
                entry["pending"][frame.index] = frame
            room = self.rooms[entry["game_id"]]
            result = asdict(frame)
            result["provenance"] = {
                **(frame.provenance or {}),
                "game_id": entry["game_id"],
                "stream_id": stream_id,
                "room_seq_observed": room["log"].sequence,
                "room_received_time_us": (time.monotonic_ns() - room["log"].start)
                // 1000,
            }
            return result

    async def capture_control(self, stream_id, action, request_id):
        engine = self.player(stream_id)["engine"]
        callback = getattr(engine, "capture_control", None)
        if callback is None:
            raise ValueError("Capture preparation is unavailable")
        return await callback(action, request_id)

    async def scene_entry_control(self, stream_id, command):
        entry = self.player(stream_id)
        room = self.rooms[entry["game_id"]]
        callback = getattr(entry["engine"], "scene_entry_control", None)
        if (
            callback is None
            or getattr(room["driver"], "max_players", 1) > 1
            or "control_caller" in room
        ):
            raise ValueError("SCENE_ENTRY_SOURCE_UNAVAILABLE")
        action, operation = command.get("action"), command.get("preparationId")
        activity = entry["input_activity"]
        if action == "hold":
            if activity.waiting not in (None, operation) or self.input_idle_expired(
                entry
            ):
                raise ValueError("SCENE_ENTRY_SOURCE_UNAVAILABLE")
            activity.pause(operation)
        try:
            result = await callback(command)
        except Exception:
            if action == "hold":
                activity.resume(operation, time.monotonic())
            raise
        if self.players.get(stream_id) is not entry:
            raise ValueError("SCENE_ENTRY_SOURCE_CHANGED")
        if action == "activate":
            activity.resume(operation, time.monotonic())
        return result

    async def area_status(self, stream_id):
        entry = self.player(stream_id)
        reader = getattr(entry["engine"], "area_status", None)
        if not reader:
            raise ValueError("AREA_PRESENTATION_UNSUPPORTED")
        result = await reader()
        if self.players.get(stream_id) is not entry:
            raise ValueError("AREA_SOURCE_CHANGED")
        return result

    async def area_frame(self, stream_id, header):
        entry = self.player(stream_id)
        reader = getattr(entry["engine"], "area_frame", None)
        if not reader:
            raise ValueError("AREA_PRESENTATION_UNSUPPORTED")
        result = await reader(header)
        if self.players.get(stream_id) is not entry:
            raise ValueError("AREA_SOURCE_CHANGED")
        return result

    async def area_presented(self, stream_id, body, *, before_send=None):
        entry = self.player(stream_id)
        room = self.rooms[entry["game_id"]]
        async with room["input_lock"]:
            if self.players.get(stream_id) is not entry:
                raise ValueError("AREA_SOURCE_CHANGED")
            writer = getattr(entry["engine"], "area_presented", None)
            if not writer:
                raise ValueError("AREA_PRESENTATION_UNSUPPORTED")
            # Receipt order uses the existing owner; native nonce stays in CLIView.
            await room["log"].event(entry["identity"], "area.presented", body)
            if self.players.get(stream_id) is not entry:
                raise ValueError("AREA_SOURCE_CHANGED")
            if before_send is None:
                return await writer(body)

            def check_dispatch():
                if self.players.get(stream_id) is not entry:
                    raise ValueError("AREA_SOURCE_CHANGED")
                before_send()

            return await writer(body, before_send=check_dispatch)

    async def event(self, stream_id, kind, payload):
        entry = self.player(stream_id)
        room = self.rooms[entry["game_id"]]
        async with room["input_lock"]:
            # This is receipt order, not an assertion of model/engine application.
            event = await room["log"].event(entry["identity"], kind, payload)
            if kind == "input":
                from .cli_room import cli_input
                from .render_contract import RenderError

                activity = entry["input_activity"]
                if activity.expired:
                    raise RenderError("PLAYER_IDLE_TIMEOUT")
                received = time.monotonic()
                value = cli_input(payload["sequence"], payload["input"])
                await entry["engine"].send_input(payload["sequence"], payload["input"])
                if (
                    meaningful_input(value)
                    or payload["input"].get("actions")
                    or activity.held.get("legacy")
                ):
                    activity.touch(received)
                activity.held["legacy"] = bool(value["heldKeys"])
            elif kind == "finish":
                await entry["engine"].finish()
            return event

    async def player_input_session(self, stream_id):
        entry = self.player(stream_id, touch=False)
        if not entry.get("source_input"):
            raise ValueError("The player input session is unavailable")
        return await self.player_transport.public_session({"stream_id": stream_id})

    async def ui_asset(self, stream_id, path):
        return await self.player(stream_id)["engine"].renderer.asset(path)

    async def render(self, stream_id, index, rgb):
        entry = self.player(stream_id)
        spec = entry["spec"]
        async with entry["render_lock"]:
            frame = entry["pending"].get(index)
            if frame is None or len(rgb) != spec.frame_bytes:
                raise ValueError("The model frame has no matching source frame")
            with stage("room_ui_render", entry["trace_id"], service="room_runtime"):
                output = await entry["engine"].renderer.render(rgb, frame, spec)
            if self.players.get(stream_id) is not entry:
                raise ValueError("The player has left the game")
            if len(output) != spec.frame_bytes:
                raise ValueError("The UI renderer returned an invalid frame")
            del entry["pending"][index]
            return output

    async def observe_player_control(self, stream_id, client_id, direction, wire):
        """Archive browser intent/SDK receipts without applying input a second time."""
        room = None
        try:
            message = json.loads(wire)
            if not isinstance(message, dict):
                return
            incoming = {
                "input.state",
                "input.action",
                "input.release",
                "input.semantic.open",
                "input.semantic.applied",
                "input.semantic.batch",
                "input.semantic.cancel",
            }
            receipts = {
                "input.receipt",
                "action.receipt",
                "input.released",
                "input.semantic.binding",
                "input.semantic.receipt",
                "input.semantic.error",
                "input.semantic.available",
            }
            if message.get("type") not in (
                incoming if direction == "to-sdk" else receipts
            ):
                return
            entry = self.player(stream_id, touch=False)
            descriptor = entry["engine"].descriptor
            if (
                type(message.get("epoch")) is not int
                or message["epoch"] != descriptor["epoch"]
            ):
                return
            activity = entry.get("input_activity")
            if activity is not None:
                activity.observe(client_id, direction, message, time.monotonic())
            room = self.rooms[entry["game_id"]]
            await room["log"].event(
                entry["identity"],
                (
                    "viewer_input_observed"
                    if direction == "to-sdk"
                    else "viewer_input_receipt"
                ),
                dict(
                    clientId=client_id,
                    sdkSessionId=descriptor["sessionId"],
                    sdkEpoch=descriptor["epoch"],
                    message=message,
                ),
            )
        except Exception:
            # Recording failure cannot take over live control. Its original
            # manifest carries an explicit failure, never a complete input claim.
            if room is not None:
                room["log"].manifest["viewer_input_archive_status"] = "failed"

    def _remember_member_retirement(self, stream_id, *, state="retired"):
        receipt = dict(
            state=state,
            source_id=stream_id,
            view_id=stream_id if state == "retired" else None,
            owner_epoch=self.epoch,
        )
        self.member_retirements[stream_id] = receipt
        if self.join_attempts.get(stream_id, {}).get("failed"):
            del self.join_attempts[stream_id]
        while len(self.member_retirements) > 2048:
            self.member_retirements.pop(next(iter(self.member_retirements)))
        return dict(receipt)

    async def _retire_failed_join(self, room, game_id, player_id, stream_id):
        # A failed join can precede players[stream_id]. Keep the original
        # driver's positive rollback (or whole empty-Room retirement) so the
        # already-reserved model parent can complete its verified release.
        if (self.rooms.get(game_id) or self.retiring.get(game_id)) is not room:
            return
        confirmed = False
        cancel_join = getattr(room["driver"], "cancel_join", None)
        if cancel_join:
            try:
                receipt = await asyncio.wait_for(cancel_join(player_id), 40)
                confirmed = (
                    isinstance(receipt, MemberCleanupReceipt)
                    and receipt.player_id == player_id
                )
            except BaseException:
                logging.getLogger(__name__).warning(
                    "Room join cleanup remains unconfirmed game_id=%s", game_id
                )
        if not room["players"]:
            try:
                confirmed = await self._end(game_id, "failed", expected_room=room)
            except BaseException:
                confirmed = False
                logging.getLogger(__name__).exception(
                    "Failed join Room retirement remains unconfirmed game_id=%s",
                    game_id,
                )
        elif self.rooms.get(game_id) is not room:
            confirmed = False
        if confirmed:
            self._remember_member_retirement(stream_id)

    async def retire_member(self, stream_id):
        """Retire only the SDK view/input owner, independent of GPU cleanup."""
        previous = self.member_retirements.get(stream_id)
        if previous is not None:
            if previous["owner_epoch"] != self.epoch:
                raise ValueError("SDK_MEMBER_CLEANUP_UNKNOWN")
            return dict(previous)
        entry = self.players.pop(stream_id, None) or self.retiring_players.get(
            stream_id
        )
        if entry is None:
            raise ValueError("SDK_MEMBER_CLEANUP_UNKNOWN")
        self.retiring_players[stream_id] = entry
        if "control_admission" in entry:
            entry["control_admission"]["leaving"] = True
        task = entry.get("sdk_close_task")
        if (
            task is not None
            and task.done()
            and (task.cancelled() or task.exception() is not None)
        ):
            task = None  # Retry the same fenced owner, never create a new view.
        if task is None:

            async def cleanup():
                # No GameRoom lock across SDK/GPU work. The old identity remains
                # fenced in retiring_players until a positive SDK receipt.
                results = await asyncio.gather(
                    asyncio.wait_for(entry["engine"].close(), 40),
                    asyncio.wait_for(entry["engine"].renderer.close(), 10),
                    return_exceptions=True,
                )
                failures = [
                    result for result in results if isinstance(result, BaseException)
                ]
                if failures:
                    raise failures[0]
                game_id = entry["game_id"]
                lock = self.locks.setdefault(game_id, asyncio.Lock())
                close_world = False
                room = self.rooms.get(game_id)
                if room:
                    # One event-loop mutation, no wait on a peer's slow SDK join.
                    room["players"].discard(stream_id)
                if not lock.locked():
                    async with lock:
                        room = self.rooms.get(game_id)
                        if (
                            room
                            and not room["players"]
                            and "control_caller" not in room
                        ):
                            room["closing"] = close_world = True
                if close_world and not await self._end(
                    game_id, "completed", expected_room=room
                ):
                    raise ValueError("SDK_WORLD_CLEANUP_UNKNOWN")
                receipt = self._remember_member_retirement(stream_id)
                if room:
                    try:
                        await room["log"].event(entry["identity"], "leave", {})
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "Player leave archive failed"
                        )
                return receipt

            task = entry["sdk_close_task"] = asyncio.create_task(cleanup())
        return dict(await asyncio.shield(task))

    async def leave(self, stream_id):
        task = self.player_closures.get(stream_id)
        if (
            task is not None
            and task.done()
            and (task.cancelled() or task.exception() is not None)
        ):
            task = None
        if task is None:
            entry = self.players.get(stream_id) or self.retiring_players.get(stream_id)
            if entry is None:
                receipt = self.member_retirements.get(stream_id)
                return receipt is not None and receipt["owner_epoch"] == self.epoch
            record = entry.get("control_admission")
            if record is not None:
                record["leaving"] = True

            async def cleanup():
                owner = getattr(self, "member_media", None)
                operations = [self.retire_member(stream_id)]
                if record is not None and owner is not None:
                    operations.append(owner.close(record))
                results = await asyncio.gather(*operations, return_exceptions=True)
                failures = [
                    result for result in results if isinstance(result, BaseException)
                ]
                if failures:
                    raise failures[0]
                self.retiring_players.pop(stream_id, None)
                return True

            task = self.player_closures[stream_id] = asyncio.create_task(cleanup())
        return await asyncio.shield(task)

    async def _heartbeat(self, game_id):
        try:
            while True:
                await asyncio.sleep(1)
                await self.directory.touch(game_id, self.owner_id, self.epoch)
                room = self.rooms[game_id]
                for stream_id in list(room["players"]):
                    # leave() withdraws a player before asynchronous cleanup.
                    # A heartbeat during that interval is not owner loss.
                    entry = self.players.get(stream_id)
                    if entry is not None and (
                        entry["input_activity"].startup_expired(time.monotonic())
                        or (
                            not entry.get("explicit_source_owner")
                            and self.input_idle_expired(entry)
                        )
                        or time.monotonic() - entry["touched"]
                        > entry.get("owner_idle_seconds", self.idle_seconds)
                        or (
                            entry.get("player_expires_at") is not None
                            and time.time() >= entry["player_expires_at"]
                        )
                    ):
                        # One failed player's GPU/SDK cleanup is not owner loss.
                        task = asyncio.create_task(self.leave(stream_id))
                        task.add_done_callback(self._observe_player_cleanup)
                end_reason = None
                if self.locks[game_id].locked():
                    # A cold CLI start/join can outlive an owner lease. Continue
                    # renewing independently; do not wait here behind that work.
                    continue
                async with self.locks[game_id]:
                    if self.rooms.get(game_id) is not room:
                        return
                    admissions = getattr(
                        getattr(self, "control", None), "admissions", {}
                    )
                    pending = any(
                        record.get("binding", {}).get("gameId") == game_id
                        and not record.get("leaving")
                        and record.get("task") is not None
                        and not record["task"].done()
                        for record in admissions.values()
                    )
                    record = room.get("control_record")
                    if (
                        record is not None
                        and time.time() * 1000 >= record["expires_at_ms"]
                    ):
                        # Fence new joins and wait for original admissions to settle
                        # before using the existing whole-room cleanup path.
                        record["retiring"] = room["closing"] = True
                        if not pending:
                            end_reason = "expired"
                    elif record is not None or room["players"] or pending:
                        # Fixed invitations can rejoin an empty managed room.
                        room.pop("empty_since", None)
                    else:
                        room.setdefault("empty_since", time.monotonic())
                        grace = 15 if room.get("ever_occupied") else 120
                        if time.monotonic() - room["empty_since"] >= grace:
                            room["closing"] = True
                            end_reason = "empty"
                if end_reason is not None:
                    await self._end(game_id, end_reason, expected_room=room)
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            # Never silently replace an authoritative world after owner/lease loss.
            await self._end(game_id, "owner_lost")

    @staticmethod
    def _observe_player_cleanup(task):
        if not task.cancelled() and task.exception() is not None:
            logging.getLogger(__name__).error(
                "Player cleanup quarantined; other members continue",
                exc_info=task.exception(),
            )

    async def _end(self, game_id, outcome, *, wait_for_log=False, expected_room=None):
        if (
            expected_room is not None
            and (self.rooms.get(game_id) or self.retiring.get(game_id))
            is not expected_room
        ):
            return False
        room = self.rooms.pop(game_id, None) or self.retiring.get(game_id)
        if room is None:
            return True
        self.retiring[game_id] = room

        def cleanup_failure(error, phase):
            log_player_failure(
                "room.cleanup",
                error,
                record={**(room.get("control_record") or {}), "game_id": game_id},
                action="retireRoom",
                owner_epoch=self.epoch,
                outcome="unknown",
                binding=None,
                phase=phase,
            )

        if room["heartbeat"] is not asyncio.current_task():
            room["heartbeat"].cancel()
            await asyncio.gather(room["heartbeat"], return_exceptions=True)
        for stream_id in list(room["players"]):
            entry = self.players.pop(stream_id, None)
            if entry:
                self.retiring_players[stream_id] = entry
        entries = {
            stream_id: entry
            for stream_id, entry in self.retiring_players.items()
            if entry["game_id"] == game_id
        }
        admissions = [
            entry["control_admission"]
            for entry in entries.values()
            if "control_admission" in entry
        ]
        media = getattr(self, "member_media", None)
        media_cleanup = None
        if media is not None:
            for admission in admissions:
                admission["leaving"] = True
            media_cleanup = asyncio.gather(
                *(
                    asyncio.wait_for(media.close(admission), 40)
                    for admission in admissions
                ),
                return_exceptions=True,
            )
        closing = [
            operation
            for entry in entries.values()
            for operation in (entry["engine"].close(), entry["engine"].renderer.close())
        ]
        if room["driver"]:
            closing.append(room["driver"].close())
        results = await asyncio.gather(
            *(asyncio.wait_for(item, 40) for item in closing), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                cleanup_failure(result, "sdk-close")
        if any(isinstance(result, BaseException) for result in results):
            outcome = "cleanup_failed"
        sdk_complete = not any(isinstance(result, BaseException) for result in results)
        if sdk_complete:
            try:
                await self.directory.touch(game_id, self.owner_id, self.epoch, end=True)
            except Exception as error:
                # The retired execution remains fenced by its original epoch.
                cleanup_failure(error, "directory-end")
        if media_cleanup is not None:
            media_results = await media_cleanup
            for result in media_results:
                if isinstance(result, BaseException):
                    cleanup_failure(result, "media-close")
            results.extend(media_results)
        if not room.get("log_finalized"):
            self.archives.finalize(room["log"], outcome)
            room["log_finalized"] = True
        complete = not any(isinstance(result, BaseException) for result in results)
        control_record = room.get("control_record")
        if wait_for_log or control_record is not None:
            try:
                await asyncio.wait_for(room["log"].finish(outcome), 40)
            except BaseException as error:
                cleanup_failure(error, "log-finish")
                complete = False
        if complete:
            if room.get("route", {}).get("dynamic_owner"):
                try:
                    await self.directory.release_room(
                        game_id, self.owner_id, self.epoch
                    )
                except Exception as error:
                    cleanup_failure(error, "capacity-release")
                    return False
            if control_record is not None:
                control_record["room_cleanup_complete"] = True
                try:
                    invocation = control_record["invocation"]
                    world = invocation["createContext"]["frozenPublication"]["world"]
                    binding = dict(
                        gameId=game_id,
                        ownerEpoch=self.epoch,
                        publicationId=world["publicationId"],
                    )
                    # Only certify the original members included in this cleanup.
                    for admission in admissions:
                        if admission.get("binding") == binding:
                            admission.update(leaving=True, status="retired")
                    await self.directory.confirm_room_cleanup(
                        control_record["caller"],
                        invocation,
                        binding,
                        self.owner_id,
                    )
                except Exception as error:
                    # Older coordinators may not support durable receipts yet.
                    cleanup_failure(error, "cleanup-receipt")
            if self.retiring.get(game_id) is room:
                self.retiring.pop(game_id)
            for stream_id, entry in entries.items():
                if self.retiring_players.get(stream_id) is entry:
                    self.retiring_players.pop(stream_id)
        return complete

    async def close(self):
        for task in list(self.pending_startups.values()):
            task.cancel()
        await asyncio.gather(*self.pending_startups.values(), return_exceptions=True)
        if getattr(self, "maintenance", None) is not None:
            await self.maintenance.cancel_for_play()
        if getattr(self, "control", None) is not None:
            await self.control.close()
        for game_id in set(self.rooms) | set(self.retiring):
            await self._end(game_id, "runtime_stopped")
        await self.archives.close()


def create_app(runtime, token):
    @asynccontextmanager
    async def lifespan(app):
        runtime.archives.start(runtime.spool)
        yield
        await runtime.close()

    app = FastAPI(title="Whitebox Game Runtime", lifespan=lifespan)
    from .room_registration import install_owner_registration

    install_owner_registration(app, runtime, token)
    from sglang.multimodal_gen.runtime.realtime.release_identity import (
        install_release_route,
    )

    install_release_route(app, "room")
    from .prefetch import install_prefetch

    prepare_world = install_prefetch(app, token)
    from .runtime_management import install_room_management

    install_room_management(app, token, runtime, prepare_world)
    from .room_control_runtime import install_runtime_room_control

    install_runtime_room_control(app, token, runtime)
    from .player_transport import install_player_runtime

    install_player_runtime(app, token, runtime)

    @app.get("/healthz")
    async def health():
        return {
            "status": "ok",
            "owner_id": runtime.owner_id,
            "owner_epoch": runtime.epoch,
        }

    @app.get("/metrics")
    async def metrics():
        from sglang.multimodal_gen.runtime.realtime.critical_path_metrics import (
            prometheus_content_type,
            prometheus_latest,
        )

        return Response(prometheus_latest(), media_type=prometheus_content_type())

    @app.post("/internal/players/join")
    async def join(
        body: dict, request: Request, authorization: str = Header(default="")
    ):
        require_internal_token(token, authorization)

        async def disconnected():
            # FastAPI has already consumed the JSON body. Read the remaining
            # ASGI disconnect directly; is_disconnected() uses a cancellation
            # scope which can swallow this watcher's own task cancellation.
            while (await request.receive())["type"] != "http.disconnect":
                pass

        joining = watcher = None
        try:
            with capture_stages() as spans:
                joining = asyncio.create_task(runtime.join(body))
                watcher = asyncio.create_task(disconnected())
                done, _ = await asyncio.wait(
                    (joining, watcher), return_when=asyncio.FIRST_COMPLETED
                )
                if watcher in done:
                    joining.cancel()
                    results = await asyncio.gather(joining, return_exceptions=True)
                    if isinstance(results[0], dict):
                        await runtime.leave(results[0]["stream_id"])
                    raise HTTPException(499, "The joining client disconnected")
                result = joining.result()
            return JSONResponse(
                result, headers={"X-Whitebox-Stages": json.dumps(spans)}
            )
        except (ValueError, KeyError, TypeError) as exc:
            logging.getLogger(__name__).exception("Room join failed")
            raise HTTPException(409, "The game could not be joined") from exc
        finally:
            pending = [task for task in (joining, watcher) if task is not None]
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    @app.post("/internal/players/{stream_id}/{operation}")
    async def operation(
        stream_id: str,
        operation: str,
        request: Request,
        authorization: str = Header(default=""),
    ):
        require_internal_token(token, authorization)
        try:
            # Bound even chunked requests; no unbounded request.body() allocation.
            data = bytearray()
            entry = runtime.players.get(stream_id)
            spec = entry["spec"] if entry is not None else runtime.spec
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > spec.frame_bytes + 65536:
                    raise ValueError("The room message is too large")
            body = msgspec.msgpack.decode(data) if data else {}
            with capture_stages() as spans:
                if operation == "frame":
                    result = await runtime.next_frame(stream_id)
                elif operation == "player_input_session":
                    result = await runtime.player_input_session(stream_id)
                elif operation == "ui_asset":
                    result = await runtime.ui_asset(stream_id, body["path"])
                elif operation == "render":
                    result = await runtime.render(stream_id, body["index"], body["rgb"])
                elif operation == "render_finish":
                    entry = runtime.player(stream_id)
                    async with entry["render_lock"]:
                        if entry["pending"]:
                            raise ValueError(
                                "The composition has pending source frames"
                            )
                        finish = getattr(entry["engine"].renderer, "finish", None)
                        if finish:
                            await finish()
                    result = None
                elif operation == "capture_control":
                    result = await runtime.capture_control(
                        stream_id, body.get("action"), body.get("request_id")
                    )
                elif operation == "scene_entry_control":
                    result = await runtime.scene_entry_control(stream_id, body)
                elif operation == "area_status":
                    result = await runtime.area_status(stream_id)
                elif operation == "startup_cancel":
                    await runtime.cancel_startup(stream_id)
                    result = {"cancelled": True}
                elif operation == "area_frame":
                    result = await runtime.area_frame(stream_id, body)
                elif operation == "area_presented":
                    result = await runtime.area_presented(stream_id, body)
                elif operation == "event":
                    if body["kind"] not in {
                        "input",
                        "prompt",
                        "references",
                        "finish",
                        "inference.init",
                    }:
                        raise ValueError("Unsupported room event")
                    result = await runtime.event(
                        stream_id, body["kind"], body["payload"]
                    )
                elif operation == "leave":
                    await runtime.leave(stream_id)
                    result = None
                elif operation == "touch":
                    # Liveness is not a game input and must not enter inputs.jsonl.
                    runtime.player(stream_id)
                    result = None
                else:
                    raise ValueError("Unsupported room operation")
            return Response(
                msgspec.msgpack.encode(result),
                media_type="application/msgpack",
                headers={"X-Whitebox-Stages": json.dumps(spans)},
            )
        except (ValueError, KeyError, TypeError, msgspec.DecodeError) as exc:
            logging.getLogger(__name__).exception(
                "Room operation failed operation=%s stream_id=%s", operation, stream_id
            )
            raise HTTPException(409, "The game stream is unavailable") from exc

    return app
