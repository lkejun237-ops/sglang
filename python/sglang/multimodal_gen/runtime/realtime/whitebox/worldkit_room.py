# SPDX-License-Identifier: Apache-2.0
"""Adapt one sealed SDK Room and its private player views to RoomRuntime.

Membership, actors and simulation remain owned by the SDK. This adapter owns
only CLI processes, private authentication slots and per-player media clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import signal
import tempfile
from pathlib import Path
from types import SimpleNamespace

from .cli_config import startup_seconds
from .cli_room import (
    NativeCLIView,
    cli_diagnostics,
    load_local_ui_package,
    prepare_delivery,
    validate_descriptor,
)
from .room_startup import (
    FactoryCleanupReceipt,
    MemberCleanupReceipt,
    drain_startup_cleanup,
    startup_cleanup_error,
)
from .rooms import checked_id, new_id
from .session import load_plugin

MAX_REPLY_BYTES = 1024 * 1024


def private_json(path, value):
    with open(
        path, "x", opener=lambda name, flags: os.open(name, flags, 0o600)
    ) as file:
        json.dump(value, file)


async def stop_process(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), 5)
    except asyncio.TimeoutError:
        pass
    finally:
        # The leader can exit before a TERM-resistant child in this private
        # process group. Reap the complete owned group, not only the leader.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def read_member_session(member, entry, *, purpose="viewer", public_origin=None):
    """Resolve a current private purpose; only viewer sessions may be published."""
    process = await asyncio.create_subprocess_exec(
        os.environ.get("WHITEBOX_NODE", "node"),
        "--import",
        "tsx",
        str(Path(__file__).with_name("worldkit_room_session.mjs")),
        cwd=entry.parents[3],
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
        limit=MAX_REPLY_BYTES,
    )
    try:
        output, errors = await asyncio.wait_for(
            process.communicate(
                json.dumps(
                    {
                        "memberFile": member["memberFile"],
                        "uiRoot": member["localResources"]["uiRoot"],
                        "purpose": purpose,
                        **(
                            {"publicOrigin": public_origin}
                            if public_origin is not None
                            else {}
                        ),
                    }
                ).encode()
            ),
            30,
        )
        if process.returncode or len(output) > MAX_REPLY_BYTES:
            if errors.strip() == b"ROOM_MODEL_SESSION_REQUIRED":
                raise ValueError("ROOM_MODEL_SESSION_REQUIRED")
            raise ValueError("ROOM_SESSION_UNAVAILABLE")
        return json.loads(output)
    finally:
        await stop_process(process)


class RoomMemberView(NativeCLIView):
    async def close(self):
        await super().close()
        # A replaced media view must never retire its successor's actor.
        await self.room.owner.leave(self.room.player_id, view=self)


class WorldkitRoom:
    def __init__(
        self,
        *,
        entry,
        world_id,
        spec,
        trace_id,
        slots,
        capture_fps,
        create_context=None,
        verified_caller=None,
    ):
        self.entry, self.world_id, self.spec, self.trace_id = (
            entry,
            world_id,
            spec,
            trace_id,
        )
        self.create_context, self.verified_caller = create_context, verified_caller
        self.external_input = create_context is not None
        self.capture_fps = capture_fps
        self.timeout = startup_seconds(os.environ)
        self.renderer_factory = load_plugin(
            os.environ.get(
                "WHITEBOX_CLI_UI_FACTORY",
                "sglang.multimodal_gen.runtime.realtime.whitebox.browser_ui:factory",
            )
        )
        self.temporary = tempfile.TemporaryDirectory(prefix="whitebox-room-")
        self.directory = Path(self.temporary.name).resolve()
        self.environment = {
            **os.environ,
            "WORLDKIT_ROOM_DIR": str(self.directory / "rooms"),
            "WORLDKIT_SESSION_DIR": str(self.directory / "sessions"),
        }
        self.slots = [
            dict(credential=secrets.token_hex(32), subject=new_id(), actorId=new_id())
            for _ in range(slots)
        ]
        private_json(self.directory / "authorization.json", self.slots)
        self.members, self.views = {}, {}
        self.process = self.diagnostics = self.prepared = None
        self.ready = None
        self.closed = False
        self.lock = asyncio.Lock()
        self.operations = set()
        self.closing = None

    def command(self, operation, *args):
        return [
            os.environ.get("WHITEBOX_NODE", "node"),
            str(self.entry),
            "room",
            operation,
            *args,
            "--timeout",
            str(self.timeout),
            "--json",
        ]

    async def call(self, operation, *args):
        process = await asyncio.create_subprocess_exec(
            *self.command(operation, *args),
            env=self.environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=MAX_REPLY_BYTES,
        )
        try:
            output, errors = await asyncio.wait_for(process.communicate(), self.timeout)
            if process.returncode or len(output) > MAX_REPLY_BYTES:
                # Only stable ASCII codes enter server logs. CLI stderr can
                # otherwise include private paths, URLs or capability values.
                for line in errors.splitlines():
                    try:
                        diagnostic = json.loads(line)
                        code = diagnostic.get("error", {}).get("code", "")
                        if (
                            diagnostic.get("kind") == "worldkit-cli-error"
                            and isinstance(code, str)
                            and re.fullmatch(r"[A-Z][A-Z0-9_]{1,100}", code)
                        ):
                            logging.getLogger(__name__).warning(
                                "Room CLI operation failed operation=%s code=%s",
                                operation,
                                code,
                            )
                    except (ValueError, AttributeError):
                        continue
                raise ValueError("ROOM_OPERATION_UNKNOWN")
            value = json.loads(output)
            if not isinstance(value, dict):
                raise ValueError("ROOM_OPERATION_UNKNOWN")
            return value
        except (asyncio.TimeoutError, json.JSONDecodeError) as error:
            raise ValueError("ROOM_OPERATION_UNKNOWN") from error
        finally:
            await stop_process(process)

    async def start(self):
        # Use the existing receipt-verified downloader, then start from that
        # exact archive. A mutable world reference cannot change this live room.
        self.prepared = await prepare_delivery(
            self.entry,
            self.world_id,
            str(self.directory),
            self.trace_id,
            **(
                {
                    "create_context": self.create_context,
                    "verified_caller": self.verified_caller,
                }
                if self.create_context is not None
                else {}
            ),
        )
        self.process = await asyncio.create_subprocess_exec(
            *self.command(
                "create",
                "--delivery",
                self.prepared["archive"],
                "--authorization-file",
                str(self.directory / "authorization.json"),
                "--width",
                str(self.spec.width),
                "--height",
                str(self.spec.height),
                "--fps",
                str(self.capture_fps),
                "--service",
                "off",
            ),
            env=self.environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=MAX_REPLY_BYTES,
        )
        self.diagnostics = asyncio.create_task(
            cli_diagnostics(self.process, self.trace_id)
        )
        line = await asyncio.wait_for(self.process.stdout.readline(), self.timeout)
        try:
            value = json.loads(line)
        except (ValueError, UnicodeError) as error:
            raise ValueError("ROOM_READY_IDENTITY_INVALID") from error
        if not isinstance(value, dict):
            raise ValueError("ROOM_READY_IDENTITY_INVALID")
        self.ready = value
        capacity, room, artifact = (
            value.get("capacity"),
            value.get("room", {}),
            value.get("artifact", {}),
        )
        if (
            value.get("kind") != "worldkit-room-ready"
            or value.get("schemaVersion") != 1
            or type(capacity) is not int
            or capacity < 1
            or value.get("playConfiguration")
            != {"playMode": "multiplayer", "multiplayer": {"maxPlayers": capacity}}
            or not isinstance(room, dict)
            or not isinstance(artifact, dict)
            or not isinstance(value.get("roomId"), str)
            or not value["roomId"]
            or value.get("roomId") != room.get("roomId")
            or room.get("runtimeHash") != artifact.get("runtimeHash")
            or room.get("contentHash") != artifact.get("sourceHash")
            or artifact != self.prepared["identity"]
        ):
            raise ValueError("ROOM_READY_IDENTITY_INVALID")
        if self.create_context is not None and value.get(
            "playConfiguration"
        ) != self.create_context["frozenPublication"].get("playConfiguration"):
            raise ValueError("ROOM_FROZEN_PLAY_CONFIGURATION_CHANGED")
        if capacity > len(self.slots):
            raise ValueError("ROOM_HOST_CAPACITY_INSUFFICIENT")
        self.max_players = capacity

    def check_open(self):
        if self.closed or self.process.returncode is not None:
            raise ValueError("ROOM_EXECUTION_UNAVAILABLE")

    async def member_view(self, player_id, member, descriptor=None):
        descriptor = descriptor or await read_member_session(
            member, self.entry, purpose="model-source"
        )
        validate_descriptor(descriptor, self.spec, fps=self.capture_fps)
        binding = descriptor.get("playerBinding", {})
        if (
            member.get("kind") != "worldkit-room-member"
            or member.get("schemaVersion") != 1
            or member.get("roomId") != self.ready["roomId"]
            or descriptor.get("accessMode") != "capability"
            or descriptor.get("media", {}).get("kind") != "pixels"
            or descriptor.get("worldBuildHash")
            != self.ready["artifact"]["worldBuildHash"]
            or binding.get("room") != self.ready["room"]
            or binding != member.get("binding")
            or descriptor.get("sessionId") != member.get("sessionId")
            or descriptor.get("epoch") != member.get("epoch")
        ):
            raise ValueError("ROOM_MEMBER_IDENTITY_INVALID")
        player_descriptor = descriptor
        if descriptor.get("modelContext") is not None:
            player_descriptor = await read_member_session(
                member, self.entry, purpose="viewer"
            )
            if any(
                player_descriptor.get(key) != descriptor.get(key)
                for key in (
                    "sessionId",
                    "epoch",
                    "baseUrl",
                    "playerBinding",
                    "worldBuildHash",
                    "uiBundleHash",
                    "audience",
                    "media",
                    "width",
                    "height",
                    "fps",
                )
            ):
                raise ValueError("ROOM_MEMBER_IDENTITY_INVALID")
        package = await asyncio.to_thread(load_local_ui_package, descriptor)
        context = SimpleNamespace(
            descriptor=descriptor,
            spec=self.spec,
            renderer_factory=self.renderer_factory,
            trace_id=self.trace_id,
            prepared_delivery={
                key: self.prepared[key]
                for key in (
                    "reference",
                    "operation",
                    "receipt",
                    "receiptSha256",
                    "identity",
                )
                if key in self.prepared
            },
            ui_package=package,
            owner=self,
            player_id=player_id,
            # Model-purpose capture has no input lease; the original player
            # channel remains the controller even in a locally hosted Room.
            external_input=self.external_input
            or descriptor.get("modelContext") is not None,
        )
        view = RoomMemberView(context)
        # Public player relay must authenticate the ordinary input capability,
        # independently of the trusted capture/model-feedback descriptor.
        view.player_descriptor = player_descriptor
        view.metadata.update(
            max_players=self.max_players,
            player_binding=binding,
            artifact=self.ready["artifact"],
        )
        try:
            await view.start()
            self.check_open()
        except BaseException:
            await NativeCLIView.close(view)
            raise
        return view

    async def join(self, *, player_id, spec):
        checked_id(player_id)
        task = asyncio.current_task()
        self.operations.add(task)
        try:
            async with self.lock:
                self.check_open()
                if spec != self.spec:
                    raise ValueError("ROOM_JOIN_UNAVAILABLE")
                record = self.members.get(player_id)
                if record is None:
                    if len(self.members) >= self.max_players:
                        raise ValueError("ROOM_PLAYER_CAPACITY_EXCEEDED")
                    used = {value["slot"] for value in self.members.values()}
                    index = next(
                        index for index in range(len(self.slots)) if index not in used
                    )
                    record = {
                        "slot": index,
                        "request_id": "join-" + new_id(),
                        "lock": asyncio.Lock(),
                    }
                    record["credential_file"] = self.directory / (
                        record["request_id"] + "-credential.json"
                    )
                    private_json(
                        record["credential_file"],
                        {"credential": self.slots[index]["credential"]},
                    )
                    self.members[player_id] = record
            # Slow media startup is serialized per member, never across players.
            async with record["lock"]:
                self.check_open()
                if player_id in self.views:
                    raise ValueError("ROOM_PLAYER_ALREADY_CONNECTED")
                if self.members.get(player_id) is not record or record.get("leaving"):
                    raise ValueError("ROOM_MEMBER_LEAVING")
                # Unknown results retain the exact request, credential and slot.
                member = await self.call(
                    "join",
                    "--room",
                    self.ready["roomId"],
                    "--credential-file",
                    str(record["credential_file"]),
                    "--request-id",
                    record["request_id"],
                )
                record["member"] = member
                self.check_open()
                try:
                    view = await self.member_view(player_id, member)
                except BaseException:
                    # Keep a blocked cleanup reserved; another player's admission
                    # cannot claim its actor/resources until the SDK reports left.
                    record["leaving"] = True
                    try:
                        await self._leave(player_id, record)
                    except Exception:
                        pass
                    raise
                self.views[player_id] = view
                return view
        finally:
            self.operations.discard(task)

    async def public_session(self, player_id, public_origin):
        self.check_open()
        record = self.members.get(player_id)
        if record is None:
            raise ValueError("ROOM_MEMBER_UNAVAILABLE")
        async with record["lock"]:
            self.check_open()
            if record.get("leaving") or player_id not in self.views:
                raise ValueError("ROOM_MEMBER_LEAVING")
            descriptor = await read_member_session(
                record["member"],
                self.entry,
                purpose="viewer",
                public_origin=public_origin,
            )
            if record.get("leaving") or self.members.get(player_id) is not record:
                raise ValueError("ROOM_MEMBER_LEAVING")
            return descriptor

    async def inspect(self, player_id=None):
        self.check_open()
        if player_id is None:
            return await self.call("inspect", "--room", self.ready["roomId"])
        return await self.call(
            "inspect", "--member-file", self.members[player_id]["member"]["memberFile"]
        )

    async def resume(self, player_id):
        task = asyncio.current_task()
        self.operations.add(task)
        try:
            record = self.members[player_id]
            async with record["lock"]:
                self.check_open()
                if record.get("leaving"):
                    raise ValueError("ROOM_MEMBER_LEAVING")
                member = record["member"]
                if "resume_from" not in record:
                    record["resume_from"] = member["binding"]
                    previous = self.views.pop(player_id, None)
                    if previous:
                        await NativeCLIView.close(previous)
                    # No automatic retry: this can commit even if stdout is lost.
                    record["member"] = await self.call(
                        "resume", "--member-file", member["memberFile"]
                    )
                else:
                    await self.inspect(player_id)
                descriptor = await read_member_session(
                    record["member"], self.entry, purpose="model-source"
                )
                binding, old = (
                    descriptor.get("playerBinding", {}),
                    record["resume_from"],
                )
                if (
                    any(
                        binding.get(key) != old.get(key)
                        for key in ("room", "playerSessionId", "viewportId", "mode")
                    )
                    or binding.get("connectionEpoch") != old["connectionEpoch"] + 1
                    or binding.get("viewportBindingRevision")
                    != old["viewportBindingRevision"] + 1
                    or binding.get("render", {}).get("sessionId")
                    == old["render"]["sessionId"]
                ):
                    raise ValueError("ROOM_RESUME_UNCONFIRMED")
                # Public readSession resolves the atomically replaced private
                # file. Reconstruct only the existing CLI result metadata.
                current = {
                    **record["member"],
                    "sessionId": descriptor["sessionId"],
                    "epoch": descriptor["epoch"],
                    "binding": binding,
                }
                view = await self.member_view(player_id, current, descriptor)
                record["member"] = current
                del record["resume_from"]
                self.views[player_id] = view
                return view
        finally:
            self.operations.discard(task)

    async def _leave(self, player_id, record):
        if "member" not in record:
            # Reconcile an interrupted join with its original journal identity.
            record["member"] = await self.call(
                "join",
                "--room",
                self.ready["roomId"],
                "--credential-file",
                str(record["credential_file"]),
                "--request-id",
                record["request_id"],
            )
        try:
            async with asyncio.timeout(self.timeout):
                result = await self.call(
                    "leave", "--member-file", record["member"]["memberFile"]
                )
                # Retirement crosses later SDK commits. Observe that exact
                # member through the original Room owner; media is already shut
                # down, so inspecting its old media session cannot prove cleanup.
                binding = record["member"]["binding"]
                while result.get("phase") == "cleaning":
                    snapshot = await self.call(
                        "inspect", "--room", self.ready["roomId"]
                    )
                    sessions = snapshot.get("sessions")
                    if not isinstance(sessions, list) or any(
                        not isinstance(row, dict) for row in sessions
                    ):
                        raise ValueError("ROOM_LEAVE_UNCONFIRMED")
                    matches = [
                        row
                        for row in sessions
                        if row.get("playerSessionId") == binding["playerSessionId"]
                    ]
                    if (
                        snapshot.get("room") != self.ready["room"]
                        or snapshot.get("ended")
                        or len(matches) != 1
                        or matches[0].get("connectionEpoch")
                        != binding["connectionEpoch"]
                        or matches[0].get("cleanupBlocked")
                    ):
                        raise ValueError("ROOM_LEAVE_UNCONFIRMED")
                    result = matches[0]
                    if result.get("phase") == "cleaning":
                        await asyncio.sleep(0.05)
                if result.get("phase") != "left":
                    raise ValueError("ROOM_LEAVE_UNCONFIRMED")
        except (ValueError, asyncio.TimeoutError) as error:
            # Missing/ended Rooms and blocked or expired observations never
            # authorize reuse of a slot whose retirement was not confirmed.
            raise ValueError("ROOM_LEAVE_UNCONFIRMED") from error
        self.members.pop(player_id, None)
        self.views.pop(player_id, None)
        record["credential_file"].unlink(missing_ok=True)

    async def leave(self, player_id, *, view=None):
        record = self.members.get(player_id)
        if record is None or self.closed:
            return
        async with record["lock"]:
            if view is not None and self.views.get(player_id) is not view:
                return
            record["leaving"] = True
            current = self.views.get(player_id)
            if current:
                await NativeCLIView.close(current)
            await self._leave(player_id, record)

    async def cancel_join(self, player_id):
        """RoomRuntime rollback before a stream entry has been published."""
        if self.closed or player_id not in self.members or player_id in self.views:
            # A duplicate join rejection cannot retire the existing player.
            return
        await self.leave(player_id)
        if player_id in self.members or player_id in self.views:
            raise ValueError("ROOM_LEAVE_UNCONFIRMED")
        return MemberCleanupReceipt(player_id)

    async def _close(self):
        self.closed = True
        running = [
            task for task in self.operations if task is not asyncio.current_task()
        ]
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        try:
            results = await asyncio.gather(
                *(NativeCLIView.close(view) for view in list(self.views.values())),
                return_exceptions=True,
            )
            if self.ready and self.process and self.process.returncode is None:
                try:
                    await self.call("close", "--room", self.ready["roomId"])
                    await asyncio.wait_for(self.process.wait(), 5)
                except (ValueError, asyncio.TimeoutError):
                    pass  # Foreground process termination is the final cleanup owner.
        finally:
            await stop_process(self.process)
            if self.diagnostics:
                await asyncio.gather(self.diagnostics, return_exceptions=True)
            self.temporary.cleanup()
        if any(isinstance(result, BaseException) for result in results):
            raise ValueError("ROOM_CLEANUP_FAILED")

    async def close(self):
        if self.closing is None:
            self.closing = asyncio.create_task(self._close())
        await asyncio.shield(self.closing)


async def worldkit_room(
    *,
    world_id,
    spec,
    trace_id,
    cli_fps=None,
    source_options=None,
    create_context=None,
    verified_caller=None,
):
    if source_options:
        raise ValueError("ROOM_SOURCE_OPTIONS_UNSUPPORTED")
    entry = Path(os.environ["WHITEBOX_CLI_ENTRYPOINT"])
    slots = os.environ.get("WHITEBOX_MAX_PLAYERS", "")
    # The SDK private authorization contract currently admits at most 128 rows.
    if (
        not entry.is_absolute()
        or not entry.is_file()
        or not slots.isdecimal()
        or not 1 <= int(slots) <= 128
    ):
        raise ValueError("ROOM_HOST_CONFIGURATION_REQUIRED")
    fps = spec.fps if cli_fps is None else cli_fps
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError("CLI_FPS_UNSUPPORTED")
    room = WorldkitRoom(
        entry=entry,
        world_id=world_id,
        spec=spec,
        trace_id=trace_id,
        slots=int(slots),
        capture_fps=fps,
        create_context=create_context,
        verified_caller=verified_caller,
    )
    try:
        await room.start()
        return room
    except BaseException as original_error:
        _, interrupted = await drain_startup_cleanup(room.close())
        error = asyncio.CancelledError() if interrupted else original_error
        if (
            not isinstance(error, (Exception, asyncio.CancelledError))
            or create_context is None
        ):
            if interrupted:
                raise error
            raise
        # close acknowledges the original process group, views and private
        # temporary files. A thrown/cancelled close never creates this receipt.
        diagnostic = error
        if (
            not isinstance(error, asyncio.CancelledError)
            and room.diagnostics
            and not room.diagnostics.cancelled()
        ):
            try:
                diagnostic = room.diagnostics.result() or error
            except Exception:
                pass  # Diagnostic failure cannot invalidate completed cleanup.
        raise startup_cleanup_error(
            diagnostic, FactoryCleanupReceipt(trace_id)
        ) from error
