# SPDX-License-Identifier: Apache-2.0
"""Control receipts around existing RoomRuntime and SDK membership owners.

Tasks outlive HTTP requests. Original operation IDs and caller bindings fence
retries; observations never invoke create/join/leave. A process restart loses
these observations, but the directory's owner epoch prevents another startup.
"""

import asyncio
import copy
import json
import time

from fastapi import Header, HTTPException

from .contracts import VideoSpec
from .diagnostics import log_player_failure
from .model_geometry import validate_room_spec
from .room_control import (
    exact,
    frozen_publication_identity,
    identities,
    unknown,
    validate_request,
)
from .room_startup import (
    FactoryCleanupReceipt,
    RoomCleanupReceipt,
    startup_cleanup_receipt,
)
from .rooms import checked_id, new_id, require_internal_token


class RoomControlRuntime:
    def __init__(self, runtime):
        self.runtime = runtime
        self.creates, self.admissions, self.leaves = {}, {}, {}
        self.tasks = set()
        self.closed = False

    def dispatch(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def drain(self):
        while self.tasks:
            await asyncio.gather(*list(self.tasks))

    async def close(self):
        self.closed = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*list(self.tasks), return_exceptions=True)

    def key(self, caller, operation_id):
        return caller["serviceId"], caller["subject"], operation_id

    def create_result(self, record):
        op = record["invocation"]["createContext"]["operation"]
        result = dict(
            operationId=op["operationId"],
            parametersSha256=op["parametersSha256"],
            status=record["status"],
        )
        if record["status"] == "succeeded":
            # A successful old create is not a resumable source after retirement
            # or lease loss. Never publish a replacement owner under this ID.
            room = self.runtime.rooms.get(record["game_id"])
            if room is None or record.get("retiring"):
                result["status"] = "unknown"
            else:
                result.update(gameId=record["game_id"], ownerEpoch=self.runtime.epoch)
        elif record["status"] == "failed":
            result.update(
                cleanupComplete=True, errorCode=record["startup_failure"].code
            )
        return result

    def member_presence(self, game_id):
        counts = dict(online=0, reconnecting=0, joining=0, cleaning=0, disconnected=0)
        for member in self.admissions.values():
            if member["binding"]["gameId"] != game_id or member["status"] == "retired":
                continue
            pipeline = member.get("media_pipeline")
            sub = pipeline.subscription if pipeline is not None else None
            if member.get("leaving") or member["status"] == "blocked":
                phase = "cleaning"
            elif (
                sub is not None
                and not sub.closed
                and sub.peer.connected.is_set()
                and not sub.peer.closed
                and not getattr(sub.peer, "error", None)
            ):
                phase = "online"
            elif (
                pipeline is not None
                and not pipeline.closed
                and (
                    pipeline.retention is not None or pipeline.rtc_reconnect is not None
                )
            ):
                phase = "reconnecting"
            elif member.get("media", {}).get("status") == "unavailable" or (
                pipeline is not None and pipeline.closed
            ):
                phase = "disconnected"
            else:
                phase = "joining"
            counts[phase] += 1
        return counts

    async def start(self, record):
        runtime = self.runtime
        game_id = record["game_id"]
        lock = runtime.locks.setdefault(game_id, asyncio.Lock())
        try:
            async with lock:
                route = await runtime.directory.resolve(
                    game_id, record["route"]["world_id"]
                )
                if route["owner_id"] != runtime.owner_id or route[
                    "owner_epoch"
                ] not in {None, runtime.epoch}:
                    raise ValueError("The room owner has changed")
                if game_id in runtime.rooms:
                    raise ValueError("The room already exists")
                if getattr(runtime, "maintenance", None) is not None:
                    await runtime.maintenance.cancel_for_play()
                body = record["invocation"]
                spec = VideoSpec(**body["videoSpec"])
                validate_room_spec(spec, runtime.spec)
                room = await runtime._start_room(
                    game_id,
                    route,
                    lock,
                    body["cliFps"],
                    None,
                    body["createContext"]["operation"]["operationId"],
                    create_context=body["createContext"],
                    verified_caller=record["caller"],
                    spec=spec,
                )
                room["control_caller"] = record["caller"]
                room["control_record"] = record
                record["status"] = "succeeded"
        except BaseException as error:
            receipt = startup_cleanup_receipt(error)
            if receipt is None:
                # Cancellation without a positive owner receipt stays unknown.
                record["status"] = "unknown"
                log_player_failure(
                    "room.start", error, record=record, outcome="unknown"
                )
                return
            record["startup_failure"] = error
            operation = record["invocation"]["createContext"]["operation"][
                "operationId"
            ]
            record["status"] = (
                "failed"
                if isinstance(receipt, RoomCleanupReceipt)
                and receipt.factory.operation_id == operation
                and receipt.owner_epoch == runtime.epoch
                else "unknown"
            )
            log_player_failure(
                "room.start", error, record=record, outcome=record["status"]
            )

    def original_create(self, caller, body, route):
        context = body["createContext"]
        key = self.key(caller, context["operation"]["operationId"])
        invocation = {k: body[k] for k in ("createContext", "videoSpec", "cliFps")}
        record = self.creates.get(key)
        if record is not None:
            if (
                json.dumps(record["invocation"], sort_keys=True)
                != json.dumps(invocation, sort_keys=True)
                or record["game_id"] != route["game_id"]
            ):
                raise ValueError("The original room operation has changed")
        elif body["action"] == "createRoom":
            if len(self.creates) >= 4096 or route["owner_epoch"] is not None:
                raise ValueError("The original room owner is unavailable")
            record = dict(
                caller=caller,
                invocation=invocation,
                game_id=route["game_id"],
                route=route,
                status="pending",
            )
            self.creates[key] = record
            record["task"] = self.dispatch(self.start(record))
        return record

    def bound_room(self, binding):
        runtime = self.runtime
        room = runtime.rooms.get(checked_id(binding["gameId"]))
        if (
            room is None
            or binding["ownerEpoch"] != runtime.epoch
            or "control_record" not in room
        ):
            raise ValueError("The original room execution is unavailable")
        record = room["control_record"]
        world = record["invocation"]["createContext"]["frozenPublication"]["world"]
        if world["publicationId"] != binding["publicationId"] or record.get("retiring"):
            raise ValueError("The original publication is unavailable")
        return room

    def admission_result(self, record, *, leave_id=None):
        result = dict(
            admissionOperationId=record["admission"]["operationId"],
            parametersSha256=record["admission"]["parametersSha256"],
            status=record["status"],
        )
        if leave_id is not None:
            result["leaveOperationId"] = leave_id
        if record["status"] == "claimed":
            result["memberId"] = record["member_id"]
        elif record["status"] == "retired":
            result["cleanupComplete"] = True
        else:
            result["errorCode"] = "ADMISSION_UNCONFIRMED"
        return result

    async def admit(self, record, room):
        runtime = self.runtime
        stream_id = record["stream_id"]
        admission_stage = "model_reserve"
        try:
            owner = getattr(runtime, "member_media", None)
            if owner is not None and not owner.bound_compute:
                owner = None  # Legacy non-bound media has no parent protocol.
            if owner is not None:
                await owner.reserve(record)
                record["sdk_join_started"] = True
                record["sdk_join_done"].clear()
            admission_stage = "sdk_join"
            try:
                joined = runtime.join(
                    dict(
                        game_id=record["binding"]["gameId"],
                        world_id=room["route"]["world_id"],
                        spec=record["spec"],
                        player_id=record["player_id"],
                        stream_id=stream_id,
                        session_id=record["session_id"],
                        user_id=record["caller"]["subject"],
                        trace_id=record["admission"]["operationId"],
                        control_caller=room["control_caller"],
                    )
                )
                await joined
            finally:
                if owner is not None:
                    record["sdk_join_done"].set()
            admission_stage = "member_identity"
            entry = runtime.player(stream_id, touch=False)
            entry["control_admission"] = record
            if record.get("model_binding") is not None:
                entry["player_expires_at"] = record[
                    "model_binding"
                ].assignment.player_expires_at
            engine = entry["engine"]
            # SDK membership identity remains authoritative. Legacy mock drivers
            # use their server-assigned player ID, never a browser supplied ID.
            descriptor = getattr(engine, "descriptor", None)
            record["member_id"] = (
                descriptor.get("playerBinding", {}).get("playerSessionId")
                if descriptor is not None
                else record["player_id"]
            )
            if not isinstance(record["member_id"], str) or not record["member_id"]:
                raise ValueError("The member identity is unavailable")
            if not record.get("leaving"):
                record["status"] = "claimed"
                entry["input_activity"].start(time.monotonic())
                if record.get("mediaLaunch") is not None:
                    owner = getattr(runtime, "member_media", None)
                    if owner is None:
                        record["media"] = dict(
                            status="unavailable", reason="MEMBER_MEDIA_UNAVAILABLE"
                        )
                    else:
                        record["media_task"] = self.dispatch(owner.start(record))
        except BaseException as error:
            log_player_failure(
                "member.admit",
                error,
                record=record,
                outcome="unknown",
                phase=admission_stage,
            )
            record["status"] = "blocked"
            record["leaving"] = True
            binding = record.get("model_binding")
            if binding is not None:
                try:
                    await binding.close()
                except Exception as cleanup_error:
                    log_player_failure(
                        "member.rollback",
                        cleanup_error,
                        record=record,
                        outcome="unknown",
                    )
                    record["cleanup_unknown"] = True
            if isinstance(error, asyncio.CancelledError):
                raise

    async def leave(self, record):
        try:
            await record["task"]
            if await self.runtime.leave(record["stream_id"]) is not True:
                raise ValueError("The original player cleanup remains unconfirmed")
            # An unresolved join may have committed a member without a stream.
            # Its cleanup must be confirmed by that original driver, not absence.
            room = self.runtime.rooms.get(record["binding"]["gameId"])
            if (
                "member_id" not in record
                and record["stream_id"] not in self.runtime.member_retirements
            ):
                raise ValueError("The original admission remains unconfirmed")
            if room is not None and record["stream_id"] in room["players"]:
                raise ValueError("The player cleanup remains unconfirmed")
            record["status"] = "retired"
        except BaseException as error:
            log_player_failure("member.leave", error, record=record, outcome="unknown")
            record["status"] = "blocked"

    async def retire(self, record):
        try:
            await record["task"]
            game = record["game_id"]
            failure = record.get("startup_failure")
            factory = startup_cleanup_receipt(failure)
            operation = record["invocation"]["createContext"]["operation"][
                "operationId"
            ]
            proven_factory = (
                isinstance(factory, FactoryCleanupReceipt)
                and factory.operation_id == operation
            )
            if record["status"] == "failed":
                record["retire_status"] = "succeeded"
                return
            if record["status"] != "succeeded" and not (
                proven_factory and game in self.runtime.retiring
            ):
                # A failed factory can have committed resources before its
                # reply disappeared. Process-local absence is not cleanup.
                record["retire_status"] = "unknown"
                return
            lock = self.runtime.locks.setdefault(game, asyncio.Lock())
            async with lock:
                # Natural shutdown may have already removed the room. Only
                # the original owner's successful cleanup can settle this call.
                room = self.runtime.rooms.get(game) or self.runtime.retiring.get(game)
                if room is None and record.get("room_cleanup_complete") is True:
                    record["retire_status"] = "succeeded"
                    return
                if room is None or (
                    record["status"] == "succeeded"
                    and room.get("control_record") is not record
                ):
                    # Revalidate after acquiring the original lifecycle lock.
                    record["retire_status"] = "unknown"
                    return
                complete = await self.runtime._end(
                    game, "completed", wait_for_log=True, expected_room=room
                )
            record["retire_status"] = "succeeded" if complete else "unknown"
            if complete:
                for admission in self.admissions.values():
                    if admission["binding"]["gameId"] == game:
                        admission["leaving"], admission["status"] = True, "retired"
        except BaseException as error:
            log_player_failure("room.retire", error, record=record, outcome="unknown")
            record["retire_status"] = "unknown"

    async def handle(self, caller, body, route):
        if self.closed or route["owner_id"] != self.runtime.owner_id:
            raise ValueError("The room owner is unavailable")
        action = body["action"]
        if "createContext" in body:
            record = self.original_create(caller, body, route)
            if record is None:
                return unknown(body)
            result = self.create_result(record)
            if action != "retireRoom":
                if (
                    action == "getOperation"
                    and result["status"] == "succeeded"
                    and getattr(self.runtime, "member_media", None) is not None
                ):
                    result["memberPresence"] = self.member_presence(result["gameId"])
                return result
            target = body["execution"]
            frozen = record["invocation"]["createContext"]["frozenPublication"]["world"]
            if target != dict(
                gameId=record["game_id"],
                ownerEpoch=self.runtime.epoch,
                publicationId=frozen["publicationId"],
            ):
                raise ValueError("The retirement target has changed")
            task = record.get("retire_task")
            if task is None or task.done() and record["retire_status"] == "unknown":
                record["retiring"] = True
                record["retire_status"] = "pending"
                record["retire_task"] = self.dispatch(self.retire(record))
            return dict(
                operationId=result["operationId"],
                parametersSha256=result["parametersSha256"],
                status=record["retire_status"],
                execution=target,
                cleanupComplete=record["retire_status"] == "succeeded",
            )

        if action in {"admitPlayer", "getAdmission"}:
            key = self.key(caller, body["admission"]["operationId"])
            record = self.admissions.get(key)
            if record is None:
                if action != "admitPlayer":
                    raise ValueError("The original admission was not found")
                room = self.bound_room(body["binding"])
                if "mediaLaunch" in body:
                    frozen = room["control_record"]["invocation"]["createContext"][
                        "frozenPublication"
                    ]
                    if body["mediaLaunch"][
                        "publicationIdentity"
                    ] != frozen_publication_identity(frozen):
                        raise ValueError("The member launch publication has changed")
                if len(self.admissions) >= 4096:
                    raise ValueError("The admission registry is full")
                record = dict(
                    caller=caller,
                    binding=body["binding"],
                    admission=body["admission"],
                    player_id=new_id(),
                    stream_id=new_id(),
                    session_id=new_id(),
                    spec=room["control_record"]["invocation"]["videoSpec"],
                    status="blocked",
                )
                if "mediaLaunch" in body:
                    record["mediaLaunch"] = copy.deepcopy(body["mediaLaunch"])
                self.admissions[key] = record
                record["task"] = self.dispatch(self.admit(record, room))
            elif (
                record["binding"] != body["binding"]
                or record["admission"] != body["admission"]
                or record.get("mediaLaunch") != body.get("mediaLaunch")
            ):
                raise ValueError("The original admission has changed")
            leave_id = body.get("leaveOperationId")
            if (
                leave_id is not None
                and self.leaves.get(self.key(caller, leave_id), {}).get("record")
                is not record
            ):
                raise ValueError("The original leave operation was not found")
            if (
                action == "getAdmission"
                and record["status"] != "retired"
                and route.get("status") == "ended"
                and route.get("game_id") == record["binding"]["gameId"]
                and route.get("owner_epoch")
                == record["binding"]["ownerEpoch"]
                == self.runtime.epoch
            ):
                # Directory termination fences this exact admission; it does
                # not prove the SDK/media owners have released their resources
                # or supersede an already verified retirement receipt.
                return dict(
                    admissionOperationId=record["admission"]["operationId"],
                    parametersSha256=record["admission"]["parametersSha256"],
                    status="blocked",
                    errorCode="ROOM_ENDED",
                    **({"leaveOperationId": leave_id} if leave_id is not None else {}),
                )
            if record["status"] == "claimed":
                try:
                    self.bound_room(record["binding"])
                    self.runtime.player(record["stream_id"], touch=False)
                except ValueError:
                    record["status"] = "blocked"
            return self.admission_result(record, leave_id=leave_id)

        binding = body["memberBinding"]
        record = self.admissions.get(self.key(caller, binding["admissionOperationId"]))
        if record is None or binding != {
            **record["binding"],
            "memberId": record.get("member_id"),
            "admissionOperationId": record["admission"]["operationId"],
        }:
            raise ValueError("The original player binding has changed")
        if action in {"getPlayerSession", "getPlayerMedia"}:
            self.bound_room(binding)
            if record["status"] != "claimed" or record.get("leaving"):
                raise ValueError("The player is unavailable")
            # Pod-local SDK descriptors are never public. The real media owner
            # must publish a reachable, parser-validated session before enabling.
            transport = getattr(self.runtime, "player_transport", None)
            if transport is None:
                raise HTTPException(503, "Public player transport is not configured")
            result = (
                await transport.get_player_media(
                    record, cursor=body.get("controlSequence")
                )
                if action == "getPlayerMedia"
                else dict(
                    status="ready", session=await transport.public_session(record)
                )
            )
            if action == "getPlayerMedia":
                self.bound_room(binding)
            if record["status"] != "claimed" or record.get("leaving"):
                raise ValueError("The player is leaving")
            return result
        op = body["operation"]
        key = self.key(caller, op["operationId"])
        leave = self.leaves.get(key)
        if leave is None:
            if len(self.leaves) >= 4096:
                raise ValueError("The leave registry is full")
            # Fence input/session publication before waiting for a late join.
            record["leaving"], record["status"] = True, "blocked"
            self.leaves[key] = leave = dict(record=record, operation=op)
        elif leave["record"] is not record or leave["operation"] != op:
            raise ValueError("The original leave operation has changed")
        task = leave.get("task")
        if record["status"] != "retired" and (task is None or task.done()):
            leave["task"] = self.dispatch(self.leave(record))
        return self.admission_result(record, leave_id=op["operationId"])


def install_runtime_room_control(app, token, runtime):
    runtime.control = RoomControlRuntime(runtime)

    @app.post("/internal/room-control")
    async def control(body: dict, authorization: str = Header(default="")):
        require_internal_token(token, authorization)
        try:
            exact(body, {"verifiedCaller", "request", "route"})
            identities(body["verifiedCaller"], {"serviceId", "subject"})
            request = validate_request(body["request"])
            return await runtime.control.handle(
                body["verifiedCaller"], request, body["route"]
            )
        except (KeyError, ValueError, TypeError):
            raise HTTPException(
                409, "The original room operation is unavailable"
            ) from None
