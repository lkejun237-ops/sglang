# SPDX-License-Identifier: Apache-2.0
"""Trusted single-play reconciliation in the original RenderHost/source journal.

WTP owns the business operation and freezes the original init. These rows only
fence execution: a retained reservation is never permission to recreate a source.
"""

import asyncio
import base64
import copy
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time

import httpx
import msgspec
from fastapi import Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .render_contract import (
    PROTOCOL,
    RenderError,
    RenderJournal,
    canonical,
    source_create_matches,
)
from .room_control import RoomControlCapabilities, exact, identities
from .rooms import checked_id, require_internal_token
from .rtc_auth import RTCCapabilities

ACTIONS = {"start", "getOperation", "getAttachment", "close"}
RETENTION = 7 * 86400
TERMINAL_STATES = {"cancelled", "failed", "expired", "lost"}


def validate_operation(value):
    exact(
        value,
        {
            "operationId",
            "parametersSha256",
            "caller",
            "frozenPublication",
            "frozenLaunchParameters",
        },
        {"_route", "controlSequence"},
    )
    json.dumps(value, allow_nan=False)
    identities(value["caller"], {"serviceId", "subject"})
    if (
        not isinstance(value["operationId"], str)
        or not 1 <= len(value["operationId"]) <= 256
        or not isinstance(value["parametersSha256"], str)
        or not re.fullmatch(r"[a-f0-9]{64}", value["parametersSha256"])
    ):
        raise ValueError("Invalid single player operation")
    # WTP's verified historical single-player scopes may omit this field.
    # Present runtime metadata must use the SDK label, not public target "single".
    frozen = value["frozenPublication"]
    if (
        not isinstance(frozen, dict)
        or frozen.get("status") != "verified"
        or frozen.get("scope") != "world-frozen-publication"
        or (
            "playConfiguration" in frozen
            and (
                not isinstance(frozen["playConfiguration"], dict)
                or frozen["playConfiguration"].get("playMode") != "single-player"
            )
        )
    ):
        raise ValueError("A frozen single player publication is required")
    world = frozen.get("world")
    if not isinstance(world, dict) or any(
        not isinstance(world.get(key), str) or not 1 <= len(world[key]) <= 256
        for key in ("sourceId", "worldId", "publicationId")
    ):
        raise ValueError("Invalid frozen world identity")
    from .runtime_management import world_reference

    world_reference(world["sourceId"] + "/" + world["worldId"])
    params = value["frozenLaunchParameters"]
    exact(params, {"initMsgpackBase64", "transport", "publicationIdentity"})
    from .cli_room import publication_identity

    selected = publication_identity(params["publicationIdentity"])
    if any(selected[key] != world[key] for key in selected if key in world) or (
        "attempt" in world
        and params["publicationIdentity"].get("attempt") != world["attempt"]
    ):
        raise ValueError("The frozen publication identity has changed")
    if params["transport"] not in {"websocket", "webrtc-rtp"}:
        raise ValueError("Invalid single player transport")
    try:
        raw = base64.b64decode(params["initMsgpackBase64"], validate=True)
        if len(raw) > 8 * 1024**2:
            raise ValueError()
        init = msgspec.msgpack.decode(raw)
        if (
            not isinstance(init, dict)
            or init.get("type") != "init"
            or init.get("control_protocol") != PROTOCOL
            or not source_create_matches(init)
            or not isinstance(init.get("whitebox"), dict)
            or init["whitebox"].get("source_id") != world["sourceId"]
            or init["whitebox"].get("world_id") != world["worldId"]
            or "game_id" in init["whitebox"]
        ):
            raise ValueError()
    except (ValueError, TypeError, KeyError, msgspec.DecodeError):
        raise ValueError("Invalid frozen single player init") from None
    if "controlSequence" in value and (
        type(value["controlSequence"]) is not int or value["controlSequence"] < 0
    ):
        raise ValueError("Invalid replay position")
    return init


class SinglePlayerControl:
    def __init__(self, host, *, now=time.time):
        self.host, self.now = host, now
        self.lock, self.tasks, self.close_tasks = asyncio.Lock(), {}, {}
        self.pending_source_failures = set()
        self.secret = secrets.token_urlsafe(32)
        host.single_player_control = self

    def _identity(self, operation):
        validate_operation(operation)
        caller = operation["caller"]
        key = (
            "single-player:"
            + hashlib.sha256(
                canonical(
                    [caller["serviceId"], caller["subject"], operation["operationId"]]
                )
            ).hexdigest()
        )
        fingerprint = hashlib.sha256(
            canonical(
                {
                    k: operation[k]
                    for k in (
                        "operationId",
                        "parametersSha256",
                        "caller",
                        "frozenPublication",
                        "frozenLaunchParameters",
                    )
                }
            )
        ).hexdigest()
        return key, fingerprint

    async def _read(self, operation):
        key, fingerprint = self._identity(operation)
        raw = await self.host.store.get(key)
        row = json.loads(raw) if raw is not None else None
        if row and row["fingerprint"] != fingerprint:
            raise ValueError("The original single player operation has changed")
        return key, row

    async def _update(self, key, change):
        for _ in range(20):
            raw = await self.host.store.get(key)
            if raw is None:
                raise ValueError("The single player operation has expired")
            row = json.loads(raw)
            change(row)
            if await self.host.store.cas(key, raw, canonical(row).decode(), RETENTION):
                return row
        raise ValueError("The single player operation is busy")

    def _result(self, operation, row, state=None):
        result = dict(
            operationId=operation["operationId"],
            parametersSha256=operation["parametersSha256"],
            state=state or (row["state"] if row else "unknown"),
        )
        if row and row.get("execution"):
            result["execution"] = copy.deepcopy(row["execution"])
        if (
            row
            and result["state"] == "failed"
            and row.get("failureCode") in {"CAPACITY_EXHAUSTED", "PLAYER_IDLE_TIMEOUT"}
        ):
            result["failureCode"] = row["failureCode"]
        return result

    async def _reserve(self, operation, state):
        key, row = await self._read(operation)
        if row is None:
            _, fingerprint = self._identity(operation)
            row = dict(
                fingerprint=fingerprint,
                state=state,
                ownerEpoch=self.host.runtime.epoch,
                execution=None,
            )
            if not await self.host.store.cas(
                key, None, canonical(row).decode(), RETENTION
            ):
                # An expired row is deliberately not reusable in SQLiteRenderStore.
                key, row = await self._read(operation)
                if row is None:
                    return key, None, False
                return key, row, False
            return key, row, True
        return key, row, False

    async def start(self, operation):
        async with self.lock:
            key, prior = await self._read(operation)
            if prior is None:
                key, row, created = await self._reserve(operation, "starting")
                if row is None:
                    return self._result(operation, None, "expired")
                if created:
                    task = asyncio.create_task(
                        self._launch(key, copy.deepcopy(operation)),
                        name="single-player-launch",
                    )
                    self.tasks[key] = task
                    task.add_done_callback(lambda done: self.tasks.pop(key, None))
            task = self.tasks.get(key)
        if task is not None:
            # HTTP cancellation/lost replies never cancel the original launch.
            await asyncio.shield(task)
        return await self.get_operation(operation)

    async def _launch(self, key, operation):
        pipeline = None
        preparation = None
        try:
            route = operation["_route"]
            if route.get("status") in {"expired", "ended"}:
                await self._update(key, lambda row: row.update(state="expired"))
                return
            if route["owner_id"] != self.host.runtime.owner_id or route.get(
                "owner_epoch"
            ) not in {None, self.host.runtime.epoch}:
                await self._update(key, lambda row: row.update(state="lost"))
                return
            init = validate_operation(operation)
            init["whitebox"]["game_id"] = checked_id(route["game_id"])
            body = dict(
                user_id=operation["caller"]["subject"],
                trace_id=operation["operationId"],
                message=init,
            )
            transport = operation["frozenLaunchParameters"]["transport"]
            if transport == "webrtc-rtp":
                from .model_geometry import session_source_spec
                from .rtc_routes import reserve_rtc

                source_spec = session_source_spec(
                    init["whitebox"], self.host.config.spec, self.host.runtime.rooms
                )
                if "source_encoding" in init["whitebox"] or (
                    source_spec.width,
                    source_spec.height,
                ) == (1280, 720):
                    # Resolve omitted resolution from the deployment/Room before
                    # allocating a source. Native admission freezes its actual
                    # capture rate and SPS for the later media attachment.
                    preparation, _ = await reserve_rtc(self.host, body)

            async def reserved(value):
                nonlocal pipeline
                pipeline = value
                if preparation is not None:
                    # No native peer exists yet. Freeze the verified SPS for
                    # the first attachment and subsequent reconnects.
                    pipeline.rtc_source_profile = body.get("source_encoding_profile")
                pipeline.single_player_operation = key
                pipeline.session.expected_publication = copy.deepcopy(
                    operation["frozenLaunchParameters"]["publicationIdentity"]
                )
                execution = dict(
                    gameId=value.game_id,
                    sourceSessionId=value.session_id,
                    sourceEpoch=value.session.source_epoch,
                    ownerId=self.host.runtime.owner_id,
                    ownerEpoch=self.host.runtime.epoch,
                    publicationId=operation["frozenPublication"]["world"][
                        "publicationId"
                    ],
                )
                await self._update(key, lambda row: row.update(execution=execution))

            # Uses exactly the original factory, source clock, model and startup.
            await self.host.attach(
                body,
                transport=transport,
                preparation=preparation,
                on_reserved=reserved,
                detached=True,
            )
            _, row = await self._read(operation)
            if row["state"] in {"cancelling", "cancelled"}:
                await pipeline.close("PLAYER_CANCELLED")
                await self._update(key, lambda row: row.update(state="cancelled"))
        except (Exception, asyncio.CancelledError) as error:
            if pipeline is not None:
                await pipeline.close("PLAYER_START_FAILED")
            logging.getLogger(__name__).warning(
                "Single player launch interrupted: %s", type(error).__name__
            )
            failure = (
                {"failureCode": error.code}
                if isinstance(error, RenderError) and error.code == "CAPACITY_EXHAUSTED"
                else {}
            )
            await self._update(
                key,
                lambda row: row.update(
                    state="cancelled"
                    if row["state"] in {"cancelling", "cancelled"}
                    else "failed",
                    **failure,
                ),
            )
            if isinstance(error, asyncio.CancelledError):
                raise
        finally:
            if preparation is not None:
                async with self.host.lock:
                    if self.host.runtime.rtc_preparation == preparation:
                        self.host.runtime.rtc_preparation = None

    async def source_failed(self, key, code):
        await self._update(
            key,
            lambda row: row.update(
                **(
                    {"state": "failed", "failureCode": code}
                    if row["state"] not in {"cancelling", "cancelled"}
                    else {}
                )
            ),
        )

    async def _released_source(self, operation, row):
        """Only the original owner's durable journal can prove retired cleanup."""
        execution = row.get("execution")
        if not execution:
            return None
        journal = RenderJournal(
            self.host.store,
            user_id=operation["caller"]["subject"],
            source_session_id=execution["sourceSessionId"],
            source_epoch=execution["sourceEpoch"],
            owner_epoch=row["ownerEpoch"],
        )
        try:
            state = await journal.read()
        except RenderError as error:
            if error.code in {"ROOM_SESSION_LOST", "SOURCE_EPOCH_MISMATCH"}:
                return None
            raise
        if (
            state.get("source_session_id") != execution["sourceSessionId"]
            or state.get("game_id") != execution["gameId"]
            or state.get("source_state") != "ended"
            or state.get("cleanup_state") != "released"
        ):
            return None
        return state

    async def get_operation(self, operation):
        key, row = await self._read(operation)
        if row is None:
            return self._result(operation, None)
        if row["state"] in TERMINAL_STATES:
            return self._result(operation, row)
        if row["ownerEpoch"] != self.host.runtime.epoch:
            # A restart cannot restore a source, but its original durable
            # release receipt can still establish the retired outcome.
            released = await self._released_source(operation, row)
            return self._result(operation, row, "failed" if released else "lost")
        execution = row.get("execution")
        pipeline = (
            self.host.sessions.get(execution["sourceSessionId"]) if execution else None
        )
        if pipeline is None:
            if key in self.pending_source_failures or any(
                key in tasks and not tasks[key].done()
                for tasks in (self.tasks, self.close_tasks)
            ):
                return self._result(operation, row)
            # The source may have finished between this read and registry
            # observation. Re-read its durable terminal outcome before lost.
            _, current = await self._read(operation)
            if current and current["state"] in TERMINAL_STATES:
                return self._result(operation, current)
            if await self._released_source(operation, row):
                return self._result(operation, row, "failed")
            return self._result(operation, row, "lost")
        if row["state"] == "cancelling":
            return self._result(operation, row)
        return self._result(
            operation, row, "ready" if pipeline.ready is not None else "starting"
        )

    async def _finish_close(self, key, pipeline):
        await pipeline.close("PLAYER_CANCELLED")
        # An independently started original teardown can make close() return
        # early. Only the original closed flag proves its teardown finished.
        if pipeline.closed:
            await self._update(key, lambda row: row.update(state="cancelled"))

    async def close(self, operation):
        async with self.lock:
            key, row, _ = await self._reserve(operation, "cancelled")
            if row is None:
                return self._result(operation, None, "expired")
            if row["state"] == "cancelled":
                return self._result(operation, row)
            same_owner = row["ownerEpoch"] == self.host.runtime.epoch
            execution = row.get("execution")
            pipeline = (
                self.host.sessions.get(execution["sourceSessionId"])
                if same_owner and execution
                else None
            )
            released = (
                await self._released_source(operation, row)
                if pipeline is None
                else None
            )
            if released is not None:
                # Exit confirms resource retirement, not successful gameplay.
                # Preserve the source outcome separately from this close receipt.
                row = await self._update(
                    key,
                    lambda current: current.update(
                        state="cancelled",
                        sourceOutcome=released.get("source_outcome"),
                        sourceCleanupState="released",
                    ),
                )
                return self._result(operation, row)
            if not same_owner:
                return self._result(operation, row, "lost")
            if row["state"] in TERMINAL_STATES:
                return self._result(operation, row)
            await self._update(key, lambda row: row.update(state="cancelling"))
            launch = self.tasks.get(key)
            existing = self.close_tasks.get(key)
            if (launch and not launch.done()) or (existing and not existing.done()):
                return self._result(operation, row, "cancelling")
            execution = row.get("execution")
            pipeline = (
                self.host.sessions.get(execution["sourceSessionId"])
                if execution
                else None
            )
            task = None
            if pipeline is not None:
                task = self.close_tasks[key] = asyncio.create_task(
                    self._finish_close(key, pipeline), name="single-player-close"
                )
                task.add_done_callback(lambda done: self.close_tasks.pop(key, None))
        if task is not None:
            # A lost close response cannot abort the original source teardown.
            await asyncio.shield(task)
        return await self.get_operation(operation)

    async def get_attachment(self, operation):
        result = await self.get_operation(operation)
        if result["state"] not in {"ready", "starting"} or "execution" not in result:
            return result
        pipeline = self.host.sessions[result["execution"]["sourceSessionId"]]
        # Do not let an early RTC attachment complete before the original init
        # creates its startup readiness events. Source-ready is not required.
        if not getattr(pipeline, "control_started", False):
            return result
        after = operation.get(
            "controlSequence",
            0 if pipeline.generation == 0 else pipeline.control_sequence,
        )
        if not pipeline.history_floor <= after <= pipeline.control_sequence:
            raise ValueError("The source replay position has expired")
        expires = int(self.now()) + 60
        message = dict(
            type="source.bind" if pipeline.session.binding_ended else "resume",
            control_protocol=PROTOCOL,
            whitebox=copy.deepcopy(pipeline.init["whitebox"]),
            source_session_id=pipeline.session_id,
            source_epoch=pipeline.session.source_epoch,
            owner_epoch=self.host.runtime.epoch,
            connection_generation=pipeline.generation,
            control_sequence=after,
            resume_user_id=pipeline.user_id,
        )
        claims = dict(
            aud="whitebox-single-attachment",
            sub=pipeline.user_id,
            exp=expires,
            messageSha256=hashlib.sha256(canonical(message)).hexdigest(),
        )
        payload = base64.urlsafe_b64encode(canonical(claims)).decode().rstrip("=")
        signature = (
            base64.urlsafe_b64encode(
                hmac.new(
                    self.secret.encode(), payload.encode(), hashlib.sha256
                ).digest()
            )
            .decode()
            .rstrip("=")
        )
        result["attachment"] = dict(
            expiresAt=expires,
            transport=pipeline.media_transport,
            message=dict(message, player_authorization=payload + "." + signature),
        )
        return result

    def authorize_attachment(self, pipeline, message, user):
        try:
            claims = RTCCapabilities(self.secret)._decode(
                message.get("player_authorization")
            )
            expected = {k: v for k, v in message.items() if k != "player_authorization"}
            if (
                claims.get("aud") != "whitebox-single-attachment"
                or claims.get("sub") != user
                or type(claims.get("exp")) is not int
                or not self.now() < claims["exp"]
                or claims.get("messageSha256")
                != hashlib.sha256(canonical(expected)).hexdigest()
                or message.get("connection_generation") != pipeline.generation
            ):
                raise ValueError()
            return dict(message, resume_token=pipeline.token)
        except (ValueError, TypeError, KeyError):
            raise ValueError(
                "The player attachment authorization has expired"
            ) from None

    async def shutdown(self):
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        await asyncio.gather(*self.close_tasks.values(), return_exceptions=True)


class SinglePlayerCapabilities(RoomControlCapabilities):
    """Separate service audience/scope using the existing signed-byte codec."""

    def __init__(self, secret, services, *, now=time.time):
        super().__init__(
            secret,
            services,
            now=now,
            audience="whitebox-single-control",
            actions=ACTIONS,
        )

    def verify(self, token, *, raw, action, origin, operation):
        claims = self._decode(token)
        if (
            claims.get("scope") != "single-player-control"
            or "origin" not in claims
            or claims["origin"] != origin
            or claims.get("service_id") != operation["caller"]["serviceId"]
            or claims.get("sub") != operation["caller"]["subject"]
        ):
            raise ValueError("Single player control access was denied")
        return super().verify(token, raw=raw, action=action)


def validate_request(body):
    exact(body, {"schemaVersion", "action", "singlePlayerOperation"})
    if (
        type(body["schemaVersion"]) is not int
        or body["schemaVersion"] != 1
        or body["action"] not in ACTIONS
    ):
        raise ValueError("Invalid single player control request")
    op = body["singlePlayerOperation"]
    if "_route" in op:
        raise ValueError("The source route is server owned")
    validate_operation(op)
    return body


def install_owner_single_player_control(app, host, token):
    control = getattr(host, "single_player_control", None) or SinglePlayerControl(host)

    @app.post("/internal/single-player-control")
    async def single_player(body: dict, authorization: str = Header(default="")):
        require_internal_token(token, authorization)
        try:
            request = validate_request(body["request"])
            op = request["singlePlayerOperation"]
            if op["caller"] != body["verifiedCaller"]:
                raise ValueError()
            route = body["route"]
            world = op["frozenPublication"]["world"]
            if (
                route["owner_id"] != host.runtime.owner_id
                or route.get("world_id") != world["sourceId"] + "/" + world["worldId"]
            ):
                raise ValueError()
            method = {
                "start": control.start,
                "getOperation": control.get_operation,
                "getAttachment": control.get_attachment,
                "close": control.close,
            }[request["action"]]
            result = await method(dict(op, _route=route))
            return JSONResponse(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, KeyError):
            raise HTTPException(
                409, "The original player operation is unavailable"
            ) from None


def install_gateway_single_player_control(app, proxy, *, verifier=None):
    from .room_control import read_request

    if verifier is None:
        secret = os.environ.get("WHITEBOX_SINGLE_CONTROL_SECRET", "")
        services = os.environ.get("WHITEBOX_SINGLE_CONTROL_SERVICES", "")
        if secret or services:
            verifier = SinglePlayerCapabilities(
                secret, {v.strip() for v in services.split(",") if v.strip()}
            )

    @app.post("/v1/whitebox/single-player-control")
    @app.post("/backends/minwm/v1/whitebox/single-player-control")
    async def single_player(request: Request):
        raw, body = await read_request(
            request, validator=validate_request, max_bytes=12 * 1024**2
        )
        if verifier is None:
            raise HTTPException(503, "Single player control is not configured")
        op = body["singlePlayerOperation"]
        try:
            authorization = request.headers.get("authorization", "")
            if not authorization.startswith("Bearer "):
                raise ValueError()
            caller = verifier.verify(
                authorization[7:],
                raw=raw,
                action=body["action"],
                origin=request.headers.get("origin"),
                operation=op,
            )
        except (ValueError, TypeError, KeyError):
            raise HTTPException(
                403, "Single player control access was denied"
            ) from None
        unknown = dict(
            operationId=op["operationId"],
            parametersSha256=op["parametersSha256"],
            state="unknown",
        )
        try:
            # Close may reserve an execution tombstone before a delayed start.
            # Reads and media attachment never allocate a directory association.
            route = await proxy.directory.control_route(
                caller, body, create=body["action"] in {"start", "close"}
            )
            if route is None:
                return JSONResponse(unknown, headers={"Cache-Control": "no-store"})
            origin = proxy.owners.get(route["owner_id"])
            if origin is None or origin.rstrip("/") != route["owner_url"].rstrip("/"):
                raise ValueError()
            response = await proxy.client.post(
                origin.rstrip("/") + "/internal/single-player-control",
                json=dict(verifiedCaller=caller, request=body, route=route),
                headers={"Authorization": "Bearer " + proxy.token},
            )
            response.raise_for_status()
            if len(response.content) > 2 * 1024**2:
                raise ValueError()
            return JSONResponse(response.json(), headers={"Cache-Control": "no-store"})
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 409:
                raise HTTPException(
                    409, "The original player operation is unavailable"
                ) from None
            return JSONResponse(unknown, headers={"Cache-Control": "no-store"})
        except httpx.HTTPError:
            return JSONResponse(unknown, headers={"Cache-Control": "no-store"})
        except ValueError:
            raise HTTPException(
                409, "The original player operation is unavailable"
            ) from None
