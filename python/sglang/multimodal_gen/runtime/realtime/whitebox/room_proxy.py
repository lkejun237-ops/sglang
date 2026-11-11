# SPDX-License-Identifier: Apache-2.0
"""Disposable Gateway relay. No model lease, RGB transport or recording owner."""

import asyncio
import json
import os
from urllib.parse import urlsplit, urlunsplit

import httpx
import msgspec
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from .render_contract import source_create_matches
from .rooms import HTTPRoomDirectory, MemoryRoomStore, RoomDirectory, checked_id, new_id


class RoomProxy:
    def __init__(self, directory, owners, token, client, connect):
        # Reuse strict origin validation; callers never choose an upstream URL.
        if owners or os.environ.get("WHITEBOX_DYNAMIC_ROOMS") != "1":
            RoomDirectory(MemoryRoomStore(), owners)
        self.directory, self.owners, self.token = directory, owners, token
        self.client, self.connect = client, connect

    @classmethod
    def from_environment(cls, connect):
        client = httpx.AsyncClient(timeout=httpx.Timeout(30, connect=5))
        token = os.environ["WHITEBOX_INTERNAL_TOKEN"]
        return cls(
            HTTPRoomDirectory(
                client, os.environ["WHITEBOX_ROOM_COORDINATOR_URL"], token
            ),
            json.loads(os.environ["WHITEBOX_ROOM_OWNERS"]),
            token,
            client,
            connect,
        )

    async def owner_origin(self, ident, epoch=None):
        # Resolve through the authenticated Coordinator, never a browser URL.
        if os.environ.get("WHITEBOX_DYNAMIC_ROOMS") == "1":
            return await self.directory.owner_origin(ident, epoch)
        origin = self.owners.get(ident)
        if origin is None:
            raise ValueError("ROOM_ROUTE_UNAVAILABLE")
        return origin.rstrip("/")

    async def close(self):
        await self.client.aclose()

    async def serve(self, websocket, *, user_id, trace_id, transport="websocket"):
        await websocket.accept()
        upstream, tasks = None, []
        terminal_forwarded = False
        rtc_create_id = None
        try:
            wire = await asyncio.wait_for(websocket.receive_bytes(), 30)
            if len(wire) > 8 * 1024**2 + 4096:
                raise ValueError("INIT_TOO_LARGE")
            message = msgspec.msgpack.decode(wire)
            if not isinstance(message, dict) or message.get("type") not in {
                "init",
                "resume",
                "source.bind",
                "world.prepare",
            }:
                raise ValueError("INIT_REQUIRED")
            if "source_encoding" in message.get("whitebox", {}):
                from .session_encoding import check_source_encoding_scope

                if message.get("type") == "init":
                    check_source_encoding_scope(message, transport=transport)
                elif transport != "webrtc-rtp" or message.get("type") not in {
                    "resume",
                    "source.bind",
                }:
                    raise ValueError("Source encoding requires native RTC")
            render_v2 = message.get("control_protocol") == "whitebox-render-v2"
            if message.get("startup") is not None and (
                not render_v2
                or message["type"] != "init"
                or os.environ.get("WHITEBOX_MODEL_STARTUP_PREPARE") != "1"
            ):
                raise ValueError("CAPABILITY_UNAVAILABLE")
            if transport == "webrtc-rtp":
                from .diagnostics import WhiteboxDiagnostic
                from .rtc_auth import RTCCapabilities

                if (
                    os.environ.get("WHITEBOX_NATIVE_RTC_ENABLED") != "1"
                    or not render_v2
                ):
                    raise WhiteboxDiagnostic("RTC_UNAVAILABLE", "transport")
                if message["type"] == "init":
                    if message.get("whitebox", {}).get("game_id"):
                        raise WhiteboxDiagnostic(
                            "RTC_AUTHORIZATION_FAILED", "validation"
                        )
                    try:
                        if not hasattr(self, "rtc_capabilities"):
                            self.rtc_capabilities = RTCCapabilities(
                                os.environ.get("WHITEBOX_RTC_AUTH_SECRET", "")
                            )
                        claims = self.rtc_capabilities.verify(
                            message.pop("rtc_authorization", None),
                            origin=websocket.headers.get("origin"),
                            world=message.get("whitebox"),
                        )
                        user_id, rtc_create_id = claims["sub"], claims["jti"]
                    except ValueError:
                        raise WhiteboxDiagnostic(
                            "RTC_AUTHORIZATION_FAILED", "validation"
                        ) from None
                elif message["type"] in {"resume", "source.bind"}:
                    # This string is not authorization. The Room checks the
                    # unguessable source capability and all epoch/owner bindings.
                    user_id = message.get("resume_user_id")
                    if not isinstance(user_id, str) or not 1 <= len(user_id) <= 256:
                        raise WhiteboxDiagnostic(
                            "RTC_AUTHORIZATION_FAILED", "validation"
                        )
                else:
                    raise WhiteboxDiagnostic("RTC_UNAVAILABLE", "validation")
            if render_v2 and os.environ.get("WHITEBOX_RENDER_SEGMENTS") != "1":
                raise ValueError("CAPABILITY_UNAVAILABLE")
            if (
                render_v2
                and message["type"] == "init"
                and not source_create_matches(message)
            ):
                raise ValueError("SOURCE_CREATE_REQUIRED")
            world = message.get("whitebox", {})
            world_id, source = world.get("world_id"), world.get("source_id")
            if not isinstance(world_id, str) or not 1 <= len(world_id) <= 256:
                raise ValueError("INVALID_WORLD")
            if source is not None:
                if (
                    not isinstance(source, str)
                    or not 1 <= len(source) <= 128
                    or "/" in source
                    or "/" in world_id
                ):
                    raise ValueError("INVALID_WORLD_SOURCE")
                world_id = source + "/" + world_id
            if message["type"] == "world.prepare":
                # This branch never allocates a game ID, Room or model lease.
                # It uses the existing playtest entry and internal Room token.
                if (
                    len(wire) > 1024
                    or os.getenv("WHITEBOX_PREFETCH_ENABLED") != "1"
                    or len(self.owners) != 1
                ):
                    raise ValueError("PREPARATION_UNAVAILABLE")

                async def prepare():
                    nonlocal terminal_forwarded
                    async with self.client.stream(
                        "POST",
                        next(iter(self.owners.values())).rstrip("/")
                        + "/internal/worlds/prepare",
                        json={"world_id": world_id},
                        timeout=httpx.Timeout(320, connect=5),
                        headers={"Authorization": "Bearer " + self.token},
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if len(line) > 8192:
                                raise ValueError("PREPARATION_TOO_LARGE")
                            value = json.loads(line)
                            if value.get("type") != "world.preparation":
                                raise ValueError("INVALID_PREPARATION")
                            await asyncio.wait_for(
                                websocket.send_bytes(msgspec.msgpack.encode(value)), 5
                            )
                            if value.get("state") in {"ready", "unavailable"}:
                                terminal_forwarded = True
                                return

                async def cancelled():
                    # Disconnect or any follow-up message ends this resource-only
                    # operation; it cannot be upgraded into a live session.
                    await websocket.receive_bytes()

                tasks = [
                    asyncio.create_task(prepare()),
                    asyncio.create_task(cancelled()),
                ]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                return
            create = not world.get("game_id") and message["type"] == "init"
            # One signed create capability maps to one persistent game route
            # across Gateway replicas; a replay cannot create a second world.
            game_id = (
                (rtc_create_id or new_id())
                if create
                else checked_id(world.get("game_id"))
            )
            route = await self.directory.resolve(game_id, world_id, create=create)
            origin = await self.owner_origin(
                route["owner_id"], route.get("owner_epoch")
            )
            if origin is None or origin.rstrip("/") != route["owner_url"].rstrip("/"):
                raise ValueError("ROOM_ROUTE_UNAVAILABLE")
            message["whitebox"] = dict(world, game_id=game_id)
            parts = urlsplit(origin)
            url = urlunsplit(
                (
                    "wss" if parts.scheme == "https" else "ws",
                    parts.netloc,
                    (
                        "/internal/render-rtc"
                        if transport == "webrtc-rtp"
                        else (
                            "/internal/render-v2" if render_v2 else "/internal/pipeline"
                        )
                    ),
                    "",
                    "",
                )
            )
            upstream = await self.connect(
                url,
                additional_headers={"Authorization": "Bearer " + self.token},
                max_size=8 * 1024**2 + 16384,
                max_queue=1,
                compression=None,
                open_timeout=10,
                close_timeout=2,
                ping_interval=10,
                ping_timeout=30,
            )
            await upstream.send(
                msgspec.msgpack.encode(
                    dict(user_id=user_id, trace_id=trace_id, message=message)
                )
            )

            async def send():
                while True:
                    data = await websocket.receive_bytes()
                    if len(data) > (8 * 1024**2 + 16384 if render_v2 else 65536):
                        raise ValueError("CONTROL_MESSAGE_TOO_LARGE")
                    await asyncio.wait_for(upstream.send(data), 5)

            async def receive():
                nonlocal terminal_forwarded
                async for data in upstream:
                    if not isinstance(data, bytes):
                        raise ValueError("INVALID_ROOM_RESPONSE")
                    await asyncio.wait_for(websocket.send_bytes(data), 5)
                    if msgspec.msgpack.decode(data).get("type") in {
                        "error",
                        "session.completed",
                    }:
                        terminal_forwarded = True
                        return

            tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except Exception as error:
            if transport == "webrtc-rtp" and upstream is None:
                from .diagnostics import public_error

                try:
                    await websocket.send_bytes(
                        msgspec.msgpack.encode(
                            dict(
                                type="error",
                                **public_error(
                                    error,
                                    trace_id,
                                    default="RTC_CONNECTION_FAILED",
                                    stage="transport",
                                ),
                                reconnectable=False,
                            )
                        )
                    )
                    terminal_forwarded = True
                except Exception:
                    pass
            # Network loss is reconnectable. Room explicitly sends terminal
            # errors; a disposable relay must not manufacture a terminal event.
            pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if upstream:
                await upstream.close()
            try:
                await websocket.close(code=1000 if terminal_forwarded else 1012)
            except Exception:
                pass

    def install(self, app):
        from .room_control import install_gateway_room_control

        install_gateway_room_control(app, self)
        from .player_control import install_gateway_single_player_control

        install_gateway_single_player_control(app, self)
        from .player_transport import install_player_gateway

        install_player_gateway(app, self)

        from .runtime_management import install_gateway_management

        install_gateway_management(
            app, self, os.environ.get("WHITEBOX_RTC_AUTH_SECRET", "")
        )

        @app.get("/v1/whitebox/owners/{owner_id}/render/sources/{source_id}")
        async def render_status(owner_id: str, source_id: str, request: Request):
            try:
                checked_id(source_id)
            except ValueError:
                raise HTTPException(404)
            try:
                origin = await self.owner_origin(owner_id)
            except (ValueError, httpx.HTTPError):
                raise HTTPException(404) from None
            if not origin or not request.headers.get("authorization", "").startswith(
                "Bearer "
            ):
                raise HTTPException(404)
            try:
                response = await self.client.get(
                    origin.rstrip("/") + "/v1/whitebox/render/sources/" + source_id,
                    params=dict(request.query_params),
                    headers={"authorization": request.headers["authorization"]},
                )
            except httpx.HTTPError:
                raise HTTPException(503, "The source owner is unavailable.")
            from fastapi.responses import Response

            return Response(
                response.content,
                status_code=response.status_code,
                media_type="application/json",
                headers={"Cache-Control": "no-store"},
            )

        # Historical capabilities can be served after a verified, copy-only
        # archive migration to the sole configured Room owner.
        if len(self.owners) == 1:

            @app.api_route("/v1/whitebox/recordings/{recording_id}", methods=["GET"])
            @app.api_route(
                "/v1/whitebox/recordings/{recording_id}/{suffix:path}",
                methods=["GET", "POST"],
            )
            async def legacy_recording(
                recording_id: str, request: Request, suffix: str = ""
            ):
                return await recording(
                    next(iter(self.owners)), recording_id, request, suffix
                )

        @app.api_route(
            "/v1/whitebox/owners/{owner_id}/recordings/{recording_id}", methods=["GET"]
        )
        @app.api_route(
            "/v1/whitebox/owners/{owner_id}/recordings/{recording_id}/{suffix:path}",
            methods=["GET", "POST"],
        )
        async def recording(
            owner_id: str, recording_id: str, request: Request, suffix: str = ""
        ):
            try:
                checked_id(recording_id)
            except ValueError:
                raise HTTPException(404)
            try:
                origin = await self.owner_origin(owner_id)
            except (ValueError, httpx.HTTPError):
                raise HTTPException(404) from None
            if not origin or any(part in {".", ".."} for part in suffix.split("/")):
                raise HTTPException(404)
            if not request.headers.get("authorization", "").startswith("Bearer "):
                raise HTTPException(403)
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > 65536:
                    raise HTTPException(413)
            path = f"/v1/whitebox/recordings/{recording_id}" + (
                "/" + suffix if suffix else ""
            )
            outgoing = self.client.build_request(
                request.method,
                origin.rstrip("/") + path,
                params=list(request.query_params.multi_items()),
                content=bytes(data),
                headers={
                    key: request.headers[key]
                    for key in ("authorization", "content-type")
                    if key in request.headers
                },
            )
            try:
                response = await self.client.send(outgoing, stream=True)
            except httpx.HTTPError:
                raise HTTPException(503, "The recording owner is unavailable")
            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers={
                    key: value
                    for key, value in response.headers.items()
                    if key.lower()
                    in {
                        "content-type",
                        "content-length",
                        "cache-control",
                        "x-content-sha256",
                        "x-content-type-options",
                    }
                },
                background=BackgroundTask(response.aclose),
            )
