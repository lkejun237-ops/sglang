# SPDX-License-Identifier: Apache-2.0
"""Fixed public SDK routes on the existing Gateway/Room relay.

No media decoding or new input owner: SDK packets and its viewer capability
cross the original authenticated hop. The directory contains route metadata only.
"""

import asyncio
import copy
import hmac
import os
import re
from dataclasses import dataclass
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx
from fastapi import Header, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, Response

from .native_pixels import MAX_HEADER, MAX_PIXELS
from .room_control import frozen_publication_identity
from .rooms import checked_id, checked_session_id, require_internal_token

MAX_UI_BYTES = 32 * 1024 * 1024
MAX_WIRE_BYTES = MAX_PIXELS + MAX_HEADER + 4
MAX_VIEWER_CONTROL_BYTES = 1024 * 1024


def origin(value):
    if not isinstance(value, str):
        raise ValueError("Player transport origins are not configured")
    url = urlsplit(value)
    local = url.hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        (url.scheme != "https" and not (url.scheme == "http" and local))
        or not url.hostname
        or url.username
        or url.password
        or url.path
        or url.query
        or url.fragment
    ):
        raise ValueError("Player transport requires exact secure origins")
    authority = url.netloc.lower()
    if (url.scheme == "https" and authority.endswith(":443")) or (
        url.scheme == "http" and authority.endswith(":80")
    ):
        authority = authority.rsplit(":", 1)[0]
    if value != url.scheme + "://" + authority or not value.isascii():
        raise ValueError("Player transport origins must be canonical")
    return value


@dataclass(frozen=True)
class PlayerTransportConfig:
    public_origin: str
    player_origin: str
    control_origin: str
    member_prefill_origin: str | None = None

    def __post_init__(self):
        origin(self.public_origin)
        origin(self.player_origin)
        origin(self.control_origin)
        if self.member_prefill_origin is not None:
            origin(self.member_prefill_origin)
        host = urlsplit(self.control_origin).hostname
        if host in {
            urlsplit(self.public_origin).hostname,
            urlsplit(self.player_origin).hostname,
            urlsplit(self.member_prefill_origin).hostname
            if self.member_prefill_origin
            else None,
        }:
            raise ValueError("Player transport must not share the control Cookie host")

    @classmethod
    def from_environment(cls):
        public = os.getenv("WHITEBOX_ROOM_PUBLIC_ORIGIN")
        player = os.getenv("WHITEBOX_PLAYER_ORIGIN")
        control = os.getenv("WHITEBOX_CONTROL_ORIGIN")
        prefill = os.getenv("WHITEBOX_MEMBER_PREFILL_PLAYER_ORIGIN")
        return (
            cls(public, player, control, prefill)
            if public or player or control or prefill
            else None
        )

    def admits_origin(self, value):
        return (
            value == self.player_origin
            or self.member_prefill_origin is not None
            and value == self.member_prefill_origin
        )

    def require_member_origin(self, value, entry):
        if value == self.player_origin:
            return
        from .member_prefill import PROTOCOL

        if value != self.member_prefill_origin or entry.get(
            "control_admission", {}
        ).get("mediaLaunch", {}).get("memberPrefill") != {"protocol": PROTOCOL}:
            raise ValueError("The player origin is unavailable for this member")


def protocols(websocket):
    values = websocket.scope.get("subprotocols", [])
    tokens = [value[13:] for value in values if value.startswith("worldkit.cap.")]
    if (
        len(values) != 2
        or "worldkit.stream.v1" not in values
        or len(tokens) != 1
        or not re.fullmatch(r"[a-f0-9]{48}", tokens[0])
    ):
        raise ValueError("Invalid SDK viewer capability")
    return values, tokens[0]


def query(request):
    values = list(request.query_params.multi_items())
    if len({key for key, _ in values}) != len(values):
        raise ValueError("Invalid SDK transport query")
    for key, value in values:
        if key == "clientId" and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
            continue
        if key == "role" and value in {"spectator", "controller"}:
            continue
        if key in {"ui", "clock"} and value in {"0", "1"}:
            continue
        raise ValueError("Invalid SDK transport query")
    return urlencode(values)


def ws_url(base, path, params=""):
    parsed = urlsplit(base)
    return urlunsplit(
        ("wss" if parsed.scheme == "https" else "ws", parsed.netloc, path, params, "")
    )


async def relay(socket, upstream, *, live=None, channel="control", observe=None):
    maximum = MAX_VIEWER_CONTROL_BYTES if channel == "control" else MAX_WIRE_BYTES

    async def to_sdk():
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if live:
                live()
            payload = message.get("bytes")
            if payload is None:
                payload = message.get("text")
            if (
                not isinstance(payload, (str, bytes))
                or len(payload.encode() if isinstance(payload, str) else payload)
                > maximum
            ):
                raise ValueError("Invalid SDK transport message")
            await upstream.send(payload)
            if observe and isinstance(payload, str):
                await observe("to-sdk", payload)

    async def from_sdk():
        async for message in upstream:
            if observe and isinstance(message, str):
                await observe("from-sdk", message)
            if live:
                live()
            if isinstance(message, bytes):
                await socket.send_bytes(message)
            else:
                await socket.send_text(message)

    tasks = [asyncio.create_task(to_sdk()), asyncio.create_task(from_sdk())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def get_bytes(client, url, *, headers, maximum):
    async with client.stream(
        "GET", url, headers=headers, follow_redirects=False
    ) as response:
        if response.status_code != 200:
            raise ValueError("The player response is unavailable")
        data = bytearray()
        async for chunk in response.aiter_bytes():
            data.extend(chunk)
            if len(data) > maximum:
                raise ValueError("The player response exceeds its budget")
        return response.headers.get("content-type", "application/octet-stream"), bytes(
            data
        )


class PlayerTransport:
    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, config

    def viewer(self, stream_id, session_id, token=None):
        entry = self.runtime.player(checked_id(stream_id), touch=False)
        admission = entry.get("control_admission")
        descriptor = getattr(entry["engine"], "player_descriptor", None)
        if descriptor is None:
            descriptor = getattr(entry["engine"], "descriptor", None)
        if (
            not admission
            or admission.get("status") != "claimed"
            or admission.get("leaving")
            or not descriptor
            or descriptor.get("sessionId") != checked_session_id(session_id)
            or descriptor.get("accessMode") != "capability"
        ):
            raise ValueError("The player is unavailable")
        if token is not None and (
            not re.fullmatch(r"[a-f0-9]{48}", token)
            or not hmac.compare_digest(token, descriptor.get("accessToken", ""))
        ):
            raise ValueError("The viewer capability is unavailable")
        return entry, descriptor

    async def public_session(self, record):
        if self.config is None:
            raise HTTPException(503, "Public player transport is not configured")
        runtime = self.runtime
        entry = runtime.player(record["stream_id"], touch=False)
        room = runtime.rooms[entry["game_id"]]
        publish = getattr(room["driver"], "public_session", None)
        if publish is None:
            raise HTTPException(503, "Public player transport is unavailable")
        session = await publish(
            entry["identity"]["player_id"], self.config.public_origin
        )
        current, private = self.viewer(
            record["stream_id"], session["sessionId"], session["accessToken"]
        )
        if current is not entry or any(
            session.get(key) != private.get(key)
            for key in (
                "epoch",
                "worldBuildHash",
                "uiBundleHash",
                "playerBinding",
                "audience",
                "media",
                "width",
                "height",
                "fps",
            )
        ):
            raise ValueError("The player execution has changed")
        await runtime.directory.publish_session(
            dict(
                session_id=session["sessionId"],
                game_id=entry["game_id"],
                owner_id=runtime.owner_id,
                owner_epoch=runtime.epoch,
                stream_id=record["stream_id"],
                epoch=session["epoch"],
            )
        )
        self.viewer(record["stream_id"], session["sessionId"], session["accessToken"])
        return session

    async def get_player_media(self, record, *, cursor=None):
        """Observe the admitted member; reading media never starts another source.

        RenderHost's existing RTC sources own separately admitted members. Until
        an original model owner explicitly binds this admission, native pixels
        are the only available output. A live unrelated model is not evidence.
        """
        session = await self.public_session(record)
        entry, _ = self.viewer(
            record["stream_id"], session["sessionId"], session["accessToken"]
        )
        binding = session.get("playerBinding", {})
        if (
            entry.get("control_admission") is not record
            or entry["game_id"] != record["binding"]["gameId"]
            or record["binding"]["ownerEpoch"] != self.runtime.epoch
            or binding.get("playerSessionId") != record["member_id"]
            or binding.get("render")
            != dict(sessionId=session["sessionId"], epoch=session["epoch"])
            or session.get("media", {}).get("kind") != "pixels"
        ):
            raise ValueError("The original player media binding has changed")
        room = self.runtime.rooms[entry["game_id"]]
        publication = frozen_publication_identity(
            room["control_record"]["invocation"]["createContext"]["frozenPublication"]
        )
        if publication["publicationId"] != record["binding"]["publicationId"]:
            raise ValueError("The original publication has changed")
        result = dict(
            status="ready",
            kind="sdk-session",
            memberBinding={
                **record["binding"],
                "memberId": record["member_id"],
                "admissionOperationId": record["admission"]["operationId"],
            },
            publication=copy.deepcopy(publication),
            session=session,
            model=copy.deepcopy(
                record.get(
                    "media",
                    dict(
                        status="unavailable", reason="MEMBER_MODEL_SOURCE_UNAVAILABLE"
                    ),
                )
            ),
        )
        owner = getattr(self.runtime, "member_media", None)
        if owner is not None:
            result.update(owner.attachment(record, session, cursor))
        return result


def install_player_runtime(app, token, runtime, *, config=None, client=None):
    config = config or PlayerTransportConfig.from_environment()
    transport = runtime.player_transport = PlayerTransport(runtime, config)

    def allowed(request, *, static=False):
        if config is None:
            raise HTTPException(503, "Public player transport is not configured")
        if not config.admits_origin(request.headers.get("origin")):
            raise HTTPException(403, "The player origin is unavailable")

    @app.get("/internal/player-streams/{stream_id}/{session_id}")
    async def session(
        stream_id: str,
        session_id: str,
        request: Request,
        authorization: str = Header(default=""),
    ):
        require_internal_token(token, authorization)
        allowed(request)
        try:
            entry, _ = transport.viewer(
                stream_id,
                session_id,
                request.headers.get("x-worldkit-viewer-capability", ""),
            )
            config.require_member_origin(request.headers.get("origin"), entry)
            session = await transport.public_session(entry["control_admission"])
            return JSONResponse(
                {**session, "status": "running"},
                headers={"Cache-Control": "private, no-store"},
            )
        except (ValueError, KeyError, httpx.HTTPError):
            raise HTTPException(404, "The player is unavailable") from None

    @app.get(
        "/internal/player-streams/{stream_id}/{session_id}/ui/{ui_hash}/{asset:path}"
    )
    async def ui(
        stream_id: str,
        session_id: str,
        ui_hash: str,
        asset: str,
        request: Request,
        authorization: str = Header(default=""),
    ):
        require_internal_token(token, authorization)
        allowed(request)
        try:
            entry, descriptor = transport.viewer(stream_id, session_id)
            config.require_member_origin(request.headers.get("origin"), entry)
            if (
                ui_hash != descriptor["uiBundleHash"]
                or not re.fullmatch(r"[a-f0-9]{64}", ui_hash)
                or not asset
                or any(part in {"", ".", ".."} for part in asset.split("/"))
                or re.search(r"[\\%?#:\x00-\x1f]", asset)
            ):
                raise ValueError()

            async def read(upstream):
                async with upstream.stream(
                    "GET",
                    descriptor["baseUrl"].rstrip("/") + "/ui/" + ui_hash + "/" + asset,
                    follow_redirects=False,
                ) as response:
                    if response.status_code != 200:
                        raise ValueError()
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > MAX_UI_BYTES:
                            raise ValueError()
                    transport.viewer(stream_id, session_id)
                    return Response(
                        bytes(data),
                        headers={
                            "content-type": response.headers.get(
                                "content-type", "application/octet-stream"
                            ),
                            "Cache-Control": "private, no-store",
                            "X-Content-Type-Options": "nosniff",
                        },
                    )

            if client is not None:
                return await read(client)
            async with httpx.AsyncClient(timeout=15, trust_env=False) as upstream:
                return await read(upstream)
        except (ValueError, KeyError, httpx.HTTPError):
            raise HTTPException(404, "The player resource is unavailable") from None

    @app.websocket("/internal/player-streams/{stream_id}/{session_id}/{channel}")
    async def wire(websocket: WebSocket, stream_id: str, session_id: str, channel: str):
        upstream = None
        try:
            require_internal_token(token, websocket.headers.get("authorization", ""))
            allowed(websocket)
            if channel not in {"control", "pixels", "media"}:
                raise ValueError()
            values, capability = protocols(websocket)
            entry, descriptor = transport.viewer(stream_id, session_id, capability)
            config.require_member_origin(websocket.headers.get("origin"), entry)
            params = query(websocket)
            from websockets.asyncio.client import connect

            upstream = await connect(
                ws_url(
                    descriptor["baseUrl"],
                    f"/v1/sessions/{session_id}/{channel}",
                    params,
                ),
                origin=descriptor["baseUrl"],
                subprotocols=values,
                max_size=(
                    MAX_VIEWER_CONTROL_BYTES if channel == "control" else MAX_WIRE_BYTES
                ),
                max_queue=2,
                compression=None,
                open_timeout=5,
                close_timeout=1,
            )
            transport.viewer(stream_id, session_id, capability)
            await websocket.accept(subprotocol="worldkit.stream.v1")

            def live():
                if runtime.player(stream_id) is not entry:
                    raise ValueError()
                transport.viewer(stream_id, session_id, capability)

            observe = getattr(runtime, "observe_player_control", None)

            async def record(direction, wire):
                await observe(
                    stream_id,
                    websocket.query_params.get("clientId", ""),
                    direction,
                    wire,
                )

            await relay(
                websocket,
                upstream,
                live=live,
                channel=channel,
                observe=record if observe and channel == "control" else None,
            )
        except Exception:
            pass
        finally:
            if upstream:
                await upstream.close()
            try:
                await websocket.close(code=1000)
            except Exception:
                pass


def install_player_gateway(app, proxy, *, config=None):
    config = config or PlayerTransportConfig.from_environment()
    # The outer Gateway CORS policy must use the same validated execution origin
    # as these stricter Session routes, rather than independently parsing env.
    app.state.whitebox_player_transport_config = config

    def allowed(request, *, static=False):
        if config is None:
            raise HTTPException(503, "Public player transport is not configured")
        if (
            not config.admits_origin(request.headers.get("origin"))
            and not (static and request.headers.get("origin") is None)
        ) or request.headers.get("host") != urlsplit(config.public_origin).netloc:
            raise HTTPException(403, "The player origin is unavailable")

    async def route(session_id):
        binding = await proxy.directory.session_route(checked_session_id(session_id))
        if binding is None or binding["owner_id"] not in proxy.owners:
            raise ValueError()
        return binding, proxy.owners[binding["owner_id"]].rstrip("/")

    @app.options("/v1/sessions/{session_id}")
    async def preflight(session_id: str, request: Request):
        allowed(request)
        if (
            request.headers.get("access-control-request-method") != "GET"
            or request.headers.get("access-control-request-headers", "").lower()
            != "authorization"
        ):
            raise HTTPException(403, "The player request is unavailable")
        return Response(
            status_code=204,
            headers={
                "access-control-allow-origin": request.headers.get("origin")
                or config.player_origin,
                "access-control-allow-methods": "GET",
                "access-control-allow-headers": "Authorization",
                "Vary": "Origin",
                "Cache-Control": "no-store",
            },
        )

    @app.get("/v1/sessions/{session_id}")
    async def session(session_id: str, request: Request):
        allowed(request, static=True)
        authorization = request.headers.get("authorization", "")
        if not re.fullmatch(r"Bearer [a-f0-9]{48}", authorization):
            raise HTTPException(403, "The player request is unavailable")
        try:
            binding, base = await route(session_id)
            _, content = await get_bytes(
                proxy.client,
                base + f"/internal/player-streams/{binding['stream_id']}/{session_id}",
                maximum=MAX_VIEWER_CONTROL_BYTES,
                headers={
                    "Authorization": "Bearer " + proxy.token,
                    "Origin": request.headers.get("origin") or config.player_origin,
                    "X-Worldkit-Viewer-Capability": authorization[7:],
                },
            )
            return Response(
                content,
                media_type="application/json",
                headers={
                    "access-control-allow-origin": request.headers.get("origin")
                    or config.player_origin,
                    "Vary": "Origin",
                    "Cache-Control": "private, no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        except (ValueError, KeyError, httpx.HTTPError):
            raise HTTPException(404, "The player is unavailable") from None

    @app.get("/v1/sessions/{session_id}/ui/{ui_hash}/{asset:path}")
    async def ui(session_id: str, ui_hash: str, asset: str, request: Request):
        allowed(request, static=True)
        try:
            binding, base = await route(session_id)
            if (
                not re.fullmatch(r"[a-f0-9]{64}", ui_hash)
                or not asset
                or re.search(r"[\\%?#:\x00-\x1f]", asset)
                or any(p in {"", ".", ".."} for p in asset.split("/"))
            ):
                raise ValueError()
            content_type, content = await get_bytes(
                proxy.client,
                base
                + f"/internal/player-streams/{binding['stream_id']}/{session_id}/ui/{ui_hash}/{asset}",
                maximum=MAX_UI_BYTES,
                headers={
                    "Authorization": "Bearer " + proxy.token,
                    "Origin": request.headers.get("origin") or config.player_origin,
                },
            )
            return Response(
                content,
                headers={
                    "content-type": content_type,
                    "access-control-allow-origin": request.headers.get("origin")
                    or config.player_origin,
                    "Vary": "Origin",
                    "Cache-Control": "private, no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        except (ValueError, KeyError, httpx.HTTPError):
            raise HTTPException(404, "The player resource is unavailable") from None

    @app.websocket("/v1/sessions/{session_id}/{channel}")
    async def wire(websocket: WebSocket, session_id: str, channel: str):
        upstream = None
        try:
            allowed(websocket)
            if channel not in {"control", "pixels", "media"}:
                raise ValueError()
            values, _ = protocols(websocket)
            params = query(websocket)
            binding, base = await route(session_id)
            from websockets.asyncio.client import connect

            upstream = await connect(
                ws_url(
                    base,
                    f"/internal/player-streams/{binding['stream_id']}/{session_id}/{channel}",
                    params,
                ),
                additional_headers={"Authorization": "Bearer " + proxy.token},
                origin=websocket.headers.get("origin"),
                subprotocols=values,
                max_size=(
                    MAX_VIEWER_CONTROL_BYTES if channel == "control" else MAX_WIRE_BYTES
                ),
                max_queue=2,
                compression=None,
                open_timeout=5,
                close_timeout=1,
            )
            await websocket.accept(subprotocol="worldkit.stream.v1")
            await relay(websocket, upstream, channel=channel)
        except Exception:
            pass
        finally:
            if upstream:
                await upstream.close()
            try:
                await websocket.close(code=1000)
            except Exception:
                pass
