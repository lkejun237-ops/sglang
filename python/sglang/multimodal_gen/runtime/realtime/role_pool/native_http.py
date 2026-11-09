# SPDX-License-Identifier: Apache-2.0
"""Managed HTTP controls and connection-scoped business authorization."""

import asyncio
import base64
import json
import os

from fastapi import Request
from fastapi.responses import JSONResponse

from .contracts import PoolError, validate_business_message


class ManagedTransport:
    def __init__(self, app, *, owner_app):
        self.app, self.owner_app = app, owner_app

    async def __call__(self, scope, receive, send):
        native = getattr(self.owner_app.state, "worker_connections", None)
        managed = getattr(native, "managed", None)
        path = scope.get("path", "")
        protected = (
            scope["type"] == "websocket"
            or scope["type"] == "http"
            and path.startswith("/v1/zing/cache")
        )
        if (
            not protected
            or managed is None
            and not os.environ.get("SGLANG_ROLE_POOL_WORKER_CONFIG")
        ):
            await self.app(scope, receive, send)
            return
        try:
            if managed is None:
                raise PoolError("MODEL_NOT_READY", 503)
            headers = dict(scope.get("headers", []))
            raw = headers.get(b"x-role-pool-claim", b"")
            if not raw or len(raw) > 16384:
                raise PoolError("INVALID_BUSINESS_CLAIM", 403)
            body = json.loads(base64.urlsafe_b64decode(raw + b"=" * (-len(raw) % 4)))
            token = headers.get(b"authorization", b"").decode().removeprefix("Bearer ")
            claim = managed.business(body, token)
            if (
                claim.get("operation") == "prefill"
                and path != "/v1/realtime_video/prefill"
            ):
                raise PoolError("INVALID_BUSINESS_OPERATION", 403)
            scope["role_pool_claim"] = claim
            scope["role_pool_service"] = managed
            task = asyncio.current_task()
            if scope["type"] == "websocket":
                managed.attach_socket(claim, task)
        except (PoolError, ValueError, TypeError, KeyError):
            if scope["type"] == "websocket":
                await send(
                    {
                        "type": "websocket.close",
                        "code": 1008,
                        "reason": "Model access denied",
                    }
                )
            else:
                await JSONResponse({"code": "MODEL_ACCESS_DENIED"}, status_code=403)(
                    scope, receive, send
                )
            return
        if scope["type"] != "websocket":
            await self.app(scope, receive, send)
            return

        async def checked_receive():
            value = await receive()
            if value.get("type") == "websocket.receive" and value.get("bytes"):
                import msgspec

                message = msgspec.msgpack.decode(value["bytes"])
                if native.service["role"] == "encoder":
                    validate_business_message(message, claim["profile_limits"])
                elif any(
                    message.get(key) != claim[key]
                    for key in ("session_id", "generation_id")
                ):
                    raise PoolError("OUTPUT_ATTACHMENT_MISMATCH")
            return value

        try:
            await self.app(scope, checked_receive, send)
        finally:
            with managed.process.lock:
                managed.sockets.discard(task)
                pending = managed.children.get(claim["session_id"])
                if (
                    native.service["role"] == "encoder"
                    and pending
                    and pending["state"] == "connected"
                ):
                    managed.child_cleaned(claim["session_id"], claim["generation_id"])
                elif native.service["role"] == "decoder":
                    managed.children.pop(claim["session_id"], None)


def install_managed_transport(app, native=None):
    if native is not None:
        app.state.worker_connections = native
    if not getattr(app.state, "role_pool_transport_installed", False):
        app.add_middleware(ManagedTransport, owner_app=app)
        app.state.role_pool_transport_installed = True


def install_managed_routes(app):
    install_managed_transport(app)

    @app.post("/managed/{action}")
    async def managed_control(action: str, request: Request):
        native = getattr(app.state, "worker_connections", None)
        managed = getattr(native, "managed", None)
        if managed is None:
            return JSONResponse({"code": "MANAGED_ROLE_UNAVAILABLE"}, status_code=503)
        raw = await request.body()
        if len(raw) > 256 * 1024:
            return JSONResponse({"code": "CONTROL_TOO_LARGE"}, status_code=413)
        try:
            body = json.loads(raw)
            token = request.headers.get("authorization", "").removeprefix("Bearer ")
            return await managed.command(action, body, token)
        except PoolError as exc:
            return JSONResponse({"code": exc.code}, status_code=exc.status)
        except (ValueError, TypeError, KeyError):
            return JSONResponse({"code": "INVALID_CONTROL_REQUEST"}, status_code=422)
