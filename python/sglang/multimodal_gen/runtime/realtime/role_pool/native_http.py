# SPDX-License-Identifier: Apache-2.0
"""Managed HTTP controls and connection-scoped business authorization."""

import asyncio
import base64
import json
import os
import re

from fastapi import Request
from fastapi.responses import JSONResponse

from .contracts import Credentials, PoolError, validate_business_message


class ManagedTransport:
    def __init__(self, app, *, owner_app):
        self.app, self.owner_app = app, owner_app

    async def __call__(self, scope, receive, send):
        native = getattr(self.owner_app.state, "worker_connections", None)
        managed = getattr(native, "managed", None)
        path = scope.get("path", "")
        # These CPU-only routes enforce their own cache-admin token. Requiring
        # a player business claim here would consume model capacity and reject
        # legitimate administrative requests before that authorization runs.
        cache_management = scope["type"] == "http" and (
            scope.get("method") == "POST"
            and path == "/v1/zing/cache/delete-world"
            or scope.get("method") == "GET"
            and re.fullmatch(r"/v1/zing/cache/operations/[a-f0-9]{32}", path)
        )
        protected = (
            scope["type"] == "websocket"
            or scope["type"] == "http"
            and path.startswith("/v1/zing/cache")
        )
        if (
            not protected
            or cache_management
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

    @app.get("/managed/discovery")
    async def discovery():
        # Non-secret bootstrap only. Full identity still requires a signature.
        native = getattr(app.state, "worker_connections", None)
        managed = getattr(native, "managed", None)
        if managed is None:
            return JSONResponse({"code": "MANAGED_ROLE_UNAVAILABLE"}, status_code=503)
        return {"control_epoch": managed.process.credentials.epoch}

    @app.post("/managed/{action}")
    async def managed_control(action: str, request: Request):
        native = getattr(app.state, "worker_connections", None)
        managed = getattr(native, "managed", None)
        raw = await request.body()
        if len(raw) > 256 * 1024:
            return JSONResponse({"code": "CONTROL_TOO_LARGE"}, status_code=413)
        if action == "attach":
            if native is None:
                return JSONResponse(
                    {"code": "MANAGED_ROLE_UNAVAILABLE"}, status_code=503
                )
            try:
                body = json.loads(raw)
                token = request.headers.get("authorization", "").removeprefix("Bearer ")
                config = body.get("worker_config")
                if (
                    not isinstance(config, dict)
                    or config.get("signing_secret_env") != "ROLE_POOL_SIGNING_KEY"
                    or config.get("role_token_env") != "ROLE_POOL_REGISTRATION_TOKEN"
                    or body.get("control_epoch") != config.get("control_epoch")
                ):
                    raise ValueError("Invalid attach configuration")
                secret = os.environ.get("ROLE_POOL_SIGNING_KEY", "")
                role_token = os.environ.get("ROLE_POOL_REGISTRATION_TOKEN", "")
                if not secret or not role_token:
                    raise PoolError("MANAGED_SECRETS_MISSING", 503)
                Credentials(secret, config["control_epoch"]).verify(
                    token, "attach", body
                )
                from .contracts import identifier, vector
                from .reporting import attach_managed_service_config

                with native.lock:
                    if native.managed is None:
                        if (
                            native.state != "idle"
                            or native.restart_required
                            or not native.ready()
                        ):
                            raise PoolError("WORKER_NOT_IDLE", 409)
                        native.managed = attach_managed_service_config(native, config)
                    else:
                        expected_dep = {
                            k: config[k]
                            for k in (
                                "pool_id",
                                "compatibility",
                                "network_domain",
                                "hardware_ids",
                            )
                        }
                        expected_capacity = vector(config["capacity"])
                        expected_profiles = {
                            identifier(k): vector(v)
                            for k, v in config["profiles"].items()
                        }
                        if (
                            native.managed.process.credentials.epoch
                            != config["control_epoch"]
                            or getattr(native.managed.process, "deployment", None)
                            != expected_dep
                            or native.managed.process.capacity != expected_capacity
                            or native.managed.process.profiles != expected_profiles
                            or native.managed.process.profile_limits
                            != config["profile_limits"]
                        ):
                            raise PoolError("ALREADY_MANAGED", 409)
                        if (
                            native.state != "idle"
                            or native.restart_required
                            or not native.ready()
                            or native.managed.state != "idle"
                            or native.managed.owner is not None
                        ):
                            raise PoolError("WORKER_NOT_IDLE", 409)
                    managed = native.managed
                identity_body = {"control_epoch": config["control_epoch"]}
                identity_token = managed.process.credentials.sign(
                    "identity",
                    identity_body,
                    expires_at=managed.process.clock() + 30,
                )
                return await managed.command("identity", identity_body, identity_token)
            except PoolError as exc:
                return JSONResponse({"code": exc.code}, status_code=exc.status)
            except (ValueError, TypeError, KeyError):
                return JSONResponse(
                    {"code": "INVALID_CONTROL_REQUEST"}, status_code=422
                )
        if managed is None:
            return JSONResponse({"code": "MANAGED_ROLE_UNAVAILABLE"}, status_code=503)
        try:
            body = json.loads(raw)
            token = request.headers.get("authorization", "").removeprefix("Bearer ")
            return await managed.command(action, body, token)
        except PoolError as exc:
            return JSONResponse({"code": exc.code}, status_code=exc.status)
        except (ValueError, TypeError, KeyError):
            return JSONResponse({"code": "INVALID_CONTROL_REQUEST"}, status_code=422)
