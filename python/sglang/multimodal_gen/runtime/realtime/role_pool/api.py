# SPDX-License-Identifier: Apache-2.0
"""Internal Coordinator API. Registration and Room controls use separate credentials."""

import hmac
import os

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .contracts import Credentials, PoolError
from .scheduler import GroupController
from .store import RedisRecords, RolePoolStore


def install_pool_routes(
    app, controller, *, role_token, room_token, operator_token=None
):
    tokens = [role_token, room_token] + ([operator_token] if operator_token else [])
    if any(not isinstance(value, str) or len(value) < 32 for value in tokens) or len(
        set(tokens)
    ) != len(tokens):
        raise ValueError("Separate role/Room credentials are required")
    store = controller.store
    app.state.role_pool_controller = controller

    async def payload(request, *, role=False, operator=False):
        if operator and (not operator_token or len(operator_token) < 32):
            raise HTTPException(503, "Operator controls are not configured")
        auth = request.headers.get("authorization", "")
        expected = "Bearer " + (
            (operator_token or "") if operator else role_token if role else room_token
        )
        if not hmac.compare_digest(auth, expected):
            raise HTTPException(403, "Role pool access denied")
        data = await request.body()
        if len(data) > 2 * 1024 * 1024:
            raise HTTPException(413, "Role pool request is too large")
        import json

        try:
            result = json.loads(data)
            if not isinstance(result, dict):
                raise ValueError()
            return result
        except ValueError:
            raise HTTPException(422, "Invalid role pool request") from None

    async def handle(request, operation, *, role=False, operator=False):
        body = await payload(request, role=role, operator=operator)
        try:
            return await operation(body)
        except PoolError as exc:
            return JSONResponse({"code": exc.code}, status_code=exc.status)
        except (KeyError, TypeError, ValueError):
            return JSONResponse({"code": "INVALID_ROLE_POOL_REQUEST"}, status_code=422)

    @app.post("/v1/role-pool/register")
    async def register(request: Request):
        return await handle(request, store.register, role=True)

    @app.post("/v1/role-pool/heartbeat")
    async def heartbeat(request: Request):
        return await handle(request, lambda b: store.heartbeat(**b), role=True)

    @app.post("/v1/role-pool/operator-check")
    async def operator_check(request: Request):
        await payload(request, operator=True)
        return {"pool_id": store.pool_id, "control_epoch": store.epoch}

    @app.post("/v1/role-pool/drain")
    async def drain(request: Request):
        return await handle(
            request, lambda b: controller.drain_process(**b), operator=True
        )

    @app.post("/v1/role-pool/process-status")
    async def process_status(request: Request):
        return await handle(request, lambda b: store.process_status(**b), operator=True)

    @app.post("/v1/role-pool/confirm-stopped")
    async def confirm_stopped(request: Request):
        return await handle(
            request, lambda b: store.confirm_stopped(**b), operator=True
        )

    @app.post("/v1/role-pool/retire-process")
    async def retire(request: Request):
        async def stopped(b):
            from .contracts import identifier

            identifier(b["stop_receipt"])
            return await store.retire_process(**b)

        return await handle(request, stopped, operator=True)

    @app.post("/v1/role-pool/capacity")
    async def capacity(request: Request):
        return await handle(request, lambda b: store.capacity(**b))

    @app.post("/v1/role-pool/admit")
    async def admit(request: Request):
        return await handle(request, lambda b: controller.admit(**b))

    @app.post("/v1/role-pool/get")
    async def get(request: Request):
        return await handle(request, lambda b: store.get(b["admission_id"]))

    @app.post("/v1/role-pool/renew")
    async def renew(request: Request):
        return await handle(request, lambda b: controller.renew(**b))

    @app.post("/v1/role-pool/business")
    async def business(request: Request):
        return await handle(request, lambda b: controller.business(**b))

    @app.post("/v1/role-pool/cancel")
    async def cancel(request: Request):
        async def stop(b):
            a = await store.cancel_intent(b["admission_id"], b["user_id"])
            if a is not None:
                return await controller.close(b["admission_id"], a["allocation_id"])
            return {"state": "cancelled"}

        return await handle(request, stop)

    @app.post("/v1/role-pool/close")
    async def close(request: Request):
        return await handle(request, lambda b: controller.close(**b))

    @app.post("/v1/role-pool/persistence")
    async def persistence(request: Request):
        async def inspect(b):
            a = await store.get(b["admission_id"])
            now = controller.clock()
            if (
                a["allocation_id"] != b["allocation_id"]
                or now >= a["player_expires_at"]
            ):
                raise PoolError("PLAYER_SESSION_ENDED")
            if (
                a["state"] == "bound"
                or a["state"] == "closing"
                and a["intent"].get("operation") == "prefill"
            ) and now < a["executor_until"]:
                await store.admissions.verify(a, store.pool_id)
            elif a["state"] == "released" and a["intent"].get("operation") == "prefill":
                # Observe the original writeback after verified compute release.
                # This local signing deadline does not renew or mutate a lease.
                a = {**a, "executor_until": min(now + 30, a["player_expires_at"])}
            else:
                raise PoolError("PLAYER_SESSION_ENDED")
            return await controller.rpc(
                a,
                "dit",
                "persistence",
                extra={"descriptor": b["descriptor"]},
                timeout=6,
            )

        return await handle(request, inspect)

    @app.post("/v1/role-pool/child-status")
    async def child_status(request: Request):
        async def observe(b):
            a = await store.get(b["admission_id"])
            if (
                a["allocation_id"] != b["allocation_id"]
                or a["state"] != "bound"
                or controller.clock()
                >= min(a["executor_until"], a["player_expires_at"])
            ):
                raise PoolError("PLAYER_SESSION_ENDED")
            await store.admissions.verify(a, store.pool_id)
            return await controller.rpc(
                a, "encoder", "child_status", extra={"session_id": b["session_id"]}
            )

        return await handle(request, observe)


def install_from_environment(app):
    config_path = os.getenv("SGLANG_ROLE_POOL_COORDINATOR_CONFIG")
    if not config_path:
        return None
    import json
    from pathlib import Path

    import redis.asyncio as redis

    cfg = json.loads(Path(config_path).read_text())
    client = redis.from_url(
        cfg["redis_url"],
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
    )
    gate_client = redis.from_url(
        cfg.get("admission_redis_url", cfg["redis_url"]),
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
    )
    store = RolePoolStore(
        RedisRecords(client, expected_run_id=cfg["redis_run_id"]),
        cfg["pool_id"],
        cfg["control_epoch"],
        admission_records=RedisRecords(
            gate_client,
            expected_run_id=cfg.get("admission_redis_run_id", cfg["redis_run_id"]),
        ),
    )
    controller = GroupController(
        store, Credentials(os.environ[cfg["signing_secret_env"]], cfg["control_epoch"])
    )
    from .contracts import identifier

    routes = cfg["capacity_routes"]
    if not isinstance(routes, dict) or cfg["default_capacity_route"] not in routes:
        raise ValueError("Role pool default capacity route is not configured")
    for route in routes.values():
        if not isinstance(route, dict) or set(route) != {
            "profile",
            "compatibility",
            "network_domain",
        }:
            raise ValueError("Invalid role pool capacity route")
        for value in route.values():
            identifier(value)
    install_pool_routes(
        app,
        controller,
        role_token=os.environ[cfg["role_token_env"]],
        room_token=os.environ[cfg["room_token_env"]],
        operator_token=os.environ[cfg["operator_token_env"]],
    )
    # No bootstrap from possibly rolled-back data. Deployment must explicitly
    # initialize the pinned authority with RolePoolStore.initialize beforehand.
    controller.owned_records_clients = [client, gate_client]
    controller.capacity_routes = cfg["capacity_routes"]
    controller.default_capacity_route = cfg["default_capacity_route"]
    return controller
