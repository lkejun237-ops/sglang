# SPDX-License-Identifier: Apache-2.0

"""HTTP entrypoint for the stateless realtime Session Coordinator."""

from __future__ import annotations

import argparse
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, fields
from types import SimpleNamespace

import uvicorn
from fastapi import FastAPI, HTTPException, Response, status

from sglang.multimodal_gen.runtime.realtime.coordinator import (
    CoordinatorRejected,
    DynamoDBCoordinatorStore,
    HTTPWorkerReservationClient,
    InMemoryCoordinatorStore,
    RealtimeCoordinator,
    RedisCoordinatorStore,
    SessionAssignment,
    WorkerHeartbeat,
    WorkerSlot,
)
from sglang.multimodal_gen.runtime.realtime.critical_path_metrics import (
    prometheus_content_type,
    prometheus_latest,
)
from sglang.multimodal_gen.runtime.utils.realtime_trace import (
    log_realtime_trace,
    normalize_trace_id,
)

logger = logging.getLogger(__name__)
_WORKER_HEARTBEAT_FIELDS = frozenset(field.name for field in fields(WorkerHeartbeat))
_WORKER_SLOT_FIELDS = frozenset(field.name for field in fields(WorkerSlot))


def _worker_slot_from_payload(payload: dict) -> WorkerSlot:
    return WorkerSlot(
        **{key: value for key, value in payload.items() if key in _WORKER_SLOT_FIELDS}
    )


def _worker_heartbeat_from_payload(payload: dict) -> WorkerHeartbeat:
    return WorkerHeartbeat(
        **{
            key: value
            for key, value in payload.items()
            if key in _WORKER_HEARTBEAT_FIELDS
        }
    )


def _assignment(payload: dict) -> SessionAssignment:
    return SessionAssignment(
        user_id=payload["user_id"],
        session_id=payload["session_id"],
        generation_id=payload["generation_id"],
        token=payload["token"],
        expires_at=float(payload["expires_at"]),
        denoiser=_worker_slot_from_payload(payload["denoiser"]),
        vae=_worker_slot_from_payload(payload["vae"]),
    )


def _parse_routing_revision_aliases(values: list[str] | None) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(
                "--routing-revision-alias must use alias=actual-revision"
            )
        alias, revision = value.split("=", 1)
        alias = alias.strip()
        revision = revision.strip()
        if not alias or not revision:
            raise SystemExit(
                "--routing-revision-alias must use alias=actual-revision"
            )
        aliases.setdefault(alias, []).append(revision)
    return aliases


def _raise_http(exc: CoordinatorRejected) -> None:
    status_code = {
        "ATTEMPT_ALREADY_OWNED": status.HTTP_409_CONFLICT,
        "USER_SESSION_LIMIT": status.HTTP_409_CONFLICT,
        "CAPACITY_EXHAUSTED": status.HTTP_429_TOO_MANY_REQUESTS,
        "LEASE_LOST": status.HTTP_409_CONFLICT,
        "WORKER_LOST": status.HTTP_409_CONFLICT,
        "COORDINATOR_STORE_BUSY": status.HTTP_503_SERVICE_UNAVAILABLE,
    }.get(exc.reason, status.HTTP_400_BAD_REQUEST)
    headers = None
    if exc.retry_after_s is not None:
        headers = {"Retry-After": str(max(1, int(exc.retry_after_s)))}
    raise HTTPException(
        status_code=status_code,
        detail={"reason": exc.reason, "retry_after_s": exc.retry_after_s},
        headers=headers,
    ) from exc


def create_app(coordinator: RealtimeCoordinator) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await coordinator.close()

    app = FastAPI(title="SGLang Realtime Coordinator", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return Response(prometheus_latest(), media_type=prometheus_content_type())

    @app.get("/v1/capacity")
    async def capacity():
        return await coordinator.capacity_snapshot()

    @app.post("/v1/workers/heartbeat", status_code=status.HTTP_204_NO_CONTENT)
    async def heartbeat(payload: dict):
        try:
            await coordinator.heartbeat(_worker_heartbeat_from_payload(payload))
        except (CoordinatorRejected, TypeError, KeyError) as exc:
            if isinstance(exc, CoordinatorRejected):
                _raise_http(exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"reason": "INVALID_WORKER_HEARTBEAT"},
            ) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/v1/sessions/admit")
    async def admit(payload: dict):
        request = dict(payload)
        trace_id = normalize_trace_id(
            request.pop("trace_id", None), fallback=str(request.get("session_id") or "")
        )
        trace_session = SimpleNamespace(
            id=request.get("session_id"),
            generation_id=request.get("generation_id"),
            trace_id=trace_id,
            trace_started_at=time.perf_counter(),
        )
        started_at = time.perf_counter()
        try:
            assignment = await coordinator.admit(**request)
        except CoordinatorRejected as exc:
            log_realtime_trace(
                logger,
                trace_session,
                "coordinator.admit_rejected",
                reason=exc.reason,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
            )
            _raise_http(exc)
        log_realtime_trace(
            logger,
            trace_session,
            "coordinator.admit_complete",
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
            denoiser_worker_id=assignment.denoiser.worker_id,
            vae_worker_id=assignment.vae.worker_id,
        )
        return asdict(assignment)

    @app.post("/v1/attempts/claim", status_code=status.HTTP_204_NO_CONTENT)
    async def claim_attempt(payload: dict):
        try:
            await coordinator.claim_attempt(
                attempt_id=str(payload["attempt_id"]),
                owner_id=str(payload["owner_id"]),
                expires_at=float(payload["expires_at"]),
            )
        except (CoordinatorRejected, KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, CoordinatorRejected):
                _raise_http(exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"reason": "INVALID_ATTEMPT_OWNER"},
            ) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/v1/sessions/renew")
    async def renew(payload: dict):
        try:
            assignment = await coordinator.renew(_assignment(payload))
        except (CoordinatorRejected, TypeError, KeyError, ValueError) as exc:
            if isinstance(exc, CoordinatorRejected):
                _raise_http(exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"reason": "INVALID_ASSIGNMENT"},
            ) from exc
        return asdict(assignment)

    @app.delete("/v1/sessions/release", status_code=status.HTTP_204_NO_CONTENT)
    async def release(payload: dict):
        try:
            await coordinator.release(_assignment(payload))
        except (TypeError, KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"reason": "INVALID_ASSIGNMENT"},
            ) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument(
        "--backend", choices=("memory", "dynamodb", "redis"), default="dynamodb"
    )
    parser.add_argument("--table-name")
    parser.add_argument("--region")
    parser.add_argument("--redis-url")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-username")
    parser.add_argument("--redis-password-env", default="REDIS_PASSWORD")
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-ssl", action="store_true")
    parser.add_argument("--redis-key-prefix", default="world-model-realtime")
    parser.add_argument("--redis-state-ttl-s", type=int, default=86400)
    parser.add_argument("--redis-lock-ttl-s", type=float, default=2.0)
    parser.add_argument("--redis-lock-timeout-s", type=float, default=1.0)
    parser.add_argument("--ttl-s", type=float, default=30.0)
    parser.add_argument("--worker-ttl-s", type=float, default=15.0)
    parser.add_argument(
        "--routing-revision-alias",
        action="append",
        help=(
            "Allow a stable routing revision to match an older worker that only "
            "reports a concrete model revision, formatted as alias=actual-revision. "
            "May be repeated."
        ),
    )
    parser.add_argument("--wait-timeout-s", type=float, default=10.0)
    parser.add_argument("--candidate-limit", type=int, default=64)
    parser.add_argument("--denoiser-capacity-limit", type=int)
    parser.add_argument("--vae-capacity-limit", type=int)
    parser.add_argument("--worker-reservation-timeout-s", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    capacity_limits = {
        role: limit
        for role, limit in (
            ("denoiser", args.denoiser_capacity_limit),
            ("vae", args.vae_capacity_limit),
        )
        if limit is not None
    }
    routing_revision_aliases = _parse_routing_revision_aliases(
        args.routing_revision_alias
    )
    if args.backend == "dynamodb":
        if not args.table_name:
            raise SystemExit("--table-name is required for DynamoDB")
        store = DynamoDBCoordinatorStore(
            args.table_name,
            ttl_s=args.ttl_s,
            worker_ttl_s=args.worker_ttl_s,
            routing_revision_aliases=routing_revision_aliases,
            region_name=args.region,
            candidate_limit=args.candidate_limit,
            capacity_limits=capacity_limits,
        )
    elif args.backend == "redis":
        store = RedisCoordinatorStore(
            redis_url=args.redis_url,
            host=args.redis_host,
            port=args.redis_port,
            username=args.redis_username,
            password=os.getenv(args.redis_password_env)
            if args.redis_password_env
            else None,
            password_env=None,
            db=args.redis_db,
            ssl=args.redis_ssl,
            key_prefix=args.redis_key_prefix,
            state_ttl_s=args.redis_state_ttl_s,
            lock_ttl_s=args.redis_lock_ttl_s,
            lock_timeout_s=args.redis_lock_timeout_s,
            ttl_s=args.ttl_s,
            worker_ttl_s=args.worker_ttl_s,
            routing_revision_aliases=routing_revision_aliases,
            candidate_limit=args.candidate_limit,
            capacity_limits=capacity_limits,
        )
    else:
        store = InMemoryCoordinatorStore(
            ttl_s=args.ttl_s,
            worker_ttl_s=args.worker_ttl_s,
            routing_revision_aliases=routing_revision_aliases,
        )
    coordinator = RealtimeCoordinator(
        store,
        wait_timeout_s=args.wait_timeout_s,
        reservation_client=HTTPWorkerReservationClient(
            timeout_s=args.worker_reservation_timeout_s
        ),
    )
    uvicorn.run(create_app(coordinator), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
