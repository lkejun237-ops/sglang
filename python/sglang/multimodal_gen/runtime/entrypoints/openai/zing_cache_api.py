# SPDX-License-Identifier: Apache-2.0
"""Metadata-only management of the configured shared Block0 disk cache."""

import asyncio
import logging
import time
from functools import lru_cache
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.cache_logging import (
    cache_event,
    cache_scope,
    log_context,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
    Block0DiskCache,
)
from sglang.multimodal_gen.runtime.realtime.cache_admin import (
    router as world_cache_router,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.shared_capacity import CapacityUnavailable

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/zing/cache", tags=["Zing cache"])
router.include_router(world_cache_router)
_ID_PATTERN = "^[0-9a-f]{64}$"


@lru_cache(maxsize=1)
def _manager(root):
    return Block0DiskCache(root)


def _call(operation, **kwargs):
    args = get_global_server_args()
    if getattr(args, "zing_block0_cache_backend", "local") == "mooncake":
        raise HTTPException(501, detail={"code": "cache_operation_unsupported"})
    root = getattr(args, "zing_block0_cache_dir", None)
    if not root:
        cache_event(
            "management_request",
            "rejected",
            operation=operation,
            reason="cache_unavailable",
            level=logging.WARNING,
        )
        raise HTTPException(503, detail={"code": "cache_unavailable"})
    try:
        with cache_scope():
            return getattr(_manager(root), operation)(**kwargs)
    except ValueError:
        logger.warning(
            "Block0 cache management configuration is invalid", exc_info=True
        )
        raise HTTPException(503, detail={"code": "cache_unavailable"}) from None
    except (OSError, CapacityUnavailable):
        logger.warning("Block0 cache management storage is unavailable", exc_info=True)
        raise HTTPException(503, detail={"code": "cache_unavailable"}) from None


async def _disk_call(request, operation, **arguments):
    client = getattr(request.app.state, "realtime_execution_client", None)
    if client is not None and hasattr(client, "cache_disk"):
        value = await client.cache_disk(operation, **arguments)
        if value.get("code"):
            code = value["code"]
            status = (
                409
                if code
                in {"cache_busy", "cache_capacity_exceeded", "connection_not_ready"}
                else 422
                if code == "invalid_cache_selection"
                else 501
                if code == "cache_operation_unsupported"
                else 503
            )
            raise HTTPException(status, detail=value)
        return value
    return _call(operation, **arguments)


@router.get("")
async def list_cache(
    request: Request,
    cache_key: str | None = Query(None, pattern=_ID_PATTERN),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    return await _disk_call(
        request, "list_entries", cache_key=cache_key, offset=offset, limit=limit
    )


@router.delete("/entries/{cache_key}")
async def delete_cache_entry(cache_key: str, request: Request):
    if len(cache_key) != 64 or any(c not in "0123456789abcdef" for c in cache_key):
        raise HTTPException(422, detail={"code": "invalid_cache_key"})
    return await _disk_call(request, "delete_entries", keys=[cache_key])


@router.post("/clear")
async def clear_cache(request: Request):
    return await _disk_call(request, "delete_entries")


class CacheSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    game_id: str
    group_id: str | None = None
    cache_key: str | None = Field(default=None, pattern=_ID_PATTERN)


async def _worker_call(
    request, operation, *, game_id=None, group_id=None, cache_key=None
):
    with cache_scope(game_id=game_id, group_id=group_id):
        started = time.perf_counter()
        level = logging.DEBUG if operation == "status" else logging.INFO
        cache_event(
            "management_request",
            "started",
            cache_key=cache_key,
            operation=operation,
            level=level,
        )
        try:
            value = await _worker_call_impl(
                request,
                operation,
                game_id=game_id,
                group_id=group_id,
                cache_key=cache_key,
            )
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            cache_event(
                "management_request",
                "rejected",
                operation=operation,
                cache_key=cache_key,
                started=started,
                reason=detail.get("code", "request_rejected"),
                level=logging.WARNING,
            )
            raise
        except BaseException as exc:
            cache_event(
                "management_request",
                "failed",
                operation=operation,
                cache_key=cache_key,
                started=started,
                reason=type(exc).__name__,
                level=logging.WARNING,
            )
            raise
        cache_event(
            "management_request",
            "completed",
            operation=operation,
            cache_key=cache_key,
            started=started,
            level=level,
        )
        return value


async def _worker_call_impl(
    request, operation, *, game_id=None, group_id=None, cache_key=None
):
    from sglang.multimodal_gen.runtime.entrypoints.control_requests import ZingCacheReq
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
        realtime_video_api as api,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.cache_index import (
        validate_labels,
    )
    from sglang.multimodal_gen.runtime.realtime.admission import AdmissionRejected

    try:
        validate_labels(game_id, group_id)
    except ValueError:
        raise HTTPException(422, detail={"code": "invalid_cache_selection"}) from None
    args = get_global_server_args()
    if args.dp_size != 1:
        raise HTTPException(409, detail={"code": "cache_requires_dp1"})
    controller = getattr(
        request.app.state, "realtime_admission_controller", None
    ) or api._get_admission_controller(args)
    lease = renewal = task = None
    try:
        if operation != "status":
            session = GenerateSession(session_id=uuid4().hex)
            lease = await controller.admit(
                api._resolve_realtime_user_id(request),
                session.id,
                session.generation_id,
                wait_for_capacity=False,
            )
            renewal = asyncio.create_task(
                api._session_watchdog(
                    session,
                    controller,
                    lease,
                    idle_timeout_s=0,
                    max_lifetime_s=0,
                    lease_ttl_s=args.realtime_session_lease_ttl_s,
                )
            )
        task = asyncio.create_task(
            (
                getattr(request.app.state, "realtime_execution_client", None)
                or api.async_scheduler_client
            ).forward(
                ZingCacheReq(
                    operation, game_id, group_id, cache_key, log_context=log_context()
                )
            )
        )
        # A cancelled HTTP wait must retain admission until file IO has finished.
        result = await asyncio.shield(task)
        if result.error:
            raise HTTPException(503, detail={"code": "cache_unavailable"})
        value = result.output
        code = value.get("code")
        if code:
            status = (
                409
                if code
                in {"cache_busy", "cache_capacity_exceeded", "connection_not_ready"}
                else 422
                if code == "invalid_cache_selection"
                else 503
            )
            raise HTTPException(status, detail=value)
        return value
    except AdmissionRejected:
        raise HTTPException(409, detail={"code": "cache_busy"}) from None
    finally:

        async def cleanup():
            try:
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)
            finally:
                if renewal is not None:
                    renewal.cancel()
                    await api._await_realtime_task(renewal)
                if lease is not None:
                    await controller.release(lease)

        await api._drain_realtime_cleanup(cleanup())


@router.post("/preload")
async def preload_cache(selection: CacheSelection, request: Request):
    return await _worker_call(request, "preload", **selection.model_dump())


@router.post("/unload")
async def unload_cache(selection: CacheSelection, request: Request):
    return await _worker_call(request, "unload", **selection.model_dump())


@router.get("/status")
async def cache_status(
    request: Request,
    game_id: str | None = None,
    group_id: str | None = None,
    cache_key: str | None = Query(None, pattern=_ID_PATTERN),
):
    return await _worker_call(
        request, "status", game_id=game_id, group_id=group_id, cache_key=cache_key
    )
