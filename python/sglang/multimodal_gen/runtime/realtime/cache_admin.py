# SPDX-License-Identifier: Apache-2.0
"""CPU-only world L3 deletion, scoped by the server's cache configuration."""

import asyncio
import hmac
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from .cache_archive import CacheArchive, world_token
from .cache_tiers import CacheTiers

router = APIRouter(tags=["Zing cache"])


class WorldSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    world_id: str = Field(min_length=1, max_length=256)


def authorize(request):
    expected = os.environ.get("SGLANG_CACHE_ADMIN_TOKEN", "")
    supplied = request.headers.get("authorization", "")
    if len(expected.encode()) < 32:
        raise HTTPException(503, "Cache management is not configured")
    if not hmac.compare_digest(supplied.encode(), ("Bearer " + expected).encode()):
        raise HTTPException(403, "Cache management authorization could not be verified")


class WorldCacheDeletion:
    """Bounded jobs retain honest pending/failed results; never stop model work."""

    def __init__(self, policy, *, archive_factory=CacheArchive):
        self.policy, self.archive_factory = policy, archive_factory
        self.jobs, self.tasks = {}, {}

    def prune(self):
        self.jobs = {
            key: value
            for key, value in self.jobs.items()
            if key in self.tasks or time.time() - value["updated_at"] < 3600
        }

    def submit(self, world):
        world_token(world)
        self.prune()
        for key, record in self.jobs.items():
            if record["world_id"] == world and key in self.tasks:
                return record
        if len(self.tasks) >= 2 or len(self.jobs) >= 32:
            raise HTTPException(429, "Cache deletion capacity is unavailable")
        key = uuid4().hex
        record = dict(
            operation_id=key,
            world_id=world,
            tier="l3",
            state="pending",
            updated_at=time.time(),
        )
        self.jobs[key] = record

        def delete():
            archive = self.archive_factory(self.policy, "management")
            try:
                return archive.delete_world(world)
            finally:
                archive.close()

        async def run():
            try:
                result = await asyncio.to_thread(delete)
                record.update(result)
            except Exception:
                record.update(state="failed", code="CACHE_DELETE_FAILED")
            finally:
                record["updated_at"] = time.time()
                self.tasks.pop(key, None)

        self.tasks[key] = asyncio.create_task(run())
        return dict(record)


def manager(request):
    value = getattr(request.app.state, "world_cache_deletion", None)
    if value is None:
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        args = get_global_server_args()
        path = getattr(args, "zing_block0_mooncake_config", None)
        if not path:
            raise HTTPException(503, "Cache management is not configured")
        try:
            config = json.loads(Path(path).read_text())
            # Disabling reads/writes does not disable explicit cleanup. Validate
            # retained region/bucket/prefix, without changing the runtime policy.
            policy = CacheTiers.parse({**config.get("tiers", {}), "l3": True})
            value = WorldCacheDeletion(policy)
        except (OSError, ValueError, TypeError):
            raise HTTPException(
                503, "Cache management configuration is invalid"
            ) from None
        request.app.state.world_cache_deletion = value
    return value


@router.post("/delete-world", status_code=202)
async def delete_world(selection: WorldSelection, request: Request):
    authorize(request)
    try:
        return manager(request).submit(selection.world_id)
    except ValueError:
        raise HTTPException(422, "Invalid cache world identity") from None


@router.get("/operations/{operation_id}")
async def operation_status(operation_id: str, request: Request):
    authorize(request)
    owner = manager(request)
    owner.prune()
    record = owner.jobs.get(operation_id)
    if record is None:
        raise HTTPException(404, "The cache operation was not found")
    return dict(record)
