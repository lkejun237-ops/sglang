# SPDX-License-Identifier: Apache-2.0
"""Optional manual owner enrollment; no process creation or automatic scale-in."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import Header, HTTPException

from .room_registry import origin
from .rooms import require_internal_token


def install_registry_routes(app, registry, token, admin_token):
    if not admin_token or admin_token == token:
        raise ValueError("A separate Room operator token is required")

    @app.post("/v1/whitebox/room-owners/enroll")
    async def enroll(body: dict, authorization: str = Header(default="")):
        require_internal_token(admin_token, authorization)
        try:
            if set(body) - {"owner_url", "initialize"}:
                raise ValueError()
            address = origin(body["owner_url"])
            async with httpx.AsyncClient(
                timeout=5, trust_env=False, follow_redirects=False
            ) as client:
                response = await client.get(
                    address + "/internal/room-owner",
                    headers={"Authorization": "Bearer " + token},
                )
                response.raise_for_status()
                if len(response.content) > 16384:
                    raise ValueError()
                descriptor = response.json()
            if descriptor["owner_url"] != address:
                raise ValueError()
            return await registry.enroll(
                descriptor, initialize=body.get("initialize") is True
            )
        except (ValueError, KeyError, TypeError, httpx.HTTPError):
            raise HTTPException(409, "The empty Room could not be enrolled") from None

    @app.post("/v1/whitebox/room-owners/heartbeat")
    async def heartbeat(body: dict, authorization: str = Header(default="")):
        require_internal_token(token, authorization)
        try:
            return await registry.heartbeat(body["owner_id"], body["owner_epoch"])
        except (ValueError, KeyError, TypeError):
            raise HTTPException(409, "The Room registration is unavailable") from None

    @app.post("/v1/whitebox/room-owners/inspect")
    async def inspect(body: dict, authorization: str = Header(default="")):
        require_internal_token(admin_token, authorization)
        try:
            return await registry.inspect(body.get("owner_id"), body.get("owner_epoch"))
        except (ValueError, KeyError, TypeError):
            raise HTTPException(409, "The Room registration is unavailable") from None

    @app.post("/v1/whitebox/room-owners/drain")
    async def drain(body: dict, authorization: str = Header(default="")):
        require_internal_token(admin_token, authorization)
        try:
            return await registry.drain(body["owner_id"], body["owner_epoch"])
        except (ValueError, KeyError, TypeError):
            raise HTTPException(409, "The original Room could not be drained") from None

    @app.post("/v1/whitebox/room-owners/remove")
    async def remove(body: dict, authorization: str = Header(default="")):
        require_internal_token(admin_token, authorization)
        try:
            row = await registry.inspect(body["owner_id"], body["owner_epoch"])
            if row["state"] == "removed":
                return row
            async with httpx.AsyncClient(
                timeout=5, trust_env=False, follow_redirects=False
            ) as client:
                response = await client.get(
                    row["owner_url"] + "/internal/room-owner",
                    headers={"Authorization": "Bearer " + token},
                )
                response.raise_for_status()
                if len(response.content) > 16384:
                    raise ValueError()
                actual = response.json()
            if (
                actual["owner_id"] != row["owner_id"]
                or actual["owner_epoch"] != row["owner_epoch"]
                or actual["empty"] is not True
            ):
                raise ValueError()
            return await registry.remove(row["owner_id"], row["owner_epoch"])
        except (ValueError, KeyError, TypeError, httpx.HTTPError):
            raise HTTPException(
                409, "The original Room is not confirmed empty"
            ) from None


def install_owner_registration(app, runtime, token):
    if os.environ.get("WHITEBOX_DYNAMIC_ROOMS") != "1":
        return
    from .native_rtc import configured_rtc_endpoints
    from .room_registry import count

    address = origin(os.environ["WHITEBOX_ROOM_ADVERTISE_URL"])
    compatibility = os.environ["WHITEBOX_ROOM_POOL_COMPATIBILITY"]
    max_views = count(int(os.environ["WHITEBOX_ROOM_PIPELINE_CAPACITY"]))
    endpoints = configured_rtc_endpoints()
    media = dict(
        public_ip=os.environ["WHITEBOX_NATIVE_RTC_PUBLIC_IP"],
        public_ports=[p for _, p, _ in endpoints]
        or [int(os.environ["WHITEBOX_NATIVE_RTC_PUBLIC_PORT"])],
    )

    @app.get("/internal/room-owner")
    async def descriptor(authorization: str = Header(default="")):
        require_internal_token(token, authorization)
        empty = not any(
            (
                runtime.rooms,
                runtime.retiring,
                runtime.players,
                runtime.retiring_players,
                runtime.pending_startups,
                any(not t.done() for t in runtime.player_closures.values()),
                any(not t.done() for t in getattr(runtime.archives, "tasks", [])),
                getattr(runtime, "render_host", None) and runtime.render_host.sessions,
                # The existing maintenance predicate also sees legacy pipeline
                # sessions while their SDK source is starting or retiring.
                getattr(runtime, "maintenance", None)
                and runtime.maintenance.is_playing(),
            )
        )
        return dict(
            owner_id=runtime.owner_id,
            owner_epoch=runtime.epoch,
            owner_url=address,
            compatibility=compatibility,
            max_rooms=runtime.max_rooms,
            max_views=max_views,
            media=media,
            empty=empty,
        )

    async def renew():
        failed = False
        while True:
            try:
                directory = runtime.directory
                response = await directory.client.post(
                    directory.url + "/v1/whitebox/room-owners/heartbeat",
                    json=dict(owner_id=runtime.owner_id, owner_epoch=runtime.epoch),
                    headers={"Authorization": "Bearer " + token},
                )
                response.raise_for_status()
                failed = False
            except (httpx.HTTPError, ValueError):
                if not failed:
                    logging.getLogger(__name__).warning(
                        "Room owner is not registered or its heartbeat failed"
                    )
                failed = True
            await asyncio.sleep(5)

    previous = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with previous(application):
            task = asyncio.create_task(renew(), name="room-owner-heartbeat")
            try:
                yield
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    app.router.lifespan_context = lifespan
