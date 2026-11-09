# SPDX-License-Identifier: Apache-2.0
"""Authenticated, bounded observations of the existing Room CLI/cache owner."""

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .maintenance_process import finish_cleanup, terminate_process
from .rooms import require_internal_token
from .rtc_auth import RTCCapabilities

OPERATIONS = {"status", "inspect", "inventory", "prepare", "remove-preview", "remove"}
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class RuntimeMaintenance:
    """Play admission cancels and drains discretionary disk/CPU work first."""

    def __init__(self, runtime):
        self.lock, self.task, self.current = asyncio.Lock(), None, None
        self.is_playing = lambda: bool(
            getattr(runtime, "rtc_preparation", None)
            or getattr(runtime, "rooms", {})
            or getattr(runtime, "players", {})
            or any(lock.locked() for lock in getattr(runtime, "locks", {}).values())
        )

    @asynccontextmanager
    async def operation(self, name):
        task = asyncio.current_task()
        async with self.lock:
            if self.is_playing() or self.task is not None:
                raise HTTPException(
                    409, "Playback has priority over cache maintenance."
                )
            self.task, self.current = task, name
        try:
            yield
        finally:
            if self.task is task:
                self.task = self.current = None

    async def cancel_for_play(self):
        task = self.task
        if task is not None and task is not asyncio.current_task():
            if not task.cancelling():
                task.cancel()
            await finish_cleanup(asyncio.gather(task, return_exceptions=True))


def world_reference(value):
    if not isinstance(value, str) or len(value) > 513:
        raise HTTPException(400, "Choose a valid world.")
    parts = value.split("/")
    if len(parts) != 2 or any(
        p in {"", ".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]{1,256}", p)
        for p in parts
    ):
        raise HTTPException(400, "Choose a valid world.")
    return value


async def request_data(request, *, gateway=False):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > 2048:
            raise HTTPException(413, "The runtime request is too large.")
    try:
        value = json.loads(data)
        allowed = {"operation", "world_id", "page", "expected_owner_epoch"}
        if gateway:
            allowed.add("source_id")
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError()
        if value.get("operation") not in OPERATIONS:
            raise ValueError()
        if (
            type(value.get("page", 0)) is not int
            or not 0 <= value.get("page", 0) <= 10000
        ):
            raise ValueError()
        if (
            value.get("world_id") is None
            and value.get("source_id") is None
            and value["operation"] in {"status", "inventory"}
        ):
            reference = None
        else:
            reference = (str(value.get("source_id")) + "/" if gateway else "") + str(
                value.get("world_id")
            )
            world_reference(reference)
        if "expected_owner_epoch" in value and (
            not isinstance(value["expected_owner_epoch"], str)
            or len(value["expected_owner_epoch"]) > 256
        ):
            raise ValueError()
        if value["operation"] in {"remove", "remove-preview"} and not value.get(
            "expected_owner_epoch"
        ):
            raise ValueError()
        return value, reference
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid runtime operation.") from None


async def disconnected(request, task):
    async def watch():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    worker, watcher = asyncio.create_task(task), asyncio.create_task(watch())
    try:
        done, _ = await asyncio.wait(
            (worker, watcher), return_when=asyncio.FIRST_COMPLETED
        )
        if worker in done:
            return worker.result()
        raise HTTPException(499, "The runtime client disconnected.")
    finally:
        for pending in (worker, watcher):
            if not pending.done() and not pending.cancelling():
                pending.cancel()
        await finish_cleanup(asyncio.gather(worker, watcher, return_exceptions=True))


async def run_runtime_operation(operation, world_id, page=0):
    entry = Path(os.getenv("WHITEBOX_CLI_ENTRYPOINT", ""))
    if not entry.is_absolute() or not entry.is_file():
        raise HTTPException(503, "The CLI is unavailable.")
    process = None
    try:
        async with asyncio.timeout(65 if operation == "inspect" else 20):
            process = await asyncio.create_subprocess_exec(
                os.getenv("WHITEBOX_NODE", "node"),
                "--import",
                "tsx",
                str(Path(__file__).with_name("worldkit_runtime_management.mjs")),
                cwd=entry.parents[3],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            process.stdin.write(
                json.dumps(
                    dict(operation=operation, world_id=world_id, page=page)
                ).encode()
            )
            await process.stdin.drain()
            process.stdin.close()
            output = bytearray()
            while chunk := await process.stdout.read(16384):
                output.extend(chunk)
                if len(output) > MAX_RESPONSE_BYTES:
                    raise ValueError("Runtime response exceeded its limit")
            await process.wait()
            result = json.loads(output)
            if not isinstance(result, dict) or result.get("type") != "world.runtime":
                raise ValueError("Invalid runtime response")
            return result
    except TimeoutError:
        raise HTTPException(504, "Runtime inspection timed out.") from None
    except (ValueError, OSError):
        raise HTTPException(503, "Runtime inspection is unavailable.") from None
    finally:
        await finish_cleanup(terminate_process(process))


def install_room_management(app, token, runtime, prepare_world):
    permits = asyncio.Semaphore(4)
    runtime.maintenance = app.state.runtime_maintenance = RuntimeMaintenance(runtime)

    @app.post("/internal/worlds/runtime")
    async def operate(request: Request, authorization: str = Header(default="")):
        require_internal_token(token, authorization)
        body, reference = await request_data(request)
        if (
            body.get("expected_owner_epoch") is not None
            and body["expected_owner_epoch"] != runtime.epoch
        ):
            raise HTTPException(409, "The Room owner changed; refresh its state.")
        operation = body["operation"]
        if operation == "prepare":
            return await prepare_world(reference)
        if permits.locked():
            raise HTTPException(429, "Runtime inspection is busy.")

        async def work():
            if operation in {"inspect", "remove-preview", "remove"}:
                async with runtime.maintenance.operation(operation):
                    return await run_runtime_operation(
                        operation, reference, body.get("page", 0)
                    )
            return await run_runtime_operation(
                operation, reference, body.get("page", 0)
            )

        async with permits:
            try:
                result = await disconnected(request, work())
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
                raise HTTPException(
                    409, "Playback cancelled cache maintenance."
                ) from None
        result.setdefault("machine", {}).update(
            ownerId=runtime.owner_id, ownerEpoch=runtime.epoch
        )
        result["maintenance"] = {"operation": runtime.maintenance.current}
        code = result.get("error", {}).get("code")
        status = (
            409
            if code in {"GAME_CACHE_BUSY", "GAME_CACHE_PINNED", "RUNTIME_SERVICE_BUSY"}
            else 503
            if result.get("state") == "unavailable"
            else 200
        )
        return JSONResponse(
            result, status_code=status, headers={"Cache-Control": "no-store"}
        )


def install_gateway_management(app, proxy, secret):
    capabilities = (
        RTCCapabilities(secret, audience="whitebox-runtime")
        if isinstance(secret, str) and len(secret.encode()) >= 32
        else None
    )

    @app.post("/backends/minwm/v1/whitebox/runtime")
    @app.post("/v1/whitebox/runtime")
    async def operate(request: Request, authorization: str = Header(default="")):
        body, reference = await request_data(request, gateway=True)
        try:
            if capabilities is None or not authorization.startswith("Bearer "):
                raise ValueError()
            capabilities.verify(
                authorization[7:],
                origin=request.headers.get("origin"),
                world=body,
                operation=body["operation"],
                scope="machine" if reference is None else "world",
            )
        except ValueError:
            raise HTTPException(
                403, "Runtime authorization could not be verified."
            ) from None
        if len(proxy.owners) != 1:
            raise HTTPException(503, "A unique Room owner is required.")
        forwarded = dict(
            operation=body["operation"], world_id=reference, page=body.get("page", 0)
        )
        if "expected_owner_epoch" in body:
            forwarded["expected_owner_epoch"] = body["expected_owner_epoch"]
        try:
            upstream = await disconnected(
                request,
                proxy.client.send(
                    proxy.client.build_request(
                        "POST",
                        next(iter(proxy.owners.values())).rstrip("/")
                        + "/internal/worlds/runtime",
                        json=forwarded,
                        headers={"Authorization": "Bearer " + proxy.token},
                        timeout=320,
                    ),
                    stream=True,
                ),
            )
        except httpx.HTTPError:
            raise HTTPException(503, "The Room is temporarily unavailable.") from None

        async def stream():
            total = 0
            try:
                async for chunk in upstream.aiter_bytes(8192):
                    total += len(chunk)
                    if total > MAX_RESPONSE_BYTES:
                        raise ValueError("Runtime response exceeded its limit")
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            stream(),
            status_code=upstream.status_code,
            media_type="application/x-ndjson"
            if body["operation"] == "prepare"
            else "application/json",
            headers={"Cache-Control": "no-store"},
            background=BackgroundTask(upstream.aclose),
        )
