# SPDX-License-Identifier: Apache-2.0
"""One bounded resource-only preparation, independent of all play/model leases."""

import asyncio
import json
import os
from contextlib import nullcontext
from pathlib import Path

from fastapi import Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from .cli_config import game_cache_directory, resource_transport
from .maintenance_process import finish_cleanup, terminate_process
from .rooms import require_internal_token


def install_prefetch(app, token):
    busy = asyncio.Lock()

    def require_preparation():
        if os.getenv("WHITEBOX_PREFETCH_ENABLED") != "1" or resource_transport(
            os.environ
        ) not in {"auto", "cdn", "oss-public", "oss-internal"}:
            raise HTTPException(404)

    @app.post("/internal/worlds/prepare")
    async def prepare(request: Request, authorization: str = Header(default="")):
        require_internal_token(token, authorization)
        require_preparation()
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 1024:
                raise HTTPException(413, "The preparation request is too large.")
        try:
            body = json.loads(data)
            world = body["world_id"]
            if not isinstance(world, str) or not 1 <= len(world) <= 256:
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise HTTPException(400, "A world identifier is required.")
        return await prepare_world(world)

    async def prepare_world(world):
        require_preparation()
        entry = Path(os.getenv("WHITEBOX_CLI_ENTRYPOINT", ""))
        cache = game_cache_directory(os.environ)
        if (
            not entry.is_absolute()
            or not entry.is_file()
            or (cache is not None and not Path(cache).is_absolute())
        ):
            raise HTTPException(503, "World preparation is unavailable.")

        maintenance = getattr(app.state, "runtime_maintenance", None)
        if maintenance is not None:
            async with maintenance.lock:
                if maintenance.is_playing() or maintenance.task is not None:
                    raise HTTPException(
                        409, "Playback has priority over cache maintenance."
                    )

        async def stream():
            if busy.locked():
                yield b'{"type":"world.preparation","state":"unavailable","code":"PREPARATION_BUSY"}\n'
                return
            await busy.acquire()
            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    os.getenv("WHITEBOX_NODE", "node"),
                    "--import",
                    "tsx",
                    str(Path(__file__).with_name("worldkit_cdn_prepare.mjs")),
                    cwd=entry.parents[3],
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    limit=16384,
                    start_new_session=True,
                )
                process.stdin.write(json.dumps({"worldId": world}).encode())
                await process.stdin.drain()
                process.stdin.close()
                async with asyncio.timeout(310):
                    async for line in process.stdout:
                        if len(line) > 8192:
                            raise ValueError("Preparation progress is too large")
                        value = json.loads(line)
                        if value.get("type") != "world.preparation":
                            raise ValueError("Invalid preparation progress")
                        yield line
                    await process.wait()
            finally:
                try:
                    await finish_cleanup(terminate_process(process))
                finally:
                    busy.release()

        async def guarded_stream():
            maintenance = getattr(app.state, "runtime_maintenance", None)
            try:
                async with (
                    maintenance.operation("prepare") if maintenance else nullcontext()
                ):
                    iterator = stream()
                    try:
                        async for chunk in iterator:
                            yield chunk
                    finally:
                        await finish_cleanup(iterator.aclose())
            except HTTPException as error:
                if error.status_code != 409:
                    raise
                yield b'{"type":"world.preparation","state":"unavailable","code":"PREPARATION_BUSY"}\n'

        return StreamingResponse(guarded_stream(), media_type="application/x-ndjson")

    return prepare_world
