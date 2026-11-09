# SPDX-License-Identifier: Apache-2.0
"""Gateway adapter for the owning room Pod; one instance per admitted H3 session."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from dataclasses import asdict

import httpx
import msgspec

from .contracts import EngineFrame
from .metrics import observe_duration, stage
from .rooms import HTTPRoomDirectory, checked_id, new_id


def observe_remote(response, trace_id):
    value = response.headers.get("X-Whitebox-Stages", "[]")
    if len(value) > 8192:
        raise ValueError("The room timing report is too large")
    for span in json.loads(value):
        # Re-emit in the gateway process so the existing per-session report sees
        # remote work. Prefix distinguishes it from local/in-process metrics.
        observe_duration(
            "remote_" + span["stage"],
            span["duration_ms"],
            trace_id,
            result=span["result"],
        )


class RemoteRenderer:
    def __init__(self, engine):
        self.engine, self.metadata = engine, engine.identity["ui"]

    async def render(self, rgb, frame, spec):
        return await self.engine.rpc("render", {"index": frame.index, "rgb": rgb})

    async def finish(self):
        await self.engine.rpc("render_finish")

    async def close(self):
        pass  # The engine owns the connection and player lifecycle.


class RemoteEngine:
    def __init__(self, client, url, token, identity, *, owned_client=True):
        self.client, self.url, self.token = client, url, token
        self.identity, self.owned_client = identity, owned_client
        self.metadata = {**identity["engine"], "transport": "room_runtime"}
        self.renderer, self.closed = RemoteRenderer(self), False

    async def rpc(self, operation, body=None, *, timeout=None):
        with stage("room_rpc_" + operation, self.identity["trace_id"]):
            response = await self.client.post(
                f"{self.url}/internal/players/{self.identity['stream_id']}/{operation}",
                content=msgspec.msgpack.encode(body or {}),
                **({"timeout": timeout} if timeout is not None else {}),
                headers={
                    "Authorization": "Bearer " + self.token,
                    "Content-Type": "application/msgpack",
                },
            )
            response.raise_for_status()
        observe_remote(response, self.identity["trace_id"])
        from .project_audio import AUDIO_PREFIX

        asset_path = (body or {}).get("path")
        sealed_audio = (
            operation == "ui_asset"
            and isinstance(asset_path, str)
            and asset_path.startswith(AUDIO_PREFIX)
        )
        if len(response.content) > (33 if sealed_audio else 16) * 1024 * 1024:
            raise ValueError("The room response is too large")
        return msgspec.msgpack.decode(response.content)

    async def frames(self):
        while not self.closed:
            value = await self.rpc("frame")
            if value is None:
                break
            yield EngineFrame(**value)

    async def ui_asset(self, path):
        return await self.rpc("ui_asset", {"path": path})

    async def note(self, kind, payload):
        return await self.rpc("event", {"kind": kind, "payload": payload})

    async def capture_control(self, action, request_id):
        return await self.rpc(
            "capture_control", dict(action=action, request_id=request_id)
        )

    async def area_status(self):
        return await self.rpc("area_status")

    async def area_frame(self, header):
        return await self.rpc("area_frame", header)

    async def area_presented(self, body):
        return await self.rpc("area_presented", body)

    async def send_input(self, sequence, payload):
        await self.note("input", {"sequence": sequence, "input": payload})

    async def touch(self):
        # Model backpressure can pause both frame reads and renders. Browser
        # liveness must therefore renew the room independently of media traffic.
        # Only liveness is idempotent. Never replay frame reads, inputs or joins:
        # those can have advanced the authoritative world before a response failed.
        for attempt in range(3):
            try:
                await self.rpc("touch", timeout=httpx.Timeout(2.0))
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in (502, 503, 504) or attempt == 2:
                    raise
            except httpx.TransportError:
                if attempt == 2:
                    raise
            await asyncio.sleep(0.1 * (attempt + 1))

    async def finish(self):
        await self.note("finish", {})

    async def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            await self.rpc("leave")
        finally:
            if self.owned_client:
                await self.client.aclose()


class RoomEngineFactory:
    def __init__(self, directory_url=None, token=None, *, client=None, directory=None):
        self.directory_url = directory_url
        self.token, self.client, self.directory = token, client, directory

    async def open_session(self, *, world_id, spec, trace_id, context):
        client = self.client or httpx.AsyncClient(
            timeout=httpx.Timeout(120, connect=5),
            limits=httpx.Limits(max_connections=4),
        )
        directory = self.directory or HTTPRoomDirectory(
            client, self.directory_url, self.token
        )
        game_id = checked_id(context["game_id"]) if context.get("game_id") else new_id()
        body = route = None
        try:
            route = await directory.resolve(
                game_id, world_id, create=not context.get("game_id")
            )
            body = dict(
                **(
                    {
                        "startup_prepare": True,
                        "startup_expires_at": time.time()
                        + min(
                            60,
                            max(0, float(context.get("startup_timeout_seconds", 60))),
                        ),
                    }
                    if context.get("startup_prepare") is True
                    else {}
                ),
                game_id=game_id,
                world_id=world_id,
                trace_id=trace_id,
                spec=asdict(spec),
                player_id=new_id(),
                stream_id=new_id(),
                session_id=context["session_id"],
                user_id=context["user_id"],
                **(
                    {"source_options": copy.deepcopy(context["source_options"])}
                    if context.get("source_options")
                    else {}
                ),
                **(
                    {"cli_fps": context["cli_fps"]}
                    if context.get("cli_fps") is not None
                    else {}
                ),
            )
            response = await client.post(
                route["owner_url"] + "/internal/players/join",
                json=body,
                headers={"Authorization": "Bearer " + self.token},
                timeout=httpx.Timeout(
                    float(os.environ.get("WHITEBOX_STARTUP_SECONDS", "90")) + 10,
                    connect=5,
                ),
            )
            response.raise_for_status()
            observe_remote(response, trace_id)
            identity = response.json()
            identity["trace_id"] = trace_id
            if (
                any(
                    identity.get(key) != body[key]
                    for key in ("game_id", "stream_id", "player_id", "session_id")
                )
                or identity.get("owner_id") != route["owner_id"]
            ):
                raise ValueError("The room returned another player's identity")
            checked_id(identity["owner_epoch"])
            if route.get("owner_epoch") not in {None, identity["owner_epoch"]}:
                raise ValueError("The game owner has changed")
            return RemoteEngine(
                client,
                route["owner_url"],
                self.token,
                identity,
                owned_client=self.client is None,
            )
        except BaseException:
            if (
                context.get("startup_prepare") is True
                and body is not None
                and route is not None
            ):
                # Cancel even if join's HTTP response was lost. The owner keeps
                # a bounded tombstone so a delayed join cannot appear afterward.
                cleanup = asyncio.create_task(
                    client.post(
                        route["owner_url"]
                        + "/internal/players/"
                        + body["stream_id"]
                        + "/startup_cancel",
                        content=msgspec.msgpack.encode({}),
                        timeout=60,
                        headers={
                            "Authorization": "Bearer " + self.token,
                            "Content-Type": "application/msgpack",
                        },
                    )
                )
                try:
                    response = await asyncio.shield(cleanup)
                    response.raise_for_status()
                    if (
                        msgspec.msgpack.decode(response.content).get("cancelled")
                        is not True
                    ):
                        raise ValueError("Startup cleanup was not confirmed")
                except BaseException:
                    cleanup.cancel()
                    await asyncio.gather(cleanup, return_exceptions=True)
                    if self.client is None:
                        await client.aclose()
                    from .render_contract import RenderError

                    raise RenderError("RELEASE_UNKNOWN") from None
            if self.client is None:
                await client.aclose()
            raise


def configured_factory():
    return RoomEngineFactory(
        os.environ["WHITEBOX_ROOM_COORDINATOR_URL"],
        os.environ["WHITEBOX_INTERNAL_TOKEN"],
    )


class EnvironmentRoomFactory:
    async def open_session(self, **kwargs):
        return await configured_factory().open_session(**kwargs)


room_engine_factory = EnvironmentRoomFactory()


def bound_room_renderer(**kwargs):
    raise RuntimeError("Room renderers must be bound to an admitted player's engine")
