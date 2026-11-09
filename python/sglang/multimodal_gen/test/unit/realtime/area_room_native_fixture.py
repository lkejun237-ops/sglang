"""Loopback-only native acceptance transport around the real Room owners.

The CLI/World/pixels/UI are genuine. Only the model worker and remote transport
are replaced by existing deterministic CPU fixtures; this launches no model/GPU.
"""

import argparse
import asyncio
import json
import logging
import mimetypes
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import msgspec
import uvicorn
from fastapi import FastAPI, Header, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from test_whitebox_render_lifecycle import StrictLeases

from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
    AsyncArchiveManager,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_room
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import RenderHost
from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
    new_id,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spool", type=Path, required=True)
    args = parser.parse_args()
    args.spool.mkdir(parents=True, exist_ok=True)
    spec = VideoSpec(160, 96)
    archives = AsyncArchiveManager(recorder_slots=3)
    runtime = RoomRuntime(
        directory=RoomDirectory(MemoryRoomStore(), {"native": "http://native"}),
        owner_id="native",
        driver_factory=cli_room,
        spool=args.spool,
        spec=spec,
        archives=archives,
    )
    config = WhiteboxConfig(
        engine_factory=None,
        ui_factory=None,
        spool=args.spool,
        spec=spec,
        room_owned=True,
        capture_clock=True,
        startup_seconds=45,
        play_seconds=300,
        drain_seconds=5,
    )
    host = RenderHost(
        runtime,
        config,
        archives,
        Reports(args.spool, archives),
        None,
        None,
        grace_seconds=5,
    )
    fault = {"drop": False, "lost_request": None, "lost_replays": 0}
    leases = StrictLeases(spec, "clip-barrier-v1")
    host.make_segment_leases = lambda *_: leases

    @asynccontextmanager
    async def lifetime(_app):
        archives.start(args.spool)
        try:
            yield
        finally:
            await host.close()
            await runtime.close()

    app = FastAPI(lifespan=lifetime)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_headers=["Authorization"],
        allow_methods=["GET"],
    )

    @app.websocket("/v1/realtime_video/generate")
    async def play(ws: WebSocket):
        await ws.accept()
        owner = sub = None
        sender = None
        try:
            first = msgspec.msgpack.decode(await ws.receive_bytes())
            if first.get("type") == "init":
                world = first["whitebox"]
                world_id = (
                    world.get("source_id", "") + "/" if world.get("source_id") else ""
                ) + world["world_id"]
                game = new_id()
                await runtime.directory.resolve(game, world_id, create=True)
                world["game_id"] = game
            owner, sub = await host.attach(
                dict(user_id="native-acceptance", trace_id="native-area", message=first)
            )

            async def send():
                while True:
                    wire = await sub.recv()
                    message = msgspec.msgpack.decode(wire)
                    if (
                        message.get("type") == "operation.result"
                        and message.get("op") == "area.presented"
                        and message.get("state") == "completed"
                    ):
                        if fault["drop"]:
                            fault.update(drop=False, lost_request=message["request_id"])
                            await ws.close(code=1011)
                            return
                        if message["request_id"] == fault["lost_request"]:
                            fault["lost_replays"] += 1
                            continue
                    await ws.send_bytes(wire)
                    if message.get("type") in {"error", "session.completed"}:
                        await ws.close()
                        return

            sender = asyncio.create_task(send())
            while True:
                owner.accept(sub, await ws.receive_bytes())
        except Exception:
            logging.getLogger(__name__).exception("Native fixture attachment ended")
            if owner is None:
                raise
        finally:
            if sender:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            if owner is not None and sub is not None:
                owner.detach(sub)

    @app.get("/v1/whitebox/owners/{owner_id}/recordings/{recording_id}/ui/{name:path}")
    async def ui_asset(
        owner_id: str,
        recording_id: str,
        name: str,
        authorization: str | None = Header(None),
    ):
        for owner in host.sessions.values():
            source = owner.session
            if (
                owner_id == "native"
                and source.recording
                and source.recording.directory.name == recording_id
                and source.report
                and authorization == "Bearer " + source.report.token
            ):
                value = await source.engine.ui_asset(name)
                return Response(
                    value,
                    media_type=mimetypes.guess_type(name)[0]
                    or "application/octet-stream",
                )
        raise HTTPException(403)

    @app.post("/test/drop-next-completed-area-result")
    async def drop_reply():
        fault["drop"] = True
        return {"armed": True}

    @app.get("/inspection")
    async def inspection():
        sources = []
        for owner in host.sessions.values():
            source = owner.session
            native = None
            native_error = None
            try:
                if source.engine is not None and hasattr(source.engine, "area_status"):
                    native = await source.engine.area_status()
            except (ValueError, ConnectionError) as error:
                native_error = str(error)
            state = source.journal.snapshot or {"operations": {}, "segment": None}
            player = runtime.players.get(getattr(source.engine, "stream_id", None))
            descriptor = (
                getattr(player.get("engine"), "descriptor", {}) if player else {}
            )
            evidence = {
                "artifact": descriptor.get("artifact", {}).get("kind"),
                "loading": descriptor.get("loading", {}).get("kind"),
                "cache_scope": descriptor.get("cache", {}).get("scope"),
                "cache_status": descriptor.get("cache", {}).get("status"),
                "has_cache_complete": "cacheComplete" in descriptor,
            }
            sources.append(
                dict(
                    native=native,
                    partial=evidence,
                    execution={
                        "artifact": descriptor.get("artifact", {}).get("kind"),
                        "startupContext": descriptor.get("artifact", {}).get(
                            "startupContext"
                        ),
                        "worldBuildHash": descriptor.get("worldBuildHash"),
                        "serviceInstanceId": descriptor.get("service", {}).get(
                            "instanceId"
                        ),
                    },
                    native_error=native_error,
                    generation=owner.generation,
                    source_session_id=source.session_id,
                    operations=[
                        {k: v for k, v in op.items() if k != "payload_sha256"}
                        for op in state["operations"].values()
                    ],
                    segment=state["segment"],
                    model_active=leases.active,
                )
            )
        return dict(sources=sources, players=len(runtime.players), fault=dict(fault))

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    print(
        json.dumps(dict(url=f"http://127.0.0.1:{listener.getsockname()[1]}")),
        flush=True,
    )
    uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on")).run(
        sockets=[listener]
    )


if __name__ == "__main__":
    main()
