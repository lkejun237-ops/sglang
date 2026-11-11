# SPDX-License-Identifier: Apache-2.0
"""One authoritative game runtime process per Pod. No deployment side effects."""

import os
from pathlib import Path

import httpx
import uvicorn

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.recording import (
    ArchiveManager,
    S3ArchiveStore,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import (
    RoomRuntime,
    create_app,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import HTTPRoomDirectory
from sglang.multimodal_gen.runtime.realtime.whitebox.session import load_plugin


def compose_room_driver(default_factory):
    """Only the verified control owner selects the sealed multiplayer driver.

    Ordinary joins retain their configured driver and lifecycle. The control
    context is supplied by RoomControlRuntime, never an ordinary player body.
    """

    async def start(**kwargs):
        if kwargs.get("create_context") is not None:
            from sglang.multimodal_gen.runtime.realtime.whitebox.worldkit_room import (
                worldkit_room,
            )

            return await worldkit_room(**kwargs)
        return await default_factory(**kwargs)

    return start


def main():
    if os.environ.get("WHITEBOX_MODEL_BACKEND") == "role_pool":
        if any(
            os.environ.get(name) != "1"
            for name in (
                "WHITEBOX_BOUND_COMPUTE",
                "WHITEBOX_ROOM_PIPELINE",
                "WHITEBOX_RENDER_SEGMENTS",
            )
        ):
            raise ValueError(
                "Role pool serving requires the Room render-v2 bound pipeline"
            )
    token = os.environ["WHITEBOX_INTERNAL_TOKEN"]
    client = httpx.AsyncClient(timeout=5)
    pipeline_config = None
    if os.environ.get("WHITEBOX_ROOM_PIPELINE") == "1":
        from sglang.multimodal_gen.runtime.realtime.whitebox.session import (
            from_environment,
        )

        pipeline_config, archives = from_environment(room=True)
    else:
        bucket = os.environ.get("WHITEBOX_OSS_BUCKET")
        store = (
            S3ArchiveStore(
                bucket,
                os.environ["WHITEBOX_OSS_ENDPOINT"],
                os.environ.get("WHITEBOX_OSS_REGION", "cn-beijing"),
            )
            if bucket
            else None
        )
        archives = ArchiveManager(store, processes=True)
    runtime = RoomRuntime(
        directory=HTTPRoomDirectory(
            client, os.environ["WHITEBOX_ROOM_COORDINATOR_URL"], token
        ),
        owner_id=os.environ["WHITEBOX_ROOM_OWNER_ID"],
        driver_factory=compose_room_driver(
            load_plugin(os.environ["WHITEBOX_ROOM_DRIVER"])
        ),
        spool=Path(
            os.environ.get("WHITEBOX_ROOM_SPOOL", "/var/lib/sglang/whitebox-rooms")
        ),
        archives=archives,
        prefix=os.environ.get("WHITEBOX_OSS_PREFIX", "white_box_game_record"),
        spec=VideoSpec(
            int(os.environ.get("WHITEBOX_WIDTH", "832")),
            int(os.environ.get("WHITEBOX_HEIGHT", "480")),
        ),
        max_players=int(os.environ.get("WHITEBOX_MAX_PLAYERS", "8")),
        max_rooms=int(os.environ.get("WHITEBOX_MAX_ROOMS", "16")),
        idle_seconds=float(os.environ.get("WHITEBOX_PLAYER_IDLE_SECONDS", "45")),
    )
    pipelines = coordinator = render_host = None
    if pipeline_config:
        from dataclasses import replace

        from websockets.asyncio.client import connect

        from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
            HTTPCoordinatorClient,
        )
        from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
            create_model_leases,
        )
        from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports
        from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
            RoomPipelines,
        )

        pipeline_config = replace(pipeline_config, spool=runtime.spool)
        coordinator = HTTPCoordinatorClient(os.environ["WHITEBOX_ROOM_COORDINATOR_URL"])
        reports = Reports(runtime.spool, archives)

        def leases(user_id, trace_id):
            return create_model_leases(
                coordinator,
                connect,
                user_id=user_id,
                trace_id=trace_id,
                model_revision=os.environ["WHITEBOX_ROUTING_REVISION"],
                vae_fingerprint="local",
                max_bytes=pipeline_config.max_message_bytes,
            )

        pipelines = RoomPipelines(
            runtime,
            pipeline_config,
            archives,
            reports,
            leases,
            max_sessions=int(os.environ.get("WHITEBOX_ROOM_PIPELINE_CAPACITY", "1")),
            grace_seconds=float(
                os.environ.get("WHITEBOX_RECONNECT_GRACE_SECONDS", "15")
            ),
        )
        if os.environ.get("WHITEBOX_RENDER_SEGMENTS") == "1":
            from sglang.multimodal_gen.runtime.realtime.whitebox.native_rtc import (
                validate_rtc_source_capacity,
            )
            from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
                RenderHost,
            )

            source_capacity = int(
                os.environ.get("WHITEBOX_ROOM_PIPELINE_CAPACITY", "1")
            )
            if os.environ.get("WHITEBOX_NATIVE_RTC_ENABLED") == "1":
                validate_rtc_source_capacity(source_capacity)
            render_host = RenderHost(
                runtime,
                pipeline_config,
                archives,
                reports,
                coordinator,
                connect,
                max_sources=source_capacity,
            )
            runtime.render_host = render_host
            render_host.lock = pipelines.lock
            render_host.peer_sessions = pipelines.sessions
            pipelines.peer_sessions = render_host.sessions
    app = create_app(runtime, token)
    if pipelines:
        runtime.maintenance.lock = pipelines.lock
        room_busy = runtime.maintenance.is_playing
        runtime.maintenance.is_playing = lambda: bool(
            room_busy()
            or any(not session.closed for session in pipelines.sessions.values())
            or (render_host and render_host.sessions)
        )
    if pipelines:
        pipelines.install(app, token)
        reports.install(app)
    if render_host:
        render_host.install(app, token)
    # HTTP client's pool is process scoped, closed on application shutdown.
    original_close = runtime.close

    async def close():
        try:
            if render_host:
                await render_host.close()
            if pipelines:
                await pipelines.close()
            await original_close()
        finally:
            await client.aclose()
            if coordinator:
                await coordinator.close()

    runtime.close = close
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("WHITEBOX_ROOM_PORT", "8092")),
        workers=1,
        ws_max_size=8 * 1024**2 + 16384,
        ws_max_queue=1,
        ws_per_message_deflate=False,
    )


if __name__ == "__main__":
    main()
