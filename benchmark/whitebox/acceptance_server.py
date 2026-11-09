#!/usr/bin/env python3
"""Isolated acceptance stack: real Redis/coordinator/gateway/reservations.

Only model, CLI and the object store are local mocks. No cloud credentials or
production data are accessed. Ctrl-C stops the dedicated throwaway Redis.
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import uvicorn
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from sglang.multimodal_gen.runtime.entrypoints.realtime_coordinator_server import (
    create_app as coordinator_app,
)
from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
    HTTPCoordinatorClient,
    create_app,
)
from sglang.multimodal_gen.runtime.realtime.coordinator import (
    HTTPWorkerReservationClient,
    RealtimeCoordinator,
    RedisCoordinatorStore,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.browser_engine import (
    factory,
    ui_factory,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.mock_worker import (
    create_app as worker_app,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.recording import ArchiveManager
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig


async def main():
    root = Path(tempfile.mkdtemp(prefix="whitebox-acceptance-NOT-TRAINING-"))
    redis = subprocess.Popen(
        [
            "redis-server",
            "--bind",
            "127.0.0.1",
            "--port",
            "16389",
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(root),
        ],
        stdout=(root / "redis.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    await asyncio.sleep(0.3)
    if redis.poll() is not None:
        raise RuntimeError(
            "Dedicated Redis could not start; refusing to use an existing server"
        )
    store = RedisCoordinatorStore(
        host="127.0.0.1",
        port=16389,
        password_env=None,
        key_prefix="whitebox-local-acceptance",
        ttl_s=15,
        worker_ttl_s=8,
    )
    coordinator = coordinator_app(
        RealtimeCoordinator(
            store,
            wait_timeout_s=2,
            reservation_client=HTTPWorkerReservationClient(timeout_s=2),
        )
    )
    worker = worker_app(
        coordinator_url="http://127.0.0.1:18088",
        public_url="http://127.0.0.1:18089",
        capacity=2,
    )

    class LocalArchive:
        async def upload(self, path, key):
            target = root / "objects" / key
            target.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copyfile, path, target)

        def playback_url(self, key):
            return "http://127.0.0.1:18087/mock-media/" + key

    gateway = create_app(
        HTTPCoordinatorClient("http://127.0.0.1:18088"),
        model_revision="whitebox-mock",
        vae_fingerprint="local",
        internal_output_url="ws://127.0.0.1:18087/v1/realtime_video/output",
        lease_renew_interval_s=3,
        whitebox_config=WhiteboxConfig(
            engine_factory=factory,
            ui_factory=ui_factory,
            spool=root / "spool",
            spec=VideoSpec(320, 192),
            play_seconds=float(os.environ.get("PLAY_SECONDS", "60")),
            object_prefix="mock-acceptance",
            model_metadata={"mode": "mock", "training_eligible": False},
        ),
        whitebox_archives=ArchiveManager(LocalArchive()),
    )
    gateway.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:4417"],
        allow_methods=["GET"],
        allow_headers=["Authorization"],
    )

    @gateway.get("/mock-media/{key:path}")
    async def media(key: str):
        path = root / "objects" / key
        if (
            not path.resolve().is_relative_to((root / "objects").resolve())
            or not path.is_file()
        ):
            raise HTTPException(404)
        return FileResponse(path, media_type="video/mp4")

    print(f"Acceptance spool and mock objects: {root}", flush=True)
    servers = [
        uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                access_log=False,
                log_level="warning",
                ws_max_size=8 * 1024 * 1024 + 4096,
                ws_max_queue=1,
            )
        )
        for app, port in [(coordinator, 18088), (worker, 18089), (gateway, 18087)]
    ]
    try:
        await asyncio.gather(*(server.serve() for server in servers))
    finally:
        redis.terminate()
        redis.wait(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())
