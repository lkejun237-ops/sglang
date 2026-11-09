# SPDX-License-Identifier: Apache-2.0
"""Deployable source echo worker with REAL coordinator reservations, NO model.

Run one replica with one active session by default. The production worker implements the
same source protocol and reservation lifecycle, so no gateway bypass is needed.
"""

import asyncio
import logging
import math
import os
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import msgspec
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from sglang.multimodal_gen.runtime.realtime.gateway_assignment import (
    parse_gateway_assignment,
)
from sglang.multimodal_gen.runtime.realtime.worker_heartbeat import (
    WorkerHeartbeatReporter,
)
from sglang.multimodal_gen.runtime.realtime.worker_reservation import (
    WorkerReservationRegistry,
    install_worker_reservation_routes,
)

logger = logging.getLogger(__name__)


def create_app(
    *,
    coordinator_url,
    public_url,
    capacity=1,
    worker_id="whitebox-mock-worker",
    startup_seconds=90,
    idle_seconds=120,
    region="cn-beijing",
):
    for value in (startup_seconds, idle_seconds):
        if not math.isfinite(value) or not 0 < value <= 900:
            raise ValueError("Mock timeouts must be finite and within (0, 900]")
    epoch = uuid4().hex
    registry = WorkerReservationRegistry(worker_epoch=epoch, capacity=capacity)
    public_url = public_url.rstrip("/")

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=3) as client:
            reporter = WorkerHeartbeatReporter(
                client,
                coordinator_url=coordinator_url,
                health_url=public_url + "/healthz",
                state_url=public_url + "/v1/realtime_worker/state",
                worker_id=worker_id,
                worker_epoch=epoch,
                role="denoiser",
                endpoint=public_url.replace("http", "ws", 1) + "/generate",
                reservation_endpoint=public_url + "/v1/realtime_worker",
                az=region,
                capacity=capacity,
                model_revision="whitebox-mock",
                vae_fingerprint="local",
            )
            task = asyncio.create_task(reporter.run(interval_s=2))
            try:
                yield
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    app = FastAPI(lifespan=lifespan)
    install_worker_reservation_routes(app, registry)

    @app.get("/healthz")
    async def health():
        return {"status": "ok", "mode": "mock", "capacity": capacity}

    @app.websocket("/generate")
    async def generate(ws: WebSocket):
        await ws.accept()
        owner = uuid4().hex
        assignment = None
        consumed = False
        try:
            assignment = parse_gateway_assignment(ws.query_params)
            if assignment is None:
                raise ValueError("A coordinator reservation is required")
            await registry.consume(
                assignment.coordinator_token,
                session_id=assignment.session_id,
                generation_id=assignment.generation_id,
                worker_epoch=assignment.worker_epoch,
                owner_id=owner,
            )
            consumed = True
            await registry.mark_runnable(assignment.coordinator_token, owner_id=owner)
            spec = None
            index = prompt_id = 0

            async def emit(**payload):
                await ws.send_bytes(msgspec.msgpack.encode(payload))

            while True:
                # Gateway enforces the user deadline. This independently bounds
                # abandoned reservations if the gateway disappears mid-stream.
                message = msgspec.msgpack.decode(
                    await asyncio.wait_for(
                        ws.receive_bytes(),
                        startup_seconds if spec is None else idle_seconds,
                    )
                )
                kind, payload = message.get("kind"), message.get("payload")
                if message.get("type") == "init":
                    if spec is not None:
                        raise ValueError("Already initialized")
                    from .contracts import VideoSpec

                    source = message["source_stream"]
                    spec = VideoSpec(source["width"], source["height"], source["fps"])
                    if (
                        source.get("format") != "rgb24"
                        or not 0 < spec.frame_bytes <= 4 * 1024 * 1024
                        or spec.fps != 24
                    ):
                        raise ValueError("Unsupported mock source")
                    await emit(
                        type="session.ready",
                        source_stream=source,
                        max_message_bytes=message["source_message_budget_bytes"],
                        max_pending_source_frames=4,
                        max_pending_source_bytes=4 * spec.frame_bytes,
                        max_source_message_bytes=spec.frame_bytes + 4096,
                    )
                elif kind == "source.append" and spec:
                    if (
                        payload["source_frame_start"] != index
                        or payload["num_frames"] != 1
                        or payload["pts_us"] != spec.pts(index)
                        or len(payload["data"]) != spec.frame_bytes
                    ):
                        raise ValueError("Invalid source timeline")
                    await emit(
                        type="source.accepted",
                        source_frame_start=index,
                        source_frame_end=index + 1,
                    )
                    await emit(
                        type="frame_batch",
                        output_frame_start=index,
                        num_frames=1,
                        pts_us=spec.pts(index),
                        width=spec.width,
                        height=spec.height,
                        format="rgb24",
                        encoding="raw",
                        payload=payload["data"],
                        event_id=prompt_id,
                        source_alignment=[
                            dict(
                                source_frame_start=index,
                                output_frame_start=index,
                                num_frames=1,
                            )
                        ],
                    )
                    index += 1
                    await emit(
                        type="source.consumed",
                        source_frame_end=index,
                        source_bytes_end=index * spec.frame_bytes,
                    )
                elif kind == "prompt":
                    prompt_id = message["event_id"]
                elif kind == "source.end":
                    if payload["source_frame_end"] != index:
                        raise ValueError("Invalid source end")
                    await emit(type="session.completed")
                    break
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            # Do not log the reservation query or arbitrary payload text.
            logger.warning("Mock worker session failed: %s", type(exc).__name__)
            await ws.close(code=1011, reason="mock worker session failed")
        finally:
            if consumed:
                await registry.release(assignment.coordinator_token, owner_id=owner)
            try:
                await ws.close()
            except RuntimeError:
                pass

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "18089"))
    app = create_app(
        coordinator_url=os.environ["COORDINATOR_URL"],
        public_url=os.environ.get("WORKER_URL", f"http://127.0.0.1:{port}"),
        capacity=int(os.environ.get("WHITEBOX_SLOTS", "1")),
        startup_seconds=float(os.environ.get("WHITEBOX_STARTUP_SECONDS", "90")),
        region=os.environ.get("WHITEBOX_REGION", "cn-beijing"),
    )
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        ws_max_size=8 * 1024 * 1024 + 4096,
        ws_max_queue=1,
        ws_per_message_deflate=False,
    )
