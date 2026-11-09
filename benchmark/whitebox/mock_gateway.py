#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LOCAL MOCK ONLY. Exercises the gateway/player/recording contract without H3.

Synthetic pixels are not model outputs and must never enter training datasets.
No Redis, OSS, GPU, CLI or production endpoint is contacted.
"""

import asyncio
import tempfile
from pathlib import Path

import msgspec
import uvicorn

from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import create_app
from sglang.multimodal_gen.runtime.realtime.coordinator import (
    InMemoryCoordinatorStore,
    RealtimeCoordinator,
    WorkerHeartbeat,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
    EngineFrame,
    VideoSpec,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig
from sglang.multimodal_gen.runtime.realtime.whitebox.ui import RgbaOverlayRenderer

events = []
spec = VideoSpec(320, 192)


class Engine:
    stopped = False

    def __init__(self):
        self.inputs = {"held_keys": []}

    async def frames(self):
        index = 0
        while not self.stopped:
            await asyncio.sleep(1 / spec.fps)
            color = [40 + index % 80, 180 if self.inputs.get("held_keys") else 80, 60]
            yield EngineFrame(
                index,
                spec.pts(index),
                bytes(color) * (spec.width * spec.height),
                {"fixture": True, "frame": index},
                dict(self.inputs),
            )
            self.inputs = {
                key: value for key, value in self.inputs.items() if key != "events"
            }
            index += 1

    async def send_input(self, sequence, payload):
        events.append(dict(kind="input", sequence=sequence, payload=payload))
        self.inputs = payload

    async def finish(self):
        self.stopped = True
        events.append(dict(kind="engine_finish"))

    async def close(self):
        self.stopped = True
        events.append(dict(kind="engine_closed"))


class MockH3:
    def __init__(self):
        self.queue = asyncio.Queue(maxsize=4)
        self.prompt_id = 0

    async def emit(self, **message):
        await self.queue.put(msgspec.msgpack.encode(message))

    async def send(self, wire):
        message = msgspec.msgpack.decode(wire)
        kind, payload = message.get("kind"), message.get("payload")
        if message["type"] == "init":
            await self.emit(
                type="session.ready",
                source_stream=message["source_stream"],
                max_message_bytes=64 * 1024 * 1024,
                max_pending_source_frames=4,
                max_pending_source_bytes=4 * spec.frame_bytes,
                max_source_message_bytes=spec.frame_bytes + 4096,
            )
        elif kind == "prompt":
            self.prompt_id = message["event_id"]
            events.append(dict(kind="prompt", event_id=self.prompt_id, prompt=payload))
        elif kind == "source.append":
            index = payload["source_frame_start"]
            await self.emit(
                type="source.accepted",
                source_frame_start=index,
                source_frame_end=index + 1,
            )
            color = payload["data"][:3]
            output = bytes([color[0], color[1], 150]) * (spec.width * spec.height)
            await self.emit(
                type="frame_batch",
                output_frame_start=index,
                num_frames=1,
                pts_us=spec.pts(index),
                width=spec.width,
                height=spec.height,
                format="rgb24",
                encoding="raw",
                payload=output,
                event_id=self.prompt_id,
                source_alignment=[
                    dict(
                        source_frame_start=index, output_frame_start=index, num_frames=1
                    )
                ],
            )
            await self.emit(
                type="source.consumed",
                source_frame_end=index + 1,
                source_bytes_end=(index + 1) * spec.frame_bytes,
            )
        elif kind == "source.end":
            await self.emit(type="session.completed")

    async def recv(self):
        return await self.queue.get()

    async def close(self):
        events.append(dict(kind="h3_mock_closed"))


class Coordinator:
    def __init__(self):
        self.store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=600)
        self.coordinator = RealtimeCoordinator(self.store, wait_timeout_s=1)

    async def admit(self, **kwargs):
        kwargs.pop("trace_id", None)
        await self.store.heartbeat(
            WorkerHeartbeat(
                worker_id="local-mock",
                role="denoiser",
                endpoint="ws://local-mock/generate",
                az="local",
                capacity=1,
                model_revision="mock-h3",
                vae_fingerprint="local",
                routing_revision="mock-h3",
                worker_epoch="mock",
                reservation_endpoint="http://local-mock/reservations",
            )
        )
        return await self.coordinator.admit(**kwargs)

    async def renew(self, assignment):
        events.append(dict(kind="lease_renewed"))
        return await self.coordinator.renew(assignment)

    async def release(self, assignment):
        await self.coordinator.release(assignment)
        events.append(dict(kind="lease_released"))

    async def health(self):
        return {"status": "ready"}

    async def capacity(self):
        return await self.coordinator.capacity_snapshot()


async def engine_factory(**_):
    return Engine()


def ui_factory(**_):
    async def overlay(frame, spec):
        return bytes([240, 210, 40, 255]) * (spec.width * 12) + bytes(
            spec.width * (spec.height - 12) * 4
        )

    return RgbaOverlayRenderer(
        overlay, metadata={"fixture": True, "bundle": "mock-yellow-header"}
    )


async def connect(*_, **__):
    return MockH3()


if __name__ == "__main__":
    spool = Path(tempfile.mkdtemp(prefix="whitebox-mock-NOT-TRAINING-"))
    config = WhiteboxConfig(
        engine_factory=engine_factory,
        ui_factory=ui_factory,
        spool=spool,
        spec=spec,
        play_seconds=120,
        object_prefix="LOCAL_MOCK_NOT_TRAINING",
    )
    app = create_app(
        Coordinator(),
        model_revision="mock-h3",
        vae_fingerprint="local",
        internal_output_url="ws://unused/internal",
        whitebox_config=config,
        connect_factory=connect,
        release_grace_s=0,
    )

    @app.get("/mock/events")
    async def inspect_events():
        return {"events": events, "spool": str(spool)}

    print(f"LOCAL MOCK ONLY, temporary recordings: {spool}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=18087)
