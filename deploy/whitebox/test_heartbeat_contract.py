"""Real worker reservation routes exercised from the supported launcher argv."""

import pytest


def test_launched_heartbeat_can_reserve_and_release_real_worker_routes(tmp_path):
    import asyncio

    import httpx
    from fastapi import FastAPI
    from model_service import commands
    from test_release import bundle

    from sglang.multimodal_gen.runtime.realtime.coordinator import (
        HTTPWorkerReservationClient,
        WorkerSlot,
    )
    from sglang.multimodal_gen.runtime.realtime.worker_reservation import (
        WorkerReservationRegistry,
        install_worker_reservation_routes,
    )

    plan = commands(bundle())
    heartbeat = plan["heartbeat"]["argv"]
    endpoint = heartbeat[heartbeat.index("--reservation-endpoint") + 1]

    async def run():
        app = FastAPI()
        registry = WorkerReservationRegistry(worker_epoch="epoch", capacity=1)
        install_worker_reservation_routes(app, registry)
        slot = WorkerSlot(
            worker_id="whitebox-th-h3",
            role="denoiser",
            endpoint="ws://10.128.1.2:80/v1/realtime_video/generate",
            az="ap-southeast-7a",
            slot_index=0,
            model_revision="step2500",
            vae_fingerprint="local",
            worker_epoch="epoch",
            reservation_endpoint=endpoint,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
            client = HTTPWorkerReservationClient(client=http)
            await client.reserve(
                slot,
                token="first",
                session_id="session",
                generation_id="gen",
                ttl_s=30,
            )
            assert (await registry.snapshot())["reserved_sessions"] == 1
            with pytest.raises(httpx.HTTPStatusError) as rejected:
                await client.reserve(
                    slot,
                    token="second",
                    session_id="other",
                    generation_id="gen",
                    ttl_s=30,
                )
            assert rejected.value.response.status_code == 409
            assert (await registry.snapshot())["reserved_sessions"] == 1
            await client.release(slot, token="first")
            assert (await registry.snapshot())["reserved_sessions"] == 0
            await client.reserve(
                slot,
                token="next",
                session_id="next",
                generation_id="gen",
                ttl_s=30,
            )
            await client.release(slot, token="next")

    asyncio.run(run())
