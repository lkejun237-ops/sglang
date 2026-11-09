# SPDX-License-Identifier: Apache-2.0
"""Real registry/store state machines; failure and delayed reply contracts."""

import asyncio
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI

from sglang.multimodal_gen.runtime.realtime.coordinator import (
    CoordinatorRejected,
    HTTPWorkerReservationClient,
    InMemoryCoordinatorStore,
    RealtimeCoordinator,
    RedisCoordinatorStore,
    WorkerHeartbeat,
)
from sglang.multimodal_gen.runtime.realtime.player_lifecycle import (
    sdk_cleanup_receipt,
)
from sglang.multimodal_gen.runtime.realtime.worker_reservation import (
    WorkerReservationRegistry,
    WorkerReservationRejected,
    install_worker_reservation_routes,
)
from sglang.multimodal_gen.test.unit.realtime.test_realtime_coordinator import (
    FakeAsyncRedis,
)


async def system(backend):
    now = [100.0]
    kwargs = dict(
        ttl_s=10, worker_ttl_s=10000, clock=lambda: now[0], wall_clock=lambda: now[0]
    )
    if backend == "redis":
        store = RedisCoordinatorStore(client=FakeAsyncRedis(), **kwargs)
    else:
        store = InMemoryCoordinatorStore(**kwargs)
    registry = WorkerReservationRegistry(
        worker_epoch="e", capacity=1, clock=lambda: now[0], max_consumed_age_s=5
    )
    app = FastAPI()
    install_worker_reservation_routes(app, registry)
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker"
    )
    coordinator = RealtimeCoordinator(
        store,
        wait_timeout_s=0,
        reservation_client=HTTPWorkerReservationClient(client=http),
        player_session_ttl_seconds=60,
    )
    await coordinator.heartbeat(
        WorkerHeartbeat(
            worker_id="w",
            role="denoiser",
            endpoint="ws://worker/bound",
            az="a",
            capacity=1,
            model_revision="m",
            vae_fingerprint="local",
            worker_epoch="e",
            reservation_endpoint="http://worker/v1/realtime_worker",
        )
    )

    async def admit(name):
        return await coordinator.admit(
            user_id=name,
            session_id=name,
            generation_id="g",
            model_revision="m",
            vae_fingerprint="local",
            player_session=True,
            wait_for_capacity=False,
        )

    return now, store, registry, coordinator, http, admit


@pytest.mark.parametrize("backend", ["memory", "redis"])
def test_close_needs_both_owners_and_lost_ack_retry_does_not_release_next_owner(
    backend,
):
    async def run():
        now, store, worker, coordinator, http, admit = await system(backend)
        try:
            a = await admit("a")
            await worker.consume(
                a.token,
                session_id="a",
                generation_id="g",
                worker_epoch="e",
                owner_id="socket-a",
            )
            receipt = sdk_cleanup_receipt(
                a,
                state="retired",
                source_id="source-a",
                owner_epoch="room-e",
                view_id="view-a",
            )
            with pytest.raises(CoordinatorRejected, match="RELEASE_PENDING"):
                await coordinator.release_verified(a, sdk_cleanup=receipt)
            await worker.release(a.token, owner_id="socket-a")
            assert (await worker.snapshot())["active_sessions"] == 1
            with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
                await admit("b")
            # The previously confirmed SDK receipt is retained by the store.
            await coordinator.release_verified(a)
            b = await admit("b")
            await coordinator.release_verified(a, sdk_cleanup=receipt)
            assert (await worker.snapshot())["reserved_sessions"] == 1
            assert (await store.verified_release_status(b))[0].token == b.token
        finally:
            await http.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "redis"])
def test_model_cleanup_alone_does_not_free_player(backend):
    async def run():
        _, _, worker, coordinator, http, admit = await system(backend)
        try:
            a = await admit("a")
            await worker.consume(
                a.token,
                session_id="a",
                generation_id="g",
                worker_epoch="e",
                owner_id="socket",
            )
            await worker.release(a.token, owner_id="socket")
            with pytest.raises(CoordinatorRejected, match="SDK_CLEANUP_PENDING"):
                await coordinator.release_verified(a)
            with pytest.raises(CoordinatorRejected, match="VERIFIED_RELEASE_REQUIRED"):
                await coordinator.release(a)
            with pytest.raises(CoordinatorRejected, match="VERIFIED_RELEASE_REQUIRED"):
                await coordinator.store.release(
                    replace(a, player_expires_at=None, admitted_at=None)
                )
            bad = sdk_cleanup_receipt(
                replace(a, session_id="other"),
                state="retired",
                source_id="s",
                owner_epoch="e",
                view_id="v",
            )
            with pytest.raises(CoordinatorRejected, match="INVALID_SDK_CLEANUP"):
                await coordinator.release_verified(a, sdk_cleanup=bad)
            assert (await worker.snapshot())["active_sessions"] == 1
        finally:
            await http.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "redis"])
def test_expiry_cannot_erase_occupied_slot_or_be_extended_by_renewal(backend):
    async def run():
        now, store, worker, coordinator, http, admit = await system(backend)
        try:
            a = await admit("a")
            await worker.consume(
                a.token,
                session_id="a",
                generation_id="g",
                worker_epoch="e",
                owner_id="socket",
            )
            now[0] += 6
            a = await coordinator.renew(a)
            assert a.admitted_at == 100 and a.player_expires_at == 160
            assert (await worker.snapshot())[
                "lifecycle"
            ] == "ready"  # Whitebox wait is not a zombie.
            now[0] = 170
            with pytest.raises(CoordinatorRejected, match="PLAYER_EXPIRED"):
                await coordinator.renew(a)
            with pytest.raises(WorkerReservationRejected, match="PLAYER_EXPIRED"):
                await worker.mark_runnable(a.token, owner_id="socket")
            assert (await store.verified_release_status(a))[1] is False
            with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
                await admit("b")
            with pytest.raises(CoordinatorRejected, match="LEASE_LOST"):
                await store.verified_release_status(replace(a, player_expires_at=9999))
        finally:
            await http.aclose()

    asyncio.run(run())


def test_cancelled_never_created_reservation_fences_late_reserve():
    async def run():
        worker = WorkerReservationRegistry(
            worker_epoch="e", capacity=1, clock=lambda: 100
        )
        identity = dict(
            session_id="a", generation_id="g", worker_epoch="e", close_id="close-a"
        )
        await worker.prepare_release("a", player_expires_at=160, **identity)
        assert (await worker.finalize_release("a", **identity))["state"] == "released"
        with pytest.raises(WorkerReservationRejected, match="RESERVATION_RELEASED"):
            await worker.reserve(
                "a",
                session_id="a",
                generation_id="g",
                worker_epoch="e",
                ttl_s=30,
                player_expires_at=160,
            )
        with pytest.raises(WorkerReservationRejected, match="IDENTITY_MISMATCH"):
            await worker.finalize_release("a", **{**identity, "close_id": "other"})
        assert (await worker.snapshot())["reserved_sessions"] == 0

    asyncio.run(run())
