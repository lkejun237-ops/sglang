import asyncio
from dataclasses import replace

import httpx
import pytest

from sglang.multimodal_gen.runtime.realtime.coordinator import (
    CoordinatorRejected,
    HTTPWorkerReservationClient,
    InMemoryCoordinatorStore,
    RealtimeCoordinator,
    RedisCoordinatorStore,
)
from sglang.multimodal_gen.test.unit.realtime.test_realtime_coordinator import (
    FakeAsyncRedis,
    _heartbeat,
)


class Reservations:
    def __init__(self):
        self.reject = False
        self.releases = 0

    async def reserve(self, *args, **kwargs):
        pass

    async def release(self, *args, **kwargs):
        pass

    async def release_verified(self, *args, **kwargs):
        self.releases += 1
        if self.reject:
            raise RuntimeError("owner still cleaning")


def test_worker_consumed_owner_and_epoch_survive_failed_cleanup():
    from sglang.multimodal_gen.runtime.realtime.worker_reservation import (
        WorkerReservationRegistry,
        WorkerReservationRejected,
    )

    async def run():
        registry = WorkerReservationRegistry(capacity=1, worker_epoch="epoch")
        await registry.reserve(
            token="token",
            session_id="session",
            generation_id="generation",
            worker_epoch="epoch",
            ttl_s=60,
        )
        await registry.consume(
            "token",
            session_id="session",
            generation_id="generation",
            worker_epoch="epoch",
            owner_id="owner",
        )
        with pytest.raises(WorkerReservationRejected, match="OWNER_MISMATCH"):
            await registry.release("token")
        await registry.quarantine("token", owner_id="owner")
        snapshot = await registry.snapshot()
        assert snapshot["lifecycle"] == "failed" and snapshot["active_sessions"] == 1
        with pytest.raises(WorkerReservationRejected):
            await registry.reserve(
                token="next",
                session_id="next",
                generation_id="next",
                worker_epoch="epoch",
                ttl_s=60,
            )
        await registry.release("token", owner_id="owner")
        assert (await registry.snapshot())["active_sessions"] == 0

    asyncio.run(run())


def make_store(redis):
    if not redis:
        return InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
    return RedisCoordinatorStore(
        host="unused", ttl_s=60, worker_ttl_s=30, client=FakeAsyncRedis()
    )


async def setup(redis=False):
    store = make_store(redis)
    reservations = Reservations()
    coordinator = RealtimeCoordinator(
        store, reservation_client=reservations, wait_timeout_s=0
    )
    await coordinator.heartbeat(
        _heartbeat("worker", "denoiser", vae_fingerprint="local")
    )
    request = dict(
        user_id="user",
        session_id="session",
        generation_id="generation",
        model_revision="minwm-r1",
        vae_fingerprint="local",
        wait_for_capacity=False,
    )
    assignment = await coordinator.admit(**request)
    return coordinator, store, reservations, assignment, request


@pytest.mark.parametrize("redis", [False, True])
def test_verified_release_replays_receipt_without_releasing_new_owner(redis):
    async def scenario():
        coor, store, client, assignment, request = await setup(redis)
        await coor.release_verified(assignment)
        successor = await coor.admit(
            **{**request, "session_id": "next", "generation_id": "next"}
        )
        calls = client.releases
        await coor.release_verified(assignment)
        assert client.releases == calls
        with pytest.raises(CoordinatorRejected):
            await coor.admit(
                **{
                    **request,
                    "user_id": "other",
                    "session_id": "other",
                    "generation_id": "other",
                }
            )
        assert successor.token != assignment.token

    asyncio.run(scenario())


@pytest.mark.parametrize("redis", [False, True])
def test_failed_release_does_not_free_slot_and_quarantine_survives_lease_removal(redis):
    async def scenario():
        coor, store, client, assignment, request = await setup(redis)
        client.reject = True
        with pytest.raises(CoordinatorRejected, match="RELEASE_PENDING"):
            await coor.release_verified(assignment)
        await coor.quarantine(assignment)
        await store.release(assignment)
        await coor.heartbeat(_heartbeat("worker", "denoiser", vae_fingerprint="local"))
        assert (await store.capacity_snapshot())["roles"]["denoiser"]["free_slots"] == 0
        with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
            await coor.admit(
                **{
                    **request,
                    "user_id": "next",
                    "session_id": "next",
                    "generation_id": "next",
                }
            )
        await coor.heartbeat(
            _heartbeat(
                "worker", "denoiser", vae_fingerprint="local", worker_epoch="new-epoch"
            )
        )
        new = await coor.admit(
            **{
                **request,
                "user_id": "next",
                "session_id": "next",
                "generation_id": "next",
            }
        )
        assert new.denoiser.worker_epoch == "new-epoch"

    asyncio.run(scenario())


@pytest.mark.parametrize("redis", [False, True])
def test_strict_release_resolves_trusted_assignment_before_network_io(redis):
    async def scenario():
        coor, store, client, assignment, _ = await setup(redis)
        forged = replace(
            assignment,
            denoiser=replace(
                assignment.denoiser, reservation_endpoint="http://untrusted"
            ),
        )
        with pytest.raises(CoordinatorRejected, match="LEASE_LOST"):
            await coor.release_verified(forged)
        with pytest.raises(CoordinatorRejected, match="LEASE_LOST"):
            await coor.quarantine(forged)
        assert client.releases == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [404, 409, 500, 204])
def test_http_release_requires_owner_cleanup_ack(status):
    async def scenario():
        _, _, _, assignment, _ = await setup()

        def respond(request):
            assert request.url.params["worker_epoch"] == "epoch-a"
            return httpx.Response(
                status,
                json={"detail": {"reason": "RESERVATION_OWNER_MISMATCH"}}
                if status != 204
                else None,
            )

        http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        client = HTTPWorkerReservationClient(client=http)
        if status == 204:
            await client.release_verified(assignment.denoiser, token=assignment.token)
        else:
            with pytest.raises(CoordinatorRejected, match="RELEASE_PENDING"):
                await client.release_verified(
                    assignment.denoiser, token=assignment.token
                )
        await http.aclose()

    asyncio.run(scenario())


def test_render_segment_cleanup_retains_ownership_without_scheduler_ack(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
        realtime_video_api,
    )

    events = []

    async def release(_):
        return False

    async def conditioning(_):
        events.append("conditioning_closed")

    monkeypatch.setattr(
        realtime_video_api, "_release_scheduler_realtime_session", release
    )
    session = SimpleNamespace(
        id="segment",
        request=SimpleNamespace(source_prompt_protocol="ordered-boundary-v2"),
        control_refresh_task=None,
        adapter=SimpleNamespace(close_conditioning=conditioning),
        input_temp_dir=str(tmp_path),
        dispose=lambda: events.append("disposed"),
    )
    with pytest.raises(RuntimeError, match="not been confirmed"):
        asyncio.run(realtime_video_api._cleanup_realtime_session(session, None, None))
    assert not events and tmp_path.exists()
