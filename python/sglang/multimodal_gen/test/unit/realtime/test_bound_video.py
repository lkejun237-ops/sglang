"""Parent reservation and native child cleanup; no GPU evidence."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import msgspec
from test_zing_h3_source_adapter import configured  # noqa
from test_zing_prefill_api import Socket, payload, service  # noqa

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    bound_video as bound,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    realtime_video_api as api,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    zing_prefill as prefill,
)
from sglang.multimodal_gen.runtime.realtime.gateway_assignment import (
    GatewayManagedConfig,
)


def test_parent_is_consumed_once_and_survives_three_prefills_and_generate(
    service, monkeypatch
):
    async def run():
        ws = Socket()
        gateway = GatewayManagedConfig(
            "parent", "generation", "token", "worker-a", "", "", "", ""
        )
        registry = NS(
            mark_runnable=AsyncMock(),
            mark_blocked=AsyncMock(),
            quarantine=AsyncMock(),
            player_deadline=AsyncMock(return_value=None),
        )
        consume = AsyncMock(return_value=registry)
        release = AsyncMock()
        calls = []
        monkeypatch.setattr(api, "_gateway_managed_config", lambda _: gateway)
        monkeypatch.setattr(api, "_consume_gateway_reservation", consume)
        monkeypatch.setattr(api, "_release_gateway_reservation", release)

        async def compute(socket, data, size, session_id, *, binding=None):
            calls.append(
                (
                    "prefill",
                    data["request_id"],
                    binding.gateway.worker_epoch,
                    session_id,
                    binding.owner_id,
                )
            )
            binding.cleaned = True
            return dict(
                type="prefill.completed",
                cache_key=data["prompt"],
                request_id=data["request_id"],
            )

        async def generate(socket, binding=None):
            raw = msgspec.msgpack.decode(await socket.receive_bytes())
            calls.append(
                (
                    "generate",
                    raw["prompt"],
                    binding.gateway.worker_epoch,
                    binding.gateway.session_id,
                    binding.owner_id,
                )
            )
            await socket.send_bytes(msgspec.msgpack.encode({"type": "session.ready"}))
            binding.cleaned = True

        monkeypatch.setattr(prefill, "run_prefill_case", compute)
        monkeypatch.setattr(api, "generate_bound_child", generate)
        task = asyncio.create_task(bound.bound_video(ws))
        assert (await ws.receive())["type"] == "bound.ready"
        for i, kind in enumerate(["prefill", "prefill", "prefill", "generate"]):
            ws.submit(
                dict(
                    type="bound.open",
                    operation_id=str(i),
                    kind=kind,
                    session_id="child" + str(i),
                    generation_id="g" + str(i),
                )
            )
            assert (await ws.receive())["type"] == "bound.opened"
            body = payload(str(i))
            body["prompt"] = "A" if i == 3 else str(i)
            ws.submit(
                dict(
                    type="bound.data",
                    operation_id=str(i),
                    payload=msgspec.msgpack.encode(body),
                )
            )
            assert (await ws.receive())["type"] == "bound.data"
            done = await ws.receive()
            assert done["cleaned"] is True
            assert release.await_count == 0
        ws.incoming.put_nowait(None)
        await task
        assert consume.await_count == release.await_count == 1
        assert len(calls) == 4 and len({x[4] for x in calls}) == 1
        assert {x[2] for x in calls} == {"worker-a"}
        assert len({x[3] for x in calls}) == 4
        assert calls[-1][1] == "A"

    asyncio.run(run())


def test_cleanup_failure_quarantines_parent_instead_of_releasing(service, monkeypatch):
    async def run():
        ws = Socket()
        gateway = GatewayManagedConfig("p", "g", "t", "a", "", "", "", "")
        registry = NS(
            quarantine=AsyncMock(), player_deadline=AsyncMock(return_value=None)
        )
        release = AsyncMock()
        monkeypatch.setattr(api, "_gateway_managed_config", lambda _: gateway)
        monkeypatch.setattr(
            api, "_consume_gateway_reservation", AsyncMock(return_value=registry)
        )
        monkeypatch.setattr(api, "_release_gateway_reservation", release)

        async def compute(*a, **k):
            k["binding"].cleaned = False
            raise RuntimeError("unknown cleanup")

        monkeypatch.setattr(prefill, "run_prefill_case", compute)

        async def close(code=1000, reason=""):
            ws.closed.append(code)
            ws.incoming.put_nowait(None)

        ws.close = close
        task = asyncio.create_task(bound.bound_video(ws))
        await ws.receive()
        ws.submit(
            dict(
                type="bound.open",
                operation_id="one",
                kind="prefill",
                session_id="s",
                generation_id="g",
            )
        )
        await ws.receive()
        ws.submit(
            dict(
                type="bound.data",
                operation_id="one",
                payload=msgspec.msgpack.encode(payload()),
            )
        )
        assert (await ws.receive())["cleaned"] is False
        await task
        assert registry.quarantine.await_count >= 1 and release.await_count == 0

    asyncio.run(run())


def test_player_deadline_cancels_blocked_input_without_freeing_uncommitted_slot(
    service, monkeypatch
):
    import time

    import pytest

    from sglang.multimodal_gen.runtime.realtime.worker_reservation import (
        WorkerReservationRegistry,
    )

    async def run():
        ws = Socket()
        gateway = GatewayManagedConfig("p", "g", "t", "a", "", "", "", "")
        registry = WorkerReservationRegistry(worker_epoch="a", capacity=1)
        deadline = time.time() + 0.25
        await registry.reserve(
            "t",
            session_id="p",
            generation_id="g",
            worker_epoch="a",
            ttl_s=30,
            player_expires_at=deadline,
        )
        ws.app.state.worker_reservations = registry
        monkeypatch.setattr(api, "_gateway_managed_config", lambda _: gateway)
        cancelled = asyncio.Event()

        async def generate(socket, binding):
            binding.cleaned = False
            try:
                await asyncio.Event().wait()
            finally:
                binding.cleaned = True
                cancelled.set()

        monkeypatch.setattr(api, "generate_bound_child", generate)
        task = asyncio.create_task(bound.bound_video(ws))
        assert (await ws.receive())["player_expires_at"] == deadline
        ws.submit(
            dict(
                type="bound.open",
                operation_id="one",
                kind="generate",
                session_id="child",
                generation_id="g1",
            )
        )
        await ws.receive()
        for _ in range(3):
            ws.submit(dict(type="bound.data", operation_id="one", payload=b"queued"))
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert cancelled.is_set()
        assert (await registry.snapshot())["active_sessions"] == 1
        assert registry._reservations["t"].cleanup_state == "cleaned_pending_commit"

    asyncio.run(run())


def test_cancel_is_read_while_child_has_queued_source_messages(service, monkeypatch):
    async def run():
        ws = Socket()
        cancelled = asyncio.Event()
        gateway = GatewayManagedConfig("p", "g", "t", "a", "", "", "", "")
        registry = NS(
            mark_blocked=AsyncMock(),
            quarantine=AsyncMock(),
            player_deadline=AsyncMock(return_value=None),
        )
        monkeypatch.setattr(api, "_gateway_managed_config", lambda _: gateway)
        monkeypatch.setattr(
            api, "_consume_gateway_reservation", AsyncMock(return_value=registry)
        )
        monkeypatch.setattr(api, "_release_gateway_reservation", AsyncMock())

        async def generate(socket, binding):
            binding.cleaned = False
            try:
                await asyncio.Event().wait()
            finally:
                binding.cleaned = True
                cancelled.set()

        monkeypatch.setattr(api, "generate_bound_child", generate)
        task = asyncio.create_task(bound.bound_video(ws))
        await ws.receive()
        ws.submit(
            dict(
                type="bound.open",
                operation_id="one",
                kind="generate",
                session_id="s",
                generation_id="g1",
            )
        )
        await ws.receive()
        for _ in range(8):
            ws.submit(
                dict(type="bound.data", operation_id="one", payload=b"queued-source")
            )
        ws.submit(dict(type="bound.cancel", operation_id="one"))
        await asyncio.wait_for(cancelled.wait(), 1)
        assert (await ws.receive())["cleaned"] is True
        ws.incoming.put_nowait(None)
        await task
        registry.quarantine.assert_not_awaited()

    asyncio.run(run())
