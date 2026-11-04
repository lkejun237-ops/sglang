# SPDX-License-Identifier: Apache-2.0
"""Connection contract without model weights or CUDA initialization."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from sglang.multimodal_gen.runtime.realtime.worker_connections import (
    ConnectionRejected,
    ServiceConnections,
    service_descriptor,
)


def topology():
    return dict(
        encoder=service_descriptor(
            "encoder",
            "enc",
            0,
            "127.0.0.1",
            http=31000,
            input_ws=30000,
            rendezvous=32000,
        ),
        dit=service_descriptor(
            "dit", "dit", 0, "127.0.0.1", http=33000, zmq=34000, rendezvous=35000
        ),
        decoder=service_descriptor(
            "decoder", "dec", 0, "127.0.0.1", http=36000, zmq=37000, output_ws=38000
        ),
    )


def manager(**overrides):
    options = dict(
        ready=lambda: True,
        build=lambda *_: None,
        destroy=lambda: None,
        rollback=lambda: None,
    )
    options.update(overrides)
    return ServiceConnections(topology()["decoder"], **options)


def request(controller, group="a"):
    reply, status = controller.query()
    assert status == 200 and reply["state"] == "available"
    return dict(
        worker_group_id=group,
        reservation_token=reply["reservation_token"],
        topology=topology(),
    )


def test_atomic_query_has_exactly_one_winner():
    controller = manager()
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(lambda _: controller.query(), range(100)))
    assert sum(body["state"] == "available" for body, _ in results) == 1
    assert all(
        "reservation_token" not in body
        for body, _ in results
        if body["state"] == "reserved"
    )


def test_expiry_is_fixed_and_queries_do_not_renew():
    now = [0]
    controller = manager(clock=lambda: now[0])
    old = request(controller)
    now[0] = 9
    assert controller.query()[0] == dict(state="reserved")
    now[0] = 10
    fresh = request(controller)
    assert old["reservation_token"] != fresh["reservation_token"]
    with pytest.raises(ConnectionRejected):
        asyncio.run(controller.build(old))


def test_build_destroy_and_old_group_cannot_destroy_new_group():
    async def run():
        calls = []
        controller = manager(
            build=lambda *_: calls.append("build"),
            destroy=lambda: calls.append("destroy"),
        )
        payload = request(controller)
        assert (await controller.build(payload))[0]["state"] == "connected"
        await controller.build(payload)
        assert calls == ["build"]
        assert controller.query()[0]["worker_group_id"] == "a"
        accepted, status = await controller.destroy(dict(worker_group_id="a"))
        assert status == 202 and accepted["state"] == "cleaning"
        await controller.operation
        await controller.destroy(dict(worker_group_id="a"))
        assert calls == ["build", "destroy"]
        await controller.build(request(controller, "b"))
        with pytest.raises(ConnectionRejected):
            await controller.destroy(dict(worker_group_id="a"))
        assert controller.query()[0]["worker_group_id"] == "b"

    asyncio.run(run())


def test_build_cleanup_stays_reserved_and_token_cannot_be_reused():
    async def run():
        cleaning, release = asyncio.Event(), asyncio.Event()

        async def fail(*_):
            raise RuntimeError("injected transport failure")

        async def rollback():
            cleaning.set()
            await release.wait()

        controller = manager(build=fail, rollback=rollback)
        payload = request(controller)
        task = asyncio.create_task(controller.build(payload))
        await cleaning.wait()
        assert controller.query()[0] == dict(state="reserved")
        release.set()
        response, status = await task
        assert status == 503 and response["state"] == "idle"
        with pytest.raises(ConnectionRejected):
            await controller.build(payload)
        assert controller.query()[0]["state"] == "available"

    asyncio.run(run())


def test_caller_cancel_does_not_interrupt_build_and_service_is_independent():
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def build(*_):
            calls.append("build")
            started.set()
            await release.wait()

        a, b = manager(build=build), manager()
        payload = request(a)
        task = asyncio.create_task(a.build(payload))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert a.query()[0]["state"] == "reserved"
        assert b.query()[0]["state"] == "available"
        duplicate = asyncio.create_task(a.build(payload))
        release.set()
        assert (await duplicate)[1] == 200
        assert calls == ["build"]

    asyncio.run(run())


def test_failure_to_clean_requires_restart():
    async def run():
        def fail(*_):
            raise RuntimeError("injected failure")

        controller = manager(build=fail, rollback=fail)
        response, status = await controller.build(request(controller))
        assert status == 503 and response["restart_required"]
        assert controller.query()[1] == 503
        assert controller.query()[0]["restart_required"]

    asyncio.run(run())


def test_topology_cannot_replace_local_service():
    controller = manager()
    payload = request(controller)
    payload["topology"] = deepcopy(payload["topology"])
    payload["topology"]["decoder"]["service_id"] = "other-service"
    with pytest.raises(ConnectionRejected, match="does not match"):
        asyncio.run(controller.build(payload))
    assert controller.query()[0] == dict(state="reserved")


def test_transport_failure_stays_quarantined_after_cleanup():
    async def run():
        rolled_back = asyncio.Event()

        async def rollback():
            rolled_back.set()

        controller = manager(rollback=rollback)
        await controller.build(request(controller))
        controller.failed()
        body, status = controller.query()
        assert status == 503 and body["restart_required"]
        await rolled_back.wait()
        await asyncio.sleep(0)
        body, status = controller.query()
        assert status == 503 and body["state"] == "unavailable"
        with pytest.raises(ConnectionRejected):
            await controller.build(
                dict(
                    worker_group_id="other",
                    reservation_token="unused",
                    topology=topology(),
                )
            )

    asyncio.run(run())


def test_http_contract_preserves_health_and_does_not_cache_reservations():
    import httpx
    from fastapi import FastAPI

    from sglang.multimodal_gen.runtime.realtime.worker_http import install_health_routes

    async def run():
        app = FastAPI()
        install_health_routes(app, lambda: dict(role="decoder", ready=True))
        controller = manager()
        controller.install(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/liveness")).json() == dict(status="ok")
            assert (await client.get("/health")).json() == dict(
                role="decoder", ready=True
            )
            available = await client.get("/get_connection_status")
            assert available.headers["cache-control"] == "no-store"
            assert available.json()["state"] == "available"
            assert (await client.get("/get_connection_status")).json() == dict(
                state="reserved"
            )
            payload = dict(
                worker_group_id="a",
                reservation_token=available.json()["reservation_token"],
                topology=topology(),
            )
            opened = await client.post("/build_group", json=payload)
            assert opened.status_code == 200 and opened.json()["state"] == "connected"
            assert (await client.get("/get_connection_status")).json()[
                "worker_group_id"
            ] == "a"
            assert (
                await client.post("/destroy_group", json=dict(worker_group_id="old"))
            ).status_code == 409
            accepted = await client.post(
                "/destroy_group", json=dict(worker_group_id="a")
            )
            assert accepted.status_code == 202
            assert accepted.json()["state"] == "cleaning"
            await controller.operation
            assert (await client.post("/build_group", content=b"{")).status_code == 422
            assert (await client.get("/get_connection_status")).json()[
                "state"
            ] == "available"

    asyncio.run(run())


def test_bridge_epoch_changes_on_reconnect_with_same_group():
    import threading

    from sglang.multimodal_gen.runtime.realtime.worker_connection_ports import (
        bridge_epoch,
    )

    class Store:
        def __init__(self):
            self.values = {}
            self.lock = threading.Lock()

        def set(self, key, value):
            with self.lock:
                self.values[key] = value.encode()

        def get(self, key):
            with self.lock:
                return self.values[key]

        def check(self, keys):
            with self.lock:
                return all(key in self.values for key in keys)

        def delete_key(self, key):
            with self.lock:
                self.values.pop(key, None)

    store = Store()
    previous = None
    for _ in range(3):
        with ThreadPoolExecutor(max_workers=2) as executor:
            sender = executor.submit(bridge_epoch, store, "same-group", 0, timeout=1)
            receiver = executor.submit(bridge_epoch, store, "same-group", 1, timeout=1)
            epoch = sender.result()
            assert epoch == receiver.result() and epoch != previous
            previous = epoch


def test_startup_routes_are_unavailable_then_use_the_installed_controller():
    import httpx
    from fastapi import FastAPI

    from sglang.multimodal_gen.runtime.realtime.worker_connections import (
        install_connection_routes,
    )

    async def run():
        app = FastAPI()
        install_connection_routes(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/get_connection_status")).status_code == 503
            assert (await client.post("/build_group", json={})).status_code == 503
            controller = manager()
            controller.install(app)
            assert (await client.get("/get_connection_status")).json()[
                "state"
            ] == "available"
            assert (
                sum(route.path == "/get_connection_status" for route in app.routes) == 1
            )

    asyncio.run(run())


def test_destroy_returns_immediately_and_queries_stay_reserved_until_cleanup_finishes():
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def destroy():
            calls.append("destroy")
            started.set()
            await release.wait()

        controller = manager(destroy=destroy)
        await controller.build(request(controller))
        accepted, status = await controller.destroy(dict(worker_group_id="a"))
        assert status == 202 and accepted == dict(
            success=True, state="cleaning", worker_group_id="a"
        )
        assert controller.state == "draining"
        assert controller.query()[0] == dict(state="reserved")
        await started.wait()
        duplicate, status = await controller.destroy(dict(worker_group_id="a"))
        assert status == 202 and duplicate == accepted
        assert calls == ["destroy"]
        assert controller.query()[0] == dict(state="reserved")
        release.set()
        await controller.operation
        assert controller.query()[0]["state"] == "available"

    asyncio.run(run())


def test_background_cleanup_failure_is_reported_by_query():
    async def run():
        async def destroy():
            raise RuntimeError("injected cleanup failure")

        controller = manager(destroy=destroy)
        await controller.build(request(controller))
        assert (await controller.destroy(dict(worker_group_id="a")))[1] == 202
        await controller.operation
        reply, status = controller.query()
        assert status == 503 and reply["state"] == "unavailable"
        assert reply["restart_required"]

    asyncio.run(run())


def test_connected_waits_for_gpu_completion(monkeypatch):
    import sys
    import threading
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.worker_connection_ports import (
        confirm_bridge_connection,
    )

    recorded, completed = threading.Event(), threading.Event()

    class Event:
        def record(self, stream):
            recorded.set()

        def query(self):
            return completed.is_set()

    class Store:
        values = {"attempt/connection-complete/1": "complete"}

        def set(self, key, value):
            self.values[key] = value

        def check(self, keys):
            return all(key in self.values for key in keys)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            device=lambda _: SimpleNamespace(type="cuda"),
            cuda=SimpleNamespace(Event=Event, current_stream=lambda _: object()),
        ),
    )
    store = Store()
    controller = manager(
        build=lambda *_: confirm_bridge_connection(
            store, "attempt", 0, "cuda:0", timeout=2
        )
    )
    payload = request(controller)

    async def run():
        task = asyncio.create_task(controller.build(payload))
        try:
            assert await asyncio.to_thread(recorded.wait, 1)
            assert controller.state == "connecting"
            assert controller.query()[0]["state"] != "connected"
            assert not task.done()
            assert "attempt/connection-complete/0" not in store.values
        finally:
            completed.set()
        body, status = await task
        assert status == 200 and body["state"] == "connected"

    asyncio.run(run())


def test_connected_requires_current_peer_confirmation():
    from sglang.multimodal_gen.runtime.realtime.worker_connection_ports import (
        confirm_bridge_connection,
    )

    class Store:
        # A previous attempt cannot confirm the new connection.
        values = {"old-attempt/connection-complete/1": "complete"}

        def set(self, key, value):
            self.values[key] = value

        def check(self, keys):
            return all(key in self.values for key in keys)

    rolled_back = []
    controller = manager(
        build=lambda *_: confirm_bridge_connection(
            Store(), "new-attempt", 0, "cpu", timeout=0.03
        ),
        rollback=lambda: rolled_back.append(True),
    )
    body, status = asyncio.run(controller.build(request(controller)))
    assert status == 503 and body["success"] is False
    assert controller.state != "connected"
    assert rolled_back == [True]


def test_successful_destroy_does_not_overwrite_concurrent_fault():
    async def run():
        async def destroy():
            controller.quarantine()

        controller = manager(destroy=destroy)
        await controller.build(request(controller))
        await controller.destroy(dict(worker_group_id="a"))
        body, status = await controller.operation
        assert status == 503 and body["restart_required"]
        assert controller.query()[0]["state"] == "unavailable"

    asyncio.run(run())


def test_completed_build_does_not_overwrite_concurrent_fault():
    def build(*_):
        controller.quarantine()

    controller = manager(build=build)
    body, status = asyncio.run(controller.build(request(controller)))
    assert status == 503 and body["restart_required"]
    assert controller.query()[0]["state"] == "unavailable"
