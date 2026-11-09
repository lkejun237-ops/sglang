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


def test_quiesce_keeps_owner_and_transports_until_destroy():
    async def run():
        calls = []
        controller = manager(
            quiesce=lambda: calls.append("finish-session"),
            destroy=lambda: calls.append("disconnect"),
        )
        payload = request(controller)
        await controller.build(payload)
        with pytest.raises(ConnectionRejected):
            await controller.quiesce(dict(worker_group_id="other"))
        assert calls == []
        for _ in range(2):
            reply, status = await controller.quiesce(dict(worker_group_id="a"))
            assert status == 200 and reply["state"] == "quiesced"
        assert calls == ["finish-session"]
        assert controller.group == "a" and controller.topology == topology()
        assert controller.query()[0] == dict(state="reserved")
        with pytest.raises(ConnectionRejected):
            await controller.build(dict(payload, worker_group_id="other"))
        await controller.destroy(dict(worker_group_id="a"))
        await controller.operation
        assert calls == ["finish-session", "disconnect"]
        assert controller.state == "idle" and controller.group is None

    asyncio.run(run())


@pytest.mark.parametrize("fail", [False, True])
def test_destroy_waits_for_quiesce_after_requester_cancellation(fail):
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def quiesce():
            started.set()
            await release.wait()
            if fail:
                raise RuntimeError("private diagnostic")
            calls.append("finished")

        controller = manager(
            quiesce=quiesce, destroy=lambda: calls.append("disconnect")
        )
        await controller.build(request(controller))
        pending = asyncio.create_task(controller.quiesce(dict(worker_group_id="a")))
        await started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert controller.state == "quiescing" and not controller.operation.done()
        closing = asyncio.create_task(controller.destroy(dict(worker_group_id="a")))
        await asyncio.sleep(0)
        assert calls == [] and not closing.done()
        release.set()
        reply, status = await closing
        if fail:
            assert status == 503 and reply["code"] == "QUIESCE_FAILED"
            assert controller.restart_required and controller.state == "failed"
            assert calls == []
            return
        assert status == 202
        await controller.operation
        assert calls == ["finished", "disconnect"] and controller.state == "idle"

    asyncio.run(run())


def test_quiesce_failure_is_sanitized_and_requires_restart():
    async def run():
        def fail():
            raise RuntimeError("private diagnostic")

        controller = manager(quiesce=fail)
        await controller.build(request(controller))
        reply, status = await controller.quiesce(dict(worker_group_id="a"))
        assert status == 503 and reply["code"] == "QUIESCE_FAILED"
        assert "private diagnostic" not in str(reply)
        assert controller.restart_required and controller.query()[1] == 503

    asyncio.run(run())


def test_unsupported_quiesce_preserves_connected_owner():
    async def run():
        controller = manager()
        await controller.build(request(controller))
        with pytest.raises(ConnectionRejected) as error:
            await controller.quiesce(dict(worker_group_id="a"))
        assert error.value.status == 501 and error.value.code == "QUIESCE_UNSUPPORTED"
        assert controller.state == "connected" and controller.group == "a"

    asyncio.run(run())


def test_encoder_quiesce_releases_session_without_disconnecting_transport():
    from types import SimpleNamespace

    from fastapi import FastAPI

    from sglang.multimodal_gen.runtime.realtime.worker_group_encoder import (
        install_encoder_group,
    )

    async def run():
        calls = []
        link = SimpleNamespace(owner=("s", "g"))

        async def drain():
            calls.append("drain-ingress")

        def forward(request):
            assert request.session_id == "s"
            calls.append("cancel-acknowledged")
            link.owner = None

        app = FastAPI()
        app.state.realtime_execution_client = object()
        app.state.connection_adapter = SimpleNamespace(
            connection_loop=asyncio.get_running_loop(), drain_connection=drain
        )
        executor = SimpleNamespace(dit_link=link, forward=forward)
        with ThreadPoolExecutor(max_workers=1) as pool:
            controller = install_encoder_group(
                app, executor, pool, topology()["encoder"], None
            )
            controller.group, controller.state = "a", "connected"
            assert (await controller.quiesce(dict(worker_group_id="a")))[1] == 200
            assert calls == ["drain-ingress", "cancel-acknowledged"]
            assert executor.dit_link is link and link.owner is None

    asyncio.run(run())


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
            assert (await client.post("/quiesce_group", json={})).status_code == 503
            controller = manager()
            controller.install(app)
            assert (
                await client.get("/get_connection_status?reserve=false")
            ).json() == {"state": "available"}
            assert controller.state == "idle" and controller.token is None
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


def test_completed_quiesce_does_not_overwrite_concurrent_fault():
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        rollbacks = []

        async def quiesce():
            started.set()
            await release.wait()

        controller = manager(quiesce=quiesce, rollback=lambda: rollbacks.append(True))
        await controller.build(request(controller))
        pending = asyncio.create_task(controller.quiesce(dict(worker_group_id="a")))
        await started.wait()
        controller.failed()
        release.set()
        body, status = await pending
        assert status == 503 and body["restart_required"]
        assert controller.query()[0]["state"] == "unavailable"
        # The active owner must finish before any transport can be torn down.
        assert rollbacks == []
        with pytest.raises(ConnectionRejected):
            await controller.destroy(dict(worker_group_id="a"))

    asyncio.run(run())


@pytest.mark.parametrize("reserved", [False, True])
def test_nonreserving_query_leaves_idle_and_expired_reservations_available(reserved):
    now = [0.0]
    controller = manager(clock=lambda: now[0])
    if reserved:
        controller.query()
        now[0] = controller.RESERVATION_SECONDS + 1
    for _ in range(2):
        assert controller.query(reserve=False) == (dict(state="available"), 200)
        assert controller.state == "idle" and controller.token is None
    reply, status = controller.query()
    assert status == 200 and reply["reservation_token"]
    token = controller.token
    assert controller.query(reserve=False) == (dict(state="reserved"), 200)
    assert controller.token == token


@pytest.mark.parametrize(
    "phase",
    [
        "validation",
        "queued",
        "opening",
        "invalid",
        "cancelled",
        "queued_cancelled",
        "opening_cancelled",
        "opening_fault",
    ],
)
def test_encoder_initialization_is_ordered_before_quiesce_and_destroy(
    monkeypatch, phase
):
    import threading
    from types import SimpleNamespace

    from fastapi import FastAPI

    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_encoder_dit_adapter import (
        ZingEncoderDiTAdapter,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_source_encoder_adapter import (
        ZingSourceEncoderAdapter,
    )
    from sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server import (
        EncoderClient,
    )
    from sglang.multimodal_gen.runtime.realtime.worker_group_encoder import (
        install_encoder_group,
    )

    async def run():
        loop = asyncio.get_running_loop()
        entered, resume = asyncio.Event(), asyncio.Event()
        thread_resume, queue_started = threading.Event(), threading.Event()
        calls = []
        app = FastAPI()
        app.state.realtime_execution_client = object()
        executor = SimpleNamespace(
            owner=None, encoder=None, conditioning=SimpleNamespace(dispose=lambda: None)
        )

        class Link:
            owner = None

            def open(self, message):
                calls.append(("open", message["session_id"], controller.state))
                if (
                    phase in ("opening", "opening_cancelled", "opening_fault")
                    and message["session_id"] == "first"
                ):
                    loop.call_soon_threadsafe(entered.set)
                    assert thread_resume.wait(timeout=5)
                self.owner = (message["session_id"], message["generation_id"])
                if phase == "opening_fault" and message["session_id"] == "first":
                    raise RuntimeError("private open failure")
                return []

            def disconnect(self):
                assert self.owner is None
                calls.append(("disconnect",))

        executor.dit_link = Link()

        def cancel(request):
            assert request.session_id == executor.dit_link.owner[0]
            calls.append(("cancel", request.session_id))
            executor.dit_link.owner = None

        executor.forward = cancel

        async def initialize(self, session, request, **kwargs):
            # This is the real parent's asynchronous prompt-token validation
            # boundary; no model work or remote owner has been accepted yet.
            if session.id == "first" and phase in (
                "validation",
                "invalid",
                "cancelled",
            ):
                entered.set()
                await resume.wait()
                if phase == "invalid":
                    raise ValueError("Invalid prompt")
            self.config = dict(width=32, height=32, fps=24, total_frames=None)
            self.source_geometry = SimpleNamespace(
                target_width=32, target_height=32, receipt=lambda: {}
            )
            self.ready = {}
            # Match the parent initialization contract without enabling either
            # optional protocol in this lifecycle-only test.
            self.condition_switch_protocol = None
            self.startup_enabled = False
            self.end = None
            self.ledger = SimpleNamespace(closed=False, accepted_frames=0)
            self.ledger.dispose = lambda: setattr(self.ledger, "closed", True)

        monkeypatch.setattr(ZingSourceEncoderAdapter, "on_init", initialize)

        class Pool(ThreadPoolExecutor):
            signal_next = False

            def submit(self, callback, *args, **kwargs):
                result = super().submit(callback, *args, **kwargs)
                if self.signal_next:
                    self.signal_next = False
                    loop.call_soon_threadsafe(entered.set)
                return result

        with Pool(max_workers=1) as pool:
            controller = install_encoder_group(
                app, executor, pool, topology()["encoder"], None
            )
            controller.state, controller.group = "connected", "a"
            client = EncoderClient(executor, pool)
            request = SimpleNamespace(seed=42, realtime_output_format="raw")
            session = SimpleNamespace(
                id="first", generation_id="g", execution_client=client
            )
            adapter = ZingEncoderDiTAdapter()
            adapter.connection_app = app
            if phase in ("queued", "queued_cancelled"):

                def block_queue():
                    queue_started.set()
                    assert thread_resume.wait(timeout=5)

                pool.submit(block_queue)
                assert queue_started.wait(timeout=5)
                pool.signal_next = True
            initializing = asyncio.create_task(adapter.on_init(session, request))
            await asyncio.wait_for(entered.wait(), timeout=5)
            closing = asyncio.create_task(controller.quiesce(dict(worker_group_id="a")))
            await asyncio.sleep(0)
            assert controller.state == "quiescing"
            assert not closing.done()
            if phase in ("cancelled", "queued_cancelled", "opening_cancelled"):
                initializing.cancel()
            else:
                resume.set()
            thread_resume.set()
            try:
                if phase == "opening":
                    await initializing
                elif phase in ("cancelled", "queued_cancelled", "opening_cancelled"):
                    with pytest.raises(asyncio.CancelledError):
                        await initializing
                elif phase == "opening_fault":
                    with pytest.raises(RuntimeError, match="private open failure"):
                        await initializing
                elif phase == "invalid":
                    with pytest.raises(ValueError, match="Invalid prompt"):
                        await initializing
                else:
                    with pytest.raises(ConditionError) as error:
                        await initializing
                    assert error.value.code == "CONDITION_BUSY"
                reply, status = await asyncio.wait_for(closing, timeout=5)
                if phase == "opening_fault":
                    assert status == 503 and reply["code"] == "QUIESCE_FAILED"
                    assert "private open failure" not in str(reply)
                    assert controller.restart_required and controller.state == "failed"
                    assert executor.dit_link.owner is None
                    assert calls == [
                        ("open", "first", "connected"),
                        ("cancel", "first"),
                    ]
                    with pytest.raises(ConnectionRejected):
                        await controller.destroy(dict(worker_group_id="a"))
                    return
                assert status == 200 and reply["state"] == "quiesced"
                assert executor.dit_link.owner is None
                assert not controller.restart_required
                if phase in ("opening", "opening_cancelled"):
                    assert calls == [
                        ("open", "first", "connected"),
                        ("cancel", "first"),
                    ]
                else:
                    assert calls == []
                if hasattr(adapter, "ledger"):
                    assert adapter.ledger.closed
                await controller.destroy(dict(worker_group_id="a"))
                await controller.operation
                assert controller.state == "idle" and executor.dit_link is None

                # Rebuild only the transport fixture; use the same real control
                # manager, model-owner queue and EncoderClient for the next init.
                controller.build_callback = lambda *_: setattr(
                    executor, "dit_link", Link()
                )
                reservation, _ = controller.query()
                await controller.build(
                    dict(
                        worker_group_id="b",
                        reservation_token=reservation["reservation_token"],
                        topology=topology(),
                    )
                )
                next_adapter = ZingEncoderDiTAdapter()
                next_adapter.connection_app = app
                await next_adapter.on_init(
                    SimpleNamespace(
                        id="next", generation_id="g2", execution_client=client
                    ),
                    request,
                )
                assert executor.dit_link.owner == ("next", "g2")
                assert calls[-1] == ("open", "next", "connected")
                await controller.quiesce(dict(worker_group_id="b"))
                await controller.destroy(dict(worker_group_id="b"))
                await controller.operation
                assert controller.state == "idle" and not controller.restart_required
            finally:
                resume.set()
                thread_resume.set()
                for task in (initializing, closing):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(initializing, closing, return_exceptions=True)

    asyncio.run(run())
