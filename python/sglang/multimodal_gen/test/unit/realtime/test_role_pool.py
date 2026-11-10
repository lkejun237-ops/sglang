# SPDX-License-Identifier: Apache-2.0
"""Real CPU control contracts: Redis transactions and native HTTP group lifecycle."""

import asyncio
import copy
import socket
import subprocess
import time
from contextlib import contextmanager

import httpx
import pytest
from fastapi import FastAPI

from sglang.multimodal_gen.runtime.realtime.role_pool.api import install_pool_routes
from sglang.multimodal_gen.runtime.realtime.role_pool.contracts import (
    Credentials,
    PoolError,
)
from sglang.multimodal_gen.runtime.realtime.role_pool.scheduler import GroupController
from sglang.multimodal_gen.runtime.realtime.role_pool.store import (
    MemoryRecords,
    RedisRecords,
    RolePoolStore,
)
from sglang.multimodal_gen.runtime.realtime.role_pool.worker import (
    ManagedService,
    ProcessAuthority,
)
from sglang.multimodal_gen.runtime.realtime.worker_connections import (
    ServiceConnections,
    service_descriptor,
)

CREATED_AT = time.time()
SECRET = "local-contract-signing-secret-32-characters"


def declaration(role, name, count=1, capacity=1):
    ports = {
        "encoder": dict(http=31000, input_ws=32000, rendezvous=33000),
        "dit": dict(http=34000, zmq=35000, rendezvous=36000),
        "decoder": dict(http=37000, zmq=38000, output_ws=39000),
    }[role]
    return dict(
        process_id=name,
        boot_epoch=name + "-boot",
        control_epoch="control-1",
        pool_id="test",
        role=role,
        compatibility="model-v1",
        network_domain="zone-a",
        capacity={"compute": capacity},
        profiles={"480": {"compute": 1}},
        profile_limits={
            "480": dict(
                source_sizes=[[32, 32]],
                target_sizes=[[32, 32]],
                max_fps=24,
                max_references=3,
                max_reference_bytes=1048576,
                max_reference_pixels=1048576,
                max_prompt_chars=16000,
                max_preview_pixels=1048576,
            )
        },
        hardware_ids=[name + "-gpu"],
        services=[
            service_descriptor(
                role, name, i, name, **{k: v + i for k, v in ports.items()}
            )
            for i in range(count)
        ],
    )


async def register(store, value):
    await store.register(value)
    await store.heartbeat(
        value["process_id"],
        value["boot_epoch"],
        sequence=0,
        state="ready",
        observations={s["service_id"]: {"state": "idle"} for s in value["services"]},
    )


def request(user="a"):
    return dict(
        admission_id="admit-" + user,
        user_id=user,
        player_id="player-" + user,
        generation_id="generation-" + user,
        profile="480",
        compatibility="model-v1",
        network_domain="zone-a",
        ttl=1800,
        created_at=CREATED_AT,
    )


def sdk_receipt(a):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.player_lifecycle import (
        sdk_cleanup_receipt,
    )

    parent = SimpleNamespace(
        token=a["allocation_id"],
        session_id=a["intent"]["player_id"],
        generation_id=a["intent"]["generation_id"],
    )
    return {
        **sdk_cleanup_receipt(
            parent,
            state="retired",
            source_id="source-a",
            owner_epoch="room-epoch",
            view_id="view-a",
        ),
        "player_id": a["intent"]["player_id"],
    }


async def populated(records=None):
    store = RolePoolStore(records or MemoryRecords(), "test", "control-1")
    await store.initialize()
    values = [
        declaration("encoder", "e", 3, 2),
        declaration("decoder", "d", 3, 3),
        *[declaration("dit", "t" + str(i)) for i in range(3)],
    ]
    for value in values:
        await register(store, value)
    return store, values


def test_three_services_do_not_oversell_shared_process_budget():
    async def run():
        store, values = await populated()
        results = await asyncio.gather(
            *(store.acquire(**request(str(i))) for i in range(20)),
            return_exceptions=True,
        )
        success = [r for r in results if isinstance(r, dict)]
        assert len(success) == 2
        assert len({a["topology"]["encoder"]["service_id"] for a in success}) == 2
        rows = await store.db.read([store.key("process", "e")])
        assert rows[store.key("process", "e")]["used"] == {"compute": 2}
        assert len(values[0]["services"]) == 3

    asyncio.run(run())


def test_idempotency_epoch_and_drain_fences():
    async def run():
        store, values = await populated()
        a = await store.acquire(**request())
        assert (await store.acquire(**request()))["allocation_id"] == a["allocation_id"]
        with pytest.raises(PoolError, match="ADMISSION_IDENTITY_CHANGED"):
            await store.acquire(**{**request(), "profile": "736"})
        with pytest.raises(PoolError, match="USER_SESSION_LIMIT"):
            await store.acquire(**{**request(), "admission_id": "other"})
        bad = copy.deepcopy(values[0])
        bad["boot_epoch"] = "new"
        with pytest.raises(PoolError, match="PROCESS_IDENTITY_CHANGED"):
            await store.register(bad)
        await store.drain("e", "e-boot")
        await store.heartbeat(
            "e",
            "e-boot",
            sequence=2,
            state="ready",
            observations={
                s["service_id"]: {"state": "idle"} for s in values[0]["services"]
            },
        )
        with pytest.raises(PoolError, match="CAPACITY_EXHAUSTED"):
            await store.acquire(**request("b"))
        assert (await store.get("admit-a"))["roles"]["encoder"] == "held"

    asyncio.run(run())


def test_stale_heartbeat_removes_candidates_without_releasing_ownership():
    async def run():
        store, _ = await populated()
        a = await store.acquire(**request())
        now = store.clock()
        store.clock = lambda: now + 11
        with pytest.raises(PoolError, match="CAPACITY_EXHAUSTED"):
            await store.acquire(**request("b"))
        assert (await store.get("admit-a"))["allocation_id"] == a["allocation_id"]

    asyncio.run(run())


class Hosts(httpx.AsyncBaseTransport):
    def __init__(self):
        self.apps = {}

    async def handle_async_request(self, request):
        return await httpx.ASGITransport(
            app=self.apps[(request.url.host, request.url.port)]
        ).handle_async_request(request)


async def native_cluster(store, values, *, build=None):
    hosts = Hosts()
    managers = {}
    for value in values:
        process = ProcessAuthority(
            capacity=value["capacity"],
            profiles=value["profiles"],
            control_epoch="control-1",
            secret=SECRET,
            boot_epoch=value["boot_epoch"],
            profile_limits=value["profile_limits"],
        )
        for service in value["services"]:
            native = ServiceConnections(
                service,
                ready=lambda: True,
                build=build or (lambda *_: None),
                destroy=lambda: None,
                rollback=lambda: None,
            )
            managed = native.managed = ManagedService(
                native, process, cleanup_timeout=0.2
            )
            app = FastAPI()
            native.install(app)
            url = httpx.URL(service["endpoints"]["http"])
            hosts.apps[(url.host, url.port)] = app
            managers[service["service_id"]] = managed
    client = httpx.AsyncClient(transport=hosts)
    controller = GroupController(
        store,
        Credentials(SECRET, "control-1"),
        client=client,
        build_timeout=1,
        cleanup_timeout=0.5,
    )
    return controller, managers, client


def test_native_http_group_lifecycle_and_next_player_reuses_capacity():
    async def run():
        store, values = await populated()
        control, managers, client = await native_cluster(store, values)
        async with client:
            a = await control.admit(**request())
            assert a["state"] == "bound"
            assert all(
                managers[s["service_id"]].snapshot()["state"] == "connected"
                for s in a["topology"].values()
            )
            assert all(
                "reservation" not in managers[s["service_id"]].snapshot()["owner"]
                for s in a["topology"].values()
            )
            # A different API replica can renew a committed player's group.
            other = GroupController(
                store, Credentials(SECRET, "control-1"), client=client
            )
            assert (await other.renew("admit-a", a["allocation_id"]))[
                "state"
            ] == "bound"
            closed = await control.close(
                "admit-a", a["allocation_id"], sdk_cleanup=sdk_receipt(a)
            )
            assert closed["state"] == "released"
            assert all(v == "released" for v in closed["roles"].values())
            assert (await control.close("admit-a", a["allocation_id"]))[
                "state"
            ] == "released"
            b = await control.admit(**request("b"))
            assert b["state"] == "bound" and b["group_id"] != a["group_id"]
            await control.close(
                "admit-b", b["allocation_id"], sdk_cleanup=sdk_receipt(b)
            )

    asyncio.run(run())


def test_destroy_unseen_fences_late_reservation():
    async def run():
        store, values = await populated()
        control, managers, client = await native_cluster(store, values)
        async with client:
            a = await store.acquire(**request())
            a = await control.claim("admit-a", a["allocation_id"])
            await control.rpc(a, "encoder", "destroy")
            with pytest.raises(PoolError, match="ALLOCATION_ALREADY_RELEASED"):
                await control.rpc(a, "encoder", "reserve")
            receipt = await control.rpc(a, "encoder", "finalize")
            assert receipt["state"] == "released"

    asyncio.run(run())


def test_native_deadline_cleans_without_control_plane_and_waits_for_final_commit():
    async def run():
        store, values = await populated()
        control, managers, client = await native_cluster(store, values)
        async with client:
            a = await control.admit(**request())
            m = managers[a["topology"]["encoder"]["service_id"]]
            now = m.process.clock()
            m.process.clock = lambda: now + 60
            await m.expire()
            await m.cleanup_task
            assert m.snapshot()["state"] == "cleaned"
            assert m.service_id in m.process.held
            assert (
                m.native.query()[0]["state"] == "available"
            )  # internal primitive only
            # Public legacy admission must be fenced in managed mode.
            url = a["topology"]["encoder"]["endpoints"]["http"]
            assert (await client.get(url + "/get_connection_status")).status_code == 403

    asyncio.run(run())


def test_unsigned_and_wrong_epoch_management_rejected():
    async def run():
        store, values = await populated()
        control, managers, client = await native_cluster(store, values)
        async with client:
            a = await store.acquire(**request())
            a = await control.claim("admit-a", a["allocation_id"])
            endpoint = a["topology"]["encoder"]["endpoints"]["http"]
            response = await client.post(
                endpoint + "/managed/reserve", json=control.body(a, "encoder")
            )
            assert response.status_code == 403
            body = control.body(a, "encoder")
            body["boot_epoch"] = "stale"
            token = control.credentials.sign(
                "reserve", body, expires_at=a["executor_until"]
            )
            response = await client.post(
                endpoint + "/managed/reserve",
                json=body,
                headers={"Authorization": "Bearer " + token},
            )
            assert response.status_code == 409
            assert managers[a["topology"]["encoder"]["service_id"]].state == "idle"

    asyncio.run(run())


@contextmanager
def redis_process(tmp_path):
    import shutil

    binary = shutil.which("redis-server")
    if not binary:
        pytest.skip("Local redis-server is unavailable")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    process = subprocess.Popen(
        [
            binary,
            "--bind",
            "127.0.0.1",
            "--port",
            str(port),
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.02)
        else:
            raise RuntimeError("Local Redis did not start")
        yield port
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_real_redis_concurrent_claims_and_restart_identity_fence(tmp_path):
    with redis_process(tmp_path) as port:

        async def run():
            import redis.asyncio as redis

            async with redis.Redis(
                host="127.0.0.1", port=port, decode_responses=True
            ) as client:
                run_id = (await client.info("server"))["run_id"]
                store, values = await populated(
                    RedisRecords(client, expected_run_id=run_id, retries=32)
                )
                # Separate store/client facades exercise optimistic Redis CAS.
                other = RolePoolStore(
                    RedisRecords(client, expected_run_id=run_id, retries=32),
                    "test",
                    "control-1",
                )
                results = await asyncio.gather(
                    *(
                        (store if i % 2 else other).acquire(**request(str(i)))
                        for i in range(12)
                    ),
                    return_exceptions=True,
                )
                assert sum(isinstance(v, dict) for v in results) == 2
                wrong = RolePoolStore(
                    RedisRecords(client, expected_run_id="old-incarnation"),
                    "test",
                    "control-1",
                )
                with pytest.raises(PoolError, match="CONTROL_STORE_CHANGED"):
                    await wrong.register(declaration("dit", "replacement"))
                assert await client.get(store.key("process", "replacement")) is None

        asyncio.run(run())


def test_shared_admission_gate_prevents_cross_pool_double_binding():
    async def run():
        shared = MemoryRecords()
        first, _ = await populated(shared)
        second = RolePoolStore(
            MemoryRecords(), "other-pool", "control-1", admission_records=shared
        )
        await second.initialize()
        a = await first.acquire(**request())
        with pytest.raises(PoolError, match="USER_SESSION_LIMIT"):
            await second.acquire(**request())
        await first.admissions.verify(a, "test")

    asyncio.run(run())


def test_cancel_before_delayed_admission_cannot_create_a_group():
    async def run():
        store, _ = await populated()
        await store.cancel_intent("admit-a", "a")
        with pytest.raises(PoolError, match="ADMISSION_CANCELLED"):
            await store.acquire(**request())
        with pytest.raises(PoolError, match="ADMISSION_NOT_FOUND"):
            await store.get("admit-a")
        assert (await store.acquire(**request("b")))["state"] == "held"

    asyncio.run(run())


def test_profile_envelope_rejects_unreserved_geometry_and_prompt_work():
    from sglang.multimodal_gen.runtime.realtime.role_pool.contracts import (
        validate_business_message,
    )

    limits = declaration("encoder", "e")["profile_limits"]["480"]
    message = dict(
        type="init",
        size="32x32",
        source_stream=dict(width=32, height=32, fps=24),
        prompt="hello",
    )
    validate_business_message(message, limits)
    with pytest.raises(PoolError, match="PROFILE_TARGET_EXCEEDED"):
        validate_business_message({**message, "size": "1280x736"}, limits)
    with pytest.raises(PoolError, match="PROFILE_PROMPT_EXCEEDED"):
        validate_business_message(
            dict(type="event", kind="prompt.update", payload={"text": "x" * 16001}),
            limits,
        )


def test_partial_build_failure_does_not_publish_bound_or_leak_healthy_roles():
    async def run():
        store, values = await populated()
        control, managers, client = await native_cluster(store, values)

        def fail(*_):
            raise RuntimeError("injected transport setup failure")

        for m in managers.values():
            if m.native.service["role"] == "decoder":
                m.native.build_callback = fail
        async with client:
            with pytest.raises(PoolError, match="GROUP_BUILD_FAILED"):
                await control.admit(**request())
            value = await store.get("admit-a")
            assert value["state"] == "awaiting_sdk"
            assert set(value["roles"].values()) == {"released"}
            assert all(not m.process.held for m in managers.values())
            await control.close(
                "admit-a", value["allocation_id"], sdk_cleanup=sdk_receipt(value)
            )

    asyncio.run(run())


def test_managed_exit_aborts_instead_of_waiting_for_dead_output_reader():
    async def run():
        store, values = await populated()
        control, managers, client = await native_cluster(store, values)
        cancelled = []

        def graceful():
            raise RuntimeError("The disconnected output cannot finish EOS")

        async with client:
            a = await control.admit(**request())
            for role, service in a["topology"].items():
                manager = managers[service["service_id"]]
                manager.native.destroy_callback = graceful
                manager.native.rollback_callback = lambda role=role: cancelled.append(
                    role
                )
            result = await control.close(
                "admit-a", a["allocation_id"], sdk_cleanup=sdk_receipt(a)
            )
            assert result["state"] == "released"
            assert set(cancelled) == {"encoder", "dit", "decoder"}
            assert not any(m.native.restart_required for m in managers.values())

    asyncio.run(run())


def test_capacity_counts_complete_groups_and_drain_does_not_evict_player():
    async def run():
        store, values = await populated()
        control, managers, client = await native_cluster(store, values)
        scope = dict(profile="480", compatibility="model-v1", network_domain="zone-a")
        assert (await store.capacity(**scope))["roles"]["model_group"][
            "free_slots"
        ] == 2
        async with client:
            a = await control.admit(**request())
            await control.drain_process("e", "e-boot")
            assert (await store.capacity(**scope))["roles"]["model_group"][
                "free_slots"
            ] == 0
            assert (await control.renew("admit-a", a["allocation_id"]))[
                "state"
            ] == "bound"
            assert not (await store.process_status("e", "e-boot"))["safe_to_stop"]
            await control.close(
                "admit-a", a["allocation_id"], sdk_cleanup=sdk_receipt(a)
            )
            await store.heartbeat(
                "e",
                "e-boot",
                sequence=1,
                state="draining",
                observations={
                    s["service_id"]: managers[s["service_id"]].snapshot(
                        include_children=False
                    )
                    for s in values[0]["services"]
                },
            )
            assert (await store.process_status("e", "e-boot"))["safe_to_stop"]
            await store.retire_process("e", "e-boot", "operator-confirmed-exit")
            new = declaration("encoder", "replacement", 3, 2)
            new["hardware_ids"] = values[0]["hardware_ids"]
            await register(store, new)
            assert (await store.capacity(**scope))["roles"]["model_group"][
                "free_slots"
            ] == 2

    asyncio.run(run())


def test_manual_exit_proof_releases_only_the_dead_role_and_old_epoch_stays_fenced():
    async def run():
        store, values = await populated()
        a = await store.acquire(**request())
        with pytest.raises(PoolError, match="PROCESS_STILL_LIVE"):
            await store.confirm_stopped("e", "e-boot", "exit-proof", "operator")
        await store.drain("e", "e-boot")
        await store.confirm_stopped("e", "e-boot", "exit-proof", "operator")
        current = await store.get("admit-a")
        assert current["roles"] == dict(encoder="released", dit="held", decoder="held")
        assert (
            current["release_evidence"]["encoder"]["kind"]
            == "operator_confirmed_process_exit"
        )
        await store.retire_process("e", "e-boot", "exit-proof")
        with pytest.raises(PoolError, match="PROCESS_EPOCH_MISMATCH"):
            await store.confirm_stopped("e", "wrong-boot", "exit-proof", "operator")

    asyncio.run(run())


def test_internal_api_separates_registration_room_and_operator_credentials():
    from sglang.multimodal_gen.runtime.realtime.role_pool.api import install_pool_routes

    async def run():
        store, values = await populated()
        control, managers, native_client = await native_cluster(store, values)
        app = FastAPI()
        install_pool_routes(
            app,
            control,
            role_token="r" * 32,
            room_token="m" * 32,
            operator_token="o" * 32,
        )
        async with (
            native_client,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://control"
            ) as client,
        ):
            body = dict(process_id="e", boot_epoch="e-boot")
            assert (
                await client.post(
                    "/v1/role-pool/drain",
                    json=body,
                    headers={"Authorization": "Bearer " + "m" * 32},
                )
            ).status_code == 403
            assert (
                await client.post(
                    "/v1/role-pool/admit",
                    json=request(),
                    headers={"Authorization": "Bearer " + "r" * 32},
                )
            ).status_code == 403
            response = await client.post(
                "/v1/role-pool/capacity",
                json=dict(
                    profile="480", compatibility="model-v1", network_domain="zone-a"
                ),
                headers={"Authorization": "Bearer " + "m" * 32},
            )
            assert (
                response.status_code == 200
                and response.json()["roles"]["model_group"]["free_slots"] == 2
            )
            response = await client.post(
                "/v1/role-pool/drain",
                json=body,
                headers={"Authorization": "Bearer " + "o" * 32},
            )
            assert (
                response.status_code == 200 and response.json()["state"] == "draining"
            )
        disabled = FastAPI()
        install_pool_routes(disabled, control, role_token="r" * 32, room_token="m" * 32)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=disabled), base_url="http://control"
        ) as client:
            assert (
                await client.post(
                    "/v1/role-pool/drain",
                    json=body,
                    headers={"Authorization": "Bearer "},
                )
            ).status_code == 503

    asyncio.run(run())


def test_process_reporter_registers_three_services_over_real_local_http_and_redis(
    tmp_path, monkeypatch
):
    import json

    import uvicorn

    from sglang.multimodal_gen.runtime.realtime.role_pool import reporting
    from sglang.multimodal_gen.runtime.realtime.role_pool.api import install_pool_routes

    with redis_process(tmp_path) as redis_port:

        async def run():
            import redis.asyncio as redis

            async with redis.Redis(
                host="127.0.0.1", port=redis_port, decode_responses=True
            ) as db:
                run_id = (await db.info("server"))["run_id"]
                store = RolePoolStore(
                    RedisRecords(db, expected_run_id=run_id), "test", "control-1"
                )
                await store.initialize()
                controller = GroupController(store, Credentials(SECRET, "control-1"))
                app = FastAPI()
                install_pool_routes(
                    app,
                    controller,
                    role_token="r" * 32,
                    room_token="m" * 32,
                    operator_token="o" * 32,
                )
                sock = socket.socket()
                sock.bind(("127.0.0.1", 0))
                sock.listen(128)
                port = sock.getsockname()[1]
                server = uvicorn.Server(
                    uvicorn.Config(app, log_level="error", lifespan="off")
                )
                task = asyncio.create_task(server.serve(sockets=[sock]))
                while not server.started:
                    await asyncio.sleep(0.01)
                declared = declaration("encoder", "auto-encoder", 3, 2)
                config = {
                    k: declared[k]
                    for k in (
                        "capacity",
                        "profiles",
                        "profile_limits",
                        "hardware_ids",
                        "control_epoch",
                        "pool_id",
                        "compatibility",
                        "network_domain",
                    )
                }
                config.update(
                    role="encoder",
                    service_count=3,
                    coordinator_url=f"http://127.0.0.1:{port}",
                    signing_secret_env="ROLE_TEST_SECRET",
                    role_token_env="ROLE_TEST_TOKEN",
                )
                path = tmp_path / "worker.json"
                path.write_text(json.dumps(config))
                monkeypatch.setenv("ROLE_TEST_SECRET", SECRET)
                monkeypatch.setenv("ROLE_TEST_TOKEN", "r" * 32)
                monkeypatch.setenv("SGLANG_ROLE_POOL_WORKER_CONFIG", str(path))
                managers = [
                    ServiceConnections(
                        s,
                        ready=lambda: True,
                        build=lambda *_: None,
                        destroy=lambda: None,
                        rollback=lambda: None,
                    )
                    for s in declared["services"]
                ]
                key = (str(path.resolve()), "auto-encoder")
                reporter = reporting._REGISTRY[key]
                try:
                    deadline = time.monotonic() + 8
                    proc = None
                    while time.monotonic() < deadline:
                        proc = (
                            await store.db.read([store.key("process", "auto-encoder")])
                        )[store.key("process", "auto-encoder")]
                        if proc and proc["state"] == "ready":
                            break
                        await asyncio.sleep(0.05)
                    assert proc and proc["state"] == "ready"
                    services = await store.db.read(
                        [
                            store.key("service", s["service_id"])
                            for s in declared["services"]
                        ]
                    )
                    assert len(services) == 3 and all(
                        v["allocation_id"] is None for v in services.values()
                    )
                    assert len(proc["manifest"]["services"]) == 3 and proc["manifest"][
                        "capacity"
                    ] == {"compute": 2.0}
                    assert all(
                        m.managed.process is managers[0].managed.process
                        for m in managers
                    )
                finally:
                    reporter.stop.set()
                    await asyncio.to_thread(reporter.thread.join, 4)
                    reporting._REGISTRY.pop(key, None)
                    server.should_exit = True
                    await task
                    await controller.shutdown()

        asyncio.run(run())


def test_uncertain_gpu_cleanup_retains_budget_and_blocks_sibling_admission():
    async def run():
        store, values = await populated()
        control, managers, client = await native_cluster(store, values)
        async with client:
            a = await control.admit(**request())
            bad = managers[a["topology"]["encoder"]["service_id"]]

            def fail():
                raise RuntimeError("Injected CUDA cleanup uncertainty")

            bad.native.rollback_callback = fail
            result = await control.close(
                "admit-a", a["allocation_id"], sdk_cleanup=sdk_receipt(a)
            )
            assert (
                result["state"] == "quarantined"
                and result["roles"]["encoder"] != "released"
            )
            assert result["roles"]["dit"] == result["roles"]["decoder"] == "released"
            assert bad.process.failed and bad.service_id in bad.process.held
            with pytest.raises(PoolError, match="PROCESS_QUARANTINED"):
                bad.process.acquire("unused-service", "another-allocation", "480")

    asyncio.run(run())


def test_sdk_receipt_survives_caller_cancellation_during_gpu_cleanup():
    async def run():
        store, values = await populated()
        control, managers, client = await native_cluster(store, values)
        async with client:
            a = await control.admit(**request())
            managed = managers[a["topology"]["encoder"]["service_id"]]
            entered, finish = asyncio.Event(), asyncio.Event()
            original = managed.native.rollback_callback

            async def slow_cleanup():
                entered.set()
                await finish.wait()
                await managed.native._call(original)

            managed.native.rollback_callback = slow_cleanup
            closing = asyncio.create_task(control.close("admit-a", a["allocation_id"]))
            await asyncio.wait_for(entered.wait(), 2)
            with_sdk = asyncio.create_task(
                control.close("admit-a", a["allocation_id"], sdk_cleanup=sdk_receipt(a))
            )
            async with asyncio.timeout(2):
                while not (await store.get("admit-a")).get("sdk_cleanup_pending"):
                    await asyncio.sleep(0.01)
            with_sdk.cancel()
            await asyncio.gather(with_sdk, return_exceptions=True)
            pending = await store.get("admit-a")
            assert pending["roles"]["encoder"] != "released"
            assert pending["sdk_cleanup_pending"]["state"] == "retired"
            finish.set()
            released = await asyncio.wait_for(closing, 3)
            assert released["state"] == "released"
            assert released["sdk_cleanup"]["view_id"] == "view-a"

    asyncio.run(run())


def test_durable_sdk_receipt_reconciles_after_request_dies_before_close_task():
    async def run():
        store, values = await populated()
        control, _, client = await native_cluster(store, values)
        async with client:
            a = await control.admit(**request())
            pending = await control.close("admit-a", a["allocation_id"])
            assert pending["state"] == "awaiting_sdk"
            await store.record_sdk_cleanup(
                "admit-a", a["allocation_id"], sdk_receipt(a)
            )
            # A concurrent closer may publish awaiting_sdk after its last read.
            await store.update(
                "admit-a",
                a["allocation_id"],
                lambda value: value.update(state="awaiting_sdk"),
            )
            assert len(await store.active_assignments()) == 1
            await control.sweep_once()
            await asyncio.gather(*list(control.recovery_tasks.values()))
            assert (await store.get("admit-a"))["state"] == "released"
            assert not await store.active_assignments()
            with pytest.raises(PoolError, match="SDK_CLEANUP_IDENTITY_CHANGED"):
                await store.record_sdk_cleanup(
                    "admit-a",
                    a["allocation_id"],
                    {**sdk_receipt(a), "view_id": "wrong"},
                )

    asyncio.run(run())


def test_sp_follower_does_not_register_a_second_boot_epoch(monkeypatch):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.role_pool.reporting import (
        attach_managed_service,
    )

    monkeypatch.setenv("SGLANG_ROLE_POOL_WORKER_CONFIG", "/not-read-by-sp-follower")
    monkeypatch.setenv("RANK", "1")
    assert attach_managed_service(SimpleNamespace(service={"role": "dit"})) is None


def test_business_socket_replay_cannot_replace_an_active_or_completed_child():
    async def run():
        store, values = await populated()
        control, managers, client = await native_cluster(store, values)
        async with client:
            a = await control.admit(**request())
            managed = managers[a["topology"]["encoder"]["service_id"]]
            claim = dict(session_id="child", generation_id="generation")
            task = object()
            managed.attach_socket(claim, task)
            with pytest.raises(PoolError, match="BUSINESS_SESSION_ALREADY_ATTACHED"):
                managed.attach_socket(claim, object())
            managed.child_started("child", "generation")
            managed.child_cleaned("child", "generation")
            managed.sockets.remove(task)
            with pytest.raises(PoolError, match="BUSINESS_SESSION_ALREADY_ATTACHED"):
                managed.attach_socket(claim, object())
            await control.close(
                "admit-a", a["allocation_id"], sdk_cleanup=sdk_receipt(a)
            )

    asyncio.run(run())


def test_gateway_does_not_divide_logical_groups_by_physical_role_count():
    import ast
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "runtime/entrypoints/realtime_gateway_server.py"
    )
    tree = ast.parse(path.read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_nonnegative_int", "_ceil_div", "_slot_capacity_summary"}
    ]
    namespace = {"Any": object}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    value = {
        "capacity_unit": "player_group",
        "roles": {"model_group": {"active_sessions": 1, "free_slots": 2}},
    }
    result = namespace["_slot_capacity_summary"](value, online_slot_cost=3)
    assert (result["used_slots"], result["free_slots"], result["total_slots"]) == (
        1,
        2,
        3,
    )
    value.pop("capacity_unit")
    assert (
        namespace["_slot_capacity_summary"](value, online_slot_cost=3)["free_slots"]
        == 0
    )


def test_native_recovery_preserves_terminal_admission_and_never_reserves_on_read():
    async def run():
        store, values = await populated()
        control, managers, client = await native_cluster(store, values)
        async with client:
            a = await control.admit(**request())
            await control.close(
                "admit-a", a["allocation_id"], sdk_cleanup=sdk_receipt(a)
            )
            service = a["topology"]["encoder"]
            managed = managers[service["service_id"]]

            def signed(action, body):
                token = control.credentials.sign(
                    action, body, expires_at=time.time() + 30
                )
                return managed.command(action, body, token)

            identity = await signed("identity", {"control_epoch": "control-1"})
            assert identity["boot_epoch"] == "e-boot"
            result = await signed(
                "recovery",
                dict(control_epoch="control-1", boot_epoch="e-boot", offset=0, limit=1),
            )
            assert (
                result["active"]["state"] == "idle"
                and result["active"]["owner"] is None
            )
            terminal = result["terminals"][0]
            assert terminal["owner"]["admission_id"] == "admit-a"
            assert terminal["owner"]["player_expires_at"] == a["player_expires_at"]
            assert terminal["owner"]["topology"] == a["topology"]
            assert "reservation" not in terminal["owner"]
            assert terminal["receipt"]["allocation_id"] == a["allocation_id"]
            assert result["history_complete"] is False
            assert managed.native.state == "idle" and managed.native.token is None

    asyncio.run(run())


def test_cancel_during_native_build_cannot_late_publish_bound():
    async def run():
        store, values = await populated()
        entered, finish = asyncio.Event(), asyncio.Event()

        async def build(*_):
            entered.set()
            await finish.wait()

        control, managers, client = await native_cluster(store, values, build=build)
        async with client:
            started = asyncio.create_task(control.admit(**request()))
            await entered.wait()
            a = await store.get("admit-a")
            closing = asyncio.create_task(
                control.close("admit-a", a["allocation_id"], sdk_cleanup=sdk_receipt(a))
            )
            while (await store.get("admit-a"))["state"] != "closing":
                await asyncio.sleep(0)
            finish.set()
            results = await asyncio.gather(started, closing, return_exceptions=True)
            assert isinstance(results[0], PoolError)
            current = await store.get("admit-a")
            assert current["state"] == "released"
            assert set(current["roles"].values()) == {"released"}
            assert all(m.state == "idle" for m in managers.values())

    asyncio.run(asyncio.wait_for(run(), 5))


def test_new_admission_clears_previous_terminal_retention_ttl():
    from sglang.multimodal_gen.runtime.realtime.role_pool.admission import AdmissionGate

    async def run():
        records = MemoryRecords()
        gate = AdmissionGate(records, "control-1")
        first = request()
        await gate.claim(first, "test")
        await gate.bind(first, "test", "first-allocation")
        await gate.release(first, "test", "first-allocation")
        key = gate.key("a")
        assert key in records.expirations
        second = {**first, "admission_id": "new-admission"}
        await gate.claim(second, "test")
        assert key not in records.expirations
        assert (await records.read([key]))[key]["state"] == "pending"

    asyncio.run(run())


def test_prefill_releases_roles_without_inventing_an_sdk_receipt():
    async def run():
        store, values = await populated()
        controller, managers, client = await native_cluster(store, values)
        async with client:
            a = await controller.admit(**request(), operation="prefill")
            assert a["intent"]["operation"] == "prefill"
            with pytest.raises(PoolError, match="INVALID_BUSINESS_ROLE"):
                await controller.business(
                    "admit-a",
                    a["allocation_id"],
                    role="decoder",
                    session_id="s",
                    generation_id="g",
                )
            claim = await controller.business(
                "admit-a",
                a["allocation_id"],
                role="encoder",
                session_id="s",
                generation_id="g",
            )
            assert claim["body"]["operation"] == "prefill"
            released = await controller.close("admit-a", a["allocation_id"])
            assert released["state"] == "released"
            assert released["sdk_cleanup"] is None
            assert set(released["roles"].values()) == {"released"}
            assert all(not manager.process.held for manager in managers.values())
            next_player = await controller.admit(**request("next"))
            # A real player still cannot bypass its original SDK retirement.
            pending = await controller.close("admit-next", next_player["allocation_id"])
            assert pending["state"] == "awaiting_sdk"
            await controller.close(
                "admit-next",
                next_player["allocation_id"],
                sdk_cleanup=sdk_receipt(next_player),
            )

    asyncio.run(run())


def test_prefill_kind_is_immutable_and_cannot_accept_fake_sdk_cleanup():
    async def run():
        store, values = await populated()
        controller, _, client = await native_cluster(store, values)
        async with client:
            a = await controller.admit(**request(), operation="prefill")
            with pytest.raises(PoolError, match="ADMISSION_IDENTITY_CHANGED"):
                await store.acquire(**request())
            with pytest.raises(PoolError, match="PREFILL_HAS_NO_SDK_OWNER"):
                await controller.close(
                    "admit-a", a["allocation_id"], sdk_cleanup=sdk_receipt(a)
                )
            await controller.close("admit-a", a["allocation_id"])

    asyncio.run(run())


def test_released_prefill_observes_disk_without_owning_or_mutating_next_player():
    async def run():
        store, values = await populated()
        controller, managers, native_client = await native_cluster(store, values)
        app = FastAPI()
        room_token = "room-prefill-observation-secret-32-chars"
        install_pool_routes(
            app,
            controller,
            role_token="role-prefill-observation-secret-32-chars",
            room_token=room_token,
        )
        async with (
            native_client,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://coordinator"
            ) as client,
        ):
            a = await controller.admit(**request(), operation="prefill")
            await controller.close("admit-a", a["allocation_id"])
            dit = managers[a["topology"]["dit"]["service_id"]]
            calls = []
            dit.native.persistence = lambda descriptor: (
                calls.append(descriptor) or {"state": "persisted"}
            )
            next_player = await controller.admit(**request("next"))
            before = copy.deepcopy(dit.owner)
            response = await client.post(
                "/v1/role-pool/persistence",
                headers={"Authorization": "Bearer " + room_token},
                json={
                    "admission_id": "admit-a",
                    "allocation_id": a["allocation_id"],
                    "descriptor": {"cache_key": "a" * 64},
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["state"] == "persisted"
            assert calls == [{"cache_key": "a" * 64}]
            assert dit.owner == before
            assert (await store.get("admit-a"))["state"] == "released"
            body = controller.body(a, "dit")
            body.update(executor_until=time.time() + 20, descriptor={})
            body["parent_generation_id"] = "other-generation"
            token = controller.credentials.sign(
                "persistence", body, expires_at=time.time() + 10
            )
            with pytest.raises(PoolError, match="GROUP_OWNER_MISMATCH"):
                await dit.command("persistence", body, token)
            await controller.close(
                "admit-next",
                next_player["allocation_id"],
                sdk_cleanup=sdk_receipt(next_player),
            )

    asyncio.run(run())


def test_four_case_direct_pool_uses_native_fences_and_releases(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[6] / "benchmark/whitebox"))
    from source_replay_role_pool import SuiteRolePool, current_pool

    async def run():
        # Actual ASGI Worker endpoints, signatures, reserve/build/renew and close.
        values = [
            declaration(role, role, capacity=1)
            for role in ("encoder", "dit", "decoder")
        ]
        for value in values:
            value["profile_limits"]["480"].update(
                source_sizes=[[832, 480]], target_sizes=[[832, 480]], max_references=12
            )
        store = RolePoolStore(MemoryRecords(), "test", "control-1")
        control, managers, client = await native_cluster(store, values)
        for managed in managers.values():
            managed.process.deployment = dict(
                compatibility="model-v1",
                network_domain="zone-a",
                hardware_ids=[managed.service_id],
            )
        monkeypatch.setenv("ROLE_POOL_SIGNING_SECRET", SECRET)
        workers = {v["role"]: v["services"][0]["endpoints"]["http"] for v in values}
        async with client:
            async with SuiteRolePool(
                workers=workers,
                client=client,
            ) as pool:
                assert current_pool.get() is pool
                assert pool.config["compatibility"] == "model-v1"
                assert pool.config["network_domain"] == "zone-a"
                assert pool.config["profile"] == "480"
                assert pool.control_epoch == "control-1"
                assert pool.revision == "model-v1"
                await pool.wait_idle()
                a = await pool.control("admit", request("suite"))
                assert a["state"] == "bound"
                assert all(m.state == "connected" for m in managers.values())
                with pytest.raises(PoolError, match="SERVICE_OCCUPIED"):
                    # A second authority cannot overwrite the first one's owner.
                    manager = next(iter(managers.values()))
                    manager.process.acquire(
                        manager.service_id, "other-allocation", "480"
                    )
                result = await pool.control(
                    "close",
                    dict(
                        admission_id=a["intent"]["admission_id"],
                        allocation_id=a["allocation_id"],
                        sdk_cleanup=sdk_receipt(a),
                    ),
                )
                assert result["state"] == "released"
                await pool.wait_idle()
                assert all(m.state == "idle" for m in managers.values())
            assert current_pool.get() is None
        await control.shutdown()

    asyncio.run(run())


def test_four_case_remote_pool_routes_and_authenticates(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[6] / "benchmark/whitebox"))
    from source_replay_role_pool import SuiteRolePool

    calls = []

    def respond(req):
        calls.append(req)
        if req.method == "GET":
            assert req.url.params["model_revision"] == "test-revision"
            return httpx.Response(200, json={"backend": "role_pool", "profile": "480"})
        assert req.headers["Authorization"] == "Bearer test-room-token"
        import json

        body = json.loads(req.content)
        assert body == {
            "profile": "480",
            "compatibility": "model-v1",
            "network_domain": "test-shanghai-copy",
        }
        return httpx.Response(
            200,
            json={
                "roles": {
                    "model_group": {
                        "free_slots": 1,
                        "active_sessions": 0,
                        "draining_workers": 0,
                    }
                }
            },
        )

    async def run():
        monkeypatch.setenv("SUITE_TEST_ROOM_TOKEN", "test-room-token")
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            async with SuiteRolePool(
                coordinator_url="http://127.0.0.1:18181",
                revision="test-revision",
                compatibility="model-v1",
                room_token_env="SUITE_TEST_ROOM_TOKEN",
                client=client,
            ) as pool:
                await pool.wait_idle()
        assert len(calls) == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    "mismatch",
    [
        "compatibility",
        "network_domain",
        "control_epoch",
        "profile_limits",
        "ambiguous",
        "wrong_secret",
    ],
)
def test_four_case_auto_discovery_fails_before_admission(monkeypatch, mismatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[6] / "benchmark/whitebox"))
    from source_replay_role_pool import SuiteRolePool

    async def run():
        values = [declaration(role, role) for role in ("encoder", "dit", "decoder")]
        for value in values:
            value["profile_limits"]["480"].update(
                source_sizes=[[832, 480]], target_sizes=[[832, 480]], max_references=12
            )
        control, managers, client = await native_cluster(
            RolePoolStore(MemoryRecords(), "test", "control-1"), values
        )
        for managed in managers.values():
            managed.process.deployment = dict(
                compatibility="model-v1",
                network_domain="zone-a",
                hardware_ids=[managed.service_id],
            )
        first = next(iter(managers.values())).process
        if mismatch in {"compatibility", "network_domain"}:
            first.deployment[mismatch] = "other"
        elif mismatch == "control_epoch":
            first.credentials = Credentials(SECRET, "other-epoch")
        elif mismatch == "profile_limits":
            first.profile_limits["480"]["max_references"] = 13
        elif mismatch == "ambiguous":
            for managed in managers.values():
                managed.process.profiles["other"] = dict(
                    managed.process.profiles["480"]
                )
                managed.process.profile_limits["other"] = dict(
                    managed.process.profile_limits["480"]
                )
        monkeypatch.setenv(
            "ROLE_POOL_SIGNING_SECRET",
            "incorrect-secret-at-least-32-characters"
            if mismatch == "wrong_secret"
            else SECRET,
        )
        workers = {v["role"]: v["services"][0]["endpoints"]["http"] for v in values}
        async with client:
            with pytest.raises((ValueError, httpx.HTTPStatusError)):
                async with SuiteRolePool(workers=workers, client=client):
                    pytest.fail("Invalid metadata must not be admitted")
            assert all(m.state == "idle" and m.owner is None for m in managers.values())
        await control.shutdown()

    asyncio.run(run())
