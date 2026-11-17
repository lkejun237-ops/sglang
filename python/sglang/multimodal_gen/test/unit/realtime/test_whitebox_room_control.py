# SPDX-License-Identifier: Apache-2.0
"""CPU-only service control contracts; no model, cloud or CLI subprocess."""

import asyncio
import base64
import hashlib
import hmac
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from whitebox_player_log_test_utils import player_log_capture as player_log_capture

from sglang.multimodal_gen.runtime.realtime.whitebox.room_control import (
    RoomControlCapabilities,
    install_gateway_room_control,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
    RoomControlRuntime,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
)

SECRET = "room-control-test-secret-" * 2
CALLER = {"serviceId": "wtp", "subject": "intranet:browser-a"}


def request_body(action="createRoom"):
    return {
        "schemaVersion": 1,
        "action": action,
        "createContext": {
            "schemaVersion": 1,
            "operation": {
                "operationId": "op-original",
                "requestId": "request-original",
                "kind": "createRoom",
                "parametersSha256": "a" * 64,
            },
            "frozenPublication": {
                "status": "verified",
                "scope": "world-frozen-publication",
                "world": {"sourceId": "s", "worldId": "w", "publicationId": "p"},
                "playConfiguration": {
                    "playMode": "multiplayer",
                    "multiplayer": {"maxPlayers": 2},
                },
            },
        },
        "videoSpec": {"width": 32, "height": 32, "fps": 24},
        "cliFps": 48,
    }


def ticket(raw, *, now=100, jti="1" * 32, **changes):
    claims = dict(
        v=3,
        aud="whitebox-room-control",
        service_id="wtp",
        sub=CALLER["subject"],
        action=json.loads(raw)["action"],
        request_sha256=hashlib.sha256(raw).hexdigest(),
        iat=now,
        exp=now + 120,
        jti=jti,
    )
    claims.update(changes)

    def encode(data):
        return base64.urlsafe_b64encode(data).decode().rstrip("=")

    payload = encode(json.dumps(claims).encode())
    return (
        payload
        + "."
        + encode(hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).digest())
    )


def test_service_ticket_is_bound_to_purpose_caller_action_bytes_and_nonce():
    raw = json.dumps(request_body()).encode()
    verify = RoomControlCapabilities(SECRET, {"wtp"}, now=lambda: 100)
    for changes in [
        dict(v=2),
        dict(aud="whitebox-rtc"),
        dict(service_id="other"),
        dict(sub=""),
        dict(action="getOperation"),
        dict(exp=100),
        dict(iat=True),
        dict(jti="x"),
        dict(request_sha256="b" * 64),
    ]:
        with pytest.raises(ValueError):
            verify.verify(ticket(raw, **changes), raw=raw, action="createRoom")
    with pytest.raises(ValueError):
        verify.verify(ticket(raw), raw=raw + b" ", action="createRoom")
    assert verify.verify(ticket(raw), raw=raw, action="createRoom") == CALLER
    with pytest.raises(ValueError):
        verify.verify(ticket(raw), raw=raw, action="createRoom")


def test_original_directory_atomically_binds_operation_to_one_random_game():
    async def run():
        store = MemoryRoomStore()
        directories = [RoomDirectory(store, {"a": "http://room"}) for _ in range(4)]
        body = request_body()
        assert await directories[0].control_route(CALLER, body, create=False) is None
        assert store.values == {}
        routes = await asyncio.gather(
            *(d.control_route(CALLER, body, create=True) for d in directories * 5)
        )
        assert all(route == routes[0] for route in routes)
        game = routes[0]["game_id"]
        assert len(game) == 32 and game[12] == "4" and game != "op-original"
        query = request_body("getOperation")
        assert (
            await directories[2].control_route(CALLER, query, create=False) == routes[0]
        )
        query["cliFps"] = 24
        with pytest.raises(ValueError, match="changed"):
            await directories[1].control_route(CALLER, query, create=False)
        stranger = {**CALLER, "subject": "intranet:browser-b"}
        assert await directories[0].control_route(stranger, body, create=False) is None
        epoch = "2" * 32
        await directories[0].touch(game, "a", epoch, end=True)
        with pytest.raises(ValueError, match="ended"):
            await directories[1].control_route(CALLER, body, create=True)
        assert len([k for k in store.values if not k.startswith("control:")]) == 1

    asyncio.run(run())


def test_gateway_returns_confirmed_close_after_original_route_expires():
    async def run():
        store = MemoryRoomStore()
        directory = RoomDirectory(store, {"a": "http://room"})
        body = request_body()
        route = await directory.control_route(CALLER, body, create=True)
        game = route["game_id"]
        epoch = "2" * 32
        execution = dict(gameId=game, ownerEpoch=epoch, publicationId="p")
        invocation = {k: body[k] for k in ("createContext", "videoSpec", "cliFps")}
        await directory.touch(game, "a", epoch)
        with pytest.raises(ValueError, match="ended owner"):
            await directory.confirm_room_cleanup(CALLER, invocation, execution, "a")
        await directory.touch(game, "a", epoch, end=True)
        with pytest.raises(ValueError, match="ended owner"):
            await directory.confirm_room_cleanup(
                CALLER, invocation, dict(execution, ownerEpoch="3" * 32), "a"
            )
        assert await directory.confirm_room_cleanup(
            CALLER, invocation, execution, "a"
        ) == {"recorded": True}
        store.values.pop(game)
        restarted = RoomDirectory(store, {"a": "http://replacement"})
        app = FastAPI()
        install_gateway_room_control(
            app,
            SimpleNamespace(directory=restarted, owners={}),
            verifier=RoomControlCapabilities(SECRET, {"wtp"}, now=lambda: 100),
        )
        raw = json.dumps(dict(request_body("retireRoom"), execution=execution)).encode()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            result = await client.post(
                "/v1/whitebox/room-control",
                content=raw,
                headers={"authorization": "Bearer " + ticket(raw)},
            )
        assert result.status_code == 200
        assert result.json() == {
            "operationId": "op-original",
            "parametersSha256": "a" * 64,
            "status": "succeeded",
            "execution": execution,
            "cleanupComplete": True,
        }

    asyncio.run(run())


def test_gateway_verifies_before_directory_and_forwards_only_original_internal_hop():
    async def run():
        calls = []
        directory = RoomDirectory(MemoryRoomStore(), {"a": "http://room"})

        def upstream(request):
            calls.append(request)
            assert str(request.url) == "http://room/internal/room-control"
            assert request.headers["authorization"] == "Bearer internal"
            forwarded = json.loads(request.content)
            assert forwarded["verifiedCaller"] == CALLER
            assert forwarded["request"] == request_body()
            assert len(forwarded["route"]["game_id"]) == 32
            return httpx.Response(
                200,
                json={
                    "operationId": "op-original",
                    "parametersSha256": "a" * 64,
                    "status": "pending",
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(upstream)
        ) as backend:
            proxy = SimpleNamespace(
                directory=directory,
                owners={"a": "http://room"},
                token="internal",
                client=backend,
            )
            app = FastAPI()
            verifier = RoomControlCapabilities(SECRET, {"wtp"}, now=lambda: 100)
            install_gateway_room_control(app, proxy, verifier=verifier)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client:
                raw = json.dumps(request_body()).encode()
                response = await client.post(
                    "/v1/whitebox/room-control",
                    content=raw,
                    headers={
                        "Authorization": "Bearer " + ticket(raw, service_id="other")
                    },
                )
                assert (
                    response.status_code == 403
                    and not directory.store.values
                    and not calls
                )
                query = json.dumps(request_body("getOperation")).encode()
                response = await client.post(
                    "/v1/whitebox/room-control",
                    content=query,
                    headers={"Authorization": "Bearer " + ticket(query, jti="2" * 32)},
                )
                assert (
                    response.json()["status"] == "unknown"
                    and not directory.store.values
                )
                response = await client.post(
                    "/v1/whitebox/room-control",
                    content=raw,
                    headers={"Authorization": "Bearer " + ticket(raw)},
                )
                assert (
                    response.status_code == 200
                    and response.json()["status"] == "pending"
                )
                assert len(calls) == 1
                response = await client.post(
                    "/v1/whitebox/room-control",
                    content=raw,
                    headers={"Authorization": "Bearer " + ticket(raw)},
                )
                assert response.status_code == 403 and len(calls) == 1

    asyncio.run(run())


@pytest.mark.parametrize("cleanup_error", [ValueError, asyncio.CancelledError])
def test_runtime_create_query_admission_leave_and_retirement_keep_original_owners(
    tmp_path,
    monkeypatch,
    cleanup_error,
):
    from sglang.multimodal_gen.runtime.entrypoints.whitebox_room_server import (
        compose_room_driver,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox import worldkit_room
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
        RoomControlRuntime,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_mock import (
        shared_mock_room,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime

    async def run():
        starts = []
        gate = asyncio.Event()

        async def factory(**kwargs):
            starts.append(kwargs)
            await gate.wait()
            kwargs.pop("create_context")
            kwargs.pop("verified_caller")
            kwargs.pop("cli_fps")
            return await shared_mock_room(**kwargs)

        async def legacy_factory(**kwargs):
            pytest.fail("Controlled creation must not use the legacy factory")

        monkeypatch.setattr(worldkit_room, "worldkit_room", factory)
        directory = RoomDirectory(MemoryRoomStore(), {"a": "http://room"})
        runtime = RoomRuntime(
            directory=directory,
            owner_id="a",
            driver_factory=compose_room_driver(legacy_factory),
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        control = RoomControlRuntime(runtime)
        runtime.control = control
        body = request_body()
        route = await directory.control_route(CALLER, body, create=True)

        async def invoke(body):
            return await control.handle(CALLER, body, route)

        try:
            results = await asyncio.gather(*(invoke(body) for _ in range(8)))
            assert all(v["status"] == "pending" for v in results)
            await asyncio.sleep(0)
            assert len(starts) == 1
            assert starts[0]["create_context"] == body["createContext"]
            assert (await invoke(request_body("getOperation")))["status"] == "pending"
            gate.set()
            await control.drain()
            outcome = await invoke(request_body("getOperation"))
            assert outcome["status"] == "succeeded"
            assert outcome["ownerEpoch"] == runtime.epoch
            assert not runtime.players  # Room creation consumes no player slot.
            binding = {
                "gameId": outcome["gameId"],
                "ownerEpoch": runtime.epoch,
                "publicationId": "p",
            }
            admission = dict(
                schemaVersion=1,
                action="admitPlayer",
                binding=binding,
                admission={
                    "operationId": "claim-a",
                    "requestId": "claim-request",
                    "parametersSha256": "b" * 64,
                },
            )
            await invoke(admission)
            await control.drain()
            claimed = await invoke({**admission, "action": "getAdmission"})
            assert claimed["status"] == "claimed" and claimed["memberId"]
            await invoke(admission)
            assert len(runtime.players) == 1
            with pytest.raises(ValueError):
                await control.handle(
                    {**CALLER, "subject": "stranger"},
                    {**admission, "action": "getAdmission"},
                    route,
                )
            member = {
                **binding,
                "memberId": claimed["memberId"],
                "admissionOperationId": "claim-a",
            }
            leave = dict(
                schemaVersion=1,
                action="leavePlayer",
                memberBinding=member,
                operation={
                    "operationId": "leave-a",
                    "requestId": "leave-request",
                    "kind": "leave",
                    "parametersSha256": "c" * 64,
                },
            )
            entry = runtime.players[next(iter(runtime.players))]
            original_close = entry["engine"].close
            cleanup_calls = []
            cleanup_started, cleanup_gate = asyncio.Event(), asyncio.Event()

            async def transient_close():
                cleanup_calls.append(entry)
                if len(cleanup_calls) == 1:
                    cleanup_started.set()
                    await cleanup_gate.wait()
                    raise cleanup_error("temporary cleanup failure")
                await original_close()

            entry["engine"].close = transient_close
            assert (await invoke(leave))["status"] == "blocked"
            await cleanup_started.wait()
            assert (await invoke(leave))["status"] == "blocked"
            assert len(cleanup_calls) == 1  # Concurrent replay shares one cleanup.
            cleanup_gate.set()
            await control.drain()
            query = {
                **admission,
                "action": "getAdmission",
                "leaveOperationId": "leave-a",
            }
            assert (await invoke(query))["status"] == "blocked"
            assert len(cleanup_calls) == 1  # Observation cannot restart cleanup.
            assert (await invoke(leave))["status"] == "blocked"
            await control.drain()
            retired = await invoke(query)
            assert retired == {
                "admissionOperationId": "claim-a",
                "parametersSha256": "b" * 64,
                "status": "retired",
                "cleanupComplete": True,
                "leaveOperationId": "leave-a",
            }
            assert len(cleanup_calls) == 2 and cleanup_calls[0] is cleanup_calls[1]
            assert await invoke(leave) == retired  # Lost replies replay the receipt.
            assert len(cleanup_calls) == 2
            assert not runtime.players and outcome["gameId"] in runtime.rooms
            close = {**request_body("retireRoom"), "execution": binding}
            assert (await invoke(close))["cleanupComplete"] is False
            await control.drain()
            result = await invoke(close)
            assert result["status"] == "succeeded" and result["cleanupComplete"] is True
            assert not runtime.rooms and len(starts) == 1
        finally:
            await runtime.close()

    asyncio.run(run())


def test_cleanup_absence_is_not_an_original_owner_receipt():
    from unittest.mock import AsyncMock

    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
        RoomControlRuntime,
    )

    async def run():
        completed = asyncio.get_running_loop().create_future()
        completed.set_result(None)
        runtime = SimpleNamespace(
            rooms={},
            retiring={},
            locks={},
            epoch="a" * 32,
            leave=AsyncMock(return_value=False),
            _end=AsyncMock(return_value=True),
        )
        control = RoomControlRuntime(runtime)
        creation = dict(
            task=completed,
            game_id="b" * 32,
            status="succeeded",
            invocation=request_body(),
        )
        await control.retire(creation)
        assert creation["retire_status"] == "unknown"
        runtime._end.assert_not_called()
        admission = dict(
            task=completed,
            stream_id="c" * 32,
            status="claimed",
            binding={"gameId": "b" * 32},
            member_id="original-member",
        )
        await control.leave(admission)
        assert admission["status"] == "blocked"

    asyncio.run(run())


@pytest.mark.parametrize("replacement", [False, True])
def test_retirement_rechecks_original_owner_after_waiting_for_lifecycle_lock(
    tmp_path, replacement
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
        RoomControlRuntime,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_mock import (
        shared_mock_room,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime

    async def run():
        async def factory(**_):
            return await shared_mock_room()

        directory = RoomDirectory(MemoryRoomStore(), {"a": "http://room"})
        runtime = RoomRuntime(
            directory=directory,
            owner_id="a",
            driver_factory=factory,
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        control = runtime.control = RoomControlRuntime(runtime)
        body = request_body()
        route = await directory.control_route(CALLER, body, create=True)
        await control.handle(CALLER, body, route)
        await control.drain()
        outcome = await control.handle(CALLER, request_body("getOperation"), route)
        close = {
            **request_body("retireRoom"),
            "execution": {
                "gameId": route["game_id"],
                "ownerEpoch": runtime.epoch,
                "publicationId": "p",
            },
        }
        lock = runtime.locks[route["game_id"]]
        await lock.acquire()
        try:
            await control.handle(CALLER, close, route)
            await asyncio.sleep(0)
            # A heartbeat can finish the old owner while control waits for lock.
            await runtime._end(route["game_id"], "owner_lost")
            successor = {"control_record": object()}
            if replacement:
                runtime.rooms[route["game_id"]] = successor
            lock.release()
            await control.drain()
            record = next(iter(control.creates.values()))
            assert record["retire_status"] == (
                "unknown" if replacement else "succeeded"
            )
            if not replacement:
                receipt = await control.handle(CALLER, close, route)
                assert receipt["cleanupComplete"] is True
            if replacement:
                assert runtime.rooms.pop(route["game_id"]) is successor
        finally:
            if lock.locked():
                lock.release()
            await runtime.close()

    asyncio.run(run())


def test_runtime_restart_and_unconfirmed_cleanup_never_recreate_or_release(tmp_path):
    from unittest.mock import AsyncMock

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
        RoomControlRuntime,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_mock import (
        shared_mock_room,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime

    async def run():
        directory = RoomDirectory(MemoryRoomStore(), {"a": "http://room"})

        async def factory(**kwargs):
            kwargs.pop("create_context")
            kwargs.pop("verified_caller")
            kwargs.pop("cli_fps")
            return await shared_mock_room(**kwargs)

        runtime = RoomRuntime(
            directory=directory,
            owner_id="a",
            driver_factory=factory,
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        runtime.control = control = RoomControlRuntime(runtime)
        body = request_body()
        route = await directory.control_route(CALLER, body, create=True)
        try:
            await control.handle(CALLER, body, route)
            await control.drain()
            route = await directory.lookup(route["game_id"])
            restarted = RoomControlRuntime(
                SimpleNamespace(owner_id="a", epoch="f" * 32)
            )
            with pytest.raises(ValueError, match="owner"):
                await restarted.handle(CALLER, body, route)
            assert (
                await restarted.handle(CALLER, request_body("getOperation"), route)
            )["status"] == "unknown"
            room = runtime.rooms[route["game_id"]]
            original = room["driver"].close
            room["driver"].close = AsyncMock(side_effect=ValueError("cleanup blocked"))
            target = dict(
                gameId=route["game_id"], ownerEpoch=runtime.epoch, publicationId="p"
            )
            close = {**request_body("retireRoom"), "execution": target}
            await control.handle(CALLER, close, route)
            await control.drain()
            record = next(iter(control.creates.values()))
            assert record["retire_status"] == "unknown"
            result = await control.handle(CALLER, close, route)
            assert result["status"] == "pending" and result["cleanupComplete"] is False
            assert route["game_id"] in runtime.retiring
            room["driver"].close = original
            # A same-operation explicit cleanup retries this retained owner;
            # querying the original creation never starts another Room.
            await control.handle(CALLER, close, route)
            await control.drain()
            result = await control.handle(CALLER, close, route)
            assert result["status"] == "succeeded" and result["cleanupComplete"] is True
            assert route["game_id"] not in runtime.retiring
        finally:
            await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "proof",
    [
        "complete",
        "cancelled-confirmed",
        "unknown",
        "wrong-operation",
        "room-withdrawn",
        "epoch-changed",
        "log-failed",
        "cancelled",
    ],
)
def test_create_failure_requires_both_original_factory_and_local_log_cleanup(
    tmp_path, monkeypatch, proof
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
        RoomControlRuntime,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import (
        RoomLog,
        RoomRuntime,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_startup import (
        FactoryCleanupReceipt,
        RoomStartupFailure,
        startup_cleanup_error,
    )

    async def run():
        starts = []

        async def factory(**kwargs):
            starts.append(kwargs)
            if proof == "unknown":
                raise ValueError("private upstream location must never escape")
            if proof == "cancelled":
                raise asyncio.CancelledError()
            if proof == "cancelled-confirmed":
                raise startup_cleanup_error(
                    asyncio.CancelledError(), FactoryCleanupReceipt(kwargs["trace_id"])
                )
            if proof == "epoch-changed":
                runtime.epoch = "f" * 32
            if proof == "room-withdrawn":
                await runtime._end(route["game_id"], "owner_lost")
            raise RoomStartupFailure(
                "CLI_STARTUP_FAILED",
                FactoryCleanupReceipt(
                    "other-operation"
                    if proof == "wrong-operation"
                    else kwargs["trace_id"]
                ),
            )

        if proof == "log-failed":

            async def broken_log(*_):
                raise ValueError("local journal cleanup failed")

            monkeypatch.setattr(RoomLog, "finish", broken_log)
        directory = RoomDirectory(MemoryRoomStore(), {"a": "http://room"})
        runtime = RoomRuntime(
            directory=directory,
            owner_id="a",
            driver_factory=factory,
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        publication_gate, publication_started = asyncio.Event(), asyncio.Event()
        if proof == "complete":
            runtime.archives.store = object()

            async def slow_publication(*_, **__):
                publication_started.set()
                await publication_gate.wait()

            runtime.archives._publish = slow_publication
        control = runtime.control = RoomControlRuntime(runtime)
        body = request_body()
        route = await directory.control_route(CALLER, body, create=True)
        try:
            await control.handle(CALLER, body, route)
            await control.drain()
            outcome = await control.handle(CALLER, request_body("getOperation"), route)
            if proof in {"complete", "cancelled-confirmed"}:
                assert outcome == dict(
                    operationId="op-original",
                    parametersSha256="a" * 64,
                    status="failed",
                    cleanupComplete=True,
                    errorCode=(
                        "CLI_STARTUP_CANCELLED"
                        if proof == "cancelled-confirmed"
                        else "CLI_STARTUP_FAILED"
                    ),
                )
                if proof == "complete":
                    await asyncio.wait_for(publication_started.wait(), 1)
                    assert runtime.archives.tasks  # OSS does not own local cleanup.
            else:
                assert outcome["status"] == "unknown"
                assert outcome.get("cleanupComplete") is not True
            assert await control.handle(CALLER, body, route) == outcome
            assert len(starts) == 1
        finally:
            publication_gate.set()
            await runtime.close()

    asyncio.run(run())


def test_exact_wtp_sender_vectors_use_original_bytes_and_digest():
    from pathlib import Path

    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control import (
        validate_request,
    )

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/room-control-wtp-fa979d90.json").read_text()
    )
    for i, vector in enumerate(fixture["requests"]):
        raw = vector["rawBodyUtf8"].encode()
        assert hashlib.sha256(raw).hexdigest() == vector["requestSha256"]
        body = validate_request(json.loads(raw))
        verifier = RoomControlCapabilities(SECRET, {"wtp"}, now=lambda: 100)
        assert (
            verifier.verify(
                ticket(raw, jti=f"{i:032x}"), raw=raw, action=body["action"]
            )
            == CALLER
        )


def test_real_http_hops_share_directory_and_do_not_publish_private_sessions(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_mock import (
        shared_mock_room,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import (
        RoomRuntime,
        create_app,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        HTTPRoomDirectory,
        install_room_routes,
    )

    async def run():
        directory = RoomDirectory(MemoryRoomStore(), {"a": "http://room"})
        coordinator = FastAPI()
        install_room_routes(coordinator, directory, "internal")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(coordinator), base_url="http://coordinator"
        ) as coordinator_client:
            remote = HTTPRoomDirectory(
                coordinator_client, "http://coordinator", "internal"
            )

            async def factory(**kwargs):
                kwargs.pop("create_context")
                kwargs.pop("verified_caller")
                kwargs.pop("cli_fps")
                return await shared_mock_room(**kwargs)

            runtime = RoomRuntime(
                directory=remote,
                owner_id="a",
                driver_factory=factory,
                spool=tmp_path,
                spec=VideoSpec(32, 32),
            )
            app = create_app(runtime, "internal")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://room"
            ) as room_client:
                gateway = FastAPI()
                proxy = SimpleNamespace(
                    directory=remote,
                    owners={"a": "http://room"},
                    token="internal",
                    client=room_client,
                )
                install_gateway_room_control(
                    gateway,
                    proxy,
                    verifier=RoomControlCapabilities(SECRET, {"wtp"}, now=lambda: 100),
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(gateway), base_url="http://gateway"
                ) as client:
                    nonce = 0

                    async def post(body, path="/v1/whitebox/room-control"):
                        nonlocal nonce
                        nonce += 1
                        raw = json.dumps(body).encode()
                        return await client.post(
                            path,
                            content=raw,
                            headers={
                                "Authorization": "Bearer "
                                + ticket(raw, jti=f"{nonce:032x}")
                            },
                        )

                    try:
                        created = await post(request_body())
                        assert (
                            created.status_code == 200
                            and created.json()["status"] == "pending"
                        )
                        await runtime.control.drain()
                        found = await post(
                            request_body("getOperation"),
                            "/backends/minwm/v1/whitebox/room-control",
                        )
                        outcome = found.json()
                        assert outcome["status"] == "succeeded"
                        binding = dict(
                            gameId=outcome["gameId"],
                            ownerEpoch=outcome["ownerEpoch"],
                            publicationId="p",
                        )
                        admission = dict(
                            schemaVersion=1,
                            action="admitPlayer",
                            binding=binding,
                            admission={
                                "operationId": "claim-a",
                                "requestId": "claim-request",
                                "parametersSha256": "b" * 64,
                            },
                        )
                        assert (await post(admission)).status_code == 200
                        await runtime.control.drain()
                        claimed = (
                            await post({**admission, "action": "getAdmission"})
                        ).json()
                        assert claimed["status"] == "claimed"
                        session = await post(
                            dict(
                                schemaVersion=1,
                                action="getPlayerSession",
                                memberBinding={
                                    **binding,
                                    "memberId": claimed["memberId"],
                                    "admissionOperationId": "claim-a",
                                },
                            )
                        )
                        assert session.status_code == 503
                        assert (
                            "127.0.0.1" not in session.text
                            and "accessToken" not in session.text
                        )
                        denied = await room_client.post(
                            "/internal/room-control", json={}
                        )
                        assert denied.status_code == 403
                    finally:
                        await runtime.close()

    asyncio.run(run())


def test_frozen_download_bridge_signs_exact_parts_without_inventory_fallback(tmp_path):
    import os
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "runtime/realtime/whitebox"
    # Resolve only the public package entry, as the deployed bridge does.
    entry = tmp_path / "entry.mjs"
    entry.write_text("")
    package = tmp_path / "node_modules/@worldkit/stream-host"
    package.mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps({"type": "module", "exports": {"./wtp": "./wtp.mjs"}})
    )
    (package / "wtp.mjs").write_text("""
import assert from 'node:assert/strict';
import {createHash, createHmac} from 'node:crypto';
export const DEFAULT_PLATFORM_URL='https://wtp.test';
export async function downloadWtpDelivery(){throw Error('INVENTORY_MUST_NOT_RUN');}
export async function downloadWtpOperationDelivery(input,directory,options){
 assert.equal(input.operation.operationId,'op-original');
 assert.equal(input.expected.receiptSha256,'a'.repeat(64));
 assert.equal(options.token,undefined);
 const nonces=[];
 for(const part of ['receipt','archive']){
  const bytes=Buffer.from(JSON.stringify({schemaVersion:1,operation:input.operation,part}));
  const token=await options.authorizeRequest(bytes);
  const [payload,signature]=token.split('.');
  assert.equal(signature,createHmac('sha256',process.env.WHITEBOX_FROZEN_READ_SECRET).update(payload).digest('base64url'));
  const claims=JSON.parse(Buffer.from(payload,'base64url'));
  assert.equal(claims.aud,'worldkit-frozen-read');
  assert.equal(claims.service_id,'sglang');
  assert.equal(claims.sub,'intranet:browser-a');
  assert.equal(claims.action,'readFrozen');
  assert.equal(claims.request_sha256,createHash('sha256').update(bytes).digest('hex'));
  nonces.push(claims.jti);
 }
 assert.notEqual(nonces[0],nonces[1]);
 return {archive:'private-archive',receipt:{runtimeHash:'c'.repeat(64)},receiptSha256:'a'.repeat(64)};
}
""")
    body = request_body()
    body["createContext"]["frozenPublication"].update(
        evidence={"receiptSha256": "a" * 64, "archiveSha256": "b" * 64},
        runtime={"runtimeHash": "c" * 64},
    )
    request = dict(
        worldId="s/w",
        directory=str(tmp_path),
        timeoutSeconds=5,
        createContext=body["createContext"],
        verifiedCaller=CALLER,
    )
    env = {
        **os.environ,
        "WHITEBOX_CLI_ENTRYPOINT": str(entry),
        "WHITEBOX_FROZEN_READ_SERVICE_ID": "sglang",
        "WHITEBOX_FROZEN_READ_SECRET": SECRET,
        "WORLD_TEST_API_TOKEN": "must-not-use",
    }
    helper = root / "worldkit_delivery.mjs"
    result = subprocess.run(
        ["node", str(helper)],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["receiptSha256"] == "a" * 64
    env.pop("WHITEBOX_FROZEN_READ_SECRET")
    failed = subprocess.run(
        ["node", str(helper)],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
    )
    assert (
        failed.returncode == 1
        and failed.stderr.strip() == "FROZEN_READ_CONFIGURATION_REQUIRED"
    )
    assert not failed.stdout


def test_failed_player_close_remains_owned_across_room_cleanup_retries(tmp_path):
    from unittest.mock import AsyncMock

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_mock import (
        shared_mock_room,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import new_id

    async def run():
        directory = RoomDirectory(MemoryRoomStore(), {"a": "http://room"})
        runtime = RoomRuntime(
            directory=directory,
            owner_id="a",
            driver_factory=shared_mock_room,
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        game, player, stream, session = (new_id() for _ in range(4))
        await directory.resolve(game, "s/w", create=True)
        try:
            await runtime.join(
                dict(
                    game_id=game,
                    player_id=player,
                    stream_id=stream,
                    session_id=session,
                    world_id="s/w",
                    spec=dict(width=32, height=32, fps=24),
                    user_id="a",
                    trace_id=new_id(),
                )
            )
            room = runtime.rooms[game]
            room["control_caller"] = CALLER
            entry = runtime.players[stream]
            original = entry["engine"].close
            entry["engine"].close = AsyncMock(side_effect=ValueError("blocked"))
            # A driver-level ack alone does not prove this player view closed.
            room["driver"].close = AsyncMock(return_value=None)
            with pytest.raises(ValueError, match="blocked"):
                await runtime.leave(stream)
            assert await runtime._end(game, "completed") is False
            assert await runtime._end(game, "completed") is False
            entry["engine"].close = original
            assert await runtime._end(game, "completed") is True
            assert game not in runtime.retiring
        finally:
            await runtime.close()

    asyncio.run(run())


def test_production_composition_preserves_ordinary_room_arguments_and_cleanup(
    tmp_path, monkeypatch
):
    from unittest.mock import AsyncMock

    from sglang.multimodal_gen.runtime.entrypoints.whitebox_room_server import (
        compose_room_driver,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox import worldkit_room
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_mock import (
        shared_mock_room,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime

    async def run():
        spec = VideoSpec(32, 32)
        driver = await shared_mock_room()
        driver.close = AsyncMock(wraps=driver.close)
        calls = []

        async def legacy_factory(*, world_id, spec, trace_id, cli_fps):
            calls.append(
                dict(world_id=world_id, spec=spec, trace_id=trace_id, cli_fps=cli_fps)
            )
            return driver

        async def controlled_factory(**kwargs):
            pytest.fail("Ordinary player input must not select the controlled factory")

        monkeypatch.setattr(worldkit_room, "worldkit_room", controlled_factory)
        runtime = RoomRuntime(
            directory=RoomDirectory(MemoryRoomStore(), {"a": "http://room"}),
            owner_id="a",
            driver_factory=compose_room_driver(legacy_factory),
            spool=tmp_path,
            spec=spec,
        )
        game, player, stream, session = (str(i) * 32 for i in range(1, 5))
        await runtime.directory.resolve(game, "s/w", create=True)
        try:
            identity = await runtime.join(
                dict(
                    game_id=game,
                    world_id="s/w",
                    player_id=player,
                    stream_id=stream,
                    session_id=session,
                    user_id="original-user",
                    trace_id="original-trace",
                    spec=dict(width=32, height=32, fps=24),
                    cli_fps=48,
                    # Only the control owner supplies _start_room kwargs. Incoming
                    # ordinary player fields cannot select a different factory.
                    create_context=request_body()["createContext"],
                )
            )
            assert calls == [
                dict(world_id="s/w", spec=spec, trace_id="original-trace", cli_fps=48)
            ]
            assert runtime.rooms[game]["driver"] is driver
            assert identity["stream_id"] == stream
            assert player in driver.players
            await runtime.leave(stream)
            assert not runtime.rooms and not runtime.players and not driver.players
            driver.close.assert_awaited_once()
        finally:
            await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "route_state", ["ended", "active", "expired", "missing", "wrong-world", "no-owner"]
)
def test_gateway_queries_exact_ended_owner_without_claiming_cleanup(route_state):
    async def run():
        directory = RoomDirectory(MemoryRoomStore(), {"a": "http://room"})
        body = request_body()
        route = await directory.control_route(CALLER, body, create=True)
        game = route["game_id"]
        await directory.touch(game, "a", "2" * 32, end=True)
        stored = await directory.lookup(game)
        if route_state == "missing":
            directory.store.values.pop(game)
        else:
            if route_state in {"active", "expired"}:
                stored["status"] = route_state
            elif route_state == "wrong-world":
                stored["world_id"] = "s/other"
            elif route_state == "no-owner":
                stored["owner_epoch"] = None
            directory.store.values[game] = json.dumps(stored)
        calls = []

        def upstream(request):
            calls.append(request)
            return httpx.Response(200, json={"status": "unknown"})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(upstream)
        ) as backend:
            app = FastAPI()
            install_gateway_room_control(
                app,
                SimpleNamespace(
                    directory=directory,
                    owners={"a": "http://room"},
                    token="internal",
                    client=backend,
                ),
                verifier=RoomControlCapabilities(SECRET, {"wtp"}, now=lambda: 100),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client:
                query = request_body("getOperation")
                raw = json.dumps(query).encode()
                result = await client.post(
                    "/v1/whitebox/room-control",
                    content=raw,
                    headers={"Authorization": "Bearer " + ticket(raw)},
                )
                if route_state == "ended":
                    assert result.json() == {"status": "unknown"}
                    assert len(calls) == 1
                    assert calls[0].url == "http://room/internal/room-control"
                    assert json.loads(calls[0].content)["request"] == query
                else:
                    assert result.json().get("errorCode") != "ROOM_ENDED"
                    if route_state in {"expired", "missing"}:
                        assert result.json()["status"] == "unknown"
                        assert calls == []
                    elif route_state in {"wrong-world", "no-owner"}:
                        assert result.status_code == 409
                        assert calls == []

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure", ["cleaned", "unconfirmed", "unreachable", "restarted"]
)
@pytest.mark.parametrize(
    "path", ["/v1/whitebox/room-control", "/backends/minwm/v1/whitebox/room-control"]
)
def test_gateway_reads_original_failed_room_cleanup_receipt(tmp_path, failure, path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import (
        RoomRuntime,
        create_app,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_startup import (
        FactoryCleanupReceipt,
        startup_cleanup_error,
    )

    async def run():
        starts = []

        async def factory(**kwargs):
            starts.append(kwargs)
            error = ValueError("The original CLI failed to start")
            if failure == "unconfirmed":
                raise error
            raise startup_cleanup_error(
                error, FactoryCleanupReceipt(kwargs["trace_id"])
            )

        directory = RoomDirectory(MemoryRoomStore(), {"a": "http://room"})
        runtime = RoomRuntime(
            directory=directory,
            owner_id="a",
            driver_factory=factory,
            spool=tmp_path / "original",
            spec=VideoSpec(32, 32),
        )
        room_app = create_app(runtime, "internal")
        replacement = None
        calls = []
        try:
            body = request_body()
            # Match the WTP journal's original subject/kind/request identity.
            operation = body["createContext"]["operation"]
            operation["operationId"] = (
                "op-"
                + hashlib.sha256(
                    json.dumps(
                        [CALLER["subject"], "createRoom", operation["requestId"]],
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            )
            route = await directory.control_route(CALLER, body, create=True)
            await runtime.control.handle(CALLER, body, route)
            await runtime.control.drain()
            ended = await directory.lookup(route["game_id"])
            assert ended["status"] == "ended"
            assert ended["owner_epoch"] == runtime.epoch
            assert len(starts) == 1

            if failure == "restarted":
                replacement = RoomRuntime(
                    directory=directory,
                    owner_id="a",
                    driver_factory=factory,
                    spool=tmp_path / "replacement",
                    spec=VideoSpec(32, 32),
                )
                assert replacement.epoch != runtime.epoch
                room_app = create_app(replacement, "internal")

            async def observe(request):
                calls.append(json.loads(request.content))

            def unreachable(request):
                raise httpx.ConnectError(
                    "The original owner is unavailable", request=request
                )

            transport = (
                httpx.MockTransport(unreachable)
                if failure == "unreachable"
                else httpx.ASGITransport(room_app)
            )
            async with httpx.AsyncClient(
                transport=transport, event_hooks={"request": [observe]}
            ) as backend:
                gateway = FastAPI()
                install_gateway_room_control(
                    gateway,
                    SimpleNamespace(
                        directory=directory,
                        owners={"a": "http://room"},
                        token="internal",
                        client=backend,
                    ),
                    verifier=RoomControlCapabilities(SECRET, {"wtp"}, now=lambda: 100),
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(gateway), base_url="http://gateway"
                ) as client:
                    query = {**body, "action": "getOperation"}
                    raw = json.dumps(query).encode()
                    for index in range(1, 3):
                        response = await client.post(
                            path,
                            content=raw,
                            headers={
                                "Authorization": "Bearer "
                                + ticket(raw, jti=f"{index:032x}")
                            },
                        )
                        assert response.status_code == 200
                        expected = {
                            "operationId": operation["operationId"],
                            "parametersSha256": "a" * 64,
                            "status": "unknown",
                        }
                        if failure == "cleaned":
                            expected.update(
                                status="failed",
                                cleanupComplete=True,
                                errorCode="CLI_STARTUP_FAILED",
                            )
                        assert response.json() == expected
            assert len(calls) == 2
            assert all(
                call == {"verifiedCaller": CALLER, "request": query, "route": ended}
                for call in calls
            )
            assert len(starts) == 1  # Queries never restart the failed CLI.
            assert not runtime.rooms and not runtime.players
            if failure == "cleaned":
                _check_wtp_failed_room_receipt(body, response.json())
        finally:
            if replacement is not None:
                await replacement.close()
            await runtime.close()

    asyncio.run(run())


def _check_wtp_failed_room_receipt(body, outcome):
    """Opt-in consumer check against the actual, explicitly selected WTP source."""
    import os
    import subprocess

    root = os.environ.get("WHITEBOX_TEST_WTP_ROOT")
    if not root:
        return
    subprocess.run(
        ["node", "--input-type=module", "-", root],
        input="""
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const root=process.argv[2];
const load=name=>import(pathToFileURL(`${root}/server/${name}.mjs`));
const {createMultiplayerRoomControl}=await load('multiplayer-room-control');
const {createMultiplayerControlOperations}=await load('multiplayer-control-operations');
const {memoryDocuments}=await load('interactive-production-test-helper');
const {body,outcome}=JSON.parse(process.env.WHITEBOX_TEST_ROOM_RECEIPT);
const subject='intranet:browser-a', op=body.createContext.operation;
const operations=createMultiplayerControlOperations({documents:memoryDocuments()});
const begun=await operations.begin({subject,kind:'createRoom',requestId:op.requestId,
 parameters:{selector:body.createContext.frozenPublication.world,videoSpec:body.videoSpec,cliFps:body.cliFps}});
assert.equal(begun.operation.operationId,outcome.operationId);
const identity={subject,operationId:outcome.operationId};
await operations.freeze({...identity,context:{createContext:body.createContext}});
await operations.issue(identity);
await operations.settle({...identity,status:'unknown',errorCode:'RUNTIME_RESULT_UNKNOWN'});
let queries=0;
const forbidden=()=>{throw Error('Observation must not create another Room');};
const service=createMultiplayerRoomControl({operations,serviceId:'wtp',
 content:{prepare:forbidden},runtime:{create:forbidden,getOperation:async()=>{queries++;return outcome;}},
 authorize:async()=>true,normalizeOptions:forbidden});
const result=await service.getOperation(identity);
assert.equal(result.operation.status,'failed');
assert.equal(result.operation.errorCode,'CLI_STARTUP_FAILED');
assert.equal(result.cleanupRequired,undefined);
assert.equal((await operations.readPrivate(identity)).retirementConfirmed,true);
assert.deepEqual(await service.getOperation(identity),result);
assert.equal(queries,1);
""",
        text=True,
        check=True,
        env={
            **os.environ,
            "WHITEBOX_TEST_ROOM_RECEIPT": json.dumps(
                {"body": body, "outcome": outcome}
            ),
        },
    )


def test_ended_admission_requires_original_caller_binding_and_owner():
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
        RoomControlRuntime,
    )

    async def run():
        runtime = SimpleNamespace(owner_id="owner", epoch="2" * 32)
        control = RoomControlRuntime(runtime)
        body = dict(
            action="getAdmission",
            binding=dict(gameId="1" * 32, ownerEpoch=runtime.epoch, publicationId="p"),
            admission=dict(operationId="admit", parametersSha256="b" * 64),
        )
        record = dict(
            binding=body["binding"], admission=body["admission"], status="blocked"
        )
        control.admissions[control.key(CALLER, "admit")] = record
        route = dict(
            game_id="1" * 32,
            owner_id="owner",
            owner_epoch=runtime.epoch,
            status="ended",
        )
        result = await control.handle(CALLER, body, route)
        assert result == dict(
            admissionOperationId="admit",
            parametersSha256="b" * 64,
            status="blocked",
            errorCode="ROOM_ENDED",
        )
        assert record["status"] == "blocked" and "cleanupComplete" not in result
        control.leaves[control.key(CALLER, "leave-original")] = dict(record=record)
        leave_query = await control.handle(
            CALLER, {**body, "leaveOperationId": "leave-original"}, route
        )
        assert leave_query == {**result, "leaveOperationId": "leave-original"}
        for altered in (
            {**body, "binding": {**body["binding"], "publicationId": "other"}},
            {**body, "admission": {**body["admission"], "operationId": "other"}},
        ):
            with pytest.raises(ValueError):
                await control.handle(CALLER, altered, route)
        with pytest.raises(ValueError):
            await control.handle({**CALLER, "subject": "other"}, body, route)
        wrong = await control.handle(CALLER, body, {**route, "owner_epoch": "3" * 32})
        assert wrong.get("errorCode") != "ROOM_ENDED"

    asyncio.run(run())


def test_ended_room_preserves_verified_retirement_after_lost_leave_response():
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
        RoomControlRuntime,
    )

    async def run():
        runtime = SimpleNamespace(owner_id="owner", epoch="2" * 32)
        control = RoomControlRuntime(runtime)
        body = dict(
            action="getAdmission",
            binding=dict(gameId="1" * 32, ownerEpoch=runtime.epoch, publicationId="p"),
            admission=dict(operationId="admit", parametersSha256="b" * 64),
            leaveOperationId="leave-original",
        )
        record = dict(
            binding=body["binding"], admission=body["admission"], status="retired"
        )
        control.admissions[control.key(CALLER, "admit")] = record
        control.leaves[control.key(CALLER, "leave-original")] = dict(record=record)
        route = dict(
            game_id="1" * 32,
            owner_id="owner",
            owner_epoch=runtime.epoch,
            status="ended",
        )
        result = await control.handle(CALLER, body, route)
        assert result == dict(
            admissionOperationId="admit",
            parametersSha256="b" * 64,
            leaveOperationId="leave-original",
            status="retired",
            cleanupComplete=True,
        )
        assert record["status"] == "retired"
        with pytest.raises(ValueError, match="original leave operation"):
            await control.handle(
                CALLER, {**body, "leaveOperationId": "wrong-leave"}, route
            )
        control.leaves[control.key(CALLER, "other-member-leave")] = dict(record={})
        with pytest.raises(ValueError, match="original leave operation"):
            await control.handle(
                CALLER, {**body, "leaveOperationId": "other-member-leave"}, route
            )

    asyncio.run(run())


def test_control_failures_log_original_identity_without_changing_unknown_or_blocked(
    player_log_capture,
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
        RoomControlRuntime,
    )

    async def run():
        async def reserve(record):
            error = ConnectionError("private reservation endpoint/token")
            error.code = "GPU_RESERVATION_FAILED"
            raise error

        runtime = SimpleNamespace(
            member_media=SimpleNamespace(bound_compute=True, reserve=reserve)
        )
        owner = RoomControlRuntime(runtime)
        record = dict(
            stream_id="log-stream-A",
            player_id="log-player-A",
            status="pending",
            binding=dict(
                gameId="log-game", ownerEpoch="log-owner", publicationId="log-pub"
            ),
            admission=dict(operationId="log-admit-A", requestId="log-request-A"),
        )
        await owner.admit(record, {})
        assert record["status"] == "blocked"
        rows = [
            json.loads(row.message)
            for row in player_log_capture.records
            if '"event":"whitebox.player.failed"' in row.message
        ]
        assert len(rows) == 1
        assert rows[0]["operationId"] == "log-admit-A"
        assert rows[0]["gameId"] == "log-game"
        assert rows[0]["error"]["code"] == "GPU_RESERVATION_FAILED"
        assert rows[0]["phase"] == "model_reserve"
        event = next(
            row
            for row in player_log_capture.records
            if '"event":"whitebox.player.failed"' in row.message
        )
        assert event.filename == "room_control_runtime.py"
        assert event.funcName == "admit"
        assert "private reservation" not in json.dumps(rows)

        async def leave(stream):
            raise RuntimeError("private release endpoint/token")

        runtime.leave = leave
        record["task"] = asyncio.get_running_loop().create_future()
        record["task"].set_result(None)
        await owner.leave(record)
        assert record["status"] == "blocked"
        rows = [
            json.loads(row.message)
            for row in player_log_capture.records
            if '"event":"whitebox.player.failed"' in row.message
        ]
        assert len(rows) == 2
        assert rows[1]["stage"] == "member.leave"
        assert rows[1]["error"]["type"] == "RuntimeError"

    asyncio.run(run())


def test_create_and_retire_exceptions_log_without_claiming_cleanup(player_log_capture):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
        RoomControlRuntime,
    )

    async def run():
        async def resolve(*_):
            raise ConnectionError("private directory error")

        runtime = SimpleNamespace(
            locks={}, directory=SimpleNamespace(resolve=resolve), epoch="log-owner"
        )
        owner = RoomControlRuntime(runtime)
        record = dict(
            game_id="log-create-game",
            route=dict(world_id="log-world"),
            status="pending",
            invocation=request_body(),
        )
        await owner.start(record)
        assert record["status"] == "unknown"
        record["task"] = asyncio.get_running_loop().create_future()
        record["task"].set_exception(RuntimeError("private cleanup failure"))
        await owner.retire(record)
        assert record["retire_status"] == "unknown"
        rows = [
            json.loads(row.message)
            for row in player_log_capture.records
            if '"event":"whitebox.player.failed"' in row.message
        ]
        assert [row["stage"] for row in rows] == ["room.start", "room.retire"]
        assert all(row["operationId"] == "op-original" for row in rows)
        assert all(row["gameId"] == "log-create-game" for row in rows)

    asyncio.run(run())


@pytest.mark.parametrize("sink_error", [BlockingIOError(), BrokenPipeError()])
def test_player_log_failure_preserves_member_cleanup(monkeypatch, sink_error):
    from sglang.multimodal_gen.runtime.realtime.whitebox import diagnostics
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
        RoomControlRuntime,
    )

    def unavailable(_):
        raise sink_error

    monkeypatch.setattr(diagnostics, "_emit_player_failure", unavailable)
    cleanup = []

    async def close():
        cleanup.append("closed")

    async def join(_):
        raise ConnectionError("controlled admission failure")

    owner = RoomControlRuntime(SimpleNamespace(join=join))
    record = dict(
        stream_id="sink-failure-stream",
        binding=dict(gameId="sink-failure-game"),
        spec={},
        player_id="sink-player",
        session_id="sink-session",
        caller=dict(subject="test"),
        admission=dict(operationId="sink-admit"),
        status="pending",
        model_binding=SimpleNamespace(close=close),
    )

    async def run():
        await owner.admit(
            record, dict(route=dict(world_id="sink-world"), control_caller={})
        )
        assert record["status"] == "blocked"
        assert record["leaving"] is True
        assert cleanup == ["closed"]

    asyncio.run(run())


def test_member_presence_separates_rtc_connection_from_retained_admission():
    owner = RoomControlRuntime(SimpleNamespace())
    connected = asyncio.Event()
    connected.set()
    peer = SimpleNamespace(connected=connected, closed=False)
    pipeline = SimpleNamespace(
        subscription=SimpleNamespace(closed=False, peer=peer),
        closed=False,
        retention=None,
        rtc_reconnect=None,
    )
    member = dict(binding={"gameId": "room"}, status="claimed", media_pipeline=pipeline)
    owner.admissions = {
        "first": member,
        "other": {**member, "binding": {"gameId": "another"}},
    }
    assert owner.member_presence("room") == dict(
        online=1, reconnecting=0, joining=0, cleaning=0, disconnected=0
    )
    pipeline.subscription = None
    pipeline.retention = object()
    assert owner.member_presence("room")["reconnecting"] == 1
    assert owner.member_presence("room")["online"] == 0
    pipeline.closed = True
    assert owner.member_presence("room")["disconnected"] == 1
    member["leaving"] = True
    assert owner.member_presence("room")["cleaning"] == 1
    member["status"] = "retired"
    assert not any(owner.member_presence("room").values())
