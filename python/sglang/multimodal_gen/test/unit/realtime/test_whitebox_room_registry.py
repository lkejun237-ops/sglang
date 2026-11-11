"""Real Redis control tests. Every server is disposable and localhost-only."""

import asyncio
import json
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager

import pytest
import redis
import redis.asyncio as aioredis

from sglang.multimodal_gen.runtime.realtime.whitebox.room_registry import RoomRegistry
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    RedisRoomStore,
    RoomDirectory,
    new_id,
    resolve_owner,
)


@contextmanager
def server():
    binary = shutil.which("redis-server")
    if binary is None:
        pytest.fail("redis-server is required for Room registry integration tests")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="room-registry-test-") as path:
        process = subprocess.Popen(
            [
                binary,
                "--bind",
                "127.0.0.1",
                "--port",
                str(port),
                "--dir",
                path,
                "--save",
                "",
                "--appendonly",
                "no",
                "--maxmemory",
                "16mb",
                "--maxmemory-policy",
                "allkeys-lru",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        client = redis.Redis(host="127.0.0.1", port=port)
        try:
            for _ in range(200):
                try:
                    client.ping()
                    break
                except redis.ConnectionError:
                    time.sleep(0.01)
            else:
                raise RuntimeError("Disposable Redis startup failed")
            yield f"redis://127.0.0.1:{port}"
        finally:
            client.close()
            process.terminate()
            process.wait(timeout=5)


def descriptor(name="a", *, rooms=2, views=2):
    return dict(
        owner_id=name,
        owner_epoch=new_id(),
        owner_url=f"http://room-{name}:8092",
        compatibility="test-v1",
        max_rooms=rooms,
        max_views=views,
        media=dict(public_ip=f"192.0.2.{ord(name) - 96}", public_ports=[443]),
        empty=True,
    )


@contextmanager
def fixture():
    with server() as url:

        async def invoke(test):
            client = aioredis.Redis.from_url(url, decode_responses=True)
            store = RedisRoomStore(lambda: client, "test:")
            now = [time.time()]
            registry = RoomRegistry(store, "test-v1", clock=lambda: now[0])
            directory = RoomDirectory(
                store, {}, registry=registry, clock=lambda: now[0]
            )
            try:
                await test(client, store, registry, directory, now)
            finally:
                await client.aclose()

        yield lambda test: asyncio.run(invoke(test))


def test_add_room_without_recreating_directory_or_gateway(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("WHITEBOX_DYNAMIC_ROOMS", "1")
    with fixture() as run:

        async def test(client, store, registry, directory, now):
            a, b = descriptor("a", rooms=1, views=2), descriptor("b")
            await registry.enroll(a, initialize=True)
            proxy = SimpleNamespace(directory=directory, owners={})
            first = await directory.resolve(new_id(), "world", create=True)
            await directory.touch(first["game_id"], "a", a["owner_epoch"])
            assert await resolve_owner(proxy, "a", a["owner_epoch"]) == a["owner_url"]
            with pytest.raises(ValueError, match="CAPACITY"):
                await directory.resolve(new_id(), "world", create=True)
            await registry.enroll(b)
            second = await directory.resolve(new_id(), "world", create=True)
            assert second["owner_id"] == "b"
            assert await resolve_owner(proxy, "b", b["owner_epoch"]) == b["owner_url"]
            assert (await directory.resolve(first["game_id"], "world"))[
                "owner_id"
            ] == "a"
            assert 0 < await client.ttl(store.prefix + registry.key) <= 86400

        run(test)


def test_concurrent_coordinators_cannot_oversubscribe_views():
    with fixture() as run:

        async def test(client, store, registry, directory, now):
            await registry.enroll(descriptor(rooms=8, views=3), initialize=True)
            other = RoomDirectory(store, {}, registry=RoomRegistry(store, "test-v1"))
            results = await asyncio.gather(
                *[
                    (directory if i % 2 else other).resolve(
                        new_id(), "world", create=True
                    )
                    for i in range(20)
                ],
                return_exceptions=True,
            )
            routes = [r for r in results if isinstance(r, dict)]
            assert len(routes) == 3
            assert (await registry.inspect("a"))["reserved_views"] == 3

        run(test)


def test_duplicate_creation_keeps_one_original_reservation():
    with fixture() as run:

        async def test(client, store, registry, directory, now):
            await registry.enroll(descriptor(), initialize=True)
            game = new_id()
            results = await asyncio.gather(
                *(directory.resolve(game, "world", create=True) for _ in range(20))
            )
            assert all(r == results[0] for r in results)
            assert (await registry.inspect("a"))["reserved_rooms"] == 1

        run(test)


def test_multiplayer_reserves_declared_views_and_draining_retains_live_routes():
    from test_whitebox_room_control import CALLER, request_body

    with fixture() as run:

        async def test(client, store, registry, directory, now):
            a = descriptor(rooms=10, views=2)
            await registry.enroll(a, initialize=True)
            route = await directory.control_route(CALLER, request_body(), create=True)
            await directory.touch(route["game_id"], "a", a["owner_epoch"])
            assert (await registry.inspect("a"))["reserved_views"] == 2
            await registry.drain("a", a["owner_epoch"])
            await registry.heartbeat("a", a["owner_epoch"])
            with pytest.raises(ValueError, match="CAPACITY"):
                await directory.resolve(new_id(), "world", create=True)
            assert await directory.owner_origin("a", a["owner_epoch"]) == a["owner_url"]
            await directory.touch(route["game_id"], "a", a["owner_epoch"])
            assert (await registry.inspect("a"))["state"] == "draining"

        run(test)


def test_claimed_game_is_not_freed_by_heartbeat_expiry_or_clock():
    with fixture() as run:

        async def test(client, store, registry, directory, now):
            a = descriptor(rooms=1, views=1)
            await registry.enroll(a, initialize=True)
            route = await directory.resolve(new_id(), "world", create=True)
            await directory.touch(route["game_id"], "a", a["owner_epoch"])
            now[0] += 100
            await registry.heartbeat("a", a["owner_epoch"])
            with pytest.raises(ValueError, match="CAPACITY"):
                await directory.resolve(new_id(), "world", create=True)
            assert (await registry.inspect("a"))["reserved_rooms"] == 1

        run(test)


def test_unclaimed_start_expires_and_late_cli_claim_is_refused():
    with fixture() as run:

        async def test(client, store, registry, directory, now):
            a = descriptor(rooms=1, views=1)
            await registry.enroll(a, initialize=True)
            route = await directory.resolve(new_id(), "world", create=True)
            now[0] += 61
            await registry.heartbeat("a", a["owner_epoch"])
            await directory.resolve(new_id(), "world", create=True)
            with pytest.raises(ValueError, match="EXPIRED"):
                await registry.claim(route["game_id"], "a", a["owner_epoch"])

        run(test)


def test_only_original_confirmed_cleanup_releases_capacity():
    with fixture() as run:

        async def test(client, store, registry, directory, now):
            a = descriptor(rooms=1, views=1)
            await registry.enroll(a, initialize=True)
            route = await directory.resolve(new_id(), "world", create=True)
            game = route["game_id"]
            await directory.touch(game, "a", a["owner_epoch"])
            with pytest.raises(ValueError):
                await directory.release_room(game, "a", a["owner_epoch"])
            with pytest.raises(ValueError):
                await directory.release_room(game, "a", new_id())
            await directory.touch(game, "a", a["owner_epoch"], end=True)
            assert (await registry.inspect("a"))["reserved_rooms"] == 1
            await directory.release_room(game, "a", a["owner_epoch"])
            await directory.release_room(game, "a", a["owner_epoch"])
            assert (await registry.inspect("a"))["reserved_rooms"] == 0
            with pytest.raises(ValueError, match="ended"):
                await directory.resolve(game, "world", create=True)
            await directory.resolve(new_id(), "world", create=True)

        run(test)


def test_registry_loss_does_not_recreate_owners_or_end_live_game():
    with fixture() as run:

        async def test(client, store, registry, directory, now):
            a = descriptor()
            await registry.enroll(a, initialize=True)
            route = await directory.resolve(new_id(), "world", create=True)
            await directory.touch(route["game_id"], "a", a["owner_epoch"])
            await client.delete(store.prefix + registry.key)  # disposable server only
            with pytest.raises(ValueError, match="UNAVAILABLE"):
                await registry.heartbeat("a", a["owner_epoch"])
            with pytest.raises(ValueError, match="UNAVAILABLE"):
                await directory.resolve(new_id(), "world", create=True)
            assert await client.get(store.prefix + registry.key) is None
            await directory.touch(route["game_id"], "a", a["owner_epoch"])

        run(test)


def test_owner_identity_and_media_endpoints_are_immutable():
    with fixture() as run:

        async def test(client, store, registry, directory, now):
            a = descriptor()
            await registry.enroll(a, initialize=True)
            assert (await registry.enroll(a))["owner_epoch"] == a["owner_epoch"]
            for changed in [
                dict(a, owner_epoch=new_id()),
                dict(a, max_views=3),
                dict(a, empty=False),
                dict(a, compatibility="other"),
            ]:
                with pytest.raises(ValueError):
                    await registry.enroll(changed)
            with pytest.raises(ValueError):
                await registry.heartbeat("a", new_id())
            with pytest.raises(ValueError, match="ENDPOINT"):
                await registry.enroll(dict(descriptor("b"), media=a["media"]))

        run(test)


def test_live_websocket_survives_enrollment_and_new_game_reaches_new_room(monkeypatch):
    import httpx
    import msgspec
    from fastapi import FastAPI, WebSocket
    from test_whitebox_rooms import server as http_server
    from websockets.asyncio.client import connect

    from sglang.multimodal_gen.runtime.realtime.whitebox.room_proxy import RoomProxy
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_registration import (
        install_registry_routes,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        HTTPRoomDirectory,
        install_room_routes,
    )

    monkeypatch.setenv("WHITEBOX_DYNAMIC_ROOMS", "1")
    token, admin = "test-internal", "test-operator"
    with fixture() as run:

        async def test(client, store, registry, directory, now):
            descriptors = {n: descriptor(n, rooms=1, views=1) for n in ("a", "b")}
            connects = {"a": 0, "b": 0}

            def room(name):
                app = FastAPI()

                @app.get("/internal/room-owner")
                async def info():
                    return descriptors[name]

                @app.websocket("/internal/pipeline")
                async def stream(ws: WebSocket):
                    assert ws.headers["authorization"] == "Bearer " + token
                    await ws.accept()
                    first = msgspec.msgpack.decode(await ws.receive_bytes())
                    route = await directory.resolve(
                        first["message"]["whitebox"]["game_id"], "world"
                    )
                    assert route["owner_id"] == name
                    connects[name] += 1
                    await directory.touch(
                        route["game_id"], name, descriptors[name]["owner_epoch"]
                    )
                    await ws.send_bytes(
                        msgspec.msgpack.encode(
                            dict(
                                type="session.ready",
                                owner=name,
                                game_id=route["game_id"],
                            )
                        )
                    )
                    try:
                        while True:
                            await ws.send_bytes(await ws.receive_bytes())
                    except Exception:
                        pass

                return app

            coordinator = FastAPI()
            install_room_routes(coordinator, directory, token)
            install_registry_routes(coordinator, registry, token, admin)
            async with (
                http_server(room("a")) as a,
                http_server(room("b")) as b,
                http_server(coordinator) as coord,
            ):
                descriptors["a"]["owner_url"], descriptors["b"]["owner_url"] = a, b
                async with httpx.AsyncClient() as http:
                    remote = HTTPRoomDirectory(http, coord, token)
                    proxy = RoomProxy(remote, {}, token, http, connect)
                    gateway = FastAPI()

                    @gateway.websocket("/play")
                    async def play(ws: WebSocket):
                        await proxy.serve(
                            ws, user_id="test", trace_id="manual-expansion"
                        )

                    async def enroll(url, initialize=False, secret=admin):
                        return await http.post(
                            coord + "/v1/whitebox/room-owners/enroll",
                            json=dict(owner_url=url, initialize=initialize),
                            headers={"Authorization": "Bearer " + secret},
                        )

                    assert (await enroll(a, True, token)).status_code == 403
                    assert (await enroll(a, True)).status_code == 200
                    async with http_server(gateway) as address:
                        async with connect(
                            address.replace("http:", "ws:") + "/play"
                        ) as old:
                            initial = msgspec.msgpack.encode(
                                dict(type="init", whitebox=dict(world_id="world"))
                            )
                            await old.send(initial)
                            first = msgspec.msgpack.decode(await old.recv())
                            assert first["owner"] == "a"
                            assert (await enroll(b)).status_code == 200
                            async with connect(
                                address.replace("http:", "ws:") + "/play"
                            ) as new:
                                await new.send(initial)
                                second = msgspec.msgpack.decode(await new.recv())
                                assert (
                                    second["owner"] == "b"
                                    and second["game_id"] != first["game_id"]
                                )
                                # Same sockets and same service objects: no restarts.
                                for sequence in range(10):
                                    payload = msgspec.msgpack.encode(
                                        dict(type="input", sequence=sequence)
                                    )
                                    await old.send(payload)
                                    await new.send(payload)
                                    assert await old.recv() == payload
                                    assert await new.recv() == payload
                            assert connects == {"a": 1, "b": 1}
                            assert (
                                await remote.owner_origin(
                                    "a", descriptors["a"]["owner_epoch"]
                                )
                                == a
                            )

        run(test)


def test_owner_descriptor_and_heartbeat_do_not_reset_registry(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import httpx
    from fastapi import FastAPI

    from sglang.multimodal_gen.runtime.realtime.whitebox.room_registration import (
        install_owner_registration,
    )

    for name, value in {
        "WHITEBOX_DYNAMIC_ROOMS": "1",
        "WHITEBOX_ROOM_ADVERTISE_URL": "http://room-a:8092",
        "WHITEBOX_ROOM_POOL_COMPATIBILITY": "test-v1",
        "WHITEBOX_ROOM_PIPELINE_CAPACITY": "2",
        "WHITEBOX_NATIVE_RTC_PUBLIC_IP": "192.0.2.1",
        "WHITEBOX_NATIVE_RTC_PUBLIC_PORT": "443",
        "WHITEBOX_NATIVE_RTC_ENDPOINTS": json.dumps(
            [
                {"listen": "0.0.0.0:40000", "public_port": 443},
                {"listen": "0.0.0.0:40001", "public_port": 444},
            ]
        ),
    }.items():
        monkeypatch.setenv(name, value)

    async def run():
        calls = []

        async def upstream(request):
            calls.append(request.url.path)
            return httpx.Response(409, json={"detail": "not enrolled"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            runtime = SimpleNamespace(
                directory=SimpleNamespace(client=client, url="http://coordinator"),
                owner_id="a",
                epoch=new_id(),
                max_rooms=2,
                rooms={},
                retiring={},
                players={},
                retiring_players={},
                pending_startups={},
                player_closures={},
                archives=SimpleNamespace(tasks=set()),
                render_host=SimpleNamespace(sessions={}),
            )
            app = FastAPI()
            install_owner_registration(app, runtime, "internal")
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://room"
                ) as http:
                    assert (await http.get("/internal/room-owner")).status_code == 403
                    reply = await http.get(
                        "/internal/room-owner",
                        headers={"Authorization": "Bearer internal"},
                    )
                    assert reply.status_code == 200
                    assert reply.json()["media"]["public_ports"] == [443, 444]
                    assert reply.json()["empty"]
                    runtime.maintenance = SimpleNamespace(is_playing=lambda: True)
                    assert not (
                        await http.get(
                            "/internal/room-owner",
                            headers={"Authorization": "Bearer internal"},
                        )
                    ).json()["empty"]
                    runtime.maintenance = SimpleNamespace(is_playing=lambda: False)
                    runtime.pending_startups["pending"] = object()
                    assert not (
                        await http.get(
                            "/internal/room-owner",
                            headers={"Authorization": "Bearer internal"},
                        )
                    ).json()["empty"]
                    await asyncio.sleep(0.01)
            assert calls == ["/v1/whitebox/room-owners/heartbeat"]

    asyncio.run(run())


def test_remove_requires_draining_and_confirmed_release():
    with fixture() as run:

        async def test(client, store, registry, directory, now):
            a = descriptor()
            await registry.enroll(a, initialize=True)
            with pytest.raises(ValueError):
                await registry.remove("a", a["owner_epoch"])
            route = await directory.resolve(new_id(), "world", create=True)
            await directory.touch(route["game_id"], "a", a["owner_epoch"])
            await registry.drain("a", a["owner_epoch"])
            with pytest.raises(ValueError):
                await registry.remove("a", a["owner_epoch"])
            await directory.touch(route["game_id"], "a", a["owner_epoch"], end=True)
            await directory.release_room(route["game_id"], "a", a["owner_epoch"])
            assert (await registry.remove("a", a["owner_epoch"]))["state"] == "removed"
            with pytest.raises(ValueError):
                await registry.heartbeat("a", a["owner_epoch"])
            with pytest.raises(ValueError):
                await directory.owner_origin("a", a["owner_epoch"])

        run(test)


def test_confirmed_original_cleanup_can_finish_after_route_lease_expiry():
    with fixture() as run:

        async def test(client, store, registry, directory, now):
            a = descriptor()
            await registry.enroll(a, initialize=True)
            route = await directory.resolve(new_id(), "world", create=True)
            await directory.touch(route["game_id"], "a", a["owner_epoch"])
            now[0] += 45
            with pytest.raises(ValueError):
                await directory.touch(route["game_id"], "a", new_id(), end=True)
            await directory.touch(route["game_id"], "a", a["owner_epoch"], end=True)
            await directory.release_room(route["game_id"], "a", a["owner_epoch"])
            assert (await registry.inspect("a"))["reserved_rooms"] == 0

        run(test)


def test_lost_route_can_only_be_retired_and_all_dynamic_keys_have_ttl():
    with fixture() as run:

        async def test(client, store, registry, directory, now):
            a = descriptor(rooms=1, views=1)
            await registry.enroll(a, initialize=True)
            route = await directory.resolve(new_id(), "world", create=True)
            await directory.touch(route["game_id"], "a", a["owner_epoch"])
            for key in await client.keys("test:*"):
                assert 0 < await client.ttl(key) <= 86400
            await client.delete(store.prefix + route["game_id"])
            with pytest.raises(ValueError):
                await directory.touch(route["game_id"], "a", a["owner_epoch"])
            with pytest.raises(ValueError):
                await directory.touch(route["game_id"], "a", new_id(), end=True)
            await directory.touch(route["game_id"], "a", a["owner_epoch"], end=True)
            assert (await directory.lookup(route["game_id"]))["status"] == "ended"
            await directory.release_room(route["game_id"], "a", a["owner_epoch"])
            assert (await registry.inspect("a"))["reserved_views"] == 0

        run(test)


def test_real_room_runtime_keeps_existing_source_and_releases_only_closed_game(
    tmp_path,
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_mock import (
        shared_mock_room,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime

    with fixture() as run:

        async def test(client, store, registry, directory, now):
            runtimes = {
                name: RoomRuntime(
                    directory=directory,
                    owner_id=name,
                    driver_factory=shared_mock_room,
                    spool=tmp_path / name,
                    spec=VideoSpec(32, 32),
                )
                for name in ("a", "b")
            }
            a = dict(descriptor("a", rooms=1, views=1), owner_epoch=runtimes["a"].epoch)
            b = dict(descriptor("b", rooms=1, views=1), owner_epoch=runtimes["b"].epoch)

            async def join(name):
                route = await directory.resolve(new_id(), "world", create=True)
                assert route["owner_id"] == name
                body = dict(
                    game_id=route["game_id"],
                    world_id="world",
                    player_id=new_id(),
                    stream_id=new_id(),
                    session_id=new_id(),
                    user_id="test-" + name,
                    trace_id="room-expansion-" + name,
                    spec=dict(width=32, height=32, fps=24),
                )
                return route, await runtimes[name].join(body)

            try:
                await registry.enroll(a, initialize=True)
                route_a, player_a = await join("a")
                assert (await runtimes["a"].next_frame(player_a["stream_id"]))[
                    "index"
                ] == 0
                await registry.enroll(b)
                route_b, player_b = await join("b")
                assert (await runtimes["b"].next_frame(player_b["stream_id"]))[
                    "index"
                ] == 0
                assert (await runtimes["a"].next_frame(player_a["stream_id"]))[
                    "index"
                ] == 1
                assert await runtimes["b"]._end(route_b["game_id"], "completed")
                assert (await registry.inspect("b"))["reserved_views"] == 0
                assert (await registry.inspect("a"))["reserved_views"] == 1
                assert (await runtimes["a"].next_frame(player_a["stream_id"]))[
                    "index"
                ] == 2
                assert (await directory.lookup(route_a["game_id"]))["owner_epoch"] == a[
                    "owner_epoch"
                ]
            finally:
                await asyncio.gather(*(r.close() for r in runtimes.values()))

        run(test)


@pytest.mark.skipif(
    not __import__("os").environ.get("WHITEBOX_TEST_CLI_SDK"),
    reason="Requires the pinned SDK and Chromium in the complete Room image",
)
def test_native_cli_keeps_running_when_second_room_is_enrolled(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_room
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore

    sdk = Path(os.environ["WHITEBOX_TEST_CLI_SDK"])

    async def run():
        with (tmp_path / "fixture.log").open("w") as error_log:
            fixture = await asyncio.create_subprocess_exec(
                "node",
                "--import",
                "tsx",
                str(Path(__file__).with_name("whitebox_wtp_fixture.mjs")),
                cwd=sdk,
                stdout=asyncio.subprocess.PIPE,
                stderr=error_log,
            )
            runtimes, pump = {}, None
            try:
                first = await asyncio.wait_for(fixture.stdout.readline(), 120)
                assert first, (tmp_path / "fixture.log").read_text()[-2000:]
                for key, value in {
                    "WHITEBOX_CLI_ENTRYPOINT": str(
                        sdk / "packages/cli/bin/worldkit.mjs"
                    ),
                    "WHITEBOX_CLI_PLATFORM_URL": json.loads(first)["url"],
                    "WORLD_TEST_API_TOKEN": "local-fixture-token",
                    "WHITEBOX_CLI_RENDERING": "software",
                    "WHITEBOX_CLI_SERVICE": "off",
                    "WHITEBOX_CLI_RESOURCE_TRANSPORT": "archive",
                }.items():
                    monkeypatch.setenv(key, value)
                store = MemoryRoomStore()
                registry = RoomRegistry(store, "test-v1")
                directory = RoomDirectory(store, {}, registry=registry)
                runtimes = {
                    name: RoomRuntime(
                        directory=directory,
                        owner_id=name,
                        driver_factory=cli_room,
                        spool=tmp_path / name,
                        spec=VideoSpec(832, 480),
                    )
                    for name in ("a", "b")
                }

                async def start(name):
                    info = dict(
                        descriptor(name, rooms=1, views=1),
                        owner_epoch=runtimes[name].epoch,
                    )
                    await registry.enroll(info, initialize=name == "a")
                    route = await directory.resolve(new_id(), "s/w", create=True)
                    assert route["owner_id"] == name
                    body = dict(
                        game_id=route["game_id"],
                        world_id="s/w",
                        player_id=new_id(),
                        stream_id=new_id(),
                        session_id=new_id(),
                        user_id="local-" + name,
                        trace_id="native-expansion-" + name,
                        spec=dict(width=832, height=480, fps=24),
                    )
                    player = await runtimes[name].join(body)
                    frame = await asyncio.wait_for(
                        runtimes[name].next_frame(player["stream_id"]), 30
                    )
                    assert len(frame["rgb"]) == 832 * 480 * 3
                    return route, player

                route_a, player_a = await asyncio.wait_for(start("a"), 150)
                cli = runtimes["a"].rooms[route_a["game_id"]]["driver"].process
                pid = cli.pid
                frames = []

                async def consume():
                    while True:
                        frame = await runtimes["a"].next_frame(player_a["stream_id"])
                        assert frame is not None
                        frames.append(frame["index"])
                        if len(frames) % 24 == 1:
                            sequence = (len(frames) - 1) // 24
                            await runtimes["a"].event(
                                player_a["stream_id"],
                                "input",
                                dict(
                                    sequence=sequence,
                                    input={
                                        "held_keys": ["KeyW"]
                                        if sequence % 2 == 0
                                        else []
                                    },
                                ),
                            )

                pump = asyncio.create_task(consume())
                route_b, player_b = await asyncio.wait_for(start("b"), 150)
                if pump.done():
                    pump.result()
                assert not pump.done()
                assert cli.pid == pid and cli.returncode is None
                assert len(frames) > 2 and frames == sorted(set(frames))
                assert route_a["game_id"] != route_b["game_id"]
                assert (await registry.inspect("a"))["reserved_views"] == 1
                assert (await registry.inspect("b"))["reserved_views"] == 1
                assert (await runtimes["b"].next_frame(player_b["stream_id"]))[
                    "index"
                ] == 1
            finally:
                if pump is not None:
                    pump.cancel()
                    await asyncio.gather(pump, return_exceptions=True)
                await asyncio.gather(*(r.close() for r in runtimes.values()))
                if fixture.returncode is None:
                    fixture.terminate()
                    try:
                        await asyncio.wait_for(fixture.wait(), 5)
                    except asyncio.TimeoutError:
                        fixture.kill()
                        await fixture.wait()

    asyncio.run(run())
