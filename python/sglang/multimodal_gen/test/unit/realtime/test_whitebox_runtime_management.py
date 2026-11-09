import asyncio
import base64
import hashlib
import hmac
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI

from sglang.multimodal_gen.runtime.realtime.whitebox.runtime_management import (
    install_gateway_management,
    install_room_management,
    run_runtime_operation,
)

SECRET = "runtime-test-secret-" * 3


def capability(operation="status", **changes):
    now = int(time.time())
    claims = dict(
        v=1,
        origin="https://wtp.test",
        aud="whitebox-runtime",
        sub="operator",
        source_id="s",
        world_id="w",
        scope="world",
        operation=operation,
        iat=now,
        exp=now + 120,
        jti="1" * 32,
    )
    claims.update(changes)
    encode = lambda value: base64.urlsafe_b64encode(value).decode().rstrip("=")
    payload = encode(json.dumps(claims).encode())
    return (
        payload
        + "."
        + encode(hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).digest())
    )


def test_gateway_management_binds_world_operation_audience_and_replay():
    async def run():
        calls = []

        def transport(request):
            calls.append(request)
            assert str(request.url) == "http://room/internal/worlds/runtime"
            assert request.headers["authorization"] == "Bearer internal-test"
            assert json.loads(request.content) == {
                "operation": "status",
                "world_id": "s/w",
                "page": 0,
            }
            return httpx.Response(200, json={"type": "world.runtime", "state": "ready"})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport)
        ) as upstream:
            proxy = SimpleNamespace(
                owners={"one": "http://room"}, token="internal-test", client=upstream
            )
            app = FastAPI()
            install_gateway_management(app, proxy, SECRET)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client:
                body = {"source_id": "s", "world_id": "w", "operation": "status"}
                for token in [
                    capability(aud="whitebox-rtc"),
                    capability(operation="remove"),
                    capability(world_id="other"),
                    capability(exp=int(time.time()) - 1),
                ]:
                    bad = await client.post(
                        "/v1/whitebox/runtime",
                        json=body,
                        headers={
                            "Origin": "https://wtp.test",
                            "Authorization": "Bearer " + token,
                        },
                    )
                    assert bad.status_code == 403
                assert calls == []
                headers = {
                    "Origin": "https://wtp.test",
                    "Authorization": "Bearer " + capability(),
                }
                good = await client.post(
                    "/v1/whitebox/runtime", json=body, headers=headers
                )
                assert good.status_code == 200
                replay = await client.post(
                    "/v1/whitebox/runtime", json=body, headers=headers
                )
                assert replay.status_code == 403
                assert len(calls) == 1

    asyncio.run(run())


def test_room_management_rejects_untrusted_arguments_before_running_cli(monkeypatch):
    async def run():
        operation = AsyncMock()
        monkeypatch.setattr(
            "sglang.multimodal_gen.runtime.realtime.whitebox.runtime_management.run_runtime_operation",
            operation,
        )
        app = FastAPI()
        install_room_management(
            app,
            "internal-test",
            SimpleNamespace(owner_id="room", epoch="epoch"),
            AsyncMock(),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://room"
        ) as client:
            body = {"operation": "status", "world_id": "s/w"}
            bad = await client.post("/internal/worlds/runtime", json=body)
            assert bad.status_code in {401, 403}
            for invalid in [
                dict(body, path="/tmp/foreign"),
                dict(body, world_id="s/../w"),
                dict(body, operation="stop"),
                dict(body, page=-1),
            ]:
                response = await client.post(
                    "/internal/worlds/runtime",
                    json=invalid,
                    headers={"Authorization": "Bearer internal-test"},
                )
                assert response.status_code == 400
            operation.assert_not_called()

    asyncio.run(run())


def test_native_official_cache_and_stopped_service_observation(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from uuid import uuid4

    sdk = os.environ.get("WHITEBOX_TEST_CLI_SDK")
    if not sdk:
        import pytest

        pytest.skip("Native maintenance contract requires WHITEBOX_TEST_CLI_SDK")
    monkeypatch.setenv(
        "WHITEBOX_CLI_ENTRYPOINT", str(Path(sdk) / "packages/cli/bin/worldkit.mjs")
    )
    monkeypatch.setenv("WHITEBOX_CLI_GAME_CACHE_DIRECTORY", str(tmp_path / "games"))
    monkeypatch.setenv("WHITEBOX_CLI_SERVICE_NAME", "console-" + uuid4().hex[:12])

    async def run():
        result = await run_runtime_operation("status", "s/w", 0)
        assert result["type"] == "world.runtime"
        assert result["service"]["status"] == "stopped"
        assert result["service"]["runningServices"] == 0
        assert result["cache"]["current"] is None
        assert result["cache"]["objectBytes"] == 0
        assert result["cache"]["verification"] == "existence"
        assert result["machine"]["host"]
        assert "cacheDirectory" not in json.dumps(result)
        assert "accessToken" not in json.dumps(result)
        inspected = await run_runtime_operation("inspect", "s/w", 0)
        assert inspected["cache"]["verification"] == "sha256"
        inventory = await run_runtime_operation("inventory", None, 0)
        assert inventory["state"] == "ready"
        assert inventory["cache"]["entries"] == []
        assert inventory["cache"]["current"] is None

    asyncio.run(run())


def test_production_gateway_and_room_factories_install_management(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_proxy import RoomProxy
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import create_app

    async def run():
        monkeypatch.setenv("WHITEBOX_RTC_AUTH_SECRET", SECRET)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"type": "world.runtime"})
            )
        ) as upstream:
            proxy = RoomProxy(
                AsyncMock(),
                {"one": "http://room"},
                "internal-test",
                upstream,
                AsyncMock(),
            )
            gateway = FastAPI()
            proxy.install(gateway)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(gateway), base_url="http://gateway"
            ) as client:
                result = await client.post(
                    "/v1/whitebox/runtime",
                    json={"source_id": "s", "world_id": "w", "operation": "status"},
                )
                assert result.status_code == 403
        room = create_app(
            SimpleNamespace(owner_id="room", epoch="epoch"), "internal-test"
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(room), base_url="http://room"
        ) as client:
            result = await client.post(
                "/internal/worlds/runtime",
                json={"world_id": "s/w", "operation": "status"},
            )
            assert result.status_code in {401, 403}

    asyncio.run(run())


def test_native_cache_status_compares_current_publication_without_preparing_a_world(
    tmp_path, monkeypatch
):
    import os
    import subprocess
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path
    from uuid import uuid4

    import pytest

    sdk = os.environ.get("WHITEBOX_TEST_CLI_SDK")
    if not sdk:
        pytest.skip("Native maintenance contract requires WHITEBOX_TEST_CLI_SDK")
    manifest_sha = hashlib.sha256(b"current-manifest").hexdigest()
    prefix = "media/test/playable/v1/s/w/pub/" + "b" * 64 + "/"
    metadata = {
        "ready": True,
        "transport": "cdn",
        "location": {
            "schemaVersion": 1,
            "bucket": "world-assets",
            "region": "oss-cn-beijing",
            "sourceId": "s",
            "worldId": "w",
            "publicationId": "pub",
            "sourceManifestSha256": "a" * 64,
            "renditionInputSha256": "b" * 64,
            "manifestSha256": manifest_sha,
            "entryPath": "index.html",
            "keyPrefix": prefix,
        },
        "entryUrl": "https://cdn.example.com/" + prefix + "index.html",
        "manifestUrl": "https://cdn.example.com/" + prefix + "playable-manifest.json",
        "publication": {
            "sourceId": "s",
            "worldId": "w",
            "publicationId": "pub",
            "renditionInputSha256": "b" * 64,
        },
    }
    requests = []
    denied = False

    class Catalog(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            assert self.path == "/api/worlds/s/w/playable-location?transport=cdn"
            assert self.headers.get("Authorization") == "Bearer selection-fixture"
            self.send_response(401 if denied else 200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(metadata).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Catalog)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    platform = f"http://127.0.0.1:{server.server_port}"
    entry = Path(sdk) / "packages/cli/bin/worldkit.mjs"
    monkeypatch.setenv("WHITEBOX_CLI_ENTRYPOINT", str(entry))
    monkeypatch.setenv("WHITEBOX_CLI_GAME_CACHE_DIRECTORY", str(tmp_path / "games"))
    monkeypatch.setenv("WHITEBOX_CLI_PLATFORM_URL", platform)
    monkeypatch.setenv("WHITEBOX_CLI_RESOURCE_TRANSPORT", "cdn")
    monkeypatch.setenv("WHITEBOX_CLI_SERVICE_NAME", "selection-" + uuid4().hex[:12])
    monkeypatch.setenv("WORLD_TEST_API_TOKEN", "selection-fixture")
    script = r"""
import {createRequire} from 'node:module';import {pathToFileURL} from 'node:url';import {writeFile} from 'node:fs/promises';import {createHash} from 'node:crypto';import path from 'node:path';
const require=createRequire(process.env.WHITEBOX_CLI_ENTRYPOINT),{openGameCache}=await import(pathToFileURL(require.resolve('@worldkit/stream-host/game-cache/store')).href);
const directory=process.env.WHITEBOX_CLI_GAME_CACHE_DIRECTORY,cache=await openGameCache(directory),hash=s=>createHash('sha256').update(s).digest('hex');
const identity={platformUrl:process.env.WHITEBOX_CLI_PLATFORM_URL,sourceId:'s',worldId:'w'},tx=await cache.begin(identity);
try{const file={sha256:hash('hello'),bytes:5,path:'world.js',category:'playable'},source=path.join(directory,'..','content');await writeFile(source,'hello');await tx.reserve([file]);await tx.retain(file,source);await tx.commit({...identity,publicationId:'pub',manifestSha256:hash('current-manifest'),sourceManifestSha256:'a'.repeat(64),renditionInputSha256:'b'.repeat(64),files:[file],pinned:false});}finally{await tx.close();}
"""
    try:
        subprocess.run(
            ["node", "--import", "tsx", "--input-type=module", "-e", script],
            cwd=sdk,
            check=True,
            capture_output=True,
            timeout=20,
        )

        async def run():
            nonlocal denied
            current = await run_runtime_operation("status", "s/w")
            assert current["cache"]["current"]["status"] == "complete"
            assert current["cache"]["selection"]["state"] == "current"
            assert (
                current["cache"]["selection"]["selected"]["manifestSha256"]
                == manifest_sha
            )
            metadata["location"]["manifestSha256"] = "c" * 64
            stale = await run_runtime_operation("status", "s/w")
            assert stale["cache"]["current"]["status"] == "complete"
            assert stale["cache"]["selection"]["state"] == "stale"
            denied = True
            unavailable = await run_runtime_operation("status", "s/w")
            assert unavailable["cache"]["selection"]["state"] == "unavailable"
            count = len(requests)
            inventory = await run_runtime_operation("inventory", None)
            assert inventory["state"] == "ready"
            assert len(requests) == count
            assert all(
                value["service"]["status"] == "stopped"
                for value in (current, stale, unavailable)
            )
            assert str(tmp_path) not in json.dumps(current)
            assert "selection-fixture" not in json.dumps(current)

        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_heavy_maintenance_drains_before_play_and_refuses_an_active_player():
    from fastapi import HTTPException

    from sglang.multimodal_gen.runtime.realtime.whitebox.runtime_management import (
        RuntimeMaintenance,
    )

    async def run():
        entered, drained = asyncio.Event(), asyncio.Event()
        maintenance = RuntimeMaintenance(SimpleNamespace(rooms={}, players={}))

        async def hashing():
            async with maintenance.operation("inspect"):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    drained.set()

        task = asyncio.create_task(hashing())
        await entered.wait()
        await maintenance.cancel_for_play()
        assert drained.is_set() and task.done()
        maintenance.is_playing = lambda: True
        try:
            async with maintenance.operation("prepare"):
                raise AssertionError("Active playback cannot share heavy maintenance")
        except HTTPException as error:
            assert error.status_code == 409

    asyncio.run(run())


def test_removal_requires_observed_owner_and_active_prefetch_is_rejected_before_headers(
    tmp_path, monkeypatch
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.prefetch import (
        install_prefetch,
    )

    async def run():
        app = FastAPI()
        runtime = SimpleNamespace(owner_id="room", epoch="epoch", rooms={}, players={})
        prepare = install_prefetch(app, "internal-test")
        install_room_management(app, "internal-test", runtime, prepare)
        app.state.runtime_maintenance.is_playing = lambda: True
        entry = tmp_path / "packages/cli/bin/worldkit.mjs"
        entry.parent.mkdir(parents=True)
        entry.write_text("")
        monkeypatch.setenv("WHITEBOX_CLI_ENTRYPOINT", str(entry))
        monkeypatch.setenv("WHITEBOX_PREFETCH_ENABLED", "1")
        monkeypatch.setenv("WHITEBOX_RESOURCE_TRANSPORT", "cdn")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://room"
        ) as client:
            headers = {"Authorization": "Bearer internal-test"}
            for operation in ["remove", "remove-preview"]:
                response = await client.post(
                    "/internal/worlds/runtime",
                    json={"operation": operation, "world_id": "s/w"},
                    headers=headers,
                )
                assert response.status_code == 400
            response = await client.post(
                "/internal/worlds/runtime",
                json={"operation": "prepare", "world_id": "s/w"},
                headers=headers,
            )
            assert response.status_code == 409
            assert response.headers["content-type"].startswith("application/json")

    asyncio.run(run())


def test_native_cache_management_preserves_shared_pinned_and_live_leased_objects(
    tmp_path, monkeypatch
):
    import os
    from pathlib import Path

    import pytest

    sdk = os.environ.get("WHITEBOX_TEST_CLI_SDK")
    if not sdk:
        pytest.skip("Native maintenance contract requires WHITEBOX_TEST_CLI_SDK")
    entry = Path(sdk) / "packages/cli/bin/worldkit.mjs"
    monkeypatch.setenv("WHITEBOX_CLI_ENTRYPOINT", str(entry))
    monkeypatch.setenv("WHITEBOX_CLI_GAME_CACHE_DIRECTORY", str(tmp_path / "games"))
    monkeypatch.setenv("WHITEBOX_CLI_PLATFORM_URL", "https://wtp.test")
    monkeypatch.setenv(
        "WHITEBOX_CLI_SERVICE_NAME",
        "console-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12],
    )
    script = r"""
import {createRequire} from 'node:module';import {pathToFileURL} from 'node:url';import {writeFile} from 'node:fs/promises';import {createHash} from 'node:crypto';import path from 'node:path';
const require=createRequire(process.env.WHITEBOX_CLI_ENTRYPOINT),{openGameCache}=await import(pathToFileURL(require.resolve('@worldkit/stream-host/game-cache/store')).href);
const directory=process.env.WHITEBOX_CLI_GAME_CACHE_DIRECTORY,cache=await openGameCache(directory),hash=s=>createHash('sha256').update(s).digest('hex');
for(const [worldId,content,pinned] of [['a','hello',false],['b','hello',false],['pinned','fixed',true],['corrupt','clean',false]]){
 const tx=await cache.begin({platformUrl:'https://wtp.test',sourceId:'s',worldId});try{
 const file={sha256:hash(content),bytes:content.length,path:'world.js',category:'playable'},source=path.join(directory,'..',worldId);
 await writeFile(source,content);await tx.reserve([file]);await tx.retain(file,source);
 await tx.commit({platformUrl:'https://wtp.test',sourceId:'s',worldId,publicationId:'p',manifestSha256:hash(worldId),sourceManifestSha256:hash('source'),renditionInputSha256:hash('rendition'),files:[file],pinned});
 }finally{await tx.close();}}
await writeFile(path.join(directory,'objects',hash('clean')),'wrong');
const held=await cache.begin({platformUrl:'https://wtp.test',sourceId:'s',worldId:'a'});
console.log('ready');await new Promise(resolve=>process.stdin.once('data',resolve));await held.close();
"""

    async def run():
        producer = await asyncio.create_subprocess_exec(
            "node",
            "--import",
            "tsx",
            "--input-type=module",
            "-e",
            script,
            cwd=sdk,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            assert await asyncio.wait_for(producer.stdout.readline(), 20) == b"ready\n"
            inventory = await run_runtime_operation("inventory", None)
            assert len(inventory["cache"]["entries"]) == 4
            assert "files" not in json.dumps(inventory)
            assert str(tmp_path) not in json.dumps(inventory)
            busy = await run_runtime_operation("remove-preview", "s/a")
            assert busy["error"]["code"] == "GAME_CACHE_BUSY"
            pinned = await run_runtime_operation("remove", "s/pinned")
            assert pinned["error"]["code"] == "GAME_CACHE_PINNED"
            corrupt = await run_runtime_operation("inspect", "s/corrupt")
            assert corrupt["cache"]["verification"] == "sha256"
            assert corrupt["cache"]["current"]["status"] == "corrupt"
            removed = await run_runtime_operation("remove", "s/b")
            assert removed["cache"]["action"]["freedBytes"] == 0
            assert (tmp_path / "b").read_text() == "hello"
        finally:
            if producer.returncode is None:
                producer.stdin.write(b"release\n")
                await producer.stdin.drain()
                producer.stdin.close()
                try:
                    await asyncio.wait_for(producer.wait(), 10)
                except asyncio.TimeoutError:
                    producer.kill()
                    await producer.wait()
                    raise
            assert producer.returncode == 0, (await producer.stderr.read()).decode()

    asyncio.run(run())


def test_native_room_admission_blocks_maintenance_before_the_driver_exists(tmp_path):
    from fastapi import HTTPException

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_mock import (
        shared_mock_room,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import (
        RoomRuntime,
        create_app,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        MemoryRoomStore,
        RoomDirectory,
        new_id,
    )

    async def run():
        directory = RoomDirectory(MemoryRoomStore(), {"room": "http://room"})
        game = new_id()
        await directory.resolve(game, "s/w", create=True)
        runtime = RoomRuntime(
            directory=directory,
            owner_id="room",
            driver_factory=shared_mock_room,
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        create_app(runtime, "internal-test")
        entered, proceed = asyncio.Event(), asyncio.Event()
        touch = directory.touch

        async def wait_touch(*args):
            entered.set()
            await proceed.wait()
            return await touch(*args)

        directory.touch = wait_touch
        joining = asyncio.create_task(
            runtime.join(
                dict(
                    game_id=game,
                    world_id="s/w",
                    player_id=new_id(),
                    stream_id=new_id(),
                    session_id=new_id(),
                    user_id="test",
                    trace_id="fixture",
                    spec={"width": 32, "height": 32, "fps": 24},
                )
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert not runtime.rooms
            refused = False
            try:
                async with runtime.maintenance.operation("inspect"):
                    pass
            except HTTPException as error:
                refused = error.status_code == 409
            assert refused, "An admitted join must not race a new heavy cache operation"
        finally:
            proceed.set()
            await joining
            await runtime.close()

    asyncio.run(run())


def test_gateway_refuses_ambiguous_owner_and_returns_unavailable_on_network_failure():
    async def run():
        calls = []

        def transport(request):
            calls.append(request)
            raise httpx.ConnectError("private upstream diagnostic", request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport)
        ) as upstream:
            proxy = SimpleNamespace(owners={}, token="internal-test", client=upstream)
            app = FastAPI()
            install_gateway_management(app, proxy, SECRET)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client:
                for index, owners in enumerate(
                    [{}, {"one": "http://a", "two": "http://b"}, {"one": "http://room"}]
                ):
                    proxy.owners = owners
                    response = await client.post(
                        "/v1/whitebox/runtime",
                        json={"operation": "status"},
                        headers={
                            "Origin": "https://wtp.test",
                            "Authorization": "Bearer "
                            + capability(
                                source_id=None,
                                world_id=None,
                                scope="machine",
                                jti=f"{index + 1:032x}",
                            ),
                        },
                    )
                    assert response.status_code == 503
                    assert "private upstream diagnostic" not in response.text
                assert len(calls) == 1

    asyncio.run(run())


def test_rtc_reservation_drains_maintenance_and_fences_new_work():
    from fastapi import HTTPException

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import reserve_rtc
    from sglang.multimodal_gen.runtime.realtime.whitebox.runtime_management import (
        RuntimeMaintenance,
    )

    async def run():
        entered, drained = asyncio.Event(), asyncio.Event()
        runtime = SimpleNamespace(rooms={}, players={}, rtc_preparation=None)
        runtime.maintenance = RuntimeMaintenance(runtime)
        host = SimpleNamespace(
            config=SimpleNamespace(spec=VideoSpec(832, 480)),
            runtime=runtime,
            lock=runtime.maintenance.lock,
            sessions={},
            peer_sessions={},
            max_sessions=1,
        )

        async def inspect():
            async with runtime.maintenance.operation("inspect"):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    drained.set()

        task = asyncio.create_task(inspect())
        await entered.wait()
        from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
            RenderError,
        )

        try:
            await reserve_rtc(
                host,
                {
                    "user_id": "u",
                    "message": {
                        "type": "init",
                        "control_protocol": "whitebox-render-v2",
                        "source_operation": "invalid",
                    },
                },
            )
            raise AssertionError("Invalid RTC admission must be rejected")
        except RenderError:
            assert not drained.is_set() and not task.done()
        await reserve_rtc(
            host,
            {
                "user_id": "u",
                "message": {
                    "type": "init",
                    "control_protocol": "whitebox-render-v2",
                    "source_operation": "create",
                },
            },
        )
        try:
            assert drained.is_set() and task.done(), (
                "RTC admission must drain discretionary maintenance"
            )
            for operation in ["prepare", "inspect", "remove"]:
                try:
                    async with runtime.maintenance.operation(operation):
                        raise AssertionError(
                            "An RTC reservation must fence maintenance"
                        )
                except HTTPException as error:
                    assert error.status_code == 409
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_repeated_cancellation_waits_for_runtime_child_reaping(tmp_path, monkeypatch):
    import os
    import signal
    import sys

    from sglang.multimodal_gen.runtime.realtime.whitebox.runtime_management import (
        RuntimeMaintenance,
        disconnected,
    )

    async def run():
        entry = tmp_path / "packages/cli/bin/worldkit.mjs"
        entry.parent.mkdir(parents=True)
        entry.write_text("")
        monkeypatch.setenv("WHITEBOX_CLI_ENTRYPOINT", str(entry))
        original = asyncio.create_subprocess_exec
        processes, ready, term, disconnect = (
            [],
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )

        async def spawn(*args, **kwargs):
            process = await original(
                sys.executable,
                "-u",
                "-c",
                'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(60)',
                **{k: v for k, v in kwargs.items() if k != "cwd"},
            )
            processes.append(process)
            assert await process.stdout.readline() == b"ready\n"
            ready.set()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        killpg = os.killpg

        def kill(pid, sig):
            if sig == signal.SIGTERM:
                term.set()
            return killpg(pid, sig)

        monkeypatch.setattr(os, "killpg", kill)

        class Request:
            async def receive(self):
                await disconnect.wait()
                return {"type": "http.disconnect"}

        maintenance = RuntimeMaintenance(SimpleNamespace(rooms={}, players={}))

        async def work():
            async with maintenance.operation("inspect"):
                return await run_runtime_operation("inspect", "s/w")

        request = asyncio.create_task(disconnected(Request(), work()))
        try:
            await ready.wait()
            play = asyncio.create_task(maintenance.cancel_for_play())
            await term.wait()
            disconnect.set()
            await asyncio.wait_for(
                asyncio.gather(play, request, return_exceptions=True), 8
            )
            assert all(p.returncode is not None for p in processes), (
                "Playback continued before child reaping"
            )
            assert maintenance.task is None
        finally:
            disconnect.set()
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
            for process in processes:
                if process.returncode is None:
                    killpg(process.pid, signal.SIGKILL)
                    await process.wait()

    asyncio.run(run())


def test_asgi_prefetch_preemption_and_disconnect_reaps_and_unlocks(
    tmp_path, monkeypatch
):
    import os
    import signal
    import sys

    from sglang.multimodal_gen.runtime.realtime.whitebox.prefetch import (
        install_prefetch,
    )

    async def run():
        entry = tmp_path / "packages/cli/bin/worldkit.mjs"
        entry.parent.mkdir(parents=True)
        entry.write_text("")
        monkeypatch.setenv("WHITEBOX_CLI_ENTRYPOINT", str(entry))
        monkeypatch.setenv("WHITEBOX_PREFETCH_ENABLED", "1")
        monkeypatch.setenv("WHITEBOX_RESOURCE_TRANSPORT", "cdn")
        app = FastAPI()
        runtime = SimpleNamespace(owner_id="room", epoch="epoch", rooms={}, players={})
        prepare = install_prefetch(app, "internal-test")
        install_room_management(app, "internal-test", runtime, prepare)
        original = asyncio.create_subprocess_exec
        processes, progress, term = [], asyncio.Event(), asyncio.Event()
        events = asyncio.Queue()
        await events.put(
            {
                "type": "http.request",
                "body": json.dumps(
                    {"operation": "prepare", "world_id": "s/w"}
                ).encode(),
                "more_body": False,
            }
        )

        async def spawn(*args, **kwargs):
            code = (
                'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print(\'{"type":"world.preparation","state":"preparing"}\',flush=True);time.sleep(60)'
                if not processes
                else 'print(\'{"type":"world.preparation","state":"ready"}\',flush=True)'
            )
            process = await original(
                sys.executable,
                "-u",
                "-c",
                code,
                **{k: v for k, v in kwargs.items() if k != "cwd"},
            )
            processes.append(process)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        killpg = os.killpg

        def kill(pid, sig):
            if sig == signal.SIGTERM:
                term.set()
            return killpg(pid, sig)

        monkeypatch.setattr(os, "killpg", kill)

        async def send(message):
            if message["type"] == "http.response.body" and b"preparing" in message.get(
                "body", b""
            ):
                progress.set()

        scope = dict(
            type="http",
            asgi={"version": "3.0", "spec_version": "2.3"},
            http_version="1.1",
            method="POST",
            scheme="http",
            path="/internal/worlds/runtime",
            raw_path=b"/internal/worlds/runtime",
            query_string=b"",
            headers=[
                (b"authorization", b"Bearer internal-test"),
                (b"content-type", b"application/json"),
            ],
            server=("room", 80),
            client=("operator", 1),
        )
        request = asyncio.create_task(app(scope, events.get, send))
        try:
            await asyncio.wait_for(progress.wait(), 5)
            play = asyncio.create_task(runtime.maintenance.cancel_for_play())
            await asyncio.wait_for(term.wait(), 5)
            await events.put({"type": "http.disconnect"})
            await asyncio.wait_for(
                asyncio.gather(play, request, return_exceptions=True), 8
            )
            assert processes[0].returncode is not None, (
                "Play must wait for terminated prewarm child"
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://room"
            ) as client:
                next_response = await client.post(
                    "/internal/worlds/runtime",
                    json={"operation": "prepare", "world_id": "s/w"},
                    headers={"Authorization": "Bearer internal-test"},
                )
                assert next_response.status_code == 200
                assert "PREPARATION_BUSY" not in next_response.text
                assert "ready" in next_response.text
                assert len(processes) == 2
        finally:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
            for process in processes:
                if process.returncode is None:
                    killpg(process.pid, signal.SIGKILL)
                    await process.wait()

    asyncio.run(run())


def test_unknown_cli_workers_are_not_reported_as_zero(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    import pytest

    sdk = os.environ.get("WHITEBOX_TEST_CLI_SDK")
    if not sdk:
        pytest.skip("Native maintenance contract requires WHITEBOX_TEST_CLI_SDK")
    monkeypatch.setenv(
        "WHITEBOX_CLI_ENTRYPOINT", str(Path(sdk) / "packages/cli/bin/worldkit.mjs")
    )
    monkeypatch.setenv("WHITEBOX_CLI_GAME_CACHE_DIRECTORY", str(tmp_path / "games"))
    monkeypatch.setenv("WHITEBOX_CLI_SERVICE_NAME", "invalid/name")
    result = asyncio.run(run_runtime_operation("status", "s/w"))
    assert result["service"]["status"] == "unknown"
    assert result["service"]["workers"] is None
