# SPDX-License-Identifier: Apache-2.0
"""Real sealed CLI -> SG -> PublicExecutionPlayer/Web Audio; opt-in, no model peer."""

import asyncio
import base64
import json
import os
import ssl
import subprocess
from pathlib import Path

import httpx
import msgspec
import pytest
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from test_whitebox_cli_integration import server
from websockets.asyncio.client import connect

from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
    AsyncArchiveManager,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_room
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import RenderHost
from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
    new_id,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig


@pytest.mark.skipif(
    not os.environ.get("WHITEBOX_TEST_CLI_SDK"),
    reason="Requires candidate SDK checkout and Chromium",
)
@pytest.mark.asyncio
async def test_real_browser_player_project_audio(tmp_path, monkeypatch):
    start_tick = 1
    assert not os.environ.get("WHITEBOX_TEST_GPU_COORDINATOR_URL"), (
        "CPU audio lane refuses GPU coordinator"
    )
    monkeypatch.setenv("WHITEBOX_TEST_PROJECT_AUDIO", "1")
    monkeypatch.setenv("WHITEBOX_TEST_AUDIO_STOP_TICK", "900")
    monkeypatch.setenv("WHITEBOX_TEST_AUDIO_START_TICK", str(start_tick))
    monkeypatch.delenv("WHITEBOX_BOUND_COMPUTE", raising=False)
    sdk = Path(os.environ["WHITEBOX_TEST_CLI_SDK"]).resolve()
    cert, key = tmp_path / "ca.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for name, value in {
        "WHITEBOX_TEST_TLS_CERT": str(cert),
        "WHITEBOX_TEST_TLS_KEY": str(key),
        "NODE_EXTRA_CA_CERTS": str(cert),
        "WHITEBOX_CLI_GAME_CACHE_DIRECTORY": str(tmp_path / "games"),
    }.items():
        monkeypatch.setenv(name, value)
    log = (tmp_path / "fixture.log").open("wb")
    fixture = await asyncio.create_subprocess_exec(
        "node",
        "--import",
        "tsx",
        str(Path(__file__).with_name("whitebox_wtp_fixture.mjs")),
        cwd=sdk,
        stdout=asyncio.subprocess.PIPE,
        stderr=log,
    )
    host = runtime = None
    try:
        line = await asyncio.wait_for(fixture.stdout.readline(), 120)
        assert line, (tmp_path / "fixture.log").read_text()[-5000:]
        startup = json.loads(line)
        url = startup["url"]
        assert startup["candidate"]["runtimeCacheHit"] is False
        (tmp_path / "audio-runtime-identity.json").write_text(
            json.dumps(startup["candidate"], indent=2)
        )
        for key, value in dict(
            WHITEBOX_CLI_ENTRYPOINT=str(sdk / "packages/cli/bin/worldkit.mjs"),
            WHITEBOX_CLI_PLATFORM_URL=url,
            WORLD_TEST_API_TOKEN="local-fixture-token",
            WHITEBOX_CLI_RENDERING="software",
            WHITEBOX_CLI_SERVICE="off",
            WHITEBOX_CLI_RESOURCE_TRANSPORT="cdn",
            WHITEBOX_CLI_STARTUP_SECONDS="90",
        ).items():
            monkeypatch.setenv(key, value)
        spec = VideoSpec(832, 480)
        archives = AsyncArchiveManager(queue_depth=256)
        directory = RoomDirectory(MemoryRoomStore(), {"audio-test": "http://loopback"})
        game = new_id()
        await directory.resolve(game, "s/w", create=True)
        runtime = RoomRuntime(
            directory=directory,
            owner_id="audio-test",
            driver_factory=cli_room,
            spool=tmp_path / "room",
            spec=spec,
            archives=archives,
        )
        reports = Reports(runtime.spool, archives)
        host = RenderHost(
            runtime,
            WhiteboxConfig(
                engine_factory=None,
                ui_factory=None,
                spool=runtime.spool,
                spec=spec,
                room_owned=True,
                startup_seconds=90,
                drain_seconds=3,
            ),
            archives,
            reports,
            None,
            None,
        )

        def forbidden(*_):
            pytest.fail("No-inference audio test attempted to create a model lease")

        host.make_segment_leases = forbidden
        app = FastAPI()
        host.install(app, "local-audio-token")
        reports.install(app)
        from sglang.multimodal_gen.runtime.realtime.whitebox.room_proxy import RoomProxy

        helper = sdk / "apps/game-player/tests/integration/project-audio-browser.mjs"
        static = tmp_path / "browser"
        build = await asyncio.create_subprocess_exec(
            "node", "--import", "tsx", str(helper), "build", str(static), cwd=sdk
        )
        assert await build.wait() == 0
        async with httpx.AsyncClient(
            trust_env=False, verify=ssl.create_default_context(cafile=str(cert))
        ) as http:
            location = (
                await http.get(
                    url + "/api/worlds/s/w/playable-location?transport=archive",
                    headers={"Authorization": "Bearer local-fixture-token"},
                )
            ).json()["location"]
            publication = {
                key: location[key]
                for key in (
                    "sourceId",
                    "worldId",
                    "publicationId",
                    "sourceManifestSha256",
                    "renditionInputSha256",
                    "manifestSha256",
                )
            }
            publication["attempt"] = 1
            init = dict(
                type="init",
                control_protocol="whitebox-render-v2",
                source_operation="create.manual",
                inference_control="manual",
                prompt="local audio browser",
                whitebox=dict(source_id="s", world_id="w"),
            )
            operation = dict(
                operationId="browser-audio",
                parametersSha256="a" * 64,
                caller=dict(serviceId="wtp", subject="local-audio"),
                frozenPublication=dict(
                    status="verified",
                    scope="world-frozen-publication",
                    world=dict(
                        sourceId="s", worldId="w", publicationId="local", attempt=1
                    ),
                ),
                frozenLaunchParameters=dict(
                    transport="websocket",
                    publicationIdentity=publication,
                    initMsgpackBase64=base64.b64encode(
                        msgspec.msgpack.encode(init)
                    ).decode(),
                ),
                _route=dict(
                    game_id=game,
                    owner_id=runtime.owner_id,
                    owner_url="http://loopback",
                    owner_epoch=runtime.epoch,
                ),
            )
            await host.single_player_control.start(operation)
            async with asyncio.timeout(100):
                while True:
                    receipt = await host.single_player_control.get_attachment(operation)
                    if receipt.get("state") == "failed":
                        pytest.fail(str(receipt))
                    if receipt.get("state") == "ready" and receipt.get("attachment"):
                        break
                    await asyncio.sleep(0.05)
            async with server(app) as address:
                gateway = FastAPI()
                sockets = set()

                @gateway.post("/drop")
                async def drop():
                    for socket in list(sockets):
                        await socket.close(code=1012)
                    return {"dropped": True}

                RoomProxy(
                    None, {"audio-test": address}, "local-audio-token", http, None
                ).install(gateway)

                @gateway.get("/")
                async def index():
                    return HTMLResponse(
                        '<!doctype html><style>html,body,#root{height:100%;margin:0}.player-dual-surface,.player-primary-surface{height:100%}</style><div id="root"></div><script src="/player.js"></script>'
                    )

                @gateway.get("/player.js")
                async def player_script():
                    return FileResponse(static / "player.js")

                @gateway.get("/player-ui-frame.js")
                async def ui_script():
                    return FileResponse(static / "player-ui-frame.js")

                @gateway.get("/descriptor")
                async def descriptor(sequence: int = 0):
                    current = await host.single_player_control.get_attachment(
                        dict(operation, controlSequence=sequence)
                    )
                    return dict(
                        kind="player",
                        handle="audio-browser",
                        mode="single",
                        status="ready",
                        identity=publication,
                        session=dict(
                            kind="whitebox",
                            publication=publication,
                            execution=current["execution"],
                            attachment=current["attachment"],
                            signallingUrl=gateway_address.replace("http", "ws")
                            + "/v1/realtime_video/generate",
                        ),
                    )

                @gateway.websocket("/v1/realtime_video/generate")
                async def relay(socket: WebSocket):
                    await socket.accept()
                    sockets.add(socket)
                    async with connect(
                        address.replace("http", "ws") + "/internal/render-v2",
                        additional_headers={
                            "Authorization": "Bearer local-audio-token"
                        },
                        max_size=16 << 20,
                    ) as upstream:
                        first = msgspec.msgpack.decode(await socket.receive_bytes())
                        await upstream.send(
                            msgspec.msgpack.encode(
                                dict(
                                    user_id="local-audio",
                                    trace_id="browser-audio",
                                    message=first,
                                )
                            )
                        )

                        async def receive():
                            while True:
                                await upstream.send(await socket.receive_bytes())

                        async def send():
                            while True:
                                await socket.send_bytes(await upstream.recv())

                        tasks = [
                            asyncio.create_task(receive()),
                            asyncio.create_task(send()),
                        ]
                        try:
                            await asyncio.wait(
                                tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                        finally:
                            sockets.discard(socket)
                            for task in tasks:
                                task.cancel()
                            await asyncio.gather(*tasks, return_exceptions=True)

                async with server(gateway) as gateway_address:
                    browser_log = (tmp_path / "browser.log").open("wb")
                    browser = await asyncio.create_subprocess_exec(
                        "node",
                        "--import",
                        "tsx",
                        str(helper),
                        "run",
                        gateway_address,
                        str(tmp_path / "browser-evidence.json"),
                        cwd=sdk,
                        stdout=browser_log,
                        stderr=browser_log,
                    )
                    try:
                        assert await asyncio.wait_for(browser.wait(), 100) == 0, (
                            tmp_path / "browser.log"
                        ).read_text()[-5000:]
                    finally:
                        if browser.returncode is None:
                            browser.terminate()
                            await browser.wait()
                        browser_log.close()
                assert len(runtime.rooms) == len(runtime.players) == 1
                assert all(
                    p.session.current_segment is None for p in host.sessions.values()
                )
                await host.single_player_control.close(operation)
    finally:
        if host:
            await host.close()
        if runtime:
            await runtime.close()
        if fixture.returncode is None:
            fixture.terminate()
            try:
                await asyncio.wait_for(fixture.wait(), 10)
            except TimeoutError:
                fixture.kill()
                await fixture.wait()
        log.close()
