# SPDX-License-Identifier: Apache-2.0
"""Original SDK wire relay: fixed routes, per-member capability, no re-encoding."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from sglang.multimodal_gen.runtime.realtime.whitebox.player_transport import (
    MAX_WIRE_BYTES,
    PlayerTransport,
    PlayerTransportConfig,
    install_player_gateway,
    install_player_runtime,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
    new_id,
)


def test_ordinary_cli_input_publishes_a_scoped_session_without_room_admission(
    monkeypatch,
):
    from sglang.multimodal_gen.runtime.realtime.whitebox import cli_room as cli_api
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import (
        CLIRoom,
        NativeCLIView,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec

    monkeypatch.setattr(cli_api, "INPUT_SESSION_RENEW_SECONDS", 0.001)

    async def run():
        private = "http://127.0.0.1:65432"
        manifest = {
            "schemaVersion": 1,
            "abiVersion": "player-input/1",
            "manifest": {"path": "player-input/manifest.json", "sha256": "a" * 64},
            "adapters": [],
        }
        grant = dict(
            sessionId="sdk-input",
            epoch=0,
            baseUrl=private,
            accessToken="b" * 48,
            accessMode="capability",
            worldBuildHash="c" * 64,
            uiBundleHash="d" * 64,
            width=32,
            height=32,
            fps=24,
            media={"kind": "pixels", "pixelFormat": "rgb24", "colorSpace": "srgb"},
            input={**manifest, "baseUrl": private + "/v1/sessions/sdk-input/"},
            audio={"manifestUrl": private + "/v1/sessions/sdk-input/audio/manifest"},
            ui={"moduleUrl": private + "/ui/" + "d" * 64 + "/module.js", "styles": []},
        )
        admin = {**grant, "accessToken": "e" * 48, "inputSession": grant}
        admin.pop("accessMode")
        room = CLIRoom(None, admin, VideoSpec(32, 32), None, "trace")
        view = NativeCLIView(room)
        room.view, room.player_id = view, "original"
        assert view.external_input and view.metadata["project_input"] is True
        assert view.player_descriptor is grant
        with pytest.raises(ValueError, match="browser"):
            await view.send_input(0, {"held_keys": ["KeyW"]})

        renewals = 0
        renewed = asyncio.Event()

        def sdk(request):
            nonlocal renewals
            renewals += 1
            if renewals >= 3:
                renewed.set()
            assert (
                request.method == "POST"
                and request.url.path == "/v1/sessions/sdk-input/input-session"
            )
            assert request.headers["authorization"] == "Bearer " + admin["accessToken"]
            return httpx.Response(200, json=grant)

        client = httpx.AsyncClient
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kwargs: client(transport=httpx.MockTransport(sdk), **kwargs),
        )
        directory = RoomDirectory(MemoryRoomStore(), {"a": "http://room"})
        game, stream, epoch = new_id(), new_id(), new_id()
        await directory.resolve(game, "local-fixture/input", create=True)
        await directory.touch(game, "a", epoch)
        entry = dict(
            game_id=game,
            engine=view,
            source_input=True,
            identity={"player_id": "original"},
        )
        runtime = SimpleNamespace(
            directory=directory,
            owner_id="a",
            epoch=epoch,
            rooms={game: {"driver": room}},
            player=lambda value, **_: (
                entry if value == stream else (_ for _ in ()).throw(ValueError())
            ),
        )
        transport = PlayerTransport(
            runtime,
            PlayerTransportConfig(
                "https://gateway.test", "https://player.test", "https://wtp.test"
            ),
        )
        session = await transport.public_session({"stream_id": stream})
        await asyncio.wait_for(renewed.wait(), 1)
        task = room.input_renew_task
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        before = renewals
        await asyncio.sleep(0.01)
        assert renewals == before  # Retirement cannot keep renewing a stale grant.
        assert session["baseUrl"] == "https://gateway.test"
        assert (
            session["input"]["baseUrl"] == "https://gateway.test/v1/sessions/sdk-input/"
        )
        assert session["accessToken"] != admin["accessToken"]
        assert "audio" not in session  # The original Whitebox owner serves audio.
        assert "playerBinding" not in session and "control_admission" not in entry
        assert (await directory.session_route("sdk-input"))["stream_id"] == stream
        with pytest.raises(ValueError):
            transport.viewer(stream, "sdk-input", admin["accessToken"])
        with pytest.raises(ValueError):
            await room.public_session("other-player", "https://gateway.test")
        entry["source_input"] = False
        with pytest.raises(ValueError):
            transport.viewer(stream, "sdk-input", grant["accessToken"])

    asyncio.run(run())


@pytest.mark.parametrize(
    "kind",
    ["input.state", "input.action", "input.semantic.batch", "input.semantic.open"],
)
def test_public_control_relay_expires_blocked_gameplay_but_not_handshake(kind):
    import json

    from sglang.multimodal_gen.runtime.realtime.whitebox.player_transport import relay

    async def run():
        sent = []
        waiting = asyncio.Event()
        sent_complete = asyncio.Event()

        class Browser:
            received = False

            async def receive(self):
                if self.received:
                    await sent_complete.wait()
                    return {"type": "websocket.disconnect"}
                self.received = True
                return {"type": "websocket.receive", "text": json.dumps({"type": kind})}

        class SDK:
            async def send(self, value):
                await asyncio.sleep(0.25)
                sent.append(value)
                sent_complete.set()

            async def __aiter__(self):
                await waiting.wait()
                yield "unused"

        if kind == "input.semantic.open":
            await relay(Browser(), SDK())
            assert len(sent) == 1
        else:
            with pytest.raises(TimeoutError):
                await relay(Browser(), SDK())
            assert sent == []

    asyncio.run(run())


def test_session_route_uses_existing_store_and_owner_epoch_fence():
    async def run():
        directory = RoomDirectory(MemoryRoomStore(), {"a": "http://room"})
        game, epoch, stream = new_id(), new_id(), new_id()
        await directory.resolve(game, "s/w", create=True)
        await directory.touch(game, "a", epoch)
        binding = dict(
            session_id="sdk-session-a",
            game_id=game,
            owner_id="a",
            owner_epoch=epoch,
            stream_id=stream,
            epoch=0,
        )
        assert await directory.session_route("sdk-session-a") is None
        assert await directory.publish_session(binding) == binding
        assert await directory.publish_session(binding) == binding
        with pytest.raises(ValueError):
            await directory.publish_session({**binding, "stream_id": new_id()})
        assert await directory.session_route("sdk-session-a") == binding
        await directory.touch(game, "a", epoch, end=True)
        with pytest.raises(ValueError):
            await directory.session_route("sdk-session-a")

    asyncio.run(run())


def test_private_http_transport_is_exact_member_and_declared_ui_only():
    async def run():
        token = "a" * 48
        entry = dict(
            game_id=new_id(),
            identity={"player_id": new_id()},
            control_admission={"status": "claimed"},
        )
        descriptor = dict(
            sessionId="sdk-a",
            epoch=0,
            accessToken=token,
            accessMode="capability",
            baseUrl="http://127.0.0.1:65432",
            uiBundleHash="b" * 64,
        )
        entry["engine"] = SimpleNamespace(descriptor=descriptor)
        calls = []

        def sdk(request):
            calls.append(request)
            assert request.url.host == "127.0.0.1" and request.url.port == 65432
            assert request.url.path == "/ui/" + "b" * 64 + "/module.js"
            assert "cookie" not in request.headers
            return httpx.Response(
                200,
                content=b"export const module = 1",
                headers={"content-type": "text/javascript"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(sdk)) as upstream:
            runtime = SimpleNamespace(
                players={"1" * 32: entry},
                player=lambda stream, **_: (
                    entry if stream == "1" * 32 else (_ for _ in ()).throw(ValueError())
                ),
                rooms={},
            )
            app = FastAPI()
            config = PlayerTransportConfig(
                "https://gateway.test", "https://player.test", "https://wtp.test"
            )
            install_player_runtime(
                app, "internal", runtime, config=config, client=upstream
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://room"
            ) as client:
                path = (
                    "/internal/player-streams/"
                    + "1" * 32
                    + "/sdk-a/ui/"
                    + "b" * 64
                    + "/module.js"
                )
                assert (await client.get(path)).status_code == 403
                headers = {
                    "authorization": "Bearer internal",
                    "origin": "https://player.test",
                }
                assert (await client.get(path, headers=headers)).status_code == 200
                assert len(calls) == 1
                bad = path.replace("sdk-a", "sdk-b")
                assert (await client.get(bad, headers=headers)).status_code == 404
                assert (
                    await client.get(path.replace("b" * 64, "c" * 64), headers=headers)
                ).status_code == 404
                assert (
                    await client.get(
                        path, headers={**headers, "origin": "https://attacker.test"}
                    )
                ).status_code == 403
                entry["control_admission"]["leaving"] = True
                assert (await client.get(path, headers=headers)).status_code == 404
                assert len(calls) == 1

    asyncio.run(run())


@pytest.mark.parametrize("model_source", [False, True])
@pytest.mark.parametrize("prefill_origin", [False, True])
def test_real_gateway_room_sdk_sockets_preserve_packets_and_member_isolation(
    model_source,
    prefill_origin,
):
    from fastapi import WebSocket
    from test_whitebox_rooms import server
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidStatus

    async def run():
        alternate = "http://127.0.0.1:5307" if prefill_origin else None
        browser_origin = alternate or "https://player.test"
        sdk = FastAPI()
        seen = []

        @sdk.get("/ui/{ui_hash}/module.js")
        async def original_ui(ui_hash: str):
            from fastapi.responses import Response

            return Response(
                "export const original = true", media_type="text/javascript"
            )

        @sdk.websocket("/v1/sessions/{session_id}/{channel}")
        async def original(socket: WebSocket, session_id: str, channel: str):
            seen.append(
                (
                    session_id,
                    channel,
                    socket.headers.get("origin"),
                    socket.headers.get("cookie"),
                )
            )
            await socket.accept(subprotocol="worldkit.stream.v1")
            try:
                if channel == "control":
                    message = await socket.receive_text()
                    await socket.send_text(message)
                else:
                    await socket.send_bytes(b"original-native-packet-\x00\xff")
                    assert await socket.receive_bytes() == b"original-credit-ack"
            finally:
                await socket.close()

        async with server(sdk) as sdk_url:
            directory = RoomDirectory(MemoryRoomStore(), {"a": "http://unused"})
            game, owner_epoch = new_id(), new_id()
            await directory.resolve(game, "s/w", create=True)
            await directory.touch(game, "a", owner_epoch)
            entries = {}
            for name, token in [("sdk-a", "a" * 48), ("sdk-b", "b" * 48)]:
                stream = new_id()
                descriptor = dict(
                    sessionId=name,
                    epoch=0,
                    accessToken=token,
                    accessMode="capability",
                    baseUrl=sdk_url,
                    uiBundleHash="b" * 64,
                )
                entries[stream] = dict(
                    engine=SimpleNamespace(
                        descriptor=(
                            dict(descriptor, accessToken="f" * 48)
                            if model_source
                            else descriptor
                        ),
                        **({"player_descriptor": descriptor} if model_source else {}),
                    ),
                    control_admission={"status": "claimed"},
                )
                if prefill_origin and name == "sdk-a":
                    entries[stream]["control_admission"]["mediaLaunch"] = {
                        "memberPrefill": {"protocol": "worldkit-member-prefill-v1"}
                    }
                await directory.publish_session(
                    dict(
                        session_id=name,
                        game_id=game,
                        owner_id="a",
                        owner_epoch=owner_epoch,
                        stream_id=stream,
                        epoch=0,
                    )
                )
            runtime = SimpleNamespace(
                players=entries, player=lambda stream, **_: entries[stream]
            )
            room = FastAPI()
            config = PlayerTransportConfig(
                "https://gateway.test",
                "https://player.test",
                "https://wtp.test",
                alternate,
            )
            install_player_runtime(room, "internal", runtime, config=config)
            async with server(room) as room_url:
                async with httpx.AsyncClient() as upstream:
                    proxy = SimpleNamespace(
                        directory=directory,
                        owners={"a": room_url},
                        token="internal",
                        client=upstream,
                    )
                    gateway = FastAPI()
                    # Bind the actual public authority once its local socket is known.
                    async with server(gateway) as gateway_url:
                        config = PlayerTransportConfig(
                            gateway_url,
                            "https://player.test",
                            "https://wtp.test",
                            alternate,
                        )
                        install_player_gateway(gateway, proxy, config=config)
                        wsbase = gateway_url.replace("http:", "ws:")
                        protocols = ["worldkit.stream.v1", "worldkit.cap." + "a" * 48]
                        async with connect(
                            wsbase + "/v1/sessions/sdk-a/control?clientId=browser-a",
                            origin=browser_origin,
                            subprotocols=protocols,
                        ) as socket:
                            assert socket.subprotocol == "worldkit.stream.v1"
                            await socket.send('{"type":"input.release","epoch":0}')
                            assert (
                                await socket.recv()
                                == '{"type":"input.release","epoch":0}'
                            )
                        async with connect(
                            wsbase + "/v1/sessions/sdk-a/pixels?clientId=browser-a",
                            origin=browser_origin,
                            subprotocols=protocols,
                        ) as socket:
                            assert (
                                await socket.recv()
                                == b"original-native-packet-\x00\xff"
                            )
                            await socket.send(b"original-credit-ack")
                        for headers in [
                            {"Origin": browser_origin},
                            {"Origin": config.player_origin},
                            {},
                        ]:
                            ui = await upstream.get(
                                gateway_url
                                + "/v1/sessions/sdk-a/ui/"
                                + "b" * 64
                                + "/module.js",
                                headers=headers,
                            )
                            assert ui.status_code == 200
                            assert ui.text == "export const original = true"
                            assert ui.headers["access-control-allow-origin"] == (
                                headers.get("Origin") or config.player_origin
                            )
                            assert ui.headers["content-type"].startswith(
                                "text/javascript"
                            )
                        for path, origin in [
                            ("sdk-b/control", "https://player.test"),
                            ("sdk-a/control", "https://evil.test"),
                            ("sdk-a/producer-control", "https://player.test"),
                        ]:
                            with pytest.raises(InvalidStatus):
                                async with connect(
                                    wsbase + "/v1/sessions/" + path,
                                    origin=origin,
                                    subprotocols=protocols,
                                ):
                                    pass
                        if prefill_origin:
                            # Even the other member's valid viewer capability is
                            # insufficient to opt an ordinary admission into this origin.
                            with pytest.raises(InvalidStatus):
                                async with connect(
                                    wsbase + "/v1/sessions/sdk-b/control",
                                    origin=browser_origin,
                                    subprotocols=[
                                        "worldkit.stream.v1",
                                        "worldkit.cap." + "b" * 48,
                                    ],
                                ):
                                    pass
                            denied = await upstream.get(
                                gateway_url
                                + "/v1/sessions/sdk-b/ui/"
                                + "b" * 64
                                + "/module.js",
                                headers={"Origin": browser_origin},
                            )
                            assert denied.status_code == 404
                        if model_source:
                            with pytest.raises(InvalidStatus):
                                async with connect(
                                    wsbase + "/v1/sessions/sdk-a/control",
                                    origin="https://player.test",
                                    subprotocols=[
                                        "worldkit.stream.v1",
                                        "worldkit.cap." + "f" * 48,
                                    ],
                                ):
                                    pass
                        assert len(seen) == 2 and all(
                            row[0] == "sdk-a" and row[2] == sdk_url and row[3] is None
                            for row in seen
                        )
                        entries[next(iter(entries))]["control_admission"]["leaving"] = (
                            True
                        )
                        with pytest.raises(InvalidStatus):
                            async with connect(
                                wsbase + "/v1/sessions/sdk-a/control",
                                origin="https://player.test",
                                subprotocols=protocols,
                            ):
                                pass

    asyncio.run(run())


def test_controlled_capture_observes_without_competing_for_browser_input():
    import json

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import CLIView
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec

    async def run():
        room = SimpleNamespace(
            external_input=True,
            prepared_delivery=None,
            descriptor={"sessionId": "s", "epoch": 0, "accessMode": "capability"},
            spec=VideoSpec(32, 32),
        )
        view = CLIView(room)
        stop = asyncio.Event()

        async def messages():
            yield json.dumps({"type": "session.ready", "epoch": 0, "role": "spectator"})
            await stop.wait()

        view.control = messages()
        task = asyncio.create_task(view._control())
        try:
            await asyncio.wait_for(view.ready.wait(), 0.5)
            assert not view.controller_ready and view.binding_sequence is None
            with pytest.raises(ValueError, match="browser"):
                await view.send_input(0, {"held_keys": ["KeyW"]})
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


async def verify_real_ordinary_input_cli(descriptor):
    """Opt-in SDK CLI contract test: actual World/CLI sockets plus Gateway/Room
    relay. The directory is local; no model, production session or GPU is used."""
    import json
    from contextlib import AsyncExitStack

    from test_whitebox_rooms import server
    from websockets.asyncio.client import connect

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import (
        CLIRoom,
        NativeCLIView,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec

    room = CLIRoom(
        None,
        descriptor,
        VideoSpec(descriptor["width"], descriptor["height"]),
        None,
        "input-contract",
    )
    view = NativeCLIView(room)
    room.view, room.player_id = view, "original"
    assert view.external_input and view.metadata["project_input"]
    directory = RoomDirectory(MemoryRoomStore(), {"a": "http://unused"})
    game, stream, epoch = new_id(), new_id(), new_id()
    await directory.resolve(game, "local-fixture/input", create=True)
    await directory.touch(game, "a", epoch)
    entry = dict(
        game_id=game, engine=view, source_input=True, identity={"player_id": "original"}
    )
    runtime = SimpleNamespace(
        directory=directory,
        owner_id="a",
        epoch=epoch,
        rooms={game: {"driver": room}},
        player=lambda value, **_: (
            entry if value == stream else (_ for _ in ()).throw(ValueError())
        ),
    )
    async with AsyncExitStack() as stack:
        room_app, gateway = FastAPI(), FastAPI()
        room_url = await stack.enter_async_context(server(room_app))
        gateway_url = await stack.enter_async_context(server(gateway))
        client = await stack.enter_async_context(httpx.AsyncClient(trust_env=False))
        config = PlayerTransportConfig(
            gateway_url, "https://player.test", "https://wtp.test"
        )
        install_player_runtime(room_app, "internal", runtime, config=config)
        install_player_gateway(
            gateway,
            SimpleNamespace(
                directory=directory,
                owners={"a": room_url},
                token="internal",
                client=client,
            ),
            config=config,
        )
        session = await runtime.player_transport.public_session({"stream_id": stream})
        assert session["accessToken"] != descriptor["accessToken"]
        assert session["input"]["manifest"] == descriptor["input"]["manifest"]
        resource = session["input"]["baseUrl"] + session["input"]["manifest"]["path"]
        response = await client.get(
            resource,
            headers={
                "Origin": config.player_origin,
                "Authorization": "Bearer " + session["accessToken"],
            },
        )
        assert (
            response.status_code == 200
            and response.json()["abiVersion"] == "player-input/1"
        )
        wrong = await client.get(
            resource,
            headers={
                "Origin": config.player_origin,
                "Authorization": "Bearer " + descriptor["accessToken"],
            },
        )
        assert wrong.status_code == 404
        url = (
            gateway_url.replace("http://", "ws://")
            + f"/v1/sessions/{session['sessionId']}/control?clientId=cli-sg-contract&ui=0&clock=0"
        )
        async with connect(
            url,
            origin=config.player_origin,
            subprotocols=[
                "worldkit.stream.v1",
                "worldkit.cap." + session["accessToken"],
            ],
        ) as control:

            async def receive(kind):
                async with asyncio.timeout(10):
                    while True:
                        value = json.loads(await control.recv())
                        assert descriptor["accessToken"] not in json.dumps(value)
                        assert session["accessToken"] not in json.dumps(value)
                        if value.get("type") == kind:
                            return value
                        if value.get("type") in {"error", "producer.error"}:
                            raise ValueError(
                                "The CLI relay failed: " + str(value.get("code"))
                            )

            assert (await receive("session.ready"))["role"] == "controller"
            await control.send(
                json.dumps(
                    dict(
                        type="input.semantic.open",
                        epoch=session["epoch"],
                        abiVersion="player-input/1",
                    )
                )
            )
            binding = await receive("input.semantic.binding")
            await control.send(
                json.dumps(
                    dict(
                        type="input.semantic.applied",
                        epoch=session["epoch"],
                        generation=binding["generation"],
                    )
                )
            )
            await control.send(
                json.dumps(
                    dict(
                        type="input.semantic.batch",
                        epoch=session["epoch"],
                        generation=binding["generation"],
                        batch=dict(
                            schemaVersion=1,
                            sequence=1,
                            stateRevision=1,
                            state={"power": 0.5},
                            events=[],
                        ),
                    )
                )
            )
            receipt = await receive("input.semantic.receipt")
            assert receipt["sequence"] == 1 and receipt["meaningful"] is True
            assert (
                receipt["clientId"] == "cli-sg-contract"
                and receipt["simulationTick"] > 0
            )


async def verify_real_public_players(room, alice, bob):
    """Called by the opt-in real sealed CLI test; substitutes no SDK transport."""
    import json
    from contextlib import AsyncExitStack

    from fastapi import FastAPI
    from test_whitebox_rooms import server
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidStatus

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_input
    from sglang.multimodal_gen.runtime.realtime.whitebox.native_pixels import (
        decode_native_packet,
    )

    directory = RoomDirectory(MemoryRoomStore(), {"a": "http://unused"})
    game, owner_epoch = new_id(), new_id()
    await directory.resolve(game, "local-fixture/room-fixture", create=True)
    await directory.touch(game, "a", owner_epoch)
    streams, entries, records = {}, {}, {}
    for name, player, view in [("a", "a" * 32, alice), ("b", "b" * 32, bob)]:
        stream = new_id()
        streams[name] = stream
        record = dict(stream_id=stream, status="claimed")
        records[name] = record
        entries[stream] = dict(
            engine=view,
            identity={"player_id": player},
            game_id=game,
            control_admission=record,
        )

    def player(stream_id, **_):
        if stream_id not in entries:
            raise ValueError("The player has left")
        return entries[stream_id]

    runtime = SimpleNamespace(
        directory=directory,
        owner_id="a",
        epoch=owner_epoch,
        player=player,
        players=entries,
        rooms={game: {"driver": room}},
    )
    async with AsyncExitStack() as stack:
        room_app, gateway = FastAPI(), FastAPI()
        room_url = await stack.enter_async_context(server(room_app))
        gateway_url = await stack.enter_async_context(server(gateway))
        client = await stack.enter_async_context(httpx.AsyncClient())
        config = PlayerTransportConfig(
            gateway_url, "https://player.test", "https://wtp.test"
        )
        install_player_runtime(room_app, "internal", runtime, config=config)
        install_player_gateway(
            gateway,
            SimpleNamespace(
                directory=directory,
                owners={"a": room_url},
                token="internal",
                client=client,
            ),
            config=config,
        )
        sessions = {
            name: await runtime.player_transport.public_session(records[name])
            for name in ("a", "b")
        }
        controls, messages, bindings = {}, {"a": [], "b": []}, {}

        async def message(name, kind=None, sequence=None):
            def matches(m):
                return (
                    kind is not None
                    and m.get("type") == kind
                    and (sequence is None or m.get("clientSequence") == sequence)
                )

            for m in messages[name]:
                if matches(m):
                    return m
            async with asyncio.timeout(20):
                while True:
                    m = json.loads(await controls[name].recv())
                    messages[name].append(m)
                    if m.get("type") in {"error", "producer.error", "session.ended"}:
                        raise ValueError(
                            "The real public player failed: "
                            + str(m.get("code", m["type"]))
                        )
                    if matches(m):
                        return m

        for name in ("a", "b"):
            session = sessions[name]
            assert session["baseUrl"] == gateway_url
            assert not set(session) & {
                "streams",
                "localResources",
                "artifact",
                "status",
            }
            assert session["ui"]["moduleUrl"].startswith(
                gateway_url + "/v1/sessions/" + session["sessionId"] + "/ui/"
            )
            protocols = ["worldkit.stream.v1", "worldkit.cap." + session["accessToken"]]
            control = await stack.enter_async_context(
                connect(
                    gateway_url.replace("http:", "ws:")
                    + "/v1/sessions/"
                    + session["sessionId"]
                    + "/control?clientId=public-"
                    + name,
                    origin=config.player_origin,
                    subprotocols=protocols,
                )
            )
            controls[name] = control
            assert (await message(name, "session.ready"))["role"] == "controller"
            if session.get("modelContext", {}).get("required"):
                await control.send(
                    json.dumps(
                        {"type": "model.context.preview", "epoch": session["epoch"]}
                    )
                )
            bindings[name] = (await message(name, "input.binding"))["bindingSequence"]
            await control.send(
                json.dumps(
                    {
                        "type": "input.binding.applied",
                        "epoch": session["epoch"],
                        "bindingSequence": bindings[name],
                    }
                )
            )
            asset = await client.get(
                session["ui"]["moduleUrl"], headers={"origin": config.player_origin}
            )
            assert (
                asset.status_code == 200
                and asset.headers["access-control-allow-origin"] == config.player_origin
            )
            live = await client.get(
                gateway_url + "/v1/sessions/" + session["sessionId"],
                headers={"Authorization": "Bearer " + session["accessToken"]},
            )
            assert live.status_code == 200 and live.json()["baseUrl"] == gateway_url
        for name, key in [("a", "KeyW"), ("b", "KeyS")]:
            await controls[name].send(
                json.dumps(
                    {
                        "type": "input.state",
                        "epoch": sessions[name]["epoch"],
                        "bindingSequence": bindings[name],
                        "input": cli_input(0, {"held_keys": [key]}),
                    }
                )
            )
            receipt = await message(name, "input.receipt", 0)
            assert receipt["sdkInputSequence"] >= 1
            snapshot = await message(name, "ui.snapshot")
            actor = room.slots[room.members[name * 32]["slot"]]["actorId"]
            other = room.slots[
                room.members[("b" if name == "a" else "a") * 32]["slot"]
            ]["actorId"]
            assert actor in json.dumps(snapshot) and other not in json.dumps(snapshot)
        # Credits and payload remain SDK-native through both relay hops.
        for name in ("a", "b"):
            session = sessions[name]
            media = await stack.enter_async_context(
                connect(
                    gateway_url.replace("http:", "ws:")
                    + "/v1/sessions/"
                    + session["sessionId"]
                    + "/pixels?clientId=public-"
                    + name,
                    max_size=MAX_WIRE_BYTES,
                    origin=config.player_origin,
                    subprotocols=[
                        "worldkit.stream.v1",
                        "worldkit.cap." + session["accessToken"],
                    ],
                )
            )
            generation = (await message(name, "media.config"))["mediaGeneration"]
            await controls[name].send(
                json.dumps(
                    {
                        "type": "media.credit",
                        "epoch": session["epoch"],
                        "mediaGeneration": generation,
                        "frames": 2,
                        "bytes": 2 * room.spec.frame_bytes,
                    }
                )
            )
            # A rejected credit arrives on control; waiting only for pixels
            # would hide the protocol error behind a media timeout.
            packet_task = asyncio.create_task(media.recv())
            failure_task = asyncio.create_task(message(name))
            try:
                async with asyncio.timeout(20):
                    done, _ = await asyncio.wait(
                        (packet_task, failure_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if failure_task in done:
                        failure_task.result()
                    packet = packet_task.result()
            finally:
                packet_task.cancel()
                failure_task.cancel()
                await asyncio.gather(packet_task, failure_task, return_exceptions=True)
            header, rgb = decode_native_packet(packet, session, room.spec)
            assert (
                header["sessionId"] == session["sessionId"]
                and len(rgb) == room.spec.frame_bytes
            )
        with pytest.raises(InvalidStatus):
            async with connect(
                gateway_url.replace("http:", "ws:")
                + "/v1/sessions/"
                + sessions["b"]["sessionId"]
                + "/control",
                origin=config.player_origin,
                subprotocols=[
                    "worldkit.stream.v1",
                    "worldkit.cap." + sessions["a"]["accessToken"],
                ],
            ):
                pass
        records["a"]["leaving"] = True
        await alice.close()
        entries.pop(streams["a"])
        await controls["b"].send(
            json.dumps(
                {
                    "type": "input.state",
                    "epoch": sessions["b"]["epoch"],
                    "bindingSequence": bindings["b"],
                    "input": cli_input(1, {"held_keys": ["KeyD"]}),
                }
            )
        )
        assert (await message("b", "input.receipt", 1))["sdkInputSequence"] >= 1
        assert not bob.renderer.closed


def test_public_transport_cannot_share_control_cookie_host():
    for public, player in [
        ("https://wtp.test", "https://player.test"),
        ("https://media.test", "https://wtp.test"),
    ]:
        with pytest.raises(ValueError):
            PlayerTransportConfig(public, player, "https://wtp.test")


def test_browser_input_observation_uses_original_room_log_and_receipt_identity():
    import json

    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime

    async def run():
        events = []

        async def event(identity, kind, payload):
            events.append((identity, kind, payload))

        log = SimpleNamespace(event=event, manifest={})
        entry = dict(
            game_id="g",
            identity={"player_id": "p"},
            engine=SimpleNamespace(descriptor={"sessionId": "sdk-a", "epoch": 2}),
        )
        runtime = SimpleNamespace(
            player=lambda *_, **__: entry, rooms={"g": {"log": log}}
        )
        for direction, message in [
            ("to-sdk", {"type": "input.state", "epoch": 2, "input": {"sequence": 0}}),
            (
                "from-sdk",
                {
                    "type": "input.receipt",
                    "epoch": 2,
                    "clientSequence": 0,
                    "sdkInputSequence": 5,
                },
            ),
        ]:
            await RoomRuntime.observe_player_control(
                runtime, "s", "browser-a", direction, json.dumps(message)
            )
        assert [row[1] for row in events] == [
            "viewer_input_observed",
            "viewer_input_receipt",
        ]
        assert all(
            row[2]["sdkSessionId"] == "sdk-a" and row[2]["clientId"] == "browser-a"
            for row in events
        )
        await RoomRuntime.observe_player_control(
            runtime,
            "s",
            "browser-a",
            "from-sdk",
            json.dumps({"type": "input.receipt", "epoch": 1}),
        )
        assert len(events) == 2

        async def fail(*_):
            raise ValueError("archive unavailable")

        log.event = fail
        await RoomRuntime.observe_player_control(
            runtime,
            "s",
            "browser-a",
            "to-sdk",
            json.dumps({"type": "input.release", "epoch": 2}),
        )
        assert log.manifest["viewer_input_archive_status"] == "failed"

    asyncio.run(run())


def test_sealed_input_resources_cross_gateway_room_with_exact_viewer_capability():
    async def run():
        capability = "a" * 48
        stream = "1" * 32
        entry = dict(
            game_id="2" * 32,
            identity={"player_id": "p"},
            control_admission={"status": "claimed"},
        )
        descriptor = dict(
            sessionId="sdk-a",
            epoch=2,
            accessMode="capability",
            accessToken=capability,
            baseUrl="http://127.0.0.1:65432",
            input={"abiVersion": "player-input/1"},
        )
        entry["engine"] = SimpleNamespace(descriptor=descriptor)
        config = PlayerTransportConfig(
            "https://gateway.test", "https://player.test", "https://wtp.test"
        )
        calls = []

        def sdk(request):
            calls.append(request)
            assert request.headers["authorization"] == "Bearer " + capability
            assert "cookie" not in request.headers
            assert request.url.path.startswith("/v1/sessions/sdk-a/player-input/")
            if request.url.path.endswith("/retired.js"):
                descriptor["epoch"] += 1
            return httpx.Response(
                200,
                content=b"frozen bytes",
                headers={"content-type": "application/octet-stream"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(sdk)) as upstream:
            runtime = SimpleNamespace(
                player=lambda sid, **_: (
                    entry if sid == stream else (_ for _ in ()).throw(ValueError())
                )
            )
            room = FastAPI()
            install_player_runtime(
                room, "internal", runtime, config=config, client=upstream
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(room), base_url="http://room"
            ) as room_client:

                async def session_route(session_id):
                    if session_id != "sdk-a":
                        raise ValueError()
                    return {"owner_id": "owner", "stream_id": stream, "epoch": 2}

                proxy = SimpleNamespace(
                    client=room_client,
                    token="internal",
                    owners={"owner": "http://room"},
                    directory=SimpleNamespace(session_route=session_route),
                )
                gateway = FastAPI()
                install_player_gateway(gateway, proxy, config=config)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(gateway),
                    base_url=config.public_origin,
                ) as browser:
                    headers = {
                        "origin": config.player_origin,
                        "authorization": "Bearer " + capability,
                    }
                    for asset in [
                        "manifest.json",
                        "module.js",
                        "worker.js",
                        "hands.wasm",
                        "model.task",
                    ]:
                        response = await browser.get(
                            "/v1/sessions/sdk-a/player-input/" + asset, headers=headers
                        )
                        assert (
                            response.status_code == 200
                            and response.content == b"frozen bytes"
                        )
                        assert (
                            response.headers["access-control-allow-origin"]
                            == config.player_origin
                        )
                    assert len(calls) == 5
                    path = "/v1/sessions/sdk-a/player-input/module.js"
                    assert (
                        await browser.get(
                            path, headers={"origin": config.player_origin}
                        )
                    ).status_code == 403
                    assert (
                        await browser.get(
                            path,
                            headers={**headers, "authorization": "Bearer " + "b" * 48},
                        )
                    ).status_code == 404
                    assert (
                        await browser.get(
                            path, headers={**headers, "origin": "https://evil.test"}
                        )
                    ).status_code == 403
                    assert (
                        await browser.get(
                            path.replace("sdk-a", "sdk-b"), headers=headers
                        )
                    ).status_code == 404
                    assert (
                        await browser.get(
                            path.replace("module.js", "%252e%252e/source.ts"),
                            headers=headers,
                        )
                    ).status_code == 404
                    preflight = await browser.options(
                        path,
                        headers={
                            "origin": config.player_origin,
                            "access-control-request-method": "GET",
                            "access-control-request-headers": "authorization",
                        },
                    )
                    assert preflight.status_code == 204
                    entry["control_admission"]["leaving"] = True
                    assert (await browser.get(path, headers=headers)).status_code == 404
                    assert len(calls) == 5
                    entry["control_admission"]["leaving"] = False
                    assert (
                        await browser.get(
                            path.replace("module.js", "retired.js"), headers=headers
                        )
                    ).status_code == 404

    asyncio.run(run())


@pytest.mark.parametrize("blocked", ["handshake-send", "archive"])
def test_public_control_queue_expires_behind_handshake_and_observation(blocked):
    import json

    from sglang.multimodal_gen.runtime.realtime.whitebox.player_transport import relay

    async def run():
        messages = asyncio.Queue()
        for kind in ["input.semantic.applied", "input.semantic.batch"]:
            messages.put_nowait(
                {"type": "websocket.receive", "text": json.dumps({"type": kind})}
            )
        started, release, never = asyncio.Event(), asyncio.Event(), asyncio.Event()
        sent = []

        async def block():
            started.set()
            await release.wait()

        class Browser:
            async def receive(self):
                return await messages.get()

        class SDK:
            async def send(self, payload):
                kind = json.loads(payload)["type"]
                if kind == "input.semantic.applied" and blocked == "handshake-send":
                    await block()
                sent.append(kind)
                if kind == "input.semantic.batch":
                    messages.put_nowait({"type": "websocket.disconnect"})

            async def __aiter__(self):
                await never.wait()
                yield "unused"

        async def observe(direction, payload):
            if (
                blocked == "archive"
                and direction == "to-sdk"
                and json.loads(payload)["type"] == "input.semantic.applied"
            ):
                await block()

        task = asyncio.create_task(relay(Browser(), SDK(), observe=observe))
        try:
            await asyncio.wait_for(started.wait(), 1)
            await asyncio.sleep(0.22)
            release.set()
            with pytest.raises(asyncio.TimeoutError, match="input"):
                await asyncio.wait_for(task, 1)
            assert "input.semantic.batch" not in sent
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_public_control_records_intent_before_an_immediate_committed_receipt():
    import json

    from sglang.multimodal_gen.runtime.realtime.whitebox.player_idle import (
        PlayerInputActivity,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.player_transport import relay

    async def run():
        activity = PlayerInputActivity()
        activity.start(0)
        incoming, receipts = asyncio.Queue(), asyncio.Queue()
        observed = asyncio.Event()
        batch = dict(
            type="input.semantic.batch",
            generation=1,
            batch=dict(sequence=1, stateRevision=1),
        )
        receipt = dict(
            type="input.semantic.receipt",
            clientId="c",
            generation=1,
            sequence=1,
            stateRevision=1,
            simulationTick=1,
            meaningful=True,
        )
        incoming.put_nowait(dict(type="websocket.receive", text=json.dumps(batch)))

        class Browser:
            async def receive(self):
                return await incoming.get()

            async def send_text(self, payload):
                observed.set()

        class SDK:
            async def send(self, payload):
                receipts.put_nowait(json.dumps(receipt))
                # A peer may consume input before a flow-controlled send returns.
                await observed.wait()

            async def __aiter__(self):
                while True:
                    yield await receipts.get()

        async def observe(direction, payload):
            activity.observe("c", direction, json.loads(payload), 10)

        task = asyncio.create_task(relay(Browser(), SDK(), observe=observe))
        try:
            await asyncio.wait_for(observed.wait(), 1)
            await asyncio.sleep(0)
            assert activity.last_input == 10
            assert not activity.pending
            assert not activity.is_expired(60, 60)
        finally:
            incoming.put_nowait(dict(type="websocket.disconnect"))
            await task

    asyncio.run(run())
