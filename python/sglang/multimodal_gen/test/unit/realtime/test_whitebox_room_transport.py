# SPDX-License-Identifier: Apache-2.0
"""Private transport regressions; optional live SDK public-API contract checks."""

import asyncio
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec


class Renderer:
    def __init__(self, **kwargs):
        self.events = []
        self.close_count = 0

    async def consume_ui(self, value):
        self.events.append(value)

    async def close(self):
        self.close_count += 1


def view_for(base, player="alice", private=True):
    descriptor = dict(
        sessionId=player,
        epoch=0,
        baseUrl=base,
        accessToken=("a" if player == "alice" else "b") * 48,
        media={"kind": "pixels", "pixelFormat": "rgb24"},
        fps=24,
        streams={
            channel: {
                "url": base.replace("http:", "ws:") + f"/v1/sessions/{player}/{path}"
            }
            for channel, path in [("ui", "control"), ("pixels", "pixels")]
        },
        ui={"catalog": {"actions": {"ping": {}}}},
        worldBuildHash="a" * 64,
        uiBundleHash="b" * 64,
    )
    if private:
        descriptor["accessMode"] = "capability"
    return NativeCLIView(
        SimpleNamespace(
            descriptor=descriptor,
            spec=VideoSpec(32, 32),
            prepared_delivery=None,
            renderer_factory=Renderer,
        )
    )


@pytest.mark.parametrize("admission", [None, "spectator"])
def test_binding_requires_prior_controller_admission(admission):
    """An input offer alone must never grant the private view control."""

    async def run():
        view = view_for("http://127.0.0.1:1")

        class Control:
            def __aiter__(self):
                return self.messages()

            async def messages(self):
                if admission:
                    yield json.dumps(
                        {"type": "session.ready", "epoch": 0, "role": admission}
                    )
                yield json.dumps(
                    {"type": "input.binding", "epoch": 0, "bindingSequence": 7}
                )

            async def send(self, wire):
                pass

        view.control = Control()
        with pytest.raises(ValueError, match="binding"):
            await view._control()
        assert not view.ready.is_set()

    asyncio.run(run())


def native_packet(player):
    pixels = bytes([17 if player == "alice" else 29]) * (32 * 32 * 3)
    header = dict(
        kind="worldkit-pixel-frame",
        schemaVersion=2,
        pixelFormat="rgb24",
        colorSpace="srgb",
        widthPixels=32,
        heightPixels=32,
        rowStrideBytes=96,
        payloadBytes=len(pixels),
        worldBuildHash="a" * 64,
        uiBundleHash="b" * 64,
        input=dict(
            protocolVersion=1,
            sessionId=player,
            epoch=0,
            mediaGeneration=1,
            outputFrameId=1,
            outputPtsUs=1,
            uiRevision=1,
            uiCompleteThroughUs=1,
            widthPixels=32,
            heightPixels=32,
            source=dict(
                presentationId=player,
                sdkEpoch=0,
                sourceFrameId=1,
                sourceTimeUs=1,
                simulationTick=1,
                worldRevision=0,
            ),
        ),
    )
    encoded = json.dumps(header).encode()
    return len(encoded).to_bytes(4, "big") + encoded + pixels


def test_private_transport_waits_binding_and_isolates_two_view_ui_pixels_and_input():
    async def run():
        from websockets.asyncio.server import serve

        connections, messages = [], {"alice": [], "bob": []}
        offer = asyncio.Event()
        received = {player: asyncio.Event() for player in messages}
        media, media_ready = {}, {player: asyncio.Event() for player in messages}

        async def socket(ws):
            parsed = urlsplit(ws.request.path)
            player, channel = parsed.path.split("/")[-2:]
            query = parse_qs(parsed.query)
            assert "token" not in query
            assert (
                ws.request.headers["Sec-WebSocket-Protocol"]
                == "worldkit.stream.v1, worldkit.cap."
                + ("a" if player == "alice" else "b") * 48
            )
            connections.append((player, query["clientId"][0]))
            if channel == "control":
                await ws.send(
                    json.dumps(
                        {"type": "session.ready", "epoch": 0, "role": "controller"}
                    )
                )
                await offer.wait()
                await ws.send(
                    json.dumps(
                        {"type": "input.binding", "epoch": 0, "bindingSequence": 7}
                    )
                )
                await ws.send(
                    json.dumps(
                        {
                            "type": "ui.snapshot",
                            "epoch": 0,
                            "snapshot": {
                                "effectiveSourceTimeUs": 0,
                                "revision": 1,
                                "state": {"player": player},
                            },
                        }
                    )
                )
                await ws.send(
                    json.dumps(
                        dict(
                            type="media.config",
                            epoch=0,
                            mediaGeneration=1,
                            config=dict(
                                kind="pixels",
                                pixelFormat="rgb24",
                                widthPixels=32,
                                heightPixels=32,
                                rowStrideBytes=96,
                                colorSpace="srgb",
                                fps=24,
                            ),
                        )
                    )
                )
                async for wire in ws:
                    message = json.loads(wire)
                    if message["type"] == "media.credit":
                        await media_ready[player].wait()
                        await media[player].send(native_packet(player))
                        continue
                    messages[player].append(message)
                    if message["type"] == "input.action":
                        received[player].set()
            else:
                media[player] = ws
                media_ready[player].set()
                await ws.wait_closed()

        async with serve(
            socket, "127.0.0.1", 0, subprotocols=["worldkit.stream.v1"]
        ) as server:
            base = "http://127.0.0.1:" + str(server.sockets[0].getsockname()[1])
            alice, bob = view_for(base), view_for(base, "bob")
            starts = [asyncio.create_task(view.start()) for view in (alice, bob)]
            try:
                # Exercise input while start is waiting for the private binding.
                while any(view.control_task is None for view in (alice, bob)):
                    await asyncio.sleep(0)
                early = asyncio.create_task(
                    alice.send_input(
                        0, {"held_keys": ["KeyW"], "actions": [{"name": "ping"}]}
                    )
                )
                await asyncio.sleep(0.01)
                assert not early.done()
                assert messages["alice"] == []
                offer.set()
                await asyncio.gather(*starts)
                await early
                await bob.send_input(
                    0, {"held_keys": ["KeyS"], "actions": [{"name": "ping"}]}
                )
                await asyncio.gather(
                    *(asyncio.wait_for(event.wait(), 1) for event in received.values())
                )
                for player, key in [("alice", "KeyW"), ("bob", "KeyS")]:
                    assert [item["type"] for item in messages[player]] == [
                        "input.binding.applied",
                        "input.state",
                        "input.action",
                    ]
                    assert all(
                        item["bindingSequence"] == 7 for item in messages[player]
                    )
                    assert messages[player][1]["input"]["heldKeys"] == [key]
                streams = [view.frames() for view in (alice, bob)]
                try:
                    frames = await asyncio.gather(
                        *(anext(stream) for stream in streams)
                    )
                    assert frames[0].rgb == bytes([17]) * (32 * 32 * 3)
                    assert frames[1].rgb == bytes([29]) * (32 * 32 * 3)
                    for frame, player in zip(frames, ("alice", "bob")):
                        assert (
                            frame.provenance["cli_frame_header"]["sessionId"] == player
                        )
                        assert frame.states["ui_events"][0]["snapshot"]["state"] == {
                            "player": player
                        }
                finally:
                    await asyncio.gather(*(stream.aclose() for stream in streams))
                assert alice.renderer.events[0]["snapshot"]["state"] == {
                    "player": "alice"
                }
                assert bob.renderer.events[0]["snapshot"]["state"] == {"player": "bob"}
                assert (
                    len({client for player, client in connections if player == "alice"})
                    == 1
                )
                assert len({client for player, client in connections}) == 2
                await alice.close()
                await alice.close()
                assert alice.renderer.close_count == 1
                assert alice.control_task.done()
                assert alice.control.state.name == "CLOSED"
                assert alice.media.state.name == "CLOSED"
                assert bob.renderer.close_count == 0
                assert not bob.closed
            finally:
                offer.set()
                for task in starts:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*starts, return_exceptions=True)
                await asyncio.gather(alice.close(), bob.close())

    asyncio.run(run())


def test_singleplayer_retains_url_token_and_does_not_require_binding():
    async def run():
        from websockets.asyncio.server import serve

        observed, received = [], asyncio.Event()

        async def socket(ws):
            parsed = urlsplit(ws.request.path)
            assert parse_qs(parsed.query)["token"] == ["a" * 48]
            assert "Sec-WebSocket-Protocol" not in ws.request.headers
            if parsed.path.endswith("/control"):
                await ws.send(
                    json.dumps(
                        {"type": "session.ready", "epoch": 0, "role": "controller"}
                    )
                )
                async for wire in ws:
                    observed.append(json.loads(wire))
                    received.set()
            else:
                await ws.wait_closed()

        async with serve(socket, "127.0.0.1", 0) as server:
            view = view_for(
                "http://127.0.0.1:" + str(server.sockets[0].getsockname()[1]),
                private=False,
            )
            try:
                await view.start()
                await view.send_input(0, {"held_keys": ["KeyW"]})
                await asyncio.wait_for(received.wait(), 1)
                assert observed[0]["type"] == "input.state"
                assert "bindingSequence" not in observed[0]
            finally:
                await view.close()

    asyncio.run(run())


def sdk_descriptor(base):
    binding = dict(
        schemaVersion=1,
        viewportId="viewport-alice",
        viewportBindingRevision=2,
        room=dict(
            protocolVersion=1,
            roomId="room",
            roomEpoch=1,
            runtimeHash="runtime",
            contentHash="content",
        ),
        playerSessionId="alice",
        connectionEpoch=2,
        mode="cloud",
        render=dict(sessionId="render-alice", epoch=3),
    )
    return dict(
        protocolVersion=1,
        sessionId="render-alice",
        epoch=3,
        worldId="world",
        worldBuildHash="a" * 64,
        uiBundleHash="b" * 64,
        baseUrl=base,
        accessToken="a" * 48,
        accessMode="capability",
        media=dict(kind="pixels", pixelFormat="rgb24", colorSpace="srgb"),
        playerBinding=binding,
        audience=dict(
            kind="player",
            roomId="room",
            roomEpoch=1,
            playerSessionId="alice",
            revision=1,
        ),
        width=160,
        height=90,
        fps=24,
        status="running",
        pixelState={},
        ui=dict(
            catalog=dict(schemaVersion=1, catalogId="ui", components={}, actions={}),
            document=dict(
                schemaVersion=1,
                catalogId="ui",
                designViewport=dict(width=32, height=32),
                spec=dict(
                    root="root",
                    elements=dict(root=dict(type="HudLayer", props={}, children=[])),
                ),
                transitions={},
            ),
            stateSchema=dict(type="object"),
            moduleUrl=base + "/ui/ui.js",
            moduleSha256="b" * 64,
            styles=[],
        ),
    )


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "render-session",
        "render-epoch",
        "audience",
        "access-mode",
        "file-mode",
        "stale-binding",
        "missing-render",
        "public-input",
    ],
)
def test_member_helper_uses_actual_sdk_read_session_and_checks_private_identity(
    tmp_path, bad
):
    sdk = os.environ.get("WHITEBOX_TEST_CLI_SDK")
    if not sdk:
        pytest.skip(
            "Set WHITEBOX_TEST_CLI_SDK to run against the installed SDK public packages"
        )

    async def run():
        requests = []

        async def http(reader, writer):
            request = await reader.readuntil(b"\r\n\r\n")
            requests.append(request.decode())
            payload = json.dumps(descriptor).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                + str(len(payload)).encode()
                + b"\r\n\r\n"
                + payload
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(http, "127.0.0.1", 0)
        descriptor = sdk_descriptor(
            "http://127.0.0.1:" + str(server.sockets[0].getsockname()[1])
        )
        connection = {
            key: descriptor[key]
            for key in ["sessionId", "baseUrl", "accessToken", "accessMode"]
        }
        if bad == "public-input":
            descriptor["input"] = dict(
                schemaVersion=1,
                abiVersion="player-input/1",
                manifest=dict(path="player-input/manifest.json", sha256="c" * 64),
                adapters=[],
                baseUrl=descriptor["baseUrl"] + "/v1/sessions/render-alice/",
            )
            descriptor["ui"]["moduleUrl"] = (
                descriptor["baseUrl"]
                + "/ui/"
                + descriptor["uiBundleHash"]
                + "/module.js"
            )
        binding = copy.deepcopy(descriptor["playerBinding"])
        if bad == "render-session":
            binding["render"]["sessionId"] = "foreign"
            descriptor["playerBinding"] = copy.deepcopy(binding)
        elif bad == "render-epoch":
            binding["render"]["epoch"] = 4
            descriptor["playerBinding"] = copy.deepcopy(binding)
        elif bad == "audience":
            descriptor["audience"]["playerSessionId"] = "bob"
        elif bad == "access-mode":
            descriptor.pop("accessMode")
        elif bad == "stale-binding":
            descriptor["playerBinding"]["connectionEpoch"] = 1
        elif bad == "missing-render":
            binding.pop("render")
        file = tmp_path / "member.json"
        file.write_text(
            json.dumps(
                dict(
                    kind="worldkit-room-member",
                    schemaVersion=1,
                    roomId="room",
                    session=connection,
                    binding=binding,
                )
            )
        )
        file.chmod(0o644 if bad == "file-mode" else 0o600)
        helper = (
            Path(__file__).parents[3]
            / "runtime/realtime/whitebox/worldkit_room_session.mjs"
        )
        entry = Path(sdk) / "packages/cli/src/main.ts"
        env = {**os.environ, "WHITEBOX_CLI_ENTRYPOINT": str(entry)}
        try:
            process = await asyncio.create_subprocess_exec(
                "node",
                "--import",
                "tsx",
                str(helper),
                cwd=sdk,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(
                    json.dumps(
                        dict(
                            memberFile=str(file),
                            uiRoot=str(tmp_path / "ui"),
                            **(
                                {"publicOrigin": "https://gateway.test"}
                                if bad == "public-input"
                                else {}
                            ),
                        )
                    ).encode()
                ),
                15,
            )
            if bad in (None, "public-input"):
                assert process.returncode == 0, stderr.decode()
                result = json.loads(stdout)
                assert (
                    result["epoch"] == 3
                )  # Minimal SessionConnection deliberately has no epoch.
                if bad == "public-input":
                    assert (
                        result["input"]["baseUrl"]
                        == "https://gateway.test/v1/sessions/render-alice/"
                    )
                    assert (
                        result["input"]["manifest"] == descriptor["input"]["manifest"]
                    )
                else:
                    assert result["streams"]["pixels"]["url"].endswith(
                        "/v1/sessions/render-alice/pixels"
                    )
                assert result["playerBinding"]["connectionEpoch"] == 2
                assert len(requests) == 1
                assert "Authorization: Bearer " + "a" * 48 in requests[0]
            else:
                assert process.returncode != 0
                assert stdout == b""
                assert stderr.strip() in {
                    b"ROOM_MEMBER_FILE_INVALID",
                    b"ROOM_MEMBER_IDENTITY_INVALID",
                }
                assert b"aaaaa" not in stderr
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())


@pytest.mark.parametrize(
    "case",
    ["model", "public", "legacy", "missing", "rotated", "foreign", "browser-model"],
)
def test_member_session_selects_private_model_purpose_without_public_leak(
    tmp_path, monkeypatch, case
):
    """Wrong descriptor selection must fail at real authenticated SDK readSession."""
    from sglang.multimodal_gen.runtime.realtime.whitebox.worldkit_room import (
        read_member_session,
    )

    sdk = os.environ.get("WHITEBOX_TEST_CLI_SDK")
    if not sdk:
        pytest.skip("Set WHITEBOX_TEST_CLI_SDK to run the SDK subprocess contract")
    entry = Path(sdk) / "packages/cli/bin/worldkit.mjs"
    monkeypatch.setenv("WHITEBOX_CLI_ENTRYPOINT", str(entry))
    monkeypatch.setenv("WHITEBOX_NODE", "node")

    async def run():
        requests = []
        ordinary, model = "a" * 48, "d" * 48
        current_model = "e" * 48 if case == "rotated" else model

        async def http(reader, writer):
            request = (await reader.readuntil(b"\r\n\r\n")).decode()
            is_model = f"Authorization: Bearer {current_model}\r\n" in request
            is_viewer = f"Authorization: Bearer {ordinary}\r\n" in request
            requests.append("model" if is_model else "viewer" if is_viewer else "bad")
            response = dict(
                descriptor, accessToken=current_model if is_model else ordinary
            )
            payload = json.dumps(response).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                + str(len(payload)).encode()
                + b"\r\n\r\n"
                + payload
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(http, "127.0.0.1", 0)
        descriptor = sdk_descriptor(
            f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        )
        descriptor["ui"]["moduleUrl"] = (
            descriptor["baseUrl"] + "/ui/" + descriptor["uiBundleHash"] + "/ui.js"
        )
        if case != "legacy":
            descriptor["modelContext"] = dict(
                protocol="worldkit-model-context-v1",
                required=True,
                configHash="c" * 64,
                gameBuildHash="a" * 64,
                compilerVersion=1,
                visible=False,
            )
        connection = {
            key: descriptor[key]
            for key in ("sessionId", "baseUrl", "accessToken", "accessMode")
        }
        record = dict(
            kind="worldkit-room-member",
            schemaVersion=1,
            roomId="room",
            session=connection,
            binding=descriptor["playerBinding"],
            token="private-room-token",
        )
        if case not in {"legacy", "missing"}:
            record["modelSession"] = dict(connection, accessToken=current_model)
            if case == "foreign":
                record["modelSession"]["sessionId"] = "foreign"
        file = tmp_path / "member-purpose.json"
        file.write_text(json.dumps(record))
        file.chmod(0o600)
        member = dict(memberFile=str(file), localResources=dict(uiRoot=str(tmp_path)))
        public_origin = (
            "https://public.example" if case in {"public", "browser-model"} else None
        )
        purpose = "viewer" if case == "public" else "model-source"
        try:
            if case in {"missing", "foreign", "browser-model"}:
                code = (
                    "ROOM_MODEL_SESSION_REQUIRED"
                    if case == "missing"
                    else "ROOM_SESSION_UNAVAILABLE"
                )
                with pytest.raises(ValueError, match=code):
                    await read_member_session(
                        member, entry, purpose=purpose, public_origin=public_origin
                    )
                if case != "missing":
                    assert requests == []
                return
            result = await read_member_session(
                member, entry, purpose=purpose, public_origin=public_origin
            )
            expected = ordinary if case in {"public", "legacy"} else current_model
            assert result["accessToken"] == expected
            assert requests == ["viewer" if expected == ordinary else "model"]
            assert result["epoch"] == 3
            assert "private-room-token" not in json.dumps(result)
            if case == "public":
                assert current_model not in json.dumps(result)
                assert "modelSession" not in result
                assert result["baseUrl"] == public_origin
                assert result["ui"]["moduleUrl"].startswith(public_origin)
            else:
                assert result["streams"]["pixels"]["url"].endswith(
                    "/v1/sessions/render-alice/pixels"
                )
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())
