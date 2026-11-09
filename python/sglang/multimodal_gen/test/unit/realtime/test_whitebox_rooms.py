# SPDX-License-Identifier: Apache-2.0
"""Local-only multiplayer acceptance: real HTTP/WS, mocked CLI/model, no cloud writes."""

import asyncio
import json
import socket
from contextlib import AsyncExitStack, asynccontextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import httpx
import msgspec
import pytest
import uvicorn
from test_realtime_coordinator import _heartbeat
from test_whitebox import H3
from websockets.asyncio.client import connect

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.room_client import (
    RemoteEngine,
    RoomEngineFactory,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.room_mock import shared_mock_room
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import (
    RoomRuntime,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import (
    create_app as runtime_app,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
    new_id,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig


def test_distributed_ids_atomic_creation_and_owner_fencing():
    async def run():
        ids = {new_id() for _ in range(10000)}
        assert len(ids) == 10000 and all(
            len(value) == 32 and value[12] == "4" for value in ids
        )
        now = [100.0]
        store = MemoryRoomStore()
        directories = [
            RoomDirectory(
                store, {"pod-a": "http://a", "pod-b": "http://b"}, clock=lambda: now[0]
            )
            for _ in range(4)
        ]
        game = new_id()
        routes = await asyncio.gather(
            *(
                directory.resolve(game, "source/world", create=True)
                for directory in directories
            )
        )
        assert all(route == routes[0] for route in routes)
        owner, epoch = routes[0]["owner_id"], new_id()
        await directories[0].touch(game, owner, epoch)
        with pytest.raises(ValueError, match="owner"):
            await directories[1].touch(game, owner, new_id())
        with pytest.raises(ValueError, match="another world"):
            await directories[2].resolve(game, "wrong")
        now[0] += 31
        with pytest.raises(ValueError, match="ended"):
            await directories[3].resolve(game, "source/world", create=True)
        with pytest.raises(ValueError, match="ended"):
            await directories[0].touch(game, owner, epoch)

    asyncio.run(run())


def test_runtime_routing_is_fixed_and_failed_join_does_not_end_existing_player(
    tmp_path,
):
    async def run():
        spec, token = VideoSpec(32, 32), "only-in-tests"
        directory = RoomDirectory(MemoryRoomStore(), {"a": "http://a", "b": "http://b"})
        created = []

        async def single_view(**kwargs):
            room = await shared_mock_room(**kwargs)
            room.max_players = 1
            created.append(room)
            return room

        game = new_id()
        route = await directory.resolve(game, "world", create=True)
        owner = route["owner_id"]
        runtime = RoomRuntime(
            directory=directory,
            owner_id=owner,
            driver_factory=single_view,
            spool=tmp_path,
            spec=spec,
        )

        def request():
            return dict(
                game_id=game,
                world_id="world",
                player_id=new_id(),
                stream_id=new_id(),
                session_id=new_id(),
                user_id="test",
                trace_id="fixture",
                spec={"width": 32, "height": 32, "fps": 24},
            )

        first = await runtime.join(request())
        # Reconfiguring the available owners must not rehash an existing room.
        directory.owners = {"new-pod": "http://new-pod"}
        assert (await directory.resolve(game, "world"))["owner_id"] == owner
        with pytest.raises(ValueError, match="another player"):
            await runtime.join(request())
        assert len(runtime.players) == 1 and len(created) == 1
        assert (await runtime.next_frame(first["stream_id"]))["index"] == 0
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=runtime_app(runtime, token)),
            base_url="http://room",
        ) as client:
            response = await client.post("/internal/players/join", json=request())
            assert response.status_code == 403
        # A new process with the same configured Pod name cannot adopt this room.
        restarted = RoomRuntime(
            directory=directory,
            owner_id=owner,
            driver_factory=single_view,
            spool=tmp_path / "restarted",
            spec=spec,
        )
        with pytest.raises(ValueError, match="owner"):
            await restarted.join(request())
        await restarted.close()
        await runtime.leave(first["stream_id"])
        await runtime.close()
        with pytest.raises(ValueError, match="ended"):
            await directory.resolve(game, "world", create=True)

    asyncio.run(run())


@asynccontextmanager
async def server(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    address = "http://127.0.0.1:" + str(sock.getsockname()[1])
    service = uvicorn.Server(
        uvicorn.Config(
            app, log_level="error", lifespan="on", timeout_graceful_shutdown=3
        )
    )
    task = asyncio.create_task(service.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(5):
            while not service.started:
                if task.done():
                    task.result()
                await asyncio.sleep(0.01)
        yield address
    finally:
        service.should_exit = True
        await asyncio.wait_for(task, 8)
        sock.close()


def test_two_gateways_share_room_but_hold_two_exclusive_slots_and_finalize_archives(
    tmp_path,
):
    from sglang.multimodal_gen.runtime.entrypoints.realtime_coordinator_server import (
        create_app as coordinator_app,
    )
    from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
        HTTPCoordinatorClient,
    )
    from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
        create_app as gateway_app,
    )
    from sglang.multimodal_gen.runtime.realtime.coordinator import (
        InMemoryCoordinatorStore,
        RealtimeCoordinator,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import ArchiveManager

    async def run():
        spec, token = VideoSpec(32, 32), "local-test-only"
        reservations, h3_connections, uploaded = {}, [], {}

        class Reservation:
            async def reserve(self, slot, **kw):
                key = (slot.worker_id, slot.slot_index)
                assert key not in reservations
                reservations[key] = kw["session_id"]

            async def release(self, slot, **kw):
                reservations.pop((slot.worker_id, slot.slot_index), None)

        class Store:
            async def upload(self, path, key):
                uploaded[key] = path.read_bytes()

            def playback_url(self, key):
                return "https://archive.invalid/" + key

        store = InMemoryCoordinatorStore(ttl_s=10, worker_ttl_s=30)
        coordinator = RealtimeCoordinator(
            store, wait_timeout_s=0, reservation_client=Reservation()
        )
        await coordinator.heartbeat(
            _heartbeat(
                "h3",
                "denoiser",
                capacity=2,
                model_revision="h3",
                vae_fingerprint="local",
            )
        )
        directory = RoomDirectory(MemoryRoomStore(), {"pod-a": "http://placeholder"})
        room_archives = ArchiveManager(Store())
        runtime = RoomRuntime(
            directory=directory,
            owner_id="pod-a",
            driver_factory=shared_mock_room,
            spool=tmp_path / "rooms",
            spec=spec,
            archives=room_archives,
        )

        async def worker_connect(url, **kwargs):
            from urllib.parse import parse_qs, urlsplit

            query = parse_qs(urlsplit(url).query)
            assert query["session_id"][0] in reservations.values()
            h3 = H3(spec)
            h3_connections.append((query, h3))
            return h3

        async with AsyncExitStack() as stack:
            runtime_url = await stack.enter_async_context(
                server(runtime_app(runtime, token))
            )
            directory.owners["pod-a"] = runtime_url
            coordinator_url = await stack.enter_async_context(
                server(
                    coordinator_app(
                        coordinator, room_directory=directory, room_token=token
                    )
                )
            )
            gateways, archives = [], []
            for i in range(2):
                manager = ArchiveManager(Store())
                archives.append(manager)
                factory = RoomEngineFactory(coordinator_url, token)
                config = WhiteboxConfig(
                    engine_factory=factory,
                    ui_factory=lambda **_: None,
                    spool=tmp_path / f"gateway-{i}",
                    spec=spec,
                    startup_seconds=5,
                    play_seconds=1.5,
                    drain_seconds=5,
                )
                client = HTTPCoordinatorClient(coordinator_url)
                app = gateway_app(
                    client,
                    model_revision="h3",
                    vae_fingerprint="local",
                    internal_output_url="ws://unused",
                    connect_factory=worker_connect,
                    whitebox_config=config,
                    whitebox_archives=manager,
                    release_grace_s=0,
                    lease_renew_interval_s=0.2,
                )
                gateways.append(await stack.enter_async_context(server(app)))

            async def join(index, user, game_id=None):
                ws = await stack.enter_async_context(
                    connect(
                        gateways[index].replace("http", "ws")
                        + "/v1/realtime_video/generate?user_id="
                        + user
                    )
                )
                await ws.send(
                    msgspec.msgpack.encode(
                        dict(
                            type="init",
                            whitebox={
                                "world_id": "source/world",
                                **({"game_id": game_id} if game_id else {}),
                            },
                            prompt=user,
                        )
                    )
                )
                return ws, msgspec.msgpack.decode(await asyncio.wait_for(ws.recv(), 5))

            first, ready_a = await join(0, "a")
            assert ready_a["type"] == "session.ready", ready_a
            game_id = ready_a["multiplayer"]["game_id"]
            second, ready_b = await join(1, "b", game_id)
            assert ready_b["multiplayer"]["game_id"] == game_id
            assert ready_a["session_id"] != ready_b["session_id"]
            assert len(runtime.rooms) == 1 and len(runtime.players) == 2
            assert len(reservations) == 2 and len(h3_connections) == 2
            assert (
                len({query["coordinator_token"][0] for query, _ in h3_connections}) == 2
            )
            third, rejected = await join(1, "c", game_id)
            assert rejected["type"] == "error" and "CAPACITY_EXHAUSTED" in str(rejected)
            assert len(h3_connections) == 2

            for ws in (first, second):
                await ws.send(
                    msgspec.msgpack.encode(
                        {
                            "type": "event",
                            "kind": "input",
                            "sequence": 0,
                            "payload": {"held_keys": ["KeyW"]},
                        }
                    )
                )
            await first.send(
                msgspec.msgpack.encode(
                    {"type": "event", "kind": "prompt", "payload": "changed-a"}
                )
            )

            async def drain(ws):
                frames = []
                while True:
                    message = msgspec.msgpack.decode(
                        await asyncio.wait_for(ws.recv(), 6)
                    )
                    if message["type"] == "session.completed":
                        assert message["frames"] == len(frames)
                        # Match the public SDK: close only after draining all
                        # media. A non-closing peer uses the bounded tail budget.
                        await ws.close()
                        return frames
                    assert message["type"] == "frame_batch", message
                    frames.append(message)

            # Finish A early; B keeps rendering and exits by its server deadline.
            await first.send(
                msgspec.msgpack.encode(
                    {"type": "event", "kind": "finish", "payload": {}}
                )
            )
            frames_a = await drain(first)
            assert len(frames_a) > 0
            async with asyncio.timeout(3):
                while len(reservations) == 2:
                    await asyncio.sleep(0.01)
            assert len(reservations) == 1 and len(runtime.rooms) == 1
            frames_b = await drain(second)
            assert len(frames_b) > len(frames_a)
            assert any(message["states"]["tick"] == 2 for message in frames_b)
            async with asyncio.timeout(3):
                while reservations or runtime.rooms:
                    await asyncio.sleep(0.01)
            await asyncio.gather(
                *(manager.close() for manager in archives), room_archives.close()
            )
            for i, ready in enumerate((ready_a, ready_b)):
                root = tmp_path / f"gateway-{i}" / ready["recording_id"]
                manifest = json.loads((root / "manifest.json").read_text())
                assert manifest["recording_status"] == "complete"
                assert manifest["upload_status"] == "complete"
                assert manifest["multiplayer"]["game_id"] == game_id
                assert "/streams/" + ready["recording_id"] in manifest["object_prefix"]
                pts = []
                for name in ("whitebox", "model", "ui_rendered"):
                    with av.open(str(root / (name + ".mp4"))) as video:
                        assert video.streams.video[0].codec_context.name == "h264"
                        pts.append([f.pts * f.time_base for f in video.decode(video=0)])
                assert pts[0] == pts[1] == pts[2]
                assert pts[0] == [Fraction(n, 24) for n in range(len(pts[0]))]
                async with httpx.AsyncClient() as http:
                    report = await http.get(
                        gateways[i]
                        + "/v1/whitebox/recordings/"
                        + ready["recording_id"],
                        headers={"Authorization": "Bearer " + ready["report_token"]},
                    )
                    assert report.status_code == 200
                    body = report.json()
                    assert body["manifest_url"] and len(body["videos"]) == 3
                    assert body["multiplayer"]["game_id"] == game_id
                    assert any(
                        row["stage"] == "whitebox.remote_room_ui_render"
                        for row in body["timings"]
                    )
            room_manifest = next((tmp_path / "rooms").glob("*/manifest.json"))
            room_data = json.loads(room_manifest.read_text())
            assert len(room_data["players"]) == 2
            events = [
                json.loads(line)
                for line in (room_manifest.parent / "inputs.jsonl")
                .read_text()
                .splitlines()
            ]
            assert [event["room_seq"] for event in events] == list(
                range(1, len(events) + 1)
            )
            assert len({e["player_id"] for e in events if e["kind"] == "input"}) == 2
            assert any(e["kind"] == "prompt" for e in events)
            assert room_data["object_prefix"] + "manifest.json" in uploaded
            with pytest.raises(ValueError, match="ended"):
                await directory.resolve(game_id, "source/world")

    asyncio.run(run())


def test_cli_v1_decodes_real_vp8_preserves_source_identity_and_sends_documented_controls():
    import numpy as np
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import (
        CLIRoom,
        decode_packet,
        validate_descriptor,
    )

    async def run():
        spec, received = VideoSpec(160, 96), []
        epoch, session_id = 0, "fixture-cli-session"
        encoder = av.CodecContext.create("libvpx", "w")
        encoder.width, encoder.height = spec.width, spec.height
        encoder.time_base, encoder.framerate = Fraction(1, 24), Fraction(24, 1)
        encoder.pix_fmt = "yuv420p"
        encoder.options = {"deadline": "realtime", "lag-in-frames": "0"}
        frame = av.VideoFrame.from_ndarray(
            np.full((spec.height, spec.width, 3), 80, dtype=np.uint8), format="rgb24"
        )
        frame.pts = 0
        payload = bytes(encoder.encode(frame)[0])
        header = dict(
            protocolVersion=1,
            sessionId=session_id,
            epoch=epoch,
            mediaGeneration=1,
            outputFrameId=12,
            outputPtsUs=500000,
            type="key",
            uiRevision=2,
            uiCompleteThroughUs=500000,
            widthPixels=spec.width,
            heightPixels=spec.height,
            payloadBytes=len(payload),
            source=dict(
                presentationId="p12",
                sdkEpoch=0,
                sourceFrameId=40,
                sourceTimeUs=500000,
                simulationTick=71,
                worldRevision=4,
            ),
        )
        encoded = json.dumps(header).encode()
        wire = len(encoded).to_bytes(4, "big") + encoded + payload
        app = FastAPI()
        controls = []

        @app.websocket("/v1/sessions/{session}/control")
        async def control(ws: WebSocket, session: str):
            assert ws.query_params["token"] == "private-local-token"
            assert ws.query_params["ui"] == "1"
            controls.append(ws.query_params["clientId"])
            await ws.accept()
            await ws.send_json(dict(type="session.ready", epoch=0, role="controller"))
            await ws.send_json(
                dict(
                    type="media.config",
                    epoch=0,
                    mediaGeneration=1,
                    config=dict(
                        codec="vp8", codedWidth=spec.width, codedHeight=spec.height
                    ),
                )
            )
            await ws.send_json(
                dict(
                    type="ui.snapshot",
                    epoch=0,
                    snapshot=dict(
                        revision=2,
                        effectiveSourceTimeUs=400000,
                        completeThroughUs=500000,
                        state={"unknown-future-state": [1, 2]},
                        activeAnimations=[],
                    ),
                )
            )
            try:
                while True:
                    received.append(await ws.receive_json())
            except WebSocketDisconnect:
                pass

        @app.websocket("/v1/sessions/{session}/media")
        async def media(ws: WebSocket, session: str):
            assert ws.query_params["clientId"] == controls[0]
            await ws.accept()
            await ws.send_bytes(wire)
            try:
                await ws.receive_bytes()
            except WebSocketDisconnect:
                pass

        class Renderer:
            metadata = {"mode": "fixture"}
            events = []

            async def consume_ui(self, event):
                self.events.append(event)

            async def render(self, rgb, frame, spec):
                assert frame.provenance["cli_frame_header"] == header
                return rgb

            async def close(self):
                pass

        async with server(app) as base:
            descriptor = dict(
                kind="worldkit-stream-session",
                schemaVersion=1,
                status="ready",
                protocolVersion=1,
                sessionId=session_id,
                epoch=0,
                baseUrl=base,
                accessToken="private-local-token",
                width=spec.width,
                height=spec.height,
                fps=spec.fps,
                ui={"catalog": {"actions": {"damage": {}}}},
                streams={
                    "video": {
                        "url": base.replace("http", "ws")
                        + f"/v1/sessions/{session_id}/media",
                        "codec": "vp8",
                    },
                    "ui": {
                        "url": base.replace("http", "ws")
                        + f"/v1/sessions/{session_id}/control"
                    },
                },
            )
            validate_descriptor(descriptor, spec)
            room = CLIRoom(
                SimpleNamespace(returncode=0),
                descriptor,
                spec,
                lambda **kw: Renderer(),
                "cli-fixture",
            )
            view = await room.join(player_id=new_id(), spec=spec)
            try:
                source = await anext(view.frames())
                assert source.index == 0 and source.pts_us == 0  # Local H3 CFR clock.
                assert (
                    source.provenance["cli_frame_header"]["source"]["simulationTick"]
                    == 71
                )
                assert len(source.rgb) == spec.frame_bytes
                assert abs(np.frombuffer(source.rgb, dtype=np.uint8).mean() - 80) < 3
                assert source.states["ui_events"][0]["snapshot"]["state"] == {
                    "unknown-future-state": [1, 2]
                }
                assert source.effective_input is None
                await view.send_input(
                    0,
                    {
                        "held_keys": ["KeyW"],
                        "events": [
                            {"type": "keydown", "code": "KeyW"},
                            {"type": "pointermove", "dx": 10, "dy": -2},
                        ],
                        "actions": [
                            {"id": "action-one", "name": "damage", "params": {}}
                        ],
                    },
                )
                await view.send_input(
                    1,
                    {
                        "held_keys": [],
                        "held_buttons": [],
                        "events": [],
                        "actions": [
                            {"id": "action-two", "name": "damage", "params": {}}
                        ],
                        "ui_source_frame": source.provenance["cli_frame_header"],
                    },
                )
                await view.finish()
                async with asyncio.timeout(2):
                    while len(received) < 5:
                        await asyncio.sleep(0.01)
                assert received[0]["input"]["yawDeltaRadians"] == -0.02
                assert received[0]["input"]["pitchDeltaRadians"] == -0.004
                assert received[0]["input"]["keyEdges"] == [
                    {"code": "KeyW", "kind": "down"}
                ]
                assert received[1] == {
                    "type": "input.action",
                    "epoch": 0,
                    "actionId": "action-one",
                    "name": "damage",
                    "params": {},
                }
                assert received[2]["type"] == "input.state"
                assert received[3] == {
                    "type": "input.action",
                    "epoch": 0,
                    "actionId": "action-two",
                    "name": "damage",
                    "params": {},
                }
                assert received[4]["type"] == "input.release"
                with pytest.raises(ValueError, match="not declared"):
                    await view.send_input(2, {"actions": [{"name": "unknown"}]})
                assert view.sequence == 1
                with pytest.raises(ValueError, match="one player"):
                    await room.join(player_id=new_id(), spec=spec)
                with pytest.raises(ValueError, match="unsupported"):
                    validate_descriptor(
                        {**descriptor, "baseUrl": "http://example.com"}, spec
                    )
                with pytest.raises(ValueError):
                    decode_packet(wire[:-1], descriptor, spec)
            finally:
                await room.close()

    asyncio.run(run())


def test_cli_launch_settings_keep_platform_credentials_out_of_argv():
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_arguments

    spec = VideoSpec(832, 480)
    env = {
        "WHITEBOX_CLI_PLATFORM_URL": "https://world-test-platform.loopit.com.cn",
        "WHITEBOX_CLI_BROWSER_EXECUTABLE": "/opt/chromium/chrome",
        "WHITEBOX_CLI_RENDERING": "software",
        "WORLD_TEST_API_TOKEN": "secret-value",
    }
    args = cli_arguments(
        Path("/opt/worldkit/packages/cli/bin/worldkit.mjs"), "source/world", spec, env
    )
    assert args[args.index("--wtp") + 1] == "source/world"
    assert args[args.index("--fps") + 1] == "24"
    assert args[args.index("--height") + 1] == "480"
    assert args[args.index("--platform-url") + 1] == env["WHITEBOX_CLI_PLATFORM_URL"]
    assert args[args.index("--browser-executable") + 1] == "/opt/chromium/chrome"
    assert "secret-value" not in args
    prepared_args = cli_arguments(
        Path("/opt/worldkit/packages/cli/bin/worldkit.mjs"),
        "source/world",
        spec,
        env,
        delivery="/tmp/private/archive.tar.gz",
    )
    assert "--wtp" not in prepared_args
    assert "--platform-url" not in prepared_args
    assert (
        prepared_args[prepared_args.index("--delivery") + 1]
        == "/tmp/private/archive.tar.gz"
    )
    assert "secret-value" not in prepared_args


def test_cli_resident_service_selection_is_explicit_for_room_owned_processes():
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_arguments

    entry, spec = Path("/sdk/packages/cli/bin/worldkit.mjs"), VideoSpec(832, 480)
    default = cli_arguments(entry, "s/w", spec, {})
    assert default.count("--service") == 1
    assert default[default.index("--service") + 1] == "auto"
    for delivery in (None, "/tmp/delivery.tar.gz"):
        args = cli_arguments(
            entry, "s/w", spec, {"WHITEBOX_CLI_SERVICE_MODE": "off"}, delivery=delivery
        )
        assert args.count("--service") == 1
        assert args[args.index("--service") + 1] == "off"
        assert "--service-name" not in args
    with pytest.raises(ValueError, match="Invalid CLI service mode"):
        cli_arguments(entry, "s/w", spec, {"WHITEBOX_CLI_SERVICE_MODE": "shared"})


def test_cli_service_aliases_cannot_select_two_process_owners():
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_arguments

    entry, spec = Path("/sdk/packages/cli/bin/worldkit.mjs"), VideoSpec(832, 480)
    for mode in ("off", "auto", "required"):
        args = cli_arguments(
            entry,
            "s/w",
            spec,
            {"WHITEBOX_CLI_SERVICE_MODE": mode, "WHITEBOX_CLI_SERVICE": mode},
        )
        assert args.count("--service") == 1
        assert args[args.index("--service") + 1] == mode
    with pytest.raises(ValueError, match="Conflicting CLI resident policy"):
        cli_arguments(
            entry,
            "s/w",
            spec,
            {"WHITEBOX_CLI_SERVICE_MODE": "off", "WHITEBOX_CLI_SERVICE": "auto"},
        )


def test_resident_workers_follow_room_source_capacity_without_changing_foreground():
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_config import (
        stream_policy_arguments,
    )

    for mode in ("auto", "required"):
        args = stream_policy_arguments(
            {"WHITEBOX_CLI_SERVICE_MODE": mode, "WHITEBOX_ROOM_PIPELINE_CAPACITY": "2"}
        )
        assert args[args.index("--workers") + 1] == "2"
    assert "--workers" not in stream_policy_arguments(
        {"WHITEBOX_CLI_SERVICE_MODE": "off", "WHITEBOX_ROOM_PIPELINE_CAPACITY": "2"}
    )
    for value in ("0", "17", "invalid"):
        with pytest.raises(ValueError):
            stream_policy_arguments({"WHITEBOX_ROOM_PIPELINE_CAPACITY": value})


def test_cli_rejects_prepared_delivery_identity_changes():
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import (
        validate_prepared_delivery,
    )

    identity = dict(
        archiveSha256="a" * 64,
        worldBuildHash="b" * 64,
        runtimeHash="c" * 64,
        sourceHash="d" * 64,
    )
    prepared = dict(identity=identity, uiBundleHash="e" * 64)
    descriptor = dict(
        artifact={**identity, "source": "local"},
        worldBuildHash=identity["worldBuildHash"],
        uiBundleHash="e" * 64,
    )
    validate_prepared_delivery(descriptor, prepared)
    for key in identity:
        with pytest.raises(ValueError, match="different prepared delivery"):
            validate_prepared_delivery(
                {**descriptor, "artifact": {**identity, key: "f" * 64}}, prepared
            )
    with pytest.raises(ValueError, match="different prepared delivery"):
        validate_prepared_delivery({**descriptor, "uiBundleHash": "f" * 64}, prepared)


def test_heartbeat_does_not_treat_a_leaving_player_as_owner_loss(tmp_path):
    async def run():
        directory = RoomDirectory(MemoryRoomStore(), {"pod": "http://room"})
        game = new_id()
        await directory.resolve(game, "world", create=True)
        runtime = RoomRuntime(
            directory=directory,
            owner_id="pod",
            driver_factory=shared_mock_room,
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        ready = await runtime.join(
            dict(
                game_id=game,
                world_id="world",
                player_id=new_id(),
                stream_id=new_id(),
                session_id=new_id(),
                user_id="a",
                trace_id="fixture",
                spec={"width": 32, "height": 32, "fps": 24},
            )
        )
        entry = runtime.players[ready["stream_id"]]
        closing, finish = asyncio.Event(), asyncio.Event()
        original = entry["engine"].close

        async def slow_close():
            closing.set()
            await finish.wait()
            await original()

        entry["engine"].close = slow_close
        leaving = asyncio.create_task(runtime.leave(ready["stream_id"]))
        try:
            await closing.wait()
            await asyncio.sleep(1.1)
            assert game in runtime.rooms
            assert not runtime.rooms[game]["heartbeat"].done()
            assert (await directory.resolve(game, "world"))["status"] == "active"
        finally:
            finish.set()
            await leaving
            await runtime.close()
        with pytest.raises(ValueError, match="ended"):
            await directory.resolve(game, "world")

    asyncio.run(run())


def test_disconnect_cancels_pending_room_startup(tmp_path):
    async def run():
        directory = RoomDirectory(MemoryRoomStore(), {"pod": "http://room"})
        game = new_id()
        await directory.resolve(game, "world", create=True)
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def cold_driver(**kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        runtime = RoomRuntime(
            directory=directory,
            owner_id="pod",
            driver_factory=cold_driver,
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        body = dict(
            game_id=game,
            world_id="world",
            player_id=new_id(),
            stream_id=new_id(),
            session_id=new_id(),
            user_id="test",
            trace_id="cancel-startup",
            spec={"width": 32, "height": 32, "fps": 24},
        )
        async with server(runtime_app(runtime, "test")) as address:
            async with httpx.AsyncClient() as client:
                task = asyncio.create_task(
                    client.post(
                        address + "/internal/players/join",
                        json=body,
                        headers={"Authorization": "Bearer test"},
                        timeout=10,
                    )
                )
                await asyncio.wait_for(entered.wait(), 2)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await asyncio.wait_for(cancelled.wait(), 2)
            async with asyncio.timeout(2):
                while True:
                    try:
                        await directory.resolve(game, "world")
                    except ValueError as error:
                        assert "ended" in str(error)
                        break
                    await asyncio.sleep(0.01)
            assert not runtime.rooms
            assert not runtime.players
            with pytest.raises(ValueError, match="ended"):
                await directory.resolve(game, "world")

    asyncio.run(run())


def test_browser_heartbeat_keeps_room_alive_during_model_startup(tmp_path):
    """A stalled H3 produces no media RPCs; only browser liveness preserves CLI."""
    from dataclasses import replace

    from test_whitebox import setup_pipeline

    async def run():
        directory = RoomDirectory(MemoryRoomStore(), {"pod": "http://room"})
        runtime = RoomRuntime(
            directory=directory,
            owner_id="pod",
            driver_factory=shared_mock_room,
            spool=tmp_path / "room",
            spec=VideoSpec(32, 32),
            idle_seconds=0.5,
        )
        upstream, h3, _, archives = setup_pipeline(
            tmp_path / "gateway", startup_seconds=5
        )
        upstream.session_id = new_id()
        model_entered, release_model = asyncio.Event(), asyncio.Event()
        original_send = h3.send

        async def stalled_model(wire):
            if msgspec.msgpack.decode(wire)["type"] == "init":
                model_entered.set()
                await release_model.wait()
            await original_send(wire)

        h3.send = stalled_model
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=runtime_app(runtime, "fixture")),
            base_url="http://room",
        ) as client:
            factory = RoomEngineFactory(
                token="fixture", client=client, directory=directory
            )
            upstream.config = replace(upstream.config, engine_factory=factory)
            try:
                await upstream.send(
                    msgspec.msgpack.encode(
                        dict(
                            type="init",
                            whitebox={"world_id": "world"},
                            prompt="preserve geometry",
                        )
                    )
                )
                await asyncio.wait_for(model_entered.wait(), 2)
                stream = next(iter(runtime.players))
                game = runtime.players[stream]["game_id"]
                baseline = runtime.rooms[game]["log"].sequence
                for _ in range(12):
                    await upstream.send(
                        msgspec.msgpack.encode(
                            dict(type="event", kind="heartbeat", payload={})
                        )
                    )
                    await asyncio.sleep(0.1)
                assert stream in runtime.players
                assert not upstream.ready.is_set()
                assert runtime.rooms[game]["log"].sequence == baseline
                # Expiry is observed by the one-second owner sweep. Its phase
                # relative to the last heartbeat is not fixed by the test.
                async with asyncio.timeout(3):
                    while stream in runtime.players:
                        await asyncio.sleep(0.01)
            finally:
                await upstream.close()
                await archives.close()
                await runtime.close()

    asyncio.run(run())


def test_room_journal_process_preserves_order_and_freezes_mutable_events(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomLog

    async def run():
        game = dict(game_id=new_id(), world_id="fixture-world")
        log = RoomLog(tmp_path, game, "fixture-prefix", process=True)
        await log.initialize()
        payload = {"keys": ["KeyW"]}
        try:
            for index in range(140):
                payload["sequence"] = index
                event = await log.event({"user_id": "a"}, "input", payload)
                assert event["room_seq"] == index + 1
            payload["keys"].clear()
            with pytest.raises(ValueError, match="too large"):
                await log.event({"user_id": "a"}, "input", {"data": "x" * 65536})
        finally:
            await log.finish("completed")
        rows = [
            json.loads(line)
            for line in (log.directory / "inputs.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 140
        assert [row["room_seq"] for row in rows] == list(range(1, 141))
        assert [row["payload"]["sequence"] for row in rows] == list(range(140))
        assert all(row["payload"]["keys"] == ["KeyW"] for row in rows)
        assert log.manifest["journal_worker"] == {"process": True, "pending": 0}
        assert log.manifest["objects"]["inputs.jsonl"]["bytes"] > 0

    asyncio.run(run())


@pytest.mark.parametrize("failure", [503, "transport"])
def test_room_touch_retries_transient_failure_and_records_errors(failure):
    from sglang.multimodal_gen.runtime.realtime.whitebox.metrics import capture_stages

    async def run():
        calls = []

        async def handler(request):
            calls.append(request)
            assert request.extensions["timeout"]["read"] == 2.0
            if len(calls) == 1:
                if failure == "transport":
                    raise httpx.ReadTimeout("fixture", request=request)
                return httpx.Response(failure)
            return httpx.Response(200, content=msgspec.msgpack.encode(None))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            engine = RemoteEngine(
                client,
                "http://room",
                "fixture",
                dict(engine={}, ui={}, trace_id="test", stream_id="a" * 32),
                owned_client=False,
            )
            with capture_stages() as spans:
                await engine.touch()
            assert len(calls) == 2
            assert [s["result"] for s in spans] == ["error", "success"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "operation,status,expected",
    [
        ("touch", 503, 3),
        ("touch", 409, 1),
        ("touch", 403, 1),
        ("frame", 503, 1),
        ("event", 503, 1),
    ],
)
def test_room_rpc_never_replays_non_idempotent_work_or_ownership_errors(
    operation, status, expected
):
    async def run():
        calls = []

        async def handler(request):
            calls.append(request)
            return httpx.Response(status)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            engine = RemoteEngine(
                client,
                "http://room",
                "fixture",
                dict(engine={}, ui={}, trace_id="test", stream_id="a" * 32),
                owned_client=False,
            )
            with pytest.raises(httpx.HTTPStatusError):
                if operation == "touch":
                    await engine.touch()
                else:
                    await engine.rpc(operation)
            assert len(calls) == expected

    asyncio.run(run())


def test_cli_capture_clock_is_independent_of_model_clock():
    from pathlib import Path

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import (
        capture_fps,
        cli_arguments,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec

    spec = VideoSpec()
    env = {"WHITEBOX_CLI_FPS": "60", "WHITEBOX_CLI_RESOURCE_TRANSPORT": "cdn"}
    args = cli_arguments(Path("/sdk/packages/cli/dist/main.js"), "world", spec, env)
    assert args[args.index("--fps") + 1] == "60"
    assert args[args.index("--resource-transport") + 1] == "cdn"
    assert spec.fps == 24
    with pytest.raises(ValueError, match="FPS"):
        capture_fps(spec, {"WHITEBOX_CLI_FPS": "61"})


def test_cli_mouse_buttons_preserve_edges_hold_renewal_and_release():
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_input

    down = cli_input(
        0,
        {
            "held_keys": ["KeyW"],
            "held_buttons": [0, 2],
            "events": [
                {"type": "pointerdown", "button": 0},
                {"type": "pointerdown", "button": 2},
                {"type": "pointermove", "dx": 10, "dy": -2},
            ],
        },
    )
    assert down["heldKeys"] == ["KeyW", "MouseLeft", "MouseRight"]
    assert down["keyEdges"] == [
        {"code": "MouseLeft", "kind": "down"},
        {"code": "MouseRight", "kind": "down"},
    ]
    assert down["yawDeltaRadians"] == -0.02
    assert down["pitchDeltaRadians"] == -0.004
    renewal = cli_input(
        1, {"held_keys": ["KeyW"], "held_buttons": [0, 2], "events": []}
    )
    assert renewal["heldKeys"] == down["heldKeys"]
    assert renewal["keyEdges"] == []
    released = cli_input(
        2,
        {
            "held_keys": [],
            "held_buttons": [],
            "events": [
                {"type": "pointerup", "button": 0},
                {"type": "pointerup", "button": 2},
            ],
        },
    )
    assert released["heldKeys"] == []
    assert released["keyEdges"] == [
        {"code": "MouseLeft", "kind": "up"},
        {"code": "MouseRight", "kind": "up"},
    ]
    assert cli_input(3, {"held_buttons": [1]})["heldKeys"] == []
    for invalid in [None, "0", [False], [3], [0] * 65]:
        with pytest.raises(ValueError, match="buttons"):
            cli_input(4, {"held_buttons": invalid})
