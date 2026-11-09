# SPDX-License-Identifier: Apache-2.0
"""SDK integration: native CLI playback and required shared client envelopes; no cloud writes."""

import asyncio
import json
import os
import socket
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import av
import httpx
import msgspec
import pytest
import uvicorn
from test_realtime_coordinator import _heartbeat
from test_whitebox import H3
from test_whitebox_interactive import Worker
from websockets.asyncio.client import connect

from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_room
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.room_client import (
    RoomEngineFactory,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import (
    RoomRuntime,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import (
    create_app as runtime_app,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig


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


@pytest.mark.skipif(
    not os.environ.get("WHITEBOX_TEST_CLI_SDK"),
    reason="Set WHITEBOX_TEST_CLI_SDK to a tested SDK checkout with Chromium",
)
@pytest.mark.parametrize("room_owned", [False, True])
def test_real_cli_two_games(tmp_path, monkeypatch, room_owned):
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
    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncArchiveManager,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import ModelLeases
    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import ArchiveManager
    from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        RoomPipelines,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_proxy import RoomProxy
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import HTTPRoomDirectory

    async def run():
        spec, token = VideoSpec(832, 480), "local-test-only"
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
        # The Room production path budgets 256 archive operations. This fixture
        # uses a local remux sidecar instead of OSS, but must not inherit the
        # legacy 24-operation queue while two native CLI worlds initialize.
        # Overflow/fail-open behavior has its own bounded-queue unit tests.
        room_archives = (AsyncArchiveManager if room_owned else ArchiveManager)(
            Store(), **({"queue_depth": 256} if room_owned else {})
        )
        runtime = RoomRuntime(
            directory=directory,
            owner_id="pod-a",
            driver_factory=cli_room,
            spool=tmp_path / "rooms",
            spec=spec,
            archives=room_archives,
        )

        async def worker_connect(url, **kwargs):
            from urllib.parse import parse_qs, urlsplit

            query = parse_qs(urlsplit(url).query)
            assert query["session_id"][0] in reservations.values()
            h3 = (Worker if room_owned else H3)(spec)
            h3_connections.append((query, h3))
            return h3

        async with AsyncExitStack() as stack:
            sdk = Path(os.environ["WHITEBOX_TEST_CLI_SDK"]).resolve()
            fixture_log = tmp_path / "fixture.log"
            fixture_stderr = stack.enter_context(fixture_log.open("wb"))
            fixture = await asyncio.create_subprocess_exec(
                "node",
                "--import",
                "tsx",
                str(Path(__file__).with_name("whitebox_wtp_fixture.mjs")),
                cwd=sdk,
                stdout=asyncio.subprocess.PIPE,
                stderr=fixture_stderr,
            )

            async def stop_fixture():
                if fixture.returncode is None:
                    fixture.terminate()
                    try:
                        await asyncio.wait_for(fixture.wait(), 5)
                    except TimeoutError:
                        fixture.kill()
                        await fixture.wait()

            stack.push_async_callback(stop_fixture)
            line = await asyncio.wait_for(fixture.stdout.readline(), 120)
            assert line, fixture_log.read_text()[-4000:]
            url = json.loads(line)["url"]
            monkeypatch.setenv(
                "WHITEBOX_CLI_ENTRYPOINT", str(sdk / "packages/cli/bin/worldkit.mjs")
            )
            monkeypatch.setenv("WHITEBOX_CLI_PLATFORM_URL", url)
            monkeypatch.setenv("WORLD_TEST_API_TOKEN", "local-fixture-token")
            monkeypatch.setenv("WHITEBOX_CLI_RENDERING", "software")
            monkeypatch.setenv("WHITEBOX_CLI_SERVICE", "off")
            monkeypatch.setenv("WHITEBOX_CLI_RESOURCE_TRANSPORT", "archive")

            @asynccontextmanager
            async def heartbeat():
                async def loop():
                    while True:
                        await coordinator.heartbeat(
                            _heartbeat(
                                "h3",
                                "denoiser",
                                capacity=2,
                                model_revision="h3",
                                vae_fingerprint="local",
                            )
                        )
                        await asyncio.sleep(1)

                task = asyncio.create_task(loop())
                try:
                    yield
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

            await stack.enter_async_context(heartbeat())
            room_app = runtime_app(runtime, token)
            # The owner uses the same Coordinator API; only its location changes.
            coordinator_url = await stack.enter_async_context(
                server(
                    coordinator_app(
                        coordinator, room_directory=directory, room_token=token
                    )
                )
            )
            if room_owned:
                config = WhiteboxConfig(
                    engine_factory=None,
                    ui_factory=None,
                    spool=tmp_path / "rooms",
                    spec=spec,
                    startup_seconds=90,
                    play_seconds=20,
                    drain_seconds=60,
                    room_owned=True,
                    playback_gop=6,
                )
                room_coordinator = HTTPCoordinatorClient(coordinator_url)
                stack.push_async_callback(room_coordinator.close)
                host = RoomPipelines(
                    runtime,
                    config,
                    room_archives,
                    Reports(runtime.spool, room_archives),
                    lambda user, trace: ModelLeases(
                        room_coordinator,
                        worker_connect,
                        user_id=user,
                        trace_id=trace,
                        model_revision="h3",
                        vae_fingerprint="local",
                        max_bytes=config.max_message_bytes,
                        renew_seconds=0.2,
                    ),
                    max_sessions=2,
                )
                host.install(room_app, token)
                host.reports.install(room_app)
                stack.push_async_callback(host.close)
            runtime_url = await stack.enter_async_context(server(room_app))
            directory.owners["pod-a"] = runtime_url
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
                    startup_seconds=90,
                    play_seconds=20,
                    drain_seconds=60,
                )
                client = HTTPCoordinatorClient(coordinator_url)
                proxy = None
                if room_owned:
                    http = await stack.enter_async_context(httpx.AsyncClient())
                    proxy = RoomProxy(
                        HTTPRoomDirectory(http, coordinator_url, token),
                        {"pod-a": runtime_url},
                        token,
                        http,
                        connect,
                    )
                app = gateway_app(
                    client,
                    model_revision="h3",
                    vae_fingerprint="local",
                    internal_output_url="ws://unused",
                    connect_factory=worker_connect,
                    whitebox_config=None if room_owned else config,
                    whitebox_archives=None if room_owned else manager,
                    whitebox_proxy=proxy,
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
                        + ("&whitebox_control=v1" if room_owned else "")
                    )
                )
                await ws.send(
                    msgspec.msgpack.encode(
                        dict(
                            type="init",
                            whitebox={
                                "world_id": "s/w",
                                **({"game_id": game_id} if game_id else {}),
                            },
                            prompt=user,
                        )
                    )
                )
                while True:
                    ready = msgspec.msgpack.decode(
                        await asyncio.wait_for(ws.recv(), 90)
                    )
                    if ready["type"] in {"session.ready", "error"}:
                        break
                if room_owned and ready["type"] == "session.ready":
                    while True:
                        state = msgspec.msgpack.decode(
                            await asyncio.wait_for(ws.recv(), 90)
                        )
                        assert state["type"] != "error", state
                        if state.get("state") == "active":
                            break
                    for stream in ("whitebox", "model"):
                        await ws.send(
                            msgspec.msgpack.encode(
                                dict(
                                    type="event",
                                    kind="playback_ack",
                                    payload=dict(
                                        protocol="whitebox-dual-v2",
                                        stream=stream,
                                        frame=-1,
                                        window=4,
                                    ),
                                )
                            )
                        )
                return ws, ready

            (first, ready_a), (second, ready_b) = await asyncio.gather(
                join(0, "a"), join(1, "b")
            )
            assert ready_a["type"] == "session.ready", ready_a
            game_id = ready_a["multiplayer"]["game_id"]
            assert ready_b["multiplayer"]["game_id"] != game_id
            assert ready_a["session_id"] != ready_b["session_id"]
            assert len(runtime.rooms) == 2 and len(runtime.players) == 2
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
            if not room_owned:
                await first.send(
                    msgspec.msgpack.encode(
                        {"type": "event", "kind": "prompt", "payload": "changed-a"}
                    )
                )

            async def drain(ws):
                frames = []
                while True:
                    message = msgspec.msgpack.decode(
                        await asyncio.wait_for(ws.recv(), 90)
                    )
                    if message["type"] == "session.completed":
                        if not room_owned:
                            assert message["frames"] == len(frames)
                        # Match the real client: close only after draining completed.
                        await ws.close()
                        return frames
                    assert message["type"] in {
                        "frame_batch",
                        "whitebox.frame",
                        "whitebox.input_received",
                        "whitebox.input_committed",
                        "whitebox.source_state",
                        "whitebox.states",
                        "whitebox.control",
                        "whitebox.capture",
                        "render.status",
                        "playback.status",
                    }, message
                    if room_owned and message["type"] in {
                        "frame_batch",
                        "whitebox.frame",
                    }:
                        await ws.send(
                            msgspec.msgpack.encode(
                                dict(
                                    type="event",
                                    kind="playback_ack",
                                    payload=dict(
                                        protocol="whitebox-dual-v2",
                                        stream=message["stream"],
                                        frame=message["output_frame_start"],
                                        window=4,
                                    ),
                                )
                            )
                        )
                    if message["type"] == "frame_batch":
                        frames.append(message)

            # Finish A early; B keeps rendering and exits by its server deadline.
            await first.send(
                msgspec.msgpack.encode(
                    {"type": "event", "kind": "finish", "payload": {}}
                )
            )
            second_drain = asyncio.create_task(drain(second))
            frames_a = await drain(first)
            assert len(frames_a) > 0
            async with asyncio.timeout(5):
                while len(reservations) == 2 or len(runtime.rooms) == 2:
                    await asyncio.sleep(0.01)
            assert len(reservations) == 1 and len(runtime.rooms) == 1
            frames_b = await second_drain
            assert len(frames_b) > len(frames_a)
            assert all("ui_events" in message["states"] for message in frames_b)
            async with asyncio.timeout(20):
                while reservations or runtime.rooms or (room_owned and host.sessions):
                    await asyncio.sleep(0.01)
            await asyncio.gather(
                *(manager.close() for manager in archives), room_archives.close()
            )
            for i, ready in enumerate((ready_a, ready_b)):
                root = (tmp_path / ("rooms" if room_owned else f"gateway-{i}")) / ready[
                    "recording_id"
                ]
                manifest = json.loads((root / "manifest.json").read_text())
                assert manifest["recording_status"] == "complete", {
                    key: manifest.get(key)
                    for key in (
                        "archive_failure",
                        "last_successful_frame",
                        "missing_frame_range",
                    )
                }
                assert manifest["upload_status"] == "complete"
                assert manifest["ui"]["mode"] == "browser"
                assert manifest["engine"]["wtp_delivery"]["reference"]["worldId"] == "w"
                assert manifest["engine"]["artifact"]["source"] == "local"
                assert (
                    manifest["engine"]["wtp_delivery"]["identity"]["archiveSha256"]
                    == manifest["engine"]["artifact"]["archiveSha256"]
                )
                assert (
                    manifest["multiplayer"]["game_id"]
                    == ready["multiplayer"]["game_id"]
                )
                assert "/streams/" + ready["recording_id"] in manifest["object_prefix"]
                pts = []
                for name in ("whitebox", "model"):
                    with av.open(str(root / (name + ".mp4"))) as video:
                        assert video.streams.video[0].codec_context.name == "h264"
                        pts.append([f.pts * f.time_base for f in video.decode(video=0)])
                assert pts[0] == pts[1]
                assert all(a < b for a, b in zip(pts[0], pts[0][1:]))
                async with httpx.AsyncClient() as http:
                    report = await http.get(
                        gateways[i]
                        + "/v1/whitebox/recordings/"
                        + ready["recording_id"],
                        headers={"Authorization": "Bearer " + ready["report_token"]},
                    )
                    assert report.status_code == 200
                    body = report.json()
                    assert body["manifest_url"] and len(body["videos"]) == 2
                    assert (
                        body["multiplayer"]["game_id"]
                        == ready["multiplayer"]["game_id"]
                    )
                    assert not any(
                        "ui_render" in row["stage"] for row in body["timings"]
                    )
                    inputs = await http.get(
                        gateways[i]
                        + "/v1/whitebox/recordings/"
                        + ready["recording_id"]
                        + "/inputs",
                        headers={"Authorization": "Bearer " + ready["report_token"]},
                    )
                    assert inputs.status_code == 200 and inputs.json()["intervals"]
                    path = next(iter(manifest["ui"]["files"]))
                    asset = await http.get(
                        gateways[i]
                        + "/v1/whitebox/recordings/"
                        + ready["recording_id"]
                        + "/ui/"
                        + path,
                        headers={"Authorization": "Bearer " + ready["report_token"]},
                    )
                    assert asset.status_code == 200
            room_manifest = next(
                p
                for p in (tmp_path / "rooms").glob("*/manifest.json")
                if json.loads(p.read_text()).get("archive_kind") == "game"
            )
            room_data = json.loads(room_manifest.read_text())
            assert len(room_data["players"]) == 1
            events = [
                json.loads(line)
                for line in (room_manifest.parent / "inputs.jsonl")
                .read_text()
                .splitlines()
            ]
            assert [event["room_seq"] for event in events] == list(
                range(1, len(events) + 1)
            )
            assert len({e["player_id"] for e in events if e["kind"] == "input"}) == 1
            assert any(e["kind"] == "input" for e in events)
            assert room_data["object_prefix"] + "manifest.json" in uploaded
            with pytest.raises(ValueError, match="ended"):
                await directory.resolve(game_id, "s/w")

    asyncio.run(run())


def test_actual_sdk_member_poll_and_display_survive_original_rtc_owner(
    tmp_path, monkeypatch
):
    import os

    import msgspec
    from test_whitebox_rtc_pipeline import Peer

    sdk = os.environ.get("WHITEBOX_TEST_CLI_SDK")
    if (
        not sdk
        or not (Path(sdk) / "packages/stream-player/src/whitebox-client.mjs").is_file()
    ):
        pytest.fail(
            "WHITEBOX_TEST_CLI_SDK must provide the shared WhiteboxClient for actual command envelopes"
        )

    from test_whitebox_player_media import model_owners

    async def run():
        env = await model_owners(tmp_path, monkeypatch)
        tasks = []
        try:
            record, _ = await env.admit("alice")
            receipt = await env.runtime.player_transport.get_player_media(record)
            pipeline = record["media_pipeline"]
            peer = Peer()
            _, sub = await env.host.attach(
                dict(
                    user_id="alice",
                    trace_id="actual-sdk-poll",
                    message=receipt["attachment"]["message"],
                ),
                transport="webrtc-rtp",
                rtc_peer=peer,
            )
            pipeline.session.startup.rtc_ready.set()
            replies = []

            async def receive():
                while not sub.closed:
                    replies.append(msgspec.msgpack.decode(await sub.recv()))

            async def deliver():
                while not sub.closed:
                    for i, (_, done) in enumerate(peer.calls):
                        if not done.done():
                            peer.finish(i)
                    await asyncio.sleep(0.01)

            tasks = [asyncio.create_task(receive()), asyncio.create_task(deliver())]

            async def until(predicate):
                async with asyncio.timeout(10):
                    while not predicate():
                        await asyncio.sleep(0.01)

            await until(
                lambda: (
                    pipeline.session.current_segment is not None
                    and pipeline.session.current_segment.first_decodable
                )
            )
            original_source = pipeline.session.source_epoch
            original_segment = pipeline.session.current_segment
            original_engine = env.runtime.player(record["stream_id"])["engine"]
            original_models = len(env.workers)
            # The original shared client polls 2s after ready. Keep both RTC
            # media and control consumers running beyond that first interval.
            await asyncio.sleep(2.05)
            for mode in ("whitebox", "model"):
                state = await pipeline.session.journal.read()
                process = await asyncio.create_subprocess_exec(
                    "node",
                    str(
                        Path(__file__).parent
                        / "fixtures/capture-member-media-commands.mjs"
                    ),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(
                        json.dumps(
                            dict(
                                ticket=receipt["attachment"]["message"],
                                revision=state["display_revision"],
                                mode=mode,
                                prefix=mode,
                            )
                        ).encode()
                    ),
                    10,
                )
                assert process.returncode == 0, stderr.decode()
                poll, display, query = json.loads(stdout)
                assert poll["type"] == display["type"] == query["type"] == "command"
                assert poll["version"] == display["version"] == query["version"] == 2
                pipeline.accept(sub, msgspec.msgpack.encode(poll))
                await until(
                    lambda: any(
                        m.get("type") == "source.status"
                        and m.get("request_id") == poll["request_id"]
                        for m in replies
                    )
                )
                pipeline.accept(sub, msgspec.msgpack.encode(display))
                await until(
                    lambda: any(
                        m.get("type") == "operation.result"
                        and m.get("request_id") == display["request_id"]
                        for m in replies
                    )
                )
                current = await pipeline.session.journal.read()
                assert (
                    current["display_mode"] == mode
                    and current["display_revision"] == state["display_revision"] + 1
                )
                # Reconnect/uncertain writes query their original operation;
                # observing it must not grant another input or model owner.
                pipeline.accept(sub, msgspec.msgpack.encode(query))
                await until(
                    lambda: any(
                        m.get("type") == "source.status"
                        and m.get("request_id") == query["request_id"]
                        for m in replies
                    )
                )
                observed = next(
                    m for m in replies if m.get("request_id") == query["request_id"]
                )
                assert observed["source_epoch"] == original_source
                assert observed["source_session_id"] == pipeline.session_id
                assert observed["connection_generation"] == sub.generation
                assert observed["operation_request_id"] == display["request_id"]
                assert observed["operation"]["request_id"] == display["request_id"]
                assert observed["operation"]["op"] == "display.configure"
                assert await pipeline.session.journal.read() == current
            state = await pipeline.session.journal.read()
            for change, code in (
                (
                    {"request_id": "stale-display", "expected_display_revision": 0},
                    "REVISION_CONFLICT",
                ),
                (
                    {"request_id": "foreign-source", "source_epoch": "foreign"},
                    "SOURCE_EPOCH_MISMATCH",
                ),
            ):
                invalid = {**display, **change}
                pipeline.accept(sub, msgspec.msgpack.encode(invalid))
                await until(
                    lambda: any(
                        m.get("type") == "operation.rejected"
                        and m.get("request_id") == invalid["request_id"]
                        and m.get("code") == code
                        for m in replies
                    )
                )
            after = await pipeline.session.journal.read()
            assert after["display_revision"] == state["display_revision"]
            assert (
                not sub.closed and not pipeline.closed and pipeline.subscription is sub
            )
            assert pipeline.session.source_epoch == original_source
            assert (
                pipeline.session.current_segment is original_segment
                and len(env.workers) == original_models
            )
            assert (
                env.runtime.player(record["stream_id"])["engine"] is original_engine
                and len(env.views) == 1
            )
            assert not original_engine.inputs
            query = {
                **poll,
                "op": "operation.get",
                "request_id": "original-display-result",
                "operation_request_id": display["request_id"],
            }
            pipeline.accept(sub, msgspec.msgpack.encode(query))
            await until(
                lambda: any(
                    message.get("type") == "source.status"
                    and message.get("request_id") == query["request_id"]
                    and (message.get("operation") or {}).get("request_id")
                    == display["request_id"]
                    for message in replies
                )
            )
            assert (await pipeline.session.journal.read())["display_revision"] == state[
                "display_revision"
            ]
            for op in (
                "render.start",
                "render.replace",
                "render.end",
                "prompt.replace",
                "source.close",
                "source.unbind",
                "area.presented",
                "unknown",
            ):
                with pytest.raises(ValueError, match="owns member input"):
                    pipeline.accept(
                        sub,
                        msgspec.msgpack.encode(
                            {**poll, "op": op, "request_id": "forbidden-" + op}
                        ),
                    )
            for message in (
                dict(
                    type="event",
                    kind="input",
                    sequence=0,
                    payload=dict(held_keys=["KeyW"]),
                ),
                dict(type="asset.upload"),
                dict(type="init"),
                dict(type="event", kind="prompt", payload="replace"),
                dict(type="event", kind="finish"),
                dict(type="event", kind="mock.frame"),
            ):
                with pytest.raises(ValueError, match="owns member input"):
                    pipeline.accept(sub, msgspec.msgpack.encode(message))
            assert not original_engine.inputs and len(env.workers) == original_models
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await env.close()

    asyncio.run(run())
