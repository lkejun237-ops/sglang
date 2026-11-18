# SPDX-License-Identifier: Apache-2.0
"""Same-admission media receipts; CPU owners and fixed public SDK wire only."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from whitebox_player_log_test_utils import player_log_capture as player_log_capture

from sglang.multimodal_gen.runtime.realtime.whitebox.player_transport import (
    PlayerTransport,
    PlayerTransportConfig,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.room_control import (
    validate_request,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
    RoomControlRuntime,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/player-media-native-v1.json").read_text()
)


async def owners():
    directory = RoomDirectory(MemoryRoomStore(), {"owner": "http://room"})
    game, epoch, stream = "1" * 32, "2" * 32, "3" * 32
    await directory.resolve(game, "s/w", create=True)
    await directory.touch(game, "owner", epoch)
    caller = dict(serviceId="wtp", subject="member-a")
    binding = dict(gameId=game, ownerEpoch=epoch, publicationId="p")
    record = dict(
        binding=binding,
        admission=dict(
            operationId="admit-a", requestId="request-a", parametersSha256="a" * 64
        ),
        caller=caller,
        member_id="alice",
        stream_id=stream,
        status="claimed",
    )
    private = copy.deepcopy(FIXTURE["response"]["session"])
    private["baseUrl"] = "http://127.0.0.1:54321"
    entry = dict(
        identity=dict(player_id="4" * 32),
        engine=SimpleNamespace(descriptor=private),
        game_id=game,
        control_admission=record,
    )
    published = copy.deepcopy(FIXTURE["response"]["session"])

    async def publish(player, public_origin):
        assert player == "4" * 32 and public_origin == "https://media.test"
        return copy.deepcopy(published)

    room = dict(
        driver=SimpleNamespace(public_session=publish),
        control_record=dict(
            invocation=dict(createContext=copy.deepcopy(FIXTURE["createContext"])),
            expires_at_ms=4_102_444_800_000,
        ),
    )
    entries = {stream: entry}

    def player(sid, **kwargs):
        if sid not in entries:
            raise ValueError("The player has left the game")
        return entries[sid]

    runtime = SimpleNamespace(
        rooms={game: room},
        directory=directory,
        player=player,
        owner_id="owner",
        epoch=epoch,
    )
    runtime.player_transport = PlayerTransport(
        runtime,
        PlayerTransportConfig(
            "https://media.test", "https://player.test", "https://control.test"
        ),
    )
    control = RoomControlRuntime(runtime)
    control.admissions[control.key(caller, "admit-a")] = record
    request = dict(
        schemaVersion=1,
        action="getPlayerMedia",
        memberBinding={
            **binding,
            "memberId": "alice",
            "admissionOperationId": "admit-a",
        },
    )
    return SimpleNamespace(
        control=control,
        runtime=runtime,
        request=request,
        caller=caller,
        route=await directory.lookup(game),
        entry=entry,
        record=record,
        published=published,
        entries=entries,
    )


def test_native_only_never_reports_model_ready():
    async def run():
        owner = await owners()
        assert validate_request(owner.request) == owner.request
        result = await owner.control.handle(owner.caller, owner.request, owner.route)
        assert result["kind"] == "sdk-session" and result["status"] == "ready"
        assert result["session"]["media"]["kind"] == "pixels"
        assert result["memberBinding"] == owner.request["memberBinding"]
        assert result == FIXTURE["response"]
        assert result["session"]["worldBuildHash"] == "a" * 64
        assert result["session"]["uiBundleHash"] == "b" * 64
        assert result["model"] == {
            "status": "unavailable",
            "reason": "MEMBER_MODEL_SOURCE_UNAVAILABLE",
        }
        assert "attachment" not in result
        # The existing consumer retains its exact response shape.
        legacy = await owner.control.handle(
            owner.caller, {**owner.request, "action": "getPlayerSession"}, owner.route
        )
        assert legacy == {"status": "ready", "session": result["session"]}

    asyncio.run(run())


@pytest.mark.parametrize(
    "changed", ["caller", "member", "admission", "leaving", "session", "epoch", "ui"]
)
def test_cross_member_or_replaced_session_is_rejected(changed):
    async def run():
        owner = await owners()
        if changed == "caller":
            owner.caller = dict(owner.caller, subject="member-b")
        elif changed == "member":
            owner.request["memberBinding"]["memberId"] = "bob"
        elif changed == "admission":
            owner.request["memberBinding"]["admissionOperationId"] = "admit-b"
        elif changed == "leaving":
            owner.record["leaving"] = True
        elif changed == "session":
            owner.published["sessionId"] = "render-bob"
        elif changed == "epoch":
            owner.published["epoch"] = 4
        else:
            owner.published["uiBundleHash"] = "c" * 64
        with pytest.raises(ValueError):
            await owner.control.handle(owner.caller, owner.request, owner.route)

    asyncio.run(run())


def test_source_envelope_retains_sdk_frame_identity_without_h264_counter_conversion():
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.native_pixels import (
        decode_native_packet,
    )

    envelope = FIXTURE["sourceEnvelope"]
    encoded = json.dumps(envelope).encode()
    wire = len(encoded).to_bytes(4, "big") + encoded + bytes(envelope["payloadBytes"])
    frame, pixels = decode_native_packet(
        wire, FIXTURE["response"]["session"], VideoSpec(160, 90)
    )
    assert frame == envelope["input"]
    assert frame["epoch"] == 3 and frame["source"]["sdkEpoch"] == 11
    assert frame["outputFrameId"] == 19 and frame["source"]["sourceFrameId"] == 47
    assert len(pixels) == 43200
    for changes in (
        {"sessionId": "render-bob"},
        {"epoch": 4},
        {"uiBundleHash": "f" * 64},
    ):
        with pytest.raises(ValueError):
            decode_native_packet(
                wire, {**FIXTURE["response"]["session"], **changes}, VideoSpec(160, 90)
            )


def test_native_receipt_rechecks_retirement_after_publication_await():
    async def run():
        owner = await owners()
        room = owner.runtime.rooms[owner.record["binding"]["gameId"]]
        original = room["driver"].public_session

        async def publish(*args):
            value = await original(*args)
            room["control_record"]["retiring"] = True
            return value

        room["driver"].public_session = publish
        with pytest.raises(ValueError, match="publication"):
            await owner.control.handle(owner.caller, owner.request, owner.route)

    asyncio.run(run())


def test_media_receipt_cannot_relabel_another_admissions_native_session():
    async def run():
        owner = await owners()
        # Both upstream descriptors agree with each other, but no longer with
        # this admission. Generic public-session equality alone cannot catch it.
        owner.entry["engine"].descriptor["playerBinding"]["playerSessionId"] = "bob"
        owner.published["playerBinding"]["playerSessionId"] = "bob"
        with pytest.raises(ValueError, match="binding"):
            await owner.control.handle(owner.caller, owner.request, owner.route)

    asyncio.run(run())


def test_signed_media_read_uses_original_gateway_directory_and_owner_hop():
    import httpx
    from fastapi import FastAPI
    from test_whitebox_room_control import SECRET, ticket

    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control import (
        RoomControlCapabilities,
        install_gateway_room_control,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
        install_runtime_room_control,
    )

    async def run():
        owner = await owners()
        app = FastAPI()
        install_runtime_room_control(app, "internal", owner.runtime)
        owner.runtime.control = owner.control
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://room"
        ) as upstream:
            gateway = FastAPI()
            proxy = SimpleNamespace(
                directory=owner.runtime.directory,
                owners={"owner": "http://room"},
                token="internal",
                client=upstream,
            )
            install_gateway_room_control(
                gateway,
                proxy,
                verifier=RoomControlCapabilities(SECRET, {"wtp"}, now=lambda: 100),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(gateway), base_url="http://gateway"
            ) as client:
                raw = json.dumps(owner.request).encode()
                auth = "Bearer " + ticket(raw, sub=owner.caller["subject"])
                result = await client.post(
                    "/v1/whitebox/room-control",
                    content=raw,
                    headers={"authorization": auth},
                )
                assert result.status_code == 200
                assert result.json() == FIXTURE["response"]
                assert result.headers["cache-control"] == "no-store"
                assert (
                    await client.post(
                        "/v1/whitebox/room-control",
                        content=raw,
                        headers={"authorization": auth},
                    )
                ).status_code == 403
                wrong = "Bearer " + ticket(
                    raw,
                    sub=owner.caller["subject"],
                    jti="2" * 32,
                    aud="whitebox-single-player-control",
                )
                assert (
                    await client.post(
                        "/v1/whitebox/room-control",
                        content=raw,
                        headers={"authorization": wrong},
                    )
                ).status_code == 403

    asyncio.run(run())


def media_launch():
    import base64

    import msgspec
    from test_whitebox_startup import request

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import PROTOCOL

    init = dict(
        type="init",
        control_protocol=PROTOCOL,
        source_operation="create",
        whitebox=dict(source_id="s", world_id="w"),
        startup=request(False),
    )
    return dict(
        transport="webrtc-rtp",
        publicationIdentity=copy.deepcopy(FIXTURE["response"]["publication"]),
        initMsgpackBase64=base64.b64encode(msgspec.msgpack.encode(init)).decode(),
    )


def test_original_admission_accepts_only_bounded_frozen_member_launch():
    request = dict(
        schemaVersion=1,
        action="admitPlayer",
        binding={
            key: FIXTURE["request"]["memberBinding"][key]
            for key in ("gameId", "ownerEpoch", "publicationId")
        },
        admission=dict(
            operationId="admit-a", requestId="request-a", parametersSha256="a" * 64
        ),
        mediaLaunch=media_launch(),
    )
    assert validate_request(request) == request
    for change in (
        dict(transport="websocket"),
        dict(initMsgpackBase64="invalid"),
        dict(
            publicationIdentity={
                **request["mediaLaunch"]["publicationIdentity"],
                "publicationId": "other",
            }
        ),
    ):
        with pytest.raises(ValueError):
            validate_request(
                {**request, "mediaLaunch": {**request["mediaLaunch"], **change}}
            )
    query = {**FIXTURE["request"], "controlSequence": 3}
    assert validate_request(query) == query
    with pytest.raises(ValueError):
        validate_request({**query, "controlSequence": True})


async def model_owners(
    tmp_path,
    monkeypatch,
    *,
    create_request=None,
    launch=None,
    worker_protocol=None,
    max_sources=1,
):
    from contextlib import asynccontextmanager
    from dataclasses import replace

    import msgspec
    from test_whitebox_interactive import BrowserUI, Engine
    from test_whitebox_render_lifecycle import OrderedWorker
    from test_whitebox_room_control import request_body

    from sglang.multimodal_gen.runtime.realtime.source_prepare import (
        PROTOCOL as PREPARE,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncArchiveManager,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import RenderHost
    from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig

    monkeypatch.setenv("WHITEBOX_NATIVE_RTC_ENABLED", "1")
    monkeypatch.setenv("WHITEBOX_MODEL_STARTUP_PREPARE", "1")
    spec = VideoSpec(160, 96)
    views, tasks, workers = {}, [], []

    class View(Engine):
        async def frames(self):
            async for frame in super().frames():
                header = copy.deepcopy(FIXTURE["sourceEnvelope"]["input"])
                header.update(
                    sessionId=self.descriptor["sessionId"],
                    widthPixels=spec.width,
                    heightPixels=spec.height,
                    outputFrameId=frame.index + 19,
                    outputPtsUs=frame.pts_us,
                    uiCompleteThroughUs=frame.pts_us,
                )
                header["source"].update(
                    sourceFrameId=frame.index + 47,
                    presentationId=self.descriptor["playerBinding"]["viewportId"],
                    sourceTimeUs=frame.pts_us,
                    simulationTick=frame.index + 100,
                )
                yield replace(
                    frame, provenance={**frame.provenance, "cli_frame_header": header}
                )

    class Driver:
        max_players = 2

        async def join(self, player_id, **kwargs):
            view = View(spec)
            view.renderer = BrowserUI()
            view.descriptor = copy.deepcopy(FIXTURE["response"]["session"])
            view.descriptor.update(
                sessionId="render-" + player_id, width=spec.width, height=spec.height
            )
            view.descriptor["playerBinding"].update(
                playerSessionId=player_id,
                viewportId="viewport-" + player_id,
                render=dict(sessionId=view.descriptor["sessionId"], epoch=3),
            )
            view.descriptor["audience"]["playerSessionId"] = player_id
            view.descriptor["ui"]["moduleUrl"] = (
                f"https://media.test/v1/sessions/{view.descriptor['sessionId']}/ui/{view.descriptor['uiBundleHash']}/module.js"
            )
            # A sealed SDK Room reports archive identity; its member model wire
            # must separately project the admission's frozen publication.
            view.metadata = dict(
                artifact=dict(
                    archiveSha256="7" * 64,
                    worldBuildHash=view.descriptor["worldBuildHash"],
                    runtimeHash="8" * 64,
                    sourceHash="9" * 64,
                )
            )
            views[player_id] = view

            async def produce():
                while not view.closed:
                    await view.push()
                    await asyncio.sleep(0.025)

            tasks.append(asyncio.create_task(produce()))
            return view

        async def public_session(self, player_id, origin):
            return copy.deepcopy(views[player_id].descriptor)

        async def close(self):
            for view in views.values():
                await view.close()

    async def factory(**kwargs):
        return Driver()

    class Worker(OrderedWorker):
        async def emit(self, message):
            if message["type"] == "session.ready":
                message.update(
                    source_prepare_protocol=worker_protocol or PREPARE,
                    source_prepare_profile_sha256="a" * 64,
                )
            await super().emit(message)

        async def send(self, wire):
            message = msgspec.msgpack.decode(wire)
            if message.get("kind") in {"source.prepare", "source.attach"}:
                await self.emit(
                    dict(
                        type=(
                            "source.prepared"
                            if message["kind"] == "source.prepare"
                            else "source.attached"
                        ),
                        identity=message["payload"]["identity"],
                    )
                )
            else:
                await super().send(wire)

    class Model:
        resource_state = "none"

        @asynccontextmanager
        async def open(self):
            worker = Worker(spec)
            workers.append(worker)
            self.resource_state = "reserved"
            try:
                yield (
                    worker,
                    SimpleNamespace(
                        session_id="m",
                        generation_id="g",
                        denoiser=SimpleNamespace(
                            worker_id="cpu-peer",
                            worker_epoch="w",
                            model_revision="test",
                        ),
                    ),
                )
            finally:
                await worker.close()
                self.resource_state = "released"

    directory = RoomDirectory(MemoryRoomStore(), {"owner": "http://room"})
    archives = AsyncArchiveManager(recorder_slots=5)
    runtime = RoomRuntime(
        directory=directory,
        owner_id="owner",
        driver_factory=factory,
        spool=tmp_path,
        spec=spec,
        archives=archives,
    )
    config = WhiteboxConfig(
        engine_factory=None,
        ui_factory=None,
        spool=tmp_path,
        spec=spec,
        room_owned=True,
        startup_seconds=15,
        drain_seconds=3,
    )
    host = RenderHost(
        runtime,
        config,
        archives,
        Reports(tmp_path, archives),
        None,
        None,
        max_sources=max_sources,
    )
    host.make_segment_leases = lambda *_: Model()
    control = RoomControlRuntime(runtime)
    runtime.control = control
    runtime.player_transport = PlayerTransport(
        runtime,
        PlayerTransportConfig(
            "https://media.test", "https://player.test", "https://control.test"
        ),
    )
    create = (
        copy.deepcopy(create_request) if create_request is not None else request_body()
    )
    if create_request is None:
        create["videoSpec"] = dict(width=spec.width, height=spec.height, fps=spec.fps)
        create["createContext"] = copy.deepcopy(FIXTURE["createContext"])
    caller = dict(serviceId="wtp", subject="alice")
    route = await directory.control_route(caller, create, create=True)
    await control.handle(caller, create, route)
    await control.drain()
    created = await control.handle(caller, {**create, "action": "getOperation"}, route)
    assert created["status"] == "succeeded"
    binding = dict(
        gameId=created["gameId"], ownerEpoch=runtime.epoch, publicationId="p"
    )

    async def admit(subject):
        who = dict(serviceId="wtp", subject=subject)
        body = dict(
            schemaVersion=1,
            action="admitPlayer",
            binding=binding,
            admission=dict(
                operationId="admit-" + subject,
                requestId="request-" + subject,
                parametersSha256="a" * 64,
            ),
            mediaLaunch=copy.deepcopy(launch) if launch is not None else media_launch(),
        )
        await control.handle(who, validate_request(body), route)
        await control.drain()
        record = control.admissions[control.key(who, body["admission"]["operationId"])]
        assert record["status"] == "claimed"
        return record, body

    async def close():
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await host.close()
        await runtime.close()
        await archives.close()

    return SimpleNamespace(
        runtime=runtime,
        host=host,
        control=control,
        admit=admit,
        close=close,
        views=views,
        workers=workers,
        route=route,
    )


def test_model_attachment_binds_same_member_source_and_ui(tmp_path, monkeypatch):
    import msgspec
    from test_whitebox_rtc_pipeline import Peer

    async def run():
        env = await model_owners(tmp_path, monkeypatch)
        try:
            record, body = await env.admit("alice")
            receipt = await env.runtime.player_transport.get_player_media(record)
            assert receipt["kind"] == "whitebox", record.get("media")
            assert receipt["model"]["status"] == "starting"
            pipeline = record["media_pipeline"]
            assert pipeline.stream_id == record["stream_id"]
            assert len(env.views) == 1
            peer = Peer()
            _, sub = await env.host.attach(
                dict(
                    user_id="alice",
                    trace_id="attach",
                    message=receipt["attachment"]["message"],
                ),
                transport="webrtc-rtp",
                rtc_peer=peer,
            )
            pipeline.session.startup.rtc_ready.set()
            async with asyncio.timeout(10):
                while not any(
                    message["stream"] == "model" for message, _ in peer.calls
                ):
                    for i, (_, done) in enumerate(peer.calls):
                        if not done.done():
                            peer.finish(i)
                    await asyncio.sleep(0.01)
            ready = await env.runtime.player_transport.get_player_media(record)
            assert ready["model"]["status"] == "ready"
            assert (
                pipeline.ready_message()["engine"].get("artifact")
                == body["mediaLaunch"]["publicationIdentity"]
            )
            # Publishing model metadata must not rewrite the SDK archive owner.
            assert (
                env.views[record["player_id"]].metadata["artifact"]["archiveSha256"]
                == "7" * 64
            )
            model = next(
                message for message, _ in peer.calls if message["stream"] == "model"
            )
            frame = model["source"]["cli_frame_header"]
            assert (
                frame["sessionId"] == ready["session"]["sessionId"]
                and frame["epoch"] == 3
            )
            assert frame["source"]["sdkEpoch"] == 11
            stale = copy.deepcopy(model)
            stale["source"]["source_epoch"] = "another-source"
            with pytest.raises(ValueError, match="source"):
                pipeline.member_source.frame(stale)
            assert (
                ready["attachment"]["message"]["sdkSession"]["uiBundleHash"] == "b" * 64
            )
            with pytest.raises(ValueError, match="owns member input"):
                pipeline.accept(
                    sub,
                    msgspec.msgpack.encode(
                        dict(kind="input", sequence=0, payload=dict(held_keys=["KeyW"]))
                    ),
                )
            assert not env.views[record["player_id"]].inputs
            assert "resume_token" not in pipeline.resume_capability_fields()
            # The original source reader is exclusive; direct/legacy reads fail.
            with pytest.raises(ValueError, match="already consumed"):
                await env.runtime.next_frame(record["stream_id"])
        finally:
            await env.close()

    asyncio.run(run())


def test_single_active_model_capacity_never_reports_second_model_ready(
    tmp_path, monkeypatch
):
    async def run():
        env = await model_owners(tmp_path, monkeypatch)
        try:
            first, _ = await env.admit("alice")
            second, body = await env.admit("bob")
            assert len(env.views) == 2 and len(env.host.sessions) == 1
            receipt = await env.runtime.player_transport.get_player_media(second)
            assert receipt["kind"] == "sdk-session"
            assert receipt["model"] == dict(
                status="unavailable", reason="CAPACITY_EXHAUSTED"
            )
            pipeline = first["media_pipeline"]
            await pipeline.close("TEST_MODEL_END")
            assert not env.views[first["player_id"]].closed
            ended = await env.runtime.player_transport.get_player_media(first)
            assert (
                ended["kind"] == "sdk-session"
                and ended["model"]["status"] == "unavailable"
            )
            await env.control.handle(second["caller"], body, env.route)
            await env.control.drain()
            assert not env.host.sessions
            assert (await env.runtime.player_transport.get_player_media(second))[
                "model"
            ] == receipt["model"]
            changed = copy.deepcopy(body)
            changed["mediaLaunch"]["initMsgpackBase64"] += "AA=="
            with pytest.raises(ValueError, match="changed"):
                await env.control.handle(second["caller"], changed, env.route)
            assert len(env.views) == 2
        finally:
            await env.close()

    asyncio.run(run())


def test_distinct_original_members_reserve_separate_media_sources(
    tmp_path, monkeypatch
):
    from test_whitebox_rtc_pipeline import Peer

    async def run():
        env = await model_owners(tmp_path, monkeypatch, max_sources=2)
        try:
            first, _ = await env.admit("alice")
            second, _ = await env.admit("bob")
            for record in (first, second):
                receipt = await env.runtime.player_transport.get_player_media(record)
                assert receipt["kind"] == "whitebox", record.get("media")
                assert receipt["model"]["status"] == "starting"
                assert record["media_pipeline"].member_source.record is record
            a, b = first["media_pipeline"], second["media_pipeline"]
            assert a.game_id == b.game_id
            assert a.session_id != b.session_id
            assert a.stream_id != b.stream_id
            assert (
                a.session.startup.ids["model_session_id"]
                != b.session.startup.ids["model_session_id"]
            )
            assert len(env.host.sessions) == 2 and len(env.views) == 2
            peers = []
            for record in (first, second):
                receipt = await env.runtime.player_transport.get_player_media(record)
                peer = Peer()
                await env.host.attach(
                    dict(
                        user_id=record["caller"]["subject"],
                        trace_id="attach",
                        message=receipt["attachment"]["message"],
                    ),
                    transport="webrtc-rtp",
                    rtc_peer=peer,
                )
                record["media_pipeline"].session.startup.rtc_ready.set()
                peers.append(peer)
            async with asyncio.timeout(10):
                while not all(
                    any(m["stream"] == "model" for m, _ in peer.calls) for peer in peers
                ):
                    for peer in peers:
                        for i, (_, done) in enumerate(peer.calls):
                            if not done.done():
                                peer.finish(i)
                    await asyncio.sleep(0.01)
            assert len(env.workers) == 2
            for record, peer in zip((first, second), peers):
                receipt = await env.runtime.player_transport.get_player_media(record)
                assert receipt["model"]["status"] == "ready"
                frame = next(m for m, _ in peer.calls if m["stream"] == "model")
                assert (
                    frame["source"]["cli_frame_header"]["sessionId"]
                    == receipt["session"]["sessionId"]
                )
        finally:
            await env.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "ending", ["leave", "model-end", "initialization-cancel", "sdk-reset"]
)
def test_two_model_members_keep_independent_lifecycles(tmp_path, monkeypatch, ending):
    import msgspec
    from test_whitebox_rtc_pipeline import Peer

    from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import (
        WhiteboxDiagnostic,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.player_media import (
        member_binding,
    )

    async def run():
        env = await model_owners(tmp_path, monkeypatch, max_sources=2)
        peers = []
        control_readers = []
        received_controls = []

        async def receive_controls(sub):
            # A real RTC client consumes its reliable control channel even while
            # another member's operation is awaiting cleanup. The media handoff
            # fixture below independently controls completion of RTP frames.
            try:
                while not sub.closed:
                    message = msgspec.msgpack.decode(await sub.recv())
                    received_controls.append((sub, message))
                    if (
                        message.get("type") in {"error", "session.completed"}
                        and sub.owner.terminal is not None
                    ):
                        return
            except WhiteboxDiagnostic as error:
                if not sub.closed or error.code != "RTC_DISCONNECTED":
                    raise

        async def until(predicate):
            async with asyncio.timeout(10):
                while not predicate():
                    for task in control_readers:
                        if task.done() and not task.cancelled():
                            task.result()
                    for peer in peers:
                        for i, (_, done) in enumerate(peer.calls):
                            if not done.done():
                                peer.finish(i)
                    await asyncio.sleep(0.01)

        async def attach(record):
            receipt = await env.runtime.player_transport.get_player_media(record)
            peer = Peer()
            pipeline, sub = await env.host.attach(
                dict(
                    user_id=record["caller"]["subject"],
                    trace_id="lifecycle-attach",
                    message=receipt["attachment"]["message"],
                ),
                transport="webrtc-rtp",
                rtc_peer=peer,
            )
            pipeline.session.startup.rtc_ready.set()
            peers.append(peer)
            control_readers.append(asyncio.create_task(receive_controls(sub)))
            return peer, sub

        def model_frames(peer):
            return [
                message for message, _ in peer.calls if message["stream"] == "model"
            ]

        try:
            # B is already producing model video before A can start or cancel.
            second, _ = await env.admit("bob")
            peer_b, sub_b = await attach(second)
            await until(lambda: bool(model_frames(peer_b)))
            first, _ = await env.admit("alice")
            a, b = first["media_pipeline"], second["media_pipeline"]
            old_a = await env.runtime.player_transport.get_player_media(first)
            if ending != "initialization-cancel":
                peer_a, _ = await attach(first)
                await until(lambda: bool(model_frames(peer_a)))
            else:
                await until(lambda: a.session.startup.prepared.is_set())
                assert not a.session.startup.rtc_ready.is_set()
                assert model_frames(peer_b)
                assert (await env.runtime.player_transport.get_player_media(first))[
                    "model"
                ]["status"] == "starting"
            reader_a, reader_b = a.member_source.reader, b.member_source.reader
            lease_a, lease_b = a.session.startup.lease, b.session.startup.lease
            worker_a, worker_b = a.session.startup.ws, b.session.startup.ws
            assert lease_a is not lease_b and worker_a is not worker_b
            assert lease_a.resource_state == lease_b.resource_state == "reserved"
            entry_b = env.runtime.player(second["stream_id"], touch=False)
            view_a, view_b = (
                env.views[first["player_id"]],
                env.views[second["player_id"]],
            )
            renderer_b = b.session.renderer
            segment_b = b.session.current_segment
            model_stream_b = b.session.current_model_stream
            before = await env.runtime.player_transport.get_player_media(second)
            if ending == "leave":
                request = dict(
                    schemaVersion=1,
                    action="leavePlayer",
                    memberBinding=member_binding(first),
                    operation=dict(
                        operationId="leave-alice",
                        requestId="leave-request",
                        kind="leave",
                        parametersSha256="c" * 64,
                    ),
                )
                await env.control.handle(
                    first["caller"], validate_request(request), env.route
                )
                await env.control.drain()
                result = await env.control.handle(first["caller"], request, env.route)
                assert (
                    result["status"] == "retired" and result["cleanupComplete"] is True
                )
                assert (
                    await env.control.handle(first["caller"], request, env.route)
                    == result
                )
                assert view_a.closed and first["stream_id"] not in env.runtime.players
            elif ending == "model-end":
                segment = a.session.current_segment
                await segment.stop()
                assert (
                    segment.closed_event.is_set()
                    and lease_a.resource_state == "released"
                )
                assert not view_a.closed and not reader_a.closed
                assert (
                    b.session.startup.lease is lease_b
                    and lease_b.resource_state == "reserved"
                )
                await env.host.member_media.close(first)
            elif ending == "initialization-cancel":
                await env.host.member_media.close(first)
                assert a.session.startup.closed and a.session.startup.task.done()
                assert not view_a.closed
                assert not any(
                    event.get("kind") == "source.append" for event in worker_a.events
                )
            else:
                # Invalidate A's frozen SDK identity, as an owner reset/replacement does.
                view_a.descriptor["epoch"] += 1
                await until(lambda: a.closed)
                assert not view_a.closed
            await until(lambda: a.closed and reader_a.closed)
            assert a.member_closed.is_set() and a.session_id not in env.host.sessions
            assert lease_a.resource_state == "released" and worker_a.closed
            assert "media_reader" not in a.member_source.entry
            # Retrying the original child cleanup never retires a native peer or leases again.
            await env.host.member_media.close(first)
            with pytest.raises(ValueError):
                await env.host.attach(
                    dict(
                        user_id="alice",
                        trace_id="old-attachment",
                        message=old_a["attachment"]["message"],
                    ),
                    transport="webrtc-rtp",
                    rtc_peer=Peer(),
                )
            assert env.host.sessions[b.session_id] is b and not b.closed
            assert env.runtime.player(second["stream_id"], touch=False) is entry_b
            assert entry_b["media_reader"] is reader_b and not reader_b.closed
            assert b.session.renderer is renderer_b and not view_b.closed
            assert (
                b.session.current_segment is segment_b
                and not segment_b.closed_event.is_set()
            )
            assert b.session.current_model_stream == model_stream_b
            assert (
                b.session.startup.lease is lease_b
                and lease_b.resource_state == "reserved"
            )
            assert b.session.startup.ws is worker_b and not worker_b.closed
            await env.runtime.event(
                second["stream_id"],
                "input",
                dict(sequence=0, input=dict(held_keys=["KeyD"])),
            )
            assert view_b.inputs == [dict(held_keys=["KeyD"])] and not view_a.inputs
            # View.frames maps index zero to the retained fixture's output frame ID.
            # next has already advanced past every generated/queued source frame,
            # so this barrier cannot be satisfied by cleanup-time backlog.
            output_frame_offset = FIXTURE["sourceEnvelope"]["input"]["outputFrameId"]
            after_cleanup_frame = view_b.next + output_frame_offset
            if ending == "leave":
                # An independent client's reliable receiver must stay live while
                # another member's action takes longer than one queue window.
                for _ in range(513):
                    await b.session.status_snapshot()
                    await asyncio.sleep(0)
            assert b.subscription is sub_b and not sub_b.closed, sub_b.reason
            command = dict(
                type="command",
                op="status.get",
                request_id="bob-after-" + ending,
                **b.session.source_identity(),
            )
            b.accept(sub_b, msgspec.msgpack.encode(command))
            await until(
                lambda: any(
                    row[2].get("type") == "source.status"
                    and row[2].get("request_id") == command["request_id"]
                    for row in b.history
                )
            )
            await until(
                lambda: (
                    model_frames(peer_b)[-1]["source"]["cli_frame_header"][
                        "outputFrameId"
                    ]
                    >= after_cleanup_frame
                )
            )
            await until(
                lambda: any(
                    subscription is sub_b
                    and message.get("type") == "source.status"
                    and message.get("request_id") == command["request_id"]
                    for subscription, message in received_controls
                )
            )
            status = next(
                message
                for subscription, message in received_controls
                if subscription is sub_b
                and message.get("type") == "source.status"
                and message.get("request_id") == command["request_id"]
            )
            assert status["source_session_id"] == b.session_id
            assert status["source_epoch"] == before["execution"]["sourceEpoch"]
            frame = model_frames(peer_b)[-1]["source"]["cli_frame_header"]
            assert (
                frame["sessionId"] == before["session"]["sessionId"]
                and frame["epoch"] == before["session"]["epoch"]
            )
            assert (
                frame["source"]["presentationId"]
                == before["session"]["playerBinding"]["viewportId"]
            )
            assert frame["uiCompleteThroughUs"] >= frame["outputPtsUs"]
            after = await env.runtime.player_transport.get_player_media(second)
            assert after["model"]["status"] == "ready"
            assert all(
                after[key] == before[key]
                for key in ("session", "memberBinding", "execution")
            )
            # Reconnect B on its original source; only the connection generation changes.
            reconnect_frame = view_b.next + output_frame_offset
            replacement_peer, replacement_sub = await attach(second)
            assert sub_b.closed and not replacement_sub.closed
            await until(
                lambda: (
                    bool(model_frames(replacement_peer))
                    and model_frames(replacement_peer)[-1]["source"][
                        "cli_frame_header"
                    ]["outputFrameId"]
                    >= reconnect_frame
                )
            )
            assert (
                b.session.startup.lease is lease_b and b.session.startup.ws is worker_b
            )
            assert lease_b.resource_state == "reserved" and len(env.workers) == 2
            assert (
                b.session.current_segment is segment_b
                and b.session.current_model_stream == model_stream_b
            )
            with pytest.raises(ValueError):
                b.accept(sub_b, msgspec.msgpack.encode(command))
        finally:
            try:
                await env.close()
            finally:
                for task in control_readers:
                    task.cancel()
                results = await asyncio.gather(*control_readers, return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException) and not isinstance(
                        result, asyncio.CancelledError
                    ):
                        raise result
        assert (
            not env.host.sessions and not env.runtime.players and not env.runtime.rooms
        )
        assert all(worker.closed for worker in env.workers)

    asyncio.run(run())


@pytest.mark.parametrize(
    "changed",
    [
        "duplicate",
        "copied-record",
        "sdk",
        "owner-epoch",
        "foreign-owner",
        "subject",
        "game",
        "legacy-existing",
        "legacy-incoming",
        "legacy-duplicate",
    ],
)
def test_member_source_dedup_keeps_original_identity_fences(
    tmp_path, monkeypatch, changed
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.player_media import (
        MemberSource,
        validate_media_launch,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
    )

    async def run():
        env = await model_owners(tmp_path, monkeypatch, max_sources=3)
        try:
            record, _ = await env.admit("alice")
            pipeline = record["media_pipeline"]
            factory = MemberSource(env.host.member_media, record)
            init = validate_media_launch(
                record["mediaLaunch"], record["binding"]["publicationId"]
            )
            init["whitebox"]["game_id"] = record["binding"]["gameId"]
            init["startup"]["request_id"] += "-another-create"
            user = record["caller"]["subject"]
            if changed == "copied-record":
                factory = MemberSource(env.host.member_media, dict(record))
            elif changed == "sdk":
                factory.engine.descriptor["playerBinding"]["viewportId"] = "replaced"
            elif changed == "owner-epoch":
                env.runtime.epoch = "9" * 32
            elif changed == "foreign-owner":
                factory = MemberSource(SimpleNamespace(host=env.host), record)
            elif changed == "subject":
                user = "bob"
            elif changed == "game":
                init["whitebox"]["game_id"] = "9" * 32
            elif changed in {"legacy-existing", "legacy-duplicate"}:
                pipeline.member_source = None
                if changed == "legacy-duplicate":
                    factory = None
            elif changed == "legacy-incoming":
                factory = None
            with pytest.raises((RenderError, ValueError)):
                await env.host.attach(
                    dict(user_id=user, trace_id="duplicate", message=init),
                    transport="webrtc-rtp",
                    detached=True,
                    member_source=factory,
                )
            assert len(env.host.sessions) == 1
            assert not pipeline.closed and len(env.views) == 1
        finally:
            # Restore only the test's classification override before owned teardown.
            if changed in {"legacy-existing", "legacy-duplicate"}:
                pipeline.member_source = MemberSource(env.host.member_media, record)
            await env.close()

    asyncio.run(run())


def test_peer_leave_or_reconnect_keeps_other_member_stream(tmp_path, monkeypatch):
    from test_whitebox_rtc_pipeline import Peer

    async def run():
        env = await model_owners(tmp_path, monkeypatch)
        try:
            first, _ = await env.admit("alice")
            second, _ = await env.admit("bob")
            receipt = await env.runtime.player_transport.get_player_media(first)
            pipeline = first["media_pipeline"]

            async def attach(message, peer):
                return await env.host.attach(
                    dict(user_id="alice", trace_id="attach", message=message),
                    transport="webrtc-rtp",
                    rtc_peer=peer,
                )

            _, sub = await attach(receipt["attachment"]["message"], Peer())
            original_source = pipeline.session.source_epoch
            for key, value in (
                ("source_epoch", "foreign-source"),
                (
                    "memberBinding",
                    {**receipt["memberBinding"], "memberId": second["member_id"]},
                ),
            ):
                altered = copy.deepcopy(receipt["attachment"]["message"])
                altered[key] = value
                with pytest.raises(ValueError):
                    await attach(altered, Peer())
                assert not sub.closed and pipeline.subscription is sub
            current = await env.runtime.player_transport.get_player_media(first)
            _, replacement = await attach(current["attachment"]["message"], Peer())
            assert sub.closed and not replacement.closed
            assert (
                pipeline.session.source_epoch == original_source and len(env.views) == 2
            )
            await env.runtime.leave(second["stream_id"])
            assert env.views[second["player_id"]].closed
            assert not env.views[first["player_id"]].closed and not pipeline.closed
            assert (
                env.runtime.player(first["stream_id"])["engine"]
                is env.views[first["player_id"]]
            )
            with pytest.raises(ValueError):
                await env.runtime.player_transport.get_player_media(second)
            await env.runtime.leave(first["stream_id"])
            assert pipeline.closed and env.views[first["player_id"]].closed
            assert replacement.closed
        finally:
            await env.close()

    asyncio.run(run())


def test_cross_member_or_source_epoch_is_rejected_before_rtc_replacement(
    tmp_path, monkeypatch
):
    from test_whitebox_rtc_pipeline import Peer

    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import reserve_rtc

    async def run():
        env = await model_owners(tmp_path, monkeypatch)
        try:
            record, _ = await env.admit("alice")
            pipeline = record["media_pipeline"]
            pipeline.init["whitebox"]["cli_fps"] = 30
            value = await env.runtime.player_transport.get_player_media(record)
            message = value["attachment"]["message"]
            assert set(message["whitebox"]) <= {
                "source_id",
                "world_id",
                "game_id",
                "scene_rendering",
            }
            assert pipeline.init["whitebox"]["cli_fps"] == 30
            _, sub = await env.host.attach(
                dict(
                    user_id="alice",
                    trace_id="attach",
                    message=value["attachment"]["message"],
                ),
                transport="webrtc-rtp",
                rtc_peer=Peer(),
            )
            fresh = await env.runtime.player_transport.get_player_media(record)
            altered = copy.deepcopy(fresh["attachment"]["message"])
            altered["sdkSession"]["epoch"] += 1
            with pytest.raises(ValueError):
                await reserve_rtc(env.host, dict(user_id="alice", message=altered))
            assert (
                not sub.closed and getattr(env.runtime, "rtc_preparation", None) is None
            )
            env.views[record["player_id"]].descriptor["epoch"] += 1
            with pytest.raises(ValueError):
                await reserve_rtc(
                    env.host,
                    dict(user_id="alice", message=fresh["attachment"]["message"]),
                )
            assert not sub.closed
            with pytest.raises(ValueError):
                await env.runtime.player_transport.get_player_media(record)
        finally:
            await env.close()

    asyncio.run(run())


def test_signed_launch_budget_rejects_oversize_before_directory_or_model_allocation():
    import base64

    import httpx
    from fastapi import FastAPI
    from test_whitebox_room_control import SECRET, ticket

    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control import (
        MAX_CONTROL_BYTES,
        MAX_MEMBER_CONTROL_BYTES,
        RoomControlCapabilities,
        install_gateway_room_control,
    )

    async def run():
        directory = RoomDirectory(MemoryRoomStore(), {"owner": "http://room"})

        async def upstream(request):
            pytest.fail("Oversized control cannot contact an owner")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(upstream)
        ) as backend:
            app = FastAPI()
            install_gateway_room_control(
                app,
                SimpleNamespace(
                    directory=directory,
                    owners={"owner": "http://room"},
                    client=backend,
                    token="internal",
                ),
                verifier=RoomControlCapabilities(SECRET, {"wtp"}, now=lambda: 100),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client:
                query = json.dumps(FIXTURE["request"]).encode()
                native = b" " * MAX_CONTROL_BYTES + query
                result = await client.post(
                    "/v1/whitebox/room-control",
                    content=native,
                    headers={"authorization": "Bearer " + ticket(native)},
                )
                assert result.status_code == 413
                launch = media_launch()
                launch["initMsgpackBase64"] = base64.b64encode(
                    b"x" * (8 * 1024**2 + 1)
                ).decode()
                body = dict(
                    schemaVersion=1,
                    action="admitPlayer",
                    binding={
                        k: FIXTURE["request"]["memberBinding"][k]
                        for k in ("gameId", "ownerEpoch", "publicationId")
                    },
                    admission=dict(
                        operationId="a", requestId="a", parametersSha256="a" * 64
                    ),
                    mediaLaunch=launch,
                )
                raw = json.dumps(body).encode()
                assert len(raw) < MAX_MEMBER_CONTROL_BYTES
                assert (
                    await client.post(
                        "/v1/whitebox/room-control",
                        content=raw,
                        headers={"authorization": "Bearer " + ticket(raw)},
                    )
                ).status_code == 400
                huge = b" " * MAX_MEMBER_CONTROL_BYTES + query
                assert (
                    await client.post("/v1/whitebox/room-control", content=huge)
                ).status_code == 413
                assert not directory.store.values

    asyncio.run(run())


def test_model_golden_preserves_original_h264_and_member_sdk_identity():
    import base64

    import av
    import msgspec

    from sglang.multimodal_gen.runtime.realtime.whitebox.native_pixels import (
        read_frame_identity,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_auth import RTCCapabilities

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/player-media-model-v1.json").read_text()
    )
    assert validate_request(fixture["request"]) == fixture["request"]
    response = fixture["response"]
    model = msgspec.msgpack.decode(base64.b64decode(fixture["modelFrameMsgpackBase64"]))
    frame = read_frame_identity(model["source"]["cli_frame_header"])
    assert model["stream"] == "model" and response["model"]["status"] == "ready"
    assert model["source"]["source_epoch"] == response["execution"]["sourceEpoch"]
    assert frame["sessionId"] == response["session"]["sessionId"]
    assert frame["epoch"] == 3 and frame["source"]["sdkEpoch"] == 11
    message = response["attachment"]["message"]
    assert message["memberBinding"] == response["memberBinding"]
    assert (
        message["sdkSession"]["playerBinding"] == response["session"]["playerBinding"]
    )
    claims = RTCCapabilities(fixture["attachmentTestKey"])._decode(
        message["member_authorization"]
    )
    assert claims["aud"] == "whitebox-member-attachment" and claims["exp"] == 160
    decoder = av.CodecContext.create("h264", "r")
    decoded = decoder.decode(av.Packet(model["payload"])) + decoder.decode(None)
    assert len(decoded) == 1 and (decoded[0].width, decoded[0].height) == (160, 96)


def test_room_retirement_closes_sdk_independently_of_model_cleanup(
    tmp_path, monkeypatch
):
    async def run():
        env = await model_owners(tmp_path, monkeypatch)
        try:
            record, _ = await env.admit("alice")
            pipeline = record["media_pipeline"]
            view = env.views[record["player_id"]]
            gate = asyncio.Event()
            original_close = env.runtime.member_media.close

            async def close(target):
                await gate.wait()
                await original_close(target)

            monkeypatch.setattr(env.runtime.member_media, "close", close)
            ending = asyncio.create_task(
                env.runtime._end(record["binding"]["gameId"], "completed")
            )
            for _ in range(200):
                if view.closed:
                    break
                await asyncio.sleep(0.001)
            assert view.closed and not ending.done()
            gate.set()
            assert await ending
            assert pipeline.closed and view.closed and not env.host.sessions
        finally:
            await env.close()

    asyncio.run(run())


@pytest.mark.parametrize("action", ["leavePlayer", "retireRoom"])
def test_explicit_original_cleanup_retries_model_after_independent_sdk_retirement(
    tmp_path, monkeypatch, action
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.player_media import (
        member_binding,
    )

    async def run():
        env = await model_owners(tmp_path, monkeypatch)
        try:
            record, admission = await env.admit("alice")
            pipeline = record["media_pipeline"]
            view = env.views[record["player_id"]]
            media = env.runtime.member_media
            # Exercise cleanup of an admitted model, not cancellation before
            # source-first capability discovery has launched the model task.
            async with asyncio.timeout(5):
                while not env.workers:
                    await asyncio.sleep(0.01)
            original_close = media.close
            attempts = []

            async def transient_close(target):
                assert target is record
                attempts.append(target)
                if len(attempts) == 1:
                    raise ValueError("Temporary member media cleanup failure")
                await original_close(target)

            monkeypatch.setattr(media, "close", transient_close)
            if action == "leavePlayer":
                request = dict(
                    schemaVersion=1,
                    action=action,
                    memberBinding=member_binding(record),
                    operation=dict(
                        operationId="leave-alice",
                        requestId="leave-request",
                        kind="leave",
                        parametersSha256="c" * 64,
                    ),
                )
                query = {
                    **admission,
                    "action": "getAdmission",
                    "leaveOperationId": "leave-alice",
                }
            else:
                room = env.runtime.rooms[record["binding"]["gameId"]]
                request = dict(
                    schemaVersion=1,
                    action=action,
                    **room["control_record"]["invocation"],
                    execution=record["binding"],
                )
                query = {
                    key: value for key, value in request.items() if key != "execution"
                }
                query["action"] = "getOperation"

            async def invoke(body):
                return await env.control.handle(
                    record["caller"], validate_request(body), env.route
                )

            await invoke(request)
            await env.control.drain()
            # SDK retirement can also end the borrowed media reader naturally;
            # that does not turn the failed original cleanup into an ACK.
            assert len(attempts) == 1 and view.closed
            result = await invoke(query)
            assert result["status"] in {"blocked", "unknown"}
            assert not result.get("cleanupComplete", False)
            assert len(attempts) == 1  # Observation cannot restart media cleanup.

            await invoke(request)
            await env.control.drain()
            result = await invoke(query if action == "leavePlayer" else request)
            assert result["status"] == (
                "retired" if action == "leavePlayer" else "succeeded"
            )
            assert result["cleanupComplete"] is True
            assert len(attempts) == 2 and pipeline.closed and view.closed
            assert not env.host.sessions and not env.runtime.players
            assert await invoke(request) == result  # A lost reply reuses the receipt.
            assert len(attempts) == 2 and len(env.views) == len(env.workers) == 1
            if action == "retireRoom":
                assert not env.runtime.rooms and not env.runtime.retiring
        finally:
            await env.close()

    asyncio.run(run())


@pytest.mark.parametrize("change", ["hash", "extra", "world", "attempt"])
def test_new_frozen_publication_identity_is_validated_before_create(change):
    from test_whitebox_room_control import request_body

    body = request_body()
    body["createContext"] = copy.deepcopy(FIXTURE["createContext"])
    assert validate_request(body) == body
    public = body["createContext"]["frozenPublication"]["publicationIdentity"]
    if change == "hash":
        public.pop("manifestSha256")
    elif change == "extra":
        public["sdkRunId"] = "not-public"
    elif change == "world":
        public["worldId"] = "other"
    else:
        public["attempt"] = 8
    with pytest.raises(ValueError):
        validate_request(body)


def test_legacy_native_session_does_not_invent_new_publication_identity():
    async def run():
        owner = await owners()
        frozen = owner.runtime.rooms[owner.record["binding"]["gameId"]][
            "control_record"
        ]["invocation"]["createContext"]["frozenPublication"]
        frozen.pop("publicationIdentity")
        legacy = await owner.control.handle(
            owner.caller, {**owner.request, "action": "getPlayerSession"}, owner.route
        )
        assert legacy == dict(status="ready", session=owner.published)
        with pytest.raises(ValueError):
            await owner.control.handle(owner.caller, owner.request, owner.route)

    asyncio.run(run())


@pytest.mark.parametrize("changed", ["missing", "hash", "attempt"])
def test_member_launch_publication_fails_before_native_and_model_allocation(
    tmp_path, monkeypatch, changed
):
    async def run():
        env = await model_owners(tmp_path, monkeypatch)
        try:
            room = next(iter(env.runtime.rooms.values()))
            frozen = room["control_record"]["invocation"]["createContext"][
                "frozenPublication"
            ]
            if changed == "missing":
                frozen.pop("publicationIdentity")
            elif changed == "hash":
                frozen["publicationIdentity"]["manifestSha256"] = "f" * 64
            else:
                frozen["publicationIdentity"]["attempt"] = 8
            with pytest.raises(ValueError):
                await env.admit("alice")
            assert (
                not env.views
                and not env.runtime.players
                and not env.host.sessions
                and not env.workers
            )
            assert not env.control.admissions
        finally:
            await env.close()

    asyncio.run(run())


def test_media_projection_preserves_complete_identity_and_not_world_run_fields():
    async def run():
        owner = await owners()
        frozen = owner.runtime.rooms[owner.record["binding"]["gameId"]][
            "control_record"
        ]["invocation"]["createContext"]["frozenPublication"]
        for world in (frozen["world"], FIXTURE["alternateInteractiveWorld"]):
            frozen["world"] = copy.deepcopy(world)
            result = await owner.runtime.player_transport.get_player_media(owner.record)
            assert result["publication"] == FIXTURE["response"]["publication"]
            assert result["publication"]["attempt"] == 7
            assert (
                not {"sdkRunId", "platformRunId", "interactiveImportId"}
                & result["publication"].keys()
            )

    asyncio.run(run())


def test_actual_wtp_frozen_context_and_launch_use_original_member_owners(
    tmp_path, monkeypatch
):
    wire = json.loads(
        (Path(__file__).parent / "fixtures/room-member-media-wtp-v1.json").read_text()
    )
    for request in wire["requests"]:
        assert validate_request(request) == request
    create = next(r for r in wire["requests"] if r["action"] == "createRoom")
    admit = next(r for r in wire["requests"] if r["action"] == "admitPlayer")

    async def run():
        env = await model_owners(
            tmp_path,
            monkeypatch,
            create_request=create,
            launch=admit["mediaLaunch"],
            worker_protocol="whitebox-startup-prepare-v2",
        )
        try:
            record, _ = await env.admit("alice")
            receipt = await env.runtime.player_transport.get_player_media(record)
            frozen = create["createContext"]["frozenPublication"]
            assert receipt["kind"] == "whitebox", record.get("media")
            assert (
                receipt["publication"]
                == frozen["publicationIdentity"]
                == admit["mediaLaunch"]["publicationIdentity"]
            )
            assert (
                record["mediaLaunch"]["initMsgpackBase64"]
                == admit["mediaLaunch"]["initMsgpackBase64"]
            )
            assert (
                env.runtime.rooms[record["binding"]["gameId"]]["control_record"][
                    "invocation"
                ]["createContext"]
                == create["createContext"]
            )
            assert {"platformRunId", "sdkRunId"} <= frozen["world"].keys()
            assert not {"platformRunId", "sdkRunId"} & receipt["publication"].keys()
            assert len(env.views) == len(env.host.sessions) == 1
            from test_whitebox_rtc_pipeline import Peer

            peer = Peer()
            _, sub = await env.host.attach(
                dict(
                    user_id="alice",
                    trace_id="wtp-v2",
                    message=receipt["attachment"]["message"],
                ),
                transport="webrtc-rtp",
                rtc_peer=peer,
            )
            startup = record["media_pipeline"].session.startup
            startup.rtc_ready.set()
            async with asyncio.timeout(10):
                while not any(
                    message["stream"] == "model" for message, _ in peer.calls
                ):
                    for i, (_, done) in enumerate(peer.calls):
                        if not done.done():
                            peer.finish(i)
                    await asyncio.sleep(0.01)
            ready = await env.runtime.player_transport.get_player_media(record)
            assert ready["model"]["status"] == "ready"
            assert startup.receipt()["protocol"] == "whitebox-startup-prepare-v2"
            assert startup.attempt == 0 and len(env.workers) == 1
            assert startup.cli_rgb_sha256 is not None
        finally:
            await env.close()

    asyncio.run(run())


def test_member_media_failure_log_retains_member_and_original_admission(
    player_log_capture,
):
    import json
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.player_media import MemberMedia

    async def run():
        owner = MemberMedia(SimpleNamespace(runtime=SimpleNamespace()))
        record = dict(
            binding=dict(
                gameId="media-log-game",
                ownerEpoch="media-log-owner",
                publicationId="media-log-pub",
            ),
            admission=dict(
                operationId="media-log-admission", requestId="media-log-request"
            ),
            member_id="media-log-member",
            mediaLaunch={},
        )
        await owner.start(record)
        assert record["media"]["status"] == "unavailable"
        await owner.start(record)
        rows = [
            json.loads(row.message)
            for row in player_log_capture.records
            if '"event":"whitebox.player.failed"' in row.message
        ]
        assert len(rows) == 1
        assert rows[0]["operationId"] == "media-log-admission"
        assert rows[0]["memberId"] == "media-log-member"
        assert rows[0]["gameId"] == "media-log-game"

    asyncio.run(run())


def test_distinct_members_can_prepare_rtc_at_the_same_time(tmp_path, monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import (
        release_reconnect_preparation,
        reserve_rtc,
    )

    async def run():
        env = await model_owners(tmp_path, monkeypatch)
        env.host.max_sessions = 2
        prepared = []
        try:
            requests = []
            for name in ("alice", "bob"):
                record, _ = await env.admit(name)
                receipt = await env.runtime.player_transport.get_player_media(record)
                requests.append(
                    dict(user_id=name, message=receipt["attachment"]["message"])
                )
            for request in requests:
                prepared.append(await reserve_rtc(env.host, request))
            assert prepared[0][1] is not prepared[1][1]
            assert prepared[0][0] != prepared[1][0]
            with pytest.raises(RenderError) as error:
                await reserve_rtc(env.host, requests[0])
            assert error.value.code == "CAPACITY_EXHAUSTED"
            release_reconnect_preparation(prepared[0][1], prepared[0][0])
            assert prepared[1][1].rtc_reconnect.reservation == prepared[1][0]
        finally:
            for reservation, pipeline in prepared:
                release_reconnect_preparation(pipeline, reservation)
            await env.close()

    asyncio.run(run())
