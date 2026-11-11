# SPDX-License-Identifier: Apache-2.0
"""Original source factory and durable SQLite journal; CPU-only execution peers."""

import asyncio
import base64
import copy
from types import SimpleNamespace as NS

import msgspec
import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox import render_host
from sglang.multimodal_gen.runtime.realtime.whitebox.player_control import (
    SinglePlayerControl,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import PROTOCOL
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import new_id


def operation():
    init = dict(
        type="init",
        control_protocol=PROTOCOL,
        source_operation="create",
        prompt="snow",
        render_mode="whitebox",
        whitebox=dict(source_id="source", world_id="world"),
    )
    return dict(
        operationId="launch-1",
        parametersSha256="a" * 64,
        caller=dict(serviceId="wtp", subject="user"),
        frozenPublication=dict(
            status="verified",
            scope="world-frozen-publication",
            world=dict(sourceId="source", worldId="world", publicationId="pub"),
            playConfiguration=dict(playMode="single-player"),
        ),
        frozenLaunchParameters=dict(
            transport="websocket",
            publicationIdentity=dict(
                sourceId="source",
                worldId="world",
                publicationId="pub",
                sourceManifestSha256="1" * 64,
                renditionInputSha256="2" * 64,
                manifestSha256="3" * 64,
            ),
            initMsgpackBase64=base64.b64encode(msgspec.msgpack.encode(init)).decode(),
        ),
        _route=dict(
            game_id=new_id(),
            owner_id="owner",
            owner_url="http://owner",
            owner_epoch=None,
        ),
    )


class Pipeline:
    media_transport = "websocket"
    """Only native/model work is replaced; RenderHost insertion/capacity are real."""
    created = []
    gate = None

    def __init__(self, host, init, user, trace):
        self.host, self.init, self.user_id = host, init, user
        self.game_id = init["whitebox"]["game_id"]
        self.session_id, self.token = new_id(), new_id()
        self.session = NS(
            source_epoch=new_id(),
            startup=None,
            binding_ended=False,
            player_remaining_seconds=1800,
        )
        self.generation, self.control_sequence, self.history_floor = 0, 3, 0
        self.closed, self.ready = False, None
        self.subscription = None
        self.created.append(self)

    def attach(self, resume=None):
        if resume is not None:
            assert resume["resume_token"] == self.token
        self.generation += 1
        self.subscription = NS(generation=self.generation)
        return self.subscription

    async def start(self):
        if self.gate:
            await self.gate.wait()
        self.ready = dict(type="session.ready")
        self.control_started = True

    async def close(self, reason=None):
        self.closed = True
        self.host.sessions.pop(self.session_id, None)


def host(tmp_path, monkeypatch, epoch=None):
    Pipeline.created, Pipeline.gate = [], None
    monkeypatch.setattr(render_host, "RenderPipeline", Pipeline)
    runtime = NS(spool=tmp_path, epoch=epoch or new_id(), owner_id="owner", rooms={})
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec

    value = render_host.RenderHost(
        runtime, NS(spec=VideoSpec()), None, None, None, None
    )
    return value, SinglePlayerControl(value)


def test_lost_start_reply_queries_same_source(tmp_path, monkeypatch):
    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        Pipeline.gate = asyncio.Event()
        request = asyncio.create_task(control.start(op))
        while not Pipeline.created and not request.done():
            await asyncio.sleep(0)
        assert Pipeline.created, await request
        request.cancel()  # Lost request/reply must not cancel the source owner.
        await asyncio.gather(request, return_exceptions=True)
        Pipeline.gate.set()
        result = await control.start(op)
        assert result["state"] == "ready"
        assert (await control.get_operation(op))["execution"] == result["execution"]
        assert len(Pipeline.created) == 1
        await control.close(op)
        await owner.close()

    asyncio.run(run())


def test_concurrent_same_operation_starts_once(tmp_path, monkeypatch):
    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        replies = await asyncio.gather(*(control.start(op) for _ in range(8)))
        assert len(Pipeline.created) == 1
        assert all(reply == replies[0] for reply in replies)
        await control.close(op)
        await owner.close()

    asyncio.run(run())


def test_same_id_different_hash_conflicts(tmp_path, monkeypatch):
    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        await control.start(op)
        changed = dict(op, parametersSha256="b" * 64)
        for method in (
            control.start,
            control.get_operation,
            control.get_attachment,
            control.close,
        ):
            with pytest.raises(ValueError, match="changed"):
                await method(changed)
        assert not Pipeline.created[0].closed
        await control.close(op)
        await owner.close()

    asyncio.run(run())


def test_owner_restart_never_recreates_unknown_source(tmp_path, monkeypatch):
    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        first = await control.start(op)
        await owner.close()
        restarted, control = host(tmp_path, monkeypatch)
        result = await control.start(op)
        assert result["state"] == "lost"
        assert result["execution"] == first["execution"]
        assert (await control.close(op))["state"] == "lost"
        assert Pipeline.created == []
        await restarted.close()

    asyncio.run(run())


def test_cancel_before_start_blocks_late_start(tmp_path, monkeypatch):
    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        assert (await control.close(op))["state"] == "cancelled"
        assert (await control.start(op))["state"] == "cancelled"
        assert Pipeline.created == []
        await owner.close()

    asyncio.run(run())


def test_cancel_during_start_releases_original_only(tmp_path, monkeypatch):
    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        Pipeline.gate = asyncio.Event()
        pending = asyncio.create_task(control.start(op))
        while not Pipeline.created and not pending.done():
            await asyncio.sleep(0)
        assert Pipeline.created, await pending
        assert (await control.close(op))["state"] in {"cancelling", "cancelled"}
        Pipeline.gate.set()
        assert (await pending)["state"] == "cancelled"
        assert len(Pipeline.created) == 1 and Pipeline.created[0].closed
        assert not owner.sessions
        await owner.close()

    asyncio.run(run())


def test_attachment_never_creates_and_short_capability_fences_original(
    tmp_path, monkeypatch
):
    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        assert (await control.get_attachment(op))["state"] == "unknown"
        assert Pipeline.created == []
        init = msgspec.msgpack.decode(
            base64.b64decode(op["frozenLaunchParameters"]["initMsgpackBase64"])
        )
        init["whitebox"]["initial_groups"] = [f"group-{i}" for i in range(1000)]
        op["frozenLaunchParameters"]["initMsgpackBase64"] = base64.b64encode(
            msgspec.msgpack.encode(init)
        ).decode()
        first = await control.start(op)
        reply = await control.get_attachment(op)
        attachment = reply["attachment"]
        assert len(attachment["message"]["player_authorization"]) < 4096
        assert "resume_token" not in attachment["message"]
        assert attachment["expiresAt"] > control.now()
        pipeline, sub = await owner.attach(
            dict(user_id="user", message=attachment["message"])
        )
        assert pipeline.session_id == first["execution"]["sourceSessionId"]
        assert sub.generation == 1 and len(Pipeline.created) == 1
        with pytest.raises(ValueError):
            await owner.attach(dict(user_id="user", message=attachment["message"]))
        newer = await control.get_attachment(op)
        control.now = lambda: newer["attachment"]["expiresAt"] + 1
        with pytest.raises(ValueError):
            await owner.attach(
                dict(user_id="user", message=newer["attachment"]["message"])
            )
        await control.close(op)
        await owner.close()

    asyncio.run(run())


def test_changed_launch_or_caller_cannot_use_original_source(tmp_path, monkeypatch):
    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        await control.start(op)
        changed = copy.deepcopy(op)
        changed["frozenPublication"]["world"]["publicationId"] = "other"
        with pytest.raises(ValueError, match="changed"):
            await control.start(changed)
        stranger = copy.deepcopy(op)
        stranger["caller"]["subject"] = "other"
        assert (await control.get_operation(stranger))["state"] == "unknown"
        assert not Pipeline.created[0].closed
        await control.close(op)
        await owner.close()

    asyncio.run(run())


def test_single_service_ticket_scope_origin_subject_and_no_public_defaults(
    tmp_path, monkeypatch
):
    import json

    import httpx
    from fastapi import FastAPI
    from test_whitebox_room_control import CALLER, SECRET, ticket

    from sglang.multimodal_gen.runtime.realtime.whitebox.player_control import (
        SinglePlayerCapabilities,
        install_gateway_single_player_control,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        MemoryRoomStore,
        RoomDirectory,
    )

    async def run():
        op = operation()
        op.pop("_route")
        op["caller"] = CALLER
        body = dict(schemaVersion=1, action="start", singlePlayerOperation=op)
        raw = json.dumps(body).encode()
        verifier = SinglePlayerCapabilities(SECRET, {"wtp"}, now=lambda: 100)
        good = dict(
            aud="whitebox-single-control",
            scope="single-player-control",
            origin="https://wtp.test",
        )
        for bad in (
            dict(scope="room-control"),
            dict(origin="https://other.test"),
            dict(sub="other"),
            dict(aud="whitebox-room-control"),
        ):
            with pytest.raises(ValueError):
                verifier.verify(
                    ticket(raw, **(good | bad)),
                    raw=raw,
                    action="start",
                    origin="https://wtp.test",
                    operation=op,
                )
        app = FastAPI()
        proxy = NS(
            directory=RoomDirectory(MemoryRoomStore(), {"owner": "http://owner"}),
            owners={"owner": "http://owner"},
            token="internal",
        )
        install_gateway_single_player_control(app, proxy, verifier=None)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/whitebox/single-player-control", content=raw
            )
            assert response.status_code == 503
        assert (
            verifier.verify(
                ticket(raw, **good),
                raw=raw,
                action="start",
                origin="https://wtp.test",
                operation=op,
            )
            == CALLER
        )

    asyncio.run(run())


def test_gateway_lost_reply_reconciles_original_directory_and_owner(
    tmp_path, monkeypatch
):
    import json

    import httpx
    from fastapi import FastAPI
    from test_whitebox_room_control import CALLER, SECRET, ticket

    from sglang.multimodal_gen.runtime.realtime.whitebox.player_control import (
        SinglePlayerCapabilities,
        install_gateway_single_player_control,
        install_owner_single_player_control,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        HTTPRoomDirectory,
        MemoryRoomStore,
        RoomDirectory,
        install_room_routes,
    )

    async def run():
        owner, control = host(tmp_path, monkeypatch)
        owner_app = FastAPI()
        install_owner_single_player_control(owner_app, owner, "internal")
        directory = RoomDirectory(MemoryRoomStore(), {"owner": "http://owner"})
        coordinator_app = FastAPI()
        install_room_routes(coordinator_app, directory, "internal")
        coordinator_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=coordinator_app),
            base_url="http://coordinator",
        )
        routed_directory = HTTPRoomDirectory(
            coordinator_client, "http://coordinator", "internal"
        )
        calls = []
        lose = True
        owner_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=owner_app), base_url="http://owner"
        )

        class Transport:
            async def post(self, url, **kwargs):
                nonlocal lose
                calls.append(kwargs["json"]["route"]["game_id"])
                reply = await owner_client.post(url, **kwargs)
                if lose:
                    lose = False
                    raise httpx.ReadTimeout("Reply was lost")
                return reply

        proxy = NS(
            directory=routed_directory,
            owners={"owner": "http://owner"},
            token="internal",
            client=Transport(),
        )
        app = FastAPI()
        install_gateway_single_player_control(
            app,
            proxy,
            verifier=SinglePlayerCapabilities(SECRET, {"wtp"}, now=lambda: 100),
        )
        op = operation()
        op.pop("_route")
        op["caller"] = CALLER
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            for number, action in enumerate(
                ("start", "getOperation", "start", "getAttachment", "close"), 1
            ):
                body = dict(schemaVersion=1, action=action, singlePlayerOperation=op)
                if action == "getAttachment":
                    stale = dict(
                        body, singlePlayerOperation=dict(op, controlSequence=999)
                    )
                    stale_raw = json.dumps(stale).encode()
                    stale_auth = ticket(
                        stale_raw,
                        jti="0" * 32,
                        aud="whitebox-single-control",
                        scope="single-player-control",
                        origin=None,
                    )
                    stale_reply = await client.post(
                        "/v1/whitebox/single-player-control",
                        content=stale_raw,
                        headers={"Authorization": "Bearer " + stale_auth},
                    )
                    assert stale_reply.status_code == 409
                raw = json.dumps(body).encode()
                auth = ticket(
                    raw,
                    jti=str(number) * 32,
                    aud="whitebox-single-control",
                    scope="single-player-control",
                    origin=None,
                )
                response = await client.post(
                    "/v1/whitebox/single-player-control",
                    content=raw,
                    headers={"Authorization": "Bearer " + auth},
                )
                assert response.status_code == 200, response.text
                reply = response.json()
                assert reply["state"] == (
                    "unknown"
                    if number == 1
                    else "cancelled"
                    if action == "close"
                    else "ready"
                )
        assert len(set(calls)) == 1 and len(Pipeline.created) == 1
        assert Pipeline.created[0].closed
        await owner_client.aclose()
        await coordinator_client.aclose()
        await owner.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "transport,producer,codec,resolution",
    [
        ("websocket", None, None, None),
        ("webrtc-rtp", None, None, None),
        ("webrtc-rtp", "canonical", None, None),
        ("webrtc-rtp", "legacy", None, None),
        ("webrtc-rtp", None, "h264", "480p"),
        ("webrtc-rtp", None, "vp9", "480p"),
        ("webrtc-rtp", None, "av1", "480p"),
        ("webrtc-rtp", None, "h264", "720p"),
        ("webrtc-rtp", None, "vp9", "720p"),
        ("webrtc-rtp", None, "av1", "720p"),
        ("webrtc-rtp", None, "h264-default", "720p"),
        ("webrtc-rtp", None, "h264-default", "deployment720"),
        ("webrtc-rtp", None, "h264-default", "deployment720-env"),
        ("webrtc-rtp", None, None, "deployment480"),
    ],
)
def test_real_original_source_factory_and_native_rtc_attachment_without_reinit(
    tmp_path, monkeypatch, transport, producer, codec, resolution
):
    import importlib
    import json
    from pathlib import Path

    from test_whitebox_interactive import BrowserUI, Engine

    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncArchiveManager,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        MemoryRoomStore,
        RoomDirectory,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import (
        release_reconnect_preparation,
        reserve_rtc,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig

    cli_owner = importlib.import_module(
        "sglang.multimodal_gen.runtime.realtime.whitebox.cli_room"
    )

    captured = None
    if producer:
        from test_whitebox_startup import V2_PROTOCOL, Lease

        from sglang.multimodal_gen.runtime.realtime.whitebox.player_control import (
            SinglePlayerCapabilities,
        )

        captured = json.loads(
            (
                Path(__file__).parent / f"fixtures/player-single-wtp-{producer}-v1.json"
            ).read_text()
        )
        verifier = SinglePlayerCapabilities(
            "fixture-original-single-secret-at-least-thirty-two",
            {"wtp"},
            now=lambda: 1793227552,
        )
        for wire in captured["wireRequests"]:
            assert json.loads(wire["rawBodyUtf8"]) == wire["body"]
            assert (
                verifier.verify(
                    wire["authorization"].removeprefix("Bearer "),
                    raw=wire["rawBodyUtf8"].encode(),
                    action=wire["body"]["action"],
                    origin=None,
                    operation=wire["body"]["singlePlayerOperation"],
                )
                == wire["body"]["singlePlayerOperation"]["caller"]
            )
        monkeypatch.setenv("WHITEBOX_MODEL_STARTUP_PREPARE", "1")

    metadata_entered, metadata_release = asyncio.Event(), asyncio.Event()

    async def checked_metadata(*args, expected_publication, **kwargs):
        metadata_entered.set()
        await metadata_release.wait()
        return 1.0  # Matched A locator; actual driver descriptor is checked next.

    monkeypatch.setattr(cli_owner, "prepare_publication", checked_metadata)
    monkeypatch.setenv(
        "WHITEBOX_CLI_ENTRYPOINT", "/cpu-fixture/sdk/packages/cli/bin/worldkit.mjs"
    )

    async def run():
        monkeypatch.setenv("WHITEBOX_NATIVE_RTC_ENABLED", "1")
        spec = (
            VideoSpec(1280, 720)
            if resolution in {"720p", "deployment720", "deployment720-env"}
            else VideoSpec(832, 480)
            if codec or resolution == "deployment480"
            else VideoSpec(32, 32)
        )
        engine = Engine(spec)
        engine.renderer = BrowserUI()
        joins = []
        op = (
            copy.deepcopy(captured["requests"][0]["singlePlayerOperation"])
            if captured
            else operation()
        )
        if not captured:
            op["frozenLaunchParameters"]["transport"] = transport
        selected = None
        if codec:
            from test_whitebox_room_control import SECRET, ticket
            from test_whitebox_session_encoding import settings

            from sglang.multimodal_gen.runtime.realtime.whitebox.player_control import (
                SinglePlayerCapabilities,
            )

            selected = settings("h264" if codec == "h264-default" else codec)
            init = msgspec.msgpack.decode(
                base64.b64decode(op["frozenLaunchParameters"]["initMsgpackBase64"])
            )
            init.update(source_operation="create.manual", inference_control="manual")
            init["whitebox"].update(
                source_encoding=selected,
                resolution="720p" if spec.width == 1280 else resolution,
                cli_fps=60 if spec.width == 1280 else 30,
            )
            op["frozenLaunchParameters"]["initMsgpackBase64"] = base64.b64encode(
                msgspec.msgpack.encode(init)
            ).decode()
            signed_op = {k: v for k, v in op.items() if k != "_route"}
            raw = json.dumps(
                dict(schemaVersion=1, action="start", singlePlayerOperation=signed_op)
            ).encode()
            auth = ticket(
                raw,
                aud="whitebox-single-control",
                scope="single-player-control",
                origin=None,
                sub=op["caller"]["subject"],
            )
            guard = SinglePlayerCapabilities(SECRET, {"wtp"}, now=lambda: 100)
            changed = copy.deepcopy(signed_op)
            changed_init = copy.deepcopy(init)
            changed_init["whitebox"]["source_encoding"]["gop"] = 12
            changed["frozenLaunchParameters"]["initMsgpackBase64"] = base64.b64encode(
                msgspec.msgpack.encode(changed_init)
            ).decode()
            changed_raw = json.dumps(
                dict(schemaVersion=1, action="start", singlePlayerOperation=changed)
            ).encode()
            with pytest.raises(ValueError):
                guard.verify(
                    auth,
                    raw=changed_raw,
                    action="start",
                    origin=None,
                    operation=changed,
                )
            assert (
                guard.verify(
                    auth, raw=raw, action="start", origin=None, operation=signed_op
                )
                == op["caller"]
            )
        if codec == "h264-default":
            del init["whitebox"]["source_encoding"]
            if resolution in {"deployment720", "deployment720-env"}:
                del init["whitebox"]["resolution"]
            if resolution == "deployment720-env":
                del init["whitebox"]["cli_fps"]
                monkeypatch.setenv("WHITEBOX_CLI_FPS", "60")
            op["frozenLaunchParameters"]["initMsgpackBase64"] = base64.b64encode(
                msgspec.msgpack.encode(init)
            ).decode()
        original = copy.deepcopy(op)

        class Driver:
            max_players = 1
            descriptor = dict(
                artifact=dict(
                    op["frozenLaunchParameters"]["publicationIdentity"],
                    worldBuildHash=op["frozenLaunchParameters"]["publicationIdentity"][
                        "manifestSha256"
                    ],
                ),
                worldBuildHash=op["frozenLaunchParameters"]["publicationIdentity"][
                    "manifestSha256"
                ],
            )

            async def join(self, **kwargs):
                joins.append(kwargs)
                return engine

            async def close(self):
                pass

        async def driver(**kwargs):
            return Driver()

        directory = RoomDirectory(MemoryRoomStore(), {"owner": "http://owner"})
        op["_route"] = await directory.resolve(
            op.get("_route", {}).get("game_id", new_id()),
            op["frozenPublication"]["world"]["sourceId"]
            + "/"
            + op["frozenPublication"]["world"]["worldId"],
            create=True,
        )
        archives = AsyncArchiveManager()
        runtime = RoomRuntime(
            directory=directory,
            owner_id="owner",
            driver_factory=driver,
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
            capture_clock=bool(codec),
        )
        owner = render_host.RenderHost(
            runtime, config, archives, Reports(tmp_path, archives), None, None
        )
        from sglang.multimodal_gen.runtime.realtime.whitebox import session_encoding

        if codec == "h264-default":
            selected = session_encoding.default_source_encoding(config)
        probes = []

        def preflight(value, fps, *, width, height):
            assert not owner.sessions and not runtime.rooms and not joins
            assert not owner.lock.locked()
            assert value == selected
            assert (width, height) == (spec.width, spec.height)
            if resolution in {"deployment720", "deployment720-env"}:
                assert fps == 60
            probes.append((copy.deepcopy(value), fps))
            return "42c029" if codec in {"h264", "h264-default"} else None

        monkeypatch.setattr(session_encoding, "preflight_source_encoding", preflight)
        control = SinglePlayerControl(owner)
        if captured:
            owner.make_segment_leases = lambda *_: Lease(V2_PROTOCOL)
        try:
            launching = asyncio.create_task(control.start(op))
            async with asyncio.timeout(5):
                while not metadata_entered.is_set() and not launching.done():
                    await asyncio.sleep(0)
            assert metadata_entered.is_set(), await launching
            early = await control.get_attachment(op)
            assert early["state"] == "starting" and "attachment" not in early
            metadata_release.set()
            result = await launching
            pipeline = next(iter(owner.sessions.values()))
            assert isinstance(pipeline.session, SourceSession)
            assert (
                pipeline.media_transport == transport and pipeline.subscription is None
            )
            if codec:
                assert len(probes) == 1
                assert pipeline.session.config.source_encoding == (
                    None if codec == "h264-default" else selected
                )
                assert config.source_encoding is None
                assert pipeline.rtc_source_profile == (
                    "42c029" if codec in {"h264", "h264-default"} else None
                )
                assert runtime.rtc_preparation is None
                assert pipeline.init["source_operation"] == "create.manual"
                assert pipeline.init["inference_control"] == "manual"
            else:
                assert not probes
            async with asyncio.timeout(5):
                while (await control.get_operation(op))["state"] != "ready":
                    await asyncio.sleep(0.01)
            if transport == "websocket":
                ready = pipeline.ready
                pipeline.ready = None
                pipeline.publish(
                    dict(type="startup.accepted", request_id="original-startup")
                )
            attach = (await control.get_attachment(op))["attachment"]
            body = dict(user_id=op["caller"]["subject"], message=attach["message"])
            if transport == "webrtc-rtp":
                if codec and codec != "h264-default":
                    changed_body = copy.deepcopy(body)
                    changed_body["message"]["whitebox"]["source_encoding"]["gop"] = 12
                    subscriber = object()
                    pipeline.subscription = subscriber
                    with pytest.raises(ValueError):
                        await reserve_rtc(owner, changed_body)
                    assert pipeline.subscription is subscriber
                    assert runtime.rtc_preparation is None
                    pipeline.subscription = None
                reservation, previous = await reserve_rtc(owner, body)
                assert previous is pipeline
                if codec:
                    assert len(probes) == 1  # Attachment reuses frozen admission.
                    receipt = pipeline.ready_message()
                    if codec == "h264-default":
                        assert "source_encoding" not in receipt
                        assert pipeline.session.config.source_encoding is None
                        assert pipeline.rtc_source_profile == "42c029"
                    else:
                        assert receipt["source_encoding"] == selected
                        receipt["source_encoding"]["gop"] = 1
                        assert pipeline.session.config.source_encoding == selected
                # Reservation/authorization only; no native peer or ICE claim.
                release_reconnect_preparation(pipeline, reservation)
                runtime.rtc_preparation = None
            else:
                same, sub = await owner.attach(body)
                assert same is pipeline
                attached = msgspec.msgpack.decode(await sub.recv())
                assert attached["type"] == "session.attached"
                assert "resume_token" not in attached
                async with asyncio.timeout(1):
                    while True:
                        message = msgspec.msgpack.decode(await sub.recv())
                        if message.get("request_id") == "original-startup":
                            assert message["type"] == "startup.accepted"
                            break
                pipeline.ready = ready
                assert "resume_token" not in pipeline.ready_message()
            assert (await control.start(op))["execution"] == result["execution"]
            assert len(joins) == 1 and len(owner.sessions) == 1
            if captured:
                assert op["frozenPublication"] == original["frozenPublication"]
                assert (
                    op["frozenLaunchParameters"] == original["frozenLaunchParameters"]
                )
                assert result["parametersSha256"] == original["parametersSha256"]
                for request in captured["requests"][1:]:
                    method = (
                        control.get_operation
                        if request["action"] == "getOperation"
                        else control.get_attachment
                    )
                    reply = await method(request["singlePlayerOperation"])
                    assert reply["execution"] == result["execution"]
                    assert reply["parametersSha256"] == original["parametersSha256"]
                assert ("playConfiguration" in op["frozenPublication"]) is not captured[
                    "legacy"
                ]
            await control.close(op)
            assert engine.closed and not owner.sessions
        finally:
            await owner.close()
            await runtime.close()
            await archives.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure", ["probe", "websocket", "invalid_fps", "attach", "cancel"]
)
def test_managed_encoding_rejection_or_interruption_releases_preparation(
    tmp_path, monkeypatch, failure
):
    from test_whitebox_session_encoding import settings

    from sglang.multimodal_gen.runtime.realtime.whitebox import session_encoding
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec

    async def run():
        owner, control = host(tmp_path, monkeypatch)
        owner.config.spec = VideoSpec(832, 480)
        op = operation()
        op["frozenLaunchParameters"]["transport"] = (
            "websocket" if failure == "websocket" else "webrtc-rtp"
        )
        init = msgspec.msgpack.decode(
            base64.b64decode(op["frozenLaunchParameters"]["initMsgpackBase64"])
        )
        init.update(source_operation="create.manual", inference_control="manual")
        init["whitebox"]["source_encoding"] = settings("h264")
        if failure == "invalid_fps":
            init["whitebox"].update(resolution="720p", cli_fps=61)
        op["frozenLaunchParameters"]["initMsgpackBase64"] = base64.b64encode(
            msgspec.msgpack.encode(init)
        ).decode()
        probes, attaches = [], []

        def preflight(*args, **kwargs):
            assert not owner.sessions and not owner.runtime.rooms
            assert not owner.lock.locked()
            probes.append(args)
            if failure == "probe":
                raise ValueError("Fixture encoder unavailable")
            return "42c029"

        original_attach = owner.attach

        async def attach(*args, **kwargs):
            attaches.append(kwargs)
            if failure in {"attach", "cancel"}:
                assert owner.runtime.rtc_preparation == kwargs["preparation"]
                assert kwargs["preparation"] is not None
                if failure == "cancel":
                    raise asyncio.CancelledError()
                raise RuntimeError("Fixture attach failed")
            return await original_attach(*args, **kwargs)

        monkeypatch.setattr(owner, "attach", attach)
        monkeypatch.setattr(session_encoding, "preflight_source_encoding", preflight)
        try:
            if failure == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await control.start(op)
            else:
                assert (await control.start(op))["state"] == "failed"
            assert not owner.sessions and not owner.runtime.rooms
            assert not Pipeline.created
            assert getattr(owner.runtime, "rtc_preparation", None) is None
            assert len(probes) == (failure in {"probe", "attach", "cancel"})
            assert len(attaches) == (failure in {"websocket", "attach", "cancel"})
            # Failed admission is a durable outcome; retry does not launch again.
            assert (await control.start(op))["state"] == "failed"
        finally:
            await owner.close()

    asyncio.run(run())


def test_reservation_shared_by_two_controllers_never_starts_twice(
    tmp_path, monkeypatch
):
    async def run():
        owner, control = host(tmp_path, monkeypatch)
        another = SinglePlayerControl(owner)
        op = operation()
        await asyncio.gather(control.start(op), another.start(op))
        assert (await another.get_operation(op))["state"] == "ready"
        assert len(Pipeline.created) == 1
        await another.close(op)
        await owner.close()

    asyncio.run(run())


def test_expired_directory_game_never_reassigns_original_operation():
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        MemoryRoomStore,
        RoomDirectory,
    )

    async def run():
        store = MemoryRoomStore()
        directory = RoomDirectory(store, {"owner": "http://owner"})
        op = operation()
        op.pop("_route")
        body = dict(schemaVersion=1, action="start", singlePlayerOperation=op)
        first = await directory.control_route(op["caller"], body, create=True)
        del store.values[first["game_id"]]  # Original route's ordinary TTL elapsed.
        late = await directory.control_route(op["caller"], body, create=True)
        assert late["game_id"] == first["game_id"]
        assert late["owner_id"] == first["owner_id"]
        assert late["status"] == "expired"
        assert first["game_id"] not in store.values

    asyncio.run(run())


def test_owner_route_and_expired_journal_cannot_recreate_source(tmp_path, monkeypatch):
    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        op["_route"]["owner_epoch"] = new_id()
        assert (await control.start(op))["state"] == "lost"
        assert not Pipeline.created
        fresh = operation()
        fresh["operationId"] = "expiring"
        await control.start(fresh)
        await control.close(fresh)
        owner.store.clock = lambda: 10**12
        assert (await control.start(fresh))["state"] == "expired"
        assert len(Pipeline.created) == 1
        await owner.close()

    asyncio.run(run())


@pytest.mark.parametrize("field", ["playConfiguration", "whitebox"])
def test_malformed_nested_launch_is_rejected_before_reservation(
    tmp_path, monkeypatch, field
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.player_control import (
        validate_operation,
    )

    op = operation()
    if field == "playConfiguration":
        op["frozenPublication"][field] = []
    else:
        init = msgspec.msgpack.decode(
            base64.b64decode(op["frozenLaunchParameters"]["initMsgpackBase64"])
        )
        init[field] = []
        op["frozenLaunchParameters"]["initMsgpackBase64"] = base64.b64encode(
            msgspec.msgpack.encode(init)
        ).decode()
    with pytest.raises(ValueError):
        validate_operation(op)


def test_second_launch_preserves_original_single_active_capacity(tmp_path, monkeypatch):
    async def run():
        owner, control = host(tmp_path, monkeypatch)
        first = operation()
        await control.start(first)
        second = operation()
        second["operationId"] = "second"
        refused = await control.start(second)
        assert refused["state"] == "failed"
        assert refused["failureCode"] == "CAPACITY_EXHAUSTED"
        assert (await control.start(second))["state"] == "failed"
        assert (await control.get_operation(first))["state"] == "ready"
        assert len(Pipeline.created) == 1 and not Pipeline.created[0].closed
        await control.close(first)
        await owner.close()

    asyncio.run(run())


def test_cli_startup_error_is_exposed_without_leaking_unknown_codes(
    tmp_path, monkeypatch
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import (
        WhiteboxDiagnostic,
    )

    async def run():
        owner, control = host(tmp_path, monkeypatch)

        async def fail(*args, **kwargs):
            raise WhiteboxDiagnostic("STREAM_DELIVERY_UI_REQUIRED")

        monkeypatch.setattr(owner, "attach", fail)
        op = operation()
        await control.start(op)
        observed = await control.get_operation(op)
        assert observed["state"] == "failed"
        assert observed["failureCode"] == "STREAM_DELIVERY_UI_REQUIRED"
        await owner.close()

    asyncio.run(run())


def test_idle_reason_survives_source_retirement_but_unknown_codes_stay_private(
    tmp_path, monkeypatch
):
    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        await control.start(op)
        key, _ = await control._read(op)
        await control.source_failed(key, "PLAYER_IDLE_TIMEOUT")
        assert (await control.get_operation(op))["failureCode"] == "PLAYER_IDLE_TIMEOUT"
        await control.source_failed(key, "private-error")
        assert "failureCode" not in await control.get_operation(op)
        await control.close(op)
        await owner.close()

    asyncio.run(run())


def test_actual_source_close_persists_idle_reason_after_cleanup(tmp_path, monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        RoomPipeline,
    )

    actual_pipeline = render_host.RenderPipeline

    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        await control.start(op)
        key, row = await control._read(op)
        original = owner.sessions[row["execution"]["sourceSessionId"]]
        pipeline = actual_pipeline.__new__(actual_pipeline)
        pipeline.__dict__.update(original.__dict__)
        pipeline.single_player_operation = key
        pipeline.terminal = dict(type="error", code="PLAYER_IDLE_TIMEOUT")
        owner.sessions[pipeline.session_id] = pipeline
        entered, release = asyncio.Event(), asyncio.Event()

        async def close_source(self, reason=None):
            owner.sessions.pop(self.session_id, None)
            entered.set()
            await release.wait()
            self.closed = True

        monkeypatch.setattr(RoomPipeline, "close", close_source)
        closing = asyncio.create_task(pipeline.close("DISCONNECTED"))
        await entered.wait()
        assert key in control.pending_source_failures
        assert (await control.get_operation(op))["state"] in {"starting", "ready"}
        attachment = await control.get_attachment(op)
        assert attachment["state"] in {"starting", "ready"}
        assert attachment["execution"] == row["execution"]
        assert "attachment" not in attachment
        assert len(Pipeline.created) == 1
        release.set()
        await closing
        result = await control.get_attachment(op)
        assert result["state"] == "failed"
        assert result["failureCode"] == "PLAYER_IDLE_TIMEOUT"
        assert key not in control.pending_source_failures
        await owner.close()

    asyncio.run(run())


def test_frozen_original_binary_startup_survives_factory_unchanged(
    tmp_path, monkeypatch
):
    from test_whitebox_startup import request as startup_request

    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        init = msgspec.msgpack.decode(
            base64.b64decode(op["frozenLaunchParameters"]["initMsgpackBase64"])
        )
        init["startup"] = startup_request()
        init.update(
            seed=42,
            references=[dict(id="original", format="png", payload=b"original-bytes")],
        )
        op["frozenLaunchParameters"]["initMsgpackBase64"] = base64.b64encode(
            msgspec.msgpack.encode(init)
        ).decode()
        await control.start(op)
        actual = copy.deepcopy(Pipeline.created[0].init)
        assert actual["whitebox"].pop("game_id") == op["_route"]["game_id"]
        assert actual == init
        await control.close(op)
        await owner.close()

    asyncio.run(run())


def test_concurrent_close_waits_for_original_room_pipeline_teardown(
    tmp_path, monkeypatch
):
    from collections import deque

    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        RoomPipeline,
    )

    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        await control.start(op)
        pipeline = Pipeline.created[0]
        started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def close_session():
            started.set()
            await release.wait()
            finished.set()

        pipeline.session.close = close_session
        pipeline.task = pipeline.commands = pipeline.retention = None
        pipeline.terminal = None
        pipeline.publish = lambda message: setattr(pipeline, "terminal", message)
        pipeline.history, pipeline.input_history = deque(), {}
        pipeline.report = NS(gateway_done=asyncio.Event(), recording=True)
        pipeline.close = RoomPipeline.close.__get__(pipeline)
        first = asyncio.create_task(control.close(op))
        await started.wait()
        second = await asyncio.wait_for(control.close(op), 1)
        assert second["state"] == "cancelling"
        assert not finished.is_set() and pipeline.session_id in owner.sessions
        assert (await control.get_operation(op))["state"] == "cancelling"
        release.set()
        assert (await first)["state"] == "cancelled"
        assert finished.is_set() and pipeline.session_id not in owner.sessions
        assert (await control.close(op))["state"] == "cancelled"
        await owner.close()

    asyncio.run(run())


@pytest.mark.parametrize("with_model", [False, True])
@pytest.mark.parametrize("fail_preflight", [False, True])
def test_frozen_publication_changed_descriptor_closes_original_source(
    tmp_path, monkeypatch, with_model, fail_preflight
):
    import importlib

    from test_whitebox_interactive import BrowserUI, Engine

    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncArchiveManager,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        MemoryRoomStore,
        RoomDirectory,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig

    cli_owner = importlib.import_module(
        "sglang.multimodal_gen.runtime.realtime.whitebox.cli_room"
    )

    async def checked_metadata(*args, expected_publication, **kwargs):
        if fail_preflight:
            from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import (
                WhiteboxDiagnostic,
            )

            raise WhiteboxDiagnostic("WTP_PLAYABLE_PUBLICATION_CHANGED")
        return 1.0  # Matched A locator; actual driver descriptor is checked next.

    monkeypatch.setattr(cli_owner, "prepare_publication", checked_metadata)
    monkeypatch.setenv(
        "WHITEBOX_CLI_ENTRYPOINT", "/cpu-fixture/sdk/packages/cli/bin/worldkit.mjs"
    )

    async def run():
        op = operation()
        if with_model:
            from test_whitebox_startup import request as startup_request

            monkeypatch.setenv("WHITEBOX_MODEL_STARTUP_PREPARE", "1")
            init = msgspec.msgpack.decode(
                base64.b64decode(op["frozenLaunchParameters"]["initMsgpackBase64"])
            )
            init["startup"] = startup_request()
            op["frozenLaunchParameters"]["initMsgpackBase64"] = base64.b64encode(
                msgspec.msgpack.encode(init)
            ).decode()
        spec = VideoSpec(32, 32)
        engine = Engine(spec)
        engine.renderer = BrowserUI()
        closed, joins = asyncio.Event(), []

        class Driver:
            max_players = 1
            descriptor = dict(
                artifact=dict(
                    sourceId="source",
                    worldId="world",
                    publicationId="replaced-publication",
                )
            )

            async def join(self, **kwargs):
                joins.append(kwargs)
                return engine

            async def close(self):
                closed.set()

        async def factory(**kwargs):
            return Driver()

        directory = RoomDirectory(MemoryRoomStore(), {"owner": "http://owner"})
        op["_route"] = await directory.resolve(
            op["_route"]["game_id"], "source/world", create=True
        )
        archives = AsyncArchiveManager()
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
        )
        owner = render_host.RenderHost(
            runtime, config, archives, Reports(tmp_path, archives), None, None
        )
        model_calls, sends = [], []
        from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
            SourceSession,
        )

        original_send = SourceSession.send

        async def observe_send(source, wire):
            sends.append(msgspec.msgpack.decode(wire)["type"])
            return await original_send(source, wire)

        monkeypatch.setattr(SourceSession, "send", observe_send)

        def model_factory(*args):
            model_calls.append(args)
            raise AssertionError("Mismatched content must not start the original model")

        owner.make_segment_leases = model_factory
        control = SinglePlayerControl(owner)
        try:
            await control.start(op)
            async with asyncio.timeout(3):
                while (result := await control.get_operation(op))[
                    "state"
                ] == "starting":
                    await asyncio.sleep(0.01)
            assert result["state"] == "failed", result
            assert closed.is_set() is (not fail_preflight)
            assert not joins and not model_calls
            assert sends == ([] if fail_preflight else ["init"])
            assert not owner.sessions and not runtime.rooms and not runtime.players
        finally:
            await owner.close()
            await runtime.close()
            await archives.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,missing",
    [(None, False), ("not-ready", False)]
    + [
        (key, missing)
        for key in (
            "publicationId",
            "manifestSha256",
            "sourceManifestSha256",
            "renditionInputSha256",
            "sourceId",
            "worldId",
            "worldBuildHash",
        )
        for missing in (False, True)
    ],
)
def test_original_cli_descriptor_fence_before_ui_or_join(
    tmp_path, monkeypatch, field, missing
):
    import importlib
    import json

    cli = importlib.import_module(
        "sglang.multimodal_gen.runtime.realtime.whitebox.cli_room"
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox import session
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import (
        WhiteboxDiagnostic,
    )

    async def run():
        expected = operation()["frozenLaunchParameters"]["publicationIdentity"]
        artifact = dict(
            expected,
            kind="wtp-playable-rendition",
            worldBuildHash=expected["manifestSha256"],
        )
        if field not in {None, "not-ready"}:
            if missing:
                artifact.pop(field)
            else:
                artifact[field] = "different"
        descriptor = dict(
            kind="worldkit-stream-session",
            schemaVersion=1,
            protocolVersion=1,
            status="ready",
            baseUrl="http://127.0.0.1:12345",
            sessionId="original",
            accessToken="original-private",
            width=32,
            height=32,
            fps=24,
            media=dict(kind="pixels", pixelFormat="rgb24"),
            artifact=artifact,
            worldBuildHash=expected["manifestSha256"],
            streams=dict(
                pixels=dict(url="ws://127.0.0.1:12345/v1/sessions/original/pixels"),
                ui=dict(url="ws://127.0.0.1:12345/v1/sessions/original/control"),
            ),
        )
        process = NS(
            returncode=None,
            stdout=asyncio.StreamReader(),
            stderr=asyncio.StreamReader(),
        )
        if field != "not-ready":
            process.stdout.feed_data(json.dumps(descriptor).encode() + b"\n")
        else:
            process.stderr.feed_data(
                json.dumps(
                    dict(
                        kind="worldkit-cli-error",
                        error=dict(code="WTP_PLAYABLE_NOT_READY"),
                        stage="startup",
                    )
                ).encode()
                + b"\n"
            )
        process.stdout.feed_eof()
        process.stderr.feed_eof()
        calls, stopped, ui = [], [], []

        async def create(*args, **kwargs):
            calls.append(args)
            return process

        async def stop(original):
            stopped.append(original)
            # This fixture substitutes the successful native DELETE, rather
            # than using process exit alone as proof of resident cleanup.
            original.http_delete_confirmed = original.descriptor is not None
            process.returncode = 0

        async def no_reconcile(*args, **kwargs):
            raise AssertionError("A frozen CLI failure must not reconcile or retry")

        monkeypatch.setattr(cli, "prepare_publication", no_reconcile)
        monkeypatch.setattr(cli.asyncio, "create_subprocess_exec", create)
        monkeypatch.setattr(cli.CLIRoom, "_stop_cli", stop)
        monkeypatch.setattr(session, "load_plugin", lambda _: lambda **kwargs: None)
        monkeypatch.setattr(
            cli, "load_local_ui_package", lambda descriptor: ui.append(descriptor) or {}
        )
        entry = tmp_path / "sdk/packages/cli/bin/worldkit.mjs"
        entry.parent.mkdir(parents=True)
        entry.write_text("")
        monkeypatch.setenv("WHITEBOX_CLI_ENTRYPOINT", str(entry))
        monkeypatch.setenv("WHITEBOX_CLI_RESOURCE_TRANSPORT", "cdn")
        monkeypatch.setenv("WHITEBOX_CLI_CACHE_UPDATE", "async")
        if field is not None:
            with pytest.raises(WhiteboxDiagnostic) as failure:
                await cli.cli_room(
                    world_id="source/world",
                    spec=VideoSpec(32, 32),
                    trace_id="original",
                    expected_publication=expected,
                )
            assert failure.value.code == (
                "WTP_PLAYABLE_NOT_READY"
                if field == "not-ready"
                else "WTP_PLAYABLE_PUBLICATION_CHANGED"
            )
            assert stopped and process.returncode == 0 and not ui
        else:
            room = await cli.cli_room(
                world_id="source/world",
                spec=VideoSpec(32, 32),
                trace_id="original",
                expected_publication=expected,
            )
            assert room.descriptor["artifact"]["publicationId"] == "pub" and ui
            await room.close()
        assert "--wtp" in calls[0] and "--delivery" not in calls[0]
        assert calls[0][calls[0].index("--cache-update") + 1] == "sync"

    asyncio.run(run())


@pytest.mark.parametrize(
    "changed",
    [
        None,
        "publicationId",
        "manifestSha256",
        "sourceManifestSha256",
        "renditionInputSha256",
        "missing",
        "not-ready",
    ],
)
def test_frozen_metadata_preflight_is_read_only_and_checks_actual_locator(
    tmp_path, changed
):
    import json
    import os
    import shutil
    import subprocess
    from pathlib import Path

    expected = operation()["frozenLaunchParameters"]["publicationIdentity"]
    location = dict(expected)
    if changed == "missing":
        location.pop("manifestSha256")
    elif changed not in {None, "not-ready"}:
        location[changed] = "replacement"
    entry = tmp_path / "sdk/packages/cli/bin/worldkit.mjs"
    entry.parent.mkdir(parents=True)
    entry.write_text("")
    package = tmp_path / "sdk/node_modules/@worldkit/stream-host"
    package.mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps(
            dict(
                type="module",
                exports={
                    "./wtp-reference": "./reference.mjs",
                    "./wtp-options": "./options.mjs",
                },
            )
        )
    )
    (package / "reference.mjs").write_text(
        'export async function resolveWtpPlayableWorld(){return {sourceId:"source",worldId:"world"};}'
    )
    (package / "options.mjs").write_text(
        'export const DEFAULT_PLATFORM_URL="https://fixture.test"; export const platformOrigin=value=>value;'
    )
    fetch = tmp_path / "fetch.mjs"
    fetch.write_text(
        'globalThis.fetch=async(url,options)=>{if(options.method&&options.method!=="GET")throw Error("FROZEN_MUST_NOT_WRITE");const q=new URL(url).searchParams;if(q.get("publicationId")!=="pub"||!q.has("manifestSha256"))throw Error("FROZEN_SELECTOR_MISSING");return new Response(process.env.FIXTURE_LOCATOR,{status:200});};'
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox import cli_room

    script = Path(cli_room.__file__).with_name("worldkit_publication_prepare.mjs")
    result = subprocess.run(
        [shutil.which("node"), "--import", str(fetch), str(script)],
        input=json.dumps(
            dict(
                worldId="source/world",
                timeoutSeconds=2,
                expectedPublication=expected,
                transport="cdn",
            )
        ),
        text=True,
        capture_output=True,
        timeout=5,
        env=dict(
            os.environ,
            WHITEBOX_CLI_ENTRYPOINT=str(entry),
            WORLD_TEST_API_TOKEN="fixture-token",
            FIXTURE_LOCATOR=json.dumps(
                dict(ready=changed != "not-ready", location=location)
            ),
        ),
    )
    if changed is None:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["publicationIdentity"] == expected
    else:
        assert (
            result.returncode == 1
            and result.stderr.strip() == "WTP_PLAYABLE_PUBLICATION_CHANGED"
        )
        assert not result.stdout


def test_frozen_selector_requires_original_hashes_and_accepts_original_attempt_zero():
    from sglang.multimodal_gen.runtime.realtime.whitebox.player_control import (
        validate_operation,
    )

    op = operation()
    op["frozenPublication"]["world"]["attempt"] = 0
    op["frozenLaunchParameters"]["publicationIdentity"]["attempt"] = 0
    assert validate_operation(op)["type"] == "init"
    for field in (
        "publicationId",
        "sourceManifestSha256",
        "renditionInputSha256",
        "manifestSha256",
    ):
        missing = copy.deepcopy(op)
        del missing["frozenLaunchParameters"]["publicationIdentity"][field]
        with pytest.raises(ValueError):
            validate_operation(missing)
    changed = copy.deepcopy(op)
    changed["frozenPublication"]["world"]["manifestSha256"] = "f" * 64
    with pytest.raises(ValueError, match="changed"):
        validate_operation(changed)


@pytest.mark.parametrize("legacy", [False, True])
def test_trusted_single_configuration_preserves_original_claims(
    tmp_path, monkeypatch, legacy
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.player_control import (
        validate_operation,
    )

    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        if legacy:
            del op["frozenPublication"]["playConfiguration"]
        frozen = copy.deepcopy(op)
        try:
            decoded = validate_operation(op)
            assert decoded == msgspec.msgpack.decode(
                base64.b64decode(op["frozenLaunchParameters"]["initMsgpackBase64"])
            )
            result = await control.start(op)
            assert result["state"] == "ready"
            attachment = await control.get_attachment(op)
            assert attachment["execution"] == result["execution"]
            assert result["parametersSha256"] == op["parametersSha256"]
            assert len(Pipeline.created) == 1 and op == frozen
            assert ("playConfiguration" in op["frozenPublication"]) is not legacy
            changed = copy.deepcopy(op)
            if legacy:
                changed["frozenPublication"]["playConfiguration"] = {
                    "playMode": "single-player"
                }
            else:
                del changed["frozenPublication"]["playConfiguration"]
            with pytest.raises(ValueError, match="changed"):
                await control.get_operation(changed)
            await control.close(op)
        finally:
            await owner.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "configuration",
    [
        None,
        {},
        [],
        "single-player",
        {"playMode": "single"},
        {"playMode": "multiplayer"},
        {"playMode": "unknown"},
    ],
)
def test_explicit_noncanonical_single_configuration_never_allocates(
    tmp_path, monkeypatch, configuration
):
    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        op["frozenPublication"]["playConfiguration"] = configuration
        try:
            with pytest.raises(ValueError, match="frozen single player"):
                await control.start(op)
            assert not Pipeline.created and not owner.sessions
        finally:
            await owner.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,value", [("status", "unverified"), ("scope", "foreign-scope"), ("world", {})]
)
@pytest.mark.parametrize("legacy", [False, True])
def test_single_configuration_never_relaxes_existing_scope_fences(
    tmp_path, monkeypatch, field, value, legacy
):
    async def run():
        owner, control = host(tmp_path, monkeypatch)
        op = operation()
        if legacy:
            del op["frozenPublication"]["playConfiguration"]
        op["frozenPublication"][field] = value
        try:
            with pytest.raises(ValueError):
                await control.start(op)
            assert not Pipeline.created and not owner.sessions
        finally:
            await owner.close()

    asyncio.run(run())


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize(
    "changed", [None, "cleanup", "state", "game", "user", "owner", "epoch", "missing"]
)
def test_retired_source_close_requires_original_durable_release(
    tmp_path, monkeypatch, changed, restart
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderJournal,
    )

    async def run():
        owner, control = host(tmp_path, monkeypatch)
        try:
            op = operation()
            started = await control.start(op)
            execution = started["execution"]
            pipeline = owner.sessions[execution["sourceSessionId"]]
            await pipeline.close()
            journal = RenderJournal(
                owner.store,
                user_id=op["caller"]["subject"],
                source_session_id=execution["sourceSessionId"],
                source_epoch=execution["sourceEpoch"],
                owner_epoch=owner.runtime.epoch,
            )
            if changed != "missing":
                await journal.create()
                updates = dict(
                    game_id=execution["gameId"],
                    source_state="ended",
                    cleanup_state="released",
                    source_outcome="failed",
                )
                if changed == "cleanup":
                    updates["cleanup_state"] = "unknown"
                if changed == "state":
                    updates["source_state"] = "ready"
                if changed == "game":
                    updates["game_id"] = new_id()
                if changed == "user":
                    updates["user_id"] = "another-user"
                if changed == "owner":
                    updates["owner_epoch"] = new_id()
                if changed == "epoch":
                    updates["source_epoch"] = new_id()
                await journal.update(lambda state: state.update(updates))
            if restart:
                original_epoch = owner.runtime.epoch
                await owner.close()
                owner, control = host(tmp_path, monkeypatch)
                assert owner.runtime.epoch != original_epoch
                journal.store = owner.store
            observed = await control.get_operation(op)
            assert observed["state"] == ("failed" if changed is None else "lost")
            if restart:
                for read in (control.start, control.get_attachment):
                    result = await read(op)
                    assert result["state"] == observed["state"]
                    assert result["execution"] == execution
            results = await asyncio.gather(*(control.close(op) for _ in range(3)))
            assert all(
                result["state"] == ("cancelled" if changed is None else "lost")
                for result in results
            )
            _, stored = await control._read(op)
            if changed is None:
                assert stored["sourceOutcome"] == "failed"
                assert stored["sourceCleanupState"] == "released"
                assert stored["execution"] == execution
                assert (await journal.read())["source_outcome"] == "failed"
                if restart:
                    assert stored["ownerEpoch"] == original_epoch
                    assert not Pipeline.created and not owner.sessions
            else:
                assert stored["state"] != "cancelled"
        finally:
            await owner.close()

    asyncio.run(run())
