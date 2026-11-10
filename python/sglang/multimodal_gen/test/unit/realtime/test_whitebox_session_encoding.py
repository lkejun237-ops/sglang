"""Signed source-only selection, default preservation and bounded preflight checks."""

import json
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox import session_encoding as selected
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig


def settings(codec="vp9"):
    value = dict(
        codec=codec,
        bitrate=1000000,
        gop=24,
        vbvMs=250,
        preset="veryfast" if codec == "h264" else "6" if codec == "vp9" else "10",
        quality=18 if codec == "h264" else 56,
        intraRefresh=False,
    )
    if codec != "h264":
        value["minQp"] = 48
    if codec == "av1":
        value["vbvMs"] = 100
    return value


@pytest.mark.parametrize("codec", ["h264", "vp9", "av1"])
def test_signed_settings_are_strict_canonical_and_bound_before_use(codec):
    value = settings(codec)
    world = {"source_encoding": value}
    assert (
        selected.bind_signed_source_encoding({"source_encoding": value}, world) == value
    )
    for claim, request in [
        ({}, world),
        ({"source_encoding": value}, {}),
        ({"source_encoding": {**value, "gop": 12}}, world),
    ]:
        with pytest.raises(ValueError):
            selected.bind_signed_source_encoding(claim, request)
    for bad in [
        None,
        {**value, "gop": True},
        {**value, "gop": 24.0},
        {**value, "bitrate": 1000000.0},
        {**value, "unknown": 0},
    ]:
        with pytest.raises(ValueError):
            selected.bind_signed_source_encoding(
                {"source_encoding": bad}, {"source_encoding": bad}
            )
    malformed = (
        {**value, "minQp": 0}
        if codec == "h264"
        else {k: v for k, v in value.items() if k != "minQp"}
    )
    with pytest.raises(ValueError, match="canonical"):
        selected.bind_signed_source_encoding(
            {"source_encoding": malformed}, {"source_encoding": malformed}
        )


@pytest.mark.parametrize("codec", ["h264", "vp9", "av1"])
def test_signed_selection_preserves_default_and_accepts_supported_codecs(codec):
    assert selected.bind_signed_source_encoding({}, {}) is None
    baseline = SimpleNamespace(source_encoding=None)
    assert selected.session_source_config(baseline, {}) is baseline
    value = settings(codec)
    assert (
        selected.bind_signed_source_encoding(
            {"source_encoding": value}, {"source_encoding": value}
        )
        == value
    )


def test_default_source_never_probes(monkeypatch):
    import asyncio

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import reserve_rtc

    def unexpected(*args):
        raise AssertionError("Default source must not probe")

    monkeypatch.setattr(selected, "preflight_source_encoding", unexpected)

    async def scenario():
        host = SimpleNamespace(
            config=SimpleNamespace(spec=VideoSpec(832, 480)),
            runtime=SimpleNamespace(rooms={}),
            lock=asyncio.Lock(),
            sessions={},
            peer_sessions={},
            max_sessions=1,
        )
        body = dict(
            user_id="user",
            message=dict(
                type="init",
                source_operation="create",
                control_protocol="whitebox-render-v2",
                whitebox={},
            ),
        )
        reservation, previous = await reserve_rtc(host, body)
        assert previous is None and host.runtime.rtc_preparation == reservation
        assert "source_encoding_profile" not in body
        assert not hasattr(host, "source_encoding_probe_lock")

    asyncio.run(scenario())


def test_private_source_config_does_not_change_model_or_archive(tmp_path):
    host = WhiteboxConfig(
        engine_factory=lambda: None,
        ui_factory=lambda: None,
        spool=tmp_path,
        room_owned=True,
        capture_clock=True,
        playback_gop=6,
        source_max_bitrate=800000,
        playback_max_bitrate=4000000,
    )
    candidate = selected.session_source_config(host, {"source_encoding": settings()})
    assert candidate is not host and host.source_encoding is None
    assert selected.source_gop(candidate) == 24 and candidate.playback_gop == 6
    assert (
        selected.source_bitrate(candidate) == 1000000
        and candidate.source_max_bitrate == 800000
    )
    assert candidate.playback_max_bitrate == host.playback_max_bitrate == 4000000
    assert candidate.archive_encoding == host.archive_encoding


def test_resume_keeps_frozen_settings():
    value = settings()
    selected.frozen_source_encoding({}, {"source_encoding": value})
    selected.frozen_source_encoding(
        {"source_encoding": value}, {"source_encoding": value}
    )
    with pytest.raises(ValueError, match="frozen"):
        selected.frozen_source_encoding(
            {"source_encoding": {**value, "gop": 12}}, {"source_encoding": value}
        )
    with pytest.raises(ValueError):
        selected.frozen_source_encoding({"source_encoding": value}, {})


@pytest.mark.parametrize(
    "transport,patch,existing",
    [
        ("websocket", {}, False),
        ("webrtc-rtp", {"type": "resume"}, False),
        ("webrtc-rtp", {"source_operation": "join"}, False),
        (
            "webrtc-rtp",
            {"whitebox": {"source_encoding": settings(), "loading": "areas"}},
            False,
        ),
        ("webrtc-rtp", {}, True),
    ],
)
def test_only_new_native_non_area_source_accepts_selection(transport, patch, existing):
    message = dict(
        type="init",
        source_operation="create",
        control_protocol="whitebox-render-v2",
        whitebox={"source_encoding": settings()},
    )
    message.update(patch)
    with pytest.raises(ValueError):
        selected.check_source_encoding_scope(
            message, transport=transport, existing_room=existing
        )


def test_preflight_checks_new_native_capability_before_codec_probe(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=b'{"sourceCodecs":["vp9"],"modelCodec":"h264"}')

    monkeypatch.setattr(selected.subprocess, "run", run)
    with pytest.raises(ValueError, match="native sender"):
        selected.preflight_source_encoding(settings(), 30, "/test/binary")
    assert calls == [["/test/binary", "--capabilities"]]


@pytest.mark.parametrize("width,height,fps", [(832, 480, 30), (1280, 720, 30)])
@pytest.mark.parametrize("codec", ["h264", "vp9", "av1"])
def test_actual_selected_encoder_preflight_decodes_three_aus(codec, width, height, fps):
    import av

    from sglang.multimodal_gen.runtime.realtime.whitebox.source_encoding import (
        encoder_implementation,
    )

    if encoder_implementation(codec) not in av.codecs_available:
        pytest.skip("Software encoder unavailable")
    report = selected._encoder_probe(settings(codec), fps, width, height)
    assert (
        report["profileLevelId"]
        if codec == "h264"
        else report["profileLevelId"] is None
    )


def test_shared_wtp_contract_fixture_is_normalized_before_signing():
    from pathlib import Path

    from sglang.multimodal_gen.runtime.realtime.whitebox.source_encoding import (
        validate_source_encoding,
    )

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/source-encoding-contract.json").read_text()
    )
    for case in fixture["valid"]:
        assert validate_source_encoding(case["input"]) == case["canonical"], case[
            "name"
        ]
        assert (
            selected.canonical_source_encoding(case["canonical"]) == case["canonical"]
        )
        if not case["inputIsCanonical"]:
            with pytest.raises(ValueError):
                selected.canonical_source_encoding(case["input"])
    for case in fixture["invalid"]:
        with pytest.raises(ValueError):
            selected.canonical_source_encoding(case["input"])


def test_upstream_manual_source_create_is_preserved():
    value = settings()
    assert (
        selected.check_source_encoding_scope(
            dict(
                type="init",
                source_operation="create.manual",
                inference_control="manual",
                control_protocol="whitebox-render-v2",
                whitebox={"source_encoding": value},
            ),
            transport="webrtc-rtp",
        )
        == value
    )


@pytest.mark.parametrize("width,height,fps", [(832, 480, 60), (1280, 720, 30)])
def test_real_binary_and_software_preflight(width, height, fps):
    import os
    from pathlib import Path

    binary = Path(
        os.environ.get(
            "WHITEBOX_TEST_RTC_BINARY",
            ".codex-tmp/production-session-source-encoding/whitebox-rtc",
        )
    ).resolve()
    if not binary.exists():
        pytest.skip("Build the exact native sender to run binary preflight")
    for codec in ["h264", "vp9", "av1"]:
        profile = selected.preflight_source_encoding(
            settings(codec),
            fps if codec == "h264" else 30,
            str(binary),
            width=width,
            height=height,
        )
        assert profile if codec == "h264" else profile is None


@pytest.mark.parametrize("codec", ["h264", "vp9", "av1"])
def test_real_ticket_tamper_fails_before_nonce_consumption(codec):
    import base64
    import hashlib
    import hmac

    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_auth import RTCCapabilities

    value = settings(codec)
    claims = dict(
        v=1,
        aud="whitebox-rtc",
        sub="user",
        origin="https://wtp.test",
        world_id="world",
        source_id="source",
        iat=100,
        exp=200,
        jti="1" * 32,
        source_encoding=value,
    )
    b64 = lambda b: base64.urlsafe_b64encode(b).decode().rstrip("=")
    payload = b64(json.dumps(claims).encode())
    token = (
        payload
        + "."
        + b64(hmac.new(b"s" * 32, payload.encode(), hashlib.sha256).digest())
    )
    guard = RTCCapabilities("s" * 32, now=lambda: 110)
    world = dict(world_id="world", source_id="source", source_encoding=value)
    for bad in [
        {k: v for k, v in world.items() if k != "source_encoding"},
        {**world, "source_encoding": {**value, "gop": 12}},
    ]:
        with pytest.raises(ValueError, match="authorization"):
            guard.verify(token, origin=claims["origin"], world=bad)
        assert not guard.used
    assert (
        guard.verify(token, origin=claims["origin"], world=world)["source_encoding"]
        == value
    )
    with pytest.raises(ValueError):
        guard.verify(token, origin=claims["origin"], world=world)


def test_direct_legacy_pipeline_cannot_apply_unsigned_source_settings():
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        RoomPipeline,
    )

    host = SimpleNamespace(runtime=SimpleNamespace(rooms={}))
    with pytest.raises(ValueError, match="native RTC"):
        RoomPipeline(
            host,
            dict(
                type="init",
                source_operation="create",
                control_protocol="whitebox-render-v2",
                whitebox=dict(source_encoding=settings()),
            ),
            "user",
            "trace",
        )


@pytest.mark.parametrize("codec", ["h264", "vp9", "av1"])
@pytest.mark.parametrize("resolution", ["480p", "720p"])
def test_rtc_preflight_is_off_loop_and_before_reservation(
    monkeypatch, codec, resolution
):
    import asyncio
    import threading

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import reserve_rtc

    async def scenario():
        host = SimpleNamespace(
            config=SimpleNamespace(spec=VideoSpec(832, 480)),
            runtime=SimpleNamespace(rooms={}),
            lock=asyncio.Lock(),
            sessions={},
            peer_sessions={},
            max_sessions=1,
        )
        body = dict(
            user_id="user",
            message=dict(
                type="init",
                source_operation="create",
                control_protocol="whitebox-render-v2",
                whitebox=dict(
                    source_encoding=settings(codec), cli_fps=30, resolution=resolution
                ),
            ),
        )
        loop_thread = threading.get_ident()
        calls = []

        def probe(value, fps, *, width, height):
            assert threading.get_ident() != loop_thread
            assert not host.lock.locked() and not getattr(
                host.runtime, "rtc_preparation", None
            )
            calls.append((value, fps, width, height))
            return "42c01f" if codec == "h264" else None

        monkeypatch.setattr(selected, "preflight_source_encoding", probe)
        reservation, previous = await reserve_rtc(host, body)
        assert previous is None and host.runtime.rtc_preparation == reservation
        assert calls == [
            (
                settings(codec),
                30 if resolution == "720p" else 60 if codec == "h264" else 30,
                1280 if resolution == "720p" else 832,
                720 if resolution == "720p" else 480,
            )
        ]
        assert body["source_encoding_profile"] == (
            "42c01f" if codec == "h264" else None
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "patch",
    [
        {"source_encoding": {**settings(), "gop": 0}},
        {"loading": "areas"},
        {"source_encoding": None},
    ],
)
def test_bad_selection_never_probes_or_reserves(monkeypatch, patch):
    import asyncio

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import reserve_rtc

    def unexpected(*args):
        raise AssertionError("Invalid source must not probe")

    monkeypatch.setattr(selected, "preflight_source_encoding", unexpected)

    async def scenario():
        host = SimpleNamespace(
            config=SimpleNamespace(spec=VideoSpec(832, 480)),
            runtime=SimpleNamespace(rooms={}),
            lock=asyncio.Lock(),
            sessions={},
            peer_sessions={},
            max_sessions=1,
        )
        body = dict(
            user_id="user",
            message=dict(
                type="init",
                source_operation="create",
                control_protocol="whitebox-render-v2",
                whitebox={"source_encoding": settings(), **patch},
            ),
        )
        from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import (
            WhiteboxDiagnostic,
        )

        with pytest.raises(WhiteboxDiagnostic):
            await reserve_rtc(host, body)
        assert not getattr(host.runtime, "rtc_preparation", None)

    asyncio.run(scenario())


def test_cancellation_does_not_release_probe_bound_early(monkeypatch):
    import asyncio
    import threading

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import reserve_rtc

    async def scenario():
        entered, release = threading.Event(), threading.Event()
        host = SimpleNamespace(
            config=SimpleNamespace(spec=VideoSpec(832, 480)),
            runtime=SimpleNamespace(rooms={}),
            lock=asyncio.Lock(),
            sessions={},
            peer_sessions={},
            max_sessions=1,
        )
        body = dict(
            user_id="user",
            message=dict(
                type="init",
                source_operation="create",
                control_protocol="whitebox-render-v2",
                whitebox={"source_encoding": settings()},
            ),
        )

        def probe(*args, **kwargs):
            entered.set()
            assert release.wait(3)

        monkeypatch.setattr(selected, "preflight_source_encoding", probe)
        task = asyncio.create_task(reserve_rtc(host, body))
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        assert host.source_encoding_probe_lock.locked()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not host.source_encoding_probe_lock.locked()
        assert not getattr(host.runtime, "rtc_preparation", None)

    asyncio.run(scenario())


def test_native_default_argv_is_unchanged_and_override_is_per_process(monkeypatch):
    import asyncio

    from sglang.multimodal_gen.runtime.realtime.whitebox.native_rtc import NativeRTC

    calls = []

    async def spawn(*args, **kwargs):
        calls.append(args)
        raise OSError("Test stops before starting a process")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    async def scenario():
        base = dict(
            executable="/native",
            listen="127.0.0.1:1",
            public_ip="127.0.0.1",
            public_port=2,
        )
        for options in [
            {},
            {"source_codec": "vp9"},
            {"source_codec": "h264", "source_h264_profile": "42c029"},
        ]:
            peer = NativeRTC(lambda _: None, **base, **options)
            with pytest.raises(OSError):
                await peer.start()
        expected = (
            "/native",
            "--listen",
            "127.0.0.1:1",
            "--public-ip",
            "127.0.0.1",
            "--public-port",
            "2",
        )
        assert calls == [
            expected,
            expected + ("--source-codec", "vp9"),
            expected + ("--source-codec", "h264", "--source-h264-profile", "42c029"),
        ]

    asyncio.run(scenario())


def test_ready_receipt_is_server_owned_and_omitted_on_default(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderPipeline,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        RoomPipeline,
    )

    monkeypatch.setattr(
        RoomPipeline,
        "ready_message",
        lambda _: dict(source_video_stream={"format": "vp9"}),
    )
    p = object.__new__(RenderPipeline)
    p.session = SimpleNamespace(
        config=SimpleNamespace(source_encoding=settings()),
        journal=SimpleNamespace(snapshot={}),
        source_epoch="s",
        player_remaining_seconds=30,
        model_context_segment=lambda _: None,
        current_model_stream=None,
    )
    p.init = {"whitebox": {"source_encoding": settings()}}
    receipt = p.ready_message()
    assert receipt["source_encoding"] == settings()
    receipt["source_encoding"]["gop"] = 1
    assert p.session.config.source_encoding["gop"] == 24
    p.init = {"whitebox": {}}
    assert "source_encoding" not in p.ready_message()


@pytest.mark.parametrize("echo", [False, True])
def test_early_h264_resume_uses_frozen_profile_before_encoder_ready(monkeypatch, echo):
    import asyncio

    import msgspec
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderPipeline,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import (
        RTCRenderPipeline,
        install_rtc_routes,
    )

    def unexpected(*args):
        raise AssertionError("Resume must not re-probe today's environment")

    monkeypatch.setattr(selected, "preflight_source_encoding", unexpected)
    monkeypatch.setattr(RenderPipeline, "__init__", lambda *args, **kwargs: None)
    previous = RTCRenderPipeline(peer=SimpleNamespace(source_h264_profile="42c029"))
    previous.init = {"whitebox": {"source_encoding": settings("h264")}}
    previous.closed = False
    previous.closing = False
    previous.terminal = None
    previous.user_id = "user"
    previous.game_id = "b" * 32
    previous.session_id = "a" * 32
    previous.session = SimpleNamespace(
        source_epoch="source",
        source_playback=None,
        config=SimpleNamespace(
            spec=VideoSpec(832, 480), source_encoding=settings("h264")
        ),
        capture_rate_hint=lambda: 60,
    )
    previous.token = "resume-secret"
    previous.history_floor = previous.control_sequence = 0
    previous.subscription = None
    previous.rtc_peer = None

    async def close(reason):
        previous.rtc_closed.set()

    previous.close = close
    calls = []

    class Peer:
        def __init__(self, callback, **options):
            calls.append(options)
            self.connected = asyncio.Event()

        async def start(self):
            pass

        async def close(self):
            pass

    async def attach(*args, **kwargs):
        raise AssertionError("Cancelled before ICE")

    host = SimpleNamespace(
        config=SimpleNamespace(source_encoding=None),
        runtime=SimpleNamespace(epoch="owner"),
        lock=asyncio.Lock(),
        sessions={previous.session_id: previous},
        attach=attach,
    )
    app = FastAPI()
    install_rtc_routes(host, app, "internal-only", peer_factory=Peer)
    world = {"game_id": previous.game_id}
    if echo:
        world["source_encoding"] = settings("h264")
    with TestClient(app) as client:
        with client.websocket_connect(
            "/internal/render-rtc", headers={"authorization": "Bearer internal-only"}
        ) as ws:
            ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        user_id="user",
                        message=dict(
                            type="resume",
                            control_protocol="whitebox-render-v2",
                            session_id=previous.session_id,
                            whitebox=world,
                            owner_epoch="owner",
                            source_epoch="source",
                            resume_token=previous.token,
                        ),
                    )
                )
            )
            configuration = msgspec.msgpack.decode(ws.receive_bytes())
            assert configuration["type"] == "rtc.configuration", configuration
            assert configuration["video_codecs"] == {
                "whitebox": "h264",
                "model": "h264",
            }
            assert calls == [{"source_codec": "h264", "source_h264_profile": "42c029"}]
            assert previous.session.source_playback is None
            ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(type="rtc.cancel", peer_id=configuration["peer_id"])
                )
            )
            assert msgspec.msgpack.decode(ws.receive_bytes())["type"] == "rtc.cancelled"
    assert previous.rtc_closed.is_set() and host.runtime.rtc_preparation is None


def test_rejected_override_does_not_end_an_existing_room_directory_lease(monkeypatch):
    import asyncio

    import msgspec
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import (
        install_rtc_routes,
    )

    events = []

    async def touch(*args, **kwargs):
        events.append("directory modified")

    def peer(*args, **kwargs):
        raise AssertionError("Rejected before native allocation")

    host = SimpleNamespace(
        config=SimpleNamespace(spec=VideoSpec(832, 480)),
        runtime=SimpleNamespace(
            rooms={"b" * 32: {}},
            owner_id="room",
            epoch="owner",
            directory=SimpleNamespace(touch=touch),
        ),
        lock=asyncio.Lock(),
        sessions={},
        peer_sessions={},
        max_sessions=1,
    )
    app = FastAPI()
    install_rtc_routes(host, app, "internal-only", peer_factory=peer)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/internal/render-rtc", headers={"authorization": "Bearer internal-only"}
        ) as ws:
            ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        user_id="user",
                        message=dict(
                            type="init",
                            source_operation="create",
                            control_protocol="whitebox-render-v2",
                            whitebox=dict(game_id="b" * 32, source_encoding=settings()),
                        ),
                    )
                )
            )
            response = msgspec.msgpack.decode(ws.receive_bytes())
            assert (
                response["type"] == "error"
                and response["code"] == "RTC_CODEC_UNSUPPORTED"
            )
    assert not events and not getattr(host.runtime, "rtc_preparation", None)


def test_live_feedback_reads_session_source_ceiling_and_preserves_host(monkeypatch):
    import asyncio

    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_transport import (
        RTCSubscription,
    )

    async def scenario():
        calls = []
        config = SimpleNamespace(
            source_encoding=settings(),
            source_max_bitrate=4_000_000,
            playback_max_bitrate=2_500_000,
        )
        host_config = SimpleNamespace(
            source_max_bitrate=8_000_000, playback_max_bitrate=9_000_000
        )
        source = SimpleNamespace(max_bitrate=1_000_000)
        sub = object.__new__(RTCSubscription)

        def update(rate):
            calls.append(rate)
            sub.closed = True
            return True

        source.set_target_bitrate = update
        sub.closed = False
        sub.peer = SimpleNamespace(delivery_stats_version=0)
        sub.owner = SimpleNamespace(
            session=SimpleNamespace(
                config=config, source_playback=source, current_segment=None
            ),
            host=SimpleNamespace(config=host_config),
        )
        sub.target_bitrate = 4_200_000
        sub.applied_bitrates = {}
        sub._record_delivery = lambda: None
        await asyncio.wait_for(sub._feedback(), 2)
        assert calls == [1_000_000]
        assert config.source_max_bitrate == 4_000_000
        assert host_config.source_max_bitrate == 8_000_000
        assert config.playback_max_bitrate == 2_500_000

    asyncio.run(scenario())


@pytest.mark.parametrize("fps", [36, 60])
@pytest.mark.parametrize("codec", ["h264", "vp9", "av1"])
def test_720p_rate_over_transport_envelope_fails_before_probe(monkeypatch, codec, fps):
    import asyncio

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import (
        WhiteboxDiagnostic,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import reserve_rtc

    def unexpected(*args, **kwargs):
        raise AssertionError("Unsupported rate must not probe")

    monkeypatch.setattr(selected, "preflight_source_encoding", unexpected)

    async def scenario():
        host = SimpleNamespace(
            config=SimpleNamespace(spec=VideoSpec(832, 480)),
            runtime=SimpleNamespace(rooms={}),
        )
        body = dict(
            user_id="user",
            message=dict(
                type="init",
                source_operation="create",
                control_protocol="whitebox-render-v2",
                whitebox=dict(
                    resolution="720p", source_encoding=settings(codec), cli_fps=fps
                ),
            ),
        )
        with pytest.raises(WhiteboxDiagnostic):
            await reserve_rtc(host, body)
        assert not getattr(host.runtime, "rtc_preparation", None)

    asyncio.run(scenario())


def test_preflight_subprocess_uses_each_requested_geometry_and_rate(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[-1] == "--capabilities":
            return SimpleNamespace(
                stdout=json.dumps(
                    dict(
                        protocol="native-video-rtp-v1",
                        sessionSourceEncoding=True,
                        sourceCodecs=["h264"],
                        modelCodec="h264",
                    )
                )
            )
        return SimpleNamespace(stdout='{"profileLevelId":"42c01f"}')

    monkeypatch.setattr(selected.subprocess, "run", run)
    for width, height, fps in [(832, 480, 60), (1280, 720, 30), (1280, 720, 24)]:
        assert (
            selected.preflight_source_encoding(
                settings("h264"), fps, "/test/binary", width=width, height=height
            )
            == "42c01f"
        )
        assert calls[-1][-3:] == [str(fps), str(width), str(height)]
    assert len(calls) == 6
