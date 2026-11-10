# SPDX-License-Identifier: Apache-2.0
"""Real source-sized RGB, H.264 and startup identity without a model/GPU."""

import asyncio
import hashlib
import io
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

import av
import msgspec
import pytest
from PIL import Image
from test_whitebox_interactive import BrowserUI, Engine, Worker, receive_until
from test_whitebox_startup import Lease, request

from sglang.multimodal_gen.runtime.realtime.source_prepare import (
    PROTOCOL,
    V2_PROTOCOL,
    condition_fingerprint,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
    InteractiveSession,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.model_geometry import (
    ModelGeometry,
    session_source_spec,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.recording import ArchiveManager
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
    new_id,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.session import (
    WhiteboxConfig,
    WhiteboxUpstream,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.startup import PlayStartup


@pytest.mark.parametrize("protocol", [PROTOCOL, V2_PROTOCOL])
@pytest.mark.parametrize("with_preview", [False, True])
def test_720_startup_keeps_cli_proof_and_private_prepare_fingerprint(
    protocol, with_preview
):
    async def run():
        spec = VideoSpec(1280, 720)
        geometry = ModelGeometry(spec)
        source = b"s" * spec.frame_bytes
        lease = Lease(protocol)
        owner = NS(
            config=NS(spec=spec),
            outgoing=asyncio.Queue(),
            user_id="user",
            session_id=new_id(),
            source_epoch=new_id(),
            host=NS(make_segment_leases=lambda *_: lease),
        )
        value = request(False, protocol=protocol)
        preview = source
        width, height = 1280, 720
        if with_preview:
            width, height = (16, 48) if protocol == V2_PROTOCOL else (1280, 720)
            preview = b"p" * (width * height * 3) if protocol == V2_PROTOCOL else source
            encoded = io.BytesIO()
            Image.frombytes("RGB", (width, height), preview).save(encoded, format="PNG")
            raw = encoded.getvalue()
            value["initial_frame"] = dict(
                format="png",
                width=width,
                height=height,
                payload=raw,
                encoded_size=len(raw),
                encoded_sha256=hashlib.sha256(raw).hexdigest(),
                rgb_sha256=hashlib.sha256(preview).hexdigest(),
            )
        startup = PlayStartup(owner, value)
        public_init = dict(
            type="init",
            size="1280x720",
            source_stream=dict(format="rgb24", width=1280, height=720, fps=24),
        )
        startup.launch(public_init)
        try:
            await startup.wait(startup.admitted)
            await startup.bind_cli(source, dict(protocol="room-source-frame-v1"))
            sent_init, prepare = lease.ws.commands
            prepared_rgb = prepare["payload"]["rgb"]
            assert sent_init["source_stream"]["height"] == 736
            assert sent_init["size"] == "1280x736"
            assert public_init["size"] == startup.init["size"] == "1280x720"
            assert startup.rgb == preview and startup.cli_rgb is source
            assert prepared_rgb == (
                geometry.pad_rgb(source) if protocol == PROTOCOL else preview
            )
            assert startup.identity["condition_fingerprint"] == condition_fingerprint(
                rgb=prepared_rgb,
                prompt="snow",
                references=[],
                seed=42,
                source_stream=sent_init["source_stream"],
                profile_sha256="a" * 64,
                protocol=protocol,
                preview_width=width,
                preview_height=height,
            )
            receipt = startup.receipt()
            assert (
                receipt["effective_rgb_sha256"] == hashlib.sha256(preview).hexdigest()
            )
            assert receipt["cli_frame_rgb_sha256"] == hashlib.sha256(source).hexdigest()
            assert (
                receipt["model_cli_frame_rgb_sha256"]
                == hashlib.sha256(geometry.pad_rgb(source)).hexdigest()
            )
            assert (
                receipt["model_condition_rgb_sha256"]
                == hashlib.sha256(prepared_rgb).hexdigest()
            )
            startup.rtc_ready.set()
            async with startup.open():
                await startup.attach()
            assert lease.opens == lease.closes == 1
        finally:
            await startup.close()
        assert startup.resource_state == "released"

    asyncio.run(run())


@pytest.mark.parametrize("resolution", ["480p", "720p"])
def test_model_boundary_and_actual_recorded_h264_keep_public_resolution(
    tmp_path, resolution
):
    async def run():
        spec = VideoSpec(832, 480) if resolution == "480p" else VideoSpec(1280, 720)
        model = ModelGeometry(spec).model
        engine, archives = Engine(spec), ArchiveManager(processes=False)

        class PatternWorker(Worker):
            async def emit(self, message):
                if message["type"] == "frame_batch":
                    # A visible bottom band must never leak into native output.
                    message["payload"] = b"d" * spec.frame_bytes + b"z" * (
                        model.frame_bytes - spec.frame_bytes
                    )
                await super().emit(message)

        worker = PatternWorker(model)

        @asynccontextmanager
        async def lease():
            try:
                yield (
                    worker,
                    NS(
                        session_id="model-session",
                        generation_id="generation",
                        denoiser=NS(worker_id="worker", model_revision="model"),
                    ),
                )
            finally:
                await worker.close()

        async def factory(**kwargs):
            assert kwargs["spec"] == spec
            return engine

        session = InteractiveSession(
            leases=NS(open=lease),
            config=WhiteboxConfig(
                engine_factory=factory,
                ui_factory=lambda **_: BrowserUI(),
                spool=tmp_path,
                startup_seconds=5,
                play_seconds=20,
                drain_seconds=5,
            ),
            archives=archives,
            session_id=new_id(),
            trace_id="resolution-test",
        )
        try:
            await session.send(
                msgspec.msgpack.encode(
                    dict(
                        type="init",
                        whitebox=dict(world_id="world", resolution=resolution),
                        prompt="test",
                    )
                )
            )
            ready = await receive_until(session, lambda m: m["type"] == "session.ready")
            assert ready["source_stream"] == dict(
                format="rgb24", width=spec.width, height=spec.height, fps=24
            )
            await receive_until(session, lambda m: m.get("state") == "active")
            await engine.push()
            frame = await receive_until(session, lambda m: m["type"] == "frame_batch")
            assert (frame["width"], frame["height"]) == (spec.width, spec.height)
            decoder = av.CodecContext.create("h264", "r")
            decoded = decoder.decode(av.Packet(frame["payload"])) + decoder.decode(None)
            assert decoded and all(
                (f.width, f.height) == (spec.width, spec.height) for f in decoded
            )
            assert abs(float(decoded[0].to_ndarray(format="rgb24").mean()) - 100) < 4
            append = next(
                e["payload"] for e in worker.events if e["kind"] == "source.append"
            )
            assert append["data"] == bytes([30]) * model.frame_bytes
            assert worker.init["size"] == f"{model.width}x{model.height}"
            assert session.upstream.init["size"] == f"{spec.width}x{spec.height}"
            await session.finish()
            await receive_until(session, lambda m: m["type"] == "session.completed")
            directory = session.recording.directory
        finally:
            await session.close()
            await archives.close()
        assert engine.closed and worker.closed
        for name in ("whitebox.mp4", "model.mp4"):
            with av.open(str(directory / name)) as video:
                stream = video.streams.video[0]
                assert (stream.width, stream.height) == (spec.width, spec.height)
                assert list(video.decode(video=0))

    asyncio.run(run())


def test_concurrent_rooms_pin_independent_specs_and_cleanup(tmp_path):
    async def run():
        directory = RoomDirectory(MemoryRoomStore(), {"room": "http://room"})
        drivers = []

        class Renderer:
            metadata = {"mode": "fixture"}

            async def render(self, rgb, _frame, spec):
                assert len(rgb) == spec.frame_bytes
                return rgb

            async def close(self):
                pass

        class Driver:
            max_players = 2

            def __init__(self, spec):
                self.spec, self.engine, self.closed = spec, Engine(spec), False
                self.engine.renderer = Renderer()

            async def join(self, *, spec, **_):
                assert spec == self.spec
                return self.engine

            async def close(self):
                self.closed = True

        async def factory(*, spec, **_):
            driver = Driver(spec)
            drivers.append(driver)
            return driver

        archives = ArchiveManager(processes=False)
        runtime = RoomRuntime(
            directory=directory,
            owner_id="room",
            driver_factory=factory,
            spool=tmp_path,
            archives=archives,
            max_rooms=2,
        )

        async def join(spec):
            game, stream = new_id(), new_id()
            await directory.resolve(game, "world", create=True)
            body = dict(
                game_id=game,
                player_id=new_id(),
                stream_id=stream,
                session_id=new_id(),
                user_id="user",
                world_id="world",
                trace_id=new_id(),
                spec=dict(width=spec.width, height=spec.height, fps=24),
            )
            await runtime.join(body)
            return body

        try:
            bodies = await asyncio.gather(join(VideoSpec()), join(VideoSpec(1280, 720)))
            assert runtime.spec == VideoSpec()
            assert {room["spec"] for room in runtime.rooms.values()} == {
                VideoSpec(),
                VideoSpec(1280, 720),
            }
            for body in bodies:
                entry = runtime.players[body["stream_id"]]
                spec = entry["spec"]
                await entry["engine"].push()
                frame = await runtime.next_frame(body["stream_id"])
                assert len(frame["rgb"]) == spec.frame_bytes
                assert (
                    await runtime.render(body["stream_id"], 0, frame["rgb"])
                    == frame["rgb"]
                )
                assert (
                    session_source_spec(
                        {"game_id": body["game_id"]}, VideoSpec(), runtime.rooms
                    )
                    == spec
                )
            original = bodies[1]
            mismatch = {
                **original,
                "player_id": new_id(),
                "stream_id": new_id(),
                "spec": dict(width=832, height=480, fps=24),
            }
            with pytest.raises(ValueError) as failure:
                await runtime.join(mismatch)
            assert failure.value.code == "RESOLUTION_UNSUPPORTED"
            assert len(drivers) == 2 and len(runtime.players) == 2
        finally:
            await runtime.close()
            await archives.close()
        assert not runtime.rooms and not runtime.players
        assert all(d.closed and d.engine.closed for d in drivers)

    asyncio.run(run())


def test_full_room_preserves_admission_error_and_existing_game_joins(tmp_path):
    from test_whitebox_interactive import Leases

    from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import MESSAGES
    from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        RoomPipelines,
    )

    async def run():
        spec = VideoSpec(32, 32)
        directory = RoomDirectory(MemoryRoomStore(), {"room": "http://room"})
        archives = ArchiveManager(processes=False)
        drivers, engines = [], []

        class Driver:
            max_players = 2

            def __init__(self, spec):
                self.spec, self.closed = spec, False

            async def join(self, *, spec, **_):
                assert spec == self.spec
                engine = Engine(spec)
                engine.renderer = BrowserUI()
                engines.append(engine)
                return engine

            async def close(self):
                self.closed = True

        async def factory(*, spec, **_):
            driver = Driver(spec)
            drivers.append(driver)
            return driver

        runtime = RoomRuntime(
            directory=directory,
            owner_id="room",
            driver_factory=factory,
            spool=tmp_path,
            spec=spec,
            archives=archives,
            max_rooms=1,
        )
        leases = Leases(spec)
        host = RoomPipelines(
            runtime,
            WhiteboxConfig(
                engine_factory=None,
                ui_factory=None,
                spool=tmp_path,
                spec=spec,
                room_owned=True,
            ),
            archives,
            Reports(tmp_path, archives),
            lambda *_: leases,
            max_sessions=2,
        )

        async def player(game, selected=spec):
            return dict(
                game_id=game,
                player_id=new_id(),
                stream_id=new_id(),
                session_id=new_id(),
                user_id="user",
                world_id="world",
                trace_id="room-admission-test",
                spec=dict(
                    width=selected.width, height=selected.height, fps=selected.fps
                ),
            )

        game, rejected_game = new_id(), new_id()
        await directory.resolve(game, "world", create=True)
        await directory.resolve(rejected_game, "world", create=True)
        first = await player(game)
        try:
            await runtime.join(first)
            existing_room = runtime.rooms[game]
            existing_player = runtime.players[first["stream_id"]]
            owner, sub = await host.attach(
                dict(
                    user_id="user",
                    trace_id="full-room-trace",
                    message=dict(
                        type="init",
                        prompt="test",
                        render_mode="whitebox",
                        whitebox=dict(game_id=rejected_game, world_id="world"),
                    ),
                )
            )
            async with asyncio.timeout(5):
                while True:
                    error = msgspec.msgpack.decode(await sub.recv())
                    if error["type"] == "error":
                        break
                await owner.task
            assert error["code"] == "CAPACITY_EXHAUSTED"
            assert error["stage"] == "admission"
            assert error["trace_id"] == "full-room-trace"
            assert error["content"] == MESSAGES["CAPACITY_EXHAUSTED"]
            assert owner.closed and not host.sessions
            assert len(drivers) == 1 and leases.opens == []
            assert runtime.rooms == {game: existing_room}
            assert runtime.players == {first["stream_id"]: existing_player}
            assert not engines[0].closed and not drivers[0].closed

            joined = await player(game)
            await runtime.join(joined)
            assert len(drivers) == 1 and len(runtime.players) == 2
            with pytest.raises(ValueError, match="another player view"):
                await runtime.join(await player(game))
            await runtime.leave(joined["stream_id"])
            await runtime.leave(first["stream_id"])
            assert not runtime.rooms and not runtime.players

            next_game = new_id()
            await directory.resolve(next_game, "world", create=True)
            await runtime.join(await player(next_game, VideoSpec(1280, 720)))
            assert runtime.rooms[next_game]["spec"] == VideoSpec(1280, 720)
            assert runtime.spec is spec and len(drivers) == 2
        finally:
            await host.close()
            await runtime.close()
            await archives.close()
        assert all(driver.closed for driver in drivers)
        assert all(engine.closed for engine in engines)

    asyncio.run(run())


@pytest.mark.parametrize(
    "resolution,code",
    [("720p", "RESOLUTION_UNSUPPORTED"), ("736p", "INVALID_RESOLUTION")],
)
def test_classic_path_rejects_unsupported_resolution_before_allocation(
    tmp_path, resolution, code
):
    async def run():
        calls = []

        async def factory(**_):
            calls.append("cli")

        async def send(_):
            calls.append("model")

        async def close():
            pass

        session = WhiteboxUpstream(
            NS(send=send, close=close),
            config=WhiteboxConfig(
                engine_factory=factory, ui_factory=None, spool=tmp_path
            ),
            archives=NS(),
            trace_id="reject",
            session_id=new_id(),
        )
        with pytest.raises(ValueError) as failure:
            await session.send(
                msgspec.msgpack.encode(
                    dict(
                        type="init",
                        prompt="test",
                        whitebox=dict(world_id="world", resolution=resolution),
                    )
                )
            )
        assert failure.value.code == code and failure.value.stage == "validation"
        assert calls == [] and session.engine is None
        await session.close()
        assert all(task.done() for task in session.tasks)

    asyncio.run(run())


@pytest.mark.parametrize(
    "resolution,code",
    [("720p", "RESOLUTION_UNSUPPORTED"), ("736p", "INVALID_RESOLUTION")],
)
def test_classic_gateway_rejects_before_coordinator_and_cli(tmp_path, resolution, code):
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
        create_app,
    )

    calls = []

    class Coordinator:
        async def admit(self, **_):
            calls.append("admit")
            raise AssertionError("Resolution must be checked before model admission")

    async def connect(*_, **__):
        calls.append("connect")
        raise AssertionError("No model connection may start")

    async def factory(**_):
        calls.append("cli")

    app = create_app(
        Coordinator(),
        model_revision="h3",
        vae_fingerprint="local",
        internal_output_url="ws://gateway/internal",
        connect_factory=connect,
        whitebox_config=WhiteboxConfig(
            engine_factory=factory, ui_factory=None, spool=tmp_path
        ),
    )
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/realtime_video/generate?user_id=user&trace_id=resolution-trace"
        ) as ws:
            ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="client_metric", stage="client_video_decode", duration_ms=1
                    )
                )
            )
            ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="init",
                        prompt="test",
                        whitebox=dict(world_id="world", resolution=resolution),
                    )
                )
            )
            error = msgspec.msgpack.decode(ws.receive_bytes())
            assert error["code"] == code and error["stage"] == "validation"
            assert error["trace_id"] == "resolution-trace"
    assert calls == []


@pytest.mark.parametrize("overflow", ["bytes", "messages"])
def test_classic_preflight_bounds_unadmitted_input(tmp_path, overflow):
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
        create_app,
    )

    calls = []

    class Coordinator:
        async def admit(self, **_):
            calls.append("admit")
            raise AssertionError("Overflow cannot allocate a model")

    app = create_app(
        Coordinator(),
        model_revision="h3",
        vae_fingerprint="local",
        internal_output_url="ws://gateway/internal",
        whitebox_config=WhiteboxConfig(
            engine_factory=None, ui_factory=None, spool=tmp_path
        ),
    )
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime_video/generate?user_id=user") as ws:
            if overflow == "messages":
                for _ in range(17):
                    ws.send_bytes(msgspec.msgpack.encode(dict(type="client_metric")))
            ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="init",
                        whitebox=dict(world_id="world"),
                        prompt="p" * (9 * 1024**2) if overflow == "bytes" else "test",
                    )
                )
            )
            error = msgspec.msgpack.decode(ws.receive_bytes())
            assert (
                error["code"] == "INVALID_CONTROL_MESSAGE"
                and error["stage"] == "validation"
            )
    assert calls == []


@pytest.mark.parametrize("kind", ["resume", "source.bind"])
@pytest.mark.parametrize(
    "selector,code",
    [("480p", "RESOLUTION_UNSUPPORTED"), ("736p", "INVALID_RESOLUTION")],
)
def test_rtc_bad_resume_resolution_preserves_the_live_attachment(kind, selector, code):
    from unittest.mock import AsyncMock, Mock

    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import reserve_rtc

    async def run():
        subscription = NS(wait_closed=AsyncMock())
        peer = NS(close=AsyncMock())
        room = {"spec": VideoSpec(1280, 720)}
        pipeline = NS(
            closed=False,
            closing=False,
            terminal=None,
            media_transport="webrtc-rtp",
            user_id="user",
            game_id="b" * 32,
            token="secret",
            history_floor=0,
            control_sequence=7,
            subscription=subscription,
            rtc_peer=peer,
            rtc_reconnect=None,
            detach=Mock(),
            session=NS(source_epoch="source", config=NS(spec=room["spec"])),
        )
        host = NS(
            runtime=NS(
                epoch="owner", rtc_preparation=None, rooms={pipeline.game_id: room}
            ),
            lock=asyncio.Lock(),
            sessions={"a" * 32: pipeline},
            coordinator=NS(admit=AsyncMock()),
        )
        body = dict(
            user_id="user",
            message=dict(
                type=kind,
                control_protocol="whitebox-render-v2",
                source_session_id="a" * 32,
                whitebox=dict(game_id=pipeline.game_id, resolution=selector),
                owner_epoch="owner",
                source_epoch="source",
                resume_token="secret",
                control_sequence=7,
            ),
        )
        with pytest.raises(ValueError) as failure:
            await reserve_rtc(host, body)
        assert failure.value.code == code and failure.value.stage == "validation"
        assert pipeline.subscription is subscription and pipeline.rtc_peer is peer
        assert pipeline.rtc_reconnect is None and host.runtime.rtc_preparation is None
        assert host.runtime.rooms[pipeline.game_id] is room
        pipeline.detach.assert_not_called()
        subscription.wait_closed.assert_not_called()
        peer.close.assert_not_called()
        host.coordinator.admit.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize("fps,level", [(24, "1f"), (30, "1f"), (60, "20")])
def test_720_capture_rate_sets_real_sps_and_keeps_every_frame(tmp_path, fps, level):
    from sglang.multimodal_gen.runtime.realtime.whitebox.playback import (
        H264PlaybackEncoder,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import H264MuxFile

    spec = VideoSpec(1280, 720)
    encoder = H264PlaybackEncoder(spec, capture_clock=True, capture_fps=fps, gop=6)
    decoder = av.CodecContext.create("h264", "r")
    mux = H264MuxFile(tmp_path / "capture.mp4", spec)
    outputs, decoded = [], []
    try:
        assert encoder.stream["codec"] == "avc1.42c0" + level
        assert encoder.stream["max_encoding_fps"] == fps
        for index in range(8):
            if index == 6:
                encoder.set_target_bitrate(800_000)
            outputs.extend(
                encoder.push(
                    index,
                    bytes([index * 20]) * spec.frame_bytes,
                    round(index * 1e6 / fps),
                )
            )
        outputs.extend(encoder.drain())
        assert [index for index, _ in outputs] == list(range(8))
        for index, encoded in outputs:
            assert encoded["codec"] == encoder.stream["codec"]
            decoded.extend(decoder.decode(av.Packet(encoded["payload"])))
            mux.append(index, encoded, pts_us=round(index * 1e6 / fps))
        mux.close()
        assert len(decoded) == 8
        assert all((frame.width, frame.height) == (1280, 720) for frame in decoded)
        with av.open(str(tmp_path / "capture.mp4")) as movie:
            assert len(list(movie.decode(video=0))) == 8
        assert encoder.context.framerate == fps
    finally:
        encoder.close()
        mux.close()


@pytest.mark.parametrize("fps", [0, 61, True, 24.0])
def test_playback_rejects_invalid_capture_rate(fps):
    from sglang.multimodal_gen.runtime.realtime.whitebox.playback import (
        H264PlaybackEncoder,
    )

    with pytest.raises(ValueError, match="capture rate"):
        H264PlaybackEncoder(VideoSpec(1280, 720), capture_clock=True, capture_fps=fps)


@pytest.mark.parametrize(
    "metadata,requested,env_fps,expected",
    [
        ({"capture_fps": 30}, 24, "60", 30),
        ({}, 30, "60", 30),
        ({}, None, "30", 30),
        ({}, None, None, 24),
    ],
)
def test_session_capture_hint_prefers_actual_source_rate(
    tmp_path, monkeypatch, metadata, requested, env_fps, expected
):
    if env_fps is None:
        monkeypatch.delenv("WHITEBOX_CLI_FPS", raising=False)
    else:
        monkeypatch.setenv("WHITEBOX_CLI_FPS", env_fps)
    session = object.__new__(InteractiveSession)
    session.config = WhiteboxConfig(
        engine_factory=None, ui_factory=None, spool=tmp_path, spec=VideoSpec(1280, 720)
    )
    session.engine = NS(metadata=metadata)
    session.requested_capture_fps = requested
    assert session.capture_rate_hint() == expected
    session.config = WhiteboxConfig(
        engine_factory=None, ui_factory=None, spool=tmp_path
    )
    assert session.capture_rate_hint() == 60


@pytest.mark.parametrize("requested,environment", [(61, None), (None, "61")])
def test_rtc_invalid_rate_rejected_before_reservation_or_source_work(
    monkeypatch, requested, environment
):
    from unittest.mock import AsyncMock

    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import reserve_rtc

    if environment is None:
        monkeypatch.delenv("WHITEBOX_CLI_FPS", raising=False)
    else:
        monkeypatch.setenv("WHITEBOX_CLI_FPS", environment)

    async def run():
        host = NS(
            config=NS(spec=VideoSpec(832, 480)),
            runtime=NS(rooms={}, rtc_preparation=None),
            lock=asyncio.Lock(),
            sessions={},
            max_sessions=1,
            coordinator=NS(admit=AsyncMock()),
        )
        body = dict(
            user_id="user",
            message=dict(
                type="init",
                source_operation="create",
                control_protocol="whitebox-render-v2",
                whitebox=dict(world_id="world", resolution="720p", cli_fps=requested),
            ),
        )
        with pytest.raises(ValueError) as failure:
            await reserve_rtc(host, body)
        if requested is not None:
            assert failure.value.code == "INVALID_CLI_FPS"
        assert host.runtime.rtc_preparation is None and host.runtime.rooms == {}
        host.coordinator.admit.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["resume", "source.bind"])
def test_rtc_invalid_rate_resume_keeps_current_media_and_room(kind):
    from unittest.mock import AsyncMock, Mock

    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import reserve_rtc

    async def run():
        subscription, peer = NS(wait_closed=AsyncMock()), NS(close=AsyncMock())
        pipeline = NS(
            closed=False,
            closing=False,
            terminal=None,
            media_transport="webrtc-rtp",
            user_id="user",
            game_id="b" * 32,
            token="secret",
            history_floor=0,
            control_sequence=7,
            subscription=subscription,
            rtc_peer=peer,
            rtc_reconnect=None,
            detach=Mock(),
            session=NS(
                source_epoch="source",
                config=NS(spec=VideoSpec(1280, 720)),
                capture_rate_hint=lambda: 61,
            ),
        )
        room = {"spec": pipeline.session.config.spec}
        host = NS(
            runtime=NS(
                epoch="owner", rtc_preparation=None, rooms={pipeline.game_id: room}
            ),
            lock=asyncio.Lock(),
            sessions={"a" * 32: pipeline},
            coordinator=NS(admit=AsyncMock()),
        )
        body = dict(
            user_id="user",
            message=dict(
                type=kind,
                control_protocol="whitebox-render-v2",
                source_session_id="a" * 32,
                whitebox=dict(game_id=pipeline.game_id),
                owner_epoch="owner",
                source_epoch="source",
                resume_token="secret",
                control_sequence=7,
            ),
        )
        with pytest.raises(ValueError) as failure:
            await reserve_rtc(host, body)
        assert (
            failure.value.code == "RTC_CODEC_UNSUPPORTED"
            and failure.value.stage == "validation"
        )
        assert pipeline.subscription is subscription and pipeline.rtc_peer is peer
        assert host.runtime.rooms[pipeline.game_id] is room
        assert pipeline.rtc_reconnect is None and host.runtime.rtc_preparation is None
        pipeline.detach.assert_not_called()
        subscription.wait_closed.assert_not_called()
        peer.close.assert_not_called()
        host.coordinator.admit.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize(
    "resolution,requested,environment",
    [
        ("480p", 60, "60"),
        ("720p", None, None),
        ("720p", 30, "60"),
        ("720p", None, "30"),
    ],
)
def test_rtc_rate_preflight_uses_effective_cli_configuration(
    monkeypatch, resolution, requested, environment
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import reserve_rtc

    if environment is None:
        monkeypatch.delenv("WHITEBOX_CLI_FPS", raising=False)
    else:
        monkeypatch.setenv("WHITEBOX_CLI_FPS", environment)

    async def run():
        host = NS(
            config=NS(spec=VideoSpec(832, 480)),
            runtime=NS(rooms={}, rtc_preparation=None),
            lock=asyncio.Lock(),
            sessions={},
            max_sessions=1,
        )
        world = dict(resolution=resolution, world_id="world")
        if requested is not None:
            world["cli_fps"] = requested
        body = dict(
            user_id="user",
            message=dict(
                type="init",
                source_operation="create",
                control_protocol="whitebox-render-v2",
                whitebox=world,
            ),
        )
        reservation, pipeline = await reserve_rtc(host, body)
        assert pipeline is None and host.runtime.rtc_preparation == reservation
        assert body["message"]["whitebox"] is world
        assert world.get("cli_fps") == requested

    asyncio.run(run())


@pytest.mark.parametrize(
    "fps,environment", [(24, "60"), (30, "60"), (60, "24"), (60, "30")]
)
def test_rtc_existing_room_uses_confirmed_descriptor_rate(
    monkeypatch, fps, environment
):
    from unittest.mock import AsyncMock

    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import reserve_rtc

    monkeypatch.setenv("WHITEBOX_CLI_FPS", environment)
    from sglang.multimodal_gen.runtime.realtime.whitebox import session_encoding

    probes = []

    def probe(settings, rate, **geometry):
        probes.append(rate)
        return "42c020"

    monkeypatch.setattr(session_encoding, "preflight_source_encoding", probe)

    async def run():
        room = dict(
            spec=VideoSpec(1280, 720), cli_fps=None, driver=NS(descriptor={"fps": fps})
        )
        host = NS(
            config=WhiteboxConfig(engine_factory=None, ui_factory=None, spool=None),
            runtime=NS(rooms={"b" * 32: room}, rtc_preparation=None),
            lock=asyncio.Lock(),
            sessions={},
            max_sessions=1,
            coordinator=NS(admit=AsyncMock()),
        )
        body = dict(
            user_id="user",
            message=dict(
                type="init",
                source_operation="create",
                control_protocol="whitebox-render-v2",
                whitebox=dict(world_id="world", game_id="b" * 32),
            ),
        )
        reservation, pipeline = await reserve_rtc(host, body)
        assert pipeline is None and host.runtime.rtc_preparation == reservation
        assert probes == ([fps] if fps > 30 else [])
        assert host.runtime.rooms["b" * 32] is room
        assert room["driver"].descriptor["fps"] == fps and room["cli_fps"] is None
        host.coordinator.admit.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize("resolution,fps", [("480p", 60), ("720p", 24), ("720p", 30)])
def test_real_session_source_model_and_paired_archive_use_pinned_capture_rate(
    tmp_path, resolution, fps
):
    async def run():
        spec = VideoSpec(832, 480) if resolution == "480p" else VideoSpec(1280, 720)
        engine = Engine(spec)
        engine.metadata = {"capture_fps": fps}
        archives = ArchiveManager(processes=False)

        class Factory:
            async def open_session(self, **_):
                return engine

        session = InteractiveSession(
            leases=NS(),
            config=WhiteboxConfig(
                engine_factory=Factory(),
                ui_factory=lambda **_: BrowserUI(),
                spool=tmp_path,
                room_owned=True,
                capture_clock=True,
                model_input_frames=18,
            ),
            archives=archives,
            session_id=new_id(),
            trace_id="capture-codec",
        )
        try:
            await session.send(
                msgspec.msgpack.encode(
                    dict(
                        type="init",
                        prompt="test",
                        render_mode="whitebox",
                        whitebox=dict(
                            world_id="world", resolution=resolution, cli_fps=fps
                        ),
                    )
                )
            )
            ready = await receive_until(session, lambda m: m["type"] == "session.ready")
            hint = 60 if resolution == "480p" else fps
            for encoder in (
                session.playback,
                session.source_playback,
                session.record_source,
            ):
                assert encoder.stream["max_encoding_fps"] == hint
                assert encoder.stream["codec"] == "avc1.42c01f"
            assert ready["source_video_stream"]["codec"] == "avc1.42c01f"
        finally:
            await session.close()
            await archives.close()
        assert engine.closed

    asyncio.run(run())


@pytest.mark.parametrize("codec", ["h264", "vp9", "av1"])
@pytest.mark.parametrize("fps", [36, 60])
def test_rtc_session_high_rate_is_not_level31_limited(tmp_path, codec, fps):
    session = object.__new__(InteractiveSession)
    session.config = WhiteboxConfig(
        engine_factory=None, ui_factory=None, spool=tmp_path, spec=VideoSpec(1280, 720)
    )
    session.engine = NS(metadata={"capture_fps": fps})
    session.rtc_level31 = True
    assert session.capture_rate_hint(codec) == fps
