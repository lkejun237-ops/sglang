# SPDX-License-Identifier: Apache-2.0
"""CPU end-to-end contracts. No Redis, OSS, CLI or GPU connection is made."""

import asyncio
import hashlib
import json
import re
from fractions import Fraction

import av
import msgspec
import pytest
from test_realtime_coordinator import FakeAsyncRedis, _heartbeat

from sglang.multimodal_gen.runtime.realtime.coordinator import (
    CoordinatorRejected,
    InMemoryCoordinatorStore,
    RealtimeCoordinator,
    RedisCoordinatorStore,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
    EngineFrame,
    VideoSpec,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.playback import H264PlaybackEncoder
from sglang.multimodal_gen.runtime.realtime.whitebox.recording import (
    ArchiveManager,
    InputIntervals,
    publish_archive,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.session import (
    WhiteboxConfig,
    WhiteboxUpstream,
)


def test_browser_engine_requires_requested_capture_and_records_applied_controls():
    import io

    from PIL import Image

    from sglang.multimodal_gen.runtime.realtime.whitebox.browser_engine import (
        BrowserEngine,
    )

    async def run():
        engine = BrowserEngine(world_id="test", spec=VideoSpec(32, 32), trace_id="test")
        stream = engine.frames()
        waiting = asyncio.create_task(anext(stream))
        request = await engine.messages.get()
        assert request["index"] == 0
        await engine.send_input(
            0, {"held_keys": ["KeyW"], "events": [{"type": "keydown"}]}
        )
        await engine.send_input(1, {"held_keys": ["KeyD"]})
        image = io.BytesIO()
        Image.new("RGB", (32, 32), (10, 20, 30)).save(image, format="PNG")
        # Only input zero was actually applied when the client captured frame 0.
        await engine.receive_frame(
            dict(
                index=0,
                applied_sequence=0,
                data=image.getvalue(),
                states={"arbitrary": [1, 2]},
            )
        )
        frame = await waiting
        assert frame.effective_input["held_keys"] == ["KeyW"]
        assert frame.states["capture"] == {"arbitrary": [1, 2]}
        with pytest.raises(ValueError, match="Unrequested"):
            await engine.receive_frame(
                dict(index=0, applied_sequence=1, data=image.getvalue())
            )
        await engine.finish()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        await stream.aclose()

    asyncio.run(run())


def test_trace_report_counts_once_and_does_not_record_sensitive_payloads(tmp_path):
    import logging

    from sglang.multimodal_gen.runtime.realtime.whitebox.reports import SessionReport
    from sglang.multimodal_gen.runtime.utils.realtime_trace import (
        emit_realtime_trace_payload,
    )

    async def run():
        report = SessionReport("bounded-trace")
        for i in range(600):
            emit_realtime_trace_payload(
                logging.getLogger(__name__),
                dict(
                    trace_id=report.trace_id,
                    event="whitebox.frame",
                    duration_ms=i,
                    prompt="must not persist",
                    payload="pixels",
                ),
            )
        rows = report.snapshot()
        assert len(rows) == 1 and rows[0]["count"] == 600
        assert rows[0]["p95_window"] == 512 and rows[0]["max_ms"] == 599
        assert "must not persist" not in json.dumps(rows)
        report.close()
        emit_realtime_trace_payload(
            logging.getLogger(__name__),
            dict(trace_id=report.trace_id, event="whitebox.frame", duration_ms=1),
        )
        assert report.snapshot()[0]["count"] == 600

    asyncio.run(run())


def test_report_requires_capability_and_survives_restart(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import Recording
    from sglang.multimodal_gen.runtime.realtime.whitebox.reports import (
        Reports,
        SessionReport,
    )

    async def run():
        archives = ArchiveManager()
        reports = Reports(tmp_path, archives)
        report = SessionReport("private-recording")
        recording = Recording(
            directory=tmp_path / ("a" * 32),
            spec=VideoSpec(32, 32),
            world_id="test",
            trace_id=report.trace_id,
            inference={},
            ui={},
            references=[],
            object_prefix="test",
        )
        reports.attach(report, recording)
        await recording.finish("aborted")
        report.close()
        app = FastAPI()
        Reports(tmp_path, archives).install(app)
        with TestClient(app) as client:
            url = "/v1/whitebox/recordings/" + recording.directory.name
            assert client.get(url).status_code == 404
            assert (
                client.get(url, headers={"Authorization": "Bearer wrong"}).status_code
                == 404
            )
            result = client.get(
                url, headers={"Authorization": "Bearer " + report.token}
            )
            assert result.status_code == 200
            assert result.json()["recording_status"] == "partial"
            assert not result.json()["videos"]
            assert (
                report.token not in (recording.directory / "manifest.json").read_text()
            )

    asyncio.run(run())


@pytest.mark.parametrize("redis", [False, True])
def test_local_decode_uses_one_real_slot_and_preserves_lease_fencing(redis):
    async def run():
        kwargs = dict(ttl_s=60, worker_ttl_s=30)
        store = (
            RedisCoordinatorStore(client=FakeAsyncRedis(), **kwargs)
            if redis
            else InMemoryCoordinatorStore(**kwargs)
        )
        reservations = []

        class Reservation:
            async def reserve(self, slot, **kwargs):
                reservations.append(("reserve", slot.worker_id))

            async def release(self, slot, **kwargs):
                reservations.append(("release", slot.worker_id))

        coordinator = RealtimeCoordinator(
            store, wait_timeout_s=0, reservation_client=Reservation()
        )
        await coordinator.heartbeat(
            _heartbeat("h3", "denoiser", model_revision="h3", vae_fingerprint="local")
        )
        request = dict(
            user_id="user",
            session_id="s",
            generation_id="g",
            model_revision="h3",
            vae_fingerprint="local",
        )
        assignment = await coordinator.admit(**request, wait_for_capacity=False)
        assert assignment.vae is None
        assert reservations == [("reserve", "h3")]
        with pytest.raises(CoordinatorRejected, match="USER_SESSION_LIMIT"):
            await coordinator.admit(**request, wait_for_capacity=False)
        renewed = await coordinator.renew(assignment)
        assert renewed.vae is None
        await coordinator.release(renewed)
        assert reservations == [("reserve", "h3"), ("release", "h3")]
        again = await coordinator.admit(**request, wait_for_capacity=False)
        await coordinator.heartbeat(
            _heartbeat(
                "h3",
                "denoiser",
                model_revision="h3",
                vae_fingerprint="local",
                worker_epoch="new",
            )
        )
        with pytest.raises(CoordinatorRejected, match="WORKER_LOST"):
            await coordinator.renew(again)
        await coordinator.release(again)

    asyncio.run(run())


def test_intervals_split_on_input_prompt_and_reference_changes(tmp_path):
    intervals = InputIntervals(tmp_path / "inputs.jsonl", VideoSpec(32, 32))
    for frame, (inputs, prompt) in enumerate(
        [
            ({"keys": [], "references": []}, "a"),
            ({"keys": [], "references": []}, "a"),
            ({"keys": ["W"], "references": []}, "a"),
            ({"keys": ["W"], "references": [{"path": "references/0.png"}]}, "b"),
        ]
    ):
        intervals.append(frame, inputs, prompt)
    intervals.close()
    rows = [
        json.loads(line)
        for line in (tmp_path / "inputs.jsonl").read_text().splitlines()
    ]
    assert [(r["start_frame"], r["end_frame"]) for r in rows] == [
        (0, 2),
        (2, 3),
        (3, 4),
    ]
    assert all(
        set(row)
        == {
            "input",
            "prompt",
            "start_frame",
            "end_frame",
            "start_time_us",
            "end_time_us",
        }
        for row in rows
    )
    assert rows[-1]["end_time_us"] == 4_000_000 // 24


class Engine:
    def __init__(self, spec, count=6):
        self.spec, self.count = spec, count
        self.closed = self.stopped = False
        self.controls = []

    async def frames(self):
        for index in range(self.count):
            if self.stopped:
                return
            yield EngineFrame(
                index,
                self.spec.pts(index),
                bytes([20]) * self.spec.frame_bytes,
                {"anything": [index, {"opaque": True}]},
                {"held_keys": [] if index < 2 else ["W"]},
            )

    async def send_input(self, sequence, payload):
        self.controls.append((sequence, payload))

    async def finish(self):
        self.stopped = True

    async def close(self):
        self.closed = True


class Renderer:
    metadata = {"bundle": "test-ui", "version": 1}

    async def render(self, rgb, frame, spec):
        assert frame.states["anything"][0] == frame.index
        return bytes([200]) * spec.frame_bytes

    async def close(self):
        pass


class H3:
    def __init__(self, spec):
        self.spec = spec
        self.queue = asyncio.Queue()
        self.events = []
        self.prompt_id = 0
        self.closed = False

    async def send(self, wire):
        message = msgspec.msgpack.decode(wire)
        if message["type"] == "init":
            self.init = message
            await self.emit(
                dict(
                    type="session.ready",
                    source_stream=message["source_stream"],
                    max_message_bytes=64 * 1024 * 1024,
                    max_pending_source_frames=2,
                    max_pending_source_bytes=self.spec.frame_bytes * 2,
                    max_source_message_bytes=self.spec.frame_bytes + 4096,
                )
            )
        elif message.get("kind") in {"source.append", "source.end", "prompt"}:
            self.events.append(message)
            payload = message["payload"]
            if message["kind"] == "prompt":
                self.prompt_id = message["event_id"]
            elif message["kind"] == "source.end":
                await self.emit(dict(type="session.completed"))
            else:
                index = payload["source_frame_start"]
                await self.emit(
                    dict(
                        type="source.accepted",
                        source_frame_start=index,
                        source_frame_end=index + 1,
                    )
                )
                await self.emit(
                    dict(
                        type="frame_batch",
                        output_frame_start=index,
                        num_frames=1,
                        pts_us=self.spec.pts(index),
                        width=self.spec.width,
                        height=self.spec.height,
                        format="rgb24",
                        encoding="raw",
                        payload=bytes([100]) * self.spec.frame_bytes,
                        event_id=self.prompt_id,
                        source_alignment=[
                            dict(
                                source_frame_start=index,
                                output_frame_start=index,
                                num_frames=1,
                            )
                        ],
                    )
                )
                await self.emit(
                    dict(
                        type="source.consumed",
                        source_frame_end=index + 1,
                        source_bytes_end=(index + 1) * self.spec.frame_bytes,
                    )
                )

    async def emit(self, message):
        await self.queue.put(msgspec.msgpack.encode(message))

    async def recv(self):
        return await self.queue.get()

    async def close(self):
        self.closed = True


def setup_pipeline(tmp_path, **options):
    spec = VideoSpec(32, 32)
    engine, h3, archives = (
        Engine(spec, options.pop("count", 6)),
        H3(spec),
        ArchiveManager(),
    )

    async def factory(**kwargs):
        return engine

    config = WhiteboxConfig(
        engine_factory=factory,
        ui_factory=lambda **_: Renderer(),
        spool=tmp_path,
        spec=spec,
        **options,
    )
    upstream = WhiteboxUpstream(
        h3, config=config, archives=archives, trace_id="test", session_id="session"
    )
    return upstream, h3, engine, archives


@pytest.mark.parametrize("steps", [1, 2, 4])
@pytest.mark.parametrize("processes", [False, True])
def test_pipeline_encodes_three_aligned_videos_and_preserves_opaque_states(
    tmp_path, steps, processes
):
    async def run():
        upstream, h3, engine, archives = setup_pipeline(
            tmp_path, num_inference_steps=steps
        )
        archives.processes = processes
        # recv is normally started before the browser sends init.
        receive = asyncio.create_task(upstream.recv())
        await asyncio.sleep(0)
        await upstream.send(
            msgspec.msgpack.encode(
                # Browser sampling fields cannot override the trusted deployment.
                dict(
                    type="init",
                    whitebox={"world_id": "world-a"},
                    prompt="a",
                    num_inference_steps=99,
                )
            )
        )
        ready = msgspec.msgpack.decode(await receive)
        assert ready["type"] == "session.ready"
        assert ready["format"] == "h264"
        assert ready["video_stream"]["bitstream_format"] == "annexb"
        messages = []
        while True:
            message = msgspec.msgpack.decode(await asyncio.wait_for(upstream.recv(), 5))
            if message["type"] == "session.completed":
                break
            upstream.media_delivered()
            messages.append(message)
        assert [m["output_frame_start"] for m in messages] == list(range(6))
        assert messages[4]["states"] == {"anything": [4, {"opaque": True}]}
        decoder = av.CodecContext.create("h264", "r")
        playback = []
        for message in messages:
            assert message["encoding"] == "h264"
            assert message["codec"] == ready["video_stream"]["codec"]
            packet = av.Packet(message["payload"])
            packet.pts = packet.dts = message["pts_us"]
            packet.time_base = Fraction(1, 1_000_000)
            playback.extend(decoder.decode(packet))
        playback.extend(decoder.decode(None))
        assert [frame.pts for frame in playback] == [m["pts_us"] for m in messages]
        assert all(
            abs(float(frame.to_ndarray(format="rgb24").mean()) - 200) < 4
            for frame in playback
        )
        await upstream.close()
        await archives.close()
        assert engine.closed and h3.closed
        assert [e["event_id"] for e in h3.events] == list(range(1, 8))
        assert h3.init["realtime_output_format"] == "raw"
        assert h3.init["num_inference_steps"] == steps
        directory = next(tmp_path.iterdir())
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["recording_status"] == "complete"
        assert manifest["timeline"]["alignment"] == "one_to_one"
        assert set(manifest["objects"]) == {
            "whitebox.mp4",
            "model.mp4",
            "ui_rendered.mp4",
            "inputs.jsonl",
            "states.jsonl",
        }
        for name, luminance in [("whitebox", 20), ("model", 100), ("ui_rendered", 200)]:
            with av.open(str(directory / (name + ".mp4"))) as container:
                decoded = list(container.decode(video=0))
                assert len(decoded) == 6
                assert [frame.pts * frame.time_base for frame in decoded] == [
                    Fraction(i, 24) for i in range(6)
                ]
                assert (
                    abs(float(decoded[0].to_ndarray(format="rgb24").mean()) - luminance)
                    < 4
                )
        rows = [
            json.loads(line)
            for line in (directory / "inputs.jsonl").read_text().splitlines()
        ]
        assert [(r["start_frame"], r["end_frame"]) for r in rows] == [(0, 2), (2, 6)]

    asyncio.run(run())


def test_h264_playback_emits_immediate_access_units_and_independent_keyframes():
    spec = VideoSpec(64, 32)
    encoder = H264PlaybackEncoder(spec)
    decoder = av.CodecContext.create("h264", "r")
    decoded = []
    for index in range(51):
        chunk = encoder.encode(index, bytes([index + 40]) * spec.frame_bytes)
        assert chunk["duration_us"] == spec.pts(index + 1) - spec.pts(index)
        assert chunk["keyframe"] == (index % spec.fps == 0)
        packet = av.Packet(chunk["payload"])
        packet.pts = packet.dts = spec.pts(index)
        packet.time_base = Fraction(1, 1_000_000)
        decoded.extend(decoder.decode(packet))
        if chunk["keyframe"]:
            units = [n for n in re.split(b"\x00\x00\x00?\x01", chunk["payload"]) if n]
            assert {7, 8, 5} <= {n[0] & 31 for n in units}
            sps = next(n for n in units if n[0] & 31 == 7)
            assert chunk["codec"] == "avc1." + sps[1:4].hex()
            fresh = av.CodecContext.create("h264", "r")
            frames = fresh.decode(packet) + fresh.decode(None)
            assert len(frames) == 1
            assert (
                abs(float(frames[0].to_ndarray(format="rgb24").mean()) - index - 40) < 4
            )
    encoder.finish()
    decoded.extend(decoder.decode(None))
    assert len(decoded) == 51
    assert [f.pts for f in decoded] == [spec.pts(i) for i in range(51)]
    with pytest.raises(ValueError, match="contiguous"):
        encoder.encode(51, bytes(spec.frame_bytes))
    encoder.close()


def test_startup_failure_wakes_receiver_and_releases_resources(tmp_path):
    async def run():
        upstream, h3, engine, archives = setup_pipeline(tmp_path, startup_seconds=0.05)
        await upstream.send(
            msgspec.msgpack.encode(
                dict(type="init", whitebox={"world_id": "a"}, prompt="p")
            )
        )
        # No delivery notification: startup must fail even if frames were rendered.
        await asyncio.sleep(0.08)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(upstream.recv(), 1)
        await upstream.close()
        await archives.close()
        assert engine.closed and h3.closed

    asyncio.run(run())


def test_archive_publishes_manifest_last_and_can_retry(tmp_path):
    async def run():
        (tmp_path / "a.mp4").write_bytes(b"video")
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                dict(
                    trace_id="test",
                    object_prefix="unique/",
                    recording_status="complete",
                    objects={
                        "a.mp4": {
                            "bytes": 5,
                            "sha256": hashlib.sha256(b"video").hexdigest(),
                        }
                    },
                    upload_status="pending",
                )
            )
        )

        class Store:
            def __init__(self):
                self.calls = []
                self.fail = True

            async def upload(self, path, key):
                self.calls.append(key)
                if self.fail:
                    raise OSError("offline")

        store = Store()
        with pytest.raises(OSError):
            await publish_archive(tmp_path, store, retries=1)
        assert store.calls == ["unique/a.mp4"]
        assert (
            json.loads((tmp_path / "manifest.json").read_text())["upload_status"]
            == "failed"
        )
        store.fail = False
        await publish_archive(tmp_path, store)
        assert store.calls[-2:] == ["unique/a.mp4", "unique/manifest.json"]
        assert (
            json.loads((tmp_path / "manifest.json").read_text())["upload_status"]
            == "complete"
        )
        (tmp_path / "a.mp4").write_bytes(b"damaged")
        calls = list(store.calls)
        with pytest.raises(ValueError, match="changed"):
            await publish_archive(tmp_path, store)
        assert store.calls == calls

    asyncio.run(run())


def test_archive_manager_resumes_finalized_failed_upload_after_restart(tmp_path):
    async def run():
        directory = tmp_path / ("b" * 32)
        directory.mkdir()
        (directory / "a.mp4").write_bytes(b"video")
        (directory / "manifest.json").write_text(
            json.dumps(
                dict(
                    trace_id="resume",
                    object_prefix="unique/",
                    recording_status="complete",
                    upload_status="failed",
                    objects={
                        "a.mp4": dict(
                            bytes=5, sha256=hashlib.sha256(b"video").hexdigest()
                        )
                    },
                )
            )
        )
        calls = []

        class Store:
            async def upload(self, path, key):
                calls.append(key)

        manager = ArchiveManager(Store())
        manager.start(tmp_path)
        for _ in range(100):
            if len(calls) == 2:
                break
            await asyncio.sleep(0.01)
        await manager.close()
        assert calls == ["unique/a.mp4", "unique/manifest.json"]
        assert (
            json.loads((directory / "manifest.json").read_text())["upload_status"]
            == "complete"
        )

    asyncio.run(run())


def test_rgba_compositor_preserves_transparent_model_pixels():
    from sglang.multimodal_gen.runtime.realtime.whitebox.ui import RgbaOverlayRenderer

    async def run():
        spec = VideoSpec(32, 32)
        frame = EngineFrame(0, 0, bytes(spec.frame_bytes), {"schema": "unrestricted"})

        async def overlay(received, spec):
            assert received.states == {"schema": "unrestricted"}
            return bytes([255, 0, 0, 255]) + bytes((spec.width * spec.height - 1) * 4)

        renderer = RgbaOverlayRenderer(overlay, metadata={"bundle": "ui-v1"})
        output = await renderer.render(bytes([80]) * spec.frame_bytes, frame, spec)
        assert output[:3] == bytes([255, 0, 0])
        assert output[3:] == bytes([80]) * (spec.frame_bytes - 3)

    asyncio.run(run())


def test_missing_init_is_bounded(tmp_path):
    async def run():
        upstream, h3, _, archives = setup_pipeline(tmp_path, startup_seconds=0.02)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(upstream.recv(), 1)
        await upstream.close()
        await archives.close()
        assert h3.closed

    asyncio.run(run())


@pytest.mark.parametrize("peer_closes", [True, False])
def test_gateway_uses_existing_generate_route_and_releases_local_assignment(
    tmp_path, monkeypatch, peer_closes
):
    import threading
    from urllib.parse import parse_qs, urlsplit

    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocket, WebSocketDisconnect

    from sglang.multimodal_gen.runtime.entrypoints import realtime_gateway_server
    from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
        create_app,
    )

    # A real websockets transport must consume late browser messages to reach
    # the close frame behind them. Model that requirement in the ASGI handshake.
    late_metric = asyncio.Event()
    drained_during_close = []
    close_done = threading.Event()
    original_observe = realtime_gateway_server._observe_gateway_client_metric
    original_close = WebSocket.close

    def observe(message, **kwargs):
        if message.get("duration_ms") == 123:
            late_metric.set()
        return original_observe(message, **kwargs)

    async def close(socket, *args, **kwargs):
        if not drained_during_close:
            try:
                await asyncio.wait_for(late_metric.wait(), 1)
                drained_during_close.append(True)
            except TimeoutError:
                drained_during_close.append(False)
        try:
            await original_close(socket, *args, **kwargs)
        finally:
            close_done.set()

    monkeypatch.setattr(
        realtime_gateway_server, "_observe_gateway_client_metric", observe
    )
    monkeypatch.setattr(WebSocket, "close", close)
    from sglang.multimodal_gen.runtime.realtime.coordinator import (
        SessionAssignment,
        WorkerSlot,
    )

    slot = WorkerSlot(
        worker_id="h3",
        role="denoiser",
        endpoint="ws://h3/v1/realtime_video/generate",
        az="beijing",
        slot_index=0,
        model_revision="h3",
        vae_fingerprint="local",
        worker_epoch="epoch",
    )
    released = []
    release_done = threading.Event()

    class Coordinator:
        async def admit(self, **kwargs):
            return SessionAssignment(
                kwargs["user_id"],
                kwargs["session_id"],
                kwargs["generation_id"],
                "lease-token",
                12345,
                slot,
                None,
            )

        async def renew(self, assignment):
            return assignment

        async def release(self, assignment):
            released.append(assignment)
            release_done.set()

        async def capacity(self, *, model_revision=None, vae_fingerprint=None):
            return {"roles": {"denoiser": {"free_slots": 1}, "vae": {"free_slots": 0}}}

    spec = VideoSpec(32, 32)
    h3, engine = H3(spec), Engine(spec, 4)

    async def connect(url, **kwargs):
        query = parse_qs(urlsplit(url).query)
        assert query["gateway_decode_mode"] == ["local"]
        assert query["coordinator_token"] == ["lease-token"]
        assert "realtime_vae_worker_url" not in query
        return h3

    async def factory(**kwargs):
        return engine

    config = WhiteboxConfig(
        engine_factory=factory,
        ui_factory=lambda **_: Renderer(),
        spool=tmp_path,
        spec=spec,
        drain_seconds=0.2,
    )
    app = create_app(
        Coordinator(),
        model_revision="h3",
        vae_fingerprint="local",
        internal_output_url="ws://gateway/internal",
        whitebox_config=config,
        connect_factory=connect,
        release_grace_s=0,
    )
    with TestClient(app) as client:
        assert client.get("/").status_code == 404
        assert client.get("/v1/slots").json()["free_slots"] == 1
        with client.websocket_connect(
            "/v1/realtime_video/generate?user_id=browser"
        ) as ws:
            ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        version=2,
                        type="client_metric",
                        stage="client_video_decode",
                        duration_ms=1,
                    )
                )
            )
            ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(version=2, type="init", whitebox={"world_id": "a"}, prompt="p")
                )
            )
            messages = []
            while True:
                message = msgspec.msgpack.decode(ws.receive_bytes())
                messages.append(message)
                if message["type"] == "session.completed":
                    break
            assert [
                m["output_frame_start"] for m in messages if m["type"] == "frame_batch"
            ] == [0, 1, 2, 3]
            ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        version=2,
                        type="client_metric",
                        stage="client_video_decode",
                        duration_ms=123,
                    )
                )
            )
            # The SDK closes after decoder drain. A peer that never closes still
            # releases capacity within the configured tail budget.
            if peer_closes:
                ws.close()
            else:
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_bytes()
            # The real peer can close before asynchronous archive/resource cleanup.
            # Keep TestClient from cancelling that cleanup on context exit.
            assert release_done.wait(5)
            # Release precedes the final socket close. Wait for that real boundary
            # too, so TestClient cannot cancel a close already waiting on metrics.
            assert close_done.wait(5)
    assert drained_during_close == [True]
    assert len(released) == 1 and released[0].user_id == "query:browser"
    assert engine.closed and h3.closed


def test_local_assignment_still_requires_every_fence_and_rejects_mixed_routes():
    from sglang.multimodal_gen.runtime.realtime.admission import AdmissionRejected
    from sglang.multimodal_gen.runtime.realtime.gateway_assignment import (
        parse_gateway_assignment,
    )

    query = dict(
        gateway_managed="1",
        gateway_decode_mode="local",
        session_id="s",
        generation_id="g",
        coordinator_token="secret",
        worker_epoch="epoch",
    )
    assert parse_gateway_assignment(query).vae_worker_url == ""
    assert parse_gateway_assignment({}) is None
    for field in ("session_id", "generation_id", "coordinator_token", "worker_epoch"):
        with pytest.raises(AdmissionRejected):
            parse_gateway_assignment({k: v for k, v in query.items() if k != field})
    with pytest.raises(AdmissionRejected):
        parse_gateway_assignment({**query, "realtime_vae_worker_url": "ws://vae"})
    with pytest.raises(AdmissionRejected):
        parse_gateway_assignment({**query, "gateway_decode_mode": "remote"})


def test_oss_client_uses_beijing_compatible_transport_without_sdk_thread_fanout(
    monkeypatch,
):
    import boto3

    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import S3ArchiveStore

    captured = {}

    def client(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(boto3, "client", client)
    store = S3ArchiveStore("test", "https://oss-cn-beijing.aliyuncs.com", "cn-beijing")
    config = captured["config"]
    assert config.s3["addressing_style"] == "virtual"
    assert config.request_checksum_calculation == "when_required"
    assert config.response_checksum_validation == "when_required"
    assert config.connect_timeout == 3 and config.read_timeout == 10
    assert not store.transfer.use_threads


def test_websocket_engine_port_closes_resources_and_requires_explicit_eof():
    from sglang.multimodal_gen.runtime.realtime.whitebox.engine import (
        EngineEnd,
        EngineEndpoint,
        WebSocketEngineFactory,
    )

    async def run():
        cleaned, connected = [], []
        spec = VideoSpec(32, 32)

        async def cleanup():
            cleaned.append(True)

        async def resolver(**_):
            return EngineEndpoint("ws://local-engine", cleanup=cleanup)

        class Socket:
            sent = []

            async def send(self, message):
                self.sent.append(message)

            async def recv(self):
                return "frame"

            async def close(self):
                self.closed = True

        socket = Socket()

        async def connect(*_, **__):
            connected.append(True)
            return socket

        def invalid(_):
            raise ValueError("invalid codec")

        factory = WebSocketEngineFactory(resolver, invalid, connect_factory=connect)
        with pytest.raises(ValueError):
            await factory(world_id="a", spec=spec, trace_id="test")
        assert cleaned == [True] and not connected

        class Codec:
            def input(self, sequence, payload):
                return str(sequence)

            def finish(self):
                return "end"

            def receive(self, wire):
                return EngineEnd(1)  # no frame was delivered

        factory = WebSocketEngineFactory(
            resolver, lambda _: Codec(), connect_factory=connect
        )
        session = await factory(world_id="a", spec=spec, trace_id="test")
        await session.send_input(0, {})
        await session.finish()
        await session.finish()
        await session.send_input(1, {})
        assert socket.sent == ["0", "end"]
        with pytest.raises(ValueError, match="EOF"):
            async for _ in session.frames():
                pass
        await session.close()
        await session.close()
        assert socket.closed and cleaned == [True, True]

    asyncio.run(run())


def test_input_evidence_and_actual_delayed_prompt_ownership_are_recorded(tmp_path):
    class DelayedH3(H3):
        async def send(self, wire):
            message = msgspec.msgpack.decode(wire)
            if message.get("kind") == "prompt":
                self.pending_prompt = message["event_id"]
                return
            if (
                message.get("kind") == "source.append"
                and message["payload"]["source_frame_start"] >= 2
            ):
                self.prompt_id = self.pending_prompt
            await super().send(wire)

    async def run():
        upstream, _, engine, archives = setup_pipeline(tmp_path)
        upstream.upstream = DelayedH3(engine.spec)
        capture = asyncio.Event()

        async def frames():
            await capture.wait()
            for index in range(4):
                yield EngineFrame(
                    index,
                    engine.spec.pts(index),
                    bytes(engine.spec.frame_bytes),
                    {"anything": [index]},
                )

        engine.frames = frames
        await upstream.send(
            msgspec.msgpack.encode(
                dict(type="init", whitebox={"world_id": "a"}, prompt="old")
            )
        )
        assert msgspec.msgpack.decode(await upstream.recv())["type"] == "session.ready"
        await upstream.send(
            msgspec.msgpack.encode(
                dict(
                    type="event", kind="input", sequence=0, payload={"held_keys": ["W"]}
                )
            )
        )
        await upstream.send(
            msgspec.msgpack.encode(
                dict(type="event", kind="input", sequence=1, payload={"held_keys": []})
            )
        )
        await upstream.send(
            msgspec.msgpack.encode(dict(type="event", kind="prompt", payload="new"))
        )
        # Even old-client telemetry never becomes an engine input or source event.
        await upstream.send(
            msgspec.msgpack.encode(
                dict(type="client_metric", stage="client_video_decode", duration_ms=1)
            )
        )
        capture.set()
        while True:
            message = msgspec.msgpack.decode(await asyncio.wait_for(upstream.recv(), 2))
            if message["type"] == "frame_batch":
                upstream.media_delivered()
            if message["type"] == "session.completed":
                break
        await upstream.close()
        await archives.close()
        directory = next(tmp_path.iterdir())
        rows = [
            json.loads(line)
            for line in (directory / "inputs.jsonl").read_text().splitlines()
        ]
        assert rows[-1]["prompt"] == "new" and rows[-1]["start_frame"] == 2
        assert rows[0]["prompt"] == "old"
        evidence = rows[0]["input"]["observed"]
        assert evidence["mapping"] == "unverified_received_before_capture"
        assert [event["sequence"] for event in evidence["events"]] == [0, 1]
        assert (
            json.loads((directory / "manifest.json").read_text())["quality"][
                "unknown_input_frames"
            ]
            == 4
        )

    asyncio.run(run())


def test_cold_start_drains_heartbeats_with_bounded_websocket_queue(tmp_path):
    from dataclasses import replace

    from websockets.asyncio.client import connect
    from websockets.legacy.server import serve

    async def run():
        upstream, h3, engine, archives = setup_pipeline(tmp_path, startup_seconds=5)
        sent = []
        original_send = h3.send

        async def observe_send(wire):
            sent.append(msgspec.msgpack.decode(wire))
            await original_send(wire)

        h3.send = observe_send

        async def cold_factory(**kwargs):
            await asyncio.sleep(1)
            return engine

        upstream.config = replace(upstream.config, engine_factory=cold_factory)

        async def handler(socket):
            async def output():
                while True:
                    await socket.send(await upstream.recv())

            task = asyncio.create_task(output())
            try:
                async for wire in socket:
                    await upstream.send(wire)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await upstream.close()

        async with serve(
            handler, "127.0.0.1", 0, max_queue=1, ping_interval=0.1, ping_timeout=0.3
        ) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}", proxy=None) as socket:
                await socket.send(
                    msgspec.msgpack.encode(
                        dict(type="init", whitebox={"world_id": "cold"}, prompt="p")
                    )
                )
                for _ in range(40):
                    await socket.send(
                        msgspec.msgpack.encode(
                            dict(type="event", kind="heartbeat", payload={})
                        )
                    )
                    await asyncio.sleep(0.015)
                ready = msgspec.msgpack.decode(await asyncio.wait_for(socket.recv(), 3))
                assert ready["type"] == "session.ready"
                # Heartbeats weren't delivered to H3 before its initialization.
                assert sent[0]["type"] == "init"
        await archives.close()
        assert engine.closed

    asyncio.run(run())


def test_manual_finish_has_its_own_drain_deadline_and_partial_archive(tmp_path):
    async def run():
        upstream, _, engine, archives = setup_pipeline(
            tmp_path, play_seconds=100, drain_seconds=0.03
        )

        async def frames():
            yield EngineFrame(
                0, 0, bytes(engine.spec.frame_bytes), {"anything": [0]}, {}
            )
            await asyncio.Event().wait()  # a stalled CLI cannot extend the slot

        engine.frames = frames
        await upstream.send(
            msgspec.msgpack.encode(
                dict(type="init", whitebox={"world_id": "a"}, prompt="p")
            )
        )
        while msgspec.msgpack.decode(await upstream.recv())["type"] != "frame_batch":
            pass
        upstream.media_delivered()
        await upstream.finish()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(upstream.recv(), 1)
        await upstream.close()
        await archives.close()
        manifest = json.loads(next(tmp_path.glob("*/manifest.json")).read_text())
        assert manifest["recording_status"] == "partial"
        assert engine.closed

    asyncio.run(run())


def test_mock_worker_cold_start_has_separate_bounded_deadline_and_releases_slot():
    import time
    from urllib.parse import urlencode

    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from sglang.multimodal_gen.runtime.realtime.whitebox.mock_worker import create_app

    app = create_app(
        coordinator_url="http://unused.invalid",
        public_url="http://unused.invalid",
        startup_seconds=0.8,
        idle_seconds=0.1,
    )
    client = TestClient(app)
    epoch = client.get("/v1/realtime_worker/state").json()["worker_epoch"]

    def reserve(token):
        response = client.post(
            "/v1/realtime_worker/reservations",
            json=dict(
                token=token,
                session_id="session",
                generation_id="generation",
                worker_epoch=epoch,
                ttl_s=10,
            ),
        )
        assert response.status_code == 204
        return "/generate?" + urlencode(
            dict(
                gateway_managed="1",
                gateway_decode_mode="local",
                coordinator_token=token,
                session_id="session",
                generation_id="generation",
                worker_epoch=epoch,
            )
        )

    with client.websocket_connect(reserve("cold-start")) as ws:
        # A real receive waits longer than the streaming idle budget before init.
        time.sleep(0.2)
        ws.send_bytes(
            msgspec.msgpack.encode(
                dict(
                    type="init",
                    source_stream=dict(width=32, height=32, fps=24, format="rgb24"),
                    source_message_budget_bytes=8192,
                )
            )
        )
        assert msgspec.msgpack.decode(ws.receive_bytes())["type"] == "session.ready"
        # After ready, the short idle deadline still bounds abandoned sessions.
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_bytes()
        assert closed.value.code == 1011
    assert client.get("/v1/realtime_worker/state").json()["active_sessions"] == 0

    with client.websocket_connect(reserve("abandoned-start")) as ws:
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_bytes()
        assert closed.value.code == 1011
    state = client.get("/v1/realtime_worker/state").json()
    assert state["active_sessions"] == state["reserved_sessions"] == 0


@pytest.mark.parametrize(
    "mode,status",
    [("zing_h3", "excluded_unvalidated"), ("mock_model", "excluded_mock")],
)
def test_unvalidated_real_model_is_not_mislabeled_mock(tmp_path, mode, status):
    from dataclasses import replace

    async def run():
        upstream, _, _, archives = setup_pipeline(tmp_path)
        upstream.config = replace(
            upstream.config, model_metadata={"mode": mode, "training_eligible": False}
        )
        try:
            await upstream.send(
                msgspec.msgpack.encode(
                    dict(type="init", whitebox={"world_id": "a"}, prompt="p")
                )
            )
            await asyncio.wait_for(upstream.ready.wait(), 2)
            assert upstream.recording.manifest["dataset_status"] == status
        finally:
            await upstream.close()
            await archives.close()

    asyncio.run(run())


def test_heartbeat_during_ui_tail_does_not_send_into_closed_model(tmp_path):
    async def run():
        upstream, h3, _, archives = setup_pipeline(tmp_path, count=1)
        ended, render = asyncio.Event(), asyncio.Event()
        original_send = h3.send

        async def send(wire):
            message = msgspec.msgpack.decode(wire)
            if ended.is_set() and message.get("kind") == "heartbeat":
                raise RuntimeError("H3 socket already closed")
            await original_send(wire)
            if message.get("kind") == "source.end":
                ended.set()

        class SlowRenderer(Renderer):
            async def render(self, rgb, frame, spec):
                await render.wait()
                return await super().render(rgb, frame, spec)

        from dataclasses import replace

        upstream.config = replace(
            upstream.config, ui_factory=lambda **kwargs: SlowRenderer()
        )
        h3.send = send
        try:
            await upstream.send(
                msgspec.msgpack.encode(
                    dict(type="init", whitebox={"world_id": "a"}, prompt="p")
                )
            )
            await asyncio.wait_for(ended.wait(), 2)
            await upstream.send(
                msgspec.msgpack.encode(dict(type="event", kind="heartbeat", payload={}))
            )
            assert not upstream.finished.is_set()
            render.set()
            messages = [
                msgspec.msgpack.decode(await asyncio.wait_for(upstream.recv(), 2))
                for _ in range(3)
            ]
            assert [m["type"] for m in messages] == [
                "session.ready",
                "frame_batch",
                "session.completed",
            ]
            assert messages[-1]["frames"] == 1
        finally:
            render.set()
            await upstream.close()
            await archives.close()

    asyncio.run(run())


def test_playback_receipts_bound_delivery_without_blocking_controls(tmp_path):
    async def run():
        upstream, h3, _, archives = setup_pipeline(tmp_path, count=12)

        def receipt(frame):
            return msgspec.msgpack.encode(
                dict(
                    type="event",
                    kind="playback_ack",
                    payload=dict(protocol="whitebox-frame-v1", window=2, frame=frame),
                )
            )

        try:
            await upstream.send(
                msgspec.msgpack.encode(
                    dict(type="init", whitebox={"world_id": "a"}, prompt="p")
                )
            )
            ready = msgspec.msgpack.decode(await asyncio.wait_for(upstream.recv(), 2))
            assert ready["playback_flow_control"] == "whitebox-frame-v1"
            await upstream.send(receipt(-1))
            for index in range(2):
                frame = msgspec.msgpack.decode(
                    await asyncio.wait_for(upstream.recv(), 2)
                )
                assert frame["output_frame_start"] == index
            pending = asyncio.create_task(upstream.recv())
            await asyncio.sleep(0.02)
            assert not pending.done()
            # Flow control is transport-only; the browser reader remains live.
            await asyncio.wait_for(
                upstream.send(
                    msgspec.msgpack.encode(
                        dict(type="event", kind="heartbeat", payload={})
                    )
                ),
                1,
            )
            event_id = upstream.event_id
            await upstream.send(receipt(1))
            assert upstream.event_id == event_id
            frame = msgspec.msgpack.decode(await asyncio.wait_for(pending, 2))
            assert frame["output_frame_start"] == 2
            await upstream.send(receipt(2))
            for index in range(3, 12):
                frame = msgspec.msgpack.decode(
                    await asyncio.wait_for(upstream.recv(), 2)
                )
                assert frame["output_frame_start"] == index
                await upstream.send(receipt(index))
            assert (
                msgspec.msgpack.decode(await asyncio.wait_for(upstream.recv(), 2))[
                    "frames"
                ]
                == 12
            )
        finally:
            await upstream.close()
            await archives.close()

    asyncio.run(run())


def test_playback_receipt_stall_and_invalid_watermarks_are_bounded():
    from sglang.multimodal_gen.runtime.realtime.whitebox.playback_credit import (
        PlaybackCredit,
    )

    async def run():
        credit = PlaybackCredit(timeout=0.02)
        await credit.acknowledge(dict(window=1, frame=-1))
        await credit.reserve(0)
        with pytest.raises(TimeoutError):
            await credit.reserve(1)
        for payload in (
            dict(window=1, frame=1),
            dict(window=True, frame=0),
            dict(window=33, frame=0),
            dict(window=2, frame=0),
        ):
            with pytest.raises(ValueError):
                await credit.acknowledge(payload)
        await credit.acknowledge(dict(window=1, frame=0))
        await credit.acknowledge(dict(window=1, frame=-1))
        await credit.reserve(1)
        assert credit.received == 0 and credit.sent == 1

    asyncio.run(run())


def test_playback_bitrate_budget_bounds_noise_and_keeps_every_frame():
    import numpy as np

    spec = VideoSpec(832, 480)
    bitrate = 1_500_000
    encoder = H264PlaybackEncoder(spec, max_bitrate=bitrate)
    decoder = av.CodecContext.create("h264", "r")
    rng = np.random.default_rng(1234)
    size, decoded = 0, 0
    try:
        for index in range(48):
            wire = encoder.encode(
                index, rng.integers(0, 256, spec.frame_bytes, dtype=np.uint8).tobytes()
            )
            size += len(wire["payload"])
            frames = decoder.decode(av.Packet(wire["payload"]))
            assert len(frames) == 1 and (frames[0].width, frames[0].height) == (
                832,
                480,
            )
            decoded += len(frames)
        encoder.finish()
        assert decoded == 48
        # Two seconds plus the declared 250 ms VBV reservoir and mux tolerance.
        assert size * 8 <= bitrate * 2.25 * 1.1
        assert encoder.stream["max_bitrate"] == bitrate
    finally:
        encoder.close()


@pytest.mark.parametrize("fail", [False, True])
def test_slow_room_heartbeat_does_not_block_playback_receipts_and_is_owned(
    tmp_path, fail
):
    async def run():
        upstream, _, engine, archives = setup_pipeline(tmp_path, count=12)
        entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = 0

        async def touch():
            nonlocal calls
            calls += 1
            entered.set()
            try:
                await release.wait()
                if fail:
                    raise ValueError("Room ownership expired")
            except asyncio.CancelledError:
                cancelled.set()
                raise

        engine.touch = touch

        async def send(kind, payload):
            await asyncio.wait_for(
                upstream.send(
                    msgspec.msgpack.encode(
                        dict(type="event", kind=kind, payload=payload)
                    )
                ),
                0.5,
            )

        async def receipt(frame):
            await send(
                "playback_ack",
                dict(protocol="whitebox-frame-v1", window=1, frame=frame),
            )

        try:
            await upstream.send(
                msgspec.msgpack.encode(
                    dict(type="init", whitebox={"world_id": "a"}, prompt="p")
                )
            )
            await asyncio.wait_for(upstream.recv(), 2)
            await receipt(-1)
            assert (
                msgspec.msgpack.decode(await asyncio.wait_for(upstream.recv(), 2))[
                    "output_frame_start"
                ]
                == 0
            )
            await send("heartbeat", {})
            await asyncio.wait_for(entered.wait(), 1)
            for _ in range(20):
                await send("heartbeat", {})
            assert calls == 1
            await receipt(0)
            assert (
                msgspec.msgpack.decode(await asyncio.wait_for(upstream.recv(), 2))[
                    "output_frame_start"
                ]
                == 1
            )
            if fail:
                release.set()
                with pytest.raises(ValueError, match="ownership expired"):
                    await asyncio.wait_for(upstream.recv(), 1)
                assert upstream.outcome == "failed"
        finally:
            await upstream.close()
            await archives.close()
        assert all(task.done() for task in upstream.tasks)
        if not fail:
            assert cancelled.is_set()

    asyncio.run(run())
