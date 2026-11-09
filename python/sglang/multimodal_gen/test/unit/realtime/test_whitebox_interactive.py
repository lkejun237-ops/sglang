# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import av
import msgspec
import pytest
from test_whitebox import H3

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
    EngineFrame,
    VideoSpec,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import (
    WhiteboxDiagnostic,
    public_error,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
    InteractiveSession,
    LatestFrame,
    ModelLeases,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.recording import ArchiveManager
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig


class BrowserUI:
    metadata = {"mode": "browser", "files": {}}

    async def close(self):
        pass


class Engine:
    def __init__(self, spec):
        self.spec = spec
        self.queue = asyncio.Queue()
        self.closed = False
        self.inputs = []
        self.next = 0

    async def frames(self):
        while (frame := await self.queue.get()) is not None:
            yield frame

    async def push(self):
        index = self.next
        self.next += 1
        await self.queue.put(
            EngineFrame(
                index,
                self.spec.pts(index),
                bytes([30]) * self.spec.frame_bytes,
                {"opaque": index},
                effective_input={"keys": list(self.inputs)},
                provenance={"capture_pts_us": index * 50000},
            )
        )

    async def send_input(self, seq, payload):
        self.inputs.append(payload)

    async def finish(self):
        await self.queue.put(None)

    async def close(self):
        self.closed = True


class Worker(H3):
    source_capacity = 34

    async def emit(self, message):
        if message["type"] == "session.ready":
            message.update(
                max_pending_source_frames=self.source_capacity,
                max_pending_source_bytes=self.spec.frame_bytes * self.source_capacity,
            )
        await super().emit(message)


class Leases:
    def __init__(self, spec):
        self.spec, self.opens, self.active = spec, [], 0
        self.fail = False

    @asynccontextmanager
    async def open(self):
        if self.fail:
            raise RuntimeError("No capacity")
        self.active += 1
        assert self.active == 1
        worker = Worker(self.spec)
        self.opens.append(worker)
        try:
            yield (
                worker,
                SimpleNamespace(
                    session_id=f"s{len(self.opens)}",
                    generation_id=f"g{len(self.opens)}",
                    denoiser=SimpleNamespace(
                        worker_id="exclusive", model_revision="test"
                    ),
                ),
            )
        finally:
            await worker.close()
            self.active -= 1


async def receive_until(session, predicate):
    async with asyncio.timeout(5):
        while True:
            message = msgspec.msgpack.decode(await session.recv())
            if message["type"] in ("frame_batch", "whitebox.frame"):
                session.media_delivered()
            if predicate(message):
                return message


async def setup(tmp_path, processes=False):
    spec = VideoSpec(32, 32)
    engine, leases, archives = (
        Engine(spec),
        Leases(spec),
        ArchiveManager(processes=processes),
    )

    async def factory(**_):
        return engine

    session = InteractiveSession(
        leases=leases,
        config=WhiteboxConfig(
            engine_factory=factory,
            ui_factory=lambda **_: BrowserUI(),
            spool=tmp_path,
            spec=spec,
            startup_seconds=5,
            play_seconds=20,
            drain_seconds=5,
        ),
        archives=archives,
        session_id="outer",
        trace_id="test-switch",
    )
    await session.send(
        msgspec.msgpack.encode(
            dict(type="init", whitebox={"world_id": "world"}, prompt="first")
        )
    )
    await receive_until(
        session, lambda m: m["type"] == "render.status" and m["state"] == "active"
    )
    return session, engine, leases, archives


async def switch(session, mode):
    payload = dict(
        request_id=f"r{session.revision + 1}",
        expected_revision=session.revision,
        mode=mode,
    )
    await session.configure(payload)
    await receive_until(session, lambda m: m.get("state") == "accepted")
    return payload


def test_ui_action_receipts_remain_distinct_and_survive_superseded_frames(tmp_path):
    async def run():
        session, engine, _leases, archives = await setup(tmp_path)
        try:
            actions = [
                dict(type="action.receipt", epoch=0, actionId=name, status="completed")
                for name in ("gate", "select", "configure")
            ]
            keyboard = dict(type="input.receipt", epoch=0, clientSequence=0)
            for index, receipts in enumerate(([keyboard, *actions], [])):
                await engine.queue.put(
                    EngineFrame(
                        index,
                        engine.spec.pts(index),
                        bytes([30]) * engine.spec.frame_bytes,
                        {},
                        observed_input={"receipts": receipts},
                    )
                )
                state = await receive_until(
                    session, lambda m: m["type"] == "whitebox.states"
                )
                assert state["observed_input"]["receipts"] == [keyboard, *actions]
        finally:
            await session.close()
            await archives.close()

    asyncio.run(run())


@pytest.mark.parametrize("processes", [False, True])
def test_abab_only_changes_display_keeps_model_and_archive_continuous(
    tmp_path, processes
):
    async def run():
        session, engine, leases, archives = await setup(tmp_path, processes)
        try:
            assert leases.active == 1 and len(leases.opens) == 1
            epoch = session.epoch
            for mode in ("whitebox", "model", "whitebox", "model"):
                request = await switch(session, mode)
                await session.configure(request)
                await receive_until(session, lambda m: m.get("duplicate"))
                await engine.push()
                frame = await receive_until(
                    session, lambda m: m["type"] == "frame_batch"
                )
                assert frame["rendering"]["mode"] == "model"
                assert frame["rendering"]["model_epoch"] == epoch
                assert frame["observed_input"]["display"]["mode"] == mode
                assert leases.active == 1 and len(leases.opens) == 1
                assert not engine.closed
            await session.finish()
            result = await receive_until(
                session, lambda m: m["type"] == "session.completed"
            )
            assert result["frames"] == result["source_frames"] == 4
        finally:
            await session.close()
            await archives.close()
        assert leases.active == 0 and engine.closed
        directory = session.recording.directory
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["recording_status"] == "complete"
        assert manifest["rendering"]["model_video"] == "continuous_model"
        rows = [
            json.loads(line)
            for line in (directory / "inputs.jsonl").read_text().splitlines()
        ]
        assert all(row["input"]["rendering"]["mode"] == "model" for row in rows)
        assert len({row["input"]["rendering"]["model_epoch"] for row in rows}) == 1
        timelines = []
        for name in ("whitebox", "model"):
            with av.open(str(directory / f"{name}.mp4")) as video:
                timelines.append(
                    [frame.pts * frame.time_base for frame in video.decode(video=0)]
                )
        assert timelines[0] == timelines[1] and len(timelines[0]) == 4

    asyncio.run(run())


def test_latest_slot_skips_unsubmitted_frames_without_unbounded_queue():
    async def run():
        latest = LatestFrame()
        for i in range(10000):
            await latest.put(i)
        assert await latest.get(0) == (10000, 9999)
        await latest.close()
        assert await latest.get(10000) is None

    asyncio.run(run())


def test_slow_display_does_not_stall_model_or_drop_archived_frames(tmp_path):
    async def run():
        session, engine, leases, archives = await setup(tmp_path)
        frames = []
        completed = asyncio.Event()

        async def drain():
            while True:
                message = msgspec.msgpack.decode(await session.recv())
                if message["type"] == "frame_batch":
                    frames.append(message)
                    session.media_delivered()
                if message["type"] == "session.completed":
                    assert message["recorded_frames"] == 20
                    assert message["frames"] == 2
                    completed.set()
                    return

        await session.playback_credit.acknowledge(dict(window=1, frame=-1))
        reader = asyncio.create_task(drain())
        try:
            await engine.push()
            async with asyncio.timeout(5):
                while not frames:
                    await asyncio.sleep(0.001)
                for _ in range(19):
                    await engine.push()
                    # Keep each source until H3 has consumed it; only display is slow.
                    while session.received < engine.next:
                        await asyncio.sleep(0.001)
            assert len(frames) == 1
            assert leases.active == 1 and len(leases.opens) == 1
            await session.playback_credit.acknowledge(dict(window=1, frame=0))
            await session.finish()
            await asyncio.wait_for(completed.wait(), 5)
            assert [f["recording_frame_index"] for f in frames] == [0, 19]
            assert all(f["keyframe"] for f in frames)
            assert session.model_preview_dropped == 18
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            await session.close()
            await archives.close()
        for name in ("whitebox", "model"):
            with av.open(str(session.recording.directory / f"{name}.mp4")) as video:
                assert sum(1 for _ in video.decode(video=0)) == 20

    asyncio.run(run())


def test_display_rejects_stale_revision_reset_and_model_parameters(tmp_path):
    async def run():
        session, engine, leases, archives = await setup(tmp_path)
        try:
            for patch, code in (
                ({"expected_revision": 99}, "REVISION_CONFLICT"),
                ({"reset": True}, "INVALID_CONFIGURATION"),
                ({"prompt": "new"}, "INVALID_CONFIGURATION"),
                ({"references": []}, "INVALID_CONFIGURATION"),
            ):
                await session.configure(
                    {
                        "request_id": str(patch),
                        "expected_revision": 0,
                        "mode": "whitebox",
                        **patch,
                    }
                )
                rejected = await receive_until(
                    session, lambda m: m.get("state") == "rejected"
                )
                assert rejected["code"] == code
                assert (
                    session.revision == 0
                    and leases.active == 1
                    and len(leases.opens) == 1
                )
            await engine.push()
            await receive_until(session, lambda m: m["type"] == "frame_batch")
        finally:
            await session.close()
            await archives.close()

    asyncio.run(run())


def test_two_users_and_cancelled_model_admission_are_isolated(tmp_path):
    async def run():
        a, ea, la, aa = await setup(tmp_path / "a")
        b, eb, lb, ab = await setup(tmp_path / "b")
        try:
            await asyncio.gather(switch(a, "model"), switch(b, "model"))
            await asyncio.gather(ea.push(), eb.push())
            ma, mb = await asyncio.gather(
                receive_until(a, lambda m: m["type"] == "frame_batch"),
                receive_until(b, lambda m: m["type"] == "frame_batch"),
            )
            assert ma["rendering"]["model_epoch"] != mb["rendering"]["model_epoch"]
            await switch(a, "whitebox")
            assert la.active == lb.active == 1
            await eb.push()
            assert (await receive_until(b, lambda m: m["type"] == "frame_batch"))[
                "rendering"
            ]["revision"] == 0
        finally:
            await asyncio.gather(a.close(), b.close())
            await asyncio.gather(aa.close(), ab.close())
        entered, finish = asyncio.Event(), asyncio.Event()
        released = []

        class Coordinator:
            async def admit(self, **_):
                entered.set()
                await finish.wait()
                return SimpleNamespace(vae=None)

            async def release(self, assignment):
                released.append(assignment)

        leases = ModelLeases(
            Coordinator(),
            None,
            user_id="user",
            model_revision="model",
            vae_fingerprint="local",
            trace_id="trace",
            max_bytes=100,
        )

        async def allocate():
            async with leases.open():
                pytest.fail("Cancelled admission must never connect.")

        task = asyncio.create_task(allocate())
        await entered.wait()
        task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(released) == 1

    asyncio.run(run())


def test_real_webrtc_unordered_datachannel_carries_decodable_h264(monkeypatch):
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

    from sglang.multimodal_gen.runtime.realtime.whitebox.playback import (
        H264PlaybackEncoder,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.webrtc import (
        HEADER,
        WebRTCMedia,
    )

    monkeypatch.delenv("WHITEBOX_RTC_ICE_SERVERS", raising=False)

    async def run():
        server = WebRTCMedia()
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        channel = peer.createDataChannel(
            "whitebox-media-v1", ordered=False, maxRetransmits=0
        )
        parts, received = {}, asyncio.Event()

        @channel.on("message")
        def receive(wire):
            magic, _, index, count, _ = HEADER.unpack(wire[: HEADER.size])
            assert magic == b"WBV1"
            parts[index] = wire[HEADER.size :]
            if len(parts) == count:
                received.set()

        codec = H264PlaybackEncoder(VideoSpec(32, 32))
        codec.independent_frames = True
        try:
            await peer.setLocalDescription(await peer.createOffer())
            answer = await server.answer(peer.localDescription.sdp)
            await peer.setRemoteDescription(
                RTCSessionDescription(type="answer", sdp=answer)
            )
            await server.wait_connected()
            encoded = codec.encode(0, bytes([80]) * 32 * 32 * 3)
            assert await server.send(
                dict(type="frame_batch", **encoded, states={"frame": 17})
            )
            await asyncio.wait_for(received.wait(), 3)
            message = msgspec.msgpack.decode(b"".join(parts[i] for i in sorted(parts)))
            assert message["states"] == {"frame": 17}
            decoder = av.CodecContext.create("h264", "r")
            decoded = decoder.decode(av.Packet(message["payload"]))
            assert len(decoded) == 1 and decoded[0].width == 32
        finally:
            codec.close()
            await peer.close()
            await server.close()

    asyncio.run(run())


def test_websocket_gateway_roundtrip_and_slot_reuse(tmp_path):
    from urllib.parse import parse_qs, urlsplit

    from test_realtime_coordinator import _heartbeat
    from test_whitebox_rooms import server
    from websockets.asyncio.client import connect

    from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
        create_app,
    )
    from sglang.multimodal_gen.runtime.realtime.coordinator import (
        InMemoryCoordinatorStore,
        RealtimeCoordinator,
    )

    async def run():
        spec = VideoSpec(32, 32)
        engine = Engine(spec)
        opened = []

        async def factory(**_):
            return engine

        async def connection(url, **_):
            opened.append(parse_qs(urlsplit(url).query))
            return Worker(spec)

        class Client(RealtimeCoordinator):
            async def admit(self, trace_id=None, **kwargs):
                return await super().admit(**kwargs)

        coor = Client(
            InMemoryCoordinatorStore(ttl_s=10, worker_ttl_s=30), wait_timeout_s=0
        )
        await coor.heartbeat(
            _heartbeat("h3", "denoiser", model_revision="h3", vae_fingerprint="local")
        )
        archives = ArchiveManager()
        app = create_app(
            coor,
            model_revision="h3",
            vae_fingerprint="local",
            internal_output_url="ws://unused",
            connect_factory=connection,
            whitebox_config=WhiteboxConfig(
                engine_factory=factory,
                ui_factory=lambda **_: BrowserUI(),
                spool=tmp_path,
                spec=spec,
                startup_seconds=8,
                drain_seconds=5,
            ),
            whitebox_archives=archives,
        )
        async with server(app) as base:
            async with connect(
                base.replace("http", "ws")
                + "/v1/realtime_video/generate?whitebox_control=v1&user_id=test-user"
            ) as ws:
                await ws.send(
                    msgspec.msgpack.encode(
                        dict(type="init", whitebox={"world_id": "world"}, prompt="p")
                    )
                )

                async def until(kind, **fields):
                    async with asyncio.timeout(5):
                        while True:
                            m = msgspec.msgpack.decode(await ws.recv())
                            if m["type"] == "error":
                                raise AssertionError(m)
                            if m["type"] == kind and all(
                                m.get(k) == v for k, v in fields.items()
                            ):
                                return m

                await until("session.ready")
                await until("render.status", state="active")
                assert len(opened) == 1
                for rev, mode in enumerate(("model", "whitebox", "model"), 1):
                    await ws.send(
                        msgspec.msgpack.encode(
                            dict(
                                type="event",
                                kind="display.configure",
                                payload=dict(
                                    request_id=str(rev),
                                    expected_revision=rev - 1,
                                    mode=mode,
                                ),
                            )
                        )
                    )
                    await until("render.status", state="accepted", revision=rev)
                    await engine.push()
                    m = await until("frame_batch")
                    assert m["rendering"]["mode"] == "model"
                assert len(opened) == 1
                await ws.send(
                    msgspec.msgpack.encode(
                        dict(type="event", kind="finish", payload={})
                    )
                )
                await until("session.completed")
        assert engine.closed

    asyncio.run(run())


def test_model_failure_does_not_readmit_or_silently_bypass(tmp_path, caplog):
    async def run():
        session, engine, leases, archives = await setup(tmp_path)
        try:
            await leases.opens[0].emit(
                dict(
                    type="error",
                    error=dict(code="GPU_MEMORY", message="private GPU details"),
                )
            )
            with pytest.raises(WhiteboxDiagnostic) as failure:
                await receive_until(session, lambda m: m["type"] == "frame_batch")
            diagnostic = public_error(failure.value, "test-switch")
            assert diagnostic["code"] == "MODEL_FAILED"
            assert diagnostic["stage"] == "model"
            assert diagnostic["upstream_code"] == "GPU_MEMORY"
            assert "private" not in str(diagnostic)
            assert "private GPU details" in caplog.text
            assert len(leases.opens) == 1
        finally:
            await session.close()
            await archives.close()
        assert leases.active == 0 and engine.closed

    asyncio.run(run())


def test_global_coordinator_slots_are_exclusive_across_interactive_users():
    from test_realtime_coordinator import _heartbeat

    from sglang.multimodal_gen.runtime.realtime.coordinator import (
        InMemoryCoordinatorStore,
        RealtimeCoordinator,
    )

    class Client(RealtimeCoordinator):
        async def admit(self, trace_id=None, **kwargs):
            return await super().admit(**kwargs)

    async def run():
        coor = Client(
            InMemoryCoordinatorStore(ttl_s=30, worker_ttl_s=60), wait_timeout_s=0
        )
        for worker in ["one", "two"]:
            await coor.heartbeat(
                _heartbeat(
                    worker, "denoiser", model_revision="h3", vae_fingerprint="local"
                )
            )

        async def connect(*args, **kwargs):
            return Worker(VideoSpec(32, 32))

        def owner(user):
            return ModelLeases(
                coor,
                connect,
                user_id=user,
                model_revision="h3",
                vae_fingerprint="local",
                trace_id=user,
                max_bytes=1048576,
            )

        a, b, c = owner("a"), owner("b"), owner("c")
        async with a.open() as (_, aa), b.open() as (_, bb):
            assert aa.denoiser.worker_id != bb.denoiser.worker_id
            with pytest.raises(Exception) as error:
                async with c.open():
                    pytest.fail("A third user must not share a busy model slot.")
            assert getattr(error.value, "reason", None) == "CAPACITY_EXHAUSTED"
        async with c.open() as (_, cc):
            assert cc.session_id not in [aa.session_id, bb.session_id]

    asyncio.run(run())


def test_model_heartbeat_survives_full_source_window_and_stops_on_close(
    tmp_path, monkeypatch
):
    from sglang.multimodal_gen.runtime.realtime.whitebox import interactive

    class ColdWorker(Worker):
        def __init__(self, spec):
            super().__init__(spec)
            self.held = []
            self.hold = True
            self.heartbeats = []

        async def emit(self, message):
            if message["type"] == "frame_batch" and self.hold:
                self.held.append(message)
            else:
                await super().emit(message)

        async def send(self, wire):
            message = msgspec.msgpack.decode(wire)
            if message.get("kind") == "heartbeat":
                self.heartbeats.append(message)
            else:
                await super().send(wire)

    monkeypatch.setattr(__import__(__name__), "Worker", ColdWorker)
    monkeypatch.setattr(interactive, "MODEL_HEARTBEAT_SECONDS", 0.01)

    async def run():
        session, engine, leases, archives = await setup(tmp_path)
        worker = leases.opens[0]
        completed = asyncio.Event()

        async def drain():
            while True:
                message = msgspec.msgpack.decode(await session.recv())
                if message["type"] in ("frame_batch", "whitebox.frame"):
                    session.media_delivered()
                if message["type"] == "session.completed":
                    completed.set()
                    return

        reader = asyncio.create_task(drain())
        try:
            async with asyncio.timeout(5):
                for index in range(worker.source_capacity):
                    await engine.push()
                    while len(worker.held) <= index:
                        await asyncio.sleep(0.001)
                await engine.push()
                baseline = len(worker.heartbeats)
                while len(worker.heartbeats) < baseline + 2:
                    await asyncio.sleep(0.001)
                assert len(worker.held) == worker.source_capacity
                assert session.received == 0  # No RGB output and no input credit.
                ids = [e["event_id"] for e in worker.events + worker.heartbeats]
                assert len(ids) == len(set(ids))
                assert all(e["payload"] == {} for e in worker.heartbeats)
                worker.hold = False
                for frame in worker.held:
                    await Worker.emit(worker, frame)
                while session.received != engine.next:
                    await asyncio.sleep(0.001)
            await session.finish()
            await asyncio.wait_for(completed.wait(), 5)
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            await session.close()
            await archives.close()
        count = len(worker.heartbeats)
        await asyncio.sleep(0.04)
        assert len(worker.heartbeats) == count and leases.active == 0

    asyncio.run(run())


def test_source_and_model_keep_same_capture_clock_and_separate_admission_wait(tmp_path):
    async def run():
        session, engine, _leases, archives = await setup(tmp_path)
        try:
            await engine.push()
            media = {}
            async with asyncio.timeout(5):
                while len(media) < 2:
                    message = msgspec.msgpack.decode(await session.recv())
                    if message["type"] in ("frame_batch", "whitebox.frame"):
                        session.media_delivered()
                        media[message["stream"]] = message
            source, model = media["whitebox"], media["model"]
            assert source["source_frame_index"] == model["source_frame_index"]
            assert (
                source["source_capture_mono_ms"] == model["source_capture_mono_ms"] > 0
            )
            timing = model["source"]["model_timing"]
            assert timing["input_admission_wait_ms"] >= 0
            assert set(timing["input_capacity_blocked_ms"]) == {
                "pair_capacity",
                "encoder_credit",
                "output_window",
            }
            assert all(
                value >= 0 for value in timing["input_capacity_blocked_ms"].values()
            )
            assert (
                abs(
                    timing["source_to_raw_output_ms"]
                    - timing["source_to_selection_ms"]
                    - timing["selection_to_dispatch_ms"]
                    - timing["dispatch_to_raw_output_ms"]
                )
                < 0.001
            )
        finally:
            await session.close()
            await archives.close()

    asyncio.run(run())
