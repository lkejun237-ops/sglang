"""RTC lifecycle tests without a real cloud, browser, or model allocation."""

import asyncio
from types import SimpleNamespace

import msgspec
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import (
    WhiteboxDiagnostic,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import RenderError
from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import (
    install_rtc_routes,
    reserve_rtc,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_transport import (
    RTCSubscription,
)


def test_cancel_before_ice_releases_startup_reservation_without_starting_world():
    events = []

    class Peer:
        def __init__(self, callback):
            self.connected = asyncio.Event()
            self.on_event = callback

        async def start(self):
            events.append("peer_started")

        async def close(self):
            events.append("peer_closed")

    class Directory:
        async def touch(self, *a, **kw):
            events.append(("route_ended", kw["end"]))

    async def attach(*a, **kw):
        events.append("world_started")
        raise AssertionError("The world must not start before ICE")

    host = SimpleNamespace(
        config=SimpleNamespace(spec=VideoSpec(32, 32)),
        runtime=SimpleNamespace(
            epoch="a" * 32, owner_id="room", directory=Directory(), rooms={}
        ),
        lock=asyncio.Lock(),
        sessions={},
        peer_sessions={},
        max_sessions=1,
        grace_seconds=15,
        attach=attach,
    )
    app = FastAPI()
    install_rtc_routes(host, app, "internal-only", peer_factory=Peer)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/internal/render-rtc", headers={"authorization": "Bearer internal-only"}
        ) as ws:
            ws.send_bytes(
                msgspec.msgpack.encode(
                    {
                        "user_id": "user",
                        "trace_id": "cancel",
                        "message": {
                            "type": "init",
                            "control_protocol": "whitebox-render-v2",
                            "source_operation": "create",
                            "whitebox": {"game_id": "b" * 32},
                        },
                    }
                )
            )
            configuration = msgspec.msgpack.decode(ws.receive_bytes())
            assert configuration["type"] == "rtc.configuration"
            assert getattr(host.runtime, "rtc_preparation", None)
            ws.send_bytes(
                msgspec.msgpack.encode(
                    {"type": "rtc.cancel", "peer_id": configuration["peer_id"]}
                )
            )
            value = msgspec.msgpack.decode(ws.receive_bytes())
            assert value["type"] == "rtc.cancelled"
            assert (
                "world_started" not in events
                and "peer_closed" in events
                and ("route_ended", True) in events
            )
            assert host.runtime.rtc_preparation is None


def test_cross_protocol_resume_cannot_replace_an_existing_source():
    async def scenario():
        p = SimpleNamespace(
            closed=False,
            terminal=None,
            media_transport="websocket",
            user_id="user",
            game_id="b" * 32,
            session=SimpleNamespace(source_epoch="s"),
            token="secret",
            subscription=None,
        )
        host = SimpleNamespace(
            runtime=SimpleNamespace(epoch="e"),
            lock=asyncio.Lock(),
            sessions={"a" * 32: p},
            peer_sessions={},
            max_sessions=1,
        )
        body = {
            "user_id": "user",
            "message": {
                "type": "resume",
                "control_protocol": "whitebox-render-v2",
                "session_id": "a" * 32,
                "whitebox": {"game_id": "b" * 32},
                "owner_epoch": "e",
                "source_epoch": "s",
                "resume_token": "secret",
            },
        }
        with pytest.raises(RenderError):
            await reserve_rtc(host, body)
        assert not getattr(host.runtime, "rtc_preparation", None)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "stop_reason, code, stage",
    [
        ("INVALID_CONTROL_MESSAGE", "INVALID_CONTROL_MESSAGE", "control"),
        ("RTC_CONTROL_OVERFLOW", "RTC_CONTROL_OVERFLOW", "control"),
        ("DISCONNECTED", "RTC_DISCONNECTED", "transport"),
    ],
)
def test_rtc_subscription_sends_without_any_initial_or_per_frame_ack(
    stop_reason, code, stage
):
    async def scenario():
        sent = []

        class Peer:
            on_event = None

            async def send_frame(self, m):
                sent.append(m["output_frame_start"])
                self.on_event(
                    {
                        "type": "frame.meta",
                        "track": "model",
                        "rtp_timestamp": len(sent) * 3000,
                        "frame": {k: v for k, v in m.items() if k != "payload"},
                    }
                )
                return {"rtp_timestamp": len(sent) * 3000}

            async def close(self):
                pass

        owner = SimpleNamespace(
            session=SimpleNamespace(
                source_epoch="s",
                current_model_stream="e",
                source_playback=None,
                current_segment=None,
                config=SimpleNamespace(
                    source_max_bitrate=1_000_000, playback_max_bitrate=2_500_000
                ),
            ),
            terminal=None,
            request_keyframe=lambda name: None,
            host=SimpleNamespace(
                config=SimpleNamespace(
                    source_max_bitrate=1_000_000, playback_max_bitrate=2_500_000
                )
            ),
            user_id="user",
        )
        sub = RTCSubscription(owner, 3, Peer())
        try:
            for i in range(12):
                sub.offer_media(
                    dict(
                        stream="model",
                        media_stream_id="e",
                        output_frame_start=i,
                        source_frame_index=i,
                        pts_us=i * 33333,
                        keyframe=i % 6 == 0,
                        payload=b"encoded",
                    )
                )
                meta = msgspec.msgpack.decode(await asyncio.wait_for(sub.recv(), 1))
                assert (
                    meta["type"] == "rtc.frame.meta" and "payload" not in meta["frame"]
                )
            assert sent == list(range(12))
            assert not hasattr(sub, "windows") and not hasattr(sub, "last_ack")
            with pytest.raises(ValueError):
                sub.ack({"frame": -1})
            sub.reset_model("new")
            sub.offer_media(
                dict(
                    stream="model",
                    media_stream_id="e",
                    output_frame_start=12,
                    keyframe=True,
                    payload=b"old",
                )
            )
            assert not sub.media["model"].items
            sub.stop(stop_reason)
            with pytest.raises(WhiteboxDiagnostic) as failure:
                await sub.recv()
            assert failure.value.code == code and failure.value.stage == stage
        finally:
            sub.stop("done")
            await sub.wait_closed()

    asyncio.run(scenario())


@pytest.mark.parametrize("complete_setup", ["attached", "timeout", "failed"])
def test_reconnect_reservation_survives_original_grace_during_bounded_ice(
    monkeypatch, complete_setup
):
    """A validated near-deadline resume must not lose its source during ICE."""
    from sglang.multimodal_gen.runtime.realtime.whitebox import rtc_routes
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import (
        RTCRenderPipeline,
        release_reconnect_preparation,
    )

    monkeypatch.setattr(rtc_routes, "RTC_RECONNECT_SETUP_SECONDS", 0.15)

    async def scenario():
        closed = []

        class Peer:
            async def close(self):
                pass

        class Subscription:
            def stop(self, reason):
                pass

            async def wait_closed(self):
                pass

        pipeline = object.__new__(RTCRenderPipeline)
        pipeline.media_transport = "webrtc-rtp"
        pipeline.user_id = "user"
        pipeline.game_id = "b" * 32
        pipeline.session = SimpleNamespace(
            source_epoch="source",
            config=SimpleNamespace(spec=VideoSpec(32, 32)),
            capture_rate_hint=lambda: 60,
        )
        pipeline.session_id = "a" * 32
        pipeline.token = "secret"
        pipeline.history_floor = pipeline.control_sequence = 0
        pipeline.subscription = Subscription()
        pipeline.retention = None
        pipeline.closed = False
        pipeline.terminal = None
        pipeline.rtc_peer = Peer()
        host = SimpleNamespace(
            runtime=SimpleNamespace(epoch="owner"),
            lock=asyncio.Lock(),
            sessions={pipeline.session_id: pipeline},
            grace_seconds=0.025,
        )
        pipeline.host = host

        async def close(reason):
            closed.append(reason)
            pipeline.closed = True
            host.sessions.pop(pipeline.session_id, None)

        pipeline.close = close
        pipeline.detach(pipeline.subscription)
        await asyncio.sleep(0.01)
        body = {
            "user_id": "user",
            "message": {
                "type": "resume",
                "control_protocol": "whitebox-render-v2",
                "session_id": pipeline.session_id,
                "whitebox": {"game_id": pipeline.game_id},
                "owner_epoch": "owner",
                "source_epoch": "source",
                "resume_token": "secret",
            },
        }
        try:
            reservation, source = await reserve_rtc(host, body)
            assert source is pipeline
            await asyncio.sleep(0.04)
            assert not closed, "Source expired while authenticated ICE was in progress"
            assert host.sessions[pipeline.session_id] is pipeline
            if complete_setup == "attached":
                # This is the ownership transition performed by attach().
                pipeline.subscription = Subscription()
                pipeline.retention.cancel()
                release_reconnect_preparation(pipeline, reservation)
                await asyncio.sleep(0.16)
                assert not closed
            elif complete_setup == "timeout":
                # No traffic or a stuck handshake cannot retain the source forever.
                await asyncio.sleep(0.16)
                assert closed == ["RECONNECT_EXPIRED"]
                with pytest.raises(RenderError):
                    await reserve_rtc(host, body)
            else:
                release_reconnect_preparation(pipeline, reservation)
                await asyncio.sleep(0.01)
                assert closed == ["RECONNECT_EXPIRED"]
                assert pipeline.rtc_reconnect is None
        finally:
            if pipeline.retention:
                pipeline.retention.cancel()
                await asyncio.gather(pipeline.retention, return_exceptions=True)

    asyncio.run(scenario())


def test_old_reconnect_cleanup_cannot_release_a_new_attempt():
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import (
        ReconnectPreparation,
        release_reconnect_preparation,
    )

    async def scenario():
        current = ReconnectPreparation("new", 1.0, asyncio.Event())
        pipeline = SimpleNamespace(rtc_reconnect=current)
        release_reconnect_preparation(pipeline, "old")
        assert pipeline.rtc_reconnect is current
        assert not current.released.is_set()
        release_reconnect_preparation(pipeline, "new")
        assert current.released.is_set() and pipeline.rtc_reconnect is None

    asyncio.run(scenario())


@pytest.mark.parametrize("finish_replay", [True, False])
def test_rtc_resume_streams_history_without_filling_live_control_queue(finish_replay):
    from collections import deque

    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import (
        RTCRenderPipeline,
    )

    async def scenario():
        class Peer:
            async def close(self):
                pass

        pipeline = object.__new__(RTCRenderPipeline)
        pipeline.closed = False
        pipeline.terminal = None
        pipeline.retention = pipeline.subscription = None
        pipeline.generation = 1
        pipeline.token = "resume-secret"
        pipeline.user_id = "user"
        pipeline.session_id = "a" * 32
        pipeline.game_id = "b" * 32
        pipeline.stream_id = "c" * 32
        pipeline.history_floor = 0
        pipeline.control_sequence = 600
        pipeline.committed_input = -1
        pipeline.history = deque()
        for index in range(1, 601):
            message = dict(type="whitebox.states", control_sequence=index)
            pipeline.history.append(
                (index, len(msgspec.msgpack.encode(message)), message)
            )
        pipeline.ready = dict(type="session.ready")
        pipeline.ready_message = lambda: dict(type="session.ready")
        pipeline.request_keyframe = lambda stream: None
        pipeline.rtc_peer = Peer()
        pipeline.session = SimpleNamespace(
            source_epoch="source",
            current_model_stream="model",
            current_segment=None,
            source_playback=None,
        )
        pipeline.host = SimpleNamespace(
            runtime=SimpleNamespace(epoch="owner"),
            history_bytes=16 * 1024**2,
            grace_seconds=15,
            config=SimpleNamespace(
                source_max_bitrate=1_000_000, playback_max_bitrate=2_500_000
            ),
        )
        sub = pipeline.attach(dict(resume_token=pipeline.token, control_sequence=0))
        try:
            assert not sub.closed, sub.reason
            assert len(sub.control) == 2  # Only attached/ready are live-queued.
            sub.offer_control(dict(type="whitebox.states", control_sequence=601))
            if not finish_replay:
                for _ in range(10):
                    await sub.recv()
                sub.stop("DISCONNECTED")
                assert not sub.replay and sub.replay_bytes == 0
                assert sub.replayed_controls == 8
                with pytest.raises(WhiteboxDiagnostic):
                    await sub.recv()
                return
            messages = [msgspec.msgpack.decode(await sub.recv()) for _ in range(603)]
            assert [v["type"] for v in messages[:2]] == [
                "session.attached",
                "session.ready",
            ]
            assert [v["control_sequence"] for v in messages[2:]] == list(range(1, 602))
            assert not sub.control and not sub.replay
            assert sub.replayed_controls == 600 and sub.replay_bytes == 0
        finally:
            sub.stop("done")
            await sub.wait_closed()

    asyncio.run(scenario())


def test_parallel_startup_begins_before_ice_and_cancel_closes_its_owner():
    events = []

    class Peer:
        def __init__(self, callback):
            self.connected = asyncio.Event()
            self.on_event = callback

        async def start(self):
            assert events == ["startup_attached"]
            events.append("peer_started")

        async def close(self):
            events.append("peer_closed")

    class Sub:
        def stop(self, reason):
            events.append("subscription_closed")

        async def wait_closed(self):
            pass

    class Pipeline:
        def __init__(self):
            self.rtc_closed = asyncio.Event()
            self.session = SimpleNamespace(
                startup=SimpleNamespace(rtc_ready=asyncio.Event())
            )

        async def close(self, reason):
            events.append("startup_closed")
            self.rtc_closed.set()

    async def attach(*args, **kwargs):
        events.append("startup_attached")
        return Pipeline(), Sub()

    async def touch(*args, **kwargs):
        raise AssertionError("The startup owner performs its own cleanup")

    host = SimpleNamespace(
        config=SimpleNamespace(spec=VideoSpec(32, 32)),
        runtime=SimpleNamespace(
            epoch="a" * 32,
            owner_id="room",
            directory=SimpleNamespace(touch=touch),
            rooms={},
        ),
        lock=asyncio.Lock(),
        sessions={},
        peer_sessions={},
        max_sessions=1,
        attach=attach,
    )
    app = FastAPI()
    install_rtc_routes(host, app, "internal-only", peer_factory=Peer)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/internal/render-rtc", headers={"authorization": "Bearer internal-only"}
        ) as ws:
            ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        user_id="u",
                        message=dict(
                            type="init",
                            control_protocol="whitebox-render-v2",
                            source_operation="create",
                            whitebox=dict(game_id="b" * 32),
                            startup=dict(protocol="whitebox-startup-prepare-v2"),
                        ),
                    )
                )
            )
            config = msgspec.msgpack.decode(ws.receive_bytes())
            assert config["type"] == "rtc.configuration"
            ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(type="rtc.cancel", peer_id=config["peer_id"])
                )
            )
            assert msgspec.msgpack.decode(ws.receive_bytes())["type"] == "rtc.cancelled"
            assert (
                events.count("startup_attached") == events.count("startup_closed") == 1
            )
            assert "peer_closed" in events and host.runtime.rtc_preparation is None


@pytest.mark.parametrize("provided_frame", [False, True])
def test_early_real_subscription_preserves_ice_callback_then_takes_media(
    provided_frame,
):
    """The real subscription replaces on_event; ICE must still reach the wire."""
    instances = []

    class Peer:
        def __init__(self, callback):
            self.on_event = callback
            self.connected = asyncio.Event()

        async def start(self):
            pass

        async def offer(self, sdp):
            assert sdp == "browser-offer"
            self.on_event(
                {"type": "candidate", "candidate": {"candidate": "server-ice"}}
            )
            return {"sdp": "server-answer", "tracks": {"whitebox": "0", "model": "1"}}

        async def candidate(self, candidate):
            assert candidate == {"candidate": "browser-saw-server-ice"}
            self.connected.set()

        def _check(self):
            pass

        async def close(self):
            pass

    class Pipeline:
        def __init__(self, peer):
            self.rtc_closed = asyncio.Event()
            self.session = SimpleNamespace(
                startup=SimpleNamespace(rtc_ready=asyncio.Event()),
                source_epoch="source",
                current_model_stream=None,
                source_playback=None,
                current_segment=None,
            )
            self.host = SimpleNamespace(
                config=SimpleNamespace(
                    source_max_bitrate=1_000_000,
                    playback_max_bitrate=2_500_000,
                )
            )
            self.terminal = None
            self.user_id = "user"
            self.rtc_peer = peer
            self.subscription = RTCSubscription(self, 1, peer)
            self.subscription.offer_control({"type": "startup.accepted"})

        def request_keyframe(self, name):
            pass

        def accept(self, sub, wire):
            assert sub is self.subscription
            assert msgspec.msgpack.decode(wire)["type"] == "probe.after_ice"
            self.rtc_peer.on_event({"type": "state", "state": "connected"})

        async def close(self, reason):
            self.subscription.stop(reason)
            await self.subscription.wait_closed()
            self.rtc_closed.set()

    async def attach(body, *, rtc_peer, **kwargs):
        pipeline = Pipeline(rtc_peer)
        instances.append(pipeline)
        return pipeline, pipeline.subscription

    host = SimpleNamespace(
        config=SimpleNamespace(spec=VideoSpec(32, 32)),
        runtime=SimpleNamespace(epoch="a" * 32, owner_id="room", rooms={}),
        lock=asyncio.Lock(),
        sessions={},
        peer_sessions={},
        max_sessions=1,
        attach=attach,
    )
    app = FastAPI()
    install_rtc_routes(host, app, "internal-only", peer_factory=Peer)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/internal/render-rtc", headers={"authorization": "Bearer internal-only"}
        ) as ws:
            startup = {"protocol": "whitebox-startup-prepare-v2"}
            if provided_frame:
                startup["initial_frame"] = {"id": "source-s0"}
            ws.send_bytes(
                msgspec.msgpack.encode(
                    {
                        "user_id": "user",
                        "message": {
                            "type": "init",
                            "control_protocol": "whitebox-render-v2",
                            "source_operation": "create",
                            "whitebox": {"game_id": "b" * 32},
                            "startup": startup,
                        },
                    }
                )
            )
            config = msgspec.msgpack.decode(ws.receive_bytes())
            assert config["type"] == "rtc.configuration"
            peer_id = config["peer_id"]
            ws.send_bytes(
                msgspec.msgpack.encode(
                    {
                        "type": "rtc.offer",
                        "peer_id": peer_id,
                        "sdp": "browser-offer",
                    }
                )
            )
            candidate = msgspec.msgpack.decode(ws.receive_bytes())
            assert candidate == {
                "type": "rtc.candidate",
                "candidate": {"candidate": "server-ice"},
                "peer_id": peer_id,
            }
            assert msgspec.msgpack.decode(ws.receive_bytes())["type"] == "rtc.answer"
            ws.send_bytes(
                msgspec.msgpack.encode(
                    {
                        "type": "rtc.candidate",
                        "peer_id": peer_id,
                        "candidate": {"candidate": "browser-saw-server-ice"},
                    }
                )
            )
            assert (
                msgspec.msgpack.decode(ws.receive_bytes())["type"] == "startup.accepted"
            )
            pipeline = instances[0]
            assert pipeline.session.startup.rtc_ready.is_set()
            assert pipeline.rtc_peer.on_event == pipeline.subscription.native_event
            ws.send_bytes(msgspec.msgpack.encode({"type": "probe.after_ice"}))
            state = msgspec.msgpack.decode(ws.receive_bytes())
            assert state["type"] == "rtc.state" and state["state"] == "connected"
            ws.send_bytes(
                msgspec.msgpack.encode({"type": "rtc.cancel", "peer_id": peer_id})
            )
            assert msgspec.msgpack.decode(ws.receive_bytes())["type"] == "rtc.cancelled"
            assert pipeline.rtc_closed.is_set()
            assert host.runtime.rtc_preparation is None


@pytest.mark.parametrize("late_consumer", [False, True])
@pytest.mark.parametrize("plan", [False, True])
@pytest.mark.parametrize("late_protocol", [False, True])
def test_managed_rtc_prewarm_capability_requires_v2_opt_in(
    monkeypatch, late_consumer, plan, late_protocol
):
    monkeypatch.setenv("WHITEBOX_MODEL_CONTEXT_PREWARM", "1")
    source_identity = dict(source_session_id="source", source_epoch="epoch")
    pipelines = []
    channel = SimpleNamespace(
        view=SimpleNamespace(client_id="consumer"),
        prewarm_supported=not plan and not late_protocol,
        prewarm_plan_supported=plan and not late_protocol,
    )

    class Peer:
        def __init__(self, callback):
            self.connected = asyncio.Event()
            self.on_event = callback

        async def start(self):
            self.connected.set()

        def _check(self):
            pass

        async def close(self):
            pass

    class Sub:
        generation = 1
        closed = False

        def __init__(self, session):
            self.queue = asyncio.Queue()
            self.session = session
            for kind in ("session.attached", "session.ready"):
                self.offer_control(
                    dict(type=kind, connection_generation=1, **source_identity)
                )

        def offer_control(self, value, *, optional=False):
            assert optional is (
                value["type"]
                in {
                    "model.context.prewarm.observation",
                    "model.context.prewarm.plan.observation",
                }
            )
            self.queue.put_nowait(msgspec.msgpack.encode(value))

        async def recv(self):
            wire = await self.queue.get()
            if msgspec.msgpack.decode(wire)["type"] == "session.ready":
                self.session.context_channel = channel
            return wire

        def stop(self, _):
            self.closed = True

        async def wait_closed(self):
            pass

    class Pipeline:
        single_player_operation = "managed-single"

        def __init__(self):
            self.rtc_closed = asyncio.Event()
            self.session = SimpleNamespace(
                startup=None,
                inference=SimpleNamespace(manual=True),
                source_identity=lambda: source_identity,
                context_channel=None if late_consumer else channel,
            )
            self.subscription = Sub(self.session)

        def accept(self, sub, wire):
            message = msgspec.msgpack.decode(wire)
            if message["type"] == "fixture.negotiate":
                channel.prewarm_supported = not plan
                channel.prewarm_plan_supported = plan
                sub.offer_control(
                    {
                        "type": "source.status",
                        "connection_generation": sub.generation,
                        **source_identity,
                    }
                )
                return
            store = getattr(
                self.session,
                "prewarm_plan_observations" if plan else "prewarm_observations",
            )
            observe = getattr(
                self.session,
                "prewarm_plan_observer" if plan else "prewarm_observer",
                None,
            )
            window = SimpleNamespace(window_id="window", opening_revision=1)

            def record(status):
                if plan:
                    return store.record(
                        window,
                        {
                            "type": "model.context.prewarm.plan",
                            "phase": status,
                            "decisionRevision": 1,
                            "planId": "plan",
                            "planRevision": 1,
                            "id": "night",
                            "candidateRevision": 1,
                            "requestId": "prefill",
                        },
                    )
                return store.record(
                    "window",
                    1,
                    status,
                    intent_id="intent",
                    candidate_revision=1,
                    request_id="prefill",
                )

            if message["type"] == "fixture.ready":
                assert observe is None
                opened = (
                    store.record(window, {"type": "model.context.prewarm.plan.window"})
                    if plan
                    else store.open("window", 1)
                )
                assert opened is not None
                assert record("memory-ready")
            elif message["type"] == "fixture.persist":
                channel.prewarm_supported = channel.prewarm_plan_supported = False
                observe(record("persisted"))
            sub.offer_control(dict(type="fixture.pong"))

        async def close(self, _):
            self.rtc_closed.set()

        def detach(self, sub):
            self.subscription = None
            sub.stop("DISCONNECTED")

    async def attach(*args, **kwargs):
        pipeline = Pipeline()
        pipelines.append(pipeline)
        return pipeline, pipeline.subscription

    host = SimpleNamespace(
        config=SimpleNamespace(spec=VideoSpec(32, 32)),
        runtime=SimpleNamespace(epoch="a" * 32, owner_id="room", rooms={}),
        lock=asyncio.Lock(),
        sessions={},
        peer_sessions={},
        max_sessions=1,
        attach=attach,
    )
    app = FastAPI()
    install_rtc_routes(host, app, "internal-only", peer_factory=Peer)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/internal/render-rtc", headers={"authorization": "Bearer internal-only"}
        ) as ws:

            def send(**value):
                ws.send_bytes(msgspec.msgpack.encode(value))

            def receive():
                return msgspec.msgpack.decode(ws.receive_bytes())

            send(
                user_id="u",
                message=dict(
                    type="init",
                    control_protocol="whitebox-render-v2",
                    source_operation="create.manual",
                    inference_control="manual",
                    whitebox=dict(game_id="b" * 32),
                ),
            )
            configuration = receive()
            assert configuration["type"] == "rtc.configuration"
            attached, ready = receive(), receive()
            supported, unchanged = (
                (ready, attached) if late_consumer else (attached, ready)
            )
            assert "model_prewarm_observation" not in unchanged
            if late_protocol:
                assert "model_prewarm_observation" not in supported
                assert "model_prewarm_plan_observation" not in supported
                send(type="fixture.negotiate")
                supported = receive()
                assert supported["type"] == "source.status"
            field = (
                "model_prewarm_plan_observation"
                if plan
                else "model_prewarm_observation"
            )
            other = (
                "model_prewarm_observation"
                if plan
                else "model_prewarm_plan_observation"
            )
            assert supported[field] == (
                "worldkit-model-prewarm-plan-v1"
                if plan
                else "worldkit-model-prewarm-v1"
            )
            assert other not in supported
            assert supported["model_prewarm_consumer_id"] == "consumer"
            if plan:
                assert not hasattr(pipelines[0].session, "prewarm_observations")
            observe_type = (
                "model.context.prewarm.plan.observe"
                if plan
                else "model.context.prewarm.observe"
            )
            send(type="fixture.ready")
            assert receive() == dict(type="fixture.pong")
            send(type=observe_type, version=2, consumer_id="foreign")
            send(type="fixture.ping")
            assert receive() == dict(type="fixture.pong")
            send(type=observe_type, version=2)
            assert receive()["status"] == "window"
            prepared = receive()
            assert prepared["status"] == "memory-ready"
            assert prepared["source_session_id"] == "source"
            assert prepared["source_epoch"] == "epoch"
            assert prepared["connection_generation"] == 1
            send(type="fixture.persist")
            assert receive()["status"] == "persisted"
            assert receive() == dict(type="fixture.pong")
            send(type="rtc.cancel", peer_id=configuration["peer_id"])
            assert receive()["type"] == "rtc.cancelled"
    assert (
        getattr(
            pipelines[0].session,
            "prewarm_plan_observer" if plan else "prewarm_observer",
        )
        is None
    )
