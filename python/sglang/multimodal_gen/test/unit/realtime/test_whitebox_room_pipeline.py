# SPDX-License-Identifier: Apache-2.0
import asyncio
import heapq
from types import SimpleNamespace

import msgspec
import pytest
from test_whitebox_interactive import BrowserUI, Engine, Leases

from sglang.multimodal_gen.runtime.realtime.whitebox import room_pipeline
from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
    AsyncArchiveManager,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports
from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
    RoomPipelines,
    Subscription,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
    new_id,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig


async def setup(tmp_path, grace=0.2, gop=6, **config_overrides):
    spec = VideoSpec(32, 32)
    engine = Engine(spec)
    engine.renderer = BrowserUI()
    leases = Leases(spec)

    class Driver:
        max_players = 1

        async def join(self, **_):
            return engine

        async def close(self):
            pass

    async def driver(**_):
        return Driver()

    directory = RoomDirectory(MemoryRoomStore(), {"room": "http://room"})
    game = new_id()
    await directory.resolve(game, "world", create=True)
    archives = AsyncArchiveManager()
    runtime = RoomRuntime(
        directory=directory,
        owner_id="room",
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
        playback_gop=gop,
        **config_overrides,
    )
    host = RoomPipelines(
        runtime,
        config,
        archives,
        Reports(tmp_path, archives),
        lambda *_: leases,
        grace_seconds=grace,
    )
    owner, sub = await host.attach(
        dict(
            user_id="u",
            trace_id="room-test",
            message=dict(
                type="init",
                prompt="world",
                render_mode="model",
                whitebox={"game_id": game, "world_id": "world"},
            ),
        )
    )
    return host, owner, sub, engine, leases


async def until(sub, predicate):
    async with asyncio.timeout(5):
        while True:
            m = msgspec.msgpack.decode(await sub.recv())
            if predicate(m):
                return m


def credit(sub):
    for stream in ("whitebox", "model"):
        sub.ack(dict(protocol="whitebox-dual-v2", stream=stream, frame=-1, window=2))


def test_diagnostic_bursts_do_not_fill_gameplay_queue_or_delay_ping(tmp_path):
    async def run():
        host, owner, sub, _engine, _leases = await setup(tmp_path)
        try:
            await until(sub, lambda message: message.get("state") == "active")
            metric = msgspec.msgpack.encode(
                dict(type="client_metric", stage="client_video_decode", duration_ms=2)
            )
            # No scheduling yield: reproduces a coalesced observer burst.
            for _ in range(300):
                owner.accept(sub, metric)
            assert owner.command_queue.empty() and owner.command_bytes == 0
            assert not sub.closed
            owner.accept(
                sub,
                msgspec.msgpack.encode(
                    dict(type="event", kind="network.ping", payload={"nonce": 7})
                ),
            )
            assert owner.command_queue.empty()
            pong = await until(
                sub, lambda message: message.get("type") == "whitebox.pong"
            )
            assert pong["nonce"] == 7
            assert not owner.session.input_events
        finally:
            await host.close()
            await host.runtime.close()

    asyncio.run(run())


def test_playback_deadline_excludes_cold_start_and_idle_producer():
    async def run():
        owner = SimpleNamespace(terminal=None, request_keyframe=lambda _: None)
        sub = Subscription(owner, 1, ack_timeout=0.05)
        waiting = asyncio.create_task(sub.recv())
        await asyncio.sleep(0.15)
        assert not waiting.done(), "cold initialization must not require playback ACKs"
        sub.offer_control(dict(type="session.ready"))
        assert msgspec.msgpack.decode(await waiting)["type"] == "session.ready"
        credit(sub)
        waiting = asyncio.create_task(sub.recv())
        await asyncio.sleep(0.15)
        assert not waiting.done(), "an idle producer has no outstanding receipt"
        sub.offer_media(dict(stream="model", keyframe=True, output_frame_start=0))
        assert msgspec.msgpack.decode(await waiting)["output_frame_start"] == 0
        await asyncio.sleep(0.02)
        sub.ack(dict(protocol="whitebox-dual-v2", stream="model", frame=0, window=2))
        waiting = asyncio.create_task(sub.recv())
        await asyncio.sleep(0.15)
        assert not waiting.done()
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)

    asyncio.run(run())


def test_playback_deadline_enforces_initial_credit_and_each_stream():
    async def run():
        owner = SimpleNamespace(terminal=None, request_keyframe=lambda _: None)
        sub = Subscription(owner, 1, ack_timeout=0.05)
        sub.offer_control(dict(type="session.ready"))
        await sub.recv()
        with pytest.raises(ConnectionError, match="PLAYBACK_RECEIPT_TIMEOUT"):
            await asyncio.wait_for(sub.recv(), 0.3)
        sub = Subscription(owner, 1, ack_timeout=0.05)
        sub.offer_control(dict(type="session.ready"))
        await sub.recv()
        credit(sub)
        sub.offer_media(dict(stream="model", keyframe=True, output_frame_start=0))
        await sub.recv()
        waiting = asyncio.create_task(sub.recv())
        for _ in range(5):
            await asyncio.sleep(0.02)
            sub.ack(
                dict(protocol="whitebox-dual-v2", stream="whitebox", frame=-1, window=2)
            )
        with pytest.raises(ConnectionError, match="PLAYBACK_RECEIPT_TIMEOUT"):
            await asyncio.wait_for(waiting, 0.3)

    asyncio.run(run())


def wire(kind, **fields):
    return msgspec.msgpack.encode(dict(type="event", kind=kind, **fields))


def test_completion_waits_for_last_playback_receipt():
    async def run():
        owner = SimpleNamespace(terminal=None, request_keyframe=lambda _: None)
        sub = Subscription(owner, 1, ack_timeout=0.1)
        sub.offer_control(dict(type="session.ready"))
        await sub.recv()
        credit(sub)
        for stream in sub.media:
            sub.offer_media(dict(stream=stream, keyframe=True, output_frame_start=0))
            await sub.recv()
        owner.terminal = dict(type="session.completed")
        waiting = asyncio.create_task(sub.recv())
        await asyncio.sleep(0.02)
        assert not waiting.done()
        sub.ack(dict(protocol="whitebox-dual-v2", stream="model", frame=0, window=2))
        await asyncio.sleep(0.02)
        assert not waiting.done()
        sub.ack(dict(protocol="whitebox-dual-v2", stream="whitebox", frame=0, window=2))
        result = msgspec.msgpack.decode(await waiting)
        delivery = result.pop("playback_delivery")
        assert all(
            s["sent_frames"] == 1 and s["dropped_frames"] == 0
            for s in delivery["streams"].values()
        )
        assert result == dict(type="session.completed", frames=1, source_frames=1)

    asyncio.run(run())


def test_byte_credit_bounds_each_stream_without_blocking_reliable_control():
    async def run():
        sub = Subscription(
            SimpleNamespace(terminal=None, request_keyframe=lambda _: None), 1
        )
        for stream in sub.media:
            sub.ack(
                dict(protocol="whitebox-dual-v2", stream=stream, frame=-1, window=32)
            )
            for index in range(2):
                sub.offer_media(
                    dict(
                        stream=stream,
                        keyframe=True,
                        output_frame_start=index,
                        payload=b"x" * 160_000,
                    )
                )
        assert msgspec.msgpack.decode(await sub.recv())["stream"] == "whitebox"
        assert msgspec.msgpack.decode(await sub.recv())["stream"] == "model"
        waiting = asyncio.create_task(sub.recv())
        try:
            await asyncio.sleep(0.01)
            assert not waiting.done(), (
                "frame credit must not permit an unbounded byte burst"
            )
            sub.offer_control(dict(type="whitebox.pong", nonce=7))
            assert msgspec.msgpack.decode(await waiting) == dict(
                type="whitebox.pong", nonce=7
            )
            sub.ack(
                dict(protocol="whitebox-dual-v2", stream="whitebox", frame=0, window=32)
            )
            next_frame = msgspec.msgpack.decode(await sub.recv())
            assert (next_frame["stream"], next_frame["media_sequence"]) == (
                "whitebox",
                1,
            )
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            sub.stop("DONE")

    asyncio.run(run())


def test_oversized_access_unit_makes_progress_but_duplicate_ack_grants_no_bytes():
    async def run():
        sub = Subscription(
            SimpleNamespace(terminal=None, request_keyframe=lambda _: None), 1
        )
        sub.ack(dict(protocol="whitebox-dual-v2", stream="model", frame=-1, window=32))
        for index in range(3):
            sub.offer_media(
                dict(
                    stream="model",
                    keyframe=True,
                    output_frame_start=index,
                    payload=b"x" * 300_000,
                )
            )
        assert msgspec.msgpack.decode(await sub.recv())["media_sequence"] == 0
        sub.ack(dict(protocol="whitebox-dual-v2", stream="model", frame=0, window=32))
        assert msgspec.msgpack.decode(await sub.recv())["media_sequence"] == 1
        sub.ack(dict(protocol="whitebox-dual-v2", stream="model", frame=0, window=32))
        waiting = asyncio.create_task(sub.recv())
        try:
            await asyncio.sleep(0.01)
            assert not waiting.done(), (
                "duplicate ACK must not release the next access unit"
            )
            sub.ack(
                dict(protocol="whitebox-dual-v2", stream="model", frame=1, window=32)
            )
            assert msgspec.msgpack.decode(await waiting)["media_sequence"] == 2
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            sub.stop("DONE")

    asyncio.run(run())


def test_delayed_media_recovers_from_a_fresh_keyframe_within_one_second(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        room_pipeline, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    requests = []
    sub = Subscription(
        SimpleNamespace(terminal=None, request_keyframe=requests.append), 1
    )
    sub.ack(dict(protocol="whitebox-dual-v2", stream="model", frame=-1, window=32))
    sub.offer_media(dict(stream="model", keyframe=True, output_frame_start=0))
    clock.now = 1.01
    sub.offer_media(dict(stream="model", keyframe=False, output_frame_start=1))
    sub.offer_media(dict(stream="model", keyframe=True, output_frame_start=2))
    frame = msgspec.msgpack.decode(asyncio.run(sub.recv()))
    assert frame["media_sequence"] == 2, (
        "a recovered peer must not replay seconds of stale video"
    )
    assert requests == ["model"]


def test_model_replacement_releases_byte_credit_and_ignores_old_stream_ack():
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderSubscription,
    )

    async def run():
        owner = SimpleNamespace(
            terminal=None,
            request_keyframe=lambda _: None,
            session=SimpleNamespace(source_epoch="source", current_model_stream="old"),
        )
        sub = RenderSubscription(owner, 1)
        sub.ack(
            dict(
                protocol="whitebox-dual-v2",
                stream="model",
                media_stream_id="old",
                frame=-1,
                window=32,
            )
        )
        sub.offer_media(
            dict(
                stream="model",
                media_stream_id="old",
                keyframe=True,
                output_frame_start=0,
                payload=b"x" * 300_000,
            )
        )
        await sub.recv()
        sub.reset_model("new")
        sub.ack(
            dict(
                protocol="whitebox-dual-v2",
                stream="model",
                media_stream_id="new",
                frame=-1,
                window=32,
            )
        )
        sub.ack(
            dict(
                protocol="whitebox-dual-v2",
                stream="model",
                media_stream_id="old",
                frame=0,
                window=32,
            )
        )
        for index in range(2):
            sub.offer_media(
                dict(
                    stream="model",
                    media_stream_id="new",
                    keyframe=True,
                    output_frame_start=index,
                    payload=b"x" * 300_000,
                )
            )
        frame = msgspec.msgpack.decode(await asyncio.wait_for(sub.recv(), 0.2))
        assert frame["media_stream_id"] == "new" and frame["output_frame_start"] == 0
        waiting = asyncio.create_task(sub.recv())
        try:
            await asyncio.sleep(0.01)
            assert not waiting.done(), (
                "old-stream ACK must not grant the new stream byte credit"
            )
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            sub.stop("DONE")

    asyncio.run(run())


def test_owner_continues_abab_reconnect_and_fences_duplicate_input(tmp_path):
    async def run():
        host, owner, sub, engine, leases = await setup(tmp_path)
        try:
            await until(sub, lambda m: m.get("state") == "active")
            credit(sub)
            owner.accept(
                sub, wire("input", sequence=0, payload={"held_keys": ["KeyW"]})
            )
            await until(sub, lambda m: m["type"] == "whitebox.input_received")
            await engine.push()
            frame = await until(sub, lambda m: m["type"] == "frame_batch")
            epoch = owner.session.epoch
            assert frame["keyframe"]
            owner.detach(sub)
            await engine.push()
            await asyncio.sleep(0.04)
            resumed = owner.attach(dict(resume_token=owner.token, control_sequence=0))
            credit(resumed)
            assert leases.active == 1 and len(leases.opens) == 1
            with pytest.raises(ValueError, match="STALE"):
                owner.accept(sub, wire("input", sequence=1, payload={}))
            owner.accept(
                resumed, wire("input", sequence=0, payload={"held_keys": ["KeyW"]})
            )
            receipt = await until(resumed, lambda m: m.get("duplicate") is True)
            assert receipt["sequence"] == 0 and len(engine.inputs) == 1
            for i, mode in enumerate(("whitebox", "model", "whitebox", "model")):
                owner.accept(
                    resumed,
                    wire(
                        "display.configure",
                        payload=dict(request_id=str(i), expected_revision=i, mode=mode),
                    ),
                )
                await until(resumed, lambda m: m.get("state") == "accepted")
                await engine.push()
            assert owner.session.epoch == epoch and len(leases.opens) == 1
            # A dead archive never tears down the model or world.
            owner.session.recording.degrade("INJECTED_DISK_FAILURE")
            await engine.push()
            await asyncio.sleep(0.05)
            assert leases.active == 1 and not owner.closed
            owner.detach(resumed)
            await asyncio.sleep(0.3)
            async with asyncio.timeout(5):
                while not owner.closed:
                    await asyncio.sleep(0.01)
            assert leases.active == 0 and engine.closed
        finally:
            await host.close()
            await host.runtime.close()

    asyncio.run(run())


def test_pair_guard_bounds_silent_model_and_frees_failed_archive_aus():
    import time

    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
        InteractiveSession,
    )

    async def run():
        frame = SimpleNamespace(rgb=b"", index=0)
        session = SimpleNamespace(
            config=SimpleNamespace(pairing_max_age_seconds=0.06),
            recording=SimpleNamespace(accepting=False),
            pair_condition=asyncio.Condition(),
            pairs={0: (frame, time.perf_counter(), {"payload": b"archive-only"})},
        )
        session.pairing_age_budget = lambda captured=None: (
            InteractiveSession.pairing_age_budget(session, captured)
        )
        with pytest.raises(TimeoutError, match="pairing age budget"):
            await asyncio.wait_for(InteractiveSession._guard_pairs(session), 0.2)
        assert session.pairs[0][0] is frame
        assert session.pairs[0][2] is None

    asyncio.run(run())


def test_subscription_discards_broken_gop_until_recovery_keyframe():
    requests = []
    owner = SimpleNamespace(request_keyframe=requests.append)
    sub = Subscription(owner, 1, media_bytes=1000)

    def offer(index, key):
        sub.offer_media(
            dict(
                type="frame_batch",
                stream="model",
                output_frame_start=index,
                keyframe=key,
                payload=b"x" * 400,
            )
        )

    offer(0, True)
    offer(1, False)
    offer(2, False)  # overflow invalidates references
    assert requests and not sub.media["model"] and sub.needs_key["model"]
    offer(3, False)
    assert not sub.media["model"]
    offer(4, True)
    assert sub.media["model"][0][2]["output_frame_start"] == 4


def test_two_real_gateways_resume_same_room_and_model_after_gateway_shutdown(tmp_path):
    import json

    import httpx
    from fastapi import FastAPI, WebSocket
    from test_whitebox_rooms import server
    from websockets.asyncio.client import connect

    from sglang.multimodal_gen.runtime.realtime.whitebox.room_proxy import RoomProxy
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import create_app

    async def run():
        host, owner, original, engine, leases = await setup(tmp_path, grace=10)
        app = create_app(host.runtime, "fixture-token")
        host.install(app, "fixture-token")
        host.reports.install(app)
        async with server(app) as room_url:
            route = json.loads(host.runtime.directory.store.values[owner.game_id])
            route["owner_url"] = room_url
            host.runtime.directory.store.values[owner.game_id] = json.dumps(route)
            host.runtime.directory.owners = {"room": room_url}
            await until(original, lambda m: m.get("state") == "active")
            epoch = owner.session.epoch
            owner.detach(original)
            async with httpx.AsyncClient() as http:
                proxies = [
                    RoomProxy(
                        host.runtime.directory,
                        {"room": room_url},
                        "fixture-token",
                        http,
                        connect,
                    )
                    for _ in range(2)
                ]

                def gateway(proxy):
                    app = FastAPI()
                    proxy.install(app)

                    @app.websocket("/v1/realtime_video/generate")
                    async def route(ws: WebSocket):
                        await proxy.serve(ws, user_id="u", trace_id="test-gateway")

                    return app

                gw1 = server(gateway(proxies[0]))
                url1 = await gw1.__aenter__()
                async with server(gateway(proxies[1])) as url2:

                    async def resume(url):
                        ws = await connect(
                            url.replace("http", "ws") + "/v1/realtime_video/generate",
                            max_size=2**20,
                        )
                        await ws.send(
                            msgspec.msgpack.encode(
                                dict(
                                    type="resume",
                                    session_id=owner.session_id,
                                    resume_token=owner.token,
                                    owner_epoch=host.runtime.epoch,
                                    control_sequence=0,
                                    whitebox=dict(
                                        world_id="world", game_id=owner.game_id
                                    ),
                                )
                            )
                        )
                        async with asyncio.timeout(5):
                            while True:
                                m = msgspec.msgpack.decode(await ws.recv())
                                assert m["type"] != "error", m
                                if m["type"] == "session.ready":
                                    break
                        for stream in ("whitebox", "model"):
                            await ws.send(
                                wire(
                                    "playback_ack",
                                    payload=dict(
                                        protocol="whitebox-dual-v2",
                                        stream=stream,
                                        frame=-1,
                                        window=2,
                                    ),
                                )
                            )
                        return ws, m

                    ws1, ready1 = await resume(url1)
                    await ws1.send(
                        wire("input", sequence=0, payload={"held_keys": ["KeyW"]})
                    )
                    async with asyncio.timeout(5):
                        while (
                            msgspec.msgpack.decode(await ws1.recv()).get("type")
                            != "whitebox.input_committed"
                        ):
                            pass
                    await gw1.__aexit__(
                        None, None, None
                    )  # actual Gateway process lifecycle / socket loss
                    assert leases.active == 1 and not owner.closed
                    ws2, ready2 = await resume(url2)
                    assert ready1["session_id"] == ready2["session_id"]
                    assert (
                        ready2["connection_generation"]
                        > ready1["connection_generation"]
                    )
                    assert ready2["last_input_sequence"] == 0
                    await engine.push()
                    async with asyncio.timeout(5):
                        while True:
                            m = msgspec.msgpack.decode(await ws2.recv())
                            if m["type"] == "frame_batch":
                                break
                    assert (
                        m["encoding"] == "h264"
                        and m["keyframe"]
                        and len(m["payload"]) < engine.spec.frame_bytes
                    )
                    assert owner.session.epoch == epoch and len(leases.opens) == 1
                    response = await http.get(
                        url2 + "/v1/" + ready2["report_path"],
                        headers={"Authorization": "Bearer " + ready2["report_token"]},
                    )
                    assert (
                        response.status_code == 200
                        and response.json()["recording_id"] == owner.stream_id
                    )
                    await ws2.close()
                    await ws1.close()
                    await host.close()

    asyncio.run(run())


def test_slow_report_disk_has_bounded_workers_and_never_blocks_live_loop(tmp_path):
    import threading
    import time

    from fastapi import HTTPException

    async def run():
        reports = Reports(tmp_path, AsyncArchiveManager())
        reports.io_timeout = 0.03
        gate = threading.Event()

        def stalled():
            gate.wait(2)

        try:
            for _ in range(2):
                with pytest.raises(HTTPException) as e:
                    await reports.run_io(stalled)
                assert e.value.status_code == 503
            before = time.monotonic()
            with pytest.raises(HTTPException):
                await reports.run_io(stalled)
            assert time.monotonic() - before < 0.02
            # Both disk workers remain occupied, but the live event loop advances.
            progressed = False
            await asyncio.sleep(0.01)
            progressed = True
            assert progressed and reports.io_active == 2
        finally:
            gate.set()
            async with asyncio.timeout(1):
                while reports.io_active:
                    await asyncio.sleep(0.01)

    asyncio.run(run())


def test_maintenance_gate_rejects_new_play_without_coordinator_or_room(
    tmp_path, monkeypatch
):
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
        create_app,
    )

    class Proxy:
        def install(self, app):
            pass

        async def serve(self, *args, **kwargs):
            raise AssertionError("Maintenance must not allocate a Room")

        async def close(self):
            pass

    class Coordinator:
        async def close(self):
            pass

    monkeypatch.setenv("WHITEBOX_ADMISSION_PAUSED", "1")
    app = create_app(
        Coordinator(),
        model_revision="test",
        vae_fingerprint="local",
        internal_output_url="ws://unused",
        whitebox_proxy=Proxy(),
    )
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/realtime_video/generate?whitebox_control=v1"
        ) as ws:
            message = msgspec.msgpack.decode(ws.receive_bytes())
            assert message["code"] == "PLAYTEST_MAINTENANCE"


@pytest.mark.parametrize(
    "worker_capacity,pairing_frames,window",
    [(34, None, 34), (68, None, 38), (68, 36, 36)],
)
def test_whitebox_preview_advances_when_model_pairing_window_is_full(
    tmp_path, monkeypatch, worker_capacity, pairing_frames, window
):
    import json

    import av
    from test_whitebox_interactive import Worker

    monkeypatch.setattr(Worker, "source_capacity", worker_capacity)
    original = Worker.emit
    held = []

    async def delayed(self, message):
        if message["type"] == "frame_batch":
            held.append(message)
        else:
            await original(self, message)

    monkeypatch.setattr(Worker, "emit", delayed)

    async def run():
        spec = VideoSpec(32, 32)
        pair_bytes = 6 * spec.frame_bytes + 2 * 65536 + 1024 * 1024
        kwargs = (
            {}
            if pairing_frames is None
            else {"pairing_max_bytes": pairing_frames * pair_bytes}
        )
        host, owner, sub, engine, leases = await setup(tmp_path, **kwargs)
        try:
            await until(sub, lambda m: m.get("state") == "active")
            credit(sub)
            async with asyncio.timeout(5):
                while not owner.session.recording.worker_ready:
                    await asyncio.sleep(0.01)
            for index in range(window):
                await asyncio.sleep(1 / 60)
                await engine.push()
                frame = await until(sub, lambda m: m["type"] == "whitebox.frame")
                sub.ack(
                    dict(
                        protocol="whitebox-dual-v2",
                        stream="whitebox",
                        frame=frame["output_frame_start"],
                        window=2,
                    )
                )
                async with asyncio.timeout(3):
                    while len(held) <= index:
                        await asyncio.sleep(0.001)
            assert len(owner.session.pairs) == window
            assert owner.session.received == 0
            assert owner.session.recording.manifest["pairing"]["capacity"] == window
            assert len(held) * pair_bytes <= owner.session.config.pairing_max_bytes
            if window == 38:
                # One clip in flight + four retained Lynn frames must still
                # permit the COMPLETE next 17-frame clip before any output ACK.
                assert len(held) - 17 - 4 == 17
            # No model output or pair credit is released. Live source must
            # continue anyway, using its own sequence and H.264 dependencies.
            for index in range(window, window + 8):
                await asyncio.sleep(1 / 60)
                if index == window + 2:
                    owner.request_keyframe("whitebox")
                await engine.push()
                async with asyncio.timeout(1):
                    frame = await until(sub, lambda m: m["type"] == "whitebox.frame")
                assert frame["output_frame_start"] == index
                assert frame["source_frame_index"] == index
                assert frame["recording_frame_index"] is None
                if index == window + 2:
                    assert frame["keyframe"]
                sub.ack(
                    dict(
                        protocol="whitebox-dual-v2",
                        stream="whitebox",
                        frame=index,
                        window=2,
                    )
                )
            assert len(held) == len(owner.session.pairs) == window
            assert owner.session.received == 0
            assert owner.session.preview_count == window + 8
            # Release the model after preview has already shown every source frame.
            # Only the newest of the eight unselected frames reaches H3.
            selected_sources = []

            async def release(message):
                await original(leases.opens[0], message)
                frame = await until(sub, lambda m: m["type"] == "frame_batch")
                selected_sources.append(frame["source_frame_index"])
                assert frame["recording_frame_index"] == len(selected_sources) - 1
                sub.ack(
                    dict(
                        protocol="whitebox-dual-v2",
                        stream="model",
                        frame=frame["output_frame_start"],
                        window=2,
                    )
                )
                # The synthetic 32px model is much faster than production.
                # Drain its archive offers rather than testing disk overload here.
                async with asyncio.timeout(3):
                    while owner.session.recording.statistics()["pending_operations"]:
                        assert not owner.session.recording.failure
                        await asyncio.sleep(0.001)

            for message in list(held):
                await release(message)
            async with asyncio.timeout(2):
                while len(held) < window + 1:
                    await asyncio.sleep(0.001)
            await release(held[-1])
            assert selected_sources == list(range(window)) + [window + 7]
            await engine.finish()
            completed = await until(sub, lambda m: m["type"] == "session.completed")
            assert completed["source_frames"] == window + 8
            assert completed["frames"] == window + 1
            directory = owner.session.recording.directory
            async with asyncio.timeout(5):
                while not owner.closed:
                    await asyncio.sleep(0.01)
        finally:
            await host.close()
            await host.runtime.close()
        assert leases.active == 0 and engine.closed
        assert owner.session.source_playback.context is None
        assert owner.session.record_source.context is None
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["recording_status"] == "complete"
        assert manifest["rendering"]["live_source_frames"] == window + 8
        assert manifest["rendering"]["live_source_superseded_frames"] == 0
        assert manifest["rendering"]["source_frames_superseded_before_model"] == 7
        pts = []
        for stream in ("whitebox", "model"):
            with av.open(str(directory / (stream + ".mp4"))) as c:
                frames = list(c.decode(video=0))
            assert len(frames) == window + 1
            pts.append([round(float(f.pts * f.time_base) * 1e6) for f in frames])
        assert pts[0] == pts[1] == [index * 50000 for index in selected_sources]

    asyncio.run(run())


def test_independent_source_preview_keeps_paired_archive_intact(tmp_path, monkeypatch):
    import json

    import av
    from test_whitebox_interactive import Worker

    original = Worker.emit
    held = []

    async def delayed(self, message):
        if message["type"] == "frame_batch":
            held.append((self, message))
        else:
            await original(self, message)

    monkeypatch.setattr(Worker, "emit", delayed)

    async def run():
        from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
            LatestFrame,
        )

        gate = asyncio.Event()
        get = LatestFrame.get

        async def gated(self, cursor):
            await gate.wait()
            return await get(self, cursor)

        monkeypatch.setattr(LatestFrame, "get", gated)
        host, owner, sub, engine, leases = await setup(tmp_path)
        source_frames, model_frames = [], []
        try:
            await until(sub, lambda m: m.get("state") == "active")
            credit(sub)
            async with asyncio.timeout(5):
                while not owner.session.recording.worker_ready:
                    await asyncio.sleep(0.01)
            # Burst capture intentionally skips CLI identities before selection.
            for _ in range(5):
                await engine.push()
            async with asyncio.timeout(2):
                while owner.session.latest.version < 5:
                    await asyncio.sleep(0.001)
            gate.set()
            for n in range(17):
                await asyncio.sleep(1 / 60)
                if n:
                    await engine.push()
                frame = await until(sub, lambda m: m["type"] == "whitebox.frame")
                source_frames.append(frame)
                sub.ack(
                    dict(
                        protocol="whitebox-dual-v2",
                        stream="whitebox",
                        frame=frame["output_frame_start"],
                        window=2,
                    )
                )
                assert frame["output_frame_start"] == n
                assert frame["recording_frame_index"] is None
                assert frame["model_pts_us"] == engine.spec.pts(
                    frame["source_frame_index"]
                )
                assert frame["source_frame_index"] >= n + 4
                async with asyncio.timeout(2):
                    while len(held) <= n:
                        await asyncio.sleep(0.001)
            # All 17 source pictures are visible before ANY model picture exists.
            async with asyncio.timeout(2):
                while len(held) != 17:
                    await asyncio.sleep(0.001)
            assert owner.session.received == 0
            assert len(owner.session.pairs) == 17
            assert all(not entry[0].rgb for entry in owner.session.pairs.values())
            paired_sources = [
                dict(au, pts_us=frame.provenance["capture_pts_us"])
                for frame, _, au in owner.session.pairs.values()
            ]
            assert owner.session.record_source is not owner.session.source_playback
            assert (
                owner.session.record_source.stream["max_bitrate"]
                == owner.session.source_playback.stream["max_bitrate"]
                == host.config.source_max_bitrate
            )
            # A subscriber's keyframe recovery only touches the live codec.
            owner.request_keyframe("whitebox")
            assert owner.session.source_playback.force_keyframe.is_set()
            assert not owner.session.record_source.force_keyframe.is_set()
            submitted = [
                e["payload"]
                for e in leases.opens[0].events
                if e["kind"] == "source.append"
            ]
            assert len(submitted) == 17
            assert all(
                p["data"] == bytes([30]) * engine.spec.frame_bytes for p in submitted
            )
            assert [p["pts_us"] for p in submitted] == [
                engine.spec.pts(i) for i in range(17)
            ]
            async with asyncio.timeout(5):
                while owner.session.recording.statistics()["pending_operations"]:
                    assert not owner.session.recording.failure
                    await asyncio.sleep(0.01)
            for worker, message in held:
                await original(worker, message)
                frame = await until(sub, lambda m: m["type"] == "frame_batch")
                model_frames.append(frame)
                sub.ack(
                    dict(
                        protocol="whitebox-dual-v2",
                        stream="model",
                        frame=frame["output_frame_start"],
                        window=2,
                    )
                )
            await engine.finish()
            await until(sub, lambda m: m["type"] == "session.completed")
            assert not owner.session.pairs
            recording = owner.session.recording
            async with asyncio.timeout(5):
                while not owner.closed:
                    await asyncio.sleep(0.01)
            await host.close()
            await host.runtime.close()
            manifest = json.loads((recording.directory / "manifest.json").read_text())
            assert manifest["recording_status"] == "complete"
            assert manifest["pairing"]["high_water_frames"] == 17
            assert manifest["rendering"]["live_source"] == "independent_latest"
            assert manifest["rendering"]["live_source_frames"] == 17
            assert manifest["rendering"]["live_source_superseded_frames"] == 4
            for name, frames in (("whitebox", paired_sources), ("model", model_frames)):
                decoder = av.CodecContext.create("h264", "r")
                live = [
                    f.to_ndarray(format="rgb24")
                    for packet in frames
                    for f in decoder.decode(av.Packet(packet["payload"]))
                ]
                with av.open(str(recording.directory / (name + ".mp4"))) as video:
                    stored = list(video.decode(video=0))
                assert len(stored) == len(live) == 17
                assert all(
                    (a == b.to_ndarray(format="rgb24")).all()
                    for a, b in zip(live, stored)
                )
                assert [round(float(f.pts * f.time_base) * 1e6) for f in stored] == [
                    f["pts_us"] for f in frames
                ]
        finally:
            await host.close()
            await host.runtime.close()

    asyncio.run(run())


# H3 clip bursts and post-archive delivery accounting.
async def replay_clip_bursts(monkeypatch, *, rtt, encode_seconds, clips=4):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        room_pipeline, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    force_key = False

    def request_keyframe(_):
        nonlocal force_key
        force_key = True

    owner = SimpleNamespace(terminal=None, request_keyframe=request_keyframe)
    sub = room_pipeline.Subscription(owner, 1)
    for stream in ("whitebox", "model"):
        sub.ack(dict(protocol="whitebox-dual-v2", stream=stream, frame=-1, window=8))
    # The production H3 decoder emits a 17-frame clip every ~1.2 seconds.
    events = [
        (clip * 1.2 + i * encode_seconds, "frame", clip * 17 + i)
        for clip in range(clips)
        for i in range(17)
    ]
    heapq.heapify(events)
    received, waits = [], []
    while events:
        clock.now, kind, index = heapq.heappop(events)
        if kind == "ack":
            sub.ack(
                dict(protocol="whitebox-dual-v2", stream="model", frame=index, window=8)
            )
        else:
            key = index % 6 == 0 or force_key
            force_key = False
            sub.offer_media(
                dict(
                    type="frame_batch",
                    stream="model",
                    keyframe=key,
                    output_frame_start=index,
                    payload=b"x" * 24000,
                )
            )
        while sub.media["model"] and sub.sent["model"] - sub.received["model"] < 8:
            offered_at = sub.media["model"][0][0]
            wire = msgspec.msgpack.decode(await sub.recv())
            received.append(wire["media_sequence"])
            waits.append(clock.now - offered_at)
            heapq.heappush(events, (clock.now + rtt, "ack", wire["output_frame_start"]))
        assert sub.media_bytes <= sub.byte_limit
    return received, waits, sub


@pytest.mark.parametrize(
    "rtt,encode_seconds", [(0.160, 0.009), (0.203, 0.006), (0.203, 0.0)]
)
def test_h3_clip_burst_survives_receipt_round_trip(monkeypatch, rtt, encode_seconds):
    received, waits, sub = asyncio.run(
        replay_clip_bursts(monkeypatch, rtt=rtt, encode_seconds=encode_seconds)
    )
    assert received == list(range(68)), (
        "Healthy ACK8 peers must receive every frame of each H3 clip"
    )
    assert max(waits) <= 2 * rtt + 1e-12
    stats = sub.delivery["streams"]["model"]
    assert (
        stats["offered_frames"],
        stats["sent_frames"],
        stats["acknowledged_frames"],
        stats["dropped_frames"],
    ) == (68, 68, 68, 0)


def test_slow_peer_still_expires_and_recovers_at_keyframe(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        room_pipeline, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    requests = []
    sub = room_pipeline.Subscription(
        SimpleNamespace(request_keyframe=requests.append), 1
    )

    def offer(index, key=False):
        sub.offer_media(
            dict(
                stream="model",
                keyframe=key,
                output_frame_start=index,
                payload=b"x" * 100,
            )
        )

    offer(0, True)
    offer(1)
    clock.now = 5.01
    offer(2)
    assert not sub.media["model"] and sub.needs_key["model"] and requests == ["model"]
    assert sub.delivery["streams"]["model"]["dropped_by_reason"] == {
        "queue_age": 2,
        "waiting_for_keyframe": 1,
    }
    offer(3, True)
    assert [x[2]["output_frame_start"] for x in sub.media["model"]] == [3]
    sub.stop("DISCONNECTED")
    stats = sub.delivery["streams"]["model"]
    assert stats["offered_frames"] == stats["dropped_frames"] == 4


@pytest.mark.parametrize(
    "payload_size,expected_reason", [(1, "queue_frames"), (100000, "queue_bytes")]
)
def test_nonreading_peer_retains_hard_bounds(payload_size, expected_reason):
    sub = room_pipeline.Subscription(
        SimpleNamespace(request_keyframe=lambda _: None), 1
    )
    for index in range(300):
        sub.offer_media(
            dict(
                stream="model",
                keyframe=index % 6 == 0,
                output_frame_start=index,
                payload=b"x" * payload_size,
            )
        )
        assert len(sub.media["model"]) <= sub.frame_limit
        assert 0 <= sub.media_bytes <= 8 * 1024**2
    stats = sub.delivery["streams"]["model"]
    assert stats["dropped_by_reason"][expected_reason] > 0
    assert stats["offered_frames"] == stats["dropped_frames"] + len(sub.media["model"])


def test_both_streams_can_hold_one_second_at_target_bitrate(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        room_pipeline, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    sub = room_pipeline.Subscription(
        SimpleNamespace(request_keyframe=lambda _: None), 1
    )
    # ~4.2 Mbps per stream including a small allowance above the nominal cap.
    for index in range(24):
        clock.now = index / 24
        for stream in ("whitebox", "model"):
            sub.offer_media(
                dict(
                    stream=stream,
                    keyframe=index % 6 == 0,
                    output_frame_start=index,
                    payload=b"x" * 22000,
                )
            )
    assert all(len(queue) == 24 for queue in sub.media.values())
    assert all(
        stats["dropped_frames"] == 0 for stats in sub.delivery["streams"].values()
    )
    assert sub.media_bytes < sub.byte_limit


@pytest.mark.parametrize(
    "batch_frames,pipeline_overlap,input_frames,native_flow",
    [
        (1, False, 18, False),
        (17, False, 18, False),
        (1, True, 18, False),
        (17, True, 18, False),
        (9, True, 9, False),
        (4, True, 9, False),
        (1, True, 25, True),
        (4, True, 25, True),
        (9, True, 25, True),
    ],
)
def test_capture_clock_stream_drains_all_preview_and_paired_frames(
    tmp_path, monkeypatch, batch_frames, pipeline_overlap, input_frames, native_flow
):
    """A continuous producer must keep the final VFR AU and its original identity."""

    from test_whitebox_interactive import Worker

    original = Worker.emit
    held = []

    async def batched(self, message):
        if message["type"] == "session.ready":
            message["source_output_release_protocol"] = "valid-source-frames-v1"
            if native_flow:
                message.update(
                    session_id="s1",
                    generation_id="g1",
                    source_demand_protocol="source-demand-v1",
                    source_flow=dict(
                        protocol="clip17-source-flow-v1",
                        input_groups=[1, 8, 8],
                        first_output_input_frames=9,
                        progressive_output=True,
                    ),
                )
        if message["type"] == "source.accepted" and native_flow:
            await original(self, message)
            end = message["source_frame_end"]
            await original(
                self,
                dict(
                    type="source.demand",
                    protocol="source-demand-v1",
                    session_id="s1",
                    generation_id="g1",
                    sequence=end,
                    accepted_source_frame_end=end,
                    dispatched_source_frame_end=end,
                    required_source_frame_end=end + 1,
                    queued_source_frames=0,
                    waiting_for_source=False,
                    input_wait_ms=0.0,
                ),
            )
            return
        if message["type"] == "frame_batch":
            held.append(message)
            if len(held) < batch_frames:
                return
            for ready in held:
                await original(self, ready)
            held.clear()
            return
        if message["type"] == "session.completed":
            for ready in held:
                await original(self, ready)
            held.clear()
        await original(self, message)

    monkeypatch.setattr(Worker, "emit", batched)

    async def run():
        import av

        host, owner, sub, engine, leases = await setup(
            tmp_path,
            capture_clock=True,
            model_input_frames=input_frames,
            pipeline_overlap=pipeline_overlap,
            model_subclip_admission=input_frames == 9,
            model_source_sampling="latest",
        )
        rows = {"whitebox": [], "model": []}
        decoders = {name: av.CodecContext.create("h264", "r") for name in rows}
        try:
            await until(sub, lambda m: m.get("state") == "active")
            credit(sub)
            assert (
                owner.session.playback.stream["rate_control_clock"]
                == "source_capture_vfr"
            )

            async def consume():
                async with asyncio.timeout(10):
                    while True:
                        message = msgspec.msgpack.decode(await sub.recv())
                        if message["type"] == "session.completed":
                            return message
                        if message["type"] == "error":
                            raise AssertionError(message)
                        if message["type"] not in ("frame_batch", "whitebox.frame"):
                            continue
                        stream = message["stream"]
                        assert (
                            len(decoders[stream].decode(av.Packet(message["payload"])))
                            == 1
                        )
                        rows[stream].append(
                            (
                                message["output_frame_start"],
                                message["source_frame_index"],
                            )
                        )
                        sub.ack(
                            dict(
                                protocol="whitebox-dual-v2",
                                stream=stream,
                                frame=message["output_frame_start"],
                                window=2,
                            )
                        )

            reader = asyncio.create_task(consume())
            # This verifies drain identity, not cold encoder latest-frame drops.
            # Observe native encode completion before feeding the timed burst;
            # VFR output itself can retain this first AU until the next capture.
            await engine.push()
            async with asyncio.timeout(5):
                while owner.session.preview_count < 1:
                    await asyncio.sleep(0.01)
            for index in range(1, 68):
                await engine.push()
                # This is a drain/identity test, not a preview-overload test.
                # A fixed 20ms delay can supersede a frame when CI is busy.
                async with asyncio.timeout(5):
                    while owner.session.preview_count <= index:
                        await asyncio.sleep(0.01)
            await engine.finish()
            completed = await reader
            assert completed["frames"] == completed["source_frames"] == 68
            assert rows["whitebox"] == rows["model"] == [(i, i) for i in range(68)]
            async with asyncio.timeout(5):
                while not owner.closed:
                    await asyncio.sleep(0.01)
            assert leases.active == 0
            pairing = owner.session.recording.manifest["pairing"]
            assert pairing["model_input_capacity"] == input_frames
            assert pairing["capacity"] == input_frames + (4 if pipeline_overlap else 0)
            assert pairing["high_water_frames"] <= pairing["capacity"]
            if native_flow:
                assert pairing["minimum_input_capacity"] == 9
                assert pairing["source_flow"]["input_groups"] == [1, 8, 8]
                assert (
                    leases.opens[0].init["source_demand_protocol"] == "source-demand-v1"
                )
        finally:
            await host.close()
            await host.runtime.close()

    asyncio.run(run())


def test_model_submission_does_not_wait_for_paired_source_encode(tmp_path):
    """A deliberately blocked archive-source encode cannot gate the RGB send."""
    import threading

    entered, release = threading.Event(), threading.Event()

    async def run():
        host, owner, sub, engine, leases = await setup(
            tmp_path, capture_clock=True, model_input_frames=18, pipeline_overlap=True
        )
        try:
            await until(sub, lambda m: m.get("state") == "active")
            credit(sub)
            codec = owner.session.record_source
            original = codec.push

            def held_encode(*args):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("Test source encoder release was not signaled")
                return original(*args)

            codec.push = held_encode
            await engine.push()
            assert await asyncio.to_thread(entered.wait, 2)
            async with asyncio.timeout(2):
                while not any(
                    event.get("kind") == "source.append"
                    for event in leases.opens[0].events
                ):
                    await asyncio.sleep(0.005)
            assert not release.is_set()
            release.set()
            await engine.finish()

            async with asyncio.timeout(5):
                while True:
                    message = msgspec.msgpack.decode(await sub.recv())
                    if message["type"] == "session.completed":
                        assert message["frames"] == message["source_frames"] == 1
                        break
                    if message["type"] in ("frame_batch", "whitebox.frame"):
                        sub.ack(
                            dict(
                                protocol="whitebox-dual-v2",
                                stream=message["stream"],
                                frame=message["output_frame_start"],
                                window=2,
                            )
                        )
        finally:
            release.set()
            await host.close()
            await host.runtime.close()

    asyncio.run(run())


def test_subclip_admission_rejects_legacy_worker_and_releases_model(tmp_path):
    async def run():
        host, owner, sub, engine, leases = await setup(
            tmp_path,
            capture_clock=True,
            model_input_frames=9,
            pipeline_overlap=True,
            model_subclip_admission=True,
        )
        try:
            message = await until(sub, lambda m: m["type"] == "error")
            assert message["type"] == "error"
            async with asyncio.timeout(5):
                while not owner.closed:
                    await asyncio.sleep(0.01)
            assert leases.active == 0
        finally:
            await host.close()
            await host.runtime.close()

    asyncio.run(run())


def test_ack_age_and_byte_ledger_survive_duplicate_ack_and_clear_on_close(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        room_pipeline, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    sub = Subscription(
        SimpleNamespace(terminal=None, request_keyframe=lambda _: None), 1
    )
    sub.ack(dict(protocol="whitebox-dual-v2", stream="model", frame=-1, window=32))
    sub.offer_media(
        dict(stream="model", keyframe=True, output_frame_start=0, payload=b"x" * 1000)
    )
    asyncio.run(sub.recv())
    clock.now = 0.5
    sub.ack(dict(protocol="whitebox-dual-v2", stream="model", frame=-1, window=32))
    stats = sub.delivery["streams"]["model"]
    assert stats["oldest_unacknowledged_ms"] == 500
    assert stats["inflight_frames"] == 1
    sub.ack(dict(protocol="whitebox-dual-v2", stream="model", frame=0, window=32))
    assert stats["max_ack_age_ms"] == 500
    assert stats["inflight_bytes"] == stats["oldest_unacknowledged_ms"] == 0
    sub.stop("DONE")
    assert sub.delivery["inflight_bytes"] == 0


def test_oversized_au_requires_shared_empty_window_and_control_still_progresses():
    async def run():
        sub = Subscription(
            SimpleNamespace(terminal=None, request_keyframe=lambda _: None), 1
        )
        for stream in sub.media:
            sub.ack(
                dict(protocol="whitebox-dual-v2", stream=stream, frame=-1, window=32)
            )
            sub.offer_media(
                dict(
                    stream=stream,
                    keyframe=True,
                    output_frame_start=0,
                    payload=b"x" * 700000,
                )
            )
        assert msgspec.msgpack.decode(await sub.recv())["stream"] == "whitebox"
        waiting = asyncio.create_task(sub.recv())
        try:
            await asyncio.sleep(0.01)
            assert not waiting.done()
            sub.offer_control(dict(type="whitebox.pong", nonce=1))
            assert msgspec.msgpack.decode(await waiting)["type"] == "whitebox.pong"
            assert sum(sub.inflight_bytes.values()) < sub.max_access_unit_bytes
            sub.ack(
                dict(protocol="whitebox-dual-v2", stream="whitebox", frame=0, window=32)
            )
            assert msgspec.msgpack.decode(await sub.recv())["stream"] == "model"
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            sub.stop("DONE")

    asyncio.run(run())


def test_overlarge_au_discards_its_dependents_until_valid_keyframe():
    requested = []
    sub = Subscription(
        SimpleNamespace(terminal=None, request_keyframe=requested.append), 1
    )
    sub.ack(dict(protocol="whitebox-dual-v2", stream="model", frame=-1, window=32))
    sub.offer_media(
        dict(
            stream="model",
            keyframe=True,
            output_frame_start=0,
            payload=b"x" * (sub.max_access_unit_bytes + 1),
        )
    )
    sub.offer_media(
        dict(stream="model", keyframe=False, output_frame_start=1, payload=b"delta")
    )
    sub.offer_media(
        dict(stream="model", keyframe=True, output_frame_start=2, payload=b"idr")
    )
    message = msgspec.msgpack.decode(asyncio.run(sub.recv()))
    assert message["media_sequence"] == 2
    assert requested == ["model"]
    reasons = sub.delivery["streams"]["model"]["dropped_by_reason"]
    assert reasons == {"access_unit_bytes": 1, "waiting_for_keyframe": 1}
    sub.stop("DONE")


def test_large_keyframe_drains_shared_credit_without_starving_either_stream():
    async def run():
        sub = Subscription(
            SimpleNamespace(terminal=None, request_keyframe=lambda _: None), 1
        )
        for stream in sub.media:
            sub.ack(
                dict(protocol="whitebox-dual-v2", stream=stream, frame=-1, window=32)
            )
        sub.offer_media(
            dict(
                stream="whitebox",
                keyframe=True,
                output_frame_start=0,
                payload=b"x" * 100000,
            )
        )
        await sub.recv()
        sub.offer_media(
            dict(
                stream="whitebox",
                keyframe=True,
                output_frame_start=1,
                payload=b"x" * 100000,
            )
        )
        sub.offer_media(
            dict(
                stream="model",
                keyframe=True,
                output_frame_start=0,
                payload=b"x" * 700000,
            )
        )
        waiting = asyncio.create_task(sub.recv())
        try:
            await asyncio.sleep(0.01)
            assert not waiting.done(), (
                "source traffic must allow a bounded oversized IDR to drain credit"
            )
            sub.offer_media(
                dict(
                    stream="whitebox",
                    keyframe=True,
                    output_frame_start=2,
                    payload=b"x" * 100000,
                )
            )
            await asyncio.sleep(0.01)
            assert not waiting.done()
            sub.ack(
                dict(protocol="whitebox-dual-v2", stream="whitebox", frame=0, window=32)
            )
            assert msgspec.msgpack.decode(await waiting)["stream"] == "model"
            sub.ack(
                dict(protocol="whitebox-dual-v2", stream="model", frame=0, window=32)
            )
            assert msgspec.msgpack.decode(await sub.recv())["stream"] == "whitebox"
            assert sub.large_au_stream is None
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            sub.stop("DONE")

    asyncio.run(run())


def test_late_archive_source_output_after_fail_open_does_not_restore_retired_pairs():
    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
        InteractiveSession,
    )

    async def run():
        owner = SimpleNamespace(
            config=SimpleNamespace(spec=VideoSpec(32, 32)),
            recording=SimpleNamespace(accepting=False),
        )
        pairs = {}
        await InteractiveSession._store_archive_source_outputs(
            owner, [(13, {"payload": b"late-source-au"})], pairs, asyncio.Condition()
        )
        assert not pairs
        owner.recording.accepting = True
        with pytest.raises(KeyError):
            await InteractiveSession._store_archive_source_outputs(
                owner,
                [(13, {"payload": b"late-source-au"})],
                pairs,
                asyncio.Condition(),
            )
        pairs[13] = ("original-source", 100, None)
        await InteractiveSession._store_archive_source_outputs(
            owner, [(13, {"payload": b"source-au"})], pairs, asyncio.Condition()
        )
        assert pairs[13] == ("original-source", 100, {"payload": b"source-au"})

    asyncio.run(run())


def test_source_debug_history_is_read_without_model_admission_and_fenced():
    async def scenario():
        calls = []
        record = {"engine": None}
        history = [{"conditionId": "source:first"}, {"conditionId": "source:second"}]

        async def refresh():
            calls.append("read")
            return history

        engine = SimpleNamespace(refresh_source_debug=refresh)
        record["engine"] = engine
        owner = object.__new__(room_pipeline.LocalRoomEngine)
        owner.closed = False
        owner.stream_id = "player"
        owner.runtime = SimpleNamespace(player=lambda _: record)
        owner._context_engine = engine
        assert await owner.refresh_source_debug() == history
        assert calls == ["read"]
        record["engine"] = object()
        with pytest.raises(Exception) as caught:
            await owner.refresh_source_debug()
        assert caught.value.code == "SOURCE_EPOCH_MISMATCH"
        assert calls == ["read"]

    asyncio.run(scenario())
