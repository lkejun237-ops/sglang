# SPDX-License-Identifier: Apache-2.0
"""Real Room orchestration against a protocol peer; no GPU performance claims."""

import asyncio
import hashlib
import io
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS
from uuid import uuid4

import msgspec
import pytest
from PIL import Image

from sglang.multimodal_gen.runtime.realtime.source_prepare import PROTOCOL, V2_PROTOCOL
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import RenderError
from sglang.multimodal_gen.runtime.realtime.whitebox.startup import (
    PlayStartup,
    StartupAssets,
)

SPEC = NS(width=32, height=32, frame_bytes=32 * 32 * 3)
RGB = bytes([255, 0, 0]) * 32 * 32


def request(with_frame=True, frame_size=(32, 32), *, protocol=PROTOCOL):
    value = dict(
        protocol=protocol,
        request_id=uuid4().hex,
        intent="model",
        fallback_policy="cli_initial_once",
        initial_render=dict(prompt="snow", seed=42, references=[]),
    )
    if protocol == V2_PROTOCOL:
        value.pop("fallback_policy")
    if with_frame:
        out = io.BytesIO()
        Image.new("RGB", frame_size, "red").save(out, format="PNG")
        raw = out.getvalue()
        value["initial_frame"] = dict(
            format="png",
            width=frame_size[0],
            height=frame_size[1],
            payload=raw,
            encoded_size=len(raw),
            encoded_sha256=hashlib.sha256(raw).hexdigest(),
            rgb_sha256=hashlib.sha256(
                Image.new("RGB", frame_size, "red").tobytes()
            ).hexdigest(),
        )
    return value


class Peer:
    def __init__(self, protocol=PROTOCOL):
        self.protocol = protocol
        self.incoming = asyncio.Queue()
        self.commands = []

    async def send(self, raw):
        value = msgspec.msgpack.decode(raw)
        self.commands.append(value)
        if value["type"] == "init":
            answer = dict(
                type="session.ready",
                source_prepare_protocol=self.protocol,
                source_prepare_profile_sha256="a" * 64,
                source_stream=value["source_stream"],
            )
        else:
            answer = dict(
                type=(
                    "source.prepared"
                    if value["kind"] == "source.prepare"
                    else "source.attached"
                ),
                identity=value["payload"]["identity"],
            )
        await self.incoming.put(msgspec.msgpack.encode(answer))

    async def recv(self):
        return await self.incoming.get()


class Lease:
    def __init__(self, protocol=PROTOCOL):
        self.ws = Peer(protocol)
        self.resource_state = "none"
        self.opens = 0
        self.closes = 0

    @asynccontextmanager
    async def open(self):
        self.opens += 1
        self.resource_state = "reserved"
        try:
            yield self.ws, NS(denoiser=NS(worker_epoch=uuid4().hex))
        finally:
            self.closes += 1
            self.resource_state = "released"


def startup(with_frame=True, *, value=None, worker_protocol=PROTOCOL):
    leases = []

    def factory(*args):
        lease = Lease(worker_protocol)
        leases.append(lease)
        return lease

    owner = NS(
        config=NS(spec=SPEC),
        outgoing=asyncio.Queue(),
        host=NS(make_segment_leases=factory),
        user_id="user",
        session_id=uuid4().hex,
        source_epoch=uuid4().hex,
    )
    root = PlayStartup(owner, request(with_frame) if value is None else value)
    root.launch(
        dict(
            type="init", source_stream=dict(format="rgb24", width=32, height=32, fps=24)
        )
    )
    return root, leases


@pytest.mark.parametrize("with_frame", [True, False])
def test_optional_s0_and_one_lease_handoff(with_frame):
    async def run():
        root, leases = startup(with_frame)
        await root.wait(root.admitted)
        if with_frame:
            await root.wait(root.prepared)
        else:
            await asyncio.sleep(0)
            assert not root.prepared.is_set()
            assert [x["type"] for x in leases[0].ws.commands] == ["init"]
        await root.bind_cli(RGB, dict(protocol="room-source-frame-v1"))
        assert len(leases) == 1
        async with root.open() as (ws, assignment):
            pending = asyncio.create_task(root.attach())
            await asyncio.sleep(0)
            assert not pending.done()
            root.rtc_ready.set()
            await pending
            assert [m.get("kind") for m in ws.commands] == [
                None,
                "source.prepare",
                "source.attach",
            ]
            assert ws.commands[1]["payload"]["rgb"] == RGB
            assert all(m.get("kind") != "source.append" for m in ws.commands)
            root.started = True
        assert leases[0].opens == leases[0].closes == 1
        await root.close()
        assert root.resource_state == "released" and leases[0].closes == 1

    asyncio.run(run())


def test_fallback_releases_old_attempt_and_keeps_original_manifest():
    async def run():
        root, leases = startup()
        await root.wait(root.prepared)
        before = dict(root.ids)
        requested_hash = root.request["initial_frame"]["rgb_sha256"]
        await root.bind_cli(bytes(len(RGB)), dict(protocol="room-source-frame-v1"))
        assert (
            len(leases) == 2
            and leases[0].closes == 1
            and root.ids != before
            and root.attempt == 1
        )
        assert root.request["initial_frame"]["rgb_sha256"] == requested_hash
        assert root.rgb == bytes(len(RGB))
        await root.close()
        assert leases[1].closes == 1

    asyncio.run(run())


def test_cancel_without_first_frame_releases_pre_reserved_worker():
    async def run():
        root, leases = startup(False)
        await root.wait(root.admitted)
        await root.close()
        await root.close()
        assert root.resource_state == "released" and leases[0].closes == 1

    asyncio.run(run())


@pytest.mark.parametrize("protocol", [PROTOCOL, V2_PROTOCOL])
def test_chunk_upload_manifest_is_bounded_ordered_and_integrity_checked(protocol):
    req = request(protocol=protocol)
    frame = req["initial_frame"]
    raw = frame.pop("payload")
    spec = SPEC if protocol == PROTOCOL else NS(width=64, height=32)
    assets = StartupAssets(req, spec)
    assert not assets.committed.is_set()
    with pytest.raises(RenderError):
        assets.chunk("unknown", 0, b"a")
    assert assets.chunk("__source_s0__", 0, raw[:10]) == 10
    assert assets.chunk("__source_s0__", 0, raw[:10]) == 10
    with pytest.raises(RenderError):
        assets.chunk("__source_s0__", 0, b"wrong")
    with pytest.raises(RenderError):
        assets.commit()
    assets.chunk("__source_s0__", 10, raw[10:])
    assets.commit()
    assets.commit()
    assert assets.rgb == RGB and assets.references == []


def test_room_pins_one_live_view_without_changing_world_clock():
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import EngineFrame

    async def run():
        root, leases = startup(False)
        root.owner.engine = NS(metadata={"world_build_hash": "world-v1"})
        root.owner.source_identity = lambda: dict(
            source_session_id=root.owner.session_id,
            source_epoch=root.owner.source_epoch,
            view_id="view-a",
        )
        await root.wait(root.admitted)

        def frame(index, tick, rgb=RGB):
            return EngineFrame(
                index,
                index * 1_000_000 // 24,
                rgb,
                {},
                provenance={
                    "cli_frame_header": {
                        "source": {
                            "presentationId": "p",
                            "sdkEpoch": 3,
                            "worldRevision": 4,
                            "simulationTick": tick,
                            "sourceTimeUs": tick * 1000,
                        }
                    }
                },
            )

        first = frame(0, 407)
        root.observe_source(1, first, 1.0)
        for i in range(1, 150):
            root.observe_source(i + 1, frame(i, 407 + i, bytes(len(RGB))), 1.0 + i)
        assert root.pinned_source == (1, (first, 1.0))
        assert root.cli_proof["source"]["simulationTick"] == 407
        await root.bind_cli(root.cli_rgb, root.cli_proof)
        assert root.rgb == RGB and len(leases) == 1
        root.source_started()
        assert root.pinned_source is None
        root.observe_source(200, frame(199, 900), 200)
        assert root.pinned_source is None  # Never repin a later source frame.
        await root.close()

    asyncio.run(run())


@pytest.mark.parametrize("protocol", [PROTOCOL, V2_PROTOCOL])
@pytest.mark.parametrize("with_frame", [True, False])
@pytest.mark.parametrize("mismatch", [True, False])
def test_source_segment_handoff_runs_real_room_pipeline_once(
    tmp_path, monkeypatch, with_frame, mismatch, protocol
):
    # This fixture provides a legacy child lease, not a bound parent slot.
    monkeypatch.delenv("WHITEBOX_BOUND_COMPUTE", raising=False)
    from test_whitebox_interactive import BrowserUI
    from test_whitebox_render_lifecycle import OrderedWorker

    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncArchiveManager,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
        EngineFrame,
        VideoSpec,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderJournal,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig

    monkeypatch.setenv("WHITEBOX_MODEL_STARTUP_PREPARE", "1")

    async def run():
        spec = VideoSpec(32, 32)
        cli_loaded = asyncio.Event()
        worker_prepared = asyncio.Event()
        allow_prepare = asyncio.Event()
        workers = []
        active = 0
        released = 0
        producer = None

        actual_rgb = bytes(len(RGB)) if mismatch else RGB

        class LiveView:
            stream_id = "view"
            metadata = {}

            def __init__(self):
                self.queue = asyncio.Queue()
                self.closed = False
                self.index = 0

            async def push(self):
                i = self.index
                self.index += 1
                await self.queue.put(
                    EngineFrame(
                        i,
                        spec.pts(i),
                        actual_rgb,
                        {},
                        provenance=dict(
                            capture_pts_us=i * 50000,
                            cli_frame_header=dict(
                                outputFrameId=i + 1,
                                source=dict(
                                    presentationId="p",
                                    sdkEpoch=1,
                                    worldRevision=1,
                                    simulationTick=500 + i,
                                    sourceTimeUs=1000000 + i * 50000,
                                ),
                            ),
                        ),
                    )
                )

            async def frames(self):
                while (frame := await self.queue.get()) is not None:
                    yield frame

            async def send_input(self, *args):
                pass

            async def finish(self):
                await self.queue.put(None)

            async def close(self):
                self.closed = True

        engine = LiveView()

        class PreparedWorker(OrderedWorker):
            async def emit(self, message):
                if message["type"] == "session.ready":
                    message["source_prepare_protocol"] = protocol
                    message["source_prepare_profile_sha256"] = "a" * 64
                await super().emit(message)

            async def send(self, wire):
                message = msgspec.msgpack.decode(wire)
                if message.get("kind") in {"source.prepare", "source.attach"}:
                    if message["kind"] == "source.prepare":
                        worker_prepared.set()
                        assert message["payload"]["rgb"] in {RGB, actual_rgb}
                        await allow_prepare.wait()
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
                nonlocal active, released
                active += 1
                assert active == 1
                self.resource_state = "reserved"
                worker = PreparedWorker(spec)
                workers.append(worker)
                try:
                    yield (
                        worker,
                        NS(
                            session_id="m",
                            generation_id="g",
                            denoiser=NS(
                                worker_id="exclusive",
                                worker_epoch="w",
                                model_revision="test",
                            ),
                        ),
                    )
                finally:
                    await worker.close()
                    active -= 1
                    released += 1
                    self.resource_state = "released"

        async def factory(**kwargs):
            nonlocal producer
            # The trusted source descriptor selects the condition writer before
            # acquiring a model lease, even for a legacy source.
            assert active == 0
            assert "startup_gate" not in kwargs  # Existing CLI interface only.
            await cli_loaded.wait()

            async def produce():
                while not engine.closed:
                    await engine.push()
                    await asyncio.sleep(0.025)

            producer = asyncio.create_task(produce())
            return engine

        archives = AsyncArchiveManager(recorder_slots=3)
        journal = RenderJournal(
            MemoryRoomStore(),
            user_id="user",
            source_session_id="source",
            source_epoch="epoch",
            owner_epoch="room",
        )
        await journal.create()
        host = NS(runtime=NS(players={}), make_segment_leases=lambda *_: Model())
        source = SourceSession(
            host,
            journal,
            source_epoch="epoch",
            user_id="user",
            session_id="source",
            trace_id="source",
            config=WhiteboxConfig(
                engine_factory=factory,
                ui_factory=lambda **_: BrowserUI(),
                spool=tmp_path,
                spec=spec,
                room_owned=True,
                startup_seconds=20,
                drain_seconds=5,
            ),
            archives=archives,
        )
        messages = []

        async def consume():
            while True:
                message = msgspec.msgpack.decode(await source.recv())
                messages.append(message)
                if message.get("type") in {"whitebox.frame", "frame_batch"}:
                    source.media_delivered()

        task = asyncio.create_task(consume())
        try:
            await source.send(
                msgspec.msgpack.encode(
                    dict(
                        type="init",
                        whitebox=dict(world_id="world"),
                        prompt="Source preview",
                        startup=request(with_frame, protocol=protocol),
                    )
                )
            )
            source.startup.rtc_ready.set()
            assert source.startup.task is None
            cli_loaded.set()
            await asyncio.wait_for(source.ready.wait(), 10)
            await source.startup.wait(source.startup.admitted)
            # Source-first admission has already opened the CLI. With no
            # supplied frame, preparation now uses its pinned first capture.
            await asyncio.wait_for(worker_prepared.wait(), 2)
            # CLI readiness survives the original startup attempt being replaced
            # by the explicit v1 mismatch fallback. Observe the source event.
            await asyncio.wait_for(source.startup.cli_ready.wait(), 10)
            # Source capture, recording and preview advance even though model
            # preparation is blocked. There is no SDK capture/release endpoint.
            async with asyncio.timeout(5):
                while source.submitted < 8 or source.preview_count < 2:
                    await asyncio.sleep(0.01)
            pinned = source.startup.pinned_source
            assert pinned[1][0].index == 0
            assert (
                pinned[1][0].provenance["cli_frame_header"]["source"]["simulationTick"]
                == 500
            )
            allow_prepare.set()
            async with asyncio.timeout(20):
                while (
                    not source.current_segment
                    or not source.current_segment.first_decodable
                ):
                    if any(m.get("type") == "error" for m in messages):
                        raise AssertionError(messages)
                    await asyncio.sleep(0.01)
            attempts = 2 if protocol == PROTOCOL and with_frame and mismatch else 1
            assert len(workers) == attempts and active == 1 and source.startup.started
            assert released == attempts - 1
            appends = [
                m for m in workers[-1].events if m.get("kind") == "source.append"
            ]
            assert (
                appends[0]["payload"]["source_frame_start"] == 0
                and appends[0]["payload"]["data"] == actual_rgb
            )
            assert (
                source.current_segment.recording.manifest["startup"]["level"]
                == "condition"
            )
        finally:
            if producer:
                producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)
            await source.close()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await archives.close()
        assert active == 0 and released == attempts and engine.closed

    asyncio.run(run())


def test_implicit_asset_commit_can_be_acknowledged_after_fast_handoff():
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderPipeline,
    )

    value = request(False)
    assets = StartupAssets(value, SPEC)
    pipeline = RenderPipeline.__new__(RenderPipeline)
    sub = NS(closed=False, generation=1)
    pipeline.subscription = sub
    pipeline.generation = 1
    pipeline.session = NS(
        startup=NS(request=value, assets=assets, transferred=True, closed=False)
    )
    replies = []
    pipeline.publish = replies.append
    pipeline.accept(
        sub,
        msgspec.msgpack.encode(
            dict(type="startup.assets.commit", request_id=value["request_id"])
        ),
    )
    assert replies == [
        dict(type="startup.assets.committed", request_id=value["request_id"])
    ]


def test_same_connection_init_retry_is_idempotent_and_fences_changed_content():
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderPipeline,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.startup import (
        startup_fingerprint,
    )

    value = request(False)
    pipeline = RenderPipeline.__new__(RenderPipeline)
    sub = NS(closed=False, generation=1)
    pipeline.subscription = sub
    pipeline.generation = 1
    pipeline.init = dict(whitebox=dict(source_id="s", world_id="w"))
    pipeline.session = NS(
        startup=NS(
            request=value, request_fingerprint=startup_fingerprint(value), closed=False
        )
    )
    replies = []
    pipeline.publish = replies.append
    message = dict(
        type="init", whitebox=dict(source_id="s", world_id="w"), startup=value
    )
    for _ in range(2):
        pipeline.accept(sub, msgspec.msgpack.encode(message))
    assert len(replies) == 2 and all(r["duplicate"] for r in replies)
    message["startup"] = {
        **value,
        "initial_render": dict(prompt="changed", references=[]),
    }
    with pytest.raises(RenderError) as error:
        pipeline.accept(sub, msgspec.msgpack.encode(message))
    assert error.value.code == "REQUEST_ID_CONFLICT"


def test_missing_cli_frame_times_out_and_does_not_keep_a_lease():
    import time

    async def run():
        root, leases = startup(False)
        root.deadline = time.monotonic() + 0.05
        with pytest.raises(TimeoutError):
            await root.task
        assert (
            not root.prepared.is_set()
            and leases[0].closes == 1
            and root.resource_state == "released"
        )
        await root.close()

    asyncio.run(run())


def test_unknown_release_prevents_fallback_allocation():
    async def run():
        root, leases = startup()
        await root.wait(root.prepared)
        original = root.close_attempt

        async def uncertain():
            await original()
            leases[0].resource_state = "release_unknown"

        root.close_attempt = uncertain
        with pytest.raises(RenderError) as failure:
            await root.bind_cli(
                bytes(len(RGB)),
                dict(protocol="room-source-frame-v1"),
            )
        assert (
            failure.value.code == "RELEASE_UNKNOWN"
            and len(leases) == 1
            and root.attempt == 0
        )
        await root.close()

    asyncio.run(run())


def test_preparing_or_cancelling_one_player_does_not_reset_shared_world(tmp_path):
    """Real Room ownership, existing shared driver interface, mocked GPU peer."""
    import time
    from dataclasses import asdict

    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncArchiveManager,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_mock import MockRoom
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        MemoryRoomStore,
        RoomDirectory,
        new_id,
    )

    async def run():
        world = MockRoom()
        factory_calls = []

        async def factory(**kwargs):
            # No new CLI argument and no per-model world clock/capture controls.
            assert set(kwargs) <= {"world_id", "spec", "trace_id", "cli_fps"}
            factory_calls.append(kwargs)
            return world

        spec = VideoSpec(32, 32)
        archives = AsyncArchiveManager(recorder_slots=3)
        directory = RoomDirectory(MemoryRoomStore(), {"room": "http://room"})
        runtime = RoomRuntime(
            directory=directory,
            owner_id="room",
            driver_factory=factory,
            spool=tmp_path,
            spec=spec,
            archives=archives,
            max_players=2,
        )
        game = new_id()
        await directory.resolve(game, "world", create=True)

        def body():
            return dict(
                game_id=game,
                player_id=new_id(),
                stream_id=new_id(),
                session_id=new_id(),
                user_id=new_id(),
                world_id="world",
                trace_id="shared-startup",
                spec=asdict(spec),
                startup_prepare=True,
                startup_expires_at=time.time() + 60,
            )

        a, b = body(), body()
        root, leases = startup(False)
        try:
            await runtime.join(a)
            await runtime.join(b)
            await root.wait(root.admitted)
            assert not root.prepared.is_set()  # Player A awaits its source frame.
            await runtime.event(
                b["stream_id"], "input", {"sequence": 0, "input": {"key": "W"}}
            )
            first = await runtime.next_frame(b["stream_id"])
            assert first["states"]["tick"] == 1 and len(first["states"]["players"]) == 2
            await root.close()
            await runtime.cancel_startup(a["stream_id"])
            assert leases[0].closes == 1 and len(factory_calls) == 1
            assert game in runtime.rooms and b["stream_id"] in runtime.players
            assert (
                b["player_id"] in world.players and a["player_id"] not in world.players
            )
            await runtime.event(
                b["stream_id"], "input", {"sequence": 1, "input": {"key": "D"}}
            )
            following = await runtime.next_frame(b["stream_id"])
            assert following["states"]["tick"] == 2
            assert following["index"] == first["index"] + 1
            assert (
                following["provenance"]["room_source_time_us"]
                > first["provenance"]["room_source_time_us"]
            )
            assert following["states"]["players"] == [b["player_id"]]
        finally:
            await root.close()
            await runtime.close()
            await archives.close()

    asyncio.run(run())


def test_unprovable_initial_state_assertion_is_rejected_before_admission():
    value = request()
    value["initial_state_id"] = "old-tick-zero-assertion"
    from sglang.multimodal_gen.runtime.realtime.whitebox.startup import validate_startup

    with pytest.raises(RenderError) as error:
        validate_startup(value, SPEC)
    assert error.value.code == "CAPABILITY_UNAVAILABLE"


@pytest.mark.parametrize("cancel_during_wait", [False, True])
def test_fallback_waits_for_heartbeat_after_verified_worker_release(cancel_during_wait):
    """A release receipt clears the lease before the busy heartbeat expires."""
    from dataclasses import replace

    from sglang.multimodal_gen.runtime.realtime.coordinator import (
        InMemoryCoordinatorStore,
        RealtimeCoordinator,
        WorkerHeartbeat,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import ModelLeases

    async def run():
        store = InMemoryCoordinatorStore(ttl_s=30, worker_ttl_s=30)
        heartbeat = WorkerHeartbeat(
            worker_id="worker",
            worker_epoch="e" * 32,
            role="denoiser",
            endpoint="ws://worker/generate",
            reservation_endpoint="http://worker",
            az="shanghai",
            capacity=1,
            model_revision="model",
            vae_fingerprint="local",
        )
        await store.heartbeat(heartbeat)
        updates = []
        reservations = []

        class Reservations:
            async def reserve(self, slot, **kwargs):
                reservations.append(kwargs["token"])
                await store.heartbeat(replace(heartbeat, active_sessions=1))

            async def release_verified(self, slot, *, token):
                async def next_heartbeat():
                    await asyncio.sleep(0.1 if cancel_during_wait else 0.05)
                    await store.heartbeat(heartbeat)

                updates.append(asyncio.create_task(next_heartbeat()))

        coordinator = RealtimeCoordinator(store, reservation_client=Reservations())

        class Client:
            async def admit(self, **kwargs):
                kwargs.pop("trace_id")
                return await coordinator.admit(**kwargs)

            async def release_verified(self, assignment):
                await coordinator.release_verified(assignment)

            async def renew(self, assignment):
                return await coordinator.renew(assignment)

        class ClosingPeer(Peer):
            async def close(self):
                pass

        async def connect(*args, **kwargs):
            return ClosingPeer()

        root, leases = startup()

        def factory(user_id, segment):
            lease = ModelLeases(
                Client(),
                connect,
                user_id=user_id,
                model_revision="model",
                vae_fingerprint="local",
                trace_id=segment["segment_id"],
                max_bytes=4 << 20,
                strict_release=True,
                session_id=segment["model_session_id"],
                generation_id=segment["generation_id"],
                capacity_wait_ms=segment["capacity_wait_ms"],
            )
            leases.append(lease)
            return lease

        root.owner.host.make_segment_leases = factory
        try:
            await root.wait(root.prepared)
            fallback = asyncio.create_task(
                root.bind_cli(bytes(len(RGB)), {"protocol": "room-source-frame-v1"})
            )
            if cancel_during_wait:
                async with asyncio.timeout(1):
                    while not store._waiting:
                        await asyncio.sleep(0.001)
                await root.close()
                await asyncio.gather(fallback, return_exceptions=True)
                assert root.closed and leases[1].resource_state == "released"
            else:
                await fallback
                assert leases[1].resource_state == "reserved"
                assert root.rgb == bytes(len(RGB))
            assert root.attempt == 1 and len(reservations) == 2
            assert leases[0].resource_state == "released"
        finally:
            await root.close()
            await asyncio.gather(*updates)
        assert all(lease.resource_state == "released" for lease in leases)
        assert not store._assignments_by_token
        assert not store._waiting

    asyncio.run(run())


@pytest.mark.parametrize("reference_count", [0, 21])
def test_v2_native_condition_preview_keeps_one_lease_and_original_cli_source(
    reference_count,
):
    from sglang.multimodal_gen.runtime.realtime.source_prepare import V2_PROTOCOL

    async def run():
        value = request()
        value["protocol"] = V2_PROTOCOL
        value.pop("fallback_policy")
        pixels = bytes([255, 0, 0]) * 16 * 48
        image = io.BytesIO()
        Image.new("RGB", (16, 48), "red").save(image, format="PNG")
        raw = image.getvalue()
        value["initial_frame"].update(
            width=16,
            height=48,
            payload=raw,
            encoded_size=len(raw),
            encoded_sha256=hashlib.sha256(raw).hexdigest(),
            rgb_sha256=hashlib.sha256(pixels).hexdigest(),
        )
        value["initial_render"]["references"] = [
            dict(
                id=f"reference-{i}",
                format="png",
                payload=raw,
                encoded_size=len(raw),
                encoded_sha256=hashlib.sha256(raw).hexdigest(),
            )
            for i in range(reference_count)
        ]
        root, leases = startup(value=value, worker_protocol=V2_PROTOCOL)
        try:
            await root.wait(root.prepared)
            before = dict(root.ids)
            cli = bytes(SPEC.frame_bytes)
            await root.bind_cli(cli, dict(protocol="room-source-frame-v1"))
            assert len(leases) == 1 and root.ids == before and root.attempt == 0
            init, prepare = leases[0].ws.commands
            assert len(init["references"]) == reference_count
            assert root.assets.reference_ids == [
                f"reference-{i}" for i in range(reference_count)
            ]
            assert init["source_prepare_protocol"] == V2_PROTOCOL
            assert prepare["payload"]["rgb"] == pixels
            assert (prepare["payload"]["width"], prepare["payload"]["height"]) == (
                16,
                48,
            )
            receipt = root.receipt()
            assert receipt["protocol"] == V2_PROTOCOL
            assert receipt["condition_preview"] == dict(
                origin="request",
                width=16,
                height=48,
                rgb_sha256=hashlib.sha256(pixels).hexdigest(),
            )
            assert receipt["cli_frame_rgb_sha256"] == hashlib.sha256(cli).hexdigest()
            root.rtc_ready.set()
            async with root.open():
                await root.attach()
            assert [m.get("kind") for m in leases[0].ws.commands] == [
                None,
                "source.prepare",
                "source.attach",
            ]
            assert leases[0].opens == leases[0].closes == 1
        finally:
            await root.close()

    asyncio.run(run())


def test_v2_rejects_legacy_fallback_and_never_downgrades_worker_protocol():
    from sglang.multimodal_gen.runtime.realtime.source_prepare import V2_PROTOCOL

    async def run():
        value = request(False)
        value["protocol"] = V2_PROTOCOL
        with pytest.raises(RenderError) as invalid:
            startup(value=value)
        assert invalid.value.code == "INVALID_COMMAND"
        value.pop("fallback_policy")
        root, leases = startup(value=value, worker_protocol=PROTOCOL)
        try:
            with pytest.raises(RenderError) as unsupported:
                await root.wait(root.prepared)
            assert unsupported.value.code == "CAPABILITY_UNAVAILABLE"
            assert len(leases[0].ws.commands) == 1
            assert leases[0].ws.commands[0]["source_prepare_protocol"] == V2_PROTOCOL
        finally:
            await root.close()

    asyncio.run(run())


def test_v2_segment_starts_with_pinned_cli_pixels_not_condition_preview():
    from unittest.mock import AsyncMock

    from sglang.multimodal_gen.runtime.realtime.source_prepare import V2_PROTOCOL
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import EngineFrame
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )

    async def run():
        segment = object.__new__(RenderSegment)
        segment.sink = NS(auto_policy=None)
        native = bytes(SPEC.frame_bytes)
        segment.leases = NS(
            is_startup=True,
            timing_reports=[],
            protocol=V2_PROTOCOL,
            rgb=b"independent-preview",
            cli_rgb_sha256=hashlib.sha256(native).hexdigest(),
        )
        segment.resume_condition_evidence = None
        segment.recording = NS(manifest={}, configure=AsyncMock())
        segment.segment = dict(start_boundary=dict(mode="next_available"))
        segment.assignment = NS(
            session_id="model",
            generation_id="generation",
            denoiser=NS(worker_id="worker", worker_epoch="epoch"),
        )
        segment.owner = NS(
            submitted=1,
            segment_start_frame=AsyncMock(),
            before_segment_source=AsyncMock(),
        )
        await segment.before_model_source(None, EngineFrame(0, 0, native, {}), 0)
        assert (
            segment.recording.manifest["start_boundary_actual"]["start_frame_sha256"]
            == hashlib.sha256(native).hexdigest()
        )
        with pytest.raises(RenderError) as wrong:
            await segment.before_model_source(
                None, EngineFrame(0, 0, bytes([1]) * len(native), {}), 0
            )
        assert wrong.value.code == "SOURCE_EPOCH_MISMATCH"

    asyncio.run(run())


@pytest.mark.parametrize(
    "count,revision", [(3, "ordinary"), (4, "large-ref"), (21, "large-ref")]
)
def test_original_large_reference_routing_keeps_one_model_lease(
    monkeypatch, count, revision
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import RenderHost

    monkeypatch.setenv("WHITEBOX_ROUTING_REVISION", "ordinary")
    monkeypatch.setenv("WHITEBOX_LARGE_REFERENCE_MODEL_REVISION", "large-ref")
    monkeypatch.setenv("WHITEBOX_DEFAULT_REFERENCE_CAPACITY", "3")
    host = NS(coordinator=None, connect=None, config=NS(max_message_bytes=64 << 20))
    lease = RenderHost.make_segment_leases(
        host,
        "user",
        dict(
            segment_id="segment",
            model_session_id="model",
            generation_id="generation",
            capacity_wait_ms=0,
            references=[str(i) for i in range(count)],
        ),
    )
    assert lease.request["model_revision"] == revision
    assert (
        lease.request["wait_for_capacity"] is False and lease.resource_state == "none"
    )


@pytest.mark.parametrize(
    "reply, upstream_code",
    [
        (
            {
                "type": "error",
                "code": "INVALID_COMMAND",
                "content": "private /model/path",
            },
            "INVALID_COMMAND",
        ),
        (
            {
                "type": "error",
                "error": {"code": "out_of_memory", "message": "private GPU details"},
            },
            "out_of_memory",
        ),
        ({"type": "error", "code": "Bearer private-token"}, None),
        ({"type": "unexpected", "payload": b"private pixels"}, None),
        ([], None),
    ],
)
def test_model_prepare_failure_keeps_origin_without_exposing_peer_body(
    reply, upstream_code, caplog
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import public_error

    async def run():
        root = object.__new__(PlayStartup)
        root.event_sequence = 0
        root.identity = {"preparation_id": "prep"}
        root.owner = NS(trace_id="test-trace")
        root.ids = {"preparation_id": "prep"}
        root.ws = Peer()

        # Use the actual event exchange, with a peer returning the failed response.
        async def send(_):
            await root.ws.incoming.put(msgspec.msgpack.encode(reply))

        root.ws.send = send
        with pytest.raises(Exception) as failure:
            await root._event("source.prepare", {}, "source.prepared")
        result = public_error(failure.value, "test-trace")
        assert result["code"] == "MODEL_PREPARATION_FAILED"
        assert result["stage"] == "model"
        assert result.get("upstream_code") == upstream_code
        assert "private" not in str(result)
        assert "model" in result["content"].lower()
        assert "test-trace" in caplog.text

    asyncio.run(run())


def test_model_prepare_known_empty_failure_is_never_an_invalid_client_command():
    from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import public_error

    result = public_error(RenderError("INITIALIZATION_FAILED"), "trace")
    assert result["code"] == "INITIALIZATION_FAILED"
    assert result["stage"] == "model"


@pytest.mark.parametrize(
    "error",
    [ConnectionError("private endpoint"), ValueError("private malformed response")],
)
@pytest.mark.parametrize("phase", ["send", "receive"])
def test_model_preparation_transport_and_decode_failure_keep_model_stage(error, phase):
    from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import public_error

    async def run():
        root = object.__new__(PlayStartup)
        root.event_sequence = 0
        root.owner = NS(trace_id="trace")
        root.ws = Peer()

        async def recv():
            raise error

        root.ws.recv = recv

        async def send(_):
            if phase == "send":
                raise error

        root.ws.send = send
        with pytest.raises(Exception) as failure:
            await root._event("source.prepare", {}, "source.prepared")
        diagnostic = public_error(failure.value, "trace")
        assert diagnostic["code"] == "MODEL_PREPARATION_FAILED"
        assert diagnostic["stage"] == "model"
        assert "private" not in str(diagnostic)

    asyncio.run(run())


@pytest.mark.parametrize("failed_operation", ["init", "source.prepare"])
def test_failed_model_preparation_releases_the_same_lease(
    monkeypatch, failed_operation
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import public_error

    original = Peer.send

    async def send(peer, raw):
        message = msgspec.msgpack.decode(raw)
        operation = message.get("kind", message["type"])
        if operation == failed_operation:
            await peer.incoming.put(
                msgspec.msgpack.encode(
                    {
                        "type": "error",
                        "error": {
                            "code": "model_prepare_failed",
                            "message": "private body",
                        },
                    }
                )
            )
        else:
            await original(peer, raw)

    monkeypatch.setattr(Peer, "send", send)

    async def run():
        root, leases = startup()
        try:
            with pytest.raises(Exception) as failure:
                await root.wait(root.prepared)
            result = public_error(failure.value, "trace")
            assert result["stage"] == "model"
            assert result["upstream_code"] == "model_prepare_failed"
        finally:
            await root.close()
        assert leases[0].opens == leases[0].closes == 1

    asyncio.run(run())


def test_unknown_render_failure_does_not_blame_client_command():
    assert RenderError("UNRECOGNIZED_SERVICE_ERROR").code == "RENDER_FAILED"


@pytest.mark.parametrize("frame_size", [(32, 32), (960, 540)])
def test_provided_preview_does_not_reprepare_for_different_cli_pixels(frame_size):
    async def run():
        root, leases = startup(
            value=request(frame_size=frame_size, protocol=V2_PROTOCOL),
            worker_protocol=V2_PROTOCOL,
        )
        try:
            await root.wait(root.prepared)
            identity, before = dict(root.identity), dict(root.ids)
            preview = bytes([255, 0, 0]) * frame_size[0] * frame_size[1]
            actual = bytes(len(RGB))
            await root.bind_cli(actual, dict(protocol="room-source-frame-v1"))
            assert len(leases) == 1 and leases[0].closes == 0
            assert (
                root.identity == identity and root.ids == before and root.attempt == 0
            )
            assert root.rgb == root.assets.rgb == preview
            command = leases[0].ws.commands[1]["payload"]
            assert (command["width"], command["height"]) == frame_size
            assert command["rgb"] == preview and root.cli_rgb == actual
            receipt = root.receipt()
            assert receipt["condition_preview"]["origin"] == "request"
            assert receipt["cli_frame_rgb_sha256"] == hashlib.sha256(actual).hexdigest()
            assert not any(event["phase"] == "fallback" for event in root.events)
        finally:
            await root.close()
        assert leases[0].closes == 1 and root.resource_state == "released"

    asyncio.run(run())


def test_unknown_release_is_reported_without_readmission():
    async def run():
        root, leases = startup(
            value=request(protocol=V2_PROTOCOL), worker_protocol=V2_PROTOCOL
        )
        await root.wait(root.prepared)
        original = root.close_attempt

        async def uncertain():
            await original()
            leases[0].resource_state = "release_unknown"

        root.close_attempt = uncertain
        await root.bind_cli(bytes(len(RGB)), dict(protocol="room-source-frame-v1"))
        assert len(leases) == 1 and root.attempt == 0
        await root.close()
        assert root.resource_state == "release_unknown" and len(leases) == 1

    asyncio.run(run())


def test_mismatching_cli_reuses_lease_while_worker_heartbeat_is_busy():
    """S0/CLI differences do not race for a second coordinator admission."""
    from dataclasses import replace

    from sglang.multimodal_gen.runtime.realtime.coordinator import (
        InMemoryCoordinatorStore,
        RealtimeCoordinator,
        WorkerHeartbeat,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import ModelLeases

    async def run():
        store = InMemoryCoordinatorStore(ttl_s=30, worker_ttl_s=30)
        heartbeat = WorkerHeartbeat(
            worker_id="worker",
            worker_epoch="e" * 32,
            role="denoiser",
            endpoint="ws://worker/generate",
            reservation_endpoint="http://worker",
            az="shanghai",
            capacity=1,
            model_revision="model",
            vae_fingerprint="local",
        )
        await store.heartbeat(heartbeat)
        updates = []
        reservations = []

        class Reservations:
            async def reserve(self, slot, **kwargs):
                reservations.append(kwargs["token"])
                await store.heartbeat(replace(heartbeat, active_sessions=1))

            async def release_verified(self, slot, *, token):
                async def next_heartbeat():
                    await asyncio.sleep(0.01)
                    await store.heartbeat(heartbeat)

                updates.append(asyncio.create_task(next_heartbeat()))

        coordinator = RealtimeCoordinator(store, reservation_client=Reservations())

        class Client:
            async def admit(self, **kwargs):
                kwargs.pop("trace_id")
                return await coordinator.admit(**kwargs)

            async def release_verified(self, assignment):
                await coordinator.release_verified(assignment)

            async def renew(self, assignment):
                return await coordinator.renew(assignment)

        class ClosingPeer(Peer):
            async def close(self):
                pass

        async def connect(*args, **kwargs):
            return ClosingPeer(V2_PROTOCOL)

        root, leases = startup(
            value=request(protocol=V2_PROTOCOL), worker_protocol=V2_PROTOCOL
        )

        def factory(user_id, segment):
            lease = ModelLeases(
                Client(),
                connect,
                user_id=user_id,
                model_revision="model",
                vae_fingerprint="local",
                trace_id=segment["segment_id"],
                max_bytes=4 << 20,
                strict_release=True,
                session_id=segment["model_session_id"],
                generation_id=segment["generation_id"],
                capacity_wait_ms=segment["capacity_wait_ms"],
            )
            leases.append(lease)
            return lease

        root.owner.host.make_segment_leases = factory
        try:
            await root.wait(root.prepared)
            identity = dict(root.identity)
            await root.bind_cli(bytes(len(RGB)), {"protocol": "room-source-frame-v1"})
            assert root.identity == identity
            assert root.attempt == 0 and len(reservations) == 1
            assert len(leases) == 1 and leases[0].resource_state == "reserved"
            assert not store._waiting
        finally:
            await root.close()
            await asyncio.gather(*updates)
        assert all(lease.resource_state == "released" for lease in leases)
        assert not store._assignments_by_token
        assert not store._waiting

    asyncio.run(run())


def test_video_start_still_rejects_a_changed_pinned_cli_frame():
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )

    segment = RenderSegment.__new__(RenderSegment)
    segment.segment = {}
    segment.leases = NS(
        is_startup=True,
        timing_reports=[],
        protocol=V2_PROTOCOL,
        cli_rgb_sha256=hashlib.sha256(RGB).hexdigest(),
    )

    async def run():
        with pytest.raises(RenderError) as error:
            await segment.before_model_source(None, NS(rgb=bytes(len(RGB))), 0)
        assert error.value.code == "SOURCE_EPOCH_MISMATCH"

    asyncio.run(run())


@pytest.mark.parametrize("generation", ["generation", "foreign-generation"])
def test_preparation_timing_stays_private_until_matching_render_handoff(generation):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from sglang.multimodal_gen.runtime.utils.zing_ttff_timing import InferenceTiming

    async def run():
        startup = object.__new__(PlayStartup)
        startup.event_sequence = 0
        startup.identity = {"preparation": "same"}
        startup.assignment = SimpleNamespace(
            session_id="session", generation_id="generation"
        )
        startup.timing_reports = []
        timing = InferenceTiming("encoder", prompt_version=1)
        timing.record("qwen", 12.0)
        report = timing.snapshot()
        startup._model_send = AsyncMock()
        startup._model_reply = AsyncMock(
            return_value={
                "type": "source.prepared",
                "identity": startup.identity,
                "timing_reports": [
                    {
                        "model_session_id": "session",
                        "generation_id": generation,
                        "timing": report,
                    }
                ],
            }
        )
        await startup._event("source.prepare", {}, "source.prepared")
        assert len(startup.timing_reports) == (1 if generation == "generation" else 0)
        if startup.timing_reports:
            buffered = startup.timing_reports[0]
            assert buffered["type"] == "model.timing"
            assert buffered["timing"]["report_id"] == report["report_id"]
            assert buffered["timing"]["stages"][0]["host_ms"] == 12.0

    asyncio.run(run())
