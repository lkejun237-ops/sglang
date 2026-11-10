# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import msgspec
import pytest
from test_whitebox_interactive import BrowserUI, Engine, Worker

from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
    AsyncArchiveManager,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
    RenderError,
    RenderJournal,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig


def test_source_ready_without_frames_does_not_start_input_idle(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.player_idle import (
        PlayerIdlePolicy,
        PlayerInputActivity,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime

    async def run():
        spec = VideoSpec(32, 32)
        engine = Engine(spec)
        engine.stream_id = "view"
        activity = PlayerInputActivity()
        entry = dict(
            input_activity=activity,
            identity={"stream_id": "view"},
            trace_id="first-frame",
        )
        runtime = SimpleNamespace(
            players={"view": entry},
            input_idle_policy=PlayerIdlePolicy(tmp_path / "idle.json"),
        )
        runtime.input_idle_expired = lambda value: RoomRuntime.input_idle_expired(
            runtime, value
        )
        archives = AsyncArchiveManager(recorder_slots=3)
        journal = RenderJournal(
            MemoryRoomStore(),
            user_id="user",
            source_session_id="source",
            source_epoch="epoch",
            owner_epoch="room",
        )
        await journal.create()

        async def factory(**_):
            return engine

        source = SourceSession(
            SimpleNamespace(runtime=runtime),
            journal,
            source_epoch="epoch",
            user_id="user",
            session_id="source",
            trace_id="first-frame",
            config=WhiteboxConfig(
                engine_factory=factory,
                ui_factory=lambda **_: BrowserUI(),
                spool=tmp_path,
                spec=spec,
                room_owned=True,
                startup_seconds=5,
                drain_seconds=2,
            ),
            archives=archives,
        )
        try:
            await source.send(
                msgspec.msgpack.encode(
                    dict(type="init", whitebox={"world_id": "world"}, prompt="first")
                )
            )
            await asyncio.wait_for(source.ready.wait(), 3)
            assert source.source_state_value == "ready"
            assert source.submitted == 0
            assert entry["explicit_source_owner"]
            assert activity.last_input is None
            assert not activity.is_expired(time.monotonic() + 61, 60)
        finally:
            await source.close()
            await archives.close()

    asyncio.run(run())


class OrderedWorker(Worker):
    def __init__(self, spec):
        super().__init__(spec)
        self.prompt_id = None
        self.switch_end = None
        self.switch_tasks = []
        self.switched = []

    async def emit(self, message):
        if message["type"] == "session.ready":
            message["source_prompt_protocol"] = "ordered-boundary-v2"
            if getattr(self, "clip_switch", False):
                message.update(
                    source_condition_switch_protocol="clip-barrier-v1",
                    prompt_reference_updates=True,
                    max_references=21,
                    max_reference_bytes=64 << 20,
                )
            if getattr(self, "release_protocol", None):
                message["source_output_release_protocol"] = self.release_protocol
        await super().emit(message)

    async def finish_switch(self):
        candidate = self.switch_candidate
        end = self.switch_end
        accepted = sum(m.get("kind") == "source.append" for m in self.events)
        await asyncio.sleep(0.15)
        assert sum(m.get("kind") == "source.append" for m in self.events) == accepted
        self.prompt_id = candidate["event_id"]
        self.switched.append(candidate["payload"])
        self.switch_end = None
        await self.emit(
            dict(
                type="control_ack",
                kind="prompt.update",
                state="kv_ready",
                event_id=self.prompt_id,
                prompt_version=candidate["payload"]["version"],
                source_frame_index=end,
            )
        )

    async def send(self, wire):
        message = msgspec.msgpack.decode(wire)
        if message.get("kind") == "prompt.update":
            if getattr(self, "clip_switch", False):
                self.switch_candidate = message
                self.switch_end = max(
                    17, ((message["payload"]["min_source_frame"] + 16) // 17) * 17
                )
                await self.emit(
                    dict(
                        type="control_ack",
                        kind="prompt.update",
                        event_id=message["event_id"],
                        state="accepted",
                        source_frame_index=self.switch_end,
                    )
                )
                if (
                    sum(m.get("kind") == "source.append" for m in self.events)
                    == self.switch_end
                ):
                    self.switch_tasks.append(asyncio.create_task(self.finish_switch()))
            else:
                self.prompt_id = message["event_id"]
                await self.emit(
                    dict(
                        type="control_ack",
                        kind="prompt.update",
                        event_id=self.prompt_id,
                        state="accepted",
                    )
                )
        else:
            if message.get("kind") == "source.append" and self.switch_end is not None:
                assert message["payload"]["source_frame_start"] < self.switch_end
            await super().send(wire)
            if (
                message.get("kind") == "source.append"
                and self.switch_end is not None
                and message["payload"]["source_frame_start"] + 1 == self.switch_end
            ):
                self.switch_tasks.append(asyncio.create_task(self.finish_switch()))

    async def close(self):
        for task in self.switch_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.switch_tasks, return_exceptions=False)
        await super().close()


def test_initial_pairs_keep_startup_budget_when_first_output_arrives():
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )

    async def run():
        segment = object.__new__(RenderSegment)
        segment.segment = {}
        segment.config = SimpleNamespace(pairing_max_age_seconds=90)
        segment.first_output_at = None
        segment.pair_condition = asyncio.Condition()
        segment.recording = SimpleNamespace(accepting=True)
        segment.pairs = {0: (None, time.perf_counter() - 100, None)}
        guard = asyncio.create_task(segment._guard_pairs())
        await asyncio.sleep(0.01)
        assert not guard.done()
        segment.first_output_at = time.perf_counter()
        async with segment.pair_condition:
            segment.pair_condition.notify_all()
        await asyncio.sleep(0.01)
        assert not guard.done()
        assert segment.pairing_age_budget() == 90
        async with segment.pair_condition:
            segment.pairs[0] = (None, time.perf_counter() - 121, None)
            segment.pair_condition.notify_all()
        with pytest.raises(TimeoutError):
            await guard

    asyncio.run(run())


def test_pair_guard_cancellation_survives_a_simultaneous_pair_notification():
    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
        InteractiveSession,
    )

    async def run():
        waiting = asyncio.Event()

        class PairCondition(asyncio.Condition):
            async def wait(self):
                waiting.set()
                return await super().wait()

        session = object.__new__(InteractiveSession)
        session.config = SimpleNamespace(pairing_max_age_seconds=90)
        session.recording = SimpleNamespace(accepting=True)
        session.pairs = {}
        session.pair_condition = PairCondition()
        guard = asyncio.create_task(session._guard_pairs())
        try:
            await waiting.wait()
            async with session.pair_condition:
                session.pair_condition.notify_all()
                # Complete the condition wait immediately before shutdown cancels
                # its caller, as the final model pair is delivered at EOS.
                asyncio.get_running_loop().call_soon(guard.cancel)
            done, _ = await asyncio.wait({guard}, timeout=0.1)
            assert guard in done, "Pair guard must not retain a completed model lease"
            assert guard.cancelled()
        finally:
            guard.cancel()
            await asyncio.gather(guard, return_exceptions=True)

    asyncio.run(run())


def test_model_frame_cancellation_survives_archive_pair_notification():
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import EngineFrame
    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
        InteractiveSession,
    )

    async def run():
        waiting = asyncio.Event()

        class PairCondition(asyncio.Condition):
            async def wait(self):
                waiting.set()
                return await super().wait()

        session = object.__new__(InteractiveSession)
        session.config = SimpleNamespace(
            room_owned=True,
            pairing_max_age_seconds=90,
            immediate_capture=False,
            spec=VideoSpec(32, 32),
        )
        session.received = session.model_preview_count = 0
        session.video_transport = None
        session.outgoing = asyncio.Queue()
        session.recording = SimpleNamespace(accepting=True, pair=lambda *_: None)
        session.pair_condition = PairCondition()
        frame = EngineFrame(0, 0, b"", {}, provenance={"model_source_frame": 0})
        session.pairs = {0: (frame, 0, None)}
        session.ready = asyncio.Event()
        session.ready.set()
        session.render_queue = asyncio.Queue()
        session.render_queue.put_nowait((0, frame, b"rgb", 0, {}, [], {}))
        session.playback = SimpleNamespace(push=lambda *_: [(0, {"payload": b"au"})])
        session.trace_id = "cancel-pair"
        delivery = asyncio.create_task(session._render())
        try:
            async with asyncio.timeout(2):
                while not waiting.is_set():
                    if delivery.done():
                        await delivery
                    await asyncio.sleep(0)
            async with session.pair_condition:
                session.pairs[0] = (frame, 0, {})
                session.pair_condition.notify_all()
                asyncio.get_running_loop().call_soon(delivery.cancel)
            done, _ = await asyncio.wait({delivery}, timeout=0.1)
            assert delivery in done, "Model delivery must not retain a stopped segment"
            assert delivery.cancelled()
        finally:
            delivery.cancel()
            await asyncio.gather(delivery, return_exceptions=True)

    asyncio.run(run())


def test_source_current_and_finishing_recorders_have_three_bounded_slots(tmp_path):
    async def run():
        archives = AsyncArchiveManager(recorder_slots=3)
        records = []
        try:
            for index in range(4):
                record = await archives.create_recording(
                    directory=tmp_path / str(index),
                    spec=VideoSpec(32, 32),
                    world_id="world",
                    trace_id=str(index),
                    inference={"prompt": "test"},
                    ui=BrowserUI.metadata,
                    references=[],
                    object_prefix="test/" + str(index),
                    source_only=index == 0,
                )
                records.append(record)
            assert all(
                record.accepting and record.failure is None for record in records[:3]
            )
            assert records[3].failure == "RECORDER_CAPACITY_EXHAUSTED"
            assert not records[3].accepting
        finally:
            await archives.close()

    asyncio.run(run())


class StrictLeases:
    def __init__(self, spec, release_protocol=None):
        self.spec = spec
        self.release_protocol = release_protocol
        self.active = 0
        self.opens = []
        self.resource_state = "none"

    @asynccontextmanager
    async def open(self):
        assert self.active == 0
        self.active += 1
        self.resource_state = "reserved"
        worker = OrderedWorker(self.spec)
        worker.source_capacity = 51
        worker.clip_switch = self.release_protocol == "clip-barrier-v1"
        worker.release_protocol = (
            "valid-source-frames-v1" if worker.clip_switch else self.release_protocol
        )
        self.opens.append(worker)
        try:
            yield (
                worker,
                SimpleNamespace(
                    session_id="model",
                    generation_id="generation",
                    denoiser=SimpleNamespace(
                        worker_id="exclusive", model_revision="test"
                    ),
                ),
            )
        finally:
            await worker.close()
            self.active -= 1
            self.resource_state = "released"


@pytest.mark.parametrize("capture_clock", [False, True])
@pytest.mark.parametrize("sampling", ["latest", "adaptive"])
@pytest.mark.parametrize(
    "release_protocol", [None, "valid-source-frames-v1", "clip-barrier-v1"]
)
def test_source_survives_four_segments_and_ordered_prompt_updates(
    tmp_path, release_protocol, capture_clock, sampling
):
    async def run():
        spec = VideoSpec(32, 32)
        engine = Engine(spec)
        engine.stream_id = "view"
        leases = StrictLeases(spec, release_protocol)
        archives = AsyncArchiveManager(recorder_slots=3)
        journal = RenderJournal(
            MemoryRoomStore(),
            user_id="user",
            source_session_id="source",
            source_epoch="epoch",
            owner_epoch="room",
        )
        await journal.create()

        async def factory(**_):
            return engine

        host = SimpleNamespace(
            runtime=SimpleNamespace(players={}), make_segment_leases=lambda *_: leases
        )
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
                capture_clock=capture_clock,
                model_source_sampling=sampling,
                startup_seconds=5,
                drain_seconds=2,
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

        async def produce():
            while True:
                await engine.push()
                await asyncio.sleep(0.025)

        async def wait(predicate):
            async with asyncio.timeout(8):
                while not predicate():
                    await asyncio.sleep(0.01)

        counter = 0

        async def command(op, **fields):
            nonlocal counter
            counter += 1
            await asyncio.sleep(0.51)
            value = dict(
                type="command",
                op=op,
                request_id=f"r{counter}",
                source_session_id="source",
                source_epoch="epoch",
                **fields,
            )
            await source.send(msgspec.msgpack.encode(value))
            return value

        tasks = []
        try:
            await source.send(
                msgspec.msgpack.encode(
                    dict(type="init", whitebox={"world_id": "world"}, prompt="first")
                )
            )
            tasks = [asyncio.create_task(consume()), asyncio.create_task(produce())]
            await wait(lambda: source.ready.is_set())
            initial_mapping = []
            if release_protocol == "clip-barrier-v1":
                import io

                from PIL import Image

                image = io.BytesIO()
                Image.new("RGB", (8, 8), "blue").save(image, format="PNG")
                asset = await source.assets.upload("png", image.getvalue())
                initial_mapping = [
                    dict(id="<Picture 1>", asset_id=asset["asset_id"], role="scene")
                ]
            for index in range(4):
                state = await journal.read()
                await command(
                    "render.start",
                    expected_render_revision=state["render_revision"],
                    prompt=f"segment {index}",
                    seed=42,
                    references=initial_mapping if index == 0 else [],
                )
                await wait(
                    lambda: (
                        source.current_segment is not None
                        and source.current_segment.segment_id
                        == journal.snapshot["segment"]["segment_id"]
                        and not source.current_segment.closed_event.is_set()
                        and source.current_segment.first_decodable
                        # first_decodable precedes the subscriber pump. In the
                        # final case unbind intentionally suppresses late media;
                        # observe a delivered frame before ending that binding.
                        and any(
                            m.get("type") == "frame_batch"
                            and m.get("segment_epoch")
                            == source.current_segment.segment_epoch
                            for m in messages
                        )
                    )
                )
                segment = source.current_segment
                assert (segment.source_view.sampler is not None) == (
                    sampling == "adaptive"
                )
                if sampling == "adaptive":
                    assert (
                        segment.recording.manifest["source_sampling"]["mode"]
                        == "adaptive"
                    )
                first_media = next(
                    m
                    for m in messages
                    if m.get("type") == "frame_batch"
                    and m.get("segment_epoch") == segment.segment_epoch
                )
                blocked = first_media["source"]["model_timing"][
                    "input_capacity_blocked_ms"
                ]
                assert ("condition_switch" in blocked) == (
                    release_protocol == "clip-barrier-v1"
                )
                assert segment.recording.manifest["pairing"]["capacity"] == (
                    34 if release_protocol else 38
                )
                evidence = segment.recording.manifest["start_boundary_actual"]
                assert (
                    len(evidence["start_frame_sha256"]) == 64
                    and evidence["source_frame_index"] >= 0
                )
                assert (
                    segment.recording.manifest["model_identity"]["generation_id"]
                    == "generation"
                )
                start_index = engine.next
                if index == 0:
                    refs = []
                    if release_protocol == "clip-barrier-v1":
                        initial_descriptors = segment.recording.references
                        assert initial_descriptors[0]["path"] == "references/00.png"
                        assert (
                            segment.active_reference_descriptors == initial_descriptors
                        )
                        assert (
                            segment.recording.directory / initial_descriptors[0]["path"]
                        ).read_bytes() == image.getvalue()
                        archive_calls = []
                        original_add = segment.recording.add_references

                        async def add_references(references, *, wait=False):
                            archive_calls.append(references)
                            return await original_add(references, wait=wait)

                        segment.recording.add_references = add_references
                        refs = [
                            dict(
                                id=f"<Picture {i + 1}>",
                                asset_id=asset["asset_id"],
                                role="scene",
                            )
                            for i in range(21)
                        ]
                    versions = (
                        (2, 3, 4) if release_protocol == "clip-barrier-v1" else (2, 3)
                    )
                    for version in versions:
                        await command(
                            "prompt.replace",
                            segment_id=segment.segment_id,
                            segment_epoch=segment.segment_epoch,
                            expected_prompt_version=version - 1,
                            prompt_version=version,
                            prompt=f"prompt {version}",
                            **(
                                {
                                    "references": refs if version == 3 else [],
                                    "source_policy": "skip_during_switch",
                                }
                                if release_protocol == "clip-barrier-v1" and version > 2
                                else {}
                            ),
                            effective={
                                "source_frame_index": start_index,
                                "policy": "at_or_after",
                            },
                        )
                        if release_protocol == "clip-barrier-v1" and version == 2:
                            await wait(
                                lambda: (
                                    journal.snapshot["segment"]["applied_version"] == 2
                                )
                            )
                            assert archive_calls == []
                            assert (
                                segment.active_reference_descriptors
                                == initial_descriptors
                            )
                            assert "references" not in leases.opens[0].switched[0]
                    await wait(
                        lambda: (
                            journal.snapshot["segment"]["applied_version"]
                            == versions[-1]
                        )
                    )
                    if release_protocol == "clip-barrier-v1":
                        assert journal.snapshot["segment"]["references"] == []
                        assert [
                            len(c["references"]) for c in leases.opens[0].switched[1:]
                        ] == [21, 0]
                        assert [len(r) for r in archive_calls] == [21, 0]
                        assert segment.dropped_before_model > 0
                        evidence = segment.recording.manifest["condition_switches"]
                        assert len(evidence) == 3 and all(
                            row["kv_ready"] for row in evidence
                        )
                        assert all(row["skipped_captures"] > 0 for row in evidence)
                        assert all(
                            row["model_source_frame"] == row["boundary"]
                            for row in evidence
                        )
                        current = segment.prompt_by_event[leases.opens[0].prompt_id]
                        assert current["reference_descriptors"] == []
                state = await journal.read()
                if index == 3:
                    await source.unbind("BINDING_BUDGET_EXHAUSTED")
                else:
                    await command(
                        "render.end",
                        expected_render_revision=state["render_revision"],
                        segment_id=segment.segment_id,
                        segment_epoch=segment.segment_epoch,
                        mode="drain",
                    )
                await wait(lambda: segment.closed_event.is_set())
                assert segment.resource_state == "released"
                assert journal.snapshot["segment"]["state"] == "ended"
                assert not engine.closed and not source.finished.is_set()
                await wait(lambda: engine.next > start_index + 2)
            assert len(leases.opens) == 4 and leases.active == 0
            media = [m for m in messages if m.get("type") == "frame_batch"]
            assert len({m["segment_epoch"] for m in media}) == 4
            assert [w.init["prompt"] for w in leases.opens] == [
                f"segment {i}" for i in range(4)
            ]
            assert all(
                w.init["source_prompt_protocol"] == "ordered-boundary-v2"
                for w in leases.opens
            )
            await source.unbind("TEST")
            assert not engine.closed
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await source.close()
            await archives.close()

    asyncio.run(run())


@pytest.mark.parametrize("play_seconds", [None, 120, 540])
def test_room_v2_reconnect_keeps_source_and_fences_old_stream_ack(
    tmp_path, monkeypatch, play_seconds
):
    monkeypatch.delenv("WHITEBOX_RENDER_PLAY_SECONDS", raising=False)
    if play_seconds is not None:
        monkeypatch.setenv("WHITEBOX_RENDER_PLAY_SECONDS", str(play_seconds))
    expected_play_seconds = 300 if play_seconds is None else play_seconds
    from test_whitebox_room_pipeline import until

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import RenderHost
    from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        RoomDirectory,
        new_id,
    )

    async def run():
        spec = VideoSpec(32, 32)
        engine = Engine(spec)
        engine.renderer = BrowserUI()
        joins = []

        class Driver:
            max_players = 1

            async def join(self, **fields):
                joins.append(fields)
                return engine

            async def close(self):
                pass

        async def driver(**_):
            return Driver()

        archives = AsyncArchiveManager(recorder_slots=3)
        directory = RoomDirectory(MemoryRoomStore(), {"room": "http://room"})
        game = new_id()
        await directory.resolve(game, "world", create=True)
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
            drain_seconds=2,
        )
        host = RenderHost(
            runtime,
            config,
            archives,
            Reports(tmp_path, archives),
            None,
            None,
            grace_seconds=0.1,
        )
        host.make_segment_leases = lambda *_: StrictLeases(spec)
        owner = sub = None
        try:
            owner, sub = await host.attach(
                dict(
                    user_id="user",
                    trace_id="trace",
                    message=dict(
                        type="init",
                        prompt="first",
                        control_protocol="whitebox-render-v2",
                        source_operation="create",
                        whitebox={"game_id": game, "world_id": "world"},
                    ),
                )
            )
            attached = await until(sub, lambda m: m["type"] == "session.attached")
            await engine.push()
            ready = await until(sub, lambda m: m["type"] == "session.ready")
            assert (
                ready["play_seconds"] == expected_play_seconds
                and ready["control_protocol"] == "whitebox-render-v2"
            )
            assert (
                ready["video_stream"]["format"] == "h264"
                and ready["model_stream_id"] is None
            )
            sub.ack(
                dict(
                    protocol="whitebox-dual-v2",
                    stream="model",
                    media_stream_id=None,
                    frame=-1,
                    window=32,
                )
            )
            assert sub.windows["model"] == 32
            assert ready["capabilities"]["play_seconds"] == expected_play_seconds
            assert owner.session.config.play_seconds == expected_play_seconds
            monkeypatch.setenv("WHITEBOX_RENDER_PLAY_SECONDS", "1")
            source_id = owner.session.source_epoch
            sub.ack(
                dict(
                    protocol="whitebox-dual-v2",
                    stream="whitebox",
                    media_stream_id=source_id,
                    frame=-1,
                    window=4,
                )
            )
            await engine.push()
            await until(sub, lambda m: m["type"] == "whitebox.frame")
            sub.reset_model("new-model")
            sub.ack(
                dict(
                    protocol="whitebox-dual-v2",
                    stream="model",
                    media_stream_id="old-model",
                    frame=500,
                    window=32,
                )
            )
            assert sub.received["model"] == -1 and sub.windows["model"] is None
            sub.ack(
                dict(
                    protocol="whitebox-dual-v2",
                    stream="model",
                    media_stream_id="new-model",
                    frame=-1,
                    window=4,
                )
            )
            assert sub.windows["model"] == 4
            host.sessions[owner.session_id] = owner
            owner.detach(sub)
            await asyncio.sleep(0.01)
            assert not engine.closed
            owner.session.play_elapsed = expected_play_seconds - 1
            assert owner.ready_message()["play_remaining_seconds"] == 1
            with pytest.raises(ValueError, match="RESUME_DENIED"):
                await host.attach(
                    dict(
                        user_id="user",
                        trace_id="trace",
                        message={
                            **attached,
                            "type": "source.bind",
                            "resume_token": "invalid",
                            "whitebox": {"game_id": game, "world_id": "world"},
                            "control_sequence": owner.control_sequence,
                        },
                    )
                )
            assert not owner.session.binding_ended and owner.generation == 1
            _, resumed = await host.attach(
                dict(
                    user_id="user",
                    trace_id="trace",
                    message={
                        **attached,
                        "type": "source.bind",
                        "whitebox": {"game_id": game, "world_id": "world"},
                        "control_sequence": owner.control_sequence,
                    },
                )
            )
            assert len(joins) == 1 and resumed.generation == 2
            owner.session.binding_ended = True
            owner.session.play_elapsed = expected_play_seconds
            await owner.session.status_snapshot("budget-check")
            status = await until(
                resumed, lambda m: m.get("request_id") == "budget-check"
            )
            assert status["remaining_play_ms"] == 0
            with pytest.raises(RenderError) as expired:
                await host.attach(
                    dict(
                        user_id="user",
                        trace_id="trace",
                        message={
                            **attached,
                            "type": "source.bind",
                            "whitebox": {"game_id": game, "world_id": "world"},
                            "control_sequence": owner.control_sequence,
                        },
                    )
                )
            assert expired.value.code == "PLAYER_EXPIRED"
            assert runtime.players[owner.stream_id]["explicit_source_owner"] is True
            stored = await host.store.find_source(owner.session_id)
            assert (
                stored["source_epoch"] == source_id
                and stored["source_state"] == "ready"
            )
            owner.detach(resumed)
            await asyncio.wait_for(owner.member_closed.wait(), 2)
            assert engine.closed and owner.session_id not in host.sessions
            assert len(joins) == 1
        finally:
            await host.close()
            await runtime.close()
            await archives.close()

    asyncio.run(run())


def test_model_eos_follows_queued_media_and_is_scoped_to_attachment():
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderSubscription,
    )

    async def run():
        owner = SimpleNamespace(
            terminal=None,
            request_keyframe=lambda _: None,
            session=SimpleNamespace(source_epoch="source", current_model_stream="e1"),
        )
        sub = RenderSubscription(owner, 7)
        for stream, epoch in [("whitebox", "source"), ("model", "e1")]:
            sub.ack(
                dict(
                    protocol="whitebox-dual-v2",
                    stream=stream,
                    media_stream_id=epoch,
                    frame=-1,
                    window=32,
                )
            )
        for index in range(3):
            sub.offer_media(
                dict(
                    type="frame_batch",
                    stream="model",
                    media_stream_id="e1",
                    keyframe=index == 0,
                    output_frame_start=index + 50,
                )
            )
        ended = dict(
            type="render.status",
            segment=dict(segment_id="s1", segment_epoch="e1", state="ended"),
        )
        sub.offer_control(ended)
        assert msgspec.msgpack.decode(await sub.recv())["type"] == "render.status"
        for index in range(3):
            packet = msgspec.msgpack.decode(await sub.recv())
            assert (
                packet["type"] == "frame_batch"
                and packet["output_frame_start"] == index
            )
        sub.offer_media(
            dict(
                type="whitebox.frame",
                stream="whitebox",
                media_stream_id="source",
                keyframe=True,
                output_frame_start=10,
            )
        )
        eos = msgspec.msgpack.decode(await sub.recv())
        assert eos == dict(
            version=2,
            type="render.eos",
            source_epoch="source",
            media_stream_id="e1",
            segment_id="s1",
            connection_generation=7,
            frames=3,
        )
        sub.offer_control(ended)
        assert msgspec.msgpack.decode(await sub.recv())["type"] == "render.status"
        assert msgspec.msgpack.decode(await sub.recv())["stream"] == "whitebox"
        assert sub.model_eos is None
        sub.reset_model("e2")
        sub.offer_control(ended)
        assert sub.model_eos is None
        sub.offer_control(
            dict(
                type="source.status",
                segment=dict(segment_id="s2", segment_epoch="e2", state="ended"),
            )
        )
        assert sub.model_eos["media_stream_id"] == "e2"
        sub.reset_model("e3")
        assert sub.model_eos is None
        sub.stop("test complete")

    asyncio.run(run())


def test_reference_control_waits_for_actual_archive_write(tmp_path):
    import io

    from PIL import Image

    async def run():
        archives = AsyncArchiveManager(recorder_slots=1)
        try:
            record = await archives.create_recording(
                directory=tmp_path / "reference-confirmation",
                spec=VideoSpec(32, 32),
                world_id="world",
                trace_id="reference-confirmation",
                inference={},
                ui=BrowserUI.metadata,
                references=[],
                object_prefix="test/reference-confirmation",
            )
            data = io.BytesIO()
            Image.new("RGB", (8, 8), "green").save(data, format="PNG")
            refs = [dict(id="<Picture 1>", format="png", payload=data.getvalue())]
            descriptors = await record.add_references(refs, wait=True)
            assert (
                record.directory / descriptors[0]["path"]
            ).read_bytes() == data.getvalue()
            record.degrade("TEST_FAILURE")
            with pytest.raises(RuntimeError, match="unavailable"):
                await record.add_references(refs, wait=True)
        finally:
            await archives.close()

    asyncio.run(run())


def test_graceful_stop_wakes_source_waiting_at_condition_barrier():
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )

    async def run():
        segment = object.__new__(RenderSegment)
        stopped = asyncio.Event()
        segment.source_view = SimpleNamespace(stopped=stopped, stop=stopped.set)
        segment.model_admission_condition = asyncio.Condition()
        segment.closed_event = asyncio.Event()
        segment.selected_frames = 17
        segment.config = SimpleNamespace(drain_seconds=0.1)

        async def sender():
            async with segment.model_admission_condition:
                await segment.model_admission_condition.wait_for(stopped.is_set)
            segment.closed_event.set()

        task = asyncio.create_task(sender())
        await asyncio.sleep(0)
        await asyncio.wait_for(segment.stop(), 0.2)
        await task

    asyncio.run(run())


def test_two_sources_reject_third_and_release_independently(tmp_path):
    from test_whitebox_room_pipeline import until

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import RenderHost
    from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        RoomDirectory,
        new_id,
    )

    async def run():
        spec = VideoSpec(32, 32)
        engines = []

        class Driver:
            max_players = 1

            async def join(self, **fields):
                engine = Engine(spec)
                engine.renderer = BrowserUI()
                engines.append(engine)
                return engine

            async def close(self):
                pass

        async def driver(**_):
            return Driver()

        archives = AsyncArchiveManager(recorder_slots=4)
        directory = RoomDirectory(MemoryRoomStore(), {"room": "http://room"})
        games = [new_id() for _ in range(3)]
        for game in games:
            await directory.resolve(game, "world", create=True)
        runtime = RoomRuntime(
            directory=directory,
            owner_id="room",
            driver_factory=driver,
            spool=tmp_path,
            spec=spec,
            archives=archives,
            max_rooms=2,
        )
        config = WhiteboxConfig(
            engine_factory=None,
            ui_factory=None,
            spool=tmp_path,
            spec=spec,
            room_owned=True,
            drain_seconds=2,
        )
        host = RenderHost(
            runtime,
            config,
            archives,
            Reports(tmp_path, archives),
            None,
            None,
            max_sources=2,
        )
        host.make_segment_leases = lambda *_: StrictLeases(spec)

        def body(index):
            return dict(
                user_id=f"user{index}",
                trace_id=f"trace{index}",
                message=dict(
                    type="init",
                    prompt="first",
                    control_protocol="whitebox-render-v2",
                    source_operation="create",
                    whitebox={"game_id": games[index], "world_id": "world"},
                ),
            )

        try:
            first, sub1 = await host.attach(body(0))
            second, sub2 = await host.attach(body(1))
            assert first.session_id != second.session_id
            assert first.session.source_epoch != second.session.source_epoch
            assert len(host.sessions) == 2
            with pytest.raises(RenderError) as rejected:
                await host.attach(body(2))
            assert rejected.value.code == "CAPACITY_EXHAUSTED"
            async with asyncio.timeout(5):
                while len(engines) != 2:
                    await asyncio.sleep(0.01)
            await engines[0].push()
            await engines[1].push()
            await until(sub1, lambda m: m["type"] == "session.ready")
            await until(sub2, lambda m: m["type"] == "session.ready")
            await first.close()
            assert engines[0].closed
            assert not engines[1].closed and not second.closed
            assert list(host.sessions) == [second.session_id]
            sub2.ack(
                dict(
                    protocol="whitebox-dual-v2",
                    stream="whitebox",
                    media_stream_id=second.session.source_epoch,
                    frame=-1,
                    window=4,
                )
            )
            await engines[1].push()
            await until(sub2, lambda m: m["type"] == "whitebox.frame")
            replacement, _ = await host.attach(body(0))
            assert replacement.session_id != first.session_id
            assert len(host.sessions) == 2
        finally:
            await host.close()
            await runtime.close()
            await archives.close()

    asyncio.run(run())


@pytest.mark.parametrize("already_draining", [True, False])
def test_end_freezes_unsent_condition_dispatch(already_draining):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )

    async def scenario():
        current = dict(
            segment_id="segment",
            state="draining" if already_draining else "streaming",
            applied_version=1,
            prompt_queue=[
                dict(
                    request_id="queued",
                    version=2,
                    text="queued",
                    effective={"source_frame_index": 0},
                )
            ],
        )
        segment = object.__new__(RenderSegment)
        segment.segment_id = "segment"
        segment.owner = SimpleNamespace(
            journal=SimpleNamespace(snapshot={"segment": current})
        )

        async def latest(_):
            return 1, (SimpleNamespace(index=0), 0)

        segment.owner.latest = SimpleNamespace(get=latest)
        segment.closed_event = asyncio.Event()
        segment.source_admission_ready = asyncio.Event()
        segment.source_admission_ready.set()
        segment.prompt_available = asyncio.Event()
        segment.prompt_available.set()
        segment.pending_prompt = None
        segment.active_reference_descriptors = []
        segment.selected_frames = 1
        segment.cancel_requested = False
        segment.source_view = SimpleNamespace(stopped=asyncio.Event())
        segment.source_send_lock = asyncio.Lock()
        segment.model_wire_lock = asyncio.Lock()
        segment.model_wire_sequence = 0
        segment.prompt_by_event = {}
        segment.config = SimpleNamespace(startup_seconds=1)
        sent = []

        async def send(wire):
            sent.append(msgspec.msgpack.decode(wire))

        if not already_draining:
            await segment.source_send_lock.acquire()
        task = asyncio.create_task(
            segment._condition_controls(SimpleNamespace(send=send))
        )
        try:
            await asyncio.sleep(0.005)
            current["state"] = "draining"
            if segment.source_send_lock.locked():
                segment.source_send_lock.release()
            await asyncio.sleep(0.005)
            assert sent == []
            assert segment.pending_prompt is None
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("seconds", [1, 300, 1800, 7200])
def test_configured_regional_play_duration(seconds, monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        configured_render_play_seconds,
    )

    monkeypatch.setenv("WHITEBOX_RENDER_PLAY_SECONDS", str(seconds))
    assert configured_render_play_seconds() == seconds


@pytest.mark.parametrize("seconds", [0, -1, 7201, "bad"])
def test_configured_regional_play_duration_rejects_invalid(seconds, monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        configured_render_play_seconds,
    )

    monkeypatch.setenv("WHITEBOX_RENDER_PLAY_SECONDS", str(seconds))
    with pytest.raises(ValueError):
        configured_render_play_seconds()


def test_binding_lifetime_preserves_regional_play_budget():
    source = SourceSession.__new__(SourceSession)
    source.config = SimpleNamespace(play_seconds=300)
    assert source.binding_lifetime_seconds == 1800
    source.config.play_seconds = 7200
    assert source.binding_lifetime_seconds == 7800


@pytest.mark.parametrize(
    "now,played,waited,ready,expired",
    [
        (1801, 1800, 0, True, False),
        (7200, 7199, 0, True, True),
        (7200, 6600, 500, True, False),
        (7800, 100, 0, True, True),
        (500, 0, 599, False, True),
    ],
)
def test_regional_deadline_counts_play_wait_and_absolute_lifetime(
    now, played, waited, ready, expired, monkeypatch
):
    from unittest.mock import AsyncMock

    from sglang.multimodal_gen.runtime.realtime.whitebox import render_source

    async def run():
        source = SourceSession.__new__(SourceSession)
        source.config = SimpleNamespace(play_seconds=7200)
        source.host = SimpleNamespace(runtime=SimpleNamespace(players={}))
        source.engine = SimpleNamespace(stream_id="deadline-test")
        source.finished = asyncio.Event()
        source.model_binding = source.control_prefill = source.startup = None
        source.last_tick, source.bound_at = now - 1, 0
        source.binding_ended = False
        source.current_segment = None
        source.source_state_value = "ready" if ready else "preparing"
        source.display_mode = "whitebox"
        source.play_elapsed, source.wait_elapsed = played, waited
        source.unbind = AsyncMock()

        async def tick_once(_):
            source.finished.set()

        monkeypatch.setattr(render_source.asyncio, "sleep", tick_once)
        monkeypatch.setattr(render_source.time, "monotonic", lambda: now)
        if expired:
            with pytest.raises(RenderError) as raised:
                await source._deadline()
            assert raised.value.code == "BINDING_BUDGET_EXHAUSTED"
        else:
            await source._deadline()
        source.unbind.assert_not_awaited()  # End the player, not only its segment.

    asyncio.run(run())


def test_recorded_video_engine_24fps_and_capture_control_hold_resume(tmp_path):
    import av
    import numpy as np

    from sglang.multimodal_gen.runtime.realtime.whitebox.recorded_engine import (
        RecordedVideoEngine,
    )

    mp4_path = tmp_path / "sample.mp4"
    spec = VideoSpec(32, 32, 24)
    container = av.open(str(mp4_path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=spec.fps)
        stream.width = spec.width
        stream.height = spec.height
        stream.pix_fmt = "yuv420p"
        for idx in range(6):
            arr = np.full((spec.height, spec.width, 3), 20 * (idx + 1), dtype=np.uint8)
            vf = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(vf):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    async def run():
        engine = RecordedVideoEngine(
            spec,
            mp4_path=mp4_path,
            max_frames=4,
            realtime_pacing=False,
            fast_frame_delay_s=0.01,
            session_id="cli-rec-1",
            epoch=2,
        )
        it = engine.frames().__aiter__()
        f0 = await anext(it)
        assert f0.index == 0
        assert f0.pts_us == 0
        assert len(f0.rgb) == spec.frame_bytes
        header0 = f0.provenance["cli_frame_header"]
        assert header0 == {
            "sessionId": "cli-rec-1",
            "epoch": 2,
            "mediaGeneration": 1,
            "outputFrameId": 0,
        }

        held = await engine.capture_control("hold", "ctx-hold-0")
        assert held == {
            "frame": {
                "sessionId": "cli-rec-1",
                "epoch": 2,
                "mediaGeneration": 1,
                "outputFrameId": 0,
            }
        }

        next_task = asyncio.create_task(anext(it))
        await asyncio.sleep(0.03)
        assert not next_task.done(), "Frame emission must stay paused while held"

        resumed = await engine.capture_control("resume", "ctx-hold-0")
        assert resumed["pausedMs"] >= 20.0
        f1 = await asyncio.wait_for(next_task, timeout=2.0)
        assert f1.index == 1
        assert f1.pts_us == spec.pts(1)
        assert f1.provenance["capture_pts_us"] > spec.pts(1)
        assert f1.provenance["cli_frame_header"]["outputFrameId"] == 1

        remaining = [f async for f in it]
        assert [f.index for f in remaining] == [2, 3]
        retired = await engine.retire()
        assert retired["state"] == "retired"

        engine_48 = RecordedVideoEngine(
            spec,
            mp4_path=mp4_path,
            max_frames=3,
            capture_fps=48,
            realtime_pacing=False,
            fast_frame_delay_s=0.0,
        )
        assert engine_48.metadata == {"capture_fps": 48, "model_fps": 24}
        frames_48 = [f async for f in engine_48.frames()]
        assert [f.pts_us for f in frames_48] == [spec.pts(0), spec.pts(1), spec.pts(2)]
        assert [f.provenance["capture_pts_us"] for f in frames_48] == [
            0,
            1_000_000 // 48,
            2_000_000 // 48,
        ]
        await engine_48.close()

    asyncio.run(run())


def test_recorded_video_engine_with_source_session_prefill_and_switch(
    tmp_path, monkeypatch
):
    import importlib.util
    from pathlib import Path

    suite_path = (
        Path(__file__).resolve().parents[6]
        / "benchmark"
        / "whitebox"
        / "run_room_sglang_source_replay.py"
    )
    spec_mod = importlib.util.spec_from_file_location("wb_four_case_suite", suite_path)
    assert spec_mod is not None and spec_mod.loader is not None
    suite_mod = importlib.util.module_from_spec(spec_mod)
    import sys

    sys.modules["wb_four_case_suite"] = suite_mod
    spec_mod.loader.exec_module(suite_mod)

    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncRecording,
    )

    add_references = AsyncRecording.add_references

    async def slow_references(self, references, *, wait=False):
        await asyncio.sleep(0.1)  # Deliberately cross an old 8 ms capture interval.
        return await add_references(self, references, wait=wait)

    monkeypatch.setattr(AsyncRecording, "add_references", slow_references)

    enqueue = suite_mod.MockBoundWebSocket._enqueue

    def delayed_persistence(self, message):
        if message.get("type") == "prefill.persisted":
            message = dict(message)
            message.pop("prefill_to_persisted_ms", None)
            asyncio.get_running_loop().call_later(0.03, enqueue, self, message)
        else:
            enqueue(self, message)

    monkeypatch.setattr(suite_mod.MockBoundWebSocket, "_enqueue", delayed_persistence)

    case_meta = {
        "case_id": "unit_lifecycle_case",
        "world_id": "world-lifecycle-unit",
        "recording_id": "rec-lifecycle-unit",
        "object_prefix": "whitebox/rec-lifecycle-unit/",
        "seed": 42,
        "video": {"width": 832, "height": 480, "fps": 24, "frames": 36},
        "prompt": "Initial wuxia courtyard scene with <Video 1> and <Picture 1>.",
        "references_mapping": [
            {"id": "ref_0", "asset_id": "sha256:" + "a" * 64, "role": "scene"}
        ],
        "objects": {
            "references/00.png": {"bytes": 128, "sha256": "a" * 64},
        },
        "condition_switches": [
            {
                "prompt_version": 2,
                "boundary": 17,
                "source_policy": "skip_during_switch",
                "prompt": "Switched wuxia courtyard scene with snow and <Picture 1>.",
                "references": [
                    {"id": "ref_0", "asset_id": "sha256:" + "a" * 64, "role": "scene"}
                ],
            }
        ],
    }

    async def run():
        cache_store = suite_mod.MockSglangCacheStore()
        res = await suite_mod._run_single_segment_flow(
            case_id="unit_lifecycle_case",
            recording_meta=case_meta,
            media_root=tmp_path / "media",
            spool_dir=tmp_path / "spool",
            spec=VideoSpec(832, 480, 24),
            mode="mock-worker",
            cache_store=cache_store,
            sglang_url=None,
            max_frames_after_last_event=4,
            full_frames=False,
            preserve_switch_boundaries=False,
            flow_mode="deterministic",
            realtime_pacing=False,
            pacing_speed=10.0,
            repeat_warm_prefill=True,
            prewarm_switch_conditions=True,
        )
        assert res["prepare"]["f0_source_frame_index"] == 0
        assert res["prepare"]["primary_prefill"]["cache_result"] == "miss"
        assert res["prepare"]["primary_prefill"]["prefill_to_persisted_ms"] >= 30
        assert (
            res["prepare"]["primary_prefill"]["persistence_timing_source"]
            == "client_receipt"
        )
        assert res["prepare"]["warm_prefill"]["cache_result"] == "l1_hit"
        assert res["whitebox_session_ttff_ms"] > 0
        assert len(res["condition_switches"]) == 1
        sw = res["condition_switches"][0]
        assert sw["executed_boundary"] == 17
        assert sw["kv_ready"] is True
        assert sw["cache_receipt"]["cache_result"] == "l1_hit"
        assert res["steady_state_flow"]["received_model_frames"] >= 21
        control_actions = [
            c[0] for c in res["steady_state_flow"]["capture_control_log"]
        ]
        assert control_actions == ["hold", "resume"]

    asyncio.run(run())


def test_four_case_sglang_suite_mock_worker_end_to_end(tmp_path):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    suite_path = (
        Path(__file__).resolve().parents[6]
        / "benchmark"
        / "whitebox"
        / "run_room_sglang_source_replay.py"
    )
    spec_mod = importlib.util.spec_from_file_location(
        "wb_four_case_suite_e2e", suite_path
    )
    assert spec_mod is not None and spec_mod.loader is not None
    suite_mod = importlib.util.module_from_spec(spec_mod)
    sys.modules["wb_four_case_suite_e2e"] = suite_mod
    spec_mod.loader.exec_module(suite_mod)

    media_dir = tmp_path / "media"
    out_dir = tmp_path / "out"
    rc = suite_mod.main(
        [
            "run",
            "--mode",
            "mock-worker",
            "--media-dir",
            str(media_dir),
            "--output-dir",
            str(out_dir),
            "--max-frames-after-last-event",
            "4",
            "--synthesize-fixtures",
        ]
    )
    assert rc == 0
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "REPORT.md"
    assert summary_path.exists()
    assert report_path.exists()
    summary = json.loads(summary_path.read_text())
    assert len(summary["cases"]) == 4
    overview = summary["metrics_overview"]
    assert overview["block0_cache"]["cold_prefill_to_memory_ms"] > 0
    assert overview["block0_cache"]["warm_cache_result"] == "l1_hit"
    assert overview["block0_cache"]["marvel_12ref_stress_prefill_ms"] > 0
    assert overview["session_ttff"]["shanghai_drive_ttff_ms"] > 0
    assert overview["session_ttff"]["marvel_overdrive_ttff_ms"] > 0
    assert overview["session_ttff"]["dark_horns_ttff_ms"] > 0
    assert overview["session_ttff"]["wuxia_jiahao_seg1_ttff_ms"] > 0
    assert overview["session_ttff"]["wuxia_jiahao_seg2_ttff_ms"] > 0
    assert len(overview["prompt_switch_ttff"]["marvel_rapid_switches"]) == 2
    ab = overview["prompt_switch_ttff"]["dark_horns_ab_comparison"]
    assert ab["unprefilled_miss"]["cache_receipt"]["cache_result"] == "miss"
    assert ab["prefilled_hit"]["cache_receipt"]["cache_result"] == "l1_hit"

    # Reuse customized valid references without requesting synthesis.
    import hashlib

    from PIL import Image

    reference = next(p for p in media_dir.glob("*/references/*") if p.is_file())
    Image.new("RGB", (128, 128), color=(254, 1, 2)).save(reference)
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in media_dir.rglob("*")
        if p.is_file()
    }
    assert (
        suite_mod.main(
            [
                "run",
                "--mode",
                "mock-worker",
                "--media-dir",
                str(media_dir),
                "--output-dir",
                str(tmp_path / "reuse"),
                "--max-frames-after-last-event",
                "4",
            ]
        )
        == 0
    )
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    secondary = summary["cases"][-1]["secondary_segment"]
    assert secondary["condition_recording_id"] != secondary["source_recording_id"]
    assert "recording_id" not in secondary
    report = report_path.read_text()
    assert "同源换条件" in report
    assert f"视频 `{secondary['source_recording_id']}`" in report
    assert f"条件 `{secondary['condition_recording_id']}`" in report


def test_four_case_cache_evidence_and_direct_token_rejection(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[6]
        / "benchmark/whitebox/run_room_sglang_source_replay.py"
    )
    spec = importlib.util.spec_from_file_location("suite_evidence_test", path)
    suite = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = suite
    spec.loader.exec_module(suite)
    for receipt in (None, {}, {"cache_result": "miss"}):
        with pytest.raises(RuntimeError, match="CACHE_EXPECTATION_FAILED"):
            suite._require_cache_result(receipt, {"l1_hit", "l2_hit"}, "warm")
    with pytest.raises(ValueError, match="real reservation"):
        suite._create_real_sglang_lease(
            "ws://127.0.0.1:1", None, VideoSpec(), "test", {}
        )
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            suite._run_single_segment_flow(
                case_id="missing",
                recording_meta={
                    "recording_id": "missing",
                    "objects": {"whitebox.mp4": {"bytes": 1, "sha256": "bad"}},
                },
                media_root=tmp_path,
                spool_dir=tmp_path,
                spec=VideoSpec(),
                mode="real-sglang",
                cache_store=suite.MockSglangCacheStore(),
                sglang_url=None,
                max_frames_after_last_event=1,
                full_frames=False,
                preserve_switch_boundaries=False,
                realtime_pacing=True,
                pacing_speed=1.0,
            )
        )


def test_four_case_oss_alias_preserves_object_hash(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    import sys
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[6]
        / "benchmark/whitebox/run_room_sglang_source_replay.py"
    )
    spec = importlib.util.spec_from_file_location("suite_oss_test", path)
    suite = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = suite
    spec.loader.exec_module(suite)
    calls = []

    def download(bucket, key, dest, endpoint, profile):
        calls.append((bucket, key))
        dest.write_bytes(b"wrong" if bucket == "leap-world-shanghai" else b"verified")

    monkeypatch.setattr(suite, "_download_oss_object", download)
    suite.prepare_recording_media(
        dict(
            recording_id="record",
            object_prefix="archive/",
            objects={
                "whitebox.mp4": dict(
                    bytes=8, sha256=hashlib.sha256(b"verified").hexdigest()
                )
            },
        ),
        tmp_path,
        oss_bucket="leap-world-shanghai",
        oss_bucket_aliases=("leap-worldmodel-shanghai",),
        oss_endpoint="oss-cn-shanghai.aliyuncs.com",
    )
    assert calls == [
        ("leap-world-shanghai", "archive/whitebox.mp4"),
        ("leap-worldmodel-shanghai", "archive/whitebox.mp4"),
    ]
    assert (tmp_path / "record/whitebox.mp4").read_bytes() == b"verified"


def test_four_case_gate_advances_from_ack_without_observation_race(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from unittest.mock import AsyncMock

    path = (
        Path(__file__).resolve().parents[6]
        / "benchmark/whitebox/run_room_sglang_source_replay.py"
    )
    spec = importlib.util.spec_from_file_location("suite_gate_test", path)
    suite = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = suite
    spec.loader.exec_module(suite)
    monkeypatch.setattr(suite.SourceSession, "segment_status", AsyncMock())
    source = object.__new__(suite._ReplaySourceSession)
    source.replay_boundaries = [17, 34]
    segment = SimpleNamespace(
        condition_switch=None,
        model_source_allowed=lambda sent: True,
        model_control=AsyncMock(return_value=True),
        _notify_source_admission=AsyncMock(),
    )

    async def run():
        await source.segment_status(segment, "initializing", phase="protocol_ready")
        assert not segment.model_source_allowed(17)
        # A complete switch may happen before capacity_available runs again.
        ack = dict(
            type="control_ack", kind="prompt.update", state="kv_ready", prompt_version=2
        )
        await segment.model_control(ack)
        assert segment.model_source_allowed(17)
        assert not segment.model_source_allowed(34)
        await segment.model_control(ack)  # Duplicate cannot advance the next gate.
        assert not segment.model_source_allowed(34)

    asyncio.run(run())


def test_four_case_cli_requires_only_three_worker_addresses(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[6]
        / "benchmark/whitebox/run_room_sglang_source_replay.py"
    )
    spec = importlib.util.spec_from_file_location("suite_auto_cli_test", path)
    suite = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = suite
    spec.loader.exec_module(suite)
    captured = {}

    async def run(data, **kwargs):
        captured.update(kwargs)
        return {"cases": []}

    monkeypatch.setattr(suite, "run_source_replay_suite", run)
    assert (
        suite.main(
            [
                "run",
                "--encoder-url",
                "http://e:1",
                "--dit-url",
                "http://t:2",
                "--decoder-url",
                "http://d:3",
            ]
        )
        == 0
    )
    assert captured["mode"] == "real-sglang"
    assert captured["synthesize_if_missing"] is False
    assert all(
        captured[k] is None
        for k in (
            "profile",
            "compatibility",
            "routing_revision",
            "control_epoch",
            "network_domain",
        )
    )


@pytest.mark.parametrize(
    "full_frames,expected_primary,expected_secondary", [(False, 4, 4), (True, 19, 9)]
)
def test_source_replay_modes_and_full_secondary_targets(
    tmp_path, monkeypatch, full_frames, expected_primary, expected_secondary
):
    import importlib.util
    import sys
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[6]
        / "benchmark/whitebox/run_room_sglang_source_replay.py"
    )
    spec = importlib.util.spec_from_file_location("source_replay_length_test", path)
    suite = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = suite
    spec.loader.exec_module(suite)

    def no_test_gate(*args, **kwargs):
        pytest.fail("Natural mode must instantiate the real SourceSession")

    monkeypatch.setattr(suite, "_ReplaySourceSession", no_test_gate)
    meta = dict(
        recording_id="primary",
        world_id="world-test",
        prompt="test <Video 1> <Picture 1>",
        video={"frames": 19},
        objects={"references/00.png": {"sha256": "a" * 64}},
        references_mapping=[
            {"id": "ref-0", "asset_id": "sha256:" + "a" * 64, "role": "scene"}
        ],
        condition_switches=[],
    )
    if full_frames:
        meta["condition_switches"] = [
            dict(
                prompt_version=2,
                boundary=17,
                prompt=meta["prompt"],
                references=meta["references_mapping"],
            )
        ]
    secondary = {**meta, "recording_id": "secondary", "video": {"frames": 9}}

    import hashlib

    from sglang.multimodal_gen.runtime.realtime.whitebox.recorded_engine import (
        _iter_rgb24_via_pyav,
    )

    for rec_id, count, offset in (("primary", 19, 0), ("secondary", 9, 200)):
        suite._synthesize_mp4(
            tmp_path / rec_id / "whitebox.mp4", 32, 32, 24, count, offset
        )
    primary_hashes = {
        hashlib.sha256(rgb).hexdigest()
        for rgb in _iter_rgb24_via_pyav(tmp_path / "primary/whitebox.mp4", 32, 32)
    }
    secondary_hashes = {
        hashlib.sha256(rgb).hexdigest()
        for rgb in _iter_rgb24_via_pyav(tmp_path / "secondary/whitebox.mp4", 32, 32)
    }

    async def run():
        result = await suite._run_single_segment_flow(
            case_id="length-test",
            recording_meta=meta,
            media_root=tmp_path,
            spool_dir=tmp_path / "spool",
            spec=VideoSpec(32, 32),
            mode="mock-worker",
            cache_store=suite.MockSglangCacheStore(),
            sglang_url=None,
            max_frames_after_last_event=4,
            full_frames=full_frames,
            preserve_switch_boundaries=False,
            flow_mode="natural",
            realtime_pacing=False,
            pacing_speed=1.0,
            secondary_segment_recording=secondary,
        )
        assert result["flow_mode"] == "natural"
        assert result["target_output_frames"] == expected_primary
        assert result["secondary_segment"]["target_output_frames"] == expected_secondary
        assert result["steady_state_flow"]["received_model_frames"] >= expected_primary
        assert (
            result["secondary_segment"]["received_model_frames"] >= expected_secondary
        )
        assert result["secondary_segment"]["source_recording_id"] == "primary"
        assert result["secondary_segment"]["condition_recording_id"] == "secondary"
        assert (
            result["secondary_segment"]["input_kind"] == "same-source-condition-change"
        )
        f0 = result["secondary_segment"]["prepare"]["f0_source_frame_sha256"]
        assert f0 in primary_hashes and f0 not in secondary_hashes

    asyncio.run(run())


def test_replay_explicit_synthesis_only_fills_missing_files(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[6]
        / "benchmark/whitebox/run_room_sglang_source_replay.py"
    )
    spec = importlib.util.spec_from_file_location("replay_synthesis_test", path)
    suite = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = suite
    spec.loader.exec_module(suite)
    rec_dir = tmp_path / "rec"
    (rec_dir / "references").mkdir(parents=True)
    (rec_dir / "whitebox.mp4").write_bytes(b"existing video")
    (rec_dir / "references/00.png").write_bytes(b"existing reference")
    meta = dict(
        case_id="case",
        recording_id="rec",
        object_prefix="archive/",
        objects={"whitebox.mp4": {}, "references/00.png": {}, "references/01.png": {}},
    )
    receipt = suite.prepare_recording_media(
        meta,
        tmp_path,
        oss_bucket="unused",
        oss_endpoint="unused",
        synthesize_fixtures=True,
    )
    assert (rec_dir / "whitebox.mp4").read_bytes() == b"existing video"
    assert (rec_dir / "references/00.png").read_bytes() == b"existing reference"
    assert receipt["synthesized_files"] == ["references/01.png"]
    assert set(receipt["reused_files"]) == {"whitebox.mp4", "references/00.png"}
    (rec_dir / "references/01.png").unlink()
    with pytest.raises(FileNotFoundError, match="explicitly use --synthesize-fixtures"):
        asyncio.run(
            suite.run_source_replay_suite(
                {"cases": [meta]},
                media_root=tmp_path,
                output_dir=tmp_path / "output",
                mode="mock-worker",
            )
        )
