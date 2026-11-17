import asyncio
import hashlib
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

import msgspec
import pytest
from test_whitebox_render_contract import command, setup

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
    EngineFrame,
    VideoSpec,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.control_prefill import (
    ControlPrefill,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.model_geometry import ModelGeometry
from sglang.multimodal_gen.runtime.realtime.whitebox.player_idle import (
    PlayerInputActivity,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_assets import SourceFrames
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import RenderError


def test_pending_writeback_retries_only_verified_clean_prefill():
    async def run(cleaned, durable):
        attempts = []
        owner = NS(
            outgoing=asyncio.Queue(),
            upstream=NS(init={}),
            model_geometry=NS(model_init=lambda init: init, pad_rgb=lambda rgb: rgb),
            config=NS(startup_seconds=2),
            source_identity=lambda: dict(source_session_id="s", source_epoch="e"),
        )

        class Child:
            def __init__(self):
                self.first = not attempts
                self.failure = "MODEL_OPERATION_FAILED" if self.first else None
                self.resource_state = "released" if cleaned else "release_unknown"
                self.completed = asyncio.Event()
                self.completed.set()

            @asynccontextmanager
            async def open(self):
                yield self, None

            async def send(self, raw):
                attempts.append(msgspec.msgpack.decode(raw))

            async def recv(self):
                if self.first:
                    raise RenderError(self.failure)
                return msgspec.msgpack.encode(
                    dict(
                        type="prefill.memory_ready",
                        request_id="next",
                        cache_key="same-key",
                    )
                )

        owner.model_binding = NS(
            assignment=NS(denoiser=NS(worker_epoch="a", worker_id="worker-a")),
            borrow=lambda *a, **kw: Child(),
        )
        control = ControlPrefill(owner)
        control.expires = time.monotonic() + 270
        control.requests = {
            "previous": dict(
                context_id="ctx",
                ready_emitted=True,
                persistence_requested=durable,
                receipt=dict(type="prefill.memory_ready"),
            ),
            "next": dict(context_id="ctx", receipt={}),
        }
        frame = EngineFrame(1, 0, b"fixed-f0", {})
        control.context_id = "ctx"
        control.first = (1, (frame, 0))
        await control._compute("next", dict(prompt="A", seed=42), [], (1, (frame, 0)))
        messages = []
        while not owner.outgoing.empty():
            messages.append(msgspec.msgpack.decode(owner.outgoing.get_nowait()))
        retry = cleaned and durable
        assert len(attempts) == (2 if retry else 1)
        assert all(
            m["source_frame"] == b"fixed-f0" and m["request_id"] == "next"
            for m in attempts
        )
        assert messages[-1]["type"] == (
            "prefill.memory_ready" if retry else "prefill.failed"
        )
        assert sum(m["type"] == "prefill.retrying" for m in messages) == int(retry)

    asyncio.run(run(True, True))
    asyncio.run(run(False, True))
    asyncio.run(run(True, False))


@pytest.mark.parametrize("scene", [False, True])
@pytest.mark.parametrize("failure_after_memory_ready", [False, True])
def test_retries_only_verified_clean_prefill(scene, failure_after_memory_ready):
    async def run(cleaned, durable):
        attempts = []
        owner = NS(
            outgoing=asyncio.Queue(),
            upstream=NS(init={}),
            model_geometry=NS(model_init=lambda init: init, pad_rgb=lambda rgb: rgb),
            config=NS(startup_seconds=2),
            source_identity=lambda: dict(source_session_id="s", source_epoch="e"),
        )

        class Child:
            def __init__(self):
                self.first = not attempts
                self.failure = "MODEL_OPERATION_FAILED" if self.first else None
                self.resource_state = "released" if cleaned else "release_unknown"
                self.completed = asyncio.Event()
                self.completed.set()

            @asynccontextmanager
            async def open(self):
                yield self, None

            async def send(self, raw):
                attempts.append(msgspec.msgpack.decode(raw))

            async def recv(self):
                if self.first and not failure_after_memory_ready:
                    raise RenderError(self.failure)
                return msgspec.msgpack.encode(
                    dict(
                        type="prefill.memory_ready",
                        request_id="next",
                        cache_key="same-key",
                    )
                )

        owner.model_binding = NS(
            assignment=NS(denoiser=NS(worker_epoch="a", worker_id="worker-a")),
            borrow=lambda *a, **kw: Child(),
        )
        control = ControlPrefill(owner)
        control.expires = time.monotonic() + 270
        control.requests = {
            "previous": dict(
                context_id="ctx",
                ready_emitted=True,
                persistence_requested=durable,
                receipt=dict(type="prefill.memory_ready"),
            ),
            "next": dict(context_id="ctx", receipt={}),
        }
        frame = EngineFrame(1, 0, b"fixed-f0", {})
        control.context_id = "ctx"
        control.first = (1, (frame, 0))
        messages = []
        if scene:
            del control.requests["previous"]
            control.requests["next"].update(
                context_id=None, startup_observer=messages.append
            )
        snapshot = dict(rgb=b"sealed-s0", width=3, height=1) if scene else None
        await control._compute(
            "next",
            dict(prompt="A", seed=42),
            [],
            (1, (frame, 0)),
            snapshot=snapshot,
            deadline=time.monotonic() + 2 if scene else None,
        )
        while not owner.outgoing.empty():
            messages.append(msgspec.msgpack.decode(owner.outgoing.get_nowait()))
        retry = cleaned and (scene or durable)
        assert len(attempts) == (2 if retry else 1)
        assert all(
            m["source_frame"] == (b"sealed-s0" if scene else b"fixed-f0")
            and m["request_id"] == "next"
            for m in attempts
        )
        assert messages[-1]["type"] == (
            "prefill.memory_ready" if retry else "prefill.failed"
        )
        assert sum(m["type"] == "prefill.retrying" for m in messages) == int(retry)
        assert (control.failure is None) == cleaned

    asyncio.run(run(True, True))
    asyncio.run(run(False, True))
    asyncio.run(run(True, False))


def test_scene_prefill_sends_native_condition_pixels_and_waits_for_child_release():
    async def run(released):
        messages, receipts = [], []
        child = NS(completed=asyncio.Event(), resource_state="in_use")

        @asynccontextmanager
        async def open_child():
            yield socket, None

        async def send(wire):
            messages.append(msgspec.msgpack.decode(wire))

        async def recv():
            return msgspec.msgpack.encode(
                dict(
                    type="prefill.memory_ready",
                    request_id=messages[-1]["request_id"],
                    cache_key="target-cache",
                )
            )

        socket = NS(send=send, recv=recv)
        child.open = open_child
        parent = NS(assignment=NS(denoiser=NS(worker_epoch="a", worker_id="b")))

        def borrow(identity, *, kind):
            assert kind == "prefill"
            return child

        parent.borrow = borrow

        async def journal():
            return {}

        def no_live_frame(_):
            pytest.fail("Scene prefill must not take a live frame")

        owner = NS(
            binding_ended=False,
            journal=NS(read=journal),
            upstream=NS(init={}),
            model_binding=parent,
            model_geometry=NS(model_init=lambda init: init, pad_rgb=no_live_frame),
            source_identity=lambda: dict(source_session_id="s", source_epoch="e"),
        )
        prefill = ControlPrefill(owner)
        snapshot = dict(rgb=bytes([30, 60, 90]) * 6, width=3, height=2)
        references = [{"id": "<Picture 1>", "format": "png", "payload": b"frozen"}]

        async def observe(receipt):
            receipts.append(receipt)

        work = asyncio.create_task(
            prefill.prepare_scene(
                dict(prompt="Arrive at the forest", seed=42),
                references,
                snapshot,
                current=lambda: True,
                observe=observe,
                deadline=time.monotonic() + 1,
            )
        )
        async with asyncio.timeout(1):
            while not messages:
                await asyncio.sleep(0)
        assert messages[0]["persist"] is False
        assert messages[0]["source_frame"] == snapshot["rgb"]
        assert messages[0]["source_frame_size"] == {"width": 3, "height": 2}
        assert messages[0]["prompt"] == "Arrive at the forest"
        assert messages[0]["references"] == references
        assert not work.done()
        assert receipts[-1]["type"] == "prefill.computing"
        child.resource_state = "released" if released else "release_unknown"
        child.completed.set()
        if released:
            assert (await work)["cache_key"] == "target-cache"
        else:
            with pytest.raises(RenderError) as error:
                await work
            assert error.value.code == "RELEASE_UNKNOWN"
        assert owner.model_binding is parent

    asyncio.run(run(True))
    asyncio.run(run(False))


@pytest.mark.parametrize("mode", ["legacy", "member", "world"])
def test_true_f0_many_conditions_idempotency_and_busy_do_not_change_active_inputs(
    mode,
):
    shared = mode == "member"
    paused = mode == "legacy"

    async def run():
        calls = []
        tasks = []
        frame = EngineFrame(
            7,
            230000,
            bytes(12),
            {},
            provenance={
                "cli_frame_header": {
                    "sessionId": "cli",
                    "epoch": 1,
                    "mediaGeneration": 1,
                    "outputFrameId": 8,
                }
            },
        )
        owner = NS(
            journal=await setup(),
            binding_ended=False,
            config=NS(startup_seconds=2),
            retained=SourceFrames(),
            outgoing=asyncio.Queue(),
            upstream=NS(
                init={
                    "type": "init",
                    "prompt": "original",
                    "size": "2x2",
                    "source_stream": {
                        "width": 2,
                        "height": 2,
                        "fps": 24,
                        "format": "rgb24",
                    },
                }
            ),
        )
        if mode == "world":
            owner.upstream.init["world_id"] = "world-tested"
        owner.model_geometry = ModelGeometry(VideoSpec(2, 2))
        owner.retained.append(8, frame, 1.0)

        async def op(m):
            await owner.outgoing.put(msgspec.msgpack.encode(m))

        async def resolve(refs):
            return refs

        async def capture(action, id):
            calls.append(("clock", action, id))
            return {"frame": frame.provenance["cli_frame_header"], "pausedMs": 1200}

        owner.source_identity = lambda: dict(
            source_session_id="source", source_epoch="epoch"
        )
        owner.engine = NS(capture_control=capture, shared_world=shared)
        owner.engine.stream_id = "view"
        activity = PlayerInputActivity()
        activity.start(time.monotonic() - 50)
        owner.host = NS(runtime=NS(players={"view": {"input_activity": activity}}))
        owner.latest = NS(version=7, value=(frame, 1.0))
        owner.assets = NS(resolve=resolve)
        owner.operation_message = op
        owner.spawn_control = lambda coro: tasks.append(asyncio.create_task(coro))

        computing, release = asyncio.Event(), asyncio.Event()

        class Child:
            def __init__(self):
                self.completed = asyncio.Event()
                self.resource_state = "released"

            @asynccontextmanager
            async def open(self):
                yield self, NS()

            async def send(self, raw):
                self.value = msgspec.msgpack.decode(raw)
                calls.append(("prefill", self.value))
                self.completed.set()

            async def recv(self):
                if mode == "world" and self.value["request_id"] == "A":
                    computing.set()
                    await release.wait()
                return msgspec.msgpack.encode(
                    dict(
                        type="prefill.memory_ready",
                        request_id=self.value["request_id"],
                        cache_key=hashlib.sha256(
                            self.value["prompt"].encode()
                        ).hexdigest(),
                        hit=False,
                    )
                )

        owner.model_binding = NS(
            assignment=NS(denoiser=NS(worker_epoch="a", worker_id="worker-a")),
            borrow=lambda *a, **k: Child(),
        )
        control = ControlPrefill(owner)
        await control.prepare(
            command("render.prepare", "ctx", expected_render_revision=0)
        )
        if not paused:
            await asyncio.sleep(0)
            control.observe(8, frame, 1.0)
        await asyncio.gather(*tasks)
        assert control.resolve("ctx")[1][0] is frame
        assert (activity.waiting is not None) is paused
        before = activity.last_input

        def event(id, prompt):
            return dict(
                type="event",
                eventType="prefill",
                version=2,
                request_id=id,
                source_session_id="source",
                source_epoch="epoch",
                payload={"context_id": "ctx", "prompt": prompt, "references": []},
            )

        for name in ["A", "B", "C"]:
            await control.submit(event(name, name))
            assert (activity.waiting is not None) is paused
            if mode == "world" and name == "A":
                from dataclasses import replace

                await asyncio.wait_for(computing.wait(), 1)
                for index in range(8, 12):
                    header = dict(
                        frame.provenance["cli_frame_header"], outputFrameId=index + 1
                    )
                    newer = replace(
                        frame,
                        index=index,
                        pts_us=frame.pts_us + 41667 * (index - 7),
                        rgb=bytes([index]) * 12,
                        provenance={"cli_frame_header": header},
                    )
                    owner.latest.version += 1
                    owner.latest.value = newer, 2.0
                    control.observe(owner.latest.version, newer, 2.0)
                assert (
                    not control.active.done() and control.resolve("ctx")[1][0] is frame
                )
                release.set()
            await control.active
            assert activity.waiting is None
            assert (activity.last_input > before) is paused
            before = activity.last_input
        native = [x[1] for x in calls if x[0] == "prefill"]
        assert len(native) == 3
        assert all(
            x["source_frame"] == owner.model_geometry.pad_rgb(frame.rgb) for x in native
        )
        assert owner.upstream.init["prompt"] == "original"
        await control.submit(event("A", "A"))
        assert activity.last_input == before
        assert len([x for x in calls if x[0] == "prefill"]) == 3
        await control.submit(event("A", "changed"))
        last = msgspec.msgpack.decode(await owner.outgoing.get())
        messages = [last]
        while not owner.outgoing.empty():
            messages.append(msgspec.msgpack.decode(owner.outgoing.get_nowait()))
        assert messages[-1]["code"] == "REQUEST_ID_CONFLICT"
        assert {
            m["cache_key"] for m in messages if m.get("type") == "prefill.memory_ready"
        } == {hashlib.sha256(x.encode()).hexdigest() for x in "ABC"}
        await owner.journal.update(lambda s: s.update(segment={"state": "streaming"}))
        await control.submit(event("playing", "D"))
        assert (
            msgspec.msgpack.decode(await owner.outgoing.get())["code"] == "PREFILL_BUSY"
        )
        assert len([x for x in calls if x[0] == "prefill"]) == 3
        if mode == "world":
            boundary = {"mode": "next_available"}
            assert control.start_frame("ctx", boundary) is None
            assert control.start_frame("ctx", {"source_frame_index": 7})[1][0] is frame
            with pytest.raises(RenderError) as error:
                control.start_frame("ctx", {"source_frame_index": 99})
            assert error.value.code == "PREFILL_INPUT_MISMATCH"
            with pytest.raises(RenderError) as error:
                await control.consume_start("ctx", boundary, frame)
            assert error.value.code == "SOURCE_EPOCH_MISMATCH"
            for field in ("epoch", "mediaGeneration", "sessionId"):
                stale = replace(
                    newer, provenance={"cli_frame_header": {**header, field: "stale"}}
                )
                with pytest.raises(RenderError) as error:
                    await control.consume_start("ctx", boundary, stale)
                assert error.value.code == "SOURCE_EPOCH_MISMATCH"
            for proof in (
                {**newer.provenance, "view_id": "other-view"},
                {
                    "cli_frame_header": {
                        **header,
                        "source": {"presentationId": "next-scene", "sdkEpoch": 2},
                    }
                },
                {
                    "cli_frame_header": {
                        **header,
                        "outputFrameId": frame.provenance["cli_frame_header"][
                            "outputFrameId"
                        ],
                    }
                },
            ):
                with pytest.raises(RenderError) as error:
                    await control.consume_start(
                        "ctx", boundary, replace(newer, provenance=proof)
                    )
                assert error.value.code == "SOURCE_EPOCH_MISMATCH"
            captured = await control.consume_start("ctx", boundary, newer)
            assert captured is frame and captured.rgb != newer.rgb
            assert control.paused_ms == 0 and control.snapshot_age_ms > 0
            message = msgspec.msgpack.decode(await owner.outgoing.get())
            assert (
                message["type"] == "prefill.source_resumed"
                and message["paused_ms"] == 0
            )
        await control.close()
        if not paused:
            assert not any(call[0] == "clock" for call in calls)
        else:
            assert ("clock", "resume", "ctx") in calls

    asyncio.run(run())


def test_resume_resets_idle_only_after_new_source_frame(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.whitebox import control_prefill

    async def run():
        now = NS(value=200)
        monkeypatch.setattr(control_prefill, "time", NS(monotonic=lambda: now.value))
        activity = PlayerInputActivity()
        activity.start(100)

        async def capture(*_):
            return {"pausedMs": 90000}

        owner = NS(
            engine=NS(stream_id="view", capture_control=capture),
            host=NS(runtime=NS(players={"view": {"input_activity": activity}})),
            latest=NS(version=8),
            outgoing=asyncio.Queue(),
        )
        control = ControlPrefill(owner)
        control.context_id, control.resumed = "ctx", False
        control._pause_idle()
        await control.resume("ctx")
        assert activity.waiting == "ctx"
        control.observe(8, EngineFrame(7, 0, b"old", {}), 0)
        assert activity.waiting == "ctx"
        control.observe(9, EngineFrame(8, 0, b"new", {}), 0)
        assert activity.waiting is None and activity.last_input == 200
        now.value = 250
        control.observe(10, EngineFrame(9, 0, b"newer", {}), 0)
        assert activity.last_input == 200
        assert activity.is_expired(260, 60)

    asyncio.run(run())


def test_expired_context_is_retired_before_resume_and_does_not_release_parent():
    async def run():
        calls = []
        parent = NS()
        owner = NS(
            latest=NS(version=8),
            outgoing=asyncio.Queue(),
            model_binding=parent,
            spawn_control=asyncio.create_task,
        )
        control = ControlPrefill(owner)
        control.context_id, control.resumed = "ctx", False
        control.first = (8, (EngineFrame(7, 0, b"f0", {}), 0))
        control.expires = 1

        async def capture(action, request):
            assert control.first is None
            calls.append((action, request))
            return {"pausedMs": 270000}

        owner.engine = NS(capture_control=capture)
        control.check_deadline(2)
        await control.recovery
        control.check_deadline(3)
        assert calls == [("resume", "ctx")]
        assert owner.model_binding is parent
        with pytest.raises(RenderError) as error:
            control.resolve("ctx")
        assert error.value.code == "START_SOURCE_EXPIRED"

    asyncio.run(run())


def test_model_first_frame_presentation_preserves_real_capture_and_ui_time():
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
        capture_pts,
        presentation_pts,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )

    segment = RenderSegment.__new__(RenderSegment)
    segment.segment = {"context_id": "ctx"}
    segment.config = NS(spec=VideoSpec(32, 32))
    segment.prefill_hold_us = 5_000_000
    original = EngineFrame(
        7,
        0,
        bytes(3072),
        {"ui": "original"},
        provenance={
            "capture_pts_us": 100_000,
            "cli_frame_header": {"source": {"sourceTimeUs": 100_000}},
        },
    )
    shown = segment.model_presentation_frame(original, 0)
    fresh = EngineFrame(
        8, 41666, bytes(3072), {}, provenance={"capture_pts_us": 5_133_333}
    )
    assert capture_pts(shown) == capture_pts(original) == 100_000
    assert (
        shown.provenance["cli_frame_header"] == original.provenance["cli_frame_header"]
    )
    assert "model_presentation_pts_us" not in original.provenance
    assert 0 < presentation_pts(fresh) - presentation_pts(shown) < 100_000
    assert segment.model_presentation_frame(fresh, 1) is fresh


def test_persistence_failure_after_compute_keeps_memory_ready_for_replay():
    async def run():
        owner = NS(
            outgoing=asyncio.Queue(),
            recording=NS(manifest={}),
            source_identity=lambda: dict(source_session_id="s", source_epoch="e"),
            model_binding=NS(
                assignment=NS(denoiser=NS(worker_epoch="a", worker_id="worker-a"))
            ),
        )
        control = ControlPrefill(owner)
        control.requests["r"] = dict(context_id="ctx", receipt={})
        await control.receipt(
            dict(
                type="prefill.failed",
                request_id="r",
                cache_key="key",
                memory_ready=True,
                code="STORAGE_FAILED",
            )
        )
        assert owner.outgoing.empty()
        await control.receipt(
            dict(type="prefill.memory_ready", request_id="r", cache_key="key")
        )
        assert (
            msgspec.msgpack.decode(await owner.outgoing.get())["type"]
            == "prefill.memory_ready"
        )
        assert (
            msgspec.msgpack.decode(await owner.outgoing.get())["type"]
            == "prefill.failed"
        )
        assert control.requests["r"]["receipt"]["type"] == "prefill.memory_ready"
        assert owner.recording.manifest["prefill"]["r"]["persistence_state"] == "failed"

    asyncio.run(run())


@pytest.mark.parametrize("member", [False, True])
def test_event_type_only_prefill_uses_source_controls_and_preserves_member_owner(
    member,
):
    from unittest.mock import Mock

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderPipeline,
    )

    pipeline = RenderPipeline.__new__(RenderPipeline)
    sub = NS(closed=False, generation=1)
    pipeline.subscription, pipeline.generation = sub, 1
    pipeline.session = NS(startup=None)
    pipeline._enqueue_command = Mock()
    pipeline.member_source = NS(check=Mock()) if member else None
    message = dict(eventType="prefill", request_id="warm-1")
    wire = msgspec.msgpack.encode(message)
    if member:
        with pytest.raises(ValueError, match="SDK player owns"):
            pipeline.accept(sub, wire)
        pipeline._enqueue_command.assert_not_called()
    else:
        pipeline.accept(sub, wire)
        pipeline._enqueue_command.assert_called_once_with(sub, wire, message)


def test_member_prefill_admission_is_explicit_and_keeps_manual_init():
    import base64
    import copy

    from test_whitebox_player_media import media_launch

    from sglang.multimodal_gen.runtime.realtime.whitebox.player_media import (
        validate_media_launch,
    )

    value = media_launch()
    publication = value["publicationIdentity"]["publicationId"]
    # An ordinary launch retains its original parser; the new marker alone
    # cannot turn eager launch bytes into a different model owner.
    validate_media_launch(value, publication)
    value["memberPrefill"] = {"protocol": "worldkit-member-prefill-v1"}
    with pytest.raises(ValueError):
        validate_media_launch(value, publication)
    init = msgspec.msgpack.decode(base64.b64decode(value["initMsgpackBase64"]))
    init.update(source_operation="create.manual", inference_control="manual")
    value["initMsgpackBase64"] = base64.b64encode(msgspec.msgpack.encode(init)).decode()
    assert validate_media_launch(value, publication) == init
    for marker in (
        {"protocol": "unknown"},
        {"protocol": "worldkit-member-prefill-v1", "prompt": "override"},
        None,
    ):
        invalid = copy.deepcopy(value)
        invalid["memberPrefill"] = marker
        with pytest.raises(ValueError):
            validate_media_launch(invalid, publication)


def member_prefill_owner():
    from sglang.multimodal_gen.runtime.realtime.whitebox.member_prefill import (
        MemberPrefill,
    )

    references = [
        dict(id=f"<Picture {i}>", asset_id="sha256:" + str(i) * 64, role="scene")
        for i in (1, 2)
    ]
    owner = NS(
        config=NS(engine_factory=NS(check=lambda: None)),
        inference=NS(
            assets=NS(
                prompt="Frozen Dune\n", seed=42, committed=NS(is_set=lambda: True)
            )
        ),
        initial_assets=references,
        engine=NS(shared_world=True),
        model_binding=NS(assignment=NS(denoiser=NS(worker_epoch="worker-a"))),
    )
    owner.member_prefill = MemberPrefill(owner)
    return owner


def test_member_prefill_condition_order_and_exact_ready_f0_are_required():
    import copy

    from sglang.multimodal_gen.runtime.realtime.whitebox.member_prefill import (
        condition_digest,
    )

    owner = member_prefill_owner()
    policy = owner.member_prefill
    frozen = policy.conditions()
    policy.validate_conditions(frozen)
    for malformed in (None, [], 1):
        with pytest.raises(RenderError):
            policy.validate_conditions(malformed)
    for change in (
        dict(prompt="Frozen Dune"),
        dict(seed=43),
        dict(references=list(reversed(frozen["references"]))),
        dict(references=[]),
    ):
        with pytest.raises(RenderError):
            policy.validate_conditions({**frozen, **change})
    detached = policy.descriptor()
    detached["references"][0]["id"] = "changed"
    assert owner.initial_assets[0]["id"] == "<Picture 1>"
    frame = EngineFrame(17, 0, b"actual-f0", {})
    owner.control_prefill = NS(
        resolve=lambda context: (
            (3, (frame, 1))
            if context == "context-a"
            else (_ for _ in ()).throw(RenderError("START_SOURCE_EXPIRED"))
        ),
        requests={},
    )
    command = {
        **copy.deepcopy(frozen),
        "context_id": "context-a",
        "start_boundary": {"source_frame_index": 17},
    }
    with pytest.raises(RenderError):
        policy.validate_start(command)
    request = dict(
        context_id="context-a",
        condition_sha256=condition_digest(frozen),
        ready_emitted=True,
        receipt=dict(worker_epoch="worker-a"),
    )
    owner.control_prefill.requests["prefill-a"] = request
    policy.validate_start(command)
    for change in (
        dict(context_id="other"),
        dict(start_boundary={"mode": "next_available"}),
        dict(start_boundary={"source_frame_index": 18}),
    ):
        with pytest.raises(RenderError):
            policy.validate_start({**command, **change})
    request["receipt"]["worker_epoch"] = "worker-b"
    with pytest.raises(RenderError):
        policy.validate_start(command)


def test_opted_in_member_accepts_only_frozen_prefill_and_keeps_sdk_input_owner():
    from unittest.mock import Mock

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderPipeline,
    )

    pipeline = RenderPipeline.__new__(RenderPipeline)
    sub = NS(closed=False, generation=1)
    pipeline.subscription, pipeline.generation = sub, 1
    owner = member_prefill_owner()
    pipeline.session = NS(startup=None, member_prefill=owner.member_prefill)
    pipeline.member_source = NS(check=Mock())
    pipeline._enqueue_command = Mock()
    good = dict(
        type="event",
        eventType="prefill",
        payload={**owner.member_prefill.conditions(), "context_id": "context-a"},
    )
    pipeline.accept(sub, msgspec.msgpack.encode(good))
    assert pipeline._enqueue_command.call_count == 1

    for message in (
        {**good, "payload": {**good["payload"], "seed": 7}},
        dict(type="asset.upload"),
        dict(kind="input"),
        dict(kind="finish"),
        dict(type="command", op="source.close"),
        dict(type="command", op="prompt.replace"),
        dict(type="command", op="model.inference.configure"),
    ):
        with pytest.raises((ValueError, RenderError)):
            pipeline.accept(sub, msgspec.msgpack.encode(message))
    assert pipeline._enqueue_command.call_count == 1


def test_member_prefill_normal_end_does_not_use_manual_failure_lifecycle():
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    async def run():
        journal = await setup()
        await journal.initialize_inference(manual=False)
        started = await journal.start(
            command(
                "render.start",
                "begin",
                expected_render_revision=0,
                prompt="Frozen",
                references=[],
            )
        )
        ended = await journal.end(
            command(
                "render.end",
                "end",
                expected_render_revision=started["render_revision"],
                segment_id=started["segment_id"],
                segment_epoch=started["segment_epoch"],
            )
        )
        await journal.update(lambda state: state["segment"].update(state="completed"))
        segment = NS(segment_id=started["segment_id"])
        source = NS(journal=journal, current_segment=segment, member_prefill=object())
        inference = SourceInference.__new__(SourceInference)
        inference.source, inference.manual = source, True
        await inference.segment_ended(segment, None)
        state = await journal.read()
        assert state["segment"]["state"] == "completed"
        assert state["inference"]["state"] != "failed"
        assert state["inference"]["enabled"] is True
        assert state["render_revision"] == ended["render_revision"]

    asyncio.run(run())
