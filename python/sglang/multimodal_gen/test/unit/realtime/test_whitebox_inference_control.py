"""Manual compute admission uses the source journal, never a browser prompt."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_whitebox_render_contract import command, setup

from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import RenderError


async def ordinary_query_source():
    journal = await setup()
    await journal.initialize_inference(manual=True)
    await journal.update(
        lambda state: state["inference"].update(enabled=True, state="enabled")
    )
    start = await journal.start(
        command(
            "render.start",
            "original-start",
            expected_render_revision=0,
            prompt="Original scene.",
            references=[dict(id="scene", asset_id="sha256:" + "a" * 64)],
        )
    )
    await journal.update(
        lambda state: state["segment"].update(state="streaming", applied_version=1)
    )
    scope = {key: start[key] for key in ("segment_id", "segment_epoch")}
    return journal, scope


def test_scene_fence_waits_for_in_flight_source_send():
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    async def run():
        lock = asyncio.Lock()
        stopped = asyncio.Event()
        segment = SimpleNamespace(
            source_send_lock=lock,
            source_view=SimpleNamespace(stop=stopped.set),
            publishing=True,
        )
        channel = SimpleNamespace(suspended=False)
        source = SimpleNamespace(
            current_segment=segment,
            current_model_stream="old",
            display_mode="model",
            journal=SimpleNamespace(snapshot={}),
            context_channel=channel,
        )
        inference = SourceInference.__new__(SourceInference)
        inference.source, inference.prewarm = source, None

        async with lock:
            fence = asyncio.create_task(inference.fence_scene())
            await asyncio.wait_for(stopped.wait(), 1)
            assert not fence.done()
            assert not segment.publishing and channel.suspended
            assert source.current_model_stream is None
        await asyncio.wait_for(fence, 1)

    asyncio.run(run())


@pytest.mark.parametrize(
    "scope", ["retiring", "retiring-error", "new-segment", "new-revision"]
)
def test_scene_fence_terminal_receipt_preserves_only_its_original_intent(scope):
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    async def run():
        journal, _ = await ordinary_query_source()
        stopped = asyncio.Event()
        segment = SimpleNamespace(
            source_send_lock=asyncio.Lock(),
            source_view=SimpleNamespace(stop=stopped.set),
            publishing=True,
        )
        source = SimpleNamespace(
            journal=journal,
            current_segment=segment,
            current_model_stream="original",
            display_mode="model",
            context_channel=None,
            status_snapshot=AsyncMock(),
        )
        owner = SourceInference(
            source,
            dict(
                source_operation="create.manual",
                inference_control="manual",
                prompt="Original scene.",
            ),
        )
        await owner.fence_scene()
        assert stopped.is_set()
        # The actual segment can finish between fence and target preparation.
        if scope == "new-segment":
            segment = source.current_segment = SimpleNamespace(
                source_view=SimpleNamespace(stop=lambda: None), publishing=True
            )
        elif scope == "new-revision":
            await journal.update(lambda state: state["inference"].update(revision=1))
        await owner.segment_ended(
            segment, "MODEL_UNAVAILABLE" if scope == "retiring-error" else None
        )
        inference = journal.snapshot["inference"]
        if scope == "retiring":
            assert inference["state"] == "enabled"
            assert "code" not in inference
            source.status_snapshot.assert_not_awaited()
        else:
            assert inference["state"] == "failed"
            assert inference["code"] == "MODEL_UNAVAILABLE"

    asyncio.run(run())


def test_ordinary_query_composes_and_clears_through_original_prompt_queue():
    async def run():
        journal, scope = await ordinary_query_source()
        read = command("model.context.intervention.get", "read", **scope)
        assert await journal.player_instruction(read) == dict(
            status="current", revision=0, prompt=None
        )
        write = command(
            "model.context.intervention",
            "query",
            **scope,
            expectedRevision=0,
            prompt="Warm sunlight.",
        )
        assert await journal.player_instruction(write) == dict(
            status="accepted", revision=1, prompt="Warm sunlight."
        )
        state = await journal.read()
        assert (
            state["segment"]["prompt_queue"][0]["text"]
            == "Original scene.\n\nWarm sunlight."
        )
        assert state["segment"]["references"] == [
            dict(id="scene", asset_id="sha256:" + "a" * 64)
        ]
        assert state["segment"]["accepted_version"] == 2
        assert await journal.player_instruction(write) == dict(
            status="duplicate", revision=1, prompt="Warm sunlight."
        )
        assert len((await journal.read())["segment"]["prompt_queue"]) == 1
        conflict = {**write, "request_id": "stale", "prompt": "Rain."}
        assert await journal.player_instruction(conflict) == dict(
            status="conflict",
            revision=1,
            prompt="Warm sunlight.",
            code="REVISION_CONFLICT",
        )
        with pytest.raises(RenderError, match="request ID"):
            await journal.player_instruction({**write, "prompt": "Changed request."})
        # Model application updates segment.prompt. Clear must still restore the
        # immutable original, rather than append to the last generated prompt.
        await journal.update(
            lambda state: state["segment"].update(
                prompt="Original scene.\n\nWarm sunlight.",
                applied_version=2,
                prompt_queue=[],
            )
        )
        cleared = await journal.player_instruction(
            command(
                "model.context.intervention",
                "clear",
                **scope,
                expectedRevision=1,
                prompt=None,
            )
        )
        assert cleared == dict(status="accepted", revision=2, prompt=None)
        assert (await journal.read())["segment"]["prompt_queue"][0][
            "text"
        ] == "Original scene."

    asyncio.run(run())


def test_ordinary_query_fences_compute_segment_and_combined_budget():
    async def run():
        journal, scope = await ordinary_query_source()
        write = command(
            "model.context.intervention",
            "query",
            **scope,
            expectedRevision=0,
            prompt="x" * 16384,
        )
        with pytest.raises(RenderError) as error:
            await journal.player_instruction(write)
        assert error.value.code == "CONDITION_INVALID"
        for change, code in [
            ({"source_epoch": "old"}, "SOURCE_EPOCH_MISMATCH"),
            ({"segment_epoch": "old"}, "SEGMENT_EPOCH_MISMATCH"),
            ({"expectedRevision": True}, "INVALID_COMMAND"),
        ]:
            with pytest.raises(RenderError) as error:
                await journal.player_instruction({**write, "prompt": "Rain.", **change})
            assert error.value.code == code
        await journal.update(lambda state: state["inference"].update(state="starting"))
        with pytest.raises(RenderError) as error:
            await journal.player_instruction({**write, "prompt": "Rain."})
        assert error.value.code == "MODEL_UNAVAILABLE"
        assert (await journal.read())["segment"]["accepted_version"] == 1

    asyncio.run(run())


def test_ordinary_query_actual_source_relays_results_before_waking_model_owner():
    import msgspec

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def run():
        journal, scope = await ordinary_query_source()
        source = object.__new__(SourceSession)
        source.journal = journal
        source.session_id, source.source_epoch = "source", "epoch"
        source.closed = source.binding_ended = False
        source.world_driven = False
        source.context_channel = None
        source.inference = SimpleNamespace(manual=True)
        from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
            LocalRoomEngine,
        )

        entry = {"engine": SimpleNamespace()}
        source.engine = LocalRoomEngine(
            SimpleNamespace(player=lambda *a, **kw: entry),
            {"stream_id": "ordinary-view", "engine": {}, "ui": {}},
        )
        source.current_segment = SimpleNamespace(prompt_available=asyncio.Event())
        source.outgoing = asyncio.Queue()
        write = command(
            "model.context.intervention",
            "query",
            **scope,
            expectedRevision=0,
            prompt="Warm sunlight.",
        )
        original_read = journal.read

        async def yielding_read():
            # Let model work run while the control owner waits for persistence.
            await asyncio.sleep(0)
            return await original_read()

        journal.read = yielding_read

        async def first_model_message():
            await source.current_segment.prompt_available.wait()
            assert not source.outgoing.empty(), "Model woke before the query receipt"
            return msgspec.msgpack.decode(source.outgoing.get_nowait())

        _, result = await asyncio.gather(
            source.context_intervention(write, await journal.read()),
            first_model_message(),
        )
        assert result["type"] == "model.context.intervention.result"
        assert result["status"] == "accepted" and result["revision"] == 1
        assert result["segment_epoch"] == scope["segment_epoch"]
        assert source.current_segment.prompt_available.is_set()

    asyncio.run(run())


def test_inference_journal_cas_duplicate_and_scope():
    async def run():
        journal = await setup()
        await journal.initialize_inference(manual=True)
        assert (await journal.read())["inference"] == dict(
            enabled=False, revision=0, state="disabled"
        )
        request = command(
            "model.inference.configure",
            "enable",
            enabled=True,
            expected_inference_revision=0,
        )
        accepted = await journal.configure_inference(request)
        assert accepted["inference"] == dict(enabled=True, revision=1, state="starting")
        assert journal.snapshot["display_mode"] == "model"
        assert journal.snapshot["display_revision"] == 1
        assert (await journal.configure_inference(request))["duplicate"]
        await journal.display(
            command(
                "display.configure",
                "debug-raw",
                mode="whitebox",
                expected_display_revision=1,
            )
        )
        assert (await journal.configure_inference(request))["duplicate"]
        assert journal.snapshot["display_mode"] == "whitebox"
        stable = await journal.configure_inference(
            {**request, "request_id": "same-intent", "expected_inference_revision": 1}
        )
        assert stable["state"] == "completed"
        assert journal.snapshot["display_mode"] == "whitebox"
        for changed, code in [
            ({"enabled": False}, "REQUEST_ID_CONFLICT"),
            ({"request_id": "stale"}, "REVISION_CONFLICT"),
            ({"request_id": "other", "source_epoch": "old"}, "SOURCE_EPOCH_MISMATCH"),
            ({"request_id": "prompt", "prompt": "browser"}, "INVALID_COMMAND"),
        ]:
            with pytest.raises(RenderError) as error:
                await journal.configure_inference({**request, **changed})
            assert error.value.code == code
        assert (await journal.read())["inference"]["revision"] == 1

    asyncio.run(run())


@pytest.mark.parametrize("bound", [False, True])
def test_manual_start_keeps_preview_without_reserving_model_parent(monkeypatch, bound):
    from sglang.multimodal_gen.runtime.realtime.whitebox import render_source, startup
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec

    async def run():
        journal = await setup()
        source = object.__new__(render_source.SourceSession)
        source.journal = journal
        source.user_id, source.session_id = "user", "source"
        source.config = SimpleNamespace(
            spec=VideoSpec(), engine_factory=None, startup_seconds=2
        )
        source.configure_resolution = lambda _: None
        parents = []

        class Parent:
            def __init__(self, lease, callback, **_):
                self.lease = lease

            def start(self):
                parents.append(self)

        from sglang.multimodal_gen.runtime.realtime.whitebox import model_binding

        monkeypatch.setattr(model_binding, "ModelBinding", Parent)
        source.host = SimpleNamespace(make_segment_leases=lambda *_: SimpleNamespace())
        monkeypatch.setenv("WHITEBOX_BOUND_COMPUTE", "1" if bound else "0")
        monkeypatch.setenv("WHITEBOX_MODEL_STARTUP_PREPARE", "1")
        monkeypatch.setattr(render_source.InteractiveSession, "_start", AsyncMock())
        monkeypatch.setattr(
            startup,
            "validate_startup",
            lambda *_: SimpleNamespace(
                prompt="trusted", seed=7, references=[], committed=asyncio.Event()
            ),
        )
        monkeypatch.setattr(
            startup,
            "PlayStartup",
            lambda *_: pytest.fail("manual init constructed PlayStartup"),
        )
        await source._start(
            dict(
                source_operation="create.manual",
                inference_control="manual",
                prompt="trusted",
                startup={"initial_render": {}},
                whitebox={"world_id": "world"},
            )
        )
        assert source.startup is None
        assert parents == []
        assert source.model_binding is None
        assert (await journal.read())["inference"]["enabled"] is False

    asyncio.run(run())


def test_manual_disable_keeps_managed_player_binding_and_prefill_available():
    from sglang.multimodal_gen.runtime.realtime.whitebox.control_prefill import (
        ControlPrefill,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    async def run():
        journal = await setup()
        binding = SimpleNamespace(
            assignment=SimpleNamespace(player_expires_at=200),
            lease=SimpleNamespace(resource_state="reserved"),
            closed=False,
            close=AsyncMock(
                side_effect=AssertionError("disabled inference released parent")
            ),
        )
        child = SimpleNamespace(
            source_view=SimpleNamespace(stop=lambda: None),
            pending_prompt=None,
            resource_state="released",
            stop=AsyncMock(),
            publishing=True,
        )
        source = SimpleNamespace(
            journal=journal,
            model_binding=binding,
            current_segment=child,
            context_channel=None,
            segment_lock=asyncio.Lock(),
            startup=None,
            display_mode="model",
            current_model_stream="child",
            revision=0,
        )
        source.control_prefill = ControlPrefill(source)
        control = SourceInference(source, {})
        old_prefill = source.control_prefill
        await control.disable()
        assert source.model_binding is binding
        assert binding.lease.resource_state == "reserved"
        assert old_prefill.closed and not source.control_prefill.closed
        assert binding.on_receipt == source.control_prefill.receipt
        child.stop.assert_awaited_once()
        binding.close.assert_not_awaited()
        assert source.display_mode == "whitebox"

    asyncio.run(run())


def test_repeated_enable_during_start_does_not_supersede_owner():
    async def run():
        journal = await setup()
        await journal.initialize_inference(manual=True)
        first = await journal.configure_inference(
            command(
                "model.inference.configure",
                "one",
                enabled=True,
                expected_inference_revision=0,
            )
        )
        second = await journal.configure_inference(
            command(
                "model.inference.configure",
                "two",
                enabled=True,
                expected_inference_revision=1,
            )
        )
        assert second["state"] == "completed"
        assert second["inference"] == first["inference"]
        assert journal.snapshot["operations"]["one"]["state"] == "accepted"
        assert (
            "superseded_by_inference_revision"
            not in journal.snapshot["operations"]["one"]
        )
        await journal.settle_inference("one", 1, state="enabled")
        assert (await journal.read())["inference"]["state"] == "enabled"

    asyncio.run(run())


def test_manual_source_repeated_enable_disable_keeps_world_and_uses_fresh_source(
    tmp_path, monkeypatch
):
    import msgspec
    from test_whitebox_render_lifecycle import (
        AsyncArchiveManager,
        BrowserUI,
        Engine,
        SourceSession,
        StrictLeases,
        VideoSpec,
        WhiteboxConfig,
    )

    async def run():
        monkeypatch.delenv("WHITEBOX_BOUND_COMPUTE", raising=False)
        spec = VideoSpec(32, 32)
        engine = Engine(spec)
        engine.stream_id = "view"
        leases = StrictLeases(spec, "clip-barrier-v1")
        archives = AsyncArchiveManager(recorder_slots=3)
        journal = await setup()

        async def factory(**_):
            return engine

        source = SourceSession(
            SimpleNamespace(
                runtime=SimpleNamespace(players={}),
                make_segment_leases=lambda *_: leases,
            ),
            journal,
            source_epoch="epoch",
            user_id="u",
            session_id="source",
            trace_id="source",
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

        tasks = []
        try:
            await source.send(
                msgspec.msgpack.encode(
                    dict(
                        type="init",
                        source_operation="create.manual",
                        inference_control="manual",
                        whitebox={"world_id": "world"},
                        prompt="frozen trusted",
                        seed=19,
                    )
                )
            )
            tasks = [asyncio.create_task(consume()), asyncio.create_task(produce())]
            await wait(
                lambda: (
                    source.ready.is_set()
                    and source.retained.frames
                    and any(m.get("type") == "session.ready" for m in messages)
                )
            )
            assert not leases.opens and leases.active == 0
            ready = next(m for m in messages if m.get("type") == "session.ready")
            assert ready["inference"] == dict(
                enabled=False, revision=0, state="disabled"
            )
            assert ready["capabilities"]["model_inference_control"] is True
            indices, identities = [], []
            for index in range(2):
                await source.command(
                    command(
                        "model.inference.configure",
                        f"enable{index}",
                        enabled=True,
                        expected_inference_revision=index * 2,
                    )
                )
                await wait(lambda: journal.snapshot["inference"]["state"] == "enabled")
                # Opt-in selects the original display owner in the same CAS;
                # no extra rate-limited display command is needed for pixels.
                assert (
                    source.display_mode == journal.snapshot["display_mode"] == "model"
                )
                assert any(
                    m.get("type") == "source.status"
                    and m.get("display_mode") == "model"
                    and m.get("inference", {}).get("revision") == index * 2 + 1
                    for m in messages
                )
                segment = source.current_segment
                await wait(lambda: segment.first_decodable)
                indices.append(
                    segment.recording.manifest["start_boundary_actual"][
                        "source_frame_index"
                    ]
                )
                identities.append(segment.segment_id)
                before = engine.next
                await source.command(
                    command(
                        "model.inference.configure",
                        f"disable{index}",
                        enabled=False,
                        expected_inference_revision=index * 2 + 1,
                    )
                )
                await wait(lambda: journal.snapshot["inference"]["state"] == "disabled")
                assert (
                    segment.closed_event.is_set()
                    and segment.resource_state == "released"
                    and leases.active == 0
                )
                assert (
                    not engine.closed
                    and not source.finished.is_set()
                    and not source.binding_ended
                )
                await wait(lambda: engine.next > before + 2)
                source.control_tokens = 4
            assert indices[1] > indices[0] and identities[0] != identities[1]
            assert [worker.init["prompt"] for worker in leases.opens] == [
                "frozen trusted"
            ] * 2
            assert [worker.init["seed"] for worker in leases.opens] == [19] * 2
            await source.command(
                command(
                    "model.inference.configure",
                    "enable-eos",
                    enabled=True,
                    expected_inference_revision=4,
                )
            )
            await wait(lambda: journal.snapshot["inference"]["state"] == "enabled")
            # A late replay of an old disable must not fence this new segment.
            await source.command(
                command(
                    "model.inference.configure",
                    "disable0",
                    enabled=False,
                    expected_inference_revision=1,
                )
            )
            assert source.display_mode == "model"
            assert source.current_model_stream is not None
            display_revision = journal.snapshot["display_revision"]
            await source.command(
                command(
                    "display.configure",
                    "display-before-eos",
                    mode="model",
                    expected_display_revision=display_revision,
                )
            )
            await source.current_segment.stop(cancel=True)
            await wait(lambda: journal.snapshot["inference"]["state"] == "failed")
            assert journal.snapshot["display_mode"] == "whitebox"
            assert (
                source.revision
                == journal.snapshot["display_revision"]
                == display_revision + 2
            )
            assert journal.snapshot["inference"]["enabled"] is True
            assert source.display_mode == "whitebox" and not source.finished.is_set()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await source.close()
            await archives.close()

    asyncio.run(run())


def test_manual_source_controls_cannot_bypass_inference_with_browser_prompt():
    import msgspec

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def run():
        journal = await setup()
        await journal.initialize_inference(manual=True)
        source = object.__new__(SourceSession)
        source.journal = journal
        source.inference = SimpleNamespace(manual=True)
        source.outgoing = asyncio.Queue()
        source.context_channel = None
        source.assets = SimpleNamespace(
            resolve=lambda _: pytest.fail("browser reached assets")
        )
        source.control_tokens, source.control_token_time = 4, 0
        source.binding_ended = False
        for op in [
            "render.start",
            "render.replace",
            "prompt.replace",
            "render.context.start",
            "render.prepare",
        ]:
            await source.command(command(op, op, prompt="browser", seed=1))
            rejected = msgspec.msgpack.decode(await source.outgoing.get())
            assert (
                rejected["type"] == "operation.rejected"
                and rejected["code"] == "INVALID_COMMAND"
            )
        assert not (await journal.read())["operations"]

    asyncio.run(run())


def test_member_manual_open_fences_existing_cli_channel_before_frames():
    from sglang.multimodal_gen.runtime.realtime.whitebox.player_media import (
        MemberSource,
    )

    async def run():
        channel = SimpleNamespace(suspended=False)
        entry = dict(
            read_lock=asyncio.Lock(),
            spec="spec",
            game_id="game",
            join_receipt=dict(stream_id="stream", engine={}, ui={}),
        )
        source = object.__new__(MemberSource)
        source.runtime = SimpleNamespace(
            rooms={"game": dict(route=dict(world_id="world"))}
        )
        source.engine = SimpleNamespace(model_context=channel)
        source.record = dict(mediaLaunch=dict(publicationIdentity={}))
        source.entry = entry
        source.check = lambda: entry
        await source.open_session(
            world_id="world",
            spec="spec",
            trace_id="trace",
            context=dict(start_suspended=True),
        )
        assert channel.suspended

    asyncio.run(run())


@pytest.mark.parametrize("cleanup_ok", [True, False])
@pytest.mark.parametrize("preview_fails", [False, True])
def test_disable_waits_for_actual_bound_parent_and_child_cleanup(
    cleanup_ok, preview_fails
):
    from contextlib import asynccontextmanager

    from test_whitebox_model_binding import Lease

    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.model_binding import (
        ModelBinding,
    )

    async def run():
        cleanup = asyncio.Event()

        class ParentLease(Lease):
            resource_state = "none"

            @asynccontextmanager
            async def open(self):
                self.resource_state = "reserved"
                try:
                    async with super().open() as opened:
                        yield opened
                finally:
                    self.resource_state = "releasing"
                    await cleanup.wait()
                    self.resource_state = (
                        "released" if cleanup_ok else "release_unknown"
                    )

        lease = ParentLease()
        binding = ModelBinding(lease, AsyncMock(), timeout=2)
        await binding.wait_ready()
        child = binding.borrow(
            dict(model_session_id="child", generation_id="generation")
        )
        context = child.open()
        await context.__aenter__()
        stopped = asyncio.Event()
        segment = SimpleNamespace(
            publishing=True,
            resource_state="reserved",
            source_view=SimpleNamespace(stop=stopped.set),
            pending_prompt=None,
        )

        async def stop(**_):
            stopped.set()
            await context.__aexit__(None, None, None)
            segment.resource_state = child.resource_state

        segment.stop = stop
        journal = await setup()
        await journal.initialize_inference(manual=False)
        source = SimpleNamespace(
            journal=journal,
            current_segment=segment,
            current_model_stream="model",
            context_channel=(
                SimpleNamespace(
                    suspended=False,
                    pause=AsyncMock(side_effect=ConnectionError("preview unavailable")),
                )
                if preview_fails
                else None
            ),
            model_binding=binding,
            control_prefill=None,
            startup=None,
            segment_lock=asyncio.Lock(),
            display_mode="model",
            operation_message=AsyncMock(),
            status_snapshot=AsyncMock(),
        )
        owner = SourceInference(
            source,
            dict(
                source_operation="create.manual",
                inference_control="manual",
                prompt="frozen",
            ),
        )
        operation = await journal.configure_inference(
            command(
                "model.inference.configure",
                "off",
                enabled=False,
                expected_inference_revision=0,
            )
        )
        worker = asyncio.create_task(owner.run(operation))
        try:
            async with asyncio.timeout(2):
                while not lease.releases:
                    await asyncio.sleep(0)
            assert stopped.is_set() and child.resource_state == "released"
            assert journal.snapshot["inference"]["state"] == "stopping"
            assert not worker.done() and lease.resource_state == "releasing"
        finally:
            cleanup.set()
            if cleanup_ok:
                await binding.close()
            else:
                with pytest.raises(RenderError):
                    await binding.close()
            await worker
        assert journal.snapshot["inference"]["state"] == (
            "disabled" if cleanup_ok else "failed"
        )
        assert source.model_binding is (None if cleanup_ok else binding)
        if not cleanup_ok:
            assert journal.snapshot["inference"]["code"] == "RELEASE_UNKNOWN"

    asyncio.run(run())


def test_harness_manual_bind_opens_preview_without_submitting_conditions(tmp_path):
    import json

    from test_whitebox_source_conditions import setup as context_setup

    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )

    async def run():
        adapter, condition, _, _, submits, _ = await context_setup(tmp_path)
        sent = []

        class Socket:
            async def send(self, value):
                sent.append(json.loads(value))

        view = SimpleNamespace(
            descriptor=dict(
                epoch=1,
                modelContext=dict(
                    protocol=condition["protocol"],
                    required=True,
                    configHash=condition["configHash"],
                    gameBuildHash=condition["source"]["gameBuildHash"],
                    compilerVersion=1,
                    visible=False,
                ),
            ),
            closed=False,
            stopped=False,
            control=Socket(),
            transport_generation=0,
        )
        view.recover_model_context_transport = AsyncMock()
        channel = CLIConditionChannel(view)
        try:
            await channel.bind(
                segment_id=condition["segmentId"],
                create_adapter=lambda *_: adapter,
                start_suspended=True,
            )
            assert channel.suspended and channel.bound.is_set()
            assert sent == [dict(type="model.context.preview", epoch=1)] and not submits
            await channel.switch_segment(condition["segmentId"])
            assert not channel.suspended
            view.recover_model_context_transport.assert_awaited_once_with(0)
        finally:
            await channel.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "source_operation,mode", [("create", "manual"), ("create.manual", None)]
)
def test_mismatched_manual_init_is_rejected_before_source_allocation(
    tmp_path, monkeypatch, source_operation, mode
):
    from test_whitebox_player_control import Pipeline, host

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import PROTOCOL

    async def run():
        owner, _ = host(tmp_path, monkeypatch)
        try:
            init = dict(
                type="init",
                control_protocol=PROTOCOL,
                source_operation=source_operation,
                whitebox=dict(game_id="1" * 32, world_id="world"),
                prompt="trusted",
            )
            if mode is not None:
                init["inference_control"] = mode
            with pytest.raises(RenderError):
                await owner.attach(dict(user_id="u", trace_id="trace", message=init))
            assert not Pipeline.created and not owner.sessions
        finally:
            await owner.close()

    asyncio.run(run())


@pytest.mark.parametrize("manual", [True, False])
def test_member_launch_strictly_pairs_manual_discriminator(manual):
    import base64

    import msgspec
    from test_whitebox_player_media import media_launch

    from sglang.multimodal_gen.runtime.realtime.whitebox.player_media import (
        validate_media_launch,
    )

    launch = media_launch()
    init = msgspec.msgpack.decode(base64.b64decode(launch["initMsgpackBase64"]))
    init["source_operation"] = "create.manual"
    if manual:
        init["inference_control"] = "manual"
    launch["initMsgpackBase64"] = base64.b64encode(
        msgspec.msgpack.encode(init)
    ).decode()
    if manual:
        assert (
            validate_media_launch(
                launch, launch["publicationIdentity"]["publicationId"]
            )["inference_control"]
            == "manual"
        )
        init["source_operation"] = "create"
        launch["initMsgpackBase64"] = base64.b64encode(
            msgspec.msgpack.encode(init)
        ).decode()
    with pytest.raises(ValueError):
        validate_media_launch(launch, launch["publicationIdentity"]["publicationId"])


def test_manual_members_default_off_and_toggle_only_their_original_source(
    tmp_path, monkeypatch
):
    import base64

    import msgspec
    from test_whitebox_player_media import media_launch, model_owners
    from test_whitebox_rtc_pipeline import Peer

    async def run():
        monkeypatch.delenv("WHITEBOX_BOUND_COMPUTE", raising=False)
        launch = media_launch()
        init = msgspec.msgpack.decode(base64.b64decode(launch["initMsgpackBase64"]))
        init.update(source_operation="create.manual", inference_control="manual")
        launch["initMsgpackBase64"] = base64.b64encode(
            msgspec.msgpack.encode(init)
        ).decode()
        env = await model_owners(tmp_path, monkeypatch, launch=launch, max_sources=2)
        peers = []

        async def wait(predicate):
            async with asyncio.timeout(10):
                while not predicate():
                    for peer in peers:
                        for index, (_, done) in enumerate(peer.calls):
                            if not done.done():
                                peer.finish(index)
                    await asyncio.sleep(0.01)

        try:
            first, _ = await env.admit("alice")
            second, _ = await env.admit("bob")
            for record in (first, second):
                receipt = await env.runtime.player_transport.get_player_media(record)
                peer = Peer()
                peers.append(peer)
                await env.host.attach(
                    dict(
                        user_id=record["caller"]["subject"],
                        trace_id="attach",
                        message=receipt["attachment"]["message"],
                    ),
                    transport="webrtc-rtp",
                    rtc_peer=peer,
                )
            a, b = first["media_pipeline"], second["media_pipeline"]
            await wait(lambda: all(p.session.ready.is_set() for p in (a, b)))
            assert not env.workers
            assert all(
                p.session.journal.snapshot["inference"]["state"] == "disabled"
                for p in (a, b)
            )
            before = [view.next for view in env.views.values()]
            await wait(
                lambda: all(
                    view.next > index for view, index in zip(env.views.values(), before)
                )
            )
            stale = dict(
                type="command",
                op="model.inference.configure",
                request_id="late-enable",
                source_session_id=a.session_id,
                source_epoch=a.session.source_epoch,
                enabled=True,
                expected_inference_revision=0,
            )
            raw = msgspec.msgpack.encode(stale)
            a.command_bytes += len(raw)
            a.command_queue.put_nowait((a.generation - 1, raw, stale))
            a.accept(
                a.subscription,
                msgspec.msgpack.encode(
                    dict(
                        type="command",
                        op="operation.get",
                        request_id="absent-query",
                        source_session_id=a.session_id,
                        source_epoch=a.session.source_epoch,
                        operation_request_id="late-enable",
                    )
                ),
            )
            await wait(
                lambda: any(
                    m.get("request_id") == "absent-query" for _, _, m in a.history
                )
            )
            absent = next(
                m for _, _, m in a.history if m.get("request_id") == "absent-query"
            )
            assert (
                absent["operation_request_id"] == "late-enable"
                and absent["operation"] is None
            )
            assert absent["connection_generation"] == a.generation and not env.workers
            request = dict(
                type="command",
                op="model.inference.configure",
                request_id="enable-a",
                source_session_id=a.session_id,
                source_epoch=a.session.source_epoch,
                enabled=True,
                expected_inference_revision=0,
            )
            a.accept(a.subscription, msgspec.msgpack.encode(request))
            a.accept(
                a.subscription,
                msgspec.msgpack.encode(
                    dict(
                        type="command",
                        op="operation.get",
                        request_id="accepted-query",
                        source_session_id=a.session_id,
                        source_epoch=a.session.source_epoch,
                        operation_request_id="enable-a",
                    )
                ),
            )
            await wait(
                lambda: any(
                    m.get("request_id") == "accepted-query" for _, _, m in a.history
                )
            )
            accepted = next(
                m for _, _, m in a.history if m.get("request_id") == "accepted-query"
            )
            assert (
                accepted["operation_request_id"] == "enable-a"
                and accepted["operation"]["request_id"] == "enable-a"
            )
            await wait(
                lambda: a.session.journal.snapshot["inference"]["state"] == "enabled"
            )
            assert len(env.workers) == 1 and b.session.current_segment is None
            scope = a.session.model_context_segment(a.session.journal.snapshot)
            for request_id, op, fields in [
                ("member-query-read", "model.context.intervention.get", {}),
                (
                    "member-query-set",
                    "model.context.intervention",
                    {"expectedRevision": 0, "prompt": "Golden sunlight."},
                ),
                (
                    "member-query-clear",
                    "model.context.intervention",
                    {"expectedRevision": 1, "prompt": None},
                ),
            ]:
                a.accept(
                    a.subscription,
                    msgspec.msgpack.encode(
                        dict(
                            type="command",
                            op=op,
                            request_id=request_id,
                            source_session_id=a.session_id,
                            source_epoch=a.session.source_epoch,
                            **scope,
                            **fields,
                        )
                    ),
                )
                await wait(
                    lambda: any(
                        m.get("request_id") == request_id for _, _, m in a.history
                    )
                )
                reply = next(
                    m for _, _, m in a.history if m.get("request_id") == request_id
                )
                assert reply["type"] == "model.context.intervention.result"
                assert reply["status"] == ("current" if not fields else "accepted")
            assert a.session.journal.snapshot["segment"]["player_instruction"] == dict(
                revision=2, prompt=None
            )
            assert b.session.current_segment is None and len(env.workers) == 1
            request.update(
                request_id="disable-a", enabled=False, expected_inference_revision=1
            )
            a.accept(a.subscription, msgspec.msgpack.encode(request))
            await wait(
                lambda: a.session.journal.snapshot["inference"]["state"] == "disabled"
            )
            assert a.session.current_segment.resource_state == "released"
            assert not any(view.closed for view in env.views.values())
            assert b.session.journal.snapshot["inference"] == dict(
                enabled=False, revision=0, state="disabled"
            )
            await env.runtime.event(
                second["stream_id"],
                "input",
                dict(sequence=0, input=dict(held_keys=["KeyD"])),
            )
            assert env.views[second["player_id"]].inputs == [dict(held_keys=["KeyD"])]
            assert not env.views[first["player_id"]].inputs
            # Member allowlist must retain original operation recovery reads.
            a.accept(
                a.subscription,
                msgspec.msgpack.encode(
                    dict(
                        type="command",
                        op="operation.get",
                        request_id="query",
                        source_session_id=a.session_id,
                        source_epoch=a.session.source_epoch,
                        operation_request_id="disable-a",
                    )
                ),
            )
            await wait(
                lambda: any(
                    m.get("type") == "source.status" and m.get("request_id") == "query"
                    for _, _, m in a.history
                )
            )
        finally:
            await env.close()

    asyncio.run(run())


def test_harness_reenable_waits_for_new_owner_and_preserves_unknown_barrier(tmp_path):
    from test_whitebox_source_conditions import setup as context_setup

    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    async def run():
        adapter, condition, journal, receipts, _, _ = await context_setup(tmp_path)
        await journal.initialize_inference(manual=False)
        first = await journal.model_condition(condition, [])
        initial_ids = (await journal.read())["model_context_segment_ids"]
        await journal.prompt_applied(
            condition["segmentId"], 1, dict(first_output_frame=0, num_frames=1)
        )
        await journal.update(
            lambda s: s["segment"].update(state="ended", resource_state="released")
        )
        await journal.update(
            lambda s: s.update(
                inference=dict(enabled=True, revision=1, state="starting")
            )
        )
        original = SimpleNamespace(
            segment_id=first["segment_id"],
            closed_event=asyncio.Event(),
            failure_code="MODEL_UNAVAILABLE",
        )
        original.closed_event.set()
        owner = SimpleNamespace(
            journal=journal,
            model_binding=None,
            closed=False,
            binding_ended=False,
            current_segment=original,
            config=SimpleNamespace(startup_seconds=1),
            start_model_binding=lambda _: None,
            operation_message=AsyncMock(),
            session_id="source",
            source_epoch="epoch",
        )
        tasks = []

        async def switch(segment_id):
            async def ready():
                await asyncio.sleep(0)
                segment = SimpleNamespace(
                    segment_id=segment_id,
                    closed_event=asyncio.Event(),
                    model_ready=asyncio.Event(),
                )
                segment.model_ready.set()
                owner.current_segment = segment

            tasks.append(asyncio.create_task(ready()))

        owner.context_channel = SimpleNamespace(
            switch_segment=switch, probe_subscription_pause=AsyncMock()
        )

        async def handoff(operation, state):
            await owner.context_channel.switch_segment(operation["segment_id"])

        owner.handoff_context_segment = handoff
        inference = SourceInference(
            owner,
            dict(
                source_operation="create.manual",
                inference_control="manual",
                prompt="frozen",
            ),
        )
        operation = dict(
            operation_id="original-inference",
            request_id="enable",
            inference=dict(revision=1, enabled=True),
        )
        try:
            await inference.enable(operation)
            assert owner.current_segment is not original
            state = await journal.read()
            child_id = next(
                key for key in state["operations"] if key.startswith("inference:")
            )
            assert child_id == "inference:original-inference:render.context.start"
            # Dispatch with a lost condition result must retain its original
            # context/operation identity even after compute itself has ended.
            await journal.update(
                lambda s: s.update(model_context_segment_ids=initial_ids)
            )
            await journal.update(
                lambda s: s["operations"][first["request_id"]].update(
                    state="reconciling", model_context_execution=dict(state="unknown")
                )
            )
            before = await journal.read()
            with pytest.raises(RenderError) as blocked:
                await inference.enable(
                    {**operation, "operation_id": "unknown-reenable"}
                )
            assert blocked.value.code == "RENDER_TRANSITION_IN_PROGRESS"
            assert await journal.read() == before
        finally:
            await asyncio.gather(*tasks)
            await adapter.close()

    asyncio.run(run())


def test_disabled_journal_fences_a_condition_that_raced_async_admission():
    async def run():
        journal = await setup()
        await journal.initialize_inference(manual=True)
        before = await journal.read()
        with pytest.raises(RenderError) as blocked:
            await journal.start(
                command(
                    "render.start",
                    "raced",
                    expected_render_revision=0,
                    prompt="trusted",
                )
            )
        assert (
            blocked.value.code == "MODEL_UNAVAILABLE" and await journal.read() == before
        )

    asyncio.run(run())


def test_binding_close_reconciles_cancelled_inflight_admission_before_free():
    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import ModelLeases
    from sglang.multimodal_gen.runtime.realtime.whitebox.model_binding import (
        ModelBinding,
    )

    async def run():
        entered, admitted, released = asyncio.Event(), asyncio.Event(), asyncio.Event()
        assignments = []

        class Coordinator:
            async def admit(self, **_):
                entered.set()
                await admitted.wait()
                assignment = SimpleNamespace(vae=None)
                assignments.append(assignment)
                return assignment

            async def release_verified(self, assignment):
                assert assignment is assignments[0]
                await released.wait()

        leases = ModelLeases(
            Coordinator(),
            None,
            user_id="u",
            model_revision="m",
            vae_fingerprint="local",
            trace_id="t",
            max_bytes=100,
            strict_release=True,
        )
        binding = ModelBinding(leases, AsyncMock(), timeout=2)
        binding.start()
        await entered.wait()
        closing = asyncio.create_task(binding.close())
        await asyncio.sleep(0)
        assert not closing.done()
        admitted.set()
        async with asyncio.timeout(1):
            while leases.resource_state != "releasing":
                await asyncio.sleep(0)
        assert not closing.done()
        released.set()
        await closing
        assert len(assignments) == 1 and leases.resource_state == "released"
        assert binding.closed and binding.stopped.is_set()

    asyncio.run(run())


def test_history_limit_preserves_disable_and_failed_cleanup_retry(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.whitebox import render_contract

    async def run():
        monkeypatch.setattr(render_contract, "MAX_OPERATIONS", 2)
        journal = await setup()
        await journal.initialize_inference(manual=True)
        await journal.update(
            lambda state: state["inference"].update(enabled=True, state="enabled")
        )
        for index in range(2):
            await journal.command(
                command("display.configure", f"content{index}"),
                lambda s, o: o.update(state="completed"),
            )
        before = await journal.read()
        normal_limit = (
            render_contract.MAX_JOURNAL_BYTES
            - render_contract.JOURNAL_SHUTDOWN_RESERVE_BYTES
        )
        pad = "x" * (normal_limit - len(render_contract.canonical(before)) - 32)
        await journal.update(lambda state: state.update(pad=pad))
        with pytest.raises(RenderError) as byte_full:
            await journal.update(lambda state: state.update(overflow="x" * 128))
        assert byte_full.value.code == "LIMIT_EXCEEDED"
        request = command(
            "model.inference.configure",
            "final-off",
            enabled=False,
            expected_inference_revision=0,
        )
        stop = await journal.configure_inference(request)
        assert stop["state"] == "accepted" and len(journal.snapshot["operations"]) == 3
        await journal.settle_inference(
            "final-off", 1, state="failed", code="RELEASE_UNKNOWN"
        )
        assert journal.snapshot["inference"]["retry"] == dict(
            request_id="final-off", expected_inference_revision=0
        )
        retry = await journal.configure_inference(request)
        assert retry["state"] == "accepted" and retry["resumed"]
        assert retry["inference"] == dict(enabled=False, revision=1, state="stopping")
        assert len(journal.snapshot["operations"]) == 3
        await journal.settle_inference("final-off", 1, state="disabled")
        assert (
            len(render_contract.canonical(journal.snapshot))
            <= render_contract.MAX_JOURNAL_BYTES
        )
        with pytest.raises(RenderError) as exhausted:
            await journal.configure_inference(
                command(
                    "model.inference.configure",
                    "extra-noop",
                    enabled=False,
                    expected_inference_revision=1,
                )
            )
        assert exhausted.value.code == "LIMIT_EXCEEDED"
        assert journal.snapshot["inference"] == dict(
            enabled=False, revision=1, state="disabled"
        )
        # Auto/internal journals retain their original full 2 MiB acceptance.
        auto = await setup()
        await auto.initialize_inference(manual=False)
        await auto.update(lambda state: state.update(pad="x" * (normal_limit + 128)))
        assert len(render_contract.canonical(auto.snapshot)) > normal_limit

    asyncio.run(run())


@pytest.mark.parametrize("entered_before_disable", [False, True])
def test_superseded_inference_intent_settles_without_forging_model_receipts(
    entered_before_disable,
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    async def run():
        journal = await setup()
        await journal.initialize_inference(manual=True)
        entered = asyncio.Event()
        source = SimpleNamespace(
            journal=journal,
            spawn_control=asyncio.create_task,
            operation_message=AsyncMock(),
            status_snapshot=AsyncMock(),
            closing=False,
        )
        owner = SourceInference(
            source,
            dict(
                source_operation="create.manual",
                inference_control="manual",
                prompt="frozen",
            ),
        )

        async def prepare(_):
            entered.set()
            await asyncio.Event().wait()

        owner.enable = AsyncMock(side_effect=prepare)
        owner.disable = AsyncMock()
        enable = await journal.configure_inference(
            command(
                "model.inference.configure",
                "enable-pending",
                enabled=True,
                expected_inference_revision=0,
            )
        )
        model = await journal.start(
            command(
                "render.start",
                "independent-model",
                expected_render_revision=0,
                prompt="frozen model condition",
            )
        )
        model_before = copy.deepcopy(
            journal.snapshot["operations"][model["request_id"]]
        )
        segment_before = copy.deepcopy(journal.snapshot["segment"])
        owner.dispatch(enable)
        if entered_before_disable:
            await entered.wait()
        stop = await journal.configure_inference(
            command(
                "model.inference.configure",
                "disable-pending",
                enabled=False,
                expected_inference_revision=1,
            )
        )
        # The accepted opposite intent settles its predecessor in the same
        # CAS, even if that predecessor never entered its coroutine body.
        during_stop = journal.snapshot
        assert during_stop["operations"]["enable-pending"]["state"] == "completed"
        assert during_stop["inference"]["state"] == "stopping"
        owner.dispatch(stop)
        await owner.task
        assert owner.enable.await_count == int(entered_before_disable)
        state = await journal.read()
        old = state["operations"]["enable-pending"]
        assert (
            old["state"] == "completed" and old["superseded_by_inference_revision"] == 2
        )
        assert state["inference"] == dict(enabled=False, revision=2, state="disabled")
        assert state["segment"] == segment_before
        assert state["operations"][model["request_id"]] == model_before

    asyncio.run(run())


def test_original_source_command_resumes_same_failed_stop_without_new_revision():
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def run():
        journal = await setup()
        await journal.initialize_inference(manual=True)
        await journal.update(
            lambda state: state["inference"].update(enabled=True, state="enabled")
        )
        request = command(
            "model.inference.configure",
            "original-stop",
            version=2,
            enabled=False,
            expected_inference_revision=0,
        )
        await journal.configure_inference(request)
        await journal.settle_inference(
            "original-stop", 1, state="failed", code="RELEASE_UNKNOWN"
        )
        source = object.__new__(SourceSession)
        source.journal = journal
        source.binding_ended = False
        source.operation_message = AsyncMock()
        source.status_snapshot = AsyncMock()
        dispatched = []
        source.inference = SimpleNamespace(
            manual=True, fence=lambda **_: None, dispatch=dispatched.append
        )
        await source.command(request)
        assert len(dispatched) == 1 and dispatched[0]["resumed"]
        assert journal.snapshot["inference"] == dict(
            enabled=False, revision=1, state="stopping"
        )
        assert list(journal.snapshot["operations"]) == ["original-stop"]

    asyncio.run(run())


def test_explicit_binding_cleanup_retry_uses_original_verified_assignment():
    from test_whitebox_model_binding import Assignment, Socket

    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import ModelLeases
    from sglang.multimodal_gen.runtime.realtime.whitebox.model_binding import (
        ModelBinding,
    )

    async def run():
        allowed = False
        admits, releases = [], []
        assignment = Assignment(
            denoiser=SimpleNamespace(worker_epoch="a", endpoint="ws://cpu-peer")
        )
        assignment.token, assignment.vae = "cpu-fixture", None

        class Coordinator:
            async def admit(self, **request):
                admits.append(request)
                return assignment

            async def release_verified(self, original):
                assert original is assignment
                releases.append(original)
                if not allowed:
                    raise ConnectionError("release receipt unavailable")

            async def quarantine(self, _):
                raise ConnectionError("quarantine receipt unavailable")

        async def connect(*_, **__):
            return Socket()

        leases = ModelLeases(
            Coordinator(),
            connect,
            user_id="u",
            model_revision="m",
            vae_fingerprint="local",
            trace_id="t",
            max_bytes=100,
            strict_release=True,
            cleanup_timeout_s=0.01,
        )
        binding = ModelBinding(leases, AsyncMock(), timeout=2)
        await binding.wait_ready()
        with pytest.raises(RenderError, match="cleaned up"):
            await binding.close()
        assert leases.resource_state == "release_unknown" and len(admits) == 1
        allowed = True
        await binding.close()
        assert leases.resource_state == "released" and leases.cleanup_code is None
        assert len(admits) == 1 and len(releases) == 2

    asyncio.run(run())


def test_failed_manual_parent_cleanup_does_not_end_world_deadline(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.whitebox import render_source

    async def run():
        journal = await setup()
        await journal.initialize_inference(manual=True)
        await journal.update(
            lambda state: state["inference"].update(
                enabled=True, state="failed", code="RELEASE_UNKNOWN"
            )
        )
        source = object.__new__(render_source.SourceSession)
        source.journal = journal
        source.inference = SimpleNamespace(manual=True)
        source.model_binding = SimpleNamespace(closed=True, assignment=None)
        source.control_prefill = source.startup = source.current_segment = None
        source.engine = None
        source.finished = asyncio.Event()
        source.config = SimpleNamespace(play_seconds=300)
        source.last_tick, source.bound_at = 0, 0
        source.binding_ended = False
        source.source_state_value, source.display_mode = "ready", "whitebox"
        source.play_elapsed = source.wait_elapsed = 0
        source.unbind = AsyncMock()

        async def once(_):
            source.finished.set()

        monkeypatch.setattr(render_source.asyncio, "sleep", once)
        monkeypatch.setattr(render_source.time, "monotonic", lambda: 1)
        await source._deadline()
        source.unbind.assert_not_awaited()
        assert source.play_elapsed == 1

    asyncio.run(run())


def test_cancelling_enable_does_not_cancel_original_segment_transition_owner():
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    async def run():
        journal = await setup()
        await journal.initialize_inference(manual=True)
        request = await journal.configure_inference(
            command(
                "model.inference.configure",
                "start-race",
                enabled=True,
                expected_inference_revision=0,
            )
        )
        entered, finish = asyncio.Event(), asyncio.Event()
        controls = []
        source = SimpleNamespace(
            journal=journal,
            closed=False,
            binding_ended=False,
            model_binding=None,
            context_channel=None,
            start_model_binding=lambda _: None,
            config=SimpleNamespace(startup_seconds=1),
            current_segment=None,
            retained=SimpleNamespace(frames=[(1, SimpleNamespace(index=1), 0, 0)]),
            initial_assets=[],
            assets=SimpleNamespace(resolve=AsyncMock(return_value=[])),
            session_id="source",
            source_epoch="epoch",
            operation_message=AsyncMock(),
            outgoing=asyncio.Queue(),
            source_identity=lambda: {},
        )

        def spawn(coroutine):
            task = asyncio.create_task(coroutine)
            controls.append(task)
            return task

        source.spawn_control = spawn

        async def transition(*_):
            entered.set()
            await finish.wait()

        source._transition_segment = transition
        owner = SourceInference(
            source,
            dict(
                source_operation="create.manual",
                inference_control="manual",
                prompt="frozen",
            ),
        )
        enabling = asyncio.create_task(owner.enable(request))
        try:
            await entered.wait()
            enabling.cancel()
            await asyncio.gather(enabling, return_exceptions=True)
            assert (
                len(controls) == 1
                and not controls[0].cancelled()
                and not controls[0].done()
            )
        finally:
            finish.set()
            enabling.cancel()
            await asyncio.gather(enabling, *controls, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("dispatched", [False, True])
def test_cancelled_harness_init_retains_unknown_only_after_original_dispatch(
    tmp_path, dispatched
):
    from test_whitebox_source_conditions import setup as context_setup

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def run():
        adapter, condition, journal, _, _, _ = await context_setup(tmp_path)
        original = await journal.model_condition(condition, [])
        segment_state = (await journal.read())["segment"]
        segment = SimpleNamespace(
            segment_id=original["segment_id"],
            segment_epoch=original["segment_epoch"],
            resource_state="released",
            archive_failure=False,
            failure_diagnostic={},
            segment=segment_state,
            model_init_dispatched=dispatched,
            pending_prompt=None,
        )
        source = object.__new__(SourceSession)
        source.journal = journal
        source.current_segment = segment
        source.inference = None
        source.source_conditions = None
        source.outgoing = asyncio.Queue()
        source.session_id, source.source_epoch = "source", "epoch"
        source.engine = SimpleNamespace(stream_id="view")
        try:
            await source.segment_finished(segment, "cancelled", None)
            result = (await journal.read())["operations"][original["request_id"]]
            restart = command(
                "render.context.start",
                "after-cancel",
                expected_render_revision=journal.snapshot["render_revision"],
            )
            if dispatched:
                assert result["model_context_execution"]["state"] == "unknown"
                assert not result.get("never_applied")
                with pytest.raises(RenderError) as blocked:
                    await journal.start_model_context_segment(restart)
                assert blocked.value.code == "RENDER_TRANSITION_IN_PROGRESS"
            else:
                assert result["never_applied"] is True
                assert (await journal.start_model_context_segment(restart))[
                    "segment_id"
                ] != original["segment_id"]
        finally:
            await adapter.close()

    asyncio.run(run())


def test_manual_harness_pause_probe_rejects_old_world_before_model_reservation():
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    async def run():
        journal = await setup()
        await journal.initialize_inference(manual=True)
        operation = await journal.configure_inference(
            command(
                "model.inference.configure",
                "enable-old-world",
                enabled=True,
                expected_inference_revision=0,
            )
        )

        async def probe(**_):
            raise RenderError("MODEL_CONTEXT_SUBSCRIPTION_PAUSE_UNSUPPORTED")

        source = SimpleNamespace(
            journal=journal,
            closed=False,
            binding_ended=False,
            model_binding=None,
            context_channel=SimpleNamespace(probe_subscription_pause=probe),
            config=SimpleNamespace(startup_seconds=1),
            start_model_binding=lambda _: pytest.fail(
                "old World acquired model compute before capability proof"
            ),
        )
        owner = SourceInference(
            source,
            dict(
                source_operation="create.manual",
                inference_control="manual",
                prompt="frozen",
            ),
        )
        with pytest.raises(RenderError) as unsupported:
            await owner.enable(operation)
        assert unsupported.value.code == "MODEL_CONTEXT_SUBSCRIPTION_PAUSE_UNSUPPORTED"

    asyncio.run(run())


@pytest.mark.parametrize("supported", [True, False, None])
def test_inactive_cli_pause_probe_requires_actual_binding_ack(tmp_path, supported):
    import json

    from test_whitebox_source_conditions import header
    from test_whitebox_source_conditions import setup as context_setup

    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )

    async def run():
        adapter, c, _, _, submits, _ = await context_setup(tmp_path)
        sent = []

        class Socket:
            async def send(self, value):
                sent.append(json.loads(value))

        view = SimpleNamespace(
            descriptor=dict(
                epoch=1,
                modelContext=dict(
                    protocol=c["protocol"],
                    required=True,
                    configHash=c["configHash"],
                    gameBuildHash=c["source"]["gameBuildHash"],
                    compilerVersion=1,
                    visible=False,
                ),
            ),
            client_id=c["audience"]["consumerId"],
            closed=False,
            stopped=False,
            control=Socket(),
            transport_generation=0,
        )
        channel = CLIConditionChannel(view)
        try:
            await channel.bind(
                segment_id=c["segmentId"],
                create_adapter=lambda *_: adapter,
                start_suspended=True,
            )
            probing = asyncio.create_task(channel.probe_subscription_pause(timeout=1))
            await asyncio.sleep(0)
            request = sent[-1]
            assert (
                request["requireSubscriptionPause"] is True
                and request["paused"] is True
            )
            assert channel.suspended and not submits
            assert await channel.frame(header(c)) is None
            channel.offer(c)
            channel.offer(dict(type="model.context.sample"))
            await asyncio.sleep(0)
            assert channel.queue.empty() and not submits
            reply = dict(
                type="model.context.subscribed",
                epoch=1,
                protocol=c["protocol"],
                segmentId=c["segmentId"],
                consumer=c["audience"],
                sampleControlFloor=1,
                paused=True,
            )
            if supported is not None:
                reply["subscriptionPause"] = supported
            channel.offer(reply)
            if supported:
                await probing
                assert await channel.frame(header(c)) is None
                assert (
                    channel.suspended
                    and not submits
                    and channel.consumer == c["audience"]
                )
                await channel.subscribe()
                assert (
                    sent[-1]["requireSubscriptionPause"] is True
                    and "paused" not in sent[-1]
                )
            else:
                with pytest.raises(RenderError) as error:
                    await probing
                assert (
                    error.value.code == "MODEL_CONTEXT_SUBSCRIPTION_PAUSE_UNSUPPORTED"
                )
                assert channel.suspended and channel.consumer is None and not submits
        finally:
            if "probing" in locals():
                probing.cancel()
                await asyncio.gather(probing, return_exceptions=True)
            await channel.close()

    asyncio.run(run())


def test_auto_source_refuses_manual_control_without_changing_legacy_binding():
    import msgspec

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def run():
        journal = await setup()
        await journal.initialize_inference(manual=False)
        source = object.__new__(SourceSession)
        source.journal = journal
        source.inference = SimpleNamespace(manual=False)
        source.context_channel = None
        source.outgoing = asyncio.Queue()
        await source.command(
            command(
                "model.inference.configure",
                "auto-off",
                enabled=False,
                expected_inference_revision=0,
            )
        )
        rejected = msgspec.msgpack.decode(await source.outgoing.get())
        assert rejected["code"] == "CAPABILITY_UNAVAILABLE"
        assert (
            not journal.snapshot["operations"]
            and journal.snapshot["inference"]["enabled"] is True
        )

    asyncio.run(run())


def test_bound_manual_room_admits_two_live_views_without_model_slots(
    tmp_path, monkeypatch
):
    import base64

    import msgspec
    from test_whitebox_player_media import media_launch, model_owners

    async def run():
        monkeypatch.setenv("WHITEBOX_BOUND_COMPUTE", "1")
        launch = media_launch()
        init = msgspec.msgpack.decode(base64.b64decode(launch["initMsgpackBase64"]))
        init.update(source_operation="create.manual", inference_control="manual")
        launch["initMsgpackBase64"] = base64.b64encode(
            msgspec.msgpack.encode(init)
        ).decode()
        env = await model_owners(tmp_path, monkeypatch, launch=launch, max_sources=2)
        try:
            first, _ = await env.admit("alice")
            second, _ = await env.admit("bob")
            async with asyncio.timeout(10):
                while not all(
                    record["media_pipeline"].session.ready.is_set()
                    for record in (first, second)
                ):
                    await asyncio.sleep(0.01)
            assert env.workers == []
            assert len(env.views) == 2
            assert all(
                record.get("model_binding") is None
                and record["media_pipeline"].session.model_binding is None
                for record in (first, second)
            )
            assert await env.runtime.leave(first["stream_id"])
            assert not env.views[second["player_id"]].closed
            await env.runtime.event(
                second["stream_id"],
                "input",
                dict(sequence=0, input=dict(held_keys=["KeyD"])),
            )
            assert env.views[second["player_id"]].inputs == [dict(held_keys=["KeyD"])]
            assert env.workers == []
        finally:
            await env.close()

    asyncio.run(run())


@pytest.mark.parametrize("applied", [True, False])
def test_disable_drains_only_dispatched_query_before_releasing_child(applied):
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        journal, scope = await ordinary_query_source()
        await journal.player_instruction(
            command(
                "model.context.intervention",
                "last-query",
                **scope,
                expectedRevision=0,
                prompt="Warm sunlight.",
            )
        )
        pending = copy.deepcopy(journal.snapshot["segment"]["prompt_queue"][0])
        await journal.player_instruction(
            command(
                "model.context.intervention",
                "queued-query",
                **scope,
                expectedRevision=1,
                prompt="Even warmer sunlight.",
            )
        )
        queued = copy.deepcopy(journal.snapshot["segment"]["prompt_queue"][-1])
        source = object.__new__(SourceSession)
        source.journal = journal
        source.outgoing = asyncio.Queue()
        source.source_identity = lambda: {}
        source._status = AsyncMock()
        source.current_model_stream, source.display_mode = "old", "model"
        source.context_channel = source.model_binding = None
        source.startup = source.control_prefill = None
        source.config = SimpleNamespace(drain_seconds=0.04)
        source.segment_lock = asyncio.Lock()
        stopped = asyncio.Event()
        segment = SimpleNamespace(
            segment_id=scope["segment_id"],
            segment=copy.deepcopy(journal.snapshot["segment"]),
            pending_prompt=pending,
            publishing=True,
            source_view=SimpleNamespace(stop=stopped.set),
            closed_event=asyncio.Event(),
            resource_state="released",
            failure_code=None,
            archive_failure=False,
        )
        source.current_segment = segment

        async def stop(*, cancel):
            stopped.set()
            await source.segment_finished(segment, "cancelled", None)
            segment.closed_event.set()

        segment.stop = stop
        owner = SourceInference(source, dict(prompt="Original scene."))
        await journal.configure_inference(
            command(
                "model.inference.configure",
                "off-pending",
                enabled=False,
                expected_inference_revision=0,
            )
        )
        disabling = asyncio.create_task(owner.disable(drain_pending=True))
        try:
            await asyncio.sleep(0.005)
            assert not stopped.is_set(), "Dispatched query still needs source frames"
            assert not segment.publishing
            assert journal.snapshot["segment"]["state"] == "draining"
            if applied:
                await journal.prompt_applied(
                    scope["segment_id"], 2, dict(first_output_frame=8, num_frames=1)
                )
                segment.pending_prompt = None
            await asyncio.wait_for(disabling, 12)
            operation = journal.snapshot["operations"][pending["request_id"]]
            assert operation["state"] == ("applied" if applied else "reconciling")
            if not applied:
                assert operation["code"] == "SEGMENT_ENDED_UNCONFIRMED"
                assert not operation.get("never_applied", False)
            assert stopped.is_set()
            queued_result = journal.snapshot["operations"][queued["request_id"]]
            assert queued_result["state"] == "cancelled"
            assert queued_result["never_applied"] is True
        finally:
            disabling.cancel()
            await asyncio.gather(disabling, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("close_requested", [False, True])
def test_normal_player_finish_does_not_report_model_child_failure(close_requested):
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    async def scenario():
        journal, _ = await ordinary_query_source()
        stopped = asyncio.Event()
        segment = SimpleNamespace(
            publishing=True,
            pending_prompt=None,
            source_view=SimpleNamespace(stop=stopped.set),
        )
        source = SimpleNamespace(
            journal=journal,
            current_segment=segment,
            close_requested=close_requested,
            context_channel=None,
            status_snapshot=AsyncMock(),
        )
        owner = SourceInference(
            source,
            dict(
                source_operation="create.manual",
                inference_control="manual",
                prompt="Original scene.",
            ),
        )
        await owner.segment_ended(segment, None)
        inference = journal.snapshot["inference"]
        if close_requested:
            assert inference["state"] == "enabled"
            assert "code" not in inference
            assert not stopped.is_set()
        else:
            assert inference["state"] == "failed"
            assert inference["code"] == "MODEL_UNAVAILABLE"
            assert stopped.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("release_sender", [True, False])
def test_disable_serializes_with_query_waiting_for_model_wire(release_sender):
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        journal, scope = await ordinary_query_source()
        await journal.player_instruction(
            command(
                "model.context.intervention",
                "query-race",
                **scope,
                expectedRevision=0,
                prompt="Warm sunlight.",
            )
        )
        update_request_id = journal.snapshot["segment"]["prompt_queue"][0]["request_id"]
        source = object.__new__(SourceSession)
        source.journal = journal
        source.current_model_stream, source.display_mode = "active", "model"
        source.context_channel = source.model_binding = None
        source.startup = source.control_prefill = None
        source.config = SimpleNamespace(drain_seconds=0.01)
        source.segment_lock = asyncio.Lock()
        source.outgoing = asyncio.Queue()
        source.source_identity = lambda: {}
        source._status = AsyncMock()
        segment = object.__new__(RenderSegment)
        source.current_segment = segment
        segment.owner, segment.segment_id = source, scope["segment_id"]
        segment.segment = copy.deepcopy(journal.snapshot["segment"])
        segment.closed_event = asyncio.Event()
        segment.source_admission_ready = asyncio.Event()
        segment.source_admission_ready.set()
        segment.prompt_available = asyncio.Event()
        segment.prompt_available.set()
        idle, stopped, sent = asyncio.Event(), asyncio.Event(), asyncio.Event()
        idle.set()
        segment.sink = SimpleNamespace(idle=idle, finish=AsyncMock())
        segment.pending_prompt = segment.condition_switch = None
        segment.active_reference_descriptors = []
        segment.selected_frames = 1
        segment.cancel_requested = False
        segment.source_view = SimpleNamespace(stopped=stopped, stop=stopped.set)
        segment.source_send_lock, segment.model_wire_lock = (
            asyncio.Lock(),
            asyncio.Lock(),
        )
        segment.model_wire_sequence = 0
        segment.prompt_by_event = {}
        from sglang.multimodal_gen.runtime.realtime.whitebox.timing_control import (
            TimingControl,
        )
        from sglang.multimodal_gen.runtime.realtime.whitebox.ttff_metrics import (
            TimingMetrics,
        )

        segment.detailed_timing = True
        segment.timing_control = TimingControl(segment)
        segment.timing_metrics = TimingMetrics()
        segment.room_timings = {}
        segment.timing_triggers = {}
        segment.config = SimpleNamespace(startup_seconds=20)
        segment.publishing = True
        segment.resource_state, segment.archive_failure = "released", False
        segment.failure_code = None
        source.latest = SimpleNamespace(
            get=AsyncMock(return_value=(1, (SimpleNamespace(index=0), 0)))
        )
        owner = SourceInference(source, dict(prompt="Original scene."))
        await segment.model_wire_lock.acquire()
        sender = asyncio.create_task(
            segment._condition_controls(
                SimpleNamespace(send=AsyncMock(side_effect=lambda _: sent.set()))
            )
        )
        stop_called = asyncio.Event()

        async def stop(*, cancel):
            assert cancel
            stop_called.set()
            stopped.set()
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            await source.segment_finished(segment, "cancelled", None)
            segment.closed_event.set()

        segment.stop = stop
        disabling = None
        try:
            async with asyncio.timeout(1):
                while segment.condition_switch is None:
                    if sender.done():
                        sender.result()
                    await asyncio.sleep(0)
            assert segment.source_send_lock.locked() and segment.pending_prompt is None
            await journal.configure_inference(
                command(
                    "model.inference.configure",
                    "off-race",
                    enabled=False,
                    expected_inference_revision=0,
                )
            )
            owner.fence(drain_pending=True)
            disabling = asyncio.create_task(owner.disable(drain_pending=True))
            await asyncio.sleep(0.005)
            assert not stopped.is_set(), (
                "Sender has not yet crossed the original dispatch boundary"
            )
            assert not stop_called.is_set()
            if release_sender:
                segment.model_wire_lock.release()
                await asyncio.wait_for(sent.wait(), 1)
                assert not stopped.is_set()
                pending = segment.pending_prompt
                assert pending is not None
                await journal.prompt_applied(
                    scope["segment_id"],
                    pending["version"],
                    dict(first_output_frame=8, num_frames=1),
                )
                segment.pending_prompt = None
            await asyncio.wait_for(disabling, 12)
            update = journal.snapshot["operations"][update_request_id]
            assert update["state"] == ("applied" if release_sender else "cancelled")
            if not release_sender:
                assert update["never_applied"] is True
                assert not sent.is_set()
            assert stopped.is_set() and stop_called.is_set()
        finally:
            if disabling is not None:
                disabling.cancel()
                await asyncio.gather(disabling, return_exceptions=True)
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            if segment.model_wire_lock.locked():
                segment.model_wire_lock.release()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "reason, expected",
    [
        ("CAPACITY_EXHAUSTED", "CAPACITY_EXHAUSTED"),
        ("USER_SESSION_LIMIT", "USER_SESSION_LIMIT"),
        ("USER_ALREADY_ACTIVE", "USER_ALREADY_ACTIVE"),
        ("private upstream detail", "INITIALIZATION_FAILED"),
    ],
)
@pytest.mark.parametrize("cleanup_unknown", [False, True])
def test_manual_bound_admission_failure_preserves_safe_reason_and_cleanup(
    reason, expected, cleanup_unknown
):
    from contextlib import asynccontextmanager

    from sglang.multimodal_gen.runtime.realtime.coordinator import CoordinatorRejected
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.model_binding import (
        ModelBinding,
    )

    async def scenario():
        journal = await setup()
        await journal.initialize_inference(manual=True)

        class RejectedLease:
            resource_state = "release_unknown" if cleanup_unknown else "none"

            @asynccontextmanager
            async def open(self):
                raise CoordinatorRejected(reason, retry_after_s=0.25)
                yield  # The real bound owner receives the admission exception.

        source = SimpleNamespace(
            journal=journal,
            closed=False,
            binding_ended=False,
            context_channel=None,
            current_segment=None,
            model_binding=None,
            startup=None,
            control_prefill=None,
            segment_lock=asyncio.Lock(),
            operation_message=AsyncMock(),
            status_snapshot=AsyncMock(),
        )
        bindings = []

        def start_binding(_):
            binding = ModelBinding(RejectedLease(), AsyncMock(), timeout=1)
            bindings.append(binding)
            source.model_binding = binding

        source.start_model_binding = start_binding
        owner = SourceInference(
            source,
            dict(
                source_operation="create.manual",
                inference_control="manual",
                prompt="frozen",
            ),
        )
        # Retry stays on the same source and uses the next inference revision.
        for index in range(2):
            operation = await journal.configure_inference(
                command(
                    "model.inference.configure",
                    f"admission-{index}",
                    enabled=True,
                    expected_inference_revision=index,
                )
            )
            await owner.run(operation)
            code = "RELEASE_UNKNOWN" if cleanup_unknown else expected
            assert journal.snapshot["inference"] == dict(
                enabled=True, revision=index + 1, state="failed", code=code
            )
            receipt = source.operation_message.await_args.args[0]
            assert receipt["request_id"] == f"admission-{index}"
            assert receipt["state"] == "failed" and receipt["code"] == code
            assert "private upstream detail" not in repr(receipt)
            assert source.display_mode == "whitebox" and not source.closed
            assert source.model_binding is (bindings[0] if cleanup_unknown else None)
        # An unknown release must block a new model reservation on retry.
        assert len(bindings) == (1 if cleanup_unknown else 2)
        assert all(binding.stopped.is_set() for binding in bindings)
        assert source.status_snapshot.await_count == 2

    asyncio.run(scenario())
