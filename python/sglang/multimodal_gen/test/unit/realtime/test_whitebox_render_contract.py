import asyncio
import copy

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
    RenderError,
    RenderJournal,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore


def run(coro):
    return asyncio.run(coro)


def test_unbind_cleanup_timeout_does_not_end_source():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        source = object.__new__(SourceSession)
        source.binding_ended = False
        source.source_state_value = "ready"
        source.current_model_stream = "model"
        source.finished = asyncio.Event()

        async def stop(**_):
            raise TimeoutError("cleanup still pending")

        source.current_segment = SimpleNamespace(
            closed_event=asyncio.Event(), stop=stop
        )
        events = []

        async def status(state, **fields):
            events.append((state, fields))

        source._status = status
        await source.unbind("BINDING_BUDGET_EXHAUSTED")
        assert source.binding_ended and not source.finished.is_set()
        assert source.current_model_stream is None and events[-1][1]["cleanup_pending"]

    run(scenario())


@pytest.mark.parametrize("play_seconds", [120, 300, 540])
def test_startup_time_is_wait_budget_not_effective_play(monkeypatch, play_seconds):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox import render_source

    async def scenario():
        clock = SimpleNamespace(now=100.0)
        monkeypatch.setattr(
            render_source, "time", SimpleNamespace(monotonic=lambda: clock.now)
        )
        source = object.__new__(render_source.SourceSession)
        source.finished = asyncio.Event()
        source.config = SimpleNamespace(play_seconds=play_seconds)
        source.engine = None
        source.bound_at = source.last_tick = 0.0
        source.binding_ended = False
        source.current_segment = None
        source.current_model_stream = None
        source.play_elapsed = source.wait_elapsed = 0.0
        source.source_state_value = "binding"
        source.display_mode = "whitebox"

        task = asyncio.create_task(source._deadline())
        await asyncio.sleep(0.12)
        assert source.wait_elapsed == 100 and source.play_elapsed == 0
        source.source_state_value = "ready"
        clock.now = 100 + play_seconds - 0.1
        await asyncio.sleep(0.12)
        assert not source.binding_ended
        clock.now += 0.2
        # The pipeline owns unified player cleanup after deadline failure;
        # the timer itself must not pretend the Slot has been released.
        with pytest.raises(RenderError) as failure:
            await asyncio.wait_for(task, 1)
        assert failure.value.code == "BINDING_BUDGET_EXHAUSTED"
        assert source.play_elapsed >= play_seconds and not source.binding_ended

    run(scenario())


def test_cancel_unstarted_successor_does_not_use_predecessor_release_state():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        journal = await setup()
        accepted = await journal.start(
            command("render.start", "new", expected_render_revision=0, prompt="next")
        )
        identity = {key: accepted[key] for key in ("segment_id", "segment_epoch")}
        ending = command(
            "render.end", "end", expected_render_revision=1, mode="cancel", **identity
        )
        operation = await journal.end(ending)
        source = object.__new__(SourceSession)
        source.journal = journal
        source.segment_lock = asyncio.Lock()
        closed = asyncio.Event()
        closed.set()
        source.current_segment = SimpleNamespace(
            segment_id="old", resource_state="released", closed_event=closed
        )
        observed = []

        async def reply(value):
            observed.append(value)

        source.operation_message = reply
        await source._end_segment(ending, operation)
        state = await journal.read()
        assert (
            state["segment"]["state"] == "cancelled"
            and state["segment"]["resource_state"] == "none"
        )
        assert observed[-1]["resource_state"] == "none"
        await journal.start(
            command("render.start", "after", expected_render_revision=2, prompt="after")
        )

    run(scenario())


def command(op, request_id, **fields):
    return {
        "op": op,
        "request_id": request_id,
        "source_session_id": "source",
        "source_epoch": "epoch",
        **fields,
    }


async def setup():
    journal = RenderJournal(
        MemoryRoomStore(),
        user_id="u",
        source_session_id="source",
        source_epoch="epoch",
        owner_epoch="owner",
    )
    await journal.create()
    await journal.update(lambda state: state.update(source_state="ready"))
    return journal


async def started():
    journal = await setup()
    reply = await journal.start(
        command(
            "render.start",
            "start",
            expected_render_revision=0,
            prompt="initial",
            references=[],
        )
    )
    await journal.update(
        lambda state: state["segment"].update(
            state="streaming", resource_state="reserved", applied_version=1
        )
    )
    return journal, {k: reply[k] for k in ["segment_id", "segment_epoch"]}


def test_start_is_exclusive_before_first_frame_and_replays_same_identity():
    async def scenario():
        journal = await setup()
        body = command(
            "render.start", "start", expected_render_revision=0, prompt="initial"
        )
        a, b = await asyncio.gather(
            journal.start(body), journal.start(copy.deepcopy(body))
        )
        assert (
            a["segment_id"] == b["segment_id"]
            and a["operation_id"] == b["operation_id"]
        )
        assert (await journal.read())["render_revision"] == 1
        with pytest.raises(RenderError) as conflict:
            await journal.start(
                {**body, "request_id": "second", "expected_render_revision": 1}
            )
        assert conflict.value.code == "RENDER_TRANSITION_IN_PROGRESS"
        with pytest.raises(RenderError) as conflict:
            await journal.start({**body, "prompt": "different"})
        assert conflict.value.code == "REQUEST_ID_CONFLICT"

    run(scenario())


def test_stale_source_and_unreleased_segment_cannot_be_reused():
    async def scenario():
        journal, identity = await started()
        with pytest.raises(RenderError) as error:
            await journal.end(
                command(
                    "render.end",
                    "end",
                    **identity,
                    source_epoch="old",
                    expected_render_revision=1,
                )
            )
        assert error.value.code == "SOURCE_EPOCH_MISMATCH"
        await journal.update(
            lambda s: s["segment"].update(
                state="failed", resource_state="release_unknown"
            )
        )
        with pytest.raises(RenderError) as error:
            await journal.start(
                command(
                    "render.start", "new", expected_render_revision=1, prompt="next"
                )
            )
        assert error.value.code == "RENDER_TRANSITION_IN_PROGRESS"

    run(scenario())


def test_display_does_not_create_or_release_a_segment():
    async def scenario():
        journal, _ = await started()
        before = (await journal.read())["segment"]
        for revision, mode in enumerate(["whitebox", "model", "whitebox", "model"]):
            await journal.display(
                command(
                    "display.configure",
                    "display" + str(revision),
                    expected_display_revision=revision,
                    mode=mode,
                )
            )
        after = await journal.read()
        assert after["segment"] == before and after["render_revision"] == 1

    run(scenario())


def test_replace_keeps_one_pending_intent_and_end_must_cancel_it_explicitly():
    async def scenario():
        journal, identity = await started()
        replacement = await journal.start(
            command(
                "render.replace",
                "replace",
                **identity,
                expected_render_revision=1,
                prompt="next",
            ),
            replace=True,
        )
        await journal.end(
            command("render.end", "old-end", **identity, expected_render_revision=2)
        )
        assert (await journal.read())["pending_replace"]["segment_id"] == replacement[
            "segment_id"
        ]
        result = await journal.end(
            command(
                "render.end",
                "cancel-all",
                **identity,
                expected_render_revision=3,
                cancel_pending_replace=True,
            )
        )
        assert result["cancelled_pending_segment"] == replacement["segment_id"]
        assert (await journal.read())["pending_replace"] is None

    run(scenario())


def test_prompt_failure_retains_version_chain_and_requires_explicit_rebase():
    async def scenario():
        journal, identity = await started()

        def update(version, **extra):
            return command(
                "prompt.replace",
                "p" + str(version),
                **identity,
                expected_prompt_version=version - 1,
                prompt_version=version,
                prompt="text" + str(version),
                effective={"source_frame_index": 100 * version},
                **extra,
            )

        await journal.prompt(update(2, one_shot_event_ids=["event"]))
        await journal.prompt(update(3))
        await journal.prompt_failed(
            identity["segment_id"], 2, never_applied=False, code="TIMEOUT"
        )
        assert len((await journal.read())["segment"]["prompt_queue"]) == 2
        await journal.prompt_failed(
            identity["segment_id"], 2, never_applied=True, code="CONDITION_INVALID"
        )
        state = await journal.read()
        assert (
            state["segment"]["accepted_version"] == 3
            and state["segment"]["applied_version"] == 1
        )
        assert state["operations"]["p3"]["code"] == "DEPENDENCY_FAILED"
        with pytest.raises(RenderError):
            await journal.prompt(update(4))
        await journal.prompt(
            update(
                4,
                base_applied_prompt_version=1,
                retry_of_request_id="p2",
                one_shot_event_ids=["event"],
            )
        )
        await journal.prompt_applied(
            identity["segment_id"],
            4,
            dict(first_output_frame=17, num_frames=8, effective_source_frame=404),
        )
        assert (await journal.read())["segment"]["applied_version"] == 4

    run(scenario())


def test_strict_prompt_is_not_accepted_until_worker_reserves_boundary():
    async def scenario():
        journal, identity = await started()
        body = command(
            "prompt.replace",
            "strict",
            **identity,
            expected_prompt_version=1,
            prompt_version=2,
            prompt="next",
            effective={"source_frame_index": 20, "policy": "strict"},
        )
        operation = await journal.prompt(body)
        assert (
            operation["state"] == "validating"
            and (await journal.read())["segment"]["accepted_version"] == 1
        )
        await journal.prompt_reserved(identity["segment_id"], "strict", accepted=False)
        assert (await journal.read())["segment"]["accepted_version"] == 1
        operation = await journal.prompt({**body, "request_id": "strict2"})
        await journal.prompt_reserved(identity["segment_id"], "strict2", accepted=True)
        assert (await journal.read())["segment"]["accepted_version"] == 2

    run(scenario())


def test_persistent_journal_reopens_and_fences_a_new_owner(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        SQLiteRenderStore,
    )

    async def scenario():
        path = tmp_path / "render-v2.sqlite3"
        store = SQLiteRenderStore(path)
        args = dict(
            user_id="u",
            source_session_id="source",
            source_epoch="epoch",
            owner_epoch="owner",
        )
        journal = RenderJournal(store, **args)
        await journal.create()
        await journal.update(lambda s: s.update(source_state="ready"))
        body = command(
            "render.start", "start", expected_render_revision=0, prompt="initial"
        )
        original = await journal.start(body)
        await store.close()
        store = SQLiteRenderStore(path)
        recovered = RenderJournal(store, **args)
        repeated = await recovered.start(body)
        assert repeated["operation_id"] == original["operation_id"]
        with pytest.raises(RenderError) as error:
            await RenderJournal(store, **{**args, "owner_epoch": "replacement"}).read()
        assert error.value.code == "ROOM_SESSION_LOST"
        await store.close()

    run(scenario())


# Worker reserves only untouched legal blocks; stale exact targets never become
# successful at-or-after updates by accident.
def test_worker_prompt_boundary_includes_admitted_prefetch():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_h3_source_adapter import (
        ZingH3SourceStreamAdapter,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan

    worker = object.__new__(ZingH3SourceStreamAdapter)
    worker.prompt_inflight = None
    worker.public_prompt_version = 1
    worker.joint_chunk_id = 0
    worker.ledger = SimpleNamespace(accepted_frames=18)
    worker.joint_plan = ChunkPlan([3, 2], True)
    payload = dict(text="new", version=2, min_source_frame=4, policy="at_or_after")
    assert worker._reserve_prompt(payload) == 26
    import pytest

    with pytest.raises(ValueError):
        worker._reserve_prompt({**payload, "policy": "strict"})
    assert (
        worker._reserve_prompt({**payload, "policy": "strict", "min_source_frame": 26})
        == 26
    )
    worker.prompt_inflight = 7
    with pytest.raises(ValueError):
        worker._reserve_prompt(payload)


def test_prompt_and_complete_reference_set_commit_as_one_version():
    async def scenario():
        journal, ids = await started()
        references = [
            dict(id=f"<Picture {i + 1}>", asset_id="sha256:" + "a" * 64, role="scene")
            for i in range(21)
        ]
        update = command(
            "prompt.replace",
            "with-21-images",
            **ids,
            expected_prompt_version=1,
            prompt_version=2,
            prompt="same text, new images",
            references=references,
            source_policy="skip_during_switch",
            effective=dict(source_frame_index=10, policy="at_or_after"),
        )
        await journal.prompt(update)
        assert journal.snapshot["segment"]["references"] == []
        references.clear()
        assert len(journal.snapshot["segment"]["prompt_queue"][0]["references"]) == 21
        await journal.prompt_applied(
            ids["segment_id"], 2, dict(first_output_frame=17, num_frames=9)
        )
        assert len(journal.snapshot["segment"]["references"]) == 21
        clear = command(
            "prompt.replace",
            "clear",
            **ids,
            expected_prompt_version=2,
            prompt_version=3,
            prompt="clear references",
            references=[],
            source_policy="skip_during_switch",
            effective=dict(source_frame_index=40, policy="at_or_after"),
        )
        await journal.prompt(clear)
        await journal.prompt_applied(
            ids["segment_id"], 3, dict(first_output_frame=34, num_frames=9)
        )
        assert journal.snapshot["segment"]["references"] == []
        with pytest.raises(RenderError):
            await journal.prompt({**clear, "request_id": "invalid", "references": None})
        with pytest.raises(RenderError, match="update time"):
            await journal.prompt(
                {
                    **clear,
                    "request_id": "strict",
                    "effective": dict(source_frame_index=50, policy="strict"),
                }
            )

    run(scenario())


@pytest.mark.parametrize(
    "configured,expected",
    [
        (None, 300),
        ("1", 1),
        ("540", 540),
        ("1800", 1800),
        ("1801", 1801),
        ("7200", 7200),
    ],
)
def test_render_play_budget_preserves_existing_environment_contract(
    monkeypatch, configured, expected
):
    from sglang.multimodal_gen.runtime.realtime.whitebox import render_source

    monkeypatch.delenv("WHITEBOX_RENDER_PLAY_SECONDS", raising=False)
    if configured is not None:
        monkeypatch.setenv("WHITEBOX_RENDER_PLAY_SECONDS", configured)
    resolver = getattr(render_source, "configured_render_play_seconds", None)
    assert callable(resolver), "Render sources must retain the configured play budget"
    assert resolver() == expected


@pytest.mark.parametrize("configured", ["0", "-1", "7201", "invalid", "1.5"])
def test_render_play_budget_rejects_invalid_configuration(monkeypatch, configured):
    from sglang.multimodal_gen.runtime.realtime.whitebox import render_source

    monkeypatch.setenv("WHITEBOX_RENDER_PLAY_SECONDS", configured)
    resolver = getattr(render_source, "configured_render_play_seconds", None)
    assert callable(resolver), "Render sources must retain the configured play budget"
    with pytest.raises(ValueError):
        resolver()


def test_full_operation_history_still_allows_one_idempotent_segment_end():
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        MAX_OPERATIONS,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        journal = await setup()
        starting = command(
            "render.start", "initial", expected_render_revision=0, prompt="initial"
        )
        started = await journal.start(starting)
        await journal.update(
            lambda state: state["operations"].update(
                {
                    f"history-{i}": dict(request_id=f"history-{i}", state="applied")
                    for i in range(MAX_OPERATIONS - 1)
                }
            )
        )
        before = await journal.read()
        with pytest.raises(RenderError) as blocked:
            await journal.display(
                command(
                    "display.configure",
                    "over-budget",
                    expected_display_revision=0,
                    mode="whitebox",
                )
            )
        assert blocked.value.code == "LIMIT_EXCEEDED"
        ending = command(
            "render.end",
            "safe-end",
            expected_render_revision=1,
            mode="cancel",
            segment_id=started["segment_id"],
            segment_epoch=started["segment_epoch"],
        )
        accepted = await journal.end(ending)
        duplicate = await journal.end(ending)
        assert (
            duplicate["duplicate"]
            and duplicate["operation_id"] == accepted["operation_id"]
        )
        current = await journal.read()
        assert current["render_revision"] == 2
        assert len(current["operations"]) == MAX_OPERATIONS + 1
        assert all(
            current["operations"][key] == value
            for key, value in before["operations"].items()
        )
        with pytest.raises(RenderError) as extra:
            await journal.end(
                dict(ending, request_id="another-end", expected_render_revision=2)
            )
        assert extra.value.code == "LIMIT_EXCEEDED"
        assert (await journal.start(starting))["duplicate"]

        # Exercise the original source-owner completion path for an unstarted model.
        owner = object.__new__(SourceSession)
        owner.journal, owner.segment_lock = journal, asyncio.Lock()
        owner.current_segment = None
        results = []

        async def reply(value):
            results.append(value)

        owner.operation_message = reply
        await owner._end_segment(ending, accepted)
        assert results[-1]["state"] == "completed"
        assert results[-1]["resource_state"] == "none"
        assert (await journal.end(ending))["state"] == "completed"
        next_source = RenderJournal(
            journal.store,
            user_id="u",
            source_session_id="next-source",
            source_epoch="next-epoch",
            owner_epoch="owner",
        )
        await next_source.create()
        assert not (await next_source.read())["operations"]
        assert len((await journal.read())["operations"]) == MAX_OPERATIONS + 1

    run(scenario())


@pytest.mark.parametrize("outcome", ["applied", "failed", "timeout", "cancel"])
def test_end_preserves_sent_condition_until_bounded_real_result(outcome):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        journal, ids = await started()
        update = await journal.prompt(
            command(
                "prompt.replace",
                "last-update",
                **ids,
                prompt="last",
                expected_prompt_version=1,
                prompt_version=2,
                effective={"source_frame_index": 0},
            )
        )
        ending = command(
            "render.end",
            "end-pending",
            **ids,
            expected_render_revision=1,
            mode="cancel" if outcome == "cancel" else "drain",
        )
        accepted = await journal.end(ending)
        source = object.__new__(SourceSession)
        source.journal, source.segment_lock = journal, asyncio.Lock()
        source.config = SimpleNamespace(drain_seconds=0.04)
        source.outgoing = asyncio.Queue()
        source.source_identity = lambda: {}
        source._status = lambda *a, **k: asyncio.sleep(0)
        segment = SimpleNamespace(
            segment_id=ids["segment_id"],
            segment=copy.deepcopy(journal.snapshot["segment"]),
            pending_prompt=copy.deepcopy(
                journal.snapshot["segment"]["prompt_queue"][0]
            ),
            closed_event=asyncio.Event(),
            resource_state="released",
            archive_failure=False,
            failure_code=None,
        )
        source.current_segment = segment

        async def stop(*, cancel):
            await source.segment_finished(
                segment, "cancelled" if cancel else "completed", None
            )
            segment.closed_event.set()

        segment.stop = stop
        ending_task = asyncio.create_task(source._end_segment(ending, accepted))
        try:
            if outcome != "cancel":
                await asyncio.sleep(0.005)
                # Stopping the segment here removes its only remaining application evidence.
                assert (
                    journal.snapshot["operations"][update["request_id"]]["state"]
                    == "accepted"
                )
                assert journal.snapshot["segment"]["state"] == "draining"
                with pytest.raises(RenderError) as denied:
                    await journal.prompt(
                        command(
                            "prompt.replace",
                            "after-end",
                            **ids,
                            prompt="extra",
                            expected_prompt_version=2,
                            prompt_version=3,
                            effective={"source_frame_index": 1},
                        )
                    )
                assert denied.value.code == "MODEL_UNAVAILABLE"
                if outcome == "applied":
                    await journal.prompt_applied(
                        ids["segment_id"], 2, dict(first_output_frame=8, num_frames=1)
                    )
                    segment.pending_prompt = None
                elif outcome == "failed":
                    await journal.prompt_failed(
                        ids["segment_id"],
                        2,
                        never_applied=True,
                        code="CONDITION_INVALID",
                    )
                    segment.pending_prompt = None
            await asyncio.wait_for(ending_task, 0.2)
            result = journal.snapshot["operations"]["last-update"]
            if outcome in {"applied", "failed"}:
                assert result["state"] == outcome
            else:
                assert result["state"] == "reconciling"
                assert result["code"] == "SEGMENT_ENDED_UNCONFIRMED"
                assert not result.get("never_applied")
            assert journal.snapshot["operations"]["end-pending"]["state"] == "completed"
        finally:
            ending_task.cancel()
            await asyncio.gather(ending_task, return_exceptions=True)

    run(scenario())


def test_late_model_readiness_does_not_reopen_draining_admission():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        journal, ids = await started()
        await journal.end(
            command("render.end", "ending", **ids, expected_render_revision=1)
        )
        source = object.__new__(SourceSession)
        source.journal = journal
        source._status = lambda *a, **k: asyncio.sleep(0)
        segment = SimpleNamespace(
            segment_id=ids["segment_id"], resource_state="reserved"
        )
        for phase in ("initializing", "streaming"):
            await source.segment_status(segment, phase, phase="first_decodable")
            assert journal.snapshot["segment"]["state"] == "draining"
        with pytest.raises(RenderError) as denied:
            await journal.prompt(
                command(
                    "prompt.replace",
                    "late",
                    **ids,
                    prompt="late",
                    expected_prompt_version=1,
                    prompt_version=2,
                    effective={"source_frame_index": 0},
                )
            )
        assert denied.value.code == "MODEL_UNAVAILABLE"

    run(scenario())


@pytest.mark.parametrize("mode", ["cancel", "drain", "drain-ready"])
def test_end_dispatch_lock_cannot_block_cancel_or_exceed_original_deadline(mode):
    from types import SimpleNamespace

    import msgspec

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        journal, ids = await started()
        before = await journal.read()
        source = object.__new__(SourceSession)
        source.journal = journal
        source.control_tokens = 4.0
        source.control_token_time = asyncio.get_running_loop().time()
        source.binding_ended = False
        source.outgoing = asyncio.Queue()
        source.source_identity = lambda: {}
        source.config = SimpleNamespace(drain_seconds=0.01)
        source.segment_lock = asyncio.Lock()
        source.control_tasks = set()
        source.current_model_stream = "model"
        lock = asyncio.Lock()
        await lock.acquire()
        segment = SimpleNamespace(
            segment_id=ids["segment_id"],
            source_send_lock=lock,
            closed_event=asyncio.Event(),
            resource_state="released",
            pending_prompt=None,
        )
        source.current_segment = segment

        async def stop(*, cancel):
            assert cancel == (mode == "cancel")
            segment.closed_event.set()

        segment.stop = stop
        ending = command(
            "render.end",
            "held-lock",
            **ids,
            expected_render_revision=1,
            mode="cancel" if mode == "cancel" else "drain",
        )
        try:
            await asyncio.wait_for(source.command(ending), 0.2)
            if mode == "drain-ready":
                # A blocked normal end must still admit exactly once after dispatch.
                await source.command(ending)
                lock.release()
            response = msgspec.msgpack.decode(
                await asyncio.wait_for(source.outgoing.get(), 10.2)
            )
            if mode in {"cancel", "drain-ready"}:
                assert response["type"] == "operation.result"
                await asyncio.wait_for(segment.closed_event.wait(), 0.2)
                assert (
                    journal.snapshot["operations"]["held-lock"]["mode"]
                    == ending["mode"]
                )
                assert journal.snapshot["render_revision"] == 2
            else:
                assert response["type"] == "operation.rejected"
                assert response["code"] == "RENDER_TRANSITION_IN_PROGRESS"
                assert await journal.read() == before
                assert not segment.closed_event.is_set()
        finally:
            if lock.locked():
                lock.release()
            for task in list(source.control_tasks):
                task.cancel()
            await asyncio.gather(*source.control_tasks, return_exceptions=True)

    run(scenario())


@pytest.mark.parametrize("followup", ["cancel", "source.close"])
def test_ordered_room_queue_can_cancel_or_close_while_end_waits_for_dispatch(followup):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import msgspec

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        RoomPipeline,
    )

    async def scenario():
        journal, ids = await started()
        source = object.__new__(SourceSession)
        source.journal = journal
        source.control_tokens = 4.0
        source.control_token_time = asyncio.get_running_loop().time()
        source.binding_ended = False
        source.outgoing = asyncio.Queue()
        source.source_identity = lambda: {}
        source.config = SimpleNamespace(drain_seconds=0.01)
        source.segment_lock = asyncio.Lock()
        source.control_tasks = set()
        source.current_model_stream = "model"
        source.ended = False
        source.finish_requested = asyncio.Event()
        engine_finished = asyncio.Event()

        async def finish_engine():
            engine_finished.set()

        source.engine = SimpleNamespace(finish=finish_engine)
        lock = asyncio.Lock()
        await lock.acquire()
        segment = SimpleNamespace(
            segment_id=ids["segment_id"],
            source_send_lock=lock,
            closed_event=asyncio.Event(),
            resource_state="released",
        )
        source.current_segment = segment

        async def stop(*, cancel):
            assert cancel
            segment.closed_event.set()
            lock.release()

        segment.stop = stop
        pipeline = object.__new__(RoomPipeline)
        pipeline.session = source
        pipeline.task = None
        pipeline.close = AsyncMock()
        source.session_id = "source"
        source.host = SimpleNamespace(
            sessions={"source": pipeline},
            runtime=SimpleNamespace(
                _observe_player_cleanup=lambda task: (
                    task.exception() if not task.cancelled() else None
                )
            ),
        )
        pipeline.generation = 1
        pipeline.command_queue = asyncio.Queue()
        pipeline.command_bytes = 0
        ending = command("render.end", "waiting-end", **ids, expected_render_revision=1)
        subsequent = (
            command(
                "render.end",
                "cancel-now",
                **ids,
                expected_render_revision=1,
                mode="cancel",
            )
            if followup == "cancel"
            else command("source.close", "close-now")
        )
        for body in (
            ending,
            ending,  # The retry shares the pending wait, without another task.
            {**ending, "mode": "cancel"},  # Same ID, different intent.
            {**ending, "request_id": "second-end"},
            subsequent,
        ):
            message = dict(type="command", **body)
            wire = msgspec.msgpack.encode(message)
            pipeline.command_bytes += len(wire)
            pipeline.command_queue.put_nowait((1, wire, message))
        worker = asyncio.create_task(pipeline._commands())
        try:
            await asyncio.wait_for(
                (
                    segment.closed_event if followup == "cancel" else engine_finished
                ).wait(),
                0.5,
            )
            state = await journal.read()
            assert "waiting-end" not in state["operations"]
            assert pipeline.command_queue.empty() and pipeline.command_bytes == 0
            rejections = []
            while not source.outgoing.empty():
                reply = msgspec.msgpack.decode(source.outgoing.get_nowait())
                if reply["type"] == "operation.rejected":
                    rejections.append((reply["request_id"], reply["code"]))
            assert ("waiting-end", "REQUEST_ID_CONFLICT") in rejections
            assert ("second-end", "RENDER_TRANSITION_IN_PROGRESS") in rejections
            if followup == "cancel":
                assert state["operations"]["cancel-now"]["mode"] == "cancel"
                async with asyncio.timeout(0.5):
                    while ("waiting-end", "REVISION_CONFLICT") not in rejections:
                        reply = msgspec.msgpack.decode(await source.outgoing.get())
                        if reply["type"] == "operation.rejected":
                            rejections.append((reply["request_id"], reply["code"]))
                await asyncio.gather(*source.control_tasks)
                assert source.outgoing.empty()  # No second task/reply for the retry.
            else:
                assert state["operations"]["close-now"]["state"] == "accepted"
                assert source.ended and source.finish_requested.is_set()
        finally:
            worker.cancel()
            tasks = list(source.control_tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(worker, *tasks, return_exceptions=True)
            if source.player_finish_task is not None:
                await source.player_finish_task
            if lock.locked():
                lock.release()

    run(scenario())


@pytest.mark.parametrize("phase", ["accepted", "applied"])
def test_room_process_sigkill_preserves_journal_and_fences_replacement(tmp_path, phase):
    """A killed Room is lost; persisted acceptance must not imply model replay."""
    import json
    import os
    import selectors
    import signal
    import subprocess
    import sys
    import urllib.error
    import urllib.request
    from pathlib import Path

    helper = Path(__file__).with_name("whitebox_render_crash_fixture.py")
    processes = []

    def start(mode):
        process = subprocess.Popen(
            [sys.executable, str(helper), str(tmp_path), phase, mode],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=20), "Room process startup timed out"
        line = process.stdout.readline()
        assert line, process.stderr.read()
        return process, json.loads(line)

    def status(port, request_id, authorized=True):
        url = f"http://127.0.0.1:{port}/v1/whitebox/render/sources/{'a' * 32}?operation_request_id={request_id}"
        headers = (
            {"Authorization": "Bearer disposable-process-test-capability"}
            if authorized
            else {}
        )
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=5
        ) as response:
            assert response.headers["Cache-Control"] == "no-store"
            return json.load(response)

    try:
        first, ready = start("seed")
        original = ready["operation"]
        before = status(ready["port"], original["request_id"])
        assert before["operation"]["state"] == phase
        os.kill(first.pid, signal.SIGKILL)
        assert first.wait(timeout=5) == -signal.SIGKILL
        second, recovered = start("recover")
        assert second.pid != first.pid
        assert recovered["fence"] == recovered["resume"] == "ROOM_SESSION_LOST"
        assert recovered["active_sources"] == 0
        after = status(recovered["port"], original["request_id"])
        assert after["source_state"] == "lost"
        assert after["owner_epoch"] == "original-owner"
        assert after["source_epoch"] == before["source_epoch"]
        assert after["segment"]["state"] == "failed"
        assert after["segment"]["resource_state"] == "release_unknown"
        assert after["operation"] == before["operation"]
        assert after["operation"]["prompt_version"] == 1
        assert after["segment"]["applied_version"] == (1 if phase == "applied" else 0)
        with pytest.raises(urllib.error.HTTPError) as denied:
            status(recovered["port"], original["request_id"], authorized=False)
        assert denied.value.code == 404
        # Querying or a rejected resume cannot allocate a new request/version.
        assert status(recovered["port"], original["request_id"]) == after
        (tmp_path / "process-recovery-evidence.json").write_text(
            json.dumps(
                dict(
                    scope="real RenderHost processes, durable SQLite and production HTTP route; deterministic model records, no GPU lease",
                    phase=phase,
                    first_pid=first.pid,
                    replacement_pid=second.pid,
                    first_exit=first.returncode,
                    replacement=recovered,
                    before=before,
                    after=after,
                ),
                indent=2,
            )
        )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()


def test_cancel_interrupts_an_admitted_end_waiting_for_condition_output():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        journal, ids = await started()
        await journal.prompt(
            command(
                "prompt.replace",
                "unsettled-prompt",
                **ids,
                prompt="last",
                expected_prompt_version=1,
                prompt_version=2,
                effective={"source_frame_index": 0},
            )
        )
        source = object.__new__(SourceSession)
        source.journal = journal
        source.segment_lock = asyncio.Lock()
        source.outgoing = asyncio.Queue()
        source.control_tasks = set()
        source.config = SimpleNamespace(drain_seconds=5)
        source.source_identity = lambda: {}
        source._status = lambda *args, **kwargs: asyncio.sleep(0)
        segment = SimpleNamespace(
            segment_id=ids["segment_id"],
            pending_prompt=journal.snapshot["segment"]["prompt_queue"][0],
            closed_event=asyncio.Event(),
            resource_state="reserved",
            failure_code=None,
            cancel_requested=False,
        )
        source.current_segment = segment
        stopped = []

        async def stop(*, cancel):
            stopped.append(cancel)
            segment.resource_state = "released"
            segment.closed_event.set()

        segment.stop = stop
        ending = command("render.end", "drain", **ids, expected_render_revision=1)
        admitted = await journal.end(ending)
        ending_task = asyncio.create_task(source._end_segment(ending, admitted))
        try:

            async def entered_drain():
                while not source.segment_lock.locked():
                    await asyncio.sleep(0)

            await asyncio.wait_for(entered_drain(), 0.2)
            assert journal.snapshot["segment"]["state"] == "draining"
            cancel = command(
                "render.end",
                "cancel-drain",
                **ids,
                expected_render_revision=2,
                mode="cancel",
            )
            await source._admit_end(
                cancel, asyncio.get_running_loop().time() + 15, None
            )
            await asyncio.wait_for(segment.closed_event.wait(), 0.2)
            await asyncio.gather(ending_task, *tuple(source.control_tasks))
            assert stopped == [True]
            assert (
                journal.snapshot["operations"]["cancel-drain"]["state"] == "completed"
            )
            assert (
                journal.snapshot["operations"]["unsettled-prompt"]["state"]
                == "accepted"
            )
        finally:
            ending_task.cancel()
            tasks = [ending_task, *source.control_tasks]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    run(scenario())
