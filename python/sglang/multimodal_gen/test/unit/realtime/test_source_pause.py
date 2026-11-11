# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace as NS

import msgspec
import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.player_idle import (
    PlayerInputActivity,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
    RenderJournal,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore
from sglang.multimodal_gen.runtime.realtime.whitebox.source_pause import SourcePause

A = "6bb6e425-d24d-48f6-a128-85a94ac98e20"
B = "6040368c-c4a4-492d-a0a2-d4c69c16d93d"


async def fixture():
    source = object.__new__(SourceSession)
    source.session_id, source.source_epoch = "source", "epoch"
    source.journal = RenderJournal(
        MemoryRoomStore(),
        user_id="user",
        source_session_id="source",
        source_epoch="epoch",
        owner_epoch="owner",
    )
    await source.journal.create()
    activity = PlayerInputActivity()
    activity.start(1)
    source.host = NS(runtime=NS(players={"view": {"input_activity": activity}}))
    source.binding_ended = source.close_requested = source.closing = False
    source.inference = NS(manual=False)
    source.control_prefill = source.member_prefill = None
    source.startup = None
    source.outgoing = asyncio.Queue()
    source.control_tasks = set()
    source.messages, source.statuses = [], []
    control = NS(calls=[], held=None, gate=None, fail_resume=False, lost_hold=False)

    async def capture(action, request_id):
        control.calls.append((action, request_id))
        if action == "hold":
            if control.held is not None:
                raise ValueError("Capture busy")
            control.held = request_id
            if control.gate is not None:
                await control.gate.wait()
            if control.lost_hold:
                raise TimeoutError("Reply lost after source stopped")
        else:
            if control.fail_resume:
                raise TimeoutError("Resume unconfirmed")
            if control.held != request_id:
                raise ValueError("Wrong capture owner")
            control.held = None
        return dict(
            requestId=request_id,
            state="held" if action == "hold" else "running",
            frame={},
        )

    source.engine = NS(
        stream_id="view",
        metadata={"mode": "worldkit_cli", "source_codec": "rgb24", "max_players": 1},
        shared_world=False,
        capture_control=capture,
    )
    source.source_pause = SourcePause(source)

    async def operation(message):
        source.messages.append(message)

    async def status(*_):
        source.statuses.append(source.source_pause.snapshot())

    source.operation_message, source.status_snapshot = operation, status
    return source, control, activity


async def send(source, op, pause_id=A, request_id=None, **extra):
    command = dict(
        type="command",
        version=2,
        op=op,
        pause_id=pause_id,
        request_id=request_id or f"{op}:{len(source.messages)}",
        source_session_id="source",
        source_epoch="epoch",
    )
    command.update(extra)
    await source.command(command)
    await asyncio.sleep(0)
    return command


async def settled(source):
    while source.control_tasks:
        await asyncio.gather(*list(source.control_tasks))
        await asyncio.sleep(0)


def test_source_pause_real_command_journal_idempotency_and_no_input_fabrication():
    async def run():
        source, control, activity = await fixture()
        first = await send(source, "source.pause")
        await settled(source)
        assert control.held == A and activity.waiting == A
        assert source.messages[-1]["state"] == "complete"
        assert source.messages[-1]["source_pause"] == dict(
            pause_id=A, state="paused", maximum_pause_ms=300000
        )
        await source.command(first)
        await send(source, "source.pause", request_id="same-id-new-request")
        await settled(source)
        assert control.calls == [("hold", A)]
        await send(source, "source.resume")
        await settled(source)
        assert control.held is None and activity.waiting is None
        assert source.statuses[-1]["state"] == "running"
        await send(source, "source.resume", request_id="resume-again")
        await send(source, "source.pause", request_id="old-pause-id")
        await settled(source)
        assert control.calls == [("hold", A), ("resume", A)]
        # Operation history remains an original receipt; status is current truth.
        await source.command(first)
        assert source.statuses[-1]["state"] == "running"

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,value",
    [
        ("shared_world", True),
        ("source_codec", "vp8"),
        ("mode", "recorded"),
        ("max_players", 2),
        ("max_players", None),
    ],
)
def test_unsupported_shared_or_non_native_source_never_calls_capture(field, value):
    async def run():
        source, control, _ = await fixture()
        if field == "shared_world":
            source.engine.shared_world = value
        else:
            source.engine.metadata[field] = value
        assert not source.source_pause.supported
        await send(source, "source.pause")
        assert (
            msgspec.msgpack.decode(source.outgoing.get_nowait())["code"]
            == "CAPABILITY_UNAVAILABLE"
        )
        assert not control.calls

    asyncio.run(run())


def test_scope_uuid_payload_conflict_and_member_prefill_rejected_before_effect():
    async def run():
        source, control, _ = await fixture()
        for extra in [
            dict(source_epoch="foreign"),
            dict(source_session_id="foreign"),
            dict(pause_id="not-a-uuid"),
            dict(unexpected=True),
            dict(version=True),
        ]:
            await send(source, "source.pause", **extra)
            assert (
                msgspec.msgpack.decode(source.outgoing.get_nowait())["type"]
                == "operation.rejected"
            )
        assert not control.calls
        source.member_prefill = NS()
        await send(source, "source.pause")
        assert (
            msgspec.msgpack.decode(source.outgoing.get_nowait())["code"]
            == "CAPABILITY_UNAVAILABLE"
        )
        source.member_prefill = None
        command = await send(source, "source.pause", request_id="fixed-request")
        await settled(source)
        await source.command({**command, "pause_id": B})
        assert (
            msgspec.msgpack.decode(source.outgoing.get_nowait())["code"]
            == "REQUEST_ID_CONFLICT"
        )
        assert control.calls == [("hold", A)]
        await source.source_pause.close()

    asyncio.run(run())


def test_prefill_other_pause_and_old_resume_cannot_take_or_release_foreign_hold():
    async def run():
        source, control, _ = await fixture()
        source.control_prefill = NS(resumed=False)
        control.held = "prefill-id"
        await send(source, "source.pause")
        await settled(source)
        assert (
            msgspec.msgpack.decode(source.outgoing.get_nowait())["code"]
            == "SOURCE_PAUSE_BUSY"
            and not control.calls
        )
        await send(source, "source.resume")
        await settled(source)
        assert source.messages[-1]["source_pause"]["state"] == "running"
        assert control.held == "prefill-id" and not control.calls
        source.control_prefill.resumed = True
        control.held = None
        await send(source, "source.pause", pause_id=B)
        await settled(source)
        await send(source, "source.pause", request_id="busy-pause")
        await settled(source)
        assert source.messages[-1]["code"] == "SOURCE_PAUSE_BUSY"
        await send(source, "source.resume", request_id="old-resume")
        await settled(source)
        assert control.held == B and source.statuses[-1]["pause_id"] == B
        await source.source_pause.close()

    asyncio.run(run())


@pytest.mark.parametrize("cancel", ["resume", "detach", "close", "source-change"])
def test_late_hold_confirmation_cannot_resurrect_pause_after_owner_invalidated(cancel):
    async def run():
        source, control, activity = await fixture()
        control.gate = asyncio.Event()
        await send(source, "source.pause")
        assert control.calls == [("hold", A)] and activity.waiting is None
        closer = None
        if cancel == "resume":
            await send(source, "source.resume")
        elif cancel == "detach":
            source.source_pause.invalidate()
        elif cancel == "close":
            closer = asyncio.create_task(source.source_pause.close())
            await asyncio.sleep(0)
        else:
            source.engine = NS()
        control.gate.set()
        await settled(source)
        if closer:
            await closer
        assert control.calls == [("hold", A), ("resume", A)]
        assert control.held is None and activity.waiting is None
        assert not any(
            m.get("source_pause", {}).get("state") == "paused" for m in source.messages
        )

    asyncio.run(run())


def test_hold_lost_reply_and_resume_failure_are_unknown_not_success_and_can_retry_same_owner():
    async def run():
        source, control, activity = await fixture()
        control.lost_hold = control.fail_resume = True
        await send(source, "source.pause")
        await settled(source)
        assert source.messages[-1]["code"] == "SOURCE_PAUSE_UNCONFIRMED"
        assert source.source_pause.snapshot() is None and activity.waiting is None
        assert control.held == A
        control.fail_resume = False
        await send(source, "source.resume")
        await settled(source)
        assert control.held is None and source.statuses[-1]["state"] == "running"

    asyncio.run(run())


def test_expiry_and_disconnect_release_only_the_original_pause_and_clear_idle_wait():
    async def run():
        source, control, activity = await fixture()
        await send(source, "source.pause")
        await settled(source)
        entry = source.source_pause.current
        assert 299 <= entry.timer.when() - asyncio.get_running_loop().time() <= 300
        source.source_pause.invalidate(entry)
        assert activity.waiting is None and source.source_pause.snapshot() is None
        await settled(source)
        assert control.held is None and source.statuses[-1]["state"] == "running"
        await send(source, "source.pause", pause_id=B)
        await settled(source)
        source.source_pause.invalidate(entry)  # old timer must not release B
        await settled(source)
        assert control.held == B
        await source.source_pause.close()

    asyncio.run(run())


def test_source_pause_blocks_prefill_prepare_without_calling_its_owner():
    async def run():
        source, control, _ = await fixture()
        await send(source, "source.pause")
        await settled(source)
        await source.command(
            dict(
                type="command",
                op="render.prepare",
                request_id="prepare",
                source_session_id="source",
                source_epoch="epoch",
            )
        )
        assert (
            msgspec.msgpack.decode(source.outgoing.get_nowait())["code"]
            == "SOURCE_PAUSE_BUSY"
        )
        await source.source_pause.close()

    asyncio.run(run())


def test_actual_pipeline_detach_and_connection_replacement_invalidate_pending_owner(
    monkeypatch,
):
    from sglang.multimodal_gen.runtime.realtime.whitebox import render_host

    async def run():
        source, control, activity = await fixture()
        pipeline = object.__new__(render_host.RenderPipeline)
        pipeline.session = source
        pipeline.host = NS(grace_seconds=60)
        pipeline.retention = None
        sub = NS(stop=lambda *_: None)
        pipeline.subscription = sub
        await send(source, "source.pause")
        await settled(source)
        pipeline.detach(sub)
        assert pipeline.subscription is None and activity.waiting is None
        pipeline.retention.cancel()
        await asyncio.gather(pipeline.retention, return_exceptions=True)
        await settled(source)
        assert control.held is None
        await send(source, "source.pause", pause_id=B)
        await settled(source)
        pipeline.generation = 2
        pipeline.resolve_area_offer = lambda *_: None
        monkeypatch.setattr(render_host, "RenderSubscription", lambda *_: sub)
        assert pipeline.create_subscription() is sub
        await settled(source)
        assert control.held is None and source.statuses[-1]["state"] == "running"

    asyncio.run(run())


def test_failed_confirmed_resume_is_not_reported_running_and_does_not_suppress_idle_forever():
    async def run():
        source, control, activity = await fixture()
        await send(source, "source.pause")
        await settled(source)
        assert activity.waiting == A
        control.fail_resume = True
        await send(source, "source.resume")
        await settled(source)
        assert source.messages[-1]["state"] == "failed"
        assert source.messages[-1]["code"] == "SOURCE_PAUSE_UNCONFIRMED"
        assert source.statuses[-1] is None and control.held == A
        assert activity.waiting is None
        control.fail_resume = False
        await send(source, "source.resume", request_id="resume-reconciled")
        await settled(source)
        assert source.statuses[-1]["state"] == "running" and control.held is None

    asyncio.run(run())


@pytest.mark.parametrize("occupied", ["pending_frame", "active", "recovery", "failure"])
def test_pause_rejects_complete_original_prefill_occupied_state(occupied):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
    )

    async def run():
        source, control, _ = await fixture()
        pending = asyncio.get_running_loop().create_future()
        source.control_prefill = NS(
            resumed=True,
            **{
                occupied: (
                    RenderError("RELEASE_UNKNOWN") if occupied == "failure" else pending
                )
            },
        )
        await send(source, "source.pause")
        assert (
            msgspec.msgpack.decode(source.outgoing.get_nowait())["code"]
            == "SOURCE_PAUSE_BUSY"
        )
        assert control.calls == [] and not source.source_pause.active
        pending.cancel()

    asyncio.run(run())


@pytest.mark.parametrize(
    "first,delay",
    [
        ("pause", "journal"),
        ("pause", "operation"),
        ("prefill", "read"),
        ("prefill", "journal"),
        ("prefill", "operation"),
    ],
)
def test_real_prefill_and_pause_admission_is_atomic_across_journal_and_reply_awaits(
    first, delay
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import EngineFrame
    from sglang.multimodal_gen.runtime.realtime.whitebox.control_prefill import (
        ControlPrefill,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
    )

    async def run():
        source, control, _ = await fixture()
        prefill = source.control_prefill = ControlPrefill(source)
        source.latest = NS(version=1, value=None)
        source.config = NS(startup_seconds=1)
        header = dict(
            sessionId="cli", epoch="cli-epoch", mediaGeneration=1, outputFrameId=0
        )
        frame = EngineFrame(
            0, 0, b"pixels", {}, provenance={"cli_frame_header": header}
        )
        source.retained = NS(frames=[(1, frame, 0, None)])
        original_capture = source.engine.capture_control

        async def capture(action, request_id):
            return {**await original_capture(action, request_id), "frame": header}

        source.engine.capture_control = capture
        gate, entered = asyncio.Event(), asyncio.Event()
        blocked = False
        target = "source.pause" if first == "pause" else "render.prepare"

        async def wait_once():
            nonlocal blocked
            if not blocked:
                blocked = True
                entered.set()
                await gate.wait()

        if delay == "journal":
            original = source.journal.command

            async def journal(command, *args, **kwargs):
                if command.get("op") == target:
                    await wait_once()
                return await original(command, *args, **kwargs)

            source.journal.command = journal
        elif delay == "operation":
            original = source.operation_message

            async def operation(value):
                if value.get("op") == target:
                    await wait_once()
                return await original(value)

            source.operation_message = operation
        else:
            original = source.journal.read

            async def read():
                await wait_once()
                return await original()

            source.journal.read = read
        prepare = dict(
            type="command",
            version=2,
            op="render.prepare",
            request_id="original-prefill",
            source_session_id="source",
            source_epoch="epoch",
            expected_render_revision=0,
        )
        if first == "pause":
            task = asyncio.create_task(send(source, "source.pause"))
            await entered.wait()
            with pytest.raises(RenderError, match="already being prepared or paused"):
                await prefill.prepare(prepare)
            assert not control.calls and source.source_pause.active
        else:
            task = asyncio.create_task(prefill.prepare(prepare))
            await entered.wait()
            await send(source, "source.pause")
            assert (
                msgspec.msgpack.decode(source.outgoing.get_nowait())["code"]
                == "SOURCE_PAUSE_BUSY"
            )
            assert (
                not control.calls and source.source_pause.prefill_admission is not None
            )
        gate.set()
        await task
        await settled(source)
        assert (
            source.source_pause.prefill_admission is None
            and not source.source_pause.pause_admissions
        )
        expected = A if first == "pause" else "original-prefill"
        assert control.calls == [("hold", expected)] and control.held == expected
        if first == "pause":
            await source.source_pause.close()
        else:
            await prefill.resume("original-prefill")
        assert (
            control.calls == [("hold", expected), ("resume", expected)]
            and control.held is None
        )

    asyncio.run(run())


def test_failed_admission_releases_only_its_reservation_for_following_owner():
    from sglang.multimodal_gen.runtime.realtime.whitebox.control_prefill import (
        ControlPrefill,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
    )

    async def run():
        source, control, _ = await fixture()
        original = source.journal.command

        async def failed(*_, **__):
            raise RenderError("CONTROL_UNAVAILABLE")

        source.journal.command = failed
        await send(source, "source.pause")
        assert (
            msgspec.msgpack.decode(source.outgoing.get_nowait())["code"]
            == "CONTROL_UNAVAILABLE"
        )
        assert not source.source_pause.active and not control.calls
        prefill = source.control_prefill = ControlPrefill(source)
        with pytest.raises(RenderError):
            await prefill.prepare(
                dict(
                    request_id="failed-prepare",
                    expected_render_revision=0,
                    source_session_id="source",
                    source_epoch="epoch",
                )
            )
        assert source.source_pause.prefill_admission is None
        source.journal.command = original
        # Prefill's original pending_frame is conservative occupied state until
        # its own cleanup; freeing the reservation does not steal that owner.
        assert prefill.pending_frame is not None
        prefill.pending_frame.cancel()

    asyncio.run(run())


def test_duplicate_receipt_remains_readable_while_prefill_busy_and_resume_prevents_late_admission():
    async def run():
        source, control, _ = await fixture()
        first = await send(source, "source.pause")
        await settled(source)
        await send(source, "source.resume")
        await settled(source)
        source.control_prefill = NS(resumed=False)
        await source.command(first)
        assert source.messages[-1]["duplicate"]
        assert source.statuses[-1]["state"] == "running"
        assert source.outgoing.empty()
        source.control_prefill.resumed = True
        await send(
            source, "source.resume", pause_id=B, request_id="cancel-before-admission"
        )
        await settled(source)
        await send(source, "source.pause", pause_id=B, request_id="late-admission")
        await settled(source)
        assert control.calls == [("hold", A), ("resume", A)]
        assert source.messages[-1]["source_pause"]["state"] == "running"

    asyncio.run(run())
