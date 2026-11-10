# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import time

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.player_idle import (
    PlayerIdlePolicy,
    PlayerInputActivity,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.room_mock import shared_mock_room
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
    new_id,
)


def state(sequence, **fields):
    return dict(type="input.state", input=dict(sequence=sequence, **fields))


def receipt(sequence, client="a"):
    return dict(type="input.receipt", clientSequence=sequence, clientId=client)


def test_input_before_first_frame_cannot_arm_or_backdate_idle_timer():
    activity = PlayerInputActivity()
    activity.touch(10)  # Legacy input path.
    activity.observe("a", "to-sdk", state(1, heldKeys=["KeyW"]), 20)
    activity.observe("a", "from-sdk", receipt(1), 21)
    activity.observe("a", "to-sdk", {"type": "input.action", "actionId": "a"}, 30)
    activity.observe(
        "a",
        "from-sdk",
        {"type": "action.receipt", "actionId": "a", "status": "completed"},
        31,
    )
    assert activity.last_input is None
    assert not activity.is_expired(1000, 60)

    activity.observe("a", "to-sdk", state(2, heldKeys=["KeyW"]), 999)
    activity.start(1000)  # First valid whitebox frame.
    activity.observe("a", "from-sdk", receipt(2), 1001)
    assert activity.last_input == 1000
    assert not activity.is_expired(1059.9, 60)
    assert activity.is_expired(1060, 60)


def test_only_valid_source_frame_arms_idle_timer(tmp_path):
    from types import SimpleNamespace as NS

    import pytest

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import EngineFrame

    async def run():
        spec = VideoSpec(32, 32)
        queue = asyncio.Queue()

        async def frames():
            while True:
                yield await queue.get()

        activity = PlayerInputActivity()
        entry = dict(
            spec=spec,
            read_lock=asyncio.Lock(),
            pending={},
            iterator=frames(),
            next=0,
            input_activity=activity,
            trace_id="first-frame",
            game_id="g",
            engine=NS(renderer=NS(metadata={"mode": "browser"})),
        )
        runtime = NS(
            players={"s": entry},
            player=lambda *args, **kwargs: entry,
            pending_frames=96,
            rooms={"g": {"log": NS(sequence=0, start=time.monotonic_ns())}},
        )
        pending = asyncio.create_task(RoomRuntime.next_frame(runtime, "s"))
        try:
            await asyncio.sleep(0)
            assert not activity.is_expired(time.monotonic() + 61, 60)
            await queue.put(EngineFrame(0, 0, b"invalid", {}))
            with pytest.raises(ValueError, match="invalid video frame"):
                await pending
            assert activity.last_input is None
            await queue.put(EngineFrame(0, 0, bytes(spec.frame_bytes), {}))
            assert (await RoomRuntime.next_frame(runtime, "s"))["index"] == 0
            first = activity.last_input
            assert first is not None
            await queue.put(EngineFrame(1, spec.pts(1), bytes(spec.frame_bytes), {}))
            await RoomRuntime.next_frame(runtime, "s")
            assert activity.last_input == first
            assert not activity.is_expired(first + 59.9, 60)
            assert activity.is_expired(first + 60, 60)
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await entry["iterator"].aclose()

    asyncio.run(run())


def test_startup_and_neutral_heartbeats_do_not_extend_play():
    activity = PlayerInputActivity()
    assert not activity.is_expired(1000, 60)  # Not playable yet.
    activity.start(1000)
    for seq in range(60):
        activity.start(1000 + seq)  # Subsequent video frames do not renew.
        activity.observe("a", "to-sdk", state(seq, heldKeys=[]), 1000 + seq)
        activity.observe("a", "from-sdk", receipt(seq), 1000 + seq)
        activity.observe("a", "to-sdk", {"type": "ping"}, 1000 + seq)
        activity.observe("a", "from-sdk", {"type": "prefill.memory_ready"}, 1000 + seq)
    assert not activity.is_expired(1059.9, 60)
    assert activity.is_expired(1060, 60)
    activity.touch(1061)
    assert activity.is_expired(1061, 60)  # Closing is not resurrected by late input.


def test_foreground_prefill_resets_full_idle_budget_only_once():
    activity = PlayerInputActivity()
    activity.start(100)
    activity.pause("ctx")
    assert not activity.is_expired(250, 60)
    activity.resume("other", 250)
    assert activity.waiting == "ctx"
    activity.resume("ctx", 250)
    assert not activity.is_expired(309.9, 60)
    activity.resume("ctx", 309)  # Duplicate completion is not new activity.
    assert activity.is_expired(310, 60)
    activity.pause("next")
    activity.resume("next", 400)
    assert activity.is_expired(400, 60)


def test_prefill_completion_before_first_source_frame_does_not_arm_idle():
    activity = PlayerInputActivity()
    activity.pause("ctx")
    activity.resume("ctx", 100)
    assert activity.last_input is None
    activity.start(200)
    assert not activity.is_expired(259.9, 60)
    assert activity.is_expired(260, 60)


def test_held_input_and_release_but_not_empty_renewals_count():
    activity = PlayerInputActivity()
    activity.start(0)
    for seq in range(1, 150):
        activity.observe("a", "to-sdk", state(seq, heldKeys=["KeyW"]), seq)
        activity.observe("a", "from-sdk", receipt(seq), seq + 0.1)
        assert not activity.is_expired(seq + 0.2, 60)
    activity.observe("a", "to-sdk", state(150, heldKeys=[]), 150)
    activity.observe("a", "from-sdk", receipt(150), 150.1)
    for seq in range(151, 210):
        activity.observe("a", "to-sdk", state(seq, heldKeys=[]), seq)
        activity.observe("a", "from-sdk", receipt(seq), seq + 0.1)
    assert activity.last_input == 150
    assert activity.is_expired(210, 60)


def test_receipts_must_match_original_input_and_use_receive_time():
    activity = PlayerInputActivity()
    activity.start(0)
    activity.observe(
        "a", "to-sdk", state(1, keyEdges=[{"kind": "down", "code": "KeyW"}]), 10
    )
    activity.observe("a", "from-sdk", receipt(1, client="b"), 11)
    assert activity.last_input == 0
    activity.observe("a", "from-sdk", receipt(1), 20)
    assert activity.last_input == 10
    activity.observe("a", "from-sdk", receipt(1), 69)
    assert activity.is_expired(70, 60)


def test_ui_action_success_only_and_camera_input():
    activity = PlayerInputActivity()
    activity.start(0)
    for status, at in [("failed", 20), ("completed", 40)]:
        message = dict(type="input.action", actionId=str(at))
        activity.observe("a", "to-sdk", message, at)
        activity.observe(
            "a",
            "from-sdk",
            dict(type="action.receipt", actionId=str(at), status=status),
            at + 1,
        )
    assert activity.last_input == 40
    activity.observe("a", "to-sdk", state(0, yawDeltaRadians=0.02), 50)
    activity.observe("a", "from-sdk", receipt(0), 51)
    assert activity.last_input == 50


def test_policy_hot_reload_disable_and_last_good_value(tmp_path):
    path = tmp_path / "player-idle.json"
    policy = PlayerIdlePolicy(path)
    assert policy.timeout(0) == 60
    path.write_text(json.dumps({"player_idle_timeout_seconds": 120}))
    assert policy.timeout(4) == 60
    assert policy.timeout(5) == 120
    for value, at in [(True, 10), (-1, 15), ("60", 20), (float("nan"), 25)]:
        path.write_text(json.dumps({"player_idle_timeout_seconds": value}))
        assert policy.timeout(at) == 120
    path.write_text('{"player_idle_timeout_seconds":0}')
    assert policy.timeout(30) == 0
    activity = PlayerInputActivity()
    activity.start(0)
    assert not activity.is_expired(1000, policy.seconds)
    path.unlink()
    assert policy.timeout(35) == 0


def test_inactive_player_leaves_without_ending_active_peer(tmp_path):
    async def run():
        directory = RoomDirectory(MemoryRoomStore(), {"a": "http://a"})
        game = new_id()
        await directory.resolve(game, "world", create=True)
        runtime = RoomRuntime(
            directory=directory,
            owner_id="a",
            driver_factory=shared_mock_room,
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )

        async def join():
            return await runtime.join(
                dict(
                    game_id=game,
                    world_id="world",
                    player_id=new_id(),
                    stream_id=new_id(),
                    session_id=new_id(),
                    user_id=new_id(),
                    trace_id="idle-test",
                    spec={"width": 32, "height": 32, "fps": 24},
                )
            )

        a, b = await join(), await join()
        try:
            await runtime.next_frame(a["stream_id"])
            await runtime.next_frame(b["stream_id"])
            first = runtime.players[a["stream_id"]]
            first["input_activity"].last_input -= 61
            # Owner/media traffic is fresh; only the input deadline expires.
            runtime.player(a["stream_id"])
            deadline = time.monotonic() + 4
            while (
                a["stream_id"] not in runtime.member_retirements
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.05)
            assert a["stream_id"] in runtime.member_retirements
            assert b["stream_id"] in runtime.players
            assert (await runtime.next_frame(b["stream_id"]))["index"] == 1
            c = await join()
            assert (await runtime.next_frame(c["stream_id"]))["index"] == 0
        finally:
            for stream in list(runtime.players):
                await runtime.leave(stream)
            await runtime.close()

    asyncio.run(run())


def test_render_deadline_ends_player_even_with_live_model_and_heartbeat(tmp_path):
    from types import SimpleNamespace as NS

    import pytest

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def run():
        activity = PlayerInputActivity()
        activity.start(time.monotonic() - 61)
        entry = dict(
            input_activity=activity, identity={"stream_id": "s"}, trace_id="idle-source"
        )
        runtime = NS(
            players={"s": entry},
            input_idle_policy=PlayerIdlePolicy(tmp_path / "config"),
        )
        runtime.input_idle_expired = lambda value: RoomRuntime.input_idle_expired(
            runtime, value
        )
        source = SourceSession.__new__(SourceSession)
        source.host = NS(runtime=runtime)
        source.engine = NS(stream_id="s")
        source.finished = asyncio.Event()
        source.model_binding = NS(closed=False)
        source.control_prefill = source.startup = None
        with pytest.raises(RenderError) as error:
            await source._deadline()
        assert error.value.code == "PLAYER_IDLE_TIMEOUT"

    asyncio.run(run())


def test_native_observer_fences_epoch_and_ignores_media(tmp_path):
    from types import SimpleNamespace as NS

    async def run():
        async def event(*args):
            pass

        activity = PlayerInputActivity()
        activity.start(1)
        entry = dict(
            input_activity=activity,
            identity={},
            game_id="g",
            engine=NS(descriptor={"sessionId": "sdk", "epoch": 2}),
        )
        runtime = NS(
            player=lambda *args, **kwargs: entry,
            rooms={"g": {"log": NS(event=event, manifest={})}},
        )

        async def observe(direction, message, epoch=2):
            await RoomRuntime.observe_player_control(
                runtime, "s", "a", direction, json.dumps(dict(message, epoch=epoch))
            )

        await observe("to-sdk", state(1, heldKeys=["KeyW"]), 1)
        await observe("from-sdk", receipt(1))
        assert activity.last_input == 1
        await observe("to-sdk", state(2, heldKeys=["KeyW"]))
        await observe("from-sdk", receipt(2))
        assert activity.last_input > 1
        last = activity.last_input
        await observe("from-sdk", {"type": "source.frame"})
        await observe("from-sdk", receipt(2))
        assert activity.last_input == last

    asyncio.run(run())


def test_input_deadline_does_not_access_cli_before_startup_finishes(monkeypatch):
    from types import SimpleNamespace as NS

    from sglang.multimodal_gen.runtime.realtime.whitebox import render_source

    async def run():
        source = render_source.SourceSession.__new__(render_source.SourceSession)
        source.finished = asyncio.Event()
        source.model_binding = source.control_prefill = source.startup = None
        source.engine = None
        source.host = NS()  # No player/CLI exists during asset preparation.
        source.last_tick = time.monotonic()
        source.bound_at = None
        source.binding_ended = False

        async def once(_):
            source.finished.set()

        monkeypatch.setattr(render_source.asyncio, "sleep", once)
        await source._deadline()

    asyncio.run(run())


def test_idle_deadline_is_inclusive_without_elapsed_subtraction_rounding():
    import math

    activity = PlayerInputActivity()
    activity.start(998.266453032)
    deadline = activity.last_input + 60
    assert not activity.is_expired(math.nextafter(deadline, -math.inf), 60)
    assert activity.is_expired(deadline, 60)


def test_semantic_activity_requires_matching_committed_receipt_and_original_send_time():
    activity = PlayerInputActivity()
    activity.start(0)

    def batch(sequence, generation=1):
        return dict(
            type="input.semantic.batch",
            generation=generation,
            batch=dict(
                sequence=sequence, stateRevision=1, state={"custom": 0.5}, events=[]
            ),
        )

    def consumed(sequence, meaningful=True, generation=1, **extra):
        return dict(
            type="input.semantic.receipt",
            clientId="a",
            generation=generation,
            sequence=sequence,
            stateRevision=1,
            simulationTick=sequence,
            meaningful=meaningful,
            **extra,
        )

    for sequence in range(1, 90):
        activity.observe("a", "to-sdk", batch(sequence), sequence)
        activity.observe("a", "from-sdk", consumed(sequence), sequence + 0.1)
    assert activity.last_input == 89
    activity.observe("a", "to-sdk", batch(90), 90)
    activity.observe("a", "from-sdk", consumed(90, False), 91)
    assert activity.last_input == 89
    activity.observe("a", "to-sdk", batch(91), 91)
    activity.observe("a", "from-sdk", consumed(91), 140)
    assert activity.last_input == 91
    assert activity.is_expired(151, 60)
    activity.observe("a", "from-sdk", consumed(91), 152)
    assert activity.last_input == 91


def test_semantic_handshake_old_generation_wrong_client_and_unknown_receipts_do_not_renew():
    activity = PlayerInputActivity()
    activity.start(0)
    batch = dict(
        type="input.semantic.batch",
        generation=1,
        batch=dict(sequence=1, stateRevision=3),
    )
    activity.observe("a", "to-sdk", batch, 20)
    activity.observe("a", "from-sdk", dict(type="input.semantic.available"), 21)
    assert activity.last_input == 0
    for packet in [
        dict(type="input.semantic.open"),
        dict(type="input.semantic.applied", generation=2),
    ]:
        activity.observe("a", "to-sdk", packet, 30)
    receipt = dict(
        type="input.semantic.receipt",
        clientId="a",
        generation=1,
        sequence=1,
        stateRevision=3,
        simulationTick=4,
        meaningful=True,
    )
    activity.observe("a", "from-sdk", receipt, 31)
    assert activity.last_input == 0
    activity.observe("a", "to-sdk", {**batch, "generation": 2}, 40)
    activity.observe("b", "from-sdk", {**receipt, "generation": 2}, 41)
    activity.observe(
        "a", "from-sdk", {**receipt, "generation": 2, "stateRevision": 4}, 41
    )
    assert activity.last_input == 0


def test_native_archive_retains_all_semantic_envelopes_without_claiming_execution():
    from types import SimpleNamespace as NS

    async def run():
        records = []

        async def event(*args):
            records.append(args)

        entry = dict(
            identity={},
            game_id="g",
            engine=NS(descriptor={"sessionId": "sdk", "epoch": 2}),
        )
        runtime = NS(
            player=lambda *args, **kwargs: entry,
            rooms={"g": {"log": NS(event=event, manifest={})}},
        )
        kinds = [("to-sdk", k) for k in ("open", "applied", "batch", "cancel")]
        kinds += [("from-sdk", k) for k in ("binding", "receipt", "error", "available")]
        for direction, kind in kinds:
            wire = dict(type="input.semantic." + kind, epoch=2, generation=3)
            await RoomRuntime.observe_player_control(
                runtime, "s", "a", direction, json.dumps(wire)
            )
        assert len(records) == 8
        assert [r[2]["message"]["type"] for r in records] == [
            "input.semantic." + k for _, k in kinds
        ]
        assert all(
            r[1] in ("viewer_input_observed", "viewer_input_receipt") for r in records
        )
        await RoomRuntime.observe_player_control(
            runtime,
            "s",
            "a",
            "from-sdk",
            json.dumps(dict(type="input.semantic.receipt", epoch=1)),
        )
        assert len(records) == 8

    asyncio.run(run())


def test_semantic_reopen_allows_a_fresh_producer_generation_after_sdk_epoch_reset():
    activity = PlayerInputActivity()
    activity.start(0)

    def batch(generation):
        return dict(
            type="input.semantic.batch",
            generation=generation,
            batch=dict(sequence=1, stateRevision=0),
        )

    activity.observe("a", "to-sdk", batch(9), 10)
    activity.observe("a", "to-sdk", dict(type="input.semantic.open"), 11)
    activity.observe("a", "to-sdk", batch(1), 20)
    activity.observe(
        "a",
        "from-sdk",
        dict(
            type="input.semantic.receipt",
            clientId="a",
            generation=1,
            sequence=1,
            stateRevision=0,
            simulationTick=3,
            meaningful=True,
        ),
        21,
    )
    assert activity.last_input == 20
