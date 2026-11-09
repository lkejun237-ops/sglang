# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from whitebox_player_log_test_utils import player_log_capture as player_log_capture

from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
    new_id,
)


def test_player_sdk_retirement_does_not_wait_for_gpu_or_close_other_view():
    async def run():
        runtime = RoomRuntime.__new__(RoomRuntime)
        runtime.epoch = new_id()
        runtime.players, runtime.retiring_players = {}, {}
        runtime.member_retirements, runtime.player_closures = {}, {}
        runtime.join_attempts = {}
        game, a, b = new_id(), new_id(), new_id()
        room = dict(players={a, b}, control_caller=object(), log=NS(event=AsyncMock()))
        runtime.rooms = {game: room}
        runtime.locks = {game: asyncio.Lock()}
        gpu_gate, gpu_started = asyncio.Event(), asyncio.Event()

        async def model_close(record):
            gpu_started.set()
            await gpu_gate.wait()

        runtime.member_media = NS(close=model_close)
        for stream in (a, b):
            runtime.players[stream] = dict(
                game_id=game,
                identity={},
                engine=NS(close=AsyncMock(), renderer=NS(close=AsyncMock())),
                control_admission={},
            )
        b_engine = runtime.players[b]["engine"]
        task = asyncio.create_task(runtime.leave(a))
        await gpu_started.wait()
        for _ in range(20):
            if a in runtime.member_retirements:
                break
            await asyncio.sleep(0)
        assert not task.done()
        assert room["players"] == {b}
        assert a not in runtime.players and a in runtime.retiring_players
        assert not runtime.locks[game].locked()
        b_engine.close.assert_not_called()
        gpu_gate.set()
        assert await task is True
        assert await runtime.leave(a) is True
        assert a not in runtime.retiring_players
        assert runtime.rooms[game] is room

    asyncio.run(run())


def test_world_has_no_fixed_lifetime_but_lost_owner_lease_still_fences_it():
    async def run():
        now = [100.0]
        directory = RoomDirectory(
            MemoryRoomStore(),
            {"a": "http://a"},
            max_seconds=40,
            lease_seconds=10,
            clock=lambda: now[0],
        )
        game, epoch = new_id(), new_id()
        await directory.resolve(game, "world", create=True)
        for _ in range(100):
            now[0] += 5
            value = await directory.touch(game, "a", epoch)
            assert value["expires_at"] is None
        assert (await directory.resolve(game, "world"))["status"] == "active"
        now[0] += 11
        with pytest.raises(ValueError, match="ended"):
            await directory.resolve(game, "world")
        assert (await directory.lookup(game))["owner_epoch"] == epoch

    asyncio.run(run())


@pytest.mark.parametrize("model_ready", [False, True])
def test_shared_admission_reserves_model_before_creating_sdk_view(
    model_ready, monkeypatch
):
    from contextlib import asynccontextmanager

    import msgspec

    from sglang.multimodal_gen.runtime.realtime.whitebox.player_media import MemberMedia
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
        RoomControlRuntime,
    )

    async def run():
        monkeypatch.setenv("WHITEBOX_BOUND_COMPUTE", "1")
        calls = []

        class Socket:
            def __init__(self):
                self.first = True
                self.ended = asyncio.Event()

            async def recv(self):
                if self.first:
                    self.first = False
                    return msgspec.msgpack.encode(
                        dict(
                            type="bound.ready",
                            protocol="bound-compute-v1",
                            worker_epoch="e",
                            player_expires_at=200,
                        )
                    )
                await self.ended.wait()
                raise ConnectionError("closed")

            async def close(self):
                self.ended.set()

        class Lease:
            resource_state = "none"

            @asynccontextmanager
            async def open(self):
                calls.append("model-reserve")
                if not model_ready:
                    raise ValueError("capacity-full")
                self.resource_state = "reserved"
                try:
                    yield (
                        Socket(),
                        NS(
                            session_id="parent",
                            generation_id="g",
                            token="t",
                            denoiser=NS(worker_epoch="e"),
                            player_expires_at=200,
                        ),
                    )
                finally:
                    self.resource_state = "released"

        players = {}

        async def join(body):
            calls.append("sdk-join")
            from sglang.multimodal_gen.runtime.realtime.whitebox.player_idle import (
                PlayerInputActivity,
            )

            players[body["stream_id"]] = dict(
                engine=NS(
                    descriptor=dict(playerBinding=dict(playerSessionId="member"))
                ),
                input_activity=PlayerInputActivity(),
            )

        runtime = NS(
            epoch="room-e", join=join, player=lambda stream, **kwargs: players[stream]
        )
        host = NS(
            runtime=runtime,
            make_segment_leases=lambda *args: Lease(),
            config=NS(startup_seconds=1),
        )
        MemberMedia(host)
        control = RoomControlRuntime(runtime)
        record = dict(
            binding=dict(gameId="game", publicationId="pub"),
            stream_id="stream",
            player_id="player",
            session_id="source",
            spec={},
            caller={"subject": "user"},
            admission={"operationId": "admit"},
            status="pending",
        )
        room = dict(route={"world_id": "world"}, control_caller=record["caller"])
        await control.admit(record, room)
        assert calls == (
            ["model-reserve", "sdk-join"] if model_ready else ["model-reserve"]
        )
        assert record["status"] == ("claimed" if model_ready else "blocked")
        if model_ready:
            assert players["stream"]["player_expires_at"] == 200
            assert record["model_binding"].lease.resource_state == "reserved"
        if record.get("model_binding"):
            await record["model_binding"].close()

    asyncio.run(run())


def test_owner_heartbeat_keeps_renewing_while_sdk_join_holds_game_lock(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.whitebox import room_runtime

    async def run():
        runtime = RoomRuntime.__new__(RoomRuntime)
        game = new_id()
        runtime.owner_id = "owner"
        runtime.epoch = new_id()
        runtime.rooms = {game: dict(players=set())}
        runtime.players = {}
        runtime.locks = {game: asyncio.Lock()}
        await runtime.locks[game].acquire()
        calls = []

        async def touch(*args):
            calls.append(args)
            if len(calls) == 3:
                raise asyncio.CancelledError

        runtime.directory = NS(touch=touch)
        runtime._end = AsyncMock()
        monkeypatch.setattr(room_runtime.asyncio, "sleep", AsyncMock())
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(runtime._heartbeat(game), 0.2)
        assert len(calls) == 3
        runtime._end.assert_not_called()
        runtime.locks[game].release()

    asyncio.run(run())


@pytest.mark.parametrize("verified", [False, True])
def test_parent_final_receipt_resolves_lost_child_ack_but_unknown_release_is_retained(
    verified, monkeypatch
):
    from unittest.mock import Mock

    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
        InteractiveSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def run():
        source = SourceSession.__new__(SourceSession)
        source._source_close_task = None
        source.session_id = "source"
        source.outcome = "completed"
        source.current_segment = NS(
            closed_event=asyncio.Event(),
            stop=AsyncMock(side_effect=ConnectionError("child ack lost")),
        )
        lease = NS(resource_state="reserved", _active_assignment=None)
        confirmed = [verified]

        async def parent_close():
            lease.resource_state = "released" if confirmed[0] else "quarantined"
            if not confirmed[0]:
                raise RenderError("RELEASE_UNKNOWN")

        source.model_binding = NS(assignment=object(), lease=lease, close=parent_close)
        source._sdk_cleanup_for = AsyncMock(return_value={"state": "retired"})
        source.startup = source.control_prefill = None
        source.tasks = set()
        source.control_tasks = []
        source.retained = NS(clear=Mock())
        source.assets = None
        snapshot = {
            "source_state": "ready",
            "operations": {"r": {"op": "source.close", "state": "accepted"}},
        }

        async def update(fn, **_):
            fn(snapshot)

        source.journal = NS(snapshot=snapshot, update=update)
        monkeypatch.setattr(InteractiveSession, "close", AsyncMock())
        if verified:
            await source.close()
            source.retained.clear.assert_called_once()
            assert snapshot["cleanup_state"] == "released"
            assert snapshot["operations"]["r"]["state"] == "completed"
        else:
            with pytest.raises(RenderError) as error:
                await source.close()
            assert error.value.code == "RELEASE_UNKNOWN"
            source.retained.clear.assert_not_called()
            assert snapshot["cleanup_state"] == "quarantined"
            # Only a later positive acknowledgement from the original parent
            # allows the explicit retry to finish; the source stays ended.
            confirmed[0] = True
            await source.close()
            assert snapshot["source_state"] == "ended"
            assert snapshot["cleanup_state"] == "released"
            source.retained.clear.assert_called_once()
        assert source._sdk_cleanup_for.await_count == (1 if verified else 2)

    asyncio.run(run())


def test_user_finish_preserves_normal_tail_before_verified_close():
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def run():
        source = SourceSession.__new__(SourceSession)
        source.player_finish_task = None
        source.close_requested = False
        source.host = NS(runtime=RoomRuntime)
        finishing = asyncio.Event()
        calls = []

        async def finish():
            finishing.set()

        source.finish = finish

        async def drain():
            await finishing.wait()
            await asyncio.sleep(0.01)
            calls.extend(["last-frame", "eos"])

        async def close():
            calls.append("cleanup")

        pipeline = NS(task=asyncio.create_task(drain()), close=close)
        source.request_player_finish(pipeline)
        original = source.player_finish_task
        source.request_player_finish(pipeline)
        assert source.player_finish_task is original
        await asyncio.wait_for(original, 1)
        assert calls == ["last-frame", "eos", "cleanup"]
        assert source.close_requested

    asyncio.run(run())


@pytest.mark.parametrize("failed", [False, True])
def test_forced_close_reports_completion_only_after_confirmed_cleanup(failed):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        RoomPipeline,
    )

    async def run():
        gate = asyncio.Event()

        async def cleanup():
            await gate.wait()
            if failed:
                raise RuntimeError("unconfirmed")

        owner = RoomPipeline.__new__(RoomPipeline)
        owner.closed = False
        owner.terminal = None
        owner.subscription = None
        owner.task = owner.commands = owner.retention = None
        owner.session = NS(close=cleanup, close_requested=True)
        owner.session_id = "source"
        owner.host = NS(sessions={"source": owner})
        owner.init = {}
        owner.history = []
        owner.history_bytes = 0
        owner.input_history = {}
        owner.report = NS(gateway_done=asyncio.Event(), recording=object())
        task = asyncio.create_task(owner.close())
        await asyncio.sleep(0)
        assert owner.terminal is None
        gate.set()
        if failed:
            with pytest.raises(RuntimeError, match="unconfirmed"):
                await task
            assert (
                owner.terminal["type"] == "error"
                and owner.terminal["code"] == "RELEASE_UNKNOWN"
            )
        else:
            await task
            assert (
                owner.terminal["type"] == "session.completed"
                and owner.terminal["cleanup_confirmed"] is True
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    "admission_state", ["running", "failed", "cancelled", "leaving"]
)
def test_empty_room_tracks_actual_admission_task(monkeypatch, admission_state):
    from sglang.multimodal_gen.runtime.realtime.whitebox import room_runtime

    async def run():
        runtime = RoomRuntime.__new__(RoomRuntime)
        game = new_id()
        runtime.owner_id, runtime.epoch = "owner", new_id()
        room = dict(players=set())
        runtime.rooms, runtime.players = {game: room}, {}
        runtime.locks = {game: asyncio.Lock()}
        pending = asyncio.get_running_loop().create_future()
        if admission_state == "failed":
            pending.set_result(None)
        elif admission_state == "cancelled":
            pending.cancel()
        runtime.control = NS(
            admissions={
                "a": dict(
                    binding=dict(gameId=game),
                    status="blocked",
                    task=pending,
                    leaving=admission_state == "leaving",
                )
            }
        )
        now, ticks = [0], []

        async def touch(*_):
            ticks.append(1)
            now[0] += 121
            if len(ticks) == 4:
                raise asyncio.CancelledError

        runtime.directory = NS(touch=touch)
        runtime._end = AsyncMock()
        monkeypatch.setattr(room_runtime.asyncio, "sleep", AsyncMock())
        monkeypatch.setattr(room_runtime.time, "monotonic", lambda: now[0])
        if admission_state == "running":
            with pytest.raises(asyncio.CancelledError):
                await runtime._heartbeat(game)
            runtime._end.assert_not_called()
        else:
            await runtime._heartbeat(game)
            runtime._end.assert_awaited_once_with(game, "empty", expected_room=room)
        pending.cancel()

    asyncio.run(run())


def test_failed_admission_logs_original_error_and_operation(player_log_capture):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
        RoomControlRuntime,
    )

    async def run():
        error = ValueError("private backend failure")
        runtime = NS(
            member_media=NS(bound_compute=True, reserve=AsyncMock(side_effect=error))
        )
        control = RoomControlRuntime(runtime)
        record = dict(
            stream_id="stream",
            admission=dict(operationId="original-admission", parametersSha256="a" * 64),
        )
        await control.admit(record, {})
        assert record["status"] == "blocked"
        events = [
            json.loads(item.getMessage())
            for item in player_log_capture.records
            if '"event":"whitebox.player.failed"' in item.getMessage()
        ]
        assert len(events) == 1
        assert events[0]["operationId"] == "original-admission"
        assert events[0]["stage"] == "member.admit"
        assert events[0]["phase"] == "model_reserve"
        assert events[0]["outcome"] == "unknown"
        assert events[0]["error"] == dict(
            code="UNCLASSIFIED_FAILURE", type="ValueError"
        )
        assert "private backend failure" not in player_log_capture.text
        assert control.admission_result(record)["errorCode"] == "ADMISSION_UNCONFIRMED"
        assert "private backend failure" not in str(control.admission_result(record))

    asyncio.run(run())


@pytest.mark.parametrize(
    "manual,prefill", [(False, False), (True, False), (True, True)]
)
def test_member_model_reservation_is_deferred_only_for_manual_preview(
    monkeypatch, manual, prefill
):
    import base64

    import msgspec
    from test_whitebox_player_media import media_launch

    from sglang.multimodal_gen.runtime.realtime.whitebox.player_media import MemberMedia

    async def run():
        launch = media_launch()
        init = msgspec.msgpack.decode(base64.b64decode(launch["initMsgpackBase64"]))
        if manual:
            init.update(source_operation="create.manual", inference_control="manual")
            launch["initMsgpackBase64"] = base64.b64encode(
                msgspec.msgpack.encode(init)
            ).decode()
        if prefill:
            launch["memberPrefill"] = {"protocol": "worldkit-member-prefill-v1"}
        host = NS(
            runtime=NS(),
            make_segment_leases=lambda *_: (_ for _ in ()).throw(
                ValueError("reservation-attempt")
            ),
        )
        owner = MemberMedia(host)
        record = dict(
            mediaLaunch=launch,
            binding=dict(publicationId=launch["publicationIdentity"]["publicationId"]),
            spec=dict(width=32, height=32, fps=24),
            caller=dict(subject="user"),
            admission=dict(operationId="admit"),
        )
        if manual and not prefill:
            await owner.reserve(record)
            assert record.get("model_binding") is None
            assert record["sdk_join_done"].is_set()
        else:
            with pytest.raises(ValueError, match="reservation-attempt"):
                await owner.reserve(record)

    asyncio.run(run())
