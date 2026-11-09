# SPDX-License-Identifier: Apache-2.0
"""An unpublished SDK member must retain its original cleanup proof."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.room_mock import MockRoom
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
    new_id,
)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_join_in_empty_room_keeps_only_confirmed_retirement(
    tmp_path, cleanup_fails
):
    async def run():
        driver = MockRoom()
        driver.join = AsyncMock(side_effect=ValueError("STREAM_SOURCE_FAILED"))
        driver.cancel_join = AsyncMock(side_effect=ValueError("SERVICE_SESSION_FAILED"))
        close = driver.close
        driver.close = (
            AsyncMock(side_effect=ValueError("cleanup blocked"))
            if cleanup_fails
            else AsyncMock(wraps=close)
        )
        runtime = RoomRuntime(
            directory=RoomDirectory(MemoryRoomStore(), {"room": "http://room"}),
            owner_id="room",
            driver_factory=AsyncMock(return_value=driver),
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        game, player, stream, session = (new_id() for _ in range(4))
        await runtime.directory.resolve(game, "s/w", create=True)
        try:
            with pytest.raises(ValueError, match="STREAM_SOURCE_FAILED"):
                await runtime.join(
                    dict(
                        game_id=game,
                        player_id=player,
                        stream_id=stream,
                        session_id=session,
                        world_id="s/w",
                        spec=dict(width=32, height=32, fps=24),
                        user_id="user",
                        trace_id="original-join",
                    )
                )
            assert stream not in runtime.players
            assert stream not in runtime.retiring_players
            driver.close.assert_awaited_once()
            if cleanup_fails:
                with pytest.raises(ValueError, match="SDK_MEMBER_CLEANUP_UNKNOWN"):
                    await runtime.retire_member(stream)
                assert (await runtime.directory.lookup(game))["status"] != "ended"
                return
            receipt = await runtime.retire_member(stream)
            assert receipt == dict(
                state="retired",
                source_id=stream,
                view_id=stream,
                owner_epoch=runtime.epoch,
            )
            assert await runtime.retire_member(stream) == receipt
            assert await runtime.leave(stream) is True
            assert (await runtime.directory.lookup(game))["status"] == "ended"
        finally:
            driver.close = close
            await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "proof_kind", ["confirmed", "wrong-player", "missing", "boolean"]
)
def test_failed_join_preserves_another_player_and_requires_original_proof(
    tmp_path, proof_kind
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_startup import (
        MemberCleanupReceipt,
    )

    async def run():
        driver = MockRoom()
        runtime = RoomRuntime(
            directory=RoomDirectory(MemoryRoomStore(), {"room": "http://room"}),
            owner_id="room",
            driver_factory=AsyncMock(return_value=driver),
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        game = new_id()
        await runtime.directory.resolve(game, "s/w", create=True)
        peer, failed = (
            {
                "game_id": game,
                "player_id": new_id(),
                "stream_id": new_id(),
                "session_id": new_id(),
                "world_id": "s/w",
                "spec": dict(width=32, height=32, fps=24),
                "user_id": name,
                "trace_id": new_id(),
            }
            for name in ["peer", "failed"]
        )
        try:
            await runtime.join(peer)
            peer_view = runtime.players[peer["stream_id"]]["engine"]
            close = driver.close
            driver.close = AsyncMock(wraps=close)
            driver.join = AsyncMock(side_effect=ValueError("join failed"))
            proof = {
                "confirmed": MemberCleanupReceipt(failed["player_id"]),
                "wrong-player": MemberCleanupReceipt(peer["player_id"]),
                "missing": None,
                "boolean": True,
            }[proof_kind]
            driver.cancel_join = AsyncMock(return_value=proof)
            with pytest.raises(ValueError, match="join failed"):
                await runtime.join(failed)
            driver.close.assert_not_called()
            assert runtime.rooms[game]["players"] == {peer["stream_id"]}
            assert not peer_view.stopped
            await peer_view.send_input(0, {"keys": ["KeyW"]})
            assert driver.inputs[peer["player_id"]]["sequence"] == 0
            if proof_kind == "confirmed":
                assert (await runtime.retire_member(failed["stream_id"]))[
                    "state"
                ] == "retired"
                assert await runtime.leave(failed["stream_id"]) is True
            else:
                with pytest.raises(ValueError, match="SDK_MEMBER_CLEANUP_UNKNOWN"):
                    await runtime.retire_member(failed["stream_id"])
        finally:
            await runtime.close()

    asyncio.run(run())


def test_repeated_cancel_waits_for_original_empty_room_cleanup(tmp_path):
    async def run():
        entered, closing, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
        driver = MockRoom()

        async def join(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        async def close():
            closing.set()
            await finish.wait()

        driver.join, driver.close = join, close
        runtime = RoomRuntime(
            directory=RoomDirectory(MemoryRoomStore(), {"room": "http://room"}),
            owner_id="room",
            driver_factory=AsyncMock(return_value=driver),
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        game, player, stream, session = (new_id() for _ in range(4))
        await runtime.directory.resolve(game, "s/w", create=True)
        task = asyncio.create_task(
            runtime.join(
                dict(
                    game_id=game,
                    player_id=player,
                    stream_id=stream,
                    session_id=session,
                    world_id="s/w",
                    spec=dict(width=32, height=32, fps=24),
                    user_id="user",
                    trace_id="cancelled-join",
                )
            )
        )
        try:
            await entered.wait()
            task.cancel()
            await closing.wait()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert stream not in runtime.member_retirements
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert (await runtime.retire_member(stream))["state"] == "retired"
        finally:
            finish.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await runtime.close()

    asyncio.run(run())


def test_worldkit_cancel_join_returns_proof_only_after_exact_member_left(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_startup import (
        MemberCleanupReceipt,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.worldkit_room import (
        WorldkitRoom,
    )

    async def run():
        room = WorldkitRoom.__new__(WorldkitRoom)
        room.closed = False
        room.timeout = 1
        room.ready = dict(
            roomId="sdk-room", room={"roomId": "sdk-room", "roomEpoch": 1}
        )
        player, other = new_id(), new_id()
        credential = tmp_path / "private.json"
        credential.write_text("{}")
        record = dict(
            lock=asyncio.Lock(),
            member=dict(
                memberFile="private-member.json",
                binding=dict(playerSessionId="sdk-player", connectionEpoch=1),
            ),
            credential_file=credential,
        )
        room.members, room.views = {player: record, other: {}}, {}
        room.call = AsyncMock(return_value={"phase": "left"})
        assert await room.cancel_join(player) == MemberCleanupReceipt(player)
        room.call.assert_awaited_once_with(
            "leave", "--member-file", "private-member.json"
        )
        assert room.members == {other: {}} and not credential.exists()
        assert await room.cancel_join(player) is None
        room.views[other] = object()
        assert await room.cancel_join(other) is None
        assert other in room.members

    asyncio.run(run())


def test_failed_bound_admission_releases_parent_and_next_player_can_join(
    tmp_path, monkeypatch
):
    from contextlib import asynccontextmanager

    import msgspec

    from sglang.multimodal_gen.runtime.realtime.whitebox.player_media import MemberMedia
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_control_runtime import (
        RoomControlRuntime,
    )

    async def run():
        monkeypatch.setenv("WHITEBOX_BOUND_COMPUTE", "1")
        reserved = set()
        receipts = []

        class Socket:
            first = True

            def __init__(self):
                self.closed = asyncio.Event()

            async def recv(self):
                if self.first:
                    self.first = False
                    return msgspec.msgpack.encode(
                        dict(
                            type="bound.ready",
                            protocol="bound-compute-v1",
                            worker_epoch="worker",
                            player_expires_at=200,
                        )
                    )
                await self.closed.wait()
                raise ConnectionError("closed")

            async def close(self):
                self.closed.set()

        class Lease:
            resource_state = "none"

            @asynccontextmanager
            async def open(self):
                assert not reserved, "Slot was not released"
                token = new_id()
                reserved.add(token)
                self.resource_state = "reserved"
                assignment = NS(
                    token=token,
                    session_id=new_id(),
                    generation_id=new_id(),
                    denoiser=NS(worker_epoch="worker"),
                    player_expires_at=200,
                )
                try:
                    yield Socket(), assignment
                finally:
                    try:
                        receipt = await self.sdk_cleanup_provider(assignment)
                        assert receipt["state"] == "retired"
                        assert receipt["session_id"] == assignment.session_id
                        receipts.append(receipt)
                        reserved.remove(token)
                        self.resource_state = "released"
                    except BaseException:
                        self.resource_state = "quarantined"
                        raise

        driver = MockRoom()
        original_join = driver.join
        driver.join = AsyncMock(side_effect=ValueError("SDK media failed"))
        driver.cancel_join = AsyncMock(side_effect=ValueError("join unconfirmed"))
        runtime = RoomRuntime(
            directory=RoomDirectory(MemoryRoomStore(), {"room": "http://room"}),
            owner_id="room",
            driver_factory=AsyncMock(return_value=driver),
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        host = NS(
            runtime=runtime,
            make_segment_leases=lambda *args: Lease(),
            config=NS(startup_seconds=1),
        )
        media = MemberMedia(host)
        runtime.member_media = media
        control = RoomControlRuntime(runtime)
        caller = {"subject": "user"}

        async def admit():
            game = new_id()
            route = await runtime.directory.resolve(game, "s/w", create=True)
            lock = runtime.locks.setdefault(game, asyncio.Lock())
            room = await runtime._start_room(game, route, lock, None, None, new_id())
            room["control_caller"] = caller
            record = dict(
                binding=dict(gameId=game, publicationId="pub"),
                stream_id=new_id(),
                player_id=new_id(),
                session_id=new_id(),
                spec=dict(width=32, height=32, fps=24),
                caller=caller,
                admission={"operationId": new_id()},
                status="pending",
            )
            record["task"] = asyncio.create_task(control.admit(record, room))
            await record["task"]
            return record

        try:
            failed = await admit()
            assert failed["status"] == "blocked" and "member_id" not in failed
            assert failed["model_binding"].lease.resource_state == "released"
            assert len(receipts) == 1 and not reserved
            await control.leave(failed)
            assert failed["status"] == "retired"
            driver.join = original_join
            following = await admit()
            assert following["status"] == "claimed" and len(reserved) == 1
            await control.leave(following)
            assert following["status"] == "retired"
            assert len(receipts) == 2 and not reserved
        finally:
            await runtime.close()

    asyncio.run(run())


def test_queued_join_after_original_room_retired_is_proven_not_created(tmp_path):
    async def run():
        driver = MockRoom()
        runtime = RoomRuntime(
            directory=RoomDirectory(MemoryRoomStore(), {"room": "http://room"}),
            owner_id="room",
            driver_factory=AsyncMock(return_value=driver),
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        game, stream = new_id(), new_id()
        await runtime.directory.resolve(game, "s/w", create=True)
        await runtime.directory.touch(game, "room", runtime.epoch, end=True)
        try:
            with pytest.raises(ValueError):
                await runtime.join(
                    dict(
                        game_id=game,
                        stream_id=stream,
                        player_id=new_id(),
                        session_id=new_id(),
                        user_id="next-user",
                        trace_id="queued",
                        world_id="s/w",
                        spec=dict(width=32, height=32, fps=24),
                    )
                )
            runtime.driver_factory.assert_not_called()
            assert await runtime.retire_member(stream) == dict(
                state="not_created",
                source_id=stream,
                view_id=None,
                owner_epoch=runtime.epoch,
            )
            assert await runtime.leave(stream) is True
        finally:
            await runtime.close()

    asyncio.run(run())


def test_duplicate_join_failure_cannot_retire_an_existing_player(tmp_path):
    async def run():
        driver = MockRoom()
        runtime = RoomRuntime(
            directory=RoomDirectory(MemoryRoomStore(), {"room": "http://room"}),
            owner_id="room",
            driver_factory=AsyncMock(return_value=driver),
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        game, stream = new_id(), new_id()
        await runtime.directory.resolve(game, "s/w", create=True)
        body = dict(
            game_id=game,
            stream_id=stream,
            player_id=new_id(),
            session_id=new_id(),
            user_id="user",
            trace_id="original",
            world_id="s/w",
            spec=dict(width=32, height=32, fps=24),
        )
        try:
            await runtime.join(body)
            view = runtime.players[stream]["engine"]
            with pytest.raises(ValueError, match="already connected"):
                await runtime.join(body)
            assert stream not in runtime.member_retirements
            assert runtime.players[stream]["engine"] is view and not view.stopped
        finally:
            await runtime.close()

    asyncio.run(run())


def test_duplicate_pending_join_cannot_claim_that_original_sdk_was_not_created(
    tmp_path,
):
    async def run():
        entered, finish = asyncio.Event(), asyncio.Event()
        driver = MockRoom()
        original_join = driver.join

        async def join(**kwargs):
            entered.set()
            await finish.wait()
            return await original_join(**kwargs)

        driver.join = join
        runtime = RoomRuntime(
            directory=RoomDirectory(MemoryRoomStore(), {"room": "http://room"}),
            owner_id="room",
            driver_factory=AsyncMock(return_value=driver),
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        game, stream = new_id(), new_id()
        await runtime.directory.resolve(game, "s/w", create=True)
        body = dict(
            game_id=game,
            stream_id=stream,
            player_id=new_id(),
            session_id=new_id(),
            user_id="user",
            trace_id="original",
            world_id="s/w",
            spec=dict(width=32, height=32, fps=24),
        )
        task = asyncio.create_task(runtime.join(body))
        try:
            await entered.wait()
            with pytest.raises(ValueError, match="still in progress"):
                await runtime.join(body)
            assert stream not in runtime.member_retirements
            assert not task.done()
            finish.set()
            await task
            assert stream in runtime.players
            assert stream not in runtime.member_retirements
            assert not runtime.join_attempts
        finally:
            finish.set()
            await asyncio.gather(task, return_exceptions=True)
            await runtime.close()

    asyncio.run(run())


def test_retry_cannot_convert_unknown_dispatched_join_into_not_created(tmp_path):
    async def run():
        driver = MockRoom()
        close = driver.close
        driver.join = AsyncMock(side_effect=ValueError("join failed"))
        driver.close = AsyncMock(side_effect=ValueError("retirement unknown"))
        factory = AsyncMock(return_value=driver)
        runtime = RoomRuntime(
            directory=RoomDirectory(MemoryRoomStore(), {"room": "http://room"}),
            owner_id="room",
            driver_factory=factory,
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        game, stream = new_id(), new_id()
        await runtime.directory.resolve(game, "s/w", create=True)
        body = dict(
            game_id=game,
            stream_id=stream,
            player_id=new_id(),
            session_id=new_id(),
            user_id="user",
            trace_id="original",
            world_id="s/w",
            spec=dict(width=32, height=32, fps=24),
        )
        try:
            with pytest.raises(ValueError, match="join failed"):
                await runtime.join(body)
            assert runtime.join_attempts[stream]["failed"]
            await runtime.directory.touch(game, "room", runtime.epoch, end=True)
            with pytest.raises(ValueError, match="unconfirmed"):
                await runtime.join(body)
            factory.assert_awaited_once()
            with pytest.raises(ValueError, match="SDK_MEMBER_CLEANUP_UNKNOWN"):
                await runtime.retire_member(stream)
        finally:
            driver.close = close
            await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("matching_operation", [False, True])
def test_original_factory_cleanup_receipt_does_not_leave_a_failed_join_fence(
    tmp_path, matching_operation, cancelled
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_startup import (
        FactoryCleanupReceipt,
        RoomStartupFailure,
        startup_cleanup_error,
    )

    async def run():
        async def factory(**kwargs):
            raise startup_cleanup_error(
                asyncio.CancelledError() if cancelled else ValueError("factory failed"),
                FactoryCleanupReceipt(
                    kwargs["trace_id"] if matching_operation else "another-operation"
                ),
            )

        runtime = RoomRuntime(
            directory=RoomDirectory(MemoryRoomStore(), {"room": "http://room"}),
            owner_id="room",
            driver_factory=factory,
            spool=tmp_path,
            spec=VideoSpec(32, 32),
        )
        game, stream = new_id(), new_id()
        await runtime.directory.resolve(game, "s/w", create=True)
        try:
            with pytest.raises(
                asyncio.CancelledError if cancelled else RoomStartupFailure
            ):
                await runtime.join(
                    dict(
                        game_id=game,
                        stream_id=stream,
                        player_id=new_id(),
                        session_id=new_id(),
                        user_id="user",
                        trace_id="original-factory",
                        world_id="s/w",
                        spec=dict(width=32, height=32, fps=24),
                    )
                )
            if matching_operation:
                assert (await runtime.retire_member(stream))["state"] == "not_created"
                assert stream not in runtime.join_attempts
            else:
                assert runtime.join_attempts[stream]["failed"]
                with pytest.raises(ValueError, match="SDK_MEMBER_CLEANUP_UNKNOWN"):
                    await runtime.retire_member(stream)
        finally:
            await runtime.close()

    asyncio.run(run())
