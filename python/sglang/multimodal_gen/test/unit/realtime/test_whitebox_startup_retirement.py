"""A failed native CLI startup must carry positive cleanup through admission."""

import asyncio
import json
import sys
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox import cli_room as cli
from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
    AsyncArchiveManager,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession
from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
    LocalRoomFactory,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
from sglang.multimodal_gen.runtime.realtime.whitebox.room_startup import (
    FactoryCleanupReceipt,
    RoomCleanupReceipt,
    RoomStartupFailure,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
    new_id,
)


@pytest.mark.parametrize(
    "sdk_code,sdk_schema,service_mode,confirmed",
    [
        ("WTP_DELIVERY_NOT_FOUND", 1, "auto", True),
        ("STREAM_DELIVERY_UI_REQUIRED", 1, "auto", True),
        ("SERVICE_QUEUE_TIMEOUT", 1, "auto", True),
        ("SERVICE_QUEUE_TIMEOUT", None, "auto", False),
        ("STREAM_STARTUP_TIMEOUT", 1, "auto", False),
        ("WTP_DELIVERY_NOT_FOUND", None, "auto", False),
        ("SERVICE_DISCONNECTED", 1, "auto", False),
        ("STREAM_START_FAILED", 1, "auto", False),
        ("STREAM_START_FAILED", 1, "off", True),
        ("STREAM_START_FAILED", None, "off", True),
    ],
)
def test_failed_cli_process_cleanup_reaches_source_receipt(
    tmp_path, monkeypatch, sdk_code, sdk_schema, service_mode, confirmed
):
    async def run():
        entry = tmp_path / "entry.mjs"
        entry.touch()
        monkeypatch.setenv("WHITEBOX_CLI_ENTRYPOINT", str(entry))
        monkeypatch.setenv("WHITEBOX_CLI_UI_FACTORY", "builtins:object")
        monkeypatch.setenv("WHITEBOX_CLI_RESOURCE_TRANSPORT", "cdn")
        monkeypatch.setenv("WHITEBOX_CLI_SERVICE_MODE", service_mode)
        monkeypatch.setattr(
            cli,
            "cli_arguments",
            lambda *a, **k: [
                sys.executable,
                "-c",
                'import sys,json;print(json.dumps({"kind":"worldkit-cli-error","schemaVersion":'
                + repr(sdk_schema)
                + ',"error":{"code":'
                + repr(sdk_code)
                + "}}),file=sys.stderr);sys.exit(7)",
            ],
        )
        processes = []
        create = asyncio.create_subprocess_exec

        async def capture(*a, **k):
            p = await create(*a, **k)
            processes.append(p)
            return p

        monkeypatch.setattr(cli.asyncio, "create_subprocess_exec", capture)
        directory = RoomDirectory(MemoryRoomStore(), {"room": "http://room"})
        game, player, stream = new_id(), new_id(), new_id()
        await directory.resolve(game, "world", create=True)
        archives = AsyncArchiveManager()
        runtime = RoomRuntime(
            directory=directory,
            owner_id="room",
            driver_factory=cli.cli_room,
            spool=tmp_path,
            spec=VideoSpec(32, 32),
            archives=archives,
        )
        factory = LocalRoomFactory(runtime, game, player, stream)
        try:
            with pytest.raises(ValueError) as caught:
                await factory.open_session(
                    world_id="world",
                    spec=VideoSpec(32, 32),
                    trace_id="original",
                    context={"session_id": new_id(), "user_id": "u"},
                )
            assert len(processes) == 1 and processes[0].returncode == 7
            assert game not in runtime.rooms
            if not confirmed:
                assert not isinstance(caught.value, RoomStartupFailure)
                with pytest.raises(ValueError, match="CLEANUP_UNKNOWN"):
                    await factory.retire_failed_start()
                return
            assert isinstance(caught.value, RoomStartupFailure)
            assert game not in runtime.retiring
            source = NS(
                engine=None,
                config=NS(engine_factory=factory),
                _sdk_retire_task=None,
                _sdk_create_done=asyncio.Event(),
                _sdk_create_started=True,
                model_binding=object(),
            )
            source._sdk_create_done.set()
            from test_player_lifecycle import system

            from sglang.multimodal_gen.runtime.realtime.coordinator import (
                CoordinatorRejected,
            )

            _, _, worker, coordinator, http, admit = await system("memory")
            try:
                assignment = await admit("first-player")
                await worker.consume(
                    assignment.token,
                    session_id=assignment.session_id,
                    generation_id=assignment.generation_id,
                    worker_epoch="e",
                    owner_id="original-socket",
                )
                receipt = await SourceSession._sdk_cleanup_for(source, assignment)
                assert receipt["state"] == "not_created"
                assert (
                    receipt["source_id"] == stream
                    and receipt["owner_epoch"] == runtime.epoch
                )
                with pytest.raises(CoordinatorRejected, match="RELEASE_PENDING"):
                    await coordinator.release_verified(assignment, sdk_cleanup=receipt)
                await worker.release(assignment.token, owner_id="original-socket")
                await coordinator.release_verified(assignment, sdk_cleanup=receipt)
                assert (await worker.snapshot())["active_sessions"] == 0
                following = await admit("next-player")
                assert following.token != assignment.token
                assert (await worker.snapshot())["reserved_sessions"] == 1
            finally:
                await http.aclose()
        finally:
            await runtime.close()

    asyncio.run(run())


def test_foreground_cleanup_failure_does_not_issue_receipt(tmp_path, monkeypatch):
    async def run():
        entry = tmp_path / "entry.mjs"
        entry.touch()
        monkeypatch.setenv("WHITEBOX_CLI_ENTRYPOINT", str(entry))
        monkeypatch.setenv("WHITEBOX_CLI_UI_FACTORY", "builtins:object")
        monkeypatch.setenv("WHITEBOX_CLI_RESOURCE_TRANSPORT", "cdn")
        monkeypatch.setenv("WHITEBOX_CLI_SERVICE_MODE", "off")
        monkeypatch.setattr(
            cli,
            "cli_arguments",
            lambda *a, **k: [sys.executable, "-c", "raise SystemExit(7)"],
        )
        monkeypatch.setattr(
            cli.CLIRoom,
            "close",
            AsyncMock(side_effect=RuntimeError("view cleanup failed")),
        )
        with pytest.raises(RuntimeError, match="view cleanup failed") as failure:
            await cli.cli_room(
                world_id="source/world", spec=VideoSpec(32, 32), trace_id="original"
            )
        assert not isinstance(failure.value, RoomStartupFailure)

    asyncio.run(run())


def test_sdk_retirement_retry_uses_the_same_original_factory():
    async def run():
        factory = NS(
            retire_failed_start=AsyncMock(
                side_effect=[
                    ValueError("SDK_MEMBER_CLEANUP_UNKNOWN"),
                    dict(
                        state="not_created",
                        source_id="original-stream",
                        owner_epoch="original-owner",
                    ),
                ]
            )
        )
        source = NS(
            engine=None,
            config=NS(engine_factory=factory),
            _sdk_retire_task=None,
            _sdk_create_done=asyncio.Event(),
            _sdk_create_started=True,
            model_binding=object(),
        )
        source._sdk_create_done.set()
        assignment = NS(
            token="original-token",
            session_id="original-session",
            generation_id="original-generation",
        )
        with pytest.raises(ValueError, match="CLEANUP_UNKNOWN"):
            await SourceSession._sdk_cleanup_for(source, assignment)
        receipt = await SourceSession._sdk_cleanup_for(source, assignment)
        assert receipt["source_id"] == "original-stream"
        assert receipt["session_id"] == "original-session"
        assert receipt["generation_id"] == "original-generation"
        assert factory.retire_failed_start.await_count == 2

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["transient", "permanent", "admission_inflight"])
def test_source_close_recovers_original_cleanup_without_another_admission(
    tmp_path, mode
):
    from test_player_lifecycle import system

    from sglang.multimodal_gen.runtime.realtime.coordinator import CoordinatorRejected
    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import ModelLeases
    from sglang.multimodal_gen.runtime.realtime.whitebox.model_binding import (
        ModelBinding,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
        RenderJournal,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig

    async def run():
        permanent = mode == "permanent"
        inflight = mode == "admission_inflight"
        _, _, worker, coordinator, http, admit = await system("memory")
        archives = AsyncArchiveManager()
        try:
            assignment = None
            if not inflight:
                assignment = await admit("original-player")
                await worker.consume(
                    assignment.token,
                    session_id=assignment.session_id,
                    generation_id=assignment.generation_id,
                    worker_epoch="e",
                    owner_id="original-socket",
                )
                # The model stopped, but neither owner has confirmed release.
                await worker.release(assignment.token, owner_id="original-socket")
                await coordinator.quarantine(assignment)
            journal = RenderJournal(
                MemoryRoomStore(),
                user_id="u",
                source_session_id="source",
                source_epoch="source-epoch",
                owner_epoch="room-epoch",
            )
            await journal.create()
            for operation in ("source.close", "source.unbind"):
                await journal.command(
                    {
                        "type": "command",
                        "op": operation,
                        "request_id": operation,
                        "source_session_id": "source",
                        "source_epoch": "source-epoch",
                    },
                    lambda *_: None,
                )

            async def retire():
                if permanent or journal.snapshot.get("cleanup_state") != "quarantined":
                    raise ValueError("SDK_MEMBER_CLEANUP_UNKNOWN")
                return {
                    "state": "not_created",
                    "source_id": "original-stream",
                    "owner_epoch": "room-epoch",
                }

            factory = NS(retire_failed_start=AsyncMock(side_effect=retire))
            source = SourceSession(
                NS(runtime=NS(players={})),
                journal,
                source_epoch="source-epoch",
                user_id="u",
                session_id="source",
                trace_id="original",
                config=WhiteboxConfig(
                    engine_factory=factory,
                    ui_factory=None,
                    spool=tmp_path,
                    spec=VideoSpec(32, 32),
                    room_owned=True,
                ),
                archives=archives,
            )
            source._sdk_create_started = True
            source._sdk_create_done.set()
            lease = ModelLeases(
                coordinator,
                AsyncMock(),
                user_id="u",
                model_revision="m",
                vae_fingerprint="local",
                trace_id="original",
                max_bytes=1024,
                strict_release=True,
                cleanup_timeout_s=0.2,
            )
            lease.sdk_cleanup_provider = source._sdk_cleanup_for
            binding = source.model_binding = ModelBinding(lease, lambda *_: None)
            original_admit = coordinator.admit
            admitted, reply = asyncio.Event(), asyncio.Event()

            async def delayed_admit(**request):
                nonlocal assignment
                request.pop("trace_id")  # HTTP client consumes the trace header.
                assignment = await original_admit(**request)
                admitted.set()
                await reply.wait()
                return assignment

            if inflight:
                coordinator.admit = AsyncMock(side_effect=delayed_admit)
                binding.start()
                await asyncio.wait_for(admitted.wait(), 1)
                assert binding.assignment is None and lease.resource_state == "none"
            else:
                lease._cleanup_assignment = binding.assignment = assignment
                lease.resource_state = "quarantined"
                binding.closed = True
                binding.task = asyncio.create_task(asyncio.sleep(0))
                await binding.task
            attempts = []
            original_close = source._close_source

            async def observed_close(**kwargs):
                attempts.append(source.config.engine_factory)
                await original_close(**kwargs)

            source._close_source = observed_close

            # Concurrent teardown callers must share the original cleanup flight.
            calls = [asyncio.create_task(source.close()) for _ in range(2)]
            if inflight:
                async with asyncio.timeout(1):
                    while not binding.task.cancelling():
                        await asyncio.sleep(0.001)
                reply.set()  # Original admit wins the cancellation race.
            async with asyncio.timeout(1):
                while journal.snapshot.get("cleanup_state") != "quarantined":
                    await asyncio.sleep(0.001)
            assert all(
                op["state"] == "accepted"
                for op in journal.snapshot["operations"].values()
            )
            # A disappearing browser/caller must not cancel backend recovery.
            calls[0].cancel()
            results = await asyncio.gather(*calls, return_exceptions=True)
            assert isinstance(results[0], asyncio.CancelledError)
            assert not source._source_close_task.cancelled()
            assert source.model_binding is binding
            assert (binding.assignment or lease._cleanup_assignment) is assignment
            lease.connect.assert_not_called()
            if inflight:
                coordinator.admit.assert_awaited_once()
                coordinator.admit = original_admit
            if permanent:
                assert isinstance(results[1], RenderError)
                assert attempts == [factory] * 3
                assert lease.resource_state != "released"
                assert journal.snapshot["cleanup_state"] == "quarantined"
                with pytest.raises(CoordinatorRejected):
                    await admit("next-player")
            else:
                assert results[1] is None
                assert attempts == [factory] * 2
                assert factory.retire_failed_start.await_count >= 2
                assert journal.snapshot["cleanup_state"] == "released"
                assert lease.resource_state == "released"
                following = await admit("next-player")
                # A late old closer cannot release the next player's reservation.
                await source.close()
                assert (await worker.snapshot())["reserved_sessions"] == 1
                assert following.token != assignment.token
            for op in journal.snapshot["operations"].values():
                assert op["state"] == ("failed" if permanent else "completed")
                assert op["resource_state"] == (
                    "quarantined" if permanent else "released"
                )
                assert op.get("code") == ("RELEASE_UNKNOWN" if permanent else None)
        finally:
            await archives.close()
            await http.aclose()

    asyncio.run(run())


def test_source_close_does_not_retry_unverified_engine_cleanup(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
        RenderJournal,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig

    async def run():
        archives = AsyncArchiveManager()
        journal = RenderJournal(
            MemoryRoomStore(),
            user_id="u",
            source_session_id="source",
            source_epoch="epoch",
            owner_epoch="owner",
        )
        await journal.create()
        source = SourceSession(
            NS(runtime=NS(players={})),
            journal,
            source_epoch="epoch",
            user_id="u",
            session_id="source",
            trace_id="source",
            config=WhiteboxConfig(
                engine_factory=None,
                ui_factory=None,
                spool=tmp_path,
                spec=VideoSpec(32, 32),
                room_owned=True,
            ),
            archives=archives,
        )
        source.engine = NS(close=AsyncMock(side_effect=ValueError("cleanup failed")))
        try:
            with pytest.raises(RenderError, match="cleaned up"):
                await source.close()
            source.engine.close.assert_awaited_once()
            assert journal.snapshot["cleanup_state"] == "quarantined"
        finally:
            await archives.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "bad", ["missing", "factory_only", "wrong_operation", "wrong_owner", "cancelled"]
)
def test_unproven_startup_keeps_sdk_cleanup_unknown(bad):
    async def run():
        cleanup = RoomCleanupReceipt(FactoryCleanupReceipt("original"), "owner")
        if bad == "factory_only":
            cleanup = cleanup.factory
        if bad == "wrong_operation":
            cleanup = RoomCleanupReceipt(FactoryCleanupReceipt("different"), "owner")
        if bad == "wrong_owner":
            cleanup = RoomCleanupReceipt(cleanup.factory, "old-owner")
        error = (
            RuntimeError("lost reply")
            if bad == "missing"
            else (
                asyncio.CancelledError()
                if bad == "cancelled"
                else RoomStartupFailure("CLI_STARTUP_FAILED", cleanup)
            )
        )
        runtime = NS(
            epoch="owner",
            join=AsyncMock(side_effect=error),
            retire_member=AsyncMock(
                side_effect=ValueError("SDK_MEMBER_CLEANUP_UNKNOWN")
            ),
        )
        factory = LocalRoomFactory(runtime, new_id(), new_id(), new_id())
        with pytest.raises(type(error)):
            await factory.open_session(
                world_id="world",
                spec=VideoSpec(32, 32),
                trace_id="original",
                context={"session_id": new_id(), "user_id": "u"},
            )
        with pytest.raises(ValueError, match="CLEANUP_UNKNOWN"):
            await factory.retire_failed_start()

    asyncio.run(run())


@pytest.mark.parametrize("point", ["before-factory", "directory", "changed-owner"])
def test_cancel_before_sdk_invocation_uses_original_owner_scope(
    tmp_path, monkeypatch, point
):
    async def run():
        directory = RoomDirectory(MemoryRoomStore(), {"room": "http://room"})
        entered = asyncio.Event()

        async def resolve(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(directory, "resolve", resolve)
        driver = AsyncMock()
        runtime = RoomRuntime(
            directory=directory,
            owner_id="room",
            driver_factory=driver,
            spool=tmp_path,
            spec=VideoSpec(32, 32),
            archives=AsyncArchiveManager(),
        )
        stream = new_id()
        factory = LocalRoomFactory(runtime, new_id(), new_id(), stream)
        task = asyncio.create_task(
            factory.open_session(
                world_id="world",
                spec=VideoSpec(32, 32),
                trace_id="original",
                context={"session_id": new_id(), "user_id": "u"},
            )
        )
        try:
            if point != "before-factory":
                await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            driver.assert_not_called()
            if point == "changed-owner":
                runtime.epoch = "replacement-owner"
                with pytest.raises(ValueError, match="CLEANUP_UNKNOWN"):
                    await factory.retire_failed_start()
            else:
                receipt = await factory.retire_failed_start()
                assert receipt == dict(
                    state="not_created", source_id=stream, owner_epoch=runtime.epoch
                )
            assert not runtime.rooms and not runtime.players
        finally:
            await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "point",
    [
        "running",
        "before-spawn",
        "spawn-return",
        "repeat-cleanup",
        "repeat-room-close",
        "timeout",
        "cleanup-error",
        "resident",
        "resident-confirmed",
        "resident-early",
        "resident-wrong-request",
    ],
)
def test_cancelled_startup_requires_positive_cleanup_before_next_player(
    tmp_path, monkeypatch, point
):
    from test_player_lifecycle import system

    from sglang.multimodal_gen.runtime.realtime.coordinator import CoordinatorRejected
    from sglang.multimodal_gen.runtime.realtime.whitebox import worldkit_room

    async def run():
        entry = tmp_path / "entry.mjs"
        entry.touch()
        monkeypatch.setenv("WHITEBOX_CLI_ENTRYPOINT", str(entry))
        monkeypatch.setenv("WHITEBOX_CLI_UI_FACTORY", "builtins:object")
        monkeypatch.setenv("WHITEBOX_CLI_RESOURCE_TRANSPORT", "cdn")
        monkeypatch.setenv(
            "WHITEBOX_CLI_SERVICE_MODE",
            "auto" if point.startswith("resident") else "off",
        )
        native_ready = tmp_path / "native-request-ready"
        native_allowed = tmp_path / "allow-native-request"
        child = "import time;time.sleep(120)"
        if point in {"resident-confirmed", "resident-early", "resident-wrong-request"}:
            request_id = "11111111-1111-4111-8111-111111111111"
            reply_id = (
                request_id
                if point in {"resident-confirmed", "resident-early"}
                else "22222222-2222-4222-8222-222222222222"
            )
            child = f"""
import signal,sys,json,pathlib,time
while {point == "resident-early"!r} and not pathlib.Path({str(native_allowed)!r}).exists(): time.sleep(0.005)
def close(*args):
    print(json.dumps(dict(kind='worldkit-service-cleanup',schemaVersion=1,
        protocol='worldkit-service-cleanup-v1',requestId={reply_id!r},
        state='retired',origin='service')),file=sys.stderr,flush=True)
    raise SystemExit(143)
signal.signal(signal.SIGTERM,close)
print(json.dumps(dict(kind='worldkit-service-request',schemaVersion=1,
    protocol='worldkit-service-cleanup-v1',requestId={request_id!r})),file=sys.stderr,flush=True)
pathlib.Path({str(native_ready)!r}).touch()
while True: signal.pause()
"""
        monkeypatch.setattr(
            cli,
            "cli_arguments",
            lambda *a, **k: [sys.executable, "-c", child],
        )
        spawned, spawn_return = asyncio.Event(), asyncio.Event()
        cleanup_entered, cleanup_return = asyncio.Event(), asyncio.Event()
        terminate_entered = asyncio.Event()
        spawn_requested, spawn_allowed = asyncio.Event(), asyncio.Event()
        room_close_entered, room_close_return = asyncio.Event(), asyncio.Event()
        if point != "before-spawn":
            spawn_allowed.set()
        if point != "repeat-room-close":
            room_close_return.set()
        if point != "spawn-return":
            spawn_return.set()
        if point != "repeat-cleanup":
            cleanup_return.set()
        original_create, original_stop = (
            asyncio.create_subprocess_exec,
            worldkit_room.stop_process,
        )
        original_terminate = cli.CLIRoom._terminate_cli

        async def terminate(room):
            terminate_entered.set()
            await original_terminate(room)

        monkeypatch.setattr(cli.CLIRoom, "_terminate_cli", terminate)
        processes = []

        async def create(*args, **kwargs):
            import os

            inherited = {fd: os.fstat(fd).st_ino for fd in kwargs.get("pass_fds", ())}
            spawn_requested.set()
            await spawn_allowed.wait()
            assert all(os.fstat(fd).st_ino == inode for fd, inode in inherited.items())
            process = await original_create(*args, **kwargs)
            processes.append(process)
            spawned.set()
            await spawn_return.wait()
            return process

        async def stop(process):
            cleanup_entered.set()
            await cleanup_return.wait()
            await original_stop(process)
            if point == "cleanup-error":
                raise RuntimeError("owned cleanup acknowledgement failed")

        monkeypatch.setattr(cli.asyncio, "create_subprocess_exec", create)
        monkeypatch.setattr(worldkit_room, "stop_process", stop)
        directory = RoomDirectory(MemoryRoomStore(), {"room": "http://room"})
        game, player, stream = new_id(), new_id(), new_id()
        await directory.resolve(game, "world", create=True)
        runtime = RoomRuntime(
            directory=directory,
            owner_id="room",
            driver_factory=cli.cli_room,
            spool=tmp_path,
            spec=VideoSpec(32, 32),
            archives=AsyncArchiveManager(),
        )
        original_end = runtime._end

        async def end(*args, **kwargs):
            room_close_entered.set()
            await room_close_return.wait()
            return await original_end(*args, **kwargs)

        monkeypatch.setattr(runtime, "_end", end)
        factory = LocalRoomFactory(runtime, game, player, stream)
        _, _, worker, coordinator, http, admit = await system("memory")
        assignment = await admit("cancelled-player")
        await worker.consume(
            assignment.token,
            session_id=assignment.session_id,
            generation_id=assignment.generation_id,
            worker_epoch="e",
            owner_id="original-socket",
        )
        task = asyncio.create_task(
            factory.open_session(
                world_id="world",
                spec=VideoSpec(32, 32),
                trace_id="cancelled-startup",
                context={"session_id": new_id(), "user_id": "u"},
            )
        )
        try:
            await asyncio.wait_for(
                (spawn_requested if point == "before-spawn" else spawned).wait(), 5
            )
            if point in {"resident-confirmed", "resident-wrong-request"}:
                async with asyncio.timeout(5):
                    while not native_ready.exists():
                        await asyncio.sleep(0.01)
            if point == "timeout":
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(0):
                        await task
            else:
                task.cancel()
                if point == "resident-early":
                    await asyncio.wait_for(terminate_entered.wait(), 5)
                    native_allowed.touch()
                if point == "before-spawn":
                    spawn_allowed.set()
                if point == "spawn-return":
                    spawn_return.set()
                if point == "repeat-cleanup":
                    await asyncio.wait_for(cleanup_entered.wait(), 5)
                    task.cancel()
                    cleanup_return.set()
                if point == "repeat-room-close":
                    await asyncio.wait_for(room_close_entered.wait(), 5)
                    task.cancel()
                    room_close_return.set()
                expected = (
                    RuntimeError if point == "cleanup-error" else asyncio.CancelledError
                )
                with pytest.raises(expected):
                    await task
            await worker.release(assignment.token, owner_id="original-socket")
            if point in {"cleanup-error", "resident", "resident-wrong-request"}:
                with pytest.raises(ValueError, match="CLEANUP_UNKNOWN"):
                    await factory.retire_failed_start()
                with pytest.raises(CoordinatorRejected, match="SDK_CLEANUP_PENDING"):
                    await coordinator.release_verified(assignment)
                with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
                    await admit("next-player")
                return
            assert task.cancelled()
            assert processes[0].returncode is not None
            assert not runtime.rooms and not runtime.players
            source = NS(
                engine=None,
                config=NS(engine_factory=factory),
                _sdk_retire_task=None,
                _sdk_create_done=asyncio.Event(),
                _sdk_create_started=True,
                model_binding=object(),
            )
            source._sdk_create_done.set()
            receipt = await SourceSession._sdk_cleanup_for(source, assignment)
            await coordinator.release_verified(assignment, sdk_cleanup=receipt)
            assert (await worker.snapshot())["active_sessions"] == 0
            following = await admit("next-player")
            assert following.token != assignment.token
        finally:
            native_allowed.touch()
            spawn_return.set()
            spawn_allowed.set()
            cleanup_return.set()
            room_close_return.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            for process in processes:
                await original_stop(process)
            await runtime.close()
            await http.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "request",
        "protocol",
        "schema",
        "origin",
        "client-retired",
        "before-request",
    ],
)
def test_native_cleanup_receipt_requires_the_same_private_cli_request(bad):
    request_id = "11111111-1111-4111-8111-111111111111"
    request = dict(
        kind="worldkit-service-request",
        schemaVersion=1,
        protocol="worldkit-service-cleanup-v1",
        requestId=request_id,
    )
    reply = dict(
        kind="worldkit-service-cleanup",
        schemaVersion=1,
        protocol="worldkit-service-cleanup-v1",
        requestId=request_id,
        state="retired",
        origin="service",
    )
    receipt = {}
    if bad != "before-request":
        assert cli.observe_service_cleanup(request, receipt)
    if bad == "request":
        reply["requestId"] = "22222222-2222-4222-8222-222222222222"
    elif bad == "protocol":
        reply["protocol"] = "other"
    elif bad == "schema":
        reply["schemaVersion"] = 2
    elif bad == "origin":
        reply["origin"] = "browser"
    elif bad == "client-retired":
        reply["origin"] = "client"
    assert cli.observe_service_cleanup(reply, receipt)
    assert bool(receipt.get("confirmed") and not receipt.get("invalid")) == (
        bad is None
    )


@pytest.mark.parametrize("confirmed", [False, True])
def test_ready_resident_cli_exit_zero_does_not_replace_its_cleanup_receipt(confirmed):
    async def run():
        stderr = asyncio.StreamReader()
        request_id = "11111111-1111-4111-8111-111111111111"
        records = [
            dict(
                kind="worldkit-service-request",
                schemaVersion=1,
                protocol="worldkit-service-cleanup-v1",
                requestId=request_id,
            )
        ]
        if confirmed:
            records.append(
                dict(
                    kind="worldkit-service-cleanup",
                    schemaVersion=1,
                    protocol="worldkit-service-cleanup-v1",
                    requestId=request_id,
                    state="retired",
                    origin="service",
                )
            )
        for value in records:
            stderr.feed_data((json.dumps(value) + "\n").encode())
        stderr.feed_eof()
        process = NS(returncode=0, stderr=stderr)
        room = cli.CLIRoom(
            process, {"sessionId": "original"}, VideoSpec(32, 32), None, "trace"
        )
        room.resident_service = True
        room.diagnostics = asyncio.create_task(
            cli.cli_diagnostics(process, "trace", service_cleanup=room.service_cleanup)
        )
        if confirmed:
            await room.close()
        else:
            with pytest.raises(ValueError, match="SDK_SERVICE_CLEANUP_UNKNOWN"):
                await room.close()

    asyncio.run(run())
