# SPDX-License-Identifier: Apache-2.0
"""A lost native CLI must not strand the single Room admission slot."""

import asyncio
from types import SimpleNamespace

import pytest
from test_whitebox_interactive import BrowserUI, Engine
from test_whitebox_room_pipeline import until

from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
    AsyncArchiveManager,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import CLIView
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import RenderHost
from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports
from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import RoomPipeline
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
    new_id,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig


@pytest.mark.parametrize("cause", ["owner_expired", "producer_cancelled"])
def test_expired_native_owner_releases_source_and_accepts_next_session(tmp_path, cause):
    async def run():
        spec = VideoSpec(32, 32)
        engines = []

        class NativeEngine(Engine):
            _wait_or_error = CLIView._wait_or_error

            def __init__(self):
                super().__init__(spec)
                self.renderer = BrowserUI()
                self.control_task = asyncio.create_task(asyncio.Event().wait())

            async def frames(self):
                while (
                    frame := await self._wait_or_error(self.queue.get())
                ) is not None:
                    yield frame

            async def close(self):
                self.closed = True
                self.control_task.cancel()
                await asyncio.gather(self.control_task, return_exceptions=True)

        class Driver:
            max_players = 1

            async def join(self, **_):
                engine = NativeEngine()
                engines.append(engine)
                return engine

            async def close(self):
                pass

        async def factory(**_):
            return Driver()

        archives = AsyncArchiveManager(recorder_slots=3)
        directory = RoomDirectory(MemoryRoomStore(), {"room": "http://room"})
        runtime = RoomRuntime(
            directory=directory,
            owner_id="room",
            driver_factory=factory,
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
            grace_seconds=2,
        )

        async def attach():
            game = new_id()
            await directory.resolve(game, "world", create=True)
            owner, sub = await host.attach(
                dict(
                    user_id="user",
                    trace_id=new_id(),
                    message=dict(
                        type="init",
                        prompt="world",
                        control_protocol="whitebox-render-v2",
                        source_operation="create",
                        whitebox={"game_id": game, "world_id": "world"},
                    ),
                )
            )
            await until(sub, lambda m: m["type"] == "session.ready")
            return owner, sub

        try:
            owner, sub = await attach()
            owner.detach(sub)
            if cause == "owner_expired":
                runtime.players[owner.stream_id]["touched"] -= 61
            else:
                producer = next(
                    t
                    for t in owner.session.tasks
                    if t.get_name() == "whitebox-engine-source"
                )
                producer.cancel()
            async with asyncio.timeout(5):
                while not owner.closed:
                    await asyncio.sleep(0.01)
            assert engines[0].closed and not runtime.players and not host.sessions
            state = await host.store.find_source(owner.session_id)
            assert state["source_state"] == "ended"
            assert state["source_outcome"] == "failed"
            assert owner.report.gateway_done.is_set()
            following, _ = await attach()
            assert following.session_id != owner.session_id and len(engines) == 2
        finally:
            await host.close()
            await runtime.close()
            await archives.close()

    asyncio.run(run())


def test_cleanup_error_frees_room_registry_but_retains_unknown_owner_and_error():
    async def run():
        async def fail():
            raise RuntimeError("encoder cleanup failed")

        owner = object.__new__(RoomPipeline)
        owner.closed = False
        owner.member_closed = asyncio.Event()
        owner.task = owner.commands = owner.retention = None
        owner.session = SimpleNamespace(close=fail)
        owner.session_id = "source"
        owner.host = SimpleNamespace(sessions={"source": owner})
        owner.init = {"private": True}
        owner.history = [1]
        owner.history_bytes = 1
        owner.input_history = {1: 1}
        owner.report = SimpleNamespace(gateway_done=asyncio.Event(), recording=object())
        with pytest.raises(RuntimeError, match="encoder cleanup failed"):
            await owner.close()
        assert owner.closed and not owner.host.sessions and not owner.init
        assert owner.report.gateway_done.is_set()
        assert owner.host.retiring_sessions["source"] is owner
        with pytest.raises(RuntimeError, match="encoder cleanup failed"):
            await owner.close()

    asyncio.run(run())


def test_cancel_startup_stops_pending_cli_and_fences_a_delayed_join(tmp_path):
    import time
    from dataclasses import asdict

    async def run():
        started, stopped = asyncio.Event(), asyncio.Event()

        async def factory(**_):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

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
        )
        game, stream = new_id(), new_id()
        await directory.resolve(game, "world", create=True)
        body = dict(
            game_id=game,
            player_id=new_id(),
            stream_id=stream,
            session_id=new_id(),
            user_id="u",
            world_id="world",
            trace_id="startup",
            spec=asdict(spec),
            startup_prepare=True,
            startup_expires_at=time.time() + 60,
        )
        pending = asyncio.create_task(runtime.join(body))
        try:
            await asyncio.wait_for(started.wait(), 5)
            await runtime.cancel_startup(stream)
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert (
                stopped.is_set()
                and not runtime.rooms
                and not runtime.players
                and not runtime.pending_startups
            )
            with pytest.raises(ValueError, match="no longer valid"):
                await runtime.join(body)
            other = new_id()
            await runtime.cancel_startup(other)
            with pytest.raises(ValueError, match="no longer valid"):
                await runtime.join({**body, "stream_id": other})
        finally:
            await runtime.close()
            await archives.close()

    asyncio.run(run())


def test_startup_http_join_cancellation_requests_exact_owner_cleanup():
    import httpx
    import msgspec

    from sglang.multimodal_gen.runtime.realtime.whitebox.room_client import (
        RoomEngineFactory,
    )

    async def run():
        joined = asyncio.Event()
        seen = []
        body = {}

        async def handle(request):
            seen.append(request.url.path)
            if request.url.path.endswith("/join"):
                import json

                body.update(json.loads(request.content))
                joined.set()
                await asyncio.Event().wait()
            return httpx.Response(
                200, content=msgspec.msgpack.encode({"cancelled": True})
            )

        async def resolve(*args, **kwargs):
            return dict(owner_url="http://room", owner_id="room", owner_epoch=new_id())

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        factory = RoomEngineFactory(
            token="test-token",
            client=client,
            directory=SimpleNamespace(resolve=resolve),
        )
        task = asyncio.create_task(
            factory.open_session(
                world_id="world",
                spec=VideoSpec(32, 32),
                trace_id="t",
                context=dict(session_id=new_id(), user_id="u", startup_prepare=True),
            )
        )
        try:
            await joined.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert seen == [
                "/internal/players/join",
                "/internal/players/" + body["stream_id"] + "/startup_cancel",
            ]
            assert body["startup_prepare"] is True and body["startup_expires_at"] > 0
        finally:
            await client.aclose()

    asyncio.run(run())


def test_startup_engine_result_that_wins_cancel_race_is_closed():
    from sglang.multimodal_gen.runtime.realtime.whitebox.startup import (
        open_startup_engine,
    )

    async def run():
        closed = []

        async def close():
            closed.append("engine")

        async def renderer_close():
            closed.append("renderer")

        engine = SimpleNamespace(
            close=close, renderer=SimpleNamespace(close=renderer_close)
        )

        async def factory():
            asyncio.get_running_loop().call_soon(owner.cancel)
            return engine

        owner = asyncio.create_task(open_startup_engine(factory, {}, 1))
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert closed == ["engine", "renderer"]

    asyncio.run(run())


def test_startup_join_cleanup_failure_is_not_hidden_by_outer_cancellation():
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.startup import (
        open_startup_engine,
    )

    async def run():
        entered = asyncio.Event()

        async def factory():
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise RenderError("RELEASE_UNKNOWN") from None

        owner = asyncio.create_task(open_startup_engine(factory, {}, 1))
        await entered.wait()
        owner.cancel()
        with pytest.raises(RenderError) as result:
            await owner
        assert result.value.code == "RELEASE_UNKNOWN"

    asyncio.run(run())
