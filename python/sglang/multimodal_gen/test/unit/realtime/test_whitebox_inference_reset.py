"""Authenticated native reset retains intent and creates fresh CPU model children."""

import asyncio
import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import msgspec
import pytest
from test_whitebox_interactive import BrowserUI, Engine
from test_whitebox_native import native_frame
from test_whitebox_render_contract import command, setup
from test_whitebox_render_lifecycle import StrictLeases

from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
    AsyncArchiveManager,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import RenderError
from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig


@pytest.mark.parametrize(
    "outcome",
    [
        "enabled",
        "double-reset",
        "display-conflict",
        "disable",
        "close",
        "cleanup-failed",
        "disabled",
    ],
)
def test_ordinary_native_reset_reopens_only_after_cleanup_and_fresh_frame(
    tmp_path, monkeypatch, outcome
):
    async def run():
        monkeypatch.delenv("WHITEBOX_BOUND_COMPUTE", raising=False)
        spec, descriptor, _ = native_frame()
        native = NativeCLIView(
            SimpleNamespace(descriptor=descriptor, spec=spec, prepared_delivery=None)
        )
        native.source_identity = ("before", 1)
        native.renderer = SimpleNamespace(consume_ui=AsyncMock())
        native.control = SimpleNamespace(send=AsyncMock())

        class Source(Engine):
            stream_id = "view"
            metadata = {}

            def bind_source_reset(self, callback):
                native.bind_source_reset(callback)

            async def push(self, identity=None):
                i = self.next
                self.next += 1
                presentation, epoch = identity or native.source_identity
                from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
                    EngineFrame,
                )

                await self.queue.put(
                    EngineFrame(
                        i,
                        spec.pts(i),
                        bytes([30]) * spec.frame_bytes,
                        {},
                        provenance=dict(
                            capture_pts_us=i * 50000,
                            cli_frame_header=dict(
                                source=dict(
                                    presentationId=presentation, sdkEpoch=epoch
                                ),
                            ),
                        ),
                    )
                )

        engine = Source(spec)

        class FreshLeases(StrictLeases):
            @asynccontextmanager
            async def open(self):
                async with super().open() as (worker, assignment):
                    assignment.session_id += str(len(self.opens))
                    assignment.generation_id += str(len(self.opens))
                    yield worker, assignment

        leases = FreshLeases(spec, "clip-barrier-v1")
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
            user_id="user",
            session_id="source",
            trace_id="reset",
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
        running = asyncio.Event()
        running.set()
        messages = []

        async def consume():
            while True:
                m = msgspec.msgpack.decode(await source.recv())
                messages.append(m)
                if m.get("type") in {"whitebox.frame", "frame_batch"}:
                    source.media_delivered()

        async def produce():
            while True:
                await running.wait()
                await engine.push()
                await asyncio.sleep(0.025)

        async def wait(predicate):
            async with asyncio.timeout(8):
                while not predicate():
                    await asyncio.sleep(0.005)

        tasks, closing = [], None
        release = asyncio.Event()
        old_stop = None
        try:
            await source.send(
                msgspec.msgpack.encode(
                    dict(
                        type="init",
                        source_operation="create.manual",
                        inference_control="manual",
                        whitebox={"world_id": "world"},
                        prompt="Frozen scene.",
                        seed=19,
                    )
                )
            )
            tasks = [asyncio.create_task(consume()), asyncio.create_task(produce())]
            await wait(lambda: source.ready.is_set() and source.retained.frames)
            if outcome != "disabled":
                await source.command(
                    command(
                        "model.inference.configure",
                        "original-enable",
                        enabled=True,
                        expected_inference_revision=0,
                    )
                )
                await wait(lambda: journal.snapshot["inference"]["state"] == "enabled")
                await wait(lambda: source.current_segment.first_decodable)
                historical = copy.deepcopy(
                    journal.snapshot["operations"]["original-enable"]
                )
                first = source.current_segment
                old_ids = {
                    k: journal.snapshot["segment"][k]
                    for k in ("segment_id", "model_session_id", "generation_id")
                }
                # A segment-local user supplement must not be replayed after reset.
                await source.command(
                    command(
                        "model.context.intervention",
                        "old-supplement",
                        segment_id=first.segment_id,
                        segment_epoch=first.segment_epoch,
                        expectedRevision=0,
                        prompt="Old supplement.",
                    )
                )
                await wait(lambda: journal.snapshot["segment"]["applied_version"] == 2)
                entered = asyncio.Event()
                old_stop = first.stop

                async def stop(**kwargs):
                    entered.set()
                    await release.wait()
                    await old_stop(**kwargs)
                    if outcome == "cleanup-failed":
                        first.resource_state = "release_unknown"
                        raise RenderError("RELEASE_UNKNOWN")

                first.stop = stop
            running.clear()
            await asyncio.sleep(0.04)
            # A preview encoder can lag one unpublished frame while holding its
            # previous frame/credit. Reset must preserve LatestFrame's pair API.
            preview_cursor = source.latest.version - 1
            reset = dict(
                type="source.reset",
                epoch=descriptor["epoch"],
                presentationId="after",
                sdkEpoch=2,
            )
            await native._source_reset(reset)
            version, preview = await asyncio.wait_for(
                source.latest.get(preview_cursor), 1
            )
            assert version > preview_cursor and preview[0].rgb
            if outcome == "disabled":
                await source.native_reset_task
                assert journal.snapshot["inference"] == dict(
                    enabled=False, revision=0, state="disabled"
                )
                assert not leases.opens
                return
            await wait(
                lambda: journal.snapshot["inference"].get("phase") == "preparing_source"
            )
            await entered.wait()
            assert len(leases.opens) == 1 and not source.inference.recovery_task.done()
            assert not source.retained.frames
            if outcome == "double-reset":
                native.pending_reset = (
                    None  # The external native media reconfiguration completed.
                )
                reset.update(presentationId="latest", sdkEpoch=3)
                await native._source_reset(reset)
            if outcome == "disable":
                await source.command(
                    command(
                        "model.inference.configure",
                        "disable-during-reset",
                        enabled=False,
                        expected_inference_revision=1,
                    )
                )
            if outcome == "close":
                closing = asyncio.create_task(source.close())
                await asyncio.sleep(0)
                assert not closing.done()
            conflict = []
            if outcome == "display-conflict":
                original_cas = journal.store.cas

                async def raced_cas(key, old, new, retention, **kwargs):
                    before, after = json.loads(old), json.loads(new)
                    if (
                        not conflict
                        and before["display_mode"] == "model"
                        and after["display_mode"] == "whitebox"
                    ):
                        conflict.append(True)
                        await journal.display(
                            command(
                                "display.configure",
                                "explicit-raw",
                                mode="whitebox",
                                expected_display_revision=before["display_revision"],
                            )
                        )
                        return False
                    return await original_cas(key, old, new, retention, **kwargs)

                journal.store.cas = raced_cas
            release.set()
            if outcome == "cleanup-failed":
                with pytest.raises(RenderError):
                    await asyncio.wait_for(asyncio.shield(source.native_reset_task), 3)
            else:
                await asyncio.wait_for(asyncio.shield(source.native_reset_task), 3)
            if outcome == "close":
                await asyncio.wait_for(closing, 3)
                assert len(leases.opens) == 1
                return
            if outcome == "cleanup-failed":
                await wait(lambda: journal.snapshot["inference"]["state"] == "failed")
                assert journal.snapshot["inference"] == dict(
                    enabled=True, revision=1, state="failed", code="RELEASE_UNKNOWN"
                )
                assert len(leases.opens) == 1
                return
            if outcome == "disable":
                await wait(lambda: journal.snapshot["inference"]["state"] == "disabled")
                running.set()
                await asyncio.sleep(0.1)
                assert len(leases.opens) == 1
                return
            # A stale in-flight capture must not refill the F0 queue after reset.
            await engine.push(("before", 1))
            await asyncio.sleep(0.05)
            assert not source.retained.frames and len(leases.opens) == 1
            fresh_index = engine.next
            running.set()
            await wait(lambda: journal.snapshot["inference"]["state"] == "enabled")
            await wait(lambda: source.current_segment.first_decodable)
            assert len(leases.opens) == 2 and leases.active == 1
            assert leases.opens[1].init["prompt"] == "Frozen scene."
            assert leases.opens[1].init["seed"] == 19
            assert all(journal.snapshot["segment"][k] != v for k, v in old_ids.items())
            assert (
                source.current_segment.recording.manifest["start_boundary_actual"][
                    "source_frame_index"
                ]
                >= fresh_index
            )
            assert journal.snapshot["operations"]["original-enable"] == historical
            assert journal.snapshot["inference"] == dict(
                enabled=True, revision=1, state="enabled"
            )
            # Model delivery is independent of display selection. The original
            # Player display controller may select it after admitting new pixels.
            assert source.display_mode == journal.snapshot["display_mode"] == "whitebox"
            if outcome == "display-conflict":
                assert conflict == [True]
                assert (
                    journal.snapshot["operations"]["explicit-raw"]["state"] == "applied"
                )
            assert (
                len(
                    [
                        o
                        for o in journal.snapshot["operations"].values()
                        if o["op"] == "model.inference.configure"
                    ]
                )
                == 1
            )
        finally:
            release.set()
            if old_stop is not None:
                first.stop = old_stop
                if first.resource_state == "release_unknown":
                    first.resource_state = "released"
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            if outcome == "cleanup-failed":
                with pytest.raises(RenderError):
                    await source.close()
            else:
                await source.close()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await archives.close()

    asyncio.run(run())


@pytest.mark.parametrize("member", [False, True])
def test_ordinary_native_reset_binding_fences_removed_and_replaced_engine(member):
    from sglang.multimodal_gen.runtime.realtime.whitebox.player_media import (
        MemberEngine,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        LocalRoomEngine,
    )

    async def run():
        spec, descriptor, _ = native_frame()
        native = NativeCLIView(
            SimpleNamespace(descriptor=descriptor, spec=spec, prepared_delivery=None)
        )
        native.source_identity = ("original", 1)
        native.renderer = SimpleNamespace(consume_ui=AsyncMock())
        native.control = SimpleNamespace(send=AsyncMock())
        runtime = SimpleNamespace(players={"stream": {"engine": native}})
        runtime.player = lambda stream: runtime.players[stream]
        engine = object.__new__(MemberEngine if member else LocalRoomEngine)
        engine.runtime, engine.stream_id, engine.closed = runtime, "stream", False
        seen = []
        engine.bind_source_reset(seen.append)
        for epoch, state in [
            (2, "current"),
            (3, "replaced"),
            (4, "removed"),
            (5, "closed"),
        ]:
            native.pending_reset = (
                None  # External native media reconfiguration completed.
            )
            if state == "replaced":
                runtime.players["stream"]["engine"] = object()
            elif state == "removed":
                runtime.players.clear()
            elif state == "closed":
                runtime.players["stream"] = {"engine": native}
                engine.closed = True
            await native._source_reset(
                dict(
                    type="source.reset",
                    epoch=descriptor["epoch"],
                    presentationId=f"reset-{epoch}",
                    sdkEpoch=epoch,
                )
            )
        assert len(seen) == 1 and seen[0]["sdkEpoch"] == 2

    asyncio.run(run())
