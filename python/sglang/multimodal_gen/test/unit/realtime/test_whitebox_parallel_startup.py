"""Managed preparation uses the original bound worker before the CLI is ready."""

import asyncio
import hashlib
import io
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS
from uuid import uuid4

import msgspec
import pytest
from PIL import Image
from test_whitebox_model_binding import Lease, Socket
from test_whitebox_startup import V2_PROTOCOL, request

from sglang.multimodal_gen.runtime.realtime.role_pool.contracts import PoolError
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.disaggregated_binding import (
    DisaggregatedModelLeases,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
    InteractiveSession,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.model_binding import ModelBinding
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import RenderError
from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession


class PreparationSocket(Socket):
    def __init__(self):
        super().__init__()
        self.preparing = asyncio.Event()
        self.release_prepare = asyncio.Event()
        self.tasks = set()
        self.operations = set()
        self.commands = []

    async def send(self, raw):
        message = msgspec.msgpack.decode(raw)
        if message["type"] != "bound.data":
            if message["type"] == "bound.open":
                self.operations.add(message["operation_id"])
            elif message["type"] == "bound.cancel":
                self.operations.discard(message["operation_id"])
            return await super().send(raw)
        self.sent.append(message)
        command = msgspec.msgpack.decode(message["payload"])
        self.commands.append(command)
        operation = message["operation_id"]
        if command["type"] == "init":
            response = dict(
                type="session.ready",
                source_prepare_protocol=V2_PROTOCOL,
                source_prepare_profile_sha256="a" * 64,
                source_stream=command["source_stream"],
            )
        elif command["kind"] == "source.prepare":
            self.preparing.set()

            async def complete():
                await self.release_prepare.wait()
                if operation in self.operations:
                    self.reply(
                        operation,
                        dict(
                            type="source.prepared",
                            identity=command["payload"]["identity"],
                        ),
                    )

            task = asyncio.create_task(complete())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            return
        else:
            assert command["kind"] == "source.attach"
            response = dict(
                type="source.attached", identity=command["payload"]["identity"]
            )
        self.reply(operation, response)

    def reply(self, operation, response):
        self.incoming.put_nowait(
            dict(
                type="bound.data",
                operation_id=operation,
                payload=msgspec.msgpack.encode(response),
            )
        )

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await super().close()


async def managed_source(
    monkeypatch,
    *,
    with_frame=True,
    managed=True,
    cli_error=None,
    references=None,
    binding_error=None,
    admission_error=None,
    cleanup_unknown=False,
):
    monkeypatch.setenv("WHITEBOX_MODEL_STARTUP_PREPARE", "1")
    monkeypatch.setenv("WHITEBOX_BOUND_COMPUTE", "1")
    source = object.__new__(SourceSession)
    source.config = NS(
        spec=VideoSpec(32, 32),
        model_target=None,
        startup_seconds=2,
        engine_factory=None,
    )
    source.session_id, source.source_epoch = uuid4().hex, uuid4().hex
    source.expected_publication = {"publicationId": "sealed"} if managed else None
    source.member_prefill = source.startup = source.engine = None
    source.closing = source.binding_ended = False
    source._sdk_create_done = asyncio.Event()
    source.outgoing = asyncio.Queue()
    source._model_transport_ready = False
    lease = Lease()
    lease.ws = PreparationSocket()

    async def receipt(_):
        pass

    @asynccontextmanager
    async def open_lease():
        if admission_error is not None:
            lease.calls += 1
            if cleanup_unknown:
                lease.resource_state = "release_unknown"
            raise admission_error
        async with Lease.open(lease) as admitted:
            yield admitted

    monkeypatch.setattr(lease, "open", open_lease)

    def start_binding(_):
        if binding_error is not None:
            raise binding_error
        source.model_binding = ModelBinding(lease, receipt, timeout=2)
        source.segment_leases = source.model_binding.borrow

    source.start_model_binding = start_binding
    message = dict(
        type="init",
        source_operation="create.manual",
        inference_control="manual",
        whitebox={"world_id": "world"},
        startup=request(with_frame, protocol=V2_PROTOCOL),
    )
    if references is not None:
        message["startup"]["initial_render"]["references"] = references
    source.configure_resolution = lambda _: None

    async def initialize_inference(*, manual):
        assert manual

    source.journal = NS(initialize_inference=initialize_inference)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def cli(self, *args):
        entered.set()
        await finish.wait()
        if cli_error is not None:
            raise cli_error

    monkeypatch.setattr(InteractiveSession, "_open_engine", cli)
    init = dict(
        type="init",
        prompt="snow",
        seed=42,
        references=[],
        source_stream=dict(format="rgb24", width=32, height=32, fps=24),
    )

    async def start(self, message):
        await self._open_engine("world", message["whitebox"], init)

    monkeypatch.setattr(InteractiveSession, "_start", start)
    task = asyncio.create_task(source._start(message))
    return source, lease, task, entered, finish


def test_managed_preview_prepares_while_cli_is_blocked(monkeypatch):
    async def run():
        source, lease, task, entered, finish = await managed_source(monkeypatch)
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.wait_for(lease.ws.preparing.wait(), 1)
            assert not task.done() and not finish.is_set()
            assert lease.calls == 1 and lease.releases == 0
            assert source.inference.manual
            assert source.startup is None  # No automatic render/display transition.
            assert [m.get("kind") for m in lease.ws.commands] == [
                None,
                "source.prepare",
            ]
            assert (
                lease.ws.commands[-1]["payload"]["rgb"] == bytes([255, 0, 0]) * 32 * 32
            )
            lease.ws.release_prepare.set()
            await source.inference.preparation.wait(
                source.inference.preparation.prepared
            )
            assert not finish.is_set()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await source.inference.close()
            await source.model_binding.close()
        assert lease.calls == lease.releases == 1
        assert not lease.ws.operations

    asyncio.run(run())


@pytest.mark.parametrize("with_frame,managed", [(False, True), (True, False)])
def test_unrequested_or_unbound_preview_does_not_start_model(
    monkeypatch, with_frame, managed
):
    async def run():
        source, lease, task, entered, _ = await managed_source(
            monkeypatch, with_frame=with_frame, managed=managed
        )
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.sleep(0)
            assert not lease.ws.commands
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await source.inference.close()
            assert source.model_binding is None

    asyncio.run(run())


@pytest.mark.parametrize("unsupported_route", [False, True])
def test_preview_admission_rejection_allows_whitebox_start(
    monkeypatch, unsupported_route
):
    async def run():
        source, lease, task, entered, finish = await managed_source(
            monkeypatch,
            binding_error=(
                RenderError("CAPABILITY_UNAVAILABLE") if unsupported_route else None
            ),
            admission_error=(
                None if unsupported_route else PoolError("CAPACITY_EXHAUSTED")
            ),
        )
        finish.set()
        await task
        assert entered.is_set()
        assert source.inference.preparation is None
        assert source.model_binding is None and source.control_prefill is None
        assert lease.calls == (0 if unsupported_route else 1)
        assert not lease.ws.commands

    asyncio.run(run())


def test_preview_admission_with_unknown_cleanup_preserves_original_owner(monkeypatch):
    async def run():
        source, lease, task, entered, finish = await managed_source(
            monkeypatch,
            admission_error=PoolError("CAPACITY_EXHAUSTED"),
            cleanup_unknown=True,
        )
        finish.set()
        with pytest.raises(PoolError, match="CAPACITY_EXHAUSTED"):
            await task
        assert not entered.is_set()
        assert source.model_binding.lease is lease
        assert lease.resource_state == "release_unknown" and lease.calls == 1
        assert source.inference.preparation is None

    asyncio.run(run())


def test_rejected_role_pool_admission_closes_client_without_reserving_compute(
    monkeypatch,
):
    async def run():
        monkeypatch.delenv("WHITEBOX_PREFERRED_SERVICE_IDS", raising=False)
        lease = DisaggregatedModelLeases(
            None,
            None,
            user_id="fixture",
            model_revision="fixture",
            vae_fingerprint="local",
            trace_id="fixture",
            max_bytes=1024,
            operation_path="/v1/realtime_video/bound",
            config={
                "profile": "fixture",
                "compatibility": "fixture",
                "network_domain": "fixture",
                "player_ttl_seconds": 300,
            },
        )
        calls = []

        async def control(operation, _):
            calls.append(operation)
            if operation == "admit":
                raise PoolError("CAPACITY_EXHAUSTED")
            assert operation == "get"
            raise PoolError("ADMISSION_NOT_FOUND", 404)

        monkeypatch.setattr(lease, "control", control)
        try:
            with pytest.raises(PoolError, match="CAPACITY_EXHAUSTED"):
                async with lease.open():
                    pytest.fail("A rejected admission cannot own compute.")
            assert calls == ["admit", "get"]
            assert lease.resource_state == "none"
            assert lease.client.is_closed
        finally:
            await lease.client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("mismatch", [None, "order", "picture_gap", "bytes"])
def test_sdk_upload_ids_match_world_picture_ordinals(monkeypatch, mismatch):
    references = []
    for index, color in enumerate(("red", "blue")):
        stream = io.BytesIO()
        Image.new("RGB", (32, 32), color).save(stream, format="PNG")
        data = stream.getvalue()
        references.append(
            dict(
                id=f"reference-{index}",
                format="png",
                payload=data,
                encoded_size=len(data),
                encoded_sha256=hashlib.sha256(data).hexdigest(),
            )
        )

    async def run():
        source, lease, task, _, _ = await managed_source(
            monkeypatch, references=references
        )
        try:
            await asyncio.wait_for(lease.ws.preparing.wait(), 1)
            candidate = source.inference.preparation
            lease.ws.release_prepare.set()
            condition = dict(
                prompt="snow",
                references=[
                    dict(picture=i + 1, format=r["format"], sha256=r["encoded_sha256"])
                    for i, r in enumerate(references)
                ],
            )
            if mismatch == "order":
                a, b = condition["references"]
                a["sha256"], b["sha256"] = b["sha256"], a["sha256"]
            elif mismatch == "picture_gap":
                condition["references"][1]["picture"] = 3
            elif mismatch == "bytes":
                condition["references"][0]["sha256"] = "b" * 64
            assert await source.inference.prepared_for_condition(condition) is (
                mismatch is None
            )
            if mismatch is None:
                resolved = [
                    dict(
                        id=f"<Picture {i + 1}>",
                        format=r["format"],
                        payload=r["payload"],
                    )
                    for i, r in enumerate(references)
                ]
                assert await source.inference.prepared_for(
                    "snow", 42, resolved, world=True
                )
                assert not source.inference.preparation_matches("snow", 42, resolved)
                assert [r["id"] for r in candidate.assets.references] == [
                    "reference-0",
                    "reference-1",
                ]
                assert lease.ws.commands[0]["references"] == candidate.assets.references
            else:
                assert candidate.resource_state == "released"
        finally:
            await finish_candidate(source, lease, task)

    asyncio.run(run())


async def finish_candidate(source, lease, task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    # These two tests explicitly own all fixture resources, including the parent.
    if source.startup is not None:
        await source.startup.close()
    await source.inference.close()
    await source.model_binding.close()


def test_world_handoff_keeps_original_child_and_pins_real_cli_pixels(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import EngineFrame
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderJournal,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore
    from sglang.multimodal_gen.runtime.realtime.whitebox.startup import (
        model_transport_ready,
    )

    async def run():
        source, lease, task, entered, finish = await managed_source(monkeypatch)
        try:
            await asyncio.wait_for(lease.ws.preparing.wait(), 1)
            model_transport_ready(source)
            lease.ws.release_prepare.set()
            assert await source.inference.prepared_for_condition(
                dict(prompt="snow", references=[])
            )
            candidate = source.inference.preparation
            original_model_ids = {
                k: candidate.ids[k] for k in ["model_session_id", "generation_id"]
            }
            finish.set()
            await task
            source.engine = NS(
                stream_id="real-view", metadata={"world_build_hash": "verified-build"}
            )
            source.journal = RenderJournal(
                MemoryRoomStore(),
                user_id="u",
                source_session_id=source.session_id,
                source_epoch=source.source_epoch,
                owner_epoch="owner",
            )
            await source.journal.create()
            await source.journal.initialize_inference(manual=True)
            await source.journal.update(
                lambda s: s["inference"].update(
                    enabled=True, revision=1, state="starting"
                )
            )
            reserved = await source.journal.reserve_model_context_segment()
            pixels = bytes([0, 0, 255]) * 32 * 32
            frame = EngineFrame(
                7,
                700,
                pixels,
                {},
                provenance={
                    "capture_pts_us": 700,
                    "cli_frame_header": {
                        "source": {
                            "presentationId": "live",
                            "sdkEpoch": 1,
                            "simulationTick": 20,
                            "sourceTimeUs": 700,
                            "worldRevision": 3,
                        }
                    },
                },
            )
            first = await source.inference.adopt_preparation(
                (8, (frame, 0.0)), context={"segmentId": reserved["segment_id"]}
            )
            assert first[1][0] is frame and first[1][0].rgb == pixels
            assert candidate.rgb == bytes([255, 0, 0]) * 32 * 32
            assert candidate.cli_rgb == pixels
            assert source.startup is candidate and source.inference.preparation is None
            assert (await source.journal.read())[
                "model_context_segment_ids"
            ] == reserved
            from test_whitebox_source_conditions import condition

            from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
                payload_hash,
            )

            committed = condition()
            committed.update(
                segmentId=reserved["segment_id"], prompt="snow", references=[]
            )
            committed["payloadHash"] = payload_hash(committed)
            await source.journal.update(lambda s: s.update(source_state="ready"))
            operation = await source.journal.model_condition(
                committed,
                [],
                seed=42,
                source_first=frame.index,
                opening_revision=1,
                prepared_model_ids=original_model_ids,
            )
            ids = (await source.journal.read())["model_context_segment_ids"]
            assert operation["segment_id"] == reserved["segment_id"]
            assert ids["segment_id"] == reserved["segment_id"]
            assert ids["segment_epoch"] == reserved["segment_epoch"]
            assert all(ids[k] == v for k, v in original_model_ids.items())
            async with candidate.open():
                await candidate.attach()
                candidate.source_started()
            assert [x.get("kind") for x in lease.ws.commands] == [
                None,
                "source.prepare",
                "source.attach",
            ]
            assert lease.calls == 1 and lease.releases == 0
            assert source.inference.assets.references == []
            assert (
                source.inference.assets.rgb is not None
            )  # Candidate does not own frozen launch assets.
        finally:
            await finish_candidate(source, lease, task)
        assert lease.calls == lease.releases == 1

    asyncio.run(run())


@pytest.mark.parametrize("mismatch", ["prompt", "references"])
def test_world_mismatch_retires_candidate_before_normal_child(monkeypatch, mismatch):
    async def run():
        source, lease, task, _, _ = await managed_source(monkeypatch)
        try:
            await asyncio.wait_for(lease.ws.preparing.wait(), 1)
            candidate = source.inference.preparation
            condition = dict(prompt="snow", references=[])
            if mismatch == "prompt":
                condition["prompt"] = "authoritative different scene"
            else:
                condition["references"] = [
                    dict(picture=1, format="png", sha256="b" * 64)
                ]
            assert not await source.inference.prepared_for_condition(condition)
            assert candidate.resource_state == "released" and not lease.ws.operations
            assert source.inference.preparation is None
            child = source.model_binding.borrow(
                dict(model_session_id=uuid4().hex, generation_id=uuid4().hex)
            )
            assert child.generation_id != candidate.ids["generation_id"]
            async with child.open():
                assert lease.calls == 1 and lease.releases == 0
        finally:
            await finish_candidate(source, lease, task)

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_cli_failure_cancels_pending_preparation_without_creating_second_child(
    monkeypatch,
    cancel,
):
    async def run():
        failure = ValueError("CLI publication changed")
        source, lease, task, entered, finish = await managed_source(
            monkeypatch, cli_error=failure
        )
        try:
            await asyncio.wait_for(lease.ws.preparing.wait(), 1)
            if cancel:
                task.cancel()
            else:
                finish.set()
            result = await asyncio.gather(task, return_exceptions=True)
            if not cancel:
                assert result == [failure]
            assert source.inference.preparation is None
            assert not lease.ws.operations
            lease.ws.release_prepare.set()
            await asyncio.sleep(0)
            assert (
                lease.calls == 1
                and len([x for x in lease.ws.sent if x["type"] == "bound.open"]) == 1
            )
            assert source.startup is None
        finally:
            await finish_candidate(source, lease, task)

    asyncio.run(run())


def test_async_preparation_cannot_bypass_changed_opening_revision(tmp_path):
    from test_whitebox_source_conditions import setup

    async def run():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def prepared(_):
            entered.set()
            await finish.wait()
            return None

        adapter, condition, journal, receipts, submits, resources = await setup(
            tmp_path, startup_release=prepared
        )
        await journal.initialize_inference(manual=True)
        await journal.update(
            lambda s: s["inference"].update(enabled=True, state="enabled")
        )
        resources.set()
        try:
            await adapter.handle(condition)
            await asyncio.wait_for(entered.wait(), 1)
            await journal.update(
                lambda s: s["inference"].update(revision=s["inference"]["revision"] + 1)
            )
            finish.set()
            async with asyncio.timeout(1):
                while receipts[-1]["phase"] != "rejected":
                    await asyncio.sleep(0)
            assert not submits
        finally:
            await adapter.close()

    asyncio.run(run())


def test_world_condition_commit_adopts_prepared_child_atomically(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from test_whitebox_source_conditions import condition

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import EngineFrame
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_assets import (
        SourceAssets,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderJournal,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
        payload_hash,
    )

    async def run():
        source, lease, task, _, finish = await managed_source(monkeypatch)
        work = []
        try:
            await asyncio.wait_for(lease.ws.preparing.wait(), 1)
            lease.ws.release_prepare.set()
            await source.inference.wait_preparation()
            candidate = source.inference.preparation
            finish.set()
            await task
            source.engine = NS(
                stream_id="view", metadata={"world_build_hash": "sealed"}
            )
            source.journal = RenderJournal(
                MemoryRoomStore(),
                user_id="u",
                source_session_id=source.session_id,
                source_epoch=source.source_epoch,
                owner_epoch="owner",
            )
            await source.journal.create()
            await source.journal.initialize_inference(manual=True)
            await source.journal.update(
                lambda s: (
                    s.update(source_state="ready"),
                    s["inference"].update(enabled=True, revision=1, state="starting"),
                )
            )
            reserved = await source.journal.reserve_model_context_segment()
            c = condition()
            c.update(segmentId=reserved["segment_id"], prompt="snow", references=[])
            c["payloadHash"] = payload_hash(c)
            native = {
                **{k: c["source"][k] for k in ["presentationId", "sdkEpoch"]},
                **{
                    k: c["effective"][k]
                    for k in ["simulationTick", "sourceFrameId", "sourceTimeUs"]
                },
                "worldRevision": 3,
            }
            frame = EngineFrame(
                10,
                1000,
                bytes(32 * 32 * 3),
                {},
                provenance={
                    "capture_pts_us": 1000,
                    "cli_frame_header": {
                        "sessionId": c["source"]["sessionId"],
                        "epoch": c["source"]["epoch"],
                        "source": native,
                    },
                },
            )
            source.latest = NS(get=AsyncMock(return_value=(11, (frame, 0.0))))
            source.assets = SourceAssets(tmp_path, memory=True)
            source.context_channel = NS(
                suspended=False, first_presented_supported=False
            )
            source.first_presented_required = False
            source.current_segment = None
            source.operation_message = AsyncMock()
            source._transition_segment = AsyncMock()

            def spawn(coro):
                t = asyncio.create_task(coro)
                work.append(t)
                return t

            source.spawn_control = spawn
            op = await source.submit_source_condition(c, [])
            await asyncio.gather(*work)
            state = await source.journal.read()
            assert source.startup is candidate and source.inference.preparation is None
            assert (
                state["segment"]["model_session_id"]
                == candidate.ids["model_session_id"]
            )
            assert state["segment"]["generation_id"] == candidate.ids["generation_id"]
            assert state["segment"]["segment_id"] == reserved["segment_id"]
            assert (
                state["model_context_segment_ids"]["generation_id"]
                == candidate.ids["generation_id"]
            )
            assert state["segment"]["start_boundary"] == {"source_frame_index": 10}
            source._transition_segment.assert_awaited_once()
            assert source._transition_segment.await_args.args[2][1][0] is frame
            assert len([x for x in lease.ws.sent if x["type"] == "bound.open"]) == 1
            assert op["op"] == "render.start"
        finally:
            await asyncio.gather(*work, return_exceptions=True)
            await finish_candidate(source, lease, task)

    asyncio.run(run())


def test_preparation_timeout_reports_failure_and_releases_child(monkeypatch):
    import time

    async def run():
        source, lease, task, _, _ = await managed_source(monkeypatch)
        try:
            await asyncio.wait_for(lease.ws.preparing.wait(), 1)
            source.inference.preparation.deadline = time.monotonic() - 1
            with pytest.raises(TimeoutError):
                await source.inference.prepared_for("snow", 42, [])
            assert isinstance(source.inference.preparation_failure, TimeoutError)
            assert not lease.ws.operations and source.inference.preparation is None
            assert source.inference.assets.rgb is not None
            assert lease.calls == 1 and lease.releases == 0
        finally:
            await finish_candidate(source, lease, task)

    asyncio.run(run())


def test_retiring_candidate_does_not_cancel_its_condition_waiter(monkeypatch):
    async def run():
        source, lease, task, _, _ = await managed_source(monkeypatch)
        waiter = None
        try:
            await asyncio.wait_for(lease.ws.preparing.wait(), 1)
            waiter = asyncio.create_task(source.inference.wait_preparation())
            await asyncio.sleep(0)
            await source.inference.discard_preparation()
            assert await waiter is False
            assert source.inference.preparation is None and not lease.ws.operations
            assert lease.calls == 1 and lease.releases == 0
        finally:
            if waiter is not None:
                await asyncio.gather(waiter, return_exceptions=True)
            await finish_candidate(source, lease, task)

    asyncio.run(run())


def test_cancelling_condition_waiter_does_not_cancel_shared_preparation(monkeypatch):
    async def run():
        source, lease, task, _, _ = await managed_source(monkeypatch)
        try:
            await asyncio.wait_for(lease.ws.preparing.wait(), 1)
            candidate = source.inference.preparation
            waiter = asyncio.create_task(source.inference.wait_preparation())
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert not candidate.task.done() and lease.ws.operations
            lease.ws.release_prepare.set()
            assert await source.inference.wait_preparation()
        finally:
            await finish_candidate(source, lease, task)

    asyncio.run(run())


def test_unknown_candidate_cleanup_never_admits_replacement(monkeypatch):
    async def run():
        source, lease, task, _, _ = await managed_source(monkeypatch)
        try:
            await asyncio.wait_for(lease.ws.preparing.wait(), 1)
            original = lease.ws.send

            async def send(raw):
                message = msgspec.msgpack.decode(raw)
                if message["type"] == "bound.cancel":
                    lease.ws.operations.discard(message["operation_id"])
                    lease.ws.incoming.put_nowait(
                        dict(
                            type="bound.completed",
                            operation_id=message["operation_id"],
                            cleaned=False,
                        )
                    )
                else:
                    await original(raw)

            monkeypatch.setattr(lease.ws, "send", send)
            with pytest.raises(RenderError) as error:
                await source.inference.prepared_for("changed", 42, [])
            assert error.value.code == "RELEASE_UNKNOWN"
            assert source.inference.preparation is not None
            assert len([x for x in lease.ws.sent if x["type"] == "bound.open"]) == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.gather(source.inference.close(), return_exceptions=True)
            await source.model_binding.close()

    asyncio.run(run())


@pytest.mark.parametrize("matching", [False, True])
def test_model_failure_is_reported_without_fallback_compute(monkeypatch, matching):
    async def run():
        source, lease, task, _, _ = await managed_source(monkeypatch)
        try:
            await asyncio.wait_for(lease.ws.preparing.wait(), 1)
            operation = next(iter(lease.ws.operations))
            lease.ws.reply(
                operation,
                dict(type="error", code="GPU_FAILED", content="private worker detail"),
            )
            await asyncio.wait_for(
                asyncio.gather(
                    source.inference.preparation.task, return_exceptions=True
                ),
                1,
            )
            with pytest.raises(ValueError) as error:
                await source.inference.prepared_for_condition(
                    dict(prompt="snow" if matching else "different", references=[])
                )
            assert error.value.code == "MODEL_PREPARATION_FAILED"
            assert "private worker detail" not in str(error.value)
            assert source.inference.preparation_failure is error.value
            assert source.inference.preparation is None
            assert len([x for x in lease.ws.sent if x["type"] == "bound.open"]) == 1
        finally:
            await finish_candidate(source, lease, task)

    asyncio.run(run())


def test_explicit_prefill_retires_optional_candidate_before_compute(monkeypatch):
    from unittest.mock import AsyncMock

    from sglang.multimodal_gen.runtime.realtime.whitebox.control_prefill import (
        ControlPrefill,
    )

    async def run():
        source, lease, task, _, _ = await managed_source(monkeypatch)
        try:
            await asyncio.wait_for(lease.ws.preparing.wait(), 1)
            source.journal = NS(
                read=AsyncMock(return_value={"segment": None, "pending_replace": None})
            )
            source.assets = NS(resolve=AsyncMock(return_value=[]))
            prefill = ControlPrefill(source)
            entered = asyncio.Event()

            async def compute(*args):
                assert source.inference.preparation is None
                assert not lease.ws.operations and lease.calls == 1
                entered.set()

            monkeypatch.setattr(prefill, "_compute", compute)
            work = await prefill.start_startup(
                dict(prompt="new authoritative", seed=42),
                [],
                None,
                current=lambda: True,
                observe=lambda _: None,
            )
            await work
            assert entered.is_set()
            assert lease.releases == 0
        finally:
            await finish_candidate(source, lease, task)

    asyncio.run(run())


def test_closed_source_cannot_start_a_late_queued_preparation(monkeypatch):
    async def run():
        source, lease, task, entered, _ = await managed_source(monkeypatch)
        source.closing = True
        result = await asyncio.gather(task, return_exceptions=True)
        await source.inference.close()
        assert getattr(result[0], "code", None) == "PLAYER_EXPIRED"
        assert not entered.is_set() and not lease.calls
        assert source.inference.preparation is None

    asyncio.run(run())
