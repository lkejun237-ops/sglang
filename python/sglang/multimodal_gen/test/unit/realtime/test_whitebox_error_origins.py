# SPDX-License-Identifier: Apache-2.0
"""Errors cross real orchestration boundaries without exposing private bodies."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

import msgspec
import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import (
    WhiteboxDiagnostic,
    public_error,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
    InteractiveSession,
    ModelLeases,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
    RenderError,
    RenderJournal,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore


def test_nonstartup_init_preserves_model_rejection_and_private_log(caplog):
    async def run():
        @asynccontextmanager
        async def lease():
            async def send(_):
                pass

            async def recv():
                return msgspec.msgpack.encode(
                    dict(type="error", code="GPU_MEMORY", content="private GPU path")
                )

            yield NS(send=send, recv=recv), None

        session = object.__new__(InteractiveSession)
        session.latest = NS(version=0)
        session.submitted = 0
        session.revision = 0
        session.display_mode = "model"
        session.epoch = "epoch"
        session.trace_id = "init-trace"
        session.outgoing = asyncio.Queue()
        session.leases = NS(open=lease)
        session.config = NS(startup_seconds=1, spec=VideoSpec(32, 32))
        session.upstream = NS(init={"source_stream": {}})
        with pytest.raises(Exception) as failure:
            await session._generation(
                0, dict(mode="model", prompt="test", references=[]), "epoch"
            )
        diagnostic = public_error(failure.value, "init-trace")
        assert diagnostic["code"] == "MODEL_PREPARATION_FAILED"
        assert diagnostic["stage"] == "model"
        assert diagnostic["upstream_code"] == "GPU_MEMORY"
        assert "private" not in str(diagnostic)
        assert "private GPU path" in caplog.text

    asyncio.run(run())


def test_release_failure_preserves_primary_error_and_independent_cleanup_status():
    async def run():
        assignment = NS(
            vae=None,
            denoiser=NS(endpoint="ws://worker", worker_epoch="worker"),
            session_id="session",
            generation_id="generation",
            token="test",
            player_expires_at=None,
        )

        class Coordinator:
            async def admit(self, **_):
                return assignment

            async def release_verified(self, _):
                raise RuntimeError("private cleanup")

            async def quarantine(self, _):
                raise RuntimeError("private cleanup")

        async def connect(*_, **__):
            async def close():
                pass

            return NS(close=close)

        lease = ModelLeases(
            Coordinator(),
            connect,
            user_id="user",
            model_revision="model",
            vae_fingerprint="local",
            trace_id="trace",
            max_bytes=1024,
            strict_release=True,
            cleanup_timeout_s=0,
        )
        primary = WhiteboxDiagnostic("MODEL_PREPARATION_FAILED", "model")
        primary.upstream_code = "GPU_MEMORY"
        with pytest.raises(Exception) as failure:
            async with lease.open():
                raise primary
        assert failure.value is primary
        diagnostic = public_error(failure.value, "trace")
        assert diagnostic["cleanup_code"] == "RELEASE_UNKNOWN"
        assert diagnostic["upstream_code"] == "GPU_MEMORY"
        assert lease.resource_state == "release_unknown"

    asyncio.run(run())


@pytest.mark.parametrize("quarantine_fails", [False, True])
def test_missing_sdk_cleanup_receipt_retains_managed_lease(quarantine_fails):
    async def run():
        calls = []

        class Coordinator:
            async def quarantine(self, assignment):
                calls.append(assignment)
                if quarantine_fails:
                    raise RuntimeError("quarantine transport failed")

            async def release_verified(self, *_, **__):
                pytest.fail("Cannot release without SDK cleanup proof")

        lease = ModelLeases(
            Coordinator(),
            None,
            user_id="u",
            model_revision="m",
            vae_fingerprint="v",
            trace_id="trace",
            max_bytes=1024,
            strict_release=True,
            cleanup_timeout_s=1,
        )
        assignment = NS(player_expires_at=100)
        with pytest.raises(RenderError) as failure:
            await lease._release_verified(assignment)
        assert failure.value.code == "RELEASE_UNKNOWN"
        assert lease.resource_state == "release_unknown"
        assert calls == [assignment]

    asyncio.run(run())


def test_model_pair_ready_cannot_swallow_simultaneous_player_cancel():
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import EngineFrame

    async def run():
        owner = None

        class ReadyAtCancel:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

            async def wait_for(self, predicate):
                assert predicate()
                asyncio.get_running_loop().call_soon(owner.cancel)
                return True

        async def blocked_output(*_):
            await asyncio.Event().wait()

        session = NS(
            received=0,
            config=NS(room_owned=True, spec=VideoSpec(32, 32)),
            model_geometry=NS(output=VideoSpec(32, 32)),
            recording=NS(accepting=True, pair=lambda *_: None),
            pair_condition=ReadyAtCancel(),
            pairs={0: (None, None, b"source-au")},
            pairing_age_budget=lambda: 1,
            _media=blocked_output,
            model_preview_count=0,
        )
        frame = EngineFrame(0, 0, bytes(32 * 32 * 3), {})
        owner = asyncio.create_task(
            InteractiveSession._deliver_model_frame(
                session, 0, frame, {"payload": b"model-au"}, 0, {}
            )
        )
        try:
            done, _ = await asyncio.wait((owner,), timeout=0.2)
            assert owner in done and owner.cancelled()
        finally:
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(run())


def test_segment_and_operation_terminal_messages_keep_the_same_primary_cause():
    async def run():
        journal = RenderJournal(
            MemoryRoomStore(),
            user_id="user",
            source_session_id="source",
            source_epoch="epoch",
            owner_epoch="room",
        )
        await journal.create()
        segment_state = dict(
            segment_id="segment",
            request_id="request",
            state="streaming",
            prompt_queue=[],
            validating_prompt=None,
            prompt_events={},
        )
        await journal.update(
            lambda value: value.update(
                segment=segment_state,
                operations={
                    "request": dict(
                        request_id="request", op="render.start", state="accepted"
                    )
                },
            )
        )
        source = object.__new__(SourceSession)
        source.journal = journal
        source.session_id, source.source_epoch = "source", "epoch"
        source.engine = NS(stream_id="view")
        source.outgoing = asyncio.Queue()
        segment = NS(
            segment_id="segment",
            resource_state="release_unknown",
            archive_failure=False,
            pending_prompt=None,
            segment=segment_state,
            failure_diagnostic=dict(
                code="MODEL_FAILED",
                stage="model",
                upstream_code="GPU_MEMORY",
                trace_id="trace",
                cleanup_code="RELEASE_UNKNOWN",
            ),
        )
        await source.segment_finished(segment, "failed", "MODEL_FAILED")
        messages = [
            msgspec.msgpack.decode(source.outgoing.get_nowait())
            for _ in range(source.outgoing.qsize())
        ]
        failed = messages[0]["segment"]
        operation = messages[-1]
        assert failed["state"] == operation["state"] == "failed"
        assert failed["error_code"] == operation["code"] == "MODEL_FAILED"
        for value in (failed, operation):
            assert value["upstream_code"] == "GPU_MEMORY"
            assert value["stage"] == "model"
            assert value["cleanup_code"] == "RELEASE_UNKNOWN"
            assert value["resource_state"] == "release_unknown"

    asyncio.run(run())


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_server_control_failure_is_not_an_invalid_player_message(error_type):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        RoomPipeline,
    )

    async def run():
        stopped = asyncio.Event()
        result = {}

        def stop(reason, *, error=None):
            result.update(reason=reason, error=error)
            stopped.set()

        async def send(_):
            raise error_type("private database unavailable")

        owner = object.__new__(RoomPipeline)
        owner.command_queue = asyncio.Queue()
        owner.command_bytes = 1
        owner.generation = 1
        owner.trace_id = "control-trace"
        owner.session_id = "source"
        owner.committed_input = -1
        owner.session = NS(send=send, input_sequence=-1)
        owner.subscription = NS(generation=1, stop=stop)
        owner.command_queue.put_nowait(
            (1, b"x", {"type": "command", "op": "source.status"})
        )
        task = asyncio.create_task(owner._commands())
        try:
            await asyncio.wait_for(stopped.wait(), 1)
            assert result["reason"] == "CONTROL_SERVICE_FAILED"
            diagnostic = public_error(result["error"], "control-trace")
            assert diagnostic["code"] == "CONTROL_SERVICE_FAILED"
            assert diagnostic["stage"] == "control"
            assert "private" not in str(diagnostic)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize(
    "failed_stage, expected",
    [("archive", "ARCHIVE_FAILED"), ("encoding", "VIDEO_ENCODING_FAILED")],
)
def test_segment_setup_does_not_blame_model_for_archive_or_encoder_failure(
    tmp_path, monkeypatch, failed_stage, expected
):
    from sglang.multimodal_gen.runtime.realtime.whitebox import render_segment
    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncArchiveManager,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig

    async def run():
        archives = AsyncArchiveManager()
        if failed_stage == "archive":

            async def create_recording(**_):
                raise RuntimeError("private archive path")

            monkeypatch.setattr(archives, "create_recording", create_recording)
        else:

            def encoder(*_, **__):
                raise RuntimeError("private encoder path")

            monkeypatch.setattr(render_segment, "H264PlaybackEncoder", encoder)

        result = {}

        async def finished(segment, outcome, code):
            result.update(
                outcome=outcome, code=code, diagnostic=segment.failure_diagnostic
            )

        owner = NS(
            config=WhiteboxConfig(
                engine_factory=None,
                ui_factory=None,
                spool=tmp_path,
                spec=VideoSpec(32, 32),
            ),
            archives=archives,
            user_id="user",
            reports=None,
            latest=NS(version=0),
            renderer=NS(metadata={}),
            recording=NS(manifest={"object_prefix": "recordings/"}),
            world_id="world",
            submitted=0,
            segment_finished=finished,
        )
        segment = RenderSegment(
            owner,
            dict(
                segment_id="segment",
                segment_epoch="epoch",
                prompt="test",
                prompt_sha256="a" * 64,
                seed=1,
                references=[],
                render_revision=1,
            ),
            [],
            NS(resource_state="none"),
        )
        try:
            await segment.run()
            assert result["outcome"] == "failed"
            assert result["code"] == expected
            assert result["diagnostic"]["stage"] == failed_stage
            assert "private" not in str(result["diagnostic"])
        finally:
            await archives.close()

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_model_event_send_keeps_transport_origin_and_cancellation(cancel):
    async def run():
        async def send(_):
            if cancel:
                raise asyncio.CancelledError()
            raise ConnectionError("private worker address")

        session = object.__new__(InteractiveSession)
        session.model_wire_lock = asyncio.Lock()
        session.model_wire_sequence = 0
        session.trace_id = "send-trace"
        with pytest.raises(BaseException) as failure:
            await session.send_model_event(NS(send=send), "source.append", {})
        if cancel:
            assert isinstance(failure.value, asyncio.CancelledError)
        else:
            diagnostic = public_error(failure.value, "send-trace")
            assert diagnostic["code"] == "MODEL_FAILED"
            assert diagnostic["stage"] == "model"
            assert diagnostic["upstream_code"] == "ConnectionError"
            assert "private" not in str(diagnostic)

    asyncio.run(run())


def test_rtc_control_failure_keeps_the_server_diagnostic_when_attachment_stops():
    from test_whitebox_rtc_pipeline import make_sub

    async def run():
        sub, _, _ = make_sub()
        error = WhiteboxDiagnostic("CONTROL_SERVICE_FAILED", "control")
        try:
            sub.stop("CONTROL_SERVICE_FAILED", error=error)
            with pytest.raises(WhiteboxDiagnostic) as failure:
                await sub.recv()
            assert failure.value is error
        finally:
            sub.stop("done")
            await sub.wait_closed()

    asyncio.run(run())


def test_coordinator_user_limit_keeps_its_admission_cause():
    from sglang.multimodal_gen.runtime.realtime.coordinator import CoordinatorRejected

    diagnostic = public_error(
        CoordinatorRejected("USER_SESSION_LIMIT"), "admission-trace"
    )
    assert diagnostic["code"] == "USER_SESSION_LIMIT"
    assert diagnostic["stage"] == "admission"


@pytest.mark.parametrize(
    "message",
    [
        {"type": "event", "kind": "input", "sequence": 2, "payload": {}},
        {"type": "event", "kind": "unknown", "payload": {}},
    ],
)
def test_client_send_validation_keeps_its_origin_through_room_controls(message):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        RoomPipeline,
    )

    async def run():
        source = object.__new__(SourceSession)
        source.ready = asyncio.Event()
        source.ready.set()
        source.video_transport = None
        source.input_sequence = -1
        source.engine = NS()
        stopped = asyncio.Event()
        result = {}

        def stop(reason, *, error=None):
            result.update(reason=reason, error=error)
            stopped.set()

        owner = object.__new__(RoomPipeline)
        owner.command_queue = asyncio.Queue()
        wire = msgspec.msgpack.encode(message)
        owner.command_bytes = len(wire)
        owner.generation = 1
        owner.trace_id = "validation-trace"
        owner.session_id = "source"
        owner.committed_input = -1
        owner.input_history = {}
        owner.session = source
        owner.subscription = NS(generation=1, stop=stop)
        owner.command_queue.put_nowait((1, wire, message))
        task = asyncio.create_task(owner._commands())
        try:
            await asyncio.wait_for(stopped.wait(), 1)
            assert result["reason"] == "INVALID_CONTROL_MESSAGE"
            diagnostic = public_error(result["error"], "validation-trace")
            assert diagnostic["code"] == "INVALID_CONTROL_MESSAGE"
            assert diagnostic["stage"] == "validation"
            assert source.input_sequence == -1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("startup_wrapper", [False, True])
@pytest.mark.parametrize(
    "primary_code", ["MODEL_STARTUP_TIMEOUT", "CONDITION_SWITCH_FAILED"]
)
def test_primary_failure_keeps_cleanup_status_after_cancelled_generation(
    tmp_path, primary_code, startup_wrapper
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncArchiveManager,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig

    async def run():
        assignment = NS(
            vae=None,
            denoiser=NS(endpoint="ws://worker", worker_epoch="worker"),
            session_id="session",
            generation_id="generation",
            token="test",
            player_expires_at=None,
        )

        class Coordinator:
            async def admit(self, **_):
                return assignment

            async def release_verified(self, _):
                raise RuntimeError("private cleanup")

            async def quarantine(self, _):
                raise RuntimeError("private cleanup")

        async def connect(*_, **__):
            async def close():
                pass

            return NS(close=close)

        lease = ModelLeases(
            Coordinator(),
            connect,
            user_id="user",
            model_revision="model",
            vae_fingerprint="local",
            trace_id="trace",
            max_bytes=1024,
            strict_release=True,
            cleanup_timeout_s=0,
        )
        archives = AsyncArchiveManager()
        result = {}

        async def finished(segment, outcome, code):
            result.update(
                outcome=outcome, code=code, diagnostic=segment.failure_diagnostic
            )

        owner = NS(
            config=WhiteboxConfig(
                engine_factory=None,
                ui_factory=None,
                spool=tmp_path,
                spec=VideoSpec(32, 32),
            ),
            archives=archives,
            user_id="user",
            reports=None,
            latest=NS(version=0),
            renderer=NS(metadata={}),
            recording=NS(manifest={"object_prefix": "recordings/"}),
            world_id="world",
            submitted=0,
            segment_finished=finished,
            capture_rate_hint=lambda: 60,
            upstream=NS(init={}),
            session_id="source",
            source_epoch="epoch",
        )
        leases = lease
        if startup_wrapper:
            from sglang.multimodal_gen.runtime.realtime.whitebox.startup import (
                PlayStartup,
            )

            leases = object.__new__(PlayStartup)
            leases.lease = lease
            # The handed-off lease owns cleanup; receipt identity has its own tests.
            leases.receipt = lambda: {}
        segment = RenderSegment(
            owner,
            dict(
                segment_id="segment",
                segment_epoch="epoch",
                prompt="test",
                prompt_sha256="a" * 64,
                seed=1,
                references=[],
                render_revision=1,
            ),
            [],
            leases,
        )

        async def timed_out_generation(*_):
            async with lease.open():
                segment.failure_code = primary_code
                if primary_code == "CONDITION_SWITCH_FAILED":
                    segment.failure_diagnostic = public_error(
                        WhiteboxDiagnostic(primary_code, "model"), "trace"
                    )
                asyncio.current_task().cancel()
                await asyncio.Event().wait()

        segment._generation = timed_out_generation
        try:
            await segment.run()
            assert result["outcome"] == "failed"
            assert result["code"] == primary_code
            assert result["diagnostic"]["code"] == primary_code
            assert result["diagnostic"]["cleanup_code"] == "RELEASE_UNKNOWN"
            assert segment.resource_state == "release_unknown"
        finally:
            await archives.close()

    asyncio.run(run())
