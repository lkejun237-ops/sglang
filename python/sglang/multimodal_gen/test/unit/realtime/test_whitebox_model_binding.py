import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace as NS

import msgspec
from whitebox_player_log_test_utils import player_log_capture as player_log_capture

from sglang.multimodal_gen.runtime.realtime.whitebox.model_binding import ModelBinding


@dataclass
class Assignment:
    session_id: str = "parent"
    generation_id: str = "parent-generation"
    denoiser: object = None


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.incoming.put_nowait(
            dict(type="bound.ready", protocol="bound-compute-v1", worker_epoch="a")
        )

    async def recv(self):
        value = await self.incoming.get()
        if value is None:
            raise ConnectionError("closed")
        return msgspec.msgpack.encode(value)

    async def send(self, raw):
        message = msgspec.msgpack.decode(raw)
        self.sent.append(message)
        op = message["operation_id"]
        if message["type"] == "bound.open":
            self.incoming.put_nowait(dict(type="bound.opened", operation_id=op))
        elif message["type"] == "bound.data":
            self.incoming.put_nowait(
                dict(type="bound.data", operation_id=op, payload=message["payload"])
            )
        elif message["type"] == "bound.cancel":
            self.incoming.put_nowait(
                dict(type="bound.completed", operation_id=op, cleaned=True)
            )

    async def close(self):
        self.incoming.put_nowait(None)


class Lease:
    def __init__(self):
        self.resource_state = "none"
        self.calls = 0
        self.releases = 0
        self.ws = Socket()

    @asynccontextmanager
    async def open(self):
        self.calls += 1
        self.resource_state = "reserved"
        try:
            yield (
                self.ws,
                Assignment(denoiser=NS(worker_epoch="a", worker_id="worker-a")),
            )
        finally:
            self.releases += 1
            self.resource_state = "released"


def test_scenes_and_prefill_borrow_same_worker_without_releasing_parent():
    async def run():
        lease = Lease()
        receipts = []

        async def receipt(m):
            receipts.append(m)

        binding = ModelBinding(lease, receipt, timeout=2)
        await binding.wait_ready()
        for i, kind in enumerate(
            ["prefill", "prefill", "prefill", "generate", "generate"]
        ):
            child = binding.borrow(
                dict(model_session_id="s" + str(i), generation_id="g" + str(i)),
                kind=kind,
            )
            async with child.open() as (socket, assignment):
                assert assignment.denoiser is binding.assignment.denoiser
                assert assignment.session_id == "s" + str(i)
                await socket.send(b"hello")
                assert await socket.recv() == b"hello"
            assert child.resource_state == "released"
            assert lease.releases == 0
        assert lease.calls == 1
        await binding.close()
        assert lease.releases == 1

    asyncio.run(run())


def test_cancelled_child_drains_pending_media_before_parent_can_start_next_scene():
    async def run():
        lease = Lease()

        async def receipt(_):
            pass

        binding = ModelBinding(lease, receipt, timeout=2)
        await binding.wait_ready()
        child = binding.borrow(dict(model_session_id="s", generation_id="g"))
        async with child.open() as (socket, _):
            for _ in range(20):
                await socket.send(b"frame")
            await asyncio.sleep(0)
        assert child.resource_state == "released"
        async with binding.borrow(
            dict(model_session_id="next", generation_id="next")
        ).open():
            pass
        assert lease.calls == 1
        await binding.close()

    asyncio.run(run())


def test_two_model_connections_log_only_the_failed_player_and_deduplicate_owner(
    player_log_capture,
):
    import json

    from sglang.multimodal_gen.runtime.realtime.whitebox import diagnostics

    async def run():
        leases = [Lease(), Lease()]
        for index, lease in enumerate(leases):
            lease.request = dict(trace_id=f"model-log-admit-{index}")
            lease.session_id, lease.generation_id = (
                f"model-session-{index}",
                f"generation-{index}",
            )

        async def receipt(_):
            pass

        bindings = [ModelBinding(lease, receipt, timeout=2) for lease in leases]
        await asyncio.gather(*(binding.wait_ready() for binding in bindings))
        leases[0].ws.incoming.put_nowait(None)
        await asyncio.wait_for(bindings[0].stopped.wait(), 2)
        assert isinstance(bindings[0].error, ConnectionError)
        assert not bindings[1].closed
        rows = [
            json.loads(row.message)
            for row in player_log_capture.records
            if '"event":"whitebox.player.failed"' in row.message
        ]
        assert len(rows) == 1
        assert rows[0]["operationId"] == "model-log-admit-0"
        assert rows[0]["workerId"] == "worker-a"
        assert rows[0]["workerEpoch"] == "a"
        assert rows[0]["modelSessionId"] == "parent"
        diagnostics.log_player_failure(
            "member.admit", bindings[0].error, binding=bindings[0]
        )
        await bindings[1].close()
        rows = [
            row
            for row in player_log_capture.records
            if '"event":"whitebox.player.failed"' in row.message
        ]
        assert len(rows) == 1
        assert leases[1].releases == 1

    asyncio.run(run())


def test_player_failure_logging_is_bounded_redacted_and_cannot_replace_business_error(
    player_log_capture, monkeypatch
):
    import json

    from sglang.multimodal_gen.runtime.realtime.whitebox import diagnostics

    error = RuntimeError("Bearer private-token https://private.invalid/path")
    error.code = "MODEL_CONNECTION_FAILED"
    record = dict(
        binding=dict(
            gameId="safe-game", ownerEpoch="safe-owner", publicationId="safe-pub"
        ),
        admission=dict(operationId="safe-admission", requestId="safe-request"),
        mediaLaunch=dict(initMsgpackBase64="private-pixels"),
        caller=dict(subject="private-user"),
    )
    diagnostics.log_player_failure("member.media", error, record=record)
    rows = [
        json.loads(row.message)
        for row in player_log_capture.records
        if '"event":"whitebox.player.failed"' in row.message
    ]
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "MODEL_CONNECTION_FAILED"
    assert rows[0]["error"]["type"] == "RuntimeError"
    assert not any(
        secret in json.dumps(rows)
        for secret in [
            "private-token",
            "private.invalid",
            "private-pixels",
            "private-user",
        ]
    )
    monkeypatch.setattr(
        diagnostics,
        "_emit_player_failure",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("sink broken")),
    )
    diagnostics.log_player_failure(
        "member.media", ValueError("another failure"), record=record
    )


def test_expected_cancellation_does_not_emit_player_failure(player_log_capture):
    from sglang.multimodal_gen.runtime.realtime.whitebox import diagnostics

    diagnostics.log_player_failure("member.leave", asyncio.CancelledError(), record={})
    assert not any(
        '"event":"whitebox.player.failed"' in row.message
        for row in player_log_capture.records
    )


def test_disabled_player_logger_does_not_read_diagnostic_context(monkeypatch):
    import logging

    from sglang.multimodal_gen.runtime.realtime.whitebox import diagnostics

    reads = []

    class Context(dict):
        def get(self, *args):
            reads.append(args)
            raise AssertionError("disabled logging must not inspect context")

    monkeypatch.setattr(logging.getLogger(diagnostics.__name__), "disabled", True)
    diagnostics.log_player_failure(
        "model.connection", RuntimeError("failed"), record=Context(active=True)
    )
    assert reads == []


def test_player_log_native_pipe_pressure_and_shutdown():
    """The deployed Linux pipe must not share Python Handler locks or hold exit."""
    import json
    import os
    import subprocess
    import sys
    import tempfile

    import pytest

    if sys.platform != "linux":
        pytest.skip("independent /proc descriptor semantics are verified in Linux")
    child = r"""
import json,logging,os,threading
from sglang.multimodal_gen.runtime.realtime.whitebox import diagnostics
called=threading.Event()
class BlockingHandler(logging.Handler):
    def emit(self,event):
        called.set()
        threading.Event().wait(5)
logger=diagnostics._player_logger
logger.handlers=[BlockingHandler()]
logger.propagate=False
logger.setLevel(logging.WARNING)
before=os.get_blocking(2)
error=RuntimeError("Bearer private-secret https://private.invalid")
error.code="CONTROLLED_PLAYER_FAILURE"
diagnostics.log_player_failure("member.admit",error,record=dict(admission=dict(operationId="native-original")))
called.wait(.1)
print(json.dumps(dict(original_blocking=before,current_blocking=os.get_blocking(2),handler_called=called.is_set())),flush=True)
"""
    for sink in ("pipe", "full-pipe", "file"):
        read_fd = write_fd = None
        target = tempfile.TemporaryFile() if sink == "file" else None
        try:
            if target is None:
                read_fd, write_fd = os.pipe()
                if sink == "full-pipe":
                    os.set_blocking(write_fd, False)
                    while True:
                        try:
                            os.write(write_fd, b"x" * 4096)
                        except BlockingIOError:
                            break
                    os.set_blocking(write_fd, True)
            result = subprocess.run(
                [sys.executable, "-c", child],
                stdout=subprocess.PIPE,
                stderr=target if target is not None else write_fd,
                text=True,
                timeout=2,
                env={
                    **os.environ,
                    "SGLANG_LIGHTWEIGHT_RUNTIME": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            assert result.returncode == 0
            state = json.loads(result.stdout)
            assert state == dict(
                original_blocking=True, current_blocking=True, handler_called=False
            )
            if target is not None:
                target.seek(0)
                assert target.read() == b""
            else:
                os.close(write_fd)
                write_fd = None
                chunks = []
                while chunk := os.read(read_fd, 65536):
                    chunks.append(chunk)
                data = b"".join(chunks)
                if sink == "pipe":
                    text = data.decode()
                    row = json.loads(text[text.index("{") :])
                    assert row["event"] == "whitebox.player.failed"
                    assert row["operationId"] == "native-original"
                    assert row["error"]["code"] == "CONTROLLED_PLAYER_FAILURE"
                    assert (
                        "private-secret" not in text and "private.invalid" not in text
                    )
                else:
                    assert b"whitebox.player.failed" not in data
        finally:
            for descriptor in (read_fd, write_fd):
                if descriptor is not None:
                    os.close(descriptor)
            if target is not None:
                target.close()


def test_room_start_failure_preserves_safe_code_and_owner_identity(player_log_capture):
    import json

    from sglang.multimodal_gen.runtime.realtime.whitebox import diagnostics

    error = RuntimeError("private wrapper")
    error.__cause__ = ValueError("ROOM_HOST_CAPACITY_INSUFFICIENT")
    diagnostics.log_player_failure(
        "room.start",
        error,
        action="createRoom",
        owner_epoch="owner-original",
        record=dict(
            game_id="room-original",
            invocation=dict(
                createContext=dict(
                    operation=dict(
                        operationId="create-original", requestId="request-original"
                    ),
                    frozenPublication=dict(
                        world=dict(publicationId="publication-original")
                    ),
                )
            ),
        ),
    )
    rows = [
        json.loads(row.message)
        for row in player_log_capture.records
        if '"event":"whitebox.player.failed"' in row.message
    ]
    assert len(rows) == 1
    assert rows[0]["ownerEpoch"] == "owner-original"
    assert rows[0]["action"] == "createRoom"
    assert rows[0]["gameId"] == "room-original"
    assert rows[0]["operationId"] == "create-original"
    assert rows[0]["error"]["cause"]["code"] == "ROOM_HOST_CAPACITY_INSUFFICIENT"
    assert "private wrapper" not in player_log_capture.text
