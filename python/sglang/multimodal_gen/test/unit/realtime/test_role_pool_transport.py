# SPDX-License-Identifier: Apache-2.0
"""Real local WebSockets verify split-channel ordering and bound-child cleanup."""

import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager

import msgspec
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from sglang.multimodal_gen.runtime.realtime.coordinator import (
    SessionAssignment,
    WorkerSlot,
)
from sglang.multimodal_gen.runtime.realtime.role_pool.transport import PooledBoundSocket
from sglang.multimodal_gen.runtime.realtime.whitebox.model_binding import ModelBinding


@pytest.mark.parametrize("persistence_available", [True, False])
@pytest.mark.parametrize("close_before_release", [True, False])
def test_public_prefill_waits_for_child_cleanup_and_observes_disk_after_release(
    persistence_available,
    close_before_release,
):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.role_pool.contracts import PoolError
    from sglang.multimodal_gen.runtime.realtime.whitebox.disaggregated_prefill import (
        _PrefillSocket,
    )

    async def run():
        cleaned, released = asyncio.Event(), asyncio.Event()

        class Bound:
            def __init__(self):
                self.queue = asyncio.Queue()
                self.queue.put_nowait(
                    msgspec.msgpack.encode(
                        dict(
                            type="bound.ready",
                            protocol="bound-compute-v1",
                            worker_epoch="worker",
                        )
                    )
                )
                self.finish = None

            def identity(self):
                return dict(admission_id="admission", allocation_id="allocation")

            async def recv(self):
                value = await self.queue.get()
                if isinstance(value, BaseException):
                    raise value
                return value

            async def send(self, wire):
                value = msgspec.msgpack.decode(wire)
                operation = value["operation_id"]
                if value["type"] == "bound.open":
                    assert value["kind"] == "prefill"
                    await self.queue.put(
                        msgspec.msgpack.encode(
                            dict(type="bound.opened", operation_id=operation)
                        )
                    )
                    return
                request = msgspec.msgpack.decode(value["payload"])
                receipt = dict(
                    type="prefill.memory_ready",
                    request_id=request["request_id"],
                    session_id="s",
                    cache_key="a" * 64,
                )
                await self.queue.put(
                    msgspec.msgpack.encode(
                        dict(
                            type="bound.data",
                            operation_id=operation,
                            payload=msgspec.msgpack.encode(receipt),
                        )
                    )
                )
                if persistence_available:
                    self.persistence_observer(receipt, {"cache_key": "a" * 64})
                else:
                    await self.queue.put(
                        msgspec.msgpack.encode(
                            dict(
                                type="prefill.failed",
                                memory_ready=True,
                                request_id=request["request_id"],
                                code="CACHE_PERSISTENCE_UNAVAILABLE",
                            )
                        )
                    )

                async def finish():
                    await cleaned.wait()
                    await self.queue.put(
                        msgspec.msgpack.encode(
                            dict(
                                type="bound.completed",
                                operation_id=operation,
                                cleaned=True,
                            )
                        )
                    )

                self.finish = asyncio.create_task(finish())

            async def close(self):
                await self.queue.put(PoolError("MODEL_CONNECTION_CLOSED"))

        async def control(action, body):
            assert action == "persistence" and body["allocation_id"] == "allocation"
            await released.wait()
            return {"state": "persisted"}

        bound = Bound()
        lease = SimpleNamespace(
            _compute_released=False, cleanup_timeout_s=1, control=control
        )
        assignment = SimpleNamespace(
            session_id="s",
            generation_id="g",
            denoiser=SimpleNamespace(worker_epoch="worker"),
        )
        socket = _PrefillSocket(lease, bound, assignment)
        try:
            await socket.start()
            assert (
                msgspec.msgpack.decode(await socket.recv())["type"] == "prefill.ready"
            )
            await socket.send(
                msgspec.msgpack.encode(
                    dict(type="prefill", request_id="request", persist=True)
                )
            )
            pending = asyncio.create_task(socket.recv())
            await asyncio.sleep(0)
            assert not pending.done()
            cleaned.set()
            ready = msgspec.msgpack.decode(await asyncio.wait_for(pending, 1))
            assert ready["type"] == "prefill.memory_ready"
            if close_before_release:
                await bound.close()
                await asyncio.sleep(0)
                assert socket.error is None
            lease._compute_released = True
            if not close_before_release:
                await bound.close()
            released.set()
            saved = msgspec.msgpack.decode(await asyncio.wait_for(socket.recv(), 1))
            assert saved["type"] == (
                "prefill.persisted" if persistence_available else "prefill.failed"
            )
        finally:
            await socket.close()
            if bound.finish is not None:
                await bound.finish

    asyncio.run(run())


@pytest.mark.parametrize("input_closes_first", [False, True])
@pytest.mark.parametrize("input_completion", ["session.completed", "input.completed"])
def test_real_split_websockets_hold_early_output_and_reuse_parent_for_prefill(
    input_closes_first,
    input_completion,
):
    async def run():
        receipts = {}
        source_sent = asyncio.Event()
        input_completed = asyncio.Event()
        events = []

        async def encoder(ws):
            raw = ws.request.headers["x-role-pool-claim"]
            claim = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
            identity = {k: claim[k] for k in ("session_id", "generation_id")}
            try:
                if ws.request.path.endswith("prefill"):
                    await ws.send(msgspec.msgpack.encode({"type": "prefill.ready"}))
                    value = msgspec.msgpack.decode(await ws.recv())
                    await ws.send(
                        msgspec.msgpack.encode(
                            {
                                "type": "prefill.completed",
                                "request_id": value["request_id"],
                                "cache_key": "a" * 64,
                                **identity,
                            }
                        )
                    )
                    await ws.wait_closed()
                    return
                assert msgspec.msgpack.decode(await ws.recv())["type"] == "init"
                await ws.send(
                    msgspec.msgpack.encode(
                        {
                            "type": "session.ready",
                            "output_attachment": {"token": "attachment"},
                            **identity,
                        }
                    )
                )
                value = msgspec.msgpack.decode(await ws.recv())
                assert value["kind"] == "source.append"
                source_sent.set()
                await asyncio.sleep(0.05)  # Decoder's frame beats input ACK.
                await ws.send(
                    msgspec.msgpack.encode(
                        dict(
                            type="source.accepted",
                            source_frame_start=0,
                            source_frame_end=1,
                        )
                    )
                )
                assert msgspec.msgpack.decode(await ws.recv())["kind"] == "source.end"
                await ws.send(
                    msgspec.msgpack.encode(
                        dict(type=input_completion, source_frames=1, **identity)
                    )
                )
                input_completed.set()
                if input_closes_first:
                    return  # Native Encoder exits before Decoder's final output.
                await ws.wait_closed()
            finally:
                receipts[identity["session_id"]] = {**identity, "cleaned": True}

        async def decoder(ws):
            attach = msgspec.msgpack.decode(await ws.recv())
            assert attach["type"] == "output.attach" and attach["token"] == "attachment"
            identity = {k: attach[k] for k in ("session_id", "generation_id")}
            await ws.send(msgspec.msgpack.encode(dict(type="output.ready", **identity)))
            await source_sent.wait()
            await ws.send(
                msgspec.msgpack.encode(
                    dict(
                        type="frame_batch_header",
                        format="rgb24",
                        output_frame_start=0,
                        num_frames=1,
                        width=32,
                        height=32,
                        **identity,
                    )
                )
            )
            await ws.send(b"x" * (32 * 32 * 3))
            await ws.send(
                msgspec.msgpack.encode(
                    dict(
                        type="source.consumed",
                        source_frame_end=1,
                        source_bytes_end=32 * 32 * 3,
                    )
                )
            )
            if input_closes_first:
                await input_completed.wait()
                await asyncio.sleep(0.05)
            await ws.send(
                msgspec.msgpack.encode(
                    dict(type="session.completed", source_frames=1, **identity)
                )
            )
            await ws.wait_closed()

        async with (
            serve(encoder, "127.0.0.1", 0) as e,
            serve(decoder, "127.0.0.1", 0) as d,
        ):
            eport = e.sockets[0].getsockname()[1]
            dport = d.sockets[0].getsockname()[1]
            group = dict(
                intent={"admission_id": "a"},
                allocation_id="allocation",
                boot_epochs={"dit": "dit-boot"},
                player_expires_at=time.time() + 1800,
                topology={
                    "encoder": {
                        "endpoints": {"input_ws": f"ws://127.0.0.1:{eport}/generate"}
                    },
                    "decoder": {
                        "endpoints": {"output_ws": f"ws://127.0.0.1:{dport}/output"}
                    },
                },
            )

            async def control(action, body):
                if action == "business":
                    return dict(body=body, token="local-wire-test")
                if action == "child-status":
                    return {"child_receipts": dict(receipts)}
                raise AssertionError(action)

            socket = PooledBoundSocket(
                control, group, connect=connect, max_bytes=8 << 20, receipt_timeout=2
            )
            slot = WorkerSlot(
                worker_id="dit",
                role="denoiser",
                endpoint="ws://unused",
                az="local",
                slot_index=0,
                model_revision="test",
                vae_fingerprint="local",
                worker_epoch="dit-boot",
            )
            assignment = SessionAssignment(
                user_id="user",
                session_id="parent",
                generation_id="parent-gen",
                token="allocation",
                expires_at=time.time() + 30,
                denoiser=slot,
                vae=None,
                player_expires_at=group["player_expires_at"],
            )

            class Lease:
                resource_state = "reserved"

                @asynccontextmanager
                async def open(self):
                    try:
                        yield socket, assignment
                    finally:
                        self.resource_state = "released"

            async def receipt(value):
                events.append(value)

            parent = ModelBinding(Lease(), receipt, timeout=3)
            await parent.wait_ready()
            child = parent.borrow(dict(model_session_id="s1", generation_id="g1"))
            async with child.open() as (ws, current):
                assert current.session_id == "s1"
                await ws.send(msgspec.msgpack.encode({"type": "init"}))
                assert (
                    msgspec.msgpack.decode(await ws.recv())["type"] == "session.ready"
                )
                await ws.send(
                    msgspec.msgpack.encode(
                        dict(
                            type="event",
                            kind="source.append",
                            payload=dict(
                                source_frame_start=0,
                                num_frames=1,
                                data=b"x" * (32 * 32 * 3),
                            ),
                        )
                    )
                )
                await ws.send(
                    msgspec.msgpack.encode(
                        dict(
                            type="event",
                            kind="source.end",
                            payload=dict(source_frame_end=1),
                        )
                    )
                )
                kinds = []
                while True:
                    value = msgspec.msgpack.decode(await ws.recv())
                    kinds.append(value["type"])
                    if value["type"] == "frame_batch_header":
                        assert await ws.recv() == b"x" * (32 * 32 * 3)
                    if value["type"] == "session.completed":
                        break
                assert kinds == [
                    "source.accepted",
                    "frame_batch_header",
                    "source.consumed",
                    "session.completed",
                ]
            assert child.resource_state == "released" and not parent.closed
            warmed = parent.borrow(
                dict(model_session_id="s2", generation_id="g2"), kind="prefill"
            )
            async with warmed.open() as (ws, _):
                await ws.send(
                    msgspec.msgpack.encode(
                        dict(type="prefill", request_id="warm-a", persist=False)
                    )
                )
                value = msgspec.msgpack.decode(await ws.recv())
                assert (
                    value["type"] == "prefill.memory_ready"
                    and value["request_id"] == "warm-a"
                )
            assert warmed.resource_state == "released" and not parent.closed
            await parent.close()

    asyncio.run(asyncio.wait_for(run(), 10))
