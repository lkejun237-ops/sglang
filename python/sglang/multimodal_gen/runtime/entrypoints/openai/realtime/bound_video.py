# SPDX-License-Identifier: Apache-2.0
"""One admitted Room connection, serial disposable compute contexts.

The reservation belongs to this socket, not to prefill or a video segment.
Only the gateway-authenticated Room can open it. Native generation and cache
code remain responsible for compute and verified scheduler cleanup.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, replace
from uuid import uuid4

import msgspec
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sglang.multimodal_gen.runtime.realtime.source_transport import (
    SourceTransportBudget,
)

router = APIRouter()
log = logging.getLogger(__name__)
PROTOCOL = "bound-compute-v1"


@dataclass
class ChildBinding:
    gateway: object
    registry: object
    owner_id: str
    cleaned: bool = True


class ChildSocket:
    """ASGI-compatible child without accepting/closing the parent transport."""

    def __init__(self, parent, operation_id, send, *, max_queued_bytes=128 << 20):
        self.parent, self.operation_id, self.send = parent, operation_id, send
        self.incoming = asyncio.Queue(maxsize=64)
        self.max_queued_bytes = max_queued_bytes
        self.queued_bytes = 0
        self.close_reason = None
        self.completion = None

    def __getattr__(self, name):
        return getattr(self.parent, name)

    async def accept(self):
        pass

    async def receive_bytes(self):
        value = await self.incoming.get()
        if value is None:
            raise WebSocketDisconnect()
        self.queued_bytes -= len(value)
        return value

    def feed(self, payload):
        # The parent keeps reading control/cancel while a child is computing.
        # Enforce both message and byte bounds instead of blocking that reader.
        if (
            self.incoming.full()
            or self.queued_bytes + len(payload) > self.max_queued_bytes
        ):
            raise ValueError("Bound child input budget exceeded")
        self.incoming.put_nowait(payload)
        self.queued_bytes += len(payload)

    async def receive(self):
        return {"type": "websocket.receive", "bytes": await self.receive_bytes()}

    async def send_bytes(self, value):
        await self.send(
            dict(type="bound.data", operation_id=self.operation_id, payload=value)
        )

    async def close(self, code=1000, reason=""):
        self.close_reason = dict(code=code, reason=reason)
        # The generation caller owns cancellation/cleanup. A virtual close must
        # never release the parent's coordinator reservation.


async def watch_persistence(send, receipt, descriptor, root, started, timeout):
    from .zing_prefill import _persistence_read

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status = await _persistence_read(
                root, descriptor, min(10, max(0.01, deadline - time.monotonic()))
            )
        except asyncio.TimeoutError:
            status = {"state": "pending"}
        if status["state"] in {"persisted", "failed"}:
            await send(
                dict(
                    **receipt,
                    type="prefill." + status["state"],
                    storage=status,
                    memory_ready=True,
                    code=status.get("code"),
                    prefill_to_persisted_ms=(time.monotonic() - started) * 1000,
                )
            )
            return
        await asyncio.sleep(0.25)
    await send(
        dict(
            **receipt,
            type="prefill.failed",
            code="PREFILL_PERSISTENCE_TIMEOUT",
            memory_ready=True,
            fatal=False,
        )
    )


@router.websocket("/bound")
async def bound_video(websocket: WebSocket):
    from . import realtime_video_api as api
    from .zing_prefill import persistence_options, run_prefill_case

    await websocket.accept()
    gateway = api._gateway_managed_config(websocket)
    if gateway is None:
        await websocket.close(code=1008, reason="Gateway reservation required")
        return
    owner = uuid4().hex
    registry = None
    active = child = binding = None
    watchers = set()
    used = set()
    send_lock = asyncio.Lock()
    args = api.get_global_server_args()
    transport = SourceTransportBudget.from_args(args)
    maximum = transport.message_bytes
    parent_task = asyncio.current_task()
    lifetime_task = None
    player_expires_at = None

    async def expire_player():
        # Independent of receive/send, Prefill locks and generation progress.
        await asyncio.sleep(max(0, player_expires_at - time.time()))
        parent_task.cancel()

    async def send(value):
        async with send_lock:
            await websocket.send_bytes(msgspec.msgpack.encode(value))

    async def run(kind, current, scope):
        started = time.monotonic()
        error = None
        try:
            if kind == "generate":
                await api.generate_bound_child(current, scope)
            else:
                raw = await current.receive_bytes()
                data = msgspec.msgpack.decode(raw)
                persist, timeout = persistence_options(data)
                receipt = await run_prefill_case(
                    current, data, len(raw), scope.gateway.session_id, binding=scope
                )
                descriptor = receipt.pop("_disk_descriptor", None)
                receipt.pop("type", None)
                receipt["prefill_to_memory_ms"] = (time.monotonic() - started) * 1000
                await current.send_bytes(
                    msgspec.msgpack.encode(dict(**receipt, type="prefill.memory_ready"))
                )
                if persist and descriptor is not None:
                    # Disk observation outlives the child compute context. The
                    # existing disk writer and cache publication are untouched.
                    task = asyncio.create_task(
                        watch_persistence(
                            send,
                            receipt,
                            descriptor,
                            args.zing_block0_cache_dir,
                            started,
                            timeout,
                        )
                    )
                    watchers.add(task)
                    task.add_done_callback(watchers.discard)
        except asyncio.CancelledError:
            error = "CANCELLED"
            raise
        except Exception:
            log.exception("Bound compute failed: %s", current.operation_id)
            error = "MODEL_OPERATION_FAILED"
        finally:
            if scope.cleaned:
                await registry.mark_blocked(gateway.coordinator_token, owner_id=owner)
            current.completion = dict(
                type="bound.completed",
                operation_id=current.operation_id,
                cleaned=scope.cleaned,
                code=error,
                close=current.close_reason,
            )
            try:
                await send(current.completion)
            except (WebSocketDisconnect, RuntimeError):
                pass
            if not scope.cleaned:
                # Unknown CUDA cleanup is not idle capacity. Keep the parent
                # quarantined; never start another child on unverified state.
                await registry.quarantine(gateway.coordinator_token, owner_id=owner)
                await websocket.close(code=1011, reason="Compute cleanup unconfirmed")

    try:
        registry = await api._consume_gateway_reservation(
            websocket, gateway, owner_id=owner
        )
        if registry is not None:
            player_expires_at = await registry.player_deadline(
                gateway.coordinator_token, owner_id=owner
            )
            if player_expires_at is not None:
                lifetime_task = asyncio.create_task(
                    expire_player(), name="player-deadline"
                )
        await send(
            dict(
                type="bound.ready",
                protocol=PROTOCOL,
                worker_epoch=gateway.worker_epoch,
                player_expires_at=player_expires_at,
            )
        )
        while True:
            raw = await websocket.receive_bytes()
            if len(raw) > maximum + 1024:
                raise ValueError("Bound message exceeds its budget")
            message = msgspec.msgpack.decode(raw)
            if not isinstance(message, dict):
                raise ValueError("Invalid bound message")
            op = message.get("operation_id")
            if message.get("type") == "bound.open":
                if active is not None and not active.done():
                    if binding is not None and binding.cleaned:
                        await api._await_realtime_task(active)
                    else:
                        raise ValueError("Only one bound operation may compute")
                if binding is not None and not binding.cleaned:
                    raise ValueError("Previous compute cleanup is unconfirmed")
                if (
                    message.get("kind") not in {"prefill", "generate"}
                    or not isinstance(op, str)
                    or not 1 <= len(op) <= 128
                    or op in used
                    or len(used) >= 1024
                    or len(watchers) >= 16
                ):
                    raise ValueError("Invalid bound operation")
                used.add(op)
                # Unique scheduler state per child, same DP owner/routing and
                # parent reservation. No second coordinator admit is performed.
                session_id = message.get("session_id")
                generation_id = message.get("generation_id")
                if not all(
                    isinstance(v, str) and 1 <= len(v) <= 128
                    for v in (session_id, generation_id)
                ):
                    raise ValueError("Invalid child identity")
                binding = ChildBinding(
                    replace(
                        gateway, session_id=session_id, generation_id=generation_id
                    ),
                    registry,
                    owner,
                )
                child = ChildSocket(
                    websocket,
                    op,
                    send,
                    max_queued_bytes=min(transport.host_bytes, 2 * maximum),
                )
                active = asyncio.create_task(run(message["kind"], child, binding))
                await send(dict(type="bound.opened", operation_id=op))
            elif message.get("type") in {"bound.data", "bound.cancel"}:
                if child is None or child.operation_id != op:
                    raise ValueError("Stale bound operation")
                if active.done():
                    if message["type"] == "bound.cancel":
                        continue  # Completion crossed the cancellation in flight.
                    raise ValueError("Completed bound operation")
                if message["type"] == "bound.cancel":
                    active.cancel()
                    done, _ = await asyncio.wait((active,), timeout=90)
                    if not done:
                        await registry.quarantine(
                            gateway.coordinator_token, owner_id=owner
                        )
                        await websocket.close(
                            code=1011, reason="Compute cleanup unconfirmed"
                        )
                        raise TimeoutError("Bound child cleanup timed out")
                    await api._await_realtime_task(active)
                else:
                    payload = message.get("payload")
                    if not isinstance(payload, bytes) or len(payload) > maximum:
                        raise ValueError("Invalid child payload")
                    child.feed(payload)
            else:
                raise ValueError("Unknown bound message")
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("Bound realtime connection failed")
    finally:
        if lifetime_task is not None:
            lifetime_task.cancel()
            await asyncio.gather(lifetime_task, return_exceptions=True)

        async def cleanup():
            if active is not None and not active.done():
                active.cancel()
            if active is not None:
                await api._await_realtime_task(active)
            remaining = list(watchers)
            for task in remaining:
                task.cancel()
            await asyncio.gather(*remaining, return_exceptions=True)
            if registry is not None:
                if binding is None or binding.cleaned:
                    await api._release_gateway_reservation(
                        registry, gateway, owner_id=owner
                    )
                else:
                    await registry.quarantine(gateway.coordinator_token, owner_id=owner)

        async def bounded_cleanup():
            task = asyncio.create_task(cleanup(), name="bound-player-cleanup")
            pending = getattr(websocket.app.state, "pending_player_cleanups", None)
            if pending is None:
                pending = websocket.app.state.pending_player_cleanups = set()
            pending.add(task)

            def finished(done):
                pending.discard(done)
                if not done.cancelled() and done.exception() is not None:
                    log.error(
                        "Bound cleanup failed session=%s",
                        gateway.session_id,
                        exc_info=done.exception(),
                    )

            task.add_done_callback(finished)
            try:
                # Retain the actual cleanup owner after timeout. Cancellation of
                # this handler is not evidence that GPU/DMA buffers are free.
                await asyncio.wait_for(asyncio.shield(task), 90)
            except Exception:
                if registry is not None:
                    await registry.quarantine(gateway.coordinator_token, owner_id=owner)
                log.exception(
                    "Player cleanup quarantined session=%s", gateway.session_id
                )

        await api._drain_realtime_cleanup(
            bounded_cleanup() if player_expires_at is not None else cleanup()
        )
