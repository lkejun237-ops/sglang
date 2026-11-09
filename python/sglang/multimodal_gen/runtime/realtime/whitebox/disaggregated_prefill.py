# SPDX-License-Identifier: Apache-2.0
"""The original standalone Prefill WS over one explicit model-only group.

Player-owned Prefill still uses ModelBinding. This adapter never creates an SDK
member and never supplies a synthetic SDK cleanup receipt. Disk observation can
outlive verified group release, exactly as with the integrated public endpoint.
"""

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

import msgspec

from ..role_pool.contracts import PoolError
from ..role_pool.transport import observe_prefill_persistence
from .disaggregated_binding import DisaggregatedModelLeases


class _PrefillSocket:
    def __init__(self, lease, bound, assignment):
        self.lease, self.bound, self.assignment = lease, bound, assignment
        self.queue = asyncio.Queue(maxsize=8)
        self.operation = uuid4().hex
        self.request = self.receipt = self.error = None
        self.opened, self.memory_ready = asyncio.Event(), asyncio.Event()
        self.closed = False
        self.pending_persistence = []
        self.reader = self.observer = None
        bound.persistence_observer = self.observe_persistence

    async def start(self):
        ready = msgspec.msgpack.decode(await self.bound.recv())
        if (
            ready.get("type") != "bound.ready"
            or ready.get("protocol") != "bound-compute-v1"
            or ready.get("worker_epoch") != self.assignment.denoiser.worker_epoch
        ):
            raise PoolError("CAPABILITY_UNAVAILABLE")
        await self.emit(dict(type="prefill.ready", capabilities={"persistence": True}))
        self.reader = asyncio.create_task(self.read())

    async def emit(self, value):
        if not self.closed:
            await self.queue.put(msgspec.msgpack.encode(value))

    async def recv(self):
        value = await self.queue.get()
        if isinstance(value, BaseException):
            raise value
        return value

    async def send(self, wire):
        value = msgspec.msgpack.decode(wire)
        if self.request is not None or value.get("type") != "prefill":
            raise PoolError("INVALID_OPERATION")
        self.request = value["request_id"]
        await self.bound.send(
            msgspec.msgpack.encode(
                dict(
                    type="bound.open",
                    kind="prefill",
                    operation_id=self.operation,
                    session_id=self.assignment.session_id,
                    generation_id=self.assignment.generation_id,
                )
            )
        )
        await asyncio.wait_for(self.opened.wait(), self.lease.cleanup_timeout_s)
        if self.error is not None:
            raise self.error
        await self.bound.send(
            msgspec.msgpack.encode(
                dict(
                    type="bound.data",
                    operation_id=self.operation,
                    payload=wire,
                )
            )
        )

    async def read(self):
        try:
            while not self.closed:
                value = msgspec.msgpack.decode(await self.bound.recv())
                if value.get("type", "").startswith("prefill."):
                    if self.memory_ready.is_set():
                        await self.emit(value)
                    else:
                        self.pending_persistence.append(value)
                    continue
                if value.get("operation_id") != self.operation:
                    raise PoolError("OPERATION_IDENTITY_MISMATCH")
                if value["type"] == "bound.opened":
                    self.opened.set()
                elif value["type"] == "bound.data":
                    receipt = msgspec.msgpack.decode(value["payload"])
                    if receipt.get("request_id") != self.request:
                        raise PoolError("OPERATION_IDENTITY_MISMATCH")
                    if receipt.get("type") == "prefill.memory_ready":
                        self.receipt = receipt
                    else:
                        await self.emit(receipt)
                elif value["type"] == "bound.completed":
                    if (
                        value.get("cleaned") is not True
                        or value.get("code")
                        or self.receipt is None
                    ):
                        raise PoolError(value.get("code") or "RELEASE_UNKNOWN")
                    await self.emit(self.receipt)
                    self.memory_ready.set()
                    for receipt in self.pending_persistence:
                        await self.emit(receipt)
                    self.pending_persistence.clear()
                    # This lease permits exactly one child. Its verified terminal
                    # receipt ends the business reader; the independent observer
                    # owns subsequent disk receipts. Parent teardown may close
                    # the bound transport before release_compute finishes.
                    return
                else:
                    raise PoolError("INVALID_BOUND_MESSAGE")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self.closed and not self.lease._compute_released:
                self.error = error
                self.opened.set()
                await self.queue.put(error)

    def observe_persistence(self, receipt, descriptor):
        if self.observer is not None:
            raise PoolError("INVALID_PREFILL_RECEIPT")

        async def observe():
            await self.memory_ready.wait()
            await observe_prefill_persistence(
                self.lease.control,
                self.bound.identity(),
                receipt,
                descriptor,
                emit=self.emit,
                closed=lambda: self.closed,
            )

        self.observer = asyncio.create_task(observe())

    async def close(self):
        self.closed = True
        tasks = [task for task in (self.reader, self.observer) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class DisaggregatedPrefillLeases(DisaggregatedModelLeases):
    @asynccontextmanager
    async def open(self):
        async with super().open() as (bound, assignment):
            socket = _PrefillSocket(self, bound, assignment)
            try:
                await socket.start()
                yield socket, assignment
            finally:
                await socket.close()
