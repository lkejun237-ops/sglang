# SPDX-License-Identifier: Apache-2.0
"""A source owns one model reservation; segments only borrow compute."""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import uuid4

import msgspec

from .diagnostics import log_player_failure
from .render_contract import RenderError


class BoundChild:
    def __init__(self, parent, kind, session_id, generation_id):
        self.parent, self.kind = parent, kind
        self.supports_condition_barrier = getattr(
            getattr(parent, "lease", None), "supports_condition_barrier", True
        )
        self.session_id, self.generation_id = session_id, generation_id
        self.id = uuid4().hex
        self.opened = asyncio.Event()
        self.completed = asyncio.Event()
        self.messages = asyncio.Queue(maxsize=8)
        self.resource_state = "none"
        self.failure = None
        self.closing = False

    async def send(self, payload):
        if self.completed.is_set():
            raise RenderError("MODEL_UNAVAILABLE")
        await self.parent.send(
            dict(type="bound.data", operation_id=self.id, payload=payload)
        )

    async def recv(self):
        incoming = asyncio.create_task(self.messages.get())
        done = asyncio.create_task(self.completed.wait())
        try:
            await asyncio.wait((incoming, done), return_when=asyncio.FIRST_COMPLETED)
            if incoming.done():
                return incoming.result()
            if not self.messages.empty():
                return self.messages.get_nowait()
            raise RenderError(self.failure or "MODEL_UNAVAILABLE")
        finally:
            for task in (incoming, done):
                task.cancel()
            await asyncio.gather(incoming, done, return_exceptions=True)

    async def close(self):
        self.closing = True
        while not self.messages.empty():
            self.messages.get_nowait()
        if not self.completed.is_set():
            await self.parent.send(dict(type="bound.cancel", operation_id=self.id))
            await asyncio.wait_for(self.completed.wait(), self.parent.timeout)
        if self.resource_state != "released":
            raise RenderError("RELEASE_UNKNOWN")

    async def retry_release(self):
        # A verified parent release covers this exact bound child, without
        # inventing a condition application or trusting an absent child receipt.
        if (
            self.parent.closed
            and getattr(self.parent.lease, "resource_state", None) == "released"
        ):
            self.resource_state = "released"
            return
        raise RenderError("RELEASE_UNKNOWN")

    @asynccontextmanager
    async def open(self):
        async with self.parent.compute_lock:
            if not self.parent.ready.is_set() or self.parent.closed:
                raise RenderError("MODEL_UNAVAILABLE")
            self.parent.child = self
            self.resource_state = "reserved"
            try:
                await self.parent.send(
                    dict(
                        type="bound.open",
                        operation_id=self.id,
                        kind=self.kind,
                        session_id=self.session_id,
                        generation_id=self.generation_id,
                    )
                )
                await asyncio.wait_for(self.opened.wait(), self.parent.timeout)
                if self.failure:
                    raise RenderError(self.failure)
                yield (
                    self,
                    replace(
                        self.parent.assignment,
                        session_id=self.session_id,
                        generation_id=self.generation_id,
                    ),
                )
            finally:
                try:
                    await self.close()
                finally:
                    if self.parent.child is self:
                        self.parent.child = None


class ModelBinding:
    def __init__(self, lease, on_receipt, *, timeout=120):
        self.lease, self.on_receipt, self.timeout = lease, on_receipt, timeout
        self.lease.operation_path = "/v1/realtime_video/bound"
        self.ready, self.stopped = asyncio.Event(), asyncio.Event()
        self.compute_lock, self.send_lock = asyncio.Lock(), asyncio.Lock()
        self.child = self.ws = self.assignment = self.task = None
        self.closed = False
        self.error = None
        self.close_task = None

    def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self._run())

    async def wait_ready(self):
        self.start()
        await asyncio.wait_for(self.ready.wait(), self.timeout)
        if self.error:
            raise self.error

    async def send(self, message):
        async with self.send_lock:
            await self.ws.send(msgspec.msgpack.encode(message))

    def borrow(self, segment, *, kind="generate"):
        return BoundChild(
            self, kind, segment["model_session_id"], segment["generation_id"]
        )

    async def _run(self):
        try:
            async with self.lease.open() as (self.ws, self.assignment):
                first = msgspec.msgpack.decode(await self.ws.recv())
                if (
                    first.get("type") != "bound.ready"
                    or first.get("protocol") != "bound-compute-v1"
                    or first.get("worker_epoch")
                    != self.assignment.denoiser.worker_epoch
                    or first.get("player_expires_at")
                    != getattr(self.assignment, "player_expires_at", None)
                ):
                    raise RenderError("CAPABILITY_UNAVAILABLE")
                self.ready.set()
                while not self.closed:
                    message = msgspec.msgpack.decode(await self.ws.recv())
                    if message.get("type", "").startswith("prefill."):
                        await self.on_receipt(message)
                        continue
                    child = self.child
                    if child is None or message.get("operation_id") != child.id:
                        raise RenderError("SOURCE_EPOCH_MISMATCH")
                    if message["type"] == "bound.opened":
                        child.opened.set()
                    elif message["type"] == "bound.data":
                        if not isinstance(message.get("payload"), bytes):
                            raise RenderError("INVALID_COMMAND")
                        if not child.closing:
                            await child.messages.put(message["payload"])
                    elif message["type"] == "bound.completed":
                        child.resource_state = (
                            "released"
                            if message.get("cleaned") is True
                            else "release_unknown"
                        )
                        child.failure = message.get("code")
                        child.completed.set()
                    else:
                        raise RenderError("INVALID_COMMAND")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self.closed:
                logging.getLogger(__name__).exception("Bound model stream failed")
            self.error = error
            if not self.closed:
                log_player_failure("model.connection", error, binding=self)
        finally:
            self.closed = True
            self.ready.set()
            if self.child:
                self.child.failure = "MODEL_UNAVAILABLE"
                self.child.opened.set()
                self.child.completed.set()
            self.stopped.set()

    async def close(self):
        # Share one in-flight closer. A later explicit close may reconcile the
        # same stopped lease after a lost release receipt, never admit a new one.
        if self.close_task is not None and self.close_task.done():
            if self.close_task.cancelled() or self.close_task.exception() is not None:
                self.close_task = None
        if self.close_task is None:
            retrying = self.closed and self.task is not None and self.task.done()

            async def cleanup():
                self.closed = True
                if self.ws is not None:
                    await self.ws.close()
                elif self.task is not None and not self.task.done():
                    self.task.cancel()
                if self.task is not None:
                    try:
                        await asyncio.wait_for(asyncio.shield(self.task), self.timeout)
                    except asyncio.CancelledError:
                        if not self.task.cancelled():
                            raise
                retry = getattr(self.lease, "retry_release", None)
                if retrying and retry is not None:
                    await retry()
                if self.lease.resource_state not in {"none", "released"}:
                    raise RenderError("RELEASE_UNKNOWN")

            self.close_task = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(self.close_task)
        except BaseException as error:
            log_player_failure("model.release", error, binding=self, outcome="unknown")
            raise
