# SPDX-License-Identifier: Apache-2.0
"""One coalesced source-demand notification, owned by a generation loop.

Demand is advisory; source acceptance/consumption still enforce hard credit.
Never make GPU dispatch wait for the shared RGB writer, or accumulate snapshots
behind it. Construct the latest coherent snapshot only after acquiring the lock.
"""

import asyncio
from contextlib import nullcontext

import msgspec.msgpack


class SourceDemandPublisher:
    def __init__(self, adapter, session, websocket):
        self.adapter, self.session, self.websocket = adapter, session, websocket
        self.failed = asyncio.get_running_loop().create_future()
        self.task = None
        self.pending = False
        self.waiting = False
        self.closed = False

    def publish(self, *, waiting=False):
        if self.closed:
            raise RuntimeError("Source demand publisher is closed")
        if self.failed.done():
            raise self.failed.result()
        # Measure actual input waits at the scheduling boundary, independently
        # of when a slow transport can publish its latest observation.
        observe = getattr(self.adapter, "observe_source_input_wait", None)
        if observe is not None:
            observe(waiting_for_input=waiting)
        self.waiting, self.pending = waiting, True
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._send(), name="source-demand-send")

    async def _send(self):
        try:
            while self.pending:
                group = getattr(self.websocket, "send_group", None)
                async with group() if group is not None else nullcontext():
                    # No await between selecting the state and serializing it.
                    self.pending = False
                    message = self.adapter.source_demand_message(
                        self.session, waiting_for_input=self.waiting
                    )
                    if message is not None:
                        await self.websocket.send_bytes(msgspec.msgpack.encode(message))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # A result (rather than an exception future) avoids unobserved
            # exceptions when cancellation and a broken writer coincide.
            if not self.failed.done():
                self.failed.set_result(error)

    async def run(self, generation):
        """Abort blocked input/work on writer failure and join both owned tasks."""
        work = asyncio.create_task(generation, name="source-demand-generation")
        try:
            await asyncio.wait((work, self.failed), return_when=asyncio.FIRST_COMPLETED)
            if self.failed.done():
                raise self.failed.result()
            return await work
        finally:
            self.closed = True
            if not work.done():
                work.cancel()
            if self.task is not None and not self.task.done():
                self.task.cancel()
            await asyncio.gather(
                work,
                *([self.task] if self.task is not None else []),
                return_exceptions=True,
            )
            self.pending = False
