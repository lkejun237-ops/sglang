# SPDX-License-Identifier: Apache-2.0
"""Single active task lease; GPU state is touched only on the encode executor."""

import asyncio
import time
import uuid
from dataclasses import dataclass

from .contract import ConditionError


def validate_task_id(value):
    if type(value) is not str or len(value) != 32:
        raise ConditionError()
    try:
        if uuid.UUID(hex=value).hex != value:
            raise ValueError()
    except ValueError:
        raise ConditionError() from None
    return value


@dataclass
class PrefixTask:
    tenant: str
    nonce: str
    handle: str
    cache: object
    last_seen: float
    busy: int = 0
    closed: bool = False
    expiry: object = None
    release: object = None


class SinglePrefixTask:
    def __init__(self, factory, executor, idle_s, clock=time.monotonic):
        self.factory, self.executor = factory, executor
        self.idle_s, self.clock = idle_s, clock
        self.active = None
        self.closed = False

    def open(self, tenant, nonce):
        validate_task_id(nonce)
        if self.closed:
            raise ConditionError("CONDITION_UNAVAILABLE")
        task = self.active
        if task is not None:
            if task.tenant == tenant and task.nonce == nonce and not task.closed:
                task.last_seen = self.clock()
                return task.handle
            raise ConditionError("CONDITION_BUSY")
        task = PrefixTask(tenant, nonce, uuid.uuid4().hex, self.factory(), self.clock())
        self.active = task
        task.expiry = asyncio.create_task(self._expire(task))
        return task.handle

    def acquire(self, tenant, handle):
        validate_task_id(handle)
        task = self.active
        if (
            task is None
            or task.closed
            or (task.tenant, task.handle) != (tenant, handle)
        ):
            raise ConditionError("CONDITION_TASK_EXPIRED")
        task.busy += 1
        task.last_seen = self.clock()
        return task

    def leave(self, task):
        task.busy -= 1
        task.last_seen = self.clock()

    async def _expire(self, task):
        while not task.closed:
            await asyncio.sleep(min(self.idle_s, 30))
            if not task.busy and self.clock() - task.last_seen >= self.idle_s:
                await self.release(task.tenant, task.handle)
                return

    async def release(self, tenant, handle):
        validate_task_id(handle)
        task = self.active
        if task is None or (task.tenant, task.handle) != (tenant, handle):
            return  # Idempotent DELETE must never clear a newer task.
        if task.release is None:
            task.closed = True
            if task.expiry is not asyncio.current_task():
                task.expiry.cancel()
            task.release = asyncio.create_task(self._drain(task))
        # Disconnection must not cancel cleanup or reopen admission early.
        await asyncio.shield(task.release)

    async def _drain(self, task):
        try:
            # Queued encodes observe closed and skip. A running forward drains
            # before this callback, on the same single-thread executor.
            await asyncio.get_running_loop().run_in_executor(
                self.executor, task.cache.clear
            )
        finally:
            if self.active is task:
                self.active = None

    async def close(self):
        self.closed = True
        if self.active is not None:
            await self.release(self.active.tenant, self.active.handle)
