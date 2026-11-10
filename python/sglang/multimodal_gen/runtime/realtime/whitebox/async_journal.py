# SPDX-License-Identifier: Apache-2.0
"""Nonblocking ordered game journal, independent of live input delivery."""

import asyncio
import copy
import multiprocessing
import queue
import time
from collections import deque

from .recording import json_line, room_event_max_bytes


def _journal(commands, replies, directory, manifest, cpu):
    from .async_recording import limit_child
    from .room_log_process import _append, _boot, _finish

    limit_child(cpu, file_bytes=64 * 1024**2)
    try:
        _boot(directory, manifest)
        replies.put(("ready", None))
        while True:
            kind, value = commands.get()
            if kind == "finish":
                replies.put(("finished", _finish(directory, *value)))
                return
            _append(value, time.monotonic())
            replies.put(("ack", None))
    except BaseException:
        replies.put(("failed", None))


class AsyncRoomJournal:
    def __init__(self, directory, trace_id, cpu):
        self.directory, self.trace_id, self.cpu = directory, trace_id, cpu
        self.queue = deque()
        self.bytes = 0
        self.failure = None
        self.manifest = None
        self.task = None
        self.ending = None
        self.wake = asyncio.Event()

    async def start(self, manifest):
        self.manifest = copy.deepcopy(manifest)
        self.task = asyncio.create_task(self._run(), name="room-input-journal")

    async def append(self, event):
        if self.failure or self.ending:
            return
        try:
            line = json_line(event)
            size = len(line.encode())
        except (ValueError, TypeError, RecursionError, MemoryError):
            self.failure = "JOURNAL_INVALID_METADATA"
            self.wake.set()
            return
        if (
            size > room_event_max_bytes(event)
            or self.bytes + size > 8 * 1024**2
            or len(self.queue) >= 128
        ):
            self.failure = "JOURNAL_QUEUE_FULL"
            self.wake.set()
            return
        self.queue.append((time.monotonic(), size, line))
        self.bytes += size
        self.wake.set()

    async def _run(self):
        ctx = multiprocessing.get_context("spawn")
        commands, replies = ctx.Queue(1), ctx.Queue(1)
        child = ctx.Process(
            target=_journal,
            args=(commands, replies, self.directory, self.manifest, self.cpu),
            daemon=True,
        )

        async def reply(timeout=5):
            async with asyncio.timeout(timeout):
                while True:
                    if self.failure:
                        raise RuntimeError("JOURNAL_WORKER_FAILED")
                    try:
                        value = replies.get_nowait()
                        if value[0] == "failed":
                            raise RuntimeError("JOURNAL_WORKER_FAILED")
                        return value
                    except queue.Empty:
                        if not child.is_alive():
                            raise RuntimeError("JOURNAL_WORKER_EXITED")
                        await asyncio.sleep(0.01)

        try:
            await asyncio.to_thread(child.start)
            await reply(15)
            while not self.failure:
                if self.queue:
                    when, size, line = self.queue.popleft()
                    if time.monotonic() - when > 5:
                        raise TimeoutError("JOURNAL_QUEUE_TOO_OLD")
                    commands.put_nowait(("append", line))
                    await reply()
                    self.bytes -= size
                elif self.ending:
                    commands.put_nowait(("finish", self.ending))
                    kind, result = await reply(15)
                    if kind != "finished":
                        raise RuntimeError("JOURNAL_FINISH_FAILED")
                    self.manifest = result
                    return
                else:
                    self.wake.clear()
                    try:
                        await asyncio.wait_for(self.wake.wait(), 0.2)
                    except TimeoutError:
                        if not child.is_alive():
                            raise RuntimeError("JOURNAL_WORKER_EXITED")
        except (Exception, asyncio.CancelledError):
            self.failure = self.failure or "JOURNAL_WORKER_FAILED"
        finally:
            if child.pid:
                if child.is_alive():
                    child.terminate()
                await asyncio.to_thread(child.join, 1)
                if child.is_alive():
                    child.kill()
                    await asyncio.to_thread(child.join, 1)
            for pipe in (commands, replies):
                pipe.cancel_join_thread()
                pipe.close()
            self.queue.clear()
            self.bytes = 0
            if self.failure:
                self.manifest.update(
                    recording_status="failed",
                    archive_failure=self.failure,
                    upload_status="failed",
                )

                from .async_recording import persist_outcome

                await persist_outcome(self.directory, self.manifest)

    async def finish(self, manifest, outcome):
        self.ending = (copy.deepcopy(manifest), outcome)
        self.wake.set()
        await self.task
        return self.manifest
