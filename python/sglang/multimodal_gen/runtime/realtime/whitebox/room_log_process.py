# SPDX-License-Identifier: Apache-2.0
"""Bounded, ordered room journal writes outside the authoritative CLI process."""

import asyncio
import copy
import hashlib
import time

from .metrics import observe_duration, stage
from .recording import json_line, write_manifest
from .recording_process import process_pool

_file = None


def _boot(directory, manifest):
    global _file
    directory.mkdir(parents=True)
    _file = (directory / "inputs.jsonl").open("w")
    write_manifest(directory, manifest)


def _append(line, enqueued):
    start = time.monotonic()
    _file.write(line)
    _file.flush()
    return (start - enqueued) * 1000, (time.monotonic() - start) * 1000


def _finish(directory, manifest, outcome):
    _file.close()
    path = directory / "inputs.jsonl"
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest.update(
        recording_status="complete" if outcome == "completed" else "partial",
        execution_outcome=outcome,
        objects={
            "inputs.jsonl": {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        },
        journal_worker={"process": True, "pending": 0},
    )
    write_manifest(directory, manifest)
    return manifest


class RoomJournal:
    def __init__(self, directory, trace_id, cpu):
        self.directory, self.trace_id = directory, trace_id
        self.pool = process_pool(cpu)
        # Each event is limited to 64 KiB; at most 128 queued plus one in flight.
        self.queue = asyncio.Queue(maxsize=128)
        self.task = None

    async def start(self, manifest):
        try:
            await asyncio.get_running_loop().run_in_executor(
                self.pool, _boot, self.directory, copy.deepcopy(manifest)
            )
            self.task = asyncio.create_task(self._run(), name="whitebox-room-journal")
        except BaseException:
            await asyncio.to_thread(self.pool.shutdown, wait=True, cancel_futures=True)
            raise

    async def append(self, event):
        line = json_line(event)
        if len(line.encode()) > 65536:
            raise ValueError("The room input is too large to archive")
        put = asyncio.create_task(self.queue.put((line, time.monotonic())))
        try:
            with stage("room_journal_backpressure", self.trace_id):
                done, _ = await asyncio.wait(
                    (put, self.task), return_when=asyncio.FIRST_COMPLETED
                )
                if self.task in done:
                    self.task.result()
                    raise RuntimeError("Room journal stopped")
                await put
        finally:
            if not put.done():
                put.cancel()
            await asyncio.gather(put, return_exceptions=True)

    async def _run(self):
        while True:
            item = await self.queue.get()
            if item is None:
                return
            wait, write = await asyncio.get_running_loop().run_in_executor(
                self.pool, _append, *item
            )
            observe_duration("room_journal_queue_wait", wait, self.trace_id)
            observe_duration("room_journal_write", write, self.trace_id)

    async def finish(self, manifest, outcome):
        try:
            if self.task.done():
                self.task.result()
            else:
                put = asyncio.create_task(self.queue.put(None))
                try:
                    done, _ = await asyncio.wait(
                        (put, self.task), return_when=asyncio.FIRST_COMPLETED
                    )
                    if self.task in done:
                        self.task.result()
                    else:
                        await self.task
                finally:
                    put.cancel()
                    await asyncio.gather(put, return_exceptions=True)
            return await asyncio.get_running_loop().run_in_executor(
                self.pool, _finish, self.directory, copy.deepcopy(manifest), outcome
            )
        finally:
            await asyncio.to_thread(self.pool.shutdown, wait=True, cancel_futures=True)
