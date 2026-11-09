# SPDX-License-Identifier: Apache-2.0
"""Drain discretionary maintenance before returning its admission ownership."""

import asyncio
import os
import signal

import anyio


async def finish_cleanup(awaitable):
    """A disconnect or repeated play cancellation must not interrupt reaping."""
    task = asyncio.ensure_future(awaitable)
    interrupted = False
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                interrupted = True
        result = task.result()
    if interrupted:
        raise asyncio.CancelledError
    return result


async def terminate_process(process):
    if process is None or process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        await asyncio.wait_for(process.wait(), 5)
    except (ProcessLookupError, asyncio.TimeoutError):
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
