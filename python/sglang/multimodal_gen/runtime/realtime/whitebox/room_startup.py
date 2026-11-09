# SPDX-License-Identifier: Apache-2.0
"""Positive cleanup receipts from the original factory and Room owners.

Absence of a driver or a process-local record cannot substitute for either
receipt. Cancellation alone proves nothing; the original owner may acknowledge
its completed cleanup while still propagating cancellation to its caller.
"""

import asyncio
from dataclasses import dataclass

from .diagnostics import WhiteboxDiagnostic


@dataclass(frozen=True)
class FactoryCleanupReceipt:
    operation_id: str


@dataclass(frozen=True)
class RoomCleanupReceipt:
    factory: FactoryCleanupReceipt
    owner_epoch: str


@dataclass
class StartupScope:
    """Original local join's record of whether Room/SDK allocation was invoked."""

    entered: bool = False
    allocation_started: bool = False


@dataclass(frozen=True)
class MemberCleanupReceipt:
    """The original driver confirmed retirement of this attempted member."""

    player_id: str


class RoomStartupFailure(WhiteboxDiagnostic):
    def __init__(self, code, cleanup: FactoryCleanupReceipt | RoomCleanupReceipt):
        super().__init__(code)
        self.cleanup = cleanup


def startup_cleanup_error(error, cleanup):
    if isinstance(error, asyncio.CancelledError):
        # asyncio.timeout checks the exact CancelledError type on supported
        # Python versions. A subclass would break timeout conversion.
        cancelled = asyncio.CancelledError(*error.args)
        cancelled._whitebox_startup_cleanup = cleanup
        cancelled.code = "CLI_STARTUP_CANCELLED"
        return cancelled
    return RoomStartupFailure(getattr(error, "code", "CLI_STARTUP_FAILED"), cleanup)


def startup_cleanup_receipt(error):
    if isinstance(error, RoomStartupFailure):
        return error.cleanup
    if type(error) is asyncio.CancelledError:
        return getattr(error, "_whitebox_startup_cleanup", None)
    return None


async def drain_startup_cleanup(awaitable):
    """Finish the original cleanup; return whether the caller was cancelled.

    Callers must propagate that cancellation after recording the owner's result.
    A failed or cancelled cleanup itself still raises and cannot issue a receipt.
    """
    from .maintenance_process import finish_cleanup

    task = asyncio.ensure_future(awaitable)
    try:
        return await finish_cleanup(task), False
    except asyncio.CancelledError:
        return task.result(), True
