# SPDX-License-Identifier: Apache-2.0
"""Atomic request admission and bounded CPU work, without artifact ownership."""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import threading
from dataclasses import dataclass, fields

from .contract import ConditionError


@dataclass(frozen=True)
class ResourceLimits:
    max_active_requests: int = 32
    max_active_requests_per_tenant: int = 16
    max_inflight_bytes: int = 8 << 30
    max_inflight_bytes_per_tenant: int = 2 << 30
    cache_bytes: int = 256 << 20
    cache_bytes_per_tenant: int = 256 << 20
    max_request_bytes: int = 512 << 20
    max_response_bytes: int = 256 << 20
    max_input_bytes: int = 512 << 20
    max_condition_bytes: int = 256 << 20
    max_condition_rows: int = 16384
    max_header_bytes: int = 65536
    max_prompt_bytes: int = 65536
    max_media: int = 16
    max_side: int = 8192
    max_decoded_pixels: int = 178956970
    max_queued_encodes: int = 16
    request_timeout_s: float = 10.0
    cpu_executor_active: int = 2
    cpu_executor_waiting: int = 4

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Conditioning limits must be finite and positive")
            if field.name != "request_timeout_s" and type(value) is not int:
                raise ValueError("Conditioning capacities must be integers")
        if self.max_header_bytes > 65536 or self.max_media > 16:
            raise ValueError("Conditioning header or media limit exceeds the protocol")
        for total, tenant in self.budgets().values():
            if tenant > total:
                raise ValueError("Tenant conditioning budget exceeds global budget")

    def budgets(self):
        return {
            "slots": (self.max_active_requests, self.max_active_requests_per_tenant),
            "inflight": (self.max_inflight_bytes, self.max_inflight_bytes_per_tenant),
        }

    @property
    def output_reservation_bytes(self):
        return self.max_condition_rows * (5120 * 2 + 8)


class Reservation:
    def __init__(self, ledger, tenant, charges):
        self.ledger, self.tenant, self.charges = ledger, tenant, dict(charges)
        self.released = False

    def release(self):
        with self.ledger._lock:
            if self.released:
                return
            self.ledger._subtract(self.tenant, self.charges)
            self.released = True

    def grow(self, **charges):
        """Atomically charge streamed bytes without one owner per tiny chunk."""
        with self.ledger._lock:
            if self.released:
                raise ValueError("Cannot grow a released conditioning reservation")
            extra = self.ledger.reserve(self.tenant, **charges)
            for key, value in charges.items():
                self.charges[key] = self.charges.get(key, 0) + value
            # Ownership moves into this reservation; totals do not change.
            extra.released = True

    def split(self, **charges):
        """Move part of this reservation to a new one; ledger totals are unchanged."""
        with self.ledger._lock:
            if self.released or any(
                value < 0 or value > self.charges.get(key, 0)
                for key, value in charges.items()
            ):
                raise ValueError("Invalid conditioning reservation split")
            for key, value in charges.items():
                self.charges[key] -= value
            return Reservation(self.ledger, self.tenant, charges)

    def reduce(self, **charges):
        """Shrink an existing reservation; growth requires a separate reserve."""
        with self.ledger._lock:
            if self.released or any(
                value < 0 or value > self.charges.get(key, 0)
                for key, value in charges.items()
            ):
                raise ValueError("Invalid conditioning reservation reduction")
            removed = {key: self.charges[key] - value for key, value in charges.items()}
            self.ledger._subtract(self.tenant, removed)
            self.charges.update(charges)


class AdmissionLedger:
    def __init__(self, budgets):
        if not budgets or any(
            not isinstance(total, int)
            or not isinstance(tenant, int)
            or total <= 0
            or tenant <= 0
            or tenant > total
            for total, tenant in budgets.values()
        ):
            raise ValueError("Invalid conditioning budgets")
        self.budgets = dict(budgets)
        self._global = {key: 0 for key in budgets}
        self._tenants = {}
        self._lock = threading.RLock()

    def reserve(self, tenant, **charges):
        if not isinstance(tenant, str) or not tenant:
            raise ValueError("Invalid conditioning tenant")
        if any(
            key not in self.budgets
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for key, value in charges.items()
        ):
            raise ValueError("Invalid conditioning charges")
        with self._lock:
            current = self._tenants.get(tenant, {})
            for key, value in charges.items():
                total, scoped = self.budgets[key]
                if (
                    self._global[key] + value > total
                    or current.get(key, 0) + value > scoped
                ):
                    raise ConditionError("CONDITION_BUSY", 429)
            current = self._tenants.setdefault(tenant, {key: 0 for key in self.budgets})
            for key, value in charges.items():
                self._global[key] += value
                current[key] += value
            return Reservation(self, tenant, charges)

    def _subtract(self, tenant, charges):
        current = self._tenants[tenant]
        for key, value in charges.items():
            self._global[key] -= value
            current[key] -= value
        if not any(current.values()):
            del self._tenants[tenant]

    def snapshot(self):
        with self._lock:
            return {
                "global": dict(self._global),
                "tenants": {key: dict(value) for key, value in self._tenants.items()},
            }


class ExecutorReservation:
    def __init__(self, executor):
        self.executor = executor
        self.started = False
        self.released = False

    def release(self):
        if not self.released:
            self.released = True
            self.executor.in_use -= 1


class BoundedExecutor:
    """No unbounded executor submit queue; cancellation waits for native work."""

    def __init__(self, active=2, waiting=4):
        if active <= 0 or waiting < 0:
            raise ValueError("Invalid conditioning executor capacity")
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=active, thread_name_prefix="conditioning-cpu"
        )
        self.capacity = active + waiting
        self.in_use = 0
        self._closed = False

    def reserve(self):
        # All calls execute on the owning asyncio loop; no await before reserve.
        if self._closed or self.in_use >= self.capacity:
            raise ConditionError("CONDITION_BUSY", 429)
        self.in_use += 1
        return ExecutorReservation(self)

    async def run(self, function, *args, **kwargs):
        return await self.run_reserved(self.reserve(), function, *args, **kwargs)

    async def run_reserved(self, reservation, function, *args, **kwargs):
        if (
            reservation.executor is not self
            or reservation.released
            or reservation.started
        ):
            raise ValueError("Invalid conditioning CPU reservation")
        reservation.started = True
        try:
            future = asyncio.get_running_loop().run_in_executor(
                self.pool, lambda: function(*args, **kwargs)
            )
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                # An async waiter disappearing cannot free buffers used by a
                # running decoder/hash operation. Repeated cancellation drains.
                while not future.done():
                    try:
                        await asyncio.shield(future)
                    except asyncio.CancelledError:
                        continue
                future.exception() if not future.cancelled() else None
                raise
        finally:
            reservation.release()

    async def close(self):
        self._closed = True
        while self.in_use:
            await asyncio.sleep(0.001)
        self.pool.shutdown(wait=True, cancel_futures=False)
