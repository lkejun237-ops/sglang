# SPDX-License-Identifier: Apache-2.0

"""Production realtime worker coordination and fenced Session leases."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Callable, Literal, Mapping, Protocol
from uuid import uuid4

from sglang.multimodal_gen.runtime.realtime.critical_path_metrics import (
    observe_coordinator_capacity_reservation_seconds,
    observe_coordinator_worker_selection_seconds,
    result_from_exception,
)

WorkerRole = Literal["denoiser", "vae"]
WorkerLifecycle = Literal["ready", "draining", "failed"]
logger = logging.getLogger(__name__)
_IDEMPOTENT_WORKER_RELEASE_REASONS = frozenset(
    {
        "RESERVATION_NOT_FOUND",
        "RESERVATION_OWNER_MISMATCH",
    }
)


class CoordinatorRejected(RuntimeError):
    def __init__(self, reason: str, *, retry_after_s: float | None = None) -> None:
        self.reason = reason
        self.retry_after_s = retry_after_s
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class WorkerHeartbeat:
    worker_id: str
    role: WorkerRole
    endpoint: str
    az: str
    capacity: int
    model_revision: str
    vae_fingerprint: str
    routing_revision: str = ""
    worker_epoch: str = ""
    lifecycle: WorkerLifecycle = "ready"
    active_sessions: int = 0
    runnable_sessions: int = 0
    blocked_sessions: int = 0
    queue_depth: int = 0
    service_time_ms: float = 0.0
    oldest_consumed_age_s: float = 0.0
    stale_consumed_reservations: int = 0
    last_progress_age_s: float = 0.0
    reservation_endpoint: str = ""
    drain_deadline: float | None = None


@dataclass(frozen=True, slots=True)
class WorkerSlot:
    worker_id: str
    role: WorkerRole
    endpoint: str
    az: str
    slot_index: int
    model_revision: str
    vae_fingerprint: str
    routing_revision: str = ""
    worker_epoch: str = ""
    lifecycle: WorkerLifecycle = "ready"
    active_sessions: int = 0
    runnable_sessions: int = 0
    blocked_sessions: int = 0
    queue_depth: int = 0
    service_time_ms: float = 0.0
    reservation_endpoint: str = ""
    drain_deadline: float | None = None
    capacity: int = 1


@dataclass(frozen=True, slots=True)
class SessionAssignment:
    user_id: str
    session_id: str
    generation_id: str
    token: str
    expires_at: float
    denoiser: WorkerSlot
    vae: WorkerSlot | None
    admitted_at: float | None = None
    player_expires_at: float | None = None

    @property
    def workers(self) -> tuple[WorkerSlot, ...]:
        return (self.denoiser,) if self.vae is None else (self.denoiser, self.vae)


class CoordinatorStore(Protocol):
    async def heartbeat(self, heartbeat: WorkerHeartbeat) -> None: ...

    async def claim_attempt(
        self, *, attempt_id: str, owner_id: str, expires_at: float
    ) -> None: ...

    async def acquire(
        self,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        model_revision: str,
        vae_fingerprint: str,
        excluded_workers: frozenset[tuple[WorkerRole, str]] = frozenset(),
    ) -> SessionAssignment: ...

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment: ...

    async def release(self, assignment: SessionAssignment) -> None: ...

    async def waiting_started(self, waiter_id: str) -> None: ...

    async def waiting_finished(self, waiter_id: str) -> None: ...

    async def capacity_snapshot(
        self, *, model_revision: str | None = None, vae_fingerprint: str | None = None
    ) -> dict[str, Any]: ...


class WorkerReservationClient(Protocol):
    async def reserve(
        self,
        slot: WorkerSlot,
        *,
        token: str,
        session_id: str,
        generation_id: str,
        ttl_s: float,
    ) -> None: ...

    async def release(self, slot: WorkerSlot, *, token: str) -> None: ...


class HTTPWorkerReservationClient:
    def __init__(
        self,
        *,
        timeout_s: float = 2.0,
        client: Any | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None

    def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def reserve(
        self,
        slot: WorkerSlot,
        *,
        token: str,
        session_id: str,
        generation_id: str,
        ttl_s: float,
        player_expires_at: float | None = None,
    ) -> None:
        response = await self._get_client().post(
            f"{slot.reservation_endpoint.rstrip('/')}/reservations",
            json={
                "token": token,
                "session_id": session_id,
                "generation_id": generation_id,
                "worker_epoch": slot.worker_epoch,
                "ttl_s": ttl_s,
                **(
                    {"player_expires_at": player_expires_at}
                    if player_expires_at is not None
                    else {}
                ),
            },
        )
        response.raise_for_status()

    async def release(self, slot: WorkerSlot, *, token: str) -> None:
        response = await self._get_client().delete(
            f"{slot.reservation_endpoint.rstrip('/')}/reservations/{token}"
        )
        if response.status_code == 404:
            return
        if response.status_code == 409:
            try:
                detail = response.json().get("detail", {})
            except (ValueError, AttributeError):
                detail = {}
            if detail.get("reason") in _IDEMPOTENT_WORKER_RELEASE_REASONS:
                return
        response.raise_for_status()

    async def release_verified(
        self, slot: WorkerSlot, *, token: str, player=None
    ) -> None:
        if player is not None:
            from .player_lifecycle import player_close_id

            payload = dict(
                session_id=player.session_id,
                generation_id=player.generation_id,
                worker_epoch=slot.worker_epoch,
                close_id=player_close_id(player),
                player_expires_at=player.player_expires_at,
            )
            base = f"{slot.reservation_endpoint.rstrip('/')}/reservations/{token}"
            prepared = await self._get_client().post(
                base + "/prepare-release", json=payload
            )
            if prepared.status_code != 204:
                raise CoordinatorRejected("RELEASE_PENDING")
            response = await self._get_client().post(base + "/finalize", json=payload)
            if response.status_code != 200:
                raise CoordinatorRejected("RELEASE_PENDING")
            receipt = response.json()
            if receipt.get("state") != "released" or any(
                receipt.get(key) != payload[key]
                for key in ("session_id", "generation_id", "worker_epoch", "close_id")
            ):
                raise CoordinatorRejected("RELEASE_PENDING")
            return
        response = await self._get_client().delete(
            f"{slot.reservation_endpoint.rstrip('/')}/reservations/{token}",
            params={"worker_epoch": slot.worker_epoch},
        )
        # OWNER_MISMATCH means a live handler still owns cleanup. A generic
        # HTTP 404 is not evidence that the assigned worker finished cleanup.
        if response.status_code != 204:
            raise CoordinatorRejected("RELEASE_PENDING")

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


@dataclass(slots=True)
class _WorkerState:
    heartbeat: WorkerHeartbeat
    updated_at: float


def _routing_revision(worker: WorkerHeartbeat | WorkerSlot) -> str:
    return worker.routing_revision or worker.model_revision


def _release_identity(assignment: SessionAssignment) -> str:
    # Lease expiry and load estimates can change during renewal; execution and
    # endpoint identities cannot. Never follow endpoints supplied only by a peer.
    return json.dumps(
        [
            assignment.user_id,
            assignment.session_id,
            assignment.generation_id,
            assignment.token,
            assignment.admitted_at,
            assignment.player_expires_at,
            [
                [
                    slot.role,
                    slot.worker_id,
                    slot.worker_epoch,
                    slot.slot_index,
                    slot.endpoint,
                    slot.reservation_endpoint,
                ]
                for slot in assignment.workers
            ],
        ],
        separators=(",", ":"),
    )


def _record_player_cleanup(closures, assignment, receipt):
    from .player_lifecycle import validate_sdk_cleanup

    current = closures.get(assignment.token)
    if receipt is None:
        if current is None:
            raise CoordinatorRejected("SDK_CLEANUP_PENDING")
        return current
    try:
        value = validate_sdk_cleanup(assignment, receipt)
    except ValueError as error:
        raise CoordinatorRejected("INVALID_SDK_CLEANUP") from error
    if current is not None and current["sdk"] != value:
        raise CoordinatorRejected("CLOSE_IDENTITY_MISMATCH")
    if current is None:
        current = dict(
            identity=_release_identity(assignment), state="releasing", sdk=value
        )
        closures[assignment.token] = current
    return current


def _retain_release_receipt(receipts, assignment, now):
    receipts[assignment.token] = dict(identity=_release_identity(assignment), at=now)
    for token in list(receipts):
        if receipts[token]["at"] + 86400 < now:
            del receipts[token]
    # Source journals retain user-visible operation results independently.
    # An evicted internal receipt returns LEASE_LOST, never guessed success.
    while len(receipts) > 2048:
        del receipts[min(receipts, key=lambda token: receipts[token]["at"])]


def _normalize_routing_revision_aliases(
    aliases: Mapping[str, object] | None,
) -> dict[str, frozenset[str]]:
    normalized: dict[str, frozenset[str]] = {}
    for alias, revisions in (aliases or {}).items():
        alias_key = str(alias).strip()
        if not alias_key:
            continue
        if isinstance(revisions, str):
            values = [revisions]
        else:
            values = [str(revision) for revision in revisions]  # type: ignore[arg-type]
        normalized[alias_key] = frozenset(
            revision.strip() for revision in values if revision.strip()
        )
    return normalized


def _matches_model_revision(
    worker: WorkerHeartbeat | WorkerSlot,
    requested_revision: str,
    aliases: Mapping[str, frozenset[str]],
) -> bool:
    if _routing_revision(worker) == requested_revision:
        return True
    return worker.model_revision in aliases.get(requested_revision, frozenset())


def _capacity_scope_requested(model_revision, vae_fingerprint) -> bool:
    if model_revision is None and vae_fingerprint is None:
        return False
    if not all(
        isinstance(value, str) and value.strip()
        for value in (model_revision, vae_fingerprint)
    ):
        raise CoordinatorRejected("INVALID_CAPACITY_SCOPE")
    return True


def _matches_worker_request(
    worker: WorkerHeartbeat | WorkerSlot,
    *,
    model_revision: str | None,
    vae_fingerprint: str | None,
    aliases: Mapping[str, frozenset[str]],
) -> bool:
    """Capacity observation and admission resolve the same worker compatibility."""
    if (
        model_revision is not None
        and worker.role == "denoiser"
        and not _matches_model_revision(worker, model_revision, aliases)
    ):
        return False
    if vae_fingerprint is not None and (
        worker.role == "vae" or vae_fingerprint == "local"
    ):
        return worker.vae_fingerprint == vae_fingerprint
    return True


class InMemoryCoordinatorStore:
    """Atomic reference store used by tests and single-process development.

    A single condition protects user leases and both worker-slot reservations.
    An assignment is committed only after a compatible Denoiser/VAE pair is
    available, so a failed admission cannot leak a partial GPU reservation.
    """

    supports_player_lifetimes = True

    def __init__(
        self,
        *,
        ttl_s: float,
        worker_ttl_s: float,
        routing_revision_aliases: Mapping[str, object] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        if worker_ttl_s <= 0:
            raise ValueError("worker_ttl_s must be positive")
        self.ttl_s = ttl_s
        self.worker_ttl_s = worker_ttl_s
        self.routing_revision_aliases = _normalize_routing_revision_aliases(
            routing_revision_aliases
        )
        self._clock = clock
        self._wall_clock = wall_clock
        self._workers: dict[str, _WorkerState] = {}
        self._assignments_by_user: dict[str, SessionAssignment] = {}
        self._assignments_by_token: dict[str, SessionAssignment] = {}
        self._slot_tokens: dict[tuple[WorkerRole, str, int], str] = {}
        self._attempt_owners: dict[str, tuple[str, float]] = {}
        self._waiting: set[str] = set()
        self._quarantined: dict[tuple[str, str], tuple[str, str]] = {}
        self._verified_releases = {}
        self._player_closures = {}
        self._condition = asyncio.Condition()

    @staticmethod
    def _validate_heartbeat(heartbeat: WorkerHeartbeat) -> None:
        if heartbeat.role not in ("denoiser", "vae"):
            raise CoordinatorRejected("INVALID_WORKER_ROLE")
        if not heartbeat.worker_id or not heartbeat.endpoint or not heartbeat.az:
            raise CoordinatorRejected("INVALID_WORKER_IDENTITY")
        if not heartbeat.worker_epoch or not heartbeat.reservation_endpoint:
            raise CoordinatorRejected("INVALID_WORKER_IDENTITY")
        if heartbeat.capacity < 1:
            raise CoordinatorRejected("INVALID_WORKER_CAPACITY")
        if heartbeat.lifecycle not in ("ready", "draining", "failed"):
            raise CoordinatorRejected("INVALID_WORKER_LIFECYCLE")
        if any(
            value < 0
            for value in (
                heartbeat.active_sessions,
                heartbeat.runnable_sessions,
                heartbeat.blocked_sessions,
                heartbeat.queue_depth,
                heartbeat.service_time_ms,
                heartbeat.oldest_consumed_age_s,
                heartbeat.stale_consumed_reservations,
                heartbeat.last_progress_age_s,
            )
        ):
            raise CoordinatorRejected("INVALID_WORKER_LOAD")

    def _release_locked(self, assignment: SessionAssignment) -> None:
        current = self._assignments_by_token.get(assignment.token)
        if current is None:
            return
        self._assignments_by_token.pop(assignment.token, None)
        if self._assignments_by_user.get(current.user_id) == current:
            self._assignments_by_user.pop(current.user_id, None)
        for slot in current.workers:
            key = (slot.role, slot.worker_id, slot.slot_index)
            if self._slot_tokens.get(key) == current.token:
                self._slot_tokens.pop(key, None)

    def _expire_locked(self, now: float) -> bool:
        expired = [
            assignment
            for assignment in self._assignments_by_token.values()
            if assignment.expires_at <= now and assignment.player_expires_at is None
        ]
        for assignment in expired:
            self._release_locked(assignment)
        for assignment in self._assignments_by_token.values():
            if assignment.player_expires_at is not None and (
                assignment.expires_at + 90 <= now
                or assignment.player_expires_at + 90 <= self._wall_clock()
            ):
                for slot in assignment.workers:
                    key = slot.role, slot.worker_id
                    value = slot.worker_epoch, assignment.token
                    if self._quarantined.get(key) != value:
                        logger.error(
                            "Player cleanup unconfirmed; quarantined worker=%s session=%s",
                            slot.worker_id,
                            assignment.session_id,
                        )
                        self._quarantined[key] = value
        return bool(expired)

    def _active_workers_locked(
        self,
        *,
        role: WorkerRole,
        now: float,
        model_revision: str | None,
        vae_fingerprint: str | None,
        excluded_workers: frozenset[tuple[WorkerRole, str]],
    ) -> list[_WorkerState]:
        workers = []
        for state in self._workers.values():
            heartbeat = state.heartbeat
            if heartbeat.role != role:
                continue
            if (heartbeat.role, heartbeat.worker_id) in excluded_workers:
                continue
            if state.updated_at + self.worker_ttl_s <= now:
                continue
            if heartbeat.lifecycle != "ready":
                continue
            if (
                self._quarantined.get((heartbeat.role, heartbeat.worker_id), (None,))[0]
                == heartbeat.worker_epoch
            ):
                continue
            if not _matches_worker_request(
                heartbeat,
                model_revision=model_revision,
                vae_fingerprint=vae_fingerprint,
                aliases=self.routing_revision_aliases,
            ):
                continue
            workers.append(state)
        return workers

    def _free_slots_locked(
        self, workers: list[_WorkerState], *, identity: str
    ) -> list[WorkerSlot]:
        slots: list[WorkerSlot] = []
        for state in workers:
            heartbeat = state.heartbeat
            known = sum(
                (heartbeat.role, heartbeat.worker_id, index) in self._slot_tokens
                for index in range(heartbeat.capacity)
            )
            remaining = heartbeat.capacity - max(known, heartbeat.active_sessions)
            for slot_index in range(heartbeat.capacity):
                if remaining <= 0:
                    break
                key = (heartbeat.role, heartbeat.worker_id, slot_index)
                if key in self._slot_tokens:
                    continue
                remaining -= 1
                slots.append(
                    WorkerSlot(
                        worker_id=heartbeat.worker_id,
                        role=heartbeat.role,
                        endpoint=heartbeat.endpoint,
                        az=heartbeat.az,
                        slot_index=slot_index,
                        model_revision=heartbeat.model_revision,
                        vae_fingerprint=heartbeat.vae_fingerprint,
                        routing_revision=_routing_revision(heartbeat),
                        worker_epoch=heartbeat.worker_epoch,
                        lifecycle=heartbeat.lifecycle,
                        active_sessions=heartbeat.active_sessions,
                        runnable_sessions=heartbeat.runnable_sessions,
                        blocked_sessions=heartbeat.blocked_sessions,
                        queue_depth=heartbeat.queue_depth,
                        service_time_ms=heartbeat.service_time_ms,
                        reservation_endpoint=heartbeat.reservation_endpoint,
                        drain_deadline=heartbeat.drain_deadline,
                        capacity=heartbeat.capacity,
                    )
                )
        slots.sort(
            key=lambda slot: (
                slot.active_sessions / slot.capacity,
                slot.queue_depth,
                slot.service_time_ms,
                hashlib.sha256(
                    f"{identity}:{slot.role}:{slot.worker_id}".encode()
                ).digest(),
                slot.worker_id,
                slot.slot_index,
            )
        )
        return slots

    async def heartbeat(self, heartbeat: WorkerHeartbeat) -> None:
        self._validate_heartbeat(heartbeat)
        async with self._condition:
            key = heartbeat.role, heartbeat.worker_id
            previous = self._quarantined.get(key)
            if previous and previous[0] != heartbeat.worker_epoch:
                self._quarantined.pop(key)
            self._workers[heartbeat.worker_id] = _WorkerState(
                heartbeat=heartbeat,
                updated_at=self._clock(),
            )
            self._condition.notify_all()

    async def claim_attempt(
        self, *, attempt_id: str, owner_id: str, expires_at: float
    ) -> None:
        if not attempt_id or not owner_id:
            raise CoordinatorRejected("INVALID_ATTEMPT_OWNER")
        async with self._condition:
            now = self._wall_clock()
            if expires_at <= now:
                raise CoordinatorRejected("INVALID_ATTEMPT_OWNER")
            # Attempt 围栏独立于 GPU lease：连接结束后仍要阻止 token 重放，
            # 因此只按 token 有效窗口惰性清理，不能随 assignment.release 删除。
            expired = [
                key for key, (_, until) in self._attempt_owners.items() if until <= now
            ]
            for key in expired:
                del self._attempt_owners[key]
            current = self._attempt_owners.get(attempt_id)
            if current is not None and current[0] != owner_id:
                raise CoordinatorRejected("ATTEMPT_ALREADY_OWNED")
            self._attempt_owners[attempt_id] = (owner_id, expires_at)

    async def waiting_started(self, waiter_id: str) -> None:
        async with self._condition:
            self._waiting.add(waiter_id)

    async def waiting_finished(self, waiter_id: str) -> None:
        async with self._condition:
            self._waiting.discard(waiter_id)

    async def capacity_snapshot(
        self, *, model_revision: str | None = None, vae_fingerprint: str | None = None
    ) -> dict[str, Any]:
        _capacity_scope_requested(model_revision, vae_fingerprint)
        async with self._condition:
            now = self._clock()
            roles = {
                role: {
                    "waiting_sessions": len(self._waiting),
                    "active_sessions": 0,
                    "queued_sessions": 0,
                    "free_slots": 0,
                    "draining_workers": 0,
                }
                for role in ("denoiser", "vae")
            }
            available = {
                (worker.heartbeat.role, worker.heartbeat.worker_id)
                for role in roles
                for worker in self._active_workers_locked(
                    role=role,
                    now=now,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                    excluded_workers=frozenset(),
                )
            }
            for state in self._workers.values():
                if state.updated_at + self.worker_ttl_s <= now:
                    continue
                heartbeat = state.heartbeat
                if not _matches_worker_request(
                    heartbeat,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                    aliases=self.routing_revision_aliases,
                ):
                    continue
                role = roles[heartbeat.role]
                known = sum(
                    assignment.expires_at > now
                    and any(
                        slot.worker_id == heartbeat.worker_id
                        for slot in assignment.workers
                    )
                    for assignment in self._assignments_by_token.values()
                )
                occupied = max(known, heartbeat.active_sessions)
                role["active_sessions"] += occupied
                role["queued_sessions"] += heartbeat.queue_depth
                if heartbeat.lifecycle == "draining":
                    role["draining_workers"] += 1
                elif (heartbeat.role, heartbeat.worker_id) in available:
                    role["free_slots"] += max(0, heartbeat.capacity - occupied)
            return {"observed_at": self._wall_clock(), "roles": roles}

    async def acquire(
        self,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        model_revision: str,
        vae_fingerprint: str,
        player_ttl_s: float | None = None,
        excluded_workers: frozenset[tuple[WorkerRole, str]] = frozenset(),
    ) -> SessionAssignment:
        if not all(
            (user_id, session_id, generation_id, model_revision, vae_fingerprint)
        ):
            raise CoordinatorRejected("INVALID_SESSION_IDENTITY")
        async with self._condition:
            now = self._clock()
            identity = f"{user_id}:{session_id}:{generation_id}"
            self._expire_locked(now)
            if user_id in self._assignments_by_user:
                raise CoordinatorRejected("USER_SESSION_LIMIT")

            denoisers = self._free_slots_locked(
                self._active_workers_locked(
                    role="denoiser",
                    now=now,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                    excluded_workers=excluded_workers,
                ),
                identity=identity,
            )
            vaes = self._free_slots_locked(
                self._active_workers_locked(
                    role="vae",
                    now=now,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                    excluded_workers=excluded_workers,
                ),
                identity=identity,
            )
            if not denoisers or (vae_fingerprint != "local" and not vaes):
                raise CoordinatorRejected("CAPACITY_EXHAUSTED", retry_after_s=0.1)

            denoiser = denoisers[0]
            vae = (
                None
                if vae_fingerprint == "local"
                else min(
                    vaes,
                    key=lambda slot: (
                        slot.az != denoiser.az,
                        slot.active_sessions / slot.capacity,
                        slot.queue_depth,
                        slot.service_time_ms,
                        hashlib.sha256(
                            f"{identity}:vae:{slot.worker_id}".encode()
                        ).digest(),
                        slot.worker_id,
                        slot.slot_index,
                    ),
                )
            )
            token = uuid4().hex
            assignment = SessionAssignment(
                user_id=user_id,
                session_id=session_id,
                generation_id=generation_id,
                token=token,
                expires_at=now + self.ttl_s,
                admitted_at=self._wall_clock() if player_ttl_s is not None else None,
                player_expires_at=self._wall_clock() + player_ttl_s
                if player_ttl_s is not None
                else None,
                denoiser=denoiser,
                vae=vae,
            )
            self._assignments_by_user[user_id] = assignment
            self._assignments_by_token[token] = assignment
            for slot in filter(None, (denoiser, vae)):
                self._slot_tokens[(slot.role, slot.worker_id, slot.slot_index)] = token
            return assignment

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment:
        async with self._condition:
            now = self._clock()
            self._expire_locked(now)
            current = self._assignments_by_token.get(assignment.token)
            if current is None or current != assignment:
                raise CoordinatorRejected("LEASE_LOST")
            if (
                current.player_expires_at is not None
                and current.player_expires_at <= self._wall_clock()
            ):
                raise CoordinatorRejected("PLAYER_EXPIRED")
            for slot in current.workers:
                state = self._workers.get(slot.worker_id)
                if (
                    state is None
                    or state.updated_at + self.worker_ttl_s <= now
                    or state.heartbeat.worker_epoch != slot.worker_epoch
                    or state.heartbeat.lifecycle == "failed"
                    or (
                        state.heartbeat.lifecycle == "draining"
                        and state.heartbeat.drain_deadline is not None
                        and state.heartbeat.drain_deadline <= self._wall_clock()
                    )
                ):
                    raise CoordinatorRejected("WORKER_LOST")
            renewed = replace(current, expires_at=now + self.ttl_s)
            self._assignments_by_token[current.token] = renewed
            self._assignments_by_user[current.user_id] = renewed
            return renewed

    async def release(self, assignment: SessionAssignment) -> None:
        async with self._condition:
            current = self._assignments_by_token.get(assignment.token)
            if assignment.player_expires_at is not None or (
                current is not None and current.player_expires_at is not None
            ):
                raise CoordinatorRejected("VERIFIED_RELEASE_REQUIRED")
            self._release_locked(assignment)
            self._condition.notify_all()

    async def quarantine(self, assignment: SessionAssignment) -> None:
        async with self._condition:
            current = self._assignments_by_token.get(assignment.token)
            if current is None:
                return
            if _release_identity(current) != _release_identity(assignment):
                raise CoordinatorRejected("LEASE_LOST")
            for slot in current.workers:
                if (slot.role, slot.worker_id) not in self._quarantined and len(
                    self._quarantined
                ) >= 2048:
                    raise CoordinatorRejected("COORDINATOR_STORE_BUSY")
                self._quarantined[(slot.role, slot.worker_id)] = (
                    slot.worker_epoch,
                    assignment.token,
                )
            self._condition.notify_all()

    async def verified_release_status(self, assignment):
        async with self._condition:
            current = self._assignments_by_token.get(assignment.token)
            if current is not None and _release_identity(current) == _release_identity(
                assignment
            ):
                return current, False
            receipt = self._verified_releases.get(assignment.token)
            if receipt and receipt["identity"] == _release_identity(assignment):
                return None, True
            raise CoordinatorRejected("LEASE_LOST")

    async def player_cleanup(self, assignment, receipt=None):
        async with self._condition:
            current = self._assignments_by_token.get(assignment.token)
            if current is None or _release_identity(current) != _release_identity(
                assignment
            ):
                raise CoordinatorRejected("LEASE_LOST")
            return _record_player_cleanup(self._player_closures, current, receipt)

    async def complete_verified_release(self, assignment):
        async with self._condition:
            current = self._assignments_by_token.get(assignment.token)
            if current is None:
                receipt = self._verified_releases.get(assignment.token)
                if receipt and receipt["identity"] == _release_identity(assignment):
                    return
                raise CoordinatorRejected("LEASE_LOST")
            if _release_identity(current) != _release_identity(assignment):
                raise CoordinatorRejected("LEASE_LOST")
            self._release_locked(current)
            _retain_release_receipt(self._verified_releases, current, self._clock())
            cleanup = self._player_closures.pop(current.token, None)
            if cleanup is not None:
                self._verified_releases[current.token]["cleanup"] = cleanup
            for slot in current.workers:
                key = slot.role, slot.worker_id
                if self._quarantined.get(key) == (slot.worker_epoch, current.token):
                    self._quarantined.pop(key)
            self._condition.notify_all()

    async def clear_quarantine(self, assignment: SessionAssignment) -> None:
        async with self._condition:
            for slot in assignment.workers:
                key = (slot.role, slot.worker_id)
                if self._quarantined.get(key) == (slot.worker_epoch, assignment.token):
                    self._quarantined.pop(key)
            self._condition.notify_all()

    async def wait_for_change(self, timeout_s: float) -> None:
        async with self._condition:
            try:
                await asyncio.wait_for(self._condition.wait(), timeout=timeout_s)
            except TimeoutError:
                pass


class DynamoDBCoordinatorStore:
    """Multi-replica Coordinator state backed by DynamoDB transactions.

    The table uses ``pk`` and ``sk`` string keys plus an ``allocation-index``
    GSI whose partition key is ``allocation_key`` and sort key is
    ``allocation_sort``. Worker heartbeats maintain allocatable slot records;
    admission atomically fences the user, Session, Denoiser slot, and VAE slot.
    """

    def __init__(
        self,
        table_name: str,
        *,
        ttl_s: float,
        worker_ttl_s: float,
        routing_revision_aliases: Mapping[str, object] | None = None,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        client: Any | None = None,
        candidate_limit: int = 24,
        capacity_limits: Mapping[WorkerRole, int] | None = None,
        wall_clock: Callable[[], float] = time.time,
        lease_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not table_name:
            raise ValueError("table_name is required")
        if ttl_s <= 0 or worker_ttl_s <= 0:
            raise ValueError("lease TTLs must be positive")
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        if capacity_limits and any(limit < 1 for limit in capacity_limits.values()):
            raise ValueError("worker capacity limits must be positive")
        self.table_name = table_name
        self.ttl_s = ttl_s
        self.worker_ttl_s = worker_ttl_s
        self.routing_revision_aliases = _normalize_routing_revision_aliases(
            routing_revision_aliases
        )
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        self.candidate_limit = candidate_limit
        self.capacity_limits = dict(capacity_limits or {})
        self._client = client
        self._wall_clock = wall_clock
        self._lease_clock = lease_clock

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "boto3 is required for DynamoDB Coordinator state"
                ) from exc
            self._client = boto3.client(
                "dynamodb",
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
            )
        return self._client

    @staticmethod
    def _slot_pk(role: WorkerRole, worker_id: str, slot_index: int) -> str:
        return f"SLOT#{role}#{worker_id}#{slot_index:04d}"

    async def claim_attempt(
        self, *, attempt_id: str, owner_id: str, expires_at: float
    ) -> None:
        await asyncio.to_thread(
            self._claim_attempt_sync,
            attempt_id=attempt_id,
            owner_id=owner_id,
            expires_at=expires_at,
        )

    def _claim_attempt_sync(
        self, *, attempt_id: str, owner_id: str, expires_at: float
    ) -> None:
        if not attempt_id or not owner_id:
            raise CoordinatorRejected("INVALID_ATTEMPT_OWNER")
        now_epoch = int(self._wall_clock())
        expires_epoch = int(expires_at)
        if expires_epoch <= now_epoch:
            raise CoordinatorRejected("INVALID_ATTEMPT_OWNER")
        client = self._get_client()
        try:
            client.put_item(
                TableName=self.table_name,
                Item={
                    "pk": {"S": f"ATTEMPT#{attempt_id}"},
                    "sk": {"S": "OWNER"},
                    "item_type": {"S": "attempt_owner"},
                    "owner_id": {"S": owner_id},
                    "owner_expires_at": {"N": str(expires_epoch)},
                    "ttl": {"N": str(expires_epoch + 86400)},
                },
                # 同一个 owner 重试必须幂等；不同连接只有在旧 token 窗口
                # 确实过期后才能覆盖，DynamoDB TTL 是否及时物理删除不影响正确性。
                ConditionExpression=(
                    "attribute_not_exists(owner_id) OR owner_expires_at <= :now OR "
                    "owner_id = :owner"
                ),
                ExpressionAttributeValues={
                    ":now": {"N": str(now_epoch)},
                    ":owner": {"S": owner_id},
                },
            )
        except client.exceptions.ConditionalCheckFailedException as exc:
            raise CoordinatorRejected("ATTEMPT_ALREADY_OWNED") from exc

    @staticmethod
    def _read_s(item: dict, key: str) -> str:
        return item[key]["S"]

    @staticmethod
    def _read_n(item: dict, key: str) -> int:
        return int(item[key]["N"])

    @staticmethod
    def _read_optional_s(item: dict, key: str, default: str = "") -> str:
        return item.get(key, {}).get("S", default)

    @staticmethod
    def _read_optional_n(item: dict, key: str, default: int | float = 0) -> int | float:
        value = item.get(key, {}).get("N")
        if value is None:
            return default
        return float(value) if "." in value else int(value)

    @staticmethod
    def _is_active_item(item: dict | None, now_epoch: int) -> bool:
        if not item:
            return False
        expires = item.get("lease_expires_at", {}).get("N")
        return expires is not None and int(expires) > now_epoch

    def _allocation_key(
        self,
        role: WorkerRole,
        *,
        model_revision: str,
        vae_fingerprint: str,
    ) -> str:
        if role == "denoiser":
            return f"DENOISER#{model_revision}"
        return f"VAE#{vae_fingerprint}"

    def _denoiser_query_revisions(self, model_revision: str) -> list[str]:
        revisions = [model_revision]
        revisions.extend(
            revision
            for revision in sorted(
                self.routing_revision_aliases.get(model_revision, ())
            )
            if revision != model_revision
        )
        return revisions

    def _slot_from_item(self, item: dict) -> WorkerSlot:
        return WorkerSlot(
            worker_id=self._read_s(item, "worker_id"),
            role=self._read_s(item, "role"),
            endpoint=self._read_s(item, "endpoint"),
            az=self._read_s(item, "az"),
            slot_index=self._read_n(item, "slot_index"),
            model_revision=self._read_s(item, "model_revision"),
            vae_fingerprint=self._read_s(item, "vae_fingerprint"),
            routing_revision=self._read_optional_s(
                item,
                "routing_revision",
                self._read_s(item, "model_revision"),
            ),
            worker_epoch=self._read_optional_s(item, "worker_epoch"),
            lifecycle=self._read_optional_s(item, "lifecycle", "ready"),
            active_sessions=int(self._read_optional_n(item, "active_sessions")),
            runnable_sessions=int(self._read_optional_n(item, "runnable_sessions")),
            blocked_sessions=int(self._read_optional_n(item, "blocked_sessions")),
            queue_depth=int(self._read_optional_n(item, "queue_depth")),
            service_time_ms=float(self._read_optional_n(item, "service_time_ms")),
            reservation_endpoint=self._read_optional_s(item, "reservation_endpoint"),
            drain_deadline=(
                float(item["drain_deadline"]["N"]) if "drain_deadline" in item else None
            ),
            capacity=int(self._read_optional_n(item, "capacity", 1)),
        )

    @staticmethod
    def _candidate_pairs(
        denoisers: list[WorkerSlot],
        vaes: list[WorkerSlot],
        *,
        identity: str,
    ) -> list[tuple[WorkerSlot, WorkerSlot]]:
        if not denoisers or not vaes:
            return []

        def available_slots_by_worker(
            slots: list[WorkerSlot],
        ) -> dict[str, int]:
            available: dict[str, int] = {}
            for slot in slots:
                available[slot.worker_id] = available.get(slot.worker_id, 0) + 1
            return available

        denoiser_free = available_slots_by_worker(denoisers)
        vae_free = available_slots_by_worker(vaes)

        def order(
            slot: WorkerSlot,
            salt: str,
            available: dict[str, int],
        ) -> tuple:
            inferred_active = max(
                0,
                slot.capacity - available.get(slot.worker_id, 0),
            )
            effective_active = max(slot.active_sessions, inferred_active)
            spread = hashlib.sha256(f"{salt}:{slot.worker_id}".encode()).digest()
            return (
                effective_active / slot.capacity,
                slot.queue_depth,
                slot.service_time_ms,
                slot.slot_index,
                spread,
                slot.worker_id,
            )

        ordered_denoisers = sorted(
            denoisers,
            key=lambda slot: order(slot, "denoiser", denoiser_free),
        )
        available_vaes = list(vaes)
        pairs: list[tuple[WorkerSlot, WorkerSlot]] = []
        for denoiser in ordered_denoisers:
            if not available_vaes:
                break
            vae_index = min(
                range(len(available_vaes)),
                key=lambda index: (
                    available_vaes[index].az != denoiser.az,
                    *order(available_vaes[index], "vae", vae_free),
                ),
            )
            pairs.append((denoiser, available_vaes.pop(vae_index)))
        return pairs

    async def heartbeat(self, heartbeat: WorkerHeartbeat) -> None:
        InMemoryCoordinatorStore._validate_heartbeat(heartbeat)
        await asyncio.to_thread(self._heartbeat_sync, heartbeat)

    def _heartbeat_sync(self, heartbeat: WorkerHeartbeat) -> None:
        capacity_limit = self.capacity_limits.get(heartbeat.role)
        if capacity_limit is not None and heartbeat.capacity > capacity_limit:
            heartbeat = replace(heartbeat, capacity=capacity_limit)
        client = self._get_client()
        now_epoch = int(self._wall_clock())
        heartbeat_expires = now_epoch + max(1, int(self.worker_ttl_s))
        client.put_item(
            TableName=self.table_name,
            Item={
                "pk": {"S": f"WORKER#{heartbeat.worker_id}"},
                "sk": {"S": "HEARTBEAT"},
                "item_type": {"S": "worker"},
                "role": {"S": heartbeat.role},
                "endpoint": {"S": heartbeat.endpoint},
                "az": {"S": heartbeat.az},
                "capacity": {"N": str(heartbeat.capacity)},
                "model_revision": {"S": heartbeat.model_revision},
                "vae_fingerprint": {"S": heartbeat.vae_fingerprint},
                "routing_revision": {"S": _routing_revision(heartbeat)},
                "worker_epoch": {"S": heartbeat.worker_epoch},
                "lifecycle": {"S": heartbeat.lifecycle},
                "active_sessions": {"N": str(heartbeat.active_sessions)},
                "runnable_sessions": {"N": str(heartbeat.runnable_sessions)},
                "blocked_sessions": {"N": str(heartbeat.blocked_sessions)},
                "queue_depth": {"N": str(heartbeat.queue_depth)},
                "service_time_ms": {"N": str(heartbeat.service_time_ms)},
                "oldest_consumed_age_s": {"N": str(heartbeat.oldest_consumed_age_s)},
                "stale_consumed_reservations": {
                    "N": str(heartbeat.stale_consumed_reservations)
                },
                "last_progress_age_s": {"N": str(heartbeat.last_progress_age_s)},
                "reservation_endpoint": {"S": heartbeat.reservation_endpoint},
                "heartbeat_expires_at": {"N": str(heartbeat_expires)},
                "allocation_key": {"S": f"CAPACITY#{heartbeat.role}"},
                "allocation_sort": {"S": f"worker#{heartbeat.worker_id}"},
                "ttl": {"N": str(heartbeat_expires + 86400)},
                **(
                    {"drain_deadline": {"N": str(heartbeat.drain_deadline)}}
                    if heartbeat.drain_deadline is not None
                    else {}
                ),
            },
        )
        allocation_key = self._allocation_key(
            heartbeat.role,
            model_revision=_routing_revision(heartbeat),
            vae_fingerprint=heartbeat.vae_fingerprint,
        )
        for slot_index in range(heartbeat.capacity):
            update = {
                "TableName": self.table_name,
                "Key": {
                    "pk": {
                        "S": self._slot_pk(
                            heartbeat.role, heartbeat.worker_id, slot_index
                        )
                    },
                    "sk": {"S": "LEASE"},
                },
                "UpdateExpression": (
                    "SET item_type = :item_type, #role = :role, "
                    "worker_id = :worker_id, endpoint = :endpoint, az = :az, "
                    "slot_index = :slot_index, model_revision = :model_revision, "
                    "routing_revision = :routing_revision, "
                    "vae_fingerprint = :vae_fingerprint, "
                    "worker_epoch = :worker_epoch, lifecycle = :lifecycle, "
                    "#capacity = :capacity, active_sessions = :active_sessions, "
                    "runnable_sessions = :runnable_sessions, "
                    "blocked_sessions = :blocked_sessions, "
                    "queue_depth = :queue_depth, service_time_ms = :service_time_ms, "
                    "reservation_endpoint = :reservation_endpoint, "
                    "heartbeat_expires_at = :heartbeat_expires, "
                    "allocation_key = :allocation_key, "
                    "allocation_sort = :allocation_sort, #ttl = :ttl"
                ),
                "ExpressionAttributeNames": {
                    "#capacity": "capacity",
                    "#role": "role",
                    "#ttl": "ttl",
                },
                "ExpressionAttributeValues": {
                    ":item_type": {"S": "worker_slot"},
                    ":role": {"S": heartbeat.role},
                    ":worker_id": {"S": heartbeat.worker_id},
                    ":endpoint": {"S": heartbeat.endpoint},
                    ":az": {"S": heartbeat.az},
                    ":slot_index": {"N": str(slot_index)},
                    ":model_revision": {"S": heartbeat.model_revision},
                    ":routing_revision": {"S": _routing_revision(heartbeat)},
                    ":vae_fingerprint": {"S": heartbeat.vae_fingerprint},
                    ":worker_epoch": {"S": heartbeat.worker_epoch},
                    ":lifecycle": {"S": heartbeat.lifecycle},
                    ":capacity": {"N": str(heartbeat.capacity)},
                    ":active_sessions": {"N": str(heartbeat.active_sessions)},
                    ":runnable_sessions": {"N": str(heartbeat.runnable_sessions)},
                    ":blocked_sessions": {"N": str(heartbeat.blocked_sessions)},
                    ":queue_depth": {"N": str(heartbeat.queue_depth)},
                    ":service_time_ms": {"N": str(heartbeat.service_time_ms)},
                    ":reservation_endpoint": {"S": heartbeat.reservation_endpoint},
                    ":heartbeat_expires": {"N": str(heartbeat_expires)},
                    ":allocation_key": {"S": allocation_key},
                    ":allocation_sort": {
                        "S": f"{heartbeat.az}#{heartbeat.worker_id}#{slot_index:04d}"
                    },
                    ":ttl": {"N": str(heartbeat_expires + 86400)},
                },
            }
            for attempt in range(3):
                try:
                    client.update_item(**update)
                    break
                except client.exceptions.TransactionConflictException:
                    if attempt == 2:
                        raise
                    time.sleep(0.01 * (2**attempt))

    async def waiting_started(self, waiter_id: str) -> None:
        await asyncio.to_thread(self._waiting_started_sync, waiter_id)

    def _waiting_started_sync(self, waiter_id: str) -> None:
        if not waiter_id:
            raise ValueError("waiter_id is required")
        client = self._get_client()
        now_epoch = int(self._wall_clock())
        expires_epoch = now_epoch + max(30, int(self.worker_ttl_s * 2))
        for role in ("denoiser", "vae"):
            client.put_item(
                TableName=self.table_name,
                Item={
                    "pk": {"S": f"CAPACITY_DEMAND#{role}#{waiter_id}"},
                    "sk": {"S": "WAIT"},
                    "item_type": {"S": "capacity_demand"},
                    "role": {"S": role},
                    "allocation_key": {"S": f"CAPACITY#{role}"},
                    "allocation_sort": {"S": f"demand#{waiter_id}"},
                    "demand_expires_at": {"N": str(expires_epoch)},
                    "ttl": {"N": str(expires_epoch + 3600)},
                },
            )

    async def waiting_finished(self, waiter_id: str) -> None:
        await asyncio.to_thread(self._waiting_finished_sync, waiter_id)

    def _waiting_finished_sync(self, waiter_id: str) -> None:
        client = self._get_client()
        for role in ("denoiser", "vae"):
            client.delete_item(
                TableName=self.table_name,
                Key={
                    "pk": {"S": f"CAPACITY_DEMAND#{role}#{waiter_id}"},
                    "sk": {"S": "WAIT"},
                },
            )

    async def capacity_snapshot(
        self, *, model_revision: str | None = None, vae_fingerprint: str | None = None
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._capacity_snapshot_sync,
            model_revision=model_revision,
            vae_fingerprint=vae_fingerprint,
        )

    def _capacity_snapshot_sync(
        self, *, model_revision: str | None = None, vae_fingerprint: str | None = None
    ) -> dict[str, Any]:
        scoped = _capacity_scope_requested(model_revision, vae_fingerprint)
        client = self._get_client()
        now_epoch = int(self._wall_clock())
        roles: dict[str, dict[str, int]] = {}
        for role in ("denoiser", "vae"):
            values = {
                "waiting_sessions": 0,
                "active_sessions": 0,
                "queued_sessions": 0,
                "free_slots": 0,
                "draining_workers": 0,
            }
            available = None
            if scoped:
                assert model_revision is not None and vae_fingerprint is not None
                # This backend cannot admit local decode; preserve that boundary.
                if vae_fingerprint == "local":
                    slots = []
                else:
                    slots = (
                        self._query_denoiser_slots_sync(
                            model_revision=model_revision,
                            vae_fingerprint=vae_fingerprint,
                            now_epoch=now_epoch,
                            limit_candidates=False,
                        )
                        if role == "denoiser"
                        else self._query_slots_sync(
                            role,
                            model_revision=model_revision,
                            vae_fingerprint=vae_fingerprint,
                            now_epoch=now_epoch,
                            limit_candidates=False,
                        )
                    )
                available = {}
                for slot in slots:
                    available[slot.worker_id] = available.get(slot.worker_id, 0) + 1
            query = {
                "TableName": self.table_name,
                "IndexName": "allocation-index",
                "KeyConditionExpression": "allocation_key = :allocation",
                "ExpressionAttributeValues": {":allocation": {"S": f"CAPACITY#{role}"}},
            }
            while True:
                response = client.query(**query)
                for item in response.get("Items", []):
                    item_type = self._read_optional_s(item, "item_type")
                    if item_type == "capacity_demand":
                        if (
                            int(self._read_optional_n(item, "demand_expires_at"))
                            > now_epoch
                        ):
                            values["waiting_sessions"] += 1
                        continue
                    if (
                        item_type != "worker"
                        or int(self._read_optional_n(item, "heartbeat_expires_at"))
                        <= now_epoch
                    ):
                        continue
                    if scoped and vae_fingerprint == "local":
                        continue
                    if scoped:
                        heartbeat = WorkerHeartbeat(
                            worker_id=self._read_s(item, "pk").split("#", 1)[1],
                            role=role,
                            endpoint=self._read_s(item, "endpoint"),
                            az=self._read_s(item, "az"),
                            capacity=int(self._read_optional_n(item, "capacity", 1)),
                            model_revision=self._read_s(item, "model_revision"),
                            vae_fingerprint=self._read_s(item, "vae_fingerprint"),
                            routing_revision=self._read_optional_s(
                                item,
                                "routing_revision",
                                self._read_s(item, "model_revision"),
                            ),
                        )
                        if not _matches_worker_request(
                            heartbeat,
                            model_revision=model_revision,
                            vae_fingerprint=vae_fingerprint,
                            aliases=self.routing_revision_aliases,
                        ):
                            continue
                    active = int(self._read_optional_n(item, "active_sessions"))
                    if (
                        scoped
                        and self._read_optional_s(item, "lifecycle", "ready") == "ready"
                    ):
                        # A reserved slot is occupied before the next worker heartbeat.
                        active = max(
                            active,
                            heartbeat.capacity - available.get(heartbeat.worker_id, 0),
                        )
                    values["active_sessions"] += active
                    values["queued_sessions"] += int(
                        self._read_optional_n(item, "queue_depth")
                    )
                    lifecycle = self._read_optional_s(item, "lifecycle", "ready")
                    if lifecycle == "draining":
                        values["draining_workers"] += 1
                    elif lifecycle == "ready":
                        values["free_slots"] += max(
                            0,
                            int(self._read_optional_n(item, "capacity", 1)) - active,
                        )
                last_evaluated_key = response.get("LastEvaluatedKey")
                if not last_evaluated_key:
                    break
                query["ExclusiveStartKey"] = last_evaluated_key
            roles[role] = values
        return {"observed_at": now_epoch, "roles": roles}

    def _query_slots_sync(
        self,
        role: WorkerRole,
        *,
        model_revision: str,
        vae_fingerprint: str,
        now_epoch: int,
        limit_candidates: bool = True,
    ) -> list[WorkerSlot]:
        query = {
            "TableName": self.table_name,
            "IndexName": "allocation-index",
            "KeyConditionExpression": "allocation_key = :allocation",
            "FilterExpression": (
                "heartbeat_expires_at > :now AND lifecycle = :ready AND "
                "(attribute_not_exists(lease_token) OR lease_expires_at <= :now)"
            ),
            "ExpressionAttributeValues": {
                ":allocation": {
                    "S": self._allocation_key(
                        role,
                        model_revision=model_revision,
                        vae_fingerprint=vae_fingerprint,
                    )
                },
                ":now": {"N": str(now_epoch)},
                ":ready": {"S": "ready"},
            },
            "Limit": self.candidate_limit,
        }
        slots: list[WorkerSlot] = []
        while not limit_candidates or len(slots) < self.candidate_limit:
            response = self._get_client().query(**query)
            slots.extend(
                self._slot_from_item(item) for item in response.get("Items", [])
            )
            last_evaluated_key = response.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break
            query["ExclusiveStartKey"] = last_evaluated_key
        return slots[: self.candidate_limit] if limit_candidates else slots

    def _query_denoiser_slots_sync(
        self,
        *,
        model_revision: str,
        vae_fingerprint: str,
        now_epoch: int,
        limit_candidates: bool = True,
    ) -> list[WorkerSlot]:
        slots: list[WorkerSlot] = []
        seen: set[tuple[str, int]] = set()
        for revision in self._denoiser_query_revisions(model_revision):
            for slot in self._query_slots_sync(
                "denoiser",
                model_revision=revision,
                vae_fingerprint=vae_fingerprint,
                now_epoch=now_epoch,
                limit_candidates=limit_candidates,
            ):
                key = (slot.worker_id, slot.slot_index)
                if key in seen:
                    continue
                seen.add(key)
                slots.append(slot)
                if limit_candidates and len(slots) >= self.candidate_limit:
                    return slots
        return slots

    async def acquire(
        self,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        model_revision: str,
        vae_fingerprint: str,
        excluded_workers: frozenset[tuple[WorkerRole, str]] = frozenset(),
    ) -> SessionAssignment:
        return await asyncio.to_thread(
            self._acquire_sync,
            user_id=user_id,
            session_id=session_id,
            generation_id=generation_id,
            model_revision=model_revision,
            vae_fingerprint=vae_fingerprint,
            excluded_workers=excluded_workers,
        )

    def _acquire_sync(
        self,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        model_revision: str,
        vae_fingerprint: str,
        excluded_workers: frozenset[tuple[WorkerRole, str]] = frozenset(),
    ) -> SessionAssignment:
        if vae_fingerprint == "local":
            raise CoordinatorRejected("LOCAL_DECODE_REQUIRES_REDIS_OR_MEMORY")
        if not all(
            (user_id, session_id, generation_id, model_revision, vae_fingerprint)
        ):
            raise CoordinatorRejected("INVALID_SESSION_IDENTITY")
        client = self._get_client()
        for query_round in range(4):
            now_epoch = int(self._wall_clock())
            expires_epoch = now_epoch + max(1, int(self.ttl_s))
            denoisers = self._query_denoiser_slots_sync(
                model_revision=model_revision,
                vae_fingerprint=vae_fingerprint,
                now_epoch=now_epoch,
            )
            vaes = self._query_slots_sync(
                "vae",
                model_revision=model_revision,
                vae_fingerprint=vae_fingerprint,
                now_epoch=now_epoch,
            )
            denoisers = [
                slot
                for slot in denoisers
                if (slot.role, slot.worker_id) not in excluded_workers
            ]
            vaes = [
                slot
                for slot in vaes
                if (slot.role, slot.worker_id) not in excluded_workers
            ]
            pairs = self._candidate_pairs(
                denoisers,
                vaes,
                identity=(f"{user_id}:{session_id}:{generation_id}:{query_round}"),
            )
            if not pairs:
                break
            for denoiser, vae in pairs:
                assignment = self._try_acquire_pair_sync(
                    client,
                    user_id=user_id,
                    session_id=session_id,
                    generation_id=generation_id,
                    denoiser=denoiser,
                    vae=vae,
                    now_epoch=now_epoch,
                    expires_epoch=expires_epoch,
                )
                if assignment is not None:
                    return assignment
            if query_round < 3:
                time.sleep(0.005 * (2**query_round))
        raise CoordinatorRejected("CAPACITY_EXHAUSTED", retry_after_s=0.25)

    def _try_acquire_pair_sync(
        self,
        client: Any,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        denoiser: WorkerSlot,
        vae: WorkerSlot,
        now_epoch: int,
        expires_epoch: int,
    ) -> SessionAssignment | None:
        token = uuid4().hex
        identity = {
            "user_id": {"S": user_id},
            "session_id": {"S": session_id},
            "generation_id": {"S": generation_id},
            "lease_token": {"S": token},
            "lease_expires_at": {"N": str(expires_epoch)},
            "ttl": {"N": str(expires_epoch + 86400)},
        }
        values = {
            ":token": {"S": token},
            ":user": {"S": user_id},
            ":session": {"S": session_id},
            ":generation": {"S": generation_id},
            ":now": {"N": str(now_epoch)},
            ":expires": {"N": str(expires_epoch)},
            ":ttl": {"N": str(expires_epoch + 86400)},
        }
        slot_updates = []
        for slot in filter(None, (denoiser, vae)):
            slot_values = {
                **values,
                ":worker_epoch": {"S": slot.worker_epoch},
                ":ready": {"S": "ready"},
            }
            slot_updates.append(
                {
                    "Update": {
                        "TableName": self.table_name,
                        "Key": {
                            "pk": {
                                "S": self._slot_pk(
                                    slot.role, slot.worker_id, slot.slot_index
                                )
                            },
                            "sk": {"S": "LEASE"},
                        },
                        "UpdateExpression": (
                            "SET lease_token = :token, user_id = :user, "
                            "session_id = :session, generation_id = :generation, "
                            "lease_expires_at = :expires, #ttl = :ttl"
                        ),
                        "ConditionExpression": (
                            "heartbeat_expires_at > :now AND "
                            "worker_epoch = :worker_epoch AND lifecycle = :ready AND "
                            "(attribute_not_exists(lease_token) OR "
                            "lease_expires_at <= :now)"
                        ),
                        "ExpressionAttributeNames": {"#ttl": "ttl"},
                        "ExpressionAttributeValues": slot_values,
                    }
                }
            )
        session_item = {
            **identity,
            "pk": {"S": f"SESSION#{session_id}"},
            "sk": {"S": "ASSIGNMENT"},
            "item_type": {"S": "session_assignment"},
            "denoiser_worker_id": {"S": denoiser.worker_id},
            "denoiser_slot": {"N": str(denoiser.slot_index)},
            "denoiser_endpoint": {"S": denoiser.endpoint},
            "denoiser_worker_epoch": {"S": denoiser.worker_epoch},
            "vae_worker_id": {"S": vae.worker_id},
            "vae_slot": {"N": str(vae.slot_index)},
            "vae_endpoint": {"S": vae.endpoint},
            "vae_worker_epoch": {"S": vae.worker_epoch},
        }
        try:
            client.transact_write_items(
                TransactItems=[
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": {
                                **identity,
                                "pk": {"S": f"USER#{user_id}"},
                                "sk": {"S": "LEASE"},
                                "item_type": {"S": "user_lease"},
                            },
                            "ConditionExpression": (
                                "attribute_not_exists(lease_token) OR "
                                "lease_expires_at <= :now"
                            ),
                            "ExpressionAttributeValues": {
                                ":now": {"N": str(now_epoch)}
                            },
                        }
                    },
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": session_item,
                            "ConditionExpression": (
                                "attribute_not_exists(lease_token) OR "
                                "lease_expires_at <= :now"
                            ),
                            "ExpressionAttributeValues": {
                                ":now": {"N": str(now_epoch)}
                            },
                        }
                    },
                    *slot_updates,
                ]
            )
        except client.exceptions.TransactionCanceledException as exc:
            getter = getattr(client, "get_item", None)
            if getter is not None:
                current_user = getter(
                    TableName=self.table_name,
                    Key={
                        "pk": {"S": f"USER#{user_id}"},
                        "sk": {"S": "LEASE"},
                    },
                    ConsistentRead=True,
                ).get("Item")
                if self._is_active_item(current_user, now_epoch):
                    raise CoordinatorRejected("USER_SESSION_LIMIT") from exc
            return None
        return SessionAssignment(
            user_id=user_id,
            session_id=session_id,
            generation_id=generation_id,
            token=token,
            expires_at=self._lease_clock() + self.ttl_s,
            denoiser=denoiser,
            vae=vae,
        )

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment:
        return await asyncio.to_thread(self._renew_sync, assignment)

    def _renew_cancellation_reason(
        self,
        client: Any,
        assignment: SessionAssignment,
        keys: list[tuple[str, str]],
        *,
        now_epoch: int,
    ) -> str | None:
        getter = getattr(client, "get_item", None)
        if getter is None:
            return "LEASE_LOST"
        for slot in assignment.workers:
            item = getter(
                TableName=self.table_name,
                Key={
                    "pk": {"S": f"WORKER#{slot.worker_id}"},
                    "sk": {"S": "HEARTBEAT"},
                },
                ConsistentRead=True,
            ).get("Item")
            if (
                not item
                or item.get("worker_epoch", {}).get("S") != slot.worker_epoch
                or int(item.get("heartbeat_expires_at", {}).get("N", "0")) <= now_epoch
                or item.get("lifecycle", {}).get("S", "ready") == "failed"
                or (
                    item.get("lifecycle", {}).get("S") == "draining"
                    and float(item.get("drain_deadline", {}).get("N", "inf"))
                    <= now_epoch
                )
            ):
                return "WORKER_LOST"
        for pk, sk in keys:
            item = getter(
                TableName=self.table_name,
                Key={"pk": {"S": pk}, "sk": {"S": sk}},
                ConsistentRead=True,
            ).get("Item")
            if item.get("lease_token", {}).get("S") != assignment.token:
                return "LEASE_LOST"
        return None

    def _renew_sync(self, assignment: SessionAssignment) -> SessionAssignment:
        client = self._get_client()
        now_epoch = int(self._wall_clock())
        expires_epoch = now_epoch + max(1, int(self.ttl_s))
        values = {
            ":token": {"S": assignment.token},
            ":expires": {"N": str(expires_epoch)},
            ":ttl": {"N": str(expires_epoch + 86400)},
        }
        keys = [
            (f"USER#{assignment.user_id}", "LEASE"),
            (f"SESSION#{assignment.session_id}", "ASSIGNMENT"),
            (
                self._slot_pk(
                    assignment.denoiser.role,
                    assignment.denoiser.worker_id,
                    assignment.denoiser.slot_index,
                ),
                "LEASE",
            ),
            (
                self._slot_pk(
                    assignment.vae.role,
                    assignment.vae.worker_id,
                    assignment.vae.slot_index,
                ),
                "LEASE",
            ),
        ]
        max_attempts = 6
        for attempt in range(max_attempts):
            worker_checks = []
            for slot in assignment.workers:
                worker_checks.append(
                    {
                        "ConditionCheck": {
                            "TableName": self.table_name,
                            "Key": {
                                "pk": {"S": f"WORKER#{slot.worker_id}"},
                                "sk": {"S": "HEARTBEAT"},
                            },
                            "ConditionExpression": (
                                "worker_epoch = :worker_epoch AND "
                                "heartbeat_expires_at > :now AND "
                                "(#lifecycle = :ready OR "
                                "(#lifecycle = :draining AND "
                                "(attribute_not_exists(drain_deadline) OR "
                                "drain_deadline > :now)))"
                            ),
                            "ExpressionAttributeNames": {"#lifecycle": "lifecycle"},
                            "ExpressionAttributeValues": {
                                ":worker_epoch": {"S": slot.worker_epoch},
                                ":now": {"N": str(now_epoch)},
                                ":ready": {"S": "ready"},
                                ":draining": {"S": "draining"},
                            },
                        }
                    }
                )
            try:
                client.transact_write_items(
                    TransactItems=worker_checks
                    + [
                        {
                            "Update": {
                                "TableName": self.table_name,
                                "Key": {"pk": {"S": pk}, "sk": {"S": sk}},
                                "UpdateExpression": (
                                    "SET lease_expires_at = :expires, #ttl = :ttl"
                                ),
                                "ConditionExpression": "lease_token = :token",
                                "ExpressionAttributeNames": {"#ttl": "ttl"},
                                "ExpressionAttributeValues": values,
                            }
                        }
                        for pk, sk in keys
                    ]
                )
                break
            except client.exceptions.TransactionCanceledException as exc:
                reason = self._renew_cancellation_reason(
                    client,
                    assignment,
                    keys,
                    now_epoch=now_epoch,
                )
                if reason is not None:
                    raise CoordinatorRejected(reason) from exc
                if attempt == max_attempts - 1:
                    raise CoordinatorRejected("LEASE_RENEW_CONFLICT") from exc
                token_jitter = (
                    int(hashlib.sha256(assignment.token.encode()).hexdigest()[:2], 16)
                    / 255
                )
                time.sleep(0.01 * (2**attempt) * (1 + token_jitter))
        return replace(assignment, expires_at=self._lease_clock() + self.ttl_s)

    async def release(self, assignment: SessionAssignment) -> None:
        await asyncio.to_thread(self._release_sync, assignment)

    def _release_sync(self, assignment: SessionAssignment) -> None:
        client = self._get_client()
        token = {":token": {"S": assignment.token}}
        try:
            client.transact_write_items(
                TransactItems=[
                    {
                        "Delete": {
                            "TableName": self.table_name,
                            "Key": {
                                "pk": {"S": f"USER#{assignment.user_id}"},
                                "sk": {"S": "LEASE"},
                            },
                            "ConditionExpression": "lease_token = :token",
                            "ExpressionAttributeValues": token,
                        }
                    },
                    {
                        "Delete": {
                            "TableName": self.table_name,
                            "Key": {
                                "pk": {"S": f"SESSION#{assignment.session_id}"},
                                "sk": {"S": "ASSIGNMENT"},
                            },
                            "ConditionExpression": "lease_token = :token",
                            "ExpressionAttributeValues": token,
                        }
                    },
                    *[
                        {
                            "Update": {
                                "TableName": self.table_name,
                                "Key": {
                                    "pk": {
                                        "S": self._slot_pk(
                                            slot.role,
                                            slot.worker_id,
                                            slot.slot_index,
                                        )
                                    },
                                    "sk": {"S": "LEASE"},
                                },
                                "UpdateExpression": (
                                    "REMOVE lease_token, user_id, session_id, "
                                    "generation_id, lease_expires_at"
                                ),
                                "ConditionExpression": "lease_token = :token",
                                "ExpressionAttributeValues": token,
                            }
                        }
                        for slot in assignment.workers
                    ],
                ]
            )
        except client.exceptions.TransactionCanceledException:
            return


class _MinimalAsyncRedisClient:
    """Small RESP2 client used when redis-py is unavailable in source patches."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str | None,
        password: str | None,
        db: int,
        ssl: bool,
        timeout_s: float,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.db = db
        self.ssl = ssl
        self.timeout_s = timeout_s
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connection_lock = asyncio.Lock()

    @staticmethod
    def _encode_command(*parts: Any) -> bytes:
        encoded = [f"*{len(parts)}\r\n".encode()]
        for part in parts:
            raw = str(part).encode()
            encoded.append(f"${len(raw)}\r\n".encode())
            encoded.append(raw)
            encoded.append(b"\r\n")
        return b"".join(encoded)

    async def _read_line(self, reader: asyncio.StreamReader) -> bytes:
        line = await reader.readline()
        if not line.endswith(b"\r\n"):
            raise RuntimeError("malformed Redis response")
        return line[:-2]

    async def _read_response(self, reader: asyncio.StreamReader) -> Any:
        prefix = await reader.readexactly(1)
        if prefix == b"+":
            return (await self._read_line(reader)).decode()
        if prefix == b"-":
            message = (await self._read_line(reader)).decode()
            raise RuntimeError(f"Redis error: {message}")
        if prefix == b":":
            return int(await self._read_line(reader))
        if prefix == b"$":
            length = int(await self._read_line(reader))
            if length < 0:
                return None
            data = await reader.readexactly(length)
            await reader.readexactly(2)
            return data.decode()
        if prefix == b"*":
            length = int(await self._read_line(reader))
            if length < 0:
                return None
            return [await self._read_response(reader) for _ in range(length)]
        raise RuntimeError(f"unsupported Redis response prefix: {prefix!r}")

    async def _connect(self) -> None:
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port, ssl=self.ssl),
            timeout=self.timeout_s,
        )
        if self.password:
            if self.username:
                await self._execute_unlocked("AUTH", self.username, self.password)
            else:
                await self._execute_unlocked("AUTH", self.password)
        if self.db:
            await self._execute_unlocked("SELECT", self.db)

    async def _close_connection(self) -> None:
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None

    async def _execute_unlocked(self, *parts: Any) -> Any:
        if self._reader is None or self._writer is None:
            await self._connect()
        assert self._reader is not None and self._writer is not None
        self._writer.write(self._encode_command(*parts))
        await asyncio.wait_for(self._writer.drain(), timeout=self.timeout_s)
        return await asyncio.wait_for(
            self._read_response(self._reader),
            timeout=self.timeout_s,
        )

    async def execute(self, *parts: Any) -> Any:
        async with self._connection_lock:
            try:
                return await self._execute_unlocked(*parts)
            except Exception:
                await self._close_connection()
                return await self._execute_unlocked(*parts)

    async def get(self, key: str) -> str | None:
        return await self.execute("GET", key)

    async def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        px: int | None = None,
        ex: int | None = None,
    ) -> bool:
        parts: list[Any] = ["SET", key, value]
        if nx:
            parts.append("NX")
        if px is not None:
            parts.extend(("PX", px))
        if ex is not None:
            parts.extend(("EX", ex))
        return await self.execute(*parts) == "OK"

    async def delete(self, key: str) -> int:
        return int(await self.execute("DEL", key))

    async def eval(self, script: str, numkeys: int, *args: Any) -> Any:
        return await self.execute("EVAL", script, numkeys, *args)

    async def aclose(self) -> None:
        async with self._connection_lock:
            await self._close_connection()


class RedisCoordinatorStore:
    """Multi-replica Coordinator state backed by a Redis-compatible service.

    The state lives in one JSON key protected by one short-lived lock key. This
    keeps all admission mutations single-slot on Redis Cluster/managed Redis
    while preserving the same session/slot fencing semantics as the reference
    in-memory store. The coordinator workload is tiny compared with model
    traffic, so this favors correctness and easy operations over sharding state
    across many Redis keys.
    """

    _RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
  return redis.call("del", KEYS[1])
end
return 0
"""

    supports_player_lifetimes = True

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        host: str = "localhost",
        port: int = 6379,
        username: str | None = None,
        password: str | None = None,
        password_env: str | None = "REDIS_PASSWORD",
        db: int = 0,
        ssl: bool = False,
        key_prefix: str = "world-model-realtime",
        state_ttl_s: int = 86400,
        lock_ttl_s: float = 2.0,
        lock_timeout_s: float = 1.0,
        lock_retry_s: float = 0.02,
        ttl_s: float,
        worker_ttl_s: float,
        routing_revision_aliases: Mapping[str, object] | None = None,
        candidate_limit: int = 64,
        capacity_limits: Mapping[WorkerRole, int] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        client: Any | None = None,
    ) -> None:
        if ttl_s <= 0 or worker_ttl_s <= 0:
            raise ValueError("lease TTLs must be positive")
        if state_ttl_s <= 0:
            raise ValueError("state_ttl_s must be positive")
        if lock_ttl_s <= 0 or lock_timeout_s <= 0 or lock_retry_s <= 0:
            raise ValueError("Redis lock timings must be positive")
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        if capacity_limits and any(limit < 1 for limit in capacity_limits.values()):
            raise ValueError("worker capacity limits must be positive")
        prefix = key_prefix.strip()
        if not prefix:
            raise ValueError("key_prefix is required")
        hash_tag = (
            prefix if prefix.startswith("{") and "}" in prefix else f"{{{prefix}}}"
        )
        self.redis_url = redis_url
        self.host = host
        self.port = int(port)
        self.username = username
        self.password = (
            password
            if password is not None
            else (os.getenv(password_env) if password_env else None)
        )
        self.db = int(db)
        self.ssl = bool(ssl)
        self.state_key = f"{hash_tag}:coordinator:state"
        self.lock_key = f"{hash_tag}:coordinator:lock"
        self.state_ttl_s = int(state_ttl_s)
        self.lock_ttl_s = float(lock_ttl_s)
        self.lock_timeout_s = float(lock_timeout_s)
        self.lock_retry_s = float(lock_retry_s)
        self.ttl_s = float(ttl_s)
        self.worker_ttl_s = float(worker_ttl_s)
        self.routing_revision_aliases = _normalize_routing_revision_aliases(
            routing_revision_aliases
        )
        self.candidate_limit = candidate_limit
        self.capacity_limits = dict(capacity_limits or {})
        self._clock = clock
        self._wall_clock = wall_clock
        self._client = client
        self._owns_client = client is None

    def _get_client(self):
        if self._client is None:
            try:
                import redis.asyncio as redis
            except ImportError:
                logger.warning("redis-py is unavailable; using minimal Redis client")
                self._client = _MinimalAsyncRedisClient(
                    host=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    db=self.db,
                    ssl=self.ssl,
                    timeout_s=max(1.0, self.lock_timeout_s),
                )
                return self._client
            kwargs = {
                "decode_responses": True,
                "socket_timeout": max(1.0, self.lock_timeout_s),
                "socket_connect_timeout": max(1.0, self.lock_timeout_s),
            }
            if self.username:
                kwargs["username"] = self.username
            if self.password:
                kwargs["password"] = self.password
            if self.redis_url:
                self._client = redis.from_url(self.redis_url, **kwargs)
            else:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    ssl=self.ssl,
                    **kwargs,
                )
        return self._client

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "schema": 1,
            "workers": {},
            "assignments_by_token": {},
            "assignments_by_user": {},
            "slot_tokens": {},
            "attempt_owners": {},
            "waiting": {},
            "quarantined_workers": {},
            "verified_releases": {},
            "player_closures": {},
        }

    async def _load_state(self, client: Any) -> dict[str, Any]:
        payload = await client.get(self.state_key)
        if not payload:
            return self._empty_state()
        if isinstance(payload, bytes):
            payload = payload.decode()
        state = json.loads(payload)
        if not isinstance(state, dict):
            raise RuntimeError("Redis Coordinator state is not a JSON object")
        for key, value in self._empty_state().items():
            state.setdefault(key, value.copy() if isinstance(value, dict) else value)
        if state.get("schema") != 1:
            raise RuntimeError(
                f"unsupported Redis Coordinator state schema: {state.get('schema')}"
            )
        return state

    async def _save_state(self, client: Any, state: dict[str, Any]) -> None:
        # A lease/Redis TTL cannot attest that a consumed player's GPU or
        # SDK resources stopped. Retain uncertain owners until verified release.
        retained = bool(state.get("quarantined_workers")) or any(
            entry.get("player_expires_at") is not None
            for entry in state.get("assignments_by_token", {}).values()
        )
        await client.set(
            self.state_key,
            json.dumps(state, separators=(",", ":"), sort_keys=True),
            **({} if retained else {"ex": self.state_ttl_s}),
        )

    async def _acquire_lock(self, client: Any) -> str:
        token = uuid4().hex
        deadline = time.monotonic() + self.lock_timeout_s
        while True:
            if await client.set(
                self.lock_key,
                token,
                nx=True,
                px=max(1, int(self.lock_ttl_s * 1000)),
            ):
                return token
            if time.monotonic() >= deadline:
                raise CoordinatorRejected("COORDINATOR_STORE_BUSY", retry_after_s=0.05)
            await asyncio.sleep(self.lock_retry_s)

    async def _release_lock(self, client: Any, token: str) -> None:
        try:
            await client.eval(self._RELEASE_LOCK_SCRIPT, 1, self.lock_key, token)
        except Exception:
            try:
                current = await client.get(self.lock_key)
                if current == token:
                    await client.delete(self.lock_key)
            except Exception:
                logger.warning(
                    "failed to release Redis Coordinator lock", exc_info=True
                )

    @asynccontextmanager
    async def _locked_state(self):
        client = self._get_client()
        token = await self._acquire_lock(client)
        try:
            state = await self._load_state(client)
            yield state
            await self._save_state(client, state)
        finally:
            await self._release_lock(client, token)

    @staticmethod
    def _slot_key(role: WorkerRole, worker_id: str, slot_index: int) -> str:
        return f"{role}|{worker_id}|{slot_index}"

    @staticmethod
    def _heartbeat_from_state(payload: dict[str, Any]) -> WorkerHeartbeat:
        field_names = {field.name for field in fields(WorkerHeartbeat)}
        return WorkerHeartbeat(
            **{key: value for key, value in payload.items() if key in field_names}
        )

    @staticmethod
    def _slot_from_state(payload: dict[str, Any]) -> WorkerSlot:
        field_names = {field.name for field in fields(WorkerSlot)}
        return WorkerSlot(
            **{key: value for key, value in payload.items() if key in field_names}
        )

    def _assignment_from_state(self, payload: dict[str, Any]) -> SessionAssignment:
        remaining_s = max(0.0, float(payload["expires_at"]) - self._wall_clock())
        return SessionAssignment(
            user_id=str(payload["user_id"]),
            session_id=str(payload["session_id"]),
            generation_id=str(payload["generation_id"]),
            token=str(payload["token"]),
            expires_at=self._clock() + remaining_s,
            admitted_at=payload.get("admitted_at"),
            player_expires_at=payload.get("player_expires_at"),
            denoiser=self._slot_from_state(payload["denoiser"]),
            vae=self._slot_from_state(payload["vae"]) if payload.get("vae") else None,
        )

    @staticmethod
    def _assignment_identity_matches(
        current: SessionAssignment,
        requested: SessionAssignment,
    ) -> bool:
        if (
            current.user_id,
            current.session_id,
            current.generation_id,
            current.token,
            current.admitted_at,
            current.player_expires_at,
        ) != (
            requested.user_id,
            requested.session_id,
            requested.generation_id,
            requested.token,
            requested.admitted_at,
            requested.player_expires_at,
        ):
            return False
        if len(current.workers) != len(requested.workers):
            return False
        for current_slot, requested_slot in zip(current.workers, requested.workers):
            if (
                current_slot.role,
                current_slot.worker_id,
                current_slot.slot_index,
                current_slot.worker_epoch,
            ) != (
                requested_slot.role,
                requested_slot.worker_id,
                requested_slot.slot_index,
                requested_slot.worker_epoch,
            ):
                return False
        return True

    def _release_token(self, state: dict[str, Any], token: str) -> None:
        current = state["assignments_by_token"].pop(token, None)
        if current is None:
            return
        user_id = str(current.get("user_id", ""))
        if state["assignments_by_user"].get(user_id) == token:
            state["assignments_by_user"].pop(user_id, None)
        for slot_payload in filter(None, (current.get("denoiser"), current.get("vae"))):
            key = self._slot_key(
                slot_payload.get("role"),
                str(slot_payload.get("worker_id")),
                int(slot_payload.get("slot_index", -1)),
            )
            if state["slot_tokens"].get(key) == token:
                state["slot_tokens"].pop(key, None)

    def _expire_state(self, state: dict[str, Any], now_epoch: float) -> None:
        expired_tokens = [
            token
            for token, assignment in state["assignments_by_token"].items()
            if float(assignment.get("expires_at", 0)) <= now_epoch
            and assignment.get("player_expires_at") is None
        ]
        for token in expired_tokens:
            self._release_token(state, token)
        for token, assignment in state["assignments_by_token"].items():
            deadline = assignment.get("player_expires_at")
            if deadline is not None and (
                float(assignment.get("expires_at", 0)) + 90 <= now_epoch
                or float(deadline) + 90 <= now_epoch
            ):
                for slot in filter(
                    None, (assignment.get("denoiser"), assignment.get("vae"))
                ):
                    worker = slot["worker_id"]
                    value = dict(worker_epoch=slot["worker_epoch"], token=token)
                    if state["quarantined_workers"].get(worker) != value:
                        logger.error(
                            "Player cleanup unconfirmed; quarantined worker=%s session=%s",
                            worker,
                            assignment["session_id"],
                        )
                        state["quarantined_workers"][worker] = value
        live_tokens = set(state["assignments_by_token"])
        state["assignments_by_user"] = {
            user_id: token
            for user_id, token in state["assignments_by_user"].items()
            if token in live_tokens
        }
        state["slot_tokens"] = {
            key: token
            for key, token in state["slot_tokens"].items()
            if token in live_tokens
        }
        state["attempt_owners"] = {
            attempt_id: owner
            for attempt_id, owner in state["attempt_owners"].items()
            if float(owner.get("expires_at", 0)) > now_epoch
        }
        state["waiting"] = {
            waiter_id: waiter
            for waiter_id, waiter in state["waiting"].items()
            if float(waiter.get("expires_at", 0)) > now_epoch
        }
        stale_worker_after_s = max(60.0, self.worker_ttl_s * 4)
        state["workers"] = {
            worker_id: worker
            for worker_id, worker in state["workers"].items()
            if float(worker.get("updated_at", 0)) + stale_worker_after_s > now_epoch
        }

    def _active_workers(
        self,
        state: dict[str, Any],
        *,
        role: WorkerRole,
        now_epoch: float,
        model_revision: str | None,
        vae_fingerprint: str | None,
        excluded_workers: frozenset[tuple[WorkerRole, str]],
    ) -> list[_WorkerState]:
        workers: list[_WorkerState] = []
        for worker in state["workers"].values():
            heartbeat = self._heartbeat_from_state(worker["heartbeat"])
            if heartbeat.role != role:
                continue
            if (heartbeat.role, heartbeat.worker_id) in excluded_workers:
                continue
            if float(worker.get("updated_at", 0)) + self.worker_ttl_s <= now_epoch:
                continue
            if heartbeat.lifecycle != "ready":
                continue
            quarantine = state["quarantined_workers"].get(heartbeat.worker_id)
            if quarantine and quarantine["worker_epoch"] == heartbeat.worker_epoch:
                continue
            if not _matches_worker_request(
                heartbeat,
                model_revision=model_revision,
                vae_fingerprint=vae_fingerprint,
                aliases=self.routing_revision_aliases,
            ):
                continue
            workers.append(
                _WorkerState(
                    heartbeat=heartbeat,
                    updated_at=float(worker.get("updated_at", 0)),
                )
            )
        return workers

    def _free_slots(
        self,
        state: dict[str, Any],
        workers: list[_WorkerState],
        *,
        identity: str,
    ) -> list[WorkerSlot]:
        slots: list[WorkerSlot] = []
        slot_tokens = state["slot_tokens"]
        for worker in workers:
            heartbeat = worker.heartbeat
            known = sum(
                self._slot_key(heartbeat.role, heartbeat.worker_id, index)
                in slot_tokens
                for index in range(heartbeat.capacity)
            )
            remaining = heartbeat.capacity - max(known, heartbeat.active_sessions)
            for slot_index in range(heartbeat.capacity):
                if remaining <= 0:
                    break
                key = self._slot_key(heartbeat.role, heartbeat.worker_id, slot_index)
                if key in slot_tokens:
                    continue
                remaining -= 1
                slots.append(
                    WorkerSlot(
                        worker_id=heartbeat.worker_id,
                        role=heartbeat.role,
                        endpoint=heartbeat.endpoint,
                        az=heartbeat.az,
                        slot_index=slot_index,
                        model_revision=heartbeat.model_revision,
                        vae_fingerprint=heartbeat.vae_fingerprint,
                        routing_revision=_routing_revision(heartbeat),
                        worker_epoch=heartbeat.worker_epoch,
                        lifecycle=heartbeat.lifecycle,
                        active_sessions=heartbeat.active_sessions,
                        runnable_sessions=heartbeat.runnable_sessions,
                        blocked_sessions=heartbeat.blocked_sessions,
                        queue_depth=heartbeat.queue_depth,
                        service_time_ms=heartbeat.service_time_ms,
                        reservation_endpoint=heartbeat.reservation_endpoint,
                        drain_deadline=heartbeat.drain_deadline,
                        capacity=heartbeat.capacity,
                    )
                )
        slots.sort(
            key=lambda slot: (
                slot.active_sessions / slot.capacity,
                slot.queue_depth,
                slot.service_time_ms,
                hashlib.sha256(
                    f"{identity}:{slot.role}:{slot.worker_id}".encode()
                ).digest(),
                slot.worker_id,
                slot.slot_index,
            )
        )
        return slots[: self.candidate_limit]

    async def heartbeat(self, heartbeat: WorkerHeartbeat) -> None:
        InMemoryCoordinatorStore._validate_heartbeat(heartbeat)
        capacity_limit = self.capacity_limits.get(heartbeat.role)
        if capacity_limit is not None and heartbeat.capacity > capacity_limit:
            heartbeat = replace(heartbeat, capacity=capacity_limit)
        async with self._locked_state() as state:
            now_epoch = self._wall_clock()
            self._expire_state(state, now_epoch)
            previous = state["quarantined_workers"].get(heartbeat.worker_id)
            if previous and previous["worker_epoch"] != heartbeat.worker_epoch:
                state["quarantined_workers"].pop(heartbeat.worker_id)
            state["workers"][heartbeat.worker_id] = {
                "heartbeat": asdict(heartbeat),
                "updated_at": now_epoch,
            }

    async def claim_attempt(
        self, *, attempt_id: str, owner_id: str, expires_at: float
    ) -> None:
        if not attempt_id or not owner_id:
            raise CoordinatorRejected("INVALID_ATTEMPT_OWNER")
        async with self._locked_state() as state:
            now_epoch = self._wall_clock()
            if expires_at <= now_epoch:
                raise CoordinatorRejected("INVALID_ATTEMPT_OWNER")
            self._expire_state(state, now_epoch)
            current = state["attempt_owners"].get(attempt_id)
            if (
                current is not None
                and current.get("owner_id") != owner_id
                and float(current.get("expires_at", 0)) > now_epoch
            ):
                raise CoordinatorRejected("ATTEMPT_ALREADY_OWNED")
            state["attempt_owners"][attempt_id] = {
                "owner_id": owner_id,
                "expires_at": float(expires_at),
            }

    async def waiting_started(self, waiter_id: str) -> None:
        if not waiter_id:
            raise ValueError("waiter_id is required")
        async with self._locked_state() as state:
            now_epoch = self._wall_clock()
            self._expire_state(state, now_epoch)
            state["waiting"][waiter_id] = {
                "expires_at": now_epoch + max(30.0, self.worker_ttl_s * 2),
            }

    async def waiting_finished(self, waiter_id: str) -> None:
        async with self._locked_state() as state:
            state["waiting"].pop(waiter_id, None)

    async def capacity_snapshot(
        self, *, model_revision: str | None = None, vae_fingerprint: str | None = None
    ) -> dict[str, Any]:
        _capacity_scope_requested(model_revision, vae_fingerprint)
        async with self._locked_state() as state:
            now_epoch = self._wall_clock()
            self._expire_state(state, now_epoch)
            roles = {
                role: {
                    "waiting_sessions": len(state["waiting"]),
                    "active_sessions": 0,
                    "queued_sessions": 0,
                    "free_slots": 0,
                    "draining_workers": 0,
                }
                for role in ("denoiser", "vae")
            }
            available = {
                (worker.heartbeat.role, worker.heartbeat.worker_id)
                for role in roles
                for worker in self._active_workers(
                    state,
                    role=role,
                    now_epoch=now_epoch,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                    excluded_workers=frozenset(),
                )
            }
            for worker in state["workers"].values():
                if float(worker.get("updated_at", 0)) + self.worker_ttl_s <= now_epoch:
                    continue
                heartbeat = self._heartbeat_from_state(worker["heartbeat"])
                if not _matches_worker_request(
                    heartbeat,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                    aliases=self.routing_revision_aliases,
                ):
                    continue
                role_values = roles[heartbeat.role]
                known = sum(
                    any(
                        slot["worker_id"] == heartbeat.worker_id
                        for slot in [assignment["denoiser"], assignment.get("vae")]
                        if slot
                    )
                    for assignment in state["assignments_by_token"].values()
                )
                occupied = max(known, heartbeat.active_sessions)
                role_values["active_sessions"] += occupied
                role_values["queued_sessions"] += heartbeat.queue_depth
                if heartbeat.lifecycle == "draining":
                    role_values["draining_workers"] += 1
                elif (heartbeat.role, heartbeat.worker_id) in available:
                    role_values["free_slots"] += max(0, heartbeat.capacity - occupied)
            return {"observed_at": now_epoch, "roles": roles}

    async def acquire(
        self,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        model_revision: str,
        vae_fingerprint: str,
        player_ttl_s: float | None = None,
        excluded_workers: frozenset[tuple[WorkerRole, str]] = frozenset(),
    ) -> SessionAssignment:
        if not all(
            (user_id, session_id, generation_id, model_revision, vae_fingerprint)
        ):
            raise CoordinatorRejected("INVALID_SESSION_IDENTITY")
        async with self._locked_state() as state:
            now_epoch = self._wall_clock()
            self._expire_state(state, now_epoch)
            user_token = state["assignments_by_user"].get(user_id)
            if user_token in state["assignments_by_token"]:
                raise CoordinatorRejected("USER_SESSION_LIMIT")

            identity = f"{user_id}:{session_id}:{generation_id}"
            denoisers = self._free_slots(
                state,
                self._active_workers(
                    state,
                    role="denoiser",
                    now_epoch=now_epoch,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                    excluded_workers=excluded_workers,
                ),
                identity=identity,
            )
            vaes = self._free_slots(
                state,
                self._active_workers(
                    state,
                    role="vae",
                    now_epoch=now_epoch,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                    excluded_workers=excluded_workers,
                ),
                identity=identity,
            )
            pairs = DynamoDBCoordinatorStore._candidate_pairs(
                denoisers,
                vaes,
                identity=identity,
            )
            if vae_fingerprint == "local":
                pairs = [(slot, None) for slot in denoisers]
            if not pairs:
                raise CoordinatorRejected("CAPACITY_EXHAUSTED", retry_after_s=0.25)
            denoiser, vae = pairs[0]
            token = uuid4().hex
            assignment_payload = {
                "user_id": user_id,
                "session_id": session_id,
                "generation_id": generation_id,
                "token": token,
                "expires_at": now_epoch + self.ttl_s,
                "admitted_at": now_epoch if player_ttl_s is not None else None,
                "player_expires_at": now_epoch + player_ttl_s
                if player_ttl_s is not None
                else None,
                "denoiser": asdict(denoiser),
                "vae": asdict(vae) if vae is not None else None,
            }
            state["assignments_by_token"][token] = assignment_payload
            state["assignments_by_user"][user_id] = token
            for slot in filter(None, (denoiser, vae)):
                state["slot_tokens"][
                    self._slot_key(slot.role, slot.worker_id, slot.slot_index)
                ] = token
            return self._assignment_from_state(assignment_payload)

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment:
        async with self._locked_state() as state:
            now_epoch = self._wall_clock()
            self._expire_state(state, now_epoch)
            current_payload = state["assignments_by_token"].get(assignment.token)
            if current_payload is None:
                raise CoordinatorRejected("LEASE_LOST")
            current = self._assignment_from_state(current_payload)
            if not self._assignment_identity_matches(current, assignment):
                raise CoordinatorRejected("LEASE_LOST")
            if (
                current.player_expires_at is not None
                and current.player_expires_at <= now_epoch
            ):
                raise CoordinatorRejected("PLAYER_EXPIRED")
            for slot in current.workers:
                worker = state["workers"].get(slot.worker_id)
                if worker is None:
                    raise CoordinatorRejected("WORKER_LOST")
                heartbeat = self._heartbeat_from_state(worker["heartbeat"])
                if (
                    float(worker.get("updated_at", 0)) + self.worker_ttl_s <= now_epoch
                    or heartbeat.worker_epoch != slot.worker_epoch
                    or heartbeat.lifecycle == "failed"
                    or (
                        heartbeat.lifecycle == "draining"
                        and heartbeat.drain_deadline is not None
                        and heartbeat.drain_deadline <= now_epoch
                    )
                ):
                    raise CoordinatorRejected("WORKER_LOST")
            current_payload["expires_at"] = now_epoch + self.ttl_s
            state["assignments_by_token"][assignment.token] = current_payload
            state["assignments_by_user"][assignment.user_id] = assignment.token
            return self._assignment_from_state(current_payload)

    async def release(self, assignment: SessionAssignment) -> None:
        async with self._locked_state() as state:
            current = state["assignments_by_token"].get(assignment.token)
            if assignment.player_expires_at is not None or (
                current is not None and current.get("player_expires_at") is not None
            ):
                raise CoordinatorRejected("VERIFIED_RELEASE_REQUIRED")
            self._release_token(state, assignment.token)

    async def quarantine(self, assignment: SessionAssignment) -> None:
        async with self._locked_state() as state:
            payload = state["assignments_by_token"].get(assignment.token)
            if payload is None:
                return
            current = self._assignment_from_state(payload)
            if _release_identity(current) != _release_identity(assignment):
                raise CoordinatorRejected("LEASE_LOST")
            for slot in current.workers:
                if (
                    slot.worker_id not in state["quarantined_workers"]
                    and len(state["quarantined_workers"]) >= 2048
                ):
                    raise CoordinatorRejected("COORDINATOR_STORE_BUSY")
                state["quarantined_workers"][slot.worker_id] = dict(
                    worker_epoch=slot.worker_epoch,
                    token=assignment.token,
                )

    async def verified_release_status(self, assignment):
        async with self._locked_state() as state:
            payload = state["assignments_by_token"].get(assignment.token)
            if payload is not None:
                current = self._assignment_from_state(payload)
                if _release_identity(current) == _release_identity(assignment):
                    return current, False
            receipt = state["verified_releases"].get(assignment.token)
            if receipt and receipt["identity"] == _release_identity(assignment):
                return None, True
            raise CoordinatorRejected("LEASE_LOST")

    async def complete_verified_release(self, assignment):
        async with self._locked_state() as state:
            payload = state["assignments_by_token"].get(assignment.token)
            if payload is None:
                receipt = state["verified_releases"].get(assignment.token)
                if receipt and receipt["identity"] == _release_identity(assignment):
                    return
                raise CoordinatorRejected("LEASE_LOST")
            current = self._assignment_from_state(payload)
            if _release_identity(current) != _release_identity(assignment):
                raise CoordinatorRejected("LEASE_LOST")
            self._release_token(state, current.token)
            _retain_release_receipt(
                state["verified_releases"], current, self._wall_clock()
            )
            cleanup = state["player_closures"].pop(current.token, None)
            if cleanup is not None:
                state["verified_releases"][current.token]["cleanup"] = cleanup
            for slot in current.workers:
                previous = state["quarantined_workers"].get(slot.worker_id)
                if previous == dict(
                    worker_epoch=slot.worker_epoch, token=current.token
                ):
                    del state["quarantined_workers"][slot.worker_id]

    async def player_cleanup(self, assignment, receipt=None):
        async with self._locked_state() as state:
            payload = state["assignments_by_token"].get(assignment.token)
            if payload is None:
                raise CoordinatorRejected("LEASE_LOST")
            current = self._assignment_from_state(payload)
            if _release_identity(current) != _release_identity(assignment):
                raise CoordinatorRejected("LEASE_LOST")
            return _record_player_cleanup(state["player_closures"], current, receipt)

    async def clear_quarantine(self, assignment: SessionAssignment) -> None:
        async with self._locked_state() as state:
            for slot in assignment.workers:
                previous = state["quarantined_workers"].get(slot.worker_id)
                if previous == dict(
                    worker_epoch=slot.worker_epoch, token=assignment.token
                ):
                    del state["quarantined_workers"][slot.worker_id]

    async def wait_for_change(self, timeout_s: float) -> None:
        await asyncio.sleep(timeout_s)

    async def close(self) -> None:
        if self._client is None or not self._owns_client:
            return
        closer = getattr(self._client, "aclose", None)
        if closer is None:
            closer = getattr(self._client, "close", None)
        if closer is not None:
            result = closer()
            if asyncio.iscoroutine(result):
                await result
        self._client = None


class RealtimeCoordinator:
    def __init__(
        self,
        store: CoordinatorStore,
        *,
        wait_timeout_s: float = 10.0,
        reservation_client: WorkerReservationClient | None = None,
        player_session_ttl_seconds: float = 1800,
    ) -> None:
        if (
            type(player_session_ttl_seconds) not in (int, float)
            or not math.isfinite(player_session_ttl_seconds)
            or not 1 <= player_session_ttl_seconds <= 86400
        ):
            raise ValueError("Invalid player session lifetime")
        self.player_session_ttl_seconds = float(player_session_ttl_seconds)
        self.store = store
        self.wait_timeout_s = max(0.0, wait_timeout_s)
        self.reservation_client = reservation_client
        self._late_cleanup_tasks: set[asyncio.Task] = set()

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    async def _retry_cleanup(self, operation, *, label: str) -> None:
        last_error: BaseException | None = None
        for attempt in range(1, 4):
            try:
                await operation()
                return
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                last_error = exc
                logger.warning(
                    "Coordinator cleanup attempt %s/3 failed for %s: %s",
                    attempt,
                    label,
                    exc,
                )
                await asyncio.sleep(0)
        assert last_error is not None
        logger.error(
            "Coordinator cleanup exhausted retries for %s",
            label,
            exc_info=last_error,
        )
        raise RuntimeError(f"coordinator cleanup failed for {label}") from last_error

    async def _compensate_assignment(
        self,
        assignment: SessionAssignment,
        *,
        strict_worker_cleanup: bool = True,
    ) -> None:
        if assignment.player_expires_at is not None:
            from .player_lifecycle import sdk_cleanup_receipt

            # Admission has not returned to Room: no SDK creation could start.
            await self.release_verified(
                assignment,
                sdk_cleanup=sdk_cleanup_receipt(
                    assignment,
                    state="not_created",
                    source_id=assignment.session_id,
                    owner_epoch="coordinator-admission",
                ),
            )
            return
        worker_failures: list[BaseException] = []
        failures: list[BaseException] = []
        if self.reservation_client is not None:
            results = await asyncio.gather(
                *(
                    self._retry_cleanup(
                        lambda slot=slot: self.reservation_client.release(
                            slot, token=assignment.token
                        ),
                        label=f"worker:{slot.role}:{slot.worker_id}",
                    )
                    for slot in assignment.workers
                ),
                return_exceptions=True,
            )
            worker_failures.extend(
                result for result in results if isinstance(result, BaseException)
            )
        try:
            await self._retry_cleanup(
                lambda: self.store.release(assignment),
                label=f"store:{assignment.session_id}:{assignment.generation_id}",
            )
        except BaseException as exc:
            failures.append(exc)
        if strict_worker_cleanup:
            failures.extend(worker_failures)
        elif worker_failures:
            logger.warning(
                "Coordinator ignored %s worker release cleanup failure(s) for "
                "session_id=%s generation_id=%s",
                len(worker_failures),
                assignment.session_id,
                assignment.generation_id,
            )
        if failures:
            raise RuntimeError(
                f"assignment compensation failed ({len(failures)} operation(s))"
            ) from failures[0]

    async def _finish_compensation(self, assignment: SessionAssignment) -> None:
        cleanup = asyncio.create_task(self._compensate_assignment(assignment))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        await cleanup

    async def _release_late_acquire(self, acquire_task: asyncio.Task) -> None:
        try:
            assignment = await acquire_task
        except BaseException:
            return
        try:
            await self._compensate_assignment(assignment)
        except BaseException:
            logger.exception("Failed to compensate a late Coordinator acquire")

    def _track_late_acquire(self, acquire_task: asyncio.Task) -> None:
        cleanup = asyncio.create_task(self._release_late_acquire(acquire_task))
        self._late_cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(self._late_cleanup_tasks.discard)

    async def _acquire_before_deadline(
        self,
        *,
        deadline: float | None,
        **request: Any,
    ) -> SessionAssignment:
        acquire_task = asyncio.create_task(self.store.acquire(**request))
        try:
            remaining = self._remaining(deadline)
            if remaining is None:
                return await acquire_task
            if remaining <= 0:
                self._track_late_acquire(acquire_task)
                raise TimeoutError
            done, _ = await asyncio.wait((acquire_task,), timeout=remaining)
            if acquire_task not in done:
                self._track_late_acquire(acquire_task)
                raise TimeoutError
            return acquire_task.result()
        except asyncio.CancelledError:
            self._track_late_acquire(acquire_task)
            raise

    async def heartbeat(self, heartbeat: WorkerHeartbeat) -> None:
        await self.store.heartbeat(heartbeat)

    async def claim_attempt(
        self, *, attempt_id: str, owner_id: str, expires_at: float
    ) -> None:
        await self.store.claim_attempt(
            attempt_id=attempt_id,
            owner_id=owner_id,
            expires_at=expires_at,
        )

    async def capacity_snapshot(
        self, *, model_revision: str | None = None, vae_fingerprint: str | None = None
    ) -> dict[str, Any]:
        if model_revision is None and vae_fingerprint is None:
            return await self.store.capacity_snapshot()
        return await self.store.capacity_snapshot(
            model_revision=model_revision, vae_fingerprint=vae_fingerprint
        )

    async def admit(
        self,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        model_revision: str,
        vae_fingerprint: str,
        wait_for_capacity: bool = True,
        player_session: bool = False,
        capacity_wait_ms: int | None = None,
    ) -> SessionAssignment:
        if type(player_session) is not bool:
            raise CoordinatorRejected("INVALID_PLAYER_SESSION")
        if player_session and not getattr(
            self.store, "supports_player_lifetimes", False
        ):
            raise CoordinatorRejected("PLAYER_LIFETIME_UNSUPPORTED")
        if capacity_wait_ms is not None and (
            type(capacity_wait_ms) is not int or not 1 <= capacity_wait_ms <= 10000
        ):
            raise CoordinatorRejected("INVALID_CAPACITY_WAIT")
        wait_timeout = self.wait_timeout_s
        if capacity_wait_ms is not None:
            wait_timeout = (
                min(wait_timeout, capacity_wait_ms / 1000)
                if wait_timeout > 0
                else capacity_wait_ms / 1000
            )
        deadline = time.monotonic() + wait_timeout if wait_timeout > 0 else None
        excluded_workers: set[tuple[WorkerRole, str]] = set()
        waiting_id: str | None = None
        try:
            while True:
                try:
                    selection_started = time.perf_counter()
                    try:
                        assignment = await self._acquire_before_deadline(
                            deadline=deadline,
                            user_id=user_id,
                            session_id=session_id,
                            generation_id=generation_id,
                            model_revision=model_revision,
                            vae_fingerprint=vae_fingerprint,
                            excluded_workers=frozenset(excluded_workers),
                            **(
                                {"player_ttl_s": self.player_session_ttl_seconds}
                                if player_session
                                else {}
                            ),
                        )
                    except BaseException as exc:
                        observe_coordinator_worker_selection_seconds(
                            time.perf_counter() - selection_started,
                            model=model_revision,
                            result=result_from_exception(exc),
                        )
                        raise
                    observe_coordinator_worker_selection_seconds(
                        time.perf_counter() - selection_started,
                        model=model_revision,
                        result="success",
                    )
                except TimeoutError as exc:
                    raise CoordinatorRejected(
                        "CAPACITY_EXHAUSTED", retry_after_s=0.1
                    ) from exc
                except CoordinatorRejected as exc:
                    if exc.reason != "CAPACITY_EXHAUSTED" or not wait_for_capacity:
                        raise
                    remaining = self._remaining(deadline)
                    if remaining is None:
                        remaining = 0.0
                    if remaining <= 0:
                        raise CoordinatorRejected(
                            "CAPACITY_EXHAUSTED", retry_after_s=exc.retry_after_s
                        ) from exc
                    if waiting_id is None:
                        waiting_id = uuid4().hex
                        await self.store.waiting_started(waiting_id)
                    wait_s = min(remaining, exc.retry_after_s or 0.1)
                    waiter = getattr(self.store, "wait_for_change", None)
                    if waiter is None:
                        await asyncio.sleep(wait_s)
                    else:
                        await waiter(wait_s)
                    continue

                if self.reservation_client is None:
                    return assignment
                failed_slot = None
                ttl_s = max(0.001, assignment.expires_at - time.monotonic())
                try:
                    for slot in assignment.workers:
                        failed_slot = slot
                        reserve = self.reservation_client.reserve(
                            slot,
                            token=assignment.token,
                            session_id=assignment.session_id,
                            generation_id=assignment.generation_id,
                            ttl_s=ttl_s,
                            **(
                                {"player_expires_at": assignment.player_expires_at}
                                if assignment.player_expires_at is not None
                                else {}
                            ),
                        )
                        remaining = self._remaining(deadline)
                        reservation_started = time.perf_counter()
                        try:
                            if remaining is None:
                                await reserve
                            elif remaining <= 0:
                                reserve.close()
                                raise TimeoutError
                            else:
                                await asyncio.wait_for(reserve, timeout=remaining)
                        except BaseException as exc:
                            observe_coordinator_capacity_reservation_seconds(
                                time.perf_counter() - reservation_started,
                                model=model_revision,
                                result=result_from_exception(exc),
                                worker_role=slot.role,
                            )
                            raise
                        observe_coordinator_capacity_reservation_seconds(
                            time.perf_counter() - reservation_started,
                            model=model_revision,
                            result="success",
                            worker_role=slot.role,
                        )
                except BaseException as exc:
                    cleanup_error = None
                    try:
                        await self._finish_compensation(assignment)
                    except BaseException as cleanup_exc:
                        cleanup_error = cleanup_exc
                        logger.exception(
                            "Coordinator failed to compensate rejected assignment"
                        )
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    if cleanup_error is not None:
                        raise CoordinatorRejected(
                            "COORDINATOR_CLEANUP_FAILED", retry_after_s=1.0
                        ) from cleanup_error
                    if failed_slot is not None:
                        excluded_workers.add((failed_slot.role, failed_slot.worker_id))
                    if (
                        not wait_for_capacity
                        or deadline is None
                        or time.monotonic() >= deadline
                    ):
                        raise CoordinatorRejected(
                            "CAPACITY_EXHAUSTED", retry_after_s=0.1
                        ) from exc
                    continue
                return assignment
        finally:
            if waiting_id is not None:
                await asyncio.gather(
                    self.store.waiting_finished(waiting_id),
                    return_exceptions=True,
                )

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment:
        return await self.store.renew(assignment)

    async def release(self, assignment: SessionAssignment) -> None:
        if assignment.player_expires_at is not None:
            raise CoordinatorRejected("VERIFIED_RELEASE_REQUIRED")
        await self._compensate_assignment(
            assignment,
            strict_worker_cleanup=False,
        )

    async def release_verified(
        self, assignment: SessionAssignment, *, sdk_cleanup=None
    ) -> None:
        if (
            self.reservation_client is None
            or not hasattr(self.reservation_client, "release_verified")
            or not hasattr(self.store, "verified_release_status")
        ):
            raise CoordinatorRejected("STRICT_RELEASE_UNAVAILABLE")
        current, released = await self.store.verified_release_status(assignment)
        if released:
            return
        if current.player_expires_at is not None:
            await self.store.player_cleanup(current, sdk_cleanup)
        results = await asyncio.gather(
            *(
                self.reservation_client.release_verified(
                    slot,
                    token=current.token,
                    **(
                        {"player": current}
                        if current.player_expires_at is not None
                        else {}
                    ),
                )
                for slot in current.workers
            ),
            return_exceptions=True,
        )
        if any(isinstance(result, BaseException) for result in results):
            raise CoordinatorRejected("RELEASE_PENDING")
        # No slot is freed before every worker has acknowledged owner cleanup.
        await self.store.complete_verified_release(current)

    async def quarantine(self, assignment: SessionAssignment) -> None:
        quarantine = getattr(self.store, "quarantine", None)
        if quarantine is None:
            raise CoordinatorRejected("STRICT_RELEASE_UNAVAILABLE")
        await quarantine(assignment)

    async def close(self) -> None:
        if self._late_cleanup_tasks:
            await asyncio.gather(*self._late_cleanup_tasks, return_exceptions=True)
        for closeable in (self.reservation_client, self.store):
            closer = getattr(closeable, "close", None)
            if closer is not None:
                await closer()
