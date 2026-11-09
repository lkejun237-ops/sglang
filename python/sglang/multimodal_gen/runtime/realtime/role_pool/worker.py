# SPDX-License-Identifier: Apache-2.0
"""Native service ownership, shared process budgets, deadlines and release proofs."""

import asyncio
import copy
import threading
import time
from uuid import uuid4

from .contracts import Credentials, PoolError, identifier, vector


class ProcessAuthority:
    def __init__(
        self,
        *,
        capacity,
        profiles,
        control_epoch,
        secret,
        boot_epoch=None,
        clock=time.time,
        profile_limits=None,
    ):
        self.capacity = vector(capacity)
        self.profiles = {identifier(k): vector(v) for k, v in profiles.items()}
        for costs in self.profiles.values():
            if set(costs) != set(self.capacity) or any(
                costs[k] > self.capacity[k] for k in costs
            ):
                raise ValueError("Profile exceeds shared process capacity")
        self.profile_limits = profile_limits or {}
        self.clock, self.boot_epoch = clock, boot_epoch or uuid4().hex
        self.credentials = Credentials(secret, control_epoch, clock=clock)
        self.lock = threading.RLock()
        self.services, self.held = {}, {}
        self.draining = False
        self.failed = False
        self.build_gate = threading.Lock()
        self.persistence_reads = set()

    def acquire(self, service_id, allocation_id, profile):
        with self.lock:
            if self.failed:
                raise PoolError("PROCESS_QUARANTINED")
            if self.draining:
                raise PoolError("PROCESS_DRAINING")
            if profile not in self.profiles:
                raise PoolError("UNSUPPORTED_PROFILE")
            existing = self.held.get(service_id)
            if existing:
                if existing != (allocation_id, profile):
                    raise PoolError("SERVICE_OCCUPIED")
                return
            used = {
                k: sum(self.profiles[p][k] for _, p in self.held.values())
                for k in self.capacity
            }
            if any(
                used[k] + self.profiles[profile][k] > self.capacity[k] * (1 + 1e-10)
                for k in used
            ):
                raise PoolError("PROCESS_CAPACITY_EXHAUSTED")
            self.held[service_id] = (allocation_id, profile)

    def release(self, service_id, allocation_id):
        with self.lock:
            if service_id in self.held:
                if self.held[service_id][0] != allocation_id:
                    raise PoolError("RELEASE_OWNER_MISMATCH")
                del self.held[service_id]


class ManagedService:
    def __init__(self, native, process, *, cleanup_timeout=90):
        self.native, self.process = native, process
        self.service_id = native.service["service_id"]
        self.owner = None
        self.state = "idle"
        self.terminal = {}
        self.terminal_at = {}
        self.terminal_recovery = {}
        self.cleanup_timeout = cleanup_timeout
        self.loop = None
        self.cleanup_task = None
        self.sockets = set()
        self.children = {}
        self.child_receipts = {}
        self.child_receipt_times = {}
        with process.lock:
            if self.service_id in process.services:
                raise ValueError("Duplicate managed service")
            process.services[self.service_id] = self
        original_build = native.build_callback

        async def build(group, topology):
            # Commit the native 10-second reservation first, then serialize
            # communicator construction for services sharing one model process.
            while not process.build_gate.acquire(blocking=False):
                with process.lock:
                    if self.state != "connecting" or process.failed:
                        raise PoolError("GROUP_BUILD_CANCELLED")
                await asyncio.sleep(0.01)
            try:
                with process.lock:
                    if self.state != "connecting" or process.failed:
                        raise PoolError("GROUP_BUILD_CANCELLED")
                return await native._call(original_build, group, topology)
            finally:
                process.build_gate.release()

        native.build_callback = build

    def _identity(self, body, *, require_owner=True):
        if body.get("boot_epoch") != self.process.boot_epoch:
            raise PoolError("PROCESS_EPOCH_MISMATCH")
        if (
            "topology" in body
            and body["topology"].get(self.native.service["role"]) != self.native.service
        ):
            raise PoolError("SERVICE_IDENTITY_MISMATCH")
        if require_owner:
            if self.owner is None or any(
                body.get(k) != self.owner[k]
                for k in ("allocation_id", "worker_group_id")
            ):
                raise PoolError("GROUP_OWNER_MISMATCH")
            for key in (
                "topology",
                "admission_id",
                "user_id",
                "player_id",
                "parent_generation_id",
                "profile",
                "profile_limits",
                "admitted_at",
                "admission_created_at",
                "player_expires_at",
                "operation",
            ):
                if body.get(key) != self.owner.get(key):
                    raise PoolError("BINDING_IDENTITY_CHANGED")
            revision = body.get("executor_revision")
            if type(revision) is not int or revision < self.owner["executor_revision"]:
                raise PoolError("STALE_EXECUTOR")
            self.owner["executor_revision"] = revision

    def snapshot(self, *, include_children=True):
        with self.process.lock, self.native.lock:
            owner = copy.deepcopy(self.owner)
            if owner is not None:
                owner.pop("reservation", None)
            return dict(
                service_id=self.service_id,
                boot_epoch=self.process.boot_epoch,
                state=self.state,
                native_state=self.native.state,
                restart_required=self.native.restart_required or self.process.failed,
                draining=self.process.draining,
                owner=owner,
                children=copy.deepcopy(self.children) if include_children else {},
                child_receipts=copy.deepcopy(self.child_receipts)
                if include_children
                else {},
            )

    def authorize(self, action, body, token):
        claims = self.process.credentials.verify(token, action, body)
        if body.get("control_epoch") != self.process.credentials.epoch:
            raise PoolError("CONTROL_EPOCH_MISMATCH")
        if action not in {"observe", "identity", "recovery"} and claims[
            "exp"
        ] > body.get("executor_until", 0):
            raise PoolError("EXECUTOR_LEASE_EXCEEDED")
        return claims

    async def command(self, action, body, token):
        self.loop = asyncio.get_running_loop()
        claims = self.authorize(action, body, token)
        with self.process.lock:
            if action == "identity":
                return dict(
                    service=copy.deepcopy(self.native.service),
                    boot_epoch=self.process.boot_epoch,
                    control_epoch=self.process.credentials.epoch,
                    capacity=copy.deepcopy(self.process.capacity),
                    profiles=copy.deepcopy(self.process.profiles),
                    profile_limits=copy.deepcopy(self.process.profile_limits),
                    deployment=copy.deepcopy(getattr(self.process, "deployment", {})),
                )
            self._identity(
                body,
                require_owner=action
                not in {
                    "observe",
                    "recovery",
                    "reserve",
                    "drain",
                    "finalize",
                    "destroy",
                    "persistence",
                },
            )
            if action == "persistence":
                identity = (body.get("allocation_id"), body.get("worker_group_id"))
                if self.owner is not None and identity == (
                    self.owner["allocation_id"],
                    self.owner["worker_group_id"],
                ):
                    self._identity(body)
                else:
                    terminal = self.terminal.get(identity)
                    original = self.terminal_recovery.get(identity)
                    if (
                        not terminal
                        or terminal.get("state") != "released"
                        or not original
                        or original.get("operation") != "prefill"
                        or self.process.clock() >= original["player_expires_at"]
                        or any(
                            body.get(k) != original.get(k)
                            for k in (
                                "operation",
                                "admission_id",
                                "user_id",
                                "player_id",
                                "parent_generation_id",
                                "topology",
                                "profile",
                                "profile_limits",
                                "admitted_at",
                                "admission_created_at",
                                "player_expires_at",
                            )
                        )
                    ):
                        raise PoolError("GROUP_OWNER_MISMATCH")
            if action == "recovery":
                offset, limit = body.get("offset", 0), body.get("limit", 32)
                if (
                    type(offset) is not int
                    or offset < 0
                    or type(limit) is not int
                    or not 1 <= limit <= 64
                ):
                    raise PoolError("INVALID_RECOVERY_PAGE", 422)
                keys = list(self.terminal)[offset : offset + limit]
                return dict(
                    active=self.snapshot(include_children=False),
                    process_capacity=dict(self.process.capacity),
                    profiles=copy.deepcopy(self.process.profiles),
                    terminals=[
                        dict(
                            receipt=self.terminal[k],
                            owner=self.terminal_recovery[k],
                            completed_at=self.terminal_at[k],
                        )
                        for k in keys
                    ],
                    next_offset=offset + len(keys)
                    if offset + len(keys) < len(self.terminal)
                    else None,
                    history_complete=False,
                )
            if action == "observe":
                return self.snapshot()
            if action == "drain":
                self.process.draining = True
                return self.snapshot()
            if action == "reserve":
                return self._reserve(body, claims)
            if action == "renew":
                deadline = body.get("player_expires_at")
                if (
                    deadline != self.owner["player_expires_at"]
                    or self.process.clock() >= deadline
                ):
                    raise PoolError("PLAYER_EXPIRED")
                if self.process.failed:
                    raise PoolError("PROCESS_QUARANTINED")
                if self.state not in {"reserved", "connecting", "connected"}:
                    raise PoolError("GROUP_CLOSING")
                if self.state == "connected" and (
                    self.native.state != "connected" or self.native.restart_required
                ):
                    self.state = "quarantined"
                    raise PoolError("GROUP_TRANSPORT_LOST")
                self.owner["lease_until"] = min(claims["exp"], deadline)
                return self.snapshot()
            if action == "finalize":
                return self._finalize(body)
            if action == "destroy":
                identity = (body.get("allocation_id"), body.get("worker_group_id"))
                if self.owner is None:
                    with self.native.lock:
                        if self.native.state != "idle":
                            raise PoolError("UNOWNED_NATIVE_STATE")
                    self._remember(
                        identity,
                        dict(
                            allocation_id=identity[0],
                            worker_group_id=identity[1],
                            service_id=self.service_id,
                            boot_epoch=self.process.boot_epoch,
                            state="released",
                        ),
                        recovery=body,
                    )
                    return self.snapshot()
                self._identity(body)
        if action == "build":
            return await self._build(body)
        if action == "destroy":
            await self.begin_cleanup()
            return self.snapshot()
        if action == "child_status":
            with self.process.lock:
                key = body.get("session_id")
                return {
                    "child_receipts": {key: copy.deepcopy(self.child_receipts[key])}
                    if key in self.child_receipts
                    else {}
                }

        if action == "persistence":
            callback = getattr(self.native, "persistence", None)
            if callback is None:
                raise PoolError("CACHE_PERSISTENCE_UNAVAILABLE", 503)
            with self.process.lock:
                if len(self.process.persistence_reads) >= 2:
                    return {"state": "pending"}
                task = asyncio.create_task(
                    asyncio.to_thread(callback, body["descriptor"])
                )
                self.process.persistence_reads.add(task)

            def done(value):
                with self.process.lock:
                    self.process.persistence_reads.discard(value)
                if not value.cancelled():
                    value.exception()

            task.add_done_callback(done)
            return await asyncio.wait_for(asyncio.shield(task), 5)
        raise PoolError("UNKNOWN_CONTROL_ACTION", 422)

    def _reserve(self, body, claims):
        now = self.process.clock()
        if body.get("operation") not in {None, "prefill"}:
            raise PoolError("INVALID_BINDING", 422)
        if self.process.failed:
            raise PoolError("PROCESS_QUARANTINED")
        required = {
            "allocation_id",
            "worker_group_id",
            "admission_id",
            "user_id",
            "player_id",
            "parent_generation_id",
            "profile",
            "player_expires_at",
            "admitted_at",
            "admission_created_at",
            "executor_revision",
            "topology",
            "profile_limits",
        }
        if not required <= body.keys() or type(body["executor_revision"]) is not int:
            raise PoolError("INVALID_BINDING", 422)
        for key in (
            "allocation_id",
            "worker_group_id",
            "admission_id",
            "user_id",
            "player_id",
            "profile",
        ):
            identifier(body[key])
        if body["profile_limits"] != self.process.profile_limits.get(body["profile"]):
            raise PoolError("PROFILE_CONTRACT_MISMATCH")
        if not now < body["player_expires_at"] or body["admitted_at"] > now + 5:
            raise PoolError("PLAYER_EXPIRED")
        if (body["allocation_id"], body["worker_group_id"]) in self.terminal:
            raise PoolError("ALLOCATION_ALREADY_RELEASED")
        if self.owner is not None:
            self._identity(body)
            if any(body[k] != self.owner[k] for k in required - {"executor_revision"}):
                raise PoolError("BINDING_IDENTITY_CHANGED")
            if self.state != "reserved":
                raise PoolError("SERVICE_OCCUPIED")
            return copy.deepcopy(self.owner["reservation"])
        if self.state != "idle" or self.native.restart_required:
            raise PoolError("SERVICE_UNAVAILABLE")
        if body["topology"].get(self.native.service["role"]) != self.native.service:
            raise PoolError("SERVICE_IDENTITY_MISMATCH")
        self.process.acquire(self.service_id, body["allocation_id"], body["profile"])
        try:
            reply, status = self.native.query()
            if status != 200 or reply["state"] != "available":
                raise PoolError("SERVICE_NOT_AVAILABLE")
        except BaseException:
            self.process.release(self.service_id, body["allocation_id"])
            raise
        self.owner = {k: copy.deepcopy(body[k]) for k in required}
        if body.get("operation") is not None:
            self.owner["operation"] = body["operation"]
        self.owner.update(
            lease_until=min(claims["exp"], body["player_expires_at"]),
            reservation=reply,
            reserved_until=now + reply["expires_in_seconds"],
        )
        self.state = "reserved"
        return copy.deepcopy(reply)

    async def _build(self, body):
        with self.process.lock:
            if self.process.failed:
                raise PoolError("PROCESS_QUARANTINED")
            if self.state == "connected":
                return self.snapshot()
            if (
                self.state not in {"reserved", "connecting"}
                or self.process.clock() >= self.owner["lease_until"]
            ):
                raise PoolError("GROUP_CLOSING")
            if body.get("topology") != self.owner["topology"]:
                raise PoolError("TOPOLOGY_CHANGED")
            self.state = "connecting"
            payload = dict(
                worker_group_id=self.owner["worker_group_id"],
                reservation_token=self.owner["reservation"]["reservation_token"],
                topology=self.owner["topology"],
            )
        reply, status = await self.native.build(payload)
        with self.process.lock:
            if self.state == "connecting" and status == 200:
                self.state = (
                    "connected"
                    if self.process.clock()
                    < min(self.owner["lease_until"], self.owner["player_expires_at"])
                    else "cleaning"
                )
            elif status != 200:
                if self.native.restart_required:
                    self.process.failed = True
                self.state = (
                    "quarantined" if self.native.restart_required else "cleaned"
                )
            closing = self.state not in {"connected", "cleaned", "quarantined"}
        if closing:
            await self.begin_cleanup()
        return self.snapshot()

    async def begin_cleanup(self):
        with self.process.lock:
            if self.state in {"idle", "cleaned"}:
                return
            self.state = "cleaning"
            for task in list(self.sockets):
                task.get_loop().call_soon_threadsafe(task.cancel)
            if self.cleanup_task is None or self.cleanup_task.done():
                self.cleanup_task = asyncio.create_task(self._cleanup())

    async def _cleanup(self):
        try:
            async with asyncio.timeout(self.cleanup_timeout):
                # Native build is shielded; do not cancel a possibly live NCCL
                # operation or let its late success overwrite a free service.
                operation = self.native.operation
                if operation is not None and not operation.done():
                    await asyncio.shield(operation)
                with self.native.lock:
                    state, group = self.native.state, self.native.group
                    if state == "reserved":
                        self.native.state, self.native.token = "idle", None
                        state = "idle"
                if state in {"connected", "draining"}:
                    # A closed browser cannot ACK graceful output EOS. Abort
                    # only this group's execution, using the native rollback
                    # callbacks which still verify threads/collectives stopped.
                    await self.native.destroy(dict(worker_group_id=group), abort=True)
                    await asyncio.shield(self.native.operation)
                with self.process.lock, self.native.lock:
                    if self.native.state != "idle" or self.native.restart_required:
                        raise PoolError("NATIVE_CLEANUP_UNCONFIRMED")
                    self.state = "cleaned"
        except BaseException:
            with self.process.lock:
                self.state = "quarantined"
                if self.native.restart_required:
                    self.process.failed = True
            # Keep ownership and capacity. No restart, no fake release.

    def _finalize(self, body):
        identity = (body.get("allocation_id"), body.get("worker_group_id"))
        if identity in self.terminal:
            return self.terminal[identity]
        self._identity(body)
        if (
            self.state != "cleaned"
            or self.sockets
            or self.native.state != "idle"
            or self.native.restart_required
        ):
            raise PoolError("RELEASE_UNCONFIRMED")
        result = dict(
            allocation_id=self.owner["allocation_id"],
            worker_group_id=self.owner["worker_group_id"],
            service_id=self.service_id,
            boot_epoch=self.process.boot_epoch,
            state="released",
        )
        self._remember(identity, result)
        self.process.release(self.service_id, self.owner["allocation_id"])
        self.owner, self.state = None, "idle"
        self.children.clear()
        self.child_receipts.clear()
        self.child_receipt_times.clear()
        return result

    async def expire(self):
        with self.process.lock:
            if self.owner is None or self.state in {"idle", "cleaned"}:
                return
            now = self.process.clock()
            expired = self.process.failed or now >= min(
                self.owner["lease_until"], self.owner["player_expires_at"]
            )
            expired = (
                expired
                or self.state == "reserved"
                and now >= self.owner["reserved_until"]
            )
        if expired:
            await self.begin_cleanup()

    def business(self, body, token):
        """Called before a WS is accepted. The lease watchdog owns cancellation."""
        with self.process.lock:
            self.authorize("business", body, token)
            self._identity(body)
            if (
                self.process.failed
                or self.native.restart_required
                or self.native.state != "connected"
                or self.state != "connected"
                or self.process.clock()
                >= min(self.owner["lease_until"], self.owner["player_expires_at"])
            ):
                raise PoolError("GROUP_NOT_READY")
            for name in ("session_id", "generation_id"):
                identifier(body.get(name))
            return dict(body)

    def child_cleaned(self, session_id, generation_id):
        with self.process.lock:
            active = self.children.get(session_id)
            if active is not None and active["generation_id"] != generation_id:
                raise PoolError("CHILD_GENERATION_MISMATCH")
            self.child_receipts[session_id] = dict(
                session_id=session_id, generation_id=generation_id, cleaned=True
            )
            self.child_receipt_times[session_id] = self.process.clock()
            self.children.pop(session_id, None)

    def attach_socket(self, claim, task):
        with self.process.lock:
            if self.children or claim["session_id"] in self.child_receipts:
                raise PoolError("BUSINESS_SESSION_ALREADY_ATTACHED")
            if len(self.child_receipts) >= 256:
                for key, at in list(self.child_receipt_times.items()):
                    if self.process.clock() - at > 180:
                        self.child_receipts.pop(key, None)
                        self.child_receipt_times.pop(key, None)
                if len(self.child_receipts) >= 256:
                    raise PoolError("BUSINESS_HISTORY_BUSY", 503)
            self.sockets.add(task)
            self.children[claim["session_id"]] = dict(
                generation_id=claim["generation_id"], state="connected"
            )

    def child_started(self, session_id, generation_id):
        with self.process.lock:
            if session_id in self.child_receipts:
                raise PoolError("BUSINESS_SESSION_ALREADY_COMPLETED")
            self.children[session_id] = dict(
                generation_id=generation_id, state="active"
            )

    def _remember(self, identity, receipt, *, recovery=None):
        now = self.process.clock()
        if identity not in self.terminal and len(self.terminal) >= 4096:
            for key, at in list(self.terminal_at.items()):
                if now - at > 180:
                    self.terminal.pop(key, None)
                    self.terminal_at.pop(key, None)
                    self.terminal_recovery.pop(key, None)
            if len(self.terminal) >= 4096:
                raise PoolError("TERMINAL_HISTORY_BUSY", 503)
        self.terminal[identity] = receipt
        self.terminal_at.setdefault(identity, now)
        if identity not in self.terminal_recovery:
            value = copy.deepcopy(recovery if recovery is not None else self.owner)
            if value is not None:
                value.pop("reservation", None)
                value["control_epoch"] = self.process.credentials.epoch
                value["boot_epoch"] = self.process.boot_epoch
            self.terminal_recovery[identity] = value
