# SPDX-License-Identifier: Apache-2.0
"""Idempotent, fenced group construction and per-role verified reclamation."""

import asyncio
import logging
import time
from uuid import uuid4

import httpx

from .contracts import ROLES, PoolError


class GroupController:
    def __init__(
        self,
        store,
        credentials,
        *,
        client=None,
        lease_seconds=30,
        build_timeout=120,
        cleanup_timeout=90,
        clock=time.time,
    ):
        self.store, self.credentials = store, credentials
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(build_timeout + 5, connect=5), trust_env=False
        )
        self.owns_client = client is None
        self.build_client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(build_timeout + 5, connect=5),
            limits=httpx.Limits(max_connections=48, max_keepalive_connections=24),
            trust_env=False,
        )
        self.cleanup_client = client or httpx.AsyncClient(
            timeout=5,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
            trust_env=False,
        )
        self.background_client = client or httpx.AsyncClient(
            timeout=6,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            trust_env=False,
        )
        self.build_slots = asyncio.Semaphore(16)
        self.recovery_tasks = {}
        self.retry_after = {}
        self.owned_records_clients = []
        self.identity, self.clock = uuid4().hex, clock
        self.lease_seconds, self.build_timeout, self.cleanup_timeout = (
            lease_seconds,
            build_timeout,
            cleanup_timeout,
        )
        self.tasks, self.close_tasks = {}, {}
        self.sweeper = None

    async def claim(self, admission_id, allocation_id):
        def claim(a):
            now = self.clock()
            if a["executor"] not in {None, self.identity} and a["executor_until"] > now:
                raise PoolError("EXECUTOR_BUSY", 503)
            if a["executor"] != self.identity:
                a["executor_revision"] = a.get("executor_revision", 0) + 1
            a.update(executor=self.identity, executor_until=now + self.lease_seconds)

        return await self.store.update(admission_id, allocation_id, claim)

    def body(self, a, role):
        body = dict(
            control_epoch=a["control_epoch"],
            boot_epoch=a["boot_epochs"][role],
            allocation_id=a["allocation_id"],
            worker_group_id=a["group_id"],
            admission_id=a["intent"]["admission_id"],
            user_id=a["intent"]["user_id"],
            player_id=a["intent"]["player_id"],
            parent_generation_id=a["intent"]["generation_id"],
            profile=a["intent"]["profile"],
            profile_limits=a["profile_limits"],
            admission_created_at=a["intent"]["created_at"],
            admitted_at=a["admitted_at"],
            player_expires_at=a["player_expires_at"],
            executor_revision=a["executor_revision"],
            executor_until=a["executor_until"],
            topology=a["topology"],
        )
        if a["intent"].get("operation") == "prefill":
            body["operation"] = "prefill"
        return body

    async def rpc(self, a, role, action, *, extra=None, timeout=None, lane=None):
        body = self.body(a, role)
        if extra:
            body.update(extra)
        deadline = min(a["executor_until"], self.clock() + self.lease_seconds)
        token = self.credentials.sign(action, body, expires_at=deadline)
        client = (
            self.cleanup_client
            if lane == "cleanup"
            else (
                self.build_client
                if action == "build"
                else self.background_client
                if action == "persistence"
                else self.client
            )
        )
        response = await client.post(
            a["topology"][role]["endpoints"]["http"].rstrip("/") + "/managed/" + action,
            json=body,
            headers={"Authorization": "Bearer " + token},
            timeout=timeout or self.build_timeout + 5,
        )
        if response.status_code >= 400:
            try:
                code = response.json().get("code", "ROLE_REQUEST_FAILED")
            except ValueError:
                code = "ROLE_REQUEST_FAILED"
            raise PoolError(code, response.status_code)
        return response.json()

    async def _change(self, a, **updates):
        def update(value):
            if (
                value["executor"] != self.identity
                or value["executor_until"] <= self.clock()
            ):
                raise PoolError("EXECUTOR_LEASE_LOST")
            if updates.get("state") in {"building", "bound"} and value["state"] not in {
                "held",
                "building",
            }:
                raise PoolError("PLAYER_SESSION_ENDED")
            value.update(updates)

        return await self.store.update(
            a["intent"]["admission_id"], a["allocation_id"], update
        )

    async def _role_state(self, a, role, state):
        def update(value):
            if (
                value["executor"] != self.identity
                or value["executor_until"] <= self.clock()
            ):
                raise PoolError("EXECUTOR_LEASE_LOST")
            if value["state"] not in {"held", "building"}:
                raise PoolError("PLAYER_SESSION_ENDED")
            value["roles"][role] = state

        return await self.store.update(
            a["intent"]["admission_id"], a["allocation_id"], update
        )

    async def admit(self, **request):
        a = await self.store.acquire(**request)
        if a["state"] == "bound":
            return await self.renew(a["intent"]["admission_id"], a["allocation_id"])
        if a["state"] not in {"held", "building"}:
            raise PoolError("PLAYER_SESSION_ENDED")
        key = a["allocation_id"]
        task = self.tasks.get(key)
        if task is None or task.done():
            if self.build_slots.locked():
                raise PoolError("ADMISSION_BUILD_BUSY", 503)
            await self.build_slots.acquire()
            task = self.tasks[key] = asyncio.create_task(self._build(a))

            def finished(done):
                self.build_slots.release()
                if self.tasks.get(key) is done:
                    self.tasks.pop(key, None)

            task.add_done_callback(finished)
        # HTTP disconnect cannot abandon a partially sent group operation.
        return await asyncio.shield(task)

    async def _keep_executor(self, a):
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            a = await self.claim(a["intent"]["admission_id"], a["allocation_id"])
            for role in ROLES:
                if a["roles"][role] in {"reserved", "build_sent", "bound"}:
                    await self.rpc(a, role, "renew", timeout=5)

    async def _build(self, a):
        await self.store.admissions.verify(a, self.store.pool_id)
        a = await self.claim(a["intent"]["admission_id"], a["allocation_id"])
        keeper = asyncio.create_task(self._keep_executor(a))
        try:
            async with asyncio.timeout(
                min(self.build_timeout, a["player_expires_at"] - self.clock())
            ):
                # Query the original managed owner on retries. A reservation
                # request is idempotent and carries the complete recovery record.
                async def prepare(role):
                    await self._role_state(a, role, "reserve_sent")
                    status = await self.rpc(a, role, "observe", timeout=5)
                    owner = status.get("owner")
                    if owner:
                        if owner.get("allocation_id") != a["allocation_id"]:
                            raise PoolError("ROLE_OWNER_CHANGED")
                        if status["state"] == "connected":
                            await self._role_state(a, role, "bound")
                            return
                        if status["state"] not in {"reserved", "connecting"}:
                            raise PoolError("ROLE_CLOSING")
                    else:
                        await self.rpc(a, role, "reserve", timeout=5)
                    await self._role_state(a, role, "reserved")

                prepared = await asyncio.gather(
                    *(prepare(r) for r in ROLES), return_exceptions=True
                )
                if any(isinstance(v, BaseException) for v in prepared):
                    raise PoolError("GROUP_RESERVATION_FAILED")
                a = await self._change(a, state="building")

                async def build(role):
                    await self._role_state(a, role, "build_sent")
                    result = await self.rpc(a, role, "build")
                    if (
                        result.get("state") != "connected"
                        or result.get("owner", {}).get("worker_group_id")
                        != a["group_id"]
                    ):
                        raise PoolError("GROUP_BUILD_UNCONFIRMED")
                    await self._role_state(a, role, "bound")

                results = await asyncio.gather(
                    *(build(r) for r in ROLES), return_exceptions=True
                )
                if any(isinstance(v, BaseException) for v in results):
                    raise PoolError("GROUP_BUILD_FAILED")
                if keeper.done():
                    keeper.result()
                if self.clock() >= a["player_expires_at"]:
                    raise PoolError("PLAYER_EXPIRED")
                return await self._change(a, state="bound", executor=None)
        except BaseException:
            keeper.cancel()
            await asyncio.gather(keeper, return_exceptions=True)
            # The complete attempted topology remains in the ledger on failure.
            await asyncio.shield(
                self.close(
                    a["intent"]["admission_id"], a["allocation_id"], sdk_cleanup=None
                )
            )
            raise
        finally:
            keeper.cancel()
            await asyncio.gather(keeper, return_exceptions=True)

    async def renew(self, admission_id, allocation_id):
        def renew(value):
            if value["state"] != "bound" or self.clock() >= min(
                value["player_expires_at"], value["executor_until"]
            ):
                raise PoolError("PLAYER_SESSION_ENDED")
            value["executor_until"] = self.clock() + self.lease_seconds

        a = await self.store.update(admission_id, allocation_id, renew)
        await self.store.admissions.verify(a, self.store.pool_id)
        results = await asyncio.gather(
            *(self.rpc(a, r, "renew", timeout=5) for r in ROLES), return_exceptions=True
        )
        failures = {
            role: {"code": getattr(value, "code", type(value).__name__)}
            if isinstance(value, BaseException)
            else {"state": value.get("state")}
            for role, value in zip(ROLES, results)
            if isinstance(value, BaseException) or value.get("state") != "connected"
        }
        if failures:
            await self.store.update(
                admission_id,
                allocation_id,
                lambda value: value.update(
                    state="quarantined",
                    renewal_failure={"at": self.clock(), "roles": failures},
                ),
            )
            raise PoolError("GROUP_UNAVAILABLE", 503)
        return a

    async def business(
        self, admission_id, allocation_id, *, role, session_id, generation_id
    ):
        if role not in {"encoder", "decoder"}:
            raise PoolError("INVALID_BUSINESS_ROLE", 422)
        a = await self.renew(admission_id, allocation_id)
        if a["intent"].get("operation") == "prefill" and role != "encoder":
            raise PoolError("INVALID_BUSINESS_ROLE", 422)
        body = self.body(a, role)
        body.update(session_id=session_id, generation_id=generation_id)
        return dict(
            body=body,
            token=self.credentials.sign(
                "business",
                body,
                expires_at=min(a["executor_until"], a["player_expires_at"]),
            ),
        )

    async def close(self, admission_id, allocation_id, *, sdk_cleanup=None):
        if sdk_cleanup is not None:
            await self.store.record_sdk_cleanup(
                admission_id, allocation_id, sdk_cleanup
            )
        task = self.close_tasks.get(allocation_id)
        if task is None or task.done():
            task = self.close_tasks[allocation_id] = asyncio.create_task(
                self._close(admission_id, allocation_id, sdk_cleanup=sdk_cleanup)
            )
            task.add_done_callback(
                lambda done: (
                    self.close_tasks.pop(allocation_id, None)
                    if self.close_tasks.get(allocation_id) is done
                    else None
                )
            )
        result = await asyncio.shield(task)
        if sdk_cleanup is not None and result["state"] == "awaiting_sdk":
            return await self.store.finish(admission_id, allocation_id, sdk_cleanup)
        return result

    async def _close(self, admission_id, allocation_id, *, sdk_cleanup=None):
        a = await self.store.get(admission_id)
        if a["state"] == "released":
            if a["allocation_id"] != allocation_id:
                raise PoolError("ALLOCATION_MISMATCH")
            await self.store.admissions.release(
                a["intent"], self.store.pool_id, allocation_id
            )
            return a

        def revoke(value):
            now = self.clock()
            if value["state"] == "released":
                return
            if (
                value["state"] == "closing"
                and value["executor"] not in {None, self.identity}
                and value["executor_until"] > now
            ):
                raise PoolError("EXECUTOR_BUSY", 503)
            # Cancellation may fence an in-progress builder immediately. This
            # is permission to destroy the original group, never to create one.
            value.setdefault("cleanup_deadline", now + self.cleanup_timeout)
            value.update(
                state="closing",
                executor=self.identity,
                executor_until=now + self.lease_seconds,
                executor_revision=value.get("executor_revision", 0) + 1,
            )

        a = await self.store.update(admission_id, allocation_id, revoke)
        if a["state"] == "released":
            return a
        keeper = asyncio.create_task(self._keep_cleanup_executor(a))

        async def close_role(role):
            current = await self.store.get(admission_id)
            if current["roles"][role] == "released":
                return
            deadline = a["cleanup_deadline"]
            first = True
            while first or self.clock() < deadline:
                first = False
                current = await self.store.get(admission_id)
                status = await self.rpc(
                    current, role, "observe", timeout=5, lane="cleanup"
                )
                if status.get("restart_required"):
                    await self.store.fail_process(
                        a["topology"][role]["worker_id"], a["boot_epochs"][role]
                    )
                owner = status.get("owner")
                if owner is None:
                    # Fence an undispatched OR late reservation before proving
                    # it absent. Native retains the original terminal identity.
                    await self.rpc(current, role, "destroy", timeout=5, lane="cleanup")
                    receipt = await self.rpc(
                        current, role, "finalize", timeout=5, lane="cleanup"
                    )
                    await self.store.release_role(
                        admission_id, allocation_id, role, receipt
                    )
                    return
                elif owner.get("allocation_id") != allocation_id:
                    raise PoolError("ROLE_OWNER_CHANGED")
                elif status["state"] == "cleaned":
                    receipt = await self.rpc(
                        current, role, "finalize", timeout=5, lane="cleanup"
                    )
                    await self.store.release_role(
                        admission_id, allocation_id, role, receipt
                    )
                    return
                await self.rpc(current, role, "destroy", timeout=5, lane="cleanup")
                await asyncio.sleep(0.05)
            raise PoolError("ROLE_CLEANUP_TIMEOUT")

        try:
            results = await asyncio.gather(
                *(close_role(r) for r in ROLES), return_exceptions=True
            )
            if any(isinstance(v, BaseException) for v in results):
                return await self._change(a, state="quarantined")
            current = await self.store.get(admission_id)
            receipt = current.get("sdk_cleanup_pending") or sdk_cleanup
            if receipt is not None or current["intent"].get("operation") == "prefill":
                return await self.store.finish(admission_id, allocation_id, receipt)
            return await self._change(a, state="awaiting_sdk", executor=None)
        finally:
            keeper.cancel()
            await asyncio.gather(keeper, return_exceptions=True)

    async def _keep_cleanup_executor(self, a):
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            await self.claim(a["intent"]["admission_id"], a["allocation_id"])

    async def drain_process(self, process_id, boot_epoch):
        value = await self.store.drain(process_id, boot_epoch)
        body = dict(
            control_epoch=self.store.epoch,
            boot_epoch=boot_epoch,
            executor_until=self.clock() + self.lease_seconds,
        )
        token = self.credentials.sign("drain", body, expires_at=body["executor_until"])
        results = await asyncio.gather(
            *(
                self.cleanup_client.post(
                    s["endpoints"]["http"].rstrip("/") + "/managed/drain",
                    json=body,
                    headers={"Authorization": "Bearer " + token},
                    timeout=5,
                )
                for s in value["manifest"]["services"]
            ),
            return_exceptions=True,
        )
        if any(isinstance(v, BaseException) or v.status_code != 200 for v in results):
            raise PoolError("DRAIN_UNCONFIRMED", 503)
        return value

    async def sweep_once(self):
        candidates = []
        for a in await self.store.active_assignments():
            now = self.clock()
            key = a["allocation_id"]
            if key in self.recovery_tasks or self.retry_after.get(key, 0) > now:
                continue
            expired = now >= min(
                a["player_expires_at"],
                a["executor_until"] or a["admitted_at"] + self.build_timeout,
            )
            closing = a["state"] in {"closing", "quarantined"} and (
                a["executor"] is None or a["executor_until"] <= now
            )
            if expired or closing or a.get("sdk_cleanup_pending"):
                candidates.append(a)
        candidates.sort(
            key=lambda a: (
                self.retry_after.get(a["allocation_id"], 0),
                a["admitted_at"],
            )
        )

        async def recover(a):
            key = a["allocation_id"]
            try:
                await self.close(a["intent"]["admission_id"], key)
            except Exception:
                pass  # Retained ledger + native deadline remain authoritative.
            finally:
                self.recovery_tasks.pop(key, None)
                self.retry_after[key] = self.clock() + 10

        for a in candidates[: max(0, 256 - len(self.recovery_tasks))]:
            key = a["allocation_id"]
            self.recovery_tasks[key] = asyncio.create_task(recover(a))
        # Forget backoff for completed groups, not their durable receipts.
        live = {a["allocation_id"] for a in await self.store.active_assignments()}
        self.retry_after = {k: v for k, v in self.retry_after.items() if k in live}

    async def run(self):
        while True:
            try:
                await self.sweep_once()
            except Exception as error:
                logging.getLogger(__name__).warning(
                    "Role reconciliation unavailable: %s", type(error).__name__
                )
            await asyncio.sleep(2)

    async def shutdown(self):
        if self.sweeper is not None:
            self.sweeper.cancel()
            await asyncio.gather(self.sweeper, return_exceptions=True)
        tasks = list(self.tasks.values()) + list(self.recovery_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        closes = list(self.close_tasks.values())
        if closes:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*closes, return_exceptions=True),
                    self.cleanup_timeout + 10,
                )
            except asyncio.TimeoutError:
                pass
        if self.owns_client:
            await asyncio.gather(
                *(
                    client.aclose()
                    for client in (
                        self.client,
                        self.build_client,
                        self.background_client,
                        self.cleanup_client,
                    )
                )
            )
        await asyncio.gather(
            *(client.aclose() for client in self.owned_records_clients)
        )
