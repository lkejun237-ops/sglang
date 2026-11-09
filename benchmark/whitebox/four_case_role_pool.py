# SPDX-License-Identifier: Apache-2.0
"""Explicit test-only role pool adapter; no discovery of deployment environments."""

import asyncio
import os
import time
from contextvars import ContextVar
from uuid import uuid4

import httpx

from sglang.multimodal_gen.runtime.realtime.role_pool.contracts import (
    Credentials,
    PoolError,
)
from sglang.multimodal_gen.runtime.realtime.role_pool.scheduler import GroupController
from sglang.multimodal_gen.runtime.realtime.role_pool.store import (
    MemoryRecords,
    RolePoolStore,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.disaggregated_binding import (
    DisaggregatedModelLeases,
)

current_pool = ContextVar("four_case_pool", default=None)


class SuiteRolePool:
    """Use an existing pool, or an isolated in-memory authority for three workers.

    Direct workers must use this authority's control epoch/signing secret and
    must not be registered with another Coordinator. All native ownership fences
    remain enforced. No drain, stop, process restart or routing write is issued.
    """

    def __init__(
        self,
        *,
        coordinator_url=None,
        workers=None,
        revision,
        compatibility,
        profile="480",
        network_domain="test-shanghai-copy",
        room_token_env="WHITEBOX_ROOM_TOKEN",
        signing_secret_env="ROLE_POOL_SIGNING_SECRET",
        control_epoch=None,
        client=None,
    ):
        self.supplied_client = client
        self.control_client = None
        self.workers = workers
        self.revision = revision
        self.control_epoch = control_epoch
        self.signing_secret_env = signing_secret_env
        self.config = dict(
            coordinator_url=coordinator_url or "http://in-memory.invalid",
            profile=profile,
            compatibility=compatibility,
            network_domain=network_domain,
            room_token_env=room_token_env,
            player_ttl_seconds=1800,
            startup_seconds=480,
        )
        self.controller = None
        self.identities = {}
        self.refresh_lock = asyncio.Lock()
        self.sequence = 0
        self.heartbeat_task = None

    async def __aenter__(self):
        self.client = self.supplied_client or httpx.AsyncClient(
            timeout=httpx.Timeout(485, connect=5), trust_env=False
        )
        try:
            if self.workers:
                if not self.control_epoch:
                    raise ValueError("Direct workers require --control-epoch")
                self.credentials = Credentials(
                    os.environ[self.signing_secret_env], self.control_epoch
                )
                store = RolePoolStore(
                    MemoryRecords(), "suite-" + uuid4().hex, self.control_epoch
                )
                await store.initialize()
                self.controller = GroupController(
                    store, self.credentials, client=self.client
                )
                from fastapi import FastAPI

                from sglang.multimodal_gen.runtime.realtime.role_pool.api import (
                    install_pool_routes,
                )

                app = FastAPI()
                self.room_token = uuid4().hex
                install_pool_routes(
                    app,
                    self.controller,
                    role_token=uuid4().hex,
                    room_token=self.room_token,
                )
                self.control_client = httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://in-memory.invalid",
                )
                for role, url in self.workers.items():
                    identity = await self.worker_rpc(
                        url, "identity", dict(control_epoch=self.control_epoch)
                    )
                    deployment = identity["deployment"]
                    for key in ("compatibility", "network_domain"):
                        if deployment.get(key) != self.config[key]:
                            raise ValueError(
                                f"Worker deployment mismatch: {role}/{key}"
                            )
                    service = identity["service"]
                    if service["role"] != role:
                        raise ValueError(f"Worker role mismatch: {role}")
                    if self.config["profile"] not in identity["profiles"]:
                        raise ValueError(f"Worker does not support profile: {role}")
                    self.identities[role] = (url, identity)
                    await store.register(
                        dict(
                            process_id=service["worker_id"],
                            boot_epoch=identity["boot_epoch"],
                            control_epoch=self.control_epoch,
                            pool_id=store.pool_id,
                            role=role,
                            compatibility=self.config["compatibility"],
                            network_domain=self.config["network_domain"],
                            capacity=identity["capacity"],
                            profiles=identity["profiles"],
                            profile_limits=identity["profile_limits"],
                            services=[service],
                            hardware_ids=deployment["hardware_ids"],
                        )
                    )
                await self.refresh()
                self.heartbeat_task = asyncio.create_task(self.report())
                self.controller.sweeper = asyncio.create_task(self.controller.run())
            else:
                # Fail before any admission when authentication is absent.
                os.environ[self.config["room_token_env"]]
                route = await self.client.get(
                    self.config["coordinator_url"].rstrip("/") + "/v1/capacity",
                    params={"model_revision": self.revision},
                )
                route.raise_for_status()
                if (
                    route.json().get("backend") != "role_pool"
                    or route.json().get("profile") != self.config["profile"]
                ):
                    raise ValueError(
                        "Routing revision does not select the requested role pool profile"
                    )
            self.token = current_pool.set(self)
            return self
        except BaseException:
            await self.shutdown()
            raise

    async def worker_rpc(self, url, action, body):
        token = self.credentials.sign(action, body, expires_at=time.time() + 10)
        response = await self.client.post(
            url.rstrip("/") + "/managed/" + action,
            json=body,
            headers={"Authorization": "Bearer " + token},
        )
        response.raise_for_status()
        return response.json()

    async def refresh(self):
        async with self.refresh_lock:
            await self._refresh()

    async def _refresh(self):
        self.sequence += 1
        for url, identity in self.identities.values():
            body = dict(
                control_epoch=self.control_epoch, boot_epoch=identity["boot_epoch"]
            )
            observation = await self.worker_rpc(url, "observe", body)
            if observation.get("restart_required") or observation.get("draining"):
                raise RuntimeError("Direct worker is unavailable")
            service = identity["service"]
            await self.controller.store.heartbeat(
                service["worker_id"],
                identity["boot_epoch"],
                sequence=self.sequence,
                state="ready",
                observations={service["service_id"]: observation},
            )

    async def report(self):
        while True:
            await asyncio.sleep(1)
            await self.refresh()

    async def capacity(self):
        body = {
            k: self.config[k] for k in ("profile", "compatibility", "network_domain")
        }
        if self.controller:
            if self.heartbeat_task and self.heartbeat_task.done():
                self.heartbeat_task.result()
            await self.refresh()
            return await self.controller.store.capacity(**body)
        return await self.control("capacity", body)

    async def wait_idle(self, timeout_s=20):
        deadline = time.monotonic() + timeout_s
        while True:
            capacity = await self.capacity()
            group = capacity["roles"]["model_group"]
            if (
                group["free_slots"] >= 1
                and not group["active_sessions"]
                and not group["draining_workers"]
            ):
                return dict(
                    status="passed", capacity=capacity, all_leases_released=True
                )
            if time.monotonic() >= deadline:
                raise RuntimeError("Test role pool is not idle")
            await asyncio.sleep(0.25)

    async def control(self, action, body):
        client = self.control_client or self.client
        token = (
            self.room_token
            if self.controller
            else os.environ[self.config["room_token_env"]]
        )
        response = await client.post(
            self.config["coordinator_url"].rstrip("/") + "/v1/role-pool/" + action,
            json=body,
            headers={"Authorization": "Bearer " + token},
        )
        if response.status_code >= 400:
            raise PoolError(
                response.json().get("code", "ROLE_POOL_REQUEST_FAILED"),
                response.status_code,
            )
        return response.json()

    def lease(self, spec, trace_id, segment):
        from websockets.asyncio.client import connect

        pool = self

        class Lease(DisaggregatedModelLeases):
            async def control(self, action, body):
                return await pool.control(action, body)

        return Lease(
            None,
            connect,
            config=self.config,
            user_id="four-case-" + trace_id,
            model_revision=self.revision,
            vae_fingerprint="local",
            trace_id=trace_id,
            profile_hint=f"{spec.width}x{spec.height}",
            max_bytes=64 << 20,
            strict_release=True,
            operation_path="/v1/realtime_video/bound",
            session_id=segment.get("model_session_id"),
            generation_id=segment.get("generation_id"),
        )

    async def shutdown(self):
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            await asyncio.gather(self.heartbeat_task, return_exceptions=True)
        if self.controller:
            await self.controller.shutdown()
        if self.control_client:
            await self.control_client.aclose()
        if self.supplied_client is None:
            await self.client.aclose()

    async def __aexit__(self, *exc):
        current_pool.reset(self.token)
        await self.shutdown()
