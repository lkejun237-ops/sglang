# SPDX-License-Identifier: Apache-2.0
"""Room lease adapter; the established ModelBinding owns its logical parent."""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx

from ..coordinator import SessionAssignment, WorkerSlot
from ..role_pool.contracts import PoolError
from ..role_pool.transport import PooledBoundSocket
from .interactive import ModelLeases
from .render_contract import RenderError


class DisaggregatedModelLeases(ModelLeases):
    supports_condition_barrier = True

    def __init__(self, *args, config=None, profile_hint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or json.loads(
            Path(os.environ["WHITEBOX_ROLE_POOL_CONFIG"]).read_text()
        )
        routes = self.config.get("routes")
        if routes is not None:
            key = self.request["model_revision"] + ":" + str(profile_hint)
            if profile_hint is None:
                choices = [
                    name
                    for name in routes
                    if name.startswith(self.request["model_revision"] + ":")
                ]
                if len(choices) == 1:
                    key = choices[0]
            if key not in routes:
                raise RenderError("CAPABILITY_UNAVAILABLE")
            self.config = {**self.config, **routes[key]}
        self.admission_id = uuid4().hex
        self.admission_created_at = time.time()
        self.player_id = self.session_id or uuid4().hex
        self.generation_id = self.generation_id or uuid4().hex
        self.group = None
        self.model_only = self.operation_path == "/v1/realtime_video/prefill"
        self.bound_socket = None
        self.close_started_at = None
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                self.config.get("startup_seconds", 480) + 10, connect=5
            ),
            trust_env=False,
        )

    async def control(self, action, body):
        response = await self.client.post(
            self.config["coordinator_url"].rstrip("/") + "/v1/role-pool/" + action,
            json=body,
            headers={
                "Authorization": "Bearer " + os.environ[self.config["room_token_env"]]
            },
        )
        if response.status_code >= 400:
            raise PoolError(
                response.json().get("code", "MODEL_UNAVAILABLE"), response.status_code
            )
        return response.json()

    def _assignment(self, a):
        service = a["topology"]["dit"]
        slot = WorkerSlot(
            worker_id=service["worker_id"],
            role="denoiser",
            endpoint=a["topology"]["encoder"]["endpoints"]["input_ws"],
            az=a["intent"]["network_domain"],
            slot_index=0,
            model_revision=self.request["model_revision"],
            vae_fingerprint="local",
            worker_epoch=a["boot_epochs"]["dit"],
        )
        return SessionAssignment(
            user_id=self.request["user_id"],
            session_id=self.player_id,
            generation_id=self.generation_id or uuid4().hex,
            token=a["allocation_id"],
            expires_at=a["executor_until"],
            denoiser=slot,
            vae=None,
            admitted_at=a["admitted_at"],
            player_expires_at=a["player_expires_at"],
        )

    async def _release_group(self):
        try:
            if self.group is None:
                self.group = await self.control(
                    "get", {"admission_id": self.admission_id}
                )
                self._active_assignment = self._assignment(self.group)

            if self.model_only:
                result = await self.control(
                    "close",
                    dict(
                        admission_id=self.admission_id,
                        allocation_id=self.group["allocation_id"],
                    ),
                )
                if result["state"] != "released":
                    raise RenderError("RELEASE_UNKNOWN")
                self.resource_state = "released"
                return

            async def sdk():
                if self.sdk_cleanup_provider is None:
                    raise RenderError("RELEASE_UNKNOWN")
                from ..player_lifecycle import validate_sdk_cleanup

                native = await self.sdk_cleanup_provider(self._active_assignment)
                receipt = {
                    **validate_sdk_cleanup(self._active_assignment, native),
                    "player_id": self.player_id,
                }
                # Submit the receipt as soon as SDK retirement is proven. The
                # Coordinator persists it before waiting for GPU teardown.
                return await self.control("close", {**identity, "sdk_cleanup": receipt})

            identity = dict(
                admission_id=self.admission_id,
                allocation_id=self.group["allocation_id"],
            )
            sdk_task = asyncio.create_task(sdk())
            model_task = asyncio.create_task(self.control("close", identity))
            try:
                if self.close_started_at is None:
                    self.close_started_at = time.monotonic()
                remaining = max(
                    0.1,
                    self.cleanup_timeout_s - (time.monotonic() - self.close_started_at),
                )
                async with asyncio.timeout(remaining):
                    results = await asyncio.gather(
                        sdk_task, model_task, return_exceptions=True
                    )
                    if any(isinstance(v, BaseException) for v in results):
                        raise RenderError("RELEASE_UNKNOWN")
                    result = results[0]
                    if result["state"] != "released":
                        raise RenderError("RELEASE_UNKNOWN")
                    self.resource_state = "released"
            finally:
                for task in (sdk_task, model_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(sdk_task, model_task, return_exceptions=True)
        except PoolError as exc:
            if exc.code == "ADMISSION_NOT_FOUND" and self.resource_state == "none":
                return
            self.resource_state = "release_unknown"
            raise
        except BaseException:
            self.resource_state = "release_unknown"
            raise

    async def retry_release(self):
        if self.resource_state not in {"none", "released"}:
            await self._release_group()

    async def release_compute(self):
        if self.model_only:
            if self._compute_released:
                return
            if self._active_renewal is not None:
                self._active_renewal.cancel()
                await asyncio.gather(self._active_renewal, return_exceptions=True)
            self._compute_released = True
            if self.bound_socket is not None:
                await self.bound_socket.close()
        await self._release_group()
        self._compute_released = True

    @asynccontextmanager
    async def open(self):
        if self.operation_path not in {
            "/v1/realtime_video/bound",
            "/v1/realtime_video/prefill",
        }:
            raise RenderError("CAPABILITY_UNAVAILABLE")
        socket = renew = None
        try:
            request = dict(
                admission_id=self.admission_id,
                created_at=self.admission_created_at,
                user_id=self.request["user_id"],
                player_id=self.player_id,
                generation_id=self.generation_id,
                profile=self.config["profile"],
                compatibility=self.config["compatibility"],
                network_domain=self.config["network_domain"],
                ttl=min(540, self.config["player_ttl_seconds"])
                if self.model_only
                else self.config["player_ttl_seconds"],
            )
            if self.model_only:
                request["operation"] = "prefill"

            async def admit_group():
                deadline = time.monotonic() + self.config.get("startup_seconds", 480)
                while True:
                    try:
                        return await self.control("admit", request)
                    except PoolError as error:
                        if (
                            error.code
                            not in {
                                "ADMISSION_BUILD_BUSY",
                                "EXECUTOR_BUSY",
                                "CONTROL_STORE_BUSY",
                                "CAPACITY_BUSY",
                            }
                            or time.monotonic() >= deadline
                        ):
                            raise
                        await asyncio.sleep(0.2)

            admit = asyncio.create_task(admit_group())
            try:
                self.group = await asyncio.shield(admit)
            except asyncio.CancelledError:
                await asyncio.shield(
                    self.control(
                        "cancel",
                        {
                            "admission_id": self.admission_id,
                            "user_id": self.request["user_id"],
                        },
                    )
                )
                try:
                    self.group = await admit
                except Exception:
                    pass
                raise
            self.resource_state = "reserved"
            self._active_assignment = self._assignment(self.group)
            socket = PooledBoundSocket(
                self.control,
                self.group,
                connect=self.connect,
                max_bytes=self.max_bytes,
                receipt_timeout=self.cleanup_timeout_s,
                on_close=lambda: setattr(
                    self, "close_started_at", self.close_started_at or time.monotonic()
                ),
            )
            self.bound_socket = socket

            async def keep():
                while True:
                    await asyncio.sleep(min(self.renew_seconds, 5))
                    try:
                        self.group = await self.control(
                            "renew",
                            dict(
                                admission_id=self.admission_id,
                                allocation_id=self.group["allocation_id"],
                            ),
                        )
                    except Exception:
                        await socket.close()
                        raise

            renew = asyncio.create_task(keep())
            self._active_renewal = renew
            yield socket, self._active_assignment
        finally:
            if renew is not None:
                renew.cancel()
                await asyncio.gather(renew, return_exceptions=True)
            if socket is not None:
                await socket.close()
            cleanup = asyncio.create_task(self._release_group())
            try:
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                await cleanup
            finally:
                # A retry needs the HTTP client; close only after positive final release.
                if self.resource_state == "released":
                    await self.client.aclose()
