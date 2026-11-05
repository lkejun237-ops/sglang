# SPDX-License-Identifier: Apache-2.0
"""Per-service connection admission and HTTP lifecycle, independent of models.

Transport callbacks own their sockets, groups and execution threads. A successful
build publishes the topology only after all local links are ready; failed builds
remain reserved until rollback finishes. No session admission is managed here.
"""

import asyncio
import inspect
import logging
import secrets
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from urllib.parse import urlsplit


class ConnectionRejected(Exception):
    def __init__(self, code, message, status=409):
        self.code, self.message, self.status = code, message, status
        super().__init__(message)


class ServiceConnections:
    RESERVATION_SECONDS = 10

    def __init__(
        self,
        service,
        *,
        ready,
        build,
        destroy,
        rollback,
        quiesce=None,
        clock=time.monotonic,
        cleanup_timeout=None,
    ):
        self.service = deepcopy(service)
        self.ready = ready
        self.build_callback, self.destroy_callback = build, destroy
        self.rollback_callback = rollback
        self.quiesce_callback = quiesce
        self.clock = clock
        self.cleanup_timeout = cleanup_timeout
        self.lock = threading.RLock()
        self.state = "idle"
        self.token = None
        self.deadline = 0
        self.group = self.topology = None
        self.restart_required = False
        self.last_destroyed = None
        self.operation = None
        self.loop = None

    def query(self, *, reserve=True):
        with self.lock:
            if self.restart_required or not self.ready():
                return dict(
                    state="unavailable",
                    restart_required=self.restart_required,
                    code="WORKER_RESTART_REQUIRED"
                    if self.restart_required
                    else "WORKER_NOT_READY",
                    message="The worker must be restarted."
                    if self.restart_required
                    else "The worker is not ready.",
                ), 503
            if self.state == "reserved" and self.clock() >= self.deadline:
                self.state, self.token = "idle", None
            if self.state == "idle":
                if not reserve:
                    return dict(state="available"), 200
                self.state = "reserved"
                self.token = secrets.token_urlsafe(32)
                self.deadline = self.clock() + self.RESERVATION_SECONDS
                return dict(
                    state="available",
                    reservation_token=self.token,
                    expires_in_seconds=self.RESERVATION_SECONDS,
                    service=deepcopy(self.service),
                ), 200
            if self.state == "connected":
                return dict(
                    state="connected",
                    worker_group_id=self.group,
                    topology=deepcopy(self.topology),
                ), 200
            return dict(state="reserved"), 200

    def _validate(self, payload):
        if not isinstance(payload, dict):
            raise ConnectionRejected(
                "INVALID_REQUEST", "The request must be an object.", 422
            )
        group, token, topology = (
            payload.get(k) for k in ("worker_group_id", "reservation_token", "topology")
        )
        if not isinstance(group, str) or not group or len(group) > 256:
            raise ConnectionRejected(
                "INVALID_GROUP", "A valid worker_group_id is required.", 422
            )
        if not isinstance(token, str) or not token:
            raise ConnectionRejected(
                "INVALID_TOKEN", "A reservation token is required.", 422
            )
        if not isinstance(topology, dict) or set(topology) != {
            "encoder",
            "dit",
            "decoder",
        }:
            raise ConnectionRejected(
                "INVALID_TOPOLOGY", "The complete worker topology is required.", 422
            )
        required = dict(
            encoder={"http", "input_ws", "rendezvous"},
            dit={"http", "zmq", "rendezvous"},
            decoder={"http", "zmq", "output_ws"},
        )
        identities = set()
        for role, node in topology.items():
            if (
                not isinstance(node, dict)
                or node.get("role") != role
                or any(
                    not isinstance(node.get(k), str) or not node[k]
                    for k in ("worker_id", "service_id")
                )
            ):
                raise ConnectionRejected(
                    "INVALID_TOPOLOGY", "Invalid worker identity.", 422
                )
            identity = node["worker_id"], node["service_id"]
            if identity in identities:
                raise ConnectionRejected(
                    "INVALID_TOPOLOGY",
                    "Worker service identities must be distinct.",
                    422,
                )
            identities.add(identity)
            endpoints = node.get("endpoints")
            if not isinstance(endpoints, dict) or set(endpoints) != required[role]:
                raise ConnectionRejected(
                    "INVALID_TOPOLOGY", "Invalid worker endpoints.", 422
                )
            for name, address in endpoints.items():
                try:
                    uri = urlsplit(address)
                    schemes = (
                        {"http", "https"}
                        if name == "http"
                        else ({"ws", "wss"} if name.endswith("ws") else {"tcp"})
                    )
                    if (
                        uri.scheme not in schemes
                        or not uri.hostname
                        or (uri.port is None or uri.port < 1)
                        or uri.username
                        or uri.password
                    ):
                        raise ValueError()
                except (ValueError, TypeError, AttributeError):
                    raise ConnectionRejected(
                        "INVALID_ENDPOINT", "Invalid endpoint address.", 422
                    ) from None
        if topology[self.service["role"]] != self.service:
            raise ConnectionRejected(
                "SERVICE_MISMATCH", "The topology does not match this service."
            )
        return group, token, deepcopy(topology)

    def cleanup_guard(self):
        if self.cleanup_timeout is None:
            return nullcontext()
        from .cleanup_watchdog import CleanupWatchdog

        return CleanupWatchdog(self.cleanup_timeout)

    def quarantine(self):
        """Reject all reuse before attempting potentially blocking cleanup."""
        with self.lock:
            self.restart_required, self.state = True, "failed"

    async def _call(self, callback, *args):
        if inspect.iscoroutinefunction(callback):
            return await callback(*args)
        result = await asyncio.to_thread(callback, *args)
        if inspect.isawaitable(result):
            return await result
        return result

    async def build(self, payload):
        self.loop = asyncio.get_running_loop()
        group, token, topology = self._validate(payload)
        with self.lock:
            matches = (
                self.group == group
                and self.topology == topology
                and self.token is not None
                and secrets.compare_digest(self.token, token)
            )
            if self.state == "connected" and matches:
                return dict(success=True, state="connected", worker_group_id=group), 200
            if self.state == "connecting" and matches:
                operation = self.operation
            else:
                if self.restart_required or not self.ready():
                    raise ConnectionRejected(
                        "WORKER_UNAVAILABLE", "The worker is unavailable.", 503
                    )
                if (
                    self.state != "reserved"
                    or self.clock() >= self.deadline
                    or self.token is None
                    or not secrets.compare_digest(self.token, token)
                ):
                    raise ConnectionRejected(
                        "RESERVATION_INVALID", "The reservation is invalid or expired."
                    )
                self.state, self.group, self.topology = "connecting", group, topology
                operation = self.operation = asyncio.create_task(
                    self._build(group, topology)
                )
        # Client disconnect must not cancel a partially constructed communicator.
        return await asyncio.shield(operation)

    async def _build(self, group, topology):
        try:
            await self._call(self.build_callback, group, topology)
            with self.lock:
                if self.restart_required:
                    raise RuntimeError("Worker failed during connection construction")
                self.state = "connected"
            return dict(success=True, state="connected", worker_group_id=group), 200
        except Exception:
            logging.exception("Worker group construction failed")
            cleaned = False
            try:
                with self.cleanup_guard():
                    await self._call(self.rollback_callback)
                cleaned = True
            except Exception:
                logging.exception("Worker group rollback failed")
            with self.lock:
                self.restart_required = self.restart_required or not cleaned
                reusable = not self.restart_required
                self.state = "idle" if reusable else "failed"
                if reusable:
                    self.last_destroyed = group
                self.token = self.group = self.topology = None
            return dict(
                success=False,
                state="idle" if reusable else "unavailable",
                worker_group_id=group,
                restart_required=not reusable,
                code="CONNECTION_FAILED",
                message="Failed to establish the worker connections.",
            ), 503

    async def quiesce(self, payload):
        """Finish the owner session while keeping all peer transports alive."""
        group = payload.get("worker_group_id") if isinstance(payload, dict) else None
        if not isinstance(group, str) or not group:
            raise ConnectionRejected(
                "INVALID_GROUP", "A worker_group_id is required.", 422
            )
        with self.lock:
            if self.group != group or self.state not in (
                "connected",
                "quiescing",
                "quiesced",
            ):
                raise ConnectionRejected(
                    "GROUP_MISMATCH", "The connection group does not match."
                )
            if self.quiesce_callback is None:
                raise ConnectionRejected(
                    "QUIESCE_UNSUPPORTED",
                    "This worker cannot finish the session before closing.",
                    501,
                )
            if self.state == "quiesced":
                return dict(success=True, state="quiesced", worker_group_id=group), 200
            if self.state == "connected":
                self.state = "quiescing"
                self.operation = asyncio.create_task(self._quiesce(group))
            operation = self.operation
        return await asyncio.shield(operation)

    async def _quiesce(self, group):
        try:
            with self.cleanup_guard():
                await self._call(self.quiesce_callback)
            with self.lock:
                if self.restart_required:
                    raise RuntimeError("Worker failed while finishing the session")
                self.state = "quiesced"
        except Exception:
            logging.exception("Worker session quiesce failed")
            with self.lock:
                self.state, self.restart_required = "failed", True
            return dict(
                success=False,
                state="unavailable",
                worker_group_id=group,
                restart_required=True,
                code="QUIESCE_FAILED",
                message="Failed to finish the current session.",
            ), 503
        return dict(success=True, state="quiesced", worker_group_id=group), 200

    async def destroy(self, payload):
        group = payload.get("worker_group_id") if isinstance(payload, dict) else None
        if not isinstance(group, str) or not group:
            raise ConnectionRejected(
                "INVALID_GROUP", "A worker_group_id is required.", 422
            )
        with self.lock:
            if self.state == "idle" and group == self.last_destroyed:
                return dict(
                    success=True, state="disconnected", worker_group_id=group
                ), 200
            if self.group != group or self.state not in (
                "connected",
                "quiescing",
                "quiesced",
                "draining",
            ):
                raise ConnectionRejected(
                    "GROUP_MISMATCH", "The connection group does not match."
                )
            pending_quiesce = self.operation if self.state == "quiescing" else None
            if self.state in ("connected", "quiesced"):
                self.state = "draining"
                self.operation = asyncio.create_task(self._destroy(group))
        if pending_quiesce is not None:
            result = await asyncio.shield(pending_quiesce)
            if result[1] != 200:
                return result
            return await self.destroy(payload)
        return dict(success=True, state="cleaning", worker_group_id=group), 202

    async def _destroy(self, group):
        try:
            with self.cleanup_guard():
                await self._call(self.destroy_callback)
        except Exception:
            logging.exception("Worker group destruction failed")
            with self.lock:
                self.state, self.restart_required = "failed", True
            return dict(
                success=False,
                state="unavailable",
                worker_group_id=group,
                restart_required=True,
                code="DISCONNECT_FAILED",
                message="Failed to close the worker connections.",
            ), 503
        with self.lock:
            if self.restart_required:
                self.state = "failed"
                return dict(
                    success=False,
                    state="unavailable",
                    worker_group_id=group,
                    restart_required=True,
                    code="WORKER_RESTART_REQUIRED",
                    message="The worker must be restarted.",
                ), 503
            self.last_destroyed = group
            self.state = "idle"
            self.token = self.group = self.topology = None
        return dict(success=True, state="disconnected", worker_group_id=group), 200

    def failed(self):
        """Called from a transport/model owner; cleanup stays off that thread."""
        with self.lock:
            if self.state in ("idle", "failed"):
                return
            if self.state in ("draining", "connecting", "quiescing"):
                self.state, self.restart_required = "failed", True
                return
            self.state, self.restart_required = "failed", True
        if self.loop is None or self.loop.is_closed():
            with self.lock:
                self.restart_required, self.state = True, "failed"
            return
        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self._failed()))

    async def _failed(self):
        self.quarantine()
        try:
            with self.cleanup_guard():
                await self._call(self.rollback_callback)
        except Exception:
            logging.exception("Failed connection cleanup requires a worker restart")
        # Even successful host cleanup does not prove a failed CUDA/NCCL
        # context reusable. Keep the worker quarantined until replacement.

    def install(self, app):
        app.state.worker_connections = self
        install_connection_routes(app)


def install_connection_routes(app):
    """Install once, including startup-unavailable replies before model load."""
    if getattr(app.state, "worker_connection_routes_installed", False):
        return
    from fastapi import Request
    from fastapi.responses import JSONResponse

    def response(value, status):
        return JSONResponse(
            value, status_code=status, headers={"Cache-Control": "no-store"}
        )

    def unavailable():
        return response(
            dict(
                state="unavailable",
                restart_required=False,
                code="WORKER_NOT_READY",
                message="The worker is not ready.",
            ),
            503,
        )

    @app.get("/get_connection_status")
    async def query(reserve: bool = True):
        controller = getattr(app.state, "worker_connections", None)
        return (
            unavailable()
            if controller is None
            else response(*controller.query(reserve=reserve))
        )

    async def command(request, operation):
        controller = getattr(app.state, "worker_connections", None)
        if controller is None:
            return unavailable()
        try:
            payload = await request.json()
        except ValueError:
            return response(
                dict(
                    success=False, code="INVALID_JSON", message="Invalid JSON request."
                ),
                422,
            )
        try:
            return response(*await getattr(controller, operation)(payload))
        except ConnectionRejected as exc:
            return response(
                dict(success=False, code=exc.code, message=exc.message), exc.status
            )

    @app.post("/build_group")
    async def build(request: Request):
        return await command(request, "build")

    @app.post("/destroy_group")
    async def destroy(request: Request):
        return await command(request, "destroy")

    @app.post("/quiesce_group")
    async def quiesce(request: Request):
        return await command(request, "quiesce")

    app.state.worker_connection_routes_installed = True


def endpoint(address):
    uri = urlsplit(address)
    return uri.hostname, uri.port


def service_descriptor(role, worker_id, index, host, **ports):
    """Advertised endpoint names follow the public contract, not CLI names."""
    if not host or host in ("0.0.0.0", "::"):
        raise ValueError("An accessible advertised host is required")
    host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    paths = dict(
        input_ws="/v1/realtime_video/generate", output_ws="/v1/realtime_video/output"
    )
    return dict(
        worker_id=worker_id,
        service_id=f"{worker_id}/{index}",
        role=role,
        endpoints={
            name: f"{'http' if name == 'http' else 'ws' if name.endswith('ws') else 'tcp'}://{host}:{port}{paths.get(name, '')}"
            for name, port in ports.items()
        },
    )
