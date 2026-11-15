# SPDX-License-Identifier: Apache-2.0
"""Small HTTP host for role workers; model execution stays on its owner thread."""

import os
import socket
import threading
import time
import weakref

_WARMUP_LOCK = threading.Lock()
_ACTIVE_SERVERS: "weakref.WeakSet[WorkerHTTPServer]" = weakref.WeakSet()
_PUBLISHED_WARMUP_BY_ROLE: dict[str, dict] = {}


def publish_role_warmup(role: str, info: dict) -> None:
    """Automatically bind warmup metadata to active (and subsequent) WorkerHTTPServers for ``role``."""
    if not isinstance(info, dict) or not info:
        return
    snapshot = dict(info)
    with _WARMUP_LOCK:
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            _PUBLISHED_WARMUP_BY_ROLE[role] = dict(snapshot)
        targets = [srv for srv in _ACTIVE_SERVERS if srv.role == role]
    for srv in targets:
        srv.record_warmup(snapshot)


def install_health_routes(app, state):
    from fastapi.responses import JSONResponse, Response

    from .critical_path_metrics import prometheus_content_type, prometheus_latest

    @app.get("/metrics")
    async def metrics():
        return Response(prometheus_latest(), media_type=prometheus_content_type())

    @app.get("/liveness")
    async def liveness():
        return {"status": "ok"}

    @app.get("/health")
    async def health():
        status = state()
        return JSONResponse(status, status_code=200 if status["ready"] else 503)


class WorkerHTTPServer:
    def __init__(self, role, host, port, *, install=None, app=None, **options):
        import uvicorn
        from fastapi import FastAPI

        self.ready = False
        self.warmup = None
        self.role = role
        with _WARMUP_LOCK:
            if not os.environ.get("PYTEST_CURRENT_TEST"):
                prior = _PUBLISHED_WARMUP_BY_ROLE.get(role)
                if prior is not None:
                    self.warmup = dict(prior)
            _ACTIVE_SERVERS.add(self)
        self.app = FastAPI() if app is None else app
        if app is None:
            install_health_routes(self.app, self.status)
            from .worker_connections import install_connection_routes

            install_connection_routes(self.app)
        if install is not None:
            install(self.app)
        # Bind in the caller: a port conflict must fail worker startup.
        self.socket = socket.socket()
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((host, port))
            self.socket.listen(128)
            self.server = uvicorn.Server(
                uvicorn.Config(
                    self.app,
                    log_level="warning",
                    lifespan="off" if app is not None else "auto",
                    **({"ws_max_queue": 1} | options),
                )
            )
            self.thread = threading.Thread(
                target=self.server.run,
                kwargs={"sockets": [self.socket]},
                name=f"{role}-http",
                daemon=True,
            )
            self.thread.start()
            deadline = time.monotonic() + 10
            while not self.server.started:
                if not self.thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("Worker HTTP startup failed")
                time.sleep(0.01)
        except BaseException:
            if hasattr(self, "server"):
                self.server.should_exit = True
            self.socket.close()
            raise

    def record_warmup(self, info):
        if isinstance(info, dict) and info:
            self.warmup = dict(info)

    def status(self):
        ready = (
            self.ready
            and not (
                getattr(getattr(self, "connections", None), "restart_required", False)
            )
            and not any(
                peer.restart_required for peer in getattr(self, "connection_peers", [])
            )
        )
        payload = {
            "role": self.role,
            "ready": ready,
        }
        if self.warmup is not None:
            payload["warmed_up"] = bool(self.warmup.get("warmed_up", True))
            payload["warmup"] = dict(self.warmup)
        return payload

    def close(self):
        self.ready = False
        with _WARMUP_LOCK:
            _ACTIVE_SERVERS.discard(self)
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.socket.close()
        if self.thread.is_alive():
            raise RuntimeError("Worker HTTP shutdown did not finish")
