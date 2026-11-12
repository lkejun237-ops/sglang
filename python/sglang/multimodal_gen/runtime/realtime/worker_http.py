# SPDX-License-Identifier: Apache-2.0
"""Small HTTP host for role workers; model execution stays on its owner thread."""

import socket
import threading
import time


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
        self.role = role
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

    def status(self):
        return {
            "role": self.role,
            "ready": self.ready
            and not (
                getattr(getattr(self, "connections", None), "restart_required", False)
            )
            and not any(
                peer.restart_required for peer in getattr(self, "connection_peers", [])
            ),
        }

    def close(self):
        self.ready = False
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.socket.close()
        if self.thread.is_alive():
            raise RuntimeError("Worker HTTP shutdown did not finish")
