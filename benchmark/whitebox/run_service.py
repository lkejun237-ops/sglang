#!/usr/bin/env python3
"""One role per process/container. Secrets come only from runtime environment."""

import os
import sys

MODULE = "sglang.multimodal_gen.runtime."
role = os.environ.get("WHITEBOX_ROLE")
if role == "coordinator":
    args = [
        MODULE + "entrypoints.realtime_coordinator_server",
        "--host=0.0.0.0",
        "--port=18081",
        "--backend=redis",
        "--redis-host=" + os.environ["REDIS_HOST"],
        "--redis-port=" + os.environ.get("REDIS_PORT", "6379"),
        "--redis-key-prefix="
        + os.environ.get("WHITEBOX_REDIS_PREFIX", "whitebox-playtest-v1"),
        "--ttl-s=30",
        "--worker-ttl-s=15",
        "--wait-timeout-s=10",
        "--denoiser-capacity-limit=" + os.environ.get("WHITEBOX_SLOTS", "2"),
    ]
    if os.environ.get("REDIS_USERNAME"):
        args.append("--redis-username=" + os.environ["REDIS_USERNAME"])
elif role == "gateway":
    args = [
        MODULE + "entrypoints.realtime_gateway_server",
        "--host=0.0.0.0",
        "--port=18080",
        "--coordinator-url=" + os.environ["COORDINATOR_URL"],
        "--model-revision="
        + os.environ.get("WHITEBOX_ROUTING_REVISION", "whitebox-mock"),
        "--vae-fingerprint=local",
        "--internal-output-url=ws://127.0.0.1:18080/v1/internal/realtime_output",
        "--lease-renew-interval-s=10",
        "--max-admission-waiters=16",
    ]
elif role == "mock-worker":
    args = [MODULE + "realtime.whitebox.mock_worker"]
elif role == "room":
    args = [MODULE + "entrypoints.whitebox_room_server"]
else:
    raise SystemExit("WHITEBOX_ROLE must be coordinator, gateway, room or mock-worker")
os.execv(sys.executable, [sys.executable, "-m", *args])
