# SPDX-License-Identifier: Apache-2.0
"""One batched process reporter, independent of the model execution thread."""

import asyncio
import atexit
import json
import logging
import os
import threading
from pathlib import Path

import httpx

from .worker import ManagedService, ProcessAuthority

_REGISTRY = {}
_LOCK = threading.RLock()


class ProcessReporter:
    def __init__(self, config, process_id, role):
        self.config, self.process_id, self.role = config, process_id, role
        self.role_token = os.environ[config["role_token_env"]]
        self.process = ProcessAuthority(
            capacity=config["capacity"],
            profiles=config["profiles"],
            profile_limits=config["profile_limits"],
            control_epoch=config["control_epoch"],
            secret=os.environ[config["signing_secret_env"]],
        )
        self.process.deployment = {
            k: config[k]
            for k in ("pool_id", "compatibility", "network_domain", "hardware_ids")
        }
        self.stop = threading.Event()
        self.sequence = 0
        self.last_error = None
        self.thread = threading.Thread(
            target=self.run, name="role-pool-reporter", daemon=True
        )
        self.thread.start()
        atexit.register(self.stop.set)

    def declaration(self):
        with self.process.lock:
            services = [m.native.service for m in self.process.services.values()]
        if len(services) != self.config["service_count"]:
            return None
        return dict(
            process_id=self.process_id,
            boot_epoch=self.process.boot_epoch,
            control_epoch=self.config["control_epoch"],
            pool_id=self.config["pool_id"],
            role=self.role,
            compatibility=self.config["compatibility"],
            network_domain=self.config["network_domain"],
            capacity=self.config["capacity"],
            profiles=self.config["profiles"],
            profile_limits=self.config["profile_limits"],
            hardware_ids=self.config["hardware_ids"],
            services=sorted(services, key=lambda s: s["service_id"]),
        )

    def run(self):
        with httpx.Client(timeout=3, trust_env=False) as client:
            base = self.config["coordinator_url"].rstrip("/") + "/v1/role-pool/"
            headers = {"Authorization": "Bearer " + self.role_token}
            registered = False
            while not self.stop.is_set():
                try:
                    declared = self.declaration()
                    if declared is not None:
                        if not registered:
                            response = client.post(
                                base + "register", json=declared, headers=headers
                            )
                            response.raise_for_status()
                            registered = True
                        with self.process.lock:
                            managers = list(self.process.services.values())
                        state = (
                            "draining"
                            if self.process.draining
                            else "ready"
                            if all(m.native.ready() for m in managers)
                            else "warming"
                        )
                        if self.process.failed or any(
                            m.native.restart_required for m in managers
                        ):
                            state = "failed"
                        self.sequence += 1
                        response = client.post(
                            base + "heartbeat",
                            json=dict(
                                process_id=self.process_id,
                                boot_epoch=self.process.boot_epoch,
                                sequence=self.sequence,
                                state=state,
                                observations={
                                    m.service_id: m.snapshot(include_children=False)
                                    for m in managers
                                },
                            ),
                            headers=headers,
                        )
                        response.raise_for_status()
                        # Watchdog must also run when control service is down.
                except Exception as error:
                    reason = type(error).__name__
                    if self.last_error != reason:
                        logging.getLogger(__name__).warning(
                            "Role registration/heartbeat unavailable: %s", reason
                        )
                    self.last_error = reason
                    # Missing heartbeats automatically stop new allocation. No
                    # credentials, data bodies, restart or process signals here.
                else:
                    self.last_error = None
                with self.process.lock:
                    managers = list(self.process.services.values())
                for managed in managers:
                    if managed.loop is not None and not managed.loop.is_closed():
                        asyncio.run_coroutine_threadsafe(managed.expire(), managed.loop)
                self.stop.wait(min(2, float(self.config.get("heartbeat_seconds", 2))))


def attach_managed_service_config(native, config, *, cache_key=None):
    if native.service["role"] == "dit" and int(os.getenv("RANK", "0")) != 0:
        # SP ranks are one DiT service. Only the rank owning HTTP may register
        # the shared descriptor/boot epoch; followers execute collectives only.
        return None
    expected_count = getattr(
        native,
        "process_service_count",
        config.get("service_count") if isinstance(config, dict) else 1,
    )
    validate_worker_config_dict(config, native.service["role"], expected_count)
    key = (
        cache_key or json.dumps(config, sort_keys=True, separators=(",", ":")),
        native.service["worker_id"],
    )
    with _LOCK:
        reporter = _REGISTRY.get(key)
        if reporter is None:
            reporter = _REGISTRY[key] = ProcessReporter(
                config, native.service["worker_id"], native.service["role"]
            )
        managed = ManagedService(native, reporter.process)
    return managed


def attach_managed_service(native):
    path = os.getenv("SGLANG_ROLE_POOL_WORKER_CONFIG")
    if not path:
        return None
    if native.service["role"] == "dit" and int(os.getenv("RANK", "0")) != 0:
        return None
    config = json.loads(Path(path).read_text())
    key = (str(Path(path).resolve()), native.service["worker_id"])
    with _LOCK:
        reporter = _REGISTRY.get(key)
        if reporter is None:
            reporter = _REGISTRY[key] = ProcessReporter(
                config, native.service["worker_id"], native.service["role"]
            )
        managed = ManagedService(native, reporter.process)
    return managed


def validate_worker_config_dict(
    config, role, service_count, *, static_connection=False
):
    if (
        not isinstance(config, dict)
        or config.get("role") != role
        or config.get("service_count") != service_count
        or static_connection
    ):
        raise ValueError(
            "Managed role configuration must match standalone role/service count"
        )
    if not config.get("hardware_ids") or not config.get("profile_limits"):
        raise ValueError(
            "Managed roles require hardware identity and validated profile limits"
        )
    # Run before any role initializes NCCL. PAI can inject MIN_NCHANNELS=4,
    # while a standalone peer builds two channels. A private cross-host group
    # can then return corrupt tensors or hang without rejecting that mismatch.
    # Keep the managed transport policy explicit; never silently override an
    # operator setting or alter the integrated/static launcher's environment.
    for name in ("NCCL_MIN_NCHANNELS", "NCCL_MIN_CTAS"):
        value = os.environ.get(name)
        if value is not None and value != "2":
            raise ValueError(
                f"Managed role transport requires {name}=2, got {value!r}; "
                "set the same channel minimum on every Encoder/DiT/Decoder launcher"
            )
    os.environ.setdefault("NCCL_MIN_NCHANNELS", "2")
    logging.getLogger(__name__).info("Managed NCCL channel minimum: 2")


def validate_worker_configuration(role, service_count, *, static_connection=False):
    path = os.getenv("SGLANG_ROLE_POOL_WORKER_CONFIG")
    if not path:
        return
    config = json.loads(Path(path).read_text())
    validate_worker_config_dict(
        config, role, service_count, static_connection=static_connection
    )
