# SPDX-License-Identifier: Apache-2.0
"""Test-only worker provisioning and connection control; no model or cloud SDK.

Explicit HTTP services always win over launch configuration. Missing roles are
started from argv manifests on the test host. This module never discovers cloud
resources, selects GPUs, or starts an alternative model/service entrypoint.
"""

import asyncio
import json
import os
import signal
import subprocess
import time
import uuid
from copy import deepcopy
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

ROLES = ("encoder", "dit", "decoder")


def http_address(value):
    value = value if "://" in value else "http://" + value
    uri = urlsplit(value)
    if (
        uri.scheme not in ("http", "https")
        or not uri.hostname
        or uri.port is None
        or uri.port < 1
        or uri.username
        or uri.password
        or uri.path not in ("", "/")
        or uri.query
        or uri.fragment
    ):
        raise ValueError("Worker HTTP address must be http[s]://host:port")
    return value.rstrip("/")


class WorkerHTTPError(RuntimeError):
    def __init__(self, address, status, body):
        body = body if isinstance(body, dict) else {}
        self.address, self.status, self.body = address, status, body
        super().__init__(
            f"Worker HTTP request failed: {address} ({status}, {body.get('code', 'UNKNOWN')})"
        )


def http_request(address, path, *, payload=None, timeout=10):
    """Direct first; use configured proxies only after a network failure."""
    request = Request(
        address + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Cache-Control": "no-store"},
        method="GET" if payload is None else "POST",
    )

    def send(opener):
        try:
            with opener.open(request, timeout=timeout) as response:
                body = json.load(response)
        except HTTPError as exc:
            try:
                body = json.loads(exc.read())
            except (ValueError, UnicodeError):
                body = {}
            raise WorkerHTTPError(address + path, exc.code, body) from None
        if not isinstance(body, dict):
            raise ValueError("Worker returned an invalid JSON response")
        return body

    try:
        return send(build_opener(ProxyHandler({})))
    except (URLError, TimeoutError, ConnectionError):
        proxy = ProxyHandler()
        if not proxy.proxies:
            raise
        return send(build_opener(proxy))


class LocalWorkers:
    """Launch only missing roles; leave all supplied services untouched."""

    def __init__(self, config_path, directory):
        self.config_path = config_path
        self.directory = Path(directory)
        self.owned = []

    def start_missing(self, addresses):
        missing = [role for role in ROLES if not addresses.get(role)]
        if not missing:
            return dict(addresses)
        if not self.config_path:
            raise ValueError(
                "Missing worker HTTP addresses require --worker-launch-config; no cluster lookup is performed"
            )
        config = json.loads(Path(self.config_path).read_text())
        if not isinstance(config, dict) or not isinstance(config.get("workers"), dict):
            raise ValueError("Launch configuration must contain a workers object")
        workers = config["workers"]
        validated = []
        # Validate every missing role before starting any process.
        for role in missing:
            entry = workers.get(role, {})
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid launch configuration for {role}")
            command, env = entry.get("command"), entry.get("env", {})
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(part, str) or not part for part in command)
                or not isinstance(env, dict)
                or any(
                    not isinstance(k, str) or not isinstance(v, str)
                    for k, v in env.items()
                )
            ):
                raise ValueError(
                    f"Launch configuration requires an argv command and string environment for {role}"
                )
            address = http_address(entry.get("http", ""))
            cwd = entry.get("cwd")
            if cwd is not None and not Path(cwd).is_dir():
                raise ValueError(f"Invalid worker working directory for {role}")
            validated.append((role, entry, address))
        result = dict(addresses)
        self.directory.mkdir(parents=True, exist_ok=True)
        for role, entry, address in validated:
            log = (self.directory / f"{role}.log").open("wb")
            try:
                process = subprocess.Popen(
                    entry["command"],
                    cwd=entry.get("cwd"),
                    env=os.environ | entry.get("env", {}),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except BaseException:
                log.close()
                raise
            self.owned.append((role, process, log))
            result[role] = address
        (self.directory / "workers.json").write_text(
            json.dumps(
                dict(
                    http=result,
                    processes=[
                        dict(role=role, pid=process.pid)
                        for role, process, _ in self.owned
                    ],
                ),
                indent=2,
            )
        )
        return result

    def check(self):
        for role, process, _ in self.owned:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Launched {role} exited with status {process.returncode}; see {self.directory / (role + '.log')}"
                )

    def close(self):
        errors = []
        for role, process, log in reversed(self.owned):
            try:
                # A live direct child cannot have had its PID reused. Never
                # signal arbitrary services found through HTTP or a port scan.
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except Exception as exc:
                errors.append(f"{role}: {type(exc).__name__}")
            finally:
                log.close()
        if errors:
            raise RuntimeError("Worker process cleanup failed: " + ", ".join(errors))

    def detach(self):
        for _, _, log in self.owned:
            log.close()


class WorkerGroupMaster:
    def __init__(
        self,
        addresses,
        *,
        launcher,
        worker_group_id=None,
        startup_timeout=900,
        control_timeout=600,
        keep_connections=False,
        request=http_request,
    ):
        self.addresses = {
            role: http_address(value) if value else None
            for role, value in addresses.items()
        }
        self.launcher = launcher
        self.group = worker_group_id or "test-" + uuid.uuid4().hex
        self.group_explicit = worker_group_id is not None
        self.startup_timeout, self.control_timeout = startup_timeout, control_timeout
        self.keep_connections = keep_connections
        self.request = request
        self.topology = None
        self.created = False
        self.build_attempted = False

    async def rpc(self, role, path, payload=None, *, timeout=None):
        return await asyncio.to_thread(
            self.request,
            self.addresses[role],
            path,
            payload=payload,
            timeout=timeout or self.control_timeout,
        )

    async def wait_ready(self, role):
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            self.launcher.check()
            try:
                reply = await self.rpc(role, "/health", timeout=5)
                if reply.get("ready") is True:
                    return
            except WorkerHTTPError as exc:
                if exc.status != 503:
                    raise
            except (URLError, TimeoutError, ConnectionError):
                pass
            await asyncio.sleep(0.2)
        raise TimeoutError(f"Worker health timed out: {role}")

    async def prepare(self):
        self.addresses = self.launcher.start_missing(self.addresses)
        await asyncio.gather(*(self.wait_ready(role) for role in ROLES))
        # Obtain all three leases together; model startup never consumes them.
        replies = await asyncio.gather(
            *(self.rpc(role, "/get_connection_status", timeout=5) for role in ROLES)
        )
        responses = dict(zip(ROLES, replies))
        if all(reply.get("state") == "connected" for reply in replies):
            if not self.group_explicit or any(
                reply.get("worker_group_id") != self.group for reply in replies
            ):
                raise RuntimeError(
                    "Existing connections require the matching --worker-group-id"
                )
            topology = replies[0].get("topology")
            if (
                not isinstance(topology, dict)
                or set(topology) != set(ROLES)
                or any(reply.get("topology") != topology for reply in replies)
            ):
                raise ValueError("Workers returned inconsistent existing topology")
            self.topology = deepcopy(topology)
            return self
        if any(reply.get("state") != "available" for reply in replies):
            raise RuntimeError(
                "All three services must be available; existing services are never replaced"
            )
        topology = {}
        for role, reply in responses.items():
            service = reply.get("service", {})
            if service.get("role") != role or not isinstance(
                reply.get("reservation_token"), str
            ):
                raise ValueError(f"Invalid reservation response for {role}")
            for key in ("worker_id", "service_id", "endpoints"):
                if not service.get(key):
                    raise ValueError(f"Incomplete service descriptor for {role}")
            topology[role] = deepcopy(service)
        self.topology = topology
        # Each recipient uses its own token. Do not write tokens to logs/results.
        payloads = {
            role: dict(
                worker_group_id=self.group,
                reservation_token=responses[role]["reservation_token"],
                topology=topology,
            )
            for role in ROLES
        }
        self.build_attempted = True
        results = await asyncio.gather(
            *(self.rpc(role, "/build_group", payloads[role]) for role in ROLES),
            return_exceptions=True,
        )
        failures = [
            role
            for role, result in zip(ROLES, results)
            if isinstance(result, BaseException)
            or result.get("success") is not True
            or result.get("state") != "connected"
            or result.get("worker_group_id") != self.group
        ]
        if failures:
            raise RuntimeError("Worker group build failed: " + ", ".join(failures))
        self.created = True
        return self

    @property
    def input_ws(self):
        return self.topology["encoder"]["endpoints"]["input_ws"]

    @property
    def output_ws(self):
        return self.topology["decoder"]["endpoints"]["output_ws"]

    async def _quiesce_encoder(self):
        try:
            reply = await self.rpc(
                "encoder", "/quiesce_group", dict(worker_group_id=self.group)
            )
        except WorkerHTTPError as exc:
            if exc.status != 409 or self.created:
                raise
        else:
            if (
                reply.get("success") is not True
                or reply.get("state") != "quiesced"
                or reply.get("worker_group_id") != self.group
            ):
                raise RuntimeError("Worker group session did not finish before cleanup")

    async def _destroy_role_until_disconnected(self, role):
        deadline = time.monotonic() + self.control_timeout
        while time.monotonic() < deadline:
            try:
                result = await self.rpc(
                    role, "/destroy_group", dict(worker_group_id=self.group)
                )
            except WorkerHTTPError as exc:
                if exc.status != 409 or self.created:
                    raise
                # A partially built role can still be connecting or already
                # rolled back. Never disconnect a different owner.
                status = await self.rpc(
                    role, "/get_connection_status?reserve=false", timeout=5
                )
                if status.get("state") == "available":
                    return
                if status.get("state") != "reserved" and not (
                    status.get("state") == "connected"
                    and status.get("worker_group_id") == self.group
                ):
                    raise RuntimeError(f"Unexpected cleanup status: {role}")
            else:
                if result.get("state") == "disconnected":
                    return
                if result.get("state") != "cleaning":
                    raise RuntimeError(f"Unexpected destroy response: {role}")
                # Repeating destroy is idempotent and does not reserve a
                # newly idle service as get_connection_status would.
            await asyncio.sleep(0.2)
        raise TimeoutError(f"Worker cleanup timed out: {role}")

    async def destroy(self):
        if not self.build_attempted and not self.created and self.topology is None:
            return

        # First drain the session with every transport alive. NCCL shutdown is
        # collective, so only then may all three peers disconnect concurrently.
        await self._quiesce_encoder()
        results = await asyncio.gather(
            *(self._destroy_role_until_disconnected(role) for role in ROLES),
            return_exceptions=True,
        )
        errors = [
            f"{role}: {type(result).__name__}"
            for role, result in zip(ROLES, results)
            if isinstance(result, BaseException)
        ]
        if errors:
            raise RuntimeError("Worker group cleanup failed: " + ", ".join(errors))

    async def __aenter__(self):
        try:
            return await self.prepare()
        except BaseException as exc:
            try:
                await self.destroy()
            except Exception as cleanup_error:
                exc.add_note(str(cleanup_error))
            try:
                self.launcher.close()
            except Exception as cleanup_error:
                exc.add_note(str(cleanup_error))
            raise

    async def __aexit__(self, kind, error, traceback):
        if kind is None and self.keep_connections:
            self.launcher.detach()
            return
        try:
            await self.destroy()
        except Exception as exc:
            if error is None:
                raise
            error.add_note(str(exc))
        finally:
            self.launcher.close()

    def summary(self):
        return dict(
            worker_group_id=self.group,
            topology=self.topology,
            reused=not self.created,
            connections_kept=self.keep_connections,
        )


def add_arguments(parser):
    for role in ROLES:
        parser.add_argument(
            f"--{role}-http",
            help=f"Existing {role} service HTTP host:port; never launched again",
        )
    parser.add_argument(
        "--worker-launch-config",
        help="JSON argv manifest used only for roles without explicit HTTP addresses",
    )
    parser.add_argument(
        "--worker-group-id",
        help="New group identity, or identity of a retained existing group",
    )
    parser.add_argument(
        "--keep-connections",
        action="store_true",
        help="Retain the group and owned workers after a successful test",
    )
    parser.add_argument("--worker-startup-timeout", type=float, default=900)
    parser.add_argument("--worker-control-timeout", type=float, default=600)
    parser.add_argument(
        "--worker-log-dir", help="Worker startup logs and process receipt directory"
    )
    parser.add_argument(
        "--direct-ws",
        action="store_true",
        help="Use the original direct WS test mode without worker lifecycle control",
    )


def from_args(args):
    addresses = {role: getattr(args, role + "_http") for role in ROLES}
    directory = args.worker_log_dir or str(
        Path(args.output).with_name(
            Path(args.output).name + ".workers-" + uuid.uuid4().hex[:8]
        )
    )
    return WorkerGroupMaster(
        addresses,
        launcher=LocalWorkers(args.worker_launch_config, directory),
        worker_group_id=args.worker_group_id,
        startup_timeout=args.worker_startup_timeout,
        control_timeout=args.worker_control_timeout,
        keep_connections=args.keep_connections,
    )
