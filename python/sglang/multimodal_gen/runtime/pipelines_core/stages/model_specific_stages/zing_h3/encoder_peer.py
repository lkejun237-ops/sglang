# SPDX-License-Identifier: Apache-2.0
"""Explicit, authenticated transport between two pods on the same node.

Loopback remains the default. This does not enable cross-node deployment:
private peers must prove the same kernel boot identity or use an explicit
control-plane-verified pod binding before any owner or NCCL bridge is created. Secrets come from a deployment file, never a request.
"""

import hashlib
import hmac
import ipaddress
import json
import os
import re
import stat
from pathlib import Path


def private_address(value):
    address = ipaddress.ip_address(value)
    if address.version != 4 or not (
        address.is_loopback
        or any(
            address in ipaddress.ip_network(network)
            for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
        )
    ):
        raise ValueError(
            "Encoder transport requires loopback or a private IPv4 address"
        )
    return str(address)


class EncoderPeer:
    def __init__(
        self,
        *,
        host="127.0.0.1",
        bridge_host="127.0.0.1",
        auth_token_file=None,
        node_binding_file=None,
    ):
        self.host = private_address(host)
        self.bridge_host = private_address(bridge_host)
        self.private = not ipaddress.ip_address(self.host).is_loopback
        if self.private != (not ipaddress.ip_address(self.bridge_host).is_loopback):
            raise ValueError(
                "Encoder peers must both use loopback or private addresses"
            )
        self.token = self.node = None
        if self.private:
            if not auth_token_file or not Path(auth_token_file).is_absolute():
                raise ValueError(
                    "Private encoder transport requires an absolute token file"
                )
            path = Path(auth_token_file)
            if stat.S_IMODE(path.stat().st_mode) & 0o077:
                raise ValueError(
                    "Encoder token file must not be accessible to other users"
                )
            token = path.read_text().strip()
            if not re.fullmatch(r"[a-f0-9]{64}", token):
                raise ValueError("Encoder transport token must contain 32 random bytes")
            self.token = token
            if node_binding_file is None:
                self.node = hashlib.sha256(
                    Path("/proc/sys/kernel/random/boot_id").read_bytes().strip()
                ).hexdigest()
            else:
                self.node = self._binding_identity(node_binding_file)

    def _binding_identity(self, filename):
        """The deployer verifies scheduler placement; workloads never forge it.

        Some PAI sandboxes expose different boot IDs on the same scheduler node.
        A binding is trusted deployment configuration, just like the peer token;
        it must be created only after inspecting both actual Pod UIDs/addresses.
        """
        path = Path(filename)
        if not path.is_absolute() or stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError("Encoder binding must be an absolute private file")
        value = json.loads(path.read_text())
        if (
            value.get("version") != 1
            or value.get("verified_by") != "pai-control-plane"
            or not isinstance(value.get("node_name"), str)
            or not value["node_name"]
        ):
            raise ValueError("Encoder binding requires verified scheduler placement")
        pods = value.get("pods")
        if not isinstance(pods, list) or len(pods) != 2:
            raise ValueError("Encoder binding requires exactly two pods")
        for pod in pods:
            if (
                not isinstance(pod, dict)
                or set(pod) != {"pod_name", "pod_uid", "host", "node_name"}
                or pod["node_name"] != value["node_name"]
                or not all(isinstance(pod[k], str) and pod[k] for k in pod)
                or private_address(pod["host"]) != pod["host"]
                or ipaddress.ip_address(pod["host"]).is_loopback
            ):
                raise ValueError("Invalid encoder pod binding")
        if any(
            len({pod[k] for pod in pods}) != 2 for k in ("pod_name", "pod_uid", "host")
        ):
            raise ValueError("Encoder binding pods must be distinct")
        local = [pod for pod in pods if pod["pod_name"] == os.environ.get("HOSTNAME")]
        if (
            len(local) != 1
            or local[0]["host"] != self.bridge_host
            or {pod["host"] for pod in pods} != {self.host, self.bridge_host}
        ):
            raise ValueError("Encoder binding does not match this pod and peer")
        canonical = dict(
            node_name=value["node_name"], pods=sorted(pods, key=lambda p: p["pod_name"])
        )
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True).encode()
        ).hexdigest()

    def fields(self):
        return (
            {"peer_token": self.token, "peer_node": self.node} if self.private else {}
        )

    def accepts(self, command):
        if not isinstance(command, dict):
            return False
        if self.private and not all(
            isinstance(command.get(key), str)
            and re.fullmatch(r"[a-f0-9]{64}", command[key]) is not None
            and hmac.compare_digest(command[key], expected)
            for key, expected in self.fields().items()
        ):
            return False
        if command.get("op") in {"open", "bridge_open"}:
            if "bridge_host" in command and command["bridge_host"] != self.host:
                return False
            if command.get("op") == "open" and "bridge_id" in command:
                return (
                    isinstance(command["bridge_id"], str)
                    and 0 < len(command["bridge_id"]) <= 256
                )
            if command.get("bridge_host", "127.0.0.1") != self.host:
                return False
            port = command.get("bridge_port")
            if type(port) is not int or not 1024 <= port <= 65535:
                return False
        return True
