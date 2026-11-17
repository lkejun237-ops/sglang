# SPDX-License-Identifier: Apache-2.0
"""RDMA-only transport selection for the Zing Block0 Mooncake Store and client."""

import os
from pathlib import Path


def _active_hcas(devices=Path("/dev/infiniband"), sys_class=Path("/sys/class")):
    names = []
    for node in sorted(devices.glob("uverbs*")):
        ibdev = sys_class / "infiniband_verbs" / node.name / "ibdev"
        try:
            name = ibdev.read_text().strip()
            ports = (sys_class / "infiniband" / name / "ports").glob("*/state")
            if any(port.read_text().strip().endswith(": ACTIVE") for port in ports):
                names.append(name)
        except FileNotFoundError:
            continue
    return ",".join(names)


def block0_rdma_devices(config):
    """Pass named HCAs to the native SDK so a host with no HCA cannot select TCP."""
    if config.get("protocol", "rdma") != "rdma":
        raise ValueError(
            'Mooncake Block0 requires "protocol": "rdma"; update the cache config'
        )
    # In Mooncake 0.3.13 these switches test getenv() for presence, including "0".
    for flag in ("MC_FORCE_TCP", "MC_USE_TENT", "MC_USE_TEV1"):
        if flag in os.environ:
            raise ValueError(f"Unset {flag} to use the Mooncake Block0 RDMA transport")
    if os.environ.get("MC_MS_AUTO_DISC", "0") != "0":
        raise ValueError(
            "Unset MC_MS_AUTO_DISC or set it to 0 to use named RDMA devices"
        )

    devices = config.get("rdma_devices", "")
    if not isinstance(devices, str):
        raise ValueError("rdma_devices must be a comma-separated list of HCA names")
    if devices:
        names = [name.strip() for name in devices.split(",")]
        if not all(names):
            raise ValueError("rdma_devices contains an empty HCA name")
        return ",".join(names)
    devices = _active_hcas()
    if not devices:
        raise ValueError(
            "No active RDMA HCA visible to Mooncake Block0; expose a verbs device or configure rdma_devices for this host"
        )
    return devices
