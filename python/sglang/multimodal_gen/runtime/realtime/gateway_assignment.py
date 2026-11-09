# SPDX-License-Identifier: Apache-2.0
"""Validate the worker's gateway assignment before consuming its reservation."""

from collections.abc import Mapping
from dataclasses import dataclass

from .admission import AdmissionRejected


@dataclass(frozen=True, slots=True)
class GatewayManagedConfig:
    session_id: str
    generation_id: str
    coordinator_token: str
    worker_epoch: str
    vae_worker_url: str
    vae_worker_epoch: str
    output_url: str
    output_token: str


def parse_gateway_assignment(query: Mapping[str, str]) -> GatewayManagedConfig | None:
    if query.get("gateway_managed") != "1":
        return None
    fields = {
        "session_id": query.get("session_id"),
        "generation_id": query.get("generation_id"),
        "coordinator_token": query.get("coordinator_token"),
        "worker_epoch": query.get("worker_epoch"),
        "vae_worker_url": query.get("realtime_vae_worker_url"),
        "vae_worker_epoch": query.get("realtime_vae_worker_epoch"),
        "output_url": query.get("gateway_output_url"),
        "output_token": query.get("gateway_output_token"),
    }
    local_decode = query.get("gateway_decode_mode") == "local"
    required = (
        ("session_id", "generation_id", "coordinator_token", "worker_epoch")
        if local_decode
        else tuple(fields)
    )
    if not all(fields[name] for name in required):
        raise AdmissionRejected("INVALID_GATEWAY_ASSIGNMENT")
    if local_decode:
        for name in (
            "vae_worker_url",
            "vae_worker_epoch",
            "output_url",
            "output_token",
        ):
            if fields[name]:
                raise AdmissionRejected("INVALID_GATEWAY_ASSIGNMENT")
            fields[name] = ""
    return GatewayManagedConfig(**fields)
