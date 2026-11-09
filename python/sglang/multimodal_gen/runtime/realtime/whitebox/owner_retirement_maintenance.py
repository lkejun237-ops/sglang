# SPDX-License-Identifier: Apache-2.0
"""Privileged, explicit reconciliation of one reviewed stopped-owner create.

No HTTP endpoint installs this module. The deployment tool supplies bounded
Docker observations on stdin; credentials remain in the original coordinator.
The default is a read-only validation, not a business cleanup receipt.
"""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import sys
from contextlib import redirect_stderr
from pathlib import Path

from .owner_retirement import retired_create_result, verify_retired_create
from .room_control import identities, validate_request
from .rooms import RedisRoomStore, RoomDirectory, control_operation_key


def decode(value, limit):
    if not isinstance(value, str) or len(value) > (limit + 2) // 3 * 4:
        raise ValueError("Retirement evidence exceeds its bound")
    data = base64.b64decode(value, validate=True)
    if not data or len(data) > limit:
        raise ValueError("Retirement evidence is empty or oversized")
    return data


async def reconcile_retirement_payload(
    directory, payload, approved_sha256, *, apply=False
):
    if not isinstance(payload, dict) or set(payload) != {
        "audit",
        "container",
        "startup_log",
        "sources",
        "references",
    }:
        raise ValueError("Invalid retirement maintenance payload")
    audit_bytes = decode(payload["audit"], 262144)
    if hashlib.sha256(audit_bytes).hexdigest() != approved_sha256:
        raise ValueError("The reviewed retirement evidence has changed")
    audit = json.loads(audit_bytes)
    caller = audit["caller"]
    identities(caller, {"serviceId", "subject"})
    key = control_operation_key(caller, audit["operation_id"])
    control_raw = await directory.store.get(key)
    if control_raw is None:
        raise ValueError("The original control record was not found")
    value = json.loads(control_raw)
    body = validate_request(
        {"schemaVersion": 1, "action": "getOperation", **value["invocation"]}
    )
    if "retirement" in value:
        if value["retirement"]["evidence_sha256"] != approved_sha256:
            raise ValueError("The original retirement proof has changed")
        return {
            "validated": True,
            "applied": True,
            "already_applied": True,
            "operation": retired_create_result(value, key),
        }
    route_raw = await directory.store.get(value["game_id"])
    if route_raw is None:
        raise ValueError("The original owner binding was not found")
    references = payload["references"]
    if not isinstance(references, dict) or set(references) != {
        "epoch",
        "stop",
        "source_review",
    }:
        raise ValueError("Original retirement references are required")
    expected = {
        "epoch": audit["epoch_observation"]["evidence_sha256"],
        "stop": audit["stop_receipt"]["evidence_sha256"],
        "source_review": audit["startup_boundary"]["review_evidence_sha256"],
    }
    for name, digest in expected.items():
        if (
            hashlib.sha256(decode(references[name], 2 * 1024 * 1024)).hexdigest()
            != digest
        ):
            raise ValueError("An original retirement reference has changed")
    proof = verify_retired_create(
        caller,
        body,
        control_raw,
        route_raw,
        audit_bytes=audit_bytes,
        approved_audit_sha256=approved_sha256,
        container=payload["container"],
        startup_log=decode(payload["startup_log"], 2 * 1024 * 1024),
        source_bytes={
            name: decode(data, 2 * 1024 * 1024)
            for name, data in payload["sources"].items()
        },
    )
    if not apply:
        return {
            "validated": True,
            "applied": False,
            "already_applied": False,
            "operationId": audit["operation_id"],
            "evidenceSha256": approved_sha256,
        }
    result = await directory.confirm_retired_create(caller, body, proof)
    # An uncertain write is never retried here. A repeated reviewed invocation
    # reads the same original record and recognizes only this exact receipt.
    current = json.loads(await directory.store.get(key))
    if current.get("retirement", {}).get("evidence_sha256") != approved_sha256:
        raise ValueError("The original retirement write could not be confirmed")
    return {
        "validated": True,
        "applied": True,
        "already_applied": False,
        "operation": result,
    }


def original_coordinator_directory():
    """Reuse the running coordinator's parsed private configuration, never URLs from stdin."""
    from sglang.multimodal_gen.runtime.entrypoints.realtime_coordinator_server import (
        _parse_args,
    )
    from sglang.multimodal_gen.runtime.realtime.coordinator import RedisCoordinatorStore

    command = Path("/proc/1/cmdline").read_bytes()
    if len(command) > 32768:
        raise ValueError("The original coordinator configuration is unavailable")
    arguments = command.decode().rstrip("\0").split("\0")
    module = "sglang.multimodal_gen.runtime.entrypoints.realtime_coordinator_server"
    if "-m" not in arguments or arguments[arguments.index("-m") + 1] != module:
        raise ValueError("Run maintenance in the original coordinator container")
    previous = sys.argv
    try:
        sys.argv = [module, *arguments[arguments.index("-m") + 2 :]]
        try:
            with redirect_stderr(io.StringIO()):
                args = _parse_args()
        except SystemExit:
            raise ValueError(
                "The original coordinator configuration is unavailable"
            ) from None
    finally:
        sys.argv = previous
    if args.backend != "redis":
        raise ValueError("The existing distributed room store is required")
    store = RedisCoordinatorStore(
        redis_url=args.redis_url,
        host=args.redis_host,
        port=args.redis_port,
        username=args.redis_username,
        password=os.getenv(args.redis_password_env)
        if args.redis_password_env
        else None,
        password_env=None,
        db=args.redis_db,
        ssl=args.redis_ssl,
        key_prefix=args.redis_key_prefix,
        ttl_s=args.ttl_s,
        worker_ttl_s=args.worker_ttl_s,
    )
    directory = RoomDirectory(
        RedisRoomStore(
            store._get_client,
            os.environ.get("WHITEBOX_ROOM_KEY_PREFIX", "whitebox:rooms:v1:"),
        ),
        json.loads(os.environ["WHITEBOX_ROOM_OWNERS"]),
        max_seconds=int(os.environ.get("WHITEBOX_GAME_SECONDS", "3600")),
    )
    return directory, store


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-audit-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    data = sys.stdin.buffer.read(12 * 1024 * 1024 + 1)
    if len(data) > 12 * 1024 * 1024:
        raise ValueError("Retirement input exceeds its bound")
    directory, store = original_coordinator_directory()
    try:
        result = await reconcile_retirement_payload(
            directory, json.loads(data), args.approved_audit_sha256, apply=args.apply
        )
        print(json.dumps(result))
    finally:
        await store.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        # Network/configuration errors can contain private connection details.
        print(
            json.dumps(
                {
                    "error": "OWNER_RETIREMENT_NOT_CONFIRMED",
                    "errorType": type(error).__name__,
                }
            )
        )
        raise SystemExit(1) from None
