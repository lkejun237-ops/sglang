# SPDX-License-Identifier: Apache-2.0
"""Retirement evidence for an original create, never a new execution owner."""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath

# The reviewed boundary spans both the CLI's awaited sealed-source preparation
# and the Room's driver-return/admission boundary. Hashing only the diagnostic
# producer cannot establish that no player/model lease could have been issued.
STARTUP_SOURCE_PATHS = frozenset(
    {
        *[
            "sglang/python/sglang/multimodal_gen/runtime/realtime/whitebox/" + name
            for name in (
                "worldkit_room.py",
                "cli_room.py",
                "room_runtime.py",
                "room_control_runtime.py",
            )
        ],
        "sdk/packages/cli/src/room-runner.ts",
        "sdk/packages/stream-host/src/creator-room-source.ts",
        "sdk/packages/stream-host/src/delivery.ts",
        "sdk/packages/creator-host/src/multiplayer/delivery.ts",
        "sglang/python/sglang/multimodal_gen/runtime/entrypoints/whitebox_room_server.py",
        "sdk/packages/cli/bin/worldkit.mjs",
        "sdk/packages/cli/src/main.ts",
        "sdk/packages/cli/src/commands/room.ts",
    }
)


def invocation_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def _time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("An exact owner timestamp is required")
    return parsed.timestamp()


@dataclass(frozen=True, init=False)
class VerifiedCreateRetirement:
    """Private maintenance result. No HTTP request deserializes this type."""

    control_raw: str
    route_raw: str
    receipt_json: str


def verify_retired_create(
    caller,
    request,
    control_raw,
    route_raw,
    *,
    audit_bytes,
    approved_audit_sha256,
    container,
    startup_log,
    source_bytes,
):
    """Verify bounded observations collected by the privileged maintenance tool.

    Approval identifies the reviewed historical evidence, including the source
    analysis that the exact sealed-loader refusal precedes driver return and any
    admission. A digest cannot perform that semantic review. The tool separately
    reads Docker state/logs/source bytes; a caller-supplied cleanup flag is never
    evidence. No normal Room-control endpoint calls this function.
    """
    from .rooms import control_operation_key

    try:
        if (
            not isinstance(audit_bytes, bytes)
            or len(audit_bytes) > 262144
            or hashlib.sha256(audit_bytes).hexdigest() != approved_audit_sha256
        ):
            raise ValueError()
        audit = json.loads(audit_bytes)
        fields = {
            "schema_version",
            "kind",
            "caller",
            "operation_id",
            "parameters_sha256",
            "game_id",
            "owner_id",
            "owner_epoch",
            "container_id",
            "image_id",
            "started_at",
            "stopped_at",
            "epoch_observation",
            "stop_receipt",
            "startup_boundary",
            "invocation_sha256",
            "control_value_sha256",
            "route_value_sha256",
        }
        if (
            set(audit) != fields
            or type(audit["schema_version"]) is not int
            or audit["schema_version"] != 1
        ):
            raise ValueError()
        value, route = json.loads(control_raw), json.loads(route_raw)
        invocation = {k: request[k] for k in ("createContext", "videoSpec", "cliFps")}
        operation = invocation["createContext"]["operation"]
        if (
            audit["kind"] != "native-create-owner-retirement"
            or audit["caller"] != caller
            or value["invocation"] != invocation
            or "retirement" in value
            or audit["invocation_sha256"] != invocation_digest(invocation)
            or audit["control_value_sha256"]
            != hashlib.sha256(control_raw.encode()).hexdigest()
            or audit["route_value_sha256"]
            != hashlib.sha256(route_raw.encode()).hexdigest()
            or audit["operation_id"] != operation["operationId"]
            or audit["parameters_sha256"] != operation["parametersSha256"]
            or audit["game_id"] != value["game_id"]
            or audit["game_id"] != route["game_id"]
            or audit["owner_id"] != route["owner_id"]
            or audit["owner_epoch"] != route["owner_epoch"]
        ):
            raise ValueError()
        started, stopped = _time(audit["started_at"]), _time(audit["stopped_at"])
        state = container["State"]
        if (
            container["Id"] != audit["container_id"]
            or container["Image"] != audit["image_id"]
            or state["Status"] != "exited"
            or state["Running"] is not False
            or state["Pid"] != 0
            or container["HostConfig"]["RestartPolicy"]["Name"] != "no"
            or state["StartedAt"] != audit["started_at"]
            or state["FinishedAt"] != audit["stopped_at"]
            or not started <= route["created_at"] < stopped
        ):
            raise ValueError()
        epoch, stop = audit["epoch_observation"], audit["stop_receipt"]
        if (
            set(epoch)
            != {"container_id", "owner_id", "owner_epoch", "at", "evidence_sha256"}
            or set(stop)
            != {"container_id", "action", "at", "finished_at", "evidence_sha256"}
            or epoch["container_id"] != audit["container_id"]
            or epoch["owner_id"] != audit["owner_id"]
            or epoch["owner_epoch"] != audit["owner_epoch"]
            or stop["container_id"] != audit["container_id"]
            or stop["action"] != "stop"
            or stop["finished_at"] != audit["stopped_at"]
            or not started <= _time(epoch["at"]) <= _time(stop["at"]) <= stopped
            or any(
                not re.fullmatch(r"[a-f0-9]{64}", item["evidence_sha256"])
                for item in (epoch, stop)
            )
        ):
            raise ValueError()
        boundary = audit["startup_boundary"]
        if (
            set(boundary)
            != {
                "classification",
                "error_code",
                "review_evidence_sha256",
                "source_sha256",
                "source_paths",
            }
            or boundary["classification"] != "sealed-loader-before-driver"
            or boundary["error_code"] != "CREATOR_MULTIPLAYER_EVIDENCE_INVALID"
            or not re.fullmatch(r"[a-f0-9]{64}", boundary["review_evidence_sha256"])
            or set(source_bytes) != STARTUP_SOURCE_PATHS
            or set(boundary["source_sha256"]) != STARTUP_SOURCE_PATHS
            or set(boundary["source_paths"]) != STARTUP_SOURCE_PATHS
            or any(
                not isinstance(path, str)
                or not path.startswith("/")
                or "\x00" in path
                or ".." in PurePosixPath(path).parts
                or PurePosixPath(path).suffix not in {".py", ".ts", ".js", ".mjs"}
                for path in boundary["source_paths"].values()
            )
            or any(
                not isinstance(data, bytes)
                or len(data) > 2 * 1024 * 1024
                or hashlib.sha256(data).hexdigest() != boundary["source_sha256"][role]
                for role, data in source_bytes.items()
            )
            or not isinstance(startup_log, bytes)
            or len(startup_log) > 2 * 1024 * 1024
        ):
            raise ValueError()
        marker = (
            "CLI startup failed trace_id="
            + operation["operationId"]
            + " code="
            + boundary["error_code"]
        )
        matched = False
        for line in startup_log.decode("utf-8", errors="replace").splitlines():
            timestamp, _, message = line.partition(" ")
            if (
                message.strip().endswith(marker)
                and max(started, route["created_at"]) <= _time(timestamp) <= stopped
            ):
                matched = True
        if not matched:
            raise ValueError()
        key = control_operation_key(caller, operation["operationId"])
        receipt = {
            "schema_version": 1,
            "control_key": key,
            "invocation_sha256": invocation_digest(invocation),
            "game_id": value["game_id"],
            "owner_id": route["owner_id"],
            "owner_epoch": route["owner_epoch"],
            "container_id": container["Id"],
            "evidence_sha256": approved_audit_sha256,
            "stopped_at": audit["stopped_at"],
        }
        retired_create_result({**value, "retirement": receipt}, key)
        proof = object.__new__(VerifiedCreateRetirement)
        object.__setattr__(proof, "control_raw", control_raw)
        object.__setattr__(proof, "route_raw", route_raw)
        object.__setattr__(proof, "receipt_json", json.dumps(receipt, sort_keys=True))
        return proof
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError):
        raise ValueError(
            "The original create retirement evidence is incomplete or changed"
        ) from None


def retired_create_result(value, control_key):
    """Project an already accepted, persisted maintenance proof.

    The business request cannot supply this field. Its caller and complete frozen
    invocation have already selected and matched the original directory record.
    A route's expiry or absence alone never reaches this projection.
    """
    proof = value["retirement"]
    fields = {
        "schema_version",
        "control_key",
        "invocation_sha256",
        "game_id",
        "owner_id",
        "owner_epoch",
        "container_id",
        "evidence_sha256",
        "stopped_at",
    }
    if not isinstance(proof, dict) or set(proof) != fields:
        raise ValueError("Invalid original owner retirement proof")
    if (
        type(proof["schema_version"]) is not int
        or proof["schema_version"] != 1
        or proof["control_key"] != control_key
        or proof["game_id"] != value["game_id"]
        or proof["invocation_sha256"] != invocation_digest(value["invocation"])
        or not isinstance(proof["owner_id"], str)
        or not 1 <= len(proof["owner_id"]) <= 256
        or any(
            not isinstance(proof[name], str) or not re.fullmatch(pattern, proof[name])
            for name, pattern in (
                ("game_id", r"[a-f0-9]{32}"),
                ("owner_epoch", r"[a-f0-9]{32}"),
                ("container_id", r"[a-f0-9]{64}"),
                ("evidence_sha256", r"[a-f0-9]{64}"),
            )
        )
    ):
        raise ValueError("The original owner retirement binding has changed")
    try:
        stopped = datetime.fromisoformat(proof["stopped_at"].replace("Z", "+00:00"))
        if stopped.tzinfo is None:
            raise ValueError()
    except (TypeError, AttributeError, ValueError):
        raise ValueError("Invalid owner retirement time") from None
    operation = value["invocation"]["createContext"]["operation"]
    return {
        "operationId": operation["operationId"],
        "parametersSha256": operation["parametersSha256"],
        "status": "failed",
        "cleanupComplete": True,
        "errorCode": "ROOM_OWNER_RETIRED",
    }
