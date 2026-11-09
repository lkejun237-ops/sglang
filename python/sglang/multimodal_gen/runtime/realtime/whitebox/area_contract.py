# SPDX-License-Identifier: Apache-2.0
"""Pure bounded AREA DTO readers. Parsing grants no Room or display authority."""

from __future__ import annotations

import copy
import json
import re

from .native_pixels import checked_json, read_frame_identity, safe_int
from .render_contract import MAX_CONTROL_BYTES, canonical

AREA_PROTOCOL = "worldkit-area-presented-v1"
IDENTITY_KEYS = set(
    "operationId areaRevision runtimeSessionId worldEpoch publication areaId selectionKey selectedGroups".split()
)
PUBLICATION_KEYS = "sourceId worldId publicationId manifestSha256 loadingSha256 sourceManifestSha256 renditionInputSha256".split()
CAPTURE_KEYS = set(
    "presentationId epoch sourceFrameId simulationTick worldRevision capturedAtMilliseconds widthPixels heightPixels".split()
)


def _object(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("AREA_CONTRACT_INVALID")
    return value


def _bounded(value):
    checked_json(value)
    if len(canonical(value)) > MAX_CONTROL_BYTES:
        raise ValueError("AREA_CONTRACT_INVALID")


def _text(value, maximum=4096):
    return (
        isinstance(value, str)
        and 0 < len(value.encode("utf-16-le", "surrogatepass")) // 2 <= maximum
    )


def _strings(value):
    if (
        not isinstance(value, list)
        or any(
            not _text(v, MAX_CONTROL_BYTES)
            or v.strip(
                "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
            )
            != v
            for v in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError("AREA_CONTRACT_INVALID")
    return value


def _sorted(value):
    # Match JS Array.sort's UTF-16 ordering for identifiers supplied by the SDK.
    return sorted(value, key=lambda text: text.encode("utf-16-be"))


def read_area_identity(value):
    _bounded(value)
    _object(value, IDENTITY_KEYS)
    if (
        not all(
            _text(value[key]) for key in ("operationId", "runtimeSessionId", "areaId")
        )
        or not safe_int(value["areaRevision"], 1)
        or not safe_int(value["worldEpoch"])
    ):
        raise ValueError("AREA_CONTRACT_INVALID")
    publication = _object(value["publication"], PUBLICATION_KEYS)
    if not all(_text(publication[key]) for key in PUBLICATION_KEYS) or any(
        not re.fullmatch(r"[a-f0-9]{64}", publication[key])
        for key in PUBLICATION_KEYS[3:]
    ):
        raise ValueError("AREA_CONTRACT_INVALID")
    selected = _strings(value["selectedGroups"])
    try:
        selection = json.loads(value["selectionKey"])
        _object(selection, ("areaId", "groupIds", "selectedGroups"))
        groups = _strings(selection["groupIds"])
        if (
            selection["areaId"] != value["areaId"]
            or selection["selectedGroups"] != selected
            or selected != _sorted(selected)
            or groups != _sorted(groups)
            or not set(selected).issubset(groups)
            or canonical(selection).decode() != value["selectionKey"]
        ):
            raise ValueError()
    except (TypeError, ValueError) as error:
        raise ValueError("AREA_CONTRACT_INVALID") from error
    return copy.deepcopy(value)


def read_area_frame(value):
    _bounded(value)
    _object(value, ("type", "epoch", "identity", "frame", "source"))
    identity = read_area_identity(value["identity"])
    frame = read_frame_identity(value["frame"])
    capture = _object(value["source"], CAPTURE_KEYS)
    source = frame["source"]
    if (
        value["type"] != "area.frame"
        or not safe_int(value["epoch"])
        or value["epoch"] != frame["epoch"]
        or not _text(capture["presentationId"], 128)
        or any(
            not safe_int(capture[k])
            for k in CAPTURE_KEYS - {"presentationId", "capturedAtMilliseconds"}
        )
        or not isinstance(capture["capturedAtMilliseconds"], (int, float))
        or isinstance(capture["capturedAtMilliseconds"], bool)
        or capture["capturedAtMilliseconds"] < 0
        or not 1 <= capture["widthPixels"] <= 8192
        or not 1 <= capture["heightPixels"] <= 8192
        or capture["presentationId"] != source["presentationId"]
        or capture["epoch"] != source["sdkEpoch"]
        or any(
            capture[key] != source[key]
            for key in ("sourceFrameId", "simulationTick", "worldRevision")
        )
    ):
        raise ValueError("AREA_CONTRACT_INVALID")
    return dict(
        type="area.frame",
        epoch=value["epoch"],
        identity=identity,
        frame=frame,
        source=copy.deepcopy(capture),
    )


def read_area_receipt(value):
    _bounded(value)
    _object(value, ("identity", "frame", "uiRevision", "uiStatus"))
    identity = read_area_identity(value["identity"])
    frame = read_frame_identity(value["frame"])
    if (
        value["uiStatus"] != "synchronized"
        or not safe_int(value["uiRevision"])
        or value["uiRevision"] != frame["uiRevision"]
    ):
        raise ValueError("AREA_CONTRACT_INVALID")
    return dict(
        identity=identity,
        frame=frame,
        uiRevision=value["uiRevision"],
        uiStatus="synchronized",
    )


PHASES = {
    "idle",
    "downloading",
    "resources-ready",
    "initializing",
    "world-ready",
    "awaiting-presented",
    "interactive",
    "cancelled",
    "failed",
    "disposed",
}


def read_area_transition(value):
    _bounded(value)
    if (
        not isinstance(value, dict)
        or set(value) - {"type", "epoch", "revision", "status", "runtime", "ack"}
        or not {"type", "epoch", "revision", "status", "runtime"}.issubset(value)
    ):
        raise ValueError("AREA_CONTRACT_INVALID")
    if (
        value["type"] != "area.transition"
        or not safe_int(value["epoch"])
        or not safe_int(value["revision"])
    ):
        raise ValueError("AREA_CONTRACT_INVALID")
    status = value["status"]
    required = {"phase", "inputHeld", "draining", "recoverable", "residentAreas"}
    if (
        not isinstance(status, dict)
        or not required.issubset(status)
        or set(status)
        - required
        - {"current", "transition", "pending", "resourceOperationId", "error"}
        or status["phase"] not in PHASES
        or not safe_int(status["residentAreas"])
        or any(
            type(status[key]) is not bool
            for key in ("inputHeld", "draining", "recoverable")
        )
    ):
        raise ValueError("AREA_CONTRACT_INVALID")
    for key in ("transition", "pending"):
        if key in status:
            read_area_identity(status[key])
    if (
        status["phase"] in {"awaiting-presented", "interactive"}
        and "transition" not in status
        or status["phase"] == "awaiting-presented"
        and status["inputHeld"] is not True
    ):
        raise ValueError("AREA_CONTRACT_INVALID")
    if "current" in status:
        _object(status["current"], ("areaId", "selectionKey"))
        if not all(
            _text(status["current"][key], MAX_CONTROL_BYTES)
            for key in ("areaId", "selectionKey")
        ):
            raise ValueError("AREA_CONTRACT_INVALID")
    if "resourceOperationId" in status and not _text(
        status["resourceOperationId"], 128
    ):
        raise ValueError("AREA_CONTRACT_INVALID")
    if "error" in status and (
        not isinstance(status["error"], str)
        or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", status["error"])
    ):
        raise ValueError("AREA_CONTRACT_INVALID")
    runtime = _object(value["runtime"], ("runtimeHash", "runtimeManifestSha256"))
    if any(
        not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v)
        for v in runtime.values()
    ):
        raise ValueError("AREA_CONTRACT_INVALID")
    if "ack" in value:
        _object(value["ack"], ("nonce",))
        if (
            not _text(value["ack"]["nonce"], 128)
            or status["phase"] != "awaiting-presented"
            or not status["inputHeld"]
        ):
            raise ValueError("AREA_CONTRACT_INVALID")
    return copy.deepcopy(value)


def read_area_result(value):
    _bounded(value)
    required = {"type", "epoch", "requestId", "operationId", "status"}
    if (
        not isinstance(value, dict)
        or not required.issubset(value)
        or set(value) - required - {"code"}
        or value["type"] != "area.presentation-result"
        or not safe_int(value["epoch"])
        or not _text(value["requestId"], 128)
        or not _text(value["operationId"], 128)
        or value["status"] not in {"confirmed", "duplicate", "rejected"}
        or value["status"] == "rejected"
        and (
            not isinstance(value.get("code"), str)
            or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", value["code"])
        )
        or value["status"] != "rejected"
        and "code" in value
    ):
        raise ValueError("AREA_CONTRACT_INVALID")
    return copy.deepcopy(value)


def area_source_options(world):
    """Explicit source intent only; publication admission remains the public SDK's job."""
    loading = world.get("loading", "complete")
    groups = world.get("initial_groups", [])
    if loading not in {"complete", "areas"} or not isinstance(groups, list):
        raise ValueError("AREA_SOURCE_OPTIONS_INVALID")
    _bounded(groups)
    _strings(groups)
    if groups and loading != "areas":
        raise ValueError("AREA_SOURCE_OPTIONS_INVALID")
    return (
        dict(loading="areas", initial_groups=list(groups))
        if loading == "areas"
        else None
    )


def validate_area_descriptor(descriptor, options):
    if not options:
        return
    loading = descriptor.get("loading", {})
    initial = loading.get("initialPreparation", {})
    if (
        descriptor.get("area") != {"protocol": AREA_PROTOCOL}
        or descriptor.get("artifact", {}).get("kind") != "wtp-area-rendition"
        or loading.get("kind") != "areas"
        or initial.get("kind") != "area-resources-ready"
        or not _text(initial.get("operationId"))
        or not _text(initial.get("areaId"))
        or initial.get("selectedGroups") != _sorted(options["initial_groups"])
        or descriptor.get("cache", {}).get("scope") != "partial-area"
        or descriptor["cache"].get("status") not in {"partial", "disabled"}
        or descriptor["cache"].get("updatePolicy") != "sync"
        or descriptor.get("cacheComplete") is True
    ):
        raise ValueError("AREA_SOURCE_ADMISSION_INVALID")
    publication = _object(initial.get("identity"), PUBLICATION_KEYS)
    if (
        not all(_text(publication[key]) for key in PUBLICATION_KEYS)
        or any(
            not re.fullmatch(r"[a-f0-9]{64}", publication[key])
            for key in PUBLICATION_KEYS[3:]
        )
        or publication["manifestSha256"] != descriptor.get("worldBuildHash")
    ):
        raise ValueError("AREA_SOURCE_ADMISSION_INVALID")
