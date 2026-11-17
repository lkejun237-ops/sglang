# SPDX-License-Identifier: Apache-2.0
"""Service-only Room control on the original Gateway -> directory -> Room hop.

The WTP journal owns business operations. The existing Room directory retains
only their immutable routing association; querying never creates a route.
"""

import hashlib
import json
import os
import re
import time

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .rooms import checked_id, resolve_owner
from .rtc_auth import RTCCapabilities

MAX_CONTROL_BYTES = 256 * 1024
MAX_MEMBER_CONTROL_BYTES = 12 * 1024**2 + 16384
ACTIONS = {
    "createRoom",
    "getOperation",
    "retireRoom",
    "admitPlayer",
    "getAdmission",
    "leavePlayer",
    "getPlayerSession",
    "getPlayerMedia",
}
BINDING_FIELDS = {"gameId", "ownerEpoch", "publicationId"}
MEMBER_FIELDS = BINDING_FIELDS | {"memberId", "admissionOperationId"}


class RoomControlCapabilities(RTCCapabilities):
    """Same signed-byte codec, separate audience, key and service allowlist."""

    def __init__(
        self,
        secret,
        services,
        *,
        now=time.time,
        audience="whitebox-room-control",
        actions=None,
    ):
        super().__init__(secret, now=now, audience=audience)
        self.actions = ACTIONS if actions is None else frozenset(actions)
        if not services or any(not isinstance(s, str) or not s for s in services):
            raise ValueError("Room control services are not configured")
        self.services = frozenset(services)

    def verify(self, token, *, raw, action):
        try:
            claims = self._decode(token)
            now = self.now()
            if (
                type(claims.get("v")) is not int
                or claims["v"] != 3
                or claims.get("aud") != self.audience
                or claims.get("service_id") not in self.services
                or claims.get("action") != action
                or action not in self.actions
                or claims.get("request_sha256") != hashlib.sha256(raw).hexdigest()
                or not isinstance(claims.get("sub"), str)
                or not 1 <= len(claims["sub"]) <= 256
                or type(claims.get("iat")) is not int
                or type(claims.get("exp")) is not int
                or not claims["iat"] <= now + 5
                or not now < claims["exp"] <= claims["iat"] + 120
                or not re.fullmatch(r"[a-f0-9]{32}", str(claims.get("jti", "")))
            ):
                raise ValueError()
            self.used = {
                key: expiry for key, expiry in self.used.items() if expiry > now
            }
            if claims["jti"] in self.used or len(self.used) >= 4096:
                raise ValueError()
            self.used[claims["jti"]] = claims["exp"]
            return {"serviceId": claims["service_id"], "subject": claims["sub"]}
        except (ValueError, TypeError, KeyError, UnicodeError):
            raise ValueError(
                "Room control authorization could not be verified"
            ) from None


def exact(value, fields, optional=()):
    if (
        not isinstance(value, dict)
        or not fields <= value.keys()
        or value.keys() - fields - set(optional)
    ):
        raise ValueError("Invalid room control fields")


def identities(value, fields):
    exact(value, fields)
    if any(
        not isinstance(value[key], str) or not 1 <= len(value[key]) <= 256
        for key in fields
    ):
        raise ValueError("Invalid room control identity")


def operation(value, kind):
    identities(value, {"operationId", "requestId", "kind", "parametersSha256"})
    if value["kind"] != kind or not re.fullmatch(
        r"[a-f0-9]{64}", value["parametersSha256"]
    ):
        raise ValueError("Invalid room control operation")


def frozen_publication_identity(frozen):
    """Read only the frozen public identity; world also contains private run IDs."""
    from .cli_room import publication_identity

    value = frozen.get("publicationIdentity")
    result = publication_identity(value)
    if "attempt" in value:
        result["attempt"] = value["attempt"]
    world = frozen["world"]
    if any(
        result[key] != world.get(key)
        for key in ("sourceId", "worldId", "publicationId")
    ) or (
        "attempt" in world
        and "attempt" in result
        and (type(world["attempt"]) is not int or world["attempt"] != result["attempt"])
    ):
        raise ValueError("The frozen public identity does not match its world")
    return result


def validate_request(body):
    if (
        not isinstance(body, dict)
        or type(body.get("schemaVersion")) is not int
        or body["schemaVersion"] != 1
    ):
        raise ValueError("Invalid room control schema")
    action = body.get("action")
    if action not in ACTIONS:
        raise ValueError("Invalid room control action")
    common = {"schemaVersion", "action"}
    if action in {"createRoom", "getOperation", "retireRoom"}:
        exact(
            body,
            common
            | {"createContext", "videoSpec", "cliFps"}
            | ({"execution"} if action == "retireRoom" else set()),
        )
        context = body["createContext"]
        exact(context, {"schemaVersion", "operation", "frozenPublication"})
        if type(context["schemaVersion"]) is not int or context["schemaVersion"] != 1:
            raise ValueError("Invalid room create context")
        operation(context["operation"], "createRoom")
        frozen = context["frozenPublication"]
        if (
            not isinstance(frozen, dict)
            or frozen.get("status") != "verified"
            or frozen.get("scope") != "world-frozen-publication"
        ):
            raise ValueError("A frozen publication is required")
        world = frozen.get("world", {})
        for key in ("sourceId", "worldId", "publicationId"):
            if not isinstance(world.get(key), str) or not 1 <= len(world[key]) <= 256:
                raise ValueError("Invalid frozen world identity")
        from .runtime_management import world_reference

        world_reference(world["sourceId"] + "/" + world["worldId"])
        if "publicationIdentity" in frozen:
            frozen_publication_identity(frozen)
        spec = body["videoSpec"]
        exact(spec, {"width", "height", "fps"})
        if (
            any(type(spec[k]) is not int for k in spec)
            or spec["fps"] != 24
            or any(not 1 <= spec[k] <= 4096 for k in ("width", "height"))
            or type(body["cliFps"]) is not int
            or not 1 <= body["cliFps"] <= 60
        ):
            raise ValueError("Invalid room video settings")
        if action == "retireRoom":
            identities(body["execution"], BINDING_FIELDS)
            if body["execution"]["publicationId"] != world["publicationId"]:
                raise ValueError("The publication has changed")
    elif action in {"admitPlayer", "getAdmission"}:
        exact(
            body,
            common | {"binding", "admission"},
            (
                {"mediaLaunch", "leaveOperationId"}
                if action == "getAdmission"
                else {"mediaLaunch"}
            ),
        )
        identities(body["binding"], BINDING_FIELDS)
        identities(body["admission"], {"operationId", "requestId", "parametersSha256"})
        if not re.fullmatch(r"[a-f0-9]{64}", body["admission"]["parametersSha256"]):
            raise ValueError("Invalid admission digest")
        if "mediaLaunch" in body:
            from .player_media import validate_media_launch

            validate_media_launch(body["mediaLaunch"], body["binding"]["publicationId"])
        if "leaveOperationId" in body and (
            not isinstance(body["leaveOperationId"], str)
            or not 1 <= len(body["leaveOperationId"]) <= 256
        ):
            raise ValueError("Invalid leave operation")
    else:
        exact(
            body,
            common
            | {"memberBinding"}
            | ({"operation"} if action == "leavePlayer" else set()),
            {"controlSequence"} if action == "getPlayerMedia" else (),
        )
        identities(body["memberBinding"], MEMBER_FIELDS)
        if "controlSequence" in body:
            from .native_pixels import safe_int

            if not safe_int(body["controlSequence"]):
                raise ValueError("Invalid member media replay position")
        if action == "leavePlayer":
            operation(body["operation"], "leave")
    return body


async def read_request(
    request, *, validator=validate_request, max_bytes=MAX_CONTROL_BYTES
):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > max_bytes:
            raise HTTPException(413, "The room control request is too large")
    try:
        # Duplicate JSON keys would make byte authentication and object meaning
        # disagree across languages. Reject them before any routing mutation.
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError()
                result[key] = value
            return result

        return bytes(data), validator(json.loads(data, object_pairs_hook=unique))
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise HTTPException(400, "Invalid room control request") from None


def unknown(body):
    if "createContext" in body:
        op = body["createContext"]["operation"]
        result = {
            "operationId": op["operationId"],
            "parametersSha256": op["parametersSha256"],
            "status": "unknown",
        }
        if body["action"] == "retireRoom":
            result.update(execution=body["execution"], cleanupComplete=False)
        return result
    if "admission" in body:
        op = body["admission"]
        result = {
            "admissionOperationId": op["operationId"],
            "parametersSha256": op["parametersSha256"],
            "status": "blocked",
            "errorCode": "ADMISSION_UNCONFIRMED",
        }
        if "leaveOperationId" in body:
            result["leaveOperationId"] = body["leaveOperationId"]
        return result
    raise HTTPException(503, "The player owner is unavailable")


def install_gateway_room_control(app, proxy, *, verifier=None):
    if verifier is None:
        secret = os.environ.get("WHITEBOX_ROOM_CONTROL_SECRET", "")
        services = os.environ.get("WHITEBOX_ROOM_CONTROL_SERVICES", "")
        if secret or services:
            verifier = RoomControlCapabilities(
                secret, {v.strip() for v in services.split(",") if v.strip()}
            )

    @app.post("/v1/whitebox/room-control")
    @app.post("/backends/minwm/v1/whitebox/room-control")
    async def control(request: Request):
        raw, body = await read_request(request, max_bytes=MAX_MEMBER_CONTROL_BYTES)
        if len(raw) > MAX_CONTROL_BYTES and not (
            body["action"] in {"admitPlayer", "getAdmission"} and "mediaLaunch" in body
        ):
            raise HTTPException(413, "The room control request is too large")
        if verifier is None:
            raise HTTPException(503, "Room control is not configured")
        try:
            authorization = request.headers.get("authorization", "")
            if not authorization.startswith("Bearer "):
                raise ValueError()
            caller = verifier.verify(authorization[7:], raw=raw, action=body["action"])
        except ValueError:
            raise HTTPException(403, "Room control access was denied") from None
        try:
            if "createContext" in body:
                route = await proxy.directory.control_route(
                    caller, body, create=body["action"] == "createRoom"
                )
            else:
                binding = body.get("binding", body.get("memberBinding"))
                route = await proxy.directory.lookup(checked_id(binding["gameId"]))
            if route is None or route.get("status") == "expired":
                return JSONResponse(
                    unknown(body), headers={"Cache-Control": "no-store"}
                )
            if "closed_execution" in route:
                exact(route, {"closed_execution"})
                if body["action"] == "retireRoom":
                    if route["closed_execution"] != body["execution"]:
                        raise ValueError("The original retirement target has changed")
                    result = dict(
                        unknown(body), status="succeeded", cleanupComplete=True
                    )
                else:
                    result = dict(
                        unknown(body),
                        status="failed",
                        cleanupComplete=True,
                        errorCode="ROOM_ENDED",
                    )
                return JSONResponse(result, headers={"Cache-Control": "no-store"})
            if "retired_create" in route:
                # Only the original authenticated directory lookup supplies a
                # retirement outcome. No runtime dispatch follows this fence.
                exact(route, {"retired_create"})
                if body["action"] not in {"createRoom", "getOperation"}:
                    return JSONResponse(
                        unknown(body), headers={"Cache-Control": "no-store"}
                    )
                result = route["retired_create"]
                exact(
                    result,
                    {
                        "operationId",
                        "parametersSha256",
                        "status",
                        "cleanupComplete",
                        "errorCode",
                    },
                )
                operation = body["createContext"]["operation"]
                if (
                    result["operationId"] != operation["operationId"]
                    or result["parametersSha256"] != operation["parametersSha256"]
                    or result["status"] != "failed"
                    or result["cleanupComplete"] is not True
                    or result["errorCode"] != "ROOM_OWNER_RETIRED"
                ):
                    raise ValueError("Invalid original owner retirement result")
                return JSONResponse(result, headers={"Cache-Control": "no-store"})
            origin = await resolve_owner(
                proxy, route["owner_id"], route.get("owner_epoch")
            )
            if origin is None or origin.rstrip("/") != route["owner_url"].rstrip("/"):
                raise ValueError()
            if body["action"] == "getOperation" and route.get("status") == "ended":
                world = body["createContext"]["frozenPublication"]["world"]
                if (
                    route.get("world_id") != world["sourceId"] + "/" + world["worldId"]
                    or not isinstance(route.get("owner_epoch"), str)
                    or not route["owner_epoch"]
                ):
                    raise ValueError("The original ended owner is unavailable")
                # control_route has verified the original caller and complete
                # invocation. Ended fences new work, but the original Room can
                # still retain its failure and positive cleanup receipt. Query
                # that owner below; the route alone proves neither outcome.
            response = await proxy.client.post(
                origin.rstrip("/") + "/internal/room-control",
                json={"verifiedCaller": caller, "request": body, "route": route},
                headers={"Authorization": "Bearer " + proxy.token},
            )
            response.raise_for_status()
            if len(response.content) > 2 * 1024 * 1024:
                raise ValueError()
            return JSONResponse(response.json(), headers={"Cache-Control": "no-store"})
        except httpx.HTTPError:
            return JSONResponse(unknown(body), headers={"Cache-Control": "no-store"})
        except ValueError:
            raise HTTPException(409, "The room operation is unavailable") from None
