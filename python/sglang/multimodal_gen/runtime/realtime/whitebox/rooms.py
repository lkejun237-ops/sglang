# SPDX-License-Identifier: Apache-2.0
"""Room ownership, separate from the coordinator's per-session GPU leases.

IDs are CSPRNG UUIDv4 (122 random bits), not clock/counter-derived identifiers.
An owner is a directly addressable runtime Pod, never a load-balanced Service.
Owner loss ends a game; moving a live CLI requires a future checkpoint protocol.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import time
from uuid import uuid4

from fastapi import Header, HTTPException


def new_id() -> str:
    return uuid4().hex


def checked_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Invalid game identifier")
    return value


def checked_session_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("Invalid SDK session identifier")
    return value


def control_operation_key(caller, operation_id):
    return (
        "control:"
        + hashlib.sha256(
            json.dumps(
                [caller["serviceId"], caller["subject"], operation_id],
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
    )


class MemoryRoomStore:
    """Single-process development/test store. Production uses RedisRoomStore."""

    def __init__(self):
        self.values = {}
        self.lock = asyncio.Lock()

    async def get(self, key):
        return self.values.get(key)

    async def cas(self, key, old, new, retention, *, expected_route=None):
        async with self.lock:
            if (
                expected_route is not None
                and self.values.get(expected_route[0]) != expected_route[1]
            ):
                return False
            if self.values.get(key) != old:
                return False
            self.values[key] = new
            return True


class RedisRoomStore:
    # Update one original record atomically. A retirement can additionally fence
    # its existing route in the same command; no global lock or hot counter.
    SCRIPT = """
local current = redis.call('GET', KEYS[1])
if #KEYS == 2 and redis.call('GET', KEYS[2]) ~= ARGV[4] then
  return 0
end
if (not current and ARGV[1] == '') or current == ARGV[1] then
  if tonumber(ARGV[3]) == 0 then
    redis.call('SET', KEYS[1], ARGV[2])
  else
    redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
  end
  return 1
end
return 0
"""

    def __init__(self, client_factory, prefix="whitebox:rooms:v1:"):
        self.client_factory, self.prefix = client_factory, prefix

    async def get(self, key):
        value = await self.client_factory().get(self.prefix + key)
        return value.decode() if isinstance(value, bytes) else value

    async def cas(self, key, old, new, retention, *, expected_route=None):
        retention = 0 if retention is None else retention
        if type(retention) is not int or retention < 0:
            raise ValueError("Invalid room record retention")
        client = self.client_factory()
        keys = [self.prefix + key]
        arguments = [old or "", new, retention]
        if expected_route is not None:
            keys.append(self.prefix + expected_route[0])
            arguments.append(expected_route[1])
        return bool(await client.eval(self.SCRIPT, len(keys), *keys, *arguments))


class RoomDirectory:
    def __init__(
        self, store, owners, *, max_seconds=3600, lease_seconds=30, clock=time.time
    ):
        from urllib.parse import urlsplit

        if not owners or max_seconds <= lease_seconds or lease_seconds < 3:
            raise ValueError("Invalid room directory settings")
        for name, url in owners.items():
            parsed = urlsplit(url)
            if (
                not name
                or parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or parsed.path not in {"", "/"}
            ):
                raise ValueError("Runtime owners require direct HTTP origins")
        self.store, self.owners = store, dict(owners)
        self.max_seconds, self.lease_seconds, self.clock = (
            max_seconds,
            lease_seconds,
            clock,
        )

    async def resolve(self, game_id, world_id, *, create=False):
        checked_id(game_id)
        if not isinstance(world_id, str) or not world_id or len(world_id) > 256:
            raise ValueError("Invalid world identifier")
        old = await self.store.get(game_id)
        if old is None and create:
            owner = max(
                self.owners,
                key=lambda key: hashlib.sha256((game_id + ":" + key).encode()).digest(),
            )
            now = self.clock()
            room = dict(
                game_id=game_id,
                world_id=world_id,
                owner_id=owner,
                owner_url=self.owners[owner].rstrip("/"),
                owner_epoch=None,
                status="starting",
                created_at=now,
                expires_at=None,
                lease_until=now + self.lease_seconds,
            )
            await self.store.cas(game_id, None, json.dumps(room), 0)
            old = await self.store.get(game_id)
        if old is None:
            raise ValueError("The game was not found")
        room = json.loads(old)
        self.validate(room)
        if room["world_id"] != world_id:
            raise ValueError("The game belongs to another world")
        return room

    async def lookup(self, game_id):
        """Observe an existing route, including tombstones, without admission."""
        value = await self.store.get(checked_id(game_id))
        return json.loads(value) if value is not None else None

    async def control_route(self, caller, body, *, create=False):
        single = "singlePlayerOperation" in body
        context = body["singlePlayerOperation"] if single else body["createContext"]
        operation = context if single else context["operation"]
        # This is an internal storage key, not another wire digest or game ID.
        key = control_operation_key(caller, operation["operationId"])
        if single:
            key = "single-" + key
        invocation = (
            {
                "sha256": hashlib.sha256(
                    json.dumps(
                        {k: v for k, v in context.items() if k != "controlSequence"},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            }
            if single
            else {k: body[k] for k in ("createContext", "videoSpec", "cliFps")}
        )
        old = await self.store.get(key)
        if old is None and create:
            value = dict(invocation=invocation, game_id=new_id())
            await self.store.cas(key, None, json.dumps(value), 0)
            old = await self.store.get(key)
        if old is None:
            return None
        value = json.loads(old)
        if json.dumps(value["invocation"], sort_keys=True) != json.dumps(
            invocation, sort_keys=True
        ):
            raise ValueError("The original room operation has changed")
        if "closed_execution" in value:
            if (
                body["action"] == "retireRoom"
                and body["execution"] != value["closed_execution"]
            ):
                raise ValueError("The original retirement target has changed")
            return {"closed_execution": value["closed_execution"]}
        if "retirement" in value:
            from .owner_retirement import retired_create_result

            # The original control record retains accepted terminal evidence;
            # an expired game route must not recreate this retired execution.
            return {"retired_create": retired_create_result(value, key)}
        world = context["frozenPublication"]["world"]
        if single:
            existing = await self.lookup(value["game_id"])
            if existing is not None:
                if "route" not in value:
                    value["route"] = existing
                    await self.store.cas(key, old, json.dumps(value), 0)
                return existing
            if "route" in value:
                return dict(value["route"], status="expired")
        if create:
            route = await self.resolve(
                value["game_id"],
                world["sourceId"] + "/" + world["worldId"],
                create=True,
            )
            if single:
                # Persist the original route before dispatch. Its tombstone
                # outlives the ordinary game TTL; expiry cannot select a new owner.
                value["route"] = route
                await self.store.cas(key, old, json.dumps(value), 0)
            return route
        return await self.lookup(value["game_id"])

    async def confirm_room_cleanup(self, caller, invocation, execution, owner_id):
        """Persist the original Room's positive cleanup result beyond route TTL."""
        operation = invocation["createContext"]["operation"]
        game_id = checked_id(execution["gameId"])
        key = control_operation_key(caller, operation["operationId"])
        old = await self.store.get(key)
        if old is None:
            raise ValueError("The original room operation was not found")
        value = json.loads(old)
        world = invocation["createContext"]["frozenPublication"]["world"]
        if (
            value["invocation"] != invocation
            or value["game_id"] != game_id
            or execution["publicationId"] != world["publicationId"]
            or "retirement" in value
        ):
            raise ValueError("The original room operation has changed")
        if value.get("closed_execution") == execution:
            return {"recorded": True}
        route_raw = await self.store.get(game_id)
        route = json.loads(route_raw) if route_raw is not None else {}
        if (
            "closed_execution" in value
            or route.get("status") != "ended"
            or route.get("game_id") != game_id
            or route.get("owner_id") != owner_id
            or route.get("owner_epoch") != execution["ownerEpoch"]
            or route.get("world_id") != world["sourceId"] + "/" + world["worldId"]
        ):
            raise ValueError("The original ended owner is unavailable")
        value["closed_execution"] = execution
        if not await self.store.cas(
            key, old, json.dumps(value), None, expected_route=(game_id, route_raw)
        ):
            current = await self.store.get(key)
            if current is None or json.loads(current) != value:
                raise ValueError("The original room cleanup binding has changed")
        return {"recorded": True}

    async def confirm_retired_create(self, caller, body, proof):
        """Privileged maintenance only; no HTTP route exposes this method.

        Existing owner epochs cannot transfer between instances. The stopped
        instance evidence and exact route are checked before the original
        control-record CAS. Its terminal fence outlives ordinary routing TTLs.
        """
        from .owner_retirement import VerifiedCreateRetirement, retired_create_result

        if not isinstance(proof, VerifiedCreateRetirement):
            raise ValueError("Verified original owner retirement evidence is required")
        key = control_operation_key(
            caller, body["createContext"]["operation"]["operationId"]
        )
        receipt = json.loads(proof.receipt_json)
        invocation = {k: body[k] for k in ("createContext", "videoSpec", "cliFps")}
        old = await self.store.get(key)
        if old is None:
            raise ValueError("The original control record was not found")
        value = json.loads(old)
        if value["invocation"] != invocation:
            raise ValueError("The original room operation has changed")
        if "retirement" in value:
            if value["retirement"] != receipt:
                raise ValueError("The original retirement proof has changed")
            return retired_create_result(value, key)
        if (
            old != proof.control_raw
            or await self.store.get(value["game_id"]) != proof.route_raw
        ):
            raise ValueError("The original retirement binding has changed")
        value["retirement"] = receipt
        result = retired_create_result(value, key)
        # Do not let an accepted positive receipt expire while WTP still needs
        # to reconcile its durable original journal. Other routing TTLs stay as-is.
        if not await self.store.cas(
            key,
            old,
            json.dumps(value),
            None,
            expected_route=(value["game_id"], proof.route_raw),
        ):
            current = await self.store.get(key)
            if current is not None:
                current = json.loads(current)
                if current == value:
                    return retired_create_result(current, key)
            raise ValueError("The original control record changed during retirement")
        return result

    async def publish_session(self, binding):
        fields = {
            "session_id",
            "game_id",
            "owner_id",
            "owner_epoch",
            "stream_id",
            "epoch",
        }
        if not isinstance(binding, dict) or set(binding) != fields:
            raise ValueError("Invalid SDK session route")
        session = checked_session_id(binding["session_id"])
        checked_id(binding["stream_id"])
        if type(binding["epoch"]) is not int or binding["epoch"] < 0:
            raise ValueError("Invalid SDK session epoch")
        room = await self.lookup(binding["game_id"])
        if room is None:
            raise ValueError("The game was not found")
        self.validate(room)
        if (
            room["owner_id"] != binding["owner_id"]
            or room["owner_epoch"] != binding["owner_epoch"]
        ):
            raise ValueError("The game owner has changed")
        key = "session:" + session
        value = json.dumps(binding, sort_keys=True)
        await self.store.cas(key, None, value, self.max_seconds + 86400)
        if await self.store.get(key) != value:
            raise ValueError("The SDK session route has changed")
        return binding

    async def session_route(self, session_id):
        value = await self.store.get("session:" + checked_session_id(session_id))
        if value is None:
            return None
        binding = json.loads(value)
        room = await self.lookup(binding["game_id"])
        if room is None:
            raise ValueError("The game was not found")
        self.validate(room)
        if (
            room["owner_id"] != binding["owner_id"]
            or room["owner_epoch"] != binding["owner_epoch"]
        ):
            raise ValueError("The game owner has changed")
        return binding

    def validate(self, room):
        now = self.clock()
        if room["status"] == "ended" or room["lease_until"] <= now:
            raise ValueError("The game has ended")

    async def touch(self, game_id, owner_id, epoch, *, end=False):
        checked_id(game_id)
        checked_id(epoch)
        for _ in range(20):
            old = await self.store.get(game_id)
            if old is None:
                raise ValueError("The game was not found")
            room = json.loads(old)
            self.validate(room)
            if room["owner_id"] != owner_id or room["owner_epoch"] not in {None, epoch}:
                raise ValueError("The game owner has changed")
            room.update(
                owner_epoch=epoch,
                status="ended" if end else "active",
                expires_at=None,
                lease_until=self.clock() + self.lease_seconds,
            )
            if await self.store.cas(
                game_id, old, json.dumps(room), 86400 if end else 0
            ):
                return room
        raise ValueError("The game directory is busy")


def require_internal_token(expected, actual):
    if not expected or not hmac.compare_digest("Bearer " + expected, actual):
        raise HTTPException(403, "Room access was denied")


def install_room_routes(app, directory, token):
    @app.post("/v1/whitebox/rooms/resolve")
    async def resolve(body: dict, authorization: str = Header(default="")):
        require_internal_token(token, authorization)
        try:
            return await directory.resolve(
                body["game_id"], body["world_id"], create=body.get("create") is True
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(409, "The game is unavailable") from exc

    @app.post("/v1/whitebox/rooms/touch")
    async def touch(body: dict, authorization: str = Header(default="")):
        require_internal_token(token, authorization)
        try:
            return await directory.touch(
                body["game_id"],
                body["owner_id"],
                body["owner_epoch"],
                end=body.get("end") is True,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(409, "The game is unavailable") from exc

    @app.post("/v1/whitebox/rooms/control-route")
    async def control_route(body: dict, authorization: str = Header(default="")):
        require_internal_token(token, authorization)
        from .room_control import identities, validate_request

        try:
            identities(body["caller"], {"serviceId", "subject"})
            request = body["request"]
            single = "singlePlayerOperation" in request
            if single:
                from .player_control import validate_request as validate_single

                request = validate_single(request)
                if request["singlePlayerOperation"]["caller"] != body["caller"]:
                    raise ValueError()
            else:
                request = validate_request(request)
            if (not single and "createContext" not in request) or (
                body.get("create") is True
                and request["action"]
                not in ({"start", "close"} if single else {"createRoom"})
            ):
                raise ValueError()
            return await directory.control_route(
                body["caller"], request, create=body.get("create") is True
            )
        except (KeyError, TypeError, ValueError):
            raise HTTPException(409, "The room operation is unavailable") from None

    @app.post("/v1/whitebox/rooms/confirm-cleanup")
    async def confirm_cleanup(body: dict, authorization: str = Header(default="")):
        require_internal_token(token, authorization)
        from .room_control import identities, validate_request

        try:
            identities(body["caller"], {"serviceId", "subject"})
            request = validate_request(body["request"])
            if request["action"] != "retireRoom":
                raise ValueError()
            invocation = {
                k: request[k] for k in ("createContext", "videoSpec", "cliFps")
            }
            return await directory.confirm_room_cleanup(
                body["caller"], invocation, request["execution"], body["owner_id"]
            )
        except (KeyError, TypeError, ValueError):
            raise HTTPException(
                409, "The original room cleanup is unavailable"
            ) from None

    @app.post("/v1/whitebox/rooms/lookup")
    async def lookup(body: dict, authorization: str = Header(default="")):
        require_internal_token(token, authorization)
        try:
            return await directory.lookup(body["game_id"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(409, "The game is unavailable") from None

    @app.post("/v1/whitebox/rooms/publish-session")
    async def publish_session(body: dict, authorization: str = Header(default="")):
        require_internal_token(token, authorization)
        try:
            return await directory.publish_session(body)
        except (KeyError, ValueError, TypeError):
            raise HTTPException(409, "The SDK session is unavailable") from None

    @app.post("/v1/whitebox/rooms/session-route")
    async def session_route(body: dict, authorization: str = Header(default="")):
        require_internal_token(token, authorization)
        try:
            return await directory.session_route(body["session_id"])
        except (KeyError, ValueError, TypeError):
            raise HTTPException(409, "The SDK session is unavailable") from None


class HTTPRoomDirectory:
    def __init__(self, client, url, token):
        self.client, self.url, self.token = client, url.rstrip("/"), token

    async def _post(self, name, body):
        response = await self.client.post(
            self.url + "/v1/whitebox/rooms/" + name,
            json=body,
            headers={"Authorization": "Bearer " + self.token},
        )
        response.raise_for_status()
        return response.json()

    async def resolve(self, game_id, world_id, *, create=False):
        return await self._post(
            "resolve", dict(game_id=game_id, world_id=world_id, create=create)
        )

    async def touch(self, game_id, owner_id, epoch, *, end=False):
        return await self._post(
            "touch",
            dict(game_id=game_id, owner_id=owner_id, owner_epoch=epoch, end=end),
        )

    async def control_route(self, caller, body, *, create=False):
        return await self._post(
            "control-route", dict(caller=caller, request=body, create=create)
        )

    async def confirm_room_cleanup(self, caller, invocation, execution, owner_id):
        return await self._post(
            "confirm-cleanup",
            dict(
                caller=caller,
                request=dict(
                    schemaVersion=1,
                    action="retireRoom",
                    execution=execution,
                    **invocation,
                ),
                owner_id=owner_id,
            ),
        )

    async def lookup(self, game_id):
        return await self._post("lookup", dict(game_id=game_id))

    async def publish_session(self, binding):
        return await self._post("publish-session", binding)

    async def session_route(self, session_id):
        return await self._post("session-route", dict(session_id=session_id))
