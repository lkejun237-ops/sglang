# SPDX-License-Identifier: Apache-2.0
"""Manually enrolled Room owners and atomic, bounded game reservations.

Registration is an operator action, never an automatic reset after state loss.
Heartbeats cannot create owners, change their incarnation, or free reservations.
Unclaimed reservations may expire; a claimed CLI needs positive cleanup proof.
"""

import copy
import hashlib
import json
import re
import time
from urllib.parse import urlsplit

from .rooms import checked_id

RETENTION = 86400
LEASE_SECONDS = 30
START_SECONDS = 60
MAX_OWNERS = 128
MAX_GAMES = 4096
MAX_BYTES = 4 << 20


def owner_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("Invalid Room owner identifier")
    return value


def origin(value):
    if not isinstance(value, str):
        raise ValueError("Invalid Room origin")
    p = urlsplit(value)
    if (
        p.scheme not in {"http", "https"}
        or not p.hostname
        or p.username
        or p.password
        or p.query
        or p.fragment
        or p.path not in {"", "/"}
    ):
        raise ValueError("Room owners require a direct HTTP origin")
    return value.rstrip("/")


def count(value, maximum=4096):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError("Invalid Room capacity")
    return value


class RoomRegistry:
    key = "dynamic-room-registry-v1"

    def __init__(self, store, compatibility, *, clock=time.time):
        if not isinstance(compatibility, str) or not 1 <= len(compatibility) <= 128:
            raise ValueError("A Room pool compatibility identity is required")
        self.store, self.compatibility, self.clock = store, compatibility, clock

    async def _read(self):
        raw = await self.store.get(self.key)
        if raw is None:
            raise ValueError("ROOM_REGISTRY_UNAVAILABLE")
        value = json.loads(raw)
        if value.get("compatibility") != self.compatibility or value.get("schema") != 1:
            raise ValueError("ROOM_POOL_MISMATCH")
        return raw, value

    async def _change(self, change, *, expected_route=None):
        for _ in range(32):
            raw, value = await self._read()
            value["owners"] = {
                k: r
                for k, r in value["owners"].items()
                if r.get("removed_at", self.clock()) + RETENTION > self.clock()
            }
            result = change(value)
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
            if len(encoded.encode()) > MAX_BYTES:
                raise ValueError("ROOM_REGISTRY_LIMIT_EXCEEDED")
            if await self.store.cas(
                self.key, raw, encoded, RETENTION, expected_route=expected_route
            ):
                return result
        raise ValueError("ROOM_REGISTRY_BUSY")

    async def enroll(self, descriptor, *, initialize=False):
        expected = {
            "owner_id",
            "owner_epoch",
            "owner_url",
            "compatibility",
            "max_rooms",
            "max_views",
            "media",
            "empty",
        }
        if (
            set(descriptor) != expected
            or descriptor["compatibility"] != self.compatibility
            or descriptor["empty"] is not True
        ):
            raise ValueError("Only an empty compatible Room can be enrolled")
        ident = owner_id(descriptor["owner_id"])
        checked_id(descriptor["owner_epoch"])
        address = origin(descriptor["owner_url"])
        count(descriptor["max_rooms"])
        count(descriptor["max_views"])
        media = descriptor["media"]
        if (
            not isinstance(media, dict)
            or set(media) != {"public_ip", "public_ports"}
            or not isinstance(media["public_ip"], str)
            or not media["public_ip"]
            or not isinstance(media["public_ports"], list)
            or not 1 <= len(media["public_ports"]) <= 128
        ):
            raise ValueError("The Room media endpoint is required")
        for port in media["public_ports"]:
            count(port, 65535)
        if len(set(media["public_ports"])) != len(media["public_ports"]):
            raise ValueError("Duplicate Room media ports")
        if initialize:
            # Explicit first-pool initialization only. Never called by heartbeat.
            await self.store.cas(
                self.key,
                None,
                json.dumps(dict(schema=1, compatibility=self.compatibility, owners={})),
                RETENTION,
            )

        def add(value):
            owners = value["owners"]
            prior = owners.get(ident)
            frozen = {k: v for k, v in descriptor.items() if k != "empty"}
            frozen["owner_url"] = address
            if prior is not None:
                if prior["descriptor"] != frozen:
                    raise ValueError("ROOM_OWNER_CONFLICT")
                return self._public(ident, prior)
            if len(owners) >= MAX_OWNERS:
                raise ValueError("ROOM_OWNER_LIMIT_EXCEEDED")
            for other in owners.values():
                d = other["descriptor"]
                if d["owner_url"] == address or (
                    d["media"]["public_ip"] == media["public_ip"]
                    and set(d["media"]["public_ports"]) & set(media["public_ports"])
                ):
                    raise ValueError("ROOM_ENDPOINT_CONFLICT")
            owners[ident] = dict(
                descriptor=frozen,
                state="ready",
                lease_until=self.clock() + LEASE_SECONDS,
                games={},
            )
            return self._public(ident, owners[ident])

        return await self._change(add)

    def _owner(self, value, ident, epoch=None):
        row = value["owners"].get(owner_id(ident))
        if (
            row is None
            or epoch is not None
            and row["descriptor"]["owner_epoch"] != epoch
        ):
            raise ValueError("ROOM_OWNER_UNAVAILABLE")
        return row

    def _public(self, ident, row):
        return dict(
            row["descriptor"],
            state=row["state"],
            lease_until=row["lease_until"],
            reserved_rooms=len(row["games"]),
            reserved_views=sum(g["views"] for g in row["games"].values()),
        )

    async def inspect(self, ident=None, epoch=None):
        _, value = await self._read()
        if ident is None:
            return [self._public(k, r) for k, r in value["owners"].items()]
        return self._public(ident, self._owner(value, ident, epoch))

    async def heartbeat(self, ident, epoch):
        checked_id(epoch)

        def renew(value):
            row = self._owner(value, ident, epoch)
            if row["state"] == "removed":
                raise ValueError("ROOM_OWNER_REMOVED")
            row["lease_until"] = self.clock() + LEASE_SECONDS
            return self._public(ident, row)

        return await self._change(renew)

    async def drain(self, ident, epoch):
        def stop(value):
            row = self._owner(value, ident, epoch)
            if row["state"] != "removed":
                row["state"] = "draining"
            return self._public(ident, row)

        return await self._change(stop)

    async def remove(self, ident, epoch):
        def remove(value):
            row = self._owner(value, ident, epoch)
            if row["state"] == "removed":
                return self._public(ident, row)
            if row["state"] != "draining" or row["games"]:
                raise ValueError("The original Room has not drained")
            row.update(state="removed", removed_at=self.clock(), lease_until=0)
            return self._public(ident, row)

        return await self._change(remove)

    async def reserve(self, game_id, world_id, views, lease_seconds):
        checked_id(game_id)
        count(views, 128)
        # The route is independently authoritative. A delayed contender must not
        # reserve again after another caller created/ended the same game.
        existing = await self.store.get(game_id)
        if existing is not None:
            return json.loads(existing)

        def allocate(value):
            now = self.clock()
            candidates = []
            for ident, row in value["owners"].items():
                if game_id in row["games"]:
                    game = row["games"][game_id]
                    if game["world_id"] != world_id or game["views"] != views:
                        raise ValueError("ROOM_RESERVATION_CONFLICT")
                    return copy.deepcopy(game["route"])
                # Once claimed, lack of a heartbeat is never cleanup evidence.
                row["games"] = {
                    k: g
                    for k, g in row["games"].items()
                    if g["claimed"] or g["claim_until"] > now
                }
                d = row["descriptor"]
                used = sum(g["views"] for g in row["games"].values())
                if (
                    row["state"] == "ready"
                    and row["lease_until"] > now
                    and len(row["games"]) < d["max_rooms"]
                    and used + views <= d["max_views"]
                ):
                    candidates.append(
                        (
                            used / d["max_views"],
                            len(row["games"]),
                            hashlib.sha256((game_id + ident).encode()).hexdigest(),
                            ident,
                        )
                    )
            if sum(len(r["games"]) for r in value["owners"].values()) >= MAX_GAMES:
                raise ValueError("ROOM_REGISTRY_LIMIT_EXCEEDED")
            if not candidates:
                raise ValueError("ROOM_CAPACITY_EXHAUSTED")
            ident = min(candidates)[-1]
            row = value["owners"][ident]
            d = row["descriptor"]
            route = dict(
                game_id=game_id,
                world_id=world_id,
                owner_id=ident,
                owner_url=d["owner_url"],
                owner_epoch=d["owner_epoch"],
                status="starting",
                created_at=now,
                expires_at=None,
                lease_until=now + lease_seconds,
                dynamic_owner=True,
            )
            row["games"][game_id] = dict(
                world_id=world_id,
                views=views,
                claimed=False,
                claim_until=now + START_SECONDS,
                route=route,
            )
            return copy.deepcopy(route)

        try:
            return await self._change(allocate, expected_route=(game_id, None))
        except ValueError:
            existing = await self.store.get(game_id)
            if existing is not None:
                return json.loads(existing)
            raise

    async def original_route(self, game_id, ident, epoch):
        _, value = await self._read()
        row = self._owner(value, ident, epoch)
        game = row["games"].get(checked_id(game_id))
        if game is None:
            raise ValueError("The original Room reservation is unavailable")
        return copy.deepcopy(game["route"])

    async def claim(self, game_id, ident, epoch):
        def claim(value):
            row = self._owner(value, ident, epoch)
            game = row["games"].get(game_id)
            if (
                game is None
                or not game["claimed"]
                and (
                    game["claim_until"] <= self.clock()
                    or row["lease_until"] <= self.clock()
                    or row["state"] != "ready"
                )
            ):
                raise ValueError("ROOM_RESERVATION_EXPIRED")
            game["claimed"] = True

        await self._change(claim)

    async def finish(self, route):
        if route["status"] != "ended":
            raise ValueError("Room cleanup must be confirmed")

        def finish(value):
            row = self._owner(value, route["owner_id"], route["owner_epoch"])
            row["games"].pop(route["game_id"], None)

        await self._change(finish, expected_route=(route["game_id"], json.dumps(route)))
