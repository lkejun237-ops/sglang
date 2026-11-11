# SPDX-License-Identifier: Apache-2.0
"""Per-record optimistic transactions. No global JSON state or remote work in locks."""

import asyncio
import copy
import time
from dataclasses import dataclass
from uuid import uuid4

from .contracts import ROLES, PoolError, canonical, digest, identifier, manifest, number


@dataclass(frozen=True)
class IndexChange:
    member: str
    score: float | None = None


@dataclass(frozen=True)
class Expiring:
    value: object
    seconds: int


class MemoryRecords:
    def __init__(self):
        self.values, self.lock = {}, asyncio.Lock()
        self.indices, self.expirations = {}, {}

    async def read(self, keys):
        async with self.lock:
            for k in keys:
                if self.expirations.get(k, float("inf")) <= time.time():
                    self.values.pop(k, None)
            return {k: copy.deepcopy(self.values.get(k)) for k in keys}

    async def keys(self, prefix):
        async with self.lock:
            return [k for k in self.values if k.startswith(prefix)]

    async def transact(self, keys, mutate):
        async with self.lock:
            for k in keys:
                if self.expirations.get(k, float("inf")) <= time.time():
                    self.values.pop(k, None)
            records = {k: copy.deepcopy(self.values.get(k)) for k in keys}
            result, writes = mutate(records)
            for key, value in writes.items():
                if isinstance(value, IndexChange):
                    index = self.indices.setdefault(key, {})
                    if value.score is None:
                        index.pop(value.member, None)
                    else:
                        index[value.member] = value.score
                elif isinstance(value, Expiring):
                    self.values[key] = copy.deepcopy(value.value)
                    self.expirations[key] = time.time() + value.seconds
                elif value is None:
                    self.values.pop(key, None)
                    self.expirations.pop(key, None)
                else:
                    self.values[key] = copy.deepcopy(value)
                    self.expirations.pop(key, None)
            return copy.deepcopy(result)

    async def index(self, key, *, minimum=0, limit=None):
        async with self.lock:
            values = self.indices.setdefault(key, {})
            for member, score in list(values.items()):
                if score <= minimum:
                    values.pop(member, None)
            result = list(values)
            if limit is not None:
                import random

                result = random.sample(result, min(limit, len(result)))
            return result, len(values)


class RedisRecords:
    """Pin the Redis process incarnation on the SAME watched connection as writes.

    A restart or promotion fails closed even when an old OPEN key survived.
    expected_run_id is supplied by deployment control, never learned on reconnect.
    """

    def __init__(self, client, *, expected_run_id, retries=16):
        self.client, self.expected_run_id, self.retries = (
            client,
            identifier(expected_run_id),
            retries,
        )

    async def read(self, keys):
        import json

        values = await self.client.mget(keys)
        return {k: json.loads(v) if v else None for k, v in zip(keys, values)}

    async def keys(self, prefix):
        return [k async for k in self.client.scan_iter(match=prefix + "*", count=128)]

    async def transact(self, keys, mutate):
        import json

        from redis.exceptions import WatchError

        for attempt in range(self.retries):
            async with self.client.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(*keys)
                    info = await pipe.info("server")
                    role = await pipe.execute_command("ROLE")
                    if info.get("run_id") != self.expected_run_id or role[0] not in (
                        "master",
                        b"master",
                    ):
                        raise PoolError("CONTROL_STORE_CHANGED", 503)
                    values = await pipe.mget(keys)
                    records = {
                        k: json.loads(v) if v else None for k, v in zip(keys, values)
                    }
                    result, writes = mutate(records)
                    pipe.multi()
                    for key, value in writes.items():
                        if isinstance(value, IndexChange):
                            if value.score is None:
                                pipe.zrem(key, value.member)
                            else:
                                pipe.zadd(key, {value.member: value.score})
                        elif isinstance(value, Expiring):
                            pipe.set(key, canonical(value.value), ex=value.seconds)
                        elif value is None:
                            pipe.delete(key)
                        else:
                            pipe.set(key, canonical(value))
                    await pipe.execute()
                    return result
                except WatchError:
                    await asyncio.sleep(min(0.001 * (attempt + 1), 0.025))
        raise PoolError("CONTROL_STORE_BUSY", 503)

    async def index(self, key, *, minimum=0, limit=None):
        async with self.client.pipeline(transaction=False) as pipe:
            if minimum > 0:
                pipe.zremrangebyscore(key, "-inf", minimum)
            pipe.zcard(key)
            if limit is None:
                pipe.zrange(key, 0, -1)
            else:
                pipe.zrandmember(key, limit)
            values = await pipe.execute()
        count, members = values[-2:]
        return members, count


class RolePoolStore:
    def __init__(
        self,
        records,
        pool_id,
        control_epoch,
        *,
        clock=time.time,
        worker_ttl=10,
        candidate_limit=64,
        admission_records=None,
    ):
        self.db, self.pool_id, self.epoch = (
            records,
            identifier(pool_id),
            identifier(control_epoch),
        )
        from .admission import AdmissionGate

        self.admissions = AdmissionGate(admission_records or records, self.epoch)
        self.prefix = "{role-pool:" + digest(pool_id)[:24] + "}:"
        self.clock, self.worker_ttl, self.candidate_limit = (
            clock,
            worker_ttl,
            candidate_limit,
        )

    def key(self, kind, identity=""):
        return self.prefix + kind + ":" + digest(identity)

    async def initialize(self):
        """Explicit administrative bootstrap only; not called by server startup."""
        key = self.key("authority")

        def create(records):
            if records[key] is not None:
                raise PoolError("AUTHORITY_ALREADY_EXISTS")
            return None, {key: dict(epoch=self.epoch, open=True)}

        await self.db.transact([key], create)

    async def _tx(self, keys, mutate):
        authority = self.key("authority")

        def checked(records):
            if records[authority] != dict(epoch=self.epoch, open=True):
                raise PoolError("ADMISSION_GATE_CLOSED", 503)
            return mutate(records)

        return await self.db.transact(list(dict.fromkeys([authority, *keys])), checked)

    def _ready_indices(self, process, writes):
        m = process["manifest"]
        for profile, costs in m["profiles"].items():
            key = self.key(
                "ready", [m["role"], m["compatibility"], m["network_domain"], profile]
            )
            valid = (
                process["state"] == "ready"
                and process.get("active_services", 0) < len(m["services"])
                and all(
                    process["used"][k] + v <= m["capacity"][k] * (1 + 1e-10)
                    for k, v in costs.items()
                )
            )
            writes[key] = IndexChange(
                m["process_id"], process["seen_at"] if valid else None
            )

    async def register(self, payload):
        value = manifest(payload)
        if value["control_epoch"] != self.epoch or value["pool_id"] != self.pool_id:
            raise PoolError("CONTROL_IDENTITY_MISMATCH")
        pk, index = (
            self.key("process", value["process_id"]),
            self.key("processes", value["role"]),
        )
        sk = [self.key("service", s["service_id"]) for s in value["services"]]
        hk = [self.key("hardware", h) for h in value["hardware_ids"]]

        def write(r):
            old = r[pk]
            if old and old["state"] != "retired":
                if old["manifest"] != value:
                    raise PoolError("PROCESS_IDENTITY_CHANGED")
                return old, {}
            for key in hk:
                if r[key] is not None:
                    raise PoolError("RESOURCE_ALREADY_REGISTERED")
            for key in sk:
                if r[key] is not None and not r[key].get("retired"):
                    raise PoolError("RESOURCE_ALREADY_REGISTERED")
            proc = dict(
                manifest=value,
                seen_at=self.clock(),
                sequence=-1,
                state="warming",
                used={k: 0.0 for k in value["capacity"]},
                observations={},
            )
            processes = r[index] or []
            writes = {pk: proc, index: sorted(set(processes + [value["process_id"]]))}
            for key, service in zip(sk, value["services"]):
                writes[key] = dict(
                    descriptor=service,
                    boot_epoch=value["boot_epoch"],
                    allocation_id=None,
                )
            for key in hk:
                writes[key] = dict(
                    process_id=value["process_id"], boot_epoch=value["boot_epoch"]
                )
            return proc, writes

        return await self._tx([pk, index, *sk, *hk], write)

    async def heartbeat(self, process_id, boot_epoch, *, sequence, state, observations):
        pk = self.key("process", process_id)
        if (
            type(sequence) is not int
            or sequence < 0
            or state not in {"ready", "warming", "draining", "failed"}
        ):
            raise PoolError("INVALID_HEARTBEAT", 422)

        def update(r):
            proc = r[pk]
            if not proc or proc["manifest"]["boot_epoch"] != boot_epoch:
                raise PoolError("PROCESS_EPOCH_MISMATCH")
            if sequence <= proc["sequence"]:
                return proc, {}
            allowed = {s["service_id"] for s in proc["manifest"]["services"]}
            if set(observations) != allowed:
                raise PoolError("INCOMPLETE_SERVICE_OBSERVATION", 422)
            proc.update(
                sequence=sequence, seen_at=self.clock(), observations=observations
            )
            # A reporter cannot un-drain an operator-drained process.
            if proc["state"] not in {
                "draining",
                "drained",
                "retired",
                "stopped",
                "failed",
            }:
                proc["state"] = state
            writes = {pk: proc}
            self._ready_indices(proc, writes)
            return proc, writes

        return await self._tx([pk], update)

    async def drain(self, process_id, boot_epoch):
        pk = self.key("process", process_id)

        def update(r):
            p = r[pk]
            if not p or p["manifest"]["boot_epoch"] != boot_epoch:
                raise PoolError("PROCESS_EPOCH_MISMATCH")
            if p["state"] in {"drained", "stopped", "retired"}:
                return p, {}
            p["state"] = "draining"
            writes = {pk: p}
            self._ready_indices(p, writes)
            return p, writes

        return await self._tx([pk], update)

    def eligible(self, proc, profile, compatibility, network):
        if (
            not proc
            or proc["state"] != "ready"
            or self.clock() - proc["seen_at"] >= self.worker_ttl
        ):
            return False
        m = proc["manifest"]
        if (
            profile not in m["profiles"]
            or m["compatibility"] != compatibility
            or m["network_domain"] != network
        ):
            return False
        return all(
            proc["used"][k] + v <= m["capacity"][k] * (1 + 1e-10)
            for k, v in m["profiles"][profile].items()
        )

    async def candidates(self, profile, compatibility, network, admission_id):
        ids = []
        truncated = False
        for role in ROLES:
            key = self.key("ready", [role, compatibility, network, profile])
            selected, total = await self.db.index(
                key, minimum=self.clock() - self.worker_ttl, limit=self.candidate_limit
            )
            ids.extend(selected)
            truncated |= total > self.candidate_limit
        keys = [self.key("process", p) for p in ids]
        rows = await self.db.read(keys) if keys else {}
        service_keys = [
            self.key("service", s["service_id"])
            for p in rows.values()
            if p
            for s in p["manifest"]["services"]
        ]
        services = await self.db.read(service_keys) if service_keys else {}
        choices = {role: [] for role in ROLES}
        for proc in rows.values():
            if not self.eligible(proc, profile, compatibility, network):
                continue
            m = proc["manifest"]
            for service in m["services"]:
                observation = proc["observations"].get(service["service_id"], {})
                registered = services[self.key("service", service["service_id"])]
                if (
                    registered
                    and registered["allocation_id"] is None
                    and observation.get("state") == "idle"
                    and not observation.get("restart_required")
                ):
                    score = max(
                        proc["used"][k] / m["capacity"][k] for k in m["capacity"]
                    )
                    choices[m["role"]].append((score, proc, service))
        return {
            role: sorted(
                values,
                key=lambda item: (
                    item[0],
                    digest([admission_id, item[2]["service_id"]]),
                ),
            )[: self.candidate_limit]
            for role, values in choices.items()
        }, truncated

    async def acquire(self, **request):
        fields = {
            "admission_id",
            "user_id",
            "player_id",
            "generation_id",
            "profile",
            "compatibility",
            "network_domain",
            "ttl",
            "created_at",
        }
        optional = {"operation"} if request.get("operation") == "prefill" else set()
        if set(request) != fields | optional:
            raise PoolError("INVALID_ADMISSION", 422)
        for key in fields - {"ttl", "created_at"}:
            identifier(request[key])
        request["ttl"] = number(request["ttl"])
        request["created_at"] = number(request["created_at"])
        if request["ttl"] > 86400:
            raise PoolError("INVALID_PLAYER_TTL", 422)
        key = self.key("admission", request["admission_id"])
        existing = (await self.db.read([key]))[key]
        if existing is not None:
            if existing["intent"] != request:
                raise PoolError("ADMISSION_IDENTITY_CHANGED")
            if existing["state"] == "released":
                return existing
        elif not self.clock() - 480 <= request["created_at"] <= self.clock() + 5:
            raise PoolError("ADMISSION_EXPIRED", 422)
        await self.admissions.claim(request, self.pool_id)
        try:
            value = await self._acquire_pool(**request)
        except PoolError as error:
            if error.code in {
                "CAPACITY_EXHAUSTED",
                "CAPACITY_BUSY",
                "PROFILE_CONTRACT_MISMATCH",
                "INVALID_PLAYER_TTL",
                "INVALID_IDENTITY",
                "ADMISSION_CANCELLED",
            }:
                # A delayed pool claim cannot dispatch before binding this gate.
                await self.admissions.release(request, self.pool_id)
            raise
        await self.admissions.bind(request, self.pool_id, value["allocation_id"])
        return value

    async def _acquire_pool(
        self,
        *,
        admission_id,
        user_id,
        player_id,
        generation_id,
        profile,
        compatibility,
        network_domain,
        ttl,
        created_at,
        operation=None,
    ):
        for v in (
            admission_id,
            user_id,
            player_id,
            profile,
            compatibility,
            network_domain,
        ):
            identifier(v)
        ttl = number(ttl)
        if ttl > 86400:
            raise PoolError("INVALID_PLAYER_TTL", 422)
        intent = dict(
            admission_id=admission_id,
            user_id=user_id,
            player_id=player_id,
            generation_id=generation_id,
            profile=profile,
            compatibility=compatibility,
            network_domain=network_domain,
            ttl=ttl,
            created_at=created_at,
        )
        if operation is not None:
            if operation != "prefill":
                raise PoolError("INVALID_ADMISSION", 422)
            intent["operation"] = operation
        ak, uk = self.key("admission", admission_id), self.key("user", user_id)
        ck = self.key("cancel", admission_id)
        existing = (await self.db.read([ak]))[ak]
        if existing:
            if existing["intent"] != intent:
                raise PoolError("ADMISSION_IDENTITY_CHANGED")
            return existing
        choices, truncated = await self.candidates(
            profile, compatibility, network_domain, admission_id
        )
        if any(not choices[r] for r in ROLES):
            if truncated:
                raise PoolError("CAPACITY_BUSY", 503)
            raise PoolError("CAPACITY_EXHAUSTED")
        # Try bounded complete tuples; conflicts never leave partial ownership.
        for attempt in range(
            min(self.candidate_limit, max(map(len, choices.values())))
        ):
            selected = {
                role: choices[role][attempt % len(choices[role])] for role in ROLES
            }
            pks = {
                r: self.key("process", v[1]["manifest"]["process_id"])
                for r, v in selected.items()
            }
            sks = {
                r: self.key("service", v[2]["service_id"]) for r, v in selected.items()
            }

            def claim(rows):
                if rows[ck] is not None:
                    raise PoolError("ADMISSION_CANCELLED")
                if rows[ak]:
                    if rows[ak]["intent"] != intent:
                        raise PoolError("ADMISSION_IDENTITY_CHANGED")
                    return rows[ak], {}
                if rows[uk] is not None:
                    raise PoolError("USER_SESSION_LIMIT")
                for role in ROLES:
                    p, s = rows[pks[role]], rows[sks[role]]
                    if (
                        not self.eligible(p, profile, compatibility, network_domain)
                        or s["allocation_id"] is not None
                        or s["boot_epoch"]
                        != selected[role][1]["manifest"]["boot_epoch"]
                    ):
                        raise PoolError("CAPACITY_CONFLICT")
                limits = rows[pks["dit"]]["manifest"]["profile_limits"][profile]
                if any(
                    rows[pks[r]]["manifest"]["profile_limits"][profile] != limits
                    for r in ROLES
                ):
                    raise PoolError("PROFILE_CONTRACT_MISMATCH")
                now, allocation = self.clock(), uuid4().hex
                value = dict(
                    intent=intent,
                    profile_limits=limits,
                    allocation_id=allocation,
                    group_id=uuid4().hex,
                    control_epoch=self.epoch,
                    state="held",
                    admitted_at=now,
                    player_expires_at=now + ttl,
                    revision=0,
                    executor=None,
                    executor_until=0,
                    topology={r: selected[r][2] for r in ROLES},
                    boot_epochs={
                        r: rows[pks[r]]["manifest"]["boot_epoch"] for r in ROLES
                    },
                    roles={r: "held" for r in ROLES},
                    sdk_cleanup=None,
                )
                writes = {
                    self.key("active"): IndexChange(admission_id, now),
                    ak: value,
                    uk: dict(admission_id=admission_id, allocation_id=allocation),
                }
                for role in ROLES:
                    p, s = rows[pks[role]], rows[sks[role]]
                    for k, cost in p["manifest"]["profiles"][profile].items():
                        p["used"][k] += cost
                    s["allocation_id"] = allocation
                    p["active_services"] = p.get("active_services", 0) + 1
                    writes[pks[role]], writes[sks[role]] = p, s
                    self._ready_indices(p, writes)
                return value, writes

            try:
                return await self._tx([ak, uk, ck, *pks.values(), *sks.values()], claim)
            except PoolError as exc:
                if exc.code != "CAPACITY_CONFLICT":
                    raise
        raise PoolError("CAPACITY_BUSY", 503)

    async def get(self, admission_id):
        key = self.key("admission", admission_id)
        value = (await self.db.read([key]))[key]
        if value is None:
            raise PoolError("ADMISSION_NOT_FOUND", 404)
        return value

    async def update(self, admission_id, allocation_id, mutate):
        key = self.key("admission", admission_id)

        def change(rows):
            value = rows[key]
            if value is None or value["allocation_id"] != allocation_id:
                raise PoolError("ALLOCATION_MISMATCH")
            result = mutate(value)
            value["revision"] += 1
            writes = {key: value}
            if value["state"] in {"released", "awaiting_sdk"}:
                writes[self.key("active")] = IndexChange(
                    admission_id,
                    self.clock()
                    if value["state"] == "awaiting_sdk"
                    and value.get("sdk_cleanup_pending")
                    else None,
                )
            return result if result is not None else value, writes

        return await self._tx([key], change)

    async def release_role(self, admission_id, allocation_id, role, receipt):
        current = await self.get(admission_id)
        if role not in ROLES:
            raise PoolError("INVALID_ROLE")
        service = current["topology"][role]
        ak, sk, pk = (
            self.key("admission", admission_id),
            self.key("service", service["service_id"]),
            self.key("process", service["worker_id"]),
        )

        def release(rows):
            a, s, p = rows[ak], rows[sk], rows[pk]
            if a["allocation_id"] != allocation_id:
                raise PoolError("ALLOCATION_MISMATCH")
            expected = dict(
                allocation_id=allocation_id,
                worker_group_id=a["group_id"],
                service_id=service["service_id"],
                boot_epoch=a["boot_epochs"][role],
                state="released",
            )
            if receipt != expected:
                raise PoolError("RELEASE_UNCONFIRMED")
            if a["roles"][role] == "released":
                return a, {}
            if (
                s["allocation_id"] != allocation_id
                or p["manifest"]["boot_epoch"] != expected["boot_epoch"]
            ):
                raise PoolError("RELEASE_OWNER_MISMATCH")
            for k, cost in p["manifest"]["profiles"][a["intent"]["profile"]].items():
                p["used"][k] = max(0.0, p["used"][k] - cost)
            s["allocation_id"] = None
            if p.get("active_services", 0) < 1:
                raise PoolError("CAPACITY_STATE_CORRUPT")
            p["active_services"] -= 1
            if p["active_services"] == 0:
                p["used"] = {k: 0.0 for k in p["used"]}
            a["roles"][role] = "released"
            writes = {ak: a, sk: s, pk: p}
            self._ready_indices(p, writes)
            return a, writes

        return await self._tx([ak, sk, pk], release)

    @staticmethod
    def _validated_sdk_cleanup(a, allocation_id, receipt):
        from types import SimpleNamespace

        from ..player_lifecycle import validate_sdk_cleanup

        if a["intent"].get("operation") == "prefill":
            raise PoolError("PREFILL_HAS_NO_SDK_OWNER")
        if (
            not isinstance(receipt, dict)
            or receipt.get("player_id") != a["intent"]["player_id"]
        ):
            raise PoolError("SDK_RELEASE_UNCONFIRMED")
        parent = SimpleNamespace(
            token=allocation_id,
            session_id=a["intent"]["player_id"],
            generation_id=a["intent"]["generation_id"],
        )
        validated = validate_sdk_cleanup(parent, receipt)
        return {**validated, "player_id": a["intent"]["player_id"]}

    async def record_sdk_cleanup(self, admission_id, allocation_id, receipt):
        """Keep positive SDK evidence even when model cleanup outlives the caller."""
        key = self.key("admission", admission_id)

        def remember(rows):
            a = rows[key]
            if a is None or a["allocation_id"] != allocation_id:
                raise PoolError("ALLOCATION_MISMATCH")
            validated = self._validated_sdk_cleanup(a, allocation_id, receipt)
            previous = a.get("sdk_cleanup_pending") or (
                {**a["sdk_cleanup"], "player_id": a["intent"]["player_id"]}
                if a.get("sdk_cleanup")
                else None
            )
            if previous is not None and previous != validated:
                raise PoolError("SDK_CLEANUP_IDENTITY_CHANGED")
            if a["state"] == "released":
                return a, {}  # Preserve terminal retention on duplicate close.
            a["sdk_cleanup_pending"] = validated
            a["revision"] += 1
            return a, {
                key: a,
                self.key("active"): IndexChange(admission_id, self.clock()),
            }

        return await self._tx([key], remember)

    async def finish(self, admission_id, allocation_id, sdk_cleanup):
        ak, current = self.key("admission", admission_id), await self.get(admission_id)
        uk = self.key("user", current["intent"]["user_id"])

        def finish(rows):
            a = rows[ak]
            if a["allocation_id"] != allocation_id or any(
                s != "released" for s in a["roles"].values()
            ):
                raise PoolError("RELEASE_UNCONFIRMED")
            if a["intent"].get("operation") == "prefill":
                # This immutable admission never created a Room/SDK member.
                # All three native role receipts above are still mandatory.
                if sdk_cleanup is not None or a.get("sdk_cleanup_pending") is not None:
                    raise PoolError("PREFILL_HAS_NO_SDK_OWNER")
                validated = None
            else:
                validated = self._validated_sdk_cleanup(a, allocation_id, sdk_cleanup)
                pending = a.get("sdk_cleanup_pending")
                if pending is not None and pending != validated:
                    raise PoolError("SDK_CLEANUP_IDENTITY_CHANGED")
                validated.pop("player_id")
            a.update(state="released", sdk_cleanup=validated)
            writes = {
                ak: Expiring(a, 86400),
                self.key("active"): IndexChange(admission_id),
            }
            if rows[uk] and rows[uk]["allocation_id"] == allocation_id:
                writes[uk] = None
            return a, writes

        value = await self._tx([ak, uk], finish)
        await self.admissions.release(value["intent"], self.pool_id, allocation_id)
        return value

    async def retire_process(self, process_id, boot_epoch, stop_receipt):
        identifier(stop_receipt)
        pk = self.key("process", process_id)
        p = (await self.db.read([pk]))[pk]
        if not p:
            raise PoolError("PROCESS_NOT_FOUND", 404)
        m = p["manifest"]
        sk = [self.key("service", s["service_id"]) for s in m["services"]]
        hk = [self.key("hardware", h) for h in m["hardware_ids"]]
        index = self.key("processes", m["role"])

        def retire(rows):
            p = rows[pk]
            if not p or p["manifest"]["boot_epoch"] != boot_epoch:
                raise PoolError("PROCESS_NOT_DRAINING")
            if p["state"] == "retired":
                if p.get("stop_receipt") != stop_receipt:
                    raise PoolError("STOP_EVIDENCE_CHANGED")
                return p, {}
            if p["state"] not in {"drained", "stopped"}:
                raise PoolError("PROCESS_NOT_DRAINING")
            if any(p["used"].values()) or any(
                rows[k]["allocation_id"] is not None for k in sk
            ):
                raise PoolError("PROCESS_STILL_OWNED")
            if p["state"] == "drained" and not p.get("drained_at"):
                raise PoolError("PROCESS_RELEASE_UNCONFIRMED")
            if (
                p["state"] == "stopped"
                and p.get("exit_evidence", {}).get("stop_receipt") != stop_receipt
            ):
                raise PoolError("STOP_EVIDENCE_CHANGED")
            p["state"] = "retired"
            p["stop_receipt"] = stop_receipt
            writes = {
                pk: Expiring(p, 86400),
                index: [v for v in rows[index] if v != process_id],
            }
            for k in sk:
                rows[k]["retired"] = True
                writes[k] = Expiring(rows[k], 86400)
            for k in hk:
                writes[k] = None
            return p, writes

        return await self._tx([pk, index, *sk, *hk], retire)

    async def active_assignments(self):
        ids, _ = await self.db.index(self.key("active"))
        keys = [self.key("admission", v) for v in ids]
        values = await self.db.read(keys) if keys else {}
        return [
            v
            for v in values.values()
            if v
            and v["state"] != "released"
            and (v["state"] != "awaiting_sdk" or v.get("sdk_cleanup_pending"))
        ]

    async def cancel_intent(self, admission_id, user_id):
        identifier(admission_id)
        identifier(user_id)
        ak, ck = self.key("admission", admission_id), self.key("cancel", admission_id)

        def cancel(rows):
            a = rows[ak]
            if a is not None and a["intent"]["user_id"] != user_id:
                raise PoolError("ADMISSION_OWNER_MISMATCH")
            if rows[ck] is not None and rows[ck]["user_id"] != user_id:
                raise PoolError("ADMISSION_OWNER_MISMATCH")
            return a, {ck: Expiring(dict(user_id=user_id), 600)}

        return await self._tx([ak, ck], cancel)

    async def capacity(self, profile, compatibility, network_domain):
        import math

        indices = [self.key("processes", r) for r in ROLES]
        registry = await self.db.read(indices)
        ids = [p for values in registry.values() for p in (values or [])]
        keys = [self.key("process", p) for p in ids]
        processes = (await self.db.read(keys)).values() if keys else []
        rows = list(processes)
        service_keys = [
            self.key("service", s["service_id"])
            for p in rows
            if p
            for s in p["manifest"]["services"]
        ]
        services = await self.db.read(service_keys) if service_keys else {}
        roles = {
            r: dict(workers=0, active_sessions=0, free_slots=0, draining_workers=0)
            for r in ROLES
        }
        classes = {}
        for p in rows:
            m = p["manifest"]
            if (
                profile not in m["profiles"]
                or m["compatibility"] != compatibility
                or m["network_domain"] != network_domain
            ):
                continue
            summary = roles[m["role"]]
            owned = sum(
                services[self.key("service", s["service_id"])]["allocation_id"]
                is not None
                for s in m["services"]
            )
            summary["active_sessions"] += owned
            if p["state"] == "draining":
                summary["draining_workers"] += 1
            if p["state"] != "ready" or self.clock() - p["seen_at"] >= self.worker_ttl:
                continue
            summary["workers"] += 1
            idle = sum(
                services[self.key("service", s["service_id"])]["allocation_id"] is None
                and p["observations"].get(s["service_id"], {}).get("state") == "idle"
                for s in m["services"]
            )
            slots = min(
                [idle]
                + [
                    max(0, math.floor((m["capacity"][k] - p["used"][k] + 1e-9) / v))
                    for k, v in m["profiles"][profile].items()
                ]
            )
            summary["free_slots"] += slots
            key = digest(m["profile_limits"][profile])
            classes.setdefault(key, {r: 0 for r in ROLES})[m["role"]] += slots
        free = sum(min(counts.values()) for counts in classes.values())
        used = max(v["active_sessions"] for v in roles.values())
        return dict(
            backend="role_pool",
            capacity_unit="player_group",
            profile=profile,
            observed_at=self.clock(),
            roles={
                "model_group": dict(
                    active_sessions=used,
                    free_slots=free,
                    draining_workers=sum(v["draining_workers"] for v in roles.values()),
                    waiting_sessions=0,
                    queued_sessions=0,
                )
            },
            role_capacity=roles,
        )

    async def process_status(self, process_id, boot_epoch):
        pk = self.key("process", process_id)
        p = (await self.db.read([pk]))[pk]
        if p is None or p["manifest"]["boot_epoch"] != boot_epoch:
            raise PoolError("PROCESS_EPOCH_MISMATCH")
        sk = [self.key("service", s["service_id"]) for s in p["manifest"]["services"]]

        def inspect(rows):
            p = rows[pk]
            if p["manifest"]["boot_epoch"] != boot_epoch:
                raise PoolError("PROCESS_EPOCH_MISMATCH")
            if (
                p["state"] == "draining"
                and self.clock() - p["seen_at"] < self.worker_ttl
                and not any(p["used"].values())
                and all(rows[k]["allocation_id"] is None for k in sk)
                and len(p["observations"]) == len(sk)
                and all(
                    v.get("state") == "idle"
                    and v.get("draining") is True
                    and v.get("owner") is None
                    for v in p["observations"].values()
                )
            ):
                p.update(state="drained", drained_at=self.clock())
            return dict(process=p, safe_to_stop=p["state"] in {"drained", "retired"}), {
                pk: p
            }

        return await self._tx([pk, *sk], inspect)

    async def confirm_stopped(self, process_id, boot_epoch, stop_receipt, operator_id):
        """Operator attestation, never a heartbeat/TTL inference of GPU death."""
        identifier(stop_receipt)
        identifier(operator_id)
        pk = self.key("process", process_id)

        def stopped(rows):
            p = rows[pk]
            if not p or p["manifest"]["boot_epoch"] != boot_epoch:
                raise PoolError("PROCESS_EPOCH_MISMATCH")
            if p.get("exit_evidence") and p["exit_evidence"] != {
                "stop_receipt": stop_receipt,
                "operator_id": operator_id,
            }:
                raise PoolError("STOP_EVIDENCE_CHANGED")
            if p["state"] == "retired":
                return p, {}
            if (
                p["state"] not in {"failed", "draining", "drained", "stopped"}
                and self.clock() - p["seen_at"] < self.worker_ttl
            ):
                raise PoolError("PROCESS_STILL_LIVE")
            p.update(
                state="stopped",
                exit_evidence={
                    "stop_receipt": stop_receipt,
                    "operator_id": operator_id,
                },
            )
            writes = {pk: p}
            self._ready_indices(p, writes)
            return p, writes

        p = await self._tx([pk], stopped)
        # Manual recovery is deliberately off the admission hot path. Match
        # original epochs, and never manufacture a native cleanup ACK.
        for a in await self.active_assignments():
            for role, s in a["topology"].items():
                if (
                    s["worker_id"] != process_id
                    or a["boot_epochs"][role] != boot_epoch
                    or a["roles"][role] == "released"
                ):
                    continue
                await self._release_exited_role(a, role, p["exit_evidence"])
        return p

    async def _release_exited_role(self, a, role, evidence):
        service = a["topology"][role]
        pk = self.key("process", service["worker_id"])
        sk = self.key("service", service["service_id"])
        ak = self.key("admission", a["intent"]["admission_id"])

        def release(rows):
            p, s, current = rows[pk], rows[sk], rows[ak]
            if (
                current["allocation_id"] != a["allocation_id"]
                or p["state"] != "stopped"
                or p["manifest"]["boot_epoch"] != a["boot_epochs"][role]
                or p.get("exit_evidence") != evidence
            ):
                raise PoolError("STOP_EVIDENCE_MISMATCH")
            if current["roles"][role] == "released":
                return current, {}
            if s["allocation_id"] != a["allocation_id"]:
                raise PoolError("RELEASE_OWNER_MISMATCH")
            for k, cost in p["manifest"]["profiles"][a["intent"]["profile"]].items():
                p["used"][k] = max(0, p["used"][k] - cost)
            if p.get("active_services", 0) < 1:
                raise PoolError("CAPACITY_STATE_CORRUPT")
            p["active_services"] -= 1
            if p["active_services"] == 0:
                p["used"] = {k: 0.0 for k in p["used"]}
            s["allocation_id"] = None
            current["roles"][role] = "released"
            current.setdefault("release_evidence", {})[role] = {
                "kind": "operator_confirmed_process_exit",
                **evidence,
            }
            current.update(state="closing", executor=None, executor_until=0)
            return current, {pk: p, sk: s, ak: current}

        return await self._tx([pk, sk, ak], release)

    async def fail_process(self, process_id, boot_epoch):
        pk = self.key("process", process_id)

        def fail(rows):
            p = rows[pk]
            if p is None or p["manifest"]["boot_epoch"] != boot_epoch:
                raise PoolError("PROCESS_EPOCH_MISMATCH")
            if p["state"] not in {"stopped", "retired"}:
                p["state"] = "failed"
            writes = {pk: p}
            self._ready_indices(p, writes)
            return p, writes

        return await self._tx([pk], fail)
