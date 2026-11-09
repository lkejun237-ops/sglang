# SPDX-License-Identifier: Apache-2.0
"""Stable admission home across resource shards. Never expire uncertain claims."""

from .contracts import PoolError, digest


class AdmissionGate:
    def __init__(self, records, control_epoch):
        self.db, self.epoch = records, control_epoch

    def key(self, user_id):
        # One transaction touches one user hash slot, independently of the pool
        # transaction. The intermediate intent is persistent and recoverable.
        return "{role-admission:" + digest(user_id)[:32] + "}:owner"

    async def claim(self, intent, pool_id):
        key = self.key(intent["user_id"])
        expected = dict(
            admission_id=intent["admission_id"],
            user_id=intent["user_id"],
            player_id=intent["player_id"],
            pool_id=pool_id,
            control_epoch=self.epoch,
            intent_hash=digest(intent),
        )

        def claim(rows):
            old = rows[key]
            if old is not None and old["state"] != "released":
                if any(old[k] != v for k, v in expected.items()):
                    raise PoolError("USER_SESSION_LIMIT")
                return old, {}
            value = {**expected, "state": "pending", "allocation_id": None}
            return value, {key: value}

        return await self.db.transact([key], claim)

    async def bind(self, intent, pool_id, allocation_id):
        key = self.key(intent["user_id"])

        def bind(rows):
            value = rows[key]
            if (
                not value
                or value["admission_id"] != intent["admission_id"]
                or value["pool_id"] != pool_id
                or value["control_epoch"] != self.epoch
                or value["state"] not in {"pending", "bound"}
                or value["allocation_id"] not in {None, allocation_id}
            ):
                raise PoolError("ADMISSION_HOME_CHANGED")
            value.update(state="bound", allocation_id=allocation_id)
            return value, {key: value}

        return await self.db.transact([key], bind)

    async def verify(self, assignment, pool_id):
        key = self.key(assignment["intent"]["user_id"])

        def check(rows):
            v = rows[key]
            if (
                not v
                or v["state"] != "bound"
                or v["control_epoch"] != self.epoch
                or v["pool_id"] != pool_id
                or v["allocation_id"] != assignment["allocation_id"]
                or v["admission_id"] != assignment["intent"]["admission_id"]
            ):
                raise PoolError("ADMISSION_HOME_UNCONFIRMED", 503)
            return v, {}

        return await self.db.transact([key], check)

    async def release(self, intent, pool_id, allocation_id=None):
        from .store import Expiring

        key = self.key(intent["user_id"])

        def release(rows):
            value = rows[key]
            if (
                not value
                or value["admission_id"] != intent["admission_id"]
                or value["pool_id"] != pool_id
                or value["allocation_id"] != allocation_id
            ):
                return None, {}
            value["state"] = "released"
            return None, {key: Expiring(value, 86400)}

        return await self.db.transact([key], release)
