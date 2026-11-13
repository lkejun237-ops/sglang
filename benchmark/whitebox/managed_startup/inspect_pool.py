"""Read retained records only; never reconcile leases or alter role readiness."""

import asyncio
import json
import os
import time
from pathlib import Path

import redis.asyncio as redis

from sglang.multimodal_gen.runtime.realtime.role_pool.store import (
    RedisRecords,
    RolePoolStore,
)


async def main():
    cfg = json.loads(
        Path(os.environ["SGLANG_ROLE_POOL_COORDINATOR_CONFIG"]).read_text()
    )
    assert cfg["pool_id"] == "adaptive-test-20261109"
    async with redis.from_url(cfg["redis_url"], decode_responses=True) as client:
        store = RolePoolStore(
            RedisRecords(client, expected_run_id=cfg["redis_run_id"]),
            cfg["pool_id"],
            cfg["control_epoch"],
        )
        rows = await store.db.read(await store.db.keys(store.prefix))
        for value in rows.values():
            if not isinstance(value, dict):
                continue
            manifest = value.get("manifest")
            if (
                isinstance(manifest, dict)
                and "process_id" in manifest
                and value.get("state") != "retired"
            ):
                print(
                    json.dumps(
                        {
                            "kind": "process",
                            "pool_id": cfg["pool_id"],
                            "process_id": manifest["process_id"],
                            "boot_epoch": manifest["boot_epoch"],
                            "role": manifest["role"],
                            "hardware_ids": manifest["hardware_ids"],
                            "state": value.get("state"),
                            "used": value.get("used"),
                            "heartbeat_age_s": time.time() - value["seen_at"],
                            "services": manifest["services"],
                        }
                    )
                )
            if (
                value.get("allocation_id")
                and "roles" in value
                and value.get("state") != "released"
            ):
                print(
                    json.dumps(
                        {
                            "kind": "allocation",
                            **{
                                key: value.get(key)
                                for key in [
                                    "allocation_id",
                                    "state",
                                    "roles",
                                    "updated_at",
                                    "expires_at",
                                ]
                            },
                        }
                    )
                )


if __name__ == "__main__":
    asyncio.run(main())
