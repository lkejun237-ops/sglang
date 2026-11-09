# SPDX-License-Identifier: Apache-2.0
"""Explicit authority bootstrap and graceful drain. Never starts/stops GPU processes."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from .store import RedisRecords, RolePoolStore


async def run(args):
    config = json.loads(Path(args.config).read_text())
    if args.action in {"native-identity", "native-recovery"}:
        from .contracts import Credentials

        if not args.service_http:
            raise ValueError("An explicit trusted service HTTP address is required")
        action = "identity" if args.action == "native-identity" else "recovery"
        body = dict(control_epoch=config["control_epoch"])
        if action == "recovery":
            if not args.boot_epoch:
                raise ValueError("The observed native boot epoch is required")
            body.update(
                boot_epoch=args.boot_epoch, offset=args.offset, limit=args.limit
            )
        token = Credentials(
            os.environ[config["signing_secret_env"]], config["control_epoch"]
        ).sign(action, body, expires_at=time.time() + 30)
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            response = await client.post(
                args.service_http.rstrip("/") + "/managed/" + action,
                json=body,
                headers={"Authorization": "Bearer " + token},
            )
            response.raise_for_status()
            return response.json()
    if args.action in {"inspect-store", "bootstrap"}:
        import redis.asyncio as redis

        async with redis.from_url(config["redis_url"], decode_responses=True) as client:
            info = await client.info("server")
            if args.action == "inspect-store":
                return dict(
                    redis_run_id=info["run_id"],
                    role=(await client.execute_command("ROLE"))[0],
                )
            if not args.new_pool:
                raise ValueError(
                    "Bootstrap requires explicit --new-pool; recovery is not automatic"
                )
            store = RolePoolStore(
                RedisRecords(client, expected_run_id=config["redis_run_id"]),
                config["pool_id"],
                config["control_epoch"],
            )
            if await store.db.keys(store.prefix):
                raise ValueError(
                    "Pool contains state; fence and reconcile it instead of bootstrapping"
                )
            await store.initialize()
            return dict(state="initialized", pool_id=config["pool_id"])
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        body = dict(process_id=args.process_id, boot_epoch=args.boot_epoch)
        if not all(body.values()):
            raise ValueError("Process ID and boot epoch are required")
        if args.action in {"retire-process", "confirm-stopped"}:
            if not args.stop_receipt:
                raise ValueError(
                    "An operator-confirmed process stop receipt is required"
                )
            body["stop_receipt"] = args.stop_receipt
            if args.action == "confirm-stopped":
                if not args.operator_id:
                    raise ValueError("Operator identity is required")
                body["operator_id"] = args.operator_id
        response = await client.post(
            config["coordinator_url"].rstrip("/") + "/v1/role-pool/" + args.action,
            json=body,
            headers={
                "Authorization": "Bearer " + os.environ[config["operator_token_env"]]
            },
        )
        response.raise_for_status()
        return response.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "inspect-store",
            "bootstrap",
            "drain",
            "process-status",
            "retire-process",
            "confirm-stopped",
            "native-identity",
            "native-recovery",
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--new-pool", action="store_true")
    parser.add_argument("--process-id")
    parser.add_argument("--boot-epoch")
    parser.add_argument("--stop-receipt")
    parser.add_argument("--operator-id")
    parser.add_argument("--service-http")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
