"""Exercise real batch handlers, Redis and managed GPU Prefill in a test lane.

The HTTP ASGI transport and signed test issuer are local to this probe. This
does not validate the deployed Gateway, WTP authorization, CLI or browser.
"""

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("SGLANG_LIGHTWEIGHT_RUNTIME", "1")

import httpx
import msgspec
from fastapi import FastAPI
from redis.asyncio import Redis
from websockets.asyncio.client import connect

from sglang.multimodal_gen.runtime.realtime.whitebox.prefill_gateway import (
    PrefillGateway,
    RedisPrefillStore,
    register_prefill_routes,
)


async def run(args):
    config = json.loads(Path(os.environ["WHITEBOX_ROLE_POOL_CONFIG"]).read_text())
    if config["compatibility"] != "cachetiers-shcopy-06d-sp1":
        raise ValueError("Require the explicitly isolated test role group")
    if args.output.exists():
        raise ValueError("Use a new result path")
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            "https://shcopy.world-test-platform.loopit.com.cn/whitebox/v1/slots"
        )
        response.raise_for_status()
        capacity = response.json()
    if capacity["free_slots"] != 1:
        raise ValueError("Test slot is not free; do not queue a probe")
    native = msgspec.msgpack.decode(args.payload.read_bytes())
    world = {"source_id": "interactive-production", "world_id": native.pop("world_id")}
    native.pop("cache_write", None)
    native.update(seed=args.seed, persist=True, persist_timeout_s=300)
    subject, prefix = (
        "batch-probe-" + uuid4().hex,
        "cachetiers:batch-probe:" + uuid4().hex + ":",
    )
    secret = secrets.token_urlsafe(32)
    redis = Redis.from_url(os.environ["CACHE_PROBE_REDIS_URL"], decode_responses=True)
    store = RedisPrefillStore(redis, prefix=prefix)
    manager = PrefillGateway(
        coordinator=None,
        connect=connect,
        secret=secret,
        model_revision="whitebox-copy-model1027-480",
        vae_fingerprint="local",
        store=store,
    )
    app = FastAPI()
    register_prefill_routes(app, manager)
    result = {
        "passed": False,
        "started_at": time.time(),
        "scope": "production batch handlers + real Redis/GPU; test ASGI transport/issuer",
        "seed": args.seed,
        "world": world,
        "redis_prefix": prefix,
    }
    args.output.write_text(json.dumps(result))

    def envelope():
        items = []
        for request_id in request_ids:
            raw = msgspec.msgpack.encode({**native, "request_id": request_id})
            now = int(time.time())
            claims = {
                "v": 2,
                "auth_method": "api-token",
                "aud": "whitebox-prefill",
                "sub": subject,
                "iat": now,
                "exp": now + 120,
                "jti": uuid4().hex,
                "operation": "submit",
                "scope": "world",
                **world,
                "request_id": request_id,
                "payload_bytes": len(raw),
                "payload_sha256": hashlib.sha256(raw).hexdigest(),
            }
            payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
            signature = base64.urlsafe_b64encode(
                hmac.new(secret.encode(), payload, hashlib.sha256).digest()
            ).rstrip(b"=")
            items.append(
                {
                    "world": world,
                    "authorization": (payload + b"." + signature).decode(),
                    "payload": raw,
                }
            )
        return msgspec.msgpack.encode({"items": items})

    request_ids = ["cachetiers-batch-" + uuid4().hex for _ in range(2)]
    result["request_ids"] = request_ids
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test-batch",
            timeout=600,
        ) as client:
            began = time.monotonic()
            response = await client.post(
                "/v1/realtime_video/prefill/batch",
                content=envelope(),
                headers={"Content-Type": "application/msgpack"},
            )
            result["first_ms"] = (time.monotonic() - began) * 1000
            result["first"] = response.json()
            assert response.status_code == 200
            assert result["first"]["state"] == "succeeded"
            assert result["first"]["durable_count"] == 2
            assert all(
                row["receipt"]["type"] == "prefill.persisted"
                for row in result["first"]["items"]
            )
            began = time.monotonic()
            repeated = await client.post(
                "/v1/realtime_video/prefill/batch",
                content=envelope(),
                headers={"Content-Type": "application/msgpack"},
            )
            result["repeat_ms"] = (time.monotonic() - began) * 1000
            result["repeat"] = repeated.json()
            assert repeated.status_code == 200 and result["repeat"] == result["first"]
        result.update(passed=True, idempotence_verified=True)
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "detail": str(error)}
        raise
    finally:
        await manager.close()
        if result["passed"]:
            keys = [key async for key in redis.scan_iter(match=prefix + "*", count=100)]
            assert len(keys) <= 32 and all(key.startswith(prefix) for key in keys)
            if keys:
                await redis.delete(*keys)
            result["status_cleanup_count"] = len(keys)
        await redis.aclose()
        result["finished_at"] = time.time()
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    asyncio.run(run(parser.parse_args()))
