#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise one frozen Prefill payload through the normal managed test pool.

This is a model/API component check, not gameplay, browser or image-quality
acceptance. Payload provenance must state whether S0 is recorded or a fixture.
No native ownership bypass, worker restart, cache eviction or routing mutation.
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("SGLANG_LIGHTWEIGHT_RUNTIME", "1")

import msgspec
from websockets.asyncio.client import connect

from sglang.multimodal_gen.runtime.realtime.whitebox.disaggregated_prefill import (
    DisaggregatedPrefillLeases,
)


def safe_value(value):
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, dict):
        return {k: safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_value(v) for v in value]
    return value


async def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    raw = Path(args.payload).read_bytes()
    payload = msgspec.msgpack.decode(raw)
    if payload.get("type") != "prefill" or not payload.get("world_id"):
        raise ValueError("A world-scoped native Prefill payload is required")
    if payload.get("force"):
        raise ValueError("This probe must not force recomputation")
    expected = config.pop("expected_workers")
    assert set(expected) == {"encoder", "dit", "decoder"}
    os.environ[config["room_token_env"]]  # Fail before admission if missing.
    lease = DisaggregatedPrefillLeases(
        None,
        connect,
        config=config,
        user_id="cache-validation-" + uuid4().hex,
        model_revision=args.model_revision,
        vae_fingerprint="local",
        trace_id=uuid4().hex,
        max_bytes=96 << 20,
        strict_release=True,
        cleanup_timeout_s=90,
        operation_path="/v1/realtime_video/prefill",
    )
    result = dict(
        scope="managed Prefill API component; not gameplay/browser acceptance",
        provenance=json.loads(Path(args.provenance).read_text()),
        payload_sha256=hashlib.sha256(raw).hexdigest(),
        request_id=payload["request_id"],
        admission_id=lease.admission_id,
        started_epoch=time.time(),
        receipts=[],
        passed=False,
    )
    (root / "intent.json").write_text(json.dumps(result, indent=2))
    began = time.monotonic()
    memory_ready = persisted = False
    try:
        async with asyncio.timeout(540):
            async with lease.open() as (socket, assignment):
                result["admission_ms"] = (time.monotonic() - began) * 1000
                actual = {
                    role: lease.group["topology"][role]["worker_id"]
                    for role in expected
                }
                if actual != expected:
                    raise RuntimeError("Unexpected test worker identity")
                result.update(
                    workers=actual,
                    boot_epochs=lease.group["boot_epochs"],
                    allocation_id=lease.group["allocation_id"],
                    session_id=assignment.session_id,
                )
                ready = msgspec.msgpack.decode(await socket.recv())
                if ready.get("type") != "prefill.ready":
                    raise RuntimeError("Prefill capability unavailable")
                sent = time.monotonic()
                await socket.send(raw)
                while True:
                    message = msgspec.msgpack.decode(await socket.recv())
                    if message.get("request_id") != payload["request_id"]:
                        raise RuntimeError("Prefill request identity mismatch")
                    result["receipts"].append(
                        dict(
                            client_elapsed_ms=(time.monotonic() - sent) * 1000,
                            message=safe_value(message),
                        )
                    )
                    (root / "receipts.json").write_text(
                        json.dumps(result["receipts"], indent=2)
                    )
                    kind = message.get("type")
                    if kind == "prefill.memory_ready":
                        if message.get("session_id") != assignment.session_id:
                            raise RuntimeError("Prefill session identity mismatch")
                        memory_ready = True
                        result["memory_ready_ms"] = (time.monotonic() - sent) * 1000
                        await lease.release_compute()
                        result["compute_released_ms"] = (time.monotonic() - sent) * 1000
                        if not payload.get("persist", False):
                            break
                    elif kind == "prefill.persisted":
                        if not memory_ready:
                            raise RuntimeError("Persistence preceded memory-ready")
                        persisted = True
                        break
                    elif kind == "prefill.failed":
                        raise RuntimeError(
                            "Native Prefill failed: " + str(message.get("code"))
                        )
                    elif kind != "prefill.persisting":
                        raise RuntimeError("Unexpected native Prefill receipt")
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        result.update(
            resource_state=lease.resource_state,
            finished_epoch=time.time(),
            elapsed_ms=(time.monotonic() - began) * 1000,
            passed=memory_ready
            and (persisted or not payload.get("persist", False))
            and lease.resource_state == "released"
            and "error" not in result,
        )
        (root / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "payload", "provenance", "model-revision", "output"):
        parser.add_argument("--" + flag, required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
