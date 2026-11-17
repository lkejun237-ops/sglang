"""Validate deployed world deletion with two new, isolated OSS fixtures.

Uses the running Encoder API and native archive transport, but no GPU request.
Retains deletion fences and failure evidence; never deletes an existing world.
"""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

from sglang.multimodal_gen.runtime.realtime.cache_archive import CacheArchive
from sglang.multimodal_gen.runtime.realtime.cache_tiers import CacheTiers


def run(args):
    policy = CacheTiers.parse(json.loads(args.config.read_text())["tiers"])
    if not policy.prefix.startswith(
        "sglang-block0-cache/v2/test-shanghai-copy/validation/"
    ):
        raise ValueError("Only the isolated Shanghai test cache is allowed")
    token = args.admin_token_file.read_text().strip()
    if len(token) < 32 or args.output.exists():
        raise ValueError("Require configured authorization and a new output path")
    suffix = uuid4().hex
    worlds = ["cache-api-probe-" + suffix + "-" + role for role in ("a", "b")]
    namespace = "cache-api-probe-" + suffix
    archive = CacheArchive(policy, namespace)
    key = hashlib.sha256(suffix.encode()).hexdigest()
    artifact = uuid4().hex
    object_key = namespace + ":" + key + ":a" + artifact + ":r0:kv:l00"
    parts = {object_key: b"isolated-api-fixture" * 4096}
    native = {"artifact_id": artifact, "objects": {k: len(v) for k, v in parts.items()}}
    result = {"worlds": worlds, "started_at": time.time(), "passed": False}
    args.output.write_text(json.dumps(result))

    def call(path, body=None, authorized=True):
        headers = {"Content-Type": "application/json"}
        if authorized:
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(
            args.url.rstrip("/") + path,
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.load(response)

    try:
        for world in worlds:
            assert (
                archive.submit(world, key, native, parts.__getitem__).result(
                    timeout=90
                )["state"]
                == "persisted"
            )
            found = archive.find(world, key)
            assert found is not None, "Published fixture failed archive validation"
            assert found["artifact_id"] == artifact
        try:
            call("/delete-world", {"world_id": worlds[0]}, authorized=False)
        except urllib.error.HTTPError as error:
            assert error.code == 403
            result["unauthorized_status"] = error.code
        else:
            raise AssertionError("Unauthenticated deletion was accepted")
        result["operations"] = []
        for index, world in enumerate(worlds):
            started = time.monotonic()
            status, record = call("/delete-world", {"world_id": world})
            assert status == 202 and record["world_id"] == world
            operation = record["operation_id"]
            while record["state"] == "pending":
                if time.monotonic() - started > 60:
                    raise TimeoutError("World deletion remained pending")
                time.sleep(0.2)
                _, record = call("/operations/" + operation)
            assert record["state"] == "completed"
            assert archive.find(world, key) is None
            if index == 0:
                assert archive.find(worlds[1], key)["artifact_id"] == artifact
                result["world_isolation"] = True
            result["operations"].append(
                {"receipt": record, "elapsed_ms": (time.monotonic() - started) * 1000}
            )
        result.update(passed=True, payload_cleanup_confirmed=True, fences_retained=True)
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "detail": str(error)}
        raise
    finally:
        archive.close()
        result["finished_at"] = time.time()
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--admin-token-file", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
