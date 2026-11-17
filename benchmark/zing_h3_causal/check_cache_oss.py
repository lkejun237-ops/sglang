"""Exercise the real OSS transport in a unique test-only cache prefix.

Credentials use boto3's existing workload environment. No model/GPU/Room calls.
The command writes and deletes only its newly created validation/<uuid>/ prefix.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from sglang.multimodal_gen.runtime.realtime.cache_archive import CacheArchive
from sglang.multimodal_gen.runtime.realtime.cache_tiers import CacheTiers


def run(config, output):
    policy = CacheTiers.parse(config["tiers"])
    if not policy.prefix.startswith("sglang-block0-cache/v2/test-shanghai-copy/"):
        raise ValueError("Cloud validation requires the Shanghai test cache prefix")
    policy = replace(policy, prefix=policy.prefix + "validation/" + uuid4().hex + "/")
    archive = CacheArchive(policy, "native-contract")
    key, artifact = "a" * 64, uuid4().hex
    world = "cache-contract-test"
    parts = {
        f"native-contract:{key}:a{artifact}:r0:kv:l00": b"first-rank-kv" * 4096,
        f"native-contract:{key}:a{artifact}:r1:kv:l00": b"second-rank-kv" * 4096,
    }
    native = dict(artifact_id=artifact, objects={k: len(v) for k, v in parts.items()})
    result = dict(
        passed=False, bucket=policy.bucket, prefix=policy.prefix, gpu_test=False
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({**result, "state": "started"}, indent=2) + "\n")
    try:
        future = archive.submit(world, key, native, parts.__getitem__)
        assert future is not None and future.result(timeout=180)["state"] == "persisted"
        assert archive.find(world, key)["artifact_id"] == artifact
        restored = {}
        assert (
            archive.restore(world, key, restored.__setitem__)["artifact_id"] == artifact
        )
        assert restored == parts
        result.update(
            write_read_complete=True,
            bytes=sum(map(len, parts.values())),
            sha256={k: hashlib.sha256(v).hexdigest() for k, v in restored.items()},
        )
        assert (
            archive.submit("other-world", key, native, parts.__getitem__).result(
                timeout=180
            )["state"]
            == "persisted"
        )
        deletion = archive.delete_world(world)
        assert deletion["state"] == "completed" and archive.find(world, key) is None
        assert archive.find("other-world", key) is not None
        result["world_isolation"] = True
        assert archive.delete_world("other-world")["state"] == "completed"
        # Exact owner prefix only, never the environment's existing cache/video.
        archive.objects.delete_prefix(policy.prefix)
        assert list(archive.objects.keys(policy.prefix)) == []
        result.update(passed=True, cleanup_confirmed=True)
    except BaseException as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        archive.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new receipt path for each cloud test")
    run(json.loads(args.config.read_text()), args.output)
