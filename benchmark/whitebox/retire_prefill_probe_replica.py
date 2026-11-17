#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Remove one owned test artifact's native copies to exercise real L3 recall.

Requires an exact persisted receipt and complete cloud artifact. Run only for
an explicitly authorized, idle test service. Never forces native lease expiry,
removes unrelated keys, edits files behind the Store or restarts a process.
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path


def main(args):
    from mooncake.store import MooncakeDistributedStore

    from sglang.multimodal_gen.runtime.realtime.block0_rdma import block0_rdma_devices
    from sglang.multimodal_gen.runtime.realtime.cache_archive import CacheArchive
    from sglang.multimodal_gen.runtime.realtime.cache_tiers import CacheTiers

    config = json.loads(Path(args.config).read_text())
    devices = block0_rdma_devices(config)
    policy = CacheTiers.parse(config["tiers"])
    if "/test-shanghai-copy/validation/" not in policy.prefix:
        raise ValueError("Only the isolated Shanghai test cache is allowed")
    if not re.fullmatch(r"[a-f0-9]{64}", args.key) or not re.fullmatch(
        r"[a-f0-9]{32}", args.artifact
    ):
        raise ValueError("Exact key and artifact identities are required")
    receipt = json.loads(Path(args.receipt).read_text())
    descriptor = receipt["descriptor"]
    if (
        descriptor["key"] != args.key
        or descriptor["artifact_id"] != args.artifact
        or descriptor["world_id"] != args.world
        or receipt["result"].get("state") != "persisted"
        or receipt["result"].get("tier") != "l3"
    ):
        raise ValueError("The owned artifact has no verified L3 receipt")
    with urllib.request.urlopen(config["master_http_url"] + "/get_all_keys") as r:
        keys = r.read().decode().splitlines()
    markers = [k for k in keys if k.endswith(":" + args.key + ":manifest")]
    if len(markers) != 1 or not markers[0].startswith(config["namespace"] + ":g3:"):
        raise ValueError("Expected exactly one native test manifest")
    marker = markers[0]
    base = marker.removesuffix(":manifest")
    namespace = base.removesuffix(":" + args.key)
    cloud = CacheArchive(policy, namespace).find(args.world, args.key)
    if cloud is None or cloud["native"]["artifact_id"] != args.artifact:
        raise ValueError("The exact cloud artifact is incomplete or unavailable")
    client = MooncakeDistributedStore()
    assert (
        client.setup(
            "127.0.0.1",
            "P2PHANDSHAKE",
            0,
            32 << 20,
            "rdma",
            devices,
            config["master_server_address"],
        )
        == 0
    )
    try:
        native = json.loads(client.get(marker))
        published = {k: v for k, v in cloud["native"].items() if k != "objects"}
        if native != published or native["artifact_id"] != args.artifact:
            raise ValueError("Cloud and native artifact identities differ")
        prefix = base + ":a" + args.artifact + ":"
        owned = [marker] + [k for k in keys if k.startswith(prefix)]
        expected = 1 + native["sp_world_size"] * (native["geometry"]["layers"] + 1)
        if (
            len(owned) != expected
            or set(owned) != {marker, *cloud["native"]["objects"]}
            or expected > 256
        ):
            raise ValueError("Incomplete native artifact inventory")
        result = dict(
            key=args.key,
            artifact_id=args.artifact,
            native_keys=owned,
            started_epoch=time.time(),
        )
        with Path(args.output).open("x") as handle:
            handle.write(json.dumps(result, indent=2))
        pending, deadline = owned, time.monotonic() + 20
        while pending:
            codes = client.batch_remove(pending)
            pending = [k for k, code in zip(pending, codes) if code not in (0, -704)]
            if not pending or time.monotonic() >= deadline:
                break
            time.sleep(0.2)
        result.update(
            removed=len(owned) - len(pending),
            pending=pending,
            finished_epoch=time.time(),
            passed=not pending,
        )
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        if pending:
            raise RuntimeError("Native leases prevented complete retirement")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "receipt", "world", "key", "artifact", "output"):
        parser.add_argument("--" + name, required=True)
    main(parser.parse_args())
