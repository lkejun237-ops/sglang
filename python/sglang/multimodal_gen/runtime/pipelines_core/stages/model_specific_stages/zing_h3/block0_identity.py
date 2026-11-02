# SPDX-License-Identifier: Apache-2.0
"""Startup-only content identities for persistent Block0 preparations."""

import hashlib
import json
import math
import os
from pathlib import Path

SCHEMA = "zing-block0-v3"


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def budget_bytes(value, name):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value <= 0
        or value * 10**9 < 1
    ):
        raise ValueError(f"{name} must be a finite positive decimal GB budget")
    return int(value * 10**9)


def startup_config(args):
    """Called only after pool admission; disabled caches need no disk/host args."""
    mode = getattr(args, "zing_block0_cache_mode", "on")
    if mode not in {"on", "off"}:
        raise ValueError("Zing Block0 cache mode must be on or off")
    if mode == "off":
        return None
    host = budget_bytes(args.zing_block0_cache_host_gb, "--zing-block0-cache-host-gb")
    disk = budget_bytes(args.zing_block0_cache_disk_gb, "--zing-block0-cache-disk-gb")
    root = args.zing_block0_cache_dir
    if not root or not Path(root).is_dir():
        raise ValueError(
            "--zing-block0-cache-dir must name an existing mounted directory"
        )
    # A missing mount must not be hidden by mkdir on a missing volume. The deployment
    # mounts and creates its cache root before starting the worker.
    if Path(root).resolve() == Path("/"):
        raise ValueError("The Block0 cache root cannot be the filesystem root")
    if not storage_mounted(Path(root).resolve()):
        raise ValueError("Block0 storage must be mounted before the worker starts")
    return host, disk, Path(root).resolve()


def storage_mounted(root):
    # Cache directories are usually below the volume mount, not mountpoints
    # themselves. Do not accept a stale cache directory on the root filesystem.
    return any(
        path != Path("/") and os.path.ismount(path) for path in (root, *root.parents)
    )


def build_identity(stage, args, *, device):
    """Trust the immutable export's config; check small metadata across SP ranks."""
    from sglang.multimodal_gen.runtime.conditioning.worker import coordinate_validation
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_sp_group,
        get_ulysses_parallel_world_size,
    )

    identity = error = None
    try:
        with open(
            Path(stage.model_path) / "zing_config.json", encoding="utf-8"
        ) as handle:
            config = json.load(handle)
        config.pop("checkpoint", None)  # Export location is not model content.
        group = get_sp_group()
        stores = stage.kv_pool._pool._slots[0].kv_cache
        identity = {
            "cache_schema_version": SCHEMA,
            "zing_config": config,
            "sp": {
                "size": group.world_size,
                "ulysses": get_ulysses_parallel_world_size(),
                "ring": group.ring_world_size,
            },
            "kv_dtype": str(stores[0].k.dtype),
        }
        digest = fingerprint(identity)
    except Exception as exc:
        error = str(exc)
        digest = None
    values = coordinate_validation((error, digest))
    if any(reason is not None for reason, _ in values) or any(
        value != values[0] for value in values
    ):
        raise ValueError(
            f"Block0 config differs or is invalid across SP ranks: {values}"
        )
    return identity
