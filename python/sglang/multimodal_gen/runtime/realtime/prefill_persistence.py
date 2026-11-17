# SPDX-License-Identifier: Apache-2.0
"""Read-only persistence receipts for a worker-selected Block0 entry.

Descriptors come from the trusted worker result, never from a public client.
No GPU reservation or LRU mutation is needed while waiting for disk writes.
"""

import hashlib
import json
import re
from pathlib import Path


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def persistence_descriptor(cache, entry, *, requested=True):
    if hasattr(cache, "request_persistence"):
        return cache.request_persistence(entry) if requested else None
    disk = getattr(cache, "disk_cache", None)
    if disk is None:
        return None
    if not hasattr(disk, "model"):
        return {
            "schema": "zing-block0-v3",
            "disk_format": 4,
            "key": entry.key,
            "identity_sha256": _fingerprint(disk.identity),
            "sp_world_size": disk.sp_world_size,
        }
    return {
        "key": entry.key,
        "generation": entry.generation,
        "model": disk.model,
        "layout": disk.layout,
        "sp_world_size": disk.sp_world_size,
    }


def inspect_block0_persistence(cache, descriptor):
    """Use the disk backend's public directory contract, including native v3."""
    if descriptor.get("schema") == "zing-block0-tier-state-v1":
        return cache.persistence_status(descriptor)
    disk = getattr(cache, "disk_cache", None)
    if disk is None:
        return {"state": "failed", "code": "CACHE_PERSISTENCE_UNAVAILABLE"}
    root = disk.cache_dir if hasattr(disk, "cache_dir") else disk.root
    return inspect_persistence(root, descriptor)


def inspect_persistence(root, descriptor):
    if descriptor.get("schema") == "zing-block0-tier-state-v1":
        from .cache_tiers import read_persistence_state

        return read_persistence_state(root, descriptor)
    if descriptor.get("schema") == "zing-block0-v3":
        return _inspect_managed_persistence(root, descriptor)
    for name in ("key", "model", "layout"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get(name, ""))):
            raise ValueError("Invalid internal cache descriptor")
    count = descriptor.get("sp_world_size")
    if type(count) is not int or not 1 <= count <= 128:
        raise ValueError("Invalid cache shard count")
    root = Path(root).resolve()
    folder = (
        root
        / "v2"
        / descriptor["model"]
        / descriptor["layout"]
        / descriptor["key"][:2]
        / descriptor["key"]
    )
    if not folder.resolve().is_relative_to(root):
        raise ValueError("Invalid internal cache path")
    try:
        if not (folder / ".complete").is_file():
            return {"state": "pending"}
        manifest_path = folder / "manifest.json"
        if manifest_path.stat().st_size > 2 << 20:
            return {"state": "failed", "code": "PREFILL_CACHE_INVALID"}
        manifest = json.loads(manifest_path.read_text())
        identity = manifest.get("identity", {})
        if (
            manifest.get("schema") != "zing-block0-v2"
            or manifest.get("early_key") != descriptor["key"]
            or manifest.get("sp_world_size") != count
            or not isinstance(manifest.get("generation"), str)
            or not manifest["generation"]
            or _fingerprint(identity.get("model")) != descriptor["model"]
            or _fingerprint(identity.get("layout")) != descriptor["layout"]
            or set(manifest.get("ranks", {})) != {str(rank) for rank in range(count)}
        ):
            return {"state": "failed", "code": "PREFILL_CACHE_INVALID"}
        shards = []
        for rank, info in manifest["ranks"].items():
            kv, conditions = (
                folder / f"rank-{rank}" / "kv.bin",
                folder / f"rank-{rank}" / "conditions.pt",
            )
            if (
                info.get("generation") != manifest["generation"]
                or not kv.resolve().is_relative_to(root)
                or not conditions.resolve().is_relative_to(root)
                or kv.stat().st_size != info.get("kv_bytes")
                or (
                    info.get("conditions_bytes") is not None
                    and conditions.stat().st_size != info["conditions_bytes"]
                )
            ):
                return {"state": "failed", "code": "PREFILL_CACHE_INVALID"}
            shards.append(
                {
                    "rank": int(rank),
                    "kv_bytes": info["kv_bytes"],
                    "conditions_bytes": conditions.stat().st_size,
                    "kv_write_host_ms": info.get("write_host_ms"),
                    "kv_fsync_host_ms": info.get("fsync_host_ms"),
                    "conditions_write_host_ms": info.get("conditions_write_host_ms"),
                }
            )
        # A completed concurrent fill may have won publication for the same key.
        # Validate the actual complete generation, not our losing candidate UUID.
        after = json.loads(manifest_path.read_text())
        if not (folder / ".complete").is_file() or any(
            after.get(k) != manifest.get(k) for k in ("generation", "ranks", "identity")
        ):
            return {"state": "pending"}
        return {
            "state": "persisted",
            "generation": manifest["generation"],
            "kv_bytes": sum(row["kv_bytes"] for row in shards),
            "conditions_bytes": sum(row["conditions_bytes"] for row in shards),
            "ranks": sorted(shards, key=lambda row: row["rank"]),
        }
    except FileNotFoundError:
        return {"state": "pending"}
    except (OSError, ValueError, TypeError, KeyError):
        return {"state": "failed", "code": "PREFILL_STORAGE_UNAVAILABLE"}


def _inspect_managed_persistence(root, descriptor):
    """Observe native v3 metadata only; never re-read/hash full KV tensors."""
    for name in ("key", "identity_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get(name, ""))):
            raise ValueError("Invalid internal managed cache descriptor")
    count = descriptor.get("sp_world_size")
    if type(count) is not int or not 1 <= count <= 128:
        raise ValueError("Invalid cache shard count")
    root = Path(root).resolve()
    folder = root / "entries" / descriptor["key"]

    def read(path):
        if not path.resolve().is_relative_to(root) or path.stat().st_size > 2 << 20:
            raise ValueError("Invalid managed cache metadata")
        return json.loads(path.read_text())

    try:
        manifest = read(folder / "manifest.json")
        if (
            manifest.get("schema") != "zing-block0-v3"
            or manifest.get("disk_format") != 4
            or manifest.get("early_key") != descriptor["key"]
            or manifest.get("sp_world_size") != count
            or _fingerprint(manifest.get("identity")) != descriptor["identity_sha256"]
        ):
            return {"state": "failed", "code": "PREFILL_CACHE_INVALID"}
        shards, observed = [], []
        for rank in range(count):
            base = folder / f"rank-{rank}"
            info = read(base / "ready.json")
            observed.append(info)
            kv, conditions = base / "kv.bin", base / "conditions.pt"
            if (
                info.get("disk_format") != 4
                or info.get("conditions_schema") != "zing-block0-conditions-v1"
                or not kv.resolve().is_relative_to(root)
                or not conditions.resolve().is_relative_to(root)
                or kv.stat().st_size != info.get("kv_bytes")
            ):
                return {"state": "pending"}
            shards.append(
                {
                    "rank": rank,
                    "kv_bytes": info["kv_bytes"],
                    "conditions_bytes": conditions.stat().st_size,
                }
            )
        # Same-key writers are interchangeable; a changing marker means retry,
        # not proof that the requested bytes have already become durable.
        if read(folder / "manifest.json") != manifest or any(
            read(folder / f"rank-{rank}" / "ready.json") != info
            for rank, info in enumerate(observed)
        ):
            return {"state": "pending"}
        return {
            "state": "persisted",
            "schema": "zing-block0-v3",
            "kv_bytes": sum(s["kv_bytes"] for s in shards),
            "conditions_bytes": sum(s["conditions_bytes"] for s in shards),
            "ranks": shards,
        }
    except FileNotFoundError:
        return {"state": "pending"}
    except (OSError, ValueError, TypeError, KeyError):
        return {"state": "failed", "code": "PREFILL_STORAGE_UNAVAILABLE"}
