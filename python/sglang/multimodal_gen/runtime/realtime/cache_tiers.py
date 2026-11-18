# SPDX-License-Identifier: Apache-2.0
"""Block0 tier policy and the pinned Mooncake replica API (no GPU work)."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class StageTimes:
    """Host spans may overlap; their sum is not end-to-end latency."""

    def __init__(self):
        self.stages = {}
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, name):
        start, status = time.perf_counter(), "executed"
        try:
            yield
        except BaseException:
            status = "error"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            with self._lock:
                old = self.stages.get(name, {})
                self.stages[name] = dict(
                    host_ms=old.get("host_ms", 0) + elapsed_ms,
                    status="error" if old.get("status") == "error" else status,
                )


@dataclass(frozen=True)
class CacheTiers:
    l1: bool = True
    l2: bool = False
    l3: bool = False
    l1_to_l2: bool = False
    l1_to_l3: bool = False
    l2_to_l3: bool = False
    retention_days: int = 7
    max_object_bytes: int = 512 << 20
    max_pending: int = 2
    deadline_s: float = 120
    online_wait_ms: float = 0
    snapshot_bytes: int = 0
    bucket: str | None = None
    endpoint: str | None = None
    region: str | None = None
    prefix: str = "sglang-block0-cache/v2/"
    state_directory: str = "/tmp/sglang-block0-state"
    disk_directory: str | None = None
    disk_bytes: int = 0
    min_free_bytes: int = 0
    allow_overlay_l2: bool = False
    oss_probe_timeout_s: float = 1.0

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Cache tier policy must be an object")
        unknown = set(value) - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(
                "Unknown cache tier settings: " + ", ".join(sorted(unknown))
            )
        policy = cls(**value)
        for key in (
            "l1",
            "l2",
            "l3",
            "l1_to_l2",
            "l1_to_l3",
            "l2_to_l3",
            "allow_overlay_l2",
        ):
            if type(getattr(policy, key)) is not bool:
                raise ValueError("Cache tier switches must be booleans")
        for edge, source, target in (
            ("l1_to_l2", "l1", "l2"),
            ("l1_to_l3", "l1", "l3"),
            ("l2_to_l3", "l2", "l3"),
        ):
            if getattr(policy, edge) and not (
                getattr(policy, source) and getattr(policy, target)
            ):
                raise ValueError("A writeback edge requires both tiers: " + edge)
        for key in ("retention_days", "max_object_bytes", "max_pending"):
            if type(getattr(policy, key)) is not int or getattr(policy, key) <= 0:
                raise ValueError("Cache bounds must be positive integers: " + key)
        for key in ("snapshot_bytes", "disk_bytes", "min_free_bytes"):
            if type(getattr(policy, key)) is not int or getattr(policy, key) < 0:
                raise ValueError("Cache byte budgets must be nonnegative: " + key)
        import math

        for key in ("deadline_s", "online_wait_ms", "oss_probe_timeout_s"):
            n = getattr(policy, key)
            if type(n) not in (float, int) or not math.isfinite(n) or n < 0:
                raise ValueError("Invalid cache deadline")
        if (
            not 0 < policy.deadline_s <= 3600
            or policy.oss_probe_timeout_s <= 0
            or policy.online_wait_ms > policy.deadline_s * 1000
        ):
            raise ValueError("Invalid cache deadline")
        if not Path(policy.state_directory).is_absolute() or Path(
            policy.state_directory
        ) == Path("/"):
            raise ValueError("Cache state requires a dedicated absolute directory")
        if policy.l2:
            if (
                not policy.disk_directory
                or not Path(policy.disk_directory).is_absolute()
            ):
                raise ValueError("Native L2 requires an absolute directory")
            if Path(policy.disk_directory) == Path("/") or not policy.disk_bytes:
                raise ValueError("Native L2 requires a dedicated budgeted directory")
        if policy.l3:
            from urllib.parse import urlparse

            endpoint = urlparse(policy.endpoint or "")
            if (
                not policy.bucket
                or not policy.region
                or endpoint.scheme != "https"
                or not endpoint.hostname
            ):
                raise ValueError(
                    "L3 requires an explicit HTTPS regional endpoint and bucket"
                )
            if (
                endpoint.username
                or endpoint.password
                or endpoint.query
                or endpoint.fragment
            ):
                raise ValueError(
                    "L3 endpoint must not contain credentials or query parameters"
                )
            from pathlib import PurePosixPath

            prefix = PurePosixPath(policy.prefix)
            if (
                not policy.prefix.startswith("sglang-block0-cache/")
                or not policy.prefix.endswith("/")
                or ".." in prefix.parts
                or "\\" in policy.prefix
            ):
                raise ValueError("L3 requires a cache-only prefix")
        return policy

    @property
    def durable_tier(self):
        return "l3" if self.l3 else "l2" if self.l2 else None

    @property
    def cloud_write(self):
        return self.l3 and (self.l1_to_l3 or self.l2_to_l3)


class NativeReplicas:
    """Use Mooncake 0.3.13 descriptors, never infer tiers from ExistKey.

    Native promotion-on-hit must be enabled by the Store operator when L2 is
    enabled. A failed/unavailable promotion remains a miss, never memory_ready.
    """

    def __init__(
        self, client, policy: CacheTiers, *, clock=time.monotonic, read_object=None
    ):
        self.client, self.policy, self.clock = client, policy, clock
        self.read_object = read_object
        if policy.l2 and not hasattr(client, "batch_get_replica_desc"):
            raise ValueError("Native L2 requires Mooncake replica descriptors")

    def layers(self, keys, *, include_filesystem=False):
        keys = tuple(keys)
        if not self.policy.l1:
            return {key: set() for key in keys}
        if not hasattr(self.client, "batch_get_replica_desc"):
            return {
                key: {"l1"} if yes == 1 else set()
                for key, yes in zip(keys, self.client.batch_is_exist(keys))
            }
        descriptors = self.client.batch_get_replica_desc(list(keys))
        result = {}
        for key in keys:
            layers = set()
            for replica in descriptors.get(key, []):
                if str(replica.status).split(".")[-1] != "COMPLETE":
                    continue
                if replica.is_memory_replica():
                    layers.add("l1")
                elif self.policy.l2 and (
                    replica.is_local_disk_replica()
                    or (include_filesystem and self._filesystem_replica(replica))
                ):
                    layers.add("l2")
            result[key] = layers
        return result

    def _filesystem_replica(self, replica):
        # Master filesystem storage publishes DISK, not the client's LOCAL_DISK variant.
        # This is durability evidence only: DISK reads do not promote to L1.
        if not self.policy.disk_directory or not replica.is_disk_replica():
            return False
        descriptor = replica.get_disk_descriptor()
        root = PurePosixPath(self.policy.disk_directory)
        path = PurePosixPath(descriptor.file_path)
        return (
            path.is_absolute()
            and ".." not in path.parts
            and path != root
            and path.is_relative_to(root)
            and 0 < descriptor.object_size <= self.policy.max_object_bytes
        )

    def memory_ready(self, keys):
        return bool(keys) and all("l1" in v for v in self.layers(keys).values())

    def wait(
        self,
        keys,
        tier,
        deadline,
        *,
        cancelled=lambda: False,
        poll_interval=0.02,
        include_filesystem=False,
    ):
        while True:
            states = self.layers(keys, include_filesystem=include_filesystem)
            if states and all(tier in v for v in states.values()):
                return True
            if cancelled() or self.clock() >= deadline:
                return False
            time.sleep(min(poll_interval, max(0, deadline - self.clock())))

    def promote(self, keys, deadline):
        states = self.layers(keys)
        if not states or any(not v for v in states.values()):
            return False
        for key, layers in states.items():
            if "l1" in layers:
                continue
            if not self.policy.l2 or self.clock() >= deadline:
                return False
            size = self.client.get_size(key)
            if not 0 < size <= self.policy.max_object_bytes:
                return False
            # Get schedules native promotion when promotion_on_hit is enabled.
            # Keep the staging object bounded and release it before the next key.
            value = (
                self.read_object(key, size)
                if self.read_object is not None
                else self.client.get(key)
            )
            if len(value) != size:
                return False
            del value
        return self.wait(keys, "l1", deadline)


def persistence_state_path(root, descriptor):
    import hashlib
    import json

    if descriptor.get("schema") != "zing-block0-tier-state-v1":
        raise ValueError("Invalid cache persistence descriptor")
    import re

    for field, length in (
        ("key", 64),
        ("artifact_id", 32),
        ("attempt_id", 32),
        ("namespace_sha256", 64),
    ):
        if not re.fullmatch(r"[a-f0-9]{%s}" % length, str(descriptor.get(field, ""))):
            raise ValueError("Invalid cache persistence identity")
    if descriptor.get("tier") not in {"l2", "l3"}:
        raise ValueError("Invalid cache persistence tier")
    from .cache_archive import world_token

    world_token(descriptor.get("world_id"))
    token = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    return Path(root) / (token + ".json")


def read_persistence_state(root, descriptor):
    import json

    path = persistence_state_path(root, descriptor)
    try:
        if path.stat().st_size > 16384:
            return {"state": "failed", "code": "CACHE_STATUS_INVALID"}
        value = json.loads(path.read_text())
        if value.get("descriptor") != descriptor:
            return {"state": "failed", "code": "CACHE_STATUS_IDENTITY_MISMATCH"}
        return value["result"]
    except FileNotFoundError:
        return {"state": "pending"}


def prune_persistence_states(root, *, retention_s=3600, now=None):
    """Only this module's hashed receipt files; run on the background IO owner."""
    import re

    cutoff = (time.time() if now is None else now) - retention_s
    for path in Path(root).glob("*.json"):
        if not re.fullmatch(r"[a-f0-9]{64}\.json", path.name) or path.is_symlink():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except FileNotFoundError:
            pass
