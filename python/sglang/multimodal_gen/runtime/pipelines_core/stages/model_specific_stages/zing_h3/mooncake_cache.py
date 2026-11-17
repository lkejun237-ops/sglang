# SPDX-License-Identifier: Apache-2.0
"""Shared Block0 tiers; Mooncake owns replicas, workers own private GPU slots."""

import atexit
import ctypes
import io
import json
import logging
import re
import socket
import threading
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, replace
from uuid import uuid4

import torch

from sglang.multimodal_gen.runtime.models.dits import zing_h3_prefix_codec as kv_codec
from sglang.multimodal_gen.runtime.realtime.block0_rdma import block0_rdma_devices
from sglang.multimodal_gen.runtime.realtime.cache_archive import ArchiveProbeTimeout

from . import preparation_cache as prep
from .block0_conditions import decode_conditions, encode_conditions
from .block0_identity import fingerprint
from .cache_index import _component, validate_labels
from .cache_logging import cache_event

# Mooncake 0.3.13 public ErrorCode values. Never force-remove leased objects.
_NOT_FOUND = -704
_ALREADY_EXISTS = -705
_KEY = re.compile(r"[0-9a-f]{64}\Z")
_UNSET = object()
logger = logging.getLogger(__name__)


def canonical_manifest(value):
    return json.dumps(
        {k: v for k, v in value.items() if k != "objects"}, sort_keys=True
    ).encode()


class CacheTransferError(RuntimeError):
    """A cache operation failed consistently across the participating SP group."""


@dataclass(frozen=True)
class SharedKV:
    shape: tuple
    dtype: torch.dtype
    keys: tuple[str, ...]
    artifact_id: str = ""
    encoding: str = kv_codec.BF16

    def numel(self):
        import math

        return math.prod(self.shape)

    def element_size(self):
        return torch.empty((), dtype=self.dtype).element_size()

    def layer_bytes(self):
        if self.encoding == kv_codec.FP8:
            return kv_codec.layer_bytes(self.shape[2], *self.shape[3:])
        return self.numel() * self.element_size() // self.shape[0]

    def storage_bytes(self):
        return self.shape[0] * self.layer_bytes()


class MooncakeBackend(prep.Block0PreparationCache):
    """Pure Store client with bounded background recall and persistence.

    ``client`` and ``list_keys`` are dependency injection points for CPU contract
    tests. Real clients enumerate immutable association keys through the existing
    Master's HTTP metadata endpoint; no read/modify/write index is maintained.
    """

    backend = "mooncake"
    mode = "on"
    cpu_capacity_bytes = 0

    def __init__(
        self,
        pool,
        *,
        identity,
        config,
        rank=0,
        world_size=1,
        backend="mooncake",
        client=None,
        list_keys=None,
        replicate_config_factory=None,
    ):
        self.kv_encoding = kv_codec.encoding_for_identity(identity)
        self.requires_raw_prefix = self.kv_encoding == kv_codec.BF16
        self.identity = (
            identity
            if self.requires_raw_prefix
            else {**identity, "mooncake_kv_encoding": self.kv_encoding}
        )
        self.log_rank = rank
        self.world_size = world_size
        self.hits = self.misses = self.evictions = 0
        self.last_lookup = {}
        self.write_failures = 0
        self.write_skips = 0
        self.pending = None
        self._artifacts = {}
        self._worlds = {}
        self._observer_targets = {}
        self._observer_contexts = {}
        self._input_s0 = {}
        self._cloud_jobs = {}
        self._durability = {}
        self._new_entries = {}
        self._loads = {}
        self._origins = {}
        self._load_lock = threading.RLock()
        from sglang.multimodal_gen.runtime.realtime.cache_tiers import CacheTiers

        self.tiers = CacheTiers.parse(config.get("tiers", {}))
        from sglang.multimodal_gen.runtime.realtime.cache_observer import CacheObserver

        self.observer = CacheObserver(self.tiers) if rank == 0 else None
        self._io = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="block0-durable"
        )
        self._recall_io = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="block0-recall"
        )
        self._pending_indices = set()
        self._known = set()
        self._registered = {}
        self._transfer_fault = None
        self._closed = False
        self._list_keys_override = list_keys
        self._namespace = config.get("namespace", "zing_b0")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self._namespace):
            raise ValueError(
                "Mooncake namespace must contain only letters, digits, _, - or ."
            )
        # Existing objects cannot acquire group membership retroactively. Keep
        # grouped entries separate from the original ungrouped wire format.
        self._namespace += ":g3:" + fingerprint(self.identity)
        self._http = config.get("master_http_url", "").rstrip("/")
        self._http_timeout = float(config.get("metadata_timeout_s", 5))
        if not 0 < self._http_timeout <= 60:
            raise ValueError(
                "Mooncake metadata timeout must be between 0 and 60 seconds"
            )
        if list_keys is None and not self._http.startswith(("http://", "https://")):
            raise ValueError(
                "Mooncake requires the Master's master_http_url for shared indexing"
            )
        self.max_rows = pool.plan.slot_rows
        slots = pool._pool._slots
        stores = slots[0].kv_cache
        first = stores[0].k
        self._device = first.device
        self._layers = len(stores)
        self._kv_shape = tuple(first.shape[1:])
        self._dtype = first.dtype
        self._metadata_bytes = config.get("metadata_buffer_bytes", 64 << 20)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="block0-put"
        )
        self.client = client
        try:
            if replicate_config_factory is None:
                from mooncake.store import ReplicateConfig

                replicate_config_factory = ReplicateConfig
            self._replicate_config_factory = replicate_config_factory
            if self.client is None:
                from mooncake.store import MooncakeDistributedStore

                rdma_devices = block0_rdma_devices(config)
                self.client = MooncakeDistributedStore()
                # SDK put/get need a registered workspace for conditions and
                # manifests. KV still transfers directly from/to GPU pointers.
                metadata_bytes = config.get("metadata_buffer_bytes", 64 << 20)
                if type(metadata_bytes) is not int or metadata_bytes <= 0:
                    raise ValueError("Mooncake metadata_buffer_bytes must be positive")
                rc = self.client.setup(
                    config.get("local_hostname")
                    or socket.gethostbyname(socket.gethostname()),
                    config.get("metadata_server", "P2PHANDSHAKE"),
                    0,
                    metadata_bytes,
                    "rdma",
                    rdma_devices,
                    config["master_server_address"],
                )
                if rc != 0:
                    raise CacheTransferError(f"Mooncake setup failed: {rc}")
            for slot in slots:
                for layer in slot.kv_cache:
                    for tensor in (layer.k, layer.v):
                        storage = tensor.untyped_storage()
                        ptr = storage.data_ptr()
                        if ptr not in self._registered:
                            rc = self.client.register_buffer(ptr, storage.nbytes())
                            if rc != 0:
                                raise CacheTransferError(
                                    f"Mooncake buffer registration failed: {rc}"
                                )
                            # Keep the allocation alive until all transfers and unregister finish.
                            self._registered[ptr] = tensor
        except BaseException:
            self.close()
            raise
        from sglang.multimodal_gen.runtime.realtime.cache_archive import CacheArchive
        from sglang.multimodal_gen.runtime.realtime.cache_tiers import NativeReplicas

        try:
            self.replicas = NativeReplicas(
                self.client, self.tiers, read_object=self._read_object
            )
            self.archive = (
                CacheArchive(self.tiers, self._namespace) if self.tiers.l3 else None
            )
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _base(self, key):
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError("Invalid Block0 cache key")
        return f"{self._namespace}:{key}"

    def _layer_keys(self, key, artifact_id=None):
        artifact_id = artifact_id or self._artifacts.get(key)
        base = f"{self._base(key)}:a{artifact_id}:r{self.log_rank}"
        return tuple(f"{base}:kv:l{layer:02d}" for layer in range(self._layers))

    def _cond_key(self, key, artifact_id=None):
        artifact_id = artifact_id or self._artifacts.get(key)
        return f"{self._base(key)}:a{artifact_id}:r{self.log_rank}:cond"

    def _manifest_key(self, key):
        return self._base(key) + ":manifest"

    def _list_objects(self):
        if self._list_keys_override is not None:
            return self._list_keys_override()
        with urllib.request.urlopen(
            self._http + "/get_all_keys", timeout=self._http_timeout
        ) as response:
            return response.read().decode("utf-8").splitlines()

    def _cache_keys(self):
        prefix, suffix = self._namespace + ":", ":manifest"
        return sorted(
            {
                item[len(prefix) : -len(suffix)]
                for item in self._list_objects()
                if item.startswith(prefix)
                and item.endswith(suffix)
                and _KEY.fullmatch(item[len(prefix) : -len(suffix)])
            }
        )

    @property
    def entry_count(self):
        # Session disposal logs this property. Never do network IO there; these
        # client-known keys are hints, while status() checks shared residency.
        return len(self._known)

    def may_contain(self, key):
        # CPU-only speculation hint. No network IO or collectives on this thread.
        return key in self._known

    @staticmethod
    def _agree(error):
        errors = prep.coordinate_validation(error)
        if any(value is not None for value in errors):
            raise CacheTransferError(
                f"Block0 transfer failed across SP ranks: {errors}"
            )

    def _replicate_config(self, cache_key, count=1):
        config = self._replicate_config_factory()
        # One native group spans every SP shard and its publication/index
        # metadata. Business game/group labels are independent associations.
        config.group_ids = [self._base(cache_key)] * count
        return config

    def _put(self, key, payload, cache_key):
        rc = self.client.put(key, payload, self._replicate_config(cache_key))
        if rc == _ALREADY_EXISTS and self.client.is_exist(key) == 1:
            return
        if rc != 0:
            raise CacheTransferError(f"Mooncake metadata write failed: {rc}")

    def _object_keys(self, key, artifact):
        result = []
        for rank in range(self.world_size):
            base = f"{self._base(key)}:a{artifact}:r{rank}"
            result.extend(f"{base}:kv:l{layer:02d}" for layer in range(self._layers))
            result.append(base + ":cond")
        return result

    def bind_world(
        self, key, world_id, *, s0_sha256=None, observer_token=None, context=None
    ):
        if world_id is None:
            return
        from sglang.multimodal_gen.runtime.realtime.cache_archive import world_token

        world_token(world_id)
        old = self._worlds.get(key)
        if old is not None and old != world_id:
            raise ValueError("A cache key cannot change world ownership")
        self._worlds[key] = world_id
        from sglang.multimodal_gen.runtime.realtime.cache_observer import (
            verify_recording,
        )

        from .cache_logging import log_context

        self._observer_targets[key] = verify_recording(observer_token, world_id)
        self._observer_contexts[key] = log_context(**(context or {}))
        if s0_sha256 is not None:
            if not isinstance(s0_sha256, str) or not _KEY.fullmatch(s0_sha256):
                raise ValueError("Invalid cache input digest")
            self._input_s0[key] = s0_sha256
        # This is only per-client provenance, not the shared cache index.
        # Bound it independently of native RAM/SSD retention.
        while len(self._worlds) > 4096:
            old_key = next(iter(self._worlds))
            if (
                old_key == key
                or old_key in self._loads
                or old_key in self._cloud_jobs
                or (self.pending and self.pending[0].key == old_key)
            ):
                self._worlds[old_key] = self._worlds.pop(old_key)
                continue
            self._worlds.pop(old_key, None)
            self._input_s0.pop(old_key, None)
            self._observer_targets.pop(old_key, None)
            self._observer_contexts.pop(old_key, None)
            self._artifacts.pop(old_key, None)
            self._known.discard(old_key)

    def _observe(self, key, event, *, target=_UNSET, context=None, **fields):
        if self.observer is None:
            return
        self.observer.offer(
            self._observer_targets.get(key) if target is _UNSET else target,
            dict(
                event=event,
                cache_key=key,
                rank=self.log_rank,
                **(
                    self._observer_contexts.get(key, {}) if context is None else context
                ),
                **fields,
            ),
        )

    def _archive_manifest(self, entry):
        keys = self._object_keys(entry.key, entry.data.artifact_id)
        sizes = {key: self.client.get_size(key) for key in keys}
        if any(type(n) is not int or n <= 0 for n in sizes.values()):
            raise CacheTransferError("A cache artifact disappeared before archival")
        return dict(self._manifest(entry), objects=sizes)

    def _import_object(self, name, data, key):
        # Registered host storage avoids a second whole-KV SDK workspace.
        value = ctypes.create_string_buffer(data)
        ptr = ctypes.addressof(value)
        rc = self.client.register_buffer(ptr, len(data))
        if rc != 0:
            raise CacheTransferError("Unable to register archive restore buffer")
        try:
            codes = self.client.batch_put_from_multi_buffers(
                [name], [[ptr]], [[len(data)]], self._replicate_config(key)
            )
            if list(codes) not in ([0], [_ALREADY_EXISTS]):
                raise CacheTransferError("Unable to restore a cache archive member")
        finally:
            self.client.unregister_buffer(ptr)

    def _export_object(self, name):
        """Large KV/conditioning objects must not depend on SDK scratch size."""
        size = self.client.get_size(name)
        if type(size) is not int or not 0 < size <= self.tiers.max_object_bytes:
            raise CacheTransferError("Cache object exceeds its transfer budget")
        return bytes(self._read_object(name, size))

    def _read_object(self, name, size):
        """Read one size-checked member using SDK scratch or registered host RAM."""
        if size <= self._metadata_bytes:
            data = self.client.get(name)
            if len(data) != size:
                raise CacheTransferError("Cache object read is incomplete")
            return data
        buffer = ctypes.create_string_buffer(size)
        ptr = ctypes.addressof(buffer)
        if self.client.register_buffer(ptr, size) != 0:
            raise CacheTransferError("Unable to register cache export buffer")
        try:
            actual = self.client.batch_get_into_multi_buffers([name], [[ptr]], [[size]])
            if list(actual) != [size]:
                raise CacheTransferError("Cache export read is incomplete")
            return buffer
        finally:
            self.client.unregister_buffer(ptr)

    def _load_tiers(self, key, marker_layers=None, *, submitted_at=None):
        from sglang.multimodal_gen.runtime.realtime.cache_tiers import StageTimes

        queued_s = 0.0 if submitted_at is None else time.monotonic() - submitted_at
        timing = StageTimes()

        def result(tier, entry=None):
            if entry is None:
                with timing.measure("conditions_deserialize"):
                    entry = self._read_local(key, require_memory=True)
            return (
                None
                if entry is None
                else {"source_layer": tier, "stages": timing.stages, "entry": entry}
            )

        if not self.tiers.l1:
            return None
        deadline = time.monotonic() + self.tiers.deadline_s
        marker = self._manifest_key(key)
        with timing.measure("native_index"):
            states = (
                marker_layers
                if marker_layers is not None
                else self.replicas.layers([marker]).get(marker, set())
            )
        if states:
            with timing.measure("native_marker_read"):
                raw = self.client.get(marker)
            header = json.loads(raw) if raw else None
            if header:
                artifact = header.get("artifact_id")
                if isinstance(artifact, str) and re.fullmatch(
                    r"[0-9a-f]{32}", artifact
                ):
                    keys = [marker, *self._object_keys(key, artifact)]
                    before = self.replicas.layers(keys)
                    if all("l1" in layers for layers in before.values()):
                        # Reading a disk-only marker can promote the entire native group.
                        return result("l2" if states == {"l2"} else "l1")
                    if self.tiers.l2:
                        try:
                            with timing.measure("l2_promotion"):
                                promoted = self.replicas.promote(keys, deadline)
                        except CacheTransferError:
                            promoted = False
                        if promoted:
                            return result("l2")
        world = self._worlds.get(key)
        if self.archive is None or world is None:
            return None
        with timing.measure("l3_metadata"):
            cloud = self.archive.probe(world, key, queued_s=queued_s)
        if cloud is None:
            return None
        native = cloud["native"]
        if (
            native.get("identity") != self.identity
            or native.get("sp_world_size") != self.world_size
        ):
            return None
        native = self.archive.restore(
            world,
            key,
            lambda name, data: self._import_object(name, data, key),
            deadline=deadline,
            found=cloud,
            timing=timing,
        )
        if native is None:
            return None
        old = self.client.get(marker)
        if old and json.loads(old).get("artifact_id") != native["artifact_id"]:
            with timing.measure("conditions_deserialize"):
                existing = self._read_local(key, require_memory=True)
            if existing is not None:
                return result("l1", existing)
            # A partial resident marker cannot mask a complete cloud artifact.
            # Native lease rejection remains a retryable miss, never force-delete.
            if self.client.remove(marker) not in (0, _NOT_FOUND):
                return None
        self._put(marker, canonical_manifest(native), key)
        keys = [marker, *self._object_keys(key, native["artifact_id"])]
        if not self.replicas.memory_ready(keys):
            return None
        with timing.measure("conditions_deserialize"):
            winner = self._read_local(key, require_memory=True)
        if winner is None:
            return None
        return result(
            "l3" if winner.data.artifact_id == native["artifact_id"] else "l1", winner
        )

    def _recall(self, key, allow_disk, resident_only):
        if not self.tiers.l1:
            return None
        marker = self._manifest_key(key)

        def consume(value):
            if not value:
                return None
            self._origins[key] = {k: v for k, v in value.items() if k != "entry"}
            entry = value.get("entry")
            if entry is not None and self.replicas.memory_ready(
                [marker, *entry.data.keys, self._cond_key(key, entry.data.artifact_id)]
            ):
                return entry
            return None

        probe_error = None
        # Preserve the actual source layer when a prior background load won.
        with self._load_lock:
            completed = self._loads.get(key)
            pending = completed is not None and not completed.done()
            if completed is not None and not pending:
                self._loads.pop(key, None)
                try:
                    origin = completed.result()
                    entry = consume(origin)
                    if entry is not None:
                        return entry
                except ArchiveProbeTimeout as exc:
                    probe_error = exc
                except Exception:
                    pass
        # A hot hit never probes OSS and never repairs durable copies.
        # Partially promoted groups can become readable before their background
        # load has finished recording where the bytes came from. Join that load.
        marker_layers = (
            None if pending else self.replicas.layers([marker]).get(marker, set())
        )
        if marker_layers and "l1" in marker_layers:
            entry = self._read_local(key, require_memory=True)
            if entry is not None:
                return entry
        if probe_error is not None:
            raise probe_error
        if resident_only or not (self.tiers.l2 or self.tiers.l3):
            return None
        with self._load_lock:
            future = self._loads.get(key)
            if future is None:
                for other, task in list(self._loads.items()):
                    if task.done():
                        self._loads.pop(other, None)
                        try:
                            source = task.result()
                            if source:
                                self._origins[other] = {
                                    k: v for k, v in source.items() if k != "entry"
                                }
                        except Exception:
                            pass
                while len(self._origins) > 64:
                    self._origins.pop(next(iter(self._origins)))
                if (
                    sum(not f.done() for f in self._loads.values())
                    >= self.tiers.max_pending
                ):
                    return None
                future = self._loads[key] = self._recall_io.submit(
                    self._load_tiers,
                    key,
                    marker_layers or None,
                    submitted_at=time.monotonic(),
                )
        timeout = (
            self.tiers.deadline_s if allow_disk else self.tiers.online_wait_ms / 1000
        )
        if allow_disk and self.tiers.l3 and not future.running() and not future.done():
            # A different key may already be downloading. An unstarted recall
            # cannot confirm a hit, and must not import stale KV after a miss.
            until = time.monotonic() + timeout
            wait([future], timeout=min(timeout, self.tiers.oss_probe_timeout_s))
            if future.cancel():
                with self._load_lock:
                    if self._loads.get(key) is future:
                        self._loads.pop(key)
                raise ArchiveProbeTimeout(
                    "OSS cache metadata probe expired in the recall queue"
                )
            timeout = max(0, until - time.monotonic())
        try:
            origin = future.result(timeout=timeout)
        except TimeoutError:
            return None
        except ArchiveProbeTimeout:
            with self._load_lock:
                if self._loads.get(key) is future:
                    self._loads.pop(key)
            raise
        except Exception:
            origin = None
        with self._load_lock:
            if self._loads.get(key) is future:
                self._loads.pop(key, None)
        if origin is None:
            return None
        return consume(origin)

    def request_persistence(self, entry):
        """Return immediately; CPU workers wait for offline durable completion."""
        import hashlib

        from sglang.multimodal_gen.runtime.realtime.cache_tiers import (
            persistence_state_path,
        )

        target, world = self.tiers.durable_tier, self._worlds.get(entry.key)
        if target is None or world is None:
            return None
        from sglang.multimodal_gen.runtime.conditioning.worker import broadcast_receipt

        descriptor = broadcast_receipt(
            dict(
                schema="zing-block0-tier-state-v1",
                key=entry.key,
                artifact_id=entry.data.artifact_id,
                world_id=world,
                attempt_id=uuid4().hex,
                namespace_sha256=hashlib.sha256(self._namespace.encode()).hexdigest(),
                tier=target,
            )
            if self.log_rank == 0
            else None
        )
        if self.log_rank != 0:
            return descriptor
        path = persistence_state_path(self.tiers.state_directory, descriptor)

        def record(result):
            import os

            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary = path.with_suffix("." + uuid4().hex + ".tmp")
            try:
                temporary.write_text(
                    json.dumps(dict(descriptor=descriptor, result=result))
                )
                os.chmod(temporary, 0o600)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

        with self._load_lock:
            old = self._durability.get(str(path))
            if old is not None:
                return descriptor
            self._durability = {
                k: f for k, f in self._durability.items() if not f.done()
            }
            if len(self._durability) >= self.tiers.max_pending:
                record({"state": "failed", "code": "CACHE_WRITEBACK_BUSY"})
                return descriptor
            record({"state": "pending"})
            fence = self.archive.submission_fence(world) if target == "l3" else None

            def persist():
                try:
                    from sglang.multimodal_gen.runtime.realtime.cache_tiers import (
                        prune_persistence_states,
                    )

                    prune_persistence_states(self.tiers.state_directory)
                    result = self._persist_entry(
                        entry, world, time.monotonic() + self.tiers.deadline_s, fence
                    )
                except Exception:
                    result = {"state": "failed", "code": "CACHE_PERSISTENCE_FAILED"}
                record(result)
                return result

            self._durability[str(path)] = self._io.submit(persist)
        return descriptor

    def persistence_status(self, descriptor):
        from sglang.multimodal_gen.runtime.realtime.cache_tiers import (
            read_persistence_state,
        )

        return read_persistence_state(self.tiers.state_directory, descriptor)

    def _persist_entry(self, entry, world, deadline, fence=None):
        keys = [
            self._manifest_key(entry.key),
            *self._object_keys(entry.key, entry.data.artifact_id),
        ]
        target = self.tiers.durable_tier
        if target == "l2":
            ready = all("l2" in v for v in self.replicas.layers(keys).values())
            if not ready and not self.tiers.l1_to_l2:
                return {"state": "failed", "code": "CACHE_WRITEBACK_DISABLED"}
            ready = ready or self.replicas.wait(keys, "l2", deadline)
            return {
                "state": "persisted" if ready else "failed",
                "tier": "l2",
                "artifact_id": entry.data.artifact_id,
            }
        self.archive.check_submission(world, fence)
        cloud = self.archive.find_durable(world, entry.key, deadline=deadline)
        if cloud is not None:
            return {
                "state": "persisted",
                "tier": "l3",
                "artifact_id": cloud["artifact_id"],
                "deduplicated": True,
                "bytes": 0,
            }
        if not self.tiers.cloud_write:
            return {"state": "failed", "code": "CACHE_WRITEBACK_DISABLED"}
        if self.tiers.l2_to_l3 and not self.tiers.l1_to_l3:
            if not self.replicas.wait(keys, "l2", deadline):
                return {"state": "failed", "code": "CACHE_L2_WRITE_TIMEOUT"}
        future = self.archive.submit(
            world,
            entry.key,
            self._archive_manifest(entry),
            self._export_object,
            expected_fence=fence,
        )
        if future is None:
            return {"state": "failed", "code": "CACHE_WRITEBACK_BUSY"}
        return future.result(timeout=max(0, deadline - time.monotonic()))

    def _start_archive(self, key):
        entry = self._new_entries.get(key)
        world = self._worlds.get(key)
        if entry is None or not world:
            return
        self._new_entries.pop(key, None)
        if self.log_rank != 0:
            return
        target = getattr(entry, "observer_target", None)
        context = dict(getattr(entry, "observer_context", {}))

        fence = (
            self.archive.submission_fence(world) if self.archive is not None else None
        )

        def observe(event, **fields):
            self._observe(
                key,
                event,
                target=target,
                context=context,
                **{"artifact_id": entry.data.artifact_id, **fields},
            )

        def archive():
            from sglang.multimodal_gen.runtime.realtime.cache_tiers import StageTimes

            timing = StageTimes()
            deadline = time.monotonic() + self.tiers.deadline_s
            keys = [
                self._manifest_key(key),
                *self._object_keys(key, entry.data.artifact_id),
            ]
            cloud = None
            if self.archive is not None and self.tiers.l1_to_l3:
                cloud = self.archive.submit(
                    world,
                    key,
                    self._archive_manifest(entry),
                    self._export_object,
                    expected_fence=fence,
                )
            ready = False
            if self.tiers.l2 and self.tiers.l1_to_l2:
                with timing.measure("l2_replica_wait"):
                    ready = self.replicas.wait(keys, "l2", deadline)
                observe(
                    "l2_writeback",
                    state="persisted" if ready else "unconfirmed",
                    tier="l2",
                    stages=dict(timing.stages),
                    completed_epoch_ms=time.time() * 1000,
                )
            else:
                observe("l2_writeback", state="disabled", tier="l2")
            if self.archive is None or not self.tiers.cloud_write:
                return {"state": "disabled", "tier": "l3"}
            if not self.tiers.l1_to_l3:
                if not ready:
                    return {
                        "state": "failed",
                        "code": "CACHE_L2_WRITE_TIMEOUT",
                        "stages": timing.stages,
                    }
                cloud = self.archive.submit(
                    world,
                    key,
                    self._archive_manifest(entry),
                    self._export_object,
                    expected_fence=fence,
                )
            return (
                cloud.result(timeout=max(0, deadline - time.monotonic()))
                if cloud is not None
                else {"state": "skipped", "reason": "queue_full"}
            )

        with self._load_lock:
            self._cloud_jobs = {
                k: f for k, f in self._cloud_jobs.items() if not f.done()
            }
            if len(self._cloud_jobs) < self.tiers.max_pending:
                future = self._cloud_jobs[key] = self._io.submit(archive)

                def observed(done):
                    try:
                        result = done.result()
                    except Exception:
                        result = {"state": "failed", "code": "CACHE_WRITEBACK_FAILED"}
                    observe("l3_writeback", **result)

                future.add_done_callback(observed)
            else:
                observe("writeback", state="skipped", reason="queue_full")

    def _check_group(self, key, associations=()):
        # Manifest first: reading the newly added member refreshes leases for
        # every current group member, including indices written before it.
        raw = self.client.get(self._manifest_key(key))
        if not raw:
            return False
        artifact = json.loads(raw).get("artifact_id")
        if not isinstance(artifact, str) or not re.fullmatch(r"[0-9a-f]{32}", artifact):
            return False
        keys = [self._manifest_key(key)]
        for rank in range(self.world_size):
            base = f"{self._base(key)}:a{artifact}:r{rank}"
            keys.extend(f"{base}:kv:l{layer:02d}" for layer in range(self._layers))
            keys.append(base + ":cond")
        keys.extend(associations)
        present = list(self.client.batch_is_exist(keys))
        if len(present) != len(keys) or any(code not in (0, 1) for code in present):
            raise CacheTransferError(f"Mooncake group check failed: {present}")
        if not present[0]:
            # Async fills can bind their index before the manifest is published.
            return False
        if present != [1] * len(keys):
            raise CacheTransferError("Mooncake published group is incomplete")
        return True

    def _manifest(self, entry):
        return dict(
            schema="zing-block0-mooncake-v3"
            if entry.data.encoding == kv_codec.BF16
            else "zing-block0-mooncake-v4",
            kv_encoding=entry.data.encoding,
            artifact_id=entry.data.artifact_id,
            baseline_s0_sha256=entry.baseline_s0_sha256,
            identity=self.identity,
            sp_world_size=self.world_size,
            geometry=prep._block0_geometry(entry),
            layer_bytes=entry.data.layer_bytes(),
        )

    def _elements_per_row(self):
        import math

        return math.prod(self._kv_shape)

    def _read_local(self, key, *, require_memory=False, manifest=None):
        if manifest is None:
            raw = self.client.get(self._manifest_key(key))
            if not raw:
                return None
            manifest = json.loads(raw)
        artifact = manifest.get("artifact_id")
        if not isinstance(artifact, str) or not re.fullmatch(r"[0-9a-f]{32}", artifact):
            return None
        geometry = manifest["geometry"]
        rows = geometry["prefix_rows"]
        expected = dict(
            layers=self._layers,
            parts=2,
            prefix_rows=rows,
            kv_shape=list(self._kv_shape),
            dtype=str(self._dtype).replace("torch.", ""),
        )
        if (
            manifest.get("schema")
            != (
                "zing-block0-mooncake-v3"
                if self.requires_raw_prefix
                else "zing-block0-mooncake-v4"
            )
            or manifest.get("kv_encoding", kv_codec.BF16) != self.kv_encoding
            or manifest.get("identity") != self.identity
            or manifest.get("sp_world_size") != self.world_size
            or type(rows) is not int
            or not 0 < rows <= self.max_rows
            or any(geometry.get(k) != v for k, v in expected.items())
            or manifest.get("layer_bytes")
            != (
                rows
                * self._elements_per_row()
                * torch.empty((), dtype=self._dtype).element_size()
                * 2
                if self.requires_raw_prefix
                else kv_codec.layer_bytes(rows, *self._kv_shape)
            )
        ):
            return None
        keys = self._layer_keys(key, artifact)
        if require_memory and not self.replicas.memory_ready(
            [*keys, self._cond_key(key, artifact)]
        ):
            return None
        if self.client.batch_is_exist([*keys, self._cond_key(key, artifact)]) != [1] * (
            self._layers + 1
        ):
            return None
        payload = self._export_object(self._cond_key(key, artifact))
        if not payload:
            return None
        conditions = decode_conditions(
            torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        )
        positions = conditions.get("positions")
        if (
            not isinstance(positions, torch.Tensor)
            or positions.shape != (rows, 3)
            or not torch.isfinite(positions).all()
        ):
            return None
        self._artifacts[key] = artifact
        return prep.PreparationEntry(
            key,
            conditions["inputs"],
            conditions.get("conditioning_identity"),
            conditions.get("request"),
            positions,
            None,
            SharedKV(
                (self._layers, 2, rows, *self._kv_shape),
                self._dtype,
                keys,
                artifact,
                self.kv_encoding,
            ),
            baseline_s0_sha256=manifest.get("baseline_s0_sha256"),
        )

    def lookup(self, key, *, allow_disk=False, protected=(), resident_only=False):
        started = time.perf_counter()
        self.finish_fill()
        entry = error = None
        try:
            entry = self._recall(key, allow_disk, resident_only)
        except ArchiveProbeTimeout:
            error = "l3_probe_timeout"
        except Exception as exc:
            error = type(exc).__name__
        rows = None if entry is None else int(entry.positions.shape[0])
        artifact = entry.data.artifact_id if entry is not None else None
        votes = prep.coordinate_validation((key, rows, artifact, error))
        hit = entry is not None and all(v == (key, rows, artifact, None) for v in votes)
        if any(v[3] == "l3_probe_timeout" for v in votes):
            error = "l3_probe_timeout"
        recalled = self._origins.pop(key, "l1") if hit else None
        origin = (
            recalled.get("source_layer") if isinstance(recalled, dict) else recalled
        )
        self.last_lookup = {
            "stages": recalled.get("stages", {}) if isinstance(recalled, dict) else {},
            "cache_result": origin + "_hit"
            if hit
            else "miss"
            if self.tiers.l1
            else "bypass",
            "local_layer": "l1",
            "backend": self.backend,
            "kv_encoding": self.kv_encoding,
            "lookup_host_ms": (time.perf_counter() - started) * 1000,
            "disk_allowed": allow_disk and self.tiers.l2,
            "source_layer": origin,
            "artifact_id": entry.data.artifact_id if hit else None,
            "baseline_s0_sha256": entry.baseline_s0_sha256 if hit else None,
            "online_s0_sha256": self._input_s0.get(key),
            "miss_reason": None
            if hit
            else "l1_disabled"
            if not self.tiers.l1
            else error or "not_resident",
            "storage": {},
        }
        self._observe(key, "lookup", **self.last_lookup)
        self.hits += int(hit)
        self.misses += int(not hit and self.tiers.l1)
        if hit:
            self._known.add(key)
        else:
            self._known.discard(key)
        cache_event(
            "lookup",
            "hit" if hit else "miss",
            cache_key=key,
            rank=self.log_rank,
            started=started,
            tier=origin,
            backend=self.backend,
            allow_disk=allow_disk and self.tiers.l2,
            reason=None if hit else error or "not_resident",
        )
        return entry if hit else None

    def restore(self, cache, entry):
        started = time.perf_counter()
        error = None
        views = []
        packed = []
        registered = []
        positions = blocks = None
        try:
            if self._transfer_fault:
                raise CacheTransferError("Cache transfer cleanup is unconfirmed")
            rows = entry.positions.shape[0]
            positions = entry.positions.to(self._device)
            blocks = torch.zeros(rows, dtype=torch.long, device=self._device)
            if entry.data.encoding != self.kv_encoding:
                raise CacheTransferError("Block0 encoding differs from this worker")
            if self.requires_raw_prefix:
                for layer, slot in enumerate(cache.storage):
                    k, v = cache._reserve_store(layer, rows, slot.k, slot.v)
                    views.append((k[:rows], v[:rows]))
                ptrs, sizes = self._buffers(views, rows)
            else:
                for _ in range(self._layers):
                    value = kv_codec.allocate(
                        rows, *self._kv_shape, device=self._device
                    )
                    packed.append(value)
                    if (
                        self.client.register_buffer(value.data_ptr(), value.numel())
                        != 0
                    ):
                        raise CacheTransferError(
                            "Compressed restore buffer registration failed"
                        )
                    registered.append(value.data_ptr())
                ptrs = [[v.data_ptr()] for v in packed]
                sizes = [[v.numel()] for v in packed]
            if self._device.type == "cuda":
                torch.cuda.current_stream(self._device).synchronize()
            codes = self.client.batch_get_into_multi_buffers(
                entry.data.keys, ptrs, sizes
            )
            if list(codes) != [sum(pair) for pair in sizes]:
                raise CacheTransferError(f"Mooncake incomplete KV read: {codes}")
        except Exception as exc:
            error = type(exc).__name__
        finally:
            # Completion includes GPU visibility, not just Python SDK return.
            if self._device.type == "cuda":
                try:
                    torch.cuda.synchronize(self._device)
                except Exception as exc:
                    error = type(exc).__name__
            if not self._release_owned_buffers(registered, packed):
                error = "CompressedRestoreUnregisterFailed"
        self._agree(error)
        return prep.PrefixRestore(
            entry,
            None,
            None,
            positions,
            blocks,
            views,
            (time.perf_counter() - started) * 1000,
            started,
            compressed_payload=tuple(packed) if packed else None,
        )

    def _release_owned_buffers(self, registered, values):
        """Keep unknown native registrations alive and stop further transfers."""
        by_ptr = {v.untyped_storage().data_ptr(): v for v in values}
        complete = True
        for ptr in registered:
            try:
                released = self.client.unregister_buffer(ptr) in (None, 0)
            except Exception:
                released = False
            if not released:
                self._registered[ptr] = by_ptr[ptr]
                self._transfer_fault = "OwnedBufferUnregisterFailed"
                complete = False
        return complete

    def _buffers(self, layers, rows, *, owned_export=False):
        ptrs, sizes = [], []
        if len(layers) != self._layers:
            raise ValueError("Block0 layer count differs from registered GPU layout")
        for pair in layers:
            if len(pair) != 2 or any(
                not v.is_contiguous() or v.shape[0] != rows for v in pair
            ):
                raise ValueError("Block0 requires contiguous K/V buffers")
            for value in pair:
                storage = value.untyped_storage()
                if storage.data_ptr() not in self._registered and not owned_export:
                    raise ValueError("Block0 buffer is outside the registered GPU pool")
                if (
                    value.dtype != self._dtype
                    or tuple(value.shape[1:]) != self._kv_shape
                    or value.device != self._device
                ):
                    raise ValueError(
                        "Block0 KV buffer geometry differs from registered layout"
                    )
            ptrs.append([v.data_ptr() for v in pair])
            sizes.append([v.numel() * v.element_size() for v in pair])
        return ptrs, sizes

    def can_store(self, rows):
        # There is no worker-local host budget. Shared Store capacity is decided
        # by put; admission can only check the registered GPU slot's row limit.
        return 0 < rows <= self.max_rows

    def fill(
        self,
        key,
        inputs,
        conditioning_identity,
        request,
        cache,
        *,
        force=False,
        required=True,
    ):
        self.finish_fill()
        if self.pending is not None:
            if required:
                self.finish_fill(wait=True)
            else:
                self.write_skips += 1
                self._observe(key, "l1_fill", state="skipped", reason="writer_busy")
                cache_event(
                    "l1_fill",
                    "skipped",
                    cache_key=key,
                    rank=self.log_rank,
                    reason="writer_busy",
                )
                return False
        prefix = cache.prefix
        rows = int(prefix.positions.shape[0])
        optional = not required
        if optional:
            tensors = [v[:rows] for pair in prefix.layers for v in pair]
            tensors.append(prefix.positions)
            tensors += (
                list(inputs.positive.values())
                + list(inputs.visual.values())
                + list(inputs.audio.values())
            )
            tensors += [
                value
                for item in inputs.condition_media or []
                for value in item.values()
                if isinstance(value, torch.Tensor)
            ]
            size = sum(v.numel() * v.element_size() for v in tensors)
            if not self.requires_raw_prefix:
                size += self._layers * kv_codec.layer_bytes(rows, *self._kv_shape)
            permitted = self.tiers.l1 and 0 < size <= self.tiers.snapshot_bytes
            if not all(prep.coordinate_validation(permitted)):
                self.write_skips += 1
                self._observe(
                    key,
                    "l1_fill",
                    state="skipped",
                    reason="snapshot_budget",
                    snapshot_bytes=size,
                )
                cache_event(
                    "l1_fill",
                    "skipped",
                    cache_key=key,
                    rank=self.log_rank,
                    reason="snapshot_budget",
                )
                return False
        from sglang.multimodal_gen.runtime.conditioning.worker import broadcast_receipt

        artifact = broadcast_receipt(uuid4().hex if self.log_rank == 0 else None)
        self._artifacts[key] = artifact
        entry = ready = error = None
        layers = []
        try:
            if self._transfer_fault:
                raise CacheTransferError("Cache transfer cleanup is unconfirmed")
            if not self.tiers.l1 or not self.can_store(rows):
                raise ValueError("Block0 prefix exceeds the available cache budget")
            owned = not self.requires_raw_prefix or getattr(
                prefix, "export_owned", False
            )
            original = (
                [(pair[0][:rows], pair[1][:rows]) for pair in prefix.layers]
                if self.requires_raw_prefix
                else [(v,) for v in kv_codec.snapshot(cache)]
            )
            if self.requires_raw_prefix:
                self._buffers(original, rows, owned_export=owned)
            # Optional online writeback owns private GPU snapshots. Queued puts
            # must never read the next prompt's mutated KV storage.
            layers = (
                [(k.detach().clone(), v.detach().clone()) for k, v in original]
                if optional and not owned
                else original
            )
            frozen = (
                replace(
                    inputs,
                    positive={
                        k: v.detach().clone() for k, v in inputs.positive.items()
                    },
                    visual={k: v.detach().clone() for k, v in inputs.visual.items()},
                    audio={k: v.detach().clone() for k, v in inputs.audio.items()},
                    conditions=deepcopy(inputs.conditions),
                    media=deepcopy(inputs.media),
                    max_area_config=deepcopy(inputs.max_area_config),
                    condition_media=None
                    if inputs.condition_media is None
                    else [
                        {
                            name: value.detach().clone()
                            if isinstance(value, torch.Tensor)
                            else deepcopy(value)
                            for name, value in item.items()
                        }
                        for item in inputs.condition_media
                    ],
                )
                if optional
                else inputs
            )
            # CPU media is also snapshotted: a later request must not retarget
            # an already queued write through an aliased input buffer.
            positions = (
                prefix.positions.detach().clone() if optional else prefix.positions
            )
            entry = prep.PreparationEntry(
                key,
                frozen,
                conditioning_identity,
                request,
                positions,
                None,
                SharedKV(
                    (self._layers, 2, rows, *self._kv_shape),
                    self._dtype,
                    self._layer_keys(key, artifact),
                    artifact,
                    self.kv_encoding,
                ),
                force=force,
                fill_started=time.perf_counter(),
                baseline_s0_sha256=self._input_s0.get(key),
            )
            entry.observer_target = self._observer_targets.get(key)
            entry.observer_context = dict(self._observer_contexts.get(key, {}))
            entry.optional_write = optional
            if self._device.type == "cuda":
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self._device))
            entry.source_copy_done = ready
        except Exception as exc:
            error = type(exc).__name__
        errors = prep.coordinate_validation(error)
        if any(value is not None for value in errors):
            if required:
                raise CacheTransferError("Block0 snapshot could not be prepared")
            self.write_skips += 1
            self._observe(key, "l1_fill", state="skipped", reason="snapshot_failed")
            cache_event(
                "l1_fill",
                "skipped",
                cache_key=key,
                rank=self.log_rank,
                reason="snapshot_failed",
            )
            return False

        def write():
            device = (
                torch.cuda.device(self._device)
                if self._device.type == "cuda"
                else nullcontext()
            )
            registered = []
            with device:
                try:
                    if ready is not None:
                        ready.synchronize()
                    entry.inputs = prep.inputs_on(frozen, "cpu", snapshot=not optional)
                    entry.positions = positions.detach().cpu().clone()
                    payload = io.BytesIO()
                    torch.save(
                        encode_conditions(
                            dict(
                                inputs=entry.inputs,
                                conditioning_identity=conditioning_identity,
                                request=request,
                                positions=entry.positions,
                            )
                        ),
                        payload,
                    )
                    ptrs, sizes = [], []
                    for pair in layers:
                        ptrs.append([v.data_ptr() for v in pair])
                        sizes.append([v.numel() * v.element_size() for v in pair])
                        if optional or owned:
                            for value in pair:
                                ptr = value.untyped_storage().data_ptr()
                                if (
                                    ptr not in registered
                                    and ptr not in self._registered
                                ):
                                    rc = self.client.register_buffer(
                                        ptr, value.untyped_storage().nbytes()
                                    )
                                    if rc != 0:
                                        raise CacheTransferError(
                                            "Snapshot buffer registration failed"
                                        )
                                    registered.append(ptr)
                    codes = list(
                        self.client.batch_put_from_multi_buffers(
                            entry.data.keys,
                            ptrs,
                            sizes,
                            self._replicate_config(key, self._layers),
                        )
                    )
                    if len(codes) != self._layers or any(
                        c not in (0, _ALREADY_EXISTS) for c in codes
                    ):
                        raise CacheTransferError("Mooncake KV write failed")
                    if (
                        self.client.batch_is_exist(entry.data.keys)
                        != [1] * self._layers
                    ):
                        raise CacheTransferError(
                            "Mooncake KV disappeared before publication"
                        )
                    data = payload.getvalue()
                    if len(data) > self.tiers.max_object_bytes:
                        raise CacheTransferError(
                            "Cache conditioning exceeds its transfer budget"
                        )
                    if len(data) <= self._metadata_bytes:
                        self._put(self._cond_key(key, artifact), data, key)
                    else:
                        self._import_object(self._cond_key(key, artifact), data, key)
                finally:
                    if not self._release_owned_buffers(
                        registered, [v for pair in layers for v in pair]
                    ):
                        raise CacheTransferError(
                            "Snapshot transfer cleanup is unconfirmed"
                        )

        future = None
        error = None
        try:
            future = self._executor.submit(write)
        except Exception as exc:
            error = type(exc).__name__
        try:
            self._agree(error)
        except BaseException:
            if future is not None:
                future.result()
            raise
        self.pending = entry, future
        if owned:
            cache.release_prefix_export()
        self._observe(
            key,
            "l1_fill",
            target=entry.observer_target,
            context=entry.observer_context,
            state="writing",
            artifact_id=artifact,
            baseline_s0_sha256=entry.baseline_s0_sha256,
        )
        return True

    def release_source_copies(self):
        # Only the D2D capture fences source storage reuse. Never wait for Store
        # or OSS from this cleanup hook; writeback owns its private snapshots.
        if self.pending is not None:
            event = getattr(self.pending[0], "source_copy_done", None)
            if event is not None:
                event.synchronize()

    def _publish_candidate(self, entry, indices):
        if self._closed:
            raise CacheTransferError("Cache provider is closing")
        for association in indices:
            self._put(association, b"1", entry.key)
        marker = self._manifest_key(entry.key)
        if self.client.is_exist(marker) == 1:
            existing = self._read_local(entry.key)
            if entry.force or existing is None:
                rc = self.client.remove(marker)
                if rc not in (0, _NOT_FOUND):
                    raise CacheTransferError("Cache publication is leased")
        self._put(
            marker,
            json.dumps(self._manifest(entry), sort_keys=True).encode(),
            entry.key,
        )
        if not self._check_group(entry.key, sorted(indices)):
            raise CacheTransferError("Mooncake manifest disappeared")
        return json.loads(self.client.get(marker))

    def _submit_metadata(self, operation, *args, **kwargs):
        try:
            return self._executor.submit(operation, *args, **kwargs)
        except Exception as error:
            future = Future()
            future.set_exception(error)
            return future

    def finish_fill(self, *, wait=None):
        """Only completed metadata futures are consumed on the generation owner.

        Explicit Prefill may wait. Optional writes advance over later dispatches;
        native publication and condition deserialization stay off that owner.
        All SP collectives remain on the owner, never on background IO threads.
        """
        if self.pending is None:
            return None
        if wait is None:
            wait = not getattr(self.pending[0], "optional_write", False)
        while self.pending is not None:
            entry, future = self.pending
            if not wait and not all(prep.coordinate_validation(future.done())):
                return None
            error = value = None
            try:
                value = future.result()
            except Exception as exc:
                error = type(exc).__name__
            errors = prep.coordinate_validation(error)
            if any(item is not None for item in errors):
                self.pending = None
                self._pending_indices.clear()
                self.write_failures += 1
                self._observe(
                    entry.key,
                    "l1_fill",
                    target=entry.observer_target,
                    context=entry.observer_context,
                    state="failed",
                    artifact_id=entry.data.artifact_id,
                    reason="publication_failed",
                )
                cache_event(
                    "l1_fill",
                    "failed",
                    cache_key=entry.key,
                    rank=self.log_rank,
                    reason="publication_failed",
                    backend=self.backend,
                    level=logging.WARNING,
                )
                return None
            phase = getattr(entry, "publication_phase", "kv")
            if phase == "kv":
                metadata = self._manifest(entry)
                if any(
                    item != metadata for item in prep.coordinate_validation(metadata)
                ):
                    self.pending = None
                    self._pending_indices.clear()
                    self.write_failures += 1
                    return None
                indices, self._pending_indices = self._pending_indices, set()
                entry.publication_phase = "publish"
                if self.log_rank == 0:
                    future = self._submit_metadata(
                        self._publish_candidate, entry, indices
                    )
                else:
                    future = Future()
                    future.set_result(None)
                self.pending = entry, future
                continue
            if phase == "publish":
                from sglang.multimodal_gen.runtime.conditioning.worker import (
                    broadcast_receipt,
                )

                header = broadcast_receipt(value if self.log_rank == 0 else None)
                if header["artifact_id"] == entry.data.artifact_id:
                    winner = entry
                else:
                    # A different complete same-key artifact won publication.
                    entry.publication_phase = "read"
                    self.pending = (
                        entry,
                        self._submit_metadata(
                            self._read_local,
                            entry.key,
                            require_memory=True,
                            manifest=header,
                        ),
                    )
                    continue
            else:
                winner = value
            artifact = winner.data.artifact_id if winner is not None else None
            artifacts = prep.coordinate_validation(artifact)
            if artifact is None or any(item != artifact for item in artifacts):
                self.pending = None
                self.write_failures += 1
                return None
            self.pending = None
            self._pending_indices.clear()
            winner.observer_target = entry.observer_target
            winner.observer_context = entry.observer_context
            self._artifacts[entry.key] = winner.data.artifact_id
            self._known.add(entry.key)
            self._observe(
                entry.key,
                "l1_fill",
                target=entry.observer_target,
                context=entry.observer_context,
                state="ready",
                artifact_id=winner.data.artifact_id,
                candidate_artifact_id=entry.data.artifact_id,
                elapsed_host_ms=(time.perf_counter() - entry.fill_started) * 1000,
            )
            cache_event(
                "l1_fill",
                "completed",
                cache_key=entry.key,
                rank=self.log_rank,
                started=entry.fill_started,
                backend=self.backend,
            )
            if entry.key in self._worlds and (
                self.tiers.l2
                or self.tiers.cloud_write
                or winner.observer_target is not None
            ):
                self._new_entries[entry.key] = winner
                self._start_archive(entry.key)
            return winner

    def register(self, key, game_id=None, group_id=None):
        self._start_archive(key)
        if game_id is None or self.log_rank != 0:
            return
        validate_labels(game_id, group_id)
        self._base(key)
        group = "_" if group_id is None else _component(group_id)
        association = f"{self._namespace}:idx:{_component(game_id)}:{group}:{key}"
        try:
            self._put(association, b"1", key)
            if self.pending is not None and self.pending[0].key == key:
                self._pending_indices.add(association)
            # A late binding adds a cold member after finish_fill's check;
            # check/refresh again so the final group includes this association.
            published = self.client.is_exist(self._manifest_key(key))
            if published == 1:
                self._check_group(key, (association,))
            elif published != 0:
                raise CacheTransferError("Mooncake publication probe failed")
        except Exception as exc:
            cache_event(
                "index_bind",
                "failed",
                cache_key=key,
                rank=self.log_rank,
                reason=type(exc).__name__,
                backend=self.backend,
                level=logging.WARNING,
            )

    def select_keys(self, game_id, group_id=None, cache_key=None):
        validate_labels(game_id, group_id)
        if game_id is None:
            raise ValueError("A cache selection requires a game ID")
        prefix = f"{self._namespace}:idx:{_component(game_id)}:"
        if group_id is not None:
            prefix += _component(group_id) + ":"
        keys = sorted(
            {
                item.rsplit(":", 1)[-1]
                for item in self._list_objects()
                if item.startswith(prefix) and _KEY.fullmatch(item.rsplit(":", 1)[-1])
            }
        )
        if cache_key is not None:
            if cache_key not in keys:
                raise ValueError(
                    "The cache key does not belong to the selected game/group"
                )
            return [cache_key]
        return keys

    def status(self, keys=None):
        from sglang.multimodal_gen.runtime.conditioning.worker import broadcast_receipt

        selection = error = None
        if self.log_rank == 0:
            try:
                selection = self._cache_keys() if keys is None else list(keys)
            except Exception as exc:
                error = type(exc).__name__
        self._agree(error)
        selection = broadcast_receipt(selection)
        local = {}
        for key in selection:
            try:
                entry = self._read_local(key)
            except Exception:
                entry = None
            if entry is not None:
                local[key] = (
                    entry.data.shape,
                    entry.data.storage_bytes(),
                )
        snapshots = prep.coordinate_validation(local)
        resident = [
            key
            for key in selection
            if key in local
            and all(
                snapshot.get(key, (None,))[0] == local[key][0] for snapshot in snapshots
            )
        ]
        states = [dict(cache_key=key, ready=key in resident) for key in selection]
        return dict(
            backend=self.backend,
            scope="shared",
            entries=len(resident),
            cache_keys=resident,
            selected=states,
            all_ready=bool(states) and all(v["ready"] for v in states),
            used_bytes=sum(
                snapshot[key][1] for snapshot in snapshots for key in resident
            ),
            budget_bytes=None,
            reserved_bytes=None,
        )

    def manage(self, operation, keys):
        if operation == "status":
            return self.status(keys)
        self.finish_fill()
        results = []
        for key in dict.fromkeys(keys):
            if operation == "preload":
                entry = self.lookup(key)
                results.append(
                    dict(cache_key=key, status="ready" if entry else "missing")
                )
            elif operation == "unload":
                self._read_local(key)
                # Remove the publication marker first. A leased marker means
                # another client may be reading; report busy without forcing it.
                rc = None
                if self.log_rank == 0:
                    try:
                        rc = self.client.remove(self._manifest_key(key))
                    except Exception:
                        rc = "remove_failed"
                from sglang.multimodal_gen.runtime.conditioning.worker import (
                    broadcast_receipt,
                )

                rc = broadcast_receipt(rc)
                if rc not in (0, _NOT_FOUND):
                    return dict(code="cache_busy", results=results)
                try:
                    codes = self.client.batch_remove(
                        [self._cond_key(key), *self._layer_keys(key)]
                    )
                    failed = len(codes) != self._layers + 1 or any(
                        c not in (0, _NOT_FOUND) for c in codes
                    )
                except Exception:
                    failed = True
                if any(prep.coordinate_validation(failed)):
                    return dict(code="cache_busy", results=results)
                self._known.discard(key)
                results.append(
                    dict(cache_key=key, status="unloaded" if rc == 0 else "missing")
                )
            else:
                raise ValueError("Unknown Block0 cache operation")
        return dict(self.status(keys), results=results)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)
        self._recall_io.shutdown(wait=True)
        if getattr(self, "archive", None) is not None:
            self.archive.close()
        self._io.shutdown(wait=True)
        if getattr(self, "observer", None) is not None:
            self.observer.close()
        if self.client is not None:
            for ptr in self._registered:
                try:
                    self.client.unregister_buffer(ptr)
                except Exception:
                    logger.warning(
                        "Cache buffer unregister failed during close", exc_info=True
                    )
            # The client must stop native transfers before any remaining owned
            # allocations can be freed. If close raises, retain their references.
            self.client.close()
            self._registered.clear()
