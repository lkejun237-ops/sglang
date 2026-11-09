# SPDX-License-Identifier: Apache-2.0
"""Memory-only Block0 provider; Mooncake owns the bytes, workers own GPU slots."""

import atexit
import io
import json
import logging
import re
import socket
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from . import preparation_cache as prep
from .block0_conditions import decode_conditions, encode_conditions
from .block0_identity import fingerprint
from .cache_index import _component, validate_labels
from .cache_logging import cache_event

# Mooncake 0.3.13 public ErrorCode values. Never force-remove leased objects.
_NOT_FOUND = -704
_ALREADY_EXISTS = -705
_KEY = re.compile(r"[0-9a-f]{64}\Z")


class CacheTransferError(RuntimeError):
    """A cache operation failed consistently across the participating SP group."""


@dataclass(frozen=True)
class SharedKV:
    shape: tuple
    dtype: torch.dtype
    keys: tuple[str, ...]

    def numel(self):
        import math

        return math.prod(self.shape)

    def element_size(self):
        return torch.empty((), dtype=self.dtype).element_size()


class MooncakeBackend(prep.Block0PreparationCache):
    """Pure Store client: no pinned KV arena, disk cache, or persistence thread.

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
        self.identity = identity
        self.log_rank = rank
        self.world_size = world_size
        self.hits = self.misses = self.evictions = 0
        self.write_failures = 0
        self.pending = None
        self._pending_indices = set()
        self._known = set()
        self._registered = {}
        self._closed = False
        self._list_keys_override = list_keys
        self._namespace = config.get("namespace", "zing_b0")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self._namespace):
            raise ValueError(
                "Mooncake namespace must contain only letters, digits, _, - or ."
            )
        # Existing objects cannot acquire group membership retroactively. Keep
        # grouped entries separate from the original ungrouped wire format.
        self._namespace += ":g2:" + fingerprint(identity)
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

                protocol = config.get("protocol", "tcp")
                if protocol not in ("tcp", "rdma"):
                    raise ValueError("Mooncake protocol must be tcp or rdma")
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
                    protocol,
                    config.get("rdma_devices", ""),
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
        atexit.register(self.close)

    def _base(self, key):
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError("Invalid Block0 cache key")
        return f"{self._namespace}:{key}"

    def _layer_keys(self, key):
        base = f"{self._base(key)}:r{self.log_rank}"
        return tuple(f"{base}:kv:l{layer:02d}" for layer in range(self._layers))

    def _cond_key(self, key):
        return f"{self._base(key)}:r{self.log_rank}:cond"

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

    def _check_group(self, key, associations=()):
        # Manifest first: reading the newly added member refreshes leases for
        # every current group member, including indices written before it.
        keys = [self._manifest_key(key)]
        for rank in range(self.world_size):
            base = f"{self._base(key)}:r{rank}"
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
            schema="zing-block0-mooncake-v2",
            identity=self.identity,
            sp_world_size=self.world_size,
            geometry=prep._block0_geometry(entry),
            layer_bytes=entry.data.shape[2]
            * torch.empty((), dtype=self._dtype).element_size()
            * 2
            * self._elements_per_row(),
        )

    def _elements_per_row(self):
        import math

        return math.prod(self._kv_shape)

    def _read_local(self, key):
        raw = self.client.get(self._manifest_key(key))
        if not raw:
            return None
        manifest = json.loads(raw)
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
            manifest.get("schema") != "zing-block0-mooncake-v2"
            or manifest.get("identity") != self.identity
            or manifest.get("sp_world_size") != self.world_size
            or type(rows) is not int
            or not 0 < rows <= self.max_rows
            or any(geometry.get(k) != v for k, v in expected.items())
            or manifest.get("layer_bytes")
            != rows
            * self._elements_per_row()
            * torch.empty((), dtype=self._dtype).element_size()
            * 2
        ):
            return None
        keys = self._layer_keys(key)
        if self.client.batch_is_exist([*keys, self._cond_key(key)]) != [1] * (
            self._layers + 1
        ):
            return None
        payload = self.client.get(self._cond_key(key))
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
        return prep.PreparationEntry(
            key,
            conditions["inputs"],
            conditions.get("conditioning_identity"),
            conditions.get("request"),
            positions,
            None,
            SharedKV((self._layers, 2, rows, *self._kv_shape), self._dtype, keys),
        )

    def lookup(self, key, *, allow_disk=False, protected=()):
        started = time.perf_counter()
        self.finish_fill()
        entry = error = None
        try:
            entry = self._read_local(key)
        except Exception as exc:
            error = type(exc).__name__
        rows = None if entry is None else int(entry.positions.shape[0])
        votes = prep.coordinate_validation((key, rows, error))
        hit = entry is not None and all(v == (key, rows, None) for v in votes)
        self.hits += int(hit)
        self.misses += int(not hit)
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
            tier="l1",
            backend=self.backend,
            allow_disk=False,
            reason=None if hit else error or "not_resident",
        )
        return entry if hit else None

    def restore(self, cache, entry):
        started = time.perf_counter()
        error = None
        views = []
        positions = blocks = None
        try:
            rows = entry.positions.shape[0]
            positions = entry.positions.to(self._device)
            blocks = torch.zeros(rows, dtype=torch.long, device=self._device)
            for layer, slot in enumerate(cache.storage):
                k, v = cache._reserve_store(layer, rows, slot.k, slot.v)
                views.append((k[:rows], v[:rows]))
            ptrs, sizes = self._buffers(views, rows)
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
        )

    def _buffers(self, layers, rows):
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
                if storage.data_ptr() not in self._registered:
                    raise ValueError("Block0 buffer is outside the registered GPU pool")
                if (
                    value.dtype != self._dtype
                    or tuple(value.shape[1:]) != self._kv_shape
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
        entry = ready = error = None
        try:
            prefix = cache.prefix
            rows = int(prefix.positions.shape[0])
            if not self.can_store(rows):
                raise ValueError("Block0 prefix exceeds the configured row limit")
            layers = [(pair[0][:rows], pair[1][:rows]) for pair in prefix.layers]
            ptrs, sizes = self._buffers(layers, rows)
            entry = prep.PreparationEntry(
                key,
                prep.inputs_on(inputs, "cpu", snapshot=True),
                conditioning_identity,
                request,
                prefix.positions.detach().cpu().clone(),
                None,
                SharedKV(
                    (self._layers, 2, rows, *self._kv_shape),
                    self._dtype,
                    self._layer_keys(key),
                ),
                force=force,
                fill_started=time.perf_counter(),
            )
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
            if self._device.type == "cuda":
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self._device))
        except Exception as exc:
            error = type(exc).__name__
        self._agree(error)

        def write():
            # Capture layers so their allocations stay alive through SDK completion.
            _ = layers
            device = (
                torch.cuda.device(self._device)
                if self._device.type == "cuda"
                else nullcontext()
            )
            with device:
                if ready is not None:
                    ready.synchronize()
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
                    raise CacheTransferError(f"Mooncake KV write failed: {codes}")
                if self.client.batch_is_exist(entry.data.keys) != [1] * self._layers:
                    raise CacheTransferError(
                        "Mooncake KV disappeared before publication"
                    )
                self._put(self._cond_key(key), payload.getvalue(), key)

        future = error = None
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
        return True

    def finish_fill(self):
        if self.pending is None:
            return None
        entry, future = self.pending
        indices, self._pending_indices = self._pending_indices, set()
        error = None
        try:
            future.result()
        except Exception as exc:
            error = type(exc).__name__
        finally:
            self.pending = None
        receipts = prep.coordinate_validation((error, self._manifest(entry)))
        if any(err is not None or meta != receipts[0][1] for err, meta in receipts):
            self.write_failures += 1
            cache_event(
                "l1_fill",
                "failed",
                cache_key=entry.key,
                rank=self.log_rank,
                reason="incomplete_shards",
                backend=self.backend,
                level=logging.WARNING,
            )
            return None
        error = None
        if self.log_rank == 0:
            try:
                # An early index can be evicted while KV is still processing.
                # Rebind successful early associations under the KV read lease
                # before publishing; immutable same-key puts are safe to reuse.
                for association in indices:
                    self._put(association, b"1", entry.key)
                self._put(
                    self._manifest_key(entry.key),
                    json.dumps(receipts[0][1], sort_keys=True).encode(),
                    entry.key,
                )
                if not self._check_group(entry.key, sorted(indices)):
                    raise CacheTransferError("Mooncake manifest disappeared")
            except Exception as exc:
                error = type(exc).__name__
        errors = prep.coordinate_validation(error)
        if any(value is not None for value in errors):
            self.write_failures += 1
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
        self._known.add(entry.key)
        cache_event(
            "l1_fill",
            "completed",
            cache_key=entry.key,
            rank=self.log_rank,
            started=entry.fill_started,
            backend=self.backend,
        )
        return entry

    def register(self, key, game_id=None, group_id=None):
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
                    entry.data.numel() * entry.data.element_size(),
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
        if self.client is not None:
            for ptr in self._registered:
                self.client.unregister_buffer(ptr)
            self._registered.clear()
            self.client.close()
