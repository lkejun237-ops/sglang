# SPDX-License-Identifier: Apache-2.0
"""Shared startup/prompt-switch snapshots for single-session deployments."""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.conditioning.worker import (
    coordinate_validation,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .block0_identity import fingerprint
from .cache_logging import cache_event, cache_scope, log_context
from .media_input import _local_media

logger = init_logger(__name__)


def preparation_request(codecs, request):
    """Freeze accepted media contents once; prompt updates reuse these contents."""
    request = {
        k: v
        for k, v in request.items()
        if k
        not in {
            "game_id",
            "group_id",
            "event_id",
            "request_id",
            "session_id",
            "force",
            "cache_write",
        }
    }
    if "messages" in request:
        prompt, media, target, conditions, source = codecs._parse_inference_request(
            request
        )
        request = {
            **{k: v for k, v in request.items() if k != "messages"},
            "prompt": prompt,
            "media": media,
            "target": target,
            "conditions": conditions,
            "source": source,
        }

    def content(value):
        if isinstance(value, bytes):
            return {"sha256": hashlib.sha256(value).hexdigest()}
        if isinstance(value, list):
            return [content(item) for item in value]
        if isinstance(value, dict):
            value = dict(value)
            if value.get("type") == "image" and "uri" in value:
                with _local_media(str(value.pop("uri")), fs=codecs.remote_read_fs) as (
                    path,
                    fs,
                ):
                    codecs.remote_read_fs = fs
                    value["payload"] = Path(path).read_bytes()
            return {name: content(item) for name, item in value.items()}
        return value

    return content(request)


def preparation_key(codecs, request, preview, config, identity):
    """One content key shared by startup and prompt switches within this stage."""
    identity = dict(
        request=preparation_request(codecs, request),
        source={name: config[name] for name in ("width", "height")},
        **identity,
        s0=None
        if preview is None
        else hashlib.sha256(
            preview["frames"].contiguous().numpy().tobytes()
        ).hexdigest(),
    )
    # Reference resizing changes the effective model inputs even when the
    # encoded images and exported checkpoint stay the same.
    reference_resize = getattr(codecs, "reference_resize", None)
    if reference_resize is not None:
        identity["reference_resize"] = asdict(reference_resize)
    return fingerprint(identity)


def inputs_on(inputs, device, *, snapshot=False):
    def tensors(values):
        return {
            name: value.detach().to(device, copy=snapshot)
            for name, value in values.items()
        }

    # condition_media stays on CPU for later prompt/reference updates.
    media = inputs.condition_media
    if snapshot and media is not None:
        media = [
            {
                name: value.detach().to("cpu", copy=True)
                if isinstance(value, torch.Tensor)
                else deepcopy(value)
                for name, value in item.items()
            }
            for item in media
        ]
    return replace(
        inputs,
        positive=tensors(inputs.positive),
        visual=tensors(inputs.visual),
        audio=tensors(inputs.audio),
        conditions=deepcopy(inputs.conditions),
        media=deepcopy(inputs.media),
        condition_media=media,
        max_area_config=deepcopy(inputs.max_area_config),
    )


@dataclass
class PreparationEntry:
    key: str
    inputs: object
    conditioning_identity: object
    request: object
    positions: torch.Tensor
    model_signature: object
    data: torch.Tensor
    slot: int
    force: bool = False
    log_context: dict | None = None
    fill_started: float | None = None


@dataclass
class PrefixRestore:
    entry: PreparationEntry
    begin: object
    end: object
    positions: torch.Tensor
    blocks: torch.Tensor
    views: list
    submit_ms: float
    started: float
    ready_at_consumer: bool | None = None
    lead_ms: float | None = None

    def commit(self, cache):
        self.ready_at_consumer = self.end.query()
        self.lead_ms = (time.perf_counter() - self.started) * 1000
        torch.cuda.current_stream(self.positions.device).wait_event(self.end)
        # L2 entries have no process-local parameter identities. The first
        # consuming model forward binds its current signature (after lazy buffer
        # initialization); subsequent forwards retain the usual mutation guard.
        if self.entry.model_signature is not None:
            cache.bind_model(self.entry.model_signature)
        cache.update(
            self.positions,
            self.blocks,
            "prefix",
            0,
            layers=self.views,
            validation=(
                self.entry.positions,
                torch.zeros_like(self.entry.positions[:, 0], dtype=torch.long),
            ),
        )


def _prompt_preview(request):
    value = (
        request.get("prompt")
        if isinstance(request, dict)
        else getattr(request, "prompt", None)
    )
    return None if value is None else str(value)[:160]


def _block0_geometry(entry):
    data = entry.data
    visual = getattr(entry.inputs, "visual_shape", None)
    return {
        "layers": int(data.shape[0]),
        "parts": int(data.shape[1]),
        "prefix_rows": int(data.shape[2]),
        "kv_shape": list(data.shape[3:]),
        "dtype": str(data.dtype).replace("torch.", ""),
        "visual_shape": list(visual) if visual is not None else None,
    }


class Block0PreparationCache:
    """A bounded pinned LRU; the existing GPU pool remains the only KV owner.

    One session writes this cache. Transfers share one stream, so a reused CPU
    slot is overwritten only after earlier restores from that slot have finished.
    Only explicit preload/prefill may promote from Tier-2 disk. Every
    published L1 entry is written back to disk on a daemon thread.
    """

    def __init__(self, pool, *, max_bytes, identity, disk_cache):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Block0 cache byte budget must be a positive integer")
        self.mode = "on"
        self.identity = identity
        self.entries = OrderedDict()
        self.pending = None
        self.hits = self.misses = self.evictions = 0
        self.disk_cache = disk_cache
        self._disk_tasks = {}
        # Reserve once, before readiness, instead of pinning gigabytes on a request.
        stores = pool._pool._slots[0].kv_cache
        first = stores[0].k
        slot_bytes = (
            len(stores)
            * 2
            * pool.plan.config.max_prefix_rows
            * first[0].numel()
            * first.element_size()
        )
        entries = max_bytes // slot_bytes
        if entries < 1:
            raise ValueError(
                "Block0 host budget cannot hold one maximum-prefix KV slot"
            )
        self.max_bytes = max_bytes
        self.data = torch.empty(
            (
                entries,
                len(stores),
                2,
                pool.plan.config.max_prefix_rows,
                *first.shape[1:],
            ),
            dtype=first.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.stream = torch.cuda.Stream(device=first.device)
        self.free_slots = list(reversed(range(entries)))

    def may_contain(self, key):
        """CPU-only hint for speculative work; lookup remains authoritative.

        Never finish a pending CUDA fill or run a collective on the prefetch
        thread. Eviction and partial-rank hits simply fall back at the boundary.
        """
        return key in self.entries

    @property
    def log_rank(self):
        return (
            getattr(self.disk_cache, "local_rank", None)
            if self.disk_cache is not None
            else (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
        )

    def lookup(self, key, *, allow_disk=False):
        with cache_scope():
            return self._lookup(key, allow_disk=allow_disk)

    def _lookup(self, key, *, allow_disk):
        started = time.perf_counter()
        self.finish_fill()
        local = self.entries.get(key)
        tier = "l1"
        # Agree before IO: a partial L1 hit must take the same SP path.
        if allow_disk:
            present = coordinate_validation(local is not None)
            if not all(present):
                local = self._promote_from_disk(key)
                tier = "l2"
        votes = coordinate_validation((key, local is not None))
        hit = local is not None and all(
            digest == key and ready for digest, ready in votes
        )
        self.hits += int(hit)
        self.misses += int(not hit)
        if hit:
            self.entries.move_to_end(key)
        cache_event(
            "lookup",
            "hit" if hit else "miss",
            cache_key=key,
            rank=self.log_rank,
            started=started,
            tier=tier,
            allow_disk=allow_disk,
            reason=None
            if hit
            else "sp_key_mismatch"
            if any(digest != key for digest, _ in votes)
            else "partial_sp_hit"
            if any(ready for _, ready in votes)
            else "not_resident",
            next_action=None
            if hit
            else "recompute"
            if not allow_disk
            else "caller_decides",
        )
        return local if hit else None

    def _take_slot(self, key):
        if key in self.entries:
            # A partial SP miss recomputes on every rank. Replace the matching
            # local entry too, so all shards describe that same preparation.
            slot = self.entries.pop(key).slot
        elif self.free_slots:
            slot = self.free_slots.pop()
        else:
            old_key, old = self.entries.popitem(last=False)
            slot = old.slot
            self.evictions += 1
            cache_event(
                "l1_evict",
                "completed",
                cache_key=old_key,
                rank=self.log_rank,
                reason="capacity",
                replacement_key=key,
            )
        task = self._disk_tasks.pop(slot, None)
        if task is not None:
            # The disk writer reads this slot directly; never overwrite it.
            task.wait()
        return slot

    def _promote_from_disk(self, key):
        disk = self.disk_cache
        if disk is None or not disk.is_enabled():
            return None
        try:
            cached = disk.lookup(key, disk.local_rank)
            if cached is None:
                return None
            conditions = cached.conditions
            inputs = conditions.get("inputs")
            positions = conditions.get("positions")
            if inputs is None or not isinstance(positions, torch.Tensor):
                cache_event(
                    "l2_read",
                    "failed",
                    cache_key=key,
                    rank=self.log_rank,
                    reason="invalid_conditions",
                    level=logging.WARNING,
                )
                return None
            if (
                positions.ndim != 2
                or positions.shape[1] != 3
                or not torch.isfinite(positions).all()
            ):
                cache_event(
                    "l2_read",
                    "failed",
                    cache_key=key,
                    rank=self.log_rank,
                    reason="invalid_positions",
                    level=logging.WARNING,
                )
                return None
            rows = int(positions.shape[0])
            if rows < 1 or rows > self.data.shape[3]:
                cache_event(
                    "l2_read",
                    "failed",
                    cache_key=key,
                    rank=self.log_rank,
                    reason="invalid_prefix_rows",
                    level=logging.WARNING,
                )
                return None
            slot = self._take_slot(key)
            try:
                # A disk read runs on the CPU, outside the DMA stream. Fence
                # earlier H2D reads before overwriting the reused pinned slot.
                self.stream.synchronize()
                data = self.data[slot, :, :, :rows]
                disk.load_kv_into(cached, data)
            except BaseException:
                self.free_slots.append(slot)
                raise
            entry = PreparationEntry(
                key,
                inputs,
                conditions.get("conditioning_identity"),
                conditions.get("request"),
                positions,
                # Old disk files may contain another process/rank's identities.
                # Never import that runtime-only guard into this cache.
                None,
                data,
                slot,
            )
            self.entries[key] = entry
            return entry
        except Exception as exc:
            cache_event(
                "l2_read",
                "failed",
                cache_key=key,
                rank=self.log_rank,
                reason=type(exc).__name__,
                level=logging.WARNING,
            )
            return None

    def restore(self, cache, entry):
        started = time.perf_counter()
        rows = entry.positions.shape[0]
        device = cache.storage[0].k.device
        # Small metadata is ready before enqueueing the large asynchronous copy.
        positions = entry.positions.to(device)
        blocks = torch.zeros(rows, dtype=torch.long, device=device)
        views = []
        for layer, slot in enumerate(cache.storage):
            k, v = cache._reserve_store(layer, rows, slot.k, slot.v)
            views.append((k[:rows], v[:rows]))
        self.stream.wait_stream(torch.cuda.current_stream(device))
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        with torch.cuda.stream(self.stream):
            begin.record()
            for layer, (k, v) in enumerate(views):
                k.copy_(entry.data[layer, 0], non_blocking=True)
                v.copy_(entry.data[layer, 1], non_blocking=True)
            end.record()
        return PrefixRestore(
            entry,
            begin,
            end,
            positions,
            blocks,
            views,
            (time.perf_counter() - started) * 1000,
            started,
        )

    def fill(self, key, inputs, conditioning_identity, request, cache, *, force=False):
        started = time.perf_counter()
        self.finish_fill()
        prefix = cache.prefix
        rows = prefix.positions.shape[0]
        if rows > self.data.shape[3]:
            raise ValueError("Block0 prefix exceeds the preallocated CPU slot")
        snapshot = inputs_on(inputs, "cpu", snapshot=True)
        positions = prefix.positions.detach().cpu().clone()
        slot = self._take_slot(key)
        data = self.data[slot, :, :, :rows]
        entry = PreparationEntry(
            key,
            snapshot,
            conditioning_identity,
            request,
            positions,
            cache._model_signature,
            data,
            slot,
            force,
            log_context(),
            started,
        )
        # Positions may stay on the host; the transfer stream owns the KV device.
        self.stream.wait_stream(torch.cuda.current_stream(self.stream.device))
        done = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            for layer, (k, v) in enumerate(prefix.layers):
                data[layer, 0].copy_(k, non_blocking=True)
                data[layer, 1].copy_(v, non_blocking=True)
            done.record()
        self.pending = (entry, done)
        cache_event(
            "l1_fill",
            "submitted",
            cache_key=key,
            rank=self.log_rank,
            context=entry.log_context,
            force=force,
        )

    def finish_fill(self):
        if self.pending is not None:
            entry, done = self.pending
            done.synchronize()
            self.entries[entry.key] = entry
            self.pending = None
            cache_event(
                "l1_fill",
                "completed",
                cache_key=entry.key,
                rank=self.log_rank,
                context=entry.log_context,
                started=entry.fill_started,
                entries=len(self.entries),
            )
            self._persist(entry)

    def _persist(self, entry):
        disk = self.disk_cache
        if disk is None or not disk.is_enabled():
            return
        done = threading.Event()
        self._disk_tasks[entry.slot] = done

        def write():
            # ContextVars do not cross threads; keep only the metadata snapshot.
            with cache_scope(entry.log_context):
                started = time.perf_counter()
                try:
                    cache_event(
                        "l2_write", "started", cache_key=entry.key, rank=self.log_rank
                    )
                    committed = self._write_disk(entry)
                    cache_event(
                        "l2_write",
                        "completed" if committed else "failed",
                        cache_key=entry.key,
                        rank=self.log_rank,
                        started=started,
                        reason=None if committed else "storage_write_skipped",
                        level=logging.INFO if committed else logging.WARNING,
                    )
                except Exception as exc:
                    cache_event(
                        "l2_write",
                        "failed",
                        cache_key=entry.key,
                        rank=self.log_rank,
                        started=started,
                        reason=type(exc).__name__,
                        level=logging.WARNING,
                    )
                finally:
                    done.set()

        threading.Thread(
            target=write,
            name=f"zing-block0-disk-{entry.slot}",
            daemon=True,
        ).start()

    def _write_disk(self, entry):
        disk = self.disk_cache
        rank = disk.local_rank
        tmp = disk.new_tmp_dir(entry.key)
        if tmp is None:
            return
        conditions = {
            "inputs": entry.inputs,
            "conditioning_identity": entry.conditioning_identity,
            "request": entry.request,
            "positions": entry.positions,
        }
        manifest = {
            "identity": self.identity,
            "sp_world_size": disk.sp_world_size,
            "geometry": _block0_geometry(entry),
            "prompt_preview": _prompt_preview(entry.request),
        }
        path = disk.save_rank_kv(entry.key, rank, entry.data, tmp)
        if path is None:
            disk.discard_tmp(tmp)
            return
        return disk.commit_entry(
            entry.key, conditions, manifest, {rank: path}, tmp, force=entry.force
        )

    def wait_for_disk(self):
        """Wait for in-flight L2 writes; used by shutdown and deterministic tests."""
        while self._disk_tasks:
            _, done = self._disk_tasks.popitem()
            done.wait()

    def register(self, key, game_id=None, group_id=None):
        if (
            game_id is None
            or self.disk_cache is None
            or self.disk_cache.local_rank != 0
        ):
            return
        from .cache_index import CacheIndex

        try:
            CacheIndex(self.disk_cache.cache_dir).bind(game_id, group_id, key)
            cache_event(
                "index_bind",
                "completed",
                cache_key=key,
                rank=self.log_rank,
                game_id=game_id,
                group_id=group_id,
            )
        except OSError as exc:
            cache_event(
                "index_bind",
                "failed",
                cache_key=key,
                rank=self.log_rank,
                game_id=game_id,
                group_id=group_id,
                reason=type(exc).__name__,
                level=logging.WARNING,
            )

    def control(self, request, *, busy):
        """Select once on SP rank zero, then execute the same operation everywhere."""
        with cache_scope(
            getattr(request, "log_context", None),
            game_id=request.game_id,
            group_id=request.group_id,
        ):
            started = time.perf_counter()
            level = logging.DEBUG if request.operation == "status" else logging.INFO
            cache_event(
                request.operation,
                "started",
                cache_key=request.cache_key,
                rank=self.log_rank,
                level=level,
            )
            try:
                value = self._control(request, busy=busy)
            except Exception as exc:
                cache_event(
                    request.operation,
                    "failed",
                    cache_key=request.cache_key,
                    rank=self.log_rank,
                    started=started,
                    reason=type(exc).__name__,
                    level=logging.WARNING,
                )
                raise
            missing = sum(
                item["status"] == "missing" for item in value.get("results", [])
            )
            code = value.get("code")
            partial = (
                request.operation == "preload"
                and not code
                and not value.get("all_ready")
            )
            cache_event(
                request.operation,
                "rejected" if code else "partial" if partial else "completed",
                cache_key=request.cache_key,
                rank=self.log_rank,
                started=started,
                reason=code or ("not_all_ready" if partial else None),
                level=logging.WARNING if code or partial else level,
                selected_count=len(value.get("results", value.get("selected", []))),
                missing_count=missing,
            )
            return value

    def _control(self, request, *, busy):
        from sglang.multimodal_gen.runtime.conditioning.worker import broadcast_receipt

        from .cache_index import CacheIndex

        busy = any(coordinate_validation(busy))
        if request.operation != "status" and busy:
            return {"code": "cache_busy"}
        selection = None
        if self.disk_cache.local_rank == 0:
            try:
                if request.game_id is not None:
                    keys = CacheIndex(self.disk_cache.cache_dir).select(
                        request.game_id, request.group_id, request.cache_key
                    )
                elif request.operation == "status" and request.group_id is None:
                    keys = None if request.cache_key is None else [request.cache_key]
                else:
                    raise ValueError("A cache selection requires a game ID")
                selection = {"keys": keys}
            except ValueError:
                selection = {"code": "invalid_cache_selection"}
            except OSError:
                logger.warning("Block0 business index is unavailable", exc_info=True)
                selection = {"code": "cache_unavailable"}
        selection = broadcast_receipt(selection)
        if "code" in selection:
            return selection
        return dict(self.manage(request.operation, selection["keys"]), busy=busy)

    def status(self, keys=None):
        local = list(self.entries)
        ranks = coordinate_validation(local)
        resident = sorted(set.intersection(*(set(items) for items in ranks)))
        selected = resident if keys is None else keys
        states = [{"cache_key": key, "ready": key in resident} for key in selected]
        return dict(
            capacity=int(self.data.shape[0]),
            entries=len(resident),
            cache_keys=resident,
            selected=states,
            all_ready=bool(states) and all(item["ready"] for item in states),
        )

    def manage(self, operation, keys):
        if operation == "status":
            return self.status(keys)
        self.finish_fill()
        if operation == "preload" and len(keys) > self.data.shape[0]:
            return {
                "code": "cache_capacity_exceeded",
                "capacity": int(self.data.shape[0]),
                "requested": len(keys),
            }
        results = []
        for key in keys:
            if operation == "preload":
                entry = self.lookup(key, allow_disk=True)
                results.append(
                    {
                        "cache_key": key,
                        "status": "ready" if entry is not None else "missing",
                    }
                )
            elif operation == "unload":
                entry = self.entries.pop(key, None)
                if entry is not None:
                    task = self._disk_tasks.pop(entry.slot, None)
                    if task is not None:
                        task.wait()
                    self.stream.synchronize()
                    self.free_slots.append(entry.slot)
                found = coordinate_validation(entry is not None)
                cache_event(
                    "l1_unload",
                    "completed" if any(found) else "missing",
                    cache_key=key,
                    rank=self.log_rank,
                )
                results.append(
                    {
                        "cache_key": key,
                        "status": "unloaded" if any(found) else "missing",
                    }
                )
            else:
                raise ValueError("Unknown Block0 cache operation")
        return dict(self.status(keys), results=results)
