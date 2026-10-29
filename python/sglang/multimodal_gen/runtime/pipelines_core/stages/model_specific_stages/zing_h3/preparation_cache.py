# SPDX-License-Identifier: Apache-2.0
"""Shared startup/prompt-switch snapshots for single-session deployments."""

import hashlib
import json
import threading
import time
import uuid
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.conditioning.worker import (
    broadcast_receipt,
    coordinate_validation,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .media_input import _local_media

logger = init_logger(__name__)


def preparation_request(codecs, request):
    """Freeze accepted media contents once; prompt updates reuse these contents."""
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


def preparation_key(codecs, request, preview, config, policy):
    """One content key shared by startup and prompt switches within this stage."""
    identity = dict(
        request=preparation_request(codecs, request),
        source={name: config[name] for name in ("width", "height", "fps")},
        policy=policy,
        s0=None
        if preview is None
        else hashlib.sha256(
            preview["frames"].contiguous().numpy().tobytes()
        ).hexdigest(),
    )
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
    generation: str


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
    A hit missing from L1 is promoted from the Tier-2 disk cache, and every
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
        return key in self.entries or (
            self.disk_cache is not None and self.disk_cache.may_contain(key)
        )

    def lookup(self, key):
        self.finish_fill()
        local = self.entries.get(key)
        if local is None:
            local = self._promote_from_disk(key)
        votes = coordinate_validation((key, local is not None))
        hit = all(digest == key and ready for digest, ready in votes)
        self.hits += int(hit)
        self.misses += int(not hit)
        if hit:
            self.entries.move_to_end(key)
        return local if hit else None

    def _take_slot(self, key):
        if key in self.entries:
            # A partial SP miss recomputes on every rank. Replace the matching
            # local entry too, so all shards describe that same preparation.
            slot = self.entries.pop(key).slot
        elif self.free_slots:
            slot = self.free_slots.pop()
        else:
            _, old = self.entries.popitem(last=False)
            slot = old.slot
            self.evictions += 1
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
                return None
            if (
                positions.ndim != 2
                or positions.shape[1] != 3
                or not torch.isfinite(positions).all()
            ):
                return None
            rows = int(positions.shape[0])
            if rows < 1 or rows > self.data.shape[3]:
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
                cached.manifest["generation"],
            )
            self.entries[key] = entry
            return entry
        except Exception as exc:
            logger.warning("Zing Block0 disk promotion failed: %s", exc)
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

    def fill(self, key, inputs, conditioning_identity, request, cache):
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
            broadcast_receipt(uuid.uuid4().hex),
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

    def finish_fill(self):
        if self.pending is not None:
            entry, done = self.pending
            done.synchronize()
            self.entries[entry.key] = entry
            self.pending = None
            self._persist(entry)

    def _persist(self, entry):
        disk = self.disk_cache
        if disk is None or not disk.is_enabled():
            return
        done = threading.Event()
        self._disk_tasks[entry.slot] = done

        def write():
            try:
                self._write_disk(entry)
            except Exception:
                logger.warning("Zing Block0 disk write failed", exc_info=True)
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
            "generation": entry.generation,
            "sp_world_size": disk.sp_world_size,
            "geometry": _block0_geometry(entry),
            "prompt_preview": _prompt_preview(entry.request),
        }
        path = disk.save_rank_kv(entry.key, rank, entry.data, tmp)
        if path is None:
            disk.discard_tmp(tmp)
            return
        disk.commit_entry(entry.key, conditions, manifest, {rank: path}, tmp)

    def wait_for_disk(self):
        """Wait for in-flight L2 writes; used by shutdown and deterministic tests."""
        while self._disk_tasks:
            _, done = self._disk_tasks.popitem()
            done.wait()
