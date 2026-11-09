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

from sglang.multimodal_gen.runtime.conditioning.worker import coordinate_validation
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .block0_identity import fingerprint
from .cache_logging import cache_event, cache_scope, log_context
from .media_input import _local_media
from .paged_kv import PAGE_ROWS, PagedKV

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
    data: object
    force: bool = False
    log_context: dict | None = None
    fill_started: float | None = None
    disk_task: threading.Event | None = None

    @property
    def page_ids(self):
        return self.data.page_ids


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

    def wait(self):
        if self.end is not None:
            self.end.synchronize()

    @property
    def cuda_ms(self):
        return None if self.end is None else self.begin.elapsed_time(self.end)

    def commit(self, cache):
        self.ready_at_consumer = self.end is None or self.end.query()
        self.lead_ms = (time.perf_counter() - self.started) * 1000
        if self.end is not None:
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
    """Block0 provider interface and shared SP management request adapter.

    Constructing the interface selects an implementation while retaining the
    existing caller API. Only the local implementation owns pages and disk IO.
    """

    def __new__(cls, *args, backend="local", **kwargs):
        if cls is Block0PreparationCache:
            if backend == "local":
                impl = LocalPagedBackend
            elif backend == "mooncake":
                from .mooncake_cache import MooncakeBackend

                impl = MooncakeBackend
            else:
                raise ValueError("Unknown Block0 cache backend")
            return object.__new__(impl)
        return object.__new__(cls)

    def lookup(self, key, *, allow_disk=False, protected=()):
        raise NotImplementedError

    def restore(self, cache, entry):
        raise NotImplementedError

    def can_store(self, rows):
        raise NotImplementedError

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
        raise NotImplementedError

    def finish_fill(self):
        raise NotImplementedError

    def may_contain(self, key):
        raise NotImplementedError

    def register(self, key, game_id=None, group_id=None):
        raise NotImplementedError

    def select_keys(self, game_id, group_id=None, cache_key=None):
        raise NotImplementedError

    def status(self, keys=None):
        raise NotImplementedError

    def manage(self, operation, keys):
        raise NotImplementedError

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

        busy = any(coordinate_validation(busy))
        if request.operation != "status" and busy:
            return {"code": "cache_busy"}
        selection = None
        if self.log_rank == 0:
            try:
                if request.game_id is not None:
                    keys = self.select_keys(
                        request.game_id, request.group_id, request.cache_key
                    )
                elif request.operation == "status" and request.group_id is None:
                    keys = None if request.cache_key is None else [request.cache_key]
                else:
                    raise ValueError("A cache selection requires a game ID")
                selection = {"keys": keys}
            except ValueError:
                selection = {"code": "invalid_cache_selection"}
            except Exception:
                logger.warning("Block0 business index is unavailable", exc_info=True)
                selection = {"code": "cache_unavailable"}
        selection = broadcast_receipt(selection)
        if "code" in selection:
            return selection
        return dict(self.manage(request.operation, selection["keys"]), busy=busy)


class LocalPagedBackend(Block0PreparationCache):
    """A bounded pinned LRU; the existing GPU pool remains the only KV owner.

    One session writes this cache. Transfers share one stream, so a reused CPU
    page is recycled only after transfers and its entry's disk writer finish.
    Only explicit preload/prefill may promote from Tier-2 disk. Every
    published L1 entry is written back to disk on a daemon thread.
    """

    def __init__(
        self,
        pool,
        *,
        max_bytes,
        identity,
        disk_cache,
        page_rows=PAGE_ROWS,
        backend="local",
    ):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Block0 cache byte budget must be a positive integer")
        self.backend = "local"
        self.mode = "on"
        self.identity = identity
        self.entries = OrderedDict()
        self.pending = None
        self.hits = self.misses = self.evictions = 0
        self.disk_cache = disk_cache
        if type(page_rows) is not int or page_rows < 1:
            raise ValueError("Block0 page rows must be a positive integer")
        # Pin once before readiness. Rows are a testable storage layout, not a
        # request-time allocation or a new deployment configuration option.
        stores = pool._pool._slots[0].kv_cache
        first = stores[0].k
        self.max_rows = pool.plan.slot_rows
        self.page_rows = page_rows
        self.page_bytes = (
            len(stores) * 2 * page_rows * first[0].numel() * first.element_size()
        )
        self.total_pages = max_bytes // self.page_bytes
        if self.total_pages < 1:
            raise ValueError("Block0 host budget cannot hold one pinned KV page")
        self.max_bytes = max_bytes
        self.pages = torch.empty(
            (len(stores), 2, self.total_pages, page_rows, *first.shape[1:]),
            dtype=first.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.stream = torch.cuda.Stream(device=first.device)
        self.free_pages = [True] * self.total_pages

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

    def lookup(self, key, *, allow_disk=False, protected=()):
        with cache_scope():
            return self._lookup(key, allow_disk=allow_disk, protected=protected)

    def _lookup(self, key, *, allow_disk, protected):
        started = time.perf_counter()
        self.finish_fill()
        local = self.entries.get(key)
        tier = "l1"
        # Partial L1 hits and unequal row counts must take the same SP disk path.
        if allow_disk:
            present = coordinate_validation(
                None if local is None else int(local.positions.shape[0])
            )
            if present[0] is None or any(rows != present[0] for rows in present):
                local = self._promote_from_disk(key, protected=protected)
                tier = "l2"
        local_rows = None if local is None else int(local.positions.shape[0])
        votes = coordinate_validation((key, local_rows))
        hit = local is not None and all(
            digest == key and rows == local_rows for digest, rows in votes
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

    def _release(self, key):
        entry = self.entries[key]
        if entry.disk_task is not None:
            entry.disk_task.wait()
        # Keep ownership until both CPU readers and DMA have retired. This also
        # makes free pages safe for synchronous disk IO, not just queued DMA.
        self.stream.synchronize()
        self._return_pages(entry.page_ids)
        del self.entries[key]

    def _return_pages(self, page_ids):
        if len(set(page_ids)) != len(page_ids) or any(
            self.free_pages[page] for page in page_ids
        ):
            raise ValueError("Block0 page was returned more than once")
        for page in page_ids:
            self.free_pages[page] = True

    def _take_pages(self, key, rows, *, protected=()):
        if type(rows) is not int or not 0 < rows <= self.max_rows:
            raise ValueError("Block0 prefix exceeds the configured row limit")
        needed = (rows + self.page_rows - 1) // self.page_rows
        if needed > self.total_pages:
            raise ValueError("Block0 prefix exceeds the pinned page budget")
        if key in self.entries:
            self._release(key)
        while self.free_pages.count(True) < needed:
            old_key = next((k for k in self.entries if k not in protected), None)
            if old_key is None:
                raise ValueError(
                    "Block0 preload selection exceeds the pinned page budget"
                )
            self._release(old_key)
            self.evictions += 1
            cache_event(
                "l1_evict",
                "completed",
                cache_key=old_key,
                rank=self.log_rank,
                reason="capacity",
                replacement_key=key,
            )
        ids = tuple(page for page, free in enumerate(self.free_pages) if free)[:needed]
        for page in ids:
            self.free_pages[page] = False
        return PagedKV(self.pages, ids, rows)

    def _promote_from_disk(self, key, *, protected=()):
        disk = self.disk_cache
        try:
            # A failed disk lookup must not leave an old shard looking ready
            # after another rank reloads to the same row count.
            if key in self.entries:
                self._release(key)
            if disk is None or not disk.is_enabled():
                return None
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
            if rows < 1 or rows > self.max_rows:
                cache_event(
                    "l2_read",
                    "failed",
                    cache_key=key,
                    rank=self.log_rank,
                    reason="invalid_prefix_rows",
                    level=logging.WARNING,
                )
                return None
            data = self._take_pages(key, rows, protected=protected)
            try:
                # A disk read runs on the CPU, outside the DMA stream. Fence
                # earlier H2D reads before overwriting the reused pinned slot.
                self.stream.synchronize()
                disk.load_kv_into(cached, data)
            except BaseException:
                self._return_pages(data.page_ids)
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
            try:
                for layer, part, start, value in entry.data.slices():
                    views[layer][part][start : start + value.shape[0]].copy_(
                        value, non_blocking=True
                    )
                end.record()
            except BaseException:
                self.stream.synchronize()
                raise
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

    def can_store(self, rows):
        return 0 < rows <= min(self.max_rows, self.total_pages * self.page_rows)

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
        started = time.perf_counter()
        self.finish_fill()
        prefix = cache.prefix
        rows = prefix.positions.shape[0]
        if rows > self.max_rows:
            raise ValueError("Block0 prefix exceeds the configured row limit")
        if not required and not self.can_store(rows):
            cache_event(
                "l1_fill",
                "skipped",
                cache_key=key,
                rank=self.log_rank,
                reason="host_budget",
            )
            return False
        snapshot = inputs_on(inputs, "cpu", snapshot=True)
        positions = prefix.positions.detach().cpu().clone()
        data = self._take_pages(key, rows)
        entry = PreparationEntry(
            key,
            snapshot,
            conditioning_identity,
            request,
            positions,
            cache._model_signature,
            data,
            force,
            log_context(),
            started,
        )
        # Own the pages even if event construction or a later copy submission
        # fails. A failed drain must never return still-active pages to the pool.
        self.pending = (entry, None)
        try:
            self.stream.wait_stream(torch.cuda.current_stream(self.stream.device))
            done = torch.cuda.Event()
            with torch.cuda.stream(self.stream):
                for layer, part, start, value in data.slices():
                    value.copy_(
                        prefix.layers[layer][part][start : start + value.shape[0]],
                        non_blocking=True,
                    )
                done.record()
            self.pending = (entry, done)
        except BaseException:
            self.finish_fill()
            raise
        cache_event(
            "l1_fill",
            "submitted",
            cache_key=key,
            rank=self.log_rank,
            context=entry.log_context,
            force=force,
        )

        return True

    def finish_fill(self):
        if self.pending is not None:
            entry, done = self.pending
            if done is None:
                self.stream.synchronize()
                self._return_pages(entry.page_ids)
                self.pending = None
                return
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
            return entry

    def _persist(self, entry):
        disk = self.disk_cache
        if disk is None or not disk.is_enabled():
            return
        done = threading.Event()
        entry.disk_task = done

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

        try:
            threading.Thread(
                target=write,
                name=f"zing-block0-disk-{entry.key[:12]}",
                daemon=True,
            ).start()
        except Exception:
            done.set()
            logger.warning("Block0 disk writer could not start", exc_info=True)

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
        for entry in list(self.entries.values()):
            if entry.disk_task is not None:
                entry.disk_task.wait()

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

    def status(self, keys=None):
        snapshots = coordinate_validation(
            (
                {
                    key: int(entry.positions.shape[0])
                    for key, entry in self.entries.items()
                },
                self.free_pages.count(True),
                self.total_pages,
                self.page_bytes,
                self.max_bytes,
            )
        )
        resident = sorted(
            key
            for key, rows in snapshots[0][0].items()
            if all(item[0].get(key) == rows for item in snapshots)
        )
        selected = resident if keys is None else keys
        states = [{"cache_key": key, "ready": key in resident} for key in selected]
        return dict(
            page_rows=self.page_rows,
            total_pages=min(item[2] for item in snapshots),
            free_pages=min(item[1] for item in snapshots),
            budget_bytes=sum(item[4] for item in snapshots),
            reserved_bytes=sum(item[2] * item[3] for item in snapshots),
            used_bytes=sum((item[2] - item[1]) * item[3] for item in snapshots),
            entries=len(resident),
            cache_keys=resident,
            selected=states,
            all_ready=bool(states) and all(item["ready"] for item in states),
        )

    def _preload_plan(self, keys):
        # Agree on residency first: partial L1 hits must all use the disk path.
        resident = coordinate_validation(
            [
                int(self.entries[k].positions.shape[0]) if k in self.entries else None
                for k in keys
            ]
        )
        local_rows = []
        for index, key in enumerate(keys):
            values = [rank[index] for rank in resident]
            if values[0] is not None and all(value == values[0] for value in values):
                rows = values[0]
            else:
                rows = (
                    None
                    if self.disk_cache is None
                    else self.disk_cache.prefix_rows(key)
                )
            local_rows.append(
                rows if type(rows) is int and 0 < rows <= self.max_rows else None
            )
        plans = coordinate_validation((self.total_pages, local_rows))
        # No rank returns before this collective. Metadata races are handled by
        # the actual load and final all-rank residency check, as before.
        rows = []
        for index in range(len(keys)):
            values = [rank[1][index] for rank in plans]
            rows.append(
                values[0]
                if values[0] is not None and all(v == values[0] for v in values)
                else 0
            )
        needed = sum((n + self.page_rows - 1) // self.page_rows for n in rows)
        total = min(rank[0] for rank in plans)
        return rows, needed, total

    def manage(self, operation, keys):
        if operation == "status":
            return self.status(keys)
        self.finish_fill()
        keys = list(dict.fromkeys(keys))
        protected = set()
        if operation == "preload":
            rows, needed, total = self._preload_plan(keys)
            if needed > total:
                return {
                    "code": "cache_capacity_exceeded",
                    "total_pages": total,
                    "requested_pages": needed,
                    "page_rows": self.page_rows,
                }
            protected = {key for key, n in zip(keys, rows) if n}
        results = []
        for key in keys:
            if operation == "preload":
                entry = (
                    self.lookup(key, allow_disk=True, protected=protected)
                    if key in protected
                    else None
                )
                results.append(
                    {
                        "cache_key": key,
                        "status": "ready" if entry is not None else "missing",
                    }
                )
            elif operation == "unload":
                found = key in self.entries
                if found:
                    self._release(key)
                votes = coordinate_validation(found)
                cache_event(
                    "l1_unload",
                    "completed" if any(votes) else "missing",
                    cache_key=key,
                    rank=self.log_rank,
                )
                results.append(
                    {
                        "cache_key": key,
                        "status": "unloaded" if any(votes) else "missing",
                    }
                )
            else:
                raise ValueError("Unknown Block0 cache operation")
        return dict(self.status(keys), results=results)

    @property
    def cpu_capacity_bytes(self):
        return self.pages.numel() * self.pages.element_size()

    @property
    def entry_count(self):
        return len(self.entries)

    def select_keys(self, game_id, group_id=None, cache_key=None):
        from .cache_index import CacheIndex

        return CacheIndex(self.disk_cache.cache_dir).select(
            game_id, group_id, cache_key
        )


def fill_prepared_prefix(
    stage,
    sampler,
    inputs,
    *,
    owner,
    key,
    identity,
    request,
    source_shape=None,
    force=False,
):
    """Fill the original Block0 cache from already prepared Encoder inputs.

    Shared by integrated prefill and the disaggregated model owner. No Qwen,
    source VAE, target sampling or decoder execution takes place here.
    """
    cache = stage.block0_cache
    with stage._use("transformer", "denoise") as transformer:
        sampler.generator = transformer
        device = next(transformer.parameters()).device
        visual = (
            inputs.positive["context"]
            .new_zeros((1, stage.packer.visual_channels, *inputs.visual_shape))
            .to(device)
        )
        layout = sampler.processor.pack_inference_layout(
            inputs, inputs.positive, visual, None
        )
        source_shape = (
            inputs.visual_shape[-2:] if source_shape is None else source_shape
        )
        lease = storage = pool_request = None
        if getattr(stage, "kv_pool", None) is not None:
            pool_request = stage.kv_request(
                sampler,
                inputs,
                layout,
                source_shape=source_shape,
                total_frames=None,
            )
            lease = stage.kv_pool.acquire(owner, pool_request)
            storage = lease.kv_cache
        state = None
        try:
            state = sampler.initialize(inputs, visual, storage=storage)
            cache.fill(
                key,
                inputs,
                identity,
                pool_request if pool_request is not None else request,
                state.cache,
                force=force,
            )
            return cache.finish_fill()
        finally:
            try:
                if state is not None:
                    state.close()
            finally:
                if lease is not None:
                    lease.release()


def updated_preparation_request(codecs, frozen, inputs, prompt, command):
    """Share prompt/reference key updates with the external Encoder."""
    request = {**frozen, "prompt": prompt}
    if "references" in command:
        active = inputs
        old_ids = {item["content_id"] for item in active.conditions}
        refs = preparation_request(codecs, {"media": command["references"]})["media"]
        request["media"] = [
            item for item in request["media"] if item["id"] not in old_ids
        ] + refs
        request["conditions"] = [
            dict(
                content_id=item["id"],
                role="reference",
                latent_modalities=["visual"],
            )
            for item in refs
        ]
    return request
