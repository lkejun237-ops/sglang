# SPDX-License-Identifier: Apache-2.0
"""Request-owned KV transactions for Zing causal generation."""

import copy
import functools
import operator
import os
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
    ZingKVCapacityError,
    ZingLayerStore,
)
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy


def _host_to_device(tensor, device):
    """Copy a host-built plan tensor without a pageable-copy stream sync.

    ``tensor.to(cuda)`` and ``torch.tensor(..., device=cuda)`` from pageable
    memory synchronize the current stream, so planning the next forward would
    wait for every queued forward (for example the previous CUDA graph) before
    the host could continue. The pinned staging block comes from the caching
    host allocator, which keeps it alive until the asynchronous copy has run;
    stream order still places the copy after the queued work that it follows.
    Same contract as ``zing_h3.host_to_device`` (not imported: circular).
    """
    device = torch.device(device)
    if tensor.device == device:
        return tensor
    if device.type != "cuda" or tensor.device.type != "cpu":
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


class ZingFixedKVWorkspace:
    """Device execution metadata over one authoritative pooled K/V store.

    A plan is validated on the host and copied into stable buffers before
    execution. Attention gathers a contiguous live prefix followed by current
    rows and explicitly zeros the padded tail. Commit only scatters current
    clean rows; it never writes padding. No device scalar is read by execution.
    """

    def __init__(self, k, v, *, current_rows):
        if (
            k.ndim != 3
            or k.shape != v.shape
            or k.dtype != v.dtype
            or k.device != v.device
        ):
            raise ValueError(
                "Fixed Zing KV requires matching [rows, heads, dim] stores"
            )
        if type(current_rows) is not int or not 0 < current_rows <= k.shape[0]:
            raise ValueError("Fixed Zing KV current row count must fit its store")
        self.k, self.v = k, v
        self.capacity, self.current_rows = k.shape[0], current_rows
        device = k.device
        self.gather_rows = torch.zeros(
            self.capacity + current_rows, dtype=torch.long, device=device
        )
        self.valid = torch.zeros(
            self.capacity + current_rows, dtype=torch.bool, device=device
        )
        self.write_rows = torch.zeros(current_rows, dtype=torch.long, device=device)
        self.cu_seqlens_k = torch.zeros(2, dtype=torch.int32, device=device)

    def owned_tensors(self):
        return (
            self.k,
            self.v,
            self.gather_rows,
            self.valid,
            self.write_rows,
            self.cu_seqlens_k,
        )

    def prepare(self, history, destinations):
        history, destinations = tuple(history), tuple(destinations)
        if (
            len(history) > self.capacity
            or len(destinations) != self.current_rows
            or len(set(destinations)) != len(destinations)
            or any(
                type(i) is not int or not 0 <= i < self.capacity
                for i in (*history, *destinations)
            )
        ):
            raise ValueError(
                "Fixed Zing KV plan contains invalid history or write rows"
            )
        live = len(history) + self.current_rows
        rows = (*history, *range(self.capacity, self.capacity + self.current_rows))
        rows = (*rows, *((0,) * (self.capacity + self.current_rows - live)))
        self.gather_rows.copy_(
            torch.tensor(rows, dtype=torch.long, device=self.k.device)
        )
        self.valid.copy_(torch.arange(self.valid.numel(), device=self.k.device) < live)
        self.write_rows.copy_(
            torch.tensor(destinations, dtype=torch.long, device=self.k.device)
        )
        self.cu_seqlens_k.copy_(
            torch.tensor([0, live], dtype=torch.int32, device=self.k.device)
        )

    def attention_view(self, current_k, current_v):
        keys = torch.cat((self.k, current_k)).index_select(0, self.gather_rows)
        values = torch.cat((self.v, current_v)).index_select(0, self.gather_rows)
        return (
            torch.where(self.valid[:, None, None], keys, 0),
            torch.where(self.valid[:, None, None], values, 0),
            self.valid,
        )

    def commit(self, current_k, current_v):
        self.k.index_copy_(0, self.write_rows, current_k)
        self.v.index_copy_(0, self.write_rows, current_v)


class ZingFixedForwardKVPlan:
    """Host-prepared fixed-shape device plan, without session bookkeeping.

    The pooled slot is the attention workspace: history is always rows
    ``[0, L)`` (live rows are compacted to the front), the forward's rows are
    written at ``[L, L + rows)``, and attention reads fixed-shape views of the
    slot bounded by ``cu_k``. Nothing is concatenated, gathered or zeroed.
    Rows past the live length are stale but finite (the pool is zero-filled
    and every write is finite), so a kernel that reads past ``cu_k`` is safe.

    The rotated-K scratch of the slot holds one RoPE rotation of raw K. A plan
    rotates the whole fixed-shape raw K itself (``rotates_history``) unless the
    plan executed right before it in the same graph already did so for equal
    history coordinates: the rows ``[0, L)`` are not written in between, so the
    rotation is still exact. Optional cross-graph reuse keeps the stable prefix
    described by the cache's last successful publication and rebuilds its tail.
    """

    def __init__(
        self,
        cache,
        current_positions,
        *,
        ends,
        clean_rows,
        prefix=False,
        after=None,
        current_blocks=None,
    ):
        pool_info = getattr(cache, "_fixed_pool_info", None)
        if pool_info is None or pool_info[0] is not cache.storage:
            stores = tuple((s.k, s.v) for s in cache.storage)
            rotated = tuple(s.rotated_k for s in cache.storage)
            store_sig = tuple(
                (
                    value.data_ptr(),
                    (tuple(value.shape), value.stride(), value.dtype, value.device),
                )
                for value in (
                    *(value for pair in stores for value in pair),
                    *rotated,
                )
            )
            pool_info = (cache.storage, stores, rotated, store_sig)
            cache._fixed_pool_info = pool_info
        _, self.stores, self.rotated, self.store_signature = pool_info
        self.capacity = cache.capacity
        self.rows = current_positions.shape[0]
        self.clean_rows, self.ends = clean_rows, tuple(ends)
        self.prefix = prefix
        if not self.ends or self.ends[-1] != self.rows:
            raise ValueError("Fixed Zing KV branches must cover all current rows")
        if not 0 <= clean_rows <= self.rows:
            raise ValueError("Fixed Zing KV clean rows are invalid")
        history = 0
        for chunk in cache._history():
            if chunk.offset != history:
                raise ValueError("Fixed Zing KV history must be the live front rows")
            history += chunk.positions.shape[0]
        if history != cache._length:
            raise ValueError("Fixed Zing KV history must cover every live row")
        # Host-known write bounds for capture rollback; historical K/V are read-only.
        self.write_start = history
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_sparse_attn import (
            is_three_tier_enabled,
        )

        compressed_active = is_three_tier_enabled() or getattr(
            cache, "_compressed_kv_enabled", False
        )
        # Every row of this forward is written into the slot, clean or not.
        if cache._length + self.rows > self.capacity:
            raise ZingKVCapacityError("Fixed Zing KV plan exceeds slot capacity")
        if compressed_active and cache.storage and len(cache.storage) > 0:
            if self.rows > cache.storage[0].rows:
                raise ZingKVCapacityError(
                    f"Fixed Zing KV forward rows {self.rows} exceed scratch buffer capacity {cache.storage[0].rows}"
                )
        mapped, historical = cache.attention_positions(
            current_positions, prefix=prefix, host_current=current_positions
        )
        self.history_positions = historical
        reusable = (
            historical is not None
            and after is not None
            and after.rotated_positions is not None
            and torch.equal(after.rotated_positions, historical)
        )
        self.rotates_history = (
            (historical is not None and not reusable)
            if not compressed_active
            else False
        )
        self.reused_history_rows = (
            history if (reusable and not compressed_active) else 0
        )
        # Without history nothing is rotated and nothing carries over.
        self.rotated_positions = (
            None
            if (historical is None or compressed_active)
            else historical
            if self.rotates_history
            else after.rotated_positions
        )
        device = self.stores[0][0].device
        write_start = 0 if compressed_active else cache._length
        # Host-built plan inputs use pinned asynchronous copies: this plan is
        # usually built while the previous forward's graph is still running.
        tensors = dict(
            current_positions=_host_to_device(mapped, device),
            write_rows=torch.arange(
                write_start, write_start + self.rows, device=device
            ),
        )
        if self.rotates_history:
            if os.environ.get("SGLANG_ZING_HISTORY_ROPE_LIVE_ROWS", "1") == "1":
                tensors["history_live_rows"] = _host_to_device(
                    torch.tensor([history], dtype=torch.int32), device
                )
            pad_rows = self.capacity - history
            tensors["positions"] = _host_to_device(
                torch.cat((historical, historical.new_zeros(pad_rows, 3))), device
            )
        previous = 0
        for branch, stop in enumerate(self.ends):
            tensors[f"cu_k.{branch}"] = _host_to_device(
                torch.tensor([0, history + stop], dtype=torch.int32), device
            )
            tensors[f"cu_q.{branch}"] = _host_to_device(
                torch.tensor([0, stop - previous], dtype=torch.int32), device
            )
            previous = stop
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_sparse_attn import (
            is_three_tier_enabled,
        )

        self._compressed_kv_stores = getattr(cache, "_compressed_kv_stores", None)
        compressed_active = is_three_tier_enabled() or getattr(
            cache, "_compressed_kv_enabled", False
        )
        if (
            self._compressed_kv_stores is None
            and compressed_active
            and hasattr(cache, "get_compressed_kv_stores")
        ):
            if cache.storage and len(cache.storage) > 0:
                s_k = cache.storage[0].k
                self._compressed_kv_stores = cache.get_compressed_kv_stores(
                    device=device,
                    num_heads=s_k.shape[1] if s_k.ndim >= 3 else 28,
                    head_dim=s_k.shape[2] if s_k.ndim >= 3 else 128,
                )
        self._compressed_kv_store = getattr(cache, "_compressed_kv_store", None)
        if (
            self._compressed_kv_stores is not None
            and len(self._compressed_kv_stores) > 0
            and compressed_active
        ):
            if (
                any(
                    s is not None and s.num_blocks == 0
                    for s in self._compressed_kv_stores
                )
                and cache._length > 0
                and hasattr(cache, "_sync_compressed_kv_store")
            ):
                cache._sync_compressed_kv_store(cache.committed)
            self.compressed_kv_by_layer = [
                s.as_kernel_dict() if s is not None else None
                for s in self._compressed_kv_stores
            ]
            self.compressed_kv = (
                self.compressed_kv_by_layer[0] if self.compressed_kv_by_layer else None
            )
            self._compressed_kv_store = self._compressed_kv_stores[0]
        elif self._compressed_kv_store is not None:
            if (
                self._compressed_kv_store.num_blocks == 0
                and cache._length > 0
                and hasattr(cache, "_sync_compressed_kv_store")
            ):
                cache._sync_compressed_kv_store(cache.committed)
            self.compressed_kv = self._compressed_kv_store.as_kernel_dict()
            self.compressed_kv_by_layer = [self.compressed_kv]
        else:
            self.compressed_kv = None
            self.compressed_kv_by_layer = None
        self.tensors = tensors
        self.compressed_attention_layout = None
        self.prefix_export_layers = None
        if compressed_active:
            from .zing_h3_compressed_view import configure_plan

            configure_plan(self, cache, current_positions, current_blocks)

    def prepare_history_reuse(self, cache):
        """Plan a stable prefix from the last successful rotation publication.

        Called after binding the effective coordinates, on the private planning
        cache only. A miss ends the reusable prefix for every layer; the remaining
        live rows are rebuilt by one device-bounded kernel. Values stay outside
        the graph key, including when the prefix grows, shrinks or becomes empty.
        New clean rows are deliberately not published until a later history pass.
        """
        if not cache.cross_graph_rope_reuse:
            return
        if not self.rotates_history:
            # The preceding plan already published exactly this history in the
            # shadow. Active forwards neither replace nor move its raw rows.
            cache.rope_reuse_stats["reused_rows"] += (
                self.reused_history_rows * cache.num_layers
            )
            return
        reused = self.write_start
        publications = {}
        for layer in range(cache.num_layers):
            cache._observe_raw_store(layer)
            ranges, _ = cache._chunk_ranges(layer)
            previous = set(cache._rotated_store_signatures.get(layer, ()))
            for entry in ranges:
                if entry not in previous:
                    reused = min(reused, entry[0])
                    break
            publications[layer] = tuple(ranges)
            cache._rotated_stores[layer] = self.rotated[layer]
        self.reused_history_rows = reused
        self.tensors["history_rope_range"] = _host_to_device(
            torch.tensor([reused, self.write_start], dtype=torch.int32),
            self.stores[0][0].device,
        )
        # Prospective state only: the owner unpublishes the real cache before
        # execution and publishes this shadow after successful completion.
        cache._rotated_store_signatures = publications
        cache.rope_reuse_stats["reused_rows"] += reused * cache.num_layers
        cache.rope_reuse_stats["rebuilt_rows"] += (
            self.write_start - reused
        ) * cache.num_layers

    def write_current(self, layer, rotated_k, k, v, tensors=None):
        """Write this forward's rows at ``[L, L + rows)``; clean raw K persists."""
        tensors = self.tensors if tensors is None else tensors
        keys, values = self.stores[layer]
        rows = tensors["write_rows"]
        self.rotated[layer].index_copy_(0, rows, rotated_k)
        values.index_copy_(0, rows, v)
        if self.clean_rows:
            keys.index_copy_(0, rows[: self.clean_rows], k[: self.clean_rows])

    def persist(self, layer, k, v, tensors=None):
        """Persist clean rows only; the replicated SP prefix never attends here."""
        tensors = self.tensors if tensors is None else tensors
        if self.clean_rows:
            keys, values = self.stores[layer]
            rows = tensors["write_rows"][: self.clean_rows]
            keys.index_copy_(0, rows, k[: self.clean_rows])
            values.index_copy_(0, rows, v[: self.clean_rows])


def tensor_identity(value):
    """Object identity plus in-place version, when the tensor tracks one."""
    version = None if torch.is_inference(value) else value._version
    return id(value), version


class _StoreLayerViews(Sequence):
    """Immutable pooled range; materialize stable per-layer views only on use.

    Fixed Graph planning consumes pool addresses and row ranges directly. Eager
    attention and snapshots can still index/iterate layers with stable Tensor
    identities, while a planning shadow replaces the range without editing the
    live transaction's existing views.
    """

    __slots__ = ("_stores", "_start", "_stop", "_views")

    def __init__(self, stores, start, stop):
        self._stores, self._start, self._stop = stores, start, stop
        self._views = {}

    def __len__(self):
        return len(self._stores)

    def __getitem__(self, layer):
        if isinstance(layer, slice):
            return tuple(self[i] for i in range(*layer.indices(len(self))))
        layer = operator.index(layer)
        if layer < 0:
            layer += len(self)
        if not 0 <= layer < len(self):
            raise IndexError("Zing KV layer index out of range")
        views = self._views.get(layer)
        if views is None:
            keys, values = self._stores[layer]
            views = (keys[self._start : self._stop], values[self._start : self._stop])
            self._views[layer] = views
        return views


@dataclass
class ZingKVChunk:
    positions: torch.Tensor
    block: int
    layers: Sequence[tuple[torch.Tensor, torch.Tensor]]
    frames: int = 0
    # Store-backed chunks borrow raw or post-RoPE K and the sole persistent V.
    # Only the absolute post-RoPE store pins a commit-time rotation signature.
    offset: int = -1
    position_signature: tuple | None = None
    identity: object = field(default_factory=object, repr=False)
    rope_parts: tuple = ()
    # CPU copy of positions; lets coordinate maps run without device reads.
    host_positions: torch.Tensor | None = None
    # L2-normalized target key mean descriptor for rolling scene sink refresh.
    descriptor: torch.Tensor | None = None
    export_owned: bool = False

    @property
    def temporal_chunk_id(self):
        return self.block // 2 - 1


@dataclass(frozen=True)
class ZingHistoryRoPEPlan:
    """Layer-independent selection for one forward; never a validity decision."""

    owner: object
    state: tuple
    chunks: tuple
    geometry: tuple
    position_signatures: tuple
    rows: int
    runs: tuple


_MAX_MOVE_PIECES = 8


def _move_rows(tensor, destination, source, rows):
    """Copy rows left in place without a temporary buffer.

    Pieces no longer than the shift never overlap, and ascending order reads
    every source row before a later piece overwrites it.
    """
    shift = source - destination
    if shift <= 0:
        raise ValueError("Zing KV compaction only moves rows left")
    if rows > _MAX_MOVE_PIECES * shift:
        # A tiny shift of a long run would launch many copies; one staged
        # copy of this slice is cheaper.
        tensor[destination : destination + rows].copy_(
            tensor[source : source + rows].clone()
        )
        return
    for start in range(0, rows, shift):
        count = min(shift, rows - start)
        tensor[destination + start : destination + start + count].copy_(
            tensor[source + start : source + start + count]
        )


def _compaction_runs(spans):
    """Merge per-chunk left moves into (destination, source, rows) runs.

    ``spans`` are (offset, rows) in row order. Adjacent moved chunks with the
    same shift form one run; unmoved chunks produce none.
    """
    runs, destination = [], 0
    for offset, rows in spans:
        if offset != destination:
            last = runs[-1] if runs else None
            if (
                last is not None
                and last[1] + last[2] == offset
                and last[1] - last[0] == offset - destination
            ):
                runs[-1] = (last[0], last[1], last[2] + rows)
            else:
                runs.append((destination, offset, rows))
        destination += rows
    return runs


def _fused_compaction_rejection(tensors, runs):
    if os.environ.get("SGLANG_ZING_FUSED_KV_COMPACT", "1") != "1":
        return "disabled"
    first = tensors[0]
    if torch.version.hip is not None or any(
        not t.is_cuda or t.device != first.device for t in tensors
    ):
        return "device"
    row_bytes = first[0].numel() * first.element_size()
    if any(
        not t.is_contiguous() or t[0].numel() * t.element_size() != row_bytes
        for t in tensors
    ):
        return "layout"
    from sglang.kernels.ops.diffusion.layout.zing_kv_compact_triton import TILE_ROWS

    if any(source - destination < TILE_ROWS for destination, source, _ in runs):
        return "shift"
    return None


@functools.lru_cache(maxsize=8)
def _device_pointers(pointers, device):
    # A pageable H2D copy would block the host; copy from pinned memory and
    # keep that buffer alive with the result until the copy has run.
    host = torch.tensor(pointers, dtype=torch.int64).pin_memory()
    return host, host.to(device, non_blocking=True)


def _fused_move_rows(tensors, runs):
    """Apply every run to every tensor, one kernel launch per run."""
    from sglang.kernels.ops.diffusion.layout.zing_kv_compact_triton import (
        fused_zing_kv_compact,
    )

    first = tensors[0]
    _, pointers = _device_pointers(tuple(t.data_ptr() for t in tensors), first.device)
    row_bytes = first[0].numel() * first.element_size()
    for destination, source, rows in runs:
        fused_zing_kv_compact(pointers, row_bytes, destination, source, rows)


class ZingH3KVCache:
    """Keep the prefix and complete source/target transactions in one store.

    Active denoising forwards never persist K/V. Positions remain absolute.
    Full-history absolute RoPE stores the K that was rotated at commit time.
    A finite window or relative coordinates keep normalized pre-RoPE K, one V,
    and a rotated-K buffer that is dropped when the attention map changes.
    Under Ulysses, every rank stores the complete cached sequence for only its
    local attention-head shard. A failed request must call clear(). The temporal
    window counts whole source/target/audio transactions. Legacy local_frames
    selects history but does not bound storage.

    With ``storage`` (one preallocated ``ZingLayerStore`` per layer, #171) the
    raw store and rotated-K scratch are views of that fixed slot: nothing is
    allocated or grown, a request that would exceed the slot raises
    ``ZingKVCapacityError``, and clear() only drops logical state and views,
    never the slot itself. The pool owns the slot; returning it is the
    lease's job, not the cache's.
    """

    def __init__(
        self,
        num_layers: int,
        local_frames: int = -1,
        sink_frames: int = 0,
        *,
        reserve_rows: int = 0,
        window_policy: ZingWindowPolicy | None = None,
        window_eviction_timing: str = "after_step0",
        rope_position_mode: str = "absolute",
        rope_max_frame_gap: int = 12,
        storage: list[ZingLayerStore] | None = None,
        retain_prefix_layers: bool = False,
    ):
        if num_layers < 1 or local_frames < -1 or sink_frames < 0:
            raise ValueError("Invalid Zing KV layer count or frame limits")
        if reserve_rows < 0:
            raise ValueError("Invalid Zing KV reserved row count")
        if window_eviction_timing not in ("before_source", "after_step0"):
            raise ValueError("Invalid Zing window eviction timing")
        self._window_eviction_timing = window_eviction_timing
        self.num_layers = num_layers
        self.local_frames = local_frames
        self.sink_frames = sink_frames
        self.window_policy = window_policy or ZingWindowPolicy()
        if self.window_policy.window_chunks is not None and (
            local_frames != -1 or sink_frames
        ):
            raise ValueError(
                "Zing temporal and legacy frame windows cannot be combined"
            )
        ZingWindowPolicy.validate_rope(rope_position_mode, rope_max_frame_gap)
        self.rope_position_mode = rope_position_mode
        self.rope_max_frame_gap = rope_max_frame_gap
        # Post-RoPE K is authoritative only while coordinates cannot move.
        self.single_store = (
            self.window_policy.window_chunks is None
            and rope_position_mode == "absolute"
        )
        self.raw_store = not self.single_store
        # Explicitly set the switch to 0 to restore whole-history invalidation.
        self.chunk_rope_reuse = (
            os.environ.get("SGLANG_ZING_CHUNK_ROPE_REUSE", "1") == "1"
        )
        # A finite request can be much longer than its live attention window.
        # Never reserve its full packed length for a window-bounded store.
        self.reserve_rows = (
            reserve_rows if self.window_policy.window_chunks is None else 0
        )
        if storage is not None:
            if not self.raw_store or self.window_policy.window_chunks is None:
                raise ValueError(
                    "Zing pooled storage requires a finite-window raw store"
                )
            if len(storage) != num_layers:
                raise ValueError("Zing pooled storage must provide every layer")
            self._logical_capacity = getattr(storage[0], "logical_capacity", None)
        else:
            self._logical_capacity = None
        self.storage = storage
        # Explicit preparation/export owns these temporary raw views until its
        # copy completes. Ordinary compressed generation retains no raw history.
        self.retain_prefix_layers = retain_prefix_layers
        # Main-store allocations/growth and compactions, counted apart from
        # derived RoPE/position metadata, which is never pooled.
        self.stats = {
            "store_allocations": 0,
            "rotated_allocations": 0,
            "compactions": 0,
            "sink_refreshes": 0,
        }
        self._sink_refresh_all_history = []
        self._init_sink_refresh_config()
        self._compressed_kv_store = None
        self._compressed_kv_stores = None
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_sparse_attn import (
            is_three_tier_enabled,
        )

        if (
            is_three_tier_enabled()
            and self.storage is not None
            and len(self.storage) > 0
        ):
            sample_k = self.storage[0].k
            self.get_compressed_kv_stores(
                device=sample_k.device,
                num_heads=sample_k.shape[1] if sample_k.ndim >= 3 else 28,
                head_dim=sample_k.shape[2] if sample_k.ndim >= 3 else 128,
            )
        self.clear()

    def _init_sink_refresh_config(self):
        raw_flag = (
            os.environ.get("ZING_SCENE_SINK_REFRESH")
            or os.environ.get("SGLANG_ZING_SCENE_SINK_REFRESH")
            or os.environ.get("ZING_SPARSE_SINK_REFRESH")
            or os.environ.get("SGLANG_ZING_SPARSE_SINK_REFRESH")
            or "auto"
        )
        env_flag = raw_flag.strip().lower()
        if env_flag in ("1", "true", "on"):
            enabled = True
        elif env_flag in ("0", "false", "off"):
            enabled = False
        else:
            raw_mode = (
                os.environ.get("SGLANG_ZING_SPARSE_ATTN_MODE")
                or os.environ.get("ZING_SPARSE_ATTN_MODE")
                or "dense"
            )
            sparse_mode = raw_mode.strip().lower()
            enabled = sparse_mode != "dense"
        self._sink_refresh_enabled = bool(
            enabled
            and self.raw_store
            and self.window_policy.window_chunks is not None
            and self.window_policy.sink_chunks == 3
        )
        self._sink_refresh_cos_thr = float(
            os.environ.get("ZING_SPARSE_SINK_REFRESH_COS_THR", "-0.35")
        )
        self._sink_refresh_cand_min_z = float(
            os.environ.get("ZING_SPARSE_SINK_REFRESH_CAND_MIN_Z", "0.0")
        )
        self._sink_refresh_streak = int(
            os.environ.get("ZING_SPARSE_SINK_REFRESH_STREAK", "16")
        )
        self._sink_refresh_cooldown = int(
            os.environ.get("ZING_SPARSE_SINK_REFRESH_COOLDOWN", "24")
        )
        self._sink_refresh_max_age = int(
            os.environ.get("ZING_SPARSE_SINK_REFRESH_MAX_AGE", "0")
        )
        self._sink_refresh_max_armed_wait = int(
            os.environ.get("ZING_SPARSE_SINK_REFRESH_MAX_ARMED_WAIT", "16")
        )
        self._sink_signal_pinned = None

    @property
    def window_eviction_timing(self):
        """A session's window policy cannot change after construction."""
        return self._window_eviction_timing

    @property
    def pooled(self):
        return self.storage is not None

    @property
    def capacity(self):
        if self.storage is None:
            return None
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_sparse_attn import (
            is_three_tier_enabled,
        )

        if is_three_tier_enabled() or getattr(self, "_compressed_kv_enabled", False):
            stores = getattr(self, "_compressed_kv_stores", None)
            if stores and stores[0] is not None:
                return stores[0].max_blocks * 64
            single = getattr(self, "_compressed_kv_store", None)
            if single is not None:
                return single.max_blocks * 64
            if getattr(self, "_logical_capacity", None) is not None:
                return self._logical_capacity
        return self.storage[0].rows

    @property
    def scratch_capacity(self):
        return None if self.storage is None else self.storage[0].rows

    def clear(self):
        projection_cache = getattr(self, "adaln_projection_cache", None)
        if projection_cache is not None:
            projection_cache.clear()
        self.adaln_projection_cache = None
        self.eviction_target_block = None
        self.prefix = None
        self._prefix_compressed_blocks = None
        self.pending_source = None
        self.committed = []
        self.active = None
        self._active_reference = None
        self._attention_positions_memo = None
        self._effective_history_source = None
        self.last_block = 0
        self.layer_layouts = None
        self._rope_coefficients = {}
        self._model_signature = None
        self._stores = {}
        self._raw_store_versions = {}
        self._raw_write_epochs = {}
        self._length = 0
        self._compressed_part_layout = ()
        self._rotated_stores = {}
        self._rotated_store_signatures = {}
        self._effective_history_positions = None
        self._attention_rope_policy = None
        self._rope_epoch = 0
        self._chunk_geometry = {}
        self._chunk_plan = None
        self.rope_reuse_stats = dict(rebuilt_rows=0, reused_rows=0, map_syncs=0)
        self.last_rope_reuse = None
        self._last_compaction_spans = ()
        if self.window_policy.active_sinks is not None:
            self.window_policy = ZingWindowPolicy(
                self.window_policy.window_chunks,
                self.window_policy.sink_chunks,
                active_sinks=None,
            )
        self._sink_refresh_state = "COOLDOWN"
        self._sink_refresh_last_chunk = 0
        self._sink_low_sim_streak = 0
        self._sink_armed_wait = 0
        self._sink_last_z_scene = 0.0
        self._sink_last_z_cand = 0.0
        self._sink_signal_staged = False
        self._sink_signal_manual = False
        self._sink_refresh_history = []
        self._sink_recent_descs = []
        if getattr(self, "_compressed_kv_stores", None) is not None:
            for store in self._compressed_kv_stores:
                if store is not None:
                    if getattr(self, "_bookkeeping_only", False):
                        store.num_blocks = 0
                    else:
                        store.clear()
        elif getattr(self, "_compressed_kv_store", None) is not None:
            self._compressed_kv_store.clear()
        if not getattr(self, "_bookkeeping_only", False):
            try:
                from sglang.multimodal_gen.runtime.models.dits.zing_h3_sparse_attn import (
                    clear_sparse_caches,
                )

                clear_sparse_caches()
            except Exception:
                pass

    def get_compressed_kv_store(
        self,
        layer_idx: int = 0,
        device="cpu",
        num_heads=16,
        head_dim=128,
        max_blocks=None,
    ):
        """Return or lazily initialize the resident ZingCompressedKVStore for layer_idx."""
        self._compressed_kv_enabled = True
        if getattr(self, "_compressed_kv_stores", None) is None:
            self._compressed_kv_stores = [None] * max(1, self.num_layers)

        if max_blocks is not None:
            max_blks = int(max_blocks)
        elif self.capacity is not None and self.capacity > 0:
            max_blks = max(64, (int(self.capacity) + 63) // 64)
            if (
                getattr(self, "window_policy", None)
                and self.window_policy.window_chunks is not None
            ):
                win_chunks = (
                    int(self.window_policy.window_chunks)
                    + max(0, int(self.window_policy.sink_chunks or 0))
                    + 8
                )
                max_blks = max(max_blks, win_chunks * 2)
        elif (
            getattr(self, "window_policy", None)
            and self.window_policy.window_chunks is not None
        ):
            win_chunks = (
                int(self.window_policy.window_chunks)
                + max(0, int(self.window_policy.sink_chunks or 0))
                + 8
            )
            max_blks = min(512, max(64, win_chunks * 4))
        else:
            max_blks = 128
        if layer_idx >= len(self._compressed_kv_stores):
            self._compressed_kv_stores.extend(
                [None] * (layer_idx + 1 - len(self._compressed_kv_stores))
            )
        if self._compressed_kv_stores[layer_idx] is None:
            pooled_slot = (
                self.storage[layer_idx]
                if self.storage is not None and 0 <= layer_idx < len(self.storage)
                else None
            )
            pooled_store = (
                getattr(pooled_slot, "compressed_kv_store", None)
                if pooled_slot is not None
                else None
            )
            if (
                pooled_store is not None
                and getattr(pooled_store, "max_blocks", 0) >= max_blks
                and getattr(pooled_store, "num_heads", None) == num_heads
                and getattr(pooled_store, "head_dim", None) == head_dim
            ):
                pooled_store.clear()
                self._compressed_kv_stores[layer_idx] = pooled_store
            else:
                from sglang.multimodal_gen.runtime.models.dits.zing_h3_sparse_attn import (
                    ZingCompressedKVStore,
                )

                store = ZingCompressedKVStore(
                    max_blocks=max_blks,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    device=device,
                )
                self._compressed_kv_stores[layer_idx] = store
                if pooled_slot is not None:
                    try:
                        pooled_slot.compressed_kv_store = store
                    except Exception:
                        pass
        self._compressed_kv_store = self._compressed_kv_stores[0]
        return self._compressed_kv_stores[layer_idx]

    def get_compressed_kv_stores(
        self,
        device="cpu",
        num_heads=16,
        head_dim=128,
        max_blocks=None,
    ) -> list:
        """Return or initialize resident ZingCompressedKVStore instances for all layers."""
        stores = []
        for layer_i in range(max(1, self.num_layers)):
            stores.append(
                self.get_compressed_kv_store(
                    layer_idx=layer_i,
                    device=device,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    max_blocks=max_blocks,
                )
            )
        return stores

    def validate_prefix_reset(self):
        prefix = self.prefix
        compressed = getattr(self, "_compressed_kv_stores", None)
        counts = self._prefix_compressed_blocks
        if getattr(self, "_compressed_kv_enabled", False):
            complete_layers = (
                compressed is not None
                and counts is not None
                and len(compressed) == len(counts) == self.num_layers
                and all(
                    store is not None and 0 < count <= store.num_blocks
                    for store, count in zip(compressed, counts)
                )
            )
        else:
            complete_layers = len(self._stores) == self.num_layers
        if (
            prefix is None
            or prefix.offset != 0
            or prefix.block != 0
            or (
                self.pending_source is not None
                and (
                    self.active is None
                    or self.pending_source.block + 1 != self.active[0]
                )
            )
            or self.eviction_target_block is not None
            or not complete_layers
        ):
            raise ValueError(
                f"Resident prefix reset requires complete static KV: offset={getattr(prefix, 'offset', None)} block={getattr(prefix, 'block', None)} source={self.pending_source is not None} eviction={self.eviction_target_block} layers={len(self._stores)}/{self.num_layers}"
            )
        for layer in self._stores:
            self._observe_raw_store(layer)
        return prefix

    def reset_history_keep_prefix(self):
        """Retire dynamic rows at a writer boundary without copying static KV.

        Prefix rows are immutable and at offset zero in the existing leased
        store. Revoke every derived position/graph publication; keep the raw
        store/version guards and model signature so the next write is checked.
        A deferred target final may be discarded just as in a normal rebuild.
        """
        prefix = self.validate_prefix_reset()
        compressed = getattr(self, "_compressed_kv_stores", None)
        if getattr(self, "_compressed_kv_enabled", False) and compressed:
            # The raw scratch rows are not authoritative in compressed mode.
            # Keep the already-quantized prefix bit-for-bit, and retire only
            # dynamic blocks while preserving graph-visible buffer addresses.
            counts = self._prefix_compressed_blocks
            for store, count in zip(compressed, counts):
                if getattr(self, "_bookkeeping_only", False):
                    store.num_blocks = count
                else:
                    store.retain_prefix_blocks(count)
            signature, layouts = self._model_signature, self.layer_layouts
            self._compressed_kv_stores = self._compressed_kv_store = None
            try:
                self.clear()
            finally:
                self._compressed_kv_stores = compressed
                self._compressed_kv_store = compressed[0]
            prefix.layers = ()
            self.prefix = prefix
            self._prefix_compressed_blocks = counts
            self._model_signature, self.layer_layouts = signature, layouts
            self._length = prefix.positions.shape[0]
            self._compressed_part_layout = ((0, self._length),)
            return
        stores = self._stores
        versions, epochs = self._raw_store_versions, self._raw_write_epochs
        signature, layouts = self._model_signature, self.layer_layouts
        self.clear()
        self.prefix, self._stores = prefix, stores
        self._raw_store_versions, self._raw_write_epochs = versions, epochs
        self._model_signature, self.layer_layouts = signature, layouts
        self._length = prefix.positions.shape[0]

    def _invalidate_rope(self, *, preserve_chunks=False):
        """Drop publications and chunk references, retaining reusable K capacity."""
        if not (self.chunk_rope_reuse and preserve_chunks):
            self._rope_coefficients.clear()
            self._rotated_store_signatures.clear()
            self._chunk_geometry.clear()
        self._effective_history_positions = None
        self._effective_history_source = None
        self._chunk_plan = None
        self._rope_epoch += 1

    def bind_attention_positions(self, history_positions):
        """Bind the effective map once per forward, independently of raw positions.

        PR #132 leaves chunk.positions unchanged when relative coordinates move.
        Compare an owned snapshot of the actual attention coordinates so repeated
        denoise steps can reuse rotations without mistaking those raw positions
        for the current map. This comparison may synchronize once per forward.
        """
        policy = (self.rope_position_mode, self.rope_max_frame_gap)
        if self.single_store and (
            self.rope_position_mode != "absolute"
            or self.window_policy.window_chunks is not None
        ):
            raise ValueError("Zing post-RoPE store requires full-history absolute RoPE")
        if policy != self._attention_rope_policy:
            self._invalidate_rope()
            self._attention_rope_policy = policy
        if self.chunk_rope_reuse and self.raw_store:
            self._bind_chunk_positions(history_positions)
            return
        if self.rope_position_mode == "absolute":
            return
        before = self._effective_history_positions
        if before is None and history_positions is None:
            return
        # attention_positions() hands back the same tensors while its inputs are
        # unchanged, so repeated denoising forwards skip the device comparison.
        if (
            before is not None
            and history_positions is not None
            and self._effective_history_source is not None
            and self._effective_history_source[0] is history_positions
            and self._effective_history_source[1] == tensor_identity(history_positions)
        ):
            return
        if (
            before is not None
            and history_positions is not None
            and before.device == history_positions.device
            and before.dtype == history_positions.dtype
            and torch.equal(before, history_positions)
        ):
            self._effective_history_source = (
                history_positions,
                tensor_identity(history_positions),
            )
            return
        self._invalidate_rope()
        if history_positions is not None:
            self._effective_history_positions = history_positions.detach().clone()
            # Hold the bound tensor so its identity cannot be reused.
            self._effective_history_source = (
                history_positions,
                tensor_identity(history_positions),
            )

    @staticmethod
    def _parts(chunk):
        return chunk.rope_parts or ((chunk.identity, chunk.positions.shape[0]),)

    def _bind_chunk_positions(self, positions):
        """Publish one exact coordinate plan shared by all attention layers.

        An unchanged effective (t, h, w) map preserves the coordinate token.
        This is necessary but not sufficient for reuse: each layer also checks
        part identity, raw K version/layout, and raw/rotated physical ranges.
        Source/target parts keep their identities through transaction merging.
        CPU layout guards reject incompatible snapshots; all remaining equality
        flags share one host read, never one synchronization per part.
        """
        chunks = self._history()
        rows = sum(c.positions.shape[0] for c in chunks)
        if positions is None:
            if rows:
                raise ValueError("Zing RoPE requires the current attention map")
            self._chunk_geometry.clear()
            self._chunk_plan = []
            self._effective_history_positions = None
            return
        if positions.shape != (rows, 3):
            raise ValueError("Zing RoPE requires the current attention map")
        candidates, comparisons, offset = [], [], 0
        for chunk in chunks:
            local = 0
            for identity, count in self._parts(chunk):
                end = offset + count
                value = positions[offset:end]
                old = self._chunk_geometry.get(identity)
                # Host and device entries never compare; a switch rebuilds.
                comparable = (
                    old is not None
                    and old[1].shape == value.shape
                    and old[1].dtype == value.dtype
                    and old[1].device == value.device
                )
                index = len(comparisons) if comparable else None
                if comparable:
                    comparisons.append((old[1] == value).all())
                candidates.append(
                    (chunk, identity, local, local + count, value, old, index)
                )
                offset = end
                local += count
        equal = torch.stack(comparisons).tolist() if comparisons else []
        if comparisons and positions.is_cuda:
            self.rope_reuse_stats["map_syncs"] += 1
        geometry, plan = {}, []
        for chunk, identity, start, end, value, old, index in candidates:
            entry = (
                old
                if index is not None and equal[index]
                else (identity, value.detach().clone(), object())
            )
            geometry[identity] = entry
            plan.append((chunk, identity, start, end, entry[1], entry[2]))
        self._chunk_geometry = geometry
        self._chunk_plan = plan
        self._effective_history_positions = positions.detach().clone()

    def _history_geometry(self, chunks):
        """Return mapped positions and stable keys shared by every layer."""
        if self.rope_position_mode == "absolute":
            return [(c.positions, self._tensor_signature(c.positions)) for c in chunks]
        positions = self._effective_history_positions
        rows = sum(c.positions.shape[0] for c in chunks)
        if positions is None or positions.shape != (rows, 3):
            if not chunks and positions is None:
                return []
            raise ValueError("Zing relative RoPE requires the current attention map")
        geometry, offset = [], 0
        for chunk in chunks:
            end = offset + chunk.positions.shape[0]
            geometry.append((positions[offset:end], (self._rope_epoch, offset, end)))
            offset = end
        return geometry

    def _history_plan_state(self):
        return (
            self._rope_epoch,
            self._length,
            self.last_block,
            id(self.prefix),
            id(self.pending_source),
            self.rope_position_mode,
            self.rope_max_frame_gap,
            self.local_frames,
            self.sink_frames,
            self.window_policy,
        )

    def prepare_history_rope_plan(self):
        """Build geometry once per forward, after binding effective positions."""
        chunks = tuple(self._history())
        return ZingHistoryRoPEPlan(
            self,
            self._history_plan_state(),
            chunks,
            tuple(self._history_geometry(chunks)),
            tuple(self._tensor_signature(c.positions) for c in chunks),
            sum(c.positions.shape[0] for c in chunks),
            tuple(self._runs(chunks)),
        )

    def owned_tensors(self):
        """Include derived storage in session memory/release accounting."""
        # A pooled slot outlives the session; count only session-owned storage.
        for chunk in self._stored_chunks():
            yield chunk.positions
            if chunk.host_positions is not None:
                yield chunk.host_positions
            if chunk.descriptor is not None:
                yield chunk.descriptor
            if not self.pooled:
                for pair in chunk.layers:
                    yield from pair
        for _, desc in getattr(self, "_sink_recent_descs", ()):
            if desc is not None:
                yield desc
        if not self.pooled:
            for pair in self._stores.values():
                yield from pair
            yield from self._rotated_stores.values()
        if getattr(self, "_compressed_kv_stores", None) is not None:
            for store in self._compressed_kv_stores:
                if store is not None:
                    yield store.k_res_fp8
                    yield store.v_res_fp8
                    yield store.k_res_scale
                    yield store.v_res_scale
                    yield store.k_blk_mean
                    yield store.v_blk_mean
                    yield store.k_sub_mean_fp8
                    yield store.v_sub_mean_fp8
                    yield store.k_sub_scale
                    yield store.v_sub_scale
                    yield store.k_blk_mean_fp8
                    yield store.v_blk_mean_fp8
                    yield store.k_blk_scale
                    yield store.v_blk_scale
                    yield store.tau_k
                    yield store.rho_kv
                    yield store.block_valid_len
                    yield store.block_delta_t
                    yield store.block_is_shifted
                    yield store.block_chunk_id
        yield from self._rope_coefficients.values()
        if self._effective_history_positions is not None:
            yield self._effective_history_positions
        for _, positions, _ in self._chunk_geometry.values():
            yield positions
        if self.active is not None:
            yield self.active[1]

    def planning_copy(self):
        """Private host transaction; all authoritative tensors keep their addresses."""
        shadow = copy.copy(self)
        shadow.__dict__ = self.__dict__.copy()
        for name, value in self.__dict__.items():
            if isinstance(value, dict):
                shadow.__dict__[name] = value.copy()
            elif isinstance(value, list):
                shadow.__dict__[name] = value.copy()
        shadow.prefix = None if self.prefix is None else copy.copy(self.prefix)
        shadow.committed = [copy.copy(chunk) for chunk in self.committed]
        shadow.pending_source = (
            None if self.pending_source is None else copy.copy(self.pending_source)
        )
        shadow._bookkeeping_only = True
        if self._compressed_kv_stores is not None:
            # Only metadata is private. GPU arrays keep their leased addresses.
            shadow._compressed_kv_stores = [
                None if store is None else copy.copy(store)
                for store in self._compressed_kv_stores
            ]
            shadow._compressed_kv_store = shadow._compressed_kv_stores[0]
        shadow._attention_positions_memo = None
        shadow._last_compaction_spans = ()
        return shadow

    def release_prefix_export(self):
        """A cache writer has taken ownership; generation retains FP8 only."""
        if getattr(self, "_compressed_kv_enabled", False):
            self.retain_prefix_layers = False
            if self.prefix is not None and self.prefix.export_owned:
                self.prefix.layers = ()
                self.prefix.export_owned = False

    def publish_plan(self, shadow):
        """Publish successful device execution's host effects exactly once."""
        if not getattr(shadow, "_bookkeeping_only", False):
            raise ValueError("Zing KV host publication requires a planning transaction")
        if shadow.cross_graph_rope_reuse:
            # Warmup, capture, replay and compaction performed cache-owned writes
            # after planning. Acknowledge those versions before the next observer.
            for layer in shadow._stores:
                shadow._mark_raw_store(layer)
        prev_last_block = self.last_block
        self.__dict__.update(shadow.__dict__)
        self._bookkeeping_only = False
        if self.last_block > prev_last_block:
            self._stage_sink_refresh_signal_async()

    def record_sink_refresh_signal(self, z_scene: float, z_cand: float = 0.0):
        """Record precomputed or test-injected scene and candidate similarity z-scores."""
        self._sink_last_z_scene = float(z_scene)
        self._sink_last_z_cand = float(z_cand)
        self._sink_signal_staged = False
        self._sink_signal_manual = True

    def _stage_sink_refresh_signal_async(self):
        """Compute target chunk descriptor and stage (z_scene, z_cand) to pinned host memory."""
        if (
            not self._sink_refresh_enabled
            or getattr(self, "_bookkeeping_only", False)
            or self._sink_signal_manual
            or not self.committed
            or (not self._stores and not getattr(self, "_compressed_kv_enabled", False))
        ):
            return
        rep_layer = self.num_layers // 2
        store = self._stores.get(rep_layer)
        compressed = getattr(self, "_compressed_kv_enabled", False)
        if store is None and not compressed:
            return
        keys = None if compressed else store[0]
        # Avoid triggering custom __torch_function__ traps (e.g. DeviceOnly in host-sync tests).
        if not compressed and type(keys) is not torch.Tensor:
            return
        seen_ids = {uid for uid, _ in self._sink_recent_descs}
        for chunk in self.committed:
            if chunk.descriptor is None:
                tgt_rows = (
                    chunk.rope_parts[-1][1]
                    if len(chunk.rope_parts) >= 2
                    else chunk.positions.shape[0]
                )
                start = chunk.offset + (chunk.positions.shape[0] - tgt_rows)
                stop = chunk.offset + chunk.positions.shape[0]
                if compressed:
                    value = self._compressed_kv_stores[rep_layer]
                    part_block = 0
                    for part_start, count in self._compressed_part_layout:
                        blocks = (count + 63) // 64
                        if part_start == start and part_start + count == stop:
                            means = value.k_blk_mean[
                                :, part_block : part_block + blocks
                            ].float()
                            weights = value.block_valid_len[
                                part_block : part_block + blocks
                            ].float()
                            vec = (
                                (means * weights[None, :, None]).sum(1)
                                / weights.sum().clamp_min(1)
                            ).reshape(-1)
                            chunk.descriptor = torch.nn.functional.normalize(
                                vec, dim=0, eps=1e-6
                            )
                            break
                        part_block += blocks
                elif 0 <= start < stop <= keys.shape[0]:
                    vec = keys[start:stop].float().mean(dim=0).reshape(-1)
                    chunk.descriptor = torch.nn.functional.normalize(
                        vec, dim=0, eps=1e-6
                    )
            if chunk.descriptor is not None and chunk.temporal_chunk_id not in seen_ids:
                self._sink_recent_descs.append(
                    (chunk.temporal_chunk_id, chunk.descriptor)
                )
                seen_ids.add(chunk.temporal_chunk_id)
        if len(self._sink_recent_descs) > 20:
            self._sink_recent_descs = self._sink_recent_descs[-20:]
        if len(self.committed) < 6 or self.committed[-1].descriptor is None:
            return
        s1_desc = self.committed[1].descriptor
        s2_desc = self.committed[2].descriptor
        cand_desc = self.committed[3].descriptor
        curr_desc = self.committed[-1].descriptor
        if s1_desc is None or s2_desc is None or cand_desc is None:
            return
        curr_id = self.committed[-1].temporal_chunk_id
        win_descs = [
            d
            for uid, d in self._sink_recent_descs
            if uid != curr_id and not self.window_policy.is_sink(uid) and d is not None
        ][-16:]
        if len(win_descs) < 2:
            win_descs = [
                c.descriptor for c in self.committed[3:-1] if c.descriptor is not None
            ]
        if len(win_descs) < 2:
            return
        hist_mat = torch.stack([s1_desc, s2_desc, cand_desc, *win_descs], dim=0)
        sims = torch.mv(hist_mat, curr_desc)
        win_sims = sims[3:]
        win_mean = win_sims.mean()
        win_std = win_sims.std(unbiased=False).clamp(min=0.05)
        sink_mean = 0.5 * (sims[0] + sims[1])
        raw_z_scene = (sink_mean - win_mean) / win_std
        z_scene = torch.where(
            (win_mean - sink_mean) >= 0.06,
            raw_z_scene,
            torch.maximum(raw_z_scene, torch.zeros_like(raw_z_scene)),
        )
        z_cand = (sims[2] - win_mean) / win_std
        pair = torch.stack((z_scene, z_cand))
        if pair.is_cuda:
            if self._sink_signal_pinned is None:
                with torch.inference_mode(False), torch.no_grad():
                    self._sink_signal_pinned = torch.zeros(
                        2, dtype=torch.float32, pin_memory=True
                    )
            self._sink_signal_pinned.copy_(pair, non_blocking=True)
            self._sink_signal_staged = True
        else:
            self._sink_last_z_scene = float(pair[0].item())
            self._sink_last_z_cand = float(pair[1].item())
            self._sink_signal_staged = False

    def _trim_chunk_to_target(self, chunk: ZingKVChunk):
        """Strip the leading source half of a merged chunk in place so only clean target rows remain."""
        if len(chunk.rope_parts) < 2:
            return
        tgt_id, tgt_rows = chunk.rope_parts[-1]
        total_rows = chunk.positions.shape[0]
        src_rows = total_rows - tgt_rows
        if src_rows <= 0 or tgt_rows <= 0:
            return
        with torch.inference_mode(False), torch.no_grad():
            chunk.positions = chunk.positions[src_rows:].detach().clone()
            if chunk.host_positions is not None:
                chunk.host_positions = chunk.host_positions[src_rows:].detach().clone()
        chunk.offset += src_rows
        chunk.rope_parts = ((tgt_id, tgt_rows),)
        chunk.position_signature = self._tensor_signature(chunk.positions)
        if self._stores and len(self._stores) == self.num_layers:
            chunk.layers = self._chunk_layers(chunk.offset, tgt_rows)
        elif chunk.layers:
            chunk.layers = tuple((k[src_rows:], v[src_rows:]) for k, v in chunk.layers)

    def _maybe_step_sink_refresh(self, target_block: int):
        """Advance the parity-matched two-step rolling scene sink refresh state machine."""
        if not self._sink_refresh_enabled:
            return
        window = self.window_policy.window_chunks
        sinks = self.window_policy.sink_chunks
        if window is None or sinks != 3 or len(self.committed) < window:
            return
        current_id = target_block // 2 - 1
        if self._sink_signal_staged and self._sink_signal_pinned is not None:
            self._sink_last_z_scene = float(self._sink_signal_pinned[0].item())
            self._sink_last_z_cand = float(self._sink_signal_pinned[1].item())
            self._sink_signal_staged = False
        self._sink_signal_manual = False

        cand = self.committed[3]
        cand_id = cand.temporal_chunk_id

        if self._sink_refresh_state == "STEP1_ODD":
            s_even_old = self.committed[1]
            even_rows = s_even_old.positions.shape[0]
            cand_tgt_rows = (
                cand.rope_parts[-1][1]
                if len(cand.rope_parts) >= 2
                else cand.positions.shape[0]
            )
            if cand_tgt_rows == even_rows or cand.positions.shape[0] == even_rows:
                if cand_tgt_rows == even_rows and cand.positions.shape[0] != even_rows:
                    self._trim_chunk_to_target(cand)
                s_odd_new = self.committed[2].temporal_chunk_id
                new_sinks = (0, s_odd_new, cand_id)
                self.window_policy = ZingWindowPolicy(
                    window, sinks, active_sinks=new_sinks
                )
                self._sink_refresh_state = "COOLDOWN"
                self._sink_refresh_last_chunk = current_id
                self._sink_low_sim_streak = 0
                self._sink_armed_wait = 0
                self.stats["sink_refreshes"] = self.stats.get("sink_refreshes", 0) + 1
                self._sink_refresh_history.append((current_id, new_sinks))
                self._sink_refresh_all_history.append(
                    (
                        current_id,
                        new_sinks,
                        round(self._sink_last_z_scene, 4),
                        round(self._sink_last_z_cand, 4),
                    )
                )
            return

        if self._sink_refresh_state == "COOLDOWN":
            if (
                current_id - self._sink_refresh_last_chunk
                >= self._sink_refresh_cooldown
            ):
                self._sink_refresh_state = "MONITOR"
                self._sink_low_sim_streak = 0
            else:
                return

        if self._sink_refresh_state == "MONITOR":
            if self._sink_last_z_scene < self._sink_refresh_cos_thr:
                self._sink_low_sim_streak += 1
            else:
                self._sink_low_sim_streak = 0
            cur_sinks = self.window_policy.effective_sinks()
            sink_age = current_id - max(cur_sinks[1], cur_sinks[2])
            age_expired = (
                self._sink_refresh_max_age > 0
                and sink_age >= self._sink_refresh_max_age
            )
            if self._sink_low_sim_streak >= self._sink_refresh_streak or age_expired:
                self._sink_refresh_state = "ARMED"
                self._sink_armed_wait = 0

        if self._sink_refresh_state == "ARMED":
            stable = (
                self._sink_last_z_cand >= self._sink_refresh_cand_min_z
                or self._sink_armed_wait >= self._sink_refresh_max_armed_wait
            )
            s_odd_old = self.committed[1]
            odd_rows = s_odd_old.positions.shape[0]
            cand_tgt_rows = (
                cand.rope_parts[-1][1]
                if len(cand.rope_parts) >= 2
                else cand.positions.shape[0]
            )
            if stable and (
                cand_tgt_rows == odd_rows or cand.positions.shape[0] == odd_rows
            ):
                if cand_tgt_rows == odd_rows and cand.positions.shape[0] != odd_rows:
                    self._trim_chunk_to_target(cand)
                s_even_old_id = self.committed[2].temporal_chunk_id
                new_sinks = (0, s_even_old_id, cand_id)
                self.window_policy = ZingWindowPolicy(
                    window, sinks, active_sinks=new_sinks
                )
                self._sink_refresh_state = "STEP1_ODD"
                self._sink_armed_wait = 0
            else:
                self._sink_armed_wait += 1

    @property
    def cross_graph_rope_reuse(self):
        return (
            self.pooled
            and self.raw_store
            and self.chunk_rope_reuse
            and os.environ.get("SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH", "1") == "1"
        )

    def begin_chunk(self, target_block):
        """Open a target boundary without changing delayed-policy history."""
        if self.eviction_target_block is not None:
            raise ValueError("Zing window selection has an unfinished boundary")
        if self.active is not None or self.pending_source is not None:
            raise ValueError("Zing window selection requires a completed boundary")
        if target_block != self.last_block + 2:
            raise ValueError("Zing window selection requires the next target block")
        self.eviction_target_block = target_block
        if self.window_eviction_timing == "before_source":
            self._evict_committed(target_block)

    def complete_step0_and_evict(self, target_block):
        """Close step zero only after both scheduler updates have succeeded."""
        if self.eviction_target_block != target_block:
            raise ValueError("Zing step-zero completion requires its open boundary")
        if self.active is None or self.active[0] != target_block:
            raise ValueError("Zing step-zero completion requires the active target")
        if (
            self.pending_source is not None
            and self.pending_source.block != target_block - 1
        ):
            raise ValueError("Zing step-zero completion has a mismatched source")
        if self.window_eviction_timing == "after_step0":
            self._evict_committed(target_block)
        self.eviction_target_block = None

    def prepare_chunk(self, target_block):
        """Compatibility boundary for callers using the original window API."""
        self.begin_chunk(target_block)
        if self.window_eviction_timing == "before_source":
            self.eviction_target_block = None

    def _evict_committed(self, target_block):
        self._maybe_step_sink_refresh(target_block)
        current_id = target_block // 2 - 1
        retained = [
            chunk
            for chunk in self.committed
            if self.window_policy.retains(chunk.temporal_chunk_id, current_id)
        ]
        if len(retained) != len(self.committed):
            self.committed = retained
            self._invalidate_rope(preserve_chunks=True)
            self._compact_store()
            self.stats["compactions"] += 1
            self._sync_compressed_kv_store(retained)
        else:
            self._last_compaction_spans = tuple(
                (c.offset, c.positions.shape[0]) for c in self._stored_chunks()
            )

    def _sync_compressed_kv_store(self, retained):
        """Compact complete source/target parts without reading device owner IDs."""
        if not getattr(self, "_compressed_kv_enabled", False):
            return
        chunks = ([self.prefix] if self.prefix is not None else []) + list(retained)
        if self.pending_source is not None:
            chunks.append(self.pending_source)
        self.move_store_rows(tuple((c.offset, c.positions.shape[0]) for c in chunks))

    def update_sink(self, operation, *, current_id):
        """Select existing clean chunks at a quiescent native block boundary."""
        if operation not in {"off", "refresh"}:
            raise ValueError("Sink operation must be off or refresh")
        policy = self.window_policy
        if policy.window_chunks is None or policy.window_chunks < 4:
            raise ValueError("Dynamic sinks require a window of at least four chunks")
        if (
            self.active is not None
            or self.pending_source is not None
            or self.eviction_target_block is not None
        ):
            raise ValueError("Sink updates require a completed KV boundary")
        if current_id != self.last_block // 2:
            raise ValueError("Sink update does not match the next model chunk")
        if operation == "off":
            if not policy.sink_chunks:
                return
            minimum = max(
                policy.min_chunk_id,
                min(policy.sink_start + policy.sink_chunks, current_id),
            )
            self.window_policy = ZingWindowPolicy(policy.window_chunks, 0, 0, minimum)
        else:
            latest = self.committed[-2:]
            if (
                len(latest) == 2
                and latest[1].temporal_chunk_id != latest[0].temporal_chunk_id + 1
            ):
                raise ValueError("Sink refresh requires consecutive recent chunks")
            start = latest[0].temporal_chunk_id if latest else current_id
            self.window_policy = ZingWindowPolicy(policy.window_chunks, 2, start, start)
        self._evict_committed(2 * current_id + 2)

    def move_store_rows(self, spans):
        """Move live raw K/V rows left; ``spans`` are (offset, rows) in row order.

        Device rows only: chunk offsets, length and rope bookkeeping stay with
        the caller (a planning transaction publishes them later).
        """
        if getattr(self, "_compressed_kv_enabled", False):
            selected, layout, block, destination = [], [], 0, 0
            # Snapshot of physical parts, independent of a host list that may
            # already have dropped/trimmed a source or promoted a target sink.
            for offset, count in self._compressed_part_layout:
                blocks = (count + 63) // 64
                if any(
                    start <= offset and offset + count <= start + rows
                    for start, rows in spans
                ):
                    selected.extend(range(block, block + blocks))
                    layout.append((destination, count))
                    destination += count
                elif any(
                    start < offset + count and offset < start + rows
                    for start, rows in spans
                ):
                    raise ValueError("Compressed compaction splits a committed part")
                block += blocks
            if destination != sum(rows for _, rows in spans):
                raise ValueError("Compressed compaction does not cover retained rows")
            for store in self._compressed_kv_stores or ():
                if store is not None:
                    if getattr(self, "_bookkeeping_only", False):
                        store.num_blocks = len(selected)
                    else:
                        store.compact_blocks(selected)
            self._compressed_part_layout = tuple(layout)
            return
        runs = _compaction_runs(spans)
        if not runs:
            return
        tensors = [tensor for pair in self._stores.values() for tensor in pair]
        if not tensors:
            return
        if _fused_compaction_rejection(tensors, runs) is None:
            _fused_move_rows(tensors, runs)
            return
        # Per chunk, a move is at most a few pieces of its shift.
        for keys, values in self._stores.values():
            destination = 0
            for source, rows in spans:
                if source != destination:
                    for tensor in (keys, values):
                        _move_rows(tensor, destination, source, rows)
                destination += rows

    def _compact_store(self):
        """Move live raw K/V to the front, using at most one chunk temporarily."""
        chunks = self._stored_chunks()
        spans = tuple((c.offset, c.positions.shape[0]) for c in chunks)
        self._last_compaction_spans = spans
        for layer in self._stores:
            self._observe_raw_store(layer)
        if getattr(self, "_compressed_kv_enabled", False) or not getattr(
            self, "_bookkeeping_only", False
        ):
            self.move_store_rows(spans)
        for layer in self._stores:
            self._mark_raw_store(layer)
        offset = 0
        for chunk in chunks:
            chunk.offset = offset
            offset += chunk.positions.shape[0]
        self._length = offset
        if getattr(self, "_compressed_kv_enabled", False):
            return  # Raw scratch is not an authoritative historical view.
        if self.pooled and len(self._stores) == self.num_layers:
            stores = tuple(self._stores[i] for i in range(self.num_layers))
            for chunk in chunks:
                chunk.layers = _StoreLayerViews(
                    stores, chunk.offset, chunk.offset + chunk.positions.shape[0]
                )
        else:
            for layer in self._stores:
                self._rebind(layer)

    def _history(self):
        chunks = self.committed
        if self.local_frames >= 0:
            first, last, count = 0, len(chunks), 0
            while first < len(chunks) and count < self.sink_frames:
                count += chunks[first].frames
                first += 1
            count = 0
            while last > first and count < self.local_frames:
                last -= 1
                count += chunks[last].frames
            chunks = chunks[:first] + chunks[last:]
        return (
            ([self.prefix] if self.prefix is not None else [])
            + chunks
            + ([self.pending_source] if self.pending_source is not None else [])
        )

    def _stored_chunks(self):
        """Every chunk holding store rows, in row order; window selection aside."""
        return (
            ([self.prefix] if self.prefix is not None else [])
            + self.committed
            + ([self.pending_source] if self.pending_source is not None else [])
        )

    def positions(self):
        chunks = self._history()
        return torch.cat([c.positions for c in chunks]) if chunks else None

    def attention_positions(self, current, *, prefix=False, host_current=None):
        """Rebuild RoPE coordinates only; cached positions/provenance stay absolute.

        Unlike MinWM token order, Zing repeats time across source, target and
        stereo audio. Compress sorted unique media timestamps together, so equal
        timestamps share a coordinate regardless of packed modality/role order.
        Prefix/reference coordinates are unchanged. Gap units are checkpoint
        temporal coordinates (not pixels, latent frames or temporal chunk IDs).
        """
        chunks = self._history()
        # Map on host copies when every history chunk has one; no device read.
        on_host = host_current is not None and all(
            c.host_positions is not None for c in chunks
        )
        if on_host:
            current = host_current
        key = (
            on_host,
            tensor_identity(current),
            prefix,
            self.rope_position_mode,
            self.rope_max_frame_gap,
            self.prefix is not None,
            tuple(
                tensor_identity(c.host_positions if on_host else c.positions)
                for c in chunks
            ),
        )
        memo = self._attention_positions_memo
        if memo is not None and memo[0] == key:
            return memo[2]
        history = self.host_positions() if on_host else self.positions()
        result = self._map_attention_positions(current, prefix, history)
        # Keep the keyed tensors alive so their identities cannot be reused.
        self._attention_positions_memo = (
            key,
            (current, *(c.host_positions if on_host else c.positions for c in chunks)),
            result,
        )
        return result

    def host_positions(self):
        chunks = self._history()
        return torch.cat([c.host_positions for c in chunks]) if chunks else None

    def _map_attention_positions(self, current, prefix, history=None):
        if history is None:
            history = self.positions()
        if self.rope_position_mode == "absolute" or prefix:
            return current, history
        prefix_rows = 0 if self.prefix is None else self.prefix.positions.shape[0]
        joined = current if history is None else torch.cat((history, current))
        mapped = joined.clone()
        media = joined[prefix_rows:, 0]
        times, inverse = torch.unique(media, sorted=True, return_inverse=True)
        if times.numel():
            anchor = (
                joined[:prefix_rows, 0].max() + 1
                if prefix_rows
                else times.new_zeros(())
            )
            gaps = times[1:] - times[:-1]
            compressed = torch.cat(
                (times.new_zeros(1), gaps.clamp(max=self.rope_max_frame_gap))
            ).cumsum(0)
            mapped[prefix_rows:, 0] = anchor + compressed[inverse]
        n = 0 if history is None else history.shape[0]
        return mapped[n:], None if history is None else mapped[:n]

    def history(self, layer):
        chunks = self._history()
        if not chunks:
            return None
        return tuple(
            torch.cat([c.layers[layer][i] for c in chunks], dim=0) for i in (0, 1)
        )

    def bind_model(self, signature):
        """Reject weight/RoPE changes while authoritative history is live.

        Changing the model requires a fresh session: rebuilding rotated keys alone
        cannot repair hidden states produced by the previous weights.
        """
        if self._model_signature is not None and signature != self._model_signature:
            if self._history() or self.active is not None:
                raise ValueError(
                    "Zing model changed within a cached session; clear it first"
                )
            self._invalidate_rope()
            # Pooled views are rebound from the slot; the slot is kept.
            self._stores.clear()
            self._rotated_stores.clear()
            self._length = 0
        self._model_signature = signature

    @staticmethod
    def _tensor_signature(value):
        # Inference tensors have no version counter. Cache-owned tensors are
        # immutable; replacements are detected by identity in either mode.
        try:
            version = value._version
        except RuntimeError:
            version = None
        return (id(value), version, tuple(value.shape), value.dtype, value.device)

    def _check_append_layout(self, layer, k, v):
        if (
            k.ndim != 3
            or k.shape != v.shape
            or k.dtype != v.dtype
            or k.device != v.device
            or (
                self.layer_layouts is not None
                and (tuple(k.shape[1:]), k.dtype, k.device) != self.layer_layouts[layer]
            )
        ):
            raise ValueError("Zing attention append changed the K/V head-shard layout")

    def _reserve_store(self, layer, needed, k, v):
        """Size authoritative K/V (raw K for windows, post-RoPE for absolute).

        Only this layer's rows are ever duplicated, so a grown or shifted store
        costs one layer of transient storage rather than the whole cache.
        """
        store = self._stores.get(layer)
        self._observe_raw_store(layer)
        if store is not None and store[0].shape[0] >= needed:
            return store
        if self.pooled:
            return self._bind_slot(layer, needed, k)
        self.stats["store_allocations"] += 1
        capacity = max(needed, self.reserve_rows)
        if store is not None and not self.reserve_rows:
            capacity = max(capacity, 2 * store[0].shape[0])
        # Version raw storage even when the caller uses inference_mode. This
        # makes unaccounted content changes observable without reading the GPU.
        with (
            torch.inference_mode(False)
            if self.chunk_rope_reuse and self.raw_store
            else nullcontext()
        ):
            keys = k.new_empty((capacity, *k.shape[1:]))
        values = v.new_empty((capacity, *v.shape[1:]))
        if store is not None and self._length:
            keys[: self._length].copy_(store[0][: self._length])
            values[: self._length].copy_(store[1][: self._length])
        self._stores[layer] = (keys, values)
        self._mark_raw_store(layer)
        # Views of the old buffer would keep it alive and read stale rows.
        self._rebind(layer)
        return self._stores[layer]

    def _observe_raw_store(self, layer):
        """Detect writes outside the cache before acknowledging owned writes.

        Normal tensor views share their allocation's version counter. A legal
        scratch-tail append must not invalidate every historical chunk, but an
        unaccounted write must. Candidate raw allocations are versioned even
        inside inference_mode, so this check also covers external in-place edits.
        """
        if not (self.chunk_rope_reuse and self.raw_store and layer in self._stores):
            return
        signature = self._tensor_signature(self._stores[layer][0])
        before = self._raw_store_versions.get(layer)
        if before is not None and before != signature:
            self._raw_write_epochs[layer] = self._raw_write_epochs.get(layer, 0) + 1
        self._raw_store_versions[layer] = signature

    def _mark_raw_store(self, layer):
        if self.chunk_rope_reuse and self.raw_store:
            self._raw_store_versions[layer] = self._tensor_signature(
                self._stores[layer][0]
            )

    def _raw_chunk_signature(self, layer, chunk, identity, start, end):
        store = self._stores[layer][0]
        if (
            isinstance(chunk.layers, _StoreLayerViews)
            and chunk.layers._stores[layer][0] is store
        ):
            begin, stop, _ = slice(chunk.layers._start, chunk.layers._stop).indices(
                store.shape[0]
            )
            rows = max(0, stop - begin)
            address = (
                store.data_ptr()
                + (begin + start) * store.stride(0) * store.element_size()
                if rows
                else 0
            )
            return (
                identity,
                self._raw_write_epochs.get(layer, 0),
                address,
                end - start,
                tuple(store.shape[1:]),
                store.stride(),
                store.dtype,
                store.device,
            )
        value = chunk.layers[layer][0]
        owned = value.untyped_storage().data_ptr() == store.untyped_storage().data_ptr()
        # Cache-owned parts can acquire a new parent view when source/target are
        # merged. Content epoch and physical range remain unchanged in that case.
        version = (
            self._raw_write_epochs.get(layer, 0)
            if owned
            else self._tensor_signature(value)
        )
        return (
            identity,
            version,
            value.data_ptr() + start * value.stride(0) * value.element_size(),
            end - start,
            tuple(value.shape[1:]),
            value.stride(),
            value.dtype,
            value.device,
        )

    def _bind_slot(self, layer, needed, k):
        """Expose this layer's fixed slot as the store; never allocate or grow."""
        slot = self.storage[layer]
        if needed > slot.rows:
            raise ZingKVCapacityError(
                f"Zing pooled KV needs {needed} rows but the slot holds {slot.rows}"
            )
        if (tuple(k.shape[1:]), k.dtype, k.device) != slot.layout:
            raise ValueError("Zing K/V head-shard layout does not match the pool slot")
        if layer not in self._stores:
            self._stores[layer] = (slot.k, slot.v)
            self._mark_raw_store(layer)
            self._rebind(layer)
        return self._stores[layer]

    def _rebind(self, layer):
        store = self._stores.get(layer)
        if store is None:
            return
        # A view's address/shape can be checked without constructing a probe
        # Tensor. Compaction calls this for every layer and retained chunk;
        # unchanged sink/prefix views must not allocate lists or slice wrappers.
        keys, values = store
        key_address, value_address = keys.data_ptr(), values.data_ptr()
        key_step = keys.stride(0) * keys.element_size()
        value_step = values.stride(0) * values.element_size()
        capacity, tail_shape = keys.shape[0], tuple(keys.shape[1:])
        for chunk in self._stored_chunks():
            start, stop = chunk.offset, chunk.offset + chunk.positions.shape[0]
            complete = len(chunk.layers) == self.num_layers
            old = chunk.layers[layer] if complete else None
            if self.chunk_rope_reuse and old is not None:
                old_k, old_v = old
                begin, end, _ = slice(start, stop).indices(capacity)
                rows = max(0, end - begin)
                if (
                    old_k.data_ptr() == (key_address + begin * key_step if rows else 0)
                    and old_v.data_ptr()
                    == (value_address + begin * value_step if rows else 0)
                    and old_k.shape == (rows, *tail_shape)
                ):
                    continue
            layers = list(chunk.layers) if complete else [None] * self.num_layers
            layers[layer] = (keys[start:stop], values[start:stop])
            chunk.layers = tuple(layers)

    def _verify_committed_positions(self, chunks):
        """A dropped raw key cannot be re-rotated, so a moved position is fatal."""
        for chunk in chunks:
            if self._tensor_signature(chunk.positions) != chunk.position_signature:
                raise ValueError(
                    "Zing single-store positions changed after commit; clear the session"
                )

    @staticmethod
    def _runs(chunks):
        runs = []
        for chunk in chunks:
            start, stop = chunk.offset, chunk.offset + chunk.positions.shape[0]
            if runs and runs[-1][1] == start:
                runs[-1][1] = stop
            else:
                runs.append([start, stop])
        return runs

    def attention_view(self, layer, k, v, write_kv):
        """Return [history + current, local heads, dim] over the post-RoPE store.

        `k` is the current block's unrotated K. `write_kv(k, v, out_k, out_v)`
        writes rotated K and V into the scratch tail past the committed length,
        or into a fresh pair when history is empty. That pair is not stored:
        the prefix commit copies it after the head shard is chosen. Committed
        rows were rotated by the forward that produced them and are never
        rotated, copied or concatenated again. A selection that skips stored
        chunks cannot be one view, so those runs are gathered instead. The
        returned views are borrowed until the next call on that layer. The
        writer only receives scratch rows, never committed history.
        """
        if not self.single_store:
            raise ValueError("Zing attention_view requires the single-store cache")
        if torch.is_grad_enabled():
            raise RuntimeError("Zing attention buffers require torch.no_grad()")
        self._check_append_layout(layer, k, v)
        chunks = self._history()
        rows = k.shape[0]
        if not chunks:
            out_k, out_v = k.new_empty(k.shape), v.new_empty(v.shape)
            write_kv(k, v, out_k, out_v)
            return out_k, out_v
        self._verify_committed_positions(chunks)
        total = self._length + rows
        keys, values = self._reserve_store(layer, total, k, v)
        out_k, out_v = keys[self._length : total], values[self._length : total]
        write_kv(k, v, out_k, out_v)
        runs = self._runs(chunks)
        if len(runs) == 1 and runs[0][1] == self._length:
            start = runs[0][0]
            return keys[start:total], values[start:total]
        runs.append([self._length, total])
        return (
            torch.cat([keys[a:b] for a, b in runs]),
            torch.cat([values[a:b] for a, b in runs]),
        )

    def raw_attention_view(
        self,
        layer,
        k,
        v,
        make_rope,
        rotate_into,
        write_current,
        *,
        rotate_batch=None,
        history_plan=None,
    ):
        """Attend from mapped K scratch plus the sole authoritative V store.

        Current K stays unrotated until write_current writes the attention tail.
        A successful clean forward commits its original K and borrows this V
        tail. History rotations are reused only while both raw K and effective
        coordinates match. Prefix/window changes invalidate that publication.
        """
        if not self.raw_store:
            raise ValueError("Zing mapped attention requires a raw K/V store")
        if torch.is_grad_enabled():
            raise RuntimeError("Zing attention buffers require torch.no_grad()")
        self._check_append_layout(layer, k, v)
        if history_plan is None:
            chunks = self._history()
        else:
            if (
                history_plan.owner is not self
                or history_plan.state != self._history_plan_state()
            ):
                raise ValueError(
                    "History RoPE plan belongs to another cache or forward state"
                )
            chunks = history_plan.chunks
            if (
                tuple(self._tensor_signature(c.positions) for c in chunks)
                != history_plan.position_signatures
            ):
                raise ValueError(
                    "History RoPE plan positions changed during the forward"
                )
        rows = k.shape[0]
        _, values = self._reserve_store(layer, self._length + rows, k, v)
        # Unpublish before writes; a failed rotation must never be reused.
        previous = self._rotated_store_signatures.pop(layer, ())
        if self.chunk_rope_reuse:
            return self._raw_chunk_attention_view(
                layer,
                k,
                v,
                chunks,
                previous,
                values,
                make_rope,
                rotate_into,
                write_current,
                rotate_batch,
            )
        if history_plan is None:
            geometry = self._history_geometry(chunks)
            history_rows = sum(c.positions.shape[0] for c in chunks)
        else:
            geometry = history_plan.geometry
            history_rows = history_plan.rows
        signatures = tuple(
            (
                position_key,
                self._tensor_signature(c.positions),
                self._tensor_signature(c.layers[layer][0]),
            )
            for c, (_, position_key) in zip(chunks, geometry)
        )
        common = 0
        for before, after in zip(previous, signatures):
            if before != after:
                break
            common += 1
        retained = sum(c.positions.shape[0] for c in chunks[:common])
        keys = self._rotated_stores.get(layer)
        capacity = values.shape[0]
        if self.pooled and keys is None:
            keys = self.storage[layer].rotated_k
            self._rotated_stores[layer] = keys
        elif keys is None or keys.shape[0] < capacity:
            self.stats["rotated_allocations"] += 1
            grown = k.new_empty((capacity, *k.shape[1:]))
            if keys is not None and retained:
                grown[:retained].copy_(keys[:retained])
            keys = grown
            self._rotated_stores[layer] = keys
        live_positions = {key for _, key in geometry}
        self._rope_coefficients = {
            key: value
            for key, value in self._rope_coefficients.items()
            if key[0] in live_positions
        }
        offset = retained
        if rotate_batch is not None:
            batch_inputs, batch_coefficients = [], []
        for chunk, (positions, position_key) in zip(chunks[common:], geometry[common:]):
            coefficient_key = (position_key, k.dtype)
            if coefficient_key not in self._rope_coefficients:
                self._rope_coefficients[coefficient_key] = make_rope(positions)
            end = offset + chunk.positions.shape[0]
            if rotate_batch is None:
                rotate_into(
                    chunk.layers[layer][0],
                    self._rope_coefficients[coefficient_key],
                    keys[offset:end],
                )
            else:
                batch_inputs.append(chunk.layers[layer][0])
                batch_coefficients.append(self._rope_coefficients[coefficient_key])
            offset = end
        if rotate_batch is not None and batch_inputs:
            rotate_batch(batch_inputs, batch_coefficients, keys[retained:history_rows])
        total = history_rows + rows
        current_values = values[self._length : self._length + rows]
        write_current(k, v, keys[history_rows:total], current_values)
        self._rotated_store_signatures[layer] = signatures
        runs = self._runs(chunks) if history_plan is None else history_plan.runs
        if not runs:
            return keys[:total], current_values
        if len(runs) == 1 and runs[0][1] == self._length:
            return keys[:total], values[runs[0][0] : self._length + rows]
        # Only the legacy selection-only frame API can leave gaps. The temporal
        # window compacts on eviction, so its V history is always a single view.
        return keys[:total], torch.cat(
            [values[a:b] for a, b in runs] + [current_values]
        )

    def _reserve_chunk_rotations(self, layer, k, capacity, ranges, reusable):
        """Bind a pool slot or grow scratch, copying only still-valid ranges."""
        keys = self._rotated_stores.get(layer)
        if self.pooled and keys is None:
            keys = self.storage[layer].rotated_k
            self._rotated_stores[layer] = keys
        elif keys is None or keys.shape[0] < capacity:
            self.stats["rotated_allocations"] += 1
            grown = k.new_empty((capacity, *k.shape[1:]))
            if keys is not None:
                for entry in ranges:
                    if entry in reusable:
                        start, end = entry[:2]
                        grown[start:end].copy_(keys[start:end])
            keys = grown
            self._rotated_stores[layer] = keys
        return keys

    @staticmethod
    def _chunk_rebuild_reasons(old, entry):
        """Explain a cache miss without changing the exact reuse predicate."""
        if old is None:
            return ("new_or_unpublished",)
        reasons = []
        if old[3] != entry[3]:
            reasons.append("coordinates")
        if old[:3] != entry[:3] or old[4][2] != entry[4][2]:
            reasons.append("storage_mapping")
        if old[4][1] != entry[4][1] or old[4][3:] != entry[4][3:]:
            reasons.append("raw_content_or_layout")
        return reasons

    def _raw_chunk_attention_view(
        self,
        layer,
        k,
        v,
        chunks,
        previous,
        values,
        make_rope,
        rotate_into,
        write_current,
        rotate_batch=None,
    ):
        ranges, offset = self._chunk_ranges(layer)
        plan = self._chunk_plan
        reusable = set(previous)
        previous_parts = {entry[4][0]: entry for entry in previous}
        accounting = dict(
            layer=layer, rebuilt_rows=0, reused_rows=0, coefficients_built=0, reasons={}
        )
        self.last_rope_reuse = accounting
        keys = self._reserve_chunk_rotations(
            layer, k, values.shape[0], ranges, reusable
        )
        live = {entry[5] for entry in plan}
        self._rope_coefficients = {
            key: value
            for key, value in self._rope_coefficients.items()
            if key[0] in live
        }
        batch_inputs, batch_coefficients = [], []
        batch_start = 0

        def flush_batch(end):
            if batch_inputs:
                rotate_batch(batch_inputs, batch_coefficients, keys[batch_start:end])
                self.rope_reuse_stats["rebuilt_rows"] += end - batch_start
                accounting["rebuilt_rows"] += end - batch_start
                batch_inputs.clear()
                batch_coefficients.clear()

        for (chunk, _, local_start, local_end, positions, key), entry in zip(
            plan, ranges
        ):
            start, end = entry[:2]
            # Exact match of the last successfully published entry: output
            # range, raw offset, coordinate token, and raw K part signature.
            if entry in reusable:
                flush_batch(start)
                self.rope_reuse_stats["reused_rows"] += end - start
                accounting["reused_rows"] += end - start
                continue
            for reason in self._chunk_rebuild_reasons(
                previous_parts.get(entry[4][0]), entry
            ):
                accounting["reasons"][reason] = (
                    accounting["reasons"].get(reason, 0) + end - start
                )
            coefficient_key = (key, k.dtype)
            if coefficient_key not in self._rope_coefficients:
                self._rope_coefficients[coefficient_key] = make_rope(positions)
                accounting["coefficients_built"] += 1
            raw = chunk.layers[layer][0][local_start:local_end]
            coefficients = self._rope_coefficients[coefficient_key]
            if rotate_batch is None:
                rotate_into(raw, coefficients, keys[start:end])
                self.rope_reuse_stats["rebuilt_rows"] += end - start
                accounting["rebuilt_rows"] += end - start
            else:
                # Group only adjacent misses: a valid middle span must never
                # be recomputed or overwritten by a contiguous batch output.
                if not batch_inputs:
                    batch_start = start
                batch_inputs.append(raw)
                batch_coefficients.append(coefficients)
        flush_batch(offset)
        total = offset + k.shape[0]
        current_values = values[self._length : self._length + k.shape[0]]
        write_current(k, v, keys[offset:total], current_values)
        # Publication follows every write, including the possibly failing tail.
        self._rotated_store_signatures[layer] = tuple(ranges)
        return keys[:total], self._chunk_value_view(chunks, values, current_values)

    def _chunk_ranges(self, layer):
        """Published-rotation entries of every history part, and their row total."""
        # Match physical output ranges as well as raw K and coordinate versions.
        # A surviving chunk shifted by compact must not read an old scratch view.
        if self._chunk_plan is None:
            if self.rope_position_mode != "absolute":
                raise ValueError(
                    "Zing relative RoPE requires the current attention map"
                )
            self._bind_chunk_positions(self.positions())
        ranges, offset = [], 0
        for chunk, identity, start, end, _, key in self._chunk_plan:
            signature = self._raw_chunk_signature(layer, chunk, identity, start, end)
            count = end - start
            ranges.append(
                (offset, offset + count, chunk.offset + start, key, signature)
            )
            offset += count
        return ranges, offset

    def reused_history_rows(self, rows):
        """History rows if every layer attends from published rotations alone.

        None when any layer would rotate history, or when the store layout
        differs from what a pooled, chunk-reusing raw store keeps fixed. Reads
        only host state; the raw-store observation it shares with
        raw_attention_view is idempotent.
        """
        if not (
            self.raw_store
            and self.pooled
            and self.chunk_rope_reuse
            and self._chunk_plan is not None
            and self._length + rows <= self.capacity
        ):
            return None
        history = None
        for layer in range(self.num_layers):
            if layer not in self._stores or layer not in self._rotated_stores:
                return None
            self._observe_raw_store(layer)
            ranges, offset = self._chunk_ranges(layer)
            previous = set(self._rotated_store_signatures.get(layer, ()))
            if any(entry not in previous for entry in ranges):
                return None
            history = offset
        return history

    def store_layout_signature(self):
        """Live rows, history row runs and pooled slot addresses a captured
        attention reads; only meaningful for a pooled store."""
        return (
            self._length,
            tuple(tuple(run) for run in self._runs(self._history())),
            tuple(
                (t.data_ptr(), tuple(t.shape), t.stride(), t.dtype, t.device)
                for store in self.storage
                for t in (store.k, store.v, store.rotated_k)
            ),
        )

    def replay_reused_attention(self):
        """Host effects of one raw_attention_view per layer that rotated nothing.

        A replayed denoise graph skips the Python of raw_attention_view, so
        publications, coefficient pruning and reuse accounting are applied
        here exactly as that call would for a fully reused history.
        """
        live = {entry[5] for entry in self._chunk_plan}
        for layer in range(self.num_layers):
            self._observe_raw_store(layer)
            self._rotated_store_signatures.pop(layer, None)
            ranges, offset = self._chunk_ranges(layer)
            self._rope_coefficients = {
                key: value
                for key, value in self._rope_coefficients.items()
                if key[0] in live
            }
            self.rope_reuse_stats["reused_rows"] += offset
            self.last_rope_reuse = dict(
                layer=layer,
                rebuilt_rows=0,
                reused_rows=offset,
                coefficients_built=0,
                reasons={},
            )
            self._rotated_store_signatures[layer] = tuple(ranges)

    def _chunk_value_view(self, chunks, values, current_values):
        """Use a contiguous V view; gather only legacy selections with gaps."""
        runs = self._runs(chunks)
        if not runs:
            return current_values
        if len(runs) == 1 and runs[0][1] == self._length:
            return values[runs[0][0] : self._length + current_values.shape[0]]
        return torch.cat([values[a:b] for a, b in runs] + [current_values])

    def stage_raw_k(self, layer, k):
        """Copy a clean forward's raw K into this layer's raw store scratch tail.

        Like the V tail written by raw_attention_view, the rows sit where the
        commit places them, so _store_chunk skips the copy. Call only after
        raw_attention_view reserved this layer for the current rows.
        """
        if not self.raw_store:
            raise ValueError("Zing raw K staging requires a raw K/V store")
        self._check_append_layout(layer, k, k)
        rows = k.shape[0]
        store = self._stores.get(layer)
        if store is None or store[0].shape[0] < self._length + rows:
            raise RuntimeError("Zing raw K staging requires a reserved store")
        tail = store[0][self._length : self._length + rows]
        self._observe_raw_store(layer)
        tail.copy_(k)
        self._mark_raw_store(layer)
        return tail

    def stage_prefix_kv(self, layer, k, v, head_start, heads):
        """Write one layer's local head shard of a replicated prefix in place.

        A replicated SP prefix computes every head on every rank. Only this
        rank's shard is written into the store tail, where the commit finds it
        and copies nothing, so no full-head copy outlives the layer.
        """
        if not self.raw_store:
            raise ValueError("Zing prefix staging requires a raw K/V store")
        k = k[:, head_start : head_start + heads]
        v = v[:, head_start : head_start + heads]
        rows = k.shape[0]
        keys, values = self._reserve_store(layer, self._length + rows, k, v)
        out_k = keys[self._length : self._length + rows]
        out_v = values[self._length : self._length + rows]
        out_k.copy_(k)
        out_v.copy_(v)
        self._mark_raw_store(layer)
        return out_k, out_v

    def _same_active_geometry(self, positions):
        """Compare with the active snapshot held on the caller's device."""
        reference = self._active_reference
        if reference is None or reference.device != positions.device:
            reference = self.active[1]
            positions = positions.to(reference.device)
        return torch.equal(reference, positions)

    def validate(self, positions, block_idx, phase, frames):
        if torch.is_grad_enabled():
            raise RuntimeError("Zing KV transactions require torch.no_grad()")
        if phase not in {"prefix", "source", "active", "final"}:
            raise ValueError("Invalid Zing KV transaction phase")
        blocks = torch.unique(block_idx)
        if blocks.numel() != 1 or positions.shape != (block_idx.numel(), 3):
            raise ValueError("A Zing KV transaction must contain one nonempty block")
        block = int(blocks.item())
        if phase == "prefix":
            if (
                block != 0
                or frames != 0
                or self.prefix is not None
                or self.last_block
                or self.active is not None
                or self.pending_source is not None
            ):
                raise ValueError(
                    "Zing prefix must be the first transaction, with block zero"
                )
        elif frames < 1:
            raise ValueError(
                "Zing source and target transactions require visual frames"
            )
        elif phase == "source":
            if (
                block != self.last_block + 1
                or block % 2 != 1
                or self.pending_source is not None
                or self.active is not None
            ):
                raise ValueError("Zing source must precede its target exactly once")
        else:
            if block != self.last_block + 2 or block % 2:
                raise ValueError("Zing target blocks must commit in consecutive order")
            if (
                self.pending_source is not None
                and self.pending_source.block + 1 != block
            ):
                raise ValueError("Zing target does not match its pending source")
            if self.active is not None and (
                self.active[0] != block or not self._same_active_geometry(positions)
            ):
                raise ValueError("Zing active target geometry changed before commit")
            if phase == "final" and self.active is None:
                raise ValueError("Zing final commit requires an active target")
        return block

    def _chunk_layers(self, offset, rows):
        if not self._stores or len(self._stores) < self.num_layers:
            return ()
        stores = tuple(self._stores[i] for i in range(self.num_layers))
        if self.pooled:
            return _StoreLayerViews(stores, offset, offset + rows)
        return tuple(
            (k[offset : offset + rows], v[offset : offset + rows]) for k, v in stores
        )

    def _store_chunk(self, positions, block, frames, layers):
        """Commit raw or post-RoPE K, reusing V already written into scratch."""
        offset = self._length
        rows = positions.shape[0]
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_sparse_attn import (
            is_three_tier_enabled,
        )

        has_compressed_kv = (
            getattr(self, "_compressed_kv_enabled", False) or is_three_tier_enabled()
        )
        bookkeeping_only = getattr(self, "_bookkeeping_only", False)
        if not has_compressed_kv:
            for layer, (k, v) in enumerate(layers):
                keys, values = self._reserve_store(layer, offset + rows, k, v)
                # A clean forward already wrote these rows into the scratch tail, at
                # the offset they commit to; copying them onto themselves is waste.
                # A store grown since then moves the destination, so the check is on
                # the address, not on the phase.
                if not bookkeeping_only:
                    for source, destination in (
                        (k, keys[offset : offset + rows]),
                        (v, values[offset : offset + rows]),
                    ):
                        if source.data_ptr() != destination.data_ptr():
                            destination.copy_(source)
                self._mark_raw_store(layer)
        self._length = offset + rows
        self._chunk_plan = None
        chunk = ZingKVChunk(positions, block, (), frames, offset)
        chunk.position_signature = self._tensor_signature(positions)
        if not has_compressed_kv:
            chunk.layers = self._chunk_layers(offset, rows)
        elif block == 0 and self.retain_prefix_layers:
            chunk.layers = (
                tuple(layers)
                if bookkeeping_only
                else tuple((k.detach().clone(), v.detach().clone()) for k, v in layers)
            )
            chunk.export_owned = not bookkeeping_only
        else:
            chunk.layers = ()

        if not getattr(self, "_bookkeeping_only", False) and layers and len(layers) > 0:
            if has_compressed_kv:
                for l_idx, layer_pair in enumerate(layers):
                    if not layer_pair or len(layer_pair) < 2:
                        continue
                    k_l, v_l = layer_pair[0], layer_pair[1]
                    store_l = (
                        self._compressed_kv_stores[l_idx]
                        if getattr(self, "_compressed_kv_stores", None) is not None
                        and l_idx < len(self._compressed_kv_stores)
                        and self._compressed_kv_stores[l_idx] is not None
                        else None
                    )
                    if store_l is None and hasattr(self, "get_compressed_kv_store"):
                        store_l = self.get_compressed_kv_store(
                            layer_idx=l_idx,
                            device=k_l.device,
                            num_heads=k_l.shape[1] if k_l.ndim >= 3 else 28,
                            head_dim=k_l.shape[2] if k_l.ndim >= 3 else 128,
                        )
                    if store_l is not None:
                        try:
                            # Prefix has its own namespace. An odd source block
                            # and its following even target belong to one chunk.
                            owner = -1 if block == 0 else (block - 1) // 2
                            store_l.append_chunk(k_l, v_l, chunk_id=owner)
                        except Exception:
                            # A partial per-layer write is not a committed KV
                            # state. Revoke it instead of advancing silently.
                            self.clear()
                            raise
        elif bookkeeping_only and has_compressed_kv:
            for store in self._compressed_kv_stores or ():
                if store is not None:
                    store.num_blocks += (rows + 63) // 64
                    if store.num_blocks > store.max_blocks:
                        raise ZingKVCapacityError(
                            "Compressed KV plan exceeds block capacity"
                        )
        if has_compressed_kv:
            self._compressed_part_layout += ((offset, rows),)
        return chunk

    def update(self, positions, block_idx, phase, frames, layers=None, validation=None):
        """Apply one transaction.

        ``validation`` optionally supplies host copies ``(positions, block_idx)``
        of the device tensors, so the phase checks never read the device.
        """
        checked = (positions, block_idx) if validation is None else validation
        block = self.validate(*checked, phase, frames)
        if phase == "active":
            if layers is not None:
                raise ValueError("Active Zing K/V must not be persisted")
            self.active = (block, positions.detach().clone())
            self._active_reference = (
                self.active[1] if validation is None else checked[0].detach().clone()
            )
            return
        if layers is None or len(layers) != self.num_layers:
            raise ValueError("Zing KV commit requires every transformer layer")
        layouts = []
        for k, v in layers:
            if (
                k.ndim != 3
                or k.shape != v.shape
                or k.shape[0] != positions.shape[0]
                or k.dtype != v.dtype
                or k.device != v.device
                or k.requires_grad
                or v.requires_grad
            ):
                raise ValueError(
                    "Zing K/V must be matching detached [rows, heads, dim] tensors"
                )
            layouts.append((tuple(k.shape[1:]), k.dtype, k.device))
        if self.layer_layouts is None:
            self.layer_layouts = tuple(layouts)
        elif tuple(layouts) != self.layer_layouts:
            raise ValueError("Zing K/V head-shard layout changed within a request")
        # Keep position snapshots versioned even inside inference_mode so
        # an in-place geometry edit invalidates derived RoPE and buffers.
        with torch.inference_mode(False), torch.no_grad():
            saved_positions = positions.detach().clone()
        # Source rows already sit immediately before their target's, so the
        # merge below only has to widen this chunk's view over the store.
        chunk = self._store_chunk(saved_positions, block, frames, layers)
        if validation is not None and checked[0].device.type == "cpu":
            with torch.inference_mode(False), torch.no_grad():
                chunk.host_positions = checked[0].detach().clone()
        if phase == "prefix":
            self.prefix = chunk
            stores = getattr(self, "_compressed_kv_stores", None)
            if stores:
                self._prefix_compressed_blocks = tuple(
                    store.num_blocks if store is not None else 0 for store in stores
                )
        elif phase == "source":
            self.pending_source = chunk
        else:
            if self.pending_source is not None:
                source = self.pending_source
                with torch.inference_mode(False), torch.no_grad():
                    source_positions, target_positions = (
                        source.positions,
                        chunk.positions,
                    )
                    if source_positions.device != target_positions.device:
                        # Graph planning keeps host metadata; a standalone eager
                        # final can supply device positions. Reuse guarded host
                        # copies instead of reading the device back at this edge.
                        if (
                            source.host_positions is not None
                            and chunk.host_positions is not None
                        ):
                            source_positions = source.host_positions
                            target_positions = chunk.host_positions
                        else:
                            target_positions = target_positions.to(
                                source_positions.device
                            )
                    combined_positions = torch.cat((source_positions, target_positions))
                rows = combined_positions.shape[0]
                merged = ZingKVChunk(
                    combined_positions, block, (), frames, source.offset
                )
                merged.rope_parts = self._parts(source) + self._parts(chunk)
                merged.position_signature = self._tensor_signature(combined_positions)
                if (
                    source.host_positions is not None
                    and chunk.host_positions is not None
                ):
                    with torch.inference_mode(False), torch.no_grad():
                        merged.host_positions = torch.cat(
                            (source.host_positions, chunk.host_positions)
                        )
                merged.layers = self._chunk_layers(source.offset, rows)
                chunk = merged
            self.committed.append(chunk)
            self.last_block = block
            self.pending_source = self.active = self._active_reference = None
            if not getattr(self, "_bookkeeping_only", False):
                self._stage_sink_refresh_signal_async()
