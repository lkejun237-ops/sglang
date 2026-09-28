# SPDX-License-Identifier: Apache-2.0
"""Request-owned KV transactions for Zing causal generation."""

import functools
import os
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
    ZingKVCapacityError,
    ZingLayerStore,
)
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy


def tensor_identity(value):
    """Object identity plus in-place version, when the tensor tracks one."""
    version = None if torch.is_inference(value) else value._version
    return id(value), version


@dataclass
class ZingKVChunk:
    positions: torch.Tensor
    block: int
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    frames: int = 0
    # Store-backed chunks borrow raw or post-RoPE K and the sole persistent V.
    # Only the absolute post-RoPE store pins a commit-time rotation signature.
    offset: int = -1
    position_signature: tuple | None = None
    identity: object = field(default_factory=object, repr=False)
    rope_parts: tuple = ()
    # CPU copy of positions; lets coordinate maps run without device reads.
    host_positions: torch.Tensor | None = None

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
        rope_position_mode: str = "absolute",
        rope_max_frame_gap: int = 12,
        storage: list[ZingLayerStore] | None = None,
    ):
        if num_layers < 1 or local_frames < -1 or sink_frames < 0:
            raise ValueError("Invalid Zing KV layer count or frame limits")
        if reserve_rows < 0:
            raise ValueError("Invalid Zing KV reserved row count")
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
        self.chunk_rope_reuse = os.environ.get("SGLANG_ZING_CHUNK_ROPE_REUSE", "1") == "1"
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
        self.storage = storage
        # Main-store allocations/growth and compactions, counted apart from
        # derived RoPE/position metadata, which is never pooled.
        self.stats = {
            "store_allocations": 0,
            "rotated_allocations": 0,
            "compactions": 0,
        }
        self.clear()

    @property
    def pooled(self):
        return self.storage is not None

    @property
    def capacity(self):
        return None if self.storage is None else self.storage[0].rows

    def clear(self):
        projection_cache = getattr(self, "adaln_projection_cache", None)
        if projection_cache is not None:
            projection_cache.clear()
        self.adaln_projection_cache = None
        self.prefix = None
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
        self._rotated_stores = {}
        self._rotated_store_signatures = {}
        self._effective_history_positions = None
        self._attention_rope_policy = None
        self._rope_epoch = 0
        self._chunk_geometry = {}
        self._chunk_plan = None
        self.rope_reuse_stats = dict(rebuilt_rows=0, reused_rows=0, map_syncs=0)
        self.last_rope_reuse = None

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
            if not self.pooled:
                for pair in chunk.layers:
                    yield from pair
        if not self.pooled:
            for pair in self._stores.values():
                yield from pair
            yield from self._rotated_stores.values()
        yield from self._rope_coefficients.values()
        if self._effective_history_positions is not None:
            yield self._effective_history_positions
        for _, positions, _ in self._chunk_geometry.values():
            yield positions
        if self.active is not None:
            yield self.active[1]

    def prepare_chunk(self, target_block):
        """Evict history before source execution; the caller closes on failure.

        One committed entry owns source + target/audio. Keep pre-RoPE K and
        absolute checkpoint positions: eviction never renumbers a retained key.
        Eviction invalidates derived maps and releases removed chunk references.
        A raw store compacts retained rows and reuses the freed tail capacity.
        """
        if self.active is not None or self.pending_source is not None:
            raise ValueError("Zing window selection requires a completed boundary")
        if target_block != self.last_block + 2:
            raise ValueError("Zing window selection requires the next target block")
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

    def _compact_store(self):
        """Move live raw K/V to the front, using at most one chunk temporarily."""
        chunks = self._stored_chunks()
        runs = _compaction_runs([(c.offset, c.positions.shape[0]) for c in chunks])
        tensors = [tensor for pair in self._stores.values() for tensor in pair]
        for layer in self._stores:
            self._observe_raw_store(layer)
        if runs and _fused_compaction_rejection(tensors, runs) is None:
            _fused_move_rows(tensors, runs)
        elif runs:
            # Per chunk, a move is at most a few pieces of its shift.
            for keys, values in self._stores.values():
                destination = 0
                for chunk in chunks:
                    rows, source = chunk.positions.shape[0], chunk.offset
                    if source != destination:
                        for tensor in (keys, values):
                            _move_rows(tensor, destination, source, rows)
                    destination += rows
        for layer in self._stores:
            self._mark_raw_store(layer)
        offset = 0
        for chunk in chunks:
            chunk.offset = offset
            offset += chunk.positions.shape[0]
        self._length = offset
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
        value = chunk.layers[layer][0]
        store = self._stores[layer][0]
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
        for chunk in self._stored_chunks():
            start, stop = chunk.offset, chunk.offset + chunk.positions.shape[0]
            layers = (
                list(chunk.layers)
                if len(chunk.layers) == self.num_layers
                else [None] * self.num_layers
            )
            if self.chunk_rope_reuse and layers[layer] is not None:
                old_k, old_v = layers[layer]
                if (
                    old_k.data_ptr() == store[0][start:stop].data_ptr()
                    and old_v.data_ptr() == store[1][start:stop].data_ptr()
                    and old_k.shape == store[0][start:stop].shape
                ):
                    continue
            layers[layer] = (store[0][start:stop], store[1][start:stop])
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
                raise ValueError("History RoPE plan positions changed during the forward")
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
        # Match physical output ranges as well as raw K and coordinate versions.
        # A surviving chunk shifted by compact must not read an old scratch view.
        if self._chunk_plan is None:
            if self.rope_position_mode != "absolute":
                raise ValueError(
                    "Zing relative RoPE requires the current attention map"
                )
            self._bind_chunk_positions(self.positions())
        plan = self._chunk_plan
        ranges, offset = [], 0
        for chunk, identity, start, end, _, key in plan:
            signature = self._raw_chunk_signature(layer, chunk, identity, start, end)
            count = end - start
            ranges.append(
                (offset, offset + count, chunk.offset + start, key, signature)
            )
            offset += count
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

    def _replace_prefix_in_place(self, prefix, old_rows, new_rows):
        """Shift committed rows within each fixed slot, then write the prefix.

        Chunks move one at a time in the direction that never overwrites an
        unread source. A chunk overlapping its destination is staged through
        the slot's rotated-K buffer, which replace_prefix has just invalidated
        and the next attention rebuilds from row 0, so the move allocates
        nothing outside the slot.
        """
        total = new_rows + self._length - old_rows
        if total > self.capacity:
            raise ZingKVCapacityError(
                f"Zing pooled KV needs {total} rows but the slot holds {self.capacity}"
            )
        shift = new_rows - old_rows
        moving = [c for c in self._stored_chunks() if c is not self.prefix]
        if shift > 0:
            moving.reverse()
        for layer in range(self.num_layers):
            self._observe_raw_store(layer)
            slot = self.storage[layer]
            if shift:
                for chunk in moving:
                    rows, start = chunk.positions.shape[0], chunk.offset
                    for tensor in (slot.k, slot.v):
                        source = tensor[start : start + rows]
                        if abs(shift) < rows:
                            source = slot.rotated_k[:rows].copy_(source)
                        tensor[start + shift : start + shift + rows].copy_(source)
            if prefix is not None:
                k, v = prefix.layers[layer]
                if (tuple(k.shape[1:]), k.dtype, k.device) != slot.layout:
                    raise ValueError("Zing prefix layout does not match the pool slot")
                slot.k[:new_rows].copy_(k)
                slot.v[:new_rows].copy_(v)
            self._stores[layer] = (slot.k, slot.v)
            self._mark_raw_store(layer)
        return total

    def replace_prefix(self, prefix):
        """Adopt a rebuilt prefix, keeping committed target rows and positions.

        Store-backed prefixes are copied and the retained tail shifted, allowing
        the donor cache to be cleared. Relative rotations must be rebuilt: a new
        prefix can change the temporal anchor even with identical raw positions.
        """
        self._invalidate_rope(preserve_chunks=True)
        if self.active is not None or self.pending_source is not None:
            raise ValueError("Zing prefix replacement requires a committed boundary")
        old_rows = 0 if self.prefix is None else self.prefix.positions.shape[0]
        new_rows = 0 if prefix is None else prefix.positions.shape[0]
        tail = self._length - old_rows
        total = new_rows + tail
        for layer in () if self.pooled else range(self.num_layers):
            store = self._stores.get(layer)
            self._observe_raw_store(layer)
            source = None if prefix is None else prefix.layers[layer]
            if new_rows == old_rows and store is not None:
                if source is not None:
                    store[0][:new_rows].copy_(source[0])
                    store[1][:new_rows].copy_(source[1])
                self._mark_raw_store(layer)
                continue
            if source is None and store is None:
                continue
            # Shifting within one buffer would overlap; rebuild this layer alone.
            template = source if source is not None else store
            capacity = max(total, self.reserve_rows)
            if store is not None:
                capacity = max(capacity, store[0].shape[0])
            with (
                torch.inference_mode(False)
                if self.chunk_rope_reuse and self.raw_store
                else nullcontext()
            ):
                keys = template[0].new_empty((capacity, *template[0].shape[1:]))
            values = template[1].new_empty((capacity, *template[1].shape[1:]))
            if source is not None:
                keys[:new_rows].copy_(source[0])
                values[:new_rows].copy_(source[1])
            if tail and store is not None:
                keys[new_rows:total].copy_(store[0][old_rows : self._length])
                values[new_rows:total].copy_(store[1][old_rows : self._length])
            self._stores[layer] = (keys, values)
            self._mark_raw_store(layer)
        if self.pooled:
            total = self._replace_prefix_in_place(prefix, old_rows, new_rows)
        shift = new_rows - old_rows
        for chunk in self._stored_chunks():
            if chunk is not self.prefix:
                chunk.offset += shift
        self._length = total
        if prefix is not None:
            with torch.inference_mode(False), torch.no_grad():
                positions = prefix.positions.detach().clone()
                host = (
                    None
                    if prefix.host_positions is None
                    else prefix.host_positions.detach().clone()
                )
            prefix = ZingKVChunk(positions, prefix.block, (), prefix.frames, 0)
            prefix.position_signature = self._tensor_signature(positions)
            prefix.host_positions = host
        self.prefix = prefix
        for layer in range(self.num_layers):
            self._rebind(layer)

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

    def _store_chunk(self, positions, block, frames, layers):
        """Commit raw or post-RoPE K, reusing V already written into scratch."""
        offset = self._length
        rows = positions.shape[0]
        for layer, (k, v) in enumerate(layers):
            keys, values = self._reserve_store(layer, offset + rows, k, v)
            # A clean forward already wrote these rows into the scratch tail, at
            # the offset they commit to; copying them onto themselves is waste.
            # A store grown since then moves the destination, so the check is on
            # the address, not on the phase.
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
        chunk.layers = tuple(
            (
                self._stores[layer][0][offset : offset + rows],
                self._stores[layer][1][offset : offset + rows],
            )
            for layer in range(self.num_layers)
        )
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
        elif phase == "source":
            self.pending_source = chunk
        else:
            if self.pending_source is not None:
                source = self.pending_source
                with torch.inference_mode(False), torch.no_grad():
                    combined_positions = torch.cat((source.positions, chunk.positions))
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
                merged.layers = tuple(
                    (
                        self._stores[layer][0][source.offset : source.offset + rows],
                        self._stores[layer][1][source.offset : source.offset + rows],
                    )
                    for layer in range(self.num_layers)
                )
                chunk = merged
            self.committed.append(chunk)
            self.last_block = block
            self.pending_source = self.active = self._active_reference = None
