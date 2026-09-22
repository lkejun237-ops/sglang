# SPDX-License-Identifier: Apache-2.0
"""Request-owned KV transactions for Zing causal generation."""

from dataclasses import dataclass

import torch

from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy


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

    @property
    def temporal_chunk_id(self):
        return self.block // 2 - 1


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
        # A finite request can be much longer than its live attention window.
        # Never reserve its full packed length for a window-bounded store.
        self.reserve_rows = (
            reserve_rows if self.window_policy.window_chunks is None else 0
        )
        self.clear()

    def clear(self):
        self.prefix = None
        self.pending_source = None
        self.committed = []
        self.active = None
        self.last_block = 0
        self.layer_layouts = None
        self._rope_coefficients = {}
        self._model_signature = None
        self._stores = {}
        self._length = 0
        self._rotated_stores = {}
        self._rotated_store_signatures = {}
        self._effective_history_positions = None
        self._attention_rope_policy = None
        self._rope_epoch = 0

    def _invalidate_rope(self):
        """Drop publications and chunk references, retaining reusable K capacity."""
        self._rope_coefficients.clear()
        self._rotated_store_signatures.clear()
        self._effective_history_positions = None
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
        if self.rope_position_mode == "absolute":
            return
        before = self._effective_history_positions
        if before is None and history_positions is None:
            return
        if (
            before is not None
            and history_positions is not None
            and before.device == history_positions.device
            and before.dtype == history_positions.dtype
            and torch.equal(before, history_positions)
        ):
            return
        self._invalidate_rope()
        if history_positions is not None:
            self._effective_history_positions = history_positions.detach().clone()

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

    def owned_tensors(self):
        """Include derived storage in session memory/release accounting."""
        for chunk in self._stored_chunks():
            yield chunk.positions
            for pair in chunk.layers:
                yield from pair
        for pair in self._stores.values():
            yield from pair
        yield from self._rotated_stores.values()
        yield from self._rope_coefficients.values()
        if self._effective_history_positions is not None:
            yield self._effective_history_positions
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
            self._invalidate_rope()
            self._compact_store()

    def _compact_store(self):
        """Move live raw K/V to the front, using at most one chunk temporarily."""
        chunks = self._stored_chunks()
        for keys, values in self._stores.values():
            destination = 0
            for chunk in chunks:
                rows, source = chunk.positions.shape[0], chunk.offset
                if source != destination:
                    # Left shifts may overlap within a chunk; clone only that
                    # source slice, never the whole multi-layer cache.
                    for tensor in (keys, values):
                        tensor[destination : destination + rows].copy_(
                            tensor[source : source + rows].clone()
                        )
                destination += rows
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

    def attention_positions(self, current, *, prefix=False):
        """Rebuild RoPE coordinates only; cached positions/provenance stay absolute.

        Unlike MinWM token order, Zing repeats time across source, target and
        stereo audio. Compress sorted unique media timestamps together, so equal
        timestamps share a coordinate regardless of packed modality/role order.
        Prefix/reference coordinates are unchanged. Gap units are checkpoint
        temporal coordinates (not pixels, latent frames or temporal chunk IDs).
        """
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
        if store is not None and store[0].shape[0] >= needed:
            return store
        capacity = max(needed, self.reserve_rows)
        if store is not None and not self.reserve_rows:
            capacity = max(capacity, 2 * store[0].shape[0])
        keys = k.new_empty((capacity, *k.shape[1:]))
        values = v.new_empty((capacity, *v.shape[1:]))
        if store is not None and self._length:
            keys[: self._length].copy_(store[0][: self._length])
            values[: self._length].copy_(store[1][: self._length])
        self._stores[layer] = (keys, values)
        # Views of the old buffer would keep it alive and read stale rows.
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

    def raw_attention_view(self, layer, k, v, make_rope, rotate_into, write_current):
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
        chunks = self._history()
        geometry = self._history_geometry(chunks)
        history_rows = sum(c.positions.shape[0] for c in chunks)
        rows = k.shape[0]
        _, values = self._reserve_store(layer, self._length + rows, k, v)
        signatures = tuple(
            (
                position_key,
                self._tensor_signature(c.positions),
                self._tensor_signature(c.layers[layer][0]),
            )
            for c, (_, position_key) in zip(chunks, geometry)
        )
        # Unpublish before writes; a failed rotation must never be reused.
        previous = self._rotated_store_signatures.pop(layer, ())
        common = 0
        for before, after in zip(previous, signatures):
            if before != after:
                break
            common += 1
        retained = sum(c.positions.shape[0] for c in chunks[:common])
        keys = self._rotated_stores.get(layer)
        capacity = values.shape[0]
        if keys is None or keys.shape[0] < capacity:
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
        for chunk, (positions, position_key) in zip(chunks[common:], geometry[common:]):
            coefficient_key = (position_key, k.dtype)
            if coefficient_key not in self._rope_coefficients:
                self._rope_coefficients[coefficient_key] = make_rope(positions)
            end = offset + chunk.positions.shape[0]
            rotate_into(
                chunk.layers[layer][0],
                self._rope_coefficients[coefficient_key],
                keys[offset:end],
            )
            offset = end
        total = history_rows + rows
        current_values = values[self._length : self._length + rows]
        write_current(k, v, keys[history_rows:total], current_values)
        self._rotated_store_signatures[layer] = signatures
        runs = self._runs(chunks)
        if not runs:
            return keys[:total], current_values
        if len(runs) == 1 and runs[0][1] == self._length:
            return keys[:total], values[runs[0][0] : self._length + rows]
        # Only the legacy selection-only frame API can leave gaps. The temporal
        # window compacts on eviction, so its V history is always a single view.
        return keys[:total], torch.cat(
            [values[a:b] for a, b in runs] + [current_values]
        )

    def replace_prefix(self, prefix):
        """Adopt a rebuilt prefix, keeping committed target rows and positions.

        Store-backed prefixes are copied and the retained tail shifted, allowing
        the donor cache to be cleared. Relative rotations must be rebuilt: a new
        prefix can change the temporal anchor even with identical raw positions.
        """
        self._invalidate_rope()
        if self.active is not None or self.pending_source is not None:
            raise ValueError("Zing prefix replacement requires a committed boundary")
        old_rows = 0 if self.prefix is None else self.prefix.positions.shape[0]
        new_rows = 0 if prefix is None else prefix.positions.shape[0]
        tail = self._length - old_rows
        total = new_rows + tail
        for layer in range(self.num_layers):
            store = self._stores.get(layer)
            source = None if prefix is None else prefix.layers[layer]
            if new_rows == old_rows and store is not None:
                if source is not None:
                    store[0][:new_rows].copy_(source[0])
                    store[1][:new_rows].copy_(source[1])
                continue
            if source is None and store is None:
                continue
            # Shifting within one buffer would overlap; rebuild this layer alone.
            template = source if source is not None else store
            capacity = max(total, self.reserve_rows)
            if store is not None:
                capacity = max(capacity, store[0].shape[0])
            keys = template[0].new_empty((capacity, *template[0].shape[1:]))
            values = template[1].new_empty((capacity, *template[1].shape[1:]))
            if source is not None:
                keys[:new_rows].copy_(source[0])
                values[:new_rows].copy_(source[1])
            if tail and store is not None:
                keys[new_rows:total].copy_(store[0][old_rows : self._length])
                values[new_rows:total].copy_(store[1][old_rows : self._length])
            self._stores[layer] = (keys, values)
        shift = new_rows - old_rows
        for chunk in self._stored_chunks():
            if chunk is not self.prefix:
                chunk.offset += shift
        self._length = total
        if prefix is not None:
            with torch.inference_mode(False), torch.no_grad():
                positions = prefix.positions.detach().clone()
            prefix = ZingKVChunk(positions, prefix.block, (), prefix.frames, 0)
            prefix.position_signature = self._tensor_signature(positions)
        self.prefix = prefix
        for layer in range(self.num_layers):
            self._rebind(layer)

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
                self.active[0] != block or not torch.equal(self.active[1], positions)
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
        self._length = offset + rows
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

    def update(self, positions, block_idx, phase, frames, layers=None):
        block = self.validate(positions, block_idx, phase, frames)
        if phase == "active":
            if layers is not None:
                raise ValueError("Active Zing K/V must not be persisted")
            self.active = (block, positions.detach().clone())
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
                merged.position_signature = self._tensor_signature(
                    combined_positions
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
            self.pending_source = self.active = None
