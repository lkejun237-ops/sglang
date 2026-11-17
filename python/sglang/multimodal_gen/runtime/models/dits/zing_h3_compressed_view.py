# SPDX-License-Identifier: Apache-2.0
"""Host layouts and bounded per-layer views for compressed Zing histories.

Persistent K is unrotated, matching the Block0 payload. Attention applies the
current complete position map to a temporary layer view before FP8 packing.
No whole-model raw history is allocated or retained. Temporary layer work and
its performance are deliberately separate from resident-pool accounting.
"""

from dataclasses import dataclass

import torch

LAYOUT_VERSION = "zing-compressed-kv-v2"
BLOCK = 64


@dataclass(frozen=True)
class TokenLayout:
    gather: tuple[int, ...]
    inverse: tuple[int, ...]
    valid_lengths: tuple[int, ...]
    raw_ends: tuple[int, ...]
    padded_ends: tuple[int, ...]

    @property
    def extent(self):
        return len(self.gather)

    @property
    def compact_gather(self):
        return tuple(
            self.gather[b * BLOCK + r]
            for b, count in enumerate(self.valid_lengths)
            for r in range(count)
        )

    def padded_end(self, raw_end):
        if raw_end == 0:
            return 0
        try:
            return self.padded_ends[self.raw_ends.index(raw_end)]
        except ValueError as error:
            raise ValueError(
                "Attention boundary splits a compressed KV part"
            ) from error


def _tiles(positions, prefix):
    """Return raw-row indices per block; never infer geometry on the device."""
    if positions.device.type != "cpu":
        raise ValueError("Compressed layout requires host positions")
    if prefix:
        return [
            list(range(i, min(i + BLOCK, len(positions))))
            for i in range(0, len(positions), BLOCK)
        ]
    from .zing_h3_sparse_attn import build_frame_2d_square_partition

    values = positions.tolist()
    result, start = [], 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end][0] == values[start][0]:
            end += 1
        coordinates = [(v[1], v[2]) for v in values[start:end]]
        hs, ws = (
            sorted({p[0] for p in coordinates}),
            sorted({p[1] for p in coordinates}),
        )
        # Degenerate/non-visual rows retain linear packing. Real visual grids
        # use the existing quadrant-major 2D tile builder.
        rectangular = min(len(hs), len(ws)) >= 2 and len(set(coordinates)) == len(
            coordinates
        ) == len(hs) * len(ws)
        if rectangular:
            lookup = {p: start + i for i, p in enumerate(coordinates)}
            raster = [lookup[(h, w)] for h in hs for w in ws]
            partition = build_frame_2d_square_partition(len(hs), len(ws), device="cpu")
            order = [raster[i] for i in partition.perm.tolist()]
            cursor = 0
            for count in partition.block_valid_len.tolist():
                result.append(order[cursor : cursor + count])
                cursor += count
        else:
            result.extend(
                list(range(i, min(i + BLOCK, end))) for i in range(start, end, BLOCK)
            )
        start = end
    return result


def token_layout(parts):
    """Each part is (host positions, is_static_prefix)."""
    gather, inverse, lengths, raw_ends, padded_ends = [], [], [], [], []
    raw = 0
    for positions, prefix in parts:
        if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
            raise ValueError("Compressed KV parts need nonempty [rows,3] positions")
        inverse.extend([-1] * len(positions))
        for indices in _tiles(positions, prefix):
            if not 0 < len(indices) <= BLOCK:
                raise ValueError("Invalid compressed tile length")
            begin = len(gather)
            for i, source in enumerate(indices):
                if inverse[raw + source] != -1:
                    raise ValueError("Compressed token mapping is not injective")
                inverse[raw + source] = begin + i
            gather.extend(raw + i for i in indices)
            gather.extend([0] * (BLOCK - len(indices)))
            lengths.append(len(indices))
        raw += len(positions)
        raw_ends.append(raw)
        padded_ends.append(len(gather))
    if any(i < 0 for i in inverse):
        raise ValueError("Compressed token mapping does not cover every row")
    return TokenLayout(
        tuple(gather),
        tuple(inverse),
        tuple(lengths),
        tuple(raw_ends),
        tuple(padded_ends),
    )


def resident_rows(parts):
    """Map compact raw rows to resident chunk-aligned FP8 block addresses."""
    indices, block = [], 0
    for count in parts:
        if count <= 0:
            raise ValueError("Resident KV part must be nonempty")
        indices.extend(block * BLOCK + i for i in range(count))
        block += (count + BLOCK - 1) // BLOCK
    return tuple(indices), block


def dequantize_layer(store, rows, blocks, *, dtype):
    """Only one layer is materialized; padded rows never enter its output."""
    if blocks == 0:
        shape = (0, store.num_heads, store.head_dim)
        empty = torch.empty(shape, dtype=dtype, device=store.device)
        return empty, empty
    if not 0 < blocks <= store.max_blocks:
        raise ValueError("Compressed history exceeds its allocation")
    # num_blocks is host publication state and can lag an earlier forward in
    # the same graph. The immutable forward plan, not that counter, bounds reads.
    k = (
        store.k_res_fp8[:blocks].float()
        * store.k_res_scale[:, :blocks].T[:, :, None, None]
    )
    k = k + store.k_blk_mean[:, :blocks].transpose(0, 1)[:, :, None, :].float()
    v = (
        store.v_res_fp8[:blocks].float()
        * store.v_res_scale[:, :blocks].T[:, :, None, None]
    )
    v = v + store.v_blk_mean[:, :blocks].transpose(0, 1)[:, :, :, None].float()
    k = k.permute(0, 2, 1, 3).reshape(-1, store.num_heads, store.head_dim)
    v = v.permute(0, 3, 1, 2).reshape(-1, store.num_heads, store.head_dim)
    return k.index_select(0, rows).to(dtype), v.index_select(0, rows).to(dtype)


def pad_rows(value, indices, valid):
    return torch.where(valid[:, None, None], value.index_select(0, indices), 0)


def pack_attention(k, v, layout, indices, valid, compact_indices, valid_lengths=None):
    """Create a complete layer view, including every current noisy row.

    Values are not published into the resident cache. A subsequent layer can
    reuse the temporary allocations; retained session memory remains FP8.
    """
    from .zing_h3_sparse_attn import ZingCompressedKVStore

    if k.shape != v.shape or k.shape[0] != len(layout.inverse):
        raise ValueError("Attention KV does not match its frozen token layout")
    padded_k, padded_v = pad_rows(k, indices, valid), pad_rows(v, indices, valid)
    # append_chunk accepts compact tile order, with explicit partial lengths.
    packed = ZingCompressedKVStore(
        len(layout.valid_lengths), k.shape[1], k.shape[2], k.device
    )
    packed.append_chunk(
        k.index_select(0, compact_indices),
        v.index_select(0, compact_indices),
        block_valid_lengths=layout.valid_lengths
        if valid_lengths is None
        else valid_lengths,
    )
    return padded_k, padded_v, packed.as_kernel_dict()


def configure_plan(plan, cache, positions, blocks):
    """Freeze all host decisions before execution/capture; never write the pool."""
    from .zing_h3_cache import _host_to_device

    device = plan.stores[0][0].device
    history_parts, history_sizes, offset = [], [], 0
    for chunk in cache._history():
        for _, count in cache._parts(chunk):
            history_parts.append(
                (plan.history_positions[offset : offset + count], chunk is cache.prefix)
            )
            history_sizes.append(count)
            offset += count
    history_layout = token_layout(history_parts)
    if blocks is None:
        blocks = torch.full(
            (len(positions),),
            0 if plan.prefix else cache.last_block + 2,
            dtype=torch.long,
        )
    if positions.device.type != "cpu" or blocks.device.type != "cpu":
        raise ValueError("Compressed forward planning requires host geometry")
    block_ids = blocks.tolist()
    boundaries = sorted(
        {
            0,
            *plan.ends,
            *[i for i in range(1, len(block_ids)) if block_ids[i] != block_ids[i - 1]],
        }
    )
    current_parts, owners, clean_parts = [], [], []
    _, base_block = resident_rows(history_sizes)
    destination = base_block
    for start, stop in zip(boundaries, boundaries[1:]):
        block = block_ids[start]
        current_parts.append((positions[start:stop], block == 0))
        if start < plan.clean_rows:
            if stop > plan.clean_rows:
                raise ValueError("Compressed clean boundary splits a KV part")
            clean_parts.append((start, stop, destination, len(owners)))
            owners.append(-1 if block == 0 else (block - 1) // 2)
            destination += (stop - start + BLOCK - 1) // BLOCK
    if destination > plan._compressed_kv_stores[0].max_blocks:
        raise ValueError("Compressed clean writes exceed resident capacity")
    plan.compressed_history_rows = offset
    plan.compressed_history_blocks = base_block
    plan.compressed_clean_parts = tuple(clean_parts)
    plan.compressed_attention_layout = token_layout(history_parts + current_parts)
    plan.compressed_query_layout = token_layout(current_parts)
    plan.attention_write_start = history_layout.extent
    plan.attention_ends = tuple(
        plan.compressed_query_layout.padded_end(end) for end in plan.ends
    )
    plan.compressed_prefix_rows = (
        cache.prefix.positions.shape[0]
        if cache.prefix
        else sum(
            stop - start
            for start, stop in zip(boundaries, boundaries[1:])
            if block_ids[start] == 0
        )
    )
    plan.compressed_prefix_blocks = (plan.compressed_prefix_rows + BLOCK - 1) // BLOCK
    plan.compressed_signature = (
        LAYOUT_VERSION,
        tuple(history_sizes),
        tuple(boundaries),
        plan.compressed_attention_layout.valid_lengths,
        plan.compressed_query_layout.valid_lengths,
        plan.compressed_attention_layout.gather,
    )
    plan.tensors["compressed_history_rows"] = _host_to_device(
        torch.tensor(resident_rows(history_sizes)[0], dtype=torch.long), device
    )
    plan.tensors["compressed_owners"] = _host_to_device(
        torch.tensor(owners, dtype=torch.int32), device
    )
    if offset:
        plan.tensors["compressed_history_positions"] = _host_to_device(
            plan.history_positions, device
        )
    for name, layout in [
        ("compressed_keys", plan.compressed_attention_layout),
        ("compressed_queries", plan.compressed_query_layout),
    ]:
        values = {
            "gather": torch.tensor(layout.gather, dtype=torch.long),
            "inverse": torch.tensor(layout.inverse, dtype=torch.long),
            "compact": torch.tensor(layout.compact_gather, dtype=torch.long),
            "lengths": torch.tensor(layout.valid_lengths, dtype=torch.int32),
            "valid": torch.tensor(
                [r < n for n in layout.valid_lengths for r in range(BLOCK)]
            ),
        }
        for key, value in values.items():
            plan.tensors[name + "_" + key] = _host_to_device(value, device)
    prefix_rows = sum(
        stop - start
        for start, stop in zip(boundaries, boundaries[1:])
        if block_ids[start] == 0
    )
    plan.prefix_export_layers = None
    if prefix_rows and cache.retain_prefix_layers:
        # Export boundaries execute eagerly: these owned buffers are never
        # hidden allocations captured by a reusable CUDA graph.
        plan.prefix_export_layers = tuple(
            (torch.empty_like(k[:prefix_rows]), torch.empty_like(v[:prefix_rows]))
            for k, v in plan.stores
        )


def forward_view(plan, layer, q, history_rope, tensors):
    from .zing_h3_rope import apply_history_rope_into

    store = plan._compressed_kv_stores[layer]
    history_k, history_v = dequantize_layer(
        store,
        tensors["compressed_history_rows"],
        plan.compressed_history_blocks,
        dtype=q.dtype,
    )
    if plan.compressed_history_rows:
        if history_rope is None:
            raise ValueError("Compressed history has no current RoPE map")
        rotated = torch.empty_like(history_k)
        apply_history_rope_into(history_k, history_rope, rotated)
        history_k = rotated
    raw_k = torch.cat((history_k, plan.rotated[layer][: plan.rows]))
    raw_v = torch.cat((history_v, plan.stores[layer][1][: plan.rows]))
    keys, values, compressed = pack_attention(
        raw_k,
        raw_v,
        plan.compressed_attention_layout,
        tensors["compressed_keys_gather"],
        tensors["compressed_keys_valid"],
        tensors["compressed_keys_compact"],
        tensors["compressed_keys_lengths"],
    )
    queries = pad_rows(
        q, tensors["compressed_queries_gather"], tensors["compressed_queries_valid"]
    )
    return queries, keys, values, compressed, raw_k, raw_v


def commit_layer(plan, layer, tensors):
    """Device-only fixed writes; publication of the block count is host-owned."""
    k, v = plan.stores[layer]
    store = plan._compressed_kv_stores[layer]
    for start, stop, destination, index in plan.compressed_clean_parts:
        store.append_chunk(
            k[start:stop],
            v[start:stop],
            chunk_id=tensors["compressed_owners"][index],
            block_start=destination,
            publish=False,
        )
    if plan.prefix_export_layers is not None:
        out_k, out_v = plan.prefix_export_layers[layer]
        out_k.copy_(k[: out_k.shape[0]])
        out_v.copy_(v[: out_v.shape[0]])


def branch_spec(plan, spec):
    """Translate logical row boundaries to the padded, frame-local tile space."""
    from dataclasses import replace

    keys, queries = plan.compressed_attention_layout, plan.compressed_query_layout
    end = keys.padded_end
    q_start, q_stop = queries.padded_end(spec.q_start), queries.padded_end(spec.q_stop)
    current, total = end(spec.s_cur_chunk), end(spec.L_k)
    kinds = None
    if spec.blk_kind_sig is not None:
        from collections import Counter

        from .zing_h3_sparse_attn import _find_contiguous_spans

        _, references, sinks, sources = spec.blk_kind_sig
        sink_spans = (
            () if not sinks else sinks if isinstance(sinks[0], tuple) else (sinks,)
        )
        mapped = []
        for block in range(current // BLOCK):
            # Preserve the existing router's classification when a 2D tile
            # covers several old linear blocks. Padding never votes.
            original = Counter(
                row // BLOCK
                for row in keys.gather[
                    block * BLOCK : block * BLOCK + keys.valid_lengths[block]
                ]
            )
            labels = Counter()
            for old, weight in original.items():
                label = (
                    "sink"
                    if any(a <= old < b for a, b in sink_spans)
                    else "source"
                    if any(a <= old < b for a, b in sources)
                    else "reference"
                    if any(a <= old < b for a, b in references)
                    else "other"
                )
                labels[label] += weight
            mapped.append(labels.most_common(1)[0][0])

        def spans(name):
            return tuple(_find_contiguous_spans([k == name for k in mapped]))

        kinds = (
            current // BLOCK,
            spans("reference"),
            spans("sink") or None,
            spans("source"),
        )
    # The router consumes a fresh view on every forward. No mask or compressed
    # payload from an earlier denoising step is reused across a window shift.
    return replace(
        spec,
        q_start=q_start,
        q_stop=q_stop,
        L_q=q_stop - q_start,
        L_k=total,
        s_cur_chunk=current,
        M_q=(q_stop - q_start) // BLOCK,
        M_k=total // BLOCK,
        M_cand=current // BLOCK,
        M_cur=(total - current) // BLOCK,
        M_prefix=min(plan.compressed_prefix_blocks, current // BLOCK),
        companion_step0_spec=None,
        blk_kind_sig=kinds,
    )


def attend(plan, layer, q, history_rope, tensors, scale, *, head_start=0):
    """Use the existing Refine/Skim/Skip router on a complete current view.

    Prefix/history reconstruction preserves the canonical dense operation.
    Sparse source/target branches never fall back to attention over raw scratch.
    Fresh routing costs more than mask reuse, but gives each denoising step and
    window its own current K/V. Reuse requires a separate validated identity.
    """
    import torch.nn.functional as F

    from . import zing_h3_sparse_attn as sparse

    queries, keys, values, compressed, raw_k, raw_v = forward_view(
        plan, layer, q, history_rope, tensors
    )
    results, start = [], 0
    specs = getattr(plan, "_sparse_branches", {})
    for branch, stop in enumerate(plan.ends):
        original = specs.get(branch)
        if original is None:
            # No padded rows in the canonical prefix/history softmax.
            visible = plan.write_start + stop
            result = F.scaled_dot_product_attention(
                q[start:stop].transpose(0, 1)[None],
                raw_k[:visible].transpose(0, 1)[None],
                raw_v[:visible].transpose(0, 1)[None],
                scale=scale,
            )[0].transpose(0, 1)
        else:
            spec = getattr(plan, "_compressed_branch_specs", {}).get(branch)
            if spec is None:
                spec = branch_spec(plan, original)
            query = queries[spec.q_start : spec.q_stop].contiguous()
            mask = None
            protected = (
                layer == 0
                or original.mode == "dense"
                or spec.chunk_id < sparse.get_min_sparse_chunk()
                or spec.M_cand <= 8
            )
            if not protected:
                source = spec.b_phase == "source"
                step0 = spec.b_phase == "active" and spec.step_idx == 0
                late = not source and not step0
                masks = sparse.fused_route_and_build_masks(
                    query,
                    None,
                    keys,
                    layer,
                    spec.tau,
                    spec.L_q,
                    spec.M_q,
                    spec.M_cand,
                    spec.M_prefix,
                    spec.safety_scale,
                    source,
                    spec.L_k if source else 0,
                    spec.M_k if source else 0,
                    spec.M_cur if source else 0,
                    step0,
                    spec.L_k if step0 else 0,
                    spec.M_k if step0 else 0,
                    spec.M_cur if step0 else 0,
                    late,
                    0,
                    spec.L_k if late else 0,
                    spec.M_k if late else 0,
                    spec.M_cand if late else 0,
                    spec.M_cur if late else 0,
                    head_start=head_start,
                    num_heads=q.shape[1],
                    blk_kind_sig=spec.blk_kind_sig,
                    v=values,
                    enable_triage_score=sparse.is_triage_score_enabled(original.mode),
                    enable_virtual_mean=sparse.is_virtual_mean_enabled(original.mode),
                    price_pi=sparse.get_sparse_price_pi(original.mode),
                    enable_three_tier=True,
                    c_rf=sparse.get_sparse_c_rf(),
                    alpha_triage=sparse.get_sparse_alpha_triage(),
                    compressed_kv=compressed,
                )
                mask = masks[0 if source else 1 if step0 else 2]
            if mask is None:
                # A full budget still runs the all-FP8 kernel (100% Refine).
                count = torch.full(
                    (q.shape[1],), spec.M_k, dtype=torch.int32, device=q.device
                )
                selected = (
                    torch.arange(spec.M_k, dtype=torch.int32, device=q.device)[
                        None, None
                    ]
                    .expand(q.shape[1], spec.M_q, -1)
                    .contiguous()
                )
                order = torch.arange(q.shape[1], dtype=torch.int32, device=q.device)
                skim_count = skim_index = None
            else:
                count, selected, order = mask.cand_cnt, mask.cand_idx, mask.head_order
                skim_count, skim_index = mask.skim_cnt, mask.skim_idx
            padded_result = sparse.launch_zing_block_sparse_attn(
                query,
                None,
                None,
                scale,
                spec.L_k,
                spec.M_cand,
                spec.M_k,
                count,
                selected,
                order,
                compressed_kv=compressed,
                skim_cnt=skim_count,
                skim_idx=skim_index,
                enable_three_tier=True,
                block_valid_len=compressed["block_valid_len"],
                require_native_compressed=True,
                enable_virtual_mean=sparse.is_virtual_mean_enabled(original.mode),
            )
            inverse = tensors["compressed_queries_inverse"][start:stop] - spec.q_start
            result = padded_result.index_select(0, inverse)
        results.append(result)
        start = stop
    commit_layer(plan, layer, tensors)
    return torch.cat(results)


def mutable_store_views(store, begin, end):
    """All physical spans written by append_chunk, for capture rollback."""
    for name, value in vars(store).items():
        if isinstance(value, torch.Tensor):
            if name in ("k_res_fp8", "v_res_fp8") or value.ndim == 1:
                yield value[begin:end]
            elif name == "v_sub_mean_fp8":
                yield value[:, :, begin * 4 : end * 4]
            elif name in ("k_sub_mean_fp8", "k_sub_scale", "v_sub_scale"):
                yield value[:, begin * 4 : end * 4]
            elif name == "v_blk_mean_fp8":
                yield value[:, :, begin:end]
            else:
                yield value[:, begin:end]
    for value in store._router_stats.values():
        yield value[:, begin:end]
