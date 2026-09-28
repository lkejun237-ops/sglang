# SPDX-License-Identifier: Apache-2.0
"""Shift one row run left in place across many same-layout tensors at once.

Zing window eviction moves the same row run in every layer's K and V store.
Each program owns one byte slice of one tensor and walks row tiles in
ascending order with a barrier per tile, so a left shift of at least
``TILE_ROWS`` rows never overwrites a row still to be read. Programs share
no bytes.
"""

import torch
import triton
import triton.language as tl

TILE_ROWS = 16
TILE_BYTES = 256


@triton.jit
def _zing_kv_compact_kernel(
    data_ptrs,
    destination,
    source,
    rows,
    ROW_BYTES: tl.constexpr,
    TILE_R: tl.constexpr,
    TILE_B: tl.constexpr,
):
    base = tl.cast(tl.load(data_ptrs + tl.program_id(0)), tl.pointer_type(tl.uint8))
    col = tl.program_id(1) * TILE_B + tl.arange(0, TILE_B)
    col_mask = col < ROW_BYTES
    offsets = tl.arange(0, TILE_R)
    for start in range(0, rows, TILE_R):
        row = start + offsets
        mask = (row < rows)[:, None] & col_mask[None, :]
        src = base + (source + row).to(tl.int64)[:, None] * ROW_BYTES + col[None, :]
        dst = (
            base + (destination + row).to(tl.int64)[:, None] * ROW_BYTES + col[None, :]
        )
        tl.store(dst, tl.load(src, mask=mask), mask=mask)
        # Warps drift between iterations; a leading warp's store must not
        # reach rows a lagging warp has not loaded yet.
        tl.debug_barrier()


def fused_zing_kv_compact(
    data_ptrs: torch.Tensor,
    row_bytes: int,
    destination: int,
    source: int,
    rows: int,
) -> None:
    """Move rows [source, source + rows) to destination in every tensor.

    ``data_ptrs`` is an int64 device tensor of base addresses of tensors with
    contiguous rows of ``row_bytes``; ``source - destination >= TILE_ROWS``.
    """
    with torch.cuda.device(data_ptrs.device):
        _zing_kv_compact_kernel[
            (data_ptrs.numel(), triton.cdiv(row_bytes, TILE_BYTES))
        ](
            data_ptrs,
            destination,
            source,
            rows,
            row_bytes,
            TILE_ROWS,
            TILE_BYTES,
            num_warps=4,
        )
