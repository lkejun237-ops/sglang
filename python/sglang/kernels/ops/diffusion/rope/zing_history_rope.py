# SPDX-License-Identifier: Apache-2.0
"""Rotate normalized history K directly into the cache, without a V copy."""

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _rotate_tile(
    X,
    C,
    OUT,
    i,
    valid,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    HS: tl.constexpr,
):
    row = i // (H * D)
    head = (i // D) % H
    d = i % D
    raw = tl.load(X + row * S + head * HS + d, valid, 0)
    x = raw.to(tl.float32)
    partner = tl.where(d < R // 2, d + R // 2, d - R // 2)
    other = tl.load(X + row * S + head * HS + partner, valid & (d < R), 0).to(
        tl.float32
    )
    other = tl.where(d < R // 2, -other, other)
    c = tl.load(C + row * R + d % (R // 2), valid & (d < R), 0).to(tl.float32)
    s = tl.load(C + row * R + R // 2 + d % (R // 2), valid & (d < R), 0).to(tl.float32)
    # Eager multiplication materializes BF16 products before adding them.
    a = (x * c).to(tl.bfloat16).to(tl.float32)
    b = (other * s).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + i, tl.where(d < R, (a + b).to(tl.bfloat16), raw), valid)


@triton.jit
def _history_rope_kernel(
    X,
    C,
    OUT,
    LIVE,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    HS: tl.constexpr,
    B: tl.constexpr,
    HAS_LIVE: tl.constexpr,
):
    if HAS_LIVE:
        live = tl.load(LIVE)
        if tl.program_id(0) * B >= live * H * D:
            return
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < N * H * D
    if HAS_LIVE:
        valid = valid & (i // (H * D) < live)
    _rotate_tile(X, C, OUT, i, valid, H, D, R, S, HS)


@triton.jit
def _history_rope_range_kernel(
    X,
    C,
    OUT,
    RANGE,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    HS: tl.constexpr,
    B: tl.constexpr,
):
    start = tl.load(RANGE) * H * D
    stop = tl.load(RANGE + 1) * H * D
    # A bounded grid consumes only dirty tiles. A full-capacity grid would
    # still schedule thousands of empty CTAs when most history is reused.
    for tile in range(tl.program_id(0), tl.cdiv(stop - start, B), tl.num_programs(0)):
        i = start + tile * B + tl.arange(0, B)
        _rotate_tile(X, C, OUT, i, i < stop, H, D, R, S, HS)


@register_custom_op(mutates_args=["out"])
def fused_zing_history_rope(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    out: torch.Tensor,
    live_rows: torch.Tensor | None = None,
    row_range: torch.Tensor | None = None,
) -> None:
    rows, heads, dim = x.shape
    with torch.cuda.device(x.device):
        if row_range is not None:
            _history_rope_range_kernel[(min(triton.cdiv(x.numel(), 512), 1024),)](
                x,
                coefficients,
                out,
                row_range,
                heads,
                dim,
                coefficients.shape[-1],
                x.stride(0),
                x.stride(1),
                512,
                num_warps=4,
                enable_fp_fusion=False,
            )
            return
        _history_rope_kernel[(triton.cdiv(x.numel(), 512),)](
            x,
            coefficients,
            out,
            live_rows,
            rows,
            heads,
            dim,
            coefficients.shape[-1],
            x.stride(0),
            x.stride(1),
            512,
            live_rows is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
