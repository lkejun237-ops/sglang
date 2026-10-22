# SPDX-License-Identifier: Apache-2.0
"""Rotate normalized history K directly into the cache, without a V copy."""

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _history_rope_kernel(
    X,
    C,
    OUT,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    HS: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < N * H * D
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


@register_custom_op(mutates_args=["out"])
def fused_zing_history_rope(
    x: torch.Tensor, coefficients: torch.Tensor, out: torch.Tensor
) -> None:
    rows, heads, dim = x.shape
    with torch.cuda.device(x.device):
        _history_rope_kernel[(triton.cdiv(x.numel(), 512),)](
            x,
            coefficients,
            out,
            rows,
            heads,
            dim,
            coefficients.shape[-1],
            x.stride(0),
            x.stride(1),
            512,
            num_warps=4,
            enable_fp_fusion=False,
        )
