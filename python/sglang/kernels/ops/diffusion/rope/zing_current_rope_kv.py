# SPDX-License-Identifier: Apache-2.0
"""Current Q/K RoPE and fixed-slot KV writes after the SP exchange."""

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _current_rope_kv_kernel(
    Q,
    K,
    V,
    C,
    ROWS,
    OUT_Q,
    RAW_K,
    ROT_K,
    OUT_V,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    CLEAN: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < N * H * D
    row, head, d = i // (H * D), (i // D) % H, i % D
    is_q = tl.program_id(1) == 0
    if is_q:
        xptr, stride_row, stride_head = Q, QS, QH
    else:
        xptr, stride_row, stride_head = K, KS, KH
    raw = tl.load(xptr + row * stride_row + head * stride_head + d, valid, 0)
    partner = tl.where(d < R // 2, d + R // 2, d - R // 2)
    other = tl.load(
        xptr + row * stride_row + head * stride_head + partner, valid & (d < R), 0
    ).to(tl.float32)
    other = tl.where(d < R // 2, -other, other)
    c = tl.load(C + row * R + d % (R // 2), valid & (d < R), 0).to(tl.float32)
    s = tl.load(C + row * R + R // 2 + d % (R // 2), valid & (d < R), 0).to(tl.float32)
    # Preserve both eager BF16 multiplication rounding boundaries; no FMA.
    a = (raw.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
    b = (other * s).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(d < R, (a + b).to(tl.bfloat16), raw)
    if is_q:
        tl.store(OUT_Q + i, rotated, valid)
    else:
        dst_row = tl.load(ROWS + row, valid, 0)
        dst = dst_row * H * D + head * D + d
        tl.store(ROT_K + dst, rotated, valid)
        value = tl.load(V + row * VS + head * VH + d, valid, 0)
        tl.store(OUT_V + dst, value, valid)
        tl.store(RAW_K + dst, raw, valid & (row < CLEAN))


@register_custom_op(mutates_args=["out_q", "raw_k", "rotated_k", "out_v"])
def fused_zing_current_rope_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    coefficients: torch.Tensor,
    rows: torch.Tensor,
    out_q: torch.Tensor,
    raw_k: torch.Tensor,
    rotated_k: torch.Tensor,
    out_v: torch.Tensor,
    clean_rows: int,
) -> None:
    n, h, d = q.shape
    with torch.cuda.device(q.device):
        _current_rope_kv_kernel[(triton.cdiv(q.numel(), 512), 2)](
            q,
            k,
            v,
            coefficients,
            rows,
            out_q,
            raw_k,
            rotated_k,
            out_v,
            n,
            h,
            d,
            coefficients.shape[-1],
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            clean_rows,
            512,
            num_warps=4,
            enable_fp_fusion=False,
        )
