# SPDX-License-Identifier: Apache-2.0
"""Optional Q/K RMSNorm kernel, preserving FP32 normalization and BF16 output."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm(
    Q,
    K,
    WQ,
    WK,
    OQ,
    OK,
    D: tl.constexpr,
    EQ: tl.constexpr,
    EK: tl.constexpr,
    AQ: tl.constexpr,
    AK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    q = tl.load(Q + row * D + i, i < D, 0).to(tl.float32)
    k = tl.load(K + row * D + i, i < D, 0).to(tl.float32)
    q = q * tl.rsqrt(tl.sum(q * q, 0) / D + EQ)
    k = k * tl.rsqrt(tl.sum(k * k, 0) / D + EK)
    if AQ:
        q *= tl.load(WQ + i, i < D, 0).to(tl.float32)
    if AK:
        k *= tl.load(WK + i, i < D, 0).to(tl.float32)
    tl.store(OQ + row * D + i, q, i < D)
    tl.store(OK + row * D + i, k, i < D)


def qk_norm(q, k, norm_q, norm_k):
    q, k = q.contiguous(), k.contiguous()
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    wq, wk = getattr(norm_q, "weight", None), getattr(norm_k, "weight", None)
    _qk_norm[(q.numel() // q.shape[-1],)](
        q,
        k,
        q if wq is None else wq,
        k if wk is None else wk,
        oq,
        ok,
        q.shape[-1],
        norm_q.eps,
        norm_k.eps,
        wq is not None,
        wk is not None,
        triton.next_power_of_2(q.shape[-1]),
        enable_fp_fusion=False,
    )
    return oq, ok
