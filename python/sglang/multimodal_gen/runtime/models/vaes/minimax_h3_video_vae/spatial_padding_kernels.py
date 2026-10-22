# SPDX-License-Identifier: Apache-2.0
"""Inference-only dense-channel spatial reflection; imported lazily on CUDA."""

import torch
import triton
import triton.language as tl


@triton.jit
def _reflect_ndhwc(
    X,
    Y,
    S0,
    S1,
    S2,
    S3,
    S4,
    C: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    TOP: tl.constexpr,
    LEFT: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if TOTAL >= 2147483648:
        index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    c = index % C
    position = index // C
    w = position % OW - LEFT
    h = position // OW % OH - TOP
    t = position // (OW * OH) % T
    # A short temporal view can still have a batch stride beyond 32-bit addressing.
    n = (position // (OW * OH * T)).to(tl.int64)
    w = tl.where(w < 0, -w, tl.where(w >= W, 2 * W - 2 - w, w))
    h = tl.where(h < 0, -h, tl.where(h >= H, 2 * H - 2 - h, h))
    value = tl.load(X + n * S0 + c * S1 + t * S2 + h * S3 + w * S4, index < TOTAL)
    tl.store(Y + index, value, index < TOTAL)


def reflect_ndhwc(x, padding):
    left, right, top, bottom = padding[:4]
    n, c, t, h, w = x.shape
    oh, ow = h + top + bottom, w + left + right
    out = torch.empty(
        (n, c, t, oh, ow),
        device=x.device,
        dtype=x.dtype,
        memory_format=torch.channels_last_3d,
    )
    _reflect_ndhwc[(triton.cdiv(out.numel(), 1024),)](
        x,
        out,
        *x.stride(),
        c,
        t,
        h,
        w,
        oh,
        ow,
        top,
        left,
        out.numel(),
        BLOCK=1024,
        num_warps=4,
    )
    return out
