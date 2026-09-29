# SPDX-License-Identifier: Apache-2.0
"""Write Zing's current BF16 K/V directly into the post-RoPE store.

The caller checks the inference-only layout contract before importing this
module. Inputs have unit dimension stride and may have QKV row/head strides;
outputs are disjoint contiguous slices of the preallocated cache. Coefficients
are [rows, rotary_dim], with half-width cosine followed by half-width sine.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _zing_rope_kv_write_kernel(
    K,
    V,
    C,
    OK,
    OV,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    KHS: tl.constexpr,
    VHS: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < N * H * D
    row = i // (H * D)
    head = (i // D) % H
    d = i % D
    x = tl.load(K + row * KS + head * KHS + d, valid, 0).to(tl.float32)
    partner = tl.where(d < R // 2, d + R // 2, d - R // 2)
    other = tl.load(K + row * KS + head * KHS + partner, valid & (d < R), 0).to(
        tl.float32
    )
    other = tl.where(d < R // 2, -other, other)
    c = tl.load(C + row * R + d % (R // 2), valid & (d < R), 0).to(tl.float32)
    s = tl.load(C + row * R + R // 2 + d % (R // 2), valid & (d < R), 0).to(tl.float32)
    # Preserve both eager BF16 multiply roundings before the add/store rounding.
    a = (x * c).to(tl.bfloat16).to(tl.float32)
    b = (other * s).to(tl.bfloat16).to(tl.float32)
    result = tl.where(d < R, a + b, x)
    value = tl.load(V + row * VS + head * VHS + d, valid, 0)
    tl.store(OK + i, result, valid)
    tl.store(OV + i, value, valid)


@register_custom_op(mutates_args=["out_k", "out_v"])
def fused_zing_rope_kv_write(
    k: torch.Tensor,
    v: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
) -> None:
    """Launch after ``_can_fuse_rope_kv_write`` accepts these tensors."""
    rows, heads, dim = k.shape
    with torch.cuda.device(k.device):
        _zing_rope_kv_write_kernel[(triton.cdiv(k.numel(), 512),)](
            k,
            v,
            cos_sin_cache,
            out_k,
            out_v,
            rows,
            heads,
            dim,
            cos_sin_cache.shape[-1],
            k.stride(0),
            v.stride(0),
            k.stride(1),
            v.stride(1),
            512,
            num_warps=4,
            enable_fp_fusion=False,
        )
