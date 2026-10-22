# SPDX-License-Identifier: Apache-2.0
"""Canonical Lynn Q261/H32/D64 preparation; eligibility lives in attention.py."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate_values(x, partner, cos, sin, column):
    partner = tl.where(column < 24, -partner.to(tl.float32), partner.to(tl.float32))
    cos = cos.to(tl.bfloat16).to(tl.float32)
    sin = sin.to(tl.bfloat16).to(tl.float32)
    # Eager Torch rounds BOTH BF16 products before their BF16 sum. Keeping
    # these round trips and disabling FMA preserves that arithmetic exactly.
    first = (x.to(tl.float32) * cos).to(tl.bfloat16).to(tl.float32)
    second = (partner * sin).to(tl.bfloat16).to(tl.float32)
    return (first + second).to(tl.bfloat16)


@triton.jit
def _query_rope(
    X,
    C,
    S,
    O,
    CB: tl.constexpr,
    CN: tl.constexpr,
    CD: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    column, token, batch = i % 64, (i // 2048) % 261, i // (261 * 2048)
    valid = i < TOTAL
    rotary = valid & (column < 48)
    x = tl.load(X + i, valid, 0)
    partner_column = tl.where(column < 24, column + 24, column - 24)
    partner = tl.load(X + i - column + partner_column, rotary, 0)
    angle = batch * CB + token * CN + column * CD
    cos, sin = tl.load(C + angle, rotary, 0), tl.load(S + angle, rotary, 0)
    result = tl.where(column < 48, _rotate_values(x, partner, cos, sin, column), x)
    tl.store(O + i, result, valid)


@triton.jit
def _keys_values(
    HK,
    HV,
    K,
    V,
    C,
    S,
    KR,
    VA,
    KC,
    VC,
    HISTORY: tl.constexpr,
    KEEP: tl.constexpr,
    CB: tl.constexpr,
    CN: tl.constexpr,
    CD: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    length = HISTORY + 261
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    column, token, batch = i % 64, (i // 2048) % length, i // (length * 2048)
    valid = i < TOTAL
    past, current = valid & (token < HISTORY), valid & (token >= HISTORY)
    past_offset = (batch * HISTORY + token) * 2048 + i % 2048
    current_offset = (batch * 261 + token - HISTORY) * 2048 + i % 2048
    raw_k = tl.where(
        past,
        tl.load(HK + past_offset, past, 0),
        tl.load(K + current_offset, current, 0),
    )
    raw_v = tl.where(
        past,
        tl.load(HV + past_offset, past, 0),
        tl.load(V + current_offset, current, 0),
    )
    partner_column = tl.where(column < 24, column + 24, column - 24)
    partner = tl.where(
        past,
        tl.load(HK + past_offset - column + partner_column, past & (column < 48), 0),
        tl.load(
            K + current_offset - column + partner_column, current & (column < 48), 0
        ),
    )
    rotary = valid & (column < 48)
    angle = batch * CB + token * CN + column * CD
    cos, sin = tl.load(C + angle, rotary, 0), tl.load(S + angle, rotary, 0)
    rotated = tl.where(
        column < 48, _rotate_values(raw_k, partner, cos, sin, column), raw_k
    )
    tl.store(KR + i, rotated, valid)
    tl.store(VA + i, raw_v, valid)
    # Attention includes all 261 current tokens. Only the 256 spatial tokens
    # enter history, in pre-RoPE form so their relative times can age next step.
    start = HISTORY + 256 - KEEP
    cache_offset = (batch * KEEP + token - start) * 2048 + i % 2048
    retained = valid & (token >= start) & (token < HISTORY + 256)
    tl.store(KC + cache_offset, raw_k, retained)
    tl.store(VC + cache_offset, raw_v, retained)


def prepare(
    q, k, v, history_k, history_v, query_rope, key_rope, spatial_tokens, cache_tokens
):
    batch = q.shape[0]
    history = history_k.shape[1]
    keep = min(history + spatial_tokens, cache_tokens)
    rotated_q = torch.empty_like(q)
    rotated_k = torch.empty(
        (batch, history + 261, 32, 64), device=q.device, dtype=q.dtype
    )
    all_v = torch.empty_like(rotated_k)
    cached_k = torch.empty((batch, keep, 32, 64), device=q.device, dtype=q.dtype)
    cached_v = torch.empty_like(cached_k)
    cos, sin = query_rope
    _query_rope[(triton.cdiv(q.numel(), 1024),)](
        q,
        cos,
        sin,
        rotated_q,
        cos.stride(0),
        cos.stride(1),
        cos.stride(-1),
        q.numel(),
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    cos, sin = key_rope
    _keys_values[(triton.cdiv(rotated_k.numel(), 1024),)](
        history_k,
        history_v,
        k,
        v,
        cos,
        sin,
        rotated_k,
        all_v,
        cached_k,
        cached_v,
        history,
        keep,
        cos.stride(0),
        cos.stride(1),
        cos.stride(-1),
        rotated_k.numel(),
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return rotated_q, rotated_k, all_v, cached_k, cached_v
