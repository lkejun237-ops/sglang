# SPDX-License-Identifier: Apache-2.0
"""Optional incremental CNN input assembly; no session or configuration state."""

import triton
import triton.language as tl


@triton.jit
def _group_partials(
    X,
    P,
    C: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    group, time, batch = row % G, (row // G) % T, row // (G * T)
    size: tl.constexpr = C // G * H * W
    i = part * BLOCK + tl.arange(0, BLOCK)
    channel = group * (C // G) + i // (H * W)
    x = tl.load(
        X + batch * S0 + channel * S1 + time * S2 + (i // W % H) * S3 + (i % W) * S4,
        i < size,
        0,
    ).to(tl.float32)
    count = tl.minimum(BLOCK, size - part * BLOCK)
    mean = tl.sum(x, 0) / count
    delta = tl.where(i < size, x - mean, 0.0)
    m2 = tl.sum(delta * delta, 0)
    tl.store(P + (row * PARTS + part) * 2, mean)
    tl.store(P + (row * PARTS + part) * 2 + 1, m2)


@triton.jit
def _group_stats(
    P,
    S,
    SIZE: tl.constexpr,
    PARTS: tl.constexpr,
    CHUNK: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    valid = i < PARTS
    count = tl.where(valid, tl.minimum(CHUNK, SIZE - i * CHUNK), 0)
    means = tl.load(P + (row * PARTS + i) * 2, valid, 0)
    m2 = tl.load(P + (row * PARTS + i) * 2 + 1, valid, 0)
    mean = tl.sum(means * count, 0) / SIZE
    # Parallel variance combination avoids subtracting E[x]^2 from E[x^2].
    delta = means - mean
    variance = tl.sum(m2 + delta * delta * count, 0) / SIZE
    tl.store(S + row * 2, mean)
    tl.store(S + row * 2 + 1, tl.rsqrt(variance + EPS))


@triton.jit
def _assemble(
    X,
    PREV,
    GAMMA,
    BETA,
    STATS,
    OUT,
    CACHE,
    C: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HISTORY: tl.constexpr,
    OFFSET: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    OUT_COUNT: tl.constexpr,
    CACHE_COUNT: tl.constexpr,
    CL: tl.constexpr,
    REFLECT: tl.constexpr,
    HAS_PREVIOUS: tl.constexpr,
    NORM: tl.constexpr,
    AFFINE: tl.constexpr,
    GROUPS: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    X3: tl.constexpr,
    X4: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    P4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    is_out = i < OUT_COUNT
    j = tl.where(is_out, i, i - OUT_COUNT)
    height = tl.where(is_out, H + 2 * PH, H)
    width = tl.where(is_out, W + 2 * PW, W)
    times = tl.where(is_out, HISTORY + T - OFFSET, max(HISTORY, 1))
    if CL:
        channel = j % C
        spatial = j // C
        xx = spatial % width
        yy = spatial // width % height
        tt = spatial // (width * height) % times
        batch = spatial // (width * height * times)
    else:
        xx = j % width
        yy = j // width % height
        tt = j // (width * height) % times
        channel = j // (width * height * times) % C
        batch = j // (width * height * times * C)
    xx -= tl.where(is_out, PW, 0)
    yy -= tl.where(is_out, PH, 0)
    inside = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
    if REFLECT:
        xx = tl.where(xx < 0, -xx, tl.where(xx >= W, 2 * W - xx - 2, xx))
        yy = tl.where(yy < 0, -yy, tl.where(yy >= H, 2 * H - yy - 2, yy))
        inside = tl.full((BLOCK,), True, tl.int1)
    source_t = tt + tl.where(is_out, OFFSET, T)
    valid = i < OUT_COUNT + CACHE_COUNT
    current_t = source_t - HISTORY
    current = valid & inside & (current_t >= 0)
    past = valid & inside & (source_t < HISTORY) & HAS_PREVIOUS
    raw = tl.load(
        X + batch * X0 + channel * X1 + current_t * X2 + yy * X3 + xx * X4, current, 0
    )
    if NORM:
        row = (batch * T + current_t) * GROUPS + channel // (C // GROUPS)
        mean = tl.load(STATS + row * 2, current, 0)
        rstd = tl.load(STATS + row * 2 + 1, current, 0)
        value = (raw.to(tl.float32) - mean) * rstd
        if AFFINE:
            value = value * tl.load(GAMMA + channel, current, 0).to(tl.float32)
            value += tl.load(BETA + channel, current, 0).to(tl.float32)
        # GroupNorm writes the activation dtype before SiLU reads it.
        value = value.to(X.dtype.element_ty).to(tl.float32)
        raw = (value * tl.sigmoid(value)).to(X.dtype.element_ty)
    previous = tl.load(
        PREV + batch * P0 + channel * P1 + source_t * P2 + yy * P3 + xx * P4, past, 0
    )
    value = tl.where(current, raw, previous)
    tl.store(OUT + i, value, valid & is_out)
    tl.store(CACHE + i - OUT_COUNT, value, valid & ~is_out)


def assemble(x, previous, history, offset, padding, mode, channels_last, norm=None):
    import torch

    b, c, t, h, w = x.shape
    ph, pw = padding
    memory_format = torch.channels_last_3d if channels_last else torch.contiguous_format
    out = torch.empty(
        (b, c, history + t - offset, h + 2 * ph, w + 2 * pw),
        dtype=x.dtype,
        device=x.device,
        memory_format=memory_format,
    )
    cache = torch.empty(
        (b, c, history, h, w),
        dtype=x.dtype,
        device=x.device,
        memory_format=memory_format,
    )
    stats = x
    groups = 1
    if norm is not None:
        groups = norm.num_groups
        size = c // groups * h * w
        parts = triton.cdiv(size, 1024)
        partials = torch.empty(
            (b * t * groups, parts, 2), device=x.device, dtype=torch.float32
        )
        stats = torch.empty((b * t * groups, 2), device=x.device, dtype=torch.float32)
        _group_partials[(b * t * groups, parts)](
            x,
            partials,
            c,
            t,
            h,
            w,
            groups,
            *x.stride(),
            parts,
            1024,
            enable_fp_fusion=False,
        )
        _group_stats[(b * t * groups,)](
            partials,
            stats,
            size,
            parts,
            1024,
            norm.eps,
            triton.next_power_of_2(parts),
            enable_fp_fusion=False,
        )
    gamma = norm.weight if norm is not None and norm.weight is not None else x
    beta = norm.bias if norm is not None and norm.bias is not None else x
    old = x if previous is None else previous
    total = out.numel() + cache.numel()
    _assemble[(triton.cdiv(total, 256),)](
        x,
        old,
        gamma,
        beta,
        stats,
        out,
        cache,
        c,
        t,
        h,
        w,
        history,
        offset,
        ph,
        pw,
        out.numel(),
        cache.numel(),
        channels_last,
        mode == "reflect",
        previous is not None,
        norm is not None,
        norm is not None and norm.weight is not None,
        groups,
        *x.stride(),
        *old.stride(),
        256,
        enable_fp_fusion=False,
    )
    return out, cache if history else None
