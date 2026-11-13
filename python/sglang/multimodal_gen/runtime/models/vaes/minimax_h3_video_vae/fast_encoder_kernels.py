# SPDX-License-Identifier: Apache-2.0
"""SM90-optimized Triton kernels for single-GPU causal H3 visual VAE encoding.

All intermediate activations in the fast single-GPU engine are stored in physical
``[B, T, H, W, C]`` row-major buffers, whose zero-copy ``permute(0, 4, 1, 2, 3)``
view is ``torch.channels_last_3d`` ``[B, C, T, H, W]``. Keeping ``C`` innermost
makes every GroupNorm reduction, residual bias fusion, causal history update, and
spatial reflection pad 100% memory-coalesced on HBM3.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@triton.jit
def _tile_pack_pad_uint8_kernel(
    FRAMES,
    PREV_HIST,
    TILE_Y0,
    TILE_X0,
    PADDED_OUT,
    NEXT_HIST,
    F_S0: tl.constexpr,
    F_S1: tl.constexpr,
    F_S2: tl.constexpr,
    F_S3: tl.constexpr,
    T_VALID: tl.constexpr,
    T_NEEDED: tl.constexpr,
    HIST: tl.constexpr,
    OFFSET: tl.constexpr,
    H_TILE: tl.constexpr,
    W_TILE: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    REFLECT: tl.constexpr,
    HAS_PREV: tl.constexpr,
    M0: tl.constexpr,
    M1: tl.constexpr,
    M2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    py = tl.program_id(2)

    H_OUT: tl.constexpr = H_TILE + 2 * PH
    W_OUT: tl.constexpr = W_TILE + 2 * PW
    T_OUT: tl.constexpr = HIST + T_NEEDED - OFFSET

    write_padded = s >= OFFSET
    write_hist = (HIST > 0) & (s >= T_NEEDED)
    if (write_padded | write_hist) == 0:
        return

    y0 = tl.load(TILE_Y0 + b)
    x0 = tl.load(TILE_X0 + b)

    # Map padded coordinate py -> unpadded tile coordinate sy in [0, H_TILE)
    sy = py - PH
    inside_y = (sy >= 0) & (sy < H_TILE)
    if REFLECT:
        sy = tl.where(sy < 0, -sy, tl.where(sy >= H_TILE, 2 * H_TILE - 2 - sy, sy))

    for px_start in range(0, W_OUT, BLOCK_W):
        px = px_start + tl.arange(0, BLOCK_W)
        mask_w = px < W_OUT
        sx = px - PW
        inside_x = (sx >= 0) & (sx < W_TILE)
        if REFLECT:
            sx = tl.where(sx < 0, -sx, tl.where(sx >= W_TILE, 2 * W_TILE - 2 - sx, sx))
            valid_val = mask_w
        else:
            valid_val = mask_w & inside_y & inside_x

        if s < HIST:
            if HAS_PREV:
                h_base = (((b * HIST + s) * H_TILE + sy) * W_TILE + sx.to(tl.int64)) * 3
                v0 = tl.load(PREV_HIST + h_base + 0, mask=valid_val, other=0.0)
                v1 = tl.load(PREV_HIST + h_base + 1, mask=valid_val, other=0.0)
                v2 = tl.load(PREV_HIST + h_base + 2, mask=valid_val, other=0.0)
            else:
                v0 = tl.zeros((BLOCK_W,), dtype=PADDED_OUT.dtype.element_ty)
                v1 = tl.zeros((BLOCK_W,), dtype=PADDED_OUT.dtype.element_ty)
                v2 = tl.zeros((BLOCK_W,), dtype=PADDED_OUT.dtype.element_ty)
        else:
            t_cur = s - HIST
            t_src = tl.minimum(t_cur, T_VALID - 1)
            gy = y0 + sy
            gx = x0 + sx
            f_off = (
                t_src.to(tl.int64) * F_S1
                + gy.to(tl.int64) * F_S2
                + gx.to(tl.int64) * F_S3
            )
            u0 = tl.load(FRAMES + 0 * F_S0 + f_off, mask=valid_val, other=0).to(
                tl.float32
            )
            u1 = tl.load(FRAMES + 1 * F_S0 + f_off, mask=valid_val, other=0).to(
                tl.float32
            )
            u2 = tl.load(FRAMES + 2 * F_S0 + f_off, mask=valid_val, other=0).to(
                tl.float32
            )
            # Match PyTorch BF16 ImageNet normalization 0-ulp bit-for-bit:
            # p = (u.to(dtype) / 255).to(dtype); out = ((p - mean_q) / std_q).to(dtype)
            d0 = (u0 / 255.0).to(PADDED_OUT.dtype.element_ty).to(tl.float32)
            d1 = (u1 / 255.0).to(PADDED_OUT.dtype.element_ty).to(tl.float32)
            d2 = (u2 / 255.0).to(PADDED_OUT.dtype.element_ty).to(tl.float32)
            n0 = ((d0 - M0).to(PADDED_OUT.dtype.element_ty).to(tl.float32) / S0).to(
                PADDED_OUT.dtype.element_ty
            )
            n1 = ((d1 - M1).to(PADDED_OUT.dtype.element_ty).to(tl.float32) / S1).to(
                PADDED_OUT.dtype.element_ty
            )
            n2 = ((d2 - M2).to(PADDED_OUT.dtype.element_ty).to(tl.float32) / S2).to(
                PADDED_OUT.dtype.element_ty
            )
            zero_c = tl.zeros((BLOCK_W,), dtype=PADDED_OUT.dtype.element_ty)
            v0 = tl.where(valid_val, n0, zero_c)
            v1 = tl.where(valid_val, n1, zero_c)
            v2 = tl.where(valid_val, n2, zero_c)

        if write_padded:
            t_out = s - OFFSET
            out_base = (
                ((b * T_OUT + t_out) * H_OUT + py) * W_OUT + px.to(tl.int64)
            ) * 3
            tl.store(PADDED_OUT + out_base + 0, v0, mask=mask_w)
            tl.store(PADDED_OUT + out_base + 1, v1, mask=mask_w)
            tl.store(PADDED_OUT + out_base + 2, v2, mask=mask_w)

        if write_hist & inside_y:
            mask_hist = mask_w & inside_x
            t_h = s - T_NEEDED
            nh_base = (
                ((b * HIST + t_h) * H_TILE + (py - PH)) * W_TILE
                + (px - PW).to(tl.int64)
            ) * 3
            tl.store(NEXT_HIST + nh_base + 0, v0, mask=mask_hist)
            tl.store(NEXT_HIST + nh_base + 1, v1, mask=mask_hist)
            tl.store(NEXT_HIST + nh_base + 2, v2, mask=mask_hist)


@triton.jit
def _tile_pack_pad_bf16_kernel(
    X,
    PREV_HIST,
    TILE_Y0,
    TILE_X0,
    PADDED_OUT,
    NEXT_HIST,
    X_S0: tl.constexpr,
    X_S1: tl.constexpr,
    X_S2: tl.constexpr,
    X_S3: tl.constexpr,
    X_S4: tl.constexpr,
    N_IN: tl.constexpr,
    T_IN: tl.constexpr,
    HIST: tl.constexpr,
    OFFSET: tl.constexpr,
    H_TILE: tl.constexpr,
    W_TILE: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    REFLECT: tl.constexpr,
    HAS_PREV: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    bn = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    py = tl.program_id(2)

    tile_id = bn // N_IN
    n_idx = bn % N_IN

    H_OUT: tl.constexpr = H_TILE + 2 * PH
    W_OUT: tl.constexpr = W_TILE + 2 * PW
    T_OUT: tl.constexpr = HIST + T_IN - OFFSET

    write_padded = s >= OFFSET
    write_hist = (HIST > 0) & (s >= T_IN)
    if (write_padded | write_hist) == 0:
        return

    y0 = tl.load(TILE_Y0 + tile_id)
    x0 = tl.load(TILE_X0 + tile_id)

    sy = py - PH
    inside_y = (sy >= 0) & (sy < H_TILE)
    if REFLECT:
        sy = tl.where(sy < 0, -sy, tl.where(sy >= H_TILE, 2 * H_TILE - 2 - sy, sy))

    for px_start in range(0, W_OUT, BLOCK_W):
        px = px_start + tl.arange(0, BLOCK_W)
        mask_w = px < W_OUT
        sx = px - PW
        inside_x = (sx >= 0) & (sx < W_TILE)
        if REFLECT:
            sx = tl.where(sx < 0, -sx, tl.where(sx >= W_TILE, 2 * W_TILE - 2 - sx, sx))
            valid_val = mask_w
        else:
            valid_val = mask_w & inside_y & inside_x

        if s < HIST:
            if HAS_PREV:
                h_base = (
                    ((bn * HIST + s) * H_TILE + sy) * W_TILE + sx.to(tl.int64)
                ) * 3
                v0 = tl.load(PREV_HIST + h_base + 0, mask=valid_val, other=0.0)
                v1 = tl.load(PREV_HIST + h_base + 1, mask=valid_val, other=0.0)
                v2 = tl.load(PREV_HIST + h_base + 2, mask=valid_val, other=0.0)
            else:
                v0 = tl.zeros((BLOCK_W,), dtype=tl.bfloat16)
                v1 = tl.zeros((BLOCK_W,), dtype=tl.bfloat16)
                v2 = tl.zeros((BLOCK_W,), dtype=tl.bfloat16)
        else:
            t_cur = s - HIST
            gy = y0 + sy
            gx = x0 + sx
            x_off = (
                n_idx * X_S0
                + t_cur.to(tl.int64) * X_S2
                + gy.to(tl.int64) * X_S3
                + gx.to(tl.int64) * X_S4
            )
            v0 = tl.load(X + 0 * X_S1 + x_off, mask=valid_val, other=0.0).to(
                tl.bfloat16
            )
            v1 = tl.load(X + 1 * X_S1 + x_off, mask=valid_val, other=0.0).to(
                tl.bfloat16
            )
            v2 = tl.load(X + 2 * X_S1 + x_off, mask=valid_val, other=0.0).to(
                tl.bfloat16
            )

        if write_padded:
            t_out = s - OFFSET
            out_base = (
                ((bn * T_OUT + t_out) * H_OUT + py) * W_OUT + px.to(tl.int64)
            ) * 3
            tl.store(PADDED_OUT + out_base + 0, v0, mask=mask_w)
            tl.store(PADDED_OUT + out_base + 1, v1, mask=mask_w)
            tl.store(PADDED_OUT + out_base + 2, v2, mask=mask_w)

        if write_hist & inside_y:
            mask_hist = mask_w & inside_x
            t_h = s - T_IN
            nh_base = (
                ((bn * HIST + t_h) * H_TILE + (py - PH)) * W_TILE
                + (px - PW).to(tl.int64)
            ) * 3
            tl.store(NEXT_HIST + nh_base + 0, v0, mask=mask_hist)
            tl.store(NEXT_HIST + nh_base + 1, v1, mask=mask_hist)
            tl.store(NEXT_HIST + nh_base + 2, v2, mask=mask_hist)


@triton.jit
def _store_with_spatial_pad1(
    PADDED_OUT,
    out_bt_base,
    y,
    x,
    c_offs,
    val,
    mask_hw,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    REFLECT: tl.constexpr,
):
    W_OUT: tl.constexpr = W + 2
    # Center write: (y + 1, x + 1)
    py_c = (y + 1).to(tl.int64)
    px_c = (x + 1).to(tl.int64)
    zero_hw = py_c * 0
    center_off = (py_c[:, None] * W_OUT + px_c[:, None]) * C + c_offs[None, :]
    tl.store(PADDED_OUT + out_bt_base + center_off, val, mask=mask_hw[:, None])

    if REFLECT:
        # Top mirror: y == 1 -> py = 0
        m_top = mask_hw & (y == 1)
        off_top = px_c[:, None] * C + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + off_top, val, mask=m_top[:, None])

        # Bottom mirror: y == H - 2 -> py = H + 1
        m_bot = mask_hw & (y == H - 2)
        off_bot = ((H + 1) * W_OUT + px_c)[:, None] * C + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + off_bot, val, mask=m_bot[:, None])

        # Left mirror: x == 1 -> px = 0
        m_left = mask_hw & (x == 1)
        off_left = (py_c * W_OUT)[:, None] * C + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + off_left, val, mask=m_left[:, None])

        # Right mirror: x == W - 2 -> px = W + 1
        m_right = mask_hw & (x == W - 2)
        off_right = (py_c * W_OUT + (W + 1))[:, None] * C + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + off_right, val, mask=m_right[:, None])

        # Four corners (broadcast pointer shape to [BLOCK_HW, C_BLK])
        m_tl = m_top & (x == 1)
        off_tl = zero_hw[:, None] + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + off_tl, val, mask=m_tl[:, None])

        m_tr = m_top & (x == W - 2)
        off_tr = (zero_hw + (W + 1) * C)[:, None] + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + off_tr, val, mask=m_tr[:, None])

        m_bl = m_bot & (x == 1)
        off_bl = (zero_hw + (H + 1) * W_OUT * C)[:, None] + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + off_bl, val, mask=m_bl[:, None])

        m_br = m_bot & (x == W - 2)
        off_br = (zero_hw + ((H + 1) * W_OUT + (W + 1)) * C)[:, None] + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + off_br, val, mask=m_br[:, None])
    else:
        zero_val = tl.zeros(val.shape, dtype=val.dtype)
        m_top = mask_hw & (y == 0)
        tl.store(
            PADDED_OUT + out_bt_base + px_c[:, None] * C + c_offs[None, :],
            zero_val,
            mask=m_top[:, None],
        )
        m_bot = mask_hw & (y == H - 1)
        off_bot = ((H + 1) * W_OUT + px_c)[:, None] * C + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + off_bot, zero_val, mask=m_bot[:, None])
        m_left = mask_hw & (x == 0)
        tl.store(
            PADDED_OUT + out_bt_base + (py_c * W_OUT)[:, None] * C + c_offs[None, :],
            zero_val,
            mask=m_left[:, None],
        )
        m_right = mask_hw & (x == W - 1)
        off_right = (py_c * W_OUT + (W + 1))[:, None] * C + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + off_right, zero_val, mask=m_right[:, None])
        m_tl = m_top & (x == 0)
        off_tl = zero_hw[:, None] + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + off_tl, zero_val, mask=m_tl[:, None])
        m_tr = m_top & (x == W - 1)
        off_tr = (zero_hw + (W + 1) * C)[:, None] + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + off_tr, zero_val, mask=m_tr[:, None])
        m_bl = m_bot & (x == 0)
        off_bl = (zero_hw + (H + 1) * W_OUT * C)[:, None] + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + off_bl, zero_val, mask=m_bl[:, None])
        m_br = m_bot & (x == W - 1)
        off_br = (zero_hw + ((H + 1) * W_OUT + (W + 1)) * C)[:, None] + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + off_br, zero_val, mask=m_br[:, None])


@triton.jit
def _gn_silu_pad_single_pass_kernel(
    X,
    RESIDUAL,
    ADD_BIAS,
    PREV_HIST,
    GAMMA,
    BETA,
    PADDED_OUT,
    NEXT_HIST,
    T_IN: tl.constexpr,
    HIST: tl.constexpr,
    OFFSET: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    GROUPS: tl.constexpr,
    G_PER_CTA: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    EPS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_ADD_BIAS: tl.constexpr,
    HAS_PREV: tl.constexpr,
    AFFINE: tl.constexpr,
    PAD_SPATIAL: tl.constexpr,
    REFLECT: tl.constexpr,
):
    bs = tl.program_id(0).to(tl.int64)
    g_blk = tl.program_id(1)

    S_TOTAL: tl.constexpr = HIST + T_IN
    T_OUT: tl.constexpr = HIST + T_IN - OFFSET
    HW: tl.constexpr = H * W
    D: tl.constexpr = C // GROUPS
    C_BLK: tl.constexpr = G_PER_CTA * D
    H_OUT: tl.constexpr = H + 2 if PAD_SPATIAL else H
    W_OUT: tl.constexpr = W + 2 if PAD_SPATIAL else W

    b = bs // S_TOTAL
    s = (bs % S_TOTAL).to(tl.int32)

    write_padded = s >= OFFSET
    write_hist = (HIST > 0) & (s >= T_IN)
    if (write_padded | write_hist) == 0:
        return

    c_start = g_blk * C_BLK
    c_offs = c_start + tl.arange(0, C_BLK)

    # Path 1: historical prefix frame (already normalized + SiLU in prior step)
    if s < HIST:
        out_bt_base = (b * T_OUT + (s - OFFSET)) * (H_OUT * W_OUT * C)
        nh_bt_base = (b * HIST + (s - T_IN)) * (HW * C)
        prev_bt_base = (b * HIST + s) * (HW * C)

        for hw_start in range(0, HW, BLOCK_HW):
            hw = hw_start + tl.arange(0, BLOCK_HW)
            mask_hw = hw < HW
            y = hw // W
            x = hw % W
            if HAS_PREV:
                in_off = hw.to(tl.int64)[:, None] * C + c_offs[None, :]
                val = tl.load(
                    PREV_HIST + prev_bt_base + in_off,
                    mask=mask_hw[:, None],
                    other=0.0,
                )
            else:
                val = tl.zeros((BLOCK_HW, C_BLK), dtype=PADDED_OUT.dtype.element_ty)

            if write_hist:
                nh_off = hw.to(tl.int64)[:, None] * C + c_offs[None, :]
                tl.store(NEXT_HIST + nh_bt_base + nh_off, val, mask=mask_hw[:, None])

            if write_padded:
                if PAD_SPATIAL:
                    _store_with_spatial_pad1(
                        PADDED_OUT,
                        out_bt_base,
                        y,
                        x,
                        c_offs,
                        val,
                        mask_hw,
                        H,
                        W,
                        C,
                        REFLECT,
                    )
                else:
                    out_off = hw.to(tl.int64)[:, None] * C + c_offs[None, :]
                    tl.store(
                        PADDED_OUT + out_bt_base + out_off, val, mask=mask_hw[:, None]
                    )
        return

    # Path 2: current frame -> single-CTA L1-resident GroupNorm + SiLU + History + Pad
    t = s - HIST
    x_bt_base = (b * T_IN + t) * (HW * C)

    if HAS_ADD_BIAS:
        add_b = tl.load(ADD_BIAS + c_offs).to(tl.float32)
    if AFFINE:
        gamma = tl.load(GAMMA + c_offs).to(tl.float32)
        beta = tl.load(BETA + c_offs).to(tl.float32)

    mean_acc = tl.zeros((G_PER_CTA,), dtype=tl.float32)
    m2_acc = tl.zeros((G_PER_CTA,), dtype=tl.float32)
    n_acc = tl.sum(tl.zeros((1,), dtype=tl.float32), 0)

    for hw_start in range(0, HW, BLOCK_HW):
        hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = hw < HW
        in_off = hw.to(tl.int64)[:, None] * C + c_offs[None, :]
        raw = tl.load(X + x_bt_base + in_off, mask=mask_hw[:, None], other=0.0).to(
            tl.float32
        )
        if HAS_RESIDUAL:
            res = tl.load(
                RESIDUAL + x_bt_base + in_off, mask=mask_hw[:, None], other=0.0
            ).to(tl.float32)
            raw = raw + res
        if HAS_ADD_BIAS:
            raw = raw + add_b[None, :]

        n_blk = tl.sum(mask_hw.to(tl.float32), 0) * D
        raw_masked = tl.where(mask_hw[:, None], raw, 0.0)
        grouped = tl.reshape(raw_masked, (BLOCK_HW, G_PER_CTA, D))
        mean_blk = tl.sum(tl.sum(grouped, 2), 0) / n_blk
        delta = tl.where(
            tl.reshape(mask_hw, (BLOCK_HW, 1, 1)),
            grouped - mean_blk[None, :, None],
            0.0,
        )
        m2_blk = tl.sum(tl.sum(delta * delta, 2), 0)

        d_mean = mean_blk - mean_acc
        n_new = n_acc + n_blk
        mean_acc = mean_acc + d_mean * (n_blk / n_new)
        m2_acc = m2_acc + m2_blk + d_mean * d_mean * (n_acc * n_blk / n_new)
        n_acc = n_new

    var = m2_acc / (HW * D)
    rstd = tl.rsqrt(var + EPS)

    mean_c = tl.reshape(tl.broadcast_to(mean_acc[:, None], (G_PER_CTA, D)), (C_BLK,))
    rstd_c = tl.reshape(tl.broadcast_to(rstd[:, None], (G_PER_CTA, D)), (C_BLK,))

    out_bt_base = (b * T_OUT + (s - OFFSET)) * (H_OUT * W_OUT * C)
    nh_bt_base = (b * HIST + (s - T_IN)) * (HW * C)

    for hw_start in range(0, HW, BLOCK_HW):
        hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = hw < HW
        y = hw // W
        x = hw % W
        in_off = hw.to(tl.int64)[:, None] * C + c_offs[None, :]
        raw = tl.load(X + x_bt_base + in_off, mask=mask_hw[:, None], other=0.0).to(
            tl.float32
        )
        if HAS_RESIDUAL:
            res = tl.load(
                RESIDUAL + x_bt_base + in_off, mask=mask_hw[:, None], other=0.0
            ).to(tl.float32)
            raw = raw + res
        if HAS_ADD_BIAS:
            raw = raw + add_b[None, :]

        normed = (raw - mean_c[None, :]) * rstd_c[None, :]
        if AFFINE:
            normed = normed * gamma[None, :] + beta[None, :]
        # Match PyTorch BF16 SiLU: norm output rounds to dtype before silu
        normed_bf16 = normed.to(PADDED_OUT.dtype.element_ty).to(tl.float32)
        act = (normed_bf16 * tl.sigmoid(normed_bf16)).to(PADDED_OUT.dtype.element_ty)

        if write_hist:
            tl.store(NEXT_HIST + nh_bt_base + in_off, act, mask=mask_hw[:, None])

        if write_padded:
            if PAD_SPATIAL:
                _store_with_spatial_pad1(
                    PADDED_OUT,
                    out_bt_base,
                    y,
                    x,
                    c_offs,
                    act,
                    mask_hw,
                    H,
                    W,
                    C,
                    REFLECT,
                )
            else:
                tl.store(PADDED_OUT + out_bt_base + in_off, act, mask=mask_hw[:, None])


@triton.jit
def _gn_partials_coalesced_kernel(
    X,
    RESIDUAL,
    ADD_BIAS,
    PARTIALS_MEAN,
    PARTIALS_M2,
    HW: tl.constexpr,
    C: tl.constexpr,
    GROUPS: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    PIXELS_PER_PART: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    C_STEP: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_ADD_BIAS: tl.constexpr,
):
    bt = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)

    D: tl.constexpr = C // GROUPS
    G_STEP: tl.constexpr = C_STEP // D

    part_hw_start = part * PIXELS_PER_PART
    part_hw_end = tl.minimum(HW, part_hw_start + PIXELS_PER_PART)
    if part_hw_start >= HW:
        for g_sub in range(0, GROUPS, G_STEP):
            g_offs = g_sub + tl.arange(0, G_STEP)
            p_idx = (bt * GROUPS + g_offs.to(tl.int64)) * NUM_PARTS + part
            tl.store(PARTIALS_MEAN + p_idx, tl.zeros((G_STEP,), dtype=tl.float32))
            tl.store(PARTIALS_M2 + p_idx, tl.zeros((G_STEP,), dtype=tl.float32))
        return

    x_bt_base = bt * (HW * C)

    for c_start in range(0, C, C_STEP):
        c_offs = c_start + tl.arange(0, C_STEP)
        g_sub = c_start // D
        if HAS_ADD_BIAS:
            add_b = tl.load(ADD_BIAS + c_offs).to(tl.float32)

        mean_acc = tl.zeros((G_STEP,), dtype=tl.float32)
        m2_acc = tl.zeros((G_STEP,), dtype=tl.float32)
        n_acc = tl.sum(tl.zeros((1,), dtype=tl.float32), 0)

        for rel_hw in range(0, PIXELS_PER_PART, BLOCK_HW):
            hw = part_hw_start + rel_hw + tl.arange(0, BLOCK_HW)
            mask_hw = hw < part_hw_end
            n_blk = tl.sum(mask_hw.to(tl.float32), 0) * D
            if n_blk > 0:
                in_off = hw.to(tl.int64)[:, None] * C + c_offs[None, :]
                raw = tl.load(
                    X + x_bt_base + in_off, mask=mask_hw[:, None], other=0.0
                ).to(tl.float32)
                if HAS_RESIDUAL:
                    res = tl.load(
                        RESIDUAL + x_bt_base + in_off,
                        mask=mask_hw[:, None],
                        other=0.0,
                    ).to(tl.float32)
                    raw = raw + res
                if HAS_ADD_BIAS:
                    raw = raw + add_b[None, :]

                raw_masked = tl.where(mask_hw[:, None], raw, 0.0)
                grouped = tl.reshape(raw_masked, (BLOCK_HW, G_STEP, D))
                mean_blk = tl.sum(tl.sum(grouped, 2), 0) / n_blk
                delta = tl.where(
                    tl.reshape(mask_hw, (BLOCK_HW, 1, 1)),
                    grouped - mean_blk[None, :, None],
                    0.0,
                )
                m2_blk = tl.sum(tl.sum(delta * delta, 2), 0)

                d_mean = mean_blk - mean_acc
                n_new = n_acc + n_blk
                mean_acc = mean_acc + d_mean * (n_blk / n_new)
                m2_acc = m2_acc + m2_blk + d_mean * d_mean * (n_acc * n_blk / n_new)
                n_acc = n_new

        g_offs = g_sub + tl.arange(0, G_STEP)
        p_idx = (bt * GROUPS + g_offs.to(tl.int64)) * NUM_PARTS + part
        tl.store(PARTIALS_MEAN + p_idx, mean_acc)
        tl.store(PARTIALS_M2 + p_idx, m2_acc)


@triton.jit
def _gn_apply_silu_pad_coalesced_kernel(
    X,
    RESIDUAL,
    ADD_BIAS,
    PREV_HIST,
    GAMMA,
    BETA,
    PARTIALS_MEAN,
    PARTIALS_M2,
    PADDED_OUT,
    NEXT_HIST,
    T_IN: tl.constexpr,
    HIST: tl.constexpr,
    OFFSET: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    GROUPS: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    PIXELS_PER_PART: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    C_BLK: tl.constexpr,
    EPS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_ADD_BIAS: tl.constexpr,
    HAS_PREV: tl.constexpr,
    AFFINE: tl.constexpr,
    PAD_SPATIAL: tl.constexpr,
    REFLECT: tl.constexpr,
):
    bs = tl.program_id(0).to(tl.int64)
    hw_blk = tl.program_id(1)
    c_blk_id = tl.program_id(2)

    S_TOTAL: tl.constexpr = HIST + T_IN
    T_OUT: tl.constexpr = HIST + T_IN - OFFSET
    HW: tl.constexpr = H * W
    D: tl.constexpr = C // GROUPS
    G_BLK: tl.constexpr = C_BLK // D
    H_OUT: tl.constexpr = H + 2 if PAD_SPATIAL else H
    W_OUT: tl.constexpr = W + 2 if PAD_SPATIAL else W

    b = bs // S_TOTAL
    s = (bs % S_TOTAL).to(tl.int32)

    write_padded = s >= OFFSET
    write_hist = (HIST > 0) & (s >= T_IN)
    if (write_padded | write_hist) == 0:
        return

    hw = hw_blk * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = hw < HW
    y = hw // W
    x = hw % W
    c_start = c_blk_id * C_BLK
    c_offs = c_start + tl.arange(0, C_BLK)
    in_off = hw.to(tl.int64)[:, None] * C + c_offs[None, :]

    out_bt_base = (b * T_OUT + (s - OFFSET)) * (H_OUT * W_OUT * C)
    nh_bt_base = (b * HIST + (s - T_IN)) * (HW * C)

    if s < HIST:
        if HAS_PREV:
            prev_bt_base = (b * HIST + s) * (HW * C)
            val = tl.load(
                PREV_HIST + prev_bt_base + in_off,
                mask=mask_hw[:, None],
                other=0.0,
            )
        else:
            val = tl.zeros((BLOCK_HW, C_BLK), dtype=PADDED_OUT.dtype.element_ty)

        if write_hist:
            tl.store(NEXT_HIST + nh_bt_base + in_off, val, mask=mask_hw[:, None])
        if write_padded:
            if PAD_SPATIAL:
                _store_with_spatial_pad1(
                    PADDED_OUT,
                    out_bt_base,
                    y,
                    x,
                    c_offs,
                    val,
                    mask_hw,
                    H,
                    W,
                    C,
                    REFLECT,
                )
            else:
                tl.store(PADDED_OUT + out_bt_base + in_off, val, mask=mask_hw[:, None])
        return

    t = s - HIST
    bt = b * T_IN + t
    x_bt_base = bt * (HW * C)

    # Combine NUM_PARTS (<= 32, power of 2) partial stats from L2 cache
    g_start = c_start // D
    g_offs = g_start + tl.arange(0, G_BLK)
    p_offs = tl.arange(0, NUM_PARTS)
    valid_pixels = tl.maximum(
        0, tl.minimum(PIXELS_PER_PART, HW - p_offs * PIXELS_PER_PART)
    )
    counts = (valid_pixels * D).to(tl.float32)  # [NUM_PARTS]
    p_idx = (bt * GROUPS + g_offs.to(tl.int64))[:, None] * NUM_PARTS + p_offs[None, :]
    means = tl.load(PARTIALS_MEAN + p_idx)  # [G_BLK, NUM_PARTS]
    m2s = tl.load(PARTIALS_M2 + p_idx)  # [G_BLK, NUM_PARTS]

    total_n: tl.constexpr = HW * D
    mean_g = tl.sum(means * counts[None, :], 1) / total_n
    delta_g = means - mean_g[:, None]
    var_g = tl.sum(m2s + delta_g * delta_g * counts[None, :], 1) / total_n
    rstd_g = tl.rsqrt(var_g + EPS)

    mean_c = tl.reshape(tl.broadcast_to(mean_g[:, None], (G_BLK, D)), (C_BLK,))
    rstd_c = tl.reshape(tl.broadcast_to(rstd_g[:, None], (G_BLK, D)), (C_BLK,))

    raw = tl.load(X + x_bt_base + in_off, mask=mask_hw[:, None], other=0.0).to(
        tl.float32
    )
    if HAS_RESIDUAL:
        res = tl.load(
            RESIDUAL + x_bt_base + in_off, mask=mask_hw[:, None], other=0.0
        ).to(tl.float32)
        raw = raw + res
    if HAS_ADD_BIAS:
        add_b = tl.load(ADD_BIAS + c_offs).to(tl.float32)
        raw = raw + add_b[None, :]

    normed = (raw - mean_c[None, :]) * rstd_c[None, :]
    if AFFINE:
        gamma = tl.load(GAMMA + c_offs).to(tl.float32)
        beta = tl.load(BETA + c_offs).to(tl.float32)
        normed = normed * gamma[None, :] + beta[None, :]
    normed_bf16 = normed.to(PADDED_OUT.dtype.element_ty).to(tl.float32)
    act = (normed_bf16 * tl.sigmoid(normed_bf16)).to(PADDED_OUT.dtype.element_ty)

    if write_hist:
        tl.store(NEXT_HIST + nh_bt_base + in_off, act, mask=mask_hw[:, None])
    if write_padded:
        if PAD_SPATIAL:
            _store_with_spatial_pad1(
                PADDED_OUT,
                out_bt_base,
                y,
                x,
                c_offs,
                act,
                mask_hw,
                H,
                W,
                C,
                REFLECT,
            )
        else:
            tl.store(PADDED_OUT + out_bt_base + in_off, act, mask=mask_hw[:, None])


@triton.jit
def _residual_bias_downsample_pad_kernel(
    X,
    RESIDUAL,
    ADD_BIAS,
    PREV_HIST,
    PADDED_OUT,
    NEXT_HIST,
    T_IN: tl.constexpr,
    HIST: tl.constexpr,
    OFFSET: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    PAD_BR: tl.constexpr,
    REFLECT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_ADD_BIAS: tl.constexpr,
    HAS_PREV: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    C_BLK: tl.constexpr,
):
    bs = tl.program_id(0).to(tl.int64)
    hw_blk = tl.program_id(1)
    c_blk_id = tl.program_id(2)

    S_TOTAL: tl.constexpr = HIST + T_IN
    T_OUT: tl.constexpr = HIST + T_IN - OFFSET
    HW: tl.constexpr = H * W
    H_OUT: tl.constexpr = H + PAD_BR
    W_OUT: tl.constexpr = W + PAD_BR

    b = bs // S_TOTAL
    s = (bs % S_TOTAL).to(tl.int32)

    write_padded = s >= OFFSET
    write_hist = (HIST > 0) & (s >= T_IN)
    if (write_padded | write_hist) == 0:
        return

    hw = hw_blk * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = hw < HW
    y = hw // W
    x = hw % W
    c_offs = c_blk_id * C_BLK + tl.arange(0, C_BLK)
    in_off = hw.to(tl.int64)[:, None] * C + c_offs[None, :]

    if s < HIST:
        if HAS_PREV:
            prev_bt_base = (b * HIST + s) * (HW * C)
            val = tl.load(
                PREV_HIST + prev_bt_base + in_off,
                mask=mask_hw[:, None],
                other=0.0,
            )
        else:
            val = tl.zeros((BLOCK_HW, C_BLK), dtype=PADDED_OUT.dtype.element_ty)
    else:
        t = s - HIST
        x_bt_base = (b * T_IN + t) * (HW * C)
        raw = tl.load(X + x_bt_base + in_off, mask=mask_hw[:, None], other=0.0).to(
            tl.float32
        )
        if HAS_RESIDUAL:
            res = tl.load(
                RESIDUAL + x_bt_base + in_off, mask=mask_hw[:, None], other=0.0
            ).to(tl.float32)
            raw = raw + res
        if HAS_ADD_BIAS:
            add_b = tl.load(ADD_BIAS + c_offs).to(tl.float32)
            raw = raw + add_b[None, :]
        val = raw.to(PADDED_OUT.dtype.element_ty)

    if write_hist:
        nh_bt_base = (b * HIST + (s - T_IN)) * (HW * C)
        tl.store(NEXT_HIST + nh_bt_base + in_off, val, mask=mask_hw[:, None])

    if write_padded:
        out_bt_base = (b * T_OUT + (s - OFFSET)) * (H_OUT * W_OUT * C)
        py = y.to(tl.int64)
        px = x.to(tl.int64)
        out_off = (py[:, None] * W_OUT + px[:, None]) * C + c_offs[None, :]
        tl.store(PADDED_OUT + out_bt_base + out_off, val, mask=mask_hw[:, None])

        if PAD_BR == 1:
            if REFLECT:
                # Bottom row py = H mirrors y = H - 2
                m_bot = mask_hw & (y == H - 2)
                off_bot = (H * W_OUT + px)[:, None] * C + c_offs[None, :]
                tl.store(PADDED_OUT + out_bt_base + off_bot, val, mask=m_bot[:, None])

                # Right col px = W mirrors x = W - 2
                m_right = mask_hw & (x == W - 2)
                off_right = (py * W_OUT + W)[:, None] * C + c_offs[None, :]
                tl.store(
                    PADDED_OUT + out_bt_base + off_right, val, mask=m_right[:, None]
                )

                # Bottom-right corner (H, W) mirrors (H - 2, W - 2)
                m_br = m_bot & (x == W - 2)
                off_br = (py * 0 + (H * W_OUT + W))[:, None] * C + c_offs[None, :]
                tl.store(PADDED_OUT + out_bt_base + off_br, val, mask=m_br[:, None])
            else:
                zero_val = tl.zeros(val.shape, dtype=val.dtype)
                m_bot = mask_hw & (y == H - 1)
                off_bot = (H * W_OUT + px)[:, None] * C + c_offs[None, :]
                tl.store(
                    PADDED_OUT + out_bt_base + off_bot, zero_val, mask=m_bot[:, None]
                )
                m_right = mask_hw & (x == W - 1)
                off_right = (py * W_OUT + W)[:, None] * C + c_offs[None, :]
                tl.store(
                    PADDED_OUT + out_bt_base + off_right,
                    zero_val,
                    mask=m_right[:, None],
                )
                m_br = m_bot & (x == W - 1)
                off_br = (py * 0 + (H * W_OUT + W))[:, None] * C + c_offs[None, :]
                tl.store(
                    PADDED_OUT + out_bt_base + off_br, zero_val, mask=m_br[:, None]
                )


@triton.jit
def _fused_add_bias_inplace_kernel(
    Y,
    RESIDUAL,
    BIAS,
    N_PIXELS,
    C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    C_BLK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    p_blk = tl.program_id(0).to(tl.int64)
    c_blk = tl.program_id(1)

    p = p_blk * BLOCK_P + tl.arange(0, BLOCK_P).to(tl.int64)
    mask_p = p < N_PIXELS
    c_offs = c_blk * C_BLK + tl.arange(0, C_BLK)
    off = p[:, None] * C + c_offs[None, :]

    y = tl.load(Y + off, mask=mask_p[:, None], other=0.0).to(tl.float32)
    r = tl.load(RESIDUAL + off, mask=mask_p[:, None], other=0.0).to(tl.float32)
    out = y + r
    if HAS_BIAS:
        b = tl.load(BIAS + c_offs).to(tl.float32)
        out = out + b[None, :]
    tl.store(Y + off, out.to(Y.dtype.element_ty), mask=mask_p[:, None])


@triton.jit
def _tile_blend_nthwc_to_ncdhw_kernel(
    TILES_NTHWC,
    IDX_CUR,
    IDX_UP,
    IDX_LEFT,
    WY0,
    WY1,
    WX0,
    WX1,
    HAS_Y,
    HAS_X,
    OUT_NCDHW,
    T_FULL: tl.constexpr,
    T_VALID: tl.constexpr,
    TILE_HW: tl.constexpr,
    OUT_HW: tl.constexpr,
    C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Input TILES_NTHWC has shape [NUM_TILES, T_FULL, TILE_H, TILE_W, C].
    # IDX_* encodes (tile_id * TILE_HW + local_hw).
    t = tl.program_id(0)
    p_blk = tl.program_id(1)
    c_blk = tl.program_id(2)

    p = p_blk * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = p < OUT_HW
    c = c_blk * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c < C
    mask = mask_p[:, None] & mask_c[None, :]

    i_cur = tl.load(IDX_CUR + p, mask=mask_p, other=0)
    tile_cur = i_cur // TILE_HW
    hw_cur = i_cur % TILE_HW
    off_cur = ((tile_cur.to(tl.int64) * T_FULL + t) * TILE_HW + hw_cur)[
        :, None
    ] * C + c[None, :]
    v_cur = tl.load(TILES_NTHWC + off_cur, mask=mask, other=0.0)

    hy = tl.load(HAS_Y + p, mask=mask_p, other=False)
    i_up = tl.load(IDX_UP + p, mask=mask_p & hy, other=0)
    tile_up = i_up // TILE_HW
    hw_up = i_up % TILE_HW
    off_up = ((tile_up.to(tl.int64) * T_FULL + t) * TILE_HW + hw_up)[:, None] * C + c[
        None, :
    ]
    v_up = tl.load(TILES_NTHWC + off_up, mask=mask & hy[:, None], other=0.0)
    wy0 = tl.load(WY0 + p, mask=mask_p, other=0.0)
    wy1 = tl.load(WY1 + p, mask=mask_p, other=1.0)
    # Match klvae.blend BF16 rounding exactly: a * w_a + b * w_b in BF16
    prod_y0 = (v_up.to(tl.float32) * wy0[:, None].to(tl.float32)).to(v_cur.dtype)
    prod_y1 = (v_cur.to(tl.float32) * wy1[:, None].to(tl.float32)).to(v_cur.dtype)
    blended_y = (prod_y0.to(tl.float32) + prod_y1.to(tl.float32)).to(v_cur.dtype)
    v_y = tl.where(hy[:, None], blended_y, v_cur)

    hx = tl.load(HAS_X + p, mask=mask_p, other=False)
    i_left = tl.load(IDX_LEFT + p, mask=mask_p & hx, other=0)
    tile_left = i_left // TILE_HW
    hw_left = i_left % TILE_HW
    off_left = ((tile_left.to(tl.int64) * T_FULL + t) * TILE_HW + hw_left)[
        :, None
    ] * C + c[None, :]
    v_left = tl.load(TILES_NTHWC + off_left, mask=mask & hx[:, None], other=0.0)
    wx0 = tl.load(WX0 + p, mask=mask_p, other=0.0)
    wx1 = tl.load(WX1 + p, mask=mask_p, other=1.0)
    prod_x0 = (v_left.to(tl.float32) * wx0[:, None].to(tl.float32)).to(v_cur.dtype)
    prod_x1 = (v_y.to(tl.float32) * wx1[:, None].to(tl.float32)).to(v_cur.dtype)
    blended_xy = (prod_x0.to(tl.float32) + prod_x1.to(tl.float32)).to(v_cur.dtype)
    v_final = tl.where(hx[:, None], blended_xy, v_y)

    # Store directly into contiguous NCDHW [1, C, T_VALID, OUT_H, OUT_W]
    out_off = (c.to(tl.int64)[None, :] * T_VALID + t) * OUT_HW + p.to(tl.int64)[:, None]
    tl.store(OUT_NCDHW + out_off, v_final, mask=mask)


def tile_pack_and_pad_uint8(
    frames: torch.Tensor,
    prev_hist: torch.Tensor | None,
    tile_y0: torch.Tensor,
    tile_x0: torch.Tensor,
    *,
    needed_frames: int,
    history: int,
    offset: int,
    tile_h: int,
    tile_w: int,
    pad_h: int,
    pad_w: int,
    reflect: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    num_tiles = int(tile_y0.numel())
    t_valid = int(frames.shape[1])
    t_out = history + needed_frames - offset
    h_out = tile_h + 2 * pad_h
    w_out = tile_w + 2 * pad_w
    device = frames.device

    padded = torch.empty(
        (num_tiles, t_out, h_out, w_out, 3), dtype=dtype, device=device
    )
    next_hist = (
        torch.empty((num_tiles, history, tile_h, tile_w, 3), dtype=dtype, device=device)
        if history > 0
        else None
    )
    means_q = [float(torch.tensor(m, dtype=dtype).item()) for m in IMAGENET_MEAN]
    stds_q = [float(torch.tensor(s, dtype=dtype).item()) for s in IMAGENET_STD]
    block_w = min(256, triton.next_power_of_2(w_out))

    _tile_pack_pad_uint8_kernel[(num_tiles, history + needed_frames, h_out)](
        frames,
        prev_hist if prev_hist is not None else padded,
        tile_y0,
        tile_x0,
        padded,
        next_hist if next_hist is not None else padded,
        F_S0=frames.stride(0),
        F_S1=frames.stride(1),
        F_S2=frames.stride(2),
        F_S3=frames.stride(3),
        T_VALID=t_valid,
        T_NEEDED=needed_frames,
        HIST=history,
        OFFSET=offset,
        H_TILE=tile_h,
        W_TILE=tile_w,
        PH=pad_h,
        PW=pad_w,
        REFLECT=reflect,
        HAS_PREV=prev_hist is not None,
        M0=means_q[0],
        M1=means_q[1],
        M2=means_q[2],
        S0=stds_q[0],
        S1=stds_q[1],
        S2=stds_q[2],
        BLOCK_W=block_w,
        num_warps=4,
    )
    return padded, next_hist


def tile_pack_and_pad_bf16(
    x: torch.Tensor,
    prev_hist: torch.Tensor | None,
    tile_y0: torch.Tensor,
    tile_x0: torch.Tensor,
    *,
    history: int,
    offset: int,
    tile_h: int,
    tile_w: int,
    pad_h: int,
    pad_w: int,
    reflect: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    n_in, _, t_in, _, _ = x.shape
    num_tiles = int(tile_y0.numel())
    bn_total = num_tiles * n_in
    t_out = history + t_in - offset
    h_out = tile_h + 2 * pad_h
    w_out = tile_w + 2 * pad_w
    device = x.device
    dtype = x.dtype

    padded = torch.empty((bn_total, t_out, h_out, w_out, 3), dtype=dtype, device=device)
    next_hist = (
        torch.empty((bn_total, history, tile_h, tile_w, 3), dtype=dtype, device=device)
        if history > 0
        else None
    )
    block_w = min(256, triton.next_power_of_2(w_out))

    _tile_pack_pad_bf16_kernel[(bn_total, history + t_in, h_out)](
        x,
        prev_hist if prev_hist is not None else padded,
        tile_y0,
        tile_x0,
        padded,
        next_hist if next_hist is not None else padded,
        X_S0=x.stride(0),
        X_S1=x.stride(1),
        X_S2=x.stride(2),
        X_S3=x.stride(3),
        X_S4=x.stride(4),
        N_IN=n_in,
        T_IN=t_in,
        HIST=history,
        OFFSET=offset,
        H_TILE=tile_h,
        W_TILE=tile_w,
        PH=pad_h,
        PW=pad_w,
        REFLECT=reflect,
        HAS_PREV=prev_hist is not None,
        BLOCK_W=block_w,
        num_warps=4,
    )
    return padded, next_hist


def fused_gn_silu_history_pad_nthwc(
    x_nthwc: torch.Tensor,
    norm: torch.nn.Module,
    *,
    prev_hist: torch.Tensor | None,
    history: int,
    offset: int,
    pad_spatial: bool,
    reflect: bool,
    add_bias: torch.Tensor | None = None,
    residual_nthwc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    b, t_in, h, w, c = x_nthwc.shape
    hw = h * w
    groups = int(norm.num_groups)
    d = c // groups
    eps = float(norm.eps)
    affine = bool(norm.affine)
    t_out = history + t_in - offset
    h_out = h + 2 if pad_spatial else h
    w_out = w + 2 if pad_spatial else w
    device = x_nthwc.device
    dtype = x_nthwc.dtype

    padded = torch.empty((b, t_out, h_out, w_out, c), dtype=dtype, device=device)
    next_hist = (
        torch.empty((b, history, h, w, c), dtype=dtype, device=device)
        if history > 0
        else None
    )

    dummy = padded
    gamma = norm.weight if affine else dummy
    beta = norm.bias if affine else dummy

    # Single-pass L1-resident kernel for spatial sizes up to 64x64 (HW <= 4096)
    if hw <= 4096:
        g_per_cta = min(groups, max(1, 64 // d))
        while groups % g_per_cta != 0:
            g_per_cta //= 2
        block_hw = min(64, max(16, triton.next_power_of_2(hw)))
        warps = 4

        grid = (b * (history + t_in), groups // g_per_cta)
        _gn_silu_pad_single_pass_kernel[grid](
            x_nthwc,
            residual_nthwc if residual_nthwc is not None else dummy,
            add_bias if add_bias is not None else dummy,
            prev_hist if prev_hist is not None else dummy,
            gamma,
            beta,
            padded,
            next_hist if next_hist is not None else dummy,
            T_IN=t_in,
            HIST=history,
            OFFSET=offset,
            H=h,
            W=w,
            C=c,
            GROUPS=groups,
            G_PER_CTA=g_per_cta,
            BLOCK_HW=block_hw,
            EPS=eps,
            HAS_RESIDUAL=residual_nthwc is not None,
            HAS_ADD_BIAS=add_bias is not None,
            HAS_PREV=prev_hist is not None,
            AFFINE=affine,
            PAD_SPATIAL=pad_spatial,
            REFLECT=reflect,
            num_warps=warps,
        )
        return padded, next_hist

    # Large spatial planes (128x128, 256x256): 2-kernel 100%-coalesced HBM3 path
    num_parts = 32 if hw >= 32768 else 16
    pixels_per_part = triton.cdiv(hw, num_parts)
    block_hw_reduce = 64
    c_step = max(d, min(128, c & -c))
    bt_total = b * t_in
    needed_elems = bt_total * groups * num_parts
    p_mean = torch.empty((needed_elems,), dtype=torch.float32, device=device)
    p_m2 = torch.empty((needed_elems,), dtype=torch.float32, device=device)

    _gn_partials_coalesced_kernel[(bt_total, num_parts)](
        x_nthwc,
        residual_nthwc if residual_nthwc is not None else dummy,
        add_bias if add_bias is not None else dummy,
        p_mean,
        p_m2,
        HW=hw,
        C=c,
        GROUPS=groups,
        NUM_PARTS=num_parts,
        PIXELS_PER_PART=pixels_per_part,
        BLOCK_HW=block_hw_reduce,
        C_STEP=c_step,
        HAS_RESIDUAL=residual_nthwc is not None,
        HAS_ADD_BIAS=add_bias is not None,
        num_warps=4,
    )

    block_hw_apply = 64
    c_blk = max(d, min(128, c & -c))
    grid_apply = (
        b * (history + t_in),
        triton.cdiv(hw, block_hw_apply),
        c // c_blk,
    )
    _gn_apply_silu_pad_coalesced_kernel[grid_apply](
        x_nthwc,
        residual_nthwc if residual_nthwc is not None else dummy,
        add_bias if add_bias is not None else dummy,
        prev_hist if prev_hist is not None else dummy,
        gamma,
        beta,
        p_mean,
        p_m2,
        padded,
        next_hist if next_hist is not None else dummy,
        T_IN=t_in,
        HIST=history,
        OFFSET=offset,
        H=h,
        W=w,
        C=c,
        GROUPS=groups,
        NUM_PARTS=num_parts,
        PIXELS_PER_PART=pixels_per_part,
        BLOCK_HW=block_hw_apply,
        C_BLK=c_blk,
        EPS=eps,
        HAS_RESIDUAL=residual_nthwc is not None,
        HAS_ADD_BIAS=add_bias is not None,
        HAS_PREV=prev_hist is not None,
        AFFINE=affine,
        PAD_SPATIAL=pad_spatial,
        REFLECT=reflect,
        num_warps=4,
    )
    return padded, next_hist


def fused_residual_bias_downsample_pad_nthwc(
    x_nthwc: torch.Tensor,
    *,
    residual_nthwc: torch.Tensor | None,
    add_bias: torch.Tensor | None,
    prev_hist: torch.Tensor | None,
    history: int,
    offset: int,
    pad_bottom_right: bool,
    reflect: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    b, t_in, h, w, c = x_nthwc.shape
    hw = h * w
    pad_br = 1 if pad_bottom_right else 0
    t_out = history + t_in - offset
    h_out = h + pad_br
    w_out = w + pad_br
    device = x_nthwc.device
    dtype = x_nthwc.dtype

    padded = torch.empty((b, t_out, h_out, w_out, c), dtype=dtype, device=device)
    next_hist = (
        torch.empty((b, history, h, w, c), dtype=dtype, device=device)
        if history > 0
        else None
    )
    dummy = padded
    block_hw = min(64, max(16, triton.next_power_of_2(hw)))
    c_blk = min(128, c & -c)
    grid = (b * (history + t_in), triton.cdiv(hw, block_hw), c // c_blk)

    _residual_bias_downsample_pad_kernel[grid](
        x_nthwc,
        residual_nthwc if residual_nthwc is not None else dummy,
        add_bias if add_bias is not None else dummy,
        prev_hist if prev_hist is not None else dummy,
        padded,
        next_hist if next_hist is not None else dummy,
        T_IN=t_in,
        HIST=history,
        OFFSET=offset,
        H=h,
        W=w,
        C=c,
        PAD_BR=pad_br,
        REFLECT=reflect,
        HAS_RESIDUAL=residual_nthwc is not None,
        HAS_ADD_BIAS=add_bias is not None,
        HAS_PREV=prev_hist is not None,
        BLOCK_HW=block_hw,
        C_BLK=c_blk,
        num_warps=4,
    )
    return padded, next_hist


def fused_add_bias_inplace_nthwc(
    y_nthwc: torch.Tensor,
    residual_nthwc: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    c = y_nthwc.shape[-1]
    n_pixels = y_nthwc.numel() // c
    block_p = 64
    c_blk = min(128, c & -c)
    grid = (triton.cdiv(n_pixels, block_p), c // c_blk)
    _fused_add_bias_inplace_kernel[grid](
        y_nthwc,
        residual_nthwc,
        bias if bias is not None else y_nthwc,
        n_pixels,
        C=c,
        BLOCK_P=block_p,
        C_BLK=c_blk,
        HAS_BIAS=bias is not None,
        num_warps=4,
    )
    return y_nthwc


def fused_tile_blend_nthwc_to_ncdhw(
    tiles_nthwc: torch.Tensor,
    lut,
    *,
    valid_latents: int | None = None,
) -> torch.Tensor:
    # tiles_nthwc: [num_tiles, t_full, tile_h, tile_w, c]
    _, t_full, tile_h, tile_w, c = tiles_nthwc.shape
    t_valid = t_full if valid_latents is None else int(valid_latents)
    out = torch.empty(
        (1, c, t_valid, lut.out_h, lut.out_w),
        dtype=tiles_nthwc.dtype,
        device=tiles_nthwc.device,
    )
    out_hw = lut.out_h * lut.out_w
    block_p = 64
    block_c = triton.next_power_of_2(c)
    grid = (t_valid, triton.cdiv(out_hw, block_p), triton.cdiv(c, block_c))
    _tile_blend_nthwc_to_ncdhw_kernel[grid](
        tiles_nthwc,
        lut.idx_cur,
        lut.idx_up,
        lut.idx_left,
        lut.wy0,
        lut.wy1,
        lut.wx0,
        lut.wx1,
        lut.has_y,
        lut.has_x,
        out,
        T_FULL=t_full,
        T_VALID=t_valid,
        TILE_HW=tile_h * tile_w,
        OUT_HW=out_hw,
        C=c,
        BLOCK_P=block_p,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out
