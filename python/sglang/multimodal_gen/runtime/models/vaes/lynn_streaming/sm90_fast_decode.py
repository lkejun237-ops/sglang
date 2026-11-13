# SPDX-License-Identifier: Apache-2.0
"""SM90 (Hopper H100) fast single-GPU decode engine for the Zing H3 Lynn StreamingDecoder.

Key optimizations while preserving causal tile streaming semantics:
1. Exact unique-grid head (`post_quant_conv` + `proj_in`) & Layer-0 QKV, eliminating
   redundant boundary patch work on step input.
2. Offline interleaved-pair Q/K channel permutation + `uint32` (`bf16x2`) coalesced
   RMSNorm + half-split 3D RoPE + 4-slot K / 6-slot V zero-copy sliding rings.
3. Shallow Full-Tile layers (`0 .. fast_start - 1`) preserve exact per-tile 2D coordinate
   anchors relative to `(0, 0)` register tokens; deep semantic layers (`fast_start .. L - 1`)
   and all FFNs + final `norm_out` / `proj_out` execute on the spatial-unique grid weighted
   by the exact bilinear weights of `StreamingDecoder._stitch_tiles`.
4. SM90 WGMMA FP8 (`e4m3fn`) projections (`torch._scaled_mm`) + Hopper FP8 FlashAttention-3
   (`sgl_kernel.flash_attn.flash_attn_with_kvcache`) captured into a 6-mode CUDA Graph pool.
"""

from importlib.util import find_spec

import torch
import torch.nn.functional as F
from torch import nn

from .execution import parameter_signature
from .geometry import split_tiles

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only CI lacks Triton
    triton = None
    tl = None
    _HAS_TRITON = False


def _is_plain_linear(module) -> bool:
    return (
        type(module) is nn.Linear
        and not module._forward_hooks
        and not module._forward_pre_hooks
        and "forward" not in module.__dict__
    )


def _ffn_modules(block):
    # The engine replaces these module calls as well as their Linear children.
    return (block.ff, block.ff.net, *block.ff.net)


def _has_custom_forward(module) -> bool:
    return bool(
        module._forward_hooks
        or module._forward_pre_hooks
        or "forward" in module.__dict__
    )


def _is_plain_rms_norm(module, width: int) -> bool:
    return (
        type(module) is nn.RMSNorm
        and module.eps is not None
        and tuple(module.normalized_shape) == (width,)
        and getattr(module, "bias", None) is None
        and not module._forward_hooks
        and not module._forward_pre_hooks
        and "forward" not in module.__dict__
    )


def _is_plain_layer_norm(module, width: int) -> bool:
    return (
        type(module) is nn.LayerNorm
        and module.eps is not None
        and tuple(module.normalized_shape) == (width,)
        and not module._forward_hooks
        and not module._forward_pre_hooks
        and "forward" not in module.__dict__
    )


def _has_sm90_fa3() -> bool:
    if not _HAS_TRITON or find_spec("sgl_kernel") is None:
        return False
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: F401
    except Exception:
        return False
    return True


def _supports_sm90_model(model, device: torch.device) -> bool:
    if (
        not _HAS_TRITON
        or device.type != "cuda"
        or torch.cuda.get_device_capability(device) != (9, 0)
        or not _has_sm90_fa3()
        or getattr(model, "compute_dtype", None) != torch.bfloat16
    ):
        return False

    cfg = model.config
    arch = model.architecture
    proto = model.protocol
    dec = model.decoder
    ratio = model.spatial_ratio
    if (
        cfg.temporal_window != 3
        or proto.teacher_window < 1
        or proto.phases < 1
        or proto.ratio != 4
        or proto.pre_padding != 3
        or ratio < 1
        or cfg.tile_size // ratio != 16
        or cfg.tile_size % ratio != 0
        or cfg.tile_overlap % ratio != 0
        or arch.get("latent_channels") != 24
        or arch.get("out_channels") != 3
        or arch.get("decoder_num_attention_heads") != 32
        or arch.get("decoder_attention_head_dim") != 64
        or arch.get("decoder_num_register_tokens") != 4
        or arch.get("decoder_ffn_mult") != 4
        or arch.get("decoder_rope_dim_ratio") != 0.75
        or len(dec.transformer_blocks) < 1
    ):
        return False

    conv = model.post_quant_conv
    if (
        type(conv) is not nn.Conv3d
        or conv.kernel_size != (1, 1, 1)
        or conv.stride != (1, 1, 1)
        or conv.padding != (0, 0, 0)
        or conv.groups != 1
        or conv._forward_hooks
        or conv._forward_pre_hooks
        or "forward" in conv.__dict__
    ):
        return False
    if not (_is_plain_linear(dec.proj_in) and _is_plain_linear(dec.proj_out)):
        return False
    if not _is_plain_layer_norm(dec.norm_out, 2048):
        return False

    for blk in dec.transformer_blocks:
        attn = blk.attn
        if any(_has_custom_forward(m) for m in _ffn_modules(blk)):
            return False
        if not (
            _is_plain_rms_norm(blk.norm1, 2048)
            and _is_plain_rms_norm(blk.norm2, 2048)
            and _is_plain_rms_norm(attn.norm_q, 64)
            and _is_plain_rms_norm(attn.norm_k, 64)
            and _is_plain_linear(attn.to_q)
            and _is_plain_linear(attn.to_k)
            and _is_plain_linear(attn.to_v)
            and len(attn.to_out) >= 1
            and _is_plain_linear(attn.to_out[0])
            and len(blk.ff.net) >= 3
            and blk.ff.net[0].__class__.__name__ == "SwiGLU"
            and hasattr(blk.ff.net[0], "proj")
            and _is_plain_linear(blk.ff.net[0].proj)
            and _is_plain_linear(blk.ff.net[2])
        ):
            return False
    return True


def _supports_step_input(device: torch.device, z_t: torch.Tensor) -> bool:
    return (
        not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and z_t.is_cuda
        and z_t.device == device
        and z_t.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and z_t.shape[0] >= 1
        and z_t.shape[-2] >= 16
        and z_t.shape[-1] >= 16
    )


def _make_qk_perm64(device: torch.device) -> torch.Tensor:
    perm = []
    for j in range(24):
        perm.append(j)
        perm.append(j + 24)
    for d in range(48, 64):
        perm.append(d)
    return torch.tensor(perm, dtype=torch.long, device=device)


def _pack_rope_tables_interleaved(
    cq: torch.Tensor, sq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    length = cq.shape[0]
    device = cq.device
    ce = torch.ones((length, 32), device=device, dtype=torch.bfloat16)
    co = torch.ones((length, 32), device=device, dtype=torch.bfloat16)
    se = torch.zeros((length, 32), device=device, dtype=torch.bfloat16)
    so = torch.zeros((length, 32), device=device, dtype=torch.bfloat16)
    ce[:, :24] = cq[:, :24].to(torch.bfloat16)
    co[:, :24] = cq[:, 24:48].to(torch.bfloat16)
    se[:, :24] = sq[:, :24].to(torch.bfloat16)
    so[:, :24] = sq[:, 24:48].to(torch.bfloat16)
    return ce.contiguous(), co.contiguous(), se.contiguous(), so.contiguous()


def _quantize_linear_colmajor(
    linear: nn.Linear, scale_mul: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wf = linear.weight.detach().float()
    ws = wf.abs().amax(dim=1).clamp_min_(1e-12) / 448.0
    wq = (
        (wf / ws[:, None])
        .clamp_(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .contiguous()
        .t()
    )
    bf = (
        linear.bias.detach().float()
        if linear.bias is not None
        else torch.zeros(wf.shape[0], device=wf.device, dtype=torch.float32)
    )
    if scale_mul is not None:
        sm = scale_mul.detach().float()
        ws = ws * sm
        bf = bf * sm
    return wq, ws.contiguous(), bf.to(torch.bfloat16).contiguous()


def _build_axis_stitch_map(
    starts, sizes, overlaps
) -> tuple[torch.Tensor, torch.Tensor]:
    idx_rows = []
    w_rows = []
    num_tiles = len(starts)
    for k in range(num_tiles):
        ov_left = overlaps[k - 1] if k > 0 else 0
        ov_right = overlaps[k] if k < num_tiles - 1 else 0
        valid_len = sizes[k] - ov_right
        for loc in range(valid_len):
            if loc < ov_left:
                prev_loc = sizes[k - 1] - ov_left + loc
                w = float(loc) / float(ov_left)
                idx_rows.append([k - 1, prev_loc, k, loc])
                w_rows.append(w)
            else:
                idx_rows.append([k, loc, k, loc])
                w_rows.append(1.0)
    return (
        torch.tensor(idx_rows, dtype=torch.int32),
        torch.tensor(w_rows, dtype=torch.float32),
    )


def _build_unique_tile_maps(
    batch_n: int,
    h_lat: int,
    w_lat: int,
    ys: list[int],
    hs: list[int],
    oy: list[int],
    xs: list[int],
    ws: list[int],
    ox: list[int],
    ratio: int,
    s_suf: int,
    device: torch.device,
) -> dict[str, int | torch.Tensor]:
    t_lat = hs[0] // ratio
    s_tile = t_lat * t_lat
    s_full = s_tile + s_suf
    ty_cnt, tx_cnt = len(ys), len(xs)
    tiles = ty_cnt * tx_cnt
    total_b = batch_n * tiles
    u_sp = batch_n * h_lat * w_lat
    u_total = u_sp + total_b * s_suf

    tile_to_u_sp = torch.empty((total_b * s_tile,), dtype=torch.int32)
    tile_to_u_full = torch.empty((total_b * s_full,), dtype=torch.int32)

    for n in range(batch_n):
        for ty in range(ty_cnt):
            y0 = ys[ty] // ratio
            for tx in range(tx_cnt):
                x0 = xs[tx] // ratio
                b = n * tiles + ty * tx_cnt + tx
                for dy in range(t_lat):
                    for dx in range(t_lat):
                        tok = dy * t_lat + dx
                        u = (n * h_lat + (y0 + dy)) * w_lat + (x0 + dx)
                        tile_to_u_sp[b * s_tile + tok] = u
                        tile_to_u_full[b * s_full + tok] = u
                for s in range(s_suf):
                    tile_to_u_full[b * s_full + s_tile + s] = u_sp + b * s_suf + s

    y_idx_pix, y_w_pix = _build_axis_stitch_map(ys, hs, oy)
    x_idx_pix, x_w_pix = _build_axis_stitch_map(xs, ws, ox)

    blend_src_full = torch.zeros((u_total, 3), dtype=torch.int32)
    blend_src_sp = torch.zeros((u_sp, 3), dtype=torch.int32)
    blend_w = torch.zeros((u_total, 3), dtype=torch.float32)

    for n in range(batch_n):
        for y_l in range(h_lat):
            ty0 = int(y_idx_pix[y_l * ratio, 0])
            dy0 = int(y_idx_pix[y_l * ratio, 1]) // ratio
            ty1 = int(y_idx_pix[y_l * ratio, 2])
            dy1 = int(y_idx_pix[y_l * ratio, 3]) // ratio
            wy = float(y_w_pix[y_l * ratio : (y_l + 1) * ratio].mean().item())

            for x_l in range(w_lat):
                tx0 = int(x_idx_pix[x_l * ratio, 0])
                dx0 = int(x_idx_pix[x_l * ratio, 1]) // ratio
                tx1 = int(x_idx_pix[x_l * ratio, 2])
                dx1 = int(x_idx_pix[x_l * ratio, 3]) // ratio
                wx = float(x_w_pix[x_l * ratio : (x_l + 1) * ratio].mean().item())

                u = (n * h_lat + y_l) * w_lat + x_l
                cands = [(n * tiles + ty1 * tx_cnt + tx1, dy1 * t_lat + dx1, wx * wy)]
                if wy < 1.0 and ty0 != ty1:
                    cands.append(
                        (
                            n * tiles + ty0 * tx_cnt + tx1,
                            dy0 * t_lat + dx1,
                            wx * (1.0 - wy),
                        )
                    )
                if wx < 1.0 and tx0 != tx1:
                    cands.append(
                        (
                            n * tiles + ty1 * tx_cnt + tx0,
                            dy1 * t_lat + dx0,
                            1.0 - wx,
                        )
                    )

                w_sum = sum(c[2] for c in cands)
                f0 = cands[0][0] * s_full + cands[0][1]
                s0 = cands[0][0] * s_tile + cands[0][1]
                for k in range(3):
                    if k < len(cands):
                        tk, pk, wk = cands[k]
                        blend_src_full[u, k] = tk * s_full + pk
                        blend_src_sp[u, k] = tk * s_tile + pk
                        blend_w[u, k] = wk / w_sum
                    else:
                        blend_src_full[u, k] = f0
                        blend_src_sp[u, k] = s0
                        blend_w[u, k] = 0.0

    for b in range(total_b):
        for s in range(s_suf):
            u = u_sp + b * s_suf + s
            idx_f = b * s_full + s_tile + s
            blend_src_full[u, 0] = idx_f
            blend_src_full[u, 1] = idx_f
            blend_src_full[u, 2] = idx_f
            blend_w[u, 0] = 1.0
            blend_w[u, 1] = 0.0
            blend_w[u, 2] = 0.0

    return {
        "U_sp": u_sp,
        "U_total": u_total,
        "tile_to_u_sp": tile_to_u_sp.to(device).contiguous(),
        "tile_to_u_full": tile_to_u_full.to(device).contiguous(),
        "blend_src_full": blend_src_full.to(device).contiguous(),
        "blend_src_sp": blend_src_sp.to(device).contiguous(),
        "blend_w": blend_w.to(device).contiguous(),
    }


if _HAS_TRITON:

    @triton.jit
    def _pack_bf16_pair_to_u32(ae_bf16, ao_bf16):
        lo = ae_bf16.to(tl.float32).to(tl.uint32, bitcast=True) >> 16
        hi = ao_bf16.to(tl.float32).to(tl.uint32, bitcast=True) & 0xFFFF0000
        return lo | hi

    @triton.jit
    def _unpack_u32_to_fp32_pair(u):
        ae = (u << 16).to(tl.float32, bitcast=True)
        ao = (u & 0xFFFF0000).to(tl.float32, bitcast=True)
        return ae, ao

    @triton.jit
    def _store_bf16_pair_to_fp8_u16(
        ptr_u16, off, ae_bf16, ao_bf16, fp8_dtype: tl.constexpr
    ):
        fe = tl.clamp(ae_bf16.to(tl.float32), -448.0, 448.0).to(fp8_dtype)
        fo = tl.clamp(ao_bf16.to(tl.float32), -448.0, 448.0).to(fp8_dtype)
        lo8 = fe.to(tl.int8, bitcast=True).to(tl.uint16) & 0xFF
        hi8 = fo.to(tl.int8, bitcast=True).to(tl.uint16) << 8
        tl.store(ptr_u16 + off, lo8 | hi8)

    @triton.jit
    def _denorm_latent_bf16_kernel(
        z_in,
        mean_ptr,
        std_ptr,
        z_out,
        hw: tl.constexpr,
        block: tl.constexpr,
    ):
        pid_nc = tl.program_id(0)
        pid_b = tl.program_id(1)
        c = pid_nc % 24
        offs = pid_b * block + tl.arange(0, block)
        mask = offs < hw
        mean = tl.load(mean_ptr + c)
        std = tl.load(std_ptr + c)
        v = tl.load(z_in + pid_nc * hw + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            z_out + pid_nc * hw + offs, (v * std + mean).to(tl.bfloat16), mask=mask
        )

    @triton.jit
    def _init_tile_backbone_kernel(
        spatial_u,
        tile_to_u_full,
        phase_emb,
        phase_idx,
        reg_tokens,
        hidden_t,
    ):
        row = tl.program_id(0)
        tok = row % 261
        cols = tl.arange(0, 2048)
        if tok < 256:
            u_sp = tl.load(tile_to_u_full + row)
            phase = tl.load(phase_idx)
            s = tl.load(spatial_u + u_sp * 2048 + cols).to(tl.float32)
            p = tl.load(phase_emb + phase * 2048 + cols).to(tl.float32)
            h = s + p
        elif tok < 260:
            h = tl.load(reg_tokens + (tok - 256) * 2048 + cols).to(tl.float32)
        else:
            h = tl.zeros([2048], dtype=tl.float32)
        tl.store(hidden_t + row * 2048 + cols, h.to(tl.bfloat16))

    @triton.jit
    def _init_l0_unique_norm1_quant_kernel(
        spatial_u,
        phase_emb,
        phase_idx,
        reg_tokens,
        norm_w,
        xq_u,
        xs_u,
        u_sp: tl.constexpr,
        eps: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, 2048)
        if row < u_sp:
            phase = tl.load(phase_idx)
            s = tl.load(spatial_u + row * 2048 + cols).to(tl.float32)
            p = tl.load(phase_emb + phase * 2048 + cols).to(tl.float32)
            h = s + p
        else:
            suf = (row - u_sp) % 5
            if suf < 4:
                h = tl.load(reg_tokens + suf * 2048 + cols).to(tl.float32)
            else:
                h = tl.zeros([2048], dtype=tl.float32)

        hf = h.to(tl.bfloat16).to(tl.float32)
        nw = tl.load(norm_w + cols).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(hf * hf, axis=0) * (1.0 / 2048.0) + eps)
        normed = (hf * rstd * nw).to(tl.bfloat16).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(normed), axis=0), 1e-12)
        scale = amax * (1.0 / 448.0)
        inv_scale = 448.0 / amax
        tl.store(
            xq_u + row * 2048 + cols,
            tl.clamp(normed * inv_scale, -448.0, 448.0).to(xq_u.dtype.element_ty),
        )
        tl.store(xs_u + row, scale)

    @triton.jit
    def _quant_bf16_rows_kernel(x_in, xq_out, xs_out):
        row = tl.program_id(0)
        cols = tl.arange(0, 2048)
        val = tl.load(x_in + row * 2048 + cols).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(val), axis=0), 1e-12)
        scale = amax * (1.0 / 448.0)
        inv_scale = 448.0 / amax
        tl.store(
            xq_out + row * 2048 + cols,
            tl.clamp(val * inv_scale, -448.0, 448.0).to(xq_out.dtype.element_ty),
        )
        tl.store(xs_out + row, scale)

    @triton.jit
    def _qkv_norm_unique_u32_kernel(
        raw_qkv_u32,
        xs_qkv,
        ws_qkv,
        b_qkv_u32,
        nq_we,
        nq_wo,
        nk_we,
        nk_wo,
        q_norm_u32,
        k_norm_u32,
        v_norm_u32,
        k_u_ring_u32,
        u_sp: tl.constexpr,
        k_slot: tl.constexpr,
        skip_suf_q: tl.constexpr,
        has_nq_w: tl.constexpr,
        has_nk_w: tl.constexpr,
        eps_q: tl.constexpr,
        eps_k: tl.constexpr,
    ):
        row = tl.program_id(0)
        hg = tl.program_id(1)
        heads = hg * 8 + tl.arange(0, 8)[:, None]
        j = tl.arange(0, 32)[None, :]
        xs = tl.load(xs_qkv + row)

        row_u32 = raw_qkv_u32 + row * 3072
        h32 = heads * 32 + j
        even_col = heads * 64 + 2 * j
        odd_col = even_col + 1

        vu = tl.load(row_u32 + 2048 + h32)
        ve_raw, vo_raw = _unpack_u32_to_fp32_pair(vu)
        vbu = tl.load(b_qkv_u32 + 2048 + h32)
        vbe, vbo = _unpack_u32_to_fp32_pair(vbu)
        wve = tl.load(ws_qkv + 4096 + even_col)
        wvo = tl.load(ws_qkv + 4096 + odd_col)
        ve_bf16 = (ve_raw * (xs * wve) + vbe).to(tl.bfloat16)
        vo_bf16 = (vo_raw * (xs * wvo) + vbo).to(tl.bfloat16)
        tl.store(
            v_norm_u32 + row * 1024 + h32, _pack_bf16_pair_to_u32(ve_bf16, vo_bf16)
        )

        ku = tl.load(row_u32 + 1024 + h32)
        ke_raw, ko_raw = _unpack_u32_to_fp32_pair(ku)
        kbu = tl.load(b_qkv_u32 + 1024 + h32)
        kbe, kbo = _unpack_u32_to_fp32_pair(kbu)
        wke = tl.load(ws_qkv + 2048 + even_col)
        wko = tl.load(ws_qkv + 2048 + odd_col)
        ke0 = (ke_raw * (xs * wke) + kbe).to(tl.bfloat16).to(tl.float32)
        ko0 = (ko_raw * (xs * wko) + kbo).to(tl.bfloat16).to(tl.float32)
        rstd_k = tl.rsqrt(
            tl.sum(ke0 * ke0 + ko0 * ko0, axis=1, keep_dims=True) * (1.0 / 64.0) + eps_k
        )
        if has_nk_w:
            ke_n = (ke0 * rstd_k * tl.load(nk_we + j)).to(tl.bfloat16)
            ko_n = (ko0 * rstd_k * tl.load(nk_wo + j)).to(tl.bfloat16)
        else:
            ke_n = (ke0 * rstd_k).to(tl.bfloat16)
            ko_n = (ko0 * rstd_k).to(tl.bfloat16)
        k_packed = _pack_bf16_pair_to_u32(ke_n, ko_n)
        tl.store(k_norm_u32 + row * 1024 + h32, k_packed)
        if row < u_sp:
            tl.store(k_u_ring_u32 + (k_slot * u_sp + row) * 1024 + h32, k_packed)

        if (not skip_suf_q) or (row < u_sp):
            qu = tl.load(row_u32 + h32)
            qe_raw, qo_raw = _unpack_u32_to_fp32_pair(qu)
            qbu = tl.load(b_qkv_u32 + h32)
            qbe, qbo = _unpack_u32_to_fp32_pair(qbu)
            wqe = tl.load(ws_qkv + even_col)
            wqo = tl.load(ws_qkv + odd_col)
            qe0 = (qe_raw * (xs * wqe) + qbe).to(tl.bfloat16).to(tl.float32)
            qo0 = (qo_raw * (xs * wqo) + qbo).to(tl.bfloat16).to(tl.float32)
            rstd_q = tl.rsqrt(
                tl.sum(qe0 * qe0 + qo0 * qo0, axis=1, keep_dims=True) * (1.0 / 64.0)
                + eps_q
            )
            if has_nq_w:
                qe_n = (qe0 * rstd_q * tl.load(nq_we + j)).to(tl.bfloat16)
                qo_n = (qo0 * rstd_q * tl.load(nq_wo + j)).to(tl.bfloat16)
            else:
                qe_n = (qe0 * rstd_q).to(tl.bfloat16)
                qo_n = (qo0 * rstd_q).to(tl.bfloat16)
            tl.store(q_norm_u32 + row * 1024 + h32, _pack_bf16_pair_to_u32(qe_n, qo_n))

    @triton.jit
    def _k_hist_from_unique_u32_kernel(
        k_u_ring_u32,
        tile_to_u_sp,
        ce_hist,
        co_hist,
        se_hist,
        so_hist,
        kr_out_u16,
        u_sp: tl.constexpr,
        history: tl.constexpr,
        slot0: tl.constexpr,
        slot1: tl.constexpr,
    ):
        row = tl.program_id(0)
        hg = tl.program_id(1)
        batch = row // history
        tok = row % history
        slot = tl.where(tok < 256, slot0, slot1)
        p_tok = tok % 256
        u_idx = tl.load(tile_to_u_sp + batch * 256 + p_tok)

        heads = hg * 8 + tl.arange(0, 8)[:, None]
        j = tl.arange(0, 32)[None, :]
        rot_mask = j < 24
        h32 = heads * 32 + j

        ku = tl.load(k_u_ring_u32 + (slot * u_sp + u_idx) * 1024 + h32)
        ke, ko = _unpack_u32_to_fp32_pair(ku)

        ce = tl.load(ce_hist + tok * 32 + j).to(tl.float32)
        co = tl.load(co_hist + tok * 32 + j).to(tl.float32)
        se = tl.load(se_hist + tok * 32 + j).to(tl.float32)
        so = tl.load(so_hist + tok * 32 + j).to(tl.float32)

        ke_rot = (
            (ke * ce).to(tl.bfloat16).to(tl.float32)
            + (-ko * se).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)
        ko_rot = (
            (ko * co).to(tl.bfloat16).to(tl.float32)
            + (ke * so).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)
        ke_fin = tl.where(rot_mask, ke_rot, ke.to(tl.bfloat16))
        ko_fin = tl.where(rot_mask, ko_rot, ko.to(tl.bfloat16))

        kv_total = history + 261
        out_off = (batch * kv_total + tok) * 1024 + h32
        _store_bf16_pair_to_fp8_u16(kr_out_u16, out_off, ke_fin, ko_fin, tl.float8e4nv)

    @triton.jit
    def _qkv_tile_rope_from_unique_u32_kernel(
        q_norm_u32,
        k_norm_u32,
        v_norm_u32,
        tile_to_u_full,
        ce_q,
        co_q,
        se_q,
        so_q,
        ce_k,
        co_k,
        se_k,
        so_k,
        oq_out_u16,
        kr_out_u16,
        v_ring_u16,
        history: tl.constexpr,
        q_out_tokens: tl.constexpr,
        v_ring_len: tl.constexpr,
        v_write_base: tl.constexpr,
        v_mirror_base: tl.constexpr,
    ):
        row = tl.program_id(0)
        hg = tl.program_id(1)
        batch = row // 261
        tok = row % 261
        u_idx = tl.load(tile_to_u_full + row)

        heads = hg * 8 + tl.arange(0, 8)[:, None]
        j = tl.arange(0, 32)[None, :]
        rot_mask = j < 24
        h32 = heads * 32 + j
        u_off = u_idx * 1024 + h32

        vu = tl.load(v_norm_u32 + u_off)
        ve, vo = _unpack_u32_to_fp32_pair(vu)
        ve_bf16 = ve.to(tl.bfloat16)
        vo_bf16 = vo.to(tl.bfloat16)
        v_dst = (batch * v_ring_len + v_write_base + tok) * 1024 + h32
        _store_bf16_pair_to_fp8_u16(v_ring_u16, v_dst, ve_bf16, vo_bf16, tl.float8e4nv)
        if v_mirror_base >= 0:
            if tok < 256:
                vm_dst = (batch * v_ring_len + v_mirror_base + tok) * 1024 + h32
                _store_bf16_pair_to_fp8_u16(
                    v_ring_u16, vm_dst, ve_bf16, vo_bf16, tl.float8e4nv
                )

        ku = tl.load(k_norm_u32 + u_off)
        ke, ko = _unpack_u32_to_fp32_pair(ku)
        cek = tl.load(ce_k + tok * 32 + j).to(tl.float32)
        cok = tl.load(co_k + tok * 32 + j).to(tl.float32)
        sek = tl.load(se_k + tok * 32 + j).to(tl.float32)
        sok = tl.load(so_k + tok * 32 + j).to(tl.float32)
        ke_rot = (
            (ke * cek).to(tl.bfloat16).to(tl.float32)
            + (-ko * sek).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)
        ko_rot = (
            (ko * cok).to(tl.bfloat16).to(tl.float32)
            + (ke * sok).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)
        ke_fin = tl.where(rot_mask, ke_rot, ke.to(tl.bfloat16))
        ko_fin = tl.where(rot_mask, ko_rot, ko.to(tl.bfloat16))

        kv_total = history + 261
        kr_dst = (batch * kv_total + history + tok) * 1024 + h32
        _store_bf16_pair_to_fp8_u16(kr_out_u16, kr_dst, ke_fin, ko_fin, tl.float8e4nv)

        if tok < q_out_tokens:
            qu = tl.load(q_norm_u32 + u_off)
            qe, qo = _unpack_u32_to_fp32_pair(qu)
            ceq = tl.load(ce_q + tok * 32 + j).to(tl.float32)
            coq = tl.load(co_q + tok * 32 + j).to(tl.float32)
            seq = tl.load(se_q + tok * 32 + j).to(tl.float32)
            soq = tl.load(so_q + tok * 32 + j).to(tl.float32)
            qe_rot = (
                (qe * ceq).to(tl.bfloat16).to(tl.float32)
                + (-qo * seq).to(tl.bfloat16).to(tl.float32)
            ).to(tl.bfloat16)
            qo_rot = (
                (qo * coq).to(tl.bfloat16).to(tl.float32)
                + (qe * soq).to(tl.bfloat16).to(tl.float32)
            ).to(tl.bfloat16)
            qe_fin = tl.where(rot_mask, qe_rot, qe.to(tl.bfloat16))
            qo_fin = tl.where(rot_mask, qo_rot, qo.to(tl.bfloat16))

            oq_dst = (batch * q_out_tokens + tok) * 1024 + h32
            _store_bf16_pair_to_fp8_u16(
                oq_out_u16, oq_dst, qe_fin, qo_fin, tl.float8e4nv
            )

    @triton.jit
    def _k_hist_tile_u32_kernel(
        k_t_ring_u32,
        ce_hist,
        co_hist,
        se_hist,
        so_hist,
        kr_out_u16,
        history: tl.constexpr,
        slot0_base: tl.constexpr,
        slot1_base: tl.constexpr,
    ):
        row = tl.program_id(0)
        hg = tl.program_id(1)
        batch = row // history
        tok = row % history
        src_tok = tl.where(tok < 256, slot0_base + tok, slot1_base + (tok - 256))

        heads = hg * 8 + tl.arange(0, 8)[:, None]
        j = tl.arange(0, 32)[None, :]
        rot_mask = j < 24
        h32 = heads * 32 + j

        ku = tl.load(k_t_ring_u32 + (batch * 1024 + src_tok) * 1024 + h32)
        ke, ko = _unpack_u32_to_fp32_pair(ku)

        ce = tl.load(ce_hist + tok * 32 + j).to(tl.float32)
        co = tl.load(co_hist + tok * 32 + j).to(tl.float32)
        se = tl.load(se_hist + tok * 32 + j).to(tl.float32)
        so = tl.load(so_hist + tok * 32 + j).to(tl.float32)

        ke_rot = (
            (ke * ce).to(tl.bfloat16).to(tl.float32)
            + (-ko * se).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)
        ko_rot = (
            (ko * co).to(tl.bfloat16).to(tl.float32)
            + (ke * so).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)
        ke_fin = tl.where(rot_mask, ke_rot, ke.to(tl.bfloat16))
        ko_fin = tl.where(rot_mask, ko_rot, ko.to(tl.bfloat16))

        kv_total = history + 261
        out_off = (batch * kv_total + tok) * 1024 + h32
        _store_bf16_pair_to_fp8_u16(kr_out_u16, out_off, ke_fin, ko_fin, tl.float8e4nv)

    @triton.jit
    def _qkv_tile_u32_kernel(
        raw_qkv_u32,
        xs_qkv,
        ws_qkv,
        b_qkv_u32,
        nq_we,
        nq_wo,
        nk_we,
        nk_wo,
        ce_q,
        co_q,
        se_q,
        so_q,
        ce_k,
        co_k,
        se_k,
        so_k,
        oq_out_u16,
        kr_out_u16,
        v_ring_u16,
        k_t_ring_u32,
        history: tl.constexpr,
        q_out_tokens: tl.constexpr,
        v_ring_len: tl.constexpr,
        v_write_base: tl.constexpr,
        v_mirror_base: tl.constexpr,
        k_slot_base: tl.constexpr,
        has_nq_w: tl.constexpr,
        has_nk_w: tl.constexpr,
        eps_q: tl.constexpr,
        eps_k: tl.constexpr,
    ):
        row = tl.program_id(0)
        hg = tl.program_id(1)
        batch = row // 261
        tok = row % 261
        xs = tl.load(xs_qkv + row)

        heads = hg * 8 + tl.arange(0, 8)[:, None]
        j = tl.arange(0, 32)[None, :]
        rot_mask = j < 24
        h32 = heads * 32 + j
        even_col = heads * 64 + 2 * j
        odd_col = even_col + 1
        row_u32 = raw_qkv_u32 + row * 3072

        vu = tl.load(row_u32 + 2048 + h32)
        ve_raw, vo_raw = _unpack_u32_to_fp32_pair(vu)
        vbu = tl.load(b_qkv_u32 + 2048 + h32)
        vbe, vbo = _unpack_u32_to_fp32_pair(vbu)
        wve = tl.load(ws_qkv + 4096 + even_col)
        wvo = tl.load(ws_qkv + 4096 + odd_col)
        ve_bf16 = (ve_raw * (xs * wve) + vbe).to(tl.bfloat16)
        vo_bf16 = (vo_raw * (xs * wvo) + vbo).to(tl.bfloat16)
        v_dst = (batch * v_ring_len + v_write_base + tok) * 1024 + h32
        _store_bf16_pair_to_fp8_u16(v_ring_u16, v_dst, ve_bf16, vo_bf16, tl.float8e4nv)
        if v_mirror_base >= 0:
            if tok < 256:
                vm_dst = (batch * v_ring_len + v_mirror_base + tok) * 1024 + h32
                _store_bf16_pair_to_fp8_u16(
                    v_ring_u16, vm_dst, ve_bf16, vo_bf16, tl.float8e4nv
                )

        ku = tl.load(row_u32 + 1024 + h32)
        ke_raw, ko_raw = _unpack_u32_to_fp32_pair(ku)
        kbu = tl.load(b_qkv_u32 + 1024 + h32)
        kbe, kbo = _unpack_u32_to_fp32_pair(kbu)
        wke = tl.load(ws_qkv + 2048 + even_col)
        wko = tl.load(ws_qkv + 2048 + odd_col)
        ke0 = (ke_raw * (xs * wke) + kbe).to(tl.bfloat16).to(tl.float32)
        ko0 = (ko_raw * (xs * wko) + kbo).to(tl.bfloat16).to(tl.float32)
        rstd_k = tl.rsqrt(
            tl.sum(ke0 * ke0 + ko0 * ko0, axis=1, keep_dims=True) * (1.0 / 64.0) + eps_k
        )
        if has_nk_w:
            ke_n = (ke0 * rstd_k * tl.load(nk_we + j)).to(tl.bfloat16)
            ko_n = (ko0 * rstd_k * tl.load(nk_wo + j)).to(tl.bfloat16)
        else:
            ke_n = (ke0 * rstd_k).to(tl.bfloat16)
            ko_n = (ko0 * rstd_k).to(tl.bfloat16)
        if tok < 256:
            tl.store(
                k_t_ring_u32 + (batch * 1024 + k_slot_base + tok) * 1024 + h32,
                _pack_bf16_pair_to_u32(ke_n, ko_n),
            )

        cek = tl.load(ce_k + tok * 32 + j).to(tl.float32)
        cok = tl.load(co_k + tok * 32 + j).to(tl.float32)
        sek = tl.load(se_k + tok * 32 + j).to(tl.float32)
        sok = tl.load(so_k + tok * 32 + j).to(tl.float32)
        kef = ke_n.to(tl.float32)
        kof = ko_n.to(tl.float32)
        ke_rot = (
            (kef * cek).to(tl.bfloat16).to(tl.float32)
            + (-kof * sek).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)
        ko_rot = (
            (kof * cok).to(tl.bfloat16).to(tl.float32)
            + (kef * sok).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)
        ke_fin = tl.where(rot_mask, ke_rot, ke_n)
        ko_fin = tl.where(rot_mask, ko_rot, ko_n)

        kv_total = history + 261
        kr_dst = (batch * kv_total + history + tok) * 1024 + h32
        _store_bf16_pair_to_fp8_u16(kr_out_u16, kr_dst, ke_fin, ko_fin, tl.float8e4nv)

        if tok < q_out_tokens:
            qu = tl.load(row_u32 + h32)
            qe_raw, qo_raw = _unpack_u32_to_fp32_pair(qu)
            qbu = tl.load(b_qkv_u32 + h32)
            qbe, qbo = _unpack_u32_to_fp32_pair(qbu)
            wqe = tl.load(ws_qkv + even_col)
            wqo = tl.load(ws_qkv + odd_col)
            qe0 = (qe_raw * (xs * wqe) + qbe).to(tl.bfloat16).to(tl.float32)
            qo0 = (qo_raw * (xs * wqo) + qbo).to(tl.bfloat16).to(tl.float32)
            rstd_q = tl.rsqrt(
                tl.sum(qe0 * qe0 + qo0 * qo0, axis=1, keep_dims=True) * (1.0 / 64.0)
                + eps_q
            )
            if has_nq_w:
                qe_n = (qe0 * rstd_q * tl.load(nq_we + j)).to(tl.bfloat16)
                qo_n = (qo0 * rstd_q * tl.load(nq_wo + j)).to(tl.bfloat16)
            else:
                qe_n = (qe0 * rstd_q).to(tl.bfloat16)
                qo_n = (qo0 * rstd_q).to(tl.bfloat16)
            qef = qe_n.to(tl.float32)
            qof = qo_n.to(tl.float32)

            ceq = tl.load(ce_q + tok * 32 + j).to(tl.float32)
            coq = tl.load(co_q + tok * 32 + j).to(tl.float32)
            seq = tl.load(se_q + tok * 32 + j).to(tl.float32)
            soq = tl.load(so_q + tok * 32 + j).to(tl.float32)
            qe_rot = (
                (qef * ceq).to(tl.bfloat16).to(tl.float32)
                + (-qof * seq).to(tl.bfloat16).to(tl.float32)
            ).to(tl.bfloat16)
            qo_rot = (
                (qof * coq).to(tl.bfloat16).to(tl.float32)
                + (qef * soq).to(tl.bfloat16).to(tl.float32)
            ).to(tl.bfloat16)
            qe_fin = tl.where(rot_mask, qe_rot, qe_n)
            qo_fin = tl.where(rot_mask, qo_rot, qo_n)

            oq_dst = (batch * q_out_tokens + tok) * 1024 + h32
            _store_bf16_pair_to_fp8_u16(
                oq_out_u16, oq_dst, qe_fin, qo_fin, tl.float8e4nv
            )

    @triton.jit
    def _update_tile_hidden_only_kernel(
        hidden_t,
        raw_out,
        xs_out,
        ws_out,
        b_out,
        tokens: tl.constexpr,
    ):
        row = tl.program_id(0)
        h_row = row if tokens == 261 else (row // tokens) * 261 + (row % tokens)
        cols = tl.arange(0, 2048)
        hp = tl.load(hidden_t + h_row * 2048 + cols).to(tl.float32)
        raw = tl.load(raw_out + row * 2048 + cols).to(tl.float32)
        xs = tl.load(xs_out + row)
        ws = tl.load(ws_out + cols)
        bs = tl.load(b_out + cols).to(tl.float32)
        tl.store(
            hidden_t + h_row * 2048 + cols,
            (hp + (raw * (xs * ws) + bs).to(tl.bfloat16).to(tl.float32)).to(
                tl.bfloat16
            ),
        )

    @triton.jit
    def _blend_hidden_rmsnorm_quant_kernel(
        hidden_t,
        blend_src,
        blend_w,
        norm_w,
        xq_u,
        xs_u,
        eps: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, 2048)
        s0 = tl.load(blend_src + row * 3 + 0)
        w0 = tl.load(blend_w + row * 3 + 0)
        w1 = tl.load(blend_w + row * 3 + 1)
        w2 = tl.load(blend_w + row * 3 + 2)

        acc = tl.load(hidden_t + s0 * 2048 + cols).to(tl.float32) * w0
        if w1 > 0.0:
            s1 = tl.load(blend_src + row * 3 + 1)
            acc = acc + tl.load(hidden_t + s1 * 2048 + cols).to(tl.float32) * w1
        if w2 > 0.0:
            s2 = tl.load(blend_src + row * 3 + 2)
            acc = acc + tl.load(hidden_t + s2 * 2048 + cols).to(tl.float32) * w2

        hf = acc.to(tl.bfloat16).to(tl.float32)
        nw = tl.load(norm_w + cols).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(hf * hf, axis=0) * (1.0 / 2048.0) + eps)
        normed = (hf * rstd * nw).to(tl.bfloat16).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(normed), axis=0), 1e-12)
        scale = amax * (1.0 / 448.0)
        inv_scale = 448.0 / amax
        tl.store(
            xq_u + row * 2048 + cols,
            tl.clamp(normed * inv_scale, -448.0, 448.0).to(xq_u.dtype.element_ty),
        )
        tl.store(xs_u + row, scale)

    @triton.jit
    def _blend_tiles_quant_fp8_kernel(
        attended,
        blend_src,
        blend_w,
        xq_out,
        xs_out,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, 2048)
        s0 = tl.load(blend_src + row * 3 + 0)
        w0 = tl.load(blend_w + row * 3 + 0)
        w1 = tl.load(blend_w + row * 3 + 1)
        w2 = tl.load(blend_w + row * 3 + 2)

        acc = tl.load(attended + s0 * 2048 + cols).to(tl.float32) * w0
        if w1 > 0.0:
            s1 = tl.load(blend_src + row * 3 + 1)
            acc = acc + tl.load(attended + s1 * 2048 + cols).to(tl.float32) * w1
        if w2 > 0.0:
            s2 = tl.load(blend_src + row * 3 + 2)
            acc = acc + tl.load(attended + s2 * 2048 + cols).to(tl.float32) * w2

        val = acc.to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(val), axis=0), 1e-12)
        scale = amax * (1.0 / 448.0)
        inv_scale = 448.0 / amax
        tl.store(
            xq_out + row * 2048 + cols,
            tl.clamp(val * inv_scale, -448.0, 448.0).to(xq_out.dtype.element_ty),
        )
        tl.store(xs_out + row, scale)

    @triton.jit
    def _swiglu_zero_spill_w16_kernel(
        raw_w1,
        xs_w1,
        ws_w1,
        b_w1,
        xq_out,
        xs_out,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, 8192)
        xs = tl.load(xs_w1 + row)
        base = raw_w1 + row * 16384

        rx = tl.load(base + cols).to(tl.float32)
        wx = tl.load(ws_w1 + cols)
        bx = tl.load(b_w1 + cols).to(tl.float32)
        xv_bf16 = (rx * (xs * wx) + bx).to(tl.bfloat16)

        rg = tl.load(base + 8192 + cols).to(tl.float32)
        wg = tl.load(ws_w1 + 8192 + cols)
        bg = tl.load(b_w1 + 8192 + cols).to(tl.float32)
        gv = (rg * (xs * wg) + bg).to(tl.bfloat16).to(tl.float32)
        silu_g = (gv * tl.sigmoid(gv)).to(tl.bfloat16).to(tl.float32)
        act = (xv_bf16.to(tl.float32) * silu_g).to(tl.bfloat16).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(act), axis=0), 1e-12)
        scale = amax * (1.0 / 448.0)
        inv_scale = 448.0 / amax
        tl.store(
            xq_out + row * 8192 + cols,
            tl.clamp(act * inv_scale, -448.0, 448.0).to(xq_out.dtype.element_ty),
        )
        tl.store(xs_out + row, scale)

    @triton.jit
    def _gather_w2_res_rmsnorm_quant_tile_kernel(
        hidden_t,
        raw_w2_u,
        xs_w2_u,
        tile_to_u_full,
        ws_w2,
        b_w2,
        norm_w,
        xq_t,
        xs_t,
        eps: tl.constexpr,
    ):
        row = tl.program_id(0)
        u = tl.load(tile_to_u_full + row)
        cols = tl.arange(0, 2048)

        hp = tl.load(hidden_t + row * 2048 + cols).to(tl.float32)
        raw = tl.load(raw_w2_u + u * 2048 + cols).to(tl.float32)
        xs = tl.load(xs_w2_u + u)
        ws = tl.load(ws_w2 + cols)
        bs = tl.load(b_w2 + cols).to(tl.float32)

        h_bf16 = (hp + (raw * (xs * ws) + bs).to(tl.bfloat16).to(tl.float32)).to(
            tl.bfloat16
        )
        tl.store(hidden_t + row * 2048 + cols, h_bf16)
        hf = h_bf16.to(tl.float32)

        nw = tl.load(norm_w + cols).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(hf * hf, axis=0) * (1.0 / 2048.0) + eps)
        normed = (hf * rstd * nw).to(tl.bfloat16).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(normed), axis=0), 1e-12)
        scale = amax * (1.0 / 448.0)
        inv_scale = 448.0 / amax
        tl.store(
            xq_t + row * 2048 + cols,
            tl.clamp(normed * inv_scale, -448.0, 448.0).to(xq_t.dtype.element_ty),
        )
        tl.store(xs_t + row, scale)

    @triton.jit
    def _blend_backbone_w2_init_hu_and_norm1_kernel(
        hidden_t,
        blend_src_full,
        blend_w,
        raw_w2_u,
        xs_w2_u,
        ws_w2,
        b_w2,
        norm1_w,
        h_u_out,
        xq_u_out,
        xs_u_out,
        eps: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, 2048)
        s0 = tl.load(blend_src_full + row * 3 + 0)
        w0 = tl.load(blend_w + row * 3 + 0)
        w1 = tl.load(blend_w + row * 3 + 1)
        w2 = tl.load(blend_w + row * 3 + 2)

        acc = tl.load(hidden_t + s0 * 2048 + cols).to(tl.float32) * w0
        if w1 > 0.0:
            s1 = tl.load(blend_src_full + row * 3 + 1)
            acc = acc + tl.load(hidden_t + s1 * 2048 + cols).to(tl.float32) * w1
        if w2 > 0.0:
            s2 = tl.load(blend_src_full + row * 3 + 2)
            acc = acc + tl.load(hidden_t + s2 * 2048 + cols).to(tl.float32) * w2

        h_blend = acc.to(tl.bfloat16).to(tl.float32)
        raw = tl.load(raw_w2_u + row * 2048 + cols).to(tl.float32)
        xs = tl.load(xs_w2_u + row)
        ws = tl.load(ws_w2 + cols)
        bs = tl.load(b_w2 + cols).to(tl.float32)

        h_bf16 = (h_blend + (raw * (xs * ws) + bs).to(tl.bfloat16).to(tl.float32)).to(
            tl.bfloat16
        )
        tl.store(h_u_out + row * 2048 + cols, h_bf16)
        hf = h_bf16.to(tl.float32)

        nw = tl.load(norm1_w + cols).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(hf * hf, axis=0) * (1.0 / 2048.0) + eps)
        normed = (hf * rstd * nw).to(tl.bfloat16).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(normed), axis=0), 1e-12)
        scale = amax * (1.0 / 448.0)
        inv_scale = 448.0 / amax
        tl.store(
            xq_u_out + row * 2048 + cols,
            tl.clamp(normed * inv_scale, -448.0, 448.0).to(xq_u_out.dtype.element_ty),
        )
        tl.store(xs_u_out + row, scale)

    @triton.jit
    def _res_rmsnorm_quant_unique_kernel(
        h_u,
        raw_proj,
        xs_prev,
        ws_scaled,
        b_scaled,
        norm_w,
        xq_out,
        xs_out,
        eps: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, 2048)

        h_prev = tl.load(h_u + row * 2048 + cols).to(tl.float32)
        raw = tl.load(raw_proj + row * 2048 + cols).to(tl.float32)
        xs = tl.load(xs_prev + row)
        ws = tl.load(ws_scaled + cols)
        bs = tl.load(b_scaled + cols).to(tl.float32)

        h_bf16 = (h_prev + (raw * (xs * ws) + bs).to(tl.bfloat16).to(tl.float32)).to(
            tl.bfloat16
        )
        tl.store(h_u + row * 2048 + cols, h_bf16)
        hf = h_bf16.to(tl.float32)

        nw = tl.load(norm_w + cols).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(hf * hf, axis=0) * (1.0 / 2048.0) + eps)
        normed = (hf * rstd * nw).to(tl.bfloat16).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(normed), axis=0), 1e-12)
        scale = amax * (1.0 / 448.0)
        inv_scale = 448.0 / amax
        tl.store(
            xq_out + row * 2048 + cols,
            tl.clamp(normed * inv_scale, -448.0, 448.0).to(xq_out.dtype.element_ty),
        )
        tl.store(xs_out + row, scale)

    @triton.jit
    def _res_layernorm_bf16_unique_kernel(
        h_u,
        raw_w2_u,
        xs_w2_u,
        ws_w2,
        b_w2,
        ln_w,
        ln_b,
        y_out,
        eps: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, 2048)

        h_prev = tl.load(h_u + row * 2048 + cols).to(tl.float32)
        raw = tl.load(raw_w2_u + row * 2048 + cols).to(tl.float32)
        xs = tl.load(xs_w2_u + row)
        ws = tl.load(ws_w2 + cols)
        bs = tl.load(b_w2 + cols).to(tl.float32)

        h = h_prev + (raw * (xs * ws) + bs).to(tl.bfloat16).to(tl.float32)
        mean = tl.sum(h, axis=0) * (1.0 / 2048.0)
        centered = h - mean
        rstd = tl.rsqrt(tl.sum(centered * centered, axis=0) * (1.0 / 2048.0) + eps)
        lw = tl.load(ln_w + cols)
        lb = tl.load(ln_b + cols)
        tl.store(y_out + row * 2048 + cols, (centered * rstd * lw + lb).to(tl.bfloat16))

    @triton.jit
    def _blend_backbone_w2_layernorm_unique_kernel(
        hidden_t,
        blend_src_full,
        blend_w,
        raw_w2_u,
        xs_w2_u,
        ws_w2,
        b_w2,
        ln_w,
        ln_b,
        y_out_u,
        eps: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, 2048)
        s0 = tl.load(blend_src_full + row * 3 + 0)
        w0 = tl.load(blend_w + row * 3 + 0)
        w1 = tl.load(blend_w + row * 3 + 1)
        w2 = tl.load(blend_w + row * 3 + 2)

        acc = tl.load(hidden_t + s0 * 2048 + cols).to(tl.float32) * w0
        if w1 > 0.0:
            s1 = tl.load(blend_src_full + row * 3 + 1)
            acc = acc + tl.load(hidden_t + s1 * 2048 + cols).to(tl.float32) * w1
        if w2 > 0.0:
            s2 = tl.load(blend_src_full + row * 3 + 2)
            acc = acc + tl.load(hidden_t + s2 * 2048 + cols).to(tl.float32) * w2

        h = acc.to(tl.bfloat16).to(tl.float32)
        raw = tl.load(raw_w2_u + row * 2048 + cols).to(tl.float32)
        xs = tl.load(xs_w2_u + row)
        ws = tl.load(ws_w2 + cols)
        bs = tl.load(b_w2 + cols).to(tl.float32)
        h = h + (raw * (xs * ws) + bs).to(tl.bfloat16).to(tl.float32)

        mean = tl.sum(h, axis=0) * (1.0 / 2048.0)
        centered = h - mean
        rstd = tl.rsqrt(tl.sum(centered * centered, axis=0) * (1.0 / 2048.0) + eps)
        lw = tl.load(ln_w + cols)
        lb = tl.load(ln_b + cols)
        tl.store(
            y_out_u + row * 2048 + cols, (centered * rstd * lw + lb).to(tl.bfloat16)
        )

    @triton.jit
    def _unpatchify_unique_rgb4_kernel(
        pix_u,
        pix_mean,
        pix_std,
        rgb_out,
        ratio: tl.constexpr,
        h_lat: tl.constexpr,
        w_lat: tl.constexpr,
        h_out: tl.constexpr,
        w_out: tl.constexpr,
        block_x: tl.constexpr,
    ):
        pid_nct = tl.program_id(0)
        y = tl.program_id(1)
        xs = tl.program_id(2) * block_x + tl.arange(0, block_x)
        mask = xs < w_out

        n_idx = pid_nct // 12
        rem = pid_nct % 12
        c_idx = rem // 4
        t_idx = rem % 4

        y_l = y // ratio
        ry = y % ratio
        x_l = xs // ratio
        rx = xs % ratio

        patch_ch = 12 * ratio * ratio
        u_row = (n_idx * h_lat + y_l) * w_lat + x_l
        ch = ((c_idx * 4 + t_idx) * ratio + ry) * ratio + rx
        val = tl.load(pix_u + u_row * patch_ch + ch, mask=mask, other=0.0).to(
            tl.float32
        )

        std = tl.load(pix_std + c_idx)
        mean = tl.load(pix_mean + c_idx)
        out_off = (((n_idx * 3 + c_idx) * 4 + t_idx) * h_out + y) * w_out + xs
        tl.store(rgb_out + out_off, val * std + mean, mask=mask)


class _SlotLease:
    __slots__ = ("engine", "slot_key", "slot", "active")

    def __init__(self, engine, slot_key: tuple[int, int, int], slot: dict):
        self.engine = engine
        self.slot_key = slot_key
        self.slot = slot
        self.active = True

    def release(self):
        if self.active:
            self.active = False
            self.slot["in_use"] = False
            self.slot = None


class LynnSm90FastEngine:
    """Single-GPU SM90 CUDA-Graph fast execution engine for `StreamingDecoder`."""

    def __init__(self, model, signature: tuple):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        self.signature = signature
        self.fa3_kv = flash_attn_with_kvcache
        device = model.latent_mean.device
        self.device = device
        self.spatial_ratio = model.spatial_ratio
        self.tile_size = model.config.tile_size
        self.tile_overlap = model.config.tile_overlap
        self.slots: dict[tuple[int, int, int], dict] = {}

        with (
            torch.inference_mode(False),
            torch.no_grad(),
            torch.amp.autocast("cuda", enabled=False),
        ):
            self.one = torch.ones((), device=device, dtype=torch.float32)
            self.latent_mean_1d = (
                model.latent_mean.detach().reshape(-1).float().contiguous()
            )
            self.latent_std_1d = (
                model.latent_std.detach().reshape(-1).float().contiguous()
            )
            self.pixel_mean_1d = (
                model.pixel_mean.detach().reshape(-1).float().contiguous()
            )
            self.pixel_std_1d = (
                model.pixel_std.detach().reshape(-1).float().contiguous()
            )
            self.phase_emb_fp32 = model.phase_embedding.detach().float().contiguous()
            self.reg_tokens_fp32 = (
                model.decoder.register_tokens.detach()
                .reshape(4, 2048)
                .float()
                .contiguous()
            )

            conv = model.post_quant_conv
            self.post_quant_w = conv.weight.detach().to(torch.bfloat16).contiguous()
            self.post_quant_b = (
                conv.bias.detach().to(torch.bfloat16).contiguous()
                if conv.bias is not None
                else None
            )

            dec = model.decoder
            self.proj_in_w = dec.proj_in.weight.detach().float().contiguous()
            self.proj_in_b = (
                dec.proj_in.bias.detach().float().contiguous()
                if dec.proj_in.bias is not None
                else None
            )
            self.proj_out_w = (
                dec.proj_out.weight.detach().to(torch.bfloat16).contiguous()
            )
            self.proj_out_b = (
                dec.proj_out.bias.detach().to(torch.bfloat16).contiguous()
                if dec.proj_out.bias is not None
                else None
            )

            gy, gx = torch.meshgrid(
                2 * (torch.arange(16, device=device).float() + 0.5) / 16 - 1,
                2 * (torch.arange(16, device=device).float() + 0.5) / 16 - 1,
                indexing="ij",
            )
            pos = torch.stack([torch.zeros_like(gy), gy, gx], dim=-1).flatten(0, 1)[
                None
            ]
            suf = pos.new_zeros((1, 5, 3))
            cq, sq = dec.rope(torch.cat([pos, suf], dim=1))
            self.rope_q = _pack_rope_tables_interleaved(
                cq.detach().reshape(261, 48), sq.detach().reshape(261, 48)
            )
            self.rope_k_cur = {0: self.rope_q}
            self.rope_k_hist = {}
            window = model.protocol.teacher_window
            for hist_frames in (1, 2):
                past_list = []
                for dist in range(hist_frames, 0, -1):
                    p = pos.clone()
                    p[..., 0] = 2 * -dist / window
                    past_list.append(p)
                ck, sk = dec.rope(torch.cat([*past_list, pos, suf], dim=1))
                h_tok = hist_frames * 256
                ck2 = ck.detach().reshape(h_tok + 261, 48)
                sk2 = sk.detach().reshape(h_tok + 261, 48)
                self.rope_k_hist[h_tok] = _pack_rope_tables_interleaved(
                    ck2[:h_tok], sk2[:h_tok]
                )
                self.rope_k_cur[h_tok] = _pack_rope_tables_interleaved(
                    ck2[h_tok:], sk2[h_tok:]
                )

            perm64 = _make_qk_perm64(device)
            watched = [
                model.post_quant_conv,
                dec.proj_in,
                dec.proj_out,
                dec.norm_out,
            ]
            self.layers = []
            for blk in dec.transformer_blocks:
                attn = blk.attn
                watched.extend(_ffn_modules(blk))
                watched.extend(
                    (
                        blk.norm1,
                        blk.norm2,
                        attn.norm_q,
                        attn.norm_k,
                        attn.to_q,
                        attn.to_k,
                        attn.to_v,
                        attn.to_out[0],
                        blk.ff.net[0].proj,
                        blk.ff.net[2],
                    )
                )
                wq_perm = (
                    attn.to_q.weight.detach()
                    .float()
                    .view(32, 64, 2048)[:, perm64, :]
                    .reshape(2048, 2048)
                )
                wk_perm = (
                    attn.to_k.weight.detach()
                    .float()
                    .view(32, 64, 2048)[:, perm64, :]
                    .reshape(2048, 2048)
                )
                wv_orig = attn.to_v.weight.detach().float()
                wf_qkv = torch.cat([wq_perm, wk_perm, wv_orig], dim=0).contiguous()
                ws_qkv = wf_qkv.abs().amax(dim=1).clamp_min_(1e-12) / 448.0
                wq_qkv = (
                    (wf_qkv / ws_qkv[:, None])
                    .clamp_(-448.0, 448.0)
                    .to(torch.float8_e4m3fn)
                    .contiguous()
                    .t()
                )

                bq_raw = (
                    attn.to_q.bias.detach().float()
                    if attn.to_q.bias is not None
                    else torch.zeros(2048, device=device, dtype=torch.float32)
                )
                bk_raw = (
                    attn.to_k.bias.detach().float()
                    if attn.to_k.bias is not None
                    else torch.zeros(2048, device=device, dtype=torch.float32)
                )
                bv_raw = (
                    attn.to_v.bias.detach().float()
                    if attn.to_v.bias is not None
                    else torch.zeros(2048, device=device, dtype=torch.float32)
                )
                bq_perm = bq_raw.view(32, 64)[:, perm64].reshape(2048)
                bk_perm = bk_raw.view(32, 64)[:, perm64].reshape(2048)
                b_qkv_bf16 = (
                    torch.cat([bq_perm, bk_perm, bv_raw], dim=0)
                    .to(torch.bfloat16)
                    .contiguous()
                )

                has_nq_w = getattr(attn.norm_q, "weight", None) is not None
                if has_nq_w:
                    nq_p = attn.norm_q.weight.detach().float()[perm64]
                    nq_we, nq_wo = nq_p[0::2].contiguous(), nq_p[1::2].contiguous()
                else:
                    nq_we = nq_wo = None
                has_nk_w = getattr(attn.norm_k, "weight", None) is not None
                if has_nk_w:
                    nk_p = attn.norm_k.weight.detach().float()[perm64]
                    nk_we, nk_wo = nk_p[0::2].contiguous(), nk_p[1::2].contiguous()
                else:
                    nk_we = nk_wo = None

                wq_out, ws_out, b_out = _quantize_linear_colmajor(
                    attn.to_out[0], blk.scale1
                )
                wq_w1, ws_w1, b_w1 = _quantize_linear_colmajor(blk.ff.net[0].proj, None)
                wq_w2, ws_w2, b_w2 = _quantize_linear_colmajor(
                    blk.ff.net[2], blk.scale2
                )

                n1w = (
                    blk.norm1.weight.detach().to(torch.bfloat16).contiguous()
                    if blk.norm1.weight is not None
                    else torch.ones(2048, device=device, dtype=torch.bfloat16)
                )
                n2w = (
                    blk.norm2.weight.detach().to(torch.bfloat16).contiguous()
                    if blk.norm2.weight is not None
                    else torch.ones(2048, device=device, dtype=torch.bfloat16)
                )

                self.layers.append(
                    {
                        "norm1_w": n1w,
                        "norm1_eps": blk.norm1.eps,
                        "wq_qkv": wq_qkv,
                        "ws_qkv": ws_qkv.contiguous(),
                        "b_qkv_u32": b_qkv_bf16.view(torch.uint32),
                        "has_nq_w": has_nq_w,
                        "has_nk_w": has_nk_w,
                        "nq_we": nq_we,
                        "nq_wo": nq_wo,
                        "nk_we": nk_we,
                        "nk_wo": nk_wo,
                        "eps_q": attn.norm_q.eps,
                        "eps_k": attn.norm_k.eps,
                        "wq_out": wq_out,
                        "ws_out": ws_out,
                        "b_out": b_out,
                        "norm2_w": n2w,
                        "norm2_eps": blk.norm2.eps,
                        "wq_w1": wq_w1,
                        "ws_w1": ws_w1,
                        "b_w1": b_w1,
                        "wq_w2": wq_w2,
                        "ws_w2": ws_w2,
                        "b_w2": b_w2,
                    }
                )
            self._watched_modules = tuple(watched)
            self._expected_config = model.config

            self.norm_out_w = (
                dec.norm_out.weight.detach().float().contiguous()
                if dec.norm_out.weight is not None
                else torch.ones(2048, device=device, dtype=torch.float32)
            )
            self.norm_out_b = (
                dec.norm_out.bias.detach().float().contiguous()
                if dec.norm_out.bias is not None
                else torch.zeros(2048, device=device, dtype=torch.float32)
            )
            self.norm_out_eps = dec.norm_out.eps

            num_layers = len(self.layers)
            if num_layers <= 4:
                self.fast_start = num_layers
                self.w2_fast_accum_layers = 0
            else:
                # Protect shallow Full-Tile layers (0..3 on 26L) so per-tile 2D anchors
                # form cleanly; execute deep semantic layers (4..25 on 26L) on unique grid.
                self.fast_start = max(1, num_layers - (num_layers * 22) // 26)
                self.w2_fast_accum_layers = (num_layers * 22) // 26

            self.qkv_u_slot_map: dict[int, int] = {}
            self.qkv_t_slot_map: dict[int, int] = {}
            for idx in range(num_layers):
                if idx == 0 or idx >= self.fast_start:
                    self.qkv_u_slot_map[idx] = len(self.qkv_u_slot_map)
                else:
                    self.qkv_t_slot_map[idx] = len(self.qkv_t_slot_map)

    def clear(self):
        for slot in self.slots.values():
            slot["graphs"].clear()
            slot["in_use"] = False
        self.slots.clear()

    def _run_attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        res = self.fa3_kv(q, k, v, causal=False)
        return res[0] if isinstance(res, tuple) else res

    def _forward_impl(self, slot: dict, mode_id: int):
        ratio = self.spatial_ratio
        batch_n = slot["N"]
        total_b = slot["B"]
        h_lat, w_lat = slot["H_lat"], slot["W_lat"]
        u_sp, u_total = slot["U_sp"], slot["U_total"]
        m_full, m_sp = total_b * 261, total_b * 256

        if mode_id == -2:
            history_tokens = 0
            v_write_base, v_mirror_base = 0, -1
            v_read_start, v_read_end = 0, 261
            k_slot_idx = 0
            s0_idx, s1_idx = 0, 0
        elif mode_id == -1:
            history_tokens = 256
            v_write_base, v_mirror_base = 256, -1
            v_read_start, v_read_end = 0, 517
            k_slot_idx = 1
            s0_idx, s1_idx = 0, 0
        else:
            s = mode_id
            history_tokens = 512
            v_write_base = (s + 2) * 256
            v_mirror_base = (s - 2) * 256 if s >= 2 else -1
            v_read_start, v_read_end = s * 256, s * 256 + 773
            s0_idx = s
            s1_idx = (s + 1) & 3
            k_slot_idx = (s + 2) & 3

        k_t_slot_base = k_slot_idx * 256
        k_t_s0 = s0_idx * 256
        k_t_s1 = s1_idx * 256

        oq_full_fp8 = torch.empty(
            (total_b, 261, 32, 64), device=self.device, dtype=torch.float8_e4m3fn
        )
        oq_sp_fp8 = torch.empty(
            (total_b, 256, 32, 64), device=self.device, dtype=torch.float8_e4m3fn
        )
        kr_buf_fp8 = torch.empty(
            (total_b, history_tokens + 261, 32, 64),
            device=self.device,
            dtype=torch.float8_e4m3fn,
        )

        ceq, coq, seq, soq = self.rope_q
        cek, cok, sek, sok = self.rope_k_cur[history_tokens]
        hist_rope = self.rope_k_hist.get(history_tokens)
        v_rings = slot["v_rings"]
        num_layers = len(self.layers)
        fast_start = self.fast_start

        hidden_t = torch.empty((m_full, 2048), device=self.device, dtype=torch.bfloat16)
        h_u = torch.empty((u_total, 2048), device=self.device, dtype=torch.bfloat16)

        xq_t_full = torch.empty(
            (m_full, 2048), device=self.device, dtype=torch.float8_e4m3fn
        )
        xs_t_full = torch.empty((m_full,), device=self.device, dtype=torch.float32)
        xq_t_sp = torch.empty(
            (m_sp, 2048), device=self.device, dtype=torch.float8_e4m3fn
        )
        xs_t_sp = torch.empty((m_sp,), device=self.device, dtype=torch.float32)

        xq_u_full = torch.empty(
            (u_total, 2048), device=self.device, dtype=torch.float8_e4m3fn
        )
        xs_u_full = torch.empty((u_total,), device=self.device, dtype=torch.float32)
        xq_u_sp = torch.empty(
            (u_sp, 2048), device=self.device, dtype=torch.float8_e4m3fn
        )
        xs_u_sp = torch.empty((u_sp,), device=self.device, dtype=torch.float32)

        xq_ff_u_full = torch.empty(
            (u_total, 8192), device=self.device, dtype=torch.float8_e4m3fn
        )
        xq_ff_u_sp = torch.empty(
            (u_sp, 8192), device=self.device, dtype=torch.float8_e4m3fn
        )

        q_norm_u32 = torch.empty(
            (u_total, 32, 32), device=self.device, dtype=torch.uint32
        )
        k_norm_u32 = torch.empty(
            (u_total, 32, 32), device=self.device, dtype=torch.uint32
        )
        v_norm_u32 = torch.empty(
            (u_total, 32, 32), device=self.device, dtype=torch.uint32
        )

        hw = h_lat * w_lat
        z_denorm = torch.empty(
            (batch_n, 24, 1, h_lat, w_lat), device=self.device, dtype=torch.bfloat16
        )
        _denorm_latent_bf16_kernel[(batch_n * 24, triton.cdiv(hw, 256))](
            slot["z_in"],
            self.latent_mean_1d,
            self.latent_std_1d,
            z_denorm,
            hw=hw,
            block=256,
            num_warps=4,
        )
        z_conv_u = F.conv3d(z_denorm, self.post_quant_w, self.post_quant_b)
        tokens_u = z_conv_u[:, :, 0].permute(0, 2, 3, 1).reshape(u_sp, 24).float()
        spatial_h_u = F.linear(tokens_u, self.proj_in_w, self.proj_in_b)

        l0 = self.layers[0]
        _init_tile_backbone_kernel[(m_full,)](
            spatial_h_u,
            slot["tile_to_u_full"],
            self.phase_emb_fp32,
            slot["phase_idx"],
            self.reg_tokens_fp32,
            hidden_t,
            num_warps=4,
        )
        _init_l0_unique_norm1_quant_kernel[(u_total,)](
            spatial_h_u,
            self.phase_emb_fp32,
            slot["phase_idx"],
            self.reg_tokens_fp32,
            l0["norm1_w"],
            xq_u_full,
            xs_u_full,
            u_sp=u_sp,
            eps=l0["norm1_eps"],
            num_warps=4,
        )

        for idx, lw in enumerate(self.layers):
            is_last = idx == num_layers - 1
            is_fast = idx >= fast_start
            qkv_is_u = (idx == 0) or is_fast
            v_ring_cur = v_rings[idx]
            v_ring_v = v_ring_cur.view(torch.uint16)
            oq_target = oq_sp_fp8 if is_last else oq_full_fp8
            oq_target_v = oq_target.view(torch.uint16)
            kr_buf_v = kr_buf_fp8.view(torch.uint16)
            q_out_tok = 256 if is_last else 261

            if qkv_is_u:
                ku_idx = self.qkv_u_slot_map[idx]
                raw_qkv_u = torch._scaled_mm(
                    xq_u_full,
                    lw["wq_qkv"],
                    scale_a=self.one,
                    scale_b=self.one,
                    out_dtype=torch.bfloat16,
                    use_fast_accum=True,
                )
                _qkv_norm_unique_u32_kernel[(u_total, 4)](
                    raw_qkv_u.view(torch.uint32),
                    xs_u_full,
                    lw["ws_qkv"],
                    lw["b_qkv_u32"],
                    lw["nq_we"],
                    lw["nq_wo"],
                    lw["nk_we"],
                    lw["nk_wo"],
                    q_norm_u32,
                    k_norm_u32,
                    v_norm_u32,
                    slot["k_u_ring_u32"][ku_idx],
                    u_sp=u_sp,
                    k_slot=k_slot_idx,
                    skip_suf_q=is_last,
                    has_nq_w=lw["has_nq_w"],
                    has_nk_w=lw["has_nk_w"],
                    eps_q=lw["eps_q"],
                    eps_k=lw["eps_k"],
                    num_warps=8,
                    enable_fp_fusion=False,
                )
                if history_tokens > 0:
                    ceh, coh, seh, soh = hist_rope
                    _k_hist_from_unique_u32_kernel[(total_b * history_tokens, 4)](
                        slot["k_u_ring_u32"][ku_idx],
                        slot["tile_to_u_sp"],
                        ceh,
                        coh,
                        seh,
                        soh,
                        kr_buf_v,
                        u_sp=u_sp,
                        history=history_tokens,
                        slot0=s0_idx,
                        slot1=s1_idx,
                        num_warps=8,
                        enable_fp_fusion=False,
                    )
                _qkv_tile_rope_from_unique_u32_kernel[(m_full, 4)](
                    q_norm_u32,
                    k_norm_u32,
                    v_norm_u32,
                    slot["tile_to_u_full"],
                    ceq,
                    coq,
                    seq,
                    soq,
                    cek,
                    cok,
                    sek,
                    sok,
                    oq_target_v,
                    kr_buf_v,
                    v_ring_v,
                    history=history_tokens,
                    q_out_tokens=q_out_tok,
                    v_ring_len=1541,
                    v_write_base=v_write_base,
                    v_mirror_base=v_mirror_base,
                    num_warps=8,
                    enable_fp_fusion=False,
                )
            else:
                kt_idx = self.qkv_t_slot_map[idx]
                raw_qkv_t = torch._scaled_mm(
                    xq_t_full,
                    lw["wq_qkv"],
                    scale_a=self.one,
                    scale_b=self.one,
                    out_dtype=torch.bfloat16,
                    use_fast_accum=True,
                )
                if history_tokens > 0:
                    ceh, coh, seh, soh = hist_rope
                    _k_hist_tile_u32_kernel[(total_b * history_tokens, 4)](
                        slot["k_t_ring_u32"][kt_idx],
                        ceh,
                        coh,
                        seh,
                        soh,
                        kr_buf_v,
                        history=history_tokens,
                        slot0_base=k_t_s0,
                        slot1_base=k_t_s1,
                        num_warps=8,
                        enable_fp_fusion=False,
                    )
                _qkv_tile_u32_kernel[(m_full, 4)](
                    raw_qkv_t.view(torch.uint32),
                    xs_t_full,
                    lw["ws_qkv"],
                    lw["b_qkv_u32"],
                    lw["nq_we"],
                    lw["nq_wo"],
                    lw["nk_we"],
                    lw["nk_wo"],
                    ceq,
                    coq,
                    seq,
                    soq,
                    cek,
                    cok,
                    sek,
                    sok,
                    oq_target_v,
                    kr_buf_v,
                    v_ring_v,
                    slot["k_t_ring_u32"][kt_idx],
                    history=history_tokens,
                    q_out_tokens=q_out_tok,
                    v_ring_len=1541,
                    v_write_base=v_write_base,
                    v_mirror_base=v_mirror_base,
                    k_slot_base=k_t_slot_base,
                    has_nq_w=lw["has_nq_w"],
                    has_nk_w=lw["has_nk_w"],
                    eps_q=lw["eps_q"],
                    eps_k=lw["eps_k"],
                    num_warps=8,
                    enable_fp_fusion=False,
                )

            attended = self._run_attn(
                oq_target, kr_buf_fp8, v_ring_cur[:, v_read_start:v_read_end]
            )

            m_cur = m_sp if is_last else m_full
            u_cur = u_sp if is_last else u_total
            xq_u_cur = xq_u_sp if is_last else xq_u_full
            xs_u_cur = xs_u_sp if is_last else xs_u_full
            xq_ff_u_cur = xq_ff_u_sp if is_last else xq_ff_u_full
            tok_cur = 256 if is_last else 261

            if is_fast:
                blend_src = slot["blend_src_sp"] if is_last else slot["blend_src_full"]
                _blend_tiles_quant_fp8_kernel[(u_cur,)](
                    attended.reshape(m_cur, 2048),
                    blend_src,
                    slot["blend_w"],
                    xq_u_cur,
                    xs_u_cur,
                    num_warps=4,
                )
                raw_out_u = torch._scaled_mm(
                    xq_u_cur,
                    lw["wq_out"],
                    scale_a=self.one,
                    scale_b=self.one,
                    out_dtype=torch.bfloat16,
                    use_fast_accum=True,
                )
                _res_rmsnorm_quant_unique_kernel[(u_cur,)](
                    h_u,
                    raw_out_u,
                    xs_u_cur,
                    lw["ws_out"],
                    lw["b_out"],
                    lw["norm2_w"],
                    xq_u_cur,
                    xs_u_cur,
                    eps=lw["norm2_eps"],
                    num_warps=4,
                )
            else:
                xq_t_cur = xq_t_sp if is_last else xq_t_full
                xs_t_cur = xs_t_sp if is_last else xs_t_full
                _quant_bf16_rows_kernel[(m_cur,)](
                    attended.reshape(m_cur, 2048), xq_t_cur, xs_t_cur, num_warps=4
                )
                raw_out_t = torch._scaled_mm(
                    xq_t_cur,
                    lw["wq_out"],
                    scale_a=self.one,
                    scale_b=self.one,
                    out_dtype=torch.bfloat16,
                    use_fast_accum=True,
                )
                _update_tile_hidden_only_kernel[(m_cur,)](
                    hidden_t,
                    raw_out_t,
                    xs_t_cur,
                    lw["ws_out"],
                    lw["b_out"],
                    tokens=tok_cur,
                    num_warps=4,
                )
                _blend_hidden_rmsnorm_quant_kernel[(u_cur,)](
                    hidden_t,
                    slot["blend_src_full"],
                    slot["blend_w"],
                    lw["norm2_w"],
                    xq_u_cur,
                    xs_u_cur,
                    eps=lw["norm2_eps"],
                    num_warps=4,
                )

            raw_w1 = torch._scaled_mm(
                xq_u_cur,
                lw["wq_w1"],
                scale_a=self.one,
                scale_b=self.one,
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )
            _swiglu_zero_spill_w16_kernel[(u_cur,)](
                raw_w1,
                xs_u_cur,
                lw["ws_w1"],
                lw["b_w1"],
                xq_ff_u_cur,
                xs_u_cur,
                num_warps=16,
            )
            w2_fast = idx < self.w2_fast_accum_layers
            raw_w2 = torch._scaled_mm(
                xq_ff_u_cur,
                lw["wq_w2"],
                scale_a=self.one,
                scale_b=self.one,
                out_dtype=torch.bfloat16,
                use_fast_accum=w2_fast,
            )

            if not is_last:
                next_lw = self.layers[idx + 1]
                if is_fast:
                    _res_rmsnorm_quant_unique_kernel[(u_total,)](
                        h_u,
                        raw_w2,
                        xs_u_cur,
                        lw["ws_w2"],
                        lw["b_w2"],
                        next_lw["norm1_w"],
                        xq_u_full,
                        xs_u_full,
                        eps=next_lw["norm1_eps"],
                        num_warps=4,
                    )
                elif (idx + 1) == fast_start:
                    _blend_backbone_w2_init_hu_and_norm1_kernel[(u_total,)](
                        hidden_t,
                        slot["blend_src_full"],
                        slot["blend_w"],
                        raw_w2,
                        xs_u_cur,
                        lw["ws_w2"],
                        lw["b_w2"],
                        next_lw["norm1_w"],
                        h_u,
                        xq_u_full,
                        xs_u_full,
                        eps=next_lw["norm1_eps"],
                        num_warps=4,
                    )
                else:
                    _gather_w2_res_rmsnorm_quant_tile_kernel[(m_full,)](
                        hidden_t,
                        raw_w2,
                        xs_u_cur,
                        slot["tile_to_u_full"],
                        lw["ws_w2"],
                        lw["b_w2"],
                        next_lw["norm1_w"],
                        xq_t_full,
                        xs_t_full,
                        eps=next_lw["norm1_eps"],
                        num_warps=4,
                    )
            else:
                normed_u = torch.empty(
                    (u_sp, 2048), device=self.device, dtype=torch.bfloat16
                )
                if is_fast:
                    _res_layernorm_bf16_unique_kernel[(u_sp,)](
                        h_u,
                        raw_w2,
                        xs_u_cur,
                        lw["ws_w2"],
                        lw["b_w2"],
                        self.norm_out_w,
                        self.norm_out_b,
                        normed_u,
                        eps=self.norm_out_eps,
                        num_warps=4,
                    )
                else:
                    _blend_backbone_w2_layernorm_unique_kernel[(u_sp,)](
                        hidden_t,
                        slot["blend_src_full"],
                        slot["blend_w"],
                        raw_w2,
                        xs_u_cur,
                        lw["ws_w2"],
                        lw["b_w2"],
                        self.norm_out_w,
                        self.norm_out_b,
                        normed_u,
                        eps=self.norm_out_eps,
                        num_warps=4,
                    )
                pix_u = F.linear(normed_u, self.proj_out_w, self.proj_out_b)
                _unpatchify_unique_rgb4_kernel[
                    (batch_n * 12, slot["H_out"], triton.cdiv(slot["W_out"], 128))
                ](
                    pix_u,
                    self.pixel_mean_1d,
                    self.pixel_std_1d,
                    slot["rgb4"],
                    ratio=ratio,
                    h_lat=h_lat,
                    w_lat=w_lat,
                    h_out=slot["H_out"],
                    w_out=slot["W_out"],
                    block_x=128,
                    num_warps=4,
                )

    def _get_or_build_slot(self, slot_key: tuple[int, int, int]) -> dict | None:
        if slot_key in self.slots:
            return self.slots[slot_key]

        # A deployment serves one active session. Keep only its most recent
        # geometry instead of retaining a Graph/KV pool for every past canvas.
        # Never evict a slot still owned by an active stream.
        if any(slot["in_use"] for slot in self.slots.values()):
            return None
        self.clear()

        batch_n, h_lat, w_lat = slot_key
        ratio = self.spatial_ratio
        h_out, w_out = h_lat * ratio, w_lat * ratio
        z_shape = (batch_n, 24, 1, h_lat, w_lat)

        with (
            torch.inference_mode(False),
            torch.no_grad(),
            torch.amp.autocast("cuda", enabled=False),
        ):
            ys, hs, oy = split_tiles(
                h_out, size=self.tile_size, overlap=self.tile_overlap, ratio=ratio
            )
            xs, ws, ox = split_tiles(
                w_out, size=self.tile_size, overlap=self.tile_overlap, ratio=ratio
            )
            tiles_total = len(ys) * len(xs)
            total_b = batch_n * tiles_total
            umaps = _build_unique_tile_maps(
                batch_n, h_lat, w_lat, ys, hs, oy, xs, ws, ox, ratio, 5, self.device
            )

            num_layers = len(self.layers)
            v_rings = [
                torch.zeros(
                    (total_b, 1541, 32, 64),
                    device=self.device,
                    dtype=torch.float8_e4m3fn,
                )
                for _ in range(num_layers)
            ]
            pool = torch.cuda.graph_pool_handle()
            slot = {
                **umaps,
                "N": batch_n,
                "H_lat": h_lat,
                "W_lat": w_lat,
                "H_out": h_out,
                "W_out": w_out,
                "B": total_b,
                "in_use": False,
                "z_in": torch.zeros(z_shape, device=self.device, dtype=torch.float32),
                "phase_idx": torch.zeros((1,), device=self.device, dtype=torch.int32),
                "k_u_ring_u32": torch.zeros(
                    (max(1, len(self.qkv_u_slot_map)), 4, umaps["U_sp"], 32, 32),
                    device=self.device,
                    dtype=torch.uint32,
                ),
                "k_t_ring_u32": torch.zeros(
                    (max(1, len(self.qkv_t_slot_map)), total_b, 1024, 32, 32),
                    device=self.device,
                    dtype=torch.uint32,
                ),
                "v_rings": v_rings,
                "rgb4": torch.zeros(
                    (batch_n, 3, 4, h_out, w_out),
                    device=self.device,
                    dtype=torch.float32,
                ),
                "pool": pool,
                "graphs": {},
            }

            # Capture all 6 temporal ring modes upfront on a private capture stream so
            # subsequent real-time steps never stall on lazy JIT/graph capture.
            caller_stream = torch.cuda.current_stream(self.device)
            capture_stream = torch.cuda.Stream(device=self.device)
            capture_stream.wait_stream(caller_stream)
            with torch.cuda.stream(capture_stream):
                for mode_id in (-2, -1, 0, 1, 2, 3):
                    for _ in range(2):
                        self._forward_impl(slot, mode_id)
                capture_stream.synchronize()
                for mode_id in (-2, -1, 0, 1, 2, 3):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
                        self._forward_impl(slot, mode_id)
                    slot["graphs"][mode_id] = graph
            caller_stream.wait_stream(capture_stream)

        self.slots[slot_key] = slot
        return slot

    def decode_step(
        self, z_t: torch.Tensor, state, phase: int, *, clamp_output: bool = False
    ) -> torch.Tensor | None:
        if not _supports_step_input(self.device, z_t):
            return None

        slot_key = (z_t.shape[0], z_t.shape[-2], z_t.shape[-1])
        step_idx = state.latent_index
        if step_idx == 0:
            slot = self._get_or_build_slot(slot_key)
            if slot is None or slot["in_use"]:
                return None
            slot["in_use"] = True
            state.sm90_lease = _SlotLease(self, slot_key, slot)
        else:
            lease = state.sm90_lease
            if (
                not isinstance(lease, _SlotLease)
                or not lease.active
                or lease.engine is not self
                or lease.slot_key != slot_key
            ):
                return None
            slot = lease.slot

        mode_id = (
            -2 if step_idx == 0 else (-1 if step_idx == 1 else ((step_idx - 2) & 3))
        )
        with (
            torch.inference_mode(False),
            torch.no_grad(),
            torch.amp.autocast("cuda", enabled=False),
        ):
            slot["z_in"].copy_(z_t)
            slot["phase_idx"].fill_(phase)
            slot["graphs"][mode_id].replay()
            rgb4 = slot["rgb4"]
            raw_view = rgb4[:, :, 3:] if phase == 0 else rgb4
            return raw_view.clamp(0, 1) if clamp_output else raw_view.clone()

    def matches_model(self, model, device: torch.device, signature: tuple) -> bool:
        if (
            self.signature != signature
            or self.device != device
            or getattr(model, "compute_dtype", None) != torch.bfloat16
            or model.config != self._expected_config
        ):
            return False
        for m in self._watched_modules:
            if _has_custom_forward(m):
                return False
        return True


def get_or_create_sm90_engine(execution, model, z_t: torch.Tensor):
    device = model.latent_mean.device
    signature = getattr(execution, "_step_signature", None)
    if signature is None:
        signature = parameter_signature(model)
    engine = execution.sm90_engine
    if engine is not None and engine.matches_model(model, device, signature):
        return engine
    if not _supports_sm90_model(model, device):
        return None
    if engine is not None:
        engine.clear()
    engine = LynnSm90FastEngine(model, signature)
    execution.sm90_engine = engine
    return engine
