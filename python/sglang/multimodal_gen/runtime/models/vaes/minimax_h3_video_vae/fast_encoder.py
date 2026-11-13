# SPDX-License-Identifier: Apache-2.0
"""High-throughput single-GPU execution engine for the MiniMax H3 visual VAE encoder.

Preserves exact spatial tiling geometry, causal receptive fields, stride phases,
transactional session state (``pending -> commit``), and zero retained session
bytes after clip reset or ``dispose()``. Shared model parameters are never
mutated in place.
"""

from __future__ import annotations

import math
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.util import find_spec

import torch
import torch.nn.functional as F

from .conv import BaseConv3d
from .norm import FusedGroupNorm3D, TemporalIsolatedGroupNorm
from .vae_cnn import Downsample3D, EncoderFCN3D

_MAX_32BIT_ELEMS = (1 << 31) - 1
_CUDNN_BENCHMARK_LIMIT_LOCK = threading.RLock()
_MODEL_METHOD_HOOKS = (
    "forward",
    "encode",
    "_encode",
    "_adaptive_encode",
    "tiled_encode",
    "split_tiles",
    "_assemble_tiles",
)


def encoder_numerical_optimizations_enabled() -> bool:
    """Allow compute rewrites that can change BF16 rounding; configure before startup."""
    name = "MINIMAX_H3_ENCODER_NUMERICAL_OPTIMIZATIONS"
    value = os.environ.get(name, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def triton_available() -> bool:
    return torch.cuda.is_available() and find_spec("triton") is not None


def _collect_encoder_submodules(model) -> tuple[torch.nn.Module, ...]:
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, EncoderFCN3D):
        return ()
    mods: list[torch.nn.Module] = [
        encoder,
        encoder.conv_in,
        encoder.norm_out,
        encoder.conv_out,
    ]
    for down in encoder.down:
        mods.append(down)
        for block in down.block:
            mods.extend((block, block.norm1, block.conv1, block.norm2, block.conv2))
            if block.in_channels != block.out_channels:
                mods.append(block.nin_shortcut)
        if hasattr(down, "downsample"):
            ds = down.downsample
            mods.append(ds)
            if isinstance(ds, Downsample3D):
                mods.append(ds.conv)
    return tuple(mods)


def has_custom_encoder_hooks_or_patches(
    model,
    *,
    assets: FastSingleGpuEncoderAssets | None = None,
    allow_quant_conv_forward: bool = False,
) -> bool:
    if any(name in model.__dict__ for name in _MODEL_METHOD_HOOKS):
        return True
    if getattr(model, "_forward_hooks", None) or getattr(
        model, "_forward_pre_hooks", None
    ):
        return True
    quant_conv = getattr(model, "quant_conv", None)
    if quant_conv is not None:
        if not allow_quant_conv_forward and "forward" in quant_conv.__dict__:
            return True
        if quant_conv._forward_hooks or quant_conv._forward_pre_hooks:
            return True
    submodules = (
        assets.guarded_modules
        if assets is not None
        else _collect_encoder_submodules(model)
    )
    for mod in submodules:
        if "forward" in mod.__dict__ or mod._forward_hooks or mod._forward_pre_hooks:
            return True
    return False


def is_fast_single_gpu_model_supported(model) -> bool:
    encoder = getattr(model, "encoder", None)
    quant_conv = getattr(model, "quant_conv", None)
    if not isinstance(encoder, EncoderFCN3D):
        return False
    if len(encoder.down) == 0 or any(len(d.block) == 0 for d in encoder.down):
        return False
    if not isinstance(quant_conv, torch.nn.Conv3d):
        return False
    if (
        quant_conv.kernel_size != (1, 1, 1)
        or quant_conv.stride != (1, 1, 1)
        or quant_conv.padding != (0, 0, 0)
        or quant_conv.groups != 1
    ):
        return False

    def _valid_norm(norm) -> bool:
        if isinstance(norm, TemporalIsolatedGroupNorm):
            pass
        elif isinstance(norm, FusedGroupNorm3D):
            if (
                not norm.use_t_isolated_gn
                or norm.use_spatial_affine
                or norm.activation != "silu"
            ):
                return False
        else:
            return False
        g = int(norm.num_groups)
        if g < 1 or (g & (g - 1)) != 0 or norm.num_channels % g != 0:
            return False
        d = norm.num_channels // g
        return d >= 1 and (d & (d - 1)) == 0

    def _valid_conv3(conv: BaseConv3d, expected_pad_hw: tuple[int, int]) -> bool:
        return (
            isinstance(conv, BaseConv3d)
            and conv.causal
            and conv.pad_mode_t == "constant"
            and conv.pad_mode in ("reflect", "constant")
            and conv.padding[1:] == expected_pad_hw
        )

    if not _valid_conv3(encoder.conv_in, encoder.conv_in.padding[1:]):
        return False
    if encoder.conv_in.in_channels != 3:
        return False

    for down in encoder.down:
        for block in down.block:
            if not (_valid_norm(block.norm1) and _valid_norm(block.norm2)):
                return False
            if block.conv1.padding[1:] not in ((0, 0), (1, 1)):
                return False
            if block.conv2.padding[1:] not in ((0, 0), (1, 1)):
                return False
            if not (
                _valid_conv3(block.conv1, block.conv1.padding[1:])
                and _valid_conv3(block.conv2, block.conv2.padding[1:])
            ):
                return False
            if block.in_channels != block.out_channels:
                sc = block.nin_shortcut
                if (
                    not isinstance(sc, BaseConv3d)
                    or sc.kernel_size != (1, 1, 1)
                    or sc.stride != (1, 1, 1)
                    or sc.padding != (0, 0, 0)
                    or sc.groups != 1
                ):
                    return False
        if hasattr(down, "downsample"):
            ds = down.downsample
            if isinstance(ds, Downsample3D):
                if ds.space_stride not in (1, 2) or not _valid_conv3(ds.conv, (0, 0)):
                    return False
            elif isinstance(ds, BaseConv3d):
                if ds.padding[1:] != (0, 0) or not _valid_conv3(ds, (0, 0)):
                    return False
            else:
                return False

    if not _valid_norm(encoder.norm_out):
        return False
    if encoder.conv_out.padding[1:] not in ((0, 0), (1, 1)) or not _valid_conv3(
        encoder.conv_out, encoder.conv_out.padding[1:]
    ):
        return False
    return True


@dataclass(frozen=True)
class TileBlendLUT:
    out_h: int
    out_w: int
    tile_h: int
    tile_w: int
    idx_cur: torch.Tensor
    idx_up: torch.Tensor
    idx_left: torch.Tensor
    wy0: torch.Tensor
    wy1: torch.Tensor
    wx0: torch.Tensor
    wx1: torch.Tensor
    has_y: torch.Tensor
    has_x: torch.Tensor


@dataclass(frozen=True)
class TileGeometryPlan:
    ys: tuple[int, ...]
    xs: tuple[int, ...]
    yo: tuple[int, ...]
    xo: tuple[int, ...]
    tile_h: int
    tile_w: int
    num_tiles: int
    vae_ratio: int
    tile_y0: torch.Tensor
    tile_x0: torch.Tensor
    blend_lut: TileBlendLUT
    geometry_tuple: tuple


@dataclass(frozen=True)
class BlockExecutionPlan:
    stage_idx: int
    block_idx: int
    norm1: torch.nn.Module
    conv1: BaseConv3d
    w1_cl: torch.Tensor
    b1: torch.Tensor | None
    norm2: torch.nn.Module
    conv2: BaseConv3d
    w2_cl: torch.Tensor
    has_shortcut: bool
    w_short_2d: torch.Tensor | None
    incoming_bias: torch.Tensor | None
    fused_end_bias: torch.Tensor | None
    fuse_into_downsample: bool
    fuse_into_norm_out: bool


@dataclass(frozen=True)
class DownsampleExecutionPlan:
    stage_idx: int
    conv: BaseConv3d
    w_cl: torch.Tensor
    pad_bottom_right: bool
    reflect: bool


@dataclass
class FastSingleGpuEncoderAssets:
    device: torch.device
    dtype: torch.dtype
    conv_in: BaseConv3d
    w_in_cl: torch.Tensor
    blocks_by_stage: tuple[tuple[BlockExecutionPlan, ...], ...]
    downsamples: dict[int, DownsampleExecutionPlan]
    tail_incoming_bias: torch.Tensor | None
    norm_out: torch.nn.Module
    conv_out: BaseConv3d
    w_out_cl: torch.Tensor
    quant_w_2d: torch.Tensor
    fused_quant_bias: torch.Tensor | None
    guarded_modules: tuple[torch.nn.Module, ...]
    geometry_lock: threading.RLock = field(default_factory=threading.RLock)
    geometry_cache: dict[tuple[int, int, bool, torch.dtype], TileGeometryPlan] = field(
        default_factory=dict
    )
    pinned_lock: threading.Lock = field(default_factory=threading.Lock)
    pinned_slots: list[torch.Tensor] = field(default_factory=list)
    pinned_events: list[torch.cuda.Event] = field(default_factory=list)
    pinned_cursor: int = 0


def stage_uint8_to_gpu(
    assets: FastSingleGpuEncoderAssets, frames: torch.Tensor
) -> torch.Tensor:
    """Async DMA uint8 frames from CPU (including non-contiguous slices) via a 2-slot pinned ring."""
    if frames.device == assets.device:
        return frames
    if (
        frames.device.type != "cpu"
        or os.environ.get("MINIMAX_H3_FAST_PINNED_H2D", "1") != "1"
    ):
        return frames.to(device=assets.device, non_blocking=False)

    needed = int(frames.numel())
    if needed == 0 or not assets.pinned_lock.acquire(blocking=False):
        return frames.to(device=assets.device, non_blocking=False)
    try:
        if not assets.pinned_slots or assets.pinned_slots[0].numel() < needed:
            # Wait for any in-flight async H2D DMA before releasing old pinned host buffers
            for ev in assets.pinned_events:
                ev.synchronize()
            # Cap reusable pinned staging at ~32 MiB per slot unless larger frame batch arrives
            cap = max(needed, 3 * 8 * 768 * 1344)
            assets.pinned_slots = [
                torch.empty((cap,), dtype=torch.uint8, pin_memory=True),
                torch.empty((cap,), dtype=torch.uint8, pin_memory=True),
            ]
            assets.pinned_events = [
                torch.cuda.Event(enable_timing=False),
                torch.cuda.Event(enable_timing=False),
            ]
            assets.pinned_cursor = 0

        slot = assets.pinned_cursor
        assets.pinned_cursor = (slot + 1) & 1
        assets.pinned_events[slot].synchronize()
        pinned_view = assets.pinned_slots[slot][:needed].view(frames.shape)
        pinned_view.copy_(frames)
        gpu_frames = torch.empty(frames.shape, dtype=torch.uint8, device=assets.device)
        gpu_frames.copy_(pinned_view, non_blocking=True)
        assets.pinned_events[slot].record(torch.cuda.current_stream(assets.device))
        return gpu_frames
    finally:
        assets.pinned_lock.release()


def _sum_biases(
    a: torch.Tensor | None, b: torch.Tensor | None, dtype: torch.dtype
) -> torch.Tensor | None:
    if a is None and b is None:
        return None
    if a is None:
        return b.detach().to(dtype=dtype).contiguous()
    if b is None:
        return a.detach().to(dtype=dtype).contiguous()
    return (a.detach().float() + b.detach().float()).to(dtype=dtype).contiguous()


def prepare_fast_single_gpu_encoder(model) -> FastSingleGpuEncoderAssets | None:
    if not encoder_numerical_optimizations_enabled():
        return None
    if not is_fast_single_gpu_model_supported(model):
        return None
    param = next(model.encoder.parameters())
    if param.device.type != "cuda" or param.dtype != torch.bfloat16:
        return None
    if next(model.quant_conv.parameters()).dtype != param.dtype:
        return None

    device = param.device
    dtype = param.dtype
    existing_cl = getattr(model, "_progressive_encoder_weights", None) or {}

    def _cl_weight(conv: torch.nn.Conv3d) -> torch.Tensor:
        w = existing_cl.get(conv)
        if w is not None and w.device == device and w.dtype == dtype:
            return w
        return conv.weight.detach().clone(memory_format=torch.channels_last_3d)

    encoder: EncoderFCN3D = model.encoder
    w_in_cl = _cl_weight(encoder.conv_in)
    pending_bias = (
        encoder.conv_in.bias.detach().clone()
        if encoder.conv_in.bias is not None
        else None
    )

    num_stages = len(encoder.down)
    blocks_by_stage: list[tuple[BlockExecutionPlan, ...]] = []
    downsamples: dict[int, DownsampleExecutionPlan] = {}

    for i, down in enumerate(encoder.down):
        has_ds = hasattr(down, "downsample")
        ds_plan: DownsampleExecutionPlan | None = None
        if has_ds:
            ds_mod = down.downsample
            if isinstance(ds_mod, Downsample3D):
                ds_conv = ds_mod.conv
                ds_plan = DownsampleExecutionPlan(
                    stage_idx=i,
                    conv=ds_conv,
                    w_cl=_cl_weight(ds_conv),
                    pad_bottom_right=(ds_mod.space_stride == 2),
                    reflect=(ds_mod.pad_mode == "reflect"),
                )
            else:
                ds_conv = ds_mod
                ds_plan = DownsampleExecutionPlan(
                    stage_idx=i,
                    conv=ds_conv,
                    w_cl=_cl_weight(ds_conv),
                    pad_bottom_right=False,
                    reflect=(ds_conv.pad_mode == "reflect"),
                )
            downsamples[i] = ds_plan

        stage_blocks: list[BlockExecutionPlan] = []
        num_blocks = len(down.block)
        for j, block in enumerate(down.block):
            is_last_in_stage = j == num_blocks - 1
            has_shortcut = block.in_channels != block.out_channels
            incoming = pending_bias

            fuse_into_ds = (
                is_last_in_stage
                and not has_shortcut
                and ds_plan is not None
                and ds_plan.conv.kernel_size != (1, 1, 1)
            )
            fuse_into_norm_out = (
                is_last_in_stage
                and i == num_stages - 1
                and not has_shortcut
                and not has_ds
            )

            if has_shortcut:
                sc = block.nin_shortcut
                w_short_2d = (
                    sc.weight.detach()
                    .reshape(sc.out_channels, sc.in_channels)
                    .contiguous()
                )
                base_b = _sum_biases(sc.bias, block.conv2.bias, torch.float32)
                if incoming is not None:
                    projected = F.linear(incoming.float(), w_short_2d.float(), base_b)
                    fused_end_bias = projected.to(dtype=dtype).contiguous()
                else:
                    fused_end_bias = (
                        base_b.to(dtype=dtype).contiguous()
                        if base_b is not None
                        else None
                    )
                pending_bias = None
            else:
                w_short_2d = None
                fused_end_bias = _sum_biases(block.conv2.bias, incoming, dtype)
                pending_bias = None

            stage_blocks.append(
                BlockExecutionPlan(
                    stage_idx=i,
                    block_idx=j,
                    norm1=block.norm1,
                    conv1=block.conv1,
                    w1_cl=_cl_weight(block.conv1),
                    b1=(
                        block.conv1.bias.detach().contiguous()
                        if block.conv1.bias is not None
                        else None
                    ),
                    norm2=block.norm2,
                    conv2=block.conv2,
                    w2_cl=_cl_weight(block.conv2),
                    has_shortcut=has_shortcut,
                    w_short_2d=w_short_2d,
                    incoming_bias=incoming,
                    fused_end_bias=fused_end_bias,
                    fuse_into_downsample=fuse_into_ds,
                    fuse_into_norm_out=fuse_into_norm_out,
                )
            )

        if ds_plan is not None:
            pending_bias = (
                ds_plan.conv.bias.detach().contiguous()
                if ds_plan.conv.bias is not None
                else None
            )
        blocks_by_stage.append(tuple(stage_blocks))

    quant_conv = model.quant_conv
    quant_w_2d = (
        quant_conv.weight.detach()
        .reshape(quant_conv.out_channels, quant_conv.in_channels)
        .contiguous()
    )
    if encoder.conv_out.bias is not None:
        fused_quant_bias = (
            F.linear(
                encoder.conv_out.bias.detach().float(),
                quant_w_2d.float(),
                (
                    quant_conv.bias.detach().float()
                    if quant_conv.bias is not None
                    else None
                ),
            )
            .to(dtype=dtype)
            .contiguous()
        )
    else:
        fused_quant_bias = (
            quant_conv.bias.detach().contiguous()
            if quant_conv.bias is not None
            else None
        )

    assets = FastSingleGpuEncoderAssets(
        device=device,
        dtype=dtype,
        conv_in=encoder.conv_in,
        w_in_cl=w_in_cl,
        blocks_by_stage=tuple(blocks_by_stage),
        downsamples=downsamples,
        tail_incoming_bias=pending_bias,
        norm_out=encoder.norm_out,
        conv_out=encoder.conv_out,
        w_out_cl=_cl_weight(encoder.conv_out),
        quant_w_2d=quant_w_2d,
        fused_quant_bias=fused_quant_bias,
        guarded_modules=_collect_encoder_submodules(model),
    )
    model._fast_single_gpu_encoder = assets
    return assets


def _has_reflect_spatial_pad(encoder: EncoderFCN3D) -> bool:
    if encoder.conv_in.pad_mode == "reflect" and encoder.conv_in.padding[1:] != (0, 0):
        return True
    if encoder.conv_out.pad_mode == "reflect" and encoder.conv_out.padding[1:] != (
        0,
        0,
    ):
        return True
    for down in encoder.down:
        for block in down.block:
            if (
                block.conv1.pad_mode == "reflect" and block.conv1.padding[1:] != (0, 0)
            ) or (
                block.conv2.pad_mode == "reflect" and block.conv2.padding[1:] != (0, 0)
            ):
                return True
        if hasattr(down, "downsample"):
            ds = down.downsample
            if (
                isinstance(ds, Downsample3D)
                and ds.space_stride == 2
                and ds.conv.pad_mode == "reflect"
            ):
                return True
    return False


def _build_tile_geometry_plan(
    model,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TileGeometryPlan:
    if height <= 0 or width <= 0:
        raise ValueError("Input spatial dimensions must be positive")
    vae_ratio = int(getattr(model, "vae_ratio", 16))
    if model.encoder_tiling:
        ys, hs, yo = model.split_tiles(height, False)
        xs, ws, xo = model.split_tiles(width, False)
    else:
        ys, hs, yo = [0], [height], []
        xs, ws, xo = [0], [width], []

    # split_tiles reports nominal sizes. Native slicing clips an edge tile to
    # the canvas, but the fused pack kernels read the full planned rectangle.
    if any(y < 0 or y + h > height for y, h in zip(ys, hs)) or any(
        x < 0 or x + w > width for x, w in zip(xs, ws)
    ):
        raise ValueError("Fast single-GPU encoder requires tiles within the canvas")

    if len(set(hs)) != 1 or len(set(ws)) != 1:
        raise ValueError("Fast single-GPU encoder requires uniform tile dimensions")
    tile_h, tile_w = int(hs[0]), int(ws[0])
    min_tile = (
        2 * vae_ratio if _has_reflect_spatial_pad(model.encoder) else max(1, vae_ratio)
    )
    if (
        tile_h < min_tile
        or tile_w < min_tile
        or tile_h % vae_ratio != 0
        or tile_w % vae_ratio != 0
    ):
        raise ValueError(
            "Tile dimensions must be divisible by vae_ratio and support spatial padding"
        )

    y0_list = [y for y in ys for _ in xs]
    x0_list = [x for _ in ys for x in xs]
    num_tiles = len(y0_list)
    tile_shapes = tuple((tile_h, tile_w) for _ in range(num_tiles))
    geometry_tuple = (
        tuple(ys),
        tuple(xs),
        tuple(yo),
        tuple(xo),
        tile_shapes,
        getattr(model, "vae_ratio", None),
    )

    yo_lat = [v // vae_ratio for v in yo]
    xo_lat = [v // vae_ratio for v in xo]
    n_rows, n_cols = len(ys), len(xs)
    th = tile_h // vae_ratio
    tw = tile_w // vae_ratio
    out_h = sum(th - (yo_lat[i] if i < n_rows - 1 else 0) for i in range(n_rows))
    out_w = sum(tw - (xo_lat[j] if j < n_cols - 1 else 0) for j in range(n_cols))

    idx_cur = torch.zeros((out_h * out_w,), dtype=torch.int32)
    idx_up = torch.zeros((out_h * out_w,), dtype=torch.int32)
    idx_left = torch.zeros((out_h * out_w,), dtype=torch.int32)
    wy0 = torch.zeros((out_h * out_w,), dtype=dtype)
    wy1 = torch.ones((out_h * out_w,), dtype=dtype)
    wx0 = torch.zeros((out_h * out_w,), dtype=dtype)
    wx1 = torch.ones((out_h * out_w,), dtype=dtype)
    has_y = torch.zeros((out_h * out_w,), dtype=torch.bool)
    has_x = torch.zeros((out_h * out_w,), dtype=torch.bool)

    y_off = 0
    for i in range(n_rows):
        rh = th - (yo_lat[i] if i < n_rows - 1 else 0)
        ey = yo_lat[i - 1] if i > 0 else 0
        if ey > 0:
            pos_y = torch.arange(ey, dtype=dtype)
            wy_a, wy_b = 1 - pos_y / ey, pos_y / ey
        x_off = 0
        for j in range(n_cols):
            cw = tw - (xo_lat[j] if j < n_cols - 1 else 0)
            ex = xo_lat[j - 1] if j > 0 else 0
            if ex > 0:
                pos_x = torch.arange(ex, dtype=dtype)
                wx_a, wx_b = 1 - pos_x / ex, pos_x / ex
            for ly in range(rh):
                for lx in range(cw):
                    p = (y_off + ly) * out_w + (x_off + lx)
                    idx_cur[p] = (i * n_cols + j) * (th * tw) + ly * tw + lx
                    if i > 0 and ly < ey:
                        has_y[p] = True
                        sy_prev = th - ey + ly
                        idx_up[p] = (
                            ((i - 1) * n_cols + j) * (th * tw) + sy_prev * tw + lx
                        )
                        wy0[p] = wy_a[ly]
                        wy1[p] = wy_b[ly]
                    if j > 0 and lx < ex:
                        has_x[p] = True
                        sx_prev = tw - ex + lx
                        idx_left[p] = (
                            (i * n_cols + (j - 1)) * (th * tw) + ly * tw + sx_prev
                        )
                        wx0[p] = wx_a[lx]
                        wx1[p] = wx_b[lx]
            x_off += cw
        y_off += rh

    lut = TileBlendLUT(
        out_h=out_h,
        out_w=out_w,
        tile_h=th,
        tile_w=tw,
        idx_cur=idx_cur.to(device=device),
        idx_up=idx_up.to(device=device),
        idx_left=idx_left.to(device=device),
        wy0=wy0.to(device=device),
        wy1=wy1.to(device=device),
        wx0=wx0.to(device=device),
        wx1=wx1.to(device=device),
        has_y=has_y.to(device=device),
        has_x=has_x.to(device=device),
    )
    return TileGeometryPlan(
        ys=tuple(ys),
        xs=tuple(xs),
        yo=tuple(yo),
        xo=tuple(xo),
        tile_h=tile_h,
        tile_w=tile_w,
        num_tiles=num_tiles,
        vae_ratio=vae_ratio,
        tile_y0=torch.tensor(y0_list, dtype=torch.int32, device=device),
        tile_x0=torch.tensor(x0_list, dtype=torch.int32, device=device),
        blend_lut=lut,
        geometry_tuple=geometry_tuple,
    )


def get_tile_geometry_plan(
    assets: FastSingleGpuEncoderAssets,
    model,
    height: int,
    width: int,
) -> TileGeometryPlan:
    key = (height, width, bool(model.encoder_tiling), assets.dtype)
    plan = assets.geometry_cache.get(key)
    if plan is not None:
        return plan
    with assets.geometry_lock:
        plan = assets.geometry_cache.get(key)
        if plan is None:
            plan = _build_tile_geometry_plan(
                model, height, width, device=assets.device, dtype=assets.dtype
            )
            assets.geometry_cache[key] = plan
        return plan


@contextmanager
def _fast_cudnn_plan_policy():
    """Allow cuDNN to benchmark top SM90 WGMMA plans instead of stopping at plan #1."""
    raw = os.environ.get("MINIMAX_H3_FAST_ENCODER_BENCHMARK_LIMIT", "8")
    target_limit = int(raw)
    with _CUDNN_BENCHMARK_LIMIT_LOCK:
        prev_limit = torch.backends.cudnn.benchmark_limit
        changed = (
            torch.backends.cudnn.benchmark
            and not torch.backends.cudnn.deterministic
            and prev_limit is not None
            and target_limit >= 0
            and prev_limit != target_limit
        )
        if changed:
            torch.backends.cudnn.benchmark_limit = target_limit
        try:
            yield
        finally:
            if changed:
                torch.backends.cudnn.benchmark_limit = prev_limit


def _conv3d_nthwc(
    padded_nthwc: torch.Tensor,
    weight_cl: torch.Tensor,
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
    groups: int,
) -> torch.Tensor:
    max_elems = int(
        os.environ.get("MINIMAX_H3_FAST_CONV_MAX_ELEMS", str(_MAX_32BIT_ELEMS))
    )
    b, t_in, h_in, w_in, _ = padded_nthwc.shape
    kt, kh, kw = weight_cl.shape[2:]
    dt, dh, dw = dilation
    st, sh, sw = stride
    t_out = (t_in - dt * (kt - 1) - 1) // st + 1
    h_out = (h_in - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in - dw * (kw - 1) - 1) // sw + 1
    c_out = weight_cl.shape[0]
    total_in = padded_nthwc.numel()
    total_out = b * t_out * h_out * w_out * c_out
    if max(total_in, total_out) <= max_elems or b == 1:
        out_5d = F.conv3d(
            padded_nthwc.permute(0, 4, 1, 2, 3),
            weight_cl,
            None,
            stride,
            0,
            dilation,
            groups,
        )
        out_nthwc = out_5d.permute(0, 2, 3, 4, 1)
        return out_nthwc if out_nthwc.is_contiguous() else out_nthwc.contiguous()

    per_tile = max(total_in // b, total_out // b)
    chunk_b = max(1, max_elems // per_tile)
    out_nthwc = torch.empty(
        (b, t_out, h_out, w_out, c_out),
        dtype=padded_nthwc.dtype,
        device=padded_nthwc.device,
    )
    for start in range(0, b, chunk_b):
        end = min(b, start + chunk_b)
        part_5d = F.conv3d(
            padded_nthwc[start:end].permute(0, 4, 1, 2, 3),
            weight_cl,
            None,
            stride,
            0,
            dilation,
            groups,
        )
        out_nthwc[start:end].copy_(part_5d.permute(0, 2, 3, 4, 1))
    return out_nthwc


def run_fast_single_gpu_step(
    model,
    assets: FastSingleGpuEncoderAssets,
    *,
    frames_uint8: torch.Tensor | None = None,
    pixels_bf16: torch.Tensor | None = None,
    needed_frames: int | None = None,
    valid_latents: int | None = None,
    current_states: dict,
) -> tuple[
    torch.Tensor,
    dict,
    int,
    dict[str, int],
    dict[str, int],
    TileGeometryPlan,
]:
    """Execute one transactional progressive step across all spatial tiles.

    ``current_states`` is read-only; all updated causal histories are returned in
    a fresh ``pending_states`` dictionary for caller commit after validation.
    """
    from .fast_encoder_kernels import (
        fused_add_bias_inplace_nthwc,
        fused_gn_silu_history_pad_nthwc,
        fused_residual_bias_downsample_pad_nthwc,
        fused_tile_blend_nthwc_to_ncdhw,
        tile_pack_and_pad_bf16,
        tile_pack_and_pad_uint8,
    )

    if (frames_uint8 is None) == (pixels_bf16 is None):
        raise ValueError("Specify exactly one of frames_uint8 or pixels_bf16")

    if frames_uint8 is not None:
        height, width = int(frames_uint8.shape[2]), int(frames_uint8.shape[3])
        t_in = int(
            needed_frames if needed_frames is not None else frames_uint8.shape[1]
        )
        n_in = 1
    else:
        n_in, _, t_in, height, width = map(int, pixels_bf16.shape)
        if n_in != 1:
            raise ValueError("Fast single-GPU tile blend expects batch size 1")

    geom = get_tile_geometry_plan(assets, model, height, width)
    num_tiles = geom.num_tiles

    pending: dict = {}
    layer_work: dict[str, int] = {}
    work = 0
    opt_counts = {"norm_input": 0, "conv_input": 0, "fallback": 0}

    def _record_work(tag, t_count: int) -> None:
        nonlocal work
        positions = num_tiles * n_in * t_count
        work += positions
        key_str = str(tag)
        layer_work[key_str] = layer_work.get(key_str, 0) + positions

    def _state_step(conv_layer: BaseConv3d, tag, t_count: int):
        hist_len = conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1)
        prev_tensor, seen = current_states.get(("fast", tag), (None, 0))
        offset = (-seen) % conv_layer.stride[0]
        if offset >= t_count:
            raise ValueError("Incremental block produced no strided output")
        return hist_len, prev_tensor, seen, offset

    with _fast_cudnn_plan_policy():
        # 1. Input conv_in: fused uint8/bf16 normalize + tile slice + history + reflect pad
        hist_in, prev_in, seen_in, off_in = _state_step(assets.conv_in, "in", t_in)
        _record_work("in", t_in)
        ph_in, pw_in = assets.conv_in.padding[1], assets.conv_in.padding[2]
        reflect_in = assets.conv_in.pad_mode == "reflect"
        if frames_uint8 is not None:
            pad_in, next_in = tile_pack_and_pad_uint8(
                frames_uint8,
                prev_in,
                geom.tile_y0,
                geom.tile_x0,
                needed_frames=t_in,
                history=hist_in,
                offset=off_in,
                tile_h=geom.tile_h,
                tile_w=geom.tile_w,
                pad_h=ph_in,
                pad_w=pw_in,
                reflect=reflect_in,
                dtype=assets.dtype,
            )
        else:
            pad_in, next_in = tile_pack_and_pad_bf16(
                pixels_bf16,
                prev_in,
                geom.tile_y0,
                geom.tile_x0,
                history=hist_in,
                offset=off_in,
                tile_h=geom.tile_h,
                tile_w=geom.tile_w,
                pad_h=ph_in,
                pad_w=pw_in,
                reflect=reflect_in,
            )
        pending[("fast", "in")] = (next_in, seen_in + t_in)
        opt_counts["conv_input"] += num_tiles
        h = _conv3d_nthwc(
            pad_in,
            assets.w_in_cl,
            assets.conv_in.stride,
            assets.conv_in.dilation,
            assets.conv_in.groups,
        )
        del pad_in

        # 2. Six downsampling stages
        for i, stage_blocks in enumerate(assets.blocks_by_stage):
            ds_plan = assets.downsamples.get(i)
            for bplan in stage_blocks:
                j = bplan.block_idx
                t_cur = int(h.shape[1])
                residual = h

                # norm1 + SiLU + history + pad -> conv1
                tag1 = (i, j, 1)
                hist1, prev1, seen1, off1 = _state_step(bplan.conv1, tag1, t_cur)
                _record_work(tag1, t_cur)
                pad1, next1 = fused_gn_silu_history_pad_nthwc(
                    h,
                    bplan.norm1,
                    prev_hist=prev1,
                    history=hist1,
                    offset=off1,
                    pad_spatial=(bplan.conv1.padding[1:] == (1, 1)),
                    reflect=(bplan.conv1.pad_mode == "reflect"),
                    add_bias=bplan.incoming_bias,
                )
                pending[("fast", tag1)] = (next1, seen1 + t_cur)
                opt_counts["norm_input"] += num_tiles
                y1 = _conv3d_nthwc(
                    pad1,
                    bplan.w1_cl,
                    bplan.conv1.stride,
                    bplan.conv1.dilation,
                    bplan.conv1.groups,
                )
                del pad1

                # norm2 + SiLU + history + pad -> conv2
                t_mid = int(y1.shape[1])
                tag2 = (i, j, 2)
                hist2, prev2, seen2, off2 = _state_step(bplan.conv2, tag2, t_mid)
                _record_work(tag2, t_mid)
                pad2, next2 = fused_gn_silu_history_pad_nthwc(
                    y1,
                    bplan.norm2,
                    prev_hist=prev2,
                    history=hist2,
                    offset=off2,
                    pad_spatial=(bplan.conv2.padding[1:] == (1, 1)),
                    reflect=(bplan.conv2.pad_mode == "reflect"),
                    add_bias=bplan.b1,
                )
                del y1
                pending[("fast", tag2)] = (next2, seen2 + t_mid)
                opt_counts["norm_input"] += num_tiles
                y2 = _conv3d_nthwc(
                    pad2,
                    bplan.w2_cl,
                    bplan.conv2.stride,
                    bplan.conv2.dilation,
                    bplan.conv2.groups,
                )
                del pad2

                if bplan.has_shortcut:
                    tag_sc = (i, j, "shortcut")
                    _record_work(tag_sc, t_cur)
                    _, seen_sc = current_states.get(("fast", tag_sc), (None, 0))
                    pending[("fast", tag_sc)] = (None, seen_sc + t_cur)
                    opt_counts["conv_input"] += num_tiles
                    short_2d = F.linear(
                        residual.view(-1, residual.shape[-1]),
                        bplan.w_short_2d,
                        bplan.fused_end_bias,
                    )
                    del residual
                    y2.view(-1, y2.shape[-1]).add_(short_2d)
                    del short_2d
                    h = y2
                elif bplan.fuse_into_downsample:
                    assert ds_plan is not None
                    t_y2 = int(y2.shape[1])
                    tag_ds = (i, "down")
                    hist_ds, prev_ds, seen_ds, off_ds = _state_step(
                        ds_plan.conv, tag_ds, t_y2
                    )
                    _record_work(tag_ds, t_y2)
                    pad_ds, next_ds = fused_residual_bias_downsample_pad_nthwc(
                        y2,
                        residual_nthwc=residual,
                        add_bias=bplan.fused_end_bias,
                        prev_hist=prev_ds,
                        history=hist_ds,
                        offset=off_ds,
                        pad_bottom_right=ds_plan.pad_bottom_right,
                        reflect=ds_plan.reflect,
                    )
                    del y2, residual
                    pending[("fast", tag_ds)] = (next_ds, seen_ds + t_y2)
                    opt_counts["conv_input"] += num_tiles
                    h = _conv3d_nthwc(
                        pad_ds,
                        ds_plan.w_cl,
                        ds_plan.conv.stride,
                        ds_plan.conv.dilation,
                        ds_plan.conv.groups,
                    )
                    del pad_ds
                    ds_plan = None
                elif bplan.fuse_into_norm_out:
                    # Defer residual + fused_end_bias directly into encoder.norm_out
                    final_y2 = y2
                    final_residual = residual
                    final_bias = bplan.fused_end_bias
                    h = y2
                else:
                    h = fused_add_bias_inplace_nthwc(y2, residual, bplan.fused_end_bias)
                    del residual

            if ds_plan is not None:
                t_ds = int(h.shape[1])
                tag_ds = (i, "down")
                hist_ds, prev_ds, seen_ds, off_ds = _state_step(
                    ds_plan.conv, tag_ds, t_ds
                )
                _record_work(tag_ds, t_ds)
                if ds_plan.conv.kernel_size == (1, 1, 1):
                    pending[("fast", tag_ds)] = (None, seen_ds + t_ds)
                    opt_counts["conv_input"] += num_tiles
                    w_ds_2d = ds_plan.w_cl.reshape(
                        ds_plan.conv.out_channels, ds_plan.conv.in_channels
                    ).contiguous()
                    out_2d = F.linear(h.view(-1, h.shape[-1]), w_ds_2d, None)
                    h = out_2d.view(*h.shape[:-1], ds_plan.conv.out_channels)
                else:
                    pad_ds, next_ds = fused_residual_bias_downsample_pad_nthwc(
                        h,
                        residual_nthwc=None,
                        add_bias=None,
                        prev_hist=prev_ds,
                        history=hist_ds,
                        offset=off_ds,
                        pad_bottom_right=ds_plan.pad_bottom_right,
                        reflect=ds_plan.reflect,
                    )
                    del h
                    pending[("fast", tag_ds)] = (next_ds, seen_ds + t_ds)
                    opt_counts["conv_input"] += num_tiles
                    h = _conv3d_nthwc(
                        pad_ds,
                        ds_plan.w_cl,
                        ds_plan.conv.stride,
                        ds_plan.conv.dilation,
                        ds_plan.conv.groups,
                    )
                    del pad_ds

        # 3. Output norm_out + SiLU + history + pad -> conv_out -> quant_conv -> blend
        last_bplan = assets.blocks_by_stage[-1][-1]
        t_final = int(h.shape[1])
        hist_out, prev_out, seen_out, off_out = _state_step(
            assets.conv_out, "out", t_final
        )
        _record_work("out", t_final)
        if last_bplan.fuse_into_norm_out:
            pad_out, next_out = fused_gn_silu_history_pad_nthwc(
                final_y2,
                assets.norm_out,
                prev_hist=prev_out,
                history=hist_out,
                offset=off_out,
                pad_spatial=(assets.conv_out.padding[1:] == (1, 1)),
                reflect=(assets.conv_out.pad_mode == "reflect"),
                add_bias=final_bias,
                residual_nthwc=final_residual,
            )
            del final_y2, final_residual, h
        else:
            pad_out, next_out = fused_gn_silu_history_pad_nthwc(
                h,
                assets.norm_out,
                prev_hist=prev_out,
                history=hist_out,
                offset=off_out,
                pad_spatial=(assets.conv_out.padding[1:] == (1, 1)),
                reflect=(assets.conv_out.pad_mode == "reflect"),
                add_bias=assets.tail_incoming_bias,
            )
            del h
        pending[("fast", "out")] = (next_out, seen_out + t_final)
        opt_counts["norm_input"] += num_tiles

        z_raw = _conv3d_nthwc(
            pad_out,
            assets.w_out_cl,
            assets.conv_out.stride,
            assets.conv_out.dilation,
            assets.conv_out.groups,
        )
        del pad_out

        # Honor instance-level monkeypatches (e.g. fault-injection tests on quant_conv.forward)
        if "forward" in model.quant_conv.__dict__:
            if assets.conv_out.bias is not None:
                z_raw.view(-1, z_raw.shape[-1]).add_(assets.conv_out.bias)
            z_5d = model.quant_conv(z_raw.permute(0, 4, 1, 2, 3).contiguous())
            q_nthwc = z_5d.permute(0, 2, 3, 4, 1).contiguous()
        else:
            q_2d = F.linear(
                z_raw.view(-1, z_raw.shape[-1]),
                assets.quant_w_2d,
                assets.fused_quant_bias,
            )
            q_nthwc = q_2d.view(*z_raw.shape[:-1], assets.quant_w_2d.shape[0])
        del z_raw

        moments_ncdhw = fused_tile_blend_nthwc_to_ncdhw(
            q_nthwc, geom.blend_lut, valid_latents=valid_latents
        )
        del q_nthwc

    return moments_ncdhw, pending, work, layer_work, opt_counts, geom


@torch.no_grad()
def encode_uint8_sequence_fast_single_gpu(
    model,
    assets: FastSingleGpuEncoderAssets,
    frames_uint8: torch.Tensor,
    *,
    frames_per_chunk: int = 17,
    temporal_ratio: int = 4,
) -> torch.Tensor:
    """Encode a complete uint8 [3, T, H, W] image or video into exact causal moments."""
    if (
        frames_uint8.ndim != 4
        or frames_uint8.shape[0] != 3
        or frames_uint8.shape[1] < 1
        or frames_uint8.dtype != torch.uint8
    ):
        raise ValueError("Expected uint8 frames of shape [3, T, H, W]")
    if frames_per_chunk != 17 or temporal_ratio != 4:
        raise ValueError("Fast sequence encoder expects 17-frame / ratio-4 H3 chunks")

    height, width = int(frames_uint8.shape[2]), int(frames_uint8.shape[3])
    get_tile_geometry_plan(assets, model, height, width)

    total_frames = int(frames_uint8.shape[1])
    outputs: list[torch.Tensor] = []
    for start in range(0, total_frames, frames_per_chunk):
        clip = frames_uint8[:, start : start + frames_per_chunk]
        n = int(clip.shape[1])
        keep_latents = 1 + math.ceil((n - 1) / temporal_ratio)

        f0 = stage_uint8_to_gpu(assets, clip[:, :1])
        m0, st, _, _, _, _ = run_fast_single_gpu_step(
            model,
            assets,
            frames_uint8=f0,
            needed_frames=1,
            valid_latents=1,
            current_states={},
        )
        del f0
        parts = [m0]

        if keep_latents > 1:
            f1 = stage_uint8_to_gpu(assets, clip[:, 1 : min(n, 9)])
            m1, st, _, _, _, _ = run_fast_single_gpu_step(
                model,
                assets,
                frames_uint8=f1,
                needed_frames=8,
                valid_latents=min(2, keep_latents - 1),
                current_states=st,
            )
            del f1
            parts.append(m1)

        if keep_latents > 3:
            f2 = stage_uint8_to_gpu(assets, clip[:, 9:n])
            m2, st, _, _, _, _ = run_fast_single_gpu_step(
                model,
                assets,
                frames_uint8=f2,
                needed_frames=8,
                valid_latents=keep_latents - 3,
                current_states=st,
            )
            del f2
            parts.append(m2)

        del st
        outputs.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=2))

    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)


@torch.no_grad()
def warmup_fast_single_gpu_encoder(
    model, assets: FastSingleGpuEncoderAssets | None = None
) -> None:
    """Pre-compile Triton kernels and benchmark SM90 cuDNN plans for 480p and 736p."""
    if not triton_available():
        return
    if assets is None:
        assets = getattr(model, "_fast_single_gpu_encoder", None)
    if assets is None:
        return

    raw_shapes = os.environ.get("MINIMAX_H3_FAST_ENCODER_WARMUP_SHAPES")
    shapes: list[tuple[int, int]] = []
    if raw_shapes is not None:
        for token in raw_shapes.split(","):
            token = token.strip().lower()
            if not token:
                continue
            h_str, _, w_str = token.partition("x")
            shapes.append((int(h_str), int(w_str)))
    elif (
        bool(getattr(model, "encoder_tiling", False))
        and int(getattr(model, "tile_size", 0)) >= 256
    ):
        shapes = [(480, 832), (736, 1280)]

    for height, width in shapes:
        st: dict = {}
        try:
            for t_step, v_lat in ((1, 1), (8, 2)):
                buf = torch.zeros(
                    (3, t_step, height, width),
                    dtype=torch.uint8,
                    device=assets.device,
                )
                _, st, _, _, _, _ = run_fast_single_gpu_step(
                    model,
                    assets,
                    frames_uint8=buf,
                    needed_frames=t_step,
                    valid_latents=v_lat,
                    current_states=st,
                )
                del buf
        finally:
            st.clear()
    if (
        bool(getattr(model, "encoder_tiling", False))
        and int(getattr(model, "tile_size", 0)) >= 256
    ):
        # Also pre-populate LUT for 736x1312 (shares B=28 conv plans with 736x1280)
        get_tile_geometry_plan(assets, model, 736, 1312)
