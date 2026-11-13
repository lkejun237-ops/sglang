# SPDX-License-Identifier: Apache-2.0
"""Session-owned incremental evaluation of the causal H3 CNN encoder.

Unlike whole-clip encoding, retained convolution history and stride phase let
one source frame produce the first stable latent before all 17 frames arrive.
Each clip advances by 1/8/8 pixels for the checkpoint's 1/2/2 latent groups;
work does not re-encode a growing prefix and state resets at clip boundaries.
This reduces input wait, not necessarily total compute: cached activations cost
memory, and smaller BF16 convolutions can differ numerically from whole clips.
"""

import logging
import os
from datetime import timedelta
from itertools import count

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .conv import BaseConv3d
from .fast_encoder import (
    encoder_numerical_optimizations_enabled,
    get_tile_geometry_plan,
    has_custom_encoder_hooks_or_patches,
    run_fast_single_gpu_step,
    stage_uint8_to_gpu,
    triton_available,
)
from .norm import FusedGroupNorm3D, TemporalIsolatedGroupNorm
from .progressive_layout import make_encoder_layout
from .spatial_padding import spatial_pad
from .vae_cnn import EncoderFCN3D

logger = logging.getLogger(__name__)
_SESSION_IDS = count()


class CausalEncoderCache:
    """Keep only convolution receptive fields and temporal stride phase.

    Updates are transactional: forward builds a new state dictionary. Cached
    tensors are never modified, so a failed layer leaves the session retryable.
    """

    def __init__(
        self, model, *, tile_mode=None, encode_mode=None, encode_dtype=None, group=None
    ):
        if not isinstance(model.encoder, EncoderFCN3D):
            raise ValueError("Progressive encoding requires EncoderFCN3D")
        for layer in model.encoder.modules():
            if isinstance(layer, torch.nn.GroupNorm) and not isinstance(
                layer, TemporalIsolatedGroupNorm
            ):
                raise ValueError(
                    "Progressive encoding requires time-isolated normalization"
                )
            if isinstance(layer, FusedGroupNorm3D) and (
                not layer.use_t_isolated_gn or layer.use_spatial_affine
            ):
                raise ValueError(
                    "Progressive encoding requires unconditional time-isolated normalization"
                )
            if isinstance(layer, BaseConv3d) and (
                not layer.causal or layer.pad_mode_t != "constant"
            ):
                raise ValueError("Progressive encoding requires causal zero padding")
        # Explicit finite tile batches are independent of legacy stack_tiling.
        value = os.environ.get("MINIMAX_H3_PROGRESSIVE_ENCODER_TILE_BATCH_SIZE")
        self.tile_batch_size = None if value is None else int(value)
        if self.tile_batch_size is not None and self.tile_batch_size < 1:
            raise ValueError("Progressive encoder tile batch size must be positive")
        self.requested_mode = tile_mode or os.environ.get(
            "MINIMAX_H3_PROGRESSIVE_ENCODER_TILING", "auto"
        )
        self.spatial_padding = os.environ.get(
            "MINIMAX_H3_PROGRESSIVE_SPATIAL_PADDING", "reflect2d"
        )
        if self.spatial_padding not in ("off", "reflect2d"):
            raise ValueError("Progressive spatial padding must be off or reflect2d")
        self._pad = spatial_pad if self.spatial_padding == "reflect2d" else F.pad
        if self.requested_mode not in ("auto", "replicated", "parallel"):
            raise ValueError(
                "Progressive encoder tiling must be auto, replicated or parallel"
            )
        from .klvae import (
            get_decode_parallel_group_coordinator,
            get_tile_parallel_state,
        )

        self.available_group = (
            group
            if group is not None
            else (
                get_decode_parallel_group_coordinator()
                if get_tile_parallel_state()[1] > 1
                else None
            )
        )
        self.group_name = "dedicated_encoder" if group is not None else "vae_decode"
        self.tile_mode = self.requested_mode
        # Prefer throughput; callers can explicitly disable performance-mode sharding.
        # Snapshot at construction, just like the tiling policy itself.
        self.performance_auto = (
            os.environ.get("MINIMAX_H3_PROGRESSIVE_ENCODER_PERFORMANCE_AUTO", "1")
            == "1"
        )
        self.encode_mode = encode_mode
        # Experimental execution choices are snapshotted per session, never read in kernels.
        self.numerical_optimizations = encoder_numerical_optimizations_enabled()
        self.fused_norm_input = self.numerical_optimizations and (
            os.environ.get("MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT", "1") == "1"
        )
        self.fused_conv_input = (
            os.environ.get("MINIMAX_H3_PROGRESSIVE_FUSED_CONV_INPUT", "1") == "1"
        )
        self.optimization_counts = {"norm_input": 0, "conv_input": 0, "fallback": 0}
        self.default_fallback_reason = None
        if self.requested_mode == "auto":
            # Retain the supported dtype, stack and group-size guards.
            if not model.encoder_tiling:
                self.default_fallback_reason = "non_tiled_encoder"
            elif self.available_group is None:
                self.default_fallback_reason = "single_rank_or_uninitialized_group"
            elif self.available_group.world_size not in (2, 4):
                self.default_fallback_reason = "unvalidated_group_size"
            elif encode_mode != "deterministic" and not (
                encode_mode == "performance" and self.performance_auto
            ):
                self.default_fallback_reason = "unvalidated_encode_mode"
            elif encode_dtype != "bfloat16":
                self.default_fallback_reason = "unvalidated_encode_dtype"
            elif model.stack_tiling and self.tile_batch_size is None:
                self.default_fallback_reason = "unvalidated_stack_tiling"
            self.tile_mode = (
                "replicated" if self.default_fallback_reason else "parallel"
            )
        self.group = self.available_group if self.tile_mode == "parallel" else None
        self.session_sequence = next(_SESSION_IDS) if self.group is not None else None
        self.last_execution = None
        self.layer_work = {}
        self._layout = None
        layouts = getattr(model, "_progressive_encoder_layouts", None)
        self._native_layout = layouts[0] if layouts else make_encoder_layout(model)
        use_channels_last = (
            os.environ.get("MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST", "1") == "1"
            and encode_mode == "performance"
            and encode_dtype == "bfloat16"
            and model.encoder.conv_in.weight.is_cuda
            and layouts is not None
        )
        self._multi_frame_layout = (
            layouts[1] if use_channels_last else self._native_layout
        )
        self.single_gpu_fast_requested = (
            os.environ.get("MINIMAX_H3_ENCODER_SINGLE_GPU_FAST", "1") == "1"
        )
        self._fast_assets = getattr(model, "_fast_single_gpu_encoder", None)
        self._use_fast_single_gpu = (
            self.single_gpu_fast_requested
            and self.numerical_optimizations
            and use_channels_last
            and self._fast_assets is not None
            and model.encoder.conv_in.weight.device.type == "cuda"
            and self.group is None
            and (self.available_group is None or self.available_group.world_size == 1)
            and self.tile_batch_size is None
            and not getattr(model, "stack_tiling", False)
            and self.spatial_padding == "reflect2d"
            and self.fused_norm_input
            and self.fused_conv_input
            and triton_available()
        )
        self.model = model
        self.states = {}
        self.temporal_positions = 0

    @property
    def channels_last(self):
        return self._multi_frame_layout.channels_last

    @property
    def fast_single_gpu(self):
        if not self._use_fast_single_gpu or torch.compiler.is_compiling():
            return False
        if (
            torch.are_deterministic_algorithms_enabled()
            or torch.backends.cudnn.deterministic
        ):
            return False
        if "forward" in self.__dict__ or type(self).forward is not _ORIG_CACHE_FORWARD:
            return False
        assets = self._fast_assets
        conv_in_w = self.model.encoder.conv_in.weight
        quant_w = self.model.quant_conv.weight
        if (
            conv_in_w.device != assets.device
            or conv_in_w.dtype != torch.bfloat16
            or conv_in_w.dtype != assets.dtype
            or quant_w.device != assets.device
            or quant_w.dtype != assets.dtype
        ):
            return False
        if has_custom_encoder_hooks_or_patches(
            self.model, assets=assets, allow_quant_conv_forward=True
        ):
            return False
        return True

    def can_encode_uint8_fast(self, height: int, width: int) -> bool:
        if "encode" in self.__dict__ or type(self).encode is not _ORIG_CACHE_ENCODE:
            return False
        if not self.fast_single_gpu:
            return False
        try:
            get_tile_geometry_plan(self._fast_assets, self.model, height, width)
        except ValueError:
            return False
        return True

    def conv(self, layer, x, states, key, layout, *, normalization=None):
        x = layout.input(x)
        history = layer.dilation[0] * (layer.kernel_size[0] - 1)
        previous, seen = self.states.get(key, (None, 0))
        offset = (-seen) % layer.stride[0]
        if offset >= x.shape[2]:
            raise ValueError("Incremental block produced no strided output")
        if normalization is not None or self.fused_conv_input:
            from .progressive_fusion import fused_conv_input

            norm = normalization[0] if normalization is not None else None
            assembled = fused_conv_input(
                x,
                previous,
                history,
                offset,
                layer.padding[1:],
                layer.pad_mode,
                # Singleton kernel dimensions make weight strides ambiguous.
                layout.input_memory_format == torch.channels_last_3d,
                norm=norm,
            )
            if assembled is not None:
                padded, next_history = assembled
                states[key] = (next_history, seen + x.shape[2])
                self.optimization_counts[
                    "norm_input" if norm is not None else "conv_input"
                ] += 1
                return layout.output(
                    F.conv3d(
                        padded,
                        layout.weight,
                        layer.bias,
                        layer.stride,
                        0,
                        layer.dilation,
                        layer.groups,
                    )
                )
            self.optimization_counts["fallback"] += 1
        if normalization is not None:
            norm, fused, normalize = normalization
            x = layout.input(normalize(norm, x, fused))
        if previous is None:
            shape = (*x.shape[:2], history, *x.shape[3:])
            previous = layout.zeros(x, shape)
        previous = layout.history(previous)
        joined = torch.cat((previous, x), 2) if history else x
        states[key] = (
            joined[:, :, -history:].clone() if history else None,
            seen + x.shape[2],
        )
        padded = self._pad(
            joined[:, :, offset:],
            (
                layer.padding[2],
                layer.padding[2],
                layer.padding[1],
                layer.padding[1],
                0,
                0,
            ),
            mode=layer.pad_mode,
        )
        out = F.conv3d(
            padded,
            layout.weight,
            layer.bias,
            layer.stride,
            0,
            layer.dilation,
            layer.groups,
        )
        return layout.output(out)

    def forward(self, x, key):
        # Singleton startup blocks retain their original layout and arithmetic.
        layout = self._multi_frame_layout if x.shape[2] > 1 else self._native_layout
        x = layout.input(x)
        states = {}
        encoder = self.model.encoder
        work = 0

        def conv(layer, value, name, norm=None, fused=False):
            nonlocal work
            positions = value.shape[0] * value.shape[2]
            work += positions
            self.layer_work[str(name)] = self.layer_work.get(str(name), 0) + positions
            normalization = None
            if norm is not None:
                if self.fused_norm_input:
                    normalization = (norm, fused, layout.norm)
                else:
                    value = layout.norm(norm, value, fused)
            return self.conv(
                layer,
                value,
                states,
                (key, name),
                layout.convs[layer],
                normalization=normalization,
            )

        h = conv(encoder.conv_in, x, "in")
        for i, down in enumerate(encoder.down):
            for j, block in enumerate(down.block):
                residual = h
                h = conv(block.conv1, h, (i, j, 1), block.norm1, block.use_fused_norm)
                h = conv(block.conv2, h, (i, j, 2), block.norm2, block.use_fused_norm)
                if block.in_channels != block.out_channels:
                    residual = conv(block.nin_shortcut, residual, (i, j, "shortcut"))
                h = h.add_(residual)
            if hasattr(down, "downsample"):
                layer = down.downsample
                if isinstance(layer, BaseConv3d):
                    h = conv(layer, h, (i, "down"))
                else:
                    if layer.space_stride == 2:
                        h = self._pad(h, (0, 1, 0, 1, 0, 0), mode=layer.pad_mode)
                    h = conv(layer.conv, h, (i, "down"))
        h = conv(encoder.conv_out, h, "out", encoder.norm_out, encoder.use_fused_norm)
        h = layout.output(h)
        return self.model.quant_conv(h), states, work

    def agree(self, value, *, error=None):
        """Bounded CPU rendezvous before entering a device collective.

        This exchanges control metadata, not shared cache tensors. ``value`` is
        a stage-tagged invariant; only the failure flag, not exception text,
        crosses ranks. See docs/docs/sglang-diffusion/zing/specs/PROGRESSIVE_TILE_PARALLEL_DESIGN.md for each stage.

        All participants must call sessions in the same order. A missing/cancelled
        worker fails the monitored barrier rather than stranding peers in NCCL.
        The owner must terminate the distributed request after any failure.
        """
        if self.group is None:
            if error is not None:
                raise error
            return
        dist.monitored_barrier(
            group=self.group.cpu_group,
            timeout=timedelta(seconds=30),
            wait_all_ranks=True,
        )
        values = [None] * self.group.world_size
        dist.all_gather_object(
            values, (error is not None, value), group=self.group.cpu_group
        )
        if any(failed for failed, _ in values):
            raise RuntimeError(
                "Progressive encoder failed on a participating rank"
            ) from error
        if any(item != values[0][1] for _, item in values):
            raise RuntimeError("Progressive encoder rank state mismatch")

    def _finalize_fast_execution(
        self,
        *,
        input_shape: list[int],
        result: torch.Tensor,
        pending: dict,
        work: int,
        layer_work: dict[str, int],
        opt_counts: dict[str, int],
        geom,
    ):
        for k, v in opt_counts.items():
            self.optimization_counts[k] = self.optimization_counts.get(k, 0) + v
        self.layer_work = dict(layer_work)
        local_ids = list(range(geom.num_tiles))
        reason = self.default_fallback_reason or (
            "disabled"
            if self.tile_mode == "replicated"
            else (
                "non_tiled_encoder"
                if not self.model.encoder_tiling
                else "single_rank_or_uninitialized_group"
            )
        )
        t_in = input_shape[2]
        execution = dict(
            channels_last_3d=True,
            fast_single_gpu=True,
            spatial_padding=self.spatial_padding,
            requested_mode=self.requested_mode,
            encode_mode=self.encode_mode,
            performance_auto=self.performance_auto,
            effective_mode="replicated",
            fallback_reason=reason,
            group=self.group_name if self.available_group is not None else None,
            group_ranks=(
                list(self.available_group.ranks)
                if self.available_group is not None
                else []
            ),
            rank=0,
            world_size=1,
            encoder_ranks=1,
            participating_group_ranks=1,
            replicated_copies=1,
            tile_count=geom.num_tiles,
            local_tile_ids=local_ids,
            backend_policy=dict(
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            ),
            stack_tiling=False,
            tile_batch_size=geom.num_tiles,
            tile_batches=[local_ids],
            local_tile_shapes=[
                [1, 3, t_in, geom.tile_h, geom.tile_w] for _ in local_ids
            ],
            input_shape=input_shape,
            output_shape=list(result.shape),
            dtype=str(result.dtype),
            temporal_positions=work,
            layer_temporal_positions=dict(layer_work),
            cache_bytes=sum(
                t.untyped_storage().nbytes()
                for t, _ in pending.values()
                if t is not None
            ),
            gather_calls=0,
            gather_input_bytes=0,
            gather_output_bytes=0,
            input_optimizations=dict(
                fused_norm_input=self.fused_norm_input,
                fused_conv_input=self.fused_conv_input,
                counts=dict(self.optimization_counts),
            ),
        )
        if self.last_execution is None:
            logger.info("Progressive encoder tile execution: %s", execution)
        self.last_execution = execution

    @torch.no_grad()
    def encode_uint8_frames(
        self,
        frames: torch.Tensor,
        *,
        needed_frames: int,
        valid_latents: int | None = None,
    ):
        height, width = int(frames.shape[2]), int(frames.shape[3])
        geom = get_tile_geometry_plan(self._fast_assets, self.model, height, width)
        local_ids = tuple(range(geom.num_tiles))
        layout = (
            (1, 3),
            (height, width),
            str(self._fast_assets.dtype),
            geom.geometry_tuple,
            local_ids,
            False,
            (local_ids,),
            False,
        )
        if self._layout is not None and layout != self._layout:
            raise ValueError("Progressive tile layout cannot change within a clip")
        gpu_frames = stage_uint8_to_gpu(self._fast_assets, frames)
        with torch.profiler.record_function("H3ProgressiveFastSingleGPU"):
            result, pending, work, layer_work, opt_counts, geom = (
                run_fast_single_gpu_step(
                    self.model,
                    self._fast_assets,
                    frames_uint8=gpu_frames,
                    needed_frames=needed_frames,
                    valid_latents=valid_latents,
                    current_states=self.states,
                )
            )
        self._layout = layout
        self._finalize_fast_execution(
            input_shape=[1, 3, int(needed_frames), height, width],
            result=result,
            pending=pending,
            work=work,
            layer_work=layer_work,
            opt_counts=opt_counts,
            geom=geom,
        )
        return result, pending, work

    @torch.no_grad()
    def encode(self, x):
        if (
            self.fast_single_gpu
            and x.ndim == 5
            and x.shape[0] == 1
            and x.shape[1] == 3
            and x.dtype == self._fast_assets.dtype
            and x.device == self._fast_assets.device
        ):
            height, width = int(x.shape[3]), int(x.shape[4])
            geom = None
            try:
                geom = get_tile_geometry_plan(
                    self._fast_assets, self.model, height, width
                )
            except ValueError:
                geom = None
            if geom is not None:
                local_ids = tuple(range(geom.num_tiles))
                layout = (
                    tuple(x.shape[:2]),
                    tuple(x.shape[3:]),
                    str(x.dtype),
                    geom.geometry_tuple,
                    local_ids,
                    False,
                    (local_ids,),
                    False,
                )
                if self._layout is not None and layout != self._layout:
                    raise ValueError(
                        "Progressive tile layout cannot change within a clip"
                    )
                with torch.profiler.record_function("H3ProgressiveFastSingleGPU"):
                    result, pending, work, layer_work, opt_counts, geom = (
                        run_fast_single_gpu_step(
                            self.model,
                            self._fast_assets,
                            pixels_bf16=x,
                            current_states=self.states,
                        )
                    )
                self._layout = layout
                self._finalize_fast_execution(
                    input_shape=list(x.shape),
                    result=result,
                    pending=pending,
                    work=work,
                    layer_work=layer_work,
                    opt_counts=opt_counts,
                    geom=geom,
                )
                return result, pending, work

        model = self.model
        pending = {}
        work = 0
        self.layer_work = {}
        rank = (
            self.available_group.rank_in_group
            if self.available_group is not None
            else 0
        )
        size = (
            self.available_group.world_size if self.available_group is not None else 1
        )
        reason = None
        if self.tile_mode == "replicated":
            reason = self.default_fallback_reason or "disabled"
        elif not model.encoder_tiling:
            reason = "non_tiled_encoder"
        elif size == 1:
            reason = "single_rank_or_uninitialized_group"
        elif (
            self.requested_mode == "auto"
            and model.stack_tiling
            and self.tile_batch_size is None
        ):
            reason = "unvalidated_stack_tiling"
        ys = xs = [0]
        yo = xo = []
        tiles = [x]
        geometry = None
        error = None
        try:
            if model.encoder_tiling:
                ys, hs, yo = model.split_tiles(x.shape[-2], False)
                xs, ws, xo = model.split_tiles(x.shape[-1], False)
                tiles = [
                    x[..., y : y + h, z : z + w]
                    for y, h in zip(ys, hs)
                    for z, w in zip(xs, ws)
                ]
                if reason is None and len(tiles) < size:
                    reason = "fewer_tiles_than_ranks"
            geometry = (
                tuple(ys),
                tuple(xs),
                tuple(yo),
                tuple(xo),
                tuple(tuple(tile.shape[-2:]) for tile in tiles),
                getattr(model, "vae_ratio", None),
            )
            # Fixed-size all_gather requires matching spatial output shapes.
            # Replicate heterogeneous edges rather than padding moments or changing
            # the existing tile geometry/assembly semantics.
            if reason is None and len({tuple(t.shape[-2:]) for t in tiles}) > 1:
                reason = "heterogeneous_tile_shapes"
            parallel = reason is None
            local_ids = (
                model._local_tile_indices(len(tiles), rank, size)
                if parallel
                else list(range(len(tiles)))
            )
            stack = (
                model.encoder_tiling
                and model.stack_tiling
                and self.tile_batch_size is None
            )
            batches = []
            if self.tile_batch_size is not None:
                shape_groups = {}
                for i in local_ids:
                    shape_groups.setdefault(tuple(tiles[i].shape), []).append(i)
                for ids in shape_groups.values():
                    batches.extend(
                        tuple(ids[start : start + self.tile_batch_size])
                        for start in range(0, len(ids), self.tile_batch_size)
                    )
            layout = (
                tuple(x.shape[:2]),
                tuple(x.shape[3:]),
                str(x.dtype),
                geometry,
                tuple(local_ids),
                stack,
                tuple(batches),
                parallel,
            )
            if self._layout is not None and layout != self._layout:
                raise ValueError("Progressive tile layout cannot change within a clip")
        except Exception as exc:
            error = exc
        self.agree(
            (
                tuple(x.shape),
                str(x.dtype),
                reason,
                geometry,
                bool(model.encoder_tiling),
                getattr(model, "stack_tiling", False),
                self.tile_batch_size,
            ),
            error=error,
        )
        error = None
        try:
            if self.tile_batch_size is not None:
                by_id = {}
                for ids in batches:
                    value = (
                        tiles[ids[0]]
                        if len(ids) == 1
                        else torch.cat([tiles[i] for i in ids], 0)
                    )
                    with torch.profiler.record_function(
                        f"H3ProgressiveTileBatch/{ids}"
                    ):
                        output, state, count = self.forward(value, ids)
                    for i, item in zip(ids, output.split(x.shape[0], dim=0)):
                        by_id[i] = item
                    pending.update(state)
                    work += count
                outputs = [by_id[i] for i in local_ids]
            elif stack:
                # The tuple fixes the batch slot -> global tile mapping for history.
                with torch.profiler.record_function(f"H3ProgressiveTiles/{local_ids}"):
                    output, pending, work = self.forward(
                        torch.cat([tiles[i] for i in local_ids], 0), tuple(local_ids)
                    )
                outputs = list(
                    output.unflatten(0, (len(local_ids), x.shape[0])).unbind(0)
                )
            else:
                outputs = []
                for i in local_ids:
                    with torch.profiler.record_function(f"H3ProgressiveTile/{i}"):
                        output, state, count = self.forward(tiles[i], i)
                    outputs.append(output)
                    pending.update(state)
                    work += count
            if parallel:
                max_tasks = (len(tiles) + size - 1) // size
                packed = outputs[0].new_zeros((max_tasks, *outputs[0].shape))
                torch.stack(outputs, out=packed[: len(outputs)])
                gathered = [torch.empty_like(packed) for _ in range(size)]
        except Exception as exc:
            error = exc
        # Include allocations in the failure rendezvous, before any rank gathers.
        self.agree(
            (
                ("computed", tuple(outputs[0].shape), str(outputs[0].dtype))
                if error is None
                else None
            ),
            error=error,
        )
        payload = 0
        if parallel:
            with torch.profiler.record_function("H3ProgressiveTileGather"):
                dist.all_gather(gathered, packed, group=self.group.device_group)
            payload = packed.numel() * packed.element_size()
            outputs = [gathered[i % size][i // size] for i in range(len(tiles))]
        error = None
        try:
            if model.encoder_tiling:
                rows = [
                    list(outputs[i : i + len(xs)])
                    for i in range(0, len(outputs), len(xs))
                ]
                result = model._assemble_tiles(
                    rows,
                    [v // model.vae_ratio for v in yo],
                    [v // model.vae_ratio for v in xo],
                )
            else:
                result = outputs[0]
            if self.channels_last:
                # Restore the public moments layout once, after low-resolution assembly.
                result = result.contiguous()
        except Exception as exc:
            error = exc
        self.agree("assembled", error=error)
        self._layout = layout
        execution = dict(
            channels_last_3d=self.channels_last,
            fast_single_gpu=False,
            spatial_padding=self.spatial_padding,
            requested_mode=self.requested_mode,
            encode_mode=self.encode_mode,
            performance_auto=self.performance_auto,
            effective_mode="parallel" if parallel else "replicated",
            fallback_reason=reason,
            group=self.group_name if self.available_group is not None else None,
            group_ranks=(
                list(self.available_group.ranks)
                if self.available_group is not None
                else []
            ),
            rank=rank,
            world_size=size,
            encoder_ranks=size if parallel else 1,
            participating_group_ranks=size,
            replicated_copies=1 if parallel else size,
            tile_count=len(tiles),
            local_tile_ids=local_ids,
            backend_policy=dict(
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            ),
            stack_tiling=bool(stack),
            tile_batch_size=self.tile_batch_size,
            tile_batches=[list(ids) for ids in batches],
            local_tile_shapes=[list(tiles[i].shape) for i in local_ids],
            input_shape=list(x.shape),
            output_shape=list(result.shape),
            dtype=str(result.dtype),
            temporal_positions=work,
            layer_temporal_positions=dict(self.layer_work),
            cache_bytes=sum(
                t.untyped_storage().nbytes()
                for t, _ in pending.values()
                if t is not None
            ),
            gather_calls=int(parallel),
            gather_input_bytes=payload,
            gather_output_bytes=payload * size,
            input_optimizations=dict(
                fused_norm_input=self.fused_norm_input,
                fused_conv_input=self.fused_conv_input,
                counts=dict(self.optimization_counts),
            ),
        )
        if self.last_execution is None:
            logger.info("Progressive encoder tile execution: %s", execution)
        self.last_execution = execution
        return result, pending, work

    def commit(self, states, work):
        self.states = states
        self.temporal_positions += work

    def clear(self):
        self.states = {}
        self._layout = None


_ORIG_CACHE_FORWARD = CausalEncoderCache.forward
_ORIG_CACHE_ENCODE = CausalEncoderCache.encode
