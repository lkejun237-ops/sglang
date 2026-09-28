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
from .norm import FusedGroupNorm3D, TemporalIsolatedGroupNorm
from .spatial_padding import spatial_pad
from .vae_cnn import EncoderFCN3D, norm_silu

logger = logging.getLogger(__name__)
_SESSION_IDS = count()


class CausalEncoderCache:
    """Keep only convolution receptive fields and temporal stride phase.

    Updates are transactional: forward builds a new state dictionary. Cached
    tensors are never modified, so a failed layer leaves the session retryable.
    """

    def __init__(self, model, *, tile_mode=None, encode_mode=None, encode_dtype=None):
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
            get_decode_parallel_group_coordinator()
            if get_tile_parallel_state()[1] > 1
            else None
        )
        self.tile_mode = self.requested_mode
        self.default_fallback_reason = None
        if self.requested_mode == "auto":
            # Promote only the numerical/stack/group policy validated in #177.
            if not model.encoder_tiling:
                self.default_fallback_reason = "non_tiled_encoder"
            elif self.available_group is None:
                self.default_fallback_reason = "single_rank_or_uninitialized_group"
            elif self.available_group.world_size not in (2, 4):
                self.default_fallback_reason = "unvalidated_group_size"
            elif encode_mode != "deterministic":
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
        self.model = model
        self.states = {}
        self.temporal_positions = 0

    def conv(self, layer, x, states, key):
        history = layer.dilation[0] * (layer.kernel_size[0] - 1)
        previous, seen = self.states.get(key, (None, 0))
        if previous is None:
            previous = x.new_zeros((*x.shape[:2], history, *x.shape[3:]))
        joined = torch.cat((previous, x), 2) if history else x
        states[key] = (
            joined[:, :, -history:].clone() if history else None,
            seen + x.shape[2],
        )
        offset = (-seen) % layer.stride[0]
        if offset >= x.shape[2]:
            raise ValueError("Incremental block produced no strided output")
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
        return F.conv3d(
            padded,
            layer.weight,
            layer.bias,
            layer.stride,
            0,
            layer.dilation,
            layer.groups,
        )

    def forward(self, x, key):
        states = {}
        encoder = self.model.encoder
        work = 0

        def conv(layer, value, name):
            nonlocal work
            positions = value.shape[0] * value.shape[2]
            work += positions
            self.layer_work[str(name)] = self.layer_work.get(str(name), 0) + positions
            return self.conv(layer, value, states, (key, name))

        h = conv(encoder.conv_in, x, "in")
        for i, down in enumerate(encoder.down):
            for j, block in enumerate(down.block):
                residual = h
                h = (
                    block.norm1(h, None)
                    if block.use_fused_norm
                    else norm_silu(h, block.norm1)
                )
                h = conv(block.conv1, h, (i, j, 1))
                h = (
                    block.norm2(h, None)
                    if block.use_fused_norm
                    else norm_silu(h, block.norm2)
                )
                h = conv(block.conv2, h, (i, j, 2))
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
        h = (
            encoder.norm_out(h, None)
            if encoder.use_fused_norm
            else norm_silu(h, encoder.norm_out)
        )
        h = conv(encoder.conv_out, h, "out")
        return self.model.quant_conv(h), states, work

    def agree(self, value, *, error=None):
        """Bounded CPU rendezvous before entering a device collective.

        This exchanges control metadata, not shared cache tensors. ``value`` is
        a stage-tagged invariant; only the failure flag, not exception text,
        crosses ranks. See PROGRESSIVE_TILE_PARALLEL_DESIGN.md for each stage.

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

    @torch.no_grad()
    def encode(self, x):
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
            ("computed", tuple(outputs[0].shape), str(outputs[0].dtype))
            if error is None
            else None,
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
        except Exception as exc:
            error = exc
        self.agree("assembled", error=error)
        self._layout = layout
        execution = dict(
            spatial_padding=self.spatial_padding,
            requested_mode=self.requested_mode,
            effective_mode="parallel" if parallel else "replicated",
            fallback_reason=reason,
            group="vae_decode" if self.available_group is not None else None,
            group_ranks=list(self.available_group.ranks)
            if self.available_group is not None
            else [],
            rank=rank,
            world_size=size,
            encoder_ranks=size if parallel else 1,
            participating_group_ranks=size,
            replicated_copies=1 if parallel else size,
            tile_count=len(tiles),
            local_tile_ids=local_ids,
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
