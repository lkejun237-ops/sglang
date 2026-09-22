# SPDX-License-Identifier: Apache-2.0
"""Session-owned incremental evaluation of the causal H3 CNN encoder.

Unlike whole-clip encoding, retained convolution history and stride phase let
one source frame produce the first stable latent before all 17 frames arrive.
Each clip advances by 1/8/8 pixels for the checkpoint's 1/2/2 latent groups;
work does not re-encode a growing prefix and state resets at clip boundaries.
This reduces input wait, not necessarily total compute: cached activations cost
memory, and smaller BF16 convolutions can differ numerically from whole clips.
"""

import torch
import torch.nn.functional as F

from .conv import BaseConv3d
from .norm import FusedGroupNorm3D, TemporalIsolatedGroupNorm
from .vae_cnn import EncoderFCN3D, norm_silu


class CausalEncoderCache:
    """Keep only convolution receptive fields and temporal stride phase.

    Updates are transactional: forward builds a new state dictionary. Cached
    tensors are never modified, so a failed layer leaves the session retryable.
    """

    def __init__(self, model):
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
        padded = F.pad(
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
            work += value.shape[0] * value.shape[2]
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
                        h = F.pad(h, (0, 1, 0, 1, 0, 0), mode=layer.pad_mode)
                    h = conv(layer.conv, h, (i, "down"))
        h = (
            encoder.norm_out(h, None)
            if encoder.use_fused_norm
            else norm_silu(h, encoder.norm_out)
        )
        h = conv(encoder.conv_out, h, "out")
        return self.model.quant_conv(h), states, work

    @torch.no_grad()
    def encode(self, x):
        model = self.model
        pending = {}
        work = 0
        if not model.encoder_tiling:
            result, pending, work = self.forward(x, 0)
        else:
            # Encoder SP remains replicated. Preserve native tile batch order.
            ys, hs, yo = model.split_tiles(x.shape[-2], False)
            xs, ws, xo = model.split_tiles(x.shape[-1], False)
            tiles = [
                x[..., y : y + h, z : z + w]
                for y, h in zip(ys, hs)
                for z, w in zip(xs, ws)
            ]
            if model.stack_tiling:
                output, pending, work = self.forward(torch.cat(tiles, 0), 0)
                outputs = output.unflatten(0, (len(tiles), x.shape[0])).unbind(0)
            else:
                outputs = []
                for i, tile in enumerate(tiles):
                    output, state, count = self.forward(tile, i)
                    outputs.append(output)
                    pending.update(state)
                    work += count
            rows = [
                list(outputs[i : i + len(xs)]) for i in range(0, len(outputs), len(xs))
            ]
            result = model._assemble_tiles(
                rows,
                [v // model.vae_ratio for v in yo],
                [v // model.vae_ratio for v in xo],
            )
        return result, pending, work

    def commit(self, states, work):
        self.states = states
        self.temporal_positions += work

    def clear(self):
        self.states = {}
