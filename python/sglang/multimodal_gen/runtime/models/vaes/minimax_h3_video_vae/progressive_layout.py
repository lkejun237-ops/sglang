# SPDX-License-Identifier: Apache-2.0
"""Prepared layout strategies for the progressive source encoder."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from .norm import progressive_norm_silu
from .vae_cnn import norm_silu


def _identity(x):
    return x


def _contiguous(x):
    return x.contiguous()


def _channels_last(x):
    return x.contiguous(memory_format=torch.channels_last_3d)


def _native_zeros(x, shape):
    return x.new_zeros(shape)


def _channels_last_zeros(x, shape):
    return torch.empty(
        shape, dtype=x.dtype, device=x.device, memory_format=torch.channels_last_3d
    ).zero_()


def _native_norm(layer, x, fused):
    return layer(x, None) if fused else norm_silu(x, layer)


def _channels_last_norm(layer, x, fused):
    return progressive_norm_silu(x, layer)


@dataclass(frozen=True)
class ConvLayout:
    weight: torch.Tensor
    input: Callable
    history: Callable
    zeros: Callable
    output: Callable


@dataclass(frozen=True)
class EncoderLayout:
    channels_last: bool
    input: Callable
    norm: Callable
    output: Callable
    convs: Mapping[torch.nn.Conv3d, ConvLayout]


def make_encoder_layout(model, *, weights=None):
    """Bind weights and conversions once; no layout decisions in conv().

    With serving copies, retain NCDHW for 1x1 convolutions and the third down
    stage: the trained-weight audit found BF16 rounding changes in its five
    convolutions. Other layers consume the prepared channels-last weights.
    Without serving copies, preserve the original path, including startup.
    """
    channels_last = weights is not None
    legacy = (
        set(model.encoder.down[2].modules())
        if channels_last and len(model.encoder.down) > 2
        else set()
    )
    convs = {}
    for layer in model.encoder.modules():
        if not isinstance(layer, torch.nn.Conv3d):
            continue
        if not channels_last:
            plan = ConvLayout(
                layer.weight, _identity, _contiguous, _native_zeros, _identity
            )
        elif layer.kernel_size == (1, 1, 1) or layer in legacy:
            plan = ConvLayout(
                layer.weight, _contiguous, _contiguous, _native_zeros, _channels_last
            )
        else:
            plan = ConvLayout(
                weights[layer],
                _identity,
                _channels_last,
                _channels_last_zeros,
                _identity,
            )
        convs[layer] = plan
    return EncoderLayout(
        channels_last,
        _channels_last if channels_last else _identity,
        _channels_last_norm if channels_last else _native_norm,
        _contiguous if channels_last else _identity,
        convs,
    )
