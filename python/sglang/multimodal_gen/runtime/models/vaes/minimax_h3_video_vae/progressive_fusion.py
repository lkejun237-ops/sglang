# SPDX-License-Identifier: Apache-2.0
"""Eligibility for optional progressive input kernels; unsupported inputs stay eager."""

from importlib.util import find_spec

import torch


def fused_conv_input(
    x, previous, history, offset, padding, mode, channels_last, norm=None
):
    """Return independent (padded input, next history), or None before any mutation.

    x/previous are NCTHW views; output follows the selected convolution layout.
    Previous values are already normalized and are never normalized again.
    """
    if (
        not x.is_cuda
        or x.ndim != 5
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or min(x.shape) < 1
        or min(x.stride()) < 1
        or history < 0
        or not 0 <= offset < x.shape[2]
        or mode not in ("reflect", "constant")
        or (torch.is_grad_enabled() and x.requires_grad)
        or torch.compiler.is_compiling()
        or find_spec("triton") is None
    ):
        return None
    ph, pw = padding
    if (
        ph < 0
        or pw < 0
        or (mode == "reflect" and (ph >= x.shape[-2] or pw >= x.shape[-1]))
    ):
        return None
    if previous is not None and (
        previous.shape != (*x.shape[:2], history, *x.shape[3:])
        or previous.device != x.device
        or previous.dtype != x.dtype
        or min(previous.stride()) < 1
    ):
        return None
    if norm is not None:
        from .norm import FusedGroupNorm3D, TemporalIsolatedGroupNorm

        if type(norm) not in (TemporalIsolatedGroupNorm, FusedGroupNorm3D):
            return None
        if norm._forward_hooks or norm._forward_pre_hooks or "forward" in norm.__dict__:
            return None
        if isinstance(norm, FusedGroupNorm3D) and (
            not norm.use_t_isolated_gn
            or norm.use_spatial_affine
            or norm.activation != "silu"
        ):
            return None
        if (
            norm.num_channels != x.shape[1]
            or x.shape[1] % norm.num_groups
            or (norm.weight is None) != (norm.bias is None)
        ):
            return None
        for tensor in (norm.weight, norm.bias):
            if tensor is not None and (
                tensor.device != x.device
                or tensor.dtype not in (x.dtype, torch.float32)
                or not tensor.is_contiguous()
                or (torch.is_grad_enabled() and tensor.requires_grad)
            ):
                return None
    from .progressive_kernels import assemble

    return assemble(x, previous, history, offset, padding, mode, channels_last, norm)
