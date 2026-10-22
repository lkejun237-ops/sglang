# SPDX-License-Identifier: Apache-2.0
"""Spatial-only padding for the progressive encoder (native Zing #293)."""

import torch
import torch.nn.functional as F


def spatial_pad(x, padding, mode):
    """Reflect spatial axes without transposing dense activation storage.

    NCDHW uses independent HW planes; NDHWC uses an HWC reflection view.
    Keep PyTorch's original path (and validation) for other layouts, temporal
    padding, cropping and invalid reflection widths. Never materialize a
    contiguous copy here: temporal history and stride slices belong to the
    caller, and copying them can erase the benefit of this substitution.
    """
    if (
        mode == "reflect"
        and x.ndim == 5
        and x.is_contiguous()
        and len(padding) == 6
        and padding[4:] == (0, 0)
        and all(0 <= p < x.shape[-1] for p in padding[:2])
        and all(0 <= p < x.shape[-2] for p in padding[2:4])
        and all(s > 0 for s in x.shape[:3])
    ):
        return F.pad(x.flatten(0, 2), padding[:4], mode=mode).unflatten(0, x.shape[:3])
    if (
        mode == "reflect"
        and x.ndim == 5
        and x.stride(1) == 1
        and x.stride(4) == x.shape[1]
        and x.stride(3) == x.shape[4] * x.shape[1]
        and x.stride(2) == x.shape[3] * x.shape[4] * x.shape[1]
        and len(padding) == 6
        and padding[4:] == (0, 0)
        and all(0 <= p < x.shape[-1] for p in padding[:2])
        and all(0 <= p < x.shape[-2] for p in padding[2:4])
        and all(s > 0 for s in x.shape[:3])
    ):
        if (
            x.is_cuda
            and torch.version.cuda is not None
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and (not torch.is_grad_enabled() or not x.requires_grad)
        ):
            from .spatial_padding_kernels import reflect_ndhwc

            return reflect_ndhwc(x, padding)
        # Native/autograd fallback: C is an unpadded spatial axis. Temporal
        # slices may have a gap between batch items.
        return F.pad(x.permute(0, 2, 3, 4, 1), (0, 0, *padding[:4]), mode=mode).permute(
            0, 4, 1, 2, 3
        )
    return F.pad(x, padding, mode=mode)
