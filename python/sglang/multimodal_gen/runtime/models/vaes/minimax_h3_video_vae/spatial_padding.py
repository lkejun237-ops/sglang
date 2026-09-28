# SPDX-License-Identifier: Apache-2.0
"""Spatial-only padding for the progressive encoder (native Zing #293)."""

import torch.nn.functional as F


def spatial_pad(x, padding, mode):
    """Use independent HW planes only when flattening BCT is a view.

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
    return F.pad(x, padding, mode=mode)
