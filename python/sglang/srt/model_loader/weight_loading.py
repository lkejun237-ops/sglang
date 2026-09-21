# SPDX-License-Identifier: Apache-2.0
"""Tensor-only weight loading shared by SRT and diffusion models."""

import torch


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Copy a checkpoint tensor, accepting differently shaped scalar tensors."""
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
        return
    assert param.size() == loaded_weight.size(), (
        f"Attempted to load weight ({loaded_weight.size()}) "
        f"into parameter ({param.size()})"
    )
    param.data.copy_(loaded_weight)
