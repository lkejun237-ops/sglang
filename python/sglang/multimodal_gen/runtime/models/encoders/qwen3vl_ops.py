# SPDX-License-Identifier: Apache-2.0
"""Opt-in Qwen3-VL adapters for existing kernels, without global dispatch changes."""

import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.attention.selector import (
    record_component_attn_backend,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def _require_cuda_half(x: torch.Tensor, operation: str) -> None:
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"{operation} requires CUDA float16/bfloat16 inputs")


class Qwen3VLHFRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        if not (
            (32 <= hidden_size < 512 and hidden_size % 32 == 0)
            or (hidden_size >= 512 and hidden_size % 512 == 0)
        ):
            raise ValueError(f"HF RMSNorm does not support width {hidden_size}")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from sglang.kernels.ops.layernorm.rmsnorm_hf import rmsnorm_hf

        _require_cuda_half(x, "HF RMSNorm")
        return rmsnorm_hf(
            x.contiguous().reshape(-1, x.shape[-1]), self.weight, self.variance_epsilon
        ).view(x.shape)


def rounded_swiglu(x: torch.Tensor) -> torch.Tensor:
    from sglang.kernels.ops.activation.activation import (
        silu_and_mul_with_activation_rounding,
    )

    _require_cuda_half(x, "Rounded SwiGLU")
    shape = x.shape[:-1]
    return silu_and_mul_with_activation_rounding(x.reshape(-1, x.shape[-1])).view(
        *shape, x.shape[-1] // 2
    )


class Qwen3VLFA4Attention(nn.Module):
    """Pinned FA4 for dense language GQA and packed, non-causal vision."""

    def __init__(self, *, softmax_scale: float, causal: bool) -> None:
        super().__init__()
        from sglang.kernels.ops.attention.flash_attention_v4 import (
            flash_attn_varlen_func,
            is_flash_attention_v4_available,
        )

        if not is_flash_attention_v4_available():
            raise RuntimeError("H3 Qwen FA4 was enabled but its kernel is unavailable")
        self.kernel = flash_attn_varlen_func
        self.softmax_scale = softmax_scale
        self.causal = causal
        # FA4 belongs to the FA backend family. No fallback reason: an explicit
        # component request for another backend must reject this selection.
        record_component_attn_backend(AttentionBackendEnum.FA)

    def forward(self, query, key, value):
        _require_cuda_half(query, "Qwen FA4")
        return self.kernel(
            query,
            key,
            value,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            num_splits=1,
            pack_gqa=None,
        )

    def forward_varlen(
        self, query, key, value, *, cu_seqlens, cu_seqlens_host, max_seqlen
    ):
        _require_cuda_half(query, "Qwen FA4")
        return self.kernel(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            num_splits=1,
            pack_gqa=None,
        )
