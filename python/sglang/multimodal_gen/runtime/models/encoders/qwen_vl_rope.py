# SPDX-License-Identifier: Apache-2.0
"""Shared SRT rotary embedding adapter for Qwen-VL text encoders."""

from typing import Any

import torch
from torch import nn

from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.utils.hf_transformers.common import get_rope_config


class QwenVLFusedMRoPE(nn.Module):
    """Encoder-owned cache shared by its layers, independent of get_rope state."""

    def __init__(self, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        # Initially share the immutable source tensor. Conversion replaces only
        # this module's buffer; it never writes to the native FP32 cache.
        self.register_buffer(
            "cos_sin_cache", rotary_emb.cos_sin_cache, persistent=False
        )

    def forward(self, rotary_emb, positions, query, key):
        from sglang.kernels.ops.attention.rotary_triton import triton_mrope_fused

        if (
            self.cos_sin_cache.device != query.device
            or self.cos_sin_cache.dtype != query.dtype
        ):
            self.cos_sin_cache = rotary_emb.cos_sin_cache.to(
                device=query.device, dtype=query.dtype
            )
        # Same kernel and arguments as MRotaryEmbedding.forward_triton, with an
        # explicitly owned cache instead of temporarily mutating that module.
        triton_mrope_fused(
            query,
            key,
            self.cos_sin_cache,
            positions,
            rotary_emb.mrope_section,
            rotary_emb.head_size,
            rotary_emb.rotary_dim,
            rotary_emb.mrope_interleaved,
            rotary_emb.mrope_interleaved_glm,
            rotary_emb.is_neox_style,
            rotary_emb._legacy_axis_map,
        )
        return query, key


def build_qwen_vl_text_rope(
    config: Any, *, mrope_interleaved: bool = False
) -> RotaryEmbedding:
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    rope_theta, rope_scaling = get_rope_config(config)
    rope_scaling = dict(rope_scaling or {})
    rope_scaling["mrope_interleaved"] = mrope_interleaved
    return get_rope(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position=config.max_position_embeddings,
        base=rope_theta,
        is_neox_style=True,
        rope_scaling=rope_scaling,
    )


def apply_qwen_vl_text_rope(
    rotary_emb: RotaryEmbedding,
    position_ids: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    fused_rope: QwenVLFusedMRoPE | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply three-axis MRoPE to batched attention tensors."""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError(
            "Qwen-VL query and key must have shape [batch, heads, sequence, head_dim]"
        )
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError(
            "Qwen-VL text position_ids must have shape [3, batch, sequence]"
        )
    batch_size, num_query_heads, sequence_length, head_dim = query.shape
    key_batch_size, num_key_value_heads, key_sequence_length, key_head_dim = key.shape
    if (key_batch_size, key_sequence_length, key_head_dim) != (
        batch_size,
        sequence_length,
        head_dim,
    ):
        raise ValueError("Qwen-VL query and key shapes are incompatible")
    if tuple(position_ids.shape[1:]) != (batch_size, sequence_length):
        raise ValueError("Qwen-VL position_ids do not match the attention input")

    query = query.transpose(1, 2).reshape(-1, num_query_heads * head_dim)
    key = key.transpose(1, 2).reshape(-1, num_key_value_heads * head_dim)
    if fused_rope is not None:
        if not query.is_cuda:
            raise ValueError("Fused Qwen MRoPE requires CUDA inputs")
        # The fused kernel writes in place. Preserve the caller's Q/K even if
        # it raises; cache conversion is owned by the encoder's fused module.
        query, key = query.clone(), key.clone()
        query, key = fused_rope(rotary_emb, position_ids.reshape(3, -1), query, key)
    else:
        # Preserve HF's bf16 arithmetic order by default.
        query, key = rotary_emb.forward_native(position_ids.reshape(3, -1), query, key)
    query = query.view(batch_size, sequence_length, num_query_heads, head_dim)
    key = key.view(batch_size, sequence_length, num_key_value_heads, head_dim)
    return query.transpose(1, 2), key.transpose(1, 2)
