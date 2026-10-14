# SPDX-License-Identifier: Apache-2.0
"""Attention backends and the streaming block of the Lynn decoder.

Ported from the Zing reference (seedleap/zing @ 8eda9c22c6, src/h3/vae_decoder/models/attention.py).
"""

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn


class NativeAttention:
    def __call__(self, q, k, v):
        return dispatch_attention_fn(q, k, v, backend="native")


class Flash4Attention:
    def __init__(self):
        from flash_attn.cute import flash_attn_func

        self.execute = flash_attn_func

    def __call__(self, q, k, v):
        # Time causality comes from which history K/V the caller passes, never from a mask.
        output = self.execute(q, k, v, causal=False, return_lse=False)
        return output[0] if isinstance(output, tuple) else output


ATTENTION_BACKENDS = {"native": NativeAttention, "flash4": Flash4Attention}


def rotate(x, rotary):
    cos, sin = (part.to(x.dtype) for part in rotary)
    width = cos.shape[-1]
    # Half-split rotation over the first `width` channels only; must not become interleaved pairs.
    a, b = x[..., :width].chunk(2, dim=-1)
    return torch.cat(
        [x[..., :width] * cos + torch.cat([-b, a], dim=-1) * sin, x[..., width:]],
        dim=-1,
    )


def streaming_block(
    block,
    hidden,
    history_k,
    history_v,
    *,
    query_rope,
    key_rope,
    spatial_tokens,
    attention,
):
    """One transformer block over the current latent; returns the K/V to cache (pre-RoPE)."""
    attn = block.attn
    normalized = block.norm1(hidden.float()).to(attn.to_q.weight.dtype)
    q = attn.to_q(normalized).unflatten(-1, (attn.heads, -1))
    k = attn.to_k(normalized).unflatten(-1, (attn.heads, -1))
    v = attn.to_v(normalized).unflatten(-1, (attn.heads, -1))
    q = attn.norm_q(q.float()).to(q.dtype)
    k = attn.norm_k(k.float()).to(k.dtype)
    # Cached K/V exclude register and zero-cls tokens.
    current_k, current_v = k[:, :spatial_tokens], v[:, :spatial_tokens]
    all_k, all_v = torch.cat([history_k, k], dim=1), torch.cat([history_v, v], dim=1)
    attended = attention(rotate(q, query_rope), rotate(all_k, key_rope), all_v)
    hidden = hidden + attn.to_out[0](attended.flatten(2)) * block.scale1
    normalized = block.norm2(hidden.float()).to(attn.to_q.weight.dtype)
    return hidden + block.ff(normalized) * block.scale2, current_k, current_v
