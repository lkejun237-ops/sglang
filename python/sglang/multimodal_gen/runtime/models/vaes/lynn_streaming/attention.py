# SPDX-License-Identifier: Apache-2.0
"""Attention backends and the streaming block of the Lynn decoder.

Ported from the Zing reference (seedleap/zing @ 8eda9c22c6, src/h3/vae_decoder/models/attention.py).
"""

from functools import cache

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


@cache
def _supports_preparation_device(device):
    properties = torch.cuda.get_device_properties(device)
    # Elementwise preparation needs no architecture-specific MMA or SM count.
    # Keep the supported families explicit; other devices use the Torch path.
    return (properties.major, properties.minor) in ((9, 0), (10, 3))


def _can_fuse_preparation(
    q, k, v, history_k, history_v, query_rope, key_rope, spatial_tokens, cache_tokens
):
    # Keep the exact shape, layout and arithmetic contract on supported devices.
    if (
        not q.is_cuda
        or q.dtype != torch.bfloat16
        or q.shape[0] not in (6, 12, 14, 28, 30)
        or q.shape[1:] != (261, 32, 64)
        or k.shape != q.shape
        or v.shape != q.shape
        or spatial_tokens != 256
        or cache_tokens != 512
        or history_k.ndim != 4
        or history_k.shape[0] != q.shape[0]
        or history_k.shape[2:] != (32, 64)
        or history_k.shape[1] not in (0, 256, 512)
        or history_v.shape != history_k.shape
    ):
        return False
    tensors = (q, k, v, history_k, history_v)
    if not all(
        t.dtype == q.dtype and t.device == q.device and t.is_contiguous()
        for t in tensors
    ):
        return False
    for rotary, length in (
        (query_rope, q.shape[1]),
        (key_rope, history_k.shape[1] + k.shape[1]),
    ):
        cos, sin = rotary
        if (
            cos.shape != (q.shape[0], length, 1, 48)
            or sin.shape != cos.shape
            or cos.stride() != sin.stride()
            or cos.dtype not in (torch.float32, torch.bfloat16)
            or sin.dtype != cos.dtype
            or cos.device != q.device
            or sin.device != q.device
        ):
            return False
    return _supports_preparation_device(q.device)


def streaming_block(
    block,
    hidden,
    history_k,
    history_v,
    *,
    query_rope,
    key_rope,
    spatial_tokens,
    cache_tokens,
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
    fused = _can_fuse_preparation(
        q,
        k,
        v,
        history_k,
        history_v,
        query_rope,
        key_rope,
        spatial_tokens,
        cache_tokens,
    )
    if fused:
        from .fused_preparation import prepare

        q, rotated_k, all_v, cached_k, cached_v = prepare(
            q,
            k,
            v,
            history_k,
            history_v,
            query_rope,
            key_rope,
            spatial_tokens,
            cache_tokens,
        )
        attended = attention(q, rotated_k, all_v)
        del q, rotated_k, all_v
    else:
        all_k = torch.cat([history_k, k], dim=1)
        all_v = torch.cat([history_v, v], dim=1)
        attended = attention(rotate(q, query_rope), rotate(all_k, key_rope), all_v)
    hidden = hidden + attn.to_out[0](attended.flatten(2)) * block.scale1
    normalized = block.norm2(hidden.float()).to(attn.to_q.weight.dtype)
    hidden = hidden + block.ff(normalized) * block.scale2
    if not fused:
        # Own only retained spatial history: views would retain the full cat,
        # including expired frames and the current special-token suffix.
        end = history_k.shape[1] + spatial_tokens
        keep = min(cache_tokens, end)
        cached_k = all_k[:, end - keep : end].clone()
        cached_v = all_v[:, end - keep : end].clone()
    return hidden, cached_k, cached_v
