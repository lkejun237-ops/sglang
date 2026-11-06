# SPDX-License-Identifier: Apache-2.0
"""Configurable Lynn block execution. Immutable weight copies are model-owned;
packed projections and compiled FFNs are invalidated when parameters change.
"""

import os
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LynnExecutionOptions:
    # Explicit options start from the unfused reference; serving uses from_env().
    packed_qkv: bool = False
    fused_qk_norm: bool = False
    compiled_ffn: bool = False

    @classmethod
    def from_env(cls):
        def flag(name, default="0"):
            value = os.environ.get("SGLANG_LYNN_" + name, default)
            if value not in ("0", "1"):
                raise ValueError(f"SGLANG_LYNN_{name} must be 0 or 1")
            return value == "1"

        return cls(
            packed_qkv=flag("PACKED_QKV", "1"),
            fused_qk_norm=flag("FUSED_QK_NORM", "1"),
            compiled_ffn=flag("COMPILED_FFN", "1"),
        )

    @property
    def block_enabled(self):
        return self.packed_qkv or self.fused_qk_norm or self.compiled_ffn


def parameter_signature(module):
    # Host metadata only; no device synchronization or tensor-value readback.
    return tuple(
        (
            id(p),
            p._version,
            p.device,
            p.dtype,
            tuple(p.shape),
            tuple(p.stride()),
            p.data_ptr(),
        )
        for p in module.parameters()
    )


def packed_projection(attn, normalized, packed):
    if packed is None:
        q, k, v = attn.to_q(normalized), attn.to_k(normalized), attn.to_v(normalized)
    else:
        weight, bias = packed
        q, k, v = F.linear(normalized, weight, bias).chunk(3, dim=-1)
        # The existing preparation path requires dense V. RMSNorm materializes Q/K.
        v = v.contiguous()
    shape = (attn.heads, -1)
    return q.unflatten(-1, shape), k.unflatten(-1, shape), v.unflatten(-1, shape)


def ffn_residual(block, hidden):
    normalized = block.norm2(hidden.float()).to(block.attn.to_q.weight.dtype)
    return hidden + block.ff(normalized) * block.scale2


class LynnBlockExecution:
    def __init__(self, options):
        self.options = options
        self.packed = {}
        self.ffns = {}
        self.counts = {"packed_qkv": 0, "compiled_ffn": 0, "fallback": 0}

    def clear(self):
        self.packed.clear()
        self.ffns.clear()

    def prepare_qkv(self, block):
        if not self.options.packed_qkv:
            return None
        attn = block.attn
        modules = (attn.to_q, attn.to_k, attn.to_v)
        if not all(
            type(m) is torch.nn.Linear
            and not m._forward_hooks
            and not m._forward_pre_hooks
            and "forward" not in m.__dict__
            for m in modules
        ):
            self.counts["fallback"] += 1
            return None
        weights = [m.weight for m in modules]
        biases = [m.bias for m in modules]
        if not all(
            w.shape == weights[0].shape
            and w.device == weights[0].device
            and w.dtype == weights[0].dtype
            for w in weights
        ):
            self.counts["fallback"] += 1
            return None
        if len({b is None for b in biases}) != 1:
            self.counts["fallback"] += 1
            return None
        signature = parameter_signature(attn)
        cached = self.packed.get(block)
        if cached is None or cached[0] != signature:
            packed = (
                torch.cat(weights).detach(),
                None if biases[0] is None else torch.cat(biases).detach(),
            )
            self.packed[block] = (signature, packed)
        self.counts["packed_qkv"] += 1
        return self.packed[block][1]

    def run(
        self,
        block,
        hidden,
        hk,
        hv,
        *,
        query_rope,
        key_rope,
        spatial_tokens,
        cache_tokens,
        attention,
    ):
        from .attention import streaming_block

        signature = parameter_signature(block)
        packed = self.prepare_qkv(block)
        ffn = None
        if self.options.compiled_ffn and hidden.is_cuda:
            cached = self.ffns.get(block)
            if cached is None or cached[0] != signature:
                self.ffns[block] = (
                    signature,
                    torch.compile(
                        partial(ffn_residual, block), fullgraph=True, dynamic=False
                    ),
                )
            ffn = self.ffns[block][1]
            self.counts["compiled_ffn"] += 1

        return streaming_block(
            block,
            hidden,
            hk,
            hv,
            query_rope=query_rope,
            key_rope=key_rope,
            spatial_tokens=spatial_tokens,
            cache_tokens=cache_tokens,
            attention=attention,
            packed_qkv=packed,
            fused_qk_norm=self.options.fused_qk_norm,
            ffn=ffn,
        )
