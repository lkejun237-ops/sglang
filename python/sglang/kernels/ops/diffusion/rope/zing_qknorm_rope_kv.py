# SPDX-License-Identifier: Apache-2.0
"""Fixed Zing QK norm, current RoPE, and KV persistence in one CUDA launch."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def _module(rotary: int):
    return load_jit(
        "zing_qknorm_rope_kv",
        rotary,
        cuda_files=["diffusion/zing_qknorm_rope_kv.cuh"],
        cuda_wrappers=[("run", f"ZingQKNormRopeKVKernel<{rotary}>::run")],
    )


@register_custom_op(mutates_args=["out_q", "raw_k", "rotated_k", "out_v"])
def fused_zing_qknorm_rope_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    coefficients: torch.Tensor,
    rows: torch.Tensor,
    out_q: torch.Tensor,
    raw_k: torch.Tensor,
    rotated_k: torch.Tensor,
    out_v: torch.Tensor,
    clean_rows: int,
    eps: float,
) -> None:
    _module(coefficients.shape[-1]).run(
        q,
        k,
        v,
        q_weight,
        k_weight,
        coefficients,
        rows,
        out_q,
        raw_k,
        rotated_k,
        out_v,
        clean_rows,
        eps,
    )
