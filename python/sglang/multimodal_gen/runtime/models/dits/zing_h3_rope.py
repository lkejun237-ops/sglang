# SPDX-License-Identifier: Apache-2.0
"""History-only RoPE dispatch; cache invalidation remains owned by the cache."""

import collections
import os

import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _apply_rope_into
from sglang.multimodal_gen.runtime.utils.profiler import maybe_record_function

HISTORY_ROPE_ROUTES = collections.Counter()


def _history_rope_rejection(x, coefficients, out, *, alias_out=None):
    if torch.is_grad_enabled():
        return "autograd"
    if not x.is_cuda or torch.version.hip is not None:
        return "device"
    if x.dtype != torch.bfloat16:
        return "dtype"
    if x.ndim != 3 or coefficients.ndim != 2 or out.shape != x.shape:
        return "shape"
    rows, heads, dim = x.shape
    rotary = coefficients.shape[-1]
    if not (
        rows > 0
        and heads > 0
        and dim > 0
        and 0 < rotary <= dim
        and rotary % 2 == 0
        and coefficients.shape[0] == rows
        and all(
            t.device == x.device and t.dtype == x.dtype for t in (coefficients, out)
        )
    ):
        return "shape_or_dtype"
    if not (
        x.stride(-1) == 1
        and x.stride(-2) == dim
        and x.stride(0) >= heads * dim
        and (rows - 1) * x.stride(0) + heads * dim < 2**31
        and coefficients.numel() < 2**31
        and coefficients.is_contiguous()
        and out.is_contiguous()
    ):
        return "layout"
    alias_out = out if alias_out is None else alias_out
    if torch._C._overlaps(alias_out, x) or torch._C._overlaps(alias_out, coefficients):
        return "alias"
    return None


def apply_history_rope_into(x, coefficients, out):
    # Opt-in until the long-window CUDA and end-to-end acceptance gates pass.
    reason = (
        "disabled"
        if os.environ.get("SGLANG_ZING_FUSED_HISTORY_ROPE", "0") != "1"
        else _history_rope_rejection(x, coefficients, out)
    )
    HISTORY_ROPE_ROUTES[reason or "fused"] += 1
    HISTORY_ROPE_ROUTES["rows"] += x.shape[0]
    with maybe_record_function("Zing.history_K_RoPE_rebuild"):
        if reason is not None:
            return _apply_rope_into(x, coefficients, out)
        from sglang.kernels.ops.diffusion.rope.zing_history_rope import (
            fused_zing_history_rope,
        )

        fused_zing_history_rope(x, coefficients, out)
        return out


def apply_history_rope_batch(xs, coefficients, out):
    """Consume already-invalid spans; never select or invalidate cache history."""
    if len(xs) != len(coefficients) or sum(x.shape[0] for x in xs) != out.shape[0]:
        raise ValueError("History RoPE batch rows or coefficient count mismatch")
    if not xs:
        return out
    offset = 0
    reason = None
    for x, table in zip(xs, coefficients):
        end = offset + x.shape[0]
        # Reject cross-chunk aliases as well as aliases within one chunk.
        rejection = _history_rope_rejection(x, table, out[offset:end], alias_out=out)
        if table.shape[-1] != coefficients[0].shape[-1]:
            rejection = "rotary_dimensions"
        reason = reason or rejection
        offset = end
    if os.environ.get("SGLANG_ZING_FUSED_HISTORY_ROPE", "0") != "1":
        reason = "disabled"
    if reason is not None:
        HISTORY_ROPE_ROUTES["grouped_fallback_" + reason] += 1
        offset = 0
        for x, table in zip(xs, coefficients):
            end = offset + x.shape[0]
            apply_history_rope_into(x, table, out[offset:end])
            offset = end
        return out
    from sglang.kernels.ops.diffusion.rope.zing_history_rope_grouped import (
        fused_zing_history_rope_grouped,
    )

    HISTORY_ROPE_ROUTES["grouped"] += 1
    HISTORY_ROPE_ROUTES["grouped_chunks"] += len(xs)
    HISTORY_ROPE_ROUTES["rows"] += out.shape[0]
    with maybe_record_function("Zing.history_K_RoPE_rebuild"):
        fused_zing_history_rope_grouped(xs, coefficients, out)
    return out


def apply_contiguous_history_rope_batch(xs, coefficients, out):
    """Ablation: view adjacent raw K rows; pack only coefficients in this call."""
    if len(xs) != len(coefficients) or sum(x.shape[0] for x in xs) != out.shape[0]:
        raise ValueError("History RoPE batch rows or coefficient count mismatch")
    if not xs:
        return out
    supported = os.environ.get("SGLANG_ZING_FUSED_HISTORY_ROPE", "0") == "1"
    offset = 0
    first = xs[0]
    for x, table in zip(xs, coefficients):
        end = offset + x.shape[0]
        supported = supported and (
            _history_rope_rejection(x, table, out[offset:end], alias_out=out) is None
            and x.stride() == first.stride()
            and x.data_ptr()
            == first.data_ptr() + offset * first.stride(0) * first.element_size()
            and x.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
            and table.shape[-1] == coefficients[0].shape[-1]
        )
        offset = end
    if not supported:
        HISTORY_ROPE_ROUTES["contiguous_fallback"] += 1
        return apply_history_rope_batch(xs, coefficients, out)
    raw = first.as_strided(out.shape, first.stride())
    with maybe_record_function("Zing.history_RoPE_coefficient_pack"):
        table = torch.cat(coefficients)
    HISTORY_ROPE_ROUTES["contiguous"] += 1
    return apply_history_rope_into(raw, table, out)
