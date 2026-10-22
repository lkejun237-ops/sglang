# SPDX-License-Identifier: Apache-2.0
"""History-only RoPE dispatch; cache invalidation remains owned by the cache."""

import collections
import functools
import importlib.metadata
import os

import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _apply_rope,
    _apply_rope_into,
)
from sglang.multimodal_gen.runtime.utils.profiler import maybe_record_function

HISTORY_ROPE_ROUTES = collections.Counter()
CURRENT_ROPE_ROUTES = collections.Counter()


def _history_rope_rejection(
    x, coefficients, out, *, alias_out=None, strided_heads=False
):
    """Return why the fused kernel cannot run, or None.

    ``out=None`` checks only the input; a fresh contiguous output of x's shape
    always passes the output checks, so the caller allocates it afterwards.
    """
    if torch.is_grad_enabled():
        return "autograd"
    if not x.is_cuda or torch.version.hip is not None:
        return "device"
    if x.dtype != torch.bfloat16:
        return "dtype"
    if x.ndim != 3 or coefficients.ndim != 2:
        return "shape"
    if out is not None and out.shape != x.shape:
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
            t.device == x.device and t.dtype == x.dtype
            for t in (coefficients, out)
            if t is not None
        )
    ):
        return "shape_or_dtype"
    if not (
        x.stride(-1) == 1
        and (x.stride(-2) == dim or (strided_heads and x.stride(-2) >= dim))
        and x.stride(0) >= (heads - 1) * x.stride(-2) + dim
        and (rows - 1) * x.stride(0) + (heads - 1) * x.stride(-2) + dim < 2**31
        and coefficients.numel() < 2**31
        and coefficients.is_contiguous()
        and (out is None or out.is_contiguous())
    ):
        return "layout"
    if out is None:
        return None
    alias_out = out if alias_out is None else alias_out
    if torch._C._overlaps(alias_out, x) or torch._C._overlaps(alias_out, coefficients):
        return "alias"
    return None


def apply_current_rope(x, coefficients):
    """Rotate the current block's Q or K into a new contiguous tensor.

    Enabled by default. The history kernel reads the row-strided QKV view in
    place of eager's cos/sin concatenations and output concatenation, including
    the interleaved head stride of the post-Ulysses Q/K views.
    """
    reason = "disabled"
    if os.environ.get("SGLANG_ZING_FUSED_CURRENT_ROPE", "1") == "1":
        # Check the input first so a rejected call allocates nothing extra.
        reason = _history_rope_rejection(x, coefficients, None, strided_heads=True)
    CURRENT_ROPE_ROUTES[reason or "fused"] += 1
    if reason is not None:
        return _apply_rope(x, coefficients)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    from sglang.kernels.ops.diffusion.rope.zing_history_rope import (
        fused_zing_history_rope,
    )

    fused_zing_history_rope(x, coefficients, out)
    return out


def apply_history_rope_into(x, coefficients, out):
    # Enabled by default; unsupported inputs retain the eager fallback.
    reason = (
        "disabled"
        if os.environ.get("SGLANG_ZING_FUSED_HISTORY_ROPE", "1") != "1"
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
    if os.environ.get("SGLANG_ZING_FUSED_HISTORY_ROPE", "1") != "1":
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
    supported = os.environ.get("SGLANG_ZING_FUSED_HISTORY_ROPE", "1") == "1"
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


CURRENT_KV_ROUTES = collections.Counter()


def apply_current_rope_kv(
    q, k, v, coefficients, plan, layer, tensors, *, default=False
):
    """Rotate Q/K and write the fixed slot; return None for the legacy path.

    The plan owns unique device write_rows and the clean prefix length. No noisy
    raw K is persisted, and no history row or cache metadata is changed here.
    """
    if (
        os.environ.get("SGLANG_ZING_FUSED_CURRENT_ROPE_KV", "1" if default else "0")
        != "1"
    ):
        return None
    raw_k, out_v = plan.stores[layer]
    rotated_k = plan.rotated[layer]
    rows = tensors["write_rows"]
    supported = (
        not torch.is_grad_enabled()
        and q.is_cuda
        and torch.version.hip is None
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and all(size > 0 for size in q.shape)
        and q.numel() < 2**31
        and q.shape == k.shape == v.shape
        and all(
            (x.shape[0] - 1) * x.stride(0) + x.shape[1] * x.stride(1) < 2**31
            for x in (q, k, v)
        )
        and coefficients.ndim == 2
        and coefficients.shape[0] == q.shape[0]
        and 0 < coefficients.shape[1] <= q.shape[-1]
        and coefficients.shape[1] % 2 == 0
        and coefficients.is_contiguous()
        and rows.ndim == 1
        and rows.shape[0] == q.shape[0]
        and rows.dtype in (torch.int32, torch.int64)
        and rows.is_contiguous()
        and all(
            x.device == q.device
            for x in (k, v, coefficients, rows, raw_k, rotated_k, out_v)
        )
        and all(
            x.dtype == q.dtype for x in (k, v, coefficients, raw_k, rotated_k, out_v)
        )
        and all(x.stride(-1) == 1 for x in (q, k, v))
        and all(
            x.is_contiguous() and x.shape[1:] == q.shape[1:]
            for x in (raw_k, rotated_k, out_v)
        )
    )
    CURRENT_KV_ROUTES["fused" if supported else "fallback"] += 1
    if not supported:
        return None
    from sglang.kernels.ops.diffusion.rope.zing_current_rope_kv import (
        fused_zing_current_rope_kv,
    )

    out_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    fused_zing_current_rope_kv(
        q, k, v, coefficients, rows, out_q, raw_k, rotated_k, out_v, plan.clean_rows
    )
    return out_q


@functools.lru_cache(maxsize=8)
def _validated_fixed_device(device):
    """The measured FA4/hardware pair; query once, outside Graph capture."""
    if device.type != "cuda" or torch.version.hip is not None:
        return False
    try:
        version = importlib.metadata.version("flash-attn-4")
    except importlib.metadata.PackageNotFoundError:
        return False
    return (
        version == "4.0.0b19"
        and torch.cuda.get_device_capability(device) == (10, 3)
        and torch.cuda.get_device_properties(device).multi_processor_count == 148
    )


def validated_fixed_480p(host, device, *, sp_size, heads, head_dim, backend, audio):
    """Recognize the measured 15x26 spatial grid on existing CPU metadata.

    A query row count alone is ambiguous across resolutions/chunk plans. Cache
    the geometry result with the immutable host layout, never read GPU positions.
    """
    if (sp_size, heads, head_dim, backend, audio) != (2, 56, 128, "fa4", False):
        return False
    if not _validated_fixed_device(device):
        return False
    key = "fixed_480p_grid"
    if key not in host.derived:
        visual = (host.row_role == 7) | (host.row_role == 10)
        positions = host.position_ids[visual, 1:]
        host.derived[key] = (
            torch.unique(positions[:, 0]).numel() == 15
            and torch.unique(positions[:, 1]).numel() == 26
            and torch.unique(positions, dim=0).shape[0] == 390
            and host.row_role.numel() in (390, 780, 1170, 1560, 2340, 2730)
            and bool(visual.all())
        )
    return host.derived[key]
