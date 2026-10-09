# SPDX-License-Identifier: Apache-2.0
"""Use the measured FA4 specialization only within its validated envelope."""

import math
import os
from collections import OrderedDict

import torch

from sglang.multimodal_gen.runtime.utils.profiler import maybe_record_function

# These are the complete post-Ulysses lengths, not rank-local sequence shards.
_VALIDATED_LENGTHS = {(880, 390), (390, 1760, 780), (780, 1760, 780), (780, 880, 390)}
_KERNELS = OrderedDict()


def eligible_geometry(lengths, history, heads, width, capability):
    return (
        tuple(lengths) in _VALIDATED_LENGTHS
        and 12246 <= history <= 55426
        and heads == 28
        and width == 128
        and capability == (10, 3)
    )


def fused_joint_attention(q, k, v, ends, scale, backend):
    mode = os.environ.get("SGLANG_ZING_JOINT_ATTENTION", "auto")
    if mode not in {"auto", "separate"}:
        raise ValueError("SGLANG_ZING_JOINT_ATTENTION must be auto or separate")
    if mode == "separate" or backend != "fa4" or not q.is_cuda:
        return None
    ends = tuple(ends)
    history = k.shape[0] - ends[-1]
    lengths = tuple(b - a for a, b in zip((0, *ends[:-1]), ends))
    if (
        q.dtype != torch.bfloat16
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or k.device != q.device
        or v.device != q.device
        or k.shape != v.shape
        or q.shape[1:] != k.shape[1:]
        or not math.isclose(scale, 128**-0.5, rel_tol=1e-12)
        or not eligible_geometry(
            lengths,
            history,
            q.shape[1],
            q.shape[-1],
            torch.cuda.get_device_capability(q.device),
        )
    ):
        return None
    # Never allocate new metadata/compile a specialization during graph capture.
    if torch.cuda.is_current_stream_capturing():
        return None
    try:
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_joint_kernel import (
            JointAttentionKernel,
        )
    except ImportError:
        # Older FA4 packages may lack the optional sparse CuTe interface.
        return None
    key = (history, ends, q.device, torch.cuda.current_stream(q.device).cuda_stream)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = JointAttentionKernel(history, ends, q.device, sparse=True)
        _KERNELS[key] = kernel
        if len(_KERNELS) > 64:
            _KERNELS.popitem(last=False)
    else:
        _KERNELS.move_to_end(key)
    # Execution failures must propagate; do not silently retry after CUDA errors.
    with maybe_record_function("Zing.FA4.joint_fused"):
        return kernel(q, k, v)
