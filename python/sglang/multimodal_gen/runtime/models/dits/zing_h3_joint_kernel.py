# SPDX-License-Identifier: Apache-2.0
"""CuTe/FA4 prefix-branch kernel, selected by the guarded runtime dispatcher.

Metadata is prepared once per geometry outside the timed/layer path. Each branch
is internally bidirectional. This is an FA4 pipeline specialization, not a new
MMA pipeline or a claim of additional inter-CTA KV sharing.
"""

import math

import cutlass.cute as cute
import torch
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.interface import flash_attn_func


def make_branch_mask(history, e0, e1, e2):
    @cute.jit
    def mask(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
        return (
            ((q_idx < e0) & (kv_idx < history + e0))
            | ((q_idx >= e0) & (q_idx < e1) & (kv_idx < history + e1))
            | ((q_idx >= e1) & (q_idx < e2) & (kv_idx < history + e2))
        )

    return mask


def block_ranges(history, ends, tile_m=256, tile_n=128):
    """Analytic full/partial block lists; no dense M x N mask materialization."""
    if (
        history < 0
        or len(ends) not in (2, 3)
        or any(b <= a for a, b in zip((0, *ends[:-1]), ends))
    ):
        raise ValueError("Expected nonempty ordered branches and nonnegative history")
    rows = []
    for start in range(0, ends[-1], tile_m):
        stop = min(start + tile_m, ends[-1])
        low = history + next(e for e in ends if start < e)
        high = history + next(e for e in ends if stop - 1 < e)
        full = list(range(low // tile_n))
        partial = list(range(low // tile_n, math.ceil(high / tile_n)))
        rows.append((full, partial))
    return rows


class JointAttentionKernel:
    def __init__(self, history, ends, device, *, sparse=True):
        self.history, self.ends = history, tuple(ends)
        rows = block_ranges(history, self.ends)
        self.mask = make_branch_mask(history, self.ends[0], self.ends[1], self.ends[-1])
        self.blocks = None
        if sparse:
            cols = math.ceil((history + self.ends[-1]) / 128)
            full_idx = torch.zeros((1, 1, len(rows), cols), dtype=torch.int32)
            mask_idx = torch.zeros_like(full_idx)
            full_cnt = torch.zeros((1, 1, len(rows)), dtype=torch.int32)
            mask_cnt = torch.zeros_like(full_cnt)
            for i, (full, partial) in enumerate(rows):
                full_cnt[0, 0, i], mask_cnt[0, 0, i] = len(full), len(partial)
                full_idx[0, 0, i, : len(full)] = torch.tensor(full, dtype=torch.int32)
                mask_idx[0, 0, i, : len(partial)] = torch.tensor(
                    partial, dtype=torch.int32
                )
            self.blocks = BlockSparseTensorsTorch(
                mask_block_cnt=mask_cnt.to(device),
                mask_block_idx=mask_idx.to(device),
                full_block_cnt=full_cnt.to(device),
                full_block_idx=full_idx.to(device),
                block_size=(256, 128),
            )

    def __call__(self, q, k, v):
        if (
            q.ndim != 3
            or q.shape[0] != self.ends[-1]
            or k.shape != v.shape
            or k.shape[0] != self.history + self.ends[-1]
            or q.shape[1:] != k.shape[1:]
            or q.shape[-1] != 128
            or q.dtype != torch.bfloat16
            or k.dtype != q.dtype
            or v.dtype != q.dtype
            or not q.is_cuda
            or k.device != q.device
            or v.device != q.device
        ):
            raise ValueError(
                "Unsupported joint attention input geometry, device or dtype"
            )
        result = flash_attn_func(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            softmax_scale=128**-0.5,
            causal=False,
            mask_mod=self.mask,
            block_sparse_tensors=self.blocks,
        )
        return (result[0] if isinstance(result, tuple) else result).squeeze(0)
