# SPDX-License-Identifier: Apache-2.0
"""History RoPE over disjoint chunks, without packing K or coefficient tensors."""

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit(do_not_specialize=["ROWS", "STARTS", "OFFSETS", "STRIDES"])
def _grouped_history_rope_kernel(
    XS,
    CS,
    OUT,
    ROWS,
    STARTS,
    OFFSETS,
    STRIDES,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    block = tl.program_id(0)
    # Select one chunk per CTA, then share the arithmetic body. Pointer values
    # remain runtime arguments; shape and alignment classes may cache variants.
    x, table = XS[0], CS[0]
    rows, start, offset, stride = ROWS[0], STARTS[0], OFFSETS[0], STRIDES[0]
    for g in tl.static_range(1, len(XS)):
        active = (block >= STARTS[g]) & (block < STARTS[g + 1])
        x = tl.where(active, XS[g], x)
        table = tl.where(active, CS[g], table)
        rows = tl.where(active, ROWS[g], rows)
        start = tl.where(active, STARTS[g], start)
        offset = tl.where(active, OFFSETS[g], offset)
        stride = tl.where(active, STRIDES[g], stride)
    i = (block - start) * B + tl.arange(0, B)
    mask = i < rows * H * D
    row = i // (H * D)
    head = (i // D) % H
    d = i % D
    raw = tl.load(x + row * stride + head * D + d, mask, 0)
    partner = tl.where(d < R // 2, d + R // 2, d - R // 2)
    other = tl.load(x + row * stride + head * D + partner, mask & (d < R), 0).to(
        tl.float32
    )
    other = tl.where(d < R // 2, -other, other)
    c = tl.load(table + row * R + d % (R // 2), mask & (d < R), 0).to(tl.float32)
    s = tl.load(table + row * R + R // 2 + d % (R // 2), mask & (d < R), 0).to(
        tl.float32
    )
    a = (raw.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
    b = (other * s).to(tl.bfloat16).to(tl.float32)
    tl.store(
        OUT + tl.cast(offset, tl.int64) * H * D + i,
        tl.where(d < R, (a + b).to(tl.bfloat16), raw),
        mask,
    )


@register_custom_op(mutates_args=["out"])
def fused_zing_history_rope_grouped(
    xs: list[torch.Tensor], coefficients: list[torch.Tensor], out: torch.Tensor
) -> None:
    heads, dim = out.shape[1:]
    rotary = coefficients[0].shape[-1]
    offset = 0
    block_size = 4096
    with torch.cuda.device(out.device):
        # Bounded variants rather than specializing on every live chunk count.
        for begin in range(0, len(xs), 32):
            inputs = xs[begin : begin + 32]
            tables = coefficients[begin : begin + 32]
            rows = [x.shape[0] for x in inputs]
            strides = [x.stride(0) for x in inputs]
            starts, offsets = [0], []
            for n in rows:
                starts.append(starts[-1] + triton.cdiv(n * heads * dim, block_size))
                offsets.append(offset)
                offset += n
            capacity = triton.next_power_of_2(len(inputs))
            for _ in range(capacity - len(inputs)):
                inputs.append(inputs[0])
                tables.append(tables[0])
                rows.append(0)
                strides.append(strides[0])
                starts.append(starts[-1])
                offsets.append(offset)
            _grouped_history_rope_kernel[(starts[-1],)](
                tuple(inputs),
                tuple(tables),
                out,
                tuple(rows),
                tuple(starts),
                tuple(offsets),
                tuple(strides),
                heads,
                dim,
                rotary,
                block_size,
                num_warps=4,
                enable_fp_fusion=False,
            )
