# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only multi-chunk launch; never installed in the model path."""

import triton
import triton.language as tl


@triton.jit
def grouped_kernel(
    XS,
    CS,
    OUT,
    ROWS: tl.constexpr,
    STARTS: tl.constexpr,
    OFFSETS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    block = tl.program_id(0)
    for g in tl.static_range(len(ROWS)):
        if (block >= STARTS[g]) & (block < STARTS[g + 1]):
            i = (block - STARTS[g]) * B + tl.arange(0, B)
            mask = i < ROWS[g] * H * D
            row = i // (H * D)
            head = (i // D) % H
            d = i % D
            raw = tl.load(XS[g] + i, mask, 0)
            x = raw.to(tl.float32)
            partner = tl.where(d < R // 2, d + R // 2, d - R // 2)
            other = tl.load(
                XS[g] + row * H * D + head * D + partner, mask & (d < R), 0
            ).to(tl.float32)
            other = tl.where(d < R // 2, -other, other)
            c = tl.load(CS[g] + row * R + d % (R // 2), mask & (d < R), 0).to(
                tl.float32
            )
            s = tl.load(CS[g] + row * R + R // 2 + d % (R // 2), mask & (d < R), 0).to(
                tl.float32
            )
            a = (x * c).to(tl.bfloat16).to(tl.float32)
            b = (other * s).to(tl.bfloat16).to(tl.float32)
            tl.store(
                OUT + OFFSETS[g] * H * D + i,
                tl.where(d < R, (a + b).to(tl.bfloat16), raw),
                mask,
            )


def grouped(xs, cs, out):
    rows = tuple(x.shape[0] for x in xs)
    heads, dim = out.shape[1:]
    starts = [0]
    offsets = [0]
    for n in rows:
        starts.append(starts[-1] + triton.cdiv(n * heads * dim, 512))
        offsets.append(offsets[-1] + n)
    grouped_kernel[(starts[-1],)](
        tuple(xs),
        tuple(cs),
        out,
        rows,
        tuple(starts),
        tuple(offsets),
        heads,
        dim,
        cs[0].shape[-1],
        512,
        num_warps=4,
        enable_fp_fusion=False,
    )
