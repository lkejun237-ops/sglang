# SPDX-License-Identifier: Apache-2.0
"""Encoder 2x2, decoder block fusions and joint candidate; retired graph cells are historical only."""

import math

CELLS = (
    ("baseline", 0, 0, 0, 0),
    ("encoder-norm", 1, 0, 0, 0),
    ("encoder-copy", 0, 1, 0, 0),
    ("encoder-all", 1, 1, 0, 0),
    ("decoder-block", 0, 0, 1, 0),
    ("decoder-sm90", 0, 0, 1, 1),
    ("all", 1, 1, 1, 1),
)


def cell_environment(name):
    _, norm, copy, block, sm90 = next(c for c in CELLS if c[0] == name)
    return {
        "MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT": str(norm),
        "MINIMAX_H3_PROGRESSIVE_FUSED_CONV_INPUT": str(copy),
        "SGLANG_LYNN_PACKED_QKV": str(block),
        "SGLANG_LYNN_FUSED_QK_NORM": str(block),
        "SGLANG_LYNN_COMPILED_FFN": str(block),
        "SGLANG_LYNN_SM90_FAST_DECODE": str(sm90),
    }


def distribution(values):
    values = sorted(values)
    if not values:
        raise ValueError("Cannot summarize an empty measurement")

    def quantile(q):
        index = (len(values) - 1) * q
        lo = math.floor(index)
        return values[lo] + (values[math.ceil(index)] - values[lo]) * (index - lo)

    return dict(
        n=len(values),
        median=quantile(0.5),
        p95=quantile(0.95),
        minimum=values[0],
        maximum=values[-1],
    )
