# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _apply_rope
from sglang.multimodal_gen.runtime.models.dits.zing_h3_rope import apply_current_rope_kv


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("clean_rows", [0, 3, 7])
@pytest.mark.parametrize("rotary", [64, 128])
@pytest.mark.parametrize("strided", [False, True])
def test_rope_kv_slot_and_graph(monkeypatch, clean_rows, rotary, strided):
    monkeypatch.setenv("SGLANG_ZING_FUSED_CURRENT_ROPE_KV", "1")
    with torch.inference_mode():
        if strided:
            packed = torch.randn(7, 3, 384, device="cuda", dtype=torch.bfloat16)
            q, k, v = packed.chunk(3, -1)
        else:
            q, k, v = [
                torch.randn(7, 3, 128, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
        coefficients = torch.randn(7, rotary, device="cuda", dtype=q.dtype)
        raw, rotated, values = [
            torch.randn(19, 3, 128, device="cuda", dtype=q.dtype) for _ in range(3)
        ]
        # Non-contiguous destination row ordering detects assumptions about [L,L+n).
        rows = torch.tensor([11, 3, 14, 8, 9, 4, 5], device="cuda")
        plan = SimpleNamespace(
            stores=((raw, values),), rotated=(rotated,), clean_rows=clean_rows
        )
        tensors = {"write_rows": rows}
        expected_raw, expected_rotated, expected_values = (
            raw.clone(),
            rotated.clone(),
            values.clone(),
        )
        expected_rotated.index_copy_(0, rows, _apply_rope(k, coefficients))
        expected_values.index_copy_(0, rows, v)
        expected_raw.index_copy_(0, rows[:clean_rows], k[:clean_rows])
        expected_q = _apply_rope(q, coefficients)
        out = apply_current_rope_kv(q, k, v, coefficients, plan, 0, tensors)
        assert torch.equal(out, expected_q)
        assert torch.equal(raw, expected_raw)
        assert torch.equal(rotated, expected_rotated)
        assert torch.equal(values, expected_values)
        # Replay must consume updated tensor contents, not captured host values.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = apply_current_rope_kv(q, k, v, coefficients, plan, 0, tensors)
        q.add_(0.5)
        graph.replay()
        assert torch.equal(out, _apply_rope(q, coefficients))
        assert torch.equal(raw, expected_raw)
        assert torch.equal(rotated, expected_rotated)
        assert torch.equal(values, expected_values)


def test_cpu_fallback_preserves_slot_buffers(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_FUSED_CURRENT_ROPE_KV", "1")
    with torch.inference_mode():
        q, k, v = [torch.randn(7, 3, 128, dtype=torch.bfloat16) for _ in range(3)]
        stores = [torch.randn(19, 3, 128, dtype=q.dtype) for _ in range(3)]
        before = [x.clone() for x in stores]
        plan = SimpleNamespace(
            stores=((stores[0], stores[2]),), rotated=(stores[1],), clean_rows=3
        )
        result = apply_current_rope_kv(
            q,
            k,
            v,
            torch.zeros(7, 128, dtype=q.dtype),
            plan,
            0,
            {"write_rows": torch.arange(7)},
            default=True,
        )
        assert result is None
        assert all(torch.equal(actual, old) for actual, old in zip(stores, before))
