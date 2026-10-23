# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _apply_rope
from sglang.multimodal_gen.runtime.models.dits.zing_h3_rope import (
    apply_current_qknorm_rope_kv,
    use_current_qknorm_rope_kv,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("clean", [0, 3, 7])
@pytest.mark.parametrize(
    "row_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize("packed", [False, True])
def test_norm_rope_kv_matches_separate_kernels_and_replay(
    monkeypatch, clean, row_dtype, packed
):
    monkeypatch.setenv("SGLANG_ZING_FUSED_QKNORM_ROPE_KV", "1")
    with torch.inference_mode():
        torch.manual_seed(17)
        module = SimpleNamespace(head_dim=128)
        for name in ("q_norm", "k_norm"):
            norm = torch.nn.RMSNorm(128, eps=1e-6, device="cuda", dtype=torch.bfloat16)
            norm.weight.copy_(torch.randn_like(norm.weight))
            setattr(module, name, norm)
        if packed:
            q, k, v = torch.randn(7, 3, 384, device="cuda", dtype=torch.bfloat16).chunk(
                3, -1
            )
        else:
            q, k, v = [
                torch.randn(7, 3, 128, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
        coefficients = torch.randn(7, 96, device="cuda", dtype=q.dtype)
        rows = torch.tensor([11, 3, 14, 8, 9, 4, 5], device="cuda", dtype=row_dtype)
        raw, rotated, values = [
            torch.randn(19, 3, 128, device="cuda", dtype=q.dtype) for _ in range(3)
        ]
        plan = SimpleNamespace(
            stores=((raw, values),), rotated=(rotated,), clean_rows=clean
        )
        tensors = {"write_rows": rows}
        assert use_current_qknorm_rope_kv(module, q, coefficients)

        def reference():
            nq, nk = q.contiguous().clone(), k.contiguous().clone()
            fused_inplace_qknorm(
                nq,
                nk,
                module.q_norm.weight,
                module.k_norm.weight,
                eps=module.q_norm.eps,
            )
            expected_raw, expected_rotated, expected_values = (
                raw.clone(),
                rotated.clone(),
                values.clone(),
            )
            expected_rotated.index_copy_(0, rows.long(), _apply_rope(nk, coefficients))
            expected_values.index_copy_(0, rows.long(), v)
            expected_raw.index_copy_(0, rows[:clean].long(), nk[:clean])
            return (
                _apply_rope(nq, coefficients),
                expected_raw,
                expected_rotated,
                expected_values,
            )

        expected = reference()
        out = apply_current_qknorm_rope_kv(
            module, q, k, v, coefficients, plan, 0, tensors
        )
        for actual, target in zip((out, raw, rotated, values), expected):
            assert torch.equal(actual, target), (
                (actual.float() - target.float()).abs().max().item()
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = apply_current_qknorm_rope_kv(
                module, q, k, v, coefficients, plan, 0, tensors
            )
        for tensor in (q, k, v, coefficients):
            tensor.add_(0.25)
        rows.copy_(
            torch.tensor([1, 10, 7, 6, 2, 12, 0], device="cuda", dtype=row_dtype)
        )
        expected = reference()
        graph.replay()
        for actual, target in zip((out, raw, rotated, values), expected):
            assert torch.equal(actual, target)


@torch.no_grad()
def test_cpu_does_not_defer_norm(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_FUSED_QKNORM_ROPE_KV", "1")
    module = SimpleNamespace(head_dim=128)
    assert not use_current_qknorm_rope_kv(
        module, torch.zeros(3, 2, 128), torch.zeros(3, 96)
    )
