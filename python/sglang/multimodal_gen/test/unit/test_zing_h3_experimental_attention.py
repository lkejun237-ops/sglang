# SPDX-License-Identifier: Apache-2.0
"""Experimental routes are explicit; CPU mocks do not validate FA4 numerics."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits import zing_h3
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (
    model as model_fixture,
    request,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_sp_cpu import (
    test_real_projections_rope_and_committed_kv_match_cpu_sp1 as check_sp_layout,
)


@pytest.mark.parametrize("backend", ["torch_sdpa", "fa4"])
@pytest.mark.parametrize("stable", ["0", "1"])
@pytest.mark.parametrize("compiled", ["0", "1"])
@torch.no_grad()
def test_model_snapshots_experiments_and_routes_refiner_and_main(
    monkeypatch, backend, stable, compiled
):
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_ATTENTION", backend)
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", stable)
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN", compiled)
    model = model_fixture.__wrapped__()
    assert model.experimental_attention_backend == backend
    assert model.experimental_stable_shapes is (stable == "1")
    assert model.experimental_compiled_adaln is (compiled == "1")
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_ATTENTION", "invalid-after-init")
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", "invalid-after-init")
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN", "invalid-after-init")
    norm_calls = []

    def fused_spy(hidden, norm, shift, scale):
        norm_calls.append(norm)
        return norm(hidden) * (1 + scale) + shift

    monkeypatch.setattr(zing_h3, "_native_compiled_adaln", fused_spy)
    original = zing_h3._attention
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs["attention_backend"])
        kwargs["attention_backend"] = "torch_sdpa"
        return original(*args, **kwargs)

    monkeypatch.setattr(zing_h3, "_attention", spy)
    packer, inputs, visual, audio = request(model)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, audio)
    prefix = sampler._select_block(layout, 0, phase="prefix", video_t=1, audio_t=1)
    sampler._forward(prefix, ZingH3KVCache(len(model.blocks)), "prefix")
    assert seen == [backend] * (len(model.token_refiner.blocks) + len(model.blocks))
    assert len(norm_calls) == (2 * len(model.blocks) if compiled == "1" else 0)


def test_default_experiments_are_off(monkeypatch):
    monkeypatch.delenv("SGLANG_ZING_EXPERIMENTAL_ATTENTION", raising=False)
    monkeypatch.delenv("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", raising=False)
    monkeypatch.delenv("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN", raising=False)
    model = model_fixture.__wrapped__()
    assert model.experimental_attention_backend == "torch_sdpa"
    assert model.experimental_stable_shapes is False
    assert model.experimental_compiled_adaln is False


@pytest.mark.parametrize(
    "variable",
    [
        "SGLANG_ZING_EXPERIMENTAL_ATTENTION",
        "SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES",
        "SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN",
    ],
)
def test_invalid_experiment_fails_at_construction(monkeypatch, variable):
    monkeypatch.setenv(variable, "invalid")
    with pytest.raises(ValueError, match=variable):
        zing_h3.ZingH3DiTModel(None, {})


def test_compiled_adaln_rejects_cpu():
    with pytest.raises(ValueError, match="requires CUDA"):
        zing_h3._native_compiled_adaln(torch.zeros(2, 4), None, None, None)


def test_fa4_rejects_mask_without_running_projection():
    with pytest.raises(ValueError, match="does not support explicit attention masks"):
        zing_h3._attention(None, None, mask=torch.ones(1, 1), attention_backend="fa4")
    with pytest.raises(ValueError, match="Unsupported"):
        zing_h3._attention(None, None, attention_backend="auto")


def test_fa4_rejects_cpu_and_missing_dependency(monkeypatch):
    cpu = torch.zeros(3, 2, 8)
    with pytest.raises(ValueError, match="requires CUDA"):
        zing_h3._fa4_attention(cpu, cpu, cpu, 0.5)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", None)
    fake_cuda = SimpleNamespace(is_cuda=True)
    with pytest.raises(RuntimeError, match="requires flash_attn.cute"):
        zing_h3._fa4_attention(fake_cuda, fake_cuda, fake_cuda, 0.5)


@pytest.mark.parametrize("tuple_result", [False, True])
def test_fa4_external_signature_and_single_document_lengths(monkeypatch, tuple_result):
    # CPU-backed duck tensors isolate the call contract, not CUDA execution.
    q = SimpleNamespace(is_cuda=True, shape=(3, 2, 8), device=torch.device("cpu"))
    k = SimpleNamespace(shape=(7, 2, 8), device=torch.device("cpu"))
    v = SimpleNamespace(shape=(7, 2, 8), device=torch.device("cpu"))
    expected = torch.zeros(3, 2, 8)
    calls = []
    fake = ModuleType("flash_attn.cute")

    def flash(**kwargs):
        calls.append(kwargs)
        return (expected, None) if tuple_result else expected

    fake.flash_attn_varlen_func = flash
    monkeypatch.setitem(sys.modules, "flash_attn.cute", fake)
    assert zing_h3._fa4_attention(q, k, v, 0.125) is expected
    call = calls[0]
    assert call["q"] is q and call["k"] is k and call["v"] is v
    assert call["cu_seqlens_q"].tolist() == [0, 3]
    assert call["cu_seqlens_k"].tolist() == [0, 7]
    assert call["cu_seqlens_q"].dtype == call["cu_seqlens_k"].dtype == torch.int32
    assert call["max_seqlen_q"] == 3 and call["max_seqlen_k"] == 7
    assert call["softmax_scale"] == 0.125
    assert call["window_size"] == (None, None)
    assert call["causal"] is call["deterministic"] is call["return_lse"] is False


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("reference_prefix", [False, True])
def test_opt_in_stable_shapes_keep_sp_cache_and_outputs(
    monkeypatch, world, reference_prefix
):
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", "1")
    model = model_fixture.__wrapped__()
    check_sp_layout(model, monkeypatch, world, reference_prefix)


@pytest.mark.parametrize("sp", [False, True])
@torch.no_grad()
def test_fa4_routes_full_query_and_history_before_output_projection(monkeypatch, sp):
    model = model_fixture.__wrapped__()
    module = model.blocks[0].attn
    hidden = torch.randn(3, 16).bfloat16()
    history = (torch.randn(4, 2, 8).bfloat16(), torch.randn(4, 2, 8).bfloat16())
    rope = torch.cat((torch.ones(4, 4), torch.zeros(4, 4)), dim=-1).bfloat16()
    reference, _ = zing_h3._attention(module, hidden, rope[:3], history, rope)
    calls = []

    def fake_fa4(q, k, v, scale):
        calls.append((q.shape, k.shape, v.shape, scale))
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )[0].transpose(0, 1)

    monkeypatch.setattr(zing_h3, "_fa4_attention", fake_fa4)
    if sp:
        monkeypatch.setattr(
            zing_h3,
            "_usp_input_all_to_all_varlen_qkv",
            lambda q, k, v, splits: torch.cat((q, k, v), dim=-1),
        )
        monkeypatch.setattr(
            zing_h3,
            "_usp_output_all_to_all_varlen",
            lambda value, splits, head_dim: value,
        )
    actual, captured = zing_h3._attention(
        module,
        hidden,
        rope[:3],
        history,
        rope,
        capture=True,
        sequence_splits=[3] if sp else None,
        attention_backend="fa4",
    )
    assert calls == [
        (
            torch.Size([3, 2, 8]),
            torch.Size([7, 2, 8]),
            torch.Size([7, 2, 8]),
            module.softmax_scale,
        )
    ]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert captured[0].shape == captured[1].shape == (3, 2, 8)


@pytest.mark.parametrize("result", [(), None, torch.zeros(2, 2, 8)])
def test_fa4_invalid_return_fails_without_fallback(monkeypatch, result):
    q = SimpleNamespace(is_cuda=True, shape=(3, 2, 8), device=torch.device("cpu"))
    fake = ModuleType("flash_attn.cute")
    fake.flash_attn_varlen_func = lambda **kwargs: result
    monkeypatch.setitem(sys.modules, "flash_attn.cute", fake)
    with pytest.raises(RuntimeError, match="Zing experimental FA4 returned"):
        zing_h3._fa4_attention(q, q, q, 0.125)
