# SPDX-License-Identifier: Apache-2.0
"""Experimental routes are explicit; CPU mocks do not validate FA4 numerics."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.dits import minimax_h3, zing_h3
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (
    cpu_execution_options,  # noqa: F401
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


def test_default_execution_uses_validated_combination(monkeypatch):
    monkeypatch.delenv("SGLANG_ZING_EXPERIMENTAL_ATTENTION", raising=False)
    monkeypatch.delenv("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", raising=False)
    monkeypatch.delenv("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN", raising=False)
    monkeypatch.delenv("SGLANG_ZING_EXPERIMENTAL_FA4_METADATA_REUSE", raising=False)
    model = model_fixture.__wrapped__()
    assert model.experimental_attention_backend == "fa4"
    assert model.experimental_stable_shapes is True
    assert model.experimental_compiled_adaln is True
    assert model.experimental_fa4_metadata_reuse is True


@pytest.mark.parametrize(
    "variable",
    [
        "SGLANG_ZING_EXPERIMENTAL_ATTENTION",
        "SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES",
        "SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN",
        "SGLANG_ZING_EXPERIMENTAL_FA4_METADATA_REUSE",
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


def test_fa4_passes_reused_metadata_to_kernel(monkeypatch):
    monkeypatch.setattr(zing_h3, "_fa4_current_stream", lambda device: "compute")
    q = SimpleNamespace(is_cuda=True, shape=(3, 2, 8), device=torch.device("cpu"))
    k = SimpleNamespace(shape=(7, 2, 8), device=torch.device("cpu"))
    v = SimpleNamespace(shape=(7, 2, 8), device=torch.device("cpu"))
    calls = []
    fake = ModuleType("flash_attn.cute")

    def flash(**kwargs):
        calls.append(kwargs)
        return torch.zeros(q.shape)

    fake.flash_attn_varlen_func = flash
    monkeypatch.setitem(sys.modules, "flash_attn.cute", fake)
    metadata = zing_h3._FA4ForwardMetadata()
    for _ in range(2):
        zing_h3._fa4_attention(q, k, v, 0.125, metadata=metadata)
    assert metadata.pairs_created == 1 and metadata.hits == 1
    assert calls[0]["cu_seqlens_q"] is calls[1]["cu_seqlens_q"]
    assert calls[0]["cu_seqlens_k"] is calls[1]["cu_seqlens_k"]
    assert calls[0]["cu_seqlens_q"].tolist() == [0, 3]
    assert calls[0]["cu_seqlens_k"].tolist() == [0, 7]
    metadata.clear()
    assert metadata.closed and metadata.keys() == ()


def test_fa4_metadata_reuses_actual_lengths_and_rejects_cross_stream(monkeypatch):
    stream = ["producer"]
    monkeypatch.setattr(zing_h3, "_fa4_current_stream", lambda device: stream[0])
    metadata = zing_h3._FA4ForwardMetadata()

    def tensors(q_rows, k_rows):
        return (
            torch.empty(q_rows, 2, 8),
            torch.empty(k_rows, 2, 8),
            torch.empty(k_rows, 2, 8),
        )

    first = metadata.lengths(*tensors(3, 7))
    assert [value.tolist() for value in first] == [[0, 3], [0, 7]]
    assert all(
        value.dtype == torch.int32 and value.device.type == "cpu" for value in first
    )
    assert metadata.lengths(*tensors(3, 7)) is first
    second = metadata.lengths(*tensors(3, 8))
    assert second is not first and second[1].tolist() == [0, 8]
    assert (metadata.pairs_created, metadata.tensors_created, metadata.hits) == (
        2,
        4,
        1,
    )
    stream[0] = "consumer"
    with pytest.raises(RuntimeError, match="across compute streams"):
        metadata.lengths(*tensors(3, 7))
    assert metadata.pairs_created == 2 and metadata.hits == 1
    metadata.clear()
    assert metadata.closed and metadata.keys() == ()
    with pytest.raises(RuntimeError, match="already released"):
        metadata.lengths(*tensors(3, 7))


@pytest.mark.parametrize("fail", [False, True])
@torch.no_grad()
def test_fa4_metadata_is_forward_local_and_cleared(monkeypatch, fail):
    # CPU-backed FA4 substitute exercises the real forward's refiner and main
    # call sites without claiming CUDA numerical acceptance.
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_ATTENTION", "fa4")
    monkeypatch.delenv("SGLANG_ZING_EXPERIMENTAL_FA4_METADATA_REUSE", raising=False)
    monkeypatch.setattr(zing_h3, "_fa4_current_stream", lambda device: "compute")
    model = model_fixture.__wrapped__()
    assert model.experimental_fa4_metadata_reuse
    owners = []
    original_metadata = zing_h3._FA4ForwardMetadata

    def new_metadata():
        owner = original_metadata()
        owners.append(owner)
        return owner

    monkeypatch.setattr(zing_h3, "_FA4ForwardMetadata", new_metadata)
    seen = []

    def cpu_fa4(q, k, v, scale, *, metadata=None):
        assert metadata is owners[-1]
        pair = metadata.lengths(q, k, v)
        seen.append((metadata, (q.shape[0], k.shape[0]), pair))
        if fail:
            raise RuntimeError("injected FA4 failure")
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )[0].transpose(0, 1)

    monkeypatch.setattr(zing_h3, "_fa4_attention", cpu_fa4)
    packer, inputs, visual, audio = request(model)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, audio)
    prefix = sampler._select_block(layout, 0, phase="prefix", video_t=1, audio_t=1)
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError, match="injected FA4 failure"):
                sampler._forward(prefix, ZingH3KVCache(len(model.blocks)), "prefix")
        else:
            sampler._forward(prefix, ZingH3KVCache(len(model.blocks)), "prefix")
        owner = owners[-1]
        calls = [call for call in seen if call[0] is owner]
        assert calls
        assert owner.closed and owner.keys() == ()
        assert owner.pairs_created == len({shape for _, shape, _ in calls})
        assert owner.hits == len(calls) - owner.pairs_created
        for _, shape, pair in calls:
            assert pair[0].tolist() == [0, shape[0]]
            assert pair[1].tolist() == [0, shape[1]]
    assert owners[0] is not owners[1]
    if not fail:
        assert len([call for call in seen if call[0] is owners[0]]) > 1
        assert all(
            first[2] is second[2]
            for first in seen
            for second in seen
            if first[0] is second[0] and first[1] == second[1]
        )
        assert not any(
            first[2] is second[2]
            for first in seen
            for second in seen
            if first[0] is not second[0]
        )


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
    metadata = object()

    def fake_fa4(q, k, v, scale, *, metadata=None):
        calls.append((q.shape, k.shape, v.shape, scale, metadata))
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
        fa4_metadata=metadata,
    )
    assert calls == [
        (
            torch.Size([3, 2, 8]),
            torch.Size([7, 2, 8]),
            torch.Size([7, 2, 8]),
            module.softmax_scale,
            metadata,
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


# Fused in-place QK norm (#157). CPU replaces the CUDA kernel with an in-place
# torch RMSNorm: this checks routing, aliasing and KV contracts, not numerics.


def _torch_inplace_qknorm(q, k, q_weight, k_weight, eps, *, head_dim):
    assert isinstance(eps, float)
    for x, weight in ((q, q_weight), (k, k_weight)):
        x.copy_(F.rms_norm(x, (head_dim,), weight, eps))


def _fake_kernel(monkeypatch):
    monkeypatch.setattr(zing_h3, "_fused_qk_norm_rejection", lambda *args: None)
    monkeypatch.setattr(zing_h3, "fused_inplace_qknorm", _torch_inplace_qknorm)


def _eager_only(monkeypatch):
    monkeypatch.setattr(zing_h3, "_fused_qk_norm_rejection", lambda *args: "forced")


class _CudaLike(torch.Tensor):
    """A CPU tensor that passes the device check, so later gates are reached."""

    @property
    def is_cuda(self):
        return True


def _qk(dtype=torch.bfloat16, rows=3, heads=2, dim=8):
    qkv = torch.randn(rows, 3 * heads * dim, dtype=dtype)
    q, k, _ = (x.reshape(rows, heads, dim) for x in qkv.chunk(3, -1))
    return q.as_subclass(_CudaLike), k.as_subclass(_CudaLike)


def _norms(dim=8, eps=1e-5, dtype=torch.bfloat16):
    return (
        torch.nn.RMSNorm(dim, eps=eps, dtype=dtype),
        torch.nn.RMSNorm(dim, eps=eps, dtype=dtype),
    )


def test_rejection_reasons(monkeypatch):
    q, k = _qk()
    qn, kn = _norms()
    reject = minimax_h3._fused_qk_norm_rejection
    assert reject(torch.zeros(1, 2, 8), torch.zeros(1, 2, 8), qn, kn, 8) == "not_cuda"
    monkeypatch.setattr(minimax_h3.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(minimax_h3, "can_use_fused_inplace_qknorm", lambda *a: True)
    with torch.no_grad():
        assert reject(q, k, qn, kn, 8) is None
        monkeypatch.setattr(torch.version, "hip", "6.0")
        assert reject(q, k, qn, kn, 8) == "hip"
        monkeypatch.setattr(torch.version, "hip", None)
        assert reject(*_qk(torch.float16), *_norms(dtype=torch.float16), 8) == (
            "unsupported_dtype"
        )
        assert reject(*_qk(torch.float32), qn, kn, 8) == "dtype_mismatch"
        assert reject(*_qk(rows=0), qn, kn, 8) == "empty"
        contiguous = torch.randn(3, 2, 16, dtype=torch.bfloat16)[..., :8]
        assert reject(contiguous.as_subclass(_CudaLike), k, qn, kn, 8) == (
            "unsupported_layout"
        )
        assert reject(q, q, qn, kn, 8) == "aliased_qk"
        # The kernel stores in place in parallel: partially overlapping Q/K
        # and rows overlapping within one tensor must be rejected, while the
        # row-interleaved Q/K of one fused projection (above) are accepted.
        base = torch.randn(3, 48, dtype=torch.bfloat16)
        views = [base[:, a : a + 16].reshape(3, 2, 8) for a in (0, 8, 16)]
        half, next_, disjoint = (v.as_subclass(_CudaLike) for v in views)
        assert reject(half, next_, qn, kn, 8) == "aliased_qk"
        assert reject(half, disjoint, qn, kn, 8) is None
        # Unequal head counts: K rows wider than Q's reach into the next Q row.
        store = torch.randn(160, dtype=torch.bfloat16)
        narrow = store.as_strided((3, 1, 8), (48, 8, 1), 0).as_subclass(_CudaLike)
        wide = store.as_strided((3, 2, 8), (48, 8, 1), 36).as_subclass(_CudaLike)
        assert reject(narrow, wide, qn, kn, 8) == "aliased_qk"
        fits = store.as_strided((3, 2, 8), (48, 8, 1), 24).as_subclass(_CudaLike)
        assert reject(narrow, fits, qn, kn, 8) is None
        flat = torch.randn(40, dtype=torch.bfloat16)
        rows = flat.as_strided((3, 2, 8), (8, 8, 1)).as_subclass(_CudaLike)
        assert reject(rows, k, qn, kn, 8) == "internal_overlap"
        assert reject(q, k, *_norms(eps=None), 8) == "eps"
        assert reject(q, k, qn, _norms(eps=1e-6)[1], 8) == "eps"
        wide = torch.nn.RMSNorm((2, 8), eps=1e-5, dtype=torch.bfloat16)
        assert reject(q, k, wide, kn, 8) == "weight_shape"
        monkeypatch.setattr(
            minimax_h3, "can_use_fused_inplace_qknorm", lambda *a: False
        )
        assert reject(q, k, qn, kn, 8) == "kernel_unavailable"
    monkeypatch.setattr(minimax_h3, "can_use_fused_inplace_qknorm", lambda *a: True)
    with torch.enable_grad():
        assert reject(q, k, qn, kn, 8) == "grad"
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert reject(q, k, qn, kn, 8) == "compiling"


@torch.no_grad()
def test_cpu_falls_back_to_eager_and_records_reason():
    zing_h3.FUSED_QK_NORM_ROUTES.clear()
    model = model_fixture.__wrapped__()
    packer, inputs, video, audio = request(model)
    ZingH3Sampler(model, packer, sigma_points=3).sample(inputs, video, audio)
    assert set(zing_h3.FUSED_QK_NORM_ROUTES) == {"not_cuda"}


@pytest.mark.parametrize(
    "cache_kwargs",
    [
        {},  # default window: raw K store, block-relative RoPE
        {"window_chunks": 2, "sink_chunks": 0},
        {"window_chunks": None, "sink_chunks": 0, "rope_position_mode": "absolute"},
    ],
)
@torch.no_grad()
def test_fused_route_matches_eager_and_captures_independent_k(
    monkeypatch, cache_kwargs
):
    packer, inputs, video, audio = request(model_fixture.__wrapped__())
    kwargs = dict(sigma_points=3, **cache_kwargs)
    commits = {False: [], True: []}
    original_update = ZingH3KVCache.update

    def update(self, positions, block_idx, phase, frames, layers=None):
        if layers is not None:
            snapshot = [(k.clone(), v.clone()) for k, v in layers]
            if enabled and self.raw_store:
                # Staged raw K is the store's own scratch tail: the commit
                # copies nothing and no QKV projection is kept alive.
                for index, (k, _) in enumerate(layers):
                    assert k.data_ptr() == (
                        self._stores[index][0][self._length].data_ptr()
                    )
            commits[enabled].append((phase, snapshot))
        return original_update(self, positions, block_idx, phase, frames, layers)

    monkeypatch.setattr(ZingH3KVCache, "update", update)
    outputs = {}
    for enabled in (False, True):
        (_fake_kernel if enabled else _eager_only)(monkeypatch)
        sampler = ZingH3Sampler(model_fixture.__wrapped__(), packer, **kwargs)
        outputs[enabled] = sampler.sample(inputs, video, audio)
    assert all(torch.equal(a, b) for a, b in zip(outputs[True], outputs[False]))
    assert len(commits[True]) == len(commits[False]) > 0
    for (phase, layers), (expected_phase, expected) in zip(
        commits[True], commits[False]
    ):
        assert phase == expected_phase
        for (k, v), (ek, ev) in zip(layers, expected):
            assert torch.equal(k, ek) and torch.equal(v, ev)


@torch.no_grad()
def test_uncached_capture_does_not_keep_qkv_alive(monkeypatch):
    # The replicated SP prefix captures without a cache append; its K is
    # held until the commit and must not borrow the fused QKV projection.
    _fake_kernel(monkeypatch)
    attn = model_fixture.__wrapped__().blocks[0].attn
    hidden = torch.randn(5, 16, dtype=attn.qkv_proj.weight.dtype)
    projections = []
    original = attn.qkv_proj.forward

    def spy(x):
        result = original(x)
        projections.append(result[0])
        return result

    monkeypatch.setattr(attn.qkv_proj, "forward", spy)
    fused, raw = zing_h3._attention(attn, hidden, capture=True)
    _eager_only(monkeypatch)
    eager, expected = zing_h3._attention(attn, hidden, capture=True)
    qkv = projections[0]
    for tensor in raw:
        assert tensor.untyped_storage().data_ptr() != (qkv.untyped_storage().data_ptr())
    assert torch.equal(fused, eager)
    assert all(torch.equal(a, b) for a, b in zip(raw, expected))
    # V is untouched by the norm.
    v = qkv.chunk(3, -1)[2].reshape_as(raw[1])
    assert torch.equal(raw[1], v)
