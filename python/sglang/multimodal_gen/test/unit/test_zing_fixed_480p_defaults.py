# SPDX-License-Identifier: Apache-2.0
"""Default admission is spatial, and segmented attention keeps device lengths."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import sglang.multimodal_gen.runtime.models.dits.zing_h3 as model_module
import sglang.multimodal_gen.runtime.models.dits.zing_h3_rope as rope_module


def _host(height=15, width=26, frames=2, role=10):
    grid = torch.cartesian_prod(
        torch.arange(frames), torch.arange(height), torch.arange(width)
    ).double()
    return SimpleNamespace(
        position_ids=grid, row_role=torch.full((grid.shape[0],), role), derived={}
    )


def _admit(host, **overrides):
    args = dict(sp_size=2, heads=56, head_dim=128, backend="fa4", audio=False)
    args.update(overrides)
    return rope_module.validated_fixed_480p(host, torch.device("cuda", 0), **args)


def test_480p_admission_uses_spatial_grid_not_query_count(monkeypatch):
    monkeypatch.setattr(rope_module, "_validated_fixed_device", lambda d: True)
    host = _host()
    assert _admit(host)
    # Same 780 total Q rows, different spatial grid.
    assert not _admit(_host(10, 26, 3))
    assert not _admit(_host(role=4))
    assert not _admit(host, sp_size=1)
    assert not _admit(host, heads=28)
    assert not _admit(host, audio=True)
    assert not _admit(host, backend="sdpa")
    # The existing immutable host object retains only a bool, no tensor graph.
    monkeypatch.setattr(
        torch, "unique", lambda *a, **kw: pytest.fail("Repeated geometry allocation")
    )
    assert _admit(host)


@pytest.mark.parametrize(
    "capability,sms,version,expected",
    [
        ((10, 3), 148, "4.0.0b19", True),
        ((10, 0), 148, "4.0.0b19", False),
        ((10, 3), 132, "4.0.0b19", False),
        ((10, 3), 148, "4.0.0b20", False),
    ],
)
def test_default_hardware_and_fa4_boundary(
    monkeypatch, capability, sms, version, expected
):
    rope_module._validated_fixed_device.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: capability)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: SimpleNamespace(multi_processor_count=sms),
    )
    monkeypatch.setattr(rope_module.importlib.metadata, "version", lambda n: version)
    try:
        assert rope_module._validated_fixed_device(torch.device("cuda", 0)) is expected
        assert not rope_module._validated_fixed_device(torch.device("cpu"))
    finally:
        rope_module._validated_fixed_device.cache_clear()


def test_explicit_rope_disable_does_not_import_or_allocate(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_FUSED_CURRENT_ROPE_KV", "0")
    assert (
        rope_module.apply_current_rope_kv(
            None, None, None, None, None, 0, None, default=True
        )
        is None
    )


@pytest.mark.parametrize(
    "qrows,flag,fixed",
    [(390, None, True), (780, None, True), (1170, None, False), (780, "0", False)],
)
def test_segmented_fa4_keeps_q1170_varlen_and_live_device_bound(
    monkeypatch, qrows, flag, fixed
):
    if flag is None:
        monkeypatch.delenv("SGLANG_ZING_FA4_FIXED", raising=False)
    else:
        monkeypatch.setenv("SGLANG_ZING_FA4_FIXED", flag)
    calls = []
    fake = ModuleType("flash_attn.cute.interface")

    def fwd(q, k, v, **kwargs):
        calls.append(kwargs)
        # A tiny CPU reference honors the live device tensor, never capacity.
        return (torch.ones_like(q) * int(kwargs["cu_seqlens_k"][1]),)

    fake._flash_attn_fwd = fwd
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", fake)
    monkeypatch.setattr(
        model_module,
        "_fa4_attention",
        lambda q, k, v, scale, metadata: torch.full_like(q, -1),
    )
    q = torch.zeros(qrows, 1, 2, dtype=torch.bfloat16)
    k = v = torch.zeros(4096, 1, 2, dtype=q.dtype)
    lengths = torch.tensor([0, 17], dtype=torch.int32)
    metadata = model_module._FixedFA4Metadata(
        {"cu_k.0": lengths, "cu_q.0": torch.tensor([0, qrows], dtype=torch.int32)}, 0
    )
    result = model_module._fixed_fa4_attention(q, k, v, 0.5, metadata, optimized=True)
    if fixed:
        assert calls[-1]["cu_seqlens_k"] is lengths
        assert calls[-1]["num_splits"] == 1 and calls[-1]["max_seqlen_k"] == 4096
        assert torch.all(result == 17)
        lengths[1] = 29
        assert torch.all(
            model_module._fixed_fa4_attention(q, k, v, 0.5, metadata, optimized=True)
            == 29
        )
    else:
        assert not calls and torch.all(result == -1)


def test_new_route_switches_participate_in_graph_keys():
    assert {"SGLANG_ZING_FUSED_CURRENT_ROPE_KV", "SGLANG_ZING_FA4_FIXED"} <= set(
        model_module._GRAPH_ROUTE_ENVS
    )
