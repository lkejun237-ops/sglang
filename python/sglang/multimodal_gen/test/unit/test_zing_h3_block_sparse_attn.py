# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Zing H3 4D Block-Sparse Attention and Single-Router-per-Chunk runtime."""

import ast
import importlib.util
import math
import os
import sys
from pathlib import Path

import torch

# Cleanly import zing_h3_sparse_attn without depending on CUDA/Triton
MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "runtime"
    / "models"
    / "dits"
    / "zing_h3_sparse_attn.py"
)
spec = importlib.util.spec_from_file_location("zing_h3_sparse_attn", str(MODULE_PATH))
sparse_mod = importlib.util.module_from_spec(spec)
sys.modules["zing_h3_sparse_attn"] = sparse_mod
spec.loader.exec_module(sparse_mod)


def test_native_attention_inputs_from_layer_and_parity_workspace():
    workspace = torch.arange(2 * 3 * 4 * 6 * 8, dtype=torch.int32).reshape(
        2, 3, 4, 6, 8
    )
    normalize = sparse_mod._native_attention_input
    first = workspace[0, 0]
    assert normalize(first) is first
    assert normalize(None) is None
    for view in (workspace[1, 2], workspace[1, 2, :, :3], first.transpose(1, 2)):
        actual = normalize(view)
        assert actual.storage_offset() == 0
        assert actual.is_contiguous()
        assert torch.equal(actual, view)


def test_sparse_mode_switching_and_envs():
    """Verify mode getters/setters and environment variable synchronization."""
    orig_mode = os.environ.get("SGLANG_ZING_SPARSE_ATTN_MODE")
    orig_scale = os.environ.get("SGLANG_ZING_SPARSE_BUDGET_SCALE")
    orig_chunk = os.environ.get("SGLANG_ZING_SPARSE_MIN_CHUNK")

    try:
        # Default mode is dense if not set
        if "SGLANG_ZING_SPARSE_ATTN_MODE" in os.environ:
            del os.environ["SGLANG_ZING_SPARSE_ATTN_MODE"]
        if "SGLANG_ZING_SPARSE_BUDGET_SCALE" in os.environ:
            del os.environ["SGLANG_ZING_SPARSE_BUDGET_SCALE"]
        assert sparse_mod.get_sparse_mode() == "dense"

        # Test switching modes
        sparse_mod.set_sparse_mode("sparse_095")
        assert sparse_mod.get_sparse_mode() == "sparse_095"
        assert os.environ["SGLANG_ZING_SPARSE_ATTN_MODE"] == "sparse_095"
        assert sparse_mod.get_sparse_budget_tau() == "0.95"

        sparse_mod.set_sparse_mode("sparse_090")
        assert sparse_mod.get_sparse_mode() == "sparse_090"
        assert os.environ["SGLANG_ZING_SPARSE_ATTN_MODE"] == "sparse_090"
        assert sparse_mod.get_sparse_budget_tau() == "0.90"

        # Test SGLANG_ZING_SPARSE_BUDGET_SCALE with dual_query_topk mode
        sparse_mod.set_sparse_mode("dual_query_topk")
        os.environ["SGLANG_ZING_SPARSE_BUDGET_SCALE"] = "0.90"
        assert sparse_mod.get_sparse_budget_tau() == "0.90"
        assert sparse_mod.read_sparse_state()["tau"] == "0.90"

        os.environ["SGLANG_ZING_SPARSE_BUDGET_SCALE"] = "0.95"
        assert sparse_mod.get_sparse_budget_tau() == "0.95"
        assert sparse_mod.read_sparse_state()["tau"] == "0.95"
        del os.environ["SGLANG_ZING_SPARSE_BUDGET_SCALE"]

        sparse_mod.set_sparse_mode("dense")
        assert sparse_mod.get_sparse_mode() == "dense"

        # Test min sparse chunk
        if "SGLANG_ZING_SPARSE_MIN_CHUNK" in os.environ:
            del os.environ["SGLANG_ZING_SPARSE_MIN_CHUNK"]
        assert sparse_mod.get_min_sparse_chunk() == 5

        sparse_mod.set_min_sparse_chunk(3)
        assert sparse_mod.get_min_sparse_chunk() == 3
        assert os.environ["SGLANG_ZING_SPARSE_MIN_CHUNK"] == "3"

        sparse_mod.set_min_sparse_chunk(5)
        assert sparse_mod.get_min_sparse_chunk() == 5

    finally:
        if orig_mode is not None:
            os.environ["SGLANG_ZING_SPARSE_ATTN_MODE"] = orig_mode
        elif "SGLANG_ZING_SPARSE_ATTN_MODE" in os.environ:
            del os.environ["SGLANG_ZING_SPARSE_ATTN_MODE"]

        if orig_scale is not None:
            os.environ["SGLANG_ZING_SPARSE_BUDGET_SCALE"] = orig_scale
        elif "SGLANG_ZING_SPARSE_BUDGET_SCALE" in os.environ:
            del os.environ["SGLANG_ZING_SPARSE_BUDGET_SCALE"]

        if orig_chunk is not None:
            os.environ["SGLANG_ZING_SPARSE_MIN_CHUNK"] = orig_chunk
        elif "SGLANG_ZING_SPARSE_MIN_CHUNK" in os.environ:
            del os.environ["SGLANG_ZING_SPARSE_MIN_CHUNK"]


def test_graph_route_envs_inclusion():
    """Verify _GRAPH_ROUTE_ENVS in zing_h3.py contains sparse attention control variables."""
    zing_h3_path = MODULE_PATH.parent / "zing_h3.py"
    assert zing_h3_path.exists(), f"File {zing_h3_path} does not exist"

    tree = ast.parse(zing_h3_path.read_text(encoding="utf-8"))
    route_envs = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "_GRAPH_ROUTE_ENVS":
                    if isinstance(stmt.value, ast.Tuple):
                        route_envs = [
                            elt.value
                            for elt in stmt.value.elts
                            if isinstance(elt, ast.Constant)
                        ]

    assert route_envs is not None, "_GRAPH_ROUTE_ENVS not found in zing_h3.py"
    assert "SGLANG_ZING_SPARSE_ATTN_MODE" in route_envs
    assert "SGLANG_ZING_SPARSE_BUDGET_SCALE" in route_envs
    assert "SGLANG_ZING_SPARSE_MIN_CHUNK" in route_envs


def test_token_per_frame_dynamic_inference():
    """Verify dynamic token-per-frame inference for 480p and 736p."""
    # 480p cases (390 tokens per frame)
    assert sparse_mod.infer_tokens_per_frame(390) == 390
    assert sparse_mod.infer_tokens_per_frame(780) == 390
    assert sparse_mod.infer_tokens_per_frame(1170) == 390

    # 736p cases (920 tokens per frame)
    assert sparse_mod.infer_tokens_per_frame(920) == 920
    assert sparse_mod.infer_tokens_per_frame(1840) == 920
    assert sparse_mod.infer_tokens_per_frame(2760) == 920

    # General fallback
    assert sparse_mod.infer_tokens_per_frame(100) == 50
    assert sparse_mod.infer_tokens_per_frame(500) == 250


def test_budget_table_resolution_proportional_scaling():
    """Verify budget scaling from 480p (M_cand_ref=434) to 736p (M_cand > 460)."""
    # 480p unscaled table
    tab_480 = sparse_mod.get_zing_h3_head_budget_table(
        "0.95", head_start=0, num_heads=56, m_cand=434
    )
    assert len(tab_480) == 50
    assert len(tab_480[0]) == 56

    # 736p scaled table (e.g. M_cand = 1024)
    m_cand_736 = 1024
    tab_736 = sparse_mod.get_zing_h3_head_budget_table(
        "0.95", head_start=0, num_heads=56, m_cand=m_cand_736
    )
    assert len(tab_736) == 50
    assert len(tab_736[0]) == 56

    scale_factor = m_cand_736 / 434.0
    for l_idx in range(50):
        for h_idx in range(56):
            k_480 = tab_480[l_idx][h_idx]
            k_736 = tab_736[l_idx][h_idx]
            # Must be clamped between 8 and m_cand_736
            assert 8 <= k_736 <= m_cand_736
            expected = min(m_cand_736, max(8, int(math.ceil(k_480 * scale_factor))))
            assert k_736 == expected, (
                f"Mismatch at layer {l_idx} head {h_idx}: got {k_736}, expected {expected}"
            )


def test_sequence_parallel_head_sharding():
    """Verify Ulysses head sharding for SP=1 (56 heads) and SP=2 (28 heads per rank)."""
    full_table_95 = sparse_mod.get_zing_h3_head_budget_table(
        "0.95", head_start=0, num_heads=56
    )
    full_table_90 = sparse_mod.get_zing_h3_head_budget_table(
        "0.90", head_start=0, num_heads=56
    )

    # SP=2: rank 0 gets heads [0:28], rank 1 gets heads [28:56]
    rank0_95 = sparse_mod.get_zing_h3_head_budget_table(
        "0.95", head_start=0, num_heads=28
    )
    rank1_95 = sparse_mod.get_zing_h3_head_budget_table(
        "0.95", head_start=28, num_heads=28
    )

    assert len(rank0_95[0]) == 28
    assert len(rank1_95[0]) == 28

    for l_idx in range(50):
        assert rank0_95[l_idx] + rank1_95[l_idx] == full_table_95[l_idx]

    rank0_90 = sparse_mod.get_zing_h3_head_budget_table(
        "0.90", head_start=0, num_heads=28
    )
    rank1_90 = sparse_mod.get_zing_h3_head_budget_table(
        "0.90", head_start=28, num_heads=28
    )
    for l_idx in range(50):
        assert rank0_90[l_idx] + rank1_90[l_idx] == full_table_90[l_idx]


def test_static_router_workspace_lifecycle():
    """Verify _StaticRouterWorkspace allocation, flat buffer sizing, and expansion."""
    dev = torch.device("cpu")
    num_heads = 28
    ws = sparse_mod._get_workspace(dev, num_heads=num_heads, req_m_q=16, req_m_cand=512)

    assert ws.num_heads == num_heads
    assert ws.max_m_q >= 16
    assert ws.max_m_cand >= 512
    # Verify flat buffer allocations
    assert ws.q_sub_flat.numel() == num_heads * (2 * ws.max_m_q * 4) * 128
    assert ws.k_sub_flat.numel() == num_heads * (ws.max_m_cand * 4) * 128
    assert ws.q_norm_flat.numel() == num_heads * (2 * ws.max_m_q * 4)
    assert ws.k_norm_flat.numel() == num_heads * (ws.max_m_cand * 4)
    assert ws.blk_scores_flat.numel() == 2 * num_heads * ws.max_m_q * ws.max_m_cand
    assert ws.top_val_flat.numel() == 2 * num_heads * ws.max_m_q * ws.max_m_cand
    assert ws.top_idx_flat.numel() == 2 * num_heads * ws.max_m_q * ws.max_m_cand
    assert ws.cand_idx_src.shape == (50, num_heads, ws.max_m_q, ws.max_m_cand)
    assert ws.cand_idx_s0.shape == (50, num_heads, ws.max_m_q, ws.max_m_cand)
    assert ws.cand_idx_s1.shape == (2, 50, num_heads, ws.max_m_q, ws.max_m_cand)

    # Request larger M_cand (e.g. for 736p with 1100 candidate blocks) -> triggers reallocation
    ws_expanded = sparse_mod._get_workspace(
        dev, num_heads=num_heads, req_m_q=16, req_m_cand=1100
    )
    assert ws_expanded.max_m_cand >= 1100
    assert (
        ws_expanded.k_sub_flat.numel() == num_heads * (ws_expanded.max_m_cand * 4) * 128
    )
    assert ws_expanded.k_norm_flat.numel() == num_heads * (ws_expanded.max_m_cand * 4)
    assert (
        ws_expanded.blk_scores_flat.numel()
        == 2 * num_heads * ws_expanded.max_m_q * ws_expanded.max_m_cand
    )


def test_kv_exemption_and_dense_guardrails():
    """Verify accuracy guardrails: layer 0 dense, warmup chunks dense, tiny candidate pool dense."""
    q = torch.randn(780, 28, 128)
    k = torch.randn(3000, 28, 128)
    v = torch.randn(3000, 28, 128)
    scale = 1.0 / math.sqrt(128)

    # 1. Warmup chunk exemption (chunk_id = 1 < min_chunk = 3)
    spec_warmup = sparse_mod.SparseBranchSpec(
        mode="sparse_095",
        tau="0.95",
        case_id="unit_test",
        chunk_id=1,
        b_phase="active",
        step_idx=0,
        step_1000=0,
        q_start=0,
        q_stop=780,
        L_q=780,
        L_k=3000,
        s_cur_chunk=2440,
        M_q=13,
        M_k=47,
        M_cand=40,
        M_cur=7,
        M_prefix=2,
    )
    # Should execute dense branch without failing
    out_warmup = sparse_mod.execute_branch_attention(
        orig_fixed_fa4=None,
        q=q,
        k=k,
        v=v,
        scale=scale,
        metadata=None,
        optimized=False,
        layer_idx=5,
        spec=spec_warmup,
        ulysses_rank=0,
        fixed_tensors={"cu_k.0": torch.tensor([0, 3000], dtype=torch.int32)},
        branch=0,
    )
    assert out_warmup.shape == q.shape
    assert torch.isfinite(out_warmup).all()

    # 2. Layer 0 exemption
    spec_layer0 = sparse_mod.SparseBranchSpec(
        mode="sparse_095",
        tau="0.95",
        case_id="unit_test",
        chunk_id=10,
        b_phase="active",
        step_idx=0,
        step_1000=0,
        q_start=0,
        q_stop=780,
        L_q=780,
        L_k=3000,
        s_cur_chunk=2440,
        M_q=13,
        M_k=47,
        M_cand=40,
        M_cur=7,
        M_prefix=2,
    )
    out_layer0 = sparse_mod.execute_branch_attention(
        orig_fixed_fa4=None,
        q=q,
        k=k,
        v=v,
        scale=scale,
        metadata=None,
        optimized=False,
        layer_idx=0,  # Layer 0 must run dense
        spec=spec_layer0,
        ulysses_rank=0,
        fixed_tensors={"cu_k.0": torch.tensor([0, 3000], dtype=torch.int32)},
        branch=0,
    )
    assert out_layer0.shape == q.shape
    assert torch.isfinite(out_layer0).all()


def test_exact_sdpa_reference():
    """Verify PyTorch SDPA reference implementation for 4D block sparse attention."""
    L_q, L_k, H, D = 128, 256, 4, 64
    BLOCK_SIZE = 64
    M_k = L_k // BLOCK_SIZE

    q = torch.randn(L_q, H, D)
    k = torch.randn(L_k, H, D)
    v = torch.randn(L_k, H, D)
    scale = 1.0 / math.sqrt(D)

    # 1 candidate block selected per head
    cand_cnt = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
    cand_idx = torch.tensor([[0], [1], [0], [1]], dtype=torch.int32)
    head_order = torch.arange(H, dtype=torch.int32)

    out = sparse_mod._exact_block_sparse_sdpa_reference(
        q,
        k,
        v,
        scale,
        L_k,
        M_cand=2,
        M_k=M_k,
        cand_cnt=cand_cnt,
        cand_idx=cand_idx,
        head_order=head_order,
    )
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_execute_fixed_block_sparse_branches_dense_parity():
    """Verify execute_fixed_block_sparse_branches in dense mode matches exact dense calculation."""
    q = torch.randn(1560, 28, 128)
    keys = torch.randn(4000, 28, 128)
    values = torch.randn(4000, 28, 128)
    scale = 1.0 / math.sqrt(128)

    class MockFixedKV:
        ends = [390, 780, 1560]
        write_start = 2440

    fixed = MockFixedKV()
    fixed_tensors = {
        "cu_k.0": torch.tensor([0, 2830], dtype=torch.int32),
        "cu_k.1": torch.tensor([0, 3220], dtype=torch.int32),
        "cu_k.2": torch.tensor([0, 4000], dtype=torch.int32),
    }

    sparse_mod.set_sparse_mode("dense")
    results = sparse_mod.execute_fixed_block_sparse_branches(
        q, keys, values, scale, index=1, fixed=fixed, fixed_tensors=fixed_tensors
    )

    assert len(results) == 3
    assert results[0].shape == (390, 28, 128)
    assert results[1].shape == (390, 28, 128)
    assert results[2].shape == (780, 28, 128)
    for res in results:
        assert torch.isfinite(res).all()


def test_cosine_gated_stale_scene_vs_character_retention():
    """Verify cosine-gated visual reference boost and stale-scene decay."""
    H, D = 4, 128
    L_q, M_q = 64, 1
    M_cand = 20
    M_prefix = 6
    L_k = M_cand * 64

    q_first = torch.zeros(L_q, H, D)
    q_first[:, :, 0] = 1.0
    q_second = q_first.clone()

    k = torch.randn(L_k, H, D) * 0.05
    # Character reference blocks (blocks 2, 3): high cosine along feature 0
    k[2 * 64 : 4 * 64, :, 0] = 1.0
    # Stale reference blocks (blocks 4, 5): negative cosine along feature 0
    k[4 * 64 : 6 * 64, :, 0] = -1.0
    # Sink blocks (blocks 6, 7): negative cosine along feature 0
    k[6 * 64 : 8 * 64, :, 0] = -1.0

    role4_spans = ((2, 6),)  # blocks 2..5 are KIND_REF_VIS
    sink_span = (6, 8)  # blocks 6..7 are KIND_SINK
    win_src_spans = ((8, 14),)  # blocks 8..13 are KIND_WIN_SRC
    blk_kind_sig = (M_cand, role4_spans, sink_span, win_src_spans)

    m_src, m_s0, m_s1 = sparse_mod.fused_route_and_build_masks(
        q_first=q_first,
        q_second=q_second,
        k=k,
        layer_idx=10,
        tau="0.95",
        L_q=L_q,
        M_q=M_q,
        M_cand=M_cand,
        M_prefix=M_prefix,
        safety_scale=1.0,
        build_src=True,
        L_k_src=L_k,
        M_k_src=M_cand,
        M_cur_src=1,
        build_s0=True,
        L_k_s0=L_k,
        M_k_s0=M_cand,
        M_cur_s0=1,
        build_s1=True,
        s_evict=0,
        L_k_s1=L_k,
        M_k_s1=M_cand,
        M_cand_s1=M_cand,
        M_cur_s1=1,
        head_start=0,
        num_heads=H,
        blk_kind_sig=blk_kind_sig,
    )

    ws = sparse_mod._get_workspace(
        q_first.device, num_heads=H, req_m_q=M_q, req_m_cand=M_cand
    )
    s0_cal = ws.blk_scores_flat[: H * M_q * M_cand].view(H, M_q, M_cand)

    # 1. Cosine-gated character reference blocks get boosted scores
    assert s0_cal[0, 0, 2] > s0_cal[0, 0, 4]
    assert s0_cal[0, 0, 3] > s0_cal[0, 0, 6]

    # 2. High-cosine character reference blocks retained in active candidate indices
    cand_head0 = m_s0.cand_idx[0, 0].tolist()
    assert 2 in cand_head0 and 3 in cand_head0
    # Stale reference and sink blocks with negative cosine suppressed
    assert 4 not in cand_head0 and 6 not in cand_head0


def test_decoupled_source_branch_and_win_src_geometry_recall():
    """Verify decoupled source branch routing and win_src 0.55 geometry weight recall."""
    H, D = 4, 128
    L_q, M_q = 64, 1
    M_cand = 20
    M_prefix = 6
    L_k = M_cand * 64

    # Step 0 query (q_first): affinity to feature 1
    q_first = torch.zeros(L_q, H, D)
    q_first[:, :, 1] = 1.0

    # Step 1 query (q_second): affinity to feature 2
    q_second = torch.zeros(L_q, H, D)
    q_second[:, :, 2] = 1.0

    # Key features:
    # Block 9: KIND_WIN_SRC (tokens 9*64 .. 10*64) has affinity to feature 1 (s0 geometry)
    # Block 15: KIND_WIN_FIN (tokens 15*64 .. 16*64) has the SAME affinity to feature 1
    k = torch.randn(L_k, H, D) * 0.05
    # Zero out feature 2 so neither block has random noise advantage on step 1
    k[:, :, 2] = 0.0
    k[9 * 64 : 10 * 64, :, 1] = 1.0
    k[15 * 64 : 16 * 64, :, 1] = 1.0

    role4_spans = ((2, 6),)
    sink_span = (6, 8)
    win_src_spans = ((8, 14),)
    blk_kind_sig = (M_cand, role4_spans, sink_span, win_src_spans)

    m_src, m_s0, m_s1 = sparse_mod.fused_route_and_build_masks(
        q_first=q_first,
        q_second=q_second,
        k=k,
        layer_idx=10,
        tau="0.95",
        L_q=L_q,
        M_q=M_q,
        M_cand=M_cand,
        M_prefix=M_prefix,
        safety_scale=1.0,
        build_src=True,
        L_k_src=L_k,
        M_k_src=M_cand,
        M_cur_src=1,
        build_s0=True,
        L_k_s0=L_k,
        M_k_s0=M_cand,
        M_cur_s0=1,
        build_s1=True,
        s_evict=0,
        L_k_s1=L_k,
        M_k_s1=M_cand,
        M_cand_s1=M_cand,
        M_cur_s1=1,
        head_start=0,
        num_heads=H,
        blk_kind_sig=blk_kind_sig,
    )

    ws = sparse_mod._get_workspace(
        q_first.device, num_heads=H, req_m_q=M_q, req_m_cand=M_cand
    )
    s0_cal = ws.blk_scores_flat[: H * M_q * M_cand].view(H, M_q, M_cand)
    s_act = ws.blk_scores_flat[H * M_q * M_cand : 2 * H * M_q * M_cand].view(
        H, M_q, M_cand
    )

    # In s0_cal, both block 9 and 15 have equal base affinity
    assert torch.isclose(s0_cal[0, 0, 9], s0_cal[0, 0, 15], atol=1e-2)

    # In s_act, block 9 (KIND_WIN_SRC) receives 0.55 weight vs block 15 receiving 0.15 weight
    assert s_act[0, 0, 9] > s_act[0, 0, 15], (
        "Block 9 (win_src) must have higher fused score than block 15"
    )


def test_non64_aligned_eviction_shift_uniqueness():
    """Verify non-64 aligned eviction shift calculation and candidate index uniqueness."""
    H, D = 4, 128
    L_q, M_q = 64, 1
    M_cand = 100
    M_prefix = 6
    L_k = M_cand * 64

    q_first = torch.randn(L_q, H, D)
    q_second = torch.randn(L_q, H, D)
    k = torch.randn(L_k, H, D)

    # Non-64 aligned eviction parameters (e.g. L_evict = 2440 tokens)
    L_evict = 2440
    s_evict = 640
    M_cand_s1 = 60

    m_src, m_s0, m_s1 = sparse_mod.fused_route_and_build_masks(
        q_first=q_first,
        q_second=q_second,
        k=k,
        layer_idx=10,
        tau="0.95",
        L_q=L_q,
        M_q=M_q,
        M_cand=M_cand,
        M_prefix=M_prefix,
        safety_scale=1.0,
        build_src=True,
        L_k_src=L_k,
        M_k_src=M_cand,
        M_cur_src=1,
        build_s0=True,
        L_k_s0=L_k,
        M_k_s0=M_cand,
        M_cur_s0=1,
        build_s1=True,
        s_evict=s_evict,
        L_k_s1=M_cand_s1 * 64,
        M_k_s1=M_cand_s1,
        M_cand_s1=M_cand_s1,
        M_cur_s1=1,
        head_start=0,
        num_heads=H,
        L_evict=L_evict,
    )

    assert m_s1 is not None
    shift_s1 = (L_evict + 32) // 64
    assert shift_s1 == 38

    for h in range(H):
        for m in range(M_q):
            total_k = int(m_s1.cand_cnt[h].item())
            row_indices = m_s1.cand_idx[h, m, :total_k].tolist()
            # Verify strict uniqueness (no duplicate zeros or duplicated blocks)
            assert len(row_indices) == len(set(row_indices)), (
                f"Head {h} row {m} has duplicate indices: {row_indices}"
            )
            # Verify all candidate indices are valid and non-negative
            for idx in row_indices:
                assert idx >= 0


def test_midstream_prefix_joint_reference_switch():
    """Verify product-reference-switch 6-branch PREFIX_PHASES & 5-branch CACHED_PREFIX_PHASES."""
    from types import SimpleNamespace

    # Build synthetic 1D Qwen text positions (256 rows: (0,0,0)..(255,0,0)) + 2D VAE visual patches
    # (2 images of 16x16 = 256 patches each, including (0,0) and diagonals (h,h))
    text_pos = torch.stack(
        [
            torch.arange(256, dtype=torch.int64),
            torch.zeros(256, dtype=torch.int64),
            torch.zeros(256, dtype=torch.int64),
        ],
        dim=1,
    )
    grid_h, grid_w = torch.meshgrid(
        torch.arange(16, dtype=torch.int64),
        torch.arange(16, dtype=torch.int64),
        indexing="ij",
    )
    img1_pos = torch.stack(
        [
            torch.full((256,), 300, dtype=torch.int64),
            grid_h.reshape(-1),
            grid_w.reshape(-1),
        ],
        dim=1,
    )
    img2_pos = torch.stack(
        [
            torch.full((256,), 301, dtype=torch.int64),
            grid_h.reshape(-1),
            grid_w.reshape(-1),
        ],
        dim=1,
    )
    full_pos_2img = torch.cat(
        [text_pos, img1_pos, img2_pos], dim=0
    )  # 768 rows = 12 blocks
    inferred = sparse_mod._infer_role4_spans_from_positions(full_pos_2img)
    assert inferred == ((256, 768),), f"Expected ((256, 768),), got {inferred}"

    # 1. 6-branch mid-stream switch to 1 reference image (256 text + 256 img1 = 512 rows = 8 blocks)
    pos_1img = torch.cat([text_pos, img1_pos], dim=0)
    role_6branch = torch.cat(
        [
            torch.zeros(256, dtype=torch.int64),
            torch.full((256,), 4, dtype=torch.int64),
            torch.full((192 + 384 + 384 + 384 + 384,), 10, dtype=torch.int64),
        ]
    )
    pref_1img = SimpleNamespace(position_signature="sig_1img", host_positions=pos_1img)
    shadow_1img = SimpleNamespace(prefix=pref_1img)
    kv_plan_6 = SimpleNamespace(
        write_start=0,
        ends=[512, 704, 1088, 1472, 1856, 2240],
    )
    packed_6 = {
        "_zing_host_layout": SimpleNamespace(row_role=role_6branch),
        "_zing_joint_phases": (
            "prefix",
            "history",
            "history",
            "history",
            "source",
            "active",
        ),
    }
    res_6 = sparse_mod.attach_sparse_specs_to_plan(
        None,
        [],
        SimpleNamespace(sink_chunks=3, retains=lambda u, cur: True),
        packed_6,
        "joint",
        kv_plan_6,
        {"mode": "sparse_095", "case_id": "product-reference-switch"},
        shadow=shadow_1img,
    )
    assert res_6 == {}
    assert sparse_mod._PREFIX_ROLE4_BY_SIG["sig_1img"] == ((256, 512),)

    # 2. Cold 5-branch CACHED_PREFIX_PHASES restore of 2-image prefix (768 rows = 12 blocks)
    pref_2img = SimpleNamespace(
        offset=0,
        positions=full_pos_2img,
        position_signature="sig_2img_cold",
        host_positions=full_pos_2img,
    )
    shadow_2img = SimpleNamespace(prefix=pref_2img)
    kv_plan_5 = SimpleNamespace(write_start=768, ends=[192, 576, 960, 1344, 1728])
    packed_5 = {
        "_zing_joint_phases": ("history", "history", "history", "source", "active"),
    }
    res_5 = sparse_mod.attach_sparse_specs_to_plan(
        None,
        [],
        SimpleNamespace(sink_chunks=3, retains=lambda u, cur: True),
        packed_5,
        "joint",
        kv_plan_5,
        {"mode": "sparse_095", "case_id": "product-reference-switch"},
        shadow=shadow_2img,
    )
    assert res_5 == {}
    assert sparse_mod._PREFIX_ROLE4_BY_SIG["sig_2img_cold"] == ((256, 768),)

    # 3. Steady-state 3-branch chunk u=6 after 2-image restore -> verify exact 5-segment block_kinds_sig
    pending_src_u5 = SimpleNamespace(
        offset=1920,
        positions=torch.zeros((384, 3), dtype=torch.int64),
        block=11,  # u = (11 - 1) // 2 = 5
    )

    def _mock_history():
        return [
            pref_2img,  # prefix: 0..768 (blks 0..11)
            SimpleNamespace(
                offset=768,
                positions=torch.zeros((384, 3), dtype=torch.int64),
                temporal_chunk_id=0,
                rope_parts=((2, 384),),
            ),  # sink u=0 (single-part from PrefixJointGraphs): 768..1152 (blks 12..17)
            SimpleNamespace(
                offset=1152,
                positions=torch.zeros((768, 3), dtype=torch.int64),
                temporal_chunk_id=4,
                rope_parts=((9, 384), (10, 384)),
            ),  # committed win u=4: src 1152..1536 (blks 18..23), fin 1536..1920 (blks 24..29)
            pending_src_u5,  # pending_source u=5: src 1920..2304 (blks 30..35)
        ]

    shadow_steady = SimpleNamespace(
        prefix=pref_2img,
        pending_source=pending_src_u5,
        _history=_mock_history,
        window_policy=SimpleNamespace(sink_chunks=3, retains=lambda u, cur: True),
    )
    hist_items = sparse_mod.extract_history_items(shadow_steady)
    kv_plan_3 = SimpleNamespace(write_start=2304, ends=[384, 768, 1152])
    block_idx = torch.cat(
        [
            torch.full(
                (384,), 12, dtype=torch.int64
            ),  # final u=5: 2304..2688 (blks 36..41)
            torch.full((384,), 13, dtype=torch.int64),  # source u=6: 2688..3072
            torch.full((384,), 14, dtype=torch.int64),  # active u=6: 3072..3456
        ]
    )
    packed_3 = {
        "_zing_host_layout": SimpleNamespace(block_idx=block_idx),
        "_zing_joint_phases": ("final", "source", "active"),
    }
    specs = sparse_mod.attach_sparse_specs_to_plan(
        None,
        hist_items,
        shadow_steady.window_policy,
        packed_3,
        "joint",
        kv_plan_3,
        {"mode": "sparse_095", "case_id": "product-reference-switch"},
        shadow=shadow_steady,
    )
    assert 1 in specs and 2 in specs
    sp_src = specs[1]
    assert sp_src.block_kinds_sig == (
        42,
        ((4, 12),),
        (12, 18),
        ((18, 24), (30, 36)),
    ), f"Unexpected block_kinds_sig: {sp_src.block_kinds_sig}"
    kind_t = sparse_mod._get_blk_kind_gpu(
        torch.device("cpu"), 42, 12, sp_src.blk_kind_sig
    ).tolist()
    assert kind_t[0:4] == [sparse_mod.KIND_TEXT] * 4
    assert kind_t[4:12] == [sparse_mod.KIND_REF_VIS] * 8
    assert kind_t[12:18] == [sparse_mod.KIND_SINK] * 6
    assert kind_t[18:24] == [sparse_mod.KIND_WIN_SRC] * 6
    assert kind_t[24:30] == [sparse_mod.KIND_WIN_FIN] * 6
    assert kind_t[30:36] == [sparse_mod.KIND_WIN_SRC] * 6
    assert kind_t[36:42] == [sparse_mod.KIND_WIN_FIN] * 6


def test_noisy_step0_rope_does_not_rescue_stale_sink_under_tau095():
    """Verify clean source_t0 mismatch evicts stale KIND_SINK even under noisy t=1000 RoPE."""
    H, D = 4, 128
    L_q, M_q = 64, 1
    M_cand = 40
    M_prefix = 4
    L_k = M_cand * 64

    torch.manual_seed(17)
    q_first = torch.zeros(L_q, H, D)
    q_second = torch.zeros(L_q, H, D)
    q_first[:, :, 0] = 2.0
    q_first[:, :, 2] = 1.2
    q_second[:, :, 1] = 2.0
    q_second[:, :, 2] = 1.2

    k = torch.randn(L_k, H, D) * 0.25
    # Block 2: <Picture 1> stale room ref; Block 3: <Picture 2/3> active character ref
    k[2 * 64 : 3 * 64, :, 0] = -1.5
    k[2 * 64 : 3 * 64, :, 1] = 0.8
    k[2 * 64 : 3 * 64, :, 2] = -0.8
    k[3 * 64 : 4 * 64, :, 0] = 0.5
    k[3 * 64 : 4 * 64, :, 1] = 0.5
    k[3 * 64 : 4 * 64, :, 2] = 2.0
    # Blocks 4..11: KIND_SINK 0s closed wall (strong noisy RoPE dim 1, negative clean dim 0)
    for b in range(4, 12):
        k[b * 64 : (b + 1) * 64, :, 0] = -1.5
        k[b * 64 : (b + 1) * 64, :, 1] = 1.4
    # Blocks 12..39: 28 sliding-window open cross-corridor blocks
    for b in range(12, 40):
        k[b * 64 : (b + 1) * 64, :, 0] = 0.8 + 0.02 * (b - 12)
        k[b * 64 : (b + 1) * 64, :, 1] = 0.6 + 0.01 * ((b - 12) % 7)

    blk_kind_sig = (M_cand, ((2, 4),), (4, 12), ((12, 26),))
    # Layer 0 row of HEAD_BUDGET_RATIOS["target_t1000"]["0.95"] has high-ratio spatial heads
    # (use layer_idx=34 or layer_idx=1 where head 2/3 have generous budgets, or safety_scale=1.5)
    m_src, m_s0, _ = sparse_mod.fused_route_and_build_masks(
        q_first=q_first,
        q_second=q_second,
        k=k,
        layer_idx=1,
        tau="0.95",
        L_q=L_q,
        M_q=M_q,
        M_cand=M_cand,
        M_prefix=M_prefix,
        safety_scale=1.0,
        build_src=True,
        L_k_src=L_k,
        M_k_src=M_cand,
        M_cur_src=1,
        build_s0=True,
        L_k_s0=L_k,
        M_k_s0=M_cand,
        M_cur_s0=1,
        build_s1=False,
        s_evict=0,
        L_k_s1=0,
        M_k_s1=0,
        M_cand_s1=0,
        M_cur_s1=0,
        head_start=6,  # layer 1 heads 6,7 have ratios 0.617, 0.694 (~25-28 blocks out of 40!)
        num_heads=H,
        blk_kind_sig=blk_kind_sig,
    )
    assert m_s0 is not None and m_src is not None
    for h in range(2):  # heads 6 and 7 are spatial heads with ~62-69% budget!
        cnt = int(m_s0.cand_cnt[h].item())
        sel_s0 = m_s0.cand_idx[h, 0, :cnt].tolist()
        assert 3 in sel_s0, (
            f"Character ref Block 3 missing on spatial head {h}: {sel_s0}"
        )
        leaked = [b for b in range(4, 12) if b in sel_s0]
        assert not leaked, f"Stale KIND_SINK wall leaked on spatial head {h}: {leaked}"
        assert 2 not in sel_s0, (
            f"Stale room ref Block 2 leaked on spatial head {h}: {sel_s0}"
        )


def test_mixed_latent_mq_budget_and_parity_isolation():
    """Verify mixed 2-latent (M_q=13) and 3-latent (M_q=19) chunks never collide in _BUDGET_CACHE or parity masks."""
    sparse_mod.clear_sparse_caches()
    dev = torch.device("cpu")
    b13 = sparse_mod._get_head_budget_spec(
        dev,
        "target_t1000",
        "0.90",
        10,
        M_k=220,
        M_cand=184,
        M_cur=36,
        M_prefix=50,
        M_q=13,
    )
    b19 = sparse_mod._get_head_budget_spec(
        dev,
        "target_t1000",
        "0.90",
        10,
        M_k=220,
        M_cand=184,
        M_cur=36,
        M_prefix=50,
        M_q=19,
    )
    assert b13 is not b19, "_BUDGET_CACHE must distinguish M_q=13 vs M_q=19"
    assert b19.K_max >= b13.K_max

    # Populate parity=1 with M_q=13 mask, then verify an M_q=19 spec with same parity=1 does NOT reuse M_q=13 mask
    dummy = torch.zeros(4, dtype=torch.int32)
    mask13 = sparse_mod.PrebuiltSparseMask(
        L_q=780,
        L_k=14569,
        M_q=13,
        M_k=228,
        M_cand=191,
        cand_cnt=dummy,
        cand_idx=torch.zeros((4, 13, 228), dtype=torch.int32),
        head_order=torch.arange(4, dtype=torch.int32),
        active_ratio=0.5,
    )
    sparse_mod._STEP0_PARITY_MASKS[(1, 13, 10)] = mask13
    assert sparse_mod._STEP0_PARITY_MASKS.get((1, 19, 10)) is None


def _zing_chunk_graphs_ctx():
    import contextlib
    import logging
    import types

    @contextlib.contextmanager
    def _ctx():
        saved = dict(sys.modules)
        try:
            try:
                sys.modules[
                    "sglang.multimodal_gen.runtime.models.dits.zing_h3_sparse_attn"
                ] = sparse_mod
                from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
                    ZingChunkGraphs,
                )

                zh3_mod = types.ModuleType(
                    "sglang.multimodal_gen.runtime.models.dits.zing_h3"
                )
                zh3_mod._FixedFA4Metadata = lambda ft, br: (ft, br)
                zh3_mod._fixed_fa4_attention = lambda q, k, v, scale, meta, **kw: (
                    torch.nn.functional.scaled_dot_product_attention(
                        q.transpose(0, 1)[None],
                        k.transpose(0, 1)[None],
                        v.transpose(0, 1)[None],
                        scale=scale,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
                sys.modules[zh3_mod.__name__] = zh3_mod

                yield ZingChunkGraphs
                return
            except ModuleNotFoundError:
                pass

            for pkg in (
                "sglang",
                "sglang.multimodal_gen",
                "sglang.multimodal_gen.runtime",
                "sglang.multimodal_gen.runtime.distributed",
                "sglang.multimodal_gen.runtime.distributed.device_communicators",
                "sglang.multimodal_gen.runtime.utils",
                "sglang.multimodal_gen.runtime.models",
                "sglang.multimodal_gen.runtime.models.dits",
            ):
                m = types.ModuleType(pkg)
                m.__path__ = []
                sys.modules[pkg] = m
            ipc_mod = types.ModuleType(
                "sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a"
            )
            ipc_mod.nccl_during_capture = contextlib.nullcontext
            sys.modules[ipc_mod.__name__] = ipc_mod

            log_mod = types.ModuleType(
                "sglang.multimodal_gen.runtime.utils.logging_utils"
            )
            log_mod.init_logger = logging.getLogger
            sys.modules[log_mod.__name__] = log_mod

            zh3_mod = types.ModuleType(
                "sglang.multimodal_gen.runtime.models.dits.zing_h3"
            )
            zh3_mod._FixedFA4Metadata = lambda ft, br: (ft, br)
            zh3_mod._fixed_fa4_attention = lambda q, k, v, scale, meta, **kw: (
                torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(0, 1)[None],
                    k.transpose(0, 1)[None],
                    v.transpose(0, 1)[None],
                    scale=scale,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
            sys.modules[zh3_mod.__name__] = zh3_mod

            sys.modules[
                "sglang.multimodal_gen.runtime.models.dits.zing_h3_sparse_attn"
            ] = sparse_mod

            dg_path = MODULE_PATH.parent / "zing_h3_denoise_graph.py"
            dg_spec = importlib.util.spec_from_file_location(
                "_standalone_zing_h3_denoise_graph", str(dg_path)
            )
            dg_mod = importlib.util.module_from_spec(dg_spec)
            dg_spec.loader.exec_module(dg_mod)
            yield dg_mod.ZingChunkGraphs
        finally:
            for k in list(sys.modules):
                if k not in saved:
                    if (
                        k.startswith("sglang")
                        or k == "_standalone_zing_h3_denoise_graph"
                    ):
                        sys.modules.pop(k, None)
            sys.modules.update(saved)

    return _ctx()


def test_chunk_graphs_shared_pool_generation_and_prefix_isolation():
    """Verify ZingChunkGraphs retains 8 shapes without mid-pool LRU eviction and drains whole generations on overflow."""
    import json
    import tempfile
    from types import SimpleNamespace

    with _zing_chunk_graphs_ctx() as ZingChunkGraphs:
        _verify_chunk_graphs_shared_pool(
            ZingChunkGraphs, json, tempfile, SimpleNamespace
        )


def _verify_chunk_graphs_shared_pool(ZingChunkGraphs, json, tempfile, SimpleNamespace):
    class MockSharedPoolBackend:
        def __init__(self):
            self.capturing = False
            self._pool = None
            self.pool_generations = 0

        def supports(self, device):
            return True

        def capture(self, fn, warmed=None):
            fn()
            if warmed is not None:
                assert warmed(True)
            if self._pool is None:
                self.pool_generations += 1
                self._pool = f"pool_gen_{self.pool_generations}"
            self.capturing = True
            try:
                outputs = fn()
            finally:
                self.capturing = False
            return (self._pool, outputs), outputs

        def replay(self, graph):
            pass

    def make_plan(sig_id: int, chunk_id: int = 12):
        sp = sparse_mod.SparseBranchSpec(
            mode="sparse_095",
            tau="0.95",
            case_id="long_session_test",
            chunk_id=chunk_id,
            b_phase="active",
            step_idx=0,
            step_1000=1000,
            q_start=0,
            q_stop=780,
            L_q=780,
            L_k=18000 + sig_id * 64,
            s_cur_chunk=17220 + sig_id * 64,
            M_q=13,
            M_k=282 + sig_id,
            M_cand=269 + sig_id,
            M_cur=13,
            M_prefix=180,
            blk_kind_sig=(
                269 + sig_id,
                ((10, 180),),
                (180, 210),
                ((210, 230 + sig_id),),
            ),
        )
        return [
            SimpleNamespace(
                signature=(f"shape_{sig_id}",),
                kv_plan=SimpleNamespace(
                    write_start=100,
                    rows=2,
                    clean_rows=2,
                    _sparse_branches={0: sp},
                ),
            )
        ]

    k, v = torch.zeros(256, 2, 8), torch.zeros(256, 2, 8)
    tensors = {"delta": torch.tensor(1.0)}

    def execute(static):
        return (static["delta"].clone(),)

    # 1. Default max_graphs=16 retains all 8 pre/post prompt_joint sparse shapes in 1 shared pool
    backend16 = MockSharedPoolBackend()
    graphs16 = ZingChunkGraphs(backend=backend16, max_graphs=16)
    for sig_id in range(8):
        plans = make_plan(sig_id, chunk_id=12 + (sig_id % 2))
        graphs16._run("steady", plans, tensors, execute, [(k, v)])  # warmup
        graphs16._run("steady", plans, tensors, execute, [(k, v)])  # capture
        graphs16._run("steady", plans, tensors, execute, [(k, v)])  # replay
    assert len(graphs16.graphs) == 8
    assert backend16.pool_generations == 1
    assert graphs16.routes[("steady", "shape_reset")] == 0
    assert graphs16.routes[("steady", "capture")] == 8
    assert graphs16.routes[("steady", "replay")] == 8

    # 2. Exceeding max_graphs=4 drains entire old generation + resets _pool BEFORE capturing new generation
    backend4 = MockSharedPoolBackend()
    graphs4 = ZingChunkGraphs(backend=backend4, max_graphs=4)
    for sig_id in range(4):
        plans = make_plan(sig_id)
        graphs4._run("steady", plans, tensors, execute, [(k, v)])
        graphs4._run("steady", plans, tensors, execute, [(k, v)])
    assert len(graphs4.graphs) == 4
    assert backend4.pool_generations == 1

    plans5 = make_plan(4)
    graphs4._run("steady", plans5, tensors, execute, [(k, v)])  # 1st sighting -> warmup
    assert len(graphs4.graphs) == 4
    graphs4._run(
        "steady", plans5, tensors, execute, [(k, v)]
    )  # 2nd sighting at capacity -> full drain + warmup
    assert graphs4.routes[("steady", "shape_reset")] == 1
    assert len(graphs4.graphs) == 0
    assert backend4._pool is None
    graphs4._run(
        "steady", plans5, tensors, execute, [(k, v)]
    )  # 2nd sighting of new gen -> capture in pool_gen_2
    assert len(graphs4.graphs) == 1
    assert backend4.pool_generations == 2

    # 3. prefix_separate ignores ZING_SPARSE_STATE_FILE mode switch mid-session
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        state_path = Path(tf.name)
        tf.write(json.dumps({"mode": "dense", "case_id": "t1"}))
    old_env = os.environ.get("ZING_SPARSE_STATE_FILE")
    try:
        os.environ["ZING_SPARSE_STATE_FILE"] = str(state_path)
        pref_backend = MockSharedPoolBackend()
        pref_graphs = ZingChunkGraphs(backend=pref_backend, max_graphs=2)
        pref_plan = [
            SimpleNamespace(
                signature=("pref_0",),
                kv_plan=SimpleNamespace(write_start=0, rows=2, clean_rows=2),
            )
        ]
        pref_graphs._run("prefix_separate", pref_plan, tensors, execute, [(k, v)])
        pref_graphs._run("prefix_separate", pref_plan, tensors, execute, [(k, v)])
        assert len(pref_graphs.graphs) == 1
        state_path.write_text(json.dumps({"mode": "sparse_095", "case_id": "t2"}))
        pref_graphs._run("prefix_separate", pref_plan, tensors, execute, [(k, v)])
        assert len(pref_graphs.graphs) == 1
        assert pref_graphs.routes[("prefix_separate", "replay")] == 1
    finally:
        if old_env is None:
            os.environ.pop("ZING_SPARSE_STATE_FILE", None)
        else:
            os.environ["ZING_SPARSE_STATE_FILE"] = old_env
        state_path.unlink(missing_ok=True)


def test_chunk_graphs_zero_recapture_across_dynamic_lk_mcand_and_prefix_switch():
    """Verify Plan A: device-tensorized sparse_meta achieves zero CUDAGraph recapture across dynamic L_k, M_cand, M_prefix, and blk_kind."""
    from types import SimpleNamespace

    with _zing_chunk_graphs_ctx() as ZingChunkGraphs:

        class MockCaptureBackend:
            def __init__(self):
                self.capturing = False
                self._pool = None
                self.captures = 0
                self.replays = 0
                self.s1_seen_in_warmup = None
                self.s1_seen_in_capture = None

            def supports(self, device):
                return True

            def capture(self, fn, warmed=None):
                fn()
                if warmed is not None:
                    assert warmed(True)
                self.capturing = True
                try:
                    outputs = fn()
                finally:
                    self.capturing = False
                self.captures += 1
                return ("graph_obj", outputs), outputs

            def replay(self, graph):
                self.replays += 1

        dev = torch.device("cpu")
        H, D = 4, 128
        ws = sparse_mod._get_workspace(dev, num_heads=H, req_m_q=13, req_m_cand=512)
        ws.cand_idx_s1[0].fill_(77)
        stores = [
            (
                torch.zeros((4096, H, D), device=dev),
                torch.zeros((4096, H, D), device=dev),
            )
        ]

        def make_chunk_plans(u_cur: int, m_cand: int, m_prefix: int):
            min_chunk = sparse_mod.get_min_sparse_chunk()
            mode_cur = "dense" if u_cur < min_chunk else "sparse_095"
            mode_next = "dense" if (u_cur + 1) < min_chunk else "sparse_095"
            plans = []
            tensors = {}
            for p_idx in range(4):
                if p_idx < 3:
                    sp = sparse_mod.SparseBranchSpec(
                        mode=mode_cur,
                        tau="0.95",
                        case_id="plan_a_test",
                        chunk_id=u_cur,
                        b_phase="active",
                        step_idx=p_idx + 1,
                        step_1000=500,
                        q_start=0,
                        q_stop=780,
                        L_q=780,
                        L_k=m_cand * 64 + 780,
                        s_cur_chunk=m_cand * 64,
                        M_q=13,
                        M_k=m_cand + 13,
                        M_cand=m_cand,
                        M_cur=13,
                        M_prefix=m_prefix,
                        blk_kind_sig=(
                            m_cand,
                            ((2, m_prefix),),
                            (m_prefix, m_prefix + 6),
                            ((m_prefix + 6, m_cand),),
                        ),
                    )
                    sb = {0: sp}
                else:
                    sp_fin = sparse_mod.SparseBranchSpec(
                        mode=mode_cur,
                        tau="0.95",
                        case_id="plan_a_test",
                        chunk_id=u_cur,
                        b_phase="final",
                        step_idx=99,
                        step_1000=1,
                        q_start=0,
                        q_stop=390,
                        L_q=390,
                        L_k=m_cand * 64 + 390,
                        s_cur_chunk=m_cand * 64,
                        M_q=7,
                        M_k=m_cand + 7,
                        M_cand=m_cand,
                        M_cur=7,
                        M_prefix=m_prefix,
                    )
                    sp_src = sparse_mod.SparseBranchSpec(
                        mode=mode_next,
                        tau="0.95",
                        case_id="plan_a_test",
                        chunk_id=u_cur + 1,
                        b_phase="source",
                        step_idx=0,
                        step_1000=0,
                        q_start=390,
                        q_stop=780,
                        L_q=390,
                        L_k=(m_cand + 12) * 64 + 390,
                        s_cur_chunk=(m_cand + 12) * 64,
                        M_q=7,
                        M_k=m_cand + 19,
                        M_cand=m_cand + 12,
                        M_cur=7,
                        M_prefix=m_prefix,
                        blk_kind_sig=(
                            m_cand + 12,
                            ((2, m_prefix),),
                            (m_prefix, m_prefix + 6),
                            ((m_prefix + 6, m_cand + 12),),
                        ),
                    )
                    sp_act0 = sparse_mod.SparseBranchSpec(
                        mode=mode_next,
                        tau="0.95",
                        case_id="plan_a_test",
                        chunk_id=u_cur + 1,
                        b_phase="active",
                        step_idx=0,
                        step_1000=1000,
                        q_start=780,
                        q_stop=1560,
                        L_q=780,
                        L_k=(m_cand + 12) * 64 + 1170,
                        s_cur_chunk=(m_cand + 12) * 64,
                        M_q=13,
                        M_k=m_cand + 31,
                        M_cand=m_cand + 12,
                        M_cur=19,
                        M_prefix=m_prefix,
                        s_evict=m_prefix * 64,
                        L_evict=384,
                        s_cur_chunk_step1=(m_cand + 6) * 64,
                        L_k_step1=(m_cand + 6) * 64 + 1170,
                        M_k_step1=m_cand + 25,
                        M_cand_step1=m_cand + 6,
                        M_cur_step1=19,
                    )
                    sp_src.companion_step0_spec = sp_act0
                    sb = {0: sp_fin, 1: sp_src, 2: sp_act0}

                kv_plan = SimpleNamespace(
                    write_start=128,
                    rows=16,
                    clean_rows=16,
                    stores=stores,
                    tensors={},
                    _sparse_branches=sb,
                )
                plan = SimpleNamespace(
                    signature=(f"static_plan_{p_idx}",),
                    kv_plan=kv_plan,
                    tensors={},
                )
                sparse_mod._populate_plan_sparse_tensors(kv_plan, sb, plan=plan)
                for k_name, v_ten in plan.tensors.items():
                    tensors[f"f{p_idx}.{k_name}"] = v_ten
                plans.append(plan)
            return plans, tensors

        backend = MockCaptureBackend()
        graphs = ZingChunkGraphs(backend=backend, max_graphs=16)

        def make_execute(u_cur: int):
            def _exec(static):
                val = int(ws.cand_idx_s1[0, 0, 0, 0, 0].item())
                if backend.capturing:
                    backend.s1_seen_in_capture = val
                else:
                    backend.s1_seen_in_warmup = val
                # Simulate joint branch overwriting ws.cand_idx_s1[0] for next chunk
                ws.cand_idx_s1[0].fill_(100 + u_cur)
                return (static["f3.kv.sparse_meta"].clone(),)

            return _exec

        # 1. Dense warmup chunks u = 1, 2, 3 (u < min_chunk = 5) -> captures 1 Dense Graph at u = 2, replays at u = 3
        for u, mc in [(1, 24), (2, 36), (3, 48)]:
            p, t = make_chunk_plans(u, m_cand=mc, m_prefix=8)
            graphs._run("chunk", p, t, make_execute(u), stores)
        assert len(graphs.graphs) == 1
        assert graphs.routes[("chunk", "capture")] == 1
        assert graphs.routes[("chunk", "replay")] == 1

        # 2. Mixed transition chunk u = 4 (min_chunk - 1, where joint lookahead u = 5 is sparse) -> 1 eager step
        p4, t4 = make_chunk_plans(4, m_cand=60, m_prefix=8)
        graphs._run("chunk", p4, t4, make_execute(4), stores)
        assert len(graphs.graphs) == 1
        assert graphs.routes[("chunk", "eager")] == 1
        assert int(ws.cand_idx_s1[0, 0, 0, 0, 0].item()) == 104

        # 3. Sparse chunks u = 5 (warmup), u = 6 (capture), u = 7..10 (replay across growing M_cand and M_prefix switch!)
        for u, mc, mp in [
            (5, 72, 8),
            (6, 96, 8),
            (7, 132, 8),
            (8, 180, 16),  # Reference switch: M_prefix jumps 8 -> 16, M_cand -> 180
            (9, 240, 16),
            (10, 312, 24),  # Reference switch 2: M_prefix jumps 16 -> 24, M_cand -> 312
        ]:
            p, t = make_chunk_plans(u, m_cand=mc, m_prefix=mp)
            graphs._run("chunk", p, t, make_execute(u), stores)
            if u == 6:
                # Verify restore() restored ws.cand_idx_s1[0] (105 from u=5) before the capturing pass!
                assert backend.s1_seen_in_warmup == 105
                assert backend.s1_seen_in_capture == 105

        # Total graphs across the entire session: 1 Dense + 1 Sparse = 2 graphs, zero shape_reset!
        assert len(graphs.graphs) == 2
        assert graphs.routes[("chunk", "capture")] == 2
        assert graphs.routes[("chunk", "shape_reset")] == 0
        # u=3 (1 dense replay) + u=7,8,9,10 (4 sparse replays) = 5 replays
        assert graphs.routes[("chunk", "replay")] == 5

        # Verify static buffer was refreshed in-place with u=10's M_cand (312 + 12 = 324) and M_prefix (24)
        sparse_entry = list(graphs.graphs.values())[1]
        refreshed_meta = sparse_entry.static["f3.kv.sparse_meta"]
        assert int(refreshed_meta[sparse_mod.META_M_CAND].item()) == 324
        assert int(refreshed_meta[sparse_mod.META_M_PREFIX].item()) == 24

        # 4. Verify execute_branch_attention end-to-end with device-tensorized sparse_meta across two dynamic configs
        q_joint = torch.randn((1560, H, D), device=dev)
        k_store = torch.randn((32768, H, D), device=dev)
        v_store = torch.randn((32768, H, D), device=dev)
        scale = 1.0 / math.sqrt(D)
        for u_test, mc_test, mp_test in [(5, 32, 6), (6, 48, 12)]:
            p_test, _ = make_chunk_plans(u_test, m_cand=mc_test, m_prefix=mp_test)
            kv_p3 = p_test[3].kv_plan
            sp_src = kv_p3._sparse_branches[1]
            sp_act0 = kv_p3._sparse_branches[2]
            out_src = sparse_mod.execute_branch_attention(
                None,
                q_joint[390:780],
                k_store,
                v_store,
                scale,
                metadata=None,
                optimized=False,
                layer_idx=10,
                spec=sp_src,
                fixed_tensors=kv_p3.tensors,
                branch=1,
                q_full=q_joint,
            )
            out_act0 = sparse_mod.execute_branch_attention(
                None,
                q_joint[780:1560],
                k_store,
                v_store,
                scale,
                metadata=None,
                optimized=False,
                layer_idx=10,
                spec=sp_act0,
                fixed_tensors=kv_p3.tensors,
                branch=2,
                q_full=q_joint,
            )
            assert out_src.shape == (390, H, D) and torch.isfinite(out_src).all()
            assert out_act0.shape == (780, H, D) and torch.isfinite(out_act0).all()
            # Step 1 plan (p_test[0]) reuses prebuilt s1 mask in slot 0 with its own sparse_num_full & meta_branch
            kv_p0 = p_test[0].kv_plan
            sp_s1 = kv_p0._sparse_branches[0]
            meta_s1 = kv_p0.tensors["sparse_meta"]
            assert int(meta_s1[sparse_mod.META_B_EVICT_LO].item()) == sp_s1.M_cand
            assert int(meta_s1[sparse_mod.META_B_EVICT_HI].item()) == sp_s1.M_cand
            assert int(meta_s1[sparse_mod.META_SHIFT_S1].item()) == 0
            assert int(meta_s1[sparse_mod.META_M_CAND_S1].item()) == sp_s1.M_cand
            assert int(meta_s1[sparse_mod.META_L_K_S1].item()) == sp_s1.L_k
            out_s1 = sparse_mod.execute_branch_attention(
                None,
                q_joint[:780],
                k_store,
                v_store,
                scale,
                metadata=None,
                optimized=False,
                layer_idx=10,
                spec=sp_s1,
                fixed_tensors=kv_p0.tensors,
                branch=0,
            )
            assert out_s1.shape == (780, H, D) and torch.isfinite(out_s1).all()

        # 5. When CUDA is available, run REAL CudaGraphBackend + ZingChunkGraphs + Triton kernels
        # across u=2..10 (dynamic L_k/M_cand growth + 2 reference switches) and assert 0.0 replay error!
        if torch.cuda.is_available():
            cuda_dev = torch.device("cuda")
            H_cu = 56
            k_cu = torch.randn((32768, H_cu, D), device=cuda_dev, dtype=torch.bfloat16)
            v_cu = torch.randn((32768, H_cu, D), device=cuda_dev, dtype=torch.bfloat16)
            q_step1_cu = torch.randn(
                (780, H_cu, D), device=cuda_dev, dtype=torch.bfloat16
            )
            q_joint_cu = torch.randn(
                (2340, H_cu, D), device=cuda_dev, dtype=torch.bfloat16
            )
            stores_cu = ((k_cu, v_cu),)
            cuda_graphs = ZingChunkGraphs(max_graphs=16)

            class _CuSampler:
                initial_block_count = 3
                initial_denoising_step_list = [1000, 750, 500, 250]
                denoising_step_list = [1000, 500]

                class _Sched:
                    timesteps = [torch.tensor(0.0), torch.tensor(0.6667)]

                video_scheduler = _Sched()
                initial_schedulers = [_Sched()]

            class _CuWindow:
                sink_chunks = 3
                window_chunks = 6

                def retains(self, u: int, u_cur: int) -> bool:
                    return u < 3 or (u_cur - u) <= 5

            class _CuHost:
                def __init__(self, b_list, role_t=None):
                    self.block_idx = torch.tensor(b_list, dtype=torch.long)
                    self.row_role = role_t

            class _CuKVPlan:
                def __init__(self, ends, write_start):
                    self.ends = tuple(ends)
                    self.write_start = write_start
                    self.rows = ends[-1]
                    self.clean_rows = ends[-1]
                    self.stores = stores_cu
                    self.rotated = (k_cu,)
                    self.tensors = {}
                    prev = 0
                    for i, stop in enumerate(ends):
                        self.tensors[f"cu_k.{i}"] = torch.tensor(
                            [0, write_start + stop], dtype=torch.int32, device=cuda_dev
                        )
                        self.tensors[f"cu_q.{i}"] = torch.tensor(
                            [0, stop - prev], dtype=torch.int32, device=cuda_dev
                        )
                        prev = stop
                    self._sparse_branches = {}

            class _CuPlan:
                def __init__(self, sig, kv_plan):
                    self.signature = sig
                    self.kv_plan = kv_plan
                    self.tensors = {f"kv.{k}": v for k, v in kv_plan.tensors.items()}

            def make_cuda_plans(u_cur: int, m_cand: int, m_prefix: int):
                L_pref = m_prefix * 64
                s_cur = m_cand * 64
                hist = [
                    ("prefix", -1, 0, L_pref),
                    ("committed", 0, L_pref, L_pref + 1170),
                    ("committed", max(1, u_cur - 1), L_pref + 1170, s_cur),
                ]
                role_pref = torch.zeros((L_pref,), dtype=torch.long)
                if L_pref > 192:
                    role_pref[192:] = 4
                sparse_mod.attach_sparse_specs_to_plan(
                    sampler=_CuSampler(),
                    shadow_before_history=[],
                    window_policy=_CuWindow(),
                    packed={"_zing_host_layout": _CuHost([0] * L_pref, role_pref)},
                    phase="prefix",
                    kv_plan=_CuKVPlan([L_pref], 0),
                    state_cfg={"mode": "sparse_095", "case_id": "test_cu"},
                )

                plans_cu = []
                # Step 1 active (t=500) of u_cur
                kv1 = _CuKVPlan([780], s_cur)
                p1 = _CuPlan("s1", kv1)
                p1.tensors["q_in"] = q_step1_cu
                sparse_mod.attach_sparse_specs_to_plan(
                    sampler=_CuSampler(),
                    shadow_before_history=hist,
                    window_policy=_CuWindow(),
                    packed={
                        "_zing_host_layout": _CuHost([u_cur * 2 + 2] * 780),
                        "_zing_host_t": torch.tensor([0.6667] * 780),
                    },
                    phase="active",
                    kv_plan=kv1,
                    state_cfg={"mode": "sparse_095", "case_id": "test_cu"},
                    plan=p1,
                )
                plans_cu.append(p1)

                # Joint lookahead (final u_cur + source u_cur+1 + active t=1000 u_cur+1)
                u_next = u_cur + 1
                s_next = s_cur + 780
                hist_j = hist + [("committed", u_cur, s_cur, s_next)]
                kvj = _CuKVPlan([780, 1560, 2340], s_next)
                pj = _CuPlan("joint", kvj)
                pj.tensors["q_in"] = q_joint_cu
                sparse_mod.attach_sparse_specs_to_plan(
                    sampler=_CuSampler(),
                    shadow_before_history=hist_j,
                    window_policy=_CuWindow(),
                    packed={
                        "_zing_joint_phases": ("final", "source", "active"),
                        "_zing_host_layout": _CuHost(
                            [u_cur * 2 + 2] * 780
                            + [u_next * 2 + 1] * 780
                            + [u_next * 2 + 2] * 780
                        ),
                        "_zing_host_t": torch.tensor(
                            [1.0] * 780 + [1.0] * 780 + [0.0] * 780
                        ),
                    },
                    phase="joint",
                    kv_plan=kvj,
                    state_cfg={"mode": "sparse_095", "case_id": "test_cu"},
                    plan=pj,
                )
                plans_cu.append(pj)

                tensors_cu = {}
                cuda_graphs._attach(plans_cu, tensors_cu)
                return plans_cu, tensors_cu

            def make_cuda_execute(plans_cu):
                def _exec(static):
                    kv_s1_t = {
                        name: static[f"f0.kv.{name}"]
                        for name in plans_cu[0].kv_plan.tensors
                    }
                    res_s1 = sparse_mod.execute_fixed_block_sparse_branches(
                        static["f0.q_in"],
                        k_cu,
                        v_cu,
                        scale,
                        10,
                        plans_cu[0].kv_plan,
                        kv_s1_t,
                        ulysses_rank=0,
                        optimized=False,
                        target_736p=False,
                    )
                    kv_j_t = {
                        name: static[f"f1.kv.{name}"]
                        for name in plans_cu[1].kv_plan.tensors
                    }
                    res_j = sparse_mod.execute_fixed_block_sparse_branches(
                        static["f1.q_in"],
                        k_cu,
                        v_cu,
                        scale,
                        10,
                        plans_cu[1].kv_plan,
                        kv_j_t,
                        ulysses_rank=0,
                        optimized=False,
                        target_736p=False,
                    )
                    return (res_s1[0], res_j[1], res_j[2])

                return _exec

            for u_c, mc_c, mp_c in [
                (2, 48, 8),
                (3, 56, 8),
                (4, 64, 8),
                (5, 76, 8),
                (6, 96, 8),
                (7, 132, 8),
                (8, 180, 16),
                (9, 240, 16),
                (10, 312, 24),
                (11, 360, 24),
            ]:
                p_cu, t_cu = make_cuda_plans(u_c, mc_c, mp_c)
                exec_fn = make_cuda_execute(p_cu)
                s1_buf_cu = sparse_mod.get_active_s1_mask_buffer(stores_cu)
                s1_before = s1_buf_cu.clone() if s1_buf_cu is not None else None
                out_graph = cuda_graphs._run("steady", p_cu, t_cu, exec_fn, stores_cu)
                if u_c >= 7:
                    s1_after_graph = (
                        s1_buf_cu.clone() if s1_buf_cu is not None else None
                    )
                    if s1_buf_cu is not None and s1_before is not None:
                        s1_buf_cu.copy_(s1_before)
                    out_eager = exec_fn(t_cu)
                    if s1_buf_cu is not None and s1_after_graph is not None:
                        s1_buf_cu.copy_(s1_after_graph)
                    for b_i, (g_t, e_t) in enumerate(zip(out_graph, out_eager)):
                        err = float((g_t - e_t).abs().max().item())
                        assert err == 0.0, (
                            f"CUDA Graph replay vs Eager mismatch at u={u_c} branch={b_i}: {err}"
                        )

            assert len(cuda_graphs.graphs) == 2
            assert cuda_graphs.routes[("steady", "capture")] == 2
            assert cuda_graphs.routes[("steady", "shape_reset")] == 0
            assert cuda_graphs.routes[("steady", "replay")] == 5


def test_compressed_kv_store_in_place_compaction_and_sizing():
    """Verify in-place compaction preserving CUDA Graph pointer stability and dynamic sizing."""
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        ZingH3KVCache,
        ZingWindowPolicy,
    )

    H, D = 4, 128
    dev = torch.device("cpu")

    # 1. Verify dynamic sizing: capacity=4096 gives max_blks=64 (not 512!)
    cache = ZingH3KVCache(
        num_layers=2,
        window_policy=ZingWindowPolicy(window_chunks=4, sink_chunks=2),
    )

    class _MockStorage:
        def __init__(self):
            self.k = torch.zeros((4096, H, D), device=dev)
            self.v = torch.zeros((4096, H, D), device=dev)
            self.rows = 4096
            self.layout = ((H, D), torch.float32, dev)

    cache.storage = (_MockStorage(), _MockStorage())
    cache._stores = {
        0: (cache.storage[0].k, cache.storage[0].v),
        1: (cache.storage[1].k, cache.storage[1].v),
    }
    assert cache.capacity == 4096

    stores = cache.get_compressed_kv_stores(device=dev, num_heads=H, head_dim=D)
    assert len(stores) == 2
    # Exactly (4096 + 63) // 64 = 64 blocks, avoiding 512-block (32k token) over-allocation!
    assert stores[0].max_blocks == 64
    assert stores[1].max_blocks == 64

    # 2. Record resident buffer data pointers
    ptr_k_res_0 = stores[0].k_res_fp8.data_ptr()
    ptr_v_res_0 = stores[0].v_res_fp8.data_ptr()
    ptr_k_mean_0 = stores[0].k_blk_mean.data_ptr()
    ptr_k_res_1 = stores[1].k_res_fp8.data_ptr()

    # Append 3 chunks (each 64 tokens = 1 block)
    for c_id in range(3):
        k0 = torch.randn((64, H, D), device=dev)
        v0 = torch.randn((64, H, D), device=dev)
        k1 = torch.randn((64, H, D), device=dev)
        v1 = torch.randn((64, H, D), device=dev)
        cache.storage[0].k[c_id * 64 : (c_id + 1) * 64] = k0
        cache.storage[0].v[c_id * 64 : (c_id + 1) * 64] = v0
        cache.storage[1].k[c_id * 64 : (c_id + 1) * 64] = k1
        cache.storage[1].v[c_id * 64 : (c_id + 1) * 64] = v1
        chunk = cache._store_chunk(
            torch.arange(c_id * 64, (c_id + 1) * 64, device=dev),
            block=(c_id + 1) * 2,
            frames=(),
            layers=[(k0, v0), (k1, v1)],
        )
        cache.committed.append(chunk)
    assert stores[0].num_blocks == 3
    assert stores[1].num_blocks == 3

    # 3. Simulate window eviction: keep only chunk 1 & 2 (evict chunk 0)
    retained_chunks = [cache.committed[1], cache.committed[2]]
    cache._sync_compressed_kv_store(retained_chunks)

    # 4. Assert 100% pointer stability across eviction (CUDA Graph safe!)
    assert stores[0].k_res_fp8.data_ptr() == ptr_k_res_0
    assert stores[0].v_res_fp8.data_ptr() == ptr_v_res_0
    assert stores[0].k_blk_mean.data_ptr() == ptr_k_mean_0
    assert stores[1].k_res_fp8.data_ptr() == ptr_k_res_1

    # And verify compacted block count is exactly 2 blocks
    assert stores[0].num_blocks == 2
    assert stores[1].num_blocks == 2
    # Verify chunk IDs inside store are now chunk 1 and chunk 2
    assert stores[0].block_chunk_id[:2].tolist() == [1, 2]
    assert stores[1].block_chunk_id[:2].tolist() == [1, 2]


def test_kv_block_stats_and_triage_mode_controls():
    """Verify Triage mode/env helpers and compute_kv_block_stats moments."""
    assert not sparse_mod.is_virtual_mean_enabled("sparse_095")
    assert not sparse_mod.is_triage_score_enabled("sparse_095")
    assert sparse_mod.is_virtual_mean_enabled("triage_095")
    assert sparse_mod.is_triage_score_enabled("triage_095")
    assert sparse_mod.is_virtual_mean_enabled("triage_090")
    assert sparse_mod.is_triage_score_enabled("triage_090")
    assert sparse_mod.get_sparse_price_pi("triage_095") == 0.0

    H, D = 4, 128
    M_cand = 6
    L_k = (M_cand + 2) * 64
    torch.manual_seed(101)
    k = torch.randn(L_k, H, D)
    v = torch.randn(L_k, H, D)
    # Make block 0 uniform
    k0 = torch.randn(1, H, D)
    v0 = torch.randn(1, H, D)
    k[:64] = k0.expand(64, H, D)
    v[:64] = v0.expand(64, H, D)

    km, vm, tk, rk = sparse_mod.compute_kv_block_stats(k, v, num_blocks=M_cand)
    assert km.shape == (H, M_cand, D)
    assert vm.shape == (H, M_cand, D)
    assert tk.shape == (H, M_cand)
    assert rk.shape == (H, M_cand)
    assert torch.allclose(km[:, 0, :].float(), k0[0].float(), atol=1e-2)
    assert torch.allclose(vm[:, 0, :].float(), v0[0].float(), atol=1e-2)
    assert float(tk[:, 0].abs().max().item()) < 1e-6
    assert float(rk[:, 0].abs().max().item()) < 1e-6
    assert bool((tk[:, 1:] > 0.05).all())
    assert bool((rk[:, 1:] > 0.01).all())


def test_virtual_mean_tiles_exact_on_uniform_and_pareto_gain():
    """Verify P0 Virtual Mean Tiles are exact on uniform skipped blocks and strictly improve L2 error."""
    torch.manual_seed(202)
    L_q, H, D = 64, 4, 128
    M_cand = 6
    M_cur = 2
    M_k = M_cand + M_cur
    L_k = M_k * 64
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(L_q, H, D)
    k = torch.randn(L_k, H, D) * 0.2
    v = torch.randn(L_k, H, D) * 0.2

    # Blocks 2..5 (skipped candidates) are uniform per block with distinct values
    for b in range(2, M_cand):
        kb = torch.randn(1, H, D) * 0.5
        vb = torch.randn(1, H, D) * 1.5
        k[b * 64 : (b + 1) * 64] = kb.expand(64, H, D)
        v[b * 64 : (b + 1) * 64] = vb.expand(64, H, D)

    # Select only current full block 6 and candidate blocks 0, 1 (skipping 2, 3, 4, 5)
    cand_cnt = torch.full((H,), 3, dtype=torch.int32)
    cand_idx = (
        torch.tensor([6, 0, 1], dtype=torch.int32)
        .view(1, 1, 3)
        .expand(H, 1, 3)
        .contiguous()
    )
    head_order = torch.arange(H, dtype=torch.int32)

    out_dense = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1)[None],
        k.transpose(0, 1)[None],
        v.transpose(0, 1)[None],
        scale=scale,
    )[0].transpose(0, 1)

    out_hard = sparse_mod.launch_zing_block_sparse_attn(
        q,
        k,
        v,
        scale,
        L_k,
        M_cand,
        M_k,
        cand_cnt,
        cand_idx,
        head_order,
        enable_virtual_mean=False,
    )
    out_vm = sparse_mod.launch_zing_block_sparse_attn(
        q,
        k,
        v,
        scale,
        L_k,
        M_cand,
        M_k,
        cand_cnt,
        cand_idx,
        head_order,
        enable_virtual_mean=True,
    )

    err_hard = float((out_hard - out_dense).norm().item() / out_dense.norm().item())
    err_vm = float((out_vm - out_dense).norm().item() / out_dense.norm().item())
    assert err_hard > 0.30, (
        f"Expected large hard-drop error on skipped blocks, got {err_hard}"
    )
    assert err_vm < 1e-3, (
        f"Expected near-zero error with virtual mean tiles, got {err_vm}"
    )

    # Add realistic intra-block Gaussian perturbations to skipped blocks 2..5
    k[2 * 64 : M_cand * 64] += torch.randn((M_cand - 2) * 64, H, D) * 0.35
    v[2 * 64 : M_cand * 64] += torch.randn((M_cand - 2) * 64, H, D) * 0.25
    out_dense_2 = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1)[None],
        k.transpose(0, 1)[None],
        v.transpose(0, 1)[None],
        scale=scale,
    )[0].transpose(0, 1)
    out_hard_2 = sparse_mod.launch_zing_block_sparse_attn(
        q,
        k,
        v,
        scale,
        L_k,
        M_cand,
        M_k,
        cand_cnt,
        cand_idx,
        head_order,
        enable_virtual_mean=False,
    )
    out_vm_2 = sparse_mod.launch_zing_block_sparse_attn(
        q,
        k,
        v,
        scale,
        L_k,
        M_cand,
        M_k,
        cand_cnt,
        cand_idx,
        head_order,
        enable_virtual_mean=True,
    )
    err_hard_2 = float(
        (out_hard_2 - out_dense_2).norm().item() / out_dense_2.norm().item()
    )
    err_vm_2 = float((out_vm_2 - out_dense_2).norm().item() / out_dense_2.norm().item())
    assert err_vm_2 < err_hard_2 * 0.25, (
        f"Expected >4x relative L2 reduction from Virtual Mean Tiles, got {err_vm_2:.4f} vs {err_hard_2:.4f}"
    )


def test_triage_second_order_complexity_routing_and_price_pi():
    """Verify P1 Triage second-order complexity routing, price_pi thresholding, and triage_095 branch execution."""
    torch.manual_seed(303)
    H, D = 4, 128
    L_q, M_q = 64, 1
    M_cand = 20
    M_prefix = 4
    M_cur = 2
    M_k = M_cand + M_cur
    L_k = M_k * 64
    scale = 1.0 / math.sqrt(D)

    q_first = torch.zeros(L_q, H, D)
    q_first[:, :, 0] = 1.0
    q_second = q_first.clone()

    k = torch.zeros(L_k, H, D)
    v = torch.zeros(L_k, H, D)

    # Block 10: uniform key/value with slightly higher mean dot product (1.05) -> 0 approx error as Virtual Mean Tile
    k[10 * 64 : 11 * 64, :, 0] = 1.05
    v[10 * 64 : 11 * 64, :, 0] = 1.0

    # Block 11: high intra-block key/value variance across tokens -> large J_t + R_t error if approximated by mean
    k[11 * 64 : 12 * 64, :, 0] = 1.00
    signs = torch.tensor([1.0 if (i % 2 == 0) else -1.0 for i in range(64)]).view(
        64, 1, 1
    )
    k[11 * 64 : 12 * 64, :, 1:33] = signs * 2.5
    v[11 * 64 : 12 * 64, :, 1:33] = signs * 2.5

    _, m_s0_base, _ = sparse_mod.fused_route_and_build_masks(
        q_first=q_first,
        q_second=q_second,
        k=k,
        v=v,
        layer_idx=10,
        tau="0.95",
        L_q=L_q,
        M_q=M_q,
        M_cand=M_cand,
        M_prefix=M_prefix,
        safety_scale=1.0,
        build_src=False,
        L_k_src=0,
        M_k_src=0,
        M_cur_src=0,
        build_s0=True,
        L_k_s0=L_k,
        M_k_s0=M_k,
        M_cur_s0=M_cur,
        build_s1=False,
        s_evict=0,
        L_k_s1=0,
        M_k_s1=0,
        M_cand_s1=0,
        M_cur_s1=0,
        head_start=0,
        num_heads=H,
        enable_triage_score=False,
    )
    assert m_s0_base is not None
    # Without Triage complexity score, uniform Block 10 ranks ahead of high-variance Block 11
    m_full_s0 = M_k - 1 - M_cand
    assert int(m_s0_base.cand_idx[0, 0, m_full_s0].item()) == 10

    _, m_s0_triage, _ = sparse_mod.fused_route_and_build_masks(
        q_first=q_first,
        q_second=q_second,
        k=k,
        v=v,
        layer_idx=10,
        tau="0.95",
        L_q=L_q,
        M_q=M_q,
        M_cand=M_cand,
        M_prefix=M_prefix,
        safety_scale=1.0,
        build_src=False,
        L_k_src=0,
        M_k_src=0,
        M_cur_src=0,
        build_s0=True,
        L_k_s0=L_k,
        M_k_s0=M_k,
        M_cur_s0=M_cur,
        build_s1=False,
        s_evict=0,
        L_k_s1=0,
        M_k_s1=0,
        M_cand_s1=0,
        M_cur_s1=0,
        head_start=0,
        num_heads=H,
        enable_triage_score=True,
    )
    assert m_s0_triage is not None
    # With Triage complexity score (J_t + R_t + 0.05), high-variance Block 11 ranks #1!
    assert int(m_s0_triage.cand_idx[0, 0, m_full_s0].item()) == 11

    # Verify Top-K-Free price_pi thresholding prunes low-probability blocks to -1 while keeping >= 4 candidates
    _, m_s0_pi, _ = sparse_mod.fused_route_and_build_masks(
        q_first=q_first,
        q_second=q_second,
        k=k,
        v=v,
        layer_idx=10,
        tau="0.95",
        L_q=L_q,
        M_q=M_q,
        M_cand=M_cand,
        M_prefix=M_prefix,
        safety_scale=1.0,
        build_src=False,
        L_k_src=0,
        M_k_src=0,
        M_cur_src=0,
        build_s0=True,
        L_k_s0=L_k,
        M_k_s0=M_k,
        M_cur_s0=M_cur,
        build_s1=False,
        s_evict=0,
        L_k_s1=0,
        M_k_s1=0,
        M_cand_s1=0,
        M_cur_s1=0,
        head_start=0,
        num_heads=H,
        enable_triage_score=True,
        price_pi=0.20,
    )
    assert m_s0_pi is not None
    cnt0 = int(m_s0_pi.cand_cnt[0].item())
    row0 = m_s0_pi.cand_idx[0, 0, :cnt0].tolist()
    valid_cands = [b for b in row0[m_full_s0:] if b >= 0]
    pruned_cands = [b for b in row0[m_full_s0:] if b == -1]
    assert len(valid_cands) >= 4
    assert len(pruned_cands) > 0

    # Verify execute_branch_attention end-to-end with mode="triage_095"
    sp_triage = sparse_mod.SparseBranchSpec(
        mode="triage_095",
        tau="0.95",
        case_id="triage_unit",
        chunk_id=8,
        b_phase="active",
        step_idx=0,
        step_1000=1000,
        q_start=0,
        q_stop=L_q,
        L_q=L_q,
        L_k=L_k,
        s_cur_chunk=M_cand * 64,
        M_q=M_q,
        M_k=M_k,
        M_cand=M_cand,
        M_cur=M_cur,
        M_prefix=M_prefix,
    )
    sparse_mod.clear_sparse_caches()
    out_triage = sparse_mod.execute_branch_attention(
        None,
        q_second,
        k,
        v,
        scale,
        metadata=None,
        optimized=False,
        layer_idx=10,
        spec=sp_triage,
        fixed_tensors={"cu_k.0": torch.tensor([0, L_k], dtype=torch.int32)},
        branch=0,
    )
    assert out_triage.shape == (L_q, H, D)
    assert torch.isfinite(out_triage).all()


def test_step2_incremental_kv_stats_and_prebuilt_mask_reuse():
    """Verify Step 2 incremental compute_kv_block_stats(start_block=...) and precomputed KV stats reuse on PrebuiltSparseMask."""
    torch.manual_seed(404)
    H, D = 4, 128
    M_cand = 12
    L_k = (M_cand + 2) * 64
    k = torch.randn(L_k, H, D, dtype=torch.bfloat16)
    v = torch.randn(L_k, H, D, dtype=torch.bfloat16)

    ws_full = sparse_mod._StaticRouterWorkspace(
        k.device, num_heads=H, max_m_q=4, max_m_cand=16
    )
    km_full, vm_full, tk_full, rk_full = sparse_mod.compute_kv_block_stats(
        k, v, num_blocks=M_cand, start_block=0, ws=ws_full
    )
    km_full = km_full.clone()
    vm_full = vm_full.clone()
    tk_full = tk_full.clone()
    rk_full = rk_full.clone()

    ws_inc = sparse_mod._StaticRouterWorkspace(
        k.device, num_heads=H, max_m_q=4, max_m_cand=16
    )
    # First compute blocks [0, 8)
    sparse_mod.compute_kv_block_stats(k, v, num_blocks=8, start_block=0, ws=ws_inc)
    # Incrementally compute newly committed blocks [8, 12)
    km_inc, vm_inc, tk_inc, rk_inc = sparse_mod.compute_kv_block_stats(
        k, v, num_blocks=M_cand, start_block=8, ws=ws_inc
    )
    assert torch.equal(km_full, km_inc)
    assert torch.equal(vm_full, vm_inc)
    assert torch.allclose(tk_full, tk_inc, atol=1e-6)
    assert torch.allclose(rk_full, rk_inc, atol=1e-6)

    # Verify fused_route_and_build_masks attaches k_blk_mean, v_blk_mean, tau_k to PrebuiltSparseMask
    q0 = torch.randn(64, H, D, dtype=torch.bfloat16)
    q1 = torch.randn(64, H, D, dtype=torch.bfloat16)
    m_src, m_s0, m_s1 = sparse_mod.fused_route_and_build_masks(
        q_first=q0,
        q_second=q1,
        k=k,
        v=v,
        layer_idx=10,
        tau="0.90",
        L_q=64,
        M_q=1,
        M_cand=M_cand,
        M_prefix=2,
        safety_scale=1.0,
        build_src=True,
        L_k_src=L_k,
        M_k_src=M_cand + 2,
        M_cur_src=2,
        build_s0=True,
        L_k_s0=L_k,
        M_k_s0=M_cand + 2,
        M_cur_s0=2,
        build_s1=True,
        s_evict=0,
        L_k_s1=L_k,
        M_k_s1=M_cand + 2,
        M_cand_s1=M_cand,
        M_cur_s1=2,
        head_start=0,
        num_heads=H,
        enable_triage_score=True,
        enable_virtual_mean=True,
        price_pi=0.05,
        fast_pi_filter=True,
    )
    assert m_src is not None and m_s0 is not None and m_s1 is not None
    assert m_src.k_blk_mean is not None
    assert m_s0.k_blk_mean is not None
    assert m_s1.k_blk_mean is not None


def test_ffn_qkv_quant_scope_selection():
    """Verify ffn-qkv selects 50-layer qkv_proj, fc1, fc2 and keeps out_proj BF16."""
    quant_path = MODULE_PATH.parent / "zing_h3_quant.py"
    assert quant_path.exists(), f"Missing {quant_path}"
    tree = ast.parse(quant_path.read_text(encoding="utf-8"))
    ns = {"fnmatch": __import__("fnmatch")}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "selected_linears":
            code = compile(
                ast.Module(body=[node], type_ignores=[]), str(quant_path), "exec"
            )
            exec(code, ns)
            break
    selected_linears = ns["selected_linears"]

    sel = selected_linears(50, "ffn-qkv")
    assert len(sel) == 50 * 3
    for i in range(50):
        assert f"blocks.{i}.attn.qkv_proj" in sel
        assert f"blocks.{i}.mlp.fc1" in sel
        assert f"blocks.{i}.mlp.fc2" in sel
        assert f"blocks.{i}.attn.out_proj" not in sel

    sel_ex = selected_linears(50, "ffn-qkv", excludes=["blocks.0.*"])
    assert len(sel_ex) == 49 * 3
    assert "blocks.0.attn.qkv_proj" not in sel_ex


def test_compressed_kv_store_chunk_boundary_and_memory_footprint():
    """Verify ZingCompressedKVStore chunk-boundary block splitting, 18.2 KB/blk footprint, and tail padding."""
    torch.manual_seed(505)
    H, D = 4, 128
    store = sparse_mod.ZingCompressedKVStore(
        max_blocks=32, num_heads=H, head_dim=D, device=torch.device("cpu")
    )
    # Verify per-block-per-head byte footprint is 18,245 B (17.82 KB vs 49,152 B raw = -62.9%)
    bytes_per_blk_head = store.bytes_per_block_per_head()
    raw_bytes_per_blk_head = 3 * 64 * 128 * 2  # 49,152 B
    assert bytes_per_blk_head == 18245, f"Expected 18245 B, got {bytes_per_blk_head}"
    reduction = 1.0 - (bytes_per_blk_head / raw_bytes_per_blk_head)
    assert reduction > 0.62, f"Expected >62% reduction, got {reduction:.4f}"

    # Verify actual allocated tensor bytes match bytes_per_block_per_head()
    per_head_blk_tensors = [
        store.k_res_fp8,
        store.v_res_fp8,
        store.k_res_scale,
        store.v_res_scale,
        store.k_blk_mean,
        store.v_blk_mean,
        store.k_sub_mean_fp8,
        store.v_sub_mean_fp8,
        store.k_sub_scale,
        store.v_sub_scale,
        store.k_blk_mean_fp8,
        store.v_blk_mean_fp8,
        store.k_blk_scale,
        store.v_blk_scale,
        store.tau_k,
        store.rho_kv,
    ]
    per_blk_meta_tensors = [
        store.block_valid_len,
        store.block_delta_t,
        store.block_is_shifted,
        store.block_chunk_id,
    ]
    actual_bytes = (
        sum(t.numel() * t.element_size() for t in per_head_blk_tensors)
        // (store.max_blocks * H)
        + sum(t.numel() * t.element_size() for t in per_blk_meta_tensors)
        // store.max_blocks
    )
    assert actual_bytes == bytes_per_blk_head, (
        f"Actual tensor bytes {actual_bytes} != reported {bytes_per_blk_head}"
    )

    # Append Chunk 0 (prefix, length 100 -> 2 blocks: 64 and 36 valid rows, never crossing chunk boundary)
    k_c0 = torch.randn(100, H, D, dtype=torch.bfloat16)
    v_c0 = torch.randn(100, H, D, dtype=torch.bfloat16)
    b0_start, b0_end = store.append_chunk(k_c0, v_c0, chunk_id=-1)
    assert (b0_start, b0_end) == (0, 2)
    assert store.num_blocks == 2
    assert store.block_valid_len[:2].tolist() == [64, 36]
    assert store.block_chunk_id[:2].tolist() == [-1, -1]
    # Verify tail rows 36..63 in block 1 have zero FP8 residuals
    assert float(store.k_res_fp8[1, :, 36:, :].float().abs().max().item()) == 0.0
    assert float(store.v_res_fp8[1, :, :, 36:].float().abs().max().item()) == 0.0

    # Append Chunk 1 (sink chunk 0, length 80 -> starts at block 2, occupies blocks 2..4 with valid lens [64, 16])
    k_c1 = torch.randn(80, H, D, dtype=torch.bfloat16)
    v_c1 = torch.randn(80, H, D, dtype=torch.bfloat16)
    b1_start, b1_end = store.append_chunk(k_c1, v_c1, chunk_id=0)
    assert (b1_start, b1_end) == (2, 4)
    assert store.num_blocks == 4
    assert store.block_valid_len[:4].tolist() == [64, 36, 64, 16]
    assert store.block_chunk_id[:4].tolist() == [-1, -1, 0, 0]


def test_compressed_kv_store_temporal_rope_homomorphism_on_window_shift():
    """Verify Theorem 3: sliding-window shift rotates block/sub-block means via 32D temporal RoPE without touching FP8 residuals."""
    torch.manual_seed(506)
    H, D = 2, 128
    store = sparse_mod.ZingCompressedKVStore(
        max_blocks=16, num_heads=H, head_dim=D, device=torch.device("cpu")
    )
    # Chunk 0 (prefix, 64 tokens) + Chunk 1 (window chunk, 100 tokens -> 2 blocks: 64 and 36)
    k_pref = torch.randn(64, H, D, dtype=torch.bfloat16)
    v_pref = torch.randn(64, H, D, dtype=torch.bfloat16)
    store.append_chunk(k_pref, v_pref, chunk_id=-1)

    k_win_raw = torch.randn(100, H, D, dtype=torch.bfloat16)
    v_win = torch.randn(100, H, D, dtype=torch.bfloat16)
    t0 = 10.0
    k_win_t0 = sparse_mod.apply_temporal_rope_32d(k_win_raw, t0).to(torch.bfloat16)
    b_start, b_end = store.append_chunk(k_win_t0, v_win, chunk_id=5)
    assert (b_start, b_end) == (1, 3)

    # Snapshot FP8 residual tensors before window shift
    k_res_before = store.k_res_fp8[:3].clone()
    v_res_before = store.v_res_fp8[:3].clone()

    # Apply sliding-window temporal shift delta_t = -3.0 to chunk_id=5 (blocks 1, 2)
    delta_t = -3.0
    store.apply_window_shift(block_indices=[1, 2], delta_t=delta_t)

    # 1. Historical FP8 residuals must be byte-for-byte untouched!
    assert torch.equal(
        store.k_res_fp8[:3].view(torch.uint8), k_res_before.view(torch.uint8)
    )
    assert torch.equal(
        store.v_res_fp8[:3].view(torch.uint8), v_res_before.view(torch.uint8)
    )
    assert store.block_is_shifted[:3].tolist() == [0, 1, 1]
    assert abs(float(store.window_delta_t) - delta_t) < 1e-6

    # 2. Rotated k_blk_mean must match fresh extraction from R_t(t0 + delta_t) k_win_raw!
    k_win_shifted = sparse_mod.apply_temporal_rope_32d(k_win_raw, t0 + delta_t).to(
        torch.bfloat16
    )
    ref_store = sparse_mod.ZingCompressedKVStore(
        max_blocks=16, num_heads=H, head_dim=D, device=torch.device("cpu")
    )
    ref_store.append_chunk(k_pref, v_pref, chunk_id=-1)
    ref_store.append_chunk(k_win_shifted, v_win, chunk_id=5)

    assert torch.allclose(
        store.k_blk_mean[:, 1:3, :].float(),
        ref_store.k_blk_mean[:, 1:3, :].float(),
        atol=2e-2,
    )

    # 3. Verify attention with inverse-rotated Q_win matches fresh shifted K!
    q = torch.randn(64, H, D, dtype=torch.bfloat16)
    q_win = sparse_mod.apply_temporal_rope_32d(q, delta_t, inverse=True).to(
        torch.bfloat16
    )
    scale = 1.0 / math.sqrt(D)
    cand_cnt = torch.full((H,), 3, dtype=torch.int32)
    cand_idx = torch.tensor([0, 1, 2], dtype=torch.int32).view(1, 1, 3).expand(H, 1, 3)
    skim_cnt = torch.zeros((H,), dtype=torch.int32)
    skim_idx = torch.zeros((H, 1, 1), dtype=torch.int32)
    head_order = torch.arange(H, dtype=torch.int32)

    out_incremental = sparse_mod.launch_zing_block_sparse_attn(
        q,
        None,
        None,
        scale,
        3 * 64,
        3,
        3,
        cand_cnt,
        cand_idx,
        head_order,
        skim_cnt=skim_cnt,
        skim_idx=skim_idx,
        compressed_kv=store.as_kernel_dict(),
        q_win=q_win,
        enable_three_tier=True,
    )
    out_fresh = sparse_mod.launch_zing_block_sparse_attn(
        q,
        None,
        None,
        scale,
        3 * 64,
        3,
        3,
        cand_cnt,
        cand_idx,
        head_order,
        skim_cnt=skim_cnt,
        skim_idx=skim_idx,
        compressed_kv=ref_store.as_kernel_dict(),
        q_win=None,
        enable_three_tier=True,
    )
    cos_sim = float(
        torch.nn.functional.cosine_similarity(
            out_incremental.float().reshape(1, -1),
            out_fresh.float().reshape(1, -1),
        ).item()
    )
    rel_l2 = float(
        (out_incremental.float() - out_fresh.float()).norm().item()
        / out_fresh.float().norm().clamp_min(1e-6).item()
    )
    assert cos_sim > 0.999, f"Expected cos_sim > 0.999, got {cos_sim}"
    assert rel_l2 < 0.03, f"Expected rel_l2 < 0.03, got {rel_l2}"


def test_three_tier_router_and_100pct_refine_dense_protection():
    """Verify Three-Tier Single-Price pi routing (Refine/Skim/Skip) and 100% Refine Dense Protection on Layer 0 & Warmup."""
    torch.manual_seed(507)
    H, D = 4, 128
    L_q, M_q = 64, 1
    M_cand = 12
    M_cur = 2
    M_k = M_cand + M_cur
    L_k = M_k * 64
    scale = 1.0 / math.sqrt(D)

    q = torch.zeros(L_q, H, D, dtype=torch.bfloat16)
    q[:, :, 0] = 1.0
    k = torch.zeros(L_k, H, D, dtype=torch.bfloat16)
    v = torch.zeros(L_k, H, D, dtype=torch.bfloat16)

    # Blocks 0..3: uniform low-variance -> Skip
    k[: 4 * 64, :, 0] = 0.2
    v[: 4 * 64, :, 0] = 0.2
    # Blocks 4..7: medium variance -> Skim (pi < J_b + R_b <= c_rf * pi)
    signs_16 = (
        torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.bfloat16)
        .repeat_interleave(16)
        .view(64, 1, 1)
    )
    for b in range(4, 8):
        k[b * 64 : (b + 1) * 64, :, 0] = 0.8
        k[b * 64 : (b + 1) * 64, :, 1:9] = signs_16 * 0.45
        v[b * 64 : (b + 1) * 64, :, 1:9] = signs_16 * 0.45
    # Blocks 8..11: high intra-block variance -> Refine (J_b + R_b > c_rf * pi)
    signs_alt = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(64)], dtype=torch.bfloat16
    ).view(64, 1, 1)
    for b in range(8, 12):
        k[b * 64 : (b + 1) * 64, :, 0] = 1.2
        k[b * 64 : (b + 1) * 64, :, 1:33] = signs_alt * 2.5
        v[b * 64 : (b + 1) * 64, :, 1:33] = signs_alt * 2.5

    # Active blocks 12..13
    k[12 * 64 :, :, 0] = 0.5
    v[12 * 64 :, :, 0] = 0.5

    _, m_s0_3t, _ = sparse_mod.fused_route_and_build_masks(
        q_first=q,
        q_second=q,
        k=k,
        v=v,
        layer_idx=10,
        tau="0.95",
        L_q=L_q,
        M_q=M_q,
        M_cand=M_cand,
        M_prefix=2,
        safety_scale=1.0,
        build_src=False,
        L_k_src=0,
        M_k_src=0,
        M_cur_src=0,
        build_s0=True,
        L_k_s0=L_k,
        M_k_s0=M_k,
        M_cur_s0=M_cur,
        build_s1=False,
        s_evict=0,
        L_k_s1=0,
        M_k_s1=0,
        M_cand_s1=0,
        M_cur_s1=0,
        head_start=0,
        num_heads=H,
        enable_triage_score=True,
        enable_virtual_mean=True,
        enable_three_tier=True,
        price_pi=0.02,
        c_rf=16.0,
    )
    assert m_s0_3t is not None
    assert m_s0_3t.skim_cnt is not None
    assert m_s0_3t.skim_idx is not None
    assert int(m_s0_3t.skim_cnt[0].item()) > 0
    refine_blks = set(
        b
        for b in m_s0_3t.cand_idx[0, 0, : int(m_s0_3t.cand_cnt[0].item())].tolist()
        if b >= 0
    )
    skim_blks = set(
        b
        for b in m_s0_3t.skim_idx[0, 0, : int(m_s0_3t.skim_cnt[0].item())].tolist()
        if b >= 0
    )
    # High-variance blocks 8..11 must be in Refine, medium-variance blocks 4..7 in Skim, disjoint!
    assert {8, 9, 10, 11}.issubset(refine_blks)
    assert len(skim_blks.intersection({4, 5, 6, 7})) >= 2
    assert len(refine_blks.intersection(skim_blks)) == 0

    # Verify alpha_triage sensitivity: tighter alpha_triage=0.08 caps Skim count at ceil(0.08 * 12) = 1
    _, m_s0_tight, _ = sparse_mod.fused_route_and_build_masks(
        q_first=q,
        q_second=q,
        k=k,
        v=v,
        layer_idx=11,
        tau="0.95",
        L_q=L_q,
        M_q=M_q,
        M_cand=M_cand,
        M_prefix=2,
        safety_scale=1.0,
        build_src=False,
        L_k_src=0,
        M_k_src=0,
        M_cur_src=0,
        build_s0=True,
        L_k_s0=L_k,
        M_k_s0=M_k,
        M_cur_s0=M_cur,
        build_s1=False,
        s_evict=0,
        L_k_s1=0,
        M_k_s1=0,
        M_cand_s1=0,
        M_cur_s1=0,
        head_start=0,
        num_heads=H,
        enable_triage_score=True,
        enable_virtual_mean=True,
        enable_three_tier=True,
        price_pi=0.02,
        c_rf=16.0,
        alpha_triage=0.08,
    )
    assert m_s0_tight is not None and m_s0_tight.skim_cnt is not None
    assert int(m_s0_tight.skim_cnt[0].item()) <= 1

    # Verify Dense Protection = 100% Refine on Layer 0 and Warmup Chunk (chunk_id=1 < min_chunk=3)
    sparse_mod.clear_sparse_caches()
    spec_l0 = sparse_mod.SparseBranchSpec(
        mode="all_fp8_3tier",
        tau="0.95",
        case_id="3tier_unit",
        chunk_id=8,
        b_phase="active",
        step_idx=0,
        step_1000=1000,
        q_start=0,
        q_stop=L_q,
        L_q=L_q,
        L_k=L_k,
        s_cur_chunk=M_cand * 64,
        M_q=M_q,
        M_k=M_k,
        M_cand=M_cand,
        M_cur=M_cur,
        M_prefix=2,
    )
    out_l0 = sparse_mod.execute_branch_attention(
        None,
        q,
        k,
        v,
        scale,
        metadata=None,
        optimized=False,
        layer_idx=0,  # Layer 0 -> 100% Refine through 3-tier All-FP8 kernel!
        spec=spec_l0,
        fixed_tensors={"cu_k.0": torch.tensor([0, L_k], dtype=torch.int32)},
        branch=0,
    )
    assert out_l0.shape == (L_q, H, D)
    assert torch.isfinite(out_l0).all()
    assert (
        sparse_mod._LAST_3TIER_EXEC_META.get("dense_protection_100pct_refine") is True
    )
    assert sparse_mod._LAST_3TIER_EXEC_META.get("layer_idx") == 0

    # Warmup chunk (chunk_id = 1 < min_chunk = 3) -> also 100% Refine!
    spec_warmup = sparse_mod.SparseBranchSpec(
        mode="all_fp8_3tier",
        tau="0.95",
        case_id="3tier_unit",
        chunk_id=1,
        b_phase="active",
        step_idx=0,
        step_1000=1000,
        q_start=0,
        q_stop=L_q,
        L_q=L_q,
        L_k=L_k,
        s_cur_chunk=M_cand * 64,
        M_q=M_q,
        M_k=M_k,
        M_cand=M_cand,
        M_cur=M_cur,
        M_prefix=2,
    )
    out_warmup = sparse_mod.execute_branch_attention(
        None,
        q,
        k,
        v,
        scale,
        metadata=None,
        optimized=False,
        layer_idx=5,
        spec=spec_warmup,
        fixed_tensors={"cu_k.0": torch.tensor([0, L_k], dtype=torch.int32)},
        branch=0,
    )
    assert out_warmup.shape == (L_q, H, D)
    assert torch.isfinite(out_warmup).all()
    assert (
        sparse_mod._LAST_3TIER_EXEC_META.get("dense_protection_100pct_refine") is True
    )
    assert sparse_mod._LAST_3TIER_EXEC_META.get("chunk_id") == 1


def test_frame_2d_square_partition_and_vectorized_total_variance_three_tier():
    """Verify SPEC Rev 7 single-frame 2D square partition, vectorized total variance stats, and GPU-vectorized 3-tier routing."""
    import torch.nn.functional as F

    torch.manual_seed(20261103)

    # 1. 736p (23x40 = 920 tokens/frame, 3 frames = 2760 tokens -> 45 blocks)
    part736 = sparse_mod.build_frame_2d_square_partition(23, 40, num_frames=3)
    assert part736.tokens_per_frame == 920
    assert part736.blocks_per_frame == 15
    assert part736.num_blocks == 45
    expected_736_lens = ([64] * 10 + [56] * 5) * 3
    assert part736.block_valid_len.tolist() == expected_736_lens
    assert int(part736.block_valid_len.sum().item()) == 2760
    assert torch.equal(
        part736.perm[part736.inv_perm], torch.arange(2760, dtype=torch.long)
    )
    assert torch.equal(
        part736.inv_perm[part736.perm], torch.arange(2760, dtype=torch.long)
    )

    # Check zero cross-frame mixing & quadrant-major 4x4 compactness on 736p
    offset = 0
    for b_idx, n_val in enumerate(expected_736_lens):
        blk_flat = part736.perm[offset : offset + n_val]
        offset += n_val
        frames = blk_flat // 920
        assert torch.all(frames == (b_idx // 15)), (
            f"Cross-frame leak in 736p block {b_idx}"
        )
        in_frame = blk_flat % 920
        rows = in_frame // 40
        cols = in_frame % 40
        assert int((rows.max() - rows.min()).item()) < 8
        assert int((cols.max() - cols.min()).item()) < 8
        if n_val == 64:
            for s_i in range(4):
                q_rows = rows[s_i * 16 : (s_i + 1) * 16]
                q_cols = cols[s_i * 16 : (s_i + 1) * 16]
                assert int((q_rows.max() - q_rows.min()).item()) == 3
                assert int((q_cols.max() - q_cols.min()).item()) == 3

    # 2. 480p (15x26 = 390 tokens/frame, 3 frames = 1170 tokens -> 21 blocks)
    part480 = sparse_mod.build_frame_2d_square_partition(15, 26, num_frames=3)
    assert part480.tokens_per_frame == 390
    assert part480.blocks_per_frame == 7
    assert part480.num_blocks == 21
    expected_480_lens = [52, 52, 52, 52, 61, 61, 60] * 3
    assert part480.block_valid_len.tolist() == expected_480_lens
    assert int(part480.block_valid_len.sum().item()) == 1170
    assert torch.equal(
        part480.perm[part480.inv_perm], torch.arange(1170, dtype=torch.long)
    )
    assert torch.equal(
        part480.inv_perm[part480.perm], torch.arange(1170, dtype=torch.long)
    )
    offset = 0
    for b_idx, n_val in enumerate(expected_480_lens):
        blk_flat = part480.perm[offset : offset + n_val]
        offset += n_val
        frames = blk_flat // 390
        assert torch.all(frames == (b_idx // 7)), (
            f"Cross-frame leak in 480p block {b_idx}"
        )
        in_frame = blk_flat % 390
        rows = in_frame // 26
        cols = in_frame % 26
        assert int((rows.max() - rows.min()).item()) < 8
        assert int((cols.max() - cols.min()).item()) <= 9

    # 3. ZingCompressedKVStore vectorized Law of Total Variance across non-64 2D partition blocks
    H, D = 4, 128
    store = sparse_mod.ZingCompressedKVStore(max_blocks=16, num_heads=H, head_dim=D)
    assert store.bytes_per_block_per_head() == 18245
    k_frame = torch.randn((390, H, D), dtype=torch.bfloat16)
    v_frame = torch.randn((390, H, D), dtype=torch.bfloat16)
    k_perm = (
        part480.permute_tokens(k_frame)
        if hasattr(part480, "permute_tokens")
        else k_frame[part480.perm[:390]]
    )
    v_perm = (
        part480.permute_tokens(v_frame)
        if hasattr(part480, "permute_tokens")
        else v_frame[part480.perm[:390]]
    )
    b_s, b_e = store.append_chunk(
        k_perm,
        v_perm,
        chunk_id=0,
        block_valid_lengths=part480.block_valid_len[:7],
    )
    assert (b_s, b_e) == (0, 7)
    ckv_480 = store.as_kernel_dict()
    assert ckv_480["block_valid_len"].tolist() == [52, 52, 52, 52, 61, 61, 60]
    assert "tau_k_between" in ckv_480
    assert "tau_k_within" in ckv_480
    assert "alpha_k" in ckv_480
    assert torch.allclose(
        ckv_480["tau_k"],
        ckv_480["tau_k_between"] + ckv_480["tau_k_within"],
        atol=1e-5,
        rtol=1e-4,
    )
    assert torch.all((ckv_480["alpha_k"] >= 0.05) & (ckv_480["alpha_k"] <= 0.95))

    # 4. Theorem 4 quota-anchored c_rf = 12.0 & vectorized 3-tier routing
    assert sparse_mod.get_sparse_c_rf() == 12.0
    M_cand = 12
    M_cur = 2
    M_k = M_cand + M_cur
    L_k = M_k * 64
    L_q = 128
    M_q = 2
    q_dir = F.normalize(torch.randn((1, 1, D), dtype=torch.float32), dim=-1)
    ortho1 = torch.randn((1, 1, D), dtype=torch.float32)
    ortho1 = F.normalize(
        ortho1 - (ortho1 * q_dir).sum(dim=-1, keepdim=True) * q_dir, dim=-1
    )
    ortho2 = torch.randn((1, 1, D), dtype=torch.float32)
    ortho2 = ortho2 - (ortho2 * q_dir).sum(dim=-1, keepdim=True) * q_dir
    ortho2 = F.normalize(
        ortho2 - (ortho2 * ortho1).sum(dim=-1, keepdim=True) * ortho1, dim=-1
    )

    q = (q_dir * 2.0).expand(L_q, H, D).contiguous().to(torch.bfloat16)
    k = torch.zeros((L_k, H, D), dtype=torch.float32)
    v = torch.randn((L_k, H, D), dtype=torch.float32) * 0.2

    # Block 0: high intra-sub-block variance (within >> between, alpha_k ~ 0.95) -> Refine
    alt_signs = (torch.arange(64, dtype=torch.float32) % 2 * 2.0 - 1.0).view(64, 1, 1)
    k[0:64] = q_dir * 3.0 + alt_signs * ortho1 * 2.5

    # Block 1: piecewise-constant sub-blocks (within = 0, between > 0, alpha_k = 0.05) -> Skim
    sub_signs = (
        torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32)
        .repeat_interleave(16)
        .view(64, 1, 1)
    )
    k[64:128] = q_dir * 3.0 + sub_signs * ortho2 * 2.0

    # Block 2: completely flat constant block (tau_k_total = 0) -> Skip
    k[128:192] = q_dir * 1.5

    # Remaining candidate blocks 3..11: moderate random background
    for b_bg in range(3, M_cand):
        s_b = b_bg * 64
        k[s_b : s_b + 64] = q_dir * (1.2 - 0.08 * b_bg) + torch.randn((64, H, D)) * 0.3

    k_bf16 = k.to(torch.bfloat16)
    v_bf16 = v.to(torch.bfloat16)
    store_3t = sparse_mod.ZingCompressedKVStore(max_blocks=M_k, num_heads=H, head_dim=D)
    store_3t.append_chunk(k_bf16, v_bf16, chunk_id=0)
    ckv_3t = store_3t.as_kernel_dict(num_blocks=M_cand)

    assert torch.all(ckv_3t["alpha_k"][:, 0] > 0.85)
    assert torch.all(ckv_3t["alpha_k"][:, 1] < 0.10)

    _, mask_s0, _ = sparse_mod.fused_route_and_build_masks(
        q_first=q,
        q_second=None,
        k=k_bf16,
        layer_idx=5,
        tau="0.95",
        L_q=L_q,
        M_q=M_q,
        M_cand=M_cand,
        M_prefix=2,
        safety_scale=1.0,
        build_src=False,
        L_k_src=0,
        M_k_src=0,
        M_cur_src=0,
        build_s0=True,
        L_k_s0=L_k,
        M_k_s0=M_k,
        M_cur_s0=M_cur,
        build_s1=False,
        s_evict=0,
        L_k_s1=0,
        M_k_s1=0,
        M_cand_s1=0,
        M_cur_s1=0,
        num_heads=H,
        v=v_bf16,
        enable_triage_score=True,
        enable_virtual_mean=True,
        enable_three_tier=True,
        price_pi=0.0,  # Quota-anchored S_(K)/c_rf
        c_rf=12.0,
        compressed_kv=ckv_3t,
    )
    assert mask_s0 is not None
    m_full_cur = max(0, M_k - 1 - M_cand)
    n_full_0 = int(mask_s0.cand_cnt[0].item())
    n_skim_0 = int(mask_s0.skim_cnt[0].item())
    ref_0 = {
        int(x)
        for x in mask_s0.cand_idx[0, 0, m_full_cur:n_full_0].tolist()
        if 0 <= int(x) < M_cand
    }
    sk_0 = {
        int(x)
        for x in mask_s0.skim_idx[0, 0, :n_skim_0].tolist()
        if 0 <= int(x) < M_cand
    }
    assert ref_0.isdisjoint(sk_0)
    assert 0 in ref_0, f"Expected high-within-variance block 0 in Refine, got {ref_0}"
    assert 1 in sk_0 and 1 not in ref_0, (
        f"Expected piecewise-constant quadrant block 1 demoted to Skim, got ref={ref_0}, sk={sk_0}"
    )
    assert 2 not in ref_0 and 2 not in sk_0, (
        f"Expected flat zero-variance block 2 routed to Skip, got ref={ref_0}, sk={sk_0}"
    )


def test_attach_sparse_specs_recognizes_refreshed_active_sinks():
    """Verify attach_sparse_specs_to_plan marks refreshed active_sinks (e.g. (0, 23, 24)) as KIND_SINK."""
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

    pref = SimpleNamespace(
        offset=0,
        positions=torch.zeros((512, 3), dtype=torch.int64),
        position_signature="sig_sink_refresh",
        host_positions=torch.zeros((512, 3), dtype=torch.int64),
    )
    sparse_mod._PREFIX_ROLE4_BY_SIG["sig_sink_refresh"] = ((256, 512),)
    try:
        # 3 refreshed sinks: u=0 (384 rows, 6 blks), u=23 (256 rows, 4 blks), u=24 (384 rows, 6 blks) -> total 16 sink blks (blks 8..24)
        # 1 sliding window chunk: u=25 (src 384 rows + fin 384 rows = 768 rows, blks 24..36)
        pending_src_u26 = SimpleNamespace(
            offset=2304,
            positions=torch.zeros((384, 3), dtype=torch.int64),
            block=53,  # u = 26
        )

        for sinks_tuple, s1_id, s1_rows, s2_id, s2_rows in [
            (
                (0, 2, 23),
                2,
                384,
                23,
                256,
            ),  # STEP1_ODD transition: (0, 2, u_odd) -> 384 + 384 + 256 = 1024 rows (16 blks)
            (
                (0, 23, 24),
                23,
                256,
                24,
                384,
            ),  # STEP2_EVEN steady:    (0, u_odd, u_odd+1) -> 384 + 256 + 384 = 1024 rows (16 blks)
        ]:

            def _mock_history(
                s1_id=s1_id, s1_rows=s1_rows, s2_id=s2_id, s2_rows=s2_rows
            ):
                return [
                    pref,
                    SimpleNamespace(
                        offset=512,
                        positions=torch.zeros((384, 3), dtype=torch.int64),
                        temporal_chunk_id=0,
                        rope_parts=((2, 384),),
                    ),
                    SimpleNamespace(
                        offset=896,
                        positions=torch.zeros((s1_rows, 3), dtype=torch.int64),
                        temporal_chunk_id=s1_id,
                        rope_parts=((s1_id * 2 + 2, s1_rows),),
                    ),
                    SimpleNamespace(
                        offset=896 + s1_rows,
                        positions=torch.zeros((s2_rows, 3), dtype=torch.int64),
                        temporal_chunk_id=s2_id,
                        rope_parts=((s2_id * 2 + 2, s2_rows),),
                    ),
                    SimpleNamespace(
                        offset=1536,
                        positions=torch.zeros((768, 3), dtype=torch.int64),
                        temporal_chunk_id=25,
                        rope_parts=((51, 384), (52, 384)),
                    ),
                    pending_src_u26,
                ]

            policy = ZingWindowPolicy(21, 3, active_sinks=sinks_tuple)
            shadow = SimpleNamespace(
                prefix=pref,
                pending_source=pending_src_u26,
                _history=_mock_history,
                window_policy=policy,
            )
            hist_items = sparse_mod.extract_history_items(shadow)
            kv_plan = SimpleNamespace(write_start=2688, ends=[384, 768, 1152])
            block_idx = torch.cat(
                [
                    torch.full(
                        (384,), 54, dtype=torch.int64
                    ),  # final u=26: 2688..3072 (blks 42..48)
                    torch.full(
                        (384,), 55, dtype=torch.int64
                    ),  # source u=27: 3072..3456
                    torch.full(
                        (384,), 56, dtype=torch.int64
                    ),  # active u=27: 3456..3840
                ]
            )
            packed = {
                "_zing_host_layout": SimpleNamespace(block_idx=block_idx),
                "_zing_joint_phases": ("final", "source", "active"),
            }
            specs = sparse_mod.attach_sparse_specs_to_plan(
                None,
                hist_items,
                policy,
                packed,
                "joint",
                kv_plan,
                {"mode": "sparse_095", "case_id": "sink_refresh_test"},
                shadow=shadow,
            )
            sp_src = specs[1]
            # Blocks 8..24 (16 blocks = 1024 rows) must all be recognized as KIND_SINK in both STEP1_ODD and STEP2_EVEN!
            assert sp_src.blk_kind_sig == (
                48,
                ((4, 8),),
                (8, 24),
                ((24, 30), (36, 42)),
            ), (
                f"Expected sink span (8, 24) for refreshed active_sinks {sinks_tuple}, got {sp_src.blk_kind_sig}"
            )
    finally:
        sparse_mod._PREFIX_ROLE4_BY_SIG.pop("sig_sink_refresh", None)


def test_product_long_prompt_stationary_sink_and_multi_ref_retention():
    """Verify Case 3A/3B (product-long-prompt) stationary room anchor and multi-reference teacup retention."""
    torch.manual_seed(20261106)
    device = torch.device("cpu")
    H, D = 2, 128
    L_q, M_q = 128, 2
    M_cand, M_prefix = 64, 16
    L_k = (M_cand + M_q) * 64

    base_dir = torch.randn(1, H, D, dtype=torch.float32)
    base_dir = base_dir / base_dir.norm(dim=-1, keepdim=True)
    ref_dom_dir = 0.85 * base_dir + 0.52 * torch.randn(1, H, D, dtype=torch.float32)
    ref_dom_dir = ref_dom_dir / ref_dom_dir.norm(dim=-1, keepdim=True)
    ref_sec_dir = 0.35 * base_dir + 0.93 * torch.randn(1, H, D, dtype=torch.float32)
    ref_sec_dir = ref_sec_dir / ref_sec_dir.norm(dim=-1, keepdim=True)

    q_src = (
        (base_dir + 0.35 * ref_sec_dir + 0.02 * torch.randn(L_q, H, D))
        .to(torch.bfloat16)
        .to(device)
    )
    q_s0 = (
        (0.45 * base_dir + 0.25 * ref_sec_dir + 0.85 * torch.randn(L_q, H, D))
        .to(torch.bfloat16)
        .to(device)
    )

    k = torch.randn(L_k, H, D, dtype=torch.bfloat16, device=device) * 0.05
    # [4..10): dominant character reference (<Picture 2>, high cos ~0.85)
    k[4 * 64 : 10 * 64] = (ref_dom_dir * 0.9 + 0.03 * torch.randn(6 * 64, H, D)).to(
        torch.bfloat16
    )
    # [10..16): secondary subject reference (<Picture 3> teacup/layout, moderate positive cos ~0.35 < mu_cr0)
    k[10 * 64 : 16 * 64] = (ref_sec_dir * 0.9 + 0.03 * torch.randn(6 * 64, H, D)).to(
        torch.bfloat16
    )
    # [16..28): stationary room sink (Sink #0..#2, high positive cos ~0.95, slightly below local window ~0.99 due to RoPE gap)
    sink_dir = 0.95 * base_dir + 0.31 * ref_sec_dir
    sink_dir = sink_dir / sink_dir.norm(dim=-1, keepdim=True)
    k[16 * 64 : 28 * 64] = (sink_dir * 1.15 + 0.01 * torch.randn(12 * 64, H, D)).to(
        torch.bfloat16
    )
    # [28..40): 12 matching spatial blocks in 6-chunk sliding window (tightly clustered cos ~0.99)
    k[28 * 64 : 40 * 64] = (base_dir * 1.0 + 0.01 * torch.randn(12 * 64, H, D)).to(
        torch.bfloat16
    )
    # [40..64): 24 other spatial blocks in 6-chunk sliding window (same direction, lower key norm 0.25 so std_w0 > 0 while std_cw0 ~ 0.01)
    k[40 * 64 : 64 * 64] = (base_dir * 0.25 + 0.005 * torch.randn(24 * 64, H, D)).to(
        torch.bfloat16
    )
    v = torch.randn_like(k)

    # 1. Stationary 6-chunk window: in-distribution Sink #0..#2 (c0 > 0.05) and secondary reference (<Picture 3>) must NOT be evicted!
    blk_kind_sig = (M_cand, ((4, 16),), (16, 28), ((28, 46),))
    mask_src, mask_s0, _ = sparse_mod.fused_route_and_build_masks(
        q_first=q_src,
        q_second=q_s0,
        k=k,
        v=v,
        layer_idx=1,
        tau="0.95",
        L_q=L_q,
        M_q=M_q,
        M_cand=M_cand,
        M_prefix=M_prefix,
        safety_scale=1.0,
        build_src=True,
        L_k_src=L_k,
        M_k_src=M_cand + M_q,
        M_cur_src=M_q,
        build_s0=True,
        L_k_s0=L_k,
        M_k_s0=M_cand + M_q,
        M_cur_s0=M_q,
        build_s1=False,
        s_evict=M_cand * 64,
        L_k_s1=L_k,
        M_k_s1=M_cand + M_q,
        M_cand_s1=M_cand,
        M_cur_s1=M_q,
        head_start=6,
        num_heads=H,
        blk_kind_sig=blk_kind_sig,
        enable_triage_score=True,
    )
    assert mask_src is not None and mask_s0 is not None
    cnt_s0 = int(mask_s0.cand_cnt[0].item())
    sel_s0 = mask_s0.cand_idx[0, :M_q, :cnt_s0]
    sink_hits_s0 = ((sel_s0 >= 16) & (sel_s0 < 28)).float().sum(dim=-1).mean().item()
    sec_ref_hits_s0 = ((sel_s0 >= 10) & (sel_s0 < 16)).float().sum(dim=-1).mean().item()
    assert sink_hits_s0 >= 6.0, (
        f"Stationary in-distribution sink was falsely evicted at Step 0: {sink_hits_s0}"
    )
    assert sec_ref_hits_s0 >= 3.0, (
        f"Secondary reference (<Picture 3> teacup) was masked out: {sec_ref_hits_s0}"
    )

    # 2. At u_cur = 3 (n_win == 0, all committed blocks 16..64 are KIND_SINK), sink must NEVER self-evict!
    blk_kind_sig_u3 = (M_cand, ((4, 16),), (16, 64), ())
    _, mask_s0_u3, _ = sparse_mod.fused_route_and_build_masks(
        q_first=q_src,
        q_second=q_s0,
        k=k,
        v=v,
        layer_idx=1,
        tau="0.95",
        L_q=L_q,
        M_q=M_q,
        M_cand=M_cand,
        M_prefix=M_prefix,
        safety_scale=1.0,
        build_src=True,
        L_k_src=L_k,
        M_k_src=M_cand + M_q,
        M_cur_src=M_q,
        build_s0=True,
        L_k_s0=L_k,
        M_k_s0=M_cand + M_q,
        M_cur_s0=M_q,
        build_s1=False,
        s_evict=M_cand * 64,
        L_k_s1=L_k,
        M_k_s1=M_cand + M_q,
        M_cand_s1=M_cand,
        M_cur_s1=M_q,
        head_start=6,
        num_heads=H,
        blk_kind_sig=blk_kind_sig_u3,
        enable_triage_score=True,
    )
    assert mask_s0_u3 is not None
    cnt_u3 = int(mask_s0_u3.cand_cnt[0].item())
    sel_u3 = mask_s0_u3.cand_idx[0, :M_q, :cnt_u3]
    sink_hits_u3 = ((sel_u3 >= 16) & (sel_u3 < 64)).float().sum(dim=-1).mean().item()
    assert sink_hits_u3 >= 20.0, (
        f"Sink self-evicted when n_win == 0 at u_cur = 3: {sink_hits_u3}"
    )


def test_production_fixed_plan_compressed_kv_lifecycle_and_three_tier_execution():
    """Verify production ZingFixedForwardKVPlan and ZingH3KVCache compressed_kv binding and three-tier execution."""
    try:
        import sglang._platform_stubs

        sglang._platform_stubs.install_platform_stubs()
    except Exception:
        pass
    from sglang.multimodal_gen.runtime.models.dits.zing_h3 import _tensor_layout
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        ZingFixedForwardKVPlan,
        ZingH3KVCache,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        allocate_layer_stores,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

    orig_3t = os.environ.get("SGLANG_ZING_SPARSE_THREE_TIER")
    orig_chunk = os.environ.get("SGLANG_ZING_SPARSE_MIN_CHUNK")
    orig_mode = sparse_mod.get_sparse_mode()

    try:
        os.environ["SGLANG_ZING_SPARSE_THREE_TIER"] = "1"
        os.environ["SGLANG_ZING_SPARSE_MIN_CHUNK"] = "1"
        sparse_mod.set_sparse_mode("dual_query_topk")

        with torch.no_grad():
            stores = allocate_layer_stores(
                8192,
                num_layers=2,
                local_heads=4,
                head_dim=128,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )
            cache = ZingH3KVCache(
                num_layers=2,
                window_policy=ZingWindowPolicy(window_chunks=10, sink_chunks=2),
                storage=stores,
            )
            assert cache._compressed_kv_store is not None
            assert len(cache._compressed_kv_stores) == 2
            assert cache._compressed_kv_stores[0].num_blocks == 0
            assert cache._compressed_kv_stores[1].num_blocks == 0

            # 1. Prefix update with distinct per-layer tensors (128 tokens = 2 blocks)
            pos_p = torch.tensor(
                [[0.0, 0.0, float(i)] for i in range(128)], dtype=torch.float32
            )
            k_p0 = torch.randn(128, 4, 128, dtype=torch.bfloat16)
            v_p0 = torch.randn(128, 4, 128, dtype=torch.bfloat16)
            k_p1 = torch.randn(128, 4, 128, dtype=torch.bfloat16) + 10.0
            v_p1 = torch.randn(128, 4, 128, dtype=torch.bfloat16) + 10.0
            cache.update(
                pos_p,
                torch.zeros(128, dtype=torch.long),
                "prefix",
                0,
                layers=[(k_p0, v_p0), (k_p1, v_p1)],
            )
            assert cache._compressed_kv_stores[0].num_blocks == 2
            assert cache._compressed_kv_stores[1].num_blocks == 2
            # Verify layer 0 and layer 1 compressed statistics are strictly distinct
            assert not torch.allclose(
                cache._compressed_kv_stores[0].k_blk_mean,
                cache._compressed_kv_stores[1].k_blk_mean,
            )

            # 2. Chunk u=1 (128 source + 128 final = 4 blocks -> total 6 blocks <= 8)
            pos_u = torch.tensor(
                [[1.0, 0.0, float(i)] for i in range(128)], dtype=torch.float32
            )
            k_u0 = torch.randn(128, 4, 128, dtype=torch.bfloat16)
            v_u0 = torch.randn(128, 4, 128, dtype=torch.bfloat16)
            k_u1 = torch.randn(128, 4, 128, dtype=torch.bfloat16) + 15.0
            v_u1 = torch.randn(128, 4, 128, dtype=torch.bfloat16) + 15.0
            cache.begin_chunk(2)
            cache.update(
                pos_u,
                torch.full((128,), 1, dtype=torch.long),
                "source",
                1,
                layers=[(k_u0, v_u0), (k_u1, v_u1)],
            )
            cache.update(pos_u, torch.full((128,), 2, dtype=torch.long), "active", 1)
            cache.complete_step0_and_evict(2)
            cache.update(
                pos_u,
                torch.full((128,), 2, dtype=torch.long),
                "final",
                1,
                layers=[(k_u0, v_u0), (k_u1, v_u1)],
            )
            assert cache._compressed_kv_stores[0].num_blocks == 6
            assert cache._compressed_kv_stores[1].num_blocks == 6
            assert not torch.allclose(
                cache._compressed_kv_stores[0].k_blk_mean,
                cache._compressed_kv_stores[1].k_blk_mean,
            )

            # Plan with M_cand <= 8: verify per-layer compressed_kv presence and recursive tensor layout
            pos_next = torch.tensor(
                [[2.0, 0.0, float(i)] for i in range(128)], dtype=torch.float32
            )
            plan = ZingFixedForwardKVPlan(
                cache, pos_next, ends=(128,), clean_rows=0, prefix=False
            )
            assert plan.compressed_kv is not None
            assert hasattr(plan, "compressed_kv_by_layer")
            assert len(plan.compressed_kv_by_layer) == 2
            ckv0 = plan.compressed_kv_by_layer[0]
            ckv1 = plan.compressed_kv_by_layer[1]
            assert not torch.allclose(ckv0["k_blk_mean"], ckv1["k_blk_mean"])

            for expected_key in (
                "k_res_fp8",
                "v_res_fp8",
                "k_blk_mean",
                "v_blk_mean",
                "alpha_k",
                "block_valid_len",
            ):
                assert expected_key in ckv0
                assert expected_key in ckv1

            layout = _tensor_layout(ckv0)
            assert isinstance(layout, tuple)
            assert len(layout) == len(ckv0)

            # Execute with M_cand <= 8 -> triggers 100% refine dense protection
            q = torch.randn(128, 4, 128, dtype=torch.bfloat16)
            k_all_0 = torch.randn(cache._length + 128, 4, 128, dtype=torch.bfloat16)
            v_all_0 = torch.randn(cache._length + 128, 4, 128, dtype=torch.bfloat16)
            k_all_1 = (
                torch.randn(cache._length + 128, 4, 128, dtype=torch.bfloat16) + 10.0
            )
            v_all_1 = (
                torch.randn(cache._length + 128, 4, 128, dtype=torch.bfloat16) + 10.0
            )
            scale = 1.0 / math.sqrt(128)
            res0 = sparse_mod.execute_fixed_block_sparse_branches(
                q,
                k_all_0,
                v_all_0,
                scale,
                index=0,
                fixed=plan,
                fixed_tensors=plan.tensors,
            )
            res1 = sparse_mod.execute_fixed_block_sparse_branches(
                q,
                k_all_1,
                v_all_1,
                scale,
                index=1,
                fixed=plan,
                fixed_tensors=plan.tensors,
            )
            assert (
                sparse_mod._LAST_3TIER_EXEC_META["dense_protection_100pct_refine"]
                is True
            )
            assert sparse_mod._LAST_3TIER_EXEC_META["M_cand"] == 6
            # Assert Layer 0 and Layer 1 outputs differ significantly (no cross-layer store reuse)
            assert not torch.allclose(res0[0], res1[0], atol=1e-2)

            # 3. Add chunks u=2, 3 -> M_cand = 14 > 8 (enters full Three-Tier execution)
            for u in (2, 3):
                pos_u = torch.tensor(
                    [[float(u), 0.0, float(i)] for i in range(128)],
                    dtype=torch.float32,
                )
                k_u0 = torch.randn(128, 4, 128, dtype=torch.bfloat16)
                v_u0 = torch.randn(128, 4, 128, dtype=torch.bfloat16)
                k_u1 = torch.randn(128, 4, 128, dtype=torch.bfloat16) + 15.0
                v_u1 = torch.randn(128, 4, 128, dtype=torch.bfloat16) + 15.0
                tb = u * 2
                cache.begin_chunk(tb)
                cache.update(
                    pos_u,
                    torch.full((128,), tb - 1, dtype=torch.long),
                    "source",
                    1,
                    layers=[(k_u0, v_u0), (k_u1, v_u1)],
                )
                cache.update(
                    pos_u, torch.full((128,), tb, dtype=torch.long), "active", 1
                )
                cache.complete_step0_and_evict(tb)
                cache.update(
                    pos_u,
                    torch.full((128,), tb, dtype=torch.long),
                    "final",
                    1,
                    layers=[(k_u0, v_u0), (k_u1, v_u1)],
                )

            assert cache._compressed_kv_stores[0].num_blocks == 14
            assert cache._compressed_kv_stores[1].num_blocks == 14
            plan = ZingFixedForwardKVPlan(
                cache, pos_next, ends=(128,), clean_rows=0, prefix=False
            )
            k_all_0 = torch.randn(cache._length + 128, 4, 128, dtype=torch.bfloat16)
            v_all_0 = torch.randn(cache._length + 128, 4, 128, dtype=torch.bfloat16)
            k_all_1 = (
                torch.randn(cache._length + 128, 4, 128, dtype=torch.bfloat16) + 10.0
            )
            v_all_1 = (
                torch.randn(cache._length + 128, 4, 128, dtype=torch.bfloat16) + 10.0
            )
            res0_3t = sparse_mod.execute_fixed_block_sparse_branches(
                q,
                k_all_0,
                v_all_0,
                scale,
                index=0,
                fixed=plan,
                fixed_tensors=plan.tensors,
            )
            res1_3t = sparse_mod.execute_fixed_block_sparse_branches(
                q,
                k_all_1,
                v_all_1,
                scale,
                index=1,
                fixed=plan,
                fixed_tensors=plan.tensors,
            )
            assert (
                sparse_mod._LAST_3TIER_EXEC_META["dense_protection_100pct_refine"]
                is False
            )
            assert sparse_mod._LAST_3TIER_EXEC_META["M_cand"] == 14
            assert sparse_mod._LAST_3TIER_EXEC_META["M_k"] == 16
            assert sparse_mod._LAST_3TIER_EXEC_META["mode"] == "dual_query_topk"
            assert not torch.allclose(res0_3t[0], res1_3t[0], atol=1e-2)

            # 4. Multi-chunk sliding window eviction synchronization across layers
            evict_cache = ZingH3KVCache(
                num_layers=2,
                window_policy=ZingWindowPolicy(window_chunks=4, sink_chunks=2),
                storage=stores,
            )
            evict_cache.update(
                pos_p,
                torch.zeros(128, dtype=torch.long),
                "prefix",
                0,
                layers=[(k_p0, v_p0), (k_p1, v_p1)],
            )
            for u in range(1, 6):
                pos_u = torch.tensor(
                    [[float(u), 0.0, float(i)] for i in range(128)],
                    dtype=torch.float32,
                )
                k_u0 = torch.randn(128, 4, 128, dtype=torch.bfloat16)
                v_u0 = torch.randn(128, 4, 128, dtype=torch.bfloat16)
                k_u1 = torch.randn(128, 4, 128, dtype=torch.bfloat16) + 15.0
                v_u1 = torch.randn(128, 4, 128, dtype=torch.bfloat16) + 15.0
                tb = u * 2
                evict_cache.begin_chunk(tb)
                evict_cache.update(
                    pos_u,
                    torch.full((128,), tb - 1, dtype=torch.long),
                    "source",
                    1,
                    layers=[(k_u0, v_u0), (k_u1, v_u1)],
                )
                evict_cache.update(
                    pos_u, torch.full((128,), tb, dtype=torch.long), "active", 1
                )
                evict_cache.complete_step0_and_evict(tb)
                evict_cache.update(
                    pos_u,
                    torch.full((128,), tb, dtype=torch.long),
                    "final",
                    1,
                    layers=[(k_u0, v_u0), (k_u1, v_u1)],
                )

            assert len(evict_cache._compressed_kv_stores) == 2
            assert (
                evict_cache._compressed_kv_stores[0].num_blocks
                == evict_cache._length // 64
            )
            assert (
                evict_cache._compressed_kv_stores[1].num_blocks
                == evict_cache._length // 64
            )
            assert not torch.allclose(
                evict_cache._compressed_kv_stores[0].k_blk_mean,
                evict_cache._compressed_kv_stores[1].k_blk_mean,
            )
    finally:
        if orig_3t is not None:
            os.environ["SGLANG_ZING_SPARSE_THREE_TIER"] = orig_3t
        else:
            os.environ.pop("SGLANG_ZING_SPARSE_THREE_TIER", None)
        if orig_chunk is not None:
            os.environ["SGLANG_ZING_SPARSE_MIN_CHUNK"] = orig_chunk
        else:
            os.environ.pop("SGLANG_ZING_SPARSE_MIN_CHUNK", None)
        sparse_mod.set_sparse_mode(orig_mode)


@torch.no_grad()
def test_compressed_kv_zero_raw16_footprint_and_more_than_half_memory_reduction():
    """Verify zero raw16 footprint, FP8 replacement, BF16 mean retention, and >50% memory cut."""
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        ZingH3KVCache,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_sparse_attn import (
        ZingCompressedKVStore,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

    num_layers = 50
    local_heads = 4
    head_dim = 128
    rows = 4096
    num_blocks = rows // 64  # 64 blocks

    # 1. Raw16 baseline calculation
    # raw16 maintains k, v, rotated_k per row (3 tensors), each BF16 (2 bytes)
    raw16_bytes_per_row = 3 * num_layers * local_heads * head_dim * 2
    raw16_total_bytes = raw16_bytes_per_row * rows

    # 2. Pure compressed KV store initialization
    orig_3t = os.environ.get("SGLANG_ZING_SPARSE_THREE_TIER")
    try:
        os.environ["SGLANG_ZING_SPARSE_THREE_TIER"] = "1"
        cache = ZingH3KVCache(
            num_layers=num_layers,
            window_policy=ZingWindowPolicy(window_chunks=16, sink_chunks=2),
            reserve_rows=rows,
        )

        stores = cache.get_compressed_kv_stores(
            device=torch.device("cpu"),
            num_heads=local_heads,
            head_dim=head_dim,
            max_blocks=num_blocks,
        )
        assert len(stores) == num_layers

        # 3. Assert raw16 KV stores are completely empty / zero footprint
        assert len(cache._stores) == 0, "Raw16 _stores must be strictly empty"
        assert len(cache._rotated_stores) == 0, (
            "Raw16 _rotated_stores must be strictly empty"
        )

        # 4. Assert types: FP8 residuals and BF16 block mean
        sample_store: ZingCompressedKVStore = stores[0]
        assert sample_store.k_res_fp8.dtype == torch.float8_e4m3fn
        assert sample_store.v_res_fp8.dtype == torch.float8_e4m3fn
        assert sample_store.k_blk_mean.dtype == torch.bfloat16
        assert sample_store.v_blk_mean.dtype == torch.bfloat16

        # 5. Measure real resident memory footprint of owned tensors
        compressed_tensors = list(cache.owned_tensors())
        compressed_total_bytes = sum(
            t.untyped_storage().nbytes() for t in compressed_tensors
        )

        # 6. Verify strictly more than 50% memory cut (reduction >= 50%)
        reduction_ratio = 1.0 - (compressed_total_bytes / float(raw16_total_bytes))
        assert reduction_ratio >= 0.50, (
            f"Expected memory reduction >= 50%, got {reduction_ratio * 100:.2f}% "
            f"({compressed_total_bytes / (1024**2):.2f} MB vs raw16 {raw16_total_bytes / (1024**2):.2f} MB)"
        )

        # 7. In-place compaction zero-recapture pointer invariance & correctness test
        pos = torch.tensor(
            [[0.0, 0.0, float(i)] for i in range(128)], dtype=torch.float32
        )
        k_chunk = torch.randn(128, local_heads, head_dim, dtype=torch.bfloat16)
        v_chunk = torch.randn(128, local_heads, head_dim, dtype=torch.bfloat16)
        cache.update(
            pos,
            torch.zeros(128, dtype=torch.long),
            "prefix",
            0,
            layers=[(k_chunk, v_chunk)] * num_layers,
        )
        assert cache._stores == {}, "Raw16 stores must remain empty after chunk commit"

        # Record pointers
        k_res_ptr_before = sample_store.k_res_fp8.data_ptr()
        k_mean_ptr_before = sample_store.k_blk_mean.data_ptr()

        # Evict to trigger in-place compaction
        cache._sync_compressed_kv_store(retained=[])

        # Assert pointers 100% stable
        assert sample_store.k_res_fp8.data_ptr() == k_res_ptr_before
        assert sample_store.k_blk_mean.data_ptr() == k_mean_ptr_before
    finally:
        if orig_3t is not None:
            os.environ["SGLANG_ZING_SPARSE_THREE_TIER"] = orig_3t
        else:
            os.environ.pop("SGLANG_ZING_SPARSE_THREE_TIER", None)


def test_production_kv_pool_real_memory_cut_with_compressed_store():
    """Verify that ZingKVPool in three-tier sparse mode shrinks raw-16 layer stores to scratch rows,

    cutting total resident pool memory by >50% compared to dense baseline, while maintaining
    session acquire and multi-chunk fixed forward plan contracts.
    """
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        ZingFixedForwardKVPlan,
        ZingH3KVCache,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_sparse_attn import (
        ZingCompressedKVStore,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        ZingKVGeometry,
        ZingKVPool,
        ZingKVPoolConfig,
        ZingKVPoolPlan,
        ZingKVRequest,
    )

    # 1. Theoretical production memory verification with dynamically sized scratch buffer
    prod_num_layers = 50
    prod_local_heads = 14
    prod_head_dim = 128

    # 1a. 480p production contract (832x480, 32352 rows, chunk=3)
    config_480 = ZingKVPoolConfig(max_target_area=832 * 480, max_target_long_edge=832)
    geometry_480 = ZingKVGeometry(
        spatial_ratio=16,
        patch_size=(1, 2, 2),
        max_short_edge=480,
        chunk_plan=ChunkPlan([3, 2]),
    )
    scratch_rows_480 = ZingKVPoolPlan.calculate_max_forward_rows(
        config_480, geometry_480
    )
    assert scratch_rows_480 == 4290
    prod_slot_rows_480 = 32352
    prod_max_blocks_480 = (prod_slot_rows_480 + 63) // 64  # 506 blocks

    # Dense baseline vs Three-Tier Sparse
    dense_bytes_480 = (
        3 * prod_slot_rows_480 * prod_num_layers * prod_local_heads * prod_head_dim * 2
    )
    scratch_bytes_480 = (
        3 * scratch_rows_480 * prod_num_layers * prod_local_heads * prod_head_dim * 2
    )
    per_block_bytes = (
        64 * prod_local_heads * prod_head_dim * 1 * 2
        + prod_local_heads * prod_head_dim * 2 * 2
    )
    compressed_bytes_480 = prod_max_blocks_480 * per_block_bytes * prod_num_layers
    total_sparse_480 = scratch_bytes_480 + compressed_bytes_480
    reduction_480 = (dense_bytes_480 - total_sparse_480) / dense_bytes_480
    assert reduction_480 > 0.50, (
        f"Expected >50% cut in 480p production, got {reduction_480 * 100:.2f}% "
        f"(dense={dense_bytes_480 / 1024**2:.2f}MB, sparse={total_sparse_480 / 1024**2:.2f}MB)"
    )

    # 1b. 736p production contract (1280x736, 49706 rows standard / 96600 rows full session, chunk=3)
    config_736 = ZingKVPoolConfig(max_target_area=1280 * 736, max_target_long_edge=1280)
    geometry_736 = ZingKVGeometry(
        spatial_ratio=16,
        patch_size=(1, 2, 2),
        max_short_edge=736,
        chunk_plan=ChunkPlan([3, 2]),
    )
    scratch_rows_736 = ZingKVPoolPlan.calculate_max_forward_rows(
        config_736, geometry_736
    )
    assert scratch_rows_736 == 10120
    # Standard 736p capacity (49,706 rows)
    prod_slot_rows_736 = 49706
    prod_max_blocks_736 = (prod_slot_rows_736 + 63) // 64
    dense_bytes_736 = (
        3 * prod_slot_rows_736 * prod_num_layers * prod_local_heads * prod_head_dim * 2
    )
    scratch_bytes_736 = (
        3 * scratch_rows_736 * prod_num_layers * prod_local_heads * prod_head_dim * 2
    )
    compressed_bytes_736 = prod_max_blocks_736 * per_block_bytes * prod_num_layers
    total_sparse_736 = scratch_bytes_736 + compressed_bytes_736
    reduction_736 = (dense_bytes_736 - total_sparse_736) / dense_bytes_736
    assert reduction_736 > 0.45, (
        f"Expected >45% cut in standard 736p production, got {reduction_736 * 100:.2f}% "
        f"(dense={dense_bytes_736 / 1024**2:.2f}MB, sparse={total_sparse_736 / 1024**2:.2f}MB)"
    )

    # Full 21-chunk 736p session (105 frames = 96,600 rows)
    full_slot_736 = 96600
    full_dense_736 = (
        3 * full_slot_736 * prod_num_layers * prod_local_heads * prod_head_dim * 2
    )
    full_comp_736 = ((full_slot_736 + 63) // 64) * per_block_bytes * prod_num_layers
    full_total_736 = scratch_bytes_736 + full_comp_736
    full_reduction_736 = (full_dense_736 - full_total_736) / full_dense_736
    assert full_reduction_736 > 0.50, (
        f"Expected >50% cut in full 736p session, got {full_reduction_736 * 100:.2f}%"
    )

    # 2. Runtime pool allocation and contracts test with safe test dimensions
    num_layers = 2
    local_heads = 4
    head_dim = 64
    slot_rows = 4096
    scratch_rows = 256

    geometry = ZingKVGeometry(
        spatial_ratio=8,
        patch_size=(1, 2, 2),
        max_short_edge=128,
        chunk_plan=ChunkPlan([1, 1], split_first_latent=False),
    )
    config = ZingKVPoolConfig(
        max_target_area=128 * 128,
        max_target_long_edge=128,
        window_chunks=8,
        sink_chunks=2,
    )
    plan = ZingKVPoolPlan.build(
        config,
        geometry,
        token_capacity=slot_rows,
    )

    layout = dict(
        num_layers=num_layers,
        local_heads=local_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    # Dense baseline pool allocation
    dense_pool = ZingKVPool.allocate(plan, compressed_kv=False, **layout)
    dense_bytes = dense_pool.allocated_bytes()
    assert dense_pool.physical_rows == slot_rows
    assert dense_bytes == 3 * slot_rows * num_layers * local_heads * head_dim * 2

    # Three-tier sparse mode pool allocation
    orig_env = os.environ.get("SGLANG_ZING_SPARSE_THREE_TIER")
    try:
        os.environ["SGLANG_ZING_SPARSE_THREE_TIER"] = "1"
        sparse_pool = ZingKVPool.allocate(
            plan, scratch_rows=scratch_rows, compressed_kv=True, **layout
        )
        sparse_raw16_bytes = sparse_pool.allocated_bytes()
        assert sparse_pool.physical_rows == scratch_rows
        assert sparse_pool.is_compressed is True
        assert (
            sparse_raw16_bytes
            == 3 * scratch_rows * num_layers * local_heads * head_dim * 2
        )

        max_blocks = (slot_rows + 63) // 64
        sample_compressed_store = ZingCompressedKVStore(
            max_blocks=max_blocks,
            num_heads=local_heads,
            head_dim=head_dim,
            device=torch.device("cpu"),
        )
        per_layer_compressed_bytes = sample_compressed_store.allocated_bytes()
        total_compressed_bytes = per_layer_compressed_bytes * num_layers

        total_sparse_resident_bytes = sparse_raw16_bytes + total_compressed_bytes
        memory_reduction = (dense_bytes - total_sparse_resident_bytes) / dense_bytes
        assert memory_reduction > 0.50, (
            f"Expected >50% resident memory cut in runtime test, got {memory_reduction * 100:.2f}% "
            f"(dense={dense_bytes / 1024**2:.2f}MB, sparse={total_sparse_resident_bytes / 1024**2:.2f}MB)"
        )

        # End-to-end Session Acquire & Multi-Chunk Forward Simulation
        session = type("Session", (), {})()
        request = ZingKVRequest(
            target_height=128,
            target_width=128,
            source_shape=None,
            prefix_rows=0,
            window_chunks=8,
            sink_chunks=2,
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        )
        lease = sparse_pool.acquire(session, request)
        assert lease is not None
        assert lease.peak_rows <= slot_rows

        cache = ZingH3KVCache(
            num_layers=num_layers,
            window_policy=config.window_policy,
            rope_position_mode="block_relative",
            storage=lease.kv_cache,
        )
        cache._compressed_kv_enabled = True
        cache.get_compressed_kv_stores(
            device=torch.device("cpu"),
            num_heads=local_heads,
            head_dim=head_dim,
            max_blocks=max_blocks,
        )

        assert cache.capacity == max_blocks * 64
        assert cache.scratch_capacity == scratch_rows

        # Simulate consecutive forwards: chunk 0, chunk 1, chunk 2
        chunk_tokens = 64
        with torch.no_grad():
            for chunk_i in range(3):
                pos_chunk = torch.tensor(
                    [[float(chunk_i), 0.0, float(i)] for i in range(chunk_tokens)],
                    dtype=torch.float32,
                )
                plan_chunk = ZingFixedForwardKVPlan(
                    cache, pos_chunk, ends=(chunk_tokens,), clean_rows=0, prefix=False
                )
                assert torch.equal(
                    plan_chunk.tensors["write_rows"],
                    torch.arange(0, chunk_tokens, device=torch.device("cpu")),
                )
                assert plan_chunk.rotates_history is False

                dummy_layers = [
                    (
                        torch.randn(
                            chunk_tokens, local_heads, head_dim, dtype=torch.bfloat16
                        ),
                        torch.randn(
                            chunk_tokens, local_heads, head_dim, dtype=torch.bfloat16
                        ),
                    )
                    for _ in range(num_layers)
                ]
                blk_tensor = torch.full(
                    (chunk_tokens,), 2 * (chunk_i + 1), dtype=torch.long
                )
                cache.update(pos_chunk, blk_tensor, "active", 1)
                cache.update(
                    pos_chunk,
                    blk_tensor,
                    "final",
                    1,
                    layers=dummy_layers,
                )
                assert cache._length == (chunk_i + 1) * chunk_tokens
                assert cache._compressed_kv_stores[0].num_blocks == (chunk_i + 1) * (
                    chunk_tokens // 64
                )

        lease.release()
    finally:
        if orig_env is not None:
            os.environ["SGLANG_ZING_SPARSE_THREE_TIER"] = orig_env
        else:
            os.environ.pop("SGLANG_ZING_SPARSE_THREE_TIER", None)


def test_production_default_allocate_and_fixed_forward_480p_and_736p():
    """Verify production entry stage.py -> ZingKVPool.allocate(plan, **layout) without scratch_rows.

    Ensures that:
    1. ZingKVPoolPlan automatically calculates scratch_rows from contract/geometry covering
       target, source, previous-final, prompt replay, and chunk length.
    2. 736p (1280x736) and 480p (832x480) allocate sufficient scratch buffer without ZingKVCapacityError
       across real ZingFixedForwardKVPlan scenarios (single frame, chunk, source+target, cross-chunk, prompt replay).
    3. Confirms exact resident memory sizing and reduction vs baseline.
    """
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        ZingFixedForwardKVPlan,
        ZingH3KVCache,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        ZingKVGeometry,
        ZingKVPool,
        ZingKVPoolConfig,
        ZingKVPoolPlan,
        ZingKVRequest,
    )

    orig_env = os.environ.get("SGLANG_ZING_SPARSE_THREE_TIER")
    try:
        os.environ["SGLANG_ZING_SPARSE_THREE_TIER"] = "1"

        # -------------------------------------------------------------
        # 1. 736p (1280x736) Production Plan & Default Allocate
        # -------------------------------------------------------------
        config_736 = ZingKVPoolConfig(
            max_target_area=1280 * 736,
            max_target_long_edge=1280,
            window_chunks=8,
            sink_chunks=2,
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        )
        geometry_736 = ZingKVGeometry(
            spatial_ratio=16,
            patch_size=(1, 2, 2),
            max_short_edge=736,
            chunk_plan=ChunkPlan([3, 2]),
        )
        plan_736 = ZingKVPoolPlan.build(
            config_736,
            geometry_736,
            token_capacity=32352,
        )
        # Expected max forward rows for 736p:
        # max_target_rows = 920, max_source_rows = 920, chunk = 3, prompt_replay = 5 * 920
        # cross_chunk = 3 * (2 * 920 + 920) = 8280
        # prompt_joint = 3 * (920 + 920) + 5 * 920 = 10120
        assert plan_736.scratch_rows == 10120

        # Production entry call (identical to stage.py:253: ZingKVPool.allocate(plan, **layout))
        num_layers = 2
        local_heads = 14
        head_dim = 128
        layout = dict(
            num_layers=num_layers,
            local_heads=local_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        pool_736 = ZingKVPool.allocate(plan_736, **layout)
        assert pool_736.is_compressed is True
        assert pool_736.physical_rows == 10120
        assert pool_736.scratch_rows == 10120

        session_736 = type("Session", (), {})()
        request_736 = ZingKVRequest(
            target_height=736,
            target_width=1280,
            source_shape=None,
            prefix_rows=512,
            window_chunks=8,
            sink_chunks=2,
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        )
        lease_736 = pool_736.acquire(session_736, request_736)
        assert lease_736 is not None

        cache_736 = ZingH3KVCache(
            num_layers=num_layers,
            window_policy=config_736.window_policy,
            rope_position_mode="block_relative",
            storage=lease_736.kv_cache,
        )
        cache_736._compressed_kv_enabled = True
        max_blocks_736 = (32352 + 63) // 64
        cache_736.get_compressed_kv_stores(
            device=torch.device("cpu"),
            num_heads=local_heads,
            head_dim=head_dim,
            max_blocks=max_blocks_736,
        )
        assert cache_736.scratch_capacity == 10120

        # Test real ZingFixedForwardKVPlan scenarios for 736p:
        # a) Single latent frame (920 rows)
        pos_single = torch.zeros((920, 3), dtype=torch.float32)
        plan_single = ZingFixedForwardKVPlan(
            cache_736, pos_single, ends=(920,), clean_rows=0, prefix=False
        )
        assert plan_single.rows == 920 <= cache_736.scratch_capacity
        assert torch.equal(
            plan_single.tensors["write_rows"],
            torch.arange(920, device=torch.device("cpu")),
        )

        # b) Single 3-frame target chunk (2760 rows)
        pos_chunk = torch.zeros((2760, 3), dtype=torch.float32)
        plan_chunk = ZingFixedForwardKVPlan(
            cache_736, pos_chunk, ends=(2760,), clean_rows=0, prefix=False
        )
        assert plan_chunk.rows == 2760 <= cache_736.scratch_capacity

        # c) Source + Target chunk (5520 rows)
        pos_st = torch.zeros((5520, 3), dtype=torch.float32)
        plan_st = ZingFixedForwardKVPlan(
            cache_736, pos_st, ends=(2760, 5520), clean_rows=0, prefix=False
        )
        assert plan_st.rows == 5520 <= cache_736.scratch_capacity

        # d) Cross-chunk joint forward with previous-final (8280 rows)
        pos_cross = torch.zeros((8280, 3), dtype=torch.float32)
        plan_cross = ZingFixedForwardKVPlan(
            cache_736, pos_cross, ends=(2760, 5520, 8280), clean_rows=0, prefix=False
        )
        assert plan_cross.rows == 8280 <= cache_736.scratch_capacity

        # e) Prompt joint replay forward (10120 rows)
        pos_prompt = torch.zeros((10120, 3), dtype=torch.float32)
        plan_prompt = ZingFixedForwardKVPlan(
            cache_736, pos_prompt, ends=(5520, 10120), clean_rows=0, prefix=False
        )
        assert plan_prompt.rows == 10120 <= cache_736.scratch_capacity

        # f) Block0 prefix/reference forward up to scratch_rows (10120 rows)
        pos_prefix_736 = torch.zeros((10120, 3), dtype=torch.float32)
        plan_prefix_736 = ZingFixedForwardKVPlan(
            cache_736,
            pos_prefix_736,
            ends=(10120,),
            clean_rows=10120,
            prefix=True,
        )
        assert plan_prefix_736.rows == 10120 <= cache_736.scratch_capacity
        lease_736.release()

        # g) Regression: prefix_rows = scratch_rows + 1 (10121) while slot_rows (32352) is still sufficient!
        # Must be rejected at admission (ZingKVPoolAdmissionError), never admitted to fail in ZingFixedForwardKVPlan.
        import pytest

        from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
            ZingKVPoolAdmissionError,
            peak_rows,
        )

        req_736_over_scratch = ZingKVRequest(
            target_height=736,
            target_width=1280,
            source_shape=None,
            prefix_rows=plan_736.scratch_rows + 1,  # 10121 > 10120
            window_chunks=8,
            sink_chunks=2,
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        )
        # Verify slot_rows is genuinely sufficient for this request's total occupancy:
        peak_736_over = peak_rows(
            geometry_736,
            config_736.window_policy,
            prefix_rows=req_736_over_scratch.prefix_rows,
            rows_per_frame=geometry_736.rows_per_frame(736, 1280),
            has_source=False,
            window_eviction_timing=req_736_over_scratch.window_eviction_timing,
        )
        assert peak_736_over <= plan_736.slot_rows
        with pytest.raises(ZingKVPoolAdmissionError) as exc_736:
            plan_736.admit(req_736_over_scratch)
        assert exc_736.value.code == "kv_pool_request_unsupported"
        assert "exceed scratch rows" in exc_736.value.detail
        with pytest.raises(ZingKVPoolAdmissionError):
            pool_736.acquire(type("Session", (), {})(), req_736_over_scratch)
        assert pool_736.available == 1

        # -------------------------------------------------------------
        # 2. 480p (832x480) Production Plan & Default Allocate
        # -------------------------------------------------------------
        config_480 = ZingKVPoolConfig(
            max_target_area=832 * 480,
            max_target_long_edge=832,
            window_chunks=8,
            sink_chunks=2,
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        )
        geometry_480 = ZingKVGeometry(
            spatial_ratio=16,
            patch_size=(1, 2, 2),
            max_short_edge=480,
            chunk_plan=ChunkPlan([3, 2]),
        )
        plan_480 = ZingKVPoolPlan.build(
            config_480,
            geometry_480,
            token_capacity=24000,
        )
        # Expected max forward rows for 480p:
        # max_target_rows = 390, max_source_rows = 390, chunk = 3, prompt_replay = 5 * 390
        # cross_chunk = 3 * (2 * 390 + 390) = 3510
        # prompt_joint = 3 * (390 + 390) + 5 * 390 = 4290
        assert plan_480.scratch_rows == 4290

        pool_480 = ZingKVPool.allocate(plan_480, **layout)
        assert pool_480.is_compressed is True
        assert pool_480.physical_rows == 4290
        assert pool_480.scratch_rows == 4290

        session_480 = type("Session", (), {})()
        request_480 = ZingKVRequest(
            target_height=480,
            target_width=832,
            source_shape=(480, 832),
            prefix_rows=512,
            window_chunks=8,
            sink_chunks=2,
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        )
        lease_480 = pool_480.acquire(session_480, request_480)
        assert lease_480 is not None

        cache_480 = ZingH3KVCache(
            num_layers=num_layers,
            window_policy=config_480.window_policy,
            rope_position_mode="block_relative",
            storage=lease_480.kv_cache,
        )
        cache_480._compressed_kv_enabled = True
        max_blocks_480 = (24000 + 63) // 64
        cache_480.get_compressed_kv_stores(
            device=torch.device("cpu"),
            num_heads=local_heads,
            head_dim=head_dim,
            max_blocks=max_blocks_480,
        )
        assert cache_480.scratch_capacity == 4290

        # Test real ZingFixedForwardKVPlan scenarios for 480p:
        # a) Single latent frame (390 rows)
        plan_480_single = ZingFixedForwardKVPlan(
            cache_480, torch.zeros((390, 3)), ends=(390,), clean_rows=0, prefix=False
        )
        assert plan_480_single.rows == 390 <= cache_480.scratch_capacity

        # b) Single 3-frame target chunk (1170 rows)
        plan_480_chunk = ZingFixedForwardKVPlan(
            cache_480, torch.zeros((1170, 3)), ends=(1170,), clean_rows=0, prefix=False
        )
        assert plan_480_chunk.rows == 1170 <= cache_480.scratch_capacity

        # c) Source + Target chunk (2340 rows)
        plan_480_st = ZingFixedForwardKVPlan(
            cache_480,
            torch.zeros((2340, 3)),
            ends=(1170, 2340),
            clean_rows=0,
            prefix=False,
        )
        assert plan_480_st.rows == 2340 <= cache_480.scratch_capacity

        # d) Cross-chunk joint forward (3510 rows)
        plan_480_cross = ZingFixedForwardKVPlan(
            cache_480,
            torch.zeros((3510, 3)),
            ends=(1170, 2340, 3510),
            clean_rows=0,
            prefix=False,
        )
        assert plan_480_cross.rows == 3510 <= cache_480.scratch_capacity

        # e) Prompt joint replay forward (4290 rows)
        plan_480_prompt = ZingFixedForwardKVPlan(
            cache_480,
            torch.zeros((4290, 3)),
            ends=(2340, 4290),
            clean_rows=0,
            prefix=False,
        )
        assert plan_480_prompt.rows == 4290 <= cache_480.scratch_capacity

        # f) Block0 prefix/reference forward up to scratch_rows (4290 rows)
        plan_480_prefix = ZingFixedForwardKVPlan(
            cache_480,
            torch.zeros((4290, 3)),
            ends=(4290,),
            clean_rows=4290,
            prefix=True,
        )
        assert plan_480_prefix.rows == 4290 <= cache_480.scratch_capacity
        lease_480.release()

        # g) Regression: 480p prefix_rows = scratch_rows + 1 (4291) while slot_rows (24000) is still sufficient!
        req_480_over_scratch = ZingKVRequest(
            target_height=480,
            target_width=832,
            source_shape=(480, 832),
            prefix_rows=plan_480.scratch_rows + 1,  # 4291 > 4290
            window_chunks=8,
            sink_chunks=2,
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        )
        peak_480_over = peak_rows(
            geometry_480,
            config_480.window_policy,
            prefix_rows=req_480_over_scratch.prefix_rows,
            rows_per_frame=geometry_480.rows_per_frame(480, 832),
            has_source=True,
            source_rows_per_frame=geometry_480.rows_per_frame(480, 832),
            window_eviction_timing=req_480_over_scratch.window_eviction_timing,
        )
        assert peak_480_over <= plan_480.slot_rows
        with pytest.raises(ZingKVPoolAdmissionError) as exc_480:
            plan_480.admit(req_480_over_scratch)
        assert exc_480.value.code == "kv_pool_request_unsupported"
        assert "exceed scratch rows" in exc_480.value.detail
        with pytest.raises(ZingKVPoolAdmissionError):
            pool_480.acquire(type("Session", (), {})(), req_480_over_scratch)
        assert pool_480.available == 1

        # h) Request-aware calculate_max_forward_rows & prefix_rows-configured plan:
        assert (
            ZingKVPoolPlan.calculate_max_forward_rows(
                config_480, geometry_480, request=req_480_over_scratch
            )
            == 4291
        )
        plan_480_large_prefix = ZingKVPoolPlan.build(
            config_480,
            geometry_480,
            token_capacity=24000,
            prefix_rows=4291,
        )
        assert plan_480_large_prefix.scratch_rows == 4291
        pool_480_large_prefix = ZingKVPool.allocate(plan_480_large_prefix, **layout)
        lease_480_lp = pool_480_large_prefix.acquire(
            type("Session", (), {})(), req_480_over_scratch
        )
        cache_480_lp = ZingH3KVCache(
            num_layers=num_layers,
            window_policy=config_480.window_policy,
            rope_position_mode="block_relative",
            storage=lease_480_lp.kv_cache,
        )
        cache_480_lp._compressed_kv_enabled = True
        plan_480_lp_fwd = ZingFixedForwardKVPlan(
            cache_480_lp,
            torch.zeros((4291, 3)),
            ends=(4291,),
            clean_rows=4291,
            prefix=True,
        )
        assert plan_480_lp_fwd.rows == 4291 <= cache_480_lp.scratch_capacity
        lease_480_lp.release()

    finally:
        if orig_env is not None:
            os.environ["SGLANG_ZING_SPARSE_THREE_TIER"] = orig_env
        else:
            os.environ.pop("SGLANG_ZING_SPARSE_THREE_TIER", None)


if __name__ == "__main__":
    test_sparse_mode_switching_and_envs()
    test_graph_route_envs_inclusion()
    test_token_per_frame_dynamic_inference()
    test_budget_table_resolution_proportional_scaling()
    test_sequence_parallel_head_sharding()
    test_static_router_workspace_lifecycle()
    test_kv_exemption_and_dense_guardrails()
    test_exact_sdpa_reference()
    test_execute_fixed_block_sparse_branches_dense_parity()
    test_cosine_gated_stale_scene_vs_character_retention()
    test_decoupled_source_branch_and_win_src_geometry_recall()
    test_non64_aligned_eviction_shift_uniqueness()
    test_midstream_prefix_joint_reference_switch()
    test_noisy_step0_rope_does_not_rescue_stale_sink_under_tau095()
    test_mixed_latent_mq_budget_and_parity_isolation()
    test_chunk_graphs_shared_pool_generation_and_prefix_isolation()
    test_chunk_graphs_zero_recapture_across_dynamic_lk_mcand_and_prefix_switch()
    test_kv_block_stats_and_triage_mode_controls()
    test_virtual_mean_tiles_exact_on_uniform_and_pareto_gain()
    test_triage_second_order_complexity_routing_and_price_pi()
    test_step2_incremental_kv_stats_and_prebuilt_mask_reuse()
    test_ffn_qkv_quant_scope_selection()
    test_compressed_kv_store_chunk_boundary_and_memory_footprint()
    test_compressed_kv_store_temporal_rope_homomorphism_on_window_shift()
    test_three_tier_router_and_100pct_refine_dense_protection()
    test_frame_2d_square_partition_and_vectorized_total_variance_three_tier()
    test_attach_sparse_specs_recognizes_refreshed_active_sinks()
    test_production_fixed_plan_compressed_kv_lifecycle_and_three_tier_execution()
    test_production_kv_pool_real_memory_cut_with_compressed_store()
    test_production_default_allocate_and_fixed_forward_480p_and_736p()
    print("All 30 unit tests passed successfully!")
