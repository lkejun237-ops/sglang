# SPDX-License-Identifier: Apache-2.0
"""Complete FP8 attention views: ragged tiles, current rows and raw scratch."""

import math

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3_compressed_view import (
    dequantize_layer,
    pack_attention,
    resident_rows,
    token_layout,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_sparse_attn import (
    ZingCompressedKVStore,
    _three_tier_compressed_sdpa_reference,
)


def positions(rows, t=0):
    p = torch.zeros(rows, 3)
    p[:, 0] = t
    p[:, 2] = torch.arange(rows)
    return p


def tensors(layout):
    return (
        torch.tensor(layout.gather),
        torch.tensor([r < n for n in layout.valid_lengths for r in range(64)]),
        torch.tensor(layout.compact_gather),
    )


def test_chunk_alignment_uses_four_blocks_for_two_65_row_parts():
    layout = token_layout([(positions(65), True), (positions(65, 1), False)])
    assert layout.valid_lengths == (64, 1, 64, 1)
    assert layout.padded_ends == (128, 256)
    assert len(layout.inverse) == 130
    values = torch.arange(130)
    padded = values[torch.tensor(layout.gather)]
    assert torch.equal(padded[torch.tensor(layout.inverse)], values)
    assert resident_rows([65, 65]) == (tuple(range(65)) + tuple(range(128, 193)), 4)


def test_router_block_roles_survive_layout_translation():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.models.dits.zing_h3_compressed_view import (
        branch_spec,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_sparse_attn import (
        SparseBranchSpec,
    )

    plan = SimpleNamespace(
        compressed_prefix_blocks=2,
        compressed_attention_layout=token_layout(
            [
                (positions(128), True),
                (positions(128, 1), False),
                (positions(64, 2), False),
            ]
        ),
        compressed_query_layout=token_layout([(positions(64, 2), False)]),
    )
    spec = SparseBranchSpec(
        "sparse_095",
        "0.95",
        "fixture",
        2,
        "active",
        0,
        1000,
        0,
        64,
        64,
        320,
        256,
        1,
        5,
        4,
        1,
        2,
        blk_kind_sig=(4, ((1, 2),), (2, 3), ((3, 4),)),
    )
    translated = branch_spec(plan, spec)
    assert translated.blk_kind_sig == (4, ((1, 2),), ((2, 3),), ((3, 4),))
    assert translated.L_k == 320 and translated.q_stop == 64


@pytest.mark.parametrize("height,width,blocks", [(15, 26, 7), (23, 40, 15)])
def test_real_visual_grids_use_frame_local_2d_tiles(height, width, blocks):
    grid = torch.tensor([[1, h, w] for h in range(height) for w in range(width)])
    layout = token_layout(
        [(positions(130), True), (grid, False), (grid + torch.tensor([1, 0, 0]), False)]
    )
    assert len(layout.valid_lengths) == 3 + 2 * blocks
    assert sum(layout.valid_lengths) == 130 + 2 * height * width
    assert sorted(layout.compact_gather) == list(range(130 + 2 * height * width))


@torch.no_grad()
def test_resident_dequantization_keeps_part_order_without_padding():
    store = ZingCompressedKVStore(8, 1, 128, "cpu")
    store.append_chunk(
        torch.full((65, 1, 128), 2.0), torch.full((65, 1, 128), 3.0), chunk_id=-1
    )
    store.append_chunk(
        torch.full((17, 1, 128), 7.0), torch.full((17, 1, 128), 11.0), chunk_id=0
    )
    indices, count = resident_rows([65, 17])
    k, v = dequantize_layer(store, torch.tensor(indices), count, dtype=torch.float32)
    assert k.shape == (82, 1, 128)
    assert torch.equal(k[:65], torch.full_like(k[:65], 2.0))
    assert torch.equal(k[65:], torch.full_like(k[65:], 7.0))
    assert torch.equal(v[65:], torch.full_like(v[65:], 11.0))


@torch.no_grad()
def test_attention_includes_current_rows_beyond_raw_scratch_capacity():
    # This is the actual kernel reference, not a dispatch mock. Zero queries
    # give an independently known weighted mean over exactly the valid rows.
    layout = token_layout(
        [(positions(130), True), (positions(17, 1), False), (positions(33, 2), False)]
    )
    keys = torch.zeros(180, 1, 128, dtype=torch.bfloat16)
    values = torch.cat(
        [
            torch.ones(130, 1, 128),
            torch.full((17, 1, 128), 3.0),
            torch.full((33, 1, 128), 5.0),
        ]
    ).bfloat16()
    _, _, view = pack_attention(keys, values, layout, *tensors(layout))
    q = torch.zeros(3, 1, 128, dtype=torch.bfloat16)
    blocks = len(layout.valid_lengths)
    args = dict(
        q=q,
        k=None,
        v=None,
        scale=1 / math.sqrt(128),
        L_k=layout.extent,
        M_cand=4,
        M_k=blocks,
        cand_cnt=torch.tensor([blocks], dtype=torch.int32),
        cand_idx=torch.arange(blocks, dtype=torch.int32).view(1, 1, -1),
        head_order=torch.tensor([0], dtype=torch.int32),
        compressed_kv=view,
        enable_virtual_mean=False,
    )
    result = _three_tier_compressed_sdpa_reference(**args)
    expected = torch.full_like(result, (130 + 17 * 3 + 33 * 5) / 180)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    values[-33:].fill_(13)
    _, _, next_view = pack_attention(keys, values, layout, *tensors(layout))
    changed = _three_tier_compressed_sdpa_reference(
        **{**args, "compressed_kv": next_view}
    )
    torch.testing.assert_close(
        changed,
        torch.full_like(changed, (130 + 17 * 3 + 33 * 13) / 180),
        atol=0,
        rtol=0,
    )
