# SPDX-License-Identifier: Apache-2.0
"""Native batch ordering and large-tensor padding regressions, without a GPU."""

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.visual_encoding import (
    SpatialReflectionPad,
    encode_clips,
)


@pytest.mark.parametrize("mode", ["reflect", "replicate", "constant"])
@pytest.mark.parametrize(
    "padding", [(0, 0, 0, 0, 0, 0), (1, 2, 2, 1, 0, 0), (1, 1, 1, 1, 1, 1)]
)
def test_padding_equivalence(mode, padding):
    values = torch.randn(2, 3, 4, 5, 6)
    expected = F.pad(values, padding, mode)
    with SpatialReflectionPad():
        actual = F.pad(values, padding, mode)
        direct = torch.ops.aten.pad.default(values, padding, mode)
    assert torch.equal(actual, expected)
    assert torch.equal(direct, expected)


def test_zero_padding_large_shape_entrypoint():
    values = torch.empty((15, 128, 17, 257, 257), device="meta")
    with SpatialReflectionPad():
        result = torch.ops.aten.pad.default(values, [0] * 6, "reflect")
    assert result is values


class TileEncoder:
    encoder_tiling = True
    vae_ratio = 1

    def __init__(self):
        self.batches = []

    def split_tiles(self, length, _):
        return ([0, 2], [2, length - 2], [0])

    def encode(self, values):
        self.batches.append(values[:, 0, 0, 0, 0].tolist())
        return values * 2

    def _assemble_tiles(self, rows, y_overlap, x_overlap):
        assert y_overlap == x_overlap == [0]
        return torch.cat([torch.cat(row, -1) for row in rows], -2)


def test_tile_major_batches_and_shape_buckets_restore_clip_order():
    model = TileEncoder()
    clips = [
        torch.arange(20).reshape(1, 1, 1, 4, 5).float() + i * 100 for i in range(3)
    ]
    actual = encode_clips(model, clips, 2)
    # Shape (1,2,2) and (1,2,3) bucket separately; each is tile-major.
    assert model.batches == [
        [0, 100],
        [200, 10],
        [110, 210],
        [2, 102],
        [202, 12],
        [112, 212],
    ]
    for result, clip in zip(actual, clips):
        assert torch.equal(result, clip * 2)


def test_untiled_batch_limit_and_tail():
    model = TileEncoder()
    model.encoder_tiling = False
    clips = [torch.full((1, 1, 2, 2, 2), float(i)) for i in range(5)]
    actual = encode_clips(model, clips, 3)
    assert model.batches == [[0, 1, 2], [3, 4]]
    for result, clip in zip(actual, clips):
        assert torch.equal(result, clip * 2)
    with pytest.raises(ValueError, match="must be positive"):
        encode_clips(model, clips, 0)
