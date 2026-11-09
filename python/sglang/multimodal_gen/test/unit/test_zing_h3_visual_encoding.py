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


def test_encode_visual_performance_syncs_peer_rank_exception_before_broadcast(
    monkeypatch,
):
    from types import SimpleNamespace

    import torch.distributed as dist

    import sglang.multimodal_gen.runtime.distributed as dist_mod
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        codecs as codecs_module,
    )

    codec = object.__new__(codecs_module.ZingH3Codecs)
    codec.vae_frames_per_chunk = 17
    codec.vae_latents_per_chunk = 5
    codec.temporal_ratio = 4
    codec.va = SimpleNamespace(latents_mean=[0.0], latents_std=[1.0])
    codec.runtime = {
        "vae_config": {
            "encode_dtype": "float32",
            "encode_mode": "performance",
            "encode_tile_batch_size": 1,
            "posterior_seed": 42,
        }
    }

    class DummyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1)
            self.quant_conv = torch.nn.Linear(1, 1)

    codec.modules = {"video_vae": DummyVAE()}
    monkeypatch.setattr(
        codecs_module,
        "encode_clips",
        lambda model, clips, batch_size: [torch.zeros(1, 4, 1, 2, 2)],
    )

    broadcast_called = []
    fake_sp_group = SimpleNamespace(
        world_size=2,
        cpu_group="sp-cpu-group",
        broadcast=lambda tensor, src=0: broadcast_called.append(True) or tensor,
    )
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist_mod, "get_sp_group", lambda: fake_sp_group)

    def fake_all_gather_object(out_list, obj, group=None):
        out_list[0] = obj
        out_list[1] = ("RuntimeError", "peer CUDA OOM during encode_clips")

    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_object)

    frames = torch.zeros(3, 1, 8, 8, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="peer SP rank"):
        codec.encode_visual(frames)
    assert broadcast_called == []
