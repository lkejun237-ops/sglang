# SPDX-License-Identifier: Apache-2.0
"""Exact eager BF16 arithmetic and pre-RoPE cache ownership on SM90/SM103."""

import math

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.attention import (
    _can_fuse_preparation,
    rotate,
)


def supported_device():
    if not torch.cuda.is_available():
        return False
    p = torch.cuda.get_device_properties(0)
    return (p.major, p.minor) in ((9, 0), (10, 3))


pytestmark = pytest.mark.skipif(
    not supported_device(), reason="Requires SM90 (H100/H200) or SM103 (B300) GPU"
)


@pytest.mark.parametrize("batch", [6, 12, 14, 28, 30])
@pytest.mark.parametrize("history", [0, 256, 512])
@pytest.mark.parametrize("rotary_dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_exact_half_split_and_bounded_spatial_history(batch, history, rotary_dtype):
    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.fused_preparation import (
        prepare,
    )

    torch.manual_seed(352)
    q, k, v = [
        torch.randn(batch, 261, 32, 64, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    hk, hv = [
        torch.randn(batch, history, 32, 64, device="cuda", dtype=q.dtype)
        for _ in range(2)
    ]
    # Identifiable special tokens must stay visible to attention and out of cache.
    k[:, -5:] = 11
    v[:, -5:] = 13
    q[0, 0, 0, :4] = torch.tensor(
        [-0.0, 0.0, torch.finfo(q.dtype).tiny, 2.0**-133], device="cuda", dtype=q.dtype
    )
    q[0, 0, 0, 24:28] = 0  # Keep the tiny first products observable in isolation.
    ropes = []
    for length in (261, history + 261):
        angle = torch.randn(batch, length, 1, 48, device="cuda") * math.pi
        cos, sin = angle.cos().to(rotary_dtype), angle.sin().to(rotary_dtype)
        cos[:, -5:] = 1
        sin[:, -5:] = 0
        ropes.append((cos, sin))
    all_k, all_v = torch.cat([hk, k], 1), torch.cat([hv, v], 1)
    end = history + 256
    start = max(0, end - 512)
    expected = (
        rotate(q, ropes[0]),
        rotate(all_k, ropes[1]),
        all_v,
        all_k[:, start:end].clone(),
        all_v[:, start:end].clone(),
    )
    assert _can_fuse_preparation(q, k, v, hk, hv, *ropes, 256, 512)
    actual = prepare(q, k, v, hk, hv, *ropes, 256, 512)
    for left, right in zip(actual, expected):
        assert left.shape == right.shape
        assert bool(left.isfinite().all())
        assert torch.equal(left, right)
    for cache in actual[-2:]:
        assert cache.is_contiguous()
        assert cache.untyped_storage().nbytes() == cache.numel() * cache.element_size()
    # The unrotated cache owns its data independently of the attention buffers.
    actual[1].zero_()
    actual[2].zero_()
    assert torch.equal(actual[3], expected[3])
    assert torch.equal(actual[4], expected[4])


@torch.no_grad()
def test_unsupported_shapes_and_layouts_keep_the_torch_path():
    q = torch.zeros(6, 261, 32, 64, device="cuda", dtype=torch.bfloat16)
    history = q[:, :0].contiguous()
    rope = (
        torch.ones(6, 261, 1, 48, device="cuda"),
        torch.zeros(6, 261, 1, 48, device="cuda"),
    )
    assert not _can_fuse_preparation(q, q, q, history, history, rope, rope, 256, 0)
    assert not _can_fuse_preparation(
        q[:1], q[:1], q[:1], history[:1], history[:1], rope, rope, 256, 512
    )
    strided = torch.zeros(6, 261, 64, 32, device="cuda", dtype=q.dtype).transpose(
        -1, -2
    )
    assert not _can_fuse_preparation(
        strided, q, q, history, history, rope, rope, 256, 512
    )
    narrow_rope = tuple(t[..., :12] for t in rope)
    assert not _can_fuse_preparation(
        q, q, q, history, history, narrow_rope, narrow_rope, 256, 512
    )


@torch.no_grad()
def test_startup_precompiles_all_supported_preparation_shapes():
    from msgspec.structs import replace
    from test_zing_lynn_stream import ARCHITECTURE, CONFIG

    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.model import (
        StreamingDecoder,
    )

    # Keep weights on CPU, as allowed during pipeline initialization. Only the
    # real preparation matrix runs on CUDA; no attention/backend is required.
    architecture = {
        **ARCHITECTURE,
        "decoder_num_layers": 1,
        "decoder_num_attention_heads": 32,
        "decoder_attention_head_dim": 64,
        "decoder_num_register_tokens": 4,
    }
    decoder = StreamingDecoder(replace(CONFIG, tile_size=64), architecture)
    decoder.to_compute("cpu", torch.bfloat16).eval().requires_grad_(False)
    assert decoder.warmup_preparation("cuda:0") == 30
