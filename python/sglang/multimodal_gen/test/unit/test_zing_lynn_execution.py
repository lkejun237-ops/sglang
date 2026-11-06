# SPDX-License-Identifier: Apache-2.0
"""Optional fusions on an actual small Lynn ViT, with nonzero residual branches."""

from importlib.util import find_spec

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.execution import (
    LynnExecutionOptions,
)
from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.model import (
    LynnDecoderConfig,
    StreamingDecoder,
)

ARCH = dict(
    temporal_downsample_factors=[1, 2, 2, 1, 1, 1],
    spatial_downsample_factors=[2, 2],
    clip_length=17,
    token_drop=3,
    latent_channels=24,
    out_channels=3,
    decoder_num_layers=2,
    decoder_num_attention_heads=2,
    decoder_attention_head_dim=16,
    decoder_num_register_tokens=2,
    decoder_ffn_mult=2,
    decoder_rope_theta=100.0,
    decoder_rope_dim_ratio=0.75,
    decoder_norm_eps=1e-5,
    latents_mean=[0.1] * 24,
    latents_std=[1.5] * 24,
)
CONFIG = LynnDecoderConfig(3, "native", 8, 4, 0, (0.5,) * 3, (0.5,) * 3)
CUDA_KERNELS = torch.cuda.is_available() and find_spec("triton") is not None


@pytest.fixture(scope="module", autouse=True)
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_model(options, device="cpu", dtype=torch.float32):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(513)
        model = StreamingDecoder(CONFIG, ARCH, execution_options=options)
        with torch.no_grad():
            for parameter in model.parameters():
                if not parameter.any():
                    parameter.normal_(std=0.2)
    return model.to_compute(device, dtype).eval().requires_grad_(False)


@pytest.mark.parametrize(
    "options",
    [
        LynnExecutionOptions(packed_qkv=True),
        LynnExecutionOptions(packed_qkv=True, fused_qk_norm=True),
        LynnExecutionOptions(compiled_ffn=True),
    ],
)
@torch.no_grad()
def test_real_cpu_stream_parity_and_resource_release(options):
    baseline = make_model(LynnExecutionOptions())
    candidate = make_model(options)
    latent = torch.randn(1, 24, 9, 3, 3)
    left, right = baseline.new_stream(), candidate.new_stream()
    old_outputs = []
    old_copies = []
    for z in latent.split(1, dim=2):
        expected, actual = left.push(z), right.push(z)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        old_outputs.append(actual)
        old_copies.append(actual.clone())
        for output, saved in zip(old_outputs, old_copies):
            assert torch.equal(output, saved)
    state = right.state
    for stream in (left, right):
        stream.finish(30)
        assert stream.closed
    assert state.execution_signature is None
    assert baseline.state_dict().keys() == candidate.state_dict().keys()
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name])
    # A second independent stream starts at phase 0 with empty history.
    torch.testing.assert_close(
        candidate.new_stream().push(latent[:, :, :1]),
        left.model.new_stream().push(latent[:, :, :1]),
        atol=2e-5,
        rtol=2e-5,
    )


@torch.no_grad()
def test_packed_weights_refresh_and_active_stream_rejects_mutation():
    model = make_model(LynnExecutionOptions(packed_qkv=True))
    latent = torch.randn(1, 24, 1, 3, 3)
    stream = model.new_stream()
    stream.push(latent)
    block = model.decoder.transformer_blocks[0]
    old = model.execution.prepare_qkv(block)[0].clone()
    block.attn.to_q.weight.add_(0.1)
    new = model.execution.prepare_qkv(block)[0]
    assert not torch.equal(old, new)
    state = stream.state
    with pytest.raises(ValueError, match="changed during"):
        stream.push(latent)
    assert stream.closed and state.execution_signature is None
    assert model.new_stream().push(latent).shape[2] == 1


def test_environment_validation_and_snapshot(monkeypatch):
    for name in (
        "PACKED_QKV",
        "FUSED_QK_NORM",
        "COMPILED_FFN",
    ):
        monkeypatch.delenv("SGLANG_LYNN_" + name, raising=False)
    defaults = LynnExecutionOptions.from_env()
    assert defaults == LynnExecutionOptions(
        packed_qkv=True, fused_qk_norm=True, compiled_ffn=True
    )
    for name in ("PACKED_QKV", "FUSED_QK_NORM", "COMPILED_FFN"):
        monkeypatch.setenv("SGLANG_LYNN_" + name, "0")
    assert LynnExecutionOptions.from_env() == LynnExecutionOptions()
    monkeypatch.setenv("SGLANG_LYNN_PACKED_QKV", "1")
    options = LynnExecutionOptions.from_env()
    monkeypatch.setenv("SGLANG_LYNN_PACKED_QKV", "0")
    assert options.packed_qkv and not LynnExecutionOptions.from_env().packed_qkv
    monkeypatch.setenv("SGLANG_LYNN_FUSED_QK_NORM", "typo")
    with pytest.raises(ValueError, match="FUSED_QK_NORM"):
        LynnExecutionOptions.from_env()


@pytest.mark.skipif(not CUDA_KERNELS, reason="Requires CUDA and Triton")
@pytest.mark.parametrize(
    "options",
    [
        LynnExecutionOptions(packed_qkv=True, fused_qk_norm=True),
        LynnExecutionOptions(compiled_ffn=True),
        LynnExecutionOptions(packed_qkv=True, fused_qk_norm=True, compiled_ffn=True),
    ],
)
@torch.no_grad()
def test_cuda_stream_parity_and_output_ownership(options):
    baseline = make_model(LynnExecutionOptions(), "cuda", torch.bfloat16)
    candidate = make_model(options, "cuda", torch.bfloat16)
    latent = torch.randn(1, 24, 9, 3, 3, device="cuda")
    a, b = baseline.new_stream(), candidate.new_stream()
    saved = []
    for z in latent.split(1, dim=2):
        expected, actual = a.push(z), b.push(z)
        torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)
        saved.append((actual, actual.clone()))
        for old, copy in saved:
            assert torch.equal(old, copy)
    state = b.state
    b.finish(30)
    assert state.execution_signature is None
