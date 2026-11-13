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
        LynnExecutionOptions(
            packed_qkv=True, fused_qk_norm=True, sm90_fast_decode=True
        ),
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
        "SM90_FAST_DECODE",
    ):
        monkeypatch.delenv("SGLANG_LYNN_" + name, raising=False)
    defaults = LynnExecutionOptions.from_env()
    assert defaults == LynnExecutionOptions(
        packed_qkv=True,
        fused_qk_norm=True,
        compiled_ffn=True,
        sm90_fast_decode=True,
    )
    for name in (
        "PACKED_QKV",
        "FUSED_QK_NORM",
        "COMPILED_FFN",
        "SM90_FAST_DECODE",
    ):
        monkeypatch.setenv("SGLANG_LYNN_" + name, "0")
    assert LynnExecutionOptions.from_env() == LynnExecutionOptions()
    monkeypatch.setenv("SGLANG_LYNN_PACKED_QKV", "1")
    options = LynnExecutionOptions.from_env()
    monkeypatch.setenv("SGLANG_LYNN_PACKED_QKV", "0")
    assert options.packed_qkv and not LynnExecutionOptions.from_env().packed_qkv
    monkeypatch.setenv("SGLANG_LYNN_SM90_FAST_DECODE", "1")
    assert LynnExecutionOptions.from_env().sm90_fast_decode
    monkeypatch.setenv("SGLANG_LYNN_SM90_FAST_DECODE", "typo")
    with pytest.raises(ValueError, match="SM90_FAST_DECODE"):
        LynnExecutionOptions.from_env()
    monkeypatch.setenv("SGLANG_LYNN_SM90_FAST_DECODE", "0")
    monkeypatch.setenv("SGLANG_LYNN_FUSED_QK_NORM", "typo")
    with pytest.raises(ValueError, match="FUSED_QK_NORM"):
        LynnExecutionOptions.from_env()


@torch.no_grad()
def test_sm90_lease_release_and_mutation_rejection(monkeypatch):
    import sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.sm90_fast_decode as sm90_mod

    released = []

    class _FakeLease:
        def __init__(self):
            self.active = True

        def release(self):
            self.active = False
            released.append(self)

    class _FakeEngine:
        def __init__(self):
            self.busy = False
            self.cleared = False

        def clear(self):
            self.cleared = True

        def decode_step(self, z_t, state, phase):
            if state.latent_index == 0:
                if self.busy:
                    return None
                lease = _FakeLease()
                self.busy = True
                orig_release = lease.release

                def _release():
                    self.busy = False
                    orig_release()

                lease.release = _release
                state.sm90_lease = lease
            elif state.sm90_lease is None:
                return None
            frames = 1 if phase == 0 else 4
            return torch.full(
                (z_t.shape[0], 3, frames, z_t.shape[-2] * 4, z_t.shape[-1] * 4),
                0.5,
                dtype=torch.float32,
            )

    fake_engine = _FakeEngine()
    monkeypatch.setattr(
        sm90_mod,
        "get_or_create_sm90_engine",
        lambda execution, model, z_t: fake_engine,
    )
    model = make_model(LynnExecutionOptions(sm90_fast_decode=True))
    latent = torch.randn(1, 24, 3, 3, 3)

    s1 = model.new_stream()
    s2 = model.new_stream()
    out1_0 = s1.push(latent[:, :, :1])
    assert out1_0.shape == (1, 3, 1, 12, 12)
    assert model.execution.counts["sm90_fast_decode"] == 1
    # Second concurrent active stream falls back cleanly without stealing s1's lease.
    out2_0 = s2.push(latent[:, :, :1])
    assert out2_0.shape == (1, 3, 1, 12, 12)
    assert model.execution.counts["sm90_fast_decode"] == 1
    s1.close()
    assert len(released) == 1 and not fake_engine.busy
    # s2 started on fallback at step 0 and must stay on fallback at step 1 even after s1 closes.
    s2.push(latent[:, :, 1:2])
    assert model.execution.counts["sm90_fast_decode"] == 1
    s2.close()

    # Parameter mutation during an active SM90 stream closes and releases lease.
    s3 = model.new_stream()
    s3.push(latent[:, :, :1])
    assert fake_engine.busy
    model.decoder.transformer_blocks[0].attn.to_q.weight.add_(0.05)
    with pytest.raises(ValueError, match="changed during"):
        s3.push(latent[:, :, 1:2])
    assert s3.closed and not fake_engine.busy


@pytest.mark.parametrize("module_path", ("ff", "ff.net", "ff.net.0", "ff.net.1"))
def test_sm90_rejects_ffn_hooks_before_and_during_stream(monkeypatch, module_path):
    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming import (
        sm90_fast_decode as fast,
    )

    arch = dict(
        ARCH,
        spatial_downsample_factors=[4, 4],
        decoder_num_attention_heads=32,
        decoder_attention_head_dim=64,
        decoder_num_register_tokens=4,
        decoder_ffn_mult=4,
    )
    with torch.device("meta"):
        model = StreamingDecoder(
            LynnDecoderConfig(3, "native", 256, 64, 0, (0.5,) * 3, (0.5,) * 3),
            arch,
            execution_options=LynnExecutionOptions(),
        )
    model.compute_dtype = torch.bfloat16
    device = torch.device("cuda:0")
    monkeypatch.setattr(fast, "_HAS_TRITON", True)
    monkeypatch.setattr(fast, "_has_sm90_fa3", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    block = model.decoder.transformer_blocks[0]
    engine = fast.LynnSm90FastEngine.__new__(fast.LynnSm90FastEngine)
    engine.signature, engine.device = (), device
    engine._expected_config = model.config
    engine._watched_modules = fast._ffn_modules(block)
    assert fast._supports_sm90_model(model, device)
    assert engine.matches_model(model, device, ())
    handle = block.get_submodule(module_path).register_forward_hook(
        lambda module, args, output: torch.zeros_like(output)
    )
    try:
        assert not fast._supports_sm90_model(model, device)
        assert not engine.matches_model(model, device, ())
    finally:
        handle.remove()


def test_sm90_active_geometry_is_not_evicted(monkeypatch):
    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming import (
        sm90_fast_decode as fast,
    )

    engine = fast.LynnSm90FastEngine.__new__(fast.LynnSm90FastEngine)
    slot = {"in_use": True, "graphs": {0: object()}}
    engine.slots = {(1, 30, 52): slot}
    # This must return before allocating or capturing any new geometry.
    assert engine._get_or_build_slot((1, 46, 80)) is None
    assert engine.slots == {(1, 30, 52): slot}
    lease = fast._SlotLease(engine, (1, 30, 52), slot)
    lease.release()
    assert not slot["in_use"] and lease.slot is None
    lease.release()


@pytest.mark.skipif(not CUDA_KERNELS, reason="Requires CUDA and Triton")
@pytest.mark.parametrize(
    "options",
    [
        LynnExecutionOptions(packed_qkv=True, fused_qk_norm=True),
        LynnExecutionOptions(compiled_ffn=True),
        LynnExecutionOptions(
            packed_qkv=True,
            fused_qk_norm=True,
            compiled_ffn=True,
            sm90_fast_decode=True,
        ),
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


@pytest.mark.skipif(not CUDA_KERNELS, reason="Requires CUDA and Triton")
@pytest.mark.parametrize("layers", (2, 5))
@torch.no_grad()
def test_sm90_shallow_and_deep_streams(layers):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Requires SM90")
    h3_arch = dict(
        ARCH,
        spatial_downsample_factors=[4, 4],
        decoder_num_layers=layers,
        decoder_num_attention_heads=32,
        decoder_attention_head_dim=64,
        decoder_num_register_tokens=4,
        decoder_ffn_mult=4,
        decoder_rope_dim_ratio=0.75,
    )
    h3_cfg = LynnDecoderConfig(3, "native", 256, 64, 0, (0.5,) * 3, (0.5,) * 3)
    with torch.random.fork_rng(devices=["cuda"]):
        torch.manual_seed(777)
        ref_m = StreamingDecoder(
            h3_cfg, h3_arch, execution_options=LynnExecutionOptions()
        )
        for p in ref_m.parameters():
            if not p.any():
                p.normal_(std=0.02)
        fast_m = StreamingDecoder(
            h3_cfg,
            h3_arch,
            execution_options=LynnExecutionOptions(sm90_fast_decode=True),
        )
        fast_m.load_state_dict(ref_m.state_dict())
    ref_m.to_compute("cuda", torch.bfloat16).eval().requires_grad_(False)
    fast_m.to_compute("cuda", torch.bfloat16).eval().requires_grad_(False)
    z_seq = torch.randn(1, 24, 7, 28, 28, device="cuda", dtype=torch.bfloat16) * 0.2
    sr, sf = ref_m.new_stream(frame_limit=22), fast_m.new_stream(frame_limit=22)
    out_r = torch.cat([sr.push(z) for z in z_seq.split(1, dim=2)], dim=2)
    out_f = torch.cat([sf.push(z) for z in z_seq.split(1, dim=2)], dim=2)
    sr.finish(22)
    sf.finish(22)
    assert fast_m.execution.counts["sm90_fast_decode"] == 7
    assert (fast_m.execution.sm90_engine.fast_start < layers) == (layers > 4)
    assert torch.isfinite(out_f).all()
    assert (out_f - out_r).abs().mean().item() < 0.02
    # Repeat determinism check
    sf2 = fast_m.new_stream(frame_limit=22)
    out_f2 = torch.cat([sf2.push(z) for z in z_seq.split(1, dim=2)], dim=2)
    sf2.finish(22)
    assert torch.equal(out_f, out_f2)
    # Steady-state overlap check rejects mid-stream weight mutation and releases lease
    sf3 = fast_m.new_stream(frame_limit=22)
    sf3.push(z_seq[:, :, :1])
    fast_m.decoder.transformer_blocks[0].attn.to_q.weight.add_(0.05)
    with pytest.raises(ValueError, match="changed during"):
        sf3.push(z_seq[:, :, 1:2])
    assert sf3.closed
    # A new canvas replaces the idle pool; returning to the old canvas must
    # rebuild it, not retain one large pool per historical shape.
    for h, w in ((32, 32), (28, 28)):
        changed = fast_m.new_stream(frame_limit=1)
        output = changed.push(
            torch.zeros(1, 24, 1, h, w, device="cuda", dtype=torch.bfloat16)
        )
        changed.finish(1)
        assert output.shape == (1, 3, 1, h * 16, w * 16)
        assert list(fast_m.execution.sm90_engine.slots) == [(1, h, w)]
