# SPDX-License-Identifier: Apache-2.0
"""Actual H3 temporal assembly on CPU; learned window kernels are fixtures."""

import math
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_codecs import (
    SourcePixelBuffer,
    StableVideoDecoder,
)
import test_zing_h3_media_contract as media_contract

codecs = media_contract.codecs
temporal_codecs = media_contract.temporal_codecs


@pytest.mark.parametrize("frames", [1, *range(2, 35), 243, 527])
@pytest.mark.parametrize("batches", [[1], [17], [3, 11, 2, 19, 7]])
def test_source_windows_ignore_network_boundaries(frames, batches):
    calls = []

    def encode(pixels):
        calls.append(pixels.clone())
        # Context-sensitive stand-in; real normalization is tested separately.
        return pixels[None, :, ::4].float() + pixels.float().mean()

    c = SimpleNamespace(
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        temporal_ratio=4,
        encode_visual_moments=encode,
    )
    stream = SourcePixelBuffer(c, capacity_frames=36)
    pixels = (
        torch.arange(3 * frames, dtype=torch.int64)
        .remainder(256)
        .byte()
        .view(3, frames, 1, 1)
    )
    results = []
    cursor = i = 0
    while cursor < frames:
        count = min(batches[i % len(batches)], frames - cursor)
        stream.append(pixels[:, cursor : cursor + count], frame_start=cursor)
        cursor += count
        i += 1
        while stream.ready:
            results.append(stream.pop())
        assert stream.buffered_frames < 17
        assert stream.pop() is None  # pause does not pad or encode
    stream.finish()
    results.extend(stream.drain())
    expected = []
    for start in range(0, frames, 17):
        window = pixels[:, start : start + 17]
        count = window.shape[1]
        if count < 17 and frames != 1:
            window = torch.cat(
                (window, window[:, -1:].expand(-1, 17 - count, -1, -1)), 1
            )
        assert torch.equal(calls[len(expected)], window)
        expected.append(encode(window)[:, :, : 1 + math.ceil((count - 1) / 4)])
    torch.testing.assert_close(
        torch.cat([r.values for r in results], 2),
        torch.cat(expected, 2),
        rtol=0,
        atol=0,
    )
    assert results[-1].source_dependency_end == frames
    assert results[0].start == 0
    assert all(a.end == b.start for a, b in zip(results, results[1:]))
    assert stream.buffered_frames == 0
    stream.dispose()
    assert stream.codecs is None and not stream.ready


@pytest.mark.parametrize("frames", [1, *range(2, 35), 243, 527])
@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("stream_cat", ["0", "1"])
def test_stable_decoder_matches_full_temporal_assembly(
    temporal_codecs, monkeypatch, frames, batch, stream_cat
):
    c, calls = temporal_codecs
    monkeypatch.setenv("MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT", stream_cat)
    clips, tail = divmod(frames - 1, 17)
    n = clips * 5 + 1 + math.ceil(tail / 4)
    latent = torch.linspace(-0.4, 0.4, n).view(1, 1, n, 1, 1).expand(1, 3, n, 1, 1)
    expected = c.decode_visual(latent, frames)
    stream = StableVideoDecoder(c, capacity_latents=16)
    output = []
    for start in range(0, n, batch):
        end = min(start + batch, n)
        stream.append(
            latent[:, :, start:end], latent_start=start, source_dependency_end=end * 17
        )
        while stream.ready:
            calls.clear()
            result = stream.pop()
            output.append(result)
            assert len(calls) <= 2
            assert all(call.shape[2] <= 7 for call in calls)
            assert result.source_dependency_end <= end * 17
        assert stream.buffered_latents <= 11
        assert stream.pop() is None
    stream.finish(total_frames=frames)
    output.extend(stream.drain())
    torch.testing.assert_close(
        torch.cat([r.values for r in output], 2), expected, rtol=0, atol=0
    )
    assert output[0].start == 0 and output[-1].end == frames
    assert all(a.end == b.start for a, b in zip(output, output[1:]))
    assert stream.buffered_latents == 0 and stream._latent is None
    stream.dispose()
    assert not stream.ready


def test_source_capacity_failure_and_kernel_retry():
    def fail(pixels):
        raise RuntimeError("kernel failure")

    c = SimpleNamespace(
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        temporal_ratio=4,
        encode_visual_moments=fail,
    )
    with pytest.raises(ValueError, match="complete VAE clip"):
        SourcePixelBuffer(c, capacity_frames=16)
    stream = SourcePixelBuffer(c, capacity_frames=17)
    pixels = torch.zeros(3, 17, 1, 1, dtype=torch.uint8)
    stream.append(pixels, frame_start=0)
    pixels.fill_(255)
    assert not stream._pixels.any()
    with pytest.raises(ValueError, match="capacity"):
        stream.append(pixels[:, :1], frame_start=17)
    with pytest.raises(RuntimeError):
        stream.pop()
    assert stream.consumed_frames == stream.emitted_latents == 0
    assert stream.buffered_frames == 17
    stream.dispose()
    with pytest.raises(ValueError, match="closed"):
        stream.append(pixels, frame_start=17)


def test_decoder_failure_dependency_and_ownership(temporal_codecs, monkeypatch):
    c, calls = temporal_codecs
    stream = StableVideoDecoder(c, capacity_latents=12)
    z = torch.zeros(1, 3, 7, 1, 1)
    stream.append(z[:, :, :5], latent_start=0, source_dependency_end=17)
    stream.append(z[:, :, 5:], latent_start=5, source_dependency_end=34)
    z.fill_(1)
    assert not stream._latent.any()
    original = c.decode_visual_range

    def fail(*args):
        raise RuntimeError("decode failure")

    monkeypatch.setattr(c, "decode_visual_range", fail)
    with pytest.raises(RuntimeError):
        stream.pop()
    assert stream.emitted_frames == stream.latent_start == 0
    assert stream.buffered_latents == 7
    monkeypatch.setattr(c, "decode_visual_range", original)
    result = stream.pop()
    assert result.source_dependency_end == 34
    assert result.end == 17
    with pytest.raises(ValueError, match="does not match"):
        stream.finish(total_frames=18)
    assert stream.total_frames is None
    stream.finish(total_frames=22)
    assert list(stream.drain())[-1].end == 22
    with pytest.raises(ValueError, match="closed"):
        stream.append(z, latent_start=7, source_dependency_end=34)


@pytest.mark.parametrize("encode_mode", ["deterministic", "performance"])
def test_source_moments_use_existing_normalization_and_precision(codecs, encode_mode):
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(1, 1)
    model.quant_conv = torch.nn.Linear(1, 1)
    seen = []

    def encode(pixels):
        assert torch.backends.cudnn.benchmark == (encode_mode == "performance")
        assert torch.backends.cudnn.deterministic == (encode_mode == "deterministic")
        assert torch.are_deterministic_algorithms_enabled() == (
            encode_mode == "deterministic"
        )
        seen.append(pixels)
        return pixels[:, :, ::4]

    model._adaptive_encode = encode
    codecs.modules = {"video_vae": model}
    codecs.runtime = {
        "vae_config": {"encode_dtype": "float32", "encode_mode": encode_mode}
    }
    pixels = torch.arange(51, dtype=torch.uint8).view(3, 17, 1, 1)
    result = codecs.encode_visual_moments(pixels)
    normalized = (
        pixels[None].float() / 255
        - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    torch.testing.assert_close(result, normalized[:, :, ::4], rtol=0, atol=0)
    with pytest.raises(ValueError, match="complete clip"):
        codecs.encode_visual_moments(pixels[:, :2])


def test_full_posterior_seed_is_not_a_streaming_noise_contract():
    # Even a continuous generator cannot preserve channel-major full-T ordering.
    full = torch.randn((1, 3, 10, 2, 2), generator=torch.Generator().manual_seed(42))
    generator = torch.Generator().manual_seed(42)
    chunks = torch.cat(
        [torch.randn((1, 3, 5, 2, 2), generator=generator) for _ in range(2)], 2
    )
    assert not torch.equal(full, chunks)


@pytest.mark.parametrize("drop", [0, 1, 2, 4])
@pytest.mark.parametrize("frames", [*range(1, 35), 243])
def test_native_token_drop_variants(temporal_codecs, drop, frames):
    c, calls = temporal_codecs
    model = c.modules["video_vae"]
    c.va.vae_token_drop = model.token_drop = drop
    model.token_overlap = (-drop) % 5
    model.frame_overlap = max(0, model.token_overlap * 4 - 3)
    full, tail = divmod(frames - 1, 17)
    tokens = full * 5 + 1 + math.ceil(tail / 4)
    latent = (
        torch.linspace(-0.4, 0.4, tokens)
        .view(1, 1, tokens, 1, 1)
        .expand(1, 3, tokens, 1, 1)
    )
    expected = c.decode_visual(latent, frames)
    stream = StableVideoDecoder(c, capacity_latents=14)
    outputs = []
    for i in range(tokens):
        stream.append(
            latent[:, :, i : i + 1], latent_start=i, source_dependency_end=243
        )
        while stream.ready:
            outputs.append(stream.pop().values)
    stream.finish(total_frames=frames)
    outputs.extend(r.values for r in stream.drain())
    torch.testing.assert_close(torch.cat(outputs, 2), expected, rtol=0, atol=0)


def test_decoder_shared_provenance_crosses_source_ids(temporal_codecs):
    from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
        PixelRange,
        Provenance,
        SourceSpan,
    )

    calls = []

    def provenance_for(pixels, dependency):
        calls.append((pixels, dependency))
        return Provenance(
            pixels, (SourceSpan(1, 0, 0, 15), SourceSpan(2, 15, 15, 2)), dependency
        )

    c, _ = temporal_codecs
    stream = StableVideoDecoder(c, capacity_latents=12, provenance_for=provenance_for)
    stream.append(torch.zeros(1, 3, 7, 1, 1), latent_start=0, source_dependency_end=34)
    result = stream.pop()
    assert calls == [(PixelRange(0, 17), 34)]
    assert result.provenance.source_dependency_end == 34
    assert [s.source_id for s in result.provenance.source_alignment] == [1, 2]
    assert [s.num_frames for s in result.provenance.source_alignment] == [15, 2]
    stream.dispose()
    assert stream.provenance_for is None


@pytest.mark.parametrize("failure", ["exception", "type", "range", "dependency"])
def test_decoder_metadata_failure_keeps_state_retryable(temporal_codecs, failure):
    from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
        PixelRange,
        Provenance,
        SourceSpan,
    )

    failing = True

    def provenance_for(pixels, dependency):
        if failing:
            if failure == "exception":
                raise RuntimeError("Alignment is unavailable")
            if failure == "type":
                return None
            if failure == "range":
                pixels = PixelRange(1, 17)
            if failure == "dependency":
                dependency = 51  # accepted watermark must not replace actual 34
        return Provenance(
            pixels,
            (SourceSpan(1, pixels.start, pixels.start, pixels.count),),
            dependency,
        )

    c, _ = temporal_codecs
    stream = StableVideoDecoder(c, capacity_latents=12, provenance_for=provenance_for)
    stream.append(torch.zeros(1, 3, 7, 1, 1), latent_start=0, source_dependency_end=34)
    retained = stream._latent
    with pytest.raises((ValueError, RuntimeError)):
        stream.pop()
    assert stream._latent is retained
    assert stream.emitted_frames == stream.latent_start == 0
    assert stream.buffered_latents == 7
    failing = False
    result = stream.pop()
    assert result.start == 0 and result.end == 17
    assert result.provenance.source_dependency_end == 34
    stream.dispose()


def test_long_decoder_releases_storage_and_autograd_history(temporal_codecs):
    import weakref

    c, calls = temporal_codecs
    stream = StableVideoDecoder(c, capacity_latents=12)
    previous = None
    for index in range(500):
        latent = torch.full((1, 3, 1, 1, 1), index / 500, requires_grad=True)
        stream.append(latent, latent_start=index, source_dependency_end=2000)
        while stream.ready:
            result = stream.pop()
            assert result.values.grad_fn is None
        calls.clear()  # fixture trace is not production codec ownership
        retained = stream._latent
        assert retained.grad_fn is None
        assert (
            retained.untyped_storage().nbytes()
            == retained.numel() * retained.element_size()
        )
        assert stream.buffered_latents <= 11
        if previous is not None:
            assert previous() is None
        previous = weakref.ref(retained)
        del retained
    stream.dispose()
    assert previous() is None


def test_streaming_posterior_sampling_preserves_partition_and_normalization(codecs):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_noise import (
        indexed_video_noise,
    )

    codecs.runtime = {"vae_config": {"posterior_seed": 23}}
    codecs.va = SimpleNamespace(latents_mean=[0.2, -0.3], latents_std=[0.5, 2.0])
    moments = torch.linspace(-2, 2, 4 * 13).reshape(1, 4, 13, 1, 1)
    kwargs = dict(noise_version="indexed-cpu-v1")
    full = codecs.sample_streaming_visual_moments(moments, latent_start=0, **kwargs)
    parts = [
        codecs.sample_streaming_visual_moments(
            moments[:, :, a:b], latent_start=a, **kwargs
        )
        for a, b in [(0, 5), (5, 6), (6, 13)]
    ]
    assert torch.equal(full, torch.cat(parts, 2))
    posterior = DiagonalGaussianDistribution(moments)
    epsilon = indexed_video_noise(
        tuple(posterior.mean.shape),
        seed=23,
        latent_start=0,
        purpose="source_posterior",
        device="cpu",
        dtype=moments.dtype,
    )
    sampled = (posterior.mean + posterior.std * epsilon).half().float()
    mean, std = codecs._stats(codecs.va, sampled)
    assert torch.equal(full, ((sampled - mean) / std).half())
