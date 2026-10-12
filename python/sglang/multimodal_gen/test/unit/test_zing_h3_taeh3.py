# SPDX-License-Identifier: Apache-2.0
"""TAEH3 temporal contract tests; no downloaded weights or GPU required."""

import math

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes.taeh3 import TAEH3Decoder
from sglang.multimodal_gen.runtime.models.vaes.taeh3.upstream import TAEHV


@pytest.fixture(scope="module")
def models():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        upstream = TAEHV(None, arch_name="taeh3").eval()
    adapter = object.__new__(TAEH3Decoder)
    torch.nn.Module.__init__(adapter)
    adapter.decoder = upstream.decoder
    yield upstream, adapter
    torch.set_num_threads(threads)


def latent_count(frames):
    full, tail = divmod(frames - 1, 17)
    return full * 5 + 1 + math.ceil(tail / 4)


def reference(upstream, latent, frames):
    count = latent.shape[2]
    padded = 5 if frames == count == 1 else max(7, count + (-(count + 3)) % 5)
    x = torch.cat((latent, latent[:, :, -1:].expand(-1, -1, padded - count, -1, -1)), 2)
    with torch.no_grad():
        return upstream.decode_video(
            x.transpose(1, 2), show_progress_bar=False
        ).transpose(1, 2)[:, :, :frames]


@pytest.mark.parametrize("frames", [1, 2, 5, 16, 17, 18, 33, 34, 35, 51, 60])
@pytest.mark.parametrize("partition", [[5], [1, 2, 2], [1], [3, 7, 2]])
def test_stream_matches_fixed_upstream(models, frames, partition):
    upstream, adapter = models
    latent = torch.randn(1, 24, latent_count(frames), 1, 2)
    expected = reference(upstream, latent, frames)
    stream = adapter.new_stream()
    pieces, cursor, index = [], 0, 0
    while cursor < latent.shape[2]:
        end = min(cursor + partition[index % len(partition)], latent.shape[2])
        pieces.append(stream.push(latent[:, :, cursor:end]))
        assert stream.tail is None or stream.tail.shape[2] <= 12
        assert stream.pending.shape[2] < 5
        cursor, index = end, index + 1
    pieces.append(stream.finish(frames))
    actual = torch.cat(pieces, dim=2)
    assert actual.shape == (1, 3, frames, 16, 32)
    assert actual.min() >= 0 and actual.max() <= 1
    # FP32 CPU batch-convolution tolerance, fixed before GPU validation.
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(adapter.decode(latent, frames), actual, atol=0, rtol=0)
    assert stream.closed and stream.tail is None and not stream.memory
    with pytest.raises(ValueError, match="closed"):
        stream.push(latent)


def test_interleaved_sessions_cancel_reset(models):
    upstream, adapter = models
    x, y = torch.randn(1, 24, 7, 1, 2), torch.randn(1, 24, 7, 1, 2)
    a, b = adapter.new_stream(), adapter.new_stream()
    a.push(x[:, :, :2])
    prefix = b.push(y[:, :, :5])
    a.close()
    a.reset()
    ax = torch.cat((a.push(x), a.finish(17)), 2)
    by = torch.cat((prefix, b.push(y[:, :, 5:]), b.finish(17)), 2)
    torch.testing.assert_close(ax, reference(upstream, x, 17), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(by, reference(upstream, y, 17), atol=2e-6, rtol=2e-5)


def test_invalid_input_and_empty_eos(models):
    stream = models[1].new_stream()
    with pytest.raises(ValueError, match="empty"):
        stream.finish(1)
    with pytest.raises(ValueError, match="N24THW"):
        stream.push(torch.zeros(1, 16, 1, 2, 2))


def test_each_latent_decoded_once_and_batch_isolation(models):
    upstream, adapter = models
    latent = torch.randn(2, 24, 12, 1, 2)
    stream = adapter.new_stream()
    consumed = []
    decode = stream._decode

    def observe(chunk):
        consumed.append(chunk.clone())
        return decode(chunk)

    stream._decode = observe
    output = [stream.push(chunk) for chunk in latent.split(1, dim=2)]
    output.append(stream.finish(39))
    assert [chunk.shape[2] for chunk in consumed] == [5, 5, 2]
    torch.testing.assert_close(torch.cat(consumed, 2), latent, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.cat(output, 2), reference(upstream, latent, 39), atol=2e-6, rtol=2e-5
    )


def test_explicit_precision_restores_caller_flags(models):
    adapter = models[1]
    latent = torch.randn(1, 24, 7, 1, 2)
    before = (
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
    )
    expected = adapter.decode(latent, 17)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = adapter.decode(latent, 17)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert before == (
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
    )


@pytest.mark.parametrize("frames", [1, 2, 5, 16, 17, 18, 33, 34, 35, 51, 60])
@pytest.mark.parametrize("partition", [[1, 2, 2], [1], [3, 7, 2]])
def test_experimental_progressive_preserves_native_contract(models, frames, partition):
    upstream, adapter = models
    latent = torch.randn(1, 24, latent_count(frames), 1, 2)
    stream = adapter.new_stream(progressive=True)
    pieces, calls, cursor, index = [], [], 0, 0
    decode = stream._decode

    def observe(chunk):
        calls.append(chunk.shape[2])
        return decode(chunk)

    stream._decode = observe
    while cursor < latent.shape[2]:
        end = min(cursor + partition[index % len(partition)], latent.shape[2])
        before = sum(calls)
        pieces.append(stream.push(latent[:, :, cursor:end]))
        assert sum(calls) - before == end - cursor
        assert stream.pending.shape[2] == 0
        assert stream.tail.shape[2] <= 12
        cursor, index = end, index + 1
    pieces.append(stream.finish(frames))
    assert all(n in (1, 2) for n in calls)
    actual = torch.cat(pieces, 2)
    assert actual.shape[2] == frames
    # Existing CPU FP32 convolution tolerance; not GPU exact acceptance.
    torch.testing.assert_close(
        actual, reference(upstream, latent, frames), atol=2e-6, rtol=2e-5
    )
    assert stream.closed and not stream.memory


def test_progressive_executes_before_rgb_tail_is_releasable(models):
    stream = models[1].new_stream(progressive=True)
    latent = torch.randn(1, 24, 5, 1, 2)
    assert stream.push(latent[:, :, :1]).shape[2] == 0
    assert stream.raw_frames == 4
    assert stream.push(latent[:, :, 1:3]).shape[2] == 0
    assert stream.raw_frames == 12
    assert stream.push(latent[:, :, 3:]).shape[2] == 5
    assert stream.raw_frames == 20
    stream.close()
    stream.reset()
    assert stream.progressive and stream.raw_frames == 0


@pytest.mark.parametrize("frames", [1, 2, 17, 18, 34, 51, 60])
def test_progressive_raw_trim_and_eos_exact_without_convolution(models, frames):
    latent = (
        torch.arange(latent_count(frames), dtype=torch.float32)
        .reshape(1, 1, -1, 1, 1)
        .expand(1, 24, -1, 1, 1)
        / 100
    )
    results = []
    for progressive in (False, True):
        stream = models[1].new_stream(progressive=progressive)

        # Four raw frames per latent; identical arithmetic for every partition.
        def raw_decode(x):
            return (
                x[:, :1]
                .transpose(1, 2)
                .repeat_interleave(4, dim=1)
                .expand(-1, -1, 12, 8, 8)
                .clone()
            )

        stream._decode = raw_decode
        chunks = [stream.push(x) for x in latent.split(1, dim=2)]
        chunks.append(stream.finish(frames))
        results.append(torch.cat(chunks, dim=2))
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
