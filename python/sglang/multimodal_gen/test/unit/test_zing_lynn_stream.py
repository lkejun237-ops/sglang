# SPDX-License-Identifier: Apache-2.0
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.model import (
    LynnDecoderConfig,
    StreamingDecoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder import (
    DECODERS,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_decoder import (
    SourceLynnDecoder,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import Provenance, SourceSpan

# H3 temporal protocol on a tiny ViT: 4x spatial ratio, 8px tiles with 4px overlap.
ARCHITECTURE = {
    "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
    "spatial_downsample_factors": [2, 2],
    "clip_length": 17,
    "token_drop": 3,
    "latent_channels": 24,
    "out_channels": 3,
    "decoder_num_layers": 2,
    "decoder_num_attention_heads": 2,
    "decoder_attention_head_dim": 16,
    "decoder_num_register_tokens": 2,
    "decoder_ffn_mult": 2,
    "decoder_rope_theta": 100.0,
    "decoder_rope_dim_ratio": 0.75,
    "decoder_norm_eps": 1e-5,
    "latents_mean": [0.1] * 24,
    "latents_std": [1.5] * 24,
}
CONFIG = LynnDecoderConfig(
    temporal_window=3,
    attention_backend="native",
    tile_size=8,
    tile_overlap=4,
    tile_batch_size=0,
    pixel_mean=(0.5, 0.5, 0.5),
    pixel_std=(0.5, 0.5, 0.5),
)
ADAPTER = DECODERS["lynn_streaming"]


@pytest.fixture(scope="module")
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        decoder = StreamingDecoder(CONFIG, ARCHITECTURE)
        with torch.no_grad():
            # Residual scales, phase embedding and register tokens start at zero,
            # which would hide the attention history from every comparison.
            for parameter in decoder.parameters():
                if not parameter.any():
                    parameter.normal_(std=0.5)
    decoder.to_compute("cpu", torch.bfloat16).eval().requires_grad_(False)
    yield decoder
    torch.set_num_threads(previous)


def _latent(count, *, seed=0, height=3, width=3):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 24, count, height, width, generator=generator)


def _latents_for(model, frames):
    count = 1
    while model.protocol.frame_count(count) < frames:
        count += 1
    return count


def _provenance(pixels, dependency):
    return Provenance(
        pixels,
        (SourceSpan(1, pixels.start, pixels.start, pixels.count),),
        dependency,
    )


def _source_decoder(model, *, capacity_latents=20, provenance_for=_provenance):
    codecs = SimpleNamespace(modules={ADAPTER.module_key: model}, decoder=ADAPTER)
    return SourceLynnDecoder(
        codecs, capacity_latents=capacity_latents, provenance_for=provenance_for
    )


def _append_valid(decoder, latent, *, latent_start, valid_frame_end, dependency=None):
    # Keep the regression executable on the old interface: fail on observable
    # output, rather than an unexpected-keyword TypeError.
    kwargs = dict(
        latent_start=latent_start,
        source_dependency_end=(valid_frame_end if dependency is None else dependency),
    )
    if "valid_frame_end" in inspect.signature(decoder.append).parameters:
        kwargs["valid_frame_end"] = valid_frame_end
    decoder.append(latent, **kwargs)


@pytest.mark.parametrize("frames", [1, 14])
def test_sp_acceptance_harness_source_path_cpu(model, monkeypatch, tmp_path, frames):
    """Exercise the acceptance caller with real Lynn streams and padded EOS."""
    import sglang.multimodal_gen.runtime.distributed as distributed
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        decoder as decoder_module,
    )

    benchmark_dir = Path(__file__).resolve().parents[5] / "benchmark" / "zing_h3_causal"
    monkeypatch.syspath_prepend(str(benchmark_dir))
    spec = importlib.util.spec_from_file_location(
        "lynn_sp_acceptance_cpu_smoke", benchmark_dir / "validate_lynn_sp_tiles.py"
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    rows = []

    class MemoryRecorder:
        def __init__(self, args, rank, world_size):
            self.env = {}

        def write(self, **row):
            rows.append(row)

        def save_env(self):
            pass

        def abort(self, reason):
            pytest.fail(reason)

    sample = SimpleNamespace(
        id=f"tiny-{frames}",
        latent=_latent(_latents_for(model, frames)),
        frames=frames,
        media_type="image" if frames == 1 else "video",
    )
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        distributed,
        "maybe_init_distributed_environment_and_model_parallel",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        distributed,
        "get_sp_group",
        lambda: SimpleNamespace(rank_in_group=0, device=torch.device("cpu")),
    )
    monkeypatch.setattr(decoder_module, "sp_tile_group", lambda: None)
    monkeypatch.setattr(ADAPTER, "build", lambda spec: {ADAPTER.module_key: model})
    monkeypatch.setattr(harness, "Recorder", MemoryRecorder)
    monkeypatch.setattr(harness, "load_inputs", lambda run: [sample])
    monkeypatch.setattr(harness, "save_frames", lambda path, values: None)

    harness.run_decode(
        SimpleNamespace(
            output=str(tmp_path),
            inputs=str(tmp_path),
            checkpoint="tiny-fixture.pt",
            checkpoint_sha256=None,
            model_dir=str(tmp_path),
            reference_output=None,
            lifecycle_sample=sample.id,
        )
    )

    path_rows = {row["variant"]: row for row in rows if row["gate"] == "paths"}
    assert set(path_rows) == {
        "offline-vs-preclamp",
        "finite-vs-offline",
        "source-vs-offline",
    }
    for row in path_rows.values():
        assert row["actual_shape"] == row["expected_shape"] == [1, 3, frames, 12, 12]
        assert row["equal"] and row["finite"]
        assert row["max_abs"] == 0 and row["mismatched"] == 0
    # This is computed by the real harness from the real source pop intervals:
    # starts at zero, no gap/overlap, ends exactly at the unpadded frame count.
    assert path_rows["source-vs-offline"]["intervals_ok"] is True


def test_chunked_stream_matches_offline_decode_bitwise(model):
    latent = _latent(10)
    stream, outputs, cursor = model.new_stream(), [], 0
    for size in (1, 2, 2, 3, 2):
        outputs.append(stream.push(latent[:, :, cursor : cursor + size]))
        cursor += size
    assert [value.shape[2] for value in outputs] == [1, 8, 8, 9, 8]
    assert stream.produced_frames == 34
    tail = stream.finish(34)
    assert tail.shape == (1, 3, 0, 12, 12) and stream.closed
    streamed = torch.cat(outputs, dim=2)
    assert streamed.shape == (1, 3, 34, 12, 12)
    assert 0 <= streamed.min() and streamed.max() <= 1
    torch.testing.assert_close(streamed, model.decode(latent, 34), atol=0, rtol=0)


def test_attention_history_reaches_later_frames(model):
    # Guards the fixture: a decoder without usable history would pass the
    # chunking comparisons while never exercising the cached K/V.
    latent = _latent(3)
    changed = latent.clone()
    changed[:, :, 0] += 1
    reference = model.new_stream().push(latent)[:, :, 1:]
    assert not torch.equal(model.new_stream().push(changed)[:, :, 1:], reference)


def test_streams_are_request_private(model):
    first, second = _latent(4, seed=1), _latent(4, seed=2)
    a, b = model.new_stream(), model.new_stream()
    interleaved = [[], []]
    for index in range(4):
        interleaved[0].append(a.push(first[:, :, index : index + 1]))
        interleaved[1].append(b.push(second[:, :, index : index + 1]))
    for values, latent in zip(interleaved, (first, second)):
        torch.testing.assert_close(
            torch.cat(values, dim=2), model.decode(latent, 13), atol=0, rtol=0
        )


@pytest.mark.parametrize("frames", [6, 7, 9])
def test_frame_limit_trims_the_tail_without_repeating_latents(model, frames):
    latent = _latent(3)
    full = model.new_stream().push(latent)
    assert full.shape[2] == 9
    limited = model.decode(latent, frames)
    assert limited.shape[2] == frames
    torch.testing.assert_close(limited, full[:, :, :frames], atol=0, rtol=0)

    stream = model.new_stream(frame_limit=frames)
    parts = [stream.push(latent[:, :, i : i + 1]) for i in range(3)]
    assert sum(value.shape[2] for value in parts) == frames
    torch.testing.assert_close(torch.cat(parts, dim=2), limited, atol=0, rtol=0)
    assert stream.finish(frames).shape[2] == 0


def test_single_image_decodes_one_frame(model):
    latent = _latent(1)
    image = model.decode(latent, 1)
    assert image.shape == (1, 3, 1, 12, 12)
    torch.testing.assert_close(image, model.new_stream().push(latent), atol=0, rtol=0)


@pytest.mark.parametrize("frames", [0, 5, 10])
def test_finish_rejects_lengths_the_latents_cannot_supply(model, frames):
    # Three latents cover frames 6..9; fewer would drop the last latent, more would need padding.
    stream = model.new_stream()
    stream.push(_latent(3))
    with pytest.raises(ValueError, match="inconsistent with latents"):
        stream.finish(frames)


def test_finish_rejects_empty_closed_and_partially_pushed_streams(model):
    with pytest.raises(ValueError, match="empty or closed"):
        model.new_stream().finish(1)
    with pytest.raises(ValueError, match="positive integer"):
        model.new_stream(frame_limit=0)

    stream = model.new_stream(frame_limit=7)
    assert stream.push(_latent(3)).shape[2] == 7
    # Nine frames are consistent with three latents, but two were never emitted.
    with pytest.raises(ValueError, match="every valid frame"):
        stream.finish(9)

    stream = model.new_stream()
    stream.push(_latent(1))
    stream.finish(1)
    with pytest.raises(ValueError, match="closed"):
        stream.push(_latent(1))
    with pytest.raises(ValueError, match="empty or closed"):
        stream.finish(1)


def test_push_rejects_malformed_latents_without_touching_the_stream(model):
    stream = model.new_stream()
    for bad in (torch.zeros(1, 24, 3, 3), torch.zeros(1, 24, 0, 3, 3)):
        with pytest.raises(ValueError, match="N24THW"):
            stream.push(bad)
    assert not stream.closed and stream.latents_seen == 0


def test_failed_push_closes_the_stream(model, monkeypatch):
    stream = model.new_stream()
    stream.push(_latent(1))
    decode_step, calls = model.decode_step, []

    def fail_on_second(z_t, state, group):
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return decode_step(z_t, state, group)

    monkeypatch.setattr(model, "decode_step", fail_on_second)
    with pytest.raises(RuntimeError, match="boom"):
        stream.push(_latent(3))
    assert stream.closed and stream.state is None
    with pytest.raises(ValueError, match="closed"):
        stream.push(_latent(1))


def test_geometry_change_is_rejected_and_closes_the_stream(model):
    stream = model.new_stream()
    stream.push(_latent(1))
    with pytest.raises(ValueError, match="geometry changed"):
        stream.push(_latent(1, height=2))
    assert stream.closed


def test_adapter_declares_stable_only_stream_with_no_lookahead(model):
    assert ADAPTER.stateful and ADAPTER.realtime_modes == ("stable",)
    assert ADAPTER.source_decoder is SourceLynnDecoder
    stable = [ADAPTER.stable_frames(model, count) for count in range(7)]
    assert stable == [0, 1, 5, 9, 13, 17, 18]
    stream = ADAPTER.new_stream(model, frame_limit=7)
    assert stream.frame_limit == 7 and stream.model is model


@pytest.mark.parametrize("frames", list(range(1, 35)))
@pytest.mark.parametrize("partition", [[1, 2, 2], [3, 2], [5], [3, 7, 2]])
def test_source_decoder_releases_all_valid_frames_and_matches_offline(
    model, frames, partition
):
    protocol = model.protocol
    latent = _latent(_latents_for(model, frames))
    decoder = _source_decoder(model)
    outputs, cursor, index, emitted = [], 0, 0, 0
    while cursor < latent.shape[2]:
        end = min(cursor + partition[index % len(partition)], latent.shape[2])
        dependency = min(frames, protocol.frame_count(end))
        _append_valid(
            decoder,
            latent[:, :, cursor:end],
            latent_start=cursor,
            valid_frame_end=dependency,
        )
        result = decoder.pop()
        # Every known valid frame leaves in this chunk, including the first frame.
        assert (result.start, result.end) == (emitted, dependency)
        assert result.values.shape[2] == result.end - result.start
        if result.end > result.start:
            assert result.provenance.source_dependency_end == dependency
        else:
            assert result.provenance is None
        assert decoder.held is None or decoder.held.shape[2] == 0
        assert decoder.emitted_frames == dependency
        assert decoder.pop() is None
        assert not decoder.ready and decoder.buffered_latents == 0
        outputs.append(result.values)
        emitted, cursor, index = result.end, end, index + 1
    decoder.finish(total_frames=frames)
    assert decoder.ready
    result = decoder.pop()
    outputs.append(result.values)
    assert (result.start, result.end) == (emitted, frames)
    assert result.provenance is None  # EOS only closes; all valid RGB already left.
    assert not decoder.ready and decoder.held is None and decoder.stream.closed
    torch.testing.assert_close(
        torch.cat(outputs, dim=2), model.decode(latent, frames), atol=0, rtol=0
    )
    decoder.dispose()
    assert decoder.closed


def test_source_decoder_checks_provenance_before_the_irreversible_decode(model):
    def stale(pixels, dependency):
        return _provenance(pixels, dependency + 1)

    decoder = _source_decoder(model, provenance_for=stale)
    _append_valid(decoder, _latent(2), latent_start=0, valid_frame_end=5)
    with pytest.raises(ValueError, match="Provenance must match"):
        decoder.pop()
    # Nothing was decoded, so the generation is still recoverable.
    assert decoder.stream.latents_seen == 0 and not decoder.stream.closed
    assert decoder.buffered_latents == 2 and decoder.emitted_frames == 0


def test_source_decoder_closes_after_a_failed_decode(model, monkeypatch):
    decoder = _source_decoder(model)
    _append_valid(decoder, _latent(2), latent_start=0, valid_frame_end=5)

    def fail(z_t, state, group):
        raise RuntimeError("boom")

    monkeypatch.setattr(model, "decode_step", fail)
    with pytest.raises(RuntimeError, match="boom"):
        decoder.pop()
    assert decoder.closed and decoder.stream.closed and not decoder.ready
    assert decoder.pop() is None
    with pytest.raises(ValueError, match="closed"):
        _append_valid(decoder, _latent(1), latent_start=2, valid_frame_end=9)


def test_source_decoder_capacity_bad_eos_and_dispose(model):
    decoder = _source_decoder(model, capacity_latents=3)
    assert not decoder.ready and decoder.pop() is None
    with pytest.raises(ValueError, match="does not match latent count"):
        decoder.finish(total_frames=1)
    _append_valid(decoder, _latent(2), latent_start=0, valid_frame_end=5)
    with pytest.raises(ValueError, match="contiguous"):
        _append_valid(decoder, _latent(1), latent_start=3, valid_frame_end=9)
    with pytest.raises(ValueError, match="N24THW"):
        _append_valid(
            decoder,
            torch.zeros(1, 23, 1, 3, 3),
            latent_start=2,
            valid_frame_end=9,
        )
    with pytest.raises(ValueError, match="pixel watermark"):
        _append_valid(
            decoder, _latent(1), latent_start=2, valid_frame_end=9, dependency=-1
        )
    with pytest.raises(ValueError, match="capacity exceeded"):
        _append_valid(decoder, _latent(2), latent_start=2, valid_frame_end=13)
    assert decoder.accepted_latents == 2 and decoder.buffered_latents == 2
    # Popping frees the bound: decoded latents are not retained.
    assert decoder.pop().end == 5
    _append_valid(decoder, _latent(3), latent_start=2, valid_frame_end=14)
    assert decoder.pop().end == 14
    # Five latents cover frames 14..17.
    for bad in (13, 18, True, 14.0):
        with pytest.raises(ValueError, match="final video frame count|latent count"):
            decoder.finish(total_frames=bad)
    decoder.finish(total_frames=14)
    with pytest.raises(ValueError, match="already closed"):
        decoder.finish(total_frames=14)
    with pytest.raises(ValueError, match="closed"):
        _append_valid(decoder, _latent(1), latent_start=5, valid_frame_end=18)
    result = decoder.pop()
    assert (result.start, result.end, result.values.shape[2]) == (14, 14, 0)
    assert decoder.pop() is None
    decoder.dispose()
    assert decoder.closed and decoder.stream.closed
    assert decoder.pending is None and decoder.held is None
    assert decoder.provenance_for is None


@pytest.mark.parametrize("bad", [-1, 0, 1, 6, True, 5.0])
def test_source_decoder_rejects_invalid_output_boundary_before_decode(model, bad):
    decoder = _source_decoder(model)
    with pytest.raises(ValueError):
        _append_valid(
            decoder, _latent(2), latent_start=0, valid_frame_end=bad, dependency=9
        )
    assert decoder.accepted_latents == decoder.stream.latents_seen == 0
    assert decoder.pending is None and not decoder.stream.closed


def test_source_decoder_partial_tail_cannot_regress_grow_or_accept_more(model):
    decoder = _source_decoder(model)
    _append_valid(decoder, _latent(3), latent_start=0, valid_frame_end=9, dependency=17)
    first = decoder.pop()
    assert (first.start, first.end) == (0, 9)
    _append_valid(
        decoder, _latent(2), latent_start=3, valid_frame_end=14, dependency=17
    )
    tail = decoder.pop()
    assert (tail.start, tail.end) == (9, 14)
    assert tail.provenance.source_dependency_end == 17
    assert decoder.pop() is None
    for total in (13, 15, 17):
        with pytest.raises(ValueError):
            decoder.finish(total_frames=total)
    with pytest.raises(ValueError):
        _append_valid(
            decoder, _latent(1), latent_start=5, valid_frame_end=18, dependency=18
        )
    assert decoder.stream.latents_seen == decoder.accepted_latents == 5
    decoder.finish(total_frames=14)
    assert decoder.pop().values.shape[2] == 0
    assert decoder.pop() is None
    decoder.dispose()


@pytest.mark.parametrize("bad", [8, 9, 14])
def test_source_decoder_rejects_nonadvancing_or_excess_boundary_atomically(model, bad):
    decoder = _source_decoder(model)
    _append_valid(decoder, _latent(3), latent_start=0, valid_frame_end=9)
    assert decoder.pop().end == 9
    with pytest.raises(ValueError):
        _append_valid(
            decoder, _latent(1), latent_start=3, valid_frame_end=bad, dependency=17
        )
    assert decoder.accepted_latents == decoder.stream.latents_seen == 3
    assert decoder.emitted_frames == 9 and decoder.pending is None
    assert decoder.pop() is None
    decoder.dispose()


def test_source_decoder_requires_explicit_pixel_validity(model):
    decoder = _source_decoder(model)
    # A dependency watermark cannot prove how many output pixels are valid.
    with pytest.raises((TypeError, ValueError)):
        decoder.append(_latent(2), latent_start=0, source_dependency_end=9)
    assert decoder.accepted_latents == decoder.stream.latents_seen == 0
    assert decoder.pending is None
    decoder.dispose()
