# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_bridge import (
    SourceChunkBridge,
)
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
from sglang.multimodal_gen.runtime.realtime.zing_contracts import Provenance, SourceSpan


def provenance(pixels, dependency):
    return Provenance(
        pixels, (SourceSpan(1, pixels.start, pixels.start, pixels.count),), dependency
    )


def bridge(plan):
    return SourceChunkBridge(
        plan,
        clip_frames=17,
        clip_latents=5,
        temporal_ratio=4,
        capacity_latents=10,
        capacity_bytes=1024,
        seed=42,
        provenance_for=provenance,
    )


@pytest.mark.parametrize("frames", list(range(1, 52)) + [240, 1440])
@pytest.mark.parametrize(
    "plan", [ChunkPlan(5), ChunkPlan([1, 2, 2]), ChunkPlan(5, True)]
)
def test_all_eos_tails_and_chunk_boundaries(frames, plan):
    b = bridge(plan)
    outputs, sources = [], []
    accepted = 0
    for start in range(0, frames, 17):
        pixels = min(17, frames - start)
        count = 1 + (pixels - 1 + 3) // 4
        source = (
            torch.arange(accepted, accepted + count).reshape(1, 1, count, 1, 1).float()
        )
        sources.append(source)
        b.append(
            source,
            latent_start=accepted,
            pixel_end=start + pixels,
            source_dependency_end=start + pixels,
        )
        accepted += count
        if start + pixels == frames:
            b.finish()
        while b.ready:
            outputs.append(b.pop())
        assert b.buffered_latents < max(plan.lengths)
        assert b.pop() is None
    assert outputs[-1].final
    assert outputs[0].provenance.pixels.start == 0
    assert outputs[-1].provenance.pixels.end == frames
    for a, c in zip(outputs, outputs[1:]):
        assert a.provenance.pixels.end == c.provenance.pixels.start
        assert a.chunk.end == c.chunk.start
    assert all(
        o.provenance.source_dependency_end >= o.provenance.pixels.end for o in outputs
    )
    torch.testing.assert_close(
        torch.cat([o.source for o in outputs], 2), torch.cat(sources, 2), rtol=0, atol=0
    )
    assert b.buffered_bytes == b.buffered_latents == 0
    b.dispose()
    assert not b.ready


def test_failed_provenance_keeps_owned_clip_and_rng():
    b = bridge(ChunkPlan(5))
    source = torch.ones(1, 1, 5, 1, 1)
    b.append(source, latent_start=0, pixel_end=17, source_dependency_end=17)
    source.zero_()
    b.provenance_for = lambda *_: (_ for _ in ()).throw(ValueError("bad mapping"))
    with pytest.raises(ValueError, match="bad mapping"):
        b.pop()
    assert b.chunk_id == b.emitted_pixels == 0 and b.buffered_latents == 5
    b.provenance_for = provenance
    assert b.pop().source.eq(1).all()


def test_target_noise_independent_of_ar_chunking():
    results = []
    for plan in [ChunkPlan(5), ChunkPlan([1, 2, 2])]:
        b, chunks = bridge(plan), []
        for i in range(2):
            b.append(
                torch.zeros(1, 1, 5, 1, 1),
                latent_start=i * 5,
                pixel_end=(i + 1) * 17,
                source_dependency_end=(i + 1) * 17,
            )
            while b.ready:
                chunks.append(b.pop().noise)
        results.append(torch.cat(chunks, 2))
    torch.testing.assert_close(*results, rtol=0, atol=0)


def test_requested_chunk_length_rephases_without_resetting_global_ranges():
    b = bridge(ChunkPlan([3, 2], split_first_latent=True))
    for start, pixel_end in ((0, 17), (5, 34)):
        b.append(
            torch.arange(start, start + 5).reshape(1, 1, 5, 1, 1).float(),
            latent_start=start,
            pixel_end=pixel_end,
            source_dependency_end=pixel_end,
        )
    lengths = [1, 2, 2, 3, 2]
    first = [b.pop(length) for length in lengths[:4]]
    assert [(item.chunk.start, item.chunk.end) for item in first] == [
        (0, 1),
        (1, 3),
        (3, 5),
        (5, 8),
    ]
    # A prompt epoch can request another local chunk-3 length here.  The wire
    # chunk id and absolute latent range continue instead of restarting.
    assert not b.ready_for(3)
    b.append(
        torch.arange(10, 13).reshape(1, 1, 3, 1, 1).float(),
        latent_start=10,
        pixel_end=43,
        source_dependency_end=43,
    )
    switched = b.pop(3)
    assert switched.chunk.chunk_id == 4
    assert (switched.chunk.start, switched.chunk.end) == (8, 11)
    assert switched.source.flatten().tolist() == [8, 9, 10]


@pytest.mark.parametrize("frames", range(1, 35))
@pytest.mark.parametrize(
    "plan", [ChunkPlan(5), ChunkPlan([1, 2, 2]), ChunkPlan(5, True)]
)
def test_progressive_slices_match_clip_baseline(frames, plan):
    results = []
    for progressive in [False, True]:
        b, chunks = bridge(plan), []
        for pixel_start in range(0, frames, 17):
            pixels = min(17, frames - pixel_start)
            count = 1 + (pixels - 1 + 3) // 4
            sizes = [1, 2, 2] if progressive else [count]
            offset = 0
            for size in sizes:
                size = min(size, count - offset)
                if not size:
                    break
                end = min(
                    pixel_start + pixels, pixel_start + 1 + (offset + size - 1) * 4
                )
                latent_start = b.accepted_latents
                b.append(
                    torch.arange(latent_start, latent_start + size)
                    .reshape(1, 1, size, 1, 1)
                    .float(),
                    latent_start=latent_start,
                    pixel_end=end,
                    # Keep the same audited lookahead in both paths for this
                    # alignment test; this is not an encoder latency claim.
                    source_dependency_end=pixel_start + pixels,
                )
                offset += size
                if end == frames:
                    b.finish()
                while b.ready:
                    chunks.append(b.pop())
        results.append(chunks)
    assert len(results[0]) == len(results[1])
    for baseline, progressive in zip(*results):
        assert baseline.chunk == progressive.chunk
        assert baseline.provenance == progressive.provenance
        assert baseline.final == progressive.final
        torch.testing.assert_close(baseline.source, progressive.source, rtol=0, atol=0)
        torch.testing.assert_close(baseline.noise, progressive.noise, rtol=0, atol=0)


def test_first_latent_and_dit_watermarks_are_independent():
    for plan, first_ready in [(ChunkPlan(5), False), (ChunkPlan([1, 2, 2]), True)]:
        b = bridge(plan)
        b.append(
            torch.ones(1, 1, 1, 1, 1),
            latent_start=0,
            pixel_end=1,
            source_dependency_end=3,
        )
        assert b.accepted_latents == 1
        assert b.ready is first_ready
        if first_ready:
            chunk = b.pop()
            assert chunk.provenance.pixels.end == 1
            assert chunk.provenance.source_dependency_end == 3
        b.append(
            torch.ones(1, 1, 2, 1, 1),
            latent_start=1,
            pixel_end=9,
            source_dependency_end=11,
        )
        b.append(
            torch.ones(1, 1, 2, 1, 1),
            latent_start=3,
            pixel_end=17,
            source_dependency_end=17,
        )
        assert b.ready


@pytest.mark.parametrize(
    "start,end,dependency,count",
    [(1, 1, 1, 1), (0, 2, 2, 1), (0, 9, 8, 3), (0, 17, 17, 6)],
)
def test_invalid_progressive_ranges_do_not_mutate(start, end, dependency, count):
    b = bridge(ChunkPlan([1, 2, 2]))
    with pytest.raises(ValueError):
        b.append(
            torch.zeros(1, 1, count, 1, 1),
            latent_start=start,
            pixel_end=end,
            source_dependency_end=dependency,
        )
    assert b.accepted_latents == b.accepted_pixels == b.buffered_bytes == 0


def test_short_eos_cannot_be_extended():
    b = bridge(ChunkPlan([1, 2, 2]))
    b.append(
        torch.zeros(1, 1, 2, 1, 1), latent_start=0, pixel_end=3, source_dependency_end=3
    )
    with pytest.raises(ValueError, match="final clip"):
        b.append(
            torch.zeros(1, 1, 1, 1, 1),
            latent_start=2,
            pixel_end=9,
            source_dependency_end=9,
        )
    b.finish()
    assert [b.pop().provenance.pixels.end, b.pop().provenance.pixels.end] == [1, 3]
