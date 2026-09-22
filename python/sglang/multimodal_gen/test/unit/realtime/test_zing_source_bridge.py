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
