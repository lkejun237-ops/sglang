# SPDX-License-Identifier: Apache-2.0
import math
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes.taeh3 import TAEH3Decoder
from sglang.multimodal_gen.runtime.models.vaes.taeh3.upstream import TAEHV
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_decoder import (
    SourceTAEH3Decoder,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import Provenance, SourceSpan


@pytest.fixture(scope="module")
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        upstream = TAEHV(None, arch_name="taeh3").eval()
    adapter = object.__new__(TAEH3Decoder)
    torch.nn.Module.__init__(adapter)
    adapter.decoder = upstream.decoder
    yield adapter
    torch.set_num_threads(previous)


@pytest.mark.parametrize("frames", [1, 2, 5, 16, 17, 18, 34, 51, 60, 240])
@pytest.mark.parametrize("partition", [[1, 2, 2], [5], [3, 7, 2]])
def test_unknown_length_source_decoder_matches_existing_decoder(
    model, frames, partition
):
    clips, tail = divmod(frames - 1, 17)
    latent = torch.randn(1, 24, clips * 5 + 1 + math.ceil(tail / 4), 1, 2)

    def provenance(pixels, dependency):
        return Provenance(
            pixels,
            (SourceSpan(1, pixels.start, pixels.start, pixels.count),),
            dependency,
        )

    decoder = SourceTAEH3Decoder(
        SimpleNamespace(modules={"taeh3_decoder": model}),
        capacity_latents=20,
        provenance_for=provenance,
    )
    outputs, cursor, index, emitted = [], 0, 0, 0
    while cursor < latent.shape[2]:
        end = min(cursor + partition[index % len(partition)], latent.shape[2])
        dependency = min(frames, math.ceil(end / 5) * 17)
        decoder.append(
            latent[:, :, cursor:end],
            latent_start=cursor,
            source_dependency_end=dependency,
        )
        result = decoder.pop()
        assert result.start == emitted
        if result.end > result.start:
            assert result.provenance.source_dependency_end == dependency
        outputs.append(result.values)
        emitted = result.end
        assert not decoder.ready and decoder.buffered_latents <= 5
        cursor, index = end, index + 1
    decoder.finish(total_frames=frames)
    result = decoder.pop()
    outputs.append(result.values)
    assert result.end == frames and not decoder.ready
    torch.testing.assert_close(
        torch.cat(outputs, dim=2), model.decode(latent, frames), atol=0, rtol=0
    )
    decoder.dispose()
    assert decoder.closed and not decoder.stream.memory


def test_capacity_bad_eos_and_cancel_release(model):
    decoder = SourceTAEH3Decoder(
        SimpleNamespace(modules={"taeh3_decoder": model}),
        capacity_latents=5,
        provenance_for=lambda *_: None,
    )
    latent = torch.zeros(1, 24, 4, 1, 2)
    decoder.append(latent, latent_start=0, source_dependency_end=13)
    with pytest.raises(ValueError, match="capacity"):
        decoder.append(latent, latent_start=4, source_dependency_end=26)
    assert decoder.accepted_latents == 4
    with pytest.raises(ValueError, match="latent count"):
        decoder.finish(total_frames=17)
    assert decoder.total_frames is None
    decoder.dispose()
    assert decoder.buffered_latents == 0 and not decoder.ready
    with pytest.raises(ValueError, match="closed"):
        decoder.append(latent, latent_start=4, source_dependency_end=26)
