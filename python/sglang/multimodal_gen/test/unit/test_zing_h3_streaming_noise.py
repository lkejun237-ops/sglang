# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_noise import (
    indexed_video_noise,
)


def noise(start, count, **kwargs):
    return indexed_video_noise(
        (1, 4, count, 2, 3),
        seed=42,
        latent_start=start,
        purpose=kwargs.pop("purpose", "source_posterior"),
        device="cpu",
        dtype=torch.float32,
        **kwargs,
    )


@pytest.mark.parametrize("batches", [[1] * 17, [5, 5, 5, 2], [3, 7, 1, 6]])
@pytest.mark.parametrize("purpose", ["source_posterior", "target_initial"])
def test_absolute_noise_is_batch_independent(batches, purpose):
    state = torch.random.get_rng_state().clone()
    expected = noise(0, 17, purpose=purpose)
    parts, start = [], 0
    for count in batches:
        parts.append(noise(start, count, purpose=purpose))
        # Replay after an unrelated session does not depend on RNG progress.
        noise(500, 2, purpose=purpose)
        assert torch.equal(parts[-1], noise(start, count, purpose=purpose))
        start += count
    assert torch.equal(expected, torch.cat(parts, dim=2))
    assert torch.equal(state, torch.random.get_rng_state())


def test_noise_domains_and_positions_are_distinct():
    assert not torch.equal(noise(0, 3), noise(0, 3, purpose="target_initial"))
    assert not torch.equal(noise(0, 3), noise(1, 3))


def test_unknown_noise_version_is_rejected():
    with pytest.raises(ValueError, match="version"):
        noise(0, 1, version="future")
