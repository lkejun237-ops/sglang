# SPDX-License-Identifier: Apache-2.0
"""Small CPU model integration; real-checkpoint CUDA acceptance is separate."""

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
    model,
    request,
)


@pytest.mark.parametrize("budget", [0, 1, 131072])
@torch.no_grad()
def test_projection_cache_preserves_every_forward_and_kv(model, budget):  # noqa: F811
    packer, inputs, visual, audio = request(model)
    snapshots = []
    for enabled in (False, True, True):
        model.adaln_projection_cache_enabled = enabled
        model.adaln_projection_cache_bytes = budget
        sampler = ZingH3Sampler(
            model,
            packer,
            solver="few_step",
            sigma_points=4,
            denoising_step_list=[1000, 750, 500, 250],
        )
        sampler.device = torch.device("cpu")
        records, caches, cache_stats = [], [], []
        original = sampler._forward

        def capture(packed, cache, update):
            output = original(packed, cache, update)
            tensors = [v.clone() for v in output]
            if cache is not None:
                caches.append(cache)
                if cache.adaln_projection_cache is not None:
                    cache_stats.append(cache.adaln_projection_cache.stats())
                positions = cache.positions()
                if positions is not None:
                    tensors.append(positions.clone())
                for layer in range(len(model.blocks)):
                    history = cache.history(layer)
                    if history is not None:
                        tensors.extend(v.clone() for v in history)
            records.append((update, tensors))
            return output

        sampler._forward = capture
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(135)
            output = sampler.sample(inputs, visual.clone(), audio.clone())
        records.append(("output", output))
        snapshots.append(records)
        assert all(c.adaln_projection_cache is None for c in caches)
        if enabled and budget == 131072:
            assert cache_stats[-1]["hits"] > 0
            assert cache_stats[-1]["bytes"] <= budget
    for candidate in snapshots[1:]:
        assert len(candidate) == len(snapshots[0])
        for (phase_a, tensors_a), (phase_b, tensors_b) in zip(snapshots[0], candidate):
            assert phase_a == phase_b and len(tensors_a) == len(tensors_b)
            for a, b in zip(tensors_a, tensors_b):
                assert a.shape == b.shape and a.dtype == b.dtype
                assert torch.isfinite(b).all() and torch.equal(a, b)


@torch.no_grad()
def test_tensor_weight_refresh_invalidates_live_session(model):  # noqa: F811
    model.adaln_projection_cache_enabled = True
    packer, inputs, visual, audio = request(model)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    sampler.device = torch.device("cpu")
    state = sampler.initialize(inputs, visual, audio)
    sampler.step(state)
    generation = model._cache_weight_generation
    model.refresh_weight_derived_caches(weights_path=None)
    assert model._cache_weight_generation == generation + 1
    with pytest.raises(ValueError, match="model changed"):
        sampler.step(state)
    assert state.cache.adaln_projection_cache is None
