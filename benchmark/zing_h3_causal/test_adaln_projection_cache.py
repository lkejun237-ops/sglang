# SPDX-License-Identifier: Apache-2.0
"""CPU lifecycle coverage; this does not certify CUDA numerical parity."""

import importlib.util
from pathlib import Path

import pytest
import torch

PATH = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/multimodal_gen/runtime/models/dits/zing_h3_adaln_cache.py"
)
SPEC = importlib.util.spec_from_file_location("zing_adaln_cache_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Cache = MODULE.ZingH3AdalnCache


class Projection(torch.nn.Module):
    def __init__(self, modalities=4, parts=6):
        super().__init__()
        self.linear = torch.nn.Linear(3, modalities * parts * 5)
        self.modalities, self.parts = modalities, parts
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.linear(x).view(-1, self.parts * 5).chunk(self.parts, -1)


@pytest.mark.parametrize("modalities,parts", [(4, 6), (1, 2)])
@torch.no_grad()
def test_exact_original_batch_and_compact_storage(modalities, parts):
    embedder = torch.nn.Linear(1, 3)
    projection = Projection(modalities, parts)
    cache = Cache(100000)
    t = torch.tensor([0.0, 250.0, 500.0])
    x = embedder(t[:, None])
    key = cache.begin(t, x, embedder, 0)
    expected = projection(x)
    first = cache.project(0, projection, x, key)
    second = cache.project(0, projection, x, key)
    assert all(torch.equal(a, b) for a, b in zip(expected, second))
    assert first is second
    assert projection.calls == 2
    assert cache.bytes == t.numel() * modalities * parts * 5 * x.element_size()
    assert cache.hits == cache.misses == 1


@torch.no_grad()
def test_timestep_order_shape_and_layer_are_not_aliased():
    embedder = torch.nn.Linear(1, 3)
    projection = Projection()
    cache = Cache(100000)
    for t in (
        torch.tensor([0.0, 250.0]),
        torch.tensor([0.0, 500.0]),
        torch.tensor([250.0, 0.0]),
        torch.tensor([250.0]),
    ):
        x = embedder(t[:, None])
        key = cache.begin(t, x, embedder, 0)
        for slot in (0, 1):
            actual = cache.project(slot, projection, x, key)
            assert all(torch.equal(a, b) for a, b in zip(actual, projection(x)))
    assert cache.misses == 8 and cache.hits == 0


@pytest.mark.parametrize("change", ["weight", "embedder", "generation", "dtype"])
@torch.no_grad()
def test_mutations_and_migrations(change):
    embedder, projection = torch.nn.Linear(1, 3), Projection()
    cache = Cache(100000)
    t = torch.tensor([0.0, 250.0])
    x = embedder(t[:, None])
    key = cache.begin(t, x, embedder, 0)
    cache.project(0, projection, x, key)
    generation = 0
    if change == "weight":
        projection.linear.weight.add_(1)
    elif change == "embedder":
        embedder.weight.add_(1)
    elif change == "generation":
        generation = 1
        projection.linear.weight.data.add_(1)
    else:
        embedder.double()
        projection.double()
        t = t.double()
    x = embedder(t[:, None])
    key = cache.begin(t, x, embedder, generation)
    actual = cache.project(0, projection, x, key)
    assert all(torch.equal(a, b) for a, b in zip(actual, projection(x)))
    assert cache.hits == 0


@torch.no_grad()
def test_budget_eviction_clear_and_session_isolation():
    embedder, projection = torch.nn.Linear(1, 3), Projection()
    cache, other = Cache(480), Cache(480)
    for t in (torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([2.0])):
        x = embedder(t[:, None])
        key = cache.begin(t, x, embedder, 0)
        cache.project(0, projection, x, key)
        assert cache.bytes <= 480
    assert cache.evictions == 2
    assert other.bytes == 0
    cache.clear()
    assert cache.bytes == 0 and not cache.entries
    cache.max_bytes = 1
    key = cache.begin(t, x, embedder, 0)
    cache.project(0, projection, x, key)
    assert cache.bytes == 0 and cache.bypasses == 1


def test_grad_and_unversioned_weights_bypass():
    embedder, projection = torch.nn.Linear(1, 3), Projection()
    t = torch.tensor([0.0])
    cache = Cache(100000)
    x = embedder(t[:, None])
    assert cache.begin(t, x, embedder, 0) is None
    cache.project(0, projection, x, None)[0].sum().backward()
    assert projection.linear.weight.grad is not None
    with torch.inference_mode():
        unversioned = Projection()
        key = cache.begin(t, x, embedder, 0)
        cache.project(0, unversioned, x, key)
    assert cache.bytes == 0


@torch.no_grad()
def test_failed_build_is_not_published():
    embedder = torch.nn.Linear(1, 3)

    class Failing(Projection):
        def forward(self, x):
            super().forward(x)
            raise RuntimeError("test failure")

    projection = Failing()
    cache = Cache(100000)
    t = torch.tensor([0.0])
    x = embedder(t[:, None])
    key = cache.begin(t, x, embedder, 0)
    with pytest.raises(RuntimeError, match="test failure"):
        cache.project(0, projection, x, key)
    assert cache.bytes == 0 and not cache.entries


@torch.no_grad()
def test_lora_state_uses_uncached_projection():
    embedder, projection = torch.nn.Linear(1, 3), Projection()
    projection.linear.disable_lora = False
    cache = Cache(100000)
    t = torch.tensor([0.0])
    x = embedder(t[:, None])
    key = cache.begin(t, x, embedder, 0)
    cache.project(0, projection, x, key)
    projection.linear.disable_lora = True
    cache.project(0, projection, x, key)
    assert projection.calls == 2 and cache.bytes == 0


@torch.no_grad()
def test_projection_storage_replacement_invalidates_without_version_bump():
    embedder, projection = torch.nn.Linear(1, 3), Projection()
    cache = Cache(100000)
    t = torch.tensor([0.0])
    x = embedder(t[:, None])
    key = cache.begin(t, x, embedder, 0)
    cache.project(0, projection, x, key)
    projection.linear.weight.data = torch.zeros_like(projection.linear.weight)
    actual = cache.project(0, projection, x, key)
    assert all(torch.equal(a, b) for a, b in zip(actual, projection(x)))
    assert cache.hits == 0
