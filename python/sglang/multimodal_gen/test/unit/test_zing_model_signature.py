# SPDX-License-Identifier: Apache-2.0
"""Signature reuse must preserve the existing model mutation boundaries."""

import weakref

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.dits.zing_h3_model_signature import (
    model_signatures,
)


def _model():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    model.register_buffer("constant", torch.ones(4))
    return model


def _legacy(model):
    values = (*model.parameters(), *model.buffers())
    graph = tuple(
        (
            id(v),
            v.data_ptr(),
            None if torch.is_inference(v) else v._version,
            (tuple(v.shape), v.stride(), v.dtype, v.device),
        )
        for v in values
    )
    cache = tuple(
        (
            id(v),
            None if torch.is_inference(v) else v._version,
            tuple(v.shape),
            v.dtype,
            v.device,
        )
        for v in values
    )
    return graph, cache, tuple(v.data_ptr() for v in values)


def test_unchanged_signatures_are_shared_with_existing_plan_keys():
    model = _model()
    first = model_signatures(model)
    held_plan_keys = [model_signatures(model).graph for _ in range(100)]
    assert all(key is first.graph for key in held_plan_keys)
    assert model_signatures(model) is first
    assert (first.graph, first.cache, first.addresses) == _legacy(model)
    assert hash(first.graph) == hash(_legacy(model)[0])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m[0].weight.add_(1),
        lambda m: m.constant.add_(1),
        lambda m: m.load_state_dict(m.state_dict()),
        lambda m: setattr(m[0], "weight", nn.Parameter(m[0].weight.clone())),
        lambda m: m.register_buffer("constant", m.constant.clone()),
        lambda m: m.register_buffer("new_constant", torch.zeros(1)),
        lambda m: delattr(m[0], "bias"),
        lambda m: m.__setitem__(1, nn.Linear(4, 4)),
        lambda m: m.double(),
    ],
)
def test_mutations_rebuild_signatures_without_changing_the_old_key(mutation):
    model = _model()
    old = model_signatures(model)
    old_graph, old_cache, old_addresses = old.graph, old.cache, old.addresses
    with torch.no_grad():
        mutation(model)
    new = model_signatures(model)
    assert new is not old
    assert (new.graph, new.cache, new.addresses) == _legacy(model)
    assert (old.graph, old.cache, old.addresses) == (
        old_graph,
        old_cache,
        old_addresses,
    )
    assert model_signatures(model) is new


def test_data_storage_swap_rekeys_graph_even_with_unchanged_identity_and_version():
    model = _model()
    old = model_signatures(model)
    model[0].weight.data = model[0].weight.data.clone()
    new = model_signatures(model)
    assert new.graph != old.graph
    assert new.addresses != old.addresses
    # Preserve cache binding semantics; Graph keys additionally guard addresses.
    assert new.cache == old.cache


def test_data_stride_change_at_the_same_address_rekeys_graph():
    model = _model()
    old = model_signatures(model)
    model[0].weight.data = model[0].weight.data.t()
    new = model_signatures(model)
    assert new.graph != old.graph
    assert new.addresses == old.addresses
    assert (new.graph, new.cache, new.addresses) == _legacy(model)


def test_inference_parameters_keep_the_versionless_rejection_signal():
    with torch.inference_mode():
        model = _model()
    state = model_signatures(model)
    assert not state.versioned
    assert (state.graph, state.cache, state.addresses) == _legacy(model)


def test_snapshot_does_not_keep_replaced_weights_alive():
    model = _model()
    old = model_signatures(model)
    reference = weakref.ref(model[0].weight)
    model[0].weight = nn.Parameter(torch.zeros(4, 4))
    assert reference() is None
    assert model_signatures(model) is not old


def test_shared_modules_keep_the_legacy_deduplication_order():
    linear = nn.Linear(4, 4)
    model = nn.Sequential(linear, linear)
    state = model_signatures(model)
    assert len(state.graph) == 2
    assert (state.graph, state.cache, state.addresses) == _legacy(model)
