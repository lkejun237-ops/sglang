# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from adaln_projection_probe import ProjectionProbe, summarize


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 4)
        self.blocks = [SimpleNamespace(adaln_proj=self.projection)]
        self.final_layer = SimpleNamespace(adaln_proj=self.projection)
        self._cache_weight_generation = 0

    def forward(self, t, self_cache_update="active"):
        return self.projection(t[:, None])


@torch.no_grad()
def test_probe_counts_phases_and_detects_unversioned_numerical_change(tmp_path):
    model = Model()
    path = tmp_path / "ledger.jsonl"
    with ProjectionProbe(model, path):
        model(torch.tensor([0.5]), self_cache_update="source")
        model(torch.tensor([0.5]))
        model.projection.weight.add_(1)
        model(torch.tensor([0.5]))
    result = summarize(path)
    assert result["unique_keys"] == 2  # two slots
    assert result["phases"]["active"]["potential_hits"] == 4
    assert result["equal_key_output_conflicts"] == 2
    assert not model._forward_pre_hooks and not model.projection._forward_hooks
