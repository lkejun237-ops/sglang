# SPDX-License-Identifier: Apache-2.0
"""Cache identity and projected output ownership; real-model parity uses the GPU benchmark."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3ConditioningProjection,
    MiniMaxH3Qwen3VLEncoder,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen_prefix import QwenPrefixCache


class CausalModel(nn.Module):
    """Deterministic causal oracle with in-place KV updates and explicit positions."""

    def __init__(self):
        super().__init__()
        self.language_model = SimpleNamespace(
            config=SimpleNamespace(
                num_key_value_heads=1, head_dim=2, hidden_size=2, num_attention_heads=1
            )
        )
        self.lengths = []
        self.fail = False

    def get_rope_index(self, ids, *args, **kwargs):
        return torch.arange(ids.numel()).view(1, 1, -1).expand(3, 1, -1), None

    def forward(
        self, input_ids, position_ids, past_key_values=None, use_cache=False, **kwargs
    ):
        class KV:
            def __init__(self):
                self.values = torch.zeros(0)

            def crop(self, n):
                self.values = self.values[:n]

        cache = past_key_values or KV()
        offset = cache.values.numel()
        assert torch.equal(
            position_ids[0, 0], torch.arange(offset, offset + input_ids.numel())
        )
        self.lengths.append(input_ids.numel())
        cache.values = torch.cat((cache.values, input_ids.flatten().float()))
        if self.fail:
            raise RuntimeError("injected failure after KV mutation")
        values = cache.values.cumsum(0)[offset:]
        hidden = torch.stack((values, values * 2), dim=-1).unsqueeze(0)
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=cache)


@pytest.fixture
def encoder(monkeypatch):
    monkeypatch.setattr(
        MiniMaxH3Qwen3VLEncoder, "device", property(lambda _: torch.device("cpu"))
    )
    obj = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
    nn.Module.__init__(obj)
    obj.model = CausalModel()
    obj.selected_lm_layer, obj.hidden_dim = 1, 2
    obj.image_token_id, obj.video_token_id = 90, 91
    obj.conditioning_projection = None
    return obj


def test_replacement_shortening_return_and_output_isolation(encoder):
    cache = QwenPrefixCache(1 << 20)
    sequence = [
        [1, 2, 3, 4, 5],
        [1, 2, 7, 4, 5],
        [1, 2, 8, 9, 4, 5],
        [1, 2],
        [1, 2, 3, 4, 5],
    ]
    for ids in sequence:
        ids = torch.tensor(ids)
        actual = encoder.encode_ids(ids, prefix_cache=cache)
        reused = cache.reused_rows
        expected = encoder.encode_ids(ids)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.fill_(123)
        assert not torch.equal(actual, cache.context)
        assert reused <= ids.numel()
    assert (
        encoder.model.lengths[2] == 3
    )  # replacement also recomputes unchanged trailing 4,5


def test_media_identity_budget_and_failure_invalidate(encoder):
    ids = torch.tensor([1, 2, 3])
    cache = QwenPrefixCache(1 << 20)
    encoder.encode_ids(ids, prefix_cache=cache, media_identity="a")
    encoder.encode_ids(ids, prefix_cache=cache, media_identity="b")
    assert cache.reused_rows == 0
    encoder.model.fail = True
    with pytest.raises(RuntimeError):
        encoder.encode_ids(
            torch.tensor([1, 2, 4]), prefix_cache=cache, media_identity="b"
        )
    assert cache.kv is None and cache.context is None
    encoder.model.fail = False
    encoder.encode_ids(ids, prefix_cache=cache)
    cache.max_bytes = 1
    encoder.encode_ids(ids, prefix_cache=cache)
    assert cache.kv is None and cache.reason == "budget"


def test_exact_hit_keeps_cache_storage_and_returns_owned_result(encoder):
    cache = QwenPrefixCache(1 << 20)
    ids = torch.tensor([1, 2, 3, 4])
    expected = encoder.encode_ids(ids, prefix_cache=cache)
    retained = (cache.kv.values, cache.ids, cache.positions, cache.context)
    actual = encoder.encode_ids(ids.clone(), prefix_cache=cache)
    assert encoder.model.lengths == [4]
    assert cache.reused_rows == 4 and cache.encoded_rows == 0
    assert all(
        before is after
        for before, after in zip(
            retained, (cache.kv.values, cache.ids, cache.positions, cache.context)
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.zero_()
    torch.testing.assert_close(cache.context, expected, rtol=0, atol=0)
    # Shortening must still compact the feature storage and crop the KV state.
    short = encoder.encode_ids(ids[:2], prefix_cache=cache)
    assert cache.kv.values.numel() == 2
    assert cache.context.untyped_storage().nbytes() == (
        cache.context.numel() * cache.context.element_size()
    )
    short.zero_()
    torch.testing.assert_close(cache.context, expected[:2], rtol=0, atol=0)
    restored = encoder.encode_ids(ids, prefix_cache=cache)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    assert encoder.model.lengths == [4, 2]


def test_projection_sink_only_applies_at_global_row_zero():
    projection = MiniMaxH3ConditioningProjection.__new__(
        MiniMaxH3ConditioningProjection
    )
    nn.Module.__init__(projection)
    projection.input_dim = projection.output_dim = 2
    projection.mean_in = projection.mean_out = torch.zeros(2)
    projection.std_in = projection.std_out = torch.ones(2)
    projection.weight = torch.eye(2)
    projection.layers = nn.ModuleList()
    projection.sink_out = torch.tensor([-7.0, -9.0])
    hidden = torch.arange(8).float().reshape(4, 2)
    full = projection(hidden)
    split = torch.cat((projection(hidden[:2]), projection(hidden[2:], token_offset=2)))
    torch.testing.assert_close(full, split, rtol=0, atol=0)


def test_cached_call_uses_real_backbone_with_native_arch_config(encoder):
    """Native ArchConfig omits optional HF output flags; exercise its real forward."""
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    from sglang.multimodal_gen.configs.models.encoders.minimax_h3_qwen3vl import (
        MiniMaxH3Qwen3VLArchConfig,
    )
    from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLModel

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = encoder.model.language_model.config
            self.embed_tokens = nn.Embedding(100, 2)

        def forward(self, *, inputs_embeds, **kwargs):
            return BaseModelOutputWithPast(
                last_hidden_state=inputs_embeds,
                past_key_values=DynamicCache(),
            )

    backbone = Qwen3VLModel.__new__(Qwen3VLModel)
    nn.Module.__init__(backbone)
    backbone.config = MiniMaxH3Qwen3VLArchConfig()
    backbone.language_model = Language()
    backbone.rope_deltas = None
    backbone.get_rope_index = encoder.model.get_rope_index
    encoder.model = backbone
    ids = torch.tensor([1, 2, 3])
    expected = encoder.encode_ids(ids)
    actual = encoder.encode_ids(ids, prefix_cache=QwenPrefixCache(1 << 20))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
