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


def test_standalone_encoder_prefix_lifecycle_and_metrics(encoder):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.conditioning.metrics import (
        ConditioningMetrics,
        install_metrics_route,
    )
    from sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server import (
        EncoderConditioning,
    )

    def encode(prompt, media, **kwargs):
        ids = torch.tensor([int(x) for x in prompt.split()])
        context = encoder.encode_ids(ids, **kwargs)
        return {
            "context": context,
            "row_role": torch.zeros(len(ids), dtype=torch.int64),
        }

    metrics = ConditioningMetrics()
    owner = EncoderConditioning(
        SimpleNamespace(encode_condition=encode),
        prefix_cache_bytes=1 << 20,
        metrics=metrics,
    )
    condition = owner.conditioner
    media = [{"type": "image", "frames": torch.zeros(3, 1, 2, 2, dtype=torch.uint8)}]
    first = condition("1 2 3", media)
    second = condition("1 2 4", media)
    assert encoder.model.lengths == [3, 1]
    cached = condition("1 2 4", media)
    torch.testing.assert_close(second["context"], cached["context"])
    assert encoder.model.lengths == [3, 1]
    assert metrics.encodes.labels("hit")._value.get() == 1
    assert metrics.encodes.labels("miss")._value.get() == 1
    assert metrics.results.labels("result_cache")._value.get() == 1
    assert metrics.tokens.labels("reused")._value.get() == 2

    changed = [{"type": "image", "frames": torch.ones_like(media[0]["frames"])}]
    condition("1 2 5", changed)
    assert encoder.model.lengths[-1] == 3
    encoder.model.fail = True
    with pytest.raises(RuntimeError, match="injected failure"):
        condition("1 2 6", changed)
    assert condition.prefix.kv is None
    encoder.model.fail = False
    condition("1 2 7", changed)
    owner.dispose()
    assert condition.prefix.kv is None
    condition("1 2 8", changed)
    assert encoder.model.lengths[-1] == 3
    torch.testing.assert_close(first["context"], encode("1 2 3", media)["context"])

    app = FastAPI()
    with pytest.raises(ValueError, match="credential"):
        install_metrics_route(app, metrics, "")
    install_metrics_route(app, metrics, "test-read-only")
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 403
        assert (
            client.get(
                "/metrics", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 403
        )
        response = client.get(
            "/metrics", headers={"Authorization": "Bearer test-read-only"}
        )
        assert response.status_code == 200
        assert (
            'world_model_conditioning_encodes_total{prefix_outcome="hit"} 1.0'
            in response.text
        )
        # No separate queue boundary exists here; do not fabricate zero wait samples.
        assert (
            "world_model_conditioning_queue_duration_seconds_count 0.0" in response.text
        )
    disabled = FastAPI()
    install_metrics_route(disabled, None, None)
    with TestClient(disabled) as client:
        assert client.get("/metrics").status_code == 404


def test_standalone_prefix_off_and_service_isolation(encoder):
    from sglang.multimodal_gen.runtime.conditioning.local_prefix import (
        SessionPrefixConditioner,
    )

    def encode(prompt, media, **kwargs):
        ids = torch.tensor([int(x) for x in prompt.split()])
        return {"context": encoder.encode_ids(ids, **kwargs), "row_role": ids * 0}

    def make(budget):
        return SessionPrefixConditioner(
            encode, "profile", prefix_bytes=budget, metrics=None, cache_bytes=0
        )

    off = make(0)
    off("1 2 3", [])
    off("1 2 4", [])
    assert encoder.model.lengths == [3, 3]
    one, two = make(1 << 20), make(1 << 20)
    one("1 2 3", [])
    two("1 2 4", [])
    one("1 2 5", [])
    assert encoder.model.lengths[-3:] == [3, 3, 1]


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


def test_prefix_metrics_over_real_http_service(encoder, monkeypatch):
    """Real TCP/client/service/cache path; only the LM is the CPU causal oracle."""
    import socket
    import threading
    import time
    import uuid

    import httpx
    import uvicorn
    from prometheus_client.parser import text_string_to_metric_families

    from sglang.multimodal_gen.runtime.conditioning.client import ConditioningClient
    from sglang.multimodal_gen.runtime.conditioning.inputs import (
        PREPROCESS_VERSION,
        content_identity,
    )
    from sglang.multimodal_gen.runtime.conditioning.service import ConditioningService

    class CPUEncoder:
        create_prefix_cache = staticmethod(QwenPrefixCache)

        def __call__(self, value):
            ids = torch.tensor([int(token) for token in value.prompt.split()])
            features = encoder.encode_ids(
                ids, prefix_cache=value.task.cache if value.task else None
            )
            return {
                "context": features.repeat(1, 2560).to(torch.bfloat16),
                "row_role": torch.zeros(ids.numel(), dtype=torch.int64),
            }

    profile = SimpleNamespace(
        profile_id="a" * 64, document={"preprocess_version": PREPROCESS_VERSION}
    )
    app = ConditioningService(
        profile=profile,
        encoder=CPUEncoder(),
        tokens={
            "token": {"tenant": "test"},
            "metrics-token": {"tenant": "ops", "admin": True},
        },
    )
    forward = app.queue._forward

    def change_cache_after_snapshot(value, key):
        result = forward(value, key)
        value.task.cache.reused_rows = value.task.cache.encoded_rows = 99
        return result

    monkeypatch.setattr(app.queue, "_forward", change_cache_after_snapshot)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.01)
        client = ConditioningClient(
            f"http://127.0.0.1:{listener.getsockname()[1]}",
            "token",
            profile.profile_id,
        )
        client.handshake()
        assert client.supports_prefix_tasks
        handle = client.open_task(uuid.uuid4().hex)
        for prompt in (
            "1 2 3 4",  # Cold: 0 reused / 4 encoded.
            "1 2 9 4",  # Replacement: 2 / 2.
            "1 2 9 4",  # Exact feature reuse: 4 / 0.
            "1 2",  # Shortening: 2 / 0.
            "8 2 9 4",  # First token changes: 0 / 4.
        ):
            result = client.encode(
                prompt,
                [],
                task_handle=handle,
                input_digest=content_identity(prompt, [], profile.profile_id),
            )
            assert type(result) is dict
            assert set(result) == {"context", "row_role"}
            assert result["context"].shape[0] == len(prompt.split())
        assert encoder.model.lengths == [4, 2, 4]
        client.release_task(handle)
        assert app.prefix_tasks.active is None
        url = f"http://127.0.0.1:{listener.getsockname()[1]}/metrics"
        with httpx.Client(trust_env=False) as scrape:
            assert scrape.get(url).status_code == 401
            assert (
                scrape.get(url, headers={"Authorization": "Bearer token"}).status_code
                == 403
            )
            response = scrape.get(
                url, headers={"Authorization": "Bearer metrics-token"}
            )
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        families = list(text_string_to_metric_families(response.text))
        names = [family.name for family in families]
        assert len(names) == len(set(names)), "Metric families must not be duplicated"
        assert "world_model_critical_path_stage_duration_seconds" in names
        samples = [
            sample
            for family in families
            if family.name.startswith("world_model_conditioning_")
            for sample in family.samples
        ]

        def value(name, **labels):
            return next(
                s.value for s in samples if s.name == name and s.labels == labels
            )

        prefix = "world_model_conditioning_"
        assert value(prefix + "encodes_total", prefix_outcome="hit") == 3
        assert value(prefix + "encodes_total", prefix_outcome="miss") == 2
        assert value(prefix + "encodes_total", prefix_outcome="unknown") == 0
        for kind, count in (("reused", 8), ("encoded", 10), ("total", 18)):
            assert value(prefix + "tokens_total", kind=kind) == count
        assert value(prefix + "reused_tokens_count") == 5
        assert value(prefix + "results_total", source="encoding") == 5
        assert all(
            set(s.labels) <= {"prefix_outcome", "kind", "source", "le"} for s in samples
        )
    finally:
        server.should_exit = True
        thread.join(10)
        listener.close()
        assert not thread.is_alive()
        assert app._closed
