# SPDX-License-Identifier: Apache-2.0
"""Control routing must preserve data delivery and the active model session."""

import ast
import asyncio
from collections import deque
from contextlib import nullcontext
from pathlib import Path

from sglang.multimodal_gen.runtime.realtime.polling_task_queue import PollingTaskQueue

LINK = (
    Path(__file__).resolve().parents[3]
    / "runtime/pipelines_core/stages/model_specific_stages/zing_h3/dit_link.py"
)


def link_classes():
    tree = ast.parse(LINK.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    scope = {
        "PollingTaskQueue": PollingTaskQueue,
        "asyncio": asyncio,
        "nullcontext": nullcontext,
    }
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(LINK), "exec"), scope)
    return scope["DiTLinkClient"], scope["DiTInputService"]


def test_lookup_preserves_interleaved_output_for_the_next_consumer():
    client_type, _ = link_classes()
    output = dict(type="dit.latents", transfer_id=0, session_id="s", generation_id="g")

    class Channel:
        replies = deque(
            [
                output,
                dict(
                    type="cache.result",
                    transfer_id=0,
                    session_id="s",
                    generation_id="g",
                    result={"hit": True},
                ),
                dict(type="input.processed", transfer_id=0),
            ]
        )

        def send(self, message):
            assert message["type"] == "cache.control"
            assert message["arguments"] == {"cache_key": "a" * 64}

        def receive(self):
            return self.replies.popleft()

    client = client_type(Channel())
    client.owner = ("s", "g")
    assert client.cache_control("lookup", cache_key="a" * 64) == {"hit": True}
    assert client.pending_results == [output]


def test_management_uses_cache_owner_without_creating_or_releasing_session():
    _, service_type = link_classes()
    replies, calls = [], []

    class Channel:
        def send(self, message):
            replies.append(message)

    def create(*args):
        raise AssertionError("Cache management must not construct a model session")

    def control(operation, arguments, *, busy):
        calls.append((operation, arguments, busy))
        return {"entries": 2}

    create.cache_control = control
    service = service_type(Channel(), create)

    async def run():
        service.queue.append(
            dict(type="cache.control", transfer_id=0, operation="status", arguments={})
        )
        await service.queue.run_once()
        service.owner = ("s", "g")
        active = service.worker = object()
        service.queue.append(
            dict(
                type="cache.control",
                transfer_id=1,
                operation="unload",
                arguments={},
                session_id="s",
                generation_id="g",
            )
        )
        await service.queue.run_once()
        assert service.worker is active
        assert service.owner == ("s", "g")

    asyncio.run(run())
    assert calls == [("status", {}, False), ("unload", {}, True)]
    assert [reply["type"] for reply in replies] == [
        "cache.result",
        "input.processed",
    ] * 2


def test_encoder_cache_hit_skips_qwen_and_reference_preparation():
    from types import SimpleNamespace as NS

    import torch

    from sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server import (
        EncoderConditioning,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_preparation import (
        prepared_message,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
        H3InferenceInputs,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A cache hit must not run Qwen or reference preparation")

    inputs = H3InferenceInputs(
        output_type="video",
        conditions=[],
        media=[],
        positive={
            "context": torch.ones(2, 4),
            "row_role": torch.zeros(2, dtype=torch.int64),
        },
        visual={"reference": torch.ones(1, 2, 1, 2, 2)},
        audio={},
        source_visual=None,
        visual_shape=(1, 2, 2),
        audio_shape=None,
        pixel_frame_count=1,
        sample_count=None,
        sample_rate=24000,
        condition_media=[{"frames": torch.zeros(3, 1, 2, 2, dtype=torch.uint8)}],
    )
    codecs = NS(
        encode_condition=forbidden,
        prepare=forbidden,
        modules={"video_vae": torch.nn.Linear(1, 1)},
    )
    encoder = EncoderConditioning(codecs)
    queries = []

    def query(operation, **arguments):
        queries.append((operation, arguments))
        return (
            {"identity": {}}
            if operation == "identity"
            else {"hit": True, "inputs": prepared_message(inputs)}
        )

    encoder.cache_query = query
    batch = NS(
        sampling_params=NS(
            prompt="p", zing_request={"prompt": "p", "media": [], "conditions": []}
        ),
        realtime_event_id=1,
        extra={
            "zing_source_command": {
                "config": {"width": 2, "height": 2, "fps": 24},
                "appends": [{"payload": {"data": bytes(12)}}],
            }
        },
    )
    output, notice = encoder.forward(batch)
    assert notice is None and output["cache_hit"] is True
    assert set(queries[1][1]) == {"cache_key", "prefill", "force"}
    assert len(queries[1][1]["cache_key"]) == 64
    assert torch.equal(encoder.inputs.visual["reference"], inputs.visual["reference"])
    assert encoder.inputs.condition_media[0]["frames"].device.type == "cpu"


def test_encoder_prefill_uses_existing_conditioner_and_never_starts_source_vae(
    monkeypatch,
):
    from types import SimpleNamespace as NS

    from sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server import (
        SourceEncoderExecutor,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        source_encoder,
    )
    from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import (
        inputs,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Prefill must not construct or run source VAE")

    monkeypatch.setattr(source_encoder, "create_source_buffer", forbidden)
    for hit in (False, True):
        calls, released = [], []
        prepared = inputs()

        class Conditioning:
            cache_key = "a" * 64
            cache_hit = hit
            cache_request = {"prompt": "p"}
            conditioner = NS(identity_state=NS(identity="fixture"))

            def forward(self, batch):
                self.inputs = prepared

            def dispose(self):
                released.append(True)

        class Link:
            def cache_control(self, operation, **arguments):
                if operation == "identity":
                    return {"identity": {}}
                assert operation == "prefill"
                assert (arguments["inputs"] is None) is hit
                assert arguments["game_id"] == "game"
                calls.append(operation)
                return {"cache_key": "a" * 64, "hit": hit, "entries": 1, "evictions": 0}

        executor = SourceEncoderExecutor(
            NS(spatial_ratio=1), conditioning=Conditioning(), dit_link=Link()
        )
        batch = NS(
            extra={
                "zing_prefill": {
                    "source_frame": bytes(12),
                    "force": not hit,
                    "game_id": "game",
                },
                "zing_source_command": {"config": {"width": 2, "height": 2, "fps": 24}},
            }
        )
        result = executor.forward([batch])
        assert result.usage["zing_prefill"]["hit"] is hit
        assert calls == ["prefill"] and released == [True]
        assert executor.encoder is None and executor.owner is None


def test_encoder_update_keys_match_prefill_after_applied_reference_replacement():
    from types import SimpleNamespace as NS

    from sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server import (
        EncoderConditioning,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        preparation_key,
    )

    def reference(identity):
        return dict(id=identity, type="image", payload=identity.encode())

    def request(prompt, identity):
        return dict(
            prompt=prompt,
            media=[reference(identity)],
            conditions=[
                dict(
                    content_id=identity, role="reference", latent_modalities=["visual"]
                )
            ],
        )

    codecs = NS(encode_condition=lambda *args: None)
    encoder = EncoderConditioning(codecs)
    queries = []

    def query(operation, **arguments):
        if operation == "identity":
            return {"identity": {}}
        queries.append(arguments["cache_key"])
        return {"hit": False}

    encoder.cache_query = query
    encoder.inputs = NS(conditions=request("p0", "old")["conditions"])
    config = dict(width=2, height=2, fps=24)
    batch = NS(
        sampling_params=NS(prompt="p0", zing_request=request("p0", "old")),
        extra={"zing_source_command": {"config": config}},
    )
    encoder._lookup(batch)
    encoder._lookup(
        batch, update=dict(event_id=1, prompt="p1", references=[reference("first")])
    )
    encoder.candidates[1] = NS(conditions=request("p1", "first")["conditions"])
    encoder.confirm_applied(1)

    # A second replacement must remove the first reference, rather than retain
    # the initial request's media and produce a key that explicit prefill lacks.
    encoder._lookup(
        batch, update=dict(event_id=2, prompt="p2", references=[reference("second")])
    )
    assert queries[-1] == preparation_key(
        codecs, request("p2", "second"), None, config, {}
    )

    # A rejected candidate must not change the references used by a later
    # text-only update's cache key.
    encoder.reject(2)
    encoder.confirm_applied(2)
    encoder._lookup(batch, update=dict(event_id=3, prompt="p3"))
    assert queries[-1] == preparation_key(
        codecs, request("p3", "first"), None, config, {}
    )
