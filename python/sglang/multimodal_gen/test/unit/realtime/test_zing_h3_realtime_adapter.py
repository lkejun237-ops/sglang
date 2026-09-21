# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
    ZingH3CausalPipelineConfig,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import realtime_adapter
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
    zing_h3_realtime_adapter as zing,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    get_realtime_model_adapter,
    has_realtime_model_adapter,
)


@pytest.fixture
def initialized(monkeypatch, tmp_path):
    (tmp_path / "zing_config.json").write_text(
        json.dumps(
            {
                "num_frame_per_block": 5,
                "split_first_latent": False,
                "h3_media_config": {"fps": 24},
            }
        )
    )
    args = SimpleNamespace(
        model_path=str(tmp_path), pipeline_config=ZingH3CausalPipelineConfig()
    )
    monkeypatch.setattr(zing, "get_global_server_args", lambda: args)
    adapter = zing.ZingH3RealtimeAdapter()
    session = GenerateSession()
    session.set_adapter(adapter)
    request = RealtimeVideoGenerationsRequest(
        type="init", prompt="first", num_frames=22
    )
    asyncio.run(adapter.on_init(session, request))
    session.set_request(request)
    return adapter, session, args


def test_registry_is_zing_specific():
    assert isinstance(
        get_realtime_model_adapter(
            SimpleNamespace(pipeline_config=ZingH3CausalPipelineConfig())
        ),
        zing.ZingH3RealtimeAdapter,
    )
    assert not has_realtime_model_adapter(
        SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig())
    )


def test_init_sets_finite_block_count_and_model_geometry(initialized):
    _, session, _ = initialized
    assert session.request.max_chunks == 2
    assert session.request.size == "896x512"
    assert session.request.fps == 24
    assert type(session.request.fps) is int


def test_prompt_event_is_snapshotted_at_next_prepared_block(initialized, monkeypatch):
    adapter, session, args = initialized
    monkeypatch.setattr(
        realtime_adapter,
        "build_sampling_params",
        lambda request_id, **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        realtime_adapter,
        "prepare_request",
        lambda **kw: SimpleNamespace(sampling_params=kw["sampling_params"], extra={}),
    )
    first = adapter.prepare_next_request(session, args, session.new_chunk())
    adapter.ingest_event(
        session,
        RealtimeEvent(type="event", kind="prompt", payload="second", event_id=11),
    )
    assert first.sampling_params.prompt == "first" and first.realtime_event_id is None
    session.generate_chunk_completed()
    second = adapter.prepare_next_request(session, args, session.new_chunk())
    assert second.sampling_params.prompt == "second" and second.realtime_event_id == 11
    assert second.realtime_session_id == first.realtime_session_id
    assert second.extra["realtime_is_final_chunk"]
    assert second.return_raw_frames


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("camera_actions", []),
        ("source.append", {"uri": "/media/next.mp4"}),
        ("prompt", ""),
        ("prompt", 12),
    ],
)
def test_invalid_events_do_not_change_pending_prompt(initialized, kind, payload):
    adapter, session, _ = initialized
    adapter.ingest_event(
        session, RealtimeEvent(type="event", kind="prompt", payload="keep", event_id=4)
    )
    with pytest.raises(ValueError):
        adapter.ingest_event(
            session, RealtimeEvent(type="event", kind=kind, payload=payload, event_id=5)
        )
    assert session.adapter_state.pending_prompt == ("keep", 4)


@pytest.mark.parametrize(
    "fields",
    [
        {"zing_generate_audio": True},
        {"condition_inputs": {"camera_actions": [["w"]]}},
        {"enable_frame_interpolation": True},
        {"fps": 30},
        {"realtime_causal_kv_cache_num_frames": 8},
    ],
)
def test_init_rejects_unsupported_features(initialized, fields):
    adapter, session, _ = initialized
    request = RealtimeVideoGenerationsRequest(type="init", prompt="scene", **fields)
    with pytest.raises(ValueError):
        asyncio.run(adapter.on_init(session, request))


def test_custom_source_request_keeps_target_horizon(initialized):
    adapter, session, _ = initialized
    custom = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "replace sky"}]},
            {
                "role": "target",
                "content": [
                    {
                        "type": "video",
                        "output": {
                            "frames": 39,
                            "width": 896,
                            "height": 512,
                        },
                    }
                ],
                "source": {"type": "video", "uri": "/tmp/source.mp4"},
            },
        ]
    }
    request = RealtimeVideoGenerationsRequest(
        type="init", prompt="ignored", zing_request=custom
    )
    asyncio.run(adapter.on_init(session, request))
    assert request.num_frames == 39 and request.max_chunks == 3
    assert request.prompt == "replace sky"


def test_zing_frame_batch_preserves_wire_event_id(initialized):
    import msgspec.msgpack

    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch

    adapter, session, _ = initialized
    messages = []

    async def send_bytes(payload):
        messages.append(msgspec.msgpack.decode(payload))

    batch = SimpleNamespace(
        block_idx=1,
        request_id="zing-block-1",
        realtime_event_id=7,
        realtime_output_format="raw",
        realtime_trace_id="trace",
    )
    result = OutputBatch(
        raw_frame_batches=[[bytes(12), bytes(12)]],
        raw_frame_content_type="application/x-raw-rgb",
        raw_frame_metadata={"width": 2, "height": 2, "channels": 3},
    )
    asyncio.run(
        adapter.send_output(
            SimpleNamespace(send_bytes=send_bytes), session, result, batch
        )
    )
    assert messages[0]["type"] == "frame_batch"
    assert messages[0]["event_id"] == 7
    assert messages[0]["chunk_index"] == 1
    assert len(messages[0]["payload"]) == 24
    assert messages[0]["num_frames"] == 2


def test_stable_transport_preserves_each_original_block_and_request(initialized):
    import msgspec.msgpack

    from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )

    adapter, session, _ = initialized
    batch = Req(
        sampling_params=ZingH3SamplingParams(request_id="current", width=2, height=2)
    )
    batch.block_idx = 2
    batch.realtime_event_id = 9
    messages = []

    async def send(payload):
        messages.append(msgspec.msgpack.decode(payload))

    result = OutputBatch(
        raw_frame_batches=[[bytes(12)], [bytes(12)]],
        raw_frame_content_type="application/x-raw-rgb",
        raw_frame_metadata={"width": 2, "height": 2, "channels": 3},
        usage={
            "zing_realtime_blocks": [
                {"block_idx": 1, "event_id": 7, "request_id": "old"},
                {"block_idx": 2, "event_id": 9, "request_id": "current"},
            ]
        },
    )
    stats = asyncio.run(
        adapter.send_output(SimpleNamespace(send_bytes=send), session, result, batch)
    )
    assert [(m["chunk_index"], m["event_id"], m["request_id"]) for m in messages] == [
        (1, 7, "old"),
        (2, 9, "current"),
    ]
    assert (
        batch.block_idx == 2
        and batch.request_id == "current"
        and batch.realtime_event_id == 9
    )
    assert stats["num_frames"] == 2


def test_stable_waiting_step_sends_no_frames(initialized):
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch

    adapter, session, _ = initialized

    async def send_bytes(payload):
        raise AssertionError("A buffered step must not send a frame")

    batch = SimpleNamespace(
        block_idx=0,
        request_id="waiting",
        realtime_output_format="raw",
        enable_upscaling=False,
        width=896,
        height=512,
    )
    stats = asyncio.run(
        adapter.send_output(
            SimpleNamespace(send_bytes=send_bytes),
            session,
            OutputBatch(raw_frame_batches=[]),
            batch,
        )
    )
    assert stats["num_frames"] == 0


@pytest.mark.parametrize("size", ["512x896", "640x384", "1024x576"])
def test_explicit_size_is_not_replaced_by_default(initialized, size):
    adapter, session, _ = initialized
    request = RealtimeVideoGenerationsRequest(type="init", prompt="scene", size=size)
    asyncio.run(adapter.on_init(session, request))
    assert request.size == size


@pytest.mark.parametrize("size", [(640, 384), (512, 896)])
def test_reference_source_request_preserves_custom_geometry(initialized, size):
    adapter, session, _ = initialized
    width, height = size
    custom = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "keep character"},
                    {"id": "character", "type": "image", "uri": "/media/ref.png"},
                ],
            },
            {
                "role": "target",
                "content": [
                    {
                        "type": "video",
                        "output": {"frames": 243, "width": width, "height": height},
                    }
                ],
                "conditions": [
                    {
                        "content_id": "character",
                        "role": "reference",
                        "latent_modalities": ["visual"],
                    }
                ],
                "source": {"type": "video", "uri": "/media/source.mp4"},
            },
        ]
    }
    request = RealtimeVideoGenerationsRequest(
        type="init", prompt="keep character", zing_request=custom
    )
    asyncio.run(adapter.on_init(session, request))
    assert request.size == f"{width}x{height}"
    assert request.num_frames == 243
    assert request.prompt == "keep character"
    assert request.zing_request == custom


def test_prompt_latest_wins_without_mutating_source_request(initialized):
    adapter, session, args = initialized
    source = {"messages": [{"role": "target", "source": {"uri": "/media/source.mp4"}}]}
    session.request.zing_request = source
    for event_id, prompt in [(8, "first update"), (9, "last update")]:
        adapter.ingest_event(
            session,
            RealtimeEvent(
                type="event", kind="prompt", payload=prompt, event_id=event_id
            ),
        )
    inputs = adapter.sample_chunk_inputs(session, args, None, 5)
    assert inputs.prompt == "last update"
    assert session.adapter_state.event_id == 9
    assert session.adapter_state.pending_prompt is None
    assert session.request.zing_request == source


@pytest.mark.parametrize(
    "lengths,split", [([1, 2, 2], False), (5, True), ([3, 2], True)]
)
def test_adapter_uses_checkpoint_cycle_for_exact_chunk_sizes(
    initialized, lengths, split
):
    from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan

    adapter, session, args = initialized
    config = Path(args.model_path) / "zing_config.json"
    runtime = json.loads(config.read_text())
    runtime.update(num_frame_per_block=lengths, split_first_latent=split)
    config.write_text(json.dumps(runtime))
    request = RealtimeVideoGenerationsRequest(
        type="init", prompt="cycle", num_frames=39
    )
    asyncio.run(adapter.on_init(session, request))
    plan = ChunkPlan(lengths, split)
    assert request.max_chunks == plan.count(12)
    sizes = [
        adapter.get_chunk_size(session, args, SimpleNamespace(index=i))
        for i in range(request.max_chunks)
    ]
    assert sizes == [c.end - c.start for c in plan.chunks(12)]
    with pytest.raises(ValueError, match="complete"):
        adapter.get_chunk_size(session, args, SimpleNamespace(index=request.max_chunks))
