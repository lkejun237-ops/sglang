# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
import torch
from test_zing_h3_source_adapter import PromptSocket, configured  # noqa: F401

from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
    ReleaseRealtimeSessionReq,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_source_encoder_adapter import (
    ZingSourceEncoderAdapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server import (
    EncoderConditioning,
    SourceEncoderExecutor,
)
from sglang.multimodal_gen.runtime.realtime.polling_task_queue import PollingTaskQueue


def test_encoder_uses_original_validation_and_explicit_capabilities(configured):  # noqa: F811
    async def run():
        adapter, session = ZingSourceEncoderAdapter(), GenerateSession()
        session.set_adapter(adapter)
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="test",
            size="32x32",
            zing_realtime_api_version=1,
            source_stream=dict(format="rgb24", width=32, height=32, fps=24),
        )
        await adapter.on_init(session, request)
        session.set_request(request)
        assert adapter.ready["output_kind"] == "source_moments"
        assert adapter.ready["encoded_components"] == ["source_vae", "qwen"]
        assert adapter.ready["prompt_updates"] is True
        assert adapter.ready["prompt_reference_updates"] is False
        with pytest.raises(ValueError, match="consecutive"):
            await adapter.ingest_wire_event(
                None,
                session,
                dict(
                    type="event",
                    kind="source.append",
                    event_id=1,
                    payload=dict(
                        client_seq=0,
                        source_frame_start=2,
                        num_frames=1,
                        pts_us=0,
                        data=bytes(3072),
                    ),
                ),
                3200,
            )
        socket = PromptSocket()
        await adapter.ingest_wire_event(
            socket,
            session,
            dict(type="event", kind="prompt", event_id=1, payload="new prompt"),
            100,
        )
        assert socket.messages == [
            dict(type="control_ack", stage="worker", kind="prompt", event_id=1)
        ]
        assert not adapter.has_control_work()  # S0 is still needed for Qwen.
        adapter.dispatched_source_frames = 1
        await asyncio.wait_for(adapter.wait_for_next_chunk(session), timeout=1)
        assert (
            adapter.sample_chunk_inputs(session, configured, None, 17).prompt
            == "new prompt"
        )
        assert not adapter.has_control_work()
        with pytest.raises(ValueError, match="non-empty"):
            await adapter.ingest_wire_event(
                socket,
                session,
                dict(type="event", kind="prompt", event_id=2, payload=" "),
                100,
            )
        adapter.dispose(session)

    asyncio.run(run())


def test_prompt_is_polled_without_source_and_keeps_original_ack(configured):  # noqa: F811
    async def run():
        adapter, session = ZingSourceEncoderAdapter(), GenerateSession()
        session.set_adapter(adapter)
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="first",
            size="32x32",
            zing_realtime_api_version=1,
            source_stream=dict(format="rgb24", width=32, height=32, fps=24),
        )
        await adapter.on_init(session, request)
        session.set_request(request)
        adapter.dispatched_source_frames = 1
        waiting = asyncio.create_task(adapter.wait_for_next_chunk(session))
        await asyncio.sleep(0)
        assert not waiting.done()
        socket = PromptSocket()
        for event, prompt in [(1, "second"), (2, "latest")]:
            await adapter.ingest_wire_event(
                socket,
                session,
                dict(type="event", kind="prompt", event_id=event, payload=prompt),
                100,
            )
        await asyncio.wait_for(waiting, timeout=1)
        assert len(socket.messages) == 2
        assert (
            adapter.sample_chunk_inputs(session, configured, None, 17).prompt
            == "latest"
        )
        assert adapter._selected_prompt_command["event_id"] == 2
        adapter.dispose(session)

    asyncio.run(run())


@pytest.mark.parametrize("work", ["source", "codec", "prompt", "eos"])
def test_encoder_polls_every_five_ms_without_arrival_notifications(monkeypatch, work):
    async def run():
        adapter = ZingSourceEncoderAdapter()
        adapter.ledger = SimpleNamespace(closed=False)
        adapter.pending_frames = 0
        adapter.more_work = False
        adapter._pending_prompt_command = None
        adapter.dispatched_source_frames = 1
        adapter.end, adapter.end_dispatched = None, False
        adapter.pending = PollingTaskQueue(
            adapter._process_encoder_batch, ready=adapter._encoder_ready
        )
        # No available Event: polling must not depend on set()/wait().
        delays = []

        async def tick(delay):
            delays.append(delay)
            if len(delays) == 2:
                if work == "source":
                    adapter.pending_frames = 17
                    adapter.pending.append({"payload": "source"})
                elif work == "codec":
                    adapter.more_work = True
                elif work == "prompt":
                    adapter._pending_prompt_command = {"prompt": "updated"}
                else:
                    adapter.end = {"frame_end": 1}
                    adapter.pending.append(dict(kind="source.end", end=adapter.end))
            assert len(delays) <= 2

        monkeypatch.setattr(asyncio, "sleep", tick)
        await adapter.wait_for_next_chunk(None)
        assert delays == [0.005, 0.005]
        # A backlog remains runnable on subsequent iterations without input.
        for _ in range(3):
            await adapter.wait_for_next_chunk(None)
        assert len(delays) == 2
        adapter.ledger.closed = True
        with pytest.raises(asyncio.CancelledError):
            await adapter.wait_for_next_chunk(None)

    asyncio.run(run())


def test_encoder_queue_calls_original_execution_path(configured, monkeypatch):  # noqa: F811
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
        realtime_video_api,
    )

    async def run():
        adapter, session = ZingSourceEncoderAdapter(), GenerateSession()
        session.set_adapter(adapter)
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="test",
            size="32x32",
            zing_realtime_api_version=1,
            source_stream=dict(format="rgb24", width=32, height=32, fps=24),
        )
        await adapter.on_init(session, request)
        session.set_request(request)
        calls = []
        batch, client, result = object(), object(), object()

        async def execute(received_batch, received_client):
            calls.append((received_batch, received_client))
            return None, result

        monkeypatch.setattr(
            realtime_video_api, "_process_realtime_generation_batch", execute
        )
        await adapter.ingest_wire_event(
            PromptSocket(),
            session,
            dict(
                type="event",
                kind="source.append",
                event_id=1,
                payload=dict(
                    client_seq=0,
                    source_frame_start=0,
                    num_frames=1,
                    pts_us=0,
                    data=bytes(3072),
                ),
            ),
            3200,
        )
        assert isinstance(adapter.pending, PollingTaskQueue)
        assert len(adapter.pending) == 1
        await adapter.wait_for_next_chunk(session)
        # The existing request builder consumes input before model dispatch.
        task = adapter.pending.popleft()
        assert task["kind"] == "source.group"
        assert task["appends"][0]["payload"]["source_frame_start"] == 0
        assert await adapter.process_batch(batch, client) == (None, result)
        assert calls == [(batch, client)]
        await adapter.close_conditioning(session)
        assert adapter._s0_reservation is None
        adapter.dispose(session)

    asyncio.run(run())


def test_encoder_continuation_does_not_depend_on_dit_work_quantum(monkeypatch):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_h3_source_adapter import (
        ZingH3SourceStreamAdapter,
    )

    calls = []

    def complete(adapter, session, result):
        calls.append(result)
        adapter.more_work = False  # Original DiT path with work quantum disabled.

    monkeypatch.setattr(ZingH3SourceStreamAdapter, "on_chunk_complete", complete)
    adapter = ZingSourceEncoderAdapter()
    for more_work in (True, False):
        result = SimpleNamespace(usage={"zing_source_more_work": more_work})
        adapter.on_chunk_complete(None, result)
        assert adapter.more_work is more_work
    assert len(calls) == 2


def test_encoder_groups_across_packets_keeps_credit_and_queues_eos(configured):  # noqa: F811
    async def run():
        adapter, session = ZingSourceEncoderAdapter(), GenerateSession()
        session.set_adapter(adapter)
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="test",
            size="32x32",
            zing_realtime_api_version=1,
            source_stream=dict(format="rgb24", width=32, height=32, fps=24),
        )
        await adapter.on_init(session, request)
        session.set_request(request)
        socket = PromptSocket()

        async def append(start, count, seq):
            await adapter.ingest_wire_event(
                socket,
                session,
                dict(
                    type="event",
                    kind="source.append",
                    event_id=seq + 1,
                    payload=dict(
                        client_seq=seq,
                        source_frame_start=start,
                        num_frames=count,
                        pts_us=start * 1_000_000 // 24,
                        data=bytes(count * 3072),
                    ),
                ),
                count * 3072 + 100,
            )

        try:
            await append(0, 3, 0)
            lease = object()
            adapter._source_reservations[0] = lease
            appends, end = adapter._take_source_input()
            assert end is None and sum(x["payload"]["num_frames"] for x in appends) == 1
            assert adapter._source_reservations[0] is lease
            assert adapter.pending_frames == 2 and not adapter._encoder_ready(
                adapter.pending
            )
            assert (
                adapter.ledger.consume(1) is None
            )  # Original 3-frame packet is incomplete.
            await append(3, 5, 1)
            assert not adapter.pending and adapter.input_arranger.buffered_frames == 7
            await append(8, 1, 2)
            assert len(adapter.pending) == 1
            appends, end = adapter._take_source_input()
            assert [x["payload"]["num_frames"] for x in appends] == [2, 5, 1]
            assert end is None and adapter.pending_frames == 0
            assert 0 not in adapter._source_reservations
            assert adapter._inflight_reservations.pop() is lease
            assert adapter.ledger.consume(9)["source_frame_end"] == 9
            await append(9, 3, 3)
            assert not adapter.pending  # Partial second eight-frame group.
            await adapter.ingest_wire_event(
                socket,
                session,
                dict(
                    type="event",
                    kind="source.end",
                    event_id=5,
                    payload=dict(source_frame_end=12),
                ),
                100,
            )
            assert [t["kind"] for t in adapter.pending] == [
                "source.group",
                "source.end",
            ]
            appends, end = adapter._take_source_input()
            assert end is None and appends[0]["payload"]["num_frames"] == 3
            assert not adapter.end_dispatched
            appends, end = adapter._take_source_input()
            assert appends == [] and end["frame_end"] == 12
            assert adapter.end_dispatched and adapter.pending_frames == 0
        finally:
            # Test-only identity markers carry no actual resource reservation.
            adapter._source_reservations.pop(0, None)
            adapter._inflight_reservations.clear()
            await adapter.close_conditioning(session)
            adapter.dispose(session)
        assert not adapter.pending and not adapter.input_arranger.pending
        assert adapter.input_arranger.buffered_frames == 0

    asyncio.run(run())


def test_qwen_initial_condition_reused_updates_are_transactional_and_release_resets():
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    encoded, prepared = [], []

    def encode(prompt, media):
        encoded.append((prompt, media))
        if prompt == "fail":
            raise ConditionError("CONDITION_INVALID", "Test rejected condition")
        return dict(
            context=torch.full((2, 3), float(len(prompt))),
            row_role=torch.tensor([0, 1]),
        )

    def prepare(request, use, **kwargs):
        prepared.append(kwargs)
        media = [kwargs["source_preview"]]
        return SimpleNamespace(
            positive=kwargs["conditioning"](request["prompt"], media),
            condition_media=media,
        )

    worker, vae_calls = executor()
    condition = EncoderConditioning(
        SimpleNamespace(prepare=prepare, encode_condition=encode)
    )
    worker.conditioning = condition

    def dispatch(index, start, count, prompt="first", event=None, **kwargs):
        request = batch(index, start, count, **kwargs)
        b = request[0]
        b.sampling_params = SimpleNamespace(
            prompt=prompt, zing_request=dict(prompt=prompt)
        )
        b.realtime_event_id, b.realtime_prompt_version = event, event
        b.extra["zing_source_command"]["config"]["fps"] = 24
        return worker.forward(request)

    initial = dispatch(0, 0, 1)
    assert initial.realtime_handoff["conditioning"]["event_id"] is None
    assert len(prepared) == len(encoded) == 1
    assert prepared[0]["retain_condition_media"] is True
    assert encoded[0][1][0]["origin"] == "source_s0"
    assert encoded[0][1][0]["frames"].shape == (3, 1, 1, 1)
    media = condition.inputs.condition_media
    assert dispatch(1, 1, 1).realtime_handoff["conditioning"] is None
    assert len(encoded) == 1
    updated = dispatch(2, 2, 0, "second", 3)
    assert updated.realtime_handoff["conditioning"]["event_id"] == 3
    assert encoded[-1][1] is media
    assert len(encoded) == 2
    # Same text/new event reuses the existing content-keyed Qwen cache.
    assert (
        dispatch(3, 2, 0, "second", 4).realtime_handoff["conditioning"]["event_id"] == 4
    )
    assert len(encoded) == 2
    active = condition.inputs.positive
    failure = dispatch(4, 2, 0, "fail", 5)
    assert failure.conditioning_notice == dict(code="CONDITION_INVALID", event_id=5)
    assert condition.inputs.positive is active and condition.prompt == "second"
    assert dispatch(5, 2, 0, "fail", 5).conditioning_notice is None
    assert len(encoded) == 3  # A failed candidate is not retried on each source block.
    end = dispatch(6, 2, 0, "second", 6, end=True)
    assert end.usage["zing_source_complete"] and len(vae_calls) == 1
    worker.forward(ReleaseRealtimeSessionReq(session_id="s"))
    assert condition.inputs is None and condition.failed_update is None
    dispatch(0, 0, 1, "next", session="next", end=True)
    assert len(prepared) == 2 and encoded[-1][0] == "next"


def executor():
    # An instrumented codec checks orchestration only; GPU replay checks actual
    # model values. Use the real SourcePixelBuffer, including EOS tail padding.
    calls = []

    def encode(pixels):
        calls.append(pixels.clone())
        return torch.ones(1, 2, 5, 1, 1)

    return SourceEncoderExecutor(
        SimpleNamespace(
            vae_frames_per_chunk=17,
            vae_latents_per_chunk=5,
            temporal_ratio=4,
            encode_visual_moments=encode,
        )
    ), calls


def batch(index, start, count, *, end=False, session="s", generation="g"):
    return [
        SimpleNamespace(
            request_id=f"request-{index}",
            realtime_session_id=session,
            realtime_generation_id=generation,
            block_idx=index,
            extra={
                "zing_source_command": dict(
                    encoder={"mode": "clip17"},
                    config=dict(width=1, height=1),
                    appends=[
                        dict(
                            payload=dict(
                                data=bytes(count * 3),
                                num_frames=count,
                                source_frame_start=start,
                            )
                        )
                    ]
                    if count
                    else [],
                    end={"frame_end": start + count} if end else None,
                )
            },
        )
    ]


def test_eos_drains_tail_and_release_allows_fresh_session():
    worker, calls = executor()
    first = worker.forward(batch(0, 0, 16))
    assert not first.realtime_handoff["source_moments"]
    tail = worker.forward(batch(1, 16, 0, end=True))
    assert tail.usage["zing_source_complete"]
    result = tail.realtime_handoff["source_moments"][0]
    assert (
        result["latent_start"],
        result["latent_end"],
        result["source_dependency_end"],
    ) == (0, 5, 16)
    assert calls[0].shape[1] == 17
    worker.forward(ReleaseRealtimeSessionReq(session_id="s"))
    again = worker.forward(batch(0, 0, 1, end=True, session="next"))
    assert again.realtime_handoff["source_moments"][0]["latent_start"] == 0


def test_wrong_owner_and_generation_cannot_touch_live_cache():
    worker, _ = executor()
    worker.forward(batch(0, 0, 1))
    active = worker.encoder
    for kwargs in (dict(session="other"), dict(generation="old")):
        with pytest.raises(ValueError, match="identity"):
            worker.forward(batch(1, 1, 1, **kwargs))
        assert active.accepted_frames == 1
    worker.forward(ReleaseRealtimeSessionReq(session_id="other"))
    assert worker.encoder is active
    worker.forward(ReleaseRealtimeSessionReq(session_id="s"))
    assert active.disposed


def test_eos_follows_all_appends_in_dispatch():
    worker, _ = executor()
    request = batch(0, 0, 17, end=True)
    request[0].extra["zing_source_command"]["appends"] += batch(0, 17, 3)[0].extra[
        "zing_source_command"
    ]["appends"]
    result = worker.forward(request)
    assert [
        v["source_dependency_end"] for v in result.realtime_handoff["source_moments"]
    ] == [17, 20]
    assert result.usage["zing_source_complete"]


@pytest.mark.parametrize("injected", [False, True])
def test_cancel_drains_selected_executor_before_releasing_session(
    monkeypatch, injected
):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
        realtime_video_api as api,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch

    async def run():
        started, finish = asyncio.Event(), asyncio.Event()
        events = []

        async def release(request):
            assert isinstance(request, ReleaseRealtimeSessionReq)
            events.append("release")

        client = SimpleNamespace(forward=release)
        session = GenerateSession()
        if injected:
            session.execution_client = client
        else:
            monkeypatch.setattr(api, "async_scheduler_client", client)

        async def process(selected, batch):
            assert selected is client
            started.set()
            await finish.wait()
            events.append("finished")
            return [], OutputBatch(raw_frame_batches=[])

        monkeypatch.setattr(api, "process_generation_batch", process)
        task = asyncio.create_task(
            api._process_realtime_generation_batch(
                object(),
                client if injected else None,
            )
        )
        await started.wait()
        cleanup = asyncio.create_task(
            api._cleanup_realtime_session(session, task, None)
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not events and not cleanup.done()
        finish.set()
        await cleanup
        assert events == ["finished", "release"]
        assert task.cancelled()

    asyncio.run(run())


def test_connected_encoder_sends_qwen_before_vae_and_keeps_native_tensors():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
        H3InferenceInputs,
    )

    worker, vae_calls = executor()
    sent = []
    positive = dict(context=torch.ones(2, 3), row_role=torch.tensor([0, 1]))
    inputs = H3InferenceInputs(
        output_type="video",
        conditions=[],
        media=[],
        positive=positive,
        visual={},
        audio={},
        source_visual=None,
        visual_shape=(5, 1, 1),
        audio_shape=None,
        pixel_frame_count=17,
        sample_count=None,
        sample_rate=24000,
    )

    class Condition:
        def __init__(self):
            self.inputs = inputs

        def forward(self, b, on_condition=None):
            from dataclasses import fields

            message = dict(
                tensors=positive,
                event_id=b.block_idx or None,
                prompt_version=b.block_idx,
            )
            if b.block_idx == 0:
                message["inputs"] = {
                    f.name: getattr(inputs, f.name)
                    for f in fields(inputs)
                    if f.name not in ("positive", "source_visual", "condition_media")
                }
            on_condition("conditioning", **message)
            return None, None

        def dispose(self):
            pass

    def send(kind, **fields):
        if kind == "conditioning":
            assert len(vae_calls) == (0 if not sent else 1)
            assert fields["tensors"]["context"] is positive["context"]
            assert ("inputs" in fields) == (not sent)
        elif kind == "source.moments":
            assert len(vae_calls) == 1 and isinstance(fields["tensor"], torch.Tensor)
        sent.append(kind)
        return []

    worker.conditioning = Condition()
    worker.dit_link = SimpleNamespace(
        send=send, release=lambda owner: sent.append("cancel")
    )
    request = batch(0, 0, 17)
    request[0].extra["zing_dit_source_records"] = [[1, 0, 17]]
    worker.forward(request)
    assert sent == ["conditioning", "source.moments"]
    # Prompt-only dispatch must not repeat first-message metadata or VAE work.
    worker.forward(batch(1, 17, 0))
    assert sent == ["conditioning", "source.moments", "conditioning"]
    worker.forward(batch(2, 17, 0, end=True))
    assert sent[-1] == "source.end"
    worker.forward(ReleaseRealtimeSessionReq(session_id="s"))
    assert sent[-1] == "cancel" and worker.owner is None


@pytest.mark.parametrize(
    "source,target", [((32, 32), (32, 32)), ((832, 480), (1280, 736))]
)
def test_connected_api_initializes_dit_before_any_source(configured, source, target):  # noqa: F811
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_encoder_dit_adapter import (
        ZingEncoderDiTAdapter,
    )

    async def run():
        width, height = target
        source_width, source_height = source
        sent = []

        async def open_dit(message):
            sent.append(message)

        adapter, session = ZingEncoderDiTAdapter(), GenerateSession()
        session.set_adapter(adapter)
        session.execution_client = SimpleNamespace(open_dit=open_dit)
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="first",
            size=f"{width}x{height}",
            seed=None,
            zing_realtime_api_version=1,
            source_stream=dict(
                format="rgb24", width=source_width, height=source_height, fps=24
            ),
        )
        await adapter.on_init(session, request)
        assert not adapter.pending and not adapter.dispatched_source_frames
        assert sent == [
            dict(
                session_id=session.id,
                generation_id=session.generation_id,
                async_inputs=True,
                seed=42,
                width=width,
                height=height,
                source_stream=dict(
                    width=source_width, height=source_height, fps=24, format="rgb24"
                ),
                source_encoding=adapter.source_geometry.receipt(),
                fps=24,
                total_frames=None,
                realtime_output_format=request.realtime_output_format,
                output_compression=getattr(request, "output_compression", None),
                realtime_preview_max_width=getattr(
                    request, "realtime_preview_max_width", None
                ),
                trace_id=session.trace_id,
            )
        ]
        assert adapter.ready["output_kind"] == "dit_latents"
        adapter.dispose(session)

    asyncio.run(run())


@pytest.mark.parametrize("stale", [False, True])
def test_decoder_rgb_handoff_uses_original_sender_and_cancels_stale_reference(
    configured,  # noqa: F811
    tmp_path,
    monkeypatch,
    stale,
):
    from sglang.multimodal_gen.runtime.disaggregation.transport.shm_publisher import (
        SharedMemoryFramePublisher,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_encoder_dit_adapter import (
        ZingEncoderDiTAdapter,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_video_api import (
        _cancel_unconsumed_output_ref,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
    from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange

    monkeypatch.setenv("SGLANG_REALTIME_VAE_SHM_DIR", str(tmp_path / "shm"))

    async def run():
        adapter, session, socket = (
            ZingEncoderDiTAdapter(),
            GenerateSession(),
            PromptSocket(),
        )
        session.set_adapter(adapter)

        async def open_dit(message):
            return [dict(type="session.opened", output_kind="rgb")]

        session.execution_client = SimpleNamespace(open_dit=open_dit)
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="test",
            size="32x32",
            zing_realtime_api_version=1,
            source_stream=dict(format="rgb24", width=32, height=32, fps=24),
        )
        await adapter.on_init(session, request)
        session.set_request(request)
        assert adapter.ready["output_kind"] == "rgb"
        await adapter.ingest_wire_event(
            socket,
            session,
            dict(
                type="event",
                kind="source.append",
                event_id=1,
                payload=dict(
                    client_seq=0,
                    source_frame_start=0,
                    num_frames=1,
                    pts_us=0,
                    data=bytes(3072),
                ),
            ),
            3200,
        )
        await adapter.ingest_wire_event(
            socket,
            session,
            dict(
                type="event",
                kind="source.end",
                event_id=2,
                payload=dict(source_frame_end=1),
            ),
            100,
        )
        provenance = adapter.ledger.provenance_for(PixelRange(0, 1), 1)
        publisher = SharedMemoryFramePublisher(timeout_seconds=2, max_in_flight=10)
        reference = publisher.reserve(
            frames=1,
            width=32,
            height=32,
            identity=dict(
                session_id=session.id,
                generation_id="stale" if stale else session.generation_id,
                request_id="request",
                chunk_index=0,
            ),
        )
        publisher.publish(reference, torch.ones(1, 3, 1, 32, 32))
        item = dict(
            type="decoder.frames",
            reference=reference,
            metadata=dict(
                format="rgb24", width=32, height=32, channels=3, bytes_per_frame=3072
            ),
            session_id=session.id,
            generation_id=session.generation_id,
            latent_start=0,
            latent_end=1,
            block_idx=0,
            event_id=7,
            prompt_version=2,
            **provenance.metadata(),
        )
        result = OutputBatch(
            realtime_handoff={"dit_latents": [item]},
            usage=dict(zing_source_consumed_end=1, zing_source_complete=True),
        )
        b = SimpleNamespace(
            request_id="request",
            block_idx=0,
            extra={},
            fps=24,
            realtime_output_format="raw",
        )
        try:
            if stale:
                with pytest.raises(ValueError, match="identity mismatch"):
                    await adapter.send_output(socket, session, result, b)
                assert adapter.ledger.output_frames == 0
            else:
                stats = await adapter.send_output(socket, session, result, b)
                frame = next(m for m in socket.messages if m["type"] == "frame_batch")
                assert frame["payload"] == bytes([255]) * 3072
                assert frame["event_id"] == 7 and frame["prompt_version"] == 2
                assert stats["num_frames"] == 1
                assert socket.messages[-1]["type"] == "session.completed"
            assert item["reference"] is None
        finally:
            _cancel_unconsumed_output_ref(result=result)
            publisher.close()
            adapter.dispose(session)

    asyncio.run(run())
    assert not list((tmp_path / "shm").glob("*"))


def test_direct_decoder_receipts_only_return_input_progress(configured):  # noqa: F811
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_encoder_dit_adapter import (
        ZingEncoderDiTAdapter,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
    from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange

    async def run():
        adapter, session, socket = (
            ZingEncoderDiTAdapter(),
            GenerateSession(),
            PromptSocket(),
        )
        session.set_adapter(adapter)

        async def open_dit(message):
            return [
                dict(
                    type="session.opened",
                    output_kind="rgb",
                    output_attachment={"token": "t"},
                )
            ]

        session.execution_client = SimpleNamespace(open_dit=open_dit)
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="test",
            size="32x32",
            zing_realtime_api_version=1,
            source_stream=dict(format="rgb24", width=32, height=32, fps=24),
        )
        await adapter.on_init(session, request)
        session.set_request(request)
        try:
            assert adapter.ready["output_transport"] == "decoder_websocket"
            await adapter.ingest_wire_event(
                socket,
                session,
                dict(
                    type="event",
                    kind="source.append",
                    event_id=1,
                    payload=dict(
                        client_seq=0,
                        source_frame_start=0,
                        num_frames=1,
                        pts_us=0,
                        data=bytes(3072),
                    ),
                ),
                3200,
            )
            await adapter.ingest_wire_event(
                socket,
                session,
                dict(
                    type="event",
                    kind="source.end",
                    event_id=2,
                    payload=dict(source_frame_end=1),
                ),
                100,
            )
            adapter._take_source_input()
            provenance = adapter.ledger.provenance_for(PixelRange(0, 1), 1)
            result = OutputBatch(
                realtime_handoff={
                    "dit_latents": [
                        dict(
                            type="decoder.output_queued",
                            session_id=session.id,
                            generation_id=session.generation_id,
                            latent_start=0,
                            latent_end=1,
                            **provenance.metadata(),
                        )
                    ]
                },
                usage=dict(zing_source_consumed_end=1, zing_source_complete=True),
            )
            stats = await adapter.send_output(
                socket, session, result, SimpleNamespace()
            )
            assert stats["num_frames"] == 0
            assert socket.messages[-1]["type"] == "input.completed"
            assert not any(
                m["type"]
                in (
                    "frame_batch",
                    "frame_batch_header",
                    "session.completed",
                    "source.consumed",
                )
                for m in socket.messages
            )
        finally:
            adapter.dispose(session)

    asyncio.run(run())


@pytest.mark.parametrize("direct_output", [False, True])
def test_decoder_credit_can_arrive_before_encoder_dispatch_returns(
    configured, direct_output
):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_encoder_dit_adapter import (
        ZingEncoderDiTAdapter,
    )

    async def run():
        adapter, session, socket = (
            ZingEncoderDiTAdapter(),
            GenerateSession(),
            PromptSocket(),
        )
        session.set_adapter(adapter)

        async def open_dit(message):
            return [
                dict(
                    type="session.opened",
                    output_kind="rgb",
                    output_attachment={"token": "t"} if direct_output else None,
                )
            ]

        session.execution_client = SimpleNamespace(open_dit=open_dit)
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="test",
            size="32x32",
            zing_realtime_api_version=1,
            source_stream=dict(format="rgb24", width=32, height=32, fps=24),
        )
        await adapter.on_init(session, request)
        session.set_request(request)

        async def append(index):
            await adapter.ingest_wire_event(
                socket,
                session,
                dict(
                    type="event",
                    kind="source.append",
                    event_id=index + 1,
                    payload=dict(
                        client_seq=index,
                        source_frame_start=index,
                        num_frames=1,
                        pts_us=index * 1_000_000 // 24,
                        data=bytes(3072),
                    ),
                ),
                3200,
            )

        try:
            capacity = adapter.ledger.limits.pending_frames
            for index in range(capacity):
                await append(index)
            appends, _ = adapter._take_source_input()
            assert sum(item["payload"]["num_frames"] for item in appends) == 1
            # Decoder's credit can reach the caller while this dispatch is
            # still in flight. No Encoder result/send_output has happened yet.
            if direct_output:
                await append(capacity)
                assert adapter.ledger.consumed_frames == 1
                assert adapter.pending_frames == capacity
                with pytest.raises(ValueError, match="source credit"):
                    await append(capacity + 1)
            else:
                with pytest.raises(ValueError, match="source credit"):
                    await append(capacity)
            assert not any(m["type"] == "source.consumed" for m in socket.messages)
        finally:
            adapter.dispose(session)

    asyncio.run(run())


def test_encoder_reuses_reference_upload_validation_and_clear(configured):
    from test_zing_h3_source_adapter import _enable_reference_updates, _reference

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.conditioning import (
        release_api_media,
    )

    _enable_reference_updates(configured)

    async def run():
        adapter, session = ZingSourceEncoderAdapter(), GenerateSession()
        session.set_adapter(adapter)
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="original",
            size="32x32",
            zing_realtime_api_version=1,
            source_stream=dict(format="rgb24", width=32, height=32, fps=24),
            references=[_reference("old")],
        )
        ws = PromptSocket()
        try:
            await adapter.on_init(session, request)
            session.set_request(request)
            assert adapter.ready["prompt_reference_updates"] is True
            assert adapter.reference_items[0]["id"] == "old"
            for event, refs in [(1, [_reference("new")]), (2, [])]:
                await adapter.ingest_wire_event(
                    ws,
                    session,
                    dict(
                        type="event",
                        kind="prompt",
                        event_id=event,
                        payload="updated",
                        references=refs,
                    ),
                    100,
                )
                assert ws.messages[-1]["type"] == "control_ack"
                assert len(adapter._pending_prompt_command["references"]) == len(refs)
            await adapter.ingest_wire_event(
                ws,
                session,
                dict(
                    type="event",
                    kind="prompt",
                    event_id=3,
                    payload="invalid",
                    references=[dict(id="bad", format="png", payload=b"invalid")],
                ),
                100,
            )
            assert ws.messages[-1]["type"] == "error"
            assert ws.messages[-1]["fatal"] is False
            assert adapter._pending_prompt_command["rejected"]
            assert adapter.reference_items[0]["id"] == "old"
        finally:
            await adapter.close_conditioning(session)
            release_api_media(session)
            adapter.dispose(session)

    asyncio.run(run())


def test_encoder_reference_candidates_use_original_codec_and_commit_on_output():
    from copy import copy

    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
        H3InferenceInputs,
    )

    media = [dict(id="old", type="image")]
    calls, sent = [], []
    positive = dict(context=torch.ones(2, 3), row_role=torch.tensor([0, 1]))
    initial = H3InferenceInputs(
        "video",
        [],
        media,
        positive,
        {"old": torch.ones(1, 2, 1, 1, 1)},
        {},
        None,
        (5, 1, 1),
        None,
        17,
        None,
        24000,
        condition_media=media,
    )

    def encode(prompt, current):
        calls.append((prompt, current))
        return positive

    def update(active, references, prompt, use, *, conditioning):
        assert active is initial or active.condition_media == []
        if prompt == "invalid":
            raise ConditionError("CONDITION_INVALID", "Invalid reference candidate")
        candidate = copy(active)
        candidate.condition_media = references
        candidate.visual = {
            item["id"]: torch.full((1, 2, 1, 1, 1), 2.0) for item in references
        }
        candidate.positive = conditioning(prompt, references)
        return candidate

    condition = EncoderConditioning(
        SimpleNamespace(
            prepare=lambda *a, **k: initial,
            encode_condition=encode,
            prepare_reference_update=update,
        ),
        encode_tensor=lambda x: x,
    )

    def run(event, refs=None, prompt="same", rejected=False, remote_rejected=False):
        b = batch(0 if condition.inputs is None else 1, 0, 1)[0]
        b.extra["zing_source_command"]["config"]["fps"] = 24
        b.sampling_params = SimpleNamespace(prompt="initial", zing_request={})
        b.realtime_event_id, b.realtime_prompt_version = None, 0
        command = dict(prompt=prompt, event_id=event)
        if refs is not None:
            command["references"] = refs
        if rejected:
            command["rejected"] = True
        b.extra["zing_prompt_update"] = command

        def publish(kind, **message):
            sent.append(dict(type=kind, **message))
            if remote_rejected:
                raise ConditionError(
                    "CONDITION_INVALID", "Reference update exceeds KV capacity"
                )

        return condition.forward(b, on_condition=publish)

    # A reference event queued before S0 remains separate from initialization.
    run(1, [dict(id="new", type="image")], prompt="initial")
    assert [m["event_id"] for m in sent] == [None, 1]
    # Unchanged text plus new images still sends Qwen and VAE together, once.
    assert set(sent[1]["tensors"]) == {"context", "row_role"}
    assert calls[-1] == ("initial", [dict(id="new", type="image")])
    assert set(sent[0]["inputs"]["visual"]) == {"old"}
    assert set(sent[1]["inputs"]["visual"]) == {"new"}
    assert condition.inputs is initial
    run(2, prompt="invalid", refs=[])
    assert sent[-1]["type"] == "conditioning.cancel"
    condition.confirm_applied(1)  # Superseded candidate must not become active.
    assert condition.inputs is initial
    run(3, [])
    assert len(sent) == 4
    assert sent[-1]["inputs"]["visual"] == {}
    assert set(sent[-1]["tensors"]) == {"context", "row_role"}
    condition.confirm_applied(3)
    assert condition.inputs.condition_media == []
    run(4, prompt="text only")
    assert len(sent) == 5
    assert "inputs" not in sent[-1]
    assert calls[-1][1] == []
    run(5, rejected=True)
    assert sent[-1]["type"] == "conditioning.cancel"
    assert condition.inputs.condition_media == []
    run(6, [dict(id="combined", type="image")], prompt="new text and image")
    assert len(sent) == 7
    assert set(sent[-1]["inputs"]["visual"]) == {"combined"}
    assert set(sent[-1]["tensors"]) == {"context", "row_role"}
    assert calls[-1] == ("new text and image", [dict(id="combined", type="image")])

    # A DiT admission rejection uses the existing notice and never re-sends
    # cancellation with the same event ID. Active references remain intact.
    before = len(sent)
    _, notice = run(7, [], prompt="too large", remote_rejected=True)
    assert notice == dict(code="CONDITION_INVALID", event_id=7)
    assert len(sent) == before + 1 and sent[-1]["type"] == "conditioning"
    assert condition.pending_inputs is None and condition.failed_update == (
        "too large",
        7,
    )
    assert condition.inputs.condition_media == []
    condition.confirm_applied(6)
    assert condition.inputs.condition_media == []
    run(8, prompt="valid after rejection")
    assert sent[-1]["event_id"] == 8 and "inputs" not in sent[-1]


@pytest.mark.parametrize("service_count", [1, 2, 3])
def test_encoder_services_load_once_and_keep_admission_and_conditions_private(
    monkeypatch, tmp_path, service_count
):
    from sglang.multimodal_gen.runtime.conditioning import native
    from sglang.multimodal_gen.runtime.distributed import parallel_state
    from sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server import (
        create_apps,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        codecs as codecs_module,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        encoder_worker,
    )
    from sglang.multimodal_gen.runtime.realtime.admission import AdmissionRejected

    calls = []
    args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(arch_config=None),
            prompt_switch_prefix_latent_frames=5,
        ),
        zing_conditioning_local_cache_bytes=1024,
        realtime_session_lease_ttl_s=60,
    )
    monkeypatch.setattr(
        encoder_worker,
        "initialize_runtime_imports",
        lambda: (SimpleNamespace(from_kwargs=lambda **kw: args), lambda _: None),
    )

    def load_qwen(*args):
        calls.append("qwen")
        return SimpleNamespace(modules={"qwen": object()})

    def load_vae(*args):
        calls.append("vae")
        return SimpleNamespace(modules={"video_vae": object()}, va=object(), runtime={})

    shared_codecs = SimpleNamespace(
        warmup_encoder=lambda: calls.append("warmup"),
        encode_condition=lambda *a: None,
        canvas_multiple=32,
        max_short_edge=768,
    )
    monkeypatch.setattr(native, "build_encoder_pipeline", load_qwen)
    monkeypatch.setattr(encoder_worker, "load_encoder", load_vae)
    monkeypatch.setattr(codecs_module, "ZingH3Codecs", lambda *a: shared_codecs)
    monkeypatch.setattr(
        parallel_state,
        "maybe_init_distributed_environment_and_model_parallel",
        lambda **kw: None,
    )
    monkeypatch.setattr(parallel_state, "cleanup_dist_env_and_memory", lambda: None)
    apps, _ = create_apps(tmp_path, service_count=service_count)

    async def run():
        async with apps[0].router.lifespan_context(apps[0]):
            clients = [app.state.realtime_execution_client for app in apps]
            assert all(c.executor.codecs is shared_codecs for c in clients)
            shared_codecs.compute_plan.validate_canvas(736, 1280)
            assert len({id(c.pool) for c in clients}) == 1
            assert len({id(c.executor.conditioning) for c in clients}) == service_count
            assert (
                len(
                    {
                        id(c.executor.conditioning.conditioner.identity_state)
                        for c in clients
                    }
                )
                == service_count
            )
            for index, app in enumerate(apps):
                controller = app.state.realtime_admission_controller
                lease = await controller.admit("user", str(index), "g")
                with pytest.raises(AdmissionRejected):
                    await controller.admit("another", "second", "g2")
                # Keep each owner active while the next independent service admits.
                app.state.test_lease = lease
            for app in apps:
                await app.state.realtime_admission_controller.release(
                    app.state.test_lease
                )

    asyncio.run(run())
    assert calls == ["qwen", "vae", "warmup"]


@pytest.mark.parametrize(
    "failure", ["condition", "model", "cuda", "cuda_after_forward"]
)
def test_encoder_owner_queue_classifies_condition_rejection_without_quarantine(
    monkeypatch, failure
):
    from concurrent.futures import ThreadPoolExecutor

    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server import (
        EncoderClient,
    )
    from sglang.multimodal_gen.runtime.realtime.worker_connections import (
        ServiceConnections,
    )

    controller = ServiceConnections(
        {"role": "encoder"},
        ready=lambda: True,
        build=lambda *args: None,
        destroy=lambda: None,
        rollback=lambda: None,
    )
    controller.state, controller.group = "connected", "capacity-recovery"
    rejected = (
        RuntimeError("Private model failure")
        if failure == "model"
        else ConditionError(
            "CONDITION_INVALID", "This request is too large for this server"
        )
    )
    completed = object()
    requests = []

    def forward(request):
        requests.append(request)
        if len(requests) == 1 and failure != "cuda_after_forward":
            raise rejected
        return completed

    cuda_error = RuntimeError("Private CUDA synchronization failure")

    def synchronize():
        if failure in ("cuda", "cuda_after_forward"):
            raise cuda_error

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    executor = SimpleNamespace(forward=forward, connection_controller=controller)

    async def run():
        with ThreadPoolExecutor(max_workers=1) as pool:
            client = EncoderClient(executor, pool)
            expected = (
                cuda_error if failure in ("cuda", "cuda_after_forward") else rejected
            )
            with pytest.raises(type(expected)) as captured:
                await client.forward("rejected-initial-condition")
            assert captured.value is expected
            if failure != "condition":
                assert controller.restart_required
                assert controller.query()[1] == 503
            else:
                assert not controller.restart_required
                assert controller.query()[0]["state"] == "connected"
                assert await client.forward("next-legal-session") is completed
                assert requests == ["rejected-initial-condition", "next-legal-session"]

    asyncio.run(run())


def test_early_connection_rejection_cleanup_preserves_existing_adapter(configured):  # noqa: F811
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_encoder_dit_adapter import (
        ZingEncoderDiTAdapter,
    )

    async def run():
        from sglang.multimodal_gen.runtime.realtime.worker_connections import (
            ServiceConnections,
        )

        active = object()
        controller = ServiceConnections(
            {"role": "encoder"},
            ready=lambda: True,
            build=lambda *args: None,
            destroy=lambda: None,
            rollback=lambda: None,
        )
        controller.state = "unavailable"
        app = SimpleNamespace(
            state=SimpleNamespace(
                worker_connections=controller,
                connection_adapter=active,
            )
        )
        adapter, session = ZingEncoderDiTAdapter(), GenerateSession()
        adapter.connection_app = app
        session.set_adapter(adapter)
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="first",
            size="32x32",
            zing_realtime_api_version=1,
            source_stream=dict(format="rgb24", width=32, height=32, fps=24),
        )
        with pytest.raises(ValueError, match="worker connections are not ready"):
            await adapter.on_init(session, request)
        await adapter.close_conditioning(session)
        adapter.dispose(session)
        assert app.state.connection_adapter is active

    asyncio.run(run())
