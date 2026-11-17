# SPDX-License-Identifier: Apache-2.0
"""Accepted API inputs -> verified original assets -> real streaming uploader.

CPU-only: model ingress and archive code are real; no GPU or live OSS is used.
"""

import asyncio
import hashlib
import json
from types import SimpleNamespace as NS

import msgspec
import pytest
from test_whitebox_stream_archive import MemoryS3, prototype
from test_zing_h3_source_adapter import configured  # noqa: F401
from test_zing_h3_source_adapter import (
    PromptSocket,
    _enable_reference_updates,
    _reference,
    initialize,
)
from test_zing_prefill_api import Socket, payload, service  # noqa: F401

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    realtime_video_api as api,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    zing_prefill,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
    zing_h3_source_adapter as source,
)
from sglang.multimodal_gen.runtime.realtime.model_input_recording import (
    accepted_input,
    input_snapshot,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.model_inputs import (
    PATH,
    ModelInputCapture,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.stream_recording import (
    StreamingData,
)


class Recording:
    def __init__(self):
        self.manifest, self.operations = {}, []

    def offer(self, kind, value):
        self.operations.append((kind, value))
        return True


def initial(**overrides):
    return dict(
        type="init",
        prompt="最终模型 Prompt\nPlayer instruction:\n下雨",
        references=[_reference("<Picture 2>", "blue"), _reference("<Picture 1>")],
        source_stream=dict(format="rgb24", width=32, height=32, fps=24),
        size="32x32",
        seed=19,
        **overrides,
    )


def test_model_validated_init_and_first_frame_archive_exact_originals(
    configured, tmp_path
):
    request = initial()
    adapter, session = initialize(
        **{k: v for k, v in request.items() if k != "type"}, record_model_inputs=True
    )
    recording = Recording()
    capture = ModelInputCapture(recording, session.id, session.generation_id)
    capture.sent(request)
    ready = adapter.session_ready_message(session)
    assert "payload" not in json.dumps(ready["model_input"])
    capture.received(ready)
    frame = bytes(range(256)) * 12
    message = dict(
        type="event",
        kind="source.append",
        event_id=1,
        payload=dict(
            client_seq=0,
            source_frame_start=0,
            num_frames=1,
            pts_us=0,
            data=frame,
        ),
    )
    capture.sent(message)
    ws = PromptSocket()
    asyncio.run(
        adapter.ingest_wire_event(
            ws, session, message, len(msgspec.msgpack.encode(message))
        )
    )
    capture.received(ws.messages[0])
    capture.close()
    assert recording.manifest["model_inputs"] == dict(
        protocol="model-input-v1",
        path=PATH,
        status="confirmed",
        accepted=2,
        applied=0,
        unconfirmed=0,
        unresolved_applied=0,
        failed=0,
    )
    kwargs, manifest = asyncio.run(prototype(tmp_path, source=True))
    manifest.update(recording.manifest)
    client = MemoryS3()
    data = StreamingData(
        client,
        {"bucket": "b"},
        kwargs,
        manifest,
        dict(
            upload_bytes=32 << 20,
            archive_bytes=2 << 30,
            part_bytes=65536,
            finish_seconds=10,
            index_key="test/index.json",
        ),
    )
    try:
        for kind, value in recording.operations:
            data.operation(kind, value)
        final = data.finish("failed", manifest)
        prefix = manifest["object_prefix"]
        rows = [
            json.loads(line)
            for line in client.objects[("b", prefix + PATH)][0].splitlines()
        ]
        assert rows[0]["prompt"] == request["prompt"]
        assert [r["id"] for r in rows[0]["references"]] == [
            r["id"] for r in request["references"]
        ]
        for actual, original in zip(rows[0]["references"], request["references"]):
            assert (
                client.objects[("b", prefix + actual["path"])][0] == original["payload"]
            )
            assert final["objects"][actual["path"]]["sha256"] == actual["sha256"]
        assert (
            client.objects[("b", prefix + rows[1]["source_frame"]["path"])][0] == frame
        )
        assert rows[1]["model_session_id"] == rows[0]["model_session_id"]
        assert final["upload_status"] == "complete"
        assert final["recording_status"] == "partial"  # no video fabricated
    finally:
        data.abort()


@pytest.mark.parametrize("references", [None, [], [_reference("<New>")]])
def test_actual_prompt_update_receipt_preserves_inherit_clear_replace(
    configured, references
):
    _enable_reference_updates(configured)
    configured.pipeline_config.decoder_type = "lynn_streaming"
    adapter, session = initialize(
        record_model_inputs=True,
        source_prompt_protocol="ordered-boundary-v2",
        source_condition_switch_protocol="clip-barrier-v1",
    )
    value = dict(
        text="CLI最终文本",
        version=2,
        min_source_frame=0,
        policy="at_or_after",
        source_policy="skip_during_switch",
    )
    if references is not None:
        value["references"] = references
    message = dict(type="event", kind="prompt.update", event_id=1, payload=value)
    recording = Recording()
    capture = ModelInputCapture(recording, session.id, session.generation_id)
    capture.sent(message)
    ws = PromptSocket()
    asyncio.run(adapter.ingest_wire_event(ws, session, message, 2048))
    receipt = ws.messages[-1]
    assert receipt["state"] == "accepted"
    capture.received(receipt)
    capture.close()
    event, _ = recording.operations[0][1]
    assert event["prompt"] == value["text"] and event["prompt_version"] == 2
    assert event["references_mode"] == ("inherit" if references is None else "replace")
    assert len(event["references"]) == len(references or [])
    # The receipt says accepted, not applied; no output has been generated.
    assert event["capture_stage"] == "model_api_accepted"
    assert adapter.public_prompt_version == 1


@pytest.mark.parametrize("corruption", ["prompt", "reference", "identity"])
def test_mismatched_model_receipt_never_publishes_candidate(corruption):
    request = initial()
    recording = Recording()
    capture = ModelInputCapture(recording, "s", "g")
    capture.sent(request)
    receipt = accepted_input(
        NS(id="s", generation_id="g"), input_snapshot("init", request)
    )
    if corruption == "prompt":
        receipt["prompt"] = "different"
    elif corruption == "reference":
        receipt["references"].reverse()
    else:
        receipt["generation_id"] = "stale"
    capture.received({"model_input": receipt})
    capture.close()
    assert not recording.operations
    assert recording.manifest["model_inputs"]["status"] == "incomplete"


def test_old_peer_missing_receipts_and_capacity_are_explicit_and_bounded():
    recording = Recording()
    capture = ModelInputCapture(recording, "s", "g")
    for event_id in range(8):
        capture.sent(
            dict(
                type="event",
                kind="prompt.update",
                event_id=event_id,
                payload=dict(text="x", version=event_id + 2),
            )
        )
    assert len(capture.pending) == 4
    capture.received(dict(type="session.ready"))
    capture.close()
    capture.close()
    assert not capture.pending and capture.pending_bytes == 0
    assert recording.manifest["model_inputs"]["unconfirmed"] == 4
    assert recording.manifest["model_inputs"]["status"] == "incomplete"


def test_unavailable_archive_does_not_raise_into_model_control():
    recording = Recording()
    recording.offer = lambda *args: False
    request = initial()
    capture = ModelInputCapture(recording, "s", "g")
    capture.sent(request)
    capture.received(
        {
            "model_input": accepted_input(
                NS(id="s", generation_id="g"), input_snapshot("init", request)
            )
        }
    )
    capture.close()
    assert recording.manifest["model_inputs"]["failed"] == 1
    assert recording.manifest["model_inputs"]["accepted"] == 0
    assert not capture.pending


def test_source_encoder_role_returns_same_accepted_input_contract(configured):
    from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
        RealtimeVideoGenerationsRequest,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_source_encoder_adapter import (
        ZingSourceEncoderAdapter,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
    )

    request = RealtimeVideoGenerationsRequest(
        **initial(), record_model_inputs=True, zing_realtime_api_version=1
    )
    adapter, session = ZingSourceEncoderAdapter(), GenerateSession()
    session.set_adapter(adapter)
    asyncio.run(adapter.on_init(session, request))
    session.set_request(request)
    ready = adapter.session_ready_message(session)
    assert ready["output_kind"] == "source_moments"
    assert ready["model_input"]["prompt"] == request.prompt
    assert [r["id"] for r in ready["model_input"]["references"]] == [
        r["id"] for r in request.references
    ]


def test_invalid_prompt_has_no_accepted_snapshot(configured):
    _enable_reference_updates(configured)
    configured.pipeline_config.decoder_type = "lynn_streaming"
    adapter, session = initialize(
        record_model_inputs=True,
        source_prompt_protocol="ordered-boundary-v2",
        source_condition_switch_protocol="clip-barrier-v1",
    )
    ws = PromptSocket()
    asyncio.run(
        adapter.ingest_wire_event(
            ws,
            session,
            dict(
                type="event",
                kind="prompt.update",
                event_id=1,
                payload=dict(
                    text="", version=2, min_source_frame=0, policy="at_or_after"
                ),
            ),
            512,
        )
    )
    assert ws.messages[-1]["state"] == "rejected"
    assert "model_input" not in ws.messages[-1]


def test_prefill_records_accepted_inputs_before_compute_failure(service, monkeypatch):
    monkeypatch.setattr(
        source.BaseRealtimeModelAdapter, "prepare_next_request", lambda *a: NS(extra={})
    )

    async def fail(batch, **kwargs):
        assert batch.extra["zing_prefill"]["source_frame"] == request["source_frame"]
        raise RuntimeError("test compute failure")

    async def released(*a, **kw):
        return True

    monkeypatch.setattr(api, "_process_realtime_generation_batch", fail)
    monkeypatch.setattr(api, "_release_scheduler_realtime_session", released)
    request = dict(payload(), record_model_inputs=True)

    async def run():
        ws = Socket()
        with pytest.raises(RuntimeError, match="test compute failure"):
            await zing_prefill.run_prefill_case(
                ws, request, len(msgspec.msgpack.encode(request)), "connection"
            )
        message = await ws.receive()
        assert message["type"] == "model.input"
        receipt = message["model_input"]
        assert receipt["request_id"] == request["request_id"]
        assert receipt["prompt"] == request["prompt"]
        assert (
            receipt["source_frame"]["sha256"]
            == hashlib.sha256(request["source_frame"]).hexdigest()
        )
        recording = Recording()
        capture = ModelInputCapture(
            recording, receipt["model_session_id"], receipt["generation_id"]
        )
        capture.sent(request)
        capture.received(message)
        capture.close()
        assert len(recording.operations) == 1

    asyncio.run(run())


def confirm_input(capture, message):
    capture.sent(message)
    kind = message["type"]
    value = message
    if kind == "event":
        kind = "prompt.update"
        value = dict(message["payload"], prompt=message["payload"]["text"])
    receipt = accepted_input(
        NS(
            id=capture.identity["model_session_id"],
            generation_id=capture.identity["generation_id"],
        ),
        input_snapshot(
            kind,
            value,
            event_id=message.get("event_id"),
            request_id=message.get("request_id"),
        ),
    )
    capture.received(dict(model_input=receipt))


def test_effective_prompt_snapshots_resolve_prefill_switch_reject_and_clear(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import EngineFrame
    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import (
        reference_descriptors,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )

    recording = Recording()
    init = initial()
    old_images = init["references"]
    changed_images = [
        _reference("<Changed 2>", "green"),
        _reference("<Changed 1>", "yellow"),
    ]
    unused_images = [_reference("<Unused>", "black")]
    # Same archive manager, distinct model contexts: a later Prefill must never
    # change the generation's current references, even when its files exist.
    for request_id, images in [
        ("original-prefill", old_images),
        ("unused-prefill", unused_images),
    ]:
        prefill_capture = ModelInputCapture(recording, request_id, request_id)
        confirm_input(
            prefill_capture,
            dict(
                init,
                type="prefill",
                request_id=request_id,
                references=images,
                source_frame=bytes(32 * 32 * 3),
            ),
        )
        prefill_capture.close()
    capture = ModelInputCapture(recording, "formal", "generation")
    confirm_input(capture, init)
    segment = object.__new__(RenderSegment)
    segment.first_output_at = None
    segment.room_timings = {}
    segment.active_reference_descriptors = reference_descriptors(old_images)
    segment.periodic_refresh = None
    segment.segment_id, segment.segment_epoch = "segment", "epoch"

    async def no_op(*args):
        return None

    segment.sink = NS(output=no_op)
    segment.owner = NS(segment_prompt_output=no_op)
    segment.prompt_by_event = {}
    expected = []

    async def output(event_id, version, prompt, images=None):
        condition = dict(
            version=version,
            text=prompt,
            sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        )
        if images is not None:
            condition["reference_descriptors"] = reference_descriptors(images)
        segment.prompt_by_event[event_id] = condition
        message = dict(event_id=event_id, output_frame_start=version * 17, num_frames=1)
        frame = EngineFrame(version * 17, version * 1000, b"", {})
        text, rendering = await segment.model_output_condition(
            message, {version * 17: (frame, 0)}, {}, {}
        )
        refs = segment.model_output_references(
            message, reference_descriptors(old_images)
        )
        capture.applied(message, text, refs, rendering["prompt_version"])
        capture.applied(
            dict(message, output_frame_start=version * 17 + 1),
            text,
            refs,
            rendering["prompt_version"],
        )
        expected.append((prompt, [r["sha256"] for r in refs]))

    async def run():
        await output(None, 1, init["prompt"], old_images)
        for version, images, used in [
            (2, None, True),
            (3, changed_images, True),
            (4, None, True),
            (5, unused_images, False),
            (6, None, True),
            (7, [], True),
            (8, None, True),
        ]:
            prompt = f"final prompt {version}"
            value = dict(text=prompt, version=version)
            if images is not None:
                value["references"] = images
            confirm_input(
                capture,
                dict(
                    type="event", kind="prompt.update", event_id=version, payload=value
                ),
            )
            if used:
                await output(version, version, prompt, images)
            else:
                capture.received(
                    dict(type="control_ack", state="rejected", event_id=version)
                )

    asyncio.run(run())
    capture.close()
    applied = [
        value[0]
        for kind, value in recording.operations
        if value[0]["capture_stage"] == "model_output_observed"
    ]
    assert len(applied) == 7
    assert [len(row["references"]) for row in applied] == [2, 2, 2, 2, 2, 0, 0]
    assert [r["id"] for r in applied[3]["references"]] == [
        r["id"] for r in changed_images
    ]
    assert all(
        row["prompt_version"] != 5 for row in applied
    )  # accepted candidate never applied
    assert all(row["references_mode"] == "resolved" for row in applied)
    assert recording.manifest["model_inputs"]["applied"] == 7
    kwargs, manifest = asyncio.run(prototype(tmp_path, source=True))
    manifest.update(recording.manifest)
    client = MemoryS3()
    data = StreamingData(
        client,
        {"bucket": "b"},
        kwargs,
        manifest,
        dict(
            upload_bytes=32 << 20,
            archive_bytes=2 << 30,
            part_bytes=65536,
            finish_seconds=10,
            index_key="index.json",
        ),
    )
    try:
        for kind, value in recording.operations:
            data.operation(kind, value)
        final = data.finish("failed", manifest)
        prefix = manifest["object_prefix"]
        rows = [
            json.loads(row)
            for row in client.objects[("b", prefix + PATH)][0].splitlines()
        ]
        resolved = [
            row for row in rows if row["capture_stage"] == "model_output_observed"
        ]
        assert [
            (row["prompt"], [d["sha256"] for d in row["references"]])
            for row in resolved
        ] == expected
        for row in resolved:
            for d in row["references"]:
                assert (
                    hashlib.sha256(
                        client.objects[("b", prefix + d["path"])][0]
                    ).hexdigest()
                    == d["sha256"]
                )
        # Two initial, two changed, one unused image and one shared RGB object.
        assert (
            len(
                [
                    key
                    for key in final["objects"]
                    if key.startswith("model-inputs/assets/")
                ]
            )
            == 6
        )
        assert final["upload_status"] == "complete"
    finally:
        data.abort()


def test_output_before_acceptance_waits_for_exact_model_receipt():
    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import (
        reference_descriptors,
    )

    recording = Recording()
    capture = ModelInputCapture(recording, "s", "g")
    init = initial()
    capture.sent(init)
    capture.applied(
        dict(event_id=None, output_frame_start=0, num_frames=1),
        init["prompt"],
        reference_descriptors(init["references"]),
        1,
    )
    assert not recording.operations
    capture.received(
        dict(
            model_input=accepted_input(
                NS(id="s", generation_id="g"), input_snapshot("init", init)
            )
        )
    )
    capture.close()
    assert [v[0]["capture_stage"] for _, v in recording.operations] == [
        "model_api_accepted",
        "model_output_observed",
    ]
    assert recording.manifest["model_inputs"]["unresolved_applied"] == 0


def test_unconfirmed_output_or_unknown_reference_never_becomes_full_snapshot():
    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import (
        reference_descriptors,
    )

    init = initial()
    for confirmed in (False, True):
        recording = Recording()
        capture = ModelInputCapture(recording, "s", "g")
        if confirmed:
            confirm_input(capture, init)
        capture.applied(
            dict(event_id=None, output_frame_start=0, num_frames=1),
            init["prompt"],
            reference_descriptors([_reference("unknown")]),
            1,
        )
        capture.close()
        assert not any(
            v[0]["capture_stage"] == "model_output_observed"
            for _, v in recording.operations
        )
        assert recording.manifest["model_inputs"]["status"] == "incomplete"


def test_applied_archive_requires_verified_existing_assets():
    from sglang.multimodal_gen.runtime.realtime.whitebox.model_inputs import (
        archive_model_input,
    )

    event = input_snapshot("init", initial())
    event.update(capture_stage="model_output_observed", references_mode="resolved")
    for d in event["references"]:
        d["path"] = ModelInputCapture.asset_path(d)
    writes = []
    with pytest.raises(ValueError, match="unavailable"):
        archive_model_input(
            event, {}, lambda *args: writes.append(args), lambda e: writes.append(e), {}
        )
    assert not writes


def test_later_capture_preserves_previous_missing_identity_failure():
    recording = Recording()
    recording.manifest["model_inputs"] = dict(
        status="incomplete", failed=1, last_error="MODEL_INPUT_IDENTITY_UNAVAILABLE"
    )
    capture = ModelInputCapture(recording, "s", "g")
    confirm_input(capture, initial())
    capture.close()
    assert recording.manifest["model_inputs"]["status"] == "incomplete"
    assert recording.manifest["model_inputs"]["accepted"] == 1
    assert recording.manifest["model_inputs"]["failed"] == 1
