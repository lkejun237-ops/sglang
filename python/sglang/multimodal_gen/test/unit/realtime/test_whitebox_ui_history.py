"""Replay keeps UI/input changes while collapsing identical capture observations."""

import copy
from collections import deque
from types import SimpleNamespace

import msgspec
import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
    InputReceiptWire,
    RoomPipeline,
)


def owner(*, negotiated=True):
    live = []
    pipeline = SimpleNamespace(
        receipt_wire=InputReceiptWire(
            {"input_receipt_protocol": "state-delta-v1"} if negotiated else {}
        ),
        reliable_control_types=RoomPipeline.reliable_control_types,
        history=deque(),
        history_bytes=0,
        history_floor=0,
        control_sequence=0,
        host=SimpleNamespace(history_bytes=16 << 20),
        subscription=SimpleNamespace(offer_control=live.append),
        member_source=None,
    )
    return pipeline, live


def observation(index, *, revision=1, events=None, receipts=None, latest=None):
    header = dict(
        protocolVersion=1,
        sessionId="native-source",
        epoch=0,
        mediaGeneration=1,
        outputFrameId=index,
        outputPtsUs=index * 50000,
        source=dict(
            presentationId="presentation",
            sdkEpoch=0,
            sourceFrameId=index,
            sourceTimeUs=index * 50000,
            simulationTick=index,
            worldRevision=10,
        ),
        uiRevision=revision,
        uiCompleteThroughUs=index * 50000,
        widthPixels=832,
        heightPixels=480,
    )
    if events is None:
        events = [
            dict(
                type="ui.clock",
                epoch=0,
                clock=dict(
                    sourceTimeUs=index * 50000,
                    simulationTick=index,
                    uiRevision=revision,
                    completeThroughUs=index * 50000,
                ),
            )
        ]
    return dict(
        version=2,
        type="whitebox.states",
        source_frame_index=index,
        pts_us=index * 50000,
        source=dict(
            cli_frame_header=header,
            capture_pts_us=index * 50000,
            capture_origin_us=0,
            source_session_id="render-source",
            source_id="render-source",
            source_epoch="scope",
            view_id="view",
            model_context_sample=dict(
                contextStateId=f"context:{index}",
                contextRevision=index,
                viewRevision=1,
                throughControlSequence=index,
                compiledConditionId="condition:fixed",
            ),
        ),
        states=dict(ui_events=events),
        observed_input=dict(
            receipts=[] if receipts is None else receipts,
            events=[],
            latest={} if latest is None else latest,
            mapping="receipt_evidence_only",
            display=dict(mode="model", revision=1),
        ),
        source_session_id="render-source",
        source_id="render-source",
        source_epoch="scope",
        view_id="view",
    )


def publish(pipeline, message):
    RoomPipeline.publish(pipeline, message)


def test_late_replay_of_700_clock_observations_keeps_first_receipt_and_latest_capture():
    pipeline, live = owner()
    receipt = dict(type="input.receipt", epoch=0, clientSequence=0, simulationTick=1)
    messages = [observation(i, receipts=[receipt]) for i in range(700)]
    frozen = copy.deepcopy(messages)
    for message in messages:
        publish(pipeline, message)
    # The browser's existing 512-message loading bound is not raised.
    assert len(pipeline.history) == 2
    assert [row[0] for row in pipeline.history] == [1, 700]
    assert pipeline.history[0][2]["observed_input"]["receipts"] == [receipt]
    assert pipeline.history[-1][2]["source"]["cli_frame_header"]["outputFrameId"] == 699
    assert (
        len(live) == 700 and live[1]["source"]["cli_frame_header"]["outputFrameId"] == 1
    )
    assert messages == frozen
    assert pipeline.history_bytes == sum(row[1] for row in pipeline.history)
    assert pipeline.history_bytes == sum(
        len(msgspec.msgpack.encode(row[2])) for row in pipeline.history
    )


@pytest.mark.parametrize(
    "boundary",
    [
        "snapshot",
        "commit",
        "reset",
        "receipt",
        "input",
        "display",
        "epoch",
        "presentation",
        "ui_revision",
        "unknown_state",
        "model_condition",
        "clock_regression",
        "model_cursor_regression",
        "cross_packet_clock_prefix",
        "malformed_display",
        "malformed_clock",
    ],
)
def test_replay_does_not_merge_across_changes_or_unrecognized_evidence(boundary):
    pipeline, _ = owner()
    first, changed, last = observation(0), observation(1), observation(2)
    if boundary in {"snapshot", "commit", "reset"}:
        changed["states"]["ui_events"] = [
            dict(
                type={
                    "snapshot": "ui.snapshot",
                    "commit": "ui.commit",
                    "reset": "source.reset",
                }[boundary]
            )
        ]
    elif boundary == "receipt":
        changed["observed_input"]["receipts"] = [
            dict(type="input.receipt", epoch=0, clientSequence=1)
        ]
    elif boundary == "input":
        changed["observed_input"]["latest"] = {"held_keys": ["KeyW"]}
    elif boundary == "display":
        changed["observed_input"]["display"]["revision"] = 2
    elif boundary == "epoch":
        changed["source"]["cli_frame_header"]["epoch"] = 1
        changed["states"]["ui_events"][0]["epoch"] = 1
    elif boundary == "presentation":
        changed["source"]["cli_frame_header"]["source"]["presentationId"] = "other"
    elif boundary == "ui_revision":
        changed["source"]["cli_frame_header"]["uiRevision"] = 2
        changed["states"]["ui_events"][0]["clock"]["uiRevision"] = 2
    elif boundary == "unknown_state":
        changed["states"]["future_state"] = {"keep": "this"}
    elif boundary == "model_condition":
        changed["source"]["model_context_sample"]["compiledConditionId"] = (
            "condition:other"
        )
    elif boundary == "clock_regression":
        first["states"]["ui_events"][0]["clock"].update(
            sourceTimeUs=100000, completeThroughUs=100000
        )
    elif boundary == "cross_packet_clock_prefix":
        first["states"]["ui_events"][0]["clock"].update(
            sourceTimeUs=100000, completeThroughUs=100000
        )
        changed["states"]["ui_events"].append(
            copy.deepcopy(changed["states"]["ui_events"][0])
        )
        changed["states"]["ui_events"][-1]["clock"].update(
            sourceTimeUs=150000, completeThroughUs=150000
        )
        last["states"]["ui_events"][0]["clock"].update(
            sourceTimeUs=200000, completeThroughUs=200000
        )
    elif boundary == "model_cursor_regression":
        first["source"]["model_context_sample"]["throughControlSequence"] = 100
    elif boundary == "malformed_display":
        changed["observed_input"]["display"]["mode"] = []
    elif boundary == "malformed_clock":
        changed["states"]["ui_events"][0]["clock"]["completeThroughUs"] = -1
    for message in (first, changed, last):
        publish(pipeline, message)
    assert len(pipeline.history) == 3
    assert pipeline.history[1][2]["states"] == changed["states"]


def test_legacy_history_keeps_every_observation_without_negotiation():
    pipeline, _ = owner(negotiated=False)
    for i in range(700):
        publish(pipeline, observation(i))
    assert len(pipeline.history) == 700


def test_resume_after_a_coalesced_cursor_returns_latest_observation_without_losing_receipt():
    import asyncio

    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        Subscription,
    )

    async def scenario():
        pipeline, _ = owner()
        pipeline.request_keyframe = lambda _: None
        pipeline.terminal = None
        receipt = dict(type="input.receipt", epoch=0, clientSequence=0)
        for index in range(700):
            publish(pipeline, observation(index, receipts=[receipt]))
        subscription = Subscription(pipeline, 2)
        subscription.replay_controls(row for row in pipeline.history if row[0] > 350)
        replay = msgspec.msgpack.decode(await subscription.recv())
        assert replay["control_sequence"] == 700
        assert replay["source_frame_index"] == 699
        assert replay["observed_input"]["receipts"] == []
        assert pipeline.history[0][2]["observed_input"]["receipts"] == [receipt]
        assert pipeline.history_floor == 0

    asyncio.run(scenario())


def test_business_control_is_a_boundary_between_observation_runs():
    pipeline, live = owner()
    for index in range(20):
        publish(pipeline, observation(index))
    control = dict(type="whitebox.control", command="keep-this-exactly")
    publish(pipeline, control)
    for index in range(20, 40):
        publish(pipeline, observation(index))
    assert [row[2]["type"] for row in pipeline.history] == [
        "whitebox.states",
        "whitebox.control",
        "whitebox.states",
    ]
    assert pipeline.history[1][2]["command"] == "keep-this-exactly"
    assert len(live) == 41


def test_business_byte_eviction_still_rejects_resume_below_the_real_history_floor():
    pipeline, _ = owner()
    pipeline.host.history_bytes = 65536
    for index in range(100):
        publish(
            pipeline,
            observation(index, events=[dict(type="ui.commit", opaque="x" * 1000)]),
        )
    assert pipeline.history_floor > 0
    assert pipeline.history_bytes <= 65536
    pipeline.closed = False
    pipeline.terminal = None
    pipeline.generation = 1
    pipeline.token = "original-capability"
    with pytest.raises(ValueError, match="STATE_HISTORY_EXPIRED"):
        RoomPipeline.attach(
            pipeline,
            dict(
                resume_token=pipeline.token, control_sequence=pipeline.history_floor - 1
            ),
        )
