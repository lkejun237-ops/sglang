# SPDX-License-Identifier: Apache-2.0
"""Regression for mixed asset uploads and named UI inputs on a Room binding."""

import asyncio
from types import SimpleNamespace

import msgspec
import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import (
    WhiteboxDiagnostic,
    public_error,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import RenderPipeline
from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
    InputReceiptWire,
    RoomPipeline,
)


def pipeline(cls=RenderPipeline):
    owner = object.__new__(cls)
    owner.generation = 1
    owner.subscription = SimpleNamespace(generation=1, closed=False)
    owner.command_queue = asyncio.Queue(maxsize=64)
    owner.command_bytes = 0
    return owner


def input_wire(sequence=0):
    return msgspec.msgpack.encode(
        dict(
            type="event",
            kind="input",
            sequence=sequence,
            payload=dict(
                held_keys=[],
                held_buttons=[],
                events=[],
                actions=[
                    dict(
                        id="action-one",
                        name="renderGateConfigure",
                        params={"epoch": "game-ui-epoch"},
                    )
                ],
                ui_source_frame={"source": {"simulationTick": 71}},
            ),
        )
    )


def test_legal_large_asset_does_not_reject_a_following_ui_input():
    owner = pipeline()
    asset = msgspec.msgpack.encode(dict(type="asset.upload", payload=b"x" * 729_992))
    control = input_wire()
    # No event-loop yield: upload and UI events can arrive in the same burst.
    owner.accept(owner.subscription, asset)
    owner.accept(owner.subscription, control)
    assert owner.command_bytes == len(asset) + len(control)
    assert owner.command_queue.get_nowait()[1] == asset
    assert owner.command_queue.get_nowait()[1] == control


@pytest.mark.parametrize("cls", [RoomPipeline, RenderPipeline])
def test_shared_control_queue_still_bounds_message_count(cls):
    owner = pipeline(cls)
    for sequence in range(64):
        owner.accept(owner.subscription, input_wire(sequence))
    before = owner.command_bytes
    with pytest.raises(WhiteboxDiagnostic) as failure:
        owner.accept(owner.subscription, input_wire(64))
    assert failure.value.code == "CONTROL_QUEUE_FULL"
    assert failure.value.stage == "control"
    assert owner.command_queue.qsize() == 64 and owner.command_bytes == before
    error = public_error(failure.value, "test-trace", stage="transport")
    assert error["code"] == "CONTROL_QUEUE_FULL" and error["stage"] == "control"


@pytest.mark.parametrize("cls", [RoomPipeline, RenderPipeline])
def test_shared_control_queue_still_bounds_bytes(cls):
    owner = pipeline(cls)
    control = input_wire()
    owner.command_bytes = owner.command_byte_limit - len(control) + 1
    before = owner.command_bytes
    with pytest.raises(WhiteboxDiagnostic) as failure:
        owner.accept(owner.subscription, control)
    assert failure.value.code == "CONTROL_QUEUE_FULL"
    assert owner.command_queue.empty() and owner.command_bytes == before


def test_action_receipts_are_compacted_only_after_reliable_delivery():
    wire = InputReceiptWire({"input_receipt_protocol": "state-delta-v1"})
    receipts = [
        dict(type="action.receipt", epoch=0, actionId=name, status="completed")
        for name in ("gate", "select", "configure")
    ]
    state = dict(type="whitebox.states", observed_input={"receipts": receipts})
    media = dict(state, type="whitebox.frame")
    assert wire.media(media)["observed_input"]["receipts"] == receipts
    assert wire.control(state)["observed_input"]["receipts"] == receipts
    assert wire.control(state)["observed_input"]["receipts"] == []
    assert wire.media(media)["observed_input"]["receipts"] == []
    changed = dict(receipts[0], status="failed", error="Unavailable")
    update = dict(state, observed_input={"receipts": [changed, *receipts[1:]]})
    assert wire.control(update)["observed_input"]["receipts"] == [changed]
    assert state["observed_input"]["receipts"] == receipts
    assert InputReceiptWire({}).control(state) is state
    for index in range(300):
        receipt = dict(receipts[0], actionId=str(index))
        wire.control(dict(state, observed_input={"receipts": [receipt]}))
    assert len(wire.history) == 256
