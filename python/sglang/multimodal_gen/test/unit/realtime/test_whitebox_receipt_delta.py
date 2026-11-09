# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
from collections import deque
from types import SimpleNamespace

import msgspec

from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
    InputReceiptWire,
    RoomPipeline,
    Subscription,
)


def receipt(sequence, epoch=0):
    return dict(
        type="input.receipt",
        clientSequence=sequence,
        epoch=epoch,
        simulationTick=sequence + 10,
        opaque={"preserved": True},
    )


def state(receipts):
    return dict(
        type="whitebox.states",
        source_frame_index=17,
        source={"identity": "unchanged"},
        states={"ui_events": ["opaque"]},
        observed_input={"receipts": receipts, "latest": {"key": "W"}},
    )


def test_repeated_history_is_removed_only_after_reliable_control_keeps_it():
    wire = InputReceiptWire({"input_receipt_protocol": "state-delta-v1"})
    original = state([receipt(i) for i in range(256)])
    frozen = copy.deepcopy(original)
    media = dict(
        original,
        type="whitebox.frame",
        stream="whitebox",
        keyframe=True,
        output_frame_start=3,
        payload=b"H264",
    )
    assert (
        wire.media(media)["observed_input"]["receipts"]
        == original["observed_input"]["receipts"]
    )
    assert not wire.history  # Droppable media cannot consume reliable evidence.
    first = wire.control(original)
    assert first["observed_input"]["receipts"] == original["observed_input"]["receipts"]
    for _ in range(120):
        compact = wire.control(original)
        assert compact["observed_input"]["receipts"] == []
        for name in ("whitebox", "model"):
            reduced = wire.media(dict(media, stream=name))
            assert reduced["observed_input"]["receipts"] == []
            assert (
                reduced["payload"] == b"H264"
                and reduced["source"] == original["source"]
            )
        assert (
            len(msgspec.msgpack.encode(compact))
            < len(msgspec.msgpack.encode(original)) / 10
        )
    assert original == frozen  # The archive producer's input was never rewritten.
    assert (
        wire.control(dict(type="session.ready"))["input_receipt_protocol"]
        == wire.protocol
    )


def test_legacy_clients_and_opaque_receipts_are_preserved():
    original = state([receipt(0), {"type": "future.receipt", "value": 3}])
    legacy = InputReceiptWire({})
    assert legacy.control(original) is original and legacy.media(original) is original
    wire = InputReceiptWire({"input_receipt_protocol": "state-delta-v1"})
    wire.control(original)
    assert wire.control(original)["observed_input"]["receipts"] == [
        original["observed_input"]["receipts"][1]
    ]
    changed = receipt(0)
    changed["simulationTick"] += 1
    assert wire.control(state([changed]))["observed_input"]["receipts"] == [changed]
    assert wire.control(state([receipt(0, epoch=1)]))["observed_input"]["receipts"] == [
        receipt(0, epoch=1)
    ]
    for i in range(300):
        wire.control(state([receipt(i, epoch=1)]))
    assert len(wire.history) == 256


def test_room_replay_retains_exact_deltas_and_media_cannot_overtake_them():
    async def run():
        owner = SimpleNamespace(
            receipt_wire=InputReceiptWire({"input_receipt_protocol": "state-delta-v1"}),
            reliable_control_types=RoomPipeline.reliable_control_types,
            history=deque(),
            history_bytes=0,
            history_floor=0,
            control_sequence=0,
            host=SimpleNamespace(history_bytes=65536),
            subscription=None,
            request_keyframe=lambda _: None,
            terminal=None,
        )
        original = state([receipt(0)])
        RoomPipeline.publish(owner, original)
        RoomPipeline.publish(owner, state([receipt(0), receipt(1)]))
        RoomPipeline.publish(owner, state([receipt(0), receipt(1)]))
        assert [
            [r["clientSequence"] for r in m["observed_input"]["receipts"]]
            for _, _, m in owner.history
        ] == [[0], [1], []]
        resumed = Subscription(owner, 2)
        for sequence, _, message in owner.history:
            if sequence > 1:  # Client already applied the first control.
                resumed.offer_control(message)
        resumed.ack(
            dict(protocol="whitebox-dual-v2", stream="model", frame=-1, window=32)
        )
        resumed.offer_media(
            owner.receipt_wire.media(
                dict(
                    state([receipt(0), receipt(1)]),
                    stream="model",
                    keyframe=True,
                    type="frame_batch",
                    output_frame_start=0,
                    payload=b"H264",
                )
            )
        )
        received = [msgspec.msgpack.decode(await resumed.recv()) for _ in range(3)]
        assert [x.get("control_sequence") for x in received] == [2, 3, None]
        assert received[0]["observed_input"]["receipts"] == [receipt(1)]
        assert received[2]["observed_input"]["receipts"] == []
        assert original["observed_input"]["receipts"] == [receipt(0)]

    asyncio.run(run())
