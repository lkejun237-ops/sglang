# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from collections import deque

import pytest
from decoder_output_client import (
    OutputReceiverThread,
    ThreadCredit,
    consume_source_credit,
    retire_source_boundaries,
    validate_split_completion,
)


def test_receiver_runs_while_sender_thread_is_blocked_without_ack():
    output_done = threading.Event()
    thread_ids = []
    saved = []

    async def receive():
        thread_ids.append(threading.get_ident())
        for frame in range(3):
            saved.append(frame)
            await asyncio.sleep(0.001)
        output_done.set()

    async def run():
        receiver = OutputReceiverThread(receive)
        try:
            # Blocking the sender loop must not stall the output loop.
            assert output_done.wait(1)
            assert saved == [0, 1, 2]
            assert thread_ids == [receiver.thread.ident]
            assert thread_ids[0] != threading.get_ident()
            await receiver.wait()
        finally:
            await receiver.close()
        assert not receiver.thread.is_alive()

    asyncio.run(run())


def test_output_disconnect_propagates_and_does_not_leak_thread():
    async def receive():
        raise RuntimeError("output disconnected")

    async def run():
        receiver = OutputReceiverThread(receive)
        try:
            with pytest.raises(RuntimeError, match="disconnected"):
                await receiver.wait()
        finally:
            await receiver.close()
        assert not receiver.thread.is_alive()

    asyncio.run(run())


def test_sender_failure_cancels_output_and_waits_for_cleanup():
    cleaned = threading.Event()
    entered = threading.Event()

    async def receive():
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    async def run():
        receiver = OutputReceiverThread(receive)
        assert entered.wait(1)
        await receiver.close()
        assert cleaned.is_set() and not receiver.thread.is_alive()

    asyncio.run(run())


@pytest.mark.parametrize(
    "missing", ["input_completed", "completed", "accepted", "consumed", "received"]
)
def test_end_validation_detects_missing_input_or_output(missing):
    state = dict(
        input_completed=True,
        completed=True,
        submitted=120,
        accepted=120,
        consumed=120,
        received=120,
    )
    validate_split_completion(state)
    state[missing] = False if missing.endswith("completed") else 119
    with pytest.raises(ValueError, match="missing"):
        validate_split_completion(state)


def test_decoder_consumed_wakes_sender_without_waiting_for_encoder_accepted():
    async def run():
        state = dict(submitted=8, accepted=0, consumed=0, consumed_bytes=0)
        boundaries = deque([(8, 96, 1)])
        credit = ThreadCredit()

        async def output():
            with credit.lock:
                consume_source_credit(
                    state,
                    boundaries,
                    dict(source_frame_end=8, source_bytes_end=96),
                    from_decoder=True,
                )
            credit.notify()

        receiver = OutputReceiverThread(output)
        try:
            await asyncio.wait_for(credit.wait_for(lambda: state["consumed"] == 8), 1)
            assert state["accepted"] == 0  # No Encoder message was needed.
            assert list(boundaries) == [(8, 96, 1)]
            # The late Encoder acceptance still has its original identity.
            with credit.lock:
                state["accepted"] = 8
                retire_source_boundaries(state, boundaries)
            assert not boundaries
            await receiver.wait()
        finally:
            await receiver.close()

    asyncio.run(run())


@pytest.mark.parametrize("end,nbytes", [(9, 108), (7, 84), (8, 95), (True, 12)])
def test_decoder_credit_must_match_a_submitted_complete_packet(end, nbytes):
    state = dict(submitted=8, accepted=0, consumed=0, consumed_bytes=0)
    boundaries = deque([(8, 96, 1)])
    with pytest.raises(ValueError, match="watermark"):
        consume_source_credit(
            state,
            boundaries,
            dict(source_frame_end=end, source_bytes_end=nbytes),
            from_decoder=True,
        )
    assert state["consumed"] == 0


def test_legacy_credit_still_requires_encoder_acceptance():
    state = dict(submitted=8, accepted=0, consumed=0, consumed_bytes=0)
    boundaries = deque([(8, 96, 1)])
    with pytest.raises(ValueError):
        consume_source_credit(
            state,
            boundaries,
            dict(source_frame_end=8, source_bytes_end=96),
            from_decoder=False,
        )
