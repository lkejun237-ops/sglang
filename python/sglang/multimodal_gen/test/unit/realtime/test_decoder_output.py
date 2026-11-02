# SPDX-License-Identifier: Apache-2.0

import queue
from concurrent.futures import ThreadPoolExecutor

import pytest

from sglang.multimodal_gen.runtime.realtime.decoder_output import DecoderOutputBroker


def setup_session():
    cancelled = []
    broker = DecoderOutputBroker(cancelled.append)
    session = broker.open(dict(session_id="s", generation_id="g", width=2, height=2))
    attachment = dict(
        type="output.attach", session_id="s", generation_id="g", token=session.token
    )
    return broker, session, attachment, cancelled


@pytest.mark.parametrize(
    "change",
    [
        {"session_id": "other"},
        {"generation_id": "old"},
        {"token": "wrong"},
        {"type": "init"},
    ],
)
def test_bad_attachment_preserves_owner(change):
    broker, session, attachment, _ = setup_session()
    with pytest.raises(ValueError):
        broker.attach(dict(attachment, **change))
    assert broker.attach(attachment) is session
    assert not session.stopped.is_set()


def test_single_active_output_and_new_generation_after_close():
    broker, session, attachment, _ = setup_session()
    broker.attach(attachment)
    with pytest.raises(ValueError):
        broker.attach(attachment)
    with pytest.raises(ValueError):
        broker.open(dict(session_id="second", generation_id="g2"))
    broker.close(session)
    replacement = broker.open(dict(session_id="s", generation_id="g2"))
    with pytest.raises(ValueError):
        broker.attach(attachment)
    broker.close(session)  # Late old cleanup must not release the new owner.
    assert broker.session is replacement


def test_ten_outputs_then_backpressure_and_eos_fifo():
    _, session, _, _ = setup_session()
    for index in range(10):
        session.put(dict(type="frames", reference=index), 0)
    with pytest.raises(TimeoutError):
        session.put(dict(type="session.completed"), 0)
    assert session.take()["reference"] == 0
    session.sent()
    session.put(dict(type="session.completed"), 0)
    for index in range(1, 10):
        assert session.take()["reference"] == index
        session.sent()
    assert session.take()["type"] == "session.completed"
    with pytest.raises(queue.Empty):
        session.take()


def test_disconnect_reclaims_inflight_and_queued_and_unblocks_producer():
    broker, session, _, cancelled = setup_session()
    session.put(dict(reference="inflight"), 0)
    session.take()
    for index in range(10):
        session.put(dict(reference=index), 0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(session.put, dict(reference="not-owned"), 30)
        broker.close(session)
        with pytest.raises(RuntimeError, match="closed"):
            pending.result(timeout=1)
    assert cancelled == ["inflight", *range(10)]
    assert session.done.is_set() and not session.completed
    session.abort()
    assert len(cancelled) == 11


def test_finish_success_does_not_cancel_previously_consumed_reference():
    broker, session, _, cancelled = setup_session()
    session.put(dict(reference="sent"), 0)
    session.take()
    session.sent()
    session.completed = True
    session.done.set()
    broker.close(session)
    assert session.completed and cancelled == []


def test_consumed_follows_current_rgb_and_uses_complete_packets_not_output_frames():
    _, session, _, _ = setup_session()
    records = [[1, 0, 8], [2, 8, 16]]
    session.put(dict(type="decoder.frames", reference="rgb", num_frames=1), 0)
    session.source_progress(12, records, 0)
    assert session.take()["type"] == "decoder.frames"
    session.sent()
    assert session.take() == dict(
        type="source.consumed", source_frame_end=8, source_bytes_end=96
    )
    session.source_progress(12, records, 0)
    with pytest.raises(queue.Empty):
        session.take()
    session.source_progress(16, records, 0)
    assert session.take()["source_frame_end"] == 16


def test_progress_without_rgb_releases_credit_without_waiting_for_video():
    _, session, _, _ = setup_session()
    records = [[1, 0, 8], [2, 8, 16]]
    session.source_progress(1, records, 0)
    assert session.queue.empty()
    session.source_progress(8, records, 0)
    assert session.take()["type"] == "source.consumed"


def test_output_failure_discards_queued_credit():
    _, session, _, cancelled = setup_session()
    session.put(dict(type="decoder.frames", reference="failed-output"), 0)
    session.source_progress(8, [[1, 0, 8]], 0)
    session.take()
    session.abort()
    assert cancelled == ["failed-output"]
    with pytest.raises(RuntimeError, match="closed"):
        session.take()


@pytest.mark.parametrize(
    "owned,records",
    [
        (17, [[1, 0, 16]]),
        (8, [[2, 4, 8]]),
        (8, [[1, 0, 4], [3, 4, 8]]),
        (8, [[1, 0, 4], [2, 5, 8]]),
    ],
)
def test_invalid_progress_cannot_grant_credit(owned, records):
    _, session, _, _ = setup_session()
    with pytest.raises(ValueError):
        session.source_progress(owned, records, 0)
    assert session.consumed_frames == 0 and session.queue.empty()
