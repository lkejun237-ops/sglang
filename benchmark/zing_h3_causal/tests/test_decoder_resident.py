# SPDX-License-Identifier: Apache-2.0
"""CPU ownership checks for the real Decoder resident GPU probe."""

from types import SimpleNamespace

import pytest
from diagnostics.decoder.benchmark_decoder_resident import consume_output


@pytest.fixture
def output_ownership():
    reference = {"owned": True}
    events = []
    service = SimpleNamespace(width=2, height=2)
    publisher = SimpleNamespace()

    def reserve(**kwargs):
        assert kwargs == dict(frames=1, width=2, height=2, identity={})
        events.append("reserve")
        return reference

    def submit(frames, value):
        assert frames == 1 and value is reference
        events.append("submit")

    def fail(value, error):
        assert value is reference and isinstance(error, RuntimeError)
        events.append("fail")

    def cancel(value):
        assert value is reference
        events.append("cancel")

    def materialize(value):
        assert value is reference
        events.append("materialize")
        return [[b"frame"]]

    publisher.reserve, publisher.fail = reserve, fail
    service.submit = submit
    return SimpleNamespace(
        service=service,
        publisher=publisher,
        cancel=cancel,
        materialize=materialize,
        events=events,
    )


def consume(state):
    return consume_output(
        state.service,
        state.publisher,
        1,
        {},
        materialize=state.materialize,
        cancel=state.cancel,
    )


def test_output_ownership_success_uses_consumer_ack(output_ownership):
    state = output_ownership
    assert consume(state) == [[b"frame"]]
    assert state.events == ["reserve", "submit", "materialize"]


def test_output_ownership_submit_failure_reclaims_unpublished_slot(output_ownership):
    state = output_ownership

    def rejected(*args):
        state.events.append("submit")
        raise RuntimeError("queue rejected")

    state.service.submit = rejected
    with pytest.raises(RuntimeError, match="queue rejected"):
        consume(state)
    assert state.events == ["reserve", "submit", "fail", "cancel"]


def test_output_ownership_consumer_failure_cancels_without_second_publish(
    output_ownership,
):
    state = output_ownership

    def failed(*args):
        state.events.append("materialize")
        raise RuntimeError("consumer disconnected")

    state.materialize = failed
    with pytest.raises(RuntimeError, match="consumer disconnected"):
        consume(state)
    assert state.events == ["reserve", "submit", "materialize", "cancel"]


def test_output_ownership_reserve_failure_has_no_reference_to_cancel(output_ownership):
    state = output_ownership

    def full(**kwargs):
        state.events.append("reserve")
        raise RuntimeError("slots full")

    state.publisher.reserve = full
    with pytest.raises(RuntimeError, match="slots full"):
        consume(state)
    assert state.events == ["reserve"]
