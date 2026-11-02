# SPDX-License-Identifier: Apache-2.0
"""No-GPU concurrency checks for the DiT communication owner."""

import threading
import time
from queue import Queue

import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_io import (
    DiTIOChannel,
)


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "Timed out waiting for the I/O owner"
        time.sleep(0.001)


class Transport:
    def __init__(self, incoming, received, closed, *, fail=None):
        self.owner = threading.get_ident()
        self.incoming, self.received, self.closed = incoming, received, closed
        self.fail = fail
        self.sent = []

    def check_owner(self):
        assert threading.get_ident() == self.owner

    def poll(self):
        self.check_owner()
        return not self.incoming.empty()

    def receive(self):
        self.check_owner()
        if self.fail == "receive":
            raise ValueError("Injected receive failure")
        value = self.incoming.get_nowait()
        self.received.append(value)
        return value

    def send(self, value):
        self.check_owner()
        if self.fail == "send":
            raise ValueError("Injected send failure")
        self.sent.append(value)

    def close(self):
        self.check_owner()
        self.closed.set()


def test_model_can_block_while_reception_fills_ten_slots_and_eos_stays_in_order():
    incoming, received, closed = Queue(), [], threading.Event()
    io = DiTIOChannel(lambda: Transport(incoming, received, closed)).start()
    try:
        for i in range(11):
            incoming.put(
                {"type": "conditioning" if i == 3 else "source.moments", "index": i}
            )
        incoming.put({"type": "source.end", "index": 11})
        # The model thread does not run an asyncio loop or consume any messages.
        # Reception must proceed independently, then stop at the queue budget.
        wait_until(lambda: io.inbox.qsize() == 10)
        assert len(received) == 10 and incoming.qsize() == 2
        assert io.transport.owner != threading.get_ident()
        # A full input queue must not prevent output from releasing peer credit.
        io.send({"type": "dit.latents"})
        assert io.transport.sent == [{"type": "dit.latents"}]
        results = []
        for _ in range(12):
            wait_until(io.poll)
            results.append(io.receive())
        assert [r["index"] for r in results] == list(range(12))
        assert results[-1]["type"] == "source.end"
    finally:
        io.close()
    assert closed.is_set() and not io.thread.is_alive()
    assert io.inbox.empty()


@pytest.mark.parametrize("fail", ["send", "receive"])
def test_transport_failure_reaches_consumer_and_exits_its_owner(fail):
    incoming, received, closed = Queue(), [], threading.Event()
    io = DiTIOChannel(lambda: Transport(incoming, received, closed, fail=fail)).start()
    try:
        if fail == "send":
            with pytest.raises((ValueError, RuntimeError)):
                io.send({"type": "reply"})
        else:
            incoming.put({"type": "source.moments"})
        assert closed.wait(3)
        with pytest.raises(RuntimeError, match="communication loop failed"):
            io.poll()
    finally:
        io.close()
    assert not io.thread.is_alive()


def test_startup_failure_is_propagated_without_leaving_a_thread():
    def fail():
        raise ValueError("Injected connection failure")

    io = DiTIOChannel(fail)
    with pytest.raises(ValueError, match="connection failure"):
        io.start()
    io.close()
    assert not io.thread.is_alive()


@pytest.mark.parametrize("count", [0, 10])
def test_idle_receiver_and_full_inbox_both_stop_without_peer_input(count):
    incoming, received, closed = Queue(), [], threading.Event()
    io = DiTIOChannel(lambda: Transport(incoming, received, closed)).start()
    for i in range(count):
        incoming.put(i)
    wait_until(lambda: io.inbox.qsize() == count)
    io.close()
    assert closed.is_set() and not io.thread.is_alive() and io.inbox.empty()
    with pytest.raises(RuntimeError, match="not running"):
        io.send({"type": "late"})


def test_pause_freezes_accepted_inbox_but_allows_results_to_be_sent():
    incoming, received, closed = Queue(), [], threading.Event()
    io = DiTIOChannel(lambda: Transport(incoming, received, closed)).start()
    try:
        incoming.put({"index": 0})
        wait_until(io.poll)
        io.pause_receiving()
        incoming.put({"index": 1})
        io.send({"type": "drain.result"})
        assert io.receive() == {"index": 0}
        assert not io.poll() and incoming.qsize() == 1
        assert io.transport.sent == [{"type": "drain.result"}]
    finally:
        io.close()


def test_one_send_failure_notifies_lifecycle_once():
    incoming, received, closed = Queue(), [], threading.Event()
    failures = []
    io = DiTIOChannel(
        lambda: Transport(incoming, received, closed, fail="send"),
        on_failure=lambda: failures.append("failed"),
    ).start()
    try:
        with pytest.raises((ValueError, RuntimeError)):
            io.send({"type": "result"})
        wait_until(lambda: len(failures) == 1)
    finally:
        io.close()
    assert failures == ["failed"]
