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


def test_blocked_startup_and_close_keep_unconfirmed_transport_owned():
    unblock, closed = threading.Event(), threading.Event()

    def connect():
        unblock.wait()
        return Transport(Queue(), [], closed)

    io = DiTIOChannel(connect, timeout_s=0.03)
    try:
        with pytest.raises(TimeoutError):
            io.start()
        assert io.thread.is_alive() and io.stopping.is_set()
        with pytest.raises(TimeoutError, match="cleanup is unconfirmed"):
            io.close()
        assert io.thread.is_alive() and not closed.is_set()
    finally:
        unblock.set()
        io.thread.join(timeout=3)
        io.close()
    assert closed.is_set() and not io.thread.is_alive()


def test_blocked_receive_cannot_claim_pause_or_clean_shutdown():
    incoming, received, closed = Queue(), [], threading.Event()
    entered, unblock = threading.Event(), threading.Event()

    class Blocked(Transport):
        def receive(self):
            self.check_owner()
            entered.set()
            unblock.wait()
            return super().receive()

    io = DiTIOChannel(
        lambda: Blocked(incoming, received, closed), timeout_s=0.03
    ).start()
    try:
        incoming.put({"type": "source.moments"})
        assert entered.wait(3)
        with pytest.raises(TimeoutError, match="pause was not confirmed"):
            io.pause_receiving()
        with pytest.raises(TimeoutError, match="cleanup is unconfirmed"):
            io.close()
        assert io.transport is not None and not closed.is_set()
    finally:
        unblock.set()
        io.thread.join(timeout=3)
        io.close()
    assert closed.is_set() and io.inbox.empty()


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


def test_dit_io_logs_first_error_json_on_exception(monkeypatch):
    import contextlib
    from types import SimpleNamespace

    import torch

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        dit_io,
    )

    logged = []

    class DummyLogger:
        def exception(self, msg, *args):
            logged.append(msg % args if args else msg)

    monkeypatch.setattr(dit_io, "logger", DummyLogger(), raising=False)
    active = [SimpleNamespace(cuda_stream=0)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: active[0])

    class ContextTransport(Transport):
        @contextlib.contextmanager
        def io_context(self):
            active[0] = SimpleNamespace(cuda_stream=5555)
            try:
                yield
            finally:
                active[0] = SimpleNamespace(cuda_stream=0)

    incoming, received, closed = Queue(), [], threading.Event()
    io = DiTIOChannel(
        lambda: ContextTransport(incoming, received, closed, fail="receive")
    ).start()
    try:
        incoming.put({"type": "source.moments"})
        assert closed.wait(3)
    finally:
        io.close()
    assert any(
        "ZING_DIT_IO_FIRST_ERROR_JSON" in entry and '"current_stream": 5555' in entry
        for entry in logged
    )


@pytest.mark.parametrize("abort_fails", [False, True])
@pytest.mark.parametrize("second_abort", [False, True])
def test_transport_abort_claim_prevents_concurrent_shutdown(abort_fails, second_abort):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_transport import (
        EncoderDiTChannel,
    )

    entered, release = threading.Event(), threading.Event()
    calls = []

    def abort():
        calls.append("abort")
        entered.set()
        assert release.wait(3)
        if abort_fails:
            raise RuntimeError("Injected native abort failure")

    channel = EncoderDiTChannel(
        SimpleNamespace(abort=abort, shutdown=lambda: calls.append("shutdown")),
        SimpleNamespace(close=lambda: calls.append("socket.close")),
        rank=0,
        device="cpu",
    )

    def io_cleanup():
        # A second fault observer must neither repeat abort nor drain the group.
        if second_abort:
            channel.abort()
        channel.close()

    with ThreadPoolExecutor(max_workers=2) as threads:
        model = threads.submit(channel.abort)
        try:
            assert entered.wait(3)
            threads.submit(io_cleanup).result(timeout=2)
            assert calls == ["abort", "socket.close"]
        finally:
            release.set()
        if abort_fails:
            with pytest.raises(RuntimeError, match="Injected native abort failure"):
                model.result(timeout=2)
        else:
            model.result(timeout=2)
    channel.abort()
    assert calls == ["abort", "socket.close"]


def test_transport_close_claim_prevents_concurrent_native_abort():
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_transport import (
        EncoderDiTChannel,
    )

    entered, release = threading.Event(), threading.Event()
    calls = []

    def shutdown():
        calls.append("shutdown")
        entered.set()
        assert release.wait(3)

    channel = EncoderDiTChannel(
        SimpleNamespace(abort=lambda: calls.append("abort"), shutdown=shutdown),
        SimpleNamespace(close=lambda: calls.append("socket.close")),
        rank=0,
        device="cpu",
    )
    with ThreadPoolExecutor(max_workers=2) as threads:
        io = threads.submit(channel.close)
        try:
            assert entered.wait(3)
            threads.submit(channel.abort).result(timeout=2)
            assert calls == ["shutdown"]
        finally:
            release.set()
        io.result(timeout=2)
    assert calls == ["shutdown", "socket.close"]


@pytest.mark.parametrize("abort_fails", [False, True])
def test_fault_abort_precedes_io_stop_notification(abort_fails):
    from concurrent.futures import ThreadPoolExecutor

    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    class FaultTransport(Transport):
        def abort(self):
            entered.set()
            assert release.wait(3)
            if abort_fails:
                raise RuntimeError("Injected native abort failure")

    io = DiTIOChannel(lambda: FaultTransport(Queue(), [], closed)).start()
    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            fault = threads.submit(io.abort)
            try:
                assert entered.wait(3)
                # Stop wakes the normal close path; publish the fault first.
                assert not io.stopping.is_set()
                assert not closed.is_set()
            finally:
                release.set()
            if abort_fails:
                with pytest.raises(RuntimeError, match="Injected native abort failure"):
                    fault.result(timeout=2)
            else:
                fault.result(timeout=2)
        assert closed.wait(3)
    finally:
        io.close()


@pytest.mark.parametrize("same_clock", [True, False])
def test_encoder_return_includes_receiver_backpressure(same_clock):
    incoming, received, closed = Queue(), [], threading.Event()

    class Sender(Transport):
        def send(self, message):
            incoming.put(dict(message))

    tx = DiTIOChannel(lambda: Sender(Queue(), [], threading.Event())).start()
    rx = DiTIOChannel(lambda: Transport(incoming, received, closed)).start()
    try:
        rx.pause_receiving()
        if not same_clock:
            tx.clock_domain = "different-host"
        start = time.perf_counter_ns()
        tx.send({"type": "source.moments", "encoder_frame_timings": [{"frame": 0}]})
        # Metadata is already sent, but the receiver cannot accept it yet.
        time.sleep(0.05)
        rx.pause_requested.clear()
        message = rx.receive()
        timing = message["encoder_frame_timings"][0]
        if same_clock:
            assert 50 <= timing["return_ms"] <= (time.perf_counter_ns() - start) / 1e6
        else:
            assert "return_ms" not in timing
        assert "_encoder_return_start_ns" not in message
        assert "_encoder_return_clock" not in message
    finally:
        tx.close()
        rx.close()
