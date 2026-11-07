# SPDX-License-Identifier: Apache-2.0
"""Real IPC and CPU request ownership; these tests do not prove GPU overlap."""

import asyncio
import threading
import time
import uuid
from types import SimpleNamespace

import msgspec.msgpack
import pytest
import zmq

from sglang.multimodal_gen.runtime.conditioning.prefetch import (
    ConditioningPrefetch,
    PromptPrefetchPublisher,
)


def wait_for(check):
    deadline = time.monotonic() + 5
    while not check():
        if time.monotonic() >= deadline:
            pytest.fail("Timed out waiting for prefetch")
        time.sleep(0.01)


class Client:
    timeout_s = 2
    clock = staticmethod(time.monotonic)

    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.deadlines = []

    def encode(self, prompt, media, *, input_digest, deadline=None):
        assert getattr(self, "result_transport", "http") == "http"
        return self._encode(prompt, media, input_digest=input_digest, deadline=deadline)

    def encode_device(self, prompt, media, *, input_digest, device, deadline=None):
        assert self.result_transport == "nccl" and device == "cuda:2"
        return self._encode(prompt, media, input_digest=input_digest, deadline=deadline)

    def _encode(self, prompt, media, *, input_digest, deadline):
        self.deadlines.append(deadline)
        self.calls.append(
            (prompt, media, input_digest, threading.current_thread().name)
        )
        self.started.set()
        assert self.release.wait(5)
        return {"context": prompt, "row_role": media}


@pytest.fixture(params=["http", "nccl"])
def channel(tmp_path, request):
    endpoint = "ipc://" + str(tmp_path / (uuid.uuid4().hex + ".sock"))
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.CONFLATE, 1)
    try:
        socket.bind(endpoint)
    except zmq.ZMQError:
        socket.close()
        context.term()
        raise
    client = Client()
    client.result_transport = request.param
    prefetch = ConditioningPrefetch(
        endpoint,
        prepare=lambda c: (c["prompt"], c.get("references", "active")),
        identity=lambda p, m: p + str(m),
        client=client,
        max_bytes=1 << 20,
        device="cuda:2" if request.param == "nccl" else None,
    )
    try:
        yield socket, prefetch, client
    finally:
        client.release.set()
        prefetch.close()
        socket.close()
        context.term()


def send(socket, prompt, **extra):
    socket.send(msgspec.msgpack.encode(dict(prompt=prompt, **extra)))


def test_rpc_starts_before_boundary_and_reuses_result(channel):
    socket, prefetch, client = channel
    client.release.clear()
    send(socket, "new")
    assert client.started.wait(5)
    assert client.calls[0][3] == "zing-qwen-prefetch"
    # Model the still-in-flight DiT owner: it has not called encode yet.
    assert prefetch.hits == 0
    client.release.set()
    wait_for(lambda: prefetch.encodes == 1)
    assert (
        prefetch.encode("new", "active", input_digest="newactive")["context"] == "new"
    )
    assert len(client.calls) == 1 and prefetch.hits == 1


def test_preparation_hit_skips_rpc_before_media_preparation(channel):
    socket, prefetch, client = channel
    prepared = []
    prefetch.prepare = lambda command: prepared.append(command)
    prefetch.should_prefetch = lambda command: False
    send(socket, "cached")
    wait_for(lambda: prefetch.cache_skips == 1)
    assert not prepared and not client.calls and prefetch.cached is None


def test_skipped_hint_can_fall_back_after_eviction_or_partial_rank_miss(channel):
    socket, prefetch, client = channel
    prefetch.should_prefetch = lambda command: False
    send(socket, "cached")
    wait_for(lambda: prefetch.cache_skips == 1)
    # The authoritative owner lookup can miss after the advisory local check.
    result = prefetch.encode("cached", "active", input_digest="cachedactive")
    assert result["context"] == "cached"
    assert len(client.calls) == 1 and prefetch.fallbacks == 1


def test_cache_hint_keeps_miss_prefetch_and_distinguishes_references(channel):
    socket, prefetch, client = channel
    prefetch.should_prefetch = lambda command: command.get("references") != "old"
    send(socket, "same", references="old")
    wait_for(lambda: prefetch.cache_skips == 1)
    send(socket, "same", references="new")
    wait_for(lambda: prefetch.encodes == 1)
    assert [call[:2] for call in client.calls] == [("same", "new")]
    assert prefetch.encode("same", "new", input_digest="samenew")["context"] == "same"
    assert len(client.calls) == 1 and prefetch.hits == 1


def test_cache_hint_failure_keeps_authoritative_boundary_fallback(channel):
    socket, prefetch, client = channel

    def invalid(command):
        raise ValueError("Invalid speculative cache key")

    prefetch.should_prefetch = invalid
    send(socket, "new")
    wait_for(lambda: prefetch.failures == 1)
    assert not client.calls
    assert (
        prefetch.encode("new", "active", input_digest="newactive")["context"] == "new"
    )
    assert len(client.calls) == 1


def test_boundary_waits_for_remaining_rpc_without_duplicate_encode(channel):
    socket, prefetch, client = channel
    client.release.clear()
    send(socket, "new")
    assert client.started.wait(5)
    result = []
    owner = threading.Thread(
        target=lambda: result.append(
            prefetch.encode("new", "active", input_digest="newactive")
        )
    )
    owner.start()
    assert not result
    client.release.set()
    owner.join(5)
    assert result[0]["context"] == "new" and len(client.calls) == 1


def test_changed_media_never_reuses_stale_result(channel):
    socket, prefetch, client = channel
    send(socket, "same", references="old")
    wait_for(lambda: prefetch.encodes == 1)
    result = prefetch.encode("same", "new", input_digest="samenew")
    assert result["row_role"] == "new"
    assert len(client.calls) == 2 and prefetch.hits == 0
    assert client.deadlines[0] is None and client.deadlines[1] is not None


def test_latest_delivered_event_is_cached_and_reused(channel):
    socket, prefetch, client = channel
    received_latest = threading.Event()
    original_prepare = prefetch.prepare

    def prepare(command):
        result = original_prepare(command)
        if command["prompt"] == "C":
            received_latest.set()
        return result

    prefetch.prepare = prepare
    client.release.clear()
    send(socket, "A")
    assert client.started.wait(5)
    send(socket, "B")
    send(socket, "C")
    client.release.set()
    assert received_latest.wait(5)
    wait_for(lambda: prefetch.cached is not None and prefetch.cached[0] == "Cactive")
    # send(C) does not acknowledge delivery. B can start before C arrives;
    # sender-side pending replacement has a separate controlled publisher test.
    calls = [c[0] for c in client.calls]
    assert calls in (["A", "C"], ["A", "B", "C"])
    encodes = prefetch.encodes
    assert prefetch.encode("C", "active", input_digest="Cactive")["context"] == "C"
    assert prefetch.encodes == encodes == len(client.calls) and prefetch.hits == 1


def test_receiver_failure_does_not_poison_boundary(channel):
    socket, prefetch, client = channel
    # Establish delivery before injecting an error into the conflating channel.
    send(socket, "old")
    wait_for(lambda: prefetch.encodes == 1)
    # Malformed private input fails preparation/identity, then fallback still works.
    socket.send(msgspec.msgpack.encode({"references": []}))
    wait_for(lambda: prefetch.failures == 1)
    assert prefetch.encode("valid", [], input_digest="valid[]")["context"] == "valid"
    assert prefetch.fallbacks == 1
    assert [call[0] for call in client.calls] == ["old", "valid"]


def test_close_drains_rpc_and_discards_result(channel):
    socket, prefetch, client = channel
    send(socket, "new")
    wait_for(lambda: prefetch.encodes == 1)
    prefetch.close()
    assert not prefetch.thread.is_alive() and prefetch.cached is None
    assert prefetch.should_prefetch is None
    prefetch.close()


def test_close_timeout_stops_late_cpu_preparation(channel, monkeypatch):
    socket, prefetch, client = channel
    preparing, release = threading.Event(), threading.Event()

    def prepare(command):
        preparing.set()
        assert release.wait(5)
        return command["prompt"], "active"

    prefetch.prepare = prepare
    join = prefetch.thread.join
    monkeypatch.setattr(prefetch.thread, "join", lambda timeout: join(0))
    try:
        send(socket, "new")
        assert preparing.wait(5)
        with pytest.raises(RuntimeError, match="did not drain"):
            prefetch.close()
        assert prefetch.thread.is_alive() and prefetch.stop.is_set()
    finally:
        release.set()
        join(5)
    assert not prefetch.thread.is_alive()
    assert not client.calls and prefetch.cached is None
    prefetch.close()


def test_conditioning_close_failure_releases_active_features():
    from sglang.multimodal_gen.runtime.conditioning.worker import (
        WorkerConditioningState,
    )

    state = WorkerConditioningState()
    state.identity, state.positive = "old", {"context": object()}
    state.media_hashes[1] = object()

    def fail_close():
        raise RuntimeError("Conditioning prefetch did not drain")

    owner = SimpleNamespace(close=fail_close)
    state.prefetch = owner
    with pytest.raises(RuntimeError, match="prefetch did not drain"):
        state.dispose()
    assert state.identity is state.positive is None and not state.media_hashes
    assert state.prefetch is owner


def test_publisher_retains_latest_and_returns_credit_on_cancel(tmp_path):
    credits = []

    def reserve(size):
        credit = SimpleNamespace(size=size, released=False)
        credit.release = lambda: setattr(credit, "released", True)
        credits.append(credit)
        return credit

    async def run():
        endpoint = "ipc://" + str(tmp_path / "no-receiver.sock")
        publisher = PromptPrefetchPublisher(endpoint, max_bytes=1024, reserve=reserve)
        publisher.submit({"prompt": "A"})
        await asyncio.sleep(0)
        publisher.submit({"prompt": "B"})
        publisher.submit({"prompt": "C"})
        assert credits[1].released
        task = publisher.task
        publisher.close()
        await asyncio.gather(task, return_exceptions=True)
        assert all(c.released for c in credits)

    asyncio.run(run())


def test_boundary_fallback_is_not_encoded_again_when_packet_arrives_late(channel):
    socket, prefetch, client = channel
    prefetch.encode("new", "active", input_digest="newactive")
    send(socket, "new")
    time.sleep(0.2)
    assert len(client.calls) == 1 and prefetch.fallbacks == 1


def test_closed_prefetch_cannot_encode(channel):
    _, prefetch, _ = channel
    prefetch.close()
    with pytest.raises(RuntimeError, match="closed"):
        prefetch.encode("new", "active", input_digest="newactive")


def test_boundary_wait_has_a_total_rpc_deadline(channel):
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    socket, prefetch, client = channel
    client.timeout_s = 0.05
    client.release.clear()
    send(socket, "old")
    assert client.started.wait(5)
    with pytest.raises(ConditionError) as error:
        prefetch.encode("new", "active", input_digest="newactive")
    assert error.value.code == "CONDITION_TIMEOUT"
    assert len(client.calls) == 1
    client.release.set()


@pytest.mark.parametrize("transport", ["http", "nccl"])
def test_worker_consumes_prefetched_features_on_owner_thread(transport):
    import torch

    from sglang.multimodal_gen.runtime.conditioning.worker import (
        RemoteConditioner,
        WorkerConditioningState,
    )

    owner = threading.current_thread().ident
    calls = []

    def record(value):
        assert threading.current_thread().ident == owner
        calls.append(value)
        return value

    positive = {
        "context": torch.ones(2, 4, dtype=torch.bfloat16),
        "row_role": torch.tensor([0, 1]),
    }
    prepared = []

    def encode(prompt, media, *, input_digest):
        prepared.append(input_digest)
        return positive

    state = WorkerConditioningState()
    state.prefetch = SimpleNamespace(encode=encode)
    remote = RemoteConditioner(
        SimpleNamespace(result_transport=transport),
        "a" * 64,
        rank=0,
        coordinate=lambda x: [record(x)],
        broadcast_receipt=record,
        broadcast_tensor=lambda x, name: record(x),
    )
    media = [
        {
            "id": "image",
            "type": "image",
            "frames": torch.zeros(3, 1, 2, 2, dtype=torch.uint8),
        }
    ]
    result = remote("new", media, device="cpu", state=state)
    assert len(prepared) == 1 and state.identity == prepared[0]
    assert torch.equal(result["context"], positive["context"])
    assert torch.equal(result["row_role"], positive["row_role"])
    assert state.positive is result and len(calls) >= 4


@pytest.mark.parametrize("transport", ["http", "nccl"])
def test_worker_prefetch_failure_preserves_active_condition(transport):
    import torch

    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.conditioning.worker import (
        RemoteConditioner,
        WorkerConditioningState,
    )

    def fail(*args, **kwargs):
        raise ConditionError("CONDITION_TIMEOUT")

    state = WorkerConditioningState()
    old = {"context": torch.ones(1, 4), "row_role": torch.tensor([0])}
    state.identity, state.positive = "old", old
    state.prefetch = SimpleNamespace(encode=fail)
    remote = RemoteConditioner(
        SimpleNamespace(result_transport=transport),
        "a" * 64,
        rank=0,
        coordinate=lambda x: [x],
        broadcast_receipt=lambda x: x,
    )
    media = [
        {
            "id": "image",
            "type": "image",
            "frames": torch.zeros(3, 1, 2, 2, dtype=torch.uint8),
        }
    ]
    with pytest.raises(ConditionError) as caught:
        remote("new", media, device="cpu", state=state)
    assert caught.value.code == "CONDITION_TIMEOUT"
    assert state.identity == "old" and state.positive is old


def test_nccl_prefetch_requires_explicit_leader_device():
    client = SimpleNamespace(result_transport="nccl")
    with pytest.raises(ValueError, match="leader device"):
        ConditioningPrefetch(
            "ipc:///unused", prepare=None, identity=None, client=client, max_bytes=1024
        )


def test_close_discards_inflight_result_and_prevents_new_encode(channel):
    socket, prefetch, client = channel
    client.release.clear()
    send(socket, "new")
    assert client.started.wait(5)
    closed = threading.Event()

    def close():
        prefetch.close()
        closed.set()

    closer = threading.Thread(target=close)
    closer.start()
    assert prefetch.stop.wait(5) and not closed.is_set()
    client.release.set()
    closer.join(5)
    assert closed.is_set() and prefetch.cached is None
    assert len(client.calls) == 1
    with pytest.raises(RuntimeError, match="closed"):
        prefetch.encode("new", "active", input_digest="newactive")


def test_nccl_background_failure_is_not_retried_on_the_poisoned_channel(channel):
    import torch

    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.conditioning.nccl import NcclReceiver

    socket, prefetch, client = channel
    # Exercise real IPC, the prefetch owner and receiver failure state together;
    # the exchange is a CPU fault injection, not a CUDA/NCCL validation.
    client.result_transport, prefetch.device = "nccl", "cuda:2"
    receiver = NcclReceiver(client)
    receiver.group, receiver.device = object(), torch.device("cuda:2")
    exchanges = []

    def fail_exchange(*args):
        exchanges.append(args)
        raise RuntimeError("injected ambiguous receive")

    receiver._exchange = fail_exchange
    client.encode_device = (
        lambda prompt, media, *, input_digest, device, deadline=None: receiver.encode(
            b"", input_digest, device, deadline=deadline
        )
    )
    send(socket, "new")
    wait_for(lambda: prefetch.failures == 1)
    assert receiver.failed and prefetch.cached is None
    with pytest.raises(ConditionError) as error:
        prefetch.encode("new", "active", input_digest="newactive")
    assert error.value.code == "CONDITION_UNAVAILABLE"
    assert len(exchanges) == 1
