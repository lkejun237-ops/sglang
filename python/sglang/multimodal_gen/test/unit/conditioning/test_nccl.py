# SPDX-License-Identifier: Apache-2.0
"""Control/lifetime tests; GPU transport still requires a multi-GPU run."""

import asyncio
import contextlib
import gc
import threading
import time
import weakref
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.conditioning.client import ConditioningClient
from sglang.multimodal_gen.runtime.conditioning.contract import (
    ConditionError,
    canonical_json,
    safe_json,
)
from sglang.multimodal_gen.runtime.conditioning.inputs import (
    PREPROCESS_VERSION,
    content_identity,
)
from sglang.multimodal_gen.runtime.conditioning.nccl import (
    NcclReceiver,
    NcclSender,
    _Records,
    result_shapes,
)
from sglang.multimodal_gen.runtime.conditioning.service import ConditioningService
from sglang.multimodal_gen.runtime.conditioning.wire import (
    decode_response,
    encode_request,
    unpack_message,
)


def condition(rows=2):
    return {
        "context": torch.ones(rows, 5120, dtype=torch.bfloat16),
        "row_role": torch.arange(rows, dtype=torch.int64) % 2,
    }


def test_metadata_checks_identity_and_capacity_before_receive():
    metadata = dict(
        schema_version=1,
        profile_id="profile",
        input_digest="input",
        bridge_id="bridge",
        rows=3,
    )
    args = dict(
        profile_id="profile",
        input_digest="input",
        bridge_id="bridge",
        max_rows=3,
        max_bytes=3 * 10248,
    )
    assert result_shapes(metadata, **args) == {"context": (3, 5120), "row_role": (3,)}
    for update in (
        {"rows": True},
        {"rows": 4},
        {"rows": 0},
        {"input_digest": "old"},
        {"profile_id": "old"},
        {"bridge_id": "old"},
    ):
        with pytest.raises(ConditionError):
            result_shapes(metadata | update, **args)
    with pytest.raises(ConditionError):
        result_shapes(metadata, **(args | {"max_bytes": 3 * 10248 - 1}))


async def chunks(values):
    for value in values:
        yield value


def test_records_handle_fragmentation_truncation_and_excess_data():
    async def run():
        stream = _Records(chunks([b'{"rows', b'":2}\n{"done":"x"}\n']))
        assert await stream.read() == {"rows": 2}
        assert await stream.read() == {"done": "x"}
        await stream.finish()
        for values in ([b'{"rows":2}'], [b"x" * 65537]):
            with pytest.raises(ConditionError):
                await _Records(chunks(values)).read()
        stream = _Records(chunks([b"{}\nextra"]))
        await stream.read()
        with pytest.raises(ConditionError):
            await stream.finish()

    asyncio.run(run())


def test_sender_rejects_wrong_owner_concurrent_request_and_failed_channel():
    sender = NcclSender("cuda:0", 10)
    sender.group, sender.bridge_id, sender.tenant = object(), "b", "t"
    with pytest.raises(ConditionError, match="") as error:
        sender.acquire("b", "other")
    assert error.value.code == "CONDITION_UNAUTHORIZED"
    sender.acquire("b", "t")
    with pytest.raises(ConditionError) as error:
        sender.acquire("b", "t")
    assert error.value.code == "CONDITION_BUSY"
    sender.busy, sender.failed = False, True
    with pytest.raises(ConditionError) as error:
        sender.acquire("b", "t")
    assert error.value.code == "CONDITION_UNAVAILABLE"


def test_receiver_never_retries_ambiguous_exchange(monkeypatch):
    receiver = NcclReceiver(SimpleNamespace(clock=time.monotonic, timeout_s=10))
    receiver.group, receiver.device = object(), torch.device("cuda:0")
    calls = []

    def fail(*args):
        calls.append(1)
        raise ConnectionError("lost completion")

    monkeypatch.setattr(receiver, "_exchange", fail)
    with pytest.raises(ConnectionError):
        receiver.encode(b"", "first", "cuda:0")
    with pytest.raises(ConditionError):
        receiver.encode(b"", "second", "cuda:0")
    assert calls == [1]


def test_config_is_opt_in_and_handshake_rejects_transport_mismatch():
    def transport(*args):
        return SimpleNamespace(
            status=200,
            body=canonical_json({"profile_id": "p", "result_transport": "nccl"}),
        )

    client = ConditioningClient(
        "http://127.0.0.1:8093", "token", "p", transport=transport
    )
    assert client.result_transport == "http"
    with pytest.raises(ConditionError):
        client.handshake()
    client = ConditioningClient(
        "http://127.0.0.1:8093",
        "token",
        "p",
        transport=transport,
        result_transport="nccl",
    )
    client.handshake()
    with pytest.raises(ConditionError):
        ConditioningClient(
            "https://remote.example", "token", "p", result_transport="nccl"
        )
    with pytest.raises(ConditionError):
        ConditioningClient(
            "http://127.0.0.1",
            "token",
            "p",
            result_transport="nccl",
            timeout_s=10,
            sp_timeout_s=20,
        )


def make_service(mode):
    class Encoder:
        gpu_id = 0

        def __init__(self):
            self.calls = []
            self.result = condition()

        def __call__(self, value):
            self.calls.append("http")
            return self.result

        def encode_device(self, value):
            self.calls.append("device")
            return self.result

    profile = SimpleNamespace(
        profile_id="a" * 64, document={"preprocess_version": PREPROCESS_VERSION}
    )
    encoder = Encoder()
    service = ConditioningService(
        profile=profile,
        encoder=encoder,
        tokens={"token": {"tenant": "t"}},
        result_transport=mode,
    )
    if service.nccl:
        service.nccl.group = object()
        service.nccl.bridge_id, service.nccl.tenant = "b", "t"
    return service


async def request(service, send, *, prompt="next", disconnect=None):
    digest = content_identity(prompt, [], service.profile.profile_id)
    body = encode_request(prompt, [], service.profile.profile_id, input_digest=digest)
    first = True
    disconnected = disconnect or asyncio.Event()

    async def receive():
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    await service(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/conditioning/encode",
            "client": ("127.0.0.1", 1234),
            "headers": [
                (b"authorization", b"Bearer token"),
                (b"content-type", b"application/octet-stream"),
                (b"content-length", str(len(body)).encode()),
                (b"x-conditioning-bridge", b"b"),
            ],
        },
        receive,
        send,
    )
    return digest


@pytest.mark.parametrize("mode", ["http", "nccl"])
def test_service_successive_requests_preserve_protocol_and_cache_policy(
    mode, monkeypatch
):
    async def run():
        service = make_service(mode)
        messages, sent = [], []

        async def send(message):
            messages.append(message)

        if service.nccl:

            def transfer(value):
                assert messages[-1]["more_body"] is True
                assert safe_json(messages[-1]["body"].strip())["rows"] == 2
                assert value is service.encoder.result
                sent.append(value)

            monkeypatch.setattr(service.nccl, "send", transfer)
            # The queue must not stage result tensors or fill the CPU LRU.
        try:
            for prompt in ("first", "second", "second"):
                messages.clear()
                digest = await request(service, send, prompt=prompt)
                assert messages[0]["status"] == 200
                payload = b"".join(m.get("body", b"") for m in messages)
                if mode == "http":
                    metadata, _ = unpack_message(payload)
                    result = decode_response(
                        payload, service.profile.profile_id, digest
                    )
                    torch.testing.assert_close(
                        result["context"], service.encoder.result["context"]
                    )
                else:
                    records = _Records(chunks([payload]))
                    metadata = await records.read()
                    assert metadata["input_digest"] == digest
                    assert await records.read() == {"done": digest}
                    assert service.cache.current_bytes == 0
                    assert service.nccl.busy is False
                assert "qwen_prefix" not in metadata
                assert "result_source" not in metadata
                assert all(n == 0 for n in service.ledger.snapshot()["global"].values())
            assert service.encoder.calls == (
                ["http"] * 2 if mode == "http" else ["device"] * 3
            )
            registry = service.metrics.registry
            assert registry.get_sample_value(
                "world_model_conditioning_encodes_total", {"prefix_outcome": "unknown"}
            ) == (2 if mode == "http" else 3)
            assert registry.get_sample_value(
                "world_model_conditioning_results_total", {"source": "result_cache"}
            ) == (1 if mode == "http" else 0)
            assert (
                registry.get_sample_value(
                    "world_model_conditioning_reused_tokens_count"
                )
                == 0
            )
        finally:
            await service.close()

    asyncio.run(run())


def test_disconnect_during_send_holds_slot_and_poisoned_channel(monkeypatch):
    async def run():
        service = make_service("nccl")
        entered, release = threading.Event(), threading.Event()
        disconnect = asyncio.Event()
        messages = []

        def transfer(value):
            entered.set()
            assert release.wait(5)

        async def send(message):
            messages.append(message)

        monkeypatch.setattr(service.nccl, "send", transfer)
        task = asyncio.create_task(request(service, send, disconnect=disconnect))
        try:
            until = time.monotonic() + 5
            while not entered.is_set() and time.monotonic() < until:
                await asyncio.sleep(0.001)
            assert entered.is_set()
            disconnect.set()
            await asyncio.sleep(0.02)
            assert not task.done()
            assert service.nccl.busy
            assert service.ledger.snapshot()["global"]["slots"] == 1
            release.set()
            await task
            assert service.nccl.failed
            assert not service.nccl.busy
            assert all(n == 0 for n in service.ledger.snapshot()["global"].values())
            messages.clear()
            await request(service, send)
            assert messages[0]["status"] == 503
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await service.close()

    asyncio.run(run())


def test_transfer_error_does_not_start_a_second_http_response(monkeypatch):
    async def run():
        service = make_service("nccl")
        messages = []

        def fail(value):
            raise ConditionError("CONDITION_INVALID", "Unexpected result device")

        async def send(message):
            messages.append(message)

        monkeypatch.setattr(service.nccl, "send", fail)
        try:
            with pytest.raises(RuntimeError, match="NCCL result transfer failed"):
                await request(service, send)
            assert sum(m["type"] == "http.response.start" for m in messages) == 1
            assert service.nccl.failed and not service.nccl.busy
            assert all(n == 0 for n in service.ledger.snapshot()["global"].values())
        finally:
            await service.close()

    asyncio.run(run())


def test_worker_uses_device_result_and_keeps_session_cache():
    from sglang.multimodal_gen.runtime.conditioning.worker import (
        RemoteConditioner,
        WorkerConditioningState,
    )

    result, calls = condition(), []

    def encode_device(prompt, media, *, input_digest, device):
        calls.append((prompt, device))
        return result

    client = SimpleNamespace(result_transport="nccl", encode_device=encode_device)
    worker = RemoteConditioner(
        client,
        "a" * 64,
        rank=0,
        coordinate=lambda v: [v],
        broadcast_receipt=lambda v: v,
        allocate=lambda shapes, device, positive: positive,
        broadcast_tensor=lambda tensor, name: tensor,
    )
    state = WorkerConditioningState()
    assert worker("prompt", [], device="cpu", state=state) is result
    assert worker("prompt", [], device="cpu", state=state) is result
    assert calls == [("prompt", "cpu")]


@pytest.mark.parametrize("rank", [0, 1])
def test_nccl_session_starts_device_prefetch_only_on_leader(rank, monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning import prefetch
    from sglang.multimodal_gen.runtime.conditioning.worker import (
        RemoteConditioner,
        WorkerConditioningState,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )

    created = []
    owner = object()

    def create(endpoint, **kwargs):
        assert kwargs["device"] == torch.device("cuda:2")
        assert kwargs["max_bytes"] == 1024
        assert callable(kwargs["encode_request"])
        assert (
            kwargs["encode_request"]("rain", [], input_digest="digest", deadline=7)
            == "shared-task"
        )
        created.append(endpoint)
        return owner

    monkeypatch.setattr(prefetch, "ConditioningPrefetch", create)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    receipts = []

    def coordinate(receipt):
        receipts.append(receipt)
        return [receipt, {"ok": True}]

    provider = RemoteConditioner(
        SimpleNamespace(result_transport="nccl") if rank == 0 else None,
        "a" * 64,
        rank=rank,
        coordinate=coordinate,
    )
    state = ZingSourceStreamState()
    conditioning_state = WorkerConditioningState()

    def shared_encode(client, prompt, media, **kwargs):
        assert client is provider.client and prompt == "rain"
        assert kwargs == dict(
            input_digest="digest", device=torch.device("cuda:2"), deadline=7
        )
        return "shared-task"

    conditioning_state.encode_remote = shared_encode
    state._start_conditioning_prefetch(
        SimpleNamespace(conditioning_provider=provider),
        SimpleNamespace(
            session=SimpleNamespace(get_or_create_state=lambda cls: conditioning_state)
        ),
        "ipc:///unused-nccl-prefetch",
        {"conditioning_prefetch_max_bytes": 1024},
    )
    assert receipts == [{"ok": True}]
    expected = owner if rank == 0 else None
    assert state.conditioning_prefetch is expected
    assert getattr(provider, "_prefetch_owner", None) is expected
    assert conditioning_state.prefetch is expected
    assert len(created) == (1 if rank == 0 else 0)


def _transport_process(rank, path):
    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.tensor_bridge import (
        bridge_group,
        receive_tensor,
        send_tensor,
    )

    store = dist.FileStore(path, 2)
    group = bridge_group(
        store,
        rank,
        prefix="zing-conditioning-bridge",
        backend="gloo",
        timeout_seconds=15,
    )
    assert not dist.is_initialized()
    for rows in (2, 7, 3):
        expected = condition(rows)
        for name in ("context", "row_role"):
            value = expected[name]
            if rank == 1:
                send_tensor(group, value, peer=0)
            else:
                result = receive_tensor(group, value.shape, value.dtype, "cpu", peer=1)
                torch.testing.assert_close(result, value, rtol=0, atol=0)


def test_private_group_transfers_variable_conditions_between_processes(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_transport_process, args=(str(tmp_path / "store"),), nprocs=2, join=True)


@pytest.mark.parametrize("fail_receive", [False, True])
def test_receiver_stream_orders_metadata_tensors_and_completion(
    monkeypatch, fail_receive
):
    import httpx

    from sglang.multimodal_gen.runtime.conditioning import nccl

    client = SimpleNamespace(
        url="http://127.0.0.1:8093",
        token="token",
        profile_id="profile",
        timeout_s=10,
        max_rows=3,
        max_bytes=40000,
        clock=time.monotonic,
    )
    receiver = NcclReceiver(client)
    receiver.group, receiver.device = object(), torch.device("cuda:0")
    ready = threading.Event()

    def synchronize():
        events.append("ready")
        ready.set()

    receiver.stream = SimpleNamespace(synchronize=synchronize)
    events = []
    metadata = dict(
        schema_version=1,
        profile_id="profile",
        input_digest="input",
        bridge_id=receiver.bridge_id,
        rows=2,
    )

    closed = threading.Event()

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            events.append("metadata")
            yield canonical_json(metadata) + b"\n"
            while not ready.is_set():
                await asyncio.sleep(0.001)
            if fail_receive:
                return
            assert events == ["metadata", "context", "row_role", "ready"]
            yield b'{"done":"input"}\n'

        async def aclose(self):
            closed.set()

    original_client, original_empty = httpx.AsyncClient, torch.empty
    monkeypatch.setattr(
        nccl.httpx,
        "AsyncClient",
        lambda **kw: original_client(
            **kw,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=Body())
            ),
        ),
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(
        torch,
        "empty",
        lambda shape, *, dtype, device: original_empty(shape, dtype=dtype),
    )

    def receive(group, value, *, peer):
        name = "context" if value.ndim == 2 else "row_role"
        events.append(name)
        if fail_receive and name == "row_role":
            raise RuntimeError("injected receive failure")
        value.copy_(condition()[name])

    monkeypatch.setattr(nccl, "receive_into", receive)
    if fail_receive:
        with pytest.raises(RuntimeError, match="injected"):
            receiver.encode(b"input", "input", "cuda:0")
        assert receiver.failed and receiver.pending is not None
        with pytest.raises(ConditionError):
            receiver.encode(b"next", "next", "cuda:0")
        # The HTTP owner has no GPU references and closes at its own deadline.
        ready.set()
    else:
        result = receiver.encode(b"input", "input", "cuda:0")
        assert receiver.pending is None and not receiver.failed
        torch.testing.assert_close(result["context"], condition()["context"])
    assert closed.wait(2)


@pytest.mark.parametrize("abandoned", [False, True])
def test_idle_queue_releases_completed_device_results(monkeypatch, abandoned):
    async def run():
        service = make_service("nccl")
        entered, release = threading.Event(), threading.Event()
        disconnected = asyncio.Event()
        references = []
        loop = asyncio.get_running_loop()
        released = asyncio.Event()
        released_count = 0

        def record_release():
            nonlocal released_count
            released_count += 1
            if released_count == 2:
                released.set()

        def tensor_released(reference):
            loop.call_soon_threadsafe(record_release)

        class Encoder:
            gpu_id = 0

            def encode_device(self, value):
                entered.set()
                if abandoned:
                    assert release.wait(5)
                result = condition()
                references.extend(
                    weakref.ref(tensor, tensor_released) for tensor in result.values()
                )
                return result

        service.encoder = service.queue.encoder = Encoder()

        def transfer(value):
            # Publishing to the request must retain its result through send.
            gc.collect()
            assert all(ref() is not None for ref in references)

        monkeypatch.setattr(service.nccl, "send", transfer)

        async def send(message):
            pass

        task = asyncio.create_task(request(service, send, disconnect=disconnected))
        try:
            if abandoned:
                until = time.monotonic() + 5
                while not entered.is_set() and time.monotonic() < until:
                    await asyncio.sleep(0.001)
                assert entered.is_set()
                disconnected.set()
            await asyncio.wait_for(task, 5)
            release.set()
            until = time.monotonic() + 5
            while service.flights and time.monotonic() < until:
                await asyncio.sleep(0.001)
            assert not service.flights
            gc.collect()
            # Publishing a Future can wake the event loop before its executor
            # thread finishes releasing temporaries. Wait for actual tensor
            # destruction, while the service remains open and idle.
            await asyncio.wait_for(released.wait(), 5)
            assert len(references) == 2 and all(ref() is None for ref in references)
            assert service.cache.current_bytes == 0 and service.nccl.pending is None
            assert all(n == 0 for n in service.ledger.snapshot()["global"].values())
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await service.close()

    asyncio.run(run())


@pytest.mark.parametrize("open_elapsed", [8, 11])
def test_receiver_open_and_exchange_share_one_deadline(monkeypatch, open_elapsed):
    now = [0]
    receiver = NcclReceiver(SimpleNamespace(clock=lambda: now[0], timeout_s=10))
    exchanges = []

    def open_bridge(device, deadline):
        assert deadline == 10
        now[0] = open_elapsed
        receiver.group, receiver.device = object(), torch.device(device)

    monkeypatch.setattr(receiver, "_open", open_bridge)
    monkeypatch.setattr(
        receiver, "_exchange", lambda body, digest, deadline: exchanges.append(deadline)
    )
    if open_elapsed < 10:
        receiver.encode(b"", "x", "cuda:0")
        assert exchanges == [10]
    else:
        with pytest.raises(ConditionError) as error:
            receiver.encode(b"", "x", "cuda:0")
        assert error.value.code == "CONDITION_TIMEOUT" and not exchanges


@pytest.mark.parametrize("budget", [3, 30])
def test_device_fallback_keeps_deadline_through_serialization_and_open(
    monkeypatch, budget
):
    from sglang.multimodal_gen.runtime.conditioning import wire

    now, observed = [10.0], []
    client = ConditioningClient(
        "http://127.0.0.1:8093",
        "token",
        "a" * 64,
        result_transport="nccl",
        clock=lambda: now[0],
        timeout_s=10,
    )
    receiver = client._nccl = NcclReceiver(client)
    original_encode = wire.encode_request

    def serialize(*args, **kwargs):
        result = original_encode(*args, **kwargs)
        now[0] += 1
        return result

    def open_bridge(device, deadline):
        observed.append(("open", deadline, now[0]))
        now[0] += 1
        receiver.group, receiver.device = object(), torch.device(device)

    def exchange(body, digest, deadline):
        observed.append(("exchange", deadline, now[0]))
        return condition()

    monkeypatch.setattr(wire, "encode_request", serialize)
    monkeypatch.setattr(receiver, "_open", open_bridge)
    monkeypatch.setattr(receiver, "_exchange", exchange)
    result = client.encode_device(
        "new", [], input_digest="b" * 64, device="cuda:2", deadline=10 + budget
    )
    end = 10 + min(budget, 10)
    assert observed == [("open", end, 11), ("exchange", end, 12)]
    assert result["context"].shape == (2, 5120)


def test_serialization_timeout_does_not_poison_unused_receiver(monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning import wire

    now = [0.0]
    client = ConditioningClient(
        "http://127.0.0.1:8093",
        "token",
        "a" * 64,
        result_transport="nccl",
        clock=lambda: now[0],
        timeout_s=10,
    )
    receiver = client._nccl = NcclReceiver(client)
    original_encode = wire.encode_request

    def serialize(*args, **kwargs):
        result = original_encode(*args, **kwargs)
        now[0] += 2
        return result

    monkeypatch.setattr(wire, "encode_request", serialize)
    with pytest.raises(ConditionError) as error:
        client.encode_device(
            "new", [], input_digest="b" * 64, device="cuda:2", deadline=1
        )
    assert error.value.code == "CONDITION_TIMEOUT"
    assert not receiver.failed and receiver.group is None and receiver.pending is None
    receiver.group, receiver.device = object(), torch.device("cuda:2")
    monkeypatch.setattr(receiver, "_exchange", lambda *args: condition())
    assert client.encode_device("new", [], input_digest="b" * 64, device="cuda:2")


def test_allocate_records_consumer_stream_before_candidate_can_be_released(monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning.worker import _allocate

    events = []
    stream = object()

    class Tensor:
        def __init__(self, name):
            self.name = name

        def to(self, *, device):
            assert device == "cuda:2"
            return self

        def record_stream(self, current):
            assert current is stream
            events.append(self.name)

    def current_stream(device):
        assert device == "cuda:2"
        return stream

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    positive = {name: Tensor(name) for name in ("context", "row_role")}
    shapes = {"context": (2, 5120), "row_role": (2,)}
    transferred = _allocate(shapes, "cuda:2", positive)
    assert all(transferred[name] is positive[name] for name in positive)
    assert events == ["context", "row_role"]


@pytest.mark.parametrize("transport", ["http", "nccl"])
def test_ready_candidate_does_not_wait_for_unrelated_receive(monkeypatch, transport):
    from sglang.multimodal_gen.runtime.conditioning.worker import (
        RemoteConditioner,
        WorkerConditioningState,
    )

    unrelated_finished = threading.Event()
    events = []
    stream = SimpleNamespace(synchronize=lambda: events.append("consumer_ready"))

    def device_barrier(*args):
        raise AssertionError("Must not wait for the unrelated receive stream")

    monkeypatch.setattr(torch.cuda, "synchronize", device_barrier)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    result = condition()
    state = WorkerConditioningState()
    state.prefetch = SimpleNamespace(encode=lambda *args, **kwargs: result)

    def broadcast(value, name):
        assert not unrelated_finished.is_set()
        assert events[0] == "consumer_ready"
        events.append(name)

    worker = RemoteConditioner(
        SimpleNamespace(result_transport=transport),
        "a" * 64,
        rank=0,
        coordinate=lambda receipt: [receipt],
        broadcast_receipt=lambda receipt: receipt,
        allocate=lambda shapes, device, positive: positive,
        broadcast_tensor=broadcast,
    )
    assert worker("new", [], device="cuda:0", state=state) is result
    assert events == ["consumer_ready", "context", "row_role"]


def test_cold_open_separates_connection_and_persistent_timeouts(monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning import nccl

    now, observed = [9.0], {}
    client = SimpleNamespace(clock=lambda: now[0], timeout_s=10.0)
    receiver = NcclReceiver(client)
    store_timeouts = []

    def store(*args, **kwargs):
        observed["connect"] = args[4].total_seconds()
        return SimpleNamespace(
            port=31000,
            set_timeout=lambda value: store_timeouts.append(value.total_seconds()),
        )

    def transport(method, path, body, timeout):
        observed["request"] = safe_json(body)
        observed["http"] = timeout
        return SimpleNamespace(
            status=200, body=canonical_json({"bridge_id": receiver.bridge_id})
        )

    def group(store, rank, *, prefix, timeout_seconds):
        observed["operation"] = timeout_seconds
        return object()

    client.transport = transport
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: object())
    monkeypatch.setattr(nccl.dist, "TCPStore", store)
    monkeypatch.setattr(nccl, "bridge_group", group)
    receiver._open("cuda:0", deadline=10.0)
    assert observed["connect"] == observed["http"] == 1
    assert observed["request"]["connect_timeout_s"] == 1
    assert observed["request"]["timeout_s"] == observed["operation"] == 10
    assert store_timeouts == [1, 10]
    calls = []
    monkeypatch.setattr(
        receiver, "_exchange", lambda body, digest, deadline: calls.append(deadline)
    )
    now[0] = 20
    receiver.encode(b"", "new", "cuda:0")
    assert calls == [30] and store_timeouts[-1] == 10


@pytest.mark.parametrize("connect_budget", [None, 1.0])
def test_sender_connection_budget_counts_executor_wait(monkeypatch, connect_budget):
    from sglang.multimodal_gen.runtime.conditioning import nccl

    now, observed = [0.0], {}
    monkeypatch.setattr(nccl.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: object())
    store_timeouts = []

    def store(*args):
        observed["connect"] = args[4].total_seconds()
        now[0] += 0.25
        return SimpleNamespace(
            set_timeout=lambda value: store_timeouts.append(value.total_seconds())
        )

    def group(store, rank, *, prefix, timeout_seconds):
        observed["operation"] = timeout_seconds
        return object()

    monkeypatch.setattr(nccl.dist, "TCPStore", store)
    monkeypatch.setattr(nccl, "bridge_group", group)

    async def run(function, *args):
        now[0] += 0.25
        return function(*args)

    document = dict(port=31000, bridge_id="b" * 32, timeout_s=10)
    if connect_budget is not None:
        document["connect_timeout_s"] = connect_budget
    sender = NcclSender("cuda:0", 10)
    asyncio.run(sender.open(document, "tenant", run))
    budget = 10 if connect_budget is None else connect_budget
    assert observed == {"connect": budget - 0.25, "operation": 10}
    assert store_timeouts == [budget - 0.5, 10]
    assert not sender.failed and not sender.busy


def test_sender_expired_queue_budget_does_not_start_store(monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning import nccl

    now = [0.0]
    monkeypatch.setattr(nccl.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Expired connect budget must not start TCPStore")

    monkeypatch.setattr(nccl.dist, "TCPStore", forbidden)

    async def run(function, *args):
        now[0] = 2
        return function(*args)

    sender = NcclSender("cuda:0", 10)
    with pytest.raises(ConditionError) as error:
        asyncio.run(
            sender.open(
                dict(port=31000, bridge_id="b" * 32, timeout_s=10, connect_timeout_s=1),
                "tenant",
                run,
            )
        )
    assert error.value.code == "CONDITION_TIMEOUT"
    assert sender.failed and not sender.busy and sender.store is None


@pytest.mark.parametrize("budget", [True, 0, -1, 11, float("nan"), "1"])
def test_sender_rejects_invalid_connection_budget(budget):
    sender = NcclSender("cuda:0", 10)
    with pytest.raises(ConditionError) as error:
        asyncio.run(
            sender.open(
                dict(
                    port=31000,
                    bridge_id="b" * 32,
                    timeout_s=10,
                    connect_timeout_s=budget,
                ),
                "tenant",
                None,
            )
        )
    assert error.value.code == "CONDITION_INVALID"
    assert not sender.busy and not sender.failed and sender.bridge_id is None


@pytest.mark.parametrize("phase", ["metadata", "completion", "eof"])
def test_receiver_deadline_interrupts_real_http_stream(monkeypatch, phase):
    """Use a real socket: HTTPX inactivity timeouts cannot catch this peer."""
    from sglang.multimodal_gen.runtime.conditioning import nccl

    stop = threading.Event()
    state = {}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def write(value):
                self.wfile.write(f"{len(value):x}\r\n".encode() + value + b"\r\n")
                self.wfile.flush()

            metadata = (
                canonical_json(
                    dict(
                        schema_version=1,
                        profile_id="p",
                        input_digest="x",
                        bridge_id=state["bridge"],
                        rows=2,
                    )
                )
                + b"\n"
            )
            completion = b'{"done":"x"}\n'
            try:
                for name, value in (("metadata", metadata), ("completion", completion)):
                    if phase == name:
                        for byte in value:
                            if stop.wait(0.05):
                                return
                            write(bytes([byte]))
                    else:
                        write(value)
                if phase == "eof":
                    stop.wait(5)
                else:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    client = ConditioningClient(
        f"http://127.0.0.1:{server.server_port}",
        "token",
        "p",
        timeout_s=0.25,
        sp_timeout_s=0.8,
        result_transport="nccl",
    )
    receiver = NcclReceiver(client)
    receiver.group, receiver.device = object(), torch.device("cuda:0")
    receiver.stream = SimpleNamespace(synchronize=lambda: None)
    state["bridge"] = receiver.bridge_id
    received = []
    original_empty = torch.empty
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(
        torch,
        "empty",
        lambda shape, *, dtype, device: original_empty(shape, dtype=dtype),
    )
    monkeypatch.setattr(
        nccl, "receive_into", lambda *args, **kwargs: received.append(time.monotonic())
    )
    try:
        started = time.monotonic()
        with pytest.raises(ConditionError) as error:
            receiver.encode(b"x", "x", "cuda:0")
        elapsed = time.monotonic() - started
        assert error.value.code == "CONDITION_TIMEOUT"
        assert elapsed < 0.8
        assert len(received) == (0 if phase == "metadata" else 2)
        assert all(at - started < client.timeout_s for at in received)
        assert receiver.failed
        assert (receiver.pending is None) == (phase == "metadata")
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize(
    "cli_transport,json_transport,expected",
    [
        (None, None, "nccl"),
        (None, "http", "http"),
        (None, "nccl", "nccl"),
        ("http", None, "http"),
        ("nccl", "http", "nccl"),
        ("http", "nccl", "http"),
    ],
)
def test_cli_transport_overrides_both_endpoint_configs(
    tmp_path, monkeypatch, cli_transport, json_transport, expected
):
    import json
    from dataclasses import asdict

    import uvicorn

    from sglang.multimodal_gen.runtime.conditioning import cli, native, service, worker
    from sglang.multimodal_gen.runtime.conditioning.contract import DeploymentProfile
    from sglang.multimodal_gen.runtime.conditioning.resources import ResourceLimits

    config = dict(
        tokens={"token": {"tenant": "test"}},
        bearer_token="token",
        native={"model_path": "model", "gpu_id": 0},
        limits=asdict(ResourceLimits()),
    )
    if json_transport is not None:
        config["result_transport"] = json_transport
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    original = path.read_bytes()
    profile = SimpleNamespace(
        profile_id="p", document={"preprocess_version": PREPROCESS_VERSION}
    )
    monkeypatch.setattr(DeploymentProfile, "load", lambda path: profile)
    argv = ["serve", "--config", str(path), "--profile", "profile.json"]
    if cli_transport:
        argv += ["--conditioning-result-transport", cli_transport]
    # Exercise the actual serve entry point: a test-side fallback would mask
    # disagreement between the service and DiT defaults.
    captured = {}
    monkeypatch.setattr(native, "NativeConditionEncoder", lambda *a, **k: None)
    monkeypatch.setattr(cli, "warm_encoder", lambda encoder: None)
    monkeypatch.setattr(service, "ConditioningService", lambda **k: captured.update(k))
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    cli.main(argv)
    assert captured["result_transport"] == expected
    monkeypatch.setattr(worker, "_sp_group", lambda: None)
    monkeypatch.setattr(ConditioningClient, "handshake", lambda self: None)
    args = SimpleNamespace(
        conditioning_config=str(path),
        conditioning_expected_profile="profile.json",
        conditioning_service_url="http://127.0.0.1:8093",
        conditioning_result_transport=cli_transport,
    )
    provider = worker.RemoteConditioner.from_server_args(args)
    assert provider.client.result_transport == expected
    assert path.read_bytes() == original


@pytest.mark.parametrize("status", [410, 500])
def test_pretransfer_expiry_keeps_nccl_channel_recoverable(monkeypatch, status):
    import httpx

    from sglang.multimodal_gen.runtime.conditioning import nccl

    client = ConditioningClient(
        "http://localhost", "token", "a" * 64, result_transport="nccl"
    )
    receiver = NcclReceiver(client)
    receiver.group, receiver.device = object(), torch.device("cuda:0")
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(status))
    monkeypatch.setattr(
        nccl.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=transport, **kwargs),
    )
    for _ in range(2):
        with pytest.raises(ConditionError) as error:
            receiver.encode(b"body", "digest", "cuda:0")
        assert error.value.code == (
            "CONDITION_TASK_EXPIRED" if status == 410 else "CONDITION_UNAVAILABLE"
        )
        assert receiver.failed is (status != 410)
        assert receiver.pending is None
