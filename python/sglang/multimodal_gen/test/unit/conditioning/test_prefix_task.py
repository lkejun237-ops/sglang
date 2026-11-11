# SPDX-License-Identifier: Apache-2.0
import asyncio
import concurrent.futures
import threading
import time
import uuid

import pytest

from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.conditioning.prefix_task import SinglePrefixTask


def test_single_owner_release_drains_and_stale_handle_cannot_clear_next_task():
    async def run():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:

            class Cache:
                cleared = False

                def clear(self):
                    self.cleared = True

            manager = SinglePrefixTask(Cache, pool, 1800)
            nonce = uuid.uuid4().hex
            handle = manager.open("a", nonce)
            assert manager.open("a", nonce) == handle
            with pytest.raises(ConditionError, match="busy"):
                manager.open("a", uuid.uuid4().hex)
            with pytest.raises(ConditionError):
                manager.acquire("b", handle)
            active = manager.acquire("a", handle)
            entered, finish = threading.Event(), threading.Event()

            def forward():
                entered.set()
                assert finish.wait(5)
                assert not active.cache.cleared

            pending = pool.submit(forward)
            while not entered.is_set():
                await asyncio.sleep(0.001)
            release = asyncio.create_task(manager.release("a", handle))
            await asyncio.sleep(0.01)
            assert active.closed and not active.cache.cleared
            with pytest.raises(ConditionError):
                manager.acquire("a", handle)
            release.cancel()
            with pytest.raises(asyncio.CancelledError):
                await release
            with pytest.raises(ConditionError):
                manager.open("a", uuid.uuid4().hex)
            finish.set()
            await asyncio.wrap_future(pending)
            await active.release
            manager.leave(active)
            assert active.cache.cleared
            second = manager.open("a", uuid.uuid4().hex)
            await manager.release("a", handle)
            assert manager.active.handle == second
            await manager.close()
            assert manager.active is None

    asyncio.run(run())


def test_idle_expiry_and_active_request_keepalive():
    async def run():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            manager = SinglePrefixTask(dict, pool, 0.01)
            handle = manager.open("a", uuid.uuid4().hex)
            task = manager.acquire("a", handle)
            await asyncio.sleep(0.03)
            assert manager.active is task
            manager.leave(task)
            await asyncio.wait_for(task.expiry, timeout=1)
            assert manager.active is None
            with pytest.raises(ConditionError) as error:
                manager.acquire("a", handle)
            assert error.value.code == "CONDITION_TASK_EXPIRED"
            await manager.close()

    asyncio.run(run())


@pytest.mark.parametrize("transport", ["http", "nccl"])
def test_worker_lease_is_released_with_session_and_no_session_uses_full_path(transport):
    import torch

    from sglang.multimodal_gen.runtime.conditioning.worker import (
        RemoteConditioner,
        WorkerConditioningState,
    )

    class Client:
        supports_prefix_tasks = True
        clock = staticmethod(time.monotonic)
        timeout_s = 10
        opened, released, handles = [], [], []

        def open_task(self, nonce, *, deadline=None):
            self.opened.append(nonce)
            return "handle"

        def release_task(self, handle):
            self.released.append(handle)

        def encode(
            self, prompt, media, *, input_digest, task_handle=None, deadline=None
        ):
            self.handles.append(task_handle)
            return {
                "context": torch.ones(2, 5120, dtype=torch.bfloat16),
                "row_role": torch.ones(2, dtype=torch.int64),
            }

    client = Client()
    client.result_transport = transport
    if transport == "nccl":
        original_encode = client.encode

        def encode_device(*args, device, **kwargs):
            assert device == "cpu"
            return original_encode(*args, **kwargs)

        def unexpected_http(*args, **kwargs):
            pytest.fail("NCCL worker must not use HTTP tensor results")

        client.encode_device = encode_device
        client.encode = unexpected_http
    remote = RemoteConditioner(client, "a" * 64, rank=0)
    state = WorkerConditioningState()
    remote("events: N/A", [], device="cpu", state=state)
    remote("events: rain", [], device="cpu", state=state)
    state.dispose()
    state.dispose()
    remote("stateless", [], device="cpu")
    assert len(client.opened) == 1
    assert client.released == ["handle"]
    assert client.handles == ["handle", "handle", None]
    with pytest.raises(ConditionError):
        remote("closed", [], device="cpu", state=state)


@pytest.mark.parametrize("expired", [False, True])
def test_registration_and_recovery_share_the_sp_wait_budget(expired):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.conditioning.client import ConditioningClient
    from sglang.multimodal_gen.runtime.conditioning.contract import canonical_json
    from sglang.multimodal_gen.runtime.conditioning.worker import (
        RemoteConditioner,
        WorkerConditioningState,
    )

    now, timeouts, announcements = [0.0], [], []
    handle = uuid.uuid4().hex

    def transport(method, path, body, timeout):
        timeouts.append(timeout)
        if expired and len(timeouts) == 1:
            now[0] += 3
            return SimpleNamespace(status=410, body=b"")
        now[0] += min(6, timeout)
        return SimpleNamespace(status=200, body=canonical_json({"task_handle": handle}))

    client = ConditioningClient(
        "http://localhost",
        "token",
        "a" * 64,
        timeout_s=10,
        sp_timeout_s=15,
        clock=lambda: now[0],
        transport=transport,
    )
    client.supports_prefix_tasks = True
    state = WorkerConditioningState()
    if expired:
        state.task_handle = uuid.uuid4().hex
    remote = RemoteConditioner(
        client,
        "a" * 64,
        rank=0,
        broadcast_receipt=lambda value: announcements.append(value) or value,
    )
    with pytest.raises(ConditionError) as error:
        remote("event: rain", [], device="cpu", state=state)
    assert error.value.code == "CONDITION_TIMEOUT"
    assert now[0] == 10
    assert timeouts == ([10, 7, 1] if expired else [10, 4])
    assert announcements == [{"ok": False, "code": "CONDITION_TIMEOUT"}]


@pytest.mark.parametrize("fails", [False, True])
def test_dispose_drains_prefetch_before_releasing_prefix_task(fails):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.conditioning.worker import (
        WorkerConditioningState,
    )

    events = []

    def close():
        events.append("prefetch")
        if fails:
            raise RuntimeError("drain failed")

    state = WorkerConditioningState()
    state.prefetch = SimpleNamespace(close=close)
    state.task_handle = "handle"
    state.task_client = SimpleNamespace(
        release_task=lambda handle: events.append(handle)
    )
    if fails:
        with pytest.raises(RuntimeError, match="drain failed"):
            state.dispose()
    else:
        state.dispose()
    assert events == ["prefetch", "handle"]
    assert state.disposed and state.task_handle is None and state.positive is None


def test_device_request_carries_prefix_handle_and_shared_deadline():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.conditioning.client import ConditioningClient
    from sglang.multimodal_gen.runtime.conditioning.inputs import content_identity
    from sglang.multimodal_gen.runtime.conditioning.wire import unpack_message

    client = ConditioningClient(
        "http://localhost",
        "token",
        "a" * 64,
        result_transport="nccl",
        timeout_s=10,
        clock=lambda: 3.0,
    )
    handle = uuid.uuid4().hex
    digest = content_identity("rain", [], "a" * 64)
    calls = []

    def receive(body, identity, device, *, deadline):
        header, _ = unpack_message(body)
        calls.append((header["task_handle"], header["deadline_ms"], deadline))
        assert identity == digest and device == "cuda:2"
        return "device-result"

    client._nccl = SimpleNamespace(encode=receive)
    assert (
        client.encode_device(
            "rain",
            [],
            input_digest=digest,
            device="cuda:2",
            task_handle=handle,
            deadline=7.0,
        )
        == "device-result"
    )
    assert calls == [(handle, 4000, 7.0)]


def test_dispose_waits_for_registration_and_releases_late_handle():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.conditioning.worker import (
        WorkerConditioningState,
    )

    state = WorkerConditioningState()
    entered, finish, disposed = threading.Event(), threading.Event(), threading.Event()
    events, errors = [], []

    def register(nonce, *, deadline):
        entered.set()
        assert finish.wait(5)
        events.append("registered")
        return "late-handle"

    client = SimpleNamespace(
        supports_prefix_tasks=True,
        clock=time.monotonic,
        timeout_s=2,
        open_task=register,
        encode=lambda *args, **kwargs: events.append("encoded"),
        release_task=lambda handle: events.append(handle),
    )

    def encode():
        try:
            state.encode_remote(client, "rain", [], input_digest="digest")
        except BaseException as error:
            errors.append(error)

    def dispose():
        try:
            state.dispose()
        except BaseException as error:
            errors.append(error)
        finally:
            disposed.set()

    worker = threading.Thread(target=encode)
    closer = threading.Thread(target=dispose)
    worker.start()
    try:
        assert entered.wait(5)
        closer.start()
        assert not disposed.wait(0.05)
    finally:
        finish.set()
        worker.join(5)
        closer.join(5)
    assert not errors and disposed.is_set()
    assert events == ["registered", "encoded", "late-handle"]
    assert state.task_handle is None
    with pytest.raises(ConditionError):
        state.encode_remote(client, "late", [], input_digest="late")
    assert events == ["registered", "encoded", "late-handle"]


def test_task_lock_wait_counts_against_boundary_deadline():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.conditioning.worker import (
        WorkerConditioningState,
    )

    state = WorkerConditioningState()
    client = SimpleNamespace(clock=time.monotonic, timeout_s=1)
    state._task_lock.acquire()
    try:
        start = time.monotonic()
        with pytest.raises(ConditionError) as error:
            state.encode_remote(
                client, "rain", [], input_digest="x", deadline=start + 0.03
            )
        assert error.value.code == "CONDITION_TIMEOUT"
        assert time.monotonic() - start < 0.5
    finally:
        state._task_lock.release()
