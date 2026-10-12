# SPDX-License-Identifier: Apache-2.0
"""R2 compatibility, physical DNS ownership and untrusted media regressions."""

if __name__ == "__main__":
    import runpy
    import sys
    from pathlib import Path

    test_file = Path(__file__).resolve()
    runner = (
        test_file.parents[4] / "benchmark/zing_h3_causal/run_conditioning_cpu_tests.py"
    )
    arguments = ["-x" if value == "-f" else value for value in sys.argv[1:]]
    sys.argv = [str(runner), str(test_file), *arguments]
    runpy.run_path(str(runner), run_name="__main__")

import asyncio
import socket
import threading
import time

import httpx
import pytest
from sglang.multimodal_gen.runtime.conditioning import client as module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "owned network/DNS resources must drain"
        time.sleep(0.005)


@pytest.fixture
def network(monkeypatch):
    owned = module._NetworkLoop()
    monkeypatch.setattr(module, "_network_loop", lambda: owned)
    try:
        yield owned
    finally:
        wait_until(lambda: owned.capacity._value == 32)
        owned.loop._default_executor.shutdown(wait=True, cancel_futures=True)
        owned.loop.call_soon_threadsafe(owned.loop.stop)
        owned.thread.join(2)
        assert not owned.thread.is_alive()
        owned.loop.close()


def test_python310_api_surface_normal_handshake(monkeypatch, network):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, stream=httpx.ByteStream(b'{"profile_id":"' + b"a" * 64 + b'"}')
        )
    )
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: original(**kwargs, transport=transport)
    )
    monkeypatch.delattr(asyncio, "timeout", raising=False)
    module.ConditioningClient(
        "https://conditioning.example", "secret", "a" * 64, timeout_s=1
    ).handshake()
    wait_until(lambda: network.capacity._value == 32)


def test_python310_future_timeout_namespace_maps_to_condition_timeout(monkeypatch):
    class LegacyFutureTimeout(Exception):
        pass

    class Future:
        def result(self, timeout):
            raise LegacyFutureTimeout()

    monkeypatch.setattr(module.concurrent.futures, "TimeoutError", LegacyFutureTimeout)
    monkeypatch.setattr(
        asyncio, "run_coroutine_threadsafe", lambda coroutine, loop: Future()
    )
    network = object.__new__(module._NetworkLoop)
    network.capacity = threading.BoundedSemaphore(32)
    network.loop = None
    try:
        with pytest.raises(module.ConditionError) as caught:
            network.run(lambda: None, 0.01)
        assert caught.value.code == "CONDITION_TIMEOUT"
        # The sync wait cannot release resources still owned by async cleanup.
        assert network.capacity._value == 31
    finally:
        network.capacity.release()


def test_python310_async_timeout_namespace_maps_to_condition_timeout(monkeypatch):
    class LegacyAsyncTimeout(Exception):
        pass

    async def expired(coroutine, timeout):
        coroutine.close()
        raise LegacyAsyncTimeout()

    monkeypatch.delattr(asyncio, "timeout", raising=False)
    monkeypatch.setattr(asyncio, "TimeoutError", LegacyAsyncTimeout)
    monkeypatch.setattr(asyncio, "wait_for", expired)
    client = module.ConditioningClient("https://example", "secret", "a" * 64)
    with pytest.raises(module.ConditionError) as caught:
        asyncio.run(
            client._async_transport("GET", "/profile", None, time.monotonic() + 1)
        )
    assert caught.value.code == "CONDITION_TIMEOUT"


def profile_server(*, drip=False):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2)
    finished = threading.Event()
    body = b'{"profile_id":"' + b"a" * 64 + b'"}'

    def serve():
        try:
            conn, _ = listener.accept()
            with conn:
                conn.recv(65536)
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: "
                    + str(len(body)).encode()
                    + b"\r\n\r\n"
                )
                if drip:
                    for byte in body:
                        conn.sendall(bytes([byte]))
                        time.sleep(0.025)
                else:
                    conn.sendall(body)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        finally:
            listener.close()
            finished.set()

    thread = threading.Thread(target=serve, daemon=True)
    port = listener.getsockname()[1]
    thread.start()
    return port, thread, finished


def test_python310_real_slow_body_absolute_deadline_closes_transport(
    monkeypatch, network
):
    monkeypatch.delattr(asyncio, "timeout", raising=False)
    port, server, finished = profile_server(drip=True)
    client = module.ConditioningClient(
        f"http://127.0.0.1:{port}", "secret", "a" * 64, timeout_s=0.15
    )
    started = time.monotonic()
    try:
        with pytest.raises(module.ConditionError) as caught:
            client.handshake()
        assert caught.value.code == "CONDITION_TIMEOUT"
        assert time.monotonic() - started < 0.35
        wait_until(lambda: network.capacity._value == 32)
        assert finished.wait(1), "timeout must close the socket seen by the real peer"
    finally:
        server.join(3)


def test_dns_timeout_bounds_physical_queue_and_drains_before_reuse(
    monkeypatch, network
):
    original = socket.getaddrinfo
    release = threading.Event()
    entered = set()
    lock = threading.Lock()

    def blocked(*args, **kwargs):
        with lock:
            entered.add(threading.get_ident())
        assert release.wait(5), "controlled resolver must be released by test cleanup"
        return original("127.0.0.1", *args[1:], **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    client = module.ConditioningClient(
        "https://conditioning.invalid:1", "secret", "a" * 64, timeout_s=0.04
    )
    executor = network.loop._default_executor
    backlog = []
    try:
        for _ in range(24):
            with pytest.raises(module.ConditionError):
                client.handshake()
            backlog.append(executor._work_queue.qsize())
        wait_until(lambda: network.capacity._value == 32)
        assert len(entered) == 2
        assert len(executor._threads) == 2
        assert max(backlog) <= 2, "two running DNS calls leave at most two queued items"
        assert executor.capacity._value == 0, (
            "cancelled queued work remains physically owned"
        )
    finally:
        release.set()
        wait_until(lambda: executor.capacity._value == 4)
        wait_until(lambda: executor._work_queue.qsize() == 0)
        wait_until(lambda: network.capacity._value == 32)
    monkeypatch.setattr(socket, "getaddrinfo", original)
    port, server, finished = profile_server()
    try:
        module.ConditioningClient(
            f"http://localhost:{port}", "secret", "a" * 64, timeout_s=1
        ).handshake()
        assert finished.wait(1)
        wait_until(lambda: executor.capacity._value == 4)
        wait_until(lambda: network.capacity._value == 32)
        assert executor._work_queue.qsize() == 0
        assert len(executor._threads) == 2
    finally:
        server.join(3)


def test_dns_shutdown_removes_cancelled_physical_work_without_permit_leak():
    executor = module._BoundedDNSExecutor()
    release = threading.Event()
    started = [threading.Event(), threading.Event()]

    def blocked(index):
        started[index].set()
        assert release.wait(3)

    try:
        running = [executor.submit(blocked, i) for i in range(2)]
        assert all(event.wait(1) for event in started)
        queued = [executor.submit(lambda: None) for _ in range(2)]
        assert all(future.cancel() for future in queued)
        assert executor._work_queue.qsize() == 2
        assert executor.capacity._value == 0
        with pytest.raises(module.ConditionError):
            executor.submit(lambda: None)
        executor.shutdown(wait=False, cancel_futures=True)
        # shutdown drains queued items; only two running OS calls remain owned.
        assert executor.capacity._value == 2
        release.set()
        for future in running:
            future.result(timeout=1)
        wait_until(lambda: executor.capacity._value == 4)
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)


def test_sync_deadline_keeps_admission_until_actual_async_cleanup(monkeypatch, network):
    cleaning, release = threading.Event(), threading.Event()

    class Stream:
        async def __aenter__(self):
            await asyncio.Future()

        async def __aexit__(self, *args):
            pass

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            cleaning.set()
            while not release.is_set():
                await asyncio.sleep(0.005)

        def stream(self, *args, **kwargs):
            return Stream()

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    client = module.ConditioningClient(
        "https://example", "secret", "a" * 64, timeout_s=0.05
    )
    started = time.monotonic()
    try:
        with pytest.raises(module.ConditionError) as caught:
            client.handshake()
        assert caught.value.code == "CONDITION_TIMEOUT"
        assert time.monotonic() - started < 0.25
        assert cleaning.wait(1)
        assert network.capacity._value == 31
    finally:
        release.set()
        wait_until(lambda: network.capacity._value == 32)


@pytest.mark.parametrize("kind", [[], {}, 1, None, True, "audio"])
def test_untrusted_json_media_type_is_400_and_never_encodes(kind):
    from test_conditioning_rpc_core import (
        PREFIX,
        assert_idle,
        components,
        media,
        request_body,
    )

    async def run():
        profile, app, client, encoder = components()
        body = request_body(
            "p",
            [media()],
            profile.profile_id,
            overrides={"media": [{"type": kind, "fps": 1, "tensor": "media_0"}]},
        )
        try:
            response = await client.post(
                PREFIX + "/encode",
                content=body,
                headers={"Content-Type": "application/octet-stream"},
            )
            assert response.status_code == 400
            assert response.json()["code"] == "CONDITION_INVALID"
            assert response.json()["message"].isascii()
            assert encoder.calls == []
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["image", "video"])
def test_valid_media_types_still_encode(kind):
    from test_conditioning_rpc_core import (
        assert_idle,
        assert_ok,
        components,
        encode,
        media,
    )

    async def run():
        profile, app, client, encoder = components()
        try:
            assert_ok(await encode(client, profile, [media(kind=kind)], prompt="p"))
            assert encoder.calls == ["p"]
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())
