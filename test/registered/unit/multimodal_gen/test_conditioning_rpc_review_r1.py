# SPDX-License-Identifier: Apache-2.0
"""R1 regressions at real socket, original forward and ASGI ownership boundaries."""

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

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import multiprocessing
import socket
import threading
import time
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.multimodal_gen.runtime.conditioning.client import ConditioningClient
from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError


def test_real_socket_header_deadline():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def server():
        conn, _ = listener.accept()
        with conn:
            conn.recv(65536)
            # Each byte arrives well within the 0.15s socket timeout, but the
            # status line alone takes > 1s. No complete HTTP response needed.
            try:
                for b in b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n":
                    conn.sendall(bytes([b]))
                    time.sleep(0.035)
            except (BrokenPipeError, ConnectionResetError):
                pass
        listener.close()

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    client = ConditioningClient(
        f"http://127.0.0.1:{listener.getsockname()[1]}",
        "secret",
        "a" * 64,
        timeout_s=0.15,
    )
    start = time.monotonic()
    with pytest.raises(ConditionError):
        client.encode("p", [], input_digest="b" * 64)
    elapsed = time.monotonic() - start
    print(f"HEADER_DEADLINE timeout_s=0.15 actual_s={elapsed:.3f}")
    thread.join(2)
    assert elapsed < 0.35


@pytest.mark.parametrize("fps", [5e-324, 1e-12, 24.0, 240.0])
def test_positive_fps_has_bounded_equivalent_original_forward(fps):
    from test_conditioning_rpc_core import Encoder, limits, media, profile

    from sglang.multimodal_gen.runtime.conditioning.wire import (
        decode_request,
        encode_request,
    )

    p = profile()
    items = [{**media(7, "video"), "fps": fps}]
    # The wire retains the caller's legal FPS; the remote boundary must make
    # its equivalent single-frame native sampling bounded.
    request = decode_request(encode_request("p", items, p.profile_id), p, limits())
    header, validated = request.header, request.media
    assert header["media"][0]["fps"] == fps
    reference = Encoder()(NS(prompt="p", media=[media(7, "video")]))
    context = multiprocessing.get_context("fork")
    entered, done = context.Event(), context.Event()
    matching = context.Value("b", False)

    def forward():
        # A forked child inherits the parent's OpenMP pool, which never runs again.
        torch.set_num_threads(1)
        entered.set()
        actual = Encoder()(NS(prompt="p", media=validated))
        matching.value = all(
            torch.equal(actual[name], reference[name]) for name in reference
        )
        done.set()

    child = context.Process(target=forward)
    child.start()
    assert entered.wait(1), "child must reach the unmodified original encode_condition"
    child.join(0.3)
    stuck = child.is_alive()
    if stuck:
        child.terminate()
    child.join()
    assert not stuck, "legal single-frame FPS must not monopolize native sampling"
    assert child.exitcode == 0
    assert done.is_set(), "the complete original forward must finish"
    assert matching.value, "context and row_role must match normal-FPS original forward"


def test_queued_timeout_drops_work_before_slow_error_send():
    import asyncio
    import sys

    sys.path.insert(0, "test/registered/unit/multimodal_gen")
    from test_conditioning_rpc_core import (
        PREFIX,
        components,
        encode,
        request_body,
        wait_until,
    )

    async def run():
        p, app, client, encoder = components()
        encoder.release.clear()
        first = asyncio.create_task(encode(client, p, prompt="running"))
        second = None
        sending, finish = asyncio.Event(), asyncio.Event()
        body = request_body("expired", [], p.profile_id, deadline_ms=100)
        consumed = False

        async def receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Future()

        async def send(message):
            if message["type"] == "http.response.start":
                assert message["status"] == 504
            if message["type"] == "http.response.body":
                sending.set()
                await finish.wait()

        try:
            await wait_until(encoder.started.is_set)
            second = asyncio.create_task(
                app(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": PREFIX + "/encode",
                        "headers": [
                            (b"authorization", b"Bearer secret"),
                            (b"content-type", b"application/octet-stream"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    },
                    receive,
                    send,
                )
            )
            await asyncio.wait_for(sending.wait(), 1)
            print("EXPIRED_QUEUE_DEPTH", app.queue.depth)
            encoder.release.set()
            await first
            await asyncio.sleep(0.1)
            print("ENCODER_CALLS_AFTER_504", encoder.calls)
            assert encoder.calls == ["running"]
        finally:
            finish.set()
            encoder.release.set()
            await asyncio.gather(
                *(t for t in (first, second) if t), return_exceptions=True
            )
            await client.aclose()
            await app.close()

    asyncio.run(run())


def test_redirect_does_not_forward_bearer_to_other_origin():
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received = []

    class Target(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.headers.get("Authorization"))
            body = ('{"profile_id":"' + "a" * 64 + '"}').encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    target = HTTPServer(("127.0.0.1", 0), Target)

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header(
                "Location", f"http://127.0.0.1:{target.server_port}/untrusted"
            )
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_):
            pass

    redirect = HTTPServer(("127.0.0.1", 0), Redirect)
    threads = [
        threading.Thread(target=s.serve_forever, daemon=True)
        for s in (target, redirect)
    ]
    for t in threads:
        t.start()
    try:
        client = ConditioningClient(
            f"http://localhost:{redirect.server_port}", "review-dummy-secret", "a" * 64
        )
        try:
            client.handshake()
        except ConditionError:
            pass
        print("REDIRECT_TARGET_AUTH", received)
        assert received == [] or received == [None]
    finally:
        for server in (target, redirect):
            server.shutdown()
            server.server_close()
        for t in threads:
            t.join()


def test_repeated_cancel_releases_request_ledger():
    import asyncio
    import sys

    sys.path.insert(0, "test/registered/unit/multimodal_gen")
    from test_conditioning_rpc_core import (
        components,
        encode,
        raw_request,
        request_body,
        wait_until,
    )

    async def run():
        p, app, client, encoder = components()
        encoder.release.clear()
        first = asyncio.create_task(encode(client, p, prompt="running"))
        second = None
        disconnect = asyncio.Event()
        try:
            await wait_until(encoder.started.is_set)
            body = request_body("queued", [], p.profile_id)
            second = asyncio.create_task(
                raw_request(
                    app,
                    body,
                    [(b"content-length", str(len(body)).encode())],
                    gate=disconnect,
                )
            )
            await wait_until(lambda: app.queue.depth == 1)
            second.cancel()
            asyncio.get_running_loop().call_soon(second.cancel)
            await asyncio.gather(second, return_exceptions=True)
            encoder.release.set()
            await first
            await asyncio.sleep(0.02)
            print("LEDGER_AFTER_DOUBLE_CANCEL", app.ledger.snapshot())
            assert app.ledger.snapshot()["global"] == {"slots": 0, "inflight": 0}
        finally:
            disconnect.set()
            encoder.release.set()
            await asyncio.gather(
                *(t for t in (first, second) if t), return_exceptions=True
            )
            await client.aclose()
            await app.close()

    asyncio.run(run())
