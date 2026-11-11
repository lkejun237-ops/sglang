# SPDX-License-Identifier: Apache-2.0
"""R3 integer boundaries and expired response serialization regressions."""

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
import time

import pytest
from test_conditioning_rpc_core import (
    PREFIX,
    Encoder,
    assert_idle,
    assert_ok,
    components,
    encode,
    limits,
    media,
    request_body,
)

from sglang.multimodal_gen.runtime.conditioning import service as service_module
from sglang.multimodal_gen.runtime.conditioning.contract import (
    ConditionError,
    canonical_json,
)
from sglang.multimodal_gen.runtime.conditioning.wire import (
    decode_request,
    encode_request,
    unpack_message,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HUGE_INTEGER = 10**400


@pytest.mark.parametrize("kind", ["image", "video"])
@pytest.mark.parametrize("fps", [HUGE_INTEGER, -HUGE_INTEGER])
def test_huge_fps_is_invalid_before_encoding(kind, fps):
    async def run():
        profile, app, client, encoder = components()
        body = request_body(
            "p",
            [media(kind=kind)],
            profile.profile_id,
            overrides={"media": [{"type": kind, "fps": fps, "tensor": "media_0"}]},
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
            assert app.queue.depth == 0
            assert app.flights == {}
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()
            assert_idle(app)

    asyncio.run(run())


def test_huge_deadline_wire_preserves_positive_integer():
    profile, app, client, _ = components()
    try:
        body = encode_request("p", [], profile.profile_id, HUGE_INTEGER)
        header, tensors = unpack_message(body)
        assert header["deadline_ms"] == HUGE_INTEGER
        assert type(header["deadline_ms"]) is int
        assert tensors == {}
        request = decode_request(body, profile, app.limits)
        decoded, items = request.header, request.media
        assert decoded == header
        assert items == []
    finally:
        asyncio.run(client.aclose())
        asyncio.run(app.close())
        assert_idle(app)


def test_huge_deadline_quick_encoder_succeeds():
    async def run():
        profile, app, client, encoder = components()
        try:
            response = await encode(
                client, profile, prompt="p", deadline_ms=HUGE_INTEGER
            )
            header, _ = assert_ok(response)
            assert header["profile_id"] == profile.profile_id
            assert encoder.calls == ["p"]
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()
            assert_idle(app)

    asyncio.run(run())


def test_huge_deadline_still_obeys_server_cap_and_native_ownership():
    async def run():
        encoder = Encoder()
        encoder.release.clear()
        profile, app, client, _ = components(
            encoder=encoder, configured_limits=limits(request_timeout_s=0.06)
        )
        started = time.monotonic()
        try:
            response = await encode(
                client, profile, prompt="p", deadline_ms=HUGE_INTEGER
            )
            elapsed = time.monotonic() - started
            assert response.status_code == 504
            assert response.json()["code"] == "CONDITION_TIMEOUT"
            assert 0.03 <= elapsed < 0.5
            assert encoder.started.is_set()
            assert encoder.calls == ["p"]
            snapshot = app.ledger.snapshot()["global"]
            assert snapshot["slots"] == 0
            assert snapshot["inflight"] > 0, (
                "running native work still owns its buffers"
            )
            assert app.queue.depth == 0
        finally:
            encoder.release.set()
            await client.aclose()
            await app.close()
            assert app.flights == {}
            assert_idle(app)

    asyncio.run(run())


@pytest.mark.parametrize(
    "deadline",
    [
        float("inf"),
        float("-inf"),
        float("nan"),
        0,
        -1,
        -HUGE_INTEGER,
        True,
        "1",
        [],
        None,
    ],
)
def test_invalid_deadlines_remain_invalid_on_wire_and_service(deadline):
    async def run():
        profile, app, client, encoder = components()
        try:
            with pytest.raises(ConditionError) as caught:
                encode_request("p", [], profile.profile_id, deadline)
            assert caught.value.code == "CONDITION_INVALID"
            body = request_body(
                "p", [], profile.profile_id, deadline_ms=deadline, allow_nonfinite=True
            )
            response = await client.post(
                PREFIX + "/encode",
                content=body,
                headers={"Content-Type": "application/octet-stream"},
            )
            assert response.status_code == 400
            assert response.json()["code"] == "CONDITION_INVALID"
            assert encoder.calls == []
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()
            assert_idle(app)

    asyncio.run(run())


@pytest.mark.parametrize(
    "delay,deadline_ms,status", [(0.12, 60, 504), (0.01, 1000, 200)]
)
def test_serialization_deadline_with_real_asgi_and_owned_buffers(
    monkeypatch, delay, deadline_ms, status
):
    original = service_module.pack_message
    ownership = []

    async def run():
        profile, app, client, encoder = components()

        def delayed_pack(*args):
            ownership.append(app.ledger.snapshot()["global"])
            time.sleep(delay)
            packed = original(*args)
            ownership.append(app.ledger.snapshot()["global"])
            return packed

        monkeypatch.setattr(service_module, "pack_message", delayed_pack)
        try:
            response = await encode(
                client, profile, prompt="p", deadline_ms=deadline_ms
            )
            assert response.status_code == status
            if status == 504:
                assert response.json()["code"] == "CONDITION_TIMEOUT"
            else:
                assert_ok(response)
            assert encoder.calls == ["p"]
            assert len(ownership) == 2
            assert all(s["slots"] == 1 and s["inflight"] > 0 for s in ownership)
            assert app.queue.depth == 0
            assert app.flights == {}
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()
            assert_idle(app)

    asyncio.run(run())


@pytest.mark.parametrize("remaining", [-0.1, 0.0, -1e-12])
def test_expired_success_send_does_not_start_and_error_grace_still_works(remaining):
    async def run():
        _, app, client, _ = components()
        app.clock = lambda: 2.0
        messages = []

        async def send(message):
            messages.append(message)

        try:
            with pytest.raises(ConditionError) as caught:
                await app._send_response(send, b"expired", None, 2.0 + remaining)
            assert caught.value.code == "CONDITION_TIMEOUT"
            assert messages == [], "expired success must not start an ASGI response"
            error = caught.value
            await app._send_response(
                send,
                canonical_json({"code": error.code, "message": error.public_message}),
                None,
                app.clock() + min(1.0, app.limits.request_timeout_s),
                status=error.status,
                content_type=b"application/json",
            )
            assert [
                m["status"] for m in messages if m["type"] == "http.response.start"
            ] == [504]
            assert b"CONDITION_TIMEOUT" in messages[-1]["body"]
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())


@pytest.mark.parametrize("remaining", [1.0, 1e-12])
def test_unexpired_send_including_near_zero_budget_succeeds(remaining):
    async def run():
        _, app, client, _ = components()
        app.clock = lambda: 2.0
        messages = []

        async def send(message):
            messages.append(message)

        try:
            await app._send_response(send, b"valid", None, 2.0 + remaining)
            assert [
                m["status"] for m in messages if m["type"] == "http.response.start"
            ] == [200]
            assert messages[-1]["body"] == b"valid"
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())
