# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import socket
import time
from urllib.parse import parse_qs, quote, urlsplit

import httpx
import pytest
import uvicorn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
    create_app,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    decode_message,
    encode_message,
)
from sglang.multimodal_gen.runtime.realtime.coordinator import (
    CoordinatorRejected,
    SessionAssignment,
    WorkerSlot,
)
from sglang.multimodal_gen.runtime.realtime.world_platform import WorldPlatformConfig


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _world_token(
    key: Ed25519PrivateKey,
    *,
    run_id: str,
    attempt_id: str,
    record_video: bool,
) -> str:
    now = int(time.time())
    claims = {
        "iss": "world-service",
        "aud": "world-session",
        "sub": "s_player",
        "uid": "u_player",
        "sid": run_id,
        "jti": attempt_id,
        "iat": now,
        "exp": now + 90,
        "session": {
            "max_lifetime_s": 90,
            "allow_free_prompt": False,
            "record_video": record_video,
        },
    }
    signing = (
        _b64url(json.dumps({"alg": "EdDSA", "typ": "JWT"}).encode())
        + "."
        + _b64url(json.dumps(claims).encode())
    )
    return signing + "." + _b64url(key.sign(signing.encode()))


class _Coordinator:
    def __init__(
        self,
        denoiser_endpoint: str,
        *,
        admit_rejection: str | None = None,
    ):
        self.denoiser_endpoint = denoiser_endpoint
        self.admit_rejection = admit_rejection
        self.admitted = []
        self.released = []
        self.released_at = None
        self.attempt_claims = []
        self.attempt_owners = {}
        self._attempt_lock = asyncio.Lock()

    async def health(self):
        return {"status": "ready"}

    async def claim_attempt(self, *, attempt_id, owner_id, expires_at):
        async with self._attempt_lock:
            self.attempt_claims.append((attempt_id, owner_id, expires_at))
            current = self.attempt_owners.get(attempt_id)
            if current is not None and current != owner_id:
                raise CoordinatorRejected("ATTEMPT_ALREADY_OWNED")
            self.attempt_owners[attempt_id] = owner_id

    async def admit(self, **request):
        self.admitted.append(request)
        if self.admit_rejection is not None:
            raise CoordinatorRejected(self.admit_rejection, retry_after_s=0.1)
        return SessionAssignment(
            user_id=request["user_id"],
            session_id=request["session_id"],
            generation_id=request["generation_id"],
            token="lease-token",
            expires_at=time.monotonic() + 60,
            denoiser=WorkerSlot(
                worker_id="denoiser-1",
                role="denoiser",
                endpoint=self.denoiser_endpoint,
                az="test-a",
                slot_index=0,
                model_revision=request["model_revision"],
                vae_fingerprint=request["vae_fingerprint"],
            ),
            vae=WorkerSlot(
                worker_id="vae-1",
                role="vae",
                endpoint="ws://vae-1:18081/v1/realtime_vae/decode",
                az="test-a",
                slot_index=0,
                model_revision="",
                vae_fingerprint=request["vae_fingerprint"],
            ),
        )

    async def renew(self, assignment):
        return assignment

    async def release(self, assignment):
        self.released_at = time.monotonic()
        self.released.append(assignment)


class _TraceQuery:
    async def query(self, trace_id, **_kwargs):
        return {
            "trace_id": trace_id,
            "events": [{"event": "gateway.ws_accepted", "trace_seq": 1}],
            "next_cursor": 1,
        }


@pytest.mark.parametrize(
    ("claim_error", "expected_code"),
    [
        (
            CoordinatorRejected("INVALID_ATTEMPT_OWNER"),
            "SESSION_UNAVAILABLE",
        ),
        (
            CoordinatorRejected("COORDINATOR_HTTP_404"),
            "SESSION_UNAVAILABLE",
        ),
        (OSError("claim response was lost"), "GATEWAY_SESSION_FAILED"),
    ],
)
def test_world_attempt_claim_failure_never_sends_lifecycle_callback(
    monkeypatch,
    claim_error,
    expected_code,
):
    callback_instances = []

    class ClaimFailingCoordinator(_Coordinator):
        async def claim_attempt(self, **_kwargs):
            raise claim_error

        async def admit(self, **_request):
            raise AssertionError("admission must not run after a failed attempt claim")

    class CallbackProbe:
        def __init__(self, _config):
            self.calls = []
            callback_instances.append(self)

        def started(self, *_args):
            raise AssertionError("started callback is not expected")

        def ended(self, *_args):
            raise AssertionError("ended callback is not expected")

        def recorded(self, *_args):
            raise AssertionError("recorded callback is not expected")

        def aborted(
            self,
            run_id,
            attempt_id,
            trace_id,
            fault,
            reason,
            established,
        ):
            self.calls.append(
                {
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "trace_id": trace_id,
                    "fault": fault,
                    "reason": reason,
                    "established": established,
                }
            )

        async def rewrite(self, *_args):
            raise AssertionError("rewrite is not expected")

        async def close(self):
            pass

    class RecordingManager:
        def start(self, **_kwargs):
            raise AssertionError("recording must not start before attempt claim")

        async def close(self):
            pass

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server.WorldCallbacks",
        CallbackProbe,
    )
    signing_key = Ed25519PrivateKey.generate()
    run_id = "11111111-1111-1111-1111-111111111111"
    attempt_id = "22222222-2222-2222-2222-222222222222"
    token = _world_token(
        signing_key,
        run_id=run_id,
        attempt_id=attempt_id,
        record_video=True,
    )
    coordinator = ClaimFailingCoordinator("ws://denoiser.invalid/generate")
    app = create_app(
        coordinator,
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        internal_output_url="ws://gateway.invalid/v1/internal/realtime_output",
        world_platform=WorldPlatformConfig(
            public_key=signing_key.public_key(),
            callback_url="http://world-service.invalid",
            callback_app_id="zing-gateway",
            callback_key_id="k1",
            callback_secret="secret",
            recording_bucket="run-videos",
            recording_region="cn-beijing",
        ),
        recording_manager=RecordingManager(),
        release_grace_s=0,
    )

    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/backends/minwm/v1/realtime_video/authorized_generate"
            f"?token={quote(token, safe='')}&trace_id=trace-claim"
        ) as websocket,
    ):
        message = decode_message(websocket.receive_bytes())
        assert message["type"] == "error"
        assert message["code"] == expected_code
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_bytes()

    # 混合版本下，旧 Coordinator 的 4xx 不能证明其它新版副本没有取得 owner；
    # 网络错误同样可能发生在条件写已提交之后。只要本连接没有成功 claim，
    # 无论拒绝是否“明确”，都没有资格修改业务侧 attempt。
    assert callback_instances[0].calls == []
    assert coordinator.admitted == []
    assert coordinator.released == []


async def _run_gateway(app, port: int):
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="off",
        )
    )
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.01)
    assert server.started
    return server, task


def test_gateway_routes_control_and_direct_vae_media_and_queries_trace_over_http():
    async def run():
        gateway_port = _free_port()
        denoiser_port = _free_port()
        denoiser_close_started_at = None

        async def denoiser(connection):
            nonlocal denoiser_close_started_at
            query = parse_qs(urlsplit(connection.request.path).query)
            session_id = query["session_id"][0]
            generation_id = query["generation_id"][0]
            output_url = query["gateway_output_url"][0]
            output_token = query["gateway_output_token"][0]

            await connection.send(encode_message("session_ready"))
            async with connect(output_url, max_size=None, compression=None) as output:
                await output.send(
                    encode_message(
                        "session_output_open",
                        session_id=session_id,
                        generation_id=generation_id,
                        token=output_token,
                    )
                )
                accepted = decode_message(await output.recv())
                assert accepted["type"] == "session_output_accepted"

                async def send_frame(index: int):
                    await output.send(
                        encode_message(
                            "frame_batch",
                            session_id=session_id,
                            generation_id=generation_id,
                            request_id="request-0",
                            chunk_index=0,
                            frame_batch_index=index,
                            payload_lengths=[4],
                            payload=b"webp",
                            content_type="image/webp",
                            width=8,
                            height=8,
                            num_frames=1,
                            is_final_frame_batch=False,
                        )
                    )

                await send_frame(0)
                denoiser_close_started_at = time.monotonic()
                await connection.close(code=1000, reason="generation complete")
                # The VAE media route is independent and can still be flushing
                # accepted frames after the Denoiser control route closes.
                await asyncio.sleep(0.05)
                await send_frame(1)
                await send_frame(2)
                await send_frame(3)
                await output.send(
                    encode_message(
                        "media_chunk_complete",
                        session_id=session_id,
                        generation_id=generation_id,
                        request_id="request-0",
                        chunk_index=0,
                        num_frames=4,
                    )
                )
                completion_accepted = decode_message(await output.recv())
                assert completion_accepted["type"] == ("media_chunk_complete_accepted")
                assert completion_accepted["chunk_index"] == 0

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{denoiser_port}/v1/realtime_video/generate"
        )
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            lingbot2_vae_fingerprint="taew2_1",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            trace_query=_TraceQuery(),
            release_grace_s=0.05,
        )

        async with serve(denoiser, "127.0.0.1", denoiser_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:
                url = (
                    f"ws://127.0.0.1:{gateway_port}/v1/realtime_video/generate"
                    "?user_id=user-a&trace_id=trace-a"
                )
                async with connect(url, max_size=None, compression=None) as browser:

                    async def send_actions_until_closed():
                        await browser.send(
                            encode_message(
                                "init",
                                max_chunks=1,
                            )
                        )
                        await browser.send(
                            encode_message(
                                "camera_actions",
                                event_id=1,
                                actions=["w"],
                            )
                        )
                        await browser.wait_closed()

                    action_task = asyncio.create_task(send_actions_until_closed())
                    messages = []
                    try:
                        while True:
                            messages.append(
                                decode_message(
                                    await asyncio.wait_for(browser.recv(), 4)
                                )
                            )
                    except ConnectionClosedOK:
                        pass
                    await action_task
                assert {message["type"] for message in messages} == {
                    "session_ready",
                    "frame_batch",
                    "media_chunk_complete",
                }
                frame_batches = [
                    message for message in messages if message["type"] == "frame_batch"
                ]
                assert [message["frame_batch_index"] for message in frame_batches] == [
                    0,
                    1,
                    2,
                    3,
                ]
                assert frame_batches[-1]["is_final_frame_batch"] is False
                media_complete = next(
                    message
                    for message in messages
                    if message["type"] == "media_chunk_complete"
                )
                assert media_complete["chunk_index"] == 0
                assert all("trace" not in message["type"] for message in messages)

                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"http://127.0.0.1:{gateway_port}"
                        "/v1/realtime_video/traces/trace-a"
                    )
                assert response.status_code == 200
                assert response.json()["events"][0]["event"] == ("gateway.ws_accepted")
            finally:
                server.should_exit = True
                await server_task

        assert len(coordinator.admitted) == 1
        assert coordinator.admitted[0]["wait_for_capacity"] is True
        assert len(coordinator.released) == 1
        assert denoiser_close_started_at is not None
        assert coordinator.released_at - denoiser_close_started_at >= 0.045

    asyncio.run(run())


@pytest.mark.parametrize("capacity_rejected", [False, True])
def test_duplicate_world_attempt_never_owns_or_releases_the_owner_session(
    monkeypatch, capacity_rejected
):
    class Callbacks:
        instances = []

        def __init__(self, _config):
            self.calls = []
            self.instances.append(self)

        def started(self, run_id, attempt_id, trace_id, max_lifetime_s):
            self.calls.append(
                (
                    "started",
                    run_id,
                    attempt_id,
                    trace_id,
                    max_lifetime_s,
                )
            )

        def ended(self, run_id, attempt_id, trace_id, reason):
            self.calls.append(("ended", run_id, attempt_id, trace_id, reason))

        def recorded(self, run_id, attempt_id, video_uri):
            self.calls.append(("recorded", run_id, attempt_id, video_uri))

        def aborted(
            self,
            run_id,
            attempt_id,
            trace_id,
            fault,
            reason,
            established,
        ):
            self.calls.append(
                (
                    "aborted",
                    run_id,
                    attempt_id,
                    trace_id,
                    fault,
                    reason,
                    established,
                )
            )

        async def rewrite(self, *_args):
            raise AssertionError("rewrite is not expected")

        async def close(self):
            pass

    class Recorder:
        def __init__(self):
            self.closed = 0

        def session_closed(self):
            self.closed += 1

    class RecordingManager:
        def __init__(self):
            self.starts = []
            self.recorders = []

        def start(self, **kwargs):
            self.starts.append(kwargs)
            recorder = Recorder()
            self.recorders.append(recorder)
            return recorder

        async def close(self):
            pass

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server.WorldCallbacks",
        Callbacks,
    )

    async def run():
        gateway_ports = [_free_port(), _free_port()]
        denoiser_port = _free_port()

        async def denoiser(connection):
            await connection.send(encode_message("session_ready"))
            await connection.wait_closed()

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{denoiser_port}/v1/realtime_video/generate",
            admit_rejection="CAPACITY_EXHAUSTED" if capacity_rejected else None,
        )
        recording_managers = [RecordingManager(), RecordingManager()]
        signing_key = Ed25519PrivateKey.generate()
        run_id = "11111111-1111-1111-1111-111111111111"
        attempt_id = "22222222-2222-2222-2222-222222222222"
        token = _world_token(
            signing_key,
            run_id=run_id,
            attempt_id=attempt_id,
            record_video=True,
        )
        world_platform = WorldPlatformConfig(
            public_key=signing_key.public_key(),
            callback_url="http://world-service.invalid",
            callback_app_id="zing-gateway",
            callback_key_id="k1",
            callback_secret="secret",
            recording_bucket="run-videos",
            recording_region="cn-beijing",
        )
        apps = [
            create_app(
                coordinator,
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
                internal_output_url=(
                    f"ws://127.0.0.1:{port}/v1/internal/realtime_output"
                ),
                world_platform=world_platform,
                recording_manager=recording_manager,
                release_grace_s=0,
            )
            for port, recording_manager in zip(
                gateway_ports, recording_managers, strict=True
            )
        ]

        async with serve(denoiser, "127.0.0.1", denoiser_port):
            running = [
                await _run_gateway(app, port)
                for app, port in zip(apps, gateway_ports, strict=True)
            ]
            try:
                urls = [
                    (
                        f"ws://127.0.0.1:{port}"
                        "/backends/minwm/v1/realtime_video/authorized_generate"
                        f"?token={quote(token, safe='')}&trace_id=trace-{index}"
                    )
                    for index, port in enumerate(gateway_ports)
                ]
                connections = await asyncio.gather(
                    *(connect(url, max_size=None, compression=None) for url in urls)
                )

                async def initial_result(connection):
                    try:
                        wire = await asyncio.wait_for(connection.recv(), timeout=2)
                    except ConnectionClosed:
                        return (
                            "closed",
                            connection.close_code,
                            connection.close_reason,
                        )
                    message = decode_message(wire)
                    if message["type"] == "error":
                        return ("error", message["code"])
                    return ("ready", message["type"])

                results = await asyncio.gather(
                    *(initial_result(connection) for connection in connections)
                )
                owner_index = next(
                    index
                    for index, result in enumerate(results)
                    if result
                    == (
                        ("error", "CAPACITY_EXHAUSTED")
                        if capacity_rejected
                        else ("ready", "session_ready")
                    )
                )
                duplicate_index = 1 - owner_index
                owner = connections[owner_index]
                duplicate = connections[duplicate_index]
                try:
                    if capacity_rejected:
                        await owner.wait_closed()
                        assert owner.close_code == 1013
                        assert owner.close_reason == "CAPACITY_EXHAUSTED"
                    for _ in range(200):
                        if any(callbacks.calls for callbacks in Callbacks.instances):
                            break
                        await asyncio.sleep(0.005)
                    callback_calls = [
                        call
                        for callbacks in Callbacks.instances
                        for call in callbacks.calls
                    ]
                    assert [call[0] for call in callback_calls] == [
                        "aborted" if capacity_rejected else "started"
                    ]

                    assert results[duplicate_index] == (
                        "closed",
                        1008,
                        "UNAUTHORIZED",
                    )
                    assert duplicate.close_code == 1008
                    assert duplicate.close_reason == "UNAUTHORIZED"

                    # 两个 app 各自持有独立 replay cache；只有共享 Coordinator
                    # 的原子 attempt fence 能保证跨 Gateway 副本仍只有一个 owner。
                    assert len(coordinator.attempt_claims) == 2
                    assert len(coordinator.attempt_owners) == 1
                    assert len(coordinator.admitted) == 1
                    assert len(coordinator.released) == 0
                    recording_starts = [
                        start
                        for manager in recording_managers
                        for start in manager.starts
                    ]
                    assert len(recording_starts) == (0 if capacity_rejected else 1)
                    if recording_starts:
                        assert recording_starts[0]["attempt_id"] == attempt_id
                finally:
                    await owner.close()

                if capacity_rejected:
                    assert coordinator.released == []
                    assert callback_calls[0][2] == attempt_id
                    assert callback_calls[0][-1] is False
                    return

                for _ in range(200):
                    callback_calls = [
                        call
                        for callbacks in Callbacks.instances
                        for call in callbacks.calls
                    ]
                    if coordinator.released and len(callback_calls) >= 2:
                        break
                    await asyncio.sleep(0.005)
                assert len(coordinator.released) == 1
                recorders = [
                    recorder
                    for manager in recording_managers
                    for recorder in manager.recorders
                ]
                assert len(recorders) == 1
                assert recorders[0].closed == 1
                assert [call[0] for call in callback_calls] == ["started", "ended"]
                assert all(call[2] == attempt_id for call in callback_calls)
            finally:
                for server, _ in running:
                    server.should_exit = True
                await asyncio.gather(*(task for _, task in running))

    asyncio.run(run())


def test_gateway_routes_named_lingbot2_websocket_through_coordinator():
    async def run():
        gateway_port = _free_port()
        lingbot2_port = _free_port()
        received = []

        async def lingbot2_denoiser(connection):
            query = parse_qs(urlsplit(connection.request.path).query)
            await connection.send(encode_message("session_ready"))
            received.append(
                {
                    "query": query,
                    "payload": await connection.recv(),
                }
            )
            await connection.close(code=1000)

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{lingbot2_port}/v1/realtime_video/generate"
        )
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            lingbot2_vae_fingerprint="taew2_1",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            lingbot2_model_revision="lingbot2-r1",
            release_grace_s=0,
        )

        async with serve(lingbot2_denoiser, "127.0.0.1", lingbot2_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:
                url = (
                    f"ws://127.0.0.1:{gateway_port}"
                    "/backends/lingbot2/v1/realtime_video/generate"
                    "?user_id=user-a&trace_id=trace-a"
                )
                payload = encode_message("camera_actions", event_id=7, actions=["w"])
                async with connect(url, max_size=None, compression=None) as browser:
                    ready = decode_message(await browser.recv())
                    assert ready["type"] == "session_ready", ready
                    await browser.send(payload)
                    await browser.wait_closed()
                assert coordinator.admitted[0]["model_revision"] == "lingbot2-r1"
                assert coordinator.admitted[0]["vae_fingerprint"] == "taew2_1"
                assert received[0]["query"]["trace_id"] == ["trace-a"]
                assert received[0]["query"]["realtime_vae_worker_url"] == [
                    "ws://vae-1:18081/v1/realtime_vae/decode"
                ]
                assert decode_message(received[0]["payload"])["event_id"] == 7
            finally:
                server.should_exit = True
                await server_task

        assert len(coordinator.released) == 1

    asyncio.run(run())
