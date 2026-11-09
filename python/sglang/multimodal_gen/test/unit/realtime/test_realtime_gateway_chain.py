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
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK
from websockets.frames import Close

from sglang.multimodal_gen.runtime.entrypoints import realtime_gateway_server
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
    role: str | None = None,
    allow_free_prompt: bool = False,
    expires_in_s: int = 90,
    play_seconds: int = 52,
    max_lifetime_s: int = 90,
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
        "exp": now + expires_in_s,
        "session": {
            "play_seconds": play_seconds,
            "max_lifetime_s": max_lifetime_s,
            "allow_free_prompt": allow_free_prompt,
            "record_video": record_video,
        },
    }
    if role is not None:
        claims["session"]["role"] = role
    signing = (
        _b64url(json.dumps({"alg": "EdDSA", "typ": "JWT"}).encode())
        + "."
        + _b64url(json.dumps(claims).encode())
    )
    return signing + "." + _b64url(key.sign(signing.encode()))


def _sealed_world_init(
    shared_secret: str,
    associated_data: str,
    *,
    playback_ack_enabled: bool = False,
) -> bytes:
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"world-service/session-payload/v1",
    ).derive(shared_secret.encode())
    nonce = b"comparison00"
    fields: dict[str, object] = {"prompt": "a quiet world"}
    if playback_ack_enabled:
        fields["playback_ack_enabled"] = True
    plaintext = encode_message("init", **fields)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, associated_data.encode())


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
            **_diagnostics,
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

    # With mixed versions, a 4xx from an old Coordinator cannot prove that
    # another new replica failed to acquire ownership; a network error can also
    # happen after the conditional write commits. A connection that did not
    # successfully claim can never mutate the business attempt.
    assert callback_instances[0].calls == []
    assert coordinator.admitted == []
    assert coordinator.released == []


def test_authorized_routes_reject_missing_invalid_expired_and_wrong_role_tokens():
    class RecordingManager:
        def start(self, **_kwargs):
            raise AssertionError("recording is not expected")

        async def close(self):
            pass

    signing_key = Ed25519PrivateKey.generate()
    run_id = "11111111-1111-1111-1111-111111111111"
    primary = _world_token(
        signing_key,
        run_id=run_id,
        attempt_id="22222222-2222-2222-2222-222222222222",
        record_video=False,
    )
    comparison = _world_token(
        signing_key,
        run_id=run_id,
        attempt_id="33333333-3333-3333-3333-333333333333",
        record_video=False,
        role="comparison",
    )
    expired_comparison = _world_token(
        signing_key,
        run_id=run_id,
        attempt_id="44444444-4444-4444-4444-444444444444",
        record_video=False,
        role="comparison",
        expires_in_s=-90,
    )
    coordinator = _Coordinator("ws://denoiser.invalid/generate")
    app = create_app(
        coordinator,
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        lingbot2_model_revision="lingbot2-r1",
        lingbot2_vae_fingerprint="taew2_1",
        internal_output_url="ws://gateway.invalid/v1/internal/realtime_output",
        world_platform=WorldPlatformConfig(
            public_key=signing_key.public_key(),
            callback_url="http://world-service.invalid",
            callback_app_id="zing-gateway",
            callback_key_id="k1",
            callback_secret="secret",
            recording_bucket="run-videos",
            recording_region="cn-shanghai",
        ),
        recording_manager=RecordingManager(),
        release_grace_s=0,
    )

    cases = [
        ("/backends/lingbot2/v1/realtime_video/authorized_generate", None),
        ("/backends/lingbot2/v1/realtime_video/authorized_generate", "bad-token"),
        (
            "/backends/lingbot2/v1/realtime_video/authorized_generate",
            expired_comparison,
        ),
        ("/backends/lingbot2/v1/realtime_video/authorized_generate", primary),
        ("/backends/minwm/v1/realtime_video/authorized_generate", comparison),
    ]
    with TestClient(app) as client:
        for path, token in cases:
            url = path if token is None else f"{path}?token={quote(token, safe='')}"
            with client.websocket_connect(url) as websocket:
                with pytest.raises(WebSocketDisconnect) as closed:
                    websocket.receive_bytes()
                assert closed.value.code == 1008
                assert closed.value.reason == "UNAUTHORIZED"

    assert coordinator.attempt_claims == []
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


def test_authorized_handler_cancellation_is_re_raised_and_gateway_owned(monkeypatch):
    callback_instances = []

    class CancelledCoordinator(_Coordinator):
        async def admit(self, **request):
            self.admitted.append(request)
            raise asyncio.CancelledError

    class Callbacks:
        def __init__(self, _config):
            self.calls = []
            callback_instances.append(self)

        def aborted(
            self,
            _run_id,
            _attempt_id,
            _trace_id,
            fault,
            reason,
            established,
            **diagnostics,
        ):
            self.calls.append((fault, reason, established, diagnostics))

        async def close(self):
            pass

    class RecordingManager:
        def start(self, **_kwargs):
            raise AssertionError("recording cannot start before admission")

        async def close(self):
            pass

    class WebSocket:
        def __init__(self, token):
            self.query_params = {"token": token, "trace_id": "trace-cancelled"}
            self.accepted = False
            self.closes = []

        async def accept(self):
            self.accepted = True

        async def close(self, *, code, reason):
            self.closes.append((code, reason))

    monkeypatch.setattr(realtime_gateway_server, "WorldCallbacks", Callbacks)
    signing_key = Ed25519PrivateKey.generate()
    run_id = "10101010-1010-1010-1010-101010101010"
    attempt_id = "20202020-2020-2020-2020-202020202020"
    token = _world_token(
        signing_key,
        run_id=run_id,
        attempt_id=attempt_id,
        record_video=False,
    )
    coordinator = CancelledCoordinator("ws://denoiser.invalid/generate")
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
            recording_region="cn-shanghai",
        ),
        recording_manager=RecordingManager(),
        release_grace_s=0,
    )
    endpoint = next(
        route.endpoint
        for route in app.routes
        if route.path == "/backends/minwm/v1/realtime_video/authorized_generate"
    )
    websocket = WebSocket(token)

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await endpoint(websocket)

    asyncio.run(run())

    assert websocket.accepted
    assert callback_instances[0].calls == [
        ("ours", "GATEWAY_SESSION_FAILED", False, {})
    ]
    assert websocket.closes == [(1000, "")]
    assert len(coordinator.attempt_claims) == 1
    assert len(coordinator.admitted) == 1
    assert coordinator.released == []


def test_authorized_routes_reject_cross_role_payload_swaps(monkeypatch):
    callback_instances = []

    class Callbacks:
        def __init__(self, _config):
            self.started_calls = []
            self.aborted_calls = []
            callback_instances.append(self)

        def started(self, run_id, attempt_id, trace_id, max_lifetime_s, admitted_at_ms):
            self.started_calls.append(
                {
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "trace_id": trace_id,
                    "max_lifetime_s": max_lifetime_s,
                }
            )

        def ended(self, *_args):
            raise AssertionError("a rejected payload must not end a run")

        def recorded(self, *_args):
            raise AssertionError("a rejected payload must not record a run")

        def aborted(
            self,
            run_id,
            attempt_id,
            trace_id,
            fault,
            reason,
            established,
            **_diagnostics,
        ):
            self.aborted_calls.append(
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
            raise AssertionError("a rejected payload must not be rewritten")

        async def close(self):
            pass

    class RecordingManager:
        def start(self, **_kwargs):
            raise AssertionError("a rejected payload must not create a recorder")

        async def close(self):
            pass

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server.WorldCallbacks",
        Callbacks,
    )

    async def run():
        gateway_port = _free_port()
        denoiser_port = _free_port()

        async def denoiser(connection):
            await connection.send(encode_message("session_ready"))
            await connection.wait_closed()

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{denoiser_port}/v1/realtime_video/generate"
        )
        signing_key = Ed25519PrivateKey.generate()
        run_id = "11111111-1111-1111-1111-111111111111"
        primary_attempt = "22222222-2222-2222-2222-222222222222"
        comparison_attempt = "33333333-3333-3333-3333-333333333333"
        primary_token = _world_token(
            signing_key,
            run_id=run_id,
            attempt_id=primary_attempt,
            record_video=False,
        )
        comparison_token = _world_token(
            signing_key,
            run_id=run_id,
            attempt_id=comparison_attempt,
            record_video=False,
            role="comparison",
        )
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            lingbot2_model_revision="lingbot2-r1",
            lingbot2_vae_fingerprint="taew2_1",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            world_platform=WorldPlatformConfig(
                public_key=signing_key.public_key(),
                callback_url="http://world-service.invalid",
                callback_app_id="zing-gateway",
                callback_key_id="k1",
                callback_secret="secret",
                recording_bucket="run-videos",
                recording_region="cn-shanghai",
            ),
            recording_manager=RecordingManager(),
            release_grace_s=0,
        )
        cases = [
            (
                "primary route with comparison payload",
                "/backends/minwm/v1/realtime_video/authorized_generate",
                primary_token,
                _sealed_world_init("secret", f"{run_id}:comparison"),
            ),
            (
                "comparison route with primary payload",
                "/backends/lingbot2/v1/realtime_video/authorized_generate",
                comparison_token,
                _sealed_world_init("secret", run_id),
            ),
        ]

        async with serve(denoiser, "127.0.0.1", denoiser_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:
                for index, (name, path, token, payload) in enumerate(cases):
                    url = (
                        f"ws://127.0.0.1:{gateway_port}{path}"
                        f"?token={quote(token, safe='')}&trace_id=trace-swap-{index}"
                    )
                    async with connect(url, max_size=None, compression=None) as browser:
                        ready = decode_message(
                            await asyncio.wait_for(browser.recv(), timeout=2)
                        )
                        assert ready["type"] == "session_ready", name
                        await browser.send(payload)
                        error = decode_message(
                            await asyncio.wait_for(browser.recv(), timeout=2)
                        )
                        assert error["type"] == "error", name
                        assert error["code"] == "SESSION_INVALID", name
                        await asyncio.wait_for(browser.wait_closed(), timeout=2)
                        assert browser.close_code == 1008, name
                        assert browser.close_reason == "SESSION_INVALID", name

                for _ in range(200):
                    if (
                        len(coordinator.released) == len(cases)
                        and callback_instances[0].aborted_calls
                    ):
                        break
                    await asyncio.sleep(0.005)
            finally:
                server.should_exit = True
                await server_task

        assert len(coordinator.attempt_claims) == 2
        assert len(coordinator.admitted) == 2
        assert len(coordinator.released) == 2
        # Only the primary route owns business lifecycle. Upstream readiness is
        # reported before the browser's first payload, then the bad payload
        # aborts that established attempt. Comparison remains callback-free.
        assert callback_instances[0].started_calls == [
            {
                "run_id": run_id,
                "attempt_id": primary_attempt,
                "trace_id": "trace-swap-0",
                "max_lifetime_s": 90,
            }
        ]
        assert callback_instances[0].aborted_calls == [
            {
                "run_id": run_id,
                "attempt_id": primary_attempt,
                "trace_id": "trace-swap-0",
                "fault": "client",
                "reason": "SESSION_INVALID",
                "established": True,
            }
        ]

    asyncio.run(run())


@pytest.mark.parametrize("client_leaves", [False, True])
def test_authorized_startup_timeout_and_normal_browser_close_are_aborted(
    monkeypatch, client_leaves
):
    callback_instances = []

    class Callbacks:
        def __init__(self, _config):
            self.calls = []
            callback_instances.append(self)

        def started(self, *_args):
            self.calls.append(("started",))

        def first_frame(self, *_args):
            raise AssertionError("no media should be delivered")

        def ended(self, *_args, **_kwargs):
            raise AssertionError("an incomplete startup cannot end normally")

        def aborted(
            self,
            _run_id,
            _attempt_id,
            _trace_id,
            fault,
            reason,
            established,
            **diagnostics,
        ):
            self.calls.append(("aborted", fault, reason, established, diagnostics))

        def recorded(self, *_args):
            raise AssertionError("recording is disabled")

        async def rewrite(self, *_args):
            raise AssertionError("rewrite is not expected")

        async def close(self):
            pass

    class RecordingManager:
        def start(self, **_kwargs):
            raise AssertionError("recording is disabled")

        async def close(self):
            pass

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server.WorldCallbacks",
        Callbacks,
    )

    async def run():
        gateway_port = _free_port()
        denoiser_port = _free_port()

        async def denoiser(connection):
            await connection.send(encode_message("session_ready"))
            assert decode_message(await connection.recv())["type"] == "init"
            await connection.wait_closed()

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{denoiser_port}/v1/realtime_video/generate"
        )
        signing_key = Ed25519PrivateKey.generate()
        run_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
        token = _world_token(
            signing_key,
            run_id=run_id,
            attempt_id="dddddddd-dddd-dddd-dddd-dddddddddddd",
            record_video=False,
            play_seconds=1,
            max_lifetime_s=7,
        )
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            world_platform=WorldPlatformConfig(
                public_key=signing_key.public_key(),
                callback_url="http://world-service.invalid",
                callback_app_id="zing-gateway",
                callback_key_id="k1",
                callback_secret="secret",
                recording_bucket="run-videos",
                recording_region="cn-shanghai",
            ),
            recording_manager=RecordingManager(),
            release_grace_s=0,
        )

        async with serve(denoiser, "127.0.0.1", denoiser_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:
                url = (
                    f"ws://127.0.0.1:{gateway_port}"
                    "/backends/minwm/v1/realtime_video/authorized_generate"
                    f"?token={quote(token, safe='')}&trace_id=trace-startup"
                )
                browser = await connect(url, max_size=None, compression=None)
                try:
                    assert decode_message(await browser.recv())["type"] == (
                        "session_ready"
                    )
                    await browser.send(_sealed_world_init("secret", run_id))
                    if client_leaves:
                        await browser.close(code=1000, reason="user left")
                    else:
                        error = decode_message(
                            await asyncio.wait_for(browser.recv(), timeout=2)
                        )
                        assert error["type"] == "error"
                        assert error["code"] == "GATEWAY_SESSION_FAILED"
                        await browser.wait_closed()
                        assert browser.close_code == 1011
                finally:
                    await browser.close()

                for _ in range(200):
                    if callback_instances and any(
                        call[0] == "aborted" for call in callback_instances[0].calls
                    ):
                        break
                    await asyncio.sleep(0.01)
            finally:
                server.should_exit = True
                await server_task

        assert len(coordinator.released) == 1
        assert callback_instances[0].calls[0] == ("started",)
        aborted = callback_instances[0].calls[1]
        assert aborted[:4] == (
            ("aborted", "client", "USER_LEFT", True)
            if client_leaves
            else ("aborted", "ours", "STARTUP_TIMEOUT", True)
        )
        if client_leaves:
            assert aborted[4]["disconnect_kind"] == "close_frame"
            assert aborted[4]["disconnect_phase"] == "before_first_frame"
            assert aborted[4]["websocket_close_code"] == 1000
            assert isinstance(aborted[4]["disconnected_at_ms"], int)
        else:
            assert aborted[4] == {}

    asyncio.run(run())


def test_authorized_upstream_close_before_finish_is_gateway_failure(monkeypatch):
    callback_instances = []

    class Callbacks:
        def __init__(self, _config):
            self.calls = []
            callback_instances.append(self)

        def started(self, *_args):
            self.calls.append(("started",))

        def first_frame(self, *_args):
            raise AssertionError("no media should be delivered")

        def ended(self, *_args, **_kwargs):
            raise AssertionError("an upstream failure cannot end normally")

        def aborted(
            self,
            _run_id,
            _attempt_id,
            _trace_id,
            fault,
            reason,
            established,
            **_diagnostics,
        ):
            self.calls.append(("aborted", fault, reason, established))

        def recorded(self, *_args):
            raise AssertionError("recording is disabled")

        async def rewrite(self, *_args):
            raise AssertionError("rewrite is not expected")

        async def close(self):
            pass

    class RecordingManager:
        def start(self, **_kwargs):
            raise AssertionError("recording is disabled")

        async def close(self):
            pass

    class EarlyClosedUpstream:
        def __init__(self):
            self.ready_sent = False
            self.sent_messages = []
            self.closed = False

        async def recv(self):
            if not self.ready_sent:
                self.ready_sent = True
                return encode_message("session_ready")
            await asyncio.Future()

        async def send(self, wire):
            message = decode_message(wire)
            self.sent_messages.append(message)
            if message.get("type") == "init":
                return
            raise ConnectionClosedOK(Close(1000, "generation complete"), None)

        async def close(self):
            self.closed = True

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server.WorldCallbacks",
        Callbacks,
    )

    async def run():
        gateway_port = _free_port()
        upstream = EarlyClosedUpstream()

        async def connect_upstream(*_args, **_kwargs):
            return upstream

        coordinator = _Coordinator("ws://denoiser.invalid/v1/realtime_video/generate")
        signing_key = Ed25519PrivateKey.generate()
        run_id = "12121212-1212-1212-1212-121212121212"
        token = _world_token(
            signing_key,
            run_id=run_id,
            attempt_id="34343434-3434-3434-3434-343434343434",
            record_video=False,
            play_seconds=1,
            max_lifetime_s=7,
        )
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            connect_factory=connect_upstream,
            world_platform=WorldPlatformConfig(
                public_key=signing_key.public_key(),
                callback_url="http://world-service.invalid",
                callback_app_id="zing-gateway",
                callback_key_id="k1",
                callback_secret="secret",
                recording_bucket="run-videos",
                recording_region="cn-shanghai",
            ),
            recording_manager=RecordingManager(),
            release_grace_s=0,
        )

        server, server_task = await _run_gateway(app, gateway_port)
        try:
            url = (
                f"ws://127.0.0.1:{gateway_port}"
                "/backends/minwm/v1/realtime_video/authorized_generate"
                f"?token={quote(token, safe='')}&trace_id=trace-upstream-close"
            )
            async with connect(url, max_size=None, compression=None) as browser:
                ready = decode_message(await asyncio.wait_for(browser.recv(), 2))
                assert ready["type"] == "session_ready"
                await browser.send(_sealed_world_init("secret", run_id))
                await browser.send(
                    encode_message("event", kind="heartbeat", payload={})
                )
                error = decode_message(
                    await asyncio.wait_for(browser.recv(), timeout=2)
                )
                assert error["type"] == "error"
                assert error["code"] == "GATEWAY_SESSION_FAILED"
                await browser.wait_closed()
                assert browser.close_code == 1011

            for _ in range(200):
                if callback_instances and any(
                    call[0] == "aborted" for call in callback_instances[0].calls
                ):
                    break
                await asyncio.sleep(0.01)
        finally:
            server.should_exit = True
            await server_task

        assert [message["type"] for message in upstream.sent_messages] == [
            "init",
            "event",
        ]
        assert upstream.closed is True
        assert len(coordinator.released) == 1
        assert callback_instances[0].calls == [
            ("started",),
            ("aborted", "ours", "GATEWAY_SESSION_FAILED", True),
        ]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("mode", "expected_reason"),
    [
        ("missing_final", "MEDIA_FINAL_MISSING"),
        ("detached_missing_final", "MEDIA_FINAL_MISSING"),
        ("stuck_worker", "FINISH_TIMEOUT"),
        ("hard_wall", "MAX_LIFETIME_REACHED"),
        ("early_final", "GATEWAY_SESSION_FAILED"),
    ],
)
def test_authorized_finish_boundary_emits_one_abort(monkeypatch, mode, expected_reason):
    callback_instances = []

    class Callbacks:
        def __init__(self, _config):
            self.calls = []
            callback_instances.append(self)

        def started(self, *_args):
            self.calls.append(("started",))

        def first_frame(self, *_args):
            self.calls.append(("first_frame",))
            if mode == "hard_wall":
                # Startup already froze the short test prep budget. Expand only
                # the nominal finish tail now, making the previously frozen hard
                # wall the earlier terminal boundary.
                realtime_gateway_server.FINISH_TAIL_SECONDS = 2.0

        def ended(self, *_args, **_kwargs):
            self.calls.append(("ended",))

        def aborted(
            self,
            _run_id,
            _attempt_id,
            _trace_id,
            fault,
            reason,
            established,
            **_diagnostics,
        ):
            self.calls.append(("aborted", fault, reason, established))

        def recorded(self, *_args):
            raise AssertionError("recording is disabled")

        async def rewrite(self, *_args):
            raise AssertionError("rewrite is not expected")

        async def close(self):
            pass

    class Recorder:
        def __init__(self):
            self.messages = []
            self.closed = 0

        def offer(self, message):
            self.messages.append(message)

        def session_closed(self):
            self.closed += 1

    class RecordingManager:
        def __init__(self):
            self.starts = []
            self.recorder = Recorder()

        def start(self, **kwargs):
            if mode != "early_final":
                raise AssertionError("recording is disabled")
            self.starts.append(kwargs)
            return self.recorder

        async def close(self):
            pass

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server.WorldCallbacks",
        Callbacks,
    )
    original_browser_send = realtime_gateway_server._BrowserSender.send

    async def detach_after_first_media(sender, wire):
        if mode == "detached_missing_final" and isinstance(wire, bytes):
            message = decode_message(wire)
            if message.get("type") == "media_chunk_complete":
                raise TimeoutError("injected browser presentation timeout")
        await original_browser_send(sender, wire)

    monkeypatch.setattr(
        realtime_gateway_server._BrowserSender,
        "send",
        detach_after_first_media,
    )
    # Keep the real one-second play clock, but shorten the test's terminal tail
    # so missing/stuck cases remain fast.
    monkeypatch.setattr(realtime_gateway_server, "FINISH_TAIL_SECONDS", 0.2)
    if mode == "hard_wall":
        monkeypatch.setattr(
            "sglang.multimodal_gen.runtime.realtime.world_platform.FINISH_TAIL_SECONDS",
            0.2,
        )

    async def run():
        gateway_port = _free_port()
        denoiser_port = _free_port()
        worker_result = asyncio.get_running_loop().create_future()
        worker_result.add_done_callback(
            lambda future: None if future.cancelled() else future.exception()
        )

        async def denoiser_impl(connection):
            query = parse_qs(urlsplit(connection.request.path).query)
            session_id = query["session_id"][0]
            generation_id = query["generation_id"][0]
            output_url = query["gateway_output_url"][0]
            output_token = query["gateway_output_token"][0]

            await connection.send(encode_message("session_ready"))
            assert decode_message(await connection.recv())["type"] == "init"
            async with connect(output_url, max_size=None, compression=None) as output:
                await output.send(
                    encode_message(
                        "session_output_open",
                        session_id=session_id,
                        generation_id=generation_id,
                        token=output_token,
                    )
                )
                assert decode_message(await output.recv())["type"] == (
                    "session_output_accepted"
                )

                async def send_frame(chunk_index):
                    await output.send(
                        encode_message(
                            "frame_batch",
                            session_id=session_id,
                            generation_id=generation_id,
                            request_id=f"request-{chunk_index}",
                            chunk_index=chunk_index,
                            frame_batch_index=0,
                            payload_lengths=[4],
                            payload=b"webp",
                            content_type="image/webp",
                            width=8,
                            height=8,
                            num_frames=1,
                        )
                    )

                async def send_completion(chunk_index, *, final):
                    await output.send(
                        encode_message(
                            "media_chunk_complete",
                            session_id=session_id,
                            generation_id=generation_id,
                            request_id=f"request-{chunk_index}",
                            chunk_index=chunk_index,
                            num_frames=1,
                            is_final_chunk=final,
                        )
                    )
                    accepted = decode_message(await output.recv())
                    assert accepted["type"] == "media_chunk_complete_accepted"

                await send_frame(0)
                await send_completion(0, final=False)
                if mode == "early_final":
                    await send_frame(1)
                    await send_completion(1, final=True)
                    await connection.wait_closed()
                    return

                finish = decode_message(await connection.recv())
                assert finish["kind"] == "server_finish"

                if mode in {"stuck_worker", "hard_wall"}:
                    # Keep the control connection open without a final. A
                    # validated final is now authoritative completion, so the
                    # timeout cases must genuinely omit it.
                    await connection.wait_closed()
                    return

                await connection.close(code=1000, reason="generation complete")
                await asyncio.sleep(0.25)

        async def denoiser(connection):
            try:
                await denoiser_impl(connection)
            except BaseException as exc:
                if not worker_result.done():
                    worker_result.set_exception(exc)
                raise
            else:
                if not worker_result.done():
                    worker_result.set_result(None)

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{denoiser_port}/v1/realtime_video/generate"
        )
        signing_key = Ed25519PrivateKey.generate()
        run_id = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
        token = _world_token(
            signing_key,
            run_id=run_id,
            attempt_id="ffffffff-ffff-ffff-ffff-ffffffffffff",
            record_video=mode == "early_final",
            play_seconds=1,
            max_lifetime_s=2 if mode == "hard_wall" else 7,
        )
        recording_manager = RecordingManager()
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            world_platform=WorldPlatformConfig(
                public_key=signing_key.public_key(),
                callback_url="http://world-service.invalid",
                callback_app_id="zing-gateway",
                callback_key_id="k1",
                callback_secret="secret",
                recording_bucket="run-videos",
                recording_region="cn-shanghai",
            ),
            recording_manager=recording_manager,
            release_grace_s=0,
        )

        async with serve(denoiser, "127.0.0.1", denoiser_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:
                url = (
                    f"ws://127.0.0.1:{gateway_port}"
                    "/backends/minwm/v1/realtime_video/authorized_generate"
                    f"?token={quote(token, safe='')}&trace_id=trace-finish-{mode}"
                )
                async with connect(url, max_size=None, compression=None) as browser:
                    assert (
                        decode_message(await browser.recv())["type"] == "session_ready"
                    )
                    await browser.send(_sealed_world_init("secret", run_id))
                    messages = []
                    try:
                        while True:
                            messages.append(
                                decode_message(
                                    await asyncio.wait_for(browser.recv(), timeout=3)
                                )
                            )
                    except ConnectionClosed:
                        pass
                    assert browser.close_code == 1011
                    assert any(message["type"] == "error" for message in messages) is (
                        mode != "detached_missing_final"
                    )

                for _ in range(100):
                    if callback_instances and any(
                        call[0] in {"ended", "aborted"}
                        for call in callback_instances[0].calls
                    ):
                        break
                    await asyncio.sleep(0.01)
            finally:
                server.should_exit = True
                await server_task

        await asyncio.wait_for(worker_result, timeout=2)
        terminal_calls = [
            call
            for call in callback_instances[0].calls
            if call[0] in {"ended", "aborted"}
        ]
        assert terminal_calls == [("aborted", "ours", expected_reason, True)]
        assert len(coordinator.released) == 1
        if mode == "early_final":
            assert len(recording_manager.starts) == 1
            assert recording_manager.recorder.closed == 1
            assert not any(
                message.get("is_final_chunk") is True
                for message in recording_manager.recorder.messages
            )
        else:
            assert recording_manager.starts == []

    asyncio.run(run())


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

        def started(self, run_id, attempt_id, trace_id, max_lifetime_s, admitted_at_ms):
            self.calls.append(
                (
                    "started",
                    run_id,
                    attempt_id,
                    trace_id,
                    max_lifetime_s,
                )
            )

        def first_frame(self, run_id, attempt_id, trace_id, first_media_at_ms):
            self.calls.append(
                (
                    "first_frame",
                    run_id,
                    attempt_id,
                    trace_id,
                    first_media_at_ms,
                )
            )

        def ended(self, run_id, attempt_id, trace_id, reason, *, ended_at_ms):
            self.calls.append(
                ("ended", run_id, attempt_id, trace_id, reason, ended_at_ms)
            )

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
            **_diagnostics,
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

                    # Each app has an independent replay cache. Only the shared
                    # Coordinator's atomic attempt fence guarantees one owner
                    # across Gateway replicas.
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
                assert [call[0] for call in callback_calls] == [
                    "started",
                    "aborted",
                ]
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


def test_authorized_lingbot_without_free_prompt_drops_direction_and_is_isolated(
    monkeypatch,
):
    callback_instances = []

    class Callbacks:
        def __init__(self, _config):
            self.calls = []
            callback_instances.append(self)

        def started(self, *_args):
            self.calls.append("started")

        def ended(self, *_args):
            self.calls.append("ended")

        def aborted(self, *_args, **_kwargs):
            self.calls.append("aborted")

        def recorded(self, *_args):
            self.calls.append("recorded")

        async def rewrite(self, *_args):
            raise AssertionError("comparison direction must not be rewritten")

        async def close(self):
            pass

    class RecordingManager:
        def __init__(self):
            self.starts = []

        def start(self, **kwargs):
            self.starts.append(kwargs)
            raise AssertionError("comparison must not create a recorder")

        async def close(self):
            pass

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server.WorldCallbacks",
        Callbacks,
    )

    async def run():
        gateway_port = _free_port()
        lingbot2_port = _free_port()
        received = []
        free_prompt_dropped = asyncio.Event()

        async def lingbot2_denoiser(connection):
            await connection.send(encode_message("session_ready"))
            received.append(decode_message(await connection.recv()))
            try:
                received.append(await asyncio.wait_for(connection.recv(), timeout=0.2))
            except TimeoutError:
                free_prompt_dropped.set()
            await connection.wait_closed()

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{lingbot2_port}/v1/realtime_video/generate"
        )
        signing_key = Ed25519PrivateKey.generate()
        run_id = "11111111-1111-1111-1111-111111111111"
        attempt_id = "33333333-3333-3333-3333-333333333333"
        token = _world_token(
            signing_key,
            run_id=run_id,
            attempt_id=attempt_id,
            record_video=False,
            role="comparison",
        )
        recording_manager = RecordingManager()
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            lingbot2_model_revision="lingbot2-r1",
            lingbot2_vae_fingerprint="taew2_1",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            world_platform=WorldPlatformConfig(
                public_key=signing_key.public_key(),
                callback_url="http://world-service.invalid",
                callback_app_id="zing-gateway",
                callback_key_id="k1",
                callback_secret="secret",
                recording_bucket="run-videos",
                recording_region="cn-shanghai",
            ),
            recording_manager=recording_manager,
            release_grace_s=0,
        )

        async with serve(lingbot2_denoiser, "127.0.0.1", lingbot2_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:
                url = (
                    f"ws://127.0.0.1:{gateway_port}"
                    "/backends/lingbot2/v1/realtime_video/authorized_generate"
                    f"?token={quote(token, safe='')}&trace_id=trace-comparison"
                )
                async with connect(url, max_size=None, compression=None) as browser:
                    assert (
                        decode_message(await browser.recv())["type"] == "session_ready"
                    )
                    await browser.send(
                        _sealed_world_init("secret", f"{run_id}:comparison")
                    )
                    await browser.send(
                        encode_message(
                            "event",
                            kind="direction",
                            text="ignore the comparison policy",
                            event_id=41,
                        )
                    )
                    await asyncio.wait_for(free_prompt_dropped.wait(), timeout=2)

                for _ in range(200):
                    if coordinator.released:
                        break
                    await asyncio.sleep(0.005)

                replay = await connect(url, max_size=None, compression=None)
                try:
                    with pytest.raises(ConnectionClosed):
                        await replay.recv()
                    assert replay.close_code == 1008
                    assert replay.close_reason == "UNAUTHORIZED"
                finally:
                    await replay.close()
            finally:
                server.should_exit = True
                await server_task

        assert len(coordinator.attempt_claims) == 1
        assert len(coordinator.admitted) == 1
        assert coordinator.admitted[0]["user_id"] == "wsvc:legacy:u_player:lingbot2"
        assert coordinator.admitted[0]["model_revision"] == "lingbot2-r1"
        assert coordinator.admitted[0]["vae_fingerprint"] == "taew2_1"
        assert len(coordinator.released) == 1
        assert received == [
            {
                "version": 2,
                "type": "init",
                "prompt": "a quiet world",
            }
        ]
        assert recording_manager.starts == []
        assert callback_instances[0].calls == []

    asyncio.run(run())


def test_authorized_lingbot_free_prompt_dispatches_before_primary_and_is_isolated(
    monkeypatch,
):
    callback_instances = []

    class Callbacks:
        def __init__(self, _config):
            self.lifecycle_calls = []
            self.rewrite_calls = []
            callback_instances.append(self)

        def started(self, *_args):
            self.lifecycle_calls.append("started")

        def ended(self, *_args):
            self.lifecycle_calls.append("ended")

        def aborted(self, *_args, **_kwargs):
            self.lifecycle_calls.append("aborted")

        def recorded(self, *_args):
            self.lifecycle_calls.append("recorded")

        async def rewrite(self, run_id, event_id, role, instruction, baseline):
            self.rewrite_calls.append(
                {
                    "run_id": run_id,
                    "event_id": event_id,
                    "role": role,
                    "instruction": instruction,
                    "baseline": baseline,
                }
            )
            return {
                "prompt": "the same rewritten rainy world",
                "change_type": "persistent",
            }

        async def close(self):
            pass

    class RecordingManager:
        def __init__(self):
            self.starts = []

        def start(self, **kwargs):
            self.starts.append(kwargs)
            raise AssertionError("comparison must not create a recorder")

        async def close(self):
            pass

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server.WorldCallbacks",
        Callbacks,
    )

    async def run():
        gateway_port = _free_port()
        lingbot2_port = _free_port()
        received = []
        direction_dispatched = asyncio.Queue()

        async def lingbot2_denoiser(connection):
            session_received = []
            received.append(session_received)
            await connection.send(encode_message("session_ready"))
            session_received.append(decode_message(await connection.recv()))
            # Comparison sends one valid direction; the primary connection that
            # follows sends one legacy direction and one coordinated direction.
            direction_count = 1 if len(received) == 1 else 2
            for _ in range(direction_count):
                session_received.append(decode_message(await connection.recv()))
                direction_dispatched.put_nowait(None)
            await connection.wait_closed()

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{lingbot2_port}/v1/realtime_video/generate"
        )
        signing_key = Ed25519PrivateKey.generate()
        run_id = "11111111-1111-1111-1111-111111111111"
        attempt_id = "33333333-3333-3333-3333-333333333333"
        comparison_token = _world_token(
            signing_key,
            run_id=run_id,
            attempt_id=attempt_id,
            record_video=False,
            role="comparison",
            allow_free_prompt=True,
        )
        primary_token = _world_token(
            signing_key,
            run_id=run_id,
            attempt_id="22222222-2222-2222-2222-222222222222",
            record_video=False,
            allow_free_prompt=True,
        )
        recording_manager = RecordingManager()
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            lingbot2_model_revision="lingbot2-r1",
            lingbot2_vae_fingerprint="taew2_1",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            world_platform=WorldPlatformConfig(
                public_key=signing_key.public_key(),
                callback_url="http://world-service.invalid",
                callback_app_id="zing-gateway",
                callback_key_id="k1",
                callback_secret="secret",
                recording_bucket="run-videos",
                recording_region="cn-shanghai",
            ),
            recording_manager=recording_manager,
            release_grace_s=0,
        )

        async with serve(lingbot2_denoiser, "127.0.0.1", lingbot2_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:
                url = (
                    f"ws://127.0.0.1:{gateway_port}"
                    "/backends/lingbot2/v1/realtime_video/authorized_generate"
                    f"?token={quote(comparison_token, safe='')}"
                    "&trace_id=trace-comparison"
                )
                # No primary socket exists: comparison must be able to arrive
                # first and wait/reuse through world-service's idempotent API.
                async with connect(url, max_size=None, compression=None) as browser:
                    assert (
                        decode_message(await browser.recv())["type"] == "session_ready"
                    )
                    await browser.send(
                        _sealed_world_init("secret", f"{run_id}:comparison")
                    )
                    # Invalid ids must never reach rewrite. In particular,
                    # Python bool must not pass as int, and values outside the
                    # JavaScript safe-integer range cannot be stable idempotency
                    # keys across the browser/JSON/Go boundary.
                    await browser.send(
                        encode_message(
                            "event",
                            kind="direction",
                            text="drop missing id",
                        )
                    )
                    for invalid_event_id in (True, "42", 0, 1 << 53):
                        await browser.send(
                            encode_message(
                                "event",
                                kind="direction",
                                text="drop invalid id",
                                event_id=invalid_event_id,
                            )
                        )
                    await browser.send(
                        encode_message(
                            "event",
                            kind="direction",
                            text="make it rain",
                            event_id=42,
                        )
                    )
                    await asyncio.wait_for(direction_dispatched.get(), timeout=2)

                assert callback_instances[0].lifecycle_calls == []
                primary_url = (
                    f"ws://127.0.0.1:{gateway_port}"
                    "/backends/minwm/v1/realtime_video/authorized_generate"
                    f"?token={quote(primary_token, safe='')}&trace_id=trace-primary"
                )
                async with connect(
                    primary_url, max_size=None, compression=None
                ) as primary_browser:
                    assert (
                        decode_message(await primary_browser.recv())["type"]
                        == "session_ready"
                    )
                    await primary_browser.send(_sealed_world_init("secret", run_id))
                    # Legacy primary clients are outside this frontend rollout.
                    # Missing event_id must keep dispatching through the old,
                    # non-coordinated rewrite request shape.
                    await primary_browser.send(
                        encode_message(
                            "event",
                            kind="direction",
                            text="legacy primary input",
                        )
                    )
                    await asyncio.wait_for(direction_dispatched.get(), timeout=2)
                    await primary_browser.send(
                        encode_message(
                            "event",
                            kind="direction",
                            text="coordinated primary input",
                            event_id=43,
                        )
                    )
                    await asyncio.wait_for(direction_dispatched.get(), timeout=2)
            finally:
                server.should_exit = True
                await server_task

        assert len(coordinator.attempt_claims) == 2
        assert len(coordinator.admitted) == 2
        assert coordinator.admitted[0]["user_id"] == "wsvc:legacy:u_player:lingbot2"
        assert coordinator.admitted[0]["model_revision"] == "lingbot2-r1"
        assert coordinator.admitted[0]["vae_fingerprint"] == "taew2_1"
        assert coordinator.admitted[1]["user_id"] == "wsvc:legacy:u_player"
        assert coordinator.admitted[1]["model_revision"] == "minwm-r1"
        assert coordinator.admitted[1]["vae_fingerprint"] == "taew2_2"
        assert len(coordinator.released) == 2
        assert received == [
            [
                {
                    "version": 2,
                    "type": "init",
                    "prompt": "a quiet world",
                },
                {
                    "version": 2,
                    "type": "event",
                    "kind": "prompt",
                    "payload": "the same rewritten rainy world",
                    "event_id": 42,
                },
            ],
            [
                {
                    "version": 2,
                    "type": "init",
                    "prompt": "a quiet world",
                },
                {
                    "version": 2,
                    "type": "event",
                    "kind": "prompt",
                    "payload": "the same rewritten rainy world",
                },
                {
                    "version": 2,
                    "type": "event",
                    "kind": "prompt",
                    "payload": "the same rewritten rainy world",
                    "event_id": 43,
                },
            ],
        ]
        assert callback_instances[0].rewrite_calls == [
            {
                "run_id": run_id,
                "event_id": 42,
                "role": "comparison",
                "instruction": "make it rain",
                "baseline": "a quiet world",
            },
            {
                "run_id": run_id,
                "event_id": None,
                "role": "primary",
                "instruction": "legacy primary input",
                "baseline": "a quiet world",
            },
            {
                "run_id": run_id,
                "event_id": 43,
                "role": "primary",
                "instruction": "coordinated primary input",
                "baseline": "the same rewritten rainy world",
            },
        ]
        assert callback_instances[0].lifecycle_calls == ["started", "aborted"]
        assert recording_manager.starts == []

    asyncio.run(run())


@pytest.mark.parametrize(
    (
        "fail_final_browser_send",
        "zero_business_input",
        "play_seconds",
        "cancel_after_completed",
    ),
    [
        pytest.param(False, False, 2, False, id="business-input-dynamic-duration"),
        pytest.param(True, False, 1, False, id="business-input-final-send-failure"),
        pytest.param(False, True, 1, False, id="zero-business-input-success"),
        pytest.param(
            False,
            True,
            1,
            True,
            id="handler-cancel-after-validated-completion",
        ),
    ],
)
def test_authorized_wall_clock_finish_uses_validated_final_independent_of_delivery(
    monkeypatch,
    fail_final_browser_send,
    zero_business_input,
    play_seconds,
    cancel_after_completed,
):
    callback_instances = []
    fence_events = {}
    finish_tail_traces = []
    browser_detach_traces = []
    rewrite_calls = []
    original_log_gateway_trace = realtime_gateway_server._log_gateway_trace

    def observe_gateway_trace(trace_id, event, **fields):
        original_log_gateway_trace(trace_id, event, **fields)
        if event == "gateway.finish_tail_upstream_closed":
            finish_tail_traces.append((trace_id, fields))
            fence_events["finish_tail_upstream_closed"].set()
        if event == "gateway.browser_delivery_detached":
            browser_detach_traces.append((trace_id, fields))

    monkeypatch.setattr(
        realtime_gateway_server,
        "_log_gateway_trace",
        observe_gateway_trace,
    )

    class Callbacks:
        def __init__(self, _config):
            self.calls = []
            callback_instances.append(self)

        def started(self, run_id, attempt_id, trace_id, max_lifetime_s, admitted_at_ms):
            self.calls.append(
                (
                    "started",
                    run_id,
                    attempt_id,
                    trace_id,
                    max_lifetime_s,
                    admitted_at_ms,
                )
            )

        def first_frame(self, run_id, attempt_id, trace_id, first_media_at_ms):
            self.calls.append(
                (
                    "first_frame",
                    run_id,
                    attempt_id,
                    trace_id,
                    first_media_at_ms,
                )
            )
            fence_events["first_media"].set()

        def ended(self, run_id, attempt_id, trace_id, reason, *, ended_at_ms):
            self.calls.append(
                ("ended", run_id, attempt_id, trace_id, reason, ended_at_ms)
            )
            if cancel_after_completed:
                asyncio.current_task().cancel()

        def aborted(
            self,
            run_id,
            attempt_id,
            trace_id,
            fault,
            reason,
            established,
            **_diagnostics,
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

        def recorded(self, *_args):
            raise AssertionError("recording is disabled")

        async def rewrite(self, *_args):
            rewrite_calls.append(_args)
            if zero_business_input:
                raise AssertionError("zero-input sessions must not request a rewrite")
            fence_events["rewrite_started"].set()
            await fence_events["absolute_deadline"].wait()
            return {
                "prompt": "this rewrite result is too late",
                "change_type": "persistent",
            }

        async def close(self):
            pass

    class RecordingManager:
        def start(self, **_kwargs):
            raise AssertionError("recording is disabled")

        async def close(self):
            pass

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server.WorldCallbacks",
        Callbacks,
    )
    if fail_final_browser_send:
        original_send = realtime_gateway_server._BrowserSender.send

        async def fail_final_send(sender, wire):
            if isinstance(wire, bytes):
                message = decode_message(wire)
                if (
                    isinstance(message, dict)
                    and message.get("type") == "media_chunk_complete"
                    and message.get("is_final_chunk") is True
                ):
                    raise RuntimeError("injected final browser send failure")
            await original_send(sender, wire)

        monkeypatch.setattr(
            realtime_gateway_server._BrowserSender,
            "send",
            fail_final_send,
        )

    async def run():
        gateway_port = _free_port()
        denoiser_port = _free_port()
        worker_result = asyncio.get_running_loop().create_future()
        worker_result.add_done_callback(
            lambda future: None if future.cancelled() else future.exception()
        )
        finish_received = asyncio.Event()
        finish_tail_transport_received = asyncio.Event()
        worker_closed = asyncio.Event()
        pre_finish_transport_received = asyncio.Event()
        allow_arbiter_finish = asyncio.Event()
        release_started = asyncio.Event()
        allow_release = asyncio.Event()
        fence_events.update(
            first_media=asyncio.Event(),
            rewrite_started=asyncio.Event(),
            absolute_deadline=asyncio.Event(),
            finish_tail_upstream_closed=asyncio.Event(),
        )
        worker_messages = []
        worker_closed_at = None
        final_sent_at = None

        original_sleep_until = realtime_gateway_server._sleep_until

        async def hold_play_timer_after_absolute_deadline(deadline):
            await original_sleep_until(deadline)
            if asyncio.current_task().get_name() == "gateway-play-deadline":
                fence_events["absolute_deadline"].set()
                await allow_arbiter_finish.wait()

        monkeypatch.setattr(
            realtime_gateway_server,
            "_sleep_until",
            hold_play_timer_after_absolute_deadline,
        )

        async def denoiser_impl(connection):
            nonlocal worker_closed_at, final_sent_at
            query = parse_qs(urlsplit(connection.request.path).query)
            session_id = query["session_id"][0]
            generation_id = query["generation_id"][0]
            output_url = query["gateway_output_url"][0]
            output_token = query["gateway_output_token"][0]

            await connection.send(encode_message("session_ready"))
            init = decode_message(await connection.recv())
            assert init["type"] == "init"
            assert init["playback_ack_enabled"] is True

            async with connect(output_url, max_size=None, compression=None) as output:
                await output.send(
                    encode_message(
                        "session_output_open",
                        session_id=session_id,
                        generation_id=generation_id,
                        token=output_token,
                    )
                )
                assert decode_message(await output.recv())["type"] == (
                    "session_output_accepted"
                )

                async def send_frame(chunk_index: int):
                    await output.send(
                        encode_message(
                            "frame_batch",
                            session_id=session_id,
                            generation_id=generation_id,
                            request_id=f"request-{chunk_index}",
                            chunk_index=chunk_index,
                            frame_batch_index=0,
                            payload_lengths=[4],
                            payload=b"webp",
                            content_type="image/webp",
                            width=8,
                            height=8,
                            num_frames=1,
                        )
                    )

                async def send_completion(chunk_index: int, *, final: bool):
                    await output.send(
                        encode_message(
                            "media_chunk_complete",
                            session_id=session_id,
                            generation_id=generation_id,
                            request_id=f"request-{chunk_index}",
                            chunk_index=chunk_index,
                            num_frames=1,
                            is_final_chunk=final,
                        )
                    )
                    accepted = decode_message(await output.recv())
                    assert accepted["type"] == "media_chunk_complete_accepted"
                    assert accepted["chunk_index"] == chunk_index

                await send_frame(0)
                await send_completion(0, final=False)

                # The play deadline is already past, but the play timer remains
                # intentionally blocked so the main arbiter has not yet set
                # world_session["finishing"]. Only tail transport may cross this
                # absolute boundary. Business-input cases additionally prove
                # that the post-deadline camera event and pre-deadline rewrite
                # result are fenced; the zero-input case sends transport only.
                for _ in range(2):
                    worker_messages.append(decode_message(await connection.recv()))
                assert [message["kind"] for message in worker_messages] == [
                    "playback_ack",
                    "heartbeat",
                ]
                pre_finish_transport_received.set()

                finish = decode_message(await connection.recv())
                worker_messages.append(finish)
                assert finish == {
                    "version": 2,
                    "type": "event",
                    "kind": "server_finish",
                    "payload": {"reason": "play_duration_elapsed"},
                }
                finish_received.set()
                for _ in range(2):
                    worker_messages.append(decode_message(await connection.recv()))
                assert [message["kind"] for message in worker_messages[3:]] == [
                    "playback_ack",
                    "heartbeat",
                ]
                finish_tail_transport_received.set()
                await connection.close(code=1000, reason="generation complete")
                worker_closed_at = time.monotonic()
                worker_closed.set()
                await asyncio.wait_for(
                    fence_events["finish_tail_upstream_closed"].wait(), timeout=1
                )
                # The worker control path is allowed to finish before the VAE
                # emits its post-EOF final marker.
                await send_frame(1)
                await send_completion(1, final=True)
                final_sent_at = time.monotonic()

        async def denoiser(connection):
            try:
                await denoiser_impl(connection)
            except BaseException as exc:
                if not worker_result.done():
                    worker_result.set_exception(exc)
                raise
            else:
                if not worker_result.done():
                    worker_result.set_result(None)

        class BlockingReleaseCoordinator(_Coordinator):
            async def release(self, assignment):
                release_started.set()
                await allow_release.wait()
                await super().release(assignment)

        coordinator = BlockingReleaseCoordinator(
            f"ws://127.0.0.1:{denoiser_port}/v1/realtime_video/generate"
        )
        signing_key = Ed25519PrivateKey.generate()
        run_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        attempt_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        token = _world_token(
            signing_key,
            run_id=run_id,
            attempt_id=attempt_id,
            record_video=False,
            allow_free_prompt=True,
            play_seconds=play_seconds,
            max_lifetime_s=play_seconds + 6,
        )
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            world_platform=WorldPlatformConfig(
                public_key=signing_key.public_key(),
                callback_url="http://world-service.invalid",
                callback_app_id="zing-gateway",
                callback_key_id="k1",
                callback_secret="secret",
                recording_bucket="run-videos",
                recording_region="cn-shanghai",
            ),
            recording_manager=RecordingManager(),
            release_grace_s=0,
        )

        async with serve(denoiser, "127.0.0.1", denoiser_port):
            server, server_task = await _run_gateway(app, gateway_port)
            release_probe = None
            try:
                url = (
                    f"ws://127.0.0.1:{gateway_port}"
                    "/backends/minwm/v1/realtime_video/authorized_generate"
                    f"?token={quote(token, safe='')}&trace_id=trace-wall-clock"
                )
                async with connect(url, max_size=None, compression=None) as browser:
                    ready = decode_message(await asyncio.wait_for(browser.recv(), 2))
                    assert ready["type"] == "session_ready"
                    await browser.send(
                        _sealed_world_init(
                            "secret",
                            run_id,
                            playback_ack_enabled=True,
                        )
                    )

                    async def exercise_session_inputs():
                        await fence_events["first_media"].wait()
                        if not zero_business_input:
                            await browser.send(
                                encode_message(
                                    "event",
                                    kind="direction",
                                    text="rewrite before the deadline",
                                    event_id=88,
                                )
                            )
                            await fence_events["rewrite_started"].wait()
                        await fence_events["absolute_deadline"].wait()
                        if not zero_business_input:
                            await browser.send(
                                encode_message(
                                    "event",
                                    kind="camera_actions",
                                    payload={"forward": 1.0},
                                    event_id=89,
                                )
                            )
                        await browser.send(
                            encode_message(
                                "event",
                                kind="playback_ack",
                                payload={
                                    "last_received_chunk": 0,
                                    "last_rendered_chunk": 0,
                                },
                            )
                        )
                        await browser.send(
                            encode_message("event", kind="heartbeat", payload={})
                        )
                        await pre_finish_transport_received.wait()
                        allow_arbiter_finish.set()
                        await finish_received.wait()
                        await browser.send(
                            encode_message(
                                "event",
                                kind="playback_ack",
                                payload={
                                    "last_received_chunk": 0,
                                    "last_rendered_chunk": 0,
                                },
                            )
                        )
                        await browser.send(
                            encode_message("event", kind="heartbeat", payload={})
                        )
                        await finish_tail_transport_received.wait()
                        await worker_closed.wait()
                        await browser.send(
                            encode_message(
                                "event",
                                kind="playback_ack",
                                payload={
                                    "last_received_chunk": 0,
                                    "last_rendered_chunk": 0,
                                },
                            )
                        )
                        await browser.send(
                            encode_message("event", kind="heartbeat", payload={})
                        )

                    ack_task = asyncio.create_task(exercise_session_inputs())

                    async def verify_terminal_before_release():
                        await asyncio.wait_for(
                            release_started.wait(), timeout=play_seconds + 2
                        )
                        try:
                            terminal_calls = [
                                call
                                for call in callback_instances[0].calls
                                if call[0] in {"ended", "aborted"}
                            ]
                            assert len(terminal_calls) == 1
                            assert terminal_calls[0][0] == "ended"
                            assert coordinator.released == []
                        finally:
                            allow_release.set()

                    release_probe = asyncio.create_task(
                        verify_terminal_before_release()
                    )
                    browser_messages = []
                    try:
                        while True:
                            browser_messages.append(
                                decode_message(
                                    await asyncio.wait_for(browser.recv(), timeout=4)
                                )
                            )
                    except ConnectionClosed:
                        pass
                    await ack_task
                    close_code = browser.close_code

                await release_probe

                for _ in range(200):
                    if callback_instances and any(
                        call[0] in {"ended", "aborted"}
                        for call in callback_instances[0].calls
                    ):
                        break
                    await asyncio.sleep(0.01)
            finally:
                allow_release.set()
                if release_probe is not None:
                    if not release_probe.done():
                        release_probe.cancel()
                    await asyncio.gather(release_probe, return_exceptions=True)
                server.should_exit = True
                await server_task

        await asyncio.wait_for(worker_result, timeout=2)
        assert worker_closed_at is not None
        assert final_sent_at is not None
        assert worker_closed_at < final_sent_at
        assert (
            sum(message.get("kind") == "server_finish" for message in worker_messages)
            == 1
        )
        assert fence_events["finish_tail_upstream_closed"].is_set()
        assert len(finish_tail_traces) == 1
        assert finish_tail_traces[0][0] == "trace-wall-clock"
        assert finish_tail_traces[0][1]["stage"] == "finishing"
        assert finish_tail_traces[0][1]["message_kind"] == "playback_ack"
        assert len(rewrite_calls) == (0 if zero_business_input else 1)
        callback_calls = callback_instances[0].calls
        first_frame = next(call for call in callback_calls if call[0] == "first_frame")
        assert callback_calls[0][0] == "started"
        assert sum(call[0] in {"ended", "aborted"} for call in callback_calls) == 1
        assert len(coordinator.released) == 1

        assert close_code == 1000
        ended = next(call for call in callback_calls if call[0] == "ended")
        assert ended[4] == "completed"
        assert ended[5] - first_frame[4] == play_seconds * 1_000
        assert not any(call[0] == "aborted" for call in callback_calls)
        if fail_final_browser_send:
            assert not any(
                message["type"] == "media_chunk_complete"
                and message.get("is_final_chunk") is True
                for message in browser_messages
            )
            assert len(browser_detach_traces) == 1
            assert browser_detach_traces[0][0] == "trace-wall-clock"
            assert browser_detach_traces[0][1]["detach_reason"] == (
                "browser_send_RuntimeError"
            )
            assert browser_detach_traces[0][1]["detach_class"] == "send_error"
            assert browser_detach_traces[0][1]["detach_phase"] == "finishing"
        else:
            assert browser_detach_traces == []
            final_message = next(
                message
                for message in browser_messages
                if message["type"] == "media_chunk_complete"
                and message.get("is_final_chunk") is True
            )
            assert final_message["chunk_index"] == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    ("role", "business_input"),
    [
        pytest.param("primary", False, id="primary-zero-input"),
        pytest.param("primary", True, id="primary-business-input"),
        pytest.param("comparison", False, id="comparison-zero-input"),
    ],
)
def test_authorized_h264_send_timeout_keeps_source_complete(
    monkeypatch, role, business_input
):
    callback_instances = []
    events = {}

    class Callbacks:
        def __init__(self, _config):
            self.calls = []
            callback_instances.append(self)

        def started(self, *_args):
            self.calls.append(("started",))

        def first_frame(self, *_args):
            self.calls.append(("first_frame",))

        def ended(self, _run_id, _attempt_id, _trace_id, reason, **_kwargs):
            self.calls.append(("ended", reason))

        def aborted(self, *_args, **_kwargs):
            self.calls.append(("aborted",))

        def recorded(self, *_args):
            self.calls.append(("recorded",))

        async def rewrite(self, *_args):
            raise AssertionError("rewrite is not expected")

        async def close(self):
            pass

    class Recorder:
        def __init__(self):
            self.messages = []
            self.closed = 0

        def offer(self, message):
            self.messages.append(message)

        def session_closed(self):
            self.closed += 1

    class RecordingManager:
        def __init__(self):
            self.starts = []
            self.recorder = Recorder()

        def start(self, **kwargs):
            self.starts.append(kwargs)
            return self.recorder

        async def close(self):
            pass

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server.WorldCallbacks",
        Callbacks,
    )
    original_send = realtime_gateway_server._BrowserSender.send

    async def block_second_h264_payload(sender, wire):
        message = None
        if isinstance(wire, bytes):
            message = decode_message(wire)
            if message.get("type") == "media_payload" and message.get("sequence") == 1:
                events["send_blocked"].set()
                try:
                    await events["never_release"].wait()
                except asyncio.CancelledError:
                    events["send_timed_out"].set()
                    raise
        await original_send(sender, wire)
        if (
            isinstance(message, dict)
            and message.get("type") == "media_payload"
            and message.get("sequence") == 0
        ):
            events["first_media"].set()

    monkeypatch.setattr(
        realtime_gateway_server._BrowserSender,
        "send",
        block_second_h264_payload,
    )
    original_close = realtime_gateway_server.WebSocket.close

    async def observe_detached_close(websocket, *args, **kwargs):
        if (
            "authorized_generate" in websocket.url.path
            and websocket.query_params.get("trace_id") == "trace-h264-timeout"
        ):
            events["close_calls"] += 1
            events["close_entered"].set()
            await events["allow_close"].wait()
        await original_close(websocket, *args, **kwargs)

    monkeypatch.setattr(
        realtime_gateway_server.WebSocket,
        "close",
        observe_detached_close,
    )

    async def run():
        gateway_port = _free_port()
        denoiser_port = _free_port()
        worker_result = asyncio.get_running_loop().create_future()
        worker_result.add_done_callback(
            lambda future: None if future.cancelled() else future.exception()
        )
        events.update(
            first_media=asyncio.Event(),
            send_blocked=asyncio.Event(),
            send_timed_out=asyncio.Event(),
            never_release=asyncio.Event(),
            final_accepted=asyncio.Event(),
            close_entered=asyncio.Event(),
            allow_close=asyncio.Event(),
            close_calls=0,
        )
        final_accepted_at = None
        worker_closed_at = None

        async def denoiser(connection):
            nonlocal final_accepted_at, worker_closed_at
            try:
                query = parse_qs(urlsplit(connection.request.path).query)
                session_id = query["session_id"][0]
                generation_id = query["generation_id"][0]
                output_url = query["gateway_output_url"][0]
                output_token = query["gateway_output_token"][0]

                await connection.send(encode_message("session_ready"))
                assert decode_message(await connection.recv())["type"] == "init"
                async with connect(
                    output_url, max_size=None, compression=None
                ) as output:
                    await output.send(
                        encode_message(
                            "session_output_open",
                            session_id=session_id,
                            generation_id=generation_id,
                            token=output_token,
                        )
                    )
                    assert decode_message(await output.recv())["type"] == (
                        "session_output_accepted"
                    )

                    async def send_media(message_type, **fields):
                        await output.send(
                            encode_message(
                                message_type,
                                session_id=session_id,
                                generation_id=generation_id,
                                **fields,
                            )
                        )

                    await send_media("media_init", codec="h264", width=8, height=8)
                    await send_media(
                        "media_payload",
                        sequence=0,
                        codec="h264",
                        payload=b"first-media",
                    )
                    await asyncio.wait_for(events["first_media"].wait(), timeout=2)
                    await send_media(
                        "media_payload",
                        sequence=1,
                        codec="h264",
                        payload=b"blocked-media",
                    )
                    await asyncio.wait_for(events["send_blocked"].wait(), timeout=2)
                    await asyncio.wait_for(events["send_timed_out"].wait(), timeout=2)

                    if business_input:
                        camera = decode_message(await connection.recv())
                        assert camera["kind"] == "camera_actions"
                        assert camera["event_id"] == 91
                    finish = decode_message(await connection.recv())
                    assert finish == {
                        "version": 2,
                        "type": "event",
                        "kind": "server_finish",
                        "payload": {"reason": "play_duration_elapsed"},
                    }

                    # More messages than the old route limit prove that the
                    # source path no longer inherits browser backpressure.
                    for sequence in range(2, 8):
                        await send_media(
                            "media_payload",
                            sequence=sequence,
                            codec="h264",
                            payload=f"tail-{sequence}".encode(),
                        )
                    await send_media(
                        "media_chunk_complete",
                        request_id="request-final",
                        chunk_index=0,
                        num_frames=8,
                        media_transport="h264",
                        is_final_chunk=True,
                    )
                    accepted = decode_message(await output.recv())
                    assert accepted["type"] == "media_chunk_complete_accepted"
                    assert accepted["chunk_index"] == 0
                    final_accepted_at = time.monotonic()
                    events["final_accepted"].set()

                    # Keep the worker control connection open. Gateway must
                    # complete from final and close this only during cleanup.
                    await connection.wait_closed()
                    worker_closed_at = time.monotonic()
            except BaseException as exc:
                if not worker_result.done():
                    worker_result.set_exception(exc)
                raise
            else:
                if not worker_result.done():
                    worker_result.set_result(None)

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{denoiser_port}/v1/realtime_video/generate"
        )
        signing_key = Ed25519PrivateKey.generate()
        run_id = "12121212-1212-1212-1212-121212121212"
        token = _world_token(
            signing_key,
            run_id=run_id,
            attempt_id="34343434-3434-3434-3434-343434343434",
            record_video=role == "primary",
            role=None if role == "primary" else "comparison",
            play_seconds=1,
            max_lifetime_s=7,
        )
        recording_manager = RecordingManager()
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            world_platform=WorldPlatformConfig(
                public_key=signing_key.public_key(),
                callback_url="http://world-service.invalid",
                callback_app_id="zing-gateway",
                callback_key_id="k1",
                callback_secret="secret",
                recording_bucket="run-videos",
                recording_region="cn-shanghai",
            ),
            recording_manager=recording_manager,
            output_queue_max_messages=2,
            browser_send_timeout_s=0.05,
            release_grace_s=0,
        )

        async with serve(denoiser, "127.0.0.1", denoiser_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:

                async def release_close_after_resource_cleanup():
                    await asyncio.wait_for(events["close_entered"].wait(), timeout=3)
                    try:
                        # A detached peer gets its one full close only after the
                        # source route and Coordinator lease are released.
                        assert len(coordinator.released) == 1
                    finally:
                        events["allow_close"].set()

                close_probe = asyncio.create_task(
                    release_close_after_resource_cleanup()
                )
                url = (
                    f"ws://127.0.0.1:{gateway_port}"
                    + (
                        "/backends/minwm/v1/realtime_video/authorized_generate"
                        if role == "primary"
                        else "/backends/lingbot2/v1/realtime_video/authorized_generate"
                    )
                    + f"?token={quote(token, safe='')}&trace_id=trace-h264-timeout"
                )
                async with connect(url, max_size=None, compression=None) as browser:
                    assert decode_message(await browser.recv())["type"] == (
                        "session_ready"
                    )
                    await browser.send(
                        _sealed_world_init(
                            "secret",
                            run_id if role == "primary" else f"{run_id}:comparison",
                        )
                    )

                    async def send_business_input_after_first_media():
                        await asyncio.wait_for(events["first_media"].wait(), timeout=2)
                        if business_input:
                            await browser.send(
                                encode_message(
                                    "event",
                                    kind="camera_actions",
                                    payload={"forward": 1.0},
                                    event_id=91,
                                )
                            )

                    input_task = asyncio.create_task(
                        send_business_input_after_first_media()
                    )
                    browser_messages = []
                    try:
                        while True:
                            browser_messages.append(
                                decode_message(
                                    await asyncio.wait_for(browser.recv(), timeout=3)
                                )
                            )
                    except ConnectionClosed:
                        pass
                    await input_task
                    assert browser.close_code == 1000
                await close_probe

                await asyncio.wait_for(events["final_accepted"].wait(), timeout=2)
                for _ in range(200):
                    if coordinator.released:
                        break
                    await asyncio.sleep(0.005)
            finally:
                server.should_exit = True
                await server_task

        await asyncio.wait_for(worker_result, timeout=2)
        assert final_accepted_at is not None
        assert worker_closed_at is not None
        assert final_accepted_at < worker_closed_at
        assert len(coordinator.released) == 1
        assert events["close_calls"] == 1
        if role == "primary":
            assert len(recording_manager.starts) == 1
            assert recording_manager.recorder.closed == 1
            assert [
                message["type"] for message in recording_manager.recorder.messages
            ] == [
                "media_init",
                *("media_payload" for _ in range(8)),
                "media_chunk_complete",
            ]
            assert [call[0] for call in callback_instances[0].calls] == [
                "started",
                "first_frame",
                "ended",
            ]
            assert callback_instances[0].calls[-1] == ("ended", "completed")
        else:
            assert recording_manager.starts == []
            assert recording_manager.recorder.messages == []
            assert recording_manager.recorder.closed == 0
            assert callback_instances[0].calls == []
        assert not any(
            message.get("type") == "media_chunk_complete"
            for message in browser_messages
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    ("detach_before_disconnect", "disconnect_mode"),
    [
        pytest.param(False, "close", id="close-frame"),
        pytest.param(True, "close", id="output-detach-then-close-frame"),
        pytest.param(False, "abort", id="connection-lost"),
    ],
)
def test_authorized_browser_disconnect_after_first_media_is_user_left(
    monkeypatch, detach_before_disconnect, disconnect_mode
):
    callback_instances = []
    presentation = {}
    transport_metrics = {"first_media": [], "input": [], "output": []}
    terminal_traces = []
    session_closed_traces = []

    class Callbacks:
        def __init__(self, _config):
            self.calls = []
            callback_instances.append(self)

        def started(self, *_args):
            self.calls.append(("started",))

        def first_frame(self, *_args):
            self.calls.append(("first_frame",))

        def ended(self, *_args, **_kwargs):
            self.calls.append(("ended",))

        def aborted(
            self,
            _run_id,
            _attempt_id,
            _trace_id,
            fault,
            reason,
            established,
            **diagnostics,
        ):
            self.calls.append(("aborted", fault, reason, established, diagnostics))

        def recorded(self, *_args):
            raise AssertionError("recording is disabled")

        async def rewrite(self, *_args):
            raise AssertionError("rewrite is not expected")

        async def close(self):
            pass

    class RecordingManager:
        def start(self, **_kwargs):
            raise AssertionError("recording is disabled")

        async def close(self):
            pass

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server.WorldCallbacks",
        Callbacks,
    )
    original_log_gateway_trace = realtime_gateway_server._log_gateway_trace

    def observe_presentation_detach(trace_id, event, **fields):
        original_log_gateway_trace(trace_id, event, **fields)
        if event == "gateway.world_session_terminal":
            terminal_traces.append((trace_id, fields))
        if event == "gateway.session_closed":
            session_closed_traces.append((trace_id, fields))
        if (
            detach_before_disconnect
            and trace_id == "trace-user-left"
            and event == "gateway.browser_delivery_detached"
        ):
            presentation["detached"].set()

    monkeypatch.setattr(
        realtime_gateway_server,
        "_log_gateway_trace",
        observe_presentation_detach,
    )
    monkeypatch.setattr(
        realtime_gateway_server,
        "observe_world_gateway_first_media",
        lambda **fields: transport_metrics["first_media"].append(fields) or True,
    )
    monkeypatch.setattr(
        realtime_gateway_server,
        "observe_world_gateway_browser_input_disconnect",
        lambda **fields: transport_metrics["input"].append(fields) or True,
    )
    monkeypatch.setattr(
        realtime_gateway_server,
        "observe_world_gateway_browser_delivery_detach",
        lambda **fields: transport_metrics["output"].append(fields) or True,
    )
    original_browser_send = realtime_gateway_server._BrowserSender.send

    async def optionally_detach_presentation(sender, wire):
        if detach_before_disconnect and isinstance(wire, bytes):
            message = decode_message(wire)
            if message.get("type") == "media_chunk_complete":
                raise TimeoutError("injected browser presentation timeout")
        await original_browser_send(sender, wire)

    monkeypatch.setattr(
        realtime_gateway_server._BrowserSender,
        "send",
        optionally_detach_presentation,
    )

    async def run():
        gateway_port = _free_port()
        denoiser_port = _free_port()
        worker_result = asyncio.get_running_loop().create_future()
        worker_result.add_done_callback(
            lambda future: None if future.cancelled() else future.exception()
        )
        presentation["detached"] = asyncio.Event()

        async def denoiser(connection):
            try:
                query = parse_qs(urlsplit(connection.request.path).query)
                session_id = query["session_id"][0]
                generation_id = query["generation_id"][0]
                output_url = query["gateway_output_url"][0]
                output_token = query["gateway_output_token"][0]

                await connection.send(encode_message("session_ready"))
                assert decode_message(await connection.recv())["type"] == "init"
                async with connect(
                    output_url, max_size=None, compression=None
                ) as output:
                    await output.send(
                        encode_message(
                            "session_output_open",
                            session_id=session_id,
                            generation_id=generation_id,
                            token=output_token,
                        )
                    )
                    assert decode_message(await output.recv())["type"] == (
                        "session_output_accepted"
                    )
                    await output.send(
                        encode_message(
                            "frame_batch",
                            session_id=session_id,
                            generation_id=generation_id,
                            request_id="request-0",
                            chunk_index=0,
                            frame_batch_index=0,
                            payload_lengths=[4],
                            payload=b"webp",
                            content_type="image/webp",
                            width=8,
                            height=8,
                            num_frames=1,
                        )
                    )
                    await output.send(
                        encode_message(
                            "media_chunk_complete",
                            session_id=session_id,
                            generation_id=generation_id,
                            request_id="request-0",
                            chunk_index=0,
                            num_frames=1,
                            is_final_chunk=False,
                        )
                    )
                    assert decode_message(await output.recv())["type"] == (
                        "media_chunk_complete_accepted"
                    )
                    await connection.wait_closed()
            except BaseException as exc:
                if not worker_result.done():
                    worker_result.set_exception(exc)
                raise
            else:
                if not worker_result.done():
                    worker_result.set_result(None)

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{denoiser_port}/v1/realtime_video/generate"
        )
        signing_key = Ed25519PrivateKey.generate()
        run_id = "56565656-5656-5656-5656-565656565656"
        token = _world_token(
            signing_key,
            run_id=run_id,
            attempt_id="78787878-7878-7878-7878-787878787878",
            record_video=False,
            play_seconds=1,
            max_lifetime_s=7,
        )
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            world_platform=WorldPlatformConfig(
                public_key=signing_key.public_key(),
                callback_url="http://world-service.invalid",
                callback_app_id="zing-gateway",
                callback_key_id="k1",
                callback_secret="secret",
                recording_bucket="run-videos",
                recording_region="cn-shanghai",
            ),
            recording_manager=RecordingManager(),
            release_grace_s=0,
        )

        async with serve(denoiser, "127.0.0.1", denoiser_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:
                url = (
                    f"ws://127.0.0.1:{gateway_port}"
                    "/backends/minwm/v1/realtime_video/authorized_generate"
                    f"?token={quote(token, safe='')}&trace_id=trace-user-left"
                )
                browser = await connect(url, max_size=None, compression=None)
                try:
                    assert decode_message(await browser.recv())["type"] == (
                        "session_ready"
                    )
                    await browser.send(_sealed_world_init("secret", run_id))
                    while True:
                        message = decode_message(
                            await asyncio.wait_for(browser.recv(), timeout=2)
                        )
                        if message["type"] == "frame_batch":
                            break
                    if detach_before_disconnect:
                        await asyncio.wait_for(
                            presentation["detached"].wait(), timeout=2
                        )
                    if disconnect_mode == "close":
                        await browser.close(code=1000, reason="user left")
                    else:
                        browser.transport.abort()
                        await browser.wait_closed()
                finally:
                    await browser.close()

                for _ in range(200):
                    if coordinator.released:
                        break
                    await asyncio.sleep(0.005)
            finally:
                server.should_exit = True
                await server_task

        await asyncio.wait_for(worker_result, timeout=2)
        assert len(coordinator.released) == 1
        assert callback_instances[0].calls[:2] == [
            ("started",),
            ("first_frame",),
        ]
        aborted = callback_instances[0].calls[2]
        assert aborted[:4] == ("aborted", "client", "USER_LEFT", True)
        diagnostics = aborted[4]
        assert diagnostics["disconnect_phase"] == "playing"
        assert diagnostics["disconnect_kind"] == (
            "close_frame" if disconnect_mode == "close" else "connection_lost"
        )
        assert diagnostics["websocket_close_code"] == (
            1000 if disconnect_mode == "close" else 1005
        )
        assert isinstance(diagnostics["disconnected_at_ms"], int)
        assert len(transport_metrics["first_media"]) == 1
        assert len(transport_metrics["input"]) == 1
        assert transport_metrics["input"][0]["phase"] == "playing"
        assert transport_metrics["input"][0]["kind"] == diagnostics["disconnect_kind"]
        assert len(transport_metrics["output"]) == (
            1 if detach_before_disconnect else 0
        )
        assert len(terminal_traces) == 1
        assert terminal_traces[0][1]["outcome"] == "aborted"
        assert (
            terminal_traces[0][1]["disconnect_kind"] == diagnostics["disconnect_kind"]
        )
        assert len(session_closed_traces) == 1
        assert session_closed_traces[0][1]["run_id"] == run_id
        assert session_closed_traces[0][1]["outcome"] == "aborted"
        assert session_closed_traces[0][1]["browser_output_detached"] is (
            detach_before_disconnect
        )

    asyncio.run(run())
