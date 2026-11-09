# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import inspect
import json
import sys

import httpx
import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints import realtime_gateway_server
from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
    _WORLD_ERROR_MESSAGES,
    HTTPCoordinatorClient,
    _browser_send_trace_fields,
    _build_world_platform,
    _build_world_platform_profiles,
    _load_world_platform_profiles,
    _parse_args,
    _parse_ui_config,
    _world_coordinator_attempt_id,
    _world_coordinator_user_id,
    _world_error,
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
from sglang.multimodal_gen.runtime.realtime.world_platform import (
    Principal,
    WorldPlatformConfig,
    WorldPlatformProfiles,
    load_public_key,
)


class _Coordinator:
    def __init__(self, *, ready=True, capacity=None):
        self.ready = ready
        self.capacity_requests = []
        self._capacity = capacity or {
            "observed_at": 1.0,
            "roles": {
                "denoiser": {
                    "waiting_sessions": 0,
                    "active_sessions": 0,
                    "queued_sessions": 0,
                    "free_slots": 1,
                    "draining_workers": 0,
                },
                "vae": {
                    "waiting_sessions": 0,
                    "active_sessions": 0,
                    "queued_sessions": 0,
                    "free_slots": 1,
                    "draining_workers": 0,
                },
            },
        }

    async def health(self):
        if not self.ready:
            raise RuntimeError("coordinator unavailable")
        return {"status": "ready"}

    async def capacity(self, *, model_revision=None, vae_fingerprint=None):
        self.capacity_requests.append(
            {"model_revision": model_revision, "vae_fingerprint": vae_fingerprint}
        )
        if isinstance(self._capacity, Exception):
            raise self._capacity
        return self._capacity


class _TraceQuery:
    def __init__(self):
        self.calls = []

    async def query(self, trace_id, **kwargs):
        self.calls.append((trace_id, kwargs))
        return {
            "trace_id": trace_id,
            "events": [{"event": "server.chunk_complete", "trace_seq": 9}],
            "next_cursor": 9,
        }


def test_gateway_parses_ui_config_for_the_served_webui():
    assert _parse_ui_config(
        '{"generationModes":["i2v","t2v"],"defaultGenerationMode":"t2v"}'
    ) == {
        "generationModes": ["i2v", "t2v"],
        "defaultGenerationMode": "t2v",
    }


def test_gateway_exposes_named_minwm_and_lingbot2_backends():
    app = create_app(
        _Coordinator(),
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        internal_output_url="ws://gateway/v1/internal/realtime_output",
        lingbot2_upstream_url="ws://lingbot2:30000/v1/realtime_video/generate",
        lingbot2_model_revision="lingbot2-r1",
    )

    routes = {route.path for route in app.routes}
    assert "/backends/minwm/v1/realtime_video/generate" in routes
    assert "/backends/lingbot2/v1/realtime_video/generate" in routes
    assert "/backends/minwm/v1/models" in routes
    assert "/backends/lingbot2/v1/models" in routes

    client = TestClient(app)
    assert client.get("/backends/minwm/v1/models").json()["data"][0]["id"] == (
        "minwm-r1"
    )
    assert client.get("/backends/lingbot2/v1/models").json()["data"][0]["id"] == (
        "lingbot2-r1"
    )


def test_world_sessions_use_independent_role_scoped_subleases():
    common = {
        "user_id": "s_per_run",
        "account_id": "u_player",
        "run_id": "run-a",
        "play_seconds": 52,
        "max_lifetime_s": 90,
        "allow_free_prompt": False,
        "record_video": False,
        "jti": "22222222-2222-2222-2222-222222222222",
    }

    primary = Principal(**common)
    comparison = Principal(**common, role="comparison")

    assert _world_coordinator_user_id(primary) == "wsvc:legacy:u_player"
    assert _world_coordinator_user_id(comparison) == "wsvc:legacy:u_player:lingbot2"
    assert (
        _world_coordinator_attempt_id(primary)
        == "legacy:22222222-2222-2222-2222-222222222222"
    )


def test_gateway_trace_events_use_the_independent_otlp_log_plane():
    source = inspect.getsource(realtime_gateway_server)

    assert "emit_realtime_trace_payload" in source
    assert "gateway.browser_send_complete" in source
    assert "gateway_browser_send_slow" in source
    assert 'logger.info(\n        "realtime_trace %s"' not in source


def test_gateway_browser_send_trace_fields_keep_media_metadata_without_payload():
    fields = _browser_send_trace_fields(
        encode_message(
            "frame_batch",
            session_id="session-a",
            generation_id="generation-a",
            request_id="request-a",
            chunk_index=3,
            frame_batch_index=2,
            num_frame_batches=4,
            payload_lengths=[17, 19],
            payload=b"opaque",
            content_type="image/webp",
            width=8,
            height=8,
            num_frames=2,
            is_final_frame_batch=False,
        )
    )

    assert fields["message_type"] == "frame_batch"
    assert fields["chunk_index"] == 3
    assert fields["frame_batch_index"] == 2
    assert fields["num_frames"] == 2
    assert fields["content_type"] == "image/webp"
    assert fields["payload_bytes"] == 36
    assert fields["payload_count"] == 2
    assert fields["wire_bytes"] > fields["payload_bytes"]
    assert "payload" not in fields


@pytest.mark.parametrize(
    ("raw_code", "code", "kind", "close_class"),
    [
        pytest.param(1000, 1000, "close_frame", "normal", id="normal"),
        pytest.param(1001, 1001, "close_frame", "going_away", id="going-away"),
        pytest.param(1006, 1006, "connection_lost", "abnormal", id="abnormal"),
        pytest.param(1015, 1015, "connection_lost", "abnormal", id="tls-local"),
        pytest.param(1008, 1008, "protocol_close", "protocol", id="policy"),
        pytest.param(1011, 1011, "close_frame", "service", id="service"),
        pytest.param(4001, 4001, "close_frame", "application", id="application"),
        pytest.param(None, 1005, "connection_lost", "abnormal", id="missing"),
        pytest.param(True, 1005, "connection_lost", "abnormal", id="bool"),
        pytest.param("1000", 1005, "connection_lost", "abnormal", id="string"),
        pytest.param(999, 1005, "connection_lost", "abnormal", id="too-low"),
        pytest.param(5000, 1005, "connection_lost", "abnormal", id="too-high"),
    ],
)
def test_browser_input_close_classification_is_stable(
    raw_code, code, kind, close_class
):
    normalized = realtime_gateway_server._normalize_websocket_close_code(raw_code)

    assert normalized == code
    assert realtime_gateway_server._classify_browser_input_close(normalized) == kind
    assert (
        realtime_gateway_server._classify_browser_input_close_metric(normalized)
        == close_class
    )


def test_receive_browser_treats_a_missing_close_code_as_no_close_frame():
    class WebSocket:
        async def receive(self):
            return {"type": "websocket.disconnect"}

    async def run():
        with pytest.raises(WebSocketDisconnect) as disconnected:
            await realtime_gateway_server._receive_browser(WebSocket())
        assert disconnected.value.code == 1005

    asyncio.run(run())


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("browser_send_TimeoutError", "send_timeout"),
        ("browser_send_RuntimeError", "send_error"),
        ("queue_message_limit", "queue_message_limit"),
        ("queue_byte_limit", "queue_byte_limit"),
        ("finish_deadline", "finish_deadline"),
        ("output_detached", "other"),
    ],
)
def test_browser_output_detach_classification_is_stable(reason, expected):
    assert realtime_gateway_server._classify_browser_output_detach(reason) == expected


def test_disconnect_facts_freeze_their_original_phase_independently():
    state = {"first_media_monotonic": 5.0, "play_deadline_monotonic": 15.0}
    input_fact = realtime_gateway_server._BrowserInputDisconnectFact(
        kind="connection_lost",
        phase=realtime_gateway_server._world_disconnect_phase(state, 10.0),
        close_code=1006,
        disconnected_at_ms=1_000,
        observed_monotonic=10.0,
    )
    output_fact = realtime_gateway_server._BrowserOutputDetachFact(
        detach_class="send_timeout",
        phase=realtime_gateway_server._world_disconnect_phase(state, 11.0),
        detached_at_ms=1_100,
        observed_monotonic=11.0,
    )

    assert (
        realtime_gateway_server._record_browser_input_disconnect(state, input_fact)
        is input_fact
    )
    assert (
        realtime_gateway_server._record_browser_output_detach(state, output_fact)
        is output_fact
    )

    state["finishing"] = True
    later_input = realtime_gateway_server._BrowserInputDisconnectFact(
        kind="close_frame",
        phase="finishing",
        close_code=1000,
        disconnected_at_ms=2_000,
        observed_monotonic=16.0,
    )
    later_output = realtime_gateway_server._BrowserOutputDetachFact(
        detach_class="finish_deadline",
        phase="finishing",
        detached_at_ms=2_000,
        observed_monotonic=16.0,
    )

    assert (
        realtime_gateway_server._record_browser_input_disconnect(state, later_input)
        is input_fact
    )
    assert (
        realtime_gateway_server._record_browser_output_detach(state, later_output)
        is output_fact
    )
    assert state["browser_input_disconnect_fact"].phase == "playing"
    assert state["browser_output_detach_fact"].phase == "playing"


def test_user_left_requires_a_frozen_browser_input_disconnect_fact():
    missing_input = {"abort_reason": "USER_LEFT", "abort_fault": "client"}
    output_only = {
        "browser_output_detach_fact": (
            realtime_gateway_server._BrowserOutputDetachFact(
                detach_class="send_error",
                phase="playing",
                detached_at_ms=1_000,
                observed_monotonic=10.0,
            )
        )
    }
    input_disconnect = realtime_gateway_server._BrowserInputDisconnectFact(
        kind="close_frame",
        phase="playing",
        close_code=1000,
        disconnected_at_ms=1_000,
        observed_monotonic=10.0,
    )

    missing = realtime_gateway_server._build_world_terminal_fact(
        missing_input,
        assignment_exists=True,
        established=True,
    )
    detached = realtime_gateway_server._build_world_terminal_fact(
        output_only,
        assignment_exists=True,
        established=True,
    )
    user_left = realtime_gateway_server._build_world_terminal_fact(
        {"browser_input_disconnect_fact": input_disconnect},
        assignment_exists=True,
        established=True,
    )

    assert (missing.fault, missing.reason) == ("ours", "GATEWAY_SESSION_FAILED")
    assert missing.classification_error == "user_left_without_input_disconnect"
    assert (detached.fault, detached.reason) == ("ours", "GATEWAY_SESSION_FAILED")
    assert detached.classification_error == "unknown_terminal_state"
    assert (user_left.fault, user_left.reason) == ("client", "USER_LEFT")
    assert user_left.input_disconnect is input_disconnect


def test_handler_cancellation_before_completion_is_gateway_owned():
    state = {}

    assert realtime_gateway_server._record_world_session_cancellation(state)
    terminal = realtime_gateway_server._build_world_terminal_fact(
        state,
        assignment_exists=True,
        established=True,
    )

    assert state["system_error_type"] == "CancelledError"
    assert (terminal.outcome, terminal.fault, terminal.reason) == (
        "aborted",
        "ours",
        "GATEWAY_SESSION_FAILED",
    )


def test_handler_cancellation_after_validated_completion_preserves_completion():
    state = {"dynamic_completed": True, "ended_at_ms": 53_000}

    assert not realtime_gateway_server._record_world_session_cancellation(state)
    terminal = realtime_gateway_server._build_world_terminal_fact(
        state,
        assignment_exists=True,
        established=True,
    )

    assert "abort_reason" not in state
    assert (terminal.outcome, terminal.reason, terminal.ended_at_ms) == (
        "ended",
        "completed",
        53_000,
    )


@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        pytest.param(
            {
                "now": 10.0,
                "hard_deadline": 10.0,
                "nominal_finish_deadline": 10.0,
                "worker_succeeded": True,
                "worker_finished_at": 10.0,
                "final_completed_at": 10.0,
            },
            "completed",
            id="final-wins-over-finish-and-hard-timers-in-the-same-tick",
        ),
        pytest.param(
            {
                "now": 10.0,
                "hard_deadline": 20.0,
                "nominal_finish_deadline": 10.0,
                "worker_succeeded": True,
                "worker_finished_at": 9.9,
                "final_completed_at": 10.001,
            },
            "MEDIA_FINAL_MISSING",
            id="late-final-cannot-rewrite-timeout-as-completed",
        ),
        pytest.param(
            {
                "now": 10.0,
                "hard_deadline": 20.0,
                "nominal_finish_deadline": 10.0,
                "worker_succeeded": True,
                "worker_finished_at": 9.9,
                "final_completed_at": None,
            },
            "MEDIA_FINAL_MISSING",
            id="worker-eof-without-final",
        ),
        pytest.param(
            {
                "now": 10.0,
                "hard_deadline": 20.0,
                "nominal_finish_deadline": 10.0,
                "worker_succeeded": False,
                "worker_finished_at": None,
                "final_completed_at": 9.9,
            },
            "completed",
            id="final-completes-with-worker-control-still-open",
        ),
        pytest.param(
            {
                "now": 10.0,
                "hard_deadline": 10.0,
                "nominal_finish_deadline": 12.0,
                "worker_succeeded": False,
                "worker_finished_at": None,
                "final_completed_at": None,
            },
            "MAX_LIFETIME_REACHED",
            id="overall-hard-wall-wins",
        ),
    ],
)
def test_world_finish_outcome_has_one_deterministic_boundary_result(facts, expected):
    assert (
        realtime_gateway_server._world_finish_outcome(
            finish_deadline=10.0,
            **facts,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("failure_at", "final_completed_at", "expected"),
    [
        pytest.param(9.9, 10.0, True, id="failure-before-final"),
        pytest.param(10.0, 10.0, False, id="final-wins-the-same-tick"),
        pytest.param(10.1, 10.0, False, id="failure-after-final"),
        pytest.param(10.1, None, False, id="failure-after-finish-deadline"),
        pytest.param(9.9, None, True, id="failure-without-final"),
        pytest.param(None, 10.0, True, id="missing-failure-time-is-conservative"),
    ],
)
def test_world_task_failure_uses_the_frozen_final_order(
    failure_at, final_completed_at, expected
):
    assert (
        realtime_gateway_server._world_failure_precedes_final(
            failure_at=failure_at,
            final_completed_at=final_completed_at,
            finish_deadline=10.0,
        )
        is expected
    )


def test_server_finish_timestamp_is_frozen_after_the_upstream_lock_is_acquired():
    class Upstream:
        def __init__(self):
            self.sent = []

        async def send(self, data):
            self.sent.append(data)

    async def run():
        upstream = Upstream()
        lock = asyncio.Lock()
        await lock.acquire()
        send_task = asyncio.create_task(
            realtime_gateway_server._send_server_finish(upstream, lock, b"finish")
        )
        await asyncio.sleep(0)
        assert not send_task.done()

        final_completed_at = asyncio.get_running_loop().time()
        lock.release()
        send_started_at = await asyncio.wait_for(send_task, timeout=0.1)

        assert send_started_at >= final_completed_at
        assert upstream.sent == [b"finish"]

    asyncio.run(run())


def test_gateway_coordinator_release_treats_lost_lease_as_idempotent():
    class Response:
        status_code = 409

        @staticmethod
        def json():
            return {"detail": {"reason": "LEASE_LOST"}}

    class Client:
        async def request(self, method, path, *, json):
            assert method == "DELETE"
            assert path == "/v1/sessions/release"
            assert json["token"] == "token-a"
            return Response()

    async def run():
        client = HTTPCoordinatorClient.__new__(HTTPCoordinatorClient)
        client._client = Client()
        slot = WorkerSlot(
            worker_id="worker-a",
            role="denoiser",
            endpoint="ws://worker-a/generate",
            az="us-east-2a",
            slot_index=0,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        await client.release(
            SessionAssignment(
                user_id="user-a",
                session_id="session-a",
                generation_id="generation-a",
                token="token-a",
                expires_at=1,
                denoiser=slot,
                vae=slot,
            )
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (404, CoordinatorRejected),
        (408, httpx.HTTPStatusError),
        (503, httpx.HTTPStatusError),
    ],
)
def test_gateway_coordinator_claim_distinguishes_rejection_from_unknown_result(
    status_code, expected_error
):
    async def handler(_request):
        return httpx.Response(status_code)

    async def run():
        client = HTTPCoordinatorClient.__new__(HTTPCoordinatorClient)
        client._client = httpx.AsyncClient(
            base_url="http://coordinator.invalid",
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(expected_error) as raised:
                await client.claim_attempt(
                    attempt_id="11111111-1111-1111-1111-111111111111",
                    owner_id="session:generation",
                    expires_at=9999999999,
                )
            if status_code == 404:
                assert raised.value.reason == "COORDINATOR_HTTP_404"
        finally:
            await client._client.aclose()

    asyncio.run(run())


def test_gateway_coordinator_capacity_client_fetches_snapshot():
    snapshot = {
        "observed_at": 42.0,
        "roles": {
            "denoiser": {
                "waiting_sessions": 0,
                "active_sessions": 2,
                "queued_sessions": 0,
                "free_slots": 5,
                "draining_workers": 0,
            },
            "vae": {
                "waiting_sessions": 0,
                "active_sessions": 2,
                "queued_sessions": 0,
                "free_slots": 12,
                "draining_workers": 0,
            },
        },
    }

    async def handler(request):
        assert request.url.path == "/v1/capacity"
        return httpx.Response(200, json=snapshot)

    async def run():
        client = HTTPCoordinatorClient.__new__(HTTPCoordinatorClient)
        client._client = httpx.AsyncClient(
            base_url="http://coordinator.invalid",
            transport=httpx.MockTransport(handler),
        )
        try:
            assert await client.capacity() == snapshot
        finally:
            await client._client.aclose()

    asyncio.run(run())


def test_gateway_exposes_slot_capacity_summary_from_coordinator():
    app = create_app(
        _Coordinator(
            capacity={
                "observed_at": 42.0,
                "roles": {
                    "denoiser": {
                        "waiting_sessions": 1,
                        "active_sessions": 3,
                        "queued_sessions": 2,
                        "free_slots": 4,
                        "draining_workers": 0,
                    },
                    "vae": {
                        "waiting_sessions": 1,
                        "active_sessions": 3,
                        "queued_sessions": 0,
                        "free_slots": 11,
                        "draining_workers": 1,
                    },
                },
            }
        ),
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        internal_output_url="ws://gateway/v1/internal/realtime_output",
    )
    client = TestClient(app)

    assert sum(route.path == "/v1/slots" for route in app.routes) == 1
    assert sum(route.path == "/backends/minwm/v1/slots" for route in app.routes) == 1

    response = client.get("/v1/slots")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "object": "realtime_slot_capacity",
        "observed_at": 42.0,
        "used_slots": 3,
        "free_slots": 4,
        "total_slots": 7,
        "roles": {
            "denoiser": {
                "used_slots": 3,
                "free_slots": 4,
                "total_slots": 7,
                "waiting_sessions": 1,
                "queued_sessions": 2,
                "draining_workers": 0,
            },
            "vae": {
                "used_slots": 3,
                "free_slots": 11,
                "total_slots": 14,
                "waiting_sessions": 1,
                "queued_sessions": 0,
                "draining_workers": 1,
            },
        },
    }
    alias = client.get("/backends/minwm/v1/slots")
    assert alias.json() == response.json()
    assert alias.headers["cache-control"] == "no-store"


def test_gateway_slot_capacity_top_level_counts_online_users_for_dual_models():
    app = create_app(
        _Coordinator(
            capacity={
                "observed_at": 42.0,
                "roles": {
                    "denoiser": {
                        "waiting_sessions": 0,
                        "active_sessions": 2,
                        "queued_sessions": 0,
                        "free_slots": 2,
                        "draining_workers": 0,
                    },
                    "vae": {
                        "waiting_sessions": 0,
                        "active_sessions": 2,
                        "queued_sessions": 0,
                        "free_slots": 2,
                        "draining_workers": 0,
                    },
                },
            }
        ),
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        internal_output_url="ws://gateway/v1/internal/realtime_output",
        ui_config={"modelSlots": ["minwm", "lingbot2"]},
    )
    client = TestClient(app)

    response = client.get("/v1/slots")

    assert response.status_code == 200
    body = response.json()
    assert body["used_slots"] == 1
    assert body["free_slots"] == 1
    assert body["total_slots"] == 2
    assert body["roles"]["denoiser"]["used_slots"] == 1
    assert body["roles"]["denoiser"]["free_slots"] == 1
    assert body["roles"]["denoiser"]["total_slots"] == 2
    assert client.get("/backends/minwm/v1/slots").json() == body


def test_gateway_slot_capacity_returns_503_when_coordinator_is_unavailable():
    app = create_app(
        _Coordinator(capacity=RuntimeError("coordinator unavailable")),
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        internal_output_url="ws://gateway/v1/internal/realtime_output",
    )

    response = TestClient(app).get("/v1/slots")

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["detail"] == "coordinator capacity unavailable"


def test_gateway_serves_trace_query_and_accepts_sanitized_client_metrics():
    query = _TraceQuery()
    app = create_app(
        _Coordinator(),
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        internal_output_url="ws://gateway/v1/internal/realtime_output",
        trace_query=query,
    )
    client = TestClient(app)

    response = client.get("/v1/realtime_video/traces/trace-a?after=7&limit=220")
    assert response.status_code == 200
    assert response.json()["next_cursor"] == 9
    assert query.calls == [("trace-a", {"after": 7, "limit": 220})]

    response = client.post(
        "/v1/realtime_video/traces/trace-a/client-events",
        json={
            "events": [
                {
                    "name": "client.chunk_first_rendered",
                    "seq": 2,
                    "display_lag_ms": 84,
                    "prompt": "must not be accepted",
                }
            ]
        },
    )
    assert response.status_code == 200
    assert response.json() == {"accepted": 1}


def test_gateway_maps_first_cloudwatch_transport_failure_to_503():
    class FailingTraceQuery:
        async def query(self, *_args, **_kwargs):
            raise OSError("CloudWatch connection reset")

    app = create_app(
        _Coordinator(),
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        internal_output_url="ws://gateway/v1/internal/realtime_output",
        trace_query=FailingTraceQuery(),
    )

    response = TestClient(app).get("/v1/realtime_video/traces/trace-a")

    assert response.status_code == 503
    assert response.json()["detail"] == "trace query unavailable"


def test_gateway_readiness_depends_on_coordinator():
    ready_client = TestClient(
        create_app(
            _Coordinator(),
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url="ws://gateway/v1/internal/realtime_output",
        )
    )
    unavailable_client = TestClient(
        create_app(
            _Coordinator(ready=False),
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url="ws://gateway/v1/internal/realtime_output",
        )
    )

    assert ready_client.get("/readyz").status_code == 200
    response = unavailable_client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["detail"] == "coordinator unavailable"


def test_gateway_readiness_tolerates_a_bounded_coordinator_probe_blip():
    coordinator = _Coordinator()
    clock = [100.0]
    client = TestClient(
        create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url="ws://gateway/v1/internal/realtime_output",
            readiness_coordinator_grace_s=30.0,
            readiness_clock=lambda: clock[0],
        )
    )

    assert client.get("/readyz").json()["coordinator"] == "ready"
    coordinator.ready = False
    clock[0] = 125.0
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["coordinator"] == "degraded"

    clock[0] = 131.0
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["detail"] == "coordinator unavailable"


def test_gateway_cli_defaults_to_a_bounded_64_waiter_queue(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "realtime_gateway_server",
            "--coordinator-url=http://coordinator:18081",
            "--model-revision=minwm-r1",
        ],
    )

    assert _parse_args().max_admission_waiters == 64


@pytest.mark.parametrize(
    ("reason", "expected_code", "expected_close_code"),
    (
        ("CAPACITY_EXHAUSTED", "CAPACITY_EXHAUSTED", 1013),
        ("ADMISSION_QUEUE_FULL", "ADMISSION_QUEUE_FULL", 1013),
        ("USER_SESSION_LIMIT", "USER_SESSION_LIMIT", 1008),
        ("LEASE_LOST", "LEASE_LOST", 1008),
        ("GATEWAY_SESSION_FAILED", "GATEWAY_SESSION_FAILED", 1011),
        ("COORDINATOR_HTTP_503", "SESSION_UNAVAILABLE", 1008),
        ("internal failure: sensitive detail", "SESSION_UNAVAILABLE", 1008),
    ),
)
def test_world_platform_errors_are_fixed_and_sanitize_unknown_reasons(
    reason, expected_code, expected_close_code
):
    code, content, close_code = _world_error(reason)

    assert code == expected_code
    assert close_code == expected_close_code
    assert content == _WORLD_ERROR_MESSAGES[expected_code]
    assert not any("\u4e00" <= char <= "\u9fff" for char in content)
    assert reason not in content


def test_world_platform_browser_errors_are_friendly_and_non_technical():
    forbidden_terms = {
        "coordinator",
        "database",
        "exception",
        "gateway",
        "model",
        "schema",
        "token",
        "worker",
    }

    for content in _WORLD_ERROR_MESSAGES.values():
        lowered = content.lower()
        assert not any(term in lowered for term in forbidden_terms)


def test_browser_sender_error_includes_the_protocol_version_and_fixed_code():
    class Socket:
        def __init__(self):
            self.wire = None

        async def send_bytes(self, wire):
            self.wire = wire

    async def run():
        socket = Socket()
        sender = realtime_gateway_server._BrowserSender(socket)
        code, content, _ = _world_error("SESSION_INVALID")
        await sender.error(content, code=code)
        return decode_message(socket.wire)

    message = asyncio.run(run())
    assert message == {
        "version": 2,
        "type": "error",
        "code": "SESSION_INVALID",
        "content": _WORLD_ERROR_MESSAGES["SESSION_INVALID"],
    }


@pytest.mark.parametrize(
    ("reason", "retry_after_s", "expected_code"),
    (
        ("CAPACITY_EXHAUSTED", 0.25, 1013),
        ("USER_SESSION_LIMIT", None, 1008),
    ),
)
def test_gateway_uses_retryable_close_semantics_for_capacity_only(
    reason, retry_after_s, expected_code
):
    class RejectingCoordinator(_Coordinator):
        async def admit(self, **_request):
            raise CoordinatorRejected(reason, retry_after_s=retry_after_s)

    app = create_app(
        RejectingCoordinator(),
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        internal_output_url="ws://gateway/v1/internal/realtime_output",
        release_grace_s=0,
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime_video/generate") as websocket:
            message = decode_message(websocket.receive_bytes())
            assert message["type"] == "error"
            assert message["reason"] == reason
            assert message.get("retry_after_s") == retry_after_s
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_bytes()
            assert closed.value.code == expected_code


def test_partial_world_config_refuses_to_start():
    """All world-platform and recording flags or none.

    A partial set used to silently skip registering authorized_generate; the
    resulting 404 is underdetermined (a fronting proxy yields the same one), so
    operators lose hours. Startup failure names the missing flags instead.
    """

    class Args:
        world_token_ed25519_pub = ""
        world_callback_url = ""
        world_callback_hmac_secret = ""
        run_video_s3_bucket = ""
        run_video_s3_region = ""

    none_given = Args()
    assert _build_world_platform(none_given) is None  # showcase-only deploy stays valid

    partial = Args()
    partial.world_callback_url = "https://zing-api.example"
    with pytest.raises(SystemExit) as exc:
        _build_world_platform(partial)
    message = str(exc.value)
    missing_part = message.split("Missing:")[1]
    assert "--world-token-ed25519-pub" in missing_part
    assert "--world-callback-hmac-secret" in missing_part
    assert "--run-video-s3-bucket" in missing_part
    assert "--run-video-s3-region" in missing_part
    assert "--world-callback-url" not in missing_part


def test_complete_world_config_carries_recording_location():
    class Args:
        world_token_ed25519_pub = base64.b64encode(bytes(32)).decode()
        world_token_ed25519_key_id = "world-cn-k1"
        world_callback_url = "https://world-service.example"
        world_callback_hmac_secret = "secret"
        run_video_s3_bucket = "run-videos"
        run_video_s3_region = "us-west-2"
        run_video_s3_endpoint = "https://s3.oss-cn-beijing.aliyuncs.com"
        run_video_uri_scheme = "oss"

    config = _build_world_platform(Args())

    assert config.profile_id == "world-cn-k1"
    assert config.recording_bucket == "run-videos"
    assert config.recording_region == "us-west-2"
    assert config.recording_endpoint_url == "https://s3.oss-cn-beijing.aliyuncs.com"
    assert config.recording_uri_scheme == "oss"


def test_profiles_file_keeps_us_and_cn_resources_in_separate_bundles(tmp_path):
    document = {
        "version": 1,
        "legacy_profile_id": "world-us-k1",
        "profiles": [
            {
                "id": "world-us-k1",
                "token_ed25519_public_key": base64.b64encode(bytes(32)).decode(),
                "token_issuer": "world-service",
                "callback": {
                    "url": "https://zing-api.loopit.me",
                    "app_id": "zing-gateway",
                    "key_id": "k1",
                    "hmac_secret": "us-secret",
                },
                "recording": {
                    "bucket": "us-runs",
                    "region": "us-east-2",
                    "uri_scheme": "s3",
                },
            },
            {
                "id": "world-cn-k1",
                "token_ed25519_public_key": base64.b64encode(bytes(32)).decode(),
                "token_issuer": "world-service",
                "callback": {
                    "url": "https://zing-api.loopit.com.cn",
                    "app_id": "zing-gateway",
                    "key_id": "k1",
                    "hmac_secret": "cn-secret",
                },
                "recording": {
                    "bucket": "cn-runs",
                    "region": "cn-beijing",
                    "endpoint_url": "https://s3.oss-cn-beijing.aliyuncs.com",
                    "uri_scheme": "oss",
                    "access_key_id": "cn-key",
                    "secret_access_key": "cn-secret-key",
                },
            },
        ],
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(document))

    profiles = _load_world_platform_profiles(str(path))

    by_id = {profile.profile_id: profile for profile in profiles.profiles}
    assert profiles.legacy_profile_id == "world-us-k1"
    assert by_id["world-us-k1"].recording_uri_scheme == "s3"
    assert by_id["world-us-k1"].recording_endpoint_url == ""
    assert by_id["world-cn-k1"].callback_url.endswith(".com.cn")
    assert by_id["world-cn-k1"].recording_uri_scheme == "oss"
    assert by_id["world-cn-k1"].recording_access_key_id == "cn-key"


def test_profiles_file_rejects_unknown_fields_and_partial_credentials(tmp_path):
    profile = {
        "id": "world-cn-k1",
        "token_ed25519_public_key": base64.b64encode(bytes(32)).decode(),
        "token_issuer": "world-service",
        "callback": {
            "url": "https://zing-api.loopit.com.cn",
            "app_id": "zing-gateway",
            "key_id": "k1",
            "hmac_secret": "secret",
        },
        "recording": {
            "bucket": "cn-runs",
            "region": "cn-beijing",
            "endpoint_url": "https://s3.oss-cn-beijing.aliyuncs.com",
            "uri_scheme": "oss",
            "access_key_id": "key-without-secret",
        },
        "callback_url_typo": "https://wrong.example",
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"version": 1, "profiles": [profile]}))

    with pytest.raises(SystemExit, match="unknown fields"):
        _load_world_platform_profiles(str(path))

    profile.pop("callback_url_typo")
    path.write_text(json.dumps({"version": 1, "profiles": [profile]}))
    with pytest.raises(SystemExit, match="both key fields"):
        _load_world_platform_profiles(str(path))

    profile["recording"].pop("access_key_id")
    profile["recording"]["endpoint_url"] = "http://oss.internal"
    path.write_text(json.dumps({"version": 1, "profiles": [profile]}))
    with pytest.raises(SystemExit, match="must use HTTPS"):
        _load_world_platform_profiles(str(path))


def test_profiles_file_cannot_be_mixed_with_legacy_flags(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text('{"version": 1, "profiles": []}')

    class Args:
        world_platform_profiles_file = str(path)
        world_token_ed25519_pub = "legacy-public-key"
        world_token_ed25519_key_id = "legacy"
        world_callback_url = ""
        world_callback_hmac_secret = ""
        run_video_s3_bucket = ""
        run_video_s3_region = ""
        run_video_s3_endpoint = ""

    with pytest.raises(SystemExit, match="cannot be combined"):
        _build_world_platform_profiles(Args())


def test_create_app_builds_an_isolated_runtime_bundle_for_each_profile(monkeypatch):
    callback_profiles = []
    recording_profiles = []

    class Callbacks:
        def __init__(self, config):
            self.config = config
            callback_profiles.append(config.profile_id)

        async def recorded(self, *_args):
            return True

        async def close(self):
            pass

    class Manager:
        async def close(self):
            pass

    def build_manager(**kwargs):
        recording_profiles.append(kwargs)
        return Manager()

    monkeypatch.setattr(realtime_gateway_server, "WorldCallbacks", Callbacks)
    monkeypatch.setattr(
        realtime_gateway_server, "create_s3_recording_manager", build_manager
    )
    public_key = load_public_key(base64.b64encode(bytes(32)).decode())
    profiles = WorldPlatformProfiles(
        [
            WorldPlatformConfig(
                profile_id="world-us-k1",
                public_key=public_key,
                callback_url="https://zing-api.loopit.me",
                callback_app_id="zing-gateway",
                callback_key_id="k1",
                callback_secret="us-secret",
                recording_bucket="us-runs",
                recording_region="us-east-2",
            ),
            WorldPlatformConfig(
                profile_id="world-cn-k1",
                public_key=public_key,
                callback_url="https://zing-api.loopit.com.cn",
                callback_app_id="zing-gateway",
                callback_key_id="k1",
                callback_secret="cn-secret",
                recording_bucket="cn-runs",
                recording_region="cn-beijing",
                recording_endpoint_url="https://s3.oss-cn-beijing.aliyuncs.com",
                recording_uri_scheme="oss",
                recording_access_key_id="cn-key",
                recording_secret_access_key="cn-secret-key",
            ),
        ]
    )

    app = create_app(
        _Coordinator(),
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        internal_output_url="ws://gateway.invalid/v1/internal/realtime_output",
        world_platform_profiles=profiles,
    )

    assert app.state.world_platform_profiles is profiles
    assert callback_profiles == ["world-us-k1", "world-cn-k1"]
    assert recording_profiles[0]["bucket"] == "us-runs"
    assert recording_profiles[0]["endpoint_url"] == ""
    assert recording_profiles[1]["bucket"] == "cn-runs"
    assert recording_profiles[1]["uri_scheme"] == "oss"
    assert recording_profiles[1]["access_key_id"] == "cn-key"


@pytest.mark.parametrize("player_enabled", [False, True])
@pytest.mark.parametrize(
    "browser_origin", [None, "https://wtp.example", "https://gc.example"]
)
def test_whitebox_cors_uses_installed_player_config(
    monkeypatch, player_enabled, browser_origin
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_proxy import RoomProxy
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        MemoryRoomStore,
        RoomDirectory,
    )

    for key in (
        "WHITEBOX_BROWSER_ORIGIN",
        "WHITEBOX_ROOM_PUBLIC_ORIGIN",
        "WHITEBOX_PLAYER_ORIGIN",
        "WHITEBOX_CONTROL_ORIGIN",
    ):
        monkeypatch.delenv(key, raising=False)
    if browser_origin:
        monkeypatch.setenv("WHITEBOX_BROWSER_ORIGIN", browser_origin)
    if player_enabled:
        monkeypatch.setenv("WHITEBOX_ROOM_PUBLIC_ORIGIN", "https://media.example")
        monkeypatch.setenv("WHITEBOX_PLAYER_ORIGIN", "https://gc.example")
        monkeypatch.setenv("WHITEBOX_CONTROL_ORIGIN", "https://control.example")
    directory = RoomDirectory(MemoryRoomStore(), {"a": "http://room"})
    proxy = RoomProxy(
        directory, {"a": "http://room"}, "private-token", httpx.AsyncClient(), None
    )
    app = create_app(
        _Coordinator(),
        model_revision="m",
        vae_fingerprint="local",
        internal_output_url="ws://output",
        whitebox_proxy=proxy,
    )
    expected = set(
        filter(None, [browser_origin, "https://gc.example" if player_enabled else None])
    )
    with TestClient(app, base_url="https://media.example") as client:
        for request_origin in (
            "https://wtp.example",
            "https://gc.example",
            "https://foreign.example",
        ):
            headers = {"Origin": request_origin}
            preflight = client.options(
                "/v1/sessions/cors-no-session",
                headers={
                    **headers,
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "authorization",
                },
            )
            if expected:
                assert preflight.status_code == (
                    200 if request_origin in expected else 400
                )
            assert preflight.headers.get("access-control-allow-origin") == (
                request_origin if request_origin in expected else None
            )
            assert "access-control-allow-credentials" not in preflight.headers
            if request_origin in expected:
                assert preflight.headers["access-control-allow-methods"] == "GET, POST"
                assert "Origin" in preflight.headers["vary"]
                assert (
                    client.options(
                        "/healthz",
                        headers={**headers, "Access-Control-Request-Method": "DELETE"},
                    ).status_code
                    == 400
                )
            response = client.get("/healthz", headers=headers)
            assert response.headers.get("access-control-allow-origin") == (
                request_origin if request_origin in expected else None
            )
            assert "access-control-allow-credentials" not in response.headers
            if expected:
                assert (
                    response.headers["access-control-expose-headers"]
                    == "X-Content-SHA256"
                )
            # Middleware acceptance never grants a Session or bypasses its exact execution Origin.
            session = client.get("/v1/sessions/cors-no-session", headers=headers)
            assert session.status_code == (403 if player_enabled else 503)
            assert session.headers.get("access-control-allow-origin") == (
                request_origin if request_origin in expected else None
            )
            assert "access-control-allow-credentials" not in session.headers
    if expected:
        policy = next(
            m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"
        )
        assert len(policy.kwargs["allow_origins"]) == len(expected)


def test_default_gateway_does_not_install_whitebox_cors(monkeypatch):
    monkeypatch.setenv("WHITEBOX_BROWSER_ORIGIN", "https://wtp.example")
    app = create_app(
        _Coordinator(),
        model_revision="m",
        vae_fingerprint="local",
        internal_output_url="ws://output",
    )
    assert not any(m.cls.__name__ == "CORSMiddleware" for m in app.user_middleware)


def test_gateway_slots_request_the_fixed_model_and_vae_pool():
    coordinator = _Coordinator()
    app = create_app(
        coordinator,
        model_revision="public-model",
        vae_fingerprint="local",
        internal_output_url="ws://gateway/v1/internal/realtime_output",
    )
    assert TestClient(app).get("/v1/slots").status_code == 200
    assert coordinator.capacity_requests == [
        {"model_revision": "public-model", "vae_fingerprint": "local"}
    ]


@pytest.mark.parametrize("vae_fingerprint", ["local", "external"])
def test_gateway_slots_preserve_complete_player_groups(vae_fingerprint):
    coordinator = _Coordinator(
        capacity={
            "backend": "role_pool",
            "capacity_unit": "player_group",
            "roles": {"model_group": {"active_sessions": 1, "free_slots": 2}},
        }
    )
    app = create_app(
        coordinator,
        model_revision="pooled-model",
        vae_fingerprint=vae_fingerprint,
        internal_output_url="ws://gateway/v1/internal/realtime_output",
        ui_config={"modelSlots": ["encoder", "dit", "decoder"]},
    )
    response = TestClient(app).get("/v1/slots")
    assert response.status_code == 200
    result = response.json()
    assert (result["used_slots"], result["free_slots"], result["total_slots"]) == (
        1,
        2,
        3,
    )
    assert set(result["roles"]) == {"model_group"}


def test_gateway_capacity_client_passes_optional_model_scope():
    async def run():
        seen = []

        async def handler(request):
            seen.append(dict(request.url.params))
            return httpx.Response(200, json={"roles": {}})

        client = HTTPCoordinatorClient.__new__(HTTPCoordinatorClient)
        client._client = httpx.AsyncClient(
            base_url="http://coordinator.invalid",
            transport=httpx.MockTransport(handler),
        )
        try:
            await client.capacity()
            await client.capacity(
                model_revision="public-model", vae_fingerprint="local"
            )
            assert seen == [
                {},
                {"model_revision": "public-model", "vae_fingerprint": "local"},
            ]
        finally:
            await client._client.aclose()

    asyncio.run(run())
