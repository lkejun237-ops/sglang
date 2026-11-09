# SPDX-License-Identifier: Apache-2.0

"""Public realtime Gateway for Coordinator-routed Denoiser/VAE sessions."""

from __future__ import annotations

import argparse
import asyncio
import binascii
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import msgspec.msgpack
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedOK

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    ProtocolViolation,
    decode_message,
    encode_message,
)
from sglang.multimodal_gen.runtime.realtime.coordinator import (
    CoordinatorRejected,
    SessionAssignment,
    WorkerSlot,
)
from sglang.multimodal_gen.runtime.realtime.critical_path_metrics import (
    infer_model_label,
    observe_client_metric_event,
    observe_stage_seconds,
    prometheus_content_type,
    prometheus_latest,
    result_from_exception,
)
from sglang.multimodal_gen.runtime.realtime.gateway import (
    AdmissionQueueFull,
    BoundedAdmissionWaiterGate,
    BrowserPlaybackAckWindow,
    GatewayOutputRegistry,
    OutputBackpressureError,
    OutputDeliveryDetached,
    OutputProtocolError,
    OutputRouteClosed,
    build_denoiser_url,
    worker_message_allowed,
    worker_message_type,
)
from sglang.multimodal_gen.runtime.realtime.world_directions import (
    DirectionCoordinator,
    parse_init_directions,
)
from sglang.multimodal_gen.runtime.realtime.world_gateway_diagnostics import (
    BrowserInputDisconnectFact as _BrowserInputDisconnectFact,
)
from sglang.multimodal_gen.runtime.realtime.world_gateway_diagnostics import (
    BrowserOutputDetachFact as _BrowserOutputDetachFact,
)
from sglang.multimodal_gen.runtime.realtime.world_gateway_diagnostics import (
    WorldTerminalFact as _WorldTerminalFact,
)
from sglang.multimodal_gen.runtime.realtime.world_gateway_diagnostics import (
    build_world_terminal_fact as _build_world_terminal_fact,
)
from sglang.multimodal_gen.runtime.realtime.world_gateway_diagnostics import (
    classify_browser_input_close as _classify_browser_input_close,
)
from sglang.multimodal_gen.runtime.realtime.world_gateway_diagnostics import (
    classify_browser_input_close_metric as _classify_browser_input_close_metric,
)
from sglang.multimodal_gen.runtime.realtime.world_gateway_diagnostics import (
    classify_browser_output_detach as _classify_browser_output_detach,
)
from sglang.multimodal_gen.runtime.realtime.world_gateway_diagnostics import (
    input_disconnect_can_end_session as _input_disconnect_can_end_session,
)
from sglang.multimodal_gen.runtime.realtime.world_gateway_diagnostics import (
    normalize_websocket_close_code as _normalize_websocket_close_code,
)
from sglang.multimodal_gen.runtime.realtime.world_gateway_diagnostics import (
    record_browser_input_disconnect as _record_browser_input_disconnect,
)
from sglang.multimodal_gen.runtime.realtime.world_gateway_diagnostics import (
    record_browser_output_detach as _record_browser_output_detach,
)
from sglang.multimodal_gen.runtime.realtime.world_gateway_diagnostics import (
    record_world_session_cancellation as _record_world_session_cancellation,
)
from sglang.multimodal_gen.runtime.realtime.world_gateway_diagnostics import (
    world_disconnect_phase as _world_disconnect_phase,
)
from sglang.multimodal_gen.runtime.realtime.world_gateway_metrics import (
    observe_world_gateway_browser_delivery_detach,
    observe_world_gateway_browser_input_disconnect,
    observe_world_gateway_first_media,
)
from sglang.multimodal_gen.runtime.realtime.world_platform import (
    FINISH_TAIL_SECONDS,
    Principal,
    SessionPayloadSealer,
    TokenError,
    TokenReplayGuard,
    WorldCallbacks,
    WorldPlatformConfig,
    WorldPlatformProfiles,
    load_public_key,
    session_payload_aad,
)
from sglang.multimodal_gen.runtime.realtime.world_recording import (
    RecordingManager,
    create_s3_recording_manager,
)
from sglang.multimodal_gen.runtime.utils.realtime_trace import (
    compact_client_trace_event,
    emit_realtime_trace_payload,
    normalize_trace_id,
)

WEBUI_ROOT = Path(__file__).resolve().parents[2] / "apps" / "realtime_webui"
logger = logging.getLogger(__name__)
_PROXY_DROP_REQUEST_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_PROXY_DROP_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "content-encoding",
        "content-length",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_IDEMPOTENT_COORDINATOR_RELEASE_REASONS = frozenset(
    {
        "LEASE_LOST",
        "WORKER_LOST",
    }
)
_WORKER_SLOT_FIELDS = frozenset(field.name for field in fields(WorkerSlot))

# Browser-facing text for the world-platform route is a product contract. Keep
# it fixed, friendly, and free of coordinator/worker details. Unknown reasons
# collapse to SESSION_UNAVAILABLE so a coordinator response can never inject
# arbitrary text into the browser message, close reason, or lifecycle callback.
_WORLD_ERROR_MESSAGES = {
    "SESSION_INVALID": "We couldn't start this world. Return to the worlds list and try again.",
    "CAPACITY_EXHAUSTED": "No spots are available right now. Your place in line is saved.",
    "USER_SESSION_LIMIT": "You are already playing somewhere else. Finish that game before starting another.",
    "INVALID_SESSION_IDENTITY": "We couldn't start this world. Your place in line is saved.",
    "LEASE_LOST": "The connection was interrupted. Return to the worlds list to continue.",
    "WORKER_LOST": "The connection was interrupted. Return to the worlds list to continue.",
    "LEASE_RENEW_CONFLICT": "The connection was interrupted. Return to the worlds list to continue.",
    "ADMISSION_QUEUE_FULL": "A lot of players are joining right now. Your place in line is saved.",
    "SESSION_UNAVAILABLE": "We couldn't start this world. Your place in line is saved.",
    "GATEWAY_SESSION_FAILED": "Something went wrong. Return to the worlds list and try again.",
}
_WORLD_RETRYABLE_CLOSE_CODES = frozenset(
    {
        "CAPACITY_EXHAUSTED",
        "ADMISSION_QUEUE_FULL",
    }
)


def _world_error(reason: str) -> tuple[str, str, int]:
    code = reason if reason in _WORLD_ERROR_MESSAGES else "SESSION_UNAVAILABLE"
    if code == "GATEWAY_SESSION_FAILED":
        close_code = 1011
    else:
        close_code = 1013 if code in _WORLD_RETRYABLE_CLOSE_CODES else 1008
    return code, _WORLD_ERROR_MESSAGES[code], close_code


def _parse_ui_config(raw: str) -> dict[str, Any]:
    try:
        config = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("UI config must be valid JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("UI config must be a JSON object")
    return config


def _proxy_request_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _PROXY_DROP_REQUEST_HEADERS
    }


def _proxy_response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in _PROXY_DROP_RESPONSE_HEADERS
    }


def _log_gateway_trace(trace_id: str, event: str, **fields: Any) -> None:
    now_ms = int(time.time() * 1000)
    payload = {
        "trace_id": trace_id,
        "event": event,
        "server_epoch_ms": now_ms,
        "trace_seq": now_ms * 1000 + time.perf_counter_ns() % 1000,
        **fields,
    }
    emit_realtime_trace_payload(logger, payload)


def _browser_send_trace_fields(wire: bytes) -> dict[str, Any]:
    fields: dict[str, Any] = {"wire_bytes": len(wire)}
    try:
        message = decode_message(wire)
    except ProtocolViolation:
        fields["message_type"] = "invalid"
        return fields

    fields["message_type"] = message.get("type")
    for name in (
        "request_id",
        "chunk_index",
        "frame_batch_index",
        "num_frame_batches",
        "is_final_frame_batch",
        "event_id",
        "action_version",
        "prompt_version",
        "num_frames",
        "content_type",
        "encoding",
        "width",
        "height",
        "source_width",
        "source_height",
        "preview_width",
        "preview_height",
    ):
        value = message.get(name)
        if value is not None:
            fields[name] = value

    payload = message.get("payload")
    if isinstance(payload, (bytes, bytearray, memoryview)):
        fields["payload_bytes"] = len(payload)
    payload_lengths = message.get("payload_lengths")
    if isinstance(payload_lengths, list):
        lengths: list[int] = []
        for item in payload_lengths:
            try:
                lengths.append(int(item))
            except (TypeError, ValueError):
                continue
        if lengths:
            fields["payload_bytes"] = sum(lengths)
            fields["payload_count"] = len(lengths)
    return fields


def _codec_from_message(message: dict[str, Any]) -> str:
    codec = str(message.get("codec") or message.get("encoding") or "").lower()
    content_type = str(message.get("content_type") or "").lower()
    if "h264" in codec or "h264" in content_type or "avc" in content_type:
        return "h264"
    if "webp" in codec or "webp" in content_type:
        return "webp"
    if "jpeg" in codec or "jpg" in codec or "jpeg" in content_type:
        return "jpeg"
    return codec or "none"


def _metric_scope_for_message(message: dict[str, Any]) -> str:
    message_type = message.get("type")
    if message_type in {"frame_batch", "media_batch", "media_payload"}:
        return "frame"
    if message_type in {"chunk_telemetry", "media_chunk_complete"}:
        return "chunk"
    return "request"


def _metric_labels_from_wire(wire: bytes) -> tuple[str, str]:
    try:
        message = decode_message(wire)
    except ProtocolViolation:
        return "none", "request"
    return _codec_from_message(message), _metric_scope_for_message(message)


def _observe_gateway_client_metric(
    message: dict[str, Any],
    *,
    model: str,
) -> int:
    if message.get("type") == "client_metric":
        return int(observe_client_metric_event(message, service="gateway", model=model))
    if message.get("type") != "client_metric_batch":
        return 0
    events = message.get("events")
    if not isinstance(events, list):
        return 0
    accepted = 0
    for event in events[:64]:
        accepted += int(
            observe_client_metric_event(event, service="gateway", model=model)
        )
    return accepted


def _worker_slot_from_payload(payload: dict[str, Any]) -> WorkerSlot:
    return WorkerSlot(
        **{key: value for key, value in payload.items() if key in _WORKER_SLOT_FIELDS}
    )


class CoordinatorClient(Protocol):
    async def health(self) -> dict[str, Any]: ...

    async def capacity(
        self, *, model_revision: str | None = None, vae_fingerprint: str | None = None
    ) -> dict[str, Any]: ...

    async def claim_attempt(
        self, *, attempt_id: str, owner_id: str, expires_at: float
    ) -> None: ...

    async def admit(self, **request: Any) -> SessionAssignment: ...

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment: ...

    async def release(self, assignment: SessionAssignment) -> None: ...


def _assignment(payload: dict[str, Any]) -> SessionAssignment:
    return SessionAssignment(
        user_id=payload["user_id"],
        session_id=payload["session_id"],
        generation_id=payload["generation_id"],
        token=payload["token"],
        expires_at=float(payload["expires_at"]),
        admitted_at=payload.get("admitted_at"),
        player_expires_at=payload.get("player_expires_at"),
        denoiser=_worker_slot_from_payload(payload["denoiser"]),
        vae=_worker_slot_from_payload(payload["vae"]) if payload.get("vae") else None,
    )


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _online_slot_cost_from_ui_config(ui_config: dict[str, Any] | None) -> int:
    if not isinstance(ui_config, dict):
        return 1
    model_slots = ui_config.get("modelSlots")
    if not isinstance(model_slots, list):
        return 1
    configured_slots = {str(slot).strip() for slot in model_slots if str(slot).strip()}
    return max(1, len(configured_slots))


def _slot_capacity_summary(
    snapshot: dict[str, Any], *, online_slot_cost: int = 1
) -> dict[str, Any]:
    online_slot_cost = (
        1
        if snapshot.get("capacity_unit") == "player_group"
        else max(1, int(online_slot_cost))
    )
    role_summaries: dict[str, dict[str, int]] = {}
    for role, values in sorted((snapshot.get("roles") or {}).items()):
        if not isinstance(values, dict):
            continue
        used_slots = _nonnegative_int(values.get("active_sessions"))
        free_slots = _nonnegative_int(values.get("free_slots"))
        role_summaries[role] = {
            "used_slots": _ceil_div(used_slots, online_slot_cost),
            "free_slots": free_slots // online_slot_cost,
            "total_slots": (used_slots + free_slots) // online_slot_cost,
            "waiting_sessions": _nonnegative_int(values.get("waiting_sessions")),
            "queued_sessions": _nonnegative_int(values.get("queued_sessions")),
            "draining_workers": _nonnegative_int(values.get("draining_workers")),
        }

    if role_summaries:
        used_slots = max(role["used_slots"] for role in role_summaries.values())
        free_slots = min(role["free_slots"] for role in role_summaries.values())
        total_slots = min(role["total_slots"] for role in role_summaries.values())
    else:
        used_slots = 0
        free_slots = 0
        total_slots = 0

    return {
        "object": "realtime_slot_capacity",
        "observed_at": snapshot.get("observed_at"),
        "used_slots": used_slots,
        "free_slots": free_slots,
        "total_slots": total_slots,
        "roles": role_summaries,
    }


class HTTPCoordinatorClient:
    def __init__(self, base_url: str, *, timeout_s: float = 15.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_s
        )

    @staticmethod
    def _raise_rejection(response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            detail = response.json().get("detail", {})
        except (ValueError, AttributeError):
            detail = {}
        reason = detail.get("reason") or f"COORDINATOR_HTTP_{response.status_code}"
        raise CoordinatorRejected(
            reason,
            retry_after_s=detail.get("retry_after_s"),
        )

    async def admit(self, **request: Any) -> SessionAssignment:
        response = await self._client.post("/v1/sessions/admit", json=request)
        self._raise_rejection(response)
        return _assignment(response.json())

    async def claim_attempt(
        self, *, attempt_id: str, owner_id: str, expires_at: float
    ) -> None:
        response = await self._client.post(
            "/v1/attempts/claim",
            json={
                "attempt_id": attempt_id,
                "owner_id": owner_id,
                "expires_at": expires_at,
            },
        )
        # Only claim-contract rejections prove that the conditional write did
        # not acquire ownership. Timeouts, rate limits, redirects, and 5xx may
        # happen after it commits and must remain indeterminate HTTP errors.
        # This classification only selects the browser error; neither failure
        # path grants callback ownership.
        definitive_rejections = {400, 404, 405, 409, 422}
        if (
            not response.is_success
            and response.status_code not in definitive_rejections
        ):
            response.raise_for_status()
        self._raise_rejection(response)

    async def health(self) -> dict[str, Any]:
        response = await self._client.get("/healthz")
        response.raise_for_status()
        return response.json()

    async def capacity(
        self, *, model_revision: str | None = None, vae_fingerprint: str | None = None
    ) -> dict[str, Any]:
        params = {
            key: value
            for key, value in {
                "model_revision": model_revision,
                "vae_fingerprint": vae_fingerprint,
            }.items()
            if value is not None
        }
        response = await self._client.get("/v1/capacity", params=params)
        response.raise_for_status()
        return response.json()

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment:
        response = await self._client.post(
            "/v1/sessions/renew", json=asdict(assignment)
        )
        self._raise_rejection(response)
        return _assignment(response.json())

    async def release(self, assignment: SessionAssignment) -> None:
        response = await self._client.request(
            "DELETE", "/v1/sessions/release", json=asdict(assignment)
        )
        if response.status_code == 404:
            return
        if response.status_code == 409:
            try:
                detail = response.json().get("detail", {})
            except (ValueError, AttributeError):
                detail = {}
            if detail.get("reason") in _IDEMPOTENT_COORDINATOR_RELEASE_REASONS:
                return
        self._raise_rejection(response)

    async def release_verified(
        self, assignment: SessionAssignment, *, sdk_cleanup=None
    ) -> None:
        response = await self._client.request(
            "DELETE",
            "/v1/sessions/release-verified",
            json={
                **asdict(assignment),
                **({"sdk_cleanup": sdk_cleanup} if sdk_cleanup is not None else {}),
            },
        )
        self._raise_rejection(response)

    async def quarantine(self, assignment: SessionAssignment) -> None:
        response = await self._client.post(
            "/v1/sessions/quarantine", json=asdict(assignment)
        )
        self._raise_rejection(response)

    async def close(self) -> None:
        await self._client.aclose()


class _BrowserSender:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self._lock = asyncio.Lock()

    async def send(self, payload: bytes | str) -> None:
        async with self._lock:
            if isinstance(payload, bytes):
                await self.websocket.send_bytes(payload)
            else:
                await self.websocket.send_text(payload)

    async def error(self, content: str, **fields: Any) -> None:
        await self.send(encode_message("error", content=content, **fields))


def _user_id(websocket: WebSocket) -> str:
    query = websocket.query_params.get("user_id")
    if query:
        return f"query:{query[:240]}"
    header = websocket.headers.get("x-user-id")
    if header:
        return f"header:{header[:240]}"
    client = websocket.client.host if websocket.client else "unknown"
    return f"client:{client}"


def _world_coordinator_user_id(principal: Principal) -> str:
    """Return the role-scoped lease key used by the Coordinator.

    A comparison is allowed alongside the primary session for the same signed
    account, while a second session of either role still hits the existing
    one-lease-per-user fence.
    """
    user_id = f"wsvc:{principal.profile_id}:{principal.lease_key}"
    if principal.role == "comparison":
        return f"{user_id}:lingbot2"
    return user_id


def _world_coordinator_attempt_id(principal: Principal) -> str:
    """Namespace the global owner fence while preserving raw jti for callbacks."""
    return f"{principal.profile_id}:{principal.jti}"


async def _receive_browser(websocket: WebSocket) -> bytes | str:
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(_normalize_websocket_close_code(message.get("code")))
    if message.get("bytes") is not None:
        return message["bytes"]
    if message.get("text") is not None:
        return message["text"]
    raise WebSocketDisconnect(1002)


# Inbound allowlist for authorized sessions. This is an allowlist rather than a
# denylist so future engine control fields do not become browser-injectable just
# because they were not explicitly blocked.
_WORLD_PASSTHROUGH_TYPES = frozenset({"client_metric", "client_metric_batch", "ack"})
# Only kinds the MinWM adapter actually ingests. "camera", "move", and "action"
# used to sit here: MinWM.ingest_event rejects all three (it accepts
# action_labels, action_weights, camera_actions, prompt, scene_cut, seed, and
# chunk_seeds), so forwarding them produced an engine-side "unsupported MinWM
# event kind" error instead of doing anything. They were dead slots that made
# the contract look wider than it is.
_WORLD_ALLOWED_EVENT_KINDS = frozenset({"camera_actions", "playback_ack", "heartbeat"})
# Free-form input no longer passes raw prompt/scene_cut through. Full scene
# descriptions are server-side assets; the browser sends only the user's raw
# instruction as kind:"direction", then the gateway asks world-service to rewrite
# and dispatch the full prompt. The direction length cap matches world-service.
_WORLD_DIRECTION_MAX_CHARS = 2000
# event_id crosses JavaScript, Python, JSON, Go, and the engine's integer field.
# Keep the accepted range identical on every hop and reject bool explicitly,
# because bool is a subclass of int in Python.
_WORLD_DIRECTION_EVENT_ID_MAX = (1 << 53) - 1
# Cap on in-flight direction/skill tasks per session. The coordinator already
# collapses a burst into "one in flight plus one queued", so normal play stays
# in the single digits; anything above this is scripted frame spam and is
# dropped. Without a cap the receive loop creates one task per frame and the
# frame rate is chosen by the attacker.
_WORLD_MAX_DIRECTION_TASKS = 8

# Init-message extension block visible only to the gateway. This matches the
# world-service zingproto.WorldExtKey value.
WORLD_EXT_KEY = "_world"


class _WorldSessionAbort(RuntimeError):
    """Fixed internal terminal reason for an authorized session."""

    def __init__(self, reason: str, *, fault: str = "ours") -> None:
        super().__init__(reason)
        self.reason = reason
        self.fault = fault


def _is_nonempty_real_media(wire: bytes) -> bool:
    """Return whether a successfully sent Gateway frame starts play time."""
    try:
        message = decode_message(wire)
    except ProtocolViolation:
        return False
    message_type = message.get("type")
    if message_type == "media_payload":
        payload = message.get("payload")
        return isinstance(payload, bytes) and bool(payload)
    if message_type == "frame_batch":
        payload = message.get("payload")
        return isinstance(payload, bytes) and bool(payload)
    return False


async def _sleep_until(deadline: float) -> None:
    await asyncio.sleep(max(0.0, deadline - asyncio.get_running_loop().time()))


async def _send_server_finish(
    upstream: Any,
    upstream_send_lock: asyncio.Lock,
    data: bytes,
) -> float:
    """Send finish and freeze the instant it owns the upstream wire lock."""
    async with upstream_send_lock:
        send_started_at = asyncio.get_running_loop().time()
        await upstream.send(data)
        return send_started_at


def _world_finish_outcome(
    *,
    now: float,
    finish_deadline: float,
    hard_deadline: float,
    nominal_finish_deadline: float,
    worker_succeeded: bool,
    worker_finished_at: float | None,
    final_completed_at: float | None,
) -> str | None:
    """Return the frozen terminal result, or None while facts may still arrive."""
    worker_completed = (
        worker_succeeded
        and isinstance(worker_finished_at, float)
        and worker_finished_at <= finish_deadline
    )
    final_completed = (
        isinstance(final_completed_at, float) and final_completed_at <= finish_deadline
    )
    # The validated H.264 final follows encoder EOF and seals the recorder.
    # Worker close handshake and browser delivery are cleanup/presentation
    # facts, so neither may invalidate media that is already complete.
    if final_completed:
        return "completed"

    late_terminal_fact = any(
        isinstance(value, float) and value > finish_deadline
        for value in (worker_finished_at, final_completed_at)
    )
    if not late_terminal_fact and now < finish_deadline:
        return None
    if hard_deadline <= nominal_finish_deadline:
        return "MAX_LIFETIME_REACHED"
    if worker_completed and not final_completed:
        return "MEDIA_FINAL_MISSING"
    return "FINISH_TIMEOUT"


def _world_failure_precedes_final(
    *,
    failure_at: float | None,
    final_completed_at: float | None,
    finish_deadline: float,
) -> bool:
    """Return whether a task failure may still invalidate source completion."""
    if isinstance(failure_at, float) and failure_at > finish_deadline:
        return False
    if not isinstance(final_completed_at, float):
        return True
    if not isinstance(failure_at, float):
        return True
    # Frozen facts on the same event-loop boundary favor the validated final,
    # matching _world_finish_outcome and avoiding task-set iteration races.
    return failure_at < final_completed_at


async def _cancel_tasks(tasks: set[asyncio.Task]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@dataclass(frozen=True)
class _WorldPlatformRuntime:
    """Resources selected together from one verified token profile."""

    config: WorldPlatformConfig
    callbacks: WorldCallbacks
    payload_sealer: SessionPayloadSealer
    recording_manager: RecordingManager


def create_app(
    coordinator: CoordinatorClient,
    *,
    model_revision: str,
    vae_fingerprint: str,
    internal_output_url: str,
    lingbot2_upstream_url: str | None = None,
    lingbot2_model_revision: str = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
    lingbot2_vae_fingerprint: str | None = None,
    output_queue_depth: int = 64,
    output_enqueue_timeout_s: float = 0.0,
    output_queue_max_bytes: int = 16 * 1024 * 1024,
    output_queue_max_messages: int = 256,
    output_drain_timeout_s: float = 5.0,
    lease_renew_interval_s: float = 10.0,
    release_grace_s: float = 0.5,
    max_admission_waiters: int = 64,
    readiness_coordinator_timeout_s: float = 1.0,
    readiness_coordinator_grace_s: float = 30.0,
    readiness_clock=time.monotonic,
    connect_factory=connect,
    ui_config: dict[str, Any] | None = None,
    trace_query=None,
    world_platform: WorldPlatformConfig | None = None,
    world_platform_profiles: WorldPlatformProfiles | None = None,
    recording_manager: RecordingManager | None = None,
    browser_send_timeout_s: float = 15.0,
    world_studio_webui_base_url: str | None = None,
    # Unauthenticated showcase routes, static pages, and trace queries. These are
    # registered by default.
    #
    # These routes must not be public. They can create sessions without lifetime,
    # queueing, or auth limits, bypassing all authorized_generate constraints.
    # The boundary is intentionally the network layer, not this process: public
    # browsers should see only authorized_generate through ingress, while the
    # internal web UI still needs these legacy routes through the cluster Service.
    # A process-level switch cannot distinguish those two traffic classes.
    #
    # Correct production isolation: expose only authorized_generate on public
    # ingress. This switch is for deployments dedicated to platform traffic with
    # no showcase traffic at all.
    disable_legacy_routes: bool = False,
    whitebox_config=None,
    whitebox_archives=None,
    whitebox_proxy=None,
    prefill_store=None,
) -> FastAPI:
    whitebox_reports = None
    if whitebox_config is not None:
        from sglang.multimodal_gen.runtime.realtime.whitebox.recording import (
            ArchiveManager,
        )
        from sglang.multimodal_gen.runtime.realtime.whitebox.reports import (
            Reports,
            SessionReport,
        )

        if vae_fingerprint != "local":
            raise ValueError("Whitebox H3 requires --vae-fingerprint local")
        whitebox_archives = whitebox_archives or ArchiveManager()
        whitebox_reports = Reports(whitebox_config.spool, whitebox_archives)
    if release_grace_s < 0:
        raise ValueError("release_grace_s must be non-negative")
    if browser_send_timeout_s <= 0:
        raise ValueError("browser_send_timeout_s must be positive")
    if world_platform is not None and world_platform_profiles is not None:
        raise ValueError(
            "world_platform and world_platform_profiles cannot both be configured"
        )
    if world_platform_profiles is None and world_platform is not None:
        world_platform_profiles = WorldPlatformProfiles.from_legacy(world_platform)
    if recording_manager is not None and (
        world_platform_profiles is None or len(world_platform_profiles.profiles) != 1
    ):
        raise ValueError(
            "recording_manager injection requires exactly one world platform profile"
        )

    # Every environment profile owns a complete, inseparable set of resources.
    # The signed key ID selects this bundle; no bucket, callback URL, or secret
    # can be supplied by the browser token payload.
    enable_legacy_routes = not disable_legacy_routes
    world_runtimes: dict[str, _WorldPlatformRuntime] = {}
    for profile in world_platform_profiles.profiles if world_platform_profiles else ():
        callbacks = WorldCallbacks(profile)
        profile_recording_manager = recording_manager or create_s3_recording_manager(
            bucket=profile.recording_bucket,
            region=profile.recording_region,
            on_recorded=callbacks.recorded,
            endpoint_url=profile.recording_endpoint_url,
            uri_scheme=profile.recording_uri_scheme,
            access_key_id=profile.recording_access_key_id,
            secret_access_key=profile.recording_secret_access_key,
            session_token=profile.recording_session_token,
        )
        world_runtimes[profile.profile_id] = _WorldPlatformRuntime(
            config=profile,
            callbacks=callbacks,
            payload_sealer=SessionPayloadSealer(profile.callback_secret),
            recording_manager=profile_recording_manager,
        )
    if output_drain_timeout_s <= 0:
        raise ValueError("output_drain_timeout_s must be positive")
    if readiness_coordinator_timeout_s <= 0:
        raise ValueError("readiness_coordinator_timeout_s must be positive")
    if readiness_coordinator_grace_s < 0:
        raise ValueError("readiness_coordinator_grace_s must be non-negative")
    registry = GatewayOutputRegistry(
        queue_depth=output_queue_depth,
        enqueue_timeout_s=output_enqueue_timeout_s,
        max_queue_bytes=output_queue_max_bytes,
        max_queue_messages=output_queue_max_messages,
    )
    admission_gate = BoundedAdmissionWaiterGate(max_waiters=max_admission_waiters)
    prefill_manager = prefill_redis = None
    if os.environ.get("WHITEBOX_PREFILL_ENABLED") == "1":
        from sglang.multimodal_gen.runtime.realtime.whitebox.prefill_gateway import (
            PrefillGateway,
            RedisPrefillStore,
        )

        if prefill_store is None:
            from redis.asyncio import Redis

            redis_url = os.environ.get("WHITEBOX_PREFILL_REDIS_URL")
            if not redis_url:
                raise ValueError("Prefill requires a shared result store")
            prefill_redis = Redis.from_url(
                redis_url,
                socket_connect_timeout=3,
                socket_timeout=3,
                health_check_interval=30,
                socket_keepalive=True,
            )
            prefill_store = RedisPrefillStore(prefill_redis)
        prefill_manager = PrefillGateway(
            coordinator=coordinator,
            connect=connect_factory,
            secret=os.environ.get("WHITEBOX_RTC_AUTH_SECRET", ""),
            model_revision=model_revision,
            vae_fingerprint=vae_fingerprint,
            store=prefill_store,
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if whitebox_archives is not None:
            whitebox_archives.start(whitebox_config.spool)
        try:
            yield
        finally:
            try:
                if prefill_manager is not None:
                    await prefill_manager.close()
                if prefill_redis is not None:
                    await prefill_redis.aclose()
                if whitebox_proxy is not None:
                    await whitebox_proxy.close()
                if whitebox_archives is not None:
                    await whitebox_archives.close()
                closed_managers: set[int] = set()
                for runtime in world_runtimes.values():
                    manager_id = id(runtime.recording_manager)
                    if manager_id not in closed_managers:
                        await runtime.recording_manager.close()
                        closed_managers.add(manager_id)
            finally:
                try:
                    for runtime in world_runtimes.values():
                        await runtime.callbacks.close()
                finally:
                    closer = getattr(coordinator, "close", None)
                    if closer is not None:
                        await closer()

    app = FastAPI(title="SGLang Realtime Gateway", lifespan=lifespan)
    from sglang.multimodal_gen.runtime.realtime.release_identity import (
        install_release_route,
    )

    install_release_route(app, "gateway")
    if prefill_manager is not None:
        from sglang.multimodal_gen.runtime.realtime.whitebox.prefill_gateway import (
            register_prefill_routes,
        )

        register_prefill_routes(app, prefill_manager)
        app.state.prefill_manager = prefill_manager
    if whitebox_proxy is not None:
        whitebox_proxy.install(app)
    if whitebox_reports is not None:
        whitebox_reports.install(app)
    if whitebox_reports is not None or whitebox_proxy is not None:
        browser_origin = os.environ.get("WHITEBOX_BROWSER_ORIGIN")
        player_config = getattr(app.state, "whitebox_player_transport_config", None)
        # Preflight is intercepted here before the Session route's own policy.
        # CORS permits reading responses; it does not grant a viewer capability.
        allowed_origins = list(
            dict.fromkeys(
                value
                for value in (
                    browser_origin,
                    player_config.player_origin if player_config else None,
                    player_config.member_prefill_origin if player_config else None,
                )
                if value
            )
        )
        if allowed_origins:
            from fastapi.middleware.cors import CORSMiddleware

            app.add_middleware(
                CORSMiddleware,
                allow_origins=allowed_origins,
                allow_methods=["GET", "POST"],
                allow_headers=["Authorization", "Content-Type"],
                expose_headers=["X-Content-SHA256"],
            )
    app.state.output_registry = registry
    if prefill_manager is not None:
        from sglang.multimodal_gen.runtime.realtime.whitebox.prefill_gateway import (
            PrefillStatusCors,
        )

        app.add_middleware(PrefillStatusCors)
    app.state.admission_gate = admission_gate
    app.state.recording_manager = (
        next(iter(world_runtimes.values())).recording_manager
        if len(world_runtimes) == 1
        else None
    )
    app.state.world_platform_profiles = world_platform_profiles
    webui_base_url = (world_studio_webui_base_url or "").strip().rstrip("/")
    last_coordinator_ready_at: float | None = None
    online_slot_cost = _online_slot_cost_from_ui_config(ui_config)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return Response(prometheus_latest(), media_type=prometheus_content_type())

    @app.get("/readyz")
    async def readyz():
        nonlocal last_coordinator_ready_at
        try:
            await asyncio.wait_for(
                coordinator.health(),
                timeout=readiness_coordinator_timeout_s,
            )
        except Exception as exc:
            now = readiness_clock()
            if (
                last_coordinator_ready_at is not None
                and now - last_coordinator_ready_at <= readiness_coordinator_grace_s
            ):
                return {
                    "status": "ready",
                    "coordinator": "degraded",
                    "last_success_age_s": round(now - last_coordinator_ready_at, 3),
                }
            raise HTTPException(
                status_code=503, detail="coordinator unavailable"
            ) from exc
        last_coordinator_ready_at = readiness_clock()
        return {"status": "ready", "coordinator": "ready"}

    @app.get("/backends/minwm/v1/slots")
    @app.get("/v1/slots")
    async def slots(response: Response):
        response.headers["Cache-Control"] = "no-store"
        try:
            snapshot = await coordinator.capacity(
                model_revision=model_revision, vae_fingerprint=vae_fingerprint
            )
            if (
                vae_fingerprint == "local"
                and snapshot.get("capacity_unit") != "player_group"
            ):
                snapshot = {
                    **snapshot,
                    "roles": {
                        "denoiser": snapshot.get("roles", {}).get("denoiser", {})
                    },
                }
            return _slot_capacity_summary(snapshot, online_slot_cost=online_slot_cost)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="coordinator capacity unavailable",
                headers={"Cache-Control": "no-store"},
            ) from exc

    @app.get("/backends/minwm/v1/models")
    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_revision}]}

    @app.get("/backends/lingbot2/v1/models")
    async def lingbot2_models():
        return {"object": "list", "data": [{"id": lingbot2_model_revision}]}

    # Internal diagnostics: returns worker id, engine hostname, adapter class, and
    # other implementation details. It shares the showcase-route switch; public
    # access must be blocked by ingress along with /metrics and /v1/models.
    if enable_legacy_routes:

        @app.get("/v1/realtime_video/traces/{trace_id}")
        async def get_trace(trace_id: str, after: int = 0, limit: int = 220):
            if trace_query is None:
                raise HTTPException(
                    status_code=503, detail="Trace query is not configured"
                )
            try:
                return await trace_query.query(trace_id, after=after, limit=limit)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("Trace query failed for trace_id=%s", trace_id)
                raise HTTPException(
                    status_code=503, detail="trace query unavailable"
                ) from exc

    @app.post("/v1/realtime_video/traces/{trace_id}/client-events")
    async def post_client_trace(trace_id: str, payload: dict):
        normalized = normalize_trace_id(trace_id, fallback="")
        if not normalized or normalized != trace_id:
            raise HTTPException(status_code=400, detail="invalid trace_id")
        raw_events = payload.get("events")
        if not isinstance(raw_events, list) or len(raw_events) > 64:
            raise HTTPException(
                status_code=400, detail="events must contain at most 64 items"
            )
        accepted = 0
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            event = compact_client_trace_event(raw_event)
            event["trace_id"] = trace_id
            event["event"] = str(event.pop("name", "client.metric"))[:128]
            event["server_epoch_ms"] = int(time.time() * 1000)
            event["trace_seq"] = (
                event["server_epoch_ms"] * 1000 + int(event.get("seq") or 0) % 1000
            )
            emit_realtime_trace_payload(logger, event)
            accepted += 1
        return {"accepted": accepted}

    @app.get("/runtime-config.js")
    async def runtime_config():
        config = ui_config or {}
        body = f"globalThis.SGLANG_REALTIME_UI_CONFIG = {json.dumps(config)};\n"
        return Response(
            body,
            media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    if webui_base_url:

        async def _proxy_world_studio_api(path: str, request: Request):
            target = f"{webui_base_url}/api/{path}"
            if request.url.query:
                target = f"{target}?{request.url.query}"
            async with httpx.AsyncClient(timeout=90.0) as client:
                upstream = await client.request(
                    request.method,
                    target,
                    headers=_proxy_request_headers(request),
                    content=await request.body(),
                )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=_proxy_response_headers(upstream),
                media_type=upstream.headers.get("content-type"),
            )

        @app.api_route(
            "/api/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        )
        async def proxy_world_studio_api(path: str, request: Request):
            return await _proxy_world_studio_api(path, request)

        @app.api_route(
            "/studio-api/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        )
        async def proxy_world_studio_api_alias(path: str, request: Request):
            return await _proxy_world_studio_api(path, request)

        @app.api_route(
            "/backends/minwm/v1/studio-api/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        )
        async def proxy_minwm_world_studio_api_alias(path: str, request: Request):
            return await _proxy_world_studio_api(path, request)

    @app.websocket("/v1/internal/realtime_output")
    async def realtime_output(websocket: WebSocket):
        await websocket.accept()
        route = None
        session_id = ""
        generation_id = ""
        output_token = ""
        try:
            opened = decode_message(await websocket.receive_bytes())
            if opened.get("type") != "session_output_open":
                raise OutputProtocolError("session_output_open is required")
            session_id = str(opened.get("session_id") or "")
            generation_id = str(opened.get("generation_id") or "")
            output_token = str(opened.get("token") or "")
            route = await registry.bind(
                session_id,
                generation_id,
                token=output_token,
            )
            await websocket.send_bytes(
                encode_message(
                    "session_output_accepted",
                    session_id=session_id,
                    generation_id=generation_id,
                )
            )
            while True:
                wire = await websocket.receive_bytes()
                message = decode_message(wire)
                enqueue_started = time.perf_counter()
                await route.put(wire)
                _log_gateway_trace(
                    route.trace_id,
                    "gateway.output_enqueued",
                    session_id=session_id,
                    generation_id=generation_id,
                    enqueue_ms=round((time.perf_counter() - enqueue_started) * 1000, 3),
                    **route.queue_metrics(),
                    **_browser_send_trace_fields(wire),
                )
                if message.get("type") == "media_chunk_complete":
                    await websocket.send_bytes(
                        encode_message(
                            "media_chunk_complete_accepted",
                            session_id=session_id,
                            generation_id=generation_id,
                            request_id=message.get("request_id"),
                            chunk_index=message.get("chunk_index"),
                        )
                    )
        except (WebSocketDisconnect, OutputRouteClosed):
            pass
        except OutputBackpressureError as exc:
            await websocket.close(code=1013, reason=str(exc))
        except (OutputProtocolError, ProtocolViolation) as exc:
            await websocket.close(code=1008, reason=str(exc))
        finally:
            if route is not None:
                await registry.unbind(session_id, generation_id, token=output_token)

    async def _generate_coordinator_session(
        websocket: WebSocket,
        *,
        selected_model_revision: str,
        selected_vae_fingerprint: str,
        principal: Principal | None = None,
        world_runtime: _WorldPlatformRuntime | None = None,
        classic_whitebox: bool = False,
    ) -> None:
        # principal set means an authorized world-platform session. Transport,
        # admission, deadline, and cleanup apply to both roles; business
        # callbacks and recording belong to primary only. principal unset means
        # the legacy showcase route, whose behavior stays byte-for-byte compatible.
        if (principal is None) != (world_runtime is None):
            raise ValueError(
                "principal and world platform runtime must be provided together"
            )
        world_callbacks = world_runtime.callbacks if world_runtime else None
        session_recording_manager = (
            world_runtime.recording_manager if world_runtime else None
        )
        payload_sealer = world_runtime.payload_sealer if world_runtime else None
        await websocket.accept()
        session_opened_monotonic = asyncio.get_running_loop().time()
        business_lifecycle_owner = principal is not None and principal.role == "primary"
        sender = _BrowserSender(websocket)
        session_id = uuid4().hex
        generation_id = uuid4().hex
        trace_id = normalize_trace_id(
            websocket.query_params.get("trace_id"), fallback=session_id
        )
        whitebox_report = (
            SessionReport(trace_id)
            if whitebox_reports is not None and principal is None
            else None
        )
        output_token = secrets.token_urlsafe(32)
        metric_model = infer_model_label(selected_model_revision)
        assignment = None
        prefetched_whitebox_init = []
        route = None
        recorder = None
        upstream = None
        tasks: set[asyncio.Task] = set()
        expected_last_chunk: int | None = None
        playback_ack_window = BrowserPlaybackAckWindow()
        first_media_event = asyncio.Event()
        admitted_monotonic: float | None = None
        admitted_at_ms: int | None = None
        # Branch-local state for world sessions: init verification, deadline, and
        # rejection reason.
        world_session: dict[str, Any] = {
            "init_verified": False,
            "attempt_owner": principal is None,
        }
        # Player-direction orchestrator created after init validation. Dynamic
        # tasks are tracked separately and canceled together in finally.
        directions: DirectionCoordinator | None = None
        direction_tasks: set[asyncio.Task] = set()
        upstream_send_lock = asyncio.Lock()

        def _freeze_browser_input_disconnect(
            exc: WebSocketDisconnect,
            *,
            observed_monotonic: float | None = None,
        ) -> _BrowserInputDisconnectFact | None:
            if principal is None:
                return None
            observed_at = (
                observed_monotonic
                if isinstance(observed_monotonic, float)
                else asyncio.get_running_loop().time()
            )
            close_code = _normalize_websocket_close_code(getattr(exc, "code", None))
            fact = _BrowserInputDisconnectFact(
                kind=_classify_browser_input_close(close_code),
                phase=_world_disconnect_phase(world_session, observed_at),
                close_code=close_code,
                disconnected_at_ms=int(time.time() * 1000),
                observed_monotonic=observed_at,
            )
            frozen = _record_browser_input_disconnect(world_session, fact)
            if frozen is fact:
                observe_world_gateway_browser_input_disconnect(
                    model=metric_model,
                    profile=principal.profile_id,
                    role=principal.role,
                    phase=fact.phase,
                    kind=fact.kind,
                    close_class=_classify_browser_input_close_metric(fact.close_code),
                )
                _log_gateway_trace(
                    trace_id,
                    "gateway.browser_input_disconnected",
                    session_id=session_id,
                    generation_id=generation_id,
                    run_id=principal.run_id,
                    attempt_id=principal.jti,
                    role=principal.role,
                    disconnect_kind=fact.kind,
                    disconnect_phase=fact.phase,
                    websocket_close_code=fact.close_code,
                    disconnected_at_ms=fact.disconnected_at_ms,
                )
            return frozen

        def _freeze_browser_output_detach(
            reason: str,
            *,
            observed_monotonic: float | None = None,
        ) -> _BrowserOutputDetachFact | None:
            if principal is None:
                return None
            observed_at = (
                observed_monotonic
                if isinstance(observed_monotonic, float)
                else asyncio.get_running_loop().time()
            )
            fact = _BrowserOutputDetachFact(
                detach_class=_classify_browser_output_detach(reason),
                phase=_world_disconnect_phase(world_session, observed_at),
                detached_at_ms=int(time.time() * 1000),
                observed_monotonic=observed_at,
            )
            frozen = _record_browser_output_detach(world_session, fact)
            if frozen is fact:
                observe_world_gateway_browser_delivery_detach(
                    model=metric_model,
                    profile=principal.profile_id,
                    role=principal.role,
                    phase=fact.phase,
                    detach_class=fact.detach_class,
                )
                play_deadline = world_session.get("play_deadline_monotonic")
                queue_metrics = route.queue_metrics() if route is not None else {}
                _log_gateway_trace(
                    trace_id,
                    "gateway.browser_delivery_detached",
                    session_id=session_id,
                    generation_id=generation_id,
                    run_id=principal.run_id,
                    attempt_id=principal.jti,
                    role=principal.role,
                    detach_reason=reason,
                    detach_class=fact.detach_class,
                    detach_phase=fact.phase,
                    detached_at_ms=fact.detached_at_ms,
                    stage=fact.phase,
                    before_play_deadline=(
                        isinstance(play_deadline, float) and observed_at < play_deadline
                    ),
                    **queue_metrics,
                )
            return frozen

        def _ensure_browser_output_detach_fact() -> None:
            if (
                principal is not None
                and route is not None
                and route.browser_detached
                and not isinstance(
                    world_session.get("browser_output_detach_fact"),
                    _BrowserOutputDetachFact,
                )
            ):
                _freeze_browser_output_detach(
                    route.browser_detach_reason or "output_detached"
                )

        def _fire_world_terminal_callback() -> None:
            """Freeze and dispatch the one business terminal callback."""
            if (
                world_session.get("terminal_callback_fired")
                or not business_lifecycle_owner
                or world_callbacks is None
                or not world_session.get("attempt_owner")
            ):
                return
            _ensure_browser_output_detach_fact()
            # Set the latch before dispatch. WorldCallbacks.fire is already
            # asynchronous; a callback task must be visible before any slow
            # transport or Coordinator cleanup yields to external services.
            world_session["terminal_callback_fired"] = True
            # started is queued after upstream setup and before the sealed init
            # arrives. An invalid init may therefore be established=True, while
            # admission failures that never queued started remain false.
            established = bool(world_session.get("started_reported"))
            terminal = _build_world_terminal_fact(
                world_session,
                assignment_exists=assignment is not None,
                established=established,
            )
            world_session["terminal_fact"] = terminal
            if terminal.classification_error is not None:
                logger.error(
                    "World gateway terminal classification failed "
                    "run_id=%s classification_error=%s",
                    principal.run_id,
                    terminal.classification_error,
                )

            terminal_at = asyncio.get_running_loop().time()
            terminal_fields: dict[str, Any] = {
                "run_id": principal.run_id,
                "attempt_id": principal.jti,
                "session_id": session_id,
                "generation_id": generation_id,
                "profile_id": principal.profile_id,
                "role": principal.role,
                "outcome": terminal.outcome,
                "fault": terminal.fault,
                "reason": terminal.reason,
                "established": terminal.established,
                "first_media_sent": isinstance(
                    world_session.get("first_media_monotonic"), float
                ),
                "browser_output_detached_at_business_terminal": (
                    terminal.output_detach is not None
                ),
                "session_duration_ms": round(
                    max(0.0, terminal_at - session_opened_monotonic) * 1000
                ),
                "last_received_chunk": playback_ack_window.last_received_chunk,
                "last_rendered_chunk": playback_ack_window.last_rendered_chunk,
            }
            gateway_instance = (os.getenv("HOSTNAME") or "").strip()
            if gateway_instance:
                terminal_fields["gateway_instance"] = gateway_instance[:253]
            first_media_at = world_session.get("first_media_monotonic")
            if isinstance(first_media_at, float):
                terminal_fields["play_elapsed_ms"] = round(
                    max(0.0, terminal_at - first_media_at) * 1000
                )
            if terminal.input_disconnect is not None:
                terminal_fields.update(
                    disconnect_kind=terminal.input_disconnect.kind,
                    disconnect_phase=terminal.input_disconnect.phase,
                    websocket_close_code=terminal.input_disconnect.close_code,
                    disconnected_at_ms=terminal.input_disconnect.disconnected_at_ms,
                )
            if terminal.output_detach is not None:
                terminal_fields.update(
                    browser_output_detach_class_at_business_terminal=(
                        terminal.output_detach.detach_class
                    ),
                    browser_output_detach_phase_at_business_terminal=(
                        terminal.output_detach.phase
                    ),
                    browser_output_detached_at_ms=(
                        terminal.output_detach.detached_at_ms
                    ),
                )
            if world_session.get("system_error_type"):
                terminal_fields["system_error_type"] = str(
                    world_session["system_error_type"]
                )[:120]
            _log_gateway_trace(
                trace_id,
                "gateway.world_session_terminal",
                **terminal_fields,
            )

            if terminal.outcome == "ended":
                assert terminal.ended_at_ms is not None
                world_callbacks.ended(
                    principal.run_id,
                    principal.jti,
                    trace_id,
                    terminal.reason,
                    ended_at_ms=terminal.ended_at_ms,
                )
                return

            disconnect_fields: dict[str, Any] = {}
            if terminal.reason == "USER_LEFT" and terminal.input_disconnect is not None:
                disconnect_fields = {
                    "disconnect_kind": terminal.input_disconnect.kind,
                    "disconnect_phase": terminal.input_disconnect.phase,
                    "websocket_close_code": terminal.input_disconnect.close_code,
                    "disconnected_at_ms": terminal.input_disconnect.disconnected_at_ms,
                }
            world_callbacks.aborted(
                principal.run_id,
                principal.jti,
                trace_id,
                fault=terminal.fault,
                reason=terminal.reason,
                established=terminal.established,
                **disconnect_fields,
            )

        try:
            if classic_whitebox and whitebox_config is not None and principal is None:
                # The legacy Whitebox path otherwise reserves H3 before seeing
                # init. Preflight only this route, then replay the exact bytes once.
                from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import (
                    WhiteboxDiagnostic,
                    public_error,
                )
                from sglang.multimodal_gen.runtime.realtime.whitebox.model_geometry import (
                    classic_source_spec,
                )

                try:
                    total_bytes = 0
                    async with asyncio.timeout(30):
                        while True:
                            payload = await _receive_browser(websocket)
                            if not isinstance(payload, bytes):
                                raise WhiteboxDiagnostic(
                                    "INVALID_CONTROL_MESSAGE", "validation"
                                )
                            total_bytes += len(payload)
                            if (
                                total_bytes > 8 * 1024**2 + 4096
                                or len(prefetched_whitebox_init) >= 17
                            ):
                                raise WhiteboxDiagnostic(
                                    "INVALID_CONTROL_MESSAGE", "validation"
                                )
                            initial = msgspec.msgpack.decode(payload)
                            if not isinstance(initial, dict):
                                raise WhiteboxDiagnostic(
                                    "INVALID_CONTROL_MESSAGE", "validation"
                                )
                            prefetched_whitebox_init.append(payload)
                            # Existing clients may send bounded decoder metrics
                            # before init. Replay them through the original observer.
                            if initial.get("type") in {
                                "client_metric",
                                "client_metric_batch",
                            }:
                                continue
                            if initial.get("type") != "init" or not isinstance(
                                initial.get("whitebox"), dict
                            ):
                                raise WhiteboxDiagnostic(
                                    "INVALID_CONTROL_MESSAGE", "validation"
                                )
                            classic_source_spec(
                                initial["whitebox"].get("resolution"),
                                whitebox_config.spec,
                            )
                            break
                except (ValueError, TimeoutError) as error:
                    diagnostic = public_error(
                        error,
                        trace_id,
                        default="INVALID_CONTROL_MESSAGE",
                        stage="validation",
                    )
                    await asyncio.wait_for(
                        sender.error(diagnostic.pop("content"), **diagnostic),
                        browser_send_timeout_s,
                    )
                    await asyncio.wait_for(
                        websocket.close(code=1008, reason=diagnostic["code"]),
                        browser_send_timeout_s,
                    )
                    return
            admitted_at = time.perf_counter()
            _log_gateway_trace(trace_id, "gateway.ws_accepted", session_id=session_id)
            if principal is not None:
                # The local replay guard covers one Gateway process; the
                # Coordinator conditional write is the global owner decision
                # across replicas. Keep the fence for at least five minutes so
                # an ended owner cannot reuse the same one-time token.
                await coordinator.claim_attempt(
                    attempt_id=_world_coordinator_attempt_id(principal),
                    owner_id=f"{session_id}:{generation_id}",
                    expires_at=max(
                        float(principal.exp) + 60.0,
                        time.time() + 300.0,
                    ),
                )
                world_session["attempt_owner"] = True
            try:
                async with admission_gate.waiter():
                    assignment = await coordinator.admit(
                        # D1: authorized sessions use the internal user id from
                        # the token. The wsvc: prefix isolates them from showcase.
                        # Key the lease on the account-level pseudonym rather
                        # than the per-run one: the coordinator's one-lease-per-
                        # user rule is meant to bound a person, and the per-run
                        # pseudonym changes with every run, so clearing the
                        # business-side ledger once lets the same person take a
                        # second seat while the old one still holds a GPU.
                        # Tokens without that claim fall back to the per-run
                        # pseudonym, so behaviour is unchanged before the
                        # world-service rollout.
                        user_id=(
                            _world_coordinator_user_id(principal)
                            if principal is not None
                            else _user_id(websocket)
                        ),
                        session_id=session_id,
                        generation_id=generation_id,
                        model_revision=selected_model_revision,
                        vae_fingerprint=selected_vae_fingerprint,
                        wait_for_capacity=True,
                        trace_id=trace_id,
                    )
            except BaseException as exc:
                observe_stage_seconds(
                    "gateway_route_admit",
                    time.perf_counter() - admitted_at,
                    service="gateway",
                    model=metric_model,
                    result=result_from_exception(exc),
                    scope="request",
                )
                raise
            observe_stage_seconds(
                "gateway_route_admit",
                time.perf_counter() - admitted_at,
                service="gateway",
                model=metric_model,
                result="success",
                scope="request",
            )
            _log_gateway_trace(
                trace_id,
                "gateway.coordinator_admit_complete",
                session_id=session_id,
                coordinator_admit_ms=round(
                    (time.perf_counter() - admitted_at) * 1000, 3
                ),
                denoiser_worker_id=assignment.denoiser.worker_id,
                vae_worker_id=assignment.vae.worker_id if assignment.vae else None,
            )
            if (
                business_lifecycle_owner
                and principal.record_video
                and world_session.get("attempt_owner")
            ):
                # Recording is a bounded observer of output accepted by this
                # route. It never sits between the VAE and browser. Only the
                # attempt owner that completed the Coordinator conditional
                # write may create a recording and its terminal callback.
                if session_recording_manager is None:
                    raise RuntimeError("world recording manager is not configured")
                recorder = session_recording_manager.start(
                    run_id=principal.run_id,
                    attempt_id=principal.jti,
                    session_id=session_id,
                    generation_id=generation_id,
                    trace_id=trace_id,
                )
            route = await registry.register(
                session_id,
                generation_id,
                token=output_token,
                trace_id=trace_id,
                recorder=recorder,
                on_browser_detach=(
                    (
                        lambda reason, observed_at: _freeze_browser_output_detach(
                            reason,
                            observed_monotonic=observed_at,
                        )
                    )
                    if principal is not None
                    else None
                ),
            )
            upstream_url = build_denoiser_url(
                assignment.denoiser.endpoint,
                session_id=session_id,
                generation_id=generation_id,
                coordinator_token=assignment.token,
                worker_epoch=assignment.denoiser.worker_epoch,
                vae_url=assignment.vae.endpoint if assignment.vae else "",
                vae_worker_epoch=assignment.vae.worker_epoch if assignment.vae else "",
                output_url=internal_output_url,
                output_token=output_token,
                trace_id=trace_id,
            )
            upstream = await connect_factory(
                upstream_url,
                max_size=whitebox_config.max_message_bytes
                if whitebox_config is not None
                else None,
                **({"max_queue": 1} if whitebox_config is not None else {}),
                compression=None,
                open_timeout=10,
                close_timeout=2,
                ping_interval=20,
                ping_timeout=20,
            )
            if (
                whitebox_config is not None
                and principal is None
                and assignment.vae is None
            ):
                from sglang.multimodal_gen.runtime.realtime.whitebox.session import (
                    WhiteboxUpstream,
                )

                upstream = WhiteboxUpstream(
                    upstream,
                    config=whitebox_config,
                    archives=whitebox_archives,
                    reports=whitebox_reports,
                    report=whitebox_report,
                    trace_id=trace_id,
                    session_id=session_id,
                    worker_metadata={
                        "exclusive_session_id": session_id,
                        "generation_id": assignment.generation_id,
                        "model_revision": assignment.denoiser.model_revision,
                        "routing_revision": assignment.denoiser.routing_revision,
                        "worker_id": assignment.denoiser.worker_id,
                        "slot_index": assignment.denoiser.slot_index,
                    },
                    user_id=assignment.user_id,
                )
            _log_gateway_trace(
                trace_id,
                "gateway.denoiser_connected",
                session_id=session_id,
            )
            if principal is not None:
                loop = asyncio.get_running_loop()
                admitted_monotonic = loop.time()
                admitted_at_ms = int(time.time() * 1000)

            def _world_business_input_fenced() -> bool:
                if world_session.get("finishing"):
                    return True
                play_deadline = world_session.get("play_deadline_monotonic")
                return isinstance(play_deadline, float) and (
                    asyncio.get_running_loop().time() >= play_deadline
                )

            async def _upstream_send(
                data,
                *,
                transport_control: bool = False,
            ) -> bool:
                # Direction/skill tasks and browser forwarding can write the same
                # upstream connection concurrently. websockets does not permit
                # concurrent send calls, so serialize them with a lock.
                async with upstream_send_lock:
                    if _world_business_input_fenced() and not transport_control:
                        return False
                    await upstream.send(data)
                    return True

            async def _direction_rewrite(
                event_id: int | None, text: str, baseline: str
            ) -> tuple[str, str]:
                # Rewrite is delegated to world-service, which owns credentials
                # and prompt assets. It uses the same HMAC-signed internal path
                # as lifecycle callbacks.
                try:
                    data = await world_callbacks.rewrite(
                        principal.run_id,
                        event_id,
                        principal.role,
                        text,
                        baseline,
                    )
                except Exception:
                    logger.warning(
                        "direction rewrite failed run_id=%s",
                        principal.run_id,
                        exc_info=True,
                    )
                    raise
                # Default change_type to one_time. That self-heals after the
                # revert if guessed wrong, while an accidental persistent mark
                # would burn a transient effect into the baseline.
                return str(data["prompt"]), str(data.get("change_type") or "one_time")

            async def _direction_dispatch(
                prompt: str, event_id, kind: str = "prompt"
            ) -> None:
                # The engine reads RealtimeEvent.payload. Passing event_id through
                # lets control_ack, frame_batch, and chunk_telemetry correlate
                # back to this player input. kind picks the transition and is
                # decided server side (compiled world), never by the browser.
                fields: dict[str, Any] = {"kind": kind, "payload": prompt}
                if event_id is not None:
                    fields["event_id"] = event_id
                await _upstream_send(encode_message("event", **fields))

            async def _direction_notify(event_id, status: str) -> None:
                # rewriting/failed/superseded are gateway-produced browser events,
                # not engine events, so they bypass the engine allowlist.
                await send_browser_with_trace(
                    encode_message(
                        "direction_status", event_id=event_id, status=status
                    ),
                    send_source="direction",
                )

            def _spawn_direction(coro, label: str) -> None:
                # Rewrite takes roughly one to two seconds, so run it in an
                # independent task and do not block the browser_to_worker receive
                # loop. Keep strong references in direction_tasks and cancel all
                # in finally. Exceptions are logged only; the main task group owns
                # session lifetime and one rewrite failure should not tear down
                # the session.
                if len(direction_tasks) >= _WORLD_MAX_DIRECTION_TASKS:
                    # Dropped silently, like every other out-of-whitelist frame:
                    # spam should stop working, not help the sender break their
                    # own session.
                    coro.close()
                    _log_gateway_trace(
                        trace_id,
                        "gateway.direction_dropped",
                        session_id=session_id,
                        reason="too_many_inflight",
                    )
                    return
                task = asyncio.create_task(coro, name=f"gateway-direction-{label}")
                direction_tasks.add(task)

                def _done(t: asyncio.Task) -> None:
                    direction_tasks.discard(t)
                    if not t.cancelled() and t.exception() is not None:
                        logger.warning(
                            "direction task failed run_id=%s",
                            principal.run_id if principal else "-",
                            exc_info=t.exception(),
                        )

                task.add_done_callback(_done)

            async def browser_to_worker():
                nonlocal expected_last_chunk, prefetched_whitebox_init
                finish_tail_upstream_closed = False
                try:
                    while True:
                        if prefetched_whitebox_init:
                            payload = prefetched_whitebox_init.pop(0)
                        else:
                            payload = await _receive_browser(websocket)
                        if principal is not None:
                            forwarded = await _world_browser_frame(payload)
                            if forwarded is None:
                                continue
                            payload = forwarded
                        control = None
                        if isinstance(payload, bytes):
                            try:
                                control = decode_message(payload)
                            except ProtocolViolation:
                                pass
                        if isinstance(control, dict) and control.get("type") in {
                            "client_metric",
                            "client_metric_batch",
                        }:
                            _observe_gateway_client_metric(control, model=metric_model)
                            if whitebox_report is not None:
                                events = (
                                    control.get("events", [])
                                    if control["type"] == "client_metric_batch"
                                    else [control]
                                )
                                for event in events[:32]:
                                    if isinstance(event, dict) and event.get(
                                        "stage"
                                    ) in {
                                        "client_video_decode",
                                        "client_render_wait",
                                        "client_receive_queue",
                                    }:
                                        whitebox_report.observe(
                                            dict(
                                                event="client." + event["stage"],
                                                duration_ms=event.get("duration_ms"),
                                            )
                                        )
                            continue
                        if isinstance(control, dict) and expected_last_chunk is None:
                            if (
                                isinstance(control, dict)
                                and control.get("type") == "init"
                            ):
                                max_chunks = int(control.get("max_chunks") or 0)
                                if max_chunks > 0:
                                    expected_last_chunk = max_chunks - 1
                        if isinstance(payload, bytes):
                            await playback_ack_window.observe_browser_message(payload)
                        tail_transport = (
                            isinstance(control, dict)
                            and control.get("type") == "event"
                            and control.get("kind") in {"playback_ack", "heartbeat"}
                        )
                        if (
                            finish_tail_upstream_closed
                            and world_session.get("finishing")
                            and tail_transport
                        ):
                            continue
                        try:
                            await _upstream_send(
                                payload,
                                transport_control=tail_transport,
                            )
                        except ConnectionClosedOK:
                            # After server_finish, the Denoiser may close before
                            # the browser's next transport-only update. That says
                            # nothing about the browser connection and must not
                            # let the input task manufacture USER_LEFT.
                            if not (
                                principal is not None
                                and world_session.get("finishing")
                                and tail_transport
                            ):
                                raise
                            finish_tail_upstream_closed = True
                            _log_gateway_trace(
                                trace_id,
                                "gateway.finish_tail_upstream_closed",
                                session_id=session_id,
                                generation_id=generation_id,
                                stage="finishing",
                                message_kind=control.get("kind"),
                            )
                except WebSocketDisconnect as exc:
                    observed_at = asyncio.get_running_loop().time()
                    world_session.setdefault("browser_finished_monotonic", observed_at)
                    _freeze_browser_input_disconnect(
                        exc,
                        observed_monotonic=observed_at,
                    )
                    raise
                except ConnectionClosedOK:
                    # Preserve the legacy showcase's existing normal-close
                    # behavior. Authorized sessions must keep upstream closure
                    # distinct from a FastAPI browser disconnect.
                    if principal is None:
                        return
                    raise
                finally:
                    world_session.setdefault(
                        "browser_finished_monotonic",
                        asyncio.get_running_loop().time(),
                    )

            async def _world_browser_frame(payload):
                """Handle inbound frames for authorized sessions.

                Returns bytes to forward to the engine, or None to drop. The
                browser may only forward the sealed session payload as the first
                frame, trigger a skill by id, or send allowlisted interaction
                events. Everything else is dropped; the allowlist prevents future
                engine parameters from becoming injectable by default.
                """
                if not isinstance(payload, bytes):
                    # Drop all text frames; constraints are defined over binary frames.
                    _log_gateway_trace(
                        trace_id,
                        "gateway.world_frame_rejected",
                        session_id=session_id,
                        reason="text_frame",
                    )
                    return None

                if _world_business_input_fenced():
                    # Business controls are fenced at T_end, but transport
                    # progress must keep flowing while the terminal media is
                    # draining. Observe/forward only playback ACK and heartbeat;
                    # in particular a browser cannot spoof server_finish.
                    try:
                        finishing_control = decode_message(payload)
                    except ProtocolViolation:
                        finishing_control = None
                    if (
                        isinstance(finishing_control, dict)
                        and finishing_control.get("type") == "event"
                        and finishing_control.get("kind")
                        in {"playback_ack", "heartbeat"}
                    ):
                        return payload
                    _log_gateway_trace(
                        trace_id,
                        "gateway.world_frame_rejected",
                        session_id=session_id,
                        reason="session_finishing",
                    )
                    return None

                if not world_session["init_verified"]:
                    # The first frame must be the sealed payload. Unseal failures,
                    # including tampering or copying across runs or roles, close
                    # the session.
                    init_message = payload_sealer.open(
                        payload,
                        session_payload_aad(principal.run_id, principal.role),
                    )
                    skills = {}
                    ext = init_message.pop(WORLD_EXT_KEY, None)
                    if isinstance(ext, dict):
                        for item in ext.get("skills") or []:
                            if isinstance(item, dict) and item.get("id"):
                                skills[str(item["id"])] = item
                    world_session["skills"] = skills
                    world_session["init_verified"] = True
                    # Direction orchestrator: baseline seed comes from init
                    # prompt, while timeline entries fold into baseline as
                    # chunk_telemetry advances.
                    nonlocal directions
                    baseline, schedule = parse_init_directions(init_message)
                    directions = DirectionCoordinator(
                        baseline=baseline,
                        schedule=schedule,
                        rewrite=_direction_rewrite,
                        dispatch=_direction_dispatch,
                        notify=_direction_notify,
                    )
                    # Strip gateway-only extensions before forwarding to the
                    # engine. encode_message has signature (message_type, *,
                    # **fields) and inserts version itself, so both keys must be
                    # removed from fields to avoid missing positional argument or
                    # duplicate keyword errors.
                    init_message.pop("version", None)
                    message_type = init_message.pop("type", "init")
                    return encode_message(message_type, **init_message)

                try:
                    control = decode_message(payload)
                except ProtocolViolation:
                    # Drop undecodable frames; never allow by default just
                    # because we cannot parse the frame.
                    _log_gateway_trace(
                        trace_id,
                        "gateway.world_frame_rejected",
                        session_id=session_id,
                        reason="undecodable",
                    )
                    return None
                if not isinstance(control, dict):
                    return None

                msg_type = control.get("type")
                if msg_type in _WORLD_PASSTHROUGH_TYPES:
                    return payload
                if msg_type != "event":
                    _log_gateway_trace(
                        trace_id,
                        "gateway.world_frame_rejected",
                        session_id=session_id,
                        reason=f"type:{msg_type}",
                    )
                    return None

                kind = control.get("kind")
                if kind == "skill":
                    # Skill activation: the browser sends only an id. The gateway
                    # looks up the prompt, so the prompt never passes through the
                    # browser.
                    payload_id = None
                    if isinstance(control.get("payload"), dict):
                        payload_id = control["payload"].get("id")
                    skill = (world_session.get("skills") or {}).get(
                        str(control.get("id") or payload_id or "")
                    )
                    if skill is None or directions is None:
                        _log_gateway_trace(
                            trace_id,
                            "gateway.world_frame_rejected",
                            session_id=session_id,
                            reason="unknown_skill",
                        )
                        return None
                    # Skill prompts are authored as full scene descriptions,
                    # equivalent to pre-rewritten directions, so use the same
                    # apply path. This also fixes two older issues: the prompt was
                    # sent under the wrong field (prompt= while the engine reads
                    # payload), and one_time skills had no revert so their effect
                    # could become permanent.
                    _spawn_direction(
                        directions.apply(
                            str(skill.get("prompt") or ""),
                            str(skill.get("change_type") or "one_time"),
                            control.get("event_id"),
                            str(skill.get("kind") or "prompt"),
                        ),
                        "skill",
                    )
                    return None
                if kind == "direction":
                    # Raw player instruction. The independent rewrite task later
                    # dispatches kind:"prompt"; forward no bytes here.
                    if not principal.allow_free_prompt:
                        # Worlds with free input disabled silently drop instead of
                        # disconnecting. Malicious injection should fail, not help
                        # the attacker end someone else's session.
                        _log_gateway_trace(
                            trace_id,
                            "gateway.free_prompt_dropped",
                            session_id=session_id,
                        )
                        return None
                    text = str(control.get("text") or "").strip()
                    event_id = control.get("event_id")
                    event_id_valid = (
                        type(event_id) is int
                        and event_id > 0
                        and event_id <= _WORLD_DIRECTION_EVENT_ID_MAX
                    )
                    if (
                        not text
                        or len(text) > _WORLD_DIRECTION_MAX_CHARS
                        or (principal.role == "comparison" and not event_id_valid)
                        or directions is None
                    ):
                        _log_gateway_trace(
                            trace_id,
                            "gateway.world_frame_rejected",
                            session_id=session_id,
                            reason="direction_invalid",
                        )
                        return None
                    # Only comparison requires a stable cross-socket id. A valid
                    # primary id participates as the rewrite owner; primary-only
                    # legacy clients may omit or misuse event_id, in which case
                    # the callback retains its old non-coordinated request shape.
                    rewrite_event_id = event_id if event_id_valid else None
                    _spawn_direction(
                        directions.submit(rewrite_event_id, text), "rewrite"
                    )
                    return None
                if kind in _WORLD_ALLOWED_EVENT_KINDS:
                    return payload
                _log_gateway_trace(
                    trace_id,
                    "gateway.world_frame_rejected",
                    session_id=session_id,
                    reason=f"kind:{kind}",
                )
                return None

            async def worker_to_browser():
                try:
                    while True:
                        wire = await upstream.recv()
                        if isinstance(wire, str):
                            raise ProtocolViolation(
                                "Denoiser control messages must be binary"
                            )
                        if principal is not None:
                            control_message = msgspec.msgpack.decode(wire)
                            if (
                                isinstance(control_message, dict)
                                and control_message.get("type") == "error"
                            ):
                                # The engine's content is unrestricted internal
                                # text. Do not copy it into logs or contracts;
                                # let the shared world error handler emit the
                                # fixed product code/message and callback.
                                logger.error(
                                    "World engine reported a session error run_id=%s",
                                    principal.run_id,
                                )
                                raise RuntimeError("world engine session failed")
                        local_message = (
                            msgspec.msgpack.decode(wire)
                            if assignment.vae is None
                            else None
                        )
                        local_allowed = isinstance(
                            local_message, dict
                        ) and local_message.get("type") in {
                            "session.ready",
                            "frame_batch",
                            "session.completed",
                            "source.accepted",
                            "source.consumed",
                        }
                        if (
                            whitebox_config is not None
                            and principal is None
                            and isinstance(local_message, dict)
                        ):
                            local_allowed = local_allowed or local_message.get(
                                "type"
                            ) in {
                                "whitebox.capture",
                                "whitebox.control",
                                "whitebox.frame",
                                "whitebox.pong",
                                "whitebox.input_received",
                            }
                        if not local_allowed and not worker_message_allowed(wire):
                            message_type = worker_message_type(wire)
                            raise ProtocolViolation(
                                f"Denoiser emitted forbidden message: {message_type}"
                            )
                        if directions is not None:
                            # Passing telemetry also advances the baseline by
                            # folding timeline entries into it. Worker control
                            # messages are raw msgpack without version fields, so
                            # use the same raw decode path as worker_message_allowed.
                            # decode_message would treat them as protocol violations.
                            # Upstream control messages are tiny, so the extra
                            # decode is negligible.
                            telemetry = msgspec.msgpack.decode(wire)
                            if (
                                isinstance(telemetry, dict)
                                and telemetry.get("type") == "chunk_telemetry"
                                and isinstance(telemetry.get("chunk_index"), int)
                            ):
                                directions.observe_chunk(telemetry["chunk_index"])
                        await send_browser_with_trace(wire, send_source="denoiser")
                        if local_allowed and local_message["type"] == "frame_batch":
                            delivered = getattr(upstream, "media_delivered", None)
                            if delivered is not None:
                                delivered()
                        if (
                            local_allowed
                            and local_message["type"] == "session.completed"
                        ):
                            if whitebox_config is not None and principal is None:
                                # send() only reaches the proxy/transport buffer.
                                # The SDK closes after receiving completed AND
                                # flushing its decoder. Let it do so, consuming
                                # late metrics meanwhile. Share the existing tail
                                # deadline so a stalled peer cannot extend a slot.
                                deadline = upstream.drain_deadline_monotonic
                                remaining = max(
                                    0.0,
                                    deadline - asyncio.get_running_loop().time(),
                                )
                                await asyncio.wait(
                                    {browser_input_task}, timeout=remaining
                                )
                            return
                except ConnectionClosedOK:
                    return
                finally:
                    world_session.setdefault(
                        "worker_finished_monotonic",
                        asyncio.get_running_loop().time(),
                    )

            async def output_to_browser():
                try:
                    while True:
                        output = await route.get_output()
                        wire = output.wire
                        codec, scope = _metric_labels_from_wire(wire)
                        observe_stage_seconds(
                            "output_pacing_queue",
                            time.monotonic() - output.enqueued_at,
                            service="gateway",
                            model=metric_model,
                            result="success",
                            codec=codec,
                            scope=scope,
                        )
                        delivered = False
                        try:
                            delivered = await send_browser_with_trace(
                                wire,
                                send_source="vae",
                                queue_fields=route.queue_metrics(),
                            )
                        finally:
                            route.task_done(delivered=delivered)
                except OutputDeliveryDetached:
                    _freeze_browser_output_detach(
                        route.browser_detach_reason or "output_detached"
                    )
                finally:
                    world_session.setdefault(
                        "output_finished_monotonic",
                        asyncio.get_running_loop().time(),
                    )

            async def send_browser_with_trace(
                wire: bytes,
                *,
                send_source: str,
                queue_fields: dict[str, Any] | None = None,
            ) -> bool:
                send_started = time.perf_counter()
                codec, scope = _metric_labels_from_wire(wire)
                send_fields = _browser_send_trace_fields(wire)
                queue_fields = queue_fields or {}
                if route.browser_detached:
                    _ensure_browser_output_detach_fact()
                    return False
                if not await playback_ack_window.allow_output(wire):
                    observe_stage_seconds(
                        "websocket_build_write",
                        time.perf_counter() - send_started,
                        service="gateway",
                        model=metric_model,
                        result="cancelled",
                        codec=codec,
                        scope=scope,
                    )
                    _log_gateway_trace(
                        trace_id,
                        "gateway.browser_send_dropped",
                        session_id=session_id,
                        generation_id=generation_id,
                        send_source=send_source,
                        drop_reason="playback_ack_window",
                        last_received_chunk=playback_ack_window.last_received_chunk,
                        last_rendered_chunk=playback_ack_window.last_rendered_chunk,
                        **queue_fields,
                        **send_fields,
                    )
                    return True
                try:
                    # D5 ghost cleanup: a dead peer may never drain its TCP write
                    # buffer, causing this await to hang forever because keepalive
                    # close is ineffective while the buffer is not empty. Bounding
                    # the send turns that case into a normal task completion and
                    # lets existing finally cleanup release the reservation.
                    await asyncio.wait_for(
                        sender.send(wire), timeout=browser_send_timeout_s
                    )
                except Exception as exc:
                    browser_send_ms = round(
                        (time.perf_counter() - send_started) * 1000, 3
                    )
                    observe_stage_seconds(
                        "websocket_build_write",
                        browser_send_ms / 1000.0,
                        service="gateway",
                        model=metric_model,
                        result=result_from_exception(exc),
                        codec=codec,
                        scope=scope,
                    )
                    _log_gateway_trace(
                        trace_id,
                        "gateway.browser_send_complete",
                        session_id=session_id,
                        generation_id=generation_id,
                        send_source=send_source,
                        send_ok=False,
                        send_ms=browser_send_ms,
                        browser_send_ms=browser_send_ms,
                        error_type=type(exc).__name__,
                        **queue_fields,
                        **send_fields,
                    )
                    if not first_media_event.is_set():
                        raise
                    reason = f"browser_send_{type(exc).__name__}"
                    if route.browser_detached or route.detach_browser(reason):
                        detached_at = asyncio.get_running_loop().time()
                        _freeze_browser_output_detach(
                            route.browser_detach_reason or reason,
                            observed_monotonic=detached_at,
                        )
                        return False
                    raise
                if (
                    principal is not None
                    and not first_media_event.is_set()
                    and _is_nonempty_real_media(wire)
                ):
                    # Freeze the monotonic and epoch facts in the same successful
                    # send turn. The epoch values are reused verbatim by every
                    # lifecycle callback retry; T_close must never replace them.
                    first_media_monotonic = asyncio.get_running_loop().time()
                    first_media_at_ms = int(time.time() * 1000)
                    world_session["first_media_monotonic"] = first_media_monotonic
                    world_session["first_media_at_ms"] = first_media_at_ms
                    world_session["play_deadline_monotonic"] = (
                        first_media_monotonic + principal.play_seconds
                    )
                    world_session["ended_at_ms"] = (
                        first_media_at_ms + principal.play_seconds * 1000
                    )
                    # From this point the player has an established media
                    # session. Browser delivery may detach without closing the
                    # source route or preventing a complete recording.
                    route.allow_browser_detach()
                    first_media_event.set()
                    observe_world_gateway_first_media(
                        model=metric_model,
                        profile=principal.profile_id,
                        role=principal.role,
                    )
                    _log_gateway_trace(
                        trace_id,
                        "gateway.first_media_sent",
                        session_id=session_id,
                        generation_id=generation_id,
                        play_seconds=principal.play_seconds,
                        first_media_at_ms=first_media_at_ms,
                        **send_fields,
                    )
                    if business_lifecycle_owner and world_callbacks is not None:
                        world_callbacks.first_frame(
                            principal.run_id,
                            principal.jti,
                            trace_id,
                            first_media_at_ms,
                        )
                browser_send_ms = round((time.perf_counter() - send_started) * 1000, 3)
                observe_stage_seconds(
                    "websocket_build_write",
                    browser_send_ms / 1000.0,
                    service="gateway",
                    model=metric_model,
                    result="success",
                    codec=codec,
                    scope=scope,
                )
                _log_gateway_trace(
                    trace_id,
                    "gateway.browser_send_complete",
                    session_id=session_id,
                    generation_id=generation_id,
                    send_source=send_source,
                    send_ok=True,
                    send_ms=browser_send_ms,
                    browser_send_ms=browser_send_ms,
                    **queue_fields,
                    **send_fields,
                )
                gateway_queue_messages = int(
                    queue_fields.get("gateway_queue_messages") or 0
                )
                gateway_queue_bytes = int(queue_fields.get("gateway_queue_bytes") or 0)
                if (
                    browser_send_ms >= 100.0
                    or gateway_queue_messages >= 128
                    or gateway_queue_bytes >= 8 * 1024 * 1024
                ):
                    logger.warning(
                        "gateway_browser_send_slow trace_id=%s session_id=%s "
                        "source=%s type=%s send_ms=%.3f queue_messages=%s "
                        "queue_bytes=%s oldest_frame_age_ms=%s wire_bytes=%s",
                        trace_id,
                        session_id,
                        send_source,
                        send_fields.get("message_type"),
                        browser_send_ms,
                        gateway_queue_messages,
                        gateway_queue_bytes,
                        queue_fields.get("gateway_oldest_frame_age_ms", 0),
                        send_fields.get("wire_bytes", 0),
                    )
                return True

            async def renew_lease():
                nonlocal assignment
                try:
                    while True:
                        await asyncio.sleep(lease_renew_interval_s)
                        if whitebox_report is not None:
                            from sglang.multimodal_gen.runtime.realtime.whitebox.metrics import (
                                stage,
                            )

                            with stage("lease_renew", trace_id):
                                assignment = await coordinator.renew(assignment)
                        else:
                            assignment = await coordinator.renew(assignment)
                finally:
                    world_session.setdefault(
                        "lease_finished_monotonic",
                        asyncio.get_running_loop().time(),
                    )

            browser_input_task = asyncio.create_task(
                browser_to_worker(), name="gateway-browser-input"
            )
            worker_control_task = asyncio.create_task(
                worker_to_browser(), name="gateway-worker-control"
            )
            output_task = asyncio.create_task(
                output_to_browser(), name="gateway-vae-output"
            )
            lease_task = asyncio.create_task(renew_lease(), name="gateway-lease-renew")
            tasks = {
                browser_input_task,
                worker_control_task,
                output_task,
                lease_task,
            }
            if principal is None:
                # Preserve the showcase path's finite max_chunks behavior. The
                # wall-clock protocol applies only to signed platform sessions.
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    exception = task.exception()
                    if exception is not None:
                        raise exception
                if (
                    route is not None
                    and assignment.vae is not None
                    and worker_control_task in done
                ):
                    try:
                        if expected_last_chunk is not None:
                            await asyncio.wait_for(
                                route.wait_until_chunk_completed(expected_last_chunk),
                                timeout=output_drain_timeout_s,
                            )
                        else:
                            await asyncio.wait_for(
                                route.wait_until_output_closed(),
                                timeout=output_drain_timeout_s,
                            )
                        await asyncio.wait_for(
                            route.join(), timeout=output_drain_timeout_s
                        )
                    except TimeoutError:
                        logger.warning(
                            "Gateway media drain timed out for session_id=%s",
                            session_id,
                        )
            else:
                if admitted_monotonic is None or admitted_at_ms is None:
                    raise RuntimeError("authorized session admission time is missing")

                startup_budget_s = (
                    principal.max_lifetime_s
                    - principal.play_seconds
                    - FINISH_TAIL_SECONDS
                )
                hard_deadline = admitted_monotonic + principal.max_lifetime_s
                startup_deadline = admitted_monotonic + startup_budget_s
                hard_timer = asyncio.create_task(
                    _sleep_until(hard_deadline), name="gateway-hard-lifetime"
                )
                startup_timer = asyncio.create_task(
                    _sleep_until(startup_deadline), name="gateway-startup-deadline"
                )
                first_media_waiter = asyncio.create_task(
                    first_media_event.wait(), name="gateway-first-media"
                )
                tasks.update({hard_timer, startup_timer, first_media_waiter})

                if business_lifecycle_owner and world_callbacks is not None:
                    world_session["started_reported"] = True
                    world_callbacks.started(
                        principal.run_id,
                        principal.jti,
                        trace_id,
                        principal.max_lifetime_s,
                        admitted_at_ms,
                    )

                def _raise_unexpected_task(task: asyncio.Task) -> None:
                    if task.cancelled():
                        raise _WorldSessionAbort("GATEWAY_SESSION_FAILED")
                    exception = task.exception()
                    if task is browser_input_task:
                        # Only Starlette's browser-side disconnect proves that
                        # the player left. Upstream closure is a separate system
                        # transport fact and remains visible as its exception.
                        if isinstance(exception, WebSocketDisconnect):
                            disconnect = world_session.get(
                                "browser_input_disconnect_fact"
                            )
                            if not isinstance(disconnect, _BrowserInputDisconnectFact):
                                logger.error(
                                    "Browser input task ended without a frozen "
                                    "disconnect fact run_id=%s",
                                    principal.run_id,
                                )
                                finished_at = world_session.get(
                                    "browser_finished_monotonic"
                                )
                                disconnect = _freeze_browser_input_disconnect(
                                    exception,
                                    observed_monotonic=(
                                        finished_at
                                        if isinstance(finished_at, float)
                                        else None
                                    ),
                                )
                            if (
                                isinstance(disconnect, _BrowserInputDisconnectFact)
                                and disconnect.phase == "finishing"
                            ):
                                return
                            raise _WorldSessionAbort("USER_LEFT", fault="client")
                    if (
                        task is output_task
                        and route.browser_detached
                        and exception is None
                    ):
                        return
                    if exception is not None:
                        raise exception
                    raise _WorldSessionAbort("GATEWAY_SESSION_FAILED")

                def _raise_completed_transport_task(
                    completed: set[asyncio.Task] | None = None,
                    *,
                    at_or_before: float | None = None,
                ) -> None:
                    timestamp_keys = (
                        (browser_input_task, "browser_finished_monotonic"),
                        (worker_control_task, "worker_finished_monotonic"),
                        (output_task, "output_finished_monotonic"),
                        (lease_task, "lease_finished_monotonic"),
                    )
                    for candidate, timestamp_key in timestamp_keys:
                        if not candidate.done():
                            continue
                        if completed is not None and candidate not in completed:
                            continue
                        finished_at = world_session.get(timestamp_key)
                        if (
                            at_or_before is not None
                            and isinstance(finished_at, float)
                            and finished_at > at_or_before
                        ):
                            continue
                        _raise_unexpected_task(candidate)

                # Startup: only a successfully delivered real media message can
                # start play time. init/ready/empty/dropped output does not win.
                startup_done, _ = await asyncio.wait(
                    {
                        browser_input_task,
                        worker_control_task,
                        output_task,
                        lease_task,
                        first_media_waiter,
                        startup_timer,
                        hard_timer,
                    },
                    return_when=asyncio.FIRST_COMPLETED,
                )
                first_media_monotonic = world_session.get("first_media_monotonic")
                if not isinstance(first_media_monotonic, float):
                    _raise_completed_transport_task(at_or_before=startup_deadline)
                    if asyncio.get_running_loop().time() >= startup_deadline:
                        raise _WorldSessionAbort("STARTUP_TIMEOUT")
                    if startup_timer in startup_done:
                        raise _WorldSessionAbort("STARTUP_TIMEOUT")
                    if hard_timer in startup_done:
                        raise _WorldSessionAbort("MAX_LIFETIME_REACHED")
                    _raise_completed_transport_task(startup_done)
                    raise RuntimeError("first media time is missing")
                if first_media_monotonic > startup_deadline:
                    # The event and timeout may wake in the same loop turn. The
                    # frozen media fact, rather than set iteration order, decides
                    # whether startup met its budget.
                    raise _WorldSessionAbort("STARTUP_TIMEOUT")
                if asyncio.get_running_loop().time() >= hard_deadline:
                    raise _WorldSessionAbort("MAX_LIFETIME_REACHED")
                startup_timer.cancel()

                play_deadline = world_session.get("play_deadline_monotonic")
                if not isinstance(play_deadline, float):
                    raise RuntimeError("play deadline is missing")
                play_timer = asyncio.create_task(
                    _sleep_until(play_deadline), name="gateway-play-deadline"
                )
                tasks.add(play_timer)
                while True:
                    play_waiters = {
                        browser_input_task,
                        worker_control_task,
                        lease_task,
                        play_timer,
                        hard_timer,
                    }
                    if not output_task.done():
                        play_waiters.add(output_task)
                    play_done, _ = await asyncio.wait(
                        play_waiters,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    _raise_completed_transport_task(at_or_before=play_deadline)
                    now_monotonic = asyncio.get_running_loop().time()
                    if now_monotonic >= hard_deadline:
                        raise _WorldSessionAbort("MAX_LIFETIME_REACHED")
                    if now_monotonic >= play_deadline:
                        break
                    if play_timer in play_done:
                        raise RuntimeError("play timer woke before its deadline")
                    _raise_completed_transport_task(play_done)

                finish_deadline = min(
                    play_deadline + FINISH_TAIL_SECONDS,
                    hard_deadline,
                )
                nominal_finish_deadline = play_deadline + FINISH_TAIL_SECONDS

                # Enter finishing without completing the outer session. Fence
                # browser sends at the upstream lock, discard rewrite results,
                # then deliver the one server-only finish control. The send is
                # bounded by the same terminal deadline: a stalled transport
                # must not let the lock wait extend either the five-second tail
                # or the overall hard lifetime.
                world_session["finishing"] = True
                if directions is not None:
                    directions.close()
                await _cancel_tasks(set(direction_tasks))
                finish_send_budget = finish_deadline - asyncio.get_running_loop().time()
                if finish_send_budget <= 0:
                    raise _WorldSessionAbort(
                        _world_finish_outcome(
                            now=finish_deadline,
                            finish_deadline=finish_deadline,
                            hard_deadline=hard_deadline,
                            nominal_finish_deadline=nominal_finish_deadline,
                            worker_succeeded=False,
                            worker_finished_at=None,
                            final_completed_at=None,
                        )
                        or "FINISH_TIMEOUT"
                    )
                try:
                    finish_send_started_at = await asyncio.wait_for(
                        _send_server_finish(
                            upstream,
                            upstream_send_lock,
                            encode_message(
                                "event",
                                kind="server_finish",
                                payload={"reason": "play_duration_elapsed"},
                            ),
                        ),
                        timeout=finish_send_budget,
                    )
                except TimeoutError as exc:
                    raise _WorldSessionAbort(
                        _world_finish_outcome(
                            now=finish_deadline,
                            finish_deadline=finish_deadline,
                            hard_deadline=hard_deadline,
                            nominal_finish_deadline=nominal_finish_deadline,
                            worker_succeeded=False,
                            worker_finished_at=None,
                            final_completed_at=None,
                        )
                        or "FINISH_TIMEOUT"
                    ) from exc
                if (
                    isinstance(route.final_completed_at, float)
                    and route.final_completed_at < finish_send_started_at
                ):
                    logger.error(
                        "World engine emitted final media before server finish "
                        "run_id=%s",
                        principal.run_id,
                    )
                    _log_gateway_trace(
                        trace_id,
                        "gateway.media_final_too_early",
                        session_id=session_id,
                        generation_id=generation_id,
                        final_completed_at=route.final_completed_at,
                        finish_send_started_at=finish_send_started_at,
                    )
                    raise _WorldSessionAbort("GATEWAY_SESSION_FAILED")
                _log_gateway_trace(
                    trace_id,
                    "gateway.finish_requested",
                    session_id=session_id,
                    generation_id=generation_id,
                    ended_at_ms=world_session.get("ended_at_ms"),
                )

                finish_timer = asyncio.create_task(
                    _sleep_until(finish_deadline), name="gateway-finish-deadline"
                )
                final_waiter = asyncio.create_task(
                    route.wait_until_final_completed(), name="gateway-media-final"
                )
                tasks.update({finish_timer, final_waiter})
                drain_waiter: asyncio.Task | None = None

                while True:
                    worker_exception = None
                    worker_cancelled = worker_control_task.cancelled()
                    if worker_control_task.done() and not worker_cancelled:
                        worker_exception = worker_control_task.exception()

                    final_completed_at = route.final_completed_at
                    if final_completed_at is not None and drain_waiter is None:
                        # Starting join only after a real final avoids accepting
                        # the route's transiently empty pre-final state.
                        drain_waiter = asyncio.create_task(
                            route.join(), name="gateway-output-drain"
                        )
                        tasks.add(drain_waiter)
                    if drain_waiter is not None and drain_waiter.done():
                        if drain_waiter.cancelled():
                            raise _WorldSessionAbort("GATEWAY_SESSION_FAILED")
                        drain_exception = drain_waiter.exception()
                        if drain_exception is not None:
                            raise drain_exception

                    worker_finished_at = world_session.get("worker_finished_monotonic")
                    route_drained_at = route.queue_drained_at
                    now_monotonic = asyncio.get_running_loop().time()
                    finish_outcome = _world_finish_outcome(
                        now=now_monotonic,
                        finish_deadline=finish_deadline,
                        hard_deadline=hard_deadline,
                        nominal_finish_deadline=nominal_finish_deadline,
                        worker_succeeded=(
                            worker_control_task.done()
                            and not worker_cancelled
                            and worker_exception is None
                        ),
                        worker_finished_at=worker_finished_at,
                        final_completed_at=final_completed_at,
                    )

                    # A system task may invalidate the session only if its
                    # frozen failure timestamp precedes the validated final.
                    # This prevents a later tail cleanup error from rewriting
                    # media that already reached encoder EOF and recording.
                    for task, finished_at in (
                        (
                            browser_input_task,
                            world_session.get("browser_finished_monotonic"),
                        ),
                        (output_task, world_session.get("output_finished_monotonic")),
                        (lease_task, world_session.get("lease_finished_monotonic")),
                    ):
                        if task.done() and _world_failure_precedes_final(
                            failure_at=finished_at,
                            final_completed_at=final_completed_at,
                            finish_deadline=finish_deadline,
                        ):
                            _raise_unexpected_task(task)

                    # A validated final is the stronger data-plane fact. A
                    # worker error that became visible first still fails the
                    # session; a later control-close error cannot invalidate
                    # media that already passed encoder EOF and sealed recording.
                    worker_failed_before_final = (
                        worker_cancelled or worker_exception is not None
                    ) and _world_failure_precedes_final(
                        failure_at=worker_finished_at,
                        final_completed_at=final_completed_at,
                        finish_deadline=finish_deadline,
                    )
                    if worker_failed_before_final:
                        if worker_exception is not None:
                            raise worker_exception
                        raise _WorldSessionAbort("GATEWAY_SESSION_FAILED")

                    if finish_outcome == "completed":
                        if not world_session.get("dynamic_completed"):
                            # Freeze the business and recording outcome as soon
                            # as the source final wins its timestamp race. The
                            # remaining browser drain is presentation cleanup
                            # only and must not delay or rewrite this result.
                            route.commit_recording_final()
                            world_session["dynamic_completed"] = True
                            logger.info(
                                "world session completed graceful finish run_id=%s "
                                "final_chunk_index=%s browser_detached=%s "
                                "route_drained=%s",
                                principal.run_id,
                                route.final_chunk_index,
                                route.browser_detached,
                                route.queue_drained_at is not None,
                            )
                            _log_gateway_trace(
                                trace_id,
                                "gateway.finish_outcome",
                                session_id=session_id,
                                generation_id=generation_id,
                                outcome="completed",
                                worker_finished=isinstance(
                                    world_session.get("worker_finished_monotonic"),
                                    float,
                                ),
                                final_completed=True,
                                browser_drained=route.queue_drained_at is not None,
                                browser_detached=route.browser_detached,
                            )
                            _fire_world_terminal_callback()
                            if recorder is not None:
                                recorder.session_closed()
                                recorder = None

                        delivery_completed = (
                            isinstance(route_drained_at, float)
                            and route_drained_at <= finish_deadline
                        )
                        if (
                            delivery_completed
                            or route.browser_detached
                            or now_monotonic >= finish_deadline
                        ):
                            if not delivery_completed and not route.browser_detached:
                                if route.detach_browser("finish_deadline"):
                                    _freeze_browser_output_detach(
                                        "finish_deadline",
                                        observed_monotonic=now_monotonic,
                                    )
                            elif route.browser_detached:
                                _ensure_browser_output_detach_fact()
                            break

                    if finish_outcome is not None and finish_outcome != "completed":
                        raise _WorldSessionAbort(finish_outcome)

                    finish_waiters: set[asyncio.Task] = {
                        task
                        for task in (
                            browser_input_task,
                            output_task,
                            lease_task,
                            finish_timer,
                            hard_timer,
                        )
                        if not task.done()
                    }
                    if not worker_control_task.done():
                        finish_waiters.add(worker_control_task)
                    if not final_waiter.done():
                        finish_waiters.add(final_waiter)
                    if drain_waiter is not None and not drain_waiter.done():
                        finish_waiters.add(drain_waiter)
                    await asyncio.wait(
                        finish_waiters,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                await _cancel_tasks({output_task})
                close_budget = finish_deadline - asyncio.get_running_loop().time()
                if close_budget > 0 and not route.browser_detached:
                    try:
                        await asyncio.wait_for(
                            websocket.close(code=1000, reason=""),
                            timeout=close_budget,
                        )
                    except Exception:
                        pass
        except _WorldSessionAbort as exc:
            world_session["abort_reason"] = exc.reason
            world_session["abort_fault"] = exc.fault
            if exc.fault == "ours":
                logger.warning(
                    "World gateway session aborted run_id=%s reason=%s",
                    principal.run_id if principal else "-",
                    exc.reason,
                )
                if route is None or not route.browser_detached:
                    try:
                        code, content, close_code = _world_error(
                            "GATEWAY_SESSION_FAILED"
                        )
                        await asyncio.wait_for(
                            sender.error(content, code=code),
                            timeout=browser_send_timeout_s,
                        )
                        await websocket.close(code=close_code, reason=code)
                    except Exception:
                        pass
        except TokenError:
            # Session payload unseal failed because of tampering, cross-run copy,
            # or a non-init payload. Externally expose only a fixed business code;
            # detailed reasons stay in server logs to avoid oracle feedback.
            world_session["init_rejected"] = True
            logger.warning(
                "world session payload rejected run_id=%s",
                principal.run_id if principal else "-",
                exc_info=True,
            )
            try:
                code, content, close_code = _world_error("SESSION_INVALID")
                await asyncio.wait_for(
                    sender.error(content, code=code),
                    timeout=browser_send_timeout_s,
                )
                await websocket.close(code=close_code, reason=code)
            except Exception:
                pass
        except CoordinatorRejected as exc:
            if principal is not None and exc.reason == "ATTEMPT_ALREADY_OWNED":
                # Cross-replica replay keeps the same close behavior as local
                # replay. It owns nothing, so it sends no error message and
                # performs no callback or lease cleanup.
                try:
                    await websocket.close(code=1008, reason="UNAUTHORIZED")
                except Exception:
                    pass
                return
            if principal is not None:
                code, content, close_code = _world_error(exc.reason)
                world_session["admit_rejected"] = code
                error_fields: dict[str, Any] = {"code": code, "reason": code}
                if exc.retry_after_s is not None:
                    error_fields["retry_after_s"] = exc.retry_after_s
            else:
                # The legacy showcase interface intentionally keeps its
                # existing wire behavior; this audit covers only the
                # world-platform route.
                code = exc.reason
                content = f"realtime admission rejected: {exc.reason}"
                close_code = 1013 if exc.reason == "CAPACITY_EXHAUSTED" else 1008
                world_session["admit_rejected"] = exc.reason
                error_fields = {
                    "reason": code,
                    "retry_after_s": exc.retry_after_s,
                }
            try:
                await asyncio.wait_for(
                    sender.error(content, **error_fields),
                    timeout=browser_send_timeout_s,
                )
                await websocket.close(code=close_code, reason=code)
            except Exception:
                pass
        except AdmissionQueueFull as exc:
            if principal is not None:
                code, content, close_code = _world_error(exc.reason)
                world_session["admit_rejected"] = code
            else:
                code = exc.reason
                content = f"realtime admission rejected: {exc.reason}"
                close_code = 1013
            try:
                await asyncio.wait_for(
                    sender.error(
                        content,
                        **(
                            {"code": code, "reason": code}
                            if principal is not None
                            else {}
                        ),
                    ),
                    timeout=browser_send_timeout_s,
                )
                await websocket.close(code=close_code, reason=code)
            except Exception:
                pass
        except asyncio.CancelledError:
            if principal is not None and _record_world_session_cancellation(
                world_session
            ):
                logger.warning(
                    "World gateway session handler was cancelled before a "
                    "terminal fact run_id=%s",
                    principal.run_id,
                )
            # Cancellation belongs to the ASGI server and must remain visible to
            # its task supervisor after lifecycle attribution and cleanup run.
            raise
        except WebSocketDisconnect as exc:
            if principal is not None:
                disconnect = _freeze_browser_input_disconnect(exc)
                if not world_session.get(
                    "dynamic_completed"
                ) and _input_disconnect_can_end_session(disconnect):
                    world_session.setdefault("abort_reason", "USER_LEFT")
                    world_session.setdefault("abort_fault", "client")
        except OutputRouteClosed:
            # Output path closed by the system side, such as VAE or route failure;
            # this is not user behavior.
            if principal is not None:
                code, content, close_code = _world_error("GATEWAY_SESSION_FAILED")
                world_session["system_error"] = code
                world_session["system_error_type"] = "OutputRouteClosed"
                try:
                    await asyncio.wait_for(
                        sender.error(content, code=code),
                        timeout=browser_send_timeout_s,
                    )
                    await websocket.close(code=close_code, reason=code)
                except Exception:
                    pass
            else:
                world_session["system_error"] = "output route closed"
        except Exception as exc:
            if whitebox_config is not None and principal is None:
                code, content, close_code = _world_error("GATEWAY_SESSION_FAILED")
                close_reason = code
                world_session["system_error"] = code
                logger.exception(
                    "Whitebox gateway session failed session_id=%s", session_id
                )
            elif principal is not None:
                code, content, close_code = _world_error("GATEWAY_SESSION_FAILED")
                world_session["system_error"] = code
                world_session["system_error_type"] = type(exc).__name__
                close_reason = code
                logger.exception(
                    "World gateway session failed run_id=%s",
                    principal.run_id,
                )
            else:
                # Preserve the legacy showcase wire behavior. Only the
                # authorized world-platform route hides exception details.
                # str() can be empty, e.g. bare TimeoutError, so guard the first
                # line while keeping the legacy message shape unchanged.
                detail = (str(exc).splitlines() or [type(exc).__name__])[0]
                content = f"realtime gateway error: {detail}"
                world_session["system_error"] = detail
                close_reason = "gateway session failed"
            if route is None or not route.browser_detached:
                try:
                    # The browser might be the non-reading peer that caused the
                    # error, so even the notification must be bounded. Otherwise
                    # we could hang before finally and leak the reservation.
                    await asyncio.wait_for(
                        sender.error(
                            content,
                            **(
                                {"code": code}
                                if principal is not None or whitebox_config is not None
                                else {}
                            ),
                        ),
                        timeout=browser_send_timeout_s,
                    )
                    await websocket.close(
                        code=close_code if principal is not None else 1011,
                        reason=close_reason,
                    )
                except Exception:
                    pass
        finally:
            # Fire the business terminal before any potentially slow cleanup.
            # The callback itself is asynchronous and attempt-fenced; resource
            # release continues independently below.
            _fire_world_terminal_callback()
            if recorder is not None:
                recorder.session_closed()
            if directions is not None:
                directions.close()  # cancel revert timer
            await _cancel_tasks(set(direction_tasks))
            await _cancel_tasks(tasks)
            if upstream is not None:
                try:
                    await upstream.close()
                except Exception:
                    logger.exception(
                        "Upstream cleanup failed for session_id=%s", session_id
                    )
                if release_grace_s:
                    await asyncio.sleep(release_grace_s)
            if route is not None:
                await registry.unregister(session_id, generation_id, token=output_token)
            if assignment is not None:
                try:
                    if whitebox_report is not None:
                        from sglang.multimodal_gen.runtime.realtime.whitebox.metrics import (
                            stage,
                        )

                        with stage("lease_release", trace_id):
                            await coordinator.release(assignment)
                    else:
                        await coordinator.release(assignment)
                except Exception:
                    logger.exception(
                        "Coordinator release failed for session_id=%s",
                        assignment.session_id,
                    )
            _ensure_browser_output_detach_fact()
            cleanup_close_code = 1000
            cleanup_close_reason = ""
            cleanup_close_succeeded = False
            cleanup_close_error_type = ""
            try:
                if route is not None and route.browser_detached:
                    # A detached presentation skips the short finish-tail close
                    # so cancellation cannot poison Starlette's close state.
                    # Release source resources and the Coordinator lease first,
                    # then make one full bounded close attempt. Source failure
                    # must retain the existing 1011 contract even if this write
                    # direction becomes available again during cleanup.
                    if world_session.get("dynamic_completed"):
                        cleanup_close_code = 1000
                        cleanup_close_reason = ""
                    else:
                        (
                            cleanup_close_reason,
                            _,
                            cleanup_close_code,
                        ) = _world_error("GATEWAY_SESSION_FAILED")
                    await asyncio.wait_for(
                        websocket.close(
                            code=cleanup_close_code,
                            reason=cleanup_close_reason,
                        ),
                        timeout=browser_send_timeout_s,
                    )
                else:
                    # Preserve the established non-detached and legacy cleanup
                    # behavior. Those paths already made their authoritative
                    # close attempt before entering finally.
                    await websocket.close(code=1000, reason="")
                cleanup_close_succeeded = True
            except Exception as exc:
                cleanup_close_error_type = type(exc).__name__

            closed_at = asyncio.get_running_loop().time()
            session_closed_fields: dict[str, Any] = {
                "session_id": session_id,
                "generation_id": generation_id,
                "session_duration_ms": round(
                    max(0.0, closed_at - session_opened_monotonic) * 1000
                ),
                "close_attempt_code": cleanup_close_code,
                "close_attempt_succeeded": cleanup_close_succeeded,
                "last_received_chunk": playback_ack_window.last_received_chunk,
                "last_rendered_chunk": playback_ack_window.last_rendered_chunk,
            }
            if cleanup_close_error_type:
                session_closed_fields["close_attempt_error_type"] = (
                    cleanup_close_error_type
                )
            terminal = world_session.get("terminal_fact")
            if isinstance(terminal, _WorldTerminalFact):
                session_closed_fields.update(
                    outcome=terminal.outcome,
                    fault=terminal.fault,
                    reason=terminal.reason,
                )
            else:
                session_closed_fields["outcome"] = "none"
            if principal is not None:
                session_closed_fields.update(
                    run_id=principal.run_id,
                    attempt_id=principal.jti,
                    profile_id=principal.profile_id,
                    role=principal.role,
                )
                gateway_instance = (os.getenv("HOSTNAME") or "").strip()
                if gateway_instance:
                    session_closed_fields["gateway_instance"] = gateway_instance[:253]
            input_disconnect = world_session.get("browser_input_disconnect_fact")
            if isinstance(input_disconnect, _BrowserInputDisconnectFact):
                session_closed_fields.update(
                    disconnect_kind=input_disconnect.kind,
                    disconnect_phase=input_disconnect.phase,
                    websocket_close_code=input_disconnect.close_code,
                    disconnected_at_ms=input_disconnect.disconnected_at_ms,
                )
            output_detach = world_session.get("browser_output_detach_fact")
            session_closed_fields["browser_output_detached"] = isinstance(
                output_detach, _BrowserOutputDetachFact
            )
            if isinstance(output_detach, _BrowserOutputDetachFact):
                session_closed_fields.update(
                    browser_output_detach_class=output_detach.detach_class,
                    browser_output_detach_phase=output_detach.phase,
                    browser_output_detached_at_ms=output_detach.detached_at_ms,
                )
            _log_gateway_trace(
                trace_id,
                "gateway.session_closed",
                **session_closed_fields,
            )
            if whitebox_report is not None:
                whitebox_report.gateway_done.set()
                if whitebox_report.recording is None:
                    whitebox_report.close()

    # Unauthenticated showcase routes used by the internal web UI. Public
    # isolation relies on ingress exposing only authorized_generate; see the
    # disable_legacy_routes notes in create_app.
    if enable_legacy_routes:

        @app.websocket("/v1/webrtc/play")
        async def whitebox_rtc(websocket: WebSocket):
            if (
                whitebox_proxy is None
                or os.environ.get("WHITEBOX_NATIVE_RTC_ENABLED") != "1"
                or os.environ.get("WHITEBOX_ADMISSION_PAUSED") == "1"
            ):
                await websocket.accept()
                await websocket.send_bytes(
                    msgspec.msgpack.encode(
                        dict(
                            type="error",
                            code="RTC_UNAVAILABLE",
                            content="WebRTC is unavailable. Please choose WebSocket before starting.",
                            reconnectable=False,
                        )
                    )
                )
                await websocket.close(code=1013)
                return
            await whitebox_proxy.serve(
                websocket,
                user_id="",
                transport="webrtc-rtp",
                trace_id=normalize_trace_id(
                    websocket.query_params.get("trace_id"), fallback=uuid4().hex
                ),
            )

        @app.websocket("/backends/lingbot2/v1/realtime_video/generate")
        async def generate_lingbot2(websocket: WebSocket):
            await _generate_coordinator_session(
                websocket,
                selected_model_revision=lingbot2_model_revision,
                selected_vae_fingerprint=(lingbot2_vae_fingerprint or vae_fingerprint),
            )

        @app.websocket("/backends/minwm/v1/realtime_video/generate")
        @app.websocket("/v1/realtime_video/generate")
        async def generate(websocket: WebSocket):
            if (
                whitebox_config is not None or whitebox_proxy is not None
            ) and os.environ.get("WHITEBOX_ADMISSION_PAUSED") == "1":
                await websocket.accept()
                await websocket.send_bytes(
                    msgspec.msgpack.encode(
                        dict(
                            version=2,
                            type="error",
                            code="PLAYTEST_MAINTENANCE",
                            content="Play is temporarily unavailable. Please try again shortly.",
                        )
                    )
                )
                await websocket.close(code=1013)
                return
            if whitebox_proxy is not None and websocket.query_params.get(
                "whitebox_control"
            ) in {"v1", "v2"}:
                await whitebox_proxy.serve(
                    websocket,
                    user_id=_user_id(websocket),
                    trace_id=normalize_trace_id(
                        websocket.query_params.get("trace_id"), fallback=uuid4().hex
                    ),
                )
                return
            if (
                whitebox_config is not None
                and websocket.query_params.get("whitebox_control") == "v1"
            ):
                from sglang.multimodal_gen.runtime.realtime.whitebox.interactive_gateway import (
                    serve,
                )

                await serve(
                    websocket,
                    coordinator=coordinator,
                    connect=connect_factory,
                    config=whitebox_config,
                    archives=whitebox_archives,
                    reports=whitebox_reports,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                    user_id=_user_id(websocket),
                    trace_id=normalize_trace_id(
                        websocket.query_params.get("trace_id"), fallback=uuid4().hex
                    ),
                    renew_seconds=lease_renew_interval_s,
                )
                return
            await _generate_coordinator_session(
                websocket,
                selected_model_revision=model_revision,
                selected_vae_fingerprint=vae_fingerprint,
                classic_whitebox=whitebox_config is not None,
            )

    if world_platform_profiles is not None:
        token_replay_guard = TokenReplayGuard()

        async def _authorized_principal(
            websocket: WebSocket, *, expected_role: str
        ) -> tuple[Principal, _WorldPlatformRuntime] | None:
            token = websocket.query_params.get("token") or ""
            try:
                principal, profile = world_platform_profiles.verify(token)
            except TokenError as exc:
                # Expose only a fixed business code. Distinguishing bad signature,
                # expiry, or aud mismatch would give token forgers a field oracle.
                # Detailed reasons stay in server logs.
                logger.warning("world token rejected: %s", exc)
                await websocket.accept()
                await websocket.close(code=1008, reason="UNAUTHORIZED")
                return None
            runtime = world_runtimes[profile.profile_id]
            if principal.role != expected_role:
                logger.warning(
                    "world token used on the wrong route run_id=%s role=%s",
                    principal.run_id,
                    principal.role,
                )
                await websocket.accept()
                await websocket.close(code=1008, reason="UNAUTHORIZED")
                return None
            if not token_replay_guard.consume(principal):
                # Only the connection that atomically consumes the jti can own
                # the attempt. A replay does not enter the shared lifecycle, so
                # it cannot acquire or release the owner's Coordinator lease or
                # send lifecycle or recorded callbacks.
                logger.warning("world token replayed run_id=%s", principal.run_id)
                await websocket.accept()
                await websocket.close(code=1008, reason="UNAUTHORIZED")
                return None
            return principal, runtime

        @app.websocket("/backends/minwm/v1/realtime_video/authorized_generate")
        async def authorized_generate(websocket: WebSocket):
            """Authorized world-platform route.

            It verifies a primary token, then reuses the shared session logic.
            Legacy unauthenticated routes continue serving showcase.
            """
            authorized = await _authorized_principal(websocket, expected_role="primary")
            if authorized is None:
                return
            principal, runtime = authorized
            await _generate_coordinator_session(
                websocket,
                selected_model_revision=model_revision,
                selected_vae_fingerprint=vae_fingerprint,
                principal=principal,
                world_runtime=runtime,
            )

        @app.websocket("/backends/lingbot2/v1/realtime_video/authorized_generate")
        async def authorized_generate_lingbot2(websocket: WebSocket):
            """Authorized comparison route using LingBot workers."""
            authorized = await _authorized_principal(
                websocket, expected_role="comparison"
            )
            if authorized is None:
                return
            principal, runtime = authorized
            await _generate_coordinator_session(
                websocket,
                selected_model_revision=lingbot2_model_revision,
                selected_vae_fingerprint=(lingbot2_vae_fingerprint or vae_fingerprint),
                principal=principal,
                world_runtime=runtime,
            )

    # Whitebox is integrated into the test platform; it needs no bundled WebUI.
    if enable_legacy_routes and whitebox_config is None:

        @app.get("/")
        async def index():
            return FileResponse(WEBUI_ROOT / "index.html")

        app.mount("/", StaticFiles(directory=WEBUI_ROOT), name="realtime-webui")
    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--lingbot2-upstream-url",
        default=os.environ.get("LINGBOT2_UPSTREAM_WS"),
    )
    parser.add_argument(
        "--lingbot2-model-revision",
        default=os.environ.get(
            "LINGBOT2_MODEL_REVISION",
            "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
        ),
    )
    parser.add_argument("--vae-fingerprint", default="taew2_2")
    parser.add_argument(
        "--lingbot2-vae-fingerprint",
        default=os.environ.get("LINGBOT2_VAE_FINGERPRINT"),
    )
    parser.add_argument(
        "--internal-output-url",
        default=os.environ.get("REALTIME_GATEWAY_OUTPUT_URL"),
    )
    parser.add_argument("--output-queue-depth", type=int, default=64)
    parser.add_argument("--output-enqueue-timeout-s", type=float, default=0.0)
    parser.add_argument("--output-queue-max-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--output-queue-max-messages", type=int, default=256)
    parser.add_argument("--output-drain-timeout-s", type=float, default=5.0)
    parser.add_argument("--lease-renew-interval-s", type=float, default=10.0)
    parser.add_argument("--release-grace-s", type=float, default=0.5)
    parser.add_argument("--max-admission-waiters", type=int, default=64)
    parser.add_argument("--readiness-coordinator-timeout-s", type=float, default=1.0)
    parser.add_argument("--readiness-coordinator-grace-s", type=float, default=30.0)
    parser.add_argument("--trace-log-group")
    # ---- World platform integration. All values are required to enable it. ----
    parser.add_argument(
        "--world-token-ed25519-pub",
        default=os.environ.get("WORLD_TOKEN_ED25519_PUB", ""),
    )
    parser.add_argument(
        "--world-token-ed25519-key-id",
        default=os.environ.get("WORLD_TOKEN_ED25519_KEY_ID", "legacy"),
        help="Expected JWS kid for the legacy single-profile configuration.",
    )
    parser.add_argument(
        "--world-platform-profiles-file",
        default=os.environ.get("WORLD_PLATFORM_PROFILES_FILE", ""),
        help=(
            "Path to a Secret-mounted JSON file containing per-token-key world "
            "platform profiles. It replaces the legacy single-profile flags."
        ),
    )
    parser.add_argument(
        "--world-callback-url", default=os.environ.get("WORLD_CALLBACK_URL", "")
    )
    parser.add_argument(
        "--world-callback-hmac-secret",
        default=os.environ.get("WORLD_CALLBACK_HMAC_SECRET", ""),
    )
    parser.add_argument(
        "--run-video-s3-bucket",
        default=os.environ.get("RUN_VIDEO_S3_BUCKET", ""),
    )
    parser.add_argument(
        "--run-video-s3-region",
        default=os.environ.get("RUN_VIDEO_S3_REGION", ""),
    )
    parser.add_argument(
        "--run-video-s3-endpoint",
        default=os.environ.get("RUN_VIDEO_S3_ENDPOINT", ""),
    )
    parser.add_argument(
        "--run-video-uri-scheme",
        choices=("s3", "oss"),
        default=os.environ.get("RUN_VIDEO_URI_SCHEME", "s3"),
    )
    parser.add_argument(
        "--disable-legacy-routes",
        action="store_true",
        help=(
            "Do not register unauthenticated showcase routes, static pages, or "
            "trace queries. They are registered by default because the internal "
            "web UI depends on them; public ingress should expose only the "
            "authorized_generate path."
        ),
    )
    parser.add_argument("--browser-send-timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--ui-config-json",
        default=os.environ.get("REALTIME_UI_CONFIG_JSON", "{}"),
    )
    parser.add_argument(
        "--world-studio-webui-base-url",
        default=os.environ.get("WORLD_STUDIO_WEBUI_BASE_URL", ""),
        help=(
            "Optional internal WebUI base URL. When set, gateway forwards "
            "/api/* to that WebUI so public domains can terminate on the "
            "gateway while keeping prompt/world APIs available."
        ),
    )
    return parser.parse_args()


def _build_world_platform(args) -> WorldPlatformConfig | None:
    """Enable world-platform routes only when all required settings are present.

    All five or none. A partial set used to silently skip route registration:
    authorized_generate then answers 404 and the operator debugs "route not
    found" instead of reading a config error. That 404 is also underdetermined
    (a proxy in front produces the same one), so the silent path wastes hours.
    Failing startup turns a misconfiguration into a ten-minute deploy failure
    that names the missing flag.
    """
    recording_bucket = (args.run_video_s3_bucket or "").strip()
    recording_region = (args.run_video_s3_region or "").strip()
    recording_endpoint = (getattr(args, "run_video_s3_endpoint", "") or "").strip()
    recording_uri_scheme = (getattr(args, "run_video_uri_scheme", "s3") or "s3").strip()
    provided = {
        "--world-token-ed25519-pub": bool(args.world_token_ed25519_pub),
        "--world-callback-url": bool(args.world_callback_url),
        "--world-callback-hmac-secret": bool(args.world_callback_hmac_secret),
        "--run-video-s3-bucket": bool(recording_bucket),
        "--run-video-s3-region": bool(recording_region),
    }
    if not any(provided.values()):
        return None  # world platform intentionally disabled (showcase-only deploy)
    missing = [flag for flag, ok in provided.items() if not ok]
    if missing:
        raise SystemExit(
            "world platform is partially configured; refusing to start with the "
            "authorized_generate route silently missing. Provide all of "
            f"{sorted(provided)} or none. Missing: {missing}"
        )
    return WorldPlatformConfig(
        profile_id=(
            getattr(args, "world_token_ed25519_key_id", "legacy") or "legacy"
        ).strip(),
        public_key=load_public_key(args.world_token_ed25519_pub),
        callback_url=args.world_callback_url,
        callback_app_id="zing-gateway",
        callback_key_id="k1",
        callback_secret=args.world_callback_hmac_secret,
        recording_bucket=recording_bucket,
        recording_region=recording_region,
        recording_endpoint_url=recording_endpoint,
        recording_uri_scheme=recording_uri_scheme,
    )


def _json_object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("world platform profile JSON contains a duplicate key")
        result[key] = value
    return result


def _strict_object(
    value: Any,
    *,
    name: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise ValueError(f"{name} is missing required fields: {missing}")
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {unknown}")
    return value


def _required_profile_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_profile_string(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value.strip()


def _validate_profile_url(value: str, *, field: str, allow_empty: bool) -> None:
    if not value and allow_empty:
        return
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field} must be an absolute HTTP or HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{field} must not contain credentials, query, or fragment")


def _load_world_platform_profiles(path: str) -> WorldPlatformProfiles:
    """Load strict, Secret-mounted per-environment Gateway configuration."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
        document = json.loads(raw, object_pairs_hook=_json_object_without_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"cannot load world platform profiles file: {exc}") from exc

    try:
        root = _strict_object(
            document,
            name="world platform profiles",
            required={"version", "profiles"},
            optional={"legacy_profile_id"},
        )
        if root["version"] != 1:
            raise ValueError("world platform profiles version must be 1")
        profile_items = root["profiles"]
        if not isinstance(profile_items, list) or not profile_items:
            raise ValueError("world platform profiles must be a non-empty list")
        if len(profile_items) > 16:
            raise ValueError("world platform profiles must contain at most 16 entries")

        profiles = []
        for index, item in enumerate(profile_items):
            profile_name = f"world platform profile {index}"
            profile = _strict_object(
                item,
                name=profile_name,
                required={
                    "id",
                    "token_ed25519_public_key",
                    "token_issuer",
                    "callback",
                    "recording",
                },
            )
            callback = _strict_object(
                profile["callback"],
                name=f"{profile_name} callback",
                required={"url", "app_id", "key_id", "hmac_secret"},
            )
            recording = _strict_object(
                profile["recording"],
                name=f"{profile_name} recording",
                required={"bucket", "region"},
                optional={
                    "endpoint_url",
                    "uri_scheme",
                    "access_key_id",
                    "secret_access_key",
                    "session_token",
                },
            )

            callback_url = _required_profile_string(
                callback["url"], field=f"{profile_name} callback.url"
            )
            _validate_profile_url(
                callback_url, field=f"{profile_name} callback.url", allow_empty=False
            )
            endpoint_url = _optional_profile_string(
                recording.get("endpoint_url"),
                field=f"{profile_name} recording.endpoint_url",
            )
            _validate_profile_url(
                endpoint_url,
                field=f"{profile_name} recording.endpoint_url",
                allow_empty=True,
            )
            if endpoint_url and urlsplit(endpoint_url).scheme != "https":
                raise ValueError(
                    f"{profile_name} recording.endpoint_url must use HTTPS"
                )
            uri_scheme = _optional_profile_string(
                recording.get("uri_scheme", "s3"),
                field=f"{profile_name} recording.uri_scheme",
            )
            if uri_scheme not in {"s3", "oss"}:
                raise ValueError(
                    f"{profile_name} recording.uri_scheme must be s3 or oss"
                )
            if uri_scheme == "oss" and not endpoint_url:
                raise ValueError(
                    f"{profile_name} recording.endpoint_url is required for OSS"
                )
            access_key_id = _optional_profile_string(
                recording.get("access_key_id"),
                field=f"{profile_name} recording.access_key_id",
            )
            secret_access_key = _optional_profile_string(
                recording.get("secret_access_key"),
                field=f"{profile_name} recording.secret_access_key",
            )
            session_token = _optional_profile_string(
                recording.get("session_token"),
                field=f"{profile_name} recording.session_token",
            )
            if bool(access_key_id) != bool(secret_access_key):
                raise ValueError(
                    f"{profile_name} recording credentials must provide both key fields"
                )
            if session_token and not access_key_id:
                raise ValueError(
                    f"{profile_name} recording.session_token requires key credentials"
                )

            profiles.append(
                WorldPlatformConfig(
                    profile_id=_required_profile_string(
                        profile["id"], field=f"{profile_name} id"
                    ),
                    public_key=load_public_key(
                        _required_profile_string(
                            profile["token_ed25519_public_key"],
                            field=f"{profile_name} token_ed25519_public_key",
                        )
                    ),
                    token_issuer=_required_profile_string(
                        profile["token_issuer"],
                        field=f"{profile_name} token_issuer",
                    ),
                    callback_url=callback_url,
                    callback_app_id=_required_profile_string(
                        callback["app_id"], field=f"{profile_name} callback.app_id"
                    ),
                    callback_key_id=_required_profile_string(
                        callback["key_id"], field=f"{profile_name} callback.key_id"
                    ),
                    callback_secret=_required_profile_string(
                        callback["hmac_secret"],
                        field=f"{profile_name} callback.hmac_secret",
                    ),
                    recording_bucket=_required_profile_string(
                        recording["bucket"], field=f"{profile_name} recording.bucket"
                    ),
                    recording_region=_required_profile_string(
                        recording["region"], field=f"{profile_name} recording.region"
                    ),
                    recording_endpoint_url=endpoint_url,
                    recording_uri_scheme=uri_scheme,
                    recording_access_key_id=access_key_id,
                    recording_secret_access_key=secret_access_key,
                    recording_session_token=session_token,
                )
            )
        legacy_profile_id = _optional_profile_string(
            root.get("legacy_profile_id"), field="legacy_profile_id"
        )
        return WorldPlatformProfiles(profiles, legacy_profile_id=legacy_profile_id)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise SystemExit(f"invalid world platform profiles file: {exc}") from exc


def _build_world_platform_profiles(args) -> WorldPlatformProfiles | None:
    profiles_file = (getattr(args, "world_platform_profiles_file", "") or "").strip()
    if profiles_file:
        legacy_values = {
            "--world-token-ed25519-pub": args.world_token_ed25519_pub,
            "--world-token-ed25519-key-id": (
                ""
                if getattr(args, "world_token_ed25519_key_id", "legacy") == "legacy"
                else getattr(args, "world_token_ed25519_key_id", "")
            ),
            "--world-callback-url": args.world_callback_url,
            "--world-callback-hmac-secret": args.world_callback_hmac_secret,
            "--run-video-s3-bucket": args.run_video_s3_bucket,
            "--run-video-s3-region": args.run_video_s3_region,
            "--run-video-s3-endpoint": getattr(args, "run_video_s3_endpoint", ""),
            "--run-video-uri-scheme": (
                ""
                if getattr(args, "run_video_uri_scheme", "s3") == "s3"
                else getattr(args, "run_video_uri_scheme", "")
            ),
        }
        mixed = sorted(flag for flag, value in legacy_values.items() if value)
        if mixed:
            raise SystemExit(
                "world platform profiles file cannot be combined with legacy "
                f"single-profile flags: {mixed}"
            )
        return _load_world_platform_profiles(profiles_file)
    legacy = _build_world_platform(args)
    return WorldPlatformProfiles.from_legacy(legacy) if legacy else None


def main() -> None:
    args = _parse_args()
    if not args.internal_output_url:
        raise SystemExit("--internal-output-url is required")
    try:
        ui_config = _parse_ui_config(args.ui_config_json)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    coordinator = HTTPCoordinatorClient(args.coordinator_url)
    trace_query = None
    if args.trace_log_group:
        try:
            import boto3
        except ImportError as exc:
            raise SystemExit("boto3 is required for CloudWatch Trace query") from exc
        from sglang.multimodal_gen.runtime.realtime.trace_query import (
            CloudWatchTraceQuery,
        )

        trace_query = CloudWatchTraceQuery(
            boto3.client("logs"), log_group=args.trace_log_group
        )
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import from_environment

    whitebox_proxy = None
    if os.environ.get("WHITEBOX_ROOM_PIPELINE") == "1":
        from sglang.multimodal_gen.runtime.realtime.whitebox.room_proxy import RoomProxy

        whitebox_proxy = RoomProxy.from_environment(connect)
        whitebox_config = whitebox_archives = None
    else:
        whitebox_config, whitebox_archives = from_environment()
    app = create_app(
        coordinator,
        whitebox_config=whitebox_config,
        whitebox_archives=whitebox_archives,
        whitebox_proxy=whitebox_proxy,
        model_revision=args.model_revision,
        vae_fingerprint=args.vae_fingerprint,
        internal_output_url=args.internal_output_url,
        lingbot2_upstream_url=args.lingbot2_upstream_url,
        lingbot2_model_revision=args.lingbot2_model_revision,
        lingbot2_vae_fingerprint=args.lingbot2_vae_fingerprint,
        output_queue_depth=args.output_queue_depth,
        output_enqueue_timeout_s=args.output_enqueue_timeout_s,
        output_queue_max_bytes=args.output_queue_max_bytes,
        output_queue_max_messages=args.output_queue_max_messages,
        output_drain_timeout_s=args.output_drain_timeout_s,
        lease_renew_interval_s=args.lease_renew_interval_s,
        release_grace_s=args.release_grace_s,
        max_admission_waiters=args.max_admission_waiters,
        readiness_coordinator_timeout_s=args.readiness_coordinator_timeout_s,
        readiness_coordinator_grace_s=args.readiness_coordinator_grace_s,
        ui_config=ui_config,
        trace_query=trace_query,
        world_platform_profiles=_build_world_platform_profiles(args),
        browser_send_timeout_s=args.browser_send_timeout_s,
        disable_legacy_routes=args.disable_legacy_routes,
        world_studio_webui_base_url=args.world_studio_webui_base_url,
    )
    ws_protocol = "auto"
    if whitebox_config or whitebox_proxy:
        from sglang.multimodal_gen.runtime.realtime.whitebox.websocket_protocol import (
            WhiteboxWebSocketProtocol,
        )

        ws_protocol = WhiteboxWebSocketProtocol
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        # Pin keepalive explicitly. Defaults are currently 20/20, but uvicorn is
        # not pinned in pyproject, and dependency drift should not change the
        # transport contract that ghost cleanup relies on.
        ws_max_size=8 * 1024 * 1024 + 4096
        if (whitebox_config or whitebox_proxy)
        else 16 * 1024 * 1024,
        ws=ws_protocol,
        ws_max_queue=1 if (whitebox_config or whitebox_proxy) else 32,
        ws_per_message_deflate=False if (whitebox_config or whitebox_proxy) else True,
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
    )


if __name__ == "__main__":
    main()
