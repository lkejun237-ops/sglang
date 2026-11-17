# SPDX-License-Identifier: Apache-2.0
"""Public diagnostic codes. Never expose arbitrary exception text or paths."""

import logging
import os
import stat

_player_logger = logging.getLogger(__name__)
# The deployment collector reads stderr; never borrow its synchronous Handler locks.


def _emit_player_failure(event):
    if not stat.S_ISFIFO(os.fstat(2).st_mode):
        return
    # dup is safe only for an already nonblocking pipe. Linux /proc opens an
    # independent description; setting flags on dup(2) would alter every writer.
    descriptor = (
        os.dup(2)
        if not os.get_blocking(2)
        else os.open("/proc/self/fd/2", os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    )
    try:
        if os.get_blocking(descriptor):
            return
        line = (
            f"{event.levelname} {event.name} [{event.filename}:{event.lineno} "
            f"{event.funcName}] {event.getMessage()}\n"
        ).encode("utf-8")
        # One atomic packet or a drop. Never block, write a partial row or retry.
        if len(line) <= os.fpathconf(descriptor, "PC_PIPE_BUF"):
            os.write(descriptor, line)
    finally:
        os.close(descriptor)


MESSAGES = {
    "SOURCE_FRAME_TIMEOUT": "The world source stopped producing frames for 15 seconds. Please restart the session.",
    "RECONNECT_EXPIRED": "The play session ended after waiting for the client connection to recover.",
    "MODEL_CONTEXT_RECOVERY_UNAVAILABLE": "The world source stopped because model context recovery was unavailable.",
    "MODEL_CONTEXT_RECOVERY_EXPIRED": "The world source did not reconnect within the model context recovery window.",
    "ARCHIVE_CAPACITY_UNAVAILABLE": "Recording is temporarily unavailable. Please try again shortly.",
    "CONTROL_QUEUE_FULL": "Game controls are arriving too quickly. Please reconnect and try again.",
    "INVALID_CONTROL_MESSAGE": "A game control could not be processed. Please check the client controls before reconnecting.",
    "RTC_UNAVAILABLE": "WebRTC is unavailable. Please choose WebSocket before starting.",
    "RTC_CODEC_UNSUPPORTED": "The selected video encoder could not start. Try different video settings or reconnect.",
    "RTC_AUTHORIZATION_FAILED": "Please sign in and verify access to this world before starting WebRTC.",
    "RTC_CONNECTION_FAILED": "The WebRTC connection could not be established. Please try again or choose WebSocket.",
    "RTC_DISCONNECTED": "The video connection was lost. Please reconnect.",
    "RTC_CONTROL_OVERFLOW": "Session updates could not be delivered in time. Please reconnect.",
    "RTC_MEDIA_STALLED": "The video connection could not keep up. Please reconnect.",
    "RTC_CLEANUP_PENDING": "The previous session is still being cleaned up. Please wait before starting again.",
    "MODEL_PREPARATION_FAILED": "The model could not prepare the initial picture. The detailed cause is unavailable here; report the trace ID.",
    "MODEL_FAILED": "The model could not continue rendering. Report the trace ID for diagnosis.",
    "MODEL_STARTUP_TIMEOUT": "The model took too long to produce its first frame.",
    "CONDITION_SWITCH_FAILED": "The rendering instructions could not be applied. Report the trace ID for diagnosis.",
    "ARCHIVE_FAILED": "The recording could not be prepared. Report the trace ID for diagnosis.",
    "VIDEO_ENCODING_FAILED": "The video could not be prepared. Report the trace ID for diagnosis.",
    "RENDER_OUTPUT_FAILED": "The rendered video could not be processed. Report the trace ID for diagnosis.",
    "CONTROL_SERVICE_FAILED": "The control service could not complete this request. Report the trace ID for diagnosis.",
    "INTERACTIVE_SESSION_FAILED": "This session could not continue. Report the trace ID for diagnosis.",
    "ROOM_PIPELINE_FAILED": "This play session could not continue.",
    "ROOM_SESSION_LOST": "This source connection was interrupted. Please reconnect.",
    "CAPACITY_EXHAUSTED": "All play slots are busy. Please try again shortly.",
    "USER_ALREADY_ACTIVE": "Another play session is already active.",
    "USER_SESSION_LIMIT": "Another play session is already active. Close it before starting again.",
    "INVALID_CLI_FPS": "Choose a whole-number capture rate between 1 and 60 FPS.",
    "INVALID_RESOLUTION": "Choose a supported playtest resolution.",
    "RESOLUTION_UNSUPPORTED": "This source does not support the requested resolution.",
    "CLI_FPS_UNSUPPORTED": "This source does not support a custom capture rate.",
    "WTP_PLAYABLE_NOT_READY": "This world's playable resources are not ready yet.",
    "WTP_PLAYABLE_PREPARATION_FAILED": "The world's playable resources could not be prepared. Please try again.",
    "WTP_PLAYABLE_PUBLICATION_CHANGED": "The world was updated while preparing. Please reconnect.",
    "WTP_PLAYABLE_AUTHENTICATION_REQUIRED": "Please sign in before preparing this world.",
    "WTP_REFERENCE_ORIGIN_MISMATCH": "Please open the world from its configured platform.",
    "STREAM_DOWNLOAD_TIMEOUT": "World resources took too long to download. Please try again.",
    "STREAM_NETWORK_FAILED": "The world resource connection was interrupted after several attempts. Please try again.",
    "STREAM_DELIVERY_UI_REQUIRED": "This delivery is missing its required UI bundle.",
    "STREAM_DELIVERY_UI_INVALID": "This delivery contains an invalid UI bundle.",
    "STREAM_STARTUP_TIMEOUT": "The world took too long to start. Please try again.",
    "STREAM_STARTUP_ABORTED": "World startup was cancelled or reached its time limit.",
    "STREAM_STARTUP_FAILED": "The world could not start. Please try again.",
    "ROOM_HOST_CAPACITY_INSUFFICIENT": "This room requires more players than the server supports.",
    "CLI_STARTUP_FAILED": "The world could not start. Please try again.",
}
STAGES = {
    "admission",
    "archive",
    "encoding",
    "rendering",
    "validation",
    "download",
    "startup",
    "source",
    "model",
    "transport",
    "control",
}


class WhiteboxDiagnostic(ValueError):
    def __init__(self, code, stage="startup"):
        self.code = code if code in MESSAGES else "CLI_STARTUP_FAILED"
        self.stage = stage if stage in STAGES else "startup"
        super().__init__(MESSAGES[self.code])


def cli_fps_override(value):
    if value is not None and (type(value) is not int or not 1 <= value <= 60):
        raise WhiteboxDiagnostic("INVALID_CLI_FPS", "validation")
    return value


def public_error(error, trace_id, *, default="ROOM_SESSION_LOST", stage="source"):
    import re

    from .render_contract import MESSAGES as RENDER_MESSAGES

    raw_code = getattr(error, "code", None)
    upstream_code = getattr(error, "upstream_code", None)
    code = getattr(error, "code", getattr(error, "reason", default))
    if not isinstance(code, str) or (
        code not in MESSAGES and code not in RENDER_MESSAGES
    ):
        if (
            upstream_code is None
            and isinstance(raw_code, str)
            and re.fullmatch(r"[A-Z][A-Z0-9_.:-]{0,95}", raw_code)
            and raw_code != default
        ):
            upstream_code = raw_code
        code = default
    phase = getattr(error, "stage", None) or (
        "admission"
        if code in {"CAPACITY_EXHAUSTED", "USER_ALREADY_ACTIVE", "USER_SESSION_LIMIT"}
        else stage
    )
    safe_trace = (
        trace_id
        if isinstance(trace_id, str)
        and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", trace_id)
        else None
    )
    result = dict(
        code=code,
        stage=phase if phase in STAGES else "source",
        trace_id=safe_trace,
        content=MESSAGES.get(
            code,
            RENDER_MESSAGES.get(
                code, "This source connection was interrupted. Please reconnect."
            ),
        ),
    )

    if isinstance(upstream_code, str) and re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_.:-]{0,95}", upstream_code
    ):
        result["upstream_code"] = upstream_code
    cleanup_code = getattr(error, "cleanup_code", None)
    if cleanup_code == "RELEASE_UNKNOWN":
        result["cleanup_code"] = cleanup_code
    return result


def model_error(message, *, trace_id, operation, code="MODEL_FAILED"):
    """Keep the model origin; peer exception bodies belong only in server logs.

    Do not serialize the whole response: it may contain frame/reference bytes.
    A valid downstream code is correlation metadata, never a client-input verdict.
    """
    import logging
    import re

    peer = message if isinstance(message, dict) else {}
    nested = peer.get("error")
    nested = nested if isinstance(nested, dict) else {}
    upstream_code = peer.get("code", nested.get("code"))
    error = WhiteboxDiagnostic(code, "model")
    error.upstream_code = (
        upstream_code
        if isinstance(upstream_code, str)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,95}", upstream_code)
        else None
    )
    detail = peer.get("content", nested.get("message"))
    # Bound the private diagnostic; never log arbitrary nested payloads/pixels.
    logging.getLogger(__name__).error(
        "Model operation failed trace_id=%s operation=%s upstream_code=%s peer_type=%r detail=%r",
        trace_id,
        operation,
        error.upstream_code,
        (
            peer.get("type", "unknown")[:128]
            if isinstance(peer.get("type", "unknown"), str)
            else "invalid"
        ),
        detail[:2048] if isinstance(detail, str) else None,
    )
    return error


def model_preparation_error(message, *, trace_id, operation):
    return model_error(
        message, trace_id=trace_id, operation=operation, code="MODEL_PREPARATION_FAILED"
    )


def log_player_failure(
    stage, error, *, record=None, binding=None, outcome="failed", phase=None
):
    """Best-effort member diagnostics through the existing container stderr collector."""
    import asyncio
    import json
    import re

    try:
        if isinstance(error, asyncio.CancelledError):
            return
        code = getattr(error, "code", None) or getattr(error, "reason", None)
        level = (
            logging.WARNING
            if outcome == "unknown"
            or code
            in {"CAPACITY_EXHAUSTED", "USER_SESSION_LIMIT", "USER_ALREADY_ACTIVE"}
            else logging.ERROR
        )
        if not _player_logger.isEnabledFor(level) or getattr(
            error, "_whitebox_failure_logged", False
        ):
            return
        record = record or {}
        binding = binding or record.get("model_binding")
        lease = getattr(binding, "lease", None)
        assignment = getattr(binding, "assignment", None) or getattr(
            lease, "_active_assignment", None
        )
        creation = record.get("invocation", {}).get("createContext", {})
        operation = record.get("admission") or creation.get("operation", {})
        world = creation.get("frozenPublication", {}).get("world", {})
        member = record.get("binding", {})
        worker = getattr(assignment, "denoiser", None)
        fields = {
            "phase": phase,
            "gameId": member.get("gameId") or record.get("game_id"),
            "ownerEpoch": member.get("ownerEpoch"),
            "publicationId": member.get("publicationId") or world.get("publicationId"),
            "worldId": world.get("worldId"),
            "operationId": operation.get("operationId")
            or getattr(lease, "request", {}).get("trace_id"),
            "requestId": operation.get("requestId"),
            "memberId": record.get("member_id"),
            "playerId": record.get("player_id"),
            "modelSessionId": getattr(assignment, "session_id", None)
            or getattr(lease, "session_id", None),
            "generationId": getattr(assignment, "generation_id", None)
            or getattr(lease, "generation_id", None),
            "workerId": getattr(worker, "worker_id", None),
            "workerEpoch": getattr(worker, "worker_epoch", None),
        }

        def failure(value, depth=0):
            value_code = getattr(value, "code", None) or getattr(value, "reason", None)
            result = {
                "code": (
                    value_code
                    if isinstance(value_code, str)
                    and re.fullmatch(r"[A-Z][A-Z0-9_]{1,95}", value_code)
                    else "UNCLASSIFIED_FAILURE"
                ),
                "type": type(value).__name__,
            }
            cause = getattr(value, "__cause__", None)
            if cause is not None and depth < 2:
                result["cause"] = failure(cause, depth + 1)
            return result

        row = dict(
            event="whitebox.player.failed",
            stage=stage,
            outcome=outcome,
            **{
                key: value
                for key, value in fields.items()
                if isinstance(value, str)
                and re.fullmatch(r"[a-zA-Z0-9_.:-]{1,256}", value)
            },
            error=failure(error),
        )
        # Preserve the originating call site for the sanitized diagnostic record.
        pathname, lineno, function, _ = _player_logger.findCaller(stacklevel=2)
        event = _player_logger.makeRecord(
            _player_logger.name,
            level,
            pathname,
            lineno,
            json.dumps(row, separators=(",", ":")),
            (),
            None,
            func=function,
        )
        error._whitebox_failure_logged = True
        _emit_player_failure(event)
    except Exception:
        # Diagnostics may be dropped; never replace the original result or cleanup.
        pass
