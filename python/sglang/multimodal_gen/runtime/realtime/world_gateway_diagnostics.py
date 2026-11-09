# SPDX-License-Identifier: Apache-2.0

"""Frozen transport and terminal facts for authorized world sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BrowserInputDisconnectFact:
    kind: str
    phase: str
    close_code: int
    disconnected_at_ms: int
    observed_monotonic: float


@dataclass(frozen=True, slots=True)
class BrowserOutputDetachFact:
    detach_class: str
    phase: str
    detached_at_ms: int
    observed_monotonic: float


@dataclass(frozen=True, slots=True)
class WorldTerminalFact:
    outcome: str
    fault: str
    reason: str
    established: bool
    ended_at_ms: int | None
    input_disconnect: BrowserInputDisconnectFact | None
    output_detach: BrowserOutputDetachFact | None
    classification_error: str | None = None


def normalize_websocket_close_code(value: Any) -> int:
    """Return an ASGI close observation without inventing a normal close frame."""
    if type(value) is not int or not 1000 <= value <= 4999:
        return 1005
    return value


def classify_browser_input_close(code: int) -> str:
    if code in {1005, 1006, 1015}:
        return "connection_lost"
    if code in {1002, 1003, 1004, 1007, 1008, 1009, 1010}:
        return "protocol_close"
    return "close_frame"


def classify_browser_input_close_metric(code: int) -> str:
    if code == 1000:
        return "normal"
    if code == 1001:
        return "going_away"
    if code in {1005, 1006, 1015}:
        return "abnormal"
    if code in {1002, 1003, 1004, 1007, 1008, 1009, 1010}:
        return "protocol"
    if 1011 <= code <= 1014:
        return "service"
    if 3000 <= code <= 4999:
        return "application"
    return "other"


def classify_browser_output_detach(reason: str) -> str:
    if reason in {
        "queue_message_limit",
        "queue_byte_limit",
        "finish_deadline",
    }:
        return reason
    if reason.startswith("browser_send_"):
        return "send_timeout" if "timeout" in reason.lower() else "send_error"
    return "other"


def world_disconnect_phase(world_session: dict[str, Any], observed_at: float) -> str:
    play_deadline = world_session.get("play_deadline_monotonic")
    if world_session.get("finishing") or (
        isinstance(play_deadline, float) and observed_at >= play_deadline
    ):
        return "finishing"
    first_media = world_session.get("first_media_monotonic")
    if not isinstance(first_media, float) or observed_at < first_media:
        return "before_first_frame"
    return "playing"


def record_browser_input_disconnect(
    world_session: dict[str, Any],
    fact: BrowserInputDisconnectFact,
) -> BrowserInputDisconnectFact:
    current = world_session.get("browser_input_disconnect_fact")
    if isinstance(current, BrowserInputDisconnectFact):
        return current
    world_session["browser_input_disconnect_fact"] = fact
    return fact


def record_browser_output_detach(
    world_session: dict[str, Any],
    fact: BrowserOutputDetachFact,
) -> BrowserOutputDetachFact:
    current = world_session.get("browser_output_detach_fact")
    if isinstance(current, BrowserOutputDetachFact):
        return current
    world_session["browser_output_detach_fact"] = fact
    return fact


def input_disconnect_can_end_session(
    fact: BrowserInputDisconnectFact | None,
) -> bool:
    return isinstance(fact, BrowserInputDisconnectFact) and fact.phase != "finishing"


def world_session_has_authoritative_terminal_fact(
    world_session: dict[str, Any],
) -> bool:
    if isinstance(world_session.get("terminal_fact"), WorldTerminalFact):
        return True
    if any(
        world_session.get(key)
        for key in (
            "dynamic_completed",
            "admit_rejected",
            "init_rejected",
            "abort_reason",
            "system_error",
        )
    ):
        return True
    return input_disconnect_can_end_session(
        world_session.get("browser_input_disconnect_fact")
    )


def record_world_session_cancellation(world_session: dict[str, Any]) -> bool:
    """Freeze an unexplained handler cancellation as a Gateway-owned failure."""
    if world_session_has_authoritative_terminal_fact(world_session):
        return False
    world_session.setdefault("abort_reason", "GATEWAY_SESSION_FAILED")
    world_session.setdefault("abort_fault", "ours")
    world_session.setdefault("system_error_type", "CancelledError")
    return True


def build_world_terminal_fact(
    world_session: dict[str, Any],
    *,
    assignment_exists: bool,
    established: bool,
) -> WorldTerminalFact:
    current = world_session.get("terminal_fact")
    if isinstance(current, WorldTerminalFact):
        return current

    input_disconnect = world_session.get("browser_input_disconnect_fact")
    if not isinstance(input_disconnect, BrowserInputDisconnectFact):
        input_disconnect = None
    output_detach = world_session.get("browser_output_detach_fact")
    if not isinstance(output_detach, BrowserOutputDetachFact):
        output_detach = None

    def terminal(
        outcome: str,
        fault: str,
        reason: str,
        *,
        ended_at_ms: int | None = None,
        classification_error: str | None = None,
    ) -> WorldTerminalFact:
        return WorldTerminalFact(
            outcome=outcome,
            fault=fault,
            reason=reason,
            established=established,
            ended_at_ms=ended_at_ms,
            input_disconnect=input_disconnect,
            output_detach=output_detach,
            classification_error=classification_error,
        )

    ended_at_ms = world_session.get("ended_at_ms")
    if world_session.get("dynamic_completed"):
        if isinstance(ended_at_ms, int):
            return terminal("ended", "", "completed", ended_at_ms=ended_at_ms)
        return terminal(
            "aborted",
            "ours",
            "GATEWAY_SESSION_FAILED",
            classification_error="completed_without_ended_at",
        )
    if world_session.get("admit_rejected"):
        return terminal("aborted", "ours", str(world_session["admit_rejected"]))
    if world_session.get("init_rejected"):
        return terminal("aborted", "client", "SESSION_INVALID")
    if world_session.get("abort_reason"):
        reason = str(world_session["abort_reason"])
        if reason != "USER_LEFT":
            return terminal(
                "aborted",
                str(world_session.get("abort_fault") or "ours"),
                reason,
            )
        if input_disconnect_can_end_session(input_disconnect):
            return terminal("aborted", "client", "USER_LEFT")
        return terminal(
            "aborted",
            "ours",
            "GATEWAY_SESSION_FAILED",
            classification_error="user_left_without_input_disconnect",
        )
    if not assignment_exists:
        return terminal("aborted", "ours", "SESSION_UNAVAILABLE")
    if world_session.get("system_error"):
        return terminal("aborted", "ours", str(world_session["system_error"])[:200])
    if input_disconnect_can_end_session(input_disconnect):
        return terminal("aborted", "client", "USER_LEFT")
    return terminal(
        "aborted",
        "ours",
        "GATEWAY_SESSION_FAILED",
        classification_error="unknown_terminal_state",
    )
