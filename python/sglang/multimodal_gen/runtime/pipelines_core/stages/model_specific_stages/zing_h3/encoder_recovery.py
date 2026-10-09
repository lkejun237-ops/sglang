# SPDX-License-Identifier: Apache-2.0
"""Token-scoped recovery for the private encoder control channel.

An unreachable worker or failed data bridge requires supervised restart; this
module never kills processes or interprets an uncertain RPC as released state.
"""

import time

import zmq


def request_abort(port, token, timeout_seconds, *, require_restart=False):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    deadline = time.monotonic() + timeout_seconds

    def remaining_ms():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise zmq.Again()
        return max(1, int(remaining * 1000))

    socket.connect(f"tcp://127.0.0.1:{port}")
    try:
        socket.setsockopt(zmq.SNDTIMEO, remaining_ms())
        socket.send_json(
            dict(op="abort", session=token, require_restart=require_restart)
        )
        socket.setsockopt(zmq.RCVTIMEO, remaining_ms())
        reply = socket.recv_json()
        if not isinstance(reply, dict) or reply.get("session") != token:
            raise ValueError("Encoder abort reply token does not match")
        return reply
    except (zmq.ZMQError, ValueError) as error:
        return dict(
            ok=False,
            session=token,
            released=False,
            code="ENCODER_RECOVERY_UNCONFIRMED",
            error_type=type(error).__name__,
        )
    finally:
        socket.close()
        context.term()


def abort_owner(command, token, last_released, quarantined, cleanup, consensus):
    """Every rank calls consensus before cleanup and before any released ACK."""
    requested = command.get("session")
    states = consensus((token, last_released, quarantined))
    reply = dict(session=requested, released=False)
    if any(state != states[0] for state in states) or any(s[2] for s in states):
        return dict(reply, ok=False, code="ENCODER_RESTART_REQUIRED"), True
    if token != requested:
        return dict(
            reply,
            ok=True,
            current_owner_unchanged=True,
            already_released=requested is not None and requested == last_released,
        ), False
    if not isinstance(requested, str) or not requested:
        return dict(reply, ok=False, code="ENCODER_INVALID_TOKEN"), False
    if command.get("require_restart"):
        return dict(reply, ok=False, code="ENCODER_RESTART_REQUIRED"), True
    try:
        cleanup()
        cleaned = True
    except Exception:
        cleaned = False
    if not all(consensus(cleaned)):
        return dict(reply, ok=False, code="ENCODER_RESTART_REQUIRED"), True
    return dict(reply, ok=True, released=True), False
