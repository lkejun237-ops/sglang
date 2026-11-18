# SPDX-License-Identifier: Apache-2.0
"""Private, session-capability-protected prompt control transport."""

import asyncio
import io
import logging
from urllib.request import Request as URLRequest
from urllib.request import urlopen

import msgspec

PROTOCOL = "dit-clip-switch-v1"
PATH = "/zing/prompt-control"
MAX_REQUEST_BYTES = 80 << 20
MAX_RESULT_BYTES = 256 << 20


def request_prompt(attachment, operation, **fields):
    """CPU caller; never submit through an Encoder/DiT model execution queue."""
    body = dict(
        session_id=attachment["session_id"],
        generation_id=attachment["generation_id"],
        **fields,
    )
    if operation == "apply":
        import torch

        target = io.BytesIO()
        torch.save(body, target)
        body = target.getvalue()
    else:
        body = msgspec.msgpack.encode(body)
    if len(body) > (MAX_RESULT_BYTES if operation == "apply" else MAX_REQUEST_BYTES):
        raise ValueError("Prompt request is too large")
    request = URLRequest(
        attachment["url"] + "/" + operation,
        data=body,
        headers={
            "Authorization": "Bearer " + attachment["token"],
            "X-Session-Id": attachment["session_id"],
            "X-Generation-Id": attachment["generation_id"],
            "Content-Type": "application/msgpack",
        },
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        raw = response.read(MAX_RESULT_BYTES + 1)
    if len(raw) > MAX_RESULT_BYTES:
        raise ValueError("Prompt response is too large")
    if operation == "result":
        import torch

        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    return msgspec.msgpack.decode(raw)


def install_prompt_routes(app):
    from fastapi import Request
    from fastapi.responses import JSONResponse, Response

    @app.post(PATH + "/{operation}")
    async def prompt_control(operation: str, request: Request):
        manager = getattr(app.state, "worker_connections", None)
        control = getattr(manager, "prompt_control", None)
        if control is None or manager.state != "connected" or manager.restart_required:
            return JSONResponse({"code": "PROMPT_CONTROL_UNAVAILABLE"}, status_code=409)
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        owner = (
            request.headers.get("x-session-id"),
            request.headers.get("x-generation-id"),
        )
        try:
            with control.lock:
                control.authenticate(owner, token)
        except ValueError:
            return JSONResponse({"code": "PROMPT_CONTROL_DENIED"}, status_code=403)
        raw = bytearray()
        async for part in request.stream():
            raw.extend(part)
            if len(raw) > (
                MAX_RESULT_BYTES if operation == "apply" else MAX_REQUEST_BYTES
            ):
                return JSONResponse(
                    {"code": "PROMPT_REQUEST_TOO_LARGE"}, status_code=413
                )
        try:
            if operation == "apply":
                import torch

                body = await asyncio.to_thread(
                    torch.load, io.BytesIO(raw), map_location="cpu", weights_only=True
                )
            else:
                body = msgspec.msgpack.decode(raw)
            if owner != (body["session_id"], body["generation_id"]):
                raise ValueError("Prompt control identity does not match")
            if operation == "begin":
                result = control.begin(owner, token, body["command"])
            elif operation == "route":
                result = await asyncio.to_thread(
                    control.route, owner, token, body["command"]
                )
            elif operation == "adopt":
                result = await asyncio.to_thread(
                    control.adopt,
                    owner,
                    token,
                    body["ticket"],
                    body["event_id"],
                    body["allow_deferred_kv"],
                )
            elif operation == "cancel":
                control.cancel(owner, token, body["ticket"])
                result = {"cancelled": True}
            elif operation == "apply":
                control.submit_candidate(
                    owner, token, body["ticket"], body["candidate"]
                )
                result = {"submitted": True}
            elif operation == "result":
                result = await asyncio.to_thread(
                    control.result, owner, token, body["ticket"]
                )

                def encode():
                    import torch

                    target = io.BytesIO()
                    torch.save(
                        {
                            k: v
                            for k, v in result.items()
                            if k not in {"admission_request", "cache_query_context"}
                        },
                        target,
                    )
                    if target.tell() > MAX_RESULT_BYTES:
                        raise ValueError("Prompt response is too large")
                    return target.getvalue()

                payload = await asyncio.to_thread(encode)
                return Response(
                    payload,
                    media_type="application/octet-stream",
                    headers={"Cache-Control": "no-store"},
                )
            else:
                raise ValueError("Unsupported prompt operation")
            return Response(
                msgspec.msgpack.encode(result),
                media_type="application/msgpack",
                headers={"Cache-Control": "no-store"},
            )
        except (ValueError, KeyError, TypeError, msgspec.DecodeError):
            return JSONResponse({"code": "PROMPT_CONTROL_REJECTED"}, status_code=409)
        except Exception:
            logging.getLogger(__name__).exception("Independent prompt query failed")
            return JSONResponse({"code": "PROMPT_QUERY_FAILED"}, status_code=503)
