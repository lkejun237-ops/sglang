# SPDX-License-Identifier: Apache-2.0
"""Serial fixed-case prefill using ordinary realtime request dispatch/cleanup."""

import asyncio
import logging
from uuid import uuid4

import msgspec.msgpack
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.cache_logging import (
    cache_event,
)
from sglang.multimodal_gen.runtime.realtime.admission import AdmissionRejected
from sglang.multimodal_gen.runtime.realtime.source_transport import (
    SourceTransportBudget,
)

from .adapters.zing_h3_source_adapter import ZingH3SourceStreamAdapter
from .conditioning import get_api_resources
from .generate_session import GenerateSession

router = APIRouter()


class PrefillRequestError(ValueError):
    pass


def _request_id(data):
    value = data.get("request_id") if isinstance(data, dict) else None
    return value if isinstance(value, str) and 1 <= len(value) <= 128 else None


def parse_prefill(data):
    if not isinstance(data, dict) or data.get("type") != "prefill":
        raise PrefillRequestError("Expected a prefill request")
    if _request_id(data) is None:
        raise PrefillRequestError("A request ID is required")
    values = dict(data)
    values.pop("request_id")
    force = values.pop("force", False)
    if type(force) is not bool:
        raise PrefillRequestError("Prefill force must be a boolean")
    frame = values.pop("source_frame", None)
    values["type"] = "init"
    return RealtimeVideoGenerationsRequest.model_validate(values), frame


async def run_prefill_case(ws, data, wire_size, session_id):
    # Reuse the normal socket admission, sampler construction, scheduler RPC
    # ownership and cancellation-safe cleanup. No second task scheduler.
    from . import realtime_video_api as api

    args = api.get_global_server_args()
    request, frame = parse_prefill(data)
    adapter = api.get_realtime_model_adapter(args).for_request(request)
    if not isinstance(adapter, ZingH3SourceStreamAdapter):
        raise PrefillRequestError("The selected model does not support prefill")
    session = GenerateSession(session_id=session_id)
    controller = api._get_admission_controller(args)
    lease = await controller.admit(
        api._resolve_realtime_user_id(ws),
        session.id,
        session.generation_id,
        wait_for_capacity=False,
    )
    renewal = asyncio.create_task(
        api._session_watchdog(
            session,
            controller,
            lease,
            idle_timeout_s=0,
            max_lifetime_s=0,
            lease_ttl_s=args.realtime_session_lease_ttl_s,
        )
    )
    try:
        session._conditioning_init_reservation = get_api_resources(
            session, args
        ).reserve_api_message(wire_size)
        session.set_adapter(adapter)
        await api._wait_for_server_warmup(ws)
        await adapter.on_init(session, request, prefill=True)
        if wire_size > adapter.max_message_bytes:
            raise PrefillRequestError("The prefill request exceeds its message budget")
        session.set_request(request)
        batch = adapter.prepare_prefill_request(session, args, frame)
        batch.extra["zing_prefill"].update(
            operation_id=data["request_id"],
            force=data.get("force", False),
            game_id=request.game_id,
            group_id=request.group_id,
        )
        _, result = await api._process_realtime_generation_batch(batch)
        receipt = (result.usage or {}).get("zing_prefill")
        if not isinstance(receipt, dict) or not receipt.get("cache_key"):
            raise RuntimeError("Prefill completed without a cache receipt")
        return dict(
            type="prefill.completed",
            request_id=data["request_id"],
            session_id=session.id,
            **receipt,
        )
    finally:

        async def cleanup():
            try:
                released = await api._cleanup_realtime_session(session, None, None)
                if not released:
                    raise RuntimeError("Prefill worker cleanup was not acknowledged")
            finally:
                renewal.cancel()
                await api._await_realtime_task(renewal)
                await controller.release(lease)

        # Success is sent only after worker/session resources and the lease
        # have been released. The long-lived cache remains on this worker.
        await api._drain_realtime_cleanup(cleanup())


@router.websocket("/prefill")
async def prefill_websocket(ws: WebSocket):
    from . import realtime_video_api as api

    await ws.accept()
    maximum = SourceTransportBudget.from_args(
        api.get_global_server_args()
    ).message_bytes
    session_id = uuid4().hex  # Stable DP routing for every case on this socket.
    active = None
    send_lock = asyncio.Lock()

    async def send(value):
        async with send_lock:
            try:
                await ws.send_bytes(msgspec.msgpack.encode(value))
            except (WebSocketDisconnect, RuntimeError):
                # Cleanup has its own lifetime even when the peer cannot read.
                pass

    async def process(data, size):
        nonlocal active
        request_id = _request_id(data)
        try:
            receipt = await run_prefill_case(ws, data, size, session_id)
            await send(receipt)
        except AdmissionRejected:
            cache_event(
                "prefill_request",
                "rejected",
                reason="prefill_busy",
                context={"operation_id": request_id, "session_id": session_id},
                level=logging.WARNING,
            )
            await send(
                dict(
                    type="prefill.failed",
                    request_id=request_id,
                    code="prefill_busy",
                    content="The service is busy. Please try again.",
                    fatal=False,
                )
            )
        except ConditionError as exc:
            cache_event(
                "prefill_request",
                "failed",
                reason=exc.code,
                context={"operation_id": request_id, "session_id": session_id},
                level=logging.WARNING,
            )
            fatal = exc.code in {"CONDITION_UNAVAILABLE", "CONDITION_TIMEOUT"}
            await send(
                dict(
                    type="prefill.failed",
                    request_id=request_id,
                    code=exc.code,
                    content=exc.public_message,
                    fatal=fatal,
                )
            )
            if fatal:
                await ws.close(code=1011)
        except (ValueError, TypeError, KeyError):
            cache_event(
                "prefill_request",
                "rejected",
                reason="prefill_invalid",
                context={"operation_id": request_id, "session_id": session_id},
                level=logging.WARNING,
            )
            await send(
                dict(
                    type="prefill.failed",
                    request_id=request_id,
                    code="prefill_invalid",
                    content="Please provide a valid prefill case.",
                    fatal=False,
                )
            )
        except Exception:
            api.logger.exception("Prefill request failed")
            await send(
                dict(
                    type="prefill.failed",
                    request_id=request_id,
                    code="prefill_failed",
                    content="The service could not complete prefill.",
                    fatal=True,
                )
            )
            await ws.close(code=1011)
        finally:
            active = None

    try:
        await send(dict(type="prefill.ready", max_message_bytes=maximum))
        while True:
            try:
                wire = await ws.receive_bytes()
            except KeyError:
                await send(
                    dict(
                        type="prefill.failed",
                        request_id=None,
                        code="prefill_invalid",
                        content="Please send a binary MessagePack case.",
                        fatal=False,
                    )
                )
                continue
            busy = active is not None
            if len(wire) > maximum:
                await send(
                    dict(
                        type="prefill.failed",
                        request_id=None,
                        code="prefill_too_large",
                        content="The prefill request is too large.",
                        fatal=False,
                    )
                )
                continue
            try:
                data = msgspec.msgpack.decode(wire)
            except msgspec.DecodeError:
                await send(
                    dict(
                        type="prefill.failed",
                        request_id=None,
                        code="prefill_invalid",
                        content="Please send a MessagePack prefill case.",
                        fatal=False,
                    )
                )
                continue
            if busy:
                await send(
                    dict(
                        type="prefill.failed",
                        request_id=_request_id(data),
                        code="prefill_busy",
                        content="Please wait for the current prefill receipt.",
                        fatal=False,
                    )
                )
                continue
            active = asyncio.create_task(process(data, len(wire)))
    except WebSocketDisconnect:
        pass
    finally:
        if active is not None:
            active.cancel()
            await api._drain_realtime_cleanup(api._await_realtime_task(active))
