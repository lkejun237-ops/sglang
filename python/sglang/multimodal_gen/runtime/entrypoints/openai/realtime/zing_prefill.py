# SPDX-License-Identifier: Apache-2.0
"""Serial fixed-case prefill using ordinary realtime request dispatch/cleanup."""

import asyncio
import logging
import time
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


_PERSISTENCE_WATCHERS = 0
_PERSISTENCE_READS = set()


def persistence_options(data):
    if not isinstance(data, dict):
        raise PrefillRequestError("Expected a prefill request")
    enabled = data.get("persist", False)
    timeout = data.get("persist_timeout_s", 180)
    if (
        type(enabled) is not bool
        or type(timeout) not in (int, float)
        or not 1 <= timeout <= 300
    ):
        raise PrefillRequestError("Invalid persistence options")
    return enabled, float(timeout)


async def _persistence_read(root, descriptor, timeout):
    from sglang.multimodal_gen.runtime.realtime.prefill_persistence import (
        inspect_persistence,
    )

    # Timed-out NFS calls may still own a thread. Keep counting them until the
    # actual call returns, rather than creating unbounded abandoned reads.
    if len(_PERSISTENCE_READS) >= 2:
        return {"state": "pending"}
    task = asyncio.create_task(asyncio.to_thread(inspect_persistence, root, descriptor))
    _PERSISTENCE_READS.add(task)

    def completed(value):
        _PERSISTENCE_READS.discard(value)
        if not value.cancelled():
            value.exception()

    task.add_done_callback(completed)
    return await asyncio.wait_for(asyncio.shield(task), timeout)


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
    values.pop("source_frame_size", None)
    persistence_options(values)
    values.pop("persist", None)
    values.pop("persist_timeout_s", None)
    values["type"] = "init"
    return RealtimeVideoGenerationsRequest.model_validate(values), frame


async def run_prefill_case(ws, data, wire_size, session_id, *, binding=None):
    # Reuse the normal socket admission, sampler construction, scheduler RPC
    # ownership and cancellation-safe cleanup. No second task scheduler.
    from . import realtime_video_api as api

    args = api.get_global_server_args()
    request, frame = parse_prefill(data)
    factory = (
        getattr(ws.app.state, "realtime_adapter_factory", None)
        or api.get_realtime_model_adapter
    )
    adapter = factory(args).for_request(request)
    if not isinstance(adapter, ZingH3SourceStreamAdapter):
        raise PrefillRequestError("The selected model does not support prefill")
    gateway = binding.gateway if binding else api._gateway_managed_config(ws)
    session = GenerateSession(
        session_id=gateway.session_id if gateway else session_id,
        generation_id=gateway.generation_id if gateway else None,
    )
    claim = ws.scope.get("role_pool_claim") if hasattr(ws, "scope") else None
    if claim is not None:
        session = GenerateSession(
            session_id=claim["session_id"], generation_id=claim["generation_id"]
        )
        session.role_pool_service = ws.scope.get("role_pool_service")
        session.role_pool_service.child_started(session.id, session.generation_id)
    session.execution_client = getattr(ws.app.state, "realtime_execution_client", None)
    controller = (
        None
        if gateway
        else (
            getattr(ws.app.state, "realtime_admission_controller", None)
            or api._get_admission_controller(args)
        )
    )
    lease = renewal = registry = None
    reservation_owner = (
        binding.owner_id if binding else (uuid4().hex if gateway else None)
    )
    try:
        if binding is not None:
            binding.cleaned = False
        if gateway:
            session.coordinator_token = gateway.coordinator_token
            if binding is not None:
                registry = binding.registry
            else:
                registry = await api._consume_gateway_reservation(
                    ws, gateway, owner_id=reservation_owner
                )
        else:
            lease = await controller.admit(
                api._resolve_realtime_user_id(
                    ws,
                    require_authenticated=getattr(
                        args, "realtime_require_authenticated_user", False
                    ),
                ),
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
        session._conditioning_init_reservation = get_api_resources(
            session, args
        ).reserve_api_message(wire_size)
        session.set_adapter(adapter)
        await api._wait_for_server_warmup(ws)
        await adapter.on_init(session, request, prefill=True)
        if wire_size > adapter.max_message_bytes:
            raise PrefillRequestError("The prefill request exceeds its message budget")
        session.set_request(request)
        batch = adapter.prepare_prefill_request(
            session,
            args,
            frame,
            source_frame_size=data.get("source_frame_size"),
            persist=data.get("persist", False),
        )
        if request.record_model_inputs:
            values = dict(
                prompt=request.prompt,
                references=adapter.reference_items,
                seed=request.seed,
                size=request.size,
                source_stream=request.source_stream,
                source_frame=frame,
                source_frame_size=data.get("source_frame_size"),
            )
            await ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="model.input",
                        **adapter.input_receipt(
                            session, "prefill", values, request_id=data["request_id"]
                        ),
                    )
                )
            )
        if registry is not None:
            await registry.mark_runnable(
                gateway.coordinator_token, owner_id=reservation_owner
            )
        batch.extra["zing_prefill"].update(
            operation_id=data["request_id"],
            force=data.get("force", False),
            game_id=request.game_id,
            group_id=request.group_id,
        )
        execution = (
            {}
            if session.execution_client is None
            else {"execution_client": session.execution_client}
        )
        _, result = await api._process_realtime_generation_batch(batch, **execution)
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
            managed = getattr(session, "role_pool_service", None)
            child_id, child_generation = session.id, session.generation_id
            released = False
            try:
                released = await api._cleanup_realtime_session(session, None, None)
                if not released:
                    raise RuntimeError("Prefill worker cleanup was not acknowledged")
            finally:
                if renewal is not None:
                    renewal.cancel()
                    await api._await_realtime_task(renewal)
                if released:
                    if lease is not None:
                        await controller.release(lease)
                    if binding is not None:
                        binding.cleaned = True
                    else:
                        await api._release_gateway_reservation(
                            registry, gateway, owner_id=reservation_owner
                        )
                    if managed is not None:
                        managed.child_cleaned(child_id, child_generation)
                elif registry is not None:
                    await registry.quarantine(
                        gateway.coordinator_token, owner_id=reservation_owner
                    )

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
        global _PERSISTENCE_WATCHERS
        request_id = _request_id(data)
        watching = False
        try:
            persist, timeout = persistence_options(data)
            if persist:
                if _PERSISTENCE_WATCHERS >= 2:
                    raise AdmissionRejected("PREFILL_STORAGE_BUSY")
                _PERSISTENCE_WATCHERS += 1
                watching = True
            started = time.monotonic()
            receipt = await run_prefill_case(ws, data, size, session_id)
            descriptor = receipt.pop("_disk_descriptor", None)
            managed = (
                ws.scope.get("role_pool_service") if hasattr(ws, "scope") else None
            )
            if persist and descriptor is None:
                await send({**receipt, "type": "prefill.memory_ready"})
                await send(
                    dict(
                        type="prefill.failed",
                        request_id=request_id,
                        code="CACHE_PERSISTENCE_UNAVAILABLE",
                        content="Saving the cache is unavailable on this service.",
                        memory_ready=True,
                        fatal=False,
                    )
                )
                return
            if persist and managed is not None:
                # The DiT owns L2; Room polls its authenticated, metadata-only
                # endpoint off the GPU command loop. Never read Encoder disk.
                await send(
                    {
                        **receipt,
                        "type": "prefill.memory_ready",
                        "_role_persistence_descriptor": descriptor,
                    }
                )
            elif not persist:
                await send(receipt)
            else:
                memory_ms = (time.monotonic() - started) * 1000
                await send(
                    {
                        **receipt,
                        "type": "prefill.memory_ready",
                        "prefill_to_memory_ms": memory_ms,
                    }
                )
                deadline = time.monotonic() + timeout
                root = api.get_global_server_args().zing_block0_cache_dir
                last_progress = time.monotonic()
                while time.monotonic() < deadline:
                    try:
                        status = await _persistence_read(
                            root,
                            descriptor,
                            max(0.01, min(10, deadline - time.monotonic())),
                        )
                    except asyncio.TimeoutError:
                        status = {"state": "pending"}
                    if status["state"] == "persisted":
                        await send(
                            {
                                **receipt,
                                "type": "prefill.persisted",
                                "storage": status,
                                "prefill_to_memory_ms": memory_ms,
                                "prefill_to_persisted_ms": (time.monotonic() - started)
                                * 1000,
                            }
                        )
                        break
                    if status["state"] == "failed":
                        await send(
                            dict(
                                type="prefill.failed",
                                request_id=request_id,
                                code=status["code"],
                                content="The cache could not be saved. Please try again later.",
                                memory_ready=True,
                                fatal=False,
                            )
                        )
                        break
                    if time.monotonic() - last_progress >= 5:
                        await send(
                            dict(type="prefill.persisting", request_id=request_id)
                        )
                        last_progress = time.monotonic()
                    await asyncio.sleep(min(0.25, max(0, deadline - time.monotonic())))
                else:
                    await send(
                        dict(
                            type="prefill.failed",
                            request_id=request_id,
                            code="PREFILL_PERSISTENCE_TIMEOUT",
                            content="Saving the cache has not been confirmed yet.",
                            memory_ready=True,
                            persistence_state="unknown",
                            fatal=False,
                        )
                    )
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
            if watching:
                _PERSISTENCE_WATCHERS -= 1
            active = None

    try:
        await send(
            dict(
                type="prefill.ready",
                max_message_bytes=maximum,
                capabilities={
                    "persistence": True,
                    "independent_preview": True,
                    "timing_schema_version": 1,
                },
            )
        )
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
