# SPDX-License-Identifier: Apache-2.0
"""Metadata preflight and bounded API transport/media ownership.

Encoding belongs to the worker. These budgets cover actual WebSocket bodies,
image verification and session-owned encoded ingress bytes, independently of RPC state.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.conditioning.resources import (
    AdmissionLedger,
    BoundedExecutor,
)
from sglang.multimodal_gen.runtime.realtime.source_transport import (
    SourceTransportBudget,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    DEFAULT_MAX_MEDIA_ITEMS,
    DEFAULT_MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_IMAGE_PIXELS,
)


@dataclass
class ApiConditioningResources:
    ledger: AdmissionLedger
    cpu: BoundedExecutor
    tenant: str
    workspace_bytes: int = 64 << 20

    def reserve_api_message(self, size):
        return self.ledger.reserve(self.tenant, ram=2 * size, inflight=2 * size)


_API_POOLS = {}


def get_api_resources(session, server_args=None):
    resources = getattr(session, "conditioning_resources", None)
    if resources is not None:
        return resources
    if server_args is None:
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        server_args = get_global_server_args()
    message_bytes = SourceTransportBudget.from_args(server_args).message_bytes
    sessions = max(1, getattr(server_args, "realtime_max_sessions", 8))
    key = message_bytes, sessions
    if key not in _API_POOLS:
        # Each session may retain accepted source packets and one verification
        # workspace. SourceLedger separately enforces frame/packet capacities.
        ram_per_session = 8 * message_bytes + (64 << 20)
        ledger = AdmissionLedger(
            {
                "ram": (sessions * ram_per_session, ram_per_session),
                "inflight": (sessions * ram_per_session, ram_per_session),
            }
        )
        _API_POOLS[key] = ledger, BoundedExecutor(active=2, waiting=sessions)
    ledger, cpu = _API_POOLS[key]
    resources = ApiConditioningResources(ledger, cpu, session.id)
    session.conditioning_resources = resources
    return resources


def validate_remote_media_metadata(media, *, max_media=DEFAULT_MAX_MEDIA_ITEMS):
    """Only metadata is read here; decoding and encoding remain worker-owned."""
    if not isinstance(media, (list, tuple)) or len(media) > max_media:
        raise ConditionError("CONDITION_INVALID", "Invalid conditioning media metadata")
    for item in media:
        if not isinstance(item, dict):
            raise ConditionError("CONDITION_INVALID", "Invalid conditioning media item")
        kind = item.get("type")
        if kind == "image":
            continue
        clip = item.get("clip")
        if (
            kind != "video"
            or not isinstance(clip, dict)
            or set(clip) != {"start", "end"}
            or any(
                type(clip[name]) not in (int, float)
                or not math.isfinite(clip[name])
                or clip[name] < 0
                for name in ("start", "end")
            )
            or clip["start"] != clip["end"]
        ):
            raise ConditionError(
                "CONDITION_INVALID",
                "Conditioning requires images or single-frame video",
            )


def validate_remote_request(request, server_args):
    if hasattr(request, "user_media"):
        raise ConditionError(
            "CONDITION_INVALID", "Please use source_stream, references and prompt."
        )
    if getattr(server_args, "zing_conditioning_mode", "local") != "remote":
        return
    max_media = (
        getattr(
            getattr(server_args, "pipeline_config", None),
            "zing_reference_max_images",
            DEFAULT_MAX_REFERENCE_IMAGES,
        )
        + 1
    )
    custom = getattr(request, "zing_request", None)
    if custom is not None:
        if not isinstance(custom, dict) or not isinstance(custom.get("messages"), list):
            raise ConditionError(
                "CONDITION_INVALID", "Invalid conditioning request metadata"
            )
        media = []
        for message in custom["messages"]:
            if not isinstance(message, dict):
                raise ConditionError(
                    "CONDITION_INVALID", "Invalid conditioning message metadata"
                )
            if message.get("role") == "user":
                content = message.get("content")
                if not isinstance(content, list) or any(
                    not isinstance(item, dict) for item in content
                ):
                    raise ConditionError(
                        "CONDITION_INVALID", "Invalid conditioning content metadata"
                    )
                media.extend(item for item in content if item.get("type") != "text")
        validate_remote_media_metadata(media, max_media=max_media)


def take_message_reservation(session):
    reservation = getattr(session, "_conditioning_message_reservation", None)
    session._conditioning_message_reservation = None
    return reservation


def release_message_reservation(session):
    reservation = take_message_reservation(session)
    if reservation is not None:
        reservation.release()


async def receive_conditioning_message(ws, session, server_args):
    """Reserve a bounded WebSocket message before asking the transport for it."""
    if getattr(server_args, "zing_conditioning_mode", "local") == "local":
        return await ws.receive_bytes()
    release_message_reservation(session)
    resources = get_api_resources(session, server_args)
    bound = SourceTransportBudget.from_args(server_args).message_bytes
    reservation = resources.reserve_api_message(bound)
    session._conditioning_message_reservation = reservation
    try:
        message = await ws.receive_bytes()
        if len(message) > bound:
            from sglang.multimodal_gen.runtime.conditioning.contract import (
                ConditionError,
            )

            raise ConditionError(
                "CONDITION_INVALID", "Streaming message exceeds its reserved size"
            )
        # Receiving keeps the actual bytes (and decoded message copy), rather
        # than retaining two maximum-sized bodies for every small init message.
        reservation.reduce(ram=2 * len(message), inflight=2 * len(message))
        return message
    except BaseException:
        release_message_reservation(session)
        raise


async def iter_conditioning_messages(ws, session, server_args):
    if getattr(server_args, "zing_conditioning_mode", "local") == "local":
        async for message in ws.iter_bytes():
            yield message
        return
    import msgspec.msgpack
    from fastapi import WebSocketDisconnect

    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    while True:
        try:
            message = await receive_conditioning_message(ws, session, server_args)
        except WebSocketDisconnect:
            return
        except ConditionError as error:
            await ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(type="error", content=error.public_message, code=error.code)
                )
            )
            return
        try:
            yield message
        finally:
            release_message_reservation(session)


def reserve_media_memory(session, byte_count):
    resources = get_api_resources(session)
    reservation = resources.ledger.reserve(
        resources.tenant, ram=byte_count, inflight=byte_count
    )
    if not hasattr(session, "_api_media_reservations"):
        session._api_media_reservations = []
    session._api_media_reservations.append(reservation)
    return reservation


def release_media_reservations(session, reservations):
    """Return media RAM credit after the owning dispatch or session completes."""
    released = set()
    for reservation in reservations:
        reservation.release()
        released.add(id(reservation))
    session._api_media_reservations = [
        item
        for item in getattr(session, "_api_media_reservations", [])
        if id(item) not in released
    ]


async def verify_presentation_upload(session, raw, format_name):
    from PIL import Image

    from sglang.multimodal_gen.runtime.realtime.zing_contracts import require

    def verify():
        with Image.open(io.BytesIO(raw)) as image:
            require(
                image.format.lower() == format_name
                and image.width * image.height <= MAX_REFERENCE_IMAGE_PIXELS,
                "The reference image format or dimensions are not supported.",
            )
            decoded_bytes = image.width * image.height * 3
            image.verify()
            return decoded_bytes

    resources = get_api_resources(session)
    workspace = resources.workspace_bytes
    reservation = resources.ledger.reserve(
        resources.tenant, ram=workspace, inflight=workspace
    )
    try:
        return await resources.cpu.run(verify)
    finally:
        reservation.release()


def release_api_media(session):
    release_media_reservations(session, getattr(session, "_api_media_reservations", []))
