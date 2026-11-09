# SPDX-License-Identifier: Apache-2.0
"""Admission-owned media children on the original RenderHost and SDK source.

The admission remains the membership/input owner. This adapter only borrows its
original source iterator, and a media read never starts or retries GPU work.
"""

import asyncio
import base64
import copy
import hashlib
import hmac
import os
import secrets
import time
from uuid import uuid4

import msgspec

from .cli_room import publication_identity
from .diagnostics import log_player_failure
from .native_pixels import read_frame_identity, safe_int
from .render_contract import PROTOCOL, canonical, source_create_matches
from .room_control import exact, frozen_publication_identity
from .room_pipeline import LocalRoomEngine
from .rtc_auth import RTCCapabilities


def validate_media_launch(value, publication_id):
    fields = {"transport", "publicationIdentity", "initMsgpackBase64"}
    if isinstance(value, dict) and "memberPrefill" in value:
        from .member_prefill import validate_profile

        validate_profile(value["memberPrefill"])
        fields.add("memberPrefill")
    exact(value, fields)
    publication = publication_identity(value["publicationIdentity"])
    if (
        value["transport"] != "webrtc-rtp"
        or publication["publicationId"] != publication_id
    ):
        raise ValueError("Invalid member media publication or transport")
    try:
        encoded = value["initMsgpackBase64"]
        if not isinstance(encoded, str) or len(encoded) > 12 * 1024**2:
            raise ValueError()
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 8 * 1024**2:
            raise ValueError()
        init = msgspec.msgpack.decode(raw)
        world = init.get("whitebox", {})
        if (
            init.get("type") != "init"
            or init.get("control_protocol") != PROTOCOL
            or not source_create_matches(init)
            or world.get("source_id") != publication["sourceId"]
            or world.get("world_id") != publication["worldId"]
            or "game_id" in world
            or not isinstance(init.get("startup"), dict)
            or "memberPrefill" in value
            and init.get("inference_control") != "manual"
        ):
            raise ValueError()
    except (ValueError, TypeError, AttributeError, msgspec.DecodeError):
        raise ValueError("Invalid frozen member media launch") from None
    return init


def member_binding(record):
    return dict(
        record["binding"],
        memberId=record["member_id"],
        admissionOperationId=record["admission"]["operationId"],
    )


def sdk_identity(descriptor):
    return {
        key: copy.deepcopy(descriptor[key])
        for key in (
            "sessionId",
            "epoch",
            "worldBuildHash",
            "uiBundleHash",
            "playerBinding",
        )
    }


class MemberSource:
    """Trusted factory for one existing member, never a second Room join."""

    def __init__(self, owner, record):
        self.owner, self.runtime, self.record = owner, owner.host.runtime, record
        self.entry = self.runtime.player(record["stream_id"], touch=False)
        self.engine = self.entry["engine"]
        self.sdk = sdk_identity(self.engine.descriptor)
        self.reader = None
        self.model_binding = record.get("model_binding")

    def check(self):
        record = self.record
        entry = self.runtime.player(record["stream_id"], touch=False)
        room = self.runtime.rooms.get(record["binding"]["gameId"])
        if (
            entry is not self.entry
            or entry["engine"] is not self.engine
            or entry.get("control_admission") is not record
            or record["status"] != "claimed"
            or record.get("leaving")
            or record["binding"]["ownerEpoch"] != self.runtime.epoch
            or sdk_identity(entry["engine"].descriptor) != self.sdk
            or self.sdk["playerBinding"]["playerSessionId"] != record["member_id"]
            or room is None
            or room["control_record"].get("retiring")
            or (
                self.reader is not None and entry.get("media_reader") is not self.reader
            )
        ):
            raise ValueError("The original member media source has changed")
        frozen = room["control_record"]["invocation"]["createContext"][
            "frozenPublication"
        ]
        selected = self.record["mediaLaunch"]["publicationIdentity"]
        if frozen_publication_identity(frozen) != selected:
            raise ValueError("The original member media publication has changed")
        return entry

    @property
    def member_prefill(self):
        self.check()
        return self.record["mediaLaunch"].get("memberPrefill")

    def is_room_peer(self, other):
        """Only distinct original member sources may share a room's game ID."""
        if not isinstance(other, MemberSource) or other.owner is not self.owner:
            return False
        # An old/replaced source still owns its reservation until normal cleanup;
        # a new admission cannot use stale state as permission to overlap it.
        other.check()
        return (
            self.record["binding"] == other.record["binding"]
            and self.entry is not other.entry
            and self.engine is not other.engine
            and all(
                self.record[key] != other.record[key]
                for key in ("member_id", "player_id", "stream_id")
            )
            and self.sdk["sessionId"] != other.sdk["sessionId"]
            and self.sdk["playerBinding"]["viewportId"]
            != other.sdk["playerBinding"]["viewportId"]
        )

    async def open_session(self, *, world_id, spec, trace_id, context):
        entry = self.check()
        if (
            entry.get("media_reader") is not None
            or entry["read_lock"].locked()
            or spec != entry["spec"]
            or world_id != self.runtime.rooms[entry["game_id"]]["route"]["world_id"]
        ):
            raise ValueError("The original member source is already consumed")
        if context.get("start_suspended") is True:
            channel = getattr(self.engine, "model_context", None)
            if channel is not None:
                channel.suspended = True
        identity = copy.deepcopy(entry["join_receipt"])
        self.reader = MemberEngine(self, identity)
        # The SDK Room retains archive hashes. Its model Player instead verifies
        # the frozen publication already checked by this original admission.
        self.reader.metadata["artifact"] = copy.deepcopy(
            self.record["mediaLaunch"]["publicationIdentity"]
        )
        entry["media_reader"] = self.reader
        return self.reader

    def authorize_attachment(self, pipeline, message, user):
        return self.owner.authorize_attachment(pipeline, message, user)

    def frame(self, message):
        self.check()
        source = message.get("source") or {}
        pipeline = self.record["media_pipeline"]
        if (
            source.get("source_epoch") != pipeline.session.source_epoch
            or source.get("source_session_id") != pipeline.session_id
            or source.get("view_id") != self.record["stream_id"]
        ):
            raise ValueError("The original member source epoch has changed")
        frame = read_frame_identity(source["cli_frame_header"])
        if (
            frame["sessionId"] != self.sdk["sessionId"]
            or frame["epoch"] != self.sdk["epoch"]
        ):
            raise ValueError("The original member frame identity has changed")
        return frame


class MemberEngine(LocalRoomEngine):
    def __init__(self, source, identity):
        super().__init__(source.runtime, identity)
        self.source = source
        self.previous_idle = {
            key: source.entry.get(key)
            for key in ("explicit_source_owner", "owner_idle_seconds")
        }

    async def frames(self):
        while not self.closed:
            self.source.check()
            frame = await self.runtime.next_frame(self.stream_id, consumer=self)
            self.source.check()
            if frame is None:
                return
            from .contracts import EngineFrame

            yield EngineFrame(**frame)

    async def send_input(self, sequence, payload):
        raise ValueError("The SDK player owns member input")

    async def finish(self):
        pass  # A model ending must not finish the member's source.

    async def close(self):
        self.closed = True
        entry = self.source.entry
        if entry.get("media_reader") is self:
            entry.pop("media_reader")
            for key, value in self.previous_idle.items():
                if value is None:
                    entry.pop(key, None)
                else:
                    entry[key] = value


class MemberMedia:
    def __init__(self, host, *, now=time.time):
        self.host, self.now = host, now
        self.secret = secrets.token_urlsafe(32)
        host.runtime.member_media = self

    @property
    def bound_compute(self):
        return os.environ.get("WHITEBOX_BOUND_COMPUTE") == "1"

    async def reserve(self, record):
        """Prepare SDK join; eager/model-prefill launches also reserve a parent."""
        from ..player_lifecycle import sdk_cleanup_receipt
        from .model_binding import ModelBinding

        # SDK join completion is also needed when manual preview has no model
        # parent yet. Its later explicit enable owns the parent cleanup receipt.
        if "sdk_join_done" not in record:
            record["sdk_join_done"] = asyncio.Event()
            record["sdk_join_done"].set()
        if record.get("model_binding") is None:
            references = []
            if record.get("mediaLaunch") is not None:
                init = validate_media_launch(
                    record["mediaLaunch"], record["binding"]["publicationId"]
                )
                if (
                    init.get("inference_control") == "manual"
                    and "memberPrefill" not in record["mediaLaunch"]
                ):
                    return
                references = init.get("references") or (init.get("startup") or {}).get(
                    "initial_render", {}
                ).get("references", [])
            from .contracts import VideoSpec
            from .model_geometry import ModelGeometry

            model_size = ModelGeometry(VideoSpec(**record["spec"])).model
            lease = self.host.make_segment_leases(
                record["caller"]["subject"],
                dict(
                    segment_id=record["admission"]["operationId"],
                    model_size=f"{model_size.width}x{model_size.height}",
                    model_session_id=uuid4().hex,
                    generation_id=uuid4().hex,
                    capacity_wait_ms=0,
                    references=references,
                ),
            )
            lease.cleanup_timeout_s = 90

            async def sdk_cleanup(assignment):
                record["leaving"] = True
                await record["sdk_join_done"].wait()
                if record.get("sdk_join_started"):
                    receipt = await self.host.runtime.retire_member(record["stream_id"])
                else:
                    receipt = dict(
                        state="not_created",
                        source_id=record["stream_id"],
                        owner_epoch=self.host.runtime.epoch,
                    )
                return sdk_cleanup_receipt(assignment, **receipt)

            async def no_child_receipt(message):
                raise ValueError("No compute child exists before SDK admission")

            lease.sdk_cleanup_provider = sdk_cleanup
            record["model_binding"] = ModelBinding(
                lease, no_child_receipt, timeout=self.host.config.startup_seconds
            )
        await record["model_binding"].wait_ready()
        if record.get("leaving"):
            raise ValueError("The player expired while joining")

    async def start(self, record):
        if "media" in record:
            return  # The admission's retained outcome is never a retry license.
        record["media"] = dict(status="starting")
        pipeline = None
        try:
            init = validate_media_launch(
                record["mediaLaunch"], record["binding"]["publicationId"]
            )
            factory = MemberSource(self, record)
            factory.check()
            init["whitebox"]["game_id"] = record["binding"]["gameId"]

            async def reserved(value):
                nonlocal pipeline
                pipeline = value
                record["media_pipeline"] = value
                value.session.expected_publication = copy.deepcopy(
                    record["mediaLaunch"]["publicationIdentity"]
                )
                factory.check()

            await self.host.attach(
                dict(
                    user_id=record["caller"]["subject"],
                    trace_id=record["admission"]["operationId"],
                    message=init,
                ),
                transport="webrtc-rtp",
                detached=True,
                on_reserved=reserved,
                member_source=factory,
            )
        except BaseException as error:
            log_player_failure("member.media-start", error, record=record)
            if pipeline is not None:
                await pipeline.close("MEMBER_MEDIA_UNAVAILABLE")
            reason = (
                "CAPACITY_EXHAUSTED"
                if getattr(error, "code", None) == "CAPACITY_EXHAUSTED"
                else "MEMBER_MEDIA_UNAVAILABLE"
            )
            record["media"] = dict(status="unavailable", reason=reason)
            if self.bound_compute:
                record["status"] = "blocked"
                record["leaving"] = True

    async def close(self, record):
        task = record.get("media_task")
        if task is not None:
            import asyncio

            if task is not asyncio.current_task():
                await asyncio.shield(task)
        pipeline = record.get("media_pipeline")
        if pipeline is not None:
            await pipeline.close("MEMBER_LEFT")
            await pipeline.member_closed.wait()
        elif record.get("model_binding") is not None:
            await record["model_binding"].close()
        if "media" in record:
            record["media"] = dict(status="unavailable", reason="MEMBER_MEDIA_ENDED")

    def attachment(self, record, session, cursor=None):
        result = copy.deepcopy(
            record.get(
                "media",
                dict(status="unavailable", reason="MEMBER_MODEL_SOURCE_UNAVAILABLE"),
            )
        )
        pipeline = record.get("media_pipeline")
        if result["status"] == "unavailable" or pipeline is None:
            return dict(model=result)
        if (
            pipeline.closed
            or getattr(pipeline, "closing", False)
            or pipeline.terminal is not None
        ):
            return dict(model=dict(status="unavailable", reason="MEMBER_MEDIA_ENDED"))
        pipeline.member_source.check()
        if sdk_identity(session) != pipeline.member_source.sdk:
            raise ValueError("The member media SDK session has changed")
        if not getattr(pipeline, "control_started", False):
            return dict(model=dict(status="starting"))
        after = (
            (0 if pipeline.generation == 0 else pipeline.control_sequence)
            if cursor is None
            else cursor
        )
        if (
            not safe_int(after)
            or not pipeline.history_floor <= after <= pipeline.control_sequence
        ):
            raise ValueError("The member media replay position has expired")
        # Startup tuning stays on the existing source. The public member
        # attachment carries only the fields accepted by the SDK wire contract.
        startup_whitebox = pipeline.init["whitebox"]
        whitebox = {
            key: copy.deepcopy(startup_whitebox[key])
            for key in ("source_id", "world_id", "game_id", "scene_rendering")
            if key in startup_whitebox
        }
        message = dict(
            type="source.bind" if pipeline.session.binding_ended else "resume",
            control_protocol=PROTOCOL,
            whitebox=whitebox,
            source_session_id=pipeline.session_id,
            source_epoch=pipeline.session.source_epoch,
            owner_epoch=self.host.runtime.epoch,
            connection_generation=pipeline.generation,
            control_sequence=after,
            resume_user_id=pipeline.user_id,
            memberBinding=member_binding(record),
            sdkSession=copy.deepcopy(pipeline.member_source.sdk),
        )
        expires = int(self.now()) + 60
        claims = dict(
            aud="whitebox-member-attachment",
            sub=pipeline.user_id,
            exp=expires,
            messageSha256=hashlib.sha256(canonical(message)).hexdigest(),
        )
        payload = base64.urlsafe_b64encode(canonical(claims)).decode().rstrip("=")
        signature = (
            base64.urlsafe_b64encode(
                hmac.new(
                    self.secret.encode(), payload.encode(), hashlib.sha256
                ).digest()
            )
            .decode()
            .rstrip("=")
        )
        segment = pipeline.session.current_segment
        model_ready = (
            segment is not None
            and segment.segment_epoch == pipeline.session.current_model_stream
            and segment.first_decodable
            and not segment.closed_event.is_set()
        )
        return dict(
            kind="whitebox",
            model=dict(status="ready" if model_ready else "starting"),
            execution=dict(
                gameId=pipeline.game_id,
                sourceSessionId=pipeline.session_id,
                sourceEpoch=pipeline.session.source_epoch,
                ownerId=self.host.runtime.owner_id,
                ownerEpoch=self.host.runtime.epoch,
                publicationId=record["binding"]["publicationId"],
            ),
            attachment=dict(
                expiresAt=expires,
                transport="webrtc-rtp",
                message=dict(message, member_authorization=payload + "." + signature),
            ),
        )

    def authorize_attachment(self, pipeline, message, user):
        try:
            pipeline.member_source.check()
            claims = RTCCapabilities(self.secret)._decode(
                message.get("member_authorization")
            )
            unsigned = {k: v for k, v in message.items() if k != "member_authorization"}
            if (
                claims.get("aud") != "whitebox-member-attachment"
                or claims.get("sub") != user
                or type(claims.get("exp")) is not int
                or not self.now() < claims["exp"]
                or claims.get("messageSha256")
                != hashlib.sha256(canonical(unsigned)).hexdigest()
                or message.get("connection_generation") != pipeline.generation
                or message.get("memberBinding")
                != member_binding(pipeline.member_source.record)
                or message.get("sdkSession") != pipeline.member_source.sdk
            ):
                raise ValueError()
            return dict(message, resume_token=pipeline.token)
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "The member attachment authorization has expired"
            ) from None
