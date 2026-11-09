# SPDX-License-Identifier: Apache-2.0
"""Version-two Room bindings. A lost viewer never directly closes its source."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import logging
import os
import time
from dataclasses import replace

import msgspec
from fastapi import Header, HTTPException, WebSocket
from fastapi.responses import JSONResponse

from .area_contract import AREA_PROTOCOL, read_area_frame, read_area_receipt
from .interactive import create_model_leases
from .render_contract import (
    PROTOCOL,
    RenderError,
    RenderJournal,
    SQLiteRenderStore,
    canonical,
    source_create_matches,
)
from .render_source import SourceSession, configured_render_play_seconds
from .room_pipeline import RoomPipeline, Subscription
from .rooms import checked_id, new_id, require_internal_token


class RenderSubscription(Subscription):
    def __init__(self, owner, generation):
        super().__init__(owner, generation)
        self.stream_ids = dict(
            whitebox=owner.session.source_epoch,
            model=owner.session.current_model_stream,
        )
        self.area_challenge = new_id()
        self.inactive_model_acknowledged = False
        self.model_eos = None
        self.model_eos_sent = None
        if self.stream_ids["model"] is None:
            self.windows["model"] = (
                16  # Inactive streams must not time out waiting for ACKs.
            )

    def reset_model(self, stream_id):
        self.model_eos = self.model_eos_sent = None
        self.media_bytes -= sum(item[1] for item in self.media["model"])
        self.media["model"].clear()
        self.inflight["model"].clear()
        self.inflight_bytes["model"] = 0
        if self.large_au_stream == "model":
            self.large_au_stream = None
        self._delivery_snapshot()
        self.stream_ids["model"] = stream_id
        self.inactive_model_acknowledged = False
        self.sent["model"] = self.received["model"] = -1
        self.windows["model"] = None if stream_id else 16
        self.needs_key["model"] = True
        self.last_ack["model"] = time.monotonic()
        self.wake.set()

    def offer_control(self, message):
        segment = message.get("segment") or {}
        stream_id = self.stream_ids["model"]
        if (
            message.get("type") in {"render.status", "source.status"}
            and segment.get("state") in {"ended", "cancelled", "failed"}
            and stream_id is not None
            and segment.get("segment_epoch") == stream_id
            and self.model_eos_sent != stream_id
        ):
            self.model_eos = dict(
                version=2,
                type="render.eos",
                source_epoch=self.owner.session.source_epoch,
                media_stream_id=stream_id,
                segment_id=segment.get("segment_id"),
                connection_generation=self.generation,
            )
        super().offer_control(message)

    async def recv(self):
        # Control status can overtake queued media. Emit a per-attachment EOS
        # only after every accepted model AU has been sent, before source-only
        # traffic can keep the decoder's final retained frame waiting forever.
        if (
            not self.closed
            and self.model_eos
            and not self.control
            and not self.media["model"]
        ):
            message = dict(self.model_eos, frames=self.sent["model"] + 1)
            self.model_eos_sent = message["media_stream_id"]
            self.model_eos = None
            return msgspec.msgpack.encode(message)
        return await super().recv()

    def offer_media(self, message):
        if message.get("media_stream_id") != self.stream_ids.get(message.get("stream")):
            return
        super().offer_media(message)

    def ack(self, payload):
        if payload.get("media_stream_id") != self.stream_ids.get(payload.get("stream")):
            return  # A late ACK must not grant credit to a replacement stream.
        if (
            payload.get("stream") == "model"
            and self.stream_ids["model"] is None
            and not self.inactive_model_acknowledged
        ):
            # The inactive sentinel suppresses a timeout; it is not a client
            # window negotiation. Accept the first legal grant (including 32).
            self.windows["model"] = None
            super().ack(payload)
            self.inactive_model_acknowledged = True
            return
        super().ack(payload)


class RenderPipeline(RoomPipeline):
    command_byte_limit = 8 * 1024**2 + 16384
    media_transport = "websocket"
    reliable_control_types = RoomPipeline.reliable_control_types | {
        "startup.accepted",
        "startup.progress",
        "startup.asset.accepted",
        "startup.assets.committed",
        "model.prepared",
        "render.status",
        "render.stream",
        "source.status",
        "operation.result",
        "operation.rejected",
        "prompt.applied",
        "area.offer",
        "area.context",
        "model.context.intervention.result",
        "model.context.instruction.receipt",
        "prefill.accepted",
        "prefill.computing",
        "prefill.memory_ready",
        "prefill.persisted",
        "prefill.failed",
        "prefill.source_resumed",
    }

    def create_session(self, **kwargs):
        epoch = new_id()
        journal = RenderJournal(
            self.host.store,
            user_id=self.user_id,
            source_session_id=self.session_id,
            source_epoch=epoch,
            owner_epoch=self.host.runtime.epoch,
        )
        return SourceSession(
            self.host,
            journal,
            source_epoch=epoch,
            **{
                **kwargs,
                "config": replace(
                    kwargs["config"], play_seconds=configured_render_play_seconds()
                ),
            },
        )

    def create_subscription(self):
        self.session.resolve_area_offer = self.resolve_area_offer
        return RenderSubscription(self, self.generation)

    def area_authority(self, require_model=True, *, allow_interactive=False):
        sub = self.subscription
        native = getattr(self.session, "area_state", None)
        state = self.session.journal.snapshot or {}
        status = native.get("status", {}) if native else {}
        if (
            sub is None
            or sub.closed
            or sub.generation != self.generation
            or self.session.binding_ended
            or self.media_transport not in {"websocket", "webrtc-rtp"}
            or not (
                status.get("phase") == "awaiting-presented"
                and status.get("inputHeld") is True
                or allow_interactive
                and status.get("phase") == "interactive"
                and status.get("inputHeld") is False
            )
            or not status.get("transition")
        ):
            return None
        mode = state.get("display_mode", "whitebox")
        segment = state.get("segment")
        if (
            require_model
            and mode == "model"
            and (
                not segment
                or segment.get("state") not in {"initializing", "streaming"}
                or segment.get("segment_epoch") != self.session.current_model_stream
            )
        ):
            return None
        return dict(
            identity=copy.deepcopy(status["transition"]),
            runtime=copy.deepcopy(native["runtime"]),
            display=dict(mode=mode, revision=state.get("display_revision", 0)),
            connection_generation=self.generation,
            challenge=sub.area_challenge,
            **(
                {
                    "condition": dict(
                        segment_id=segment["segment_id"],
                        segment_epoch=segment["segment_epoch"],
                        render_revision=state["render_revision"],
                        prompt_version=segment["accepted_version"],
                        binding=segment.get(
                            "accepted_area_binding", segment.get("area_binding")
                        ),
                    )
                }
                if mode == "model" and segment
                else {}
            ),
        )

    def area_context(self):
        if (
            getattr(self.session, "area_state", None) is None
            or self.subscription is None
        ):
            return None
        authority = self.area_authority(require_model=False, allow_interactive=True)
        return dict(
            version=2,
            type="area.context",
            protocol=AREA_PROTOCOL,
            source_session_id=self.session.session_id,
            source_epoch=self.session.source_epoch,
            status=(
                self.session.area_state["status"]["phase"] if authority else "inactive"
            ),
            **(authority or {"connection_generation": self.generation}),
        )

    def area_offer(self, message):
        authority = self.area_authority()
        if not authority or message.get("stream") != authority["display"]["mode"]:
            return None
        try:
            evidence = read_area_frame((message.get("source") or {}).get("area_frame"))
        except ValueError:
            return None
        if canonical(evidence["identity"]) != canonical(authority["identity"]):
            return None
        output = {
            key: copy.deepcopy(message[key])
            for key in ("stream", "media_stream_id", "output_frame_start", "pts_us")
        }
        offered = dict(
            version=2,
            type="area.offer",
            protocol=AREA_PROTOCOL,
            source_session_id=self.session.session_id,
            source_epoch=self.session.source_epoch,
            **authority,
            frame=evidence["frame"],
            output=output,
        )
        if output["stream"] == "model":
            rendering = message.get("rendering") or {}
            actual = rendering.get("area_condition") or {}
            expected = authority["condition"]
            if (
                canonical(actual.get("binding")) != canonical(expected["binding"])
                or canonical((actual.get("binding") or {}).get("identity"))
                != canonical(authority["identity"])
                or actual.get("prompt_version") != expected["prompt_version"]
                or rendering.get("segment_id") != expected["segment_id"]
                or rendering.get("segment_epoch") != expected["segment_epoch"]
                or output["media_stream_id"] != expected["segment_epoch"]
            ):
                return None
            offered["condition"] = dict(expected, actual=copy.deepcopy(actual))
            # The opaque reference is retained only in the existing bounded reliable history.
            offered["model"] = dict(mapping_id=new_id())
        return offered

    def resolve_area_offer(self, command, *, allow_interactive=False):
        authority = self.area_authority(allow_interactive=allow_interactive)
        if not authority or command.get("connection_generation") != self.generation:
            raise RenderError("AREA_PRESENTATION_STALE")
        try:
            receipt = read_area_receipt(command.get("receipt"))
        except ValueError as error:
            raise RenderError("AREA_RECEIPT_INVALID") from error
        for _, _, offer in reversed(self.history):
            if offer.get("type") != "area.offer":
                continue
            if (
                command.get("challenge") != authority["challenge"]
                or offer.get("challenge") != authority["challenge"]
                or canonical(command.get("display")) != canonical(authority["display"])
                or canonical(offer["display"]) != canonical(authority["display"])
                or canonical(offer["identity"]) != canonical(authority["identity"])
                or canonical(offer["runtime"]) != canonical(authority["runtime"])
                or canonical(receipt["identity"]) != canonical(offer["identity"])
                or canonical(receipt["frame"]) != canonical(offer["frame"])
                or canonical(command.get("model")) != canonical(offer.get("model"))
            ):
                continue
            if authority["display"]["mode"] == "model":
                expected = authority["condition"]
                if any(
                    canonical(offer["condition"].get(key)) != canonical(value)
                    for key, value in expected.items()
                ):
                    continue
            return copy.deepcopy(offer)
        raise RenderError("AREA_PRESENTATION_STALE")

    def attachment_fields(self):
        return dict(
            control_protocol=PROTOCOL,
            cleanup_timeout_ms=90000,
            source_session_id=self.session_id,
            source_epoch=self.session.source_epoch,
            view_id=self.stream_id,
        )

    async def start(self):
        expected = getattr(self.session, "expected_publication", None)
        if getattr(self, "member_source", None) is not None:
            self.member_source.check()
        elif expected is not None:
            from pathlib import Path

            from .cli_room import prepare_publication

            # Read-only authorization precedes SourceSession.send/PlayStartup,
            # so a replaced publication cannot start model conditioning.
            world = expected["sourceId"] + "/" + expected["worldId"]
            await prepare_publication(
                Path(os.environ["WHITEBOX_CLI_ENTRYPOINT"]),
                world,
                self.host.config.startup_seconds,
                expected_publication=expected,
            )
        await self.session.journal.create()
        await self.session.journal.update(
            lambda state: state.update(
                game_id=self.game_id,
                stream_id=self.stream_id,
                world=self.init["whitebox"],
                capability_sha256=hashlib.sha256(self.token.encode()).hexdigest(),
            )
        )
        await super().start()
        self.control_started = True

    async def close(self, reason=None):
        relay = getattr(self, "sdk_control_relay", None)
        if relay is not None:
            relay.stop()
            if relay.task is not None:
                await asyncio.gather(relay.task, return_exceptions=True)
        terminal = self.terminal or {}
        failing = (
            hasattr(self, "single_player_operation")
            and not self.closed
            and not getattr(self, "closing", False)
            and terminal.get("code")
            in {"WTP_PLAYABLE_PUBLICATION_CHANGED", "PLAYER_IDLE_TIMEOUT"}
        )
        if not failing:
            await super().close(reason)
            return
        control, key = self.host.single_player_control, self.single_player_operation
        # Preserve the known terminal reason across registry removal and its
        # durable write, including idle retirement after a successful launch.
        control.pending_source_failures.add(key)
        try:
            await super().close(reason)
        finally:
            try:
                if self.closed:
                    await control.source_failed(key, terminal["code"])
            finally:
                control.pending_source_failures.discard(key)

    def ready_message(self):
        value = super().ready_message()
        state = self.session.journal.snapshot or {}
        value.update(
            room_pipeline_protocol="room-render-v2",
            sdk_control_relay=getattr(self, "member_source", None) is not None,
            control_protocol=PROTOCOL,
            source_epoch=self.session.source_epoch,
            play_remaining_seconds=self.session.player_remaining_seconds,
            render_revision=state.get("render_revision", 0),
            display_revision=state.get("display_revision", 0),
            model_context_segment=self.session.model_context_segment(state),
            inference=copy.deepcopy(state.get("inference")),
            current_segment={
                k: v
                for k, v in (state.get("segment") or {}).items()
                if k
                not in {
                    "prompt_queue",
                    "prompt_events",
                    "validating_prompt",
                    "periodic_refresh",
                    "refresh_condition_version",
                }
            }
            or None,
            model_stream_id=self.session.current_model_stream,
        )
        if "source_encoding" in self.init.get("whitebox", {}):
            value["source_encoding"] = copy.deepcopy(
                self.session.config.source_encoding
            )
        segment = getattr(self.session, "current_segment", None)
        if (
            getattr(self.session.config, "model_target", None) is not None
            and segment
            and segment.playback
        ):
            value["model_video_stream"] = segment.playback.stream
        if value.get("video_stream") is None:
            value["video_stream"] = value["source_video_stream"]
        if (
            self.media_transport in {"websocket", "webrtc-rtp"}
            and value.get("engine", {}).get("area_protocol") == AREA_PROTOCOL
        ):
            value["capabilities"] = {
                **value.get("capabilities", {}),
                "area_presented": AREA_PROTOCOL,
                "area_mapping": dict(
                    whitebox=True, model=True, transports=[self.media_transport]
                ),
            }
        return value

    def request_keyframe(self, stream):
        if stream == "whitebox":
            return super().request_keyframe(stream)
        segment = getattr(self.session, "current_segment", None)
        if segment and segment.playback:
            segment.playback.request_keyframe()

    def publish(self, message):
        if message.get("type") in {"whitebox.frame", "frame_batch"}:
            context = self.area_context()
            previous = next(
                (
                    row[2]
                    for row in reversed(self.history)
                    if row[2].get("type") == "area.context"
                ),
                None,
            )
            if context is not None and canonical(context) != canonical(
                {
                    key: value
                    for key, value in (previous or {}).items()
                    if key != "control_sequence"
                }
            ):
                super().publish(context)
            offer = self.area_offer(message)
            if offer is not None:
                super().publish(offer)
                if self.media_transport == "webrtc-rtp":
                    # Carry this exact history offer with its RTP-mapped frame.
                    # Newer reliable offers can overtake queued or dropped video.
                    message = dict(message, area_offer=copy.deepcopy(offer))
        if message.get("type") == "render.stream" and self.subscription:
            self.subscription.reset_model(message.get("media_stream_id"))
        super().publish(message)

    def detach(self, sub):
        sub.stop("DISCONNECTED")
        if self.subscription is not sub:
            return
        self.subscription = None
        relay = getattr(self, "sdk_control_relay", None)
        if relay is not None:
            relay.stop()
        if self.retention:
            self.retention.cancel()

        async def expire():
            await asyncio.sleep(self.host.grace_seconds)
            if self.subscription is None:
                await self.close("PLAYER_DISCONNECTED")

        self.retention = asyncio.create_task(expire())

    def accept(self, sub, wire):
        if (
            sub is not self.subscription
            or sub.closed
            or sub.generation != self.generation
        ):
            raise RenderError("SOURCE_EPOCH_MISMATCH")
        if len(wire) > 8 * 1024**2 + 16384:
            raise RenderError("LIMIT_EXCEEDED")
        message = msgspec.msgpack.decode(wire)
        if isinstance(message, dict) and message.get("type") in {
            "sdk.control.open",
            "sdk.control",
        }:
            try:
                if getattr(self, "member_source", None) is None:
                    raise RenderError("CAPABILITY_UNAVAILABLE")
                self.member_source.check()
                from .sdk_control_relay import SDKControlRelay

                relay = getattr(self, "sdk_control_relay", None)
                if message["type"] == "sdk.control.open":
                    if set(message) != {"type", "client_id"}:
                        raise RenderError("INVALID_COMMAND")
                    if (
                        relay is not None
                        and not relay.closed
                        and relay.subscription is sub
                    ):
                        if relay.client_id != message["client_id"]:
                            raise RenderError("INVALID_COMMAND")
                        return
                    if relay is not None:
                        relay.stop()
                    relay = SDKControlRelay(self, sub, message["client_id"])
                    self.sdk_control_relay = relay
                    relay.task = asyncio.create_task(relay.run())
                else:
                    if set(message) != {"type", "payload"}:
                        raise RenderError("INVALID_COMMAND")
                    if relay is None or relay.subscription is not sub:
                        raise RenderError("CONTROL_UNAVAILABLE")
                    relay.enqueue(message["payload"])
            except RenderError as error:
                sub.offer_control(
                    dict(type="sdk.control", state="rejected", code=error.code)
                )
            return
        if getattr(self, "member_source", None) is not None:
            self.member_source.check()
            # Declared context owners may reserve lifecycle or relay instructions.
            # The ordered SourceSession/journal path still validates its exact
            # fields, source scope and revision; no prompt or seed is accepted.
            allowed = {
                "status.get",
                "operation.get",
                "display.configure",
                "model.inference.configure",
                "model.sink",
            }
            member_prefill = getattr(self.session, "member_prefill", None)
            if member_prefill is not None:
                allowed.discard("model.inference.configure")
                allowed.update(
                    {
                        "render.prepare",
                        "render.prepare.cancel",
                        "render.start",
                        "render.end",
                    }
                )
            prefill_event = (
                member_prefill is not None
                and message.get("type") == "event"
                and message.get("eventType") == "prefill"
            )
            if prefill_event:
                member_prefill.validate_conditions(message.get("payload", {}))
            if (
                getattr(self.session, "world_driven", False)
                and getattr(self.session, "context_channel", None) is not None
            ):
                allowed.update(
                    {
                        "render.context.start",
                        "model.context.intervention",
                        "model.context.intervention.get",
                    }
                )
            elif (
                getattr(getattr(self.session, "inference", None), "manual", False)
                and member_prefill is None
            ):
                allowed.update(
                    {"model.context.intervention", "model.context.intervention.get"}
                )
            forbidden_command = (
                message.get("type") == "command" and message.get("op") not in allowed
            )
            if (
                forbidden_command
                or message.get("eventType") is not None
                and not prefill_event
                or message.get("kind") in {"input", "prompt", "finish", "mock.frame"}
                or message.get("type") in {"init", "asset.upload"}
            ):
                raise ValueError(
                    "The SDK player owns member input and frozen model launch"
                )
        frozen_startup = self.session.startup or getattr(
            getattr(self.session, "inference", None), "startup", None
        )
        if message.get("type") == "init" and frozen_startup is not None:
            from .startup import startup_fingerprint

            startup = frozen_startup
            same_world = all(
                message.get("whitebox", {}).get(key) == self.init["whitebox"].get(key)
                for key in (
                    "source_id",
                    "world_id",
                    "cli_fps",
                    "scene_rendering",
                    "resolution",
                    "source_encoding",
                )
            )
            if (
                startup.closed
                or not same_world
                or message.get("inference_control")
                != self.init.get("inference_control")
                or message.get("source_operation") != self.init.get("source_operation")
                or startup_fingerprint(message.get("startup"))
                != startup.request_fingerprint
            ):
                raise RenderError("REQUEST_ID_CONFLICT")
            self.publish(
                dict(
                    type="startup.accepted",
                    protocol=startup.request["protocol"],
                    request_id=startup.request["request_id"],
                    max_asset_chunk_bytes=1 << 20,
                    duplicate=True,
                )
            )
            return
        if message.get("type", "").startswith("startup."):
            startup = frozen_startup
            if (
                startup is None
                or message.get("request_id") != startup.request["request_id"]
                or startup.closed
            ):
                raise RenderError("INVALID_COMMAND")
            kind = message["type"]
            if kind == "startup.asset.chunk":
                end = startup.assets.chunk(
                    message.get("asset_id"),
                    message.get("offset"),
                    message.get("payload"),
                )
                self.publish(
                    dict(
                        type="startup.asset.accepted",
                        request_id=message["request_id"],
                        asset_id=message["asset_id"],
                        offset=end,
                    )
                )
            elif kind == "startup.assets.commit":
                startup.assets.commit()
                self.publish(
                    dict(
                        type="startup.assets.committed",
                        request_id=message["request_id"],
                    )
                )
            else:
                raise RenderError("INVALID_COMMAND")
            return
        if (
            message.get("type") in {"command", "asset.upload"}
            or message.get("eventType") is not None
        ):
            limit = (
                8 * 1024**2 + 16384
                if message.get("type") == "asset.upload"
                else 128 * 1024
            )
            if len(wire) > limit:
                raise RenderError("LIMIT_EXCEEDED")
            self._enqueue_command(sub, wire, message)
            return
        super().accept(sub, wire)


class RenderHost:
    def __init__(
        self,
        runtime,
        config,
        archives,
        reports,
        coordinator,
        connect,
        *,
        max_sources=1,
        grace_seconds=15,
    ):
        self.runtime, self.config, self.archives, self.reports = (
            runtime,
            config,
            archives,
            reports,
        )
        self.coordinator, self.connect = coordinator, connect
        self.max_sessions, self.grace_seconds, self.history_bytes = (
            max_sources,
            grace_seconds,
            16 * 1024**2,
        )
        from .condition_refresh import ConditionRefreshPolicy

        self.condition_refresh_policy = ConditionRefreshPolicy(
            runtime.spool / ".runtime" / "condition-refresh.json"
        )
        self.sessions = {}
        self.lock = asyncio.Lock()
        self.store = SQLiteRenderStore(runtime.spool / ".render-v2" / "control.sqlite3")
        from .player_media import MemberMedia

        self.member_media = MemberMedia(self)

    def make_segment_leases(self, user_id, segment):
        revision = os.environ["WHITEBOX_ROUTING_REVISION"]
        large_revision = os.environ.get("WHITEBOX_LARGE_REFERENCE_MODEL_REVISION")
        default_limit = int(os.environ.get("WHITEBOX_DEFAULT_REFERENCE_CAPACITY", "3"))
        if large_revision and len(segment.get("references") or []) > default_limit:
            revision = large_revision
        return create_model_leases(
            self.coordinator,
            self.connect,
            user_id=user_id,
            profile_hint=segment.get("model_size"),
            trace_id=segment["segment_id"],
            model_revision=revision,
            vae_fingerprint="local",
            max_bytes=self.config.max_message_bytes,
            strict_release=True,
            session_id=segment["model_session_id"],
            generation_id=segment["generation_id"],
            capacity_wait_ms=segment["capacity_wait_ms"],
        )

    async def attach(
        self,
        body,
        *,
        transport="websocket",
        rtc_peer=None,
        preparation=None,
        on_reserved=None,
        detached=False,
        member_source=None,
    ):
        from .diagnostics import cli_fps_override

        user, trace, message = (
            body.get("user_id"),
            body.get("trace_id"),
            body.get("message"),
        )
        if (
            not isinstance(user, str)
            or not user
            or len(user) > 256
            or not isinstance(message, dict)
        ):
            raise RenderError("INVALID_SOURCE")
        if message.get("control_protocol") != PROTOCOL:
            raise RenderError("CAPABILITY_UNAVAILABLE")
        kind = message.get("type")
        if kind == "init" and isinstance(message.get("whitebox"), dict):
            from .session_encoding import check_source_encoding_scope

            check_source_encoding_scope(message, transport=transport)
            cli_fps_override(message["whitebox"].get("cli_fps"))
            from .model_geometry import session_source_spec

            session_source_spec(
                message["whitebox"], self.config.spec, self.runtime.rooms
            )
        async with self.lock:
            if getattr(self.runtime, "maintenance", None) is not None:
                await self.runtime.maintenance.cancel_for_play()
            reserved = getattr(self.runtime, "rtc_preparation", None)
            if reserved is not None and reserved != preparation:
                raise RenderError("CAPACITY_EXHAUSTED")
            if kind in {"resume", "source.bind"}:
                source_id = message.get("source_session_id", message.get("session_id"))
                pipeline = self.sessions.get(checked_id(source_id))
                if (
                    pipeline is None
                    or pipeline.user_id != user
                    or pipeline.game_id != message["whitebox"]["game_id"]
                    or message.get("owner_epoch") != self.runtime.epoch
                ):
                    raise RenderError("ROOM_SESSION_LOST")
                if message.get("source_epoch") != pipeline.session.source_epoch:
                    raise RenderError("SOURCE_EPOCH_MISMATCH")
                if pipeline.media_transport != transport:
                    raise RenderError("CAPABILITY_UNAVAILABLE")
                from .session_encoding import frozen_source_encoding

                if "source_encoding" in message["whitebox"]:
                    frozen_source_encoding(
                        message["whitebox"], pipeline.init["whitebox"]
                    )
                source = pipeline.session
                if (
                    source.binding_ended
                    or getattr(source, "closing", False)
                    or getattr(source, "close_requested", False)
                    or source.player_remaining_seconds <= 0
                ):
                    raise RenderError("PLAYER_EXPIRED")
                # Reattach the same valid identity; never revive an ended player
                # or reset its reservation-time deadline.
                if hasattr(pipeline, "single_player_operation"):
                    message = self.single_player_control.authorize_attachment(
                        pipeline, message, user
                    )
                if getattr(pipeline, "member_source", None) is not None:
                    message = pipeline.member_source.authorize_attachment(
                        pipeline, message, user
                    )
                if rtc_peer is not None:
                    pipeline.rtc_peer = rtc_peer
                subscription = pipeline.attach(message)
                return pipeline, subscription
            if kind != "init" or not source_create_matches(message):
                raise RenderError("INVALID_SOURCE")
            requested_startup = message.get("startup")
            if isinstance(requested_startup, dict) and any(
                pipeline.user_id == user
                and (
                    startup := pipeline.session.startup
                    or getattr(
                        getattr(pipeline.session, "inference", None), "startup", None
                    )
                )
                is not None
                and startup.request["request_id"] == requested_startup.get("request_id")
                for pipeline in self.sessions.values()
            ):
                # Reconnection uses the existing capability/epoch fenced resume
                # route. A second create cannot silently reserve another slot.
                raise RenderError("STARTUP_ALREADY_EXISTS")
            if (
                len(self.sessions) + len(getattr(self, "peer_sessions", {}))
                >= self.max_sessions
            ):
                raise RenderError("CAPACITY_EXHAUSTED")
            if member_source is not None:
                from .player_media import MemberSource

                if (
                    not isinstance(member_source, MemberSource)
                    or member_source.owner is not self.member_media
                    or transport != "webrtc-rtp"
                    or user != member_source.record["caller"]["subject"]
                    or message["whitebox"]["game_id"]
                    != member_source.record["binding"]["gameId"]
                ):
                    raise RenderError("INVALID_SOURCE")
                member_source.check()
            if any(
                p.game_id == message["whitebox"]["game_id"]
                and (
                    member_source is None
                    or not member_source.is_room_peer(getattr(p, "member_source", None))
                )
                for p in self.sessions.values()
            ):
                raise RenderError("INVALID_SOURCE")
            if transport == "webrtc-rtp":
                if os.environ.get("WHITEBOX_NATIVE_RTC_ENABLED") != "1":
                    raise RenderError("CAPABILITY_UNAVAILABLE")
                from .rtc_routes import RTCRenderPipeline

                pipeline = RTCRenderPipeline(
                    self,
                    message,
                    user,
                    trace,
                    peer=rtc_peer,
                    **(
                        {"member_source": member_source}
                        if member_source is not None
                        else {}
                    ),
                )
            elif transport == "websocket":
                pipeline = RenderPipeline(self, message, user, trace)
            else:
                raise RenderError("CAPABILITY_UNAVAILABLE")
            self.sessions[pipeline.session_id] = pipeline
            sub = None if detached else pipeline.attach()
        # Only registry insertion holds the host lock, never CLI or asset startup.
        try:
            if on_reserved is not None:
                await on_reserved(pipeline)
            await pipeline.start()
            frozen_startup = pipeline.session.startup or getattr(
                getattr(pipeline.session, "inference", None), "startup", None
            )
            if frozen_startup is not None:
                if transport == "websocket" and pipeline.session.startup is not None:
                    pipeline.session.startup.rtc_ready.set()
                pipeline.publish(
                    dict(
                        type="startup.accepted",
                        request_id=frozen_startup.request["request_id"],
                        protocol=frozen_startup.request["protocol"],
                        max_asset_chunk_bytes=1 << 20,
                    )
                )
        except BaseException:
            await pipeline.close("INITIALIZATION_FAILED")
            raise
        return pipeline, sub

    async def close(self):
        if hasattr(self, "single_player_control"):
            await self.single_player_control.shutdown()
        await asyncio.gather(
            *(p.close("ROOM_STOPPED") for p in list(self.sessions.values())),
            return_exceptions=True,
        )
        await self.store.close()

    def install(self, app, token):
        from .player_control import install_owner_single_player_control

        install_owner_single_player_control(app, self, token)
        if os.environ.get("WHITEBOX_NATIVE_RTC_ENABLED") == "1":
            from .rtc_routes import install_rtc_routes

            install_rtc_routes(self, app, token)

        @app.get("/v1/whitebox/render/sources/{source_id}")
        async def status(
            source_id: str,
            authorization: str = Header(default=""),
            operation_request_id: str | None = None,
        ):
            try:
                checked_id(source_id)
            except ValueError:
                raise HTTPException(404)
            state = await self.store.find_source(source_id)
            supplied = authorization.removeprefix("Bearer ")
            if (
                state is None
                or not authorization.startswith("Bearer ")
                or not hmac.compare_digest(
                    hashlib.sha256(supplied.encode()).hexdigest(),
                    state.get("capability_sha256", ""),
                )
            ):
                raise HTTPException(404)

            def segment(value):
                return {
                    k: v
                    for k, v in (value or {}).items()
                    if k
                    not in {
                        "prompt_queue",
                        "prompt_events",
                        "validating_prompt",
                        "periodic_refresh",
                        "refresh_condition_version",
                    }
                } or None

            current = segment(state["segment"])
            lost = state["owner_epoch"] != self.runtime.epoch
            if (
                lost
                and current
                and current["resource_state"] not in {"none", "released"}
            ):
                current.update(resource_state="release_unknown", state="failed")
            operation = (
                state["operations"].get(operation_request_id)
                if operation_request_id
                else None
            )
            if operation:
                operation = {
                    k: v for k, v in operation.items() if k != "payload_sha256"
                }
            return JSONResponse(
                dict(
                    source_session_id=source_id,
                    source_epoch=state["source_epoch"],
                    source_state="lost" if lost else state["source_state"],
                    inference=copy.deepcopy(state.get("inference")),
                    owner_epoch=state["owner_epoch"],
                    render_revision=state["render_revision"],
                    display_revision=state["display_revision"],
                    segment=current,
                    pending_replace=segment(state["pending_replace"]),
                    operation=operation,
                ),
                headers={"Cache-Control": "no-store"},
            )

        @app.websocket("/internal/render-v2")
        async def render(websocket: WebSocket):
            require_internal_token(token, websocket.headers.get("authorization", ""))
            await websocket.accept()
            pipeline = sub = None
            trace = None
            tasks = []
            try:
                wire = await asyncio.wait_for(websocket.receive_bytes(), 30)
                if len(wire) > 8 * 1024**2 + 16384:
                    raise RenderError("LIMIT_EXCEEDED")
                body = msgspec.msgpack.decode(wire)
                trace = body.get("trace_id") if isinstance(body, dict) else None
                pipeline, sub = await self.attach(body)

                async def receive():
                    while True:
                        pipeline.accept(sub, await websocket.receive_bytes())

                async def send():
                    while True:
                        wire = await sub.recv()
                        await asyncio.wait_for(websocket.send_bytes(wire), 10)
                        if msgspec.msgpack.decode(wire).get("type") in {
                            "session.completed",
                            "error",
                        }:
                            return

                tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            except Exception as error:
                logging.getLogger(__name__).warning(
                    "Render connection interrupted: %s",
                    type(error).__name__,
                    exc_info=True,
                )
                from .diagnostics import public_error

                try:
                    await websocket.send_bytes(
                        msgspec.msgpack.encode(
                            dict(
                                type="error",
                                **public_error(error, trace),
                                reconnectable=pipeline is not None,
                            )
                        )
                    )
                except Exception:
                    pass
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if pipeline and sub:
                    pipeline.detach(sub)
                try:
                    await websocket.close()
                except Exception:
                    pass
