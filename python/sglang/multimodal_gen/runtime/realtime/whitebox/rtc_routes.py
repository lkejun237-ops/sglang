# SPDX-License-Identifier: Apache-2.0
"""Opt-in RTC startup reservation and same-protocol attachment recovery."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import os
from dataclasses import dataclass

import msgspec
from fastapi import WebSocket

from .diagnostics import WhiteboxDiagnostic, public_error
from .native_rtc import NativeRTC
from .opening_presentation import PROTOCOL as FIRST_PRESENTED_PROTOCOL
from .playback import level31_supports
from .prewarm_observation import (
    OBSERVE,
    PrewarmPlayerObservations,
    is_prewarm_observation_request,
)
from .render_contract import PROTOCOL, RenderError, source_create_matches
from .render_host import RenderPipeline
from .rooms import checked_id, new_id, require_internal_token
from .rtc_transport import RTCSubscription

RTC_RECONNECT_SETUP_SECONDS = 13.0


@dataclass
class ReconnectPreparation:
    reservation: str
    deadline: float
    released: asyncio.Event


def release_reconnect_preparation(pipeline, reservation):
    preparation = getattr(pipeline, "rtc_reconnect", None)
    if preparation is not None and preparation.reservation == reservation:
        preparation.released.set()
        pipeline.rtc_reconnect = None


class RTCRenderPipeline(RenderPipeline):
    media_transport = "webrtc-rtp"

    def __init__(self, *args, peer, **kwargs):
        self.rtc_peer = peer
        self.rtc_source_profile = getattr(peer, "source_h264_profile", None)
        self.rtc_closed = asyncio.Event()
        self.rtc_reconnect = None
        self.rtc_reconnect_deadline = None
        super().__init__(*args, **kwargs)

    def create_subscription(self):
        self.session.resolve_area_offer = self.resolve_area_offer
        sub = RTCSubscription(self, self.generation, self.rtc_peer)
        self.session.resolve_first_presentation_offer = (
            sub.resolve_first_presentation_offer
        )
        self.session.first_presented_player_ready = lambda: (
            self.subscription is sub
            and not sub.closed
            and sub.first_presentation_offers is not None
        )
        return sub

    async def close(self, reason=None):
        try:
            await super().close(reason)
        finally:
            if self.closed:
                self.rtc_closed.set()

    def detach(self, sub):
        sub.stop("DISCONNECTED")
        if self.subscription is not sub:
            return
        self.subscription = None
        if self.retention:
            self.retention.cancel()
        if not self.closed and self.terminal is None:
            self.rtc_reconnect_deadline = (
                asyncio.get_running_loop().time() + self.host.grace_seconds
            )

            async def expire():
                await asyncio.sleep(self.host.grace_seconds)
                # A resume accepted before expiry owns one bounded ICE attempt.
                # Keep the source alive through that attempt, not through an
                # unlimited sequence of failed reconnects.
                preparation = self.rtc_reconnect
                if preparation is not None:
                    with contextlib.suppress(TimeoutError):
                        async with asyncio.timeout_at(preparation.deadline):
                            await preparation.released.wait()
                if self.subscription is None:
                    logging.getLogger(__name__).info(
                        "RTC reconnect grace expired source_session_id=%s",
                        self.session_id,
                    )
                    await self.close("RECONNECT_EXPIRED")

            self.retention = asyncio.create_task(expire())


class RTCStartCancelled(Exception):
    pass


def validate_rtc_capture(spec, fps):
    """Validate the public capture contract independently of codec levels."""
    if type(fps) is not int or not 1 <= fps <= 60:
        raise WhiteboxDiagnostic("RTC_CODEC_UNSUPPORTED", "validation")


async def reserve_rtc(host, body):
    """Reserve only startup capacity; no world, CLI or model is started here."""
    message = body.get("message")
    user = body.get("user_id")
    if (
        not isinstance(message, dict)
        or message.get("control_protocol") != PROTOCOL
        or not isinstance(user, str)
        or not 1 <= len(user) <= 256
        or message.get("type") not in {"init", "resume", "source.bind"}
    ):
        raise RenderError("INVALID_SOURCE")
    if message["type"] == "init":
        from .cli_room import capture_fps
        from .model_geometry import session_source_spec

        world = message.get("whitebox", {})
        spec = session_source_spec(
            message.get("whitebox", {}), host.config.spec, host.runtime.rooms
        )
        room = host.runtime.rooms.get(world.get("game_id"))
        from .session_encoding import (
            check_source_encoding_scope,
            preflight_source_encoding,
        )

        try:
            selected = check_source_encoding_scope(
                message, transport="webrtc-rtp", existing_room=room is not None
            )
        except ValueError:
            raise WhiteboxDiagnostic("RTC_CODEC_UNSUPPORTED", "validation") from None
        driver = (room or {}).get("driver")
        # An existing CLI descriptor already froze the actual capture rate.
        # Re-reading today's environment must not change that Room's contract.
        fps = getattr(driver, "descriptor", {}).get("fps")
        if fps is None:
            fps = getattr(driver, "metadata", {}).get("capture_fps")
        if fps is None:
            fps = capture_fps(
                spec, os.environ, room.get("cli_fps") if room else world.get("cli_fps")
            )
        validate_rtc_capture(spec, fps)
        # Legacy default H.264 keeps its old negotiation only inside its known
        # envelope. Larger rates need the actual paired encoder SPS too.
        if selected is None and not level31_supports(spec, fps):
            from .session_encoding import default_source_encoding

            selected = default_source_encoding(host.config)
        if selected is not None:
            if (
                len(host.sessions) + len(getattr(host, "peer_sessions", {}))
                >= host.max_sessions
                or getattr(host.runtime, "rtc_preparation", None) is not None
            ):
                raise RenderError("CAPACITY_EXHAUSTED")
            # No host lock, CLI, native listener, model or source attachment yet.
            probe_lock = getattr(host, "source_encoding_probe_lock", None)
            if probe_lock is None:
                host.source_encoding_probe_lock = probe_lock = asyncio.Lock()
            if probe_lock.locked():
                raise RenderError("CAPACITY_EXHAUSTED")
            async with probe_lock:
                try:
                    from .playback import capture_rate_hint

                    nominal = (
                        capture_rate_hint(spec, fps)
                        if selected["codec"] == "h264"
                        else fps
                    )
                    probe = asyncio.create_task(
                        asyncio.to_thread(
                            preflight_source_encoding,
                            selected,
                            nominal,
                            width=spec.width,
                            height=spec.height,
                        )
                    )
                    try:
                        body["source_encoding_profile"] = await asyncio.shield(probe)
                    except asyncio.CancelledError:
                        # Cancellation must not release this concurrency bound
                        # while the bounded subprocess probe is still running.
                        await asyncio.gather(probe, return_exceptions=True)
                        raise
                except Exception:
                    raise WhiteboxDiagnostic(
                        "RTC_CODEC_UNSUPPORTED", "validation"
                    ) from None
    reservation = new_id()
    previous = None
    pipeline = None
    member_resume = False
    async with host.lock:
        if getattr(host.runtime, "rtc_preparation", None) is not None:
            raise RenderError("CAPACITY_EXHAUSTED")
        member_preparing = any(
            getattr(item, "rtc_reconnect", None) is not None
            for item in host.sessions.values()
        )
        if message["type"] == "init":
            if member_preparing:
                raise RenderError("CAPACITY_EXHAUSTED")
            if not source_create_matches(message):
                raise RenderError("INVALID_SOURCE")
            if (
                len(host.sessions) + len(getattr(host, "peer_sessions", {}))
                >= host.max_sessions
            ):
                raise RenderError("CAPACITY_EXHAUSTED")
        else:
            pipeline = host.sessions.get(
                checked_id(message.get("source_session_id", message.get("session_id")))
            )
            if pipeline is not None and hasattr(pipeline, "single_player_operation"):
                message = host.single_player_control.authorize_attachment(
                    pipeline, message, user
                )
            if (
                pipeline is not None
                and getattr(pipeline, "member_source", None) is not None
            ):
                message = pipeline.member_source.authorize_attachment(
                    pipeline, message, user
                )
            if (
                pipeline is None
                or pipeline.closed
                or getattr(pipeline, "closing", False)
                or pipeline.terminal is not None
                or pipeline.media_transport != "webrtc-rtp"
                or pipeline.user_id != user
                or pipeline.game_id != message.get("whitebox", {}).get("game_id")
                or message.get("owner_epoch") != host.runtime.epoch
                or message.get("source_epoch") != pipeline.session.source_epoch
                or not hmac.compare_digest(
                    str(message.get("resume_token", "")), pipeline.token
                )
            ):
                raise RenderError("ROOM_SESSION_LOST")
            cursor = message.get("control_sequence", 0)
            if (
                type(cursor) is not int
                or not pipeline.history_floor <= cursor <= pipeline.control_sequence
            ):
                raise RenderError("SOURCE_EPOCH_MISMATCH")
            from .session_encoding import frozen_source_encoding

            if "source_encoding" in message.get("whitebox", {}):
                frozen_source_encoding(message["whitebox"], pipeline.init["whitebox"])
            requested = message.get("whitebox", {}).get("resolution")
            if requested is not None:
                from .model_geometry import resolution_spec

                if (
                    resolution_spec(requested, pipeline.session.config.spec)
                    != pipeline.session.config.spec
                ):
                    raise WhiteboxDiagnostic("RESOLUTION_UNSUPPORTED", "validation")
            validate_rtc_capture(
                pipeline.session.config.spec,
                pipeline.session.capture_rate_hint(),
            )
            member_resume = getattr(pipeline, "member_source", None) is not None
            if getattr(pipeline, "rtc_reconnect", None) is not None or (
                member_preparing and not member_resume
            ):
                raise RenderError("CAPACITY_EXHAUSTED")
            previous = pipeline.subscription
            expires = getattr(pipeline, "rtc_reconnect_deadline", None)
            now = asyncio.get_running_loop().time()
            if previous is None and expires is not None and now >= expires:
                raise RenderError("ROOM_SESSION_LOST")
            pipeline.rtc_reconnect = ReconnectPreparation(
                reservation, now + RTC_RECONNECT_SETUP_SECONDS, asyncio.Event()
            )
            if previous is not None:
                pipeline.detach(previous)
        if getattr(host.runtime, "maintenance", None) is not None:
            await host.runtime.maintenance.cancel_for_play()
        # Existing member sources already hold their own admission and use
        # rtc_reconnect to fence duplicate negotiation for that source.
        if not member_resume:
            host.runtime.rtc_preparation = reservation
    try:
        if pipeline is not None:
            async with asyncio.timeout_at(pipeline.rtc_reconnect.deadline):
                if previous is not None:
                    await previous.wait_closed()
                if pipeline.rtc_peer is not None:
                    await pipeline.rtc_peer.close()
    except BaseException:
        release_reconnect_preparation(pipeline, reservation)
        async with host.lock:
            if getattr(host.runtime, "rtc_preparation", None) == reservation:
                host.runtime.rtc_preparation = None
        raise
    return reservation, pipeline


def install_rtc_routes(host, app, token, *, peer_factory=NativeRTC):
    @app.websocket("/internal/render-rtc")
    async def render_rtc(websocket: WebSocket):
        require_internal_token(token, websocket.headers.get("authorization", ""))
        await websocket.accept()
        peer = pipeline = sub = reservation = body = previous_pipeline = None
        trace = None
        tasks = []
        attached_sent = False
        cancelled = False
        terminal_sent = False
        failure = None
        debug = None
        prewarm_store = prewarm_observer = None
        plan_store = plan_observer = None
        first_presented_requested = False
        prewarm_available = os.environ.get("WHITEBOX_MODEL_CONTEXT_PREWARM") == "1"
        debug_available = os.environ.get("WHITEBOX_PROMPT_DEBUG") == "1"
        signals = asyncio.Queue(maxsize=64)

        def signal(message):
            if message.get("type") == "candidate":
                value = dict(type="rtc.candidate", candidate=message["candidate"])
            elif message.get("type") == "state":
                value = dict(type="rtc.state", state=message["state"])
            else:
                return
            if signals.full():
                raise RuntimeError("RTC signaling budget exceeded")
            signals.put_nowait(dict(value, peer_id=reservation))

        async def send_signals():
            while True:
                value = await signals.get()
                try:
                    await asyncio.wait_for(
                        websocket.send_bytes(msgspec.msgpack.encode(value)), 5
                    )
                finally:
                    signals.task_done()

        try:
            wire = await asyncio.wait_for(websocket.receive_bytes(), 30)
            if len(wire) > 8 * 1024**2 + 16384:
                raise RenderError("LIMIT_EXCEEDED")
            body = msgspec.msgpack.decode(wire)
            if not isinstance(body, dict):
                raise RenderError("INVALID_SOURCE")
            trace = body.get("trace_id")
            reservation, previous_pipeline = await reserve_rtc(host, body)
            from .session_encoding import session_source_config

            world = body["message"].get("whitebox", {})
            session_config = (
                getattr(previous_pipeline.session, "config", host.config)
                if previous_pipeline is not None
                else session_source_config(host.config, world)
            )
            explicit = (
                "source_encoding"
                in getattr(previous_pipeline, "init", {}).get("whitebox", {})
                if previous_pipeline is not None
                else "source_encoding" in world
            )
            codec = (getattr(session_config, "source_encoding", None) or {}).get(
                "codec", "h264"
            )
            peer_options = {}
            profile = (
                previous_pipeline.rtc_source_profile
                if previous_pipeline is not None
                else body.get("source_encoding_profile")
            )
            if explicit or profile is not None:
                peer_options["source_codec"] = codec
                if codec == "h264":
                    peer_options["source_h264_profile"] = profile
            peer = peer_factory(signal, **peer_options)
            if previous_pipeline is None and body["message"].get("startup") is not None:
                pipeline, sub = await host.attach(
                    body, transport="webrtc-rtp", rtc_peer=peer, preparation=reservation
                )
                # The early subscription takes the native callback, but its
                # media/control writer only runs after ICE connects. Keep the
                # negotiation callback until that writer can own the peer.
                peer.on_event = signal
            tasks.append(asyncio.create_task(send_signals()))
            deadline = (
                45
                if previous_pipeline is None
                else max(
                    0,
                    previous_pipeline.rtc_reconnect.deadline
                    - asyncio.get_running_loop().time(),
                )
            )
            async with asyncio.timeout(deadline):
                await peer.start()
                await signals.put(
                    dict(
                        type="rtc.configuration",
                        peer_id=reservation,
                        protocol=(
                            "native-h264-rtp-v1"
                            if codec == "h264"
                            else "native-video-rtp-v1"
                        ),
                        video_codecs={
                            "whitebox": codec,
                            "model": "h264",
                        },
                        delivery_stats_version=1,
                        ice_servers=[],
                        tracks=["whitebox", "model"],
                        negotiation_timeout_ms=int(deadline * 1000),
                    )
                )
                connected = asyncio.create_task(peer.connected.wait())
                try:
                    while not connected.done():
                        receive = asyncio.create_task(websocket.receive_bytes())
                        try:
                            done, _ = await asyncio.wait(
                                (connected, receive),
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if receive not in done:
                                continue
                            wire = receive.result()
                            if len(wire) > 256 << 10:
                                raise RenderError("LIMIT_EXCEEDED")
                            message = msgspec.msgpack.decode(wire)
                            if message.get("type") == "rtc.cancel" and message.get(
                                "peer_id"
                            ) in {None, reservation}:
                                cancelled = True
                                raise RTCStartCancelled()
                            if (
                                message.get("type", "").startswith("startup.")
                                and pipeline is not None
                            ):
                                pipeline.accept(sub, wire)
                                continue
                            if message.get("peer_id") != reservation:
                                raise RenderError("SOURCE_EPOCH_MISMATCH")
                            if message.get("type") == "rtc.offer":
                                first_presented_requested = (
                                    message.get("model_context_first_presented")
                                    == FIRST_PRESENTED_PROTOCOL
                                )
                                # Older cached players reject unknown rtc.* events.
                                # Statistics are opt-in, independent of media support.
                                peer.delivery_stats_version = (
                                    1
                                    if type(message.get("delivery_stats_version"))
                                    is int
                                    and message["delivery_stats_version"] == 1
                                    else 0
                                )
                                result = await peer.offer(message.get("sdp"))
                                await signals.put(
                                    dict(
                                        type="rtc.answer",
                                        sdp=result["sdp"],
                                        tracks=result["tracks"],
                                        peer_id=reservation,
                                    )
                                )
                            elif message.get("type") == "rtc.candidate":
                                await peer.candidate(message.get("candidate"))
                            else:
                                raise RenderError("INVALID_COMMAND")
                        finally:
                            receive.cancel()
                            await asyncio.gather(receive, return_exceptions=True)
                finally:
                    connected.cancel()
                    await asyncio.gather(connected, return_exceptions=True)
                peer._check()
                await signals.join()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            tasks.clear()
            if pipeline is None:
                pipeline, sub = await host.attach(
                    body, transport="webrtc-rtp", rtc_peer=peer, preparation=reservation
                )
            else:
                peer.on_event = sub.native_event
            first_presented_available = (
                first_presented_requested
                and isinstance(pipeline, RTCRenderPipeline)
                and getattr(pipeline, "member_source", None) is None
            )
            if first_presented_available:
                sub.enable_first_presentation()
            from .startup import model_transport_ready

            model_transport_ready(pipeline.session)
            async with host.lock:
                release_reconnect_preparation(previous_pipeline, reservation)
                if getattr(host.runtime, "rtc_preparation", None) == reservation:
                    host.runtime.rtc_preparation = None

            def get_prewarm_store(message):
                session = pipeline.session
                if (
                    not prewarm_available
                    or not hasattr(pipeline, "single_player_operation")
                    or getattr(pipeline, "member_source", None) is not None
                    or getattr(getattr(session, "engine", None), "shared_world", False)
                    or not getattr(getattr(session, "inference", None), "manual", False)
                    or sub is not pipeline.subscription
                    or sub.closed
                    or message.get("connection_generation") != sub.generation
                ):
                    return None
                channel = getattr(session, "context_channel", None)
                consumer_id = getattr(getattr(channel, "view", None), "client_id", None)
                source = session.source_identity()
                scope = {
                    key: source[key] for key in ("source_session_id", "source_epoch")
                }
                if not consumer_id or any(
                    message.get(key) != value for key, value in scope.items()
                ):
                    return None
                scope["consumer_id"] = consumer_id
                from .prewarm_observation import PrewarmPlanObservations

                protocols = (
                    (
                        "prewarm_plan_observations",
                        PrewarmPlanObservations,
                        "prewarm_plan_supported",
                    ),
                    (
                        "prewarm_observations",
                        PrewarmPlayerObservations,
                        "prewarm_supported",
                    ),
                )
                # A retained store pins the source's negotiated protocol through
                # inference-off and reconnect; an empty store cannot prove support.
                for attribute, _, _ in protocols:
                    store = getattr(session, attribute, None)
                    if store is not None and store.source == scope:
                        return store
                for attribute, observation, supported in protocols:
                    if getattr(channel, supported, False):
                        try:
                            store = observation(**scope)
                        except ValueError:
                            return None
                        setattr(session, attribute, store)
                        return store
                return None

            def get_member_plan_store(message):
                from .prewarm_observation import (
                    PrewarmPlanObservations,
                    dynamic_member_plan_scope,
                )

                if (
                    not prewarm_available
                    or sub is not pipeline.subscription
                    or sub.closed
                    or message.get("connection_generation") != sub.generation
                ):
                    return None
                session = pipeline.session
                scope = dynamic_member_plan_scope(
                    session, getattr(pipeline, "member_source", None), message
                )
                if scope is None:
                    return None
                store = getattr(session, "prewarm_plan_observations", None)
                if store is None or store.source != scope:
                    store = session.prewarm_plan_observations = PrewarmPlanObservations(
                        **scope
                    )
                return store

            def observe_plan():
                session = pipeline.session

                def observe(event):
                    if (
                        event is not None
                        and session is pipeline.session
                        and sub is pipeline.subscription
                        and not sub.closed
                        and getattr(session, "prewarm_plan_observations", None)
                        is plan_store
                        and all(event.get(k) == v for k, v in plan_store.source.items())
                    ):
                        sub.offer_control(
                            plan_store.wire(event, sub.generation), optional=True
                        )

                session.prewarm_plan_observer = observe
                for event in plan_store.replay(sub.generation):
                    sub.offer_control(event, optional=True)
                return observe

            def observe_prewarm():
                session = pipeline.session

                def observe(event):
                    if (
                        event is not None
                        and session is pipeline.session
                        and sub is pipeline.subscription
                        and not sub.closed
                        and getattr(session, "prewarm_observations", None)
                        is prewarm_store
                        and all(
                            event.get(key) == value
                            for key, value in prewarm_store.source.items()
                        )
                    ):
                        sub.offer_control(
                            prewarm_store.wire(event, sub.generation), optional=True
                        )

                session.prewarm_observer = observe
                for event in prewarm_store.replay(sub.generation):
                    sub.offer_control(event, optional=True)
                return observe

            async def send_debug(queue):
                while True:
                    value = await queue.get()
                    await asyncio.wait_for(
                        websocket.send_bytes(msgspec.msgpack.encode(value)), 5
                    )

            def open_debug():
                # Debug payloads belong to one authenticated subscription and
                # never enter the reliable replay ring or the recording.
                queue = asyncio.Queue(maxsize=8)
                latest = {}

                def observe(condition, receipt):
                    if sub is not pipeline.subscription or sub.closed:
                        return
                    latest["model"] = condition
                    if queue.full():
                        queue.get_nowait()
                    queue.put_nowait(
                        dict(
                            type="model.context.debug.condition",
                            condition_id=condition["conditionId"],
                            payload_sha256=condition["payloadHash"],
                            prompt=condition["prompt"],
                            references=[
                                {
                                    key: ref[key]
                                    for key in ("picture", "format", "sha256")
                                }
                                for ref in condition["references"]
                            ],
                            receipt={
                                key: receipt[key]
                                for key in (
                                    "phase",
                                    "requestId",
                                    "promptVersion",
                                    "reason",
                                )
                                if receipt.get(key) is not None
                            },
                        )
                    )

                async def poll_source():
                    previous = None
                    while sub is pipeline.subscription and not sub.closed:
                        reader = getattr(
                            pipeline.session.engine, "refresh_source_debug", None
                        )
                        if reader is not None:
                            try:
                                conditions = await asyncio.wait_for(reader(), 1) or []
                                if conditions != previous:
                                    for key in list(latest):
                                        if key.startswith("source:"):
                                            del latest[key]
                                    rows = []
                                    for condition in conditions:
                                        latest[condition["conditionId"]] = condition
                                        rows.append(
                                            dict(
                                                condition_id=condition["conditionId"],
                                                payload_sha256=condition["payloadHash"],
                                                prompt=condition["prompt"],
                                                event_id=condition["eventId"],
                                                occurrence_id=condition.get(
                                                    "occurrenceId"
                                                ),
                                                simulation_tick=condition[
                                                    "simulationTick"
                                                ],
                                                references=[
                                                    {
                                                        key: ref[key]
                                                        for key in (
                                                            "picture",
                                                            "format",
                                                            "sha256",
                                                        )
                                                    }
                                                    for ref in condition["references"]
                                                ],
                                                receipt=condition["receipt"],
                                            )
                                        )
                                    if queue.full():
                                        queue.get_nowait()
                                    queue.put_nowait(
                                        dict(
                                            type="model.context.debug.source.history",
                                            rows=rows,
                                        )
                                    )
                                    previous = conditions
                            except Exception:
                                if previous:
                                    for key in list(latest):
                                        if key.startswith("source:"):
                                            del latest[key]
                                    if queue.full():
                                        queue.get_nowait()
                                    queue.put_nowait(
                                        dict(
                                            type="model.context.debug.source.history",
                                            rows=[],
                                        )
                                    )
                                    previous = None
                        await asyncio.sleep(0.5)

                session = pipeline.session
                session.condition_debug = observe
                if session.source_conditions is not None:
                    session.source_conditions.debug = observe
                session.emit_applied_prompt_debug()
                sender = asyncio.create_task(send_debug(queue))
                return (
                    observe,
                    latest,
                    queue,
                    sender,
                    asyncio.create_task(poll_source()),
                )

            async def stop_debug(subscription):
                observe, _, _, sender, poller = subscription
                session = pipeline.session
                if session.condition_debug is observe:
                    session.condition_debug = None
                if (
                    session.source_conditions is not None
                    and session.source_conditions.debug is observe
                ):
                    session.source_conditions.debug = None
                sender.cancel()
                poller.cancel()
                await asyncio.gather(sender, poller, return_exceptions=True)

            async def send_debug_reference(message):
                _, conditions, queue, _, _ = debug
                condition_id, digest = (
                    message.get("condition_id"),
                    message.get("sha256"),
                )
                latest = next(
                    (
                        row
                        for row in conditions.values()
                        if row and row["conditionId"] == condition_id
                    ),
                    {},
                )
                ref = next(
                    (
                        ref
                        for ref in latest.get("references", ())
                        if ref["sha256"] == digest
                    ),
                    None,
                )
                reply = dict(
                    type="model.context.debug.reference",
                    condition_id=condition_id,
                    sha256=digest,
                )
                if ref is not None and latest.get("conditionId") == condition_id:
                    try:
                        async with asyncio.timeout(3):
                            if "asset" in ref:
                                resolved = await pipeline.session.assets.resolve(
                                    [ref["asset"]]
                                )
                                data = resolved[0]["payload"]
                            else:
                                data = await pipeline.session.engine.read_model_context_resource(
                                    ref["resource"]
                                )
                        if (
                            sub is pipeline.subscription
                            and not sub.closed
                            and any(row is latest for row in conditions.values())
                            and isinstance(data, bytes)
                            and len(data) <= 8 * 1024**2
                            and hashlib.sha256(data).hexdigest() == digest
                        ):
                            reply.update(format=ref["format"], bytes=data)
                    except Exception:
                        pass
                if sub is pipeline.subscription and not sub.closed and not queue.full():
                    queue.put_nowait(reply)

            async def receive():
                nonlocal cancelled, debug, prewarm_observer, plan_observer
                while True:
                    wire = await websocket.receive_bytes()
                    if len(wire) > 8 * 1024**2 + 16384:
                        raise RenderError("LIMIT_EXCEEDED")
                    message = msgspec.msgpack.decode(wire)
                    if message.get("type") == "rtc.cancel":
                        if sub is not pipeline.subscription or message.get(
                            "peer_id"
                        ) not in {None, reservation}:
                            raise RenderError("SOURCE_EPOCH_MISMATCH")
                        cancelled = True
                        return
                    elif message.get("type") == "rtc.candidate":
                        if message.get("peer_id") != reservation:
                            raise RenderError("SOURCE_EPOCH_MISMATCH")
                        await peer.candidate(message.get("candidate"))
                    elif message.get("type", "").startswith("rtc."):
                        raise RenderError("INVALID_COMMAND")
                    elif message.get("type") == "model.context.prewarm.plan.observe":
                        from .prewarm_observation import is_plan_observation_request

                        if (
                            plan_store is not None
                            and plan_observer is None
                            and sub is pipeline.subscription
                            and not sub.closed
                            and is_plan_observation_request(message)
                        ):
                            plan_observer = observe_plan()
                    elif message.get("type") == OBSERVE:
                        if (
                            prewarm_store is not None
                            and prewarm_observer is None
                            and sub is pipeline.subscription
                            and not sub.closed
                            and is_prewarm_observation_request(message)
                        ):
                            prewarm_observer = observe_prewarm()
                    elif message.get("type") == "model.context.debug.subscribe":
                        if (
                            set(message) == {"type"}
                            and debug is None
                            and debug_available
                            and sub is pipeline.subscription
                            and not sub.closed
                        ):
                            debug = open_debug()
                    elif message.get("type") == "model.context.debug.unsubscribe":
                        if set(message) == {"type"} and debug is not None:
                            await stop_debug(debug)
                            debug = None
                    elif message.get("type") == "model.context.debug.reference.get":
                        if (
                            debug is not None
                            and set(message) == {"type", "condition_id", "sha256"}
                            and isinstance(message["condition_id"], str)
                            and isinstance(message["sha256"], str)
                            and sub is pipeline.subscription
                            and not sub.closed
                        ):
                            await send_debug_reference(message)
                    else:
                        pipeline.accept(sub, wire)

            async def send():
                nonlocal attached_sent, terminal_sent, prewarm_store, plan_store
                while True:
                    wire = await sub.recv()
                    message = msgspec.msgpack.decode(wire)
                    kind = message.get("type")
                    if kind in {"session.attached", "session.ready", "source.status"}:
                        extras = {}
                        if first_presented_available and kind != "source.status":
                            extras["model_context_first_presented"] = (
                                FIRST_PRESENTED_PROTOCOL
                            )
                        if debug_available and kind == "session.attached":
                            extras.update(
                                prompt_debug_available=True, prompt_debug_toggle=True
                            )
                        if prewarm_store is None and plan_store is None:
                            store = get_prewarm_store(message)
                            if store is not None:
                                extras.update(store.capability)
                                if isinstance(store, PrewarmPlayerObservations):
                                    prewarm_store = store
                                else:
                                    plan_store = store
                        if plan_store is None:
                            plan_store = get_member_plan_store(message)
                            if plan_store is not None:
                                extras.update(plan_store.capability)
                        if extras:
                            wire = msgspec.msgpack.encode(dict(message, **extras))
                    await asyncio.wait_for(websocket.send_bytes(wire), 5)
                    if kind == "session.attached":
                        attached_sent = True
                    if kind in {"error", "session.completed"}:
                        terminal_sent = kind == "session.completed"
                        return

            tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except RTCStartCancelled:
            pass
        except Exception as error:
            logging.getLogger(__name__).warning(
                "RTC attachment failed trace_id=%s", trace, exc_info=True
            )
            if not isinstance(error, (RenderError, WhiteboxDiagnostic)):
                error = WhiteboxDiagnostic(
                    "RTC_DISCONNECTED" if attached_sent else "RTC_CONNECTION_FAILED",
                    "transport",
                )
            failure = dict(
                type="error",
                **public_error(error, trace, stage="transport"),
                reconnectable=pipeline is not None and attached_sent,
            )
        finally:
            if (
                prewarm_observer is not None
                and getattr(pipeline.session, "prewarm_observer", None)
                is prewarm_observer
            ):
                pipeline.session.prewarm_observer = None
            if (
                plan_observer is not None
                and getattr(pipeline.session, "prewarm_plan_observer", None)
                is plan_observer
            ):
                pipeline.session.prewarm_plan_observer = None
            if debug is not None:
                await stop_debug(debug)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            cleanup_failed = False
            try:
                if pipeline and sub:
                    if cancelled or (
                        not attached_sent and body["message"]["type"] == "init"
                    ):
                        await pipeline.close("STARTUP_CANCELLED")
                        await asyncio.wait_for(pipeline.rtc_closed.wait(), 15)
                    elif terminal_sent:
                        await asyncio.wait_for(pipeline.rtc_closed.wait(), 15)
                    else:
                        pipeline.detach(sub)
                if cancelled and pipeline is None and previous_pipeline is not None:
                    await previous_pipeline.close("CANCELLED")
                    await asyncio.wait_for(previous_pipeline.rtc_closed.wait(), 15)
            except Exception:
                cleanup_failed = True
                logging.getLogger(__name__).exception(
                    "RTC source cleanup was not confirmed"
                )
            finally:
                if sub:
                    sub.stop("DISCONNECTED")
                    with contextlib.suppress(Exception):
                        await sub.wait_closed()
                if peer:
                    try:
                        await peer.close()
                    except Exception:
                        cleanup_failed = True
            async with host.lock:
                release_reconnect_preparation(previous_pipeline, reservation)
                if (
                    reservation is not None
                    and getattr(host.runtime, "rtc_preparation", None) == reservation
                ):
                    host.runtime.rtc_preparation = None
            if (
                pipeline is None
                and reservation is not None
                and body
                and body.get("message", {}).get("type") == "init"
            ):
                with contextlib.suppress(Exception):
                    await host.runtime.directory.touch(
                        body["message"]["whitebox"]["game_id"],
                        host.runtime.owner_id,
                        host.runtime.epoch,
                        end=True,
                    )
            if cleanup_failed:
                failure = dict(
                    type="error",
                    code="RTC_CLEANUP_PENDING",
                    stage="transport",
                    content="The previous session is still being cleaned up. Please wait before starting again.",
                    reconnectable=False,
                )
            if cancelled and not cleanup_failed:
                with contextlib.suppress(Exception):
                    await websocket.send_bytes(
                        msgspec.msgpack.encode(
                            dict(type="rtc.cancelled", peer_id=reservation)
                        )
                    )
            elif failure is not None:
                with contextlib.suppress(Exception):
                    await websocket.send_bytes(msgspec.msgpack.encode(failure))
            with contextlib.suppress(Exception):
                await websocket.close()
