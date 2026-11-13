# SPDX-License-Identifier: Apache-2.0
"""Room-owned continuous H3 sessions and replaceable encoded-media subscribers.

No Gateway identity owns a world or model lease. Each attachment is fenced by a
Room-issued generation; losing a subscriber starts a bounded retention period.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import logging
import secrets
import time
from collections import OrderedDict, deque
from dataclasses import asdict, replace

import msgspec
from fastapi import WebSocket

from .contracts import EngineFrame
from .diagnostics import WhiteboxDiagnostic
from .interactive import InteractiveSession
from .model_geometry import session_source_spec
from .reports import SessionReport
from .room_client import RemoteRenderer
from .rooms import checked_id, new_id, require_internal_token


class LocalRoomEngine:
    def __init__(self, runtime, identity):
        self.runtime, self.identity = runtime, identity
        self.stream_id = identity["stream_id"]
        self.metadata = {**identity["engine"], "transport": "room_local"}
        self.renderer = RemoteRenderer(self)
        self.closed = False

    async def rpc(self, operation, body=None):
        if operation == "render_finish":
            return
        raise ValueError("Room pipelines require browser-owned rendering")

    async def frames(self):
        while not self.closed:
            frame = await self.runtime.next_frame(self.stream_id)
            if frame is None:
                return
            yield EngineFrame(**frame)

    async def capture_control(self, action, request_id):
        return await self.runtime.capture_control(self.stream_id, action, request_id)

    @property
    def shared_world(self):
        entry = self.runtime.player(self.stream_id, touch=False)
        room = self.runtime.rooms[entry["game_id"]]
        return getattr(room["driver"], "max_players", 1) > 1 or "control_caller" in room

    async def area_status(self):
        return await self.runtime.area_status(self.stream_id)

    async def area_frame(self, header):
        return await self.runtime.area_frame(self.stream_id, header)

    async def area_presented(self, body, *, before_send=None):
        return await self.runtime.area_presented(
            self.stream_id, body, before_send=before_send
        )

    async def send_input(self, sequence, payload):
        await self.note("input", {"sequence": sequence, "input": payload})

    async def note(self, kind, payload):
        return await self.runtime.event(self.stream_id, kind, payload)

    async def touch(self):
        self.runtime.player(self.stream_id)

    async def finish(self):
        await self.note("finish", {})

    async def player_input_session(self):
        return await self.runtime.player_input_session(self.stream_id)

    async def ui_asset(self, path):
        return await self.runtime.ui_asset(self.stream_id, path)

    @property
    def model_context_capability(self):
        engine = self.runtime.player(self.stream_id)["engine"]
        return getattr(engine, "descriptor", {}).get("modelContext")

    async def bind_model_context(
        self, *, segment_id, create_adapter, start_suspended=False
    ):
        engine = self.runtime.player(self.stream_id)["engine"]
        binder = getattr(engine, "bind_model_context", None)
        if binder is None:
            return None
        self._context_engine = engine

        def create(envelope, emit, check):
            def fenced():
                check()
                if (
                    self.closed
                    or self.runtime.player(self.stream_id)["engine"] is not engine
                ):
                    raise ValueError("MODEL_CONTEXT_STALE_SOURCE")

            return create_adapter(envelope, emit, fenced)

        return await binder(
            segment_id=segment_id,
            create_adapter=create,
            **({"start_suspended": True} if start_suspended else {}),
        )

    def check_model_context(self):
        if self.closed or self.runtime.player(self.stream_id)["engine"] is not getattr(
            self, "_context_engine", None
        ):
            from .render_contract import RenderError

            raise RenderError("SOURCE_EPOCH_MISMATCH")

    async def refresh_source_debug(self):
        engine = getattr(self, "_context_engine", None)
        reader = getattr(engine, "refresh_source_debug", None)
        if reader is None:
            return None
        self.check_model_context()
        value = await reader()
        self.check_model_context()
        return value

    async def read_model_context_resource(self, handle):
        engine = self.runtime.player(self.stream_id)["engine"]
        if engine is not getattr(self, "_context_engine", None):
            raise ValueError("MODEL_CONTEXT_STALE_SOURCE")
        return await engine.read_model_context_resource(handle)

    async def client_messages(self):
        engine = self.runtime.player(self.stream_id)["engine"]
        if hasattr(engine, "client_messages"):
            async for message in engine.client_messages():
                yield message

    async def close(self):
        if not self.closed:
            await self.runtime.leave(self.stream_id)
            self.closed = True

    async def retire(self):
        receipt = await self.runtime.retire_member(self.stream_id)
        self.closed = True
        return receipt


class LocalRoomFactory:
    def __init__(self, runtime, game_id, player_id, stream_id):
        self.runtime, self.game_id, self.player_id, self.stream_id = (
            runtime,
            game_id,
            player_id,
            stream_id,
        )
        self.startup_started = False
        self.startup_owner_epoch = runtime.epoch

    async def open_session(self, *, world_id, spec, trace_id, context):
        from .room_startup import StartupScope

        self.startup_started = True
        self.startup_scope = StartupScope()
        self.startup_cleanup = None
        try:
            return await self._open_session(
                world_id=world_id, spec=spec, trace_id=trace_id, context=context
            )
        except BaseException as error:
            from .room_startup import (
                FactoryCleanupReceipt,
                RoomCleanupReceipt,
                startup_cleanup_receipt,
            )

            receipt = startup_cleanup_receipt(error)
            if (
                isinstance(receipt, RoomCleanupReceipt)
                and receipt.factory.operation_id == trace_id
                and receipt.owner_epoch == self.runtime.epoch
                and receipt.owner_epoch == self.startup_owner_epoch
            ):
                self.startup_cleanup = receipt
            elif (
                self.startup_scope.entered
                and not self.startup_scope.allocation_started
                and self.runtime.epoch == self.startup_owner_epoch
            ):
                # The original runtime observed this join but never invoked a
                # Room or driver allocation. Absence alone is not proof.
                self.startup_cleanup = RoomCleanupReceipt(
                    FactoryCleanupReceipt(trace_id), self.startup_owner_epoch
                )
            raise

    async def retire_failed_start(self):
        receipt = getattr(self, "startup_cleanup", None)
        if (
            receipt is not None
            and receipt.owner_epoch == self.runtime.epoch
            or not self.startup_started
            and self.startup_owner_epoch == self.runtime.epoch
        ):
            return dict(
                state="not_created",
                source_id=self.stream_id,
                owner_epoch=self.startup_owner_epoch,
            )
        # A lost join reply may still have created this exact member. Only its
        # owner can acknowledge retirement; an absent entry remains unknown.
        return await self.runtime.retire_member(self.stream_id)

    async def _open_session(self, *, world_id, spec, trace_id, context):
        identity = await self.runtime.join(
            dict(
                game_id=self.game_id,
                player_id=self.player_id,
                stream_id=self.stream_id,
                world_id=world_id,
                spec=asdict(spec),
                trace_id=trace_id,
                session_id=context["session_id"],
                user_id=context["user_id"],
                **({"start_suspended": True} if context.get("start_suspended") else {}),
                **(
                    {
                        "expected_publication": copy.deepcopy(
                            context["expected_publication"]
                        )
                    }
                    if context.get("expected_publication") is not None
                    else {}
                ),
                **(
                    {"source_options": copy.deepcopy(context["source_options"])}
                    if context.get("source_options")
                    else {}
                ),
                **(
                    {"cli_fps": context["cli_fps"]}
                    if context.get("cli_fps") is not None
                    else {}
                ),
            ),
            startup_scope=self.startup_scope,
        )
        return LocalRoomEngine(self.runtime, identity)


class InputReceiptWire:
    """Owner-scoped deltas; recordings and clients without negotiation stay intact."""

    protocol = "state-delta-v1"

    def __init__(self, init):
        self.enabled = init.get("input_receipt_protocol") == self.protocol
        self.history = OrderedDict()

    def control(self, message):
        if not self.enabled:
            return message
        if message.get("type") == "session.ready":
            return dict(message, input_receipt_protocol=self.protocol)
        if message.get("type") == "whitebox.states":
            return self.compact(message, remember=True)
        return message

    def media(self, message):
        return self.compact(message) if self.enabled else message

    def compact(self, message, *, remember=False):
        """Remove only evidence already retained in the reliable control history.

        Media with unseen evidence retains it as a compatibility fallback. It
        must not advance the index: media can be dropped while control is replayed.
        Never edit the producer envelope used by recordings or replay history.
        """
        observed = message.get("observed_input")
        if not isinstance(observed, dict) or not isinstance(
            observed.get("receipts"), list
        ):
            return message
        receipts = []
        for receipt in observed["receipts"]:
            key = None
            if isinstance(receipt, dict) and receipt.get("type") == "input.receipt":
                epoch = receipt.get("epoch")
                sequence = receipt.get("clientSequence", receipt.get("sequence"))
                if (
                    type(epoch) is int
                    and epoch >= 0
                    and type(sequence) is int
                    and sequence >= 0
                ):
                    key = epoch, sequence
            elif isinstance(receipt, dict) and receipt.get("type") == "action.receipt":
                epoch, action_id = receipt.get("epoch"), receipt.get("actionId")
                if (
                    type(epoch) is int
                    and epoch >= 0
                    and isinstance(action_id, str)
                    and 0 < len(action_id) <= 128
                ):
                    key = "action", epoch, action_id
            if key is None or self.history.get(key) != receipt:
                receipts.append(receipt)
                if remember and key is not None:
                    self.history.pop(key, None)
                    self.history[key] = copy.deepcopy(receipt)
                    while len(self.history) > 256:
                        self.history.popitem(last=False)
        return dict(message, observed_input=dict(observed, receipts=receipts))

    def repeated_observation(self, previous, current, earlier=None):
        """Only replay storage may replace adjacent captures with no new intent.

        Live controls and the already written source recording remain complete.
        A malformed or unfamiliar envelope is retained for its original consumer
        to validate; it must never disappear as an apparently redundant clock.
        """
        if not self.enabled:
            return False
        before, after = _ui_observation(previous), _ui_observation(current)
        if before is None or after is None or before[0] != after[0]:
            return False

        def backwards(older, newer):
            return any(now < old for old, now in zip(older[1], newer[1])) or (
                older[2] is not None
                and newer[2] is not None
                and any(first < last for first, last in zip(newer[2], older[1][-4:]))
            )

        prior = _ui_observation(earlier)
        if prior is not None and prior[0] == before[0] and backwards(prior, before):
            # Preserve a backward observation even after a later clock recovers.
            return False
        return not backwards(before, after) and after[1][0] > before[1][0]


def _ui_observation(message):
    """Recognize only the existing native capture/clock DTO, conservatively."""
    from .native_pixels import read_frame_identity, safe_int

    scope_fields = {"source_session_id", "source_id", "source_epoch", "view_id"}
    if (
        not isinstance(message, dict)
        or message.get("type") != "whitebox.states"
        or message.get("version") != 2
        or set(message)
        - {
            "version",
            "type",
            "source_frame_index",
            "pts_us",
            "source",
            "states",
            "observed_input",
            "control_sequence",
            *scope_fields,
        }
    ):
        return None
    states, observed, source = (
        message.get(key) for key in ("states", "observed_input", "source")
    )
    # This classifier authorizes sparse replay compaction, not live state
    # admission. Audio requires exact historical proof for every source frame,
    # so audio-bearing states must retain their complete original envelopes.
    if isinstance(states, dict) and "audio_snapshot" in states:
        return None
    if (
        not isinstance(states, dict)
        or set(states) != {"ui_events"}
        or not isinstance(states["ui_events"], list)
        or len(states["ui_events"]) > 512
        or not isinstance(observed, dict)
        or set(observed) - {"receipts", "events", "latest", "mapping", "display"}
        or observed.get("receipts") != []
        or observed.get("events") != []
        or not isinstance(source, dict)
        or set(source)
        - {
            "cli_frame_header",
            "capture_pts_us",
            "capture_origin_us",
            "model_context_sample",
            *scope_fields,
        }
    ):
        return None
    latest, display = observed.get("latest"), observed.get("display")
    if (
        observed.get("mapping") != "receipt_evidence_only"
        or not isinstance(latest, dict)
        or set(latest) - {"held_keys", "held_buttons", "actions", "events"}
        or any(not isinstance(value, list) for value in latest.values())
        or latest.get("events", []) != []
        or not isinstance(display, dict)
        or set(display) != {"mode", "revision"}
        or not isinstance(display["mode"], str)
        or display["mode"] not in {"whitebox", "model"}
        or not safe_int(display["revision"])
    ):
        return None
    if any(
        not isinstance(key, str) or not 0 < len(key) <= 128
        for key in latest.get("held_keys", [])
    ) or any(
        not safe_int(button) or button > 31 for button in latest.get("held_buttons", [])
    ):
        return None
    for action in latest.get("actions", []):
        if (
            not isinstance(action, dict)
            or set(action) - {"id", "name", "params"}
            or not isinstance(action.get("name"), str)
            or not action["name"]
            or "id" in action
            and not isinstance(action["id"], str)
            or "params" in action
            and not isinstance(action["params"], dict)
        ):
            return None
    if any(
        not isinstance(message.get(key), str)
        or not message[key]
        or source.get(key) != message[key]
        for key in scope_fields
    ):
        return None
    try:
        header = read_frame_identity(source.get("cli_frame_header"))
        clocks = []
        for event in states["ui_events"]:
            if (
                not isinstance(event, dict)
                or set(event) != {"type", "epoch", "clock"}
                or event["type"] != "ui.clock"
                or not safe_int(event["epoch"])
                or event["epoch"] != header["epoch"]
            ):
                return None
            clock = event["clock"]
            if (
                not isinstance(clock, dict)
                or set(clock)
                != {"sourceTimeUs", "simulationTick", "uiRevision", "completeThroughUs"}
                or any(not safe_int(value) for value in clock.values())
                or clock["uiRevision"] != header["uiRevision"]
                or clock["completeThroughUs"] < clock["sourceTimeUs"]
            ):
                return None
            if clocks and any(clock[key] < clocks[-1][key] for key in clock):
                return None
            clocks.append(clock)
        times = [
            message.get("source_frame_index"),
            message.get("pts_us"),
            source.get("capture_pts_us"),
            source.get("capture_origin_us"),
        ]
        if any(not safe_int(value) for value in times):
            return None
        model = source.get("model_context_sample")
        model_scope = None
        if model is not None:
            if (
                not isinstance(model, dict)
                or set(model)
                != {
                    "contextStateId",
                    "contextRevision",
                    "viewRevision",
                    "throughControlSequence",
                    "compiledConditionId",
                }
                or not isinstance(model["contextStateId"], str)
                or not model["contextStateId"]
                or any(
                    not safe_int(model[key])
                    for key in (
                        "contextRevision",
                        "viewRevision",
                        "throughControlSequence",
                    )
                )
                or model["compiledConditionId"] is not None
                and not isinstance(model["compiledConditionId"], str)
            ):
                return None
            # Capture-state IDs/cursors vary on every tick. The compiled model
            # condition and view are the retained business boundary, not that clock.
            model_scope = (model["viewRevision"], model["compiledConditionId"])
        native = header["source"]
        progress = (
            times[0],
            times[1],
            times[2],
            header["outputFrameId"],
            header["outputPtsUs"],
            native["sourceFrameId"],
            native["sourceTimeUs"],
            native["simulationTick"],
            header["uiCompleteThroughUs"],
        )
        if model is not None:
            progress += (model["contextRevision"], model["throughControlSequence"])
        if clocks:
            progress += tuple(
                clocks[-1][key]
                for key in (
                    "sourceTimeUs",
                    "simulationTick",
                    "uiRevision",
                    "completeThroughUs",
                )
            )
        identity = {
            key: value
            for key, value in header.items()
            if key
            not in {"source", "outputFrameId", "outputPtsUs", "uiCompleteThroughUs"}
        }
        identity["source"] = {
            key: value
            for key, value in native.items()
            if key not in {"sourceFrameId", "sourceTimeUs", "simulationTick"}
        }
        key = json.dumps(
            (
                identity,
                {key: message[key] for key in sorted(scope_fields)},
                times[3],
                observed,
                model_scope,
                bool(clocks),
            ),
            sort_keys=True,
            allow_nan=False,
        )
        first_clock = (
            tuple(
                clocks[0][name]
                for name in (
                    "sourceTimeUs",
                    "simulationTick",
                    "uiRevision",
                    "completeThroughUs",
                )
            )
            if clocks
            else None
        )
        return key, progress, first_clock
    except (ValueError, TypeError, KeyError, OverflowError):
        return None


class Subscription:
    """Bounded dependency-aware media queue; reliable control is separately bound."""

    def __init__(
        self,
        owner,
        generation,
        *,
        media_bytes=8 * 1024**2,
        control_bytes=16 * 1024**2,
        ack_timeout=15.0,
    ):
        if ack_timeout <= 0:
            raise ValueError("Playback receipt timeout must be positive")
        self.owner, self.generation = owner, generation
        self.byte_limit, self.control_limit = media_bytes, control_bytes
        # Allow healthy clip/ACK bursts, but never replay several seconds of old
        # live video after a slow connection recovers. Archives are independent.
        self.frame_limit = 120
        self.max_media_age = 1.0
        self.control = deque()
        self.control_bytes = 0
        self.media = {name: deque() for name in ("whitebox", "model")}
        self.media_bytes = 0
        self.sent = {name: -1 for name in self.media}
        self.received = dict(self.sent)
        # A frame window alone can put megabytes ahead of reliable state/pongs
        # in the same TCP connection. Bound encoded bytes independently per
        # stream; ACKs release them cumulatively without changing the protocol.
        self.inflight_byte_limit = 256 * 1024
        self.shared_inflight_byte_limit = 512 * 1024
        # A large IDR can exceed the normal window, but only one bounded AU
        # may occupy the whole attachment at a time.
        self.max_access_unit_bytes = 1024 * 1024
        self.inflight = {name: deque() for name in self.media}
        self.inflight_bytes = {name: 0 for name in self.media}
        self.large_au_stream = None
        self.windows = {name: None for name in self.media}
        self.needs_key = {name: True for name in self.media}
        self.wake = asyncio.Event()
        self.closed = False
        self.reason = "DISCONNECTED"
        self.next_stream = "whitebox"
        self.ack_timeout = ack_timeout
        self.ready_for_acks = False
        self.last_ack = {name: time.monotonic() for name in self.media}
        report = getattr(owner, "report", None)
        self.delivery = report.playback_delivery if report is not None else {}
        self.delivery.setdefault("scope", "room_subscriptions")
        self.delivery["limits"] = dict(
            frames_per_stream=self.frame_limit,
            max_queue_age_ms=self.max_media_age * 1000,
            shared_media_bytes=self.byte_limit,
            inflight_bytes_per_stream=self.inflight_byte_limit,
            shared_inflight_bytes=self.shared_inflight_byte_limit,
            max_access_unit_bytes=self.max_access_unit_bytes,
        )
        self.delivery["attachments"] = self.delivery.get("attachments", 0) + 1
        self.delivery.setdefault("high_water_bytes", 0)
        streams = self.delivery.setdefault("streams", {})
        for name in self.media:
            streams.setdefault(
                name,
                dict(
                    offered_frames=0,
                    sent_frames=0,
                    acknowledged_frames=0,
                    dropped_frames=0,
                    high_water_frames=0,
                    max_queue_wait_ms=0.0,
                    max_inflight_bytes=0,
                    max_ack_age_ms=0.0,
                    byte_credit_waits=0,
                    abandoned_unacknowledged_frames=0,
                    dropped_by_reason={},
                ),
            )

    def _delivery_snapshot(self):
        now = time.monotonic()
        self.delivery["inflight_bytes"] = sum(self.inflight_bytes.values())
        for name, entries in self.inflight.items():
            stats = self.delivery["streams"][name]
            stats["inflight_bytes"] = self.inflight_bytes[name]
            stats["inflight_frames"] = len(entries)
            stats["oldest_unacknowledged_ms"] = (
                max(0, (now - entries[0][2]) * 1000) if entries else 0.0
            )
            values = self.media[name]
            stats["queued_frames"] = len(values)
            stats["oldest_queued_ms"] = (
                max(0, (now - values[0][0]) * 1000) if values else 0.0
            )

    def _dropped(self, stream, reason, count):
        if count:
            stats = self.delivery["streams"][stream]
            stats["dropped_frames"] += count
            reasons = stats["dropped_by_reason"]
            reasons[reason] = reasons.get(reason, 0) + count

    def _discard_queue(self, stream, reason):
        if self.large_au_stream == stream:
            self.large_au_stream = None
        values = self.media[stream]
        self._dropped(stream, reason, len(values))
        self.media_bytes -= sum(item[1] for item in values)
        values.clear()
        self.needs_key[stream] = True
        self.owner.request_keyframe(stream)

    def stop(self, reason, *, error=None):
        self.closed, self.reason = True, reason
        self.error = error
        self.control.clear()
        for stream, values in self.media.items():
            self._dropped(stream, "attachment_closed", len(values))
            values.clear()
            # They may already have reached the browser: account for the
            # uncertainty separately from known unsent media drops.
            self.delivery["streams"][stream]["abandoned_unacknowledged_frames"] += len(
                self.inflight[stream]
            )
            self.inflight[stream].clear()
            self.inflight_bytes[stream] = 0
        self.control_bytes = self.media_bytes = 0
        self._delivery_snapshot()
        self.wake.set()

    def replay_controls(self, history):
        for _, _, message in history:
            self.offer_control(message)

    def offer_control(self, message):
        if self.closed:
            return
        if message.get("type") == "session.ready" and not self.ready_for_acks:
            self.ready_for_acks = True
            for name in self.media:
                self.last_ack[name] = time.monotonic()
        wire = msgspec.msgpack.encode(message)
        if (
            len(wire) + self.control_bytes > self.control_limit
            or len(self.control) >= 4096
        ):
            self.stop("CONTROL_CONSUMER_SLOW")
            return
        self.control.append(wire)
        self.control_bytes += len(wire)
        self.wake.set()

    def offer_media(self, message):
        if self.closed:
            return
        stream = message["stream"]
        values = self.media[stream]
        stats = self.delivery["streams"][stream]
        stats["offered_frames"] += 1
        size = len(msgspec.msgpack.encode(message))
        if size > self.max_access_unit_bytes:
            # Reject an exceptional AU as a whole; never feed its dependent
            # deltas into a decoder after skipping that reference.
            self._discard_queue(stream, "access_unit_bytes")
            self._dropped(stream, "access_unit_bytes", 1)
            return
        stale = bool(values and time.monotonic() - values[0][0] > self.max_media_age)
        reason = (
            "queue_age"
            if stale
            else (
                "queue_frames"
                if len(values) >= self.frame_limit
                else (
                    "queue_bytes" if self.media_bytes + size > self.byte_limit else None
                )
            )
        )
        if reason:
            self._discard_queue(stream, reason)
        if self.needs_key[stream] and not message["keyframe"]:
            self._dropped(stream, "waiting_for_keyframe", 1)
            return
        if size > self.byte_limit or self.media_bytes + size > self.byte_limit:
            self.needs_key[stream] = True
            self._dropped(stream, "queue_bytes", 1)
            return
        self.needs_key[stream] = False
        values.append((time.monotonic(), size, message))
        self.media_bytes += size
        stats["high_water_frames"] = max(stats["high_water_frames"], len(values))
        self.delivery["high_water_bytes"] = max(
            self.delivery["high_water_bytes"], self.media_bytes
        )
        self.wake.set()

    def ack(self, payload):
        stream = payload.get("stream")
        frame, window = payload.get("frame"), payload.get("window")
        if (
            payload.get("protocol") != "whitebox-dual-v2"
            or stream not in self.media
            or type(window) is not int
            or not 1 <= window <= 32
            or type(frame) is not int
            or not -1 <= frame <= self.sent[stream]
            or self.windows[stream] not in (None, window)
        ):
            raise ValueError("Invalid playback receipt")
        self.windows[stream] = window
        self.delivery["streams"][stream]["acknowledged_frames"] += max(
            0, frame - self.received[stream]
        )
        self.received[stream] = max(frame, self.received[stream])
        outstanding = self.inflight[stream]
        while outstanding and outstanding[0][0] <= self.received[stream]:
            _, size, sent_at = outstanding.popleft()
            self.inflight_bytes[stream] -= size
            stats = self.delivery["streams"][stream]
            stats["max_ack_age_ms"] = max(
                stats["max_ack_age_ms"], (time.monotonic() - sent_at) * 1000
            )
        self._delivery_snapshot()
        self.last_ack[stream] = time.monotonic()
        self.wake.set()

    async def recv(self):
        while not self.closed:
            self._delivery_snapshot()
            # Initialization and an idle producer aren't missing browser receipts.
            # Check each stream even while another stream continuously wakes us.
            if self.ready_for_acks and any(
                (self.windows[name] is None or self.sent[name] > self.received[name])
                and time.monotonic() - self.last_ack[name] > self.ack_timeout
                for name in self.media
            ):
                self.stop("PLAYBACK_RECEIPT_TIMEOUT")
                break
            if self.control:
                wire = self.control.popleft()
                self.control_bytes -= len(wire)
                return wire
            order = (
                self.next_stream,
                "model" if self.next_stream == "whitebox" else "whitebox",
            )
            if self.large_au_stream is not None:
                order = (self.large_au_stream,)
            for stream in order:
                window = self.windows[stream]
                values = self.media[stream]
                if values and time.monotonic() - values[0][0] > self.max_media_age:
                    self._discard_queue(stream, "queue_age")
                if (
                    self.media[stream]
                    and window is not None
                    and self.sent[stream] - self.received[stream] < window
                ):
                    offered_at, size, message = self.media[stream][0]
                    wire = msgspec.msgpack.encode(
                        dict(
                            message,
                            media_sequence=message["output_frame_start"],
                            output_frame_start=self.sent[stream] + 1,
                            connection_generation=self.generation,
                        )
                    )
                    if len(wire) > self.max_access_unit_bytes:
                        self._discard_queue(stream, "access_unit_bytes")
                        continue
                    shared = sum(self.inflight_bytes.values())
                    oversized = len(wire) > self.inflight_byte_limit
                    blocked = (
                        shared != 0
                        if oversized
                        else self.inflight_bytes[stream] + len(wire)
                        > self.inflight_byte_limit
                        or shared + len(wire) > self.shared_inflight_byte_limit
                    )
                    if blocked:
                        self.delivery["streams"][stream]["byte_credit_waits"] += 1
                        if oversized:
                            # Stop admitting the other stream until this bounded
                            # IDR gets a turn; otherwise a continuous source can
                            # prevent the shared window from ever emptying.
                            self.large_au_stream = stream
                            break
                        continue
                    # One oversized AU may progress when this stream is empty.
                    # Splitting/dropping it would break H.264 reference integrity.
                    self.media[stream].popleft()
                    self.large_au_stream = None
                    self.media_bytes -= size
                    stats = self.delivery["streams"][stream]
                    stats["sent_frames"] += 1
                    stats["max_queue_wait_ms"] = max(
                        stats["max_queue_wait_ms"],
                        (time.monotonic() - offered_at) * 1000,
                    )
                    if self.sent[stream] == self.received[stream]:
                        self.last_ack[stream] = time.monotonic()
                    self.sent[stream] += 1
                    self.inflight[stream].append(
                        (self.sent[stream], len(wire), time.monotonic())
                    )
                    self.inflight_bytes[stream] += len(wire)
                    stats["max_inflight_bytes"] = max(
                        stats["max_inflight_bytes"], self.inflight_bytes[stream]
                    )
                    self.next_stream = "model" if stream == "whitebox" else "whitebox"
                    self._delivery_snapshot()
                    # Wire delivery sequence is per attachment; source / model /
                    # archive identities remain unchanged across reconnections.
                    return wire
            if self.owner.terminal is not None:
                # A client that cannot drain promptly does not hold the model or
                # block termination. Wait for delivered media receipts as well;
                # otherwise closing races the client's final playback ACK.
                if self.owner.terminal["type"] == "error" or (
                    not any(self.media.values())
                    and all(
                        self.sent[name] == self.received[name] for name in self.media
                    )
                ):
                    terminal = dict(self.owner.terminal)
                    if terminal["type"] == "session.completed":
                        terminal.update(
                            frames=self.sent["model"] + 1,
                            source_frames=self.sent["whitebox"] + 1,
                            playback_delivery=copy.deepcopy(self.delivery),
                        )
                    return msgspec.msgpack.encode(terminal)
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), min(1, self.ack_timeout))
            except TimeoutError:
                pass
        if getattr(self, "error", None) is not None:
            raise self.error
        raise ConnectionError(self.reason)


class RoomPipeline:
    command_byte_limit = 512 * 1024
    reliable_control_types = {
        "whitebox.states",
        "whitebox.control",
        "whitebox.capture",
        "whitebox.model_timing",
        "model.cache",
    }

    def __init__(self, host, init, user_id, trace_id, *, member_source=None):
        if "source_encoding" in init.get("whitebox", {}):
            from .session_encoding import check_source_encoding_scope

            check_source_encoding_scope(
                init,
                transport=getattr(self, "media_transport", "websocket"),
                existing_room=init["whitebox"].get("game_id") in host.runtime.rooms,
            )
        self.trace_id = trace_id
        self.host, self.init, self.user_id = host, copy.deepcopy(init), user_id
        self.receipt_wire = InputReceiptWire(self.init)
        self.game_id = checked_id(init["whitebox"]["game_id"])
        spec = session_source_spec(
            init["whitebox"], host.config.spec, host.runtime.rooms
        )
        self.session_id, self.player_id, self.stream_id = new_id(), new_id(), new_id()
        self.token = secrets.token_urlsafe(32)
        self.generation = 0
        self.subscription = None
        self.retention = None
        self.ready = None
        self.first_media_at = None
        self.terminal = None
        self.history = deque()
        self.history_bytes = 0
        self.control_sequence = 0
        self.input_history = OrderedDict()
        self.committed_input = -1
        self.history_floor = 0
        self.keyframe_requests = {}
        self.command_queue = asyncio.Queue(maxsize=64)
        self.command_bytes = 0
        self.report = SessionReport(trace_id)
        self.member_source = member_source
        self.member_closed = asyncio.Event()
        if member_source is not None:
            self.player_id = member_source.record["player_id"]
            self.stream_id = member_source.record["stream_id"]
        factory = member_source or LocalRoomFactory(
            host.runtime, self.game_id, self.player_id, self.stream_id
        )
        from .session_encoding import session_source_config

        self.session = self.create_session(
            config=replace(
                session_source_config(host.config, init["whitebox"]),
                engine_factory=factory,
                spec=spec,
            ),
            archives=host.archives,
            trace_id=trace_id,
            session_id=self.session_id,
            user_id=user_id,
            reports=host.reports,
            report=self.report,
        )
        self.task = None
        self.commands = None
        self.closed = False

    def create_session(self, **kwargs):
        return InteractiveSession(
            leases=self.host.make_leases(self.user_id, kwargs["trace_id"]), **kwargs
        )

    def create_subscription(self):
        return Subscription(self, self.generation)

    def attachment_fields(self):
        return {}

    def resume_capability_fields(self):
        # Service-launched sources use a generation-fenced short capability.
        # Their long-lived owner token must never enter browser control events.
        return (
            {}
            if hasattr(self, "single_player_operation")
            or getattr(self, "member_source", None) is not None
            else {"resume_token": self.token}
        )

    def request_keyframe(self, stream):
        now = time.monotonic()
        if now - self.keyframe_requests.get(stream, -float("inf")) < 0.25:
            return
        self.keyframe_requests[stream] = now
        encoder = (
            self.session.source_playback
            if stream == "whitebox"
            else self.session.playback
        )
        if encoder and hasattr(encoder, "request_keyframe"):
            encoder.request_keyframe()

    def attach(self, resume=None):
        if self.terminal is not None or self.closed or getattr(self, "closing", False):
            raise ValueError("SESSION_ENDED")
        after = 0
        if self.generation:
            if not isinstance(resume, dict) or not hmac.compare_digest(
                str(resume.get("resume_token", "")), self.token
            ):
                raise ValueError("RESUME_DENIED")
            after = resume.get("control_sequence", 0)
            if type(after) is not int or not 0 <= after <= self.control_sequence:
                raise ValueError("INVALID_RESUME_POSITION")
            if after < self.history_floor:
                raise ValueError("STATE_HISTORY_EXPIRED")
            requested = resume.get("whitebox", {}).get("resolution")
            if requested is not None:
                from .model_geometry import resolution_spec

                if (
                    resolution_spec(requested, self.session.config.spec)
                    != self.session.config.spec
                ):
                    raise WhiteboxDiagnostic("RESOLUTION_UNSUPPORTED", "validation")
        if self.retention:
            self.retention.cancel()
            self.retention = None
        if self.subscription:
            self.subscription.stop("CONNECTION_REPLACED")
        self.generation += 1
        sub = self.subscription = self.create_subscription()
        sub.offer_control(
            dict(
                version=2,
                type="session.attached",
                session_id=self.session_id,
                game_id=self.game_id,
                **self.resume_capability_fields(),
                owner_epoch=self.host.runtime.epoch,
                connection_generation=self.generation,
                reconnect_grace_seconds=self.host.grace_seconds,
                last_input_sequence=self.committed_input,
                **self.attachment_fields(),
            )
        )
        if self.ready:
            sub.offer_control(self.ready_message())
        if self.ready or hasattr(self, "single_player_operation"):
            sub.replay_controls(row for row in self.history if row[0] > after)
        for stream in ("whitebox", "model"):
            self.request_keyframe(stream)
        return sub

    def ready_message(self):
        return dict(
            self.ready,
            connection_generation=self.generation,
            **self.resume_capability_fields(),
            room_pipeline_protocol="room-pipeline-v1",
            play_remaining_seconds=(
                max(
                    0,
                    self.host.config.play_seconds
                    - (time.monotonic() - self.first_media_at),
                )
                if self.first_media_at
                else self.host.config.play_seconds
            ),
            owner_epoch=self.host.runtime.epoch,
            reconnect_grace_seconds=self.host.grace_seconds,
            last_input_sequence=self.committed_input,
            display_revision=self.session.revision,
            display_mode=self.session.display_mode,
            report_path=f"whitebox/owners/{self.host.runtime.owner_id}/recordings/{self.stream_id}",
        )

    def detach(self, sub):
        sub.stop("DISCONNECTED")
        if self.subscription is not sub:
            return
        self.subscription = None
        if self.terminal is None and not self.closed:

            async def expire():
                await asyncio.sleep(self.host.grace_seconds)
                if self.subscription is None:
                    await self.close("RECONNECT_EXPIRED")

            self.retention = asyncio.create_task(expire())

    def publish(self, message):
        kind = message.get("type")
        if getattr(self, "member_source", None) is not None and kind not in {
            "error",
            "session.completed",
        }:
            self.member_source.check()
            if kind in {"whitebox.frame", "frame_batch"}:
                self.member_source.frame(message)
        if kind == "session.ready":
            self.ready = self.receipt_wire.control(message)
            if self.subscription:
                self.subscription.offer_control(self.ready_message())
            return
        if kind in {"whitebox.frame", "frame_batch"}:
            if self.subscription:
                self.subscription.offer_media(self.receipt_wire.media(message))
            if self.first_media_at is None:
                self.first_media_at = time.monotonic()
            # Preserve the configured play deadline even during brief disconnects.
            self.session.media_delivered()
            return
        if kind in {"session.completed", "error"}:
            self.terminal = message
            if self.subscription:
                self.subscription.wake.set()
            return
        if kind in self.reliable_control_types:
            # Compact once before storing. Reconnect replays these exact deltas
            # after the client's applied cursor, rather than compacting again.
            message = self.receipt_wire.control(message)
            self.control_sequence += 1
            message = dict(message, control_sequence=self.control_sequence)
            size = len(msgspec.msgpack.encode(message))
            if size > self.host.history_bytes:
                self.history_floor = self.control_sequence
                if self.subscription:
                    self.subscription.stop("STATE_MESSAGE_TOO_LARGE")
                return
            if self.history and self.receipt_wire.repeated_observation(
                self.history[-1][2],
                message,
                self.history[-2][2] if len(self.history) > 1 else None,
            ):
                _, old_size, _ = self.history.pop()
                self.history_bytes -= old_size
            self.history.append((self.control_sequence, size, message))
            self.history_bytes += size
            while self.history_bytes > self.host.history_bytes:
                sequence, size, _ = self.history.popleft()
                self.history_bytes -= size
                self.history_floor = sequence
        if self.subscription:
            self.subscription.offer_control(message)

    async def start(self):
        await self.session.send(msgspec.msgpack.encode(self.init))
        self.commands = asyncio.create_task(
            self._commands(), name="room-player-controls"
        )
        self.task = asyncio.create_task(self._run(), name="room-player-pipeline")

    def accept(self, sub, wire):
        if (
            sub is not self.subscription
            or sub.closed
            or sub.generation != self.generation
        ):
            raise ValueError("STALE_CONNECTION")
        if len(wire) > 65536:
            raise ValueError("CONTROL_MESSAGE_TOO_LARGE")
        message = msgspec.msgpack.decode(wire)
        if not isinstance(message, dict):
            raise ValueError("INVALID_CONTROL_MESSAGE")
        if message.get("kind") == "playback_ack":
            sub.ack(message.get("payload", {}))
            return
        if message.get("kind") == "playback.keyframe":
            stream = message.get("payload", {}).get("stream")
            if stream not in sub.media:
                raise ValueError("INVALID_STREAM")
            sub._discard_queue(stream, "keyframe_requested")
            return
        if message.get("kind") == "heartbeat":
            self.session.room_touch_requested.set()
            return
        if message.get("type") in {"client_metric", "client_metric_batch"}:
            # Decoder callbacks can arrive in a burst after main-thread/network
            # stalls. Diagnostics must not consume the bounded input queue.
            self.session.observe_client_metric(message)
            return
        if message.get("kind") == "network.ping":
            payload = message.get("payload")
            nonce = payload.get("nonce") if isinstance(payload, dict) else None
            if type(nonce) is not int or not 0 <= nonce < 2**53:
                raise ValueError("Invalid network probe")
            self.publish(
                dict(
                    version=2,
                    type="whitebox.pong",
                    nonce=nonce,
                    server_epoch_ms=time.time() * 1000,
                )
            )
            return
        if message.get("type") == "init":
            raise ValueError("SESSION_ALREADY_INITIALIZED")
        self._enqueue_command(sub, wire, message)

    def _enqueue_command(self, sub, wire, message):
        # Render assets and gameplay inputs share one ordered queue and therefore
        # must use the same byte budget, regardless of their per-message limits.
        if (
            self.command_queue.full()
            or self.command_bytes + len(wire) > self.command_byte_limit
        ):
            raise WhiteboxDiagnostic("CONTROL_QUEUE_FULL", "control")
        self.command_bytes += len(wire)
        self.command_queue.put_nowait((sub.generation, wire, message))

    async def _commands(self):
        while True:
            generation, wire, message = await self.command_queue.get()
            self.command_bytes -= len(wire)
            if generation != self.generation:
                continue
            applying = False
            # Replies to operation.get retain the exact ordered ingress owner,
            # even if the subscriber is replaced before the outgoing pump runs.
            self.session.control_generation = generation
            try:
                if message.get("kind") == "input":
                    sequence = message.get("sequence")
                    digest = hashlib.sha256(
                        json.dumps(
                            message.get("payload"),
                            sort_keys=True,
                            allow_nan=False,
                            separators=(",", ":"),
                        ).encode()
                    ).digest()
                    if type(sequence) is not int:
                        raise ValueError("INVALID_INPUT_SEQUENCE")
                    if sequence <= self.session.input_sequence:
                        if self.input_history.get(sequence) != digest:
                            raise ValueError("INPUT_REPLAY_MISMATCH")
                        self.publish(
                            dict(
                                version=2,
                                type="whitebox.input_committed",
                                sequence=sequence,
                                duplicate=True,
                                server_epoch_ms=time.time() * 1000,
                            )
                        )
                        continue
                    applying = True
                    await self.session.send(wire)
                    self.input_history[sequence] = digest
                    self.committed_input = sequence
                    self.publish(
                        dict(
                            version=2,
                            type="whitebox.input_committed",
                            sequence=sequence,
                            server_epoch_ms=time.time() * 1000,
                        )
                    )
                    while len(self.input_history) > 256:
                        self.input_history.popitem(last=False)
                else:
                    applying = True
                    await self.session.send(wire)
            except Exception as error:
                logging.getLogger(__name__).exception(
                    "Room control failed trace_id=%s source_session_id=%s "
                    "generation=%s input_sequence=%s committed_input=%s wire_bytes=%s",
                    self.trace_id,
                    self.session_id,
                    generation,
                    self.session.input_sequence,
                    self.committed_input,
                    len(wire),
                )
                if (
                    message.get("kind") == "input"
                    and self.session.input_sequence > self.committed_input
                ):
                    await self.close("INPUT_APPLICATION_FAILED")
                    return
                # An invalid peer cannot replace the world or affect another
                # player; fence this attachment and allow bounded recovery.
                if self.subscription and self.subscription.generation == generation:
                    if applying:
                        from .diagnostics import public_error

                        diagnostic = public_error(
                            error,
                            self.trace_id,
                            default="CONTROL_SERVICE_FAILED",
                            stage="control",
                        )
                        failure = WhiteboxDiagnostic(
                            "CONTROL_SERVICE_FAILED", "control"
                        )
                        # Public render codes also travel through this boundary;
                        # retain their sanitized identity without an exception body.
                        failure.code, failure.stage = (
                            diagnostic["code"],
                            diagnostic["stage"],
                        )
                        failure.upstream_code = diagnostic.get("upstream_code")
                        failure.cleanup_code = diagnostic.get("cleanup_code")
                        self.subscription.stop(diagnostic["code"], error=failure)
                    else:
                        self.subscription.stop("INVALID_CONTROL_MESSAGE")

    async def _run(self):
        try:
            while True:
                message = msgspec.msgpack.decode(await self.session.recv())
                self.publish(message)
                if message.get("type") == "session.completed":
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            from .diagnostics import public_error

            self.publish(
                dict(
                    version=2,
                    type="error",
                    **public_error(exc, self.trace_id, default="ROOM_PIPELINE_FAILED"),
                )
            )
        finally:
            await self.close()

    async def close(self, reason=None):
        if not hasattr(self, "member_closed"):
            self.member_closed = asyncio.Event()
        if self.closed:
            if getattr(self, "close_error", None) is not None:
                raise self.close_error
            return
        if getattr(self, "closing", False):
            # An external membership/Host close must wait for the original
            # borrowed-media teardown. Its own cancelled worker must not wait
            # on the caller that is draining that same worker.
            if asyncio.current_task() not in {
                self.task,
                self.commands,
                self.retention,
            }:
                await self.member_closed.wait()
                if getattr(self, "close_error", None) is not None:
                    raise self.close_error
            return
        self.closing = True
        if reason and self.terminal is None:
            self.publish(
                dict(
                    version=2,
                    type="error",
                    code=reason,
                    content="This play session has ended.",
                )
            )
        tasks = [
            task
            for task in (self.task, self.commands, self.retention)
            if task and task is not asyncio.current_task()
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await self.session.close()
            if self.terminal is None and getattr(
                self.session, "close_requested", False
            ):
                # A soft-drain timeout can cancel the outgoing reader before
                # its EOS. Publish completion only after actual owner cleanup.
                self.publish(
                    dict(version=2, type="session.completed", cleanup_confirmed=True)
                )
        except BaseException as error:
            self.close_error = error
            if hasattr(self, "terminal"):
                self.publish(
                    dict(
                        version=2,
                        type="error",
                        code="RELEASE_UNKNOWN",
                        stage="cleanup",
                        content="Player resource cleanup could not be confirmed.",
                    )
                )
            if not hasattr(self.host, "retiring_sessions"):
                self.host.retiring_sessions = {}
            self.host.retiring_sessions[self.session_id] = self
            raise
        finally:
            # Session teardown already attempts engine/renderer/archive cleanup
            # in finally blocks. A cleanup error must not retain a dead source
            # in the single-session admission registry or strand its report.
            try:
                if (
                    getattr(self, "member_source", None) is not None
                    and self.subscription is not None
                ):
                    self.subscription.stop("MEMBER_MEDIA_ENDED")
                    await self.subscription.wait_closed()
            finally:
                self.closed = True
                self.member_closed.set()
                self.host.sessions.pop(self.session_id, None)
                self.init = {}
                self.history.clear()
                self.history_bytes = 0
                self.input_history.clear()
                self.report.gateway_done.set()
                if self.report.recording is None:
                    self.report.close()


class RoomPipelines:
    def __init__(
        self,
        runtime,
        config,
        archives,
        reports,
        make_leases,
        *,
        grace_seconds=15,
        max_sessions=1,
        history_bytes=16 * 1024**2,
    ):
        if not 0 < grace_seconds <= 60 or max_sessions < 1 or history_bytes < 65536:
            raise ValueError("Invalid Room pipeline budgets")
        self.runtime, self.config, self.archives, self.reports, self.make_leases = (
            runtime,
            config,
            archives,
            reports,
            make_leases,
        )
        self.grace_seconds, self.max_sessions, self.history_bytes = (
            grace_seconds,
            max_sessions,
            history_bytes,
        )
        self.sessions = {}
        self.lock = asyncio.Lock()

    async def attach(self, body):
        user_id, trace_id, message = body["user_id"], body["trace_id"], body["message"]
        if not isinstance(user_id, str) or not user_id or len(user_id) > 256:
            raise ValueError("INVALID_PLAYER_IDENTITY")
        if message.get("type") == "init":
            session_source_spec(
                message.get("whitebox", {}), self.config.spec, self.runtime.rooms
            )
        async with self.lock:
            if getattr(self.runtime, "maintenance", None) is not None:
                await self.runtime.maintenance.cancel_for_play()
            if getattr(self.runtime, "rtc_preparation", None) is not None:
                raise ValueError("CAPACITY_EXHAUSTED")
            if message.get("type") == "resume":
                session = self.sessions.get(checked_id(message["session_id"]))
                if (
                    session is None
                    or session.user_id != user_id
                    or session.game_id != message["whitebox"]["game_id"]
                    or message.get("owner_epoch") != self.runtime.epoch
                ):
                    raise ValueError("ROOM_SESSION_LOST")
                return session, session.attach(message)
            if message.get("type") != "init":
                raise ValueError("INIT_REQUIRED")
            if (
                sum(not session.closed for session in self.sessions.values())
                + len(getattr(self, "peer_sessions", {}))
                >= self.max_sessions
            ):
                raise ValueError("CAPACITY_EXHAUSTED")
            for key, session in list(self.sessions.items()):
                if len(self.sessions) <= 128:
                    break
                if session.closed:
                    del self.sessions[key]
            session = RoomPipeline(self, message, user_id, trace_id)
            self.sessions[session.session_id] = session
            sub = session.attach()
            try:
                await session.start()
            except BaseException:
                await session.close("INITIALIZATION_FAILED")
                raise
            return session, sub

    async def close(self):
        await asyncio.gather(
            *(
                session.close("ROOM_STOPPED")
                for session in list(self.sessions.values())
            ),
            return_exceptions=True,
        )

    def install(self, app, token):
        @app.websocket("/internal/pipeline")
        async def pipeline(websocket: WebSocket):
            require_internal_token(token, websocket.headers.get("authorization", ""))
            await websocket.accept()
            session = sub = None
            tasks = []
            try:
                wire = await asyncio.wait_for(websocket.receive_bytes(), 30)
                if len(wire) > 8 * 1024**2 + 16384:
                    raise ValueError("INIT_TOO_LARGE")
                body = msgspec.msgpack.decode(wire)
                session, sub = await self.attach(body)

                async def receive():
                    while True:
                        session.accept(sub, await websocket.receive_bytes())

                async def send():
                    while True:
                        value = await sub.recv()
                        await asyncio.wait_for(websocket.send_bytes(value), 10)
                        if msgspec.msgpack.decode(value).get("type") in {
                            "session.completed",
                            "error",
                        }:
                            return

                tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            except Exception as exc:
                try:
                    codes = {
                        "CAPACITY_EXHAUSTED",
                        "ROOM_SESSION_LOST",
                        "RESUME_DENIED",
                        "SESSION_ENDED",
                        "STATE_HISTORY_EXPIRED",
                        "INVALID_RESUME_POSITION",
                        "INVALID_CONTROL_MESSAGE",
                    }
                    code = str(exc) if str(exc) in codes else "ROOM_CONNECTION_CLOSED"
                    await websocket.send_bytes(
                        msgspec.msgpack.encode(
                            dict(
                                version=2,
                                type="error",
                                code=code,
                                reconnectable=code == "ROOM_CONNECTION_CLOSED"
                                and session is not None,
                                content="The connection was interrupted. Please reconnect.",
                            )
                        )
                    )
                except Exception:
                    pass
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if session and sub:
                    session.detach(sub)
                try:
                    await websocket.close()
                except Exception:
                    pass
