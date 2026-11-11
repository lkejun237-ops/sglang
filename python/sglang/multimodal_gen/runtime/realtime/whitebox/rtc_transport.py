# SPDX-License-Identifier: Apache-2.0
"""RTC-only handoff and reliable control; deliberately no WS media ACK window."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import math
import os
import time
from collections import deque

import msgspec

from .diagnostics import WhiteboxDiagnostic, public_error
from .opening_presentation import OpeningPresentationOffers
from .render_contract import RenderError
from .rooms import new_id
from .rtc_playout import ModelPresentationClock
from .session_encoding import source_bitrate

CODEC_ENCODER_MIN_BITRATE = 100_000
# Hard quality floors; native transport preserves the matching send budget.
MODEL_ENCODER_MIN_BITRATE = int(os.environ.get("WHITEBOX_MODEL_MIN_BITRATE", "1000000"))
if not 0 < MODEL_ENCODER_MIN_BITRATE <= 100_000_000:
    raise ValueError("WHITEBOX_MODEL_MIN_BITRATE must be between 1 and 100000000 bps")
SOURCE_ENCODER_MIN_BITRATE = 300_000


def encoder_budget(rate, source_ceiling, model_ceiling, *, model_active):
    """Preserve the configured quality floors; native pacing uses the same floor."""
    source_min = min(source_ceiling, SOURCE_ENCODER_MIN_BITRATE)
    if not model_active:
        return {"whitebox": min(source_ceiling, max(source_min, int(rate * 0.9)))}
    model_min = min(model_ceiling, MODEL_ENCODER_MIN_BITRATE)
    budget = max(source_min + model_min, int(rate * 0.9))
    share = budget * model_ceiling // (source_ceiling + model_ceiling)
    model = min(model_ceiling, max(model_min, share))
    return {
        "whitebox": min(source_ceiling, max(source_min, budget - model)),
        "model": model,
    }


class AUQueue:
    """Only unsubmitted AUs; native RTP owns network loss and retransmissions."""

    def __init__(
        self,
        request_keyframe,
        *,
        clock=time.monotonic,
        age=0.2,
        frames=8,
        byte_limit=1 << 20,
    ):
        self.request_keyframe, self.clock = request_keyframe, clock
        self.age, self.frame_limit, self.byte_limit = age, frames, byte_limit
        self.items, self.bytes, self.needs_key = deque(), 0, True
        self.last_request = -float("inf")
        self.drops = {
            "expired": 0,
            "capacity": 0,
            "awaiting_keyframe": 0,
            "replaced": 0,
        }
        self.high_water = 0
        self.offered = 0
        self.taken = 0
        self.wait_ms_total = 0.0
        self.wait_ms_max = 0.0

    def request(self):
        now = self.clock()
        if now - self.last_request >= 0.25:
            self.last_request = now
            self.request_keyframe()

    def clear(self, reason):
        self.drops[reason] += len(self.items)
        self.items.clear()
        self.bytes = 0
        self.needs_key = True

    def _trim(self):
        now = self.clock()
        if not self.items:
            return
        expired = now - self.items[0][0] > self.age
        if (
            not expired
            and len(self.items) <= self.frame_limit
            and self.bytes <= self.byte_limit
        ):
            return
        reason = "expired" if expired else "capacity"
        suffix_bytes = 0
        # Keep the newest complete IDR chain that still fits every budget.
        for start in range(len(self.items) - 1, -1, -1):
            at, size, frame = self.items[start]
            suffix_bytes += size
            if (
                frame["keyframe"]
                and now - at <= self.age
                and len(self.items) - start <= self.frame_limit
                and suffix_bytes <= self.byte_limit
                and all(
                    self.items[i + 1][2]["output_frame_start"]
                    == self.items[i][2]["output_frame_start"] + 1
                    for i in range(start, len(self.items) - 1)
                )
            ):
                for _ in range(start):
                    _, size, _ = self.items.popleft()
                    self.bytes -= size
                    self.drops[reason] += 1
                return
        self.clear(reason)

    def offer(self, message):
        self.offered += 1
        self._trim()
        metadata = {key: value for key, value in message.items() if key != "payload"}
        size = len(message["payload"]) + len(msgspec.msgpack.encode(metadata))
        if size > self.byte_limit:
            self.clear("capacity")
            self.drops["capacity"] += 1
            return
        if self.needs_key:
            if not message.get("keyframe"):
                self.drops["awaiting_keyframe"] += 1
                return
            self.needs_key = False
        self.items.append((self.clock(), size, message))
        self.bytes += size
        self.high_water = max(self.high_water, self.bytes)
        self._trim()

    def take(self, *, trim=True):
        if trim:
            self._trim()
        if not self.items:
            if self.needs_key:
                self.request()
            return None
        at, size, message = self.items.popleft()
        self.bytes -= size
        self.taken += 1
        wait_ms = max(0.0, (self.clock() - at) * 1000)
        self.wait_ms_total += wait_ms
        self.wait_ms_max = max(self.wait_ms_max, wait_ms)
        return message

    def peek_size(self):
        self._trim()
        if not self.items and self.needs_key:
            self.request()
        return self.items[0][1] if self.items else None


class RTCSubscription:
    """A reliable control subscription plus two expiring native RTP handoffs."""

    def __init__(self, owner, generation, peer):
        self.owner, self.generation, self.peer = owner, generation, peer
        self.area_challenge = new_id()
        self.first_presentation_offers = None
        self.closed = False
        self.reason = "DISCONNECTED"
        self.wake = asyncio.Event()
        self.control = deque()
        self.control_bytes = 0
        self.replay = deque()
        self.replay_bytes = self.replay_bootstrap = 0
        self.replay_total = self.replayed_controls = 0
        self.model_clock = (
            ModelPresentationClock()
            if getattr(owner.host.config, "rtc_model_cadence", False)
            else None
        )
        self.media = {
            name: AUQueue(
                lambda name=name: owner.request_keyframe(name),
                # Model output arrives in blocks; a 250 ms encoder VBV burst
                # cannot drain inside the source preview's 200 ms deadline.
                # Keep at most one short model burst, without changing the
                # source preview freshness, byte cap, or native frame credits.
                age=0.5 if name == "model" and self.model_clock is not None else 0.2,
                frames=17 if name == "model" and self.model_clock is not None else 8,
            )
            for name in ("whitebox", "model")
        }
        self.stream_ids = {
            "whitebox": owner.session.source_epoch,
            "model": owner.session.current_model_stream,
        }
        self.sent = {"whitebox": 0, "model": 0}
        self.last_rtp = {"whitebox": None, "model": None}
        self.last_offered = {"whitebox": -1, "model": -1}
        self.model_eos = None
        self.model_eos_sent = None
        self.handoffs = {}
        self.last_sent_index = {name: -1 for name in self.media}
        self.send_service = {name: 0.0 for name in self.media}
        self.accepted = {name: 0 for name in self.media}
        self.departed = {name: 0 for name in self.media}
        self.failed = {name: 0 for name in self.media}
        self.cancelled = {name: 0 for name in self.media}
        self.media_failed = False
        self.handoff_ms_total = {name: 0.0 for name in self.media}
        self.handoff_ms_max = {name: 0.0 for name in self.media}
        self.pacer_samples = {name: 0 for name in self.media}
        self.pacer_ms_total = {name: 0.0 for name in self.media}
        self.pacer_ms_max = {name: 0.0 for name in self.media}
        self.target_bitrate = None
        self.receiver_bitrate = None
        self.applied_bitrates = {}
        self.report_delivery = None
        self.last_delivery_log = 0.0
        report = getattr(owner, "report", None)
        if report is not None:
            delivery = report.playback_delivery.setdefault(
                "rtc", {"attachment_count": 0, "recent_attachments": []}
            )
            delivery["attachment_count"] += 1
            self.report_delivery = {}
            delivery["recent_attachments"].append(self.report_delivery)
            del delivery["recent_attachments"][:-16]
        self.media_task = asyncio.create_task(
            self._send_media(), name="rtc-media-handoff"
        )
        self.feedback_task = asyncio.create_task(
            self._feedback(), name="rtc-encoder-feedback"
        )
        self.cleanup_task = None
        peer.on_event = self.native_event

    def stop(self, reason, *, error=None):
        if self.closed:
            return
        self.closed = True
        self.reason = reason
        self.error = error
        self.wake.set()
        for queue in self.media.values():
            queue.clear("replaced")
        self.control.clear()
        self.control_bytes = 0
        self.replay.clear()
        self.replay_bytes = self.replay_bootstrap = 0
        self.media_task.cancel()
        self.feedback_task.cancel()
        self.cleanup_task = asyncio.create_task(
            self._close(), name="rtc-attachment-cleanup"
        )

    async def _close(self):
        await asyncio.gather(
            self.media_task, self.feedback_task, return_exceptions=True
        )
        try:
            await self.peer.close()
        finally:
            self._record_delivery()

    async def wait_closed(self):
        if self.cleanup_task:
            await self.cleanup_task

    def reset_model(self, stream_id):
        self.media["model"].clear("replaced")
        if self.model_clock is not None:
            self.model_clock = ModelPresentationClock()
        self.stream_ids["model"] = stream_id
        self.last_rtp["model"] = None
        self.last_offered["model"] = -1
        self.sent["model"] = 0
        self.last_sent_index["model"] = -1
        self.model_eos = None
        self.model_eos_sent = None
        self.wake.set()

    def _discard_queue(self, stream, reason):
        self.media[stream].clear("replaced")
        self.media[stream].request()

    def ack(self, _):
        raise ValueError("RTC media does not use application playback receipts")

    def enable_first_presentation(self):
        self.first_presentation_offers = OpeningPresentationOffers()

    def resolve_first_presentation_offer(self, command, state):
        if (
            self.closed
            or self.owner.subscription is not self
            or command["connection_generation"] != self.generation
            or self.first_presentation_offers is None
        ):
            raise RenderError("MODEL_PRESENTATION_STALE", retryable=True)
        return self.first_presentation_offers.resolve(
            self.owner.session, command["offer_id"], command["ui_revision"], state=state
        )

    def native_event(self, message):
        if self.closed:
            return
        kind = message.get("type")
        if kind == "frame.meta":
            frame = message["frame"]
            if getattr(self.owner, "member_source", None) is not None:
                try:
                    self.owner.member_source.frame(frame)
                except (ValueError, KeyError):
                    self.stop("MEMBER_MEDIA_CHANGED")
                    return
            stream = message["track"]
            if frame.get("media_stream_id") != self.stream_ids[stream]:
                return
            if stream == "model" and self.first_presentation_offers is not None:
                offer = self.first_presentation_offers.offer(self.owner.session, frame)
                if offer is not None:
                    frame = dict(frame, first_presentation_offer=offer)
            self.offer_control(
                dict(
                    type="rtc.frame.meta",
                    track=stream,
                    rtp_timestamp=message["rtp_timestamp"],
                    connection_generation=self.generation,
                    frame=frame,
                )
            )
        elif kind == "keyframe" and message.get("track") in self.media:
            self.media[message["track"]].request()
        elif kind == "bitrate":
            self.target_bitrate = message["bitrate"]
            self.receiver_bitrate = message.get("receiver_bitrate", message["bitrate"])
        elif kind == "accepted" and message.get("track") in self.media:
            self.accepted[message["track"]] += 1
        elif kind == "state":
            self.offer_control(
                {
                    "type": "rtc.state",
                    "state": message["state"],
                    "connection_generation": self.generation,
                }
            )
            if message["state"] in {"failed", "closed"}:
                self.offer_control(
                    {
                        "type": "error",
                        "code": "RTC_DISCONNECTED",
                        "content": "The video connection was lost. Please reconnect.",
                        "reconnectable": True,
                    }
                )

    def replay_controls(self, history):
        # Reuse the already bounded history, without putting it all into the
        # live 512-message queue or dropping ordered UI/input deltas.
        self.replay_bootstrap = len(self.control)
        for row in history:
            if (
                row[1] > 8 << 20
                or self.replay_bytes + row[1] > self.owner.host.history_bytes
            ):
                self.stop("RTC_CONTROL_OVERFLOW")
                return
            self.replay.append(row)
            self.replay_bytes += row[1]
        self.replay_total = len(self.replay)
        self.wake.set()

    def _control_wire(self, message):
        segment = message.get("segment") or {}
        if (
            message.get("type") in {"render.status", "source.status"}
            and segment.get("state") in {"ended", "cancelled", "failed"}
            and segment.get("segment_epoch") == self.stream_ids["model"]
            and self.stream_ids["model"] is not None
            and self.model_eos_sent != self.stream_ids["model"]
        ):
            self.model_eos = dict(
                type="rtc.eos",
                track="model",
                source_epoch=self.owner.session.source_epoch,
                media_stream_id=self.stream_ids["model"],
                connection_generation=self.generation,
            )
        if message.get("type") == "session.attached":
            message = dict(
                message, media_transport="webrtc-rtp", resume_user_id=self.owner.user_id
            )
        elif message.get("type") == "session.ready":
            message = dict(
                message, media_transport="webrtc-rtp", playback_flow_control=None
            )
        return msgspec.msgpack.encode(message)

    def offer_control(self, message, *, optional=False):
        if self.closed:
            return
        wire = self._control_wire(message)
        if self._control_overflow(len(wire)):
            if optional:
                return
            self._discard_optional_controls()
            if self._control_overflow(len(wire)):
                self.stop("RTC_CONTROL_OVERFLOW")
                return
        self.control.append((time.monotonic(), message["type"], wire, optional))
        self.control_bytes += len(wire)
        self.wake.set()

    def _control_overflow(self, size):
        return len(self.control) >= 512 or self.control_bytes + size > 8 << 20

    def _discard_optional_controls(self):
        retained = deque()
        bootstrap = 0
        for index, entry in enumerate(self.control):
            if entry[3]:
                self.control_bytes -= len(entry[2])
            else:
                retained.append(entry)
                if index < self.replay_bootstrap:
                    bootstrap += 1
        self.control = retained
        self.replay_bootstrap = bootstrap

    def offer_media(self, message):
        stream = message.get("stream")
        if (
            self.closed
            or stream not in self.media
            or message.get("media_stream_id") != self.stream_ids[stream]
        ):
            return
        self.last_offered[stream] = message["output_frame_start"]
        if stream == "model" and self.model_clock is not None:
            # The immutable source message also feeds archives and WS consumers.
            # Only the native sender consumes this attachment-local RTP clock.
            message = dict(
                message, rtc_pts_us=self.model_clock.observe(message["pts_us"])
            )
        if not self.media[stream].items and not self._inflight(stream):
            # An idle track cannot accumulate unlimited priority over a live one.
            self.send_service[stream] = max(self.send_service.values())
        self.media[stream].offer(message)
        self.wake.set()

    async def _send_media(self):
        try:
            while not self.closed:
                if getattr(self.owner, "member_source", None) is not None:
                    self.owner.member_source.check()
                self.wake.clear()
                for task in list(self.handoffs):
                    if not task.done():
                        continue
                    self._complete_handoff(task)
                    self.wake.set()
                # Fair by encoded bytes, with the configured model/source ratio.
                # Credits cover IPC and native pacing, not remote playback ACKs.
                while len(self.handoffs) < 4:
                    candidates = []
                    used = sum(value[2] for value in self.handoffs.values())
                    limit = self._handoff_budget()
                    for stream, queue in self.media.items():
                        size = queue.peek_size()
                        if (
                            size is None
                            or self.stream_ids[stream] is None
                            or self._inflight(stream) >= 2
                        ):
                            continue
                        # One oversized IDR may run alone, under AU/native hard caps.
                        if used and used + size > limit:
                            continue
                        candidates.append((self.send_service[stream], stream, size))
                    if not candidates:
                        break
                    _, stream, size = min(candidates)
                    # No await between peek and removal: charge the same AU even
                    # if an expiry boundary passes during this scheduler iteration.
                    message = self.media[stream].take(trim=False)
                    if message is None:
                        continue
                    ceiling = (
                        self.owner.session.config.playback_max_bitrate
                        if stream == "model"
                        else source_bitrate(self.owner.session.config)
                    )
                    self.send_service[stream] += size / max(1, ceiling)
                    task = asyncio.create_task(self.peer.send_frame(message))
                    self.handoffs[task] = (stream, message, size, time.monotonic())
                    task.add_done_callback(lambda _: self.wake.set())
                try:
                    await asyncio.wait_for(self.wake.wait(), 0.01)
                except asyncio.TimeoutError:
                    pass  # Expire unsubmitted AUs even if both producers are idle.
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.media_failed = True
            failure = WhiteboxDiagnostic("RTC_MEDIA_STALLED", "transport")
            failure.upstream_code = getattr(error, "upstream_code", None)
            self.offer_control(
                dict(
                    type="error",
                    **public_error(failure, getattr(self.owner, "trace_id", None)),
                    reconnectable=True,
                )
            )
        finally:
            for task in self.handoffs:
                task.cancel()
            await asyncio.gather(*self.handoffs, return_exceptions=True)
            for task in list(self.handoffs):
                with contextlib.suppress(Exception):
                    self._complete_handoff(task)
            self.wake.set()

    def _complete_handoff(self, task):
        stream, message, _, started = self.handoffs.pop(task)
        if task.cancelled():
            self.cancelled[stream] += 1
            return
        try:
            result = task.result()
        except Exception:
            self.failed[stream] += 1
            raise
        self.departed[stream] += 1
        elapsed = (time.monotonic() - started) * 1000
        self.handoff_ms_total[stream] += elapsed
        self.handoff_ms_max[stream] = max(self.handoff_ms_max[stream], elapsed)
        paced = result.get("pacer_duration_ms")
        if isinstance(paced, (int, float)) and math.isfinite(paced) and paced >= 0:
            self.pacer_samples[stream] += 1
            self.pacer_ms_total[stream] += paced
            self.pacer_ms_max[stream] = max(self.pacer_ms_max[stream], paced)
        if message["media_stream_id"] == self.stream_ids[stream]:
            self.sent[stream] += 1
            if message["output_frame_start"] > self.last_sent_index[stream]:
                self.last_sent_index[stream] = message["output_frame_start"]
                self.last_rtp[stream] = result["rtp_timestamp"]

    def _inflight(self, stream=None):
        return sum(
            1
            for value in self.handoffs.values()
            if stream is None or value[0] == stream
        )

    def _handoff_budget(self):
        # 75ms soft drain budget, charged for metadata as well as encoded bytes.
        # The native pacer independently retains its 2MiB/1s hard failure bounds.
        return min(1 << 20, max(1, int((self.target_bitrate or 4_000_000) * 0.075 / 8)))

    async def _feedback(self):
        last_snapshot = 0.0
        while not self.closed:
            await asyncio.sleep(0.5)
            if time.monotonic() - last_snapshot >= 1:
                self._record_delivery()
                if getattr(self.peer, "delivery_stats_version", 0) == 1:
                    self.offer_control(
                        dict(
                            type="rtc.delivery",
                            connection_generation=self.generation,
                            streams=self.snapshot(),
                        )
                    )
                last_snapshot = time.monotonic()
            rate = self.target_bitrate
            if type(rate) is not int or rate <= 0:
                continue
            source = self.owner.session.source_playback
            segment = self.owner.session.current_segment
            model = segment.playback if segment else None
            targets = encoder_budget(
                rate,
                source_bitrate(self.owner.session.config),
                self.owner.session.config.playback_max_bitrate,
                model_active=model is not None
                and model.context is not None
                and not model.drained,
            )
            for name, encoder in (("whitebox", source), ("model", model)):
                if encoder is None or name not in targets:
                    continue
                target = targets[name]
                last = self.applied_bitrates.get((name, id(encoder)))
                floor = min(
                    (
                        MODEL_ENCODER_MIN_BITRATE
                        if name == "model"
                        else SOURCE_ENCODER_MIN_BITRATE
                    ),
                    encoder.max_bitrate,
                )
                if (
                    last is not None
                    and last >= floor
                    and abs(target - last) < last * 0.1
                ):
                    continue
                if not await asyncio.to_thread(encoder.set_target_bitrate, target):
                    continue
                self.applied_bitrates = {
                    key: value
                    for key, value in self.applied_bitrates.items()
                    if key[0] != name
                }
                self.applied_bitrates[(name, id(encoder))] = target
                logging.getLogger("uvicorn.error").info(
                    "WHITEBOX_RTC_ENCODER_BUDGET %s",
                    json.dumps(
                        dict(
                            track=name,
                            receiver_bitrate=getattr(self, "receiver_bitrate", None)
                            or rate,
                            transport_target_bitrate=rate,
                            encoder_target_bitrate=target,
                            encoder_min_bitrate=floor,
                            encoder_max_bitrate=encoder.max_bitrate,
                            quality_floor_limited=sum(targets.values())
                            > int(
                                (getattr(self, "receiver_bitrate", None) or rate) * 0.9
                            ),
                        ),
                        separators=(",", ":"),
                    ),
                )

    def snapshot(self):
        sampled = time.monotonic() * 1000
        segment = self.owner.session.current_segment
        return {
            name: dict(
                media_stream_id=self.stream_ids[name],
                sample_mono_ms=sampled,
                produced_frames=(
                    getattr(segment, "raw_model_frames", None)
                    if name == "model"
                    else getattr(self.owner.session, "submitted", None)
                ),
                handoff_frames=self.sent[name],
                offered_frames=queue.offered,
                submitted_frames=queue.taken,
                accepted_frames=self.accepted[name],
                departed_frames=self.departed[name],
                failed_frames=self.failed[name],
                cancelled_frames=self.cancelled[name],
                inflight_frames=self._inflight(name),
                inflight_bytes=sum(
                    v[2] for v in self.handoffs.values() if v[0] == name
                ),
                handoff_budget_bytes=self._handoff_budget(),
                oldest_queued_ms=(
                    max(0.0, (queue.clock() - queue.items[0][0]) * 1000)
                    if queue.items
                    else 0.0
                ),
                queue_wait_ms_mean=(
                    queue.wait_ms_total / queue.taken if queue.taken else None
                ),
                queue_wait_ms_max=queue.wait_ms_max if queue.taken else None,
                handoff_ms_mean=(
                    self.handoff_ms_total[name] / self.departed[name]
                    if self.departed[name]
                    else None
                ),
                handoff_ms_max=(
                    self.handoff_ms_max[name] if self.departed[name] else None
                ),
                native_pacer_samples=self.pacer_samples[name],
                native_pacer_ms_mean=(
                    self.pacer_ms_total[name] / self.pacer_samples[name]
                    if self.pacer_samples[name]
                    else None
                ),
                native_pacer_ms_max=(
                    self.pacer_ms_max[name] if self.pacer_samples[name] else None
                ),
                queued_frames=len(queue.items),
                queued_bytes=queue.bytes,
                high_water_bytes=queue.high_water,
                dropped=copy.deepcopy(queue.drops),
                encoder_target_bitrate=next(
                    (
                        rate
                        for (track, _), rate in self.applied_bitrates.items()
                        if track == name
                    ),
                    None,
                ),
            )
            for name, queue in self.media.items()
        }

    def _record_delivery(self):
        now = time.monotonic()
        if self.closed or now - self.last_delivery_log >= 5:
            self.last_delivery_log = now
            logging.getLogger("uvicorn.error").info(
                "WHITEBOX_RTC_DELIVERY %s",
                json.dumps(
                    dict(
                        connection_generation=self.generation,
                        closed=self.closed,
                        reason=self.reason if self.closed else None,
                        transport_target_bitrate=self.target_bitrate,
                        receiver_bitrate=self.receiver_bitrate,
                        streams=self.snapshot(),
                    ),
                    separators=(",", ":"),
                ),
            )
        if self.report_delivery is not None:
            self.report_delivery.update(
                connection_generation=self.generation,
                closed=self.closed,
                reason=self.reason if self.closed else None,
                control_replay_total=self.replay_total,
                control_replay_read=self.replayed_controls,
                control_replay_remaining=len(self.replay),
                control_replay_bytes=self.replay_bytes,
                streams=self.snapshot(),
            )

    async def recv(self):
        while not self.closed:
            if self.control and (self.replay_bootstrap or not self.replay):
                at, kind, wire, _ = self.control.popleft()
                self.control_bytes -= len(wire)
                if self.replay_bootstrap:
                    self.replay_bootstrap -= 1
                if kind == "rtc.frame.meta" and time.monotonic() - at > 1:
                    continue
                return wire
            if self.replay:
                _, size, message = self.replay.popleft()
                self.replay_bytes -= size
                self.replayed_controls += 1
                # send_bytes can complete synchronously. Give capture, inputs
                # and native RTP service time during a long control replay.
                if self.replayed_controls % 32 == 0:
                    await asyncio.sleep(0)
                    if self.closed:
                        break
                return self._control_wire(message)
            if (
                self.model_eos
                and not self.media_failed
                and not self.media["model"].items
                and not self._inflight("model")
            ):
                message, self.model_eos = self.model_eos, None
                self.model_eos_sent = message["media_stream_id"]
                return msgspec.msgpack.encode(
                    dict(
                        message,
                        final_rtp_timestamp=self.last_rtp["model"],
                        final_offered_frame=self.last_offered["model"],
                        delivery=self.snapshot()["model"],
                    )
                )
            if (
                self.owner.terminal
                and not self.media_failed
                and not any(q.items for q in self.media.values())
                and not self.handoffs
            ):
                return msgspec.msgpack.encode(
                    dict(self.owner.terminal, rtc_delivery=self.snapshot())
                )
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), 0.1)
            except asyncio.TimeoutError:
                pass
        if getattr(self, "error", None) is not None:
            raise self.error
        if self.reason in {"INVALID_CONTROL_MESSAGE", "RTC_CONTROL_OVERFLOW"}:
            raise WhiteboxDiagnostic(self.reason, "control")
        raise WhiteboxDiagnostic("RTC_DISCONNECTED", "transport")
