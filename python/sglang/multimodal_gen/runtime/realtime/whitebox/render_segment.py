# SPDX-License-Identifier: Apache-2.0
"""A model execution child. It borrows source frames and never owns the CLI."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import os
import time
from dataclasses import replace
from itertools import islice

import msgspec

from ..source_prepare import V2_PROTOCOL
from .contracts import capture_pts
from .diagnostics import WhiteboxDiagnostic, public_error
from .interactive import InteractiveSession
from .playback import H264PlaybackEncoder
from .render_contract import RenderError, canonical
from .reports import SessionReport
from .sink_control import SinkControl
from .source_sampling import AdaptiveSourceSampler


class SegmentSource:
    def __init__(self, source):
        self.source = source
        self.stopped = asyncio.Event()
        self.first = None
        self.start_after = source.latest.version
        self.context_boundary = None
        self.sampler = (
            AdaptiveSourceSampler()
            if getattr(
                getattr(source, "config", None), "model_source_sampling", "latest"
            )
            == "adaptive"
            else None
        )
        self.sampling_fence = None

    @property
    def version(self):
        return self.first[0] - 1 if self.first is not None else self.start_after

    def has_capture(self, after):
        return self.context_boundary is None and (
            self.first is not None or self.source.latest.version > after
        )

    def select_start(self, boundary):
        if boundary == {"mode": "next_available"}:
            self.start_after = self.source.latest.version
        else:
            self.first = self.source.resolve_start(boundary)
            if self.first is None:
                raise RenderError("START_SOURCE_EXPIRED")

    async def get(self, after):
        if self.stopped.is_set():
            return None
        if self.first is not None:
            first, self.first = self.first, None
            self.sampling_fence = first[0]
            return first
        frame = asyncio.create_task(self.source.latest.get(after))
        stopped = asyncio.create_task(self.stopped.wait())
        try:
            done, _ = await asyncio.wait(
                (frame, stopped), return_when=asyncio.FIRST_COMPLETED
            )
            if stopped in done:
                return None
            result = frame.result()
            while result is not None and self.context_boundary is not None:
                from .source_conditions import source_reached

                if source_reached(result[1][0], self.context_boundary):
                    self.context_boundary = None
                    self.sampling_fence = result[0]
                    break
                # Keep the same stop fence while advancing past pre-boundary captures.
                frame = asyncio.create_task(self.source.latest.get(result[0]))
                done, _ = await asyncio.wait(
                    (frame, stopped), return_when=asyncio.FIRST_COMPLETED
                )
                if stopped in done:
                    return None
                result = frame.result()
            return result
        finally:
            for task in (frame, stopped):
                task.cancel()
            await asyncio.gather(frame, stopped, return_exceptions=True)

    def stop(self):
        self.stopped.set()

    async def wait_for_capture(self, after):
        """Observe availability without consuming a pinned/context boundary."""
        if self.first is not None or self.stopped.is_set():
            return
        capture = asyncio.create_task(self.source.latest.get(after))
        stopped = asyncio.create_task(self.stopped.wait())
        try:
            await asyncio.wait((capture, stopped), return_when=asyncio.FIRST_COMPLETED)
        finally:
            capture.cancel()
            stopped.cancel()
            await asyncio.gather(capture, stopped, return_exceptions=True)

    def select_model_source(self, latest, *, after, **demand):
        if self.sampler is None:
            return latest
        inputs = getattr(self.source, "sampling_inputs", None)
        control = dict(
            control_sequence=(
                inputs.revision,
                inputs.acknowledged(
                    (latest[1][0].observed_input or {}).get("receipts", [])
                ),
            )
            if inputs is not None
            else self.source.input_sequence,
            semantic_control=inputs is not None,
        )
        pinned = self.sampling_fence == latest[0]
        if pinned:
            self.sampler.reset()
            demand["startup"] = True
            self.sampling_fence = None
        demand["urgent"] = (
            demand["urgent"] or self.source.latest.closed or self.stopped.is_set()
        )
        if not pinned and not self.sampler.admit(
            latest,
            after=after,
            **control,
            **{key: value for key, value in demand.items() if key != "now"},
        ):
            return None
        return self.sampler.choose(
            latest,
            tuple(
                islice(
                    reversed(self.source.retained.frames), self.sampler.max_candidates
                )
            ),
            after=after,
            **control,
            **demand,
        )


class RenderSegment(InteractiveSession):
    render_segment = True

    def __init__(self, owner, state, references, leases):
        self.owner, self.segment = owner, copy.deepcopy(state)
        self.segment_id, self.segment_epoch = (
            state["segment_id"],
            state["segment_epoch"],
        )
        self.source_view = SegmentSource(owner)
        self.source_view.context_boundary = copy.deepcopy(state.get("source_context"))
        self.references = references
        self.model_ws = self.assignment = None
        self.model_ready = asyncio.Event()
        self.model_init_dispatched = False
        self.closed_event = asyncio.Event()
        self.first_decodable = False
        self.first_output_at = None
        self.cancel_requested = False
        self.publishing = True
        self.failure_code = None
        self.failure_diagnostic = {}
        self.archive_failure = False
        self.resource_state = "none"
        self.prompt_by_event = {
            None: dict(
                version=1,
                text=state["prompt"],
                sha256=state["prompt_sha256"],
                **(
                    {"area_binding": copy.deepcopy(state["area_binding"])}
                    if "area_binding" in state
                    else {}
                ),
            )
        }
        self.pending_prompt = None
        self.periodic_refresh = None
        self.sink = SinkControl(self)
        self.prompt_available = asyncio.Event()
        self.source_admission_ready = asyncio.Event()
        self.condition_switch = None
        self.clip_switch = False
        self.source_resume_after = 0
        self.resume_condition_evidence = None
        self.last_selected_source = None
        self.selected_frames = 0
        self.generation_task = self.render_task = self.output_task = None
        super().__init__(
            leases=leases,
            config=owner.config,
            archives=owner.archives,
            trace_id=state["segment_id"],
            session_id=state["segment_id"],
            user_id=owner.user_id,
            reports=owner.reports,
            report=SessionReport(state["segment_id"]) if owner.reports else None,
        )
        self.latest = self.source_view
        self.epoch = self.segment_epoch
        self.ready = asyncio.Event()
        self.preview_done.set()

    async def _deadline(self):
        await self.closed_event.wait()

    async def _room_heartbeat(self):
        await self.closed_event.wait()

    async def _status(self, state, **fields):
        if state == "active":
            return  # Protocol readiness and a decodable first frame are distinct.
        await self.owner.segment_status(self, "initializing", phase=state, **fields)

    async def model_protocol_ready(self, ws, assignment, ready):
        if ready.get("source_prompt_protocol") != "ordered-boundary-v2":
            raise RenderError("CAPABILITY_UNAVAILABLE")
        if getattr(self.leases, "bound_source", None) is not None:
            self.pinned_start = self.leases.bound_source
        if getattr(self, "pinned_start", None) is not None:
            self.latest.first = self.pinned_start
            self.pinned_start = None
        else:
            self.latest.select_start(self.segment["start_boundary"])
        self.model_ws, self.assignment = ws, assignment
        self.prompt_reference_updates = ready.get("prompt_reference_updates") is True
        self.clip_switch = (
            ready.get("source_condition_switch_protocol") == "clip-barrier-v1"
        )
        self.condition_early_release = (
            self.clip_switch and ready.get("source_condition_early_release") is True
        )
        self.model_reference_limit = ready.get("max_references", 9)
        large_revision = os.environ.get("WHITEBOX_LARGE_REFERENCE_MODEL_REVISION")
        if large_revision and self.assignment.denoiser.model_revision != large_revision:
            self.model_reference_limit = min(
                self.model_reference_limit,
                int(os.environ.get("WHITEBOX_DEFAULT_REFERENCE_CAPACITY", "3")),
            )
        self.model_reference_bytes = min(
            ready.get("max_reference_bytes", 8 << 20),
            ready.get("max_message_bytes", self.config.max_message_bytes) - (128 << 10),
        )
        self.active_reference_descriptors = copy.deepcopy(self.recording.references)
        self.prompt_by_event[None]["reference_descriptors"] = (
            self.active_reference_descriptors
        )
        self.resource_state = "reserved"
        self.sink.supported = ready.get("sink_updates") is True
        self.model_ready.set()
        await self.owner.segment_status(
            self,
            "initializing",
            phase="protocol_ready",
            sink={
                "supported": self.sink.supported,
                "enabled": None,
                "pending_request_id": None,
            },
            model_session_id=assignment.session_id,
            generation_id=assignment.generation_id,
            prompt_reference_updates=self.clip_switch
            and ready.get("prompt_reference_updates") is True,
            condition_switch_protocol=ready.get("source_condition_switch_protocol"),
            max_references=self.model_reference_limit,
        )
        if self.clip_switch:
            self.tasks.add(asyncio.create_task(self._condition_controls(ws)))
            policy = getattr(self.owner.host, "condition_refresh_policy", None)
            if policy is not None:
                from .condition_refresh import PeriodicConditionRefresh

                self.periodic_refresh = PeriodicConditionRefresh(self, policy)
                self.tasks.add(asyncio.create_task(self.periodic_refresh.run()))

    def model_source_allowed(self, sent):
        switch = self.condition_switch
        if switch is None:
            return True
        boundary = switch.get("boundary")
        if boundary is None:
            boundary = switch.get("provisional_boundary")
        return boundary is not None and sent < boundary

    def model_output_references(self, message, references):
        # model_output_condition has selected this output event's actual media.
        # A prompt-only update inherits those media, not the init reference list.
        return self.active_reference_descriptors

    async def _notify_source_admission(self):
        async with self.model_admission_condition:
            self.model_admission_condition.notify_all()

    async def _condition_controls(self, ws):
        try:
            await self.source_admission_ready.wait()
            while not self.closed_event.is_set():
                await self.prompt_available.wait()
                self.prompt_available.clear()
                await self.sink.idle.wait()
                current = self.owner.journal.snapshot.get("segment")
                if (
                    not current
                    or current["segment_id"] != self.segment_id
                    or current["state"] not in {"initializing", "streaming"}
                    or current["applied_version"] < 1
                    or self.pending_prompt is not None
                ):
                    continue
                queue = current["prompt_queue"]
                if not queue:
                    continue
                item = queue[0]
                if item["effective"].get("policy", "at_or_after") != "at_or_after":
                    raise RenderError("EFFECTIVE_BOUNDARY_UNAVAILABLE")
                effective = item["effective"]
                requested = effective.get(
                    "source_frame_index", effective.get("source_pts_us")
                )
                version = -1
                while True:
                    captured = await self.owner.latest.get(version)
                    if captured is None:
                        return
                    version, (frame, _) = captured
                    actual = (
                        frame.index
                        if "source_frame_index" in effective
                        else capture_pts(frame)
                    )
                    if item.get("source_context") is not None:
                        from .source_conditions import source_reached

                        if source_reached(frame, item["source_context"]):
                            break
                    elif actual >= requested:
                        break
                descriptors = self.active_reference_descriptors
                if "references" in item:
                    references = await self.owner.assets.resolve(item["references"])
                    if (
                        len(references) > self.model_reference_limit
                        or sum(len(ref["payload"]) for ref in references)
                        > self.model_reference_bytes
                    ):
                        result = await self.owner.journal.prompt_failed(
                            self.segment_id,
                            item["version"],
                            never_applied=True,
                            code="LIMIT_EXCEEDED",
                        )
                        await self.owner.operation_message(result)
                        self.prompt_available.set()
                        continue
                    # Confirm storage before the worker can use new images. Text-only
                    # changes retain the applied paths, including initial numbered files.
                    async with asyncio.timeout(self.config.startup_seconds):
                        descriptors = await self.recording.add_references(
                            references, wait=True
                        )
                candidate = {
                    **item,
                    "strict": False,
                    "reference_descriptors": descriptors,
                }
                loop = asyncio.get_running_loop()
                switch = dict(
                    accepted=loop.create_future(),
                    ready=loop.create_future(),
                    boundary=None,
                    started_at=time.perf_counter(),
                )
                async with self.source_send_lock:
                    current = self.owner.journal.snapshot.get("segment")
                    if (
                        self.source_view.stopped.is_set()
                        or self.cancel_requested
                        or not current
                        or current["segment_id"] != self.segment_id
                        or current["state"] not in {"initializing", "streaming"}
                    ):
                        return
                    if item.get(
                        "periodic_refresh"
                    ) and not await self.owner.journal.mark_periodic_dispatch(
                        self.segment_id, item["request_id"]
                    ):
                        self.prompt_available.set()
                        continue
                    if getattr(self, "condition_early_release", False):
                        watermark = max(self.selected_frames, 17)
                        switch["provisional_boundary"] = ((watermark + 16) // 17) * 17
                    self.condition_switch = switch

                    def remember(event_id):
                        candidate["event_id"] = event_id
                        self.pending_prompt = candidate
                        self.prompt_by_event[event_id] = candidate

                    fields = dict(
                        text=item["text"],
                        version=item["version"],
                        min_source_frame=self.selected_frames,
                        policy="at_or_after",
                        source_policy="skip_during_switch",
                    )
                    if "references" in item:
                        fields["references"] = references
                    await self.send_model_event(
                        ws, "prompt.update", fields, before_send=remember
                    )
                # The worker reserves B, then receives only the remaining old clip.
                async with asyncio.timeout(self.config.startup_seconds):
                    accepted = await switch["accepted"]
                    if accepted.get("state") == "accepted":
                        boundary = accepted.get("source_frame_index")
                        if (
                            type(boundary) is not int
                            or boundary < self.selected_frames
                            or boundary % 17
                        ):
                            raise RenderError("CONDITION_INVALID")
                        switch["boundary"] = boundary
                        await self._notify_source_admission()
                        ready = await switch["ready"]
                        if (
                            ready.get("state") in {"kv_ready", "source_ready"}
                            and ready.get("prompt_version") != item["version"]
                        ):
                            raise RenderError("REVISION_CONFLICT")
                # Sampling resumes after a fresh capture; never replay frames from the pause.
                async with self.source_send_lock:
                    self.source_resume_after = self.owner.latest.version
                    self.resume_condition_evidence = dict(
                        prompt_version=item["version"],
                        boundary=switch["boundary"],
                        previous_source=self.last_selected_source,
                        skipped_captures=max(
                            0,
                            self.source_resume_after
                            - getattr(
                                self,
                                "last_selected_capture_version",
                                self.source_resume_after,
                            ),
                        ),
                        switch_host_ms=(time.perf_counter() - switch["started_at"])
                        * 1000,
                        kv_ready=accepted.get("state") == "accepted"
                        and ready.get("state") == "kv_ready",
                        source_ready=accepted.get("state") == "accepted"
                        and ready.get("state") == "source_ready",
                    )
                    self.condition_switch = None
                await self._notify_source_admission()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logging.getLogger(__name__).exception(
                "Condition switch failed for segment %s", self.segment_id
            )
            self.failure_code = "CONDITION_SWITCH_FAILED"
            self.failure_diagnostic = public_error(
                error, self.trace_id, default="CONDITION_SWITCH_FAILED", stage="model"
            )
            self.failure_code = self.failure_diagnostic["code"]
            if self.generation_task is not None:
                self.generation_task.cancel()

    def model_presentation_frame(self, frame, local):
        if local != 0 or not self.segment.get("context_id"):
            return frame
        # F0 remains the actual capture used by prefill. Place its video sample
        # near resume, one source interval before fresh F1. Do not relabel its
        # capture time, input receipts, UI source clock or full-source recording.
        shift = max(0, self.prefill_hold_us - self.config.spec.pts(1))
        return replace(
            frame,
            provenance={
                **(frame.provenance or {}),
                "model_presentation_pts_us": capture_pts(frame) + shift,
                "prefill_hold_us": self.prefill_hold_us,
            },
        )

    async def before_model_source(self, ws, frame, local):
        context_id = self.segment.get("context_id")
        if local == 0 and context_id is not None:
            prepared = self.owner.control_prefill.resolve(context_id)
            if frame.index != prepared[1][0].index or frame.rgb != prepared[1][0].rgb:
                raise RenderError("PREFILL_INPUT_MISMATCH")
            await self.owner.control_prefill.resume(context_id)
            self.prefill_hold_us = round(self.owner.control_prefill.paused_ms * 1000)
        if getattr(self.leases, "is_startup", False) and local == 0:
            # V2's independently sized condition preview is not a video frame.
            # Bind video zero to the pinned original CLI frame instead.
            matches = (
                hashlib.sha256(frame.rgb).hexdigest() == self.leases.cli_rgb_sha256
                if getattr(self.leases, "protocol", None) == V2_PROTOCOL
                else frame.rgb == self.leases.rgb
            )
            if not matches:
                raise RenderError("SOURCE_EPOCH_MISMATCH")
        if self.resume_condition_evidence is not None:
            evidence = self.resume_condition_evidence
            self.resume_condition_evidence = None
            evidence.update(
                resumed_source_frame=frame.index,
                resumed_source_pts_us=capture_pts(frame),
                model_source_frame=local,
            )
            self.recording.manifest.setdefault("condition_switches", []).append(
                evidence
            )
            await self.recording.configure()
        self.last_selected_source = dict(
            source_frame_index=frame.index, source_pts_us=capture_pts(frame)
        )
        if local == 0:
            evidence = dict(
                source_frame_index=frame.index,
                source_pts_us=capture_pts(frame),
                start_frame_sha256=hashlib.sha256(frame.rgb).hexdigest(),
            )
            self.recording.manifest["start_boundary_actual"] = evidence
            self.recording.manifest["start_boundary_requested"] = self.segment[
                "start_boundary"
            ]
            self.recording.manifest["model_identity"] = dict(
                model_session_id=self.assignment.session_id,
                generation_id=self.assignment.generation_id,
                worker_id=self.assignment.denoiser.worker_id,
                worker_epoch=getattr(self.assignment.denoiser, "worker_epoch", None),
            )
            await self.recording.configure()
            await self.owner.segment_start_frame(self, evidence)
        self.selected_frames = local + 1
        self.submitted = self.owner.submitted
        await self.owner.before_segment_source(self, ws, frame, local)

    async def model_control(self, message):
        if await self.sink.control(message):
            return True
        if message.get("type") == "model.cache":
            from .cache_receipt import public_cache_receipt

            usage = message.get("usage")
            cache = public_cache_receipt(
                usage.get("zing_block0_cache") if isinstance(usage, dict) else None
            )
            if (
                cache is not None
                and self.assignment is not None
                and message.get("model_session_id") == self.assignment.session_id
                and message.get("generation_id") == self.assignment.generation_id
            ):
                await self.owner.segment_cache_receipt(self, cache)
            return True
        return await self.owner.segment_model_control(self, message)

    async def model_output_condition(self, message, pending, selection, rendering):
        if self.first_output_at is None:
            self.first_output_at = time.perf_counter()
        sink = await self.sink.output(message)
        condition = self.prompt_by_event.get(message.get("event_id"))
        if condition is None:
            raise RenderError("CONDITION_INVALID")
        self.active_reference_descriptors = condition.get(
            "reference_descriptors", self.active_reference_descriptors
        )
        first = pending[message["output_frame_start"]][0]
        if "area_binding" in condition:
            condition.setdefault(
                "effective_source",
                dict(source_frame_index=first.index, source_pts_us=capture_pts(first)),
            )
        await self.owner.segment_prompt_output(self, condition, message, first)
        if self.periodic_refresh is not None:
            self.periodic_refresh.observe(message)
        # Output provenance is ordered. Once a newer version is visible, older
        # text can no longer appear in a subsequent block (including decoder
        # tail frames); the archive has already received its full snapshots.
        self.prompt_by_event = {
            key: value
            for key, value in self.prompt_by_event.items()
            if value["version"] >= condition["version"]
        }
        return condition["text"], dict(
            rendering,
            segment_id=self.segment_id,
            segment_epoch=self.segment_epoch,
            **({"sink": sink} if sink is not None else {}),
            prompt_version=condition["version"],
            prompt_sha256=condition["sha256"],
            **(
                {
                    "area_condition": dict(
                        binding=copy.deepcopy(condition["area_binding"]),
                        event_id=message.get("event_id"),
                        prompt_version=condition["version"],
                        prompt_sha256=condition["sha256"],
                        reference_sha256=hashlib.sha256(
                            canonical(self.active_reference_descriptors)
                        ).hexdigest(),
                        seed=self.segment["seed"],
                        effective_source=copy.deepcopy(condition["effective_source"]),
                    )
                }
                if "area_binding" in condition
                else {}
            ),
        )

    def pairing_age_budget(self, captured=None):
        # The first request can compile kernels and encode its references. Its
        # startup inputs retain the advertised 120 s budget even while the first
        # decoded batch drains; subsequent captures use the normal steady limit.
        if self.first_output_at is None or (
            captured is not None and captured < self.first_output_at
        ):
            return max(
                390 if self.segment.get("context_id") else 120,
                self.config.pairing_max_age_seconds,
            )
        return self.config.pairing_max_age_seconds

    async def _media(
        self, stream, index, frame, encoded, captured, recording_index=None
    ):
        if not self.publishing:
            return
        if stream != "model":
            raise ValueError("A model segment cannot publish source preview")
        await super()._media(stream, index, frame, encoded, captured, recording_index)
        if not self.first_decodable:
            if not encoded.get("keyframe"):
                raise ValueError("A new segment must start with a keyframe")
            self.first_decodable = True
            await self.owner.segment_status(
                self,
                "streaming",
                phase="first_decodable",
                first_decodable=True,
                source_frame_index=frame.index,
                source_pts_us=capture_pts(frame),
                output_frame=recording_index,
            )

    async def _pump(self):
        while True:
            message = msgspec.msgpack.decode(await self.outgoing.get())
            if message.get("type") == "session.completed":
                return
            if self.publishing:
                message.update(
                    segment_id=self.segment_id,
                    segment_epoch=self.segment_epoch,
                    media_stream_id=self.segment_epoch,
                    source_session_id=self.owner.session_id,
                    source_epoch=self.owner.source_epoch,
                )
                await self.owner.segment_message(self, message)

    async def run(self):
        state = self.segment
        spec = self.config.spec
        outcome = "failed"
        failure_default, failure_stage = "ARCHIVE_FAILED", "archive"
        try:
            if self.source_view.stopped.is_set():
                outcome = "cancelled"
                return
            self.recording = await self.archives.create_recording(
                directory=self.config.spool / self.segment_id,
                spec=spec,
                **(
                    {"model_spec": self.model_geometry.output}
                    if self.config.model_target is not None
                    else {}
                ),
                world_id=self.owner.world_id,
                trace_id=self.trace_id,
                inference=dict(
                    prompt=state["prompt"],
                    seed=state["seed"],
                    deployment=self.config.model_metadata,
                ),
                ui=self.owner.renderer.metadata,
                references=self.references,
                object_prefix=self.owner.recording.manifest["object_prefix"]
                + "segments/"
                + self.segment_id,
                **(
                    {"archive_encoding": self.config.archive_encoding}
                    if self.config.independent_archive
                    and getattr(self.archives, "supports_rgb", False)
                    else {}
                ),
            )
            if getattr(self.recording, "raw_archive", False):
                await self.recording.prepare_raw()
            failure_default, failure_stage = "VIDEO_ENCODING_FAILED", "encoding"
            self.playback = H264PlaybackEncoder(
                self.model_geometry.output,
                max_bitrate=self.config.playback_max_bitrate,
                gop=self.config.playback_gop,
                capture_clock=self.config.capture_clock,
                immediate_capture=self.config.immediate_capture,
                capture_fps=(
                    self.model_geometry.output.fps
                    if self.config.model_target is not None
                    else self.owner.capture_rate_hint()
                ),
            )
            self.record_source = (
                None
                if getattr(self.recording, "raw_archive", False)
                else H264PlaybackEncoder(
                    spec,
                    max_bitrate=self.config.source_max_bitrate,
                    gop=self.config.playback_gop,
                    capture_clock=self.config.capture_clock,
                    capture_fps=self.owner.capture_rate_hint(),
                )
            )
            if self.config.model_target is not None:
                await self.owner.outgoing.put(
                    msgspec.msgpack.encode(
                        dict(
                            version=2,
                            type="render.stream",
                            segment_id=self.segment_id,
                            segment_epoch=self.segment_epoch,
                            media_stream_id=self.segment_epoch,
                            render_revision=self.segment["render_revision"],
                            video_stream=self.playback.stream,
                            **self.owner.source_identity(),
                        )
                    )
                )
            self.playback.independent_frames = self.config.playback_gop == 1
            if self.record_source:
                self.record_source.independent_frames = self.config.playback_gop == 1
            self.browser_ui = True
            self.recording.manifest.update(
                model_geometry=self.model_geometry.receipt(),
                recording_kind="render_segment",
                dataset_status="excluded_unvalidated",
                source_session_id=self.owner.session_id,
                source_epoch=self.owner.source_epoch,
                segment_id=self.segment_id,
                segment_epoch=self.segment_epoch,
                **(
                    {"context_id": state["context_id"]}
                    if state.get("context_id")
                    else {}
                ),
                references_mapping=state["references"],
                rendering=dict(
                    protocol="whitebox-render-v2",
                    model_video="render_segment",
                    live_source="independent_source",
                    source_frames_captured=0,
                ),
                video_streams=(
                    {}
                    if getattr(self.recording, "raw_archive", False)
                    else dict(
                        whitebox=self.record_source.stream, model=self.playback.stream
                    )
                ),
            )
            if getattr(self.leases, "is_startup", False):
                self.recording.manifest["startup"] = self.leases.receipt()
            failure_default, failure_stage = "ARCHIVE_FAILED", "archive"
            await self.recording.configure()
            if self.source_view.stopped.is_set():
                outcome = "cancelled"
                return
            if self.reports and self.report:
                self.reports.attach(self.report, self.recording)
            self.upstream.init = dict(
                self.owner.upstream.init,
                prompt=state["prompt"],
                references=self.references,
                seed=state["seed"],
                source_prompt_protocol="ordered-boundary-v2",
            )
            if getattr(self.leases, "supports_condition_barrier", True):
                self.upstream.init["source_condition_switch_protocol"] = (
                    "clip-barrier-v1"
                )
                self.upstream.init["source_condition_early_release"] = True
            else:
                self.upstream.init.pop("source_condition_switch_protocol", None)
                self.upstream.init.pop("source_condition_early_release", None)
            self.ready.set()
            self.output_task = asyncio.create_task(self._pump())
            self.render_task = asyncio.create_task(self._render())
            self.generation_task = asyncio.create_task(
                self._generation(
                    state["render_revision"],
                    dict(
                        mode="model", prompt=state["prompt"], references=self.references
                    ),
                    self.segment_epoch,
                )
            )

            async def startup_deadline():
                await asyncio.sleep(120)
                if not self.first_decodable:
                    self.failure_code = "MODEL_STARTUP_TIMEOUT"
                    self.generation_task.cancel()

            self.tasks.add(asyncio.create_task(startup_deadline()))
            done, _ = await asyncio.wait(
                (self.generation_task, self.render_task, self.output_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                failure_default, failure_stage = "RENDER_FAILED", "rendering"
                if task in (self.render_task, self.output_task):
                    failure_default = "RENDER_OUTPUT_FAILED"
                task.result()
            failure_default, failure_stage = "RENDER_FAILED", "rendering"
            await self.generation_task
            await self.render_queue.put(None)
            failure_default = "RENDER_OUTPUT_FAILED"
            await asyncio.wait_for(self.render_task, self.config.drain_seconds)
            await self.output_task
            outcome = "completed"
        except asyncio.CancelledError:
            outcome = "failed" if self.failure_code else "cancelled"
            if self.failure_code and not self.failure_diagnostic:
                self.failure_diagnostic = public_error(
                    WhiteboxDiagnostic(self.failure_code, "model"), self.trace_id
                )
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "Render segment failed: %s", self.segment_id
            )
            self.failure_diagnostic = public_error(
                exc, self.trace_id, default=failure_default, stage=failure_stage
            )
            self.failure_code = self.failure_diagnostic["code"]
            outcome = "failed"
        finally:
            self.publishing = False
            self.source_view.stop()
            for task in (self.generation_task, self.render_task, self.output_task):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(
                    t
                    for t in (self.generation_task, self.render_task, self.output_task)
                    if t
                ),
                return_exceptions=True,
            )
            self.resource_state = getattr(
                self.leases, "resource_state", "release_unknown"
            )
            # Cancellation carries the primary timeout/switch verdict separately
            # from the lease context. Read cleanup only after that context drained.
            if getattr(self.leases, "cleanup_code", None) == "RELEASE_UNKNOWN":
                self.failure_diagnostic["cleanup_code"] = "RELEASE_UNKNOWN"
            if self.recording:
                if getattr(self.leases, "is_startup", False):
                    self.recording.manifest["startup"] = self.leases.receipt()
                self.archive_failure = bool(getattr(self.recording, "failure", None))
                try:
                    self.recording.manifest["rendering"].update(
                        source_frames_captured=self.owner.submitted,
                        selected_source_frames=self.selected_frames,
                        model_frames=self.received,
                    )
                    await self.recording.configure()
                    if self.report:
                        self.report.gateway_done.set()
                    self.archives.finalize(self.recording, outcome, report=self.report)
                except Exception:
                    self.archive_failure = True
                    if self.report:
                        self.report.close()
            elif self.report:
                self.report.close()
            for encoder in (self.playback, self.record_source):
                if encoder:
                    await asyncio.to_thread(encoder.close)
            for task in self.tasks:
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.pairs.clear()
            try:
                await self.owner.segment_finished(self, outcome, self.failure_code)
            finally:
                self.closed_event.set()

    async def stop(self, *, cancel=False):
        self.source_view.stop()
        if hasattr(self, "model_admission_condition"):
            await self._notify_source_admission()
        self.cancel_requested = cancel
        if cancel or not self.selected_frames:
            self.publishing = False
            if self.generation_task:
                self.generation_task.cancel()
        await asyncio.wait_for(
            asyncio.shield(self.closed_event.wait()), self.config.drain_seconds + 10
        )

    async def retry_release(self):
        if not self.closed_event.is_set():
            raise RenderError("RELEASE_UNKNOWN")
        retry = getattr(self.leases, "retry_release", None)
        if retry is None:
            raise RenderError("RELEASE_UNKNOWN")
        await retry()
        self.resource_state = self.leases.resource_state
        current = self.owner.journal.snapshot["segment"]
        if not current or current["segment_id"] != self.segment_id:
            raise RenderError("SEGMENT_EPOCH_MISMATCH")
        await self.owner.segment_status(self, current["state"])
