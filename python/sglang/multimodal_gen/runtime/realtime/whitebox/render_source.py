# SPDX-License-Identifier: Apache-2.0
"""One authoritative source with independently replaceable model children."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import os
import time
from dataclasses import replace
from uuid import uuid4

import msgspec

from .area_contract import AREA_PROTOCOL
from .contracts import capture_pts
from .interactive import InteractiveSession, LatestFrame
from .model_geometry import validate_reference_geometry
from .opening_presentation import CONFIRM as CONFIRM_MODEL_PRESENTATION
from .opening_presentation import PROTOCOL as FIRST_PRESENTED_PROTOCOL
from .project_audio import public_ui_metadata
from .render_assets import SourceAssets, SourceFrames
from .render_contract import (
    MAX_REFERENCE_BYTES,
    MAX_REFERENCE_IMAGES,
    PROTOCOL,
    RenderError,
    area_binding,
    canonical,
    identity,
    prompt_text,
    reference_parameters,
    segment_parameters,
)
from .render_segment import RenderSegment


def configured_render_play_seconds():
    """Freeze the configured regional render budget at session creation."""
    seconds = int(os.environ.get("WHITEBOX_RENDER_PLAY_SECONDS", "300"))
    if not 1 <= seconds <= 7200:
        raise ValueError("WHITEBOX_RENDER_PLAY_SECONDS must be between 1 and 7200")
    return seconds


class SourceLatest(LatestFrame):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    async def put(self, value):
        frame, captured = value
        frame = replace(
            frame,
            provenance={**(frame.provenance or {}), **self.owner.source_identity()},
        )
        self.owner.retained.append(self.version + 1, frame, captured)
        prefill = getattr(self.owner, "control_prefill", None)
        if prefill is not None:
            prefill.observe(self.version + 1, frame, captured)
        if getattr(self.owner, "startup", None) is not None:
            # SourceFrames reserves 8 MiB for one pinned model start. Do not
            # block capture, preview, inputs or another player's world here.
            self.owner.startup.observe_source(self.version + 1, frame, captured)
        await super().put((frame, captured))


class SourceSession(InteractiveSession):
    source_only = True
    defer_startup_until_source = True
    startup = None
    model_binding = control_prefill = None
    close_requested = False
    player_finish_task = _source_close_task = None
    first_presented_required = False
    first_presented_player_ready = None

    def __init__(self, host, journal, *, source_epoch, **kwargs):
        self.host, self.journal, self.source_epoch = host, journal, source_epoch
        self.retained = SourceFrames()
        self.current_segment = self.segment_task = None
        self.segment_lock = asyncio.Lock()
        self.binding_ended = False
        self.bound_at = time.monotonic()
        self.play_elapsed = self.wait_elapsed = 0.0
        self.last_tick = time.monotonic()
        self.source_state_value = "binding"
        self.assets = None
        self.initial_assets = []
        self.control_tasks = set()
        self.pending_end = None
        self.request_cache = {}
        self.segment_snapshots = {}
        self.current_model_stream = None
        self.control_tokens = 4.0
        self.control_token_time = time.monotonic()
        self.area_state = None
        self.resolve_area_offer = None
        self.resolve_first_presentation_offer = None
        self.protocol = PROTOCOL
        self.source_conditions = None
        self.condition_debug = None
        self.context_channel = None
        self.context_seed = 42
        self.context_startup_request_id = None
        self.startup = None
        self.model_binding = self.control_prefill = None
        super().__init__(leases=None, **kwargs)
        self.latest = SourceLatest(self)
        self._sdk_create_started = False
        self._sdk_create_done = asyncio.Event()
        self._sdk_create_done.set()
        self._sdk_retire_task = self._source_close_task = None
        self.close_requested = False
        self.player_finish_task = None
        from .source_pause import SourcePause

        self.source_pause = SourcePause(self)

    @property
    def binding_lifetime_seconds(self):
        binding = self.model_binding
        if binding is not None and binding.assignment is not None:
            assignment = binding.assignment
            if (
                assignment.player_expires_at is not None
                and assignment.admitted_at is not None
            ):
                return assignment.player_expires_at - assignment.admitted_at
        # Legacy non-bound protocol only.
        return max(1800, self.config.play_seconds + 600)

    def source_identity(self):
        return dict(
            source_session_id=self.session_id,
            source_id=self.session_id,
            source_epoch=self.source_epoch,
            view_id=getattr(self.engine, "stream_id", None),
        )

    async def _start(self, message):
        self.configure_resolution(message)
        from .inference_control import SourceInference

        self.inference = SourceInference(self, message)
        self.member_prefill = None
        profile = getattr(self.config.engine_factory, "member_prefill", None)
        if profile is not None:
            from .member_prefill import MemberPrefill, validate_profile

            validate_profile(profile)
            if not self.inference.manual or self.inference.assets is None:
                raise RenderError("CONDITION_INVALID")
            self.member_prefill = MemberPrefill(self)
        await self.journal.initialize_inference(
            manual=self.inference.manual and self.member_prefill is None
        )
        # Ordinary manual preview owns only the live SDK source. Explicit
        # enable obtains its parent Slot; member-prefill already requests compute.
        if not self.inference.manual or self.member_prefill is not None:
            self.start_model_binding(message)
        if self.member_prefill is not None and self.control_prefill is None:
            raise RenderError("CAPABILITY_UNAVAILABLE")
        if message.get("startup") is not None and not self.inference.manual:
            from .startup import PlayStartup

            if os.environ.get("WHITEBOX_MODEL_STARTUP_PREPARE") != "1":
                raise RenderError("CAPABILITY_UNAVAILABLE")
            if message.get("whitebox", {}).get("loading") is not None:
                raise RenderError("CAPABILITY_UNAVAILABLE")
            if message.get("references") not in (None, []):
                raise RenderError("CONDITION_INVALID")
            self.startup = PlayStartup(self, message["startup"])
        world = message["whitebox"]
        self.world_id = (
            world.get("source_id", "") + "/" if world.get("source_id") else ""
        ) + world["world_id"]
        self.initial_message = copy.deepcopy(message)
        await super()._start(
            {
                **message,
                "prompt": message.get("prompt") or "Source preview",
                "render_mode": "whitebox",
            }
        )

    def start_model_binding(self, message):
        if os.environ.get("WHITEBOX_BOUND_COMPUTE") == "1":
            from .control_prefill import ControlPrefill
            from .model_binding import ModelBinding

            prebound = getattr(self.config.engine_factory, "model_binding", None)
            lease = (
                prebound.lease
                if prebound is not None
                else self.host.make_segment_leases(
                    self.user_id,
                    dict(
                        segment_id=self.session_id,
                        model_size=f"{self.model_geometry.model.width}x{self.model_geometry.model.height}",
                        model_session_id=uuid4().hex,
                        generation_id=uuid4().hex,
                        capacity_wait_ms=0,
                        references=message.get("references")
                        or (message.get("startup") or {})
                        .get("initial_render", {})
                        .get("references", []),
                    ),
                )
            )
            lease.cleanup_timeout_s = 90
            lease.sdk_cleanup_provider = self._sdk_cleanup_for
            self.control_prefill = ControlPrefill(self)
            self.model_binding = prebound or ModelBinding(
                lease, self.control_prefill.receipt, timeout=self.config.startup_seconds
            )
            self.model_binding.on_receipt = self.control_prefill.receipt
            self.model_binding.start()

    async def source_model_ready(self, init):
        """A declared world condition source owns init, including member sources.

        Source-first admission is necessary: the descriptor's pinned capability
        is unavailable before opening the source, and a manual preparation must
        never race the first complete world-context commit.
        """
        if getattr(self.engine, "metadata", {}).get("project_input"):
            self.sdk_input_session = await self.engine.player_input_session()
            if not isinstance(self.sdk_input_session.get("input"), dict):
                raise RenderError("CAPABILITY_UNAVAILABLE")
        capability = getattr(self.engine, "model_context_capability", None)
        if capability is None:
            capability = getattr(self.engine, "descriptor", {}).get("modelContext")
        if capability is not None:
            if self.member_prefill is not None:
                raise RenderError("CONDITION_INVALID")
            from .source_conditions import validate_capability

            validate_capability(capability)
            self.first_presented_required = (
                capability.get("firstPresentedProtocol") == FIRST_PRESENTED_PROTOCOL
            )
            self.context_seed = (
                self.inference.assets.seed
                if getattr(self, "inference", None) and self.inference.assets
                else init.get("seed", 42)
            )
            if self.startup is not None:
                if self.startup.task is not None:
                    raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
                self.context_startup_ids = copy.deepcopy(self.startup.ids)
                # Launch intent and deterministic seed outlive preparation;
                # the World owns prompt/images and the journal alone owns init.
                self.context_startup_request_id = self.startup.request["request_id"]
                self.context_seed = self.startup.assets.seed
                self.display_mode = "model"
                await self.startup.close()
                self.startup = None
            self.world_driven = True
            # These belonged to the legacy startup request, not the fixed world
            # configuration. Only the condition's scoped handles supply images.
            self.initial_message["references"] = []
            return
        if self.startup is not None and not getattr(
            getattr(self, "inference", None), "manual", False
        ):
            self.startup.launch(init)
            await self.startup.wait(self.startup.admitted)

    async def _open_engine(self, world_id, world, init):
        if self.model_binding is not None:
            await self.model_binding.wait_ready()
        if getattr(self, "closing", False):
            raise RenderError("PLAYER_EXPIRED")
        self._sdk_create_started = True
        self._sdk_create_done.clear()
        try:
            await super()._open_engine(world_id, world, init)
        finally:
            self._sdk_create_done.set()

    async def _sdk_cleanup_for(self, assignment):
        from ..player_lifecycle import sdk_cleanup_receipt

        self.closing = True
        if self._sdk_retire_task is not None and self._sdk_retire_task.done():
            if (
                self._sdk_retire_task.cancelled()
                or self._sdk_retire_task.exception() is not None
            ):
                self._sdk_retire_task = None  # Retry only this original SDK owner.
        if self._sdk_retire_task is None:

            async def retire():
                await self._sdk_create_done.wait()
                if self.engine is None:
                    factory = self.config.engine_factory
                    if (
                        getattr(factory, "model_binding", None) is self.model_binding
                        and getattr(factory, "record", None) is not None
                    ):
                        return await factory.runtime.retire_member(
                            factory.record["stream_id"]
                        )
                    if self._sdk_create_started:
                        retire_failed_start = getattr(
                            factory, "retire_failed_start", None
                        )
                        if retire_failed_start is not None:
                            return await retire_failed_start()
                        # A timed-out join may have committed. Absence isn't a
                        # positive terminal receipt; retain/quarantine the Slot.
                        raise RenderError("RELEASE_UNKNOWN")
                    return dict(
                        state="not_created",
                        source_id=self.session_id,
                        owner_epoch=self.source_epoch,
                    )
                if not hasattr(self.engine, "retire"):
                    raise RenderError("RELEASE_UNKNOWN")
                return await self.engine.retire()

            self._sdk_retire_task = asyncio.create_task(retire())
        receipt = await asyncio.shield(self._sdk_retire_task)
        return sdk_cleanup_receipt(assignment, **receipt)

    @property
    def player_remaining_seconds(self):
        binding = self.model_binding
        if binding is not None and binding.assignment is not None:
            deadline = binding.assignment.player_expires_at
            if deadline is not None:
                return max(0, deadline - time.time())
        return max(0, self.config.play_seconds - self.play_elapsed)

    def segment_leases(self, segment):
        if self.model_binding is not None:
            return self.model_binding.borrow(segment)
        return self.host.make_segment_leases(self.user_id, segment)

    async def _read_h3(self):
        if not self.browser_ui:
            raise RenderError("CAPABILITY_UNAVAILABLE")
        if self.playback:
            await asyncio.to_thread(self.playback.close)
            self.playback = None
        self.assets = SourceAssets(
            self.recording.directory,
            memory=getattr(self.recording, "diskless", False),
        )
        for ref in self.initial_message.get("references", []):
            asset = await self.assets.upload(ref["format"], ref["payload"])
            self.initial_assets.append(
                dict(id=ref["id"], asset_id=asset["asset_id"], role="scene")
            )
        if getattr(self, "inference", None) and self.inference.assets is not None:
            async with asyncio.timeout(self.config.startup_seconds):
                await self.inference.assets.committed.wait()
            for ref in self.inference.assets.references:
                asset = await self.assets.upload(ref["format"], ref["payload"])
                self.initial_assets.append(
                    dict(id=ref["id"], asset_id=asset["asset_id"], role="scene")
                )
        if self.startup is not None:
            async with asyncio.timeout(
                max(0, self.startup.deadline - time.monotonic())
            ):
                await self.startup.assets.committed.wait()
            for ref in self.startup.assets.references:
                asset = await self.assets.upload(ref["format"], ref["payload"])
                self.initial_assets.append(
                    dict(id=ref["id"], asset_id=asset["asset_id"], role="scene")
                )
        self.recording.manifest.update(
            recording_kind="source",
            source_session_id=self.session_id,
            dataset_status="excluded_unvalidated",
            source_epoch=self.source_epoch,
            rendering=dict(
                protocol=PROTOCOL,
                live_source="independent_latest",
                model_video="separate_segments",
            ),
            video_streams=(
                {}
                if getattr(self.recording, "raw_archive", False)
                else {"whitebox": self.source_playback.stream}
            ),
            render_segments=[],
        )
        self.recording.manifest.pop("client_ui", None)
        entry = self.host.runtime.players.get(self.engine.stream_id)
        if entry is not None:
            entry.update(explicit_source_owner=True, owner_idle_seconds=60)
            # Source readiness does not prove that the CLI produced a frame.
            # RoomRuntime.next_frame arms input inactivity after validation.
        await self.recording.configure()
        self.source_state_value = "ready"
        self.last_tick = time.monotonic()
        await self.journal.update(
            lambda state: state.update(
                source_state="ready", source_identity=self.source_identity()
            )
        )
        model_startup = (
            self.startup is not None or self.context_startup_request_id is not None
        )
        if model_startup:
            self.display_mode = "model"
            self.revision = 1
            await self.journal.update(
                lambda state: state.update(display_mode="model", display_revision=1)
            )
        ready = dict(
            version=2,
            type="session.ready",
            control_protocol=PROTOCOL,
            session_id=self.session_id,
            **self.source_identity(),
            play_seconds=self.config.play_seconds,
            format="h264",
            video_stream=None,
            source_video_stream=self.source_playback.stream,
            **(
                {"sdk_input_session": self.sdk_input_session}
                if getattr(self, "sdk_input_session", None) is not None
                else {}
            ),
            **(
                {
                    "model_output": dict(
                        width=self.model_geometry.output.width,
                        height=self.model_geometry.output.height,
                        fps=self.model_geometry.output.fps,
                    )
                }
                if self.config.model_target is not None
                else {}
            ),
            playback_flow_control="whitebox-dual-v2",
            timeline="source_capture_vfr",
            independent_streams=True,
            continuous_model=False,
            world_driven=getattr(self, "world_driven", False),
            model_parameters_mutable=True,
            recording_id=self.recording.directory.name,
            report_token=self.report.token if self.report else None,
            **(
                {"project_audio": self.engine.metadata["project_audio"]}
                if getattr(self.engine, "metadata", {}).get("project_audio")
                else {}
            ),
            engine=getattr(self.engine, "metadata", {}),
            ui=public_ui_metadata(self.renderer.metadata),
            object_prefix=self.recording.manifest["object_prefix"],
            source_stream=self.upstream.init["source_stream"],
            initial_references=self.initial_assets,
            capabilities=dict(
                protocol=PROTOCOL,
                render_segments=True,
                source_pause=self.source_pause.supported,
                model_inference_control=self.inference.manual
                and self.member_prefill is None,
                model_context_instruction=self.inference.manual
                and self.member_prefill is None,
                audio_output=False,
                independent_first_frame=False,
                control_prefill_protocol=(
                    "whitebox-control-prefill-v1" if self.control_prefill else None
                ),
                prefill_max_requests=256 if self.control_prefill else 0,
                prefill_parallelism=1 if self.control_prefill else 0,
                model_binding=(
                    dict(
                        worker_epoch=self.model_binding.assignment.denoiser.worker_epoch
                    )
                    if self.model_binding
                    else None
                ),
                reference_semantics="ordinary_visual_references",
                max_references=MAX_REFERENCE_IMAGES,
                max_reference_bytes=MAX_REFERENCE_BYTES,
                max_prompt_chars=16384,
                control_bytes=128 << 10,
                source_history_bytes=64 << 20,
                play_seconds=self.config.play_seconds,
                startup_wait_seconds=600,
                segment_startup_seconds=120,
                binding_lifetime_seconds=self.binding_lifetime_seconds,
                reconnect_grace_seconds=15,
                cleanup_timeout_ms=90000,
                **(
                    {"startup_prepare_protocol": self.startup.request["protocol"]}
                    if self.startup
                    else {}
                ),
            ),
        )
        if model_startup:
            ready["startup_request_id"] = (
                self.startup.request["request_id"]
                if self.startup
                else self.context_startup_request_id
            )
            ready.update(display_mode="model", display_revision=1)
        ready["inference"] = copy.deepcopy(self.journal.snapshot.get("inference"))
        if self.member_prefill is not None:
            ready["member_prefill"] = self.member_prefill.descriptor()
        await self.bind_source_conditions()
        if getattr(self, "world_driven", False):
            ready["capabilities"].update(
                model_context="worldkit-model-context-v1",
                model_context_instruction=True,
            )
            ready["model_context_segment"] = self.model_context_segment(
                await self.journal.read()
            )
        elif self.inference.manual and self.member_prefill is None:
            ready["model_context_segment"] = self.model_context_segment(
                await self.journal.read()
            )
        await self.outgoing.put(msgspec.msgpack.encode(ready))
        self.ready.set()
        self._spawn(self._preview(), "source-preview")
        if self.startup is not None:
            self._spawn(self._finish_startup(), "model-prepare-handoff")
        await self.preview_done.wait()
        self.source_state_value = "ended"
        if self.current_segment:
            await self.current_segment.stop()

        def source_ended(state):
            state["source_state"] = "ended"
            for operation in state["operations"].values():
                if (
                    operation["op"] == "source.close"
                    and operation["state"] == "accepted"
                ):
                    operation.update(
                        state="completed",
                        source_state="ended",
                        resource_state=(
                            self.current_segment.resource_state
                            if self.current_segment
                            else "none"
                        ),
                    )

        await self.journal.update(source_ended)
        self.recording.manifest["rendering"].update(
            source_frames_captured=self.submitted,
            recorded_preview_frames=self.preview_count,
            preview_superseded_frames=self.preview_dropped,
        )
        await self.recording.configure()
        self.outcome = "completed"
        self.finished.set()
        await self.outgoing.put(
            msgspec.msgpack.encode(
                dict(
                    version=2,
                    type="session.completed",
                    source_frames=self.preview_count,
                    **self.source_identity(),
                )
            )
        )

    async def bind_source_conditions(self):
        binder = getattr(self.engine, "bind_model_context", None)
        capability = getattr(self.engine, "model_context_capability", None)
        if capability is None:
            capability = getattr(self.engine, "descriptor", {}).get("modelContext")
        if capability is None:
            return
        if binder is None:
            raise RenderError("CAPABILITY_UNAVAILABLE")
        ids = await self.journal.reserve_model_context_segment(
            getattr(self, "context_startup_ids", None)
        )
        from .source_conditions import SourceConditions

        def create(envelope, emit, check):
            # A native reset reuses this factory with a fresh segment. Keep each
            # callback's identity frozen, including a late result from an old one.
            receipt_ids = copy.deepcopy(
                self.journal.snapshot.get("model_context_segment_ids")
            )
            if receipt_ids and receipt_ids["segment_id"] != envelope["segmentId"]:
                receipt_ids = None

            def fenced():
                check()
                if self.closed or self.binding_ended:
                    raise RenderError("SOURCE_EPOCH_MISMATCH")
                member = getattr(self.engine, "source", None)
                if member is not None:
                    member.check()

            def capabilities():
                segment = self.current_segment
                return dict(
                    max_references=getattr(
                        segment,
                        "model_reference_limit",
                        int(
                            os.environ.get("WHITEBOX_INITIAL_REFERENCE_CAPACITY", "20")
                        ),
                    ),
                    max_reference_bytes=getattr(
                        segment, "model_reference_bytes", MAX_REFERENCE_BYTES
                    ),
                    reference_updates=getattr(segment, "clip_switch", False)
                    and getattr(segment, "prompt_reference_updates", False),
                )

            def startup_window():
                window = getattr(getattr(self, "inference", None), "prewarm", None)
                return (
                    window
                    if window is not None and window.segment_id == envelope["segmentId"]
                    else None
                )

            def startup_release(condition):
                channel = self.context_channel
                window = startup_window()
                if (
                    channel is None
                    or channel.suspended
                    or window is None
                    or window.task is None
                    and not (
                        getattr(channel, "prewarm_supported", False)
                        or getattr(channel, "prewarm_plan_supported", False)
                    )
                ):
                    return None
                if getattr(self.engine, "shared_world", False) and not getattr(
                    channel, "prewarm_plan_supported", False
                ):
                    return None
                if window.task is None and getattr(
                    channel, "prewarm_plan_supported", False
                ):
                    from .startup_prewarm_plan import StartupPrewarmPlan

                    if not isinstance(window, StartupPrewarmPlan):
                        window = self.inference.prewarm = StartupPrewarmPlan(
                            self,
                            window.segment_id,
                            window.opening_revision,
                            window.opening_deadline,
                        )
                window.append_supported = getattr(
                    channel, "prewarm_append_supported", False
                )
                return window.gate(
                    condition,
                    emit=emit,
                    resolve=lambda value: adapter.prepare_references(
                        value, capabilities()
                    ),
                    check=fenced,
                )

            def startup_effective(condition):
                channel = self.context_channel
                window = channel.prewarm_window() if channel is not None else None
                return window.effective_for(condition) if window is not None else None

            def first_presented_result(operation):
                self.spawn_control(self.operation_message(operation))

            prewarm_enabled = os.environ.get("WHITEBOX_MODEL_CONTEXT_PREWARM") == "1"
            adapter = SourceConditions(
                envelope=envelope,
                assets=self.assets,
                journal=self.journal,
                read_resource=self.engine.read_model_context_resource,
                emit=emit,
                submit=self.submit_source_condition,
                capabilities=capabilities,
                validate_reference=validate_reference_geometry,
                check=fenced,
                first_presented_supported=lambda: getattr(
                    self.context_channel, "first_presented_supported", False
                ),
                first_presented_result=first_presented_result,
                **(
                    dict(
                        startup_release=startup_release,
                        startup_effective=startup_effective,
                    )
                    if prewarm_enabled
                    else {}
                ),
            )
            if prewarm_enabled:
                adapter.startup_window = startup_window

            async def instruction_receipt(instruction, phase, reason):
                fenced()
                if receipt_ids is None:
                    raise RenderError("SOURCE_EPOCH_MISMATCH")
                await self.emit_instruction_receipt(
                    instruction, phase, reason, receipt_ids
                )

            adapter.instruction_receipt = instruction_receipt
            adapter.debug = getattr(self, "condition_debug", None)
            self.source_conditions = adapter
            return adapter

        self.context_channel = await binder(
            segment_id=ids["segment_id"],
            create_adapter=create,
            **({"start_suspended": True} if not self.inference_enabled() else {}),
        )
        self.context_channel.on_recovery_failed = self.context_recovery_failed

    async def context_recovery_failed(self, code, *, source_reset=None):
        segment = self.current_segment
        if source_reset is not None:
            # The channel already fenced this adapter's generation. Completion
            # must publish to the journal, without re-entering its stale writer.
            self.source_conditions = None
        if segment is not None:
            # Fence model admission synchronously before preview-only pixels
            # are released. Its independent cleanup must not stop the World.
            segment.publishing = False
            segment.source_view.stop()
            if source_reset is None:
                self.spawn_control(segment.stop(cancel=True))
        if source_reset is None:
            await self._status("context_recovery_required", code=code)
        else:
            # Close may cancel the outgoing reader before native retirement ends.
            # The source owns this notice; it must not gate original cleanup.
            self.spawn_control(self._status("context_recovery_required", code=code))
        inference = getattr(self, "inference", None)
        state = self.journal.snapshot
        admitted = state.get("segment")
        if (
            source_reset is not None
            and getattr(inference, "manual", False)
            and (
                state["inference"]["state"] == "starting"
                or admitted is not None
                and segment is not None
                and admitted["segment_id"] != segment.segment_id
            )
        ):
            await inference.retire_preparation(code)
            # Ordinary disable owns cancellation of an admitted, unsent successor.
            # Retire that journal generation, never a stale current_segment pointer.
            admitted = self.journal.snapshot.get("segment")
            if admitted is not None:
                await self.journal.retire_model_context_source(
                    admitted["segment_id"], source_reset
                )
            return
        if segment is not None and source_reset is not None:
            # Native reset retires this execution, not its historical outcome.
            # Wait outside the shared CLI reader for the original child to close.
            await segment.stop(cancel=True)
            if not segment.closed_event.is_set():
                raise RenderError("RELEASE_UNKNOWN")
            await self.journal.retire_model_context_source(
                segment.segment_id, source_reset
            )

    def inference_enabled(self):
        return (self.journal.snapshot or {}).get("inference", {}).get("enabled", True)

    def emit_prewarm_observation(self, event):
        observer = getattr(self, "prewarm_observer", None)
        if event is not None and observer is not None:
            observer(event)

    def emit_prewarm_plan_observation(self, event):
        observer = getattr(self, "prewarm_plan_observer", None)
        if event is not None and observer is not None:
            observer(event)

    def record_startup_prewarm(self, window, message, decision=None):
        if message["type"].startswith("model.context.prewarm.plan"):
            from .prewarm_observation import PrewarmPlanObservations

            source = self.source_identity()
            scope = dict(
                source_session_id=source["source_session_id"],
                source_epoch=source["source_epoch"],
                consumer_id=message["audience"]["consumerId"],
            )
            store = getattr(self, "prewarm_plan_observations", None)
            if store is None or store.source != scope:
                store = self.prewarm_plan_observations = PrewarmPlanObservations(
                    **scope
                )
            self.emit_prewarm_plan_observation(store.record(window, message, decision))
            return
        from .prewarm_observation import PrewarmPlayerObservations

        store = getattr(self, "prewarm_observations", None)
        if message["type"] == "model.context.prewarm.window":
            identity = self.source_identity()
            scope = dict(
                source_session_id=identity["source_session_id"],
                source_epoch=identity["source_epoch"],
                consumer_id=message["audience"]["consumerId"],
            )
            if store is None or store.source != scope:
                store = self.prewarm_observations = PrewarmPlayerObservations(**scope)
            event = store.open(
                window.window_id, window.opening_revision, segment_id=window.segment_id
            )
            if event is not None:
                self.prewarm_consumption = None
            self.emit_prewarm_observation(event)
        elif store is not None:
            fields = {
                target: message[key]
                for key, target in (
                    ("intentId", "intent_id"),
                    ("candidateRevision", "candidate_revision"),
                    ("requestId", "request_id"),
                    ("reason", "reason"),
                    ("durationMs", "duration_ms"),
                )
                if key in message
            }
            if message["phase"] == "memory-ready" and "cacheKey" in message:
                fields["cache_key"] = message["cacheKey"]
            if decision and "eventId" in decision:
                fields.update(
                    event_id=decision["eventId"],
                    semantic_sha256=decision["contentHash"],
                )
            self.emit_prewarm_observation(
                store.record(
                    window.window_id,
                    message["decisionRevision"],
                    message["phase"],
                    **fields,
                )
            )

    def track_prewarm_consumption(self, window, condition, operation):
        if window is not None and hasattr(window, "entries"):
            if window.retired or window.binding_identity() != window.binding:
                return
            if condition["segmentId"] != window.segment_id:
                return
            matches = [
                item["entry"]
                for item in window.entries.values()
                if item["memory_ready"]
                and not item["cancelled"]
                and all(
                    condition[key] == item["entry"][key]
                    for key in ("lifecycle", "prompt", "references")
                )
            ]
            if matches:
                # Associate only once: identical nominees must not multiply
                # a native formal request's hit/miss denominator.
                consumed = getattr(self, "prewarm_plan_consumption", {})
                consumed[(operation["segment_epoch"], operation["prompt_version"])] = (
                    window,
                    matches[0],
                    operation.copy(),
                )
                while len(consumed) > 64:
                    del consumed[next(iter(consumed))]
                self.prewarm_plan_consumption = consumed
            return
        store = getattr(self, "prewarm_observations", None)
        computed = window.computed if window is not None else None
        if (
            store is None
            or not store.window
            or computed is None
            or store.window["window_id"] != window.window_id
            or getattr(self, "prewarm_consumption", None) is not None
            or operation["op"] != "prompt.replace"
            or condition["segmentId"] != window.segment_id
            or condition["lifecycle"] != computed["lifecycle"]
            or condition["prompt"] != computed["prompt"]
            or condition["references"] != computed["references"]
        ):
            return
        self.prewarm_consumption = (
            window.window_id,
            operation["request_id"],
            operation["prompt_version"],
            operation["segment_id"],
            operation["segment_epoch"],
        )
        self.emit_prewarm_observation(
            store.record(
                window.window_id,
                computed["decisionRevision"],
                "fulfilled",
                intent_id=computed["intentId"],
                candidate_revision=computed["candidateRevision"],
                segment_id=operation["segment_id"],
                segment_epoch=operation["segment_epoch"],
            )
        )

    def require_opening_presentation_player(self):
        if self.first_presented_required and (
            self.first_presented_player_ready is None
            or not self.first_presented_player_ready()
        ):
            raise RenderError("MODEL_PRESENTATION_UNSUPPORTED")

    def require_opening_presentation_channel(self, segment_id):
        segment = (self.journal.snapshot or {}).get("segment") or {}
        if (
            self.first_presented_required
            and segment.get("context_segment_id") != segment_id
        ):
            self.require_opening_presentation_player()
            if not getattr(self.context_channel, "first_presented_supported", False):
                raise RenderError("MODEL_PRESENTATION_UNSUPPORTED")

    async def submit_source_condition(self, condition, references):
        # Called only by the authenticated source channel, never browser commands.
        if not self.inference_enabled() or getattr(
            self.context_channel, "suspended", False
        ):
            raise RenderError("MODEL_UNAVAILABLE")
        if getattr(self, "first_presented_required", False):
            self.require_opening_presentation_channel(condition["segmentId"])
        resolved = await self.assets.resolve(references)
        if not self.inference_enabled() or getattr(
            self.context_channel, "suspended", False
        ):
            raise RenderError("MODEL_UNAVAILABLE")
        if getattr(self, "first_presented_required", False):
            self.require_opening_presentation_channel(condition["segmentId"])
        window = getattr(getattr(self, "inference", None), "prewarm", None)
        startup_check = {}
        first = None
        if (
            window is not None
            and window.segment_id == condition["segmentId"]
            and (
                window.condition is not None
                or getattr(self.context_channel, "prewarm_supported", False)
            )
        ):
            first = window.first_for(condition)
            startup_check = dict(
                opening_revision=window.opening_revision,
                startup_prepared=window.prepared_for(condition),
            )
            if first is not None:
                startup_check["source_first"] = first[1][0].index
        if getattr(self.context_channel, "first_presented_supported", False):
            startup_check["first_presented"] = True
        operation = await self.journal.model_condition(
            condition,
            references,
            seed=getattr(self, "context_seed", 42),
            **startup_check,
        )
        if operation.get("duplicate"):
            return operation
        if operation["op"] == "render.start":
            if window is not None and hasattr(window, "entries"):
                self.track_prewarm_consumption(window, condition, operation)
            if first is not None:
                window.first = None
            self.current_model_stream = operation["segment_epoch"]
            await self.outgoing.put(
                msgspec.msgpack.encode(
                    dict(
                        version=2,
                        type="render.stream",
                        segment_id=operation["segment_id"],
                        segment_epoch=operation["segment_epoch"],
                        media_stream_id=operation["segment_epoch"],
                        render_revision=operation["render_revision"],
                        **self.source_identity(),
                    )
                )
            )
            self.spawn_control(self._transition_segment(operation, resolved, first))
        elif self.current_segment is not None:
            self.track_prewarm_consumption(window, condition, operation)
            self.current_segment.prompt_available.set()
        # The durable action proceeds even if its external receipt is lost.
        await self.operation_message(operation)
        return operation

    async def _render(self):
        await self.finished.wait()

    async def source_state(self, state):
        state = dict(state, **self.source_identity())
        state["source"] = {**(state.get("source") or {}), **self.source_identity()}
        await super().source_state(state)

    async def _media(
        self, stream, index, frame, encoded, captured, recording_index=None
    ):
        if stream != "whitebox":
            raise RenderError("INVALID_COMMAND")
        archived = replace(
            frame,
            index=index,
            pts_us=self.config.spec.pts(index),
            provenance={
                **(frame.provenance or {}),
                "source_frame_index": frame.index,
                "capture_pts_us": capture_pts(frame),
            },
        )
        if getattr(self.recording, "raw_archive", False):
            pass  # Every captured RGB was offered before the live latest-frame boundary.
        elif hasattr(self.recording, "source_frame"):
            self.recording.source_frame(archived, encoded)
        else:
            await self.recording.encoded_video(
                index, encoded, name="whitebox", pts_us=capture_pts(frame)
            )
            await self.recording.write_metadata(archived, "")
        if self.binding_ended:
            return
        await self.outgoing.put(
            msgspec.msgpack.encode(
                await self.area_media(
                    dict(
                        version=2,
                        type="whitebox.frame",
                        stream="whitebox",
                        output_frame_start=index,
                        num_frames=1,
                        pts_us=capture_pts(frame),
                        source_frame_index=frame.index,
                        source_capture_mono_ms=captured * 1000,
                        media_stream_id=self.source_epoch,
                        source=frame.provenance,
                        states=frame.states,
                        observed_input=frame.observed_input,
                        width=self.config.spec.width,
                        height=self.config.spec.height,
                        server_epoch_ms=time.time() * 1000,
                        pipeline_ms=(time.perf_counter() - captured) * 1000,
                        **self.source_identity(),
                        **encoded,
                    )
                )
            )
        )

    async def _deadline(self):
        while not self.finished.is_set():
            await asyncio.sleep(0.1)
            now = time.monotonic()
            if (
                self.model_binding is not None
                and self.model_binding.closed
                and not getattr(getattr(self, "inference", None), "manual", False)
                and self.inference_enabled()
                and not getattr(self, "closing", False)
            ):
                raise RenderError("MODEL_UNAVAILABLE")
            prefill = self.control_prefill
            if prefill is not None:
                prefill.check_deadline(now)
            if (
                self.startup is not None
                and not self.startup.started
                and now >= self.startup.deadline
            ):
                raise TimeoutError("Play startup deadline exceeded")
            entry = (
                self.host.runtime.players.get(self.engine.stream_id)
                if self.engine is not None
                else None
            )
            if entry is not None and self.host.runtime.input_idle_expired(entry):
                raise RenderError("PLAYER_IDLE_TIMEOUT")
            elapsed = now - self.last_tick
            self.last_tick = now
            if self.bound_at is None or self.binding_ended:
                continue
            model = self.current_segment
            playable = (
                self.source_state_value == "ready"
                and not (
                    getattr(self, "source_pause", None) is not None
                    and self.source_pause.active
                )
                and not (
                    self.control_prefill is not None
                    and not self.control_prefill.resumed
                )
                and (
                    self.display_mode == "whitebox"
                    or (
                        model is not None
                        and model.segment_epoch == self.current_model_stream
                        and model.first_decodable
                        and not model.closed_event.is_set()
                    )
                )
            )
            if playable:
                self.play_elapsed += elapsed
            else:
                self.wait_elapsed += elapsed
            if self.model_binding is not None:
                if (
                    self.model_binding.assignment is not None
                    and self.player_remaining_seconds <= 0
                ):
                    raise RenderError("PLAYER_EXPIRED")
            elif (
                self.play_elapsed >= self.config.play_seconds
                or self.wait_elapsed >= 600
                or now - self.bound_at >= self.binding_lifetime_seconds
            ):
                raise RenderError("BINDING_BUDGET_EXHAUSTED")

    async def _status(self, state, **fields):
        if getattr(self, "closing", False):
            return
        await self.outgoing.put(
            msgspec.msgpack.encode(
                dict(
                    version=2,
                    type="render.status",
                    inference=copy.deepcopy(
                        (self.journal.snapshot or {}).get("inference")
                    ),
                    state=state,
                    **self.source_identity(),
                    **fields,
                )
            )
        )

    async def send(self, wire):
        message = msgspec.msgpack.decode(wire)
        if not isinstance(message, dict):
            raise RenderError("INVALID_COMMAND")
        if message.get("eventType") is not None:
            if self.control_prefill is None:
                raise RenderError("CAPABILITY_UNAVAILABLE")
            if getattr(self, "member_prefill", None) is not None:
                self.member_prefill.validate_conditions(message.get("payload", {}))
            await self.control_prefill.submit(message)
        elif message.get("type") == "command":
            await self.command(message)
        elif message.get("type") == "asset.upload":
            await self.upload_asset(message)
        elif message.get("kind") == "prompt":
            raise RenderError("INVALID_COMMAND")
        else:
            await super().send(wire)

    async def upload_asset(self, message):
        operation = None
        try:
            if getattr(getattr(self, "inference", None), "manual", False):
                raise RenderError("INVALID_COMMAND")
            if (
                getattr(self, "world_driven", False)
                or getattr(self, "context_channel", None) is not None
            ):
                raise RenderError("INVALID_COMMAND")
            state = await self.journal.read()
            self.journal.validate_scope(state, message)
            if self.assets is None or self.binding_ended:
                raise RenderError("INVALID_SOURCE")
            if set(message) - {
                "type",
                "version",
                "request_id",
                "source_session_id",
                "source_epoch",
                "format",
                "payload",
            }:
                raise RenderError("INVALID_COMMAND")
            data = message.get("payload")
            if not isinstance(data, bytes) or len(data) > 8 << 20:
                raise RenderError("LIMIT_EXCEEDED")
            intent = {k: v for k, v in message.items() if k not in {"type", "payload"}}
            intent.update(
                op="asset.upload",
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
            operation = await self.journal.command(
                intent, lambda s, o: o.update(state="accepted")
            )
            if operation.get("state") == "complete":
                await self.operation_message({**operation, "duplicate": True})
                return
            asset = await self.assets.upload(message.get("format"), data)
            operation = await self.journal.result(
                message["request_id"], state="complete", asset=asset
            )
            await self.operation_message(operation)
        except RenderError as error:
            if operation is not None and error.code != "CONTROL_UNAVAILABLE":
                await self.journal.result(
                    message["request_id"], state="failed", code=error.code
                )
            await self.outgoing.put(
                msgspec.msgpack.encode(error.message(message.get("request_id")))
            )

    async def command(self, command):
        request_id = command.get("request_id")
        try:
            if (
                getattr(getattr(self, "inference", None), "manual", False)
                and getattr(self, "member_prefill", None) is None
                and command.get("op")
                in {
                    "render.start",
                    "render.replace",
                    "prompt.replace",
                    "render.context.start",
                    "render.prepare",
                    "render.prepare.cancel",
                }
            ):
                raise RenderError("INVALID_COMMAND")
            if getattr(self, "member_prefill", None) is not None and command.get(
                "op"
            ) in {
                "model.inference.configure",
                "render.replace",
                "prompt.replace",
                "render.context.start",
            }:
                raise RenderError("INVALID_COMMAND")
            if (
                getattr(self, "world_driven", False)
                or getattr(self, "context_channel", None) is not None
            ) and command.get("op") in {
                "prompt.replace",
                "render.start",
                "render.replace",
            }:
                raise RenderError("INVALID_COMMAND")
            state = await self.journal.read()
            self.journal.validate_scope(state, command)
            if command.get("op") in {
                "model.context.intervention",
                "model.context.intervention.get",
            }:
                await self.context_intervention(command, state)
                return
            previous = state["operations"].get(request_id)
            if previous and command.get("op") not in {
                "model.inference.configure",
                "source.pause",
                "source.resume",
            }:
                if (
                    previous["payload_sha256"]
                    != hashlib.sha256(canonical(command)).hexdigest()
                ):
                    raise RenderError("REQUEST_ID_CONFLICT")
                if previous["op"] == "render.context.start":
                    await self.handoff_context_segment(previous, state)
                await self.operation_message({**previous, "duplicate": True})
                return
            pending_end = getattr(self, "pending_end", None)
            if pending_end is not None and request_id == pending_end.get("request_id"):
                if canonical(command) != canonical(pending_end):
                    raise RenderError("REQUEST_ID_CONFLICT")
                return
            op = command.get("op")
            if (
                self.startup is not None
                and not self.startup.started
                and op not in {"source.close", "status.get", "operation.get"}
            ):
                raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
            if op in {"source.pause", "source.resume"}:
                if self.binding_ended or self.close_requested:
                    raise RenderError("INVALID_SOURCE")
                await self.source_pause.command(command)
                return
            if (
                op == "render.prepare"
                and getattr(self, "source_pause", None) is not None
                and self.source_pause.active
            ):
                raise RenderError("SOURCE_PAUSE_BUSY")
            if op == "model.inference.configure":
                if not getattr(getattr(self, "inference", None), "manual", False):
                    raise RenderError("CAPABILITY_UNAVAILABLE")
                if command.get("enabled") is True:
                    self.require_opening_presentation_player()
                source_generation = getattr(
                    getattr(self, "context_channel", None), "source_generation", None
                )
                result = await self.journal.configure_inference(command)
                state = await self.journal.read()
                self.display_mode = state["display_mode"]
                self.revision = state["display_revision"]
                # A delayed duplicate stop is historical evidence, not a new
                # fence on a source that has since enabled a fresh segment.
                if not state["inference"]["enabled"]:
                    self.inference.fence(drain_pending=True)
                await self.operation_message(result)
                await self.status_snapshot()
                if result["state"] == "accepted" and not result.get("duplicate"):
                    if command["enabled"] and source_generation != getattr(
                        getattr(self, "context_channel", None),
                        "source_generation",
                        None,
                    ):
                        # The receipt/status awaits cannot carry old source intent
                        # into a reset generation, even before a task existed.
                        result = await self.journal.settle_inference(
                            result["request_id"],
                            result["inference"]["revision"],
                            state="failed",
                            code="MODEL_CONTEXT_STALE_SOURCE",
                        )
                        await self.operation_message(result)
                        await self.status_snapshot()
                    else:
                        self.inference.dispatch(result)
                return
            if op in {"status.get", "operation.get"}:
                await self.status_snapshot(
                    request_id, command.get("operation_request_id")
                )
                return
            if op in {
                "model.sink",
                "display.configure",
                "render.start",
                "render.replace",
                "render.end",
                "render.context.start",
                "prompt.replace",
            } and any(
                item.get("op") == "area.presented"
                and item.get("state") == "dispatching"
                for item in state["operations"].values()
            ):
                raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
            now = time.monotonic()
            self.control_tokens = min(
                4.0, self.control_tokens + 2 * (now - self.control_token_time)
            )
            self.control_token_time = now
            if self.control_tokens < 1:
                raise RenderError("LIMIT_EXCEEDED")
            self.control_tokens -= 1
            if op == "area.presented":
                await self.prepare_area_confirmation(command)
                return
            if op == CONFIRM_MODEL_PRESENTATION:
                await self.confirm_opening_presentation(command)
                return
            if op in {"source.close", "source.unbind"}:
                result = await self.journal.command(
                    command, lambda s, o: o.update(state="accepted")
                )
                await self.operation_message(result)
                pipeline = self.host.sessions.get(self.session_id)
                if pipeline is not None:
                    self.request_player_finish(pipeline)
                else:
                    raise RenderError("ROOM_SESSION_LOST")
                return
            if self.binding_ended or self.close_requested:
                raise RenderError("LIMIT_EXCEEDED")
            if op == "model.sink":
                current = self.current_segment
                if current is None:
                    raise RenderError("MODEL_UNAVAILABLE")
                await current.sink.command(command)
                return
            if op == "render.context.start":
                if self.context_channel is None:
                    raise RenderError("CAPABILITY_UNAVAILABLE")
                result = await self.journal.start_model_context_segment(command)
                await self.handoff_context_segment(result, await self.journal.read())
                await self.operation_message(result)
                return
            if op == "display.configure":
                if command.get("mode") == "model" and not self.inference_enabled():
                    raise RenderError("MODEL_UNAVAILABLE")
                result = await self.journal.display(command)
                self.display_mode = command["mode"]
                self.revision = result["display_revision"]
                await self.operation_message(result)
                return
            if op == "render.prepare.cancel":
                if self.control_prefill is None:
                    raise RenderError("CAPABILITY_UNAVAILABLE")
                await self.control_prefill.cancel(command)
                return
            if op == "render.prepare":
                if self.control_prefill is None:
                    raise RenderError("CAPABILITY_UNAVAILABLE")
                await self.control_prefill.prepare(command)
                return
            if op in {"render.start", "render.replace"}:
                if getattr(self, "member_prefill", None) is not None:
                    self.member_prefill.validate_start(command)
                segment_parameters(command)
                # Validate all assets before recording a replacement intent.
                references = await self.assets.resolve(command.get("references", []))
                if getattr(self, "member_prefill", None) is not None:
                    self.member_prefill.validate_start(command)
                first = None
                boundary = command.get("start_boundary", {"mode": "next_available"})
                prefill = self.control_prefill
                if (
                    prefill is not None
                    and prefill.active is not None
                    and not prefill.active.done()
                ):
                    raise RenderError("PREFILL_BUSY")
                if command.get("context_id") is not None:
                    if prefill is None:
                        raise RenderError("CAPABILITY_UNAVAILABLE")
                    first = prefill.resolve(command["context_id"])
                    if boundary != {"source_frame_index": first[1][0].index}:
                        raise RenderError("PREFILL_INPUT_MISMATCH")
                if first is None and boundary != {"mode": "next_available"}:
                    first = self.retained.exact(boundary)
                    if first is None:
                        raise RenderError("START_SOURCE_EXPIRED")
                await self.validate_area_binding(command)
                result = await self.journal.start(
                    command, replace=op == "render.replace"
                )
                await self.operation_message(result)
                self.current_model_stream = result["segment_epoch"]
                await self.outgoing.put(
                    msgspec.msgpack.encode(
                        dict(
                            version=2,
                            type="render.stream",
                            segment_id=result["segment_id"],
                            segment_epoch=result["segment_epoch"],
                            media_stream_id=result["segment_epoch"],
                            render_revision=result["render_revision"],
                            **self.source_identity(),
                        )
                    )
                )
                self.spawn_control(self._transition_segment(result, references, first))
                return
            if op == "render.end":
                deadline = (
                    asyncio.get_running_loop().time() + self.config.drain_seconds + 10
                )
                send_lock = getattr(self.current_segment, "source_send_lock", None)
                if command.get("mode") != "cancel":
                    if pending_end is not None:
                        raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
                    if send_lock is not None:
                        # The ordered Room queue must still admit cancel/close
                        # while prompt dispatch holds the drain boundary.
                        self.pending_end = command
                        self.spawn_control(
                            self._admit_end(command, deadline, send_lock)
                        )
                        return
                await self._admit_end(command, deadline, send_lock)
                return
            if op == "prompt.replace":
                segment = self.current_segment
                if (
                    segment is not None
                    and getattr(segment, "clip_switch", False)
                    and command.get("effective", {}).get("policy") == "strict"
                ):
                    raise RenderError("EFFECTIVE_BOUNDARY_UNAVAILABLE")
                if "references" in command:
                    refs = reference_parameters(command["references"])
                    if segment is None or not getattr(segment, "clip_switch", False):
                        raise RenderError("CAPABILITY_UNAVAILABLE")
                    if len(refs) > segment.model_reference_limit:
                        raise RenderError("LIMIT_EXCEEDED")
                    await self.assets.resolve(refs)
                if command.get("source_policy") and (
                    segment is None or not getattr(segment, "clip_switch", False)
                ):
                    raise RenderError("CAPABILITY_UNAVAILABLE")
                await self.validate_area_binding(command)
                result = await self.journal.prompt(command)
                await self.operation_message(result)
                if segment is not None:
                    segment.prompt_available.set()
                return
            raise RenderError("INVALID_COMMAND")
        except RenderError as error:
            await self.outgoing.put(msgspec.msgpack.encode(error.message(request_id)))

    async def _admit_end(self, command, deadline, send_lock):
        try:
            # Serialize drain admission with prompt dispatch, but keep the CAS
            # outside the lock timeout so a durable end is never cancelled midway.
            if command.get("mode") == "cancel" or send_lock is None:
                result = await self.journal.end(command)
            else:
                try:
                    async with asyncio.timeout_at(deadline):
                        await send_lock.acquire()
                except TimeoutError as error:
                    raise RenderError("RENDER_TRANSITION_IN_PROGRESS") from error
                try:
                    result = await self.journal.end(command)
                finally:
                    send_lock.release()
            if command.get("mode") == "cancel":
                current = self.current_segment
                if current and current.segment_id == command["segment_id"]:
                    # Wake the admitted end's condition drain before taking its
                    # cleanup lock. The original prompt keeps its journal identity.
                    current.cancel_requested = True
            await self.operation_message(result)
            if command.get("mode") == "cancel":
                self.current_model_stream = None
                await self.outgoing.put(
                    msgspec.msgpack.encode(
                        dict(
                            version=2,
                            type="render.stream",
                            media_stream_id=None,
                            render_revision=result["render_revision"],
                            **self.source_identity(),
                        )
                    )
                )
            self.spawn_control(self._end_segment(command, result, deadline=deadline))
        except RenderError as error:
            await self.outgoing.put(
                msgspec.msgpack.encode(error.message(command.get("request_id")))
            )
        finally:
            if getattr(self, "pending_end", None) is command:
                self.pending_end = None

    @staticmethod
    def model_context_segment(state):
        ids = state.get("model_context_segment_ids") or (
            state.get("segment") if state.get("manual_inference_journal") else None
        )
        return {k: ids[k] for k in ("segment_id", "segment_epoch")} if ids else None

    async def context_intervention(self, command, state):
        if not getattr(self, "world_driven", False) and not getattr(
            getattr(self, "inference", None), "manual", False
        ):
            raise RenderError("CAPABILITY_UNAVAILABLE")
        allowed = {
            "type",
            "version",
            "op",
            "request_id",
            "source_session_id",
            "source_epoch",
            "segment_id",
            "segment_epoch",
        }
        if command["op"] == "model.context.intervention":
            allowed.update(("expectedRevision", "prompt"))
            if (
                type(command.get("expectedRevision")) is not int
                or not 0 <= command["expectedRevision"] <= 2**53 - 1
            ):
                raise RenderError("INVALID_COMMAND")
            if "prompt" not in command:
                raise RenderError("INVALID_COMMAND")
            if command["prompt"] is not None:
                prompt_text(command["prompt"])
        if set(command) - allowed:
            raise RenderError("INVALID_COMMAND")
        identity(command.get("request_id"))
        ids = self.model_context_segment(state)
        if ids is None or any(command.get(k) != v for k, v in ids.items()):
            raise RenderError("SEGMENT_EPOCH_MISMATCH")
        if (
            self.closed
            or self.binding_ended
            or state["source_state"] != "ready"
            or self.context_channel is not None
            and self.context_channel.segment != ids["segment_id"]
        ):
            raise RenderError("INVALID_SOURCE")
        # Only a Harness source binds a context engine. Ordinary sources are
        # fenced by their source journal and member lifetime below.
        check_owner = (
            getattr(self.engine, "check_model_context", None)
            if getattr(self, "world_driven", False)
            else None
        )
        if check_owner is not None:
            check_owner()
        member = getattr(self.engine, "source", None)
        if member is not None:
            member.check()
        if getattr(self, "world_driven", False):
            if self.context_channel is None:
                raise RenderError("CAPABILITY_UNAVAILABLE")
            result = await self.context_channel.intervention(command)
        else:
            result = await self.journal.player_instruction(command)
        # A source/member/segment replacement during the await cannot publish a
        # successful answer against the replacement's browser identity.
        current = await self.journal.read()
        self.journal.validate_scope(current, command)
        if (
            self.closed
            or self.binding_ended
            or self.model_context_segment(current) != ids
        ):
            raise RenderError("SEGMENT_EPOCH_MISMATCH")
        if check_owner is not None:
            check_owner()
        if member is not None:
            member.check()
        await self.outgoing.put(
            msgspec.msgpack.encode(
                dict(
                    type="model.context.intervention.result",
                    request_id=command["request_id"],
                    **self.source_identity(),
                    **ids,
                    **{
                        k: result[k]
                        for k in ("status", "revision", "prompt", "code")
                        if k in result
                    },
                )
            )
        )

        # Queue the correlated receipt before model work can publish its progress.
        if (
            not getattr(self, "world_driven", False)
            and result["status"] == "accepted"
            and self.current_segment is not None
        ):
            self.current_segment.prompt_available.set()

    async def handoff_context_segment(self, operation, state):
        if self.closed or self.binding_ended:
            return
        ids = state.get("model_context_segment_ids")
        if ids and ids["segment_id"] == operation["segment_id"]:
            # Only a genuine, journal-reserved successor opens another window.
            # Retries of the same handoff reuse that window and original operation.
            inference = state.get("inference", {})
            if (
                os.environ.get("WHITEBOX_MODEL_CONTEXT_PREWARM") == "1"
                and inference.get("enabled")
                and getattr(getattr(self, "inference", None), "manual", False)
            ):
                await self.inference.open_prewarm(
                    operation["segment_id"], inference["revision"]
                )
            if getattr(
                getattr(self, "inference", None), "manual", False
            ) and not self.inference.current(inference["revision"]):
                return
            await self.context_channel.switch_segment(operation["segment_id"])

    async def validate_area_binding(self, command):
        binding = area_binding(command).get("area_binding")
        if (
            not binding
            and getattr(self.engine, "metadata", {}).get("area_protocol")
            != AREA_PROTOCOL
        ):
            return
        if not binding:
            raise RenderError("CONDITION_INVALID")
        native = await self.refresh_area_state()
        if not native or canonical(native["status"].get("transition")) != canonical(
            binding["identity"]
        ):
            raise RenderError("AREA_PRESENTATION_STALE")

    async def refresh_area_state(self):
        reader = getattr(self.engine, "area_status", None)
        if reader is None:
            raise RenderError("CAPABILITY_UNAVAILABLE")
        try:
            self.area_state = await reader()
        except (ValueError, ConnectionError) as error:
            self.area_state = None
            raise RenderError("AREA_PRESENTATION_STALE") from error
        return self.area_state

    async def area_media(self, message):
        if getattr(self.engine, "metadata", {}).get("area_protocol") != AREA_PROTOCOL:
            return message
        try:
            await self.refresh_area_state()
            header = (message.get("source") or {}).get("cli_frame_header")
            reader = getattr(self.engine, "area_frame", None)
            if header is not None and reader is not None:
                evidence = await reader(header)
                if evidence is not None:
                    message = dict(
                        message, source=dict(message["source"], area_frame=evidence)
                    )
        except (RenderError, ValueError, ConnectionError):
            self.area_state = None
        return message

    async def prepare_area_confirmation(self, command):
        required = {
            "op",
            "request_id",
            "source_session_id",
            "source_epoch",
            "connection_generation",
            "challenge",
            "receipt",
            "display",
        }
        if (
            set(command) - required - {"type", "version", "model"}
            or not required.issubset(command)
            or command.get("type", "command") != "command"
            or "version" in command
            and (type(command["version"]) is not int or command["version"] != 2)
        ):
            raise RenderError("AREA_RECEIPT_INVALID")
        if self.resolve_area_offer is None or self.binding_ended:
            raise RenderError("CAPABILITY_UNAVAILABLE")
        await self.refresh_area_state()
        self.resolve_area_offer(command)
        result = await self.journal.command(
            command, lambda s, operation: operation.update(state="accepted")
        )
        await self.operation_message(result)
        self.spawn_control(self.confirm_area(command))

    def replay_opening_presentation(self):
        adapter = self.source_conditions
        if adapter is not None and not adapter.closed:
            adapter.schedule_first_presented()

    async def confirm_opening_presentation(self, command):
        required = {
            "type",
            "op",
            "request_id",
            "source_session_id",
            "source_epoch",
            "connection_generation",
            "offer_id",
            "ui_revision",
        }
        if (
            not required.issubset(command)
            or set(command) - required - {"version"}
            or command["type"] != "command"
            or (
                "version" in command
                and (type(command["version"]) is not int or command["version"] != 2)
            )
            or type(command["connection_generation"]) is not int
            or not 0 < command["connection_generation"] < 2**53
            or type(command["ui_revision"]) is not int
            or not 0 <= command["ui_revision"] < 2**53
        ):
            raise RenderError("INVALID_COMMAND")
        identity(command["offer_id"])
        resolve_offer = self.resolve_first_presentation_offer
        if resolve_offer is None:
            raise RenderError("CAPABILITY_UNAVAILABLE")
        result = await self.journal.confirm_opening_presentation(command, resolve_offer)
        self.replay_opening_presentation()
        await self.operation_message(result)

    async def confirm_area(self, command):
        request_id = command["request_id"]
        sent = False
        try:
            await self.refresh_area_state()
            self.resolve_area_offer(command)
            result = await self.journal.result(request_id, state="dispatching")

            def before_send():
                nonlocal sent
                # The Room input lock, receipt log and CLI status may all wait.
                # Revoke a detached/replaced attachment at the actual native send.
                self.resolve_area_offer(command)
                sent = True

            confirmation = await self.engine.area_presented(
                dict(request_id=request_id, receipt=copy.deepcopy(command["receipt"])),
                before_send=before_send,
            )
            if (
                confirmation.get("status") not in {"confirmed", "duplicate"}
                or confirmation.get("operationId")
                != command["receipt"]["identity"]["operationId"]
            ):
                raise RenderError("AREA_RECEIPT_INVALID")
            # The confirmed effect itself releases the native hold. A capture may
            # observe it before this reply; every other authority fence still applies.
            self.resolve_area_offer(command, allow_interactive=True)
            result = await self.journal.result(
                request_id,
                state="completed",
                native_status=confirmation["status"],
                area_operation_id=confirmation["operationId"],
            )
        except (
            ValueError,
            ConnectionError,
            TimeoutError,
            asyncio.CancelledError,
        ) as error:
            result = await self.journal.result(
                request_id,
                state="reconciling" if sent else "rejected",
                code=(
                    "AREA_PRESENTATION_UNKNOWN"
                    if sent
                    else getattr(error, "code", "AREA_PRESENTATION_STALE")
                ),
            )
        await self.operation_message(result)

    def spawn_control(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.control_tasks.add(task)
        task.add_done_callback(self.control_tasks.discard)

        def completed(task):
            if task.cancelled():
                return
            error = task.exception()
            if error:
                logging.getLogger(__name__).error(
                    "Render control did not complete", exc_info=error
                )

        task.add_done_callback(completed)
        return task

    async def emit_instruction_receipt(self, instruction, phase, reason, ids):
        # Small business outcome: no prompt, images or debug subscription required.
        await self.outgoing.put(
            msgspec.msgpack.encode(
                dict(
                    type="model.context.instruction.receipt",
                    **self.source_identity(),
                    **ids,
                    requestId=instruction["requestId"],
                    revision=instruction["revision"],
                    phase=phase,
                    **({"reason": reason} if reason else {}),
                )
            )
        )

    async def operation_message(self, operation):
        if operation.get("periodic_refresh"):
            logging.getLogger(__name__).info(
                "condition_refresh request=%s state=%s version=%s code=%s",
                operation["request_id"],
                operation["state"],
                operation["prompt_version"],
                operation.get("code"),
            )
            return
        if getattr(self, "closing", False):
            return
        instruction = operation.get("instruction_result")
        if (
            instruction
            and instruction.get("status") == "accepted"
            and operation.get("state") in {"applied", "failed", "rejected"}
        ):
            current = (self.journal.snapshot or {}).get("segment")
            if (
                current
                and current["segment_id"] == operation.get("segment_id")
                and current["segment_epoch"] == operation.get("segment_epoch")
            ):
                await self.emit_instruction_receipt(
                    {
                        "requestId": operation["request_id"],
                        "revision": instruction["revision"],
                    },
                    operation["state"],
                    operation.get("code"),
                    {
                        "segment_id": current["segment_id"],
                        "segment_epoch": current["segment_epoch"],
                    },
                )
        adapter = getattr(self, "source_conditions", None)
        if (
            adapter is not None
            and not adapter.closed
            and operation.get("model_context") is not None
            and operation["model_context"]["segmentId"] == adapter.envelope["segmentId"]
        ):
            await adapter.original(operation["model_context"])
        message = {
            k: v
            for k, v in operation.items()
            if k
            not in {
                "payload_sha256",
                "model_context_lifecycle",
                "first_presented",
                "first_presented_confirmation_id",
            }
        }
        message["inference"] = copy.deepcopy(
            (self.journal.snapshot or {}).get("inference")
        )
        await self.outgoing.put(
            msgspec.msgpack.encode(
                dict(
                    version=2,
                    type="operation.result",
                    **self.source_identity(),
                    **message,
                )
            )
        )

    async def status_snapshot(self, request_id=None, operation_request_id=None):
        state = await self.journal.read()

        def summary(segment):
            if segment is None:
                return None
            return {
                k: v
                for k, v in segment.items()
                if k
                not in {
                    "prompt_queue",
                    "prompt_events",
                    "validating_prompt",
                    "periodic_refresh",
                    "refresh_condition_version",
                }
            }

        operation = state["operations"].get(operation_request_id)
        if (
            operation is not None
            and operation.get("op") == CONFIRM_MODEL_PRESENTATION
            and operation["state"] == "accepted"
        ):
            self.replay_opening_presentation()

        await self.outgoing.put(
            msgspec.msgpack.encode(
                dict(
                    version=2,
                    type="source.status",
                    request_id=request_id,
                    operation_request_id=operation_request_id,
                    connection_generation=getattr(self, "control_generation", None),
                    **self.source_identity(),
                    source_state=self.source_state_value,
                    source_pause=(
                        self.source_pause.snapshot()
                        if getattr(self, "source_pause", None) is not None
                        else None
                    ),
                    inference=copy.deepcopy(state.get("inference")),
                    render_revision=state["render_revision"],
                    display_revision=state["display_revision"],
                    display_mode=self.display_mode,
                    remaining_play_ms=self.player_remaining_seconds * 1000,
                    startup_elapsed_ms=self.wait_elapsed * 1000,
                    binding_ended=self.binding_ended,
                    segment=summary(state["segment"]),
                    model_context_segment=self.model_context_segment(state),
                    pending_replace=summary(state["pending_replace"]),
                    operation=(
                        {
                            k: v
                            for k, v in operation.items()
                            if k
                            not in {
                                "model_context_lifecycle",
                                "first_presented",
                                "first_presented_confirmation_id",
                            }
                        }
                        if operation is not None
                        else None
                    ),
                )
            )
        )

    def resolve_start(self, boundary):
        return self.retained.exact(boundary)

    async def unbind(self, reason):
        if self.binding_ended:
            return
        self.binding_ended = True
        self.current_model_stream = None
        cleanup_pending = False
        if self.current_segment and not self.current_segment.closed_event.is_set():
            try:
                await self.current_segment.stop()
            except Exception:
                cleanup_pending = True
                logging.getLogger(__name__).warning(
                    "Model cleanup continues after source unbind", exc_info=True
                )
        await self._status(
            "unbound",
            code=reason,
            source_state=self.source_state_value,
            cleanup_pending=cleanup_pending,
        )

    async def _transition_segment(self, operation, references, first):
        async with self.segment_lock:
            if not self.inference_enabled():
                return
            if self.current_segment and not self.current_segment.closed_event.is_set():
                self.current_segment.publishing = False
                await self.current_segment.stop()
            state = await self.journal.read()
            pending = state["pending_replace"]
            if pending and pending["segment_id"] == operation["segment_id"]:
                if state["segment"]["resource_state"] not in {"none", "released"}:
                    await self.journal.result(
                        operation["request_id"],
                        state="reconciling",
                        code="RELEASE_UNKNOWN",
                    )
                    await self.status_snapshot()
                    return
                await self.journal.update(
                    lambda value: value.update(
                        segment=value["pending_replace"], pending_replace=None
                    )
                )
                state = await self.journal.read()
            segment = state["segment"]
            if (
                segment is None
                or segment["segment_id"] != operation["segment_id"]
                or segment["state"] not in {"accepted", "waiting_capacity"}
                or self.binding_ended
                or not self.inference_enabled()
            ):
                return
            if segment.get("context_segment_id") and getattr(
                self.context_channel, "suspended", False
            ):
                # Recovery may fail after the original commit CAS but before
                # this child acquires its lease. Preview pixels cannot init it.
                def cancel_unstarted(value):
                    value["segment"].update(state="cancelled", resource_state="none")
                    result = value["operations"][operation["request_id"]]
                    result.update(
                        state="cancelled",
                        code="MODEL_CONTEXT_RECOVERY_REQUIRED",
                        never_applied=True,
                    )
                    return result

                await self.operation_message(
                    await self.journal.update(cancel_unstarted)
                )
                return
            leases = (
                self.startup
                if self.startup is not None
                and self.startup.ids["segment_id"] == segment["segment_id"]
                else self.segment_leases(segment)
            )
            current = RenderSegment(self, segment, references, leases)
            current.pinned_start = first
            self.current_segment = current
            self.recording.manifest["render_segments"].append(
                dict(
                    segment_id=current.segment_id,
                    segment_epoch=current.segment_epoch,
                    recording_id=current.segment_id,
                    region_id=segment["region_id"],
                )
            )
            await self.recording.configure()
            self.segment_task = asyncio.create_task(current.run())
            self.segment_task.add_done_callback(
                lambda task: task.exception() if not task.cancelled() else None
            )

    async def _finish_startup(self):
        startup = self.startup
        await startup.wait(startup.cli_ready)
        await startup.bind_cli(startup.cli_rgb, startup.cli_proof)
        if getattr(self, "closing", False) or self.binding_ended:
            return
        await self._start_prepared_segment()

    async def _start_prepared_segment(self):
        startup = self.startup
        command = dict(
            type="command",
            op="render.start",
            request_id=(
                startup.request["request_id"]
                if startup.attempt == 0
                else startup.ids["preparation_id"]
            ),
            source_session_id=self.session_id,
            source_epoch=self.source_epoch,
            expected_render_revision=self.journal.snapshot["render_revision"],
            prompt=startup.assets.prompt,
            seed=startup.assets.seed,
            references=self.initial_assets,
            start_boundary={"source_frame_index": startup.pinned_source[1][0].index},
        )
        operation = await self.journal.start(command, startup_ids=startup.ids)
        self.current_model_stream = operation["segment_epoch"]
        await self.operation_message(operation)
        await self.outgoing.put(
            msgspec.msgpack.encode(
                dict(
                    version=2,
                    type="render.stream",
                    segment_id=operation["segment_id"],
                    segment_epoch=operation["segment_epoch"],
                    media_stream_id=operation["segment_epoch"],
                    render_revision=operation["render_revision"],
                    **self.source_identity(),
                )
            )
        )
        await self._transition_segment(
            operation, startup.assets.references, startup.pinned_source
        )

    async def drain_pending_condition(self, segment, deadline):
        # A dispatched condition needs real source frames until its original
        # output/rejection arrives. Expiry preserves unknown, never application.
        clock = asyncio.get_running_loop().time
        while (
            segment.pending_prompt is not None
            and not getattr(segment, "cancel_requested", False)
            and not segment.closed_event.is_set()
            and not segment.failure_code
            and clock() < deadline
        ):
            operation = self.journal.prompt_operation(
                self.journal.snapshot,
                self.journal.snapshot["segment"],
                segment.pending_prompt,
            )
            if operation["state"] in {"applied", "failed", "rejected", "cancelled"}:
                break
            await asyncio.sleep(min(0.01, max(0, deadline - clock())))

    async def _end_segment(self, command, operation, *, deadline=None):
        async with self.segment_lock:
            current = self.current_segment
            if (
                current
                and current.segment_id == command["segment_id"]
                and not current.closed_event.is_set()
            ):
                cancel = command.get("mode") == "cancel"
                clock = asyncio.get_running_loop().time
                if deadline is None:
                    deadline = clock() + self.config.drain_seconds + 10
                if not cancel and not getattr(current, "failure_code", None):
                    await self.drain_pending_condition(current, deadline - 10)
                # Share stop's original total bound, rather than stacking a new
                # condition wait on top of its existing output/release timeout.
                async with asyncio.timeout_at(deadline):
                    await current.stop(
                        cancel=cancel or getattr(current, "cancel_requested", False)
                    )
            elif current is None or current.segment_id != command["segment_id"]:

                def cancel_unstarted(state):
                    segment = state["segment"]
                    if segment and segment["segment_id"] == command["segment_id"]:
                        segment.update(state="cancelled", resource_state="none")
                        state["operations"][segment["request_id"]].update(
                            state="cancelled", resource_state="none"
                        )

                await self.journal.update(cancel_unstarted)
                current = None
            resource_state = current.resource_state if current else "none"
            result = await self.journal.result(
                operation["request_id"],
                state=(
                    "completed"
                    if resource_state in {"none", "released"}
                    else "reconciling"
                ),
                resource_state=resource_state,
            )
            await self.operation_message(result)

    async def segment_status(self, segment, state, **fields):
        def update(value):
            current = value["segment"]
            if current and current["segment_id"] == segment.segment_id:
                # Late protocol/first-frame readiness cannot reopen admission
                # after render.end has frozen this segment's dispatch boundary.
                effective_state = (
                    "draining"
                    if current["state"] == "draining"
                    and state in {"initializing", "streaming"}
                    else state
                )
                current.update(
                    {
                        "state": effective_state,
                        "resource_state": segment.resource_state,
                        **fields,
                    }
                )
                return {
                    k: v
                    for k, v in current.items()
                    if k
                    not in {
                        "prompt_queue",
                        "prompt_events",
                        "validating_prompt",
                        "periodic_refresh",
                        "refresh_condition_version",
                    }
                }

        result = await self.journal.update(
            update, settlement=state in {"ended", "cancelled", "failed"}
        )
        if result:
            await self._status(result["state"], segment=result)

    async def segment_start_frame(self, segment, evidence):
        await self.segment_status(
            segment,
            "initializing",
            phase="source_selected",
            start_boundary_actual=evidence,
        )

    async def segment_message(self, segment, message):
        if self.binding_ended or getattr(self, "closing", False):
            return
        if (
            message.get("stream") == "model"
            and segment.segment_epoch != self.current_model_stream
        ):
            return
        message = await self.area_media(message)
        await self.outgoing.put(msgspec.msgpack.encode(message))

    async def segment_cache_receipt(self, segment, cache):
        if (
            self.current_segment is not segment
            or self.current_model_stream != segment.segment_epoch
        ):
            return
        plan_consumed = getattr(self, "prewarm_plan_consumption", {}).get(
            (segment.segment_epoch, cache["prompt_version"])
        )
        plan_store = getattr(self, "prewarm_plan_observations", None)
        if plan_consumed and plan_store:
            window, entry, operation = plan_consumed
            if (
                not window.retired
                and window.binding_identity() == window.binding
                and plan_store.window["window_id"] == window.window_id
            ):
                self.emit_prewarm_plan_observation(
                    plan_store.record_cache(window, entry, operation, cache)
                )
        consumed = getattr(self, "prewarm_consumption", None)
        store = getattr(self, "prewarm_observations", None)
        if consumed is not None and store is not None:
            window_id, formal_request, version, segment_id, segment_epoch = consumed
            if (version, segment_id, segment_epoch) == (
                cache["prompt_version"],
                segment.segment_id,
                segment.segment_epoch,
            ):
                self.emit_prewarm_observation(
                    store.record_cache(
                        window_id,
                        formal_request_id=formal_request,
                        prompt_version=version,
                        segment_id=segment_id,
                        segment_epoch=segment_epoch,
                        cache=cache,
                    )
                )
        message = dict(
            type="model.cache",
            segment_id=segment.segment_id,
            segment_epoch=segment.segment_epoch,
            **self.source_identity(),
            model_session_id=segment.assignment.session_id,
            generation_id=segment.assignment.generation_id,
            prompt_version=cache["prompt_version"],
            usage={"zing_block0_cache": cache},
        )
        # Persist bounded per-version evidence with the segment, as well as
        # sending the receipt over the existing reliable control connection.
        receipts = segment.recording.manifest.setdefault("model_cache_receipts", [])
        if not any(r["prompt_version"] == cache["prompt_version"] for r in receipts):
            receipts.append(copy.deepcopy(cache))
            del receipts[:-64]
        await self.outgoing.put(msgspec.msgpack.encode(message))

    async def segment_finished(self, segment, outcome, code):
        if getattr(segment, "sink", None) is not None:
            await segment.sink.finish()
        diagnostic = getattr(segment, "failure_diagnostic", {})
        state = (
            "ended"
            if outcome == "completed"
            else "cancelled"
            if outcome == "cancelled"
            else "failed"
        )
        await self.segment_status(
            segment,
            state,
            resource_state=segment.resource_state,
            recording_state="failed" if segment.archive_failure else "finalizing",
            error_code=code,
            **{key: value for key, value in diagnostic.items() if key != "code"},
        )

        def settle_pending(value):
            current = value["segment"]
            if not current or current["segment_id"] != segment.segment_id:
                return
            pending = list(current["prompt_queue"])
            if current.get("validating_prompt"):
                pending.append(current["validating_prompt"])
            for item in pending:
                sent = (
                    segment.pending_prompt
                    and segment.pending_prompt["request_id"] == item["request_id"]
                )
                operation = self.journal.prompt_operation(value, current, item)
                if sent:
                    operation.update(
                        state="reconciling", code="SEGMENT_ENDED_UNCONFIRMED"
                    )
                else:
                    operation.update(
                        state="cancelled", code="SEGMENT_ENDED", never_applied=True
                    )
                for event in item["one_shot_event_ids"]:
                    if event in current["prompt_events"]:
                        current["prompt_events"][event]["state"] = (
                            "unknown" if sent else "never_applied"
                        )
            current["prompt_queue"] = []
            current["validating_prompt"] = None

        await self.journal.update(settle_pending, settlement=True)
        original = self.journal.snapshot["operations"][segment.segment["request_id"]]
        execution = original.get("model_context_execution", {})
        context_result = {}
        if original.get("model_context") and execution.get("state") not in {
            "applied",
            "failed",
            "rejected",
            "cancelled",
        }:
            if getattr(segment, "model_init_dispatched", False):
                # Child cancellation/release cannot prove whether a dispatched
                # init took effect before its first output/receipt was lost.
                context_result["model_context_execution"] = dict(state="unknown")
            else:
                context_result.update(
                    never_applied=True,
                    model_context_execution=dict(state=state, never_applied=True),
                )
        result = await self.journal.result(
            segment.segment["request_id"],
            state=state,
            resource_state=segment.resource_state,
            recording_id=segment.segment_id,
            **diagnostic,
            **context_result,
        )
        await self.operation_message(result)
        if getattr(self, "inference", None) is not None:
            await self.inference.segment_ended(segment, code)

    async def before_segment_source(self, segment, ws, frame, local):
        if getattr(segment, "clip_switch", False):
            return
        # Only one candidate crosses the worker boundary at a time. The durable
        # Room queue retains later versions until actual output proves use.
        if segment.pending_prompt is not None or segment.sink.busy:
            return
        state = self.journal.snapshot
        current = state["segment"]
        if (
            not current
            or current["segment_id"] != segment.segment_id
            or current["state"] not in {"initializing", "streaming"}
        ):
            return
        if current["applied_version"] < 1:
            return
        queue = current["prompt_queue"]
        item = queue[0] if queue else current.get("validating_prompt")
        if item is None:
            return
        effective = item["effective"]
        actual = (
            frame.index if "source_frame_index" in effective else capture_pts(frame)
        )
        requested = effective.get("source_frame_index", effective.get("source_pts_us"))
        if item.get("source_context") is not None:
            from .source_conditions import source_reached

            if not source_reached(frame, item["source_context"]):
                return
        elif actual < requested:
            return
        if effective.get("policy") == "strict" and actual != requested:
            result = await self.journal.prompt_reserved(
                segment.segment_id, item["request_id"], accepted=False
            )
            await self.operation_message(result)
            return
        candidate = {**item, "strict": effective.get("policy") == "strict"}

        def remember(event_id):
            candidate["event_id"] = event_id
            segment.pending_prompt = candidate
            segment.prompt_by_event[event_id] = candidate

        await segment.send_model_event(
            ws,
            "prompt.update",
            dict(
                text=item["text"],
                version=item["version"],
                min_source_frame=local,
                policy=effective.get("policy", "at_or_after"),
            ),
            before_send=remember,
        )

    async def segment_model_control(self, segment, message):
        pending = segment.pending_prompt
        if pending is None or message.get("event_id") != pending["event_id"]:
            return False
        if (
            message.get("type") == "control_ack"
            and message.get("kind") == "prompt.update"
            and (
                message.get("state") == "kv_ready"
                or message.get("state") == "source_ready"
                and getattr(segment, "condition_early_release", False)
            )
        ):
            switch = getattr(segment, "condition_switch", None)
            boundary = switch.get("boundary") if switch else None
            if switch and boundary is None and switch["accepted"].done():
                boundary = switch["accepted"].result().get("source_frame_index")
            if switch and message.get("source_frame_index") == boundary:
                if not switch["ready"].done():
                    switch["ready"].set_result(message)
            return True
        if (
            message.get("type") == "control_ack"
            and message.get("kind") == "prompt.update"
        ):
            switch = getattr(segment, "condition_switch", None)
            if switch and not switch["accepted"].done():
                switch["accepted"].set_result(message)
            accepted = message.get("state") == "accepted"
            if pending["strict"]:
                result = await self.journal.prompt_reserved(
                    segment.segment_id,
                    pending["request_id"],
                    accepted=accepted,
                    code=message.get("code"),
                )
                await self.operation_message(result)
            elif not accepted:
                result = await self.journal.prompt_failed(
                    segment.segment_id,
                    pending["version"],
                    never_applied=True,
                    code=message.get("code", "CONDITION_INVALID"),
                )
                await self.operation_message(result)
            if not accepted:
                segment.pending_prompt = None
            return True
        if message.get("type") == "error" and message.get("fatal") is False:
            switch = getattr(segment, "condition_switch", None)
            if switch and not switch["ready"].done():
                switch["ready"].set_result({"state": "rejected"})
            result = await self.journal.prompt_failed(
                segment.segment_id,
                pending["version"],
                never_applied=True,
                code=message.get("code", "CONDITION_INVALID"),
            )
            await self.operation_message(result)
            segment.pending_prompt = None
            return True
        return False

    def emit_applied_prompt_debug(self, request_id=None):
        debug = getattr(self, "condition_debug", None)
        if debug is None or getattr(self, "world_driven", False):
            return
        segment = (self.journal.snapshot or {}).get("segment")
        if not segment or not segment["applied_version"]:
            return
        references = []
        for picture, asset in enumerate(segment["references"], 1):
            descriptor = self.assets.assets.get(asset["asset_id"])
            if descriptor is not None:
                references.append(
                    dict(
                        picture=picture,
                        format=descriptor["format"],
                        sha256=descriptor["sha256"],
                        asset=asset,
                    )
                )
        debug(
            dict(
                conditionId=f"{segment['segment_id']}:{segment['applied_version']}",
                payloadHash=hashlib.sha256(
                    canonical(
                        dict(
                            prompt=segment["prompt"],
                            references=references,
                        )
                    )
                ).hexdigest(),
                prompt=segment["prompt"],
                references=references,
            ),
            dict(
                phase="applied",
                promptVersion=segment["applied_version"],
                **({"requestId": request_id} if request_id else {}),
            ),
        )

    async def segment_prompt_output(self, segment, condition, message, frame):
        state = self.journal.snapshot
        current = state["segment"]
        version = condition["version"]
        if (
            current
            and current["segment_id"] == segment.segment_id
            and version > current["applied_version"]
        ):
            result = await self.journal.prompt_applied(
                segment.segment_id,
                version,
                dict(
                    first_output_frame=message["output_frame_start"],
                    num_frames=message["num_frames"],
                    effective_source_frame=frame.index,
                    effective_source_pts_us=capture_pts(frame),
                ),
            )
            if not result.get("periodic_refresh"):
                self.emit_applied_prompt_debug(result.get("request_id"))
            await self.operation_message(result)
            await self._status(
                "prompt_applied",
                segment_id=segment.segment_id,
                prompt_version=version,
                prompt_sha256=condition["sha256"],
                effective_source_frame=frame.index,
                effective_source_pts_us=capture_pts(frame),
            )
            if segment.pending_prompt and segment.pending_prompt["version"] == version:
                segment.pending_prompt = None
            segment.prompt_available.set()

    def request_player_finish(self, pipeline):
        """Bound normal tail drain, then use the same verified player close."""
        if self.player_finish_task is not None:
            return
        self.close_requested = True

        async def finish_player():
            deadline = time.monotonic() + 10
            finish = asyncio.create_task(self.finish())
            try:
                done, _ = await asyncio.wait((finish,), timeout=10)
                if done:
                    try:
                        finish.result()
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "Player soft drain failed"
                        )
                if pipeline.task is not None and not pipeline.task.done():
                    await asyncio.wait(
                        (pipeline.task,), timeout=max(0, deadline - time.monotonic())
                    )
                await pipeline.close()
            finally:
                if not finish.done():
                    finish.cancel()
                await asyncio.gather(finish, return_exceptions=True)

        # The closer must not be a control child which its own teardown cancels.
        self.player_finish_task = asyncio.create_task(finish_player())
        self.player_finish_task.add_done_callback(
            self.host.runtime._observe_player_cleanup
        )

    async def close(self):
        if self._source_close_task is not None and self._source_close_task.done():
            if (
                self._source_close_task.cancelled()
                or self._source_close_task.exception() is not None
            ):
                self._source_close_task = None
        if self._source_close_task is None:
            self.closing = True
            self._source_close_task = asyncio.create_task(self._close_with_retry())
        await asyncio.shield(self._source_close_task)

    async def _close_with_retry(self):
        # Browser teardown has no caller left to retry. Keep the original SDK
        # and model owners alive for bounded reconciliation, inside the shared
        # close flight. Only their positive receipts may clear quarantine.
        binding = self.model_binding
        for delay in (1, 3, None):
            try:
                await self._close_source()
            except RenderError as error:
                # Cancellation can finish the original in-flight admission
                # during cleanup, before either active assignment was published.
                lease = getattr(binding, "lease", None)
                assignment = (
                    getattr(binding, "assignment", None)
                    or getattr(lease, "_active_assignment", None)
                    or getattr(lease, "_cleanup_assignment", None)
                )
                if (
                    error.code != "RELEASE_UNKNOWN"
                    or delay is None
                    or self.model_binding is not binding
                    or not getattr(lease, "strict_release", False)
                    or getattr(assignment, "player_expires_at", None) is None
                    or lease.resource_state == "none"
                ):
                    if error.code == "RELEASE_UNKNOWN":
                        await self._settle_source_close(uncertain=True)
                    raise
                logging.getLogger(__name__).warning(
                    "Retrying original player cleanup source=%s delay_s=%s",
                    self.session_id,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                await self._settle_source_close(uncertain=False)
                return

    async def _settle_source_close(self, *, uncertain):
        def terminal(state):
            for operation in state["operations"].values():
                if (
                    operation["op"] in {"source.close", "source.unbind"}
                    and operation["state"] == "accepted"
                ):
                    operation.update(
                        state="failed" if uncertain else "completed",
                        source_state="ended",
                        resource_state="quarantined" if uncertain else "released",
                    )
                    if uncertain:
                        operation["code"] = "RELEASE_UNKNOWN"

        if self.journal.snapshot and any(
            operation["op"] in {"source.close", "source.unbind"}
            and operation["state"] == "accepted"
            for operation in self.journal.snapshot["operations"].values()
        ):
            await self.journal.update(terminal, settlement=True)

    async def _close_source(self):
        # SDK input/view retirement starts independently of a stuck model child.
        sdk = None
        if self.model_binding is not None:
            assignment = (
                self.model_binding.assignment
                or self.model_binding.lease._active_assignment
            )
            if assignment is not None:
                sdk = asyncio.create_task(self._sdk_cleanup_for(assignment))
        failures = []
        try:
            pause = getattr(self, "source_pause", None)
            if pause is not None:
                try:
                    await pause.close()
                except Exception as error:
                    failures.append(error)
            if getattr(self, "inference", None) is not None:
                try:
                    await self.inference.close()
                except Exception as error:
                    failures.append(error)
            # Stop capture users before invalidating their context channel.
            stages = [task for task in self.tasks if task is not asyncio.current_task()]
            for task in stages:
                task.cancel()
            await asyncio.gather(*stages, return_exceptions=True)
            if getattr(self, "context_channel", None) is not None:
                try:
                    await self.context_channel.close()
                except Exception as error:
                    failures.append(error)
            if self.current_segment and not self.current_segment.closed_event.is_set():
                try:
                    await self.current_segment.stop(cancel=True)
                except Exception as error:
                    failures.append(error)
            for owner in (self.startup, self.control_prefill, self.model_binding):
                if owner is not None:
                    try:
                        await owner.close()
                    except Exception as error:
                        failures.append(error)
            for task in list(self.control_tasks):
                task.cancel()
            await asyncio.gather(*self.control_tasks, return_exceptions=True)
            if sdk is not None:
                try:
                    await sdk
                except Exception as error:
                    failures.append(error)
        except BaseException as error:
            failures.append(error)
            raise
        finally:
            try:
                await super().close()
            except Exception as error:
                failures.append(error)
            verified = (
                self.model_binding is not None
                and self.model_binding.lease.resource_state == "released"
            )
            uncertain = (bool(failures) and not verified) or (
                self.model_binding is not None
                and self.model_binding.lease.resource_state not in {"none", "released"}
            )
            if failures and verified:
                logging.getLogger(__name__).warning(
                    "Player release verified after subordinate close errors source=%s errors=%s",
                    self.session_id,
                    [type(error).__name__ for error in failures],
                )
            self.source_state_value = "ended"
            if self.journal.snapshot and (
                self.journal.snapshot["source_state"] != "ended"
                or self.journal.snapshot.get("cleanup_state")
                != ("quarantined" if uncertain else "released")
            ):

                def terminal(state):
                    state.update(
                        source_state="ended",
                        source_outcome=self.outcome,
                        cleanup_state="quarantined" if uncertain else "released",
                        cleanup_warnings=len(failures) if verified else 0,
                    )

                await self.journal.update(terminal, settlement=True)
        if uncertain:
            raise RenderError("RELEASE_UNKNOWN") from (
                failures[0] if failures else None
            )
        self.retained.clear()
        if self.assets is not None:
            self.assets.release_memory()
