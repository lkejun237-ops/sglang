# SPDX-License-Identifier: Apache-2.0
"""Existing Encoder input contract, returning the dedicated DiT's latents."""

from copy import copy

import msgspec.msgpack

from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    RealtimeChunkInputs,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    empty_frame_send_stats,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange, require
from sglang.multimodal_gen.runtime.realtime.zing_source_stream import CLIP_BARRIER_V2

from .zing_h3_source_adapter import ZingH3SourceStreamAdapter
from .zing_source_encoder_adapter import ZingSourceEncoderAdapter


class ZingEncoderDiTAdapter(ZingSourceEncoderAdapter):
    supports_sink_updates = True
    # The Encoder has no local Decoder config. Admission is completed against
    # the real DiT handshake below; an old peer cannot silently skip the fence.
    external_condition_barrier = True

    async def on_init(self, session, request, *, prefill=False):
        import asyncio

        self.connection_admission = asyncio.Lock()
        self._prompt_preparation_event = None
        self._selected_prompt_preparation = None
        connection_app = getattr(self, "connection_app", None)
        if connection_app is not None:
            controller = connection_app.state.worker_connections
            # Registration and the connected check share quiesce's state lock.
            # Publish the initializer before its first asynchronous validation.
            with controller.lock:
                require(
                    controller.state == "connected",
                    "The worker connections are not ready.",
                )
                self.connection_loop = asyncio.get_running_loop()
                self.connection_draining = False
                connection_app.state.connection_adapter = self
        async with self.connection_admission:
            await super().on_init(session, request, prefill=prefill)
            # No Qwen/VAE invocation and no wait for S0: forward session parameters
            # as their own ZMQ message before accepting source input.
            if prefill:
                return
            opened = await session.execution_client.open_dit(
                dict(
                    session_id=session.id,
                    generation_id=session.generation_id,
                    async_inputs=True,
                    return_stage_timing=bool(
                        getattr(request, "return_stage_timing", False)
                    ),
                    detailed_timing=self.config.get("detailed_timing", False),
                    seed=request.seed
                    if request.seed is not None
                    else ZingH3SamplingParams.seed,
                    width=self.source_geometry.target_width,
                    height=self.source_geometry.target_height,
                    source_stream=dict(
                        width=self.config["width"],
                        height=self.config["height"],
                        fps=self.config["fps"],
                        format="rgb24",
                    ),
                    source_encoding=self.source_geometry.receipt(),
                    fps=self.config["fps"],
                    total_frames=self.config["total_frames"],
                    realtime_output_format=request.realtime_output_format,
                    output_compression=getattr(request, "output_compression", None),
                    realtime_preview_max_width=getattr(
                        request, "realtime_preview_max_width", None
                    ),
                    trace_id=getattr(session, "trace_id", None),
                    **(
                        {"source_condition_switch_protocol": CLIP_BARRIER_V2}
                        if self.condition_switch_protocol == CLIP_BARRIER_V2
                        else {}
                    ),
                )
            )
            self.output_kind = next(
                (
                    v["output_kind"]
                    for v in (opened or [])
                    if v["type"] == "session.opened"
                ),
                "dit_latents",
            )
            if self.condition_switch_protocol:
                require(
                    any(
                        value.get("source_condition_switch_protocol")
                        == "clip-barrier-v1"
                        and (
                            self.condition_switch_protocol != CLIP_BARRIER_V2
                            or value.get("async_inputs") is True
                        )
                        for value in (opened or [])
                        if value.get("type") == "session.opened"
                    ),
                    "The connected model does not support condition switching.",
                )
            if self.startup_enabled:
                require(
                    any(
                        self.startup_protocol
                        in value.get("source_prepare_protocols", [])
                        for value in (opened or [])
                        if value.get("type") == "session.opened"
                    ),
                    "The connected model does not support source preparation.",
                )
            self.timing_revision = 0
            self.timing_updates = any(
                v.get("timing_updates") is True
                for v in (opened or [])
                if v["type"] == "session.opened"
            )
            self.ready.update(
                output_kind=self.output_kind, timing_updates=self.timing_updates
            )
            for value in opened or []:
                if value["type"] != "session.opened":
                    continue
                if (
                    self.condition_switch_protocol == "clip-barrier-v1"
                    and getattr(request, "source_condition_early_release", False)
                    and value.get("async_inputs") is True
                ):
                    self.ready["source_condition_early_release"] = True
                if (
                    value.get("source_output_release_protocol")
                    == "valid-source-frames-v1"
                ):
                    self.ready["source_output_release_protocol"] = value[
                        "source_output_release_protocol"
                    ]
                    if self.source_encoder["mode"] == "progressive" and self.clip == 17:
                        flow = value.get("source_flow")
                        if flow is not None:
                            self.ready["source_flow"] = flow
            self.output_attachment = next(
                (
                    v.get("output_attachment")
                    for v in (opened or [])
                    if v["type"] == "session.opened"
                ),
                None,
            )
            if self.output_attachment:
                self.ready.update(
                    output_transport="decoder_websocket",
                    output_attachment=self.output_attachment,
                )
            self.latents_sent = 0

    def _condition_control_ready(self):
        command = self._pending_prompt_command
        if command is None or "source_boundary" not in command:
            return False
        boundary = command["source_boundary"]
        # Encoder dispatch is the local watermark. The ordered, synchronous
        # DiT barrier drains all earlier messages and checks the true committed
        # GPU boundary. Waiting for a future source batch here would deadlock
        # Room, which has deliberately stopped at this complete codec clip.
        return (
            self.end is None
            and not self.more_work
            and not self.pending
            and not self.input_arranger.buffered_frames
            and self.ledger.accepted_frames == boundary
            and self.dispatched_source_frames == boundary
        )

    def _has_dit_output(self):
        app = getattr(self, "connection_app", None)
        client = getattr(getattr(app, "state", None), "realtime_execution_client", None)
        link = getattr(getattr(client, "executor", None), "dit_link", None)
        return (
            link is not None
            and self.dispatched_source_frames > 0
            and not self.end_dispatched
            and getattr(link, "has_output_notifications", lambda: False)()
        )

    def _condition_prepare_ready(self):
        command = self._pending_prompt_command
        return (
            (
                getattr(self, "condition_switch_protocol", None) == CLIP_BARRIER_V2
                or getattr(self, "ready", {}).get("source_condition_early_release")
                is True
            )
            and command is not None
            and "source_boundary" in command
            and not command.get("rejected")
            and command["event_id"] != getattr(self, "_prompt_preparation_event", None)
            and self.dispatched_source_frames > 0
            and self.end is None
            and not self.more_work
            and not self.pending
            and getattr(self, "_pending_sink_command", None) is None
            and getattr(self, "_pending_sink_auto_command", None) is None
            and not self._condition_control_ready()
        )

    def has_control_work(self):
        if self.startup_enabled and self.startup_phase != "attached":
            return self.startup_command is not None
        if not self.condition_switch_protocol:
            return super().has_control_work() or self._has_dit_output()
        return (
            self._condition_control_ready()
            or self._condition_prepare_ready()
            or self._has_dit_output()
            or (
                (
                    getattr(self, "_pending_sink_command", None) is not None
                    or getattr(self, "_pending_sink_auto_command", None) is not None
                )
                and self.dispatched_source_frames > 0
                and not self.end_dispatched
            )
        )

    async def wait_for_next_chunk(self, session):
        if self.startup_enabled and self.startup_phase != "attached":
            return await ZingH3SourceStreamAdapter.wait_for_next_chunk(self, session)
        return await super().wait_for_next_chunk(session)

    def is_control_work(self, session):
        # Like startup preparation, this RPC does not consume a source command
        # index or max_chunks budget and cannot overlap another active dispatch.
        return self._condition_prepare_ready() or super().is_control_work(session)

    def clear_state(self, session):
        self._prompt_preparation_event = None
        self._selected_prompt_preparation = None
        super().clear_state(session)
        app = getattr(self, "connection_app", None)
        if app is not None and getattr(app.state, "connection_adapter", None) is self:
            app.state.connection_adapter = None

    async def ingest_wire_event(self, ws, session, data, message_bytes):
        async with self.connection_admission:
            app = getattr(self, "connection_app", None)
            if getattr(self, "connection_draining", False) or (
                app is not None and app.state.worker_connections.state != "connected"
            ):
                await ws.send_bytes(
                    msgspec.msgpack.encode(
                        dict(
                            type="error",
                            fatal=False,
                            code="CONNECTION_DRAINING",
                            content="The connection is closing. Please wait before starting again.",
                        )
                    )
                )
                return True
            if data.get("type") == "event" and data.get("kind") == "timing.configure":
                await self._configure_timing(ws, session, data, message_bytes)
                return True
            return await super().ingest_wire_event(ws, session, data, message_bytes)

    async def _configure_timing(self, ws, session, data, message_bytes):
        from sglang.multimodal_gen.runtime.utils.zing_ttff_timing import (
            validate_timing_update,
        )

        payload, event_id = data.get("payload"), data.get("event_id")
        require(
            self.timing_updates and not self.ledger.closed and self.end is None,
            "Timing control is unavailable for this session.",
        )
        require(
            message_bytes <= self.max_message_bytes
            and isinstance(payload, dict)
            and set(payload)
            == {"session_id", "generation_id", "enabled", "timing_revision"}
            and payload.get("session_id") == session.id
            and payload.get("generation_id") == session.generation_id
            and type(event_id) is int
            and event_id > self.ledger.event_id,
            "Please refresh the session status before changing timing.",
        )
        try:
            validate_timing_update(payload, self.timing_revision)
        except ValueError:
            require(False, "Please refresh the timing status before changing it.")
        self.ledger.event_id = event_id
        receipt = await session.execution_client.configure_timing(payload)
        self.timing_revision = payload["timing_revision"]
        self.config["detailed_timing"] = payload["enabled"]
        session.mark_client_activity()
        await ws.send_bytes(
            msgspec.msgpack.encode(
                dict(
                    type="control_ack",
                    kind="timing.configure",
                    event_id=event_id,
                    model_session_id=session.id,
                    generation_id=session.generation_id,
                    enabled=receipt["enabled"],
                    timing_revision=receipt["timing_revision"],
                    boundaries=receipt["boundaries"],
                )
            )
        )

    async def drain_connection(self):
        import asyncio

        async with self.connection_admission:
            self.connection_draining = True
            if not hasattr(self, "ledger") or self.ledger.closed:
                return
            if self.end is None and self.ledger.accepted_frames:
                # Administrative EOS may end a finite request before its declared
                # target. Public source.end keeps its original strict validation.
                self.ledger.total_frames = self.ledger.accepted_frames
                event_id = self.ledger.event_id + 1
                self.ledger.finish(self.ledger.accepted_frames, event_id=event_id)
                self.end = dict(
                    frame_end=self.ledger.accepted_frames, event_id=event_id
                )
                self.pending.append(dict(kind="source.end", end=self.end))
                self.available.set()
        while (
            not self.ledger.closed
            and self.ledger.accepted_frames
            and not getattr(self, "connection_eos_done", False)
        ):
            await asyncio.sleep(0.005)

        self.ledger.dispose()

    def sample_chunk_inputs(self, session, server_args, chunk, chunk_size):
        if getattr(chunk, "control_only", False) and self._condition_prepare_ready():
            self._selected_prompt_preparation = copy(self._pending_prompt_command)
            self._prompt_preparation_event = self._pending_prompt_command["event_id"]
            # Keep old source batches on their old condition until the complete
            # clip boundary. Prepared is neither accepted activation nor applied.
            return RealtimeChunkInputs(prompt=session.request.prompt)
        return super().sample_chunk_inputs(session, server_args, chunk, chunk_size)

    def prepare_next_request(self, session, server_args, chunk):
        batch = super().prepare_next_request(session, server_args, chunk)
        if getattr(self, "_selected_prompt_preparation", None) is not None:
            batch.extra["zing_prompt_prepare"] = self._selected_prompt_preparation
            self._selected_prompt_preparation = None
        if self.ready.get("source_condition_early_release"):
            update = batch.extra.get("zing_prompt_update")
            if update is not None and "source_boundary" in update:
                update["allow_early_release"] = True
        batch.extra["zing_dit_source_records"] = [
            [r.source_id, r.start, r.end] for r in self.ledger.records
        ]
        return batch

    def _take_source_input(self):
        if getattr(self, "_selected_prompt_preparation", None) is not None:
            # A source group may arrive after the control dispatch was selected.
            # Leave it queued; preparation must not consume or relabel its frames.
            return [], None
        appends, end = super()._take_source_input()
        if self.output_attachment:
            # Execution now owns these inputs. Decoder can return client credit
            # before this dispatch returns to the Encoder API, so update local
            # admission here. Only Decoder publishes source.consumed on the wire.
            self.ledger.consume(self.dispatched_source_frames)
        return appends, end

    async def send_output(self, ws, session, result, batch):
        try:
            return await self._send_connected_output(ws, session, result, batch)
        except BaseException:
            app = getattr(self, "connection_app", None)
            if app is not None:
                app.state.worker_connections.failed()
            raise
        finally:
            if result.usage.get("zing_source_complete"):
                self.connection_eos_done = True

    async def _send_connected_output(self, ws, session, result, batch):
        total = empty_frame_send_stats()
        if result.usage.get("zing_startup_ready") is not None:
            ready = dict(result.usage["zing_startup_ready"])
            reports = result.realtime_handoff.get("timing_reports", [])
            if reports:
                ready["timing_reports"] = [
                    dict(
                        model_session_id=session.id,
                        generation_id=session.generation_id,
                        timing=report,
                    )
                    for report in reports
                ]
            await ws.send_bytes(msgspec.msgpack.encode(ready))
            return total
        for notice in result.realtime_handoff.get("conditioning_notices", []):
            await ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="error",
                        fatal=False,
                        code=notice["code"],
                        event_id=notice["event_id"],
                        content="The prompt update could not be applied. Please try again.",
                    )
                )
            )
        source_ready = result.usage.get("zing_condition_source_ready")
        if source_ready is not None:
            await ws.send_bytes(msgspec.msgpack.encode(source_ready))
        for item in result.realtime_handoff["dit_latents"]:
            require(
                item["latent_start"] == self.latents_sent, "Invalid DiT latent order."
            )
            require(
                (item["session_id"], item["generation_id"])
                == (session.id, session.generation_id),
                "Invalid DiT result identity.",
            )
            pixels = PixelRange(
                item["output_frame_start"],
                item["output_frame_start"] + item["num_frames"],
            )
            provenance = self.ledger.provenance_for(
                pixels, item["source_dependency_end"]
            )
            expected = provenance.metadata()
            require(
                all(item[k] == value for k, value in expected.items()),
                "Invalid DiT result alignment.",
            )
            if self.output_attachment:
                require(
                    item["type"] in ("decoder.output_queued", "decoder.input_sent")
                    and item.get("reference") is None,
                    "Invalid direct Decoder receipt.",
                )
                # Pixels bypass this input WS, so its zero send counters cannot
                # arm the hard lifetime. Start it at the first validated server
                # dispatch; client heartbeats/withheld playback ACKs cannot
                # retain the single KV owner forever after output disconnect.
                session.mark_playable()
            elif self.output_kind == "rgb":
                from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
                    OutputBatch,
                )

                require(item["type"] == "decoder.frames", "Invalid video output.")
                sub_batch = copy(batch)
                sub_batch.extra = dict(
                    batch.extra,
                    zing_source_output_blocks=[
                        dict(
                            item,
                            provenance=provenance,
                            request_id=batch.request_id,
                            block_idx=item["block_idx"],
                        )
                    ],
                )
                decoded = OutputBatch(
                    raw_frame_shared_memory_ref=item["reference"],
                    raw_frame_metadata=item["metadata"],
                )
                try:
                    stats = await self.output_adapter.send(
                        ws, session, decoded, sub_batch
                    )
                finally:
                    # The original output adapter consumes or cancels ownership.
                    item["reference"] = decoded.raw_frame_shared_memory_ref
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        total[key] += value
                    elif value:
                        total[key] = value
            else:
                require(item["type"] == "dit.latents", "Invalid latent output.")
                await ws.send_bytes(msgspec.msgpack.encode(item))
            self.ledger.delivered(pixels)
            self.latents_sent = item["latent_end"]
        if not self.output_attachment:
            consumed = self.ledger.consume(result.usage["zing_source_consumed_end"])
            if consumed is not None:
                await ws.send_bytes(msgspec.msgpack.encode(consumed))
        if result.usage.get("zing_condition_ready") is not None:
            await ws.send_bytes(
                msgspec.msgpack.encode(result.usage["zing_condition_ready"])
            )
        for report in result.realtime_handoff.get("timing_reports", []):
            await ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="model.timing",
                        model_session_id=session.id,
                        generation_id=session.generation_id,
                        timing=report,
                    )
                )
            )
        for report in result.realtime_handoff.get("cache_reports", []):
            from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
                public_cache_receipt,
            )

            public = public_cache_receipt(report)
            if public is not None:
                await ws.send_bytes(
                    msgspec.msgpack.encode(
                        dict(
                            type="model.cache",
                            model_session_id=session.id,
                            generation_id=session.generation_id,
                            prompt_version=public["prompt_version"],
                            usage={"zing_block0_cache": public},
                        )
                    )
                )
        if result.usage["zing_source_complete"]:
            completion = self.ledger.completed()
            if self.output_kind == "rgb":
                if self.output_attachment:
                    completion = dict(completion, type="input.completed")
                await ws.send_bytes(msgspec.msgpack.encode(completion))
                return total
            await ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="session.completed",
                        output_kind="dit_latents",
                        session_id=session.id,
                        generation_id=session.generation_id,
                        source_frames=self.ledger.output_frames,
                        latent_frames=self.latents_sent,
                    )
                )
            )
        return total
