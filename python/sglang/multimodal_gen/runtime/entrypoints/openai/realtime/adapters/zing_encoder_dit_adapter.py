# SPDX-License-Identifier: Apache-2.0
"""Existing Encoder input contract, returning the dedicated DiT's latents."""

from copy import copy

import msgspec.msgpack

from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    empty_frame_send_stats,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange, require

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
                        == self.condition_switch_protocol
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
            self.ready.update(output_kind=self.output_kind)
            for value in opened or []:
                if value["type"] != "session.opened":
                    continue
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

    def has_control_work(self):
        if self.startup_enabled and self.startup_phase != "attached":
            return self.startup_command is not None
        if not self.condition_switch_protocol:
            return super().has_control_work() or self._has_dit_output()
        return (
            self._condition_control_ready()
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

    def clear_state(self, session):
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
            return await super().ingest_wire_event(ws, session, data, message_bytes)

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

    def prepare_next_request(self, session, server_args, chunk):
        batch = super().prepare_next_request(session, server_args, chunk)
        batch.extra["zing_dit_source_records"] = [
            [r.source_id, r.start, r.end] for r in self.ledger.records
        ]
        return batch

    def _take_source_input(self):
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
            await ws.send_bytes(
                msgspec.msgpack.encode(result.usage["zing_startup_ready"])
            )
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
        for report in result.realtime_handoff.get("cache_reports", []):
            from sglang.multimodal_gen.runtime.realtime.whitebox.cache_receipt import (
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
