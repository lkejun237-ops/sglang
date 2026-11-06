# SPDX-License-Identifier: Apache-2.0
"""Source-VAE and Qwen output on the existing realtime source input contract.

This staged endpoint does not run reference encoding, DiT or decoding.
The original video adapter and its capabilities remain unchanged.
"""

import asyncio

import msgspec.msgpack

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    empty_frame_send_stats,
)
from sglang.multimodal_gen.runtime.realtime.polling_task_queue import PollingTaskQueue
from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange, require
from sglang.multimodal_gen.runtime.realtime.zing_encoder_input import (
    EncoderInputArranger,
)

from .zing_h3_source_adapter import ZingH3SourceStreamAdapter


class ZingSourceEncoderAdapter(ZingH3SourceStreamAdapter):
    supports_sink_updates = False

    def _initialize_source_admission(self, session, limits):
        # Inputs already belong to this process; remote prefetch IPC would
        # duplicate delivery and wait for a peer which does not exist here.
        self.source_encoder = dict(self.source_encoder)
        self.source_encoder.pop("remote", None)
        self.source_append_overlap = False
        self.admission_socket = self.admission_context = None

    async def on_init(self, session, request, *, prefill=False):
        require(not request.profile, "Profiling is not supported by this endpoint yet.")
        await super().on_init(session, request, prefill=prefill)
        self.input_arranger = EncoderInputArranger(
            frame_bytes=self.config["width"] * self.config["height"] * 3,
            fps=self.config["fps"],
        )
        self.pending = PollingTaskQueue(
            self._process_encoder_batch,
            ready=self._encoder_ready,
            arrange=self.input_arranger,
        )
        self.ready.update(
            output_kind="source_moments",
            generation_id=session.generation_id,
            prompt_updates=True,
            encoded_components=["source_vae", "qwen"],
        )
        self.latents_sent = 0

    def prepare_prefill_request(self, session, server_args, source_frame):
        from dataclasses import asdict

        batch = super().prepare_prefill_request(session, server_args, source_frame)
        batch.extra["zing_source_command"]["source_geometry"] = asdict(
            self.source_geometry
        )
        return batch

    async def ingest_wire_event(self, ws, session, data, message_bytes):
        accepted = await super().ingest_wire_event(ws, session, data, message_bytes)
        if data.get("kind") == "source.end":
            # Admission already validated EOS. Flush staging before enqueuing it.
            self.pending.append(dict(kind="source.end", end=self.end))
        return accepted

    def _take_source_input(self):
        # Consume exactly one arranged group. Prompt-only and codec-continuation
        # dispatches retain their existing behavior without consuming source.
        if self.more_work or not self.pending:
            return [], None
        task = self.pending.popleft()
        if task["kind"] == "source.end":
            self.end_dispatched = True
            return [], task["end"]
        self.pending_frames -= task["num_frames"]
        self.dispatched_source_frames += task["num_frames"]
        for seq in task["release_client_seqs"]:
            reservation = self._source_reservations.pop(seq, None)
            if reservation is not None:
                self._inflight_reservations.append(reservation)
        return task["appends"], None

    def has_control_work(self):
        # After S0 has initialized Qwen, a prompt event can run even while the
        # source producer is idle. The original adapter keeps its own policy.
        return (
            (
                self._pending_prompt_command is not None
                or getattr(self, "_pending_sink_command", None) is not None
            )
            and self.dispatched_source_frames > 0
            and not self.end_dispatched
        )

    def _encoder_ready(self, queue):
        if self.ledger.closed:
            raise asyncio.CancelledError("Source session has closed")
        # Include buffered codec work and controls even when input is exhausted.
        return self.more_work or bool(queue) or self.has_control_work()

    async def wait_for_next_chunk(self, session):
        await self.pending.wait_ready()

    @staticmethod
    async def _process_encoder_batch(queue, batch, execution_client):
        from ..realtime_video_api import _process_realtime_generation_batch

        return await _process_realtime_generation_batch(batch, execution_client)

    async def process_batch(self, batch, execution_client):
        return await self.pending.process(batch, execution_client)

    def on_chunk_complete(self, session, result):
        super().on_chunk_complete(session, result)
        # This executor always bounds output to ten slices, independently of
        # the original DiT work-quantum setting. Drain remaining codec work.
        self.more_work = (result.usage or {}).get("zing_source_more_work", False)

    def clear_state(self, session):
        super().clear_state(session)
        if hasattr(self, "input_arranger"):
            self.input_arranger.clear()

    async def send_output(self, ws, session, result, batch):
        condition = result.realtime_handoff.get("conditioning")
        if condition is not None:
            # This is an encoded condition, not a DiT prompt-activation ACK.
            await ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="conditioning",
                        session_id=session.id,
                        generation_id=session.generation_id,
                        **condition,
                    )
                )
            )
        for item in result.realtime_handoff["source_moments"]:
            require(item["latent_start"] == self.latents_sent, "Invalid latent order.")
            await ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="source.moments",
                        session_id=session.id,
                        generation_id=session.generation_id,
                        **item,
                    )
                )
            )
            # This ledger tracks source coverage. No RGB frames were generated.
            self.ledger.delivered(
                PixelRange(self.ledger.output_frames, item["source_dependency_end"])
            )
            self.latents_sent = item["latent_end"]
        consumed = self.ledger.consume(result.usage["zing_source_consumed_end"])
        if consumed is not None:
            await ws.send_bytes(msgspec.msgpack.encode(consumed))
        if result.usage["zing_source_complete"]:
            self.ledger.completed()  # Require EOS, full coverage and released credit.
            await ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="session.completed",
                        output_kind="source_moments",
                        session_id=session.id,
                        generation_id=session.generation_id,
                        source_frames=self.ledger.output_frames,
                        latent_frames=self.latents_sent,
                    )
                )
            )
        return empty_frame_send_stats()
