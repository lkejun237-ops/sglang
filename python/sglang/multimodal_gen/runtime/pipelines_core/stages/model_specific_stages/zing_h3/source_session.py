# SPDX-License-Identifier: Apache-2.0
"""Worker-owned continuous RGB session; reuse codec and source-ready kernels."""

import time
import json
import os
from collections import deque
from contextlib import contextmanager

import torch
import psutil

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState
from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange
from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
    SourceLedger,
    SourceLimits,
)

from .incremental import SourceReadySession
from .source_bridge import SourceChunkBridge
from .source_decoder import SourceTAEH3Decoder
from .streaming_codecs import SourcePixelBuffer, StableVideoDecoder
from .streaming_noise import STREAMING_NOISE_VERSION

logger = init_logger(__name__)


class ZingSourceStreamState(BaseRealtimeState):
    def __init__(self):
        self.inputs = self.sampler = self.ledger = self.pixels = self.bridge = (
            self.decoder
        ) = None
        self.blocks = deque()
        self.command_index = 0
        self.prompt = self.prompt_event = None
        self.closed = False

    def initialize(self, stage, batch, config):
        params = batch.sampling_params
        stage._validate_sampling_params(params)
        sampler = stage._build_sampler(params)
        if sampler.window_policy.window_chunks is None:
            raise ValueError("Continuous source requires a bounded attention window")
        self.policy = {
            "window_chunks": sampler.window_policy.window_chunks,
            "sink_chunks": sampler.window_policy.sink_chunks,
            "rope_position_mode": sampler.rope_position_mode,
            "rope_max_frame_gap": sampler.rope_max_frame_gap,
            "video_decoder": stage.codecs.video_decoder_name,
        }
        self.inputs = stage.codecs.prepare(
            stage._build_request(batch),
            stage._use,
            retain_condition_media=True,
        )
        if self.inputs.audio_shape is not None or self.inputs.source_visual is not None:
            raise ValueError("Continuous source requires a video-only static prefix")
        self.prompt = params.prompt
        self.ledger = SourceLedger(**config)
        clip, tokens = (
            stage.codecs.vae_frames_per_chunk,
            stage.codecs.vae_latents_per_chunk,
        )
        max_chunk = max(stage.packer.chunk_plan.lengths)
        # One input dispatch is at most two codec clips. All capacities remain
        # independent of total duration, including single-frame network appends.
        self.pixels = SourcePixelBuffer(stage.codecs, capacity_frames=3 * clip)
        self.bridge = SourceChunkBridge(
            stage.packer.chunk_plan,
            clip_frames=clip,
            clip_latents=tokens,
            temporal_ratio=stage.codecs.temporal_ratio,
            capacity_latents=max_chunk + tokens,
            capacity_bytes=256 << 20,
            seed=params.seed[0] if isinstance(params.seed, list) else params.seed,
            provenance_for=self.ledger.provenance_for,
        )
        self.sampler = SourceReadySession(
            sampler,
            self.inputs,
            seed=self.bridge.seed,
            max_pending_chunks=1,
            max_pending_bytes=256 << 20,
        )
        decoder_type = (
            SourceTAEH3Decoder
            if stage.codecs.video_decoder_name == "taeh3_decoder"
            else StableVideoDecoder
        )
        self.decoder = decoder_type(
            stage.codecs,
            capacity_latents=2 * tokens + max_chunk + tokens,
            provenance_for=self.ledger.provenance_for,
        )

    @torch.no_grad()
    def run(self, stage, batch):
        if self.closed or batch.block_idx != self.command_index:
            raise ValueError("Continuous source commands must be consecutive")
        self.session_id = batch.realtime_session_id
        command = batch.extra["zing_source_command"]
        timings, outputs, metadata = {}, [], []

        @contextmanager
        def timed(name):
            start = time.perf_counter()
            try:
                yield
            finally:
                timings[name] = (
                    timings.get(name, 0.0) + (time.perf_counter() - start) * 1000
                )

        try:
            if self.ledger is None:
                config = dict(command["config"])
                config["limits"] = SourceLimits(**config["limits"])
                with timed("prepare_host_ms"):
                    self.initialize(stage, batch, config)
            if (
                batch.sampling_params.prompt != self.prompt
                or batch.realtime_event_id != self.prompt_event
            ):
                with timed("prompt_host_ms"), stage._use("text_encoder"):
                    positive = stage.codecs.encode_condition(
                        batch.sampling_params.prompt, self.inputs.condition_media
                    )
                    self.sampler.update_prompt(positive, batch.realtime_event_id)
                self.prompt, self.prompt_event = (
                    batch.sampling_params.prompt,
                    batch.realtime_event_id,
                )
            count = sum(x["payload"]["num_frames"] for x in command["appends"])
            if count > 2 * stage.codecs.vae_frames_per_chunk:
                raise ValueError(
                    "Continuous source dispatch exceeds its bounded capacity"
                )
            for item in command["appends"]:
                payload = item["payload"]
                self.ledger.accept(
                    payload,
                    event_id=item["event_id"],
                    message_bytes=item["message_bytes"],
                )
                with timed("input_copy_host_ms"):
                    pixels = torch.frombuffer(
                        bytearray(payload["data"]), dtype=torch.uint8
                    )
                    pixels = pixels.reshape(
                        payload["num_frames"], self.ledger.height, self.ledger.width, 3
                    )
                    self.pixels.append(
                        pixels.permute(3, 0, 1, 2),
                        frame_start=payload["source_frame_start"],
                    )
            final = command.get("end")
            if final is not None:
                self.ledger.finish(final["frame_end"], event_id=final["event_id"])
                self.pixels.finish()
            while self.pixels.ready:
                with timed("encoder_host_ms"), stage._use("video_vae"):
                    encoded = self.pixels.pop()
                    source = stage.codecs.sample_streaming_visual_moments(
                        encoded.values,
                        latent_start=encoded.start,
                        noise_version=STREAMING_NOISE_VERSION,
                    )
                with timed("bridge_host_ms"):
                    self.bridge.append(
                        source,
                        latent_start=encoded.start,
                        pixel_end=encoded.source_dependency_end,
                        source_dependency_end=encoded.source_dependency_end,
                    )
                self._generate_ready(stage, batch, outputs, metadata, timed)
                self.ledger.consume(self.bridge.accepted_pixels)
            if final is not None:
                self.bridge.finish()
                self._generate_ready(stage, batch, outputs, metadata, timed)
                if not self.sampler.ended:
                    self.sampler.finish()
                if not self.sampler.complete:
                    raise ValueError("Continuous source sampling did not drain")
                self.decoder.finish(total_frames=self.ledger.accepted_frames)
                self._decode_ready(stage, outputs, metadata, timed)
                self.ledger.consume(self.bridge.accepted_pixels)
                self.ledger.completed()
            self.command_index += 1
            usage = {
                "zing_realtime_blocks": metadata,
                "zing_source_consumed_end": self.bridge.accepted_pixels,
                "zing_source_complete": final is not None,
                "zing_source_stage_host_ms": timings,
                "zing_source_memory": {
                    **self.sampler.memory_snapshot(),
                    "codec_pixel_frames": self.pixels.buffered_frames,
                    "bridge_latents": self.bridge.buffered_latents,
                    "bridge_bytes": self.bridge.buffered_bytes,
                    "decoder_latents": self.decoder.buffered_latents,
                    "metadata_batches": len(self.ledger.records),
                    "pending_blocks": len(self.blocks),
                },
            }
            memory = usage["zing_source_memory"]
            memory["host_rss_bytes"] = psutil.Process().memory_info().rss
            if torch.cuda.is_available():
                memory.update(
                    gpu_allocated_bytes=torch.cuda.memory_allocated(),
                    gpu_reserved_bytes=torch.cuda.memory_reserved(),
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                )
            logger.info(
                "zing_source_step %s",
                json.dumps(
                    {
                        "session_id": batch.realtime_session_id,
                        "request_id": batch.request_id,
                        "dispatch_index": batch.block_idx,
                        "rank": torch.distributed.get_rank()
                        if torch.distributed.is_initialized()
                        else 0,
                        "pid": os.getpid(),
                        "source_frames": self.ledger.accepted_frames,
                        "output_frames": self.ledger.output_frames,
                        "ended": final is not None,
                        "stage_host_ms": timings,
                        "policy": self.policy,
                        "memory": memory,
                    }
                ),
                main_process_only=False,
                local_main_process_only=False,
            )
            if outputs:
                batch.sampling_params.num_frames = sum(x.shape[1] for x in outputs)
                result = OutputBatch(output=outputs, metrics=batch.metrics, usage=usage)
            else:
                result = OutputBatch(
                    raw_frame_batches=[], metrics=batch.metrics, usage=usage
                )
            if final is not None:
                self.dispose()
            return result
        except BaseException:
            self.dispose()
            raise

    def _generate_ready(self, stage, batch, outputs, metadata, timed):
        while self.bridge.ready:
            with timed("bridge_host_ms"):
                chunk = self.bridge.pop()
                self.sampler.append(
                    chunk.source,
                    chunk.noise,
                    provenance=chunk.provenance,
                    final=chunk.final,
                )
            with (
                timed("dit_host_ms"),
                stage._use("transformer", "denoise") as transformer,
            ):
                self.sampler.sampler.generator = transformer
                result = self.sampler.step()
            self.blocks.append(
                {
                    "block_idx": result.chunk.chunk_id,
                    "request_id": batch.request_id,
                    "event_id": result.prompt_event_id,
                    "frame_end": result.provenance.pixels.end,
                }
            )
            self.decoder.append(
                result.visual,
                latent_start=result.chunk.start,
                source_dependency_end=result.provenance.source_dependency_end,
            )
            self._decode_ready(stage, outputs, metadata, timed)

    def _decode_ready(self, stage, outputs, metadata, timed):
        while self.decoder.ready:
            with (
                timed("decoder_host_ms"),
                stage._use(stage.codecs.video_decoder_name, "decode"),
            ):
                decoded = self.decoder.pop()
            start = decoded.start
            while start < decoded.end:
                block = self.blocks[0]
                end = min(decoded.end, block["frame_end"])
                if end <= start:
                    raise ValueError("Output control provenance is not continuous")
                pixels = PixelRange(start, end)
                outputs.append(
                    decoded.values[
                        0, :, start - decoded.start : end - decoded.start
                    ].contiguous()
                )
                metadata.append(
                    {**block, **decoded.provenance.slice(pixels).metadata()}
                )
                self.ledger.delivered(pixels)
                if end == block["frame_end"]:
                    self.blocks.popleft()
                start = end

    def dispose(self):
        for state in (
            self.sampler,
            self.ledger,
            self.pixels,
            self.bridge,
            self.decoder,
        ):
            if state is not None:
                state.dispose()
        if self.sampler is not None:
            memory = self.sampler.memory_snapshot()
            memory.update(
                codec_pixel_frames=self.pixels.buffered_frames if self.pixels else 0,
                bridge_latents=self.bridge.buffered_latents if self.bridge else 0,
                decoder_latents=self.decoder.buffered_latents if self.decoder else 0,
                metadata_batches=len(self.ledger.records) if self.ledger else 0,
            )
            logger.info(
                "zing_source_disposed %s",
                json.dumps(
                    dict(session_id=getattr(self, "session_id", None), memory=memory)
                ),
                main_process_only=False,
                local_main_process_only=False,
            )
        self.inputs = self.sampler = self.ledger = self.pixels = self.bridge = (
            self.decoder
        ) = None
        self.blocks.clear()
        self.prompt = self.prompt_event = None
        self.closed = True
