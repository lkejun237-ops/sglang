# SPDX-License-Identifier: Apache-2.0
"""Worker-owned continuous RGB session; reuse codec and source-ready kernels."""

import json
import os
import time
from collections import deque
from contextlib import contextmanager, nullcontext

import psutil
import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState
from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange
from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
    SourceLedger,
    SourceLimits,
    default_source_encoder,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .incremental import SourceReadySession
from .source_bridge import SourceChunkBridge
from .source_decoder import SourceTAEH3Decoder
from .source_encoder import create_source_buffer
from .streaming_codecs import StableVideoDecoder
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
        self.kv_lease = None

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
        self.qwen_init_host_ms = None

        @contextmanager
        def prepare_use(name):
            start = time.perf_counter()
            with stage._use(name) as component:
                yield component
            if name == "text_encoder":
                # Includes weight activation and condition encoding; host duration
                # does not imply CUDA completion.
                self.qwen_init_host_ms = (time.perf_counter() - start) * 1000

        source_preview = None
        command = batch.extra["zing_source_command"]
        if command.get("source_preview", False):
            # Read S0 from the same accepted bytes later fed to the source VAE.
            # This bounded CPU tensor crosses neither a file nor a second IPC.
            payload = command["appends"][0]["payload"]
            width, height = config["width"], config["height"]
            pixels = torch.frombuffer(
                bytearray(payload["data"][: width * height * 3]), dtype=torch.uint8
            ).reshape(height, width, 3)
            source_preview = dict(
                type="video",
                id="<Video 1>",
                clip=dict(start=0, end=0),
                frames=pixels.permute(2, 0, 1).unsqueeze(1),
                fps=float(config["fps"]),
            )
        self.inputs = stage.codecs.prepare(
            stage._build_request(batch),
            prepare_use,
            retain_condition_media=True,
            source_preview=source_preview,
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
        encoder = batch.extra["zing_source_command"].get(
            "encoder", default_source_encoder()
        )
        self.policy["source_encoder"] = encoder
        with self._encoder_use(stage):
            self.pixels = create_source_buffer(
                stage.codecs, encoder, capacity_frames=3 * clip
            )
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
            acquire_storage=self._pooled_storage(stage, batch.session, sampler),
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

    def _pooled_storage(self, stage, session, sampler):
        """Lease a slot at the first step; the ready session never sees the pool."""
        if stage.kv_pool is None:
            return None
        sampler.prefix_workspace = stage.kv_pool.workspace

        def acquire(positive, ready):
            request = stage.kv_request(
                sampler,
                self.inputs,
                positive,
                source_shape=ready.source.shape,
                total_frames=None,
            )
            self.kv_lease = stage.kv_pool.acquire(session, request)
            return self.kv_lease.kv_cache

        return acquire

    @torch.no_grad()
    def run(self, stage, batch):
        if self.closed or batch.block_idx != self.command_index:
            raise ValueError("Continuous source commands must be consecutive")
        self.session_id = batch.realtime_session_id
        command = batch.extra["zing_source_command"]
        measure_cuda = command.get("measure_cuda", False)
        timings, outputs, metadata = {}, [], []
        self.progress = []  # Per-dispatch, bounded by input/bridge capacities.

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
            if type(measure_cuda) is not bool:
                raise ValueError("Source CUDA measurement flag must be boolean")
            if self.ledger is None:
                config = dict(command["config"])
                config["limits"] = SourceLimits(**config["limits"])
                with timed("prepare_host_ms"):
                    self.initialize(stage, batch, config)
                if self.qwen_init_host_ms is not None:
                    timings["qwen_init_host_ms"] = self.qwen_init_host_ms
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
                    ).permute(3, 0, 1, 2)
                # The shared factory may advance its cache in append/finish,
                # not only pop. Keep the declared component active throughout.
                with timed("encoder_host_ms"), self._encoder_use(stage):
                    self.pixels.append(
                        pixels,
                        frame_start=payload["source_frame_start"],
                    )
            final = command.get("end")
            if final is not None:
                self.ledger.finish(final["frame_end"], event_id=final["event_id"])
                with timed("encoder_host_ms"), self._encoder_use(stage):
                    self.pixels.finish()
            while self.pixels.ready:
                with timed("encoder_host_ms"), stage._use("video_vae"):
                    encoded = self.pixels.pop()
                    if (
                        encoded is None
                        or encoded.values.ndim != 5
                        or encoded.start != self.bridge.accepted_latents
                        or encoded.end != encoded.start + encoded.values.shape[2]
                        or encoded.source_dependency_end > self.ledger.accepted_frames
                    ):
                        raise ValueError(
                            "Source encoder returned an invalid stable slice"
                        )
                    source = stage.codecs.sample_streaming_visual_moments(
                        encoded.values,
                        latent_start=encoded.start,
                        noise_version=STREAMING_NOISE_VERSION,
                    )
                    if measure_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                self._record_progress(
                    "latent_ready" if measure_cuda else "latent_submitted",
                    latent_start=encoded.start,
                    latent_end=encoded.end,
                    source_dependency_end=encoded.source_dependency_end,
                )
                with timed("bridge_host_ms"):
                    clip_index, last_offset = divmod(
                        encoded.end - 1, self.bridge.clip_latents
                    )
                    pixel_end = (
                        clip_index * self.bridge.clip_frames
                        + 1
                        + last_offset * self.bridge.ratio
                    )
                    if final is not None:
                        pixel_end = min(pixel_end, self.ledger.accepted_frames)
                    self.bridge.append(
                        source,
                        latent_start=encoded.start,
                        pixel_end=pixel_end,
                        source_dependency_end=encoded.source_dependency_end,
                    )
                self._generate_ready(stage, batch, outputs, metadata, timed)
                # Release only source alignment already transferred downstream.
                # Encoder lookahead/cache ownership never grants extra credit.
                self.ledger.consume(self.bridge.accepted_pixels)
            if final is not None:
                if self.bridge.accepted_pixels != self.ledger.accepted_frames:
                    raise ValueError("Source encoder did not drain its input")
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
                "zing_source_progress": self.progress,
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
            if self.policy["source_encoder"]["mode"] == "progressive":
                memory["encoder_state"] = vars(self.pixels.state_usage())
            memory["host_rss_bytes"] = psutil.Process().memory_info().rss
            if torch.cuda.is_available():
                memory.update(
                    gpu_allocated_bytes=torch.cuda.memory_allocated(),
                    gpu_reserved_bytes=torch.cuda.memory_reserved(),
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
            # Diagnostic only; synthetic samplers in tests have no KV cache.
            counters = getattr(self.sampler, "kv_counters", None)
            if counters is not None:
                memory.update(counters())
            if self.kv_lease is not None and not self.kv_lease.released:
                memory["kv_slot_id"] = self.kv_lease.slot_id
            logger.info(
                "zing_source_step %s",
                json.dumps(
                    {
                        "session_id": batch.realtime_session_id,
                        "request_id": batch.request_id,
                        "dispatch_index": batch.block_idx,
                        "rank": (
                            torch.distributed.get_rank()
                            if torch.distributed.is_initialized()
                            else 0
                        ),
                        "pid": os.getpid(),
                        "source_frames": self.ledger.accepted_frames,
                        "output_frames": self.ledger.output_frames,
                        "ended": final is not None,
                        "stage_host_ms": timings,
                        "progress": self.progress,
                        "first_source_received_unix_ns": command.get(
                            "first_source_received_unix_ns"
                        ),
                        "dispatch_unix_ns": command.get("dispatch_unix_ns"),
                        "source_preview": command.get("source_preview", False),
                        "source_wait_host_ms": command.get("source_wait_host_ms"),
                        "source_preview_rgb_sha256": command.get(
                            "source_preview_sha256"
                        ),
                        "source_fps": self.ledger.fps,
                        "measure_cuda": measure_cuda,
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

    def _encoder_use(self, stage):
        # The legacy buffer only copies raw CPU pixels on append/finish; do not
        # introduce an extra weight activation into the frozen A/B baseline.
        return (
            stage._use("video_vae")
            if self.policy["source_encoder"]["mode"] == "progressive"
            else nullcontext()
        )

    def _record_progress(self, kind, **fields):
        # Host submission timestamps intentionally do not claim CUDA completion.
        # GPU-ready latency must come from separately synchronized benchmark
        # measurements or CUDA events, with the measurement mode frozen first.
        self.progress.append(
            dict(
                kind=kind,
                host_unix_ns=time.time_ns(),
                source_frames=self.ledger.accepted_frames,
                consumed_frames=self.ledger.consumed_frames,
                **fields,
            )
        )

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
                self._record_progress(
                    "dit_started",
                    chunk_id=chunk.chunk.chunk_id,
                    latent_start=chunk.chunk.start,
                    latent_end=chunk.chunk.end,
                    source_dependency_end=chunk.provenance.source_dependency_end,
                )
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
        try:
            self._dispose_states()
        finally:
            # Return the slot only after the ready session cleared its cache.
            if self.kv_lease is not None:
                self.kv_lease.release()
                self.kv_lease = None

    def _dispose_states(self):
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
            if (
                self.pixels is not None
                and self.policy["source_encoder"]["mode"] == "progressive"
            ):
                memory["encoder_state"] = vars(self.pixels.state_usage())
            logger.info(
                "zing_source_disposed %s",
                json.dumps(
                    dict(
                        session_id=getattr(self, "session_id", None),
                        rank=(
                            torch.distributed.get_rank()
                            if torch.distributed.is_initialized()
                            else 0
                        ),
                        pid=os.getpid(),
                        memory=memory,
                    )
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
