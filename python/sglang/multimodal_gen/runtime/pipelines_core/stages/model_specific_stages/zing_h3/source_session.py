# SPDX-License-Identifier: Apache-2.0
"""Worker-owned continuous RGB session; reuse codec and source-ready kernels."""

import json
import os
import time
from collections import deque
from contextlib import closing, contextmanager, nullcontext

import psutil
import torch

from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState
from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import ZingKVPoolAdmissionError
from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
    SourceLedger,
    SourceLimits,
    default_source_encoder,
    source_work_quantum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.profiler import maybe_record_function
from sglang.multimodal_gen.runtime.utils.zing_stage_timing import (
    handoff,
    stage_span,
    trace_for_batch,
)

from .incremental import SourceReadySession
from .source_bridge import SourceChunk, SourceChunkBridge
from .source_encoder import create_source_buffer
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
        self.failed_update = None
        self.closed = False
        self.component_capture = None
        self.kv_lease = None
        self.work_quantum = source_work_quantum()
        self.end_received = self.bridge_finished = False
        self.dispatch_steps = 0

    def initialize(self, stage, batch, config, trace=None):
        params = batch.sampling_params
        stage._validate_sampling_params(params)
        command = batch.extra["zing_source_command"]
        sampling_options = (
            {"source_sampling_schedule": command["sampling_schedule"]}
            if "sampling_schedule" in command
            else {}
        )
        sampler = stage._build_realtime_sampler(params, **sampling_options)
        if sampler.window_policy.window_chunks is None:
            raise ValueError("Continuous source requires a bounded attention window")
        self.policy = {
            "window_chunks": sampler.window_policy.window_chunks,
            "window_eviction_timing": getattr(
                sampler, "window_eviction_timing", "after_step0"
            ),
            "sink_chunks": sampler.window_policy.sink_chunks,
            "rope_position_mode": sampler.rope_position_mode,
            "rope_max_frame_gap": sampler.rope_max_frame_gap,
            "video_decoder": stage.codecs.video_decoder_name,
            "source_work_quantum": self.work_quantum,
        }
        if getattr(sampler, "joint_forward", "off") != "off":
            self.policy["joint_forward"] = sampler.joint_forward
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
                origin="source_s0",
            )
        self.inputs = stage.codecs.prepare(
            stage._build_request(batch),
            prepare_use,
            retain_condition_media=True,
            source_preview=source_preview,
            conditioning=stage._conditioning_consumer(batch),
            stage_trace=trace,
        )
        if self.inputs.audio_shape is not None or self.inputs.source_visual is not None:
            raise ValueError("Continuous source requires a video-only static prefix")
        self.prompt = params.prompt
        self.prompt_event = batch.realtime_event_id
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
                stage.codecs,
                encoder,
                capacity_frames=3 * clip,
                session_id=batch.realtime_session_id,
                stage_trace=trace_for_batch(batch),
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
            max_pending_chunks=2
            if getattr(sampler, "joint_forward", "off") == "cross_chunk"
            else 1,
            max_pending_bytes=256 << 20,
            acquire_storage=self._pooled_storage(stage, batch.session, sampler),
            prompt_switch_prefix_latent_frames=(
                stage.prompt_switch_prefix_latent_frames
            ),
            initial_prompt_event_id=batch.realtime_event_id,
            initial_prompt_version=getattr(batch, "realtime_prompt_version", None),
        )
        self.decoder = stage.codecs.decoder.source_decoder(
            stage.codecs,
            capacity_latents=2 * tokens + max_chunk + tokens,
            provenance_for=self.ledger.provenance_for,
        )
        capture_dir = os.environ.get("SGLANG_ZING_COMPONENT_CAPTURE_DIR")
        if capture_dir and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            from .component_capture import ComponentCapture

            self.component_capture = ComponentCapture(
                capture_dir,
                stage,
                self.inputs,
                params,
                self.bridge.seed,
                batch.realtime_session_id,
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

    def _admit_prompt_update(self, stage, condition):
        """Check the new condition before it can replace active or pending KV."""
        if getattr(stage, "kv_pool", None) is None:
            return
        # Pack one placeholder target to reuse the authoritative reference-row
        # accounting. No model forward, random draw, lease or KV write occurs.
        packer = stage.packer
        _, height, width = self.inputs.visual_shape
        visual = condition["context"].new_zeros(
            (1, packer.visual_channels, 1, height, width)
        )
        layout = packer.pack_inference_layout(self.inputs, condition, visual, None)
        request = stage.kv_request(
            self.sampler.sampler,
            self.inputs,
            layout,
            source_shape=(height, width),
            total_frames=None,
        )
        try:
            stage.kv_pool.plan.admit(request)
        except ZingKVPoolAdmissionError as exc:
            logger.warning("Prompt update rejected by KV pool: %s", exc.detail)
            raise ConditionError("CONDITION_INVALID", exc.detail) from exc

    def _update_prompt(self, stage, batch):
        """Reject invalid candidates without disturbing the active generation."""
        notice = None
        update = (batch.sampling_params.prompt, batch.realtime_event_id)
        discard = getattr(self.sampler, "discard_pending_prompt", None)
        if discard is not None:
            discard()
        try:
            positive = stage.encode_condition_for_batch(
                batch,
                batch.sampling_params.prompt,
                self.inputs.condition_media,
            )
            self._admit_prompt_update(stage, positive)
        except ConditionError as exc:
            self.failed_update = update
            notice = {"code": exc.code, "event_id": batch.realtime_event_id}
        else:
            self.sampler.update_prompt(
                positive,
                batch.realtime_event_id,
                getattr(batch, "realtime_prompt_version", None),
            )
            self.prompt, self.prompt_event = update
            self.failed_update = None
        return notice

    @torch.no_grad()
    def run(self, stage, batch):
        if self.closed or batch.block_idx != self.command_index:
            raise ValueError("Continuous source commands must be consecutive")
        self.session_id = batch.realtime_session_id
        command = batch.extra["zing_source_command"]
        measure_cuda = command.get("measure_cuda", False)
        trace = trace_for_batch(batch)
        timings, outputs, metadata = {}, [], []
        self.progress = []  # Per-dispatch, bounded by input/bridge capacities.
        self.dispatch_steps = 0

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
                    self.initialize(stage, batch, config, trace)
                if self.qwen_init_host_ms is not None:
                    timings["qwen_init_host_ms"] = self.qwen_init_host_ms
            notice = None
            update = (batch.sampling_params.prompt, batch.realtime_event_id)
            if update != getattr(self, "failed_update", None) and update != (
                self.prompt,
                self.prompt_event,
            ):
                with timed("prompt_host_ms"):
                    notice = self._update_prompt(stage, batch)
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
                if self.component_capture is not None:
                    self.component_capture.append_frames(
                        "input", pixels, payload["source_frame_start"]
                    )
                # The shared factory may advance its cache in append/finish,
                # not only pop. Keep the declared component active throughout.
                with timed("encoder_host_ms"), self._encoder_use(stage):
                    self.pixels.append(
                        pixels,
                        frame_start=payload["source_frame_start"],
                    )
            final = command.get("end")
            if final is not None:
                if self.end_received:
                    raise ValueError("Source end was already accepted")
                self.end_received = True
                self.ledger.finish(final["frame_end"], event_id=final["event_id"])
                with timed("encoder_host_ms"), self._encoder_use(stage):
                    self.pixels.finish()
            self._generate_ready(stage, batch, outputs, metadata, timed, trace)
            while self.pixels.ready and self._within_work_budget():
                with (
                    timed("encoder_host_ms"),
                    stage._use("video_vae"),
                    maybe_record_function("Zing.source_encode"),
                ):
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
                if self.component_capture is not None:
                    self.component_capture.append_source(encoded, source)
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
                    if self.end_received:
                        pixel_end = min(pixel_end, self.ledger.accepted_frames)
                    self.bridge.append(
                        source,
                        latent_start=encoded.start,
                        pixel_end=pixel_end,
                        source_dependency_end=encoded.source_dependency_end,
                    )
                self._generate_ready(
                    stage,
                    batch,
                    outputs,
                    metadata,
                    timed,
                    trace,
                    flush=not self.pixels.ready,
                )
                # Release only source alignment already transferred downstream.
                # Encoder lookahead/cache ownership never grants extra credit.
                self.ledger.consume(self.bridge.accepted_pixels)
            completed = False
            if self.end_received and not self.pixels.ready:
                if self.bridge.accepted_pixels != self.ledger.accepted_frames:
                    raise ValueError("Source encoder did not drain its input")
                if not self.bridge_finished:
                    self.bridge.finish()
                    self.bridge_finished = True
                self._generate_ready(stage, batch, outputs, metadata, timed, trace)
                if not self._bridge_ready() and not self._pending_chunks():
                    if not self.sampler.ended:
                        self.sampler.finish()
                    if not self.sampler.complete:
                        raise ValueError("Continuous source sampling did not drain")
                    self.decoder.finish(total_frames=self.ledger.accepted_frames)
                    self._decode_ready(stage, outputs, metadata, timed, trace, eos=True)
                    self.ledger.consume(self.bridge.accepted_pixels)
                    self.ledger.completed()
                    if self.component_capture is not None:
                        self.component_capture.finish()
                    completed = True
                    # Record this rank's Host completion after its final decode.
                    # The collector drains queued records before result delivery.
                    with stage_span(
                        trace,
                        "Request",
                        cuda=trace is not None and trace.measure_cuda,
                        component="continuous_source_complete",
                        source_encoder_role=(
                            "dedicated"
                            if "remote" in self.policy["source_encoder"]
                            else "local"
                        ),
                    ) as span:
                        pass
                    handoff(
                        span,
                        complete=True,
                        output_frame_range=[0, self.ledger.accepted_frames],
                    )
            more_work = not completed and (
                self.pixels.ready or self._bridge_ready() or self._pending_chunks()
            )
            self.command_index += 1
            usage = {
                "zing_realtime_blocks": metadata,
                "zing_source_consumed_end": self.bridge.accepted_pixels,
                "zing_source_complete": completed,
                "zing_source_more_work": bool(more_work),
                "zing_source_dispatch_steps": self.dispatch_steps,
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
            if self.policy.get("joint_forward", "off") != "off":
                usage["zing_h3"] = self.sampler.metrics
                usage["zing_joint_wait"] = command.get("joint_wait")
            memory = usage["zing_source_memory"]
            if self.policy["source_encoder"]["mode"] == "progressive":
                memory["encoder_state"] = vars(self.pixels.state_usage())
                if hasattr(self.pixels, "queue_snapshot"):
                    memory["encoder_queues"] = self.pixels.queue_snapshot()
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
                        "ended": completed,
                        "stage_host_ms": timings,
                        "progress": self.progress,
                        "first_source_received_unix_ns": command.get(
                            "first_source_received_unix_ns"
                        ),
                        "dispatch_unix_ns": command.get("dispatch_unix_ns"),
                        "source_preview": command.get("source_preview", False),
                        "source_wait_host_ms": command.get("source_wait_host_ms"),
                        "joint_wait": command.get("joint_wait"),
                        "joint": usage.get("zing_h3"),
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
                result = OutputBatch(
                    output=outputs,
                    metrics=batch.metrics,
                    usage=usage,
                    conditioning_notice=notice,
                )
            else:
                result = OutputBatch(
                    raw_frame_batches=[],
                    metrics=batch.metrics,
                    usage=usage,
                    conditioning_notice=notice,
                )
            if completed:
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

    def _within_work_budget(self):
        return not self.work_quantum or self.dispatch_steps < self.work_quantum

    def _pending_chunks(self):
        return getattr(self.sampler, "pending", ())

    def _next_bridge_count(self):
        if hasattr(self.sampler, "next_chunk_size"):
            return self.sampler.next_chunk_size
        upcoming = self.bridge.plan.chunk(
            self.bridge.chunk_id,
            self.bridge.accepted_latents if self.bridge.ended else None,
        )
        return None if upcoming is None else upcoming.end - upcoming.start

    def _bridge_ready(self):
        count = self._next_bridge_count()
        return count is not None and self.bridge.ready_for(count)

    def _pop_bridge(self):
        count = self._next_bridge_count()
        return None if count is None else self.bridge.pop(count)

    def _generate_ready(
        self, stage, batch, outputs, metadata, timed, trace=None, *, flush=True
    ):
        if getattr(self.sampler.sampler, "joint_forward", "off") == "cross_chunk":
            return self._generate_joint_ready(
                stage, batch, outputs, metadata, timed, trace, flush=flush
            )
        while self._bridge_ready() and self._within_work_budget():
            with timed("bridge_host_ms"):
                chunk = self._pop_bridge()
                self.sampler.append(
                    chunk.source,
                    chunk.noise,
                    provenance=chunk.provenance,
                    final=chunk.final,
                )
            with stage_span(
                trace,
                "DiT",
                cuda=trace is not None and trace.measure_cuda,
                input_ids=[
                    f"{batch.realtime_session_id}:latent:{chunk.chunk.start}:{chunk.chunk.end}"
                ],
                latent_range=[chunk.chunk.start, chunk.chunk.end],
                source_dependency_end=chunk.provenance.source_dependency_end,
                block_index=chunk.chunk.chunk_id,
            ) as span:
                with (
                    timed("dit_host_ms"),
                    stage._use("transformer", "denoise") as transformer,
                    maybe_record_function("Zing.source_dit"),
                ):
                    with maybe_record_function(
                        f"Zing.clock_anchor.unix_ns{time.time_ns()}"
                    ):
                        pass
                    self.sampler.sampler.generator = transformer
                    self._record_progress(
                        "dit_started",
                        chunk_id=chunk.chunk.chunk_id,
                        latent_start=chunk.chunk.start,
                        latent_end=chunk.chunk.end,
                        source_dependency_end=chunk.provenance.source_dependency_end,
                    )
                    result = self.sampler.step()
                    self.dispatch_steps += 1
            handoff(
                span,
                output_ids=[
                    f"{batch.realtime_session_id}:generated-latent:{result.chunk.start}:{result.chunk.end}"
                ],
                output_latent_range=[result.chunk.start, result.chunk.end],
            )
            if self.component_capture is not None:
                self.component_capture.append(chunk, result)
            self.blocks.append(
                {
                    "block_idx": result.chunk.chunk_id,
                    "request_id": batch.request_id,
                    "event_id": result.prompt_event_id,
                    "prompt_version": result.prompt_version,
                    "frame_end": result.provenance.pixels.end,
                }
            )
            self.decoder.append(
                result.visual,
                latent_start=result.chunk.start,
                source_dependency_end=result.provenance.source_dependency_end,
                valid_frame_end=result.provenance.pixels.end,
            )
            self._decode_ready(stage, outputs, metadata, timed, trace)

    def _generate_joint_ready(
        self, stage, batch, outputs, metadata, timed, trace=None, *, flush=True
    ):
        while (
            self._bridge_ready() or self.sampler.pending
        ) and self._within_work_budget():
            capacity = self.sampler.max_pending_chunks
            if self.work_quantum:
                capacity = min(capacity, self.work_quantum - self.dispatch_steps)
            while self._bridge_ready() and len(self.sampler.pending) < capacity:
                if self.sampler.pending:
                    count = self._next_bridge_count()
                    values = self.bridge.values[:, :, :count]
                    needed = values.numel() * (values.element_size() + 4)
                    del values
                    # One-chunk fallback before popping transfers any ownership.
                    if (
                        self.sampler.pending_bytes + needed
                        > self.sampler.max_pending_bytes
                    ):
                        reasons = self.sampler.sampler.last_run_metrics[
                            "joint_fallbacks"
                        ]
                        reasons["ready_queue_capacity"] = (
                            reasons.get("ready_queue_capacity", 0) + 1
                        )
                        flush = True
                        break
                with timed("bridge_host_ms"):
                    chunk = self._pop_bridge()
                    self.sampler.append(
                        chunk.source,
                        chunk.noise,
                        provenance=chunk.provenance,
                        final=chunk.final,
                    )
                    del chunk
            if not flush and len(self.sampler.pending) < capacity:
                return
            # The transformer lease ends before every decoder call. Closing on
            # failure clears any prefetched next active target as well.
            with closing(self.sampler.drain_ready()) as ready:
                while True:
                    capture_chunk = None
                    dit_fields = {}
                    if self.sampler.pending:
                        pending = self.sampler.pending[0]
                        dit_fields = {
                            "input_ids": [
                                f"{batch.realtime_session_id}:latent:{pending.chunk.start}:{pending.chunk.end}"
                            ],
                            "latent_range": [
                                pending.chunk.start,
                                pending.chunk.end,
                            ],
                            "source_dependency_end": (
                                pending.provenance.source_dependency_end
                            ),
                            "block_index": pending.chunk.chunk_id,
                        }
                        # Do not retain the ReadySourceChunk's source/noise GPU
                        # tensors across drain_ready() and Decoder execution.
                        del pending
                    with (
                        stage_span(
                            trace,
                            "DiT",
                            cuda=trace is not None and trace.measure_cuda,
                            **dit_fields,
                        ) as span,
                        timed("dit_host_ms"),
                        stage._use("transformer", "denoise") as transformer,
                        maybe_record_function("Zing.source_dit"),
                    ):
                        with maybe_record_function(
                            f"Zing.clock_anchor.unix_ns{time.time_ns()}"
                        ):
                            pass
                        if self.sampler.sampler is not None:
                            self.sampler.sampler.generator = transformer
                        if self.sampler.pending:
                            chunk = self.sampler.pending[0]
                            if self.component_capture is not None:
                                capture_chunk = SourceChunk(
                                    chunk=chunk.chunk,
                                    source=chunk.source,
                                    noise=chunk.noise,
                                    provenance=chunk.provenance,
                                    final=(
                                        self.bridge.ended
                                        and chunk.chunk.end
                                        == self.bridge.accepted_latents
                                    ),
                                )
                            self._record_progress(
                                "dit_started",
                                chunk_id=chunk.chunk.chunk_id,
                                latent_start=chunk.chunk.start,
                                latent_end=chunk.chunk.end,
                                source_dependency_end=chunk.provenance.source_dependency_end,
                            )
                            del chunk
                        result = next(ready, None)
                    if result is None:
                        handoff(span, output_ids=[], no_output=True)
                        break
                    handoff(
                        span,
                        output_ids=[
                            f"{batch.realtime_session_id}:generated-latent:{result.chunk.start}:{result.chunk.end}"
                        ],
                        output_latent_range=[result.chunk.start, result.chunk.end],
                    )
                    self.dispatch_steps += 1
                    if self.component_capture is not None:
                        self.component_capture.append(capture_chunk, result)
                        del capture_chunk
                    self.blocks.append(
                        {
                            "block_idx": result.chunk.chunk_id,
                            "request_id": batch.request_id,
                            "event_id": result.prompt_event_id,
                            "prompt_version": result.prompt_version,
                            "frame_end": result.provenance.pixels.end,
                        }
                    )
                    self.decoder.append(
                        result.visual,
                        latent_start=result.chunk.start,
                        source_dependency_end=result.provenance.source_dependency_end,
                        valid_frame_end=result.provenance.pixels.end,
                    )
                    self._decode_ready(stage, outputs, metadata, timed, trace)

    def _decode_ready(self, stage, outputs, metadata, timed, trace=None, *, eos=False):
        while self.decoder.ready:
            latent_end = getattr(self.decoder, "accepted_latents", None)
            span_fields = {"final": bool(eos)}
            if latent_end is not None:
                span_fields.update(
                    input_ids=[f"{self.session_id}:generated-latent:0:{latent_end}"],
                    input_latent_range=[0, latent_end],
                )
            with stage_span(
                trace,
                "Decoder",
                cuda=trace is not None and trace.measure_cuda,
                **span_fields,
            ) as span:
                with (
                    timed("decoder_host_ms"),
                    stage._use(stage.codecs.video_decoder_name, "decode"),
                    maybe_record_function("Zing.source_decode"),
                ):
                    with maybe_record_function(
                        "Zing.source_decode.eos" if eos else "Zing.source_decode.chunk"
                    ):
                        decoded = self.decoder.pop()
            if decoded.end > decoded.start:
                handoff(
                    span,
                    output_ids=[
                        f"{self.session_id}:frames:{decoded.start}:{decoded.end}"
                    ],
                    output_frame_range=[decoded.start, decoded.end],
                )
            else:
                handoff(span, output_ids=[], no_output=True)
            if self.component_capture is not None:
                self.component_capture.append_frames(
                    "decoder", decoded.values[0], decoded.start
                )
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
        self.failed_update = None
        self.closed = True
