# SPDX-License-Identifier: Apache-2.0
"""Worker-owned continuous RGB session; reuse codec and source-ready kernels."""

import hashlib
import json
import logging
import os
import time
from collections import deque
from contextlib import closing, contextmanager, nullcontext
from dataclasses import replace

import psutil
import torch

from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState
from sglang.multimodal_gen.runtime.realtime.source_prepare import (
    PROTOCOL,
    V2_PROTOCOL,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange
from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
    SourceLedger,
    SourceLimits,
    default_source_encoder,
    remote_decode_progressive,
    source_decode_quantum,
    source_work_quantum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.profiler import maybe_record_function
from sglang.multimodal_gen.runtime.utils.zing_stage_timing import (
    handoff,
    stage_span,
    trace_for_batch,
)

from .block0_telemetry import Block0Telemetry
from .cache_logging import batch_context, cache_event, cache_scope
from .incremental import SourceReadySession
from .media_geometry import resolve_source_geometry
from .prompt_update import (
    admit_prompt_update,
)
from .remote_decoder import create_decoder
from .source_bridge import SourceChunk, SourceChunkBridge, source_pixel_end
from .source_encoder import create_source_buffer
from .streaming_noise import STREAMING_NOISE_VERSION

logger = init_logger(__name__)


def source_preview(config, data):
    """Construct identical S0 conditioning for prefill and normal generation."""
    width, height = config["width"], config["height"]
    size = width * height * 3
    if not isinstance(data, bytes) or len(data) < size:
        raise ValueError("Source preview requires one complete RGB24 frame")
    pixels = torch.frombuffer(bytearray(data[:size]), dtype=torch.uint8).reshape(
        height, width, 3
    )
    return dict(
        type="video",
        id="<Video 1>",
        clip=dict(start=0, end=0),
        frames=pixels.permute(2, 0, 1).unsqueeze(1),
        fps=float(config["fps"]),
        origin="source_s0",
    )


def _source_key_policy(stage, sampler, work_quantum, command):
    """Policy fields that must be identical for a warmup and a live session."""
    defer_final = command.get(
        "defer_final", os.environ.get("SGLANG_ZING_SOURCE_DEFER_FINAL", "0") == "1"
    )
    # Joint scheduling is independent of whether a prefix cache is available.
    prompt_joint = command.get(
        "prompt_joint", os.environ.get("SGLANG_ZING_SOURCE_PROMPT_JOINT", "0") == "1"
    )
    if type(defer_final) is not bool or type(prompt_joint) is not bool:
        raise ValueError("Source scheduling flags must be boolean")
    decoder = command.get("decoder")
    progressive_remote = command.get(
        "remote_progressive",
        bool(decoder and "remote" in decoder) and remote_decode_progressive(),
    )
    if type(progressive_remote) is not bool:
        raise ValueError("Remote decode progressive policy must be boolean")
    # Preserve Shanghai's bounded remote-output path with the q0 lookahead budget.
    defer_final = defer_final or (
        progressive_remote and getattr(sampler, "joint_forward", "off") == "cross_chunk"
    )
    decode_quantum = source_decode_quantum(
        command.get("decode_quantum"), work_quantum=work_quantum
    )
    if decode_quantum and stage.codecs.video_decoder_name != "lynn_decoder":
        raise ValueError("Source decode quantum requires the Lynn decoder")
    if defer_final and work_quantum != 1 and not progressive_remote:
        raise ValueError("Deferred source final currently requires work quantum 1")
    if sampler.window_policy.window_chunks is None:
        raise ValueError("Continuous source requires a bounded attention window")
    policy = {
        "window_chunks": sampler.window_policy.window_chunks,
        "window_eviction_timing": getattr(
            sampler, "window_eviction_timing", "after_step0"
        ),
        "sink_chunks": sampler.window_policy.sink_chunks,
        "rope_position_mode": sampler.rope_position_mode,
        "rope_max_frame_gap": sampler.rope_max_frame_gap,
        "video_decoder": stage.codecs.video_decoder_name,
        "source_work_quantum": work_quantum,
        "source_decode_quantum": decode_quantum,
        "source_defer_final": defer_final,
        "source_prompt_joint": prompt_joint,
    }
    if getattr(sampler, "joint_forward", "off") != "off":
        policy["joint_forward"] = sampler.joint_forward
    return policy, defer_final, prompt_joint


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
        self.conditioning_prefetch = None
        self.kv_lease = None
        self.pending_output = None
        self.remote_decoder = False
        # Resolve once from the API's deployment command, not a worker-local env.
        self.work_quantum = None
        self.decode_quantum = None
        self.end_received = self.bridge_finished = False
        self.decoder_finished = False
        self.source_geometry = None
        self.dispatch_steps = 0
        self.decode_steps = 0
        self.startup_identity = None
        self.startup_attached = False
        self.startup_rgb_sha256 = None
        self.startup_protocol = None
        self.startup_preview_key = None
        self.block0_cache = self.block0_entry = self.block0_restore = None
        self.block0_prefix = self.block0_request = None
        self.block0_key = None
        self.block0_lookup_ms = 0.0
        self.block0_output_start = 0
        self.pending_block0 = None
        self.block0_telemetry = None
        self.block0_reported = False
        self.block0_prompt_version = 1
        self.pending_block0_prompt_version = None
        self.pending_block0_lookup = None
        self.pending_block0_started = None
        self.cache_write = False
        self.cache_labels = {}

    def initialize(self, stage, batch, config, trace=None):
        self.block0_telemetry = Block0Telemetry()
        self.block0_prompt_version = (
            getattr(batch, "realtime_prompt_version", None) or 1
        )
        params = batch.sampling_params
        stage._validate_sampling_params(params)
        command = batch.extra["zing_source_command"]
        self.cache_write = command.get("cache_write", False)
        self.cache_labels = {
            name: command.get(name) for name in ("game_id", "group_id")
        }
        self.work_quantum = source_work_quantum(command.get("work_quantum"))
        sampling_options = (
            {"source_sampling_schedule": command["sampling_schedule"]}
            if "sampling_schedule" in command
            else {}
        )
        sampler = stage._build_realtime_sampler(params, **sampling_options)
        self.policy, defer_final, prompt_joint = _source_key_policy(
            stage, sampler, self.work_quantum, command
        )
        self.decode_quantum = self.policy["source_decode_quantum"]
        sampler.prompt_joint = prompt_joint
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

        preview = None
        if command.get("source_preview", False):
            # S0 may use native preview geometry independently of the CLI stream.
            startup = command.get("startup", {})
            if startup.get("op") == "prepare":
                preview = source_preview(
                    {
                        **config,
                        **(
                            {"width": startup["width"], "height": startup["height"]}
                            if startup.get("protocol", PROTOCOL) == V2_PROTOCOL
                            else {}
                        ),
                    },
                    startup["rgb"],
                )
            else:
                preview = source_preview(
                    config, command["appends"][0]["payload"]["data"]
                )
        request = stage._build_request(batch)
        self.block0_cache = getattr(stage, "block0_cache", None)
        log_rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        if self.block0_cache is None:
            cache_event(
                "lookup",
                "disabled",
                reason="cache_disabled",
                next_action="recompute",
                rank=log_rank,
            )
        if not self.cache_write:
            cache_event(
                "cache_write", "disabled", reason="cache_write_false", rank=log_rank
            )
        if self.block0_cache is not None:
            from .preparation_cache import (
                inputs_on,
                preparation_key,
                preparation_request,
            )

            started = time.perf_counter()
            self.block0_input_request = preparation_request(stage.codecs, request)
            # Freeze the same key inputs for later prompt changes. self.policy
            # gains reporting-only encoder/decoder fields after initialization.
            self.block0_key_args = (
                stage.codecs,
                preview,
                dict(config),
                self.block0_cache.identity,
            )
            self.block0_key = preparation_key(
                stage.codecs,
                self.block0_input_request,
                preview,
                config,
                self.block0_cache.identity,
            )
            self.block0_entry = self.block0_cache.lookup(self.block0_key)
            self.block0_lookup_ms = (time.perf_counter() - started) * 1000
            lookup = getattr(self.block0_cache, "last_lookup", {})
            self.block0_telemetry.cache_result = lookup.get("cache_result", "miss")
            self.block0_telemetry.miss_reason = lookup.get("miss_reason")
            self.block0_telemetry.lookup_details = dict(lookup)
            self.block0_telemetry.record("cache_lookup", self.block0_lookup_ms)
        else:
            self.block0_telemetry.cache_result = "bypass"
        if self.block0_entry is None:
            self.inputs = stage.codecs.prepare(
                request,
                prepare_use,
                retain_condition_media=True,
                source_preview=preview,
                conditioning=stage._conditioning_consumer(batch),
                stage_trace=trace,
                preparation_metrics=self.block0_telemetry,
            )
        if self.block0_entry is not None:
            from sglang.multimodal_gen.runtime.conditioning.worker import (
                WorkerConditioningState,
            )

            # Consume the preparation that produced the cached KV.
            device = next(stage.modules["transformer"].parameters()).device
            stage.admit_cached_preparation(self.block0_entry)
            self.inputs = inputs_on(self.block0_entry.inputs, device)
            self.block0_telemetry.skip(
                "qwen_prepare", "qwen_remote_prepare", "reference_vae", "block0_forward"
            )
            condition = batch.session.get_or_create_state(WorkerConditioningState)
            condition.identity = self.block0_entry.conditioning_identity
            condition.positive = self.inputs.positive
        if self.block0_cache is not None:
            from sglang.multimodal_gen.runtime.conditioning.worker import (
                WorkerConditioningState,
            )

            self.block0_inputs = self.inputs
            self.block0_condition_identity = batch.session.get_or_create_state(
                WorkerConditioningState
            ).identity
        if self.inputs.audio_shape is not None or self.inputs.source_visual is not None:
            raise ValueError("Continuous source requires a video-only static prefix")
        ratio = stage.codecs.spatial_ratio
        self.source_geometry = resolve_source_geometry(
            (config["height"], config["width"]),
            tuple(n * ratio for n in self.inputs.visual_shape[-2:]),
            stage.codecs.pipeline_config,
            stage.codecs.runtime["h3_media_config"],
        )
        from dataclasses import asdict

        if command.get("source_geometry") != asdict(self.source_geometry):
            raise ValueError("Source geometry differs between adapter and worker")
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
        if "remote" in encoder:
            encoder = {
                **encoder,
                "remote": {
                    **encoder["remote"],
                    "source_geometry": asdict(self.source_geometry),
                },
            }
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
            target_shape=(stage.packer.visual_channels, *self.inputs.visual_shape[-2:]),
        )
        cached_joint = (
            prompt_joint and getattr(sampler, "joint_forward", "off") == "cross_chunk"
        )
        self.sampler = SourceReadySession(
            sampler,
            self.inputs,
            seed=self.bridge.seed,
            max_pending_chunks=(
                2 if getattr(sampler, "joint_forward", "off") == "cross_chunk" else 1
            ),
            max_pending_bytes=256 << 20,
            acquire_storage=self._pooled_storage(stage, batch.session, sampler),
            prompt_switch_prefix_latent_frames=(
                stage.prompt_switch_prefix_latent_frames
            ),
            initial_prompt_event_id=batch.realtime_event_id,
            initial_prompt_version=getattr(batch, "realtime_prompt_version", None),
            defer_final=defer_final,
            prompt_joint=prompt_joint,
            prefix_loader=(
                self._load_block0_prefix
                if cached_joint and self.block0_cache is not None
                else None
            ),
        )
        if self.block0_cache is not None:
            self.sampler.prepare_prompt_prefix = self._prepare_block0_switch
        if self.block0_entry is not None:
            if not cached_joint:
                self.sampler.restore_initial_prefix = self._restore_block0
            # The cached admission already resolved its layout.
            self.kv_lease = stage.kv_pool.acquire(
                batch.session, self.block0_entry.request
            )
            self.sampler.cache = self.sampler._new_cache(self.kv_lease.kv_cache)
            self.block0_restore = self.block0_cache.restore(
                self.sampler.cache, self.block0_entry
            )
        elif self.block0_cache is not None and self.cache_write:
            self.sampler.capture_initial_prefix = self._remember_block0
        endpoint = command.get("conditioning_prefetch_endpoint")
        if endpoint is not None:
            self._start_conditioning_prefetch(stage, batch, endpoint, command)
        capture_dir = os.environ.get("SGLANG_ZING_COMPONENT_CAPTURE_DIR")
        decoder_policy = batch.extra["zing_source_command"].get("decoder")
        capture_scope = os.environ.get("SGLANG_ZING_COMPONENT_CAPTURE_SCOPE", "full")
        if capture_dir and capture_scope not in ("full", "dit"):
            raise ValueError("Component capture scope must be full or dit")
        if (
            capture_dir
            and capture_scope == "full"
            and decoder_policy
            and "remote" in decoder_policy
        ):
            raise ValueError("Component capture requires the local decoder")
        self.decoder = create_decoder(
            stage.codecs,
            policy=decoder_policy,
            capacity_latents=2 * tokens + max_chunk + tokens,
            provenance_for=self.ledger.provenance_for,
            session_id=batch.realtime_session_id,
        )
        self.remote_decoder = bool(decoder_policy and "remote" in decoder_policy)
        self.remote_progressive = self.remote_decoder and command.get(
            "remote_progressive", remote_decode_progressive()
        )
        self.policy["remote_decode_progressive"] = self.remote_progressive
        self.policy["video_decoder_role"] = (
            "dedicated" if self.remote_decoder else "dit"
        )
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
                scope=capture_scope,
                replay_policy=dict(
                    work_quantum=self.work_quantum,
                    schedule=dict(
                        solver=sampler.solver
                        if hasattr(sampler, "solver")
                        else params.zing_solver,
                        denoising_step_list=sampler.denoising_step_list,
                        initial_denoising_step_list=sampler.initial_denoising_step_list,
                        initial_block_count=sampler.initial_block_count,
                    ),
                    defer_final=defer_final,
                    prompt_joint=prompt_joint,
                    initial_prompt_event_id=batch.realtime_event_id,
                    initial_prompt_version=getattr(
                        batch, "realtime_prompt_version", None
                    ),
                ),
            )
            self.component_capture.remember_prompt(self.sampler.prompt)

    @torch.no_grad()
    def prefill(
        self,
        request,
        *,
        stage,
        batch,
        config=None,
        source_preview=None,
        source_shape=None,
        trace=None,
        force=False,
    ):
        with cache_scope(batch_context(batch)):
            started = time.perf_counter()
            rank = getattr(getattr(stage, "block0_cache", None), "log_rank", None)
            self.prefill_reused = False
            cache_event(
                "prefill",
                "started",
                rank=rank,
                reason="force_recompute" if force else None,
            )
            try:
                entry = self._prefill(
                    request,
                    stage=stage,
                    batch=batch,
                    config=config,
                    source_preview=source_preview,
                    source_shape=source_shape,
                    trace=trace,
                    force=force,
                )
                if entry is None:
                    raise ValueError("Block0 prefill did not publish a cache entry")
            except Exception as exc:
                cache_event(
                    "prefill",
                    "failed",
                    rank=rank,
                    started=started,
                    reason=type(exc).__name__,
                    level=logging.WARNING,
                )
                raise
            cache_event(
                "prefill",
                "completed",
                cache_key=entry.key,
                rank=rank,
                started=started,
                reason="reused" if self.prefill_reused else "recomputed",
                l2_persisted=None,
            )
            return entry

    def _prefill(
        self,
        request,
        *,
        stage,
        batch,
        config,
        source_preview,
        source_shape,
        trace,
        force,
    ):
        """Warm one Block0 entry without source history, sampling or decoding.

        Only the Qwen/reference preparation and the Block0 prefix forward run.
        The resulting L1 slot is immediately visible to normal source sessions,
        and disk persistence is queued on the existing daemon writer.
        """
        self.block0_telemetry = Block0Telemetry()
        cache = getattr(stage, "block0_cache", None)
        if cache is None:
            raise ValueError("Block0 prefill requires an enabled Block0 cache")
        command = batch.extra.get("zing_source_command", {})
        config = dict(config or command.get("config") or {})
        if not {"width", "height", "fps"} <= config.keys():
            raise ValueError("Block0 prefill requires width, height and fps")
        params = batch.sampling_params
        stage._validate_sampling_params(params)
        sampler = stage._build_realtime_sampler(
            params,
            **(
                {"source_sampling_schedule": command["sampling_schedule"]}
                if "sampling_schedule" in command
                else {}
            ),
        )
        self.sampler = sampler
        self.work_quantum = source_work_quantum(command.get("work_quantum"))
        self.policy, _, _ = _source_key_policy(
            stage, sampler, self.work_quantum, command
        )

        from .preparation_cache import preparation_key, preparation_request

        input_request = preparation_request(stage.codecs, request)
        key = preparation_key(
            stage.codecs, input_request, source_preview, config, cache.identity
        )
        with self.block0_telemetry.measure("cache_lookup"):
            entry = None if force else cache.lookup(key, allow_disk=True)
        lookup = (
            {"cache_result": "miss", "miss_reason": "force_recompute"}
            if force
            else getattr(cache, "last_lookup", {})
        )
        self.block0_telemetry.cache_result = lookup.get("cache_result", "miss")
        self.block0_telemetry.miss_reason = lookup.get("miss_reason")
        self.block0_telemetry.lookup_details = dict(lookup)
        if entry is not None:
            self.prefill_reused = True
            self.block0_telemetry.skip(
                "qwen_prepare", "qwen_remote_prepare", "reference_vae", "block0_forward"
            )
            ensure_disk = getattr(cache, "ensure_disk", None)
            if batch.extra.get("zing_prefill", {}).get("persist") and callable(
                ensure_disk
            ):
                with self.block0_telemetry.measure("persist_submit"):
                    entry = ensure_disk(entry)
            self.block0_telemetry.finish()
            return entry

        inputs = stage.codecs.prepare(
            request,
            stage._use,
            retain_condition_media=True,
            source_preview=source_preview,
            conditioning=stage._conditioning_consumer(batch),
            stage_trace=trace,
            preparation_metrics=self.block0_telemetry,
        )
        if inputs.audio_shape is not None or inputs.source_visual is not None:
            raise ValueError("Block0 prefill requires a video-only static prefix")
        from sglang.multimodal_gen.runtime.conditioning.worker import (
            WorkerConditioningState,
        )

        condition = batch.session.get_or_create_state(WorkerConditioningState)
        identity = condition.identity
        with stage._use("transformer", "denoise") as transformer:
            sampler.generator = transformer
            device = next(transformer.parameters()).device
            visual = (
                inputs.positive["context"]
                .new_zeros((1, stage.packer.visual_channels, *inputs.visual_shape))
                .to(device)
            )
            layout = sampler.processor.pack_inference_layout(
                inputs, inputs.positive, visual, None
            )
            source_shape = (
                inputs.visual_shape[-2:] if source_shape is None else source_shape
            )
            lease = storage = pool_request = None
            if getattr(stage, "kv_pool", None) is not None:
                pool_request = stage.kv_request(
                    sampler,
                    inputs,
                    layout,
                    source_shape=source_shape,
                    total_frames=None,
                )
                lease = stage.kv_pool.acquire(batch.session, pool_request)
                storage = lease.kv_cache
            state = None
            try:
                cuda_begin = cuda_end = None
                if device.type == "cuda":
                    cuda_begin, cuda_end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    cuda_begin.record()
                with self.block0_telemetry.measure("block0_forward"):
                    state = sampler.initialize(inputs, visual, storage=storage)
                if cuda_end is not None:
                    cuda_end.record()
                with self.block0_telemetry.measure("d2h_and_l1_publish"):
                    stored = cache.fill(
                        key,
                        inputs,
                        identity,
                        pool_request if pool_request is not None else request,
                        state.cache,
                        force=force,
                    )
                    if stored is False:
                        raise ConditionError(
                            "CONDITION_BUSY", "Block0 CPU writeback slots are busy"
                        )
                    entry = cache.finish_fill()
                if cuda_end is not None and cuda_end.query():
                    self.block0_telemetry.cuda["block0_forward"] = (
                        cuda_begin.elapsed_time(cuda_end)
                    )
                self.block0_telemetry.finish()
                return entry
            finally:
                try:
                    if state is not None:
                        state.close()
                finally:
                    if lease is not None:
                        lease.release()

    def _pooled_storage(self, stage, session, sampler):
        """Lease a slot at the first step; the ready session never sees the pool."""
        if stage.kv_pool is None:
            return None

        def acquire(positive, ready):
            request = stage.kv_request(
                sampler,
                self.inputs,
                positive,
                source_shape=ready.source.shape,
                total_frames=None,
            )
            self.block0_request = request
            self.kv_lease = stage.kv_pool.acquire(session, request)
            return self.kv_lease.kv_cache

        return acquire

    def _restore_block0(self, cache):
        if self.block0_restore is None:
            self.block0_restore = self.block0_cache.restore(cache, self.block0_entry)
        self.block0_restore.commit(cache)

    def _load_block0_prefix(self, cache, inputs, prompt, prefix):
        """Consume the startup entry selected by the preparation cache.

        Prompt switches use prepare_prompt_prefix so L1/L2 lookup and epoch
        writeback remain owned by the same provider as the serial consumer.
        """
        if self.sampler.applied_prompt is not None or self.block0_entry is None:
            return False
        self._restore_block0(cache)
        return True

    def _remember_block0(self, prefix):
        self.block0_prefix = prefix
        # Only pixels generated under this prefix qualify, even if old output
        # is still queued in the decoder when the switch occurs.
        self.block0_output_start = self.sampler.committed_pixel
        if self.block0_telemetry is not None:
            self.block0_telemetry.record(
                "prefix_ready_elapsed",
                (time.perf_counter() - self.block0_telemetry.started) * 1000,
            )

    def _prepare_block0_switch(self, candidate):
        """Called at the actual switch boundary, before clearing the live slot."""
        lookup_started = time.perf_counter()
        if candidate.reuse_prefix:
            self.block0_prefix = self.block0_restore = self.block0_entry = None
            self.pending_block0 = None
            self.block0_telemetry = Block0Telemetry()
            self.block0_telemetry.cache_result = "bypass"
            self.block0_telemetry.lookup_details = {
                "reuse_source": "resident_gpu_prefix"
            }
            self.block0_telemetry.skip(
                "cache_lookup",
                "qwen_prepare",
                "reference_vae",
                "block0_forward",
                "restore",
            )
            self.block0_prompt_version = (
                candidate.prompt_version or self.block0_prompt_version + 1
            )
            self.block0_reported = False
            self.sampler.capture_initial_prefix = None
            return None
        self.block0_cache.finish_fill()
        key, request, inputs, identity, pool_request = self.pending_block0
        self.pending_block0 = None
        self.block0_key, self.block0_input_request = key, request
        self.block0_inputs, self.block0_condition_identity = inputs, identity
        self.block0_request = pool_request
        self.block0_prefix = None
        self.block0_entry = self.block0_cache.lookup(key)
        # This is the actual activation boundary, not merely an accepted update.
        # Keep a prior L2 promotion visible instead of relabelling its subsequent
        # in-memory lookup as an end-to-end L1 hit.
        final_lookup = dict(getattr(self.block0_cache, "last_lookup", {}))
        first_lookup = self.pending_block0_lookup or final_lookup
        lookup = (
            first_lookup
            if self.block0_entry is not None
            and first_lookup.get("cache_result") != "miss"
            else final_lookup
        )
        self.block0_telemetry = Block0Telemetry()
        self.block0_telemetry.started = self.pending_block0_started or lookup_started
        self.block0_telemetry.cache_result = lookup.get("cache_result", "miss")
        self.block0_telemetry.miss_reason = lookup.get("miss_reason")
        self.block0_telemetry.lookup_details = dict(lookup)
        self.block0_telemetry.record("cache_lookup", lookup.get("lookup_host_ms", 0))
        self.block0_prompt_version = (
            self.pending_block0_prompt_version or self.block0_prompt_version + 1
        )
        self.block0_reported = False
        self.block0_restore = None
        self.pending_block0_lookup = self.pending_block0_prompt_version = None
        self.pending_block0_started = None
        self.sampler.capture_initial_prefix = None
        if self.block0_entry is None:
            self.sampler.capture_initial_prefix = (
                self._remember_block0 if self.cache_write else None
            )
            return None

        def restore(cache):
            self.block0_restore = self.block0_cache.restore(cache, self.block0_entry)
            self.block0_restore.commit(cache)

        return restore

    def _fill_block0_after_output(self):
        # Called on the dispatch after the first RGB output has been returned.
        if (
            self.cache_write
            and self.block0_prefix is not None
            and self.ledger.output_frames > self.block0_output_start
        ):
            prefix, self.block0_prefix = self.block0_prefix, None
            if self.sampler.cache.prefix is prefix:
                stored = self.block0_cache.fill(
                    self.block0_key,
                    self.block0_inputs,
                    self.block0_condition_identity,
                    self.block0_request,
                    self.sampler.cache,
                    required=False,
                )
                if stored:
                    self.block0_cache.register(self.block0_key, **self.cache_labels)

    def _block0_update_key(self, stage, prompt, command):
        """Share the CPU request identity between prefetch and owner lookup."""
        from .preparation_cache import preparation_key, preparation_request

        request = {**self.block0_input_request, "prompt": prompt}
        if "references" in command:
            active = self.inputs
            old_ids = {item["content_id"] for item in active.conditions}
            refs = preparation_request(stage.codecs, {"media": command["references"]})[
                "media"
            ]
            request["media"] = [
                item for item in request["media"] if item["id"] not in old_ids
            ] + refs
            request["conditions"] = [
                dict(
                    content_id=item["id"],
                    role="reference",
                    latent_modalities=["visual"],
                )
                for item in refs
            ]
        codecs, preview, config, identity = self.block0_key_args
        return preparation_key(codecs, request, preview, config, identity), request

    def _start_conditioning_prefetch(self, stage, batch, endpoint, command):
        from sglang.multimodal_gen.runtime.conditioning.prefetch import (
            ConditioningPrefetch,
        )
        from sglang.multimodal_gen.runtime.conditioning.worker import (
            RemoteConditioner,
            WorkerConditioningState,
            _identity,
        )

        provider = stage.conditioning_provider
        if not isinstance(provider, RemoteConditioner):
            raise ConditionError(
                "CONDITION_INVALID", "Prefetch requires remote conditioning"
            )

        def prepare(update):
            # Capture the current worker-owned media once; a later commit replaces
            # self.inputs, but never mutates these normalized CPU tensors.
            active = self.inputs
            media = active.condition_media
            if "references" in update:
                media, _, _ = stage.codecs.prepare_reference_media(
                    active, update["references"]
                )
            return update["prompt"], media

        def identity(prompt, media):
            provider.client.validate_input(prompt, media)
            return _identity(
                prompt,
                media,
                provider.profile_id,
                WorkerConditioningState(),
                provider.preprocess_version,
            )

        def should_prefetch(update):
            if "references" not in update and update["prompt"] == self.prompt:
                return False
            if self.block0_cache is None:
                return True
            key, _ = self._block0_update_key(stage, update["prompt"], update)
            return not self.block0_cache.may_contain(key)

        receipt = {"ok": True}
        if provider.rank == 0:
            try:
                previous = getattr(provider, "_prefetch_owner", None)
                if previous is not None and previous.thread.is_alive():
                    raise ConditionError(
                        "CONDITION_BUSY", "Previous conditioning work has not drained"
                    )
                condition = batch.session.get_or_create_state(WorkerConditioningState)
                device = (
                    torch.device("cuda", torch.cuda.current_device())
                    if getattr(provider.client, "result_transport", "http") == "nccl"
                    else None
                )

                def encode_request(prompt, media, *, input_digest, deadline=None):
                    return condition.encode_remote(
                        provider.client,
                        prompt,
                        media,
                        input_digest=input_digest,
                        device=device,
                        deadline=deadline,
                    )

                self.conditioning_prefetch = ConditioningPrefetch(
                    endpoint,
                    prepare=prepare,
                    identity=identity,
                    client=provider.client,
                    max_bytes=command["conditioning_prefetch_max_bytes"],
                    device=device,
                    encode_request=encode_request,
                    should_prefetch=(
                        should_prefetch if self.block0_cache is not None else None
                    ),
                )
                provider._prefetch_owner = self.conditioning_prefetch
                condition.prefetch = self.conditioning_prefetch
            except Exception:
                logger.exception("Conditioning prefetch startup failed")
                receipt = {"ok": False}
        if not all(item["ok"] for item in provider.coordinate(receipt)):
            if self.conditioning_prefetch is not None:
                self.conditioning_prefetch.close()
                self.conditioning_prefetch = None
            raise ConditionError(
                "CONDITION_UNAVAILABLE", "Conditioning prefetch startup failed"
            )

    def _admit_prompt_update(self, stage, condition, inputs=None):
        if getattr(stage, "kv_pool", None) is None:
            return
        return admit_prompt_update(
            stage,
            self.sampler.sampler,
            self.inputs if inputs is None else inputs,
            condition,
            source_shape=tuple(
                n // stage.codecs.spatial_ratio
                for n in self.source_geometry.encoded_shape
            ),
        )

    def _update_prompt(self, stage, batch):
        """Reject invalid candidates without disturbing the active generation."""
        update_started = time.perf_counter()
        notice = None
        command = getattr(batch, "extra", {}).get("zing_prompt_update", {})
        update = (
            command.get("prompt", batch.sampling_params.prompt),
            command.get("event_id", batch.realtime_event_id),
        )
        discard = getattr(self.sampler, "discard_pending_prompt", None)
        if discard is not None:
            discard()
        self.pending_block0 = None
        if command.get("rejected"):
            self.failed_update = update
            return None  # API already sent the nonfatal error for this event.
        # Reuse only the live applied identity, never a queued/prefetched one.
        # Omitted references preserve that identity. Explicit replacement is
        # deliberately conservative and goes through normal content admission.
        from sglang.multimodal_gen.runtime.conditioning.worker import (
            coordinate_validation,
        )

        live = getattr(self.sampler, "prompt", None)
        cache = getattr(self.sampler, "cache", None)
        reuse = (
            "references" not in command
            and update[0] == self.prompt
            and live is not None
            and live.event_id == self.prompt_event
            and self.sampler.applied_prompt is live
            and cache is not None
            and cache.prefix is not None
        )
        votes = coordinate_validation((reuse, self.prompt_event))
        if reuse and all(v == votes[0] and v[0] for v in votes):
            self.sampler.update_prompt(
                live.condition,
                update[1],
                getattr(batch, "realtime_prompt_version", None),
                min_source_frame=command.get("min_source_frame"),
                reuse_prefix=True,
            )
            self.prompt, self.prompt_event = update
            self.failed_update = None
            return None
        candidate = entry = None
        try:
            if self.block0_cache is not None:
                from sglang.multimodal_gen.runtime.conditioning.worker import (
                    coordinate_validation,
                )

                from .preparation_cache import inputs_on

                key_error = None
                try:
                    key, request = self._block0_update_key(stage, update[0], command)
                except (OSError, ValueError) as exc:
                    key_error = exc
                if not all(coordinate_validation(key_error is None)):
                    raise ConditionError(
                        "CONDITION_INVALID", "Reference media preparation failed"
                    ) from key_error
                entry = self.block0_cache.lookup(key)
                lookup_details = dict(getattr(self.block0_cache, "last_lookup", {}))
            if entry is not None:
                stage.admit_cached_preparation(
                    entry,
                    prompt_update=True,
                    prompt_joint=getattr(self.sampler, "prompt_joint", False),
                )
                candidate = inputs_on(
                    entry.inputs, self.inputs.positive["context"].device
                )
                positive = candidate.positive
            elif "references" in command:
                if getattr(stage, "prompt_switch_prefix_latent_frames", None) != 5:
                    raise ConditionError(
                        "CONDITION_INVALID",
                        "Reference updates require prompt prefix rebuilding",
                    )
                candidate = stage.codecs.prepare_reference_update(
                    self.inputs,
                    command["references"],
                    update[0],
                    stage._use,
                    conditioning=lambda prompt, media: stage.encode_condition_for_batch(
                        batch, prompt, media
                    ),
                )
                positive = candidate.positive
            else:
                positive = stage.encode_condition_for_batch(
                    batch, update[0], self.inputs.condition_media
                )
            # All ranks agree on capacity before any cache is changed.
            from sglang.multimodal_gen.runtime.conditioning.worker import (
                coordinate_validation,
            )

            admission_error = None
            try:
                pool_request = self._admit_prompt_update(stage, positive, candidate)
            except ConditionError as exc:
                admission_error = exc
            if not all(coordinate_validation(admission_error is None)):
                raise ConditionError(
                    "CONDITION_INVALID", "Reference update exceeds KV capacity"
                ) from admission_error
        except ConditionError as exc:
            self.failed_update = update
            notice = {"code": exc.code, "event_id": update[1]}
        else:
            if self.block0_cache is not None:
                from sglang.multimodal_gen.runtime.conditioning.worker import (
                    WorkerConditioningState,
                )

                state = batch.session.get_or_create_state(WorkerConditioningState)
                if entry is not None:
                    state.identity, state.positive = (
                        entry.conditioning_identity,
                        positive,
                    )
                prepared = replace(
                    self.inputs if candidate is None else candidate, positive=positive
                )
                self.pending_block0 = (
                    key,
                    request,
                    prepared,
                    state.identity,
                    pool_request,
                )
                self.pending_block0_prompt_version = (
                    command.get("prompt_version")
                    or getattr(batch, "realtime_prompt_version", None)
                    or self.block0_prompt_version + 1
                )
                self.pending_block0_lookup = lookup_details
                self.pending_block0_started = update_started
            kwargs = {} if candidate is None else {"inputs": candidate}
            boundary = command.get("min_source_frame")
            if boundary is not None:
                kwargs["min_source_frame"] = boundary
            self.sampler.update_prompt(
                positive,
                update[1],
                getattr(batch, "realtime_prompt_version", None),
                **kwargs,
            )
            if self.component_capture is not None:
                self.component_capture.remember_prompt(self.sampler.pending_prompt)
            self.prompt, self.prompt_event = update
            self.failed_update = None
        return notice

    @torch.no_grad()
    def run(self, stage, batch):
        self.session_id = batch.realtime_session_id
        startup = batch.extra["zing_source_command"].get("startup")
        if startup is not None:
            return self._run_startup(stage, batch, startup)
        if self.closed or batch.block_idx != self.command_index:
            raise ValueError("Continuous source commands must be consecutive")
        command = batch.extra["zing_source_command"]
        if self.startup_identity is not None:
            if not self.startup_attached:
                raise ValueError("Source is not attached")
            if (
                self.startup_protocol == PROTOCOL
                and self.ledger.accepted_frames == 0
                and command["appends"]
            ):
                size = self.ledger.width * self.ledger.height * 3
                data = command["appends"][0]["payload"]["data"][:size]
                if hashlib.sha256(data).hexdigest() != self.startup_rgb_sha256:
                    raise ValueError("Initial source frame differs from preparation")
        if self.work_quantum is None:
            self.work_quantum = source_work_quantum(command.get("work_quantum"))
        if self.decode_quantum is None:
            self.decode_quantum = source_decode_quantum(
                command.get("decode_quantum"), work_quantum=self.work_quantum
            )
        measure_cuda = command.get("measure_cuda", False)
        trace = trace_for_batch(batch)
        timings, outputs, metadata = {}, [], []
        self.progress = []  # Per-dispatch, bounded by input/bridge capacities.
        self.dispatch_steps = 0
        self.decode_steps = 0
        self.pending_output = None

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
            # Every initialized session has negotiated input/encoding/output geometry.
            assert self.source_geometry is not None
            if self.remote_decoder:
                self.decoder.bind_dispatch(
                    request_id=batch.request_id,
                    chunk_index=batch.block_idx,
                    generation_id=batch.realtime_generation_id,
                )
            self._fill_block0_after_output()
            sink_command = batch.extra.get("zing_sink_update")
            if sink_command is not None:
                if self.component_capture is not None:
                    raise ValueError(
                        "Component capture does not support dynamic sink updates"
                    )
                self.sampler.update_sink(
                    sink_command["operation"], sink_command["event_id"]
                )
            notice = None
            condition_ready = None
            applied = getattr(self.sampler, "applied_prompt", None)
            if applied is not None and applied.inputs is not None:
                self.inputs = applied.inputs
            prompt_command = batch.extra.get("zing_prompt_update", {})
            switch_end = prompt_command.get("source_boundary")
            if switch_end is not None:
                from sglang.multimodal_gen.runtime.conditioning.worker import (
                    coordinate_validation,
                )

                invalid_boundary = (
                    type(switch_end) is not int
                    or switch_end < 17
                    or switch_end % 17
                    or command["appends"]
                    or command.get("end") is not None
                    or self.ledger.accepted_frames != switch_end
                    or self.sampler.committed_pixel != switch_end
                    or self.pixels.ready
                    or self.bridge.buffered_latents
                    or self._pending_chunks()
                    or (self.decode_quantum and (self.decoder.ready or self.blocks))
                )
                if not all(coordinate_validation(not invalid_boundary)):
                    raise ValueError("The source condition barrier is not ready")
            update = (
                prompt_command.get("prompt", batch.sampling_params.prompt),
                prompt_command.get("event_id", batch.realtime_event_id),
            )
            if update != getattr(self, "failed_update", None) and update != (
                self.prompt,
                self.prompt_event,
            ):
                if self.block0_cache is not None:
                    self.block0_cache.finish_fill()
                with timed("prompt_host_ms"):
                    notice = self._update_prompt(stage, batch)
                if switch_end is not None and notice is None:
                    with timed("prompt_kv_rebuild_host_ms"):
                        condition_ready = (
                            self.sampler.activate_prompt_at_source_boundary(switch_end)
                        )
                        if self.sampler.device.type == "cuda":
                            completed_event = torch.cuda.Event()
                            completed_event.record(
                                torch.cuda.current_stream(self.sampler.device)
                            )
                            completed_event.synchronize()
                        receipt = (
                            self.session_id,
                            condition_ready["event_id"],
                            condition_ready["prompt_version"],
                            switch_end,
                        )
                        if any(
                            other != receipt for other in coordinate_validation(receipt)
                        ):
                            raise ValueError(
                                "The condition switch could not be confirmed"
                            )
                    condition_ready.update(
                        state="kv_ready", kind="prompt.update", type="control_ack"
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
                if self.component_capture is not None:
                    self.component_capture.append_frames(
                        "input", pixels, payload["source_frame_start"]
                    )
                # The shared factory may advance its cache in append/finish,
                # not only pop. Keep the declared component active throughout.
                if "remote" not in self.policy["source_encoder"]:
                    with timed("source_resize_host_ms"):
                        pixels = self.source_geometry.resize(pixels)
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
            # Continue the already committed decoder slice before scheduling
            # another DiT block. A returned slice must reach the transport now.
            if self.decode_quantum and not self.remote_decoder and self.decoder.ready:
                self._decode_ready(stage, outputs, metadata, timed, trace)
            self._generate_ready(stage, batch, outputs, metadata, timed, trace)
            # Return a completed progressive AR block before collecting more
            # encoder output when another AR block is already pending. A GPU
            # encoder handoff here can otherwise synchronize the prepared joint
            # lookahead and delay publication of the completed block. Drain the
            # remaining prefix after the last ready AR block instead, so it does
            # not require an otherwise empty RPC before accepting new source.
            while self.pixels.ready and (
                self._within_work_budget()
                or (
                    getattr(self, "remote_progressive", False)
                    and not self._pending_chunks()
                    and not self._bridge_ready()
                )
            ):
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
                    if (
                        tuple(n * stage.codecs.spatial_ratio for n in source.shape[-2:])
                        != self.source_geometry.encoded_shape
                    ):
                        raise ValueError("Source latent differs from encoded geometry")
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
                    pixel_end = source_pixel_end(
                        encoded.end,
                        self.bridge,
                        frame_end=self.ledger.accepted_frames
                        if self.end_received
                        else None,
                    )
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
                    if not self.decoder_finished:
                        self.decoder.finish(total_frames=self.ledger.accepted_frames)
                        self.decoder_finished = True
                    self._decode_ready(stage, outputs, metadata, timed, trace, eos=True)
                    self.ledger.consume(self.bridge.accepted_pixels)
                    completed = not self.decoder.ready
                    if completed:
                        self.ledger.completed()
                        if self.component_capture is not None:
                            self.component_capture.finish()
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
                            complete=completed,
                            output_frame_range=[0, self.ledger.accepted_frames],
                        )
            more_work = not completed and (
                self.pixels.ready
                or self._bridge_ready()
                or self._pending_chunks()
                or self.decoder.ready
            )
            self.command_index += 1
            if self.remote_decoder and not completed:
                # One reference per dispatch: the decoder host publishes while
                # the DiT proceeds to its next dispatch. EOS already drained.
                self._decode_ready(stage, outputs, metadata, timed, trace)
                if getattr(self, "remote_progressive", False):
                    more_work = (
                        self.pixels.ready
                        or self._bridge_ready()
                        or bool(self._pending_chunks())
                        or self.decoder.ready
                    )
            prefetch = self.conditioning_prefetch
            block0_report = self._take_block0_report(metadata)
            usage = {
                "zing_conditioning_prefetch": (
                    None
                    if prefetch is None
                    else dict(
                        encodes=prefetch.encodes,
                        hits=prefetch.hits,
                        cache_skips=prefetch.cache_skips,
                        failures=prefetch.failures,
                        fallbacks=prefetch.fallbacks,
                    )
                ),
                "zing_realtime_blocks": metadata,
                "zing_source_consumed_end": self.bridge.accepted_pixels,
                "zing_source_committed_end": self.sampler.committed_pixel,
                "zing_condition_ready": condition_ready,
                "zing_source_complete": completed,
                "zing_source_more_work": bool(more_work),
                "zing_source_dispatch_steps": self.dispatch_steps,
                "zing_source_decode_steps": self.decode_steps,
                "zing_source_stage_host_ms": timings,
                "zing_block0_cache": block0_report,
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
            if "remote" in self.policy["source_encoder"]:
                usage["zing_source_append_capacity_frames"] = (
                    self.pixels.append_capacity_frames
                )
            if self.policy.get("joint_forward", "off") != "off" or self.policy.get(
                "source_prompt_joint", False
            ):
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
                        "block0_cache": usage["zing_block0_cache"],
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
            if self.pending_output is not None:
                output = self.pending_output
                if output.reference is not None:
                    batch.sampling_params.num_frames = output.frames
                    result = OutputBatch(
                        raw_frame_shared_memory_ref=output.reference,
                        raw_frame_metadata=output.transport_metadata,
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
            elif outputs:
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

    def _run_startup(self, stage, batch, startup):
        from sglang.multimodal_gen.runtime.realtime.source_prepare import (
            preparation_identity,
            validate_preview_rgb,
        )

        try:
            identity = preparation_identity(startup.get("identity"))
            protocol = startup.get("protocol", PROTOCOL)
            if protocol not in {PROTOCOL, V2_PROTOCOL}:
                raise ValueError("Unsupported startup protocol")
            if self.startup_protocol is not None and protocol != self.startup_protocol:
                raise ValueError("Preparation protocol changed")
            command = batch.extra["zing_source_command"]
            if (
                self.closed
                or command["appends"]
                or command.get("end")
                or self.command_index
            ):
                raise ValueError("Preparation requires an unused source generation")
            if self.startup_identity is not None and identity != self.startup_identity:
                raise ValueError("Preparation identity mismatch")
            if startup["op"] == "prepare":
                config = dict(command["config"])
                rgb = startup["rgb"]
                width, height = config["width"], config["height"]
                if protocol == V2_PROTOCOL:
                    width, height = startup.get("width"), startup.get("height")
                validate_preview_rgb(rgb, width, height)
                digest = hashlib.sha256(rgb).hexdigest()
                preview_key = (width, height, digest)
                if self.ledger is None:
                    config["limits"] = SourceLimits(**config["limits"])
                    self.initialize(stage, batch, config, trace_for_batch(batch))
                    self.startup_identity, self.startup_rgb_sha256 = identity, digest
                    self.startup_protocol = protocol
                    self.startup_preview_key = preview_key
                    if self.remote_decoder:
                        self.decoder.bind_dispatch(
                            request_id=batch.request_id,
                            chunk_index=batch.block_idx,
                            generation_id=batch.realtime_generation_id,
                        )
                elif preview_key != self.startup_preview_key:
                    raise ValueError("Preparation pixels changed")
                # Readiness includes this rank's CUDA dependencies, not merely
                # constructing Python state. No sampler/source/decoder step runs.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                receipt = dict(
                    type="source.prepared", identity=identity, level="condition"
                )
            elif startup["op"] == "attach" and self.startup_identity is not None:
                self.startup_attached = True
                receipt = dict(type="source.attached", identity=identity)
            else:
                raise ValueError("Invalid preparation transition")
            from sglang.multimodal_gen.runtime.conditioning.worker import (
                coordinate_validation,
            )

            if any(value != receipt for value in coordinate_validation(receipt)):
                raise ValueError("Source preparation ranks disagree")
            return OutputBatch(
                raw_frame_batches=[],
                metrics=batch.metrics,
                usage={
                    "zing_startup_ready": receipt,
                    "zing_source_consumed_end": 0,
                    "zing_source_committed_end": 0,
                    "zing_source_complete": False,
                    "zing_realtime_blocks": [],
                },
            )
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
        return (
            (not getattr(self, "remote_progressive", False) or self.dispatch_steps < 1)
            and (not self.work_quantum or self.dispatch_steps < self.work_quantum)
            and (
                self.remote_decoder
                or not self.decode_quantum
                or (not self.decoder.ready and self.decode_steps < self.decode_quantum)
            )
        )

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
                self.component_capture.append(chunk, result, prompt=self.sampler.prompt)
            self.blocks.append(
                {
                    "block_idx": result.chunk.chunk_id,
                    "request_id": batch.request_id,
                    "event_id": result.prompt_event_id,
                    "prompt_version": result.prompt_version,
                    "sink_event_id": getattr(result, "sink_event_id", None),
                    "sink_chunks": getattr(result, "sink_chunks", 0),
                    "sink_chunk_ids": getattr(result, "sink_chunk_ids", ()),
                    "frame_end": result.provenance.pixels.end,
                }
            )
            self.decoder.append(
                result.visual,
                latent_start=result.chunk.start,
                source_dependency_end=result.provenance.source_dependency_end,
                valid_frame_end=result.provenance.pixels.end,
            )
            if not self.remote_decoder or (
                getattr(self, "remote_progressive", False) and not self.end_received
            ):
                # Start D1 before collecting a future encoder prefix. At EOS,
                # retain the single final pop below: finish may itself need a
                # receipt and must not replace this dispatch's RGB reference.
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
            if (
                not flush
                and len(self.sampler.pending) < capacity
                and not getattr(self, "remote_progressive", False)
            ):
                return
            # One-output progressive dispatches must not stall a ready block
            # on the next encoder pop merely to fill the joint ready queue.
            # The transformer lease ends before every decoder call. Closing on
            # failure clears any prefetched next active target as well.
            ready_iterator = (
                self.sampler.drain_ready(max_outputs=1)
                if getattr(self, "remote_progressive", False)
                else self.sampler.drain_ready()
            )
            with closing(ready_iterator) as ready:
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
                        self.component_capture.append(
                            capture_chunk, result, prompt=self.sampler.prompt
                        )
                        del capture_chunk
                    self.blocks.append(
                        {
                            "block_idx": result.chunk.chunk_id,
                            "request_id": batch.request_id,
                            "event_id": result.prompt_event_id,
                            "prompt_version": result.prompt_version,
                            "sink_event_id": getattr(result, "sink_event_id", None),
                            "sink_chunks": getattr(result, "sink_chunks", 0),
                            "sink_chunk_ids": getattr(result, "sink_chunk_ids", ()),
                            "frame_end": result.provenance.pixels.end,
                        }
                    )
                    self.decoder.append(
                        result.visual,
                        latent_start=result.chunk.start,
                        source_dependency_end=result.provenance.source_dependency_end,
                        valid_frame_end=result.provenance.pixels.end,
                    )
                    if not self.remote_decoder or (
                        getattr(self, "remote_progressive", False)
                        and not self.end_received
                    ):
                        self._decode_ready(stage, outputs, metadata, timed, trace)

    def _decode_ready(self, stage, outputs, metadata, timed, trace=None, *, eos=False):
        while self.decoder.ready and (
            self.remote_decoder
            or not self.decode_quantum
            or self.decode_steps < self.decode_quantum
        ):
            latent_end = getattr(self.decoder, "accepted_latents", None)
            span_fields = {"final": bool(eos)}
            if latent_end is not None:
                span_fields.update(
                    input_ids=[f"{self.session_id}:generated-latent:0:{latent_end}"],
                    input_latent_range=[0, latent_end],
                )
            if self.remote_decoder:
                span_fields["decoder_role"] = "dedicated"
            with stage_span(
                trace,
                "Decoder",
                cuda=trace is not None and trace.measure_cuda,
                **span_fields,
            ) as span:
                with (
                    timed("decoder_host_ms"),
                    (
                        nullcontext()
                        if self.remote_decoder
                        else stage._use(stage.codecs.video_decoder_name, "decode")
                    ),
                    maybe_record_function("Zing.source_decode"),
                ):
                    with maybe_record_function(
                        "Zing.source_decode.eos" if eos else "Zing.source_decode.chunk"
                    ):
                        decoded = (
                            self.decoder.pop(max_latents=1)
                            if self.decode_quantum and not self.remote_decoder
                            else self.decoder.pop()
                        )
                        self.decode_steps += 1
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
            if decoded.values is None:
                # The decoder host publishes the pixels; only the reference
                # and the per-block metadata cross back to the DiT.
                self.pending_output = decoded
            elif self.component_capture is not None:
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
                if decoded.values is not None:
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

    def _take_block0_report(self, metadata):
        """Publish the matching version once, using native transfer events only.

        PrefixRestore exposes H2D begin/end events, not consumer-wait events.
        Incomplete GPU timing stays absent; reporting must never synchronize.
        """
        block0_report = None
        if (
            self.block0_telemetry is not None
            and not self.block0_reported
            and metadata
            and any(
                (m.get("prompt_version") or 1) == self.block0_prompt_version
                for m in metadata
            )
        ):
            self.block0_telemetry.finish()
            transfer = self.block0_restore
            if (
                transfer is not None
                and transfer.end is not None
                and transfer.end.query()
            ):
                self.block0_telemetry.cuda["restore"] = transfer.begin.elapsed_time(
                    transfer.end
                )
            block0_report = {
                "key": self.block0_key,
                "cache_key": self.block0_key,
                "prompt_version": self.block0_prompt_version,
                **self.block0_telemetry.snapshot(),
            }
            self.block0_reported = True
        return block0_report

    def dispose(self):
        # Revoke computation immediately, but keep every failed resource owner
        # reachable. A failed abort cannot authorize reuse of its pooled KV.
        self.closed = True
        failures = []
        try:
            if self.block0_cache is not None:
                self.block0_cache.finish_fill()
                transfer = self.block0_restore
                if transfer is not None:
                    transfer.wait()
                logger.info(
                    "zing_block0_cache %s",
                    json.dumps(
                        dict(
                            mode=self.block0_cache.mode,
                            key=self.block0_key,
                            hit=self.block0_entry is not None,
                            prepare_bypassed=self.block0_entry is not None,
                            lookup_host_ms=self.block0_lookup_ms,
                            restore_submit_host_ms=None
                            if transfer is None
                            else transfer.submit_ms,
                            restore_cuda_ms=None
                            if transfer is None
                            else transfer.cuda_ms,
                            ready_at_consumer=None
                            if transfer is None
                            else transfer.ready_at_consumer,
                            restore_lead_host_ms=None
                            if transfer is None
                            else transfer.lead_ms,
                            cpu_capacity_bytes=self.block0_cache.cpu_capacity_bytes,
                            page_rows=getattr(self.block0_cache, "page_rows", None),
                            total_pages=getattr(self.block0_cache, "total_pages", None),
                            free_pages=(
                                self.block0_cache.free_pages.count(True)
                                if hasattr(self.block0_cache, "free_pages")
                                else None
                            ),
                            ready=bool(self.block0_cache.entry_count),
                            entries=self.block0_cache.entry_count,
                            evictions=self.block0_cache.evictions,
                            hits=self.block0_cache.hits,
                            misses=self.block0_cache.misses,
                            rank=(
                                torch.distributed.get_rank()
                                if torch.distributed.is_initialized()
                                else 0
                            ),
                        )
                    ),
                    main_process_only=False,
                    local_main_process_only=False,
                )
        except Exception as error:
            failures.append(("block0", error))
        try:
            if self.conditioning_prefetch is not None:
                self.conditioning_prefetch.close()
                self.conditioning_prefetch = None
        except Exception as error:
            failures.append(("conditioning", error))
        try:
            self._dispose_states()
        except Exception as error:
            failures.append(("source_roles", error))
        if failures:
            raise RuntimeError(
                "Source cleanup remains unconfirmed: "
                + ", ".join(name for name, _ in failures)
            ) from failures[0][1]
        self.block0_cache = self.block0_entry = self.block0_restore = None
        self.block0_inputs = self.block0_prefix = None
        self.pending_block0 = self.block0_key_args = None
        self.block0_input_request = None
        if self.kv_lease is not None:
            self.kv_lease.release()
            self.kv_lease = None

    def _dispose_states(self):
        owners = {
            name: getattr(self, name)
            for name in ("sampler", "ledger", "pixels", "bridge", "decoder")
        }
        failures = []
        for name, state in owners.items():
            if state is None:
                continue
            try:
                state.dispose()
            except Exception as error:
                failures.append((name, error))
            else:
                setattr(self, name, None)
        if failures:
            raise RuntimeError(
                "Source role cleanup failed: " + ", ".join(name for name, _ in failures)
            ) from failures[0][1]
        sampler, pixels, bridge, decoder, ledger = (
            owners[name]
            for name in ("sampler", "pixels", "bridge", "decoder", "ledger")
        )
        if sampler is not None:
            memory = sampler.memory_snapshot()
            memory.update(
                codec_pixel_frames=pixels.buffered_frames if pixels else 0,
                bridge_latents=bridge.buffered_latents if bridge else 0,
                decoder_latents=decoder.buffered_latents if decoder else 0,
                metadata_batches=len(ledger.records) if ledger else 0,
            )
            if (
                pixels is not None
                and self.policy["source_encoder"]["mode"] == "progressive"
            ):
                memory["encoder_state"] = vars(pixels.state_usage())
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
