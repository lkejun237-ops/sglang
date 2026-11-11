# SPDX-License-Identifier: Apache-2.0
"""Version-one RGB source on the shared realtime scheduler and transport."""

import asyncio
import hashlib
import json
import logging
import math
import os
import tempfile
import time
import uuid
from collections import deque
from copy import copy, deepcopy
from dataclasses import asdict
from pathlib import Path

import msgspec.msgpack
import zmq
import zmq.asyncio
from PIL import Image  # noqa: F401 - preserve the existing CPU validation seam

from sglang.multimodal_gen.configs.zing_sampling import (
    normalize_source_sampling_schedule,
)
from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.disaggregation.role_launcher import (
    _with_disagg_role_policies,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    BaseRealtimeModelAdapter,
    RealtimeChunkInputs,
    build_realtime_sampling_params,
)
from sglang.multimodal_gen.runtime.realtime.source_prepare import (
    PROTOCOL as PREPARE_PROTOCOL,
)
from sglang.multimodal_gen.runtime.realtime.source_prepare import (
    STARTUP_SECONDS,
    V2_PROTOCOL,
    condition_fingerprint,
    preparation_identity,
    rgb_digest,
    validate_preview_rgb,
)
from sglang.multimodal_gen.runtime.realtime.source_transport import (
    PENDING_BATCHES,
    SourceTransportBudget,
    check,
    validate_source_dimensions,
)
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    DEFAULT_MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_IMAGE_BYTES,
    PixelRange,
    ReferenceLimits,
    require,
)
from sglang.multimodal_gen.runtime.realtime.zing_input_budget import (
    ZingInputBudget,
    count_prompt_tokens,
)
from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
    SourceLedger,
    SourceLimits,
    default_source_encoder,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

from .zing_reference_upload import prepare_reference_upload


def _with_source_decoder_policy(profile, port):
    """Overlay the launcher-assigned decoder port on the attested export."""
    return _with_disagg_role_policies(profile, decoder_port=port)


def _with_source_pipeline_policy(
    profile,
    *,
    encoder_mode=None,
    encoder_prefetch=None,
    async_output=None,
    joint_forward=None,
):
    """Overlay deployment-only switches without mutating the exported profile."""
    result = deepcopy(profile)
    remote = result.get("source_encoder", {}).get("remote")
    if encoder_mode is not None or encoder_prefetch is True:
        if remote is None:
            raise ValueError("Source encoder CLI policy requires a remote Encoder")
    if remote is not None:
        remote["mode"] = encoder_mode if encoder_mode is not None else "overlap"
        if encoder_prefetch is None:
            encoder_prefetch = remote["mode"] == "overlap"
    if encoder_prefetch is not None and remote is not None:
        remote["cross_dispatch_prefetch"] = encoder_prefetch
    if async_output is not None:
        result["async_source_output"] = async_output
    if joint_forward is not None:
        result["joint_forward"] = joint_forward
    return result


logger = logging.getLogger(__name__)


def _source_sampling_schedule(args, profile):
    """Freeze a whole deployment plan, with explicit CLI settings taking precedence."""
    profile_has_initial = "initial_block_count" in profile
    profile = normalize_source_sampling_schedule(profile)
    solver = getattr(args, "zing_solver", None)
    count = getattr(args, "zing_initial_block_count", 0)
    is_explicit = getattr(args, "is_arg_explicitly_set", None)
    explicit_cli = (
        any(
            is_explicit(name)
            for name in (
                "zing_solver",
                "zing_denoising_step_list",
                "zing_initial_block_count",
                "zing_initial_denoising_step_list",
            )
        )
        if is_explicit is not None
        else solver is not None or count == 3
    )
    if not explicit_cli:
        if profile_has_initial:
            return profile
        # Legacy profiles name only the regular grid. An explicit count=0 is
        # now required to retain a single-stage deployment.
        return dict(
            solver="few_step",
            denoising_step_list=[1000, 500],
            initial_block_count=3,
            initial_denoising_step_list=[1000, 750, 500, 250],
        )
    grid = (
        getattr(args, "zing_denoising_step_list", None)
        if solver is not None
        else profile["denoising_step_list"]
    )
    return normalize_source_sampling_schedule(
        dict(
            solver=solver or profile["solver"],
            denoising_step_list=grid,
            initial_block_count=count,
            initial_denoising_step_list=getattr(
                args, "zing_initial_denoising_step_list", None
            ),
        )
    )


class ZingH3SourceStreamAdapter(BaseRealtimeModelAdapter):
    supports_sink_updates = True

    def create_state(self):
        return None  # Initialized atomically after negotiation validation.

    async def on_init(self, session, request, *, prefill=False):
        self.init_started_ns = time.time_ns()
        self.startup_protocol = request.source_prepare_protocol
        self.startup_enabled = self.startup_protocol is not None
        self.startup_phase = "waiting" if self.startup_enabled else None
        self.startup_identity = self.startup_command = None
        self.startup_deadline = time.monotonic() + STARTUP_SECONDS
        self.worker_initialized = False
        self._source_reservations, self._inflight_reservations = {}, []
        self._s0_reservation = None
        self._pending_reference_upload = None
        self._inflight_reference_uploads = []
        self._pending_prompt_command = self._selected_prompt_command = None
        self._pending_sink_command = None
        self.prompt_prefetch = None
        self.conditioning_prefetch_endpoint = None
        self.initial_prompt = request.prompt
        require(
            not self.startup_enabled
            or os.environ.get("SGLANG_ZING_SOURCE_PREPARE") == "1",
            "This worker does not support early preparation.",
        )
        require(
            not self.startup_enabled
            or type(request.seed) is int
            and 0 <= request.seed < 2**32,
            "Please provide a fixed model seed for early preparation.",
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.cache_index import (
            validate_labels,
        )

        validate_labels(request.game_id, request.group_id)
        self.cache_options = dict(
            cache_write=request.cache_write,
            game_id=request.game_id,
            group_id=request.group_id,
        )
        args = get_global_server_args()
        idle_seconds = getattr(args, "realtime_session_idle_timeout_s", STARTUP_SECONDS)
        require(
            not self.startup_enabled
            or idle_seconds == 0
            or idle_seconds >= STARTUP_SECONDS,
            "This worker's session timeout is too short for early preparation.",
        )
        from ..conditioning import get_api_resources, validate_remote_request

        validate_remote_request(request, args)
        get_api_resources(session, args)
        require(
            getattr(args, "realtime_vae_backend", "local") == "local"
            and not session.vae_worker_url,
            "This model requires local video decoding.",
        )
        require(
            type(getattr(request, "zing_realtime_api_version", None)) is int
            and request.zing_realtime_api_version == 1,
            "Please select streaming API version 1.",
        )
        for name in (
            "zing_request",
            "prompt_video_frame",
            "source_preview",
            "first_frame",
            "video_path",
            "video_url",
            "task",
            "conditions",
            "target",
            "source",
            "max_chunks",
            "generation_mode",
            "condition_inputs",
            "zing_window_chunks",
            "zing_sink_chunks",
            "zing_rope_position_mode",
            "zing_rope_max_frame_gap",
            "zing_window_eviction_timing",
            "zing_solver",
            "zing_denoising_step_list",
            "zing_initial_block_count",
            "zing_initial_denoising_step_list",
            "realtime_causal_sink_size",
            "realtime_causal_kv_cache_num_frames",
            "negative_prompt",
            "diffusers_kwargs",
        ):
            require(
                getattr(request, name, None) is None,
                "Please use the continuous source and reference fields for this session.",
            )
        require(
            request.prompt.strip()
            and request.n in (None, 1)
            and request.num_outputs_per_prompt in (None, 1),
            "Please request one video with a prompt.",
        )
        require(
            getattr(request, "zing_realtime_decode_mode", "stable") == "stable",
            "This session requires stable rendering.",
        )
        require(
            not any(
                (
                    request.enable_frame_interpolation,
                    request.enable_upscaling,
                    request.enable_teacache,
                    getattr(request, "zing_generate_audio", False),
                )
            ),
            "Video enhancement and audio are not supported for this session.",
        )
        config_bytes = (Path(args.model_path) / "zing_config.json").read_bytes()
        runtime = json.loads(config_bytes)
        # Explicit deployment attestation binds solver/schedule to this model
        # export. A request cannot silently override checkpoint semantics.
        profile = json.loads(
            (Path(args.model_path) / "zing_realtime_profile.json").read_text()
        )
        profile = _with_disagg_role_policies(
            profile,
            encoder_port=getattr(args, "disagg_encoder_port", None),
            decoder_port=getattr(args, "disagg_decoder_port", None),
            encoder_ports=getattr(args, "disagg_encoder_ports", None),
            decoder_ports=getattr(args, "disagg_decoder_ports", None),
        )
        profile = _with_source_pipeline_policy(
            profile,
            encoder_mode=getattr(args, "zing_source_encoder_mode", None),
            encoder_prefetch=getattr(args, "zing_source_encoder_prefetch", None),
            async_output=getattr(args, "zing_source_async_output", None),
            joint_forward=getattr(args, "zing_source_joint_forward", None),
        )
        # require raises ContractError explicitly and is unaffected by python -O.
        require(
            runtime.get("causal") is True
            and profile.get("model_config_sha256")
            == hashlib.sha256(config_bytes).hexdigest()
            and profile.get("solver") == "few_step"
            and isinstance(profile.get("denoising_step_list"), list)
            and profile.get("noise_version") == "indexed-cpu-v1",
            "The deployed model is not configured for streaming.",
        )
        self.sampling_schedule = _source_sampling_schedule(args, profile)
        regular_steps = len(self.sampling_schedule["denoising_step_list"])
        require(
            request.num_inference_steps in (None, regular_steps),
            "The requested step count does not match this deployment. "
            "Omit it to use the server settings.",
        )
        require(
            profile.get("window_eviction_timing", "after_step0")
            in ("before_source", "after_step0"),
            "The deployed streaming eviction settings are invalid.",
        )
        joint_mode = profile.get("joint_forward", "cross_chunk")
        wait_ms = profile.get("joint_wait_ms", "auto")
        require(
            joint_mode in ("off", "source_first", "cross_chunk")
            and (
                wait_ms == "auto"
                or (
                    type(wait_ms) in (int, float)
                    and math.isfinite(wait_ms)
                    and 0 < wait_ms <= 1000
                    and joint_mode == "cross_chunk"
                )
            ),
            "The deployed streaming settings are invalid.",
        )
        self.joint_wait = None
        self.joint_plan = ChunkPlan(
            runtime.get("num_frame_per_block", 5),
            runtime.get("split_first_latent", False),
        )
        self.dispatched_source_frames = 0
        self.joint_chunk_id = 0
        require(
            "guidance_scale" not in request.model_fields_set
            or request.guidance_scale == profile["guidance_scale"],
            "The requested guidance does not match this model.",
        )
        # Only the server deployment profile selects the window; clients cannot override it.
        window = profile.get(
            "source_attention_window", {"window_chunks": 21, "sink_chunks": 2}
        )
        require(
            isinstance(window, dict)
            and set(window) == {"window_chunks", "sink_chunks"}
            and type(window["window_chunks"]) is int
            and type(window["sink_chunks"]) is int
            and 1 <= window["window_chunks"] <= 21
            and 0 <= window["sink_chunks"] < window["window_chunks"],
            "The deployed model is not configured for streaming.",
        )
        self.source_attention_window = dict(window)
        self.sink_updates = (
            self.supports_sink_updates
            and self.joint_plan.lengths == (3, 2)
            and not self.joint_plan.split_first_latent
            and window["window_chunks"] >= 4
            and window["sink_chunks"] in (0, 2)
        )
        source = getattr(request, "source_stream", None)
        target_width, target_height = map(int, request.size.split("x"))
        fps = int(runtime["h3_media_config"]["fps"])
        require(
            isinstance(source, dict)
            and set(source) == {"format", "width", "height", "fps"}
            and source["format"] == "rgb24"
            and source["fps"] == fps
            and all(
                type(source[k]) is int and source[k] > 0 for k in ("width", "height")
            )
            and request.fps in (None, fps),
            "Please provide matching RGB source dimensions and frame rate.",
        )
        width, height = source["width"], source["height"]
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.media_geometry import (
            resolve_source_geometry,
        )

        validate_source_dimensions(target_width, target_height, args.pipeline_config)
        try:
            self.source_geometry = resolve_source_geometry(
                (height, width),
                (target_height, target_width),
                args.pipeline_config,
                runtime["h3_media_config"],
            )
        except ValueError as exc:
            logger.warning("Source geometry rejected: %s", exc)
            check(
                False,
                "source_geometry_unsupported",
                "This video size is not supported by the server.",
            )
        transport = SourceTransportBudget.from_args(args)
        transport.validate_sessions(
            max(
                getattr(args, "realtime_max_sessions", 8),
                getattr(args, "realtime_max_sessions_per_worker", 8),
            )
        )
        transport = transport.negotiate(
            getattr(request, "source_message_budget_bytes", None)
        )
        self.max_message_bytes = transport.message_bytes
        clip = int(args.pipeline_config.vae_config.arch_config.vae_clip_length)
        message_bytes = transport.source_message_bytes(width, height, clip)
        limits = SourceLimits(
            clip,
            message_bytes,
            PENDING_BATCHES * clip,
            PENDING_BATCHES * clip * width * height * 3,
            8 * clip,
        )
        require(
            request.realtime_preview_max_width in (None, target_width),
            "This session requires the full output dimensions.",
        )
        self.max_references = getattr(
            args.pipeline_config,
            "zing_reference_max_images",
            DEFAULT_MAX_REFERENCE_IMAGES,
        )
        reference_limits = ReferenceLimits(self.max_references)
        references = getattr(request, "references", []) or []
        require(
            isinstance(references, list) and len(references) <= self.max_references,
            f"Please provide at most {self.max_references} reference images.",
        )
        from ..conditioning import get_api_resources

        self.input_budget = ZingInputBudget.from_config(args.pipeline_config)
        prompt_tokens = await get_api_resources(session, args).cpu.run(
            count_prompt_tokens, args.model_path, request.prompt
        )
        self.input_budget.validate(prompt_tokens, len(references))
        self._budget_reference_count = len(references)
        self._budget_reference_update = None
        self.reference_max_decoded_bytes = getattr(
            args.pipeline_config, "zing_reference_max_decoded_bytes", None
        )
        if self.reference_max_decoded_bytes is None:
            self.reference_max_decoded_bytes = reference_limits.decoded_reference_bytes
        reference_decoded_bytes = 0
        seen, media_bytes = set(), 0
        self.reference_items = []
        for item in references:
            require(
                isinstance(item, dict)
                and set(item) == {"id", "format", "payload"}
                and isinstance(item.get("id"), str)
                and item["id"]
                and item["id"] not in seen
                and item.get("format") in ("jpeg", "webp", "png")
                and isinstance(item.get("payload"), bytes),
                "Please provide uniquely named reference images with supported image bytes.",
            )
            require(
                item["id"] != "<Video 1>",
                "Please use a different name for this image.",
            )
            seen.add(item["id"])
            require(
                len(item["payload"]) <= MAX_REFERENCE_IMAGE_BYTES,
                "The reference image is too large.",
            )
            media_bytes += len(item["payload"])
            require(
                media_bytes <= reference_limits.reference_bytes,
                "The reference images are too large.",
            )
            from ..conditioning import reserve_media_memory, verify_presentation_upload

            decoded_bytes = await verify_presentation_upload(
                session, item["payload"], item["format"]
            )
            if self.reference_max_decoded_bytes is not None:
                reference_decoded_bytes += decoded_bytes
                require(
                    reference_decoded_bytes <= self.reference_max_decoded_bytes,
                    "The reference images exceed the configured decoded size limit.",
                )
            reserve_media_memory(session, len(item["payload"]))
            # Keep caller-supplied encoded bytes; init and updates share the
            # worker's in-memory image decoder. No temporary media files.
            descriptor = dict(
                type="image",
                id=item["id"],
                format=item["format"],
                payload=item["payload"],
            )
            self.reference_items.append(descriptor)
        self.source_preview_sha256 = None
        self.source_preview_bytes = None
        self.config = dict(
            width=width,
            height=height,
            fps=fps,
            limits=asdict(limits),
            total_frames=request.num_frames,
        )
        self.ledger = SourceLedger(**{**self.config, "limits": limits})
        self.pending, self.available = deque(), asyncio.Event()
        self._source_reservations, self._inflight_reservations = {}, []
        self._s0_reservation = None
        self.pending_frames, self.end, self.end_dispatched = 0, None, False
        from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
            remote_decode_progressive,
            source_decode_quantum,
            source_work_quantum,
        )

        self.source_work_quantum = source_work_quantum(
            getattr(args, "zing_source_work_quantum", None)
        )
        self.source_defer_final = getattr(args, "zing_source_defer_final", None)
        if self.source_defer_final is None:
            self.source_defer_final = (
                os.environ.get("SGLANG_ZING_SOURCE_DEFER_FINAL", "0") == "1"
            )
        self.source_prompt_joint = getattr(args, "zing_source_prompt_joint", None)
        if self.source_prompt_joint is None:
            self.source_prompt_joint = (
                os.environ.get("SGLANG_ZING_SOURCE_PROMPT_JOINT", "0") == "1"
            )
        self.remote_progressive = (
            bool(profile.get("source_decoder", {}).get("remote"))
            and remote_decode_progressive()
        )
        if (
            self.source_defer_final
            and self.source_work_quantum != 1
            and not self.remote_progressive
        ):
            raise ValueError("Deferred source final currently requires work quantum 1")
        self.source_decode_quantum = source_decode_quantum(
            work_quantum=self.source_work_quantum
        )
        self.source_decoder_overlap = getattr(args, "zing_source_decoder_overlap", None)
        if self.source_decoder_overlap is None:
            self.source_decoder_overlap = (
                self.source_work_quantum == 1
                and "remote" in profile.get("source_decoder", {})
                and getattr(args, "realtime_max_sessions", 1) == 1
                and getattr(args, "realtime_max_sessions_per_worker", 1) == 1
            )
        if self.source_decoder_overlap and (
            self.source_work_quantum != 1
            or "remote" not in profile.get("source_decoder", {})
        ):
            raise ValueError(
                "Source decoder overlap requires a dedicated decoder and source quantum 1"
            )
        self.more_work = False
        self.clip, self.profile = clip, profile
        self.reference_ids = [ref["id"] for ref in self.reference_items]
        self.reference_updates = getattr(
            args.pipeline_config, "prompt_switch_prefix_latent_frames", None
        ) == 5 and self.joint_plan.lengths == (3, 2)
        self.source_encoder = profile.get("source_encoder", default_source_encoder())
        self.source_append_overlap = bool(
            self.source_work_quantum
            and self.source_encoder.get("remote", {}).get("cross_dispatch_prefetch")
        )
        self.append_capacity_frames = None  # First dispatch uses existing limits.
        self.pending_source_consumed_end = 0
        self.admission_socket = self.admission_context = None
        if not prefill:
            self._initialize_source_admission(session, limits)
        conditioning_prefetch = getattr(args, "zing_conditioning_prefetch", None)
        if conditioning_prefetch is None:
            conditioning_prefetch = (
                getattr(args, "zing_conditioning_mode", "local") == "remote"
                and getattr(args, "realtime_max_sessions", 1) == 1
                and getattr(args, "realtime_max_sessions_per_worker", 1) == 1
            )
        if not prefill and conditioning_prefetch:
            from sglang.multimodal_gen.runtime.conditioning.prefetch import (
                PromptPrefetchPublisher,
            )

            if session.input_temp_dir is None:
                session.input_temp_dir = tempfile.mkdtemp(
                    prefix="sglang_zing_source_", dir="/tmp"
                )
            self.conditioning_prefetch_endpoint = "ipc://" + str(
                Path(session.input_temp_dir) / (uuid.uuid4().hex + ".sock")
            )
            resources = get_api_resources(session, args)
            self.prompt_prefetch = PromptPrefetchPublisher(
                self.conditioning_prefetch_endpoint,
                max_bytes=self.max_message_bytes,
                reserve=lambda size: resources.ledger.reserve(
                    resources.tenant, ram=size, inflight=size
                ),
            )
        self.first_source_received_unix_ns = None
        self.prompt_event = None
        self.ordered_prompts = (
            getattr(request, "source_prompt_protocol", None) == "ordered-boundary-v2"
        )
        self.prompt_boundary = None
        self.committed_source_frames = 0
        self._source_switch_end = None
        self.condition_switch_protocol = getattr(
            request, "source_condition_switch_protocol", None
        )
        require(
            self.condition_switch_protocol is None
            or (
                self.ordered_prompts
                and self.reference_updates
                and (
                    getattr(args.pipeline_config, "decoder_type", None)
                    == "lynn_streaming"
                    or getattr(self, "external_condition_barrier", False)
                )
            ),
            "This session does not support source condition switching.",
        )
        self.public_prompt_version = 1
        self.prompt_inflight = None
        request.fps, request.num_inference_steps, request.guidance_scale = (
            fps,
            regular_steps,
            profile["guidance_scale"],
        )
        request.realtime_output_format = request.realtime_output_format or "raw"
        request.realtime_preview_max_width = target_width
        self.ready = {
            "type": "session.ready",
            "zing_realtime_api_version": 1,
            "session_id": session.id,
            "num_inference_steps": regular_steps,
            "sampling_schedule": self.sampling_schedule,
            "realtime_output_format": request.realtime_output_format,
            "source_stream": source,
            "source_encoding": self.source_geometry.receipt(),
            "max_message_bytes": transport.message_bytes,
            "max_source_batch_frames": limits.batch_frames,
            "max_source_message_bytes": limits.message_bytes,
            "max_pending_source_frames": limits.pending_frames,
            "max_pending_source_bytes": limits.pending_bytes,
            "max_references": self.max_references,
            "sink_updates": self.sink_updates,
            "input_budget": self.input_budget.metadata(),
            "prompt_reference_updates": self.reference_updates,
            "max_reference_bytes": reference_limits.reference_bytes,
            "max_reference_image_bytes": MAX_REFERENCE_IMAGE_BYTES,
            "max_reference_decoded_bytes": self.reference_max_decoded_bytes,
        }
        if self.startup_enabled:
            self.ready["source_prepare_protocol"] = self.startup_protocol
            self.preparation_profile = hashlib.sha256(
                json.dumps(
                    dict(
                        profile=profile,
                        sampling_schedule=self.sampling_schedule,
                        model_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                        source_work_quantum=self.source_work_quantum,
                        source_decode_quantum=self.source_decode_quantum,
                        defer_final=self.source_defer_final,
                        prompt_joint=self.source_prompt_joint,
                        remote_progressive=self.remote_progressive,
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            self.ready["source_prepare_profile_sha256"] = self.preparation_profile
        if self.ordered_prompts:
            self.ready["source_prompt_protocol"] = "ordered-boundary-v2"
        if self.condition_switch_protocol:
            self.ready["source_condition_switch_protocol"] = (
                self.condition_switch_protocol
            )
        if getattr(args.pipeline_config, "decoder_type", None) == "lynn_streaming":
            self.ready["source_output_release_protocol"] = "valid-source-frames-v1"

    def _initialize_source_admission(self, session, limits):
        """Initialize private source prefetch; ledger and Encoder still enforce capacity."""
        self.admission_socket = self.admission_context = None
        remote = self.source_encoder.get("remote", {})
        if remote.get("cross_dispatch_prefetch", False):
            require(
                remote.get("mode") == "overlap",
                "Source prefetch requires an overlap deployment.",
            )
            if session.input_temp_dir is None:
                session.input_temp_dir = tempfile.mkdtemp(
                    prefix="sglang_zing_source_", dir="/tmp"
                )
            endpoint = "ipc://" + str(
                Path(session.input_temp_dir) / (uuid.uuid4().hex + ".sock")
            )
            self.source_encoder = {
                **self.source_encoder,
                "remote": {**remote, "admission_endpoint": endpoint},
            }
            self.admission_context = zmq.asyncio.Context()
            self.admission_socket = self.admission_context.socket(zmq.PUSH)
            self.admission_socket.setsockopt(zmq.LINGER, 0)
            self.admission_socket.setsockopt(zmq.SNDHWM, limits.pending_frames + 1)
            try:
                # PUSH/PULL sends accepted inputs one way; LINGER=0 avoids waiting on shutdown.
                # SNDHWM bounds messages, not the separately validated frame or byte budgets.
                self.admission_socket.bind(endpoint)
            except BaseException:
                self.admission_socket.close()
                self.admission_context.term()
                self.admission_socket = self.admission_context = None
                raise

    def input_receipt(self, session, kind, value, *, event_id=None, request_id=None):
        if not getattr(getattr(session, "request", None), "record_model_inputs", False):
            return {}
        from sglang.multimodal_gen.runtime.realtime.model_input_recording import (
            accepted_input,
            input_snapshot,
        )

        return {
            "model_input": accepted_input(
                session,
                input_snapshot(kind, value, event_id=event_id, request_id=request_id),
            )
        }

    def session_ready_message(self, session):
        if not getattr(getattr(session, "request", None), "record_model_inputs", False):
            return self.ready
        if not hasattr(self, "_model_input_ready"):
            self._model_input_ready = self.input_receipt(
                session,
                "init",
                dict(
                    prompt=self.initial_prompt,
                    references=self.reference_items,
                    seed=session.request.seed,
                    size=session.request.size,
                    source_stream=session.request.source_stream,
                ),
            )
        return {**self.ready, **self._model_input_ready}

    async def _admit_source(self, op, payload):
        if self.admission_socket is None:
            return
        # append carries frame start/count/shape and a separate RGB part; finish carries EOS.
        # Headers validate order, capacity and EOS separately from open/pop/close RPCs.
        header = {
            "op": op,
            "width": self.config["width"],
            "height": self.config["height"],
            **{key: value for key, value in payload.items() if key != "data"},
        }
        parts = [json.dumps(header).encode()]
        if op == "append":
            parts.append(payload["data"])
        await asyncio.wait_for(self.admission_socket.send_multipart(parts), timeout=120)

    async def ingest_wire_event(self, ws, session, data, message_bytes):
        check(
            message_bytes <= self.max_message_bytes,
            "source_message_too_large",
            "The message exceeds this session's streaming size limit.",
        )
        kind = data.get("kind")
        if kind == "heartbeat":
            return False
        require(data.get("type") == "event", "Please send a valid streaming event.")
        event_id, payload = data.get("event_id"), data.get("payload")
        if kind in {"source.prepare", "source.attach"}:
            require(
                type(event_id) is int and event_id > 0,
                "Please provide a valid control sequence.",
            )
            await self._startup_control(ws, session, kind, payload)
            session.mark_client_activity()
            return True
        if self.startup_enabled:
            require(
                self.startup_phase == "attached", "Please wait for source attachment."
            )
        if kind == "source.append":
            if (
                self.startup_protocol == PREPARE_PROTOCOL
                and self.ledger.accepted_frames == 0
            ):
                frame_bytes = self.config["width"] * self.config["height"] * 3
                require(
                    isinstance(payload, dict)
                    and isinstance(payload.get("data"), bytes)
                    and rgb_digest(payload["data"][:frame_bytes])
                    == self.source_preview_sha256,
                    "The initial source frame does not match its preparation.",
                )
            if self._source_switch_end is not None:
                require(
                    isinstance(payload, dict)
                    and type(payload.get("source_frame_start")) is int
                    and type(payload.get("num_frames")) is int
                    and payload["source_frame_start"] + payload["num_frames"]
                    <= self._source_switch_end,
                    "Source input must wait for the condition switch to finish.",
                )
            received_ns = time.time_ns()
            record = self.ledger.accept(
                payload, event_id=event_id, message_bytes=message_bytes
            )
            if self.first_source_received_unix_ns is None:
                self.first_source_received_unix_ns = received_ns
                if not self.startup_enabled:
                    # Observe S0 without encoding it or creating a second transport.
                    # The worker reads it from the original accepted append bytes.
                    frame_bytes = self.config["width"] * self.config["height"] * 3
                    from ..conditioning import get_api_resources

                    resources = get_api_resources(session)
                    if self._s0_reservation is None:
                        self._s0_reservation = resources.ledger.reserve(
                            resources.tenant, ram=frame_bytes, inflight=frame_bytes
                        )
                    self.source_preview_bytes = payload["data"][:frame_bytes]
                    self.source_preview_sha256 = hashlib.sha256(
                        payload["data"][:frame_bytes]
                    ).hexdigest()
            # Only expose work after accepted is on the same serialized writer.
            acceptance = record.message()
            if payload["source_frame_start"] == 0:
                frame_bytes = self.config["width"] * self.config["height"] * 3
                acceptance.update(
                    self.input_receipt(
                        session,
                        "source.first",
                        dict(
                            source_frame=payload["data"][:frame_bytes],
                            source_frame_index=0,
                            width=self.config["width"],
                            height=self.config["height"],
                        ),
                        event_id=event_id,
                    )
                )
            await ws.send_bytes(msgspec.msgpack.encode(acceptance))
            from ..conditioning import take_message_reservation

            reservation = take_message_reservation(session)
            if reservation is not None:
                charges = dict(reservation.charges)
                reduced = {key: 0 for key in charges}
                reduced.update(ram=2 * message_bytes, inflight=2 * message_bytes)
                reservation.reduce(**reduced)
                self._source_reservations[payload["client_seq"]] = reservation
            self.pending.append(
                dict(payload=payload, event_id=event_id, message_bytes=message_bytes)
            )
            self.pending_frames += payload["num_frames"]
            self.available.set()
            # The first dispatch creates the admission receiver. Publish accepted
            # work before awaiting its connection; the producer still validates
            # admission identity before encoding, in either arrival order.
            await self._admit_source("append", payload)
        elif kind == "source.end":
            require(
                isinstance(payload, dict), "Please provide the final source position."
            )
            self.ledger.finish(payload.get("source_frame_end"), event_id=event_id)
            await self._admit_source(
                "finish", {"source_frame_end": self.ledger.accepted_frames}
            )
            self.end = dict(frame_end=self.ledger.accepted_frames, event_id=event_id)
            self.available.set()
        elif kind == "prompt.update" and self.ordered_prompts:
            await self._ingest_ordered_prompt_event(ws, session, event_id, payload)
        elif kind == "prompt" and not self.ordered_prompts:
            await self._ingest_prompt_event(ws, session, data, event_id, payload)
        elif kind in {"sink.off", "sink.refresh"}:
            require(self.sink_updates, "This deployment does not support sink updates.")
            require(payload == {}, "Please send an empty sink update payload.")
            self.ledger.validate_event_id(event_id)
            require(not self.ledger.ended, "The source stream has ended.")
            self.ledger.event_id = event_id
            self._pending_sink_command = dict(
                operation=kind.split(".")[1], event_id=event_id
            )
            self.available.set()
            await ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="control_ack",
                        kind=kind,
                        event_id=event_id,
                        stage="accepted",
                    )
                )
            )
        else:
            raise ValueError(
                "This session supports source frames, prompt and sink updates only."
            )
        session.mark_client_activity()
        return True

    async def _startup_control(self, ws, session, kind, payload):
        require(self.startup_enabled, "Early preparation was not negotiated.")
        require(isinstance(payload, dict), "Please provide a startup command.")
        identity = preparation_identity(payload.get("identity"))
        if self.startup_identity is not None:
            require(
                identity == self.startup_identity, "The startup identity has changed."
            )
        if kind == "source.prepare":
            native_preview = self.startup_protocol == V2_PROTOCOL
            fields = (
                {"identity", "rgb", "width", "height"}
                if native_preview
                else {"identity", "rgb"}
            )
            require(set(payload) == fields, "Invalid preparation fields.")
            rgb = payload["rgb"]
            width = payload["width"] if native_preview else self.config["width"]
            height = payload["height"] if native_preview else self.config["height"]
            try:
                validate_preview_rgb(rgb, width, height)
            except ValueError as error:
                raise ValueError("Please provide a valid starting image.") from error
            fingerprint = condition_fingerprint(
                rgb=rgb,
                prompt=self.initial_prompt,
                references=[
                    {k: r[k] for k in ("id", "format", "payload")}
                    for r in self.reference_items
                ],
                seed=session.request.seed,
                source_stream=session.request.source_stream,
                profile_sha256=self.preparation_profile,
                protocol=self.startup_protocol,
                preview_width=width,
                preview_height=height,
            )
            require(
                fingerprint == identity["condition_fingerprint"],
                "The preparation inputs have changed.",
            )
            if self.startup_phase != "waiting":
                require(
                    rgb_digest(rgb) == self.source_preview_sha256,
                    "The initial source frame has changed.",
                )
                receipt = (
                    dict(type="source.prepared", identity=identity, level="condition")
                    if self.worker_initialized
                    else dict(type="source.prepare.accepted", identity=identity)
                )
                await ws.send_bytes(msgspec.msgpack.encode(receipt))
                return
            from ..conditioning import get_api_resources

            resources = get_api_resources(session)
            self._s0_reservation = resources.ledger.reserve(
                resources.tenant, ram=len(rgb), inflight=len(rgb)
            )
            self.source_preview_bytes, self.source_preview_sha256 = rgb, rgb_digest(rgb)
            self.startup_identity = identity
            self.startup_phase = "preparing"
            self.startup_command = dict(op="prepare", identity=identity, rgb=rgb)
            if native_preview:
                # Internal worker boundary: legacy omitted versions remain v1.
                self.startup_command.update(
                    protocol=self.startup_protocol, width=width, height=height
                )
            await ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(type="source.prepare.accepted", identity=identity)
                )
            )
        else:
            require(set(payload) == {"identity"}, "Invalid attachment fields.")
            require(
                self.startup_phase in {"prepared", "attaching", "attached"},
                "The model is not prepared.",
            )
            if self.startup_phase == "attached":
                await ws.send_bytes(
                    msgspec.msgpack.encode(
                        dict(type="source.attached", identity=identity)
                    )
                )
                return
            if self.startup_phase == "prepared":
                self.startup_command = dict(op="attach", identity=identity)
                if self.startup_protocol == V2_PROTOCOL:
                    self.startup_command["protocol"] = self.startup_protocol
                self.startup_phase = "attaching"
        self.available.set()

    def is_control_work(self, session):
        return self.startup_command is not None

    async def _ingest_prompt_event(self, ws, session, data, event_id, payload):
        """Supersede pending work, prepare one candidate, then reject or ACK."""
        from ..conditioning import validate_remote_request

        validate_remote_request(session.request, get_global_server_args())
        self.ledger.validate_event_id(event_id)
        require(
            isinstance(payload, str) and payload.strip(),
            "Please provide a non-empty prompt.",
        )
        self.ledger.event_id = event_id
        self._pending_prompt_command = None
        if self._pending_reference_upload is not None:
            self._pending_reference_upload.release(session)
            self._pending_reference_upload = None
        command = dict(prompt=payload, event_id=event_id)
        try:
            from ..conditioning import get_api_resources

            args = get_global_server_args()
            tokens = await get_api_resources(session, args).cpu.run(
                count_prompt_tokens, args.model_path, payload
            )
            references = data.get("references")
            reference_count = (
                len(references)
                if isinstance(references, list)
                else self._budget_reference_count
            )
            self.input_budget.validate(tokens, reference_count)
            if "references" in data:
                require(
                    self.reference_updates,
                    "This session does not support reference updates.",
                )
                self._pending_reference_upload = await prepare_reference_upload(
                    session,
                    data["references"],
                    max_references=self.max_references,
                    max_decoded_bytes=self.reference_max_decoded_bytes,
                )
                command["references"] = self._pending_reference_upload.items
        except (
            ValueError,
            OSError,
            SyntaxError,
            Image.DecompressionBombError,
            ConditionError,
        ) as exc:
            # Still dispatch the rejection: a newer event supersedes any
            # older candidate already waiting at the worker boundary.
            command["rejected"] = True
            await ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="error",
                        code=exc.code
                        if isinstance(exc, ConditionError)
                        else "CONDITION_INVALID",
                        fatal=False,
                        event_id=event_id,
                        content=(
                            exc.public_message
                            if isinstance(exc, ConditionError)
                            else "This session does not support reference updates."
                            if not self.reference_updates
                            else "The prompt update could not be prepared. Please check the reference images and try again."
                        ),
                    )
                )
            )
        self._pending_prompt_command = command
        if self.prompt_prefetch is not None:
            self.prompt_prefetch.submit(command)
        self.available.set()
        if not command.get("rejected"):
            await ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="control_ack",
                        stage="worker",
                        kind="prompt",
                        event_id=event_id,
                        **self.input_receipt(
                            session, "prompt.update", command, event_id=event_id
                        ),
                    )
                )
            )

    async def _ingest_ordered_prompt_event(self, ws, session, event_id, payload):
        from ..conditioning import validate_remote_request

        validate_remote_request(session.request, get_global_server_args())
        self.ledger.validate_event_id(event_id)
        self.ledger.event_id = event_id
        upload = None
        code = "EFFECTIVE_BOUNDARY_UNAVAILABLE"
        try:
            boundary = self._reserve_prompt(payload)
            command = dict(
                prompt=payload["text"],
                event_id=event_id,
                min_source_frame=boundary,
                prompt_version=payload["version"],
            )
            if self.condition_switch_protocol:
                command["source_boundary"] = boundary
            if "references" in payload:
                code = "CONDITION_INVALID"
                require(
                    self.reference_updates,
                    "This session does not support reference updates.",
                )
                upload = await prepare_reference_upload(
                    session,
                    payload["references"],
                    max_references=self.max_references,
                    max_decoded_bytes=self.reference_max_decoded_bytes,
                )
                command["references"] = upload.items
        except (
            ValueError,
            OSError,
            SyntaxError,
            Image.DecompressionBombError,
            ConditionError,
        ):
            if upload is not None:
                upload.release(session)
            await ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="control_ack",
                        kind="prompt.update",
                        event_id=event_id,
                        state="rejected",
                        code=code,
                    )
                )
            )
            return
        self._pending_reference_upload = upload
        self._pending_prompt_command = command
        self.prompt_inflight, self.prompt_boundary = event_id, boundary
        if self.condition_switch_protocol:
            self._source_switch_end = boundary
        self.available.set()
        await ws.send_bytes(
            msgspec.msgpack.encode(
                dict(
                    type="control_ack",
                    kind="prompt.update",
                    event_id=event_id,
                    state="accepted",
                    source_frame_index=boundary,
                    prompt_version=payload["version"],
                    **self.input_receipt(
                        session,
                        "prompt.update",
                        dict(command, version=payload["version"]),
                        event_id=event_id,
                    ),
                )
            )
        )

    def has_control_work(self):
        return self._condition_control_ready() or (
            self._pending_sink_command is not None
            and self.dispatched_source_frames > 0
            and not self.end_dispatched
        )

    def _condition_control_ready(self):
        command = self._pending_prompt_command
        if command is None or "source_boundary" not in command:
            return False
        boundary = command["source_boundary"]
        return (
            self.end is None
            and not self.more_work
            and not self.pending
            and self.ledger.accepted_frames == boundary
            and self.dispatched_source_frames == boundary
            and self.committed_source_frames == boundary
        )

    def _reserve_prompt(self, payload):
        """Reserve beyond all admitted work, including Encoder/joint lookahead.

        RGB prefetch does not bind conditions. The accepted watermark is a
        conservative upper bound on any DiT work already submitted by this
        adapter. Native model blocks, rather than transport batches, define the
        next admissible boundary. Room holds the bounded durable prompt queue;
        this worker accepts one handoff until a nonempty output proves its use.
        """
        required = {"text", "version", "min_source_frame", "policy"}
        if (
            not isinstance(payload, dict)
            or not required <= set(payload)
            or set(payload) - required - {"references", "source_policy"}
        ):
            raise ValueError("Invalid prompt control")
        from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
            prompt_text,
        )

        prompt_text(payload["text"])
        target, version = payload["min_source_frame"], payload["version"]
        if (
            self.prompt_inflight is not None
            or type(target) is not int
            or target < 0
            or type(version) is not int
            or version <= self.public_prompt_version
            or payload["policy"] not in {"strict", "at_or_after"}
        ):
            raise ValueError("Invalid prompt boundary")
        watermark = max(target, self.ledger.accepted_frames, 17)
        if getattr(self, "condition_switch_protocol", None):
            if (
                payload.get("source_policy") != "skip_during_switch"
                or payload["policy"] != "at_or_after"
            ):
                raise ValueError("The condition switch requires a complete source clip")
            return ((watermark + self.clip - 1) // self.clip) * self.clip
        if "source_policy" in payload:
            raise ValueError("This session does not support source skipping")
        chunk_id = self.joint_chunk_id
        # Pixel boundaries use the same clip17/latent5 geometry as SourceBridge.
        while True:
            latent = self.joint_plan.chunk(chunk_id).start
            clips, offset = divmod(latent, 5)
            start = clips * 17 + (0 if offset == 0 else 1 + (offset - 1) * 4)
            if start >= watermark:
                break
            chunk_id += 1
        if payload["policy"] == "strict" and start != target:
            raise ValueError("The exact boundary is unavailable")
        return start

    async def wait_for_next_chunk(self, session):
        if self.startup_enabled and self.startup_phase != "attached":
            while self.startup_command is None:
                if self.ledger.closed:
                    raise asyncio.CancelledError("Source session has closed")
                self.available.clear()
                remaining = self.startup_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Source preparation deadline exceeded")
                await asyncio.wait_for(self.available.wait(), remaining)
            return
        # The first accepted frame supplies Qwen's source presentation;
        # subsequent codec readiness is independent of transport packet size.
        minimum = 1
        while (
            not self.more_work
            and not self._condition_control_ready()
            and self.pending_frames < minimum
            and (self.end is None or self.end_dispatched)
        ):
            if self.ledger.closed:
                raise asyncio.CancelledError("Source session has closed")
            self.available.clear()
            await self.available.wait()

        if self.more_work or self._condition_control_ready():
            # This dispatch does not collect source. Do not repeat the previous
            # dispatch's wait in the timing receipt for a resumed output block.
            self.joint_wait = dict(
                budget_ms=0.0,
                target_frames=0,
                policy="continuation",
                actual_ms=0.0,
                reason="continuation" if self.more_work else "condition_ready",
            )
            return
        target, automatic_budget = self._joint_wait_target()
        configured_budget = self.profile.get("joint_wait_ms", "auto")
        budget = automatic_budget if configured_budget == "auto" else configured_budget
        started = time.monotonic()
        deadline = started + budget / 1000
        reason = "disabled"
        try:
            while self.profile.get("joint_forward", "cross_chunk") == "cross_chunk":
                if self.ledger.closed:
                    reason = "cancelled"
                    raise asyncio.CancelledError("Source session has closed")
                if (
                    self._pending_prompt_command is not None
                    or getattr(self, "_pending_sink_command", None) is not None
                ):
                    reason = "prompt"
                    break
                if self.end is not None:
                    reason = "eos"
                    break
                if self.pending_frames >= target:
                    reason = "ready"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "timeout" if budget else "disabled"
                    break
                self.available.clear()
                try:
                    await asyncio.wait_for(self.available.wait(), remaining)
                except asyncio.TimeoutError:
                    reason = "timeout"
                    break
        except asyncio.CancelledError:
            reason = "cancelled"
            raise
        finally:
            self.joint_wait = dict(
                budget_ms=budget,
                target_frames=target,
                policy="auto" if configured_budget == "auto" else "fixed",
                actual_ms=(time.monotonic() - started) * 1000,
                reason=reason,
            )

    def _joint_wait_target(self):
        if self.source_encoder["mode"] != "progressive":
            # Preserve first-pixel presentation while condition preparation runs.
            target = 1 if not self.dispatched_source_frames else self.clip
            return target, min(
                1000.0,
                max(0, target - self.pending_frames) * 1000.0 / self.config["fps"],
            )

        # The progressive clip17 codec emits at pixel watermarks 1/9/17,
        # yielding latent ends 1/3/5. AR chunks are a separate checkpoint plan.
        def dependency(end):
            clips, last = divmod(end - 1, 5)
            return clips * 17 + (1 if last == 0 else 9 if last < 3 else 17)

        while (
            dependency(self.joint_plan.chunk(self.joint_chunk_id).end)
            <= self.dispatched_source_frames
        ):
            self.joint_chunk_id += 1
        # The first dispatch never waits for the next block. Progressive
        # remote output continues to wait only for its current codec dependency.
        single_output = self.source_work_quantum == 1 or self.remote_progressive
        lookahead = 0 if single_output or not self.dispatched_source_frames else 1
        required = self.joint_plan.chunk(self.joint_chunk_id + lookahead)
        target = min(
            self.clip, dependency(required.end) - self.dispatched_source_frames
        )
        # Wait only for the missing source at its declared frame rate, with a
        # hard deadline. Interrupts and the dispatch/credit limits still win.
        missing = max(0, target - self.pending_frames)
        return target, min(1000.0, missing * 1000.0 / self.config["fps"])

    async def close_conditioning(self, session):
        if not hasattr(self, "_source_reservations"):
            return  # A connected-worker check can reject before on_init owns resources.
        if self._pending_reference_upload is not None:
            self._pending_reference_upload.release(session)
            self._pending_reference_upload = None
        for upload in self._inflight_reference_uploads:
            upload.release(session)
        self._inflight_reference_uploads.clear()
        for reservation in (
            *self._source_reservations.values(),
            *self._inflight_reservations,
        ):
            reservation.release()
        self._source_reservations.clear()
        self._inflight_reservations.clear()
        if self._s0_reservation is not None:
            self._s0_reservation.release()
            self._s0_reservation = None

    def get_chunk_size(self, session, server_args, chunk):
        return self.clip

    def sample_chunk_inputs(self, session, server_args, chunk, chunk_size):
        if self._pending_prompt_command is not None and (
            "source_boundary" not in self._pending_prompt_command
            or self._condition_control_ready()
        ):
            command = self._pending_prompt_command
            session.request.prompt = command["prompt"]
            self.public_prompt_version = command.get(
                "prompt_version", self.public_prompt_version
            )
            self.prompt_event = command["event_id"]
            self._selected_prompt_command = command
            self._pending_prompt_command = None
            if self._pending_reference_upload is not None:
                self._inflight_reference_uploads.append(self._pending_reference_upload)
                self._pending_reference_upload = None
        return RealtimeChunkInputs(prompt=session.request.prompt)

    def get_realtime_event_id(self, session):
        return self.prompt_event

    def build_sampling_params(
        self, session, server_args, chunk, chunk_inputs, chunk_size
    ):
        params = build_realtime_sampling_params(
            chunk.request_id,
            request=session.request,
            chunk_inputs=chunk_inputs,
            num_frames=self.clip,
            num_inference_steps=len(self.sampling_schedule["denoising_step_list"]),
            chunk_size=chunk_size,
        )
        # Only locally validated media enter the internal canonical request.
        # The wire API still rejects arbitrary zing_request/source objects.
        # Worker inputs persist after the first dispatch. Do not pickle the
        # initial encoded images again on every chunk; replacements travel only
        # in zing_prompt_update and the worker owns the active media snapshot.
        initial_media = (
            self.reference_items
            if session.generate_chunk_cnt == 0 and not self.worker_initialized
            else []
        )
        content = list(initial_media)
        content.append(dict(type="text", text=chunk_inputs.prompt))
        params.zing_request = dict(
            messages=[
                dict(role="user", content=content),
                dict(
                    role="target",
                    content=[
                        dict(
                            type="video",
                            output=dict(
                                frames=self.clip,
                                width=self.source_geometry.target_width,
                                height=self.source_geometry.target_height,
                            ),
                        )
                    ],
                    conditions=[
                        dict(
                            content_id=item["id"],
                            role="reference",
                            latent_modalities=["visual"],
                        )
                        for item in self.reference_items
                        if initial_media
                    ],
                ),
            ]
        )
        params.zing_joint_forward = self.profile.get("joint_forward", "cross_chunk")
        params.zing_window_eviction_timing = self.profile.get(
            "window_eviction_timing", "after_step0"
        )
        params.zing_solver = self.sampling_schedule["solver"]
        params.zing_denoising_step_list = list(
            self.sampling_schedule["denoising_step_list"]
        )
        params.zing_realtime_decode_mode = "stable"
        params.zing_window_chunks = self.source_attention_window["window_chunks"]
        params.zing_sink_chunks = self.source_attention_window["sink_chunks"]
        params.zing_rope_position_mode, params.zing_rope_max_frame_gap = (
            "block_relative",
            12,
        )
        return params

    def prepare_prefill_request(
        self,
        session,
        server_args,
        source_frame,
        *,
        source_frame_size=None,
        persist=False,
    ):
        geometry = (
            source_frame_size
            if source_frame_size is not None
            else {"width": self.config["width"], "height": self.config["height"]}
        )
        require(
            isinstance(geometry, dict) and set(geometry) == {"width", "height"},
            "Please provide a valid prefill frame size.",
        )
        try:
            validate_preview_rgb(source_frame, geometry["width"], geometry["height"])
        except ValueError as exc:
            raise ValueError(
                "Please provide exactly one valid RGB24 prefill frame."
            ) from exc
        batch = super().prepare_next_request(session, server_args, session.new_chunk())
        batch.extra["zing_prefill"] = dict(
            source_frame=source_frame, source_frame_size=dict(geometry), persist=persist
        )
        batch.extra["zing_source_command"] = dict(
            **self.cache_options,
            config=self.config,
            source_geometry=asdict(self.source_geometry),
            sampling_schedule=self.sampling_schedule,
            work_quantum=self.source_work_quantum,
            decode_quantum=self.source_decode_quantum,
            remote_progressive=self.remote_progressive,
            defer_final=self.source_defer_final,
            prompt_joint=self.source_prompt_joint,
        )
        return batch

    def _take_source_input(self):
        appends, count = [], 0
        while (
            (not self.more_work or self.source_append_overlap)
            and self.pending
            and count < self.clip
        ):
            size = self.pending[0]["payload"]["num_frames"]
            capacity = 2 * self.clip
            if self.source_append_overlap and self.append_capacity_frames is not None:
                capacity = min(capacity, self.append_capacity_frames)
            if count + size > capacity:
                require(
                    count > 0 or self.more_work,
                    "Source input cannot progress within the configured capacity.",
                )
                break
            item = self.pending.popleft()
            reservation = self._source_reservations.pop(
                item["payload"]["client_seq"], None
            )
            if reservation is not None:
                self._inflight_reservations.append(reservation)
            count += item["payload"]["num_frames"]
            appends.append(item)
        self.pending_frames -= count
        self.dispatched_source_frames += count
        end = self.end if not self.pending and not self.end_dispatched else None
        if self.end_dispatched and not self.more_work:
            raise ValueError("Source end was already dispatched")
        self.end_dispatched = self.end_dispatched or end is not None
        return appends, end

    def prepare_next_request(self, session, server_args, chunk):
        batch = super().prepare_next_request(session, server_args, chunk)
        if self.startup_enabled:
            batch.extra["realtime_preparation_id"] = self.startup_identity[
                "preparation_id"
            ]
        if self.ordered_prompts:
            batch.realtime_prompt_version = self.public_prompt_version
        if self.startup_command is not None:
            require(chunk.control_only, "Startup work must use a control dispatch.")
            batch.extra["zing_source_command"] = dict(
                config=self.config,
                source_geometry=asdict(self.source_geometry),
                sampling_schedule=self.sampling_schedule,
                work_quantum=self.source_work_quantum,
                decode_quantum=self.source_decode_quantum,
                remote_progressive=self.remote_progressive,
                defer_final=self.source_defer_final,
                prompt_joint=self.source_prompt_joint,
                appends=[],
                end=None,
                encoder=self.source_encoder,
                decoder=self.profile.get("source_decoder"),
                startup=self.startup_command,
                source_preview=True,
            )
            self.startup_command = None
            return batch
        if getattr(self, "_pending_sink_command", None) is not None:
            batch.extra["zing_sink_update"] = self._pending_sink_command
            self._pending_sink_command = None
        if self._selected_prompt_command is not None:
            batch.extra["zing_prompt_update"] = self._selected_prompt_command
            self._selected_prompt_command = None
        update = batch.extra.get("zing_prompt_update", {})
        if "references" in update and not update.get("rejected"):
            count = len(update["references"])
            # Until output confirms this update, the prior snapshot may remain
            # active after a failed or superseded update. Keep the larger count.
            self._budget_reference_count = max(self._budget_reference_count, count)
            self._budget_reference_update = (update["event_id"], count)
        if batch.block_idx == 0 and ("references" in update or update.get("rejected")):
            # Initialize the original epoch before applying a queued update;
            # an early invalid reference must not change its prompt or identity.
            batch.sampling_params.prompt = self.initial_prompt
            batch.sampling_params.zing_request["messages"][0]["content"][-1]["text"] = (
                self.initial_prompt
            )
            batch.realtime_event_id = None
        appends, end = self._take_source_input()
        batch.extra["zing_reference_ids"] = self.reference_ids
        batch.extra["zing_source_command"] = dict(
            **self.cache_options,
            config=self.config,
            source_geometry=asdict(self.source_geometry),
            conditioning_prefetch_endpoint=self.conditioning_prefetch_endpoint,
            conditioning_prefetch_max_bytes=self.max_message_bytes,
            sampling_schedule=self.sampling_schedule,
            async_output=self.profile.get("async_source_output", False),
            work_quantum=self.source_work_quantum,
            decode_quantum=self.source_decode_quantum,
            remote_progressive=self.remote_progressive,
            defer_final=self.source_defer_final,
            prompt_joint=self.source_prompt_joint,
            appends=appends,
            end=end,
            encoder=self.source_encoder,
            decoder=self.profile.get("source_decoder"),
            first_source_received_unix_ns=self.first_source_received_unix_ns,
            source_preview=True,
            source_preview_sha256=self.source_preview_sha256,
            source_wait_host_ms=(
                (self.first_source_received_unix_ns - self.init_started_ns) / 1e6
                if self.first_source_received_unix_ns is not None
                else None
            ),
            joint_wait=self.joint_wait,
            dispatch_unix_ns=time.time_ns(),
            measure_cuda=self.profile.get("measure_source_cuda", False),
            prompt_boundary=self.prompt_boundary,
        )
        return batch

    def on_chunk_complete(self, session, result):
        super().on_chunk_complete(session, result)
        receipt = (result.usage or {}).get("zing_startup_ready")
        if receipt is not None:
            require(
                receipt.get("identity") == self.startup_identity,
                "Invalid startup receipt.",
            )
            expected = (
                "source.prepared"
                if self.startup_phase == "preparing"
                else "source.attached"
            )
            require(receipt.get("type") == expected, "Unexpected startup receipt.")
            self.startup_phase = (
                "prepared" if expected == "source.prepared" else "attached"
            )
            self.worker_initialized = True
            self.available.set()
            return

        ready_condition = (result.usage or {}).get("zing_condition_ready")
        if getattr(self, "_budget_reference_update", None) is not None:
            event_id, count = self._budget_reference_update
            if (
                ready_condition is not None
                and ready_condition.get("event_id") == event_id
            ) or any(
                block["event_id"] == event_id
                for block in (result.usage or {}).get("zing_realtime_blocks", [])
            ):
                self._budget_reference_count = count
                self._budget_reference_update = None
        for upload in self._inflight_reference_uploads:
            upload.release(session)
        self._inflight_reference_uploads.clear()
        committed = (result.usage or {}).get(
            "zing_source_committed_end", self.committed_source_frames
        )
        require(
            type(committed) is int
            and self.committed_source_frames
            <= committed
            <= self.ledger.accepted_frames,
            "Invalid committed source boundary.",
        )
        self.committed_source_frames = committed
        if ready_condition is not None:
            self._source_switch_end = None
        more_work = (result.usage or {}).get("zing_source_more_work", False)
        require(type(more_work) is bool, "Invalid source continuation state.")
        self.more_work = bool(
            (self.source_work_quantum or getattr(self, "remote_progressive", False))
            and more_work
        )
        if self.ordered_prompts:
            usage = result.usage or {}
            if (
                ready_condition is not None
                and ready_condition.get("event_id") == self.prompt_inflight
            ) or any(
                block.get("event_id", block.get("prompt_event_id"))
                == self.prompt_inflight
                and block.get("num_frames", 0) > 0
                for block in usage.get("zing_realtime_blocks", [])
            ):
                self.prompt_inflight = None
            notice = getattr(result, "conditioning_notice", None)
            if notice and notice.get("event_id") == self.prompt_inflight:
                self.prompt_inflight = None
                self._source_switch_end = None
        if self.source_append_overlap:
            capacity = (result.usage or {}).get("zing_source_append_capacity_frames")
            require(
                type(capacity) is int and capacity >= 0,
                "Invalid source append capacity.",
            )
            self.append_capacity_frames = capacity
        if self.source_decoder_overlap:
            consumed = (result.usage or {}).get("zing_source_consumed_end")
            require(
                type(consumed) is int
                and self.pending_source_consumed_end
                <= consumed
                <= self.dispatched_source_frames,
                "Invalid source ownership watermark.",
            )
            # The worker watermark may fall inside a transport packet. The
            # ledger grants credit only for whole packets covered by it.
            self.pending_source_consumed_end = consumed
        for reservation in self._inflight_reservations:
            reservation.release()
        self._inflight_reservations.clear()
        if (result.usage or {}).get("zing_source_complete"):
            session.request.max_chunks = session.generate_chunk_cnt

    async def send_output(self, ws, session, result, batch):
        usage = result.usage or {}
        if usage.get("zing_startup_ready") is not None:
            from ..realtime_output_adapter import empty_frame_send_stats

            await ws.send_bytes(msgspec.msgpack.encode(usage["zing_startup_ready"]))
            return empty_frame_send_stats()
        blocks = usage.get("zing_realtime_blocks", [])
        source_blocks = []
        for metadata in blocks:
            provenance = self.ledger.provenance_for(
                PixelRange(
                    metadata["output_frame_start"],
                    metadata["output_frame_start"] + metadata["num_frames"],
                ),
                metadata["source_dependency_end"],
            )
            require(
                provenance.metadata()
                == {key: metadata[key] for key in provenance.metadata()},
                "The generated video has invalid source alignment.",
            )
            source_blocks.append({**metadata, "provenance": provenance})
        sub_batch = copy(batch)
        sub_batch.extra = {**batch.extra, "zing_source_output_blocks": source_blocks}
        total = await super().send_output(ws, session, result, sub_batch)
        from sglang.multimodal_gen.runtime.realtime.whitebox.cache_receipt import (
            public_cache_receipt,
        )

        cache = public_cache_receipt(usage.get("zing_block0_cache"))
        if cache is not None:
            await ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="model.cache",
                        model_session_id=session.id,
                        generation_id=session.generation_id,
                        prompt_version=cache["prompt_version"],
                        usage={"zing_block0_cache": cache},
                    )
                )
            )
        for block in source_blocks:
            self.ledger.delivered(block["provenance"].pixels)
        if self.source_decoder_overlap:
            await self.flush_source_credit(ws)
        else:
            consumed = self.ledger.consume(usage["zing_source_consumed_end"])
            if consumed is not None:
                await ws.send_bytes(msgspec.msgpack.encode(consumed))
        if usage.get("zing_condition_ready") is not None:
            await ws.send_bytes(msgspec.msgpack.encode(usage["zing_condition_ready"]))
        if usage.get("zing_source_complete"):
            await ws.send_bytes(msgspec.msgpack.encode(self.ledger.completed()))
        return total

    async def flush_source_credit(self, ws):
        """Merge validated progress without a second sender or stale RGB watermark."""
        if not self.source_decoder_overlap:
            return
        # Both the output task and the input waiter use this entry point.
        # Re-read after acquiring the reentrant output group lock.
        async with ws.send_group():
            if self.pending_source_consumed_end > self.ledger.consumed_frames:
                consumed = self.ledger.consume(self.pending_source_consumed_end)
                if consumed is not None:
                    await ws.send_bytes(msgspec.msgpack.encode(consumed))

    def clear_state(self, session):
        if getattr(self, "prompt_prefetch", None) is not None:
            self.prompt_prefetch.close()
            self.prompt_prefetch = None
        if getattr(self, "admission_socket", None) is not None:
            self.admission_socket.close()
            self.admission_context.term()
            self.admission_socket = self.admission_context = None
        if hasattr(self, "ledger"):
            self.ledger.dispose()
            self.pending.clear()
            self.available.set()
            self._pending_prompt_command = self._selected_prompt_command = None
            self._pending_sink_command = None
            self.reference_items.clear()
            self.reference_ids.clear()
            self.source_preview_sha256 = None
            self.source_preview_bytes = None
            self.startup_command = self.startup_identity = None
