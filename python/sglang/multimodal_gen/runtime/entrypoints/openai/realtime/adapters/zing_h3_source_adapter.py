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
    def create_state(self):
        return None  # Initialized atomically after negotiation validation.

    async def on_init(self, session, request, *, prefill=False):
        self.init_started_ns = time.time_ns()
        self._source_reservations, self._inflight_reservations = {}, []
        self._s0_reservation = None
        self._pending_reference_upload = None
        self._inflight_reference_uploads = []
        self._pending_prompt_command = self._selected_prompt_command = None
        self.prompt_prefetch = None
        self.conditioning_prefetch_endpoint = None
        self.initial_prompt = request.prompt
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
        # Only the server deployment profile selects the window; clients cannot override 21/3.
        window = profile.get(
            "source_attention_window", {"window_chunks": 21, "sink_chunks": 3}
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
        if self.source_defer_final and self.source_work_quantum != 1:
            raise ValueError("Deferred source final currently requires work quantum 1")
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
            "prompt_reference_updates": self.reference_updates,
            "max_reference_bytes": reference_limits.reference_bytes,
            "max_reference_image_bytes": MAX_REFERENCE_IMAGE_BYTES,
            "max_reference_decoded_bytes": self.reference_max_decoded_bytes,
        }

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

    def session_ready_message(self, session):
        return self.ready

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
        if kind == "source.append":
            received_ns = time.time_ns()
            record = self.ledger.accept(
                payload, event_id=event_id, message_bytes=message_bytes
            )
            if self.first_source_received_unix_ns is None:
                self.first_source_received_unix_ns = received_ns
                # Observe S0 without encoding it or creating a second transport.
                # The worker reads it from the original accepted append bytes.
                frame_bytes = self.config["width"] * self.config["height"] * 3
                from ..conditioning import get_api_resources

                resources = get_api_resources(session)
                self._s0_reservation = resources.ledger.reserve(
                    resources.tenant, ram=frame_bytes, inflight=frame_bytes
                )
                self.source_preview_bytes = payload["data"][:frame_bytes]
                self.source_preview_sha256 = hashlib.sha256(
                    payload["data"][:frame_bytes]
                ).hexdigest()
            # Only expose work after accepted is on the same serialized writer.
            await ws.send_bytes(msgspec.msgpack.encode(record.message()))
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
        elif kind == "prompt":
            await self._ingest_prompt_event(ws, session, data, event_id, payload)
        else:
            raise ValueError(
                "This session supports source frames and prompt updates only."
            )
        session.mark_client_activity()
        return True

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
        if "references" in data:
            try:
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
                    )
                )
            )

    async def wait_for_next_chunk(self, session):
        # The first accepted frame supplies Qwen's source presentation;
        # subsequent codec readiness is independent of transport packet size.
        minimum = 1
        while (
            not self.more_work
            and self.pending_frames < minimum
            and (self.end is None or self.end_dispatched)
        ):
            if self.ledger.closed:
                raise asyncio.CancelledError("Source session has closed")
            self.available.clear()
            await self.available.wait()

        if self.more_work:
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
                if self._pending_prompt_command is not None:
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
        # Start as soon as the first chunk is ready. Only later dispatches
        # wait for the following chunk to enable cross-chunk joint execution.
        target_chunk = self.joint_plan.chunk(
            self.joint_chunk_id + (1 if self.dispatched_source_frames else 0)
        )
        target = min(
            self.clip, dependency(target_chunk.end) - self.dispatched_source_frames
        )
        # Wait only for the missing source at its declared frame rate, with a
        # hard deadline. Interrupts and the dispatch/credit limits still win.
        missing = max(0, target - self.pending_frames)
        return target, min(1000.0, missing * 1000.0 / self.config["fps"])

    async def close_conditioning(self, session):
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
        if self._pending_prompt_command is not None:
            command = self._pending_prompt_command
            session.request.prompt = command["prompt"]
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
        initial_media = self.reference_items if session.generate_chunk_cnt == 0 else []
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

    def prepare_prefill_request(self, session, server_args, source_frame):
        require(
            isinstance(source_frame, bytes)
            and len(source_frame) == self.config["width"] * self.config["height"] * 3,
            "Please provide exactly one RGB24 source frame for prefill.",
        )
        batch = super().prepare_next_request(session, server_args, session.new_chunk())
        batch.extra["zing_prefill"] = dict(source_frame=source_frame)
        batch.extra["zing_source_command"] = dict(
            **self.cache_options,
            config=self.config,
            sampling_schedule=self.sampling_schedule,
            work_quantum=self.source_work_quantum,
            defer_final=self.source_defer_final,
            prompt_joint=self.source_prompt_joint,
        )
        return batch

    def prepare_next_request(self, session, server_args, chunk):
        batch = super().prepare_next_request(session, server_args, chunk)
        if self._selected_prompt_command is not None:
            batch.extra["zing_prompt_update"] = self._selected_prompt_command
            self._selected_prompt_command = None
        update = batch.extra.get("zing_prompt_update", {})
        if batch.block_idx == 0 and ("references" in update or update.get("rejected")):
            # Initialize the original epoch before applying a queued update;
            # an early invalid reference must not change its prompt or identity.
            batch.sampling_params.prompt = self.initial_prompt
            batch.sampling_params.zing_request["messages"][0]["content"][-1]["text"] = (
                self.initial_prompt
            )
            batch.realtime_event_id = None
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
        )
        return batch

    def on_chunk_complete(self, session, result):
        super().on_chunk_complete(session, result)
        for upload in self._inflight_reference_uploads:
            upload.release(session)
        self._inflight_reference_uploads.clear()
        more_work = (result.usage or {}).get("zing_source_more_work", False)
        require(type(more_work) is bool, "Invalid source continuation state.")
        self.more_work = bool(self.source_work_quantum and more_work)
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
        for block in source_blocks:
            self.ledger.delivered(block["provenance"].pixels)
        if self.source_decoder_overlap:
            await self.flush_source_credit(ws)
        else:
            consumed = self.ledger.consume(usage["zing_source_consumed_end"])
            if consumed is not None:
                await ws.send_bytes(msgspec.msgpack.encode(consumed))
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
            self.reference_items.clear()
            self.reference_ids.clear()
            self.source_preview_sha256 = None
            self.source_preview_bytes = None
