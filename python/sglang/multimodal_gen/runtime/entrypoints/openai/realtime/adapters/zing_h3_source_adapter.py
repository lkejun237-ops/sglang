# SPDX-License-Identifier: Apache-2.0
"""Version-one RGB source on the shared realtime scheduler and transport."""

import asyncio
import hashlib
import json
import math
import tempfile
import time
import uuid
from collections import deque
from copy import copy
from dataclasses import asdict
from pathlib import Path

import msgspec.msgpack
import zmq
import zmq.asyncio
from PIL import Image  # noqa: F401 - preserve the existing CPU validation seam

from sglang.multimodal_gen.configs.zing_sampling import (
    normalize_source_sampling_schedule,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    BaseRealtimeModelAdapter,
    RealtimeChunkInputs,
    build_realtime_sampling_params,
)
from sglang.multimodal_gen.runtime.realtime.source_transport import (
    PENDING_BATCHES,
    REFERENCE_BYTES,
    SourceTransportBudget,
    check,
    validate_source_dimensions,
)
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange, require
from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
    SourceLedger,
    SourceLimits,
    default_source_encoder,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args


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
            denoising_step_list=[1000, 750],
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

    async def on_init(self, session, request):
        self.init_started_ns = time.time_ns()
        self._source_reservations, self._inflight_reservations = {}, []
        self._s0_reservation = None
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
        width, height = map(int, request.size.split("x"))
        fps = int(runtime["h3_media_config"]["fps"])
        require(
            source == dict(format="rgb24", width=width, height=height, fps=fps)
            and request.fps in (None, fps),
            "Please provide matching RGB source dimensions and frame rate.",
        )
        validate_source_dimensions(width, height, args.pipeline_config)
        transport = SourceTransportBudget.from_args(args)
        transport.validate_sessions(
            max(
                getattr(args, "realtime_max_sessions", 8),
                getattr(args, "realtime_max_sessions_per_worker", 8),
            )
        )
        client_budget = getattr(request, "source_message_budget_bytes", None)
        check(
            client_budget is None
            or (
                type(client_budget) is int and client_budget == transport.message_bytes
            ),
            "source_transport_mismatch",
            "The client and service streaming size limits do not match.",
        )
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
            request.realtime_preview_max_width in (None, width),
            "This session requires the full output dimensions.",
        )
        references = getattr(request, "references", []) or []
        media = getattr(request, "user_media", None)
        require(
            media is None or not references,
            "Please use either user media or reference images.",
        )
        if media is None:
            require(
                isinstance(references, list) and len(references) <= 9,
                "Please provide at most nine reference images.",
            )
            media = [dict(ref, type="image", reference=True) for ref in references]
        require(
            isinstance(media, list) and len(media) <= 10,
            "Please provide at most ten media items.",
        )
        # The causal source contract supplies Video 1 automatically. Explicit
        # typed video media already defines presentation (including an independent
        # preview), so preserve it instead of adding or replacing it.
        self.auto_source_preview = not any(
            isinstance(item, dict) and item.get("type") == "video" for item in media
        )
        require(
            len(media) + int(self.auto_source_preview) <= 10,
            "Please provide at most ten media items including source preview.",
        )
        seen, reference_paths, media_bytes = set(), [], 0
        self.user_media, self.reference_items = [], []
        for i, item in enumerate(media):
            require(
                isinstance(item, dict)
                and not set(item)
                - {"type", "id", "format", "payload", "clip", "fps", "reference"}
                and item.get("type") in ("image", "video")
                and isinstance(item.get("id"), str)
                and item["id"]
                and item["id"] not in seen
                and item.get("format") in ("jpeg", "webp", "png")
                and isinstance(item.get("payload"), bytes)
                and type(item.get("reference", False)) is bool,
                "Please provide uniquely named image or video media with supported image bytes.",
            )
            video = item["type"] == "video"
            require(
                not video
                or (
                    isinstance(item.get("clip"), dict)
                    and set(item["clip"]) == {"start", "end"}
                    and item["clip"]["start"] == item["clip"]["end"]
                    and all(
                        type(value) in (int, float)
                        and math.isfinite(value)
                        and value >= 0
                        for value in item["clip"].values()
                    )
                    and type(item.get("fps")) in (int, float)
                    and 0 < item["fps"] <= 240
                    and not item.get("reference", False)
                ),
                "Please provide a single-frame video with its frame rate and matching clip times.",
            )
            require(
                video or not ({"clip", "fps"} & set(item)),
                "Image media cannot have a video timeline.",
            )
            require(
                not self.auto_source_preview or item["id"] != "<Video 1>",
                "Please use a different name for this image.",
            )
            seen.add(item["id"])
            media_bytes += len(item["payload"])
            require(
                media_bytes <= REFERENCE_BYTES, "The reference images are too large."
            )
            from ..conditioning import reserve_media_storage, verify_presentation_upload

            await verify_presentation_upload(session, item["payload"], item["format"])
            reserve_media_storage(session, len(item["payload"]))
            # Workers read reference media from the session's private directory.

            if session.input_temp_dir is None:
                session.input_temp_dir = tempfile.mkdtemp(
                    prefix="sglang_zing_source_", dir="/tmp"
                )
            path = Path(session.input_temp_dir) / f"media-{i}.{item['format']}"
            path.write_bytes(item["payload"])
            descriptor = dict(type=item["type"], id=item["id"], uri=str(path))
            if video:
                descriptor.update(clip=dict(item["clip"]), presentation_fps=item["fps"])
            self.user_media.append(descriptor)
            if item.get("reference", False):
                self.reference_items.append(descriptor)
                reference_paths.append(str(path))
        require(
            len(self.reference_items) <= 9,
            "Please provide at most nine reference images.",
        )
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

        self.source_work_quantum = source_work_quantum()
        self.more_work = False
        self.clip, self.reference_paths, self.profile = clip, reference_paths, profile
        self.reference_ids = [ref["id"] for ref in self.reference_items]
        self.source_encoder = profile.get("source_encoder", default_source_encoder())
        self._initialize_source_admission(session, limits)
        self.first_source_received_unix_ns = None
        self.pending_prompt = None
        self.prompt_event = None
        request.fps, request.num_inference_steps, request.guidance_scale = (
            fps,
            regular_steps,
            profile["guidance_scale"],
        )
        request.realtime_output_format = request.realtime_output_format or "raw"
        request.realtime_preview_max_width = width
        self.ready = {
            "type": "session.ready",
            "zing_realtime_api_version": 1,
            "session_id": session.id,
            "num_inference_steps": regular_steps,
            "sampling_schedule": self.sampling_schedule,
            "realtime_output_format": request.realtime_output_format,
            "source_stream": source,
            "max_message_bytes": transport.message_bytes,
            "max_source_batch_frames": limits.batch_frames,
            "max_source_message_bytes": limits.message_bytes,
            "max_pending_source_frames": limits.pending_frames,
            "max_pending_source_bytes": limits.pending_bytes,
            "max_references": 9,
            "max_reference_bytes": REFERENCE_BYTES,
            "user_media_formats": ["jpeg", "webp", "png"],
            "max_user_media": 10,
            "user_media_video_clip": {"start": 0, "end": 0},
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
                if self.auto_source_preview:
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
            from ..conditioning import validate_remote_request

            validate_remote_request(session.request, get_global_server_args())
            self.ledger.validate_event_id(event_id)
            require(
                isinstance(payload, str) and payload.strip(),
                "Please provide a non-empty prompt.",
            )
            self.ledger.event_id = event_id
            self.pending_prompt = payload, event_id
            self.available.set()
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
        else:
            raise ValueError(
                "This session supports source frames and prompt updates only."
            )
        session.mark_client_activity()
        return True

    async def wait_for_next_chunk(self, session):
        # Progressive readiness belongs to the codec, never a transport packet
        # size. Keep the original collection policy only for the explicit A/B
        # baseline selected in the deployment profile.
        minimum = (
            1
            if self.auto_source_preview or self.source_encoder["mode"] == "progressive"
            else self.clip
        )
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
                if self.pending_prompt is not None:
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
            target = (
                1
                if self.auto_source_preview and not self.dispatched_source_frames
                else self.clip
            )
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
        following = self.joint_plan.chunk(self.joint_chunk_id + 1)
        target = min(
            self.clip, dependency(following.end) - self.dispatched_source_frames
        )
        # Wait only for the missing source at its declared frame rate, with a
        # hard deadline. Interrupts and the dispatch/credit limits still win.
        missing = max(0, target - self.pending_frames)
        return target, min(1000.0, missing * 1000.0 / self.config["fps"])

    async def close_conditioning(self, session):
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
        if self.pending_prompt is not None:
            session.request.prompt, self.prompt_event = self.pending_prompt
            self.pending_prompt = None
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
        params.image_path = self.reference_paths
        # Only locally validated media enter the internal canonical request.
        # The wire API still rejects arbitrary zing_request/source objects.
        content = list(self.user_media)
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
                                width=self.config["width"],
                                height=self.config["height"],
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

    def prepare_next_request(self, session, server_args, chunk):
        batch = super().prepare_next_request(session, server_args, chunk)
        appends, count = [], 0
        while not self.more_work and self.pending and count < self.clip:
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
            config=self.config,
            sampling_schedule=self.sampling_schedule,
            async_output=self.profile.get("async_source_output", False),
            appends=appends,
            end=end,
            encoder=self.source_encoder,
            first_source_received_unix_ns=self.first_source_received_unix_ns,
            source_preview=self.auto_source_preview,
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
        more_work = (result.usage or {}).get("zing_source_more_work", False)
        require(type(more_work) is bool, "Invalid source continuation state.")
        self.more_work = bool(self.source_work_quantum and more_work)
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
        consumed = self.ledger.consume(usage["zing_source_consumed_end"])
        if consumed is not None:
            await ws.send_bytes(msgspec.msgpack.encode(consumed))
        if usage.get("zing_source_complete"):
            await ws.send_bytes(msgspec.msgpack.encode(self.ledger.completed()))
        return total

    def clear_state(self, session):
        if getattr(self, "admission_socket", None) is not None:
            self.admission_socket.close()
            self.admission_context.term()
            self.admission_socket = self.admission_context = None
        if hasattr(self, "ledger"):
            self.ledger.dispose()
            self.pending.clear()
            self.available.set()
            self.pending_prompt = None
            self.reference_paths.clear()
            self.reference_items.clear()
            self.reference_ids.clear()
            self.source_preview_sha256 = None
            self.source_preview_bytes = None
            self.user_media.clear()
