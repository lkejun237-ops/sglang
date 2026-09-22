# SPDX-License-Identifier: Apache-2.0
"""Version-one RGB source on the shared realtime scheduler and transport."""

import asyncio
import hashlib
import io
import json
import tempfile
import time
from collections import deque
from copy import copy
from dataclasses import asdict
from pathlib import Path

import msgspec.msgpack
from PIL import Image

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    BaseRealtimeModelAdapter,
    RealtimeChunkInputs,
    build_realtime_sampling_params,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange, require
from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
    SourceLedger,
    SourceLimits,
    default_source_encoder,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args


class ZingH3SourceStreamAdapter(BaseRealtimeModelAdapter):
    def create_state(self):
        return None  # Initialized atomically after negotiation validation.

    async def on_init(self, session, request):
        args = get_global_server_args()
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
            "zing_solver",
            "zing_denoising_step_list",
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
            request.num_inference_steps in (None, 4)
            and getattr(request, "zing_realtime_decode_mode", "stable") == "stable",
            "This session requires stable four-step rendering.",
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
        require(
            profile.get("model_config_sha256")
            == hashlib.sha256(config_bytes).hexdigest()
            and profile.get("solver") == "few_step"
            and isinstance(profile.get("denoising_step_list"), list)
            and len(profile["denoising_step_list"]) == 4
            and profile.get("noise_version") == "indexed-cpu-v1",
            "The deployed model is not configured for four-step streaming.",
        )
        require(
            "guidance_scale" not in request.model_fields_set
            or request.guidance_scale == profile["guidance_scale"],
            "The requested guidance does not match this model.",
        )
        source = getattr(request, "source_stream", None)
        width, height = map(int, request.size.split("x"))
        fps = int(runtime["h3_media_config"]["fps"])
        require(
            source == dict(format="rgb24", width=width, height=height, fps=fps)
            and request.fps in (None, fps),
            "Please provide matching RGB source dimensions and frame rate.",
        )
        require(
            min(width, height) > 0 and width % 16 == height % 16 == 0,
            "Please use video dimensions divisible by 16.",
        )
        require(
            request.realtime_preview_max_width in (None, width),
            "This session requires the full output dimensions.",
        )
        references = getattr(request, "references", []) or []
        require(
            isinstance(references, list) and len(references) <= 9,
            "Please provide at most nine reference images.",
        )
        seen, reference_paths, reference_bytes = set(), [], 0
        for i, ref in enumerate(references):
            require(
                isinstance(ref, dict)
                and isinstance(ref.get("id"), str)
                and ref["id"]
                and ref["id"] not in seen
                and ref.get("format") in ("jpeg", "webp", "png")
                and isinstance(ref.get("payload"), bytes),
                "Please provide uniquely named JPEG, WebP, or PNG reference images.",
            )
            seen.add(ref["id"])
            reference_bytes += len(ref["payload"])
            require(reference_bytes <= 8 << 20, "The reference images are too large.")
            with Image.open(io.BytesIO(ref["payload"])) as image:
                require(
                    image.format.lower() == ref["format"]
                    and image.width * image.height <= 16_777_216,
                    "The reference image format or dimensions are not supported.",
                )
                image.verify()
            if session.input_temp_dir is None:
                session.input_temp_dir = tempfile.mkdtemp(prefix="sglang_zing_source_")
            path = Path(session.input_temp_dir) / f"reference-{i}.{ref['format']}"
            path.write_bytes(ref["payload"])
            reference_paths.append(str(path))
        self.reference_items = [
            dict(type="image", id=ref["id"], uri=path)
            for ref, path in zip(references, reference_paths)
        ]
        # Prompt-only source snapshot: never an aligned source or a sixth
        # visual reference. Video semantics preserve the original prompt IDs.
        self.prompt_video = None
        frame = getattr(request, "prompt_video_frame", None)
        if frame is not None:
            require(
                isinstance(frame, dict)
                and isinstance(frame.get("id"), str)
                and frame["id"]
                and frame["id"] not in seen
                and frame.get("format") == "png"
                and isinstance(frame.get("payload"), bytes),
                "Please provide a uniquely named PNG prompt video frame.",
            )
            require(
                reference_bytes + len(frame["payload"]) <= 8 << 20,
                "The reference images are too large.",
            )
            with Image.open(io.BytesIO(frame["payload"])) as image:
                require(
                    image.format == "PNG" and image.width * image.height <= 16_777_216,
                    "The reference image format or dimensions are not supported.",
                )
                image.verify()
            if session.input_temp_dir is None:
                session.input_temp_dir = tempfile.mkdtemp(prefix="sglang_zing_source_")
            path = Path(session.input_temp_dir) / "prompt-video-frame.png"
            path.write_bytes(frame["payload"])
            self.prompt_video = dict(
                type="video", id=frame["id"], uri=str(path), clip=dict(start=0, end=0)
            )
        clip = int(args.pipeline_config.vae_config.arch_config.vae_clip_length)
        require(
            clip * width * height * 3 + 4096 <= 32 << 20,
            "The source dimensions exceed the supported streaming size.",
        )
        limits = SourceLimits(
            clip,
            (clip * width * height * 3) + 4096,
            4 * clip,
            4 * clip * width * height * 3,
            8 * clip,
        )
        self.config = dict(
            width=width,
            height=height,
            fps=fps,
            limits=asdict(limits),
            total_frames=request.num_frames,
        )
        self.ledger = SourceLedger(**{**self.config, "limits": limits})
        self.pending, self.available = deque(), asyncio.Event()
        self.pending_frames, self.end, self.end_dispatched = 0, None, False
        self.clip, self.reference_paths, self.profile = clip, reference_paths, profile
        self.reference_ids = [ref["id"] for ref in references]
        self.source_encoder = profile.get("source_encoder", default_source_encoder())
        self.first_source_received_unix_ns = None
        self.pending_prompt = None
        self.prompt_event = None
        request.fps, request.num_inference_steps, request.guidance_scale = (
            fps,
            4,
            profile["guidance_scale"],
        )
        request.realtime_output_format = request.realtime_output_format or "raw"
        request.realtime_preview_max_width = width
        self.ready = {
            "type": "session.ready",
            "zing_realtime_api_version": 1,
            "session_id": session.id,
            "num_inference_steps": 4,
            "realtime_output_format": request.realtime_output_format,
            "source_stream": source,
            "max_source_batch_frames": limits.batch_frames,
            "max_source_message_bytes": limits.message_bytes,
            "max_pending_source_frames": limits.pending_frames,
            "max_pending_source_bytes": limits.pending_bytes,
            "max_references": 9,
            "max_reference_bytes": 8 << 20,
            "prompt_video_frame_format": "png",
        }

    def session_ready_message(self, session):
        return self.ready

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
            # Only expose work after accepted is on the same serialized writer.
            await ws.send_bytes(msgspec.msgpack.encode(record.message()))
            self.pending.append(
                dict(payload=payload, event_id=event_id, message_bytes=message_bytes)
            )
            self.pending_frames += payload["num_frames"]
            self.available.set()
        elif kind == "source.end":
            require(
                isinstance(payload, dict), "Please provide the final source position."
            )
            self.ledger.finish(payload.get("source_frame_end"), event_id=event_id)
            self.end = dict(frame_end=self.ledger.accepted_frames, event_id=event_id)
            self.available.set()
        elif kind == "prompt":
            self.ledger.validate_event_id(event_id)
            require(
                isinstance(payload, str) and payload.strip(),
                "Please provide a non-empty prompt.",
            )
            self.ledger.event_id = event_id
            self.pending_prompt = payload, event_id
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
        minimum = 1 if self.source_encoder["mode"] == "progressive" else self.clip
        while self.pending_frames < minimum and self.end is None:
            self.available.clear()
            await self.available.wait()

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
            num_inference_steps=4,
            chunk_size=chunk_size,
        )
        params.image_path = self.reference_paths
        # Only locally validated files enter the internal canonical request.
        # The wire API still rejects arbitrary zing_request/source objects.
        content = ([self.prompt_video] if self.prompt_video else []) + list(
            self.reference_items
        )
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
        params.zing_solver = self.profile["solver"]
        params.zing_denoising_step_list = self.profile["denoising_step_list"]
        params.zing_realtime_decode_mode = "stable"
        params.zing_window_chunks, params.zing_sink_chunks = 21, 3
        params.zing_rope_position_mode, params.zing_rope_max_frame_gap = (
            "block_relative",
            12,
        )
        return params

    def prepare_next_request(self, session, server_args, chunk):
        batch = super().prepare_next_request(session, server_args, chunk)
        appends, count = [], 0
        while self.pending and count < self.clip:
            item = self.pending.popleft()
            count += item["payload"]["num_frames"]
            appends.append(item)
        self.pending_frames -= count
        end = self.end if not self.pending else None
        if self.end_dispatched:
            raise ValueError("Source end was already dispatched")
        self.end_dispatched = end is not None
        batch.extra["zing_reference_ids"] = self.reference_ids
        batch.extra["zing_source_command"] = dict(
            config=self.config,
            appends=appends,
            end=end,
            encoder=self.source_encoder,
            first_source_received_unix_ns=self.first_source_received_unix_ns,
            dispatch_unix_ns=time.time_ns(),
            measure_cuda=self.profile.get("measure_source_cuda", False),
        )
        return batch

    def on_chunk_complete(self, session, result):
        super().on_chunk_complete(session, result)
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
        if hasattr(self, "ledger"):
            self.ledger.dispose()
            self.pending.clear()
            self.available.set()
            self.pending_prompt = None
            self.reference_paths.clear()
            self.reference_items.clear()
            self.reference_ids.clear()
            self.prompt_video = None
