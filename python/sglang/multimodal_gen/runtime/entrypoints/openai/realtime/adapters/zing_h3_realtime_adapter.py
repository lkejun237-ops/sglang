# SPDX-License-Identifier: Apache-2.0
"""Finite, video-only Zing AR sessions on the shared realtime protocol."""

import json
import math
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    BaseRealtimeModelAdapter,
    RealtimeChunkInputs,
    build_realtime_sampling_params,
    save_realtime_first_frame,
)
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
from sglang.multimodal_gen.runtime.server_args import get_global_server_args


@dataclass
class ZingH3RealtimeControlState:
    chunk_plan: ChunkPlan = field(default_factory=ChunkPlan)
    latent_frames: int = 0
    pending_prompt: tuple[str, int | None] | None = None
    event_id: int | None = None


class ZingH3RealtimeAdapter(BaseRealtimeModelAdapter):
    def create_state(self):
        return ZingH3RealtimeControlState()

    async def on_init(self, session, request):
        args = get_global_server_args()
        if getattr(args, "realtime_vae_backend", "local") != "local" or getattr(
            session, "vae_worker_url", None
        ):
            raise ValueError("This model does not support remote video decoding yet.")
        runtime = json.loads((Path(args.model_path) / "zing_config.json").read_text())
        state = session.adapter_state
        try:
            state.chunk_plan = ChunkPlan(
                runtime["num_frame_per_block"], runtime["split_first_latent"]
            )
        except ValueError as exc:
            raise ValueError(
                "The model has an invalid video chunk configuration."
            ) from exc
        if getattr(request, "zing_realtime_decode_mode", "immediate") not in {
            "immediate",
            "stable",
        }:
            raise ValueError("Please choose immediate or stable video decoding.")
        if request.n not in (None, 1) or request.num_outputs_per_prompt not in (
            None,
            1,
        ):
            raise ValueError("Please request one video per streaming session.")
        if request.condition_inputs:
            raise ValueError("This model supports prompt updates only.")
        if getattr(request, "zing_generate_audio", False):
            raise ValueError("Audio streaming is not supported yet.")
        custom = getattr(request, "zing_request", None)
        if custom is not None:
            try:
                messages = {message["role"]: message for message in custom["messages"]}
                target = messages["target"]["content"][0]
                prompt = next(
                    item["text"]
                    for item in messages["user"]["content"]
                    if item["type"] == "text"
                )
                output = target["output"]
                frames, height, width = (
                    int(output[key]) for key in ("frames", "height", "width")
                )
                if target["type"] == "video_audio" or min(frames, height, width) < 1:
                    raise ValueError("Invalid streaming target")
            except (ValueError, TypeError, KeyError, IndexError, StopIteration) as exc:
                raise ValueError(
                    "Please provide a valid video-only generation request."
                ) from exc
            request.prompt = prompt
            request.num_frames = frames
            request.size = f"{width}x{height}"
        if request.video_path or request.video_url:
            raise ValueError(
                "Please provide the source video in the generation request."
            )
        if request.enable_frame_interpolation or request.enable_upscaling:
            raise ValueError("Video enhancement is not supported while streaming.")
        if request.realtime_causal_kv_cache_num_frames not in (
            None,
            -1,
        ) or request.realtime_causal_sink_size not in (None, 0):
            raise ValueError("This model requires the complete video history.")
        fps = float(runtime["h3_media_config"]["fps"])
        if not fps.is_integer() or fps <= 0:
            raise ValueError("The model's frame rate is not supported for streaming.")
        if request.fps is not None and request.fps != fps:
            raise ValueError("The requested frame rate is not supported by this model.")
        request.fps = int(fps)
        request.num_frames = (
            request.num_frames if request.num_frames is not None else 22
        )
        if request.num_frames < 1:
            raise ValueError("Request at least one video frame.")
        va = args.pipeline_config.vae_config.arch_config
        full, tail = divmod(request.num_frames - 1, va.vae_clip_length)
        latent_frames = (
            full * math.ceil(va.vae_clip_length / va.temporal_compression_ratio)
            + 1
            + math.ceil(tail / va.temporal_compression_ratio)
        )
        state.latent_frames = latent_frames
        blocks = state.chunk_plan.count(latent_frames)
        request.max_chunks = min(request.max_chunks or blocks, blocks)
        if custom is None and "size" not in request.model_fields_set:
            request.size = "896x512"
        await save_realtime_first_frame(session, request)

    def ingest_event(self, session, event):
        if event.kind != "prompt":
            raise ValueError("This model supports prompt updates only.")
        if not isinstance(event.payload, str) or not event.payload:
            raise ValueError("Please provide a non-empty prompt.")
        session.adapter_state.pending_prompt = (event.payload, event.event_id)
        return f"kind=prompt, prompt_len={len(event.payload)}"

    def get_chunk_size(self, session, server_args, chunk):
        state = session.adapter_state
        planned = state.chunk_plan.chunk(chunk.index, state.latent_frames)
        if planned is None:
            raise ValueError("The video session is complete.")
        return planned.end - planned.start

    def sample_chunk_inputs(self, session, server_args, chunk, chunk_size):
        state = session.adapter_state
        if state.pending_prompt is not None:
            session.request.prompt, state.event_id = state.pending_prompt
            state.pending_prompt = None
        return RealtimeChunkInputs(prompt=session.request.prompt)

    def build_sampling_params(
        self, session, server_args, chunk, chunk_inputs, chunk_size
    ):
        request = session.request
        params = build_realtime_sampling_params(
            chunk.request_id,
            request=request,
            chunk_inputs=chunk_inputs,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            chunk_size=chunk_size,
        )
        # Preserve the model-specific solver without changing the wire envelope.
        for name in (
            "zing_solver",
            "zing_denoising_step_list",
            "zing_request",
            "zing_realtime_decode_mode",
        ):
            value = getattr(request, name, None)
            if value is not None:
                setattr(params, name, value)
        return params

    def get_realtime_event_id(self, session):
        return session.adapter_state.event_id

    def clear_state(self, session):
        session.adapter_state.pending_prompt = None
        session.adapter_state.event_id = None

    async def send_output(self, ws, session, result, batch):
        blocks = (result.usage or {}).get("zing_realtime_blocks")
        if not blocks:
            return await super().send_output(ws, session, result, batch)
        if result.raw_frame_batches is None or len(blocks) != len(
            result.raw_frame_batches
        ):
            raise ValueError("Zing decoded block metadata does not match frame batches")
        total = None
        for metadata, frames in zip(blocks, result.raw_frame_batches):
            block_batch, block_result = copy(batch), copy(result)
            block_batch.sampling_params = copy(batch.sampling_params)
            block_batch.block_idx = metadata["block_idx"]
            block_batch.request_id = metadata["request_id"]
            block_batch.realtime_event_id = metadata["event_id"]
            block_result.raw_frame_batches = [frames]
            stats = await super().send_output(ws, session, block_result, block_batch)
            if total is None:
                total = stats
            else:
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        total[key] += value
        return total
