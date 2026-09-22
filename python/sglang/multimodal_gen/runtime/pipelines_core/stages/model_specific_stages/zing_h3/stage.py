# SPDX-License-Identifier: Apache-2.0
import time
from contextlib import contextmanager, nullcontext
from copy import copy

import torch

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.utils.profiler import maybe_record_function
from sglang.multimodal_gen.runtime.utils.zing_diagnostics import (
    current_diagnostic,
    diagnostic_request,
)

from .codecs import ZingH3Codecs
from .packing import ZingH3Packer
from .realtime import ZingH3RealtimeState
from .request_adapter import build_h3_request
from .sampling import ZingH3Sampler


class ZingH3GenerationStage(PipelineStage):
    def __init__(self, modules, config, runtime):
        super().__init__()
        self.modules, self.config, self.runtime = modules, config, runtime
        self.codecs = ZingH3Codecs(modules, config, runtime)
        rope = runtime["rope_config"]
        self.packer = ZingH3Packer(
            config.dit_config.arch_config,
            num_frame_per_block=runtime["num_frame_per_block"],
            split_first_latent=runtime["split_first_latent"],
            spatial_extent=rope["spatial_extent"],
            frame_scale=rope["frame_scale"],
            visual_token_frame_spans=rope["visual_token_frame_spans"],
        )

    def component_uses(self, server_args, stage_name=None):
        uses = [
            ("text_encoder", "encode"),
            ("video_vae", "encode"),
            ("audio_vae", "encode"),
            ("transformer", "denoise"),
            (self.codecs.video_decoder_name, "decode"),
            ("audio_vae", "decode"),
        ]
        return [
            ComponentUse(
                stage_name or self.__class__.__name__,
                name,
                phase=phase,
                start_at_stage_entry=False,
            )
            for name, phase in uses
        ]

    @contextmanager
    def _use(self, name, phase="encode"):
        with self.use_declared_component(
            component_name=name, module=self.modules[name], phase=phase
        ) as module:
            self.modules[name] = module
            module.eval()
            yield module

    def _validate_sampling_params(self, params):
        if params.num_outputs_per_prompt != 1:
            raise ValueError("Zing requires one output per request")
        if (
            params.negative_prompt
            or params.guidance_rescale
            or params.cfg_normalization
        ):
            raise ValueError(
                "Zing CFG uses a target-only unconditional branch without a negative prompt or rescaling"
            )
        if params.quality != "lossless":
            raise ValueError("Zing currently implements only the dense quality path")
        # Playback timestamps use params.fps; this never throttles inference.
        # The caller encodes the output file from params.fps; audio length is
        # computed from the export's fps, so the two must agree.
        if float(params.fps) != self.codecs.target_fps:
            raise ValueError(
                f"Zing generates at {self.codecs.target_fps:g} fps; "
                f"requested fps={params.fps}"
            )

    def _build_request(self, batch):
        params = batch.sampling_params
        canonical = build_h3_request(
            params, batch.extra, media_fps=self.codecs.target_fps
        )
        if canonical is not None:
            return canonical
        request = batch.extra.get("zing_request")
        if request is None:
            request = getattr(params, "zing_request", None)
        if request is None:
            if params.video_path:
                raise ValueError(
                    "Use the target message source field in zing_request for aligned source video"
                )
            content = [{"type": "text", "text": params.prompt}]
            conditions = []
            images = params.image_path or []
            if isinstance(images, str):
                images = [images]
            for index, path in enumerate(images):
                reference_ids = batch.extra.get("zing_reference_ids")
                media_id = (
                    reference_ids[index]
                    if reference_ids is not None
                    else f"image-{index}"
                )
                content.append({"type": "image", "id": media_id, "uri": path})
                conditions.append(
                    {
                        "content_id": media_id,
                        "role": "reference",
                        "latent_modalities": ["visual"],
                    }
                )
            output_type = (
                "video_audio"
                if getattr(params, "zing_generate_audio", False)
                else "image"
                if params.num_frames == 1
                else "video"
            )
            request = {
                "messages": [
                    {"role": "user", "content": content},
                    {
                        "role": "target",
                        "content": [
                            {
                                "type": output_type,
                                "output": {
                                    "frames": params.num_frames,
                                    "height": params.height,
                                    "width": params.width,
                                },
                            }
                        ],
                        "conditions": conditions,
                    },
                ]
            }
        source = batch.extra.get("source", getattr(params, "source", None))
        if source is not None:
            if (
                getattr(params, "zing_request", None) is not None
                or batch.extra.get("zing_request") is not None
            ):
                raise ValueError(
                    "Specify source inside zing_request or use the shared H3 fields"
                )
            request["messages"][1]["source"] = source
        return request

    def _build_sampler(self, params):
        # Validate sampling settings before encoding inputs or loading the DiT.
        return ZingH3Sampler(
            self.modules["transformer"],
            self.packer,
            solver=getattr(params, "zing_solver", "euler"),
            sigma_points=params.num_inference_steps,
            denoising_step_list=getattr(params, "zing_denoising_step_list", None),
            video_shift=self.runtime["timestep_shift"],
            audio_shift=self.runtime["audio_timestep_shift"],
            num_train_timesteps=self.runtime["num_train_timestep"],
            guidance_scale=params.guidance_scale,
            window_chunks=getattr(params, "zing_window_chunks", 21),
            sink_chunks=getattr(params, "zing_sink_chunks", 3),
            rope_position_mode=getattr(
                params, "zing_rope_position_mode", "block_relative"
            ),
            rope_max_frame_gap=getattr(params, "zing_rope_max_frame_gap", 12),
        )

    @torch.no_grad()
    @diagnostic_request
    def forward(self, batch, server_args):
        if getattr(batch, "realtime_session_id", None):
            return self._forward_realtime(batch, server_args)
        params = batch.sampling_params
        self._validate_sampling_params(params)
        request = self._build_request(batch)
        sampler = self._build_sampler(params)
        component_seconds = {}
        diagnostic = current_diagnostic()

        def synchronize():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        def finish_timing(name, started):
            synchronize()
            duration = time.perf_counter() - started
            component_seconds[name] = duration
            if batch.metrics is not None:
                batch.metrics.record_stage(name, duration)

        with set_forward_context(current_timestep=0, attn_metadata=None):
            synchronize()
            started = time.perf_counter()
            with maybe_record_function("Zing.conditioning"):
                inputs = self.codecs.prepare(request, self._use)
            if diagnostic:
                diagnostic.record(
                    "conditioning",
                    positive=inputs.positive,
                    visual=inputs.visual,
                    audio=inputs.audio,
                    source_visual=inputs.source_visual,
                    visual_shape=inputs.visual_shape,
                    audio_shape=inputs.audio_shape,
                )
            finish_timing("ZingH3Conditioning", started)
            with self._use("transformer", "denoise") as transformer:
                # The residency manager may return a replacement module.
                sampler.generator = transformer
                device = next(transformer.parameters()).device
                devices = (
                    [
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    ]
                    if device.type == "cuda"
                    else []
                )
                synchronize()
                started = time.perf_counter()
                with torch.random.fork_rng(devices=devices):
                    seed = (
                        params.seed[0] if isinstance(params.seed, list) else params.seed
                    )
                    torch.manual_seed(int(seed))
                    if diagnostic:
                        diagnostic.record("seeded")
                    video = torch.randn(
                        (1, self.packer.visual_channels, *inputs.visual_shape),
                        device=device,
                    )
                    audio = (
                        None
                        if inputs.audio_shape is None
                        else torch.randn((1, *inputs.audio_shape), device=device)
                    )
                    if diagnostic:
                        diagnostic.record("initial_noise", video=video, audio=audio)
                    capture = (
                        diagnostic.prefix_modules(transformer)
                        if diagnostic
                        else nullcontext()
                    )
                    with maybe_record_function("Zing.denoise"), capture:
                        video, audio = sampler.sample(inputs, video, audio)
                    if diagnostic:
                        diagnostic.record("final_latents", video=video, audio=audio)
                finish_timing("ZingH3Denoise", started)
            with self._use(self.codecs.video_decoder_name, "decode") as vae:
                synchronize()
                started = time.perf_counter()
                video = video.to(next(vae.parameters()).device)
                with maybe_record_function("Zing.video_decode"):
                    frames = self.codecs.decode_visual(video, inputs.pixel_frame_count)
                height, width = (
                    n * self.codecs.spatial_ratio for n in inputs.visual_shape[1:]
                )
                frames = frames[..., :height, :width].contiguous()
                finish_timing("ZingH3VideoDecode", started)
            waveform = None
            if audio is not None:
                with self._use("audio_vae", "decode") as vae:
                    synchronize()
                    started = time.perf_counter()
                    with maybe_record_function("Zing.audio_decode"):
                        waveform = self.codecs.decode_audio(
                            audio.to(next(vae.parameters()).device), inputs.sample_count
                        )
                    finish_timing("ZingH3AudioDecode", started)
            params.num_frames, params.height, params.width = (
                inputs.pixel_frame_count,
                height,
                width,
            )
            params.fps = 1 if inputs.output_type == "image" else self.codecs.target_fps
            return OutputBatch(
                output=frames,
                audio=waveform,
                audio_sample_rate=inputs.sample_rate if waveform is not None else None,
                trajectory_latents=(
                    video.detach()
                    if getattr(params, "zing_return_latents", False)
                    else None
                ),
                metrics=batch.metrics,
                usage={
                    "zing_h3": {
                        **sampler.last_run_metrics,
                        "component_seconds": component_seconds,
                    }
                },
            )

    def _forward_realtime(self, batch, server_args):
        if batch.session is None:
            raise ValueError("Zing realtime request is missing its worker session")
        if "zing_source_command" in batch.extra:
            from .source_session import ZingSourceStreamState

            with set_forward_context(current_timestep=0, attn_metadata=None):
                return batch.session.get_or_create_state(ZingSourceStreamState).run(
                    self, batch
                )
        state = batch.session.get_or_create_state(ZingH3RealtimeState)
        try:
            with set_forward_context(current_timestep=0, attn_metadata=None):
                return self._generate_realtime_block(batch, state)
        except BaseException:
            state.dispose()
            raise

    def _prepare_realtime_state(self, batch, state):
        params = batch.sampling_params
        self._validate_sampling_params(params)
        if state.sampling is None:
            if batch.block_idx != 0:
                raise ValueError("Zing realtime session must start at block zero")
            state.sampler = self._build_sampler(params)
            request = self._build_request(batch)
            state.inputs = self.codecs.prepare(
                request, self._use, retain_condition_media=True
            )
            if state.inputs.audio_shape is not None:
                raise ValueError("Audio streaming is not supported yet.")
            state.prompt = self.codecs._parse_inference_request(request)[0]
        elif batch.block_idx != state.sampling.index:
            raise ValueError("Zing realtime blocks must be requested in order")
        mode = getattr(params, "zing_realtime_decode_mode", "immediate")
        if mode not in {"immediate", "stable"}:
            raise ValueError("Invalid Zing realtime decode mode")
        if self.codecs.video_decoder_name == "taeh3_decoder" and mode != "stable":
            raise ValueError("TAEH3 realtime decoding requires stable mode")
        return mode

    def _sample_realtime_block(self, batch, state):
        params = batch.sampling_params
        inputs, sampler = state.inputs, state.sampler
        prompt_changed = params.prompt != state.prompt
        if prompt_changed:
            inputs = copy(inputs)
            with self._use("text_encoder"):
                inputs.positive = self.codecs.encode_condition(
                    params.prompt, inputs.condition_media
                )
        with self._use("transformer", "denoise") as transformer:
            sampler.generator = transformer
            device = next(transformer.parameters()).device
            seed = params.seed[0] if isinstance(params.seed, list) else params.seed
            with state.random_stream(device, seed):
                if state.sampling is None:
                    visual = torch.randn(
                        (1, self.packer.visual_channels, *inputs.visual_shape),
                        device=device,
                    )
                    state.sampling = sampler.initialize(inputs, visual)
                _, end = sampler.step(
                    state.sampling, inputs=inputs if prompt_changed else None
                )
            state.inputs = inputs
            state.prompt = params.prompt
        return end

    def _queue_realtime_output(self, batch, state, end, mode, final):
        inputs = state.inputs
        full, tail = divmod(end - 1, self.codecs.vae_latents_per_chunk)
        frame_end = min(
            inputs.pixel_frame_count,
            full * self.codecs.vae_frames_per_chunk
            + 1
            + tail * self.codecs.temporal_ratio,
        )
        if state.sampling.index == len(state.sampling.spans):
            frame_end = inputs.pixel_frame_count
        state.pending_blocks.append(
            {
                "block_idx": batch.block_idx,
                "request_id": getattr(batch, "request_id", None),
                "event_id": getattr(batch, "realtime_event_id", None),
                "frame_end": frame_end,
            }
        )
        # A full H3 temporal decode window needs chunk + overlap latent frames.
        # Only its leading clip is final; its overlap can still be blended later.
        chunk = self.codecs.vae_latents_per_chunk
        overlap = (-self.codecs.va.vae_token_drop) % chunk if mode == "stable" else 0
        stable_end = max(0, (end - overlap) // chunk) * self.codecs.vae_frames_per_chunk
        if self.codecs.video_decoder_name == "taeh3_decoder":
            stable_end = max(0, (end // 5) * 17 - 12)
        ready = [
            block
            for block in state.pending_blocks
            if mode == "immediate" or final or block["frame_end"] <= stable_end
        ]
        return ready

    def _decode_realtime_output(self, state, end, ready, final=False):
        inputs = state.inputs
        first_frame = state.emitted_frames
        with self._use(self.codecs.video_decoder_name, "decode"):
            latent = state.sampling.visual[:, :, :end]
            if self.codecs.video_decoder_name == "taeh3_decoder":
                if state.video_decoder is None:
                    state.video_decoder = self.modules["taeh3_decoder"].new_stream()
                stream = state.video_decoder
                parts = [stream.push(latent[:, :, state.decoded_latents : end])]
                state.decoded_latents = end
                if final:
                    parts.append(stream.finish(ready[-1]["frame_end"]))
                if state.decoder_frames is not None:
                    parts.insert(0, state.decoder_frames)
                available = torch.cat(parts, dim=2)
                count = ready[-1]["frame_end"] - first_frame
                if available.shape[2] < count:
                    raise ValueError("TAEH3 output is not ready")
                decoded = available[:, :, :count]
                state.decoder_frames = available[:, :, count:].clone()
            else:
                decoded = self.codecs.decode_visual_range(
                    latent, first_frame, ready[-1]["frame_end"]
                )
            height, width = (
                n * self.codecs.spatial_ratio for n in inputs.visual_shape[1:]
            )
            outputs = []
            for block in ready:
                frames = decoded[
                    0,
                    :,
                    state.emitted_frames - first_frame : block["frame_end"]
                    - first_frame,
                    :height,
                    :width,
                ].contiguous()
                if frames.shape[1] < 1:
                    raise ValueError("Zing realtime block produced no video frames")
                outputs.append(frames)
                state.emitted_frames = block["frame_end"]
        state.pending_blocks = state.pending_blocks[len(ready) :]
        return outputs, height, width

    def _generate_realtime_block(self, batch, state):
        params = batch.sampling_params
        mode = self._prepare_realtime_state(batch, state)
        end = self._sample_realtime_block(batch, state)
        final = state.sampling.index == len(state.sampling.spans) or batch.extra.get(
            "realtime_is_final_chunk", False
        )
        ready = self._queue_realtime_output(batch, state, end, mode, final)
        trajectory = (
            state.sampling.visual[:, :, :end].detach().clone()
            if getattr(params, "zing_return_latents", False)
            else None
        )
        if not ready:
            return OutputBatch(
                # An explicit empty frame batch is a successful buffered step;
                # an entirely absent output is reserved for scheduler failures.
                raw_frame_batches=[],
                metrics=batch.metrics,
                trajectory_latents=trajectory,
            )
        outputs, height, width = self._decode_realtime_output(
            state, end, ready, final=final
        )
        params.num_frames, params.height, params.width = (
            sum(x.shape[1] for x in outputs),
            height,
            width,
        )
        params.fps = self.codecs.target_fps
        # Immediate mode keeps its existing single-tensor transport contract.
        result = OutputBatch(
            output=outputs[0].unsqueeze(0) if mode == "immediate" else outputs,
            metrics=batch.metrics,
            trajectory_latents=trajectory,
            usage={"zing_realtime_blocks": ready} if mode == "stable" else None,
        )
        if final:
            state.dispose()
        return result
