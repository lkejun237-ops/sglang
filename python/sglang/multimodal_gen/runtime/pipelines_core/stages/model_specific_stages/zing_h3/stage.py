# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager

import torch

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage

from .codecs import ZingH3Codecs
from .packing import ZingH3Packer
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
            ("video_vae", "decode"),
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
        # The caller encodes the output file from params.fps; audio length is
        # computed from the export's fps, so the two must agree.
        if float(params.fps) != self.codecs.target_fps:
            raise ValueError(
                f"Zing generates at {self.codecs.target_fps:g} fps; "
                f"requested fps={params.fps}"
            )

    def _build_request(self, batch):
        params = batch.sampling_params
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
                media_id = f"image-{index}"
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
        )

    @torch.no_grad()
    def forward(self, batch, server_args):
        params = batch.sampling_params
        self._validate_sampling_params(params)
        request = self._build_request(batch)
        sampler = self._build_sampler(params)

        with set_forward_context(current_timestep=0, attn_metadata=None):
            inputs = self.codecs.prepare(request, self._use)
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
                with torch.random.fork_rng(devices=devices):
                    seed = (
                        params.seed[0] if isinstance(params.seed, list) else params.seed
                    )
                    torch.manual_seed(int(seed))
                    video = torch.randn(
                        (1, self.packer.visual_channels, *inputs.visual_shape),
                        device=device,
                    )
                    audio = (
                        None
                        if inputs.audio_shape is None
                        else torch.randn((1, *inputs.audio_shape), device=device)
                    )
                    video, audio = sampler.sample(inputs, video, audio)
            with self._use("video_vae", "decode") as vae:
                video = video.to(next(vae.parameters()).device)
                frames = self.codecs.decode_visual(video, inputs.pixel_frame_count)
                height, width = (
                    n * self.codecs.spatial_ratio for n in inputs.visual_shape[1:]
                )
                frames = frames[..., :height, :width].contiguous()
            waveform = None
            if audio is not None:
                with self._use("audio_vae", "decode") as vae:
                    waveform = self.codecs.decode_audio(
                        audio.to(next(vae.parameters()).device), inputs.sample_count
                    )
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
            )
