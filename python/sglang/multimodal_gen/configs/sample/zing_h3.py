# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

# Zing generates video and audio on the checkpoint's fixed 24 fps time base.
# Output files are encoded by the caller from this field, so a user override
# would change the video duration without changing the audio.
ZING_H3_FPS = 24


@dataclass
class ZingH3SamplingParams(SamplingParams):
    height: int = 512
    width: int = 896
    num_frames: int = 22
    fps: int = field(default=ZING_H3_FPS, init=False)
    num_inference_steps: int = 50  # Euler sigma points; 49 denoising forwards.
    guidance_scale: float = 1.0
    negative_prompt: str = ""
    zing_request: dict | None = None
    zing_solver: str = "euler"
    zing_denoising_step_list: list[int] | None = None
    zing_generate_audio: bool = False
    zing_return_latents: bool = False

    @classmethod
    def video_request_extra_fields(cls):
        return frozenset(
            {
                "zing_request",
                "zing_solver",
                "zing_denoising_step_list",
                "zing_generate_audio",
                "zing_return_latents",
            }
        )

    def build_request_extra(self):
        extra = super().build_request_extra()
        extra.update(
            {name: getattr(self, name) for name in self.video_request_extra_fields()}
        )
        return extra

    def _adjust(self, server_args):
        super()._adjust(server_args)
        self.fps = ZING_H3_FPS

    def _validate(self):
        # CLI and API paths assign fields after construction; pin the time base
        # the same way MiniMaxH3SamplingParams does.
        self.fps = ZING_H3_FPS
        super()._validate()
        # Zing emits paired video/audio on the fixed 24 fps time base. Frame
        # interpolation changes the frame count but not the audio length, and
        # the other post-processing switches have no Zing contract. Reject them
        # the way MiniMaxH3SamplingParams does instead of ignoring them.
        if self.enable_frame_interpolation:
            raise ValueError(
                "Zing does not support enable_frame_interpolation: video and "
                "audio share the fixed 24 fps time base"
            )
        if self.enable_upscaling:
            raise ValueError(
                "Zing does not support enable_upscaling: the output is the "
                "resolved target canvas"
            )
        if self.enable_teacache:
            raise ValueError(
                "Zing does not support enable_teacache: the block-causal "
                "denoise loop has no TeaCache contract"
            )

    def prepare_synthetic_warmup_request_for_queue(self, req, server_args):
        # The generic warmup may request one step; Euler needs a start and an
        # endpoint even when executing only one denoising forward.
        self.num_inference_steps = max(2, int(self.num_inference_steps))
        self.apply_request_extra(req)
