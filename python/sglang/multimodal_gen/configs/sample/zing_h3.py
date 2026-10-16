# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class ZingH3SamplingParams(SamplingParams):
    height: int = 480
    width: int = 832
    num_frames: int = 22
    adjust_frames: bool = False
    enable_sequence_shard: bool | None = True
    # Media playback time base, never a cap on measured generation throughput.
    fps: int = 24
    num_inference_steps: int = 50  # Euler sigma points; 49 denoising forwards.
    guidance_scale: float = 1.0
    negative_prompt: str = ""
    # Share H3 public request fields. source extends references with paired input.
    task: str | None = None
    conditions: list[dict] | None = None
    target: dict | None = None
    source: dict | None = None
    zing_request: dict | None = None
    zing_solver: str = "euler"
    zing_denoising_step_list: list[int] | None = None
    zing_generate_audio: bool = False
    zing_return_latents: bool = False
    zing_realtime_decode_mode: str = "stable"
    zing_joint_forward: str = "cross_chunk"
    zing_window_eviction_timing: str = "after_step0"
    zing_window_chunks: int | None = 21
    zing_sink_chunks: int = 3
    zing_rope_position_mode: str = "block_relative"
    zing_rope_max_frame_gap: int = 12

    def _adjust(self, server_args):
        enable_sequence_shard = self.enable_sequence_shard
        sp_degree = getattr(server_args, "sp_degree", 1) or 1
        if sp_degree > 1 and enable_sequence_shard is False:
            raise ValueError(
                "Zing H3 with sp_degree > 1 requires enable_sequence_shard=True."
            )
        if enable_sequence_shard is None or enable_sequence_shard:
            self.adjust_frames = False
        super()._adjust(server_args)
        if enable_sequence_shard is None or enable_sequence_shard:
            self.enable_sequence_shard = True
            self.adjust_frames = False

    @classmethod
    def video_request_extra_fields(cls):
        return frozenset(
            {
                "task",
                "conditions",
                "target",
                "source",
                "zing_request",
                "zing_solver",
                "zing_denoising_step_list",
                "zing_generate_audio",
                "zing_return_latents",
                "zing_realtime_decode_mode",
                "zing_joint_forward",
                "zing_window_eviction_timing",
                "zing_window_chunks",
                "zing_sink_chunks",
                "zing_rope_position_mode",
                "zing_rope_max_frame_gap",
            }
        )

    def build_request_extra(self):
        extra = super().build_request_extra()
        extra.update(
            {name: getattr(self, name) for name in self.video_request_extra_fields()}
        )
        return extra

    def _validate(self):
        super()._validate()
        from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

        if self.zing_window_eviction_timing not in ("before_source", "after_step0"):
            raise ValueError(
                "Zing window eviction timing must be before_source or after_step0"
            )
        if self.zing_joint_forward not in {"off", "source_first", "cross_chunk"}:
            raise ValueError(
                "Zing joint_forward must be off, source_first, or cross_chunk"
            )
        ZingWindowPolicy(self.zing_window_chunks, self.zing_sink_chunks)
        ZingWindowPolicy.validate_rope(
            self.zing_rope_position_mode, self.zing_rope_max_frame_gap
        )
        # Zing emits paired video/audio on the checkpoint time base. Frame
        # interpolation changes the frame count but not the audio length, and
        # the other post-processing switches have no Zing contract. Reject them
        # the way MiniMaxH3SamplingParams does instead of ignoring them.
        if self.enable_frame_interpolation:
            raise ValueError(
                "Zing does not support enable_frame_interpolation: video and "
                "audio share the checkpoint time base"
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
