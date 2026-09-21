# SPDX-License-Identifier: Apache-2.0
"""Zing1 causal architecture and deployment contract."""

import os
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)


@dataclass
class ZingH3DiTArchConfig(MiniMaxH3DiTArchConfig):
    num_modalities: int = 4
    adaln_out_features: int = 24 * 5376
    checkpoint_uses_diffusers_layout: bool = True


@dataclass
class ZingH3DiTConfig(MiniMaxH3DiTConfig):
    arch_config: ZingH3DiTArchConfig = field(default_factory=ZingH3DiTArchConfig)


@dataclass
class ZingH3CausalPipelineConfig(MiniMaxH3PipelineConfig):
    dit_config: ZingH3DiTConfig = field(default_factory=ZingH3DiTConfig)
    should_use_guidance: bool = True
    vae_decode_precision: str = "fp32"
    video_decoder_backend: str = "native"
    taeh3_checkpoint_path: str | None = None
    taeh3_decode_precision: str = "fp32"

    @property
    def requires_audio_output(self):
        return False

    def get_model_deployment_config(self):
        return ModelDeploymentConfig(
            speed_mode_enable_torch_compile_by_default=False,
            auto_enable_cfg_parallel=False,
            supports_cfg_parallel=False,
            dit_layerwise_offload_modes=("memory",),
        )

    def validate_quality_deployment(self, server_args):
        self.validate_server_args(server_args)

    def validate_server_args(self, server_args):
        if self.video_decoder_backend not in {"native", "taeh3"}:
            raise ValueError("Unknown Zing video decoder backend")
        if self.video_decoder_backend == "taeh3":
            if not self.taeh3_checkpoint_path:
                raise ValueError("TAEH3 requires a local checkpoint path")
            if self.taeh3_decode_precision not in {"fp32", "fp16", "bf16"}:
                raise ValueError("Unsupported TAEH3 decode precision")
        tp_size = getattr(server_args, "tp_size", 1) or 1
        sp_degree = getattr(server_args, "sp_degree", 1) or 1
        ulysses_degree = getattr(server_args, "ulysses_degree", 1) or 1
        ring_degree = getattr(server_args, "ring_degree", 1) or 1
        if tp_size != 1:
            raise ValueError("Zing causal inference requires TP=1")
        if ring_degree != 1:
            raise ValueError("Zing causal inference supports Ulysses SP only")
        if sp_degree != ulysses_degree:
            raise ValueError(
                "Zing causal inference requires --sp-degree == --ulysses-degree"
            )
        heads = self.dit_config.arch_config.num_attention_heads
        if heads % ulysses_degree:
            raise ValueError(
                f"Zing attention heads ({heads}) must be divisible by "
                f"ulysses_degree ({ulysses_degree})"
            )
        for name in (
            "enable_cfg_parallel",
            "enable_torch_compile",
        ):
            if getattr(server_args, name, False):
                raise ValueError(f"Zing dense inference does not support {name}")
        if sp_degree > 1:
            required = {
                "SGLANG_ZING_EXPERIMENTAL_ATTENTION": "fa4",
                "SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES": "1",
                "SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN": "1",
            }
            missing = [
                f"{name}={value}"
                for name, value in required.items()
                if os.environ.get(name, value) != value
            ]
            if missing:
                raise ValueError(
                    "Zing SP requires the numerically validated configuration: "
                    + ", ".join(missing)
                )
        selected = self.resolve_transformer_attention_backend(server_args)
        if (
            selected is not None
            and getattr(selected, "name", str(selected)).upper() != "TORCH_SDPA"
        ):
            raise ValueError("Zing dense inference requires TORCH_SDPA attention")
