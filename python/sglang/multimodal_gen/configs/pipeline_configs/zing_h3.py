# SPDX-License-Identifier: Apache-2.0
"""Zing1 causal architecture and deployment contract."""

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
        for name in ("tp_size", "sp_degree", "ulysses_degree", "ring_degree"):
            if getattr(server_args, name, 1) not in (None, 1):
                raise ValueError("Zing dense inference requires TP=1 and SP=1")
        for name in (
            "enable_cfg_parallel",
            "enable_torch_compile",
            "use_fsdp_inference",
        ):
            if getattr(server_args, name, False):
                raise ValueError(f"Zing dense inference does not support {name}")
        selected = self.resolve_transformer_attention_backend(server_args)
        if (
            selected is not None
            and getattr(selected, "name", str(selected)).upper() != "TORCH_SDPA"
        ):
            raise ValueError("Zing dense inference requires TORCH_SDPA attention")
