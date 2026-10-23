# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3-VL encoder configuration for MiniMax H3."""

import os
from dataclasses import dataclass, field, fields

from sglang.multimodal_gen.configs.models.encoders.qwen3vl import (
    Qwen3VLArchConfig,
    Qwen3VLConfig,
    Qwen3VLExecutionOptions,
)

MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER = 50


def h3_qwen_execution_disabled() -> bool:
    names = [
        f"SGLANG_H3_QWEN_{item.name.upper()}"
        for item in fields(Qwen3VLExecutionOptions)
    ]
    return all(os.environ.get(name) == "0" for name in (*names, "SGLANG_H3_QWEN_FP8"))


def h3_qwen_fp8_enabled() -> bool:
    value = os.environ.get("SGLANG_H3_QWEN_FP8", "1")
    if value not in ("0", "1"):
        raise ValueError("SGLANG_H3_QWEN_FP8 must be 0 or 1")
    return value == "1"


def resolve_h3_qwen_online_quantization(explicit: str | None) -> str | None:
    """Resolve the H3-only FP8 alias before the shared loader owns conversion."""
    if h3_qwen_fp8_enabled():
        if explicit not in (None, "fp8"):
            raise ValueError(
                "SGLANG_H3_QWEN_FP8=1 conflicts with the encoder quantization override"
            )
        return "fp8"
    if explicit == "fp8":
        raise ValueError("SGLANG_H3_QWEN_FP8=0 conflicts with explicit encoder FP8")
    return explicit


@dataclass
class MiniMaxH3Qwen3VLArchConfig(Qwen3VLArchConfig):
    """The checkpoint is Qwen3-VL-32B, consumed at hidden_states[50]."""

    architectures: list[str] = field(
        default_factory=lambda: ["MiniMaxH3Qwen3VLEncoder"]
    )
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_hidden_layers: int = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    text_len: int = 262144
    hidden_state_skip_layer: int = 0
    checkpoint_num_hidden_layers: int = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
    conditioning_projection_path: str | None = None


@dataclass
class MiniMaxH3Qwen3VLConfig(Qwen3VLConfig):
    arch_config: MiniMaxH3Qwen3VLArchConfig = field(
        default_factory=MiniMaxH3Qwen3VLArchConfig
    )

    def post_diffusers_config_update(self) -> None:
        """Select the in-tree extractor after loading the HF architecture."""

        arch = self.arch_config
        arch.checkpoint_num_hidden_layers = int(arch.text_config.num_hidden_layers)
        arch.architectures = ["MiniMaxH3Qwen3VLEncoder"]
        arch.hidden_size = int(arch.text_config.hidden_size)
        arch.intermediate_size = int(arch.text_config.intermediate_size)
        arch.num_attention_heads = int(arch.text_config.num_attention_heads)
        arch.num_key_value_heads = int(arch.text_config.num_key_value_heads)
        arch.head_dim = int(arch.text_config.head_dim)
        arch.num_hidden_layers = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        arch.text_config.num_hidden_layers = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        arch.text_config.output_hidden_states = False
        arch.text_config.use_cache = False


__all__ = [
    "MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER",
    "MiniMaxH3Qwen3VLArchConfig",
    "MiniMaxH3Qwen3VLConfig",
]
