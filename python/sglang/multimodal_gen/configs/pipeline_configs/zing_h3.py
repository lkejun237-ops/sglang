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
from sglang.multimodal_gen.configs.pipeline_configs.zing_h3_decoder import (
    LEGACY_DECODER_FIELDS,
    DecoderSpec,
    normalize_decoder_spec,
)


def resolve_zing_modality_tags(num_modalities, tags=None):
    """Resolve legacy canonical tags or validate native four/five-branch routing."""
    roles = ("video", "text", "audio", "source_visual")
    if type(num_modalities) is not int or num_modalities not in (4, 5):
        raise ValueError("Zing requires four or five AdaLN modalities")
    if tags is None:
        tags = dict(zip(roles, range(4)))
        if num_modalities == 5:
            tags["condition_visual"] = 4
    required = set(roles)
    if (
        not isinstance(tags, dict)
        or not required <= tags.keys() <= required | {"condition_visual"}
        or any(type(value) is not int for value in tags.values())
    ):
        raise ValueError("Unsupported Zing modality routing")
    tags = dict(tags)
    tags.setdefault("condition_visual", tags["video"])
    if tags["condition_visual"] != tags["video"]:
        roles += ("condition_visual",)
    if len(roles) != num_modalities or sorted(tags[role] for role in roles) != list(
        range(num_modalities)
    ):
        raise ValueError(
            "Zing projection tags must match the contiguous modality count"
        )
    return tags


@dataclass
class ZingH3DiTArchConfig(MiniMaxH3DiTArchConfig):
    num_modalities: int = 4
    adaln_out_features: int = 24 * 5376
    checkpoint_uses_diffusers_layout: bool = True
    # None preserves canonical routing for existing version-1 exports.
    modality_tags: dict[str, int] | None = None
    ar_condition_timestep: float = 1.0
    reference_visual_timestep: float = 1.0


@dataclass
class ZingH3DiTConfig(MiniMaxH3DiTConfig):
    arch_config: ZingH3DiTArchConfig = field(default_factory=ZingH3DiTArchConfig)


@dataclass
class ZingH3CausalPipelineConfig(MiniMaxH3PipelineConfig):
    dit_config: ZingH3DiTConfig = field(default_factory=ZingH3DiTConfig)
    should_use_guidance: bool = True
    vae_decode_precision: str = "fp32"
    # Flat legacy selection; None means "not chosen", so an explicit taeh3 stays visible.
    video_decoder_backend: str | None = None
    taeh3_checkpoint_path: str | None = None
    taeh3_decode_precision: str | None = None
    # Nested decoder selection; with neither this nor a legacy choice the default is Lynn.
    decoder_config: dict | None = None
    # Immutable offline table; None keeps the explicit resident baseline.
    zing_adaln_table_path: str | None = None
    zing_adaln_table_max_bytes: int = 536870912
    # Realtime KV pool (#171). None disables it and keeps per-session growth.
    zing_kv_pool_size: int | None = None
    zing_kv_pool_max_target_area: int | None = None
    zing_kv_pool_max_target_long_edge: int | None = None
    zing_kv_pool_max_prefix_rows: int | None = None
    zing_kv_pool_window_chunks: int = 21
    zing_kv_pool_sink_chunks: int = 3
    zing_kv_pool_rope_position_mode: str = "block_relative"
    zing_kv_pool_rope_max_frame_gap: int = 12
    # Runtime-owned rolling target history used when a WebSocket prompt changes.
    # None keeps the existing prompt-update path for exports without 1,2,2 geometry.
    prompt_switch_prefix_latent_frames: int | None = None

    # Legacy decoder fields the caller supplied; a default must not look like a choice.
    _explicit_decoder_fields = frozenset()

    def update_config_from_dict(self, args, prefix: str = "") -> None:
        dotted = f"{prefix}." if prefix.strip() else ""
        self._explicit_decoder_fields = frozenset(
            name
            for name in LEGACY_DECODER_FIELDS
            if args.get(dotted + name) is not None
        )
        super().update_config_from_dict(args, prefix)

    def update_pipeline_config(self, source_pipeline_dict) -> None:
        self._explicit_decoder_fields = frozenset(
            name
            for name in LEGACY_DECODER_FIELDS
            if source_pipeline_dict.get(name) is not None
        )
        super().update_pipeline_config(source_pipeline_dict)

    def resolve_decoder_spec(self) -> DecoderSpec:
        explicit = self._explicit_decoder_fields or frozenset(
            name
            for name in LEGACY_DECODER_FIELDS
            if getattr(self, name) != type(self).__dataclass_fields__[name].default
        )
        return normalize_decoder_spec(
            self.decoder_config,
            video_decoder_backend=self.video_decoder_backend,
            taeh3_checkpoint_path=self.taeh3_checkpoint_path,
            taeh3_decode_precision=self.taeh3_decode_precision,
            vae_decode_precision=self.vae_decode_precision,
            explicit_legacy=explicit,
        )

    @property
    def decoder_type(self) -> str:
        return self.resolve_decoder_spec().type

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
        if self.prompt_switch_prefix_latent_frames is not None and (
            type(self.prompt_switch_prefix_latent_frames) is not int
            or self.prompt_switch_prefix_latent_frames != 5
        ):
            raise ValueError(
                "prompt_switch_prefix_latent_frames currently only supports 5"
            )
        if self.zing_adaln_table_path is not None:
            if (
                type(self.zing_adaln_table_max_bytes) is not int
                or self.zing_adaln_table_max_bytes <= 0
            ):
                raise ValueError("Zing AdaLN table budget must be positive")
            if getattr(server_args, "lora_path", None):
                raise ValueError(
                    "Zing AdaLN tables require offline-merged LoRA weights"
                )
            if (getattr(server_args, "sp_degree", 1) or 1) != 1:
                raise ValueError(
                    "Zing AdaLN tables currently support SP1 only; SP2 exactness is unverified"
                )
        decoder = self.resolve_decoder_spec()
        if decoder.type == "lynn_streaming":
            # Offload of the mixed BF16/FP32 decoder is unvalidated; keep it resident.
            residency = getattr(server_args, "explicit_residency_mode", None)
            mode = residency("lynn_decoder") if residency is not None else None
            if mode not in (None, "resident"):
                raise ValueError("lynn_streaming requires a resident decoder")
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
        from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
            ZingKVPoolConfig,
        )

        ZingKVPoolConfig.from_pipeline_config(
            self,
            max_sessions=getattr(server_args, "realtime_max_sessions_per_worker", 1),
        )
        selected = self.resolve_transformer_attention_backend(server_args)
        if (
            selected is not None
            and getattr(selected, "name", str(selected)).upper() != "TORCH_SDPA"
        ):
            raise ValueError("Zing dense inference requires TORCH_SDPA attention")
