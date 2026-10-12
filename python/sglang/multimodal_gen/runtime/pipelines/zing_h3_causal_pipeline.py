# SPDX-License-Identifier: Apache-2.0
"""Native Zing1/H3 causal inference with shared H3 codecs."""

import json
import math
import os
from pathlib import Path

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
    ZingH3CausalPipelineConfig,
    resolve_zing_modality_tags,
)
from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
from sglang.multimodal_gen.runtime.conditioning.worker import (
    EXCLUDED_ENCODER_COMPONENTS,
    RemoteConditioner,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
    ZingH3GenerationStage,
)


class ZingH3CausalPipeline(ComposedPipelineBase):
    pipeline_name = "ZingH3CausalPipeline"
    pipeline_config_cls = ZingH3CausalPipelineConfig
    sampling_params_cls = ZingH3SamplingParams
    is_video_pipeline = True
    _required_config_modules = [
        "processor",
        "text_encoder",
        "tokenizer",
        "video_vae",
        "audio_vae",
        "transformer",
    ]

    def __init__(
        self,
        model_path,
        server_args,
        required_config_modules=None,
        loaded_modules=None,
        executor=None,
    ):
        self.conditioning_mode = getattr(server_args, "zing_conditioning_mode", "local")
        if self.conditioning_mode not in {"local", "remote"}:
            raise ValueError("Unsupported Zing conditioning mode")
        required = list(
            self._required_config_modules
            if required_config_modules is None
            else required_config_modules
        )
        if self.conditioning_mode != "local":
            conflicts = set(loaded_modules or {}) & EXCLUDED_ENCODER_COMPONENTS
            for name in ("component_paths", "component_weights_paths"):
                conflicts.update(
                    set(getattr(server_args, name, {}) or {})
                    & EXCLUDED_ENCODER_COMPONENTS
                )
            direct = (
                getattr(server_args, "component_direct_gpu_weight_loading", {}) or {}
            )
            selected = (
                {name for name, enabled in direct.items() if enabled}
                if isinstance(direct, dict)
                else set(direct)
            )
            conflicts.update(selected & EXCLUDED_ENCODER_COMPONENTS)
            if conflicts:
                raise ValueError(
                    "External Zing conditioning conflicts with explicit encoder component loading"
                )
            required = [
                name for name in required if name not in EXCLUDED_ENCODER_COMPONENTS
            ]
        self.conditioning_loader_receipt = {
            "mode": self.conditioning_mode,
            "resolved_weight_components": [],
        }
        super().__init__(
            model_path,
            server_args,
            required_config_modules=required,
            loaded_modules=loaded_modules,
            executor=executor,
        )
        self.conditioning_loader_receipt["loaded_components"] = sorted(self.modules)

    def _resolve_component_path(self, server_args, module_name, load_module_name):
        if self.conditioning_mode != "local" and (
            module_name in EXCLUDED_ENCODER_COMPONENTS
            or load_module_name in EXCLUDED_ENCODER_COMPONENTS
        ):
            raise ValueError(
                "External Zing conditioning cannot resolve encoder weights"
            )
        result = super()._resolve_component_path(
            server_args, module_name, load_module_name
        )
        self.conditioning_loader_receipt["resolved_weight_components"].append(
            module_name
        )
        return result

    def _load_worker_config(self):
        """Resolve only worker files; a trimmed export need not contain Qwen."""
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            maybe_download_model,
        )

        subfolder = self.server_args.model_subfolder
        if subfolder is not None:
            subfolder = os.path.normpath(subfolder)
            if (
                os.path.isabs(subfolder)
                or subfolder == ".."
                or subfolder.startswith(f"..{os.sep}")
            ):
                raise ValueError(
                    "Zing model subfolder must remain inside the model repository"
                )
        prefix = "" if subfolder is None else f"{subfolder}/"
        patterns = [f"{prefix}model_index.json", f"{prefix}zing_config.json"] + [
            f"{prefix}{name}/**" for name in self._required_config_modules
        ]
        root = Path(self.model_path)
        if not root.is_dir():
            root = Path(
                maybe_download_model(
                    self.model_path,
                    allow_patterns=patterns,
                    revision=self.server_args.revision,
                    force_diffusers_model=False,
                )
            )
        self.model_path = str(root if subfolder is None else root / subfolder)
        index = json.loads((Path(self.model_path) / "model_index.json").read_text())
        if (
            index.get("_class_name") != self.pipeline_name
            or "_diffusers_version" not in index
        ):
            raise ValueError("Zing worker export has an invalid model index")
        index = {
            name: value
            for name, value in index.items()
            if name not in EXCLUDED_ENCODER_COMPONENTS
        }
        for name in self._required_config_modules:
            if name not in index or (
                index[name] is not None
                and not (Path(self.model_path) / name).is_dir()
                and name not in (getattr(self.server_args, "component_paths", {}) or {})
            ):
                raise ValueError(
                    "Zing worker export is missing a required worker component"
                )
        return index

    def _load_config(self):
        index = (
            super()._load_config()
            if self.conditioning_mode == "local"
            else self._load_worker_config()
        )
        self.zing_runtime = json.loads(
            (Path(self.model_path) / "zing_config.json").read_text()
        )
        cfg = self.zing_runtime
        version = cfg.get("format_version")
        if (
            type(version) is not int
            or version not in (1, 2)
            or cfg.get("causal") is not True
        ):
            raise ValueError(
                "Zing requires a version-1 or version-2 causal export; use export_checkpoint.py"
            )
        if version == 2:
            if (
                cfg.get("conversion_version") != "split-adaln-concat-v1"
                or cfg.get("input_adaln_layout") not in ("single", "split")
                or cfg.get("runtime_adaln_layout") != "single"
            ):
                raise ValueError(
                    "Zing version-2 export has unsupported AdaLN conversion metadata"
                )
            arch = json.loads(
                (Path(self.model_path) / "transformer/config.json").read_text()
            )
            tags = resolve_zing_modality_tags(
                arch["num_modalities"], arch.get("modality_tags")
            )
            roles = ["video", "text", "audio", "source_visual"]
            if tags["condition_visual"] != tags["video"]:
                roles.append("condition_visual")
            if cfg.get("adaln_modality_order") != sorted(roles, key=tags.__getitem__):
                raise ValueError(
                    "Zing export AdaLN order conflicts with transformer routing"
                )
            for name in ("ar_condition_timestep", "reference_visual_timestep"):
                value = cfg.get(name)
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                    or value != arch.get(name)
                ):
                    raise ValueError(
                        "Zing export condition timesteps conflict with transformer config"
                    )
        if cfg.get("local_attn_size") != -1 or cfg.get("sink_size") != 0:
            raise ValueError(
                "Zing dense runtime requires unlimited absolute-position history and sink_size=0"
            )
        # Deployment policy is independent of the checkpoint's export policy.
        encode_mode = os.environ.get("SGLANG_ZING_VISUAL_ENCODE_MODE", "performance")
        if encode_mode not in {"performance", "deterministic"}:
            raise ValueError(
                "SGLANG_ZING_VISUAL_ENCODE_MODE must be performance or deterministic"
            )
        cfg["vae_config"]["encode_mode"] = encode_mode
        return index

    def create_pipeline_stages(self, server_args):
        config = server_args.pipeline_config
        if config.video_decoder_backend == "taeh3":
            import torch
            from sglang.multimodal_gen.runtime.models.vaes.taeh3 import (
                WEIGHTS_SHA256,
                TAEH3Decoder,
            )

            dtype = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }[config.taeh3_decode_precision]
            self.modules["taeh3_decoder"] = TAEH3Decoder(
                config.taeh3_checkpoint_path, WEIGHTS_SHA256
            ).to(dtype=dtype)
        provider = (
            None
            if self.conditioning_mode == "local"
            else RemoteConditioner.from_server_args(server_args)
        )
        stage = ZingH3GenerationStage(
            self.modules,
            server_args.pipeline_config,
            self.zing_runtime,
            conditioning_mode=self.conditioning_mode,
            conditioning_provider=provider,
            local_cache_bytes=server_args.zing_conditioning_local_cache_bytes,
        )
        stage.conditioning_loader_receipt = self.conditioning_loader_receipt
        # Realtime KV slots are allocated here, before the worker is ready.
        stage.initialize_realtime_kv_cache_pool(server_args)
        self.add_stage(stage)


EntryClass = ZingH3CausalPipeline
