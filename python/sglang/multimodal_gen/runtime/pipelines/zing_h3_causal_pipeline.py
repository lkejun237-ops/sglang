# SPDX-License-Identifier: Apache-2.0
"""Native Zing1/H3 causal inference with shared H3 codecs."""

import json
import os
from pathlib import Path

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
    ZingH3CausalPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
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

    def _load_config(self):
        index = super()._load_config()
        self.zing_runtime = json.loads(
            (Path(self.model_path) / "zing_config.json").read_text()
        )
        cfg = self.zing_runtime
        if cfg.get("format_version") != 1 or cfg.get("causal") is not True:
            raise ValueError(
                "Zing requires a version-1 causal export; use export_checkpoint.py"
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
        stage = ZingH3GenerationStage(
            self.modules, server_args.pipeline_config, self.zing_runtime
        )
        # Realtime KV slots are allocated here, before the worker is ready.
        stage.initialize_realtime_kv_cache_pool(server_args)
        self.add_stage(stage)


EntryClass = ZingH3CausalPipeline
