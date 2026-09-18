# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    FastH3PipelineConfig,
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
    ZingH3CausalPipelineConfig,
)
from sglang.multimodal_gen.registry import (
    get_model_info,
    get_pipeline_class,
    get_registered_pipeline_names,
)
from sglang.multimodal_gen.runtime.pipelines.zing_h3_causal_pipeline import (
    ZingH3CausalPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)


@pytest.mark.parametrize(
    "directory", ["zing-export", "MiniMax-H3-zing-export", "MiniMax-H3"]
)
def test_explicit_zing_model_index_resolves_native_pipeline(tmp_path, directory):
    model_path = tmp_path / directory
    model_path.mkdir()
    (model_path / "transformer").mkdir()
    (model_path / "vae").mkdir()
    (model_path / "model_index.json").write_text(
        json.dumps(
            {"_class_name": "ZingH3CausalPipeline", "_diffusers_version": "0.37.0"}
        )
    )
    info = get_model_info(str(model_path), backend="sglang")
    assert info.pipeline_cls is ZingH3CausalPipeline
    assert info.pipeline_config_cls is ZingH3CausalPipelineConfig
    assert "ZingH3CausalPipeline" in get_registered_pipeline_names()
    assert get_pipeline_class("ZingH3CausalPipeline") is ZingH3CausalPipeline


def test_zing_keeps_h3_and_fasth3_resolution_distinct():
    h3 = get_model_info("MiniMaxAI/MiniMax-H3")
    fast = get_model_info("FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree")
    assert h3.pipeline_config_cls is MiniMaxH3PipelineConfig
    assert fast.pipeline_config_cls is FastH3PipelineConfig
    assert h3.pipeline_cls is not ZingH3CausalPipeline
    assert fast.pipeline_cls is not ZingH3CausalPipeline


def test_zing_declares_four_modalities_and_reuses_codec_configs():
    zing, h3 = ZingH3CausalPipelineConfig(), MiniMaxH3PipelineConfig()
    arch = zing.dit_config.arch_config
    assert arch.num_modalities == 4
    assert arch.adaln_out_features == 4 * 6 * arch.hidden_size == 129024
    assert h3.dit_config.arch_config.num_modalities == 3
    assert h3.dit_config.arch_config.adaln_out_features == 96768
    assert (
        arch.final_adaln_out_features
        == h3.dit_config.arch_config.final_adaln_out_features
    )
    assert type(zing.vae_config) is type(h3.vae_config)
    assert type(zing.audio_vae_config) is type(h3.audio_vae_config)
    assert type(zing.text_encoder_configs[0]) is type(h3.text_encoder_configs[0])
    assert zing.should_use_guidance
    assert not zing.requires_audio_output
    assert zing.get_model_deployment_config().dit_layerwise_offload_modes == ("memory",)
    files = ["diffusion_pytorch_model-00001-of-00003.safetensors"]
    assert zing.select_vae_weight_files(
        files, "/model/video_vae", "video_vae", "fp32"
    ) == ["/model/video_vae/source/model.safetensors"]


def test_zing_registers_executable_stages_and_rejects_unsupported_parallelism():
    from types import SimpleNamespace

    from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )

    assert issubclass(ZingH3CausalPipeline, ComposedPipelineBase)
    assert ZingH3CausalPipeline.sampling_params_cls is ZingH3SamplingParams
    assert callable(ZingH3GenerationStage.forward)
    with pytest.raises(ValueError, match="TP=1 and SP=1"):
        ZingH3CausalPipelineConfig().validate_server_args(SimpleNamespace(tp_size=2))


def test_zing_pins_output_fps_to_checkpoint_time_base():
    from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams

    with pytest.raises(TypeError):
        ZingH3SamplingParams(prompt="clip", fps=30)
    params = ZingH3SamplingParams(prompt="clip")
    assert params.fps == 24
    # CLI/API paths assign fields after construction; validation re-pins it.
    params.fps = 30
    params.check_sampling_param()
    assert params.fps == 24


@pytest.mark.parametrize(
    "flag", ["enable_frame_interpolation", "enable_upscaling", "enable_teacache"]
)
def test_zing_rejects_post_processing_switches(flag):
    from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams

    params = ZingH3SamplingParams(prompt="clip")
    params.check_sampling_param()
    setattr(params, flag, True)
    with pytest.raises(ValueError, match=flag):
        params.check_sampling_param()
    with pytest.raises(ValueError, match=flag):
        ZingH3SamplingParams(prompt="clip", **{flag: True})


def test_zing_warmup_keeps_euler_endpoint():
    from types import SimpleNamespace

    from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams

    params = ZingH3SamplingParams(prompt="warmup")
    params.num_inference_steps = 1
    req = SimpleNamespace(extra={})
    params.prepare_synthetic_warmup_request_for_queue(req, None)
    assert params.num_inference_steps == 2
    assert req.extra["zing_solver"] == "euler"
    assert req.extra["zing_return_latents"] is False
