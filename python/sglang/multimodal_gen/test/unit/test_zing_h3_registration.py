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
    assert zing.prompt_switch_prefix_latent_frames is None
    assert zing.get_model_deployment_config().dit_layerwise_offload_modes == ("memory",)
    files = ["diffusion_pytorch_model-00001-of-00003.safetensors"]
    assert zing.select_vae_weight_files(
        files, "/model/video_vae", "video_vae", "fp32"
    ) == ["/model/video_vae/source/model.safetensors"]


def test_zing_registers_executable_stages_and_accepts_ulysses_sp4(monkeypatch):
    from types import SimpleNamespace

    from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )

    assert issubclass(ZingH3CausalPipeline, ComposedPipelineBase)
    assert ZingH3CausalPipeline.sampling_params_cls is ZingH3SamplingParams
    assert callable(ZingH3GenerationStage.forward)
    with pytest.raises(ValueError, match="requires TP=1"):
        ZingH3CausalPipelineConfig(
            taeh3_checkpoint_path="/model/taeh3.pth"
        ).validate_server_args(SimpleNamespace(tp_size=2))

    with pytest.raises(ValueError, match="only supports 5"):
        ZingH3CausalPipelineConfig(
            prompt_switch_prefix_latent_frames=4
        ).validate_server_args(SimpleNamespace())
    args = SimpleNamespace(
        tp_size=1,
        sp_degree=4,
        ulysses_degree=4,
        ring_degree=1,
        enable_cfg_parallel=False,
        enable_torch_compile=False,
        use_fsdp_inference=False,
        attention_backend=None,
        resolve_component_attention_backend=lambda _name: (None, None),
    )
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_ATTENTION", "fa4")
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", "1")
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN", "1")
    ZingH3CausalPipelineConfig(
        taeh3_checkpoint_path="/model/taeh3.pth",
        prompt_switch_prefix_latent_frames=5,
    ).validate_server_args(args)
    args.sp_degree = 2
    with pytest.raises(ValueError, match="sp-degree == --ulysses-degree"):
        ZingH3CausalPipelineConfig(
            taeh3_checkpoint_path="/model/taeh3.pth"
        ).validate_server_args(args)


def test_adaln_table_is_limited_to_sp1():
    from types import SimpleNamespace

    def args(sp):
        return SimpleNamespace(
            sp_degree=sp,
            ulysses_degree=sp,
            attention_backend=None,
            resolve_component_attention_backend=lambda _name: (None, None),
        )

    config = ZingH3CausalPipelineConfig(
        taeh3_checkpoint_path="/model/taeh3.pth", zing_adaln_table_path="/table"
    )
    config.validate_server_args(args(1))
    with pytest.raises(ValueError, match="SP1 only"):
        config.validate_server_args(args(2))


@pytest.mark.parametrize("fps", [24, 30])
def test_zing_preserves_fps_and_validates_against_export(fps):
    from types import SimpleNamespace

    from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )

    params = ZingH3SamplingParams(prompt="clip", fps=fps)
    params.check_sampling_param()
    assert params.fps == fps
    stage = object.__new__(ZingH3GenerationStage)
    stage.codecs = SimpleNamespace(target_fps=float(fps))
    stage._validate_sampling_params(params)
    params.fps = fps + 1
    with pytest.raises(ValueError, match="requested fps"):
        stage._validate_sampling_params(params)
    assert params.fps == fps + 1  # Reject a mismatch; never silently rewrite it.


@pytest.mark.parametrize("sp", [1, 2, 4])
@pytest.mark.parametrize("attention", [None, "torch_sdpa", "fa4"])
@pytest.mark.parametrize("stable", [None, "0", "1"])
@pytest.mark.parametrize("compiled", [None, "0", "1"])
def test_zing_sp_requires_validated_numerical_configuration(
    monkeypatch, sp, attention, stable, compiled
):
    from types import SimpleNamespace

    for name, value in (
        ("SGLANG_ZING_EXPERIMENTAL_ATTENTION", attention),
        ("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", stable),
        ("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN", compiled),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    args = SimpleNamespace(
        sp_degree=sp,
        ulysses_degree=sp,
        attention_backend=None,
        resolve_component_attention_backend=lambda _name: (None, None),
    )
    config = ZingH3CausalPipelineConfig(taeh3_checkpoint_path="/model/taeh3.pth")
    effective = (attention or "fa4", stable or "1", compiled or "1")
    if sp == 1 or effective == ("fa4", "1", "1"):
        config.validate_server_args(args)
    else:
        with pytest.raises(ValueError, match="numerically validated configuration"):
            config.validate_server_args(args)


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


def test_zing_defaults_to_lynn_and_requires_explicit_weights():
    from types import SimpleNamespace

    from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams

    args = SimpleNamespace(
        attention_backend=None,
        resolve_component_attention_backend=lambda _name: (None, None),
    )
    config = ZingH3CausalPipelineConfig()
    assert config.video_decoder_backend is None
    assert ZingH3SamplingParams().zing_realtime_decode_mode == "stable"
    with pytest.raises(ValueError, match="lynn_streaming \\(the default decoder\\)"):
        config.validate_server_args(args)
    config.decoder_config = {
        "checkpoint_path": "/model/lynn.pt",
        "model_dir": "/model/lynn",
    }
    config.validate_server_args(args)
    assert config.decoder_type == "lynn_streaming"

    taeh3 = ZingH3CausalPipelineConfig(video_decoder_backend="taeh3")
    with pytest.raises(ValueError, match="TAEH3 requires a local checkpoint path"):
        taeh3.validate_server_args(args)
    taeh3.taeh3_checkpoint_path = "/model/taeh3.pth"
    taeh3.validate_server_args(args)
    taeh3.taeh3_decode_precision = "invalid"
    with pytest.raises(ValueError, match="Unsupported TAEH3 decode precision"):
        taeh3.validate_server_args(args)
    ZingH3CausalPipelineConfig(video_decoder_backend="native").validate_server_args(
        args
    )


@pytest.mark.parametrize(
    "version,tags",
    [
        (1, None),
        (2, None),
        (2, dict(video=2, text=0, audio=3, source_visual=1, condition_visual=2)),
        (2, dict(video=4, text=0, audio=3, source_visual=1, condition_visual=2)),
    ],
)
def test_causal_export_versions_validate_routing_and_timesteps(
    tmp_path, monkeypatch, version, tags
):
    cfg = dict(
        format_version=version,
        causal=True,
        local_attn_size=-1,
        sink_size=0,
        vae_config={},
    )
    roles = ["video", "text", "audio", "source_visual"]
    count = 4
    if tags is not None and tags["condition_visual"] != tags["video"]:
        roles.append("condition_visual")
        count = 5
    arch = dict(
        num_modalities=count,
        modality_tags=tags,
        ar_condition_timestep=0.875,
        reference_visual_timestep=0.999,
    )
    if version == 2:
        cfg.update(
            conversion_version="split-adaln-concat-v1",
            input_adaln_layout="split",
            runtime_adaln_layout="single",
            adaln_modality_order=sorted(roles, key=tags.__getitem__) if tags else roles,
            ar_condition_timestep=0.875,
            reference_visual_timestep=0.999,
        )
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer/config.json").write_text(json.dumps(arch))
    monkeypatch.setattr(
        ComposedPipelineBase, "_load_config", lambda self: {"index": True}
    )
    pipeline = object.__new__(ZingH3CausalPipeline)
    pipeline.model_path = str(tmp_path)
    pipeline.conditioning_mode = "local"

    def load(config):
        (tmp_path / "zing_config.json").write_text(json.dumps(config))
        return pipeline._load_config()

    assert load(cfg) == {"index": True}
    assert pipeline.zing_runtime["vae_config"]["encode_mode"] == "performance"
    with pytest.raises(ValueError, match="causal export"):
        load(dict(cfg, format_version=True))
    if version == 2:
        for key, value in [
            ("conversion_version", "other"),
            ("input_adaln_layout", "mixed"),
            ("runtime_adaln_layout", "split"),
            ("adaln_modality_order", ["video"]),
            ("ar_condition_timestep", 1),
            ("reference_visual_timestep", float("nan")),
        ]:
            with pytest.raises(ValueError):
                load(dict(cfg, **{key: value}))
