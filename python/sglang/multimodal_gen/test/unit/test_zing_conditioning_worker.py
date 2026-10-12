# SPDX-License-Identifier: Apache-2.0
"""Applicable pre-RPC regressions retained under the approved v6 redesign."""

import inspect
from types import SimpleNamespace

import pytest


@pytest.fixture
def pipeline_modules():
    global \
        ZingH3CausalPipeline, \
        ComposedPipelineBase, \
        ZingH3Codecs, \
        ZingH3GenerationStage
    from sglang.multimodal_gen.runtime.pipelines.zing_h3_causal_pipeline import (
        ZingH3CausalPipeline,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
        ComposedPipelineBase,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        ZingH3Codecs,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )


def args(mode, **overrides):
    return SimpleNamespace(
        zing_conditioning_mode=mode,
        conditioning_expected_profile="profile.json",
        conditioning_cache_root="cache",
        conditioning_config="admin.json",
        component_paths={},
        component_direct_gpu_weight_loading=[],
        **overrides,
    )


@pytest.mark.parametrize("mode", ["remote"])
@pytest.mark.usefixtures("pipeline_modules")
def test_external_mode_filters_before_parent_loader(monkeypatch, mode):
    captured = {}

    def parent_init(
        self,
        model_path,
        server_args,
        required_config_modules=None,
        loaded_modules=None,
        executor=None,
    ):
        captured["required"] = required_config_modules or self._required_config_modules

    monkeypatch.setattr(ComposedPipelineBase, "__init__", parent_init)
    original = list(ZingH3CausalPipeline._required_config_modules)
    ZingH3CausalPipeline("trimmed-export", args(mode))
    assert set(captured["required"]) == {"video_vae", "audio_vae", "transformer"}
    assert ZingH3CausalPipeline._required_config_modules == original


@pytest.mark.usefixtures("pipeline_modules")
def test_local_mode_preserves_required_modules(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        ComposedPipelineBase,
        "__init__",
        lambda self, *a, **kw: captured.update(
            required=kw.get("required_config_modules") or self._required_config_modules
        ),
    )
    ZingH3CausalPipeline("export", args("local"))
    assert "text_encoder" in captured["required"]


@pytest.mark.parametrize("component", ["text_encoder", "tokenizer", "processor"])
@pytest.mark.parametrize("conflict", ["injected", "override", "direct_gpu"])
@pytest.mark.usefixtures("pipeline_modules")
def test_external_mode_rejects_explicit_encoder_loading(
    monkeypatch, component, conflict
):
    monkeypatch.setattr(ComposedPipelineBase, "__init__", lambda *a, **kw: None)
    options = args("remote")
    loaded = None
    if conflict == "injected":
        loaded = {component: object()}
    elif conflict == "override":
        options.component_paths = {component: "/external/component"}
    else:
        options.component_direct_gpu_weight_loading = [component]
    with pytest.raises(ValueError, match="conditioning"):
        ZingH3CausalPipeline("export", options, loaded_modules=loaded)


@pytest.mark.usefixtures("pipeline_modules")
def test_prepare_has_external_condition_entrypoint():
    assert "conditioning" in inspect.signature(ZingH3Codecs.prepare).parameters


@pytest.mark.parametrize("mode", ["remote"])
@pytest.mark.usefixtures("pipeline_modules")
def test_external_stage_has_no_encoder_residency_declaration(mode):
    stage = object.__new__(ZingH3GenerationStage)
    stage.conditioning_mode = mode
    stage.codecs = SimpleNamespace(video_decoder_name="video_vae")
    uses = stage.component_uses(args(mode))
    assert "text_encoder" not in {use.component_name for use in uses}


@pytest.mark.usefixtures("pipeline_modules")
def test_trimmed_export_configuration_does_not_require_qwen_directories(
    tmp_path, monkeypatch
):
    import json
    import os

    components = {
        name: ["sglang", "Test"]
        for name in ZingH3CausalPipeline._required_config_modules
    }
    for name in ("video_vae", "audio_vae", "transformer"):
        (tmp_path / name).mkdir()
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "ZingH3CausalPipeline",
                "_diffusers_version": "0.37.0",
                **components,
            }
        )
    )
    (tmp_path / "zing_config.json").write_text(
        json.dumps(
            dict(
                format_version=1,
                causal=True,
                local_attn_size=-1,
                sink_size=0,
                vae_config={},
            )
        )
    )
    stage = object.__new__(ZingH3CausalPipeline)
    stage.conditioning_mode = "remote"
    stage.model_path = str(tmp_path)
    stage._required_config_modules = ["video_vae", "audio_vae", "transformer"]
    stage.server_args = SimpleNamespace(model_subfolder=None, revision=None)
    # The local-only model helper must not repair a trimmed export by fetching Qwen.
    import sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base as composed

    monkeypatch.setattr(composed, "maybe_download_model", lambda path, **kw: path)
    try:
        config = stage._load_config()
        error = None
    except ValueError as exc:
        config, error = None, exc
    assert error is None, f"Trimmed export was rejected: {error}"
    assert "text_encoder" not in config
    assert not os.path.exists(tmp_path / "text_encoder")


@pytest.mark.usefixtures("pipeline_modules")
def test_automatic_source_s0_origin_survives_parse_and_cpu_decode(tmp_path):
    import numpy as np
    from PIL import Image

    codecs = object.__new__(ZingH3Codecs)
    codecs.runtime = {"causal": True}
    codecs.conditioning_mode = "remote"
    codecs.spatial_ratio, codecs.patch_size, codecs.max_short_edge = 8, (1, 2, 2), 32
    codecs.vae_frames_per_chunk, codecs.vae_latents_per_chunk, codecs.temporal_ratio = (
        17,
        5,
        4,
    )
    codecs.remote_read_fs = None
    path = tmp_path / "s0.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(path)
    request = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "prompt"}]},
            {
                "role": "target",
                "content": [
                    {
                        "type": "video",
                        "output": {"frames": 1, "height": 16, "width": 16},
                    }
                ],
                "source": {"type": "video", "uri": str(path)},
            },
        ]
    }
    media = codecs._parse_inference_request(request)[1]
    assert media[0].get("origin") == "source_s0"
    decoded = codecs._decode_media(
        {**media[0], "presentation_fps": 24.0}, need_audio=False
    )
    assert decoded["origin"] == "source_s0"
    assert decoded["fps"] == 24.0
    assert decoded["clip"] == {"start": 0, "end": 0}
    explicit = {
        "type": "video",
        "id": "<Video 1>",
        "uri": str(path),
        "clip": {"start": 0, "end": 0},
        "presentation_fps": 24.0,
    }
    request["messages"][0]["content"].append(explicit)
    media = codecs._parse_inference_request(request)[1]
    assert len(media) == 1
    assert media[0].get("origin", "explicit") == "explicit"
