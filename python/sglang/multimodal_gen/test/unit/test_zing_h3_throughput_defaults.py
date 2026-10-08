# SPDX-License-Identifier: Apache-2.0
"""Deployment policy must override export policy without rewriting checkpoints."""

import json
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.pipelines.zing_h3_causal_pipeline import (
    ZingH3CausalPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)


@pytest.mark.parametrize("export_mode", ["deterministic", "performance"])
@pytest.mark.parametrize("override", [None, "performance", "deterministic"])
def test_visual_deployment_policy(tmp_path, monkeypatch, export_mode, override):
    checkpoint = tmp_path / "zing_config.json"
    checkpoint.write_text(
        json.dumps(
            dict(
                format_version=1,
                causal=True,
                local_attn_size=-1,
                sink_size=0,
                vae_config=dict(encode_mode=export_mode, encode_dtype="bfloat16"),
            )
        )
    )
    original = checkpoint.read_bytes()
    monkeypatch.delenv("SGLANG_ZING_VISUAL_ENCODE_MODE", raising=False)
    if override is not None:
        monkeypatch.setenv("SGLANG_ZING_VISUAL_ENCODE_MODE", override)
    index = {"sentinel": True}
    monkeypatch.setattr(ComposedPipelineBase, "_load_config", lambda self: index)
    pipeline = SimpleNamespace(model_path=tmp_path)
    assert ZingH3CausalPipeline._load_config(pipeline) is index
    assert pipeline.zing_runtime["vae_config"] == dict(
        encode_mode=override or "performance", encode_dtype="bfloat16"
    )
    assert checkpoint.read_bytes() == original


def test_invalid_visual_deployment_policy(tmp_path, monkeypatch):
    (tmp_path / "zing_config.json").write_text(
        json.dumps(
            dict(
                format_version=1,
                causal=True,
                local_attn_size=-1,
                sink_size=0,
                vae_config=dict(encode_mode="deterministic"),
            )
        )
    )
    monkeypatch.setenv("SGLANG_ZING_VISUAL_ENCODE_MODE", "fast")
    monkeypatch.setattr(ComposedPipelineBase, "_load_config", lambda self: {})
    with pytest.raises(ValueError, match="SGLANG_ZING_VISUAL_ENCODE_MODE"):
        ZingH3CausalPipeline._load_config(SimpleNamespace(model_path=tmp_path))
