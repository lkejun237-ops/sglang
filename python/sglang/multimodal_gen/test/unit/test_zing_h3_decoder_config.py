# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
    ZingH3CausalPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.zing_h3_decoder import (
    normalize_decoder_spec,
)


def _spec(decoder_config=None, explicit=(), **legacy):
    return normalize_decoder_spec(
        decoder_config,
        **{"video_decoder_backend": None, "vae_decode_precision": "fp32", **legacy},
        explicit_legacy=frozenset(explicit),
    )


def test_native_reference_selection_preserves_precision():
    spec = _spec(video_decoder_backend="native", vae_decode_precision="fp16")
    assert (spec.type, spec.precision) == ("official_h3", "fp16")


def test_default_is_lynn_and_requires_its_checkpoint():
    with pytest.raises(
        ValueError, match=r"lynn_streaming \(the default decoder\) requires"
    ):
        _spec()
    spec = _spec({"checkpoint_path": "/a.pt", "model_dir": "/m"})
    assert (spec.type, spec.precision, spec.checkpoint_path, spec.model_dir) == (
        "lynn_streaming",
        "bf16",
        "/a.pt",
        "/m",
    )


def test_equivalent_native_legacy_and_nested_values_are_accepted():
    spec = _spec(
        {"type": "official_h3", "precision": "bf16"},
        explicit={"video_decoder_backend", "vae_decode_precision"},
        video_decoder_backend="native",
        vae_decode_precision="bf16",
    )
    assert (spec.type, spec.precision) == ("official_h3", "bf16")


@pytest.mark.parametrize(
    "decoder_config,legacy,explicit",
    [
        (
            {"type": "lynn_streaming", "checkpoint_path": "/a.pt", "model_dir": "/m"},
            {"video_decoder_backend": "native"},
            {"video_decoder_backend"},
        ),
        (
            {"type": "official_h3", "precision": "fp16"},
            {"vae_decode_precision": "bf16"},
            {"vae_decode_precision"},
        ),
    ],
)
def test_conflicting_selections_fail(decoder_config, legacy, explicit):
    with pytest.raises(ValueError, match="conflicts with decoder_config"):
        _spec(decoder_config, explicit=explicit, **legacy)


@pytest.mark.parametrize(
    "decoder_config,message",
    [
        ({"type": "mystery"}, "Unknown Zing decoder type"),
        ({"type": "taeh3"}, "Unknown Zing decoder type"),
        ({"type": "official_h3", "bogus": 1}, "Unknown decoder_config field"),
        ({"type": "official_h3", "checkpoint_path": "/a.pth"}, "not supported yet"),
        ({"type": "official_h3", "precision": "int8"}, "Unsupported official_h3"),
        ({"checkpoint_path": "/a.pt"}, "the default decoder"),
        (
            {
                "type": "lynn_streaming",
                "checkpoint_path": "/a.pt",
                "model_dir": "/m",
                "weights": "swa",
            },
            "Unsupported lynn_streaming weights",
        ),
        (
            {
                "type": "lynn_streaming",
                "checkpoint_path": "/a.pt",
                "model_dir": "/m",
                "precision": "fp16",
            },
            "Unsupported lynn_streaming precision",
        ),
        (
            {
                "type": "lynn_streaming",
                "checkpoint_path": "/a.pt",
                "model_dir": "/m",
                "options": {"activation_checkpointing": True},
            },
            "activation_checkpointing off",
        ),
    ],
)
def test_invalid_configs_rejected_without_fallback(decoder_config, message):
    with pytest.raises(ValueError, match=message):
        _spec(decoder_config)


@pytest.mark.parametrize("backend", ["mystery", "taeh3"])
def test_removed_or_unknown_legacy_backend_is_rejected(backend):
    with pytest.raises(ValueError, match="Unknown Zing video decoder backend"):
        _spec(video_decoder_backend=backend)


def test_lynn_spec_redacts_filesystem_paths():
    spec = _spec(
        {
            "type": "lynn_streaming",
            "checkpoint_path": "/secret/a.pt",
            "model_dir": "/secret/m",
            "weights": "ema",
        }
    )
    assert spec.redacted() == {
        "type": "lynn_streaming",
        "weights": "ema",
        "precision": "bf16",
        "options": {},
    }


def test_pipeline_tracks_explicit_legacy_fields_from_kwargs():
    config = ZingH3CausalPipelineConfig()
    config.update_config_from_dict(
        {
            "video_decoder_backend": "native",
            "decoder_config": {"type": "lynn_streaming"},
        }
    )
    with pytest.raises(ValueError, match="conflicts with decoder_config"):
        config.resolve_decoder_spec()


def test_pipeline_json_selects_lynn_without_legacy_fields():
    config = ZingH3CausalPipelineConfig()
    config.update_pipeline_config(
        {"decoder_config": {"checkpoint_path": "/m/lynn.pt", "model_dir": "/m"}}
    )
    assert config.resolve_decoder_spec().checkpoint_path == "/m/lynn.pt"
    assert config.decoder_type == "lynn_streaming"
