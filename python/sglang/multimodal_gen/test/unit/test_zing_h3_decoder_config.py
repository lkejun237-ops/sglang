# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
    ZingH3CausalPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.zing_h3_decoder import (
    normalize_decoder_spec,
)

LEGACY_DEFAULTS = dict(
    video_decoder_backend=None,
    taeh3_checkpoint_path=None,
    taeh3_decode_precision=None,
    vae_decode_precision="fp32",
)


def _spec(decoder_config=None, explicit=(), **legacy):
    return normalize_decoder_spec(
        decoder_config,
        **{**LEGACY_DEFAULTS, **legacy},
        explicit_legacy=frozenset(explicit),
    )


def test_legacy_fields_map_to_decoder_classes():
    taeh3 = _spec(taeh3_checkpoint_path="/m/taeh3.pth", taeh3_decode_precision="bf16")
    assert (taeh3.type, taeh3.checkpoint_path, taeh3.precision) == (
        "taeh3",
        "/m/taeh3.pth",
        "bf16",
    )
    official = _spec(video_decoder_backend="native", vae_decode_precision="fp16")
    assert (official.type, official.precision) == ("official_h3", "fp16")


def test_explicit_legacy_taeh3_still_selects_taeh3():
    with pytest.raises(ValueError, match="TAEH3 requires a local checkpoint path"):
        _spec(video_decoder_backend="taeh3", explicit={"video_decoder_backend"})
    spec = _spec(video_decoder_backend="taeh3", taeh3_checkpoint_path="/m/taeh3.pth")
    assert (spec.type, spec.precision) == ("taeh3", "fp32")


def test_default_is_lynn_and_requires_its_checkpoint():
    with pytest.raises(
        ValueError, match=r"lynn_streaming \(the default decoder\) requires"
    ):
        _spec()


def test_nested_config_defaults_to_lynn():
    spec = _spec({"checkpoint_path": "/a.pt", "model_dir": "/m"})
    assert (spec.type, spec.checkpoint_path, spec.model_dir, spec.precision) == (
        "lynn_streaming",
        "/a.pt",
        "/m",
        "bf16",
    )


def test_nested_config_fills_gaps_from_explicit_legacy_fields():
    spec = _spec(
        {"type": "taeh3"},
        explicit={"taeh3_checkpoint_path", "taeh3_decode_precision"},
        taeh3_checkpoint_path="/m/taeh3.pth",
        taeh3_decode_precision="fp16",
    )
    assert (spec.checkpoint_path, spec.precision) == ("/m/taeh3.pth", "fp16")


def test_equivalent_legacy_and_nested_values_are_accepted():
    spec = _spec(
        {"type": "official_h3", "precision": "bf16"},
        explicit={"video_decoder_backend", "vae_decode_precision"},
        video_decoder_backend="native",
        vae_decode_precision="bf16",
    )
    assert (spec.type, spec.precision) == ("official_h3", "bf16")


@pytest.mark.parametrize(
    "decoder_config,explicit,legacy",
    [
        (
            {"type": "official_h3"},
            {"video_decoder_backend"},
            {"video_decoder_backend": "taeh3"},
        ),
        (
            {"type": "taeh3", "checkpoint_path": "/a.pth"},
            {"taeh3_checkpoint_path"},
            {"taeh3_checkpoint_path": "/b.pth"},
        ),
        (
            {"type": "taeh3", "checkpoint_path": "/a.pth", "precision": "fp16"},
            {"taeh3_decode_precision"},
            {"taeh3_decode_precision": "bf16"},
        ),
        (
            {"checkpoint_path": "/a.pt", "model_dir": "/m"},
            {"taeh3_checkpoint_path"},
            {"taeh3_checkpoint_path": "/b.pth"},
        ),
        (
            {"type": "official_h3", "precision": "fp16"},
            {"vae_decode_precision"},
            {"vae_decode_precision": "bf16"},
        ),
        (
            {"type": "official_h3"},
            {"taeh3_checkpoint_path"},
            {"taeh3_checkpoint_path": "/b.pth"},
        ),
    ],
)
def test_conflicting_legacy_and_nested_fields_fail(decoder_config, explicit, legacy):
    with pytest.raises(ValueError, match="conflicts with decoder_config"):
        _spec(decoder_config, explicit=explicit, **legacy)


def test_dataclass_defaults_are_not_mistaken_for_user_choices():
    spec = _spec({"type": "official_h3"})
    assert spec.type == "official_h3"


@pytest.mark.parametrize(
    "decoder_config,message",
    [
        ({"type": "mystery"}, "Unknown Zing decoder type"),
        ({"type": "taeh3", "bogus": 1}, "Unknown decoder_config field: bogus"),
        (
            {"type": "taeh3", "checkpoint_path": "/a.pth", "precision": "int8"},
            "Unsupported TAEH3 decode precision",
        ),
        (
            {"type": "taeh3", "checkpoint_path": "/a.pth", "weights": "ema"},
            "does not support decoder_config.weights",
        ),
        ({"checkpoint_path": "/a.pt"}, "the default decoder"),
        ({"type": "official_h3", "checkpoint_path": "/a.pth"}, "not supported yet"),
        ({"type": "lynn_streaming", "checkpoint_path": "/a.pt"}, "requires"),
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
                "options": {"activation_checkpointing": True},
            },
            "activation_checkpointing off",
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
    ],
)
def test_invalid_nested_configs_are_rejected_without_fallback(decoder_config, message):
    with pytest.raises(ValueError, match=message):
        _spec(decoder_config)


def test_unknown_legacy_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown Zing video decoder backend"):
        _spec(video_decoder_backend="mystery")


def test_lynn_spec_and_redacted_form():
    spec = _spec(
        {
            "type": "lynn_streaming",
            "checkpoint_path": "/secret/a.pt",
            "model_dir": "/secret/m",
            "weights": "ema",
        }
    )
    assert spec.precision == "bf16"
    assert spec.redacted() == {
        "type": "lynn_streaming",
        "weights": "ema",
        "precision": "bf16",
        "options": {},
    }


def test_pipeline_config_tracks_explicit_legacy_fields_from_kwargs():
    config = ZingH3CausalPipelineConfig()
    args = {"video_decoder_backend": "native", "decoder_config": {"type": "taeh3"}}
    config.update_config_from_dict(args)
    with pytest.raises(ValueError, match="conflicts with decoder_config"):
        config.resolve_decoder_spec()


def test_pipeline_config_tracks_explicit_legacy_fields_from_json_dict():
    config = ZingH3CausalPipelineConfig()
    config.update_pipeline_config(
        {
            "taeh3_checkpoint_path": "/m/taeh3.pth",
            "decoder_config": {"type": "taeh3", "checkpoint_path": "/m/taeh3.pth"},
        }
    )
    assert config.resolve_decoder_spec().checkpoint_path == "/m/taeh3.pth"
    assert config.decoder_type == "taeh3"


def test_default_pipeline_config_selects_lynn_and_fails_without_its_checkpoint():
    with pytest.raises(ValueError, match="the default decoder"):
        ZingH3CausalPipelineConfig().resolve_decoder_spec()
    lynn = ZingH3CausalPipelineConfig(
        decoder_config={"checkpoint_path": "/a.pt", "model_dir": "/m"}
    )
    assert lynn.decoder_type == "lynn_streaming"


def test_pipeline_config_explicit_legacy_taeh3_still_works():
    # "taeh3" used to be the dataclass default; an explicit value must still select it.
    explicit = ZingH3CausalPipelineConfig(
        video_decoder_backend="taeh3", taeh3_checkpoint_path="/m/taeh3.pth"
    )
    assert explicit.decoder_type == "taeh3"
    assert (
        ZingH3CausalPipelineConfig(taeh3_checkpoint_path="/m/taeh3.pth").decoder_type
        == "taeh3"
    )
    from_dict = ZingH3CausalPipelineConfig()
    from_dict.update_pipeline_config(
        {"video_decoder_backend": "taeh3", "taeh3_checkpoint_path": "/m/taeh3.pth"}
    )
    assert from_dict.decoder_type == "taeh3"
    assert ZingH3CausalPipelineConfig(video_decoder_backend="native").decoder_type == (
        "official_h3"
    )
