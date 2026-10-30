# SPDX-License-Identifier: Apache-2.0
"""CPU conversion oracle: exact weights; FP32 GEMM atol=2e-6, rtol=1e-5."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

spec = importlib.util.spec_from_file_location(
    "zing_export",
    Path(__file__).resolve().parents[4]
    / "benchmark/zing_h3_causal/export_checkpoint.py",
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def fixture(layout="split", tags=None):
    arch = dict(
        dim=8,
        num_heads=2,
        head_dim=4,
        ffn_dim=16,
        in_dim=2,
        patch_size=[1, 2, 2],
        audio_in_dim=2,
        text_dim=8,
        freq_dim=4,
        time_embed_hidden_dim=8,
        time_embed_dim=4,
        num_layers=2,
        num_refiner_layers=1,
        num_modalities=4,
        modality_tags=dict(zip(exporter.ADALN_MODALITIES, range(4))),
    )
    if tags is not None:
        arch["modality_tags"] = tags
        arch.pop("num_modalities")
    torch.manual_seed(205)
    state = {
        key: torch.randn(shape)
        for key, shape in exporter.expected_shapes(arch, layout).items()
    }
    return arch, state


def test_split_projection_and_independent_source():
    arch, state = fixture()
    converted, layout = exporter.normalize_adaln_checkpoint(state, arch)
    assert layout == "split"
    x = torch.randn(3, arch["time_embed_dim"])
    for i in range(arch["num_layers"]):
        prefix = f"backbone.transformer_blocks.{i}.adaln_proj."
        expected = torch.cat(
            [
                torch.nn.functional.linear(
                    x,
                    state[prefix + m + "_linear.weight"],
                    state[prefix + m + "_linear.bias"],
                )
                for m in exporter.ADALN_MODALITIES
            ],
            -1,
        )
        actual = torch.nn.functional.linear(
            x, converted[prefix + "linear.weight"], converted[prefix + "linear.bias"]
        )
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        for j, m in enumerate(exporter.ADALN_MODALITIES):
            assert torch.equal(
                converted[prefix + "linear.weight"][j * 48 : (j + 1) * 48],
                state[prefix + m + "_linear.weight"],
            )
        assert not torch.equal(
            converted[prefix + "linear.weight"][:48],
            converted[prefix + "linear.weight"][-48:],
        )


def test_legacy_weights_are_unchanged():
    arch, state = fixture("single")
    converted, layout = exporter.normalize_adaln_checkpoint(state, arch)
    assert layout == "single"
    assert all(converted[k] is v for k, v in state.items())


@pytest.mark.parametrize(
    "error", ["missing", "mixed", "shape", "tags", "official", "lora", "dtype"]
)
def test_invalid_checkpoint_rejected(error):
    arch, state = fixture()
    prefix = "backbone.transformer_blocks.0.adaln_proj."
    if error == "missing":
        del state[prefix + "source_visual_linear.bias"]
    elif error == "mixed":
        state[prefix + "linear.weight"] = torch.zeros(192, 4)
    elif error == "shape":
        state[prefix + "text_linear.weight"] = torch.zeros(47, 4)
    elif error == "tags":
        arch["modality_tags"]["source_visual"] = 0
    elif error == "official":
        arch["num_modalities"] = 3
    elif error == "lora":
        state[prefix + "source_visual_linear.lora_A.weight"] = torch.randn(3, 4)
    else:
        state[prefix + "text_linear.weight"] = state[
            prefix + "text_linear.weight"
        ].double()
    with pytest.raises(ValueError):
        exporter.normalize_adaln_checkpoint(state, arch)


def test_unequal_rank_native_merged_branches_preserved():
    arch, state = fixture()
    prefix = "backbone.transformer_blocks.0.adaln_proj."
    for m, rank in zip(exporter.ADALN_MODALITIES, [2, 3, 4, 7]):
        state[prefix + m + "_linear.weight"] += torch.randn(48, rank) @ torch.randn(
            rank, 4
        )
    converted, _ = exporter.normalize_adaln_checkpoint(state, arch)
    for i, m in enumerate(exporter.ADALN_MODALITIES):
        assert torch.equal(
            converted[prefix + "linear.weight"][i * 48 : (i + 1) * 48],
            state[prefix + m + "_linear.weight"],
        )


@pytest.mark.parametrize("layout", ["single", "split"])
@pytest.mark.parametrize(
    "tags",
    [
        None,
        dict(video=2, text=0, audio=3, source_visual=1, condition_visual=2),
        dict(video=4, text=0, audio=3, source_visual=1, condition_visual=2),
    ],
)
def test_split_export_requires_and_preserves_provenance(tmp_path, layout, tags):
    arch, state = fixture(layout, tags)
    arch.update(
        eps=1e-6,
        qk_norm_eps=1e-6,
        final_norm_eps=1e-6,
        rope_config=dict(theta=10000, freq_dim=1),
        local_attn_size=-1,
        sink_size=0,
    )
    cfg = dict(
        generator_config=arch,
        causal=True,
        num_frame_per_block=[3, 2],
        split_first_latent=False,
        num_train_timestep=1000,
        timestep_shift=5,
        audio_timestep_shift=3,
        h3_media_config={},
        condition_encoder_config=dict(output_hidden_layer=50),
        vae_config=dict(encode_mode="deterministic"),
        lora_config=dict(enabled=True),
        ar_condition_timestep=0.875,
    )
    config = tmp_path / "effective.yaml"
    config.write_text(yaml.safe_dump(cfg))
    checkpoint = tmp_path / "model.pt"
    torch.save(state, checkpoint)
    components = tmp_path / "components"
    for name in [
        "text_encoder",
        "tokenizer",
        "processor",
        "audio_vae",
        "video_vae/source",
    ]:
        (components / name).mkdir(parents=True)
    (components / "video_vae/source/model.safetensors").write_bytes(b"fixture")
    destination = tmp_path / "export"
    for provenance in [
        None,
        dict(
            native_sha="a" * 40,
            checkpoint_source="ema",
            component_versions={"h3": "fixture"},
        ),
    ]:
        with pytest.raises(ValueError, match="provenance"):
            exporter.export(
                checkpoint, config, components, destination, provenance=provenance
            )
        assert not destination.exists()
    provenance = dict(
        native_sha="a" * 40,
        checkpoint_source="generator",
        component_versions={"h3": "fixture"},
        lora_merged=True,
        original_config_sha256=exporter.file_sha256(config),
    )
    original_config = config.read_bytes()
    for override in [{"split_first_latent": True}, {"num_frame_per_block": [1, 2, 2]}]:
        config.write_text(yaml.safe_dump(dict(cfg, **override)))
        with pytest.raises(ValueError, match="original effective training config"):
            exporter.export(
                checkpoint, config, components, destination, provenance=provenance
            )
        assert not destination.exists()
    config.write_bytes(original_config)
    exporter.export(
        checkpoint,
        config,
        components,
        destination,
        provenance=provenance,
        shard_bytes=1024,
    )
    runtime = json.loads((destination / "zing_config.json").read_text())
    assert runtime["format_version"] == 2
    assert runtime["input_adaln_layout"] == layout
    assert runtime["runtime_adaln_layout"] == "single"
    order = exporter.adaln_modalities(arch)
    assert runtime["adaln_modality_order"] == list(order)
    model_config = json.loads((destination / "transformer/config.json").read_text())
    assert model_config["modality_tags"] == arch["modality_tags"]
    assert model_config["num_modalities"] == len(order)
    assert model_config["adaln_out_features"] == 6 * len(order) * arch["dim"]
    assert model_config["final_adaln_out_features"] == 2 * arch["dim"]
    assert (
        runtime["ar_condition_timestep"]
        == model_config["ar_condition_timestep"]
        == 0.875
    )
    assert (
        runtime["reference_visual_timestep"]
        == model_config["reference_visual_timestep"]
        == (0.999 if layout == "split" else 1.0)
    )
    assert runtime["checkpoint_sha256"] == exporter.file_sha256(checkpoint)
    assert runtime["training_effective_config"] == cfg
    assert runtime["num_frame_per_block"] == [3, 2]
    assert runtime["split_first_latent"] is False
    assert runtime["training_config_sha256"] == provenance["original_config_sha256"]
    assert runtime["provenance"] == provenance
    weights = {}
    for path in (destination / "transformer").glob("*.safetensors"):
        weights.update(load_file(path))
    converted, _ = exporter.normalize_adaln_checkpoint(state, arch)
    for key in converted:
        assert torch.equal(converted[key].to(weights[key]), weights[key])


def test_five_branch_checkpoint_retains_independent_condition_visual():
    arch, _ = fixture()
    arch.pop("num_modalities")  # Native effective config derives it from tags.
    arch["modality_tags"]["condition_visual"] = 4
    state = {
        key: torch.randn(shape)
        for key, shape in exporter.expected_shapes(arch, "split").items()
    }
    converted, layout = exporter.normalize_adaln_checkpoint(state, arch)
    assert layout == "split"
    for layer in range(arch["num_layers"]):
        prefix = f"backbone.transformer_blocks.{layer}.adaln_proj."
        for suffix in ("weight", "bias"):
            for index, modality in enumerate(exporter.adaln_modalities(arch)):
                assert torch.equal(
                    converted[prefix + "linear." + suffix][
                        index * 48 : (index + 1) * 48
                    ],
                    state[prefix + modality + "_linear." + suffix],
                )
    arch["num_modalities"] = 4
    with pytest.raises(ValueError, match="count conflicts"):
        exporter.normalize_adaln_checkpoint(state, arch)


def test_training_metadata_redacts_authentication_without_mutating_input():
    config = {
        "wandb_config": {"key": "test-credential", "project": "test-project"},
        "api_key": "test-api-credential",
        "nested": [{"password": "test-password", "dim": 8}],
        "generator_config": {"patch_size": [1, 2, 2]},
    }
    clean = exporter.redact_config_secrets(config)
    assert clean["wandb_config"] == {"key": "[REDACTED]", "project": "test-project"}
    assert clean["api_key"] == "[REDACTED]"
    assert clean["nested"] == [{"password": "[REDACTED]", "dim": 8}]
    assert clean["generator_config"] == config["generator_config"]
    assert config["wandb_config"]["key"] == "test-credential"


@pytest.mark.parametrize(
    "tags",
    [
        dict(video=2, text=0, audio=3, source_visual=1, condition_visual=2),
        dict(video=4, text=0, audio=3, source_visual=1, condition_visual=2),
    ],
)
def test_permuted_split_weights_and_biases_follow_native_tag_order(tags):
    arch, state = fixture(tags=tags)
    converted, _ = exporter.normalize_adaln_checkpoint(state, arch)
    order = tuple(
        sorted(
            (
                name
                for name in tags
                if name != "condition_visual" or tags[name] != tags["video"]
            ),
            key=tags.__getitem__,
        )
    )
    x = torch.randn(3, arch["time_embed_dim"])
    for layer in range(arch["num_layers"]):
        prefix = f"backbone.transformer_blocks.{layer}.adaln_proj."
        for suffix in ("weight", "bias"):
            for index, role in enumerate(order):
                assert torch.equal(
                    converted[prefix + "linear." + suffix][
                        index * 48 : (index + 1) * 48
                    ],
                    state[prefix + role + "_linear." + suffix],
                )
        expected = torch.cat(
            [
                torch.nn.functional.linear(
                    x,
                    state[prefix + role + "_linear.weight"],
                    state[prefix + role + "_linear.bias"],
                )
                for role in order
            ],
            dim=-1,
        )
        actual = torch.nn.functional.linear(
            x, converted[prefix + "linear.weight"], converted[prefix + "linear.bias"]
        )
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    for key in state:
        if ".adaln_proj." not in key:
            assert converted[key] is state[key]


@pytest.mark.parametrize(
    "tags",
    [
        {},
        dict(video=0, text=1, audio=2, source_visual=3, condition_visual=1),
        dict(video=0, text=1, audio=2, source_visual=5),
        dict(video=False, text=1, audio=2, source_visual=3),
        dict(video=0, text=1, audio=2, source_visual=3, extra=4),
        dict(video=0, text=1, audio=2, source_visual=3, condition_visual=-1),
    ],
)
def test_invalid_native_routing_is_rejected(tags):
    arch, state = fixture()
    arch["modality_tags"] = tags
    with pytest.raises(ValueError):
        exporter.normalize_adaln_checkpoint(state, arch)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.9"])
@pytest.mark.parametrize("name", ["ar_condition_timestep", "reference_visual_timestep"])
def test_invalid_condition_timesteps_rejected(value, name):
    with pytest.raises(ValueError, match="condition timesteps"):
        exporter.condition_timesteps({name: value}, "split")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
