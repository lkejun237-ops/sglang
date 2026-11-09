"""Real-file preparation regressions migrated from the retired regional launcher."""

import json

import pytest
from config import render
from prepare_assets import digest, prepare, verify_prepared
from test_release import bundle


@pytest.fixture
def serving_export(tmp_path):
    candidate = bundle()
    source = tmp_path / "source"
    source.mkdir()
    (source / "zing_config.json").write_text(
        json.dumps(
            {
                "causal": True,
                "h3_media_config": {"fps": 24},
                "num_frame_per_block": [3, 2],
                "split_first_latent": False,
                "checkpoint_sha256": candidate["policy"]["model"]["checkpoint_sha256"],
            }
        )
    )
    (source / "model_index.json").write_text("{}")
    (source / "zing_realtime_profile.json").write_text('{"original":true}')
    for name in (
        "transformer",
        "text_encoder",
        "processor",
        "tokenizer",
        "video_vae",
        "audio_vae",
    ):
        (source / name).mkdir()
    (source / "transformer/config.json").write_text('{"model":"fixture"}')
    decoder = tmp_path / "decoder.pth"
    decoder.write_bytes(b"fixture decoder")
    architecture = tmp_path / "lynn"
    architecture.mkdir()
    (architecture / "config.json").write_text(
        json.dumps({"clip_length": 17, "token_drop": 3})
    )
    return dict(
        source=source,
        decoder=decoder,
        decoder_model_dir=architecture,
        output=tmp_path / "prepared",
        bundle=candidate,
        config_sha=digest(source / "zing_config.json"),
        decoder_sha=digest(decoder),
    )


def test_prepare_preserves_source_and_freezes_shared_policy(serving_export):
    receipt = prepare(**serving_export)
    root = serving_export["output"]
    policy = serving_export["bundle"]["policy"]["model"]
    profile = json.loads((root / "model/zing_realtime_profile.json").read_text())
    pipeline = json.loads((root / "pipeline.json").read_text())
    assert json.loads(
        (serving_export["source"] / "zing_realtime_profile.json").read_text()
    ) == {"original": True}
    assert profile["denoising_step_list"] == policy["denoising_steps"]
    assert profile["initial_block_count"] == 3
    assert profile["source_attention_window"] == {"window_chunks": 6, "sink_chunks": 0}
    assert pipeline["zing_kv_pool_window_chunks"] == 6
    assert pipeline["zing_kv_pool_sink_chunks"] == 0
    assert (
        profile["window_eviction_timing"]
        == pipeline["zing_kv_pool_window_eviction_timing"]
        == "after_step0"
    )
    assert pipeline["zing_adaln_mode"] == "resident"
    assert profile["source_encoder"]["remote"]["cross_dispatch_prefetch"] is True
    assert verify_prepared(root, serving_export["bundle"]) == receipt
    with pytest.raises(ValueError, match="already exists"):
        prepare(**serving_export)


@pytest.fixture
def lynn_export(serving_export, tmp_path):
    architecture = serving_export["decoder_model_dir"]
    table = tmp_path / "adaln"
    table.mkdir()
    (table / "table.safetensors").write_bytes(b"fixture table")
    (table / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "zing-adaln-table-v1",
                "identity": {
                    "config_sha256": digest(
                        serving_export["source"] / "transformer/config.json"
                    )
                },
                "provenance": json.loads(
                    (serving_export["source"] / "zing_config.json").read_text()
                ),
                "table_sha256": digest(table / "table.safetensors"),
            }
        )
    )
    candidate = serving_export["bundle"]
    candidate["policy"]["model"].update(adaln_mode="table", initial_blocks=0)
    candidate = render(candidate["policy"], candidate["bindings"])
    return dict(
        serving_export,
        bundle=candidate,
        decoder_type="lynn_streaming",
        decoder_model_dir=architecture,
        adaln_table=table,
    )


def test_lynn_and_table_assets_are_bound_to_the_receipt(lynn_export):
    receipt = prepare(**lynn_export)
    root = lynn_export["output"]
    pipeline = json.loads((root / "pipeline.json").read_text())
    assert pipeline["decoder_config"] == {
        "type": "lynn_streaming",
        "precision": "bf16",
        "checkpoint_path": str(lynn_export["decoder"]),
        "model_dir": str(lynn_export["decoder_model_dir"]),
    }
    assert len(receipt["asset_checks"]) == 4
    assert (
        json.loads((root / "model/zing_realtime_profile.json").read_text())[
            "async_source_output"
        ]
        is True
    )
    assert verify_prepared(root, lynn_export["bundle"]) == receipt
    (lynn_export["decoder_model_dir"] / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="asset identities changed"):
        verify_prepared(root, lynn_export["bundle"])


@pytest.mark.parametrize(
    "mutation", ["mixed", "table", "identity", "architecture", "policy"]
)
def test_invalid_lynn_assets_fail_before_writes(lynn_export, mutation):
    if mutation in ("mixed", "policy"):
        candidate = lynn_export["bundle"]
        candidate["policy"]["model"].update(
            {"initial_blocks": 3} if mutation == "mixed" else {"adaln_mode": "resident"}
        )
        lynn_export["bundle"] = render(candidate["policy"], candidate["bindings"])
    elif mutation == "table":
        (lynn_export["adaln_table"] / "table.safetensors").write_bytes(b"changed")
    elif mutation == "identity":
        path = lynn_export["adaln_table"] / "manifest.json"
        data = json.loads(path.read_text())
        data["identity"]["config_sha256"] = "0" * 64
        path.write_text(json.dumps(data))
    else:
        (lynn_export["decoder_model_dir"] / "config.json").write_text("{}")
    with pytest.raises(ValueError):
        prepare(**lynn_export)
    assert not lynn_export["output"].exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "config",
        "decoder",
        "incomplete",
        "nested",
        "checkpoint",
        "blocks",
        "fps",
        "bundle",
    ],
)
def test_bad_exports_fail_before_writes(serving_export, mutation):
    if mutation in ("config", "decoder"):
        serving_export[mutation + "_sha"] = "0" * 64
    elif mutation == "incomplete":
        (serving_export["source"] / "model_index.json").unlink()
    elif mutation == "nested":
        serving_export["output"] = serving_export["source"] / "prepared"
    elif mutation == "bundle":
        serving_export["bundle"]["config_sha256"] = "0" * 64
    else:
        path = serving_export["source"] / "zing_config.json"
        data = json.loads(path.read_text())
        data.update(
            {
                "checkpoint": {"checkpoint_sha256": "0" * 64},
                "blocks": {"split_first_latent": True},
                "fps": {"h3_media_config": {"fps": 30}},
            }[mutation]
        )
        path.write_text(json.dumps(data))
        serving_export["config_sha"] = digest(path)
    with pytest.raises(ValueError):
        prepare(**serving_export)
    assert not serving_export["output"].exists()


@pytest.mark.parametrize(
    "file",
    ["model/zing_realtime_profile.json", "pipeline.json", "model/zing_config.json"],
)
def test_changed_prepared_files_are_rejected(serving_export, file):
    prepare(**serving_export)
    (serving_export["output"] / file).write_text("{}")
    with pytest.raises(ValueError, match="identities changed"):
        verify_prepared(serving_export["output"], serving_export["bundle"])


def test_other_release_bundle_is_rejected(serving_export):
    prepare(**serving_export)
    with pytest.raises(ValueError, match="release bundle"):
        verify_prepared(serving_export["output"], bundle("shanghai"))


def test_generated_outputs_cannot_enter_checkout(tmp_path):
    from output_paths import REPO, outside_checkout

    with pytest.raises(ValueError, match="outside"):
        outside_checkout(REPO / "launch.json")
    link = tmp_path / "repo-link"
    link.symlink_to(REPO)
    with pytest.raises(ValueError, match="outside"):
        outside_checkout(link / "worker.secret.json")
    assert outside_checkout(tmp_path / "private.json") == tmp_path / "private.json"


@pytest.mark.parametrize("kind", ["taeh3", "official_h3"])
def test_unsupported_service_decoder_is_rejected_before_writes(serving_export, kind):
    with pytest.raises(ValueError, match="supports only lynn_streaming"):
        prepare(**serving_export, decoder_type=kind)
    assert not serving_export["output"].exists()


def test_prepared_decoder_passes_actual_runtime_schema(serving_export):
    from prepare_assets import validate_decoder_config

    prepare(**serving_export)
    value = json.loads((serving_export["output"] / "pipeline.json").read_text())[
        "decoder_config"
    ]
    spec = validate_decoder_config(value)
    assert spec.type == "lynn_streaming" and spec.precision == "bf16"
    with pytest.raises(
        ValueError,
        match="requires decoder_config.checkpoint_path and decoder_config.model_dir",
    ):
        validate_decoder_config({**value, "model_dir": None})
    with pytest.raises(ValueError, match="Unknown Zing decoder type"):
        validate_decoder_config({**value, "type": "taeh3"})
