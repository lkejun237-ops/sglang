# SPDX-License-Identifier: Apache-2.0
import copy

import pytest
from prepare_encoder_overlap import (
    digest,
    set_request_resolution,
    validate_deployment_manifest,
)


def request():
    return {
        "messages": [
            {"content": [{"type": "image", "id": str(i)} for i in range(6)]},
            {
                "source": {"uri": "source.mp4"},
                "content": [
                    {
                        "type": "video",
                        "output": {"frames": 120, "width": 832, "height": 480},
                    }
                ],
            },
        ],
        "metadata": {
            "smoke_contract": {
                "target": {"width": 832, "height": 480, "frames": 120, "fps": 24}
            }
        },
    }


@pytest.mark.parametrize("width,height", [(832, 480), (896, 512), (1280, 736)])
def test_target_resolution_preserves_business_inputs(width, height):
    value = request()
    original = copy.deepcopy(value)
    set_request_resolution(value, width, height)
    assert value["messages"][0] == original["messages"][0]
    assert value["messages"][1]["source"] == original["messages"][1]["source"]
    assert value["messages"][1]["content"][0]["output"] == {
        "frames": 120,
        "width": width,
        "height": height,
    }
    assert value["metadata"]["smoke_contract"]["target"] == {
        "frames": 120,
        "fps": 24,
        "width": width,
        "height": height,
    }


@pytest.mark.parametrize("width,height", [(0, 480), (833, 480), (832, 481), (832, 496)])
def test_invalid_resolution_rejected_without_mutation(width, height):
    value = request()
    original = copy.deepcopy(value)
    with pytest.raises(ValueError):
        set_request_resolution(value, width, height)
    assert value == original


@pytest.mark.parametrize("changed", [None, "model", "weights", "architecture", "steps"])
def test_preparation_uses_manifest_identity_not_one_checkpoint(tmp_path, changed):
    (tmp_path / "zing_config.json").write_text('{"checkpoint": "another-valid-model"}')
    weights = tmp_path / "lynn.pt"
    weights.write_bytes(b"another-checkpoint")
    (tmp_path / "config.json").write_text("{}")
    profile = dict(
        model_config_sha256=digest(tmp_path / "zing_config.json"),
        solver="few_step",
        denoising_step_list=[900, 600, 300, 100],
        noise_version="indexed-cpu-v1",
        source_encoder={"mode": "progressive"},
    )
    manifest = dict(
        realtime_profile=profile,
        decoder_checkpoint_sha256=digest(weights),
        decoder_config={"type": "lynn_streaming", "model_dir": str(tmp_path)},
        decoder_architecture_sha256=digest(tmp_path / "config.json"),
    )
    original = copy.deepcopy(manifest)
    if changed == "model":
        (tmp_path / "zing_config.json").write_text("changed")
    elif changed == "weights":
        weights.write_bytes(b"changed")
    elif changed == "architecture":
        (tmp_path / "config.json").write_text("changed")
    elif changed == "steps":
        profile["denoising_step_list"] = [900]
    if changed:
        with pytest.raises(ValueError):
            validate_deployment_manifest(manifest, tmp_path, weights)
    else:
        assert validate_deployment_manifest(manifest, tmp_path, weights) == profile
        assert manifest == original
