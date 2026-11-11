# SPDX-License-Identifier: Apache-2.0
"""Applicable pre-RPC regressions retained under the approved v6 redesign."""

import copy
import importlib
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[2] / "runtime/conditioning/contract.py"


@pytest.fixture
def api():
    if not SOURCE.is_file():
        pytest.skip("Feature assertion above owns the initial red gate")
    return importlib.import_module(
        "sglang.multimodal_gen.runtime.conditioning.contract"
    )


@pytest.fixture
def profile(api):
    return api.DeploymentProfile(
        dict(
            schema_version=1,
            model={"sha256": "1" * 64},
            projection=None,
            tokenizer={"sha256": "2" * 64},
            processor={"sha256": "3" * 64},
            runtime={"video_sample_fps": 2},
            implementation_version="1",
            preprocess_version="rgb-v1",
            numerical_environment={
                "encoder_tp": 1,
                "precision": "bf16",
                "gpu": "fixture",
                "cc": "9.0",
                "backend": "fixture",
                "software": {"torch": "fixture"},
            },
        )
    )


def test_profile_content_hash_changes_and_schema_rejects_unknown(api, profile):
    for key in (
        "model",
        "projection",
        "tokenizer",
        "processor",
        "runtime",
        "numerical_environment",
    ):
        changed = copy.deepcopy(profile.document)
        changed[key] = (
            {"changed": True}
            if key != "numerical_environment"
            else changed[key] | {"gpu": "other"}
        )
        assert api.DeploymentProfile(changed).profile_id != profile.profile_id
    with pytest.raises(api.ConditionError):
        api.DeploymentProfile(profile.document | {"unknown": 1})


def test_internal_error_detail_and_safe_public_message(api):
    error = api.ConditionError(
        "CONDITION_INVALID", "Invalid conditioning artifact at /private/secret"
    )
    assert "conditioning artifact" in str(error)
    assert "/private/secret" not in error.public_message
    assert error.code == "CONDITION_INVALID"


def test_error_http_status_compatibility_and_unauthorized(api):
    assert api.ConditionError("CONDITION_BUSY", 429).status == 429
    error = api.ConditionError(
        "CONDITION_UNAUTHORIZED", "Ingress conditioning key is revoked"
    )
    assert error.status == 401
    assert "key" not in error.public_message and "revoked" not in error.public_message
    assert api.ConditionError("CONDITION_TIMEOUT").status == 504


@pytest.mark.parametrize("missing", ["gpu", "cc", "backend", "precision", "software"])
def test_profile_requires_frozen_numeric_environment(api, profile, missing):
    document = profile.document
    del document["numerical_environment"][missing]
    with pytest.raises(api.ConditionError):
        api.DeploymentProfile(document)


@pytest.mark.parametrize("max_media", [1, 3, 33])
def test_rpc_media_capacity_is_configurable_not_capped_at_22(api, profile, max_media):
    import torch

    from sglang.multimodal_gen.runtime.conditioning.inputs import content_identity
    from sglang.multimodal_gen.runtime.conditioning.resources import ResourceLimits
    from sglang.multimodal_gen.runtime.conditioning.wire import (
        decode_request,
        encode_request,
    )

    media = [
        dict(
            type="video" if i == 0 else "image",
            fps=24 if i == 0 else 1,
            frames=torch.full((3, 1, 2, 4), i, dtype=torch.uint8),
        )
        for i in range(max_media)
    ]
    body = encode_request("prompt", media, profile.profile_id)
    request = decode_request(body, profile, ResourceLimits(max_media=max_media))
    header, decoded = request.header, request.media
    assert len(decoded) == max_media
    assert header["input_digest"] == content_identity(
        "prompt", media, profile.profile_id
    )
    assert torch.equal(decoded[-1]["frames"], media[-1]["frames"])
    extra = encode_request("prompt", [*media, media[-1]], profile.profile_id)
    with pytest.raises(api.ConditionError, match="media list"):
        decode_request(extra, profile, ResourceLimits(max_media=max_media))
    if max_media > 22:
        with pytest.raises(api.ConditionError, match="media list"):
            decode_request(body, profile, ResourceLimits())
        changed = [*media[:-1], dict(media[-1], frames=media[-1]["frames"] + 1)]
        assert (
            content_identity("prompt", changed, profile.profile_id)
            != header["input_digest"]
        )
