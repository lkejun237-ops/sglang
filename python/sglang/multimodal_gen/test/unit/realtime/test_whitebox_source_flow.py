# SPDX-License-Identifier: Apache-2.0
"""Negotiated native dependencies are independent of region and input inventory."""

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
    progressive_source_dependency,
    progressive_source_flow,
)


@pytest.mark.parametrize(
    "latent_end,pixels",
    [(1, 1), (2, 9), (3, 9), (4, 17), (5, 17), (6, 18), (8, 26), (10, 34)],
)
def test_dependency_follows_encoder_groups_not_output_pixel_count(latent_end, pixels):
    assert progressive_source_dependency(latent_end) == pixels


@pytest.mark.parametrize(
    "lengths,split,minimum",
    [([3, 2], False, 9), ([3, 2], True, 1), ([5], False, 17), ([2, 3], False, 9)],
)
def test_first_output_dependency_comes_from_the_native_plan(lengths, split, minimum):
    plan = ChunkPlan(lengths, split_first_latent=split)
    flow = progressive_source_flow(plan, progressive_output=True)
    assert flow["first_output_input_frames"] == minimum
    assert flow["input_groups"] == [1, 8, 8]
    assert flow["progressive_output"] is True


def test_room_uses_verified_native_minimum_without_a_region_override(tmp_path):
    config = WhiteboxConfig(
        engine_factory=lambda: None,
        ui_factory=lambda: None,
        spool=tmp_path,
        room_owned=True,
        capture_clock=True,
        model_input_frames=25,
        pipeline_overlap=True,
    )
    assert config.minimum_model_input_frames == 18
    ready = {
        "source_output_release_protocol": "valid-source-frames-v1",
        "source_flow": progressive_source_flow(
            ChunkPlan([3, 2], split_first_latent=False), progressive_output=True
        ),
    }
    assert config.negotiated_minimum_model_input_frames(ready) == 9
    assert config.model_input_window(ready["source_output_release_protocol"]) == 25
    assert config.negotiated_minimum_model_input_frames({}) == 18
    assert config.model_input_window() == 29


@pytest.mark.parametrize(
    "change",
    [
        {"first_output_input_frames": True},
        {"first_output_input_frames": 0},
        {"input_groups": [17]},
        {"progressive_output": "1"},
        {"protocol": "unknown"},
    ],
)
def test_room_rejects_incomplete_or_invalid_flow_claims(tmp_path, change):
    config = WhiteboxConfig(
        engine_factory=lambda: None, ui_factory=lambda: None, spool=tmp_path
    )
    flow = progressive_source_flow(
        ChunkPlan([3, 2], split_first_latent=False), progressive_output=True
    )
    flow.update(change)
    with pytest.raises(ValueError, match="Invalid negotiated source flow"):
        config.negotiated_minimum_model_input_frames(
            {
                "source_flow": flow,
                "source_output_release_protocol": "valid-source-frames-v1",
            }
        )
    with pytest.raises(ValueError, match="Invalid negotiated source flow"):
        config.negotiated_minimum_model_input_frames({"source_flow": flow})
