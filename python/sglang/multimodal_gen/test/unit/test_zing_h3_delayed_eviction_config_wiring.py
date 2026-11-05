# SPDX-License-Identifier: Apache-2.0
"""Additional configuration boundaries and visible session policy contracts."""

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
    ZingSourceStreamState,
)
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import ZingKVPoolConfig
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy
from sglang.multimodal_gen.test.unit.realtime.test_zing_h3_source_adapter import (  # noqa: F401
    configured,
    initialize,
)


@pytest.mark.parametrize("policy", ["before_source", "after_step0"])
@pytest.mark.usefixtures("configured")
def test_source_api_cannot_override_server_eviction_policy(policy):
    with pytest.raises(ValueError):
        initialize(zing_window_eviction_timing=policy)


@pytest.mark.parametrize("policy", ["wrong", None, [], 1])
def test_launch_rejects_invalid_eviction_policy(policy):
    with pytest.raises(ValueError, match="eviction"):
        ZingKVPoolConfig(window_eviction_timing=policy)


@pytest.mark.parametrize("policy", ["before_source", "after_step0"])
def test_source_state_reports_explicit_eviction_policy(policy):
    state = ZingSourceStreamState()
    params = ZingH3SamplingParams(zing_window_eviction_timing=policy)
    sampler = SimpleNamespace(
        window_policy=ZingWindowPolicy(3, 1),
        rope_position_mode="block_relative",
        rope_max_frame_gap=12,
        window_eviction_timing=policy,
        joint_forward="off",
    )

    class StopBeforeCodec(RuntimeError):
        pass

    def stop(*args):
        raise StopBeforeCodec

    stage = SimpleNamespace(
        _validate_sampling_params=lambda p: p._validate(),
        _build_sampler=lambda p: sampler,
        _build_realtime_sampler=lambda p: sampler,
        codecs=SimpleNamespace(video_decoder_name="stable", prepare=stop),
        _build_request=stop,
    )
    batch = SimpleNamespace(sampling_params=params, extra={"zing_source_command": {}})
    with pytest.raises(StopBeforeCodec):
        state.initialize(stage, batch, {})
    assert state.policy["window_eviction_timing"] == policy
