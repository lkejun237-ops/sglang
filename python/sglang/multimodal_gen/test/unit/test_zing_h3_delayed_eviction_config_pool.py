# SPDX-License-Identifier: Apache-2.0
"""Independent delayed-eviction configuration and staged capacity contracts."""

import importlib.util
import inspect
import json
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
    ZingKVGeometry,
    ZingKVPoolAdmissionError,
    ZingKVPoolConfig,
    ZingKVPoolPlan,
    ZingKVRequest,
    peak_rows,
)
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy
from sglang.multimodal_gen.test.unit.realtime.test_zing_h3_source_adapter import (  # noqa: F401
    configured,
    initialize,
)

FIELD = "zing_window_eviction_timing"
POLICY = "window_eviction_timing"


def _declares(obj, name):
    assert name in inspect.signature(obj).parameters, (
        f"{obj.__name__} must explicitly declare {name}; joint mode cannot select eviction semantics"
    )


@pytest.mark.parametrize("joint", ["off", "source_first", "cross_chunk"])
def test_default_sampling_policy_is_independent_of_joint(joint):
    params = ZingH3SamplingParams(zing_joint_forward=joint)
    assert getattr(params, FIELD, None) == "after_step0"
    assert params.build_request_extra()[FIELD] == "after_step0"


@pytest.mark.parametrize("policy", ["before_source", "after_step0"])
def test_request_extra_preserves_explicit_policy(policy):
    assert FIELD in {f.name for f in fields(ZingH3SamplingParams)}
    params = ZingH3SamplingParams(**{FIELD: policy})
    assert FIELD in params.video_request_extra_fields()
    assert params.build_request_extra()[FIELD] == policy


@pytest.mark.parametrize("bad", ["typo", None, 1, [], "after_step1"])
def test_invalid_sampling_policy_is_rejected_in_english(bad):
    params = ZingH3SamplingParams()
    setattr(params, FIELD, bad)
    with pytest.raises(ValueError, match="eviction") as error:
        params._validate()
    assert str(error.value).isascii()


@pytest.mark.parametrize("policy", [None, "before_source", "after_step0"])
@pytest.mark.parametrize("joint", ["off", "source_first", "cross_chunk"])
def test_source_profile_to_sampling_stage_to_pool_request(
    configured, monkeypatch, policy, joint
):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
        zing_h3_source_adapter as source,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )

    path = Path(configured.model_path) / "zing_realtime_profile.json"
    profile = json.loads(path.read_text())
    profile["joint_forward"] = joint
    if policy is not None:
        profile[POLICY] = policy
    path.write_text(json.dumps(profile))
    adapter, session = initialize()
    monkeypatch.setattr(
        source,
        "build_realtime_sampling_params",
        lambda *a, **kw: ZingH3SamplingParams(),
    )
    params = adapter.build_sampling_params(
        session,
        None,
        SimpleNamespace(request_id="policy"),
        SimpleNamespace(prompt="snow"),
        17,
    )
    expected = policy or "after_step0"
    assert getattr(params, FIELD, None) == expected
    stage = object.__new__(ZingH3GenerationStage)
    stage.modules = {"transformer": SimpleNamespace()}
    stage.packer = SimpleNamespace()
    stage.runtime = dict(
        timestep_shift=5.0, audio_timestep_shift=3.0, num_train_timestep=1000
    )
    sampler = stage._build_sampler(params)
    assert getattr(sampler, POLICY, None) == expected
    # The realtime server builds a different sampler; it must follow the same policy.
    assert getattr(stage._build_realtime_sampler(params), POLICY, None) == expected
    stage.codecs = SimpleNamespace(spatial_ratio=16)
    request = stage.kv_request(
        sampler,
        SimpleNamespace(visual_shape=(5, 2, 2), audio_shape=None),
        {"block_idx": torch.tensor([0, 1])},
        source_shape=(5, 2, 2),
        total_frames=5,
    )
    assert getattr(request, POLICY, None) == expected


def test_source_profile_rejects_invalid_policy(configured):
    path = Path(configured.model_path) / "zing_realtime_profile.json"
    profile = json.loads(path.read_text())
    profile[POLICY] = "after_step1"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError) as error:
        initialize()
    assert str(error.value).isascii()


def _oracle(
    geometry,
    window,
    *,
    prefix_rows,
    rows_per_frame,
    has_source,
    total_frames=None,
    has_audio=False,
    audio_length=None,
    timing="after_step0",
):
    """Enumerate real stage occupancies without invoking the capacity implementation.

    Before source, after source, step0 scratch, after eviction, and final are
    distinct stages. The old live set is selected against the previous chunk.
    Cross union has the same raw row peak: previous final replaces its scratch
    while the next source and next target occupy the transaction tail.
    """
    committed, peak = {}, prefix_rows
    chunks = (
        geometry.chunk_plan.chunks(total_frames)
        if total_frames is not None
        else [geometry.chunk_plan.chunk(i) for i in range(256)]
    )
    for chunk in chunks:
        i = chunk.chunk_id
        old = sum(committed.values())
        retained = {j: n for j, n in committed.items() if window.retains(j, i)}
        if timing == "before_source":
            old = sum(retained.values())
        visual = rows_per_frame * (chunk.end - chunk.start)
        source = visual if has_source else 0
        target = visual
        if has_audio:
            target += 2 * geometry.audio_frames(
                chunk.start,
                chunk.end,
                tail=total_frames is not None and chunk.end == total_frames,
                audio_length=audio_length,
            )
        stages = (
            old,
            old + source,
            old + source + target,
            sum(retained.values()) + source + target,
        )
        peak = max(peak, prefix_rows + max(stages))
        committed = {**retained, i: source + target}
    return peak


@pytest.mark.parametrize("lengths,split", [(5, False), (5, True), ([1, 4, 2], False)])
@pytest.mark.parametrize("window,sink", [(1, 0), (3, 1), (4, 3), (21, 3)])
@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("timing", ["before_source", "after_step0"])
def test_capacity_matches_independent_stage_enumeration(
    lengths, split, window, sink, source, finite, timing
):
    _declares(peak_rows, POLICY)
    geometry = ZingKVGeometry(
        16,
        (1, 2, 2),
        720,
        ChunkPlan(lengths, split),
        video_t_span=lambda n: n * 5 / 3,
        time_span_period=3,
    )
    policy = ZingWindowPolicy(window, sink)
    kwargs = dict(
        prefix_rows=7,
        rows_per_frame=3,
        has_source=source,
        total_frames=83 if finite else None,
        has_audio=True,
        audio_length=145 if finite else None,
    )
    expected = _oracle(geometry, policy, timing=timing, **kwargs)
    assert (
        peak_rows(geometry, policy, **kwargs, window_eviction_timing=timing) == expected
    )


def test_delayed_peak_is_not_window_plus_one_times_longest_chunk():
    _declares(peak_rows, POLICY)
    geometry = ZingKVGeometry(16, (1, 2, 2), 720, ChunkPlan([1, 4, 2], True))
    policy = ZingWindowPolicy(4, 3)
    kwargs = dict(prefix_rows=7, rows_per_frame=3, has_source=True)
    expected = _oracle(geometry, policy, **kwargs)
    assert expected != 7 + (policy.window_chunks + 1) * 4 * 3 * 2
    assert (
        peak_rows(geometry, policy, **kwargs, window_eviction_timing="after_step0")
        == expected
    )


def test_current_capacity_formula_underestimates_delayed_stage_peak():
    """Behavioral red light independent of any newly proposed API name."""
    geometry = ZingKVGeometry(16, (1, 2, 2), 720, ChunkPlan([1, 4, 2], True))
    window = ZingWindowPolicy(4, 3)
    kwargs = dict(prefix_rows=7, rows_per_frame=3, has_source=True)
    expected = _oracle(geometry, window, **kwargs)
    # Once the explicit API exists select it; on the baseline exercise its
    # actual legacy result so this fails on occupancy, rather than imports.
    extra = (
        {POLICY: "after_step0"}
        if POLICY in inspect.signature(peak_rows).parameters
        else {}
    )
    assert peak_rows(geometry, window, **kwargs, **extra) == expected


def _benchmark_module(name):
    root = Path(__file__).resolve().parents[5]
    path = root / "benchmark" / "zing_h3_causal" / (name + ".py")
    spec = importlib.util.spec_from_file_location("delayed_eviction_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("policy", [None, "before_source", "after_step0"])
def test_checkpoint_benchmark_cli_explicit_policy(policy):
    module = _benchmark_module("validate_real_checkpoint")
    parser = module.parser()
    # Check declaration first so baseline absence is an assertion failure,
    # rather than argparse exiting before the semantic contract is inspected.
    subparser = next(
        a
        for a in parser._actions
        if hasattr(a, "choices") and isinstance(a.choices, dict)
    ).choices["sglang"]
    assert "--window-eviction-timing" in subparser._option_string_actions
    argv = [
        "sglang",
        "--request",
        "request.json",
        "--output",
        "out",
        "--sglang-root",
        ".",
        "--model",
        "model",
    ]
    if policy is not None:
        argv += ["--window-eviction-timing", policy]
    args = parser.parse_args(argv)
    assert args.window_eviction_timing == (policy or "after_step0")


def _pool_config(timing):
    _declares(ZingKVPoolConfig, POLICY)
    return ZingKVPoolConfig(
        max_target_area=32 * 32,
        max_target_long_edge=32,
        window_chunks=3,
        sink_chunks=1,
        window_eviction_timing=timing,
    )


def _pool_request(timing):
    _declares(ZingKVRequest, POLICY)
    return ZingKVRequest(
        32, 32, (32, 32), 7, 3, 1, "block_relative", 12, window_eviction_timing=timing
    )


@pytest.mark.parametrize("timing", ["before_source", "after_step0"])
def test_pool_launch_and_admission_use_same_policy(timing):
    config = _pool_config(timing)
    geometry = ZingKVGeometry(16, (1, 2, 2), 32, ChunkPlan([1, 4, 2]))
    expected = _oracle(
        geometry,
        config.window_policy,
        prefix_rows=7,
        rows_per_frame=1,
        has_source=True,
        timing=timing,
    )
    plan = ZingKVPoolPlan.build(config, geometry, token_capacity=expected)
    assert plan.slot_rows == expected
    assert plan.admit(_pool_request(timing)) == expected
    other = "before_source" if timing == "after_step0" else "after_step0"
    with pytest.raises(ZingKVPoolAdmissionError) as error:
        plan.admit(_pool_request(other))
    assert error.value.code == "kv_pool_request_unsupported"
    assert error.value.message.isascii()


def test_capacity_shortfall_rejects_without_changing_policy():
    config = _pool_config("after_step0")
    geometry = ZingKVGeometry(16, (1, 2, 2), 32, ChunkPlan([1, 4, 2]))
    plan = ZingKVPoolPlan.build(
        config,
        geometry,
        token_capacity=_oracle(
            geometry,
            config.window_policy,
            prefix_rows=7,
            rows_per_frame=1,
            has_source=True,
            timing="after_step0",
        ),
    )
    limited = replace(plan, slot_rows=plan.slot_rows - 1)
    request = _pool_request("after_step0")
    with pytest.raises(ZingKVPoolAdmissionError):
        limited.admit(request)
    assert request.window_eviction_timing == "after_step0"
    assert limited.config.window_eviction_timing == "after_step0"


def test_pipeline_launch_contract_propagates_policy():
    _declares(ZingKVPoolConfig, POLICY)
    pipeline = SimpleNamespace(
        zing_kv_pool_window_chunks=3,
        zing_kv_pool_sink_chunks=1,
        zing_kv_pool_rope_position_mode="block_relative",
        zing_kv_pool_rope_max_frame_gap=12,
        zing_kv_pool_window_eviction_timing="after_step0",
    )
    config = ZingKVPoolConfig.from_pipeline_config(
        pipeline, max_sessions=1, mem_fraction_static=0.8
    )
    assert config.window_eviction_timing == "after_step0"


def test_pipeline_default_uses_delayed_eviction():
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3CausalPipelineConfig,
    )

    assert (
        ZingH3CausalPipelineConfig().zing_kv_pool_window_eviction_timing
        == "after_step0"
    )


def test_pool_default_uses_delayed_eviction():
    config = ZingKVPoolConfig(window_chunks=3, sink_chunks=1)
    assert config.window_eviction_timing == "after_step0"


def test_request_default_uses_delayed_eviction():
    request = ZingKVRequest(32, 32, (32, 32), 7, 3, 1, "block_relative", 12)
    assert request.window_eviction_timing == "after_step0"


def test_capacity_default_uses_delayed_eviction():
    config = ZingKVPoolConfig(window_chunks=3, sink_chunks=1)
    geometry = ZingKVGeometry(16, (1, 2, 2), 32, ChunkPlan([1, 4, 2]))
    assert peak_rows(
        geometry, config.window_policy, prefix_rows=7, rows_per_frame=1, has_source=True
    ) == _oracle(
        geometry,
        config.window_policy,
        prefix_rows=7,
        rows_per_frame=1,
        has_source=True,
        timing="after_step0",
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
