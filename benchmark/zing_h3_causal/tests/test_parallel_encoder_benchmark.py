# SPDX-License-Identifier: Apache-2.0
"""Acceptance must distinguish actual tile work from a successful fallback."""

import pytest
from benchmark_parallel_encoder import validate_execution


def execution():
    return dict(
        group="dedicated_encoder",
        world_size=2,
        encoder_ranks=2,
        participating_group_ranks=2,
        replicated_copies=1,
        effective_mode="parallel",
        fallback_reason=None,
        gather_calls=1,
        gather_input_bytes=4096,
        local_tile_ids=[0, 2],
    )


def test_actual_parallel_execution_is_accepted():
    validate_execution(execution(), 2)


@pytest.mark.parametrize(
    "change",
    [
        dict(
            effective_mode="replicated",
            fallback_reason="unvalidated_encode_mode",
            encoder_ranks=1,
            replicated_copies=2,
        ),
        dict(group="vae_decode"),
        dict(world_size=4),
        dict(encoder_ranks=1),
        dict(participating_group_ranks=1),
        dict(replicated_copies=2),
        dict(gather_calls=0),
        dict(gather_input_bytes=0),
        dict(local_tile_ids=[]),
    ],
)
def test_replicated_or_incomplete_execution_is_not_parallel_acceptance(change):
    row = execution()
    row.update(change)
    with pytest.raises(RuntimeError):
        validate_execution(row, 2)


def test_missing_receipt_is_not_success():
    with pytest.raises(RuntimeError, match="missing"):
        validate_execution(None, 2)


def test_single_rank_baseline_does_not_claim_multi_rank_work():
    row = dict(
        group="dedicated_encoder",
        world_size=1,
        encoder_ranks=1,
        effective_mode="replicated",
        fallback_reason="single_rank_or_uninitialized_group",
    )
    validate_execution(row, 1)
    with pytest.raises(RuntimeError):
        validate_execution(row, 2)


@pytest.mark.parametrize(
    "entry",
    [
        "benchmark_parallel_encoder",
        "probe_encoder_transport_gpu",
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker",
    ],
)
def test_entrypoint_bootstrap_in_fresh_process_without_sitecustomize(entry):
    import json
    import os
    import site
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    paths = [str(root / "python"), str(root / "benchmark/zing_h3_causal")]
    paths += site.getsitepackages() + [site.getusersitepackages()]
    script = f"""
import sys, importlib
sys.path[:0] = {json.dumps(paths)}
assert 'sitecustomize' not in sys.modules
assert 'sglang.multimodal_gen.configs.pipeline_configs' not in sys.modules
if sys.platform == 'darwin':
    from sglang._platform_stubs import install_platform_stubs
    install_platform_stubs()
assert 'sglang.multimodal_gen.configs.pipeline_configs' not in sys.modules
entry = importlib.import_module({entry!r})
ServerArgs, setter = entry.initialize_runtime_imports()
assert ServerArgs.__name__ == 'ServerArgs' and callable(setter)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
assert callable(get_global_server_args)
import torch
assert not torch.cuda.is_initialized()
print('ENTRYPOINT_BOOTSTRAP_OK')
"""
    env = dict(os.environ, SGLANG_LIGHTWEIGHT_RUNTIME="1")
    env.pop("PYTHONPATH", None)
    # -S suppresses all automatic site/sitecustomize initialization. Dependency
    # directories are added as plain paths, without processing .pth files.
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ENTRYPOINT_BOOTSTRAP_OK" in result.stdout


def test_encoder_limit_override_is_explicit_local_and_preserves_model_mode():
    from benchmark_parallel_encoder import configure_encoder_policy

    runtime = {
        "vae_config": {"encode_mode": "performance", "encode_dtype": "bfloat16"},
        "dit": {"steps": 4},
    }
    assert configure_encoder_policy(runtime, None) == runtime
    effective = configure_encoder_policy(runtime, 1)
    assert effective["vae_config"] == dict(
        runtime["vae_config"], cudnn_benchmark_limit=1
    )
    assert effective["dit"] == runtime["dit"]
    assert "cudnn_benchmark_limit" not in runtime["vae_config"]
    for invalid in (True, 0, 2, -1, "1"):
        with pytest.raises(ValueError):
            configure_encoder_policy(runtime, invalid)


def test_policy_requires_actual_context_values_and_frozen_configuration():
    from benchmark_parallel_encoder import validate_encode_policy

    config = dict(
        encode_mode="performance", encode_dtype="bfloat16", cudnn_benchmark_limit=1
    )
    policy = dict(
        config,
        cudnn_benchmark=True,
        cudnn_deterministic=False,
        deterministic_algorithms=False,
    )
    validate_encode_policy({"encode_policy": policy}, config)
    for change in (
        {"cudnn_benchmark_limit": 10},
        {"encode_dtype": "float32"},
        {"cudnn_benchmark": False},
    ):
        with pytest.raises(RuntimeError):
            validate_encode_policy({"encode_policy": dict(policy, **change)}, config)
    with pytest.raises(RuntimeError):
        validate_encode_policy({}, config)


def test_transport_numeric_diagnostic_keeps_failure_and_rejects_structure_mismatch():
    import torch
    from probe_encoder_transport_gpu import compare_moments

    expected = torch.ones(2)
    assert compare_moments(expected, expected)["passed"] is True
    row = compare_moments(expected + 0.01, expected)
    assert row["passed"] is False and row["max_abs"] > 0
    assert compare_moments(expected * float("nan"), expected)["passed"] is False
    with pytest.raises(RuntimeError, match="shape or dtype"):
        compare_moments(expected[:1], expected)
