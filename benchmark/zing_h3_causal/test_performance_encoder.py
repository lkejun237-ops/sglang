# SPDX-License-Identifier: Apache-2.0
"""Probe evidence gates; CPU tests do not establish GPU acceptance."""

import pytest
import torch
from validate_performance_encoder import (
    compare,
    numerical_failure_blocks,
    verify_execution,
)


def test_diagnostic_tolerance_does_not_relabel_bitwise_difference():
    reference = torch.ones(8, dtype=torch.float64)
    result = compare(reference + 0.001, reference)
    assert result["passed"] and not result["bitwise"]
    assert result["relative_l2"] == pytest.approx(0.001)
    assert result["max_abs"] == pytest.approx(0.001)
    assert compare(reference, reference)["bitwise"]


def test_absolute_and_relative_gates_are_both_required():
    assert not compare(torch.ones(8) * 1.01, torch.ones(8))["passed"]
    reference = torch.ones(8) * 1000
    # Small relative L2 must not hide an absolute outlier.
    assert not compare(reference + 0.5, reference)["passed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_never_passes_even_with_identical_bytes(value):
    x = torch.tensor([value])
    result = compare(x, x)
    assert not result["passed"] and not result["bitwise"] and not result["finite"]


def test_shape_and_dtype_mismatch_fail_closed():
    assert not compare(torch.ones(2), torch.ones(3))["passed"]
    assert not compare(torch.ones(2), torch.ones(2).double())["passed"]


def execution():
    return {
        "rows": [
            {
                "tile_execution": dict(
                    local_tile_ids=[0, 2, 4, 6, 8, 10],
                    tile_count=12,
                    requested_mode="auto",
                    encoder_ranks=2,
                    effective_mode="parallel",
                )
            }
        ]
    }


def test_replicated_fallback_cannot_pass_partition_gate():
    timing = execution()
    verify_execution(timing, 2, 0)
    timing["rows"][0]["tile_execution"]["effective_mode"] = "replicated"
    with pytest.raises(ValueError, match="partition"):
        verify_execution(timing, 2, 0)


@pytest.mark.parametrize(
    "ids", [[0, 2, 4, 6, 8, 8], [1, 3, 5, 7, 9, 11], list(range(12))]
)
def test_wrong_or_replicated_tile_ids_fail(ids):
    timing = execution()
    timing["rows"][0]["tile_execution"]["local_tile_ids"] = ids
    with pytest.raises(ValueError, match="partition"):
        verify_execution(timing, 2, 0)


def test_report_only_retains_failure_without_blocking_collection():
    result = compare(torch.ones(8) * 2, torch.ones(8))
    assert not result["passed"]
    assert numerical_failure_blocks(result, required="passed", report_only=False)
    assert not numerical_failure_blocks(result, required="passed", report_only=True)
    assert not result["passed"]


@pytest.mark.parametrize("actual", [torch.ones(3), torch.tensor([float("nan")])])
def test_report_only_still_blocks_invalid_outputs(actual):
    result = compare(actual, torch.ones(1))
    assert numerical_failure_blocks(result, required="passed", report_only=True)
