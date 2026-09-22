# SPDX-License-Identifier: Apache-2.0
"""Validator unit tests only; these do not certify a progressive VAE kernel."""

import math

import pytest
import torch

from validate_progressive_encoder import (
    check_slices,
    comparison,
    numerical_failure_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
    expected_progressive_slices,
)


@pytest.mark.parametrize("frames", range(1, 35))
def test_tail_count_and_real_watermarks(frames):
    rows = expected_progressive_slices(frames)
    clips, tail = divmod(frames - 1, 17)
    assert rows[0] == (0, 1, 1)
    assert rows[-1][1] == clips * 5 + 1 + math.ceil(tail / 4)
    assert rows[-1][2] <= frames
    assert all(a[1] == b[0] for a, b in zip(rows, rows[1:]))
    assert all(0 <= s < e and e - s <= 2 and 1 <= d <= frames for s, e, d in rows)


def test_cross_clip_resets_absolute_indices_do_not():
    assert expected_progressive_slices(34) == [
        (0, 1, 1),
        (1, 3, 9),
        (3, 5, 17),
        (5, 6, 18),
        (6, 8, 26),
        (8, 10, 34),
    ]
    assert expected_progressive_slices(19) == [
        (0, 1, 1),
        (1, 3, 9),
        (3, 5, 17),
        (5, 6, 18),
        (6, 7, 19),
    ]


@pytest.mark.parametrize("frames", [0, -1, True, 1.5])
def test_invalid_frame_count(frames):
    with pytest.raises(ValueError):
        expected_progressive_slices(frames)


def timeline():
    return [
        dict(start=s, end=e, dependency=d, accepted=d, eos=False)
        for s, e, d in expected_progressive_slices(17)
    ]


@pytest.mark.parametrize(
    "mutation",
    ["future", "duplicate", "missing", "early_eos", "delayed", "false_watermark"],
)
def test_rejects_invalid_evidence(mutation):
    rows = timeline()
    if mutation == "future":
        rows[1]["accepted"] = 8
    if mutation == "duplicate":
        rows[1]["start"] = 0
    if mutation == "missing":
        rows.pop()
    if mutation == "early_eos":
        rows[0]["eos"] = True
    if mutation == "delayed":
        rows[0]["accepted"] = 17
    if mutation == "false_watermark":
        rows[1]["dependency"] = 17
    with pytest.raises(ValueError):
        check_slices(rows, 17, progressive=True)


def test_numerical_gate_does_not_allow_nan_or_silent_tolerance():
    assert comparison(torch.ones(2), torch.ones(2), atol=0, rtol=0)["passed"]
    assert not comparison(torch.ones(2), torch.ones(2) + 1e-5, atol=0, rtol=0)["passed"]
    assert not comparison(
        torch.tensor([float("nan")]), torch.tensor([float("nan")]), atol=0, rtol=0
    )["passed"]
    with pytest.raises(ValueError):
        comparison(torch.ones(1), torch.ones(2), atol=0, rtol=0)


def test_diagnostic_parity_preserves_failure_and_rejects_nonfinite():
    result = comparison(torch.ones(2), torch.ones(2) + 0.125, atol=0, rtol=0)
    assert numerical_failure_blocks(result, report_only=False)
    assert not numerical_failure_blocks(result, report_only=True)
    assert result["passed"] is False
    for value in (float("nan"), float("inf")):
        bad = comparison(torch.tensor([value]), torch.ones(1), atol=0, rtol=0)
        assert numerical_failure_blocks(bad, report_only=True)
