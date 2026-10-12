# SPDX-License-Identifier: Apache-2.0
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "issue269_audit", Path(__file__).parents[1] / "audit_historical.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def evidence(ref="same", source="same"):
    return {
        "warmup_conditions": [
            {"operation": "encode_condition", "result": {"sha256": "qwen"}},
            {"operation": "encode_visual", "result": {"sha256": ref}},
            {
                "operation": "sample_streaming_visual_moments",
                "result": {"sha256": source},
            },
        ],
        "warmup_moments_sha256": ["moments"],
    }


def test_distingishes_rank_mismatch_from_cross_process_stage():
    summary = {
        "arms": {
            "perf-tile-off": {
                "evidence": {"0": evidence(), "1": evidence(ref="rank1")}
            },
            "perf-tile-cross": {
                "evidence": {
                    "0": evidence(source="different"),
                    "1": evidence(ref="rank1", source="different"),
                }
            },
        }
    }
    report = audit.audit(summary)["arms"]
    assert report["perf-tile-off"]["first_vs_off"] is None
    assert report["perf-tile-off"]["rank_mismatches"][0]["operation"] == "encode_visual"
    assert (
        report["perf-tile-cross"]["first_vs_off"]["operation"]
        == "sample_streaming_visual_moments"
    )
    assert report["perf-tile-cross"]["source_moments_rank_equal"]


def test_does_not_compare_misaligned_rank_events():
    left = evidence()
    right = copy.deepcopy(left)
    right["warmup_conditions"].reverse()
    with pytest.raises(ValueError, match="operation order"):
        audit.audit({"arms": {"perf-tile-off": {"evidence": {"0": left, "1": right}}}})
