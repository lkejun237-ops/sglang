# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest
import torch

from compare_adaln_diagnostics import compare

_path = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/multimodal_gen/runtime/utils/zing_diagnostics.py"
)
_spec = importlib.util.spec_from_file_location("zing_diagnostic_store_test", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ZingDiagnostic = _module.ZingDiagnostic


def test_independent_exact_stores_deduplicate_and_preserve_rng(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_TENSOR_STORE", "1")
    before = torch.random.get_rng_state().clone()
    for name in ("left", "right"):
        capture = ZingDiagnostic(tmp_path / name)
        for _ in range(3):
            capture.record(
                "cache", history=[torch.arange(12).reshape(3, 4)], active=None
            )
        assert len(list((tmp_path / name / "tensors").glob("*.pt"))) == 1
    assert torch.equal(before, torch.random.get_rng_state())
    result = compare(tmp_path / "left", tmp_path / "right")
    assert result["tensor_occurrences"] == 3
    assert result["unique_blob_pairs_torch_equal"] == 1


def test_dtype_shape_are_part_of_blob_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_TENSOR_STORE", "1")
    capture = ZingDiagnostic(tmp_path / "capture")
    x = torch.zeros(4, dtype=torch.int32)
    capture.record("shapes", a=x, b=x.reshape(2, 2), c=x.view(torch.float32))
    assert len(list((capture.directory / "tensors").glob("*.pt"))) == 3


def test_budget_fails_without_unbounded_write(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_TENSOR_STORE", "1")
    monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_TENSOR_STORE_BYTES", "1")
    capture = ZingDiagnostic(tmp_path / "capture")
    with pytest.raises(RuntimeError, match="budget exceeded"):
        capture.record("too_large", x=torch.ones(4))
    assert not list((capture.directory / "tensors").glob("*.pt"))


@pytest.mark.parametrize("difference", ["value", "nonfinite", "corrupt"])
def test_comparison_rejects_mismatch_or_corruption(tmp_path, monkeypatch, difference):
    monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_TENSOR_STORE", "1")
    for name in ("left", "right"):
        x = torch.ones(3)
        if difference == "value" and name == "right":
            x[0] = 2
        elif difference == "nonfinite":
            x[0] = float("nan")
        ZingDiagnostic(tmp_path / name).record("output", x=x)
    if difference == "corrupt":
        path = next((tmp_path / "right" / "tensors").glob("*.pt"))
        torch.save(torch.zeros(3), path)
    with pytest.raises(ValueError):
        compare(tmp_path / "left", tmp_path / "right")


def test_only_startup_rng_is_outside_request_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_TENSOR_STORE", "1")
    with torch.random.fork_rng():
        for i, name in enumerate(("left", "right")):
            torch.manual_seed(i)
            capture = ZingDiagnostic(tmp_path / name)
            capture.record("conditioning", x=torch.ones(3))
            torch.manual_seed(42)
            capture.record("seeded")
            capture.record("initial_noise", x=torch.randn(3))
        result = compare(tmp_path / "left", tmp_path / "right")
        assert result["startup_rng_snapshots_before_seed"] == 1
        assert result["request_rng_snapshots_compared"] == 2
        import json

        path = tmp_path / "right/events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events[-1]["rng"]["cpu"] = "changed"
        path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        with pytest.raises(ValueError, match="request RNG"):
            compare(tmp_path / "left", tmp_path / "right")


def test_parallel_store_is_exact_and_recaptures_mutated_values(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_TENSOR_STORE", "1")
    for name, workers in (("left", 1), ("right", 4)):
        monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_TENSOR_STORE_WORKERS", str(workers))
        capture = ZingDiagnostic(tmp_path / name)
        tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).t()
        capture.record("first", values=[tensor] * 20)
        tensor.add_(1)
        capture.record("mutated", values=[tensor] * 20)
        # Concurrent duplicate writes must not create multiple blobs or reuse
        # an earlier value just because the tensor identity is unchanged.
        assert len(list((capture.directory / "tensors").glob("*.pt"))) == 2
        assert capture.tensor_bytes == sum(
            p.stat().st_size for p in (capture.directory / "tensors").glob("*.pt")
        )
    result = compare(tmp_path / "left", tmp_path / "right")
    assert result["tensor_occurrences"] == 40
    assert result["unique_blob_pairs_torch_equal"] == 2


def test_parallel_budget_failure_finishes_pending_work(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_TENSOR_STORE", "1")
    monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_TENSOR_STORE_WORKERS", "4")
    monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_TENSOR_STORE_BYTES", "1")
    capture = ZingDiagnostic(tmp_path / "capture")
    with pytest.raises(RuntimeError, match="budget exceeded"):
        capture.record("too_large", values=[torch.ones(4) for _ in range(20)])
    assert not (capture.directory / "events.jsonl").exists()
    assert not list((capture.directory / "tensors").glob("*.pt"))
