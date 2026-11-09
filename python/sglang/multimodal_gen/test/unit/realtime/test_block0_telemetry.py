import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.block0_telemetry import (
    Block0Telemetry,
)


def test_overlapping_spans_and_missing_work_are_not_added_to_elapsed():
    now = [0.0]
    metrics = Block0Telemetry(clock=lambda: now[0])
    with metrics.measure("prepare"):
        now[0] = 0.1
        with metrics.measure("qwen"):
            now[0] = 0.3
        now[0] = 0.4
    metrics.skip("disk_read")
    report = metrics.snapshot()
    assert report["elapsed_host_ms"] == pytest.approx(400)
    assert report["stages"]["prepare"]["host_ms"] == pytest.approx(400)
    assert report["stages"]["qwen"]["host_ms"] == pytest.approx(200)
    assert report["stages"]["disk_read"] == {"host_ms": None, "status": "skipped"}
    metrics.record("qwen", 10)
    assert report["stages"]["qwen"]["host_ms"] == pytest.approx(200)


def test_failed_span_keeps_error_status():
    metrics = Block0Telemetry()
    with pytest.raises(OSError):
        with metrics.measure("disk_read"):
            raise OSError("private detail")
    report = metrics.snapshot()
    assert report["stages"]["disk_read"]["status"] == "error"
    assert "private" not in str(report)
