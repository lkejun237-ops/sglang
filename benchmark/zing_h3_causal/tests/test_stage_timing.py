# SPDX-License-Identifier: Apache-2.0
"""Boundaries and delayed CUDA collection, without importing model code."""

import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/multimodal_gen/runtime/utils/zing_stage_timing.py"
)
spec = importlib.util.spec_from_file_location("zing_stage_timing", PATH)
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


class Event:
    def __init__(self, ready=True):
        self.ready = ready
        self.records = 0

    def record(self):
        self.records += 1

    def query(self):
        return self.ready

    def elapsed_time(self, end):
        assert end.ready
        return 12.5

    def synchronize(self):
        raise AssertionError("Timing must not synchronize")


def test_host_return_and_tensor_handoff_are_distinct():
    rows = []
    ticks = iter([100, 160, 190])
    trace = timing.StageTrace(
        "session", "dispatch", rows.append, clock=lambda: next(ticks)
    )
    with trace.span("DiT", input_ids=["condition"], output_ids=["latent"]) as span:
        pass
    assert not rows  # Handoff is recorded by the caller, after function return.
    span.handoff()
    row, start, end = rows[0]
    assert row["host_enter_ns"] == 100
    assert row["host_return_ns"] == 160
    assert row["host_handoff_ns"] == 190
    assert row["host_call_ms"] == pytest.approx(0.00006)
    assert row["enqueue_ns"] is None  # No queue boundary was observed.
    assert row["request_id"] == "session"
    assert row["dispatch_id"] == "dispatch"
    assert row["input_ids"] == ["condition"]
    assert row["output_ids"] == ["latent"]
    assert start is end is None


def test_cuda_samples_wait_for_completion_without_synchronizing():
    rows = []
    start, end = Event(), Event(False)
    trace = timing.StageTrace("s", "d", rows.append, event_factory=lambda: (start, end))
    with trace.span("Decoder", cuda=True) as span:
        pass
    span.handoff()
    row, first, last = rows[0]
    assert start.records == end.records == 1
    assert timing.resolve_cuda(row, first, last) is False
    assert "cuda_stream_ms" not in row
    end.ready = True
    assert timing.resolve_cuda(row, first, last) is True
    assert row["cuda_stream_ms"] == 12.5
    assert row["cuda_status"] == "complete"


def test_stage_exception_preserves_failed_span_and_original_error():
    rows = []
    trace = timing.StageTrace("s", "d", rows.append)
    with pytest.raises(ValueError, match="model failed"):
        with trace.span("Encoder"):
            raise ValueError("model failed")
    row, _, _ = rows[0]
    assert row["status"] == "error"
    assert row["host_handoff_ns"] is None
    assert row["error_type"] == "ValueError"


def test_unavailable_cuda_is_missing_instead_of_host_fallback():
    rows = []

    def unavailable():
        raise RuntimeError("No CUDA")

    trace = timing.StageTrace("s", "d", rows.append, event_factory=unavailable)
    with trace.span("DiT", cuda=True) as span:
        pass
    span.handoff()
    row, start, end = rows[0]
    assert timing.resolve_cuda(row, start, end)
    assert row["cuda_status"] == "error"
    assert row["cuda_stream_ms"] is None


def test_disabled_collection_does_not_construct_events_or_writer(monkeypatch):
    monkeypatch.delenv("SGLANG_ZING_STAGE_TIMING", raising=False)
    monkeypatch.setattr(timing, "get_writer", lambda *a: pytest.fail("Writer started"))
    assert timing.trace_for_batch(object()) is None


def test_double_handoff_cannot_duplicate_sample():
    rows = []
    trace = timing.StageTrace("s", "d", rows.append)
    with trace.span("DiT") as span:
        pass
    span.handoff()
    span.handoff()
    assert len(rows) == 1


def test_health_publication_interruption_keeps_last_complete_snapshot(
    monkeypatch, tmp_path
):
    path = tmp_path / "health.json"
    previous = {"written": 1, "pending": 0}
    timing.publish_health(path, previous)
    original_replace = Path.replace

    def interrupted_replace(self, target):
        assert json.loads(path.read_text()) == previous
        raise OSError("Publication interrupted")

    monkeypatch.setattr(Path, "replace", interrupted_replace)
    with pytest.raises(OSError, match="Publication interrupted"):
        timing.publish_health(path, {"written": 2, "pending": 0})
    assert json.loads(path.read_text()) == previous
    monkeypatch.setattr(Path, "replace", original_replace)
    timing.publish_health(path, {"written": 3, "pending": 0})
    assert json.loads(path.read_text()) == {"written": 3, "pending": 0}


def test_missing_dispatch_identity_is_recorded_without_interrupting_generation(
    monkeypatch, tmp_path
):
    rows = []
    monkeypatch.setenv("SGLANG_ZING_STAGE_TIMING", "host")
    monkeypatch.setenv("SGLANG_ZING_STAGE_TIMING_DIR", str(tmp_path))
    monkeypatch.setattr(
        timing, "get_writer", lambda directory: SimpleNamespace(submit=rows.append)
    )
    trace = timing.trace_for_batch(SimpleNamespace(realtime_session_id="session"))
    with trace.span("DiT") as span:
        pass
    span.handoff()
    assert rows[0][0]["dispatch_id"] is None
    assert rows[0][0]["dispatch_id_status"] == "missing"


def test_writer_overflow_is_visible_and_completed_events_are_drained(tmp_path):
    writer = timing.Writer(tmp_path, capacity=1, event_timeout=2)
    start, end = Event(), Event(False)
    writer.submit(({"cuda_status": "pending"}, start, end))
    deadline = time.monotonic() + 2
    try:
        while True:
            try:
                if json.loads(writer.health_path.read_text())["pending"] == 1:
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            assert time.monotonic() < deadline
            time.sleep(0.01)
        writer.submit(({"cuda_status": "not_requested"}, None, None))
        writer.submit(({"cuda_status": "not_requested"}, None, None))
        assert writer.dropped == 1
    finally:
        end.ready = True
        writer.stop.set()
        writer.thread.join(timeout=2)
    assert not writer.thread.is_alive()
    rows = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["cuda_stream_ms"] == 12.5
    health = json.loads(writer.health_path.read_text())
    assert health["dropped"] == 1
    assert health["pending"] == health["queued"] == 0
