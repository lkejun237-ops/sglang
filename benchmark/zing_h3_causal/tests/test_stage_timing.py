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


def test_component_worker_trace_keeps_external_request_identity(monkeypatch, tmp_path):
    rows = []
    monkeypatch.setenv("SGLANG_ZING_STAGE_TIMING", "host")
    monkeypatch.setenv("SGLANG_ZING_STAGE_TIMING_DIR", str(tmp_path))
    monkeypatch.setattr(
        timing, "get_writer", lambda directory: SimpleNamespace(submit=rows.append)
    )
    trace = timing.trace_for_ids(
        "client-session",
        "client-session:source:0",
        enabled=True,
        identity={"worker_role": "dedicated_source_encoder"},
    )
    with trace.span("Encoder", component="source_vae") as span:
        pass
    span.handoff(latent_range=[0, 5], source_dependency_end=17)
    row = rows[0][0]
    assert row["request_id"] == "client-session"
    assert row["dispatch_id"] == "client-session:source:0"
    assert row["worker_role"] == "dedicated_source_encoder"
    assert row["generation_id"] is None
    assert row["latent_range"] == [0, 5]
    assert row["source_dependency_end"] == 17


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
    trace = timing.trace_for_batch(
        SimpleNamespace(
            realtime_session_id="session",
            realtime_generation_id="gen-1",
            extra={"return_stage_timing": True},
        )
    )
    with trace.span("DiT") as span:
        pass
    span.handoff()
    assert rows[0][0]["dispatch_id"] is None
    assert rows[0][0]["dispatch_id_status"] == "missing"
    assert rows[0][0]["generation_id"] == "gen-1"


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


def test_profile_waits_for_cuda_and_collects_only_its_session(tmp_path):
    import asyncio

    writer = timing.Writer(tmp_path)
    end = Event(False)
    trace = timing.StageTrace(
        "current",
        "current_d",
        writer.submit,
        measure_cuda=True,
        event_factory=lambda: (Event(), end),
        identity={"rank": 0, "pid": 1},
    )
    with trace.span("DiT", cuda=True) as span:
        pass
    span.handoff()
    trace.finish()

    async def collect():
        task = asyncio.create_task(
            timing.collect_profile(tmp_path, "current", [0], 1, timeout=1)
        )
        await asyncio.sleep(0.05)
        assert not task.done()
        end.ready = True
        return await task

    try:
        profile = asyncio.run(collect())
        assert profile["status"] == "complete"
        assert len(profile["stages"]) == 1
        assert profile["stages"][0]["cuda_stream_ms"] == 12.5
        other = asyncio.run(
            timing.collect_profile(tmp_path, "previous", [0], 1, timeout=0)
        )
        assert other["status"] == "incomplete"
        assert other["stages"] == []
    finally:
        end.ready = True
        writer.stop.set()
        writer.thread.join(2)


def test_profile_requires_matching_dispatches_from_every_rank(tmp_path):
    import asyncio

    for rank in (0, 1):
        rows = [
            {
                "request_id": "s",
                "generation_id": "current",
                "dispatch_id": "s_d",
                "rank": rank,
                "stage": "DiT",
                "cuda_status": "not_requested",
            }
        ]
        (tmp_path / f"stage-timing-pid{rank}.jsonl").write_text(
            json.dumps({**rows[0], "generation_id": "previous", "dispatch_id": "s_old"})
            + "\n"
            + json.dumps(rows[0])
            + "\n"
        )
        timing.publish_health(
            tmp_path / f"stage-timing-pid{rank}-health.json",
            {
                "profile": {
                    "request_id": "s",
                    "generation_id": "current",
                    "rank": rank,
                    "pid": rank,
                    "dispatches": {"s_d": 1},
                }
            },
        )
    profile = asyncio.run(
        timing.collect_profile(
            tmp_path, "s", [0, 1], 1, generation_id="current", timeout=0
        )
    )
    assert profile["status"] == "complete"
    assert len(profile["stages"]) == 2
    assert {r["rank"] for r in profile["stages"]} == {0, 1}
    (tmp_path / "stage-timing-pid1-health.json").unlink()
    profile = asyncio.run(
        timing.collect_profile(
            tmp_path, "s", [0, 1], 1, generation_id="current", timeout=0
        )
    )
    assert profile["status"] == "incomplete"
    assert profile["missing_ranks"] == [1]


def test_profile_does_not_accept_a_finished_marker_with_missing_spans(tmp_path):
    import asyncio

    (tmp_path / "stage-timing-pid0.jsonl").write_text("")
    timing.publish_health(
        tmp_path / "stage-timing-pid0-health.json",
        {"profile": {"request_id": "s", "rank": 0, "pid": 0, "dispatches": {"s_d": 1}}},
    )
    profile = asyncio.run(timing.collect_profile(tmp_path, "s", [0], 1, timeout=0))
    assert profile["status"] == "incomplete"
    assert profile["health"][0]["dropped"] == 1


def test_streaming_report_checks_server_ranges_and_collector_health(tmp_path):
    from build_streaming_report import load_server_evidence

    directory = tmp_path / "timing"
    directory.mkdir()
    hardware = {
        "active_roles": {"dit_decoder": [0, 1], "source_encoder": [2]},
        "devices": [],
    }
    (tmp_path / "hardware.json").write_text(json.dumps(hardware))
    base = {
        "request_id": "session",
        "dispatch_id": "dispatch",
        "dispatch_id_status": "present",
        "status": "ok",
        "cuda_status": "complete",
        "cuda_stream_ms": 0,
        "host_call_ms": 1,
        "clock_domain": "CLOCK_MONOTONIC:test",
    }
    workers = [
        [
            dict(
                base,
                rank=0,
                stage="Encoder",
                component="source_vae",
                worker_role="dedicated_source_encoder",
                latent_range=[0, 1],
            )
        ],
        *[
            [
                dict(base, rank=rank, stage="DiT", output_latent_range=[0, 1]),
                dict(base, rank=rank, stage="Decoder", output_frame_range=[0, 1]),
                dict(
                    base,
                    rank=rank,
                    stage="Request",
                    component="continuous_source_complete",
                ),
            ]
            for rank in range(2)
        ],
    ]
    for pid, rows in enumerate(workers):
        path = directory / f"stage-timing-pid{pid}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        path.with_name(path.stem + "-health.json").write_text(
            json.dumps({"written": len(rows), "dropped": 0, "pending": 0, "queued": 0})
        )
    bundles = [
        {
            "config": {"cases": [{"label": "case", "frames": 1}]},
            "requests": [
                {
                    "label": "case",
                    "success": True,
                    "server_session_id": "session",
                    "phase": "measured",
                    "index": 0,
                }
            ],
        }
    ]
    server = load_server_evidence(tmp_path, bundles)
    assert server["validation"]["collector_complete"]
    assert server["validation"]["sessions"]["session"]["complete"]
    assert all(row["cuda_stream_ms"] == 0 for row in server["spans"])
    # A rank whose frame range starts too late must not be certified complete.
    workers[2][1]["output_frame_range"] = [1, 2]
    (directory / "stage-timing-pid2.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in workers[2])
    )
    server = load_server_evidence(tmp_path, bundles)
    assert not server["validation"]["sessions"]["session"]["complete"]
    snapshot = directory / "stage-timing-pid0-health.json"
    snapshot.write_text(json.dumps({"written": 1, "dropped": 0, "pending": 1}))
    assert not load_server_evidence(tmp_path, bundles)["validation"][
        "collector_complete"
    ]


def test_request_files_can_be_removed_without_affecting_next_session(tmp_path):
    writer = timing.Writer(tmp_path)
    try:
        for request_id in ("first", "next"):
            trace = timing.StageTrace(
                request_id, request_id + "_d", writer.submit, identity={"rank": 0}
            )
            with trace.span("DiT") as span:
                pass
            span.handoff()
            trace.finish()
        deadline = time.monotonic() + 2
        while writer.written != 2 or not writer.queue.empty():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        timing.remove_profile_files(tmp_path, "first")
        assert not list(
            tmp_path.glob(f"*-session{timing.session_file_tag('first')}.jsonl")
        )
        profile = __import__("asyncio").run(
            timing.collect_profile(tmp_path, "next", [0], 1, timeout=1)
        )
        assert len(profile["stages"]) == 1
        assert profile["stages"][0]["request_id"] == "next"
        assert profile["stages"][0]["cuda_status"] == "not_requested"
    finally:
        writer.stop.set()
        writer.thread.join(2)


def test_continuous_host_profile_includes_dedicated_encoder(tmp_path):
    import asyncio

    tag = timing.session_file_tag("session")
    for pid, rows in [
        (1, [dict(stage="Encoder", worker_role="dedicated_source_encoder", rank=0)]),
        (2, [dict(stage="Request", component="continuous_source_complete", rank=0)]),
        (3, [dict(stage="Request", component="continuous_source_complete", rank=1)]),
    ]:
        path = tmp_path / f"stage-timing-pid{pid}-session{tag}.jsonl"
        path.write_text(
            "".join(
                json.dumps(dict(row, request_id="session", status="ok", host_call_ms=1))
                + "\n"
                for row in rows
            )
        )
        path.with_name(f"stage-timing-pid{pid}-health.json").write_text(
            json.dumps(dict(pending=0, queued=0, error=None))
        )
    profile = asyncio.run(
        timing.collect_continuous_profile(tmp_path, "session", [0, 1], timeout=0)
    )
    assert profile["status"] == "complete"
    assert len(profile["stages"]) == 3
    missing = asyncio.run(
        timing.collect_continuous_profile(tmp_path, "session", [0, 1, 2], timeout=0)
    )
    assert missing["status"] == "incomplete"


def test_request_flag_controls_capture_even_with_legacy_cuda_env(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_STAGE_TIMING", "cuda")
    monkeypatch.setattr(
        timing,
        "get_writer",
        lambda *args: pytest.fail("Ordinary request started a writer"),
    )
    assert (
        timing.trace_for_batch(SimpleNamespace(extra={"return_stage_timing": False}))
        is None
    )
    assert timing.trace_for_ids("session", "dispatch") is None


def test_requested_host_trace_uses_automatic_shared_directory(monkeypatch):
    import subprocess
    import sys

    monkeypatch.delenv("SGLANG_ZING_STAGE_TIMING", raising=False)
    monkeypatch.delenv("SGLANG_ZING_STAGE_TIMING_DIR", raising=False)
    with timing.deployment_timing_directory() as directory:
        with timing.deployment_timing_directory() as nested:
            assert nested == directory
        child_directory = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import os;print(os.environ['SGLANG_ZING_STAGE_TIMING_DIR'])",
            ],
            text=True,
        ).strip()
        assert Path(child_directory) == directory
        trace = timing.trace_for_batch(
            SimpleNamespace(
                realtime_session_id="s",
                request_id="d",
                extra={"return_stage_timing": True},
            )
        )
        assert trace.directory == str(directory)
        assert trace.identity["mode"] == "host" and not trace.measure_cuda
        trace.event_factory = lambda: pytest.fail(
            "Host request constructed CUDA events"
        )
        with trace.span("DiT") as span:
            pass
        span.handoff()
        trace.finish()
    assert not directory.exists()
    assert timing._WRITER is None


def test_directory_override_is_a_base_and_survives_failed_launch(monkeypatch, tmp_path):
    import os

    sentinel = tmp_path / "keep.json"
    sentinel.write_text("original")
    monkeypatch.setenv("SGLANG_ZING_STAGE_TIMING_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="Launch failed"):
        with timing.deployment_timing_directory() as directory:
            assert directory.parent == tmp_path
            (directory / "partial.jsonl").write_text("partial")
            raise RuntimeError("Launch failed")
    assert not directory.exists()
    assert sentinel.read_text() == "original"
    assert os.environ["SGLANG_ZING_STAGE_TIMING_DIR"] == str(tmp_path)


def test_encoder_writer_can_follow_a_new_deployment_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(timing, "_WRITER", None)
    first = timing.get_writer(tmp_path / "first")
    second = timing.get_writer(tmp_path / "second")
    try:
        assert not first.thread.is_alive()
        assert second.path.parent == tmp_path / "second"
        assert first is not second
    finally:
        second.close()
        timing._WRITER = None


@pytest.mark.parametrize("acknowledge", [True, False, "timeout"])
def test_stage_profile_ack_or_disconnect_removes_only_delivered_session(
    monkeypatch, tmp_path, acknowledge
):
    import asyncio

    import msgspec.msgpack
    from starlette.websockets import WebSocketDisconnect

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
        realtime_video_api,
    )
    from sglang.multimodal_gen.runtime.utils import zing_stage_timing as timing

    monkeypatch.setenv("SGLANG_ZING_STAGE_TIMING", "host")
    monkeypatch.setenv("SGLANG_ZING_STAGE_TIMING_DIR", str(tmp_path))
    monkeypatch.setattr(
        realtime_video_api,
        "get_global_server_args",
        lambda: SimpleNamespace(num_gpus=2),
    )
    for sid in ("current", "next"):
        (
            tmp_path / f"stage-timing-pid1-session{timing.session_file_tag(sid)}.jsonl"
        ).write_text("{}\n")

    async def collect(*args, **kwargs):
        return dict(
            type="profile_result",
            session_id="current",
            status="complete",
            stages=[],
            health=[],
        )

    monkeypatch.setattr(timing, "collect_profile", collect)

    class Socket:
        async def send_bytes(self, wire):
            assert msgspec.msgpack.decode(wire)["session_id"] == "current"
            assert len(list(tmp_path.glob("*.jsonl"))) == 2

        async def receive_bytes(self):
            if acknowledge == "timeout":
                raise asyncio.TimeoutError()
            if not acknowledge:
                raise WebSocketDisconnect()
            return msgspec.msgpack.encode(
                dict(type="profile_ack", session_id="current")
            )

    asyncio.run(realtime_video_api._send_stage_profile(Socket(), "current", 1))
    assert not list(
        tmp_path.glob(f"*-session{timing.session_file_tag('current')}.jsonl")
    )
    assert (
        len(list(tmp_path.glob(f"*-session{timing.session_file_tag('next')}.jsonl")))
        == 1
    )


@pytest.mark.parametrize(
    "close_result", [(1001, "cancelled"), (1011, "generation failed"), None]
)
def test_aborted_profile_files_are_removed_after_worker_cleanup(
    monkeypatch, tmp_path, close_result
):
    import asyncio

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
        realtime_video_api,
    )
    from sglang.multimodal_gen.runtime.utils import zing_stage_timing as timing

    monkeypatch.setenv("SGLANG_ZING_STAGE_TIMING_DIR", str(tmp_path))
    path = (
        tmp_path
        / f"stage-timing-pid1-session{timing.session_file_tag('current')}.jsonl"
    )
    path.write_text("partial")
    next_path = (
        tmp_path / f"stage-timing-pid1-session{timing.session_file_tag('next')}.jsonl"
    )
    next_path.write_text("next")
    session = SimpleNamespace(
        id="current",
        next_chunk_index=1,
        generation_id="g",
        request=SimpleNamespace(return_stage_timing=True, source_stream=None),
    )
    cleaned = []

    async def cleanup(*args):
        assert path.exists()
        cleaned.append(True)

    async def forbidden_profile(*args):
        pytest.fail("Aborted session attempted profile delivery")

    class Socket:
        async def close(self, **kwargs):
            assert cleaned and not path.exists()

    monkeypatch.setattr(realtime_video_api, "_cleanup_realtime_session", cleanup)
    monkeypatch.setattr(realtime_video_api, "_send_stage_profile", forbidden_profile)
    asyncio.run(
        realtime_video_api._cleanup_realtime_session_then_close(
            Socket(), session, None, None, close_result
        )
    )
    assert cleaned and not path.exists() and next_path.exists()


@pytest.mark.parametrize("entrypoint", ["launch_server", "dispatch_launch"])
@pytest.mark.parametrize("fail", [False, True])
def test_serving_entrypoints_own_timing_directory(monkeypatch, entrypoint, fail):
    import ast
    import os
    from pathlib import Path

    # Execute the actual launcher wrappers without importing GPU worker dependencies.
    module = ast.parse((PATH.parents[1] / "launch_server.py").read_text())
    wrapper = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == entrypoint
    )
    scope = {"ServerArgs": object}

    monkeypatch.delenv("SGLANG_ZING_STAGE_TIMING", raising=False)
    monkeypatch.delenv("SGLANG_ZING_STAGE_TIMING_DIR", raising=False)
    directories = []

    def launch(*args, **kwargs):
        directory = Path(os.environ["SGLANG_ZING_STAGE_TIMING_DIR"])
        assert directory.is_dir()
        directories.append(directory)
        (directory / "partial.jsonl").write_text("partial")
        if fail:
            raise RuntimeError("Launch failed")

    scope["_" + entrypoint] = launch
    exec(
        compile(
            ast.Module(body=[wrapper], type_ignores=[]),
            str(PATH.parents[1] / "launch_server.py"),
            "exec",
        ),
        scope,
    )
    if fail:
        with pytest.raises(RuntimeError, match="Launch failed"):
            scope[entrypoint](SimpleNamespace())
    else:
        scope[entrypoint](SimpleNamespace())
    assert not directories[0].exists()
    assert "SGLANG_ZING_STAGE_TIMING_DIR" not in os.environ


@pytest.mark.parametrize("publication", ["missing-health", "partial-line"])
def test_continuous_collector_retries_in_progress_publication(tmp_path, publication):
    import asyncio

    row = dict(
        request_id="s",
        rank=0,
        stage="Request",
        component="continuous_source_complete",
        status="ok",
        host_call_ms=1,
    )
    path = tmp_path / f"stage-timing-pid1-session{timing.session_file_tag('s')}.jsonl"
    health = tmp_path / "stage-timing-pid1-health.json"
    path.write_text(
        json.dumps(row) + "\n" if publication == "missing-health" else '{"request_id":'
    )
    if publication == "partial-line":
        health.write_text(json.dumps(dict(pending=0, queued=0)))

    async def collect():
        task = asyncio.create_task(
            timing.collect_continuous_profile(tmp_path, "s", [0], timeout=1)
        )
        await asyncio.sleep(0.02)
        assert not task.done()
        path.write_text(json.dumps(row) + "\n")
        health.write_text(json.dumps(dict(pending=0, queued=0)))
        result = await task
        assert result["status"] == "complete" and result["stages"] == [row]

    asyncio.run(collect())
