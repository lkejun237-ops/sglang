import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.inference_timing import (
    public_timing,
    trigger_source,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.recording import Recording
from sglang.multimodal_gen.runtime.utils.zing_ttff_timing import (
    InferenceTiming,
    TimingBuffer,
)


def test_delayed_cuda_updates_without_synchronizing():
    class Event:
        ready = False

        def record(self):
            pass

        def query(self):
            return self.ready

        def elapsed_time(self, other):
            return 12.5

        def synchronize(self):
            raise AssertionError("Telemetry must never synchronize")

    end = Event()
    timing = InferenceTiming(
        "dit",
        event_id=4,
        prompt_version=3,
        device=SimpleNamespace(type="cuda"),
        event_factory=lambda: (Event(), end),
    )
    with timing.measure("joint_step0", cuda=True):
        pass
    buffer = TimingBuffer()
    buffer.add(timing)
    first = buffer.drain()[0]
    assert first["status"] == "pending"
    assert first["stages"][0]["device_elapsed_ms"] is None
    assert buffer.drain() == []
    end.ready = True
    second = buffer.drain()[0]
    assert second["report_id"] == first["report_id"]
    assert second["revision"] > first["revision"]
    assert second["status"] == "complete"
    assert second["stages"][0]["device_elapsed_ms"] == 12.5
    assert first["stages"][0]["device_elapsed_ms"] is None
    assert buffer.drain() == []


def test_final_pending_is_incomplete_and_skipped_is_not_zero():
    event = SimpleNamespace(record=lambda: None, query=lambda: False)
    timing = InferenceTiming(
        "encoder",
        prompt_version=1,
        device=SimpleNamespace(type="cuda"),
        event_factory=lambda: (event, event),
    )
    with timing.measure("source_vae", cuda=True):
        pass
    timing.skip("reference_vae")
    buffer = TimingBuffer()
    buffer.add(timing)
    value = buffer.drain(final=True)[0]
    assert value["status"] == "incomplete"
    assert value["stages"][0]["device_status"] == "unresolved"
    assert value["stages"][1]["host_ms"] is None
    assert not buffer.pending


def test_public_timing_bounds_and_private_fields():
    timing = InferenceTiming("dit", prompt_version=2, event_id=7)
    timing.record("history_kv_rebuild", float("nan"))
    timing.finished = True
    raw = dict(timing.snapshot(), prompt="SECRET", tensors={"secret": 1})
    public = public_timing(raw)
    assert "SECRET" not in str(public)
    assert public["stages"][0]["host_ms"] is None
    assert public_timing(public) == public
    assert public_timing(dict(raw, prompt_version=True)) is None
    assert public_timing(dict(raw, stages=[{}] * 129)) is None
    assert (
        trigger_source({"periodic_refresh": True, "request_id": "x"})
        == "periodic_refresh"
    )
    assert trigger_source({"source_context": {}, "request_id": "x"}) == "world_context"


def test_archive_and_export_keep_late_updates_and_unknown_cases(tmp_path):
    root = Path(__file__).resolve().parents[6]
    spec = importlib.util.spec_from_file_location(
        "summarize_ttff", root / "benchmark/whitebox/summarize_ttff.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recording = Recording.__new__(Recording)
    recording.directory, recording.manifest = tmp_path, {}
    timing = InferenceTiming("dit", prompt_version=2, event_id=17)
    timing.record("history_kv_rebuild", 200)
    timing.finished = True
    report = timing.snapshot()
    asyncio.run(
        recording.write_inference_timing(dict(report, revision=1, status="pending"))
    )
    asyncio.run(
        recording.write_inference_timing(
            dict(report, revision=2, trigger_source="user_control")
        )
    )
    for version, stage, elapsed in (
        (1, "segment_setup_to_source_dispatch", 400),
        (2, "sink_idle_wait", 20),
    ):
        room = InferenceTiming("room", prompt_version=version)
        room.record(stage, elapsed)
        room.finished = True
        asyncio.run(recording.write_inference_timing(room.snapshot()))
    manifest = dict(
        recording.manifest,
        recording_id="case",
        model_identity={"generation_id": "g"},
        condition_switches=[{"prompt_version": 2}, {"prompt_version": 3}],
        model_cache_receipts=[{"prompt_version": 2, "cache_result": "l1_hit"}],
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    rows, stages = module.read_case(tmp_path, "prod-shanghai")
    assert len(rows) == 3 and len(stages) == 3
    assert rows[0]["segment_setup_to_source_dispatch_ms"] == 400
    assert rows[1]["sink_idle_wait_ms"] == 20
    assert rows[1]["timing_status"] == "incomplete"  # Other roles are absent.
    assert rows[1]["cache_result"] == "l1_hit"
    assert rows[2]["cache_result"] == "not_recorded"
    assert rows[2]["timing_status"] == "not_recorded"
    summary = module.summarize(rows)
    assert summary["prompt_switch"]["cache_hit_rate"] == 1
    assert (
        summary["first_frame"]["latency_ms"]["segment_setup_to_source_dispatch"]["p50"]
        == 400
    )
    assert summary["prompt_switch"]["latency_ms"]["sink_idle_wait"]["n"] == 1


@pytest.mark.parametrize("timing_first", [False, True])
def test_optional_timing_never_consumes_required_queue_or_sequences(timing_first):
    from collections import deque

    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncRecording,
    )

    async def run():
        record = object.__new__(AsyncRecording)
        record.manifest = {}
        record.accepting = True
        record.rejected = 0
        record.items, record.timing_items = deque(), deque()
        record.pending = None
        record.pending_bytes = record.timing_bytes = 0
        record.failure = None
        record.manager = SimpleNamespace(max_age=10, queue_depth=1, queue_bytes=1)
        record.wake = asyncio.Event()
        record._offered_sequence = 0
        if timing_first:
            await record.write_inference_timing({"revision": 1})
        assert record.offer("pair", "video", size=1)
        if not timing_first:
            await record.write_inference_timing({"revision": 1})
        assert record.accepting and record.failure is None
        assert record.items[0][2] == "pair"
        assert record._offered_sequence == 1 and record.pending_bytes == 1
        assert len(record.timing_items) == 1
        for revision in range(record.TIMING_QUEUE_DEPTH + 1):
            await record.write_inference_timing({"revision": revision + 2})
        assert len(record.timing_items) == record.TIMING_QUEUE_DEPTH
        assert record.manifest["inference_timing"]["incomplete"]
        record.timing_items.clear()
        record.timing_bytes = 0
        await record.write_inference_timing({"revision": 99})
        assert record.manifest["inference_timing"]["incomplete"]
        assert record._offered_sequence == 1

    asyncio.run(run())


@pytest.fixture
def timing_registry(monkeypatch):
    from prometheus_client import CollectorRegistry, Histogram, generate_latest

    from sglang.multimodal_gen.runtime.realtime import critical_path_metrics as metrics

    registry = CollectorRegistry()
    histogram = Histogram(
        "world_model_critical_path_stage_duration_seconds",
        "Test registry",
        metrics.CRITICAL_PATH_LABELS,
        buckets=metrics.CRITICAL_PATH_BUCKETS,
        registry=registry,
    )
    metrics._critical_path_child.cache_clear()
    monkeypatch.setattr(metrics, "CRITICAL_PATH_STAGE_SECONDS", histogram)
    monkeypatch.setattr(metrics, "prometheus_latest", lambda: generate_latest(registry))
    yield registry
    metrics._critical_path_child.cache_clear()


def metric_values(registry, suffix="count"):
    return {
        sample.labels["stage"]: sample.value
        for metric in registry.collect()
        for sample in metric.samples
        if sample.name.endswith("_" + suffix)
    }


def test_metrics_wait_for_cuda_and_deduplicate_revisions(timing_registry):
    from sglang.multimodal_gen.runtime.realtime.whitebox.ttff_metrics import (
        TimingMetrics,
    )

    event = SimpleNamespace(record=lambda: None, query=lambda: False)
    event.elapsed_time = lambda other: 125
    timing = InferenceTiming(
        "dit",
        prompt_version=2,
        device=SimpleNamespace(type="cuda"),
        event_factory=lambda: (event, event),
    )
    with timing.measure("history_kv_rebuild", cuda=True):
        pass
    timing.skip("condition_activate")
    timing.record("user_supplied_label", 123)
    timing.finished = True
    metrics = TimingMetrics()
    pending = timing.snapshot()
    metrics.observe(pending, trigger="user_control")
    metrics.observe(dict(pending, status="incomplete"), trigger="user_control")
    assert metric_values(timing_registry) == {}
    event.query = lambda: True
    complete = timing.snapshot()
    for report in (complete, complete, pending, dict(complete, revision=999)):
        metrics.observe(report, trigger="user_control")
    assert metric_values(timing_registry) == {
        "whitebox_ttff_prompt_switch_dit_host_history_kv_rebuild": 1,
        "whitebox_ttff_prompt_switch_dit_cuda_history_kv_rebuild": 1,
    }
    assert (
        metric_values(timing_registry, "sum")[
            "whitebox_ttff_prompt_switch_dit_cuda_history_kv_rebuild"
        ]
        == 0.125
    )


def test_metrics_missing_values_and_bounded_replay(timing_registry):
    from sglang.multimodal_gen.runtime.realtime.whitebox.ttff_metrics import (
        TimingMetrics,
    )

    metrics = TimingMetrics()
    for version in range(1, 20):
        timing = InferenceTiming("room", prompt_version=version)
        timing.record("source_dispatch_to_first_rgb", 1000)
        timing.record("room_accept_to_first_rgb", float("nan"))
        timing.skip("sink_idle_wait")
        timing.finished = True
        report = timing.snapshot()
        metrics.observe(report, trigger="periodic_refresh")
        metrics.observe(report, trigger="unknown")
        metrics.observe(report, trigger="initial" if version == 1 else "user_control")
        if version == 1:
            first = report
    before = metric_values(timing_registry)
    metrics.observe(first, trigger="initial")  # Pruned IDs cannot be re-counted.
    assert metric_values(timing_registry) == before
    assert before == {
        "whitebox_ttff_initial_room_host_source_dispatch_to_first_rgb": 1,
        "whitebox_ttff_prompt_switch_room_host_source_dispatch_to_first_rgb": 18,
    }
    assert len(metrics.seen) == metrics.VERSION_WINDOW
    for _ in range(metrics.REPORTS_PER_VERSION + 10):
        timing = InferenceTiming("room", prompt_version=19)
        timing.finished = True
        metrics.observe(timing.snapshot(), trigger="user_control")
    assert len(metrics.seen[19]) == metrics.REPORTS_PER_VERSION


def test_room_timing_metrics_reach_http_without_recording(timing_registry):
    import httpx

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import create_app
    from sglang.multimodal_gen.runtime.realtime.whitebox.ttff_metrics import (
        TimingMetrics,
    )

    async def run():
        segment = object.__new__(RenderSegment)
        segment.detailed_timing = True
        from sglang.multimodal_gen.runtime.realtime.whitebox.timing_control import (
            TimingControl,
        )

        segment.timing_control = TimingControl(segment)
        segment.timing_metrics = TimingMetrics()
        segment.timing_triggers = {1: "initial", 2: "user_control"}
        segment.recording = None
        segment.assignment = SimpleNamespace(session_id="s", generation_id="g")
        timing = InferenceTiming("encoder", prompt_version=1)
        timing.record("qwen_prepare", 650)
        timing.finished = True
        message = dict(
            type="model.timing",
            model_session_id="s",
            generation_id="wrong",
            timing=timing.snapshot(),
        )
        await segment.model_control(message)
        assert not metric_values(timing_registry)
        message["generation_id"] = "g"
        await segment.model_control(message)
        await segment.model_control(message)
        for version, stage in (
            (1, "segment_setup_to_first_rgb"),
            (2, "room_accept_to_first_rgb"),
        ):
            room = InferenceTiming("room", prompt_version=version)
            room.record(stage, 1250)
            room.finished = True
            await segment._archive_timing(room.snapshot())
        app = create_app(SimpleNamespace(), "test-token")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://room"
        ) as client:
            response = await client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert (
            'stage="whitebox_ttff_initial_encoder_host_qwen_prepare"' in response.text
        )
        assert (
            'stage="whitebox_ttff_prompt_switch_room_host_room_accept_to_first_rgb"'
            in response.text
        )
        assert "report_id=" not in response.text
        assert all(count == 1 for count in metric_values(timing_registry).values())
        assert (
            metric_values(timing_registry, "sum")[
                "whitebox_ttff_initial_room_host_segment_setup_to_first_rgb"
            ]
            == 1.25
        )

    asyncio.run(run())


def test_ttff_metric_allowlist_and_collector_budget(timing_registry):
    from sglang.multimodal_gen.runtime.realtime.critical_path_metrics import (
        observe_stage_ms,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.telemetry import (
        CLI_STARTUP_STAGES,
        TTFF_METRIC_STAGES,
    )

    for stage in TTFF_METRIC_STAGES | CLI_STARTUP_STAGES:
        assert observe_stage_ms(stage, 250, service="room_runtime", model="zing_h3")
    samples = [
        sample for metric in timing_registry.collect() for sample in metric.samples
    ]
    # Existing all-in-one collector accepts at most 4000 samples per target.
    # Leave room for the pre-existing Room histogram/recording series.
    assert len(samples) < 2800
    assert all(
        len(value) <= 100 for sample in samples for value in sample.labels.values()
    )
    assert not observe_stage_ms("whitebox_ttff_user_controlled", 1)


def test_room_admission_clock_rejects_recovery_and_invalid_values():
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_timing import (
        room_admission,
        room_elapsed_ms,
    )

    marker = room_admission()
    assert room_elapsed_ms(marker, marker["at"] + 2) == 2000
    assert (
        room_elapsed_ms(dict(marker, clock_id="previous-process"), marker["at"] + 2)
        is None
    )
    for bad in (None, True, float("nan"), float("inf")):
        assert room_elapsed_ms(dict(marker, at=bad), marker["at"] + 2) is None
    assert room_elapsed_ms(marker, marker["at"] - 1) is None


@pytest.mark.parametrize("ready_state", ["kv_ready", "source_ready"])
def test_room_prompt_path_uses_receipt_arrival_and_closes_to_first_rgb(
    monkeypatch, tmp_path, timing_registry, ready_state
):
    from sglang.multimodal_gen.runtime.realtime.whitebox import (
        inference_timing,
        render_segment,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    clock = SimpleNamespace(now=10.0)

    def monotonic():
        return clock.now

    monkeypatch.setattr(
        inference_timing, "time", SimpleNamespace(perf_counter=monotonic)
    )
    monkeypatch.setattr(
        render_segment,
        "time",
        SimpleNamespace(perf_counter=monotonic, monotonic=monotonic),
    )
    monkeypatch.setattr(
        render_segment,
        "InferenceTiming",
        lambda *a, **kw: InferenceTiming(*a, clock=monotonic, **kw),
    )

    async def run():
        item = dict(
            version=2,
            request_id="replace-2",
            text="scene",
            sha256="a" * 64,
            effective={"source_frame_index": 17},
            room_admission=inference_timing.room_admission(),
        )
        frame = SimpleNamespace(index=17)

        async def latest(_):
            clock.now = 12.0
            return 1, (frame, 0)

        owner = SimpleNamespace(
            journal=SimpleNamespace(
                snapshot={
                    "segment": dict(
                        segment_id="segment",
                        state="streaming",
                        applied_version=1,
                        prompt_queue=[item],
                    )
                }
            ),
            latest=SimpleNamespace(get=latest, version=25),
            segment_prompt_output=AsyncMock(),
            source_identity=lambda: dict(
                source_session_id="source", source_epoch="epoch"
            ),
        )

        async def control(segment, message, **kwargs):
            return await SourceSession.segment_model_control(
                owner, segment, message, **kwargs
            )

        owner.segment_model_control = control
        segment = object.__new__(render_segment.RenderSegment)
        segment.owner = owner
        segment.deferred_kv_ready = ready_state == "source_ready"
        segment.segment_id, segment.segment_epoch = "segment", "segment-epoch"
        from sglang.multimodal_gen.runtime.realtime.whitebox.ttff_metrics import (
            TimingMetrics,
        )

        segment.detailed_timing = True
        from sglang.multimodal_gen.runtime.realtime.whitebox.timing_control import (
            TimingControl,
        )

        segment.timing_control = TimingControl(segment)
        segment.timing_metrics = TimingMetrics()
        segment.room_timings, segment.timing_triggers = {}, {}
        segment.closed_event = asyncio.Event()
        segment.source_admission_ready, segment.prompt_available = (
            asyncio.Event(),
            asyncio.Event(),
        )
        segment.source_admission_ready.set()
        segment.prompt_available.set()
        idle = asyncio.Event()
        idle.set()
        segment.sink = SimpleNamespace(
            idle=idle,
            control=AsyncMock(return_value=False),
            output=AsyncMock(return_value=None),
        )
        segment.pending_prompt = segment.condition_switch = None
        segment.active_reference_descriptors = []
        segment.source_send_lock, segment.model_wire_lock = (
            asyncio.Lock(),
            asyncio.Lock(),
        )
        segment.model_admission_condition = asyncio.Condition()
        segment.source_view = SimpleNamespace(stopped=asyncio.Event())
        segment.cancel_requested = False
        segment.selected_frames = 17
        segment.last_selected_source = None
        segment.prompt_by_event = {}
        segment.model_wire_sequence = 0
        segment.config = SimpleNamespace(startup_seconds=1)
        segment.first_output_at = None
        segment.periodic_refresh = None
        segment.assignment = None
        segment.recording = Recording.__new__(Recording)
        segment.recording.directory, segment.recording.manifest = tmp_path, {}

        async def send(_):
            event_id = segment.pending_prompt["event_id"]
            for at, state in ((14.2, "accepted"), (14.8, ready_state)):
                clock.now = at
                await segment.model_control(
                    dict(
                        type="control_ack",
                        kind="prompt.update",
                        state=state,
                        event_id=event_id,
                        source_frame_index=17,
                        prompt_version=2,
                    )
                )
            # Both receipts arrived before the send coroutine returns. Counting
            # the later await-resume timestamp would hide their actual interval.
            clock.now = 15.0

        await segment.model_wire_lock.acquire()
        clock.now = 11.0
        task = asyncio.create_task(
            segment._condition_controls(SimpleNamespace(send=send))
        )
        try:
            async with asyncio.timeout(1):
                while segment.condition_switch is None:
                    await asyncio.sleep(0)
                clock.now = 14.0
                segment.model_wire_lock.release()
                while (
                    segment.pending_prompt is None
                    or segment.pending_prompt.get("timing_resumed_at") is None
                ):
                    if task.done():
                        task.result()
                    await asyncio.sleep(0)
            clock.now = 17.0
            segment.model_source_dispatch(17, 17.0)
            segment.model_source_dispatch(18, 18.0)
            clock.now = 20.0
            await segment.model_output_condition(
                dict(
                    event_id=segment.pending_prompt["event_id"], output_frame_start=17
                ),
                {17: (frame, 0)},
                None,
                {},
            )
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        report = json.loads((tmp_path / "inference-timing.jsonl").read_text())
        values = {v["stage"]: v["host_ms"] for v in report["stages"]}
        assert values["room_accept_to_condition_dequeue"] == 1000
        assert values["model_wire_lock_wait"] == 2000
        assert values["room_accept_to_model_send"] == 4000
        assert values["prompt_send_to_accepted"] == pytest.approx(200)
        assert values["accepted_to_condition_ready"] == pytest.approx(600)
        assert values["condition_ready_to_resume"] == pytest.approx(200)
        assert values["resume_to_fresh_source_dispatch"] == 2000
        assert values["source_dispatch_to_first_rgb"] == 3000
        assert (
            sum(
                row["stage"] == "resume_to_fresh_source_dispatch"
                for row in report["stages"]
            )
            == 1
        )
        assert values["room_accept_to_first_rgb"] == 10000
        assert (
            metric_values(timing_registry)[
                "whitebox_ttff_prompt_switch_room_host_room_accept_to_first_rgb"
            ]
            == 1
        )
        assert metric_values(timing_registry, "sum")[
            "whitebox_ttff_prompt_switch_room_host_accepted_to_condition_ready"
        ] == pytest.approx(0.6)
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                dict(
                    recording_id="case",
                    condition_switches=[
                        {"prompt_version": 2, "readiness_state": ready_state}
                    ],
                )
            )
        )
        root = Path(__file__).resolve().parents[6]
        spec = importlib.util.spec_from_file_location(
            "summarize_ttff", root / "benchmark/whitebox/summarize_ttff.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows, _ = module.read_case(tmp_path, "test-shanghai-copy")
        row = rows[1]
        assert row["readiness_state"] == ready_state
        assert row["room_path_status"] == "complete"
        assert row["room_path_closure_error_ms"] == pytest.approx(0)
        assert row["client_send_to_room_accept_ms"] is None
        stats = module.summarize(rows)["prompt_switch"]["latency_ms"]
        assert stats["room_accept_to_first_rgb"]["p50"] == 10000
        assert stats["client_send_to_room_accept"]["n"] == 0
        assert stats["client_send_to_room_accept"]["missing"] == 1

    asyncio.run(run())


def test_basic_ttff_remains_observable_without_detailed_reports(timing_registry):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.ttff_metrics import (
        TimingMetrics,
    )

    async def run():
        segment = object.__new__(RenderSegment)
        segment.detailed_timing = False
        from sglang.multimodal_gen.runtime.realtime.whitebox.timing_control import (
            TimingControl,
        )

        segment.timing_control = TimingControl(segment)
        segment.timing_metrics = TimingMetrics()
        segment.timing_triggers = {1: "initial", 2: "user_control"}
        segment.recording = SimpleNamespace(write_inference_timing=AsyncMock())
        for version, total in [
            (1, "segment_setup_to_first_rgb"),
            (2, "room_accept_to_first_rgb"),
        ]:
            timing = segment._new_room_timing(version)
            timing.record("model_wire_lock_wait", 500)
            timing.record(total, 1500)
            timing.finished = True
            assert [s["stage"] for s in timing.snapshot()["stages"]] == [total]
            await segment._archive_timing(timing.snapshot())
        assert len(metric_values(timing_registry)) == 2
        await segment.model_control(
            {"type": "model.timing", "timing": {"untrusted": True}}
        )
        segment.recording.write_inference_timing.assert_not_awaited()

    asyncio.run(run())


def test_filtered_measurement_never_creates_cuda_events():
    def forbidden():
        raise AssertionError("Disabled measurements must not allocate CUDA events")

    timing = InferenceTiming(
        "room",
        allowed_stages={"segment_setup_to_first_rgb"},
        device=SimpleNamespace(type="cuda"),
        event_factory=forbidden,
    )
    with timing.measure("qwen_prepare", cuda=True):
        pass
    assert timing.stages == [] and timing.pending == []


def test_room_dynamic_timing_revisions_identity_idempotence_and_cancellation():
    from test_whitebox_render_contract import command
    from test_whitebox_sink_control import fixture, until

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.timing_control import (
        TimingControl,
    )

    async def run():
        sink, owner, scope, sent, receipts, tasks = await fixture()
        segment = sink.segment
        segment.detailed_timing = False
        segment.assignment = SimpleNamespace(
            session_id="model", generation_id="generation"
        )
        segment.recording = SimpleNamespace(manifest={})
        control = segment.timing_control = TimingControl(segment)
        control.supported = True
        for revision, enabled in enumerate([True, False, True], 1):
            request = command(
                "model.timing.configure",
                f"timing-{revision}",
                **scope,
                enabled=enabled,
                expected_timing_revision=revision - 1,
            )
            await control.command(request)
            await until(lambda: len(sent) == revision)
            assert control.revision == revision - 1  # accepted isn't applied
            ack = dict(
                type="control_ack",
                kind="timing.configure",
                event_id=sent[-1]["event_id"],
                model_session_id="model",
                generation_id="stale",
                enabled=enabled,
                timing_revision=revision,
                boundaries=dict(
                    encoder=dict(command_index=3),
                    dit=dict(source_frame_end=17),
                    decoder=dict(output_frame_start=9),
                ),
            )
            assert await control.control(ack)
            assert control.revision == revision - 1
            ack["generation_id"] = "generation"
            await control.control(ack)
            assert control.revision == revision and segment.detailed_timing is enabled
            assert receipts[-1]["state"] == "completed"
            await control.command(request)  # journal duplicate sends nothing
            assert len(sent) == revision and receipts[-1]["duplicate"]
        assert control.ever_enabled
        with pytest.raises(RenderError) as error:
            await control.command(
                command(
                    "model.timing.configure",
                    "stale",
                    **scope,
                    enabled=False,
                    expected_timing_revision=0,
                )
            )
        assert error.value.code == "REVISION_CONFLICT"
        await control.command(
            command(
                "model.timing.configure",
                "cancel",
                **scope,
                enabled=False,
                expected_timing_revision=3,
            )
        )
        await until(lambda: len(sent) == 4)
        await control.finish()
        assert receipts[-1]["state"] == "reconciling"
        assert control.pending is None
        assert owner.journal.snapshot["segment"]["timing"]["pending_request_id"] is None
        await asyncio.gather(*tasks)
        fresh = TimingControl(SimpleNamespace(detailed_timing=False))
        assert fresh.revision == 0 and not fresh.ever_enabled

    asyncio.run(run())


def test_room_dynamic_timing_timeout_and_unsupported_peer():
    from test_whitebox_render_contract import command
    from test_whitebox_sink_control import fixture

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.timing_control import (
        TimingControl,
    )

    async def run():
        sink, owner, scope, sent, receipts, tasks = await fixture()
        segment = sink.segment
        segment.detailed_timing = False
        segment.assignment = SimpleNamespace(session_id="s", generation_id="g")
        segment.config.startup_seconds = 0.001
        control = TimingControl(segment)
        request = command(
            "model.timing.configure",
            "toggle",
            **scope,
            enabled=True,
            expected_timing_revision=0,
        )
        with pytest.raises(RenderError) as error:
            await control.command(request)
        assert error.value.code == "CAPABILITY_UNAVAILABLE" and not sent
        control.supported = True
        await control.command(request)
        await asyncio.gather(*tasks)
        assert receipts[-1]["state"] == "reconciling"
        assert control.pending and not segment.detailed_timing
        with pytest.raises(RenderError) as error:
            await control.command(dict(request, request_id="second"))
        assert error.value.code == "TIMING_UPDATE_BUSY"
        await control.finish()

    asyncio.run(run())


def test_dynamic_timing_real_encoder_dit_control_chain_and_decoder_boundary():
    from concurrent.futures import ThreadPoolExecutor

    import msgspec

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_encoder_dit_adapter import (
        ZingEncoderDiTAdapter,
    )
    from sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server import (
        EncoderClient,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker import (
        configure_decoder_timing,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_link import (
        DiTInputService,
        DiTLinkClient,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.queued_dit import (
        QueuedDiTSession,
    )

    class Channel:
        def __init__(self):
            self.replies = []

        def send(self, message):
            self.service.queue.append(message)
            asyncio.run(self.service.queue.run_once())

        def receive(self):
            return self.replies.pop(0)

    workers, entries = [], []

    def create(init, output):
        worker = QueuedDiTSession(
            session_id=init["session_id"],
            generation_id=init["generation_id"],
            create_session=None,
            bridge=SimpleNamespace(accepted_pixels=17),
            sample_moments=None,
            on_output=output,
        )
        workers.append(worker)
        return worker

    def decoder(init):
        entry = dict(
            timing_identity=init,
            timing_revision=0,
            detailed_timing=False,
            service=SimpleNamespace(submitted_frames=9),
        )
        entries.append(entry)
        return SimpleNamespace(
            remote=SimpleNamespace(timing_updates=True, oneway=False),
            configure_timing=lambda m: configure_decoder_timing(entry, m),
            close=lambda: None,
        )

    channel = Channel()
    channel.service = DiTInputService(
        SimpleNamespace(send=channel.replies.append), create, create_decoder=decoder
    )
    link = DiTLinkClient(channel)
    executor = SimpleNamespace(dit_link=link, owner=None, command_index=4)
    with ThreadPoolExecutor(max_workers=1) as pool:
        client = EncoderClient(executor, pool)

        async def run():
            session = SimpleNamespace(
                id="s",
                generation_id="g",
                execution_client=client,
                mark_client_activity=lambda: None,
            )
            opened = await client.open_dit(dict(session_id="s", generation_id="g"))
            assert opened[0]["timing_updates"]
            adapter = ZingEncoderDiTAdapter()
            adapter.connection_admission = asyncio.Lock()
            adapter.timing_updates = True
            adapter.timing_revision = 0
            adapter.ledger = SimpleNamespace(closed=False, event_id=0)
            adapter.end = None
            adapter.max_message_bytes = 4096
            adapter.config = {"detailed_timing": False}
            receipts = []

            async def send_bytes(value):
                receipts.append(msgspec.msgpack.decode(value))

            ws = SimpleNamespace(send_bytes=send_bytes)
            for revision, enabled in enumerate([True, False, True], 1):
                message = dict(
                    type="event",
                    kind="timing.configure",
                    event_id=revision,
                    payload=dict(
                        session_id="s",
                        generation_id="g",
                        enabled=enabled,
                        timing_revision=revision,
                    ),
                )
                await adapter.ingest_wire_event(ws, session, message, 200)
                assert executor.timing_override["enabled"] is enabled
                assert workers[0].detailed_timing is enabled
                assert entries[0]["detailed_timing"] is enabled
                assert receipts[-1]["boundaries"] == dict(
                    encoder=dict(command_index=4),
                    dit=dict(source_frame_end=17),
                    decoder=dict(ok=True, output_frame_start=9),
                )
            for bad in [
                dict(message["payload"], generation_id="old"),
                dict(message["payload"], enabled=1),
                message["payload"],
            ]:
                with pytest.raises(ValueError):
                    await adapter.ingest_wire_event(
                        ws, session, dict(message, event_id=4, payload=bad), 200
                    )
            assert adapter.timing_revision == 3 and len(receipts) == 3
            # Release is serialized in the same execution pool; no prior override survives open.
            workers[0].close = lambda: setattr(workers[0], "closed", True)
            await asyncio.get_running_loop().run_in_executor(pool, link.release, "s")
            await client.open_dit(dict(session_id="s2", generation_id="g2"))
            assert executor.timing_override is None
            assert (
                not workers[-1].detailed_timing and entries[-1]["timing_revision"] == 0
            )

        asyncio.run(run())


def test_decoder_timing_rejects_stale_owner_without_mutation():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker import (
        configure_decoder_timing,
    )

    entry = dict(
        timing_identity=dict(session_id="s", generation_id="g"),
        timing_revision=0,
        detailed_timing=False,
        service=SimpleNamespace(submitted_frames=12),
    )
    for message in [
        dict(session_id="s", generation_id="stale", enabled=True, timing_revision=1),
        dict(session_id="s", generation_id="g", enabled=True, timing_revision=2),
    ]:
        with pytest.raises(ValueError):
            configure_decoder_timing(entry, message)
        assert entry["timing_revision"] == 0 and not entry["detailed_timing"]


def test_off_accepts_inflight_role_reports_without_enabling_future_room_stages(
    timing_registry,
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.timing_control import (
        TimingControl,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.ttff_metrics import (
        TimingMetrics,
    )

    async def run():
        segment = object.__new__(RenderSegment)
        segment.detailed_timing = False
        segment.timing_control = TimingControl(segment)
        segment.timing_control.ever_enabled = True
        segment.timing_control.revision = 2
        segment.timing_metrics = TimingMetrics()
        segment.timing_triggers = {2: "user_control"}
        segment.segment_id, segment.segment_epoch = "segment", "epoch"
        segment.assignment = SimpleNamespace(
            session_id="s", generation_id="g", denoiser=None
        )
        segment.owner = SimpleNamespace(source_identity=lambda: {})
        segment.recording = SimpleNamespace(write_inference_timing=AsyncMock())
        report = InferenceTiming("decoder", prompt_version=2, timing_revision=1)
        report.record("vae_decode", 120)
        report.finished = True
        message = dict(
            type="model.timing",
            model_session_id="s",
            generation_id="old",
            timing=report.snapshot(),
        )
        await segment.model_control(message)
        segment.recording.write_inference_timing.assert_not_awaited()
        await segment.model_control(dict(message, generation_id="g"))
        segment.recording.write_inference_timing.assert_awaited_once()
        assert (
            segment.recording.write_inference_timing.call_args.args[0][
                "timing_revision"
            ]
            == 1
        )
        new = segment._new_room_timing(3)
        new.record("source_dispatch_to_first_rgb", 100)
        new.record("room_accept_to_first_rgb", 300)
        assert [x["stage"] for x in new.stages] == ["room_accept_to_first_rgb"]

    asyncio.run(run())


def test_timing_cancel_during_journal_commit_never_sends_to_model():
    from test_whitebox_render_contract import command
    from test_whitebox_sink_control import fixture

    from sglang.multimodal_gen.runtime.realtime.whitebox.timing_control import (
        TimingControl,
    )

    async def run():
        sink, owner, scope, sent, receipts, tasks = await fixture()
        segment = sink.segment
        segment.detailed_timing = False
        segment.assignment = SimpleNamespace(session_id="s", generation_id="g")
        control = TimingControl(segment)
        control.supported = True
        original = owner.journal.update
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(change, **kwargs):
            if (
                control.pending
                and control.pending["event_id"] is not None
                and not entered.is_set()
            ):
                entered.set()
                await release.wait()
            return await original(change, **kwargs)

        owner.journal.update = delayed
        await control.command(
            command(
                "model.timing.configure",
                "cancel-race",
                **scope,
                enabled=True,
                expected_timing_revision=0,
            )
        )
        await entered.wait()
        segment.source_view.stopped.set()
        await control.finish()
        release.set()
        await asyncio.gather(*tasks)
        assert not sent
        assert (
            owner.journal.snapshot["operations"]["cancel-race"]["state"]
            == "reconciling"
        )

    asyncio.run(run())


def test_export_distinguishes_dynamic_capture_from_disabled_and_mixed_windows(tmp_path):
    root = Path(__file__).resolve().parents[6]
    spec = importlib.util.spec_from_file_location(
        "timing_export_dynamic", root / "benchmark/whitebox/summarize_ttff.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = dict(
        detailed_timing=False, timing_changes=[dict(enabled=True, timing_revision=1)]
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    values = []
    for role in ("room", "encoder", "dit", "decoder"):
        report = InferenceTiming(role, prompt_version=2, timing_revision=1)
        report.record("example_stage", 2)
        report.finished = True
        values.append(report.snapshot())

    def write():
        (tmp_path / "inference-timing.jsonl").write_text(
            "".join(json.dumps(v) + "\n" for v in values)
        )

    write()
    rows, stages = module.read_case(tmp_path, "test")
    assert rows[-1]["timing_status"] == "complete"
    assert all(row["timing_revision"] == 1 for row in stages)
    values[-1]["timing_revision"] = 3
    write()
    rows, _ = module.read_case(tmp_path, "test")
    assert rows[-1]["timing_status"] == "incomplete"
