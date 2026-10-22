"""CPU regressions for profiling evidence; no CUDA or checkpoint required."""

import importlib.util
import json
from pathlib import Path

import analyze_sp_profile as analyzer
import pytest
import torch
from check_profile_outputs import check_outputs


def event(name, ts, dur, cat="user_annotation", **args):
    return {
        "name": name,
        "ts": ts,
        "dur": dur,
        "ph": "X",
        "pid": 1,
        "tid": 1,
        "cat": cat,
        "args": args,
    }


def trace_events():
    return [
        event("Zing.causal_block1.latent5-10.history_rows42", 0, 100),
        event("Zing.denoise_step.block1.step0", 1, 40),
        event("Zing.forward.active", 2, 35),
        event("Zing.AdaLN", 3, 10),
        event("aten::linear", 4, 5, "cpu_op", **{"External id": 7}),
        event(
            "cudaLaunchKernel",
            5,
            1,
            "cuda_runtime",
            correlation=11,
            **{"External id": 7},
        ),
        # GPU executes after the CPU scope AND block have ended.
        event("gemm", 200, 20, "kernel", correlation=11),
        event("Zing.forward.commit", 50, 30),
        event("Zing.RoPE_and_KV_append", 51, 10),
        event("aten::cat", 52, 4, "cpu_op", **{"External id": 8}),
        event("cudaLaunchKernel", 53, 1, "cuda_runtime", correlation=12),
        event("copy", 230, 5, "kernel", correlation=12),
    ]


def cross(events):
    return analyzer.region_block_evidence(
        events,
        [e for e in events if analyzer.BLOCK_PATTERN.search(e["name"])],
        [e for e in events if analyzer.REGION_PATTERN.search(e["name"])],
        10,
        analyzer.event_kind,
    )


def test_async_kernel_and_commit_attribution():
    result = cross(trace_events())
    active, commit = result["rows"]
    assert active["cuda_kernels"]["inclusive_sum_ms"] == 0.02
    assert active["history_rows"] == 42
    assert active["latent_span"] == [5, 10]
    assert active["step"] == "Zing.denoise_step.block1.step0"
    assert commit["forward"] == "Zing.forward.commit"
    assert commit["step"] is None
    assert commit["cuda_kernels"]["count"] == 1


def test_conflicting_launch_id_stays_unassigned():
    events = trace_events()
    events.append(event("cudaLaunchKernel", 54, 1, "cuda_runtime", correlation=11))
    result = cross(events)
    assert result["rows"][0]["cuda_kernels"]["count"] == 0
    assert result["correlation_status"]["ambiguous_or_unscoped_correlation"] == 1


def test_other_thread_is_not_a_parent():
    events = trace_events()
    events[0]["tid"] = 99
    assert all(row["block"] is None for row in cross(events)["rows"])


def test_summarize_exposes_cross_table(tmp_path):
    path = tmp_path / "rank0.json"
    path.write_text(json.dumps({"traceEvents": trace_events()}))
    result = analyzer.summarize(path, 10)
    assert len(result["region_by_causal_block"]["rows"]) == 2


@pytest.fixture
def outputs(tmp_path):
    runs = {}
    shapes = {"visual_latents": [1, 2, 3], "audio": [1, 8]}
    for sp in (1, 2, 4):
        root = tmp_path / f"sp{sp}"
        root.mkdir()
        runs[str(sp)] = str(root)
        for i in range(3):
            torch.save(
                {"visual_latents": torch.ones(1, 2, 3), "audio": torch.ones(1, 8)},
                root / f"sglang-{i}-raw.pt",
            )
    return runs, shapes


def test_raw_matrix_comparisons(outputs):
    result = check_outputs(*outputs)
    assert result["valid"] and result["matrix_complete"]
    assert len(result["files"]) == 9
    assert len(result["comparisons"]) == 8
    assert result["strict_parity"] == "not_evaluated_no_frozen_tolerance"


@pytest.mark.parametrize("damage", ["nan", "shape", "missing_audio", "missing_file"])
def test_later_raw_failure_invalidates_matrix(outputs, damage):
    runs, shapes = outputs
    path = Path(runs["4"]) / "sglang-2-raw.pt"
    value = torch.load(path, weights_only=True)
    if damage == "nan":
        value["audio"][0, -1] = float("nan")
    elif damage == "shape":
        value["visual_latents"] = torch.ones(3)
    elif damage == "missing_audio":
        value.pop("audio")
    if damage == "missing_file":
        path.unlink()
    else:
        torch.save(value, path)
    result = check_outputs(runs, shapes)
    assert not result["valid"]
    assert not result["files"][-1]["valid"]
    assert len(result["comparisons"]) == 7


def test_shape_spec_is_not_inferred_from_results(outputs):
    runs, shapes = outputs
    shapes["audio"] = [2, 8]
    assert not check_outputs(runs, shapes)["valid"]


def test_rank_memory_preserves_rank_values():
    source = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/multimodal_gen/runtime/utils/rank_memory.py"
    )
    spec = importlib.util.spec_from_file_location("rank_memory", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Distributed:
        def get_world_size(self, group):
            assert group == "cpu"
            return 2

        def all_gather_object(self, rows, local, group):
            rows[:] = [
                local,
                (3, {"peak_allocated_mb": 100}, {"peak_reserved_mb": 110}),
            ]

    result = module.gather_rank_memory(
        Distributed(), "cpu", 2, {"peak_allocated_mb": 10}, {"peak_reserved_mb": 40}
    )
    assert result["rank_2_load_peak"]["peak_allocated_mb"] == 10
    assert result["rank_3_load_peak"]["peak_allocated_mb"] == 100
    assert result["rank_2_request_peak"]["peak_reserved_mb"] == 40
    assert len(result) == 4


def test_harness_preserves_warmup_and_profile_metrics(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    import validate_real_checkpoint as harness

    calls = []

    class Generator:
        @staticmethod
        def from_pretrained(**kwargs):
            assert kwargs["warmup_mode"] == "off"
            return Generator()

        def generate(self, sampling_params_kwargs):
            params = sampling_params_kwargs
            calls.append(params)
            return SimpleNamespace(
                samples=torch.ones(1, 3, 2, 2, 2),
                trajectory_latents=torch.ones(1, 2, 3),
                audio=torch.ones(1, 8),
                peak_memory_mb=40,
                usage=None,
                metrics={
                    "memory_snapshots": {
                        "rank_0_load_peak": {
                            "peak_allocated_mb": 1,
                            "peak_reserved_mb": 2,
                        },
                        "rank_0_request_peak": {
                            "peak_allocated_mb": len(calls),
                            "peak_reserved_mb": 40,
                        },
                    }
                },
                output_file_path="mock.mp4",
            )

        def shutdown(self):
            calls.append("shutdown")

    module = ModuleType("sglang.multimodal_gen.runtime.entrypoints.diffusion_generator")
    module.DiffGenerator = Generator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(harness, "_hardware", lambda: {"mock": True})
    monkeypatch.setattr(harness, "_source_commit", lambda *_: "test")
    request = Path(__file__).with_name("request_10s.json")
    args = harness.parser().parse_args(
        [
            "sglang",
            "--sglang-root",
            str(tmp_path),
            "--model",
            "mock",
            "--request",
            str(request),
            "--output",
            str(tmp_path),
            "--warmup-runs",
            "1",
            "--measured-runs",
            "3",
            "--save-all-raw",
            "--collect-rank-memory",
            "--profile-run",
        ]
    )
    harness.run_sglang(args)
    report = json.loads((tmp_path / "sglang.json").read_text())
    assert len(report["warmup_metrics"]) == 1
    assert len(report["measured_metrics"]) == 3
    assert (
        report["profile_metrics"]["memory_snapshots"]["rank_0_request_peak"][
            "peak_allocated_mb"
        ]
        == 5
    )
    assert report["collect_rank_memory"] is True
    assert [c["profile"] for c in calls[:-1]] == [False, False, False, False, True]
    assert calls[-1] == "shutdown"
    assert all((tmp_path / f"sglang-{i}-raw.pt").exists() for i in range(3))


def test_raw_cli_failure_writes_report(outputs, tmp_path):
    import subprocess
    import sys

    runs, shapes = outputs
    (Path(runs["2"]) / "sglang-1-raw.pt").unlink()
    spec = tmp_path / "shapes.json"
    spec.write_text(json.dumps(shapes))
    report = tmp_path / "report.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("check_profile_outputs.py")),
        "--expected-shapes",
        str(spec),
        "--output",
        str(report),
    ]
    for sp, directory in runs.items():
        command.extend(["--run", f"{sp}={directory}"])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert json.loads(report.read_text())["valid"] is False


def test_missing_rank_memory_fails_before_reporting_success():
    from validate_real_checkpoint import _require_rank_memory

    metrics = {
        "memory_snapshots": {
            "rank_0_load_peak": {"peak_allocated_mb": 1, "peak_reserved_mb": 2},
            "rank_0_request_peak": {"peak_allocated_mb": 3, "peak_reserved_mb": 4},
        }
    }
    _require_rank_memory(metrics, 1)
    with pytest.raises(RuntimeError, match="rank_1_load_peak"):
        _require_rank_memory(metrics, 2)


def _gloo_memory_worker(rank, rendezvous, output):
    from datetime import timedelta

    distributed = torch.distributed
    distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        source = (
            Path(__file__).resolve().parents[2]
            / "python/sglang/multimodal_gen/runtime/utils/rank_memory.py"
        )
        spec = importlib.util.spec_from_file_location("rank_memory", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        records = module.gather_rank_memory(
            distributed,
            distributed.group.WORLD,
            rank,
            {"peak_allocated_mb": 10 + rank},
            {"peak_reserved_mb": 20 + rank},
        )
        Path(output, f"rank{rank}.json").write_text(json.dumps(records))
    finally:
        distributed.destroy_process_group()


def test_two_cpu_processes_gather_distinct_rank_records(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    torch.multiprocessing.spawn(
        _gloo_memory_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    records = [
        json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in (0, 1)
    ]
    assert records[0] == records[1]
    assert records[0]["rank_0_load_peak"]["peak_allocated_mb"] == 10
    assert records[0]["rank_1_load_peak"]["peak_allocated_mb"] == 11


def allocator_snapshot(events, blocks):
    return dict(device_traces=[events], segments=[dict(device=0, blocks=blocks)])


def allocation(action, addr, size, time):
    return dict(
        action=action,
        addr=addr,
        size=size,
        time_us=time,
        frames=[dict(filename="model.py", line=9, name="forward")],
    )


def live_block(addr, size):
    return dict(
        address=addr,
        size=size,
        requested_size=size,
        state="active_allocated",
        frames=[],
    )


def test_allocation_replay_separates_pending_free_and_old_resident_bytes():
    from allocation_replay import final_live, replay, stack_groups

    events = [
        allocation("alloc", 2, 30, 1),
        allocation("free_requested", 2, 30, 2),
        allocation("alloc", 3, 40, 3),
        allocation("free_completed", 2, 30, 4),
        allocation("free_requested", 1, 100, 5),
        allocation("free_completed", 1, 100, 6),
    ]
    snapshot = allocator_snapshot(events, [live_block(3, 40)])
    result = replay(events, final_live(snapshot, 0), 0, buckets=1)
    assert result["closure"] and result["baseline_bytes"] == 100
    assert result["new_peak_bytes"] == 40
    assert result["new_active_peak_bytes"] == 70
    assert result["net_peak_bytes"] == 40
    assert result["end_allocated_bytes"] == 40
    assert sum(g["bytes"] for g in stack_groups(result["peak_live"])) == 40
    assert max(p[1] + p[2] for p in result["points"]) == 70
    assert min(p[3] for p in result["points"]) == -60


def test_allocation_replay_address_reuse_creates_a_new_allocation():
    from allocation_replay import final_live, replay

    events = [
        allocation("alloc", 2, 40, 1),
        allocation("free_requested", 2, 40, 2),
        allocation("free_completed", 2, 40, 3),
        allocation("alloc", 2, 80, 5),
    ]
    result = replay(
        events, final_live(allocator_snapshot(events, [live_block(2, 80)]), 0), 4
    )
    assert result["baseline_bytes"] == 0 and result["new_peak_bytes"] == 80
    assert result["peak_live"][0][1][2] == 3


def test_allocation_replay_rejects_inconsistent_final_snapshot():
    from allocation_replay import final_live, replay

    events = [allocation("alloc", 2, 40, 1)]
    with pytest.raises(ValueError, match="mismatch"):
        replay(
            events, final_live(allocator_snapshot(events, [live_block(2, 80)]), 0), 0
        )


def test_partial_history_keeps_the_actual_retained_start():
    from analyze_memory import analyze

    events = [allocation("alloc", 2, 40, 8)]
    row = analyze(
        allocator_snapshot(events, [live_block(2, 40)]),
        dict(
            device=0,
            start_us=2,
            history_may_be_truncated=True,
            case="capture",
            pid=7,
            role="server",
        ),
    )
    assert row["partial"] and row["recorded_start_seconds"] == 6e-6
    assert row["points"][0][0] == 6e-6 and row["closure"]


def test_memory_probe_preserves_peak_resets_and_stops_sampling_when_disabled(
    tmp_path, monkeypatch
):
    import importlib
    import sys
    from types import SimpleNamespace

    import memory_probe

    probe = importlib.reload(memory_probe)
    state = dict(peak=100, reserved=150)

    def reset(device=None):
        state.update(peak=20, reserved=30)

    cuda = SimpleNamespace(
        set_device=lambda _: None,
        current_device=lambda: 0,
        memory_allocated=lambda _: 20,
        memory_reserved=lambda _: 30,
        max_memory_allocated=lambda _: state["peak"],
        max_memory_reserved=lambda _: state["reserved"],
        reset_peak_memory_stats=reset,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setenv("ZING_BENCHMARK_ROLE", "server")
    monkeypatch.setenv("ZING_BENCHMARK_MEMORY_DIR", str(tmp_path))
    probe.install()
    cuda.reset_peak_memory_stats(0)
    assert probe.INTERVAL_PEAKS[0] == dict(allocated=100, reserved=150)
    probe.ENABLED = False
    state["peak"] = 90
    cuda.reset_peak_memory_stats(0)
    assert state["peak"] == 20 and probe.INTERVAL_PEAKS[0]["allocated"] == 100
    rows = [
        json.loads(line)
        for p in (tmp_path / "memory").glob("samples-*.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert [v["kind"] for v in rows] == ["peak_reset_observed"]
    importlib.reload(probe)


def test_memory_probe_repeated_labels_keep_distinct_snapshots(tmp_path, monkeypatch):
    import importlib
    import sys
    from types import SimpleNamespace

    import memory_probe

    probe = importlib.reload(memory_probe)
    monkeypatch.setenv("ZING_BENCHMARK_ROLE", "server")
    monkeypatch.setenv("ZING_BENCHMARK_MEMORY_DIR", str(tmp_path))
    control = tmp_path / "memory-control.json"
    snapshots = []
    histories = []

    def write_command(sequence, operation="snapshot"):
        control.write_text(
            json.dumps(
                dict(sequence=sequence, label="after-request", operation=operation)
            )
        )

    def dump(path):
        from pathlib import Path

        snapshots.append(path)
        Path(path).write_text(str(len(snapshots)))

    cuda = SimpleNamespace(
        set_device=lambda _: None,
        current_device=lambda: 0,
        get_device_name=lambda _: "mock GPU",
        synchronize=lambda _: None,
        memory_allocated=lambda _: 20,
        memory_reserved=lambda _: 30,
        max_memory_allocated=lambda _: 40,
        max_memory_reserved=lambda _: 50,
        reset_peak_memory_stats=lambda _: None,
        memory=SimpleNamespace(
            _record_memory_history=lambda **kw: histories.append(kw),
            _dump_snapshot=dump,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=cuda, __version__="mock")
    )
    monkeypatch.setattr(
        probe.threading, "Thread", lambda target, **kw: SimpleNamespace(start=target)
    )

    def next_command(_):
        sequence = json.loads(control.read_text())["sequence"] + 1
        assert sequence <= 3, "Observer did not stop after disable"
        write_command(sequence, "disable" if sequence == 3 else "snapshot")

    monkeypatch.setattr(probe.time, "sleep", next_command)
    write_command(1)
    probe.start()
    assert len(set(snapshots)) == 2
    from pathlib import Path

    assert [Path(p).read_text() for p in snapshots] == ["1", "2"]
    acks = [
        json.loads(p.read_text())
        for p in sorted((tmp_path / "memory").glob("ack-*.json"))
    ]
    assert len(acks) == 3 and all(a["passed"] for a in acks)
    assert acks[0]["snapshot"] != acks[1]["snapshot"]
    assert acks[2]["snapshot"] is None and not probe.ENABLED
    assert histories[-1] == dict(enabled=None, clear_history=True)
    assert not list((tmp_path / "memory").glob("ERROR-*.txt"))
    importlib.reload(probe)
