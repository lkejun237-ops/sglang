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
