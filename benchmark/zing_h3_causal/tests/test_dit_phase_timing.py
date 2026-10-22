# SPDX-License-Identifier: Apache-2.0
import pytest
from bench_dit_phases import keep_layer
from dit_phase_timing import check_forward_rows, distribution, summarize


def row(chunk, frames, *, run=0, kind="block", mode="block", **kw):
    return dict(
        run=run,
        run_kind="measured",
        chunk=chunk,
        latent_frames=frames,
        kind=kind,
        mode=mode,
        step=None,
        q_rows=3,
        prefix_rows=5,
        history_rows=7,
        kv_rows=15,
        wall_ms=10,
        cuda_ms=8,
        **kw,
    )


def test_cycle_sums_paired_samples_before_reducing():
    rows = []
    for run, first, second in [(0, 1, 100), (1, 50, 60), (2, 100, 1)]:
        a, b = row(3, 3, run=run), row(4, 2, run=run)
        a["wall_ms"], b["wall_ms"] = first, second
        rows += [a, b]
    result = summarize(rows)["cycles"][0]
    assert result["wall_ms"]["median"] == 101  # sum of medians would be 110
    assert result["wall_ms"]["count"] == 3


def test_no_cycle_from_startup_tail_or_different_runs():
    assert not summarize([row(1, 3), row(2, 2), row(3, 3), row(4, 2, run=1)])["cycles"]


def test_warmup_and_histories_are_not_pooled():
    a = row(3, 3, kind="denoise", mode="forward")
    b = dict(a, run=1, history_rows=8)
    warmup = dict(a, run_kind="warmup", wall_ms=999)
    result = summarize([a, b, warmup])["groups"]
    assert len(result) == 2
    assert all(r["wall_ms"]["count"] == 1 for r in result)


@pytest.mark.parametrize("bad", [[], [float("nan")], [float("inf")], [-1]])
def test_bad_timings_cannot_become_speed_results(bad):
    with pytest.raises(ValueError):
        distribution(bad)


def test_filter_preserves_refiner_and_excludes_unselected_blocks():
    assert keep_layer("backbone.transformer_blocks.0.attn.weight", 1)
    assert not keep_layer("backbone.transformer_blocks.1.attn.weight", 1)
    assert not keep_layer("blocks.49.norm.weight", 1)
    assert keep_layer("backbone.token_refiner.blocks.1.norm.weight", 1)
    assert keep_layer("backbone.proj_in.weight", 1)


def test_missing_or_duplicate_steps_are_rejected_even_when_counts_match():
    rows = [
        dict(row(0, 3, kind="denoise", mode="forward"), step=step) for step in (0, 0)
    ]
    rows.append(row(0, 3, kind="clean_commit", mode="forward"))
    with pytest.raises(ValueError, match="denoise step"):
        check_forward_rows(rows, [dict(frames=3)], 2, dict(denoise=2, commit=1))


def _profile_trace():
    def event(name, cat, ts, dur, **args):
        return dict(name=name, cat=cat, ph="X", ts=ts, dur=dur, pid=1, tid=1, args=args)

    events = [event("PhaseBenchmark.cycle", "user_annotation", 0, 1000)]
    operations = [
        ("Zing.RoPE_and_KV_append", "user_annotation"),
        ("aten::mm", "cpu_op"),
        ("Zing.FA4", "user_annotation"),
        ("aten::mm", "cpu_op"),
        ("aten::gelu", "cpu_op"),
        ("Zing.RoPE_and_KV_append", "user_annotation"),
    ]
    for index, (chunk, frames) in enumerate([(13, 3), (14, 2)]):
        events.append(
            event(
                f"PhaseBenchmark.chunk.{chunk}.latent{frames}",
                "user_annotation",
                index * 400,
                400,
            )
        )
        for step, (name, cat) in enumerate(operations):
            start = index * 400 + step * 40 + 10
            kind = "source" if step == 0 else "final" if step == 5 else "active"
            correlation = index * 6 + step
            events += [
                event(f"Zing.forward.{kind}", "user_annotation", start, 25),
                event(name, cat, start + 2, 10),
                event(
                    "cudaLaunchKernel",
                    "cuda_runtime",
                    start + 3,
                    1,
                    correlation=correlation,
                ),
                event(
                    "generic_kernel",
                    "kernel",
                    10000 + correlation * 1000,
                    (step + 1) * 100,
                    correlation=correlation,
                ),
            ]
    return dict(traceEvents=events)


def test_profile_categories_are_exclusive_and_follow_launch_not_gpu_time():
    from dit_phase_profile import analyze_profile

    result = analyze_profile(_profile_trace(), [13, 14])
    costs = {k: v["cuda_ms"] for k, v in result["categories"].items()}
    assert costs == pytest.approx(
        {"GEMM": 1.2, "Attention": 0.6, "KV/RoPE": 1.4, "Other": 1.0}
    )
    assert (
        sum(c["kernels"] for c in result["categories"].values())
        == result["kernel_count"]
        == 12
    )
    assert result["total_kernel_ms"] == pytest.approx(4.2)
    assert result["attributed_kernel_fraction"] == 1
    assert len(result["forward_coverage"]) == 2
    assert result["top_names"]["GEMM"][0]["count"] == 4


def test_profile_rejects_missing_phase_or_orphan_kernels():
    from dit_phase_profile import analyze_profile

    trace = _profile_trace()
    trace["traceEvents"] = [
        e
        for e in trace["traceEvents"]
        if e["name"] != "PhaseBenchmark.chunk.14.latent2"
    ]
    with pytest.raises(ValueError, match="each requested sampler chunk"):
        analyze_profile(trace, [13, 14])
    trace = _profile_trace()
    trace["traceEvents"].append(
        dict(
            name="orphan",
            cat="kernel",
            ph="X",
            ts=99999,
            dur=50,
            pid=1,
            tid=1,
            args={"correlation": 999},
        )
    )
    with pytest.raises(ValueError, match="correlation coverage"):
        analyze_profile(trace, [13, 14])


def _async_nccl_trace():
    trace = _profile_trace()
    trace["traceEvents"] += [
        dict(
            name="cuLaunchKernelEx",
            cat="cuda_driver",
            ph="X",
            ts=300,
            dur=1,
            pid=1,
            tid=2,
            args={"correlation": 999},
        ),
        dict(
            name="ncclDevKernel_SendRecv",
            cat="kernel",
            ph="X",
            ts=30000,
            dur=2000,
            pid=0,
            tid=141,
            args={"correlation": 999},
        ),
    ]
    return trace


def test_async_nccl_communication_requires_unique_launch_inside_cpu_cycle():
    from dit_phase_profile import analyze_profile

    result = analyze_profile(_async_nccl_trace(), [13, 14])
    assert result["categories"]["Other"]["cuda_ms"] == pytest.approx(3)
    assert result["kernel_count"] == 13 and result["attributed_time_fraction"] == 1
    assert result["original_scope_attributed_time_fraction"] == pytest.approx(4.2 / 6.2)
    assert result["async_communication"]["kernels"] == 1
    assert (
        result["async_communication"]["receipts"][0]["method"]
        == "unique_async_launch_in_cycle"
    )


@pytest.mark.parametrize(
    "invalid", ["duplicate", "outside", "different_process", "external", "non_nccl"]
)
def test_async_launch_does_not_rescue_ambiguous_or_unrelated_work(invalid):
    from dit_phase_profile import analyze_profile

    trace = _async_nccl_trace()
    launch, kernel = trace["traceEvents"][-2:]
    if invalid == "duplicate":
        trace["traceEvents"].append(dict(launch))
    elif invalid == "outside":
        launch["ts"] = 1100
    elif invalid == "different_process":
        launch["pid"] = 2
    elif invalid == "external":
        kernel["args"]["External id"] = 5
    else:
        kernel["name"] = "unscoped_elementwise"
    with pytest.raises(ValueError, match="correlation coverage"):
        analyze_profile(trace, [13, 14])
