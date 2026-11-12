# SPDX-License-Identifier: Apache-2.0
"""Measure collector overhead only, without model weights or serving changes.

Use SGLANG_LIGHTWEIGHT_RUNTIME=1 and PYTHONPATH=python. Each repeat compares
bare work, the disabled timer, and the enabled timer. CUDA synchronization is
outside the measured call loop, once at each batch boundary. This is not a
model benchmark: the matrix sizes are probes, not frozen product-case inputs.
Set PROMETHEUS_MULTIPROC_DIR before Python starts to measure that mode too.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch
from prometheus_client import REGISTRY

from sglang.multimodal_gen.runtime.realtime import zing_component_metrics as metrics


def sample(name, labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def tags(width, height, measurement):
    return dict(
        component="dit",
        operation="collector_probe",
        service=metrics.service_label("dit"),
        lane=metrics.lane_label(),
        rank=os.environ.get("RANK", "0"),
        resolution=f"{width}x{height}",
        measurement=measurement,
    )


def summary(values):
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "n": len(ordered),
    }


def run(args):
    cuda = args.device == "cuda"
    if cuda:
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() == 1, "Expose one isolated GPU only"
        torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(694)
    results = []
    for width, height in ((832, 480), (1280, 736)):
        if cuda:
            left = torch.randn(height, 1024, device="cuda", dtype=torch.bfloat16)
            right = torch.randn(1024, width, device="cuda", dtype=torch.bfloat16)
            output = torch.empty(height, width, device="cuda", dtype=torch.bfloat16)

        for kernels in (1, 32) if cuda else (0,):
            iterations = args.iterations

            def work():
                if cuda:
                    for _ in range(kernels):
                        torch.mm(left, right, out=output)

            for _ in range(args.warmup):
                work()
            if cuda:
                torch.cuda.synchronize()
            for repeat in range(args.repeats):
                # Rotate arm order to reduce drift bias. All three arms use
                # exactly the same preallocated matrices and call count.
                arms = ("bare", "disabled", "enabled")
                order = arms[repeat % 3 :] + arms[: repeat % 3]
                for arm in order:
                    os.environ["SGLANG_ZING_COMPONENT_METRICS"] = (
                        "1" if arm == "enabled" else "0"
                    )
                    label = tags(width, height, "cuda" if cuda else "host")
                    completed_label = {
                        k: v for k, v in label.items() if k != "measurement"
                    }
                    completed_before = sample(
                        "sglang_zing_component_completed_frames_total", completed_label
                    )
                    before_count = sample(
                        "sglang_zing_component_duration_seconds_count", label
                    )
                    before_sum = sample(
                        "sglang_zing_component_duration_seconds_sum", label
                    )
                    before_frames = sample("sglang_zing_component_frames_total", label)
                    drop_before = {
                        reason: sample(
                            "sglang_zing_component_timing_dropped_total",
                            {
                                "component": "dit",
                                "reason": reason,
                            },
                        )
                        for reason in ("queue_full", "timeout", "event_error")
                    }
                    if cuda:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        resident = {
                            "allocated": torch.cuda.memory_allocated(),
                            "reserved": torch.cuda.memory_reserved(),
                        }
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                    begin = time.perf_counter()
                    for _ in range(iterations):
                        if arm == "bare":
                            work()
                        else:
                            with metrics.component_timer(
                                "dit",
                                operation="collector_probe",
                                frames=17,
                                width=width,
                                height=height,
                            ):
                                work()
                    enqueue = time.perf_counter() - begin
                    if cuda:
                        end.record()
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - begin
                    gpu_ms = start.elapsed_time(end) if cuda else None
                    wait = time.perf_counter()
                    while arm == "enabled":
                        observed = (
                            sample(
                                "sglang_zing_component_duration_seconds_count", label
                            )
                            - before_count
                        )
                        dropped = sum(
                            sample(
                                "sglang_zing_component_timing_dropped_total",
                                {
                                    "component": "dit",
                                    "reason": reason,
                                },
                            )
                            - old
                            for reason, old in drop_before.items()
                        )
                        if (
                            observed + dropped >= iterations
                            or time.perf_counter() - wait > 5
                        ):
                            break
                        time.sleep(0.005)
                    count = (
                        sample("sglang_zing_component_duration_seconds_count", label)
                        - before_count
                    )
                    frames = (
                        sample("sglang_zing_component_frames_total", label)
                        - before_frames
                    )
                    duration = (
                        sample("sglang_zing_component_duration_seconds_sum", label)
                        - before_sum
                    )
                    assert frames == 17 * count, (frames, count)
                    completed_frames = (
                        sample(
                            "sglang_zing_component_completed_frames_total",
                            completed_label,
                        )
                        - completed_before
                    )
                    assert completed_frames == (
                        17 * iterations if arm == "enabled" else 0
                    )
                    results.append(
                        {
                            "width": width,
                            "height": height,
                            "kernels_per_call": kernels,
                            "repeat": repeat,
                            "arm": arm,
                            "calls": iterations,
                            "enqueue_us_per_call": enqueue * 1e6 / iterations,
                            "completed_us_per_call": elapsed * 1e6 / iterations,
                            "outer_cuda_ms": gpu_ms,
                            "observed_calls": count,
                            "observed_frames": frames,
                            "completed_frames": completed_frames,
                            "observed_duration_ms": duration * 1000,
                            "collector_drain_ms": (time.perf_counter() - wait) * 1000,
                            "dropped": {
                                reason: sample(
                                    "sglang_zing_component_timing_dropped_total",
                                    {
                                        "component": "dit",
                                        "reason": reason,
                                    },
                                )
                                - old
                                for reason, old in drop_before.items()
                            },
                            "resident_bytes": resident if cuda else None,
                            "peak_bytes": {
                                "allocated": torch.cuda.max_memory_allocated(),
                                "reserved": torch.cuda.max_memory_reserved(),
                            }
                            if cuda
                            else None,
                        }
                    )
    report = {
        "scope": "collector_microbenchmark_not_model_or_e2e",
        "runtime_sha": args.runtime_sha,
        "tool_sha": args.tool_sha,
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "gpu": str(torch.cuda.get_device_properties(0)) if cuda else None,
        "multiprocess": bool(os.environ.get("PROMETHEUS_MULTIPROC_DIR")),
        "warmup_calls": args.warmup,
        "results": results,
        "source_sha256": {
            Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (metrics.__file__, __file__)
        },
    }
    grouped = []
    for width, height in ((832, 480), (1280, 736)):
        for kernels in (1, 32) if cuda else (0,):
            item = {"width": width, "height": height, "kernels_per_call": kernels}
            for arm in ("bare", "disabled", "enabled"):
                rows = [
                    r
                    for r in results
                    if (r["width"], r["height"], r["kernels_per_call"], r["arm"])
                    == (width, height, kernels, arm)
                ]
                item[arm] = summary([r["completed_us_per_call"] for r in rows])
                item[arm]["dropped"] = sum(sum(r["dropped"].values()) for r in rows)
            item["enabled_vs_disabled_percent"] = (
                item["enabled"]["median"] / item["disabled"]["median"] - 1
            ) * 100
            grouped.append(item)
    report["summary"] = grouped
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("scope", "device", "multiprocess", "summary")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--runtime-sha", required=True)
    parser.add_argument("--tool-sha", required=True)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
