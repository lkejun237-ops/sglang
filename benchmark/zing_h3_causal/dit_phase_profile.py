# SPDX-License-Identifier: Apache-2.0
"""Independent cycle profile and exclusive CUDA-kernel cost attribution."""

import gzip
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from analyze_sp_profile import correlate_blocks, event_kind, timing, trace_id

CATEGORIES = ("GEMM", "Attention", "KV/RoPE", "Other")


def async_nccl_launch_scopes(events, scopes):
    """Recognize only unique, unscoped NCCL driver launches inside one CPU cycle.

    GPU timestamps never determine ownership. Keep the actual driver event as
    the derived scope; this does not fabricate a profiler annotation.
    """
    by_id = defaultdict(list)
    for event in events:
        key = trace_id(event, "correlation")
        if key is not None:
            by_id[key].append(event)
    launches, receipts = [], []
    for kernel in events:
        if event_kind(kernel) != "cuda_kernel" or not re.search(
            r"\bncclDevKernel_SendRecv\b", kernel.get("name", "")
        ):
            continue
        key = trace_id(kernel, "correlation")
        pair = by_id.get(key, [])
        if len(pair) != 2:
            continue
        launch = next((e for e in pair if e is not kernel), None)
        if (
            launch is None
            or "cuda_driver" not in launch.get("cat", "").lower()
            or launch.get("name") not in ("cuLaunchKernel", "cuLaunchKernelEx")
            or any(trace_id(e, "externalid") is not None for e in pair)
        ):
            continue

        def encloses(scope, same_thread=False):
            return (
                scope.get("pid") == launch.get("pid")
                and (not same_thread or scope.get("tid") == launch.get("tid"))
                and scope["ts"] <= launch["ts"]
                and launch["ts"] + launch["dur"] <= scope["ts"] + scope["dur"]
            )

        cycles = [
            s for s in scopes if s["name"] == "PhaseBenchmark.cycle" and encloses(s)
        ]
        if len(cycles) != 1 or any(encloses(s, same_thread=True) for s in scopes):
            continue
        launches.append(launch)
        receipts.append(
            dict(
                method="unique_async_launch_in_cycle",
                category="Other",
                kernel=kernel["name"],
                correlation=key,
                cuda_ms=kernel["dur"] / 1000,
                launch_pid=launch.get("pid"),
                launch_tid=launch.get("tid"),
                launch_ts_us=launch["ts"],
                launch_duration_us=launch["dur"],
            )
        )
    return launches, receipts


def analyze_profile(trace, chunks, steps=4):
    if len(chunks) != 2 or chunks[1] != chunks[0] + 1:
        raise ValueError("Profile requires one consecutive 3+2 cycle")
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    kernels = [e for e in events if event_kind(e) == "cuda_kernel"]
    if not kernels:
        raise ValueError("Cycle profile contains no CUDA kernels; check CUPTI")
    annotations = [e for e in events if event_kind(e) == "cpu_annotation"]
    forward_coverage = []
    for chunk, frames in zip(chunks, (3, 2)):
        owners = [
            e
            for e in annotations
            if e["name"] == f"PhaseBenchmark.chunk.{chunk}.latent{frames}"
        ]
        if len(owners) != 1:
            raise ValueError(
                "Profile must contain each requested sampler chunk exactly once"
            )
        owner = owners[0]
        forwards = [
            e
            for e in annotations
            if (e.get("pid"), e.get("tid")) == (owner.get("pid"), owner.get("tid"))
            and owner["ts"] <= e["ts"]
            and e["ts"] + e["dur"] <= owner["ts"] + owner["dur"]
            and e["name"].startswith("Zing.forward.")
        ]
        names = [e["name"] for e in forwards]
        if sorted(names) != sorted(
            [
                "Zing.forward.source",
                *["Zing.forward.active"] * steps,
                "Zing.forward.final",
            ]
        ):
            raise ValueError(
                "Incomplete source/denoise/commit coverage in profiled chunk"
            )
        rows, _ = correlate_blocks(events, forwards, 0)
        if any(not row["cuda_kernels"]["count"] for row in rows):
            raise ValueError("Profile forward scope has no associated CUDA kernels")
        forward_coverage.append(
            dict(
                chunk=chunk,
                latent_frames=frames,
                calls=[
                    dict(kind=e["name"], kernels=row["cuda_kernels"]["count"])
                    for e, row in zip(forwards, rows)
                ],
            )
        )
    scopes = []
    labels = {}
    for event in events:
        kind, name = event_kind(event), event.get("name", "")
        category = None
        if kind == "cpu_annotation":
            if name == "PhaseBenchmark.cycle":
                category = "Other"
            elif name in ("Zing.FA4", "Zing.SDPA"):
                category = "Attention"
            elif name in (
                "Zing.RoPE_and_KV_append",
                "Zing.build_RoPE_and_history_positions",
                "Zing.KV_history_concat",
                "Zing.history_RoPE_plan",
            ):
                category = "KV/RoPE"
        elif kind == "cpu_op" and re.fullmatch(
            r"aten::(?:mm|bmm|addmm|_scaled_mm)", name
        ):
            category = "GEMM"
        if category:
            scopes.append(event)
            labels[name] = category
    if not any(e["name"] == "PhaseBenchmark.cycle" for e in scopes):
        raise ValueError("Missing cycle scope")
    # The existing analyzer assigns each kernel to its smallest enclosing CPU
    # owner through launch correlation, never asynchronous GPU timestamp clipping.
    rows, status = correlate_blocks(events, scopes, 20, group_by_name=True)
    original_unknown = dict(status["unassigned_cuda_kernels"])
    launches, async_receipts = async_nccl_launch_scopes(events, scopes)
    if launches:
        scopes.extend(launches)
        labels.update({launch["name"]: "Other" for launch in launches})
        rows, status = correlate_blocks(events, scopes, 20, group_by_name=True)
        if original_unknown["count"] - status["unassigned_cuda_kernels"][
            "count"
        ] != len(async_receipts):
            raise ValueError("Async communication ownership changed unrelated kernels")
    costs = {name: dict(cuda_ms=0.0, kernels=0) for name in CATEGORIES}
    names = {name: [] for name in CATEGORIES}
    for row in rows:
        category = labels[row["name"]]
        costs[category]["cuda_ms"] += row["cuda_kernels"]["inclusive_sum_ms"]
        costs[category]["kernels"] += row["cuda_kernels"]["count"]
        names[category].extend(row["cuda_kernels"]["top_names"])
    unknown = status["unassigned_cuda_kernels"]
    costs["Other"]["cuda_ms"] += unknown["inclusive_sum_ms"]
    costs["Other"]["kernels"] += unknown["count"]
    names["Other"].extend(unknown["top_names"])
    total = timing(kernels)
    if (
        unknown["count"] / total["count"] > 0.05
        or unknown["inclusive_sum_ms"] / total["inclusive_sum_ms"] > 0.05
    ):
        raise ValueError(
            "Profile kernel launch correlation coverage is below 95 percent"
        )
    if (
        sum(c["kernels"] for c in costs.values()) != total["count"]
        or abs(sum(c["cuda_ms"] for c in costs.values()) - total["inclusive_sum_ms"])
        > 1e-6
    ):
        raise ValueError("Kernel category accounting does not conserve total cost")
    top = {}
    for category, entries in names.items():
        merged = {}
        for entry in entries:
            item = merged.setdefault(
                entry["name"], dict(name=entry["name"], count=0, inclusive_sum_ms=0)
            )
            item["count"] += entry["count"]
            item["inclusive_sum_ms"] += entry["inclusive_sum_ms"]
        top[category] = sorted(
            merged.values(), key=lambda r: r["inclusive_sum_ms"], reverse=True
        )[:20]
    return dict(
        schema_version=1,
        chunks=list(chunks),
        categories=costs,
        total_kernel_ms=total["inclusive_sum_ms"],
        kernel_count=total["count"],
        attributed_kernel_fraction=1 - unknown["count"] / total["count"],
        attributed_time_fraction=1
        - unknown["inclusive_sum_ms"] / total["inclusive_sum_ms"],
        correlation=status,
        original_scope_attributed_time_fraction=1
        - original_unknown["inclusive_sum_ms"] / total["inclusive_sum_ms"],
        async_communication=dict(
            kernels=len(async_receipts),
            cuda_ms=sum(r["cuda_ms"] for r in async_receipts),
            receipts=async_receipts,
        ),
        forward_coverage=forward_coverage,
        top_names=top,
        semantics="Exclusive kernel-work ms in one profiled 3+2 cycle; launch correlation plus CPU scopes; unassigned kernels retained in Other. NCCL SendRecv with unique unscoped driver launch inside the CPU cycle is explicitly retained in Other/Communication. Not wall-time percentages.",
    )


def profile_cycle(sampler, inputs, args, chunks, expected_hash):
    import hashlib

    import torch

    torch.manual_seed(args.seed)
    visual = torch.randn(
        (1, sampler.generator.arch.latents_dim, *inputs.visual_shape), device="cuda"
    )
    with torch.no_grad():
        state = sampler.initialize(inputs, visual)
        try:
            while state.index < chunks[0]:
                sampler.step(state)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as profiler:
                started = time.perf_counter()
                with torch.profiler.record_function("PhaseBenchmark.cycle"):
                    for chunk, frames in zip(chunks, (3, 2)):
                        if state.index != chunk:
                            raise ValueError("Profile chunks are not consecutive")
                        if state.spans[chunk][3] - state.spans[chunk][2] != frames:
                            raise ValueError(
                                "Profile phase is not the expected 3/2 latent geometry"
                            )
                        with torch.profiler.record_function(
                            f"PhaseBenchmark.chunk.{chunk}.latent{frames}"
                        ):
                            sampler.step(state)
                    torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - started) * 1000
            peak = torch.cuda.max_memory_allocated()
            while state.index < len(state.spans):
                sampler.step(state)
            output = state.visual
            digest = hashlib.sha256(
                output.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            if not bool(torch.isfinite(output).all()) or digest != expected_hash:
                raise ValueError("Profile repetition changed latent output")
        finally:
            state.close()
            del sampler._forward_counts
    path = Path(args.output) / "cycle.trace.json.gz"
    profiler.export_chrome_trace(str(path))
    with gzip.open(path, "rt") as stream:
        result = analyze_profile(
            json.load(stream), chunks, len(args.denoising_step_list)
        )
    result.update(
        profile_wall_ms=wall_ms,
        profile_peak_allocated=peak,
        output_sha256=digest,
        trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    (Path(args.output) / "profile.json").write_text(json.dumps(result, indent=2))
    return result
