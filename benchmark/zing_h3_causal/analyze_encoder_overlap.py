# SPDX-License-Identifier: Apache-2.0
"""CPU-only matrix summary; missing evidence remains unknown, never zero."""

import argparse
import gzip
import json
import re
import statistics
from pathlib import Path

from analyze_sp_profile import PATTERNS, correlate_blocks, event_kind
from encoder_overlap_config import layouts, selected_layouts


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def intersection_us(left, right):
    left, right = merged(left), merged(right)
    i = j = 0
    total = 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        total += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] <= b[1]:
            i += 1
        else:
            j += 1
    return total


def trace_intervals(path, scope_name):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        trace = json.load(f)
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X" and "dur" in e]
    anchors = [
        e
        for e in events
        if str(e.get("name", "")).startswith("Zing.clock_anchor.unix_ns")
    ]
    base = trace.get("baseTimeNanoseconds")
    if not isinstance(base, (int, float)) or not anchors:
        return None, "Missing clock base or wall-clock anchor"
    errors = [
        abs(base + e["ts"] * 1000 - int(e["name"].split("unix_ns")[1])) for e in anchors
    ]
    if max(errors) > 5_000_000:
        return None, "Trace clock failed the 5 ms wall-clock alignment check"
    scopes = [
        e
        for e in events
        if event_kind(e) == "cpu_annotation" and e.get("name") == scope_name
    ]
    if scope_name == "Zing.encoder_cross_dispatch":
        scopes = []
        pattern = re.compile(
            r"^Zing\.encoder_prefetch\.latent_start_(\d+)\.source_end_(\d+)\.dispatch_end_(\d+)$"
        )
        for event in events:
            match = pattern.match(str(event.get("name", "")))
            if event_kind(event) != "cpu_annotation" or match is None:
                continue
            latent, source, dispatched = map(int, match.groups())
            phase = latent % 5
            if phase not in (0, 1, 3):
                continue
            dependency = min(source, latent // 5 * 17 + {0: 1, 1: 9, 3: 17}[phase])
            if dependency > dispatched:
                scopes.append(event)
    # #178 exposes per-tile compute scopes. Gather is deliberately excluded.
    if not scopes and scope_name == "Zing.encoder_tiles":
        scopes = [
            e
            for e in events
            if event_kind(e) == "cpu_annotation"
            and str(e.get("name", "")).startswith(
                ("H3ProgressiveTile/", "H3ProgressiveTiles/")
            )
        ]
    if scope_name in (
        "Zing.encoder_tiles",
        "Zing.encoder_cross_dispatch",
        "Zing.source_dit",
    ) or scope_name.startswith("Zing.source_decode"):
        collective = re.compile(PATTERNS["collectives"], re.I)
        events = [
            e
            for e in events
            if not (
                event_kind(e) == "cuda_kernel"
                and collective.search(str(e.get("name", "")))
            )
        ]
    if not scopes:
        return None, "Missing requested scope"
    rows, status = correlate_blocks(events, scopes, 5, include_intervals=True)
    intervals = [
        (a + base / 1000, b + base / 1000)
        for r in rows
        for a, b in r["kernel_intervals_us"]
    ]
    if not intervals:
        return None, "No kernels attributed to the requested scope"
    per_scope = []
    for scope, row in zip(scopes, rows):
        active = merged(row["kernel_intervals_us"])
        per_scope.append(
            dict(
                name=scope["name"],
                host_start_us=scope["ts"],
                cuda_active_ms=sum(b - a for a, b in active) / 1000,
                cuda_span_ms=(active[-1][1] - active[0][0]) / 1000 if active else None,
            )
        )
    return merged(intervals), dict(
        attribution=status, clock_max_error_ns=max(errors), scopes=per_scope
    )


def overlap_evidence(cell, sp, enc):
    if not enc:
        return dict(status="shared_baseline", observed_overlap_ms=None)
    encoder = sorted((cell / "encoder-traces").glob("*.trace.json*"))
    dit = sorted((cell / "traces").glob("*.trace.json*"))
    if len(encoder) != enc or not all(
        any(f"global-rank{r}." in p.name for p in dit) for r in range(sp)
    ):
        return dict(status="missing_rank_traces", observed_overlap_ms=None)
    streams = {"encoder": [], "dit": []}
    evidence = []
    for kind, paths, scope in (
        ("encoder", encoder, "Zing.encoder_tiles"),
        ("dit", dit, "Zing.source_dit"),
    ):
        for path in paths:
            intervals, status = trace_intervals(path, scope)
            evidence.append(dict(path=str(path), status=status))
            if intervals is None:
                return dict(
                    status="unverified_clock_or_attribution",
                    observed_overlap_ms=None,
                    evidence=evidence,
                )
            streams[kind].extend(intervals)
    duration = intersection_us(streams["encoder"], streams["dit"]) / 1000
    future_intervals, decoder_intervals, future_evidence = [], [], []
    future_verified = decoder_verified = True
    for path in encoder:
        intervals, status = trace_intervals(path, "Zing.encoder_cross_dispatch")
        future_evidence.append(dict(path=str(path), status=status))
        if intervals is None:
            future_verified = False
        else:
            future_intervals.extend(intervals)
    for path in dit:
        intervals, status = trace_intervals(path, "Zing.source_decode")
        if intervals is None:
            decoder_verified = False
        else:
            decoder_intervals.extend(intervals)
    cross_dispatch_dit_ms = (
        intersection_us(future_intervals, streams["dit"]) / 1000
        if future_verified
        else None
    )
    cross_dispatch_decoder_ms = (
        intersection_us(future_intervals, decoder_intervals) / 1000
        if future_verified and decoder_verified
        else None
    )
    return dict(
        status="aligned_kernel_evidence",
        observed_overlap_ms=duration,
        cross_dispatch_dit_overlap_ms=cross_dispatch_dit_ms,
        cross_dispatch_decoder_overlap_ms=cross_dispatch_decoder_ms,
        cross_dispatch_evidence=future_evidence,
        evidence=evidence,
        semantics="Union across ranks on one node; communication and decoder kernels excluded; profiler run only",
    )


def decoder_evidence(cell):
    paths = sorted((cell / "traces").glob("*global-rank0.*trace.json*"))
    if len(paths) != 1:
        return dict(status="unknown", reason="Expected one leader profile trace")
    intervals, evidence = trace_intervals(paths[0], "Zing.source_decode")
    if intervals is None:
        return dict(status="unknown", reason=evidence, path=str(paths[0]))
    eos_intervals, eos_evidence = trace_intervals(paths[0], "Zing.source_decode.eos")
    rows = evidence["scopes"]
    values = sorted(
        row["cuda_active_ms"] for row in rows if row["cuda_span_ms"] is not None
    )
    return dict(
        status="collected",
        path=str(paths[0]),
        cuda_active_ms=sum(b - a for a, b in intervals) / 1000,
        block_p50_cuda_active_ms=statistics.median(values),
        block_p95_cuda_active_ms=values[min(len(values) - 1, int(len(values) * 0.95))],
        scopes=rows,
        evidence=evidence,
        eos_cost_status=(
            "collected"
            if eos_intervals is not None
            else "unknown: missing attributed EOS scope"
        ),
        eos_cuda_active_ms=(
            sum(b - a for a, b in eos_intervals) / 1000
            if eos_intervals is not None
            else None
        ),
        eos_evidence=eos_evidence,
        blocking_status="requires timeline review: active kernel time does not prove the full DiT stall",
    )


def strict_dit_evidence(root, size):
    rows = []
    for rank in range(size):
        path = root / f"dit-sp{size}" / f"rank{rank}" / "summary.json"
        value = (
            json.loads(path.read_text()).get("strict_replay_passed")
            if path.is_file()
            else None
        )
        rows.append(dict(rank=rank, strict_replay_passed=value))
    status = (
        "failed"
        if any(row["strict_replay_passed"] is False for row in rows)
        else (
            "passed"
            if all(row["strict_replay_passed"] is True for row in rows)
            else "unknown"
        )
    )
    return dict(status=status, ranks=rows)


def component_rows(root, prefix, sizes):
    rows = []
    for size in sizes:
        directory = root / f"{prefix}{size}"
        paths = [directory / f"rank{r}" / "measured.json" for r in range(size)]
        if not all(p.is_file() for p in paths):
            rows.append(dict(gpus=size, status="missing", median_seconds=None))
            continue
        runs = [json.loads(p.read_text()) for p in paths]
        if any(len(r) != 5 for r in runs):
            raise ValueError("Expected five unprofiled repetitions on every rank")
        times = [max(r[i]["seconds"] for r in runs) for i in range(5)]
        median = statistics.median(times)
        rows.append(
            dict(
                gpus=size,
                status="collected",
                seconds=times,
                median_seconds=median,
                stable=(max(times) - min(times)) / median <= 0.1,
            )
        )
    baseline = rows[0]["median_seconds"]
    for row in rows:
        row["speedup_vs_one"] = (
            baseline / row["median_seconds"]
            if baseline and row["median_seconds"]
            else None
        )
    return rows


def soak_evidence(root):
    selection = root / "provisional-selection.json"
    if not selection.is_file():
        return dict(status="missing")
    name = json.loads(selection.read_text())["layout"][0]
    cell = root / f"{name}-paced-soak"
    events_path = cell / "measured-0/events.jsonl"
    summary_path = cell / "measured-0/summary.json"
    if not events_path.is_file() or not summary_path.is_file():
        return dict(status="missing", layout=name)
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    sid = next(e["message"]["session_id"] for e in events if e["kind"] == "ready")
    deliveries = [e for e in events if e["kind"] == "frames"]
    steps, output_snapshots = [], {}
    for line in (cell / "server.log").read_text().splitlines():
        if "zing_source_step " in line:
            row = json.loads(line.split("zing_source_step ", 1)[1])
            if row["session_id"] == sid:
                steps.append(row)
        elif "zing_source_output " in line:
            row = json.loads(line.split("zing_source_output ", 1)[1])
            if row["session_id"] == sid:
                output_snapshots[row["request_id"]] = row["memory"]
    for row in steps:
        if row.get("request_id") in output_snapshots:
            row["memory"]["output_transport"] = output_snapshots[row["request_id"]]
            row["memory"].update(
                {
                    key: value
                    for key, value in output_snapshots[row["request_id"]].items()
                    if key.startswith("gpu_")
                }
            )
    windows = []
    for start in (0, 30, 60, 90):
        latency = sorted(
            e["block_capture_to_output_seconds"]
            for e in deliveries
            if start <= e["header"]["output_frame_start"] / 24 < start + 30
        )
        ranks = {}
        for rank in sorted({r["rank"] for r in steps}):
            samples = [
                r["memory"]
                for r in steps
                if r["rank"] == rank
                and start <= (r["source_frames"] - 1) / 24 < start + 30
            ]
            ranks[rank] = {
                k: dict(
                    first=samples[0][k],
                    last=samples[-1][k],
                    maximum=max(s[k] for s in samples),
                )
                for k in (
                    "gpu_allocated_bytes",
                    "gpu_reserved_bytes",
                    "gpu_peak_allocated_bytes",
                    "gpu_peak_reserved_bytes",
                    "codec_pixel_frames",
                    "bridge_bytes",
                    "pending_blocks",
                )
                if samples and all(k in s for s in samples)
            }
            queues = [s["encoder_queues"] for s in samples if "encoder_queues" in s]
            output_queues = [
                sample["output_transport"]
                for sample in samples
                if "output_transport" in sample
            ]
            if output_queues:
                ranks[rank]["output_transport_maxima"] = {
                    key: max(q[key] for q in output_queues) for key in output_queues[0]
                }
            if queues:
                ranks[rank]["encoder_queue_maxima"] = {
                    k: max(q[k] for q in queues) for k in queues[0]
                }
        windows.append(
            dict(
                source_seconds=[start, start + 30],
                blocks=len(latency),
                p50_seconds=statistics.median(latency) if latency else None,
                p95_seconds=(
                    latency[min(len(latency) - 1, int(len(latency) * 0.95))]
                    if latency
                    else None
                ),
                ranks=ranks,
            )
        )
    return dict(
        status="collected_requires_growth_review",
        layout=name,
        summary=json.loads(summary_path.read_text()),
        windows=windows,
        encoder_memory_log=str(cell / "encoder.log"),
        semantics="Windows follow source time; first/last/max are observations, not an automatic stability pass",
    )


def analyze(root):
    plan_path = root / "matrix-plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
    visible_gpus = plan.get("visible_gpus")
    dit = component_rows(root, "dit-sp", (1, 2, 4, 8))
    for row in dit:
        row["numerical_gate"] = strict_dit_evidence(root, row["gpus"])
    encoder = component_rows(root, "encoder-", (1, 2, 4))
    endpoint = []
    specs = (
        selected_layouts([row[0] for row in plan["layouts"]])
        if "layouts" in plan
        else layouts()
    )
    for name, sp, enc, mode in specs:
        for paced in (False, True):
            cell = root / (name + ("-paced" if paced else "-unpaced"))
            paths = [cell / f"measured-{i}" / "summary.json" for i in range(5)]
            entry = dict(
                name=name,
                paced=paced,
                active_gpus=sp + enc,
                sp=sp,
                encoder_gpus=enc,
                mode=mode,
            )
            if not all(p.is_file() for p in paths):
                endpoint.append(dict(entry, status="missing"))
                continue
            runs = [json.loads(p.read_text()) for p in paths]
            times = [r["elapsed_seconds"] for r in runs]
            p95s = [r["block_capture_to_output_p95_seconds"] for r in runs]
            valid = all(
                r["ended"]
                and all(
                    r[k] == 720
                    for k in ("submitted", "accepted", "consumed", "received")
                )
                for r in runs
            )
            entry.update(
                status="collected",
                output_complete=valid,
                elapsed_median=statistics.median(times),
                ttff_median=statistics.median(r["ttff_seconds"] for r in runs),
                capture_p95_median=statistics.median(
                    r["block_capture_to_output_p95_seconds"] for r in runs
                ),
                per_frame_capture_p95_median=statistics.median(
                    r["source_to_output_p95_seconds"] for r in runs
                ),
                dependency_p95_median=statistics.median(
                    r["dependency_to_output_p95_seconds"] for r in runs
                ),
                stable=(max(times) - min(times)) / statistics.median(times) <= 0.1,
                p95_stable=(max(p95s) - min(p95s)) / statistics.median(p95s) <= 0.1,
                block_p50_median=statistics.median(
                    r["block_capture_to_output_p50_seconds"] for r in runs
                ),
                playback_stall_seconds_median=statistics.median(
                    r["stall_seconds"] for r in runs
                ),
                output_hashes=[r["output_sha256"] for r in runs],
                active_gpu_seconds=statistics.median(times) * (sp + enc),
                allocated_gpu_seconds=(
                    statistics.median(times) * visible_gpus if visible_gpus else None
                ),
                kernel_overlap=overlap_evidence(cell, sp, enc),
                shared_decoder=decoder_evidence(cell),
            )
            endpoint.append(entry)
    for row in endpoint:
        if row["status"] != "collected" or row["mode"] != "overlap":
            continue
        serial = next(
            (
                r
                for r in endpoint
                if r["encoder_gpus"] == row["encoder_gpus"]
                and r["mode"] == "serial"
                and r["paced"] == row["paced"]
                and r["status"] == "collected"
            ),
            None,
        )
        row["serial_output_exact"] = (
            len(set(row["output_hashes"] + serial["output_hashes"])) == 1
            if serial
            else None
        )
        row["serial_output_gate"] = (
            "unknown"
            if serial is None
            else (
                "passed"
                if row["serial_output_exact"]
                and row["output_complete"]
                and serial["output_complete"]
                else "failed"
            )
        )
        row["latency_reduction_vs_serial"] = (
            1 - row["capture_p95_median"] / serial["capture_p95_median"]
            if serial
            else None
        )
    ratio = dit[-1]["speedup_vs_one"]
    e2, e4 = encoder[1]["median_seconds"], encoder[2]["median_seconds"]
    return dict(
        matrix_plan=plan,
        dit=dit,
        encoder=encoder,
        encoder_4_vs_2=e2 / e4 if e2 and e4 else None,
        sp8_speedup_ge_6=ratio >= 6 if ratio is not None else None,
        endpoints=endpoint,
        decoder=(
            json.loads((root / "decoder/summary.json").read_text())
            if (root / "decoder/summary.json").is_file()
            else None
        ),
        soak=soak_evidence(root),
        acceptance="not_established",
        remaining_gates=[
            "Shared-versus-dedicated moments/latent/DiT/RGB parity",
            "Lifecycle and witnessed backpressure",
            "Media and tile seams",
            "120 s growth review",
            "Decoder blocking and EOS timeline review",
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(analyze(args.root), indent=2))


if __name__ == "__main__":
    main()
