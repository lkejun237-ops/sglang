# SPDX-License-Identifier: Apache-2.0
"""Read original stall receipts, delivery events, stage spans and memory evidence.

This read-only collector never declares acceptance. Run after writers stop;
formal/warmup/diagnostic requests, cases and ranks remain separate. Nested stages
must not be added together as E2E. Driver samples are lower bounds on true peaks.
"""

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import re
import statistics
import sys
import time
from pathlib import Path

P = Path


def reconstruct_source_latencies(events, paced):
    captures = {}
    for event in events:
        if event.get("kind") != "source_send":
            continue
        for frame in range(event["start"], event["start"] + event["count"]):
            captures[frame] = event["capture_seconds"] + (
                (frame - event["start"]) / 24 if paced else 0
            )
    values = []
    for event in events:
        if event.get("kind") != "frames":
            continue
        h = event["header"]
        for frame in range(
            h["output_frame_start"], h["output_frame_start"] + h["num_frames"]
        ):
            values.append(event["received_seconds"] - captures[frame])
    return values


def stats(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return dict(count=0, median=None, p95=None, p99=None, min=None, max=None)

    def q(t):
        pos = (len(values) - 1) * t
        i = int(pos)
        j = min(i + 1, len(values) - 1)
        return values[i] + (values[j] - values[i]) * (pos - i)

    return dict(
        count=len(values),
        median=statistics.median(values),
        p95=q(0.95),
        p99=q(0.99),
        min=values[0],
        max=values[-1],
    )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--root", action="append", required=True, help="Unique label=/absolute/frozen/run"
)
parser.add_argument("--harness-source", type=Path, required=True)
parser.add_argument("--source-revision", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.harness_source / "benchmark/zing_h3_causal"))
importlib.import_module("frozen_source").verify_source(
    args.harness_source, args.source_revision
)
analysis_path = (
    args.harness_source / "benchmark/zing_h3_causal/analyze_stream_stalls.py"
)
spec = importlib.util.spec_from_file_location("frozen_delivery_analysis", analysis_path)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
roots = {}
for item in args.root:
    label, separator, path = item.partition("=")
    run_root = Path(path)
    if (
        not separator
        or not label
        or label in roots
        or not run_root.is_absolute()
        or not (run_root / "work/results").is_dir()
        or not (run_root / "CPU_READY.json").is_file()
    ):
        raise ValueError(
            "Expected unique labels and existing absolute frozen run roots"
        )
    roots[label] = run_root
all_data = dict(
    time=time.time(),
    regression_accepted=False,
    scope="Read-only original metadata analysis; no GPU commands or tensor reads. Collection is not acceptance.",
    harness_revision=args.source_revision,
    delivery_analysis_sha256=hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
    roots={},
)
for root, run_root in roots.items():
    R = run_root / "work/results"
    out = dict(groups=[], stage_groups=[], memory_groups=[], delivery_reanalysis=[])
    groups = collections.defaultdict(list)
    windows = []
    for p in R.rglob("requests.jsonl"):
        for line in p.read_text().splitlines():
            if not line:
                continue
            r = json.loads(line)
            event_path = (
                P(r["events_path"])
                if r.get("events_path")
                else (P(r["media_path"]).parent if r.get("media_path") else p.parent)
                / "events.jsonl"
            )
            raw = event_path.read_bytes() if event_path.exists() else b""
            prior = r.get("stall_check") or {}
            events = [json.loads(line) for line in raw.decode().splitlines() if line]
            updated = analysis.analyze_events(
                events,
                fps=prior.get("fps", 24),
                startup_buffer_seconds=prior.get("startup_buffer_seconds", 0.25),
                stall_threshold_seconds=prior.get("stall_threshold_seconds", 0.25),
            )
            updated = {k: v for k, v in updated.items() if k != "underflows"}
            out["delivery_reanalysis"].append(
                dict(
                    request_id=r["request_id"],
                    event_path=str(event_path),
                    event_sha256=hashlib.sha256(raw).hexdigest(),
                    original_stall_check=prior,
                    recomputed=updated,
                )
            )
            r["stall_check"] = updated
            values = reconstruct_source_latencies(
                events, r.get("input_mode") == "paced"
            )
            observed = stats(values)
            expected = (r.get("client_observation") or {}).get(
                "source_to_output_seconds", {}
            )
            if r.get("success") and expected.get("count"):
                assert observed["count"] == expected["count"] and all(
                    abs(observed[k] - expected[k]) < 1e-6
                    for k in ["median", "p95", "max"]
                ), (
                    "Source capture-to-output reconstruction differs from original client summary"
                )
            expected_files = list((p.parent / "cases").rglob("prompt-updates.json"))
            updates = (r.get("client_observation") or {}).get("prompt_updates", [])
            expected_updates = (
                json.loads(expected_files[0].read_text())
                if len(expected_files) == 1
                else None
            )
            r["_condition_updates"] = {
                "expected_manifest": str(expected_files[0])
                if len(expected_files) == 1
                else None,
                "expected_manifest_sha256": hashlib.sha256(
                    expected_files[0].read_bytes()
                ).hexdigest()
                if len(expected_files) == 1
                else None,
                "expected_boundaries": [
                    v["source_frame_start"] for v in expected_updates
                ]
                if expected_updates is not None
                else None,
                "observed_boundaries": [v.get("source_frame_start") for v in updates],
                "status_counts": dict(
                    collections.Counter(v.get("status", "missing") for v in updates)
                ),
                "accepted": expected_updates is not None
                and len(updates) == len(expected_updates)
                and [v.get("source_frame_start") for v in updates]
                == [v["source_frame_start"] for v in expected_updates]
                and all(
                    v.get("status") == "applied"
                    and isinstance(v.get("latency_ms"), (int, float))
                    and v["latency_ms"] >= 0
                    for v in updates
                ),
            }
            r["_source_latency_values"] = values
            r["_dependency_values"] = [
                ev.get("dependency_to_output_seconds")
                for ev in events
                if ev.get("kind") == "frames"
            ]
            r["_block_capture_values"] = [
                ev.get("block_capture_to_output_seconds")
                for ev in events
                if ev.get("kind") == "frames"
            ]
            metric = r.get("metrics") or {}
            end = metric.get("e2e_seconds")
            first_rgb = metric.get("ttff_seconds")
            window = (
                end - first_rgb
                if all(isinstance(v, (int, float)) for v in [end, first_rgb])
                else 0
            )
            for key in ["underflow_count", "stall_count", "severe_underflow_count"]:
                r["stall_check"][key + "_per_minute_after_first_rgb"] = (
                    r["stall_check"].get(key, 0) * 60 / window if window > 0 else None
                )
            rel = str(p.relative_to(R))
            label = rel.split("/")[0]
            case = r.get("case_id", "unknown")
            phase = next(
                (part for part in p.parts if part.startswith(("S1-", "S2-"))), label
            )
            if phase.startswith("S1-"):
                phase = phase.rsplit("-", 1)[0]
            if phase.startswith("S2-"):
                phase = phase.rsplit("-", 1)[0]
            if re.search(r"-formal\d+$", phase):
                phase = re.sub(r"formal\d+$", "formal", phase)
            elif re.search(r"-diagnostic\d+$", phase):
                phase = re.sub(r"diagnostic\d+$", "diagnostic", phase)
            elif phase.startswith("S3-") and re.search(r"-warm\d+$", phase):
                phase = re.sub(r"warm\d+$", "warm", phase)
            # Cold process restarts remain independently identifiable, never pooled with warm requests.
            key = (case, phase, r.get("phase", "unobserved"))
            groups[key].append(r)
            obs = r.get("client_observation") or {}
            first = obs.get("first_output")
            ttff = (r.get("metrics") or {}).get("ttff_seconds")
            elapsed = obs.get("elapsed_seconds")
            if all(isinstance(v, (int, float)) for v in [first, ttff, elapsed]):
                start = first - ttff
                windows.append(
                    dict(
                        start_ns=int(start * 1e9),
                        end_ns=int((start + elapsed) * 1e9),
                        clock=obs.get("client_clock_domain"),
                        label=str(event_path.parent.relative_to(R)),
                        case_id=case,
                        phase=phase,
                    )
                )
    for (case, phase, client_phase), rows in sorted(groups.items()):
        g = dict(
            artifact_mode_counts=dict(
                collections.Counter(
                    (r.get("media") or {}).get("mode", "legacy_full_rgb_file")
                    for r in rows
                )
            ),
            case=case,
            phase=phase,
            client_phase=client_phase,
            requests=len(rows),
            successful=sum(
                bool(r.get("success") and r.get("delivery_complete")) for r in rows
            ),
            failures=[
                dict(
                    request_id=r.get("request_id"),
                    error=r.get("error"),
                    delivery_complete=r.get("delivery_complete"),
                    output_frames=r.get("output_frames"),
                )
                for r in rows
                if not (r.get("success") and r.get("delivery_complete"))
            ],
            metrics={},
        )
        for k in [
            "ttff_seconds",
            "e2e_seconds",
            "fps",
            "post_ttff_fps",
            "tpoc_seconds",
        ]:
            g["metrics"][k] = stats([(r.get("metrics") or {}).get(k) for r in rows])
        g["expected_request_count"] = (
            1
            if client_phase == "warmup"
            else 3
            if phase in ["S1-measured", "S2-paced"]
            or (
                phase.endswith("-formal")
                or (
                    phase.endswith("-component-diagnostic")
                    and "-cold-component-" not in phase
                )
            )
            or re.search(r"^(candidate|baseline)-\d+x\d+-diagnostic$", phase)
            or (phase.startswith("S3-") and phase.endswith("-warm"))
            else 1
        )
        g["acceptance_eligible"] = (
            client_phase in {"warmup", "measured"}
            and g["successful"] == g["requests"] == g["expected_request_count"]
        )
        g["metrics"]["source_to_output_seconds"] = stats(
            [v for r in rows for v in r["_source_latency_values"]]
        )
        g["metrics"]["dependency_send_to_output_seconds"] = stats(
            [v for r in rows for v in r["_dependency_values"]]
        )
        g["metrics"]["block_capture_to_output_seconds"] = stats(
            [v for r in rows for v in r["_block_capture_values"]]
        )
        g["metrics"]["icl_seconds"] = stats(
            [v for r in rows for v in r.get("icl_values_seconds", [])]
        )
        for k in [
            "source_to_output_p50_seconds",
            "source_to_output_p95_seconds",
            "tail_drain_seconds",
            "credit_wait_seconds",
            "elapsed_seconds",
            "effective_delivery_fps",
            "completion_overhead_seconds",
        ]:
            g["metrics"][k] = stats(
                [(r.get("client_observation") or {}).get(k) for r in rows]
            )
        for k in [
            "underflow_count",
            "stall_count",
            "severe_underflow_count",
            "total_underflow_seconds",
            "max_underflow_seconds",
            "max_batch_gap_seconds",
            "max_source_schedule_lag_seconds",
            "max_credit_wait_seconds",
            "underflow_count_per_minute_after_first_rgb",
            "stall_count_per_minute_after_first_rgb",
            "severe_underflow_count_per_minute_after_first_rgb",
        ]:
            g["metrics"][k] = stats([r.get("stall_check", {}).get(k) for r in rows])
        updates = [
            u
            for r in rows
            for u in (r.get("client_observation") or {}).get("prompt_updates", [])
        ]
        g["metrics"]["prompt_switch_ttff_ms"] = stats(
            [u.get("latency_ms") for u in updates if u.get("status") == "applied"]
        )
        g["prompt_update_status_counts"] = dict(
            collections.Counter(u.get("status", "missing") for u in updates)
        )
        g["condition_update_checks"] = [
            dict(request_id=r["request_id"], **r["_condition_updates"]) for r in rows
        ]
        g["condition_updates_accepted"] = all(
            r["_condition_updates"]["accepted"] for r in rows
        )
        g["request_ids"] = [r["request_id"] for r in rows]
        out["groups"].append(g)
    stages = collections.defaultdict(list)
    profiles = []
    seen = set()
    raw_files = []
    stage_examples = []

    def add_stage(row, label, source):
        # Span sequences can restart in a later session in the same worker.
        # Only deduplicate the same observed span, never another request.
        identity = tuple(
            row.get(key)
            for key in (
                "clock_domain",
                "pid",
                "request_id",
                "generation_id",
                "span_id",
                "host_enter_ns",
            )
        )
        if identity in seen:
            return
        seen.add(identity)
        block = row.get("block_index")
        phase = "not_applicable"
        if row.get("stage") == "DiT":
            phase = (
                "drain_without_output"
                if row.get("no_output")
                else "initial_four_step"
                if isinstance(block, int) and block < 3
                else "steady_two_step"
                if isinstance(block, int)
                else "unknown_block"
            )
        key = (
            label,
            row.get("stage"),
            row.get("component"),
            row.get("rank"),
            row.get("worker_role"),
            row.get("scope"),
            phase,
        )
        stages[key].append(row)
        if not stage_examples or (
            row.get("stage"),
            row.get("component"),
            phase,
        ) not in {
            (v["stage"], v["component"], v["block_phase"]) for v in stage_examples
        }:
            stage_examples.append(
                dict(
                    stage=row.get("stage"),
                    component=row.get("component"),
                    block_phase=phase,
                    row=row,
                    source=source,
                )
            )

    for p in R.rglob("profile.json"):
        d = json.loads(p.read_text())
        profiles.append(
            dict(
                path=str(p),
                session_id=d.get("session_id"),
                status=d.get("status"),
                health=d.get("health"),
                expected_ranks=d.get("expected_ranks"),
            )
        )
        for row in d.get("stages", []):
            add_stage(row, str(p.parent.relative_to(R)), str(p))
    for p in (R / "timing").rglob("*.jsonl"):
        count = 0
        for line in p.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            start = row.get("host_enter_ns")
            matches = [
                w
                for w in windows
                if w["clock"] == row.get("clock_domain")
                and isinstance(start, int)
                and w["start_ns"] <= start <= w["end_ns"]
            ]
            if (
                row.get("component") == "qwen"
                and row.get("prompt_sha256")
                == "c6cf1309cd700e5a84e18d0b1d5877b9a608141037ac40445d484398256fc56c"
            ):
                label = (
                    str(p.parent.relative_to(R))
                    + "/startup_synthetic_warmup_before_ready"
                )
            elif len(matches) == 1:
                label = matches[0]["label"]
            else:
                label = (
                    str(p.parent.relative_to(R))
                    + "/unattributed_not_pooled_with_requests"
                )
            add_stage(row, label, str(p))
            count += 1
        raw_files.append(dict(path=str(p), rows=count))
    for key, rows in stages.items():
        out["stage_groups"].append(
            dict(
                label=key[0],
                stage=key[1],
                component=key[2],
                rank=key[3],
                worker_role=key[4],
                scope=key[5],
                block_phase=key[6],
                keys=list(rows[0]),
                samples=len(rows),
                cuda_status_counts=dict(
                    collections.Counter(r.get("cuda_status") for r in rows)
                ),
                numeric_fields={
                    k: stats([r.get(k) for r in rows])
                    for k in rows[0]
                    if k.endswith(("_ms", "_seconds"))
                },
            )
        )

    request_totals = collections.defaultdict(list)
    for key, values in stages.items():
        request_totals[key[:6]].extend(values)
    out["stage_request_totals"] = []
    for key, values in request_totals.items():
        fields = {}
        for field in ["host_call_ms", "cuda_stream_ms"]:
            measured = [
                v[field] for v in values if isinstance(v.get(field), (int, float))
            ]
            fields[field] = dict(
                total=sum(measured) if len(measured) == len(values) else None,
                observed=len(measured),
                expected=len(values),
            )
        out["stage_request_totals"].append(
            dict(
                label=key[0],
                stage=key[1],
                component=key[2],
                rank=key[3],
                worker_role=key[4],
                scope=key[5],
                samples=len(values),
                fields=fields,
            )
        )
    stage_combined = collections.defaultdict(list)
    stage_labels = collections.defaultdict(set)
    for key, rows in stages.items():
        normalized = re.sub(r"(diagnostic)\d+(?=/|$)", r"\1", key[0])
        k = (normalized, *key[1:])
        stage_combined[k].extend(rows)
        stage_labels[k].add(key[0])
    out["stage_comparison_groups"] = []
    for key, rows in stage_combined.items():
        out["stage_comparison_groups"].append(
            dict(
                label=key[0],
                stage=key[1],
                component=key[2],
                rank=key[3],
                worker_role=key[4],
                scope=key[5],
                block_phase=key[6],
                request_count=len(stage_labels[key]),
                request_labels=sorted(stage_labels[key]),
                samples=len(rows),
                numeric_fields={
                    k: stats([r.get(k) for r in rows])
                    for k in rows[0]
                    if k.endswith(("_ms", "_seconds"))
                },
            )
        )
    out["stage_examples"] = stage_examples
    out["raw_timing_files"] = raw_files
    out["request_windows"] = windows
    mem = collections.defaultdict(list)
    for p in (R / "memory").glob("*.jsonl"):
        for line in p.read_text().splitlines():
            if line:
                r = json.loads(line)
                matches = [
                    w
                    for w in windows
                    if w["start_ns"] <= r["monotonic_ns"] <= w["end_ns"]
                ]
                label = (
                    matches[0]["label"]
                    if len(matches) == 1
                    else "outside_request_or_ambiguous"
                )
                mem[(r["role"], r["pid"], r["rank"], r["phase"], label)].append(r)
    for (role, pid, rank, phase, label), rows in mem.items():
        out["memory_groups"].append(
            dict(
                role=role,
                pid=pid,
                rank=rank,
                phase=phase,
                request_label=label,
                samples=len(rows),
                scope=rows[0]["scope"],
                first_monotonic_ns=rows[0]["monotonic_ns"],
                last_monotonic_ns=rows[-1]["monotonic_ns"],
                metrics={
                    k: stats([r[k] / (1024**3) for r in rows])
                    for k in [
                        "allocated_bytes",
                        "reserved_bytes",
                        "peak_allocated_bytes",
                        "peak_reserved_bytes",
                    ]
                },
            )
        )

    memory_combined = collections.defaultdict(list)
    memory_labels = collections.defaultdict(set)
    for (role, pid, rank, phase, label), rows in mem.items():
        normalized = re.sub(r"(diagnostic)\d+(?=/|$)", r"\1", label)
        k = (role, rank, phase, normalized)
        memory_combined[k].extend(rows)
        memory_labels[k].add(label)
    out["memory_comparison_groups"] = []
    for (role, rank, phase, label), rows in memory_combined.items():
        out["memory_comparison_groups"].append(
            dict(
                role=role,
                rank=rank,
                phase=phase,
                request_label=label,
                request_count=len(memory_labels[(role, rank, phase, label)]),
                pids=sorted({r["pid"] for r in rows}),
                samples=len(rows),
                scope=rows[0]["scope"],
                metrics={
                    k: stats([r[k] / 1024**3 for r in rows])
                    for k in [
                        "allocated_bytes",
                        "reserved_bytes",
                        "peak_allocated_bytes",
                        "peak_reserved_bytes",
                    ]
                },
            )
        )
    telemetry = R / "gpu-telemetry.log"
    out["driver_memory_groups"] = []
    out["driver_clock_alignment"] = dict(
        request_attribution_available=False,
        reason="No recorded telemetry timezone or wall/monotonic pairing; file mtime is never used as a clock anchor",
    )
    out["driver_unattributed_memory_groups"] = []
    if telemetry.exists():
        driver = collections.defaultdict(list)
        with telemetry.open() as stream:
            for row in csv.DictReader(stream, skipinitialspace=True):
                if row.get("memory.used [MiB]") in [None, "[N/A]"]:
                    continue
                driver[row["uuid"]].append(
                    float(row["memory.used [MiB]"].split()[0]) / 1024
                )
        for uuid, values in driver.items():
            out["driver_unattributed_memory_groups"].append(
                dict(
                    uuid=uuid,
                    driver_used_gib=stats(values),
                    first_sample_gib=values[0],
                    last_sample_gib=values[-1],
                    scope="Whole-run physical GPU samples across phases/resolutions; request/geometry attribution unavailable. Sampled maximum is a lower bound, not a request or component peak",
                )
            )
        digest = hashlib.sha256()
        with telemetry.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        out["driver_original"] = dict(
            path=str(telemetry),
            bytes=telemetry.stat().st_size,
            sha256=digest.hexdigest(),
        )
    out["gc_startup_log_receipts"] = {}
    for log in R.glob("*-dit.log"):
        receipts = {}
        for line in log.read_text(errors="replace").splitlines():
            if "ZING_STATIC_GC_READY_JSON " in line:
                row = json.loads(line.split("ZING_STATIC_GC_READY_JSON ", 1)[1])
                receipts[row["pid"]] = row
        if receipts:
            out["gc_startup_log_receipts"][log.name] = list(receipts.values())
    ready = R.parent.parent / "CPU_READY.json"
    # R is root/work/results; the immutable CPU receipt is at root.
    if ready.exists():
        out["source_revision"] = json.loads(ready.read_text())["source_revision"]
        out["runtime_source_revision"] = json.loads(ready.read_text()).get(
            "runtime_source_revision", out["source_revision"]
        )
    out["memory_probe_boundaries"] = dict(
        Qwen_process="NativeConditionEncoder.encode_device begin/end; not the later complete RPC transport peak",
        DiT_process_including_reference_VAE="ZingSourceStreamState.initialize/dispose",
        source_Encoder_process="H3ProgressiveEncoderSession.__init__/dispose",
        Decoder_process="_DecodeService.__init__/dispose return; cancellation can leave asynchronous cleanup after this snapshot",
    )
    config = R.parent.parent / "runner-config.json"
    if config.exists():
        c = json.loads(config.read_text())
        out["deployment_config"] = {
            k: c.get(k)
            for k in [
                "sp",
                "mem_fraction_static",
                "role_devices",
                "allow_shared_role_devices",
                "ports",
                "source",
                "model",
                "pipeline",
                "frozen_inputs",
                "scheduler_allocated_gpus",
                "active_gpus",
                "unused_allocated_gpus",
            ]
        }
    out["profiles"] = profiles
    for name in ["STATUS.json", "HARDWARE.json", "ALLOCATION.json"]:
        p = R / name
        if p.exists():
            out[name] = json.loads(p.read_text())
    out["gc_startup_receipts"] = {
        p.name: json.loads(p.read_text()) for p in R.glob("*-STATIC_GC.json")
    }
    out["scope_notes"] = [
        "Formal and CUDA diagnostic requests remain separate",
        "Allocator values cover complete process on current device; reference VAE shares DiT process",
        "Memory rows matched by local monotonic request intervals; outside or ambiguous rows remain explicit; Qwen synthetic warmup is excluded from real requests",
        "Stage spans can overlap and nest; do not sum into E2E",
        "Cross-host monotonic clocks are never subtracted",
    ]
    all_data["roots"][root] = out
with args.output.open("x") as stream:
    json.dump(all_data, stream, indent=2, allow_nan=False)
print(
    "REPORT_EVIDENCE_READY",
    [
        (k, len(v["groups"]), len(v["stage_groups"]))
        for k, v in all_data["roots"].items()
    ],
)
