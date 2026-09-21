#!/usr/bin/env python3
"""Summarize per-rank torch.profiler Chrome traces without adding nested timings.

Usage: python analyze_sp_profile.py TRACE_DIRECTORY --output summary.json
Only Chrome complete (X) events are timed. Units in the output are milliseconds.
Name matches are evidence candidates, not proof of a particular bottleneck.
"""

import argparse
import gzip
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

PATTERNS = {
    "collectives": r"nccl|all.?to.?all|all.?gather|reduce.?scatter|all.?reduce|ulysses|c10d",
    "gemm": r"gemm|cublas|aten::(?:mm|bmm|addmm|linear)$|nvjet",
    "sdpa": r"scaled_dot_product|flash.*att|flash_(?:fwd|bwd)|attention|fmha",
    "adaln_candidate": r"adaln|ada_layer|adaptive.*norm|modulat",
    "kv_concat_candidate": r"kv.?cache|kv.?concat|aten::cat$|concat",
    "rope_candidate": r"rope|rotary",
    "cpu_sync": r"synchronize|cudaEventSynchronize|cudaStreamSynchronize|aten::item$|_local_scalar_dense",
    "decode": r"decode|taehv|vae",
    "device_to_host": r"dtoh|device.to.host",
}
REGION_PATTERN = re.compile(
    r"^Zing\.(?:AdaLN|KV_history_concat|build_RoPE|RoPE_and_KV_append|Ulysses|SDPA|video_decode|audio_decode|conditioning)",
    re.I,
)
BLOCK_PATTERN = re.compile(r"causal[ _:/-]*block|causal.*block_index", re.I)


def region_block_evidence(events, block_events, region_events, limit, classify):
    """Correlate kernels to individual region occurrences, then label CPU parents.

    GPU timestamps need not fall inside host ranges. Ambiguous IDs retain the
    same rejection semantics as the existing block/region summaries.
    """
    steps = [
        e
        for e in events
        if classify(e) == "cpu_annotation"
        and str(e.get("name", "")).startswith("Zing.denoise_step.")
    ]
    forwards = [
        e
        for e in events
        if classify(e) == "cpu_annotation"
        and str(e.get("name", "")).startswith("Zing.forward.")
    ]

    def parent(scope, candidates):
        matches = [
            e
            for e in candidates
            if (e.get("pid"), e.get("tid")) == (scope.get("pid"), scope.get("tid"))
            and e["ts"] <= scope["ts"]
            and scope["ts"] + scope["dur"] <= e["ts"] + e["dur"]
        ]
        if not matches:
            return None
        matches.sort(key=lambda e: e["dur"])
        if len(matches) > 1 and matches[0]["dur"] == matches[1]["dur"]:
            return None
        return matches[0]

    stats, status = correlate_blocks(events, region_events, limit, classify)
    rows = []
    for scope, evidence in zip(region_events, stats):
        block = parent(scope, block_events)
        step = parent(scope, steps)
        forward = parent(scope, forwards)
        name = block["name"] if block else None
        history = re.search(r"history_rows(\d+)", name or "")
        latent = re.search(r"\.latent(\d+)-(\d+)", name or "")
        rows.append(
            {
                "region": scope["name"],
                "block": name,
                "block_start_us": block["ts"] if block else None,
                "history_rows": int(history[1]) if history else None,
                "latent_span": [int(latent[1]), int(latent[2])] if latent else None,
                "step": step["name"] if step else None,
                "forward": forward["name"] if forward else None,
                "pid": scope.get("pid"),
                "tid": scope.get("tid"),
                "host_start_us": scope["ts"],
                "host_duration_ms": scope["dur"] / 1000,
                **evidence,
            }
        )
    return {
        "rows": rows,
        "correlation_status": status,
        "semantics": "Per region occurrence; block/step/forward are same-thread enclosing host scopes. Null parents remain unknown. GPU kernels use launch IDs, not timestamp clipping. Do not add these rows to the separate block or global region views.",
    }


def union_us(events):
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in events)
    total = 0.0
    end = float("-inf")
    for start, stop in intervals:
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def timing(events):
    if not events:
        return {"count": 0, "inclusive_sum_ms": 0, "interval_union_ms": 0, "span_ms": 0}
    return {
        "count": len(events),
        "inclusive_sum_ms": sum(e["dur"] for e in events) / 1000,
        "interval_union_ms": union_us(events) / 1000,
        "span_ms": (
            max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events)
        )
        / 1000,
    }


def event_kind(event, gpu_pids=(), gpu_threads=()):
    category = str(event.get("cat", "")).lower()
    if "kernel" in category:
        return "cuda_kernel"
    if "gpu_memcpy" in category or "gpu_memset" in category:
        return "cuda_memory"
    is_gpu = (
        event.get("pid") in gpu_pids
        or (event.get("pid"), event.get("tid")) in gpu_threads
    )
    if "gpu_user_annotation" in category or (is_gpu and "user_annotation" in category):
        return "cuda_annotation"
    if is_gpu:
        return "cuda_other"
    if "cuda_runtime" in category or "cuda_driver" in category:
        return "cpu_cuda_runtime"
    if "cpu_op" in category:
        return "cpu_op"
    if "user_annotation" in category:
        return "cpu_annotation"
    return "other"


def top_names(events, limit):
    grouped = defaultdict(list)
    for event in events:
        grouped[str(event.get("name", "<unnamed>"))].append(event)
    ordered = sorted(
        grouped.items(), key=lambda pair: sum(e["dur"] for e in pair[1]), reverse=True
    )
    return [{"name": name, **timing(group)} for name, group in ordered[:limit]]


def trace_id(event, field):
    for key, value in event.get("args", {}).items():
        if str(key).lower().replace(" ", "").replace("_", "") == field:
            return str(value) if value is not None else None
    return None


def correlate_blocks(events, block_events, limit, kind=event_kind, group_by_name=False):
    """Attribute launches by CPU nesting, then kernels by IDs, never GPU time."""
    by_thread = defaultdict(list)
    for index, block in enumerate(block_events):
        by_thread[(block.get("pid"), block.get("tid"))].append((index, block))
    cpu = [
        e for e in events if kind(e) in ("cpu_op", "cpu_annotation", "cpu_cuda_runtime")
    ]
    cpu_threads = defaultdict(list)
    for event in cpu:
        cpu_threads[(event.get("pid"), event.get("tid"))].append(event)
    owners = {}
    for thread, group in cpu_threads.items():
        scopes = sorted(by_thread[thread], key=lambda pair: pair[1]["ts"])
        cursor, active = 0, []
        for event in sorted(group, key=lambda e: e["ts"]):
            while cursor < len(scopes) and scopes[cursor][1]["ts"] <= event["ts"]:
                active.append(scopes[cursor])
                cursor += 1
            active = [(i, b) for i, b in active if b["ts"] + b["dur"] >= event["ts"]]
            enclosing = [
                (i, b)
                for i, b in active
                if event["ts"] + event["dur"] <= b["ts"] + b["dur"]
            ]
            owners[id(event)] = (
                min(enclosing, key=lambda pair: pair[1]["dur"])[0]
                if enclosing
                else None
            )
    external = defaultdict(set)
    for event in cpu:
        if kind(event) == "cpu_cuda_runtime":
            continue
        key = trace_id(event, "externalid")
        if key is not None:
            external[key].add(owners[id(event)])

    def unique(mapping, key):
        values = mapping.get(key, set())
        return next(iter(values)) if len(values) == 1 and None not in values else None

    correlations = defaultdict(set)
    for event in cpu:
        if kind(event) != "cpu_cuda_runtime":
            continue
        owner = owners[id(event)]
        external_owner = unique(external, trace_id(event, "externalid"))
        # Runtime on a launch worker may be outside the main CPU scope.
        if owner is None:
            owner = external_owner
        elif external_owner is not None and external_owner != owner:
            owner = None
        key = trace_id(event, "correlation")
        if key is not None:
            correlations[key].add(owner)
    assigned = defaultdict(list)
    methods = defaultdict(lambda: defaultdict(int))
    status = defaultdict(int)
    unassigned = []
    for event in events:
        if kind(event) != "cuda_kernel":
            continue
        corr_key = trace_id(event, "correlation")
        ext_key = trace_id(event, "externalid")
        corr_owner = unique(correlations, corr_key)
        ext_owner = unique(external, ext_key)
        if corr_owner is not None and ext_owner is not None and corr_owner != ext_owner:
            status["conflicting_ids"] += 1
            unassigned.append(event)
            continue
        # A present but ambiguous launch ID cannot be rescued by a weaker ID.
        if corr_key in correlations and corr_owner is None:
            status["ambiguous_or_unscoped_correlation"] += 1
            unassigned.append(event)
            continue
        owner = corr_owner if corr_owner is not None else ext_owner
        if owner is None:
            status["unmatched"] += 1
            unassigned.append(event)
            continue
        method = "runtime_correlation" if corr_owner is not None else "external_id"
        assigned[owner].append(event)
        methods[owner][method] += 1
        status["matched"] += 1
    scope_groups = defaultdict(list)
    if group_by_name:
        merged, merged_methods = (
            defaultdict(list),
            defaultdict(lambda: defaultdict(int)),
        )
        for index, scope in enumerate(block_events):
            name = scope["name"]
            scope_groups[name].append(scope)
            merged[name].extend(assigned[index])
            for method, count in methods[index].items():
                merged_methods[name][method] += count
        assigned, methods = merged, merged_methods
    keys = sorted(scope_groups) if group_by_name else range(len(block_events))
    results = []
    collective_regex = re.compile(PATTERNS["collectives"], re.I)
    for index in keys:
        kernels = assigned[index]
        collectives = [
            e for e in kernels if collective_regex.search(str(e.get("name", "")))
        ]
        devices = defaultdict(list)
        for event in kernels:
            devices[
                str(event.get("args", {}).get("device", event.get("pid", "unknown")))
            ].append(event)
        results.append(
            {
                **(
                    {"name": index, "host_scopes": timing(scope_groups[index])}
                    if group_by_name
                    else {}
                ),
                "attribution_methods": dict(methods[index]),
                "cuda_kernels": {
                    **timing(kernels),
                    "top_names": top_names(kernels, limit),
                },
                "cuda_kernels_per_device": {
                    device: timing(group) for device, group in devices.items()
                },
                "cuda_collectives": {
                    **timing(collectives),
                    "top_names": top_names(collectives, limit),
                },
            }
        )
    return results, {
        **dict(status),
        "unassigned_cuda_kernels": {
            **timing(unassigned),
            "top_names": top_names(unassigned, limit),
        },
    }


def summarize(path, limit):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        trace = json.load(handle)
    raw_events = trace.get("traceEvents") if isinstance(trace, dict) else trace
    if not isinstance(raw_events, list):
        return None  # Ignore request/result JSON next to traces.
    events = [
        e
        for e in raw_events
        if isinstance(e, dict)
        and e.get("ph") == "X"
        and isinstance(e.get("ts"), (int, float))
        and isinstance(e.get("dur"), (int, float))
        and e["dur"] >= 0
    ]
    if not events:
        return None
    metadata = [
        {
            "name": e.get("name"),
            "pid": e.get("pid"),
            "tid": e.get("tid"),
            "args": e.get("args"),
        }
        for e in raw_events
        if isinstance(e, dict)
        and e.get("ph") == "M"
        and e.get("name") in ("process_name", "thread_name", "process_labels")
    ]
    gpu_pids = {
        e.get("pid")
        for e in metadata
        if e.get("name") in ("process_labels", "process_name")
        and re.search(r"\b(?:GPU|CUDA)\b", str(e.get("args", {})), re.I)
    }
    gpu_threads = {
        (e.get("pid"), e.get("tid"))
        for e in metadata
        if e.get("name") == "thread_name"
        and re.search(r"\bstream\s+\d+", str(e.get("args", {}).get("name", "")), re.I)
    }

    def classify(event):
        return event_kind(event, gpu_pids, gpu_threads)

    grouped = defaultdict(list)
    for event in events:
        grouped[classify(event)].append(event)
    candidates = {}
    for label, pattern in PATTERNS.items():
        regex = re.compile(pattern, re.I)
        matches = {}
        for kind, group in grouped.items():
            selected = [e for e in group if regex.search(str(e.get("name", "")))]
            if selected:
                matches[kind] = {
                    **timing(selected),
                    "top_names": top_names(selected, limit),
                }
        candidates[label] = matches
    origin = min(e["ts"] for e in events)
    blocks = []
    block_events = []
    for e in sorted(events, key=lambda item: item["ts"]):
        if classify(e) not in (
            "cpu_annotation",
            "cpu_op",
        ) or not BLOCK_PATTERN.search(str(e.get("name", ""))):
            continue
        block_events.append(e)
        blocks.append(
            {
                "name": e.get("name"),
                "pid": e.get("pid"),
                "tid": e.get("tid"),
                "start_from_trace_ms": (e["ts"] - origin) / 1000,
                "cpu_span_ms": e["dur"] / 1000,
                "args": e.get("args", {}),
            }
        )
    block_gpu, correlation_status = correlate_blocks(
        events, block_events, limit, classify
    )
    for block, gpu in zip(blocks, block_gpu):
        block.update(gpu)
    region_events = [
        e
        for e in events
        if classify(e) == "cpu_annotation"
        and REGION_PATTERN.search(str(e.get("name", "")))
    ]
    regions, region_status = correlate_blocks(
        events, region_events, limit, classify, group_by_name=True
    )
    devices = {}
    for e in grouped["cuda_kernel"] + grouped["cuda_memory"]:
        args = e.get("args", {})
        device = str(args.get("device", e.get("pid", "unknown")))
        devices.setdefault(device, []).append(e)
    return {
        "file": str(path.resolve()),
        "distributed_info": trace.get("distributedInfo")
        if isinstance(trace, dict)
        else None,
        "device_properties": trace.get("deviceProperties")
        if isinstance(trace, dict)
        else None,
        "metadata": metadata,
        "complete_event_count": len(events),
        "unhandled_begin_end_event_count": sum(
            isinstance(e, dict) and e.get("ph") in ("B", "E") for e in raw_events
        ),
        "trace_span_ms": timing(events)["span_ms"],
        "by_kind": {
            kind: {**timing(group), "top_names": top_names(group, limit)}
            for kind, group in sorted(grouped.items())
        },
        "cuda_activity_per_device": {
            device: timing(group) for device, group in devices.items()
        },
        "name_match_evidence": candidates,
        "gpu_annotation_spans": [
            {
                "name": e.get("name"),
                "pid": e.get("pid"),
                "tid": e.get("tid"),
                "start_from_trace_ms": (e["ts"] - origin) / 1000,
                "gpu_annotation_span_ms": e["dur"] / 1000,
                "args": e.get("args", {}),
            }
            for e in grouped["cuda_annotation"]
        ],
        "causal_block_spans": blocks,
        "region_by_causal_block": region_block_evidence(
            events, block_events, region_events, limit, classify
        ),
        "causal_block_kernel_correlation_status": correlation_status,
        "named_region_cuda_evidence": {
            "selection_regex": REGION_PATTERN.pattern,
            "semantics": "Each kernel belongs only to its innermost selected host region. Name totals merge all occurrences. Outer scopes exclude inner-region kernels; host_scopes remain inclusive CPU intervals. Unassigned includes work outside selected regions and unresolved/ambiguous IDs. This is a separate view from causal blocks; do not sum the two views.",
            "regions": regions,
            "correlation_status": region_status,
        },
        "warnings": (
            []
            if blocks
            else [
                "No named causal block spans found; history-dependent timing cannot be inferred."
            ]
        ),
    }


def summarize_safe(path, limit):
    try:
        return summarize(path, limit), None
    except (OSError, ValueError, TypeError) as exc:
        return None, {"file": str(path), "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Trace directory (recursive), or a single trace file"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel trace processes; each worker loads one complete trace into memory",
    )
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.input.exists():
        parser.error("input does not exist")
    paths = (
        sorted(set(args.input.rglob("*.json")) | set(args.input.rglob("*.json.gz")))
        if args.input.is_dir()
        else [args.input]
    )
    paths = [path for path in paths if path.resolve() != args.output.resolve()]
    if args.workers == 1:
        results = map(summarize_safe, paths, repeat(args.top))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # map preserves input order, including error records.
            results = list(executor.map(summarize_safe, paths, repeat(args.top)))
    summaries, errors = [], []
    for result, error in results:
        if result is not None:
            summaries.append(result)
        if error is not None:
            errors.append(error)
    report = {
        "units": "milliseconds",
        "semantics": [
            "inclusive_sum_ms sums event durations; nested CPU ops and concurrent CUDA streams can double count. It is not wall time.",
            "interval_union_ms merges overlapping intervals within the reported group, across its threads/streams; it is observed activity coverage, not hardware utilization.",
            "span_ms is first-start to last-end, including gaps; trace_span_ms is profiler coverage, not unprofiled API latency.",
            "Do not sum groups, candidates, ranks or CPU/CUDA durations to obtain wall time; candidate categories overlap.",
            "Causal block CPU spans measure host scopes; asynchronous GPU completion requires synchronized scopes or separate CUDA-event measurements.",
            "GPU annotations are separate device-side scopes, identified by category or GPU process/stream metadata. They are excluded from host causal blocks and External id ownership; do not sum mirrored host and device annotations.",
            "Block CUDA attribution uses same-thread nested CPU scopes plus runtime correlation / External id; GPU timestamps are never clipped to CPU blocks. Ambiguous or conflicting IDs remain unattributed. Nested block scopes assign to the innermost block only. CUDA graphs may lack resolvable IDs.",
            "Name matches identify candidates only. Generic elementwise kernels cannot be attributed to AdaLN/RoPE without explicit annotations or correlation analysis.",
            "Profiler overhead is not corrected. Compare against separately measured warm unprofiled runs.",
        ],
        "traces": summaries,
        "read_errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "trace_count": len(summaries),
                "read_error_count": len(errors),
            }
        )
    )
    if errors or not summaries:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
