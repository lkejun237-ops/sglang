# SPDX-License-Identifier: Apache-2.0
"""Load already-replayed memory records without importing torch or pickle."""

import csv
import hashlib
import json
import math
import statistics


def sampled_series(directory):
    """Preserve recorded extrema in bounded curves; never infer unsampled peaks."""

    def reduce(points):
        if not points:
            return []
        low, high = points[0][0], points[-1][0]
        buckets = {}
        for point in points:
            buckets.setdefault(
                int(500 * (point[0] - low) / max(high - low, 0.001)), []
            ).append(point)
        return [
            point
            for values in buckets.values()
            for point in sorted(
                set(
                    [
                        tuple(min(values, key=lambda v: v[1])),
                        tuple(max(values, key=lambda v: v[1])),
                    ]
                )
            )
        ]

    records = []
    sources = []
    roots = (
        [directory / "memory"]
        if (directory / "memory").exists()
        else sorted(directory.glob("results/*/observations/memory"))
    )
    for root in roots:
        identities = [
            json.loads(p.read_text()) for p in sorted(root.glob("process-*.json"))
        ]
        samples = {}
        for identity in identities:
            path = root / f"samples-{identity['role']}-{identity['pid']}.jsonl"
            samples[identity["pid"]] = [
                json.loads(line) for line in path.read_text().splitlines()
            ]
            sources.extend([path, root / f"process-{identity['pid']}.json"])
        ready = [
            v["monotonic"]
            for values in samples.values()
            for v in values
            if v.get("kind") == "memory_command" and v.get("label") == "ready"
        ]
        begin = max(ready) if ready else float("-inf")
        by_device = {}
        for identity in identities:
            visible = (identity["visible_devices"] or "").split(",")
            key = (
                visible[identity["device"]]
                if len(visible) > identity["device"] and visible[identity["device"]]
                else str(identity["device"])
            )
            group = by_device.setdefault(key, dict(label=key, series=[]))
            for metric in ["allocated", "reserved"]:
                points = [
                    [v["monotonic"], v[metric]]
                    for v in samples[identity["pid"]]
                    if v["monotonic"] >= begin
                ]
                if points:
                    group["series"].append(
                        dict(
                            name=f"{identity['role']} {identity['device']} {metric}",
                            points=reduce(points),
                        )
                    )
        configuration = directory if root == directory / "memory" else root.parents[1]
        nvml = configuration / "gpu-memory.jsonl"
        if nvml.exists():
            sources.append(nvml)
            values = {}
            for line in nvml.read_text().splitlines():
                sample = json.loads(line)
                if sample["monotonic"] < begin:
                    continue
                for text in sample["rows"]:
                    index, uuid, gpu_name, used, _ = next(csv.reader([text]))
                    key = uuid.strip() if uuid.strip() in by_device else index.strip()
                    if key in by_device:
                        by_device[key]["label"] = gpu_name.strip() + " / " + key
                        values.setdefault(key, []).append(
                            [sample["monotonic"], float(used) * 2**20]
                        )
            for key, points in values.items():
                by_device[key]["series"].insert(
                    0, dict(name="NVML", points=reduce(points))
                )
        records.extend(
            dict(mode=configuration.name, **value)
            for value in by_device.values()
            if value["series"]
        )
    return records, sources


def summarize_counters(directory, name):
    """Summarize this configuration's checkpoint counters and optional NVML log."""
    memory = directory / "memory"
    processes = []
    physical = {}
    for path in sorted(memory.glob("process-*.json")):
        identity = json.loads(path.read_text())
        acks = [
            json.loads(p.read_text())
            for p in memory.glob(f"ack-*-{identity['pid']}.json")
        ]
        ready = next((v["counters"] for v in acks if v["label"] == "ready"), {})
        label = identity["role"] + " / device " + str(identity["device"])
        visible = (identity["visible_devices"] or "").split(",")
        device = (
            visible[identity["device"]]
            if len(visible) > identity["device"] and visible[identity["device"]]
            else str(identity["device"])
        )
        physical.setdefault(device, []).append(label)
        row = dict(
            role=label,
            pid=identity["pid"],
            device=identity["device"],
            ready_allocated=ready.get("allocated"),
            ready_reserved=ready.get("reserved"),
        )
        for key in ["allocated", "reserved"]:
            row["peak_" + key] = max(
                (v["counters"]["peak_" + key] for v in acks if v["label"] != "ready"),
                default=None,
            )
        processes.append(row)
    gpu_samples = {}
    source = directory / "gpu-memory.jsonl"
    if source.exists():
        for line in source.read_text().splitlines():
            for text in json.loads(line)["rows"]:
                index, uuid, gpu_name, used, _ = next(csv.reader([text]))
                key = uuid.strip() if uuid.strip() in physical else index.strip()
                if key in physical:
                    gpu_samples.setdefault(key, dict(name=gpu_name.strip(), values=[]))[
                        "values"
                    ].append(float(used) * 2**20)
    gpus = [
        dict(
            role=" + ".join(physical[key]),
            name=value["name"],
            ready_sample=None,
            peak_sample=max(value["values"]),
            increment_sample=None,
            complete=False,
        )
        for key, value in gpu_samples.items()
    ]
    return (
        {
            name: dict(
                processes=processes,
                gpus=gpus,
                scope="Observed checkpoint and NVML sampling windows; no inferred pre-request NVML baseline",
            )
        }
        if processes
        else None
    )


def summarize_timing(rows):
    """Recompute caches from the per-request values preserved by the exporter."""

    def stats(values):
        if not values or not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Invalid independent timing samples")
        return dict(median=statistics.median(values), min=min(values), max=max(values))

    for row in rows:
        requests = row["requests"]
        if not requests:
            raise ValueError("Missing independent timing requests")
        row["partial"] = (
            row["partial"] or row["count"] != len(requests) or bool(row.get("invalid"))
        )
        row["count"] = len(requests)
        row["client"] = {
            key: stats([r["client"][key] for r in requests])
            for key in ["ttff_seconds", "e2e_seconds", "post_ttff_fps"]
        }
        stages = set(requests[0]["stage_cuda_ms"])
        if any(set(r["stage_cuda_ms"]) != stages for r in requests):
            raise ValueError("Inconsistent independent timing stage coverage")
        row["stages"] = {
            key: stats([r["stage_cuda_ms"][key] for r in requests]) for key in stages
        }
    return rows


def check_replay(value):
    if value["schema_version"] != 1 or not value["profiles"]:
        raise ValueError("Expected nonempty version 1 allocator replay")
    identities = set()
    for row in value["profiles"]:
        identity = (row["case"], row["pid"], row["device"])
        if identity in identities or row["closure"] is not True:
            raise ValueError("Duplicate or unclosed memory profile")
        identities.add(identity)
        for key in (
            "baseline_bytes",
            "new_peak_bytes",
            "new_active_peak_bytes",
            "net_peak_bytes",
            "reconstructed_peak_bytes",
            "peak_seconds",
            "recorded_start_seconds",
            "trace_events",
        ):
            if not isinstance(row[key], (int, float)) or not math.isfinite(row[key]):
                raise ValueError("Non-finite memory statistic: " + key)
        if not isinstance(row["partial"], bool) or any(
            row[k] < 0
            for k in [
                "baseline_bytes",
                "new_peak_bytes",
                "new_active_peak_bytes",
                "net_peak_bytes",
            ]
        ):
            raise ValueError("Invalid memory accounting")
        if (
            row["new_active_peak_bytes"] < row["new_peak_bytes"]
            or row["reconstructed_peak_bytes"]
            != row["baseline_bytes"] + row["net_peak_bytes"]
        ):
            raise ValueError("Inconsistent memory peaks")
        if not row["points"] or any(
            len(p) < 4 or not all(math.isfinite(n) for n in p) for p in row["points"]
        ):
            raise ValueError("Invalid allocation timeline")
        if any(b[0] < a[0] for a, b in zip(row["points"], row["points"][1:])):
            raise ValueError("Allocation timeline is not ordered")
        if (
            sum(g["bytes"] for g in row["peak_stacks"]) + row["other_peak_stack_bytes"]
            != row["new_peak_bytes"]
        ):
            raise ValueError("Allocation stack bytes do not conserve")


def load_memory_run(name, directory):
    path = directory / "MEMORY_REPLAY.json"
    replay = json.loads(path.read_text())
    check_replay(replay)
    result = dict(
        kind="runtime_memory",
        name=name,
        replay=replay,
        sources={path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
    )
    for filename, key in [
        ("COUNTER_OVERVIEW.json", "counters"),
        ("results/H200_TIMING.json", "independent_timing"),
        ("INDEPENDENT_TIMING.json", "independent_timing"),
        ("HARDWARE_ADMISSION.json", "hardware"),
        ("PROFILE_COMPARISON.json", "comparison"),
    ]:
        source = directory / filename
        if source.exists():
            if key in result:
                raise ValueError("Multiple independent timing inputs; choose one")
            result[key] = json.loads(source.read_text())
            result["sources"][filename] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
    if "counters" not in result:
        counters = summarize_counters(directory, name)
        if counters:
            result["counters"] = counters
            for source in [
                *sorted((directory / "memory").glob("*.json")),
                *(
                    [directory / "gpu-memory.jsonl"]
                    if (directory / "gpu-memory.jsonl").exists()
                    else []
                ),
            ]:
                result["sources"][source.relative_to(directory).as_posix()] = (
                    hashlib.sha256(source.read_bytes()).hexdigest()
                )
    if "independent_timing" in result:
        summarize_timing(result["independent_timing"])
    series, sources = sampled_series(directory)
    if series:
        result["counter_series"] = series
        result["sources"].update(
            {
                p.relative_to(directory).as_posix(): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in sources
            }
        )
    return result
