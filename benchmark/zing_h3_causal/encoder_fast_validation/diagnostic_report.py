# SPDX-License-Identifier: Apache-2.0
"""Read request-bracketed allocator counters and Prometheus histogram deltas."""

import json
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families


def read(path):
    return json.loads(Path(path).read_text())


def samples(text):
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
    }


def component_delta(before, after):
    a, b = samples(before), samples(after)
    delta = {key: value - a.get(key, 0) for key, value in b.items()}
    groups = {}
    prefix = "sglang_zing_component_duration_seconds"
    for (name, labels), value in delta.items():
        if not name.startswith(prefix):
            continue
        label = dict(labels)
        boundary = label.pop("le", None)
        key = tuple(sorted(label.items()))
        item = groups.setdefault(key, {"count": 0, "sum": 0, "buckets": []})
        if name == prefix + "_count":
            item["count"] = value
        elif name == prefix + "_sum":
            item["sum"] = value
        elif boundary is not None:
            item["buckets"].append((float(boundary), value))
    rows = []
    for labels, item in groups.items():
        if item["count"] <= 0:
            continue
        row = dict(labels)
        row.update(count=item["count"], mean_ms=1000 * item["sum"] / item["count"])
        for quantile in (0.5, 0.95):
            lower = 0
            for upper, count in sorted(item["buckets"]):
                if count >= item["count"] * quantile:
                    row[f"p{int(quantile * 100)}_ms_interval"] = [
                        lower * 1000,
                        upper * 1000 if upper != float("inf") else None,
                    ]
                    break
                lower = upper
        rows.append(row)
    dropped = [
        {"labels": dict(labels), "count": value}
        for (name, labels), value in delta.items()
        if name == "sglang_zing_component_timing_dropped_total" and value > 0
    ]
    return {
        "calls": rows,
        "dropped": dropped,
        "scope": "one diagnostic request; host call/current CUDA stream; quantiles are histogram bucket bounds, not exact request quantiles",
    }


def memory_pair(root, case_id):
    variants = {}
    components = {}
    for variant in ("baseline", "candidate"):
        folder = root / variant / "diagnostics" / case_id
        receipt = read(folder / "allocator.json")
        assert receipt["case_id"] == case_id and receipt["requests"] == 1
        before = {v["pid"]: v for v in receipt["before"]}
        after = {v["pid"]: v for v in receipt["after"]}
        assert set(before) == set(after)
        driver = read(folder / "measured/driver-memory.json")["samples"]
        assert driver
        hardware = read(root / variant / "HARDWARE.json")["devices"]
        driver_values = {}
        for sample in driver:
            for line in sample["rows"]:
                index, uuid, used, utilization = [v.strip() for v in line.split(",")]
                driver_values.setdefault(uuid.lower().removeprefix("gpu-"), []).append(
                    float(used) * (1 << 20)
                )
        by_role = {}
        for pid, entry in before.items():
            end = after[pid]
            key = (entry["role"], entry["device"])
            assert key not in by_role and all(
                entry[k] == end[k] for k in ("role", "device", "visible_devices")
            )
            physical = int(entry["visible_devices"].split(",")[entry["device"]])
            uuid = hardware[physical]["uuid"].lower().removeprefix("gpu-")
            by_role[key] = {
                "gpu_uuid": uuid,
                "driver_first_sample": driver_values[uuid][0],
                "driver_peak_sampled": max(driver_values[uuid]),
                "resident_allocated": entry["counters"]["allocated"],
                "resident_reserved": entry["counters"]["reserved"],
                "peak_allocated": end["counters"]["peak_allocated"],
                "peak_reserved": end["counters"]["peak_reserved"],
            }
        variants[variant] = by_role
        components[variant] = {
            role: component_delta(
                (folder / f"before-{role}.prom").read_text(),
                (folder / f"after-{role}.prom").read_text(),
            )
            for role in ("encoder", "dit", "decoder")
        }
    assert variants["baseline"].keys() == variants["candidate"].keys()
    rows = [
        {"role": role, "rank": rank, **{v: variants[v][(role, rank)] for v in variants}}
        for role, rank in sorted(variants["baseline"])
    ]
    return {
        "requests_per_variant": 1,
        "unit": "bytes",
        "rows": rows,
        "scope": "whole-role process allocator; Encoder includes Qwen/reference/source VAE; allocator resident after per-case warmup, interval peaks bracket one complete request; driver first sample is taken within the request, not a resident measurement",
    }, components
