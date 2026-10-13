# SPDX-License-Identifier: Apache-2.0
"""Build an offline HTML report from new bench_client records (no GPU required)."""

import argparse
import copy
import hashlib
import html
import json
import os
import statistics
from pathlib import Path

from bench_client import calculate_metrics, summarize


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(name, directory, timing=None):
    config = json.loads((directory / "config.json").read_text())
    requests = read_jsonl(directory / "requests.jsonl")
    chunks = read_jsonl(directory / "chunks.jsonl")
    identities = [r["request_id"] for r in requests]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate client request identity")
    if any(c["request_id"] not in identities for c in chunks):
        raise ValueError("Chunk has no matching client request")
    cases = {c["label"]: c for c in config["cases"]}
    for row in requests:
        case = cases[row["label"]]
        if row["case_id"] != case["case_id"]:
            raise ValueError("Client request does not match configured case")
        batches = [c for c in chunks if c["request_id"] == row["request_id"]]
        frames = sum(c["num_frames"] for c in batches)
        if frames != row["output_frames"] or len(batches) != row["chunk_count"]:
            raise ValueError("Client request and chunk counts disagree")
        if row["success"] and frames != case["frames"]:
            raise ValueError("Successful request has incomplete frame coverage")
        row["metrics"] = calculate_metrics(batches)
        row["icl_values_seconds"] = [
            b["arrival_seconds"] - a["arrival_seconds"]
            for a, b in zip(batches, batches[1:])
        ]
        row["rtf"] = (
            row["metrics"]["e2e_seconds"] / (frames / case["init"]["fps"])
            if frames and row["metrics"]["e2e_seconds"] is not None
            else None
        )
    timing = timing or directory.parent / "timing"
    stages, health, issues = [], [], []
    profiles = directory / "profiles"
    if profiles.is_dir():
        for request in requests:
            path = profiles / (request["request_id"] + ".json")
            if not path.is_file():
                if config.get("receive_stage_timing"):
                    issues.append(
                        f"No profile result received: {request['request_id']}"
                    )
                continue
            profile = json.loads(path.read_text())
            stages.extend(profile["stages"])
            health.extend({"file": path.name, **value} for value in profile["health"])
            if profile["status"] != "complete":
                issues.append(
                    f"Server profile {profile['status']}: {request['request_id']}"
                )
    elif config.get("receive_stage_timing"):
        issues.append("No server profile results received")
    elif timing.is_dir():
        for path in sorted(timing.glob("stage-timing-pid*.jsonl")):
            stages.extend(read_jsonl(path))
        for path in sorted(timing.glob("*-health.json")):
            try:
                value = json.loads(path.read_text())
                health.append({"file": path.name, **value})
                if any(value.get(k) for k in ("dropped", "pending", "queued", "error")):
                    issues.append(f"Collector incomplete: {path.name}")
            except (ValueError, OSError):
                issues.append(f"Unreadable collector health: {path.name}")
    dispatches = {c["server_request_id"] for c in chunks}
    if any(s.get("dispatch_id") not in dispatches for s in stages):
        issues.append("Stage records contain unmatched dispatch identities")
    if stages and not health:
        issues.append("No readable collector health snapshots")
    if any(s.get("status") != "ok" for s in stages):
        issues.append("Stage records contain failed spans")
    if any(s.get("cuda_status") in {"error", "timeout", "pending"} for s in stages):
        issues.append("CUDA samples are incomplete")
    sources = {}
    for path in [
        directory / "config.json",
        directory / "requests.jsonl",
        directory / "chunks.jsonl",
        *sorted(profiles.glob("*.json")),
    ]:
        if path.exists():
            sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "name": name,
        "directory": str(directory.resolve()),
        "config": config,
        "requests": requests,
        "chunks": chunks,
        "summary": summarize(requests, config["cases"], config["measured_runs"]),
        "stages": stages,
        "health": health,
        "issues": issues,
        "sources": sources,
    }


def evidence(runs, output):
    """Adapt raw runs to the existing 5s results page, without changing layout."""
    case_ids = {c["case_id"] for run in runs for c in run["config"]["cases"]}
    if len(case_ids) != 1:
        raise ValueError(
            "Reference page compares one case; generate separate pages for different cases"
        )
    modes, components = {}, {}
    for run in runs:
        case = run["config"]["cases"][0]
        measured = [r for r in run["requests"] if r["phase"] == "measured"]
        ids = {r["request_id"] for r in measured}
        chunks = [c for c in run["chunks"] if c["request_id"] in ids]
        dispatches = {c["server_request_id"] for c in chunks}
        spans = [s for s in run["stages"] if s.get("dispatch_id") in dispatches]
        summaries = []
        for request in measured:
            if not request["success"]:
                continue
            req_ids = {
                c["server_request_id"]
                for c in chunks
                if c["request_id"] == request["request_id"]
            }
            group = [s for s in spans if s.get("dispatch_id") in req_ids]
            for rank in sorted({s["rank"] for s in group}):
                rows = [s for s in group if s["rank"] == rank]
                stages = {}
                for stage in ("Conditioning", "Encoder", "DiT", "Decoder"):
                    selected = [s for s in rows if s["stage"] == stage]
                    complete = selected and all(
                        s.get("cuda_status") == "complete"
                        and s.get("cuda_stream_ms") is not None
                        for s in selected
                    )
                    stages[stage] = {
                        "count": len(selected),
                        "host_call_ms_sum": sum(s["host_call_ms"] for s in selected),
                        "cuda_stream_ms_sum": sum(s["cuda_stream_ms"] for s in selected)
                        if complete
                        else None,
                    }
                summaries.append(
                    {
                        "client_request_id": request["request_id"],
                        "model_request_id": rows[0]["request_id"],
                        "rank": rank,
                        "stages": stages,
                    }
                )
        modes[run["name"]] = {
            "summary": run["summary"][case["label"]],
            "requests": measured,
            "chunks": chunks,
            "stageRequests": summaries,
            "spans": spans,
            "source": os.path.relpath(
                Path(run["directory"]) / "chunks.jsonl", output.parent
            ),
        }
    chosen = next(
        (
            m
            for m in modes.values()
            if any(s.get("cuda_status") == "complete" for s in m["spans"])
        ),
        None,
    )
    for component in ("qwen", "reference_vae", "source_vae"):
        values = []
        if chosen:
            for request in chosen["requests"]:
                if not request["success"]:
                    continue
                dispatches = {
                    c["server_request_id"]
                    for c in chosen["chunks"]
                    if c["request_id"] == request["request_id"]
                }
                selected = [
                    s
                    for s in chosen["spans"]
                    if s.get("dispatch_id") in dispatches
                    and s["rank"] == 0
                    and s.get("component") == component
                ]
                if selected and all(
                    s.get("cuda_status") == "complete"
                    and s.get("cuda_stream_ms") is not None
                    for s in selected
                ):
                    values.append(sum(s["cuda_stream_ms"] for s in selected) / 1000)
        components[component] = statistics.median(values) if values else None
    return {
        "modes": modes,
        "components": components,
        "frames": runs[0]["config"]["cases"][0]["frames"],
        "runs": runs,
    }


def export_results(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"format": "zing-benchmark-results", "version": 1, "data": data},
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_report(runs, output, json_output=None):
    case_ids = list(
        dict.fromkeys(c["case_id"] for run in runs for c in run["config"]["cases"])
    )
    datasets = []
    for case_id in case_ids:
        selected_runs = []
        for run in runs:
            cases = [c for c in run["config"]["cases"] if c["case_id"] == case_id]
            if not cases:
                continue
            selected = copy.deepcopy(run)
            selected["config"]["cases"] = cases
            selected["requests"] = [
                r for r in selected["requests"] if r["case_id"] == case_id
            ]
            ids = {r["request_id"] for r in selected["requests"]}
            selected["chunks"] = [
                c for c in selected["chunks"] if c["request_id"] in ids
            ]
            dispatches = {c["server_request_id"] for c in selected["chunks"]}
            selected["stages"] = [
                s for s in selected["stages"] if s.get("dispatch_id") in dispatches
            ]
            selected_runs.append(selected)
        datasets.append(evidence(selected_runs, output))
    data = datasets[0]
    payload = json.dumps(
        {"datasets": datasets} if len(datasets) > 1 else data,
        ensure_ascii=False,
        allow_nan=False,
    )
    for char, escape in [
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ]:
        payload = payload.replace(char, escape)
    template = Path(__file__).with_name("benchmark_report.html").read_text()
    runs = data["runs"]
    case = runs[0]["config"]["cases"][0]
    config = runs[0]["config"]
    deployment = config.get("deployment", {})
    stage_rows = [
        row for mode in data["modes"].values() for row in mode["stageRequests"]
    ]

    def counts(stage):
        values = sorted({r["stages"][stage]["count"] for r in stage_rows})
        return " / ".join(map(str, values)) if values else "未测"

    mode_options = "".join(
        '<option value="'
        + html.escape(name, quote=True)
        + ('" selected>' if name == "cuda" else '">')
        + html.escape(name)
        + "</option>"
        for name in data["modes"]
    )
    successful = sum(m["summary"]["succeeded"] for m in data["modes"].values())
    failed = sum(m["summary"]["failed"] for m in data["modes"].values())
    replacements = {
        "CASE": case["label"],
        "IDENTITY": "ZING · " + case["case_id"] + " · 本次原始测量记录",
        "INPUT": f"固定 source 视频 + {case.get('reference_count', '未记录')} 项参考条件，输出 {case['frames']} 帧 / {case['init']['fps']} FPS / {case.get('width', '未记录')} × {case.get('height', '未记录')}。模型常驻、并发 1、请求全部串行。",
        "RUN_COUNTS": " · ".join(
            f"{r['name']}：{r['config'].get('warmup_runs', 0)} 次预热 + {r['config']['measured_runs']} 次正式"
            for r in runs
        ),
        "MODES": " → ".join(data["modes"]),
        "PLACEMENT": str(deployment.get("layout", "部署信息见版本记录")),
        "HARDWARE": "硬件按运行者配置记录",
        "ENCODER_COUNT": counts("Encoder"),
        "DIT_COUNT": counts("DiT"),
        "DECODER_COUNT": counts("Decoder"),
        "OUTCOME": f"本次正式请求成功 {successful} 次、失败 {failed} 次。客户端指标按原始到达记录复算；阶段数据按货号关联展示。页面展示不等于严格 ledger、完整媒体解码或人工画质已验收。",
        "VERSION": json.dumps(
            [
                {
                    "name": r["name"],
                    "client_sha": r["config"].get("client_sha"),
                    "deployment": r["config"].get("deployment"),
                    "sampling": r["config"].get("sampling"),
                    "source_sha256": r["sources"],
                    "warmup_count": sum(q["phase"] == "warmup" for q in r["requests"]),
                }
                for r in runs
            ],
            ensure_ascii=False,
            indent=2,
        ),
        "ISSUES": "；".join(issue for r in runs for issue in r["issues"])
        or "未发现 health 读入或已记录的丢样异常；采集完整性仍需严格验收。",
    }
    for key, value in replacements.items():
        template = template.replace("@@" + key + "@@", html.escape(value))
    links = " · ".join(
        '<a href="'
        + html.escape(
            os.path.relpath(Path(r["directory"]) / "config.json", output.parent),
            quote=True,
        )
        + '">'
        + html.escape(r["name"])
        + " 原始配置</a>"
        for r in runs
    )
    template = template.replace("@@SOURCE_LINKS@@", links).replace(
        "@@MODE_OPTIONS@@", mode_options
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("@@DATA@@", payload), encoding="utf-8")
    json_path = json_output or output.with_suffix(".json")
    export_results(data, json_path)
    if len(datasets) > 1:
        bundle = json.loads(json_path.read_text())
        bundle["results"] = datasets
        json_path.write_text(
            json.dumps(bundle, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def named_path(value):
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Expected NAME=DIRECTORY")
    return name, Path(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=named_path, action="append", required=True)
    parser.add_argument("--timing", type=named_path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    names = [name for name, _ in args.run]
    if len(set(names)) != len(names):
        parser.error("Run names must be unique")
    timings = dict(args.timing)
    if any(name not in names for name in timings):
        parser.error("Timing names must match run names")
    build_report(
        [load_run(name, path, timings.get(name)) for name, path in args.run],
        args.output,
        args.json_output,
    )
    print(args.output.resolve())
