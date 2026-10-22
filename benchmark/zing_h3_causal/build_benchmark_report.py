# SPDX-License-Identifier: Apache-2.0
"""Build an offline HTML report from client or DiT phase records (no GPU required)."""

import argparse
import copy
import hashlib
import html
import json
import os
import statistics
from pathlib import Path

from bench_client import calculate_metrics, summarize
from dit_phase_timing import summarize as summarize_phases
from memory_results import load_memory_run


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(name, directory, timing=None):
    config = json.loads((directory / "config.json").read_text())
    if "chunk_plan" in config:
        return load_phase_run(name, directory, config)
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


def load_phase_run(name, directory, config):
    """Recompute phase statistics from raw calls, never a cached summary."""
    samples = read_jsonl(directory / "samples.jsonl")
    runs = json.loads((directory / "runs.json").read_text())
    if not samples or not runs or not any(r["run_kind"] == "measured" for r in samples):
        raise ValueError("Phase report requires raw samples and measured runs")
    if (
        not all(r["finite"] is True for r in runs)
        or len({r["output_sha256"] for r in runs}) != 1
    ):
        raise ValueError("Phase runs must have finite, identical outputs")
    summary = summarize_phases(samples)
    options = config.get("options", {})
    comparison = dict(
        source=config.get("source_archive_sha256"),
        conditioning=config.get("conditioning_sha256"),
        configs=[
            config.get(k)
            for k in (
                "model_config_sha256",
                "runtime_config_sha256",
                "vae_config_sha256",
            )
        ],
        weights=config.get("selected_weights"),
        layers=config.get("layer_ids"),
        sp=config.get("sp"),
        backend=config.get("effective_backend"),
        graph=config.get("graph"),
        routes=config.get("kernel_route_envs"),
        sampling={
            k: options.get(k)
            for k in ("seed", "window_chunks", "sink_chunks", "denoising_step_list")
        },
        geometry=[
            {
                k: group[k]
                for k in (
                    "mode",
                    "kind",
                    "chunk",
                    "latent_frames",
                    "step",
                    "q_rows",
                    "prefix_rows",
                    "history_rows",
                    "kv_rows",
                )
            }
            for group in summary["groups"]
        ],
    )
    profile_path = directory / "profile.json"
    return dict(
        kind="dit_phases",
        name=name,
        config=config,
        runs=runs,
        samples=samples,
        summary=summary,
        comparison_key=hashlib.sha256(
            json.dumps(comparison, sort_keys=True).encode()
        ).hexdigest()
        if comparison["source"]
        else None,
        profile=json.loads(profile_path.read_text()) if profile_path.exists() else None,
        sources={
            filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            for filename in (
                "config.json",
                "runs.json",
                "samples.jsonl",
                *(["profile.json"] if profile_path.exists() else []),
            )
        },
    )


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
    chosen = next((m for m in modes.values() if m["spans"]), None)
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
                    s.get("host_call_ms") is not None for s in selected
                ):
                    values.append(sum(s["host_call_ms"] for s in selected) / 1000)
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


def write_report_page(data, output, *, template=None):
    """Shared offline page shell for finite and continuous-source evidence."""
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
    for char, escape in [
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ]:
        payload = payload.replace(char, escape)
    template = template or Path(__file__).with_name("benchmark_report.html").read_text()
    script = Path(__file__).with_name("streaming_report.js").read_text()
    # Finite builders fill these fields; streaming metadata is rendered from data.
    for key in (
        "CASE",
        "IDENTITY",
        "INPUT",
        "RUN_COUNTS",
        "MODES",
        "PLACEMENT",
        "HARDWARE",
        "ENCODER_COUNT",
        "DIT_COUNT",
        "DECODER_COUNT",
        "OUTCOME",
        "VERSION",
        "ISSUES",
        "SOURCE_LINKS",
        "MODE_OPTIONS",
    ):
        template = template.replace(
            "@@" + key + "@@", "Benchmark" if key == "CASE" else ""
        )
    template = template.replace(
        "@@PHASE_SCRIPT@@", Path(__file__).with_name("phase_report.js").read_text()
    )
    template = template.replace(
        "@@MEMORY_SCRIPT@@", Path(__file__).with_name("memory_report.js").read_text()
    )
    template = template.replace("@@STREAM_SCRIPT@@", script).replace(
        "@@DATA@@", payload
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template, encoding="utf-8")


def save_report(datasets, output, json_output, included, template=None, memory_runs=()):
    """Keep every saved result available in one offline page and JSON bundle."""
    for label, path in included:
        bundle = json.loads(path.read_text())
        if (
            bundle.get("format") != "zing-benchmark-results"
            or bundle.get("version") != 1
        ):
            raise ValueError("Expected a version 1 benchmark results JSON")
        results = bundle.get("results", [bundle.get("data")])
        if (
            not isinstance(results, list)
            or not results
            or not all(isinstance(r, dict) for r in results)
        ):
            raise ValueError("Saved report has no result datasets")
        datasets.extend(dict(result, report_label=label) for result in results)
    datasets.extend(load_memory_run(name, path) for name, path in memory_runs)
    payload = {"datasets": datasets} if len(datasets) > 1 else datasets[0]
    write_report_page(payload, output, template=template)
    path = json_output or output.with_suffix(".json")
    export_results(datasets[0], path)
    if len(datasets) > 1:
        bundle = json.loads(path.read_text())
        bundle["results"] = datasets
        path.write_text(
            json.dumps(bundle, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def build_report(runs, output, json_output=None, include_results=(), memory_runs=()):
    if any(run.get("kind") == "dit_phases" for run in runs):
        if not all(run.get("kind") == "dit_phases" for run in runs):
            raise ValueError(
                "Build phase and client reports separately, then open both JSON files in the page"
            )
        data = dict(kind="dit_phases", phase_runs=runs)
        save_report(
            [data], output, json_output, include_results, memory_runs=memory_runs
        )
        return
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
    save_report(
        datasets,
        output,
        json_output,
        include_results,
        template=template,
        memory_runs=memory_runs,
    )


def named_path(value):
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Expected NAME=DIRECTORY")
    return name, Path(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=named_path, action="append", default=[])
    parser.add_argument(
        "--memory-run",
        type=named_path,
        action="append",
        default=[],
        metavar="NAME=DIRECTORY",
        help="Import an already-replayed memory profile; no pickle loading",
    )
    parser.add_argument("--timing", type=named_path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--include-results",
        type=named_path,
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="Embed an existing saved report alongside the new results",
    )
    args = parser.parse_args()
    if not args.run and not args.include_results and not args.memory_run:
        parser.error("Provide --run, --memory-run or --include-results")
    names = [name for name, _ in args.run]
    if len(set(names)) != len(names):
        parser.error("Run names must be unique")
    timings = dict(args.timing)
    if any(name not in names for name in timings):
        parser.error("Timing names must match run names")
    if args.run:
        build_report(
            [load_run(name, path, timings.get(name)) for name, path in args.run],
            args.output,
            args.json_output,
            args.include_results,
            args.memory_run,
        )
    else:
        save_report(
            [],
            args.output,
            args.json_output,
            args.include_results,
            memory_runs=args.memory_run,
        )
    print(args.output.resolve())
