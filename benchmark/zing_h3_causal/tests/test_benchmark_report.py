# SPDX-License-Identifier: Apache-2.0
"""Report contracts: use new records, preserve failures, never invent metrics."""

import json

import pytest
from build_benchmark_report import build_report, load_run


def fixture_run(tmp_path, frames=120):
    directory = tmp_path / "client"
    directory.mkdir()
    config = {
        "cases": [
            {
                "label": "case",
                "case_id": "new-input",
                "frames": frames,
                "init": {"fps": 24},
            }
        ],
        "measured_runs": 3,
    }
    rows, chunks = [], []
    for name, phase, success, count in [
        ("warm", "warmup", True, frames),
        ("ok", "measured", True, frames),
        ("failed", "measured", False, 12),
    ]:
        rows.append(
            {
                "request_id": name,
                "label": "case",
                "case_id": "new-input",
                "phase": phase,
                "success": success,
                "output_frames": count,
                "chunk_count": 2,
                "metrics": {"fps": 999},
                "error": None if success else "Disconnected",
            }
        )
        for i in range(2):
            chunks.append(
                {
                    "request_id": name,
                    "server_request_id": f"{name}-{i}",
                    "arrival_seconds": [2, 10][i]
                    if phase == "measured"
                    else [50, 100][i],
                    "num_frames": count // 2,
                }
            )
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "requests.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (directory / "chunks.jsonl").write_text("\n".join(map(json.dumps, chunks)))
    return directory


def fixture_phase_run(tmp_path):
    directory = tmp_path / "phases"
    directory.mkdir()
    config = dict(
        layers=1, sp=1, device="L20D", chunk_plan=dict(num_frame_per_block=[3, 2])
    )
    samples, runs = [], []
    for index, kind in [(0, "warmup"), (1, "measured"), (2, "measured")]:
        runs.append(
            dict(
                mode="block",
                run=index,
                run_kind=kind,
                finite=True,
                output_sha256="a" * 64,
            )
        )
        for chunk, frames in [(0, 1), (1, 2), (2, 2), (3, 3), (4, 2), (5, 1)]:
            samples.append(
                dict(
                    mode="block",
                    kind="block",
                    chunk=chunk,
                    latent_frames=frames,
                    run=index,
                    run_kind=kind,
                    step=None,
                    q_rows=None,
                    prefix_rows=None,
                    history_rows=None,
                    kv_rows=None,
                    wall_ms=999 if kind == "warmup" else 10 + index,
                    cuda_ms=8,
                )
            )
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "runs.json").write_text(json.dumps(runs))
    (directory / "samples.jsonl").write_text("\n".join(map(json.dumps, samples)))
    return directory


def test_phase_report_recomputes_samples_preserves_scope_and_exports(tmp_path):
    directory = fixture_phase_run(tmp_path)
    # The display must work without summary.json, and ignore stale cached stats.
    (directory / "summary.json").write_text(
        json.dumps(dict(cycles=[dict(wall_ms=999)]))
    )
    run = load_run("L20D K1", directory)
    assert len(run["summary"]["cycles"]) == 1
    assert run["summary"]["cycles"][0]["wall_ms"]["median"] == 23
    assert all(g["wall_ms"]["count"] == 2 for g in run["summary"]["groups"])
    report = tmp_path / "phases.html"
    build_report([run], report)
    bundle = json.loads(report.with_suffix(".json").read_text())
    assert bundle["data"]["kind"] == "dit_phases"
    assert bundle["data"]["phase_runs"][0]["samples"] == run["samples"]
    assert "@@PHASE_SCRIPT@@" not in report.read_text()
    assert "PhaseReport.render(data)" in report.read_text()


def test_phase_report_rejects_changed_outputs_and_bad_timing(tmp_path):
    directory = fixture_phase_run(tmp_path)
    path = directory / "runs.json"
    runs = json.loads(path.read_text())
    runs[-1]["output_sha256"] = "b" * 64
    path.write_text(json.dumps(runs))
    with pytest.raises(ValueError, match="identical outputs"):
        load_run("bad", directory)
    runs[-1]["output_sha256"] = "a" * 64
    path.write_text(json.dumps(runs))
    path = directory / "samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["wall_ms"] = -1
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="nonnegative"):
        load_run("bad", directory)


def test_phase_and_client_reports_keep_separate_measurement_scopes(tmp_path):
    phase = load_run("phase", fixture_phase_run(tmp_path))
    client = load_run("client", fixture_run(tmp_path))
    with pytest.raises(ValueError, match="separately"):
        build_report([phase, client], tmp_path / "mixed.html")
    old = tmp_path / "old.html"
    build_report([client], old)
    combined = tmp_path / "combined.html"
    build_report(
        [phase],
        combined,
        include_results=[("Earlier client run", old.with_suffix(".json"))],
    )
    bundle = json.loads(combined.with_suffix(".json").read_text())
    assert len(bundle["results"]) == 2
    assert bundle["results"][0]["kind"] == "dit_phases"
    assert bundle["results"][1]["report_label"] == "Earlier client run"
    assert bundle["results"][1]["modes"]["client"]["chunks"] == client["chunks"][2:]
    embedded = json.loads(
        combined.read_text()
        .split('<script id="evidence" type="application/json">', 1)[1]
        .split("</script>", 1)[0]
    )
    assert embedded["datasets"] == bundle["results"]


@pytest.mark.parametrize("frames", [120, 720])
def test_new_records_recomputed_without_summary_and_warmup_excluded(tmp_path, frames):
    run = load_run("fresh", fixture_run(tmp_path, frames))
    summary = run["summary"]["case"]
    assert summary["succeeded"] == summary["failed"] == summary["not_run_measured"] == 1
    assert summary["metrics"]["fps"]["median"] == frames / 10
    assert summary["metrics"]["ttff_seconds"]["median"] == 2
    assert summary["icl_seconds"]["count"] == 1
    assert run["requests"][1]["rtf"] == pytest.approx(10 / (frames / 24))
    assert run["requests"][2]["output_frames"] == 12
    assert run["stages"] == []


def test_missing_count_fails_instead_of_showing_success(tmp_path):
    directory = fixture_run(tmp_path)
    path = directory / "chunks.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match="counts disagree"):
        load_run("broken", directory)


def test_damaged_health_is_visible_and_never_silently_accepted(tmp_path):
    directory = fixture_run(tmp_path)
    timing = tmp_path / "timing"
    timing.mkdir()
    (timing / "stage-timing-pid1-health.json").write_text("")
    run = load_run("old", directory)
    assert run["issues"] == [
        "Unreadable collector health: stage-timing-pid1-health.json"
    ]


def test_preflight_only_preserves_not_run_counts(tmp_path):
    directory = fixture_run(tmp_path)
    (directory / "requests.jsonl").unlink()
    (directory / "chunks.jsonl").unlink()
    run = load_run("preflight", directory)
    assert run["summary"]["case"]["not_run_measured"] == 3
    assert run["summary"]["case"]["metrics"]["fps"]["median"] is None


def test_embedded_prompt_cannot_close_data_script(tmp_path):
    run = load_run("fresh", fixture_run(tmp_path))
    attack = "</script><script>window.pwned=true</script>&\u2028@@PHASE_SCRIPT@@"
    run["config"]["prompt"] = attack
    path = tmp_path / "report.html"
    build_report([run], path)
    html = path.read_text()
    assert attack not in html
    payload = html.split('<script id="evidence" type="application/json">', 1)[1].split(
        "</script>", 1
    )[0]
    assert json.loads(payload)["runs"][0]["config"]["prompt"] == attack
    assert "@@DATA@@" not in html


def test_portable_json_is_exported_alongside_html(tmp_path):
    run = load_run("new", fixture_run(tmp_path))
    report = tmp_path / "report.html"
    build_report([run], report)
    bundle = json.loads(report.with_suffix(".json").read_text())
    assert bundle["format"] == "zing-benchmark-results"
    assert bundle["version"] == 1
    mode = bundle["data"]["modes"]["new"]
    assert mode["summary"]["succeeded"] == 1
    assert mode["summary"]["failed"] == 1
    assert len(mode["requests"]) == 2
    assert len(bundle["data"]["runs"][0]["requests"]) == 3  # Includes warmup.
    separate = tmp_path / "shared" / "results.json"
    build_report([run], report, separate)
    assert json.loads(separate.read_text()) == bundle


def test_three_cases_export_as_independent_results(tmp_path):
    import copy

    run = load_run("current", fixture_run(tmp_path))
    cases = [("5s", 120, 6), ("30s-wasteland", 720, 7), ("30s-town", 720, 7)]
    runs = []
    for label, frames, references in cases:
        current = copy.deepcopy(run)
        case = current["config"]["cases"][0]
        case.update(
            label=label, case_id=label, frames=frames, reference_count=references
        )
        for request in current["requests"]:
            request.update(label=label, case_id=label)
            if request["success"]:
                request["output_frames"] = frames
        for chunk in current["chunks"]:
            if chunk["request_id"] in ["warm", "ok"]:
                chunk["num_frames"] = frames // 2
        current["summary"] = {label: current["summary"].pop("case")}
        current["summary"][label]["case_id"] = label
        runs.append(current)
    path = tmp_path / "all.html"
    build_report(runs, path)
    data = json.loads(path.with_suffix(".json").read_text())
    assert len(data["results"]) == 3
    assert [d["frames"] for d in data["results"]] == [120, 720, 720]
    assert [
        d["runs"][0]["config"]["cases"][0]["reference_count"] for d in data["results"]
    ] == [6, 7, 7]
    assert all(len(d["modes"]) == 1 for d in data["results"])


def test_received_profiles_build_portable_report_without_server_files(tmp_path):
    directory = fixture_run(tmp_path)
    profiles = directory / "profiles"
    profiles.mkdir()
    stages = [
        {
            "request_id": "session",
            "dispatch_id": "ok-0",
            "rank": rank,
            "stage": "DiT",
            "status": "ok",
            "cuda_status": "complete",
            "cuda_stream_ms": 12.5,
            "host_call_ms": 1.0,
        }
        for rank in (0, 1)
    ]
    (profiles / "ok.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "stages": stages,
                "health": [
                    {"rank": rank, "written": 1, "dropped": 0, "pending": 0}
                    for rank in (0, 1)
                ],
            }
        )
    )
    run = load_run("cuda", directory)
    assert run["stages"] == stages
    assert len(run["health"]) == 2
    assert not run["issues"]
    output = tmp_path / "report.html"
    build_report([run], output)
    exported = json.loads(output.with_suffix(".json").read_text())
    assert exported["data"]["modes"]["cuda"]["spans"] == stages
    profile = json.loads((profiles / "ok.json").read_text())
    profile["status"] = "incomplete"
    (profiles / "ok.json").write_text(json.dumps(profile))
    assert any("incomplete" in issue for issue in load_run("cuda", directory)["issues"])


def test_streaming_report_uses_shared_page_and_portable_envelope(tmp_path):
    from build_streaming_report import build_report as build_streaming

    measurement = {
        "config": {
            "transport": "continuous_source",
            "prompt": "literal @@CASE@@ and @@DATA@@",
            "cases": [
                {"label": "new-scene", "case_id": "scene-identity", "frames": 120}
            ],
        },
        "requests": [],
        "summary": {},
    }
    page, exported = tmp_path / "report.html", tmp_path / "report.json"
    build_streaming([measurement], page, json_output=exported)
    bundle = json.loads(exported.read_text())
    assert bundle["format"] == "zing-benchmark-results"
    assert bundle["version"] == 1
    assert bundle["data"]["kind"] == "continuous_source"
    assert bundle["data"]["streaming"] == measurement
    assert 'id="result-file"' in page.read_text()
    assert 'id="finite-view"' in page.read_text()
    assert 'id="streaming-view"' in page.read_text()

    embedded = (
        page.read_text()
        .split('<script id="evidence" type="application/json">', 1)[1]
        .split("</script>", 1)[0]
    )
    assert json.loads(embedded)["streaming"] == measurement


def test_received_continuous_host_profile_needs_no_server_files_or_cuda(tmp_path):
    from build_streaming_report import load_received_profiles

    base = dict(
        request_id="s", host_call_ms=1, status="ok", clock_domain="CLOCK_MONOTONIC:test"
    )
    spans = [
        dict(
            base,
            rank=0,
            stage="Encoder",
            component="source_vae",
            worker_role="dedicated_source_encoder",
            latent_range=[0, 1],
        )
    ]
    for rank in (0, 1):
        spans.extend(
            [
                dict(base, rank=rank, stage="DiT", output_latent_range=[0, 1]),
                dict(base, rank=rank, stage="Decoder", output_frame_range=[0, 1]),
                dict(
                    base,
                    rank=rank,
                    stage="Request",
                    component="continuous_source_complete",
                ),
            ]
        )
    cell = tmp_path / "case/measured-0"
    cell.mkdir(parents=True)
    (cell / "profile.json").write_text(
        json.dumps(
            dict(
                session_id="s",
                status="complete",
                expected_ranks=[0, 1],
                stages=spans,
                health=[],
            )
        )
    )
    (cell / "summary.json").write_text(
        json.dumps(dict(session_id="s", client_clock_domain="CLOCK_MONOTONIC:test"))
    )
    bundle = dict(
        config=dict(cases=[dict(label="case", frames=1)]),
        requests=[
            dict(
                request_id="client",
                label="case",
                phase="measured",
                index=0,
                success=True,
            )
        ],
    )
    server = load_received_profiles(tmp_path, bundle)
    assert server["validation"]["sessions"]["s"]["complete"]
    assert server["spans"] == spans
    assert server["clients"]["s"]["client_clock_domain"] == "CLOCK_MONOTONIC:test"
    assert not (tmp_path / "timing").exists()


@pytest.mark.parametrize("rank_count", [1, 2])
def test_received_profile_supports_local_encoder_and_still_requires_remote_encoder(
    tmp_path, rank_count
):
    from build_streaming_report import load_received_profiles

    base = dict(
        request_id="s", host_call_ms=1, status="ok", clock_domain="CLOCK_MONOTONIC:test"
    )
    spans = []
    for rank in range(rank_count):
        spans.extend(
            [
                dict(base, rank=rank, stage="DiT", output_latent_range=[0, 1]),
                dict(base, rank=rank, stage="Decoder", output_frame_range=[0, 1]),
                dict(
                    base,
                    rank=rank,
                    stage="Request",
                    component="continuous_source_complete",
                    source_encoder_role="local",
                ),
            ]
        )
    cell = tmp_path / "case/measured-0"
    cell.mkdir(parents=True)
    profile = dict(
        session_id="s",
        status="complete",
        expected_ranks=list(range(rank_count)),
        stages=spans,
        health=[],
    )
    (cell / "summary.json").write_text(json.dumps(dict(session_id="s")))
    bundle = dict(
        config=dict(cases=[dict(label="case", frames=1)]),
        requests=[
            dict(
                request_id="client",
                label="case",
                phase="measured",
                index=0,
                success=True,
            )
        ],
    )
    (cell / "profile.json").write_text(json.dumps(profile))
    assert load_received_profiles(tmp_path, bundle)["validation"]["sessions"]["s"][
        "complete"
    ]
    for row in spans:
        if row["stage"] == "Request":
            row["source_encoder_role"] = "dedicated"
    (cell / "profile.json").write_text(json.dumps(profile))
    assert not load_received_profiles(tmp_path, bundle)["validation"]["sessions"]["s"][
        "complete"
    ]


def test_phase_comparison_uses_exact_assets_and_keeps_runtime_versions(tmp_path):
    directory = fixture_phase_run(tmp_path)
    path = directory / "config.json"
    config = json.loads(path.read_text())
    assert load_run("historical", directory)["comparison_key"] is None
    config.update(
        source_archive_sha256="source",
        selected_weights={"layer": {"sha256": "weights"}},
        conditioning_sha256="input",
        kernel_route_envs={"FUSED_HISTORY_ROPE": "1"},
        torch="2.11",
    )
    path.write_text(json.dumps(config))
    first = load_run("first", directory)
    config["torch"] = "2.13"
    path.write_text(json.dumps(config))
    second = load_run("second", directory)
    assert first["comparison_key"] == second["comparison_key"]
    assert second["config"]["torch"] == "2.13"
    config["selected_weights"]["layer"]["sha256"] = "other-checkpoint"
    path.write_text(json.dumps(config))
    assert load_run("different", directory)["comparison_key"] != first["comparison_key"]


def test_cli_rebuilds_saved_reports_without_raw_run(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    directory = fixture_run(tmp_path)
    original = tmp_path / "original.html"
    build_report([load_run("original", directory)], original)
    output = tmp_path / "rebuilt.html"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "build_benchmark_report.py"),
            "--include-results",
            f"saved={original.with_suffix('.json')}",
            "--output",
            str(output),
        ],
        check=True,
    )
    before = json.loads(original.with_suffix(".json").read_text())["data"]
    after = json.loads(output.with_suffix(".json").read_text())["data"]
    assert after == dict(before, report_label="saved")
    assert "FullChainReport" in output.read_text()


def memory_fixture(tmp_path):
    from analyze_memory import analyze

    trace = [
        dict(
            action="alloc",
            addr=1,
            size=40,
            time_us=5,
            frames=[dict(filename="model.py", line=7, name="forward")],
        )
    ]
    snapshot = dict(
        device_traces=[trace],
        segments=[
            dict(
                device=0,
                blocks=[
                    dict(
                        address=1,
                        size=40,
                        requested_size=40,
                        state="active_allocated",
                        frames=[],
                    )
                ],
            )
        ],
    )
    row = analyze(
        snapshot,
        dict(
            device=0,
            start_us=0,
            history_may_be_truncated=False,
            case="graph / warmup",
            pid=9,
            role="server",
        ),
    )
    directory = tmp_path / "memory-run"
    directory.mkdir()
    (directory / "MEMORY_REPLAY.json").write_text(
        json.dumps(dict(schema_version=1, profiles=[row], errors=[]))
    )
    return directory


def test_memory_reports_share_saved_result_import_export(tmp_path):
    from build_benchmark_report import save_report
    from memory_results import load_memory_run

    directory = memory_fixture(tmp_path)
    result = load_memory_run("new-device", directory)
    output = tmp_path / "all.html"
    save_report([], output, None, [], memory_runs=[("new-device", directory)])
    data = json.loads(output.with_suffix(".json").read_text())["data"]
    assert data == result and data["kind"] == "runtime_memory"
    assert (
        "MemoryReport" in output.read_text()
        and "@@MEMORY_SCRIPT@@" not in output.read_text()
    )
    # Importing the saved JSON does not need the original pickle or directory.
    save_report(
        [], tmp_path / "imported.html", None, [("saved", output.with_suffix(".json"))]
    )
    assert (
        json.loads((tmp_path / "imported.json").read_text())["data"]["replay"]
        == result["replay"]
    )


def test_memory_loader_rejects_unclosed_or_nonconserving_replay(tmp_path):
    from memory_results import load_memory_run

    directory = memory_fixture(tmp_path)
    path = directory / "MEMORY_REPLAY.json"
    value = json.loads(path.read_text())
    value["profiles"][0]["closure"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="unclosed"):
        load_memory_run("bad", directory)
    value["profiles"][0]["closure"] = True
    value["profiles"][0]["peak_stacks"][0]["bytes"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="conserve"):
        load_memory_run("bad", directory)


def test_memory_json_validation_in_browser(tmp_path):
    import subprocess
    from pathlib import Path

    directory = memory_fixture(tmp_path)
    subprocess.run(
        [
            "node",
            str(Path(__file__).with_name("check_memory_report.cjs")),
            str(directory / "MEMORY_REPLAY.json"),
        ],
        check=True,
    )


def test_memory_timing_caches_are_recomputed_from_requests():
    from memory_results import summarize_timing

    row = dict(
        count=2,
        partial=False,
        client={"stale": 999},
        stages={"stale": 999},
        requests=[
            dict(
                client=dict(ttff_seconds=t, e2e_seconds=t + 4, post_ttff_fps=8),
                stage_cuda_ms={"DiT": t * 1000},
            )
            for t in [1, 3]
        ],
    )
    summarize_timing([row])
    assert row["client"]["ttff_seconds"] == dict(median=2, min=1, max=3)
    assert row["stages"]["DiT"]["median"] == 2000
    row["requests"][1]["stage_cuda_ms"] = {"Encoder": 1000}
    with pytest.raises(ValueError, match="coverage"):
        summarize_timing([row])


def test_checkpoint_summary_excludes_load_peak_and_does_not_sum_shared_gpu(tmp_path):
    from memory_results import summarize_counters

    memory = tmp_path / "memory"
    memory.mkdir()
    for pid, role in [(1, "qwen"), (2, "encoder")]:
        (memory / f"process-{pid}.json").write_text(
            json.dumps(dict(pid=pid, role=role, device=0, visible_devices="GPU-shared"))
        )
        for seq, label, peak in [(1, "ready", 999), (2, "after", 200)]:
            (memory / f"ack-{seq}-{pid}.json").write_text(
                json.dumps(
                    dict(
                        label=label,
                        counters=dict(
                            allocated=100,
                            reserved=110,
                            peak_allocated=peak,
                            peak_reserved=peak + 10,
                        ),
                    )
                )
            )
    (tmp_path / "gpu-memory.jsonl").write_text(
        json.dumps(dict(monotonic=1, rows=["0, GPU-shared, GPU, 300, 0"])) + "\n"
    )
    value = summarize_counters(tmp_path, "configuration")["configuration"]
    assert [v["peak_allocated"] for v in value["processes"]] == [200, 200]
    assert len(value["gpus"]) == 1 and value["gpus"][0]["peak_sample"] == 300 * 2**20
    assert value["gpus"][0]["ready_sample"] is None
