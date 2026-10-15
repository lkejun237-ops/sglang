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
    attack = "</script><script>window.pwned=true</script>&\u2028"
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
