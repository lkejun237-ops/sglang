# SPDX-License-Identifier: Apache-2.0
"""An independent ledger must reject missing or misattributed stage evidence."""

import json

import pytest
from analyze_stage_timing import analyze


@pytest.fixture
def ledger(tmp_path):
    client, timing = tmp_path / "client", tmp_path / "timing"
    client.mkdir()
    timing.mkdir()
    (client / "requests.jsonl").write_text(
        json.dumps({"request_id": "client", "phase": "measured", "success": True})
        + "\n"
    )
    (client / "chunks.jsonl").write_text(
        json.dumps(
            {
                "request_id": "client",
                "server_request_id": "dispatch",
                "chunk_index": 0,
                "num_frames": 120,
            }
        )
        + "\n"
    )
    rows = []
    for rank in (0, 1):
        base = dict(
            request_id="session",
            dispatch_id="dispatch",
            rank=rank,
            clock_domain="same",
            status="ok",
            mode="host",
            host_enter_ns=1,
            host_return_ns=2,
            host_handoff_ns=3,
            host_call_ms=0.000001,
            cuda_stream_ms=None,
        )
        for index, component in enumerate(
            ["qwen", *["reference_vae"] * 6, "source_vae"]
        ):
            rows.append(
                {
                    **base,
                    "span_id": f"{rank}:encoder:{index}",
                    "stage": "Encoder",
                    "component": component,
                }
            )
        rows.append(
            {
                **base,
                "span_id": f"{rank}:dit",
                "stage": "DiT",
                "block_index": 0,
                "latent_range": [0, 35],
            }
        )
        rows.append(
            {
                **base,
                "span_id": f"{rank}:decoder",
                "stage": "Decoder",
                "output_frame_range": [0, 120],
                "output_blocks": [{"dispatch_id": "dispatch"}],
            }
        )
        rows.append(
            {**base, "span_id": f"{rank}:conditioning", "stage": "Conditioning"}
        )
        (timing / f"stage-timing-pid{rank}-health.json").write_text(
            json.dumps(dict(written=11, dropped=0, pending=0, queued=0, error=None))
        )
    path = timing / "stage-timing-pid0.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path, path, rows


def test_complete_ledger_retains_rank_local_costs(ledger):
    directory, _, _ = ledger
    report = analyze(directory)
    assert report["valid"]
    assert report["span_count"] == 22
    assert {r["rank"] for r in report["requests"]} == {0, 1}
    assert report["requests"][0]["stages"]["Encoder"]["count"] == 8
    assert report["requests"][0]["stages"]["DiT"]["cuda_stream_ms_sum"] is None


@pytest.mark.parametrize(
    "fault", ["missing_rank", "wrong_delivery", "cuda_missing", "dropped"]
)
def test_incomplete_or_misattributed_evidence_cannot_pass(ledger, fault):
    directory, path, rows = ledger
    if fault == "missing_rank":
        rows = [row for row in rows if row["rank"] == 0]
    elif fault == "wrong_delivery":
        next(row for row in rows if row["stage"] == "Decoder")["output_blocks"] = [
            {"dispatch_id": "other_request"}
        ]
    elif fault == "cuda_missing":
        rows[0].update(mode="cuda", cuda_status="timeout")
    else:
        health = directory / "timing/stage-timing-pid0-health.json"
        health.write_text(
            json.dumps(dict(written=11, dropped=1, pending=0, queued=0, error=None))
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        analyze(directory)


def test_long_seven_reference_ledger_uses_exact_contract(ledger):
    directory, path, rows = ledger
    chunk = json.loads((directory / "client/chunks.jsonl").read_text())
    chunk["num_frames"] = 720
    (directory / "client/chunks.jsonl").write_text(json.dumps(chunk) + "\n")
    for rank in (0, 1):
        extra = next(
            r
            for r in rows
            if r["rank"] == rank and r.get("component") == "reference_vae"
        ).copy()
        extra["span_id"] = f"{rank}:encoder:seventh"
        rows.append(extra)
    for row in rows:
        if row["stage"] == "Decoder":
            row["output_frame_range"] = [0, 720]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert analyze(directory, expected_frames=720, expected_references=7)["valid"]
    with pytest.raises(ValueError, match="incomplete"):
        analyze(directory)


def test_conditioner_without_qwen_span_requires_explicit_unmeasured_scope(ledger):
    directory, path, rows = ledger
    rows = [r for r in rows if r.get("component") != "qwen"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Encoder coverage"):
        analyze(directory)
    report = analyze(directory, expected_qwen=0)
    assert report["valid"]
    assert report["qwen_scope"] == "not_measured_conditioner_callback"
    assert report["requests"][0]["stages"]["Encoder"]["count"] == 7


def test_indoor_three_reference_case_is_supported_by_analysis_cli(ledger):
    import subprocess
    import sys
    from pathlib import Path

    directory, path, rows = ledger
    chunk_path = directory / "client/chunks.jsonl"
    chunk = json.loads(chunk_path.read_text())
    chunk["num_frames"] = 720
    chunk_path.write_text(json.dumps(chunk) + "\n")
    rows = [
        r
        for r in rows
        if r.get("component") != "qwen"
        and not (
            r.get("component") == "reference_vae"
            and int(r["span_id"].rsplit(":", 1)[1]) > 3
        )
    ]
    for row in rows:
        if row["stage"] == "Decoder":
            row["output_frame_range"] = [0, 720]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = directory / "analysis.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "analyze_stage_timing.py"),
            str(directory),
            "--case",
            "30s-indoor",
            "--output",
            str(output),
        ],
        check=True,
    )
    report = json.loads(output.read_text())
    assert report["valid"]
    assert report["requests"][0]["stages"]["Encoder"]["count"] == 4
