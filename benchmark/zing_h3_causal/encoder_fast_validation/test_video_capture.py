# SPDX-License-Identifier: Apache-2.0
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_artifacts import MediaCapture


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_full_video_keeps_every_frame_at_original_size_and_24fps(tmp_path):
    capture = MediaCapture(tmp_path, 32, 16, 24, mode="video")
    for i in range(24):
        capture.write(bytes([i * 10, 20, 100]) * (32 * 16))
    receipt = capture.finish()
    assert receipt["output_frames"] == 24
    assert receipt["sampled_frame_indices"] == []
    assert not (tmp_path / "video-spool.rgb").exists()
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_read_frames,avg_frame_rate",
                "-of",
                "json",
                str(tmp_path / "output.mp4"),
            ],
            text=True,
        )
    )["streams"][0]
    assert (info["width"], info["height"], int(info["nb_read_frames"])) == (32, 16, 24)
    assert info["avg_frame_rate"] == "24/1"


def test_full_video_rejects_unbounded_or_extra_frames(tmp_path):
    with pytest.raises(ValueError, match="expected frame count"):
        MediaCapture(tmp_path / "unknown", 32, 16, None, mode="video")
    capture = MediaCapture(tmp_path / "bounded", 32, 16, 1, mode="video")
    with pytest.raises(RuntimeError, match="frozen frame count"):
        capture.write(bytes(2 * 32 * 16 * 3))
    capture.finish()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_review_builder_checks_inputs_and_escapes_prompt(tmp_path):
    from build_video_review import build

    case = {
        "label": "scene",
        "case_id": "case1",
        "display_name": "场景",
        "frames": 24,
        "width": 32,
        "height": 16,
        "prompt": "</script><h1>test",
        "prompt_updates": [],
        "reference_ids": [],
        "assets": [],
    }
    config = {"seed": 42, "case_collection": {"name": "frozen"}, "cases": [case]}
    for variant in ["baseline", "candidate"]:
        root = tmp_path / "run" / variant
        visual = root / "visual"
        visual.mkdir(parents=True)
        (visual / "config.json").write_text(json.dumps(config))
        (visual / "requests.jsonl").write_text(
            json.dumps(
                {"case_id": "case1", "phase": "measured", "input_sha256": "same"}
            )
            + "\n"
        )
        perf = root / "performance"
        perf.mkdir()
        (perf / "summary.json").write_text(json.dumps({"scene": {"metrics": {}}}))
        cap = MediaCapture(visual / "scene/measured-0", 32, 16, 24, mode="video")
        cap.write(bytes(32 * 16 * 3 * 24))
        cap.finish()
    output = tmp_path / "page"
    build(tmp_path / "run", output)
    assert len(list((output / "media").glob("*.mp4"))) == 2
    page = (output / "index.html").read_text()
    assert "\\u003c/script>" in page
    row = json.loads((tmp_path / "run/candidate/visual/requests.jsonl").read_text())
    row["input_sha256"] = "different"
    (tmp_path / "run/candidate/visual/requests.jsonl").write_text(
        json.dumps(row) + "\n"
    )
    with pytest.raises(AssertionError):
        build(tmp_path / "run", tmp_path / "bad-page")


def test_component_histogram_delta_keeps_bucket_bounds_and_drop_count():
    from diagnostic_report import component_delta

    before = """sglang_zing_component_duration_seconds_count{component="encoder",measurement="cuda"} 10
sglang_zing_component_duration_seconds_sum{component="encoder",measurement="cuda"} 1
sglang_zing_component_duration_seconds_bucket{component="encoder",measurement="cuda",le="0.1"} 10
sglang_zing_component_duration_seconds_bucket{component="encoder",measurement="cuda",le="0.2"} 10
sglang_zing_component_duration_seconds_bucket{component="encoder",measurement="cuda",le="+Inf"} 10
"""
    after = before.replace("} 10", "} 12").replace("} 1\n", "} 1.3\n")
    after = after.replace('le="0.1"} 12', 'le="0.1"} 11')
    after += 'sglang_zing_component_timing_dropped_total{component="encoder",reason="queue_full"} 1\n'
    result = component_delta(before, after)
    row = result["calls"][0]
    assert row["count"] == 2
    assert row["mean_ms"] == pytest.approx(150)
    assert row["p50_ms_interval"] == [0, 100]
    assert row["p95_ms_interval"] == [100, 200]
    assert result["dropped"][0]["count"] == 1


def test_memory_report_matches_gpu_uuid_and_requires_same_request_workers(tmp_path):
    from diagnostic_report import memory_pair

    for index, variant in enumerate(("baseline", "candidate")):
        base = tmp_path / variant
        folder = base / "diagnostics/case"
        (folder / "measured").mkdir(parents=True)
        (base / "HARDWARE.json").write_text(json.dumps({"devices": [{"uuid": "abc"}]}))
        entry = {
            "pid": 100 + index,
            "role": "encoder",
            "device": 0,
            "visible_devices": "0",
            "counters": {
                "allocated": 6,
                "reserved": 8,
                "peak_allocated": 10,
                "peak_reserved": 12,
            },
        }
        (folder / "allocator.json").write_text(
            json.dumps(
                {"case_id": "case", "requests": 1, "before": [entry], "after": [entry]}
            )
        )
        # nvidia-smi uses physical index7; CUDA logical index0 maps by UUID.
        (folder / "measured/driver-memory.json").write_text(
            json.dumps({"samples": [{"rows": ["7, GPU-abc, 12, 0"]}]})
        )
        for role in ("encoder", "dit", "decoder"):
            for when in ("before", "after"):
                (folder / f"{when}-{role}.prom").write_text("")
    memory, _ = memory_pair(tmp_path, "case")
    assert memory["rows"][0]["baseline"]["driver_peak_sampled"] == 12 << 20
    path = tmp_path / "candidate/diagnostics/case/allocator.json"
    receipt = json.loads(path.read_text())
    receipt["after"][0]["pid"] += 1
    path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):
        memory_pair(tmp_path, "case")
