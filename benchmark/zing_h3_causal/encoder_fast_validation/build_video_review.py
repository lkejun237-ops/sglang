# SPDX-License-Identifier: Apache-2.0
"""Build a self-contained synchronized baseline/candidate video review folder."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def build(root, output):
    output.mkdir(parents=True, exist_ok=False)
    media = output / "media"
    media.mkdir()
    variants = ("baseline", "candidate")
    configs = {v: read(root / v / "visual/config.json") for v in variants}
    assert configs["baseline"]["seed"] == configs["candidate"]["seed"]
    assert (
        configs["baseline"]["case_collection"]
        == configs["candidate"]["case_collection"]
    )
    summaries = {v: read(root / v / "performance/summary.json") for v in variants}
    requests = {
        v: {
            row["case_id"]: row
            for row in [
                json.loads(line)
                for line in (root / v / "visual/requests.jsonl")
                .read_text()
                .splitlines()
            ]
            if row["phase"] == "measured"
        }
        for v in variants
    }
    cases = []
    for case in configs["baseline"]["cases"]:
        label = case["label"]
        other = next(c for c in configs["candidate"]["cases"] if c["label"] == label)
        for field in (
            "case_id",
            "frames",
            "width",
            "height",
            "prompt",
            "prompt_updates",
            "reference_ids",
        ):
            assert case.get(field) == other.get(field), (label, field)
        item = {
            k: case[k] for k in ("case_id", "display_name", "frames", "width", "height")
        }
        item["updates"] = [
            {"frame": u["source_frame_start"], "prompt": u.get("prompt", "")}
            for u in case.get("prompt_updates", [])
        ]
        item["source_width"] = case.get("source_width", case["width"])
        item["source_height"] = case.get("source_height", case["height"])
        item["prompt"] = case["prompt"]
        assert case["assets"] == other["assets"], label
        assert (
            requests["baseline"][case["case_id"]]["input_sha256"]
            == requests["candidate"][case["case_id"]]["input_sha256"]
        ), label
        item["input_sha256"] = requests["baseline"][case["case_id"]]["input_sha256"]
        item["applied_updates"] = {
            v: requests[v][case["case_id"]]
            .get("streaming", {})
            .get("prompt_updates", [])
            for v in variants
        }
        item["metrics"] = {v: summaries[v][label] for v in variants}
        for variant in variants:
            applied = item["applied_updates"][variant]
            assert [u["source_frame_start"] for u in applied] == [
                u["source_frame_start"] for u in case.get("prompt_updates", [])
            ], (label, variant)
            assert all(u["status"] == "applied" for u in applied), (label, variant)
        item["videos"] = {}
        for variant in variants:
            folder = root / variant / "visual" / label / "measured-0"
            receipt = read(folder / "media.json")
            assert (
                receipt["mode"] == "video"
                and receipt["output_frames"] == case["frames"]
                and receipt["error"] is None
            )
            src = folder / "output.mp4"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(src), "-f", "null", "-"], check=True
            )
            probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-count_frames",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=nb_read_frames,width,height,avg_frame_rate",
                        "-of",
                        "json",
                        str(src),
                    ],
                    text=True,
                )
            )["streams"][0]
            assert int(probe["nb_read_frames"]) == case["frames"]
            assert (probe["width"], probe["height"], probe["avg_frame_rate"]) == (
                case["width"],
                case["height"],
                "24/1",
            )
            dst = media / (label + "-" + variant + ".mp4")
            shutil.copy2(src, dst)
            item["videos"][variant] = {
                "path": str(dst.relative_to(output)),
                "bytes": dst.stat().st_size,
                "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
                "rgb_sha256": receipt["output_sha256"],
            }
        if (root / "baseline/diagnostics/GATE.json").exists():
            from diagnostic_report import memory_pair

            for variant in variants:
                assert read(root / variant / "diagnostics/GATE.json")["passed"]
            item["memory"], item["components"] = memory_pair(root, case["case_id"])
        cases.append(item)
    data = {
        "cases": cases,
        "seed": configs["baseline"]["seed"],
        "source": read(root / "SOURCE.json") if (root / "SOURCE.json").exists() else {},
        "hardware": read(root / "baseline/HARDWARE.json")
        if (root / "baseline/HARDWARE.json").exists()
        else {},
        "commands": {
            v: read(root / v / "commands.json")
            for v in variants
            if (root / v / "commands.json").exists()
        },
        "timing": {
            v: {
                k: read(root / v / "performance/config.json").get(k)
                for k in (
                    "timing_origin",
                    "timing_boundaries",
                    "input_mode",
                    "media_mode",
                    "warmup_runs",
                    "measured_runs",
                    "seed",
                    "batch_pattern",
                )
            }
            for v in variants
            if (root / v / "performance/config.json").exists()
        },
        "scope": "Complete 24 FPS lossy videos; performance uses a separate metrics-only pass. Component timings/memory not measured.",
    }
    if any("memory" in case for case in cases):
        data["scope"] = (
            "Separate metrics timing, whole-role allocator diagnostics and full 24 FPS lossy videos. Encoder subcomponent memory is shared."
        )
        data["title"] = "数值优化开关 A/B · 完整视频对比"
        data["description"] = (
            "同源码、模型、输入与seed；两个FAST开关固定开启，只切换数值优化许可。显存诊断与正式性能/视频分开。"
        )
        data["labels"] = {
            "baseline": "BASELINE · 数值优化关闭",
            "candidate": "CANDIDATE · 数值优化默认开启",
        }
    (output / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    template = Path(__file__).with_name("video_review.html").read_text()
    (output / "index.html").write_text(
        template.replace(
            "__DATA_JSON__",
            json.dumps(data, ensure_ascii=False).replace("<", "\\u003c"),
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build(a.root, a.output)
