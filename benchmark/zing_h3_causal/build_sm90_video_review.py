# SPDX-License-Identifier: Apache-2.0
"""Build synchronized source/baseline/candidate video review from completed runs."""

import argparse
import hashlib
import html
import json
import shutil
import statistics
import subprocess
from pathlib import Path


def retime_source(source, destination):
    """Keep every decoded frame, replacing all container timing with 24 FPS."""
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(source),
            ]
        )
    )["streams"][0]
    # A raw frame pipe also discards the input packet duration. setpts alone
    # leaves an extra final-frame duration with some VFR files on FFmpeg 5.
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-threads",
            "2",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "settb=1/24,setpts=N",
            "-enc_time_base",
            "1:24",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-n",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{info['width']}x{info['height']}",
                "-framerate",
                "24",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-preset",
                "fast",
                "-crf",
                "10",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(destination),
            ],
            stdin=decoder.stdout,
            check=True,
        )
    finally:
        decoder.stdout.close()
        status = decoder.wait()
    if status:
        raise subprocess.CalledProcessError(status, decoder.args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-sha", help="Full pushed commit of this report builder")
    p.add_argument("--run-label", default="Decoder 开关对照")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    frozen = (args.results / "experiment.json").is_file()
    cases_root = args.results / ("cases" if frozen else "decoder-block/client/cases")
    for case_path in sorted(cases_root.glob("*/streaming-case.json")):
        case = json.loads(case_path.read_text())
        name = case["label"]
        assets_by_local = {a["local_uri"]: a for a in case["assets"]}

        def reference_identity(path, media_id):
            asset = assets_by_local[Path(path).as_uri()]
            return {
                "id": media_id,
                **{k: asset[k] for k in ("source_uri", "sha256", "bytes")},
            }

        row = {
            "id": name,
            "title": case.get("display_name", name),
            "frames": case["frames"],
            "width": case["width"],
            "height": case["height"],
            "prompt": case["prompt"],
            "updates": case.get("prompt_updates", []),
            "references": [
                reference_identity(p, i)
                for p, i in zip(case["references"], case["reference_ids"], strict=True)
            ],
        }
        media = args.output / "media" / name
        media.mkdir(parents=True)
        for cell in ("decoder-block", "decoder-sm90"):
            folder = (
                args.results / cell / name
                if frozen
                else args.results / cell / "client" / name / "measured-0"
            )
            summary = json.loads((folder / "summary.json").read_text())
            if summary.get("success") is False:
                raise ValueError(f"Failed request: {folder}")
            receipt = json.loads((folder / "media.json").read_text())
            if receipt["mode"] != "full" or receipt["output_frames"] != case["frames"]:
                raise ValueError(f"Incomplete full video: {folder}")
            if frozen:
                frozen_identity = summary["frozen_latents"]["files"]
                if cell == "decoder-block":
                    baseline_identity = frozen_identity
                elif frozen_identity != baseline_identity:
                    raise ValueError("Decoder cells consumed different frozen latents")
                row[cell + "_timing"] = {
                    k: summary[k]
                    for k in ("sample_count", "median_ms", "p95_ms", "scope")
                }
                formal = [run for run in summary["runs"] if run["phase"] == "formal"]
                host = sorted(
                    push["host_ms"]
                    for run in formal
                    for push in run["pushes"]
                    if push["frames"]
                )
                row[cell + "_metrics"] = dict(
                    cuda_median_ms=summary["median_ms"],
                    cuda_p95_ms=summary["p95_ms"],
                    host_median_ms=statistics.median(host),
                    host_p95_ms=host[int(0.95 * (len(host) - 1))],
                    samples=summary["sample_count"],
                    formal_runs=len(formal),
                    switch_environment=summary.get("switch_environment"),
                    source_sha=summary["source_sha"],
                    **{
                        f"resident_{kind}_mib": statistics.median(
                            run["resident"][kind] for run in formal
                        )
                        / (1 << 20)
                        for kind in ("allocated", "reserved")
                    },
                    **{
                        f"peak_{kind}_mib": max(run["peak"][kind] for run in formal)
                        / (1 << 20)
                        for kind in ("allocated", "reserved")
                    },
                )
                data = args.output / "data" / cell / name
                data.mkdir(parents=True)
                shutil.copyfile(folder / "summary.json", data / "summary.json")
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
                        "-show_streams",
                        "-of",
                        "json",
                        str(src),
                    ]
                )
            )
            stream = probe["streams"][0]
            if (stream["width"], stream["height"], int(stream["nb_read_frames"])) != (
                case["width"],
                case["height"],
                case["frames"],
            ):
                raise ValueError(f"Encoded dimensions/frame count mismatch: {src}")
            dest = media / (cell + ".mp4")
            shutil.copyfile(src, dest)
            row[cell] = str(dest.relative_to(args.output))
            row[cell + "_rgb_sha256"] = receipt["output_sha256"]
        source = Path(case["source_video"])
        # Product recordings are VFR. Replay consumes frames at 24 FPS, so
        # retime the preview by frame index without duplicating/dropping frames.
        source_preview = media / "source.mp4"
        retime_source(source, source_preview)
        source_probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-show_streams",
                    "-of",
                    "json",
                    str(source_preview),
                ]
            )
        )["streams"][0]
        if (
            int(source_probe["nb_read_frames"]) != case["frames"]
            or abs(float(source_probe["duration"]) - case["frames"] / 24) > 1 / 24
        ):
            raise ValueError("Source preview retiming changed frame coverage")
        row["source_preview"] = (
            "Original frames retimed by index to replay 24 FPS; no sampling, repetition or FPS conversion"
        )
        row["source"] = str((media / "source.mp4").relative_to(args.output))
        # Do not expose remote absolute input paths in the published update data.
        row["updates"] = [
            {
                "source_frame_start": u["source_frame_start"],
                "prompt": u["prompt"],
                **(
                    {
                        "references": [
                            reference_identity(r["path"], r["id"])
                            for r in u["references"]
                        ]
                    }
                    if "references" in u
                    else {}
                ),
            }
            for u in row["updates"]
        ]
        rows.append(row)
    if len(rows) != 8:
        raise ValueError("Expected all 8 frozen product cases")
    provenance = {
        "human_visual_acceptance": "pending",
        "builder_source_sha": args.source_sha,
        "comparison": (
            "identical frozen generated DiT tensors, verified by SHA256 before both decoders"
            if frozen
            else "independent complete replays with identical inputs, seed and sampling; DiT latents are not frozen"
        ),
        "performance_scope": (
            "offline isolated Decoder push timing; production three-role E2E unverified"
            if frozen
            else "single GPU with text CPU offload; not production three-role acceptance"
        ),
        "cells": {},
    }
    for cell in ("decoder-block", "decoder-sm90"):
        if frozen:
            experiment = json.loads((args.results / "experiment.json").read_text())
            provenance["cells"][cell] = dict(
                experiment,
                source_sha=experiment.get("phase_sources", {}).get(
                    cell, experiment["source_sha"]
                ),
                sm90_fast_decode=cell == "decoder-sm90",
            )
            continue
        provenance["cells"][cell] = json.loads(
            (args.results / cell / "deployment.json").read_text()
        )
        dest = args.output / "data" / cell
        dest.mkdir(parents=True)
        for name in ("results.json", "config.json", "case-set.json"):
            source = args.results / cell / "client" / name
            if source.is_file():
                shutil.copyfile(source, dest / name)
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2)
    )
    (args.output / "cases.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2)
    )
    template = Path(__file__).with_name("sm90_video_review.html").read_text()
    (args.output / "index.html").write_text(
        template.replace("__RUN_LABEL__", html.escape(args.run_label))
        .replace(
            "__METHOD_SUMMARY__",
            (
                "两组 Decoder 读取完全相同的冻结 DiT latent；解码前逐文件校验 SHA256"
                if frozen
                else "两组独立完整回放，输入、seed 与采样一致，DiT latent 未冻结"
            ),
        )
        .replace(
            "__PERFORMANCE_SCOPE__",
            (
                "本轮仅测离线 Decoder，生产三角色 E2E 未验证"
                if frozen
                else "单卡 text CPU offload，不能代表生产三角色性能"
            ),
        )
        .replace(
            "__CASES_JSON__",
            json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c"),
        )
    )
    files = [
        {
            "path": str(f.relative_to(args.output)),
            "bytes": f.stat().st_size,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        }
        for f in sorted(args.output.rglob("*"))
        if f.is_file()
    ]
    (args.output / "files.json").write_text(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
