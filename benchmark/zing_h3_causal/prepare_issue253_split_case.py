# SPDX-License-Identifier: Apache-2.0
"""Prepare one exact Issue #253 case for the dedicated Encoder baseline."""

import argparse
import json
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

from prepare_encoder_overlap import digest, validate_deployment_manifest
from prepare_whitebox_video_smoke import load_cases, materialize_case


def media_path(item):
    return Path(unquote(urlparse(item["uri"]).path))


def probe_video(path):
    return json.loads(
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
                str(path),
            ],
            text=True,
        )
    )["streams"][0]


def first_rgb(path):
    return subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ]
    )


def prepare(args):
    manifest = json.loads(args.deployment_manifest.read_text())
    profile = validate_deployment_manifest(
        manifest, args.model.resolve(), args.taeh3_weights.resolve()
    )
    case = load_cases(args.case_manifest)[args.case]
    user, target = case["messages"]
    outputs = [x["output"] for x in target["content"] if x["type"] == "video"]
    if len(outputs) != 1 or target.get("source", {}).get("type") != "video":
        raise ValueError("Case must contain one video output and one video source")
    spec = outputs[0]
    frames, width, height = (spec[k] for k in ("frames", "width", "height"))
    fps = (
        case.get("metadata", {})
        .get("smoke_contract", {})
        .get("target", {})
        .get("fps", 24)
    )
    if fps != 24 or min(frames, width, height) < 1:
        raise ValueError("Issue #253 cases require nonempty 24 FPS video")
    conditions = target.get("conditions", [])
    reference_ids = [x.get("content_id") for x in conditions]
    if (
        not reference_ids
        or len(reference_ids) != len(set(reference_ids))
        or any(x.get("latent_modalities") != ["visual"] for x in conditions)
    ):
        raise ValueError("Reference conditions are incomplete or duplicated")
    media = {x.get("id"): x for x in user.get("content", []) if x.get("id")}
    if any(media.get(key, {}).get("type") != "image" for key in reference_ids):
        raise ValueError("Reference conditions do not resolve to ordered images")

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    request, receipt = materialize_case(case, root / "case", media_root=args.media_root)
    (root / "request.json").write_text(json.dumps(request, indent=2))
    local_user, local_target = request["messages"]
    source = media_path(local_target["source"])
    prompt_videos = [x for x in local_user["content"] if x.get("type") == "video"]
    if len(prompt_videos) != 1 or prompt_videos[0].get("id") != "<Video 1>":
        raise ValueError("Case must retain its single <Video 1> prompt video")
    prompt_video = media_path(prompt_videos[0])
    if first_rgb(source) != first_rgb(prompt_video):
        raise ValueError(
            "Continuous source preview differs from prompt video frame zero"
        )
    probe = probe_video(source)
    if int(probe["nb_read_frames"]) != frames or probe["avg_frame_rate"] != "24/1":
        raise ValueError("Source frame count or FPS differs from the frozen case")

    rgb = root / "source.rgb"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(source),
            "-vf",
            f"scale={width}:{height}:flags=bilinear",
            "-frames:v",
            str(frames),
            "-an",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            str(rgb),
        ],
        check=True,
    )
    if rgb.stat().st_size != frames * width * height * 3:
        raise ValueError("Decoded source is incomplete")
    prompt_frame = root / "prompt-frame0.png"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(prompt_video),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            str(prompt_frame),
        ],
        check=True,
    )

    model = root / "model"
    model.mkdir()
    for source_file in args.model.resolve().iterdir():
        if source_file.name != "zing_realtime_profile.json":
            (model / source_file.name).symlink_to(source_file)
    (model / "zing_realtime_profile.json").write_text(json.dumps(profile, indent=2))
    prepared = {
        "case_id": args.case,
        "case_manifest": str(args.case_manifest.resolve()),
        "case_manifest_sha256": digest(args.case_manifest),
        "expected_frames": frames,
        "reference_ids": reference_ids,
        "measured_runs": 3,
        "stage_timing": "host",
        "run_lifecycle_probe": False,
        "encoder_tile_mode": "parallel",
        "cudnn_benchmark_limit": 1,
        "width": width,
        "height": height,
        "model": str(model),
        "request": str(root / "request.json"),
        "source_rgb": str(rgb),
        "taeh3_weights": str(args.taeh3_weights.resolve()),
        "prompt_video_frame": str(prompt_frame),
        "prompt_video_id": "<Video 1>",
        "layouts": ["enc1-sp2-serial", "enc1-sp2-overlap"],
    }
    (root / "input.json").write_text(json.dumps(prepared, indent=2))
    receipt.update(
        case_id=args.case,
        frames=frames,
        width=width,
        height=height,
        fps=fps,
        reference_ids=reference_ids,
        source_probe=probe,
        source_rgb={
            "path": str(rgb),
            "bytes": rgb.stat().st_size,
            "sha256": digest(rgb),
        },
        request={
            "path": str(root / "request.json"),
            "sha256": digest(root / "request.json"),
        },
        source_extension="none; exact frozen source decoded once",
        deployment_manifest_sha256=digest(args.deployment_manifest),
    )
    (root / "PREP_COMPLETE.json").write_text(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--taeh3-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
