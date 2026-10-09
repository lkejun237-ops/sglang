# SPDX-License-Identifier: Apache-2.0
"""Zero-GPU preparation from explicit deployment and case manifests."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

from prepare_whitebox_video_smoke import DEFAULT_CASE, DEFAULT_MANIFEST, load_cases


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def set_request_resolution(request, width, height):
    """Keep source and six references intact; update only target geometry."""
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("Width must be a positive multiple of 32 and height of 32")
    target = request["messages"][1]
    outputs = [x["output"] for x in target["content"] if x["type"] == "video"]
    if len(outputs) != 1:
        raise ValueError("Expected one target video output")
    outputs[0].update(width=width, height=height)
    request["metadata"]["smoke_contract"]["target"].update(width=width, height=height)


def validate_deployment_manifest(manifest, model, taeh3_weights):
    """Reject changed weights/configuration before creating prepared outputs."""
    profile = dict(manifest["realtime_profile"])
    if profile.get("model_config_sha256") != digest(model / "zing_config.json"):
        raise ValueError("Model configuration differs from the deployment manifest")
    if manifest.get("taeh3_sha256") != digest(taeh3_weights):
        raise ValueError("TAEH3 checkpoint differs from the deployment manifest")
    if (
        profile.get("solver") != "few_step"
        or len(profile.get("denoising_step_list", [])) != 4
        or profile.get("noise_version") != "indexed-cpu-v1"
        or profile.get("source_encoder", {}).get("mode") != "progressive"
    ):
        raise ValueError(
            "Deployment manifest must select four-step progressive streaming"
        )
    return profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deployment-manifest", type=Path, required=True)
    p.add_argument("--case-manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--case", default=DEFAULT_CASE)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--taeh3-weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32:
        p.error("Width must be a positive multiple of 32 and height of 32")
    manifest = json.loads(args.deployment_manifest.read_text())
    profile = validate_deployment_manifest(
        manifest, args.model.resolve(), args.taeh3_weights
    )
    case = load_cases(args.case_manifest)[args.case]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    model = root / "model"
    model.mkdir()
    for source in args.model.resolve().iterdir():
        if source.name != "zing_realtime_profile.json":
            (model / source.name).symlink_to(source)
    (model / "zing_realtime_profile.json").write_text(json.dumps(profile, indent=2))
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("prepare_whitebox_video_smoke.py")),
            "--case",
            args.case,
            "--manifest",
            str(args.case_manifest),
            "--media-root",
            str(args.media_root),
            "--output-dir",
            str(root / "case"),
        ],
        check=True,
    )
    request = json.loads((root / "case/request.json").read_text())
    set_request_resolution(request, args.width, args.height)
    refs = [x for x in request["messages"][0]["content"] if x["type"] == "image"]
    assert [x["id"] for x in refs] == [f"<Picture {i}>" for i in range(1, 7)]
    videos = [x for x in request["messages"][0]["content"] if x["type"] == "video"]
    assert len(videos) == 1 and videos[0]["id"] == "<Video 1>"
    assert videos[0]["uri"] == request["messages"][1]["source"]["uri"]
    assets = root / "assets"
    assets.mkdir()
    for index, ref in enumerate(refs):
        source = Path(unquote(urlparse(ref["uri"]).path))
        target = assets / f"reference-{index}{source.suffix}"
        shutil.copyfile(source, target)
        ref["uri"] = target.as_uri()
    source = Path(unquote(urlparse(request["messages"][1]["source"]["uri"]).path))
    target = assets / ("source" + source.suffix)
    shutil.copyfile(source, target)
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
                str(target),
            ],
            text=True,
        )
    )["streams"][0]
    contract = case["metadata"]["smoke_contract"]
    fps = contract["target"]["fps"]
    if (
        int(probe["nb_read_frames"]) != contract["source_stream_frames"]
        or probe["avg_frame_rate"] != f"{fps}/1"
    ):
        raise ValueError("Source frames or frame rate differ from the case manifest")
    request["messages"][1]["source"]["uri"] = target.as_uri()
    videos[0]["uri"] = target.as_uri()
    prompt_frame = assets / "source-frame0.png"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(target),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            str(prompt_frame),
        ],
        check=True,
    )

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

    if first_rgb(target) != first_rgb(prompt_frame):
        raise ValueError("Prompt snapshot does not match original source frame zero")
    (root / "request.json").write_text(json.dumps(request, indent=2))
    for name, frames in (("source.rgb", 720), ("soak.rgb", 2880)):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-stream_loop",
                "-1",
                "-i",
                str(target),
                "-vf",
                f"scale={args.width}:{args.height}:flags=bilinear",
                "-frames:v",
                str(frames),
                "-an",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                str(root / name),
            ],
            check=True,
        )
        assert (root / name).stat().st_size == frames * args.width * args.height * 3
    inputs = dict(
        encoder_tile_mode="parallel",
        cudnn_benchmark_limit=1,
        width=args.width,
        height=args.height,
        model=str(model),
        request=str(root / "request.json"),
        source_rgb=str(root / "source.rgb"),
        soak_rgb=str(root / "soak.rgb"),
        taeh3_weights=str(args.taeh3_weights.resolve()),
        prompt_video_frame=str(prompt_frame),
        prompt_video_id="<Video 1>",
    )
    (root / "input.json").write_text(json.dumps(inputs, indent=2))
    files = [
        *assets.iterdir(),
        root / "source.rgb",
        root / "soak.rgb",
        root / "request.json",
        model / "zing_realtime_profile.json",
    ]
    git_identity = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )
    receipt = dict(
        width=args.width,
        height=args.height,
        case=args.case,
        case_manifest_sha256=digest(args.case_manifest),
        deployment_manifest_sha256=digest(args.deployment_manifest),
        source_extension="repeat original source; loop boundaries are synthetic",
        source_probe=probe,
        source_commit=git_identity.stdout.strip()
        if git_identity.returncode == 0
        else None,
        preparer_sha256=digest(Path(__file__)),
        files=[
            dict(path=str(f), bytes=f.stat().st_size, sha256=digest(f)) for f in files
        ],
        model_files=[
            dict(path=str(f), bytes=f.stat().st_size)
            for f in args.model.rglob("*")
            if f.is_file()
        ],
    )
    (root / "PREP_COMPLETE.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
