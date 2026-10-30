# SPDX-License-Identifier: Apache-2.0
"""Download and verify the frozen product replay assets without running inference."""

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from case_sets import load_case_set
from PIL import Image
from prepare_whitebox_video_smoke import expected_assets, local_source_path, sha256_file


def archive_timing(manifest, intervals):
    """Keep source PTS, recording wall time and fixed-FPS replay time distinct."""
    intervals = list(intervals)
    start = min(row["start_time_us"] for row in intervals)
    end = max(row["end_time_us"] for row in intervals)
    return {
        "source_pts_start_us": start,
        "source_pts_end_us": end,
        "source_pts_span_seconds": (end - start) / 1e6,
        "recording_wall_seconds": (
            datetime.fromisoformat(manifest["finished_at"])
            - datetime.fromisoformat(manifest["created_at"])
        ).total_seconds(),
        "replay_duration_seconds": manifest["rendering"]["model_frames"] / 24,
    }


def prepare(media_root, profile, *, ffprobe="ffprobe"):
    receipt, seen = [], set()
    for member in load_case_set("product-case-20261029")["members"]:
        case = member["request"]
        region = case["provenance"]["region"]
        for uri, identity in expected_assets(case).items():
            if uri in seen:
                continue
            seen.add(uri)
            path = local_source_path(uri, media_root)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
                    staged = Path(temporary) / "asset"
                    subprocess.run(
                        [
                            "aliyun",
                            "oss",
                            "cp",
                            uri,
                            str(staged),
                            "--profile",
                            profile,
                            "-e",
                            f"oss-{region}.aliyuncs.com",
                        ],
                        check=True,
                    )
                    verify_asset(staged, identity)
                    os.replace(staged, path)
            verify_asset(path, identity)
            if uri != case["messages"][1]["source"]["uri"]:
                with Image.open(path) as image:
                    image.verify()
            receipt.append({"uri": uri, "path": str(path.resolve()), **identity})
        source = local_source_path(case["messages"][1]["source"]["uri"], media_root)
        # Decode the entire video: header duration/nb_frames is not sufficient.
        video = json.loads(
            subprocess.check_output(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=width,height,nb_read_frames",
                    "-of",
                    "json",
                    str(source),
                ],
                text=True,
            )
        )["streams"][0]
        expected = case["messages"][1]["content"][0]["output"]
        if (video["width"], video["height"], int(video["nb_read_frames"])) != (
            expected["width"],
            expected["height"],
            expected["frames"],
        ):
            raise ValueError(
                f"Decoded source geometry or frame count differs: {source}"
            )
    return receipt


def verify_asset(path, identity):
    if (
        path.stat().st_size != identity["bytes"]
        or sha256_file(path) != identity["sha256"]
    ):
        raise ValueError(f"Asset differs from frozen archive: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--profile", required=True, help="Existing Aliyun CLI profile")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    assets = prepare(args.media_root, args.profile, ffprobe=args.ffprobe)
    args.media_root.mkdir(parents=True, exist_ok=True)
    (args.media_root / "product-case-20261029-assets.json").write_text(
        json.dumps({"status": "verified", "assets": assets}, indent=2) + "\n"
    )
