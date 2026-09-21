# SPDX-License-Identifier: Apache-2.0
"""Check every measured raw pair against request-derived H3 shapes and invariants."""

import argparse
import json
import math
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--sp-sizes", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    target = json.loads(Path(args.request).read_text())["messages"][1]["content"][0]
    frames, height, width = (target["output"][k] for k in ("frames", "height", "width"))
    full, tail = divmod(frames - 1, 17)
    expected = {
        "visual_latents": [
            1,
            24,
            full * 5 + 1 + math.ceil(tail / 4),
            height // 16,
            width // 16,
        ],
        "audio": (
            [2, int(frames / 24 * 32000)] if target["type"] == "video_audio" else None
        ),
    }
    report = {
        "shape_basis": "H3: 16x spatial, 5 latents per 17 frames, temporal tail stride 4; 24 FPS, stereo 32 kHz",
        "expected": expected,
        "pairs": [],
        "valid": True,
    }
    for sp in args.sp_sizes:
        reference = None
        for index in range(args.repeats):
            pair = {}
            for backend in ("native", "taeh3"):
                path = Path(args.root) / f"sp{sp}" / backend / f"sglang-{index}-raw.pt"
                data = torch.load(path, map_location="cpu", weights_only=True)
                for key, shape in expected.items():
                    value = data.get(key)
                    if shape is None:
                        assert value is None, (path, key, "unexpected value")
                    else:
                        assert value is not None and list(value.shape) == shape, (
                            path,
                            key,
                            shape,
                        )
                        assert torch.isfinite(value).all(), (path, key, "non-finite")
                pair[backend] = data
            if reference is None:
                reference = pair["native"]
            row = {"sp": sp, "repeat": index}
            for key in expected:
                a, b = pair["native"].get(key), pair["taeh3"].get(key)
                row[key + "_decoder_exact"] = a is b if a is None else torch.equal(a, b)
                row[key + "_repeat_exact"] = (
                    a is reference.get(key)
                    if a is None
                    else torch.equal(a, reference[key])
                )
            report["pairs"].append(row)
            report["valid"] &= all(v for k, v in row.items() if k.endswith("_exact"))
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["valid"]:
        raise RuntimeError("Decoder switching or repeated requests changed raw samples")


if __name__ == "__main__":
    main()
