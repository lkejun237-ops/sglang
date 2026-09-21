#!/usr/bin/env python3
"""CPU-only checks of every measured raw archive; no model/GPU imports.

--expected-shapes is a frozen JSON object with visual_latents and audio shapes.
Derive it from the verified package/request, not from the observed first output.
A valid report proves shape/finiteness only, never strict numerical parity.
"""

import argparse
import json
from pathlib import Path

import torch
from validate_real_checkpoint import _comparison


def load_raw(path, shapes):
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        result = {}
        for key in ("visual_latents", "audio"):
            value = data.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{key} must be a tensor")
            if list(value.shape) != shapes[key]:
                raise ValueError(f"{key}: shape {list(value.shape)} != {shapes[key]}")
            if value.numel() == 0 or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{key}: empty or non-finite tensor")
            result[key] = value
        return result, None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def compare_raw(actual, expected):
    return {
        key: _comparison(actual[key], expected[key])
        for key in ("visual_latents", "audio")
    }


def check_outputs(runs, shapes, measured_runs=3):
    if measured_runs < 2:
        raise ValueError("At least two measured runs are required")
    if "1" not in runs:
        raise ValueError("SP1 is required as the comparison baseline")
    for key in ("visual_latents", "audio"):
        shape = shapes.get(key)
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(n) is not int or n <= 0 for n in shape)
        ):
            raise ValueError(f"Invalid expected shape for {key}")
    files, comparisons = [], []
    # Retain only the SP1 reference and one configuration reference at a time.
    baseline, baseline_error = load_raw(Path(runs["1"]) / "sglang-0-raw.pt", shapes)
    for sp, directory in sorted(runs.items(), key=lambda item: int(item[0])):
        reference, reference_error = load_raw(
            Path(directory) / "sglang-0-raw.pt", shapes
        )
        for index in range(measured_runs):
            path = Path(directory) / f"sglang-{index}-raw.pt"
            actual, error = (
                (reference, reference_error) if index == 0 else load_raw(path, shapes)
            )
            files.append(
                {
                    "sp": int(sp),
                    "run": index,
                    "path": str(path),
                    "valid": error is None,
                    "error": error,
                }
            )
            if index and actual is not None and reference is not None:
                comparisons.append(
                    {
                        "kind": "repeat",
                        "sp": int(sp),
                        "run": index,
                        "reference_run": 0,
                        "metrics": compare_raw(actual, reference),
                    }
                )
            del actual
        if sp != "1" and reference is not None and baseline_error is None:
            comparisons.append(
                {
                    "kind": "cross_sp",
                    "sp": int(sp),
                    "run": 0,
                    "reference_sp": 1,
                    "metrics": compare_raw(reference, baseline),
                }
            )
    return {
        "valid": all(row["valid"] for row in files),
        "matrix_complete": set(runs) == {"1", "2", "4"},
        "strict_parity": "not_evaluated_no_frozen_tolerance",
        "media_validation": "not_evaluated_use_separate_decode_checks",
        "expected_shapes": shapes,
        "files": files,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="SP=DIRECTORY")
    parser.add_argument("--expected-shapes", type=Path, required=True)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for value in args.run:
        sp, separator, directory = value.partition("=")
        if not separator or sp not in {"1", "2", "4"} or not directory or sp in runs:
            parser.error("--run must be a unique SP=DIRECTORY with SP 1, 2, or 4")
        runs[sp] = directory
    report = check_outputs(
        runs, json.loads(args.expected_shapes.read_text()), args.measured_runs
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
