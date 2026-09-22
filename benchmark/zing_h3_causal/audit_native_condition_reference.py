#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run pinned native conditioning preprocessing in a separate Python environment.

Consumes immutable decoded media/serving artifacts from audit_source_video_condition.
No SGLang imports or learned model weights. Does not alter the serving environment.
"""

import argparse
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from audit_source_video_condition import NATIVE_REVISION, VideoCapture, load_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-audit", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--decoded-media", type=Path)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if version("transformers") != "5.5.3":
        raise ValueError("Native reference requires Transformers 5.5.3")
    packet_path = args.decoded_media or args.serving_audit / "decoded_media.pt"
    packet = torch.load(packet_path, map_location="cpu", weights_only=True)
    serving = torch.load(
        args.serving_audit / "preprocessing.pt", map_location="cpu", weights_only=True
    )
    serving_report = json.loads((args.serving_audit / "report.json").read_text())
    if packet["request_sha256"] != serving_report["request_sha256"]:
        raise ValueError("Decoded media and serving request hashes differ")
    config = {
        "pretrained_dir": str(args.model_root),
        "condition_encoder_config": packet["condition_encoder_config"],
        "generator_config": {"modality_tags": {"video": 0, "text": 1}},
    }
    with tempfile.TemporaryDirectory(prefix="zing-native-reference-") as directory:
        native, source_hash = load_native(None, directory, args.native_source)
        processor = native.H3ConditionInputProcessor(config)
        capture = VideoCapture(processor.processor.video_processor)
        processor.processor.video_processor = capture
        refs = []
        for item in packet["media"]:
            kind, pixels = item["type"], item.get("frames")
            refs.append(
                native.H3ConditionReference(
                    kind="video" if kind == "video_audio" else kind,
                    image=pixels[:, 0].permute(1, 2, 0).numpy()
                    if kind == "image"
                    else None,
                    frames=pixels.permute(1, 2, 3, 0).numpy()
                    if kind in {"video", "video_audio"}
                    else None,
                    fps=item.get("fps"),
                    has_audio=kind in {"audio", "video_audio"},
                )
            )
        expected = processor.prepare(packet["prompt"], refs)
        model = object.__new__(Qwen3VLModel)
        torch.nn.Module.__init__(model)
        model.config = AutoConfig.from_pretrained(
            str(args.model_root), subfolder="text_encoder"
        )
        kwargs = {
            key: value
            for key, value in expected.model_inputs.items()
            if key
            in {"input_ids", "attention_mask", "image_grid_thw", "video_grid_thw"}
        }
        if "mm_token_type_ids" in inspect.signature(model.get_rope_index).parameters:
            kwargs["mm_token_type_ids"] = expected.model_inputs["mm_token_type_ids"]
        rope = model.get_rope_index(**kwargs)
        checks = {
            key: key in serving["serving"]
            and torch.equal(value, serving["serving"][key])
            for key, value in expected.model_inputs.items()
        }
        checks["field_set"] = set(expected.model_inputs) == set(serving["serving"])
        checks["h3_modality_tags"] = torch.equal(
            expected.modality_ids, serving["serving_tags"]
        )
        checks["backend_rope"] = all(
            torch.equal(a, b) for a, b in zip(rope, serving["rope"]["serving"])
        )
        with np.load(args.serving_audit / "padded_pixels.npz") as arrays:
            serving_keys = sorted(
                key for key in arrays.files if key.startswith("serving_")
            )
            checks["padded_pixels"] = len(serving_keys) == len(capture.videos) and all(
                np.array_equal(video, arrays[f"serving_{index}"])
                for index, video in enumerate(capture.videos)
            )
        checks["sampling_disabled"] = all(
            call["do_sample_frames"] is False for call in capture.calls
        )
        checks["video_call_count"] = len(capture.calls) == int(bool(capture.videos))
        component_hashes = {
            str(path.relative_to(args.model_root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for folder in ("tokenizer", "processor")
            for path in sorted((args.model_root / folder).rglob("*"))
            if path.is_file()
        }
        checks["component_file_sha256"] = (
            component_hashes == serving_report["component_file_sha256"]
        )
        torch.save(
            {
                "model_inputs": expected.model_inputs,
                "tags": expected.modality_ids,
                "rope": rope,
            },
            args.output / "native_reference.pt",
        )
        np.savez(
            args.output / "native_padded_pixels.npz",
            **{str(i): v for i, v in enumerate(capture.videos)},
        )
        packages = {
            name: version(name)
            for name in ("torch", "torchvision", "transformers", "numpy")
        }
        report = {
            "native_revision": NATIVE_REVISION,
            "native_source_sha256": source_hash,
            "decoded_media_sha256": hashlib.sha256(
                packet_path.read_bytes()
            ).hexdigest(),
            "request_sha256": packet["request_sha256"],
            "serving_report_sha256": hashlib.sha256(
                (args.serving_audit / "report.json").read_bytes()
            ).hexdigest(),
            "packages": packages,
            "checks": checks,
            "preprocessing_equal": all(checks.values()),
            "full_native_dependency_match": packages["torch"].split("+")[0] == "2.9.1"
            and packages["torchvision"].split("+")[0] == "0.24.1",
            "feature_validation": "not_run",
            "e2e_validation": "not_run",
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if report["preprocessing_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
