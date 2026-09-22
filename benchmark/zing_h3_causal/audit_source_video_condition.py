#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare actual CPU conditioning artifacts against issue #167's pinned native.

Uses the real checkpoint processor and decoded request media, but does not run
Qwen/DiT. A passed report is preprocessing evidence, not feature or E2E evidence.
"""

import argparse
import hashlib
import importlib.util
import inspect
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

NATIVE_REVISION = "967df2fd539590b4fa88462ddb67276e5f9a8192"


class VideoCapture:
    def __init__(self, processor):
        self.processor = processor
        self.calls = []
        self.videos = []

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *, videos, **kwargs):
        self.calls.append(dict(kwargs))
        self.videos.extend(np.array(video, copy=True) for video in videos)
        return self.processor(videos=videos, **kwargs)


def load_native(repo, directory, native_source=None):
    source = (
        native_source.read_bytes()
        if native_source
        else subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "show",
                f"{NATIVE_REVISION}:src/h3/modules/condition.py",
            ]
        )
    )
    if (
        hashlib.sha256(source).hexdigest()
        != "df600e78ab897c5384371c6e96d7228b18ef4c9be49f918fc46630c06fafd661"
    ):
        raise ValueError("Native source does not match the pinned issue #167 revision")
    path = Path(directory) / "native_condition.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("audit_native_condition", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(source).hexdigest()


def load_serving_processor(root):
    from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
        AutoProcessorLoader,
        TokenizerLoader,
    )

    model_index = json.loads((root / "model_index.json").read_text())
    tokenizer_loader = TokenizerLoader(device=torch.device("cpu"))
    tokenizer_loader.component_architecture = model_index["tokenizer"][1]
    return SimpleNamespace(
        tokenizer=tokenizer_loader.load_customized(
            str(root / "tokenizer"), None, "tokenizer"
        ),
        processor=AutoProcessorLoader(device=torch.device("cpu")).load_customized(
            str(root / "processor"), None, "processor"
        ),
    )


def compare_rope(model_root, expected, captured):
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel as NativeModel,
    )
    from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import (
        Qwen3VLModel as ServingModel,
    )

    config = AutoConfig.from_pretrained(str(model_root), subfolder="text_encoder")
    positions = {}
    for name, cls, inputs in [
        ("native", NativeModel, expected),
        ("serving", ServingModel, captured),
    ]:
        # RoPE has no learned weights; call the actual backend implementation.
        model = object.__new__(cls)
        torch.nn.Module.__init__(model)
        model.config = config
        kwargs = {
            key: value
            for key, value in inputs.items()
            if key
            in {"input_ids", "attention_mask", "image_grid_thw", "video_grid_thw"}
        }
        if "mm_token_type_ids" in inspect.signature(cls.get_rope_index).parameters:
            kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"]
        positions[name] = model.get_rope_index(**kwargs)
    return positions, all(
        torch.equal(left, right)
        for left, right in zip(positions["native"], positions["serving"])
    )


def main():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        ZingH3Codecs,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    native_input = parser.add_mutually_exclusive_group(required=True)
    native_input.add_argument("--native-repo", type=Path)
    native_input.add_argument("--native-source", type=Path)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-decoded-only", action="store_true")
    args = parser.parse_args()
    # A revision is immutable; never overwrite earlier failed/successful evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    request = json.loads(args.request.read_text())
    runtime = json.loads(args.runtime_config.read_text())
    cfg = runtime["condition_encoder_config"]
    config = {
        "pretrained_dir": str(args.model_root),
        "condition_encoder_config": cfg,
        "generator_config": {"modality_tags": {"video": 0, "text": 1}},
    }
    user = next(message for message in request["messages"] if message["role"] == "user")
    prompts = [item["text"] for item in user["content"] if item["type"] == "text"]
    if len(prompts) != 1:
        raise ValueError("Audit requires exactly one user prompt")
    codec = object.__new__(ZingH3Codecs)
    codec.runtime = runtime
    codec.remote_read_fs = None
    # Decode through serving's real explicit type/clip path. Qwen has no VAE
    # resize/resample here; source/conditions retain their independent tracks.
    media = [
        codec._decode_media(item, need_audio=False)
        for item in user["content"]
        if item["type"] != "text"
    ]
    torch.save(
        {
            "media": media,
            "prompt": prompts[0],
            "condition_encoder_config": cfg,
            "request_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest(),
        },
        args.output / "decoded_media.pt",
    )
    if args.export_decoded_only:
        return 0
    captured = {}

    def capture_encoder(ids, **features):
        captured.update(
            input_ids=ids[None], attention_mask=torch.ones_like(ids[None]), **features
        )
        return torch.empty(len(ids), 0)  # deliberately no synthetic feature verdict

    with tempfile.TemporaryDirectory(prefix="zing-native-condition-") as directory:
        native, source_hash = load_native(
            args.native_repo, directory, args.native_source
        )
        native_processor = native.H3ConditionInputProcessor(config)
        serving_processor = load_serving_processor(args.model_root)
        native_capture = VideoCapture(native_processor.processor.video_processor)
        serving_capture = VideoCapture(serving_processor.processor.video_processor)
        native_processor.processor.video_processor = native_capture
        serving_processor.processor.video_processor = serving_capture
        refs = []
        for item in media:
            kind = item["type"]
            pixels = item.get("frames")
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
        expected = native_processor.prepare(prompts[0], refs)
        codec.modules = {
            "tokenizer": serving_processor.tokenizer,
            "processor": serving_processor.processor,
            "text_encoder": SimpleNamespace(encode_ids=capture_encoder),
        }
        result = codec.encode_condition(prompts[0], media)
        # The SGLang backend derives image/video masks from token IDs. Record
        # processor type IDs and compare the actual two backend RoPE functions.
        captured["mm_token_type_ids"] = torch.tensor(
            serving_processor.processor.create_mm_token_type_ids(
                captured["input_ids"].tolist()
            ),
            dtype=torch.long,
        )
        actual_tags = 1 - result["row_role"].cpu()
        rope, rope_equal = compare_rope(
            args.model_root, expected.model_inputs, captured
        )
        checks = {
            key: key in captured and torch.equal(value, captured[key])
            for key, value in expected.model_inputs.items()
        }
        checks["backend_rope"] = rope_equal
        checks["native_transformers_version"] = version("transformers") == "5.5.3"
        checks["field_set"] = set(expected.model_inputs) == set(captured)
        checks["h3_modality_tags"] = torch.equal(expected.modality_ids, actual_tags)
        checks["video_call_count"] = (
            len(native_capture.calls)
            == len(serving_capture.calls)
            == int(bool(native_capture.videos))
        )
        checks["sampling_disabled"] = all(
            call["do_sample_frames"] is False for call in serving_capture.calls
        )
        checks["padded_pixels"] = len(native_capture.videos) == len(
            serving_capture.videos
        ) and all(
            np.array_equal(left, right)
            for left, right in zip(native_capture.videos, serving_capture.videos)
        )
        torch.save(
            {
                "native": expected.model_inputs,
                "serving": captured,
                "native_tags": expected.modality_ids,
                "serving_tags": actual_tags,
                "rope": rope,
            },
            args.output / "preprocessing.pt",
        )
        np.savez(
            args.output / "padded_pixels.npz",
            **{
                f"{name}_{index}": video
                for name, capture in [
                    ("native", native_capture),
                    ("serving", serving_capture),
                ]
                for index, video in enumerate(capture.videos)
            },
        )
        report = {
            "native_revision": NATIVE_REVISION,
            "sglang_commit": os.environ.get("SGLANG_COMMIT"),
            "audit_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "component_file_sha256": {
                str(path.relative_to(args.model_root)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for folder in ("tokenizer", "processor")
                for path in sorted((args.model_root / folder).rglob("*"))
                if path.is_file()
            },
            "native_condition_sha256": source_hash,
            "request_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest(),
            "runtime_config_sha256": hashlib.sha256(
                args.runtime_config.read_bytes()
            ).hexdigest(),
            "model_root": str(args.model_root),
            "checks": checks,
            "passed": all(checks.values()),
            "feature_validation": "not_run",
            "backend_rope_validation": "passed" if rope_equal else "failed",
            "packages": {
                name: version(name)
                for name in ("torch", "transformers", "torchvision", "numpy")
            },
            "e2e_validation": "not_run",
            "mm_token_type_ids": "derived using checkpoint processor; backend consumes token IDs",
            "presentation": serving_processor.tokenizer.decode(
                captured["input_ids"][0]
            ),
            "context_rows": int(actual_tags.numel()),
            "video_processor_calls": serving_capture.calls,
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
