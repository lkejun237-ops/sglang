# SPDX-License-Identifier: Apache-2.0
"""Strict loader for the delivered Lynn StreamingDecoder checkpoint."""

import json
import math
from pathlib import Path

import torch

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3_decoder import DecoderSpec

from .attention import ATTENTION_BACKENDS
from .model import LynnDecoderConfig, StreamingDecoder

# Values the delivered checkpoint was trained and accepted with; anything else is unvalidated.
_EXPECTED = {
    "temporal_window": 3,
    "tile_size": 256,
    "tile_overlap": 64,
    "tile_batch_size": 0,
}
_EXPECTED_ARCH = {"clip_length": 17, "token_drop": 3}
_PRECISION_DTYPES = {"bf16": "bfloat16"}


def read_architecture(model_dir) -> dict:
    architecture = json.loads((Path(model_dir) / "config.json").read_text())
    channels = architecture["latent_channels"]
    for name in ("latents_mean", "latents_std"):
        values = architecture[name]
        if len(values) != channels or not all(math.isfinite(v) for v in values):
            raise ValueError(f"config.json {name} must be {channels} finite values")
    if not all(v > 0 for v in architecture["latents_std"]):
        raise ValueError("config.json latents_std values must be greater than zero")
    for name, expected in _EXPECTED_ARCH.items():
        if architecture[name] != expected:
            raise ValueError(
                f"Unsupported lynn_streaming {name}: {architecture[name]!r}"
            )
    return architecture


def _decoder_config(section: dict) -> LynnDecoderConfig:
    if section.get("first_latent_pad_config", {}).get("enable", False):
        raise ValueError(
            "lynn_streaming checkpoints with first_latent_pad are not supported yet"
        )
    if section["attention_backend"] not in ATTENTION_BACKENDS:
        raise ValueError(
            f"Unknown lynn_streaming attention backend: {section['attention_backend']!r}"
        )
    for name, expected in _EXPECTED.items():
        if section[name] != expected:
            raise ValueError(f"Unsupported lynn_streaming {name}: {section[name]!r}")
    if len(section["pixel_mean"]) != 3 or len(section["pixel_std"]) != 3:
        raise ValueError("lynn_streaming pixel mean/std must have 3 values")
    return LynnDecoderConfig(
        temporal_window=section["temporal_window"],
        attention_backend=section["attention_backend"],
        tile_size=section["tile_size"],
        tile_overlap=section["tile_overlap"],
        tile_batch_size=section["tile_batch_size"],
        pixel_mean=tuple(section["pixel_mean"]),
        pixel_std=tuple(section["pixel_std"]),
    )


def select_weights(spec: DecoderSpec, checkpoint_section: dict) -> str:
    if spec.weights is not None:
        return spec.weights
    return "ema" if checkpoint_section["export_ema"] else "student"


def load_lynn_decoder(spec: DecoderSpec, *, device) -> StreamingDecoder:
    architecture = read_architecture(spec.model_dir)
    # Trusted CPFS checkpoint; mmap avoids reading the optimizer state into host memory.
    payload = torch.load(
        spec.checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    config = payload["config"]
    # The reference decodes under autocast(training_config.dtype); a different choice is unvalidated.
    trained_dtype = config["training_config"]["dtype"]
    if trained_dtype != _PRECISION_DTYPES[spec.precision]:
        raise ValueError(
            f"lynn_streaming precision {spec.precision!r} does not match the checkpoint compute dtype {trained_dtype!r}"
        )
    dtype = getattr(torch, trained_dtype)
    decoder_config = _decoder_config(config["decoder_config"])
    key = select_weights(spec, config["checkpoint_config"])
    model = StreamingDecoder(decoder_config, architecture)
    model.load_state_dict(payload[key], strict=True)
    del payload
    model.to_compute(device, dtype).eval().requires_grad_(False)
    return model
