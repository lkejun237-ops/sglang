# SPDX-License-Identifier: Apache-2.0
"""Export a merged Zing generator and pinned H3 components for SGLang.

The output directory must not exist. Weights remain in the training checkpoint's
separate-QKV layout; the native loader fuses QKV and swaps SwiGLU value/gate.
"""

import argparse
import hashlib
import json
import math
import re
import warnings
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

ADALN_MODALITIES = ("video", "text", "audio", "source_visual")
CONVERSION_VERSION = "split-adaln-concat-v1"


def redact_config_secrets(value, path=()):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            secret = key.lower() in {
                "api_key",
                "access_key_id",
                "access_key_secret",
                "secret_access_key",
                "password",
                "token",
                "authorization",
            } or (key.lower() == "key" and "wandb_config" in path)
            result[key] = (
                "[REDACTED]" if secret else redact_config_secrets(item, path + (key,))
            )
        return result
    if isinstance(value, list):
        return [redact_config_secrets(item, path) for item in value]
    return value


def adaln_modalities(arch):
    """Projection names in native tag order; condition may share video."""
    tags = arch["modality_tags"]
    required = set(ADALN_MODALITIES)
    if (
        not isinstance(tags, dict)
        or not required <= tags.keys() <= required | {"condition_visual"}
        or any(type(value) is not int for value in tags.values())
    ):
        raise ValueError("Unsupported Zing modality routing")
    modalities = ADALN_MODALITIES
    if tags.get("condition_visual", tags["video"]) != tags["video"]:
        modalities += ("condition_visual",)
    if sorted(tags[name] for name in modalities) != list(range(len(modalities))):
        raise ValueError("Zing projection tags must be unique and contiguous")
    count = arch.get("num_modalities", len(modalities))
    if type(count) is not int or count != len(modalities):
        raise ValueError("Zing modality count conflicts with routing")
    return tuple(sorted(modalities, key=tags.__getitem__))


def condition_timesteps(cfg, input_layout):
    values = (
        cfg.get("ar_condition_timestep", 1.0),
        cfg.get("reference_visual_timestep", 0.999 if input_layout == "split" else 1.0),
    )
    if any(
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in values
    ):
        raise ValueError("Zing condition timesteps must be finite and within [0, 1]")
    return values


def expected_shapes(arch, adaln_layout="single"):
    if adaln_layout not in {"single", "split"}:
        raise ValueError("AdaLN layout must be single or split")
    h, inner, ff = arch["dim"], arch["num_heads"] * arch["head_dim"], arch["ffn_dim"]
    patch = arch["in_dim"] * math.prod(arch["patch_size"])
    shapes = {}

    def linear(name, output, input, bias=True):
        shapes["backbone." + name + ".weight"] = (output, input)
        if bias:
            shapes["backbone." + name + ".bias"] = (output,)

    def norm(name, size):
        shapes["backbone." + name + ".weight"] = (size,)

    linear("proj_in", h, patch)
    linear("audio_proj_in", h, arch["audio_in_dim"])
    linear("context_embedder", h, arch["text_dim"])
    linear("time_embedder.linear_1", arch["time_embed_hidden_dim"], arch["freq_dim"])
    linear(
        "time_embedder.linear_2", arch["time_embed_dim"], arch["time_embed_hidden_dim"]
    )
    norm("token_refiner.final_norm", h)
    norm("norm_out.norm", h)
    linear("norm_out.linear", 2 * h, arch["time_embed_dim"])
    linear("proj_out", patch, h)
    linear("audio_proj_out", arch["audio_in_dim"], h)
    for family, count in (
        ("transformer_blocks", arch["num_layers"]),
        ("token_refiner.refiner_blocks", arch["num_refiner_layers"]),
    ):
        for i in range(count):
            prefix = f"{family}.{i}."
            norm(prefix + "norm1", h)
            norm(prefix + "norm2", h)
            for part in ("to_q", "to_k", "to_v"):
                linear(prefix + "attn." + part, inner, h, False)
            linear(prefix + "attn.to_out.0", h, inner, False)
            norm(prefix + "attn.norm_q", arch["head_dim"])
            norm(prefix + "attn.norm_k", arch["head_dim"])
            linear(prefix + "ff.net.0.proj", 2 * ff, h, False)
            linear(prefix + "ff.net.2", h, ff, False)
            if family == "transformer_blocks":
                if adaln_layout == "single":
                    linear(
                        prefix + "adaln_proj.linear",
                        6 * len(adaln_modalities(arch)) * h,
                        arch["time_embed_dim"],
                    )
                else:
                    for modality in adaln_modalities(arch):
                        linear(
                            prefix + f"adaln_proj.{modality}_linear",
                            6 * h,
                            arch["time_embed_dim"],
                        )
    return shapes


def normalize_adaln_checkpoint(state, arch):
    """Validate the entire merged checkpoint before converting any tensor.

    Official three-modality initialization belongs to native Zing. In particular,
    never replace a trained source branch with video or merge LoRA here.
    """
    modalities = adaln_modalities(arch)
    split = any(".adaln_proj." in key and "_linear." in key for key in state)
    layout = "split" if split else "single"
    shapes = expected_shapes(arch, layout)
    if state.keys() != shapes.keys():
        raise ValueError(
            f"Incomplete or unexpected Zing checkpoint keys ({layout}): "
            f"missing={sorted(shapes.keys() - state.keys())[:8]}, "
            f"unexpected={sorted(state.keys() - shapes.keys())[:8]}"
        )
    for key, shape in shapes.items():
        value = state[key]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or not value.is_floating_point()
        ):
            raise ValueError(f"Invalid Zing checkpoint tensor: {key}; expected {shape}")
    result = dict(state)
    if split:
        for i in range(arch["num_layers"]):
            prefix = f"backbone.transformer_blocks.{i}.adaln_proj."
            for suffix in ("weight", "bias"):
                parts = [
                    result.pop(prefix + m + "_linear." + suffix) for m in modalities
                ]
                if len({part.dtype for part in parts}) != 1:
                    raise ValueError("Split AdaLN branches must have the same dtype")
                result[prefix + "linear." + suffix] = torch.cat(parts, dim=0)
    return result, layout


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(
    checkpoint,
    config_path,
    components,
    output,
    shard_bytes=4 * 1024**3,
    video_vae_source=None,
    provenance=None,
):
    checkpoint, config_path, components, output = map(
        Path, (checkpoint, config_path, components, output)
    )
    if checkpoint.is_dir():
        raise ValueError(
            "Zing training checkpoints use DCP directories. Export the selected "
            "generator or EMA component to model.pt with Zing's "
            "src/scripts/export_model_pt.py, then pass that file to --checkpoint."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing merged Zing checkpoint: {checkpoint}")
    cfg = yaml.safe_load(config_path.read_text())
    arch = cfg["generator_config"]
    if cfg["causal"] is not True:
        raise ValueError("Export requires a causal checkpoint")
    modalities = adaln_modalities(arch)
    if (
        arch["rope_config"]["theta"] != 10000
        or arch["local_attn_size"] != -1
        or arch["sink_size"] != 0
    ):
        raise ValueError("Export requires absolute RoPE, unlimited history and no sink")
    if cfg["condition_encoder_config"]["output_hidden_layer"] != 50:
        raise ValueError("The shared H3 conditioner requires hidden_states[50]")
    if cfg["vae_config"]["encode_mode"] not in ("deterministic", "performance"):
        raise ValueError(
            "Unsupported Zing visual encode_mode; expected deterministic or performance"
        )
    component_sources = {
        name: components / name
        for name in ("text_encoder", "tokenizer", "processor", "audio_vae")
    }
    component_sources["video_vae"] = components / (
        "video_vae" if (components / "video_vae").is_dir() else "vae"
    )
    for name, path in component_sources.items():
        if not path.is_dir():
            raise ValueError(f"Missing shared H3 component: {name}: {path}")
    video_vae_source = (
        Path(video_vae_source)
        if video_vae_source is not None
        else component_sources["video_vae"] / "source" / "model.safetensors"
    )
    if not video_vae_source.is_file():
        raise ValueError(
            "Missing SGLang H3 video VAE source/model.safetensors; provide "
            "--video-vae-source from the same pinned H3 component revision"
        )
    if output.exists():
        raise FileExistsError(f"Export destination already exists: {output}")
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if (
        not isinstance(state, dict)
        or not state
        or any(
            not isinstance(v, torch.Tensor) or not k.startswith("backbone.")
            for k, v in state.items()
        )
    ):
        raise ValueError(
            "Expected a merged generator state dict with backbone.* tensor keys"
        )
    state, input_layout = normalize_adaln_checkpoint(state, arch)
    provenance = {} if provenance is None else dict(provenance)
    if (
        input_layout == "split"
        or provenance
        or cfg.get("lora_config", {}).get("enabled")
    ):
        if (
            not re.fullmatch(r"[0-9a-f]{40}", provenance.get("native_sha", ""))
            or provenance.get("checkpoint_source") not in {"generator", "ema"}
            or not provenance.get("component_versions")
            or (
                cfg.get("lora_config", {}).get("enabled")
                and provenance.get("lora_merged") is not True
            )
        ):
            raise ValueError(
                "Export provenance requires native_sha, explicit generator/ema checkpoint_source, component_versions, and lora_merged=true for LoRA"
            )
    else:
        warnings.warn(
            "Legacy single-linear export has no native/component provenance",
            stacklevel=2,
        )
    condition_t, reference_t = condition_timesteps(cfg, input_layout)
    output.mkdir(parents=True)
    transformer = output / "transformer"
    transformer.mkdir()
    names = {
        "dim": "hidden_size",
        "num_heads": "num_attention_heads",
        "head_dim": "attention_head_dim",
        "num_layers": "num_layers",
        "num_refiner_layers": "token_refiner_num_layers",
        "ffn_dim": "ffn_hidden_size",
        "in_dim": "latents_dim",
        "audio_in_dim": "audio_latents_dim",
        "text_dim": "text_dim",
        "freq_dim": "timestep_input_dim",
        "time_embed_hidden_dim": "time_embed_hidden_size",
        "time_embed_dim": "time_embed_dim",
        "patch_size": "patch_size",
        "eps": "norm_eps",
        "qk_norm_eps": "qk_norm_eps",
        "final_norm_eps": "final_norm_eps",
    }
    model_config = {dst: arch[src] for src, dst in names.items()}
    model_config.update(
        _class_name="ZingH3DiTModel",
        num_modalities=len(modalities),
        modality_tags=arch["modality_tags"],
        adaln_out_features=6 * len(modalities) * arch["dim"],
        final_adaln_out_features=2 * arch["dim"],
        rope_inv_freq_len=arch["rope_config"]["freq_dim"],
        checkpoint_uses_diffusers_layout=True,
        ar_condition_timestep=condition_t,
        reference_visual_timestep=reference_t,
    )
    (transformer / "config.json").write_text(json.dumps(model_config, indent=2) + "\n")
    fp32_prefixes = (
        "backbone.proj_in.",
        "backbone.audio_proj_in.",
        "backbone.time_embedder.",
        "backbone.proj_out.",
        "backbone.audio_proj_out.",
    )
    weight_map, shard, size, total = {}, {}, 0, 0

    def flush():
        nonlocal shard, size
        if not shard:
            return
        name = f"diffusion_pytorch_model-{len(set(weight_map.values())) + 1:05d}.safetensors"
        save_file(shard, str(transformer / name))
        weight_map.update({key: name for key in shard})
        shard, size = {}, 0

    for key, value in state.items():
        dtype = torch.float32 if key.startswith(fp32_prefixes) else torch.bfloat16
        value = value.detach().to(dtype).contiguous()
        count = value.numel() * value.element_size()
        if shard and size + count > shard_bytes:
            flush()
        shard[key] = value
        size += count
        total += count
    flush()
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2
        )
        + "\n"
    )
    for name, path in component_sources.items():
        bundled_source = path / "source" / "model.safetensors"
        if (
            name != "video_vae"
            or video_vae_source.resolve() == bundled_source.resolve()
        ):
            (output / name).symlink_to(path.resolve(), target_is_directory=True)
            continue
        # Some training snapshots only retain the official Diffusers VAE
        # shards. Build a read-only component view and pin SGLang's equivalent
        # fused-QKV source weights without modifying the shared snapshot.
        destination = output / name
        destination.mkdir()
        for child in path.iterdir():
            if child.name == "source":
                continue
            (destination / child.name).symlink_to(
                child.resolve(), target_is_directory=child.is_dir()
            )
        source = destination / "source"
        source.mkdir()
        (source / "model.safetensors").symlink_to(video_vae_source.resolve())
    runtime = {
        name: cfg[name]
        for name in (
            "causal",
            "num_frame_per_block",
            "split_first_latent",
            "num_train_timestep",
            "timestep_shift",
            "audio_timestep_shift",
            "h3_media_config",
            "condition_encoder_config",
            "vae_config",
        )
    }
    runtime.update(
        format_version=2,
        conversion_version=CONVERSION_VERSION,
        input_adaln_layout=input_layout,
        runtime_adaln_layout="single",
        adaln_modality_order=list(modalities),
        ar_condition_timestep=condition_t,
        reference_visual_timestep=reference_t,
        provenance=provenance,
        checkpoint_sha256=file_sha256(checkpoint),
        training_effective_config=redact_config_secrets(cfg),
        rope_config=arch["rope_config"],
        local_attn_size=arch["local_attn_size"],
        sink_size=arch["sink_size"],
        checkpoint=str(Path(checkpoint).resolve()),
        training_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
    )
    (output / "zing_config.json").write_text(json.dumps(runtime, indent=2) + "\n")
    index = {
        "_class_name": "ZingH3CausalPipeline",
        "_diffusers_version": "0.37.0",
        "transformer": ["diffusers", "ZingH3DiTModel"],
        "video_vae": ["diffusers", "MiniMaxH3VideoVAE"],
        "audio_vae": ["diffusers", "MiniMaxH3AudioVAE"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "tokenizer": ["transformers", "Qwen2TokenizerFast"],
        "processor": ["transformers", "Qwen3VLProcessor"],
    }
    # Publish identity last: an interrupted export cannot be selected as a pipeline.
    (output / "model_index.json").write_text(json.dumps(index, indent=2) + "\n")
    return {
        "output": str(output.resolve()),
        "tensors": len(state),
        "weight_bytes": total,
        "shards": len(set(weight_map.values())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "components", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--video-vae-source",
        help=(
            "SGLang-compatible H3 video VAE source/model.safetensors from the "
            "same component revision; required when --components lacks it"
        ),
    )
    parser.add_argument(
        "--provenance",
        help="JSON with native_sha, checkpoint_source, component_versions and lora_merged",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                args.checkpoint,
                args.config,
                args.components,
                args.output,
                video_vae_source=args.video_vae_source,
                provenance=json.loads(Path(args.provenance).read_text())
                if args.provenance
                else None,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
