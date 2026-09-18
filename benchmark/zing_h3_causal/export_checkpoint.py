# SPDX-License-Identifier: Apache-2.0
"""Export a merged Zing generator and pinned H3 components for SGLang.

The output directory must not exist. Weights remain in the training checkpoint's
separate-QKV layout; the native loader fuses QKV and swaps SwiGLU value/gate.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file


def expected_shapes(arch):
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
                linear(prefix + "adaln_proj.linear", 24 * h, arch["time_embed_dim"])
    return shapes


def export(
    checkpoint,
    config_path,
    components,
    output,
    shard_bytes=4 * 1024**3,
    video_vae_source=None,
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
    if cfg["causal"] is not True or arch["num_modalities"] != 4:
        raise ValueError("Export requires a four-modality causal checkpoint")
    if arch["modality_tags"] != {"video": 0, "text": 1, "audio": 2, "source_visual": 3}:
        raise ValueError("Unsupported Zing modality routing")
    if (
        arch["rope_config"]["theta"] != 10000
        or arch["local_attn_size"] != -1
        or arch["sink_size"] != 0
    ):
        raise ValueError("Export requires absolute RoPE, unlimited history and no sink")
    if cfg["condition_encoder_config"]["output_hidden_layer"] != 50:
        raise ValueError("The shared H3 conditioner requires hidden_states[50]")
    if cfg["vae_config"]["encode_mode"] != "deterministic":
        raise ValueError("Zing dense export requires deterministic visual encoding")
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
    shapes = expected_shapes(arch)
    if state.keys() != shapes.keys():
        raise ValueError(
            f"Incomplete or unexpected Zing checkpoint keys: missing={sorted(shapes.keys() - state.keys())[:8]}, unexpected={sorted(state.keys() - shapes.keys())[:8]}"
        )
    for key, shape in shapes.items():
        if tuple(state[key].shape) != shape or not state[key].is_floating_point():
            raise ValueError(
                f"Invalid Zing checkpoint tensor: {key}; expected {shape}, got {tuple(state[key].shape)} {state[key].dtype}"
            )
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
        num_modalities=4,
        adaln_out_features=24 * arch["dim"],
        final_adaln_out_features=2 * arch["dim"],
        rope_inv_freq_len=arch["rope_config"]["freq_dim"],
        checkpoint_uses_diffusers_layout=True,
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
        format_version=1,
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
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                args.checkpoint,
                args.config,
                args.components,
                args.output,
                video_vae_source=args.video_vae_source,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
