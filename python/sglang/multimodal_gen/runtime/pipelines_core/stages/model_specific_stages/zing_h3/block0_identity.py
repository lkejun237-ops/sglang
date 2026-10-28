# SPDX-License-Identifier: Apache-2.0
"""Startup-only content identities for persistent Block0 preparations."""

import hashlib
import json
import math
import os
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

SCHEMA = "zing-block0-v2"


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def budget_bytes(value, name):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value <= 0
        or value * 10**9 < 1
    ):
        raise ValueError(f"{name} must be a finite positive decimal GB budget")
    return int(value * 10**9)


def startup_config(args):
    """Called only after pool admission; disabled caches need no disk/host args."""
    mode = getattr(args, "zing_block0_cache_mode", "on")
    if mode not in {"on", "off"}:
        raise ValueError("Zing Block0 cache mode must be on or off")
    if mode == "off":
        return None
    host = budget_bytes(args.zing_block0_cache_host_gb, "--zing-block0-cache-host-gb")
    disk = budget_bytes(args.zing_block0_cache_disk_gb, "--zing-block0-cache-disk-gb")
    root = args.zing_block0_cache_dir
    if not root or not Path(root).is_dir():
        raise ValueError(
            "--zing-block0-cache-dir must name an existing mounted directory"
        )
    # A missing mount must not be hidden by mkdir('/cpfs/...'). The deployment
    # mounts and creates its cache root before starting the worker.
    if Path(root).resolve() == Path("/"):
        raise ValueError("The Block0 cache root cannot be the filesystem root")
    if not storage_mounted(Path(root).resolve()):
        raise ValueError("Block0 storage must be mounted before the worker starts")
    return host, disk, Path(root).resolve()


def storage_mounted(root):
    # Cache directories are usually below the volume mount, not mountpoints
    # themselves. Do not accept a stale /cpfs directory on the root filesystem.
    return any(
        path != Path("/") and os.path.ismount(path) for path in (root, *root.parents)
    )


def _assets(path):
    """Reuse the conditioning inventory's complete-content hashing and symlink checks."""
    from sglang.multimodal_gen.runtime.conditioning.native import _component_identity

    path = Path(path)
    identity = _component_identity(path.parent, path.name)
    # Component labels and relative filenames are stable across mount locations.
    return {
        str(Path(k).relative_to(path.name)): v for k, v in identity["files"].items()
    }


def _scan_models(stage, args):
    from sglang.multimodal_gen.runtime.conditioning.contract import DeploymentProfile
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
        checkpoint_identity,
        file_digest,
    )

    root = Path(stage.model_path)

    def component(name):
        return Path(
            args.model_paths.get(name) or args.component_paths.get(name) or root / name
        )

    transformer = stage.modules["transformer"]
    table = getattr(transformer, "adaln_table", None)
    if table is not None:
        if not table.checkpoint_verified:
            raise ValueError("Block0 requires a verified AdaLN checkpoint identity")
        dit = table.manifest["identity"]
    else:
        # Resolve overrides with the same resolver used by the transformer loader.
        from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
            _server_args_for_transformer_component,
            resolve_transformer_checkpoint_files,
        )

        files = resolve_transformer_checkpoint_files(
            _server_args_for_transformer_component(args, "transformer", "transformer"),
            str(component("transformer")),
        )
        if not files.safetensors:
            raise ValueError(
                "Block0 requires identifiable transformer checkpoint shards"
            )
        dit = checkpoint_identity(
            files.safetensors,
            files.config_path or component("transformer") / "config.json",
        )
    if stage.conditioning_mode == "local":
        qwen = {
            name: _assets(component(name))
            for name in ("text_encoder", "tokenizer", "processor")
        }
    else:
        # The remote service verifies these assets at its own startup; the
        # authenticated handshake has already checked this expected profile.
        profile = DeploymentProfile.load(args.conditioning_expected_profile)
        qwen = {
            name: profile.document[name]
            for name in ("model", "projection", "tokenizer", "processor")
        }
    models = {"dit": dit, "qwen": qwen, "video_vae": _assets(component("video_vae"))}
    overrides = {}
    for name, value in getattr(args, "component_weights_paths", {}).items():
        if name in {"text_encoder", "video_vae"}:
            path = Path(value)
            overrides[name] = (
                _assets(path) if path.is_dir() else {"sha256": file_digest(path)}
            )
    if overrides:
        models["weight_overrides"] = overrides
    lora = getattr(args, "lora_path", None)
    if lora:
        path = Path(lora)
        models["lora"] = (
            _assets(path) if path.is_dir() else {"sha256": file_digest(path)}
        )
    return models


def build_identity(stage, args, *, device):
    """Scan Qwen/VAE once per SP group, reusing AdaLN's verified DiT digest.

    Scan failures are broadcast too, so the other ranks never wait forever for
    a successful rank-zero result. Models are immutable for this worker lifetime.
    """
    from sglang.multimodal_gen.runtime.conditioning.worker import (
        broadcast_receipt,
        coordinate_validation,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_sp_group,
    )

    group = get_sp_group()
    receipt = None
    if group.rank_in_group == 0:
        try:
            receipt = {"ok": True, "models": _scan_models(stage, args)}
        except Exception as exc:
            receipt = {"ok": False, "error": str(exc)}
    receipt = broadcast_receipt(receipt)
    if not receipt["ok"]:
        raise ValueError("Block0 asset scan failed: " + receipt["error"])

    local = error = None
    try:
        contract = _layout_contract(stage, args, device, group)
        table = getattr(stage.modules["transformer"], "adaln_table", None)
        local = {
            "layout": fingerprint(contract),
            "dit": None if table is None else table.manifest["identity"],
        }
    except Exception as exc:
        error = str(exc)
    values = coordinate_validation((error, local))
    if any(reason is not None for reason, _ in values) or any(
        value != values[0] for value in values
    ):
        raise ValueError(
            f"Block0 model/layout contract differs or is invalid across SP ranks: {values}"
        )
    return {"model": receipt["models"], "layout": contract}


def _layout_contract(stage, args, device, group):
    import torch

    from sglang.multimodal_gen.runtime.conditioning.native import _execution_options
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_tensor_model_parallel_world_size,
        get_ulysses_parallel_world_size,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
        arithmetic_contract,
        file_digest,
    )

    model = stage.modules["transformer"]
    table = getattr(model, "adaln_table", None)
    runtime = Path(__file__).parents[4]
    source_files = [
        *Path(__file__).parent.glob("*.py"),
        *(runtime / "models/dits").glob("zing_h3*.py"),
        *(runtime / "models/vaes/minimax_h3_video_vae").rglob("*.py"),
    ]
    software = {}
    for name in (
        "transformers",
        "triton",
        "safetensors",
        "einops",
        "diffusers",
        "flash-attn",
    ):
        try:
            software[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            software[name] = None
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.startswith(
            (
                "SGLANG_ZING_",
                "SGLANG_H3_",
                "MINIMAX_H3_",
                "SGLANG_DIFFUSION_MINIMAX_H3_",
            )
        )
        and not name.startswith(("SGLANG_ZING_BLOCK0_", "SGLANG_ZING_DISK_CACHE_"))
    }
    contract = {
        "schema": SCHEMA,
        "sp": group.world_size,
        "ulysses": get_ulysses_parallel_world_size(),
        "ring": group.ring_world_size,
        "tp": get_tensor_model_parallel_world_size(),
        "arch": asdict(stage.config.dit_config.arch_config),
        "vae_arch": asdict(stage.config.vae_config.arch_config),
        "dtype": str(next(model.parameters()).dtype),
        "qwen_execution": _execution_options()
        if stage.conditioning_mode == "local"
        else None,
        "precisions": {
            name: getattr(stage.config, name, None)
            for name in ("dit_precision", "vae_precision", "text_encoder_precision")
        },
        "arithmetic": arithmetic_contract(device),
        "software": software,
        "cudnn": {
            "version": torch.backends.cudnn.version(),
            "benchmark": torch.backends.cudnn.benchmark,
            "deterministic": torch.backends.cudnn.deterministic,
            "tf32": torch.backends.cudnn.allow_tf32,
        },
        "adaln": None
        if table is None
        else {
            name: table.manifest[name]
            for name in ("table_sha256", "layout", "arithmetic", "schedule")
        },
        "runtime": {
            name: stage.runtime[name]
            for name in (
                "vae_config",
                "rope_config",
                "condition_encoder_config",
                "h3_media_config",
                "num_frame_per_block",
                "split_first_latent",
            )
        },
        "environment": environment,
        "quantization": {
            "global": args.quantization,
            "components": args.component_quantizations,
            "ignored_layers": args.quantization_ignored_layers,
            "component_ignored_layers": args.component_quantization_ignored_layers,
        },
        "component_precisions": args.component_precisions,
        "lora": {
            name: getattr(args, name)
            for name in (
                "lora_scale",
                "lora_alpha",
                "lora_merge_mode",
                "lora_weight_name",
            )
        },
        "execution": {
            "compile": args.enable_torch_compile,
            "cuda_graph": args.enable_cuda_graph,
            "attention_backend": str(args.attention_backend),
            "attention_backend_config": args.attention_backend_config,
            "component_attention_backends": args.component_attention_backends,
        },
        "implementation": {
            str(path.relative_to(runtime)): file_digest(path)
            for path in sorted(source_files)
        },
    }
    if stage.conditioning_mode != "local":
        contract["conditioning_profile"] = stage.conditioning_provider.profile_id
    return contract
