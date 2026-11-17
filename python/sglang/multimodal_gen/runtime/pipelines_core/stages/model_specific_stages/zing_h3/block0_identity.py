# SPDX-License-Identifier: Apache-2.0
"""Startup-only content identities for persistent Block0 preparations."""

import hashlib
import json
import math
import os
from pathlib import Path

SCHEMA = "zing-block0-v3"


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


def host_capacity(args):
    directory = getattr(args, "zing_block0_cache_capacity_dir", None)
    total = getattr(args, "zing_block0_cache_host_total_gb", None)
    total = None if total is None else budget_bytes(total, "host total")
    inherited = os.environ.get("SGLANG_CACHE_CAPACITY_DIR")
    if inherited is not None:
        planned = int(os.environ["SGLANG_CACHE_HOST_TOTAL_BYTES"])
        quota = int(os.environ["SGLANG_CACHE_INSTANCE_HOST_BYTES"])
        if (
            directory is not None
            and Path(directory).resolve() != Path(inherited).resolve()
        ):
            raise ValueError("Instance cannot bypass its machine capacity directory")
        if total is not None and total != planned:
            raise ValueError(
                "Instance machine cache budget differs from its launch plan"
            )
        if budget_bytes(args.zing_block0_cache_host_gb, "host cache") > quota:
            raise ValueError("Instance L1 exceeds its admitted launch plan")
        directory, total = inherited, planned
    if (directory is None) != (total is None):
        raise ValueError(
            "Host cache admission requires both capacity directory and machine total"
        )
    return directory, total


def startup_config(args):
    """Called only after pool admission; disabled caches need no disk/host args."""
    mode = getattr(args, "zing_block0_cache_mode", "on")
    if mode not in {"on", "off"}:
        raise ValueError("Zing Block0 cache mode must be on or off")
    if mode == "off":
        return None
    backend = getattr(args, "zing_block0_cache_backend", "local")
    if backend == "mooncake":
        path = getattr(args, "zing_block0_mooncake_config", None)
        if not path:
            raise ValueError("Mooncake Block0 caching requires a client config file")
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise ValueError("Mooncake client config must be a JSON object")
        return {"backend": backend, "config": config}
    if backend != "local":
        raise ValueError("Unknown Block0 cache backend")
    host = budget_bytes(args.zing_block0_cache_host_gb, "--zing-block0-cache-host-gb")
    disk = budget_bytes(args.zing_block0_cache_disk_gb, "--zing-block0-cache-disk-gb")
    root = args.zing_block0_cache_dir
    if not root or not Path(root).is_dir():
        raise ValueError(
            "--zing-block0-cache-dir must name an existing mounted directory"
        )
    # A missing mount must not be hidden by mkdir on a missing volume. The deployment
    # mounts and creates its cache root before starting the worker.
    if Path(root).resolve() == Path("/"):
        raise ValueError("The Block0 cache root cannot be the filesystem root")
    storage = getattr(args, "zing_block0_cache_storage", "mounted")
    if storage not in {"mounted", "local"}:
        raise ValueError("Block0 storage must be mounted or local")
    if storage == "mounted" and not storage_mounted(Path(root).resolve()):
        raise ValueError("Block0 storage must be mounted before the worker starts")
    capacity, total = host_capacity(args)
    if total is not None and total < host:
        raise ValueError("Instance L1 budget exceeds the machine cache budget")
    reserve = getattr(args, "zing_block0_cache_min_free_gb", 0)
    if (
        isinstance(reserve, bool)
        or not isinstance(reserve, (int, float))
        or not math.isfinite(reserve)
        or reserve < 0
    ):
        raise ValueError("Disk reserve must be finite and nonnegative")
    budget_bytes(getattr(args, "zing_block0_cache_inflight_gb", 128), "disk inflight")
    writers = getattr(args, "zing_block0_cache_max_writers", 2)
    if type(writers) is not int or not 1 <= writers <= 64:
        raise ValueError("Disk writer count must be in 1..64")
    return host, disk, Path(root).resolve()


def capacity_config(args):
    """SP ranks must agree on admission domains and policies before allocating."""
    directory, total = host_capacity(args)
    root = (
        getattr(args, "zing_block0_cache_budget_root", None)
        or args.zing_block0_cache_dir
    )
    return (
        None if directory is None else str(Path(directory).resolve()),
        total,
        str(Path(root).resolve()),
        getattr(args, "zing_block0_cache_min_free_gb", 0),
        getattr(args, "zing_block0_cache_inflight_gb", 128),
        getattr(args, "zing_block0_cache_max_writers", 2),
        getattr(args, "zing_block0_cache_gc_high_watermark", 1.0),
        getattr(args, "zing_block0_cache_gc_low_watermark", 0.9),
        getattr(args, "zing_block0_cache_gc_stale_s", 600.0),
    )


def storage_mounted(root):
    # Cache directories are usually below the volume mount, not mountpoints
    # themselves. Do not accept a stale cache directory on the root filesystem.
    return any(
        path != Path("/") and os.path.ismount(path) for path in (root, *root.parents)
    )


def execution_identity(model):
    """Resolved server recipe, independent of worker IDs and deployment paths.

    KV is BF16 even with FP8 weights. Its dtype alone cannot distinguish the
    preparations produced by different quantization or sparse-attention recipes.
    """
    from sglang.multimodal_gen.runtime.models.dits import zing_h3_sparse_attn as sparse

    quant = model.quant_config
    precision = {
        "weight_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "quantization": None,
    }
    if quant is not None:
        precision["quantization"] = {
            "mode": quant.get_name(),
            "selected_linears": sorted(quant.selected),
            "activation_scheme": quant.backend.activation_scheme,
            "weight_block_size": quant.backend.weight_block_size,
            "coverage": dict(quant.coverage),
        }
    mode = sparse.get_sparse_mode()
    attention = {"mode": mode}
    if mode != "dense":
        attention.update(
            min_chunk=sparse.get_min_sparse_chunk(),
            budget_tau=sparse.get_sparse_budget_tau(mode),
            three_tier=sparse.is_three_tier_enabled(mode),
            virtual_mean=sparse.is_virtual_mean_enabled(mode),
            triage_score=sparse.is_triage_score_enabled(mode),
            price_pi=sparse.get_sparse_price_pi(mode),
            c_rf=sparse.get_sparse_c_rf(mode),
            alpha_triage=sparse.get_sparse_alpha_triage(mode),
        )
    if sparse.is_three_tier_enabled(mode):
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_compressed_view import (
            LAYOUT_VERSION,
        )

        attention["three_tier"] = True
        attention["compressed_layout"] = LAYOUT_VERSION
    return {"precision": precision, "attention": attention}


def build_identity(stage, args, *, device):
    """Trust the immutable export's config; check small metadata across SP ranks."""
    from sglang.multimodal_gen.runtime.conditioning.worker import coordinate_validation
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_sp_group,
        get_ulysses_parallel_world_size,
    )

    identity = error = None
    try:
        with open(
            Path(stage.model_path) / "zing_config.json", encoding="utf-8"
        ) as handle:
            config = json.load(handle)
        config.pop("checkpoint", None)  # Export location is not model content.
        group = get_sp_group()
        stores = stage.kv_pool._pool._slots[0].kv_cache
        identity = {
            "cache_schema_version": SCHEMA,
            "zing_config": config,
            "sp": {
                "size": group.world_size,
                "ulysses": get_ulysses_parallel_world_size(),
                "ring": group.ring_world_size,
            },
            "kv_dtype": str(stores[0].k.dtype),
            "execution": execution_identity(stage.modules["transformer"]),
        }
        digest = fingerprint(identity)
    except Exception as exc:
        error = str(exc)
        digest = None
    values = coordinate_validation((error, digest))
    if any(reason is not None for reason, _ in values) or any(
        value != values[0] for value in values
    ):
        raise ValueError(
            f"Block0 config differs or is invalid across SP ranks: {values}"
        )
    return identity
