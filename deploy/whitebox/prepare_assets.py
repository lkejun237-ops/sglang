"""Prepare an isolated H3 serving view without changing the shared checkpoint."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

from config import fingerprint, validate
from output_paths import outside_checkout


def validate_decoder_config(value):
    # Execute the real CPU-only schema without importing the eager GPU pipeline
    # registry. Preparation and decoder_worker therefore share one contract.
    schema = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/multimodal_gen/configs/pipeline_configs/zing_h3_decoder.py"
    )
    normalize = runpy.run_path(str(schema))["normalize_decoder_spec"]
    spec = normalize(value, video_decoder_backend=None, vae_decode_precision="fp32")
    if spec.type != "lynn_streaming":
        raise ValueError("The four-GPU decoder service supports only lynn_streaming")
    return spec


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def prepare(
    *,
    source,
    decoder,
    output,
    bundle,
    config_sha,
    decoder_sha,
    decoder_type="lynn_streaming",
    decoder_model_dir=None,
    adaln_table=None,
):
    source, decoder, output = (Path(p).resolve() for p in (source, decoder, output))
    outside_checkout(output)
    policy = validate(bundle["policy"])
    if bundle["config_sha256"] != fingerprint(
        {"policy": policy, "bindings": bundle["bindings"]}
    ):
        raise ValueError("Release bundle fingerprint changed")
    m = policy["model"]
    if output == source or output.is_relative_to(source):
        raise ValueError("Serving output must be outside the source checkpoint")
    if output.exists():
        raise ValueError("Serving output already exists; use a new revision")
    if (
        digest(source / "zing_config.json") != config_sha
        or digest(decoder) != decoder_sha
    ):
        raise ValueError("Model configuration or decoder identity changed")

    if decoder_type != "lynn_streaming":
        raise ValueError("The four-GPU decoder service supports only lynn_streaming")
    asset_checks = {}
    config = {
        "zing_kv_pool_window_chunks": m["window_chunks"],
        "zing_kv_pool_sink_chunks": m["sink_chunks"],
        "zing_kv_pool_rope_position_mode": "block_relative",
        "zing_kv_pool_rope_max_frame_gap": 12,
        "zing_kv_pool_window_eviction_timing": "after_step0",
        "prompt_switch_prefix_latent_frames": 5,
        "zing_reference_max_images": 21,
        "decoder_config": {
            "type": decoder_type,
            "checkpoint_path": str(decoder),
            "precision": "bf16",
        },
    }
    if (m["adaln_mode"] == "table") != (adaln_table is not None):
        raise ValueError("AdaLN assets must match the shared release policy")
    if adaln_table is None:
        # Preserve this launcher's existing resident projection path explicitly;
        # No unprepared offline table may be selected by a runtime default.
        config["zing_adaln_mode"] = "resident"
    if decoder_model_dir is None:
        raise ValueError("Lynn requires a local decoder model directory")
    decoder_model_dir = Path(decoder_model_dir).resolve()
    architecture_path = decoder_model_dir / "config.json"
    architecture = json.loads(architecture_path.read_text())
    if architecture.get("clip_length") != 17 or architecture.get("token_drop") != 3:
        raise ValueError("Unsupported Lynn architecture")
    asset_checks[str(architecture_path)] = digest(architecture_path)
    config["decoder_config"]["model_dir"] = str(decoder_model_dir)
    validate_decoder_config(config["decoder_config"])
    if adaln_table is not None:
        if m["initial_blocks"]:
            raise ValueError("Mixed schedules do not support offline AdaLN tables")
        adaln_table = Path(adaln_table).resolve()
        manifest_path = adaln_table / "manifest.json"
        table_path = adaln_table / "table.safetensors"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "zing-adaln-table-v1":
            raise ValueError("Unsupported AdaLN table schema")
        table_sha = digest(table_path)
        if table_sha != manifest.get("table_sha256"):
            raise ValueError("AdaLN table checksum mismatch")
        transformer_config = source / "transformer/config.json"
        if manifest.get("identity", {}).get("config_sha256") != digest(
            transformer_config
        ) or manifest.get("provenance") != json.loads(
            (source / "zing_config.json").read_text()
        ):
            raise ValueError("AdaLN table belongs to a different model configuration")
        asset_checks[str(transformer_config)] = digest(transformer_config)
        asset_checks[str(manifest_path)] = digest(manifest_path)
        asset_checks[str(table_path)] = table_sha
        config["zing_adaln_table_path"] = str(adaln_table)
        # Runtime admission also verifies shard/provenance, arithmetic, complete
        # timestep batches and the actual scheduler. This is only host preflight.
    runtime = json.loads((source / "zing_config.json").read_text())
    if (
        runtime.get("num_frame_per_block") != [3, 2]
        or runtime.get("split_first_latent") is not False
        or runtime.get("checkpoint_sha256") != m["checkpoint_sha256"]
        or runtime.get("causal") is not True
        or runtime.get("h3_media_config", {}).get("fps") != 24
    ):
        raise ValueError("Expected a causal 24 fps export")
    for name in (
        "model_index.json",
        "transformer",
        "text_encoder",
        "processor",
        "tokenizer",
        "video_vae",
        "audio_vae",
    ):
        if not (source / name).exists():
            raise ValueError("The model export is incomplete: " + name)
    profile = {
        "solver": "few_step",
        "denoising_step_list": list(m["denoising_steps"]),
        "initial_denoising_step_list": list(m["initial_steps"]),
        "initial_block_count": m["initial_blocks"],
        "source_attention_window": {
            "window_chunks": m["window_chunks"],
            "sink_chunks": m["sink_chunks"],
        },
        "noise_version": "indexed-cpu-v1",
        "window_eviction_timing": "after_step0",
        "source_encoder": {
            "mode": "progressive",
            "contract_version": "h3-clip17-progressive-v1",
            "factory": "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder:create_session",
            "remote": {
                "port": 31001,
                "mode": "overlap" if m["source_work_quantum"] else "serial",
                "capacity_bytes": 268435456,
                "timeout_seconds": 120,
                "cross_dispatch_prefetch": bool(m["source_work_quantum"]),
            },
        },
        "async_source_output": bool(m["source_work_quantum"]),
        "model_config_sha256": config_sha,
        "guidance_scale": 1.0,
    }
    output.mkdir(parents=True, exist_ok=False)
    model = output / "model"
    model.mkdir()
    for item in source.iterdir():
        if item.name != "zing_realtime_profile.json":
            (model / item.name).symlink_to(item)
    (model / "zing_realtime_profile.json").write_text(
        json.dumps(profile, indent=2) + "\n"
    )
    (output / "pipeline.json").write_text(json.dumps(config, indent=2) + "\n")
    receipt = {
        "schema_version": 1,
        "model_source": str(source),
        "model_config_sha256": config_sha,
        "decoder_source": str(decoder),
        "decoder_sha256": decoder_sha,
        "decoder_type": decoder_type,
        "asset_checks": asset_checks,
        "serving_profile_sha256": digest(model / "zing_realtime_profile.json"),
        "decoder_config_sha256": digest(output / "pipeline.json"),
        "config_sha256": bundle["config_sha256"],
        "schema": "whitebox-prepared-assets-v1",
        "source_modified": False,
        "weight_verification": "shared read-only export; full weight hashes not asserted",
    }
    (output / "PREP_COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def verify_prepared(root, bundle):
    """Verify the prepared view and original decoder/table bytes before launch."""
    root = Path(root)
    receipt = json.loads((root / "PREP_COMPLETE.json").read_text())
    if (
        receipt.get("schema") != "whitebox-prepared-assets-v1"
        or receipt.get("config_sha256") != bundle["config_sha256"]
    ):
        raise ValueError("Prepared assets do not match the release bundle")
    checks = [
        (root / "model/zing_config.json", "model_config_sha256"),
        (Path(receipt["decoder_source"]), "decoder_sha256"),
        (root / "model/zing_realtime_profile.json", "serving_profile_sha256"),
        (root / "pipeline.json", "decoder_config_sha256"),
    ]
    if any(digest(path) != receipt[key] for path, key in checks):
        raise ValueError("Prepared serving identities changed")
    if any(
        digest(path) != expected for path, expected in receipt["asset_checks"].items()
    ):
        raise ValueError("Prepared decoder or AdaLN asset identities changed")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "source",
        "decoder",
        "output",
        "bundle",
        "config-sha",
        "decoder-sha",
    ):
        parser.add_argument("--" + key, required=True)
    parser.add_argument(
        "--decoder-type", choices=("lynn_streaming",), default="lynn_streaming"
    )
    parser.add_argument("--decoder-model-dir", required=True)
    parser.add_argument("--adaln-table")
    args = parser.parse_args()
    values = vars(args)
    values["bundle"] = json.loads(Path(args.bundle).read_text())
    receipt = prepare(**values)
    print(
        json.dumps(
            {
                "prepared": str(Path(args.output).resolve()),
                "profile_sha256": receipt["serving_profile_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
