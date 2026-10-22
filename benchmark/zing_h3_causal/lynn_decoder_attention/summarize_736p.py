# SPDX-License-Identifier: Apache-2.0
"""Validate full-model 736p service measurements with common component offload."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from summarize_e2e import summarize_service, validate_startup


def summarize(archive, startup_archive):
    assert archive["counts"] == {"720p": {"complete": 6, "failed": False}}
    files, previous = archive["files"], startup_archive["files"]
    assert files["READY.json"]["passed"] and files["TRANSPORT_READY.json"]["passed"]
    assert files["READY.json"]["measured_requests_per_server"] == 1
    assert files["REUSED_REGRESSIONS.json"]["source_hashes_identical"]
    source_files = {
        name: digest
        for name, digest in files["READY.json"]["files"].items()
        if name.startswith("python/")
    }
    previous_sources = {
        name: digest
        for name, digest in previous["READY.json"]["files"].items()
        if name.startswith("python/")
    }
    assert previous_sources and all(
        source_files[name] == digest for name, digest in previous_sources.items()
    )
    common = files["READY.json"]["common_files"]
    assert set(source_files) == set(previous_sources) | set(common)
    assert all(source_files[name] == digest for name, digest in common.items())
    assert files["graph-gpu-unit.xml.json"] == dict(
        tests=3, errors=0, failures=0, skipped=0
    )
    assert files["gpu-unit.xml.json"] == dict(tests=19, errors=0, failures=0, skipped=0)
    assert files["READY.json"]["baseline_sha"] == previous["READY.json"]["baseline_sha"]
    assert len(files["hardware.json"]) == 2
    assert files["hardware.json"] == previous["hardware.json"]
    capture_memory = []
    for rep in range(3):
        for variant in ("baseline", "candidate"):
            prefix = f"e2e/720p/rep{rep}/{variant}/"
            assert files[prefix + "model-architecture.json"]["num_layers"] == 50
            command = files[prefix + "command.json"]
            for flag, value in {
                "--text-encoder-cpu-offload": "true",
                "--vae-cpu-offload": "true",
                "--dit-cpu-offload": "false",
                "--dit-layerwise-offload": "false",
                "--use-fsdp-inference": "false",
                "--num-gpus": "2",
                "--sp-degree": "2",
                "--ulysses-degree": "2",
                "--ring-degree": "1",
                "--zing-denoising-step-list": "1000,750,500,250",
            }.items():
                assert command[command.index(flag) + 1] == value
            pipeline = files[prefix + "pipeline.json"]
            assert pipeline["decoder_config"]["precision"] == "bf16"
            for key, value in dict(
                size=1,
                max_target_area=1312 * 736,
                max_target_long_edge=1312,
                max_prefix_rows=16384,
                window_chunks=21,
                sink_chunks=3,
            ).items():
                assert pipeline["zing_kv_pool_" + key] == value
            cell = files[prefix + "COMPLETE.json"]
            assert cell["text_encoder_cpu_offload"] and cell["vae_cpu_offload"]
            assert cell["kv_pool"]
            for pool in cell["kv_pool"]:
                assert pool["slot_rows"] == 116940
                assert pool["pool_allocated_bytes"] == 125733888000
                assert pool["prefix_workspace_bytes"] == 17616076800
                assert pool["dtype"] == "torch.bfloat16"
            snapshot = files[prefix + "CAPTURE_SNAPSHOTS.json"]
            captures = snapshot["captures"]
            assert captures and all(
                0 < row["snapshot_bytes"] < row["store_bytes"] == 83822592000
                for row in captures
            )
            capture_memory.append(dict(rep=rep, variant=variant, **snapshot))
    report = summarize_service(files, prefix="e2e/720p", measured_requests=1)
    report["scope"] = (
        "Full-model 1312x736/SP2/TP1, common text encoder and VAE component CPU "
        "offload; 3 independent server pairs, 1 measured 720-frame request per "
        "server; warmup excluded."
    )
    validate_startup(previous)
    report["startup_proof_reused"] = True
    report["hardware"] = files["hardware.json"]
    report["input_sha256"] = files["e2e/720p/rep0/baseline/COMPLETE.json"]["requests"][
        1
    ]["summary"]["input_sha256"]
    report["runtime_source_files"] = source_files
    report["common_capture_fix_files"] = common
    report["graph_gpu_unit_result"] = files["graph-gpu-unit.xml.json"]
    report["preparation_gpu_unit_result"] = files["gpu-unit.xml.json"]
    report["kv_pool_budget"] = files["KV_PLAN.json"]["736p"]
    report["capture_memory"] = capture_memory
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("startup_archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.archive, "rt") as stream:
        archive = json.load(stream)
    with gzip.open(args.startup_archive, "rt") as stream:
        startup_archive = json.load(stream)
    report = summarize(archive, startup_archive)
    report["readback_sha256"] = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    report["startup_readback_sha256"] = hashlib.sha256(
        args.startup_archive.read_bytes()
    ).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
