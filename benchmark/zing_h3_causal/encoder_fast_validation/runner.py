# SPDX-License-Identifier: Apache-2.0
"""Compare fast encoder off/on with complete frozen product requests.

Inputs and runtime source are supplied separately; this file only orchestrates
existing production worker entrypoints and the standard continuous-source client.
"""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_config(root):
    config = json.loads((root / "RUN_CONFIG.json").read_text())
    assert config["dit_sp_degree"] in (1, 2)
    assert config["compute_capability"] in ([9, 0], [10, 3])
    assert config["resolutions"] and all(
        size in ([832, 480], [1280, 736]) for size in config["resolutions"]
    )
    return config


def prepare(root, output, revision):
    from bench_streaming_source import prepare_case
    from case_sets import load_case_set

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        initialize_runtime_imports,
    )

    config_run = run_config(root)
    sp = config_run["dit_sp_degree"]
    initialize_runtime_imports()
    from importlib.metadata import version

    from sglang.multimodal_gen.runtime.realtime.source_transport import (
        SourceTransportBudget,
    )

    SourceTransportBudget().websocket_options()
    versions = {
        name: version(name) for name in ["uvicorn", "websockets", "torch", "triton"]
    }
    from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
        default_source_encoder,
    )

    profile_path = root / "model/zing_realtime_profile.json"
    profile = json.loads(profile_path.read_text())
    write(root / "original-model-profile.json", profile)
    profile["source_encoder"] = default_source_encoder()
    profile.pop("source_decoder", None)
    write(profile_path, profile)
    config = json.loads((root / "model/zing_config.json").read_text())
    assert (
        config["checkpoint_sha256"]
        == "89c3a67d77214e57d810e6a4e5554c7c919ae929db0174e6439dfadb53a1d731"
    )
    assert config["split_first_latent"] is False
    assert config["num_frame_per_block"] == [3, 2]
    pipeline = {
        "zing_kv_cache_dtype": "bfloat16",
        "zing_kv_pool_window_chunks": 6,
        "zing_kv_pool_sink_chunks": 3,
        "zing_kv_pool_rope_position_mode": "block_relative",
        "zing_kv_pool_rope_max_frame_gap": 12,
        "zing_kv_pool_window_eviction_timing": "after_step0",
        "zing_adaln_mode": "resident",
        "decoder_config": {
            "type": "lynn_streaming",
            "checkpoint_path": str(root / "lynn/decoder.pt"),
            "model_dir": str(root / "lynn"),
            "precision": "bf16",
            "weights": "ema",
        },
    }
    write(root / "pipeline.json", pipeline)
    cases = []
    for member in load_case_set("product-case-20261103")["members"]:
        case = prepare_case(
            member["case_id"],
            root / "media",
            root / "preflight" / member["case_id"],
            manifest=member["manifest"],
            case_id=member["case_id"],
        )
        cases.append(case)
    for case in cases:
        # The frozen collection owns source/output geometry, frame counts and updates.
        print("PREPARED_CASE", case["case_id"], case["frames"], flush=True)
    write(root / "CASES.json", cases)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=size=32x32:rate=1",
            "-frames:v",
            "1",
            "-c:v",
            "libx264",
            "-y",
            str(root / "codec-check.mp4"),
        ],
        check=True,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    ServerArgs.from_kwargs(
        model_path=str(root / "model"),
        pipeline_config_path=str(root / "pipeline.json"),
        num_gpus=sp,
        sp_degree=sp,
        ulysses_degree=sp,
        ring_degree=1,
        mem_fraction_static=0.85,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        performance_mode="speed",
        warmup_mode="off",
        realtime_max_sessions=1,
        realtime_max_sessions_per_worker=1,
    )
    controls = {
        str(p.relative_to(root)): digest(p)
        for folder in [root / "source", root / "harness", root / "transport-deps"]
        for p in folder.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    controls.update(
        {
            n: digest(root / n)
            for n in [
                "RUN_CONFIG.json",
                "pipeline.json",
                "CASES.json",
                "model/zing_config.json",
                "model/model_index.json",
                "model/zing_realtime_profile.json",
                "original-model-profile.json",
            ]
        }
    )
    receipt = {
        "passed": True,
        "packages": versions,
        "encoder_session_max_lifetime_s": 7200,
        "runtime_revision": revision,
        "controls": controls,
        "checkpoint_sha256": config["checkpoint_sha256"],
        "model_config_sha256": digest(root / "model/zing_config.json"),
        "cases": len(cases),
        "source_resize": "ffmpeg scale=W:H:flags=bilinear; full original frames and frozen assets",
        "model": str((root / "model").resolve()),
    }
    write(root / "CPU_READY.json", receipt)
    write(output / "CPU_READY.json", receipt)
    print("CPU_READY", receipt["cases"], flush=True)


async def measure_variant(root, output, fast_flag):
    import torch
    from benchmark_artifacts import ExperimentRetention
    from worker_group_master import LocalWorkers, WorkerGroupMaster

    ready = json.loads((root / "CPU_READY.json").read_text())
    config_run = run_config(root)
    sp = config_run["dit_sp_degree"]
    assert ready["passed"]
    for relative, expected in ready["controls"].items():
        assert digest(root / relative) == expected, relative
    assert digest(root / "model/zing_config.json") == ready["model_config_sha256"]
    hardware = [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "uuid": str(torch.cuda.get_device_properties(i).uuid),
            "cc": list(torch.cuda.get_device_capability(i)),
            "bytes": torch.cuda.get_device_properties(i).total_memory,
        }
        for i in range(torch.cuda.device_count())
    ]
    assert len(hardware) == sp + 2 and all(
        g["cc"] == config_run["compute_capability"]
        and g["bytes"] >= config_run["min_gpu_memory_bytes"]
        for g in hardware
    )
    write(
        output / "HARDWARE.json",
        {
            "devices": hardware,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "run_config": config_run,
        },
    )
    owned = LocalWorkers(root / "workers.json", output / "workers")
    retention = ExperimentRetention(output)
    cache = retention.create_cache("scratch/block0-cache")
    module = "sglang.multimodal_gen.runtime."
    model = [
        "--model",
        str(root / "model"),
        "--advertise-host",
        "127.0.0.1",
        "--service-count",
        "1",
    ]
    commands = {
        "encoder": [
            sys.executable,
            "-u",
            "-m",
            module + "entrypoints.zing_source_encoder_server",
            *model,
            "--worker-id",
            "main-product-encoder",
            "--port",
            "24100",
            "--http-port",
            "24110",
            "--rendezvous-port",
            "24120",
            "--session-max-lifetime-s",
            "7200",
        ],
        "dit": [
            sys.executable,
            "-u",
            "-m",
            *(
                [
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={sp}",
                    "-m",
                ]
                if sp > 1
                else []
            ),
            module + "pipelines_core.stages.model_specific_stages.zing_h3.dit_worker",
            *model,
            "--worker-id",
            "main-product-dit",
            "--port",
            "24200",
            "--http-port",
            "24210",
            "--rendezvous-port",
            "24220",
            "--sp-degree",
            str(sp),
            "--pipeline-config-path",
            str(root / "pipeline.json"),
            "--mem-fraction-static",
            "0.85",
            "--zing-solver",
            "few_step",
            "--zing-denoising-step-list",
            "1000,500",
            "--zing-initial-denoising-step-list",
            "1000,750,500,250",
            "--zing-initial-block-count",
            "3",
            "--zing-source-joint-forward",
            "cross_chunk",
            "--zing-source-defer-final",
            "--zing-block0-cache-mode",
            "off",
            "--zing-block0-cache-dir",
            str(cache),
            "--timeout-seconds",
            "600",
        ],
        "decoder": [
            sys.executable,
            "-u",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            "-m",
            module
            + "pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker",
            *model,
            "--worker-id",
            "main-product-decoder",
            "--port",
            "24300",
            "--http-port",
            "24310",
            "--output-port",
            "24320",
            "--decoder-config",
            json.dumps(
                json.loads((root / "pipeline.json").read_text())["decoder_config"]
            ),
            "--timeout-seconds",
            "600",
        ],
    }
    devices = {
        "encoder": "0",
        "dit": ",".join(str(i) for i in range(1, sp + 1)),
        "decoder": str(sp + 1),
    }
    workers = {
        "workers": {
            role: {
                "command": commands[role],
                "http": f"http://127.0.0.1:{port}",
                "cwd": str(root / "source"),
                "env": dict(
                    CUDA_VISIBLE_DEVICES=devices[role],
                    **(
                        {
                            "MINIMAX_H3_ENCODER_SINGLE_GPU_FAST": (
                                "1"
                                if config_run.get("comparison") == "numerical"
                                else fast_flag
                            ),
                            "MINIMAX_H3_FAST_VISUAL_ENCODE": (
                                "1"
                                if config_run.get("comparison") == "numerical"
                                else fast_flag
                            ),
                            "MINIMAX_H3_ENCODER_NUMERICAL_OPTIMIZATIONS": (
                                fast_flag
                                if config_run.get("comparison") == "numerical"
                                else "1"
                            ),
                        }
                        if role == "encoder"
                        else {}
                    ),
                ),
            }
            for role, port in [("encoder", 24110), ("dit", 24210), ("decoder", 24310)]
        }
    }
    if config_run.get("diagnostics"):
        for role, entry in workers["workers"].items():
            metrics_dir = output / "prometheus" / role
            metrics_dir.mkdir(parents=True)
            entry["env"].update(
                PYTHONPATH=str(root / "harness/observer_bootstrap")
                + ":"
                + os.environ["PYTHONPATH"],
                ZING_BENCHMARK_MEMORY_DIR=str(output / "diagnostics"),
                ZING_BENCHMARK_ROLE=role,
                ZING_BENCHMARK_MEMORY_HISTORY="0",
                PROMETHEUS_MULTIPROC_DIR=str(metrics_dir),
            )
    write(root / "workers.json", workers)
    write(output / "commands.json", workers)
    group = WorkerGroupMaster(
        {"encoder": None, "dit": None, "decoder": None},
        launcher=owned,
        worker_group_id="main-product-" + str(os.getpid()),
        startup_timeout=1800,
        control_timeout=600,
    )
    client = None
    passed = False
    try:
        await group.prepare()
        write(output / "group.json", group.summary())
        if config_run.get("diagnostics"):
            from diagnostic import diagnose

            await diagnose(root, output, group, owned, expected_workers=sp + 2)
        for pass_name, media_mode, warmups, repeats in [
            ("performance", "metrics", 1, 3),
            ("visual", "video", 0, 1),
        ]:
            expected = 8 * (warmups + repeats)
            out = output / pass_name
            cmd = [
                sys.executable,
                str(root / "source/benchmark/zing_h3_causal/bench_streaming_source.py"),
                "--url",
                group.input_ws,
                "--decoder-url",
                group.output_ws,
                "--case-set",
                "product-case-20261103",
                "--media-root",
                str(root / "media"),
                "--output-dir",
                str(out),
                "--seed",
                "42",
                "--batch-pattern",
                "1",
                "--timeout",
                "7200",
                "--warmup-runs",
                str(warmups),
                "--measured-runs",
                str(repeats),
                "--no-stage-timing",
                "--media-mode",
                media_mode,
            ]
            write(
                output / "STATUS.json",
                {
                    "phase": "measuring",
                    "fast_flag": fast_flag,
                    "command": cmd,
                    "time": time.time(),
                },
            )
            with (output / (pass_name + "-client.log")).open("w") as log:
                client = subprocess.Popen(
                    cmd,
                    cwd=root / "source",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while client.poll() is None:
                    owned.check()
                    rows = out / "requests.jsonl"
                    if rows.exists():
                        completed = [
                            json.loads(line)
                            for line in rows.read_text().splitlines()
                            if line.strip()
                        ]
                        write(
                            output / "STATUS.json",
                            {
                                "phase": "measuring",
                                "fast_flag": fast_flag,
                                "requests": len(completed),
                                "last": completed[-1] if completed else None,
                                "time": time.time(),
                            },
                        )
                    await asyncio.sleep(5)
            rows = [
                json.loads(line)
                for line in (out / "requests.jsonl").read_text().splitlines()
            ]
            ok = (
                client.returncode == 0
                and len(rows) == expected
                and all(r.get("success") and r.get("delivery_complete") for r in rows)
            )
            write(
                out / "GATE.json",
                {
                    "passed": ok,
                    "expected": expected,
                    "actual": len(rows),
                    "exit_code": client.returncode,
                },
            )
            if not ok:
                raise RuntimeError("Incomplete full product benchmark")
        passed = True
    finally:
        if client is not None and client.poll() is None:
            os.killpg(client.pid, signal.SIGTERM)
            try:
                client.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(client.pid, signal.SIGKILL)
                client.wait(timeout=10)
        try:
            await group.destroy()
            write(output / "GROUP_CLEANUP.json", {"passed": True})
        finally:
            owned.close()
            retention.finish(
                workers_stopped=all(p.poll() is not None for _, p, _ in owned.owned)
            )
        write(
            output / "STATUS.json",
            {"phase": "complete" if passed else "failed", "time": time.time()},
        )


async def measure(root, output):
    from benchmark_artifacts import check_retention

    for label, flag in [("baseline", "0"), ("candidate", "1")]:
        target = output / label
        target.mkdir()
        write(
            output / "STATUS.json",
            {"phase": "running", "variant": label, "time": time.time()},
        )
        await measure_variant(root, target, flag)
    check_retention(output)
    write(output / "STATUS.json", {"phase": "complete", "time": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "measure"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    args = parser.parse_args()

    def terminate(*unused):
        raise KeyboardInterrupt("Termination requested")

    signal.signal(signal.SIGTERM, terminate)
    if args.mode == "prepare":
        prepare(args.root, args.output, args.revision)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        try:
            asyncio.run(measure(args.root, args.output))
        except BaseException:
            (args.output / "FAILURE.txt").write_text(traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
