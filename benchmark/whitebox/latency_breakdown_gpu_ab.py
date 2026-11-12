#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Same-code timing on/off experiment, derived from the successful PR690 B300 harness.

Uses an isolated E1 + DiT SP2 + D1 group and the existing Room replay runner.
Inputs: /work/source_bundle.tar.gz, wtp_clip_16s.tar.gz, run-plan.json;
prepared runtime: /mnt/ref_env/prepared.tar.gz. --prepare-only performs CPU preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def extract_tar(archive: Path, dst: Path, *, trusted: bool = False) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        if trusted:
            tf.extractall(dst, filter="fully_trusted")
        else:
            tf.extractall(dst, filter="data")


def wait_worker_ready(
    role: str,
    proc: subprocess.Popen[bytes],
    http_url: str,
    log_path: Path,
    timeout_s: float = 900.0,
) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = ""
            if log_path.exists():
                tail = log_path.read_text(errors="replace")[-6000:]
            raise RuntimeError(
                f"Worker {role} exited prematurely (rc={proc.returncode}). Log tail:\n{tail}"
            )
        try:
            with opener.open(f"{http_url}/health", timeout=3) as r1:
                health = json.loads(r1.read().decode("utf-8"))
            with opener.open(f"{http_url}/managed/discovery", timeout=3) as r2:
                disc = json.loads(r2.read().decode("utf-8"))
            if health.get("ready") is True and "control_epoch" in disc:
                return {"health": health, "discovery": disc}
        except Exception:
            pass
        time.sleep(1.5)
    tail = log_path.read_text(errors="replace")[-6000:] if log_path.exists() else ""
    raise TimeoutError(
        f"Worker {role} readiness timed out after {timeout_s}s. Log tail:\n{tail}"
    )


def stop_processes(procs: list[tuple[str, subprocess.Popen[bytes], object]]) -> None:
    for _, proc, _ in reversed(procs):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 20.0
    for _, proc, _ in reversed(procs):
        if proc.poll() is None:
            try:
                proc.wait(timeout=max(0.5, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5.0)
    for _, _, fh in procs:
        try:
            fh.close()
        except Exception:
            pass


def create_warmup_clip(src_clip: Path, dst_clip: Path, warmup_s: float = 3.0) -> None:
    if dst_clip.exists():
        shutil.rmtree(dst_clip)
    shutil.copytree(src_clip, dst_clip)
    meta_path = dst_clip / "case_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fps = int(meta.get("capture_fps", 30))
    frames = max(30, int(round(warmup_s * fps)))
    meta["case_id"] = meta["case_id"] + "-warmup"
    meta["duration_s"] = warmup_s
    meta["frames"] = frames
    meta["seed"] = int(
        meta.get("seed", 42)
    )  # Warm the same conditioning/cache key in both arms.
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Keep only inputs within warmup_s
    in_path = dst_clip / "inputs.jsonl"
    kept = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if float(row.get("rel_time_s", 0.0)) <= warmup_s:
            kept.append(json.dumps(row, ensure_ascii=False))
    in_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    work = Path(os.environ.get("LATENCY_AB_WORK", "/work"))
    plan = json.loads((work / "run-plan.json").read_text())
    results_dir = work / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ref_env = Path("/mnt/ref_env")
    root = Path(tempfile.mkdtemp(prefix="sglang-latency-ab-"))

    write_json(results_dir / "STATUS.json", {"phase": "staging", "ts": time.time()})
    print(
        "[1/6] Extracting source bundle, prepared env, and WTP 16s clip...", flush=True
    )
    extract_tar(work / "source_bundle.tar.gz", root / "source")
    extract_tar(work / "wtp_clip_16s.tar.gz", root / "wtp_clip_16s")
    extract_tar(ref_env / "prepared.tar.gz", root / "prepared_base", trusted=True)
    prep_run = root / "prepared_base/latency-matrix-runtime/run"
    assert (prep_run / "kernels/src").is_dir(), f"Missing kernels/src under {prep_run}"
    assert (prep_run / "transport-deps").is_dir(), (
        f"Missing transport-deps under {prep_run}"
    )
    assert (prep_run / "deps").is_dir(), f"Missing deps under {prep_run}"

    py_path = ":".join(
        [
            str(prep_run / "kernels/src"),
            str(prep_run / "transport-deps"),
            str(prep_run / "deps"),
            str(root / "source/python"),
        ]
    )

    # Stage model & lynn symlinks locally so profile metadata can be normalized safely
    model_src = Path(plan["model_path"])
    lynn_src = Path(plan["lynn_path"])
    assert model_src.is_dir(), f"Missing model_src: {model_src}"
    assert lynn_src.is_dir(), f"Missing lynn_src: {lynn_src}"

    (root / "model").mkdir()
    for item in model_src.iterdir():
        dst = root / "model" / item.name
        if item.name in (
            "zing_config.json",
            "zing_realtime_profile.json",
            "model_index.json",
        ):
            shutil.copy2(item, dst)
        else:
            dst.symlink_to(item.resolve(), target_is_directory=item.is_dir())

    (root / "lynn").mkdir()
    for item in lynn_src.iterdir():
        (root / "lynn" / item.name).symlink_to(
            item.resolve(), target_is_directory=item.is_dir()
        )

    # Normalize zing_realtime_profile.json using our runtime helper
    norm_code = """
import json, pathlib
from sglang.multimodal_gen.runtime.realtime.zing_source_stream import default_source_encoder
p = pathlib.Path(r'''MODEL_PROFILE''')
prof = json.loads(p.read_text())
prof["source_encoder"] = default_source_encoder()
prof.pop("source_decoder", None)
prof["initial_block_count"] = 0
prof.pop("initial_denoising_step_list", None)
p.write_text(json.dumps(prof, indent=2) + "\\n")
print(json.dumps(prof.get("source_attention_window") or {"window_chunks": 6, "sink_chunks": 3}))
""".replace("MODEL_PROFILE", str(root / "model/zing_realtime_profile.json"))
    attn_win_raw = subprocess.check_output(
        [sys.executable, "-c", norm_code],
        env={**os.environ, "PYTHONPATH": py_path, "SGLANG_LIGHTWEIGHT_RUNTIME": "1"},
        text=True,
    ).strip()
    attn_win = json.loads(attn_win_raw.splitlines()[-1])
    win_chunks = int(attn_win.get("window_chunks", 6))
    sink_chunks = int(attn_win.get("sink_chunks", 3))

    pipeline = {
        "zing_kv_pool_window_chunks": win_chunks,
        "zing_kv_pool_sink_chunks": sink_chunks,
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
    write_json(root / "pipeline.json", pipeline)

    runtime_env = {
        **os.environ,
        "PYTHONPATH": py_path,
        "SGLANG_LIGHTWEIGHT_RUNTIME": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    actual_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root / "source", text=True
    ).strip()
    if actual_sha != plan["source_revision"]:
        raise ValueError("Source bundle does not match the frozen run plan")
    check_code = """
import pathlib, subprocess, shutil
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import queued_dit, dit_worker
from sglang.multimodal_gen.runtime.entrypoints import zing_source_encoder_server
from sglang.multimodal_gen.runtime.realtime import decoder_output
assert pathlib.Path(queued_dit.__file__).resolve().is_relative_to(pathlib.Path.cwd() / 'python')
assert queued_dit.QueuedDiTSession.__init__.__kwdefaults__['return_stage_timing'] is False
for tool in ('ffmpeg', 'ffprobe'):
    assert shutil.which(tool), tool
assert 'libx264' in subprocess.check_output(['ffmpeg', '-hide_banner', '-encoders'], stderr=subprocess.STDOUT, text=True)
print('Runtime imports, timing default and codecs passed')
"""
    subprocess.run(
        [sys.executable, "-c", check_code],
        cwd=root / "source",
        env=runtime_env,
        check=True,
    )
    # Prepare an explicitly derived diagnostic 736 clip; never relabel as a frozen product case.
    clip736 = root / "wtp_clip_736"
    shutil.copytree(root / "wtp_clip_16s", clip736)
    meta_path = clip736 / "case_meta.json"
    meta = json.loads(meta_path.read_text())
    source_video = root / "wtp_clip_16s" / meta["whitebox_mp4"]["path"]
    dest_video = clip736 / meta["whitebox_mp4"]["path"]
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source_video),
            "-vf",
            "scale=1280:736:flags=bilinear",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            str(dest_video),
        ],
        check=True,
    )
    meta.update(
        width=1280,
        height=736,
        case_id=meta["case_id"] + "-derived-736",
        input_derivation="bilinear scale from the 832x480 WTP diagnostic clip",
    )
    meta["whitebox_mp4"].update(
        bytes=dest_video.stat().st_size,
        sha256=hashlib.sha256(dest_video.read_bytes()).hexdigest(),
    )
    write_json(meta_path, meta)
    preflight = dict(
        source_revision=actual_sha,
        image=plan["image"],
        input_clip_sha256=hashlib.sha256(
            (work / "wtp_clip_16s.tar.gz").read_bytes()
        ).hexdigest(),
        model_path=str(model_src),
        lynn_path=str(lynn_src),
        runtime_imports="passed",
        resolutions=[[832, 480], [1280, 736]],
    )
    write_json(results_dir / "PREP_COMPLETE.json", preflight)
    if args.prepare_only:
        return 0

    # Probe GPUs
    smi_out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    gpus = []
    for line in smi_out.splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        gpus.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mib": int(parts[3]),
            }
        )
    assert len(gpus) >= 4, f"Expected >=4 GPUs on B300 job, got {len(gpus)}: {gpus}"
    write_json(
        results_dir / "HARDWARE.json",
        {
            "gpus": gpus,
            "attention_window": attn_win,
            "role_topology": {
                "encoder": [gpus[0]["uuid"]],
                "dit_sp2": [gpus[1]["uuid"], gpus[2]["uuid"]],
                "decoder": [gpus[3]["uuid"]],
            },
        },
    )

    # Ephemeral isolated role-pool credentials (never written to disk or logs)
    signing_secret = uuid.uuid4().hex + uuid.uuid4().hex
    role_token = uuid.uuid4().hex + uuid.uuid4().hex
    control_epoch = "isolated-epoch-" + uuid.uuid4().hex[:12]

    profile_limits = {
        "480": {
            "source_sizes": [[832, 480]],
            "target_sizes": [[832, 480]],
            "max_fps": 24,
            "max_references": 21,
            "max_reference_bytes": 67108864,
            "max_reference_pixels": 33554432,
            "max_prompt_chars": 65536,
            "max_preview_pixels": 16777216,
        }
    }
    profile_limits["736"] = {
        **profile_limits["480"],
        "source_sizes": [[1280, 736]],
        "target_sizes": [[1280, 736]],
    }
    hw_map = {
        "encoder": [gpus[0]["uuid"]],
        "dit": [gpus[1]["uuid"], gpus[2]["uuid"]],
        "decoder": [gpus[3]["uuid"]],
    }
    for role in ("encoder", "dit", "decoder"):
        wcfg = {
            "role": role,
            "service_count": 1,
            "coordinator_url": "http://127.0.0.1:9",
            "pool_id": "isolated-latency-pool",
            "control_epoch": control_epoch,
            "compatibility": "b300-zing-h3-latency",
            "network_domain": "isolated-b300-pod",
            "capacity": {"slots": 1},
            "profiles": {"480": {"slots": 1}, "736": {"slots": 1}},
            "profile_limits": profile_limits,
            "hardware_ids": hw_map[role],
            "signing_secret_env": "ROLE_POOL_SIGNING_KEY",
            "role_token_env": "ROLE_POOL_REGISTRATION_TOKEN",
            "heartbeat_seconds": 30,
        }
        write_json(root / f"worker_cfg_{role}.json", wcfg)

    sys.path.insert(0, str(root / "source/benchmark/zing_h3_causal"))
    from benchmark_artifacts import ExperimentRetention

    retention = ExperimentRetention(root / "owned-artifacts")
    cache_dir = retention.create_cache("scratch/block0-cache")
    (cache_dir / "capacity").mkdir(parents=True, exist_ok=True)

    base_env = dict(
        os.environ,
        PYTHONPATH=py_path,
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        NCCL_MIN_NCHANNELS="2",
        NCCL_MIN_CTAS="2",
        NCCL_IB_DISABLE="1",
        NCCL_NET="Socket",
        NCCL_DEBUG="WARN",
        NCCL_GRAPH_REGISTER="0",
        SGLANG_ZING_EXPERIMENTAL_ATTENTION="fa4",
        SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES="1",
        SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN="1",
        SGLANG_ZING_DENOISE_CUDA_GRAPH="1",
        SGLANG_ZING_DENOISE_CUDA_GRAPH_SP="1",
        SGLANG_ZING_SOURCE_PROMPT_JOINT="0",
        SGLANG_ZING_SPARSE_ATTN_MODE="dense",
        SGLANG_ZING_SPARSE_THREE_TIER="0",
        SGLANG_ZING_QUANT_SCOPE="ffn-attention",
        SGLANG_ZING_FUSED_PRE_LINEAR_QUANT="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        ROLE_POOL_SIGNING_KEY=signing_secret,
        ROLE_POOL_REGISTRATION_TOKEN=role_token,
    )
    for qflag in (
        "HF_RMSNORM",
        "FUSED_MROPE",
        "FUSED_GATE_UP",
        "TEXT_FA4",
        "VISION_FA4",
        "FP8",
    ):
        base_env[f"SGLANG_H3_QWEN_{qflag}"] = "0"
    for dvar in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        base_env.pop(dvar, None)

    common_model_args = [
        "--model",
        str(root / "model"),
        "--advertise-host",
        "127.0.0.1",
        "--service-count",
        "1",
    ]
    worker_specs = [
        (
            "encoder",
            "0",
            "http://127.0.0.1:24110",
            [
                sys.executable,
                "-u",
                "-m",
                "sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server",
                *common_model_args,
                "--worker-id",
                "latency-encoder-0",
                "--port",
                "24100",
                "--http-port",
                "24110",
                "--rendezvous-port",
                "24120",
                "--session-max-lifetime-s",
                "7200",
            ],
        ),
        (
            "dit",
            "1,2",
            "http://127.0.0.1:24210",
            [
                sys.executable,
                "-u",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "-m",
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_worker",
                *common_model_args,
                "--worker-id",
                "latency-dit-0",
                "--port",
                "24200",
                "--http-port",
                "24210",
                "--rendezvous-port",
                "24220",
                "--sp-degree",
                "2",
                "--pipeline-config-path",
                str(root / "pipeline.json"),
                "--mem-fraction-static",
                "0.7",
                "--zing-solver",
                "few_step",
                "--zing-denoising-step-list",
                "1000,500",
                "--zing-initial-block-count",
                "0",
                "--zing-source-joint-forward",
                "cross_chunk",
                "--zing-source-defer-final",
                "--zing-block0-cache-mode",
                "on",
                "--zing-block0-cache-backend",
                "local",
                "--zing-block0-cache-host-gb",
                "128",
                "--zing-block0-cache-disk-gb",
                "200",
                "--zing-block0-cache-dir",
                str(cache_dir),
                "--zing-block0-cache-storage",
                "local",
                "--zing-block0-cache-budget-root",
                str(cache_dir),
                "--zing-block0-cache-capacity-dir",
                str(cache_dir / "capacity"),
                "--zing-block0-cache-host-total-gb",
                "128",
                "--zing-block0-cache-min-free-gb",
                "32",
                "--zing-block0-cache-inflight-gb",
                "64",
                "--zing-block0-cache-max-writers",
                "2",
                "--zing-block0-cache-gc-high-watermark",
                "0.9",
                "--zing-block0-cache-gc-low-watermark",
                "0.8",
                "--zing-block0-cache-gc",
                "--quantization",
                "fp8",
                "--timeout-seconds",
                "600",
            ],
        ),
        (
            "decoder",
            "3",
            "http://127.0.0.1:24310",
            [
                sys.executable,
                "-u",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=1",
                "-m",
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker",
                *common_model_args,
                "--worker-id",
                "latency-decoder-0",
                "--port",
                "24300",
                "--http-port",
                "24310",
                "--output-port",
                "24320",
                "--decoder-config",
                json.dumps(pipeline["decoder_config"]),
                "--timeout-seconds",
                "600",
            ],
        ),
    ]

    logs_dir = results_dir / "worker_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[str, subprocess.Popen[bytes], object]] = []

    try:
        print(
            "[2/6] Starting isolated Encoder, DiT (SP2 FP8), and Decoder workers...",
            flush=True,
        )
        write_json(
            results_dir / "STATUS.json",
            {"phase": "starting_workers", "ts": time.time()},
        )
        for role, cuda_devs, _, cmd in worker_specs:
            jit = root / "jit" / role
            for sub in ("inductor", "triton", "cuda", "extensions"):
                (jit / sub).mkdir(parents=True, exist_ok=True)
            role_env = dict(
                base_env,
                CUDA_VISIBLE_DEVICES=cuda_devs,
                SGLANG_ROLE_POOL_WORKER_CONFIG=str(root / f"worker_cfg_{role}.json"),
                TORCHINDUCTOR_CACHE_DIR=str(jit / "inductor"),
                TRITON_CACHE_DIR=str(jit / "triton"),
                CUDA_CACHE_PATH=str(jit / "cuda"),
                TORCH_EXTENSIONS_DIR=str(jit / "extensions"),
            )
            log_path = logs_dir / f"{role}.log"
            fh = log_path.open("wb")
            proc = subprocess.Popen(
                cmd,
                cwd=str(root / "source"),
                env=role_env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            procs.append((role, proc, fh))

        t_boot = time.monotonic()
        readiness = {}
        for role, (_, proc, _), (_, _, url, _) in zip(
            ("encoder", "dit", "decoder"), procs, worker_specs
        ):
            readiness[role] = wait_worker_ready(
                role, proc, url, logs_dir / f"{role}.log"
            )
            print(f"  -> {role} ready at {url}", flush=True)
        boot_s = round(time.monotonic() - t_boot, 2)
        write_json(
            results_dir / "WORKERS_READY.json",
            {"boot_seconds": boot_s, "readiness": readiness},
        )

        replay_env = dict(
            os.environ,
            PYTHONPATH=py_path,
            SGLANG_LIGHTWEIGHT_RUNTIME="1",
            ROLE_POOL_SIGNING_SECRET=signing_secret,
            WHITEBOX_INFERENCE_STEPS="2",
        )
        replay_script = str(
            root / "source/benchmark/whitebox/run_latency_breakdown_replay.py"
        )

        write_json(
            results_dir / "CONFIG.json",
            dict(
                source_revision=actual_sha,
                pipeline=pipeline,
                preflight=preflight,
                commands=[
                    dict(role=role, cuda_devices=devices, command=cmd)
                    for role, devices, _, cmd in worker_specs
                ],
                parameters=dict(
                    steps=2,
                    pacing_speed=1.0,
                    flow="Room natural adaptive",
                    measured_order=["off", "on", "on", "off", "off", "on"],
                    group="E1 + DiT SP2 + D1",
                    dit_precision="FP8",
                    decoder_precision="BF16",
                    initial_block_count=0,
                ),
            ),
        )
        arm_results = []
        monitor_stop = threading.Event()
        active_arm = {"name": "idle"}

        def monitor_memory():
            with (results_dir / "gpu-memory.jsonl").open("w") as fh:
                while not monitor_stop.is_set():
                    raw = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    )
                    fh.write(
                        json.dumps(
                            dict(
                                time_s=time.time(),
                                arm=active_arm["name"],
                                rows=raw.splitlines(),
                            )
                        )
                        + "\n"
                    )
                    fh.flush()
                    monitor_stop.wait(1)

        monitor = threading.Thread(target=monitor_memory, daemon=True)
        monitor.start()
        try:
            for height, clip in ((480, root / "wtp_clip_16s"), (736, clip736)):
                warmup_clip = root / f"warmup-{height}"
                create_warmup_clip(clip, warmup_clip, warmup_s=3.0)
                sequence = [
                    ("warmup-off", "off", warmup_clip),
                    ("warmup-on", "on", warmup_clip),
                ]
                sequence += [
                    (f"measured-{i}-{mode}", mode, clip)
                    for i, mode in enumerate(("off", "on", "on", "off", "off", "on"))
                ]
                for label, mode, input_clip in sequence:
                    arm_name = f"{height}-{label}"
                    active_arm["name"] = arm_name
                    write_json(
                        results_dir / "STATUS.json",
                        dict(phase="measuring", arm=arm_name),
                    )
                    out = root / arm_name
                    command = [
                        sys.executable,
                        "-u",
                        replay_script,
                        "--clip-dir",
                        str(input_clip),
                        "--output-dir",
                        str(out),
                        "--mode",
                        "real-sglang",
                        "--sampling",
                        "adaptive",
                        "--pacing-speed",
                        "1.0",
                        "--num-inference-steps",
                        "2",
                        "--resolution",
                        str(height),
                        "--stage-timing",
                        mode,
                        "--encoder-url",
                        "http://127.0.0.1:24110",
                        "--dit-url",
                        "http://127.0.0.1:24210",
                        "--decoder-url",
                        "http://127.0.0.1:24310",
                        "--profile",
                        str(height),
                        "--control-epoch",
                        control_epoch,
                        "--server-revision",
                        actual_sha,
                    ]
                    print("Running", arm_name, flush=True)
                    with (results_dir / (arm_name + ".log")).open("w") as log:
                        subprocess.run(
                            command,
                            cwd=root / "source",
                            env=replay_env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=600,
                        )
                    data = json.loads((out / "latency-breakdown-data.json").read_text())
                    summary = data["summary"]
                    if mode == "on" and summary["model_timing_chunks"] < 1:
                        raise ValueError("Timing-on produced no stage telemetry")
                    if mode == "off" and summary["model_timing_chunks"] != 0:
                        raise ValueError("Timing-off still produced stage telemetry")
                    saved = results_dir / arm_name
                    saved.mkdir()
                    for name in (
                        "latency-breakdown-data.json",
                        "index.html",
                        "latency-breakdown-overview.png",
                    ):
                        shutil.copy2(out / name, saved / name)
                    arm_results.append(
                        dict(
                            height=height,
                            arm=label,
                            stage_timing=mode,
                            measured=label.startswith("measured"),
                            summary=summary,
                        )
                    )
                    write_json(results_dir / "arms.json", arm_results)
                    active_arm["name"] = "idle"
            comparison = []
            for height in (480, 736):
                row = dict(height=height, metrics={})
                for metric in (
                    "session_ttff_ms",
                    "room_output_elapsed_s",
                    "room_output_fps",
                    "source_to_raw_output_ms",
                    "dispatch_to_raw_output_ms",
                ):

                    def value(summary):
                        item = summary.get(metric)
                        return item.get("median") if isinstance(item, dict) else item

                    vals = {
                        mode: [
                            value(a["summary"])
                            for a in arm_results
                            if a["height"] == height
                            and a["measured"]
                            and a["stage_timing"] == mode
                        ]
                        for mode in ("off", "on")
                    }
                    if any(v is None for values in vals.values() for v in values):
                        continue
                    off, on = (statistics.median(vals[mode]) for mode in ("off", "on"))
                    row["metrics"][metric] = dict(
                        off=off,
                        on=on,
                        delta=on - off,
                        relative_percent=100 * (on / off - 1) if off else None,
                        samples=vals,
                    )
                comparison.append(row)
            write_json(results_dir / "comparison.json", comparison)
        finally:
            monitor_stop.set()
            monitor.join(timeout=10)

        # Record peak GPU memory at completion before stopping workers
        smi_after = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        write_json(
            results_dir / "STATUS.json",
            {
                "phase": "benchmarks_completed",
                "ts": time.time(),
                "gpu_memory_after_run": smi_after.splitlines(),
            },
        )
        print(
            "[6/6] Measurements persisted. Stopping workers and validating cleanup.",
            flush=True,
        )
        return 0
    except BaseException:
        err_txt = traceback.format_exc()
        print("JOB_FAILED:\n" + err_txt, file=sys.stderr, flush=True)
        write_json(
            results_dir / "STATUS.json",
            {"phase": "failed", "ts": time.time(), "error": err_txt[-8000:]},
        )
        return 1
    finally:
        stop_processes(procs)
        workers_stopped = all(proc.poll() is not None for _, proc, _ in procs)
        try:
            retention.finish(workers_stopped=workers_stopped)
            for name in ("cache-cleanup.json", "retention.json"):
                shutil.copy2(retention.root / name, results_dir / name)
            from benchmark_artifacts import check_retention

            write_json(
                results_dir / "total-retention.json", check_retention(results_dir)
            )
            status = json.loads((results_dir / "STATUS.json").read_text())
            if status["phase"] == "benchmarks_completed":
                write_json(
                    results_dir / "STATUS.json", dict(phase="succeeded", ts=time.time())
                )
        except BaseException as exc:
            # Preserve the refusal and enough evidence to inspect it; never
            # remove links or relax the shared cleanup guard to force success.
            unsafe = []
            for cache, _, _ in retention.caches:
                for path in cache.rglob("*"):
                    if path.is_symlink() or path.is_mount():
                        unsafe.append(
                            dict(
                                path=str(path.relative_to(retention.root)),
                                target=os.readlink(path) if path.is_symlink() else None,
                            )
                        )
            write_json(
                results_dir / "CLEANUP.json",
                dict(
                    status="failed",
                    workers_stopped=workers_stopped,
                    unsafe_paths=unsafe,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    previous_status=json.loads(
                        (results_dir / "STATUS.json").read_text()
                    ),
                ),
            )
            write_json(
                results_dir / "STATUS.json",
                dict(
                    phase="cleanup_failed",
                    measurements_complete=(results_dir / "comparison.json").exists(),
                    workers_stopped=workers_stopped,
                    ts=time.time(),
                ),
            )
            raise
        finally:
            retention.close()


if __name__ == "__main__":
    raise SystemExit(main())
