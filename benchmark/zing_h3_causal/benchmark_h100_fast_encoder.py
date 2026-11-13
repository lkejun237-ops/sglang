# SPDX-License-Identifier: Apache-2.0
"""Shanghai H100 single-GPU A/B benchmark and numerical/memory verification for Zing H3 Encoder.

Measures real trained MiniMax H3 visual VAE weights across 832x480 (480p), 1280x736 (736p),
and 1312x736 (736p wide), comparing Baseline (MINIMAX_H3_ENCODER_SINGLE_GPU_FAST=0,
MINIMAX_H3_FAST_VISUAL_ENCODE=0) against Candidate (default enabled).
Profiler trace capture runs in a strictly separate post-timing pass when --profile is set.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
    ZingH3CausalPipelineConfig,
)
from sglang.multimodal_gen.runtime.conditioning.cache import ConditioningCache
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
    _consume_vae_checkpoint_arch_metadata,
)
from sglang.multimodal_gen.runtime.loader.utils import load_model_state_dict
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
    ZingH3Codecs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
    create_session,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args


@contextmanager
def _temp_env(**overrides: str):
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[k] = v
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _stats_summary(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    n = len(ordered)
    p95_idx = min(n - 1, max(0, math.ceil(0.95 * n) - 1))
    return {
        "median_ms": round(statistics.median(ordered), 3),
        "mean_ms": round(statistics.fmean(ordered), 3),
        "p95_ms": round(ordered[p95_idx], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "samples_ms": [round(x, 3) for x in samples_ms],
    }


def _tensor_error_metrics(cand: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    c = cand.detach().float()
    r = ref.detach().float()
    diff = (c - r).abs()
    rmse = float(diff.pow(2).mean().sqrt().item())
    ref_rms = float(r.pow(2).mean().sqrt().clamp_min(1e-12).item())
    cos = float(F.cosine_similarity(c.reshape(1, -1), r.reshape(1, -1), dim=1).item())
    return {
        "max_abs": round(float(diff.max().item()), 6),
        "mean_abs": round(float(diff.mean().item()), 6),
        "rmse": round(rmse, 6),
        "rrmse": round(rmse / ref_rms, 6),
        "cosine_sim": round(cos, 8),
    }


def _query_gpu_hardware(device_idx: int) -> dict:
    props = torch.cuda.get_device_properties(device_idx)
    info = {
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "total_memory_mib": round(props.total_memory / (1024 * 1024), 1),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    try:
        import triton

        info["triton_version"] = triton.__version__
    except Exception:
        info["triton_version"] = None
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used,clocks.sm,clocks.mem,power.draw,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        ).strip()
        info["nvidia_smi"] = smi.splitlines()
    except Exception as exc:
        info["nvidia_smi_error"] = str(exc)
    return info


def load_trained_encoder(root: Path, device: torch.device) -> ZingH3Codecs:
    set_global_server_args(
        ServerArgs.from_kwargs(
            model_path=str(root),
            pipeline_config=ZingH3CausalPipelineConfig(
                video_decoder_backend="native", zing_adaln_mode="resident"
            ),
            num_gpus=1,
            attention_backend="torch_sdpa",
            warmup_mode="off",
        )
    )
    config = MiniMaxH3VideoVAEConfig(load_decoder=False)
    config.update_model_arch(json.loads((root / "video_vae/config.json").read_text()))
    model = MiniMaxH3VideoVAE(config)

    weights = root / "video_vae/source/model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as f:
        full_meta = {
            k: f.get_tensor(k)
            for k in f.keys()
            if not k.startswith(("decoder.", "post_quant_conv."))
        }
    _consume_vae_checkpoint_arch_metadata(full_meta, config, model.state_dict())
    load_model_state_dict(model, full_meta, strict=True, assign=True)
    del full_meta

    model = model.to(device=device, dtype=torch.bfloat16).eval()
    model.stack_tiling = False

    codecs = object.__new__(ZingH3Codecs)
    codecs.modules = {"video_vae": model}
    codecs.va = config.arch_config
    if codecs.va.latents_mean is None or codecs.va.latents_std is None:
        raw_cfg = json.loads((root / "video_vae/config.json").read_text())
        codecs.va = SimpleNamespace(
            latents_mean=raw_cfg.get("latents_mean", [0.0] * 24),
            latents_std=raw_cfg.get("latents_std", [1.0] * 24),
        )
    codecs.temporal_ratio = codecs.va.temporal_compression_ratio
    codecs.spatial_ratio = codecs.va.spatial_compression_ratio
    codecs.vae_frames_per_chunk = codecs.va.vae_clip_length
    codecs.vae_latents_per_chunk = math.ceil(
        codecs.vae_frames_per_chunk / codecs.temporal_ratio
    )
    codecs.runtime = json.loads((root / "zing_config.json").read_text())
    codecs.reference_cache = ConditioningCache(256 << 20)
    codecs.prepare_visual_encoder_weights()
    return codecs


def make_deterministic_video(
    frames: int, height: int, width: int, seed: int = 20261112
) -> torch.Tensor:
    """Create smooth spatio-temporal gradient + texture uint8 [3, T, H, W] on CPU."""
    gen = torch.Generator(device="cpu").manual_seed(seed + height * 17 + width)
    y = torch.linspace(0.0, 1.0, height, dtype=torch.float32).view(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, width, dtype=torch.float32).view(1, 1, 1, width)
    t = torch.linspace(0.0, 1.0, frames, dtype=torch.float32).view(1, frames, 1, 1)
    r = (0.55 * x + 0.30 * y + 0.15 * torch.sin(t * 6.283)) * 220.0
    g = (0.25 * x + 0.60 * y + 0.15 * torch.cos(t * 6.283)) * 220.0
    b = (0.40 * (1.0 - x) + 0.40 * y + 0.20 * t) * 220.0
    base = torch.cat(
        [
            r.expand(1, frames, height, width),
            g.expand(1, frames, height, width),
            b.expand(1, frames, height, width),
        ],
        dim=0,
    )
    noise = torch.randint(
        0, 32, (3, frames, height, width), generator=gen, dtype=torch.uint8
    ).float()
    return (base + noise).clamp(0, 255).to(torch.uint8).contiguous()


def measure_mode_on_canvas(
    codecs: ZingH3Codecs,
    video_cpu: torch.Tensor,
    *,
    fast_flag: str,
    repeats: int,
    warmup_iters: int,
    device: torch.device,
) -> tuple[dict, dict[str, torch.Tensor]]:
    _, total_frames, height, width = video_cpu.shape
    video_gpu = video_cpu.to(device=device)

    with _temp_env(
        MINIMAX_H3_ENCODER_SINGLE_GPU_FAST=fast_flag,
        MINIMAX_H3_FAST_VISUAL_ENCODE=fast_flag,
    ):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        resident_alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)

        # Untimed warmup passes for exact canvas shape
        t_warm0 = time.perf_counter()
        for _ in range(warmup_iters):
            ws = create_session(codecs, capacity_frames=total_frames)
            ws.append(video_cpu[:, :9], frame_start=0)
            _ = ws.pop()
            _ = ws.pop()
            ws.dispose()
            _ = codecs.encode_visual(video_cpu[:, :1])
        torch.cuda.synchronize(device)
        warmup_ms = (time.perf_counter() - t_warm0) * 1000.0

        torch.cuda.reset_peak_memory_stats(device)

        pop_t1_e2e_samples: list[float] = []
        pop_t1_gpu_samples: list[float] = []
        pop_t8_e2e_samples: list[float] = []
        pop_t8_gpu_samples: list[float] = []
        stream_total_samples: list[float] = []
        vis_t1_samples: list[float] = []
        vis_17f_samples: list[float] = []

        captured_moments: torch.Tensor | None = None
        captured_vis_t1: torch.Tensor | None = None
        captured_vis_m17: torch.Tensor | None = None
        last_exec_snapshot: dict | None = None

        # 1. Measure progressive streaming (CPU uint8 -> pop(T=1), pop(T=8), full 34f stream)
        for rep in range(repeats):
            sess = create_session(codecs, capacity_frames=total_frames)
            torch.cuda.synchronize(device)

            t_stream_start = time.perf_counter()

            # First frame T=1 (end-to-end CPU append + pop)
            t0 = time.perf_counter()
            sess.append(video_cpu[:, :1], frame_start=0)
            out0 = sess.pop()
            torch.cuda.synchronize(device)
            t_first = (time.perf_counter() - t0) * 1000.0
            pop_t1_e2e_samples.append(t_first)

            # Second chunk T=8 (end-to-end CPU append + pop)
            t1 = time.perf_counter()
            sess.append(video_cpu[:, 1:9], frame_start=1)
            out1 = sess.pop()
            torch.cuda.synchronize(device)
            t_second = (time.perf_counter() - t1) * 1000.0
            pop_t8_e2e_samples.append(t_second)

            # Remaining frames (9..total_frames)
            emitted = [out0.values, out1.values]
            sess.append(video_cpu[:, 9:total_frames], frame_start=9)
            sess.finish()
            while sess.ready:
                emitted.append(sess.pop().values)
            torch.cuda.synchronize(device)
            t_total = (time.perf_counter() - t_stream_start) * 1000.0
            stream_total_samples.append(t_total)

            if rep == repeats - 1:
                captured_moments = torch.cat(emitted, dim=2).detach().cpu()
                last_exec_snapshot = dict(sess.cache.last_execution or {})
            sess.dispose()

        # 2. Measure GPU-resident pop(T=1) and pop(T=8) (pure GPU kernel + conv latency)
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_mid = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        for _ in range(repeats):
            sess_g = create_session(codecs, capacity_frames=17)
            sess_g.append(video_gpu[:, :9], frame_start=0)
            torch.cuda.synchronize(device)
            ev_start.record()
            _ = sess_g.pop()
            ev_mid.record()
            _ = sess_g.pop()
            ev_end.record()
            torch.cuda.synchronize(device)
            pop_t1_gpu_samples.append(float(ev_start.elapsed_time(ev_mid)))
            pop_t8_gpu_samples.append(float(ev_mid.elapsed_time(ev_end)))
            sess_g.dispose()

        # 3. Measure encode_visual(T=1) (single reference image) and encode_visual(T=17)
        for rep in range(repeats):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            lat1 = codecs.encode_visual(video_cpu[:, :1])
            torch.cuda.synchronize(device)
            vis_t1_samples.append((time.perf_counter() - t0) * 1000.0)
            if rep == repeats - 1:
                captured_vis_t1 = lat1.detach().cpu()

        for rep in range(max(3, repeats // 2)):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = codecs.encode_visual(video_cpu[:, :17])
            torch.cuda.synchronize(device)
            vis_17f_samples.append((time.perf_counter() - t0) * 1000.0)

        captured_vis_m17 = (
            codecs.encode_visual_moments(video_cpu[:, :17]).detach().cpu()
        )

        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)

        # Verify session cleanup leaves zero retained bytes
        probe_sess = create_session(codecs, capacity_frames=17)
        probe_sess.append(video_cpu[:, :9], frame_start=0)
        _ = probe_sess.pop()
        retained_mid_clip = int(probe_sess.state_usage().retained_bytes)
        probe_sess.dispose()
        retained_after_dispose = int(probe_sess.state_usage().retained_bytes)
        del probe_sess
        gc.collect()
        torch.cuda.synchronize(device)
        after_dispose_alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)

    pop_t8_per_frame = [x / 8.0 for x in pop_t8_e2e_samples]
    pop_t8_gpu_per_frame = [x / 8.0 for x in pop_t8_gpu_samples]
    stream_per_frame = [x / total_frames for x in stream_total_samples]
    vis_17f_per_frame = [x / 17.0 for x in vis_17f_samples]

    metrics = {
        "fast_flag": fast_flag,
        "warmup_ms": round(warmup_ms, 2),
        "pop_t1_e2e": _stats_summary(pop_t1_e2e_samples),
        "pop_t1_gpu": _stats_summary(pop_t1_gpu_samples),
        "encode_visual_t1": _stats_summary(vis_t1_samples),
        "pop_t8_e2e": _stats_summary(pop_t8_e2e_samples),
        "pop_t8_e2e_per_frame": _stats_summary(pop_t8_per_frame),
        "pop_t8_gpu": _stats_summary(pop_t8_gpu_samples),
        "pop_t8_gpu_per_frame": _stats_summary(pop_t8_gpu_per_frame),
        "stream_total": _stats_summary(stream_total_samples),
        "stream_per_frame": _stats_summary(stream_per_frame),
        "stream_median_fps": round(1000.0 / statistics.median(stream_per_frame), 2),
        "encode_visual_17f": _stats_summary(vis_17f_samples),
        "encode_visual_17f_per_frame": _stats_summary(vis_17f_per_frame),
        "memory": {
            "resident_allocated_mb": round(resident_alloc_mb, 2),
            "peak_allocated_mb": round(peak_alloc_mb, 2),
            "peak_reserved_mb": round(peak_reserved_mb, 2),
            "after_dispose_allocated_mb": round(after_dispose_alloc_mb, 2),
            "retained_bytes_mid_clip": retained_mid_clip,
            "retained_bytes_after_dispose": retained_after_dispose,
        },
        "last_execution": last_exec_snapshot,
    }
    tensors = {
        "progressive_moments": captured_moments,
        "encode_visual_t1": captured_vis_t1,
        "encode_visual_moments_17f": captured_vis_m17,
    }
    return metrics, tensors


def capture_profiler_trace(
    codecs: ZingH3Codecs,
    video_cpu: torch.Tensor,
    trace_dir: Path,
    label: str,
    device: torch.device,
) -> list[dict]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    with _temp_env(
        MINIMAX_H3_ENCODER_SINGLE_GPU_FAST="1",
        MINIMAX_H3_FAST_VISUAL_ENCODE="1",
    ):
        sess = create_session(codecs, capacity_frames=17)
        sess.append(video_cpu[:, :9], frame_start=0)
        torch.cuda.synchronize(device)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            with torch.profiler.record_function("pop_t1"):
                _ = sess.pop()
            with torch.profiler.record_function("pop_t8"):
                _ = sess.pop()
            torch.cuda.synchronize(device)
        sess.dispose()

    trace_path = trace_dir / f"trace_{label}.json.gz"
    prof.export_chrome_trace(str(trace_path))

    rows = []
    for item in sorted(
        prof.key_averages(),
        key=lambda e: getattr(e, "self_device_time_total", 0.0),
        reverse=True,
    )[:20]:
        cuda_us = float(getattr(item, "self_device_time_total", 0.0))
        if cuda_us <= 0:
            continue
        rows.append(
            {
                "name": item.key,
                "count": item.count,
                "self_cuda_ms": round(cuda_us / 1000.0, 3),
                "cpu_total_ms": round(float(item.cpu_time_total) / 1000.0, 3),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="unknown")
    parser.add_argument(
        "--resolutions",
        default="832x480,1280x736,1312x736",
        help="Comma-separated WxH list",
    )
    parser.add_argument("--frames", type=int, default=34)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    args.output.mkdir(parents=True, exist_ok=True)
    hw_info = _query_gpu_hardware(0)

    t_load0 = time.perf_counter()
    codecs = load_trained_encoder(args.model, device)
    load_sec = time.perf_counter() - t_load0

    t_warm_svc0 = time.perf_counter()
    codecs.warmup_source_vae()
    startup_warmup_sec = time.perf_counter() - t_warm_svc0

    report: dict = {
        "revision": args.revision,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hw_info,
        "load_seconds": round(load_sec, 3),
        "startup_warmup_source_vae_seconds": round(startup_warmup_sec, 3),
        "frames_per_stream": args.frames,
        "repeats": args.repeats,
        "resolutions": {},
        "targets_check": {},
    }

    canvas_list: list[tuple[int, int]] = []
    for item in args.resolutions.split(","):
        item = item.strip().lower()
        if not item:
            continue
        w_s, _, h_s = item.partition("x")
        canvas_list.append((int(w_s), int(h_s)))

    all_targets_met = True
    for width, height in canvas_list:
        key = f"{width}x{height}"
        print(
            f"\n=== Benchmarking {key} (frames={args.frames}, repeats={args.repeats}) ===",
            flush=True,
        )
        video_cpu = make_deterministic_video(args.frames, height, width)

        base_metrics, base_tensors = measure_mode_on_canvas(
            codecs,
            video_cpu,
            fast_flag="0",
            repeats=args.repeats,
            warmup_iters=args.warmup_iters,
            device=device,
        )
        cand_metrics, cand_tensors = measure_mode_on_canvas(
            codecs,
            video_cpu,
            fast_flag="1",
            repeats=args.repeats,
            warmup_iters=args.warmup_iters,
            device=device,
        )

        prog_err = _tensor_error_metrics(
            cand_tensors["progressive_moments"], base_tensors["progressive_moments"]
        )
        prog_mean_err = _tensor_error_metrics(
            cand_tensors["progressive_moments"][:, :24],
            base_tensors["progressive_moments"][:, :24],
        )
        prog_logvar_err = _tensor_error_metrics(
            cand_tensors["progressive_moments"][:, 24:],
            base_tensors["progressive_moments"][:, 24:],
        )
        vis_t1_err = _tensor_error_metrics(
            cand_tensors["encode_visual_t1"], base_tensors["encode_visual_t1"]
        )
        vis_m17_err = _tensor_error_metrics(
            cand_tensors["encode_visual_moments_17f"],
            base_tensors["encode_visual_moments_17f"],
        )

        speedups = {
            "pop_t1_e2e_speedup": round(
                base_metrics["pop_t1_e2e"]["median_ms"]
                / cand_metrics["pop_t1_e2e"]["median_ms"],
                2,
            ),
            "pop_t1_gpu_speedup": round(
                base_metrics["pop_t1_gpu"]["median_ms"]
                / cand_metrics["pop_t1_gpu"]["median_ms"],
                2,
            ),
            "encode_visual_t1_speedup": round(
                base_metrics["encode_visual_t1"]["median_ms"]
                / cand_metrics["encode_visual_t1"]["median_ms"],
                2,
            ),
            "pop_t8_e2e_speedup": round(
                base_metrics["pop_t8_e2e"]["median_ms"]
                / cand_metrics["pop_t8_e2e"]["median_ms"],
                2,
            ),
            "stream_per_frame_speedup": round(
                base_metrics["stream_per_frame"]["median_ms"]
                / cand_metrics["stream_per_frame"]["median_ms"],
                2,
            ),
            "encode_visual_17f_speedup": round(
                base_metrics["encode_visual_17f"]["median_ms"]
                / cand_metrics["encode_visual_17f"]["median_ms"],
                2,
            ),
        }

        target_ms = 20.0 if height <= 480 and width <= 832 else 60.0
        gate = {
            "target_ms": target_ms,
            "pop_t1_e2e_median_ms": cand_metrics["pop_t1_e2e"]["median_ms"],
            "pop_t1_gpu_median_ms": cand_metrics["pop_t1_gpu"]["median_ms"],
            "encode_visual_t1_median_ms": cand_metrics["encode_visual_t1"]["median_ms"],
            "stream_per_frame_median_ms": cand_metrics["stream_per_frame"]["median_ms"],
            "pop_t8_per_frame_median_ms": cand_metrics["pop_t8_e2e_per_frame"][
                "median_ms"
            ],
            "retained_bytes_after_dispose": cand_metrics["memory"][
                "retained_bytes_after_dispose"
            ],
            "passed": (
                cand_metrics["pop_t1_e2e"]["median_ms"] < target_ms
                and cand_metrics["encode_visual_t1"]["median_ms"] < target_ms
                and cand_metrics["stream_per_frame"]["median_ms"] < target_ms
                and cand_metrics["pop_t8_e2e_per_frame"]["median_ms"] < target_ms
                and cand_metrics["memory"]["retained_bytes_after_dispose"] == 0
            ),
        }
        all_targets_met = all_targets_met and gate["passed"]
        report["targets_check"][key] = gate

        entry = {
            "baseline": base_metrics,
            "candidate": cand_metrics,
            "speedups": speedups,
            "accuracy_vs_baseline": {
                "progressive_moments_all": prog_err,
                "progressive_moments_mean": prog_mean_err,
                "progressive_moments_logvar": prog_logvar_err,
                "encode_visual_t1_latent": vis_t1_err,
                "encode_visual_moments_17f": vis_m17_err,
            },
        }
        report["resolutions"][key] = entry
        print(
            json.dumps(
                {
                    "canvas": key,
                    "baseline_pop_t1_ms": base_metrics["pop_t1_e2e"]["median_ms"],
                    "candidate_pop_t1_ms": cand_metrics["pop_t1_e2e"]["median_ms"],
                    "baseline_vis_t1_ms": base_metrics["encode_visual_t1"]["median_ms"],
                    "candidate_vis_t1_ms": cand_metrics["encode_visual_t1"][
                        "median_ms"
                    ],
                    "baseline_stream_ms_per_frame": base_metrics["stream_per_frame"][
                        "median_ms"
                    ],
                    "candidate_stream_ms_per_frame": cand_metrics["stream_per_frame"][
                        "median_ms"
                    ],
                    "speedups": speedups,
                    "moments_rrmse": prog_err["rrmse"],
                    "gate_passed": gate["passed"],
                },
                indent=2,
            ),
            flush=True,
        )

    if args.profile:
        report["profiler_top_cuda_ops"] = {}
        for width, height in canvas_list[:2]:
            key = f"{width}x{height}"
            vid = make_deterministic_video(17, height, width)
            report["profiler_top_cuda_ops"][key] = capture_profiler_trace(
                codecs, vid, args.output / "traces", key, device
            )

    report["all_targets_met"] = all_targets_met
    out_file = args.output / "h100_fast_encoder_report.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"\nSaved complete report to {out_file} (all_targets_met={all_targets_met})",
        flush=True,
    )
    return 0 if all_targets_met else 2


if __name__ == "__main__":
    raise SystemExit(main())
