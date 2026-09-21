# SPDX-License-Identifier: Apache-2.0
"""Run and compare real-checkpoint Zing reference and SGLang inference.

Run ``reference`` and ``sglang`` in separate processes so the 37B model and its
codec modules are released between implementations. ``compare`` consumes their
raw FP32 tensors and writes one machine-readable validation report.
"""

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _init_reference_distributed(sp_size):
    if sp_size == 1:
        return torch.device("cuda"), 0, None
    from runtime.distributed import (
        get_sp_process_group,
        launch_distributed_job,
    )

    launch_distributed_job(backend="nccl", sp_size=sp_size)
    local_rank = int(os.environ["LOCAL_RANK"])
    return torch.device(f"cuda:{local_rank}"), local_rank, get_sp_process_group()


def _hardware(device=None):
    result = {
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device() if device is None else device
        free, total = torch.cuda.mem_get_info(device)
        props = torch.cuda.get_device_properties(device)
        result.update(
            gpu=props.name,
            compute_capability=f"{props.major}.{props.minor}",
            total_memory_mib=total // 2**20,
            free_memory_mib=free // 2**20,
        )
    return result


def _load_request(path):
    request = json.loads(Path(path).read_text())
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("Validation requires one two-message Zing request")
    prompt = next(
        item["text"] for item in messages[0]["content"] if item.get("type") == "text"
    )
    return request, prompt


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_commit(path, environment_name):
    """Report an unpacked source archive's recorded commit without requiring .git."""
    configured = os.environ.get(environment_name)
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _move_reference_inputs(inputs, device):
    def move(values):
        return {key: value.to(device) for key, value in values.items()}

    inputs.positive = move(inputs.positive)
    inputs.visual = move(inputs.visual)
    inputs.audio = move(inputs.audio)
    if inputs.source_visual is not None:
        inputs.source_visual = inputs.source_visual.to(device)
    return inputs


def run_reference(args):
    minwm_root = Path(args.minwm_root).resolve()
    sys.path.insert(0, str(minwm_root / "src"))
    from configs.configuration import PretrainedConfig
    from dataloader.processors.h3_packed import H3PackedProcessor
    from message_inference import write_generation_output
    from omegaconf import OmegaConf
    from pipeline import GenerationOutput, PipelineBase
    from runtime.seed import set_seed

    device, local_rank, sp_group = _init_reference_distributed(args.sp_size)
    request, _ = _load_request(args.request)
    config = PretrainedConfig.from_pretrained(args.config)
    config = OmegaConf.merge(
        config,
        OmegaConf.from_dotlist(
            [
                "pipeline=UnifiedH3InferencePipeline",
                "causal=true",
                f"sample_solver={args.solver}",
                f"sampling_steps={args.sigma_points}",
                f"guidance_scale={args.guidance_scale}",
                "return_latents=true",
                f"low_memory={str(args.low_memory).lower()}",
                "lora_config.enabled=false",
                f"inference_fsdp_config.shard_with_sp={str(args.sp_size > 1).lower()}",
                "generator_ckpt=null",
                f"pretrained_dir={args.components}",
                f"generator_config.checkpoint_dir={args.components}/transformer_ref",
            ]
            + (
                ["denoising_step_list=" + json.dumps(args.denoising_step_list)]
                if args.solver == "few_step"
                else []
            )
        ),
    )
    OmegaConf.resolve(config)

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    input_cache = destination / "reference_inputs.pt"
    condition_cache = destination / "reference_condition.pt"

    # The CPU path remains useful for memory-constrained GPUs. High-memory
    # devices can run the conditioner on CUDA so wall-clock comparisons include
    # the normal resident-device path.
    set_seed(args.seed)
    condition_started = time.perf_counter()
    processor = H3PackedProcessor(config)
    condition_seconds = 0.0
    if local_rank == 0:
        condition_device = (
            device if args.condition_device == "cuda" else torch.device("cpu")
        )
        inputs = processor.prepare_inference(request, device=condition_device)
        _sync()
        condition_seconds = time.perf_counter() - condition_started
        temporary = input_cache.with_suffix(".tmp")
        torch.save(inputs, temporary)
        temporary.replace(input_cache)
        torch.save(
            {key: value.detach().cpu() for key, value in inputs.positive.items()},
            condition_cache,
        )
        if args.low_memory:
            processor._condition_input = None
            processor._condition_encoder = None
            gc.collect()
    if sp_group is not None:
        import torch.distributed as dist

        dist.barrier(group=sp_group)
    if local_rank != 0:
        inputs = torch.load(input_cache, map_location="cpu", weights_only=False)
    inputs = _move_reference_inputs(inputs, device)

    started = time.perf_counter()
    pipeline = PipelineBase.from_pretrained(
        config,
        args.checkpoint,
        device,
        low_memory=args.low_memory,
        sp_group=sp_group,
    )
    pipeline.processor = processor
    load_seconds = time.perf_counter() - started
    set_seed(args.seed)
    _sync()
    started = time.perf_counter()
    visual = torch.randn(
        (1, processor.visual_channels, *inputs.visual_shape),
        device=device,
        dtype=torch.float32,
    )
    audio = (
        None
        if inputs.audio_shape is None
        else torch.randn((1, *inputs.audio_shape), device=device, dtype=torch.float32)
    )
    pipeline._validate_inputs(inputs)
    visual, audio = pipeline._sample(inputs, visual, audio)
    video, waveform = processor.decode_inference(visual, audio, inputs, device=device)
    output = GenerationOutput(
        video=video,
        fps=1.0 if inputs.output_type == "image" else float(processor.target_fps),
        audio=waveform,
        audio_sample_rate=(
            processor.audio_sample_rate if waveform is not None else None
        ),
        visual_latents=visual,
        audio_latents=audio,
    )
    _sync()
    generation_seconds = time.perf_counter() - started
    peak_memory_mib = torch.cuda.max_memory_allocated(device) // 2**20
    if sp_group is not None:
        import torch.distributed as dist

        metrics = torch.tensor(
            [
                condition_seconds,
                load_seconds,
                generation_seconds,
                peak_memory_mib,
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX, group=sp_group)
        condition_seconds, load_seconds, generation_seconds, peak_memory_mib = (
            metrics.tolist()
        )
    if local_rank != 0:
        dist.barrier(group=sp_group)
        return
    torch.save(
        {
            "frames": output.video.detach().cpu(),
            "visual_latents": output.visual_latents.detach().cpu(),
            "audio_latents": (
                None
                if output.audio_latents is None
                else output.audio_latents.detach().cpu()
            ),
            "audio": (None if output.audio is None else output.audio.detach().cpu()),
            "audio_sample_rate": output.audio_sample_rate,
        },
        destination / "reference.pt",
    )
    media = write_generation_output(output, destination, "reference")
    _write_json(
        destination / "reference.json",
        {
            "implementation": "minWM-reference",
            "commit": _source_commit(args.minwm_root, "ZING_COMMIT"),
            "hardware": _hardware(device),
            "sp_size": args.sp_size,
            "seed": args.seed,
            "sigma_points": args.sigma_points,
            "solver": args.solver,
            "denoising_step_list": (
                args.denoising_step_list if args.solver == "few_step" else None
            ),
            "low_memory": args.low_memory,
            "condition_device": args.condition_device,
            "guidance_scale": args.guidance_scale,
            "condition_seconds": condition_seconds,
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "peak_memory_mib": peak_memory_mib,
            "media": str(media),
        },
    )
    if sp_group is not None:
        dist.barrier(group=sp_group)


def configure_execution(args):
    """Apply launch options before workers start; never read them from weights.

    Keep the legacy environment names for compatibility with deployed workers.
    """
    os.environ["SGLANG_ZING_EXPERIMENTAL_ATTENTION"] = args.dit_attention_backend
    os.environ["SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES"] = (
        "1" if args.sp_preserve_projection_shapes else "0"
    )
    os.environ["SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN"] = (
        "1" if args.compile_adaln else "0"
    )


def _require_rank_memory(metrics, sp_size):
    snapshots = metrics.get("memory_snapshots", {})
    for rank in range(sp_size):
        for phase in ("load", "request"):
            name = f"rank_{rank}_{phase}_peak"
            value = snapshots.get(name, {})
            for field in ("peak_allocated_mb", "peak_reserved_mb"):
                peak = value.get(field)
                if (
                    not isinstance(peak, (int, float))
                    or not np.isfinite(peak)
                    or peak < 0
                ):
                    raise RuntimeError(
                        f"Missing or invalid per-rank memory: {name}.{field}"
                    )


def run_sglang(args):
    configure_execution(args)
    if args.collect_rank_memory:
        os.environ["SGLANG_DIFFUSION_COLLECT_RANK_MEMORY"] = "1"
        os.environ["SGLANG_DIFFUSION_STAGE_LOGGING"] = "1"
    else:
        os.environ.pop("SGLANG_DIFFUSION_COLLECT_RANK_MEMORY", None)
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    if args.warmup_runs < 0 or args.measured_runs < 1:
        raise ValueError("warmup_runs must be non-negative and measured_runs positive")
    request, prompt = _load_request(args.request)
    target_output = next(
        item["output"] for item in request["messages"][1]["content"] if "output" in item
    )
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    use_fsdp = args.dit_placement == "fsdp"
    use_layerwise_offload = args.dit_placement == "layerwise"
    generator = DiffGenerator.from_pretrained(
        model_path=args.model,
        num_gpus=args.sp_size,
        sp_degree=args.sp_size,
        ulysses_degree=args.sp_size,
        ring_degree=1,
        encoder_parallel=args.encoder_parallel,
        local_mode=True,
        attention_backend="torch_sdpa",
        use_fsdp_inference=use_fsdp,
        dit_layerwise_offload=use_layerwise_offload,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=args.text_encoder_offload,
        layerwise_offload_components=(
            ["text_encoder", "dit"]
            if use_layerwise_offload
            else (["text_encoder"] if args.text_encoder_offload else [])
        ),
        performance_mode=("memory" if use_layerwise_offload else "speed"),
        pin_cpu_memory=False,
        vae_cpu_offload=args.vae_cpu_offload,
        warmup_mode="off",
    )
    load_seconds = time.perf_counter() - started
    try:

        def generate(run_name, *, save_output, profile=False):
            generated = generator.generate(
                sampling_params_kwargs={
                    "prompt": prompt,
                    "num_frames": target_output["frames"],
                    "height": target_output["height"],
                    "width": target_output["width"],
                    "profile": profile,
                    "profile_all_stages": profile,
                    "zing_request": request,
                    "zing_solver": args.solver,
                    "zing_denoising_step_list": (
                        args.denoising_step_list if args.solver == "few_step" else None
                    ),
                    "zing_return_latents": True,
                    "num_inference_steps": args.sigma_points,
                    "guidance_scale": args.guidance_scale,
                    "seed": args.seed,
                    "save_output": save_output,
                    "return_file_paths_only": False,
                    "output_path": str(destination),
                    "output_file_name": run_name,
                }
            )

            if args.collect_rank_memory:
                _require_rank_memory(generated.metrics, args.sp_size)
            return generated

        warmup_seconds = []
        warmup_metrics = []
        for index in range(args.warmup_runs):
            started = time.perf_counter()
            warmup_result = generate(f"sglang-warmup-{index}", save_output=False)
            warmup_seconds.append(time.perf_counter() - started)
            warmup_metrics.append(warmup_result.metrics)
            del warmup_result
        generation_seconds_all = []
        results = []
        for index in range(args.measured_runs):
            started = time.perf_counter()
            results.append(generate(f"sglang-{index}", save_output=True))
            generation_seconds_all.append(time.perf_counter() - started)
            if args.save_all_raw:
                item = results[-1]
                torch.save(
                    {
                        "visual_latents": torch.as_tensor(item.trajectory_latents)
                        .detach()
                        .cpu(),
                        "audio": None
                        if item.audio is None
                        else torch.as_tensor(item.audio).detach().cpu(),
                    },
                    destination / f"sglang-{index}-raw.pt",
                )
        profile_seconds = None
        result = results[0]
        generation_seconds = statistics.median(generation_seconds_all)
        if result is None or result.trajectory_latents is None:
            raise RuntimeError("SGLang returned no Zing result or final latent")
        sample = result.samples
        if isinstance(sample, (tuple, list)) and len(sample) == 2:
            sample = sample[0]
        frames = torch.as_tensor(sample).detach().cpu()
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)
        latents = torch.as_tensor(result.trajectory_latents).detach().cpu()
        torch.save(
            {
                "frames": frames,
                "visual_latents": latents,
                "audio": (
                    None
                    if result.audio is None
                    else torch.as_tensor(result.audio).detach().cpu()
                ),
            },
            destination / "sglang.pt",
        )
        _write_json(
            destination / "sglang.json",
            {
                "implementation": "sglang",
                "commit": _source_commit(args.sglang_root, "SGLANG_COMMIT"),
                "hardware": _hardware(),
                "sp_size": args.sp_size,
                "dit_attention_backend": args.dit_attention_backend,
                "sp_preserve_projection_shapes": args.sp_preserve_projection_shapes,
                "compile_adaln": args.compile_adaln,
                "dit_placement": args.dit_placement,
                "encoder_parallel": args.encoder_parallel,
                "vae_cpu_offload": args.vae_cpu_offload,
                "text_encoder_offload": args.text_encoder_offload,
                "seed": args.seed,
                "sigma_points": args.sigma_points,
                "solver": args.solver,
                "denoising_step_list": (
                    args.denoising_step_list if args.solver == "few_step" else None
                ),
                "guidance_scale": args.guidance_scale,
                "load_seconds": load_seconds,
                "generation_seconds": generation_seconds,
                "generation_seconds_all": generation_seconds_all,
                "profile_seconds_separate": profile_seconds,
                "profile_included_in_measurements": False,
                "correctness_diagnostics_enabled": bool(
                    os.environ.get("SGLANG_ZING_DIAGNOSTICS_DIR")
                ),
                "request": request,
                "finite": {
                    "frames": bool(torch.isfinite(frames).all()),
                    "visual_latents": bool(torch.isfinite(latents).all()),
                    "audio": result.audio is not None
                    and bool(torch.isfinite(torch.as_tensor(result.audio)).all()),
                },
                "shapes": {
                    "frames": list(frames.shape),
                    "visual_latents": list(latents.shape),
                    "audio": None
                    if result.audio is None
                    else list(torch.as_tensor(result.audio).shape),
                },
                "warmup_seconds": warmup_seconds,
                "warmup_metrics": warmup_metrics,
                "collect_rank_memory": args.collect_rank_memory,
                "peak_memory_mib": result.peak_memory_mb,
                "metrics": result.metrics,
                "measured_metrics": [item.metrics for item in results],
                "usage": getattr(result, "usage", None),
                "media": result.output_file_path,
            },
        )
        if args.profile_run:
            started = time.perf_counter()
            profile_result = generate("sglang-profile", save_output=True, profile=True)
            profile_seconds = time.perf_counter() - started
            report_path = destination / "sglang.json"
            report = json.loads(report_path.read_text())
            report["profile_seconds_separate"] = profile_seconds
            report["profile_metrics"] = profile_result.metrics
            _write_json(report_path, report)
    finally:
        generator.shutdown()


def _comparison(actual, expected):
    actual, expected = actual.float(), expected.float()
    if actual.shape != expected.shape:
        return {
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    error = (actual - expected).abs()
    denom = expected.square().mean().sqrt().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten().double(), expected.flatten().double(), dim=0
    )
    return {
        "shape_match": True,
        "shape": list(actual.shape),
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        # torch.quantile rejects tensors above 2**24 elements on CUDA-enabled
        # builds, even when the tensor is on the CPU. Full-resolution video
        # comparisons exceed that limit, while NumPy's partition path does not.
        "p99_abs_error": float(np.quantile(error.numpy(), 0.99)),
        "relative_rmse": float(error.square().mean().sqrt() / denom),
        "cosine_similarity": float(cosine),
    }


def _optional_comparison(actual, expected):
    if actual is None or expected is None:
        return {
            "present_match": actual is None and expected is None,
            "actual_present": actual is not None,
            "expected_present": expected is not None,
        }
    return {"present_match": True, **_comparison(actual, expected)}


def _canonical_audio(value):
    if value is None:
        return None
    value = torch.as_tensor(value)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(
            f"Expected audio [channels, samples], got {tuple(value.shape)}"
        )
    return value


def _quality(frames):
    frames = frames.float()
    if frames.ndim != 5:
        raise ValueError(f"Expected BCTHW frames, got {tuple(frames.shape)}")
    temporal = (frames[:, :, 1:] - frames[:, :, :-1]).abs()
    dx = frames[..., 1:] - frames[..., :-1]
    dy = frames[..., 1:, :] - frames[..., :-1, :]
    luminance = 0.2126 * frames[:, 0] + 0.7152 * frames[:, 1] + 0.0722 * frames[:, 2]
    duplicate_fraction = 0.0
    if temporal.numel():
        per_pair = temporal.permute(0, 2, 1, 3, 4).flatten(2).amax(2)
        duplicate_fraction = float((per_pair < 1 / 255).float().mean())
    return {
        "finite": bool(torch.isfinite(frames).all()),
        "minimum": float(frames.min()),
        "maximum": float(frames.max()),
        "mean": float(frames.mean()),
        "std": float(frames.std()),
        "clipped_fraction": float(((frames <= 0) | (frames >= 1)).float().mean()),
        "temporal_mean_abs_delta": (
            float(temporal.mean()) if temporal.numel() else 0.0
        ),
        "temporal_duplicate_fraction": duplicate_fraction,
        "horizontal_gradient_mean": float(dx.abs().mean()),
        "vertical_gradient_mean": float(dy.abs().mean()),
        "luminance_mean": float(luminance.mean()),
        "luminance_std": float(luminance.std()),
    }


def _quality_delta(actual, expected):
    return {
        key: actual[key] - expected[key]
        for key in actual
        if isinstance(actual[key], float) and key in expected
    }


def _summarize(values):
    values = values.double()
    return {
        "minimum": float(values.min()),
        "mean": float(values.mean()),
        "maximum": float(values.max()),
    }


def _frame_quality(actual, expected):
    """Compare aligned BCTHW videos frame by frame in the normalized [0, 1] range."""
    actual, expected = actual.float(), expected.float()
    if actual.shape != expected.shape:
        return {
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    if actual.ndim != 5:
        raise ValueError(f"Expected BCTHW frames, got {tuple(actual.shape)}")

    difference = actual - expected
    reduction_dims = (0, 1, 3, 4)
    mae = difference.abs().mean(dim=reduction_dims)
    mse = difference.square().mean(dim=reduction_dims)
    rmse = mse.sqrt()
    psnr = -10.0 * torch.log10(mse.clamp_min(torch.finfo(mse.dtype).tiny))

    # Compute the standard SSIM formula over local 11x11 uniform windows. Process
    # one video frame at a time to keep peak host memory small for 896x512 output.
    c1, c2 = 0.01**2, 0.03**2
    ssim = []
    for frame_index in range(actual.shape[2]):
        x = actual[:, :, frame_index]
        y = expected[:, :, frame_index]
        mu_x = torch.nn.functional.avg_pool2d(x, 11, stride=1)
        mu_y = torch.nn.functional.avg_pool2d(y, 11, stride=1)
        sigma_x = torch.nn.functional.avg_pool2d(x * x, 11, stride=1) - mu_x.square()
        sigma_y = torch.nn.functional.avg_pool2d(y * y, 11, stride=1) - mu_y.square()
        sigma_xy = torch.nn.functional.avg_pool2d(x * y, 11, stride=1) - mu_x * mu_y
        value = (
            (2 * mu_x * mu_y + c1)
            * (2 * sigma_xy + c2)
            / ((mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2))
        )
        ssim.append(value.mean())
    ssim = torch.stack(ssim)

    return {
        "shape_match": True,
        "mae": [float(value) for value in mae],
        "rmse": [float(value) for value in rmse],
        "psnr_db": [float(value) for value in psnr],
        "ssim_uniform_11x11": [float(value) for value in ssim],
        "summary": {
            "mae": _summarize(mae),
            "rmse": _summarize(rmse),
            "psnr_db": _summarize(psnr),
            "ssim_uniform_11x11": _summarize(ssim),
        },
    }


def compare(args):
    reference = torch.load(Path(args.output) / "reference.pt", weights_only=True)
    sglang = torch.load(Path(args.output) / "sglang.pt", weights_only=True)
    reference_frames = reference["frames"].permute(0, 2, 1, 3, 4)
    latent = _comparison(sglang["visual_latents"], reference["visual_latents"])
    frames = _comparison(sglang["frames"], reference_frames)
    audio = _optional_comparison(
        _canonical_audio(sglang.get("audio")),
        _canonical_audio(reference.get("audio")),
    )
    sglang_quality = _quality(sglang["frames"])
    reference_quality = _quality(reference_frames)
    report = {
        "latent": latent,
        "frames": frames,
        "audio": audio,
        "quality": sglang_quality,
        "reference_quality": reference_quality,
        "quality_delta": _quality_delta(sglang_quality, reference_quality),
        "per_frame": _frame_quality(sglang["frames"], reference_frames),
        "numeric_pass": bool(
            latent.get("shape_match")
            and latent.get("finite")
            and latent["max_abs_error"] <= args.latent_atol
            and latent["relative_rmse"] <= args.latent_rrmse
            and audio.get("present_match")
            and (
                not audio.get("actual_present")
                or (
                    audio.get("shape_match")
                    and audio.get("finite")
                    and audio["max_abs_error"] <= args.audio_atol
                    and audio["relative_rmse"] <= args.audio_rrmse
                )
            )
        ),
        "thresholds": {
            "latent_max_abs_error": args.latent_atol,
            "latent_relative_rmse": args.latent_rrmse,
            "audio_max_abs_error": args.audio_atol,
            "audio_relative_rmse": args.audio_rrmse,
        },
    }
    _write_json(Path(args.output) / "comparison.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["numeric_pass"]:
        raise SystemExit(1)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--request", required=True)
    common.add_argument("--output", required=True)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--sigma-points", type=int, default=50)
    common.add_argument("--solver", choices=("euler", "few_step"), default="euler")
    common.add_argument(
        "--denoising-step-list",
        type=int,
        nargs="+",
        default=[1000, 750, 500, 250],
    )
    common.add_argument("--guidance-scale", type=float, default=1.0)
    ref = sub.add_parser("reference", parents=[common])
    ref.add_argument("--minwm-root", required=True)
    ref.add_argument("--config", required=True)
    ref.add_argument("--checkpoint", required=True)
    ref.add_argument("--components", required=True)
    ref.add_argument("--sp-size", type=int, default=1)
    ref.add_argument(
        "--low-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ref.add_argument(
        "--condition-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    sglang = sub.add_parser("sglang", parents=[common])
    sglang.add_argument("--sglang-root", required=True)
    sglang.add_argument("--model", required=True)
    sglang.add_argument("--sp-size", type=int, default=1)
    sglang.add_argument(
        "--dit-placement",
        choices=("layerwise", "resident", "fsdp"),
        default="layerwise",
    )
    sglang.add_argument(
        "--vae-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    sglang.add_argument(
        "--text-encoder-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    sglang.add_argument(
        "--encoder-parallel", choices=("replicate", "auto", "fold"), default="replicate"
    )
    sglang.add_argument(
        "--dit-attention-backend",
        "--experimental-attention",
        dest="dit_attention_backend",
        choices=("torch_sdpa", "fa4"),
        default="fa4",
        help="Launch-time Zing DiT backend, independent of checkpoint weights (default: fa4)",
    )
    sglang.add_argument(
        "--sp-preserve-projection-shapes",
        "--experimental-stable-shapes",
        dest="sp_preserve_projection_shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep SP1 projection row counts across SP: replicate prefix and gather "
            "hidden rows before output projection (default: enabled)"
        ),
    )
    sglang.add_argument("--profile-run", action="store_true")
    sglang.add_argument(
        "--collect-rank-memory",
        action="store_true",
        help="Preserve load/request peaks per rank; includes CPU gathering overhead",
    )
    sglang.add_argument(
        "--compile-adaln",
        "--experimental-compiled-adaln",
        dest="compile_adaln",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Native compiled RMSNorm and modulation rounding (default: enabled)",
    )
    sglang.add_argument(
        "--save-all-raw",
        action="store_true",
        help="Archive every measured latent/audio result for repeatability checks",
    )
    sglang.add_argument("--warmup-runs", type=int, default=0)
    sglang.add_argument("--measured-runs", type=int, default=1)
    check = sub.add_parser("compare")
    check.add_argument("--output", required=True)
    check.add_argument("--latent-atol", type=float, default=0.03)
    check.add_argument("--latent-rrmse", type=float, default=0.01)
    check.add_argument("--audio-atol", type=float, default=0.1)
    check.add_argument("--audio-rrmse", type=float, default=0.05)
    return root


def main():
    args = parser().parse_args()
    if not torch.cuda.is_available() and args.mode != "compare":
        raise RuntimeError("Real-checkpoint validation requires CUDA")
    modes = {"reference": run_reference, "sglang": run_sglang, "compare": compare}
    modes[args.mode](args)


if __name__ == "__main__":
    main()
