# SPDX-License-Identifier: Apache-2.0
"""torchrun: exact progressive parity, then isolated 1/2/4 GPU encoder timing."""

import argparse
import copy
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from prepare_encoder_overlap import digest


def configure_encoder_policy(runtime, limit):
    """Return a frozen Encoder-only runtime; never mutate process-wide backends."""
    if limit is not None and (type(limit) is not int or limit != 1):
        raise ValueError("cudnn_benchmark_limit must be 1 or null")
    effective = copy.deepcopy(runtime)
    if limit is not None:
        effective["vae_config"]["cudnn_benchmark_limit"] = limit
    return effective


from encoder_execution_evidence import validate_encode_policy, validate_execution


def stream(
    codecs,
    pixels,
    group,
    pattern,
    capture,
    control_group=None,
    execution_failure_path=None,
):
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.parallel_encoder import (
        create_encoder_cache,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
        H3ProgressiveEncoderSession,
        expected_progressive_slices,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_noise import (
        STREAMING_NOISE_VERSION,
    )

    world_size = dist.get_world_size(group) if group is not None else None
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive import (
        CausalEncoderCache,
    )

    cache = (
        CausalEncoderCache(
            codecs.modules["video_vae"],
            tile_mode="replicated",
            encode_mode=codecs.runtime["vae_config"]["encode_mode"],
            encode_dtype=codecs.runtime["vae_config"]["encode_dtype"],
        )
        if group is None
        else create_encoder_cache(
            codecs.modules["video_vae"],
            group,
            control_group=control_group,
            encode_mode=codecs.runtime["vae_config"]["encode_mode"],
            encode_dtype=codecs.runtime["vae_config"]["encode_dtype"],
        )
    )
    session = H3ProgressiveEncoderSession(codecs, capacity_frames=34, cache=cache)
    values, sampled, intervals, slices, executions = [], [], [], [], []
    cursor = index = 0
    torch.cuda.synchronize()
    started = time.perf_counter()

    def drain():
        while session.ready:
            begin = time.perf_counter()
            item = session.pop()
            execution = copy.deepcopy(session.cache.last_execution or {})
            executions.append(execution)
            try:
                validate_encode_policy(execution, codecs.runtime["vae_config"])
                if group is not None:
                    validate_execution(execution, world_size)
            except RuntimeError as exc:
                if execution_failure_path is not None:
                    execution_failure_path.write_text(
                        json.dumps(
                            dict(error=str(exc), encoder_executions=executions),
                            indent=2,
                        )
                    )
                session.dispose()
                raise
            torch.cuda.synchronize()
            intervals.append(time.perf_counter() - begin)
            slices.append((item.start, item.end, item.source_dependency_end))
            if capture:
                values.append(item.values.cpu())
                sampled.append(
                    codecs.sample_streaming_visual_moments(
                        item.values,
                        latent_start=item.start,
                        noise_version=STREAMING_NOISE_VERSION,
                    ).cpu()
                )

    while cursor < pixels.shape[1]:
        count = min(pattern[index % len(pattern)], pixels.shape[1] - cursor)
        session.append(pixels[:, cursor : cursor + count], frame_start=cursor)
        cursor += count
        index += 1
        drain()
    session.finish()
    drain()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert slices == expected_progressive_slices(pixels.shape[1])
    stats = dict(
        seconds=elapsed,
        pop_seconds=intervals,
        slices=slices,
        state=vars(session.state_usage()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        encoder_execution=session.cache.last_execution,
        encoder_executions=executions,
        parallel_execution_passed=bool(executions) if group is not None else None,
    )
    session.dispose()
    return stats, (torch.cat(values, 2), torch.cat(sampled, 2)) if capture else None


def compare(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    delta = actual.float() - expected.float()
    return dict(
        passed=bool(torch.isfinite(actual).all() and torch.equal(actual, expected)),
        max_abs=delta.abs().max().item(),
        rrmse=(delta.square().sum() / expected.float().square().sum().clamp_min(1e-30))
        .sqrt()
        .item(),
    )


def initialize_runtime_imports():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        initialize_runtime_imports as initialize,
    )

    return initialize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-rgb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument(
        "--cudnn-benchmark-limit",
        type=int,
        default=None,
        help="Encoder-only optional policy override; omitted preserves the frozen runtime",
    )
    parser.add_argument(
        "--tile-mode",
        choices=("parallel", "auto"),
        default="parallel",
        help="Explicit existing tile policy; multi-rank fallback always fails acceptance",
    )
    args = parser.parse_args()
    ServerArgs, set_global_server_args = initialize_runtime_imports()
    # Change only the existing experimental tile-distribution selector. The
    # business encode mode, dtype and backend numerical policy remain frozen.
    os.environ["MINIMAX_H3_PROGRESSIVE_ENCODER_TILING"] = args.tile_mode
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    control_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=30))

    set_global_server_args(
        ServerArgs.from_kwargs(
            model_path=str(args.model),
            num_gpus=1,
            video_decoder_backend="native",
            attention_backend="torch_sdpa",
            warmup_mode="off",
        )
    )
    out = args.output / f"rank{dist.get_rank()}"
    out.mkdir(parents=True, exist_ok=False)
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        load_encoder,
    )

    codecs = load_encoder(args.model)
    original_runtime = copy.deepcopy(codecs.runtime)
    codecs.runtime = configure_encoder_policy(
        codecs.runtime, args.cudnn_benchmark_limit
    )
    frame_bytes = args.width * args.height * 3
    assert args.source_rgb.stat().st_size % frame_bytes == 0
    frames = args.source_rgb.stat().st_size // frame_bytes
    data = np.memmap(
        args.source_rgb,
        dtype=np.uint8,
        mode="r",
        shape=(frames, args.height, args.width, 3),
    )
    pixels = torch.from_numpy(np.array(data)).permute(3, 0, 1, 2)
    manifest = dict(
        world_size=dist.get_world_size(),
        execution_schema_version=2,
        requested_tile_mode=args.tile_mode,
        reference_tile_mode="replicated",
        frames=frames,
        width=args.width,
        height=args.height,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        runtime=codecs.runtime,
        original_runtime=original_runtime,
        model_config_sha256=digest(args.model / "zing_config.json"),
        requested_cudnn_benchmark_limit=args.cudnn_benchmark_limit,
        effective_cudnn_benchmark_limit=codecs.runtime["vae_config"].get(
            "cudnn_benchmark_limit"
        ),
        policy_baseline="Reference and candidate use this same effective Encoder runtime; no parity claim against a different benchmark_limit policy",
        tolerance={"atol": 0, "rtol": 0},
        source_sha256=digest(args.source_rgb),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    acceptance = []
    # Real-sized tiles, every EOS residue, cross-clip caches and packet patterns.
    for count in range(1, 35):
        reference_stats, expected = stream(codecs, pixels[:, :count], None, [1], True)
        for pattern in ([1], [3, 2, 7, 5], [17]):
            stats, actual = stream(
                codecs,
                pixels[:, :count],
                dist.group.WORLD,
                pattern,
                True,
                control_group,
                out / "failed-execution.json",
            )
            reference_policies = [
                x["encode_policy"] for x in reference_stats["encoder_executions"]
            ]
            candidate_policies = [
                x["encode_policy"] for x in stats["encoder_executions"]
            ]
            if reference_policies != candidate_policies:
                raise RuntimeError("Reference and candidate Encoder policies differ")
            row = dict(
                reference_encoder_executions=reference_stats["encoder_executions"],
                frames=count,
                pattern=pattern,
                slices=stats["slices"],
                moments=compare(actual[0], expected[0]),
                latent=compare(actual[1], expected[1]),
                encoder_executions=stats["encoder_executions"],
                parallel_execution_passed=stats["parallel_execution_passed"],
            )
            acceptance.append(row)
            (out / "acceptance.json").write_text(json.dumps(acceptance, indent=2))
            ok = torch.tensor(
                int(row["moments"]["passed"] and row["latent"]["passed"]), device="cuda"
            )
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if not ok.item():
                torch.save(
                    dict(actual=actual, expected=expected), out / "failed-tensors.pt"
                )
                raise RuntimeError(
                    "Parallel encoder failed exact acceptance; performance matrix is blocked"
                )
    warmup, _ = stream(
        codecs,
        pixels,
        dist.group.WORLD,
        [1, 3, 17, 2, 5],
        False,
        control_group,
        out / "failed-execution.json",
    )
    (out / "warmup.json").write_text(json.dumps(warmup, indent=2))
    runs = []
    for _ in range(args.measured_runs):
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        stats, _ = stream(
            codecs,
            pixels,
            dist.group.WORLD,
            [1, 3, 17, 2, 5],
            False,
            control_group,
            out / "failed-execution.json",
        )
        runs.append(stats)
        (out / "measured.json").write_text(json.dumps(runs, indent=2))
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        profile, _ = stream(
            codecs,
            pixels,
            dist.group.WORLD,
            [1, 3, 17, 2, 5],
            False,
            control_group,
            out / "failed-execution.json",
        )
    (out / "profile.json").write_text(json.dumps(profile, indent=2))
    prof.export_chrome_trace(str(out / "encoder.trace.json"))
    seconds = [r["seconds"] for r in runs]
    (out / "summary.json").write_text(
        json.dumps(
            dict(
                median=statistics.median(seconds),
                min=min(seconds),
                max=max(seconds),
                stable=(max(seconds) - min(seconds)) / statistics.median(seconds)
                <= 0.1,
                profile_seconds_separate=profile["seconds"],
                acceptance_passed=True,
                parallel_execution_passed=True,
                requested_tile_mode=args.tile_mode,
            ),
            indent=2,
        )
    )
    (out / "COMPLETE").write_text("Exact progressive parity and timing complete\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
