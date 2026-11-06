# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint VAE ablations on identical pixels/latents, without DiT.

Component timings do not include file I/O or numerical comparison. Each case
uses a new stream. Retired workspace/Graph cells remain in historical reports only.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from vae_execution_matrix import cell_environment, distribution


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare(actual, reference, *, rgb=False):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError("Output shape or dtype differs from baseline")
    # Chunk large RGB tensors on CPU to keep diagnostics out of GPU timings.
    a, b = actual.flatten(), reference.flatten()
    total = equal = squared = reference_squared = absolute = maximum = 0
    for x, y in zip(a.split(1 << 20), b.split(1 << 20), strict=True):
        x, y = x.double(), y.double()
        if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise ValueError("Nonfinite output")
        d = x - y
        total += x.numel()
        equal += int((x == y).sum())
        squared += float(d.square().sum())
        reference_squared += float(y.square().sum())
        absolute += float(d.abs().sum())
        maximum = max(maximum, float(d.abs().max()))
    result = dict(
        shape=list(actual.shape),
        dtype=str(actual.dtype),
        finite=True,
        exact=equal == total,
        equal_fraction=equal / total,
        max_abs=maximum,
        mae=absolute / total,
        rmse=math.sqrt(squared / total),
        rrmse=math.sqrt(squared / max(reference_squared, 1e-300)),
    )
    if rgb:
        result["psnr_range_1_db"] = (
            None if squared == 0 else -10 * math.log10(squared / total)
        )
        result["psnr_exact_infinite"] = squared == 0
    return result


def memory():
    return dict(
        allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
    )


@torch.inference_mode()
def encoder(args):
    from benchmark_encoder_layout import load_encoder

    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive import (
        CausalEncoderCache,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
        H3ProgressiveEncoderSession,
    )

    codecs, weights = load_encoder(args.model, 0)
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_noise import (
        STREAMING_NOISE_VERSION,
    )

    codecs.va = SimpleNamespace(**codecs.runtime["vae_config"])
    model = codecs.modules["video_vae"]
    pixels = (
        torch.from_numpy(
            np.fromfile(args.source_rgb, dtype=np.uint8).reshape(
                args.frames, args.height, args.width, 3
            )
        )
        .permute(3, 0, 1, 2)
        .contiguous()
    )
    codecs.prepare_visual_encoder_weights()
    codecs.warmup_encoder()

    def run(capture=False):
        cache = CausalEncoderCache(
            model, encode_mode="performance", encode_dtype="bfloat16"
        )
        stream = H3ProgressiveEncoderSession(
            codecs, capacity_frames=args.frames, cache=cache
        )
        stream.append(pixels, frame_start=0)
        rows, outputs = [], []
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            for index in range(args.frames // 17 * 3):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                before = time.perf_counter()
                value = stream.pop()
                end.record()
                end.synchronize()
                rows.append(
                    dict(
                        index=index,
                        wall_ms=1000 * (time.perf_counter() - before),
                        cuda_ms=start.elapsed_time(end),
                        execution=cache.last_execution,
                    )
                )
                if capture:
                    outputs.append(value.values.detach().cpu())
            elapsed = time.perf_counter() - started
            stream.finish()
            return dict(wall_seconds=elapsed, calls=rows, memory=memory()), outputs
        finally:
            stream.dispose()

    reference = reference_latent = None
    for cell in args.cells or (
        "baseline",
        "encoder-norm",
        "encoder-copy",
        "encoder-all",
    ):
        os.environ.update(cell_environment(cell))
        out = args.output / cell
        out.mkdir()
        print("BEGIN", cell, flush=True)
        cold, values = run(capture=True)
        moments = torch.cat(values, dim=2)
        # Same indexed posterior noise and FP16 rounding as the source codec.
        latent = codecs.sample_streaming_visual_moments(
            moments.cuda(), latent_start=0, noise_version=STREAMING_NOISE_VERSION
        ).cpu()
        if reference is None:
            reference = moments
            reference_latent = latent
            torch.save(moments, args.output / "baseline-moments.pt")
            torch.save(latent, args.output / "baseline-latents.pt")
        comparison = compare(moments, reference)
        write(
            out / "numerics.json",
            dict(
                moments=comparison, normalized_latent=compare(latent, reference_latent)
            ),
        )
        if cell == "encoder-copy" and not comparison["exact"]:
            raise RuntimeError("Pure-copy optimization must be exact")
        warmup, _ = run()
        runs = [run()[0] for _ in range(args.repeats)]
        write(
            out / "timing.json",
            dict(
                cold_untimed_capture=cold,
                warmup=warmup,
                runs=runs,
                total_seconds=distribution([r["wall_seconds"] for r in runs]),
                weights=str(weights),
            ),
        )
        print("PASS", cell, comparison, flush=True)


@torch.inference_mode()
def decoder(args):
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3_decoder import (
        normalize_decoder_spec,
    )
    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.execution import (
        LynnBlockExecution,
        LynnExecutionOptions,
    )
    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.loader import (
        load_lynn_decoder,
    )

    spec = normalize_decoder_spec(
        json.loads(args.decoder_config.read_text()),
        video_decoder_backend=None,
        vae_decode_precision="fp32",
    )
    # Match the worker's loading context. Parameters need version counters for
    # packed-weight invalidation even when the measured forward uses inference.
    with torch.inference_mode(False), torch.no_grad():
        model = load_lynn_decoder(spec, device="cuda").eval()
    latents = torch.load(args.latents, map_location="cpu", weights_only=True).cuda()
    if latents.shape[:3] != (1, 24, args.frames // 17 * 5):
        raise ValueError("Frozen decoder input has unexpected latent shape")

    def run(capture=False):
        stream = model.new_stream(frame_limit=args.frames)
        rows, outputs, preclamp = [], [], []
        original = model.decode_step

        def capture_step(*values):
            result, state = original(*values)
            preclamp.append(result.detach().cpu())
            return result, state

        if capture:
            model.decode_step = capture_step
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            for index, value in enumerate(latents.split(1, 2)):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                before = time.perf_counter()
                result = stream.push(value)
                end.record()
                end.synchronize()
                counts = dict(model.execution.counts)
                rows.append(
                    dict(
                        index=index,
                        phase=index % model.protocol.phases,
                        wall_ms=1000 * (time.perf_counter() - before),
                        cuda_ms=start.elapsed_time(end),
                        frames=result.shape[2],
                        model_execution_counts=counts,
                    )
                )
                if capture:
                    outputs.append(result.detach().cpu())
            elapsed = time.perf_counter() - started
            stream.finish(args.frames)
            return dict(
                wall_seconds=elapsed,
                calls=rows,
                memory=memory(),
                model_execution_counts=dict(model.execution.counts),
            ), (outputs, preclamp)
        finally:
            stream.close()
            if capture:
                model.decode_step = original

    reference = reference_preclamp = None
    for cell in args.cells or (
        "baseline",
        "decoder-block",
    ):
        os.environ.update(cell_environment(cell))
        model.execution = LynnBlockExecution(LynnExecutionOptions.from_env())
        out = args.output / cell
        out.mkdir()
        print("BEGIN", cell, flush=True)
        cold, (values, raw_values) = run(capture=True)
        rgb = torch.cat(values, dim=2)
        raw = torch.cat(raw_values, dim=2)[:, :, : args.frames]
        if reference is None:
            reference = rgb
            reference_preclamp = raw
        comparison = compare(rgb, reference, rgb=True)
        uint8 = compare(
            (rgb * 255).clamp(0, 255).to(torch.uint8),
            (reference * 255).clamp(0, 255).to(torch.uint8),
        )
        write(
            out / "numerics.json",
            dict(
                preclamp_rgb=compare(raw, reference_preclamp, rgb=True),
                rgb_0_1=comparison,
                wire_uint8_truncation=uint8,
            ),
        )
        warmup, _ = run()
        runs = [run()[0] for _ in range(args.repeats)]
        steady = [r["wall_ms"] for run in runs for r in run["calls"] if r["index"] >= 3]
        write(
            out / "timing.json",
            dict(
                cold_untimed_capture=cold,
                warmup=warmup,
                runs=runs,
                total_seconds=distribution([r["wall_seconds"] for r in runs]),
                steady_push_ms=distribution(steady),
            ),
        )
        print("PASS", cell, comparison, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("encoder", "decoder"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--source-rgb", type=Path)
    parser.add_argument("--decoder-config", type=Path)
    parser.add_argument("--latents", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, default=34)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--cells", nargs="+", help="Explicit subset, starting with baseline"
    )
    args = parser.parse_args()
    allowed = {
        "baseline",
        *(
            ("encoder-norm", "encoder-copy", "encoder-all")
            if args.component == "encoder"
            else ("decoder-block",)
        ),
    }
    if args.cells is not None and (
        args.cells[0] != "baseline"
        or len(set(args.cells)) != len(args.cells)
        or not set(args.cells) <= allowed
    ):
        parser.error("Select unique cells of this component, starting with baseline")
    if args.frames < 34 or args.frames % 17 or args.repeats < 3:
        parser.error("Use at least 34 frames in complete 17-frame clips, and 3 repeats")
    if args.component == "encoder" and (args.model is None or args.source_rgb is None):
        parser.error("Encoder requires --model and --source-rgb")
    if args.component == "decoder" and (
        args.decoder_config is None or args.latents is None
    ):
        parser.error("Decoder requires --decoder-config and --latents")
    os.environ.update(cell_environment("baseline"))
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    write(
        args.output / "scope.json",
        dict(
            component=args.component,
            frames=args.frames,
            width=args.width,
            height=args.height,
            repeats=args.repeats,
            torch=str(torch.__version__),
            cuda=str(torch.version.cuda),
            gpu=str(torch.cuda.get_device_properties(0)),
            input_latents=str(args.latents),
            input_rgb=str(args.source_rgb),
        ),
    )
    (encoder if args.component == "encoder" else decoder)(args)
    write(args.output / "COMPLETE.json", dict(completed=True))


if __name__ == "__main__":
    main()
