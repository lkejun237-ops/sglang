# SPDX-License-Identifier: Apache-2.0
"""Trained H3 source Encoder A/B; no DiT or Decoder performance claims.

Run with python (one GPU) or torchrun --nproc-per-node=2 (SP2 tile sharding).
The input is packed THWC RGB bytes. Default 34 frames cover startup 1/2/2
and one subsequent 3/2 scheduler group. Timing and profiling are separate.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
    AutoencoderKL,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive import (
    CausalEncoderCache,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_cnn import (
    EncoderFCN3D,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
    ZingH3Codecs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
    H3ProgressiveEncoderSession,
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_encoder(root, device):
    # Match MiniMaxH3VideoVAE's fixed EncoderFCN3D, without constructing its
    # unused 36-layer reference Decoder. Strict state loading checks all shapes.
    model = nn.Module()
    with torch.device("meta"):
        model.encoder = EncoderFCN3D(
            ch=128,
            ch_mult=[1, 2, 2, 4, 4, 8],
            space_down=[2, 2, 2, 2, 1, 1],
            time_down=[1, 2, 2, 1, 1, 1],
            num_res_blocks=2,
            in_channels=3,
            z_channels=24,
            double_z=True,
            padding_mode="reflect",
            causal=True,
            use_t_isolated_gn=True,
        )
        model.quant_conv = nn.Conv3d(48, 48, 1)
    weights = root / "video_vae/source/model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as f:
        state = {
            k: f.get_tensor(k)
            for k in f.keys()
            if k.startswith(("encoder.", "quant_conv."))
        }
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device=device, dtype=torch.bfloat16).eval()
    model.encoder_tiling = True
    model.stack_tiling = model.parallel_tiling = False
    model.tile_size, model.tile_overlap_min, model.vae_ratio = 256, 64, 16
    model._blend_weight_cache = {}
    for name in ("split_tiles", "blend", "_assemble_tiles", "_local_tile_indices"):
        setattr(model, name, MethodType(getattr(AutoencoderKL, name), model))
    codecs = object.__new__(ZingH3Codecs)
    codecs.modules = {"video_vae": model}
    codecs.vae_frames_per_chunk, codecs.vae_latents_per_chunk, codecs.temporal_ratio = (
        17,
        5,
        4,
    )
    codecs.runtime = json.loads((root / "zing_config.json").read_text())
    assert codecs.runtime["vae_config"]["encode_mode"] == "performance"
    assert codecs.runtime["vae_config"]["encode_dtype"] == "bfloat16"
    return codecs, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-rgb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=34)
    parser.add_argument("--tile-batch-size", type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert args.frames >= 34 and args.frames % 17 == 0
    assert args.tile_batch_size is None or args.tile_batch_size > 0
    torch.set_grad_enabled(False)
    rank, size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    group = None
    if size > 1:
        assert size == 2, "This acceptance harness validates SP2 only"
        dist.init_process_group("nccl")
        cpu_group = dist.new_group(backend="gloo")
        group = SimpleNamespace(
            world_size=size,
            rank_in_group=rank,
            ranks=list(range(size)),
            cpu_group=cpu_group,
            device_group=dist.group.WORLD,
        )
    output = args.output / f"rank{rank}"
    output.mkdir(parents=True, exist_ok=False)
    codecs, weights = load_encoder(args.model, device)
    model = codecs.modules["video_vae"]
    originals = {
        name: (p.data_ptr(), p.stride()) for name, p in model.named_parameters()
    }
    os.environ["MINIMAX_H3_PROGRESSIVE_ENCODER_TILING"] = (
        "parallel" if group else "replicated"
    )
    key = "MINIMAX_H3_PROGRESSIVE_ENCODER_TILE_BATCH_SIZE"
    if args.tile_batch_size is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = str(args.tile_batch_size)
    frame_bytes = args.width * args.height * 3
    assert args.source_rgb.stat().st_size % frame_bytes == 0
    total = args.source_rgb.stat().st_size // frame_bytes
    assert total >= args.frames
    source = np.memmap(
        args.source_rgb,
        dtype=np.uint8,
        mode="r",
        shape=(total, args.height, args.width, 3),
    )
    pixels = (
        torch.from_numpy(np.array(source[: args.frames]))
        .permute(3, 0, 1, 2)
        .contiguous()
    )
    scope = dict(
        source=str(args.source_rgb),
        source_sha256=digest(args.source_rgb),
        weights=str(weights),
        weights_sha256=digest(weights),
        width=args.width,
        height=args.height,
        frames=args.frames,
        sp=size,
        tp=1,
        tile_batch_size=args.tile_batch_size,
        repeats=3,
        dtype="bfloat16",
        runtime_policy=codecs.runtime["vae_config"],
        device=torch.cuda.get_device_name(),
        cc=torch.cuda.get_device_capability(),
        memory=torch.cuda.get_device_properties(device).total_memory,
        sm=torch.cuda.get_device_properties(device).multi_processor_count,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        scope="Source Encoder only; no DiT, references, Decoder, or end-to-end performance claim",
    )
    write(output / "scope.json", scope)
    records, numerical, baseline = [], [], None

    def run(save=False):
        if group:
            # Hashing, file export and warmup can finish at different times.
            # Synchronize outside timing, before the bounded cache rendezvous.
            dist.barrier(group=cpu_group)
        cache = CausalEncoderCache(
            model, encode_mode="performance", encode_dtype="bfloat16", group=group
        )
        stream = H3ProgressiveEncoderSession(
            codecs, capacity_frames=args.frames, cache=cache
        )
        stream.append(pixels, frame_start=0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        rows, values = [], []
        for i in range(args.frames // 17 * 3):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            host = time.perf_counter()
            with torch.profiler.record_function(f"EncoderSource/{i}"):
                value = stream.pop()
            end.record()
            end.synchronize()
            rows.append(
                dict(
                    call=i,
                    latent_start=value.start,
                    latent_count=value.end - value.start,
                    gpu_ms=start.elapsed_time(end),
                    wall_ms=1000 * (time.perf_counter() - host),
                    cache_bytes=cache.last_execution["cache_bytes"],
                    execution=cache.last_execution,
                )
            )
            if save:
                values.append(value.values.cpu())
        stream.finish()
        stream.dispose()
        assert stream.state_usage().retained_bytes == 0
        return rows, values, torch.cuda.max_memory_allocated()

    for label, enabled in (("ncdhw", "0"), ("ndhwc", "1")):
        os.environ["MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST"] = enabled
        torch.cuda.synchronize()
        start = time.perf_counter()
        codecs.prepare_visual_encoder_weights()
        codecs.warmup_encoder()
        torch.cuda.synchronize()
        prep_ms = 1000 * (time.perf_counter() - start)
        assert originals == {
            name: (p.data_ptr(), p.stride()) for name, p in model.named_parameters()
        }
        weight_bytes = sum(
            w.numel() * w.element_size()
            for w in model._progressive_encoder_weights.values()
        )
        warmup_rows, _, warmup_peak = run()
        write(
            output / f"{label}-warmup.json",
            dict(
                calls=warmup_rows,
                peak_allocated=warmup_peak,
                excluded_from_formal_timing=True,
            ),
        )
        for repeat in range(3):
            rows, values, peak = run(save=repeat == 0)
            if repeat == 0:
                if baseline is None:
                    baseline = values
                else:
                    for actual, expected in zip(values, baseline):
                        delta = actual.float() - expected.float()
                        numerical.append(
                            dict(
                                equal=torch.equal(actual, expected),
                                finite=bool(torch.isfinite(actual).all()),
                                max_abs=delta.abs().max().item(),
                                rrmse=(
                                    delta.square().sum()
                                    / expected.float().square().sum().clamp_min(1e-30)
                                )
                                .sqrt()
                                .item(),
                            )
                        )
                    write(output / "numerics.json", numerical)
                torch.save(values, output / f"{label}-latents.pt")
            row = dict(
                layout=label,
                repeat=repeat,
                calls=rows,
                peak_allocated=peak,
                peak_reserved=torch.cuda.max_memory_reserved(),
                weight_layout_bytes=weight_bytes,
                preparation_ms=prep_ms,
            )
            records.append(row)
            write(output / "timing.json", records)
            print(
                "TIMING",
                rank,
                label,
                repeat,
                [round(c["gpu_ms"], 3) for c in rows],
                flush=True,
            )
        if args.profile:
            conv = F.conv3d
            strides = []

            def observed(x, weight, *a, **kw):
                y = conv(x, weight, *a, **kw)
                strides.append(
                    dict(
                        shape=list(x.shape),
                        input_stride=list(x.stride()),
                        output_stride=list(y.stride()),
                        input_channels_last=x.is_contiguous(
                            memory_format=torch.channels_last_3d
                        ),
                        output_channels_last=y.is_contiguous(
                            memory_format=torch.channels_last_3d
                        ),
                    )
                )
                return y

            with (
                patch.object(F, "conv3d", observed),
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                ) as prof,
            ):
                run()
            prof.export_chrome_trace(str(output / f"{label}.trace.json"))
            write(output / f"{label}-strides.json", strides)
    passed = bool(numerical) and all(x["equal"] and x["finite"] for x in numerical)
    write(
        output / "COMPLETE.json",
        dict(passed=passed, exact_numerics=passed, parameter_storage_unchanged=True),
    )
    if group:
        dist.destroy_process_group()
    assert passed, "Exact BF16 numerical gate failed; original data retained"


if __name__ == "__main__":
    main()
