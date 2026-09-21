# SPDX-License-Identifier: Apache-2.0
"""Single-GPU decoder gate using an existing, unmodified H3 DiT latent."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.models.vaes.taeh3 import (
    SOURCE_COMMIT,
    WEIGHTS_SHA256,
    TAEH3Decoder,
    decoder_precision,
)
from sglang.multimodal_gen.runtime.models.vaes.taeh3.upstream import TAEHV


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frames", type=int, default=240)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "source_commit": SOURCE_COMMIT,
        "weights_sha256": WEIGHTS_SHA256,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "memory_mib": torch.cuda.get_device_properties(0).total_memory // 2**20,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "latent_path": args.latent,
        "dtype_results": {},
        "quality_accepted": False,
    }
    data = torch.load(args.latent, map_location="cpu", weights_only=True)
    latent = data["visual_latents"]
    report["latent_shape"] = list(latent.shape)
    report["latent_sha256"] = hashlib.sha256(
        latent.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    # Predeclared dtype gates; report failures without relaxing thresholds.
    for label, dtype, atol, rtol in [
        ("fp32", torch.float32, 2e-5, 2e-4),
        ("fp16", torch.float16, 2e-3, 2e-3),
        ("bf16", torch.bfloat16, 2e-2, 2e-2),
    ]:
        adapter = TAEH3Decoder(args.weights, WEIGHTS_SHA256).cuda().to(dtype)
        reference = TAEHV(args.weights, arch_name="taeh3").cuda().to(dtype).eval()
        x = latent.cuda().to(dtype)
        count = x.shape[2]
        padded = 5 if args.frames == count == 1 else max(7, count + (-(count + 3)) % 5)
        px = torch.cat((x, x[:, :, -1:].expand(-1, -1, padded - count, -1, -1)), 2)
        with torch.inference_mode():
            started = time.perf_counter()
            with decoder_precision(dtype):
                expected = reference.decode_video(
                    px.transpose(1, 2), show_progress_bar=False
                ).transpose(1, 2)[:, :, : args.frames]
            torch.cuda.synchronize()
            reference_seconds = time.perf_counter() - started
            actual = adapter.decode(x, args.frames)
            diff = (actual.float() - expected.float()).abs()
            match = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
            row = {
                "fp32_convolution": "ieee; TF32 disabled for both implementations",
                "atol": atol,
                "rtol": rtol,
                "upstream_match": match,
                "max_abs": diff.max().item(),
                "rmse": diff.square().mean().sqrt().item(),
                "reference_cold_seconds": reference_seconds,
            }
            del expected, diff, reference
            row["partitions"] = {}
            for partition in ([5], [1, 2, 2], [3, 7, 2]):
                stream = adapter.new_stream()
                parts, position, index = [], 0, 0
                while position < count:
                    end = min(position + partition[index % len(partition)], count)
                    parts.append(stream.push(x[:, :, position:end]))
                    position, index = end, index + 1
                parts.append(stream.finish(args.frames))
                joined = torch.cat(parts, 2)
                row["partitions"][str(partition)] = {
                    "exact": bool(torch.equal(joined, actual)),
                    "max_abs": (joined - actual).abs().max().item(),
                    "frames": joined.shape[2],
                }
                del joined, parts
            times, cuda_times = [], []
            adapter.decode(x, args.frames)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                start_event, end_event = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                torch.cuda.synchronize()
                started = time.perf_counter()
                start_event.record()
                result = adapter.decode(x, args.frames)
                end_event.record()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
                cuda_times.append(start_event.elapsed_time(end_event) / 1000)
                del result
            row.update(
                wall_seconds=times,
                cuda_seconds=cuda_times,
                median_seconds=statistics.median(times),
                fps=args.frames / statistics.median(times),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            )
            # Separate synchronized samples: do not contaminate the full-decode
            # timings above with per-chunk synchronization or instrumentation.
            chunk_rows = []
            for repeat in range(3):
                stream = adapter.new_stream()
                chunks = list(x.split(5, dim=2))
                for index in range(len(chunks) + 1):
                    torch.cuda.synchronize()
                    start_event.record()
                    started = time.perf_counter()
                    result = (
                        stream.push(chunks[index])
                        if index < len(chunks)
                        else stream.finish(args.frames)
                    )
                    end_event.record()
                    torch.cuda.synchronize()
                    chunk_rows.append(
                        {
                            "repeat": repeat,
                            "index": index,
                            "kind": (
                                "first"
                                if index == 0
                                else ("finish" if index == len(chunks) else "steady")
                            ),
                            "input_latents": (
                                chunks[index].shape[2] if index < len(chunks) else 0
                            ),
                            "output_frames": result.shape[2],
                            "wall_ms": (time.perf_counter() - started) * 1000,
                            "cuda_ms": start_event.elapsed_time(end_event),
                        }
                    )
                    del result
            row["stream_chunks"] = chunk_rows
            row["stream_chunk_summary"] = {}
            for kind in ("first", "steady", "finish"):
                samples = sorted(r["wall_ms"] for r in chunk_rows if r["kind"] == kind)
                if samples:
                    row["stream_chunk_summary"][kind] = {
                        "count": len(samples),
                        "p50_ms": statistics.median(samples),
                        "p95_ms": samples[
                            min(len(samples) - 1, (95 * len(samples) + 99) // 100 - 1)
                        ],
                    }
            if "frames" in data and list(data["frames"].shape) == list(actual.shape):
                native = data["frames"].float().cuda()
                mse = (actual.float() - native).square().mean()
                row["historical_native_psnr"] = (-10 * torch.log10(mse)).item()
                row["quality_note"] = (
                    "Historical native frames only; same-image main rerun and subjective review still required."
                )
                del native
            torch.save(
                {"frames": actual.cpu(), "latent_sha256": report["latent_sha256"]},
                out / f"taeh3-{label}.pt",
            )
        report["dtype_results"][label] = row
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({label: row}), flush=True)
        del adapter, actual, x, px
        torch.cuda.empty_cache()
    assert all(
        r["upstream_match"] and all(v["exact"] for v in r["partitions"].values())
        for r in report["dtype_results"].values()
    ), "TAEH3 correctness gate failed"


if __name__ == "__main__":
    main()
