# SPDX-License-Identifier: Apache-2.0
"""Same captured latent sequence, actual single-vs-parallel Lynn decode; no DiT."""

import argparse
import datetime
import json
import os
import pathlib
import time

p = argparse.ArgumentParser()
p.add_argument("--capture", required=True)
p.add_argument("--pipeline", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.realtime.zing_nccl import configure_zing_nccl

configure_zing_nccl()
from sglang.multimodal_gen.configs.pipeline_configs.zing_h3_decoder import (
    normalize_decoder_spec,
)
from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.loader import (
    load_lynn_decoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker import (
    _TileGroup,
)

P = pathlib.Path
out = P(a.out)
out.mkdir(parents=True, exist_ok=True)
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=1800))
world = dist.get_world_size()
cpu = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=1800))
device = dist.new_group(backend="nccl", timeout=datetime.timedelta(seconds=1800))
group = _TileGroup(rank=rank, world_size=world, cpu_group=cpu, device_group=device)
with (out / f"rank{rank}-STARTED.json").open("x") as marker:
    marker.write(
        json.dumps(
            dict(
                rank=rank,
                world=world,
                scope="Fresh full Decoder comparison; production Decoder NCCL policy",
            )
        )
    )


def write(name, value):
    (out / (f"rank{rank}-" + name + ".json")).write_text(json.dumps(value, indent=2))


complete = json.loads((P(a.capture) / "COMPLETE.json").read_text())
assert complete["input_frames"] == 15921 and complete["chunks"] == 1873
rows = [
    torch.load(
        P(a.capture) / f"chunk-{i:05d}.pt", map_location="cpu", weights_only=False
    )
    for i in range(complete["chunks"])
]
frames = rows[-1]["chunk"].provenance.pixels.end
assert frames == 15921
latents = [r["expected_visual"].cuda() for r in rows]
del rows
spec = normalize_decoder_spec(
    json.loads(P(a.pipeline).read_text())["decoder_config"],
    video_decoder_backend=None,
    vae_decode_precision="fp32",
)
model = load_lynn_decoder(spec, device=torch.device("cuda", rank)).eval()
comparison = []
parallel = model.new_stream(frame_limit=frames, group=group)
single = model.new_stream(frame_limit=frames) if rank == 0 else None
try:
    for i, z in enumerate(latents):
        reference = single.push(z) if rank == 0 else None
        result = parallel.push(z)
        if rank == 0:
            assert (
                reference.shape == result.shape
                and torch.isfinite(result).all().item()
                and torch.isfinite(reference).all().item()
            )
            difference = (reference - result).float()
            comparison.append(
                dict(
                    chunk=i,
                    frames=result.shape[2],
                    elements=result.numel(),
                    exact=torch.equal(reference, result),
                    max_abs=difference.abs().max().item() if result.numel() else 0.0,
                    squared_error_sum=difference.square()
                    .sum(dtype=torch.float64)
                    .item(),
                )
            )
        if rank == 0 and i in [0, 1, 2, 100, 500, 1000, 1500, 1872] and result.shape[2]:
            from PIL import Image

            for label, tensor in [("single", reference), ("parallel", result)]:
                array = (
                    tensor[0, :, 0]
                    .permute(1, 2, 0)
                    .clamp(0, 1)
                    .mul(255)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                Image.fromarray(array).save(out / f"chunk-{i:05d}-{label}.png")
        del reference, result
        if rank == 0 and (i % 100 == 0 or i == 1872):
            write(
                "parity-progress",
                dict(chunks=i + 1, comparison=comparison[-1], time=time.time()),
            )
            print("PARITY_PROGRESS", i + 1, flush=True)
    parallel.finish(frames)
    if rank == 0:
        single.finish(frames)
        write(
            "comparison",
            dict(
                chunks=len(comparison),
                frames=sum(v["frames"] for v in comparison),
                all_exact=all(v["exact"] for v in comparison),
                max_abs=max(v["max_abs"] for v in comparison),
                rows=comparison,
                scope="Actual captured full 15921-frame latents, same model weights; all output elements checked without full RGB retention. This untimed comparison is not speed evidence.",
            ),
        )
finally:
    parallel.close()
    if single is not None:
        single.close()
dist.barrier(group=cpu)
runs = []
for mode in ["single", "parallel"]:
    for trial in range(3):
        dist.barrier(group=cpu)
        if mode == "single" and rank != 0:
            dist.barrier(group=cpu)
            continue
        stream = model.new_stream(
            frame_limit=frames, group=group if mode == "parallel" else None
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        resident = dict(
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
        )
        events = []
        emitted = 0
        started = time.perf_counter()
        try:
            for i, z in enumerate(latents):
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                result = stream.push(z)
                end.record()
                emitted += result.shape[2]
                events.append((begin, end, result.shape[2]))
                del result
            stream.finish(frames)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert emitted == frames
        finally:
            stream.close()
        record = dict(
            mode=mode,
            trial=trial,
            rank=rank,
            world=world,
            frames=emitted,
            seconds=elapsed,
            fps=emitted / elapsed,
            blocks=[
                dict(index=i, cuda_elapsed_ms=begin.elapsed_time(end), frames=n)
                for i, (begin, end, n) in enumerate(events)
            ],
            resident_before=resident,
            after_cleanup=dict(
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
            ),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            scope="Isolated Lynn CUDA stream elapsed, three full original-case sessions; no inter-role transport. Loaded frozen inputs/model/workspaces share process memory. No accuracy scans during timing.",
        )
        runs.append(record)
        write("measured", runs)
        print("MEASURED", mode, trial, rank, elapsed, flush=True)
        del events
        dist.barrier(group=cpu)
write(
    "COMPLETE",
    dict(rank=rank, world=world, collected=True, independent_acceptance_required=True),
)
dist.destroy_process_group()
