# SPDX-License-Identifier: Apache-2.0
"""Isolated Lynn replay in the captured DiT chunk order."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3_decoder import (
    normalize_decoder_spec,
)
from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.loader import (
    load_lynn_decoder,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--decoder-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    count = json.loads((args.capture / "COMPLETE.json").read_text())["chunks"]
    rows = [
        torch.load(
            args.capture / f"chunk-{i:05d}.pt", map_location="cpu", weights_only=False
        )
        for i in range(count)
    ]
    frames = rows[-1]["chunk"].provenance.pixels.end
    latents = [row["expected_visual"].cuda() for row in rows]
    spec = normalize_decoder_spec(
        json.loads(args.decoder_config.read_text()),
        video_decoder_backend=None,
        vae_decode_precision="fp32",
    )
    if spec.type != "lynn_streaming":
        raise ValueError("Decoder replay requires Lynn")
    model = load_lynn_decoder(spec, device="cuda").eval()

    @torch.no_grad()
    def run(capture=False):
        stream = model.new_stream(frame_limit=frames)
        outputs, intervals = [], []
        emitted = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        for value in latents:
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            output = stream.push(value)
            b.record()
            intervals.append((a, b, output.shape[2]))
            emitted += output.shape[2]
            if capture:
                outputs.append(output)
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        output = stream.finish(frames)
        b.record()
        # Lynn emits frames during push; finish validates the final frame limit.
        torch.cuda.synchronize()
        stats = dict(
            seconds=time.perf_counter() - start,
            frames=emitted,
            pushes=[dict(ms=x.elapsed_time(y), frames=n) for x, y, n in intervals],
            eos_ms=a.elapsed_time(b),
        )
        stream.close()
        assert emitted == frames
        return stats, torch.cat(outputs, 2) if capture else None

    _, reference = run(capture=True)
    runs = []
    for _ in range(5):
        stats, _ = run()
        runs.append(stats)
    _, check = run(capture=True)
    exact = bool(torch.equal(reference, check) and torch.isfinite(check).all())
    if not exact:
        raise RuntimeError("Repeated Lynn replay is not exact")
    del reference, check
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        profile, _ = run()
    prof.export_chrome_trace(str(args.output / "decoder.trace.json"))
    active = sorted(row["ms"] for run in runs for row in run["pushes"] if row["frames"])
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(
                runs=runs,
                exact_repeat=exact,
                median_seconds=statistics.median(r["seconds"] for r in runs),
                emitting_push_p50_ms=statistics.median(active),
                emitting_push_p95_ms=active[int(0.95 * (len(active) - 1))],
                profile_seconds_separate=profile["seconds"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
