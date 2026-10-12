# SPDX-License-Identifier: Apache-2.0
"""Isolated FP16 TAEH3 replay in the captured DiT chunk order."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.models.vaes.taeh3 import TAEH3Decoder, WEIGHTS_SHA256


@torch.no_grad()
def validate_progressive(model, latents, frames):
    """Same captured values and precision, strict old-versus-new output gate."""
    import math

    source = torch.cat(latents, dim=2)
    rows = []
    for length in sorted({*range(1, 35), 51, 60, frames}):
        if length > frames:
            continue
        clips, tail = divmod(length - 1, 17)
        count = clips * 5 + 1 + math.ceil(tail / 4)
        if source.shape[2] < count:
            raise ValueError("Captured latents are incomplete for decoder acceptance")
        values = source[:, :, :count]
        outputs = []
        for progressive in (False, True):
            stream = model.new_stream(progressive=progressive)
            parts, cursor, index = [], 0, 0
            try:
                while cursor < count:
                    end = min(count, cursor + (1, 2, 2)[index % 3])
                    parts.append(stream.push(values[:, :, cursor:end]).cpu())
                    cursor, index = end, index + 1
                parts.append(stream.finish(length).cpu())
                outputs.append(torch.cat(parts, dim=2))
            finally:
                stream.close()
        reference, candidate = outputs
        shape_ok = reference.shape == candidate.shape and reference.shape[2] == length
        finite = bool(
            torch.isfinite(reference).all() and torch.isfinite(candidate).all()
        )
        rows.append(
            dict(
                frames=length,
                latents=count,
                shape_equal=shape_ok,
                finite=finite,
                exact=shape_ok and finite and torch.equal(reference, candidate),
                max_abs=(
                    (reference.float() - candidate.float()).abs().max().item()
                    if shape_ok
                    else None
                ),
            )
        )
    return dict(
        passed=all(row["exact"] for row in rows),
        atol=0,
        rtol=0,
        partition=[1, 2, 2],
        reference="five-latent",
        candidate="progressive",
        cases=rows,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--progressive", action="store_true")
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
    model = TAEH3Decoder(args.weights, WEIGHTS_SHA256).cuda().half().eval()

    if args.progressive:
        acceptance = validate_progressive(model, latents, frames)
        (args.output / "progressive-acceptance.json").write_text(
            json.dumps(acceptance, indent=2)
        )
        if not acceptance["passed"]:
            raise RuntimeError(
                "Progressive TAEH3 does not match five-latent decoding exactly"
            )

    @torch.no_grad()
    def run(capture=False):
        stream = model.new_stream(progressive=args.progressive)
        outputs, intervals = [], []
        emitted = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        for value in latents:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            a.record()
            output = stream.push(value)
            b.record()
            intervals.append((a, b, output.shape[2]))
            emitted += output.shape[2]
            if capture:
                outputs.append(output)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        a.record()
        output = stream.finish(frames)
        b.record()
        emitted += output.shape[2]
        torch.cuda.synchronize()
        stats = dict(
            seconds=time.perf_counter() - start,
            frames=emitted,
            pushes=[dict(ms=x.elapsed_time(y), frames=n) for x, y, n in intervals],
            eos_ms=a.elapsed_time(b),
        )
        if capture:
            outputs.append(output)
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
        raise RuntimeError("Repeated TAEH3 replay is not exact")
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
                progressive=args.progressive,
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
