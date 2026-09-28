# SPDX-License-Identifier: Apache-2.0
"""Independent-process tile batch probe; encoder evidence is not service acceptance."""

import argparse
import json
import os
from pathlib import Path

import torch
from frozen_source import verify_source
from validate_incremental_codecs import digest, load_codecs
from validate_progressive_encoder import (
    candidate_future_checks,
    comparison,
    lifecycle_checks,
    run_stream,
)
from validate_progressive_tiles import encoder_server_args, exact, indexed

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
    create_session,
)


def factory(batch_size):
    def make(codecs, *, capacity_frames):
        os.environ["MINIMAX_H3_PROGRESSIVE_ENCODER_TILE_BATCH_SIZE"] = str(batch_size)
        return create_session(codecs, capacity_frames=capacity_frames)

    return make


@torch.inference_mode()
def run(args):
    from sglang.multimodal_gen.runtime.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args

    verify_source(
        Path(__file__).resolve().parents[2], args.source_sha, args.source_manifest
    )
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank)
        initialize_model_parallel(sequence_parallel_degree=world, ulysses_degree=world)
    out = Path(args.output) / f"rank{rank}"
    out.mkdir(parents=True, exist_ok=False)
    if digest(args.rgb) != args.rgb_sha256:
        raise ValueError("Source RGB hash mismatch")
    raw = Path(args.rgb).read_bytes()
    if len(raw) % (480 * 832 * 3) or len(raw) < max(args.frames) * 480 * 832 * 3:
        raise ValueError("Insufficient complete frozen 832x480 RGB24 frames")
    pixels = (
        torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        .reshape(-1, 480, 832, 3)
        .permute(3, 0, 1, 2)
    )
    set_global_server_args(encoder_server_args(args.model))
    codecs, weights = load_codecs(Path(args.model))
    model = codecs.modules["video_vae"]
    model.stack_tiling = False
    dtype = getattr(torch, codecs.runtime["vae_config"]["encode_dtype"])
    model.encoder.to(dtype=dtype)
    model.quant_conv.to(dtype=dtype)
    os.environ["MINIMAX_H3_PROGRESSIVE_ENCODER_TILING"] = "auto"
    prop = torch.cuda.get_device_properties(local_rank)
    frozen = dict(
        vars(args),
        rank=rank,
        world=world,
        runtime=codecs.runtime,
        weights_sha256=digest(weights),
        config_sha256=digest(Path(args.model) / "zing_config.json"),
        gpu_uuid=str(prop.uuid),
        gpu_name=prop.name,
        memory_bytes=prop.total_memory,
        compute_capability=[prop.major, prop.minor],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        numerical_gate="bitwise; no tolerance",
        service_acceptance="not_run",
    )
    (out / "frozen.json").write_text(json.dumps(frozen, indent=2))
    report = dict(status="incomplete", correctness=[], timings=[], profiles=[])

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2))

    save()
    try:
        if args.phase == "lifecycle":
            # Compare the candidate to itself for isolation/ownership; cross-batch
            # strict numerical acceptance is independently recorded in correctness.
            report["future"] = candidate_future_checks(
                codecs, pixels[:, :120], factory(args.batch_size), out
            )
            if not all(row["passed"] for row in report["future"]):
                raise ValueError("Future pixel isolation failed")
            report["lifecycle"] = lifecycle_checks(
                codecs,
                pixels[:, :120],
                factory(args.batch_size),
                atol=0,
                rtol=0,
                output=out,
                reference_factory=factory(args.batch_size),
            )
        elif args.phase == "correctness":
            for frames in [*range(1, 35), 120]:
                source = pixels[:, :frames]
                expected, base = run_stream(
                    codecs, source, factory(1), [1], progressive=True
                )
                expected_sampled = indexed(codecs, expected, base["rows"])
                for pattern in ([1], [1, 2, 2], [17], [3, 8, 2, 11]):
                    actual, timing = run_stream(
                        codecs,
                        source,
                        factory(args.batch_size),
                        pattern,
                        progressive=True,
                    )
                    sampled = indexed(codecs, actual, timing["rows"])
                    row = dict(frames=frames, pattern=pattern, timeline=timing)
                    for label, a, b in (
                        ("moments", actual, expected),
                        ("posterior", sampled, expected_sampled),
                    ):
                        row[label] = {**comparison(a, b, atol=0, rtol=0), **exact(a, b)}
                    report["correctness"].append(row)
                    if (
                        not all(row[k]["passed"] for k in ("moments", "posterior"))
                        and "first_difference" not in report
                    ):
                        artifact = out / "first-difference.pt"
                        torch.save(
                            dict(
                                actual=actual,
                                expected=expected,
                                actual_posterior=sampled,
                                expected_posterior=expected_sampled,
                            ),
                            artifact,
                        )
                        report["first_difference"] = dict(
                            path=artifact.name,
                            sha256=digest(artifact),
                            frames=frames,
                            pattern=pattern,
                        )
                    save()
                    if not all(row[k]["finite"] for k in ("moments", "posterior")):
                        raise ValueError("Nonfinite encoder output")
            report["numerical_passed"] = all(
                c[k]["passed"]
                for c in report["correctness"]
                for k in ("moments", "posterior")
            )
        else:
            for frames in args.frames:
                source = pixels[:, :frames]
                # Includes both 8-frame history states and clip resets.
                run_stream(
                    codecs,
                    source,
                    factory(args.batch_size),
                    [1],
                    progressive=True,
                    capture=False,
                )
                if args.phase == "timing":
                    for repeat in range(3):
                        _, timing = run_stream(
                            codecs,
                            source,
                            factory(args.batch_size),
                            [1],
                            progressive=True,
                            capture=False,
                        )
                        report["timings"].append(
                            dict(frames=frames, repeat=repeat, **timing)
                        )
                        save()
                else:
                    trace = out / f"batch{args.batch_size}-frames{frames}.trace.json"
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=True,
                    ) as prof:
                        run_stream(
                            codecs,
                            source,
                            factory(args.batch_size),
                            [1],
                            progressive=True,
                            capture=False,
                        )
                    prof.export_chrome_trace(str(trace))
                    report["profiles"].append(
                        dict(path=trace.name, sha256=digest(trace))
                    )
        report["status"] = "encoder_probe_complete_not_service_acceptance"
        save()
    except Exception as error:
        report.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "rgb",
        "rgb-sha256",
        "source-sha",
        "source-manifest",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 3, 6), required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[120])
    parser.add_argument(
        "--phase",
        choices=("correctness", "lifecycle", "timing", "profile"),
        required=True,
    )
    args = parser.parse_args()
    if min(args.frames) < 17:
        parser.error("Timing/profile streams must cover a full 17-frame clip")
    run(args)
