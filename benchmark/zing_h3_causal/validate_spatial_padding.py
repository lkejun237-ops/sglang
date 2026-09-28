# SPDX-License-Identifier: Apache-2.0
"""Independent-process #196 encoder probe; service/media acceptance is separate.

Run off/reflect2d/reflect2d/off in four processes on the same device(s). Compare
saved moments and posteriors by bytes. This tool never relaxes the exact gate.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from frozen_source import verify_source
from validate_incremental_codecs import digest, load_codecs
from validate_progressive_encoder import (
    candidate_future_checks,
    lifecycle_checks,
    run_stream,
)
from validate_progressive_tiles import encoder_server_args, indexed

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
    create_session,
)


@torch.inference_mode()
def run(args):
    from sglang.multimodal_gen.runtime.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args

    verify_source(Path(__file__).resolve().parents[2], args.source_sha)
    os.environ["MINIMAX_H3_PROGRESSIVE_SPATIAL_PADDING"] = args.mode
    os.environ.pop("MINIMAX_H3_PROGRESSIVE_ENCODER_TILING", None)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if world > 1:
        init_distributed_environment(world_size=world, rank=rank, local_rank=local)
        initialize_model_parallel(sequence_parallel_degree=world, ulysses_degree=world)
    output = Path(args.output) / f"rank{rank}"
    output.mkdir(parents=True, exist_ok=False)
    if digest(args.rgb) != args.rgb_sha256:
        raise ValueError("Frozen RGB SHA256 mismatch")
    frame_bytes = 832 * 480 * 3
    raw = Path(args.rgb).read_bytes()
    if len(raw) % frame_bytes or len(raw) < max(args.frames) * frame_bytes:
        raise ValueError("Source RGB is incomplete")
    pixels = (
        torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        .reshape(-1, 480, 832, 3)
        .permute(3, 0, 1, 2)
    )
    set_global_server_args(encoder_server_args(args.model))
    codecs, weights = load_codecs(Path(args.model))
    model = codecs.modules["video_vae"]
    model.stack_tiling = False
    assert codecs.runtime["vae_config"]["encode_mode"] == "deterministic"
    dtype = getattr(torch, codecs.runtime["vae_config"]["encode_dtype"])
    model.encoder.to(dtype=dtype)
    model.quant_conv.to(dtype=dtype)
    prop = torch.cuda.get_device_properties(local)
    report = dict(
        status="incomplete",
        args=vars(args),
        rank=rank,
        world=world,
        gpu_uuid=str(prop.uuid),
        gpu_name=prop.name,
        memory_bytes=prop.total_memory,
        cc=[prop.major, prop.minor],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        weights_sha256=digest(weights),
        runtime=codecs.runtime,
        cases=[],
        timings=[],
    )

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2))

    def execute(source, pattern, capture=False):
        result, timing = run_stream(
            codecs, source, create_session, pattern, progressive=True, capture=capture
        )
        for row in timing["rows"]:
            receipt = row["tile_execution"]
            if receipt["encoder_ranks"] != world or receipt["tile_count"] != 12:
                raise ValueError("Expected actual 12-tile sharding")
            if len(receipt["local_tile_ids"]) != 12 // world:
                raise ValueError("Unexpected local tile assignment")
            assert receipt["spatial_padding"] == args.mode
        return result, timing

    save()
    try:
        # All three causal states, repeated to warm the exact kernels.
        for _ in range(2):
            execute(pixels[:, :34], [1])
        if args.correctness:
            for frames in [*range(1, 35), 120]:
                for pattern in ([1], [1, 3, 17, 2, 5]):
                    moments, timing = execute(pixels[:, :frames], pattern, True)
                    posterior = indexed(codecs, moments, timing["rows"])
                    name = f"frames{frames}-pattern{'-'.join(map(str, pattern))}.pt"
                    torch.save(
                        dict(moments=moments, posterior=posterior), output / name
                    )
                    report["cases"].append(
                        dict(frames=frames, pattern=pattern, raw=name, timeline=timing)
                    )
                    save()
            report["future"] = candidate_future_checks(
                codecs, pixels[:, :120], create_session, output
            )
            if not all(r["passed"] for r in report["future"]):
                raise ValueError("Future dependency check failed")
            report["lifecycle"] = lifecycle_checks(
                codecs,
                pixels[:, :120],
                create_session,
                atol=0,
                rtol=0,
                output=output,
                report_numerical_only=False,
                reference_factory=create_session,
            )
            save()
        for frames in args.frames:
            execute(pixels[:, :frames], [1])
            for repeat in range(3):
                _, timing = execute(pixels[:, :frames], [1])
                report["timings"].append(dict(frames=frames, repeat=repeat, **timing))
                save()
        if args.profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof:
                execute(pixels[:, :17], [1])
            prof.export_chrome_trace(str(output / "encoder.trace.json.gz"))
        report["status"] = "encoder_probe_complete_not_service_acceptance"
        save()
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        save()
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--rgb", required=True)
    p.add_argument("--rgb-sha256", required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--mode", choices=["off", "reflect2d"], required=True)
    p.add_argument("--frames", type=int, nargs="+", default=[240])
    p.add_argument("--output", required=True)
    p.add_argument("--correctness", action="store_true")
    p.add_argument("--profile", action="store_true")
    run(p.parse_args())
